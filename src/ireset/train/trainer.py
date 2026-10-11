"""Single-GPU trainer for both generators (plan sections 9.1/3.3).

Pilot conventions: RTX 3090, micro-batch 4 with gradient accumulation 4
(effective 16), FP32 (AMP deliberately off for the pilot so loss scales and
FK-loss gradients stay simple to audit), AdamW, gradient clipping. Records
per-step wall time, peak CUDA memory and the validation loss history;
checkpoints keep the tokenizer statistics so a checkpoint is
self-contained.

Formal best selection (W1 plan §5.2/§5.3, review fix 9): when a
``formal_eval`` callback is provided, ``best.pt`` is selected ONLY by the
demonstration-macro generation metric the callback measures over the FULL
expected val set, evaluated every ``formal_eval_every`` optimizer steps —
never by the window-validation loss (a window loss may improve while the
macro metric worsens). Missing/incomplete formal evidence means selection
is insufficient: no ``best.pt`` is written for that evaluation, and if no
evaluation ever completes, ``best.pt`` simply does not exist.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ireset.diffusion.generator import DiffusionG1
from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1
from ireset.train.losses import LossAssembler, LossWeights


def _worker_thread_limits(worker_id: int) -> None:
    """DataLoader worker boot (cpu_parallel_optimization_plan section 4):
    each worker is I/O + tensor-preprocess only, so its native pools are
    pinned to ONE thread -- outer batching provides the parallelism and
    CPU workers never touch CUDA. Environment first (any library the
    worker imports later), then torch for the already-imported runtime."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, "1")
    torch.set_num_threads(1)


@dataclass
class TrainConfig:
    model_type: str = "diffusion"          # "diffusion" | "deterministic"
    steps: int = 4000                      # optimizer steps
    micro_batch: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 20260913
    num_workers: int = 4
    val_every: int = 250
    log_every: int = 25
    device: str = "cuda"
    #: dedicated seed for the FIXED validation t/noise stream (W1 §5.3);
    #: None falls back to ``seed``. Validation never touches the global RNG.
    val_seed: int | None = None
    # backbone (plan 7.1 pilot: d_model 240, 4 layers, 6 heads, window 31)
    d_model: int = 240
    n_layers: int = 4
    n_heads: int = 6
    ffn_mult: int = 4
    time_window: int = 31
    # diffusion
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    loss: LossWeights = field(default_factory=LossWeights)

    def backbone_kwargs(self) -> dict:
        return dict(
            d_model=self.d_model, n_layers=self.n_layers, n_heads=self.n_heads,
            ffn_mult=self.ffn_mult, time_window=self.time_window,
        )


class Trainer:
    def __init__(self, cfg: TrainConfig, tokenizer, train_ds, val_ds, out_dir: Path,
                 identity: dict | None = None,
                 formal_eval=None, formal_eval_every: int | None = None) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train_ds, self.val_ds = train_ds, val_ds
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        #: formal-generation best-selection hook (W1 §5.3, review fix 9):
        #: callable(model, conditioner) -> {"n_expected", "n_ok",
        #: "macro_mpjpe_m", "per_demo": [...]}; complete evidence required
        self.formal_eval = formal_eval
        self.formal_eval_every = formal_eval_every
        if (formal_eval is None) != (formal_eval_every is None):
            raise ValueError("formal_eval and formal_eval_every come together")

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if cfg.model_type == "diffusion":
            self.model = DiffusionG1(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_schedule=cfg.beta_schedule,
                beta_start=cfg.beta_start,
                beta_end=cfg.beta_end,
                **cfg.backbone_kwargs(),
            ).to(cfg.device)
        else:
            self.model = DeterministicG1(**cfg.backbone_kwargs()).to(cfg.device)
        self.conditioner = ConditionEncoder(cfg.d_model).to(cfg.device)
        self.loss_fn = LossAssembler(tokenizer, cfg.loss, cfg.device)
        self.opt = torch.optim.AdamW(
            [*self.model.parameters(), *self.conditioner.parameters()],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        g = torch.Generator().manual_seed(cfg.seed)
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.micro_batch, shuffle=True,
            num_workers=cfg.num_workers, generator=g, drop_last=True,
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=_worker_thread_limits,
        )
        # the val loader keeps its OWN generator: creating a DataLoader
        # iterator draws a base seed from the loader's generator, and with
        # ``generator=None`` that draw comes from the GLOBAL stream —
        # validation must never perturb the training RNG (W1 §5.3)
        g_val = torch.Generator().manual_seed(cfg.seed)
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.micro_batch, shuffle=False,
            num_workers=0, drop_last=False, generator=g_val,
        )

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.cfg.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _forward(self, batch, noise_generator: torch.Generator | None = None):
        B = batch["x0"].shape[0]
        memory = self.conditioner(batch)
        if self.cfg.model_type == "diffusion":
            if noise_generator is not None:
                # validation: FIXED t/noise drawn from a dedicated CPU
                # generator — reproducible, and never touching the global
                # RNG stream the training steps sample from
                t = torch.randint(0, self.cfg.num_train_timesteps, (B,),
                                  generator=noise_generator).to(self.cfg.device)
                noise = torch.randn(batch["x0"].shape,
                                    generator=noise_generator).to(self.cfg.device)
            else:
                t = torch.randint(0, self.cfg.num_train_timesteps, (B,), device=self.cfg.device)
                noise = torch.randn_like(batch["x0"])
            x_t = self.model.add_noise(batch["x0"], noise, t)
            x0_pred = self.model(x_t, t, memory)
        else:
            x0_pred = self.model(memory, batch["x0"].shape[1])
        return self.loss_fn(x0_pred, batch)

    def _run_formal_eval(self) -> dict:
        """One formal-generation evaluation through the registered callback.

        Any exception is REGISTERED as incomplete evidence — the run keeps
        going, but that evaluation can never select ``best.pt``."""
        assert self.formal_eval is not None
        was_training = self.model.training
        self.model.eval(); self.conditioner.eval()
        try:
            with torch.no_grad():
                out = dict(self.formal_eval(self.model, self.conditioner))
        except Exception as exc:  # noqa: BLE001 -- evidence, not a crash
            out = {"n_expected": None, "n_ok": 0, "macro_mpjpe_m": None,
                   "complete": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if was_training:
                self.model.train(); self.conditioner.train()
        complete = (
            out.get("n_expected") is not None
            and out.get("n_ok") == out.get("n_expected")
            and isinstance(out.get("macro_mpjpe_m"), (int, float))
            and math.isfinite(float(out["macro_mpjpe_m"])))
        out["complete"] = bool(complete)
        return out

    def run(self) -> dict:
        cfg = self.cfg
        history, val_history = [], []
        formal_history: list[dict] = []
        best_val = float("inf")
        best_formal = float("inf")
        best_formal_step = None
        step, ms_ema = 0, None
        data_iter = iter(self.train_loader)
        self.model.train(); self.conditioner.train()
        t_start = time.perf_counter()

        while step < cfg.steps:
            opt_step_t0 = time.perf_counter()
            self.opt.zero_grad(set_to_none=True)
            parts_acc: dict[str, float] = {}
            for _ in range(cfg.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                batch = self._to_device(batch)
                loss, parts = self._forward(batch)
                (loss / cfg.grad_accum).backward()
                for k, v in parts.items():
                    parts_acc[k] = parts_acc.get(k, 0.0) + v / cfg.grad_accum
            gnorm = torch.nn.utils.clip_grad_norm_(
                [*self.model.parameters(), *self.conditioner.parameters()], cfg.grad_clip)
            self.opt.step()
            step += 1

            wall = (time.perf_counter() - opt_step_t0) * 1e3
            ms_ema = wall if ms_ema is None else 0.98 * ms_ema + 0.02 * wall
            history.append({"step": step, **{k: round(v, 6) for k, v in parts_acc.items()},
                            "gnorm": float(gnorm), "ms": round(wall, 1)})
            if step % cfg.log_every == 0:
                last = history[-1]
                print(f"[{cfg.model_type}] step {step}/{cfg.steps} "
                      f"total={last['total']:.4f} fk={last.get('fk_m2', float('nan')):.5f} "
                      f"geo={last.get('geodesic_rad2', float('nan')):.5f} "
                      f"ms/step={ms_ema:.0f}", flush=True)
            if step % cfg.val_every == 0 or step == cfg.steps:
                val = self.validate()
                val_history.append({"step": step, **val})
                print(f"[{cfg.model_type}] VAL step {step}: total={val['total']:.4f}", flush=True)
                self.save("last.pt", step, val, identity=self.identity)
                if self.formal_eval is not None:
                    # W1 §5.3 best rule (review fix 9): best.pt comes ONLY
                    # from complete formal macro evidence — the window loss
                    # never selects, and incomplete evidence selects nothing
                    if step % self.formal_eval_every == 0 or step == cfg.steps:
                        formal = self._run_formal_eval()
                        formal_history.append({"step": step, **formal})
                        macro = formal["macro_mpjpe_m"]
                        print(f"[{cfg.model_type}] FORMAL step {step}: "
                              f"complete={formal['complete']} macro={macro}",
                              flush=True)
                        if (formal["complete"]
                                and float(macro) < best_formal):
                            best_formal = float(macro)
                            best_formal_step = step
                            self.save("best.pt", step, val,
                                      identity=self.identity, formal=formal)
                elif val["total"] < best_val:
                    best_val = val["total"]
                    self.save("best.pt", step, val, identity=self.identity)

        summary = {
            "config": asdict(cfg),
            "identity": self.identity,
            "steps": step,
            "wall_s": round(time.perf_counter() - t_start, 1),
            "ms_per_step_ema": round(ms_ema, 1),
            "peak_cuda_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available() else None,
            "best_val": best_val,
            "best_selection_rule": ("formal_macro_mpjpe" if self.formal_eval
                                    is not None else "window_val_loss"),
            "best_formal_macro_mpjpe_m": (best_formal
                                          if self.formal_eval is not None
                                          else None),
            "best_formal_step": best_formal_step,
            "n_train_windows": len(self.train_ds),
            "n_val_windows": len(self.val_ds),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(
            {**summary, "val_history": val_history,
             "formal_history": formal_history}, indent=2, default=float))
        (self.out_dir / "loss_history.json").write_text(json.dumps(history))
        return summary

    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> dict:
        """Loss validation over the val dataset.

        W1 contract (plan §5.3): EVERY admitted val window participates —
        no 40-batch truncation (``max_batches`` stays available only as an
        explicit debug knob); batch parts are weighted by SAMPLE count so a
        smaller final batch cannot skew the mean; the diffusion t/noise
        stream is FIXED through a dedicated generator (``val_seed``) and
        the global RNG is never reseeded — validation must not perturb the
        training random sequence or the DataLoader order.
        """
        self.model.eval(); self.conditioner.eval()
        g = torch.Generator()
        g.manual_seed(self.cfg.val_seed if self.cfg.val_seed is not None
                      else self.cfg.seed)
        acc: dict[str, float] = {}
        n_samples, n_batches = 0, 0
        for i, batch in enumerate(self.val_loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = self._to_device(batch)
            _, parts = self._forward(batch, noise_generator=g)
            w = batch["x0"].shape[0]
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v * w
            n_samples += w
            n_batches += 1
        self.model.train(); self.conditioner.train()
        out = {k: v / max(n_samples, 1) for k, v in acc.items()}
        out["n_windows_evaluated"] = n_samples
        out["n_val_windows_total"] = len(self.val_ds)
        out["n_batches"] = n_batches
        return out

    def save(self, name: str, step: int, val: dict,
             identity: dict | None = None, formal: dict | None = None) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "conditioner": self.conditioner.state_dict(),
                "step": step,
                "val": val,
                "formal": formal,
                "config": asdict(self.cfg),
                "root_stats": {"mean": self.tokenizer.root_stats.mean,
                               "std": self.tokenizer.root_stats.std},
                # W1 model identity (plan §5.3): what this checkpoint is
                # bound to — consumers refuse a checkpoint whose identity
                # does not match the manifest/cache/stats they hold
                "identity": identity,
            },
            self.out_dir / name,
        )
