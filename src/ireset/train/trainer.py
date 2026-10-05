"""Single-GPU trainer for both generators (plan sections 9.1/3.3).

Pilot conventions: RTX 3090, micro-batch 4 with gradient accumulation 4
(effective 16), FP32 (AMP deliberately off for the pilot so loss scales and
FK-loss gradients stay simple to audit), AdamW, gradient clipping. Records
per-step wall time, peak CUDA memory and the validation loss history;
checkpoints keep the tokenizer statistics so a checkpoint is
self-contained.
"""

from __future__ import annotations

import json
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
    # backbone (plan 7.1 pilot: d_model 240, 4 layers, 6 heads, window 31)
    d_model: int = 240
    n_layers: int = 4
    n_heads: int = 6
    ffn_mult: int = 4
    time_window: int = 31
    # diffusion
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    loss: LossWeights = field(default_factory=LossWeights)

    def backbone_kwargs(self) -> dict:
        return dict(
            d_model=self.d_model, n_layers=self.n_layers, n_heads=self.n_heads,
            ffn_mult=self.ffn_mult, time_window=self.time_window,
        )


class Trainer:
    def __init__(self, cfg: TrainConfig, tokenizer, train_ds, val_ds, out_dir: Path) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train_ds, self.val_ds = train_ds, val_ds
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if cfg.model_type == "diffusion":
            self.model = DiffusionG1(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_schedule=cfg.beta_schedule,
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
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.micro_batch, shuffle=False,
            num_workers=0, drop_last=False,
        )

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.cfg.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _forward(self, batch):
        B = batch["x0"].shape[0]
        memory = self.conditioner(batch)
        if self.cfg.model_type == "diffusion":
            t = torch.randint(0, self.cfg.num_train_timesteps, (B,), device=self.cfg.device)
            noise = torch.randn_like(batch["x0"])
            x_t = self.model.add_noise(batch["x0"], noise, t)
            x0_pred = self.model(x_t, t, memory)
        else:
            x0_pred = self.model(memory, batch["x0"].shape[1])
        return self.loss_fn(x0_pred, batch)

    def run(self) -> dict:
        cfg = self.cfg
        history, val_history = [], []
        best_val = float("inf")
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
                self.save("last.pt", step, val)
                if val["total"] < best_val:
                    best_val = val["total"]
                    self.save("best.pt", step, val)

        summary = {
            "config": asdict(cfg),
            "steps": step,
            "wall_s": round(time.perf_counter() - t_start, 1),
            "ms_per_step_ema": round(ms_ema, 1),
            "peak_cuda_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available() else None,
            "best_val": best_val,
            "n_train_windows": len(self.train_ds),
            "n_val_windows": len(self.val_ds),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(
            {**summary, "val_history": val_history}, indent=2, default=float))
        (self.out_dir / "loss_history.json").write_text(json.dumps(history))
        return summary

    @torch.no_grad()
    def validate(self, max_batches: int = 40) -> dict:
        self.model.eval(); self.conditioner.eval()
        torch.manual_seed(self.cfg.seed)                    # reproducible val noise
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg.seed)
        acc, n = {}, 0
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            batch = self._to_device(batch)
            _, parts = self._forward(batch)
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
        self.model.train(); self.conditioner.train()
        return {k: v / max(n, 1) for k, v in acc.items()}

    def save(self, name: str, step: int, val: dict) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "conditioner": self.conditioner.state_dict(),
                "step": step,
                "val": val,
                "config": asdict(self.cfg),
                "root_stats": {"mean": self.tokenizer.root_stats.mean,
                               "std": self.tokenizer.root_stats.std},
            },
            self.out_dir / name,
        )
