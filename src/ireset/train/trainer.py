"""Single-GPU training with finite-first checks and preserved W1-D evidence.

Initialization, noise draws, loader flow, losses and optimizer arithmetic keep
 their accepted first-success behavior. Execution checks surround those calls;
 they do not repair numerical failures. ``budget_deadline`` is an absolute
 perf_counter work deadline with the driver's shutdown reserve subtracted.
 Formal callbacks may accept that keyword and must expose ``expected_demos``
 (and optionally ``demo_lengths``); only complete registered rows select best.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ireset.diffusion.generator import DiffusionG1
from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1
from ireset.train.losses import LossAssembler, LossWeights


class TrainingNumericalError(RuntimeError):
    """A checked numerical failure; ``evidence`` identifies the failing value."""

    def __init__(self, reason: str, **evidence) -> None:
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


class TrainingBudgetExceeded(RuntimeError):
    """No further work unit may start, including validation and normal saves."""

    def __init__(self, unit: str) -> None:
        super().__init__(f"budget_deadline:{unit}")
        self.unit = unit


def _finite_number(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if torch.is_tensor(value):
        return (value.dtype != torch.bool and not value.is_complex()
                and value.numel() == 1 and bool(torch.isfinite(value).all()))
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _positive_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0


def _rotation6d_valid(r6: torch.Tensor) -> bool:
    # Check the orthogonal component actually normalized by the frozen decoder.
    # Checking only the two raw norms accepts parallel nonzero columns.
    with torch.no_grad():
        a, b = r6[..., :3].detach(), r6[..., 3:6].detach()
        orth = b - (a * b).sum(-1, keepdim=True) * a / (a * a).sum(-1, keepdim=True).clamp_min(1e-12)
        na, nb = a.norm(dim=-1), orth.norm(dim=-1)
        nc = torch.linalg.cross(a, orth, dim=-1).norm(dim=-1)
        return bool((torch.isfinite(na) & torch.isfinite(nb) & torch.isfinite(nc)
                     & (na > 1e-6) & (nb > 1e-6) & (nc > 1e-12)).all())


def _json_evidence(value, invalid: list[str], path: str = ""):
    """Make bad evidence writable *after* its validity has been adjudicated.

    Nonfinite fields become null with their exact paths retained, never a
    numeric passing value. This is serialization, not an acceptance check.
    """
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_evidence(v, invalid, f"{path}.{k}" if path else str(k))
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_evidence(v, invalid, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        invalid.append(path)
        return None
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _worker_thread_limits(worker_id: int) -> None:
    """DataLoader workers keep one native thread; CUDA stays in the parent."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, "1")
    torch.set_num_threads(1)


@dataclass
class TrainConfig:
    model_type: str = "diffusion"
    steps: int = 4000
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
    val_seed: int | None = None
    d_model: int = 240
    n_layers: int = 4
    n_heads: int = 6
    ffn_mult: int = 4
    time_window: int = 31
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    loss: LossWeights = field(default_factory=LossWeights)

    def backbone_kwargs(self) -> dict:
        return dict(d_model=self.d_model, n_layers=self.n_layers,
                    n_heads=self.n_heads, ffn_mult=self.ffn_mult,
                    time_window=self.time_window)


class Trainer:
    def __init__(self, cfg: TrainConfig, tokenizer, train_ds, val_ds, out_dir: Path,
                 identity: dict | None = None,
                 formal_eval=None, formal_eval_every: int | None = None,
                 budget_deadline: float | None = None) -> None:
        self.cfg, self.tokenizer = cfg, tokenizer
        self.train_ds, self.val_ds = train_ds, val_ds
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.formal_eval, self.formal_eval_every = formal_eval, formal_eval_every
        if (formal_eval is None) != (formal_eval_every is None):
            raise ValueError("formal_eval and formal_eval_every come together")
        if formal_eval_every is not None and not _positive_int(formal_eval_every):
            raise ValueError("formal_eval_every must be a positive integer")
        if budget_deadline is not None and not _finite_number(budget_deadline):
            raise ValueError("budget_deadline must be finite")
        self.budget_deadline = budget_deadline
        # Snapshot independently registered identities before a callback can
        # mutate its attributes or report its own smaller denominator.
        expected = getattr(formal_eval, "expected_demos", None)
        self.formal_expected_demos = tuple(expected) if isinstance(expected, (list, tuple)) else None
        lengths = getattr(formal_eval, "demo_lengths", None)
        self.formal_demo_lengths = dict(lengths) if isinstance(lengths, dict) else None
        self._checkpoint_records: dict[str, dict] = {}
        self._preserving_evidence = False
        self._failure: dict | None = None
        self._last_validation_evidence: dict | None = None
        self._last_formal_evidence: dict | None = None

        # Preserve the exact accepted initialization/RNG order.
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if cfg.model_type == "diffusion":
            self.model = DiffusionG1(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_schedule=cfg.beta_schedule, beta_start=cfg.beta_start,
                beta_end=cfg.beta_end, **cfg.backbone_kwargs(),
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
        g_val = torch.Generator().manual_seed(cfg.seed)
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.micro_batch, shuffle=False,
            num_workers=0, drop_last=False, generator=g_val,
        )

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.cfg.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _synchronize(self) -> None:
        if torch.device(self.cfg.device).type == "cuda":
            torch.cuda.synchronize(self.cfg.device)

    def _budget_allows_new_unit(self) -> bool:
        return self.budget_deadline is None or time.perf_counter() < self.budget_deadline

    def _check_deadline(self, unit: str) -> None:
        if not self._budget_allows_new_unit():
            raise TrainingBudgetExceeded(unit)

    def _params_finite(self) -> bool:
        return all(bool(torch.isfinite(p).all())
                   for p in (*self.model.parameters(), *self.conditioner.parameters()))

    def _check_params(self) -> None:
        for side, module in (("model", self.model), ("conditioner", self.conditioner)):
            for name, p in module.named_parameters():
                if not bool(torch.isfinite(p).all()):
                    raise TrainingNumericalError("nonfinite_params", field=f"{side}.{name}")

    def _check_gradients(self) -> None:
        for side, module in (("model", self.model), ("conditioner", self.conditioner)):
            for name, p in module.named_parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise TrainingNumericalError("nonfinite_gradient", field=f"{side}.{name}")

    def _check_batch(self, batch: dict) -> None:
        if not isinstance(batch, dict) or not torch.is_tensor(batch.get("x0")):
            raise TrainingNumericalError("invalid_input", field="batch.x0")
        x0 = batch["x0"]
        if x0.ndim != 3 or x0.shape[-1] != 38 or min(x0.shape[:2]) < 1:
            raise TrainingNumericalError("invalid_input", field="batch.x0", shape=list(x0.shape))
        for key, value in batch.items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise TrainingNumericalError("nonfinite_input", field=f"batch.{key}")
        if not _rotation6d_valid(x0[..., 3:9]):
            raise TrainingNumericalError("degenerate_input_rotation6d", field="batch.x0[3:9]")
        quat = batch.get("box_quat")
        if torch.is_tensor(quat):
            n = quat.norm(dim=-1)
            if not bool((torch.isfinite(n) & (n > 1e-6)).all()):
                raise TrainingNumericalError("degenerate_input_quaternion", field="batch.box_quat")

    def _check_losses(self, loss, parts) -> None:
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise TrainingNumericalError("invalid_loss", field="loss")
        if not bool(torch.isfinite(loss).all()):
            raise TrainingNumericalError("nonfinite_loss", field="loss")
        required = {"pos", "rot6d", "joint", "geodesic_rad2", "vel", "total"}
        if self.cfg.loss.w_fk > 0:
            required.add("fk_m2")
            if self.cfg.loss.w_contact > 0:
                required.add("contact_m2")
        if not isinstance(parts, dict) or not required <= parts.keys():
            missing = sorted(required - parts.keys()) if isinstance(parts, dict) else sorted(required)
            raise TrainingNumericalError("missing_loss_fields", missing=missing)
        for key, value in parts.items():
            if not _finite_number(value):
                raise TrainingNumericalError("nonfinite_loss_part", field=f"parts.{key}", value=repr(value))

    def _forward(self, batch, noise_generator: torch.Generator | None = None):
        self._check_batch(batch)
        B = batch["x0"].shape[0]
        memory = self.conditioner(batch)
        if not torch.is_tensor(memory) or not bool(torch.isfinite(memory).all()):
            raise TrainingNumericalError("nonfinite_condition", field="condition_memory")
        if self.cfg.model_type == "diffusion":
            if noise_generator is not None:
                t = torch.randint(0, self.cfg.num_train_timesteps, (B,),
                                  generator=noise_generator).to(self.cfg.device)
                noise = torch.randn(batch["x0"].shape,
                                    generator=noise_generator).to(self.cfg.device)
            else:
                t = torch.randint(0, self.cfg.num_train_timesteps, (B,), device=self.cfg.device)
                noise = torch.randn_like(batch["x0"])
            x_t = self.model.add_noise(batch["x0"], noise, t)
            if not bool(torch.isfinite(noise).all()) or not bool(torch.isfinite(x_t).all()):
                raise TrainingNumericalError("nonfinite_diffusion_input", field="noise_or_x_t")
            x0_pred = self.model(x_t, t, memory)
        else:
            x0_pred = self.model(memory, batch["x0"].shape[1])
        if not torch.is_tensor(x0_pred) or x0_pred.shape != batch["x0"].shape:
            raise TrainingNumericalError("invalid_prediction", field="x0_pred")
        if not bool(torch.isfinite(x0_pred).all()):
            raise TrainingNumericalError("nonfinite_prediction", field="x0_pred")
        if not _rotation6d_valid(x0_pred[..., 3:9]):
            raise TrainingNumericalError("degenerate_prediction_rotation6d", field="x0_pred[3:9]")
        loss, parts = self.loss_fn(x0_pred, batch)
        self._check_losses(loss, parts)
        return loss, parts

    @contextmanager
    def _evaluation_state(self):
        """Restore independent modes and global RNG, including exceptional exits.

        Loader generators are deliberately untouched: validation still follows
        the accepted iterator flow and advances its own base-seed stream.
        """
        modes = [(m, m.training) for root in (self.model, self.conditioner) for m in root.modules()]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            self.model.eval(); self.conditioner.eval()
            yield
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def _formal_evidence(self, raw: dict) -> dict:
        gaps: list[str] = []
        expected = self.formal_expected_demos
        roster_valid = (expected is not None and len(expected) > 0
                        and all(isinstance(d, str) and d for d in expected)
                        and len(set(expected)) == len(expected))
        if not roster_valid:
            gaps.append("registered_roster_missing_or_invalid")
        expected = expected if roster_valid else ()
        rows = raw.get("per_demo")
        if not isinstance(rows, list):
            gaps.append("per_demo_not_a_list")
            rows = []
        row_names = [r.get("demo") for r in rows if isinstance(r, dict)]
        if (len(row_names) != len(rows) or any(not isinstance(d, str) for d in row_names)
                or len(set(d for d in row_names if isinstance(d, str))) != len(row_names)
                or len(rows) != len(expected) or set(d for d in row_names if isinstance(d, str)) != set(expected)):
            gaps.append("per_demo_roster_mismatch")
        values, ok_demos = [], set()
        lengths = self.formal_demo_lengths
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                gaps.append(f"row_not_a_mapping:{i}")
                continue
            demo, T, metric = row.get("demo"), row.get("T"), row.get("mpjpe_m")
            valid = (isinstance(demo, str) and demo in expected
                     and row.get("status") == "ok" and row.get("numerical_valid") is True
                     and _positive_int(T) and _finite_number(metric) and float(metric) >= 0)
            if lengths is not None:
                valid = valid and _positive_int(lengths.get(demo)) and T == lengths.get(demo)
            if not valid:
                gaps.append(f"invalid_demo_evidence:{demo if isinstance(demo, str) else i}")
            else:
                values.append(float(metric))
                ok_demos.add(demo)
        n_expected, n_ok = len(expected), len(ok_demos)
        if not _positive_int(raw.get("n_expected")) or raw["n_expected"] != n_expected:
            gaps.append("n_expected_mismatch")
        reported_ok = raw.get("n_ok")
        if (not isinstance(reported_ok, (int, np.integer)) or isinstance(reported_ok, (bool, np.bool_))
                or reported_ok != n_ok):
            gaps.append("n_ok_mismatch")
        macro = sum(values) / len(values) if values and len(values) == n_expected and not gaps else None
        reported_macro = raw.get("macro_mpjpe_m")
        if macro is not None and (not _finite_number(macro) or not _finite_number(reported_macro)
                                  or float(reported_macro) != macro):
            gaps.append("macro_not_reproducible")
        if raw.get("error") is not None:
            gaps.append("callback_error")
        if raw.get("budget_stopped") is True:
            gaps.append("budget_deadline")
        invalid: list[str] = []
        out = _json_evidence(raw, invalid)
        if invalid:
            out["nonfinite_evidence_fields"] = invalid
            gaps.append("nonfinite_formal_evidence")
        out.update(n_expected=n_expected, n_ok=n_ok,
                   expected_demos=list(expected),
                   macro_mpjpe_m=macro if not gaps else None,
                   complete=not gaps and n_ok == n_expected and n_expected > 0,
                   evidence_gaps=gaps)
        return out

    def _run_formal_eval(self) -> dict:
        """Callback errors retain incomplete evidence; no callback can pick best by assertion."""
        assert self.formal_eval is not None
        raw = {}
        try:
            with self._evaluation_state(), torch.no_grad():
                self._check_deadline("formal_eval_start")
                self._check_params()
                try:
                    signature = inspect.signature(self.formal_eval)
                except (TypeError, ValueError):
                    signature = None
                accepts_deadline = signature is not None and any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    or (p.name == "budget_deadline" and p.kind != inspect.Parameter.POSITIONAL_ONLY)
                    for p in signature.parameters.values())
                result = (self.formal_eval(self.model, self.conditioner, budget_deadline=self.budget_deadline)
                          if accepts_deadline else self.formal_eval(self.model, self.conditioner))
                if not isinstance(result, dict):
                    raise TypeError("formal callback result must be a mapping")
                raw = result
                self._synchronize()
                self._check_params()
                self._check_deadline("formal_eval_complete")
        except TrainingBudgetExceeded as exc:
            raw = {**raw, "budget_stopped": True, "error": str(exc)}
        except TrainingNumericalError:
            self._last_formal_evidence = self._formal_evidence({**raw, "error": "nonfinite_callback_parameters"})
            raise
        except Exception as exc:
            raw = {**raw, "error": f"{type(exc).__name__}: {exc}"}
        except BaseException as exc:
            self._last_formal_evidence = self._formal_evidence({**raw, "error": f"{type(exc).__name__}: {exc}"})
            raise
        self._last_formal_evidence = self._formal_evidence(raw)
        return self._last_formal_evidence

    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> dict:
        """All val windows, sample-weighted parts, fixed dedicated t/noise draws."""
        acc: dict[str, float] = {}
        n_samples, n_batches = 0, 0

        def evidence(complete: bool, error: str | None = None) -> dict:
            out = {k: v / max(n_samples, 1) for k, v in acc.items()}
            out.update(n_windows_evaluated=n_samples, n_val_windows_total=len(self.val_ds),
                       n_batches=n_batches, complete=complete)
            if error is not None:
                out["error"] = error
            return out

        try:
            with self._evaluation_state():
                self._check_deadline("validation_start")
                self._check_params()
                g = torch.Generator().manual_seed(
                    self.cfg.val_seed if self.cfg.val_seed is not None else self.cfg.seed)
                for i, batch in enumerate(self.val_loader):
                    if max_batches is not None and i >= max_batches:
                        break
                    self._check_deadline(f"validation_batch_{i + 1}_start")
                    batch = self._to_device(batch)
                    self._check_batch(batch)
                    loss, parts = self._forward(batch, noise_generator=g)
                    self._check_losses(loss, parts)
                    w = batch["x0"].shape[0]
                    for k, v in parts.items():
                        acc[k] = acc.get(k, 0.0) + v * w
                        if not _finite_number(acc[k]):
                            raise TrainingNumericalError("nonfinite_validation_aggregate", field=k)
                    n_samples += w
                    n_batches += 1
                    self._synchronize()
                    self._check_deadline(f"validation_batch_{i + 1}_complete")
                self._check_params()
                self._check_deadline("validation_complete")
                if n_samples == 0:
                    raise TrainingNumericalError("empty_validation", field="val_loader")
        except BaseException as exc:
            self._last_validation_evidence = evidence(False, f"{type(exc).__name__}: {exc}")
            raise
        self._last_validation_evidence = evidence(n_samples == len(self.val_ds))
        return self._last_validation_evidence

    def save(self, name: str, step: int, val: dict,
             identity: dict | None = None, formal: dict | None = None) -> None:
        """Atomic checkpoint; a failed replacement preserves an earlier valid best.

        Shutdown evidence may use the reserved tail after the work deadline,
        with unusable/uncertain parameters explicitly marked diagnostic.
        """
        if not self._preserving_evidence:
            self._check_deadline(f"save_{name}_start")
            self._check_params()
        if val.get("step", step) != step:
            raise ValueError("checkpoint validation evidence belongs to another step")
        finite = self._params_finite()
        usable = finite and not (self._failure or {}).get("optimizer_update_uncertain", False)
        payload = {
            "model": self.model.state_dict(), "conditioner": self.conditioner.state_dict(),
            "step": step, "val": val, "formal": formal, "config": asdict(self.cfg),
            "root_stats": {"mean": self.tokenizer.root_stats.mean, "std": self.tokenizer.root_stats.std},
            "identity": identity, "parameters_finite": finite, "checkpoint_usable": usable,
            "diagnostic_only": not usable, "failure": self._failure,
        }
        path = self.out_dir / name
        tmp = path.with_name(path.name + ".tmp")
        try:
            self._synchronize()
            torch.save(payload, tmp)
            self._synchronize()
            if not self._preserving_evidence:
                self._check_deadline(f"save_{name}_serialized")
            os.replace(tmp, path)
            self._checkpoint_records[name] = {
                "name": name, "role": "best" if name == "best.pt" else "last",
                "step": step, "parameters_finite": finite, "checkpoint_usable": usable,
            }
            if not self._preserving_evidence:
                self._check_deadline(f"save_{name}_complete")
        finally:
            if tmp.exists():
                tmp.unlink()

    def run(self) -> dict:
        cfg = self.cfg
        history, val_history, formal_history = [], [], []
        best_val, best_formal = float("inf"), float("inf")
        best_formal_step = None
        step, ms_ema = 0, None
        n_windows = len(self.train_ds)
        batches_per_epoch = n_windows // cfg.micro_batch if cfg.micro_batch else 0
        discarded_per_epoch = n_windows - batches_per_epoch * cfg.micro_batch
        micro_batches_total = samples_total = backward_micro_batches = epochs_started = epochs_completed = 0
        skipped_units: list[str] = []
        stop_reason, primary_exception = None, None
        preservation_errors: list[str] = []
        phase, n_micro, n_samples = "startup", 0, 0
        update_in_flight = False
        t_start = time.perf_counter()
        self._checkpoint_records = {}
        self._failure = None
        self._last_validation_evidence = self._last_formal_evidence = None
        try:
            self._check_params()
            data_iter = iter(self.train_loader)
            epochs_started = 1
            self.model.train(); self.conditioner.train()
            while step < cfg.steps:
                phase = "optimizer_step"
                self._check_deadline(f"optimizer_step_{step + 1}")
                self._check_params()
                self._synchronize()
                opt_step_t0 = time.perf_counter()
                self.opt.zero_grad(set_to_none=True)
                parts_acc: dict[str, float] = {}
                n_micro, n_samples = 0, 0
                for micro in range(cfg.grad_accum):
                    phase = f"micro_batch_{micro + 1}"
                    self._check_deadline(f"step_{step + 1}_{phase}_start")
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        epochs_completed += 1
                        data_iter = iter(self.train_loader)
                        epochs_started += 1
                        batch = next(data_iter)
                    micro_batches_total += 1
                    if isinstance(batch, dict) and torch.is_tensor(batch.get("x0")) and batch["x0"].ndim > 0:
                        consumed = int(batch["x0"].shape[0])
                        samples_total += consumed
                        n_samples += consumed
                    self._check_deadline(f"step_{step + 1}_{phase}_loaded")
                    batch = self._to_device(batch)
                    self._check_batch(batch)
                    loss, parts = self._forward(batch)
                    self._check_losses(loss, parts)
                    self._check_deadline(f"step_{step + 1}_{phase}_forward")
                    (loss / cfg.grad_accum).backward()
                    n_micro += 1
                    backward_micro_batches += 1
                    self._check_gradients()
                    for k, v in parts.items():
                        parts_acc[k] = parts_acc.get(k, 0.0) + v / cfg.grad_accum
                        if not _finite_number(parts_acc[k]):
                            raise TrainingNumericalError("nonfinite_loss_aggregate", field=k)
                    self._check_deadline(f"step_{step + 1}_{phase}_backward")
                phase = "gradient_clip"
                self._check_gradients()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [*self.model.parameters(), *self.conditioner.parameters()], cfg.grad_clip)
                if not _finite_number(gnorm):
                    raise TrainingNumericalError("nonfinite_gradnorm", field="gnorm")
                self._check_gradients()
                self._check_params()
                self._check_deadline(f"optimizer_update_{step + 1}")
                phase = "optimizer_update"
                update_in_flight = True
                self.opt.step()
                update_in_flight = False
                step += 1  # actual update returned; record its losses even if it poisoned parameters
                self._synchronize()
                wall = (time.perf_counter() - opt_step_t0) * 1e3
                ms_ema = wall if ms_ema is None else 0.98 * ms_ema + 0.02 * wall
                history.append({"step": step, "n_micro": n_micro, "n_samples": n_samples,
                                **{k: round(v, 6) for k, v in parts_acc.items()},
                                "gnorm": float(gnorm), "lr": cfg.lr, "ms": round(wall, 1)})
                self._check_params()
                self._check_deadline(f"optimizer_step_{step}_complete")
                if step % cfg.log_every == 0:
                    last = history[-1]
                    print(f"[{cfg.model_type}] step {step}/{cfg.steps} "
                          f"total={last['total']:.4f} fk={last.get('fk_m2', 0.0):.5f} "
                          f"geo={last['geodesic_rad2']:.5f} ms/step={ms_ema:.0f}", flush=True)
                if step % cfg.val_every == 0 or step == cfg.steps:
                    phase = "validation"
                    self._last_validation_evidence = None
                    val = self.validate()
                    val_history.append({"step": step, **val})
                    print(f"[{cfg.model_type}] VAL step {step}: total={val['total']:.4f}", flush=True)
                    phase = "save_last"
                    self.save("last.pt", step, val, identity=self.identity)
                    if self.formal_eval is None and val["total"] < best_val:
                        phase = "save_best"
                        self.save("best.pt", step, val, identity=self.identity)
                        best_val = val["total"]
                # Formal cadence is independent of loss-validation cadence.
                if self.formal_eval is not None and (step % self.formal_eval_every == 0 or step == cfg.steps):
                    phase = "formal_eval"
                    self._last_formal_evidence = None
                    formal = self._run_formal_eval()
                    formal_history.append({"step": step, **formal})
                    self._check_deadline(f"formal_eval_at_step_{step}")
                    if formal["complete"] and formal["macro_mpjpe_m"] < best_formal:
                        phase = "save_best"
                        # A skipped/different-step validation cannot be attached as current evidence.
                        val = val_history[-1] if val_history and val_history[-1]["step"] == step else {}
                        self.save("best.pt", step, val, identity=self.identity, formal=formal)
                        best_formal, best_formal_step = formal["macro_mpjpe_m"], step
        except BaseException as exc:
            if isinstance(exc, TrainingBudgetExceeded):
                stop_reason = "budget_deadline"
                skipped_units.append(exc.unit)
            elif isinstance(exc, TrainingNumericalError):
                stop_reason = f"{exc.reason}_at_step_{step if phase == 'optimizer_update' else step + 1}"
            else:
                stop_reason = f"exception:{type(exc).__name__}"
                primary_exception = (exc, exc.__traceback__)
            self._failure = {"phase": phase, "step": step,
                             "attempted_step": step + (phase.startswith("micro") or phase in ("optimizer_step", "gradient_clip", "optimizer_update")),
                             "n_micro_completed": n_micro, "n_samples_consumed": n_samples,
                             "optimizer_update_uncertain": update_in_flight,
                             "error": f"{type(exc).__name__}: {exc}",
                             **(exc.evidence if isinstance(exc, TrainingNumericalError) else {})}
            if phase == "validation" and self._last_validation_evidence is not None:
                val_history.append({"step": step, **self._last_validation_evidence})
            if phase == "formal_eval" and self._last_formal_evidence is not None:
                if not formal_history or formal_history[-1]["step"] != step:
                    formal_history.append({"step": step, **self._last_formal_evidence})

        # Preserve every independently writable artifact. A secondary save/write
        # failure must not replace the original exception or invent step zero.
        self._preserving_evidence = True
        try:
            last = self._checkpoint_records.get("last.pt")
            if last is None or last["step"] != step or self._failure is not None:
                val = (val_history[-1] if val_history and val_history[-1]["step"] == step
                       and val_history[-1].get("complete") is True else {})
                try:
                    self.save("last.pt", step, val, identity=self.identity)
                except BaseException as exc:
                    preservation_errors.append(f"last.pt:{type(exc).__name__}:{exc}")
                    if primary_exception is None:
                        primary_exception = (exc, exc.__traceback__)
            checkpoint_index = []
            for name in ("best.pt", "last.pt"):
                record = self._checkpoint_records.get(name)
                if record is None:
                    continue
                try:
                    sha = hashlib.sha256((self.out_dir / name).read_bytes()).hexdigest()
                    checkpoint_index.append({**record, "sha256": sha})
                except BaseException as exc:
                    preservation_errors.append(f"hash_{name}:{type(exc).__name__}:{exc}")
                    if primary_exception is None:
                        primary_exception = (exc, exc.__traceback__)
            # Selection is a successfully saved checkpoint, not an optimistic
            # assignment made before a failing/late save.
            best_record = self._checkpoint_records.get("best.pt")
            if self.formal_eval is not None and best_record is not None:
                chosen = next((f for f in formal_history if f["step"] == best_record["step"] and f["complete"]), None)
                best_formal_step = chosen["step"] if chosen else None
                best_formal = chosen["macro_mpjpe_m"] if chosen else float("inf")
            summary = {
                "config": asdict(cfg), "identity": self.identity, "steps": step,
                "wall_s": None, "ms_per_step_ema": round(ms_ema, 1) if ms_ema is not None else None,
                "peak_cuda_mem_gb": (round(torch.cuda.max_memory_allocated(self.cfg.device) / 2**30, 2)
                                     if torch.device(self.cfg.device).type == "cuda" else None),
                "best_val": best_val if math.isfinite(best_val) else None,
                "best_selection_rule": "formal_macro_mpjpe" if self.formal_eval is not None else "window_val_loss",
                "best_formal_macro_mpjpe_m": best_formal if self.formal_eval is not None and math.isfinite(best_formal) else None,
                "best_formal_step": best_formal_step,
                "selection_status": ("best_by_formal_macro" if best_formal_step is not None else
                                     "no_qualified_best" if self.formal_eval is not None else "window_loss_selection"),
                "budget_deadline_s": None if self.budget_deadline is None else round(self.budget_deadline - t_start, 1),
                "budget_stopped": stop_reason == "budget_deadline", "stop_reason": stop_reason,
                "skipped_units": skipped_units, "failure": self._failure,
                "preservation_errors": preservation_errors,
                "train_loader": {
                    "n_windows": n_windows, "micro_batch": cfg.micro_batch,
                    "batches_per_epoch": batches_per_epoch,
                    "drop_last_discarded_per_epoch": discarded_per_epoch,
                    "epochs_started": epochs_started, "epochs_completed": epochs_completed,
                    "micro_batches_total": micro_batches_total, "samples_total": samples_total,
                    "backward_micro_batches_total": backward_micro_batches,
                },
                "n_train_windows": n_windows, "n_val_windows": len(self.val_ds),
                "checkpoint_index": checkpoint_index,
            }
            if history:
                summary["first_100_steps_ms_mean"] = round(float(np.mean([h["ms"] for h in history[:100]])), 1)

            def refresh() -> None:
                if not self._budget_allows_new_unit():
                    summary["budget_stopped"] = True
                    if summary["stop_reason"] is None:
                        summary["stop_reason"] = "budget_deadline"
                summary["complete"] = (step == cfg.steps and summary["stop_reason"] is None
                                       and not preservation_errors)
                summary["status"] = "crashed" if primary_exception is not None else "completed" if summary["complete"] else "incomplete"
                summary["wall_s"] = round(time.perf_counter() - t_start, 1)

            def write_json(name, data) -> None:
                tmp = self.out_dir / (name + ".tmp")
                bad: list[str] = []
                safe = _json_evidence(data, bad)
                if bad:
                    raise ValueError(f"nonfinite persisted evidence:{bad}")
                tmp.write_text(json.dumps(safe, indent=2, allow_nan=False))
                os.replace(tmp, self.out_dir / name)

            for name, data in (("loss_history.json", history), ("val_history.json", val_history),
                               ("formal_history.json", formal_history), ("checkpoint_index.json", checkpoint_index)):
                try:
                    write_json(name, data)
                except BaseException as exc:
                    preservation_errors.append(f"{name}:{type(exc).__name__}:{exc}")
                    if primary_exception is None:
                        primary_exception = (exc, exc.__traceback__)
                refresh()  # includes hashing and each final evidence write
            refresh()
            try:
                write_json("summary.json", {**summary, "val_history": val_history, "formal_history": formal_history})
                was_complete = summary["complete"]
                refresh()
                if was_complete and not summary["complete"]:
                    write_json("summary.json", {**summary, "val_history": val_history, "formal_history": formal_history})
                    refresh()
            except BaseException as exc:
                preservation_errors.append(f"summary.json:{type(exc).__name__}:{exc}")
                if primary_exception is None:
                    primary_exception = (exc, exc.__traceback__)
                refresh()
        finally:
            self._preserving_evidence = False
        if primary_exception is not None:
            exc, traceback = primary_exception
            exc.training_summary = summary
            raise exc.with_traceback(traceback)
        return summary
