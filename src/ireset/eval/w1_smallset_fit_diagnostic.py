"""W1 I-phase small-set fit diagnostic (plan v0.4 sections 6/7).

Independent diagnostic machinery for the conditional small-set fitting study:
the D1/D6 fitting driver REUSES the audited ``Trainer`` update unit
(``predict``/``begin_update``/``accumulate_micro``/``clip_gradients``/
``step_optimizer``) -- it never re-implements forward/backward/AdamW -- and
adds the guards the plan requires around those units:

- §6.3 runtime hard gates: the frozen production runtime is applied and
  re-checked before every training/probe/replay unit; parameter/buffer
  dtype+device are verified after each construction or strict load; a
  synthetic-CPU guard (clearly labelled, no production equivalence claim)
  is used by the pure-synthetic acceptance path only.
- §7.1/§7.2 per-window physical metrics with float64 reductions over the
  frozen ``metric_decode_v1`` CPU decode, plus full spike visibility
  (per-frame curves, p95/max/worst coordinates); ``spike_gate_status`` is
  frozen to ``not_evaluated`` -- a mean-gate pass never claims spikes are
  excluded.
- §7.3 probe/save machinery: registered probes, dedicated probe generator,
  exactly one conditioner + one backbone forward per probe (delta-counted),
  initial/last diagnostic checkpoints.
- §7.4 items 1-3 local verification: coverage multiset, CPU recompute of
  every saved probe from disk raw x0 (bitwise decode binding), D1 handoff
  receipt consumed by D6; items 4-5 (fresh bitwise replay and mismatch
  verdicts) live in :func:`run_f_verification`.
- §7.6 status layering: stage reports only publish ``local_*`` fields and
  ``final_verification_status="pending_F"``; unprefixed final fields are
  validator-rejected.

This module contains NO real-data entry point.  The synthetic acceptance
(:func:`run_synthetic_acceptance`) builds an offline synthetic world; the
real R freeze and the real D1/D6/F executions are later, separately
authorized deliverables (plan §8.3).
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ireset.embodied.g1_spec import KPT_TO_SMPLH
from ireset.eval.w1_n0_coordinate_flags import (
    npz_members,
    sha256_file,
    verify_return_coverage,
    _write_summary_sealed,
)
from ireset.eval.w1_real_fk_audit import (
    RUNTIME_REQUIREMENTS,
    configure_production_runtime,
    environment_facts,
    runtime_facts_problems,
)
from ireset.kinematics.rotation_adapter import rot6d_to_mat_torch
from ireset.model.tokens import RootStats, StateTokenizer
from ireset.train.losses import LossAssembler, LossWeights
from ireset.train.trainer import Trainer, TrainConfig, TrainingNumericalError

SCHEMA = "w1_smallset_fit_diagnostic_v1"
STAGE_REPORT_SCHEMA = "w1_smallset_fit_stage_report_v1"
HANDOFF_SCHEMA = "w1_smallset_fit_handoff_v1"
F_REPORT_SCHEMA = "w1_smallset_fit_f_report_v1"
ARTIFACTS_INDEX_SCHEMA = "w1_smallset_fit_artifacts_index_v1"
PROBE_INDEX_SCHEMA = "w1_smallset_fit_probe_index_v1"
LEDGER_SCHEMA = "w1_smallset_fit_cost_ledger_v1"

METRIC_DECODE_SCHEMA = "metric_decode_v1"
METRIC_DECODE_DOMAIN = "cpu_float64_from_saved_x0"
PRODUCTION_DECODE_DOMAIN = "production_fp32"

T_WINDOW = 60
J_JOINTS = 29
S_SITES = len(KPT_TO_SMPLH)
assert S_SITES == 15, "FK site roster must stay the frozen 15-entry KPT_TO_SMPLH"

# §7.1 mean-fit gate values (frozen study targets, not acceptance thresholds).
GATE_ROOT_POS_RMSE_M = 0.01
GATE_ROOT_ROT_MEAN_RAD = math.pi / 180.0
GATE_JOINT_RMSE_RAD = 0.05
GATE_FK_MPJPE_M = 0.01

# §7 frozen fitting plan values (the real registered defaults).
FIT_STEPS = 1000
PROBE_EVERY = 10
GRAD_ACCUM = 4
D1_MICRO_BATCH = 1
D6_MICRO_BATCH = 6
PLAN_SEED = 20261008
PLAN_VAL_SEED = 20261008
PLAN_LR = 1e-4
PLAN_WEIGHT_DECAY = 1e-5
PLAN_GRAD_CLIP = 1.0
PLAN_D_MODEL, PLAN_N_LAYERS, PLAN_N_HEADS = 240, 4, 6
PLAN_FFN_MULT = 4
PLAN_TIME_WINDOW = 31
PLAN_NUM_TRAIN_TIMESTEPS = 1000

SPIKE_GATE_STATUS = "not_evaluated"
FK_REPLAY_SCOPE = "initial_last_only"

# §8.1 budget caps: (wall cap seconds, in-cap sealing reserve seconds).
BUDGET_CAPS = {"R": (120.0, 15.0), "D1": (1800.0, 60.0),
               "D6": (3600.0, 120.0), "F": (600.0, 60.0)}

# §6.2: the conditioner must never read the target state or supervision masks.
COND_LEAK_FORBIDDEN_KEYS = ("x0", "hand_contact_supervision_mask",
                            "hand_proximity_mask")

# Frozen saved-array record table (per window): name -> (shape, dtype).
RECORD_SHAPES: dict[str, tuple[tuple[int, ...], str]] = {
    "raw_x0_pred": ((T_WINDOW, 38), "float32"),
    "md_pos": ((T_WINDOW, 3), "float64"),
    "md_R": ((T_WINDOW, 3, 3), "float64"),
    "md_q": ((T_WINDOW, J_JOINTS), "float64"),
    "prod_pos": ((T_WINDOW, 3), "float32"),
    "prod_quat": ((T_WINDOW, 4), "float32"),
    "prod_q": ((T_WINDOW, J_JOINTS), "float32"),
    "fk_pred": ((T_WINDOW, S_SITES, 3), "float32"),
    "err_root_pos_t": ((T_WINDOW,), "float64"),
    "err_root_rot_t": ((T_WINDOW,), "float64"),
    "err_joint_tj": ((T_WINDOW, J_JOINTS), "float64"),
    "err_fk_ts": ((T_WINDOW, S_SITES), "float64"),
}
TEACHER_RECORDS = ("raw_x0_tgt", "md_pos", "md_R", "md_q",
                   "prod_pos", "prod_quat", "prod_q", "fk_tgt")
# the teacher raw target / FK target share the frozen pred shapes
TEACHER_TABLE = {**RECORD_SHAPES,
                 "raw_x0_tgt": RECORD_SHAPES["raw_x0_pred"],
                 "fk_tgt": RECORD_SHAPES["fk_pred"]}
COND_KEYS = ("kps_obj", "kps_clip", "box_t_clip", "box_quat", "contact")

I_WITHDRAW_ON_EXCEED = {
    "local_complete_evidence": False,
    "local_mean_fit_passed": None,
    "local_conclusion": "incomplete",
    "label_verdict": "budget_exceeded_after_seal",
}

F_WITHDRAW_ON_EXCEED = {
    "complete_evidence": False, "replay_passed": False,
    "mean_fit_passed": None, "verdict": "budget_exceeded_after_seal"}


class FitDiagnosticError(RuntimeError):
    """Fail-closed diagnostic error; the stage records it as a failure."""


class BudgetExceeded(FitDiagnosticError):
    """A unit may not start: the work deadline has passed."""


# --------------------------------------------------------------------------
# budget / runtime gates
# --------------------------------------------------------------------------

@dataclass
class StageBudget:
    """Monotonic work/hard deadline budget for one stage (plan §8.2)."""

    stage: str
    wall_cap_s: float
    reserve_s: float
    t0: float = field(default_factory=time.perf_counter)
    clock: Callable[[], float] = time.perf_counter
    exceeded: bool = False
    stopped_at: str = ""

    @property
    def work_deadline(self) -> float:
        return self.t0 + self.wall_cap_s - self.reserve_s

    @property
    def hard_deadline(self) -> float:
        return self.t0 + self.wall_cap_s

    def check(self, point: str) -> None:
        if self.clock() >= self.work_deadline:
            self.exceeded = True
            self.stopped_at = point
            raise BudgetExceeded(f"work deadline reached at {point}")

    def facts(self) -> dict:
        now = self.clock() - self.t0
        return {"stage": self.stage, "t0_perf_counter": self.t0,
                "wall_cap_s": self.wall_cap_s, "reserve_s": self.reserve_s,
                "work_deadline_s": self.wall_cap_s - self.reserve_s,
                "hard_deadline_s": self.wall_cap_s,
                "elapsed_s_at_facts": now,
                "budget_exceeded": self.exceeded, "stopped_at": self.stopped_at}


def budget_for_stage(stage: str, t0: float | None = None,
                     clock: Callable[[], float] = time.perf_counter) -> StageBudget:
    """The §8.1 budget for a registered stage."""
    cap, reserve = BUDGET_CAPS[stage]
    return StageBudget(stage=stage, wall_cap_s=cap, reserve_s=reserve,
                       t0=t0 if t0 is not None else clock(), clock=clock)


class ProductionRuntimeGuard:
    """§6.3: apply the audited production runtime and re-check per unit."""

    def __init__(self, expected_device: str = "cuda",
                 expected_dtype: torch.dtype = torch.float32) -> None:
        self.expected_device = expected_device
        self.expected_dtype = expected_dtype
        self.applied = configure_production_runtime()
        self.apply_problems = list(self.applied["problems"])

    def facts(self) -> dict:
        cur = {k: environment_facts().get(k) for k in RUNTIME_REQUIREMENTS}
        return {"kind": "production", "requirements": dict(RUNTIME_REQUIREMENTS),
                "applied": self.applied, "current": cur,
                "expected_device": self.expected_device,
                "expected_dtype": str(self.expected_dtype)}

    def check_unit(self, point: str) -> None:
        problems = runtime_facts_problems(environment_facts())
        if problems:
            raise FitDiagnosticError(f"runtime_drift:{point}:{sorted(problems)}")


class SyntheticRuntimeGuard:
    """Offline synthetic guard: records units, claims NO production parity."""

    def __init__(self) -> None:
        self.units_checked = 0

    def facts(self) -> dict:
        return {"kind": "synthetic_cpu",
                "production_equivalence_claimed": False,
                "requirements": dict(RUNTIME_REQUIREMENTS),
                "note": "synthetic acceptance path; the CUDA/float32 "
                        "production runtime is applied only by "
                        "ProductionRuntimeGuard"}

    def check_unit(self, point: str) -> None:
        self.units_checked += 1


def parameter_contract_problems(modules: dict[str, torch.nn.Module],
                                expected_device: str,
                                expected_dtype: torch.dtype) -> list[str]:
    """§6.3: floating parameters AND buffers must match the frozen contract."""
    problems: list[str] = []
    for side, module in modules.items():
        for name, p in module.named_parameters():
            if p.device.type != expected_device:
                problems.append(f"param_device:{side}.{name}:{p.device.type}")
            if p.dtype != expected_dtype:
                problems.append(f"param_dtype:{side}.{name}:{p.dtype}")
        for name, b in module.named_buffers():
            if b.is_floating_point():
                if b.device.type != expected_device:
                    problems.append(f"buffer_device:{side}.{name}:{b.device.type}")
                if b.dtype != expected_dtype:
                    problems.append(f"buffer_dtype:{side}.{name}:{b.dtype}")
    return problems


# --------------------------------------------------------------------------
# plan / registration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FitPlan:
    stage: str
    steps: int = FIT_STEPS
    probe_every: int = PROBE_EVERY
    grad_accum: int = GRAD_ACCUM
    seed: int = PLAN_SEED
    val_seed: int | None = PLAN_VAL_SEED
    model_type: str = "deterministic"
    lr: float = PLAN_LR
    weight_decay: float = PLAN_WEIGHT_DECAY
    grad_clip: float = PLAN_GRAD_CLIP
    d_model: int = PLAN_D_MODEL
    n_layers: int = PLAN_N_LAYERS
    n_heads: int = PLAN_N_HEADS
    ffn_mult: int = PLAN_FFN_MULT
    time_window: int = PLAN_TIME_WINDOW
    synthetic: bool = False

    def __post_init__(self) -> None:
        if self.stage not in ("D1", "D6"):
            raise FitDiagnosticError(f"unknown fit stage {self.stage!r}")
        if self.steps <= 0 or self.steps % self.probe_every != 0:
            raise FitDiagnosticError("steps must be positive and a multiple of probe_every")
        if self.probe_every <= 0 or self.grad_accum <= 0:
            raise FitDiagnosticError("probe_every/grad_accum must be positive")
        if not self.synthetic:
            frozen = {"steps": FIT_STEPS, "probe_every": PROBE_EVERY,
                      "grad_accum": GRAD_ACCUM, "seed": PLAN_SEED,
                      "val_seed": PLAN_VAL_SEED, "lr": PLAN_LR,
                      "weight_decay": PLAN_WEIGHT_DECAY,
                      "grad_clip": PLAN_GRAD_CLIP, "d_model": PLAN_D_MODEL,
                      "n_layers": PLAN_N_LAYERS, "n_heads": PLAN_N_HEADS,
                      "ffn_mult": PLAN_FFN_MULT, "time_window": PLAN_TIME_WINDOW}
            for key, want in frozen.items():
                if getattr(self, key) != want:
                    raise FitDiagnosticError(
                        f"non-synthetic plan deviates from frozen {key}")

    @property
    def micro_batch(self) -> int:
        return D1_MICRO_BATCH if self.stage == "D1" else D6_MICRO_BATCH

    @property
    def probe_steps(self) -> tuple[int, ...]:
        return tuple(range(0, self.steps + 1, self.probe_every))

    @property
    def n_probes(self) -> int:
        return len(self.probe_steps)

    def as_dict(self) -> dict:
        return asdict(self)


def frozen_plan_defaults_problems() -> list[str]:
    """The registered defaults must match the plan-frozen constants."""
    problems = []
    d1 = FitPlan("D1")
    d6 = FitPlan("D6")
    if (d1.micro_batch, d6.micro_batch) != (D1_MICRO_BATCH, D6_MICRO_BATCH):
        problems.append("micro_batch_defaults")
    if d1.n_probes != FIT_STEPS // PROBE_EVERY + 1:
        problems.append("probe_count_default")
    if d1.probe_steps[0] != 0 or d1.probe_steps[-1] != FIT_STEPS:
        problems.append("probe_schedule_endpoints")
    if d1.steps != 1000 or d1.probe_every != 10:
        problems.append("steps_probe_defaults")
    if d1.grad_accum != GRAD_ACCUM:
        problems.append("grad_accum_default")
    if (d1.lr, d1.weight_decay, d1.grad_clip, d1.seed) != (
            PLAN_LR, PLAN_WEIGHT_DECAY, PLAN_GRAD_CLIP, PLAN_SEED):
        problems.append("optimizer_defaults")
    if d6.steps != d1.steps or d6.micro_batch != D6_MICRO_BATCH:
        problems.append("d6_defaults")
    return problems


@dataclass(frozen=True)
class WindowKey:
    demo: str
    start: int
    end: int

    @property
    def window_id(self) -> str:
        return f"{self.demo}:{self.start}:{self.end}"


def canonical_window_order(windows) -> tuple:
    return tuple(sorted(windows, key=lambda w: (w.demo, w.start, w.end)))


@dataclass(frozen=True)
class FitRegistration:
    """The registered window set for one fit stage (plan §3.2 shape)."""

    stage: str
    windows: tuple[WindowKey, ...]
    splits: tuple[str, ...]

    def problems(self) -> list[str]:
        problems: list[str] = []
        if not self.windows:
            return ["registration_empty"]
        if len(self.windows) != len(self.splits):
            problems.append("registration_split_length_mismatch")
        ids = [w.window_id for w in self.windows]
        if len(set(ids)) != len(ids):
            problems.append("registration_duplicate_window")
        if list(self.windows) != list(canonical_window_order(self.windows)):
            problems.append("registration_not_canonical_order")
        want = 1 if self.stage == "D1" else 6
        if len(self.windows) != want:
            problems.append(f"registration_window_count:{len(self.windows)}:want{want}")
        for w, split in zip(self.windows, self.splits):
            if not isinstance(w.demo, str) or not w.demo:
                problems.append("registration_demo_invalid")
            if w.start != 0 or w.end - w.start != T_WINDOW:
                problems.append(f"registration_window_span:{w.window_id}")
            if split != "train":
                problems.append(f"registration_split_not_train:{w.window_id}:{split}")
        return problems

    @property
    def window_ids(self) -> tuple[str, ...]:
        return tuple(w.window_id for w in self.windows)

    def as_dict(self) -> dict:
        return {"stage": self.stage,
                "windows": [{"demo": w.demo, "start": w.start, "end": w.end,
                             "split": s}
                            for w, s in zip(self.windows, self.splits)]}


# --------------------------------------------------------------------------
# metric_decode_v1 (plan §7.5)
# --------------------------------------------------------------------------

def _rot6d_valid_f64(r6: np.ndarray) -> np.ndarray:
    """Mirror of the trainer's rotation-validity predicate, in float64."""
    a, b = r6[..., :3], r6[..., 3:6]
    na = np.linalg.norm(a, axis=-1)
    orth = b - (a * b).sum(-1, keepdims=True) * a / np.maximum(
        (a * a).sum(-1, keepdims=True), 1e-12)
    nb = np.linalg.norm(orth, axis=-1)
    nc = np.linalg.norm(np.cross(a, orth), axis=-1)
    return (na > 1e-6) & (nb > 1e-6) & (nc > 1e-12)


class MetricDecoder:
    """``metric_decode_v1``: canonical CPU float64 decode of saved x0.

    Constants are EXACT promotions of the production-effective float32
    values; the 6D decode reuses the existing dtype-generic
    ``rot6d_to_mat_torch`` on CPU float64; the joint map is the clean-side
    production map (clamp into [-1, 1]).  Saves root position, root rotation
    MATRIX and 29 physical joint angles in the frozen clip frame.  No
    network, FK or optimizer is constructed here.
    """

    SCHEMA = METRIC_DECODE_SCHEMA
    DOMAIN = METRIC_DECODE_DOMAIN

    def __init__(self, mean_f32, std_f32, lo_f32, hi_f32) -> None:
        self.mean = np.asarray(mean_f32, dtype=np.float32).astype(np.float64)
        self.std = np.asarray(std_f32, dtype=np.float32).astype(np.float64)
        self.lo = np.asarray(lo_f32, dtype=np.float32).astype(np.float64)
        self.hi = np.asarray(hi_f32, dtype=np.float32).astype(np.float64)
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise FitDiagnosticError("metric_decode_root_stats_shape")
        if self.lo.shape != (J_JOINTS,) or self.hi.shape != (J_JOINTS,):
            raise FitDiagnosticError("metric_decode_joint_range_shape")

    @property
    def source_constants_f32(self) -> dict:
        return {"mean": np.ascontiguousarray(self.mean.astype(np.float32)),
                "std": np.ascontiguousarray(self.std.astype(np.float32)),
                "lo": np.ascontiguousarray(self.lo.astype(np.float32)),
                "hi": np.ascontiguousarray(self.hi.astype(np.float32))}

    def verify_x0(self, x0) -> list[str]:
        problems: list[str] = []
        if not isinstance(x0, np.ndarray):
            return ["x0_not_array"]
        if x0.dtype != np.float32:
            problems.append(f"x0_dtype:{x0.dtype}")
        if x0.ndim != 2 or x0.shape != (T_WINDOW, 38):
            problems.append(f"x0_shape:{tuple(x0.shape)}")
            return problems
        if not np.isfinite(x0).all():
            problems.append("x0_nonfinite")
            return problems
        if not bool(_rot6d_valid_f64(x0[:, 3:9]).all()):
            problems.append("x0_degenerate_rotation6d")
        return problems

    def decode(self, x0: np.ndarray) -> dict:
        problems = self.verify_x0(x0)
        if problems:
            raise FitDiagnosticError(f"metric_decode_input_invalid:{problems}")
        x = x0.astype(np.float64)
        pos = x[:, 0:3] * self.std + self.mean
        R = rot6d_to_mat_torch(
            torch.from_numpy(np.ascontiguousarray(x[:, 3:9]))).numpy()
        u = np.clip(x[:, 9:38], -1.0, 1.0)
        q = self.lo + (u + 1.0) * (self.hi - self.lo) / 2.0
        return {"pos": pos, "R": R, "q": q}


def verify_rotation_matrix(R: np.ndarray) -> list[str]:
    problems: list[str] = []
    if not isinstance(R, np.ndarray) or R.dtype != np.float64 \
            or R.shape != (T_WINDOW, 3, 3):
        return ["rotation_matrix_shape_or_dtype"]
    if not np.isfinite(R).all():
        return ["rotation_matrix_nonfinite"]
    eye = np.eye(3, dtype=np.float64)
    orth = float(np.abs(np.einsum("tij,tkj->tik", R, R) - eye).max())
    det = np.linalg.det(R)
    if orth > 1e-6:
        problems.append(f"rotation_matrix_not_orthonormal:{orth:.3e}")
    if float(np.abs(det - 1.0).max()) > 1e-6:
        problems.append(f"rotation_matrix_det:{float(np.abs(det - 1.0).max()):.3e}")
    return problems


def so3_geodesic_angles_rad(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Per-frame shortest SO(3) geodesic angle in [0, pi] (scipy)."""
    rel = Rotation.from_matrix(R_a) * Rotation.from_matrix(R_b).inv()
    return rel.magnitude()


def _p95_linear(x: np.ndarray) -> float:
    return float(np.quantile(np.asarray(x, dtype=np.float64), 0.95,
                             method="linear"))


def _max_first(x: np.ndarray) -> tuple[float, tuple[int, ...]]:
    """Max with ties broken by first occurrence in C-order (min frame, then
    joint/site fixed order)."""
    arr = np.asarray(x, dtype=np.float64)
    flat = arr.reshape(-1)
    idx = int(np.argmax(flat))
    coord = tuple(int(v) for v in np.unravel_index(idx, arr.shape))
    return float(flat[idx]), coord


def window_spike_stats(err: np.ndarray) -> dict:
    m, coord = _max_first(err)
    return {"p95": _p95_linear(err), "max": m, "worst_coordinate": list(coord)}


def compute_window_metrics(dec_pred: dict, dec_tgt: dict,
                           fk_pred: np.ndarray, fk_tgt: np.ndarray) -> dict:
    """§7.1 four per-window metrics + §7.2 spike visibility (float64).

    Root position/rotation/joint consume the (re-)decoded values; FK MPJPE
    consumes the saved production FP32 FK points (float64 reduction only).
    """
    for R in (dec_pred["R"], dec_tgt["R"]):
        problems = verify_rotation_matrix(R)
        if problems:
            raise FitDiagnosticError(f"metric_rotation_invalid:{problems}")
    err_root_pos_t = np.linalg.norm(dec_pred["pos"] - dec_tgt["pos"], axis=1)
    root_pos_rmse = float(np.sqrt(np.mean(err_root_pos_t ** 2)))
    err_root_rot_t = so3_geodesic_angles_rad(dec_pred["R"], dec_tgt["R"])
    root_rot_mean = float(np.mean(err_root_rot_t))
    err_joint_tj = np.abs(dec_pred["q"] - dec_tgt["q"])
    joint_rmse = float(np.sqrt(np.mean(err_joint_tj ** 2)))
    err_fk_ts = np.linalg.norm(fk_pred.astype(np.float64)
                               - fk_tgt.astype(np.float64), axis=-1)
    fk_mpjpe = float(np.mean(err_fk_ts))
    gates = {
        "root_position_rmse_m": root_pos_rmse <= GATE_ROOT_POS_RMSE_M,
        "root_rotation_mean_rad": root_rot_mean <= GATE_ROOT_ROT_MEAN_RAD,
        "joint_rmse_rad": joint_rmse <= GATE_JOINT_RMSE_RAD,
        "teacher_fk_mpjpe_m": fk_mpjpe <= GATE_FK_MPJPE_M,
    }
    return {
        "root_position_rmse_m": root_pos_rmse,
        "root_rotation_mean_rad": root_rot_mean,
        "joint_rmse_rad": joint_rmse,
        "teacher_fk_mpjpe_m": fk_mpjpe,
        "gates": gates,
        "window_mean_fit_passed": bool(all(gates.values())),
        "spike": {
            "root_position_per_frame_m": window_spike_stats(err_root_pos_t),
            "root_rotation_per_frame_rad": window_spike_stats(err_root_rot_t),
            "joint_per_frame_per_joint_rad": window_spike_stats(err_joint_tj),
            "fk_per_frame_per_site_m": window_spike_stats(err_fk_ts),
        },
    }


def normalized_group_errors(x0_pred: np.ndarray, x0_tgt: np.ndarray) -> dict:
    """Unweighted pos/rot6d/joint normalized-space error groups (S0 §6.2-4)."""
    d2 = (x0_pred.astype(np.float64) - x0_tgt.astype(np.float64)) ** 2
    return {"pos": float(d2[:, 0:3].mean()), "rot6d": float(d2[:, 3:9].mean()),
            "joint": float(d2[:, 9:38].mean()),
            "motion_reconstruction": float(d2[:, 0:3].mean() + d2[:, 3:9].mean()
                                           + d2[:, 9:38].mean())}


def aggregate_local_verdicts(final_windows: list[dict]) -> dict:
    """Step-final aggregation: the macro average NEVER substitutes the
    per-window conjunction."""
    per_window = [bool((w.get("metrics") or {}).get("window_mean_fit_passed"))
                  for w in final_windows]
    macro = None
    if final_windows:
        values = [w.get("metrics", {}).get("root_position_rmse_m")
                  for w in final_windows]
        if all(isinstance(v, float) for v in values):
            macro = float(np.mean(values))
    return {"local_mean_fit_passed": (bool(all(per_window))
                                      if per_window else None),
            "n_windows": len(final_windows),
            "macro_root_position_rmse_m": macro,
            "macro_is_not_a_gate": True}


# --------------------------------------------------------------------------
# guards: conditioner leak, counting wrappers, RNG
# --------------------------------------------------------------------------

class _AccessTrackingDict(dict):
    def __init__(self, base: dict) -> None:
        super().__init__(base)
        self.accessed: set[str] = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed.add(key)
        return super().get(key, default)


def conditioner_leak_guard_forward(inner: torch.nn.Module, batch: dict):
    """Reject, at the conditioner boundary, any read of the target state or
    supervision masks (plan §6.2 target-leak counterexample)."""
    if not isinstance(batch, dict):
        raise FitDiagnosticError("conditioner_batch_not_a_mapping")
    tracked = _AccessTrackingDict(batch)
    original = getattr(inner, "_original_forward", None)
    if original is None:
        original = types.MethodType(type(inner).forward, inner)
    out = original(tracked)
    leaked = sorted(k for k in tracked.accessed if k in COND_LEAK_FORBIDDEN_KEYS)
    if leaked:
        raise FitDiagnosticError(f"conditioner_target_leak:{leaked}")
    return out


def patch_conditioner_leak_guard(conditioner: torch.nn.Module) -> None:
    if not hasattr(conditioner, "_original_forward"):
        conditioner._original_forward = conditioner.forward
    conditioner.forward = types.MethodType(conditioner_leak_guard_forward,
                                           conditioner)


class CountingFK:
    """Duck-type wrapper counting production FK batch calls."""

    def __init__(self, inner, counter: dict) -> None:
        self._inner = inner
        self._counter = counter

    def site_world_positions(self, *args, **kwargs):
        self._counter["production_fk_batch_calls"] = \
            self._counter.get("production_fk_batch_calls", 0) + 1
        return self._inner.site_world_positions(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class CountingLoss:
    """Duck-type wrapper counting loss evaluations; delegates attributes."""

    def __init__(self, inner, counter: dict) -> None:
        self._inner = inner
        self._counter = counter

    def __call__(self, *args, **kwargs):
        self._counter["loss_calls"] = self._counter.get("loss_calls", 0) + 1
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _forward_hook_counter(module: torch.nn.Module, counter: dict, key: str):
    def hook(_module, _inputs):
        counter[key] = counter.get(key, 0) + 1

    return module.register_forward_pre_hook(hook)


def _rng_atom_equal(a, b) -> bool:
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) \
            and type(a) is type(b):
        return len(a) == len(b) and all(
            _rng_atom_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) \
            and a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def capture_global_rng() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def global_rng_equal(a: dict, b: dict) -> bool:
    for key in ("python", "numpy", "torch_cpu", "torch_cuda"):
        if (key in a) != (key in b):
            return False
        if key in a and not _rng_atom_equal(a[key], b[key]):
            return False
    return True


# --------------------------------------------------------------------------
# saved-array IO
# --------------------------------------------------------------------------

def _check_record(record_name: str, key: str, arr, table: dict) -> list[str]:
    want_shape, want_dtype = table[record_name]
    problems: list[str] = []
    if not isinstance(arr, np.ndarray):
        return [f"record_not_array:{key}"]
    if arr.dtype != np.dtype(want_dtype):
        problems.append(f"record_dtype:{key}:{arr.dtype}")
    if arr.shape != want_shape:
        problems.append(f"record_shape:{key}:{tuple(arr.shape)}:want{want_shape}")
    if arr.dtype != np.bool_ and not np.issubdtype(arr.dtype, np.integer) \
            and not np.isfinite(arr).all():
        problems.append(f"record_nonfinite:{key}")
    return problems


def write_npz_records(path: Path, records: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **records)


def read_npz_records(path: Path, table: dict, prefix_map: dict[str, str]) -> dict:
    """Read records declared by ``prefix_map`` (record -> npz key), enforcing
    the frozen shape/dtype table before any value is consumed."""
    with np.load(path) as data:
        out = {}
        for record, key in prefix_map.items():
            if key not in data:
                raise FitDiagnosticError(f"record_missing:{Path(path).name}:{key}")
            arr = data[key]
            problems = _check_record(record, key, arr, table)
            if problems:
                raise FitDiagnosticError(f"record_invalid:{Path(path).name}:{problems}")
            out[record] = arr
    return out


# late artifacts whose bytes are bound by SHA FIELDS elsewhere (report ←
# return proof + candidate; proof ← report bytes; candidate ← consumed by
# resolve_handoff): they are written after the index and must never appear
# in it, or the index↔report SHA chain would be circular
LATE_BOUND_ARTIFACTS = ("stage_report.json", "return_observation.json",
                        "handoff_candidate.json")


def build_artifacts_index(stage_dir: Path) -> dict:
    files = {}
    for p in sorted(Path(stage_dir).rglob("*")):
        if not p.is_file() or p.name == "artifacts_index.json" \
                or p.name in LATE_BOUND_ARTIFACTS:
            continue
        files[str(p.relative_to(stage_dir))] = {
            "sha256": sha256_file(p), "bytes": p.stat().st_size}
    return {"schema": ARTIFACTS_INDEX_SCHEMA, "files": files}


def verify_artifacts_index(stage_dir: Path, index: dict,
                           required: Sequence[str] | None = None) -> list[str]:
    problems: list[str] = []
    if not isinstance(index, dict) or index.get("schema") != ARTIFACTS_INDEX_SCHEMA:
        return ["artifacts_index_schema"]
    files = index.get("files") or {}
    # a required file absent from the index is a structural failure even
    # though the (possibly empty) iteration below finds nothing to check
    for rel in required or ():
        if rel not in files:
            problems.append(f"artifact_required_missing:{rel}")
    for rel, meta in files.items():
        p = Path(stage_dir) / rel
        if not p.is_file():
            problems.append(f"artifact_missing:{rel}")
            continue
        if p.stat().st_size != meta.get("bytes"):
            problems.append(f"artifact_bytes:{rel}")
        if sha256_file(p) != meta.get("sha256"):
            problems.append(f"artifact_sha256:{rel}")
    return problems


# --------------------------------------------------------------------------
# attempt ledger (identity / billing)
# --------------------------------------------------------------------------

class AttemptLedger:
    """One row per (stage, attempt), pre-registered running, updated in place.

    A crash (SIGKILL) leaves a ``running`` row; disk attempt directories and
    ledger rows must agree; unknown history is a stop condition, never a
    zero (plan §8.2, S0 precedent)."""

    def __init__(self, run_root: Path, stage: str) -> None:
        self.run_root = Path(run_root)
        self.stage = stage
        self.path = self.run_root / "costs.json"

    def _rows(self) -> list[dict]:
        if not self.path.is_file():
            return []
        doc = json.loads(self.path.read_text())
        rows = doc.get("invocations") if isinstance(doc, dict) else None
        return rows if isinstance(rows, list) else []

    def prior_state(self) -> tuple[float | None, list[str]]:
        """(prior cost seconds | None=unknown, problems)."""
        problems: list[str] = []
        stage_dir = self.run_root / self.stage
        disk = ({d.name for d in stage_dir.glob("attempt_*") if d.is_dir()}
                if stage_dir.exists() else set())
        ledger: dict[str, float] = {}
        total = 0.0
        for row in self._rows():
            if row.get("stage") != self.stage:
                continue
            attempt = row.get("attempt")
            if attempt in ledger:
                return None, [f"ledger_duplicate_attempt:{attempt}"]
            wall = row.get("wall_s")
            if not isinstance(wall, (int, float)) or isinstance(wall, bool) \
                    or not math.isfinite(float(wall)) or float(wall) < 0:
                return None, [f"ledger_cost_invalid:{attempt}:{wall!r}"]
            if row.get("status") == "running":
                return None, [f"ledger_running_row:{attempt}"]
            ledger[attempt] = float(wall)
            total += float(wall)
        if disk != set(ledger):
            problems.append(f"ledger_disk_disagreement:disk{sorted(disk)}:"
                            f"ledger{sorted(ledger)}")
            return None, problems
        if not ledger:
            return 0.0, []
        return total, []

    def begin(self, attempt: str) -> None:
        self._upsert(attempt, 0.0, {}, status="running")

    def finish(self, attempt: str, wall_s: float, status: str,
               extra: dict | None = None) -> None:
        self._upsert(attempt, wall_s, extra or {}, status=status)

    def _upsert(self, attempt: str, wall: float, extra: dict, status: str) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        rows = self._rows()
        row = {"stage": self.stage, "attempt": attempt, "status": status,
               "wall_s": round(float(wall), 6), **extra}
        for i, existing in enumerate(rows):
            if existing.get("stage") == self.stage and existing.get("attempt") == attempt:
                keep_ts = existing.get("timestamp")
                rows[i] = {**row, "timestamp": keep_ts}
                break
        else:
            rows.append(row)
        self.path.write_text(json.dumps({"schema": LEDGER_SCHEMA,
                                         "invocations": rows}, indent=1))


def remaining_budget_or_refuse(run_root: Path, stage: str,
                               budget: StageBudget) -> float:
    """Refuse to start when history is unknown or the remaining wall cannot
    cover the cap; a failed attempt never resets the budget (§6.2)."""
    ledger = AttemptLedger(run_root, stage)
    prior, problems = ledger.prior_state()
    if prior is None:
        raise FitDiagnosticError(f"budget_history_unknown:{stage}:{problems}")
    remaining = budget.wall_cap_s - prior
    if remaining <= budget.reserve_s:
        raise FitDiagnosticError(
            f"budget_exhausted:{stage}:prior{prior:.3f}:cap{budget.wall_cap_s}")
    return remaining


# --------------------------------------------------------------------------
# R-phase metadata-only guard (plan §3.2)
# --------------------------------------------------------------------------

class MetadataOnlyReader:
    """R may only read registration/manifest/cache metadata: JSON files and
    NPZ headers.  Decoding array values before a billed D window is a hard
    refusal."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def read_json(self, path):
        self.reads.append(f"json:{path}")
        return json.loads(Path(path).read_text())

    def npz_members(self, path) -> dict:
        self.reads.append(f"npz_header:{path}")
        return npz_members(Path(path))

    def load_arrays(self, path):
        raise FitDiagnosticError("r_metadata_only_array_read_forbidden")

    def torch_load(self, path):
        raise FitDiagnosticError("r_metadata_only_checkpoint_read_forbidden")


def materialize_for_fit(world, billing: StageBudget | None):
    """Array materialization is only legal inside a started, not-exceeded
    D1/D6 billing interval; R and inter-stage contexts are refused."""
    if billing is None:
        raise FitDiagnosticError("materialization_without_billed_interval")
    if billing.stage not in ("D1", "D6"):
        raise FitDiagnosticError(f"materialization_stage_not_fit:{billing.stage}")
    if billing.exceeded:
        raise FitDiagnosticError("materialization_after_budget_exceeded")
    billing.check("materialize_for_fit")
    return world


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def probe_generator_seed(plan: FitPlan) -> int:
    return plan.val_seed if plan.val_seed is not None else plan.seed


def run_probe(*, trainer: Trainer, probe_batch: dict, plan: FitPlan, step: int,
              probe_idx: int, probe_gen: torch.Generator, training_gen,
              loss_fn, budget: StageBudget, counters: dict,
              decoder: MetricDecoder, teacher: dict | None,
              window_ids: Sequence[str]) -> tuple[dict, dict]:
    """One fixed probe: exactly one conditioner + one backbone forward over
    the whole registered batch, eval/no_grad, global RNG saved and restored.

    Returns (index entry, arrays dict).  ``teacher`` carries the per-window
    frozen references (computed once per stage by the caller).
    """
    point = f"probe_{probe_idx}_step_{step}"
    budget.check(point)
    if probe_gen is training_gen:
        raise FitDiagnosticError("probe_generator_is_training_generator")
    if not isinstance(probe_gen, torch.Generator):
        raise FitDiagnosticError("probe_generator_not_a_torch_generator")
    cond_before = counters.get("conditioner_forward_calls", 0)
    back_before = counters.get("backbone_forward_calls", 0)
    rng_before = capture_global_rng()
    probe_gen.manual_seed(probe_generator_seed(plan))
    batch = {k: (v.to(trainer.cfg.device) if torch.is_tensor(v) else v)
             for k, v in probe_batch.items()}
    try:
        with trainer._evaluation_state(), torch.no_grad():
            _, x0_pred = trainer.predict(batch, noise_generator=probe_gen)
    finally:
        cond_delta = counters.get("conditioner_forward_calls", 0) - cond_before
        back_delta = counters.get("backbone_forward_calls", 0) - back_before
    if cond_delta != 1 or back_delta != 1:
        raise FitDiagnosticError(
            "probe_forward_count_violation:conditioner"
            f"{cond_delta}:backbone{back_delta}")
    rng_after = capture_global_rng()
    rng_ok = global_rng_equal(rng_before, rng_after)
    if not rng_ok:
        raise FitDiagnosticError("probe_rng_violation:global_rng_changed")

    x0_pred_np = x0_pred.detach().to("cpu").numpy()
    x0_tgt_np = batch["x0"].detach().to("cpu").numpy()
    B = x0_pred_np.shape[0]
    if probe_batch["x0"].shape[0] != B or len(window_ids) != B:
        raise FitDiagnosticError("probe_batch_shape_changed")
    windows_out, arrays = [], {}
    for b in range(B):
        raw = np.ascontiguousarray(x0_pred_np[b])
        tgt = np.ascontiguousarray(x0_tgt_np[b])
        problems = decoder.verify_x0(raw) + decoder.verify_x0(tgt)
        if problems:
            raise FitDiagnosticError(f"probe_x0_invalid:{problems}")
        counters["metric_decode_calls"] = \
            counters.get("metric_decode_calls", 0) + 1
        dec_p = decoder.decode(raw)
        dec_t = teacher["md"][b]
        counters["production_decode_calls"] = \
            counters.get("production_decode_calls", 0) + 1
        pos_p, quat_p, q_p = loss_fn.decode_clip(x0_pred.detach()[b:b + 1])
        flat = lambda a: a.reshape(-1, a.shape[-1])
        kp_p = loss_fn.fk.site_world_positions(
            flat(pos_p), flat(quat_p), flat(q_p), list(KPT_TO_SMPLH)
        ).view(T_WINDOW, S_SITES, 3)
        kp_p_np = kp_p.detach().cpu().numpy()
        kp_t_np = np.asarray(teacher["fk_tgt"][b])
        metrics = compute_window_metrics(dec_p, dec_t, kp_p_np, kp_t_np)
        err_fk_ts = np.linalg.norm(kp_p_np.astype(np.float64)
                                   - kp_t_np.astype(np.float64), axis=-1)
        prefix = f"w{b}__"
        arrays.update({
            prefix + "raw_x0_pred": raw,
            prefix + "md_pos": dec_p["pos"], prefix + "md_R": dec_p["R"],
            prefix + "md_q": dec_p["q"],
            prefix + "prod_pos": pos_p[0].detach().cpu().numpy(),
            prefix + "prod_quat": quat_p[0].detach().cpu().numpy(),
            prefix + "prod_q": q_p[0].detach().cpu().numpy(),
            prefix + "fk_pred": kp_p_np,
            prefix + "err_root_pos_t": np.linalg.norm(
                dec_p["pos"] - dec_t["pos"], axis=1),
            prefix + "err_root_rot_t": so3_geodesic_angles_rad(
                dec_p["R"], dec_t["R"]),
            prefix + "err_joint_tj": np.abs(dec_p["q"] - dec_t["q"]),
            prefix + "err_fk_ts": err_fk_ts,
        })
        windows_out.append({
            "window_id": window_ids[b],
            "batch_pos": b,
            "metrics": metrics,
            "normalized_groups": normalized_group_errors(raw, tgt),
            "spike_gate_status": SPIKE_GATE_STATUS,
        })
    entry = {"idx": probe_idx, "step": step,
             "forward_counts": {"conditioner": cond_delta, "backbone": back_delta},
             "probe_rng": {"global_rng_unchanged": rng_ok,
                           "dedicated_generator": True},
             "windows": windows_out}
    budget.check(point + "_complete")
    return entry, arrays


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

class FitDriver:
    """D1/D6 fitting over the audited Trainer update units (plan §7).

    The driver never re-implements the objective: every micro goes through
    ``trainer.accumulate_micro`` (forward + scaled backward + finite checks),
    clipping through ``trainer.clip_gradients`` and the update through
    ``trainer.step_optimizer``.  Budget guards bracket every unit because the
    shared unit methods themselves do not check deadlines.
    """

    def __init__(self, *, trainer: Trainer, plan: FitPlan,
                 registration: FitRegistration, loss_fn,
                 decoder: MetricDecoder, probe_batch: dict,
                 budget: StageBudget, runtime_guard, identity: dict,
                 stage_dir: Path, training_gen: torch.Generator | None = None,
                 authorization_includes_d6: bool = False,
                 dataset_window_keys: Sequence[WindowKey] | None = None) -> None:
        if registration.stage != plan.stage:
            raise FitDiagnosticError("registration_stage_mismatch")
        registration_problems = registration.problems()
        if registration_problems:
            raise FitDiagnosticError(
                f"registration_invalid:{registration_problems}")
        if dataset_window_keys is not None:
            got = [w.window_id for w in canonical_window_order(dataset_window_keys)]
            want = [w.window_id for w in registration.windows]
            if got != want:
                raise FitDiagnosticError(
                    f"dataset_registration_window_mismatch:got{got[:4]}:want{want[:4]}")
        self._check_trainer_config(trainer, plan)
        if float(trainer.cfg.loss.w_contact) != 0.0:
            raise FitDiagnosticError("contact_loss_must_be_disabled")
        self.trainer, self.plan, self.registration = trainer, plan, registration
        self.decoder, self.budget = decoder, budget
        self.runtime_guard = runtime_guard
        self.identity = dict(identity)
        self.stage_dir = Path(stage_dir)
        self.authorization_includes_d6 = bool(authorization_includes_d6)
        self.counters: dict = {"conditioner_forward_calls": 0,
                               "backbone_forward_calls": 0,
                               "loss_calls": 0, "adamw_step_calls": 0,
                               "backward_calls": 0, "clip_calls": 0,
                               "probe_calls": 0, "checkpoint_saves": 0,
                               "production_decode_calls": 0,
                               "production_fk_batch_calls": 0,
                               "metric_decode_calls": 0,
                               "window_consumptions_total": 0,
                               "epochs_restarted": 0}
        # production-chain counting wrappers (audit-only; every call is
        # delegated unchanged so production numbers are untouched).  The FK
        # wrapper must sit on the INNER loss assembler because the loss
        # executes its own ``self.fk`` calls; rebinding is idempotent for
        # worlds whose loss assembler is shared across drivers.
        self.loss_inner = loss_fn
        self.loss_fn = CountingLoss(loss_fn, self.counters)
        trainer.loss_fn = self.loss_fn
        inner_fk = getattr(loss_fn, "fk", None)
        if isinstance(inner_fk, CountingFK):
            inner_fk = inner_fk._inner
        if inner_fk is not None:
            loss_fn.fk = CountingFK(inner_fk, self.counters)
        patch_conditioner_leak_guard(trainer.conditioner)
        self.cond_handle = _forward_hook_counter(
            trainer.conditioner, self.counters, "conditioner_forward_calls")
        self.back_handle = _forward_hook_counter(
            trainer.model, self.counters, "backbone_forward_calls")
        self.training_gen = training_gen
        self.probe_batch = probe_batch
        self.probes: list[dict] = []
        self.history: list[dict] = []
        self.teacher: dict | None = None
        self.teacher_arrays: dict = {}
        self.probe_arrays_by_idx: dict[int, dict] = {}
        self.failure: dict | None = None
        self._phase = "init"
        self._step = 0
        self._clip_pre: dict | None = None
        self.expected_matrix = self._expected_call_matrix(plan)

    # -- configuration binding ------------------------------------------------

    @staticmethod
    def _check_trainer_config(trainer: Trainer, plan: FitPlan) -> None:
        cfg = trainer.cfg
        pairs = (("steps", plan.steps), ("micro_batch", plan.micro_batch),
                 ("grad_accum", plan.grad_accum), ("seed", plan.seed),
                 ("lr", plan.lr), ("weight_decay", plan.weight_decay),
                 ("grad_clip", plan.grad_clip), ("model_type", plan.model_type),
                 ("d_model", plan.d_model), ("n_layers", plan.n_layers),
                 ("n_heads", plan.n_heads), ("ffn_mult", plan.ffn_mult),
                 ("time_window", plan.time_window))
        problems = [f"config_{k}:{getattr(cfg, k)}:want{v}"
                    for k, v in pairs if getattr(cfg, k) != v]
        if cfg.val_seed != plan.val_seed:
            problems.append(f"config_val_seed:{cfg.val_seed}:want{plan.val_seed}")
        if problems:
            raise FitDiagnosticError(f"trainer_plan_binding:{problems}")
        want_windows = 1 if plan.stage == "D1" else 6
        ds_len = len(trainer.train_ds)
        if ds_len != want_windows:
            raise FitDiagnosticError(f"dataset_window_count:{ds_len}:want{want_windows}")
        if ds_len // cfg.micro_batch < 1:
            raise FitDiagnosticError("empty_loader_drop_last")

    def _expected_call_matrix(self, plan: FitPlan) -> dict:
        micros = plan.steps * plan.grad_accum
        n_w = len(self.registration.windows)
        return {"conditioner_forward_calls": micros + plan.n_probes,
                "backbone_forward_calls": micros + plan.n_probes,
                "loss_calls": micros,
                "adamw_step_calls": plan.steps,
                "backward_calls": micros,
                "clip_calls": plan.steps,
                "probe_calls": plan.n_probes,
                "checkpoint_saves": 2,
                "production_decode_calls": n_w * (plan.n_probes + 1),
                "production_fk_batch_calls": 2 * micros + n_w * (plan.n_probes + 1),
                "metric_decode_calls": n_w * (plan.n_probes + 1),
                "window_consumptions_total": micros * plan.micro_batch,
                "epochs_restarted": None}

    def call_matrix_problems(self, snapshot: dict | None = None) -> list[str]:
        problems = []
        source = self.counters if snapshot is None else snapshot
        for key, want in self.expected_matrix.items():
            if want is None:
                continue
            got = source.get(key, 0)
            if got != want:
                problems.append(f"call_matrix:{key}:actual{got}:expected{want}")
        return problems

    # -- helpers --------------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.trainer.cfg.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _ensure_teacher(self) -> None:
        if self.teacher is not None:
            return
        self.budget.check("teacher_reference_build")
        x0_tgt = self.probe_batch["x0"].detach().cpu().numpy()
        md, prod, fk = [], [], []
        with torch.no_grad():
            for b in range(x0_tgt.shape[0]):
                self.counters["metric_decode_calls"] = \
                    self.counters.get("metric_decode_calls", 0) + 1
                md.append(self.decoder.decode(np.ascontiguousarray(x0_tgt[b])))
                self.counters["production_decode_calls"] = \
                    self.counters.get("production_decode_calls", 0) + 1
                pos, quat, q = self.loss_fn.decode_clip(
                    self.probe_batch["x0"].detach()[b:b + 1]
                    .to(self.trainer.cfg.device))
                flat = lambda a: a.reshape(-1, a.shape[-1])
                kp = self.loss_fn.fk.site_world_positions(
                    flat(pos), flat(quat), flat(q), list(KPT_TO_SMPLH)
                ).view(T_WINDOW, S_SITES, 3)
                prod.append((pos[0].detach().cpu().numpy(),
                             quat[0].detach().cpu().numpy(),
                             q[0].detach().cpu().numpy()))
                fk.append(kp.detach().cpu().numpy())
        self.teacher = {"md": md, "fk_tgt": fk}
        for b in range(x0_tgt.shape[0]):
            self.teacher_arrays.update({
                f"w{b}__raw_x0_tgt": np.ascontiguousarray(x0_tgt[b]),
                f"w{b}__md_pos": md[b]["pos"], f"w{b}__md_R": md[b]["R"],
                f"w{b}__md_q": md[b]["q"],
                f"w{b}__prod_pos": prod[b][0], f"w{b}__prod_quat": prod[b][1],
                f"w{b}__prod_q": prod[b][2], f"w{b}__fk_tgt": fk[b]})

    def _run_probe(self, step: int, idx: int) -> None:
        self.budget.check(f"probe_{idx}_step_{step}_start")
        self.runtime_guard.check_unit(f"probe_{idx}_step_{step}_start")
        self._ensure_teacher()
        self.counters["probe_calls"] = self.counters.get("probe_calls", 0) + 1
        gen = torch.Generator()
        entry, arrays = run_probe(
            trainer=self.trainer, probe_batch=self.probe_batch, plan=self.plan,
            step=step, probe_idx=idx, probe_gen=gen,
            training_gen=self.training_gen, loss_fn=self.loss_fn,
            budget=self.budget, counters=self.counters, decoder=self.decoder,
            teacher=self.teacher,
            window_ids=self.registration.window_ids)
        self.probes.append(entry)
        self.probe_arrays_by_idx[idx] = arrays

    def _save_checkpoint(self, name: str, step: int) -> None:
        self.budget.check(f"checkpoint_{name}_start")
        self.runtime_guard.check_unit(f"checkpoint_{name}_start")
        self.counters["checkpoint_saves"] = \
            self.counters.get("checkpoint_saves", 0) + 1
        self.trainer.save(name, step, {"step": step}, identity=self.identity)

    # -- main flow ------------------------------------------------------------

    def run(self) -> dict:
        summary: dict = {"schema": SCHEMA, "stage": self.plan.stage,
                         "plan": self.plan.as_dict(),
                         "registration": self.registration.as_dict(),
                         "identity": self.identity,
                         "runtime": self.runtime_guard.facts(),
                         "budget": self.budget.facts()}
        try:
            self._preflight()
            self._train_and_probe()
            self._write_stage_artifacts()
            # the matrix is judged on the units the STAGE itself executed;
            # the local verification's own decode work is counted separately
            matrix_snapshot = dict(self.counters)
            verification_counters: dict = {"production_decode_calls": 0,
                                           "metric_decode_calls": 0}
            verification = verify_stage_local(
                self.stage_dir, self.plan, self.registration, self.decoder,
                production_decoder=self.loss_fn,
                counters=verification_counters)
            self.budget.check("local_verification_complete")
            summary.update(verification)
            summary["verification_counters"] = verification_counters
            summary["failure"] = None
            summary["local_complete_evidence"] = bool(verification["problems"] == [])
            summary["local_mean_fit_passed"] = verification["local_mean_fit_passed"]
            summary["local_conclusion"] = (
                "incomplete" if not summary["local_complete_evidence"]
                else ("fit_insufficient" if summary["local_mean_fit_passed"] is False
                      else "complete_fit"))
            summary["call_matrix"] = {
                "expected": self.expected_matrix, "actual": matrix_snapshot,
                "problems": self.call_matrix_problems(snapshot=matrix_snapshot)}
            if summary["call_matrix"]["problems"]:
                summary["local_complete_evidence"] = False
                summary["local_conclusion"] = "incomplete"
            summary["spike_gate_status"] = SPIKE_GATE_STATUS
            summary["final_verification_status"] = "pending_F"
            summary["fk_replay_scope"] = FK_REPLAY_SCOPE
        except BaseException as exc:  # noqa: BLE001 -- record, then surface
            summary["failure"] = {
                "type": type(exc).__name__, "message": str(exc)[:500],
                "phase": self._phase, "step": self._step,
                "counters": dict(self.counters)}
            summary["local_complete_evidence"] = False
            summary["local_mean_fit_passed"] = None
            summary["local_conclusion"] = "incomplete"
            summary["spike_gate_status"] = SPIKE_GATE_STATUS
            summary["final_verification_status"] = "pending_F"
            summary["call_matrix"] = {"expected": self.expected_matrix,
                                      "actual": dict(self.counters),
                                      "problems": ["stage_failed"]}
        finally:
            self.cond_handle.remove()
            self.back_handle.remove()
            summary["counters"] = dict(self.counters)
            summary["budget"] = self.budget.facts()
        return summary

    def _preflight(self) -> None:
        self._phase = "preflight"
        self.budget.check("preflight")
        self.runtime_guard.check_unit("preflight")
        apply_problems = getattr(self.runtime_guard, "apply_problems", None)
        if apply_problems:
            raise FitDiagnosticError(f"runtime_apply_failed:{apply_problems}")
        problems = parameter_contract_problems(
            {"model": self.trainer.model, "conditioner": self.trainer.conditioner},
            getattr(self.runtime_guard, "expected_device", "cpu"),
            getattr(self.runtime_guard, "expected_dtype", torch.float32))
        if problems:
            raise FitDiagnosticError(f"parameter_contract:{problems}")

    def _train_and_probe(self) -> None:
        plan, trainer = self.plan, self.trainer
        data_iter = iter(trainer.train_loader)
        ever_yielded = False
        step = 0
        self._phase = "probe_step0"
        self._run_probe(0, 0)
        self._save_checkpoint("initial.pt", 0)
        trainer.model.train()
        trainer.conditioner.train()
        while step < plan.steps:
            step += 1
            self._step = step
            self._phase = f"step_{step}"
            self.budget.check(f"step_{step}_start")
            self.runtime_guard.check_unit(f"step_{step}_start")
            trainer.begin_update()
            parts_acc: dict[str, float] = {}
            n_micro, n_samples = 0, 0
            for micro in range(plan.grad_accum):
                self._phase = f"step_{step}_micro_{micro + 1}"
                self.budget.check(f"step_{step}_micro_{micro + 1}_start")
                self.runtime_guard.check_unit(
                    f"step_{step}_micro_{micro + 1}_start")
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if not ever_yielded:
                        raise FitDiagnosticError("empty_loader_drop_last")
                    self.counters["epochs_restarted"] += 1
                    data_iter = iter(trainer.train_loader)
                    batch = next(data_iter)
                ever_yielded = True
                n_samples += int(batch["x0"].shape[0])
                self.counters["window_consumptions_total"] += \
                    int(batch["x0"].shape[0])
                self.budget.check(f"step_{step}_micro_{micro + 1}_loaded")
                batch = self._to_device(batch)

                def pre_backward(_s=step, _m=micro):
                    self.budget.check(f"step_{_s}_micro_{_m + 1}_pre_backward")
                    self.runtime_guard.check_unit(
                        f"step_{_s}_micro_{_m + 1}_pre_backward")

                def post_backward():
                    self.counters["backward_calls"] += 1

                loss, parts = trainer.accumulate_micro(
                    batch, pre_backward=pre_backward, post_backward=post_backward)
                n_micro += 1
                for k, v in parts.items():
                    parts_acc[k] = parts_acc.get(k, 0.0) + float(v) / plan.grad_accum
                self.budget.check(f"step_{step}_micro_{micro + 1}_backward")
                self.runtime_guard.check_unit(
                    f"step_{step}_micro_{micro + 1}_backward")
            self._phase = f"step_{step}_clip"
            self.budget.check(f"step_{step}_clip")
            self.runtime_guard.check_unit(f"step_{step}_clip")
            self._check_clip_coverage_pre()
            self.counters["clip_calls"] += 1
            gnorm = trainer.clip_gradients()
            self._check_clip_coverage_post()
            self.budget.check(f"step_{step}_pre_optimizer_step")
            self.runtime_guard.check_unit(f"step_{step}_pre_optimizer_step")
            self._phase = f"step_{step}_optimizer_update"
            self.counters["adamw_step_calls"] += 1
            trainer.step_optimizer()
            trainer._check_params()
            self.history.append({"step": step, "n_micro": n_micro,
                                 "n_samples": n_samples,
                                 **{k: float(v) for k, v in parts_acc.items()},
                                 "gnorm": float(gnorm)})
            self.budget.check(f"step_{step}_complete")
            if step % plan.probe_every == 0:
                self._phase = f"probe_step{step}"
                self._run_probe(step, step // plan.probe_every)
        self._phase = "save_last"
        self._save_checkpoint("last.pt", plan.steps)

    def _check_clip_coverage_pre(self) -> None:
        """Every parameter must enter the joint clip (§6.2)."""
        self._clip_pre = {
            name: (p.grad.detach().abs().clone() if p.grad is not None else None)
            for name, p in self.trainer._named_parameters()}

    def _check_clip_coverage_post(self) -> None:
        pre = self._clip_pre
        self._clip_pre = None
        if pre is None:
            raise FitDiagnosticError("clip_coverage_pre_missing")
        named = dict(self.trainer._named_parameters())
        missing = sorted(set(pre) - set(named))
        if missing:
            raise FitDiagnosticError(f"clip_coverage_param_missing:{missing[:3]}")
        posts = {name: (p.grad.detach().abs() if p.grad is not None else None)
                 for name, p in named.items()}
        coef = None
        for name, pre_g in pre.items():
            post_g = posts.get(name)
            if pre_g is not None and post_g is not None and pre_g.numel() \
                    and post_g.numel():
                pmax, qmax = float(pre_g.max()), float(post_g.max())
                if pmax > 0 and qmax > 0:
                    coef = qmax / pmax
                    break
        if coef is not None and coef < 1.0 - 1e-12:
            for name, pre_g in pre.items():
                post_g = posts.get(name)
                if (pre_g is None) != (post_g is None):
                    raise FitDiagnosticError(f"clip_coverage_grad_side:{name}")
                if pre_g is None or pre_g.numel() == 0:
                    continue
                expect = pre_g * coef
                if not torch.allclose(post_g, expect, rtol=1e-4, atol=1e-12):
                    raise FitDiagnosticError(f"clip_coverage_unscaled_param:{name}")

    # -- artifacts ------------------------------------------------------------

    def _write_stage_artifacts(self) -> None:
        self._phase = "write_artifacts"
        self.budget.check("write_artifacts")
        probes_dir = self.stage_dir / "probes"
        probes_dir.mkdir(parents=True, exist_ok=True)
        write_npz_records(self.stage_dir / "teacher_reference.npz",
                          self.teacher_arrays)
        write_npz_records(self.stage_dir / "probe_inputs.npz", {
            f"w{b}__{k}": self.probe_batch[k].detach().cpu().numpy()[b]
            for b in range(self.probe_batch["x0"].shape[0])
            for k in ("x0", *COND_KEYS)})
        probe_index = {"schema": PROBE_INDEX_SCHEMA,
                       "plan": self.plan.as_dict(),
                       "window_ids": list(self.registration.window_ids),
                       "record_table": {k: [list(v[0]), v[1]]
                                        for k, v in RECORD_SHAPES.items()},
                       "probes": self.probes,
                       "teacher_reference": "teacher_reference.npz",
                       "probe_inputs": "probe_inputs.npz",
                       "history_unrounded": self.history}
        (self.stage_dir / "probe_index.json").write_text(
            json.dumps(probe_index, indent=1, allow_nan=False))
        for idx, arrays in self.probe_arrays_by_idx.items():
            step = self.probes[idx]["step"]
            write_npz_records(
                probes_dir / f"probe_{idx:04d}_step_{step:06d}.npz", arrays)
        index = build_artifacts_index(self.stage_dir)
        (self.stage_dir / "artifacts_index.json").write_text(
            json.dumps(index, indent=1, allow_nan=False))


RETURN_OBSERVATION_SCHEMA = "n0_return_observation_v1"


def _read_json_object(path: Path) -> tuple[dict | None, list[str]]:
    """Parse a JSON artifact that MUST be an object; parseable but
    non-object payloads are structural verification failures, never
    crashes (P2: a list-shaped return proof must fail closed)."""
    try:
        doc = json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001
        return None, [f"{Path(path).name}_unreadable:{type(exc).__name__}"]
    if not isinstance(doc, dict):
        return None, [f"{Path(path).name}_not_a_mapping"]
    return doc, []


def _return_proof_problems(stage_dir: Path) -> list[str]:
    """Full consumption of the outer return observation: existence, object
    shape and schema, covered, byte-bound to the sealed report, the proof's
    registered cost and wall cap BOUND to the report's sealed values, and an
    internally consistent coverage chain (elapsed <= registered <= cap)."""
    stage_dir = Path(stage_dir)
    path = stage_dir / "return_observation.json"
    if not path.is_file():
        return ["return_observation_missing"]
    proof, problems = _read_json_object(path)
    if proof is None:
        return problems
    if proof.get("schema") != RETURN_OBSERVATION_SCHEMA:
        problems.append(f"return_observation_schema:{proof.get('schema')!r}")
    if proof.get("covered") is not True:
        problems.append("return_observation_not_covered")
    report_path = stage_dir / "stage_report.json"
    report, rep_problems = _read_json_object(report_path)
    problems += [f"report_{p}" for p in rep_problems]
    if report is None:
        problems.append("return_observation_report_unreadable")
        return problems
    if proof.get("summary_sha256") != sha256_file(report_path):
        problems.append("return_observation_summary_sha_mismatch")
    seal = report.get("seal") or {}
    registered = seal.get("measured_wall_s_at_summary")
    cap = seal.get("wall_cap_s")
    if not isinstance(registered, (int, float)) or isinstance(registered, bool) \
            or not math.isfinite(float(registered)) or float(registered) < 0:
        problems.append("report_registered_cost_invalid")
        registered = None
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) \
            or not math.isfinite(float(cap)) or float(cap) < 0:
        problems.append("report_wall_cap_invalid")
        cap = None
    proof_reg = proof.get("registered_measured_s")
    proof_cap = proof.get("wall_cap_s")
    # the proof's budget numbers are claims; the SEALED REPORT is the
    # authority — a self-consistent proof with invented cost/cap values is
    # refused even when its internal chain looks covered
    if registered is not None and proof_reg != registered:
        problems.append(f"return_observation_registered_mismatch:"
                        f"{proof_reg!r}:report{registered!r}")
    if cap is not None and proof_cap != cap:
        problems.append(f"return_observation_wall_cap_mismatch:"
                        f"{proof_cap!r}:report{cap!r}")
    elapsed = proof.get("elapsed_s")
    for name, v in (("elapsed_s", elapsed), ("registered_measured_s", proof_reg),
                    ("wall_cap_s", proof_cap)):
        if not isinstance(v, (int, float)) or isinstance(v, bool) \
                or not math.isfinite(float(v)) or float(v) < 0:
            problems.append(f"return_observation_field_invalid:{name}")
    if not problems:
        if not (float(elapsed) <= float(proof_reg) <= float(proof_cap)):
            problems.append("return_observation_coverage_invalid")
    return problems


def write_handoff_candidate(stage_dir: Path, summary: dict) -> dict | None:
    """Write the D1 receipt candidate AFTER the sealed report exists: it
    binds the report bytes but is NOT consumable alone (plan §7.4 item 3)."""
    stage_dir = Path(stage_dir)
    if summary.get("stage") != "D1":
        return None
    report_path = stage_dir / "stage_report.json"
    candidate = {
        "schema": HANDOFF_SCHEMA, "stage": "D1",
        "stage_report_sha256": sha256_file(report_path)
        if report_path.is_file() else None,
        "artifacts_index_sha256": summary.get("artifacts_index_sha256"),
        "local_complete_evidence": summary.get("local_complete_evidence"),
        "local_mean_fit_passed": summary.get("local_mean_fit_passed"),
        "coverage": summary.get("coverage"),
        "final_windows": [
            {"window_id": w.get("window_id"),
             "window_mean_fit_passed":
                 (w.get("metrics") or {}).get("window_mean_fit_passed")}
            for w in summary.get("final_windows") or []],
        "note": "candidate_only; handoff_ready is resolved from disk facts "
                "(seal + return observation + re-verification + authorization)",
    }
    (stage_dir / "handoff_candidate.json").write_text(
        json.dumps(candidate, indent=1, allow_nan=False))
    return candidate


def finish_stage(stage_dir: Path, summary: dict, budget: StageBudget) -> dict:
    """Seal the stage report (publish-verify seal), write the D1 handoff
    candidate, re-read the sealed report (the necessary post-seal read), and
    only then run the final return-coverage verification — every ordinary
    I/O lands BEFORE the coverage sample, so no per-stage tail write or
    tail read escapes the checked window (P1: a slow candidate write or a
    slow final read can no longer hide behind a covered proof; P1 round-4:
    the report re-read itself moved inside the window).  When the coverage
    fails, ``verify_return_coverage`` withdraws the on-disk report
    fail-closed; the returned view mirrors that deterministic rewrite
    (costs only increase) instead of spending post-sample I/O on a
    re-read."""
    stage_dir = Path(stage_dir)
    index_path = stage_dir / "artifacts_index.json"
    summary = {**summary, "artifacts_index_sha256":
               sha256_file(index_path) if index_path.is_file() else None}
    path = stage_dir / "stage_report.json"
    seal = _write_summary_sealed(path, summary, budget,
                                 withdraw=dict(I_WITHDRAW_ON_EXCEED),
                                 clock=budget.clock)
    candidate = write_handoff_candidate(stage_dir, summary)
    doc = json.loads(path.read_text())
    cov = verify_return_coverage(path, budget,
                                 withdraw=dict(I_WITHDRAW_ON_EXCEED),
                                 clock=budget.clock)
    if not cov["covered"]:
        doc = {**doc, **I_WITHDRAW_ON_EXCEED,
               "seal": {**(doc.get("seal") or {}),
                        "status": "budget_exceeded_after_seal",
                        "return_coverage_withdrawn": True}}
    return {"stage_report": str(path), "seal": seal, "return_coverage": cov,
            "handoff_candidate": candidate, "summary_doc": doc}


# --------------------------------------------------------------------------
# local verification (§7.4 items 1-3)
# --------------------------------------------------------------------------

def _metrics_equal(saved: dict, recomputed: dict, path: str = "") -> list[str]:
    problems: list[str] = []
    for key, value in recomputed.items():
        if key not in saved:
            problems.append(f"metric_missing:{path}{key}")
            continue
        sv = saved[key]
        if isinstance(value, dict):
            problems += _metrics_equal(sv if isinstance(sv, dict) else {},
                                       value, f"{path}{key}.")
        elif isinstance(value, bool):
            if not isinstance(sv, bool) or sv != value:
                problems.append(f"metric_bool_mismatch:{path}{key}")
        elif isinstance(value, float):
            if isinstance(sv, bool) or not isinstance(sv, (int, float)) \
                    or float(sv) != value:
                problems.append(f"metric_value_mismatch:{path}{key}")
        else:
            if sv != value:
                problems.append(f"metric_field_mismatch:{path}{key}")
    extra = sorted(set(saved) - set(recomputed))
    if extra:
        problems.append(f"metric_extra_fields:{path}{extra[:3]}")
    return problems


def _required_stage_artifacts(plan: FitPlan) -> list[str]:
    """The frozen per-stage artifact set; an index missing any entry is a
    structural failure (P1: an empty index must never verify)."""
    required = ["teacher_reference.npz", "probe_inputs.npz",
                "probe_index.json", "initial.pt", "last.pt"]
    required += [f"probes/probe_{i:04d}_step_{s:06d}.npz"
                 for i, s in enumerate(plan.probe_steps)]
    return required


def _production_decode_binding_problems(saved: dict, raw: np.ndarray,
                                        production_decoder, counters: dict | None,
                                        tag: str) -> list[str]:
    """Re-derive the production FP32 decode from the saved raw x0 through the
    SAME production ``decode_clip`` and bitwise-bind the saved quantities.

    The production decode is not model/FK/optimizer work, so this stays
    inside the §7.4 item-2 recompute; FK points remain production-registered
    observations (state binding stays ``not_evaluated``)."""
    if production_decoder is None:
        return []
    with torch.no_grad():
        x = torch.as_tensor(np.ascontiguousarray(raw),
                            device=production_decoder.mean.device)
        pos, quat, q = production_decoder.decode_clip(x)
    if counters is not None:
        counters["production_decode_calls"] = \
            counters.get("production_decode_calls", 0) + 1
    problems: list[str] = []
    for key, arr in (("prod_pos", pos), ("prod_quat", quat), ("prod_q", q)):
        saved_arr = saved.get(key)
        fresh = arr.detach().cpu().numpy()
        if saved_arr is None or saved_arr.dtype != fresh.dtype \
                or saved_arr.shape != fresh.shape \
                or not np.array_equal(saved_arr.view(np.uint32),
                                      fresh.view(np.uint32)):
            problems.append(f"production_decode_binding_mismatch:{tag}:{key}")
    return problems


def verify_stage_local(stage_dir: Path, plan: FitPlan,
                       registration: FitRegistration,
                       decoder: MetricDecoder,
                       production_decoder=None,
                       counters: dict | None = None) -> dict:
    """§7.4 items 1-3 for one stage: coverage multiset, full CPU recompute of
    every saved probe AND the frozen teacher references from disk raw x0
    (bitwise decode bindings on both sides), production-decode re-derivation
    bindings when the production chain is available, and checkpoint
    role/step/dtype binding."""
    stage_dir = Path(stage_dir)
    problems: list[str] = []
    incomplete = {"problems": problems, "coverage": None,
                  "local_mean_fit_passed": None, "final_windows": [],
                  "decode_binding_status": "not_evaluated",
                  "teacher_decode_binding_status": "not_evaluated",
                  "production_decode_binding_status": "not_evaluated",
                  "fk_metric_reduction_status": "not_evaluated",
                  "intermediate_fk_state_binding_status": "not_evaluated",
                  "teacher_fk_state_binding_status": "not_evaluated",
                  "spike_gate_status": SPIKE_GATE_STATUS}
    index_path = stage_dir / "artifacts_index.json"
    if not index_path.is_file():
        problems.append("artifacts_index_missing")
        return dict(incomplete)
    index = json.loads(index_path.read_text())
    problems += verify_artifacts_index(stage_dir, index,
                                       required=_required_stage_artifacts(plan))
    # the stage report registers the index SHA; consume it whenever the
    # report already exists (F / handoff contexts)
    report_path = stage_dir / "stage_report.json"
    if report_path.is_file():
        report, report_problems = _read_json_object(report_path)
        if report is None:
            problems.append(f"stage_report_invalid:{report_problems[:2]}")
        elif report.get("artifacts_index_sha256") != sha256_file(index_path):
            problems.append("artifacts_index_sha_mismatch")
    probe_index_path = stage_dir / "probe_index.json"
    if not probe_index_path.is_file():
        problems.append("probe_index_missing")
        return dict(incomplete)
    pidx = json.loads(probe_index_path.read_text())
    if pidx.get("schema") != PROBE_INDEX_SCHEMA:
        problems.append("probe_index_schema")

    # 1. coverage multiset ----------------------------------------------------
    want: dict = {}
    for step in plan.probe_steps:
        for wid in registration.window_ids:
            want[(step, wid)] = want.get((step, wid), 0) + 1
    got: dict = {}
    window_ids = list(registration.window_ids)
    for entry in pidx.get("probes") or []:
        step = entry.get("step")
        for w in entry.get("windows") or []:
            key = (step, w.get("window_id"))
            got[key] = got.get(key, 0) + 1
    missing = sorted(f"{k[0]}:{k[1]}" for k in want if want[k] > got.get(k, 0))
    extra = sorted(f"{k[0]}:{k[1]}" for k in got if k not in want)
    duplicated = sorted(f"{k[0]}:{k[1]}" for k in want
                        if k in got and got[k] > want[k])
    for tag, items in (("coverage_missing", missing),
                       ("coverage_extra", extra),
                       ("coverage_duplicate", duplicated)):
        if items:
            problems.append(f"{tag}:{items[:4]}")

    # 2. teacher reference + probe inputs --------------------------------------
    # the teacher metric decode is re-derived from the SAVED RAW teacher x0
    # and bitwise-bound against the saved md_* arrays; a teacher-derived
    # replacement (e.g. the last prediction) can never pass silently
    teacher_records: dict[int, dict] = {}
    teacher_ok: dict[int, bool] = {}
    teacher_path = stage_dir / "teacher_reference.npz"
    if teacher_path.is_file():
        for b in range(len(window_ids)):
            try:
                teacher_records[b] = read_npz_records(
                    teacher_path, TEACHER_TABLE,
                    {r: f"w{b}__{r}" for r in TEACHER_RECORDS})
            except FitDiagnosticError as exc:
                problems.append(f"teacher_reference_invalid:{exc}")
                break
    else:
        problems.append("teacher_reference_missing")
    re_teacher: dict[int, dict] = {}
    for b, recs in teacher_records.items():
        raw_tgt = recs["raw_x0_tgt"]
        problems += [f"teacher_raw_invalid:w{b}:{p}"
                     for p in decoder.verify_x0(raw_tgt)]
        problems += verify_rotation_matrix(recs["md_R"])
        try:
            re_tmd = decoder.decode(raw_tgt)
        except FitDiagnosticError:
            teacher_ok[b] = False
            continue
        if counters is not None:
            counters["metric_decode_calls"] = \
                counters.get("metric_decode_calls", 0) + 1
        re_teacher[b] = re_tmd
        bound = all(np.array_equal(recs[key].view(np.uint64), arr.view(np.uint64))
                    for key, arr in (("md_pos", re_tmd["pos"]),
                                     ("md_R", re_tmd["R"]),
                                     ("md_q", re_tmd["q"])))
        teacher_ok[b] = bound
        if not bound:
            problems.append(f"teacher_decode_binding_mismatch:w{b}")
        problems += _production_decode_binding_problems(
            recs, raw_tgt, production_decoder, counters, f"teacher:w{b}")
    inputs_path = stage_dir / "probe_inputs.npz"
    if inputs_path.is_file():
        for b in range(len(window_ids)):
            try:
                recs = read_npz_records(
                    inputs_path,
                    {"raw_x0_pred": RECORD_SHAPES["raw_x0_pred"]},
                    {"raw_x0_pred": f"w{b}__x0"})
                if b in teacher_records and not np.array_equal(
                        recs["raw_x0_pred"].view(np.uint32),
                        teacher_records[b]["raw_x0_tgt"].view(np.uint32)):
                    problems.append(f"probe_inputs_teacher_x0_mismatch:w{b}")
            except FitDiagnosticError as exc:
                problems.append(f"probe_inputs_invalid:{exc}")
                break
    else:
        problems.append("probe_inputs_missing")

    # 3. recompute every probe -------------------------------------------------
    final_windows: list[dict] = []
    final_step = plan.steps
    any_redecode = False
    any_prod_binding = production_decoder is not None
    for entry in pidx.get("probes") or []:
        step, idx = entry.get("step"), entry.get("idx")
        npz_path = stage_dir / "probes" / f"probe_{idx:04d}_step_{step:06d}.npz"
        if not npz_path.is_file():
            problems.append(f"probe_npz_missing:idx{idx}")
            continue
        fc = entry.get("forward_counts") or {}
        if fc.get("conditioner") != 1 or fc.get("backbone") != 1:
            problems.append(f"probe_forward_counts:idx{idx}:{fc}")
        for b, wout in enumerate(entry.get("windows") or []):
            if wout.get("window_id") != window_ids[b]:
                problems.append(f"probe_window_order:idx{idx}:w{b}")
            try:
                recs = read_npz_records(
                    npz_path, RECORD_SHAPES,
                    {r: f"w{b}__{r}" for r in RECORD_SHAPES})
            except FitDiagnosticError as exc:
                problems.append(f"probe_records_invalid:idx{idx}:{exc}")
                continue
            raw = recs["raw_x0_pred"]
            raw_problems = decoder.verify_x0(raw)
            problems += [f"probe_raw_invalid:idx{idx}:{p}" for p in raw_problems]
            problems += verify_rotation_matrix(recs["md_R"])
            problems += _production_decode_binding_problems(
                recs, raw, production_decoder, counters, f"probe{idx}:w{b}")
            if not teacher_ok.get(b, False):
                # the teacher binding failed for this window: its saved
                # derived quantities are untrusted, so nothing may be
                # consumed from them (no metrics, no final window)
                continue
            if raw_problems:
                continue
            tgt_md = re_teacher.get(b)
            if tgt_md is None:
                continue
            # bitwise decode binding: re-decode from disk raw x0
            re_md = decoder.decode(raw)
            any_redecode = True
            if counters is not None:
                counters["metric_decode_calls"] = \
                    counters.get("metric_decode_calls", 0) + 1
            for key, arr in (("md_pos", re_md["pos"]), ("md_R", re_md["R"]),
                             ("md_q", re_md["q"])):
                saved = recs[key]
                if not np.array_equal(saved.view(np.uint64), arr.view(np.uint64)):
                    problems.append(f"decode_binding_mismatch:idx{idx}:w{b}:{key}")
            # error curves recompute from the re-decoded values on BOTH sides
            re_err = {
                "err_root_pos_t": np.linalg.norm(
                    re_md["pos"] - tgt_md["pos"], axis=1),
                "err_root_rot_t": so3_geodesic_angles_rad(
                    re_md["R"], tgt_md["R"]),
                "err_joint_tj": np.abs(re_md["q"] - tgt_md["q"]),
                "err_fk_ts": np.linalg.norm(
                    recs["fk_pred"].astype(np.float64)
                    - teacher_records[b]["fk_tgt"].astype(np.float64), axis=-1)}
            for key, arr in re_err.items():
                if not np.array_equal(recs[key], arr):
                    problems.append(f"error_curve_mismatch:idx{idx}:w{b}:{key}")
            # metrics recompute (float64, from re-decoded + saved FK points)
            re_metrics = compute_window_metrics(
                re_md, tgt_md, recs["fk_pred"], teacher_records[b]["fk_tgt"])
            problems += [f"probe_metrics_mismatch:idx{idx}:w{b}:{p}"
                         for p in _metrics_equal(wout.get("metrics") or {},
                                                 re_metrics)]
            if wout.get("spike_gate_status") != SPIKE_GATE_STATUS:
                problems.append(f"spike_gate_status_invalid:idx{idx}:w{b}")
            if step == final_step:
                final_windows.append({"window_id": wout.get("window_id"),
                                      "step": step, "metrics": re_metrics})
    total_rows = sum(got.values())
    coverage = {"expected_rows": sum(want.values()), "actual_rows": total_rows,
                "missing": missing, "extra": extra, "duplicated": duplicated}

    # 4. checkpoint role/step binding -------------------------------------------
    problems += _checkpoint_binding_problems(stage_dir, plan, decoder)

    complete = problems == [] and len(final_windows) == len(window_ids)
    mean_fit = (bool(all(w["metrics"]["window_mean_fit_passed"]
                         for w in final_windows))
                if complete and final_windows else None)
    macro = aggregate_local_verdicts(final_windows) if final_windows else None
    return {"problems": problems, "coverage": coverage,
            "local_mean_fit_passed": mean_fit, "final_windows": final_windows,
            "macro": macro,
            "decode_binding_status": ("recomputed_bitwise" if any_redecode
                                      else "not_evaluated"),
            "teacher_decode_binding_status": ("recomputed_bitwise"
                                              if re_teacher else "not_evaluated"),
            "production_decode_binding_status":
                ("recomputed_bitwise" if any_prod_binding else "not_evaluated"),
            "fk_metric_reduction_status": ("recomputed_from_saved_fk_points"
                                           if final_windows else "not_evaluated"),
            "intermediate_fk_state_binding_status": "not_evaluated",
            "teacher_fk_state_binding_status": "not_evaluated",
            "spike_gate_status": SPIKE_GATE_STATUS}


def _checkpoint_binding_problems(stage_dir: Path, plan: FitPlan,
                                 decoder: MetricDecoder) -> list[str]:
    problems: list[str] = []
    for name, want_step in (("initial.pt", 0), ("last.pt", plan.steps)):
        path = Path(stage_dir) / name
        if not path.is_file():
            problems.append(f"checkpoint_missing:{name}")
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"checkpoint_unreadable:{name}:{type(exc).__name__}")
            continue
        if not isinstance(ck, dict) or ck.get("step") != want_step:
            problems.append(f"checkpoint_step_mismatch:{name}:"
                            f"{ck.get('step') if isinstance(ck, dict) else '?'}")
        rs = ck.get("root_stats") if isinstance(ck, dict) else None
        if not isinstance(rs, dict) or "mean" not in rs or "std" not in rs:
            problems.append(f"checkpoint_root_stats_missing:{name}")
        else:
            src = decoder.source_constants_f32
            for key in ("mean", "std"):
                if not np.array_equal(
                        np.ascontiguousarray(rs[key], dtype=np.float32).view(np.uint32),
                        src[key].view(np.uint32)):
                    problems.append(f"checkpoint_root_stats_mismatch:{name}:{key}")
        cfg = ck.get("config") if isinstance(ck, dict) else None
        if not isinstance(cfg, dict) or cfg.get("model_type") != plan.model_type \
                or cfg.get("steps") != plan.steps \
                or cfg.get("micro_batch") != plan.micro_batch:
            problems.append(f"checkpoint_config_mismatch:{name}")
        if isinstance(ck, dict) and ck.get("parameters_finite") is not True:
            problems.append(f"checkpoint_params_nonfinite:{name}")
        # the frozen contract is CUDA/float32 parameters: any saved floating
        # tensor that is not float32 must be refused BEFORE a strict load can
        # silently down-cast it (P1: dtype tamper)
        if isinstance(ck, dict):
            for side in ("model", "conditioner"):
                tensors = ck.get(side)
                if not isinstance(tensors, dict):
                    problems.append(f"checkpoint_{side}_not_a_mapping:{name}")
                    continue
                for key, tensor in tensors.items():
                    if torch.is_tensor(tensor) and tensor.is_floating_point() \
                            and tensor.dtype != torch.float32:
                        problems.append(
                            f"checkpoint_param_dtype:{name}:{side}.{key}:"
                            f"{tensor.dtype}")
    return problems


def local_report_problems(doc: dict) -> list[str]:
    """§7.6 validator: a stage report may never publish final-namespace
    fields; ``final_verification_status`` stays ``pending_F``."""
    problems: list[str] = []
    if doc.get("schema") not in (STAGE_REPORT_SCHEMA, SCHEMA):
        problems.append(f"report_schema:{doc.get('schema')}")
    for key in ("complete_evidence", "mean_fit_passed", "replay_passed"):
        if key in doc:
            problems.append(f"final_namespace_field_in_local_report:{key}")
    if doc.get("final_verification_status") != "pending_F":
        problems.append(
            f"final_verification_status:{doc.get('final_verification_status')}")
    if doc.get("spike_gate_status") != SPIKE_GATE_STATUS:
        problems.append(f"spike_gate_status:{doc.get('spike_gate_status')}")
    if doc.get("local_complete_evidence") is False and \
            doc.get("local_mean_fit_passed") is not None:
        problems.append("local_mean_fit_passed_without_complete_evidence")
    return problems


# --------------------------------------------------------------------------
# D1 -> D6 handoff (§7.4 item 3)
# --------------------------------------------------------------------------

def build_handoff_candidate(stage_dir: Path, plan: FitPlan,
                            verification: dict) -> dict:
    """Candidate receipt binding the local facts; NOT consumable alone —
    the seal/return facts and the authorization are resolved on disk later."""
    return {"schema": HANDOFF_SCHEMA, "stage": plan.stage,
            "stage_dir": str(stage_dir),
            "stage_report_sha256":
                (sha256_file(Path(stage_dir) / "stage_report.json")
                 if (Path(stage_dir) / "stage_report.json").is_file() else None),
            "coverage": verification.get("coverage"),
            "final_windows": [
                {"window_id": w["window_id"],
                 "window_mean_fit_passed": w["metrics"]["window_mean_fit_passed"]}
                for w in verification.get("final_windows", [])],
            "local_mean_fit_passed": verification.get("local_mean_fit_passed"),
            "note": "candidate_only; handoff_ready is resolved from disk facts"}


def resolve_handoff(stage_dir: Path, plan: FitPlan, decoder: MetricDecoder,
                    registration: FitRegistration, authorization: dict | None,
                    budget: StageBudget,
                    production_decoder=None) -> dict:
    """D6-entry re-verification of the D1 receipt INSIDE D6's billed window.

    ``handoff_ready`` requires ALL of: the sealed stage report is consumable,
    the bound outer return observation exists and is internally consistent
    (covered, byte-bound to the report, elapsed <= registered <= cap), the
    handoff candidate exists and binds the current report bytes with
    complete/passing local facts, the full local recompute (from disk raw x0,
    teacher side included) passes, the mean-fit gate passed locally, and the
    authorization explicitly includes D6."""
    stage_dir = Path(stage_dir)
    problems: list[str] = []
    budget.check("handoff_reverify")
    report_path = stage_dir / "stage_report.json"
    if not report_path.is_file():
        return {"handoff_ready": False, "problems": ["stage_report_missing"]}
    doc, report_problems = _read_json_object(report_path)
    if doc is None:
        return {"handoff_ready": False,
                "problems": ["stage_report_invalid"] + report_problems[:3]}
    seal = doc.get("seal") or {}
    if seal.get("status") != "sealed_within_budget" or not seal.get("terminal"):
        problems.append("stage_seal_not_consumable")
    problems += _return_proof_problems(stage_dir)
    candidate_path = stage_dir / "handoff_candidate.json"
    if not candidate_path.is_file():
        problems.append("handoff_candidate_missing")
    else:
        candidate = json.loads(candidate_path.read_text())
        if candidate.get("stage_report_sha256") != sha256_file(report_path):
            problems.append("handoff_candidate_report_sha_mismatch")
        if candidate.get("local_complete_evidence") is not True:
            problems.append("handoff_candidate_not_complete")
        if candidate.get("local_mean_fit_passed") is not True:
            problems.append("handoff_candidate_mean_fit_not_passed")
    verification = verify_stage_local(stage_dir, plan, registration, decoder,
                                      production_decoder=production_decoder)
    problems += verification["problems"]
    mean_fit = verification["local_mean_fit_passed"]
    if mean_fit is not True:
        problems.append("d1_mean_fit_gate_not_passed")
    includes = (authorization or {}).get("includes") or []
    if "D6" not in includes:
        problems.append("authorization_does_not_include_d6")
    if plan.stage != "D1":
        problems.append("handoff_source_not_d1")
    return {"handoff_ready": problems == [] and mean_fit is True,
            "problems": problems, "local_mean_fit_passed": mean_fit,
            "coverage": verification.get("coverage")}


# --------------------------------------------------------------------------
# F: fresh bitwise replay + final verdicts (§7.4 items 4-5, §7.6)
# --------------------------------------------------------------------------

def _fresh_probe(model, conditioner, batch: dict, plan: FitPlan,
                 loss_fn, counters: dict) -> tuple[np.ndarray, list[dict]]:
    """Fresh probe through the SAME production interfaces: exactly one
    conditioner + one backbone forward, then the fresh production FP32
    decode + FK points per window."""
    device = batch["x0"].device
    with torch.no_grad():
        conditioner.eval()
        model.eval()
        memory = conditioner(batch)
        if plan.model_type == "diffusion":
            gen = torch.Generator()
            gen.manual_seed(probe_generator_seed(plan))
            t = torch.randint(0, PLAN_NUM_TRAIN_TIMESTEPS,
                              (batch["x0"].shape[0],), generator=gen).to(device)
            noise = torch.randn(batch["x0"].shape, generator=gen).to(device)
            x_t = model.add_noise(batch["x0"], noise, t)
            x0_pred = model(x_t, t, memory)
        else:
            x0_pred = model(memory, batch["x0"].shape[1])
    counters["conditioner_forward_calls"] = \
        counters.get("conditioner_forward_calls", 0) + 1
    counters["backbone_forward_calls"] = \
        counters.get("backbone_forward_calls", 0) + 1
    x0_np = x0_pred.detach().to("cpu").numpy()
    per_window = []
    flat = lambda a: a.reshape(-1, a.shape[-1])
    for b in range(x0_np.shape[0]):
        counters["production_decode_calls"] = \
            counters.get("production_decode_calls", 0) + 1
        pos, quat, q = loss_fn.decode_clip(x0_pred.detach()[b:b + 1])
        counters["production_fk_batch_calls"] = \
            counters.get("production_fk_batch_calls", 0) + 1
        kp = loss_fn.fk.site_world_positions(
            flat(pos), flat(quat), flat(q), list(KPT_TO_SMPLH)
        ).view(T_WINDOW, S_SITES, 3)
        per_window.append({
            "prod_pos": pos[0].detach().cpu().numpy(),
            "prod_quat": quat[0].detach().cpu().numpy(),
            "prod_q": q[0].detach().cpu().numpy(),
            "fk_pred": kp.detach().cpu().numpy()})
    return x0_np, per_window


def _checkpoint_tensor_problems(ck: dict, model, conditioner) -> list[str]:
    """P1: full key-set/shape/dtype/finiteness contract against the FRESH
    modules BEFORE any ``load_state_dict`` copy — strict load would silently
    down-cast a float64 tensor and the post-load check could never see it."""
    problems: list[str] = []
    for side, module in (("model", model), ("conditioner", conditioner)):
        want = module.state_dict()
        got = ck.get(side)
        if not isinstance(got, dict):
            problems.append(f"checkpoint_{side}_not_a_mapping")
            continue
        if set(got) != set(want):
            problems.append(f"checkpoint_{side}_key_set_mismatch")
            continue
        for key, want_t in want.items():
            got_t = got[key]
            if not torch.is_tensor(got_t):
                problems.append(f"checkpoint_{side}.{key}_not_tensor")
                continue
            if got_t.shape != want_t.shape:
                problems.append(f"checkpoint_{side}.{key}_shape")
            if got_t.dtype != want_t.dtype:
                problems.append(f"checkpoint_{side}.{key}_dtype:{got_t.dtype}")
            if got_t.is_floating_point() and not bool(torch.isfinite(got_t).all()):
                problems.append(f"checkpoint_{side}.{key}_nonfinite")
    return problems


def _load_state_dicts_checked(path: Path, model, conditioner) -> list[str]:
    """Verify the saved tensor contract FIRST; only a clean checkpoint is
    copied into the live modules."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    problems = _checkpoint_tensor_problems(ck, model, conditioner)
    if problems:
        return problems
    model.load_state_dict(ck["model"], strict=True)
    conditioner.load_state_dict(ck["conditioner"], strict=True)
    return []


def _f_identity_problems(stage_dir: Path, plan: FitPlan,
                         decoder: MetricDecoder, index: dict) -> list[str]:
    problems = verify_artifacts_index(
        Path(stage_dir), index, required=_required_stage_artifacts(plan))
    for name, want_step in (("initial.pt", 0), ("last.pt", plan.steps)):
        path = Path(stage_dir) / name
        if not path.is_file():
            problems.append(f"checkpoint_missing:{name}")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or ck.get("step") != want_step:
            problems.append(f"checkpoint_step_mismatch:{name}")
        cfg = ck.get("config") or {}
        if cfg.get("model_type") != plan.model_type:
            problems.append(f"checkpoint_model_type:{name}")
        rs = ck.get("root_stats") or {}
        src = decoder.source_constants_f32
        for key in ("mean", "std"):
            if not np.array_equal(
                    np.ascontiguousarray(rs.get(key), dtype=np.float32).view(np.uint32),
                    src[key].view(np.uint32)):
                problems.append(f"checkpoint_root_stats_mismatch:{name}:{key}")
    return problems


def _saved_probe_entry(stage_dir: Path, idx: int, want_step: int) -> dict | None:
    pidx = json.loads((Path(stage_dir) / "probe_index.json").read_text())
    for entry in pidx.get("probes") or []:
        if entry.get("idx") == idx and entry.get("step") == want_step:
            return entry
    return None


def _saved_probe_records(stage_dir: Path, idx: int, b: int) -> dict:
    pidx = json.loads((Path(stage_dir) / "probe_index.json").read_text())
    entry = next(e for e in pidx["probes"] if e.get("idx") == idx)
    npz_path = (Path(stage_dir) / "probes" /
                f"probe_{idx:04d}_step_{entry['step']:06d}.npz")
    return read_npz_records(
        npz_path, RECORD_SHAPES,
        {r: f"w{b}__{r}" for r in ("raw_x0_pred", "prod_pos", "prod_quat",
                                   "prod_q", "fk_pred", "md_pos", "md_R",
                                   "md_q")})


def _teacher_records(stage_dir: Path, b: int) -> dict:
    return read_npz_records(
        Path(stage_dir) / "teacher_reference.npz", TEACHER_TABLE,
        {r: f"w{b}__{r}" for r in TEACHER_RECORDS})


def _teacher_reference_checked(stage_dir: Path, decoder: MetricDecoder,
                               b: int, counters: dict | None = None) -> tuple[dict, list[str]]:
    """Teacher reference with the metric decode re-derived from the saved RAW
    teacher x0 and bitwise-bound against the saved md_* arrays (P1: a
    teacher-derived replacement can never feed the F metrics)."""
    problems: list[str] = []
    recs = _teacher_records(stage_dir, b)
    problems += [f"teacher_raw_invalid:w{b}:{p}"
                 for p in decoder.verify_x0(recs["raw_x0_tgt"])]
    re_tmd = decoder.decode(recs["raw_x0_tgt"])
    if counters is not None:
        counters["metric_decode_calls"] = \
            counters.get("metric_decode_calls", 0) + 1
    for key, arr in (("md_pos", re_tmd["pos"]), ("md_R", re_tmd["R"]),
                     ("md_q", re_tmd["q"])):
        if not np.array_equal(recs[key].view(np.uint64), arr.view(np.uint64)):
            problems.append(f"teacher_decode_binding_mismatch:w{b}:{key}")
    recs = {**recs, "md_pos": re_tmd["pos"], "md_R": re_tmd["R"],
            "md_q": re_tmd["q"]}
    return recs, problems


def run_f_verification(*, stage_specs: list[dict], out_dir: Path,
                       budget: StageBudget, runtime_guard, build_modules,
                       authorization: dict | None) -> dict:
    """One F window: per stage re-run items 1-3, then fresh bitwise replay of
    the initial/last endpoints (exactly 2 fresh probes per stage).  Any
    mismatch is recorded as ``replay_mismatch`` and fails the final verdict —
    no tolerance loosening, no retry, no stage-dir rewrite."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_guard.check_unit("f_entry")
    stages_out: list[dict] = []
    evidence_all_ok = True
    for spec in stage_specs:
        stage_dir = Path(spec["stage_dir"])
        plan: FitPlan = spec["plan"]
        registration: FitRegistration = spec["registration"]
        decoder: MetricDecoder = spec["decoder"]
        loss_fn = spec["loss_fn"]
        res: dict = {"stage": plan.stage, "stage_dir": str(stage_dir)}
        budget.check(f"f_stage_{plan.stage}_entry")
        report_path = stage_dir / "stage_report.json"
        doc, report_problems = _read_json_object(report_path)
        if doc is None:
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "stage_report_invalid",
                        "report_problems": report_problems[:4]})
            stages_out.append(res)
            evidence_all_ok = False
            continue
        seal = doc.get("seal") or {}
        if seal.get("status") != "sealed_within_budget":
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "stage_seal_not_consumable"})
            stages_out.append(res)
            evidence_all_ok = False
            continue
        if doc.get("local_complete_evidence") is not True:
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "local_complete_evidence_false"})
            stages_out.append(res)
            evidence_all_ok = False
            continue
        proof_problems = _return_proof_problems(stage_dir)
        if proof_problems:
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "return_observation_missing_or_uncovered",
                        "return_proof_problems": proof_problems[:6]})
            stages_out.append(res)
            evidence_all_ok = False
            continue
        f_verify_counters: dict = {"production_decode_calls": 0,
                                   "metric_decode_calls": 0}
        verification = verify_stage_local(
            stage_dir, plan, registration, decoder,
            production_decoder=loss_fn, counters=f_verify_counters)
        res["local_reverify_problems"] = verification["problems"][:10]
        res["verification_counters"] = dict(f_verify_counters)
        res["intermediate_fk_state_binding_status"] = \
            verification["intermediate_fk_state_binding_status"]
        res["teacher_fk_state_binding_status"] = \
            verification["teacher_fk_state_binding_status"]
        res["fk_replay_scope"] = FK_REPLAY_SCOPE
        if verification["problems"]:
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "local_reverify_failed"})
            stages_out.append(res)
            evidence_all_ok = False
            continue
        identity_problems = _f_identity_problems(
            stage_dir, plan, decoder,
            json.loads((stage_dir / "artifacts_index.json").read_text()))
        if identity_problems:
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "f_identity_problems",
                        "identity_problems": identity_problems[:10]})
            stages_out.append(res)
            evidence_all_ok = False
            continue
        res["replay_enabled"] = True
        f_counters: dict = {"conditioner_forward_calls": 0,
                            "backbone_forward_calls": 0,
                            "production_fk_batch_calls": 0,
                            "production_decode_calls": 0,
                            "metric_decode_calls": 0, "probe_calls": 0}
        replay_problems: list[str] = []
        mean_fit_values: list[bool] = []
        window_ids = list(registration.window_ids)
        # construction sits INSIDE the guarded region: a deadline crossed by
        # the pre-verification checks refuses BEFORE any module exists (P1)
        try:
            budget.check(f"f_stage_{plan.stage}_pre_construct")
            model, conditioner = build_modules()
            patch_conditioner_leak_guard(conditioner)
            runtime_guard.check_unit("f_modules_constructed")
            problems = parameter_contract_problems(
                {"model": model, "conditioner": conditioner},
                getattr(runtime_guard, "expected_device", "cpu"),
                getattr(runtime_guard, "expected_dtype", torch.float32))
            if problems:
                res.update({"replay_enabled": False, "replay_passed": None,
                            "skip_reason": "f_parameter_contract",
                            "parameter_problems": problems[:10]})
                stages_out.append(res)
                evidence_all_ok = False
                continue
            for role, want_step in (("initial", 0), ("last", plan.steps)):
                budget.check(f"f_probe_{role}")
                load_problems = _load_state_dicts_checked(
                    stage_dir / f"{role}.pt", model, conditioner)
                if load_problems:
                    # refuse BEFORE any copy: a strict load would silently
                    # down-cast mistyped tensors (P1: dtype tamper)
                    replay_problems.append(
                        f"checkpoint_tensor_contract:{load_problems[:3]}")
                    continue
                budget.check(f"f_probe_{role}_loaded")
                runtime_guard.check_unit(f"f_probe_{role}_loaded")
                param_problems = parameter_contract_problems(
                    {"model": model, "conditioner": conditioner},
                    getattr(runtime_guard, "expected_device", "cpu"),
                    getattr(runtime_guard, "expected_dtype", torch.float32))
                if param_problems:
                    replay_problems.append(f"loaded_param_contract:{param_problems[:3]}")
                    continue
                f_counters["probe_calls"] += 1
                # rebuild the probe batch from the SAVED inputs (disk binding)
                batch_arrays = {}
                for b in range(len(window_ids)):
                    recs = read_npz_records(
                        stage_dir / "probe_inputs.npz", RECORD_SHAPES,
                        {"raw_x0_pred": f"w{b}__x0"})
                    batch_arrays.setdefault("x0", []).append(recs["raw_x0_pred"])
                    for key in COND_KEYS:
                        with np.load(stage_dir / "probe_inputs.npz") as data:
                            batch_arrays.setdefault(key, []).append(
                                np.asarray(data[f"w{b}__{key}"]))
                batch = {k: torch.from_numpy(np.ascontiguousarray(np.stack(v)))
                         for k, v in batch_arrays.items()}
                device = next(model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
                budget.check(f"f_probe_{role}_inputs_ready")
                # a precision drift during the input reads must refuse the
                # fresh numeric call itself (P1: budget-only checks are not
                # enough at this boundary)
                runtime_guard.check_unit(f"f_probe_{role}_inputs_ready")
                x0_fresh, fresh_prod = _fresh_probe(
                    model, conditioner, batch, plan, loss_fn, f_counters)
                budget.check(f"f_probe_{role}_fresh_complete")
                runtime_guard.check_unit(f"f_probe_{role}_fresh_complete")
                idx = 0 if role == "initial" else plan.n_probes - 1
                saved_entry = _saved_probe_entry(stage_dir, idx, want_step)
                if saved_entry is None:
                    replay_problems.append(f"saved_probe_missing:{role}")
                    continue
                for b, wout in enumerate(saved_entry["windows"]):
                    raw_f = np.ascontiguousarray(x0_fresh[b])
                    problems += [f"fresh_x0_invalid:{p}"
                                 for p in decoder.verify_x0(raw_f)]
                    saved = _saved_probe_records(stage_dir, idx, b)
                    if not np.array_equal(raw_f.view(np.uint32),
                                          saved["raw_x0_pred"].view(np.uint32)):
                        replay_problems.append(f"replay_mismatch:x0:{role}:w{b}")
                    for key, fresh_arr in (("prod_pos", fresh_prod[b]["prod_pos"]),
                                           ("prod_quat", fresh_prod[b]["prod_quat"]),
                                           ("prod_q", fresh_prod[b]["prod_q"]),
                                           ("fk_pred", fresh_prod[b]["fk_pred"])):
                        if not np.array_equal(
                                np.ascontiguousarray(fresh_arr).view(np.uint32),
                                saved[key].view(np.uint32)):
                            replay_problems.append(
                                f"replay_mismatch:{key}:{role}:w{b}")
                    # fresh metric_decode + metrics from the fresh x0 against
                    # the RE-DERIVED teacher decode (never the saved md_*)
                    teacher, teacher_problems = _teacher_reference_checked(
                        stage_dir, decoder, b, f_counters)
                    replay_problems += teacher_problems[:3]
                    # P1 (round-4): the fresh metric decode is a numeric unit
                    # of its own — the work deadline is re-checked at this
                    # boundary, so a deadline crossed during the teacher
                    # re-verification refuses the decode (the BudgetExceeded
                    # is caught below and the stage closes as incomplete).
                    budget.check(f"f_fresh_metrics_{role}_w{b}")
                    f_counters["metric_decode_calls"] += 1
                    fresh_md = decoder.decode(raw_f)
                    fresh_metrics = compute_window_metrics(
                        fresh_md, {"pos": teacher["md_pos"], "R": teacher["md_R"],
                                   "q": teacher["md_q"]},
                        saved["fk_pred"], teacher["fk_tgt"])
                    metric_problems = _metrics_equal(
                        wout.get("metrics") or {}, fresh_metrics)
                    if metric_problems:
                        replay_problems.append(
                            f"fresh_metrics_mismatch:{role}:w{b}:"
                            f"{metric_problems[:2]}")
                    if role == "last":
                        mean_fit_values.append(
                            fresh_metrics["window_mean_fit_passed"])
        except FitDiagnosticError as exc:
            replay_problems.append(f"fresh_replay_error:{exc}")
        except TrainingNumericalError as exc:
            replay_problems.append(f"fresh_replay_numerical:{exc}")
        res["replay_problems"] = replay_problems[:20]
        res["fresh_counters"] = dict(f_counters)
        res["replay_passed"] = replay_problems == []
        res["complete_evidence"] = res["replay_passed"] is True
        res["mean_fit_passed"] = (bool(all(mean_fit_values))
                                  if res["replay_passed"] and mean_fit_values
                                  else None)
        if not res["replay_passed"]:
            res["verdict"] = "replay_mismatch"
            evidence_all_ok = False
        stages_out.append(res)
    all_replayed = bool(stages_out) and all(
        s.get("replay_passed") is True for s in stages_out)
    # P1 (round-4): the completion claim must itself be affordable inside
    # the work deadline — a claim sampled past the deadline closes as
    # incomplete (replay facts stay recorded; the verdict cannot pass).
    claim_ok = budget.clock() < budget.work_deadline
    if not claim_ok and not budget.exceeded:
        budget.exceeded = True
        budget.stopped_at = "f_final_claim"
    final = {
        "schema": F_REPORT_SCHEMA,
        "stages": stages_out,
        "replay_passed": (True if all_replayed and evidence_all_ok
                          else (False if not evidence_all_ok else None)),
        "complete_evidence": (True if all_replayed and evidence_all_ok
                              else (False if not evidence_all_ok else None)),
        "mean_fit_passed": None,
        "tolerance_policy": "bitwise_no_tolerance",
        "fk_replay_scope": FK_REPLAY_SCOPE,
        "spike_gate_status": SPIKE_GATE_STATUS,
        "final_verification_status": "final_F",
        "budget": budget.facts(),
        "identity": {"module": "ireset.eval.w1_smallset_fit_diagnostic"},
    }
    if not claim_ok:
        final["claim_deadline_exceeded"] = True
        final["complete_evidence"] = False
        final["mean_fit_passed"] = None
        final["verdict"] = "incomplete"
    elif final["complete_evidence"] is True and final["replay_passed"] is True:
        values = [s.get("mean_fit_passed") for s in stages_out]
        final["mean_fit_passed"] = (bool(all(v is True for v in values))
                                    if all(v is not None for v in values) else None)
        final["verdict"] = ("fit_pass" if final["mean_fit_passed"] is True
                            else "fit_insufficient")
    else:
        final["verdict"] = "incomplete"
    path = out_dir / "f_report.json"
    # P1 (round-4): F covers its OWN final report re-read with a return-
    # coverage verification — the outer acceptance budget alone must not be
    # the only guard against a slow tail read; the necessary read happens
    # BEFORE the coverage sample and no I/O follows the passing sample.
    seal = _write_summary_sealed(path, final, budget,
                                 withdraw=dict(F_WITHDRAW_ON_EXCEED),
                                 clock=budget.clock)
    doc = json.loads(path.read_text())
    cov = verify_return_coverage(path, budget,
                                 withdraw=dict(F_WITHDRAW_ON_EXCEED),
                                 clock=budget.clock)
    if not cov["covered"]:
        doc = {**doc, **F_WITHDRAW_ON_EXCEED,
               "seal": {**(doc.get("seal") or {}),
                        "status": "budget_exceeded_after_seal",
                        "return_coverage_withdrawn": True}}
    return {"f_report": str(path), "seal": seal, "return_coverage": cov,
            "summary_doc": doc}


# --------------------------------------------------------------------------
# synthetic world (offline acceptance fixture)
# --------------------------------------------------------------------------

class SyntheticWindowDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict],
                 window_keys: list[WindowKey] | None = None) -> None:
        self.items = items
        self.window_keys = list(window_keys) if window_keys is not None else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class SyntheticFK:
    """Differentiable deterministic 15-site FK stub (offline tests only).

    site_i = pos + R @ offset_i + c_i * q[:, :3].  Never claimed equivalent
    to the production G1 FK; it lets the probe/verification machinery run
    without locked assets."""

    def __init__(self, n_sites: int = S_SITES, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.offsets = torch.tensor(
            rng.uniform(-0.2, 0.2, (n_sites, 3)), dtype=torch.float32)
        self.coeffs = torch.tensor(
            rng.uniform(-0.02, 0.02, (n_sites, 3)), dtype=torch.float32)

    def site_world_positions(self, pos, quat, q, names):
        n_sites = self.offsets.shape[0]
        if len(list(names)) != n_sites:
            raise FitDiagnosticError("synthetic_fk_site_count_mismatch")
        R = _quat_to_mat_f32(quat)
        return (pos[:, None, :]
                + torch.einsum("nij,sj->nsi", R, self.offsets.to(pos.device))
                + q[:, None, :3] * self.coeffs.to(pos.device)[None])


def _quat_to_mat_f32(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                     1 - 2 * (x * x + y * y)], -1),
    ], -2)


@dataclass
class SyntheticWorld:
    seed: int
    tokenizer: StateTokenizer
    registration: FitRegistration
    train_ds: SyntheticWindowDataset
    probe_batch: dict
    decoder: MetricDecoder
    loss_fn: LossAssembler
    fk_inner: SyntheticFK
    d_model: int
    easy_fit: bool

    def registration_for(self, stage: str) -> FitRegistration:
        if stage not in ("D1", "D6"):
            raise FitDiagnosticError(f"unknown stage {stage}")
        windows = self.registration.windows
        if stage == "D1":
            return FitRegistration("D1", windows=(windows[0],),
                                   splits=(self.registration.splits[0],))
        return FitRegistration("D6", windows=windows,
                               splits=self.registration.splits)

    def view_for(self, stage: str) -> dict:
        """Stage view: the registered window subset as dataset + probe batch
        (D1 takes the canonical-first window, plan §3.2)."""
        reg = self.registration_for(stage)
        items = self.train_ds.items[:len(reg.windows)]
        ds = SyntheticWindowDataset(items, window_keys=list(reg.windows))
        probe_batch = {k: torch.stack([it[k] for it in items])
                       for k in ("x0", *COND_KEYS)}
        return {"ds": ds, "probe_batch": probe_batch, "registration": reg}

    def build_trainer(self, plan: FitPlan, out_dir: Path,
                      budget_deadline: float | None,
                      identity: dict | None = None,
                      train_ds: SyntheticWindowDataset | None = None) -> Trainer:
        cfg = TrainConfig(
            model_type=plan.model_type, steps=plan.steps,
            micro_batch=plan.micro_batch, grad_accum=plan.grad_accum,
            lr=plan.lr, weight_decay=plan.weight_decay,
            grad_clip=plan.grad_clip, seed=plan.seed, num_workers=0,
            val_seed=plan.val_seed, d_model=plan.d_model,
            n_layers=plan.n_layers, n_heads=plan.n_heads,
            ffn_mult=plan.ffn_mult, time_window=plan.time_window,
            device="cpu", loss=LossWeights(w_contact=0.0))
        ds = train_ds if train_ds is not None else self.train_ds
        return Trainer(cfg, self.tokenizer, ds, ds,
                       Path(out_dir), identity=identity,
                       budget_deadline=budget_deadline,
                       loss_fn=self.loss_fn)


def make_synthetic_world(seed: int = 20261107, n_windows: int = 6,
                         d_model: int = 16, easy_fit: bool = True,
                         device: str = "cpu") -> SyntheticWorld:
    """Offline synthetic world: synthetic tokenizer/joint range, teacher
    windows, conditioner channels, synthetic FK, loss assembler with
    w_contact=0 and an injected static frame (no asset compile)."""
    joint_range = np.stack([-2.0 + 0.02 * np.arange(J_JOINTS),
                            2.0 - 0.02 * np.arange(J_JOINTS)], axis=1)
    mean = np.array([0.10, 0.20, 0.90], dtype=np.float32)
    std = np.array([0.50, 0.50, 0.30], dtype=np.float32)
    stats = RootStats(mean=mean, std=std)
    tokenizer = StateTokenizer(joint_range.astype(np.float64), stats)
    decoder = MetricDecoder(mean, std, joint_range[:, 0], joint_range[:, 1])
    windows = [WindowKey(f"synth_{i:02d}", 0, T_WINDOW) for i in range(n_windows)]
    registration = FitRegistration(
        stage="D1", windows=canonical_window_order(windows),
        splits=tuple(["train"] * len(windows)))
    items = [_synthetic_item(seed, wi, tokenizer)
             for wi in range(len(registration.windows))]
    train_ds = SyntheticWindowDataset(items)
    probe_batch = {k: torch.stack([it[k] for it in items])
                   for k in ("x0", *COND_KEYS)}
    fk = SyntheticFK(seed=seed)
    loss_fn = LossAssembler(tokenizer, LossWeights(w_contact=0.0), device,
                            use_fk=True, fk=fk,
                            static_frame=(np.eye(3), np.zeros(3)))
    return SyntheticWorld(seed=seed, tokenizer=tokenizer,
                          registration=registration, train_ds=train_ds,
                          probe_batch=probe_batch, decoder=decoder,
                          loss_fn=loss_fn, fk_inner=fk, d_model=d_model,
                          easy_fit=easy_fit)


def _synthetic_item(seed: int, wi: int, tokenizer: StateTokenizer) -> dict:
    rng = np.random.default_rng(seed + 101 * wi)
    # easy-fit target: the zero-normalized state (root at the stats mean,
    # identity 6D, joints mid-range); decoded pos = mean, R = I, q = mid.
    x0 = np.zeros((T_WINDOW, 38), dtype=np.float32)
    x0[:, 3] = 1.0
    x0[:, 7] = 1.0
    yaw = np.full(T_WINDOW, 0.02 * wi)
    box_t = np.tile(np.array([0.4, 0.0, 0.3 + 0.01 * wi]), (T_WINDOW, 1))
    box_quat = np.stack([np.cos(yaw / 4), np.zeros(T_WINDOW),
                         np.zeros(T_WINDOW), np.sin(yaw / 4)], axis=1)
    kps = rng.uniform(-0.3, 0.3, (1, 52, 3)).astype(np.float32)
    kps = np.repeat(kps, T_WINDOW, axis=0)
    flags = np.zeros((T_WINDOW, 4), dtype=np.float32)
    flags[:, 0] = 1.0
    return {
        "x0": torch.as_tensor(x0, dtype=torch.float32),
        "kps_obj": torch.as_tensor(kps, dtype=torch.float32),
        "kps_clip": torch.as_tensor(kps + 0.01, dtype=torch.float32),
        "box_t_clip": torch.as_tensor(box_t, dtype=torch.float32),
        "box_quat": torch.as_tensor(box_quat, dtype=torch.float32),
        "contact": torch.as_tensor(flags, dtype=torch.float32),
    }


def write_acceptance_outputs(out_dir: Path, facts: dict, budget: StageBudget,
                             withdraw: dict | None = None) -> dict:
    """Write the acceptance summary under the publish-verify seal and return
    the on-disk document (the authority for the CLI's exit decision)."""
    out_dir = Path(out_dir)
    path = out_dir / "acceptance_summary.json"
    seal = _write_summary_sealed(path, facts, budget,
                                 withdraw=dict(withdraw or {}),
                                 clock=budget.clock)
    return {"summary": str(path), "seal": seal,
            "summary_doc": json.loads(path.read_text())}


# --------------------------------------------------------------------------
# synthetic acceptance + counterexample battery
# --------------------------------------------------------------------------

def build_modules_factory(world: SyntheticWorld, plan: FitPlan):
    def factory():
        if plan.model_type == "diffusion":
            from ireset.diffusion.generator import DiffusionG1
            model = DiffusionG1(num_train_timesteps=PLAN_NUM_TRAIN_TIMESTEPS,
                                **_backbone_kwargs(plan))
        else:
            from ireset.model.deterministic import DeterministicG1
            model = DeterministicG1(**_backbone_kwargs(plan))
        from ireset.model.conditioner import ConditionEncoder
        conditioner = ConditionEncoder(plan.d_model)
        return model, conditioner
    return factory


def _backbone_kwargs(plan: FitPlan) -> dict:
    return {"d_model": plan.d_model, "n_layers": plan.n_layers,
            "n_heads": plan.n_heads, "ffn_mult": plan.ffn_mult,
            "time_window": plan.time_window}


def _run_fit_stage(world: SyntheticWorld, plan: FitPlan, stage_dir: Path,
                   budget: StageBudget, authorization: dict | None):
    view = world.view_for(plan.stage)
    trainer = world.build_trainer(plan, stage_dir,
                                  budget_deadline=budget.work_deadline,
                                  train_ds=view["ds"])
    training_gen = torch.Generator().manual_seed(plan.seed)
    driver = FitDriver(
        trainer=trainer, plan=plan,
        registration=view["registration"],
        loss_fn=world.loss_fn, decoder=world.decoder,
        probe_batch=view["probe_batch"], budget=budget,
        runtime_guard=SyntheticRuntimeGuard(),
        identity={"module": "ireset.eval.w1_smallset_fit_diagnostic",
                  "synthetic": True, "stage": plan.stage},
        stage_dir=stage_dir, training_gen=training_gen,
        dataset_window_keys=view["ds"].window_keys,
        authorization_includes_d6="D6" in ((authorization or {}).get("includes")
                                           or []))
    summary = driver.run()
    finished = finish_stage(stage_dir, summary, budget)
    return driver, summary, finished


def _mini_stage(tmp: Path, stage: str, *, n_windows: int = 6, steps: int = 2,
                probe_every: int = 1, seed: int = 20261107, d_model: int = 16,
                lr: float = 0.02, world: SyntheticWorld | None = None,
                finish: bool = False):
    """One miniature fit stage for the counterexamples."""
    world = world or make_synthetic_world(seed=seed, n_windows=n_windows,
                                          d_model=d_model)
    plan = FitPlan(stage, steps=steps, probe_every=probe_every, synthetic=True,
                   d_model=d_model, n_layers=1, n_heads=4, ffn_mult=2, lr=lr)
    sdir = Path(tmp) / stage.lower()
    budget = StageBudget(stage=stage, wall_cap_s=600.0, reserve_s=10.0)
    view = world.view_for(stage)
    trainer = world.build_trainer(plan, sdir, budget_deadline=budget.work_deadline,
                                  train_ds=view["ds"])
    driver = FitDriver(
        trainer=trainer, plan=plan,
        registration=view["registration"], loss_fn=world.loss_fn,
        decoder=world.decoder, probe_batch=view["probe_batch"], budget=budget,
        runtime_guard=SyntheticRuntimeGuard(), identity={"synthetic": True},
        stage_dir=sdir,
        training_gen=torch.Generator().manual_seed(plan.seed),
        dataset_window_keys=view["ds"].window_keys,
        authorization_includes_d6=True)
    summary = driver.run()
    finished = finish_stage(sdir, summary, budget) if finish else None
    return world, plan, driver, summary, sdir, budget, finished


def run_synthetic_acceptance(budget: StageBudget, out_dir: Path,
                             seed: int = 20261107,
                             d1_steps: int = 300, d1_probe_every: int = 30,
                             d6_steps: int = 300, d6_probe_every: int = 30,
                             d1_lr: float = 0.03, d6_lr: float = 0.02) -> dict:
    """Offline end-to-end acceptance: mini D1 -> local verify -> handoff ->
    mini D6 (only when the D1 mean gate passed) -> F fresh bitwise replay ->
    counterexample battery.  Every gate outcome is recorded honestly; the
    acceptance asserts pipeline facts, never a forced fit pass."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    facts: dict = {"schema": SCHEMA, "kind": "synthetic_acceptance", "seed": seed}
    world = make_synthetic_world(seed=seed, n_windows=6, d_model=16)
    facts["frozen_plan_defaults_problems"] = frozen_plan_defaults_problems()
    auth = {"includes": ["D6"]}
    d1_plan = FitPlan("D1", steps=d1_steps, probe_every=d1_probe_every,
                      synthetic=True, d_model=16, n_layers=1, n_heads=4,
                      ffn_mult=2, lr=d1_lr)
    budget.check("acceptance_d1_entry")
    _, d1_summary, d1_fin = _run_fit_stage(world, d1_plan, out_dir / "d1",
                                           budget, auth)
    facts["d1"] = {
        "local_complete_evidence": d1_summary.get("local_complete_evidence"),
        "local_mean_fit_passed": d1_summary.get("local_mean_fit_passed"),
        "local_conclusion": d1_summary.get("local_conclusion"),
        "seal": (d1_fin["summary_doc"].get("seal") or {}).get("status"),
        "return_coverage": d1_fin["return_coverage"].get("covered"),
        "call_matrix_problems": d1_summary["call_matrix"]["problems"],
        "report_problems": local_report_problems(
            {k: v for k, v in d1_fin["summary_doc"].items() if k != "seal"})}
    d6_summary = None
    d6_plan = None
    if facts["d1"]["local_mean_fit_passed"] is True:
        handoff = resolve_handoff(out_dir / "d1", d1_plan, world.decoder,
                                  world.registration_for("D1"), auth, budget,
                                  production_decoder=world.loss_fn)
        facts["d1_handoff"] = {"handoff_ready": handoff["handoff_ready"],
                               "problems": handoff["problems"][:5]}
        if handoff["handoff_ready"]:
            d6_plan = FitPlan("D6", steps=d6_steps, probe_every=d6_probe_every,
                              synthetic=True, d_model=16, n_layers=1, n_heads=4,
                              ffn_mult=2, lr=d6_lr)
            budget.check("acceptance_d6_entry")
            _, d6_summary, d6_fin = _run_fit_stage(world, d6_plan,
                                                   out_dir / "d6", budget, auth)
            facts["d6"] = {
                "local_complete_evidence": d6_summary.get("local_complete_evidence"),
                "local_mean_fit_passed": d6_summary.get("local_mean_fit_passed"),
                "local_conclusion": d6_summary.get("local_conclusion"),
                "seal": (d6_fin["summary_doc"].get("seal") or {}).get("status"),
                "call_matrix_problems": d6_summary["call_matrix"]["problems"]}
    else:
        facts["d1_handoff"] = {
            "handoff_ready": False,
            "skipped_reason": "d1_mean_fit_gate_not_passed",
            "note": "D6 is never appended as a retry (plan §7.2)"}
    # F verifies whatever stages completed locally; a complete negative
    # result is verifiable too (plan §7.3).
    f_specs = []
    if facts["d1"]["local_complete_evidence"]:
        f_specs.append({"stage_dir": out_dir / "d1", "plan": d1_plan,
                        "registration": world.registration_for("D1"),
                        "decoder": world.decoder, "loss_fn": world.loss_fn})
    if d6_summary is not None and d6_summary.get("local_complete_evidence"):
        f_specs.append({"stage_dir": out_dir / "d6", "plan": d6_plan,
                        "registration": world.registration_for("D6"),
                        "decoder": world.decoder, "loss_fn": world.loss_fn})
    if f_specs:
        budget.check("acceptance_f_entry")
        f_out = run_f_verification(
            stage_specs=f_specs, out_dir=out_dir / "f",
            budget=budget_for_stage("F", t0=budget.clock()),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world, d1_plan),
            authorization=auth)
        facts["f"] = {k: f_out["summary_doc"].get(k)
                      for k in ("complete_evidence", "replay_passed",
                                "mean_fit_passed", "verdict")}
        facts["f"]["return_coverage"] = f_out["return_coverage"]["covered"]
        facts["f"]["stages"] = [
            {k: s.get(k) for k in ("stage", "replay_enabled", "replay_passed",
                                   "skip_reason", "replay_problems")}
            for s in f_out["summary_doc"].get("stages", [])]
    battery = run_counterexample_battery(budget, seed=seed + 1)
    facts["counterexample_battery"] = battery
    facts["counterexample_ids"] = sorted(battery)
    facts["all_counterexamples_caught"] = bool(battery) and all(
        v["caught"] for v in battery.values())
    facts["budget"] = budget.facts()
    return facts


# -- counterexample battery ---------------------------------------------------

COUNTEREXAMPLE_IDS = (
    "registration_tamper", "missing_window", "duplicate_window",
    "val_test_mixing", "conditioner_target_leak", "empty_loader_drop_last",
    "nonfinite_prediction_stops", "saved_metric_missing_or_nonfinite",
    "param_outside_joint_clip", "call_matrix_miscount",
    "probe_uses_training_generator", "probe_forward_count_violation",
    "checkpoint_root_stats_misbinding", "work_after_deadline_stops",
    "complete_claim_after_deadline_withdrawn", "r_metadata_array_read_forbidden",
    "unbilled_materialization_refused", "sigkill_running_row_stops",
    "new_attempt_budget_not_reset", "f_identity_failure_no_forward",
    "window_gate_not_masked_by_macro", "metric_formula_and_units",
    "rotation_spike_visibility", "metric_schema_row_offset_truncation",
    "fresh_gate_pass_but_bitwise_mismatch", "intermediate_curve_tamper",
    "d1_forge_pass_boolean", "initial_last_role_swap", "tf32_runtime_drift",
    "param_dtype_device_mismatch", "d1_handoff_cannot_publish_final",
    "f_failure_preserves_local_facts", "replay_mismatch_no_loosening",
    "decode_tamper_synced_rejected",
    "fk_self_consistent_replacement_keeps_limitation",
    "teacher_derived_replacement_rejected",
    "return_proof_binding_rejected",
    "empty_artifacts_index_rejected",
    "checkpoint_dtype_tamper_rejected",
    "runtime_drift_mid_micro_stops",
    "f_forward_after_deadline_refused",
    "proof_budget_binding_rejected",
    "candidate_write_after_covered_refused",
    "candidate_write_chain_control_passes",
    "f_claim_deadline_refused",
    "stage_report_reread_inside_coverage_window",
    "f_report_reread_covered_by_f",
    "f_construct_after_deadline_refused",
    "f_runtime_drift_before_fresh_refused",
    "proof_not_a_mapping_survives_f",
)


class _FakeClock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _full_resign(sdir: Path) -> None:
    """Re-sign the whole derived binding chain so the on-disk state is
    SELF-CONSISTENT after a value tamper.  Order is strictly one-way
    (index → report → return proof → candidate): the index excludes the
    late-bound files, the report registers the index SHA, the proof binds
    the final report bytes and the candidate binds the report as well.
    Only re-derivation from primary quantities (raw x0, constants, fresh
    replay) can still catch the fault — which is exactly what the reviewed
    gates must do."""
    (sdir / "artifacts_index.json").write_text(
        json.dumps(build_artifacts_index(sdir), indent=1, allow_nan=False))
    report_path = sdir / "stage_report.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text())
    report["artifacts_index_sha256"] = sha256_file(sdir / "artifacts_index.json")
    report_path.write_text(json.dumps(report, indent=1, allow_nan=False))
    obs_path = sdir / "return_observation.json"
    if obs_path.is_file():
        obs = json.loads(obs_path.read_text())
        obs["summary_sha256"] = sha256_file(report_path)
        obs_path.write_text(json.dumps(obs, indent=1))
    cand_path = sdir / "handoff_candidate.json"
    if cand_path.is_file():
        cand = json.loads(cand_path.read_text())
        cand["stage_report_sha256"] = sha256_file(report_path)
        cand_path.write_text(json.dumps(cand, indent=1))


def run_counterexample_battery(budget: StageBudget, seed: int = 20261107) -> dict:
    """Every §6.2/v0.3/v0.4 synthetic counterexample; each case injects one
    specific fault and asserts the specific gate catches it."""
    results: dict[str, dict] = {}

    def record(cid: str, caught: bool, detail: str = "") -> None:
        results[cid] = {"caught": bool(caught), "detail": detail[:400]}

    with tempfile.TemporaryDirectory(prefix="w1_fit_cx_") as td:
        tmp = Path(td)

        # --- registration family ----------------------------------------------
        w6 = make_synthetic_world(seed=seed, n_windows=6, d_model=16)
        regs = w6.registration_for("D6")
        bad = FitRegistration("D6", windows=canonical_window_order(
            [WindowKey("synth_00", 0, 60)]
            + [WindowKey(f"synth_{i:02d}", 0, 60) for i in range(1, 5)]
            + [WindowKey("synth_XX", 0, 60)]), splits=("train",) * 6)
        caught = False
        try:
            FitDriver(trainer=None, plan=FitPlan("D6", steps=1, probe_every=1,
                                                 synthetic=True, d_model=16,
                                                 n_layers=1, n_heads=4,
                                                 ffn_mult=2),
                      registration=bad, loss_fn=None, decoder=w6.decoder,
                      probe_batch=None, budget=StageBudget("D6", 60.0, 5.0),
                      runtime_guard=SyntheticRuntimeGuard(), identity={},
                      stage_dir=tmp / "reg_tamper",
                      dataset_window_keys=list(regs.windows))
        except FitDiagnosticError as exc:
            caught = ("registration_invalid" in str(exc)
                      or "dataset_registration_window_mismatch" in str(exc))
        record("registration_tamper", caught, "renamed demo rejected")

        short = FitRegistration("D6", windows=canonical_window_order(
            list(regs.windows)[:5]), splits=("train",) * 5)
        record("missing_window", bool(short.problems()), str(short.problems()[:2]))
        dup = FitRegistration("D6", windows=canonical_window_order(
            list(regs.windows[:5]) + [regs.windows[0]]),
            splits=("train",) * 6)
        record("duplicate_window", bool(dup.problems()), str(dup.problems()[:2]))
        mixed = FitRegistration("D6", windows=regs.windows,
                                splits=("train",) * 5 + ("val",))
        record("val_test_mixing", bool(mixed.problems()),
               str(mixed.problems()[:2]))

        # --- conditioner target leak -------------------------------------------
        # direct guard unit: honest keys pass, the target state is rejected
        honest = {"kps_obj": torch.zeros(1), "kps_clip": torch.zeros(1),
                  "box_quat": torch.zeros(1), "box_t_clip": torch.zeros(1),
                  "contact": torch.zeros(1)}
        passed = conditioner_leak_guard_forward(_Noop(), honest) == "ok"
        caught = False
        try:
            conditioner_leak_guard_forward(_LeakyNoop(), {"x0": torch.zeros(1)})
        except FitDiagnosticError as exc:
            caught = "conditioner_target_leak" in str(exc)
        # end-to-end: a conditioner that reads x0 dies at the boundary
        world1 = make_synthetic_world(seed=seed + 3, n_windows=1, d_model=16)
        plan1 = FitPlan("D1", steps=1, probe_every=1, synthetic=True, d_model=16,
                        n_layers=1, n_heads=4, ffn_mult=2, lr=0.02)
        trainer_l = world1.build_trainer(plan1, tmp / "leak", None)
        trainer_l.conditioner = _LeakyConditioner(plan1.d_model)
        patch_conditioner_leak_guard(trainer_l.conditioner)
        e2e = False
        try:
            trainer_l.begin_update()
            batch = next(iter(trainer_l.train_loader))
            trainer_l.accumulate_micro(batch)
        except FitDiagnosticError as exc:
            e2e = "conditioner_target_leak" in str(exc)
        record("conditioner_target_leak", passed and caught and e2e,
               f"guard unit + end-to-end (x0 read at the conditioner boundary)")

        # --- empty loader --------------------------------------------------------
        plan6 = FitPlan("D6", steps=1, probe_every=1, synthetic=True, d_model=16,
                        n_layers=1, n_heads=4, ffn_mult=2, seed=1, val_seed=1)
        tiny_ds = SyntheticWindowDataset(w6.train_ds.items[:2],
                                         window_keys=list(w6.registration.windows[:2]))
        cfg = TrainConfig(model_type="deterministic", steps=1, micro_batch=6,
                          grad_accum=4, seed=1, val_seed=1, num_workers=0,
                          device="cpu", d_model=16, n_layers=1, n_heads=4,
                          ffn_mult=2, loss=LossWeights(w_contact=0.0))
        trainer6 = Trainer(cfg, w6.tokenizer, tiny_ds, tiny_ds,
                           tmp / "empty_loader", loss_fn=w6.loss_fn)
        caught = False
        try:
            FitDriver(trainer=trainer6, plan=plan6,
                      registration=w6.registration_for("D6"),
                      loss_fn=w6.loss_fn, decoder=w6.decoder,
                      probe_batch=w6.probe_batch,
                      budget=StageBudget("D6", 60.0, 5.0),
                      runtime_guard=SyntheticRuntimeGuard(), identity={},
                      stage_dir=tmp / "empty_loader")
        except FitDiagnosticError as exc:
            caught = ("dataset_window_count" in str(exc)
                      or "empty_loader" in str(exc))
        record("empty_loader_drop_last", caught, "2 windows vs micro_batch 6")

        # --- nonfinite prediction stops -------------------------------------------
        trainer1 = world1.build_trainer(plan1, tmp / "nonfinite", None)
        driver1 = FitDriver(trainer=trainer1, plan=plan1,
                            registration=world1.registration_for("D1"),
                            loss_fn=world1.loss_fn, decoder=world1.decoder,
                            probe_batch=world1.probe_batch,
                            budget=StageBudget("D1", 120.0, 5.0),
                            runtime_guard=SyntheticRuntimeGuard(),
                            identity={}, stage_dir=tmp / "nonfinite",
                            training_gen=torch.Generator().manual_seed(1))
        original_predict = trainer1.predict

        def nan_predict(batch, **kwargs):
            memory, x0_pred = original_predict(batch, **kwargs)
            return memory, x0_pred * float("nan")

        trainer1.predict = nan_predict
        summary_bad = driver1.run()
        caught = (summary_bad.get("local_complete_evidence") is False
                  and "nonfinite" in json.dumps(summary_bad.get("failure") or {}))
        record("nonfinite_prediction_stops", caught,
               str((summary_bad.get("failure") or {}).get("message", ""))[:120])

        # --- saved metric missing --------------------------------------------------
        world_v, plan_v, _, _, sdir_v, _, _ = _mini_stage(
            tmp / "verif", "D1", n_windows=1, steps=1, seed=seed + 4)
        pidx = json.loads((sdir_v / "probe_index.json").read_text())
        for entry in pidx["probes"]:
            for w in entry["windows"]:
                w["metrics"].pop("joint_rmse_rad", None)
        (sdir_v / "probe_index.json").write_text(
            json.dumps(pidx, indent=1, allow_nan=False))
        ver = verify_stage_local(sdir_v, plan_v, world_v.registration_for("D1"),
                                 world_v.decoder)
        record("saved_metric_missing_or_nonfinite",
               any("metric_missing" in p or "probe_metrics_mismatch" in p
                   for p in ver["problems"]), str(ver["problems"][:2]))

        # --- clip coverage -----------------------------------------------------------
        world_c, plan_c, driver_c, _, _, _, _ = _mini_stage(
            tmp / "clip", "D1", n_windows=1, steps=1, seed=seed + 5)
        caught = False
        try:
            driver_c.trainer.begin_update()
            batch = next(iter(driver_c.trainer.train_loader))
            driver_c.trainer.accumulate_micro(batch)
            for p_ in driver_c.trainer.model.parameters():
                if p_.grad is not None:
                    p_.grad.mul_(1e6)
            driver_c._check_clip_coverage_pre()
            # the fault: clip only the model, leaving the conditioner out
            torch.nn.utils.clip_grad_norm_(
                list(driver_c.trainer.model.parameters()), plan_c.grad_clip)
            driver_c._check_clip_coverage_post()
        except FitDiagnosticError as exc:
            caught = "clip_coverage" in str(exc)
        record("param_outside_joint_clip", caught,
               "conditioner grads left unclipped while the model was clipped")

        # --- call matrix miscount -------------------------------------------------------
        driver_m = FitDriver.__new__(FitDriver)
        driver_m.counters = {"conditioner_forward_calls": 3, "adamw_step_calls": 1}
        driver_m.expected_matrix = {"conditioner_forward_calls": 4,
                                    "adamw_step_calls": 1}
        problems = driver_m.call_matrix_problems()
        record("call_matrix_miscount", any("call_matrix" in p for p in problems),
               str(problems[:2]))

        # --- probe generator guards --------------------------------------------------------
        world_p, plan_p, driver_p, _, _, _, _ = _mini_stage(
            tmp / "probe", "D1", n_windows=1, steps=1, seed=seed + 6)
        caught = False
        try:
            run_probe(trainer=driver_p.trainer, probe_batch=world_p.probe_batch,
                      plan=plan_p, step=0, probe_idx=0,
                      probe_gen=driver_p.training_gen,
                      training_gen=driver_p.training_gen,
                      loss_fn=driver_p.loss_fn,
                      budget=StageBudget("D1", 60.0, 5.0), counters={},
                      decoder=world_p.decoder, teacher=None,
                      window_ids=world_p.registration_for("D1").window_ids)
        except FitDiagnosticError as exc:
            caught = "probe_generator_is_training_generator" in str(exc)
        record("probe_uses_training_generator", caught, "training gen rejected")

        caught = False
        inner_cond = driver_p.trainer.conditioner.forward.__self__
        original_forward = inner_cond.forward

        def six_call_forward(self, batch):
            out = None
            for _ in range(6):
                out = original_forward(batch)
            return out

        driver_p.trainer.conditioner.forward = types.MethodType(
            six_call_forward, inner_cond)
        try:
            run_probe(trainer=driver_p.trainer, probe_batch=world_p.probe_batch,
                      plan=plan_p, step=0, probe_idx=1,
                      probe_gen=torch.Generator(), training_gen=None,
                      loss_fn=driver_p.loss_fn,
                      budget=StageBudget("D1", 60.0, 5.0),
                      counters=driver_p.counters, decoder=world_p.decoder,
                      teacher=None,
                      window_ids=world_p.registration_for("D1").window_ids)
        except FitDiagnosticError as exc:
            caught = "probe_forward_count_violation" in str(exc)
        finally:
            driver_p.trainer.conditioner.forward = types.MethodType(
                conditioner_leak_guard_forward, inner_cond)
        record("probe_forward_count_violation", caught, "6 conditioner calls")

        # --- checkpoint root-stats misbinding ------------------------------------------------
        world_k, plan_k, _, _, sdir_k, _, _ = _mini_stage(
            tmp / "ckbind", "D1", n_windows=1, steps=1, seed=seed + 7)
        ck = torch.load(sdir_k / "last.pt", map_location="cpu", weights_only=False)
        ck["root_stats"] = {"mean": ck["root_stats"]["mean"] + 100.0,
                            "std": ck["root_stats"]["std"]}
        torch.save(ck, sdir_k / "last.pt")
        _full_resign(sdir_k)
        ver = verify_stage_local(sdir_k, plan_k, world_k.registration_for("D1"),
                                 world_k.decoder)
        record("checkpoint_root_stats_misbinding",
               any("root_stats_mismatch" in p for p in ver["problems"]),
               str([p for p in ver["problems"] if "root_stats" in p][:2]))

        # --- controlled clock: work after deadline ---------------------------------------------
        budget_d = StageBudget(stage="D1", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=_FakeClock(1e9))
        caught = False
        try:
            budget_d.check("some_unit")
        except BudgetExceeded:
            caught = True
        record("work_after_deadline_stops", caught,
               "the unit guard refuses past the work deadline")

        # --- complete claim after a deadline write: withdrawn -------------------------------------
        summary_fake = {"local_complete_evidence": True,
                        "local_mean_fit_passed": True,
                        "local_conclusion": "complete_fit"}
        seal_budget = StageBudget(stage="D1", wall_cap_s=10.0, reserve_s=2.0,
                                  t0=0.0, clock=_FakeClock(20.0))
        seal_path = tmp / "seal" / "stage_report.json"
        seal_path.parent.mkdir(parents=True, exist_ok=True)
        _write_summary_sealed(seal_path, summary_fake, seal_budget,
                              withdraw=dict(I_WITHDRAW_ON_EXCEED),
                              clock=seal_budget.clock)
        doc = json.loads(seal_path.read_text())
        caught = (doc["seal"]["terminal"] is True
                  and doc["seal"]["status"] == "budget_exceeded_after_seal"
                  and doc.get("local_complete_evidence") is False
                  and doc.get("local_mean_fit_passed") is None)
        record("complete_claim_after_deadline_withdrawn", caught,
               f"seal={doc['seal']['status']}")

        # --- R metadata-only guards -----------------------------------------------------------------
        reader = MetadataOnlyReader()
        caught = caught2 = False
        try:
            reader.load_arrays(tmp / "any.npz")
        except FitDiagnosticError as exc:
            caught = "r_metadata_only_array_read_forbidden" in str(exc)
        try:
            reader.torch_load(tmp / "any.pt")
        except FitDiagnosticError as exc:
            caught2 = "r_metadata_only_checkpoint_read_forbidden" in str(exc)
        record("r_metadata_array_read_forbidden", caught and caught2,
               "array and checkpoint reads refused")

        caught = caught2 = False
        try:
            materialize_for_fit(object(), None)
        except FitDiagnosticError as exc:
            caught = "materialization_without_billed_interval" in str(exc)
        try:
            materialize_for_fit(object(), StageBudget("R", 10.0, 2.0))
        except FitDiagnosticError as exc:
            caught2 = "materialization_stage_not_fit" in str(exc)
        record("unbilled_materialization_refused", caught and caught2,
               "no-billing / R-stage refusals")

        # --- ledger family ---------------------------------------------------------------------------
        run_root = tmp / "ledger"
        (run_root / "D1" / "attempt_001").mkdir(parents=True)
        ledger = AttemptLedger(run_root, "D1")
        ledger.begin("attempt_001")  # row stays running (simulated SIGKILL)
        prior, problems = ledger.prior_state()
        record("sigkill_running_row_stops",
               prior is None and any("running_row" in p for p in problems),
               str(problems[:1]))

        run_root2 = tmp / "ledger2"
        (run_root2 / "D1" / "attempt_001").mkdir(parents=True)
        ledger2 = AttemptLedger(run_root2, "D1")
        ledger2.begin("attempt_001")
        ledger2.finish("attempt_001", 110.0, "done", {"verdict": "incomplete"})
        budget2 = StageBudget(stage="D1", wall_cap_s=120.0, reserve_s=15.0)
        caught = False
        try:
            remaining_budget_or_refuse(run_root2, "D1", budget2)
        except FitDiagnosticError as exc:
            caught = "budget_exhausted" in str(exc)
        record("new_attempt_budget_not_reset", caught,
               "prior 110s of a 120s cap leaves less than the reserve")

        # --- F identity failure: no forward -------------------------------------------------------------
        world_f, plan_f, _, _, sdir_f, _, _ = _mini_stage(
            tmp / "fident", "D1", n_windows=1, steps=1, seed=seed + 8, finish=True)
        ck = torch.load(sdir_f / "last.pt", map_location="cpu", weights_only=False)
        ck["root_stats"] = {"mean": ck["root_stats"]["mean"] + 5.0,
                            "std": ck["root_stats"]["std"]}
        torch.save(ck, sdir_f / "last.pt")
        _full_resign(sdir_f)
        built = {"count": 0}

        def counting_build():
            built["count"] += 1
            return build_modules_factory(world_f, plan_f)()

        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_f, "plan": plan_f,
                          "registration": world_f.registration_for("D1"),
                          "decoder": world_f.decoder,
                          "loss_fn": world_f.loss_fn}],
            out_dir=tmp / "fident_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=counting_build, authorization=None)
        stage0 = f_out["summary_doc"]["stages"][0]
        record("f_identity_failure_no_forward",
               stage0.get("replay_enabled") is False
               and stage0.get("skip_reason") in ("f_identity_problems",
                                                "local_reverify_failed")
               and stage0.get("fresh_counters") is None
               and built["count"] == 0,
               f"skip={stage0.get('skip_reason')} builds={built['count']}")

        # --- gate/metric family --------------------------------------------------------------------------
        final_windows = [
            {"window_id": f"w{i}",
             "metrics": {"window_mean_fit_passed": True}} for i in range(5)
        ] + [{"window_id": "w5",
              "metrics": {"window_mean_fit_passed": False}}]
        agg = aggregate_local_verdicts(final_windows)
        agg_pass = aggregate_local_verdicts(final_windows[:1])
        record("window_gate_not_masked_by_macro",
               agg["local_mean_fit_passed"] is False
               and agg["macro_root_position_rmse_m"] is None
               and agg["macro_is_not_a_gate"] is True
               and agg_pass["local_mean_fit_passed"] is True,
               f"conjunction={agg['local_mean_fit_passed']} "
               f"(all-pass control={agg_pass['local_mean_fit_passed']})")

        tgt_dec = {"pos": np.zeros((T_WINDOW, 3)),
                   "R": np.tile(np.eye(3), (T_WINDOW, 1, 1)),
                   "q": np.zeros((T_WINDOW, J_JOINTS))}
        pred_dec = {"pos": np.zeros((T_WINDOW, 3)),
                    "R": np.tile(np.eye(3), (T_WINDOW, 1, 1)),
                    "q": np.zeros((T_WINDOW, J_JOINTS))}
        pred_dec["pos"][:, 0] = 0.03  # every frame +3cm on x only
        pred_dec["q"][:, 0] = 0.04
        fk_p = np.zeros((T_WINDOW, S_SITES, 3))
        fk_p[:, 0, 0] = 0.02
        metrics = compute_window_metrics(pred_dec, tgt_dec, fk_p,
                                         np.zeros_like(fk_p))
        formula_ok = (
            abs(metrics["root_position_rmse_m"] - 0.03) < 1e-12
            and abs(metrics["joint_rmse_rad"] - 0.04 / math.sqrt(J_JOINTS)) < 1e-12
            and abs(metrics["teacher_fk_mpjpe_m"] - 0.02 / S_SITES) < 1e-12
            and metrics["root_position_rmse_m"]
            != float(np.sqrt(0.03 ** 2 / 3.0)))
        record("metric_formula_and_units", formula_ok,
               f"rmse={metrics['root_position_rmse_m']:.4f} "
               f"joint={metrics['joint_rmse_rad']:.4f} "
               f"mpjpe={metrics['teacher_fk_mpjpe_m']:.5f}")

        angles = np.zeros(T_WINDOW)
        angles[17] = math.radians(60.0)
        rot_mean = float(np.mean(angles))
        record("rotation_spike_visibility",
               rot_mean <= GATE_ROOT_ROT_MEAN_RAD + 1e-12
               and _max_first(angles) == (math.radians(60.0), (17,))
               and SPIKE_GATE_STATUS == "not_evaluated",
               f"mean={rot_mean:.5f}rad passes the 1deg mean gate; max "
               f"{_max_first(angles)[0]:.5f}rad at frame "
               f"{_max_first(angles)[1][0]} stays visible (spikes NOT excluded)")

        # --- metric schema: missing record / truncation ------------------------------------------------------
        world_s, plan_s, _, _, sdir_s, _, _ = _mini_stage(
            tmp / "schema", "D1", n_windows=1, steps=1, seed=seed + 9)
        npz_path = next((sdir_s / "probes").glob("probe_*.npz"))
        with np.load(npz_path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays.pop("w0__md_q")
        write_npz_records(npz_path, arrays)
        _full_resign(sdir_s)
        ver = verify_stage_local(sdir_s, plan_s, world_s.registration_for("D1"),
                                 world_s.decoder)
        record("metric_schema_row_offset_truncation",
               any("record_missing" in p or "record_invalid" in p
                   for p in ver["problems"]), str(ver["problems"][:2]))

        # --- fresh gate pass but arrays differ bitwise --------------------------------------------------------
        world_b, plan_b, _, _, sdir_b, _, _ = _mini_stage(
            tmp / "bitwise", "D1", n_windows=1, steps=1, seed=seed + 10,
            finish=True)
        pidx0 = json.loads((sdir_b / "probe_index.json").read_text())
        _e0 = pidx0["probes"][0]
        npz_path = (sdir_b / "probes" /
                    f"probe_{_e0['idx']:04d}_step_{_e0['step']:06d}.npz")
        with np.load(npz_path) as data:
            arrays = {k: data[k] for k in data.files}
        perturbed = arrays["w0__raw_x0_pred"].copy()
        perturbed[0, 0] = np.nextafter(perturbed[0, 0], np.float32(np.inf))
        arrays["w0__raw_x0_pred"] = perturbed
        dec = world_b.decoder.decode(perturbed)
        tgt = read_npz_records(sdir_b / "teacher_reference.npz", TEACHER_TABLE,
                               {r: f"w0__{r}" for r in TEACHER_RECORDS})
        arrays["w0__md_pos"] = dec["pos"]
        arrays["w0__md_R"] = dec["R"]
        arrays["w0__md_q"] = dec["q"]
        arrays["w0__err_root_pos_t"] = np.linalg.norm(
            dec["pos"] - tgt["md_pos"], axis=1)
        arrays["w0__err_root_rot_t"] = so3_geodesic_angles_rad(
            dec["R"], tgt["md_R"])
        arrays["w0__err_joint_tj"] = np.abs(dec["q"] - tgt["md_q"])
        # a fully self-consistent tamper also re-derives the production
        # decode of the perturbed raw x0 (the production-decode binding
        # would otherwise catch the swap already)
        with torch.no_grad():
            _pos, _quat, _q = world_b.loss_fn.decode_clip(
                torch.as_tensor(perturbed))
        arrays["w0__prod_pos"] = _pos.numpy()
        arrays["w0__prod_quat"] = _quat.numpy()
        arrays["w0__prod_q"] = _q.numpy()
        write_npz_records(npz_path, arrays)
        pidx = json.loads((sdir_b / "probe_index.json").read_text())
        re_metrics = compute_window_metrics(
            dec, {"pos": tgt["md_pos"], "R": tgt["md_R"], "q": tgt["md_q"]},
            arrays["w0__fk_pred"], tgt["fk_tgt"])
        pidx["probes"][0]["windows"][0]["metrics"] = re_metrics
        (sdir_b / "probe_index.json").write_text(
            json.dumps(pidx, indent=1, allow_nan=False))
        _full_resign(sdir_b)
        ver = verify_stage_local(sdir_b, plan_b, world_b.registration_for("D1"),
                                 world_b.decoder)
        self_consistent = ver["problems"] == []
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_b, "plan": plan_b,
                          "registration": world_b.registration_for("D1"),
                          "decoder": world_b.decoder,
                          "loss_fn": world_b.loss_fn}],
            out_dir=tmp / "bitwise_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_b, plan_b),
            authorization=None)
        stage0 = f_out["summary_doc"]["stages"][0]
        caught = (self_consistent
                  and stage0.get("replay_passed") is False
                  and any("replay_mismatch:x0" in p
                          for p in stage0.get("replay_problems", []))
                  and f_out["summary_doc"]["mean_fit_passed"] is None
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("fresh_gate_pass_but_bitwise_mismatch", caught,
               f"self-consistent saved arrays pass local verify; F replay: "
               f"{stage0.get('replay_problems', [])[:1]}")

        # --- intermediate curve tamper -------------------------------------------------------------------------
        world_i, plan_i, _, _, sdir_i, _, _ = _mini_stage(
            tmp / "curve", "D1", n_windows=1, steps=2, probe_every=1,
            seed=seed + 11)
        pidx = json.loads((sdir_i / "probe_index.json").read_text())
        mid = next(e for e in pidx["probes"] if e["step"] == 1)
        mid["windows"][0]["metrics"]["spike"]["root_position_per_frame_m"][
            "p95"] += 1e-6
        (sdir_i / "probe_index.json").write_text(
            json.dumps(pidx, indent=1, allow_nan=False))
        ver = verify_stage_local(sdir_i, plan_i, world_i.registration_for("D1"),
                                 world_i.decoder)
        record("intermediate_curve_tamper",
               any("probe_metrics_mismatch" in p for p in ver["problems"]),
               "mid-probe p95 tampered")

        # --- D1 forge pass boolean -------------------------------------------------------------------------------
        world_g, plan_g, _, _, sdir_g, _, _ = _mini_stage(
            tmp / "forge", "D1", n_windows=1, steps=1, seed=seed + 12)
        pidx = json.loads((sdir_g / "probe_index.json").read_text())
        for w in pidx["probes"][-1]["windows"]:
            w["metrics"]["window_mean_fit_passed"] = True
            w["metrics"]["gates"] = {k: True for k in w["metrics"]["gates"]}
        (sdir_g / "probe_index.json").write_text(
            json.dumps(pidx, indent=1, allow_nan=False))
        ver = verify_stage_local(sdir_g, plan_g, world_g.registration_for("D1"),
                                 world_g.decoder)
        record("d1_forge_pass_boolean",
               any("probe_metrics_mismatch" in p for p in ver["problems"])
               and ver["local_mean_fit_passed"] is None,
               "forged booleans rejected by the raw recompute")

        # --- initial/last role or step swap ------------------------------------------------------------------------
        world_r, plan_r, _, _, sdir_r, _, _ = _mini_stage(
            tmp / "roleswap", "D1", n_windows=1, steps=1, seed=seed + 13)
        init_bytes = (sdir_r / "initial.pt").read_bytes()
        last_bytes = (sdir_r / "last.pt").read_bytes()
        (sdir_r / "initial.pt").write_bytes(last_bytes)
        (sdir_r / "last.pt").write_bytes(init_bytes)
        ver = verify_stage_local(sdir_r, plan_r, world_r.registration_for("D1"),
                                 world_r.decoder)
        record("initial_last_role_swap",
               any("checkpoint_step_mismatch" in p for p in ver["problems"]),
               "swapped checkpoints detected by the embedded step")

        # --- decode tamper synced with metrics/hashes ---------------------------------------------------------------
        world_dt, plan_dt, _, _, sdir_dt, _, _ = _mini_stage(
            tmp / "decodetamper", "D1", n_windows=1, steps=1, seed=seed + 14)
        npz_path = next((sdir_dt / "probes").glob("probe_*.npz"))
        with np.load(npz_path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["w0__md_pos"] = arrays["w0__md_pos"] + 0.001
        write_npz_records(npz_path, arrays)
        _full_resign(sdir_dt)
        ver = verify_stage_local(sdir_dt, plan_dt,
                                 world_dt.registration_for("D1"),
                                 world_dt.decoder)
        record("decode_tamper_synced_rejected",
               any("decode_binding_mismatch" in p for p in ver["problems"]),
               str([p for p in ver["problems"] if "decode_binding" in p][:1]))

        # --- FK self-consistent replacement keeps the limitation ------------------------------------------------------
        world_fk, plan_fk, _, _, sdir_fk, _, _ = _mini_stage(
            tmp / "fkreplace", "D1", n_windows=1, steps=1, seed=seed + 15)
        pidx_fk = json.loads((sdir_fk / "probe_index.json").read_text())
        _efk = pidx_fk["probes"][0]
        npz_path = (sdir_fk / "probes" /
                    f"probe_{_efk['idx']:04d}_step_{_efk['step']:06d}.npz")
        with np.load(npz_path) as data:
            arrays = {k: data[k] for k in data.files}
        replaced = arrays["w0__fk_pred"] + 0.004
        tgt = _teacher_records(sdir_fk, 0)
        arrays["w0__fk_pred"] = replaced
        arrays["w0__err_fk_ts"] = np.linalg.norm(
            replaced.astype(np.float64) - tgt["fk_tgt"].astype(np.float64),
            axis=-1)
        write_npz_records(npz_path, arrays)
        pidx = json.loads((sdir_fk / "probe_index.json").read_text())
        re_metrics = compute_window_metrics(
            {"pos": arrays["w0__md_pos"], "R": arrays["w0__md_R"],
             "q": arrays["w0__md_q"]},
            {"pos": tgt["md_pos"], "R": tgt["md_R"], "q": tgt["md_q"]},
            replaced, tgt["fk_tgt"])
        pidx["probes"][0]["windows"][0]["metrics"] = re_metrics
        (sdir_fk / "probe_index.json").write_text(
            json.dumps(pidx, indent=1, allow_nan=False))
        _full_resign(sdir_fk)
        ver = verify_stage_local(sdir_fk, plan_fk,
                                 world_fk.registration_for("D1"),
                                 world_fk.decoder)
        record("fk_self_consistent_replacement_keeps_limitation",
               ver["problems"] == []
               and ver["intermediate_fk_state_binding_status"] == "not_evaluated",
               "self-consistent FK swap verifies; the intermediate-FK "
               "limitation stays not_evaluated (never upgraded to passed)")

        # --- TF32 / runtime drift --------------------------------------------------------------------------------------
        guard = ProductionRuntimeGuard(expected_device="cpu")
        saved_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            caught = False
            try:
                guard.check_unit("drift_point")
            except FitDiagnosticError as exc:
                caught = "runtime_drift" in str(exc)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        record("tf32_runtime_drift", caught, "tf32 flip detected per unit")

        # --- param dtype/device mismatch ---------------------------------------------------------------------------------
        world_dd, plan_dd, _, _, _, _, _ = _mini_stage(
            tmp / "dtype", "D1", n_windows=1, steps=1, seed=seed + 16)
        trainer_dd = world_dd.build_trainer(plan_dd, tmp / "dtype2", None)
        trainer_dd.conditioner.register_parameter(
            "bad_float64_param",
            torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)))
        driver_dd = FitDriver(
            trainer=trainer_dd, plan=plan_dd,
            registration=world_dd.registration_for("D1"),
            loss_fn=world_dd.loss_fn, decoder=world_dd.decoder,
            probe_batch=world_dd.probe_batch,
            budget=StageBudget("D1", 60.0, 5.0),
            runtime_guard=SyntheticRuntimeGuard(), identity={},
            stage_dir=tmp / "dtype2",
            training_gen=torch.Generator().manual_seed(1))
        caught = False
        try:
            driver_dd._preflight()
        except FitDiagnosticError as exc:
            caught = "parameter_contract" in str(exc)
        record("param_dtype_device_mismatch", caught, "float64 param refused")

        # --- D1 handoff cannot publish final ------------------------------------------------------------------------------
        world_h, plan_h, _, summary_h, sdir_h, budget_h, _ = _mini_stage(
            tmp / "handoff", "D1", n_windows=1, steps=1, seed=seed + 17)
        fin = finish_stage(sdir_h, summary_h, budget_h)
        doc = fin["summary_doc"]
        report_ok = local_report_problems(
            {k: v for k, v in doc.items() if k != "seal"}) == []
        final_blocked = ("complete_evidence" not in doc
                         and "mean_fit_passed" not in doc
                         and doc.get("final_verification_status") == "pending_F")
        handoff = resolve_handoff(sdir_h, plan_h, world_h.decoder,
                                  world_h.registration_for("D1"),
                                  {"includes": []}, StageBudget("D6", 60.0, 5.0),
                                  production_decoder=world_h.loss_fn)
        record("d1_handoff_cannot_publish_final",
               report_ok and final_blocked and handoff["handoff_ready"] is False
               and any("authorization_does_not_include_d6" in p
                       for p in handoff["problems"]),
               f"pending_F + no final fields + auth refused; local fit gate "
               f"= {doc.get('local_mean_fit_passed')}")

        # --- F failure preserves local facts / no loosening ------------------------------------------------------------
        world_p2, plan_p2, _, summary_p2, sdir_p2, budget_p2, _ = _mini_stage(
            tmp / "preserve", "D1", n_windows=1, steps=1, seed=seed + 18)
        finish_stage(sdir_p2, summary_p2, budget_p2)
        report_before = sha256_file(sdir_p2 / "stage_report.json")
        ck = torch.load(sdir_p2 / "last.pt", map_location="cpu",
                        weights_only=False)
        ck["root_stats"] = {"mean": ck["root_stats"]["mean"] - 3.0,
                            "std": ck["root_stats"]["std"]}
        torch.save(ck, sdir_p2 / "last.pt")
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_p2, "plan": plan_p2,
                          "registration": world_p2.registration_for("D1"),
                          "decoder": world_p2.decoder,
                          "loss_fn": world_p2.loss_fn}],
            out_dir=tmp / "preserve_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_p2, plan_p2),
            authorization=None)
        caught = (sha256_file(sdir_p2 / "stage_report.json") == report_before
                  and f_out["summary_doc"]["complete_evidence"] is False
                  and f_out["summary_doc"]["mean_fit_passed"] is None
                  and f_out["summary_doc"]["tolerance_policy"]
                  == "bitwise_no_tolerance")
        record("f_failure_preserves_local_facts", caught,
               "stage report bytes unchanged after a failing F")
        record("replay_mismatch_no_loosening",
               f_out["summary_doc"]["verdict"] == "incomplete"
               and f_out["summary_doc"]["mean_fit_passed"] is None
               and f_out["summary_doc"]["tolerance_policy"]
               == "bitwise_no_tolerance",
               "identity-failed stage never reaches a fresh probe; no "
               "tolerance loosening path exists")

        # --- P1: teacher-derived quantities replaced by the last prediction ---
        world_t, plan_t, _, _, sdir_t, _, _ = _mini_stage(
            tmp / "teacher_swap", "D1", n_windows=1, steps=2, probe_every=1,
            seed=seed + 30, finish=True)
        pidx_t = json.loads((sdir_t / "probe_index.json").read_text())
        last_entry = pidx_t["probes"][-1]
        last_npz = (sdir_t / "probes" /
                    f"probe_{last_entry['idx']:04d}_step_{last_entry['step']:06d}.npz")
        with np.load(last_npz) as data:
            last_arrays = {k: data[k] for k in data.files}
        # keep raw_x0_tgt; replace every teacher-derived quantity with the
        # last prediction's derived quantities, then re-sync every probe's
        # metrics against the fake teacher and re-sign the whole chain
        with np.load(sdir_t / "teacher_reference.npz") as data:
            teacher_arrays = {k: data[k] for k in data.files}
        for key in ("md_pos", "md_R", "md_q", "prod_pos", "prod_quat",
                    "prod_q", "fk_tgt"):
            src_key = "w0__fk_pred" if key == "fk_tgt" else f"w0__{key}"
            teacher_arrays[f"w0__{key}"] = last_arrays[src_key]
        write_npz_records(sdir_t / "teacher_reference.npz", teacher_arrays)
        fake_md = {"pos": teacher_arrays["w0__md_pos"],
                   "R": teacher_arrays["w0__md_R"],
                   "q": teacher_arrays["w0__md_q"]}
        for entry in pidx_t["probes"]:
            npz_p = (sdir_t / "probes" /
                     f"probe_{entry['idx']:04d}_step_{entry['step']:06d}.npz")
            with np.load(npz_p) as data:
                p_arrays = {k: data[k] for k in data.files}
            p_arrays["w0__err_root_pos_t"] = np.linalg.norm(
                p_arrays["w0__md_pos"] - fake_md["pos"], axis=1)
            p_arrays["w0__err_root_rot_t"] = so3_geodesic_angles_rad(
                p_arrays["w0__md_R"], fake_md["R"])
            p_arrays["w0__err_joint_tj"] = np.abs(
                p_arrays["w0__md_q"] - fake_md["q"])
            p_arrays["w0__err_fk_ts"] = np.linalg.norm(
                p_arrays["w0__fk_pred"].astype(np.float64)
                - teacher_arrays["w0__fk_tgt"].astype(np.float64), axis=-1)
            write_npz_records(npz_p, p_arrays)
            for w in entry["windows"]:
                w["metrics"] = compute_window_metrics(
                    {"pos": p_arrays["w0__md_pos"], "R": p_arrays["w0__md_R"],
                     "q": p_arrays["w0__md_q"]}, fake_md,
                    p_arrays["w0__fk_pred"], teacher_arrays["w0__fk_tgt"])
        (sdir_t / "probe_index.json").write_text(
            json.dumps(pidx_t, indent=1, allow_nan=False))
        _full_resign(sdir_t)
        ver = verify_stage_local(sdir_t, plan_t,
                                 world_t.registration_for("D1"),
                                 world_t.decoder,
                                 production_decoder=world_t.loss_fn)
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_t, "plan": plan_t,
                          "registration": world_t.registration_for("D1"),
                          "decoder": world_t.decoder,
                          "loss_fn": world_t.loss_fn}],
            out_dir=tmp / "teacher_swap_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_t, plan_t),
            authorization=None)
        caught = (any("teacher_decode_binding_mismatch" in p
                      for p in ver["problems"])
                  and ver["local_mean_fit_passed"] is None
                  and f_out["summary_doc"]["complete_evidence"] is False
                  and f_out["summary_doc"]["mean_fit_passed"] is None)
        record("teacher_derived_replacement_rejected", caught,
               f"teacher=last-prediction swap fully re-signed; verify: "
               f"{[p for p in ver['problems'] if 'teacher' in p][:1]}")

        # --- P1: return-proof binding / existence ---------------------------------
        world_r2, plan_r2, _, _, sdir_r2, _, _ = _mini_stage(
            tmp / "proof", "D1", n_windows=1, steps=1, seed=seed + 31,
            finish=True)
        obs = json.loads((sdir_r2 / "return_observation.json").read_text())
        obs["summary_sha256"] = "0" * 64
        (sdir_r2 / "return_observation.json").write_text(
            json.dumps(obs, indent=1))
        handoff = resolve_handoff(sdir_r2, plan_r2, world_r2.decoder,
                                  world_r2.registration_for("D1"),
                                  {"includes": ["D6"]},
                                  StageBudget("D6", 60.0, 5.0),
                                  production_decoder=world_r2.loss_fn)
        proof_bound = (handoff["handoff_ready"] is False
                       and any("return_observation_summary_sha_mismatch" in p
                               for p in handoff["problems"]))
        (sdir_r2 / "return_observation.json").unlink()
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_r2, "plan": plan_r2,
                          "registration": world_r2.registration_for("D1"),
                          "decoder": world_r2.decoder,
                          "loss_fn": world_r2.loss_fn}],
            out_dir=tmp / "proof_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_r2, plan_r2),
            authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (proof_bound
                  and s0.get("replay_enabled") is False
                  and s0.get("skip_reason") == "return_observation_missing_or_uncovered"
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("return_proof_binding_rejected", caught,
               f"zero-SHA proof refused by handoff; deleted proof refused by F "
               f"(skip={s0.get('skip_reason')})")

        # --- P1: empty artifacts index ------------------------------------------------
        world_e, plan_e, _, _, sdir_e, _, _ = _mini_stage(
            tmp / "emptyidx", "D1", n_windows=1, steps=1, seed=seed + 32,
            finish=True)
        (sdir_e / "artifacts_index.json").write_text(json.dumps(
            {"schema": ARTIFACTS_INDEX_SCHEMA, "files": {}}))
        ver = verify_stage_local(sdir_e, plan_e,
                                 world_e.registration_for("D1"),
                                 world_e.decoder,
                                 production_decoder=world_e.loss_fn)
        caught = (any("artifact_required_missing" in p for p in ver["problems"])
                  and any("artifacts_index_sha_mismatch" in p
                          for p in ver["problems"])
                  and ver["local_mean_fit_passed"] is None)
        record("empty_artifacts_index_rejected", caught,
               f"empty index + unbound report SHA; "
               f"{len(ver['problems'])} problems")

        # --- P1: checkpoint dtype tamper ------------------------------------------------
        world_c4, plan_c4, _, _, sdir_c4, _, _ = _mini_stage(
            tmp / "ckptdtype", "D1", n_windows=1, steps=1, seed=seed + 33,
            finish=True)
        for name in ("initial.pt", "last.pt"):
            ck = torch.load(sdir_c4 / name, map_location="cpu",
                            weights_only=False)
            ck["model"] = {k: v.double() for k, v in ck["model"].items()}
            ck["conditioner"] = {k: v.double()
                                 for k, v in ck["conditioner"].items()}
            torch.save(ck, sdir_c4 / name)
        _full_resign(sdir_c4)
        ver = verify_stage_local(sdir_c4, plan_c4,
                                 world_c4.registration_for("D1"),
                                 world_c4.decoder,
                                 production_decoder=world_c4.loss_fn)
        built = {"count": 0}

        def counting_build_c4():
            built["count"] += 1
            return build_modules_factory(world_c4, plan_c4)()

        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_c4, "plan": plan_c4,
                          "registration": world_c4.registration_for("D1"),
                          "decoder": world_c4.decoder,
                          "loss_fn": world_c4.loss_fn}],
            out_dir=tmp / "ckptdtype_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=counting_build_c4, authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (any("checkpoint_param_dtype" in p for p in ver["problems"])
                  and ver["local_mean_fit_passed"] is None
                  and s0.get("replay_enabled") is False
                  and s0.get("fresh_counters") is None
                  and built["count"] == 0)
        record("checkpoint_dtype_tamper_rejected", caught,
               f"float64 checkpoint refused before any load; "
               f"builds={built['count']} skip={s0.get('skip_reason')}")

        # --- P1: runtime drift between micro units ----------------------------------------
        world_r5 = make_synthetic_world(seed=seed + 34, n_windows=1, d_model=16)
        plan_r5 = FitPlan("D1", steps=2, probe_every=2, synthetic=True,
                          d_model=16, n_layers=1, n_heads=4, ffn_mult=2,
                          lr=0.02)
        view_r5 = world_r5.view_for("D1")
        budget_r5 = StageBudget("D1", 300.0, 10.0)
        trainer_r5 = world_r5.build_trainer(plan_r5, tmp / "runtimedrift",
                                            budget_deadline=None,
                                            train_ds=view_r5["ds"])

        class _DriftAfter(SyntheticRuntimeGuard):
            def __init__(self, fail_after: int) -> None:
                super().__init__()
                self._fail_after = fail_after
                self.calls = 0

            def check_unit(self, point: str) -> None:
                self.calls += 1
                if self.calls > self._fail_after:
                    raise FitDiagnosticError(
                        f"runtime_drift:{point}:tf32_enabled_simulated")

        guard_r5 = _DriftAfter(fail_after=6)  # preflight, step-0 probe,
        # initial checkpoint, step start, micro-1 start and its pre-backward
        # pass; the post-backward unit check refuses (drift mid-step)
        driver_r5 = FitDriver(
            trainer=trainer_r5, plan=plan_r5,
            registration=view_r5["registration"],
            loss_fn=world_r5.loss_fn, decoder=world_r5.decoder,
            probe_batch=view_r5["probe_batch"], budget=budget_r5,
            runtime_guard=guard_r5, identity={}, stage_dir=tmp / "runtimedrift",
            training_gen=torch.Generator().manual_seed(plan_r5.seed))
        summary_r5 = driver_r5.run()
        caught = (summary_r5["local_complete_evidence"] is False
                  and "runtime_drift" in str(
                      (summary_r5.get("failure") or {}).get("message", ""))
                  and summary_r5["counters"]["backward_calls"] == 1
                  and summary_r5["counters"]["adamw_step_calls"] == 0)
        record("runtime_drift_mid_micro_stops", caught,
               f"drift after micro 1; backward_calls="
               f"{summary_r5['counters']['backward_calls']}, steps="
               f"{summary_r5['counters']['adamw_step_calls']}")

        # --- P1: F forward after the work deadline -----------------------------------------
        world_f6, plan_f6, _, _, sdir_f6, _, _ = _mini_stage(
            tmp / "fdeadline", "D1", n_windows=1, steps=1, seed=seed + 35,
            finish=True)

        class _JumpClock:
            """Pass the first two budget checks (stage entry, probe start),
            then jump past the work deadline — the load and input reads land
            beyond it and no fresh forward may start."""

            def __init__(self, passes: int, t_jump: float) -> None:
                self.n = 0
                self.passes = passes
                self.t_jump = t_jump

            def __call__(self) -> float:
                self.n += 1
                return 0.0 if self.n <= self.passes else self.t_jump

        budget_f6 = StageBudget(stage="F", wall_cap_s=300.0, reserve_s=10.0,
                                t0=0.0, clock=_JumpClock(passes=2,
                                                         t_jump=1e9))
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_f6, "plan": plan_f6,
                          "registration": world_f6.registration_for("D1"),
                          "decoder": world_f6.decoder,
                          "loss_fn": world_f6.loss_fn}],
            out_dir=tmp / "fdeadline_f", budget=budget_f6,
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_f6, plan_f6),
            authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        fc6 = s0.get("fresh_counters") or {}
        caught = (fc6.get("conditioner_forward_calls") == 0
                  and fc6.get("backbone_forward_calls") == 0
                  and any("deadline" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("f_forward_after_deadline_refused", caught,
               f"load crossed the deadline; fresh forwards="
               f"{fc6.get('conditioner_forward_calls')}")

        # --- P1(r3): proof budget numbers must bind the SEALED report ---------
        world_p3, plan_p3, _, _, sdir_p3, _, _ = _mini_stage(
            tmp / "proofbudget", "D1", n_windows=1, steps=1, seed=seed + 36,
            finish=True)
        obs = json.loads((sdir_p3 / "return_observation.json").read_text())
        # self-consistent invented chain 700<=800<=900 while the sealed
        # report registers a much smaller cost under a smaller cap
        obs["elapsed_s"] = 700.0
        obs["registered_measured_s"] = 800.0
        obs["wall_cap_s"] = 900.0
        (sdir_p3 / "return_observation.json").write_text(json.dumps(obs, indent=1))
        proof_problems = _return_proof_problems(sdir_p3)
        handoff = resolve_handoff(sdir_p3, plan_p3, world_p3.decoder,
                                  world_p3.registration_for("D1"),
                                  {"includes": ["D6"]},
                                  StageBudget("D6", 60.0, 5.0),
                                  production_decoder=world_p3.loss_fn)
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_p3, "plan": plan_p3,
                          "registration": world_p3.registration_for("D1"),
                          "decoder": world_p3.decoder,
                          "loss_fn": world_p3.loss_fn}],
            out_dir=tmp / "proofbudget_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_p3, plan_p3),
            authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (any("return_observation_registered_mismatch" in p
                      or "return_observation_wall_cap_mismatch" in p
                      for p in proof_problems)
                  and handoff["handoff_ready"] is False
                  and s0.get("replay_enabled") is False
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("proof_budget_binding_rejected", caught,
               f"proof 700<=800<=900 vs sealed report; handoff refused, "
               f"F skip={s0.get('skip_reason')}")

        # --- P1(r4): the candidate write is inside the final coverage window --
        # control FIRST: with no injected latency the same seal → candidate →
        # read → coverage chain must close clean; the attack below is only
        # meaningful against a path that passes when nothing is injected.
        world_c3, plan_c3, _, summary_c3, sdir_c3, _, _ = _mini_stage(
            tmp / "candslow", "D1", n_windows=1, steps=1, seed=seed + 37)
        clock_holder = {"t": 0.0}

        def _holder_clock() -> float:
            return clock_holder["t"]

        control_budget = StageBudget(stage="D1", wall_cap_s=600.0,
                                     reserve_s=10.0, t0=0.0,
                                     clock=_holder_clock)
        fin_ok = finish_stage(sdir_c3, summary_c3, control_budget)
        control_ok = (fin_ok["summary_doc"]["seal"]["status"]
                      == "sealed_within_budget"
                      and fin_ok["return_coverage"]["covered"] is True
                      and (sdir_c3 / "handoff_candidate.json").is_file())
        record("candidate_write_chain_control_passes", control_ok,
               "no injected latency: seal covered, candidate written "
               f"(seal={fin_ok['summary_doc']['seal']['status']}, "
               f"covered={fin_ok['return_coverage']['covered']})")

        # inject the latency INTO the candidate write: with the fixed order
        # (seal → candidate → read → coverage) the jump lands before the
        # final sample and withdraws the completion claim; a coverage check
        # that ran before the candidate write would have stayed covered.
        # The clock is SHARED with the budget so the seal/coverage loops see
        # the same timeline as the budget checks (P2 round-4: a fake budget
        # clock next to a real seal clock makes the case pass vacuously).
        world_c4, plan_c4, _, summary_c4, sdir_c4, _, _ = _mini_stage(
            tmp / "candslow2", "D1", n_windows=1, steps=1, seed=seed + 37)
        slow_budget = StageBudget(stage="D1", wall_cap_s=600.0, reserve_s=10.0,
                                  t0=0.0, clock=_holder_clock)
        mod = sys.modules[__name__]
        real_writer = mod.write_handoff_candidate

        def _slow_writer(sdir_, summary_):
            out = real_writer(sdir_, summary_)
            clock_holder["t"] = 1000.0
            return out

        mod.write_handoff_candidate = _slow_writer
        try:
            fin = finish_stage(sdir_c4, summary_c4, slow_budget)
        finally:
            mod.write_handoff_candidate = real_writer
        doc_c3 = fin["summary_doc"]
        on_disk = json.loads((sdir_c4 / "stage_report.json").read_text())
        handoff = resolve_handoff(sdir_c4, plan_c4, world_c4.decoder,
                                  world_c4.registration_for("D1"),
                                  {"includes": ["D6"]},
                                  StageBudget("D6", 60.0, 5.0),
                                  production_decoder=world_c4.loss_fn)
        caught = (control_ok
                  and doc_c3["seal"]["status"] == "budget_exceeded_after_seal"
                  and on_disk["seal"]["status"] == "budget_exceeded_after_seal"
                  and fin["return_coverage"]["covered"] is False
                  and handoff["handoff_ready"] is False)
        record("candidate_write_after_covered_refused", caught,
               "slow candidate write lands inside the checked window; "
               f"seal={doc_c3['seal']['status']} handoff="
               f"{handoff['handoff_ready']}")

        # --- P1(r4): a deadline crossed at the teacher re-check return refuses
        # the fresh metric decode and the claim itself cannot pass -----------
        world_f7, plan_f7, _, _, sdir_f7, _, _ = _mini_stage(
            tmp / "fclaimdl", "D1", n_windows=1, steps=1, seed=seed + 41,
            finish=True)
        claim_holder = {"t": 0.0}
        claim_budget = StageBudget(stage="F", wall_cap_s=300.0, reserve_s=10.0,
                                   t0=0.0, clock=lambda: claim_holder["t"])
        tchk_calls = {"n": 0}
        real_tchk = mod._teacher_reference_checked

        def _jumping_teacher(*a, **kw):
            out = real_tchk(*a, **kw)
            tchk_calls["n"] += 1
            if tchk_calls["n"] >= 2:
                # the LAST endpoint's teacher re-check just returned: jump
                # inside the wall cap but past the WORK deadline
                claim_holder["t"] = claim_budget.work_deadline + 5.0
            return out

        mod._teacher_reference_checked = _jumping_teacher
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_f7, "plan": plan_f7,
                              "registration": world_f7.registration_for("D1"),
                              "decoder": world_f7.decoder,
                              "loss_fn": world_f7.loss_fn}],
                out_dir=tmp / "fclaimdl_f", budget=claim_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_f7, plan_f7),
                authorization=None)
        finally:
            mod._teacher_reference_checked = real_tchk
        s0 = f_out["summary_doc"]["stages"][0]
        fc7 = s0.get("fresh_counters") or {}
        caught = (s0.get("replay_passed") is False
                  and any("f_fresh_metrics_last" in p
                          for p in s0.get("replay_problems", []))
                  and (fc7.get("metric_decode_calls") or 0) < 4
                  and f_out["summary_doc"].get("claim_deadline_exceeded") is True
                  and f_out["summary_doc"]["complete_evidence"] is False
                  and f_out["summary_doc"]["verdict"] == "incomplete"
                  and f_out["return_coverage"]["covered"] is True)
        record("f_claim_deadline_refused", caught,
               "deadline crossed at the teacher re-check return; the fresh "
               "metric decode refused and the claim closed incomplete "
               f"(problems={s0.get('replay_problems', [])[:1]})")

        # --- P1(r4): the post-seal report RE-READ is inside the window too ----
        # (the reviewer's round-4 attack: a slow final read used to hide
        # behind an already-covered proof; the read now precedes the sample)
        world_c5, plan_c5, _, summary_c5, sdir_c5, _, _ = _mini_stage(
            tmp / "candslow3", "D1", n_windows=1, steps=1, seed=seed + 37)
        read_holder = {"t": 0.0}
        read_budget = StageBudget(stage="D1", wall_cap_s=600.0, reserve_s=10.0,
                                  t0=0.0, clock=lambda: read_holder["t"])
        real_read_text_b = Path.read_text

        def _slow_report_read(path, *a, **kw):
            value = real_read_text_b(path, *a, **kw)
            if path.name == "stage_report.json" \
                    and sys._getframe(1).f_code.co_name == "finish_stage":
                read_holder["t"] = 1000.0
            return value

        mod.Path.read_text = _slow_report_read
        try:
            fin_r = finish_stage(sdir_c5, summary_c5, read_budget)
        finally:
            mod.Path.read_text = real_read_text_b
        on_disk_r = json.loads((sdir_c5 / "stage_report.json").read_text())
        caught = (control_ok
                  and fin_r["summary_doc"]["seal"]["status"]
                  == "budget_exceeded_after_seal"
                  and on_disk_r["seal"]["status"] == "budget_exceeded_after_seal"
                  and fin_r["return_coverage"]["covered"] is False)
        record("stage_report_reread_inside_coverage_window", caught,
               "slow post-seal report read lands inside the checked window; "
               f"seal={fin_r['summary_doc']['seal']['status']} "
               f"covered={fin_r['return_coverage']['covered']}")

        # --- P1(r4): F's own final report re-read is covered by F's own
        # return-coverage verification (not just the outer budget) -----------
        world_f8, plan_f8, _, _, sdir_f8, _, _ = _mini_stage(
            tmp / "freread", "D1", n_windows=1, steps=1, seed=seed + 42,
            finish=True)
        f_holder = {"t": 0.0}
        f_read_budget = StageBudget(stage="F", wall_cap_s=300.0, reserve_s=10.0,
                                    t0=0.0, clock=lambda: f_holder["t"])
        real_read_text_c = Path.read_text

        def _slow_f_report_read(path, *a, **kw):
            value = real_read_text_c(path, *a, **kw)
            if path.name == "f_report.json" \
                    and sys._getframe(1).f_code.co_name == "run_f_verification":
                f_holder["t"] = 1000.0
            return value

        mod.Path.read_text = _slow_f_report_read
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_f8, "plan": plan_f8,
                              "registration": world_f8.registration_for("D1"),
                              "decoder": world_f8.decoder,
                              "loss_fn": world_f8.loss_fn}],
                out_dir=tmp / "freread_f", budget=f_read_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_f8, plan_f8),
                authorization=None)
        finally:
            mod.Path.read_text = real_read_text_c
        caught = (f_out["return_coverage"]["covered"] is False
                  and f_out["summary_doc"]["verdict"]
                  == "budget_exceeded_after_seal"
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("f_report_reread_covered_by_f", caught,
               "slow post-seal F report read flips F's own coverage to "
               f"withdrawn (verdict={f_out['summary_doc']['verdict']})")

        # --- P1(r3): F refuses BEFORE construction past the deadline ----------
        world_f4, plan_f4, _, _, sdir_f4, _, _ = _mini_stage(
            tmp / "fconstruct", "D1", n_windows=1, steps=1, seed=seed + 38,
            finish=True)
        built = {"count": 0}

        def counting_build_f4():
            built["count"] += 1
            return build_modules_factory(world_f4, plan_f4)()

        # one passing budget check (stage entry), then everything refused:
        # the pre-construct check fires before any module exists
        budget_f4 = StageBudget(stage="F", wall_cap_s=300.0, reserve_s=10.0,
                                t0=0.0, clock=_JumpClock(passes=1, t_jump=1e9))
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_f4, "plan": plan_f4,
                          "registration": world_f4.registration_for("D1"),
                          "decoder": world_f4.decoder,
                          "loss_fn": world_f4.loss_fn}],
            out_dir=tmp / "fconstruct_f", budget=budget_f4,
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=counting_build_f4, authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        fc4 = s0.get("fresh_counters") or {}
        caught = (built["count"] == 0
                  and fc4.get("conditioner_forward_calls", 0) == 0
                  and any("deadline" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("f_construct_after_deadline_refused", caught,
               f"deadline crossed by the pre-verification phase; builds="
               f"{built['count']}, fresh forwards="
               f"{fc4.get('conditioner_forward_calls')}, problems="
               f"{s0.get('replay_problems', [])[:1]}")

        # --- P1(r3): runtime drift during the input reads refuses the fresh call
        world_f5, plan_f5, _, _, sdir_f5, _, _ = _mini_stage(
            tmp / "fdriftfresh", "D1", n_windows=1, steps=1, seed=seed + 39,
            finish=True)

        class _DriftAtInputsReady(SyntheticRuntimeGuard):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def check_unit(self, point: str) -> None:
                self.calls += 1
                if point.endswith("_inputs_ready"):
                    raise FitDiagnosticError(
                        f"runtime_drift:{point}:tf32_enabled_simulated")

        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_f5, "plan": plan_f5,
                          "registration": world_f5.registration_for("D1"),
                          "decoder": world_f5.decoder,
                          "loss_fn": world_f5.loss_fn}],
            out_dir=tmp / "fdriftfresh_f",
            budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=_DriftAtInputsReady(),
            build_modules=build_modules_factory(world_f5, plan_f5),
            authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        fc5 = s0.get("fresh_counters") or {}
        caught = (fc5.get("conditioner_forward_calls") == 0
                  and fc5.get("backbone_forward_calls") == 0
                  and any("runtime_drift" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("f_runtime_drift_before_fresh_refused", caught,
               f"drift at inputs_ready; fresh forwards="
               f"{fc5.get('conditioner_forward_calls')}")

        # --- P2(r3): a list-shaped return proof must fail closed, not crash ---
        world_p6, plan_p6, _, _, sdir_p6, _, _ = _mini_stage(
            tmp / "prooflist", "D1", n_windows=1, steps=1, seed=seed + 40,
            finish=True)
        (sdir_p6 / "return_observation.json").write_text("[]")
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_p6, "plan": plan_p6,
                          "registration": world_p6.registration_for("D1"),
                          "decoder": world_p6.decoder,
                          "loss_fn": world_p6.loss_fn}],
            out_dir=tmp / "prooflist_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_p6, plan_p6),
            authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        report_written = (Path(tmp) / "prooflist_f" / "f_report.json").is_file()
        handoff = resolve_handoff(sdir_p6, plan_p6, world_p6.decoder,
                                  world_p6.registration_for("D1"),
                                  {"includes": ["D6"]},
                                  StageBudget("D6", 60.0, 5.0),
                                  production_decoder=world_p6.loss_fn)
        caught = (report_written
                  and s0.get("replay_enabled") is False
                  and any("not_a_mapping" in p
                          for p in s0.get("return_proof_problems", []))
                  and handoff["handoff_ready"] is False
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("proof_not_a_mapping_survives_f", caught,
               f"list-shaped proof -> structured refusal; f_report written="
               f"{report_written}")

    return results


class _Noop:
    def forward(self, batch):
        return "ok"


class _LeakyNoop:
    def forward(self, batch):
        _ = batch["x0"]
        return "ok"


class _LeakyConditioner(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        from ireset.model.conditioner import ConditionEncoder
        self.inner = ConditionEncoder(d_model)

    def forward(self, batch):
        _ = batch["x0"]  # the fault: the conditioner reads the target state
        return self.inner(batch)
