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

import hashlib
import json
import math
import random
import shutil
import struct
import sys
import tempfile
import time
import types
import zipfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence
from unittest.mock import patch

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
# P2 (round-13): the saved probe conditions are re-read by F to rebuild the
# batch; a missing key or a wrong dtype/shape must be a structured refusal,
# never a KeyError/TypeError after modules have been constructed.
CONDITION_SHAPES = {
    "kps_obj": (T_WINDOW, 52, 3),
    "kps_clip": (T_WINDOW, 52, 3),
    "box_t_clip": (T_WINDOW, 3),
    "box_quat": (T_WINDOW, 4),
    "contact": (T_WINDOW, 4),
}

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
    artifact_space: object | None = field(default=None, repr=False)

    def closure_write_options(self) -> dict:
        if self.artifact_space is None:
            return {}
        return {"write_json": self.artifact_space.write_closure_json}

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
    if _loss_weight_values(LossWeights(w_contact=0.0)) != LOSS_WEIGHT_EXPECTED:
        problems.append("loss_weight_defaults")
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
            # P2 (round-12): isfinite/decode are invalid on a non-float dtype
            return problems
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
    """Duck-type wrapper counting production FK batch calls; optional
    pre/post unit guards bracket EACH internal loss FK call (P1 round-11)."""

    def __init__(self, inner, counter: dict, pre_call=None, post_call=None) -> None:
        self._inner = inner
        self._counter = counter
        self._pre_call = pre_call
        self._post_call = post_call

    def site_world_positions(self, *args, **kwargs):
        if self._pre_call is not None:
            self._pre_call()
        self._counter["production_fk_batch_calls"] = \
            self._counter.get("production_fk_batch_calls", 0) + 1
        out = self._inner.site_world_positions(*args, **kwargs)
        if self._post_call is not None:
            self._post_call()
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class CountingLoss:
    """Duck-type wrapper counting loss evaluations; delegates attributes."""

    def __init__(self, inner, counter: dict, pre_call=None) -> None:
        self._inner = inner
        self._counter = counter
        self._pre_call = pre_call

    def __call__(self, *args, **kwargs):
        # P1 (round-11): the ACTUAL weights are re-checked at the real loss
        # boundary — a post-construction mutation cannot consume the tampered
        # objective while the checkpoint still records the frozen one.
        if self._pre_call is not None:
            self._pre_call()
        self._counter["loss_calls"] = self._counter.get("loss_calls", 0) + 1
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _forward_hook_counter(module: torch.nn.Module, counter: dict, key: str):
    # P2 (round-11): count at the forward ENTRY boundary.  For the backbone
    # the counter is registered AFTER the admission gate, so a refused
    # backbone is never counted while an entered forward that later raises
    # still is (a post-forward hook would miss that executed call).
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
    # P2 (round-12): only numeric arrays can reach isfinite — a string dtype
    # is already a recorded dtype failure and must not raise TypeError before
    # the incomplete report can be published.
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        problems.append(f"record_nonfinite:{key}")
    return problems


def write_npz_records(path: Path, records: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **records)


def read_npz_records(path: Path, table: dict, prefix_map: dict[str, str]) -> dict:
    """Read records declared by ``prefix_map`` (record -> npz key), enforcing
    the frozen shape/dtype table before any value is consumed.

    P2 (round-13): an archive containing object arrays fails inside NpzFile
    (allow_pickle=False) with ValueError — that is a structured refusal, not
    an uncaught exception.  Pickle is never enabled.
    P2 (round-14): a CRC-corrupted member raises ``zipfile.BadZipFile`` while
    the archive is read; that belongs to the same structured refusal or F
    exits without its incomplete report."""
    try:
        with np.load(path) as data:
            out = {}
            for record, key in prefix_map.items():
                if key not in data:
                    raise FitDiagnosticError(
                        f"record_missing:{Path(path).name}:{key}")
                arr = data[key]
                problems = _check_record(record, key, arr, table)
                if problems:
                    raise FitDiagnosticError(
                        f"record_invalid:{Path(path).name}:{problems}")
                out[record] = arr
    except FitDiagnosticError:
        raise
    except (ValueError, OSError, TypeError, EOFError,
            zipfile.BadZipFile, zlib.error) as exc:
        raise FitDiagnosticError(
            f"record_archive_unreadable:{Path(path).name}:"
            f"{type(exc).__name__}") from exc
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
    files = index.get("files")
    if not isinstance(files, dict):
        return ["artifacts_index_files_not_a_mapping"]
    # a required file absent from the index is a structural failure even
    # though the (possibly empty) iteration below finds nothing to check
    for rel in required or ():
        if rel not in files:
            problems.append(f"artifact_required_missing:{rel}")
    for rel, meta in files.items():
        if not isinstance(meta, dict):
            problems.append(f"artifact_meta_not_a_mapping:{rel}")
            continue
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
        # P1 (round-9): each window's decode/production-decode/FK/reduction
        # is its own numeric unit — a completed unit returning past the work
        # deadline refuses the NEXT unit instead of finishing the batch
        budget.check(f"{point}_w{b}_decode")
        raw = np.ascontiguousarray(x0_pred_np[b])
        tgt = np.ascontiguousarray(x0_tgt_np[b])
        problems = decoder.verify_x0(raw) + decoder.verify_x0(tgt)
        if problems:
            raise FitDiagnosticError(f"probe_x0_invalid:{problems}")
        counters["metric_decode_calls"] = \
            counters.get("metric_decode_calls", 0) + 1
        dec_p = decoder.decode(raw)
        dec_t = teacher["md"][b]
        budget.check(f"{point}_w{b}_prod_decode")
        counters["production_decode_calls"] = \
            counters.get("production_decode_calls", 0) + 1
        pos_p, quat_p, q_p = loss_fn.decode_clip(x0_pred.detach()[b:b + 1])
        flat = lambda a: a.reshape(-1, a.shape[-1])
        budget.check(f"{point}_w{b}_fk")
        kp_p = loss_fn.fk.site_world_positions(
            flat(pos_p), flat(quat_p), flat(q_p), list(KPT_TO_SMPLH)
        ).view(T_WINDOW, S_SITES, 3)
        kp_p_np = kp_p.detach().cpu().numpy()
        kp_t_np = np.asarray(teacher["fk_tgt"][b])
        budget.check(f"{point}_w{b}_metrics")
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

# P1 (round-10): the frozen effective objective is NOT just the 14 optimizer/
# architecture fields — the loss weights are part of the training target and
# the mandatory zero contact weight.  Plan §7 line 144: "损失权重不变且
# w_contact=0".  A file-hash-consistent checkpoint can still claim a different
# objective; every consumer reconciles these seven fields semantically.
LOSS_WEIGHT_FIELDS = ("w_pos", "w_rot6d", "w_joint", "w_geodesic",
                      "w_fk", "w_contact", "w_vel")
LOSS_WEIGHT_EXPECTED = {"w_pos": 1.0, "w_rot6d": 1.0, "w_joint": 1.0,
                        "w_geodesic": 0.5, "w_fk": 2.0, "w_contact": 0.0,
                        "w_vel": 0.5}

# P1 (round-13): the audited Trainer's AdamW defaults are part of the update
# arithmetic as well; the ACTUAL groups must carry exactly these values.
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_MAXIMIZE = False
ADAMW_AMSGRAD = False
# P1 (round-14): these flags select DIFFERENT arithmetic inside the installed
# Torch AdamW (decoupled vs coupled weight decay, foreach/fused dispatch and
# capturable/differentiable paths).  The frozen effective configuration is the
# default decoupled formula with no explicit dispatch override, so each field
# must be exactly this value on EVERY actual parameter group.
ADAMW_ARITHMETIC_FLAGS = {
    "decoupled_weight_decay": True,
    "foreach": None,
    "fused": None,
    "capturable": False,
    "differentiable": False,
}


def _loss_weight_values(weights) -> dict | None:
    if isinstance(weights, dict):
        return weights
    if weights is not None and all(hasattr(weights, key)
                                   for key in LOSS_WEIGHT_FIELDS):
        return {key: getattr(weights, key) for key in LOSS_WEIGHT_FIELDS}
    return None


def _loss_weight_problems(weights, label: str) -> list[str]:
    """Seven frozen loss weights, item by item (bool forgery refused)."""
    values = _loss_weight_values(weights)
    if values is None:
        return [f"{label}_not_a_mapping"]
    problems: list[str] = []
    for key in LOSS_WEIGHT_FIELDS:
        want = LOSS_WEIGHT_EXPECTED[key]
        got = values.get(key)
        # P2 (round-11): a Tensor/ndarray value must be a mismatch, never an
        # ambiguous Boolean conversion that escapes the incomplete closure.
        if not _json_value_equal(got, want):
            problems.append(f"{label}:{key}:{got!r}:want{want!r}")
    extra = sorted(set(values) - set(LOSS_WEIGHT_FIELDS))
    if extra:
        problems.append(f"{label}_extra_fields:{extra[:3]}")
    return problems


def _optimizer_state_problems(opt, expected_step: int) -> list[str]:
    """P1 (round-15): the frozen claim is a FROM-SCRATCH, step-0 AdamW.  A
    preloaded (step, exp_avg, exp_avg_sq) silently changes every update while
    all frozen hyperparameters and the parameter roster still match.  At the
    declared step 0 the actual state must be EMPTY; afterwards every optimizer
    parameter must have an entry with the driver's completed step count and
    shape-aligned finite moments."""
    state = getattr(opt, "state", None)
    if not isinstance(state, dict):
        return ["optimizer_state_missing"]
    if not _is_json_int(expected_step) or expected_step < 0:
        return [f"optimizer_expected_step_invalid:{expected_step!r}"]
    if expected_step == 0:
        return ([] if not state
                else [f"optimizer_state_not_empty:{len(state)}"])
    problems: list[str] = []
    groups = getattr(opt, "param_groups", None)
    if not isinstance(groups, list) or not groups or any(
            not isinstance(group, dict) or not isinstance(group.get("params"), list)
            for group in groups):
        return ["optimizer_state_parameter_roster_invalid"]
    expected_ids = {id(param) for group in groups for param in group["params"]}
    state_ids = {id(param) for param in state}
    missing = len(expected_ids - state_ids)
    extra = len(state_ids - expected_ids)
    if missing or extra:
        problems.append(f"optimizer_state_roster_mismatch:missing{missing}:extra{extra}")
    for idx, (param, entry) in enumerate(state.items()):
        if not isinstance(entry, dict):
            problems.append(f"optimizer_state_entry_not_a_mapping:{idx}")
            continue
        step = entry.get("step")
        if torch.is_tensor(step):
            if step.numel() != 1:
                problems.append(f"optimizer_state_step_not_scalar:{idx}")
                continue
            try:
                value = float(step.detach().cpu().item())
            except (TypeError, ValueError):
                problems.append(f"optimizer_state_step_unreadable:{idx}")
                continue
        elif isinstance(step, bool) or not isinstance(step, (int, float)):
            problems.append(f"optimizer_state_step_unreadable:{idx}")
            continue
        else:
            value = float(step)
        if not math.isfinite(value) or value != float(expected_step):
            problems.append(
                f"optimizer_state_step:{idx}:{value!r}:want{expected_step}")
        param_shape = getattr(param, "shape", None)
        for key in ("exp_avg", "exp_avg_sq"):
            moment = entry.get(key)
            if (not torch.is_tensor(moment) or param_shape is None
                    or moment.shape != param_shape):
                problems.append(f"optimizer_state_{key}_shape:{idx}")
                continue
            if not bool(torch.isfinite(moment).all()):
                problems.append(f"optimizer_state_{key}_nonfinite:{idx}")
            elif key == "exp_avg_sq" and bool((moment < 0).any()):
                problems.append(f"optimizer_state_{key}_negative:{idx}")
    return problems


def _optimizer_binding_problems(trainer, plan: FitPlan, *,
                                effective_lr: float | None = None) -> list[str]:
    """P1 (round-12/13): the ACTUAL AdamW parameter groups are the update
    arithmetic — ``trainer.cfg`` being correct is not enough.  Every group's
    lr/weight_decay/betas/eps/maximize/amsgrad must equal the frozen
    contract, and the groups must cover EXACTLY the model+conditioner
    parameters (a dropped conditioner parameter would never be updated)."""
    opt = getattr(trainer, "opt", None)
    if opt is None:
        return ["optimizer_missing"]
    groups = getattr(opt, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        return ["optimizer_param_groups_missing"]
    problems: list[str] = []
    actual_ids: list[int] = []
    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            problems.append(f"optimizer_group_not_a_mapping:{idx}")
            continue
        for key, want in (("lr", plan.lr if effective_lr is None else effective_lr),
                          ("weight_decay", plan.weight_decay),
                          ("betas", ADAMW_BETAS),
                          ("eps", ADAMW_EPS),
                          ("maximize", ADAMW_MAXIMIZE),
                          ("amsgrad", ADAMW_AMSGRAD),
                          *ADAMW_ARITHMETIC_FLAGS.items()):
            got = group.get(key)
            if not _json_value_equal(got, want):
                problems.append(
                    f"optimizer_group_mismatch:{idx}:{key}:"
                    f"{got!r}:want{want!r}")
        params = group.get("params")
        if not isinstance(params, list):
            problems.append(f"optimizer_group_params_not_a_list:{idx}")
            continue
        actual_ids.extend(id(p) for p in params if torch.is_tensor(p))
    model = getattr(trainer, "model", None)
    conditioner = getattr(trainer, "conditioner", None)
    if model is not None and conditioner is not None:
        frozen = [f"{side}.{name}" for side, module in
                  (("model", model), ("conditioner", conditioner))
                  for name, param in module.named_parameters()
                  if not param.requires_grad]
        if frozen:
            problems.append(f"optimizer_parameter_frozen:{frozen[:3]}")
        expected_ids = [id(p) for p in (*model.parameters(),
                                        *conditioner.parameters())]
        missing = len(set(expected_ids) - set(actual_ids))
        extra = len(set(actual_ids) - set(expected_ids))
        if (len(actual_ids) != len(expected_ids)
                or set(actual_ids) != set(expected_ids)):
            problems.append(
                f"optimizer_parameter_roster_mismatch:missing{missing}:"
                f"extra{extra}:actual{len(actual_ids)}:want{len(expected_ids)}")
    return problems


# P1 (round-10): the stage identity must be anchored OUTSIDE the stage report.
# The driver embeds the canonical plan/registration digests into the identity
# it writes; F, the handoff and the local verification recompute those
# digests from the plan/registration THEY were given (the external anchor)
# and refuse a coordinated report+checkpoint identity rewrite.  A caller
# cannot override a binding field with a different value.
IDENTITY_PLAN_SHA_KEY = "plan_sha256"
IDENTITY_REGISTRATION_SHA_KEY = "registration_sha256"


def _canonical_sha256(obj) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _stage_identity_binding(plan: FitPlan,
                            registration: FitRegistration) -> dict:
    return {IDENTITY_PLAN_SHA_KEY: _canonical_sha256(plan.as_dict()),
            IDENTITY_REGISTRATION_SHA_KEY:
                _canonical_sha256(registration.as_dict())}


def _synthetic_stage_identity(plan: FitPlan,
                              registration: FitRegistration) -> dict:
    """The full identity a ``_mini_stage``/battery driver is constructed
    with; every F/handoff consumer must be given this external value."""
    return {"synthetic": True, **_stage_identity_binding(plan, registration)}


def _identity_binding_problems(identity, plan: FitPlan,
                               registration: FitRegistration) -> list[str]:
    if not isinstance(identity, dict):
        return ["stage_identity_not_a_mapping"]
    want = _stage_identity_binding(plan, registration)
    problems = [f"stage_identity_binding_mismatch:{key}"
                for key, value in want.items()
                if not _json_value_equal(identity.get(key), value)]
    return problems


def _checkpoint_root_stats_problems(rs, name: str,
                                    decoder: MetricDecoder) -> list[str]:
    """Container/type check BEFORE the float32 conversion (P2 round-10: a
    wrong-shaped mapping used to raise TypeError and kill F without a
    report)."""
    if not isinstance(rs, dict) or "mean" not in rs or "std" not in rs:
        return [f"checkpoint_root_stats_missing:{name}"]
    src = decoder.source_constants_f32
    problems: list[str] = []
    for key in ("mean", "std"):
        try:
            arr = np.asarray(rs[key])
            if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
                raise TypeError
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        except (TypeError, ValueError):
            problems.append(f"checkpoint_root_stats_unreadable:{name}:{key}")
            continue
        if arr.shape != src[key].shape or not np.array_equal(
                arr.view(np.uint32), src[key].view(np.uint32)):
            problems.append(f"checkpoint_root_stats_mismatch:{name}:{key}")
    return problems


def expected_call_matrix(plan: FitPlan, n_windows: int) -> dict:
    """Exact §8.2 call matrix for one registered fit stage.

    Module-level so the R admission freezes the SAME numbers the driver
    enforces at run time (single definition, no mirrored copy).
    """
    micros = plan.steps * plan.grad_accum
    return {"conditioner_forward_calls": micros + plan.n_probes,
            "backbone_forward_calls": micros + plan.n_probes,
            "loss_calls": micros,
            "adamw_step_calls": plan.steps,
            "backward_calls": micros,
            "clip_calls": plan.steps,
            "probe_calls": plan.n_probes,
            "checkpoint_saves": 2,
            "production_decode_calls": n_windows * (plan.n_probes + 1),
            "production_fk_batch_calls": 2 * micros
            + n_windows * (plan.n_probes + 1),
            "metric_decode_calls": n_windows * (plan.n_probes + 1),
            "window_consumptions_total": micros * plan.micro_batch,
            "epochs_restarted": None}


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
        # P1 (round-10): the ACTUAL loss assembler is the training target too;
        # matching ``cfg.loss`` alone lets a changed objective through.
        loss_fn_weight_problems = _loss_weight_problems(
            getattr(loss_fn, "w", None), "loss_fn_weights")
        if loss_fn_weight_problems:
            raise FitDiagnosticError(
                f"loss_weight_binding:{loss_fn_weight_problems}")
        opt_problems = _optimizer_binding_problems(trainer, plan)
        if opt_problems:
            raise FitDiagnosticError(f"optimizer_binding:{opt_problems}")
        # P1 (round-15): the step-0 empty state is part of the frozen claim.
        state_problems = _optimizer_state_problems(trainer.opt, 0)
        if state_problems:
            raise FitDiagnosticError(
                f"optimizer_state_binding:{state_problems}")
        self.trainer, self.plan, self.registration = trainer, plan, registration
        # completed native optimizer steps; the ACTUAL state counters must
        # follow it exactly (round-15).
        self._completed_steps = 0
        self.decoder, self.budget = decoder, budget
        self.runtime_guard = runtime_guard
        # P1 (round-10): the stage identity is anchored to the plan and the
        # registration (external inputs of every consumer); a conflicting
        # caller-supplied binding field is refused, never silently overridden.
        binding = _stage_identity_binding(plan, registration)
        declared = dict(identity)
        for key, value in binding.items():
            if key in declared and not _json_value_equal(declared[key], value):
                raise FitDiagnosticError(f"identity_binding_conflict:{key}")
        self.identity = {**declared, **binding}
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
        self.loss_fn = CountingLoss(loss_fn, self.counters,
                                    pre_call=self._check_loss_weights)
        trainer.loss_fn = self.loss_fn
        inner_fk = getattr(loss_fn, "fk", None)
        if isinstance(inner_fk, CountingFK):
            inner_fk = inner_fk._inner
        # P1 (round-11): the LossAssembler's OWN decode→FK chains are numeric
        # units too.  Each FK call is bracketed by pre/post deadline checks
        # (an FK returning past the deadline refuses the following teacher
        # decode/FK), and each decode_clip call is bracketed the same way.
        self._real_fk = inner_fk
        if inner_fk is not None:
            loss_fn.fk = CountingFK(inner_fk, self.counters,
                                    pre_call=self._guard_loss_fk_pre,
                                    post_call=self._guard_loss_fk_post)
        self._real_decode_clip = getattr(loss_fn, "decode_clip", None)
        if self._real_decode_clip is not None:
            loss_fn.decode_clip = self._guarded_loss_decode
        patch_conditioner_leak_guard(trainer.conditioner)
        # P1 (round-12): the conditioner is a numeric unit of its own — an
        # expiry during predict's input checks refuses the conditioner (the
        # gate precedes the counting hook, so a refused call is not counted).
        self.cond_gate_handle = trainer.conditioner.register_forward_pre_hook(
            self._pre_conditioner_gate)
        self.cond_handle = _forward_hook_counter(
            trainer.conditioner, self.counters, "conditioner_forward_calls")
        # P1 (round-9): the conditioner→backbone and backbone→loss boundaries
        # INSIDE the shared micro/predict unit are guarded with module-level
        # hooks (the shared Trainer itself is untouched): a completed unit
        # returning past the work deadline refuses the next unit of the
        # chain instead of only stopping at pre_backward.
        self._guard_point: str | None = None
        self.guard_pre_handle = trainer.model.register_forward_pre_hook(
            self._pre_backbone_gate)
        # P2 (round-10/11): count at the admission boundary — a refused
        # backbone never enters forward (not counted), while a forward that
        # has actually been entered (even if it later raises) is counted.
        # Registering the counter AFTER the admission gate gives exactly that
        # order; a post-forward hook would miss the entered-but-failed case.
        self.back_handle = _forward_hook_counter(
            trainer.model, self.counters, "backbone_forward_calls")
        self.guard_post_handle = trainer.model.register_forward_hook(
            self._pre_loss_gate)
        # P1 (round-10): ``Trainer.predict`` performs its own output-validity
        # checks AFTER the model's forward hook.  Those checks are time too:
        # the loss/FK chain must not start when they cross the work deadline.
        # The shared Trainer source stays untouched — the instance-bound
        # predictor is wrapped at the exact predict→loss boundary.
        self._real_predict = trainer.predict
        trainer.predict = self._guarded_predict
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
        # P1 (round-10): the seven loss weights are part of the frozen target
        problems += _loss_weight_problems(getattr(cfg, "loss", None),
                                          "config_loss")
        if problems:
            raise FitDiagnosticError(f"trainer_plan_binding:{problems}")
        want_windows = 1 if plan.stage == "D1" else 6
        ds_len = len(trainer.train_ds)
        if ds_len != want_windows:
            raise FitDiagnosticError(f"dataset_window_count:{ds_len}:want{want_windows}")
        if ds_len // cfg.micro_batch < 1:
            raise FitDiagnosticError("empty_loader_drop_last")

    def _expected_call_matrix(self, plan: FitPlan) -> dict:
        return expected_call_matrix(plan, len(self.registration.windows))

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

    def _pre_conditioner_gate(self, _module, _inputs) -> None:
        if self._guard_point is not None:
            self.budget.check(self._guard_point + "_pre_conditioner")
            # P1 (round-14): predict's input validity checks are NOT a
            # precision gate — a drift enabled during them refuses the
            # conditioner itself, before its forward is entered.
            self.runtime_guard.check_unit(self._guard_point + "_pre_conditioner")
            # P1 (round-14): the runtime read is itself work; its elapsed time
            # must not buy a late admission for the conditioner.
            self.budget.check(
                self._guard_point + "_pre_conditioner_post_runtime")

    def _pre_backbone_gate(self, _module, _inputs) -> None:
        if self._guard_point is not None:
            self.budget.check(self._guard_point + "_pre_backbone")
            # P1 (round-13): precision/runtime drift during the conditioner
            # refuses the backbone before it starts.
            self.runtime_guard.check_unit(self._guard_point + "_pre_backbone")
            # P1 (round-14): charge the runtime read itself before admission.
            self.budget.check(
                self._guard_point + "_pre_backbone_post_runtime")

    def _pre_loss_gate(self, _module, _inputs, _output) -> None:
        if self._guard_point is not None:
            self.budget.check(self._guard_point + "_pre_loss")
            # P1 (round-13): a drift during the backbone refuses the loss/FK
            # chain before it starts.
            self.runtime_guard.check_unit(self._guard_point + "_pre_loss")
            # P1 (round-14): charge the runtime read itself before admission.
            self.budget.check(self._guard_point + "_pre_loss_post_runtime")

    def _check_loss_weights(self) -> None:
        """The live objective is re-validated at run time (preflight, probe,
        micro, save and the actual loss call) — construction-time equality
        does not freeze an independently mutable ``loss_fn.w`` object."""
        problems = _loss_weight_problems(
            getattr(self.loss_inner, "w", None), "loss_fn_weights")
        if problems:
            raise FitDiagnosticError(f"loss_weight_drift:{problems}")

    def _check_optimizer_binding(self) -> None:
        problems = _optimizer_binding_problems(self.trainer, self.plan)
        if problems:
            raise FitDiagnosticError(f"optimizer_binding:{problems}")

    def _check_optimizer_state(self) -> None:
        problems = _optimizer_state_problems(self.trainer.opt,
                                             self._completed_steps)
        if problems:
            raise FitDiagnosticError(f"optimizer_state_drift:{problems}")

    def _check_runtime_bindings(self) -> None:
        """Config + actual loss object + ACTUAL AdamW parameter groups and
        their initial/advancing state are re-checked at every numeric
        boundary; any independent mutation is a drift, not a silently
        consumed update."""
        self._check_trainer_config(self.trainer, self.plan)
        self._check_loss_weights()
        self._check_optimizer_binding()
        self._check_optimizer_state()

    def _guard_loss_fk_pre(self) -> None:
        if self._guard_point is not None:
            self.budget.check(self._guard_point + "_loss_fk_pre")
            # P1 (round-14): the production decode may have returned under a
            # drifted precision/runtime; the FK call itself must refuse.
            self.runtime_guard.check_unit(self._guard_point + "_loss_fk_pre")
            # P1 (round-15): the runtime read is itself work — charge it
            # before the FK is admitted.
            self.budget.check(
                self._guard_point + "_loss_fk_pre_post_runtime")

    def _guard_loss_fk_post(self) -> None:
        if self._guard_point is not None:
            self.budget.check(self._guard_point + "_loss_fk_post")

    def _guarded_loss_decode(self, *args, **kwargs):
        """Bracket each LossAssembler decode under the work deadline: a
        decode returning past it refuses the FK call that follows, and a
        decode not yet started past it never starts (P1 round-11)."""
        if self._guard_point is not None:
            self.budget.check(self._guard_point + "_loss_decode_pre")
            self.runtime_guard.check_unit(self._guard_point + "_loss_decode_pre")
            # P1 (round-15): charge the runtime read before the decode starts.
            self.budget.check(
                self._guard_point + "_loss_decode_pre_post_runtime")
        out = self._real_decode_clip(*args, **kwargs)
        if self._guard_point is not None:
            # P1 (round-14): a drift enabled while the decode ran refuses the
            # following FK chain at this boundary too.
            self.budget.check(self._guard_point + "_loss_decode_post")
            self.runtime_guard.check_unit(self._guard_point + "_loss_decode_post")
            # P1 (round-15): the post-decode runtime read is charged too.
            self.budget.check(
                self._guard_point + "_loss_decode_post_post_runtime")
        return out

    def _guarded_predict(self, *args, **kwargs):
        """Predict→loss admission: the post-backbone validity checks inside
        ``Trainer.predict`` are part of the preceding numeric unit, so their
        elapsed time is charged before the loss (and its FK calls) start."""
        out = self._real_predict(*args, **kwargs)
        if self._guard_point is not None:
            self.budget.check(self._guard_point + "_post_predict")
        return out

    def _ensure_teacher(self) -> None:
        if self.teacher is not None:
            return
        self.budget.check("teacher_reference_build")
        x0_tgt = self.probe_batch["x0"].detach().cpu().numpy()
        md, prod, fk = [], [], []
        with torch.no_grad():
            for b in range(x0_tgt.shape[0]):
                # P1 (round-9): every window is its own decode/production/
                # FK unit chain; a completed decode returning past the work
                # deadline refuses the units that follow
                self.budget.check(f"teacher_window_w{b}_decode")
                self.counters["metric_decode_calls"] = \
                    self.counters.get("metric_decode_calls", 0) + 1
                md.append(self.decoder.decode(np.ascontiguousarray(x0_tgt[b])))
                # Conversion may launch device work; the metric decode must
                # not admit it after the deadline. The prod-decode check below
                # also charges the conversion before runtime/FK admission.
                self.budget.check(f"teacher_window_w{b}_target_to")
                target = (self.probe_batch["x0"].detach()[b:b + 1]
                          .to(self.trainer.cfg.device))
                point = f"teacher_window_w{b}_prod_decode"
                self.budget.check(point)
                self.runtime_guard.check_unit(point)
                self.budget.check(point + "_post_runtime")
                self.counters["production_decode_calls"] = \
                    self.counters.get("production_decode_calls", 0) + 1
                pos, quat, q = self.loss_fn.decode_clip(target)
                flat = lambda a: a.reshape(-1, a.shape[-1])
                point = f"teacher_window_w{b}_fk"
                self.budget.check(point)
                self.runtime_guard.check_unit(point)
                self.budget.check(point + "_post_runtime")
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
        # P1 (round-11/12): live objective, config and ACTUAL AdamW groups
        self._check_runtime_bindings()
        self._ensure_teacher()
        self.counters["probe_calls"] = self.counters.get("probe_calls", 0) + 1
        gen = torch.Generator()
        self._guard_point = f"probe_{idx}_step_{step}"
        try:
            entry, arrays = run_probe(
                trainer=self.trainer, probe_batch=self.probe_batch, plan=self.plan,
                step=step, probe_idx=idx, probe_gen=gen,
                training_gen=self.training_gen, loss_fn=self.loss_fn,
                budget=self.budget, counters=self.counters, decoder=self.decoder,
                teacher=self.teacher,
                window_ids=self.registration.window_ids)
        finally:
            self._guard_point = None
        self.probes.append(entry)
        self.probe_arrays_by_idx[idx] = arrays

    def _save_checkpoint(self, name: str, step: int) -> None:
        self.budget.check(f"checkpoint_{name}_start")
        self.runtime_guard.check_unit(f"checkpoint_{name}_start")
        self._check_runtime_bindings()
        self.budget.check(f"checkpoint_{name}_binding_read")
        def save():
            self.counters["checkpoint_saves"] = self.counters.get("checkpoint_saves", 0) + 1
            self.trainer.save(name, step, {"step": step}, identity=self.identity)

        space = self.budget.artifact_space
        if space is None:
            save()
        else:
            with space.checkpoint(self.stage_dir / name, (self.trainer.model, self.trainer.conditioner),
                                  check=self.budget.check):
                save()

    # -- main flow ------------------------------------------------------------

    def run(self) -> dict:
        summary: dict = {"schema": SCHEMA, "stage": self.plan.stage,
                         "plan": self.plan.as_dict(),
                         "registration": self.registration.as_dict(),
                         "identity": self.identity,
                         "runtime": self.runtime_guard.facts(),
                         "budget": self.budget.facts()}
        # P2 (round-7): the verification's executed-unit counters are kept
        # OUTSIDE the stage training matrix and are recorded on BOTH the
        # success and the failure path — a deadline stop must not erase how
        # many verification units actually ran.
        verification_counters: dict = {"production_decode_calls": 0,
                                       "metric_decode_calls": 0}
        try:
            self._preflight()
            self._train_and_probe()
            self._write_stage_artifacts()
            # the matrix is judged on the units the STAGE itself executed;
            # the local verification's own decode work is counted separately
            matrix_snapshot = dict(self.counters)
            # P1 (round-6): the local recompute runs under the SAME stage
            # budget — a deadline crossed inside it refuses the next decode
            # instead of finishing the verification first (the driver's
            # closure records the stage as incomplete).
            self._phase = "local_verification"
            verification = self._verify_saved_stage(verification_counters)
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
            summary["verification_counters"] = dict(verification_counters)
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
            self.cond_gate_handle.remove()
            self.back_handle.remove()
            self.guard_pre_handle.remove()
            self.guard_post_handle.remove()
            self.trainer.predict = self._real_predict
            # P1/P2 (round-11): restore the shared loss assembler's injected
            # wrappers so a later driver (D6) or F sees the raw methods.  The
            # decode wrapper was an INSTANCE attribute shadowing the class
            # method, so deleting it restores normal class-level dispatch
            # (and class-level observation patches in the battery).
            if "decode_clip" in self.loss_inner.__dict__:
                del self.loss_inner.__dict__["decode_clip"]
            self.loss_inner.fk = self._real_fk
            summary["counters"] = dict(self.counters)
            summary["budget"] = self.budget.facts()
        return summary

    def _preflight(self) -> None:
        self._phase = "preflight"
        self.budget.check("preflight")
        self.runtime_guard.check_unit("preflight")
        # P1 (round-11/12): freeze-check the LIVE objective, config and
        # actual AdamW groups before any numeric unit.
        self._check_runtime_bindings()
        apply_problems = getattr(self.runtime_guard, "apply_problems", None)
        if apply_problems:
            raise FitDiagnosticError(f"runtime_apply_failed:{apply_problems}")
        problems = parameter_contract_problems(
            {"model": self.trainer.model, "conditioner": self.trainer.conditioner},
            getattr(self.runtime_guard, "expected_device", "cpu"),
            getattr(self.runtime_guard, "expected_dtype", torch.float32))
        if problems:
            raise FitDiagnosticError(f"parameter_contract:{problems}")

    def _verify_saved_stage(self, counters: dict) -> dict:
        return verify_stage_local(
            self.stage_dir, self.plan, self.registration, self.decoder,
            production_decoder=self.loss_fn, counters=counters,
            budget=self.budget, expected_identity=self.identity)

    def _admit_collated(self, batch: dict, step: int, micro: int) -> None:
        """Extension gate on the exact collated object, before consumption."""

    def _count_collated_windows(self, n_windows: int) -> None:
        self.counters["window_consumptions_total"] += n_windows

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
            self._before_update(step)
            self.budget.check(f"step_{step}_after_update_admission")
            trainer.begin_update()
            parts_acc: dict[str, float] = {}
            n_micro, n_samples = 0, 0
            for micro in range(plan.grad_accum):
                self._phase = f"step_{step}_micro_{micro + 1}"
                self.budget.check(f"step_{step}_micro_{micro + 1}_start")
                self.runtime_guard.check_unit(
                    f"step_{step}_micro_{micro + 1}_start")
                self._check_runtime_bindings()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if not ever_yielded:
                        raise FitDiagnosticError("empty_loader_drop_last")
                    self.counters["epochs_restarted"] += 1
                    data_iter = iter(trainer.train_loader)
                    batch = next(data_iter)
                ever_yielded = True
                self._admit_collated(batch, step, micro + 1)
                n_windows = int(batch["x0"].shape[0])
                n_samples += n_windows
                self._count_collated_windows(n_windows)
                self.budget.check(f"step_{step}_micro_{micro + 1}_loaded")
                batch = self._to_device(batch)

                def pre_backward(_s=step, _m=micro):
                    self.budget.check(f"step_{_s}_micro_{_m + 1}_pre_backward")
                    self.runtime_guard.check_unit(
                        f"step_{_s}_micro_{_m + 1}_pre_backward")
                    # The runtime read is work too; it must not admit a
                    # backward that starts after the work deadline.
                    self.budget.check(
                        f"step_{_s}_micro_{_m + 1}_pre_backward_post_runtime")

                def post_backward():
                    self.counters["backward_calls"] += 1

                # P1 (round-9): the guard point arms the model pre-forward /
                # forward hooks, so a conditioner returning past the deadline
                # refuses the backbone (and the backbone the loss/FK chain)
                self._guard_point = f"step_{step}_micro_{micro + 1}"
                try:
                    loss, parts = trainer.accumulate_micro(
                        batch, pre_backward=pre_backward,
                        post_backward=post_backward)
                finally:
                    self._guard_point = None
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
            # P1 (round-12): the forensics read is time too — an expiry
            # during it refuses the clip that follows.
            self.budget.check(f"step_{step}_clip_coverage_pre")
            self._check_runtime_bindings()
            # P1 (round-13): the binding reads are time too — charge them
            # before admitting the clip itself.
            self.budget.check(f"step_{step}_clip_binding_read")
            self.counters["clip_calls"] += 1
            gnorm = trainer.clip_gradients()
            self._check_clip_coverage_post()
            self.budget.check(f"step_{step}_pre_optimizer_step")
            self.runtime_guard.check_unit(f"step_{step}_pre_optimizer_step")
            self._check_runtime_bindings()
            # P1 (round-13): same boundary before the actual AdamW step.
            self.budget.check(f"step_{step}_step_binding_read")
            self._phase = f"step_{step}_optimizer_update"
            self._optimizer_step_entry(step)
            self.budget.check(f"step_{step}_after_step_admission")
            self._step_optimizer()
            # P1 (round-15): one executed native step advances the frozen
            # counter exactly once; the ACTUAL optimizer state must follow it
            # (a stale/preloaded step is a refusal, never a silent update).
            self._completed_steps += 1
            self._check_optimizer_state()
            trainer._check_params()
            row = {"step": step, "n_micro": n_micro, "n_samples": n_samples,
                   **{k: float(v) for k, v in parts_acc.items()},
                   "gnorm": float(gnorm)}
            self._record_update(row)
            self.budget.check(f"step_{step}_complete")
            if step % plan.probe_every == 0:
                self._phase = f"probe_step{step}"
                self._run_probe(step, step // plan.probe_every)
        self._phase = "save_last"
        self._save_checkpoint("last.pt", plan.steps)

    def _before_update(self, step: int) -> None:
        """Extension admission only; the constant-LR path remains a no-op."""

    def _optimizer_step_entry(self, step: int) -> None:
        """Read-only extension evidence immediately before the native step."""

    def _step_optimizer(self) -> None:
        self.counters["adamw_step_calls"] += 1
        self.trainer.step_optimizer()

    def _record_update(self, row: dict) -> None:
        self.history.append(row)

    def _check_clip_coverage_pre(self) -> None:
        """Every trainable parameter must have a gradient for the joint clip."""
        missing = [name for name, p in self.trainer._named_parameters()
                   if p.grad is None]
        if missing:
            raise FitDiagnosticError(f"clip_coverage_grad_missing:{missing[:3]}")
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
        for name, pre_g in pre.items():
            if (pre_g is None) != (posts.get(name) is None):
                raise FitDiagnosticError(f"clip_coverage_grad_side:{name}")
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
                if pre_g is None or pre_g.numel() == 0:
                    continue
                expect = pre_g * coef
                if not torch.allclose(post_g, expect, rtol=1e-4, atol=1e-12):
                    raise FitDiagnosticError(f"clip_coverage_unscaled_param:{name}")

    # -- artifacts ------------------------------------------------------------

    def _write_work_json(self, path, doc):
        space = self.budget.artifact_space
        if space is None:
            path.write_text(json.dumps(doc, indent=1, allow_nan=False))
        else:
            space.write_json(path, doc, check=self.budget.check)

    def _write_work_npz(self, path, records):
        space = self.budget.artifact_space
        if space is None:
            write_npz_records(path, records)
        else:
            space.write_npz(path, records, check=self.budget.check)

    def _write_stage_artifacts(self, *, arrays_already_persisted: bool = False) -> None:
        self._phase = "write_artifacts"
        self.budget.check("write_artifacts")
        probes_dir = self.stage_dir / "probes"
        probes_dir.mkdir(parents=True, exist_ok=True)
        if not arrays_already_persisted:
            self._write_work_npz(self.stage_dir / "teacher_reference.npz",
                                 self.teacher_arrays)
            self._write_work_npz(self.stage_dir / "probe_inputs.npz", {
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
        self._write_work_json(self.stage_dir / "probe_index.json", probe_index)
        if not arrays_already_persisted:
            for idx, arrays in self.probe_arrays_by_idx.items():
                step = self.probes[idx]["step"]
                self._write_work_npz(
                    probes_dir / f"probe_{idx:04d}_step_{step:06d}.npz", arrays)
        index = build_artifacts_index(self.stage_dir)
        self._write_work_json(self.stage_dir / "artifacts_index.json", index)


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
    # P2 (round-8): a parseable but wrong-shaped seal is a structural
    # failure, never an AttributeError
    seal = report.get("seal")
    if not isinstance(seal, dict):
        problems.append("report_seal_not_a_mapping")
        seal = {}
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


def write_handoff_candidate(stage_dir: Path, summary: dict, *, write_json=None) -> dict | None:
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
    path = stage_dir / "handoff_candidate.json"
    if write_json is None:
        path.write_text(json.dumps(candidate, indent=1, allow_nan=False))
    else:
        write_json(path, candidate)
    return candidate


def _coverage_return_view(cov: dict) -> tuple[dict, dict]:
    """The coverage function returns ``final_doc`` — the FINAL document
    snapshot exactly as left on disk (covered: the entry doc; withdrawn: the
    last written withdrawn doc, including multi-round costs and the
    ``cost_coverage_unproven`` flag).  The returned view therefore matches
    the disk field-for-field without reconstructing the withdrawal
    semantics and without post-sample I/O (P2 rounds 5-6)."""
    final_doc = cov["final_doc"]
    return final_doc, dict(final_doc.get("seal") or {})


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
    write_options = budget.closure_write_options()
    seal = _write_summary_sealed(path, summary, budget,
                                 withdraw=dict(I_WITHDRAW_ON_EXCEED),
                                 clock=budget.clock, **write_options)
    candidate = write_handoff_candidate(stage_dir, summary, **write_options)
    doc = json.loads(path.read_text())
    cov = verify_return_coverage(path, budget,
                                  withdraw=dict(I_WITHDRAW_ON_EXCEED),
                                  clock=budget.clock, **write_options)
    doc, seal_view = _coverage_return_view(cov)
    return {"stage_report": str(path), "seal": seal_view,
            "return_coverage": cov,
            "seal_loop": seal,
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
            if not isinstance(sv, dict):
                problems.append(f"metric_container_mismatch:{path}{key}")
            else:
                problems += _metrics_equal(sv, value, f"{path}{key}.")
        elif isinstance(value, bool):
            if not _json_value_equal(sv, value):
                problems.append(f"metric_bool_mismatch:{path}{key}")
        elif isinstance(value, int):
            # P2 (round-11): an index/coordinate is an exact JSON integer —
            # 3.0 or true in an int field is a type violation, not equality.
            if not _json_value_equal(sv, value):
                problems.append(f"metric_int_type_or_value:{path}{key}")
        elif isinstance(value, float):
            if not _json_value_equal(sv, value):
                problems.append(f"metric_value_mismatch:{path}{key}")
        elif isinstance(value, (list, tuple)):
            if not _json_value_equal(sv, value):
                problems.append(f"metric_list_type_or_value:{path}{key}")
        else:
            if not _json_value_equal(sv, value):
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


def _is_json_int(value) -> bool:
    """A JSON integer (bools are ints in Python but never valid indices)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _json_value_equal(got, want) -> bool:
    """Type-strict JSON value comparison that never lets Tensor/ndarray
    equality reach an ``if`` (P2 round-11: a multi-element container used to
    raise RuntimeError/ValueError instead of closing as a structured
    refusal)."""
    if want is None:
        return got is None
    if isinstance(want, bool):
        return isinstance(got, bool) and got == want
    if isinstance(want, int):
        return _is_json_int(got) and got == want
    if isinstance(want, float):
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            return False
        # P2 (round-12): a huge Python int (e.g. 10**400) makes float()
        # raise OverflowError — a mismatch, never an F crash.
        try:
            return float(got) == want
        except (OverflowError, ValueError):
            return False
    if isinstance(want, str):
        return isinstance(got, str) and got == want
    if isinstance(want, dict):
        return (isinstance(got, dict) and set(got) == set(want)
                and all(_json_value_equal(got[key], want[key]) for key in want))
    if isinstance(want, (list, tuple)):
        return (isinstance(got, (list, tuple)) and len(got) == len(want)
                and all(_json_value_equal(g, w) for g, w in zip(got, want)))
    return False


def verify_stage_local(stage_dir: Path, plan: FitPlan,
                       registration: FitRegistration,
                       decoder: MetricDecoder,
                       production_decoder=None,
                       counters: dict | None = None,
                       budget: StageBudget | None = None,
                       expected_identity: dict | None = None) -> dict:
    """§7.4 items 1-3 for one stage: coverage multiset, full CPU recompute of
    every saved probe AND the frozen teacher references from disk raw x0
    (bitwise decode bindings on both sides), production-decode re-derivation
    bindings when the production chain is available, and checkpoint
    role/step/dtype binding.

    When ``budget`` is given, every numeric unit of the recompute (teacher
    decode, teacher production decode, probe production decode, probe
    decode, metric reduction) is guarded: a deadline crossed inside the
    saved-quantity verification refuses the NEXT unit (P1 round-5 — the
    caller decides how the stage closes; the checks never run silently
    past the work deadline)."""
    stage_dir = Path(stage_dir)
    check = budget.check if budget is not None else (lambda _point: None)
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
    index, index_problems = _read_json_object(index_path)
    if index is None:
        problems.append(f"artifacts_index_unreadable:{index_problems[:2]}")
        return dict(incomplete)
    problems += verify_artifacts_index(stage_dir, index,
                                       required=_required_stage_artifacts(plan))
    # the stage report registers the index SHA; consume it whenever the
    # report already exists (F / handoff contexts)
    report_path = stage_dir / "stage_report.json"
    report_identity: dict | None = None
    identity_failed = False
    if report_path.is_file():
        report, report_problems = _read_json_object(report_path)
        if report is None:
            problems.append(f"stage_report_invalid:{report_problems[:2]}")
            identity_failed = True
        else:
            if report.get("artifacts_index_sha256") != sha256_file(index_path):
                problems.append("artifacts_index_sha_mismatch")
            # P1 (round-9): the stage's declared identity is the semantic
            # anchor the checkpoint identities must equal
            declared = report.get("identity")
            if isinstance(declared, dict):
                report_identity = declared
            else:
                problems.append("stage_report_identity_invalid")
                identity_failed = True
    if expected_identity is not None:
        if report_identity is not None and not _json_value_equal(
                report_identity, expected_identity):
            problems.append("stage_report_identity_mismatch")
            identity_failed = True
        identity_to_bind = expected_identity
    else:
        identity_to_bind = report_identity
    if identity_to_bind is not None:
        # P1 (round-10): the identity is anchored to the plan/registration
        # this verifier was given — a coordinated report+checkpoint rewrite
        # cannot make a foreign identity self-consistent.
        binding_problems = _identity_binding_problems(identity_to_bind, plan,
                                                      registration)
        if binding_problems:
            identity_failed = True
        problems += binding_problems
    probe_index_path = stage_dir / "probe_index.json"
    if not probe_index_path.is_file():
        problems.append("probe_index_missing")
        return dict(incomplete)
    # P2 (round-8): a parseable but WRONG-SHAPE container is a structural
    # failure routed into the normal incomplete closure — never an
    # uncaught AttributeError that would leave F without a report.
    pidx, pidx_problems = _read_json_object(probe_index_path)
    if pidx is None:
        problems.append(f"probe_index_invalid:{pidx_problems[:2]}")
        return dict(incomplete)
    if pidx.get("schema") != PROBE_INDEX_SCHEMA:
        problems.append("probe_index_schema")
    probes_raw = pidx.get("probes")
    if not isinstance(probes_raw, list):
        problems.append("probe_index_probes_not_a_list")
        probes: list = []
    else:
        probes = probes_raw

    # 1. coverage multiset ----------------------------------------------------
    want: dict = {}
    for step in plan.probe_steps:
        for wid in registration.window_ids:
            want[(step, wid)] = want.get((step, wid), 0) + 1
    got: dict = {}
    window_ids = list(registration.window_ids)
    for entry in probes:
        if not isinstance(entry, dict):
            problems.append("probe_entry_not_a_mapping")
            continue
        step = entry.get("step")
        idx_raw = entry.get("idx")
        # P2 (round-9): type-check BEFORE hash-key construction — a list
        # step/window_id used to raise TypeError and kill F without a report
        if not _is_json_int(step) or not _is_json_int(idx_raw):
            problems.append(
                f"probe_entry_index_invalid:idx{idx_raw!r}:step{step!r}")
            continue
        windows_raw = entry.get("windows")
        if not isinstance(windows_raw, list):
            problems.append(f"probe_windows_not_a_list:idx{idx_raw}")
            continue
        for w in windows_raw:
            if not isinstance(w, dict):
                problems.append(f"probe_window_not_a_mapping:idx{idx_raw}")
                continue
            wid = w.get("window_id")
            if not isinstance(wid, str):
                problems.append(
                    f"probe_window_id_invalid:idx{idx_raw}:{wid!r}")
                continue
            key = (step, wid)
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
            # P1 (round-7): every disk read is a unit — a read returning
            # past the work deadline refuses the NEXT read
            check(f"saved_teacher_read_w{b}")
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
        check(f"saved_teacher_decode_w{b}")
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
        check(f"saved_teacher_prod_decode_w{b}")
        problems += _production_decode_binding_problems(
            recs, raw_tgt, production_decoder, counters, f"teacher:w{b}")
    inputs_path = stage_dir / "probe_inputs.npz"
    if inputs_path.is_file():
        for b in range(len(window_ids)):
            check(f"saved_inputs_read_w{b}")
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
    for entry in probes:
        if not isinstance(entry, dict):
            continue
        step, idx = entry.get("step"), entry.get("idx")
        # P2 (round-9): index/step must be JSON integers BEFORE any format
        # spec; a wrong-typed entry is a structural problem, never a
        # ValueError/TypeError that escapes the incomplete closure
        if not _is_json_int(step) or not _is_json_int(idx):
            problems.append(
                f"probe_entry_index_invalid:idx{idx!r}:step{step!r}")
            continue
        windows_raw = entry.get("windows")
        if not isinstance(windows_raw, list):
            continue  # already reported by the coverage pass
        if len(windows_raw) != len(window_ids):
            problems.append(f"probe_windows_count:idx{idx}:"
                            f"{len(windows_raw)}:want{len(window_ids)}")
            continue
        npz_path = stage_dir / "probes" / f"probe_{idx:04d}_step_{step:06d}.npz"
        if not npz_path.is_file():
            problems.append(f"probe_npz_missing:idx{idx}")
            continue
        fc = entry.get("forward_counts")
        # P2 (round-11): exact JSON integers — true == 1 in Python but is not
        # an executed forward count.
        if (not isinstance(fc, dict)
                or not _is_json_int(fc.get("conditioner"))
                or not _is_json_int(fc.get("backbone"))
                or fc.get("conditioner") != 1 or fc.get("backbone") != 1):
            problems.append(f"probe_forward_counts:idx{idx}:{fc!r}")
        for b, wout in enumerate(windows_raw):
            if not isinstance(wout, dict):
                problems.append(f"probe_window_not_a_mapping:idx{idx}:w{b}")
                continue
            if wout.get("window_id") != window_ids[b]:
                problems.append(f"probe_window_order:idx{idx}:w{b}")
            check(f"saved_probe_read_{idx}")
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
            check(f"saved_probe_prod_decode_{idx}_w{b}")
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
            check(f"saved_probe_decode_{idx}_w{b}")
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
            check(f"saved_metrics_{idx}_w{b}")
            re_metrics = compute_window_metrics(
                re_md, tgt_md, recs["fk_pred"], teacher_records[b]["fk_tgt"])
            saved_metrics = wout.get("metrics")
            # P2 (round-10): a list-shaped metrics container used to reach
            # ``saved[key]`` and raise TypeError; it is a structural problem
            # that must still close as an incomplete report.
            if not isinstance(saved_metrics, dict):
                problems.append(
                    f"probe_metrics_not_a_mapping:idx{idx}:w{b}")
            else:
                problems += [f"probe_metrics_mismatch:idx{idx}:w{b}:{p}"
                             for p in _metrics_equal(saved_metrics, re_metrics)]
            if wout.get("spike_gate_status") != SPIKE_GATE_STATUS:
                problems.append(f"spike_gate_status_invalid:idx{idx}:w{b}")
            if step == final_step:
                final_windows.append({"window_id": wout.get("window_id"),
                                      "step": step, "metrics": re_metrics})
    total_rows = sum(got.values())
    coverage = {"expected_rows": sum(want.values()), "actual_rows": total_rows,
                "missing": missing, "extra": extra, "duplicated": duplicated}

    # 4. checkpoint role/step/config/identity binding ---------------------------
    problems += _checkpoint_binding_problems(
        stage_dir, plan, decoder, budget=budget,
        expected_identity=identity_to_bind,
        counters=counters,
        # P1 (round-15): an already-refused identity skips the reference
        # construction (it is a real, budgeted numeric unit).
        skip_state_contract=identity_failed)

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


# P1 (round-9): file-hash self-consistency is not a semantic binding — the
# checkpoint's embedded effective configuration and stage identity must be
# reconciled ITEM BY ITEM against the frozen plan and the stage's declared
# identity before any construction/load can lean on them.
CHECKPOINT_CONFIG_FIELDS = ("steps", "micro_batch", "grad_accum", "seed",
                            "lr", "weight_decay", "grad_clip", "model_type",
                            "d_model", "n_layers", "n_heads", "ffn_mult",
                            "time_window", "val_seed")


def _checkpoint_config_problems(cfg, plan: FitPlan, name: str) -> list[str]:
    """Full frozen-config contract of one saved checkpoint (14 plan fields +
    the seven frozen loss weights, P1 round-10)."""
    if not isinstance(cfg, dict):
        return [f"checkpoint_config_missing:{name}"]
    problems: list[str] = []
    for key in CHECKPOINT_CONFIG_FIELDS:
        want = getattr(plan, key)
        got = cfg.get(key)
        # P2 (round-11): safe type-strict equality — a Tensor/ndarray value
        # cannot reach an ``if`` and crash F before its incomplete report.
        if not _json_value_equal(got, want):
            problems.append(
                f"checkpoint_config_mismatch:{name}:{key}:{got!r}:want{want!r}")
    loss_cfg = cfg.get("loss")
    if not isinstance(loss_cfg, dict):
        problems.append(f"checkpoint_config_loss_missing:{name}")
    else:
        problems += [f"checkpoint_config_mismatch:{name}:{p}"
                     for p in _loss_weight_problems(loss_cfg, "loss")]
    return problems


def _checkpoint_identity_problems(ck: dict, name: str,
                                  expected_identity: dict | None) -> list[str]:
    """A saved stage identity is only meaningful when it EQUALS the external
    declared identity; a coordinated (re-signed) rewrite is a refusal."""
    if expected_identity is None:
        return []
    got = ck.get("identity")
    if not isinstance(got, dict) or not _json_value_equal(got, expected_identity):
        return [f"checkpoint_identity_mismatch:{name}"]
    return []


_STATE_CONTRACT_CACHE: dict[tuple, dict[str, dict[str, tuple]]] = {}


def _plan_fresh_modules(plan: FitPlan, budget: StageBudget | None = None,
                        counters: dict | None = None):
    """P1 (round-14): fresh modules for the FROZEN plan's architecture, from
    the same constructors ``Trainer`` uses — never trained, never forwarded.

    P1 (round-15): each construction is its own unit — a model returning past
    the work deadline refuses the conditioner construction that follows."""
    check = budget.check if budget is not None else (lambda _point: None)
    if plan.model_type == "diffusion":
        from ireset.diffusion.generator import DiffusionG1
        model = DiffusionG1(num_train_timesteps=PLAN_NUM_TRAIN_TIMESTEPS,
                            **_backbone_kwargs(plan))
    else:
        from ireset.model.deterministic import DeterministicG1
        model = DeterministicG1(**_backbone_kwargs(plan))
    if counters is not None:
        counters["reference_model_constructions"] = counters.get(
            "reference_model_constructions", 0) + 1
    check("saved_state_contract_model")
    from ireset.model.conditioner import ConditionEncoder
    conditioner = ConditionEncoder(plan.d_model)
    if counters is not None:
        counters["reference_conditioner_constructions"] = counters.get(
            "reference_conditioner_constructions", 0) + 1
    check("saved_state_contract_conditioner")
    return model, conditioner


def _plan_state_contract(plan: FitPlan,
                         budget: StageBudget | None = None,
                         counters: dict | None = None
                         ) -> dict[str, dict[str, tuple[tuple[int, ...], str]]]:
    """P1 (round-14): two checkpoints agreeing with EACH OTHER is not
    architecture legality — both can drop the same key, reshape a tensor or
    retype it.  The frozen plan (independent of the artifacts) defines the
    legal state roster: every key, shape and dtype a freshly constructed
    module carries.  Construction is a budgeted, RNG-isolated metadata unit
    (no forward, no optimizer, global torch RNG restored)."""
    key = (plan.model_type, plan.d_model, plan.n_layers, plan.n_heads,
           plan.ffn_mult, plan.time_window)
    cached = _STATE_CONTRACT_CACHE.get(key)
    if cached is not None:
        if counters is not None:
            counters["reference_contract_cache_hits"] = counters.get(
                "reference_contract_cache_hits", 0) + 1
        return cached
    if budget is not None:
        budget.check("saved_state_contract_build")
    rng_state = torch.random.get_rng_state()
    try:
        model, conditioner = _plan_fresh_modules(plan, budget, counters)
        contract = {
            "model": {name: (tuple(t.shape), str(t.dtype))
                      for name, t in model.state_dict().items()},
            "conditioner": {name: (tuple(t.shape), str(t.dtype))
                            for name, t in conditioner.state_dict().items()},
        }
    finally:
        torch.random.set_rng_state(rng_state)
    _STATE_CONTRACT_CACHE[key] = contract
    return contract


def _checkpoint_architecture_problems(
        ck: dict, name: str,
        contract: dict[str, dict[str, tuple[tuple[int, ...], str]]]
) -> list[str]:
    """Per-endpoint key-set/shape/dtype contract against the frozen plan."""
    problems: list[str] = []
    for side in ("model", "conditioner"):
        got = ck.get(side)
        if not isinstance(got, dict):
            continue  # reported by _checkpoint_state_problems
        # P2 (round-15): a readable checkpoint may carry mixed-primitive keys;
        # sorting them raises TypeError and kills F without a report.  Key
        # TYPE is validated before any ordering.
        bad_keys = [key for key in got if not isinstance(key, str)]
        if bad_keys:
            problems.append(
                f"checkpoint_{side}_key_type_not_string:{name}:"
                f"{sorted(repr(k) for k in bad_keys)[:3]}")
            continue
        want = contract[side]
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        if missing or extra:
            problems.append(
                f"checkpoint_{side}_key_set_not_frozen:{name}:"
                f"missing{missing[:3]}:extra{extra[:3]}")
            continue
        for key, (shape, dtype) in want.items():
            tensor = got[key]
            if not torch.is_tensor(tensor):
                continue  # reported by _checkpoint_state_problems
            if tuple(tensor.shape) != shape:
                problems.append(
                    f"checkpoint_{side}.{key}_shape_not_frozen:{name}:"
                    f"{tuple(tensor.shape)}:want{shape}")
            if str(tensor.dtype) != dtype:
                problems.append(
                    f"checkpoint_{side}.{key}_dtype_not_frozen:{name}:"
                    f"{tensor.dtype}:want{dtype}")
    return problems


def _checkpoint_state_problems(ck: dict, name: str
                               ) -> tuple[dict[str, frozenset], list[str]]:
    """P1 (round-13): actual tensor facts and the per-side state key sets;
    the self-reported ``parameters_finite`` flag is never trusted."""
    problems: list[str] = []
    key_sets: dict[str, frozenset] = {}
    for side in ("model", "conditioner"):
        tensors = ck.get(side)
        if not isinstance(tensors, dict):
            problems.append(f"checkpoint_{side}_not_a_mapping:{name}")
            continue
        key_sets[side] = frozenset(tensors.keys())
        if not tensors:
            problems.append(f"checkpoint_{side}_state_empty:{name}")
        for key, tensor in tensors.items():
            if not torch.is_tensor(tensor):
                problems.append(f"checkpoint_{side}.{key}_not_tensor:{name}")
                continue
            # Shape and dtype alone do not imply a dense tensor: sparse COO/
            # CSR tensors may not support isfinite/all or strict copying.
            if tensor.layout != torch.strided:
                problems.append(
                    f"checkpoint_tensor_layout:{name}:{side}.{key}:{tensor.layout}")
                continue
            if not tensor.is_floating_point():
                continue
            if tensor.dtype != torch.float32:
                problems.append(
                    f"checkpoint_param_dtype:{name}:{side}.{key}:{tensor.dtype}")
            if not bool(torch.isfinite(tensor).all()):
                problems.append(f"checkpoint_{side}.{key}_nonfinite:{name}")
    return key_sets, problems


def _checkpoint_binding_problems(stage_dir: Path, plan: FitPlan,
                                 decoder: MetricDecoder,
                                 budget: StageBudget | None = None,
                                 expected_identity: dict | None = None,
                                 skip_state_contract: bool = False,
                                 counters: dict | None = None) -> list[str]:
    """Checkpoint role/step/root-stats/config/identity binding; when
    ``budget`` is given, every checkpoint read is guarded (P1 round-7: a
    reduction returning past the work deadline refuses the checkpoint
    loads)."""
    check = budget.check if budget is not None else (lambda _point: None)
    problems: list[str] = []
    state_key_sets: dict[str, dict[str, frozenset]] = {}
    loaded: list[tuple[str, dict]] = []
    for name, want_step in (("initial.pt", 0), ("last.pt", plan.steps)):
        path = Path(stage_dir) / name
        if not path.is_file():
            problems.append(f"checkpoint_missing:{name}")
            continue
        check(f"saved_checkpoint_read_{name}")
        if counters is not None:
            counters["checkpoint_inspection_reads"] = counters.get(
                "checkpoint_inspection_reads", 0) + 1
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"checkpoint_unreadable:{name}:{type(exc).__name__}")
            continue
        if not isinstance(ck, dict):
            problems.append(f"checkpoint_not_a_mapping:{name}")
            continue
        step = ck.get("step")
        # P2 (round-11): bool/float step used to pass Python's == rule
        if not _is_json_int(step) or step != want_step:
            problems.append(f"checkpoint_step_mismatch:{name}:{step!r}")
        rs = ck.get("root_stats")
        problems += _checkpoint_root_stats_problems(rs, name, decoder)
        problems += _checkpoint_config_problems(ck.get("config"), plan, name)
        problems += _checkpoint_identity_problems(ck, name, expected_identity)
        if ck.get("parameters_finite") is not True:
            problems.append(f"checkpoint_params_nonfinite:{name}")
        # P1 (round-13): the self-reported ``parameters_finite`` flag is not
        # evidence — scan the ACTUAL tensors and bind the state-dict key sets
        # across the two endpoints (a deleted key in one checkpoint is a
        # structural mismatch even with a re-signed artifact chain).
        key_sets, state_problems = _checkpoint_state_problems(ck, name)
        problems += state_problems
        loaded.append((name, ck))
        state_key_sets[name] = key_sets
    if len(state_key_sets) == 2:
        for side in ("model", "conditioner"):
            initial = state_key_sets["initial.pt"].get(side)
            last = state_key_sets["last.pt"].get(side)
            if initial is not None and last is not None and initial != last:
                problems.append(f"checkpoint_state_key_set_mismatch:{side}")
    # P1 (round-15): the reference construction is a real, budget-guarded
    # numeric unit (model then conditioner).  It is only worth doing for a
    # stage whose declared/per-checkpoint identity is already sound — an
    # identity refusal must not silently construct reference modules.
    identity_failed = skip_state_contract or any(
        "identity" in p for p in problems)
    if loaded and not identity_failed:
        contract = _plan_state_contract(plan, budget, counters)
        for name, ck in loaded:
            problems += _checkpoint_architecture_problems(ck, name, contract)
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
                    budget: StageBudget, *,
                    expected_identity: dict,
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
    if not isinstance(expected_identity, dict):
        # P1 (round-11): the handoff identity anchor is the caller's external
        # declaration — a coordinated report+checkpoint rewrite is caught here.
        return {"handoff_ready": False, "problems": ["handoff_identity_not_declared"],
                "local_mean_fit_passed": None, "coverage": None}
    budget.check("handoff_reverify")
    report_path = stage_dir / "stage_report.json"
    if not report_path.is_file():
        return {"handoff_ready": False, "problems": ["stage_report_missing"]}
    doc, report_problems = _read_json_object(report_path)
    if doc is None:
        return {"handoff_ready": False,
                "problems": ["stage_report_invalid"] + report_problems[:3]}
    seal = doc.get("seal")
    if not isinstance(seal, dict):
        # P2 (round-8): wrong-shaped seal is a structural refusal, never a
        # crash on the following attribute access
        return {"handoff_ready": False,
                "problems": problems + ["stage_seal_not_a_mapping"]}
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
    # P1 (round-6): the handoff recompute runs under the SAME D6 budget — a
    # deadline crossed inside it refuses the NEXT decode and the handoff
    # (never a handoff_ready=true produced after the work deadline).
    try:
        verification = verify_stage_local(stage_dir, plan, registration,
                                          decoder,
                                          production_decoder=production_decoder,
                                          budget=budget,
                                          expected_identity=expected_identity)
    except BudgetExceeded as exc:
        problems.append(f"handoff_deadline_exceeded:{exc}")
        return {"handoff_ready": False, "problems": problems,
                "local_mean_fit_passed": None, "coverage": None}
    problems += verification["problems"]
    mean_fit = verification["local_mean_fit_passed"]
    # P1 (round-7): the handoff claim itself must be affordable inside the
    # work deadline — a claim sampled past it refuses the handoff (the
    # verification may have returned from a slow final unit at the edge)
    if budget.clock() >= budget.work_deadline:
        budget.exceeded = True
        budget.stopped_at = budget.stopped_at or "handoff_final_claim"
        problems.append("handoff_deadline_exceeded:handoff_final_claim")
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
                 loss_fn, counters: dict,
                 budget: StageBudget | None = None,
                 runtime_guard=None) -> tuple[np.ndarray, list[dict]]:
    """Fresh probe through the SAME production interfaces: exactly one
    conditioner + one backbone forward, then the fresh production FP32
    decode + FK points per window.  When ``budget`` is given, every window's
    production decode and FK call is guarded (P1 round-7: a decode returning
    past the work deadline refuses the following decode/FK units)."""
    check = budget.check if budget is not None else (lambda _point: None)
    device = batch["x0"].device
    with torch.no_grad():
        conditioner.eval()
        model.eval()
        # P1 (round-14): the caller's last admission is the runtime read at
        # ``inputs_ready``; its elapsed time AND a drift it may have raced
        # with are both refused here, before the conditioner is entered.
        check("f_fresh_entry")
        if runtime_guard is not None:
            runtime_guard.check_unit("f_fresh_entry")
            # P1 (round-15): the runtime read is itself work — charge it
            # before the conditioner is admitted.
            check("f_fresh_entry_post_runtime")
        # P2 (round-12): count at the ENTRY of the actual numeric call — an
        # executed conditioner/backbone that raises afterwards is still an
        # actual call; a budget refusal BEFORE the call is not.
        counters["conditioner_forward_calls"] = \
            counters.get("conditioner_forward_calls", 0) + 1
        memory = conditioner(batch)
        # P1 (round-13): the precision/runtime contract is re-checked at the
        # conditioner→backbone boundary too — a drift enabled while the
        # conditioner ran refuses the backbone before it starts.
        if runtime_guard is not None:
            runtime_guard.check_unit("f_fresh_backbone")
        # P1 (round-8): the conditioner and backbone are independent numeric
        # units — a conditioner returning past the work deadline refuses the
        # backbone forward that follows.
        check("f_fresh_backbone")
        if plan.model_type == "diffusion":
            gen = torch.Generator()
            gen.manual_seed(probe_generator_seed(plan))
            t = torch.randint(0, PLAN_NUM_TRAIN_TIMESTEPS,
                              (batch["x0"].shape[0],), generator=gen).to(device)
            noise = torch.randn(batch["x0"].shape, generator=gen).to(device)
            x_t = model.add_noise(batch["x0"], noise, t)
            counters["backbone_forward_calls"] = \
                counters.get("backbone_forward_calls", 0) + 1
            x0_pred = model(x_t, t, memory)
        else:
            counters["backbone_forward_calls"] = \
                counters.get("backbone_forward_calls", 0) + 1
            x0_pred = model(memory, batch["x0"].shape[1])
        if runtime_guard is not None:
            runtime_guard.check_unit("f_fresh_backbone_complete")
        # P1 (round-14): charge the runtime read before the per-window
        # production decode/FK chain starts.
        check("f_fresh_backbone_complete")
    x0_np = x0_pred.detach().to("cpu").numpy()
    per_window = []
    flat = lambda a: a.reshape(-1, a.shape[-1])
    for b in range(x0_np.shape[0]):
        check(f"f_fresh_prod_decode_w{b}")
        counters["production_decode_calls"] = \
            counters.get("production_decode_calls", 0) + 1
        pos, quat, q = loss_fn.decode_clip(x0_pred.detach()[b:b + 1])
        check(f"f_fresh_fk_w{b}")
        # P1 (round-14): the production decode may have returned under a
        # drifted precision/runtime — the FK itself must refuse.
        if runtime_guard is not None:
            runtime_guard.check_unit(f"f_fresh_fk_w{b}")
            # P1 (round-15): charge the runtime read before the FK starts.
            check(f"f_fresh_fk_w{b}_post_runtime")
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
            if got_t.layout != torch.strided:
                problems.append(f"checkpoint_tensor_layout:{side}.{key}:{got_t.layout}")
                continue
            if got_t.shape != want_t.shape:
                problems.append(f"checkpoint_{side}.{key}_shape")
            if got_t.dtype != want_t.dtype:
                problems.append(f"checkpoint_{side}.{key}_dtype:{got_t.dtype}")
            if got_t.is_floating_point() and not bool(torch.isfinite(got_t).all()):
                problems.append(f"checkpoint_{side}.{key}_nonfinite")
    return problems


def _load_state_dicts_checked(path: Path, model, conditioner,
                              budget: StageBudget | None = None,
                              counters: dict | None = None) -> list[str]:
    """Verify the saved tensor contract FIRST; only a clean checkpoint is
    copied into the live modules.  When ``budget`` is given, the read→load
    and model→conditioner boundaries are guarded (P1 round-8: each strict
    load is an independent unit)."""
    check = budget.check if budget is not None else (lambda _point: None)
    if counters is not None:
        counters["checkpoint_loads"] = counters.get("checkpoint_loads", 0) + 1
    ck = torch.load(path, map_location="cpu", weights_only=False)
    check(f"f_strict_load_{Path(path).stem}_pre")
    problems = _checkpoint_tensor_problems(ck, model, conditioner)
    if problems:
        return problems
    check(f"f_strict_load_{Path(path).stem}_model")
    if counters is not None:
        counters["strict_module_loads"] = counters.get("strict_module_loads", 0) + 1
    model.load_state_dict(ck["model"], strict=True)
    check(f"f_strict_load_{Path(path).stem}_conditioner")
    if counters is not None:
        counters["strict_module_loads"] = counters.get("strict_module_loads", 0) + 1
    conditioner.load_state_dict(ck["conditioner"], strict=True)
    return []


def _f_identity_problems(stage_dir: Path, plan: FitPlan,
                         decoder: MetricDecoder, index: dict,
                         budget: StageBudget | None = None,
                         expected_identity: dict | None = None,
                         counters: dict | None = None) -> list[str]:
    """P1 (round-9/11): the identity helper's checkpoint reads are units too —
    each read is guarded, the embedded configuration is reconciled against
    the frozen plan FIELD BY FIELD and the embedded stage identity against
    the CALLER'S external identity declaration (never the report's copy)."""
    check = budget.check if budget is not None else (lambda _point: None)
    problems = verify_artifacts_index(
        Path(stage_dir), index, required=_required_stage_artifacts(plan))
    state_key_sets: dict[str, dict[str, frozenset]] = {}
    for name, want_step in (("initial.pt", 0), ("last.pt", plan.steps)):
        path = Path(stage_dir) / name
        if not path.is_file():
            problems.append(f"checkpoint_missing:{name}")
            continue
        check(f"f_identity_checkpoint_read_{name}")
        if counters is not None:
            counters["checkpoint_inspection_reads"] = counters.get(
                "checkpoint_inspection_reads", 0) + 1
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"checkpoint_unreadable:{name}:{type(exc).__name__}")
            continue
        if not isinstance(ck, dict):
            problems.append(f"checkpoint_not_a_mapping:{name}")
            continue
        step = ck.get("step")
        if not _is_json_int(step) or step != want_step:
            problems.append(f"checkpoint_step_mismatch:{name}")
        problems += _checkpoint_config_problems(ck.get("config"), plan, name)
        problems += _checkpoint_identity_problems(ck, name, expected_identity)
        rs = ck.get("root_stats")
        problems += _checkpoint_root_stats_problems(rs, name, decoder)
        key_sets, state_problems = _checkpoint_state_problems(ck, name)
        problems += state_problems
        state_key_sets[name] = key_sets
    if len(state_key_sets) == 2:
        for side in ("model", "conditioner"):
            initial = state_key_sets["initial.pt"].get(side)
            last = state_key_sets["last.pt"].get(side)
            if initial is not None and last is not None and initial != last:
                problems.append(f"checkpoint_state_key_set_mismatch:{side}")
    return problems


def _saved_probe_entry(stage_dir: Path, idx: int, want_step: int) -> dict | None:
    """Saved probe index entry; a wrong-shaped container is treated as a
    missing entry (P2 round-8: structural failures close as skips, never
    AttributeError)."""
    pidx, _problems = _read_json_object(Path(stage_dir) / "probe_index.json")
    if pidx is None:
        return None
    probes = pidx.get("probes")
    if not isinstance(probes, list):
        return None
    for entry in probes:
        if not isinstance(entry, dict):
            continue
        if entry.get("idx") == idx and entry.get("step") == want_step \
                and isinstance(entry.get("windows"), list):
            return entry
    return None


def _saved_probe_records(stage_dir: Path, idx: int, b: int) -> dict:
    pidx, _problems = _read_json_object(Path(stage_dir) / "probe_index.json")
    if pidx is None:
        raise FitDiagnosticError("probe_index_invalid")
    probes = pidx.get("probes")
    if not isinstance(probes, list):
        raise FitDiagnosticError("probe_index_probes_not_a_list")
    entry = next((e for e in probes
                  if isinstance(e, dict) and e.get("idx") == idx), None)
    if entry is None or not isinstance(entry.get("step"), int):
        raise FitDiagnosticError(f"probe_entry_invalid:idx{idx}")
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
                               b: int, counters: dict | None = None,
                               budget: StageBudget | None = None) -> tuple[dict, list[str]]:
    """Teacher reference with the metric decode re-derived from the saved RAW
    teacher x0 and bitwise-bound against the saved md_* arrays (P1: a
    teacher-derived replacement can never feed the F metrics).  When
    ``budget`` is given, the teacher decode is guarded: a read returning
    past the work deadline refuses it (P1 round-7)."""
    check = budget.check if budget is not None else (lambda _point: None)
    problems: list[str] = []
    recs = _teacher_records(stage_dir, b)
    problems += [f"teacher_raw_invalid:w{b}:{p}"
                 for p in decoder.verify_x0(recs["raw_x0_tgt"])]
    check(f"f_teacher_decode_w{b}")
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
                       authorization: dict | None,
                       finalize_evidence: Callable[[dict], None] | None = None) -> dict:
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
        # P1 (round-11): the stage identity anchor is the caller's FULL
        # external declaration.  Only the plan/registration digests are
        # derived; every other field (stage/module/implementation/...) is
        # compared against this external value, so a coordinated rewrite of
        # the report and both checkpoint identities cannot self-authenticate.
        expected_identity = spec.get("identity")
        res: dict = {"stage": plan.stage, "stage_dir": str(stage_dir)}
        if not isinstance(expected_identity, dict):
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "stage_identity_not_declared"})
            stages_out.append(res)
            evidence_all_ok = False
            continue
        # P2 (round-5): a deadline crossed by an earlier stage must stop the
        # SUBSEQUENT stages without crashing the F closure — the stage-entry
        # check is non-raising here and the incomplete report still lands.
        if budget.clock() >= budget.work_deadline:
            budget.exceeded = True
            budget.stopped_at = budget.stopped_at or f"f_stage_{plan.stage}_entry"
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "stage_deadline_stop",
                        "deadline_stop": f"f_stage_{plan.stage}_entry"})
            stages_out.append(res)
            evidence_all_ok = False
            break
        report_path = stage_dir / "stage_report.json"
        doc, report_problems = _read_json_object(report_path)
        if doc is None:
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "stage_report_invalid",
                        "report_problems": report_problems[:4]})
            stages_out.append(res)
            evidence_all_ok = False
            continue
        # P2 (round-8): a parseable but wrong-shaped seal must land in the
        # normal incomplete closure (skip + report), never crash the run
        seal = doc.get("seal")
        if not isinstance(seal, dict) or \
                seal.get("status") != "sealed_within_budget":
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
        try:
            extension_verifier = spec.get("verify_extension")
            if extension_verifier is not None:
                budget.check("f_extension_precheck")
                extension_problems = extension_verifier(budget)
                budget.check("f_extension_precheck_complete")
                if extension_problems:
                    res.update(replay_enabled=False, replay_passed=None,
                               skip_reason="extension_evidence_invalid",
                               extension_problems=extension_problems[:20])
                    stages_out.append(res)
                    evidence_all_ok = False
                    continue
            verification = verify_stage_local(
                stage_dir, plan, registration, decoder,
                production_decoder=loss_fn, counters=f_verify_counters,
                budget=budget, expected_identity=expected_identity)
        except BudgetExceeded as exc:
            # P1 (round-5): the saved-quantity recompute is deadline-guarded
            # too; a deadline crossed inside it refuses the next decode and
            # stops this AND every later stage — the F report (with the
            # executed counters) still closes honestly below.
            res.update({"replay_enabled": False, "replay_passed": None,
                        "skip_reason": "stage_deadline_stop",
                        "deadline_stop": str(exc),
                        "verification_counters": dict(f_verify_counters)})
            stages_out.append(res)
            evidence_all_ok = False
            break
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
        # P1 (round-9): every checkpoint read inside the identity helper is
        # a unit; a read returning past the deadline refuses the next one and
        # closes this AND every later stage as a deadline stop (no build).
        index_doc, index_problems = _read_json_object(
            stage_dir / "artifacts_index.json")
        if index_doc is None:
            identity_problems = [
                f"artifacts_index_invalid:{index_problems[:2]}"]
        else:
            try:
                identity_problems = _f_identity_problems(
                    stage_dir, plan, decoder, index_doc, budget=budget,
                    expected_identity=expected_identity, counters=f_verify_counters)
            except BudgetExceeded as exc:
                res.update({"replay_enabled": False, "replay_passed": None,
                            "skip_reason": "stage_deadline_stop",
                            "deadline_stop": str(exc)})
                stages_out.append(res)
                evidence_all_ok = False
                break
        res["verification_counters"] = dict(f_verify_counters)
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
                    stage_dir / f"{role}.pt", model, conditioner,
                    budget=budget, counters=f_counters)
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
                # P1 (round-8): EVERY read is a unit — a read returning past
                # the work deadline refuses the next raw/condition read
                batch_arrays = {}
                for b in range(len(window_ids)):
                    budget.check(f"f_probe_{role}_input_w{b}_raw")
                    recs = read_npz_records(
                        stage_dir / "probe_inputs.npz", RECORD_SHAPES,
                        {"raw_x0_pred": f"w{b}__x0"})
                    batch_arrays.setdefault("x0", []).append(recs["raw_x0_pred"])
                    for key in COND_KEYS:
                        budget.check(f"f_probe_{role}_input_w{b}_{key}")
                        npz_key = f"w{b}__{key}"
                        try:
                            with np.load(stage_dir / "probe_inputs.npz") as data:
                                if npz_key not in data:
                                    raise FitDiagnosticError(
                                        f"condition_missing:w{b}:{key}")
                                arr = np.asarray(data[npz_key])
                        except FitDiagnosticError:
                            raise
                        except (ValueError, OSError, TypeError, EOFError,
                                zipfile.BadZipFile, zlib.error) as exc:
                            # P2 (round-14): a CRC-corrupted member is a
                            # structured archive refusal too.
                            raise FitDiagnosticError(
                                f"condition_archive_unreadable:w{b}:{key}:"
                                f"{type(exc).__name__}") from exc
                        # P2 (round-13): the frozen condition contract is
                        # checked BEFORE the array reaches torch/model code.
                        if arr.dtype != np.dtype(np.float32):
                            raise FitDiagnosticError(
                                f"condition_dtype:w{b}:{key}:{arr.dtype}")
                        if arr.shape != CONDITION_SHAPES[key]:
                            raise FitDiagnosticError(
                                f"condition_shape:w{b}:{key}:"
                                f"{tuple(arr.shape)}:"
                                f"want{CONDITION_SHAPES[key]}")
                        if not np.isfinite(arr).all():
                            raise FitDiagnosticError(
                                f"condition_nonfinite:w{b}:{key}")
                        batch_arrays.setdefault(key, []).append(arr)
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
                    model, conditioner, batch, plan, loss_fn, f_counters,
                    budget=budget, runtime_guard=runtime_guard)
                budget.check(f"f_probe_{role}_fresh_complete")
                runtime_guard.check_unit(f"f_probe_{role}_fresh_complete")
                idx = 0 if role == "initial" else plan.n_probes - 1
                saved_entry = _saved_probe_entry(stage_dir, idx, want_step)
                if saved_entry is None:
                    replay_problems.append(f"saved_probe_missing:{role}")
                    continue
                # P2 (round-9): a saved window list longer/shorter than the
                # registered batch is a structural replay failure, never an
                # IndexError that escapes the F report
                saved_windows = saved_entry["windows"]
                if len(saved_windows) != len(window_ids):
                    replay_problems.append(
                        f"saved_probe_windows_count:{role}:"
                        f"{len(saved_windows)}:want{len(window_ids)}")
                    continue
                for b, wout in enumerate(saved_windows):
                    if not isinstance(wout, dict):
                        replay_problems.append(
                            f"saved_probe_window_not_a_mapping:{role}:w{b}")
                        continue
                    raw_f = np.ascontiguousarray(x0_fresh[b])
                    problems += [f"fresh_x0_invalid:{p}"
                                 for p in decoder.verify_x0(raw_f)]
                    # P1 (round-8): the per-window endpoint artifact read is
                    # its own unit — a reduction returning past the deadline
                    # refuses the next window's read
                    budget.check(f"f_saved_probe_read_{role}_w{b}")
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
                    # P1 (round-6): the saved-quantity read AND the bitwise
                    # comparisons above are I/O work — a deadline crossed
                    # during them must refuse the teacher decode that
                    # follows (and be caught below as a replay failure).
                    budget.check(f"f_teacher_recheck_{role}_w{b}")
                    teacher, teacher_problems = _teacher_reference_checked(
                        stage_dir, decoder, b, f_counters, budget=budget)
                    replay_problems += teacher_problems[:3]
                    # P1 (round-4): the fresh metric decode is a numeric unit
                    # of its own — the work deadline is re-checked at this
                    # boundary, so a deadline crossed during the teacher
                    # re-verification refuses the decode (the BudgetExceeded
                    # is caught below and the stage closes as incomplete).
                    budget.check(f"f_fresh_metrics_{role}_w{b}")
                    f_counters["metric_decode_calls"] += 1
                    fresh_md = decoder.decode(raw_f)
                    # P1 (round-6): the fresh decode is a numeric unit too —
                    # a deadline crossed during it refuses the metric
                    # reduction that follows.
                    budget.check(f"f_fresh_metrics_reduce_{role}_w{b}")
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
    # Optional, versioned diagnostics attach their CPU/file-only comparison
    # BEFORE F's seal. A slow/failed comparison cannot buy a late fit claim.
    if finalize_evidence is not None:
        try:
            budget.check("f_extension_evidence")
            finalize_evidence(final)
            budget.check("f_extension_evidence_complete")
        except (FitDiagnosticError, OSError, ValueError, TypeError, KeyError) as exc:
            final.update(complete_evidence=False, mean_fit_passed=None,
                         verdict="incomplete", extension_error=str(exc)[:500])
    path = out_dir / "f_report.json"
    # P1 (round-4): F covers its OWN final report re-read with a return-
    # coverage verification — the outer acceptance budget alone must not be
    # the only guard against a slow tail read; the necessary read happens
    # BEFORE the coverage sample and no I/O follows the passing sample.
    seal = _write_summary_sealed(path, final, budget,
                                  withdraw=dict(F_WITHDRAW_ON_EXCEED),
                                  clock=budget.clock, **budget.closure_write_options())
    doc = json.loads(path.read_text())
    cov = verify_return_coverage(path, budget,
                                  withdraw=dict(F_WITHDRAW_ON_EXCEED),
                                  clock=budget.clock, **budget.closure_write_options())
    doc, seal_view = _coverage_return_view(cov)
    return {"f_report": str(path), "seal": seal_view,
            "return_coverage": cov, "seal_loop": seal, "summary_doc": doc}


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
                                  clock=budget.clock, **budget.closure_write_options())
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


def _observe_returned_report(path: Path, budget: StageBudget, withdraw: dict) -> dict:
    """Recheck coverage *after* the stage function actually returned.

    Its in-function observation cannot cover execution between that sample
    and the caller regaining control.  The replacement observation remains
    bound to the same sealed bytes; on failure the existing withdrawal path
    makes the report non-consumable.
    """
    return verify_return_coverage(path, budget, withdraw=dict(withdraw),
                                  clock=budget.clock)


def _mini_stage(tmp: Path, stage: str, *, n_windows: int = 6, steps: int = 2,
                probe_every: int = 1, seed: int = 20261107, d_model: int = 16,
                lr: float = 0.02, world: SyntheticWorld | None = None,
                finish: bool = False, artifact_space=None):
    """One miniature fit stage for the counterexamples."""
    world = world or make_synthetic_world(seed=seed, n_windows=n_windows,
                                          d_model=d_model)
    plan = FitPlan(stage, steps=steps, probe_every=probe_every, synthetic=True,
                   d_model=d_model, n_layers=1, n_heads=4, ffn_mult=2, lr=lr)
    sdir = Path(tmp) / stage.lower()
    budget = StageBudget(stage=stage, wall_cap_s=600.0, reserve_s=10.0, artifact_space=artifact_space)
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
    budget.check("acceptance_world_entry")
    world = make_synthetic_world(seed=seed, n_windows=6, d_model=16)
    budget.check("acceptance_world_constructed")
    facts["frozen_plan_defaults_problems"] = frozen_plan_defaults_problems()
    auth = {"includes": ["D6"]}
    d1_plan = FitPlan("D1", steps=d1_steps, probe_every=d1_probe_every,
                      synthetic=True, d_model=16, n_layers=1, n_heads=4,
                      ffn_mult=2, lr=d1_lr)
    budget.check("acceptance_d1_entry")
    d1_driver, d1_summary, d1_fin = _run_fit_stage(
        world, d1_plan, out_dir / "d1", budget, auth)
    d1_returned = budget.clock() - budget.t0
    d1_cov = _observe_returned_report(out_dir / "d1" / "stage_report.json",
                                      budget, I_WITHDRAW_ON_EXCEED)
    d1_fin["return_coverage"] = d1_cov
    d1_fin["summary_doc"] = d1_cov["final_doc"]
    d1_doc = d1_fin["summary_doc"]
    facts["d1"] = {
        "local_complete_evidence": d1_doc.get("local_complete_evidence"),
        "local_mean_fit_passed": d1_doc.get("local_mean_fit_passed"),
        "local_conclusion": d1_doc.get("local_conclusion"),
        "seal": (d1_doc.get("seal") or {}).get("status"),
        "return_coverage": d1_fin["return_coverage"].get("covered"),
        "stage_returned_elapsed_s": d1_returned,
        "call_matrix_problems": d1_summary["call_matrix"]["problems"],
        "report_problems": local_report_problems(
            {k: v for k, v in d1_fin["summary_doc"].items() if k != "seal"})}
    d6_summary = None
    d6_plan = None
    if (facts["d1"]["local_complete_evidence"] is True
            and facts["d1"]["local_mean_fit_passed"] is True
            and facts["d1"]["return_coverage"] is True):
        handoff = resolve_handoff(out_dir / "d1", d1_plan, world.decoder,
                                  world.registration_for("D1"), auth, budget,
                                  expected_identity=d1_driver.identity,
                                  production_decoder=world.loss_fn)
        facts["d1_handoff"] = {"handoff_ready": handoff["handoff_ready"],
                               "problems": handoff["problems"][:5]}
        if handoff["handoff_ready"]:
            d6_plan = FitPlan("D6", steps=d6_steps, probe_every=d6_probe_every,
                              synthetic=True, d_model=16, n_layers=1, n_heads=4,
                              ffn_mult=2, lr=d6_lr)
            budget.check("acceptance_d6_entry")
            d6_driver, d6_summary, d6_fin = _run_fit_stage(
                world, d6_plan, out_dir / "d6", budget, auth)
            d6_returned = budget.clock() - budget.t0
            d6_cov = _observe_returned_report(out_dir / "d6" / "stage_report.json",
                                              budget, I_WITHDRAW_ON_EXCEED)
            d6_fin["return_coverage"] = d6_cov
            d6_fin["summary_doc"] = d6_cov["final_doc"]
            d6_doc = d6_fin["summary_doc"]
            facts["d6"] = {
                "local_complete_evidence": d6_doc.get("local_complete_evidence"),
                "local_mean_fit_passed": d6_doc.get("local_mean_fit_passed"),
                "local_conclusion": d6_doc.get("local_conclusion"),
                "seal": (d6_doc.get("seal") or {}).get("status"),
                "return_coverage": d6_cov.get("covered"),
                "stage_returned_elapsed_s": d6_returned,
                "call_matrix_problems": d6_summary["call_matrix"]["problems"]}
    else:
        facts["d1_handoff"] = {
            "handoff_ready": False,
            "skipped_reason": "d1_incomplete_or_mean_fit_gate_not_passed",
            "note": "D6 is never appended as a retry (plan §7.2)"}
    # F verifies whatever stages completed locally; a complete negative
    # result is verifiable too (plan §7.3).
    f_specs = []
    if facts["d1"]["local_complete_evidence"]:
        f_specs.append({"stage_dir": out_dir / "d1", "plan": d1_plan,
                        "registration": world.registration_for("D1"),
                        "decoder": world.decoder, "loss_fn": world.loss_fn,
                        "identity": d1_driver.identity})
    if (d6_summary is not None
            and facts["d6"]["local_complete_evidence"] is True
            and facts["d6"]["return_coverage"] is True):
        f_specs.append({"stage_dir": out_dir / "d6", "plan": d6_plan,
                        "registration": world.registration_for("D6"),
                        "decoder": world.decoder, "loss_fn": world.loss_fn,
                        "identity": d6_driver.identity})
    if f_specs:
        budget.check("acceptance_f_entry")
        f_budget = budget_for_stage("F", t0=budget.clock())
        f_out = run_f_verification(
            stage_specs=f_specs, out_dir=out_dir / "f",
            budget=f_budget,
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world, d1_plan),
            authorization=auth)
        f_returned = f_budget.clock() - f_budget.t0
        f_cov = _observe_returned_report(out_dir / "f" / "f_report.json",
                                         f_budget, F_WITHDRAW_ON_EXCEED)
        f_out["return_coverage"] = f_cov
        f_out["summary_doc"] = f_cov["final_doc"]
        facts["f"] = {k: f_out["summary_doc"].get(k)
                      for k in ("complete_evidence", "replay_passed",
                                "mean_fit_passed", "verdict")}
        facts["f"]["return_coverage"] = f_out["return_coverage"]["covered"]
        facts["f"]["stage_returned_elapsed_s"] = f_returned
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
    "f_saved_recompute_deadline_stop",
    "f_second_stage_deadline_stop_reports",
    "d_driver_recompute_deadline_stop",
    "handoff_recompute_deadline_refused",
    "f_teacher_recheck_deadline_stop",
    "f_fresh_metrics_reduce_deadline_stop",
    "withdrawal_view_matches_disk_multi_round",
    "withdrawal_view_matches_disk_exhausted",
    "f_fresh_probe_deadline_stop",
    "f_teacher_decode_deadline_stop",
    "saved_teacher_read_deadline_stop",
    "handoff_checkpoint_read_deadline_stop",
    "handoff_final_claim_deadline_refused",
    "f_fresh_backbone_deadline_stop",
    "f_strict_load_deadline_stop",
    "f_input_rebuild_deadline_stop",
    "f_saved_probe_read_deadline_stop",
    "bad_containers_report_incomplete_f",
    "f_construct_after_deadline_refused",
    "f_runtime_drift_before_fresh_refused",
    "proof_not_a_mapping_survives_f",
    "checkpoint_config_binding_rejected",
    "checkpoint_identity_binding_rejected",
    "teacher_window_units_deadline_stop",
    "probe_window_units_deadline_stop",
    "training_micro_internal_deadline_stop",
    "f_identity_checkpoint_read_deadline_stop",
    "f_completed_conditioner_counter_kept",
    "nested_probe_schema_containers_report_incomplete",
    "loss_weight_binding_rejected",
    "stage_identity_external_anchor",
    "post_backbone_validation_tail_stop",
    "refused_backbone_not_counted",
    "bad_metrics_and_root_stats_report_incomplete",
    "loss_internal_fk_units_deadline_stop",
    "live_loss_weight_drift_refused",
    "stage_identity_full_external_anchor",
    "bad_checkpoint_config_containers_report_incomplete",
    "executed_failing_backbone_counted",
    "typed_counts_and_coordinates_rejected",
    "effective_optimizer_binding_rejected",
    "conditioner_and_clip_pre_boundaries_stop",
    "f_entered_failed_forwards_counted",
    "huge_config_scalars_report_incomplete",
    "bad_npz_dtype_reports_incomplete",
    "optimizer_extended_binding_rejected",
    "binding_read_boundaries_stop",
    "checkpoint_tensor_facts_bound",
    "runtime_drift_between_numeric_units_stop",
    "object_npz_reports_incomplete",
    "condition_contract_reports_incomplete",
    "optimizer_arithmetic_flags_rejected",
    "runtime_check_time_charged",
    "f_runtime_check_time_charged",
    "checkpoint_architecture_contract_bound",
    "conditioner_and_fk_precision_admission",
    "f_fresh_fk_precision_admission",
    "crc_corrupt_npz_reports_incomplete",
    "optimizer_initial_state_bound",
    "decode_fk_runtime_read_time_charged",
    "f_fresh_runtime_read_time_charged",
    "state_contract_construction_guarded",
    "mixed_checkpoint_keys_report_incomplete",
    "trainable_parameter_roster_bound",
    "pre_backward_runtime_read_time_charged",
    "teacher_reference_precision_admission",
    "bad_report_identity_no_reference_construction",
    "sparse_checkpoint_layout_report_incomplete",
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
                          "loss_fn": world_f.loss_fn,
                          "identity": _synthetic_stage_identity(plan_f, world_f.registration_for("D1"))}],
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
                          "loss_fn": world_b.loss_fn,
                          "identity": _synthetic_stage_identity(plan_b, world_b.registration_for("D1"))}],
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
        # the roster binding now covers every model+conditioner parameter, so
        # rebuild the native optimizer to include the deliberately bad tensor;
        # the refusal under test is the PREFLIGHT dtype contract, not the
        # roster check
        trainer_dd.opt = torch.optim.AdamW(
            [*trainer_dd.model.parameters(),
             *trainer_dd.conditioner.parameters()],
            lr=plan_dd.lr, weight_decay=plan_dd.weight_decay)
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
                                  expected_identity=_synthetic_stage_identity(
                                      plan_h, world_h.registration_for("D1")),
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
                          "loss_fn": world_p2.loss_fn,
                          "identity": _synthetic_stage_identity(plan_p2, world_p2.registration_for("D1"))}],
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
                          "loss_fn": world_t.loss_fn,
                          "identity": _synthetic_stage_identity(plan_t, world_t.registration_for("D1"))}],
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
                                  expected_identity=_synthetic_stage_identity(
                                      plan_r2, world_r2.registration_for("D1")),
                                  production_decoder=world_r2.loss_fn)
        proof_bound = (handoff["handoff_ready"] is False
                       and any("return_observation_summary_sha_mismatch" in p
                               for p in handoff["problems"]))
        (sdir_r2 / "return_observation.json").unlink()
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_r2, "plan": plan_r2,
                          "registration": world_r2.registration_for("D1"),
                          "decoder": world_r2.decoder,
                          "loss_fn": world_r2.loss_fn,
                          "identity": _synthetic_stage_identity(plan_r2, world_r2.registration_for("D1"))}],
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
                          "loss_fn": world_c4.loss_fn,
                          "identity": _synthetic_stage_identity(plan_c4, world_c4.registration_for("D1"))}],
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

        class _DriftAt(SyntheticRuntimeGuard):
            def __init__(self, fail_point: str) -> None:
                super().__init__()
                self._fail_point = fail_point
                self.calls = 0

            def check_unit(self, point: str) -> None:
                self.calls += 1
                if point == self._fail_point:
                    raise FitDiagnosticError(
                        f"runtime_drift:{point}:tf32_enabled_simulated")

        # fail exactly at the post-backward unit of micro 1: the first micro's
        # backward has executed (backward_calls=1) but the step must not
        # reach clip/optimizer (round-13 added conditioner/backbone/loss
        # boundary runtime checks, so a positional count is no longer stable)
        guard_r5 = _DriftAt("step_1_micro_1_backward")
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
            """Pass the first two budget checks (stage entry, first saved-
            quantity check), then jump past the work deadline — the saved
            recompute, the load and the input reads land beyond it and no
            fresh forward may start."""

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
                          "loss_fn": world_f6.loss_fn,
                          "identity": _synthetic_stage_identity(plan_f6, world_f6.registration_for("D1"))}],
            out_dir=tmp / "fdeadline_f", budget=budget_f6,
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_f6, plan_f6),
            authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        fc6 = s0.get("fresh_counters") or {}
        stop6 = str(s0.get("deadline_stop") or "")
        caught = (fc6.get("conditioner_forward_calls", 0) == 0
                  and fc6.get("backbone_forward_calls", 0) == 0
                  and s0.get("skip_reason") == "stage_deadline_stop"
                  and "deadline" in stop6
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("f_forward_after_deadline_refused", caught,
               f"saved recompute crossed the deadline at {stop6[:40]}; "
               f"fresh forwards={fc6.get('conditioner_forward_calls', 0)}")

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
                                  expected_identity=_synthetic_stage_identity(
                                      plan_p3, world_p3.registration_for("D1")),
                                  production_decoder=world_p3.loss_fn)
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_p3, "plan": plan_p3,
                          "registration": world_p3.registration_for("D1"),
                          "decoder": world_p3.decoder,
                          "loss_fn": world_p3.loss_fn,
                          "identity": _synthetic_stage_identity(plan_p3, world_p3.registration_for("D1"))}],
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
                                  expected_identity=_synthetic_stage_identity(
                                      plan_c4, world_c4.registration_for("D1")),
                                  production_decoder=world_c4.loss_fn)
        caught = (control_ok
                  and doc_c3["seal"]["status"] == "budget_exceeded_after_seal"
                  and on_disk["seal"]["status"] == "budget_exceeded_after_seal"
                  and fin["return_coverage"]["covered"] is False
                  and fin["seal"]["status"] == "budget_exceeded_after_seal"
                  and (doc_c3.get("budget") or {}).get(
                      "measured_wall_s_at_summary")
                  == (on_disk.get("budget") or {}).get(
                      "measured_wall_s_at_summary")
                  == fin["return_coverage"]["registered_measured_s"]
                  and handoff["handoff_ready"] is False)
        record("candidate_write_after_covered_refused", caught,
               "slow candidate write lands inside the checked window; "
               f"seal={doc_c3['seal']['status']} handoff="
               f"{handoff['handoff_ready']} cost="
               f"{(doc_c3.get('budget') or {}).get('measured_wall_s_at_summary')}")

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
                              "loss_fn": world_f7.loss_fn,
                              "identity": _synthetic_stage_identity(plan_f7, world_f7.registration_for("D1"))}],
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
                  and fin_r["return_coverage"]["covered"] is False
                  and fin_r["seal"]["status"] == "budget_exceeded_after_seal"
                  and (fin_r["summary_doc"].get("budget") or {}).get(
                      "measured_wall_s_at_summary")
                  == (on_disk_r.get("budget") or {}).get(
                      "measured_wall_s_at_summary")
                  == fin_r["return_coverage"]["registered_measured_s"])
        record("stage_report_reread_inside_coverage_window", caught,
               "slow post-seal report read lands inside the checked window; "
               f"seal={fin_r['summary_doc']['seal']['status']} "
               f"covered={fin_r['return_coverage']['covered']} cost="
               f"{(fin_r['summary_doc'].get('budget') or {}).get('measured_wall_s_at_summary')}")

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
                              "loss_fn": world_f8.loss_fn,
                              "identity": _synthetic_stage_identity(plan_f8, world_f8.registration_for("D1"))}],
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

        # --- P1(r5): the saved-quantity recompute is deadline-guarded ---------
        # (the reviewer's round-5 attack: a clock jump at the FIRST decode
        # return still let the remaining canonical decodes, production
        # decodes and metric reductions run past the work deadline)
        world_f9, plan_f9, _, _, sdir_f9, _, _ = _mini_stage(
            tmp / "fsaveddl", "D1", n_windows=1, steps=1, seed=seed + 43,
            finish=True)
        saved_holder = {"t": 0.0}
        saved_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                                   t0=0.0, clock=lambda: saved_holder["t"])
        dec_calls = {"n": 0}
        real_decode = type(world_f9.decoder).decode

        def _jumping_decode(self, *a, **kw):
            out = real_decode(self, *a, **kw)
            dec_calls["n"] += 1
            if dec_calls["n"] == 1:
                # first decode returns inside the cap but past the WORK
                # deadline: every following numeric unit must be refused
                saved_holder["t"] = saved_budget.work_deadline + 5.0
            return out

        mod.MetricDecoder.decode = _jumping_decode
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_f9, "plan": plan_f9,
                              "registration": world_f9.registration_for("D1"),
                              "decoder": world_f9.decoder,
                              "loss_fn": world_f9.loss_fn,
                              "identity": _synthetic_stage_identity(plan_f9, world_f9.registration_for("D1"))}],
                out_dir=tmp / "fsaveddl_f", budget=saved_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_f9, plan_f9),
                authorization=None)
        finally:
            mod.MetricDecoder.decode = real_decode
        s0 = f_out["summary_doc"]["stages"][0]
        stop9 = str(s0.get("deadline_stop") or "")
        caught = (dec_calls["n"] == 1
                  and "saved_teacher_prod_decode" in stop9
                  and s0.get("skip_reason") == "stage_deadline_stop"
                  and f_out["summary_doc"]["verdict"] == "incomplete"
                  and (Path(tmp) / "fsaveddl_f" / "f_report.json").is_file()
                  and f_out["return_coverage"]["covered"] is True)
        record("f_saved_recompute_deadline_stop", caught,
               "one canonical decode ran; the production decode after the "
               f"deadline jump was refused at {stop9[:48]} "
               f"(decodes={dec_calls['n']})")

        # --- P2(r5): a deadline crossed in D1 stops D6; the F report and the
        # return proof still land (the uncaught second-stage entry check used
        # to leave no f_report.json at all) ------------------------------------
        world_g1, plan_g1, _, _, sdir_g1, _, _ = _mini_stage(
            tmp / "fstopd1", "D1", n_windows=1, steps=1, seed=seed + 44,
            finish=True)
        d6_plan_mini = FitPlan("D6", steps=1, probe_every=1, synthetic=True,
                               d_model=16, n_layers=1, n_heads=4, ffn_mult=2,
                               lr=0.02)
        stop_holder = {"t": 0.0}
        stop_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                                  t0=0.0, clock=lambda: stop_holder["t"])
        stop_calls = {"n": 0}
        real_tchk_g = mod._teacher_reference_checked

        def _jump_at_first_teacher(*a, **kw):
            out = real_tchk_g(*a, **kw)
            stop_calls["n"] += 1
            if stop_calls["n"] == 1:
                stop_holder["t"] = stop_budget.work_deadline + 5.0
            return out

        mod._teacher_reference_checked = _jump_at_first_teacher
        try:
            f_out = run_f_verification(
                stage_specs=[
                    {"stage_dir": sdir_g1, "plan": plan_g1,
                     "registration": world_g1.registration_for("D1"),
                     "decoder": world_g1.decoder,
                     "loss_fn": world_g1.loss_fn,
                     "identity": _synthetic_stage_identity(plan_g1, world_g1.registration_for("D1"))},
                    {"stage_dir": sdir_g1, "plan": d6_plan_mini,
                     "registration": world_g1.registration_for("D1"),
                     "decoder": world_g1.decoder,
                     "loss_fn": world_g1.loss_fn,
                     "identity": _synthetic_stage_identity(d6_plan_mini, world_g1.registration_for("D1"))},
                ],
                out_dir=tmp / "fstopd1_f", budget=stop_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_g1, plan_g1),
                authorization=None)
        finally:
            mod._teacher_reference_checked = real_tchk_g
        doc_g = f_out["summary_doc"]
        stages_g = doc_g.get("stages") or []
        f_dir_g = Path(tmp) / "fstopd1_f"
        caught = ((f_dir_g / "f_report.json").is_file()
                  and (f_dir_g / "return_observation.json").is_file()
                  and len(stages_g) == 2
                  and stages_g[0].get("replay_passed") is False
                  and stages_g[1].get("skip_reason") == "stage_deadline_stop"
                  and doc_g.get("claim_deadline_exceeded") is True
                  and doc_g.get("verdict") == "incomplete"
                  and doc_g.get("complete_evidence") is False)
        record("f_second_stage_deadline_stop_reports", caught,
               "D1 crossed the deadline; D6 did not start and the F report + "
               f"return proof still landed (stages={len(stages_g)}, "
               f"verdict={doc_g.get('verdict')})")

        # --- P1(r6): the D driver's local recompute runs under the stage
        # budget — a deadline crossed inside it refuses the next decode -------
        world_h1, plan_h1, _, _, _, _, _ = _mini_stage(
            tmp / "ddlverify", "D1", n_windows=1, steps=1, seed=seed + 45)
        ddir = tmp / "ddlverify2"
        d_holder = {"t": 0.0}
        d_budget = StageBudget(stage="D1", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: d_holder["t"])
        d_view = world_h1.view_for("D1")
        # the trainer's own save deadline compares against the real clock;
        # with the controlled (fake-epoch) budget it is disabled — the
        # driver's per-unit budget checks remain the guard under test
        trainer_h1 = world_h1.build_trainer(
            plan_h1, ddir, None, train_ds=d_view["ds"])
        driver_h1 = FitDriver(
            trainer=trainer_h1, plan=plan_h1,
            registration=d_view["registration"], loss_fn=world_h1.loss_fn,
            decoder=world_h1.decoder, probe_batch=d_view["probe_batch"],
            budget=d_budget, runtime_guard=SyntheticRuntimeGuard(),
            identity={"synthetic": True}, stage_dir=ddir,
            training_gen=torch.Generator().manual_seed(plan_h1.seed),
            dataset_window_keys=d_view["ds"].window_keys,
            authorization_includes_d6=True)
        real_decode_d = mod.MetricDecoder.decode
        d_calls = {"n": 0}

        def _jump_in_verify(self, *a, **kw):
            out = real_decode_d(self, *a, **kw)
            if sys._getframe(1).f_code.co_name == "verify_stage_local":
                d_calls["n"] += 1
                if d_calls["n"] == 1:
                    d_holder["t"] = d_budget.work_deadline + 5.0
            return out

        mod.MetricDecoder.decode = _jump_in_verify
        try:
            summary_h1 = driver_h1.run()
        finally:
            mod.MetricDecoder.decode = real_decode_d
        failure_h1 = summary_h1.get("failure") or {}
        caught = (failure_h1.get("type") == "BudgetExceeded"
                  and failure_h1.get("phase") == "local_verification"
                  and d_calls["n"] == 1
                  and (summary_h1.get("verification_counters") or {}).get(
                      "metric_decode_calls") == 1
                  and summary_h1.get("local_complete_evidence") is False
                  and summary_h1.get("local_conclusion") == "incomplete"
                  and (summary_h1.get("budget") or {}).get("budget_exceeded")
                  is True)
        record("d_driver_recompute_deadline_stop", caught,
               "driver recompute crossed the deadline after one decode; the "
               f"stage closed incomplete (decodes={d_calls['n']}, "
               f"failure={failure_h1.get('message', '')[:40]})")

        # --- P1(r6): the handoff recompute runs under the D6 budget and a
        # crossed deadline refuses the handoff --------------------------------
        world_h2, plan_h2, _, _, sdir_h2, _, _ = _mini_stage(
            tmp / "hdlverify", "D1", n_windows=1, steps=1, seed=seed + 46,
            finish=True)
        h_holder = {"t": 0.0}
        h_budget = StageBudget(stage="D6", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: h_holder["t"])
        real_decode_hd = mod.MetricDecoder.decode
        h_calls = {"n": 0}

        def _jump_in_handoff(self, *a, **kw):
            out = real_decode_hd(self, *a, **kw)
            if sys._getframe(1).f_code.co_name == "verify_stage_local":
                h_calls["n"] += 1
                if h_calls["n"] == 1:
                    h_holder["t"] = h_budget.work_deadline + 5.0
            return out

        mod.MetricDecoder.decode = _jump_in_handoff
        try:
            handoff_h = resolve_handoff(
                sdir_h2, plan_h2, world_h2.decoder,
                world_h2.registration_for("D1"), {"includes": ["D6"]},
                h_budget,
                expected_identity=_synthetic_stage_identity(
                    plan_h2, world_h2.registration_for("D1")),
                production_decoder=world_h2.loss_fn)
        finally:
            mod.MetricDecoder.decode = real_decode_hd
        caught = (handoff_h["handoff_ready"] is False
                  and any("handoff_deadline_exceeded" in p
                          for p in handoff_h["problems"])
                  and h_budget.exceeded is True
                  and h_calls["n"] == 1)
        record("handoff_recompute_deadline_refused", caught,
               "handoff recompute crossed the deadline after one decode; "
               f"handoff refused (decodes={h_calls['n']}, "
               f"problems={handoff_h['problems'][:1]})")

        # --- P1(r6): a slow saved-quantity read refuses the teacher decode ----
        world_h3, plan_h3, _, _, sdir_h3, _, _ = _mini_stage(
            tmp / "fteachdl", "D1", n_windows=1, steps=1, seed=seed + 47,
            finish=True)
        t_holder = {"t": 0.0}
        t_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: t_holder["t"])
        tchk_seen = {"n": 0}
        real_saved_h = mod._saved_probe_records
        real_tchk_h = mod._teacher_reference_checked

        def _slow_saved(*a, **kw):
            out = real_saved_h(*a, **kw)
            t_holder["t"] = t_budget.work_deadline + 5.0
            return out

        def _counting_tchk(*a, **kw):
            tchk_seen["n"] += 1
            return real_tchk_h(*a, **kw)

        mod._saved_probe_records = _slow_saved
        mod._teacher_reference_checked = _counting_tchk
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_h3, "plan": plan_h3,
                              "registration": world_h3.registration_for("D1"),
                              "decoder": world_h3.decoder,
                              "loss_fn": world_h3.loss_fn,
                              "identity": _synthetic_stage_identity(plan_h3, world_h3.registration_for("D1"))}],
                out_dir=tmp / "fteachdl_f", budget=t_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_h3, plan_h3),
                authorization=None)
        finally:
            mod._saved_probe_records = real_saved_h
            mod._teacher_reference_checked = real_tchk_h
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (tchk_seen["n"] == 0
                  and any("f_teacher_recheck" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["verdict"] == "incomplete")
        record("f_teacher_recheck_deadline_stop", caught,
               "slow saved read crossed the deadline; the teacher decode "
               f"never started (tchk={tchk_seen['n']}, "
               f"problems={s0.get('replay_problems', [])[:1]})")

        # --- P1(r6): a slow fresh decode refuses the metric reduction ---------
        world_h4, plan_h4, _, _, sdir_h4, _, _ = _mini_stage(
            tmp / "freducedl", "D1", n_windows=1, steps=1, seed=seed + 48,
            finish=True)
        r_holder = {"t": 0.0}
        r_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: r_holder["t"])
        real_decode_r = mod.MetricDecoder.decode

        def _jump_after_fresh_decode(self, *a, **kw):
            out = real_decode_r(self, *a, **kw)
            if sys._getframe(1).f_code.co_name == "run_f_verification":
                r_holder["t"] = r_budget.work_deadline + 5.0
            return out

        mod.MetricDecoder.decode = _jump_after_fresh_decode
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_h4, "plan": plan_h4,
                              "registration": world_h4.registration_for("D1"),
                              "decoder": world_h4.decoder,
                              "loss_fn": world_h4.loss_fn,
                              "identity": _synthetic_stage_identity(plan_h4, world_h4.registration_for("D1"))}],
                out_dir=tmp / "freducedl_f", budget=r_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_h4, plan_h4),
                authorization=None)
        finally:
            mod.MetricDecoder.decode = real_decode_r
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (any("f_fresh_metrics_reduce" in p
                      for p in s0.get("replay_problems", []))
                  and s0.get("mean_fit_passed") is None
                  and f_out["summary_doc"]["verdict"] == "incomplete")
        record("f_fresh_metrics_reduce_deadline_stop", caught,
               "slow fresh decode crossed the deadline; the metric reduction "
               f"was refused (problems={s0.get('replay_problems', [])[:1]})")

        # --- P2(r6): multi-round and exhausted withdrawals return the FINAL
        # on-disk document — no reconstructed withdrawal semantics -------------
        import ireset.eval.w1_n0_coordinate_flags as _n0mod
        real_fsync_n0 = _n0mod._write_json_fsync

        def _run_withdrawal_case(name: str, mutator) -> tuple[dict, dict]:
            world_v, plan_v, _, summary_v, sdir_v, _, _ = _mini_stage(
                tmp / name, "D1", n_windows=1, steps=1, seed=seed + 49)
            v_holder = {"t": 0.0}
            v_budget = StageBudget(stage="D1", wall_cap_s=100.0,
                                   reserve_s=10.0, t0=0.0,
                                   clock=lambda: v_holder["t"])
            real_read_v = Path.read_text
            state = {"t_prev": 0.0, "t_cur": 0.0}

            def _slow_read_v(path, *a, **kw):
                value = real_read_v(path, *a, **kw)
                if path.name == "stage_report.json" \
                        and sys._getframe(1).f_code.co_name == "finish_stage":
                    v_holder["t"] = 1000.0
                    state["t_cur"] = 1000.0
                return value

            def _slow_fsync(path, doc):
                if path.name == "return_observation.json" \
                        and sys._getframe(1).f_code.co_name \
                        == "verify_return_coverage":
                    t_next = mutator(state)
                    state["t_prev"], state["t_cur"] = state["t_cur"], t_next
                    v_holder["t"] = t_next
                return real_fsync_n0(path, doc)

            mod.Path.read_text = _slow_read_v
            _n0mod._write_json_fsync = _slow_fsync
            try:
                fin_v = finish_stage(sdir_v, summary_v, v_budget)
            finally:
                mod.Path.read_text = real_read_v
                _n0mod._write_json_fsync = real_fsync_n0
            on_disk_v = json.loads((sdir_v / "stage_report.json").read_text())
            return fin_v, on_disk_v

        fin_m, disk_m = _run_withdrawal_case(
            "withmulti", lambda state: 1008.0)
        # round 2 is forced by the slow proof write: the registered elapsed
        # stays the ENTRY value on disk and the return must match it
        caught = (fin_m["return_coverage"]["covered"] is False
                  and json.loads(json.dumps(fin_m["summary_doc"])) == disk_m
                  and disk_m["seal"]["return_coverage_elapsed_s"] == 1000.0
                  and disk_m["seal"]["measured_wall_s_at_summary"]
                  == fin_m["return_coverage"]["registered_measured_s"]
                  and disk_m["seal"]["measured_wall_s_at_summary"] >= 1027.0)
        record("withdrawal_view_matches_disk_multi_round", caught,
               "multi-round withdrawal: returned doc == disk doc, entry "
               f"elapsed={disk_m['seal']['return_coverage_elapsed_s']}, "
               f"measured={disk_m['seal']['measured_wall_s_at_summary']}")

        fin_x, disk_x = _run_withdrawal_case(
            "withexhaust",
            lambda state: state["t_cur"] + 2.0 * (state["t_cur"]
                                                  - state["t_prev"]) + 10.0)
        caught = (fin_x["return_coverage"]["covered"] is False
                  and fin_x["return_coverage"]["cost_coverage_proven"] is False
                  and json.loads(json.dumps(fin_x["summary_doc"])) == disk_x
                  and disk_x["seal"].get("cost_coverage_unproven") is True
                  and disk_x["seal"]["measured_wall_s_at_summary"]
                  == fin_x["return_coverage"]["registered_measured_s"])
        record("withdrawal_view_matches_disk_exhausted", caught,
               "exhausted withdrawal: returned doc == disk doc incl. "
               f"cost_coverage_unproven="
               f"{disk_x['seal'].get('cost_coverage_unproven')}")

        # --- P1(r7): the fresh probe's per-window production decode + FK are
        # guarded — a decode returning past the deadline refuses both ---------
        world_i1, plan_i1, _, _, sdir_i1, _, _ = _mini_stage(
            tmp / "ffreshdl", "D6", n_windows=6, steps=1, seed=seed + 50,
            finish=True)
        i_holder = {"t": 0.0}
        i_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: i_holder["t"])
        real_clip_i = type(world_i1.loss_fn).decode_clip
        real_site_i = type(world_i1.loss_fn.fk).site_world_positions
        clip_calls = {"n": 0}
        site_calls = {"n": 0}

        def _jump_clip(self, *a, **kw):
            out = real_clip_i(self, *a, **kw)
            if sys._getframe(1).f_code.co_name == "_fresh_probe":
                clip_calls["n"] += 1
                if clip_calls["n"] == 1:
                    i_holder["t"] = i_budget.work_deadline + 5.0
            return out

        def _count_site(self, *a, **kw):
            site_calls["n"] += 1
            return real_site_i(self, *a, **kw)

        type(world_i1.loss_fn).decode_clip = _jump_clip
        type(world_i1.loss_fn.fk).site_world_positions = _count_site
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_i1, "plan": plan_i1,
                              "registration": world_i1.registration_for("D6"),
                              "decoder": world_i1.decoder,
                              "loss_fn": world_i1.loss_fn,
                              "identity": _synthetic_stage_identity(plan_i1, world_i1.registration_for("D6"))}],
                out_dir=tmp / "ffreshdl_f", budget=i_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_i1, plan_i1),
                authorization=None)
        finally:
            type(world_i1.loss_fn).decode_clip = real_clip_i
            type(world_i1.loss_fn.fk).site_world_positions = real_site_i
        s0 = f_out["summary_doc"]["stages"][0]
        fc_i = s0.get("fresh_counters") or {}
        caught = (clip_calls["n"] == 1
                  and site_calls["n"] == 0
                  and fc_i.get("production_decode_calls") == 1
                  and fc_i.get("production_fk_batch_calls") == 0
                  and any("f_fresh_fk" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["verdict"] == "incomplete")
        record("f_fresh_probe_deadline_stop", caught,
               "first fresh production decode crossed the deadline; the next "
               f"decode/FK were refused (decodes={clip_calls['n']}, "
               f"fk={site_calls['n']})")

        # --- P1(r7): the teacher re-check's read→decode boundary is guarded ----
        world_i2, plan_i2, _, _, sdir_i2, _, _ = _mini_stage(
            tmp / "fteachdec", "D1", n_windows=1, steps=1, seed=seed + 51,
            finish=True)
        j_holder = {"t": 0.0}
        j_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: j_holder["t"])
        real_trec_i = mod._teacher_records
        real_decode_t = type(world_i2.decoder).decode
        tdec_calls = {"n": 0}

        def _slow_trec(*a, **kw):
            out = real_trec_i(*a, **kw)
            if sys._getframe(1).f_code.co_name == "_teacher_reference_checked":
                j_holder["t"] = j_budget.work_deadline + 5.0
            return out

        def _count_tdecode(self, *a, **kw):
            if sys._getframe(1).f_code.co_name == "_teacher_reference_checked":
                tdec_calls["n"] += 1
            return real_decode_t(self, *a, **kw)

        mod._teacher_records = _slow_trec
        type(world_i2.decoder).decode = _count_tdecode
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_i2, "plan": plan_i2,
                              "registration": world_i2.registration_for("D1"),
                              "decoder": world_i2.decoder,
                              "loss_fn": world_i2.loss_fn,
                              "identity": _synthetic_stage_identity(plan_i2, world_i2.registration_for("D1"))}],
                out_dir=tmp / "fteachdec_f", budget=j_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_i2, plan_i2),
                authorization=None)
        finally:
            mod._teacher_records = real_trec_i
            type(world_i2.decoder).decode = real_decode_t
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (tdec_calls["n"] == 0
                  and any("f_teacher_decode" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["verdict"] == "incomplete")
        record("f_teacher_decode_deadline_stop", caught,
               "teacher read crossed the deadline; the teacher decode never "
               f"started (teacher decodes={tdec_calls['n']})")

        # --- P1(r7): the saved-quantity per-window reads are guarded ----------
        world_i3, plan_i3, _, _, sdir_i3, _, _ = _mini_stage(
            tmp / "savedreads", "D6", n_windows=6, steps=1, seed=seed + 52,
            finish=True)
        k_holder = {"t": 0.0}
        k_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: k_holder["t"])
        real_npz_read = mod.read_npz_records
        read_calls = {"n": 0}

        def _slow_npz_read(path, *a, **kw):
            out = real_npz_read(path, *a, **kw)
            if sys._getframe(1).f_code.co_name == "verify_stage_local":
                read_calls["n"] += 1
                if read_calls["n"] == 1:
                    k_holder["t"] = k_budget.work_deadline + 5.0
            return out

        mod.read_npz_records = _slow_npz_read
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_i3, "plan": plan_i3,
                              "registration": world_i3.registration_for("D6"),
                              "decoder": world_i3.decoder,
                              "loss_fn": world_i3.loss_fn,
                              "identity": _synthetic_stage_identity(plan_i3, world_i3.registration_for("D6"))}],
                out_dir=tmp / "savedreads_f", budget=k_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_i3, plan_i3),
                authorization=None)
        finally:
            mod.read_npz_records = real_npz_read
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (read_calls["n"] == 1
                  and s0.get("skip_reason") == "stage_deadline_stop"
                  and "saved_teacher_read_w1" in str(s0.get("deadline_stop") or ""))
        record("saved_teacher_read_deadline_stop", caught,
               "first saved teacher read crossed the deadline; the second "
               f"read was refused (reads={read_calls['n']})")

        # --- P1(r7): handoff checkpoint reads and the final claim are guarded -
        world_i4, plan_i4, _, _, sdir_i4, _, _ = _mini_stage(
            tmp / "hdlckpt", "D1", n_windows=1, steps=1, seed=seed + 53,
            finish=True)
        plan_i4_windows = len(world_i4.registration_for("D1").window_ids)
        total_reductions = len(plan_i4.probe_steps) * plan_i4_windows
        l_holder = {"t": 0.0}
        l_budget = StageBudget(stage="D6", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: l_holder["t"])
        real_metrics_i = mod.compute_window_metrics
        real_load_i = mod.torch.load
        red_calls = {"n": 0}
        load_calls = {"n": 0}

        def _jump_metrics(*a, **kw):
            out = real_metrics_i(*a, **kw)
            if sys._getframe(1).f_code.co_name == "verify_stage_local":
                red_calls["n"] += 1
                if red_calls["n"] >= total_reductions:
                    l_holder["t"] = l_budget.work_deadline + 5.0
            return out

        def _count_ckpt_load(*a, **kw):
            if sys._getframe(1).f_code.co_name == "_checkpoint_binding_problems":
                load_calls["n"] += 1
            return real_load_i(*a, **kw)

        mod.compute_window_metrics = _jump_metrics
        mod.torch.load = _count_ckpt_load
        try:
            handoff_l = resolve_handoff(
                sdir_i4, plan_i4, world_i4.decoder,
                world_i4.registration_for("D1"), {"includes": ["D6"]},
                l_budget,
                expected_identity=_synthetic_stage_identity(
                    plan_i4, world_i4.registration_for("D1")),
                production_decoder=world_i4.loss_fn)
        finally:
            mod.compute_window_metrics = real_metrics_i
            mod.torch.load = real_load_i
        caught = (handoff_l["handoff_ready"] is False
                  and any("handoff_deadline_exceeded" in p
                          for p in handoff_l["problems"])
                  and load_calls["n"] == 0
                  and l_budget.exceeded is True)
        record("handoff_checkpoint_read_deadline_stop", caught,
               "last metric reduction crossed the deadline; the checkpoint "
               f"reads never started (loads={load_calls['n']})")

        # --- P1(r7): the handoff claim itself is re-checked after the loads ---
        world_i5, plan_i5, _, _, sdir_i5, _, _ = _mini_stage(
            tmp / "hdlclaim", "D1", n_windows=1, steps=1, seed=seed + 54,
            finish=True)
        m_holder = {"t": 0.0}
        m_budget = StageBudget(stage="D6", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: m_holder["t"])
        real_load_m = mod.torch.load
        loads_m = {"n": 0}

        def _jump_after_last_load(*a, **kw):
            out = real_load_m(*a, **kw)
            if sys._getframe(1).f_code.co_name == "_checkpoint_binding_problems":
                loads_m["n"] += 1
                if loads_m["n"] >= 2:
                    m_holder["t"] = m_budget.work_deadline + 5.0
            return out

        mod.torch.load = _jump_after_last_load
        try:
            handoff_m = resolve_handoff(
                sdir_i5, plan_i5, world_i5.decoder,
                world_i5.registration_for("D1"), {"includes": ["D6"]},
                m_budget,
                expected_identity=_synthetic_stage_identity(
                    plan_i5, world_i5.registration_for("D1")),
                production_decoder=world_i5.loss_fn)
        finally:
            mod.torch.load = real_load_m
        caught = (handoff_m["handoff_ready"] is False
                  and any("handoff_deadline_exceeded:handoff_final_claim" in p
                          for p in handoff_m["problems"])
                  and loads_m["n"] == 2
                  and m_budget.exceeded is True)
        record("handoff_final_claim_deadline_refused", caught,
               "last checkpoint read crossed the deadline; the handoff claim "
               f"was refused (loads={loads_m['n']}, "
               f"problems={handoff_m['problems'][:1]})")

        # --- P1(r8): a conditioner returning past the deadline refuses the
        # backbone forward (independent unit boundary) ------------------------
        world_n1, plan_n1, _, _, sdir_n1, _, _ = _mini_stage(
            tmp / "fbackbone", "D1", n_windows=1, steps=1, seed=seed + 55,
            finish=True)
        n_holder = {"t": 0.0}
        n_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: n_holder["t"])
        base_factory = build_modules_factory(world_n1, plan_n1)
        cond_calls = {"n": 0}
        backbone_calls = {"n": 0}

        def _wrapped_factory():
            model_, cond_ = base_factory()
            real_cond_fwd = cond_.forward
            real_model_fwd = model_.forward

            def cond_forward(*a, **kw):
                out = real_cond_fwd(*a, **kw)
                cond_calls["n"] += 1
                n_holder["t"] = n_budget.work_deadline + 5.0
                return out

            def model_forward(*a, **kw):
                backbone_calls["n"] += 1
                return real_model_fwd(*a, **kw)

            cond_.forward = cond_forward
            model_.forward = model_forward
            return model_, cond_

        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_n1, "plan": plan_n1,
                          "registration": world_n1.registration_for("D1"),
                          "decoder": world_n1.decoder,
                          "loss_fn": world_n1.loss_fn,
                          "identity": _synthetic_stage_identity(plan_n1, world_n1.registration_for("D1"))}],
            out_dir=tmp / "fbackbone_f", budget=n_budget,
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=_wrapped_factory, authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (cond_calls["n"] == 1
                  and backbone_calls["n"] == 0
                  and any("f_fresh_backbone" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["verdict"] == "incomplete")
        record("f_fresh_backbone_deadline_stop", caught,
               "conditioner returned past the deadline; the backbone forward "
               f"never started (cond={cond_calls['n']}, "
               f"backbone={backbone_calls['n']})")

        # --- P1(r8): checkpoint-read→strict-load and model→conditioner load
        # boundaries are guarded ---------------------------------------------
        world_n2, plan_n2, _, _, sdir_n2, _, _ = _mini_stage(
            tmp / "fstrict", "D1", n_windows=1, steps=1, seed=seed + 56,
            finish=True)
        real_load_n = mod.torch.load
        real_lsd_n = torch.nn.Module.load_state_dict

        def _strict_run(out_name: str, jump_after_load: bool) -> tuple[dict, dict]:
            holder = {"t": 0.0}
            budget_ = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                                  t0=0.0, clock=lambda: holder["t"])
            loads = {"n": 0}

            def _jump_after_read(*a, **kw):
                value = real_load_n(*a, **kw)
                if not jump_after_load \
                        and sys._getframe(1).f_code.co_name \
                        == "_load_state_dicts_checked":
                    holder["t"] = budget_.work_deadline + 5.0
                return value

            def _count_lsd(self, *a, **kw):
                out = real_lsd_n(self, *a, **kw)
                if sys._getframe(1).f_code.co_name == "_load_state_dicts_checked":
                    loads["n"] += 1
                    if jump_after_load and loads["n"] == 1:
                        holder["t"] = budget_.work_deadline + 5.0
                return out

            mod.torch.load = _jump_after_read
            torch.nn.Module.load_state_dict = _count_lsd
            try:
                result = run_f_verification(
                    stage_specs=[{"stage_dir": sdir_n2, "plan": plan_n2,
                                  "registration": world_n2.registration_for("D1"),
                                  "decoder": world_n2.decoder,
                                  "loss_fn": world_n2.loss_fn,
                                  "identity": _synthetic_stage_identity(plan_n2, world_n2.registration_for("D1"))}],
                    out_dir=tmp / out_name, budget=budget_,
                    runtime_guard=SyntheticRuntimeGuard(),
                    build_modules=build_modules_factory(world_n2, plan_n2),
                    authorization=None)
            finally:
                mod.torch.load = real_load_n
                torch.nn.Module.load_state_dict = real_lsd_n
            return result, loads

        f_read, loads_read = _strict_run("fstrict_read_f", False)
        s0 = f_read["summary_doc"]["stages"][0]
        read_ok = (loads_read["n"] == 0
                   and any("f_strict_load_initial_pre" in p
                           for p in s0.get("replay_problems", [])))
        f_model, loads_model = _strict_run("fstrict_model_f", True)
        s0 = f_model["summary_doc"]["stages"][0]
        model_ok = (loads_model["n"] == 1
                    and any("f_strict_load_initial_conditioner" in p
                            for p in s0.get("replay_problems", [])))
        caught = (read_ok and model_ok
                  and f_read["summary_doc"]["verdict"] == "incomplete"
                  and f_model["summary_doc"]["verdict"] == "incomplete")
        record("f_strict_load_deadline_stop", caught,
               "read past deadline: 0 strict loads; model load past deadline: "
               f"conditioner load refused (loads={loads_read['n']}/"
               f"{loads_model['n']})")

        # --- P1(r8): every input-rebuild read is guarded (D6 six windows) ----
        world_n3, plan_n3, _, _, sdir_n3, _, _ = _mini_stage(
            tmp / "finputs", "D6", n_windows=6, steps=1, seed=seed + 57,
            finish=True)
        o_holder = {"t": 0.0}
        o_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: o_holder["t"])
        real_read_o = mod.read_npz_records
        real_np_load = np.load
        raw_reads = {"n": 0}
        cond_reads = {"n": 0}

        def _slow_raw_read(path, *a, **kw):
            out = real_read_o(path, *a, **kw)
            if sys._getframe(1).f_code.co_name == "run_f_verification":
                raw_reads["n"] += 1
                if raw_reads["n"] == 1:
                    o_holder["t"] = o_budget.work_deadline + 5.0
            return out

        def _count_cond_read(*a, **kw):
            if sys._getframe(1).f_code.co_name == "run_f_verification":
                cond_reads["n"] += 1
            return real_np_load(*a, **kw)

        mod.read_npz_records = _slow_raw_read
        np.load = _count_cond_read
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_n3, "plan": plan_n3,
                              "registration": world_n3.registration_for("D6"),
                              "decoder": world_n3.decoder,
                              "loss_fn": world_n3.loss_fn,
                              "identity": _synthetic_stage_identity(plan_n3, world_n3.registration_for("D6"))}],
                out_dir=tmp / "finputs_f", budget=o_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_n3, plan_n3),
                authorization=None)
        finally:
            mod.read_npz_records = real_read_o
            np.load = real_np_load
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (raw_reads["n"] == 1
                  and cond_reads["n"] == 0
                  and any("f_probe_initial_input_w0_kps_obj" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["verdict"] == "incomplete")
        record("f_input_rebuild_deadline_stop", caught,
               "first raw input read crossed the deadline; the condition "
               f"reads and windows never started "
               f"(raw={raw_reads['n']}, cond={cond_reads['n']})")

        # --- P1(r8): the per-window endpoint artifact read is guarded --------
        world_n4, plan_n4, _, _, sdir_n4, _, _ = _mini_stage(
            tmp / "fendpoint", "D6", n_windows=6, steps=1, seed=seed + 58,
            finish=True)
        p_holder = {"t": 0.0}
        p_budget = StageBudget(stage="F", wall_cap_s=100.0, reserve_s=10.0,
                               t0=0.0, clock=lambda: p_holder["t"])
        real_saved_p = mod._saved_probe_records
        real_metrics_p = mod.compute_window_metrics
        saved_reads = {"n": 0}

        def _count_saved_read(*a, **kw):
            out = real_saved_p(*a, **kw)
            if sys._getframe(1).f_code.co_name == "run_f_verification":
                saved_reads["n"] += 1
            return out

        def _jump_after_first_reduction(*a, **kw):
            out = real_metrics_p(*a, **kw)
            if sys._getframe(1).f_code.co_name == "run_f_verification":
                p_holder["t"] = p_budget.work_deadline + 5.0
            return out

        mod._saved_probe_records = _count_saved_read
        mod.compute_window_metrics = _jump_after_first_reduction
        try:
            f_out = run_f_verification(
                stage_specs=[{"stage_dir": sdir_n4, "plan": plan_n4,
                              "registration": world_n4.registration_for("D6"),
                              "decoder": world_n4.decoder,
                              "loss_fn": world_n4.loss_fn,
                              "identity": _synthetic_stage_identity(plan_n4, world_n4.registration_for("D6"))}],
                out_dir=tmp / "fendpoint_f", budget=p_budget,
                runtime_guard=SyntheticRuntimeGuard(),
                build_modules=build_modules_factory(world_n4, plan_n4),
                authorization=None)
        finally:
            mod._saved_probe_records = real_saved_p
            mod.compute_window_metrics = real_metrics_p
        s0 = f_out["summary_doc"]["stages"][0]
        caught = (saved_reads["n"] == 1
                  and any("f_saved_probe_read_initial_w1" in p
                          for p in s0.get("replay_problems", []))
                  and f_out["summary_doc"]["verdict"] == "incomplete")
        record("f_saved_probe_read_deadline_stop", caught,
               "first reduction crossed the deadline; the next window's "
               f"endpoint read was refused (reads={saved_reads['n']})")

        # --- P2(r8): wrong-shaped containers close as incomplete reports,
        # never crash F without a report -------------------------------------
        world_n5, plan_n5, _, _, sdir_n5, _, _ = _mini_stage(
            tmp / "badidx", "D1", n_windows=1, steps=1, seed=seed + 59,
            finish=True)
        (sdir_n5 / "probe_index.json").write_text("[]")
        f_bad_idx = run_f_verification(
            stage_specs=[{"stage_dir": sdir_n5, "plan": plan_n5,
                          "registration": world_n5.registration_for("D1"),
                          "decoder": world_n5.decoder,
                          "loss_fn": world_n5.loss_fn,
                          "identity": _synthetic_stage_identity(plan_n5, world_n5.registration_for("D1"))}],
            out_dir=tmp / "badidx_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_n5, plan_n5),
            authorization=None)
        s0 = f_bad_idx["summary_doc"]["stages"][0]
        idx_ok = ((Path(tmp) / "badidx_f" / "f_report.json").is_file()
                  and (Path(tmp) / "badidx_f" /
                       "return_observation.json").is_file()
                  and s0.get("skip_reason") == "local_reverify_failed"
                  and any("probe_index_invalid" in p
                          for p in s0.get("local_reverify_problems", []))
                  and f_bad_idx["summary_doc"]["verdict"] == "incomplete")
        world_n6, plan_n6, _, _, sdir_n6, _, _ = _mini_stage(
            tmp / "badseal", "D1", n_windows=1, steps=1, seed=seed + 60,
            finish=True)
        rep = json.loads((sdir_n6 / "stage_report.json").read_text())
        rep["seal"] = [1]
        (sdir_n6 / "stage_report.json").write_text(json.dumps(rep))
        f_bad_seal = run_f_verification(
            stage_specs=[{"stage_dir": sdir_n6, "plan": plan_n6,
                          "registration": world_n6.registration_for("D1"),
                          "decoder": world_n6.decoder,
                          "loss_fn": world_n6.loss_fn,
                          "identity": _synthetic_stage_identity(plan_n6, world_n6.registration_for("D1"))}],
            out_dir=tmp / "badseal_f", budget=StageBudget("F", 300.0, 10.0),
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=build_modules_factory(world_n6, plan_n6),
            authorization=None)
        s0 = f_bad_seal["summary_doc"]["stages"][0]
        seal_ok = ((Path(tmp) / "badseal_f" / "f_report.json").is_file()
                   and (Path(tmp) / "badseal_f" /
                        "return_observation.json").is_file()
                   and s0.get("skip_reason") == "stage_seal_not_consumable"
                   and f_bad_seal["summary_doc"]["verdict"] == "incomplete")
        caught = idx_ok and seal_ok
        record("bad_containers_report_incomplete_f", caught,
               "probe_index=[] and seal=[1] both close as incomplete F "
               f"reports with proofs (idx={s0.get('skip_reason')})")

        # --- P1(r3): F refuses BEFORE construction past the deadline ----------
        world_f4, plan_f4, _, _, sdir_f4, _, _ = _mini_stage(
            tmp / "fconstruct", "D1", n_windows=1, steps=1, seed=seed + 38,
            finish=True)
        built = {"count": 0}

        def counting_build_f4():
            built["count"] += 1
            return build_modules_factory(world_f4, plan_f4)()

        # one passing stage-entry check, then the saved-quantity recompute is
        # refused past the deadline: no construction, no forward
        budget_f4 = StageBudget(stage="F", wall_cap_s=300.0, reserve_s=10.0,
                                t0=0.0, clock=_JumpClock(passes=1, t_jump=1e9))
        f_out = run_f_verification(
            stage_specs=[{"stage_dir": sdir_f4, "plan": plan_f4,
                          "registration": world_f4.registration_for("D1"),
                          "decoder": world_f4.decoder,
                          "loss_fn": world_f4.loss_fn,
                          "identity": _synthetic_stage_identity(plan_f4, world_f4.registration_for("D1"))}],
            out_dir=tmp / "fconstruct_f", budget=budget_f4,
            runtime_guard=SyntheticRuntimeGuard(),
            build_modules=counting_build_f4, authorization=None)
        s0 = f_out["summary_doc"]["stages"][0]
        fc4 = s0.get("fresh_counters") or {}
        stop4 = str(s0.get("deadline_stop") or "")
        caught = (built["count"] == 0
                  and fc4.get("conditioner_forward_calls", 0) == 0
                  and s0.get("skip_reason") == "stage_deadline_stop"
                  and "deadline" in stop4
                  and f_out["summary_doc"]["complete_evidence"] is False)
        record("f_construct_after_deadline_refused", caught,
               f"deadline crossed by the pre-verification phase at {stop4[:40]}; "
               f"builds={built['count']}, fresh forwards="
               f"{fc4.get('conditioner_forward_calls', 0)}")

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
                          "loss_fn": world_f5.loss_fn,
                          "identity": _synthetic_stage_identity(plan_f5, world_f5.registration_for("D1"))}],
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
                          "loss_fn": world_p6.loss_fn,
                          "identity": _synthetic_stage_identity(plan_p6, world_p6.registration_for("D1"))}],
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
                                  expected_identity=_synthetic_stage_identity(
                                      plan_p6, world_p6.registration_for("D1")),
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

        # ==================================================================
        # Round-9 review counterexamples
        # ==================================================================
        trainer_mod = sys.modules[Trainer.__module__]

        def _r9_world(tag_seed: int):
            return make_synthetic_world(seed=tag_seed, n_windows=6, d_model=16)

        def _r9_driver(root: Path, role: str, budget_,
                       world_) -> tuple[FitPlan, "FitDriver", Path]:
            plan_ = FitPlan(role, steps=1, probe_every=1, synthetic=True,
                            d_model=16, n_layers=1, n_heads=4, ffn_mult=2,
                            lr=0.02)
            sdir_ = Path(root) / role.lower()
            view_ = world_.view_for(role)
            trainer_ = world_.build_trainer(
                plan_, sdir_, budget_deadline=budget_.work_deadline,
                train_ds=view_["ds"])
            driver_ = FitDriver(
                trainer=trainer_, plan=plan_,
                registration=view_["registration"], loss_fn=world_.loss_fn,
                decoder=world_.decoder, probe_batch=view_["probe_batch"],
                budget=budget_, runtime_guard=SyntheticRuntimeGuard(),
                identity={"synthetic": True}, stage_dir=sdir_,
                training_gen=torch.Generator().manual_seed(plan_.seed),
                dataset_window_keys=view_["ds"].window_keys,
                authorization_includes_d6=True)
            return plan_, driver_, sdir_

        # --- P1(r9): checkpoint config/identity semantic binding -----------
        world_r9 = _r9_world(seed + 61)
        fixtures_r9: dict = {}
        for role in ("D1", "D6"):
            _, plan_r9, _, report_r9, sdir_r9, _, _ = _mini_stage(
                tmp / "r9fixtures", role, n_windows=6, steps=1,
                world=world_r9, finish=True)
            assert report_r9["local_complete_evidence"] is True
            fixtures_r9[role] = (plan_r9, sdir_r9)

        def _run_f_r9(name: str, budget_, factory=None,
                      directory=None, guard=None) -> dict:
            plan_r9, sdir_r9 = fixtures_r9["D1"]
            return run_f_verification(
                stage_specs=[{"stage_dir": directory or sdir_r9,
                              "plan": plan_r9,
                              "registration": world_r9.registration_for("D1"),
                              "decoder": world_r9.decoder,
                              "loss_fn": world_r9.loss_fn,
                              "identity": _synthetic_stage_identity(plan_r9, world_r9.registration_for("D1"))}],
                out_dir=tmp / name, budget=budget_,
                runtime_guard=(guard or SyntheticRuntimeGuard()),
                build_modules=(factory
                               or build_modules_factory(world_r9, plan_r9)),
                authorization={"includes": ["D6"]})

        config_bind: dict = {}
        identity_bind = False
        for variant in ("lr", "grad_accum", "identity"):
            copydir = tmp / f"r9bind_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            for ck_name in ("initial.pt", "last.pt"):
                path = copydir / ck_name
                ck = torch.load(path, map_location="cpu", weights_only=False)
                if variant == "lr":
                    ck["config"]["lr"] = 123.0
                elif variant == "grad_accum":
                    ck["config"]["grad_accum"] = 99
                else:
                    ck["identity"] = {"foreign_stage": True}
                torch.save(ck, path)
            _full_resign(copydir)
            local = verify_stage_local(
                copydir, fixtures_r9["D1"][0],
                world_r9.registration_for("D1"), world_r9.decoder,
                production_decoder=world_r9.loss_fn)
            out = _run_f_r9(f"r9bind_{variant}_f",
                            StageBudget("F", 300.0, 10.0), directory=copydir)
            if variant == "identity":
                identity_bind = (
                    any("checkpoint_identity_mismatch" in p
                        for p in local["problems"])
                    and out["summary_doc"]["complete_evidence"] is False)
            else:
                config_bind[variant] = (
                    any("checkpoint_config_mismatch" in p
                        for p in local["problems"])
                    and out["summary_doc"]["complete_evidence"] is False)
        caught = config_bind.get("lr", False) \
            and config_bind.get("grad_accum", False)
        record("checkpoint_config_binding_rejected", caught,
               "re-signed lr=123 / grad_accum=99 checkpoints refused by the "
               f"full frozen-config contract (lr={config_bind.get('lr')}, "
               f"accum={config_bind.get('grad_accum')})")
        record("checkpoint_identity_binding_rejected", identity_bind,
               "re-signed foreign checkpoint identity refused against the "
               "stage's declared identity")

        # --- P1(r9): teacher/probe multi-window unit chains ----------------
        def _stack_has(code) -> bool:
            frame = sys._getframe(1)
            while frame is not None:
                if frame.f_code is code:
                    return True
                frame = frame.f_back
            return False

        tail_results: dict = {}
        for context in ("teacher", "probe"):
            world_t = _r9_world(seed + 62)
            holder_t = {"t": 0.0}
            budget_t = StageBudget("D6", 100.0, 10.0, t0=0.0,
                                   clock=lambda: holder_t["t"])
            _, driver_t, _ = _r9_driver(tmp / f"r9tail_{context}", "D6",
                                        budget_t, world_t)
            scope = (FitDriver._ensure_teacher.__code__ if context == "teacher"
                     else run_probe.__code__)
            counts_t = dict(metric_decodes=0, late_metric_decodes=0,
                            production_decodes=0, late_production_decodes=0,
                            fk_calls=0, late_fk_calls=0,
                            metric_reductions=0, late_metric_reductions=0)
            real_md = world_t.decoder.decode
            real_pd = world_t.loss_fn.decode_clip
            real_fk_t = world_t.fk_inner.site_world_positions
            real_metrics_t = compute_window_metrics

            def md_tail(*a, _scope=scope, _counts=counts_t, _real=real_md,
                        _holder=holder_t, _budget=budget_t, **kw):
                in_scope = _stack_has(_scope)
                if in_scope:
                    _counts["metric_decodes"] += 1
                    _counts["late_metric_decodes"] += int(
                        _holder["t"] >= _budget.work_deadline)
                value = _real(*a, **kw)
                if in_scope and _counts["metric_decodes"] == 1:
                    _holder["t"] = 91
                return value

            def pd_observe(*a, _scope=scope, _counts=counts_t,
                           _real=real_pd, _holder=holder_t,
                           _budget=budget_t, **kw):
                if _stack_has(_scope):
                    _counts["production_decodes"] += 1
                    _counts["late_production_decodes"] += int(
                        _holder["t"] >= _budget.work_deadline)
                return _real(*a, **kw)

            def fk_observe_t(*a, _scope=scope, _counts=counts_t,
                             _real=real_fk_t, _holder=holder_t,
                             _budget=budget_t, **kw):
                if _stack_has(_scope):
                    _counts["fk_calls"] += 1
                    _counts["late_fk_calls"] += int(
                        _holder["t"] >= _budget.work_deadline)
                return _real(*a, **kw)

            def metric_observe_t(*a, _scope=scope, _counts=counts_t,
                                 _real=real_metrics_t, _holder=holder_t,
                                 _budget=budget_t, **kw):
                if _stack_has(_scope):
                    _counts["metric_reductions"] += 1
                    _counts["late_metric_reductions"] += int(
                        _holder["t"] >= _budget.work_deadline)
                return _real(*a, **kw)

            with patch.object(trainer_mod, "time",
                              types.SimpleNamespace(
                                  perf_counter=budget_t.clock)), \
                    patch.object(world_t.decoder, "decode", md_tail), \
                    patch.object(world_t.loss_fn, "decode_clip", pd_observe), \
                    patch.object(world_t.fk_inner, "site_world_positions",
                                 fk_observe_t), \
                    patch.object(mod, "compute_window_metrics",
                                 metric_observe_t):
                summary_t = driver_t.run()
            tail_results[context] = {
                **counts_t,
                "failure": (summary_t.get("failure") or {}).get("type")}
        teacher_ok = (tail_results["teacher"]["late_metric_decodes"] == 0
                      and tail_results["teacher"]["late_production_decodes"] == 0
                      and tail_results["teacher"]["late_fk_calls"] == 0
                      and tail_results["teacher"]["failure"] == "BudgetExceeded")
        probe_ok = (tail_results["probe"]["late_metric_decodes"] == 0
                    and tail_results["probe"]["late_production_decodes"] == 0
                    and tail_results["probe"]["late_fk_calls"] == 0
                    and tail_results["probe"]["late_metric_reductions"] == 0
                    and tail_results["probe"]["failure"] == "BudgetExceeded")
        record("teacher_window_units_deadline_stop", teacher_ok,
               "first teacher decode crossed the deadline; the following "
               f"per-window units never started ({tail_results['teacher']})")
        record("probe_window_units_deadline_stop", probe_ok,
               "first probe decode crossed the deadline; the following "
               f"per-window units never started ({tail_results['probe']})")

        # --- P1(r9): conditioner -> backbone -> loss inside the micro ------
        world_u = _r9_world(seed + 63)
        holder_u = {"t": 0.0}
        budget_u = StageBudget("D1", 100.0, 10.0, t0=0.0,
                               clock=lambda: holder_u["t"])
        _, driver_u, _ = _r9_driver(tmp / "r9micro", "D1", budget_u, world_u)
        counts_u = dict(training_conditioner_calls=0,
                        training_backbone_calls=0,
                        late_training_backbone_calls=0,
                        training_loss_calls=0, late_training_loss_calls=0,
                        training_fk_calls=0, late_training_fk_calls=0)
        real_fk_u = world_u.fk_inner.site_world_positions
        real_loss_u = CountingLoss.__call__

        def conditioner_tail_u(module, _inputs, _output):
            if module.training:
                counts_u["training_conditioner_calls"] += 1
                holder_u["t"] = 91

        def backbone_start_u(module, _inputs):
            if module.training:
                counts_u["training_backbone_calls"] += 1
                counts_u["late_training_backbone_calls"] += int(
                    holder_u["t"] >= budget_u.work_deadline)

        def loss_start_u(self, *a, **kw):
            if _stack_has(Trainer.accumulate_micro.__code__):
                counts_u["training_loss_calls"] += 1
                counts_u["late_training_loss_calls"] += int(
                    holder_u["t"] >= budget_u.work_deadline)
            return real_loss_u(self, *a, **kw)

        def train_fk_u(*a, **kw):
            if _stack_has(Trainer.accumulate_micro.__code__):
                counts_u["training_fk_calls"] += 1
                counts_u["late_training_fk_calls"] += int(
                    holder_u["t"] >= budget_u.work_deadline)
            return real_fk_u(*a, **kw)

        handle_cu = driver_u.trainer.conditioner.register_forward_hook(
            conditioner_tail_u)
        handle_bu = driver_u.trainer.model.register_forward_pre_hook(
            backbone_start_u)
        with patch.object(trainer_mod, "time",
                          types.SimpleNamespace(
                              perf_counter=budget_u.clock)), \
                patch.object(CountingLoss, "__call__", loss_start_u), \
                patch.object(world_u.fk_inner, "site_world_positions",
                             train_fk_u):
            summary_u = driver_u.run()
        handle_cu.remove()
        handle_bu.remove()
        caught = (counts_u["late_training_backbone_calls"] == 0
                  and counts_u["late_training_loss_calls"] == 0
                  and counts_u["late_training_fk_calls"] == 0
                  and (summary_u.get("failure") or {}).get("type")
                  == "BudgetExceeded"
                  and summary_u["counters"]["backward_calls"] == 0
                  and summary_u["counters"]["adamw_step_calls"] == 0)
        record("training_micro_internal_deadline_stop", caught,
               "conditioner returned past the deadline; the backbone/loss/FK "
               f"chain never started ({counts_u})")

        # --- P1(r9): F identity helper checkpoint reads are guarded --------
        identity_tail: dict = {}
        for jump_at in ("local_last_checkpoint", "identity_first_checkpoint"):
            clock_i = _FakeClock(1)
            budget_i = StageBudget("F", 100.0, 10.0, t0=0.0, clock=clock_i)
            counts_i = dict(identity_checkpoint_reads=0,
                            late_identity_checkpoint_reads=0,
                            module_builds=0)
            real_load_i = torch.load
            base_factory_i = build_modules_factory(world_r9,
                                                   fixtures_r9["D1"][0])

            def count_factory_i(_counts=counts_i, _factory=base_factory_i):
                _counts["module_builds"] += 1
                return _factory()

            def identity_reads_i(path, *a, _jump=jump_at, _counts=counts_i,
                                 _real=real_load_i, _clock=clock_i,
                                 _budget=budget_i, **kw):
                caller = sys._getframe(1).f_code
                is_identity = caller is _f_identity_problems.__code__
                if is_identity:
                    _counts["identity_checkpoint_reads"] += 1
                    _counts["late_identity_checkpoint_reads"] += int(
                        _clock.t >= _budget.work_deadline)
                ck = _real(path, *a, **kw)
                if (_jump == "local_last_checkpoint"
                        and caller is _checkpoint_binding_problems.__code__
                        and Path(path).name == "last.pt"):
                    _clock.t = 91
                elif (_jump == "identity_first_checkpoint" and is_identity
                        and _counts["identity_checkpoint_reads"] == 1):
                    _clock.t = 91
                return ck

            with patch.object(torch, "load", identity_reads_i):
                out_i = _run_f_r9(f"r9identity_{jump_at}_f", budget_i,
                                  factory=count_factory_i)
            s0 = out_i["summary_doc"]["stages"][0]
            identity_tail[jump_at] = (
                counts_i["late_identity_checkpoint_reads"] == 0
                and counts_i["module_builds"] == 0
                and s0.get("skip_reason") == "stage_deadline_stop"
                and out_i["summary_doc"]["complete_evidence"] is False)
        caught = all(identity_tail.values())
        record("f_identity_checkpoint_read_deadline_stop", caught,
               "identity-helper checkpoint reads are guarded at both the "
               f"entry and inter-read boundaries ({identity_tail})")

        # --- P2(r9): the completed conditioner count survives ---------------
        clock_c = _FakeClock(1)
        budget_c = StageBudget("F", 100.0, 10.0, t0=0.0, clock=clock_c)
        forward_counts = dict(conditioner_actual=0, backbone_actual=0)
        base_factory_c = build_modules_factory(world_r9,
                                               fixtures_r9["D1"][0])

        def forward_factory_c(_factory=base_factory_c, _counts=forward_counts,
                              _clock=clock_c):
            model_, cond_ = _factory()

            def cond_end(*_a):
                _counts["conditioner_actual"] += 1
                _clock.t = 91

            def model_start(*_a):
                _counts["backbone_actual"] += 1

            cond_.register_forward_hook(cond_end)
            model_.register_forward_pre_hook(model_start)
            return model_, cond_

        out_c = _run_f_r9("r9count_f", budget_c, factory=forward_factory_c)
        disk_c = json.loads(
            (tmp / "r9count_f" / "f_report.json").read_text())
        ret_count = out_c["summary_doc"]["stages"][0]["fresh_counters"]
        disk_count = disk_c["stages"][0]["fresh_counters"]
        caught = (forward_counts["conditioner_actual"] == 1
                  and forward_counts["backbone_actual"] == 0
                  and ret_count.get("conditioner_forward_calls") == 1
                  and disk_count.get("conditioner_forward_calls") == 1
                  and ret_count.get("backbone_forward_calls") == 0
                  and disk_count.get("backbone_forward_calls") == 0)
        record("f_completed_conditioner_counter_kept", caught,
               "the completed conditioner call stays recorded when the "
               f"backbone gate refuses (returned={ret_count.get('conditioner_forward_calls')}, "
               f"disk={disk_count.get('conditioner_forward_calls')})")

        # --- P2(r9): nested wrong-shaped probe containers -------------------
        schema_results: dict = {}
        for variant in ("step_string", "window_id_list",
                        "forward_counts_list", "extra_window"):
            copydir = tmp / f"r9schema_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            path = copydir / "probe_index.json"
            pidx = json.loads(path.read_text())
            entry = pidx["probes"][0]
            if variant == "step_string":
                entry["step"] = "bad"
            elif variant == "window_id_list":
                entry["windows"][0]["window_id"] = []
            elif variant == "forward_counts_list":
                entry["forward_counts"] = [1]
            else:
                entry["windows"].append(dict(entry["windows"][0]))
            path.write_text(json.dumps(pidx))
            _full_resign(copydir)
            error = None
            out_s = None
            try:
                out_s = _run_f_r9(f"r9schema_{variant}_f",
                                  StageBudget("F", 300.0, 10.0),
                                  directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir = tmp / f"r9schema_{variant}_f"
            schema_results[variant] = (
                error is None
                and (fdir / "f_report.json").is_file()
                and (fdir / "return_observation.json").is_file()
                and out_s is not None
                and out_s["summary_doc"]["verdict"] == "incomplete")
        caught = all(schema_results.values())
        record("nested_probe_schema_containers_report_incomplete", caught,
               "step/window_id/forward_counts/extra-window shape faults all "
               f"close as incomplete F reports ({schema_results})")

        # ==================================================================
        # Round-10 review counterexamples
        # ==================================================================
        # --- P1(r10): the seven loss weights are part of the frozen target --
        frozen_plan_r10 = FitPlan("D1")
        frozen_cfg_r10 = TrainConfig(
            model_type=frozen_plan_r10.model_type,
            steps=frozen_plan_r10.steps,
            micro_batch=frozen_plan_r10.micro_batch,
            grad_accum=frozen_plan_r10.grad_accum,
            seed=frozen_plan_r10.seed, val_seed=frozen_plan_r10.val_seed,
            loss=LossWeights(w_fk=999.0, w_contact=0.0))
        frozen_rejected = False
        try:
            FitDriver._check_trainer_config(
                types.SimpleNamespace(cfg=frozen_cfg_r10, train_ds=[None]),
                frozen_plan_r10)
        except FitDiagnosticError as exc:
            frozen_rejected = "config_loss" in str(exc)
        frozen_problems = _checkpoint_config_problems(
            asdict(frozen_cfg_r10), frozen_plan_r10, "metadata_only.pt")
        world_l = _r9_world(seed + 64)
        plan_l = FitPlan("D1", steps=1, probe_every=1, synthetic=True,
                         d_model=16, n_layers=1, n_heads=4, ffn_mult=2,
                         lr=0.02)
        view_l = world_l.view_for("D1")
        trainer_l = world_l.build_trainer(plan_l, tmp / "r10loss", None,
                                          train_ds=view_l["ds"])
        trainer_l.cfg.loss.w_fk = 999.0
        world_l.loss_fn.w.w_fk = 999.0
        objective_rejected = False
        try:
            FitDriver(trainer=trainer_l, plan=plan_l,
                      registration=view_l["registration"],
                      loss_fn=world_l.loss_fn, decoder=world_l.decoder,
                      probe_batch=view_l["probe_batch"],
                      budget=StageBudget("D1", 60.0, 5.0),
                      runtime_guard=SyntheticRuntimeGuard(), identity={},
                      stage_dir=tmp / "r10loss",
                      training_gen=torch.Generator().manual_seed(plan_l.seed),
                      dataset_window_keys=view_l["ds"].window_keys)
        except FitDiagnosticError as exc:
            objective_rejected = ("config_loss" in str(exc)
                                  or "loss_weight_binding" in str(exc))
        loss_bind: dict = {}
        for variant in ("w_fk", "w_contact", "loss_missing"):
            copydir = tmp / f"r10loss_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            for ck_name in ("initial.pt", "last.pt"):
                path = copydir / ck_name
                ck = torch.load(path, map_location="cpu", weights_only=False)
                if variant == "loss_missing":
                    del ck["config"]["loss"]
                else:
                    ck["config"]["loss"][variant] = (
                        999.0 if variant == "w_fk" else 1.0)
                torch.save(ck, path)
            _full_resign(copydir)
            local = verify_stage_local(
                copydir, fixtures_r9["D1"][0],
                world_r9.registration_for("D1"), world_r9.decoder,
                production_decoder=world_r9.loss_fn)
            out = _run_f_r9(f"r10loss_{variant}_f",
                            StageBudget("F", 300.0, 10.0), directory=copydir)
            loss_bind[variant] = (
                any("loss" in p for p in local["problems"])
                and out["summary_doc"]["complete_evidence"] is False
                and out["summary_doc"]["verdict"] == "incomplete"
                and (tmp / f"r10loss_{variant}_f" / "f_report.json").is_file())
        record("loss_weight_binding_rejected",
               frozen_rejected and bool(frozen_problems)
               and objective_rejected and all(loss_bind.values()),
               "w_fk=999 (config+actual target), w_contact=1 and a missing "
               "loss block are all refused; the metadata-only frozen-plan "
               f"omission is rejected too ({loss_bind})")

        # --- P1(r10): the stage identity is anchored to plan+registration ---
        copydir = tmp / "r10identity"
        shutil.copytree(fixtures_r9["D1"][1], copydir)
        foreign = {"foreign_stage": True}
        report_path = copydir / "stage_report.json"
        report = json.loads(report_path.read_text())
        report["identity"] = foreign
        report_path.write_text(json.dumps(report))
        for ck_name in ("initial.pt", "last.pt"):
            path = copydir / ck_name
            ck = torch.load(path, map_location="cpu", weights_only=False)
            ck["identity"] = foreign
            torch.save(ck, path)
        _full_resign(copydir)
        local = verify_stage_local(
            copydir, fixtures_r9["D1"][0],
            world_r9.registration_for("D1"), world_r9.decoder,
            production_decoder=world_r9.loss_fn)
        out_i = _run_f_r9("r10identity_f", StageBudget("F", 300.0, 10.0),
                          directory=copydir)
        caught = (any("stage_identity_binding_mismatch" in p
                      for p in local["problems"])
                  and out_i["summary_doc"]["complete_evidence"] is False
                  and out_i["summary_doc"]["verdict"] == "incomplete"
                  and (tmp / "r10identity_f" / "f_report.json").is_file())
        record("stage_identity_external_anchor", caught,
               "a coordinated report+checkpoint identity rewrite is refused "
               "against the plan/registration digests the verifier was given")

        # --- P1(r10): predict's post-backbone validity tail -----------------
        world_t2 = _r9_world(seed + 65)
        holder_t2 = {"t": 0.0}
        budget_t2 = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                clock=lambda: holder_t2["t"])
        _, driver_t2, _ = _r9_driver(tmp / "r10tail", "D1", budget_t2,
                                     world_t2)
        counts_t2 = dict(validity_in_predict=0, loss_calls=0,
                         late_loss_calls=0, fk_calls=0, late_fk_calls=0)
        real_valid_t2 = getattr(trainer_mod, "_rotation6d_valid")
        real_loss_t2 = CountingLoss.__call__
        real_fk_t2 = world_t2.fk_inner.site_world_positions

        def valid_tail(*a, _driver=driver_t2, _holder=holder_t2,
                       _counts=counts_t2, **kw):
            value = real_valid_t2(*a, **kw)
            if (sys._getframe(1).f_code is Trainer.predict.__code__
                    and _driver.trainer.model.training):
                _counts["validity_in_predict"] += 1
                _holder["t"] = 91
            return value

        def loss_tail_u(self, *a, _counts=counts_t2, _holder=holder_t2,
                        _budget=budget_t2, **kw):
            _counts["loss_calls"] += 1
            _counts["late_loss_calls"] += int(
                _holder["t"] >= _budget.work_deadline)
            return real_loss_t2(self, *a, **kw)

        def fk_tail_u(*a, _counts=counts_t2, _holder=holder_t2,
                      _budget=budget_t2, **kw):
            _counts["fk_calls"] += 1
            _counts["late_fk_calls"] += int(
                _holder["t"] >= _budget.work_deadline)
            return real_fk_t2(*a, **kw)

        with patch.object(trainer_mod, "time",
                          types.SimpleNamespace(
                              perf_counter=budget_t2.clock)), \
                patch.object(trainer_mod, "_rotation6d_valid", valid_tail), \
                patch.object(CountingLoss, "__call__", loss_tail_u), \
                patch.object(world_t2.fk_inner, "site_world_positions",
                             fk_tail_u):
            summary_t2 = driver_t2.run()
        failure_t2 = summary_t2.get("failure") or {}
        caught = (counts_t2["validity_in_predict"] == 1
                  and counts_t2["late_loss_calls"] == 0
                  and counts_t2["late_fk_calls"] == 0
                  and failure_t2.get("type") == "BudgetExceeded"
                  and "post_predict" in failure_t2.get("message", "")
                  and summary_t2["counters"]["backward_calls"] == 0
                  and summary_t2["counters"]["adamw_step_calls"] == 0)
        record("post_backbone_validation_tail_stop", caught,
               "a deadline crossed inside predict's output-validity checks "
               f"refuses the loss/FK chain ({counts_t2}, "
               f"failure={failure_t2.get('message', '')[:48]})")

        # --- P2(r10): a refused backbone is not counted ---------------------
        world_c2 = _r9_world(seed + 66)
        holder_c2 = {"t": 0.0}
        budget_c2 = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                clock=lambda: holder_c2["t"])
        _, driver_c2, _ = _r9_driver(tmp / "r10count", "D1", budget_c2,
                                     world_c2)
        actual_c2 = dict(conditioner=0, backbone=0)
        real_forward_c2 = driver_c2.trainer.model.forward

        def observe_backbone_c2(*a, _counts=actual_c2,
                                _real=real_forward_c2, **kw):
            _counts["backbone"] += 1
            return _real(*a, **kw)

        def conditioner_end_c2(module, _inputs, _output):
            actual_c2["conditioner"] += 1
            if module.training:
                holder_c2["t"] = 91

        handle_c2 = driver_c2.trainer.conditioner.register_forward_hook(
            conditioner_end_c2)
        with patch.object(trainer_mod, "time",
                          types.SimpleNamespace(
                              perf_counter=budget_c2.clock)), \
                patch.object(driver_c2.trainer.model, "forward",
                             observe_backbone_c2):
            summary_c2 = driver_c2.run()
        handle_c2.remove()
        failure_c2 = summary_c2.get("failure") or {}
        caught = (actual_c2["conditioner"] == 2
                  and actual_c2["backbone"] == 1
                  and summary_c2["counters"]["conditioner_forward_calls"] == 2
                  and summary_c2["counters"]["backbone_forward_calls"] == 1
                  and (failure_c2.get("counters") or {}).get(
                      "backbone_forward_calls") == 1)
        record("refused_backbone_not_counted", caught,
               "the backbone admission hook refuses before forward runs; "
               f"actual={actual_c2}, reported="
               f"{summary_c2['counters']['backbone_forward_calls']}")

        # --- P2(r10): bad metrics/root_stats containers report incomplete ---
        schema_r10: dict = {}
        for variant in ("metric_list", "metric_nested_list", "root_stats"):
            copydir = tmp / f"r10schema_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            if variant.startswith("metric"):
                path = copydir / "probe_index.json"
                pidx = json.loads(path.read_text())
                pidx["probes"][0]["windows"][0]["metrics"] = (
                    ["root_position_rmse_m"] if variant == "metric_list"
                    else [[]])
                path.write_text(json.dumps(pidx))
            else:
                path = copydir / "initial.pt"
                ck = torch.load(path, map_location="cpu", weights_only=False)
                ck["root_stats"]["mean"] = {"bad": 1}
                torch.save(ck, path)
            _full_resign(copydir)
            error = None
            out_s = None
            try:
                out_s = _run_f_r9(f"r10schema_{variant}_f",
                                  StageBudget("F", 300.0, 10.0),
                                  directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir = tmp / f"r10schema_{variant}_f"
            schema_r10[variant] = (
                error is None
                and (fdir / "f_report.json").is_file()
                and (fdir / "return_observation.json").is_file()
                and out_s is not None
                and out_s["summary_doc"]["verdict"] == "incomplete")
        record("bad_metrics_and_root_stats_report_incomplete",
               all(schema_r10.values()),
               "list/nested-list metrics and a wrong-shaped root_stats "
               f"mapping all close as incomplete F reports ({schema_r10})")

        # ==================================================================
        # Round-11 review counterexamples
        # ==================================================================
        # --- P1(r11): LossAssembler's own decode->FK chain is guarded -------
        loss_units_r11: dict = {}
        for inject_after in ("none", "pred_decode", "pred_fk"):
            world_u2 = _r9_world(seed + 67)
            holder_u2 = {"t": 0.0}
            budget_u2 = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                    clock=lambda: holder_u2["t"])
            _, driver_u2, _ = _r9_driver(tmp / f"r11units_{inject_after}",
                                         "D1", budget_u2, world_u2)
            counts_u2 = dict(training_decodes=0, late_training_decodes=0,
                             training_fks=0, late_training_fks=0)
            real_decode_u2 = world_u2.loss_fn.decode_clip
            real_fk_u2 = world_u2.fk_inner.site_world_positions

            def decode_u2(*a, _c=counts_u2, _real=real_decode_u2,
                          _holder=holder_u2, _budget=budget_u2,
                          _inj=inject_after, **kw):
                active = _stack_has(Trainer.accumulate_micro.__code__)
                if active:
                    _c["training_decodes"] += 1
                    _c["late_training_decodes"] += int(
                        _holder["t"] >= _budget.work_deadline)
                value = _real(*a, **kw)
                if (active and _inj == "pred_decode"
                        and _c["training_decodes"] == 1):
                    _holder["t"] = 91.0
                return value

            def fk_u2(*a, _c=counts_u2, _real=real_fk_u2,
                      _holder=holder_u2, _budget=budget_u2,
                      _inj=inject_after, **kw):
                active = _stack_has(Trainer.accumulate_micro.__code__)
                if active:
                    _c["training_fks"] += 1
                    _c["late_training_fks"] += int(
                        _holder["t"] >= _budget.work_deadline)
                value = _real(*a, **kw)
                if (active and _inj == "pred_fk"
                        and _c["training_fks"] == 1):
                    _holder["t"] = 91.0
                return value

            with patch.object(trainer_mod, "time",
                              types.SimpleNamespace(
                                  perf_counter=budget_u2.clock)), \
                    patch.object(world_u2.loss_fn, "decode_clip", decode_u2), \
                    patch.object(world_u2.fk_inner, "site_world_positions",
                                 fk_u2):
                summary_u2 = driver_u2.run()
            loss_units_r11[inject_after] = {
                **counts_u2,
                "failure": (summary_u2.get("failure") or {}).get("type"),
                "complete": summary_u2.get("local_complete_evidence"),
                "backward": summary_u2["counters"]["backward_calls"],
                "step": summary_u2["counters"]["adamw_step_calls"]}
        caught = (loss_units_r11["none"]["training_decodes"] == 8
                  and loss_units_r11["none"]["training_fks"] == 8
                  and loss_units_r11["none"]["complete"] is True
                  and all(loss_units_r11[inj]["late_training_decodes"] == 0
                          and loss_units_r11[inj]["late_training_fks"] == 0
                          and loss_units_r11[inj]["failure"] == "BudgetExceeded"
                          and loss_units_r11[inj]["backward"] == 0
                          and loss_units_r11[inj]["step"] == 0
                          for inj in ("pred_decode", "pred_fk")))
        record("loss_internal_fk_units_deadline_stop", caught,
               "decode/FK inside the production loss are bracketed: a "
               f"completed unit crossing the deadline starts no later unit "
               f"({loss_units_r11})")

        # --- P1(r11): live loss weights are re-checked at run time ----------
        world_l2 = _r9_world(seed + 68)
        holder_l2 = {"t": 0.0}
        budget_l2 = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                clock=lambda: holder_l2["t"])
        _, driver_l2, sdir_l2 = _r9_driver(tmp / "r11livedrift", "D1",
                                           budget_l2, world_l2)
        world_l2.loss_fn.w.w_fk = 999.0
        used_l2: list = []
        real_loss_l2 = LossAssembler.__call__

        def observe_loss_l2(self, *a, _used=used_l2, _real=real_loss_l2, **kw):
            if _stack_has(Trainer.accumulate_micro.__code__):
                _used.append(float(self.w.w_fk))
            return _real(self, *a, **kw)

        with patch.object(trainer_mod, "time",
                          types.SimpleNamespace(
                              perf_counter=budget_l2.clock)), \
                patch.object(LossAssembler, "__call__", observe_loss_l2):
            summary_l2 = driver_l2.run()
        failure_l2 = summary_l2.get("failure") or {}
        out_l2 = _run_f_r9("r11livedrift_f", StageBudget("F", 300.0, 10.0),
                           directory=sdir_l2)
        caught = (used_l2 == []
                  and failure_l2.get("type") == "FitDiagnosticError"
                  and "loss_weight_drift" in failure_l2.get("message", "")
                  and summary_l2.get("local_complete_evidence") is False
                  and out_l2["summary_doc"]["complete_evidence"] is False)
        record("live_loss_weight_drift_refused", caught,
               "post-construction loss_fn.w drift is refused before any "
               f"micro consumes it (uses={used_l2}, "
               f"failure={failure_l2.get('type')})")

        # --- P1(r11): the FULL external identity is the anchor --------------
        copydir = tmp / "r11identity_full"
        shutil.copytree(fixtures_r9["D1"][1], copydir)
        report_path = copydir / "stage_report.json"
        report = json.loads(report_path.read_text())
        original_identity = dict(report["identity"])
        foreign = {**original_identity, "synthetic": False, "stage": "D6",
                   "module": "foreign.module",
                   "implementation_sha256": "b" * 64}
        report["identity"] = foreign
        report_path.write_text(json.dumps(report))
        for ck_name in ("initial.pt", "last.pt"):
            path = copydir / ck_name
            ck = torch.load(path, map_location="cpu", weights_only=False)
            ck["identity"] = foreign
            torch.save(ck, path)
        _full_resign(copydir)
        local_no_anchor = verify_stage_local(
            copydir, fixtures_r9["D1"][0],
            world_r9.registration_for("D1"), world_r9.decoder,
            production_decoder=world_r9.loss_fn)
        local_external = verify_stage_local(
            copydir, fixtures_r9["D1"][0],
            world_r9.registration_for("D1"), world_r9.decoder,
            production_decoder=world_r9.loss_fn,
            expected_identity=original_identity)
        out_i2 = _run_f_r9("r11identity_full_f",
                           StageBudget("F", 300.0, 10.0), directory=copydir)
        caught = (any("stage_report_identity_mismatch" in p
                      or "checkpoint_identity_mismatch" in p
                      for p in local_external["problems"])
                  and out_i2["summary_doc"]["complete_evidence"] is False
                  and out_i2["summary_doc"]["verdict"] == "incomplete"
                  and (tmp / "r11identity_full_f" / "f_report.json").is_file())
        record("stage_identity_full_external_anchor", caught,
               "coordinated report+checkpoint identity rewrite (digests kept) "
               "is refused against the caller's full external identity "
               f"(no_anchor={local_no_anchor['problems'][:2]})")

        # --- P2(r11): bad config value containers close as incomplete -------
        bad_cfg_r11: dict = {}
        for variant in ("loss_tensor", "lr_array"):
            copydir = tmp / f"r11cfg_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            path = copydir / "initial.pt"
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if variant == "loss_tensor":
                ck["config"]["loss"]["w_fk"] = torch.tensor([2.0, 2.0])
            else:
                ck["config"]["lr"] = np.array([fixtures_r9["D1"][0].lr,
                                               fixtures_r9["D1"][0].lr])
            torch.save(ck, path)
            _full_resign(copydir)
            error, out_c = None, None
            try:
                out_c = _run_f_r9(f"r11cfg_{variant}_f",
                                  StageBudget("F", 300.0, 10.0),
                                  directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir = tmp / f"r11cfg_{variant}_f"
            bad_cfg_r11[variant] = (
                error is None
                and (fdir / "f_report.json").is_file()
                and (fdir / "return_observation.json").is_file()
                and out_c is not None
                and out_c["summary_doc"]["verdict"] == "incomplete")
        record("bad_checkpoint_config_containers_report_incomplete",
               all(bad_cfg_r11.values()),
               "Tensor/ndarray config values are type-strict mismatches, "
               f"never an F crash ({bad_cfg_r11})")

        # --- P2(r11): an entered-but-failing forward is still counted -------
        world_c3 = _r9_world(seed + 69)
        holder_c3 = {"t": 0.0}
        budget_c3 = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                clock=lambda: holder_c3["t"])
        _, driver_c3, _ = _r9_driver(tmp / "r11failedfw", "D1",
                                     budget_c3, world_c3)
        actual_c3 = dict(numeric=0, training=0)
        real_forward_c3 = driver_c3.trainer.model.forward

        def fail_after_forward_c3(*a, _counts=actual_c3,
                                  _real=real_forward_c3, **kw):
            value = _real(*a, **kw)
            _counts["numeric"] += 1
            if driver_c3.trainer.model.training:
                _counts["training"] += 1
                raise TrainingNumericalError("synthetic_failure_after_forward")
            return value

        with patch.object(trainer_mod, "time",
                          types.SimpleNamespace(
                              perf_counter=budget_c3.clock)), \
                patch.object(driver_c3.trainer.model, "forward",
                             fail_after_forward_c3):
            summary_c3 = driver_c3.run()
        failure_c3 = summary_c3.get("failure") or {}
        caught = (actual_c3["numeric"] == 2
                  and actual_c3["training"] == 1
                  and summary_c3["counters"]["backbone_forward_calls"] == 2
                  and (failure_c3.get("counters") or {}).get(
                      "backbone_forward_calls") == 2)
        record("executed_failing_backbone_counted", caught,
               "forward-entry counting after the admission gate: a refused "
               "backbone is not counted, an entered-then-failing one is "
               f"({actual_c3}, reported="
               f"{summary_c3['counters']['backbone_forward_calls']})")

        # --- P2(r11): integer/boolean strictness ----------------------------
        typed_r11: dict = {}
        for variant in ("checkpoint_step", "forward_counts",
                        "worst_coordinate"):
            copydir = tmp / f"r11typed_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            if variant == "checkpoint_step":
                for filename, step in (("initial.pt", False),
                                       ("last.pt",
                                        float(fixtures_r9["D1"][0].steps))):
                    path = copydir / filename
                    ck = torch.load(path, map_location="cpu",
                                    weights_only=False)
                    ck["step"] = step
                    torch.save(ck, path)
            else:
                path = copydir / "probe_index.json"
                pidx = json.loads(path.read_text())
                for entry in pidx["probes"]:
                    if variant == "forward_counts":
                        entry["forward_counts"] = {"conditioner": True,
                                                   "backbone": True}
                    else:
                        for wout in entry["windows"]:
                            for stats in wout["metrics"]["spike"].values():
                                stats["worst_coordinate"] = [
                                    float(v)
                                    for v in stats["worst_coordinate"]]
                path.write_text(json.dumps(pidx))
            _full_resign(copydir)
            local_t = verify_stage_local(
                copydir, fixtures_r9["D1"][0],
                world_r9.registration_for("D1"), world_r9.decoder,
                production_decoder=world_r9.loss_fn)
            error, out_t = None, None
            try:
                out_t = _run_f_r9(f"r11typed_{variant}_f",
                                  StageBudget("F", 300.0, 10.0),
                                  directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            typed_r11[variant] = (
                bool(local_t["problems"])
                and error is None
                and out_t is not None
                and out_t["summary_doc"]["complete_evidence"] is False
                and out_t["summary_doc"]["verdict"] == "incomplete")
        record("typed_counts_and_coordinates_rejected",
               all(typed_r11.values()),
               "bool/float checkpoint step, true forward counts and float "
               f"worst coordinates are type violations ({typed_r11})")

        # ==================================================================
        # Round-12 review counterexamples
        # ==================================================================
        # --- P1(r12): the ACTUAL AdamW parameter groups are bound ----------
        opt_results: dict = {}
        for field, changed in (("lr", 0.125), ("weight_decay", 0.25)):
            world_o = _r9_world(seed + 70)
            holder_o = {"t": 0.0}
            budget_o = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                   clock=lambda: holder_o["t"])
            _, driver_o, sdir_o = _r9_driver(tmp / f"r12opt_{field}", "D1",
                                             budget_o, world_o)
            for group in driver_o.trainer.opt.param_groups:
                group[field] = changed
            steps_o: list = []
            real_step_o = driver_o.trainer.opt.step

            def observe_step_o(*a, _steps=steps_o, _real=real_step_o, **kw):
                _steps.append(True)
                return _real(*a, **kw)

            with patch.object(trainer_mod, "time",
                              types.SimpleNamespace(
                                  perf_counter=budget_o.clock)), \
                    patch.object(driver_o.trainer.opt, "step",
                                 observe_step_o):
                summary_o = driver_o.run()
            failure_o = summary_o.get("failure") or {}
            out_o = _run_f_r9(f"r12opt_{field}_f",
                              StageBudget("F", 300.0, 10.0),
                              directory=sdir_o)
            opt_results[field] = (
                steps_o == []
                and failure_o.get("type") == "FitDiagnosticError"
                and "optimizer_binding" in failure_o.get("message", "")
                and summary_o.get("local_complete_evidence") is False
                and out_o["summary_doc"]["complete_evidence"] is False)
        record("effective_optimizer_binding_rejected",
               all(opt_results.values()),
               "actual AdamW param-group lr/weight_decay mutations are "
               f"refused before any step ({opt_results})")

        # --- P1(r12): conditioner entry and clip-forensics boundaries -------
        boundary_r12: dict = {}
        for variant in ("none", "predict_input_check", "clip_coverage_pre"):
            world_b = _r9_world(seed + 71)
            holder_b = {"t": 0.0}
            budget_b = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                   clock=lambda: holder_b["t"])
            _, driver_b, _ = _r9_driver(tmp / f"r12bound_{variant}", "D1",
                                        budget_b, world_b)
            counts_b = dict(training_conditioner_calls=0,
                            late_conditioner_calls=0, clip_calls=0,
                            late_clip_calls=0, injections=0)
            real_check_b = driver_b.trainer._check_batch
            real_cond_b = driver_b.trainer.conditioner.forward
            real_cov_b = driver_b._check_clip_coverage_pre
            real_clip_b = driver_b.trainer.clip_gradients

            def check_batch_b(batch, _d=driver_b, _c=counts_b,
                              _holder=holder_b, _real=real_check_b,
                              _inj=variant):
                value = _real(batch)
                if (_inj == "predict_input_check"
                        and sys._getframe(1).f_code is Trainer.predict.__code__
                        and _d.trainer.model.training
                        and _c["injections"] == 0):
                    _holder["t"] = 91.0
                    _c["injections"] += 1
                return value

            def conditioner_b(*a, _d=driver_b, _c=counts_b,
                              _holder=holder_b, _real=real_cond_b, **kw):
                if _d.trainer.conditioner.training:
                    _c["training_conditioner_calls"] += 1
                    _c["late_conditioner_calls"] += int(
                        _holder["t"] >= budget_b.work_deadline)
                return _real(*a, **kw)

            def coverage_b(*a, _c=counts_b, _holder=holder_b,
                           _real=real_cov_b, _inj=variant):
                value = _real(*a)
                if _inj == "clip_coverage_pre":
                    _holder["t"] = 91.0
                    _c["injections"] += 1
                return value

            def clip_b(*a, _c=counts_b, _holder=holder_b,
                       _real=real_clip_b, **kw):
                _c["clip_calls"] += 1
                _c["late_clip_calls"] += int(
                    _holder["t"] >= budget_b.work_deadline)
                return _real(*a, **kw)

            with patch.object(trainer_mod, "time",
                              types.SimpleNamespace(
                                  perf_counter=budget_b.clock)), \
                    patch.object(driver_b.trainer, "_check_batch",
                                 check_batch_b), \
                    patch.object(driver_b.trainer.conditioner, "forward",
                                 conditioner_b), \
                    patch.object(driver_b, "_check_clip_coverage_pre",
                                 coverage_b), \
                    patch.object(driver_b.trainer, "clip_gradients", clip_b):
                summary_b = driver_b.run()
            failure_b = summary_b.get("failure") or {}
            boundary_r12[variant] = {
                **counts_b,
                "failure": failure_b.get("type"),
                "steps": summary_b["counters"]["adamw_step_calls"],
                "complete": summary_b.get("local_complete_evidence")}
        caught = (boundary_r12["none"]["training_conditioner_calls"] == 4
                  and boundary_r12["none"]["clip_calls"] == 1
                  and boundary_r12["none"]["complete"] is True
                  and boundary_r12["predict_input_check"][
                      "late_conditioner_calls"] == 0
                  and boundary_r12["predict_input_check"][
                      "training_conditioner_calls"] == 0
                  and boundary_r12["predict_input_check"]["failure"]
                  == "BudgetExceeded"
                  and boundary_r12["predict_input_check"]["steps"] == 0
                  and boundary_r12["clip_coverage_pre"]["late_clip_calls"] == 0
                  and boundary_r12["clip_coverage_pre"]["clip_calls"] == 0
                  and boundary_r12["clip_coverage_pre"]["failure"]
                  == "BudgetExceeded"
                  and boundary_r12["clip_coverage_pre"]["steps"] == 0)
        record("conditioner_and_clip_pre_boundaries_stop", caught,
               "expiry in predict's input check refuses the conditioner; "
               "expiry in the clip forensics refuses the clip "
               f"({boundary_r12})")

        # --- P2(r12): F fresh probes count entered-then-failing forwards ----
        f_failed_r12: dict = {}
        for variant in ("conditioner", "backbone"):
            actual_f = {"conditioner": 0, "backbone": 0}
            base_factory_f = build_modules_factory(world_r9,
                                                   fixtures_r9["D1"][0])

            def failing_factory_f(_base=base_factory_f, _counts=actual_f,
                                  _variant=variant):
                model_f, cond_f = _base()
                real_cond_f, real_back_f = cond_f.forward, model_f.forward

                def cond_w(*a, **kw):
                    value = real_cond_f(*a, **kw)
                    _counts["conditioner"] += 1
                    if _variant == "conditioner":
                        raise TrainingNumericalError(
                            "synthetic_f_failure_after_conditioner")
                    return value

                def back_w(*a, **kw):
                    value = real_back_f(*a, **kw)
                    _counts["backbone"] += 1
                    if _variant == "backbone":
                        raise TrainingNumericalError(
                            "synthetic_f_failure_after_backbone")
                    return value

                cond_f.forward, model_f.forward = cond_w, back_w
                return model_f, cond_f

            out_f = _run_f_r9(f"r12ff_{variant}_f",
                              StageBudget("F", 300.0, 10.0),
                              factory=failing_factory_f)
            s0_f = out_f["summary_doc"]["stages"][0]
            fc_f = s0_f.get("fresh_counters") or {}
            disk_f = json.loads(
                (tmp / f"r12ff_{variant}_f" / "f_report.json").read_text())
            dfc_f = disk_f["stages"][0].get("fresh_counters") or {}
            key_f = variant + "_forward_calls"
            f_failed_r12[variant] = (
                actual_f[variant] == 1
                and fc_f.get(key_f) == 1
                and dfc_f.get(key_f) == 1
                and out_f["summary_doc"]["complete_evidence"] is False)
        record("f_entered_failed_forwards_counted",
               all(f_failed_r12.values()),
               "F fresh conditioner/backbone entry counters keep executed "
               f"then failing forwards ({f_failed_r12})")

        # --- P2(r12): huge config scalars and bad NPZ dtypes ----------------
        huge_r12: dict = {}
        for variant in ("lr", "w_fk"):
            copydir = tmp / f"r12huge_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            path = copydir / "initial.pt"
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if variant == "lr":
                ck["config"]["lr"] = 10 ** 400
            else:
                ck["config"]["loss"]["w_fk"] = 10 ** 400
            torch.save(ck, path)
            _full_resign(copydir)
            error, out_h = None, None
            try:
                out_h = _run_f_r9(f"r12huge_{variant}_f",
                                  StageBudget("F", 300.0, 10.0),
                                  directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir = tmp / f"r12huge_{variant}_f"
            huge_r12[variant] = (
                error is None
                and (fdir / "f_report.json").is_file()
                and (fdir / "return_observation.json").is_file()
                and out_h is not None
                and out_h["summary_doc"]["verdict"] == "incomplete")
        record("huge_config_scalars_report_incomplete",
               all(huge_r12.values()),
               "10**400 config lr / loss weight are type-strict mismatches, "
               f"never an OverflowError ({huge_r12})")

        bad_npz_r12: dict = {}
        for variant in ("raw", "fk"):
            copydir = tmp / f"r12npz_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            path = copydir / "probes" / "probe_0000_step_000000.npz"
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            key = "w0__raw_x0_pred" if variant == "raw" else "w0__fk_pred"
            arrays[key] = arrays[key].astype("U32")
            np.savez(path, **arrays)
            _full_resign(copydir)
            error, out_n = None, None
            try:
                out_n = _run_f_r9(f"r12npz_{variant}_f",
                                  StageBudget("F", 300.0, 10.0),
                                  directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir = tmp / f"r12npz_{variant}_f"
            bad_npz_r12[variant] = (
                error is None
                and (fdir / "f_report.json").is_file()
                and (fdir / "return_observation.json").is_file()
                and out_n is not None
                and out_n["summary_doc"]["verdict"] == "incomplete")
        record("bad_npz_dtype_reports_incomplete",
               all(bad_npz_r12.values()),
               "string-dtype raw/FK NPZ records are dtype failures, never "
               f"isfinite TypeErrors ({bad_npz_r12})")

        # ==================================================================
        # Round-13 review counterexamples
        # ==================================================================
        # --- P1(r13): the extended AdamW contract and parameter roster -----
        opt_r13: dict = {}
        for variant in ("betas", "eps", "maximize", "omit_conditioner"):
            world_o3 = _r9_world(seed + 75)
            holder_o3 = {"t": 0.0}
            budget_o3 = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                    clock=lambda: holder_o3["t"])
            _, driver_o3, sdir_o3 = _r9_driver(tmp / f"r13opt_{variant}",
                                               "D1", budget_o3, world_o3)
            group_o3 = driver_o3.trainer.opt.param_groups[0]
            if variant == "omit_conditioner":
                group_o3["params"] = list(driver_o3.trainer.model.parameters())
            else:
                group_o3[variant] = {
                    "betas": (0.5, 0.9), "eps": 0.5,
                    "maximize": True}[variant]
            steps_o3: list = []
            real_step_o3 = driver_o3.trainer.opt.step

            def observe_step_o3(*a, _steps=steps_o3, _real=real_step_o3, **kw):
                _steps.append(True)
                return _real(*a, **kw)

            with patch.object(trainer_mod, "time",
                              types.SimpleNamespace(
                                  perf_counter=budget_o3.clock)), \
                    patch.object(driver_o3.trainer.opt, "step",
                                 observe_step_o3):
                summary_o3 = driver_o3.run()
            failure_o3 = summary_o3.get("failure") or {}
            out_o3 = _run_f_r9(f"r13opt_{variant}_f",
                               StageBudget("F", 300.0, 10.0),
                               directory=sdir_o3)
            opt_r13[variant] = (
                steps_o3 == []
                and failure_o3.get("type") == "FitDiagnosticError"
                and "optimizer_binding" in failure_o3.get("message", "")
                and summary_o3.get("local_complete_evidence") is False
                and out_o3["summary_doc"]["complete_evidence"] is False)
        record("optimizer_extended_binding_rejected",
               all(opt_r13.values()),
               "betas/eps/maximize mutations and a dropped conditioner "
               f"parameter roster are refused before any step ({opt_r13})")

        # --- P1(r13): the binding reads themselves are charged --------------
        bind_r13: dict = {}
        for variant in ("none", "clip_binding_read", "step_binding_read"):
            world_b3 = _r9_world(seed + 76)
            holder_b3 = {"t": 0.0}
            budget_b3 = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                    clock=lambda: holder_b3["t"])
            _, driver_b3, _ = _r9_driver(tmp / f"r13bind_{variant}", "D1",
                                         budget_b3, world_b3)
            calls_b3 = dict(bindings_in_clip=0, injections=0, clip=0,
                            late_clip=0, opt_step=0, late_opt_step=0)
            real_bind_b3 = driver_b3._check_runtime_bindings
            real_clip_b3 = driver_b3.trainer.clip_gradients
            real_step_b3 = driver_b3.trainer.opt.step

            def bindings_b3(_real=real_bind_b3, _d=driver_b3, _c=calls_b3,
                            _holder=holder_b3, _inj=variant):
                value = _real()
                if _d._phase == "step_1_clip":
                    _c["bindings_in_clip"] += 1
                    target = {"clip_binding_read": 1,
                              "step_binding_read": 2}.get(_inj)
                    if _c["bindings_in_clip"] == target:
                        _holder["t"] = 91.0
                        _c["injections"] += 1
                return value

            def clip_b3(*a, _c=calls_b3, _holder=holder_b3, _budget=budget_b3,
                        _real=real_clip_b3, **kw):
                _c["clip"] += 1
                _c["late_clip"] += int(
                    _holder["t"] >= _budget.work_deadline)
                return _real(*a, **kw)

            def step_b3(*a, _c=calls_b3, _holder=holder_b3, _budget=budget_b3,
                        _real=real_step_b3, **kw):
                _c["opt_step"] += 1
                _c["late_opt_step"] += int(
                    _holder["t"] >= _budget.work_deadline)
                return _real(*a, **kw)

            with patch.object(trainer_mod, "time",
                              types.SimpleNamespace(
                                  perf_counter=budget_b3.clock)), \
                    patch.object(driver_b3, "_check_runtime_bindings",
                                 bindings_b3), \
                    patch.object(driver_b3.trainer, "clip_gradients",
                                 clip_b3), \
                    patch.object(driver_b3.trainer.opt, "step", step_b3):
                summary_b3 = driver_b3.run()
            failure_b3 = summary_b3.get("failure") or {}
            bind_r13[variant] = {
                **calls_b3,
                "failure": failure_b3.get("type"),
                "steps": summary_b3["counters"]["adamw_step_calls"],
                "complete": summary_b3.get("local_complete_evidence")}
        caught = (bind_r13["none"]["complete"] is True
                  and bind_r13["none"]["late_clip"] == 0
                  and bind_r13["none"]["late_opt_step"] == 0
                  and bind_r13["clip_binding_read"]["clip"] == 0
                  and bind_r13["clip_binding_read"]["late_clip"] == 0
                  and bind_r13["clip_binding_read"]["failure"]
                  == "BudgetExceeded"
                  and bind_r13["clip_binding_read"]["steps"] == 0
                  and bind_r13["step_binding_read"]["opt_step"] == 0
                  and bind_r13["step_binding_read"]["late_opt_step"] == 0
                  and bind_r13["step_binding_read"]["failure"]
                  == "BudgetExceeded"
                  and bind_r13["step_binding_read"]["steps"] == 0)
        record("binding_read_boundaries_stop", caught,
               "elapsed time of the new binding reads is charged before "
               f"clip/AdamW step ({bind_r13})")

        # --- P1(r13): actual checkpoint tensor facts and key-set binding ----
        facts_r13: dict = {}
        for variant in ("none", "nan_parameter", "missing_parameter"):
            copydir = tmp / f"r13ck_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            mutated_key = None
            if variant != "none":
                path = copydir / "last.pt"
                ck = torch.load(path, map_location="cpu", weights_only=False)
                mutated_key = next(
                    key for key, value in ck["model"].items()
                    if value.is_floating_point() and value.numel() > 1)
                if variant == "nan_parameter":
                    ck["model"][mutated_key].reshape(-1)[0] = float("nan")
                else:
                    del ck["model"][mutated_key]
                assert ck["parameters_finite"] is True
                torch.save(ck, path)
                _full_resign(copydir)
            plan_f3 = fixtures_r9["D1"][0]
            reg_f3 = world_r9.registration_for("D1")
            local_f3 = verify_stage_local(
                copydir, plan_f3, reg_f3, world_r9.decoder,
                production_decoder=world_r9.loss_fn,
                expected_identity=_synthetic_stage_identity(plan_f3, reg_f3))
            handoff_f3 = resolve_handoff(
                copydir, plan_f3, world_r9.decoder, reg_f3,
                {"includes": ["D6"]}, StageBudget("D6", 60.0, 5.0),
                expected_identity=_synthetic_stage_identity(plan_f3, reg_f3),
                production_decoder=world_r9.loss_fn)
            facts_r13[variant] = {
                "mutated_key": mutated_key,
                "problems": local_f3["problems"][:4],
                "handoff_ready": handoff_f3["handoff_ready"],
                "handoff_problems": handoff_f3["problems"][:6]}
        caught = (facts_r13["none"]["problems"] == []
                  and any("nonfinite" in p
                          for p in facts_r13["nan_parameter"]["problems"])
                  and facts_r13["nan_parameter"]["handoff_ready"] is False
                  and any("nonfinite" in p
                          for p in facts_r13["nan_parameter"]["handoff_problems"])
                  and any("state_key_set_mismatch" in p
                          for p in facts_r13["missing_parameter"]["problems"])
                  and facts_r13["missing_parameter"]["handoff_ready"] is False
                  and any("state_key_set_mismatch" in p
                          for p in facts_r13["missing_parameter"]["handoff_problems"]))
        record("checkpoint_tensor_facts_bound", caught,
               "actual NaN tensors and a deleted state key are refused by "
               f"local verification and the handoff ({facts_r13})")

        # --- P1(r13): runtime precision re-check inside forward units -------
        def _runtime_snapshot_r13():
            return (torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32,
                    torch.get_float32_matmul_precision())

        def _runtime_restore_r13(state):
            torch.set_float32_matmul_precision(state[2])
            torch.backends.cuda.matmul.allow_tf32 = state[0]
            torch.backends.cudnn.allow_tf32 = state[1]

        runtime_r13: dict = {}
        for role in ("D", "F"):
            for injected in (False, True):
                runtime_state = _runtime_snapshot_r13()
                counts_r13 = dict(injected=0, backbone_under_tf32=0,
                                  backbone=0, conditioner=0)
                try:
                    guard_r13 = ProductionRuntimeGuard(expected_device="cpu")
                    assert guard_r13.apply_problems == []
                    if role == "D":
                        world_rt = _r9_world(seed + 77)
                        holder_rt = {"t": 0.0}
                        budget_rt = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                                clock=lambda: holder_rt["t"])
                        _, driver_rt, _ = _r9_driver(tmp / f"r13rt_D_{injected}",
                                                     "D1", budget_rt, world_rt)
                        driver_rt.runtime_guard = guard_r13
                        real_cond_rt = driver_rt.trainer.conditioner.forward
                        real_back_rt = driver_rt.trainer.model.forward

                        def cond_rt(*a, _c=counts_r13, _real=real_cond_rt,
                                    _inj=injected, _trainer=driver_rt.trainer,
                                    **kw):
                            value = _real(*a, **kw)
                            _c["conditioner"] += 1
                            if (_inj and _c["injected"] == 0
                                    and _trainer.model.training):
                                torch.backends.cuda.matmul.allow_tf32 = True
                                _c["injected"] += 1
                            return value

                        def back_rt(*a, _c=counts_r13, _real=real_back_rt, **kw):
                            _c["backbone"] += 1
                            _c["backbone_under_tf32"] += int(
                                torch.backends.cuda.matmul.allow_tf32)
                            return _real(*a, **kw)

                        driver_rt.trainer.conditioner.forward = cond_rt
                        driver_rt.trainer.model.forward = back_rt
                        with patch.object(trainer_mod, "time",
                                          types.SimpleNamespace(
                                              perf_counter=budget_rt.clock)):
                            summary_rt = driver_rt.run()
                        facts_rt = dict(counts_r13)
                        facts_rt.update({
                            "failure": (summary_rt.get("failure") or {}).get(
                                "type"),
                            "complete": summary_rt.get(
                                "local_complete_evidence")})
                    else:
                        base_rt = build_modules_factory(
                            world_r9, fixtures_r9["D1"][0])

                        def factory_rt(_base=base_rt, _c=counts_r13,
                                       _inj=injected):
                            model_rt, cond_rt2 = _base()
                            real_c2, real_b2 = (cond_rt2.forward,
                                                model_rt.forward)

                            def cw(*a, **kw):
                                value = real_c2(*a, **kw)
                                _c["conditioner"] += 1
                                if _inj and _c["injected"] == 0:
                                    torch.backends.cuda.matmul.allow_tf32 = True
                                    _c["injected"] += 1
                                return value

                            def bw(*a, **kw):
                                _c["backbone"] += 1
                                _c["backbone_under_tf32"] += int(
                                    torch.backends.cuda.matmul.allow_tf32)
                                return real_b2(*a, **kw)

                            cond_rt2.forward, model_rt.forward = cw, bw
                            return model_rt, cond_rt2

                        out_rt = _run_f_r9(
                            f"r13rt_F_{injected}_f",
                            StageBudget("F", 300.0, 10.0),
                            factory=factory_rt, guard=guard_r13)
                        s0_rt = out_rt["summary_doc"]["stages"][0]
                        facts_rt = dict(counts_r13)
                        facts_rt.update({
                            "failure": s0_rt.get("skip_reason"),
                            "complete": out_rt["summary_doc"][
                                "complete_evidence"]})
                    runtime_r13[f"{role}_{injected}"] = facts_rt
                finally:
                    _runtime_restore_r13(runtime_state)
        caught = (runtime_r13["D_False"]["backbone_under_tf32"] == 0
                  and runtime_r13["D_False"]["complete"] is True
                  and runtime_r13["D_True"]["injected"] == 1
                  and runtime_r13["D_True"]["backbone_under_tf32"] == 0
                  and runtime_r13["D_True"]["complete"] is False
                  and runtime_r13["F_False"]["backbone_under_tf32"] == 0
                  and runtime_r13["F_False"]["complete"] is True
                  and runtime_r13["F_True"]["injected"] == 1
                  and runtime_r13["F_True"]["backbone_under_tf32"] == 0
                  and runtime_r13["F_True"]["backbone"] == 0
                  and runtime_r13["F_True"]["complete"] is False)
        record("runtime_drift_between_numeric_units_stop", caught,
               "a TF32 flip after the conditioner is refused at the next "
               f"forward boundary in D and F ({runtime_r13})")

        # --- P2(r13): object NPZ archives and condition contracts -----------
        object_r13: dict = {}
        for variant in ("object_raw", "object_fk"):
            copydir = tmp / f"r13obj_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            path = copydir / "probes" / "probe_0000_step_000000.npz"
            key = ("w0__raw_x0_pred" if variant == "object_raw"
                   else "w0__fk_pred")
            with np.load(path) as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays[key] = arrays[key].astype(object)
            np.savez(path, **arrays)
            _full_resign(copydir)
            error, out_obj = None, None
            try:
                out_obj = _run_f_r9(f"r13obj_{variant}_f",
                                    StageBudget("F", 300.0, 10.0),
                                    directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir = tmp / f"r13obj_{variant}_f"
            object_r13[variant] = (
                error is None
                and (fdir / "f_report.json").is_file()
                and (fdir / "return_observation.json").is_file()
                and out_obj is not None
                and out_obj["summary_doc"]["verdict"] == "incomplete")
        record("object_npz_reports_incomplete",
               all(object_r13.values()),
               "object-dtype NPZ archives are refused without enabling "
               f"pickle and still publish the F report ({object_r13})")

        cond_r13: dict = {}
        for variant in ("unicode_contact", "missing_contact"):
            copydir = tmp / f"r13cond_{variant}"
            shutil.copytree(fixtures_r9["D1"][1], copydir)
            path = copydir / "probe_inputs.npz"
            with np.load(path) as archive:
                arrays = {k: archive[k] for k in archive.files}
            if variant == "missing_contact":
                del arrays["w0__contact"]
            else:
                arrays["w0__contact"] = arrays["w0__contact"].astype("U32")
            np.savez(path, **arrays)
            _full_resign(copydir)
            error, out_c = None, None
            try:
                out_c = _run_f_r9(f"r13cond_{variant}_f",
                                  StageBudget("F", 300.0, 10.0),
                                  directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir = tmp / f"r13cond_{variant}_f"
            cond_r13[variant] = (
                error is None
                and (fdir / "f_report.json").is_file()
                and (fdir / "return_observation.json").is_file()
                and out_c is not None
                and out_c["summary_doc"]["verdict"] == "incomplete")
        record("condition_contract_reports_incomplete",
               all(cond_r13.values()),
               "a missing or string-dtype probe condition is a structured "
               f"contract refusal, never KeyError/TypeError ({cond_r13})")

        # ==================================================================
        # Round-14 review counterexamples
        # ==================================================================
        # --- P1(r14): the live AdamW arithmetic flags are frozen ------------
        opt_flags_r14: dict = {}
        for variant in ("decoupled_weight_decay", "foreach", "fused"):
            world_of = _r9_world(seed + 79)
            holder_of = {"t": 0.0}
            budget_of = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                    clock=lambda: holder_of["t"])
            _, driver_of, _ = _r9_driver(tmp / f"r14opt_{variant}", "D1",
                                         budget_of, world_of)
            group_of = driver_of.trainer.opt.param_groups[0]
            group_of[variant] = (False if variant == "decoupled_weight_decay"
                                 else True)
            steps_of: list = []
            real_step_of = driver_of.trainer.opt.step

            def observe_step_of(*a, _real=real_step_of, _steps=steps_of, **kw):
                _steps.append(1)
                return _real(*a, **kw)

            with patch.object(trainer_mod, "time",
                              types.SimpleNamespace(
                                  perf_counter=budget_of.clock)), \
                    patch.object(driver_of.trainer.opt, "step",
                                 observe_step_of):
                summary_of = driver_of.run()
            failure_of = summary_of.get("failure") or {}
            opt_flags_r14[variant] = (
                steps_of == []
                and failure_of.get("type") == "FitDiagnosticError"
                and "optimizer_binding" in failure_of.get("message", "")
                and summary_of.get("local_complete_evidence") is False)
        record("optimizer_arithmetic_flags_rejected",
               all(opt_flags_r14.values()),
               "decoupled_weight_decay/foreach/fused mutations of the ACTUAL "
               f"AdamW groups are refused before any step ({opt_flags_r14})")

        # --- P1(r14): the runtime read's own time is inside the deadline ----
        runtime_time_r14: dict = {}
        for injected in (False, True):
            world_rt2 = _r9_world(seed + 80)
            holder_rt2 = {"t": 0.0}
            budget_rt2 = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                     clock=lambda: holder_rt2["t"])
            _, driver_rt2, _ = _r9_driver(tmp / f"r14rt_{injected}", "D1",
                                          budget_rt2, world_rt2)
            calls_rt2 = {"injections": 0, "backbone": 0, "late_backbone": 0}
            base_check_rt2 = driver_rt2.runtime_guard.check_unit
            real_back_rt2 = driver_rt2.trainer.model.forward

            def runtime_check_rt2(point, _base=base_check_rt2,
                                  _ct=holder_rt2, _c=calls_rt2,
                                  _inj=injected):
                value = _base(point)
                if (_inj and point == "step_1_micro_1_pre_backbone"
                        and _c["injections"] == 0):
                    _ct["t"] = 91.0
                    _c["injections"] += 1
                return value

            def back_rt2(*a, _c=calls_rt2, _ct=holder_rt2,
                         _bg=budget_rt2, _real=real_back_rt2, **kw):
                _c["backbone"] += 1
                _c["late_backbone"] += int(_ct["t"] >= _bg.work_deadline)
                return _real(*a, **kw)

            with patch.object(driver_rt2.runtime_guard, "check_unit",
                              runtime_check_rt2), \
                    patch.object(driver_rt2.trainer.model, "forward",
                                 back_rt2), \
                    patch.object(trainer_mod, "time",
                                 types.SimpleNamespace(
                                     perf_counter=budget_rt2.clock)):
                summary_rt2 = driver_rt2.run()
            failure_rt2 = summary_rt2.get("failure") or {}
            runtime_time_r14[str(injected)] = {
                **calls_rt2, "failure": failure_rt2.get("type"),
                "message": failure_rt2.get("message", ""),
                "complete": summary_rt2.get("local_complete_evidence")}
        caught = (runtime_time_r14["False"]["late_backbone"] == 0
                  and runtime_time_r14["False"]["complete"] is True
                  and runtime_time_r14["True"]["injections"] == 1
                  and runtime_time_r14["True"]["late_backbone"] == 0
                  and runtime_time_r14["True"]["failure"] == "BudgetExceeded"
                  and "pre_backbone_post_runtime"
                  in runtime_time_r14["True"]["message"]
                  and runtime_time_r14["True"]["complete"] is False)
        record("runtime_check_time_charged", caught,
               "a slow precision read at pre_backbone is charged before the "
               f"backbone is admitted ({runtime_time_r14})")

        f_time_r14: dict = {}
        for injected in (False, True):
            holder_ft = {"t": 0.0}
            budget_ft = StageBudget("F", 100.0, 10.0, t0=0.0,
                                    clock=lambda: holder_ft["t"])
            guard_ft = SyntheticRuntimeGuard()
            calls_ft = {"injections": 0, "conditioner": 0,
                        "late_conditioner": 0}
            base_check_ft = guard_ft.check_unit
            base_factory_ft = build_modules_factory(world_r9,
                                                    fixtures_r9["D1"][0])

            def check_ft(point, _base=base_check_ft, _ct=holder_ft,
                         _c=calls_ft, _inj=injected):
                value = _base(point)
                if (_inj and point == "f_probe_initial_inputs_ready"
                        and _c["injections"] == 0):
                    _ct["t"] = 91.0
                    _c["injections"] += 1
                return value

            def factory_ft(_base=base_factory_ft, _c=calls_ft,
                           _ct=holder_ft, _bg=budget_ft):
                model_ft, cond_ft = _base()
                real_c_ft = cond_ft.forward

                def cond_ft2(*a, **kw):
                    _c["conditioner"] += 1
                    _c["late_conditioner"] += int(
                        _ct["t"] >= _bg.work_deadline)
                    return real_c_ft(*a, **kw)

                cond_ft.forward = cond_ft2
                return model_ft, cond_ft

            with patch.object(guard_ft, "check_unit", check_ft):
                out_ft = _run_f_r9(f"r14ft_{injected}_f", budget_ft,
                                   factory=factory_ft, guard=guard_ft)
            fdir_ft = tmp / f"r14ft_{injected}_f"
            f_time_r14[str(injected)] = {
                **calls_ft,
                "verdict": out_ft["summary_doc"]["verdict"],
                "complete": out_ft["summary_doc"]["complete_evidence"],
                "report": (fdir_ft / "f_report.json").is_file(),
                "proof": (fdir_ft / "return_observation.json").is_file()}
        caught = (f_time_r14["False"]["late_conditioner"] == 0
                  and f_time_r14["False"]["complete"] is True
                  and f_time_r14["True"]["injections"] == 1
                  and f_time_r14["True"]["late_conditioner"] == 0
                  and f_time_r14["True"]["conditioner"] == 0
                  and f_time_r14["True"]["complete"] is False
                  and f_time_r14["True"]["report"]
                  and f_time_r14["True"]["proof"]
                  and f_time_r14["True"]["verdict"] == "incomplete")
        record("f_runtime_check_time_charged", caught,
               "a slow precision read at F inputs_ready is charged before the "
               f"conditioner is admitted ({f_time_r14})")

        # --- P1(r14): checkpoint legality is defined by the FROZEN plan -----
        arch_r14: dict = {}
        plan_ar, sdir_ar = fixtures_r9["D1"]
        ident_ar = _synthetic_stage_identity(
            plan_ar, world_r9.registration_for("D1"))
        for variant in ("same_key_missing_both", "wrong_shape_last",
                        "int32_last"):
            copydir = tmp / f"r14arch_{variant}"
            shutil.copytree(sdir_ar, copydir)
            for endpoint in ("initial.pt", "last.pt"):
                if variant != "same_key_missing_both" \
                        and endpoint != "last.pt":
                    continue
                path = copydir / endpoint
                ck = torch.load(path, map_location="cpu", weights_only=False)
                key = "net.root_in.weight"
                value = ck["model"][key]
                if variant == "same_key_missing_both":
                    del ck["model"][key]
                elif variant == "wrong_shape_last":
                    ck["model"][key] = value.reshape(-1)
                else:
                    ck["model"][key] = value.to(torch.int32)
                torch.save(ck, path)
            _full_resign(copydir)
            local_ar = verify_stage_local(
                copydir, plan_ar, world_r9.registration_for("D1"),
                world_r9.decoder, production_decoder=world_r9.loss_fn,
                expected_identity=ident_ar)
            handoff_ar = resolve_handoff(
                copydir, plan_ar, world_r9.decoder,
                world_r9.registration_for("D1"), {"includes": ["D6"]},
                StageBudget("D6", 120.0, 10.0),
                expected_identity=ident_ar,
                production_decoder=world_r9.loss_fn)
            out_ar = _run_f_r9(f"r14arch_{variant}_f",
                               StageBudget("F", 300.0, 10.0),
                               directory=copydir)
            arch_r14[variant] = {
                "local": local_ar["problems"],
                "mean": local_ar["local_mean_fit_passed"],
                "handoff_ready": handoff_ar["handoff_ready"],
                "handoff": handoff_ar["problems"],
                "f_complete": out_ar["summary_doc"]["complete_evidence"]}
        caught = (
            any("key_set_not_frozen"
                in p for p in arch_r14["same_key_missing_both"]["local"])
            and any("shape_not_frozen"
                    in p for p in arch_r14["wrong_shape_last"]["local"])
            and any("dtype_not_frozen"
                    in p for p in arch_r14["int32_last"]["local"])
            and all(r["mean"] is None and r["handoff_ready"] is False
                    and r["f_complete"] is False
                    for r in arch_r14.values()))
        record("checkpoint_architecture_contract_bound", caught,
               "two endpoints agreeing on a deleted key, a wrong shape or an "
               "int32 tensor are refused against the plan-derived roster "
               f"({ {k: v['local'][:2] for k, v in arch_r14.items()} })")

        # --- P1(r14): conditioner and FK precision admission ----------------
        def _in_fresh_probe() -> bool:
            frame = sys._getframe(1)
            try:
                while frame:
                    if frame.f_code is _fresh_probe.__code__:
                        return True
                    frame = frame.f_back
                return False
            finally:
                del frame

        precision_r14: dict = {}
        for variant in ("after_input_check", "after_loss_decode"):
            runtime_state = _runtime_snapshot_r13()
            counts_p = {"injections": 0, "conditioner_under_tf32": 0,
                        "fk_under_tf32": 0, "conditioner": 0, "fk": 0}
            try:
                guard_p = ProductionRuntimeGuard(expected_device="cpu")
                assert guard_p.apply_problems == []
                world_p = _r9_world(seed + 81)
                holder_p = {"t": 0.0}
                budget_p = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                       clock=lambda: holder_p["t"])
                _, driver_p, _ = _r9_driver(tmp / f"r14prec_{variant}", "D1",
                                            budget_p, world_p)
                driver_p.runtime_guard = guard_p
                real_check_batch_p = driver_p.trainer._check_batch
                real_decode_p = driver_p._real_decode_clip
                real_cond_p = driver_p.trainer.conditioner.forward
                real_fk_p = driver_p._real_fk.site_world_positions

                def maybe_inject_p(_c=counts_p, _d=driver_p):
                    if _d._guard_point == "step_1_micro_1" \
                            and _c["injections"] == 0:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        _c["injections"] += 1

                def check_batch_p(*a, _real=real_check_batch_p,
                                  _inj=variant, **kw):
                    value = _real(*a, **kw)
                    if _inj == "after_input_check":
                        maybe_inject_p()
                    return value

                def decode_p(*a, _real=real_decode_p, _inj=variant, **kw):
                    value = _real(*a, **kw)
                    if _inj == "after_loss_decode":
                        maybe_inject_p()
                    return value

                def cond_p(*a, _c=counts_p, _real=real_cond_p, **kw):
                    _c["conditioner"] += 1
                    _c["conditioner_under_tf32"] += int(
                        torch.backends.cuda.matmul.allow_tf32)
                    return _real(*a, **kw)

                def fk_p(*a, _c=counts_p, _real=real_fk_p, **kw):
                    _c["fk"] += 1
                    _c["fk_under_tf32"] += int(
                        torch.backends.cuda.matmul.allow_tf32)
                    return _real(*a, **kw)

                driver_p.trainer._check_batch = check_batch_p
                driver_p._real_decode_clip = decode_p
                driver_p.trainer.conditioner.forward = cond_p
                with patch.object(driver_p._real_fk, "site_world_positions",
                                  fk_p), \
                        patch.object(trainer_mod, "time",
                                     types.SimpleNamespace(
                                         perf_counter=budget_p.clock)):
                    summary_p = driver_p.run()
                failure_p = summary_p.get("failure") or {}
                precision_r14[variant] = {
                    **counts_p, "failure": failure_p.get("type"),
                    "message": failure_p.get("message", ""),
                    "complete": summary_p.get("local_complete_evidence")}
            finally:
                _runtime_restore_r13(runtime_state)
        caught = (precision_r14["after_input_check"]["injections"] == 1
                  and precision_r14["after_input_check"][
                      "conditioner_under_tf32"] == 0
                  and precision_r14["after_input_check"]["failure"]
                  == "FitDiagnosticError"
                  and "pre_conditioner" in precision_r14["after_input_check"][
                      "message"]
                  and precision_r14["after_input_check"]["complete"] is False
                  and precision_r14["after_loss_decode"]["injections"] == 1
                  and precision_r14["after_loss_decode"][
                      "fk_under_tf32"] == 0
                  and precision_r14["after_loss_decode"]["failure"]
                  == "FitDiagnosticError"
                  and precision_r14["after_loss_decode"]["complete"] is False)
        record("conditioner_and_fk_precision_admission", caught,
               "a TF32 flip after the input check / after the production "
               "decode is refused BEFORE the conditioner / FK call "
               f"({precision_r14})")

        f_precision_r14: dict = {}
        runtime_state = _runtime_snapshot_r13()
        counts_fp = {"injections": 0, "fresh_fk": 0, "fresh_fk_under_tf32": 0}
        try:
            guard_fp = ProductionRuntimeGuard(expected_device="cpu")
            assert guard_fp.apply_problems == []
            base_decode_fp = world_r9.loss_fn.decode_clip
            base_fk_fp = world_r9.loss_fn.fk.site_world_positions

            def decode_fp(*a, _real=base_decode_fp, _c=counts_fp, **kw):
                value = _real(*a, **kw)
                if _in_fresh_probe() and _c["injections"] == 0:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    _c["injections"] += 1
                return value

            def fk_fp(*a, _c=counts_fp, _real=base_fk_fp, **kw):
                if _in_fresh_probe():
                    _c["fresh_fk"] += 1
                    _c["fresh_fk_under_tf32"] += int(
                        torch.backends.cuda.matmul.allow_tf32)
                return _real(*a, **kw)

            with patch.object(world_r9.loss_fn, "decode_clip", decode_fp), \
                    patch.object(world_r9.loss_fn.fk, "site_world_positions",
                                 fk_fp):
                out_fp = _run_f_r9("r14fprec_f",
                                   StageBudget("F", 300.0, 10.0),
                                   guard=guard_fp)
            fdir_fp = tmp / "r14fprec_f"
            f_precision_r14 = {
                **counts_fp,
                "verdict": out_fp["summary_doc"]["verdict"],
                "complete": out_fp["summary_doc"]["complete_evidence"],
                "report": (fdir_fp / "f_report.json").is_file(),
                "proof": (fdir_fp / "return_observation.json").is_file()}
        finally:
            _runtime_restore_r13(runtime_state)
        caught = (f_precision_r14["injections"] == 1
                  and f_precision_r14["fresh_fk_under_tf32"] == 0
                  and f_precision_r14["complete"] is False
                  and f_precision_r14["report"]
                  and f_precision_r14["proof"]
                  and f_precision_r14["verdict"] == "incomplete")
        record("f_fresh_fk_precision_admission", caught,
               "a TF32 flip after the F fresh production decode refuses the "
               f"FK call and still publishes the report ({f_precision_r14})")

        # --- P2(r14): CRC-corrupted NPZ members are structured refusals -----
        def _corrupt_member_crc(path: Path, key: str) -> None:
            with zipfile.ZipFile(path) as zf:
                info = zf.getinfo(key + ".npy")
                assert info.compress_type == zipfile.ZIP_STORED
            data = bytearray(path.read_bytes())
            namelen, extralen = struct.unpack_from(
                "<HH", data, info.header_offset + 26)
            payload = info.header_offset + 30 + namelen + extralen
            data[payload + info.file_size - 1] ^= 1
            path.write_bytes(data)

        crc_r14: dict = {}
        for variant in ("probe_raw_crc", "condition_contact_crc"):
            copydir = tmp / f"r14crc_{variant}"
            shutil.copytree(sdir_ar, copydir)
            if variant == "probe_raw_crc":
                path = copydir / "probes" / "probe_0000_step_000000.npz"
                key = "w0__raw_x0_pred"
            else:
                path = copydir / "probe_inputs.npz"
                key = "w0__contact"
            _corrupt_member_crc(path, key)
            _full_resign(copydir)
            error, out_crc = None, None
            try:
                out_crc = _run_f_r9(f"r14crc_{variant}_f",
                                    StageBudget("F", 300.0, 10.0),
                                    directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir_crc = tmp / f"r14crc_{variant}_f"
            crc_r14[variant] = (
                error is None
                and (fdir_crc / "f_report.json").is_file()
                and (fdir_crc / "return_observation.json").is_file()
                and out_crc is not None
                and out_crc["summary_doc"]["verdict"] == "incomplete")
        record("crc_corrupt_npz_reports_incomplete",
               all(crc_r14.values()),
               "CRC-corrupted raw/condition NPZ members are structural "
               f"refusals that still publish the F report ({crc_r14})")

        # ==================================================================
        # Round-15 review counterexamples
        # ==================================================================
        # --- P1(r15): the from-scratch step-0 optimizer state is frozen -----
        opt_state_r15: dict = {}
        for injected in (False, True):
            world_os = _r9_world(seed + 91)
            holder_os = {"t": 0.0}
            budget_os = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                    clock=lambda: holder_os["t"])
            _, driver_os, _ = _r9_driver(tmp / f"r15optstate_{injected}", "D1",
                                         budget_os, world_os)
            target_os = driver_os.trainer._named_parameters()[0][1]
            if injected:
                driver_os.trainer.opt.state[target_os] = {
                    "step": torch.tensor(17.0),
                    "exp_avg": torch.full_like(target_os, 0.1),
                    "exp_avg_sq": torch.full_like(target_os, 0.01)}
            steps_os: list = []
            real_step_os = driver_os.trainer.opt.step

            def observe_step_os(*a, _real=real_step_os, _steps=steps_os, **kw):
                _steps.append(1)
                return _real(*a, **kw)

            with patch.object(driver_os.trainer.opt, "step",
                              observe_step_os), \
                    patch.object(trainer_mod, "time",
                                 types.SimpleNamespace(
                                     perf_counter=budget_os.clock)):
                summary_os = driver_os.run()
            failure_os = summary_os.get("failure") or {}
            opt_state_r15[str(injected)] = {
                "steps": len(steps_os),
                "failure": failure_os.get("type"),
                "message": failure_os.get("message", ""),
                "complete": summary_os.get("local_complete_evidence")}
        caught = (opt_state_r15["False"]["steps"] == 1
                  and opt_state_r15["False"]["complete"] is True
                  and opt_state_r15["True"]["steps"] == 0
                  and opt_state_r15["True"]["failure"] == "FitDiagnosticError"
                  and "optimizer_state" in opt_state_r15["True"]["message"]
                  and opt_state_r15["True"]["complete"] is False)
        record("optimizer_initial_state_bound", caught,
               "a preloaded momentum/step-17 state is refused at preflight "
               f"before any update ({opt_state_r15})")

        # --- P1(r15): the new decode/FK runtime reads are time too ----------
        d_time_r15: dict = {}
        for point in ("none", "step_1_micro_1_loss_decode_pre",
                      "step_1_micro_1_loss_fk_pre"):
            world_dt = _r9_world(seed + 92)
            holder_dt = {"t": 0.0}
            budget_dt = StageBudget("D1", 100.0, 10.0, t0=0.0,
                                    clock=lambda: holder_dt["t"])
            _, driver_dt, _ = _r9_driver(tmp / f"r15dtime_{point}", "D1",
                                         budget_dt, world_dt)
            calls_dt = {"injections": 0, "decode": 0, "late_decode": 0,
                        "fk": 0, "late_fk": 0}
            real_check_dt = driver_dt.runtime_guard.check_unit
            real_decode_dt = driver_dt._real_decode_clip
            real_fk_dt = driver_dt._real_fk.site_world_positions

            def check_dt(where, _base=real_check_dt, _point=point,
                         _ct=holder_dt, _c=calls_dt):
                value = _base(where)
                if where == _point and _c["injections"] == 0:
                    _ct["t"] = 91.0
                    _c["injections"] += 1
                return value

            def decode_dt(*a, _real=real_decode_dt, _c=calls_dt,
                          _ct=holder_dt, _bg=budget_dt, **kw):
                _c["decode"] += 1
                _c["late_decode"] += int(
                    _ct["t"] >= _bg.work_deadline)
                return _real(*a, **kw)

            def fk_dt(*a, _real=real_fk_dt, _c=calls_dt, _ct=holder_dt,
                      _bg=budget_dt, **kw):
                _c["fk"] += 1
                _c["late_fk"] += int(_ct["t"] >= _bg.work_deadline)
                return _real(*a, **kw)

            driver_dt._real_decode_clip = decode_dt
            with patch.object(driver_dt.runtime_guard, "check_unit",
                              check_dt), \
                    patch.object(driver_dt._real_fk, "site_world_positions",
                                 fk_dt), \
                    patch.object(trainer_mod, "time",
                                 types.SimpleNamespace(
                                     perf_counter=budget_dt.clock)):
                summary_dt = driver_dt.run()
            failure_dt = summary_dt.get("failure") or {}
            d_time_r15[point] = {
                **calls_dt, "failure": failure_dt.get("type"),
                "message": failure_dt.get("message", ""),
                "complete": summary_dt.get("local_complete_evidence")}
        caught = (d_time_r15["none"]["late_decode"] == 0
                  and d_time_r15["none"]["late_fk"] == 0
                  and d_time_r15["none"]["complete"] is True
                  and d_time_r15["step_1_micro_1_loss_decode_pre"][
                      "injections"] == 1
                  and d_time_r15["step_1_micro_1_loss_decode_pre"][
                      "late_decode"] == 0
                  and d_time_r15["step_1_micro_1_loss_decode_pre"][
                      "failure"] == "BudgetExceeded"
                  and "loss_decode_pre_post_runtime"
                  in d_time_r15["step_1_micro_1_loss_decode_pre"]["message"]
                  and d_time_r15["step_1_micro_1_loss_fk_pre"][
                      "injections"] == 1
                  and d_time_r15["step_1_micro_1_loss_fk_pre"][
                      "late_fk"] == 0
                  and d_time_r15["step_1_micro_1_loss_fk_pre"][
                      "failure"] == "BudgetExceeded"
                  and "loss_fk_pre_post_runtime"
                  in d_time_r15["step_1_micro_1_loss_fk_pre"]["message"])
        record("decode_fk_runtime_read_time_charged", caught,
               "a slow precision read at the decode/FK admissions is charged "
               f"before the numeric unit starts ({d_time_r15})")

        f_time_r15: dict = {}
        for point in ("none", "f_fresh_entry", "f_fresh_fk_w0"):
            holder_ft2 = {"t": 0.0}
            budget_ft2 = StageBudget("F", 100.0, 10.0, t0=0.0,
                                     clock=lambda: holder_ft2["t"])
            guard_ft2 = SyntheticRuntimeGuard()
            calls_ft2 = {"injections": 0, "conditioner": 0,
                         "late_conditioner": 0, "fresh_fk": 0,
                         "late_fresh_fk": 0}
            base_check_ft2 = guard_ft2.check_unit
            base_factory_ft2 = build_modules_factory(world_r9,
                                                     fixtures_r9["D1"][0])
            base_fk_ft2 = world_r9.loss_fn.fk.site_world_positions

            def check_ft2(where, _base=base_check_ft2, _point=point,
                          _ct=holder_ft2, _c=calls_ft2):
                value = _base(where)
                if where == _point and _c["injections"] == 0:
                    _ct["t"] = 91.0
                    _c["injections"] += 1
                return value

            def factory_ft2(_base=base_factory_ft2, _c=calls_ft2,
                            _ct=holder_ft2, _bg=budget_ft2):
                model_ft2, cond_ft2 = _base()
                real_c_ft2 = cond_ft2.forward

                def cond_ft3(*a, **kw):
                    _c["conditioner"] += 1
                    _c["late_conditioner"] += int(
                        _ct["t"] >= _bg.work_deadline)
                    return real_c_ft2(*a, **kw)

                cond_ft2.forward = cond_ft3
                return model_ft2, cond_ft2

            def fk_ft2(*a, _real=base_fk_ft2, _c=calls_ft2,
                       _ct=holder_ft2, _bg=budget_ft2, **kw):
                if _in_fresh_probe():
                    _c["fresh_fk"] += 1
                    _c["late_fresh_fk"] += int(
                        _ct["t"] >= _bg.work_deadline)
                return _real(*a, **kw)

            with patch.object(guard_ft2, "check_unit", check_ft2), \
                    patch.object(world_r9.loss_fn.fk, "site_world_positions",
                                 fk_ft2):
                out_ft2 = _run_f_r9(f"r15ftime_{point}_f", budget_ft2,
                                    factory=factory_ft2, guard=guard_ft2)
            fdir_ft2 = tmp / f"r15ftime_{point}_f"
            f_time_r15[point] = {
                **calls_ft2,
                "verdict": out_ft2["summary_doc"]["verdict"],
                "complete": out_ft2["summary_doc"]["complete_evidence"],
                "report": (fdir_ft2 / "f_report.json").is_file(),
                "proof": (fdir_ft2 / "return_observation.json").is_file()}
        caught = (f_time_r15["none"]["late_conditioner"] == 0
                  and f_time_r15["none"]["late_fresh_fk"] == 0
                  and f_time_r15["none"]["complete"] is True
                  and f_time_r15["f_fresh_entry"]["injections"] == 1
                  and f_time_r15["f_fresh_entry"]["late_conditioner"] == 0
                  and f_time_r15["f_fresh_entry"]["conditioner"] == 0
                  and f_time_r15["f_fresh_entry"]["complete"] is False
                  and f_time_r15["f_fresh_fk_w0"]["injections"] == 1
                  and f_time_r15["f_fresh_fk_w0"]["late_fresh_fk"] == 0
                  and f_time_r15["f_fresh_fk_w0"]["fresh_fk"] == 0
                  and f_time_r15["f_fresh_fk_w0"]["complete"] is False
                  and all(r["report"] and r["proof"]
                          and r["verdict"] == "incomplete"
                          for r in (f_time_r15["f_fresh_entry"],
                                    f_time_r15["f_fresh_fk_w0"])))
        record("f_fresh_runtime_read_time_charged", caught,
               "slow precision reads at F's fresh entry/FK admission are "
               f"charged before the conditioner/FK starts ({f_time_r15})")

        # --- P1(r15): the reference construction is budgeted and skipped on
        # an already-refused identity ---------------------------------------
        def _inside_code(code) -> bool:
            frame = sys._getframe(1)
            try:
                while frame:
                    if frame.f_code is code:
                        return True
                    frame = frame.f_back
                return False
            finally:
                del frame

        from ireset.model.conditioner import ConditionEncoder
        from ireset.model.deterministic import DeterministicG1
        real_model_init = DeterministicG1.__init__
        real_cond_init = ConditionEncoder.__init__
        contract_r15: dict = {}
        for variant in ("none", "model_returns_late", "identity_bad"):
            holder_sc = {"t": 0.0}
            budget_sc = StageBudget("F", 100.0, 10.0, t0=0.0,
                                    clock=lambda: holder_sc["t"])
            events_sc: list = []
            _STATE_CONTRACT_CACHE.clear()
            before_rng_sc = torch.get_rng_state().clone()

            def model_init_sc(obj, *a, _base=real_model_init,
                              _variant=variant, _ct=holder_sc,
                              _events=events_sc, **kw):
                metadata = _inside_code(_plan_fresh_modules.__code__)
                _events.append({"unit": "model", "metadata": metadata,
                                "start": _ct["t"]})
                _base(obj, *a, **kw)
                if metadata and _variant == "model_returns_late":
                    _ct["t"] = 91.0

            def cond_init_sc(obj, *a, _base=real_cond_init,
                             _ct=holder_sc, _events=events_sc, **kw):
                _events.append({"unit": "conditioner",
                                "metadata": _inside_code(
                                    _plan_fresh_modules.__code__),
                                "start": _ct["t"]})
                return _base(obj, *a, **kw)

            ident_sc = dict(_synthetic_stage_identity(
                plan_ar, world_r9.registration_for("D1")))
            if variant == "identity_bad":
                ident_sc["implementation_sha256"] = "b" * 64
            with patch.object(DeterministicG1, "__init__", model_init_sc), \
                    patch.object(ConditionEncoder, "__init__", cond_init_sc):
                out_sc = run_f_verification(
                    stage_specs=[{
                        "stage_dir": sdir_ar, "plan": plan_ar,
                        "registration": world_r9.registration_for("D1"),
                        "decoder": world_r9.decoder,
                        "loss_fn": world_r9.loss_fn,
                        "identity": ident_sc}],
                    out_dir=tmp / f"r15sc_{variant}",
                    budget=budget_sc,
                    runtime_guard=SyntheticRuntimeGuard(),
                    build_modules=build_modules_factory(world_r9, plan_ar),
                    authorization={"includes": ["D6"]})
            s0_sc = out_sc["summary_doc"]["stages"][0]
            contract_r15[variant] = {
                "events": events_sc,
                "rng_unchanged": torch.equal(before_rng_sc,
                                             torch.get_rng_state()),
                "module_builds": 1 if any(not e["metadata"]
                                          for e in events_sc) else 0,
                "complete": out_sc["summary_doc"]["complete_evidence"],
                "skip": s0_sc.get("skip_reason"),
                "local": s0_sc.get("local_reverify_problems") or []}
        caught = (
            [e["unit"] for e in contract_r15["none"]["events"]]
            == ["model", "conditioner"] * 2
            and contract_r15["none"]["complete"] is True
            and [e["unit"] for e in contract_r15["model_returns_late"]["events"]]
            == ["model"]
            and contract_r15["model_returns_late"]["complete"] is False
            and contract_r15["model_returns_late"]["rng_unchanged"] is True
            and contract_r15["model_returns_late"]["module_builds"] == 0
            and contract_r15["identity_bad"]["events"] == []
            and contract_r15["identity_bad"]["complete"] is False
            and contract_r15["identity_bad"]["rng_unchanged"] is True
            and contract_r15["identity_bad"]["module_builds"] == 0
            and any("stage_report_identity_mismatch" in p
                    for p in contract_r15["identity_bad"]["local"]))
        record("state_contract_construction_guarded", caught,
               "a model construction returning past the deadline refuses the "
               "conditioner; a refused identity constructs nothing "
               f"({ {k: [e['unit'] for e in v['events']] for k, v in contract_r15.items()} })")

        # --- P2(r15): mixed-primitive checkpoint keys are a structured fault -
        mixed_r15: dict = {}
        for injected in (False, True):
            copydir = tmp / f"r15mixed_{injected}"
            shutil.copytree(sdir_ar, copydir)
            if injected:
                path = copydir / "last.pt"
                ck = torch.load(path, map_location="cpu", weights_only=False)
                ck["model"][17] = torch.zeros(1, dtype=torch.float32)
                ck["model"]["unexpected"] = torch.zeros(1,
                                                        dtype=torch.float32)
                torch.save(ck, path)
                _full_resign(copydir)
            error, out_m = None, None
            try:
                out_m = _run_f_r9(f"r15mixed_{injected}_f",
                                  StageBudget("F", 300.0, 10.0),
                                  directory=copydir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            fdir_m = tmp / f"r15mixed_{injected}_f"
            local_m = ((out_m["summary_doc"]["stages"][0].get(
                "local_reverify_problems") or []) if out_m is not None else [])
            mixed_r15[str(injected)] = (
                error is None
                and (fdir_m / "f_report.json").is_file()
                and (fdir_m / "return_observation.json").is_file()
                and out_m is not None
                and ((not injected
                      and out_m["summary_doc"]["complete_evidence"] is True)
                     or (injected
                         and out_m["summary_doc"]["verdict"] == "incomplete"
                         and any("key_type_not_string" in p
                                 for p in local_m))))
        record("mixed_checkpoint_keys_report_incomplete",
               all(mixed_r15.values()),
               "mixed int/str checkpoint keys are refused before any ordering "
               f"and still publish the F report ({mixed_r15})")

        # A frozen parameter remains in AdamW's roster but has no gradient,
        # state or update; membership alone must not make D complete.
        trainable_r16: dict = {}
        for variant, name in (("control", None),
                              ("model", "model.net.root_out.weight"),
                              ("conditioner", "conditioner.kpt_proj.weight")):
            world_tr = _r9_world(seed + 101)
            clock_tr = _FakeClock(1.0)
            budget_tr = StageBudget("D1", 100.0, 10.0, t0=0.0, clock=clock_tr)
            _, driver_tr, _ = _r9_driver(tmp / f"r16trainable_{variant}",
                                         "D1", budget_tr, world_tr)
            if name is not None:
                dict(driver_tr.trainer._named_parameters())[name].requires_grad_(False)
            with patch.object(trainer_mod, "time",
                              types.SimpleNamespace(perf_counter=clock_tr)):
                summary_tr = driver_tr.run()
            trainable_r16[variant] = {
                "complete": summary_tr["local_complete_evidence"],
                "steps": summary_tr["counters"]["adamw_step_calls"],
                "forwards": summary_tr["counters"]["conditioner_forward_calls"],
                "message": (summary_tr.get("failure") or {}).get("message", "")}
        caught = (trainable_r16["control"]["complete"] is True
                  and trainable_r16["control"]["steps"] == 1
                  and all(not trainable_r16[v]["complete"]
                          and trainable_r16[v]["steps"] == 0
                          and trainable_r16[v]["forwards"] == 0
                          and "optimizer_parameter_frozen" in trainable_r16[v]["message"]
                          for v in ("model", "conditioner")))
        record("trainable_parameter_roster_bound", caught,
               f"frozen model/conditioner parameter refuses before forward ({trainable_r16})")

        # The runtime check immediately before backward can itself cross the
        # work deadline. Count the actual backward entry, not a later summary.
        backward_r16: dict = {}
        for injected in (False, True):
            world_bw = _r9_world(seed + 102)
            clock_bw = _FakeClock(1.0)
            budget_bw = StageBudget("D1", 100.0, 10.0, t0=0.0, clock=clock_bw)
            _, driver_bw, _ = _r9_driver(tmp / f"r16backward_{injected}",
                                         "D1", budget_bw, world_bw)
            calls_bw = {"injections": 0, "backward": 0, "late": 0}
            base_check_bw = driver_bw.runtime_guard.check_unit
            base_backward_bw = torch.Tensor.backward

            def check_bw(point, _base=base_check_bw, _calls=calls_bw,
                         _clock=clock_bw, _injected=injected):
                value = _base(point)
                if _injected and point == "step_1_micro_1_pre_backward":
                    _calls["injections"] += 1
                    _clock.t = 91.0
                return value

            def backward_bw(tensor, *a, _base=base_backward_bw,
                            _calls=calls_bw, _clock=clock_bw,
                            _budget=budget_bw, **kw):
                _calls["backward"] += 1
                _calls["late"] += int(_clock.t >= _budget.work_deadline)
                return _base(tensor, *a, **kw)

            with patch.object(driver_bw.runtime_guard, "check_unit", check_bw), \
                    patch.object(torch.Tensor, "backward", backward_bw), \
                    patch.object(trainer_mod, "time",
                                 types.SimpleNamespace(perf_counter=clock_bw)):
                summary_bw = driver_bw.run()
            backward_r16[str(injected)] = {
                **calls_bw, "complete": summary_bw["local_complete_evidence"],
                "recorded": summary_bw["counters"]["backward_calls"],
                "steps": summary_bw["counters"]["adamw_step_calls"],
                "message": (summary_bw.get("failure") or {}).get("message", "")}
        caught = (backward_r16["False"]["complete"] is True
                  and backward_r16["False"]["backward"] == 4
                  and backward_r16["True"]["complete"] is False
                  and backward_r16["True"]["injections"] == 1
                  and backward_r16["True"]["backward"] == 0
                  and backward_r16["True"]["late"] == 0
                  and backward_r16["True"]["recorded"] == 0
                  and backward_r16["True"]["steps"] == 0
                  and "pre_backward_post_runtime" in backward_r16["True"]["message"])
        record("pre_backward_runtime_read_time_charged", caught,
               f"runtime read crossing the deadline refuses backward ({backward_r16})")

        # Teacher references precede the guarded probe/forward chain. Check
        # real runtime drift and slow runtime reads at both numeric units.
        teacher_r16: dict = {}
        for variant in ("control", "drift_after_decode", "slow_decode_gate",
                        "slow_fk_gate"):
            runtime_before = _runtime_snapshot_r13()
            try:
                world_te = _r9_world(seed + 103)
                clock_te = _FakeClock(1.0)
                budget_te = StageBudget("D1", 100.0, 10.0, t0=0.0, clock=clock_te)
                _, driver_te, _ = _r9_driver(tmp / f"r16teacher_{variant}",
                                             "D1", budget_te, world_te)
                driver_te.runtime_guard = ProductionRuntimeGuard(expected_device="cpu")
                calls_te = {"injections": 0, "decode": 0, "fk": 0,
                            "late_decode": 0, "late_fk": 0, "bad_fk": 0}
                base_check_te = driver_te.runtime_guard.check_unit
                base_decode_te = driver_te._real_decode_clip
                base_fk_te = driver_te._real_fk.site_world_positions

                def check_te(point, _base=base_check_te, _v=variant,
                             _clock=clock_te, _calls=calls_te):
                    value = _base(point)
                    target = {"slow_decode_gate": "teacher_window_w0_prod_decode",
                              "slow_fk_gate": "teacher_window_w0_fk"}.get(_v)
                    if point == target and _calls["injections"] == 0:
                        _calls["injections"] += 1
                        _clock.t = 91.0
                    return value

                def decode_te(*a, _base=base_decode_te, _v=variant,
                              _clock=clock_te, _budget=budget_te,
                              _calls=calls_te, **kw):
                    _calls["decode"] += 1
                    _calls["late_decode"] += int(_clock.t >= _budget.work_deadline)
                    out = _base(*a, **kw)
                    if _v == "drift_after_decode" and _calls["injections"] == 0:
                        _calls["injections"] += 1
                        torch.backends.cuda.matmul.allow_tf32 = True
                    return out

                def fk_te(*a, _base=base_fk_te, _clock=clock_te,
                          _budget=budget_te, _calls=calls_te, **kw):
                    _calls["fk"] += 1
                    _calls["late_fk"] += int(_clock.t >= _budget.work_deadline)
                    _calls["bad_fk"] += int(torch.backends.cuda.matmul.allow_tf32)
                    return _base(*a, **kw)

                driver_te._real_decode_clip = decode_te
                with patch.object(driver_te.runtime_guard, "check_unit", check_te), \
                        patch.object(driver_te._real_fk, "site_world_positions", fk_te), \
                        patch.object(trainer_mod, "time",
                                     types.SimpleNamespace(perf_counter=clock_te)):
                    summary_te = driver_te.run()
                teacher_r16[variant] = {
                    **calls_te, "complete": summary_te["local_complete_evidence"],
                    "recorded_decode": summary_te["counters"]["production_decode_calls"],
                    "recorded_fk": summary_te["counters"]["production_fk_batch_calls"],
                    "message": (summary_te.get("failure") or {}).get("message", "")}
            finally:
                _runtime_restore_r13(runtime_before)
        caught = (teacher_r16["control"]["complete"] is True
                  and teacher_r16["drift_after_decode"]["injections"] == 1
                  and teacher_r16["drift_after_decode"]["fk"] == 0
                  and teacher_r16["drift_after_decode"]["bad_fk"] == 0
                  and "runtime_drift" in teacher_r16["drift_after_decode"]["message"]
                  and teacher_r16["slow_decode_gate"]["decode"] == 0
                  and teacher_r16["slow_decode_gate"]["recorded_decode"] == 0
                  and teacher_r16["slow_fk_gate"]["fk"] == 0
                  and teacher_r16["slow_fk_gate"]["recorded_fk"] == 0
                  and all(not teacher_r16[v]["complete"]
                          and teacher_r16[v]["injections"] == 1
                          and teacher_r16[v]["late_decode"] == 0
                          and teacher_r16[v]["late_fk"] == 0
                          for v in ("drift_after_decode", "slow_decode_gate",
                                    "slow_fk_gate")))
        record("teacher_reference_precision_admission", caught,
               f"teacher decode/FK have runtime and charged deadline gates ({teacher_r16})")

        # A missing/non-object identity has already failed the identity
        # gate, just like a mismatched identity. A cold contract cache must
        # not build reference modules on any of those paths.
        identity_r16: dict = {}
        for variant in ("control", "missing", "list", "null"):
            copydir = tmp / f"r16identity_{variant}"
            shutil.copytree(sdir_ar, copydir)
            if variant != "control":
                report_path = copydir / "stage_report.json"
                report = json.loads(report_path.read_text())
                if variant == "missing":
                    del report["identity"]
                else:
                    report["identity"] = [] if variant == "list" else None
                report_path.write_text(json.dumps(report))
                _full_resign(copydir)
            _STATE_CONTRACT_CACHE.clear()
            before_rng = torch.get_rng_state().clone()
            events: list = []

            def model_init_id(obj, *a, _base=real_model_init,
                              _events=events, **kw):
                _events.append(("model", _inside_code(_plan_fresh_modules.__code__)))
                return _base(obj, *a, **kw)

            def cond_init_id(obj, *a, _base=real_cond_init,
                             _events=events, **kw):
                _events.append(("conditioner", _inside_code(_plan_fresh_modules.__code__)))
                return _base(obj, *a, **kw)

            with patch.object(DeterministicG1, "__init__", model_init_id), \
                    patch.object(ConditionEncoder, "__init__", cond_init_id):
                out_id = _run_f_r9(f"r16identity_{variant}_f",
                                   StageBudget("F", 300.0, 10.0),
                                   directory=copydir)
            doc_id = out_id["summary_doc"]
            fdir_id = tmp / f"r16identity_{variant}_f"
            identity_r16[variant] = {
                "events": events, "rng_unchanged": torch.equal(before_rng, torch.get_rng_state()),
                "complete": doc_id["complete_evidence"], "verdict": doc_id["verdict"],
                "local": doc_id["stages"][0].get("local_reverify_problems", []),
                "report": (fdir_id / "f_report.json").is_file(),
                "proof": (fdir_id / "return_observation.json").is_file()}
        caught = (identity_r16["control"]["complete"] is True
                  and len(identity_r16["control"]["events"]) == 4
                  and all(identity_r16[v]["events"] == []
                          and identity_r16[v]["rng_unchanged"]
                          and identity_r16[v]["complete"] is False
                          and identity_r16[v]["verdict"] == "incomplete"
                          and identity_r16[v]["report"] and identity_r16[v]["proof"]
                          and "stage_report_identity_invalid" in identity_r16[v]["local"]
                          for v in ("missing", "list", "null")))
        record("bad_report_identity_no_reference_construction", caught,
               f"bad report identities refuse cold reference construction ({identity_r16})")

        sparse_r16: dict = {}
        for variant in ("control", "model_last", "conditioner_initial"):
            copydir = tmp / f"r16sparse_{variant}"
            shutil.copytree(sdir_ar, copydir)
            if variant != "control":
                side = "model" if variant == "model_last" else "conditioner"
                key = "net.root_out.weight" if side == "model" else "kpt_proj.weight"
                path = copydir / ("last.pt" if side == "model" else "initial.pt")
                ck = torch.load(path, map_location="cpu", weights_only=False)
                ck[side][key] = ck[side][key].to_sparse()
                torch.save(ck, path)
                _full_resign(copydir)
            builds: list = []
            base_factory = build_modules_factory(world_r9, plan_ar)

            def factory_sparse(_base=base_factory, _builds=builds):
                _builds.append(1)
                return _base()

            out_sp = _run_f_r9(f"r16sparse_{variant}_f",
                               StageBudget("F", 300.0, 10.0),
                               directory=copydir, factory=factory_sparse)
            doc_sp = out_sp["summary_doc"]
            fdir_sp = tmp / f"r16sparse_{variant}_f"
            sparse_r16[variant] = {
                "builds": len(builds), "complete": doc_sp["complete_evidence"],
                "verdict": doc_sp["verdict"],
                "local": doc_sp["stages"][0].get("local_reverify_problems", []),
                "report": (fdir_sp / "f_report.json").is_file(),
                "proof": (fdir_sp / "return_observation.json").is_file()}
        caught = (sparse_r16["control"]["complete"] is True
                  and sparse_r16["control"]["builds"] == 1
                  and all(sparse_r16[v]["complete"] is False
                          and sparse_r16[v]["builds"] == 0
                          and sparse_r16[v]["verdict"] == "incomplete"
                          and sparse_r16[v]["report"] and sparse_r16[v]["proof"]
                          and any("checkpoint_tensor_layout" in p
                                  for p in sparse_r16[v]["local"])
                          for v in ("model_last", "conditioner_initial")))
        record("sparse_checkpoint_layout_report_incomplete", caught,
               f"non-strided tensors refuse before isfinite/load ({sparse_r16})")

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
