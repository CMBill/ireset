"""W1 real network-parameter gradient and optimizer-update audit (I/C).

Implementation of ``docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md``
(§10.1 I/C scope).  The module has two entry modes:

* ``synthetic`` -- a pure-synthetic analytic fixture: the full production
  ``DeterministicG1`` / ``DiffusionG1`` / ``ConditionEncoder`` architecture is
  constructed from a fixed random seed with a bounded output-head gain, the
  four registered windows come from analytic worlds, and the FK is the
  audited 29-joint/15-site synthetic chain.  The chain exercises
  R0 -> P -> I -> F, the full parameter roster, the ordered micro/noise
  matrix, accumulation, joint clip, effective-float32 AdamW and the
  cross-path sensitivity classification.  No real checkpoint, dataset, cache,
  G1 asset or MuJoCo model is read.
* ``real`` -- the registered path.  It refuses before any real array read or
  ``torch.load`` unless a hash-bound authorization record (plan section 10.2,
  all ten boxes unchecked in this repository state), a frozen synthetic
  calibration/fp32 contract and a passed admission package are supplied.

Even a passing covered contract is ``objective_coverage="partial"``:
``full_objective_contract_passed=None``, ``factor_selected=False``,
``auto_execute=False``, ``final_interaction_method="not_ready"`` and
``production_integration=False``.  Selection, DDIM, validation, parameter FD
and production checkpoint writes are hard zero counters.

No disk access happens at import time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch

from ireset.diffusion.generator import DiffusionG1
from ireset.eval.training_objective_audit import (
    SyntheticWorld,
    build_synthetic_world,
    make_synthetic_batch,
)
from ireset.eval.w1_real_fk_audit import (
    DOMAIN as FK_DOMAIN,
    GROUP_GATE,
    SITE_NAMES,
    SyntheticChain,
    _SyntheticFkAdapter,
    _canonical_sha256,
    _hash_array,
    _jsonable,
    _num,
    _count,
    _nonneg,
    build_synthetic_chain,
    configure_production_runtime,
    RUNTIME_REQUIREMENTS,
    decode_jacobian_np,
    effective_constants,
    environment_facts,
    fk_loss_and_gradient,
    independent_decode,
    read_json,
    runtime_facts_problems,
    sha256_bytes,
    sha256_file,
    synthetic_fk_jac_np,
    synthetic_fk_np,
    write_json,
)
from ireset.eval.w1_real_objective_audit import (
    independent_scalars,
    independent_term_elements,
    independent_vjp,
)
from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1
from ireset.model.tokens import RootStats, StateTokenizer
from ireset.paths import PROJECT_ROOT, path_for_record
from ireset.train.losses import LossAssembler, LossWeights
from ireset.train.trainer import (
    TrainConfig,
    Trainer,
    _optimizer_state_snapshot,
)

# ---------------------------------------------------------------------------
# identity and frozen matrix
# ---------------------------------------------------------------------------

AUDIT_SCHEMA = "w1_real_training_update_audit_v1"
COHORT_ID = "w1_real_training_update_audit_v1"
EXECUTION_ID = "w1_real_training_update_audit_v1"
AUTHORIZATION_SCHEMA = "w1_real_training_update_authorization_v1"
FP32_SCHEMA = "w1_real_training_update_fp32_contract_v1"
TOLERANCE_SCHEMA = "w1_real_training_update_tolerance_contract_v1"
CALIBRATION_SCHEMA = "w1_real_training_update_synthetic_calibration_v1"
ADMISSION_SCHEMA = "w1_real_training_update_admission_package_v1"
REGISTRATION_SCHEMA = "w1_real_training_update_registration_v1"
ACCOUNTING_SCHEMA = "pre_registered_v1"

WINDOW = 60
STRIDE = 30
WINDOW_IDS = ("W0", "W1", "W2", "W3")
MODELS = ("deterministic", "diffusion")
UPDATES = (1, 2)
NOISE_SEED = 20261008
NOISE_PROTOCOL = "w1_real_training_update_noise_v1"
ROSTER_SEED = 20261029
FIXTURE_SEED = 20261029
FIXTURE_HEAD_GAIN = 0.05
FIXTURE_DELTA_AMPLITUDE = 0.25

WINDOW_SPECS = (
    ("sub10_largebox_063", 0, 60, 133),
    ("sub10_largebox_063", 30, 90, 133),
    ("sub10_largebox_085", 0, 60, 148),
    ("sub10_largebox_085", 30, 90, 148),
)

#: Ordered micro matrix from plan section 3.1.  Each entry is four window
#: slots; duplicates are deliberate so the deterministic model does not
#: collapse to one window multiset.
MICRO_SCHEDULE = {
    1: (("W0", "W0", "W1", "W2"),
        ("W1", "W2", "W2", "W3"),
        ("W0", "W1", "W3", "W3"),
        ("W0", "W1", "W2", "W3")),
    2: (("W0", "W0", "W2", "W3"),
        ("W1", "W1", "W2", "W3"),
        ("W0", "W1", "W2", "W2"),
        ("W0", "W1", "W3", "W3")),
}
MICRO_COUNT = 4
DRAW_UNITS_PER_PATH = len(UPDATES) * MICRO_COUNT

#: Effective production configuration frozen by the plan (section 2.1).
MODEL_CONFIG = {
    "d_model": 240, "n_layers": 4, "n_heads": 6, "ffn_mult": 4,
    "time_window": 31, "micro_batch": 4, "grad_accum": 4,
    "lr": 1e-4, "weight_decay": 1e-5, "grad_clip": 1.0,
    "num_train_timesteps": 1000, "beta_schedule": "scaled_linear",
    "beta_start": 1e-4, "beta_end": 0.02,
}
ADAM_DEFAULTS = {"betas": (0.9, 0.999), "eps": 1e-8}
LOSS_WEIGHTS = LossWeights(w_pos=1.0, w_rot6d=1.0, w_joint=1.0,
                           w_geodesic=0.5, w_fk=2.0, w_contact=0.0,
                           w_vel=0.5)
TERMS = ("pos", "rot6d", "joint", "geodesic_rad2", "fk_m2", "vel", "L5",
         "total")
FK_TERM = "fk_m2"

STAGE_BUDGETS = {
    "R0": {"budget_s": 240.0, "reserve_s": 30.0, "attempts": 2},
    "P": {"budget_s": 900.0, "reserve_s": 120.0, "attempts": 2},
    "I": {"budget_s": 900.0, "reserve_s": 120.0, "attempts": 2},
    "F": {"budget_s": 1200.0, "reserve_s": 180.0, "attempts": 2},
}
TOTAL_BUDGET_S = 3240.0
SEAL_TAIL_RESERVE_S = 1.0
SEAL_VERIFY_ROUNDS = 4
STAGE_DIRS = {"R0": "registration", "P": "production", "I": "independent",
              "F": "finalize"}
HISTORY_ROOT = Path("derived/experiments/omniretarget_first")

#: Per-attempt nominal call/work matrix (plan section 7.2).  Cumulative caps
#: are twice the stage column; the cohort cap is twice the full-chain column.
CALL_KINDS = (
    "checkpoint_loads", "model_constructions", "conditioner_constructions",
    "native_optimizer_constructions", "conditioner_forwards",
    "backbone_forwards", "native_add_noise_batches", "cpu_draw_units",
    "production_loss_assembler", "production_float32_fk_batch",
    "production_parameter_backward", "independent_output_vjp",
    "independent_total_vjp", "native_global_clip",
    "independent_clip_formula", "native_adamw_steps",
    "explicit_reference_updates", "diagnostic_parameter_update_events",
    "anchor_state_restores", "optimizer_formula_checks",
    "reference_mj_forward", "reference_mj_jac_site",
    "scene_setup_mj_forward", "micro_execution_units",
)
STAGE_NOMINAL = {
    "R0": {**{kind: 0 for kind in CALL_KINDS}, "cpu_draw_units": 8},
    "P": {
        "checkpoint_loads": 2, "model_constructions": 2,
        "conditioner_constructions": 2, "native_optimizer_constructions": 2,
        "conditioner_forwards": 16, "backbone_forwards": 16,
        "native_add_noise_batches": 8, "cpu_draw_units": 8,
        "production_loss_assembler": 16,
        "production_float32_fk_batch": 32,
        "production_parameter_backward": 16,
        "independent_output_vjp": 0, "independent_total_vjp": 0,
        "native_global_clip": 4, "independent_clip_formula": 0,
        "native_adamw_steps": 4, "explicit_reference_updates": 0,
        "diagnostic_parameter_update_events": 4,
        "anchor_state_restores": 0, "optimizer_formula_checks": 0,
        "reference_mj_forward": 0, "reference_mj_jac_site": 0,
        "scene_setup_mj_forward": 1, "micro_execution_units": 16,
    },
    "I": {
        "checkpoint_loads": 2, "model_constructions": 2,
        "conditioner_constructions": 2, "native_optimizer_constructions": 0,
        "conditioner_forwards": 16, "backbone_forwards": 16,
        "native_add_noise_batches": 8, "cpu_draw_units": 8,
        "production_loss_assembler": 0,
        "production_float32_fk_batch": 0,
        "production_parameter_backward": 0,
        "independent_output_vjp": 16, "independent_total_vjp": 16,
        "native_global_clip": 0, "independent_clip_formula": 4,
        "native_adamw_steps": 0, "explicit_reference_updates": 4,
        "diagnostic_parameter_update_events": 4,
        "anchor_state_restores": 2, "optimizer_formula_checks": 4,
        "reference_mj_forward": 7920, "reference_mj_jac_site": 57600,
        "scene_setup_mj_forward": 0, "micro_execution_units": 16,
    },
    "F": {
        "checkpoint_loads": 4, "model_constructions": 4,
        "conditioner_constructions": 4, "native_optimizer_constructions": 2,
        "conditioner_forwards": 32, "backbone_forwards": 32,
        "native_add_noise_batches": 16, "cpu_draw_units": 24,
        "production_loss_assembler": 16,
        "production_float32_fk_batch": 32,
        "production_parameter_backward": 16,
        "independent_output_vjp": 16, "independent_total_vjp": 16,
        "native_global_clip": 4, "independent_clip_formula": 4,
        "native_adamw_steps": 4, "explicit_reference_updates": 4,
        "diagnostic_parameter_update_events": 8,
        "anchor_state_restores": 2, "optimizer_formula_checks": 4,
        "reference_mj_forward": 7920, "reference_mj_jac_site": 57600,
        "scene_setup_mj_forward": 1, "micro_execution_units": 32,
    },
}
CALL_CAPS = {kind: 2 * sum(STAGE_NOMINAL[stage][kind]
                           for stage in STAGE_BUDGETS)
             for kind in CALL_KINDS}
#: The chain-level cap is two full chains (each stage at its nominal value).
CHAIN_NOMINAL = {kind: sum(STAGE_NOMINAL[stage][kind]
                           for stage in STAGE_BUDGETS)
                 for kind in CALL_KINDS}
for _kind in CALL_KINDS:
    CALL_CAPS[_kind] = 2 * int(CHAIN_NOMINAL[_kind])

ZERO_COUNTERS = (
    "parameter_fd", "state_fd", "ddim_calls", "validation_calls",
    "selection_calls", "production_checkpoint_writes",
    "contact_branch_executions",
)
ZERO_DECLARATIONS = ZERO_COUNTERS

UPDATE_TERMS = ("pos", "rot6d", "joint", "geodesic_rad2", "fk_m2", "vel")
COTANGENT_WEIGHTS = {
    "pos": 1.0, "rot6d": 1.0, "joint": 1.0,
    "geodesic_rad2": 0.5, "fk_m2": 2.0, "vel": 0.5,
}
#: Calibrated absolute equivalence for a mathematically zero reference
#: gradient (float32 autograd leaves cancellation noise ~1e-9).  The draft's
#: 1e-30 proposal was not viable; this is proof-tested in the calibration.
ZERO_GRADIENT_ABS = 1e-6
#: Only tensors whose both norms sit inside this deep-noise band may use the
#: absolute floor.  Observed cancellation noise is ~1e-12, so a 1e-8 tensor
#: is a nonzero comparison and must pass relative-L2/cosine (or report a
#: failure); the floor may not erase it.
ZERO_GRADIENT_NOISE_SCALE = 1e-10
ZERO_STATE_ABS = 1e-6
COTANGENT_ORDER = ("pos", "rot6d", "joint", "geodesic_rad2", "fk_m2", "vel")
COTANGENT_ORDER_INDEX = {k: i for i, k in enumerate(COTANGENT_ORDER)}

#: Numeric gates from plan section 6.1.  Thresholds are proposals until the
#: synthetic calibration proves ``4 x observed`` stays inside each one.
TOLERANCE_PROPOSALS = {
    "alpha_x_t": {"atol": 2e-6, "rtol": 2e-6},
    "loss_scalar": {"atol": 2e-6, "rtol": 2e-5},
    "output_cotangent": {"atol": 2e-8, "rtol": 2e-4},
    "parameter_gradient": {"atol": 1e-8, "rtol": 2e-4},
    "global_norm_clip": {"atol": 1e-7, "rtol": 2e-4},
    "theta_cross_path": {"atol": 2e-7, "rtol": 2e-6},
    "state_cross_path": {"atol": 0.0, "rtol": 2e-3},
    "delta_cross_path": {"atol": 0.0, "rtol": 5e-4},
    "delta_same_input": {"atol": 0.0, "rtol": 5e-4},
    "independent_float64": {"atol": 1e-12, "rtol": 1e-9},
}
TOLERANCE_RULES = {
    "order": ["structure", "dtype", "non_empty", "shape", "domain", "finite",
              "tolerance"],
    "scalar_formula": "abs(prod-ref) <= atol + rtol*abs(ref)",
    "calibration_rule": "4 * observed_max_ratio <= 1.0",
    # A delta is only compared when its half-ULP write bound is at most a
    # quarter of the gate; otherwise it is a precision gap.  Using the gate
    # itself as the observability bound makes 4x calibration impossible.
    "delta_observability_fraction": 0.25,
    "bitwise_gates": [
        "t_epsilon", "x_t_replay", "memory_replay", "prediction_replay",
        "same_input_adamw_state", "step_continuity", "f_replay",
    ],
    "bool_is_not_numeric": True,
    "counts_are_integers": True,
    "thresholds_are_proposals": True,
}
#: Fraction of the delta gate below which a float32 write is comfortably
#: observable.  See TOLERANCE_RULES["delta_observability_fraction"].
DELTA_OBSERVABILITY_FRACTION = 0.25

BOUNDARIES = {
    "objective_coverage": "partial",
    "full_objective_contract_passed": None,
    "attribution": "attribution_insufficient",
    "factor_selected": False,
    "auto_execute": False,
    "final_interaction_method": "not_ready",
    "production_integration": False,
    "training_updates": 16,
    "parameter_fd": 0,
    "state_fd": 0,
    "ddim_calls": 0,
    "checkpoint_writes": 0,
    "contact_branch_executions": 0,
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    return time.perf_counter()


def _ensure_deadline(deadline: float | None, reserve: float = 0.0,
                     unit: str = "unit") -> None:
    if deadline is None:
        return
    if time.perf_counter() + float(reserve) >= float(deadline):
        raise TimeoutError(f"budget_deadline:{unit}")


def _sha_array(a: np.ndarray | torch.Tensor) -> str:
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    return _hash_array(np.ascontiguousarray(np.asarray(a)))


def _to_np(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _bitwise_equal_tensor(a: Any, b: Any) -> bool:
    try:
        a_t = torch.as_tensor(a)
        b_t = torch.as_tensor(b)
    except (TypeError, ValueError):
        return False
    return (a_t.dtype == b_t.dtype and a_t.shape == b_t.shape
            and torch.equal(a_t, b_t))


def _array_shape_problems(value: Any, *, name: str,
                          shape: Sequence[int] | None = None,
                          dtype: Any = None) -> list[str]:
    problems: list[str] = []
    try:
        a = np.asarray(value)
    except (TypeError, ValueError):
        return [f"{name}:not_array"]
    if a.dtype == object or a.dtype == np.bool_:
        problems.append(f"{name}:dtype")
    if a.size == 0:
        problems.append(f"{name}:empty")
    if shape is not None and tuple(a.shape) != tuple(shape):
        problems.append(f"{name}:shape:{a.shape}!={tuple(shape)}")
    if dtype is not None and np.dtype(a.dtype) != np.dtype(dtype):
        problems.append(f"{name}:dtype:{a.dtype}!={np.dtype(dtype)}")
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        problems.append(f"{name}:nonfinite")
    return problems


def _json_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _row(key: str, **fields: Any) -> dict:
    return {"key": key, **fields}


def _summarize(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    failures = sorted({str(r.get("failure")) for r in rows if r.get("failure")})
    gaps = sorted({str(g) for r in rows for g in (r.get("gaps") or [])})
    return [f for f in failures if f], [g for g in gaps if g]


def _call_counts_from_balance(balance: Any) -> dict[str, int] | None:
    if not isinstance(balance, Mapping):
        return None
    if balance.get("schema") != "w1_real_training_update_call_budget_v1":
        return None
    counts = balance.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(CALL_KINDS):
        return None
    out = {}
    for kind in CALL_KINDS:
        value = counts.get(kind)
        if not _json_int(value) or int(value) > int(CALL_CAPS[kind]):
            return None
        out[kind] = int(value)
    if balance.get("within_caps") is not True:
        return None
    return out


class CallCounter:
    """Monotone 24-kind counter; every snapshot must cover all kinds."""

    schema = "w1_real_training_update_call_budget_v1"

    def __init__(self, caps: Mapping[str, int], initial: Mapping[str, int] | None = None) -> None:
        self.caps = {k: int(v) for k, v in caps.items()}
        self.counts = {k: 0 for k in self.caps}
        for key, value in (initial or {}).items():
            if key not in self.caps:
                raise ValueError(f"unknown counter {key}")
            self.counts[key] = int(value)

    def add(self, kind: str, n: int = 1) -> None:
        if kind not in self.counts:
            raise KeyError(f"unknown counter {kind}")
        if type(n) is not int or n < 0:
            raise ValueError("call increment must be a non-negative int")
        value = self.counts[kind] + n
        if value > self.caps[kind]:
            raise UpdateCallBudgetExceeded(
                f"{kind}:{value}>{self.caps[kind]}")
        self.counts[kind] = value

    def snapshot(self) -> dict:
        counts = {k: int(v) for k, v in self.counts.items()}
        return {
            "schema": self.schema,
            "counts": counts,
            "within_caps": all(counts[k] <= self.caps[k] for k in self.caps),
        }


class UpdateCallBudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# parameter roster and row catalog
# ---------------------------------------------------------------------------


def _model_factory(model_type: str, config: Mapping[str, Any], device: str):
    kwargs = dict(d_model=int(config["d_model"]),
                  n_layers=int(config["n_layers"]),
                  n_heads=int(config["n_heads"]),
                  ffn_mult=int(config["ffn_mult"]),
                  time_window=int(config["time_window"]))
    if model_type == "diffusion":
        return DiffusionG1(
            num_train_timesteps=int(config["num_train_timesteps"]),
            beta_schedule=str(config["beta_schedule"]),
            beta_start=float(config["beta_start"]),
            beta_end=float(config["beta_end"]), **kwargs).to(device)
    return DeterministicG1(**kwargs).to(device)


def _controlled_head_init(model: torch.nn.Module) -> None:
    """Bounded head gain keeps random-init predictions in the valid domain."""
    with torch.no_grad():
        model.net.root_out.bias.copy_(
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                         dtype=model.net.root_out.bias.dtype,
                         device=model.net.root_out.bias.device))
        model.net.root_out.weight.mul_(FIXTURE_HEAD_GAIN)
        model.net.joint_out.bias.zero_()
        model.net.joint_out.weight.mul_(FIXTURE_HEAD_GAIN)


def _parameter_entries(module: torch.nn.Module, side: str,
                       start: int) -> list[dict]:
    entries: list[dict] = []
    for offset, (name, param) in enumerate(module.named_parameters()):
        ptr = param.untyped_storage().data_ptr()
        entries.append({
            "name": f"{side}.{name}",
            "side": side,
            "local_name": name,
            "dtype": str(param.dtype).replace("torch.", ""),
            "shape": [int(x) for x in param.shape],
            "numel": int(param.numel()),
            "requires_grad": bool(param.requires_grad),
            "group": 0,
            "index": int(start + offset),
            "storage_ptr": int(ptr),
            "storage_offset": int(param.storage_offset()),
        })
    return entries


def build_parameter_roster(config: Mapping[str, Any] | None = None) -> dict:
    """Full-architecture roster; random/empty weights, no checkpoint read."""
    config = dict(MODEL_CONFIG if config is None else config)
    models: dict[str, dict] = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(ROSTER_SEED)
        for model_type in MODELS:
            model = _model_factory(model_type, config, "cpu")
            conditioner = ConditionEncoder(int(config["d_model"]))
            entries = _parameter_entries(model, "model", 0)
            entries += _parameter_entries(conditioner, "conditioner",
                                          len(entries))
            by_ptr: dict[int, list[str]] = {}
            for entry in entries:
                by_ptr.setdefault(entry["storage_ptr"], []).append(entry["name"])
            for entry in entries:
                shared = by_ptr[entry["storage_ptr"]]
                entry["shared_with"] = sorted(n for n in shared
                                              if n != entry["name"])
                entry.pop("storage_ptr")
            names = [e["name"] for e in entries]
            if len(set(names)) != len(names):
                raise ValueError("duplicate parameter name in roster")
            models[model_type] = {
                "parameters": entries,
                "n_parameters": len(entries),
                "n_elements": sum(e["numel"] for e in entries),
                "state_dict_keys": sorted(
                    [f"model.{k}" for k in model.state_dict()]
                    + [f"conditioner.{k}" for k in conditioner.state_dict()]),
            }
    n_det = models["deterministic"]["n_parameters"]
    n_diff = models["diffusion"]["n_parameters"]
    record = {
        "schema": "w1_real_training_update_parameter_roster_v1",
        "config": config,
        "models": models,
        "n_det": n_det,
        "n_diff": n_diff,
        "n_total": n_det + n_diff,
        "order_rule": "model named_parameters then conditioner named_parameters",
        "optimizer_group": 0,
        "shared_storage_rule": "identical untyped storage data_ptr",
    }
    record["sha256"] = _canonical_sha256(
        {k: record[k] for k in ("models", "n_det", "n_diff", "n_total",
                                "config", "order_rule", "optimizer_group")})
    return record


def roster_problems(actual: Mapping[str, Any] | None,
                    expected: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return ["roster_not_a_mapping"]
    problems: list[str] = []
    if actual.get("schema") != expected.get("schema"):
        problems.append("roster_schema")
    if actual.get("sha256") != expected.get("sha256"):
        problems.append("roster_sha256")
    for model_type in MODELS:
        a = (actual.get("models") or {}).get(model_type) or {}
        e = (expected.get("models") or {}).get(model_type) or {}
        ap = a.get("parameters")
        ep = e.get("parameters")
        if not isinstance(ap, list) or not isinstance(ep, list):
            problems.append(f"roster_parameters:{model_type}")
            continue
        if len(ap) != len(ep):
            problems.append(f"roster_count:{model_type}")
        for i, (x, y) in enumerate(zip(ap, ep)):
            for key in ("name", "dtype", "shape", "numel", "requires_grad",
                        "group", "index", "shared_with"):
                if x.get(key) != y.get(key):
                    problems.append(f"roster_entry:{model_type}:{i}:{key}")
    return problems


def live_roster(trainer: Trainer, model_type: str) -> dict:
    entries = _parameter_entries(trainer.model, "model", 0)
    entries += _parameter_entries(trainer.conditioner, "conditioner",
                                  len(entries))
    by_ptr: dict[int, list[str]] = {}
    for entry in entries:
        by_ptr.setdefault(entry["storage_ptr"], []).append(entry["name"])
    for entry in entries:
        shared = by_ptr[entry["storage_ptr"]]
        entry["shared_with"] = sorted(n for n in shared if n != entry["name"])
        entry.pop("storage_ptr")
    return {
        "schema": "w1_real_training_update_parameter_roster_v1",
        "models": {model_type: {"parameters": entries,
                                "n_parameters": len(entries),
                                "n_elements": sum(e["numel"] for e in entries)}},
        "sha256": None,
    }


def optimizer_roster_problems(trainer: Trainer,
                              roster: Mapping[str, Any],
                              model_type: str) -> list[str]:
    if trainer.opt is None:
        return ["optimizer_missing"]
    groups = trainer.opt.param_groups
    if len(groups) != 1:
        return ["optimizer_group_count"]
    params = list(groups[0]["params"])
    expected = (roster.get("models", {}).get(model_type, {})
                .get("parameters") or [])
    names = [n for n, _ in trainer._named_parameters()]
    problems: list[str] = []
    if len(params) != len(expected):
        problems.append("optimizer_param_count")
    if names != [e.get("name") for e in expected]:
        problems.append("optimizer_param_order")
    if len({id(p) for p in params}) != len(params):
        problems.append("optimizer_duplicate_parameter")
    if len({id(p) for p in trainer.model.parameters()}
           & {id(p) for p in trainer.conditioner.parameters()}):
        problems.append("optimizer_shared_module_parameter")
    return problems


def row_catalog(roster: Mapping[str, Any]) -> dict:
    rows: list[str] = ["model_setup:deterministic", "model_setup:diffusion"]
    for model in MODELS:
        for u in UPDATES:
            for m in range(MICRO_COUNT):
                rows.append(f"input:{model}:u{u}:m{m}")
                for term in TERMS:
                    rows.append(f"objective:{model}:u{u}:m{m}:{term}")
                rows.append(f"cotangent:{model}:u{u}:m{m}")
            rows.append(f"update_global:{model}:u{u}")
        rows.append(f"step_continuity:{model}:1->2")
        entries = roster["models"][model]["parameters"]
        for u in UPDATES:
            for m in range(MICRO_COUNT):
                for entry in entries:
                    rows.append(
                        f"micro_param:{model}:u{u}:m{m}:{entry['name']}")
        for entry in entries:
            for u in UPDATES:
                rows.append(f"accumulated:{model}:u{u}:{entry['name']}")
                rows.append(f"clipped:{model}:u{u}:{entry['name']}")
                rows.append(f"adamw:{model}:u{u}:{entry['name']}")
    return {
        "schema": "w1_real_training_update_row_catalog_v1",
        "rows": rows,
        "keys_sha256": _canonical_sha256(sorted(rows)),
        "counts": dict(Counter(r.split(":", 1)[0] for r in rows)),
        "n_rows": len(rows),
        "n_parameters": int(roster["n_total"]),
        "formula": "14N+168",
    }


# ---------------------------------------------------------------------------
# noise protocol and fixture batches
# ---------------------------------------------------------------------------


def draw_noise_path(seed: int = NOISE_SEED) -> list[dict]:
    """Eight CPU t/epsilon pairs in update -> micro order."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out: list[dict] = []
    for u in UPDATES:
        for m in range(MICRO_COUNT):
            before = g.get_state().clone()
            t = torch.randint(0, 1000, (4,), generator=g, device="cpu",
                              dtype=torch.int64)
            eps = torch.randn((4, WINDOW, 38), generator=g, device="cpu",
                              dtype=torch.float32)
            out.append({
                "update": u, "micro": m,
                "t": t.detach().clone(),
                "epsilon": eps.detach().clone(),
                "generator_state_before": before,
                "generator_state_after": g.get_state().clone(),
            })
    return out


def alpha_cumprod_reference(config: Mapping[str, Any] | None = None) -> np.ndarray:
    config = dict(MODEL_CONFIG if config is None else config)
    steps = int(config["num_train_timesteps"])
    betas = np.linspace(math.sqrt(float(config["beta_start"])),
                        math.sqrt(float(config["beta_end"])),
                        steps, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


def effective_alpha(alpha_cumprod: np.ndarray) -> np.ndarray:
    """Production scheduler stores alpha_cumprod in binary32; upcast it."""
    return np.asarray(alpha_cumprod, dtype=np.float32).astype(np.float64)


def independent_x_t(x0: np.ndarray, epsilon: np.ndarray, t: np.ndarray,
                    alpha_cumprod: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    t = np.asarray(t, dtype=np.int64)
    a = alpha_cumprod[t][:, None, None]
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * epsilon


def _window_arrays(world: SyntheticWorld, start: int, kp_seed: int) -> dict:
    end = start + WINDOW
    rng = np.random.default_rng(int(kp_seed))
    kp = rng.normal(0.0, 0.1, size=(WINDOW, 52, 3)).astype(np.float32)
    return {
        "kps_obj": kp,
        "kps_clip": (kp + np.array([0.05, -0.02, 0.01],
                                   dtype=np.float32)).astype(np.float32),
        "box_t_clip": np.broadcast_to(np.asarray(world.center, dtype=np.float32),
                                      (WINDOW, 3)).copy(),
        "box_quat": np.asarray(world.box_quat[start:end], dtype=np.float32),
        "contact": np.asarray(world.contact[start:end], dtype=np.float32),
        "wfc": np.asarray(world.wfc, dtype=np.float32),
    }


class _EmptyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


class _PlaceholderDataset(torch.utils.data.Dataset):
    """Four tiny items; the audit never iterates the native loader."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index):
        return {"x0": torch.zeros(1, 2, 38)}


class CountingLoss:
    """Read-only call counter around the production loss assembler."""

    def __init__(self, inner, counter: CallCounter) -> None:
        self.inner = inner
        self.counter = counter
        self.last_observer: dict | None = None

    def __call__(self, pred, batch):
        self.counter.add("production_loss_assembler", 1)
        observer: dict = {}
        loss, parts = self.inner(pred, batch, observer=observer)
        self.last_observer = observer
        return loss, parts

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class ForbiddenLoss:
    """Guards the independent stage: production loss must not execute there."""

    def __call__(self, pred, batch):
        raise RuntimeError("production_loss_assembler_called_in_independent_stage")


class CountingFk:
    """Count production FK batches at the real call site."""

    def __init__(self, inner, counter: CallCounter) -> None:
        self.inner = inner
        self.counter = counter

    def site_world_positions(self, *args, **kwargs):
        self.counter.add("production_float32_fk_batch", 1)
        return self.inner.site_world_positions(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _rng_state() -> dict:
    return {
        "python": __import__("random").getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_initialized() else None),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    import random

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _seed_stage_rng(seed: int = ROSTER_SEED) -> None:
    """Give each P/I invocation the same ambient RNG origin.

    No production draw depends on the ambient RNG (noise comes from explicit
    CPU generators and batches are fixed), but the RNG digests recorded per
    micro are otherwise process-path dependent and cannot be replayed.
    """
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(int(seed))


def _rng_digest(state: Mapping[str, Any]) -> dict:
    return {
        "python": sha256_bytes(repr(state["python"]).encode()),
        "numpy": sha256_bytes(state["numpy"][1].tobytes()),
        "torch_cpu": sha256_bytes(state["torch_cpu"].numpy().tobytes()),
        "torch_cuda": (None if state.get("torch_cuda") is None else
                       [sha256_bytes(s.numpy().tobytes())
                        for s in state["torch_cuda"]]),
    }

# ---------------------------------------------------------------------------
# synthetic fixture and provider
# ---------------------------------------------------------------------------


def _loss_assembler(tokenizer: StateTokenizer, chain: SyntheticChain,
                    counter: CallCounter | None, device: str) -> LossAssembler:
    """Production ``LossAssembler`` methods with the analytic FK stand-in.

    The real constructor compiles the locked MuJoCo scene/URDF; the synthetic
    audit must not read those assets.  Every consumed field is set explicitly
    and the FK adapter is the audited synthetic chain (``w_contact=0`` keeps
    the box-geom fields unread).
    """
    asm = object.__new__(LossAssembler)
    asm.w = LOSS_WEIGHTS
    asm.device = device
    asm.mean = torch.tensor(tokenizer.root_stats.mean, dtype=torch.float32,
                           device=device)
    asm.std = torch.tensor(tokenizer.root_stats.std, dtype=torch.float32,
                          device=device)
    asm.lo = torch.tensor(tokenizer.lo, dtype=torch.float32, device=device)
    asm.hi = torch.tensor(tokenizer.hi, dtype=torch.float32, device=device)
    adapter = _SyntheticFkAdapter(chain)
    asm.fk = (CountingFk(adapter, counter) if counter is not None else adapter)
    asm.R_g = torch.eye(3, dtype=torch.float32, device=device)
    asm.x_g = torch.zeros(3, dtype=torch.float32, device=device)
    return asm


def _angle_between_rot6d(a: np.ndarray, b: np.ndarray,
                         constants: Mapping[str, np.ndarray]) -> np.ndarray:
    da = independent_decode(a.astype(np.float64),
                            constants["mean32"].astype(np.float64),
                            constants["std32"].astype(np.float64),
                            constants["lo32"].astype(np.float64),
                            constants["hi32"].astype(np.float64))
    db = independent_decode(b.astype(np.float64),
                            constants["mean32"].astype(np.float64),
                            constants["std32"].astype(np.float64),
                            constants["lo32"].astype(np.float64),
                            constants["hi32"].astype(np.float64))
    rel = np.einsum("tij,tkj->tik", da["R"], db["R"])
    trace = np.trace(rel, axis1=1, axis2=2)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


class SyntheticTrainingProvider:
    """Analytic four-window fixture with the full production architecture."""

    mode = "synthetic"

    def __init__(self, config: Mapping[str, Any] | None = None,
                 *, seed: int = FIXTURE_SEED, device: str = "cuda") -> None:
        self.config = dict(MODEL_CONFIG if config is None else config)
        self.seed = int(seed)
        self.device = device
        self.counter: CallCounter | None = None
        self.fixture_dir: Path | None = None
        self.fixture: dict | None = None
        self.tokenizer: StateTokenizer | None = None
        self.chain: SyntheticChain | None = None
        self.constants: dict | None = None
        self.problems: list[str] = []
        self.fixture_seconds: float | None = None

    def bind_counter(self, counter: CallCounter) -> None:
        self.counter = counter

    # --- fixture ---------------------------------------------------------

    def _build_tokenizer(self) -> StateTokenizer:
        self.worlds = [build_synthetic_world(T=WINDOW + STRIDE,
                                             seed=self.seed + 101 + i)
                       for i in range(2)]
        self.tokenizer = self.worlds[0].tokenizer
        self.chain = build_synthetic_chain()
        self.constants = effective_constants(
            self.tokenizer.root_stats.mean, self.tokenizer.root_stats.std,
            self.tokenizer.lo, self.tokenizer.hi)
        return self.tokenizer

    def _window_meta(self) -> list[dict]:
        return [
            {"id": WINDOW_IDS[i], "demo": spec[0], "start": spec[1],
             "end": spec[2], "T": spec[3]}
            for i, spec in enumerate(WINDOW_SPECS)
        ]

    def _fixture_targets(self, base_states: Mapping[str, Mapping[str, Any]]) -> dict:
        # targets = deterministic base prediction + registered perturbation
        targets: dict[str, np.ndarray] = {}
        for i, key in enumerate(WINDOW_IDS):
            pred = base_states[key]["prediction"]
            delta = (FIXTURE_DELTA_AMPLITUDE
                     * np.sin(0.7 * np.arange(1, 39, dtype=np.float64) * 0.3
                              + i * 1.1)).astype(np.float32)
            target = (pred + delta).astype(np.float32)
            target[..., 9:38] = np.clip(target[..., 9:38], -0.9, 0.9)
            targets[key] = target
        return targets

    def _load_state(self, model_type: str) -> dict:
        path = self.fixture_dir / f"state_{model_type}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return payload

    def prepare_fixture(self, directory: Path) -> dict:
        t0 = _now()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.fixture_dir = directory
        tokenizer = self._build_tokenizer()
        windows = self._window_meta()
        states: dict[str, dict] = {}
        fixture_forwards = 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            det = _model_factory("deterministic", self.config, "cpu")
            _controlled_head_init(det)
            det_cond = ConditionEncoder(int(self.config["d_model"]))
            torch.manual_seed(self.seed + 1)
            diff = _model_factory("diffusion", self.config, "cpu")
            _controlled_head_init(diff)
            diff_cond = ConditionEncoder(int(self.config["d_model"]))
            states = {
                "deterministic": {"model": det.state_dict(),
                                  "conditioner": det_cond.state_dict()},
                "diffusion": {"model": diff.state_dict(),
                              "conditioner": diff_cond.state_dict()},
            }
            for model_type in MODELS:
                torch.save(states[model_type],
                           directory / f"state_{model_type}.pt")
        # window condition arrays and analytic targets
        window_arrays: dict[str, dict] = {}
        raw_batches: dict[str, dict] = {}
        for i, key in enumerate(WINDOW_IDS):
            wi = 0 if i < 2 else 1
            start = WINDOW_SPECS[i][1]
            arrays = _window_arrays(self.worlds[wi], start,
                                    kp_seed=self.seed + 500 + i)
            window_arrays[key] = arrays
            batch = {name: torch.tensor(value, dtype=torch.float32,
                                        device="cpu").unsqueeze(0)
                     for name, value in arrays.items()}
            batch["x0"] = torch.tensor(self.worlds[wi].x0[start:start + WINDOW],
                                       dtype=torch.float32,
                                       device="cpu").unsqueeze(0)
            raw_batches[key] = batch
        # deterministic base prediction for the targets
        det.eval(); det_cond.eval()
        base_states: dict[str, dict] = {}
        with torch.no_grad():
            for key in WINDOW_IDS:
                memory = det_cond(raw_batches[key])
                pred = det(memory, WINDOW)
                base_states[key] = {"prediction": pred[0].numpy().copy(),
                                    "memory_sha256": _sha_array(memory)}
                fixture_forwards += 2
        targets = self._fixture_targets(base_states)
        for key in WINDOW_IDS:
            window_arrays[key]["x0"] = targets[key]
        # registered noise matrix and independent x_t
        noise = draw_noise_path(NOISE_SEED)
        alpha = alpha_cumprod_reference(self.config)
        x_t_ref: list[np.ndarray] = []
        for unit in noise:
            slots = MICRO_SCHEDULE[unit["update"]][unit["micro"]]
            x0 = np.stack([window_arrays[w]["x0"] for w in slots], axis=0)
            x_t_ref.append(independent_x_t(x0, unit["epsilon"].numpy(),
                                           unit["t"].numpy(),
                                           effective_alpha(alpha)))
        # diffusion pre-check: predictions must stay in the rotation domain
        diff.eval(); diff_cond.eval()
        max_angle = 0.0
        domain_bad = 0
        with torch.no_grad():
            for unit in noise:
                slots = MICRO_SCHEDULE[unit["update"]][unit["micro"]]
                batch = {
                    "x0": torch.stack([torch.tensor(window_arrays[w]["x0"])
                                       for w in slots]),
                    **{name: torch.stack([torch.tensor(window_arrays[w][name])
                                          for w in slots])
                       for name in ("kps_obj", "kps_clip", "box_t_clip",
                                    "box_quat", "contact")},
                    "wfc": torch.stack([torch.tensor(window_arrays[w]["wfc"])
                                        for w in slots]),
                }
                memory = diff_cond(batch)
                t = unit["t"]
                noise_t = unit["epsilon"]
                x_t = diff.add_noise(batch["x0"], noise_t, t)
                pred = diff(x_t, t, memory)
                fixture_forwards += 2
                angle = _angle_between_rot6d(pred.numpy().reshape(-1, 38),
                                             batch["x0"].numpy().reshape(-1, 38),
                                             self.constants)
                max_angle = max(max_angle, float(angle.max()))
                dec = independent_decode(pred.numpy().reshape(-1, 38).astype(np.float64),
                                         self.constants["mean32"].astype(np.float64),
                                         self.constants["std32"].astype(np.float64),
                                         self.constants["lo32"].astype(np.float64),
                                         self.constants["hi32"].astype(np.float64))
                domain_bad += int((~dec["ok"]).sum())
        # freeze the static None/zero gradient contract from a fixture pass
        none_params: dict[str, list[str]] = {}
        zero_params: dict[str, list[str]] = {}
        for model_type, model, conditioner in (
                ("deterministic", det, det_cond),
                ("diffusion", diff, diff_cond)):
            model.train(); conditioner.train()
            unit = noise[0]
            slots = MICRO_SCHEDULE[unit["update"]][unit["micro"]]
            batch = {
                "x0": torch.stack([torch.tensor(window_arrays[w]["x0"])
                                   for w in slots]),
                **{name: torch.stack([torch.tensor(window_arrays[w][name])
                                      for w in slots])
                   for name in ("kps_obj", "kps_clip", "box_t_clip",
                                "box_quat", "contact")},
                "wfc": torch.stack([torch.tensor(window_arrays[w]["wfc"])
                                    for w in slots]),
            }
            memory = conditioner(batch)
            if model_type == "diffusion":
                pred = model(batch["x0"], unit["t"], memory)
            else:
                pred = model(memory, WINDOW)
            pred.square().mean().backward()
            none_params[model_type], zero_params[model_type] = [], []
            named = ([("model." + n, p) for n, p in model.named_parameters()]
                     + [("conditioner." + n, p)
                        for n, p in conditioner.named_parameters()])
            for name, param in named:
                grad = param.grad
                if grad is None:
                    none_params[model_type].append(name)
                elif float(grad.abs().max()) == 0.0:
                    zero_params[model_type].append(name)
            model.zero_grad(set_to_none=True)
            conditioner.zero_grad(set_to_none=True)
            fixture_forwards += 1
        max_angle = max(max_angle, 0.0)
        self.problems = []
        if max_angle > 1.0:
            self.problems.append(f"fixture_prediction_angle:{max_angle:.6g}")
        if domain_bad:
            self.problems.append(f"fixture_prediction_domain_bad:{domain_bad}")
        metadata = {
            "schema": "w1_real_training_update_synthetic_fixture_v1",
            "mode": "synthetic",
            "cohort_id": COHORT_ID,
            "seed": self.seed,
            "config": self.config,
            "windows": windows,
            "window_ids": list(WINDOW_IDS),
            "micro_schedule": {str(u): [list(x) for x in MICRO_SCHEDULE[u]]
                               for u in UPDATES},
            "noise_seed": NOISE_SEED,
            "noise_protocol": NOISE_PROTOCOL,
            "draw_units": len(noise),
            "alpha_cumprod_sha256": _hash_array(alpha),
            "x_t_reference_sha256": [_hash_array(a) for a in x_t_ref],
            "target_source": ("deterministic_fixture_forward_plus_registered_"
                              "perturbation"),
            "fixture_delta_amplitude": FIXTURE_DELTA_AMPLITUDE,
            "fixture_head_gain": FIXTURE_HEAD_GAIN,
            "fixture_forward_calls": fixture_forwards,
            "fixture_prediction_max_angle_rad": max_angle,
            "fixture_prediction_domain_bad": domain_bad,
            "none_gradient_parameters": none_params,
            "zero_gradient_parameters": zero_params,
            "problems": list(self.problems),
            "state_sha256": {
                model_type: {
                    "model": _canonical_sha256(
                        {k: _hash_array(v) for k, v in
                         states[model_type]["model"].items()}),
                    "conditioner": _canonical_sha256(
                        {k: _hash_array(v) for k, v in
                         states[model_type]["conditioner"].items()}),
                } for model_type in MODELS},
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        np.savez(directory / "windows.npz",
                 **{f"{key}__{name}": value
                    for key, arrays in window_arrays.items()
                    for name, value in arrays.items()})
        np.savez(directory / "noise.npz",
                 alpha_cumprod=alpha,
                 **{f"u{unit['update']}m{unit['micro']}__t": unit["t"].numpy()
                    for unit in noise},
                 **{f"u{unit['update']}m{unit['micro']}__epsilon":
                    unit["epsilon"].numpy() for unit in noise},
                 **{f"u{unit['update']}m{unit['micro']}__x_t": x_t_ref[i]
                    for i, unit in enumerate(noise)})
        roster = build_parameter_roster(self.config)
        write_json(directory / "roster.json", roster)
        metadata["roster_sha256"] = roster["sha256"]
        metadata["roster_n_total"] = roster["n_total"]
        metadata["windows_sha256"] = sha256_file(directory / "windows.npz")
        metadata["noise_sha256"] = sha256_file(directory / "noise.npz")
        write_json(directory / "fixture.json", metadata)
        self.fixture = metadata
        self.fixture_seconds = _now() - t0
        self.roster = roster
        self.window_arrays = window_arrays
        self.noise = noise
        return metadata

    # --- provider API ----------------------------------------------------

    def load_fixture(self) -> dict:
        if self.fixture is None:
            meta = read_json(self.fixture_dir / "fixture.json")
            self.fixture = meta
            self.roster = read_json(self.fixture_dir / "roster.json")
            data = np.load(self.fixture_dir / "windows.npz")
            self.window_arrays = {}
            for key in WINDOW_IDS:
                self.window_arrays[key] = {
                    name: data[f"{key}__{name}"]
                    for name in ("x0", "kps_obj", "kps_clip", "box_t_clip",
                                 "box_quat", "contact", "wfc")}
            noise_data = np.load(self.fixture_dir / "noise.npz")
            self.noise = draw_noise_path(NOISE_SEED)
            for unit in self.noise:
                u, m = unit["update"], unit["micro"]
                if not np.array_equal(unit["t"].numpy(),
                                      noise_data[f"u{u}m{m}__t"]):
                    self.problems.append(f"fixture_noise_t:{u}:{m}")
                if not np.array_equal(unit["epsilon"].numpy(),
                                      noise_data[f"u{u}m{m}__epsilon"]):
                    self.problems.append(f"fixture_noise_epsilon:{u}:{m}")
                unit["x_t"] = noise_data[f"u{u}m{m}__x_t"]
            # tokenizer/chain/constants are analytic and rebuilt on demand
            self._build_tokenizer()
            self.constants = effective_constants(
                self.tokenizer.root_stats.mean,
                self.tokenizer.root_stats.std,
                self.tokenizer.lo, self.tokenizer.hi)
        return self.fixture

    def micro_batch(self, update: int, micro: int) -> dict:
        slots = MICRO_SCHEDULE[update][micro]
        out: dict[str, torch.Tensor] = {}
        for name in ("x0", "kps_obj", "kps_clip", "box_t_clip", "box_quat",
                     "contact", "wfc"):
            out[name] = torch.stack(
                [torch.tensor(self.window_arrays[w][name], dtype=torch.float32)
                 for w in slots]).to(self.device)
        return out

    def build_trainer(self, model_type: str, *, with_optimizer: bool,
                      counter: CallCounter, loss_mode: str = "production",
                      out_dir: Path | None = None) -> Trainer:
        cfg = TrainConfig(
            model_type=model_type,
            steps=len(UPDATES),
            micro_batch=int(self.config["micro_batch"]),
            grad_accum=int(self.config["grad_accum"]),
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"]),
            grad_clip=float(self.config["grad_clip"]),
            seed=ROSTER_SEED,
            num_workers=0, val_every=10 ** 9, log_every=10 ** 9,
            device=self.device, val_seed=ROSTER_SEED,
            d_model=int(self.config["d_model"]),
            n_layers=int(self.config["n_layers"]),
            n_heads=int(self.config["n_heads"]),
            ffn_mult=int(self.config["ffn_mult"]),
            time_window=int(self.config["time_window"]),
            num_train_timesteps=int(self.config["num_train_timesteps"]),
            beta_schedule=str(self.config["beta_schedule"]),
            beta_start=float(self.config["beta_start"]),
            beta_end=float(self.config["beta_end"]),
            loss=LOSS_WEIGHTS,
        )
        if loss_mode == "production":
            loss = CountingLoss(_loss_assembler(self.tokenizer, self.chain,
                                                counter, self.device), counter)
        elif loss_mode == "forbidden":
            loss = ForbiddenLoss()
        else:
            raise ValueError(f"unknown loss_mode {loss_mode}")
        out = Path(out_dir or (self.fixture_dir / "_trainer_tmp"))
        out.mkdir(parents=True, exist_ok=True)
        with torch.random.fork_rng(devices=[]):
            trainer = Trainer(
                cfg, self.tokenizer, _PlaceholderDataset(),
                _PlaceholderDataset(), out,
                loss_fn=loss, build_optimizer=with_optimizer)
        counter.add("model_constructions", 1)
        counter.add("conditioner_constructions", 1)
        if with_optimizer:
            counter.add("native_optimizer_constructions", 1)
        state = self._load_state(model_type)
        counter.add("checkpoint_loads", 1)
        trainer.model.load_state_dict(state["model"], strict=True)
        trainer.conditioner.load_state_dict(state["conditioner"],
                                            strict=True)
        trainer.model.train(); trainer.conditioner.train()
        return trainer

# ---------------------------------------------------------------------------
# effective float32 AdamW evaluation graph (plan section 4.3.1)
# ---------------------------------------------------------------------------

getcontext().prec = 80
_F32_INF = float(np.finfo(np.float32).max)


def _f32_bits(value: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", float(np.float32(value))))[0])


def _f32_from_bits(bits: int) -> float:
    bits &= 0xFFFFFFFF
    return float(np.float32(struct.unpack("<f", struct.pack("<I", bits))[0]))


def _f32_round_fraction(value: Fraction) -> float:
    """Exact round-to-nearest-ties-to-even of a rational to binary32."""
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    mag = abs(value)
    if mag > Fraction(_F32_INF):
        return sign * math.inf
    approx = np.float32(float(mag))
    base = _f32_bits(approx)
    best: tuple | None = None
    for delta in (-2, -1, 0, 1, 2):
        bits = base + delta
        if bits < 0 or bits > 0x7F800000:
            continue
        candidate = _f32_from_bits(bits)
        if not math.isfinite(candidate):
            continue
        distance = abs(Fraction(candidate) - mag)
        even = (_f32_bits(candidate) & 1) == 0
        key = (distance, 0 if even else 1)
        if best is None or key < best[0]:
            best = (key, candidate)
    if best is None:
        raise ValueError("no float32 candidate")
    return sign * best[1]


def _exact_fma32(a: float, b: float, c: float) -> float:
    return _f32_round_fraction(Fraction(a) * Fraction(b) + Fraction(c))


def _exact_div32(a: float, b: float) -> float:
    return _f32_round_fraction(Fraction(a) / Fraction(b))


def _exact_sqrt32(v: float) -> float:
    fr = Fraction(v)
    root = (Decimal(fr.numerator) / Decimal(fr.denominator)).sqrt()
    return _f32_round_fraction(Fraction(root))


def _f32_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _fma32_array(a: Any, b: Any, c: Any) -> tuple[np.ndarray, int]:
    """IEEE binary32 FMA: exact product+sum, one round, exact fallbacks."""
    a32, b32, c32 = _f32_array(a), _f32_array(b), _f32_array(c)
    a64 = a32.astype(np.float64)
    b64 = b32.astype(np.float64)
    c64 = c32.astype(np.float64)
    product = a64 * b64                      # exact for binary32 operands
    total = product + c64
    bb = total - product
    err = (product - (total - bb)) + (c64 - bb)   # exact rounding error
    result = total.astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        r64 = result.astype(np.float64)
        up = np.nextafter(result, np.float32(np.inf)).astype(np.float64)
        down = np.nextafter(result, np.float32(-np.inf)).astype(np.float64)
        mid_up = (r64 + up) * 0.5
        mid_down = (down + r64) * 0.5
        unsafe = np.where(
            err > 0, np.abs(total - mid_up) <= np.abs(err),
            np.where(err < 0, np.abs(total - mid_down) <= np.abs(err),
                     False))
        unsafe &= np.isfinite(total) & np.isfinite(r64)
    fallbacks = 0
    if unsafe.any():
        flat = result.reshape(-1)
        for index in np.flatnonzero(unsafe.reshape(-1)):
            flat[index] = np.float32(_exact_fma32(
                float(a32.reshape(-1)[index]),
                float(b32.reshape(-1)[index]),
                float(c32.reshape(-1)[index])))
            fallbacks += 1
    return result, fallbacks


def _div32_array(a: Any, b: Any) -> tuple[np.ndarray, int]:
    """Correctly rounded binary32 division with exact midpoint fallbacks."""
    a32, b32 = _f32_array(a), _f32_array(b)
    q = a32.astype(np.float64) / b32.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        result = q.astype(np.float32)
        r64 = result.astype(np.float64)
        up = np.nextafter(result, np.float32(np.inf)).astype(np.float64)
        down = np.nextafter(result, np.float32(-np.inf)).astype(np.float64)
        mid_up = (r64 + up) * 0.5
        mid_down = (down + r64) * 0.5
        spacing = np.abs(np.spacing(q))
        unsafe = (np.minimum(np.abs(q - mid_up), np.abs(q - mid_down))
                  <= 2.0 * spacing) & np.isfinite(q) & np.isfinite(r64)
    fallbacks = 0
    if unsafe.any():
        flat = result.reshape(-1)
        for index in np.flatnonzero(unsafe.reshape(-1)):
            flat[index] = np.float32(_exact_div32(
                float(a32.reshape(-1)[index]),
                float(b32.reshape(-1)[index])))
            fallbacks += 1
    return result, fallbacks


def _sqrt32_array(v: Any) -> tuple[np.ndarray, int]:
    """sqrtf is correctly rounded from a binary32 input (53-bit theorem)."""
    v32 = _f32_array(v)
    with np.errstate(invalid="ignore"):
        result = np.sqrt(v32.astype(np.float64)).astype(np.float32)
    return result, 0


def _half_ulp32(value: Any) -> np.ndarray:
    """Half the larger adjacent binary32 spacing, in float64."""
    x = np.asarray(value, dtype=np.float32)
    x64 = x.astype(np.float64)
    with np.errstate(invalid="ignore"):
        up = np.nextafter(x, np.float32(np.inf)).astype(np.float64)
        down = np.nextafter(x, np.float32(-np.inf)).astype(np.float64)
        gap = np.maximum(np.abs(up - x64), np.abs(x64 - down))
    gap = np.where(np.isfinite(gap), gap, 0.0)
    return 0.5 * gap


def _delta_write_bound(theta32: Any) -> np.ndarray:
    """Plan section 6.2: half the larger adjacent finite float32 spacing."""
    return _half_ulp32(theta32)


def host_constants(step: int, adams: Mapping[str, Any] | None = None,
                   lr: float = 1e-4, weight_decay: float = 1e-5) -> dict:
    adams = dict(ADAM_DEFAULTS if adams is None else adams)
    beta1, beta2 = (float(x) for x in adams["betas"])
    eps = float(adams["eps"])
    bc1 = 1.0 - beta1 ** int(step)
    bc2 = 1.0 - beta2 ** int(step)
    return {
        "step": int(step),
        "beta1": beta1, "beta2": beta2, "lr": float(lr),
        "weight_decay": float(weight_decay), "eps": eps,
        "bc1": bc1,
        "bc2_sqrt": math.sqrt(bc2),
        "h": -float(lr) / bc1,
        "d": 1.0 - float(lr) * float(weight_decay),
        "bc1_32": float(np.float32(bc1)),
        "bc2_sqrt_32": float(np.float32(math.sqrt(bc2))),
        "h_32": float(np.float32(-float(lr) / bc1)),
        "d_32": float(np.float32(1.0 - float(lr) * float(weight_decay))),
        "a_32": float(np.float32(1.0 - beta1)),
        "b2_32": float(np.float32(beta2)),
        "one_minus_b2_32": float(np.float32(1.0 - beta2)),
        "eps_32": float(np.float32(eps)),
    }


def adamw_math_reference(theta: Any, m: Any, v: Any, g: Any, step: int,
                         *, m_none: bool = False, v_none: bool = False,
                         host: Mapping[str, Any] | None = None) -> dict:
    """Float64 algebraic reference; never applied as an update."""
    host = dict(host_constants(step) if host is None else host)
    theta64 = np.asarray(theta, dtype=np.float64)
    g64 = np.asarray(g, dtype=np.float64)
    m64 = np.zeros_like(g64) if m_none else np.asarray(m, dtype=np.float64)
    v64 = np.zeros_like(g64) if v_none else np.asarray(v, dtype=np.float64)
    beta1, beta2 = host["beta1"], host["beta2"]
    m_new = beta1 * m64 + (1.0 - beta1) * g64
    v_new = beta2 * v64 + (1.0 - beta2) * g64 * g64
    mhat = m_new / (1.0 - beta1 ** int(step))
    vhat = v_new / (1.0 - beta2 ** int(step))
    theta_new = ((1.0 - host["lr"] * host["weight_decay"]) * theta64
                 - host["lr"] * mhat / (np.sqrt(vhat) + host["eps"]))
    return {"m": m_new, "v": v_new, "mhat": mhat, "vhat": vhat,
            "theta": theta_new, "step": int(step)}


def adamw_effective_fp32(theta: Any, m: Any, v: Any, g: Any, step: int,
                         *, m_none: bool = False, v_none: bool = False,
                         host: Mapping[str, Any] | None = None) -> dict:
    """The frozen node graph, evaluated node by node on binary32 values."""
    host = dict(host_constants(step) if host is None else host)
    theta32 = _f32_array(theta)
    g32 = _f32_array(g)
    m_old = (np.zeros_like(theta32) if m_none else _f32_array(m))
    v_old = (np.zeros_like(theta32) if v_none else _f32_array(v))
    theta_decay = (theta32 * np.float32(host["d_32"])).astype(np.float32)
    z = (g32 - m_old).astype(np.float32)
    m_new, m_fb = _fma32_array(np.float32(host["a_32"]), z, m_old)
    v_scaled = (np.float32(host["b2_32"]) * v_old).astype(np.float32)
    g2 = (g32 * g32).astype(np.float32)
    v_new, v_fb = _fma32_array(np.float32(host["one_minus_b2_32"]), g2,
                               v_scaled)
    s, sqrt_fb = _sqrt32_array(v_new)
    b, b_fb = _div32_array(s, np.float32(host["bc2_sqrt_32"]))
    denom = (b + np.float32(host["eps_32"])).astype(np.float32)
    ratio, r_fb = _div32_array(m_new, denom)
    theta_new, t_fb = _fma32_array(np.float32(host["h_32"]), ratio,
                                   theta_decay)
    delta_ref64 = ((theta_decay.astype(np.float64)
                    - theta32.astype(np.float64))
                   + np.float64(host["h_32"])
                   * ratio.astype(np.float64))
    return {
        "theta": theta_new, "theta_decay": theta_decay,
        "m": m_new, "v": v_new, "s": s, "b": b, "denom": denom,
        "ratio": ratio, "z": z, "g2": g2, "v_scaled": v_scaled,
        "delta_ref64": delta_ref64,
        "host": host,
        "rounding_fallbacks": {
            "m": m_fb, "v": v_fb, "b": b_fb, "ratio": r_fb,
            "theta": t_fb, "sqrt": sqrt_fb,
        },
        "state_initialized": (not m_none) and (not v_none),
    }


# --- interval propagation (plan section 6.4) -------------------------------

def _r32_down(value: Any) -> np.ndarray:
    """Largest binary32 not greater than each real value (as float64)."""
    v = np.asarray(value, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        f = v.astype(np.float32)
        f64 = f.astype(np.float64)
        out = np.where(f64 > v, np.nextafter(f, np.float32(-np.inf)), f)
    return out.astype(np.float64)


def _r32_up(value: Any) -> np.ndarray:
    """Smallest binary32 not less than each real value (as float64)."""
    v = np.asarray(value, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        f = v.astype(np.float32)
        f64 = f.astype(np.float64)
        out = np.where(f64 < v, np.nextafter(f, np.float32(np.inf)), f)
    return out.astype(np.float64)


def _iv_square(lo, hi):
    lo2 = lo * lo
    hi2 = hi * hi
    cross = (lo <= 0) & (hi >= 0)
    lower = np.where(cross, 0.0, np.minimum(lo2, hi2))
    upper = np.maximum(lo2, hi2)
    return lower, upper


def _iv_div_positive(lo1, hi1, lo2, hi2):
    """Real corner bounds of an interval quotient (rounding applied later)."""
    q1 = lo1 / lo2
    q2 = lo1 / hi2
    q3 = hi1 / lo2
    q4 = hi1 / hi2
    lo = np.minimum(np.minimum(q1, q2), np.minimum(q3, q4))
    hi = np.maximum(np.maximum(q1, q2), np.maximum(q3, q4))
    return lo, hi


def effective_input_budget(g_i: np.ndarray) -> np.ndarray:
    """Per-coordinate input set admitted by the gradient gate itself.

    The interval must cover every gradient that ``compare_gradient`` can
    accept.  A tensor in the deep cancellation band is admitted by the
    calibrated absolute floor; a small non-deep tensor by the group gate
    (relative-L2 <= 2e-4 of its norm); otherwise by the elementwise gate.
    Using a tighter invented tensor budget would declare legal inputs
    unverifiable.
    """
    g = np.asarray(g_i, dtype=np.float64)
    norm = float(np.linalg.norm(g))
    if norm <= ZERO_GRADIENT_NOISE_SCALE:
        return np.full_like(g, ZERO_GRADIENT_ABS)
    if norm <= ZERO_GRADIENT_ABS:
        return np.full_like(g, float(GROUP_GATE["relative_l2"]) * norm)
    return float(TOLERANCE_PROPOSALS["parameter_gradient"]["atol"]) + float(
        TOLERANCE_PROPOSALS["parameter_gradient"]["rtol"]) * np.abs(g)


def propagate_intervals(g_i: np.ndarray, theta: Any, m: Any, v: Any,
                        step: int, *, m_none: bool = False,
                        v_none: bool = False,
                        host: Mapping[str, Any] | None = None) -> dict:
    """Outer enclosure of the effective graph over the allowed input set.

    Every prescribed binary32 node is applied monotone-wise, then rounded
    outwards to binary32: z=R32(g-m), m=R32(a*z+m), g2=R32(g*g),
    v=R32(c*g2+v_scaled), s=sqrtf(v), b=R32(s/bc2), den=R32(b+eps),
    ratio=R32(m/den), theta=R32(h*ratio+theta_decay).  Clipping an interval at
    the central float64 value is not an enclosure.
    """
    host = dict(host_constants(step) if host is None else host)
    g64 = np.asarray(g_i, dtype=np.float64)
    budget = effective_input_budget(g64)
    g_lo = g64 - budget
    g_hi = g64 + budget
    g_lo = np.maximum(g_lo, -_F32_INF)
    g_hi = np.minimum(g_hi, _F32_INF)
    theta32 = _f32_array(theta)
    theta_before = theta32.astype(np.float64)
    m_old = (np.zeros_like(theta32) if m_none else _f32_array(m)).astype(np.float64)
    v_old32 = (np.zeros_like(theta32) if v_none else _f32_array(v))
    # z = R32(g - m_old)
    z_lo = _r32_down(g_lo - m_old)
    z_hi = _r32_up(g_hi - m_old)
    a = np.float64(host["a_32"])
    # m = R32(a * z + m_old), a > 0 -> monotone in z
    m_lo = _r32_down(a * z_lo + m_old)
    m_hi = _r32_up(a * z_hi + m_old)
    # v_scaled = R32(beta2 * v_old): one fixed binary32 value
    v_scaled = (np.float32(host["b2_32"]) * v_old32).astype(np.float32)
    v_scaled64 = v_scaled.astype(np.float64)
    # g2 = R32(g * g); square crosses zero -> include 0
    g2_real_lo, g2_real_hi = _iv_square(g_lo, g_hi)
    g2_lo = _r32_down(g2_real_lo)
    g2_hi = _r32_up(g2_real_hi)
    b2 = np.float64(host["one_minus_b2_32"])
    # v = R32(c * g2 + v_scaled), c > 0 -> monotone in g2
    v_lo = _r32_down(b2 * g2_lo + v_scaled64)
    v_hi = _r32_up(b2 * g2_hi + v_scaled64)
    v_lo = np.maximum(v_lo, 0.0)
    # s = sqrtf(v), b = R32(s / bc2), den = R32(b + eps)
    s_lo = _r32_down(np.sqrt(v_lo))
    s_hi = _r32_up(np.sqrt(v_hi))
    bc2 = np.float64(host["bc2_sqrt_32"])
    b_lo = _r32_down(s_lo / bc2)
    b_hi = _r32_up(s_hi / bc2)
    den_lo = _r32_down(b_lo + host["eps_32"])
    den_hi = _r32_up(b_hi + host["eps_32"])
    den_lo = np.maximum(den_lo, np.finfo(np.float64).tiny)
    # ratio = R32(m / den), den > 0: real extrema at rectangle corners
    r_real_lo, r_real_hi = _iv_div_positive(m_lo, m_hi, den_lo, den_hi)
    r_lo = _r32_down(r_real_lo)
    r_hi = _r32_up(r_real_hi)
    # theta_decay = R32(theta * d)
    theta_decay = (theta32 * np.float32(host["d_32"])).astype(np.float32)
    theta_decay64 = theta_decay.astype(np.float64)
    # theta = R32(h * ratio + theta_decay), h < 0 -> decreasing in ratio
    h32 = np.float64(host["h_32"])
    t_lo = _r32_down(h32 * r_hi + theta_decay64)
    t_hi = _r32_up(h32 * r_lo + theta_decay64)
    d_lo = np.nextafter(t_lo - theta_before, -np.inf)
    d_hi = np.nextafter(t_hi - theta_before, np.inf)
    return {
        "budget": budget,
        "g_lo": g_lo, "g_hi": g_hi,
        "K_m": (m_lo, m_hi), "K_v": (v_lo, v_hi),
        "K_theta": (t_lo, t_hi),
        "K_delta": (d_lo, d_hi),
        "theta_before": theta_before,
        "host": host,
    }

# ---------------------------------------------------------------------------
# stage artifacts: P (production two-step updates)
# ---------------------------------------------------------------------------


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{str(k): np.asarray(v) for k, v in arrays.items()})
    return sha256_file(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _state_arrays(state: Mapping[str, Any] | None, prefix: str,
                  arrays: dict) -> dict:
    """Copy m/v/step for one parameter into the artifact dict."""
    if state is None:
        return {"present": False}
    out = {"present": True}
    for key, tag in (("exp_avg", "m"), ("exp_avg_sq", "v")):
        value = state.get(key)
        if value is not None:
            arrays[f"{prefix}/{tag}"] = _to_np(value)
            out[tag] = True
    step = state.get("step")
    if step is not None:
        arrays[f"{prefix}/step"] = _to_np(step)
        raw = float(step) if torch.is_tensor(step) else float(step)
        out["step"] = raw
    return out


def _register_counting_hooks(trainer: Trainer, counter: CallCounter,
                             model_type: str) -> list:
    handles = [
        trainer.conditioner.register_forward_hook(
            lambda *args: counter.add("conditioner_forwards")),
        trainer.model.register_forward_hook(
            lambda *args: counter.add("backbone_forwards")),
    ]
    if model_type == "diffusion":
        original = trainer.model.add_noise

        def counted_add_noise(*args, **kwargs):
            counter.add("native_add_noise_batches", 1)
            return original(*args, **kwargs)

        trainer.model.add_noise = counted_add_noise
    return handles


def _noise_unit_index() -> dict[tuple[int, int], int]:
    return {(u, m): (u - 1) * MICRO_COUNT + m
            for u in UPDATES for m in range(MICRO_COUNT)}


def run_p_stage(root: Path, stage: str, attempt: str, t0: float,
                budget: Mapping[str, Any], provider: SyntheticTrainingProvider,
                counter: CallCounter, roster: Mapping[str, Any],
                *, out_dir: Path | None = None,
                fixture_error: str | None = None,
                guard: Callable[[str], None] | None = None) -> dict:
    base = Path(out_dir or (stage_dir(root, stage) / attempt))
    base.mkdir(parents=True, exist_ok=True)
    _seed_stage_rng()
    deadline = t0 + float(budget["remaining_s"]) - float(budget["reserve_s"])

    def check(unit: str) -> None:
        _ensure_deadline(deadline, 0.0, unit)
        if guard is not None:
            guard(unit)

    results: dict[str, dict] = {}
    rows: list[dict] = []
    failures: list[str] = []
    gaps: list[str] = list([fixture_error] if fixture_error else [])
    noise = provider.noise
    unit_index = _noise_unit_index()
    rows.append(_row("model_setup:deterministic",
                     model="deterministic", roster_sha256=roster["sha256"],
                     n_parameters=roster["n_det"], status="synthetic_verified"))
    rows.append(_row("model_setup:diffusion",
                     model="diffusion", roster_sha256=roster["sha256"],
                     n_parameters=roster["n_diff"], status="synthetic_verified"))
    for model_type in MODELS:
        _ensure_deadline(deadline, 0.0, "p_model")
        check("p_model")
        model_arrays: dict[str, Any] = {}
        presence: dict[str, str] = {}
        state_presence: dict[str, dict] = {}
        step_facts: dict[str, Any] = {}
        report: dict[str, Any] = {"model": model_type, "updates": {},
                                  "problems": [], "failures": []}
        trainer = provider.build_trainer(model_type, with_optimizer=True,
                                         counter=counter, loss_mode="production",
                                         out_dir=base / f"_trainer_{model_type}")
        handles = _register_counting_hooks(trainer, counter, model_type)
        try:
            roster_problems_ = roster_problems(
                _live_roster_record(trainer, model_type, roster), roster)
            report["roster_problems"] = roster_problems_
            if roster_problems_:
                failures += [f"p_roster:{p}" for p in roster_problems_]
            opt_problems = optimizer_roster_problems(trainer, roster,
                                                     model_type)
            report["optimizer_problems"] = opt_problems
            if opt_problems:
                failures += [f"p_optimizer:{p}" for p in opt_problems]
            named = trainer._named_parameters()
            theta_init = {name: param.detach().clone()
                          for name, param in named}
            for name, value in theta_init.items():
                model_arrays[f"theta_init/{name}"] = _to_np(value)
            prev_post: dict[str, Any] | None = None
            prev_state: dict[str, Any] | None = None
            for u in UPDATES:
                _ensure_deadline(deadline, 0.0, f"p_update_{u}")
                check(f"p_update_{u}")
                step_obs: dict = {}
                micro_records: list[dict] = []
                trainer.begin_update()
                for m in range(MICRO_COUNT):
                    _ensure_deadline(deadline, 0.0, f"p_u{u}_m{m}")
                    check(f"p_u{u}_m{m}")
                    unit_i = unit_index[(u, m)]
                    unit = noise[unit_i]
                    gen = None
                    if model_type == "diffusion":
                        gen = torch.Generator(device="cpu")
                        gen.set_state(unit["generator_state_before"])
                        counter.add("cpu_draw_units")
                    obs: dict = {}
                    rng_before = _rng_state()
                    counter.add("micro_execution_units")
                    loss, parts = trainer.accumulate_micro(
                        batch=provider.micro_batch(u, m),
                        noise_generator=gen, observer=obs,
                        pre_backward=lambda: check(f"p_backward_{u}_{m}"),
                        post_backward=lambda: counter.add(
                            "production_parameter_backward"))
                    rng_after = _rng_state()
                    record = {
                        "update": u, "micro": m,
                        "windows": list(MICRO_SCHEDULE[u][m]),
                        "loss": float(loss.detach()),
                        "parts": dict(parts),
                        "prediction_hook_fires": int(
                            obs.get("prediction_hook_fires", -1)),
                        "rng_before": _rng_digest(rng_before),
                        "rng_after": _rng_digest(rng_after),
                    }
                    if record["prediction_hook_fires"] != 1:
                        failures.append(
                            f"p_prediction_hook:{model_type}:u{u}:m{m}:"
                            f"{record['prediction_hook_fires']}")
                    pred = obs["forward_capture"]["x0_pred"]
                    preds = obs.get("prediction_grads") or []
                    if len(preds) != 1:
                        failures.append(
                            f"p_prediction_grad_count:{model_type}:u{u}:m{m}"
                            f":{len(preds)}")
                    else:
                        model_arrays[f"pred/u{u}/m{m}"] = _to_np(pred)
                        model_arrays[f"pred_grad/u{u}/m{m}"] = _to_np(preds[0])
                        record["prediction_sha256"] = _sha_array(pred)
                        record["prediction_grad_sha256"] = _sha_array(
                            preds[0])
                        record["prediction_grad_norm"] = float(
                            preds[0].float().norm())
                    memory = obs["forward_capture"]["memory"]
                    record["memory_sha256"] = _sha_array(memory)
                    cap = obs["forward_capture"]
                    if model_type == "diffusion":
                        if not torch.equal(cap["t"].cpu(), unit["t"]):
                            failures.append(
                                f"p_noise_t:{model_type}:u{u}:m{m}")
                        if not torch.equal(cap["noise"].cpu(), unit["epsilon"]):
                            failures.append(
                                f"p_noise_epsilon:{model_type}:u{u}:m{m}")
                        if gen is not None and not torch.equal(
                                gen.get_state(), unit["generator_state_after"]):
                            failures.append(
                                f"p_generator_state:{model_type}:u{u}:m{m}")
                        model_arrays[f"t/u{u}/m{m}"] = _to_np(cap["t"])
                        model_arrays[f"noise/u{u}/m{m}"] = _to_np(cap["noise"])
                        model_arrays[f"x_t/u{u}/m{m}"] = _to_np(cap["x_t"])
                        x0_ref = provider.window_arrays  # documentation only
                        slots = MICRO_SCHEDULE[u][m]
                        x0 = np.stack([provider.window_arrays[w]["x0"]
                                       for w in slots], axis=0)
                        alpha = alpha_cumprod_reference(provider.config)
                        x_t_ref = independent_x_t(
                            x0, unit["epsilon"].numpy(), unit["t"].numpy(),
                            effective_alpha(alpha))
                        diff = np.abs(cap["x_t"].detach().cpu().numpy().astype(
                            np.float64) - x_t_ref)
                        bound = (TOLERANCE_PROPOSALS["alpha_x_t"]["atol"]
                                 + TOLERANCE_PROPOSALS["alpha_x_t"]["rtol"]
                                 * np.abs(x_t_ref))
                        record["x_t_max_ratio"] = float((diff / bound).max())
                        record["alpha_t_hash"] = _sha_array(unit["t"].numpy())
                    loss_obs = getattr(trainer.loss_fn, "last_observer", None)
                    if isinstance(loss_obs, dict) and loss_obs.get("L5") is not None:
                        record["L5"] = float(loss_obs["L5"].detach())
                    contributions = obs.get("leaf_contributions") or {}
                    counts = obs.get("leaf_counts") or {}
                    accumulated = obs.get("accumulated") or {}
                    for name, param in named:
                        key = f"u{u}/m{m}/{name}"
                        grad = accumulated.get(name)
                        component = contributions.get(name)
                        fired = int(counts.get(name, 0))
                        if grad is None:
                            presence[key] = "none"
                            if component is not None:
                                failures.append(
                                    f"p_leaf_none_with_component:{name}")
                        else:
                            model_arrays[f"accum/u{u}/m{m}/{name}"] = _to_np(grad)
                            if component is not None:
                                model_arrays[
                                    f"contrib/u{u}/m{m}/{name}"] = _to_np(component)
                            presence[key] = ("zero" if float(grad.abs().max())
                                             == 0.0 else "nonzero")
                            if fired != 1:
                                failures.append(
                                    f"p_leaf_hook_count:{name}:{fired}")
                    micro_records.append(record)
                check(f"p_clip_{u}")
                gnorm = trainer.clip_gradients(observer=step_obs)
                counter.add("native_global_clip")
                check(f"p_step_{u}")
                trainer.step_optimizer(observer=step_obs)
                counter.add("native_adamw_steps")
                counter.add("diagnostic_parameter_update_events")
                update_record = {
                    "gnorm": float(gnorm),
                    "clip_coef": float(step_obs.get("clip_coef", 0.0)),
                    "clip_triggered": bool(step_obs.get("clip_triggered")),
                    "micros": micro_records,
                }
                if not _nonneg(update_record["gnorm"]):
                    failures.append(f"p_gnorm:{model_type}:u{u}")
                for name, param in named:
                    before = step_obs["theta_before"].get(name)
                    after = step_obs["theta_after"].get(name)
                    pre = step_obs["preclip"].get(name)
                    post = step_obs["postclip"].get(name)
                    if before is not None:
                        model_arrays[f"theta_before/u{u}/{name}"] = _to_np(before)
                    if after is not None:
                        model_arrays[f"theta_after/u{u}/{name}"] = _to_np(after)
                    if pre is not None:
                        model_arrays[f"preclip/u{u}/{name}"] = _to_np(pre)
                    if post is not None:
                        model_arrays[f"postclip/u{u}/{name}"] = _to_np(post)
                    state_before = (step_obs.get("state_before") or {}).get(name)
                    state_after = (step_obs.get("state_after") or {}).get(name)
                    state_presence[f"u{u}/{name}"] = {
                        "before": _state_arrays(
                            state_before, f"state_before/u{u}/{name}",
                            model_arrays),
                        "after": _state_arrays(
                            state_after, f"state_after/u{u}/{name}",
                            model_arrays),
                    }
                    step_val = state_presence[f"u{u}/{name}"]["after"].get("step")
                    step_facts[f"u{u}/{name}"] = step_val
                # step 1 post -> step 2 pre continuity
                if prev_post is not None:
                    cont = _continuity_problems(
                        prev_post, step_obs["theta_before"], "theta")
                    cont += _continuity_problems(
                        prev_state, step_obs["state_before"], "state")
                    update_record["continuity_problems"] = cont
                    if cont:
                        failures += [f"p_continuity:{model_type}:u{u}:{p}"
                                     for p in cont]
                report["updates"][f"u{u}"] = update_record
                prev_post = {name: p.detach().clone()
                             for name, p in trainer._named_parameters()}
                prev_state = step_obs["state_after"]
        finally:
            for handle in handles:
                handle.remove()
        report["none_parameters"] = sorted(
            name for name, p in trainer._named_parameters()
            if all(presence.get(f"u{u}/m{m}/{name}") == "none"
                   for u in UPDATES for m in range(MICRO_COUNT)))
        model_arrays["presence"] = json.dumps(presence).encode()
        npz_path = base / f"p_{model_type}.npz"
        arrays_sha = _save_npz(npz_path, model_arrays)
        report["arrays_path"] = path_for_record(npz_path)
        report["arrays_sha256"] = arrays_sha
        report["leaf_presence"] = presence
        report["state_presence"] = state_presence
        report["step_facts"] = step_facts
        results[model_type] = report
        for u in UPDATES:
            rows.append(_row(
                f"input:{model_type}:u{u}:m0",
                model=model_type, update=u, micro=0,
                windows=list(MICRO_SCHEDULE[u][0]),
                status="synthetic_verified"))
    # input rows are per micro; the loop above only emitted m0 as a header.
    rows = [r for r in rows if not r["key"].startswith("input:")]
    for model_type in MODELS:
        for u in UPDATES:
            for m in range(MICRO_COUNT):
                rec = results[model_type]["updates"][f"u{u}"]["micros"][m]
                rows.append(_row(
                    f"input:{model_type}:u{u}:m{m}", model=model_type,
                    update=u, micro=m, windows=rec["windows"],
                    prediction_sha256=rec.get("prediction_sha256"),
                    prediction_grad_sha256=rec.get(
                        "prediction_grad_sha256"),
                    memory_sha256=rec["memory_sha256"],
                    prediction_hook_fires=rec["prediction_hook_fires"],
                    status="synthetic_verified"))
    catalog = row_catalog(roster)
    row_problems = _stage_row_problems(rows, catalog, "P")
    complete_evidence = (
        not row_problems
        and all(results[model].get("arrays_path") for model in MODELS))
    report = {
        "schema": "w1_real_training_update_p_stage_v1",
        "mode": "synthetic",
        "stage": stage, "attempt": attempt,
        "models": results,
        "rows": rows,
        "failures": failures,
        "gaps": gaps,
        "evidence_problems": row_problems,
        "complete_evidence": bool(complete_evidence),
        "numeric_passed": not failures,
        "call_delta": None,
        "passed": not failures and not gaps,
    }
    write_json(base / "p_rows.json", rows)
    write_json(base / "p_report.json", report)
    return {"passed": not failures and not gaps, "failures": failures,
            "gaps": gaps, "rows": rows,
            "complete_evidence": bool(complete_evidence),
            "evidence_problems": row_problems,
            "numeric_passed": not failures,
            "paths": {
                model: results[model]["arrays_path"] for model in MODELS},
            "report": path_for_record(base / "p_report.json"),
            "arrays_sha256": {model: results[model]["arrays_sha256"]
                              for model in MODELS},
            "call_delta": None}


def _live_roster_record(trainer: Trainer, model_type: str,
                        roster: Mapping[str, Any]) -> dict:
    entries = _parameter_entries(trainer.model, "model", 0)
    entries += _parameter_entries(trainer.conditioner, "conditioner",
                                  len(entries))
    by_ptr: dict[int, list[str]] = {}
    for entry in entries:
        by_ptr.setdefault(entry["storage_ptr"], []).append(entry["name"])
    for entry in entries:
        shared = by_ptr[entry["storage_ptr"]]
        entry["shared_with"] = sorted(n for n in shared if n != entry["name"])
        entry.pop("storage_ptr")
    expected = dict(roster)
    models = dict(expected.get("models") or {})
    models[model_type] = {"parameters": entries,
                          "n_parameters": len(entries),
                          "n_elements": sum(e["numel"] for e in entries)}
    expected["models"] = models
    return expected


def _continuity_problems(previous: Mapping[str, Any] | None,
                         current: Mapping[str, Any] | None,
                         label: str) -> list[str]:
    if previous is None or current is None:
        return [f"{label}_missing"]
    if set(previous) != set(current):
        return [f"{label}_keys"]
    problems: list[str] = []
    for name in previous:
        a, b = previous[name], current[name]
        if a is None or b is None:
            if (a is None) != (b is None):
                problems.append(f"{label}:{name}:presence")
            continue
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            problems.extend(_continuity_state(a, b, label, name))
        elif not _bitwise_equal_tensor(a, b):
            problems.append(f"{label}:{name}")
    return problems


def _continuity_state(a: Mapping[str, Any], b: Mapping[str, Any],
                      label: str, name: str) -> list[str]:
    problems: list[str] = []
    keys = {"exp_avg", "exp_avg_sq", "step"}
    if set(a) != set(b):
        return [f"{label}:{name}:state_keys"]
    for key in keys & set(a):
        if not _bitwise_equal_tensor(a[key], b[key]):
            problems.append(f"{label}:{name}:{key}")
    return problems

# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------


def compare_numeric(got: Any, ref: Any, spec: Mapping[str, Any],
                    *, name: str) -> dict:
    raw_g, raw_r = np.asarray(got), np.asarray(ref)
    if (raw_g.dtype == np.bool_ or raw_r.dtype == np.bool_
            or raw_g.dtype.kind in "OUS" or raw_r.dtype.kind in "OUS"):
        return {"problems": [f"{name}:bool_or_non_numeric"], "max_abs": None,
                "max_ratio": None, "ok": False}
    g = raw_g.astype(np.float64)
    r = raw_r.astype(np.float64)
    problems: list[str] = []
    if g.shape != r.shape:
        return {"problems": [f"{name}:shape"], "max_abs": None,
                "max_ratio": None, "ok": False}
    if g.size == 0:
        return {"problems": [f"{name}:empty"], "max_abs": None,
                "max_ratio": None, "ok": False}
    if not (np.isfinite(g).all() and np.isfinite(r).all()):
        return {"problems": [f"{name}:nonfinite"], "max_abs": None,
                "max_ratio": None, "ok": False}
    diff = np.abs(g - r)
    bound = float(spec["atol"]) + float(spec["rtol"]) * np.abs(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_array = np.where(bound > 0.0, diff / bound,
                               np.where(diff == 0.0, 0.0, np.inf))
    ratio = float(ratio_array.max())
    ok = math.isfinite(ratio) and ratio <= 1.0
    if not ok:
        problems.append(f"{name}:tolerance")
    return {"problems": problems, "max_abs": float(diff.max()),
            "max_ratio": ratio if math.isfinite(ratio) else None,
            "ok": ok}


def _zero_noise_equivalence(g: np.ndarray, r: np.ndarray, *,
                            max_abs_diff: float, zero_abs: float,
                            noise_scale: float) -> dict:
    """Absolute zero-floor equivalence, restricted to genuine cancellation.

    The calibrated floor only excuses float32 cancellation noise around a
    mathematically zero tensor.  It must not excuse erasing a nonzero
    reference (fully or in a tensor whose scale exceeds the observed noise
    band); those remain nonzero comparisons for the relative-L2/cosine gate.
    """
    rn = float(np.linalg.norm(r))
    gn = float(np.linalg.norm(g))
    erased = (r != 0.0) & (g == 0.0)
    full_erasure = rn > 0.0 and gn == 0.0
    record = {
        "zero_abs": zero_abs,
        "noise_scale": noise_scale,
        "nonzero_erased": int(np.count_nonzero(erased)),
        "full_erasure": bool(full_erasure),
        "max_abs_diff": float(max_abs_diff),
    }
    ok = (max_abs_diff <= zero_abs
          and max(rn, gn) <= noise_scale
          and not full_erasure)
    if ok:
        record["zero_noise_floor"] = zero_abs
    return {"ok": ok, "record": record}


GRADIENT_SLICE_POS = {"pos": slice(0, 3), "rot6d": slice(3, 9),
                      "joint": slice(9, 38)}


def _gradient_group_gate(got: Any, ref: Any, *,
                         name: str) -> tuple[list[str], dict]:
    """p/rot6d/q group relative-L2/cosine gate for a 38-dim tensor.

    The whole-tensor relative-L2 can hide an exactly-zero group when the
    remaining groups are large; the plan requires the three registered groups
    to pass their own gates.  A group whose reference norm is below the
    nonzero band uses the calibrated absolute zero-noise floor only when the
    production norm is inside the same deep-noise band; an exactly zero
    production group with a non-zero reference is an erasure.
    """
    g_all = np.asarray(got, dtype=np.float64)
    r_all = np.asarray(ref, dtype=np.float64)
    if g_all.shape != r_all.shape or g_all.ndim == 0:
        return [f"{name}:group_shape"], {}
    if g_all.shape[-1] != 38:
        return [f"{name}:group_width"], {}
    flat_g = g_all.reshape(-1, 38)
    flat_r = r_all.reshape(-1, 38)
    problems: list[str] = []
    record: dict[str, dict] = {}
    for label, sl in GRADIENT_SLICE_POS.items():
        g = flat_g[:, sl].reshape(-1)
        r = flat_r[:, sl].reshape(-1)
        diff = g - r
        rn = float(np.linalg.norm(r))
        gn = float(np.linalg.norm(g))
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        entry: dict[str, Any] = {
            "reference_norm": rn, "production_norm": gn,
            "max_abs_diff": max_abs, "coordinate_count": int(g.size),
            "relative_l2": None, "cosine": None, "ok": True,
        }
        if rn >= GROUP_GATE["nonzero_reference_norm"]:
            rel = float(np.linalg.norm(diff) / rn)
            cos = (float(np.dot(g, r) / (gn * rn))
                   if gn > 0.0 and rn > 0.0 else None)
            ok = (rel <= GROUP_GATE["relative_l2"]
                  and cos is not None and cos >= GROUP_GATE["cosine_min"]
                  and gn > 0.0)
            entry.update(relative_l2=rel, cosine=cos)
            if not ok:
                problems.append(f"{name}:group_{label}")
        else:
            # The absolute floor is only valid when both norms sit inside the
            # deep cancellation band; a production group far outside it is a
            # real difference, not zero noise.
            deep = max(rn, gn) <= ZERO_GRADIENT_NOISE_SCALE
            ok = (deep and max_abs <= ZERO_GRADIENT_ABS
                  and not (rn > 0.0 and gn == 0.0))
            entry["tiny_gate_abs"] = ZERO_GRADIENT_ABS
            entry["tiny_noise_scale"] = ZERO_GRADIENT_NOISE_SCALE
            if not ok:
                problems.append(f"{name}:tiny_group_{label}")
        entry["ok"] = bool(ok)
        record[label] = entry
    return problems, record


def compare_gradient(got: Any, ref: Any, spec: Mapping[str, Any],
                     *, name: str) -> dict:
    """Elementwise gate plus relative-L2/cosine for a non-tiny tensor."""
    base = compare_numeric(got, ref, spec, name=name)
    g = np.asarray(got, dtype=np.float64).reshape(-1)
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    if g.size == 0 or g.shape != r.shape:
        return base
    rn = float(np.linalg.norm(r))
    gn = float(np.linalg.norm(g))
    diff = g - r
    record = {
        "reference_norm": rn, "production_norm": gn,
        "relative_l2": (float(np.linalg.norm(diff) / rn)
                        if rn >= 1e-10 else None),
        "cosine": (float(np.dot(g, r) / (gn * rn))
                   if gn > 0.0 and rn > 0.0 else None),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }
    problems = list(base["problems"])
    zero_abs = float(ZERO_GRADIENT_ABS)
    noise_scale = float(ZERO_GRADIENT_NOISE_SCALE)
    tiny_abs = float(GROUP_GATE["tiny_abs"])
    if (rn <= zero_abs and gn <= zero_abs
            and max(rn, gn) <= noise_scale):
        zero = _zero_noise_equivalence(
            g, r, max_abs_diff=record["max_abs_diff"], zero_abs=zero_abs,
            noise_scale=noise_scale)
        record.update(zero["record"])
        if zero["ok"]:
            # Mathematically-zero cancellation noise only; no nonzero
            # coordinate was erased and both sides are inside the deep band.
            record["zero_noise_floor"] = zero_abs
        else:
            record["zero_structure_mismatch"] = True
            problems.append(f"{name}:zero_erased")
    elif rn >= 1e-10:
        if not (record["relative_l2"] <= GROUP_GATE["relative_l2"]
                and record["cosine"] >= GROUP_GATE["cosine_min"] and gn > 0):
            problems.append(f"{name}:group")
    else:
        # Reference below the nonzero band.  A production tensor outside the
        # deep-noise band is not excused by the absolute floor.
        record["tiny_equivalence_abs"] = tiny_abs
        if max(gn, rn) > tiny_abs or record["max_abs_diff"] > tiny_abs:
            problems.append(f"{name}:tiny")
    return {**base, "problems": problems, "groups": record,
            "ok": not problems}


def compare_cotangent_gradient(got: Any, ref: Any,
                               spec: Mapping[str, Any], *,
                               name: str) -> dict:
    """Whole-tensor gate plus the registered p/rot6d/joint group gates."""
    base = compare_gradient(got, ref, spec, name=name)
    problems = list(base["problems"])
    group_problems, group_record = _gradient_group_gate(got, ref, name=name)
    problems += group_problems
    groups = dict(base.get("groups") or {})
    groups["group_gates"] = group_record
    group_ratio = 0.0
    for entry in group_record.values():
        rel = entry.get("relative_l2")
        if rel is not None:
            group_ratio = max(
                group_ratio,
                float(rel) / float(GROUP_GATE["relative_l2"]))
    return {**base, "problems": problems, "groups": groups,
            "group_gate_problems": group_problems,
            "group_max_ratio": group_ratio,
            "ok": not problems}


def _float(x: Any) -> float:
    return float(x.detach()) if torch.is_tensor(x) else float(x)


# ---------------------------------------------------------------------------
# independent FK reference context
# ---------------------------------------------------------------------------


class ReferenceContext:
    """One cache per I/F-I process: four target windows (240 states)."""

    def __init__(self, provider: SyntheticTrainingProvider,
                 counter: CallCounter, *,
                 guard: Callable[[str], None] | None = None) -> None:
        self.provider = provider
        self.counter = counter
        self._guard = guard
        tokenizer = provider.tokenizer
        constants = provider.constants
        self.const = constants
        self.mean = constants["mean32"].astype(np.float64)
        self.std = constants["std32"].astype(np.float64)
        self.lo = constants["lo32"].astype(np.float64)
        self.hi = constants["hi32"].astype(np.float64)
        self.chain = provider.chain
        self.target: dict[str, dict] = {}
        for key in WINDOW_IDS:
            self._check(f"reference_target_{key}")
            x0 = np.asarray(provider.window_arrays[key]["x0"],
                            dtype=np.float32).reshape(-1, 38)
            decoded = independent_decode(x0.astype(np.float64), self.mean,
                                         self.std, self.lo, self.hi)
            kp = synthetic_fk_np(self.chain, decoded["pos"], decoded["R"],
                                 decoded["q"])
            self.target[key] = {"kp": kp.reshape(-1, len(SITE_NAMES), 3),
                                "ok": bool(decoded["ok"].all()),
                                "boundary": bool(decoded["clamp_boundary"].any())}
            self.counter.add("reference_mj_forward", x0.shape[0])
        self.problems = [
            f"target_domain:{key}"
            for key, rec in self.target.items() if not rec["ok"]]

    def _check(self, unit: str,
               guard: Callable[[str], None] | None = None) -> None:
        active = guard if guard is not None else self._guard
        if active is not None:
            active(unit)

    def targets_for(self, slots: Sequence[str]) -> np.ndarray:
        return np.concatenate([self.target[w]["kp"] for w in slots], axis=0)

    def evaluate(self, pred_np: np.ndarray, slots: Sequence[str], *,
                 guard: Callable[[str], None] | None = None) -> dict:
        self._check("reference_evaluate", guard)
        x0 = np.asarray(pred_np, dtype=np.float32).reshape(-1, 38)
        decoded = independent_decode(x0.astype(np.float64), self.mean,
                                     self.std, self.lo, self.hi)
        self._check("reference_fk", guard)
        kp = synthetic_fk_np(self.chain, decoded["pos"], decoded["R"],
                             decoded["q"])
        self.counter.add("reference_mj_forward", x0.shape[0])
        self._check("reference_jacobian", guard)
        _, jac = synthetic_fk_jac_np(self.chain, decoded["pos"], decoded["R"],
                                     decoded["q"])
        self.counter.add("reference_mj_forward", x0.shape[0])
        self.counter.add("reference_mj_jac_site",
                         x0.shape[0] * len(SITE_NAMES))
        Rt = np.swapaxes(decoded["R"], 1, 2)
        J35 = np.array(jac, copy=True)
        J35[..., 3:6] = np.einsum("nsck,nkj->nscj", jac[..., 3:6], Rt)
        D, ddom = decode_jacobian_np(x0, self.std, self.lo, self.hi)
        J38 = np.einsum("nsck,nkj->nscj", J35, D)
        kp_target = self.targets_for(slots)
        ref = fk_loss_and_gradient(J38, kp.reshape(-1, len(SITE_NAMES), 3),
                                   kp_target)
        problems: list[str] = []
        if not decoded["ok"].all():
            problems.append("pred_decode_domain")
        if decoded["clamp_boundary"].any():
            problems.append("pred_clamp_boundary")
        if not ddom["ok"].all():
            problems.append("pred_jacobian_domain")
        return {
            "elements": ref["elements"],
            "l_fk": ref["l_fk"],
            "grad_raw": ref["grad_raw"].reshape(-1, WINDOW, 38),
            "kp": kp,
            "problems": problems,
        }


def _p_sidecar(p_dir: Path, model_type: str) -> dict:
    report = read_json(Path(p_dir) / "p_report.json")
    rec = dict(report["models"][model_type])
    arrays = _load_npz(Path(p_dir) / f"p_{model_type}.npz")
    rec["arrays"] = arrays
    return rec


def _param_slice(arrays: Mapping[str, np.ndarray], prefix: str,
                 name: str) -> np.ndarray | None:
    key = f"{prefix}/{name}"
    return arrays.get(key)


def _state_values(rec: Mapping[str, Any], tag: str, u: int,
                  name: str) -> dict:
    arrays = rec["arrays"]
    presence = (rec.get("state_presence") or {}).get(f"u{u}/{name}") or {}
    side = presence.get(tag) or {}
    if not side.get("present"):
        return {"present": False, "m": None, "v": None, "step": None}
    return {
        "present": True,
        "m": arrays.get(f"state_{tag}/u{u}/{name}/m"),
        "v": arrays.get(f"state_{tag}/u{u}/{name}/v"),
        "step": side.get("step"),
        "step_array": arrays.get(f"state_{tag}/u{u}/{name}/step"),
    }


def _clip_independent(grads: Mapping[str, np.ndarray | None],
                      grad_clip: float) -> dict:
    total = 0.0
    elems = 0
    for value in grads.values():
        if value is None:
            continue
        a = np.asarray(value, dtype=np.float64)
        total += float(np.sum(a * a))
        elems += int(a.size)
    norm = math.sqrt(total) if elems else 0.0
    # Native clip_grad_norm_ computes clip/(norm+1e-6) and scales whenever the
    # coefficient is below one; norm == clip still shrinks the gradient.
    coef = min(1.0, float(grad_clip) / (norm + 1e-6))
    post = {name: (None if value is None else
                   (np.asarray(value, np.float32)
                    * np.float32(coef)).astype(np.float32))
            for name, value in grads.items()}
    return {"norm": norm, "coef": coef, "postclip": post,
            "triggered": bool(coef < 1.0)}


def _step_json(value: Any) -> Any:
    if value is None:
        return None
    raw = float(value)
    if not math.isfinite(raw):
        return "nonfinite"
    return int(raw) if raw == int(raw) else raw

# ---------------------------------------------------------------------------
# I (independent reference stage)
# ---------------------------------------------------------------------------


def _bitwise_array(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    x = np.asarray(a)
    y = np.asarray(b)
    return (x.dtype == y.dtype and x.shape == y.shape
            and np.array_equal(x, y))


def _copy_theta(trainer: Trainer, theta: Mapping[str, np.ndarray]) -> None:
    with torch.no_grad():
        for name, param in trainer._named_parameters():
            value = theta.get(name)
            if value is None:
                continue
            param.data.copy_(torch.from_numpy(np.asarray(value)).to(
                param.device, param.dtype))


def _zeros_like_theta(theta: Mapping[str, np.ndarray]) -> dict:
    return {name: np.zeros_like(np.asarray(value, dtype=np.float32))
            for name, value in theta.items()}


def _accumulate_f32(acc: dict, grads: Mapping[str, Any]) -> None:
    for name, value in grads.items():
        if value is None:
            continue
        cur = None if acc.get(name) is None else acc[name]
        add = np.asarray(value, dtype=np.float32)
        acc[name] = add.copy() if cur is None else (cur + add).astype(np.float32)


def _formula_state_problems(eff: Mapping[str, Any], theta: Any, m: Any,
                            v: Any, step_measured: Any, *,
                            m_none: bool, v_none: bool,
                            expected_step: int) -> list[str]:
    problems: list[str] = []
    if not _bitwise_array(eff["theta"], theta):
        problems.append("theta")
    if m is not None and not _bitwise_array(eff["m"], m):
        problems.append("m")
    if v is not None and not _bitwise_array(eff["v"], v):
        problems.append("v")
    if step_measured is not None:
        try:
            value = float(step_measured)
        except (TypeError, ValueError):
            problems.append("step_type")
            return problems
        if not math.isfinite(value):
            problems.append("step_finite")
        elif value != int(value):
            problems.append("step_not_integer")
        elif int(value) != int(expected_step):
            problems.append("step_value")
    return problems


def _zero_state_problems(ref_m: Any, ref_v: Any, got_m: Any, got_v: Any,
                         *, m_none: bool, v_none: bool) -> list[str]:
    problems: list[str] = []
    for tag, ref, got in (("m", ref_m, got_m), ("v", ref_v, got_v)):
        r = np.asarray(ref, dtype=np.float64)
        g = np.asarray(got, dtype=np.float64)
        if r.shape != g.shape:
            problems.append(f"{tag}_shape")
            continue
        if (r < 0).any() and tag == "v":
            problems.append("ref_v_negative")
        if (g < 0).any() and tag == "v":
            problems.append("got_v_negative")
        nonzero_erased = (r != 0.0) & (g == 0.0)
        zero_invented = (r == 0.0) & (g != 0.0)
        if nonzero_erased.any():
            problems.append(f"{tag}_nonzero_erased")
        if zero_invented.any():
            problems.append(f"{tag}_zero_invented")
        sign = (np.sign(r) != np.sign(g)) & (r != 0.0) & (g != 0.0)
        if sign.any():
            problems.append(f"{tag}_sign")
    return problems


def _delta_observability(delta_ref64: Any, delta_measured: Any,
                         bound: Any, theta_new: Any, theta_before: Any,
                         *, gate_rtol: float) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(delta_ref64, dtype=np.float64)
    measured = np.asarray(delta_measured, dtype=np.float64)
    delta_bound = np.asarray(bound, dtype=np.float64)
    changed = np.asarray(theta_new, np.float32) != np.asarray(theta_before,
                                                              np.float32)
    observable = ((ref != 0.0)
                  & (delta_bound
                     <= DELTA_OBSERVABILITY_FRACTION * float(gate_rtol)
                     * np.abs(ref))
                  & changed)
    ok = np.where(
        ref == 0.0, measured == 0.0,
        np.where(observable,
                 np.abs(measured - ref) <= float(gate_rtol) * np.abs(ref),
                 False))
    return observable, ok


def _state_cross_path_ratio(a: Any, b: Any,
                            spec: Mapping[str, Any]) -> tuple[float, int]:
    """Maximum normalized error over *every* coordinate the gate compares.

    The raw state gate is ``abs(a-b) <= atol + rtol*abs(b)`` with no floor, so
    calibration must use the same denominator and may not skip small states;
    only coordinates where both sides are exactly zero are equivalent without
    a ratio.  A non-zero difference at ``b == 0`` is unbounded.
    """
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(spec["atol"]) + float(spec["rtol"]) * np.abs(bb)
    nonzero = denom > 0.0
    unbounded = (~nonzero) & (aa != bb)
    if unbounded.any():
        return math.inf, int(unbounded.sum())
    if not nonzero.any():
        return 0.0, 0
    ratio = np.abs(aa[nonzero] - bb[nonzero]) / denom[nonzero]
    return float(ratio.max()), int(nonzero.sum())


def _cross_path_flags(p_m: Any, i_m: Any, p_v: Any, i_v: Any,
                      p_theta: Any, i_theta: Any,
                      p_delta_measured: Any, p_delta_ref: Any,
                      i_delta_ref: Any, p_obs: Any, i_obs: Any,
                      spec: Mapping[str, Any], *,
                      delta_rtol: float = 5e-4) -> dict:
    """Raw cross-path gates; do not apply the interval classification here.

    The delta close gate compares P's *measured* update with I's *unrounded*
    reference delta (plan section 6.2); comparing two measured, rounded
    updates lets the final float32 write hide a cross-path difference.  The
    zero/non-zero presence check compares the two unrounded references, so an
    unobservable but non-zero reference on both sides is not a divergence.
    """
    out: dict[str, Any] = {}
    for tag, a, b in (("m", p_m, i_m), ("v", p_v, i_v),
                      ("theta", p_theta, i_theta)):
        if a is None or b is None:
            out[f"{tag}_ok"] = a is None and b is None
            continue
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape != bb.shape or not (np.isfinite(aa).all()
                                        and np.isfinite(bb).all()):
            out[f"{tag}_ok"] = False
            continue
        tiny_abs = float(ZERO_STATE_ABS)
        # The raw cross-path gate stays a relative comparison: an absolute
        # "both tiny" exemption would hide a 50% difference or a one-sided
        # zero.  Zero/sign divergence is recorded for classification and the
        # interval (not the floor) decides whether a failure is a gap.
        zero_disagree = ((aa == 0.0) != (bb == 0.0))
        sign_disagree = ((np.sign(aa) != np.sign(bb))
                         & (aa != 0.0) & (bb != 0.0))
        close = ((np.abs(aa - bb)
                  <= float(spec["atol"]) + float(spec["rtol"]) * np.abs(bb))
                 | ((aa == 0.0) & (bb == 0.0)))
        out[f"{tag}_tiny_zero_floor"] = tiny_abs
        out[f"{tag}_zero_disagreement"] = bool(zero_disagree.any())
        out[f"{tag}_sign_disagreement"] = bool(sign_disagree.any())
        out[f"{tag}_close"] = bool(close.all())
        out[f"{tag}_ok"] = bool(close.all())
    p_delta_nz = np.asarray(p_delta_ref, np.float64) != 0.0
    i_delta_nz = np.asarray(i_delta_ref, np.float64) != 0.0
    out["delta_presence_mismatch"] = bool(
        (p_delta_nz != i_delta_nz).any())
    mask = np.asarray(p_obs, bool) & np.asarray(i_obs, bool)
    one_sided = np.asarray(p_obs, bool) != np.asarray(i_obs, bool)
    out["delta_one_sided_observable"] = bool(one_sided.any())
    if out["delta_presence_mismatch"] or out["delta_one_sided_observable"]:
        out["delta_close"] = False
        out["delta_ok"] = False
    elif mask.any():
        a = np.asarray(p_delta_measured, np.float64)[mask]
        b = np.asarray(i_delta_ref, np.float64)[mask]
        close = np.abs(a - b) <= float(delta_rtol) * np.abs(b)
        out["delta_close"] = bool(close.all())
        out["delta_ok"] = bool(close.all())
    else:
        out["delta_close"] = True
        out["delta_ok"] = True
    return out


def _inside(interval: tuple, value: Any, mask: Any | None = None) -> bool:
    lo, hi = interval
    v = np.asarray(value, dtype=np.float64)
    ok = (v >= np.asarray(lo) - 1e-300) & (v <= np.asarray(hi) + 1e-300)
    if mask is not None:
        ok = ok | (~np.asarray(mask, bool))
    return bool(ok.all())


def _classify_adamw(flags: Mapping[str, Any], same_input_ok: bool,
                    inside: Mapping[str, bool], *,
                    zero_problems: Sequence[str]) -> dict:
    failures: list[str] = []
    gaps: list[str] = []
    if not same_input_ok:
        failures.append("same_input_formula")
    failures.extend(zero_problems)
    raw_ok = all(bool(flags.get(f"{tag}_ok", True))
                 for tag in ("m", "v", "theta", "delta"))
    if failures:
        classification = "direct_failure"
    elif not raw_ok:
        if all(bool(inside.get(k, False))
               for k in ("m", "v", "theta", "delta")):
            classification = "cross_path_sensitivity_gap"
            gaps.append("cross_path_sensitivity_gap")
        else:
            classification = "cross_path_bound_unverifiable"
            gaps.append("cross_path_bound_unverifiable")
    else:
        classification = "passed"
    return {"classification": classification, "failures": failures,
            "gaps": gaps, "raw_cross_path_passed": not failures and raw_ok,
            "same_input_formula_passed": same_input_ok}


def _adamw_param_row(key: str, *, name: str, model: str, u: int,
                     p_pre: Mapping[str, Any], p_post: Mapping[str, Any],
                     i_post: Mapping[str, Any], p_g: np.ndarray | None,
                     i_g: np.ndarray | None,
                     p_step: Any, i_step: Any, host: Mapping[str, Any],
                     arrays: dict, tolerance: Mapping[str, Any]) -> dict:
    """Build one adamw_parameter_state row and its buffers."""
    if p_g is None or i_g is None:
        return _row(key, model=model, update=u, parameter=name,
                    status="none_parameter", failure=None, gaps=[],
                    classification="none_parameter",
                    same_input_formula_passed=True,
                    raw_cross_path_passed=True,
                    delta_observable_p=False, delta_observable_i=False,
                    delta_ref_p=0.0, delta_ref_i=0.0,
                    delta_measured_p=0.0, delta_measured_i=0.0,
                    bound_delta_p=0.0, bound_delta_i=0.0)
    theta_pre = np.asarray(p_pre["theta"], np.float32)
    p_theta = np.asarray(p_post["theta"], np.float32)
    p_m = None if p_post["m"] is None else np.asarray(p_post["m"], np.float32)
    p_v = None if p_post["v"] is None else np.asarray(p_post["v"], np.float32)
    i_theta = np.asarray(i_post["theta"], np.float32)
    i_m = None if i_post["m"] is None else np.asarray(i_post["m"], np.float32)
    i_v = None if i_post["v"] is None else np.asarray(i_post["v"], np.float32)
    m_none = p_pre["m_none"]
    v_none = p_pre["v_none"]
    m_old = None if m_none else np.asarray(p_pre["m"], np.float32)
    v_old = None if v_none else np.asarray(p_pre["v"], np.float32)
    pe = adamw_effective_fp32(theta_pre, m_old, v_old, p_g, int(p_step),
                              m_none=m_none, v_none=v_none, host=host)
    ie = adamw_effective_fp32(theta_pre, m_old, v_old, i_g, int(i_step),
                              m_none=m_none, v_none=v_none, host=host)
    p_formula = _formula_state_problems(pe, p_theta, p_m, p_v,
                                        p_post.get("step"),
                                        m_none=m_none, v_none=v_none,
                                        expected_step=u)
    i_formula = _formula_state_problems(ie, i_theta, i_m, i_v,
                                        i_step,
                                        m_none=m_none, v_none=v_none,
                                        expected_step=u)
    zero_problems = (_zero_state_problems(pe["m"], pe["v"], p_m, p_v,
                                          m_none=m_none, v_none=m_none)
                     + _zero_state_problems(ie["m"], ie["v"], i_m, i_v,
                                            m_none=m_none, v_none=m_none))
    delta_p_measured = p_theta.astype(np.float64) - theta_pre.astype(np.float64)
    delta_i_measured = i_theta.astype(np.float64) - theta_pre.astype(np.float64)
    b_p = _delta_write_bound(pe["theta"])
    b_i = _delta_write_bound(ie["theta"])
    gate_same = float(tolerance["delta_same_input"]["rtol"])
    gate_cross = float(tolerance["delta_cross_path"]["rtol"])
    p_obs, p_ok = _delta_observability(pe["delta_ref64"], delta_p_measured,
                                       b_p, pe["theta"], theta_pre,
                                       gate_rtol=gate_same)
    i_obs, i_ok = _delta_observability(ie["delta_ref64"], delta_i_measured,
                                       b_i, ie["theta"], theta_pre,
                                       gate_rtol=gate_same)
    iv = propagate_intervals(i_g, theta_pre, m_old if m_old is not None else 0.0,
                             v_old if v_old is not None else 0.0,
                             int(i_step), m_none=m_none, v_none=v_none)
    p_m64 = None if p_m is None else p_m.astype(np.float64)
    p_v64 = None if p_v is None else p_v.astype(np.float64)
    i_m64 = None if i_m is None else i_m.astype(np.float64)
    i_v64 = None if i_v is None else i_v.astype(np.float64)
    theta_spec = tolerance["theta_cross_path"]
    flags = _cross_path_flags(
        p_m64, i_m64, p_v64, i_v64,
        p_theta.astype(np.float64), i_theta.astype(np.float64),
        delta_p_measured, pe["delta_ref64"], ie["delta_ref64"],
        p_obs, i_obs, tolerance["state_cross_path"], delta_rtol=gate_cross)
    theta_close = ((np.abs(p_theta.astype(np.float64)
                           - i_theta.astype(np.float64))
                    <= float(theta_spec["atol"])
                    + float(theta_spec["rtol"])
                    * np.abs(i_theta.astype(np.float64)))
                   | ((p_theta == 0.0) & (i_theta == 0.0)))
    theta_zero = (p_theta == 0.0) != (i_theta == 0.0)
    theta_sign = ((np.sign(p_theta) != np.sign(i_theta))
                  & (p_theta != 0.0) & (i_theta != 0.0))
    flags["theta_close"] = bool(theta_close.all())
    flags["theta_zero_disagreement"] = bool(theta_zero.any())
    flags["theta_sign_disagreement"] = bool(theta_sign.any())
    flags["theta_ok"] = bool(theta_close.all())
    m_mask = np.ones_like(i_g, dtype=bool) if not m_none else np.zeros_like(
        i_g, dtype=bool)
    v_mask = np.ones_like(i_g, dtype=bool) if not v_none else np.zeros_like(
        i_g, dtype=bool)
    inside = {
        "m": (_inside(iv["K_m"], p_m if p_m is not None else 0.0, m_mask)
              and _inside(iv["K_m"], i_m if i_m is not None else 0.0, m_mask)
              and _inside(iv["K_m"], pe["m"], m_mask)
              and _inside(iv["K_m"], ie["m"], m_mask)),
        "v": (_inside(iv["K_v"], p_v if p_v is not None else 0.0, v_mask)
              and _inside(iv["K_v"], i_v if i_v is not None else 0.0, v_mask)
              and _inside(iv["K_v"], pe["v"], v_mask)
              and _inside(iv["K_v"], ie["v"], v_mask)),
        "theta": (_inside(iv["K_theta"], p_theta)
                  and _inside(iv["K_theta"], i_theta)
                  and _inside(iv["K_theta"], pe["theta"])
                  and _inside(iv["K_theta"], ie["theta"])),
        "delta": (_inside(iv["K_delta"], delta_p_measured)
                  and _inside(iv["K_delta"], delta_i_measured)
                  and _inside(iv["K_delta"], pe["delta_ref64"])
                  and _inside(iv["K_delta"], ie["delta_ref64"])),
    }
    same_input_ok = not (p_formula or i_formula)
    verdict = _classify_adamw(flags, same_input_ok, inside,
                              zero_problems=zero_problems)
    p_nz = np.asarray(pe["delta_ref64"]) != 0.0
    i_nz = np.asarray(ie["delta_ref64"]) != 0.0
    p_bad = (~p_ok & p_obs) | ((~p_nz) & (np.asarray(delta_p_measured) != 0.0))
    i_bad = (~i_ok & i_obs) | ((~i_nz) & (np.asarray(delta_i_measured) != 0.0))
    if p_bad.any() or i_bad.any():
        verdict["failures"].append("same_input_delta")
        verdict["classification"] = "direct_failure"
    if (p_nz & ~p_obs).any() or (i_nz & ~i_obs).any():
        if "parameter_delta_precision_gap" not in verdict["gaps"]:
            verdict["gaps"].append("parameter_delta_precision_gap")
        if verdict["classification"] == "passed":
            verdict["classification"] = "audit_incomplete"
    # Buffers for F: raw inputs and every derived node.
    keyp = key.replace(":", "_").replace("/", "_")
    arrays[f"{keyp}/theta_pre"] = theta_pre
    arrays[f"{keyp}/theta_p"] = p_theta
    arrays[f"{keyp}/theta_i"] = i_theta
    for tag, val in (("m_p", p_m), ("v_p", p_v), ("m_i", i_m), ("v_i", i_v)):
        if val is not None:
            arrays[f"{keyp}/{tag}"] = val
    arrays[f"{keyp}/delta_ref_p"] = pe["delta_ref64"]
    arrays[f"{keyp}/delta_ref_i"] = ie["delta_ref64"]
    arrays[f"{keyp}/delta_measured_p"] = delta_p_measured
    arrays[f"{keyp}/delta_measured_i"] = delta_i_measured
    arrays[f"{keyp}/bound_delta_p"] = b_p
    arrays[f"{keyp}/bound_delta_i"] = b_i
    arrays[f"{keyp}/K_m_lo"], arrays[f"{keyp}/K_m_hi"] = iv["K_m"]
    arrays[f"{keyp}/K_v_lo"], arrays[f"{keyp}/K_v_hi"] = iv["K_v"]
    arrays[f"{keyp}/K_theta_lo"], arrays[f"{keyp}/K_theta_hi"] = iv["K_theta"]
    arrays[f"{keyp}/K_delta_lo"], arrays[f"{keyp}/K_delta_hi"] = iv["K_delta"]
    arrays[f"{keyp}/budget"] = iv["budget"]
    arrays[f"{keyp}/obs_p"] = p_obs.astype(np.uint8)
    arrays[f"{keyp}/obs_i"] = i_obs.astype(np.uint8)
    arrays[f"{keyp}/ok_p"] = p_ok.astype(np.uint8)
    arrays[f"{keyp}/ok_i"] = i_ok.astype(np.uint8)
    arrays[f"{keyp}/same_input_p"] = np.asarray(
        p_formula == [], dtype=np.uint8)
    arrays[f"{keyp}/same_input_i"] = np.asarray(
        i_formula == [], dtype=np.uint8)
    arrays[f"{keyp}/math_p_theta"] = adamw_math_reference(
        theta_pre, m_old if m_old is not None else 0.0,
        v_old if v_old is not None else 0.0, p_g, int(p_step),
        m_none=m_none, v_none=v_none, host=host)["theta"]
    arrays[f"{keyp}/math_i_theta"] = adamw_math_reference(
        theta_pre, m_old if m_old is not None else 0.0,
        v_old if v_old is not None else 0.0, i_g, int(i_step),
        m_none=m_none, v_none=v_none, host=host)["theta"]
    theta_bound = (float(theta_spec["atol"])
                   + float(theta_spec["rtol"]) * np.abs(i_theta.astype(np.float64)))
    max_theta_ratio = float(np.max(np.abs(
        p_theta.astype(np.float64) - i_theta.astype(np.float64)) / theta_bound))
    state_ratios = []
    state_applicable = 0
    state_unbounded = False
    for a, b in ((p_m, i_m), (p_v, i_v)):
        if a is None or b is None:
            continue
        ratio, count = _state_cross_path_ratio(
            a, b, tolerance["state_cross_path"])
        state_applicable += count
        if not math.isfinite(ratio):
            state_unbounded = True
        else:
            state_ratios.append(ratio)
    max_state_ratio = (None if state_unbounded
                       else max(state_ratios, default=0.0))
    delta_mask = np.asarray(p_obs, bool) & np.asarray(i_obs, bool)
    if delta_mask.any():
        delta_ref_i = np.asarray(ie["delta_ref64"], np.float64)
        max_delta_ratio = float(np.max(
            np.abs(np.asarray(delta_p_measured)[delta_mask]
                   - delta_ref_i[delta_mask])
            / (gate_cross * np.abs(delta_ref_i[delta_mask]))))
    else:
        max_delta_ratio = 0.0
    same_delta = []
    for obs, ref, measured in ((p_obs, pe["delta_ref64"],
                                delta_p_measured),
                               (i_obs, ie["delta_ref64"],
                                delta_i_measured)):
        mask = np.asarray(obs, bool)
        if mask.any():
            same_delta.append(float(np.max(
                np.abs(np.asarray(measured)[mask] - np.asarray(ref)[mask])
                / (gate_same * np.abs(np.asarray(ref)[mask])))))
    max_same_delta_ratio = max(same_delta, default=0.0)
    return _row(
        key, model=model, update=u, parameter=name,
        status="synthetic_verified",
        classification=verdict["classification"],
        max_theta_ratio=max_theta_ratio,
        max_state_ratio=max_state_ratio,
        state_ratio_applicable=state_applicable,
        state_ratio_unbounded=state_unbounded,
        max_delta_ratio=max_delta_ratio,
        max_same_delta_ratio=max_same_delta_ratio,
        same_input_formula_passed=verdict["same_input_formula_passed"],
        raw_cross_path_passed=verdict["raw_cross_path_passed"],
        same_input_problems_p=p_formula, same_input_problems_i=i_formula,
        zero_problems=zero_problems, cross_path_flags=flags,
        inside=inside,
        failure=";".join(verdict["failures"]) or None,
        gaps=verdict["gaps"],
        delta_observable_p=bool(p_obs.any()),
        delta_observable_i=bool(i_obs.any()),
        delta_ref_p=float(np.max(np.abs(pe["delta_ref64"]))),
        delta_ref_i=float(np.max(np.abs(ie["delta_ref64"]))),
        delta_measured_p=float(np.max(np.abs(delta_p_measured))),
        delta_measured_i=float(np.max(np.abs(delta_i_measured))),
        bound_delta_p=float(np.max(b_p)),
        bound_delta_i=float(np.max(b_i)),
        rounding_fallbacks_p=pe["rounding_fallbacks"],
        rounding_fallbacks_i=ie["rounding_fallbacks"],
    )

# --- I stage ---------------------------------------------------------------

EVIDENCE_FAILURE_PREFIXES = {
    "i_memory_replay", "i_prediction_replay", "i_noise_t", "i_noise_epsilon",
    "i_x_t_replay", "i_alpha_x_t", "i_roster",
    "i_unexpected_none_gradient", "i_step_tensor_missing", "i_step_tensor",
    "i_output_cotangent_missing", "i_objective_missing", "i_stage_aborted",
}


def _expected_stage_rows(catalog: Mapping[str, Any], stage: str) -> list[str]:
    if stage == "P":
        return [key for key in catalog["rows"]
                if key.startswith(("model_setup:", "input:"))]
    return [key for key in catalog["rows"]
            if not key.startswith(("model_setup:", "input:"))]


def _stage_row_problems(rows: Sequence[Mapping[str, Any]],
                        catalog: Mapping[str, Any], stage: str) -> list[str]:
    want = Counter(_expected_stage_rows(catalog, stage))
    got = Counter(str(r.get("key")) for r in rows)
    problems: list[str] = []
    if got != want:
        missing = want - got
        extra = got - want
        if missing:
            problems.append(f"{stage.lower()}_rows_missing:{len(missing)}")
        if extra:
            problems.append(f"{stage.lower()}_rows_extra:{len(extra)}")
        if any(v > 1 for v in got.values()):
            problems.append(f"{stage.lower()}_rows_duplicate")
    for row in rows:
        if row.get("status") == "missing":
            problems.append(f"{stage.lower()}_row_evidence:{row.get('key')}")
            break
    return problems


def _i_evidence_failures(failures: Sequence[str]) -> list[str]:
    return [f for f in failures
            if f.split(":", 1)[0] in EVIDENCE_FAILURE_PREFIXES]


def _declared_tolerance(tolerance: Mapping[str, Any]) -> dict:
    return dict(tolerance.get("tolerances") or TOLERANCE_PROPOSALS)


def run_i_stage(root: Path, stage: str, attempt: str, t0: float,
                budget: Mapping[str, Any], provider: SyntheticTrainingProvider,
                counter: CallCounter, roster: Mapping[str, Any],
                p_dir: Path, *, tolerance: Mapping[str, Any],
                out_dir: Path | None = None,
                guard: Callable[[str], None] | None = None) -> dict:
    base = Path(out_dir or (stage_dir(root, stage) / attempt))
    base.mkdir(parents=True, exist_ok=True)
    _seed_stage_rng()
    deadline = t0 + float(budget["remaining_s"]) - float(budget["reserve_s"])

    def check(unit: str) -> None:
        _ensure_deadline(deadline, 0.0, unit)
        if guard is not None:
            guard(unit)

    tol = _declared_tolerance(tolerance)
    p_reports = {model: _p_sidecar(p_dir, model) for model in MODELS}
    check("i_reference_context")
    ref_ctx = ReferenceContext(provider, counter, guard=check)
    failures: list[str] = list(ref_ctx.problems)
    gaps: list[str] = list(ref_ctx.problems)
    rows: list[dict] = []
    results: dict[str, dict] = {}
    host_cache: dict[int, dict] = {u: host_constants(
        u, ADAM_DEFAULTS, float(provider.config["lr"]),
        float(provider.config["weight_decay"])) for u in UPDATES}

    for model_type in MODELS:
        _ensure_deadline(deadline, 0.0, f"i_model_{model_type}")
        check(f"i_model_{model_type}")
        p_rec = p_reports[model_type]
        trainer = provider.build_trainer(model_type, with_optimizer=False,
                                         counter=counter,
                                         loss_mode="forbidden",
                                         out_dir=base / f"_trainer_{model_type}")
        handles = _register_counting_hooks(trainer, counter, model_type)
        model_arrays: dict[str, Any] = {}
        report: dict[str, Any] = {"model": model_type, "updates": {},
                                  "problems": []}
        names = [name for name, _ in trainer._named_parameters()]
        roster_actual = _live_roster_record(trainer, model_type, roster)
        problems = roster_problems(roster_actual, roster)
        if problems:
            failures += [f"i_roster:{p}" for p in problems]
        try:
            theta_init = {name: p_rec["arrays"][f"theta_init/{name}"]
                          for name in names}
            for u in UPDATES:
                _ensure_deadline(deadline, 0.0, f"i_u{u}")
                check(f"i_u{u}")
                pre_theta = {name: np.asarray(
                    p_rec["arrays"][f"theta_before/u{u}/{name}"], np.float32)
                    for name in names}
                _copy_theta(trainer, pre_theta)
                if u == 2:
                    counter.add("anchor_state_restores")
                    report.setdefault("anchor_restores", []).append({
                        "update": u,
                        "theta_matches_p_before": all(
                            np.array_equal(pre_theta[name],
                                           p_rec["arrays"][
                                               f"theta_before/u{u}/{name}"])
                            for name in names),
                    })
                pre_state = {name: _state_values(p_rec, "before", u, name)
                             for name in names}
                m_none = {name: not pre_state[name]["present"]
                          or pre_state[name]["m"] is None for name in names}
                v_none = {name: not pre_state[name]["present"]
                          or pre_state[name]["v"] is None for name in names}
                i_accum: dict[str, Any] = {name: None for name in names}
                i_micro_grads: dict[int, dict[str, Any]] = {}
                objective_facts: dict[int, dict] = {}
                cotangent_facts: dict[int, dict] = {}
                unit_index = _noise_unit_index()
                for m in range(MICRO_COUNT):
                    _ensure_deadline(deadline, 0.0, f"i_u{u}_m{m}")
                    check(f"i_u{u}_m{m}")
                    unit = provider.noise[unit_index[(u, m)]]
                    gen = None
                    if model_type == "diffusion":
                        gen = torch.Generator(device="cpu")
                        gen.set_state(unit["generator_state_before"])
                        counter.add("cpu_draw_units")
                    cap: dict = {}
                    counter.add("micro_execution_units")
                    memory, pred = trainer.predict(
                        provider.micro_batch(u, m), noise_generator=gen,
                        capture=cap)
                    batch = provider.micro_batch(u, m)
                    p_pred = p_rec["arrays"].get(f"pred/u{u}/m{m}")
                    p_mem = (p_rec.get("updates", {}).get(f"u{u}", {})
                             .get("micros", [{}] * MICRO_COUNT)[m]
                             .get("memory_sha256"))
                    mem_ok = p_mem is not None and _sha_array(memory) == p_mem
                    pred_ok = (p_pred is not None
                               and _bitwise_array(pred.detach().cpu().numpy(),
                                                  p_pred))
                    if not mem_ok:
                        failures.append(f"i_memory_replay:{model_type}:u{u}:m{m}")
                    if not pred_ok:
                        failures.append(f"i_prediction_replay:{model_type}:u{u}:m{m}")
                    if model_type == "diffusion":
                        if not torch.equal(cap["t"].cpu(), unit["t"]):
                            failures.append(f"i_noise_t:{model_type}:u{u}:m{m}")
                        if not torch.equal(cap["noise"].cpu(), unit["epsilon"]):
                            failures.append(
                                f"i_noise_epsilon:{model_type}:u{u}:m{m}")
                        p_x_t = p_rec["arrays"].get(f"x_t/u{u}/m{m}")
                        if p_x_t is None or not _bitwise_array(
                                cap["x_t"].detach().cpu().numpy(), p_x_t):
                            failures.append(
                                f"i_x_t_replay:{model_type}:u{u}:m{m}")
                        slots = MICRO_SCHEDULE[u][m]
                        x0 = np.stack([provider.window_arrays[w]["x0"]
                                       for w in slots], axis=0)
                        alpha = alpha_cumprod_reference(provider.config)
                        x_t_ref = independent_x_t(
                            x0, unit["epsilon"].numpy(), unit["t"].numpy(),
                            effective_alpha(alpha))
                        cmp_alpha = compare_numeric(
                            cap["x_t"].detach().cpu().numpy().astype(np.float64),
                            x_t_ref, tol["alpha_x_t"],
                            name=f"alpha_x_t:{model_type}:u{u}:m{m}")
                        if not cmp_alpha["ok"]:
                            failures.append(
                                f"i_alpha_x_t:{model_type}:u{u}:m{m}")
                    pred_det = pred.detach()
                    pred_np = pred_det.cpu().numpy().astype(np.float32).reshape(
                        -1, 38)
                    check(f"i_ref_fk_{u}_{m}")
                    ref = ref_ctx.evaluate(pred_np, MICRO_SCHEDULE[u][m],
                                           guard=check)
                    for gap in ref["problems"]:
                        gaps.append(f"i_fk_{gap}:{model_type}:u{u}:m{m}")
                    check(f"i_output_vjp_{u}_{m}")
                    pred64 = pred_det.cpu().double()
                    target64 = batch["x0"].detach().cpu().double()
                    domain = {"min_angle_rad": 1e-6, "min_sin_theta": 1e-3}
                    elements = independent_term_elements(
                        pred64, target64, rotation_domain=domain)
                    fk_elements = torch.tensor(
                        ref["elements"], dtype=torch.float64).reshape(
                        MICRO_COUNT, WINDOW, len(SITE_NAMES))
                    vjp, vjp_facts = independent_vjp(
                        pred_det.cpu(), batch["x0"].detach().cpu(),
                        fk_elements, LOSS_WEIGHTS, rotation_domain=domain)
                    counter.add("independent_output_vjp")
                    if not vjp_facts["geodesic_domain_ok"]:
                        gaps.append(
                            f"i_geodesic_domain:{model_type}:u{u}:m{m}:"
                            f"{vjp_facts['geodesic_domain_bad_count']}")
                    fk_grad = torch.tensor(ref["grad_raw"],
                                           dtype=torch.float64)
                    total_cot = (
                        COTANGENT_WEIGHTS["pos"] * vjp[0]
                        + COTANGENT_WEIGHTS["rot6d"] * vjp[1]
                        + COTANGENT_WEIGHTS["joint"] * vjp[2]
                        + COTANGENT_WEIGHTS["geodesic_rad2"] * vjp[3]
                        + COTANGENT_WEIGHTS["vel"] * vjp[4]
                        + COTANGENT_WEIGHTS["fk_m2"] * fk_grad)
                    scalars = independent_scalars(elements, LOSS_WEIGHTS,
                                                  fk_elements)
                    total_check = (scalars["L5"]
                                   + COTANGENT_WEIGHTS["fk_m2"] * scalars["fk_m2"])
                    if abs(float(total_check) - float(scalars["total"])) > (
                            1e-10 + 1e-10 * abs(float(total_check))):
                        failures.append(
                            f"i_objective_compose:{model_type}:u{u}:m{m}")
                    # output cotangent vs P
                    p_grad = p_rec["arrays"].get(f"pred_grad/u{u}/m{m}")
                    cot = (total_cot / 4.0).numpy()
                    cot_cmp = compare_cotangent_gradient(
                        p_grad, cot, tol["output_cotangent"],
                        name=f"cotangent:{model_type}:u{u}:m{m}")
                    if p_grad is None:
                        failures.append(
                            f"i_output_cotangent_missing:{model_type}:u{u}:m{m}")
                    elif not cot_cmp["ok"]:
                        failures.append(
                            f"i_output_cotangent:{model_type}:u{u}:m{m}")
                        failures += [
                            f"i_output_cotangent_{p}"
                            for p in (cot_cmp.get("group_gate_problems") or [])]
                    cotangent_facts[m] = {
                        "comparison": cot_cmp, "total_cot": cot,
                        "scalars": {k: float(v) for k, v in scalars.items()},
                        "domain": vjp_facts,
                    }
                    rows.append(_row(
                        f"cotangent:{model_type}:u{u}:m{m}",
                        model=model_type, update=u, micro=m,
                        windows=list(MICRO_SCHEDULE[u][m]),
                        production_sha256=_sha_array(p_grad)
                        if p_grad is not None else None,
                        independent_sha256=_sha_array(cot),
                        max_abs=cot_cmp.get("max_abs"),
                        max_ratio=cot_cmp.get("max_ratio"),
                        groups=cot_cmp.get("groups"),
                        group_gates=(cot_cmp.get("groups") or {}).get(
                            "group_gates"),
                        group_gate_problems=cot_cmp.get(
                            "group_gate_problems"),
                        group_max_ratio=cot_cmp.get("group_max_ratio"),
                        status="synthetic_verified" if cot_cmp["ok"]
                        else "mismatch",
                        failure=None if cot_cmp["ok"]
                        else "output_cotangent"))
                    model_arrays[f"cotangent/u{u}/m{m}"] = cot
                    model_arrays[f"fk_elements/u{u}/m{m}"] = ref["elements"]
                    # parameter VJP through the real network
                    params = [p for _, p in trainer._named_parameters()]
                    cot32 = (total_cot / 4.0).to(dtype=torch.float32,
                                                 device=pred.device)
                    check(f"i_vjp_{u}_{m}")
                    grads = torch.autograd.grad(
                        pred, params, grad_outputs=cot32,
                        allow_unused=True, retain_graph=False)
                    counter.add("independent_total_vjp")
                    grad_map = {name: (None if g is None else
                                       g.detach().cpu().numpy())
                                for name, g in zip(names, grads)}
                    i_micro_grads[m] = grad_map
                    unknown_none = [n for n, g in grad_map.items()
                                    if g is None]
                    if unknown_none:
                        failures.append(
                            f"i_unexpected_none_gradient:{model_type}:u{u}:m{m}"
                            f":{','.join(unknown_none[:3])}")
                    _accumulate_f32(i_accum, grad_map)
                    # micro parameter rows vs P contributions
                    for name in names:
                        p_val = p_rec["arrays"].get(
                            f"contrib/u{u}/m{m}/{name}")
                        i_val = grad_map.get(name)
                        presence = (p_rec.get("leaf_presence") or {}).get(
                            f"u{u}/m{m}/{name}")
                        if p_val is None and i_val is None:
                            status = "none"
                            cmp_ok = True
                            cmp_rec = {"problems": [], "ok": True}
                        elif p_val is None or i_val is None:
                            status = "presence_mismatch"
                            cmp_ok = False
                            cmp_rec = {"problems": ["presence"], "ok": False}
                            failures.append(
                                f"i_micro_presence:{model_type}:u{u}:m{m}:{name}")
                        else:
                            status = "synthetic_verified"
                            cmp_rec = compare_gradient(
                                p_val, i_val, tol["parameter_gradient"],
                                name=f"micro_param:{name}")
                            cmp_ok = cmp_rec["ok"]
                            if not cmp_ok:
                                failures.append(
                                    f"i_micro_param:{model_type}:u{u}:m{m}:{name}")
                        rows.append(_row(
                            f"micro_param:{model_type}:u{u}:m{m}:{name}",
                            model=model_type, update=u, micro=m,
                            parameter=name, presence=presence,
                            status=status, comparison=cmp_rec,
                            failure=None if cmp_ok else "micro_parameter"))
                    # objective rows
                    p_micro = p_rec["updates"][f"u{u}"]["micros"][m]
                    for term in TERMS:
                        if term == "L5":
                            prod = p_micro.get("L5")
                        else:
                            prod = p_micro["parts"].get(term)
                        indep = scalars.get(term)
                        if prod is None or indep is None:
                            failures.append(
                                f"i_objective_missing:{model_type}:u{u}:m{m}:{term}")
                            rows.append(_row(
                                f"objective:{model_type}:u{u}:m{m}:{term}",
                                model=model_type, update=u, micro=m,
                                term=term, status="missing",
                                failure="objective_evidence"))
                            continue
                        cmp_rec = compare_numeric(
                            np.asarray([prod]), np.asarray([float(indep)]),
                            tol["loss_scalar"],
                            name=f"objective:{term}")
                        if not cmp_rec["ok"]:
                            failures.append(
                                f"i_objective:{model_type}:u{u}:m{m}:{term}")
                            gaps += [f"i_objective_domain:{model_type}:u{u}:m{m}"
                                     for g in (vjp_facts["geodesic_domain_ok"]
                                               if term in ("geodesic_rad2", "total",
                                                           "L5") else True,)
                                     if not g]
                        rows.append(_row(
                            f"objective:{model_type}:u{u}:m{m}:{term}",
                            model=model_type, update=u, micro=m, term=term,
                            production=float(prod),
                            independent=float(indep),
                            max_abs=cmp_rec.get("max_abs"),
                            max_ratio=cmp_rec.get("max_ratio"),
                            status="synthetic_verified" if cmp_rec["ok"]
                            else "mismatch",
                            failure=None if cmp_rec["ok"] else "objective"))
                # independent clip of the I gradient
                check(f"i_clip_{u}")
                clip = _clip_independent(i_accum, float(provider.config["grad_clip"]))
                counter.add("independent_clip_formula")
                p_gnorm = float(p_rec["updates"][f"u{u}"]["gnorm"])
                p_coef = float(p_rec["updates"][f"u{u}"]["clip_coef"])
                norm_cmp = compare_numeric(
                    np.asarray([p_gnorm]), np.asarray([clip["norm"]]),
                    tol["global_norm_clip"], name="gnorm")
                coef_cmp = compare_numeric(
                    np.asarray([p_coef]), np.asarray([clip["coef"]]),
                    tol["global_norm_clip"], name="clip_coef")
                if not (norm_cmp["ok"] and coef_cmp["ok"]):
                    failures.append(f"i_global_clip:{model_type}:u{u}")
                rows.append(_row(
                    f"update_global:{model_type}:u{u}", model=model_type,
                    update=u, production_gnorm=p_gnorm,
                    independent_gnorm=clip["norm"],
                    production_coef=p_coef, independent_coef=clip["coef"],
                    production_triggered=bool(
                        p_rec["updates"][f"u{u}"]["clip_triggered"]),
                    independent_triggered=clip["triggered"],
                    norm_comparison=norm_cmp,
                    coef_comparison=coef_cmp,
                    status="synthetic_verified"
                    if norm_cmp["ok"] and coef_cmp["ok"] else "mismatch",
                    failure=None if norm_cmp["ok"] and coef_cmp["ok"]
                    else "global_clip"))
                check(f"i_update_{u}")
                # final accumulated/clipped and AdamW state rows
                for name in names:
                    p_accum = p_rec["arrays"].get(f"accum/u{u}/m3/{name}")
                    p_contributions = [p_rec["arrays"].get(
                        f"contrib/u{u}/m{m}/{name}") for m in range(MICRO_COUNT)]
                    p_sum = (None if all(x is None for x in p_contributions)
                             else np.sum([np.asarray(x, np.float64)
                                          for x in p_contributions
                                          if x is not None], axis=0))
                    acc_cmp = compare_gradient(
                        p_accum, p_sum, tol["parameter_gradient"],
                        name=f"accumulated_sum:{name}") if p_accum is not None \
                        and p_sum is not None else {"ok": p_accum is None
                                                    and p_sum is None,
                                                    "problems": []}
                    i_acc = i_accum.get(name)
                    if (p_accum is None) != (i_acc is None):
                        cross_cmp = {"ok": False, "problems": ["presence"]}
                        acc_ok = False
                    elif p_accum is None and i_acc is None:
                        cross_cmp = {"ok": True, "problems": []}
                        acc_ok = bool(acc_cmp.get("ok"))
                    else:
                        cross_cmp = compare_gradient(
                            p_accum, i_acc, tol["parameter_gradient"],
                            name=f"accumulated_cross:{name}")
                        acc_ok = bool(acc_cmp.get("ok")) and bool(cross_cmp["ok"])
                    if not acc_ok:
                        failures.append(
                            f"i_accumulated:{model_type}:u{u}:{name}")
                    rows.append(_row(
                        f"accumulated:{model_type}:u{u}:{name}",
                        model=model_type, update=u, parameter=name,
                        production=p_accum is not None,
                        independent=i_acc is not None,
                        sum_comparison=acc_cmp,
                        cross_comparison=cross_cmp,
                        status="synthetic_verified" if acc_ok else "mismatch",
                        failure=None if acc_ok else "accumulated"))
                    p_post = p_rec["arrays"].get(f"postclip/u{u}/{name}")
                    p_theta_after = p_rec["arrays"].get(
                        f"theta_after/u{u}/{name}")
                    p_pre = p_rec["arrays"].get(f"preclip/u{u}/{name}")
                    p_self = (None if p_pre is None else
                              (np.asarray(p_pre, np.float32)
                               * np.float32(p_coef)).astype(np.float32))
                    i_post = clip["postclip"].get(name)
                    clip_cmp_p = compare_gradient(
                        p_post, p_self, tol["parameter_gradient"],
                        name=f"clipped_self_p:{name}") if p_post is not None \
                        else {"ok": p_post is None and p_self is None,
                              "problems": []}
                    i_self = (None if i_acc is None else
                              (np.asarray(i_acc, np.float32)
                               * np.float32(clip["coef"])).astype(np.float32))
                    clip_cmp_i = compare_gradient(
                        i_post, i_self, tol["parameter_gradient"],
                        name=f"clipped_self_i:{name}") if i_post is not None \
                        else {"ok": i_post is None and i_self is None,
                              "problems": []}
                    if p_post is None or i_post is None:
                        clip_cross = {"ok": p_post is None and i_post is None,
                                      "problems": []}
                    else:
                        clip_cross = compare_gradient(
                            p_post, i_post, tol["parameter_gradient"],
                            name=f"clipped_cross:{name}")
                    clip_ok = (bool(clip_cmp_p.get("ok"))
                               and bool(clip_cmp_i.get("ok"))
                               and bool(clip_cross["ok"])
                               and (p_post is None) == (i_post is None))
                    if not clip_ok:
                        failures.append(
                            f"i_clipped:{model_type}:u{u}:{name}")
                    rows.append(_row(
                        f"clipped:{model_type}:u{u}:{name}",
                        model=model_type, update=u, parameter=name,
                        production=p_post is not None,
                        independent=i_post is not None,
                        self_comparison=clip_cmp_p,
                        cross_comparison=clip_cross,
                        status="synthetic_verified" if clip_ok else "mismatch",
                        failure=None if clip_ok else "clipped"))
                    # AdamW row
                    if p_post is None or i_post is None:
                        rows.append(_row(
                            f"adamw:{model_type}:u{u}:{name}",
                            model=model_type, update=u, parameter=name,
                            status="none_parameter",
                            classification="none_parameter",
                            same_input_formula_passed=True,
                            raw_cross_path_passed=True,
                            delta_observable_p=False,
                            delta_observable_i=False,
                            delta_ref_p=0.0, delta_ref_i=0.0,
                            delta_measured_p=0.0, delta_measured_i=0.0,
                            bound_delta_p=0.0, bound_delta_i=0.0,
                            failure=None, gaps=[]))
                        continue
                    p_pre_state = _state_values(p_rec, "before", u, name)
                    p_post_state = _state_values(p_rec, "after", u, name)
                    step_arr = p_post_state.get("step_array")
                    if step_arr is None:
                        failures.append(
                            f"i_step_tensor_missing:{model_type}:u{u}:{name}")
                    else:
                        step_np = np.asarray(step_arr)
                        if (step_np.ndim != 0
                                or step_np.dtype != np.float32
                                or not np.isfinite(step_np).all()
                                or float(step_np) != float(u)):
                            failures.append(
                                f"i_step_tensor:{model_type}:u{u}:{name}")
                    i_eff = adamw_effective_fp32(
                        pre_theta[name],
                        None if m_none[name] else p_pre_state["m"],
                        None if v_none[name] else p_pre_state["v"],
                        i_post, u,
                        m_none=m_none[name], v_none=v_none[name],
                        host=host_cache[u])
                    model_arrays[f"i_theta/u{u}/{name}"] = i_eff["theta"]
                    model_arrays[f"i_m/u{u}/{name}"] = i_eff["m"]
                    model_arrays[f"i_v/u{u}/{name}"] = i_eff["v"]
                    row = _adamw_param_row(
                        f"adamw:{model_type}:u{u}:{name}",
                        name=name, model=model_type, u=u,
                        p_pre={"theta": pre_theta[name],
                               "m": p_pre_state["m"],
                               "v": p_pre_state["v"],
                               "m_none": m_none[name],
                               "v_none": v_none[name]},
                        p_post={"theta": p_theta_after,
                                "m": p_post_state["m"],
                                "v": p_post_state["v"],
                                "step": p_post_state["step"]},
                        i_post={"theta": i_eff["theta"], "m": i_eff["m"],
                                "v": i_eff["v"]},
                        p_g=p_post, i_g=i_post,
                        p_step=u, i_step=u, host=host_cache[u],
                        arrays=model_arrays, tolerance=tol)
                    rows.append(row)
                    if row.get("failure"):
                        failures.append(
                            f"i_adamw:{model_type}:u{u}:{name}:{row['failure']}")
                    if row.get("gaps"):
                        gaps += [f"i_adamw_gap:{model_type}:u{u}:{name}:{g}"
                                 for g in row["gaps"]]
                # The explicit reference update is the AdamW computation
                # above; count it only after it actually executed.
                counter.add("explicit_reference_updates")
                counter.add("diagnostic_parameter_update_events")
                counter.add("optimizer_formula_checks")
                # step continuity row (once per model, at update 2)
                p_next = {n: p_rec["arrays"].get(f"theta_before/u2/{n}")
                          for n in names}
                p_after1 = {n: p_rec["arrays"].get(f"theta_after/u1/{n}")
                            for n in names}
                cont_problems = _continuity_problems(p_after1, p_next, "theta")
                state_after1 = {n: _state_values(p_rec, "after", 1, n)
                                for n in names}
                state_before2 = {n: _state_values(p_rec, "before", 2, n)
                                 for n in names}
                cont_problems += _continuity_problems(
                    state_after1, state_before2, "state")
                if u == 2:
                    rows.append(_row(
                        f"step_continuity:{model_type}:1->2",
                        model=model_type, update_from=1, update_to=2,
                        theta_restore_matches=all(
                            np.array_equal(pre_theta[name],
                                           p_rec["arrays"][
                                               f"theta_before/u2/{name}"])
                            for name in names),
                        continuity_problems=cont_problems,
                        status="synthetic_verified" if not cont_problems
                        else "mismatch",
                        failure=None if not cont_problems else "continuity"))
                    if cont_problems:
                        failures += [f"i_continuity:{model_type}:{p}"
                                     for p in cont_problems]
                report["updates"][f"u{u}"] = {
                    "objective": {str(m): cotangent_facts[m]["scalars"]
                                  for m in cotangent_facts},
                    "gnorm": p_gnorm, "independent_gnorm": clip["norm"],
                    "coef": p_coef, "independent_coef": clip["coef"],
                }
        finally:
            for handle in handles:
                handle.remove()
        npz_path = base / f"i_{model_type}.npz"
        arrays_sha = _save_npz(npz_path, model_arrays)
        report["arrays_path"] = path_for_record(npz_path)
        report["arrays_sha256"] = arrays_sha
        results[model_type] = report
    catalog = row_catalog(roster)
    evidence_failures = _i_evidence_failures(failures) + list(ref_ctx.problems)
    row_problems = _stage_row_problems(rows, catalog, "I")
    complete_evidence = (
        not evidence_failures and not row_problems
        and all(results[model].get("arrays_path") for model in MODELS))
    report = {
        "schema": "w1_real_training_update_i_stage_v1",
        "mode": "synthetic",
        "stage": stage, "attempt": attempt,
        "models": results,
        "rows": rows,
        "failures": failures,
        "gaps": gaps,
        "evidence_failures": evidence_failures,
        "evidence_problems": row_problems,
        "complete_evidence": bool(complete_evidence),
        "numeric_passed": not failures,
        # Numeric mismatches are complete evidence and must reach F, which
        # replays and preserves them.  Gaps do not block F either.
        "passed": not failures,
    }
    write_json(base / "i_rows.json", rows)
    write_json(base / "i_report.json", report)
    return {"passed": not failures, "failures": failures,
            "gaps": gaps, "rows": rows,
            "complete_evidence": bool(complete_evidence),
            "evidence_failures": evidence_failures,
            "evidence_problems": row_problems,
            "numeric_passed": not failures,
            "paths": {model: results[model]["arrays_path"]
                      for model in MODELS},
            "report": path_for_record(base / "i_report.json"),
            "arrays_sha256": {model: results[model]["arrays_sha256"]
                              for model in MODELS}}

# ---------------------------------------------------------------------------
# F (fresh subprocess replay and independent verification)
# ---------------------------------------------------------------------------


def _npz_compare_problems(saved_dir: Path, replay_dir: Path,
                          models: Sequence[str],
                          *, prefix: str) -> list[str]:
    problems: list[str] = []
    for model in models:
        a_path = Path(saved_dir) / f"{prefix}_{model}.npz"
        b_path = Path(replay_dir) / f"{prefix}_{model}.npz"
        if not a_path.is_file() or not b_path.is_file():
            problems.append(f"npz_missing:{prefix}:{model}")
            continue
        a = _load_npz(a_path)
        b = _load_npz(b_path)
        if set(a) != set(b):
            problems.append(f"npz_keys:{prefix}:{model}")
            continue
        for key in sorted(a):
            x, y = a[key], b[key]
            if x.dtype != y.dtype or x.shape != y.shape:
                problems.append(f"npz_shape:{prefix}:{model}:{key}")
                continue
            if x.dtype.kind in "fc":
                if not np.array_equal(x, y):
                    problems.append(f"npz_value:{prefix}:{model}:{key}")
            elif x.tobytes() != y.tobytes():
                problems.append(f"npz_value:{prefix}:{model}:{key}")
    return problems


def _row_key_counts(rows: Sequence[Mapping[str, Any]]) -> Counter:
    return Counter(str(r.get("key")) for r in rows)


def _row_compare_problems(saved: Sequence[Mapping[str, Any]],
                          fresh: Sequence[Mapping[str, Any]],
                          expected: Sequence[str]) -> list[str]:
    """Compare saved and fresh rows as multisets, never via a key->row dict."""
    problems: list[str] = []
    want = Counter(expected)
    got = _row_key_counts(fresh)
    if got != want:
        missing = want - got
        extra = got - want
        if missing:
            problems.append(f"rows_missing:{len(missing)}")
        if extra:
            problems.append(f"rows_extra:{len(extra)}")
        if any(v > 1 for v in got.values()):
            problems.append("rows_duplicate")
    saved_keys = _row_key_counts(saved)
    if saved_keys != got:
        if saved_keys - got:
            problems.append(f"saved_rows_missing:{len(saved_keys - got)}")
        if got - saved_keys:
            problems.append(f"saved_rows_extra:{len(got - saved_keys)}")
        if any(v > 1 for v in saved_keys.values()):
            problems.append("saved_rows_duplicate")
    saved_by: dict[str, list[str]] = {}
    fresh_by: dict[str, list[str]] = {}
    for row in saved:
        saved_by.setdefault(str(row.get("key")), []).append(
            _canonical_sha256(row))
    for row in fresh:
        fresh_by.setdefault(str(row.get("key")), []).append(
            _canonical_sha256(row))
    for key in sorted(set(saved_by) & set(fresh_by)):
        if Counter(saved_by[key]) != Counter(fresh_by[key]):
            problems.append(f"row_drift:{key}")
    return problems


ROW_EVIDENCE_STATUSES = {"synthetic_verified", "none", "none_parameter",
                         "mismatch", "presence_mismatch"}


def _row_evidence_problems(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    for row in rows:
        if row.get("status") not in ROW_EVIDENCE_STATUSES:
            problems.append(f"row_evidence:{row.get('key')}:"
                            f"{row.get('status')}")
    return problems


def _report_binding_problems(saved_dir: Path, replay_dir: Path, *,
                             stage: str) -> list[str]:
    """Bind saved reports, RNG facts and rows to the fresh replay.

    F must not trust the saved summary: a doctored total, an all-zero RNG
    digest or a duplicated row is only visible by reconciling the report and
    both row multisets with the independently replayed run.
    """
    prefix = "p" if stage == "P" else "i"
    problems: list[str] = []
    try:
        saved = read_json(Path(saved_dir) / f"{prefix}_report.json")
        fresh = read_json(Path(replay_dir) / f"{prefix}_report.json")
    except (OSError, ValueError, TypeError) as exc:
        return [f"report_unreadable:{stage}:{type(exc).__name__}"]
    if not isinstance(saved, Mapping) or not isinstance(fresh, Mapping):
        return [f"report_not_mapping:{stage}"]
    if saved.get("schema") != fresh.get("schema"):
        problems.append(f"report_schema:{stage}")
    if stage == "P":
        keys = ("none_parameters", "leaf_presence", "state_presence",
                "step_facts", "roster_problems", "optimizer_problems",
                "failures", "gaps")
        micro_keys = ("update", "micro", "windows", "loss", "parts",
                      "prediction_hook_fires", "rng_before", "rng_after",
                      "prediction_sha256", "prediction_grad_sha256",
                      "prediction_grad_norm", "memory_sha256", "L5",
                      "x_t_max_ratio", "alpha_t_hash")
    else:
        keys = ("failures", "gaps", "evidence_failures",
                "evidence_problems", "complete_evidence", "numeric_passed")
        micro_keys = ()
    for model in MODELS:
        a = (saved.get("models") or {}).get(model)
        b = (fresh.get("models") or {}).get(model)
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            problems.append(f"report_model_missing:{stage}:{model}")
            continue
        for key in keys:
            if _canonical_sha256(a.get(key)) != _canonical_sha256(b.get(key)):
                problems.append(f"report_drift:{stage}:{model}:{key}")
        if a.get("arrays_sha256") != b.get("arrays_sha256"):
            problems.append(f"report_arrays_sha:{stage}:{model}")
        npz_path = Path(saved_dir) / f"{prefix}_{model}.npz"
        if not npz_path.is_file():
            problems.append(f"report_npz_missing:{stage}:{model}")
        elif a.get("arrays_sha256") != sha256_file(npz_path):
            problems.append(f"report_npz_sha:{stage}:{model}")
        au = a.get("updates") or {}
        bu = b.get("updates") or {}
        if set(au) != set(bu):
            problems.append(f"report_updates:{stage}:{model}")
            continue
        for u in UPDATES:
            ua = au.get(f"u{u}") or {}
            ub = bu.get(f"u{u}") or {}
            for key in ("gnorm", "clip_coef", "clip_triggered",
                        "independent_gnorm", "independent_coef",
                        "objective", "coef", "continuity_problems"):
                if key in ua or key in ub:
                    if _canonical_sha256(ua.get(key)) != _canonical_sha256(
                            ub.get(key)):
                        problems.append(
                            f"report_update_drift:{stage}:{model}:u{u}:{key}")
            if not micro_keys:
                continue
            ma = ua.get("micros") or []
            mb = ub.get("micros") or []
            if len(ma) != MICRO_COUNT or len(mb) != MICRO_COUNT:
                problems.append(f"report_micros:{stage}:{model}:u{u}")
                continue
            for m, (x, y) in enumerate(zip(ma, mb)):
                for key in micro_keys:
                    if key in x or key in y:
                        if _canonical_sha256(x.get(key)) != _canonical_sha256(
                                y.get(key)):
                            problems.append(
                                f"report_micro_drift:{stage}:{model}:u{u}"
                                f":m{m}:{key}")
    rows_path = Path(saved_dir) / f"{prefix}_rows.json"
    try:
        rows = read_json(rows_path)
    except (OSError, ValueError, TypeError):
        problems.append(f"rows_unreadable:{stage}")
    else:
        if _canonical_sha256(rows) != _canonical_sha256(saved.get("rows")):
            problems.append(f"report_rows_file_disagree:{stage}")
    return problems


def _worker_main(spec_path: str, out_dir: str) -> int:
    """Spawned F worker: fresh models, disk re-read, full re-verification.

    The worker inherits the parent's absolute deadline and re-checks the
    running parent registration before every read, factory, micro, backward
    and update.  A result snapshot is written in ``finally`` so the parent can
    settle only calls that were actually made; a killed worker leaves no
    snapshot and therefore no invented zero.
    """
    spec = read_json(Path(spec_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter = CallCounter(spec["remaining_caps"])
    deadline = float(spec["deadline_perf_counter"])
    reserve = float(spec.get("reserve_s", 0.0))
    root = Path(spec["run_root"])
    attempt = str(spec["attempt"])
    parent_record = {"registration_sha256": spec.get("registration_sha256"),
                     "reserve_s": reserve}
    problems: list[str] = []
    failures: list[str] = []
    gaps: list[str] = []
    runtime: dict = {}
    provider = None
    p_replay = None
    i_replay = None
    saved_rows: list = []
    saved_p_rows: list = []
    fresh_rows: list = []
    schedule_ok = False
    p_problems: list[str] = []
    i_problems: list[str] = []
    row_problems: list[str] = []
    report_problems: list[str] = []
    call_problems: list[str] = []
    aborted = False

    def guard(unit: str) -> None:
        # ``deadline`` is already the work deadline (the seal reserve was
        # subtracted once when the stage started), so do not subtract it again.
        _ensure_deadline(deadline, 0.0, unit)
        admission = unit_admission(root, "F", attempt, parent_record,
                                   deadline=deadline)
        if admission:
            raise TimeoutError(f"unit_admission:{unit}:{','.join(admission)}")

    try:
        guard("worker_start")
        runtime = configure_production_runtime()
        if runtime["problems"]:
            problems += [f"worker_runtime:{p}" for p in runtime["problems"]]
        guard("fixture_load")
        provider = SyntheticTrainingProvider(spec["config"],
                                             device=spec.get("device", "cuda"))
        provider.fixture_dir = Path(spec["fixture_dir"])
        provider.load_fixture()
        roster = read_json(provider.fixture_dir / "roster.json")
        # Rebuild the R0 schedule once (counted) and compare with the fixture.
        guard("schedule_rebuild")
        schedule = draw_noise_path(NOISE_SEED)
        counter.add("cpu_draw_units", DRAW_UNITS_PER_PATH)
        fixture_noise = provider.noise
        schedule_ok = True
        for a, b in zip(schedule, fixture_noise):
            if (not torch.equal(a["t"], b["t"])
                    or not torch.equal(a["epsilon"], b["epsilon"])):
                schedule_ok = False
        if not schedule_ok:
            problems.append("schedule_rebuild_mismatch")
        p_dir = out / "production_replay"
        i_dir = out / "independent_replay"
        guard("p_replay_start")
        p_replay = run_p_stage(
            out, "P", "F-P", _now(),
            {"remaining_s": max(0.0, deadline - _now() + reserve),
             "reserve_s": reserve}, provider, counter, roster,
            out_dir=p_dir, guard=guard)
        guard("p_replay_compare")
        p_problems = _npz_compare_problems(Path(spec["p_dir"]), p_dir, MODELS,
                                           prefix="p")
        problems += p_problems
        guard("i_replay_start")
        i_replay = run_i_stage(
            out, "I", "F-I", _now(),
            {"remaining_s": max(0.0, deadline - _now() + reserve),
             "reserve_s": reserve}, provider, counter, roster,
            p_dir, tolerance=spec["tolerance"], out_dir=i_dir, guard=guard)
        guard("i_replay_compare")
        i_problems = _npz_compare_problems(Path(spec["i_dir"]), i_dir, MODELS,
                                           prefix="i")
        problems += i_problems
        guard("row_binding")
        try:
            saved_rows = read_json(Path(spec["i_dir"]) / "i_rows.json")
            saved_p_rows = read_json(Path(spec["p_dir"]) / "p_rows.json")
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"saved_rows_unreadable:{type(exc).__name__}")
            saved_rows, saved_p_rows = [], []
        fresh_rows = list(i_replay["rows"])
        catalog = row_catalog(roster)
        all_fresh = list(p_replay["rows"]) + list(fresh_rows)
        row_problems = _row_compare_problems(
            list(saved_rows) + list(saved_p_rows), all_fresh, catalog["rows"])
        row_problems += _row_evidence_problems(all_fresh)
        expected_p = _expected_stage_rows(catalog, "P")
        expected_i = _expected_stage_rows(catalog, "I")
        if _row_key_counts(saved_p_rows) != Counter(expected_p):
            row_problems.append("saved_p_rows_scope")
        if _row_key_counts(saved_rows) != Counter(expected_i):
            row_problems.append("saved_i_rows_scope")
        problems += row_problems
        report_problems = (
            _report_binding_problems(Path(spec["p_dir"]), p_dir, stage="P")
            + _report_binding_problems(Path(spec["i_dir"]), i_dir, stage="I"))
        problems += report_problems
        # Replay failures are measured numeric evidence, not evidence loss:
        # they are preserved in ``failures`` so F can conclude a mismatch.
        failures += list(p_replay["failures"]) + list(i_replay["failures"])
        gaps += list(p_replay["gaps"]) + list(i_replay["gaps"])
        if not p_replay.get("complete_evidence", False):
            problems.append("p_replay_incomplete_evidence")
        if not i_replay.get("complete_evidence", False):
            problems.append("i_replay_incomplete_evidence")
        expected_f = dict(STAGE_NOMINAL["F"])
        expected_f["scene_setup_mj_forward"] = 0
        counts = counter.snapshot()["counts"]
        call_problems = [
            f"f_call_matrix:{kind}:{counts[kind]}!={expected_f[kind]}"
            for kind in CALL_KINDS if counts[kind] != expected_f[kind]]
        problems += call_problems
    except TimeoutError as exc:
        aborted = True
        problems.append(f"f_worker_aborted:{exc}")
    except Exception as exc:  # noqa: BLE001 -- a broken worker is evidence
        aborted = True
        problems.append(f"f_worker_exception:{type(exc).__name__}:{exc}")
    finally:
        snapshot = counter.snapshot()
        result = {
            "schema": "w1_real_training_update_f_worker_v1",
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "python": sys.version.split()[0],
            "runtime": runtime,
            "fixture_dir": (path_for_record(provider.fixture_dir)
                            if provider is not None else None),
            "p_replay": p_replay,
            "i_replay": None if i_replay is None else {
                "complete_evidence": i_replay.get("complete_evidence"),
                "numeric_passed": i_replay.get("numeric_passed"),
                "rows": len(i_replay.get("rows") or []),
                "report": i_replay.get("report"),
            },
            "i_replay_rows": len(fresh_rows),
            "schedule_rebuild_ok": schedule_ok,
            "npz_problems": p_problems + i_problems,
            "row_problems": row_problems,
            "report_problems": report_problems,
            "call_matrix_problems": call_problems,
            "problems": problems,
            "aborted": aborted,
            "counts": snapshot,
            "complete_evidence": not problems,
            "failures": failures,
            "gaps": gaps,
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        try:
            write_json(out / "worker_result.json", result)
        except (OSError, ValueError, TypeError):
            return 3
    return 0 if not problems else 2


def run_f_stage(root: Path, stage: str, attempt: str, t0: float,
                budget: Mapping[str, Any], provider: SyntheticTrainingProvider,
                counter: CallCounter, roster: Mapping[str, Any],
                p_dir: Path, i_dir: Path, *, tolerance: Mapping[str, Any],
                out_dir: Path | None = None,
                project_root: Path = PROJECT_ROOT) -> dict:
    base = Path(out_dir or (stage_dir(root, stage) / attempt))
    base.mkdir(parents=True, exist_ok=True)
    deadline = t0 + float(budget["remaining_s"]) - float(budget["reserve_s"])
    base_counts = counter.snapshot()["counts"]
    remaining = {kind: int(CALL_CAPS[kind]) - int(base_counts[kind])
                 for kind in CALL_KINDS}
    spec = {
        "fixture_dir": path_for_record(provider.fixture_dir),
        "p_dir": path_for_record(p_dir),
        "i_dir": path_for_record(i_dir),
        "config": provider.config,
        "device": provider.device,
        "tolerance": dict(tolerance),
        "remaining_caps": remaining,
        "deadline_perf_counter": deadline,
        "reserve_s": float(budget["reserve_s"]),
        "run_root": path_for_record(Path(root)),
        "stage": stage,
        "attempt": attempt,
        "registration_sha256": None,
    }
    try:
        rows = [r for r in load_index(root)
                if r.get("stage") == stage and r.get("attempt") == attempt]
    except (OSError, ValueError, TypeError):
        rows = []
    if len(rows) == 1:
        spec["registration_sha256"] = rows[0].get(
            "execution_registration_sha256")
    spec_path = base / "worker_spec.json"
    write_json(spec_path, spec)
    worker_dir = base / "worker"
    entry = (
        "import sys;"
        f"sys.path.insert(0, {str(Path(project_root) / 'src')!r});"
        "from ireset.eval.w1_real_training_update_audit import _worker_main;"
        "_worker_main(sys.argv[1], sys.argv[2])")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(project_root) / "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    problems: list[str] = []
    result: dict | None = None
    returncode = -1
    stdout = stderr = ""
    wall = 0.0
    if time.perf_counter() >= deadline:
        problems.append("f_stage_deadline_before_worker")
        result = {"problems": list(problems), "counts": None,
                  "failures": [], "gaps": [], "aborted": True}
    else:
        timeout = max(0.1, min(float(budget["remaining_s"]),
                               deadline - time.perf_counter()))
        started = _now()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", entry, str(spec_path), str(worker_dir)],
                cwd=str(project_root), env=env, capture_output=True, text=True,
                timeout=timeout)
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = -9
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        wall = _now() - started
        result_path = worker_dir / "worker_result.json"
        result = read_json(result_path) if result_path.is_file() else None
    process = {
        "pid": None if result is None else result.get("pid"),
        "ppid": None if result is None else result.get("ppid"),
        "parent_pid": os.getpid(),
        "python": sys.executable,
        "exit_code": returncode,
        "wall_s": wall,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "exit_reason": ("ok" if returncode == 0 else
                        "timeout" if returncode == -9 else
                        "not_spawned" if returncode == -1 else "nonzero_exit"),
    }
    if result is None:
        problems.append("f_worker_result_missing")
        result = {"problems": [], "counts": None, "failures": [], "gaps": [],
                  "aborted": True}
    problems += list(result.get("problems") or [])
    if returncode != 0 and "f_stage_deadline_before_worker" not in problems:
        problems.append(f"f_worker_exit:{returncode}")
    if (process["pid"] is not None
            and process["pid"] == process["parent_pid"]):
        problems.append("f_worker_not_a_fresh_process")
    if process["ppid"] is not None and process["ppid"] != os.getpid():
        problems.append("f_worker_ppid")
    counts = (result.get("counts") or {}).get("counts") \
        if isinstance(result.get("counts"), Mapping) else None
    merged = True
    if not isinstance(counts, Mapping):
        problems.append("f_worker_counts_missing")
        merged = False
    else:
        try:
            for kind in CALL_KINDS:
                value = counts.get(kind)
                if not _json_int(value) or int(value) < 0:
                    raise ValueError(kind)
                counter.add(kind, int(value))
        except (KeyError, ValueError, UpdateCallBudgetExceeded) as exc:
            problems.append(f"f_worker_counts:{type(exc).__name__}:{exc}")
            merged = False
    expected_f = dict(STAGE_NOMINAL["F"])
    expected_f["scene_setup_mj_forward"] = 0
    if merged:
        problems += [f"f_call_matrix:{kind}:{int(counts.get(kind, -1))}"
                     f"!={expected_f[kind]}" for kind in CALL_KINDS
                     if int(counts.get(kind, -1)) != expected_f[kind]]
    failures = list(result.get("failures") or [])
    gaps = [g for g in (result.get("gaps") or []) if isinstance(g, str)]
    verification_problems = list(result.get("problems") or []) + problems
    complete = not verification_problems
    covered = complete and not failures and not gaps
    conclusion = ("real_training_update_covered_contract_passed" if covered
                  else "audit_incomplete" if gaps or not complete
                  else "real_training_update_mismatch")
    final = {
        "schema": "w1_real_training_update_final_v1",
        "mode": "synthetic",
        "cohort_id": COHORT_ID,
        "attempt": attempt,
        "conclusion": conclusion,
        "complete_evidence": complete,
        "covered_contract_passed": covered,
        "verification_problems": verification_problems,
        "failures": failures,
        "gaps": gaps,
        "gap_kinds": dict(Counter(g.split(":")[-1] for g in gaps)),
        "worker": process,
        "worker_result": result,
        "worker_snapshot_ok": bool(merged),
        "deadline_perf_counter": deadline,
        "call_delta": dict(counts) if merged else None,
        "call_balance": counter.snapshot(),
        "training_updates": int(counter.counts["native_adamw_steps"]
                                + counter.counts["explicit_reference_updates"]),
        "zero_counters": {k: 0 for k in ZERO_COUNTERS},
        "coverage": {
            "objective_coverage": "partial",
            "full_objective_contract_passed": None,
            "attribution": "attribution_insufficient",
            "factor_selected": False,
            "auto_execute": False,
            "final_interaction_method": "not_ready",
            "production_integration": False,
        },
        "boundaries": dict(BOUNDARIES),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    final_path = base / "training_update_final.json"
    write_json(final_path, final)
    return {
        "conclusion": conclusion,
        "complete_evidence": complete,
        "covered_contract_passed": covered,
        "verification_problems": verification_problems,
        "failures": failures,
        "gaps": gaps,
        "worker": process,
        "worker_result": result,
        "final": final,
        "report": path_for_record(final_path),
        "summary": final,
        "call_delta": dict(counts) if merged else None,
        "units": [],
    }

# ---------------------------------------------------------------------------
# cohort/attempt accounting (semantics carried from the audited ledger)
# ---------------------------------------------------------------------------


def stage_dir(root: Path, stage: str) -> Path:
    return Path(root) / STAGE_DIRS[stage]


def _cohort_path(root: Path) -> Path:
    return Path(root) / "cohort.json"


def _index_path(root: Path) -> Path:
    return Path(root) / "attempts_index.json"


def _ledger_path(root: Path) -> Path:
    return Path(root) / "costs.json"


def _load_rows(path: Path, key: str) -> list[dict]:
    if not Path(path).is_file():
        return []
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        raise ValueError(f"{path}: missing list {key!r}")
    return [r for r in data[key] if isinstance(r, dict)]


def load_index(root: Path) -> list[dict]:
    return _load_rows(_index_path(root), "attempts")


def load_ledger(root: Path) -> list[dict]:
    return _load_rows(_ledger_path(root), "invocations")


def save_index(root: Path, rows: list[dict]) -> None:
    write_json(_index_path(root), {"attempts": rows})


def save_ledger(root: Path, rows: list[dict]) -> None:
    write_json(_ledger_path(root), {"invocations": rows})


def _replace_row(rows: list[dict], stage: str, attempt: str,
                 entry: dict) -> tuple[list[dict], bool]:
    hits = [i for i, row in enumerate(rows)
            if row.get("stage") == stage and row.get("attempt") == attempt]
    if len(hits) > 1:
        raise ValueError(f"duplicate {stage}/{attempt} rows")
    if hits:
        rows[hits[0]] = {**rows[hits[0]], **entry,
                         "stage": stage, "attempt": attempt}
        return rows, True
    rows.append({"stage": stage, "attempt": attempt, **entry})
    return rows, False


def upsert_index(root: Path, stage: str, attempt: str, entry: dict) -> None:
    rows = load_index(root)
    rows, _ = _replace_row(rows, stage, attempt, entry)
    save_index(root, rows)


def upsert_ledger(root: Path, stage: str, attempt: str, entry: dict) -> None:
    rows = load_ledger(root)
    rows, _ = _replace_row(rows, stage, attempt, entry)
    save_ledger(root, rows)


def upsert_cost(root: Path, stage: str, attempt: str, measured_wall_s: float,
                extra: dict | None = None, status: str = "running") -> None:
    measured = float(measured_wall_s)
    if not _nonneg(measured):
        raise ValueError("measured wall time must be finite and non-negative")
    upsert_ledger(root, stage, attempt, {
        "status": status,
        "complete": bool((extra or {}).get("complete", False)),
        "measured_wall_s": measured,
        "wall_s": round(measured, 1),
        "accounting_schema": ACCOUNTING_SCHEMA,
        "cohort_id": COHORT_ID,
        **(extra or {}),
    })


def _disk_attempts(root: Path, stage: str) -> set[str]:
    base = stage_dir(root, stage)
    if not base.is_dir():
        return set()
    return {p.name for p in base.iterdir()
            if p.is_dir() and re.fullmatch(r"attempt_\d+", p.name)}


def attempt_identity() -> dict:
    return {
        "execution_id": EXECUTION_ID,
        "cohort_id": COHORT_ID,
        "accounting_schema": ACCOUNTING_SCHEMA,
        "audit_schema": AUDIT_SCHEMA,
    }


def write_cohort(root: Path, *, mode: str,
                 authorization_sha: str | None = None) -> None:
    path = _cohort_path(root)
    existing = read_json(path) if path.is_file() else {}
    if existing and existing.get("cohort_id") != COHORT_ID:
        raise ValueError("cohort.json belongs to another cohort")
    record = {
        **attempt_identity(),
        "mode": mode,
        "authorization_sha256": authorization_sha,
        "stage_budgets": {k: dict(v) for k, v in STAGE_BUDGETS.items()},
        "total_budget_s": TOTAL_BUDGET_S,
        "call_caps": dict(CALL_CAPS),
    }
    if existing:
        record = {**existing, **record}
    write_json(path, record)


def allocate_attempt(root: Path, stage: str) -> str:
    seen = (_disk_attempts(root, stage)
            | {r.get("attempt") for r in load_index(root)
               if r.get("stage") == stage}
            | {r.get("attempt") for r in load_ledger(root)
               if r.get("stage") == stage})
    nums = [int(name.split("_")[1]) for name in seen
            if isinstance(name, str) and re.fullmatch(r"attempt_\d+", name)]
    return f"attempt_{max(nums, default=0) + 1:03d}"


def _row_identity_problems(row: Mapping[str, Any], *, kind: str) -> list[str]:
    problems: list[str] = []
    if row.get("accounting_schema") != ACCOUNTING_SCHEMA:
        problems.append(f"{kind}_schema_missing")
    if row.get("cohort_id") != COHORT_ID:
        problems.append(f"{kind}_cohort_missing")
    if not isinstance(row.get("complete"), bool):
        problems.append(f"{kind}_completion_invalid")
    if row.get("status") not in (
            "running", "publishing", "done", "crashed", "timeout", "incomplete"):
        problems.append(f"{kind}_status_invalid")
    return problems


def _call_balance_problems(index_row: Mapping[str, Any],
                           ledger_row: Mapping[str, Any]) -> list[str]:
    a = index_row.get("call_balance")
    b = ledger_row.get("call_balance")
    if a is None and b is None:
        return []
    if a is None or b is None:
        return ["call_balance_single_sided"]
    ca = _call_counts_from_balance(a)
    cb = _call_counts_from_balance(b)
    if ca is None or cb is None or ca != cb:
        return ["call_balance_disagreement"]
    return []


def _reservation_problems(root: Path) -> list[str]:
    problems: list[str] = []
    for stage in STAGE_BUDGETS:
        base = stage_dir(root, stage)
        if not base.is_dir():
            continue
        try:
            dirs = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            problems.append("reservation_dir_unreadable")
            continue
        for directory in dirs:
            path = directory / "call_reservations.json"
            if not path.is_file():
                continue
            try:
                data = read_json(path)
            except (OSError, ValueError, TypeError):
                problems.append(
                    f"reservation_unreadable:{stage}:{directory.name}")
                continue
            if not isinstance(data, Mapping) or data.get("consumed") is None:
                problems.append(
                    f"reservation_unsettled:{stage}:{directory.name}")
    return problems


def stage_history(root: Path, stage: str,
                  *, allow_current: tuple[str, str] | None = None) -> float | None:
    try:
        index = [r for r in load_index(root) if r.get("stage") == stage]
        ledger = [r for r in load_ledger(root) if r.get("stage") == stage]
        disk = _disk_attempts(root, stage)
        index_by = {r.get("attempt"): r for r in index}
        ledger_by = {r.get("attempt"): r for r in ledger}
        if (len(index_by) != len(index) or len(ledger_by) != len(ledger)
                or None in index_by or None in ledger_by):
            return None
        if set(disk) != set(index_by) or set(disk) != set(ledger_by):
            return None
        if _reservation_problems(root):
            return None
        total = 0.0
        for attempt, row in index_by.items():
            if allow_current == (stage, attempt):
                continue
            e = ledger_by[attempt]
            problems = _row_identity_problems(row, kind="index")
            problems += _row_identity_problems(e, kind="ledger")
            problems += _call_balance_problems(row, e)
            if problems:
                return None
            if row.get("status") != e.get("status"):
                return None
            if row.get("complete") is not e.get("complete"):
                return None
            status = row.get("status")
            if status not in ("done", "crashed", "timeout", "incomplete"):
                return None
            if status != "done" and row.get("complete") is not False:
                return None
            measured = e.get("measured_wall_s")
            rounded = e.get("wall_s")
            index_measured = row.get("measured_wall_s")
            if not (_nonneg(measured) and _nonneg(rounded)
                    and _nonneg(index_measured)):
                return None
            if float(index_measured) != float(measured):
                return None
            if float(rounded) != round(float(measured), 1):
                return None
            total += float(measured)
        return total
    except (OSError, ValueError, TypeError, KeyError):
        return None


def total_history(root: Path, *,
                  allow_current: tuple[str, str] | None = None) -> float | None:
    total = 0.0
    for stage in STAGE_BUDGETS:
        value = stage_history(root, stage, allow_current=allow_current)
        if value is None:
            return None
        total += value
    return total


def _cohort_roots(fixed_root: Path) -> list[Path]:
    base = Path(fixed_root)
    if not base.is_dir():
        return []
    return [p for p in sorted(base.iterdir()) if p.is_dir()]


def _attribution(marker: Mapping[str, Any], index, ledger,
                 broken: str | None) -> tuple[str, str | None]:
    matching_marker = marker.get("cohort_id") == COHORT_ID
    matching_rows = any(
        r.get("cohort_id") == COHORT_ID
        and r.get("accounting_schema") == ACCOUNTING_SCHEMA
        for r in (*(index or []), *(ledger or [])))
    if not (matching_marker or matching_rows):
        if broken and marker:
            return "unknown", f"unreadable_{broken}"
        return ("legacy" if not marker else "foreign"), None
    if broken:
        return "unknown", f"unreadable_{broken}"
    if marker and marker.get("cohort_id") not in (None, COHORT_ID):
        return "unknown", "foreign_cohort"
    if marker and marker.get("execution_id") not in (None, EXECUTION_ID):
        return "unknown", "foreign_execution"
    return "member", None


def cohort_history(fixed_root: Path, *, exclude_root: Path | None = None,
                   allow_current: tuple[str, str] | None = None) -> dict:
    runs: list[dict] = []
    by_stage = {stage: 0.0 for stage in STAGE_BUDGETS}
    total = 0.0
    for sibling in _cohort_roots(fixed_root):
        if exclude_root is not None and sibling.resolve() == Path(exclude_root).resolve():
            continue
        marker: dict = {}
        marker_path = sibling / "cohort.json"
        if marker_path.is_file():
            try:
                marker = read_json(marker_path)
            except (OSError, ValueError):
                return {"known": False, "reason": f"unreadable_cohort:{sibling.name}",
                        "runs": runs}
            if not isinstance(marker, dict):
                return {"known": False, "reason": f"unreadable_cohort:{sibling.name}",
                        "runs": runs}
        index, bi = _read_registration(sibling, "index")
        ledger, bl = _read_registration(sibling, "ledger")
        kind, reason = _attribution(marker, index, ledger, bi or bl)
        if kind in ("foreign", "legacy"):
            continue
        if kind == "unknown":
            return {"known": False, "reason": f"{reason}:{sibling.name}",
                    "runs": runs}
        stage_costs = {}
        for stage in STAGE_BUDGETS:
            value = stage_history(
                sibling, stage,
                allow_current=(allow_current
                               if sibling.resolve()
                               == Path(exclude_root or "").resolve() else None))
            if value is None:
                return {"known": False,
                        "reason": f"unknown_history:{sibling.name}:{stage}",
                        "runs": runs}
            stage_costs[stage] = value
            by_stage[stage] += value
            total += value
        runs.append({"run": path_for_record(sibling),
                     "stage_cost_s": stage_costs})
    return {"known": True, "runs": runs, "stage_cost_s": by_stage,
            "total_s": total}


def _read_registration(root: Path, which: str) -> tuple[list[dict] | None, str | None]:
    try:
        rows = load_index(root) if which == "index" else load_ledger(root)
    except (OSError, ValueError, TypeError):
        return None, which
    return rows, None


def cohort_attempts(fixed_root: Path, stage: str, *,
                    exclude_root: Path | None = None) -> int | None:
    count = 0
    for sibling in _cohort_roots(fixed_root):
        if exclude_root is not None and sibling.resolve() == Path(exclude_root).resolve():
            continue
        marker = {}
        marker_path = sibling / "cohort.json"
        try:
            marker = read_json(marker_path) if marker_path.is_file() else {}
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(marker, dict):
            return None
        index, bi = _read_registration(sibling, "index")
        ledger, bl = _read_registration(sibling, "ledger")
        kind, _ = _attribution(marker, index, ledger, bi or bl)
        if kind in ("foreign", "legacy"):
            continue
        if kind == "unknown" or bi or bl:
            return None
        index_attempts = {r.get("attempt") for r in (index or [])
                          if r.get("stage") == stage}
        ledger_attempts = {e.get("attempt") for e in (ledger or [])
                           if e.get("stage") == stage}
        disk = _disk_attempts(sibling, stage)
        if (index_attempts != ledger_attempts or index_attempts != disk
                or None in index_attempts):
            return None
        count += len(index_attempts)
    return count


def prior_call_counts(root: Path, *, fixed_root: Path,
                      mode: str = "real") -> dict[str, int] | None:
    roots: list[Path] = []
    if mode == "real":
        for sibling in _cohort_roots(fixed_root):
            if sibling.resolve() == Path(root).resolve():
                continue
            marker = {}
            try:
                if (sibling / "cohort.json").is_file():
                    marker = read_json(sibling / "cohort.json")
            except (OSError, ValueError, TypeError):
                return None
            index, bi = _read_registration(sibling, "index")
            ledger, bl = _read_registration(sibling, "ledger")
            kind, _ = _attribution(marker, index, ledger, bi or bl)
            if kind in ("foreign", "legacy"):
                continue
            if kind == "unknown" or bi or bl:
                return None
            roots.append(sibling)
    roots.append(Path(root))
    totals = {kind: 0 for kind in CALL_KINDS}
    for run_root in roots:
        if _reservation_problems(run_root):
            return None
        try:
            index = [r for r in load_index(run_root)
                     if r.get("cohort_id") == COHORT_ID
                     and r.get("accounting_schema") == ACCOUNTING_SCHEMA]
            ledger = [e for e in load_ledger(run_root)
                      if e.get("cohort_id") == COHORT_ID
                      and e.get("accounting_schema") == ACCOUNTING_SCHEMA]
        except (OSError, ValueError, TypeError):
            return None
        index_by = {(r.get("stage"), r.get("attempt")): r for r in index}
        ledger_by = {(e.get("stage"), e.get("attempt")): e for e in ledger}
        if (len(index_by) != len(index) or len(ledger_by) != len(ledger)
                or set(index_by) != set(ledger_by)
                or any(k[0] is None or k[1] is None for k in index_by)):
            return None
        previous = {kind: 0 for kind in CALL_KINDS}
        for key, row in index_by.items():
            if row.get("status") in ("running", "publishing"):
                return None
            if row.get("status") not in ("done", "crashed", "timeout", "incomplete"):
                return None
            counts = _call_counts_from_balance(row.get("call_balance"))
            ledger_counts = _call_counts_from_balance(
                ledger_by[key].get("call_balance"))
            if counts is None or ledger_counts != counts:
                return None
            for kind in CALL_KINDS:
                if counts[kind] < previous[kind]:
                    return None
                previous[kind] = counts[kind]
        for kind in CALL_KINDS:
            totals[kind] = max(totals[kind], previous[kind])
    return totals


def budget_for(root: Path, stage: str, *, fixed_root: Path,
               mode: str = "real") -> dict:
    if stage not in STAGE_BUDGETS:
        raise ValueError(f"unknown stage {stage}")
    spec = STAGE_BUDGETS[stage]
    local = stage_history(root, stage)
    local_total = total_history(root)
    if local is None or local_total is None:
        raise SystemExit(f"local {stage} cost history unknown")
    if mode == "synthetic":
        prior_stage, prior_total = 0.0, 0.0
        used = len({r.get("attempt") for r in load_index(root)
                    if r.get("stage") == stage})
    else:
        history = cohort_history(fixed_root, exclude_root=root)
        if not history.get("known"):
            raise SystemExit(f"cohort history unknown: {history.get('reason')}")
        prior_stage = float(history.get("stage_cost_s", {}).get(stage, 0.0))
        prior_total = float(history.get("total_s", 0.0))
        used = cohort_attempts(fixed_root, stage, exclude_root=root)
        if used is None:
            raise SystemExit("cohort attempt history unknown")
        used += len({r.get("attempt") for r in load_index(root)
                     if r.get("stage") == stage})
    if used >= spec["attempts"]:
        raise SystemExit(f"{stage} attempt limit reached: {used} >= {spec['attempts']}")
    remaining_stage = spec["budget_s"] - local - prior_stage
    remaining_total = TOTAL_BUDGET_S - local_total - prior_total
    remaining = min(remaining_stage, remaining_total)
    if remaining <= spec["reserve_s"]:
        raise SystemExit(
            f"{stage} budget exhausted: local {local:.1f}s + prior "
            f"{prior_stage:.1f}s of {spec['budget_s']}s")
    return {
        "stage": stage, "budget_s": spec["budget_s"],
        "reserve_s": spec["reserve_s"], "attempts": spec["attempts"],
        "local_stage_s": local, "prior_stage_s": prior_stage,
        "prior_total_s": prior_total, "remaining_s": remaining,
        "attempts_used": used,
    }


def pre_register_attempt(root: Path, stage: str, attempt: str, t0: float,
                         *, scope: dict, budget: dict,
                         registration_sha: str | None = None,
                         call_balance: Mapping[str, Any] | None = None) -> dict:
    if int(budget.get("attempts_used", 0)) >= int(budget.get("attempts", 0)):
        raise SystemExit(f"{stage} attempt limit reached before registration")
    marker = _cohort_path(root)
    if marker.is_file() and read_json(marker).get("cohort_id") not in (None, COHORT_ID):
        raise ValueError("cohort marker mismatch")
    elapsed = max(0.0, time.perf_counter() - t0)
    entry = {
        "status": "running", "complete": False,
        "measured_wall_s": elapsed,
        "accounting_schema": ACCOUNTING_SCHEMA,
        "cohort_id": COHORT_ID,
        "execution_registration_sha256": registration_sha,
        "scope": scope, "budget": budget,
        "call_balance": dict(call_balance) if call_balance else None,
        "deadline_perf_counter": t0 + float(budget["remaining_s"]),
        "allocated_wall_s": time.time(),
    }
    upsert_index(root, stage, attempt, entry)
    upsert_cost(root, stage, attempt, elapsed,
                {"complete": False, "scope": scope, "budget": budget,
                 "call_balance": dict(call_balance) if call_balance else None},
                "running")
    return entry


@dataclass
class SealResult:
    sealed: bool
    row: dict
    problems: list[str] = field(default_factory=list)
    terminal: bool = False


def _report_hashes(row: dict) -> dict:
    out = dict(row)
    for key in ("report", "summary"):
        rel = out.get(key)
        if rel:
            p = Path(rel)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.is_file():
                out[f"{key}_sha256"] = sha256_file(p)
    return out


def finish_attempt(root: Path, stage: str, attempt: str, t0: float,
                   budget: dict, row: dict, *, status: str = "done",
                   extra: dict | None = None,
                   cost_floor: float | None = None) -> SealResult:
    """running -> publishing -> seal; non-consumable on unproven coverage.

    ``cost_floor`` preserves a previously published cost when an attempt is
    later withdrawn; withdrawal must never reduce the billed wall time.
    """
    extra = dict(extra or {})
    floor = 0.0 if cost_floor is None else float(cost_floor)
    if not _nonneg(floor):
        raise ValueError("cost floor must be finite and non-negative")
    (stage_dir(root, stage) / attempt).mkdir(parents=True, exist_ok=True)
    row = {**row, "stage": stage, "attempt": attempt,
           "accounting_schema": ACCOUNTING_SCHEMA, "cohort_id": COHORT_ID}
    balance = extra.get("call_balance")
    if isinstance(balance, Mapping):
        row["call_balance"] = dict(balance)
    row = _report_hashes(row)
    writes: list[float] = []
    problems: list[str] = []

    def timed(fn, *args):
        start = time.perf_counter()
        fn(*args)
        writes.append(time.perf_counter() - start)

    def reserve() -> float:
        return max(SEAL_TAIL_RESERVE_S, 2.0 * max(writes, default=0.0))

    def elapsed() -> float:
        return max(floor, time.perf_counter() - t0)

    def publish(rw: dict) -> dict:
        sampled = elapsed()
        rw = {**rw, "status": "publishing", "measured_wall_s": sampled,
              "sampled_wall_s": sampled}
        timed(upsert_index, root, stage, attempt, rw)
        timed(upsert_cost, root, stage, attempt, sampled,
              {**extra, "complete": False,
               "incomplete_reason": rw.get("incomplete_reason"),
               "sampled_wall_s": sampled}, "publishing")
        return rw

    remaining = float(budget["remaining_s"])
    overrun = elapsed() >= remaining
    if overrun:
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_after_registration")
        status = "incomplete"
    row = publish(row)
    if not overrun and elapsed() >= remaining:
        overrun = True
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_after_publication")
        status = "incomplete"
        row = publish(row)

    def terminal(rw: dict, st: str, cost: float, tail: float) -> dict:
        sampled = elapsed()
        rw = {**rw, "status": st, "measured_wall_s": cost,
              "sampled_wall_s": sampled, "seal_tail_reserve_s": tail}
        timed(upsert_index, root, stage, attempt, rw)
        timed(upsert_cost, root, stage, attempt, cost,
              {**extra, "complete": bool(rw.get("complete", False)),
               "incomplete_reason": rw.get("incomplete_reason"),
               "sampled_wall_s": sampled, "seal_tail_reserve_s": tail}, st)
        return rw

    tail = reserve()
    if not overrun and elapsed() + tail >= remaining:
        overrun = True
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_before_seal")
        status = "incomplete"
        row = publish(row)
    try:
        row = terminal(row, status, elapsed() + tail, tail)
    except (OSError, ValueError, TypeError):
        row.update(status="publishing", complete=False,
                   incomplete_reason="seal_write_failed")
        try:
            row = publish(row)
        except (OSError, ValueError, TypeError):
            pass
        problems.append("seal_write_failed")
        return SealResult(False, row, problems)
    for _ in range(SEAL_VERIFY_ROUNDS):
        post = elapsed()
        if not overrun and post >= remaining:
            overrun = True
            row.update(complete=False,
                       incomplete_reason="budget_exceeded_after_seal")
            status = "incomplete"
            tail = reserve()
            row = publish(row)
            row = terminal(row, status, post + tail, tail)
            continue
        if post <= float(row.get("measured_wall_s", 0.0)):
            break
        tail = reserve()
        if not overrun and post + tail >= remaining:
            overrun = True
            row.update(complete=False,
                       incomplete_reason="seal_tail_reserve_insufficient")
            status = "incomplete"
            row = publish(row)
            row = terminal(row, status, post + tail, tail)
            continue
        row = terminal(row, status, post + tail, tail)
    else:
        row.update(complete=False, status="publishing",
                   incomplete_reason="seal_tail_reserve_unverifiable")
        try:
            row = publish(row)
        except (OSError, ValueError, TypeError):
            problems.append("seal_publish_failed")
        problems.append("seal_tail_reserve_unverifiable")
        return SealResult(False, row, problems)
    ok = not overrun and row.get("complete") is True
    terminal_state = row.get("status") in ("done", "crashed", "timeout",
                                           "incomplete")
    return SealResult(ok, row, problems, terminal_state)


def stage_state(root: Path, stage: str) -> tuple[dict | None, list[str]]:
    problems: list[str] = []
    try:
        index = [r for r in load_index(root) if r.get("stage") == stage]
        ledger = [r for r in load_ledger(root) if r.get("stage") == stage]
    except (OSError, ValueError, TypeError):
        return None, ["stage_registration_unreadable"]
    disk = _disk_attempts(root, stage)
    index_by = {r.get("attempt"): r for r in index}
    ledger_by = {e.get("attempt"): e for e in ledger}
    if (len(index_by) != len(index) or len(ledger_by) != len(ledger)
            or set(index_by) != set(ledger_by) or set(disk) != set(index_by)):
        return None, ["stage_attempt_sets_disagree"]
    if not index_by:
        return None, ["stage_not_registered"]
    terminal = [a for a, r in index_by.items()
                if r.get("status") in ("done", "crashed", "timeout",
                                       "incomplete")]
    if len(terminal) != 1:
        return None, ["stage_not_terminal"] + _reservation_problems(root)
    attempt = terminal[0]
    row = index_by[attempt]
    problems += _reservation_problems(root)
    problems += _row_identity_problems(row, kind="index")
    problems += _row_identity_problems(ledger_by[attempt], kind="ledger")
    problems += _call_balance_problems(row, ledger_by[attempt])
    if row.get("complete") is not True:
        problems.append("attempt_not_complete")
    if row.get("status") != "done" or ledger_by[attempt].get("status") != "done":
        problems.append("attempt_status_not_done")
    for key in ("report", "summary"):
        if row.get(key) and not (PROJECT_ROOT / row[key]).is_file():
            problems.append(f"{key}_missing")
    return row, problems


def unit_admission(root: Path, stage: str, attempt: str,
                   parent: Mapping[str, Any], *, deadline: float) -> list[str]:
    problems: list[str] = []
    rows = [r for r in load_index(root)
            if r.get("stage") == stage and r.get("attempt") == attempt]
    if len(rows) != 1:
        return ["unit_parent_not_registered"]
    row = rows[0]
    if row.get("status") != "running":
        problems.append("unit_parent_not_running")
    if row.get("cohort_id") != COHORT_ID:
        problems.append("unit_parent_cohort")
    if row.get("execution_registration_sha256") != parent.get(
            "registration_sha256"):
        problems.append("unit_parent_registration")
    # ``deadline`` is the absolute work deadline; the seal reserve is already
    # removed by the caller and must not be deducted a second time.
    if time.perf_counter() >= float(deadline):
        problems.append("unit_deadline")
    return problems


def _stage_unit_guard(root: Path, stage: str, attempt: str, *,
                      registration_sha: str | None,
                      deadline: float) -> Callable[[str], None]:
    """Admission check before every unit of a registered P/I attempt."""
    parent = {"registration_sha256": registration_sha, "reserve_s": 0.0}

    def guard(unit: str) -> None:
        problems = unit_admission(root, stage, attempt, parent,
                                  deadline=deadline)
        if problems:
            raise TimeoutError(f"unit_admission:{unit}:{','.join(problems)}")

    return guard


def reserve_stage_calls(root: Path, stage: str, attempt: str,
                        reserved: Mapping[str, Any]) -> Path:
    path = stage_dir(root, stage) / attempt / "call_reservations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = reserved.get("counts") if isinstance(reserved, Mapping) else None
    if not isinstance(counts, Mapping) or set(counts) != set(CALL_KINDS):
        raise ValueError("reservation must cover all call kinds")
    for kind, value in counts.items():
        if not _json_int(value):
            raise ValueError(f"reservation {kind} is not a JSON integer")
    write_json(path, {"schema": "w1_real_training_update_reservation_v1",
                      "reserved": {k: int(v) for k, v in counts.items()},
                      "consumed": None})
    return path


def settle_stage_calls(root: Path, stage: str, attempt: str,
                       consumed: Mapping[str, Any]) -> Path:
    path = stage_dir(root, stage) / attempt / "call_reservations.json"
    data = read_json(path) if path.is_file() else None
    if not isinstance(data, Mapping) or not isinstance(data.get("reserved"),
                                                       Mapping):
        raise ValueError("reservation missing")
    counts = consumed.get("counts") if isinstance(consumed, Mapping) else None
    if not isinstance(counts, Mapping) or set(counts) != set(CALL_KINDS):
        raise ValueError("consumed snapshot must cover all call kinds")
    for kind in CALL_KINDS:
        value = counts.get(kind)
        if not _json_int(value):
            raise ValueError(f"consumed {kind} is not a JSON integer")
        if int(value) > int(data["reserved"][kind]):
            raise ValueError(f"consumed {kind} exceeds reservation")
    write_json(path, {**data, "consumed": {k: int(v) for k, v in counts.items()}})
    return path


def _report_claims_complete(report_path: Path | None) -> bool:
    if report_path is None or not Path(report_path).is_file():
        return False
    try:
        data = read_json(Path(report_path))
    except (OSError, ValueError, TypeError):
        return True  # unreadable report cannot be proven non-claiming
    if not isinstance(data, Mapping):
        return True
    return bool(data.get("complete_evidence")
                or data.get("covered_contract_passed")
                or data.get("passed"))


def _attempt_row(root: Path, stage: str, attempt: str) -> dict:
    try:
        rows = [r for r in load_index(root)
                if r.get("stage") == stage and r.get("attempt") == attempt]
    except (OSError, ValueError, TypeError):
        return {"attempt": attempt}
    return dict(rows[0]) if len(rows) == 1 else {"attempt": attempt}


def _amend_final_report(report_path: Path | None,
                        problems: Sequence[str]) -> bool:
    """Make a disk summary retract its completion claim before resealing."""
    if report_path is None or not Path(report_path).is_file():
        return False
    try:
        data = read_json(Path(report_path))
        if not isinstance(data, dict):
            return False
        current = data.get("verification_problems")
        merged = list(current) if isinstance(current, list) else []
        for item in problems:
            if item not in merged:
                merged.append(item)
        data.update(conclusion="audit_incomplete", complete_evidence=False,
                    covered_contract_passed=False,
                    verification_problems=merged)
        write_json(Path(report_path), data)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _valid_call_snapshot(snapshot: Any) -> bool:
    if not isinstance(snapshot, Mapping) or set(snapshot) != set(CALL_KINDS):
        return False
    return all(_json_int(snapshot[kind]) and int(snapshot[kind]) >= 0
               for kind in CALL_KINDS)


def _settle_f_calls(root: Path, stage: str, attempt: str,
                    snapshot: Any) -> tuple[bool, list[str]]:
    """Settle only a complete worker snapshot; otherwise leave it unknown."""
    if not _valid_call_snapshot(snapshot):
        return False, ["f_call_settlement_unknown"]
    try:
        settle_stage_calls(root, stage, attempt,
                           {"counts": {kind: int(snapshot[kind])
                                       for kind in CALL_KINDS}})
        return True, []
    except (OSError, ValueError, KeyError) as exc:
        return False, [f"reservation_settle:{type(exc).__name__}"]


def _call_balance_delta_problems(before: Any, after: Any,
                                 consumed: Any) -> list[str]:
    """Reconcile the final cumulative call snapshot with the reservation."""
    prior = _call_counts_from_balance(before)
    final = _call_counts_from_balance(after)
    if prior is None:
        return ["call_balance_prior_unknown"]
    if final is None:
        return ["call_balance_final_invalid"]
    if not _valid_call_snapshot(consumed):
        return ["call_consumed_unknown"]
    problems = []
    for kind in CALL_KINDS:
        if final[kind] - prior[kind] != int(consumed[kind]):
            problems.append(f"call_balance_delta:{kind}")
    return problems


def withdraw_attempt_claim(root: Path, stage: str, attempt: str,
                           report_path: Path | None, row: dict,
                           reasons: Sequence[str], *, reason: str,
                           t0: float, budget: dict,
                           call_balance: Mapping[str, Any] | None = None
                           ) -> tuple[dict, list[str], bool]:
    """Rewrite a published pass claim to incomplete and reseal honestly.

    The rewrite and reseal follow the same publishing -> terminal protocol as
    ``finish_attempt`` and keep the previously billed wall time as a floor.
    If report rewriting or seal-cost coverage cannot be proven, the row is
    left ``publishing`` (non-consumable) and the third value is False.
    ``call_balance`` is the post-execution cumulative snapshot; a withdrawal
    of a running attempt must not restore the pre-execution balance.
    """
    problems: list[str] = []
    row = dict(row)
    if call_balance is not None:
        row["call_balance"] = dict(call_balance)
    old_cost = float(row.get("measured_wall_s", 0.0) or 0.0)
    if not _nonneg(old_cost):
        old_cost = 0.0
    row.update(complete=False, status="incomplete", incomplete_reason=reason,
               withdrawn_reasons=list(reasons))
    rewritten = report_path is not None and Path(report_path).is_file()
    if rewritten:
        try:
            data = read_json(Path(report_path))
            if not isinstance(data, dict):
                raise ValueError("report_not_mapping")
            data.update(conclusion="audit_incomplete",
                        complete_evidence=False,
                        covered_contract_passed=False,
                        withdrawn_reasons=list(reasons))
            write_json(Path(report_path), data)
            row["report"] = path_for_record(Path(report_path))
            row["summary"] = path_for_record(Path(report_path))
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"withdraw_report:{type(exc).__name__}")
            # The pass claim cannot be rewritten: at least publish a
            # non-consumable publishing row so the old claim is not sealed as
            # a successful terminal state.
            try:
                upsert_index(root, stage, attempt,
                             {**row, "status": "publishing",
                              "measured_wall_s": max(
                                  old_cost, time.perf_counter() - t0),
                              "incomplete_reason": "withdraw_report_failed"})
                upsert_cost(root, stage, attempt,
                            max(old_cost, time.perf_counter() - t0),
                            {"complete": False,
                             "incomplete_reason": "withdraw_report_failed",
                             "withdrawn_reasons": list(reasons),
                             **({"call_balance": dict(call_balance)}
                                if call_balance is not None else {})},
                            "publishing")
            except (OSError, ValueError, TypeError) as exc2:
                problems.append(f"withdraw_publish:{type(exc2).__name__}")
            return row, problems, False
    sealed = finish_attempt(
        root, stage, attempt, t0, budget, row, status="incomplete",
        extra={"stage": stage, "withdrawn_reasons": list(reasons),
               "withdrawn_reason": reason,
               **({"call_balance": dict(call_balance)}
                  if call_balance is not None else {})},
        cost_floor=old_cost)
    problems += list(sealed.problems)
    try:
        _row, state_problems = stage_state(root, stage)
    except (OSError, ValueError, TypeError):
        state_problems = ["stage_unreadable"]
    unknown = [p for p in state_problems
               if p not in ("attempt_not_complete", "attempt_status_not_done")]
    proven = bool(sealed.terminal and not unknown and not sealed.problems)
    return sealed.row, problems, proven

# ---------------------------------------------------------------------------
# parent anchors, implementation identity and R0
# ---------------------------------------------------------------------------

OBJECTIVE_PARENT = Path(
    "derived/experiments/omniretarget_first/"
    "20261027_b45cd7c_w1_real_objective_audit")
FK_PARENT = Path(
    "derived/experiments/omniretarget_first/"
    "20261028_2083249_w1_real_fk_audit")
PARENT_FILES = {
    "objective_registration": (
        OBJECTIVE_PARENT / "registration/attempt_001/registration.json",
        "ecc11368219a33c95a6038769398fd11789dcdd04430760f54ae3a84ba58a66a"),
    "objective_admission": (
        OBJECTIVE_PARENT / "registration/attempt_001/admission.json",
        "61c5b16d883d0fa084632a548bd7428e2fe580540990feed47926546bd32ce95"),
    "objective_p_report": (
        OBJECTIVE_PARENT / "production/attempt_001/p_report.json",
        "1ea15ddc2da7a0acb5125f1ecc92e1ea7e0cc74589e7a7ad5055b26055b7cdcb"),
    "objective_arrays": (
        OBJECTIVE_PARENT / "production/attempt_001/objective_real_arrays.npz",
        "4cf20acb13fcdee207e0492a3b07818cc9ed31202409177fa1ed3f13086fd2c8"),
    "objective_gradients": (
        OBJECTIVE_PARENT / "production/attempt_001/gradient_real_arrays.npz",
        "49b21c6c5d82a19e92f2c0d870b1d7f036986604c009c6195e6f86a035765ea7"),
    "objective_summary": (
        OBJECTIVE_PARENT / "finalize/attempt_001/audit_summary.json",
        "0a839071125a156273b3d579a564965cd0edb196b8f900b05417edd0cb00db97"),
    "objective_verification": (
        OBJECTIVE_PARENT / "finalize/attempt_001/independent_verification.json",
        "f0bf351bd0bd184f52198ac481293bae9127607ec98864010768430a4dcbd961"),
    "fk_registration": (
        FK_PARENT / "registration/attempt_001/registration.json",
        "93a7096fd58139d52351763befc673f9be49c42882fc898ac063f9582a33d6bb"),
    "fk_final": (
        FK_PARENT / "finalize/attempt_001/fk_audit_final.json",
        "c1b8e0e7136b053dbed6d9710993fae3112896d8560571237aefb4c3a35ab674"),
    "fk_verification": (
        FK_PARENT / "finalize/attempt_001/independent_verification.json",
        "9432985de7cb6f3dc19ab6b19876d897846bd9f7373df9fc45bed456530bb647"),
}
CHECKPOINTS = {
    "deterministic": (
        Path("derived/experiments/omniretarget_first/20261013_be00c50_w1_de/"
             "formal/deterministic/attempt_001/best.pt"),
        "7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff"),
    "diffusion": (
        Path("derived/experiments/omniretarget_first/20261013_be00c50_w1_de/"
             "formal/diffusion/attempt_001/best.pt"),
        "1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1"),
}
TRAIN_CONFIG_PATH = Path("configs/train/omniretarget_motion.yaml")
TRAIN_CONFIG_SHA = "0e2cfa96849ee2de14ac84b7343289090c606f78116bb79114ff81b046b521e0"

IMPLEMENTATION_FILES = (
    "src/ireset/eval/w1_real_training_update_audit.py",
    "scripts/audit_w1_real_training_update.py",
    "tests/test_w1_real_training_update_audit.py",
    "src/ireset/train/trainer.py",
    "src/ireset/train/losses.py",
    "src/ireset/model/backbone.py",
    "src/ireset/model/conditioner.py",
    "src/ireset/diffusion/generator.py",
    "src/ireset/eval/w1_real_fk_audit.py",
    "src/ireset/eval/w1_real_objective_audit.py",
    "configs/train/omniretarget_motion.yaml",
    "docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md",
)
PREREG_DRAFT = IMPLEMENTATION_FILES[-1]


def _git_identity(project_root: Path) -> dict:
    try:
        head = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout
        diff = subprocess.run(
            ["git", "-C", str(project_root), "diff", "HEAD"],
            capture_output=True, check=True).stdout
        return {"head": head, "dirty": bool(status.strip()),
                "status_sha256": sha256_bytes(status.encode()),
                "tracked_diff_sha256": sha256_bytes(diff)}
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": None, "status_sha256": None,
                "tracked_diff_sha256": None}


def implementation_identity(project_root: Path) -> dict:
    root = Path(project_root)
    files = {}
    for rel in IMPLEMENTATION_FILES:
        p = root / rel
        files[rel] = sha256_file(p) if p.is_file() else None
    return {"schema": "w1_real_training_update_implementation_v1",
            "files": files, "git": _git_identity(root)}


def _fixture_schema_problems(provider: SyntheticTrainingProvider) -> list[str]:
    problems: list[str] = []
    fixture = provider.fixture or {}
    windows = fixture.get("windows")
    expected = [{"id": WINDOW_IDS[i], "demo": spec[0], "start": spec[1],
                 "end": spec[2], "T": spec[3]}
                for i, spec in enumerate(WINDOW_SPECS)]
    if windows != expected:
        problems.append("fixture_windows")
    schedule = {str(u): [list(x) for x in MICRO_SCHEDULE[u]] for u in UPDATES}
    if fixture.get("micro_schedule") != schedule:
        problems.append("fixture_schedule")
    if fixture.get("noise_seed") != NOISE_SEED:
        problems.append("fixture_noise_seed")
    for key in WINDOW_IDS:
        arrays = provider.window_arrays[key]
        problems += _array_shape_problems(arrays["x0"], name=f"{key}.x0",
                                          shape=(WINDOW, 38), dtype=np.float32)
        problems += _array_shape_problems(arrays["kps_obj"],
                                          name=f"{key}.kps_obj",
                                          shape=(WINDOW, 52, 3),
                                          dtype=np.float32)
        problems += _array_shape_problems(arrays["kps_clip"],
                                          name=f"{key}.kps_clip",
                                          shape=(WINDOW, 52, 3),
                                          dtype=np.float32)
        problems += _array_shape_problems(arrays["box_t_clip"],
                                          name=f"{key}.box_t_clip",
                                          shape=(WINDOW, 3), dtype=np.float32)
        problems += _array_shape_problems(arrays["box_quat"],
                                          name=f"{key}.box_quat",
                                          shape=(WINDOW, 4), dtype=np.float32)
        problems += _array_shape_problems(arrays["contact"],
                                          name=f"{key}.contact",
                                          shape=(WINDOW, 4), dtype=np.float32)
        problems += _array_shape_problems(arrays["wfc"], name=f"{key}.wfc",
                                          shape=(4, 4), dtype=np.float32)
        quat = np.asarray(arrays["box_quat"], dtype=np.float64)
        norms = np.linalg.norm(quat, axis=-1)
        if not ((norms > 1e-6) & np.isfinite(norms)).all():
            problems.append(f"{key}.box_quat_norm")
        decoded = independent_decode(
            np.asarray(arrays["x0"], np.float32).astype(np.float64),
            provider.constants["mean32"].astype(np.float64),
            provider.constants["std32"].astype(np.float64),
            provider.constants["lo32"].astype(np.float64),
            provider.constants["hi32"].astype(np.float64))
        if not decoded["ok"].all():
            problems.append(f"{key}.target_domain")
    return problems


def _artifact_classes(roster: Mapping[str, Any]) -> dict:
    """Byte upper bound per artifact class for both attempt slots.

    Counts come from the actual save sites, not a fixed copy multiplier:
    P stores theta_init + 2 per micro (accum, contrib) + 8 update states;
    I stores 17 float64 AdamW/propagation buffers, 7 float32 row buffers and
    3 float32 explicit-update arrays per update; F stores a P replay plus an
    I replay.  Two attempts per stage are retained under the cohort cap.
    """
    numel = sum(int(e["numel"]) for model in MODELS
                for e in roster["models"][model]["parameters"])
    param_bytes = numel * 4
    p_copies = (1                                    # theta_init
                + 2 * MICRO_COUNT * len(UPDATES)     # accum + contrib
                + 8 * len(UPDATES))                  # theta/pre/post/state
    i_f64_copies = 17 * len(UPDATES)                 # AdamW + intervals
    i_f32_copies = (7 + 3) * len(UPDATES)            # row + explicit update
    forward_bytes = (MICRO_COUNT * len(UPDATES) * len(MODELS)
                     * 5 * int(MODEL_CONFIG["micro_batch"]) * WINDOW * 38 * 4)
    p_json = 32 * 2 ** 20
    i_json = 64 * 2 ** 20
    f_json = 64 * 2 ** 20
    p_bytes = int(param_bytes * p_copies + forward_bytes + p_json)
    i_bytes = int(param_bytes * (i_f64_copies * 2 + i_f32_copies)
                  + forward_bytes + i_json)
    f_bytes = int(p_bytes + i_bytes + f_json)
    fixture_bytes = 64 * 2 ** 20
    registration_bytes = 8 * 2 ** 20
    attempts = 2
    cohort_bytes = int(attempts * (p_bytes + i_bytes + f_bytes)
                       + fixture_bytes + registration_bytes)
    classes = {
        "parameter_bytes": param_bytes,
        "p_parameter_array_bytes": int(param_bytes * p_copies),
        "i_float64_propagation_bytes": int(param_bytes * i_f64_copies * 2),
        "i_float32_array_bytes": int(param_bytes * i_f32_copies),
        "forward_array_bytes_per_attempt": int(forward_bytes),
        "p_json_bytes": p_json,
        "i_json_bytes": i_json,
        "f_json_bytes": f_json,
        "fixture_bytes": fixture_bytes,
        "registration_bytes": registration_bytes,
        "p_attempt_bytes": p_bytes,
        "i_attempt_bytes": i_bytes,
        "f_attempt_bytes": f_bytes,
        "attempts_per_stage": attempts,
    }
    return {"classes": classes, "cohort_bytes": cohort_bytes,
            "one_chain_bytes": int(p_bytes + i_bytes + f_bytes
                                   + fixture_bytes)}


def _resource_bound(roster: Mapping[str, Any]) -> dict:
    artifacts = _artifact_classes(roster)
    total = int(artifacts["cohort_bytes"])
    classes = dict(artifacts["classes"])
    classes["one_chain_bytes"] = int(artifacts["one_chain_bytes"])
    classes["cohort_bytes"] = total
    return {
        **classes,
        "estimated_artifacts_bytes": total,
        "estimated_artifacts_gib": round(total / 2 ** 30, 3),
        "one_chain_gib": round(artifacts["one_chain_bytes"] / 2 ** 30, 3),
        "artifact_cap_gib": 16,
        "host_peak_cap_gib": 32,
        "gpu_peak_cap_gib": 22,
        "disk_free_min_gib": 20,
        "within_cap": total <= 16 * 2 ** 30,
    }


def run_r0(root: Path, stage: str, attempt: str, t0: float,
           budget: Mapping[str, Any], provider: SyntheticTrainingProvider,
           counter: CallCounter, roster: Mapping[str, Any],
           *, mode: str, tolerance: Mapping[str, Any],
           project_root: Path = PROJECT_ROOT,
           admission_package: Mapping[str, Any] | None = None,
           authorization: Mapping[str, Any] | None = None) -> dict:
    base = stage_dir(root, stage) / attempt
    base.mkdir(parents=True, exist_ok=True)
    fixture = provider.fixture or {}
    problems: list[str] = list(fixture.get("problems") or [])
    gaps: list[str] = []
    # tolerance + fp32 contract
    problems += [f"tolerance:{p}" for p in tolerance_contract_problems(
        tolerance, recompute=(mode == "real"))]
    fp32_ok = True
    if mode == "real":
        fp32_ok = bool(tolerance.get("fp32_contract"))
        problems += [f"fp32:{p}" for p in tolerance.get("fp32_problems", [])
                     if p]
    # roster/schema/noise
    roster_here = roster_problems(provider.roster, roster)
    problems += [f"roster:{p}" for p in roster_here]
    problems += [f"fixture:{p}" for p in _fixture_schema_problems(provider)]
    schedule = draw_noise_path(NOISE_SEED)
    counter.add("cpu_draw_units", DRAW_UNITS_PER_PATH)
    noise_ok = True
    for a, b in zip(schedule, provider.noise):
        if (not torch.equal(a["t"], b["t"])
                or not torch.equal(a["epsilon"], b["epsilon"])):
            noise_ok = False
            problems.append("r0_noise_matrix")
            break
    alpha = alpha_cumprod_reference(provider.config)
    if _hash_array(alpha) != fixture.get("alpha_cumprod_sha256"):
        problems.append("r0_alpha_hash")
    if not noise_ok:
        gaps.append("r0_noise_matrix")
    catalog = row_catalog(roster)
    if catalog["n_rows"] != 14 * int(roster["n_total"]) + 168:
        problems.append("r0_row_formula")
    # runtime
    runtime = configure_production_runtime()
    problems += [f"runtime:{p}" for p in runtime["problems"]]
    env = environment_facts()
    if provider.device == "cuda" and not torch.cuda.is_available():
        problems.append("r0_cuda_unavailable")
    # resources
    bound = _resource_bound(roster)
    if not bound["within_cap"]:
        problems.append("r0_artifact_bound")
    import shutil
    free = shutil.disk_usage(project_root).free
    if free < bound["disk_free_min_gib"] * 2 ** 30:
        problems.append("r0_disk_free")
    # parent anchors / checkpoints (metadata only)
    parents = {}
    for name, (rel, want) in PARENT_FILES.items():
        p = Path(project_root) / rel
        actual = sha256_file(p) if p.is_file() else None
        parents[name] = {"path": path_for_record(p), "expected": want,
                         "actual": actual,
                         "match": actual is not None and actual == want}
        if mode == "real":
            if actual is None:
                problems.append(f"r0_parent_missing:{name}")
            elif actual != want:
                problems.append(f"r0_parent_drifted:{name}")
    checkpoints = {}
    for model, (rel, want) in CHECKPOINTS.items():
        p = Path(project_root) / rel
        actual = sha256_file(p) if p.is_file() else None
        checkpoints[model] = {"path": path_for_record(p), "expected": want,
                              "actual": actual,
                              "match": actual is not None and actual == want,
                              "step": 4000}
        if mode == "real" and actual != want:
            problems.append(f"r0_checkpoint:{model}")
    config_path = Path(project_root) / TRAIN_CONFIG_PATH
    config_sha = sha256_file(config_path) if config_path.is_file() else None
    if config_sha != TRAIN_CONFIG_SHA:
        problems.append("r0_train_config_sha")
    impl = implementation_identity(project_root)
    missing_impl = [rel for rel, sha in impl["files"].items() if sha is None]
    if missing_impl:
        problems.append(f"r0_implementation_missing:{missing_impl}")
    if admission_package is not None:
        expected_files = (admission_package.get("implementation") or {}).get(
            "files") or {}
        if expected_files != impl["files"]:
            problems.append("r0_implementation_drift")
    registration = {
        "schema": REGISTRATION_SCHEMA,
        "mode": mode,
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "stage": stage,
        "attempt": attempt,
        "window_keys": [{"id": WINDOW_IDS[i], "demo": spec[0],
                         "start": spec[1], "end": spec[2], "T": spec[3]}
                        for i, spec in enumerate(WINDOW_SPECS)],
        "micro_schedule": {str(u): [list(x) for x in MICRO_SCHEDULE[u]]
                           for u in UPDATES},
        "noise": {"seed": NOISE_SEED, "protocol": NOISE_PROTOCOL,
                  "draw_units": DRAW_UNITS_PER_PATH,
                  "t_epsilon_sha256": [_sha_array(u["t"])
                                       + _sha_array(u["epsilon"])
                                       for u in schedule],
                  "alpha_cumprod_sha256": _hash_array(alpha)},
        "roster": roster,
        "row_catalog": {k: catalog[k] for k in
                        ("n_rows", "counts", "keys_sha256", "formula")},
        "config": provider.config,
        "budget": dict(budget),
        "call_caps": dict(CALL_CAPS),
        "call_nominal": {k: dict(v) for k, v in STAGE_NOMINAL.items()},
        "tolerances": tolerance.get("tolerances"),
        "tiny_zero_floors": {"gradient_abs": ZERO_GRADIENT_ABS,
                             "gradient_noise_scale": ZERO_GRADIENT_NOISE_SCALE,
                             "state_abs": ZERO_STATE_ABS},
        "parents": parents,
        "checkpoints": checkpoints,
        "train_config_sha256": config_sha,
        "implementation": impl,
        "runtime": runtime,
        "environment": env,
        "resource_bound": bound,
        "disk_free_bytes": int(free),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    registration_path = base / "registration.json"
    write_json(registration_path, registration)
    registration_sha = sha256_file(registration_path)
    admission = {
        "schema": "w1_real_training_update_r0_admission_v1",
        "mode": mode,
        "passed": not problems,
        "problems": problems,
        "gaps": gaps,
        "registration_sha256": registration_sha,
        "registration": path_for_record(registration_path),
        "fixture_problems": list(fixture.get("problems") or []),
        "fixture_forward_calls": fixture.get("fixture_forward_calls"),
        "fixture_prediction_max_angle_rad":
            fixture.get("fixture_prediction_max_angle_rad"),
        "zero_gradient_parameters":
            fixture.get("zero_gradient_parameters"),
        "none_gradient_parameters":
            fixture.get("none_gradient_parameters"),
        "call_balance": counter.snapshot(),
        "zero_counters": {k: 0 for k in ZERO_COUNTERS},
        "parent_files_verified": all(v["match"] for v in parents.values())
        if mode == "real" else False,
        "checkpoint_files_verified": all(v["match"]
                                         for v in checkpoints.values())
        if mode == "real" else False,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    admission_path = base / "admission.json"
    write_json(admission_path, admission)
    gaps += list(fixture.get("problems") or [])
    return {
        "passed": not problems,
        "problems": problems,
        "gaps": gaps,
        "registration_sha256": registration_sha,
        "registration": path_for_record(registration_path),
        "admission": path_for_record(admission_path),
        "fp32_ok": fp32_ok,
        "call_balance": counter.snapshot(),
    }

# ---------------------------------------------------------------------------
# effective-float32 contract and synthetic calibration
# ---------------------------------------------------------------------------

NODE_GRAPH_TEXT = (
    "host: bc1=1-beta1**step, bc2_sqrt=(1-beta2**step)**0.5, "
    "h=-lr/bc1, d=1-lr*weight_decay (binary64)\n"
    "decay: theta_decay=R32(R32(d)*theta)\n"
    "m: a=R32(1-beta1); z=R32(g-m_old); m_new=FMA32(a,z,m_old)\n"
    "v: v_scaled=R32(R32(beta2)*v_old); g2=R32(g*g); "
    "v_new=FMA32(R32(1-beta2),g2,v_scaled)\n"
    "denominator: s=R32(sqrt(v_new)); b=R32(s/R32(bc2_sqrt)); "
    "denom=R32(b+R32(eps))\n"
    "theta: ratio=R32(m_new/denom); theta_new=FMA32(R32(h),ratio,theta_decay)"
)


def _torch_adamw_probe(theta: np.ndarray, m: np.ndarray, v: np.ndarray,
                       g: np.ndarray, step: int, device: str,
                       *, lr: float = 1e-4, weight_decay: float = 1e-5,
                       adams: Mapping[str, Any] | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    adams = dict(ADAM_DEFAULTS if adams is None else adams)
    p = torch.tensor(np.asarray(theta, np.float32), dtype=torch.float32,
                     device=device, requires_grad=True)
    opt = torch.optim.AdamW([p], lr=lr, weight_decay=weight_decay,
                            betas=tuple(adams["betas"]),
                            eps=float(adams["eps"]))
    state = opt.state[p]
    state["step"] = torch.tensor(float(step - 1), dtype=torch.float32,
                                 device=device)
    state["exp_avg"] = torch.tensor(np.asarray(m, np.float32),
                                    dtype=torch.float32, device=device)
    state["exp_avg_sq"] = torch.tensor(np.asarray(v, np.float32),
                                       dtype=torch.float32, device=device)
    p.grad = torch.tensor(np.asarray(g, np.float32), dtype=torch.float32,
                          device=device)
    opt.step()
    return (p.detach().cpu().numpy(), state["exp_avg"].detach().cpu().numpy(),
            state["exp_avg_sq"].detach().cpu().numpy(),
            float(state["step"].detach().cpu()))


def _fp32_probe_cases() -> list[dict]:
    rng = np.random.default_rng(20261029)
    n = 64
    theta = (rng.choice([-1.0, 1.0], n) * 10 ** rng.uniform(-2, 1, n)
             ).astype(np.float32)
    v = (10 ** rng.uniform(-14, 2, n)).astype(np.float32)
    m = (rng.choice([-1.0, 1.0], n) * 10 ** rng.uniform(-12, -1, n)
         ).astype(np.float32)
    g = (rng.choice([-1.0, 1.0], n) * 10 ** rng.uniform(-12, -1, n)
         ).astype(np.float32)
    mask = rng.random(n) < 0.4
    g[mask] = (m[mask] + rng.choice([-1.0, 1.0], int(mask.sum()))
               * 10 ** rng.uniform(-10, -4, int(mask.sum()))).astype(np.float32)
    return [
        {"name": "plan_1ulp_pair", "theta": np.array([0.0, 0.0], np.float32),
         "m": np.array([0.001, 0.001], np.float32),
         "v": np.array([1.0000000116860974e-7, 1.0000000116860974e-7],
                       np.float32),
         "g": np.array([-0.008999999612569809, -0.008999998681247234],
                       np.float32), "step": 2},
        {"name": "random_adversarial", "theta": theta, "m": m, "v": v,
         "g": g, "step": 2},
        {"name": "subnormal", "theta": np.zeros(3, np.float32),
         "m": np.zeros(3, np.float32),
         "v": np.array([1e-40, 1e-38, 1e-45], np.float32),
         "g": np.array([1e-20, 1e-19, 1e-23], np.float32), "step": 2},
        {"name": "first_step", "theta": np.array([0.5, -0.5], np.float32),
         "m": np.zeros(2, np.float32), "v": np.zeros(2, np.float32),
         "g": np.array([0.01, -0.02], np.float32), "step": 1},
    ]


def build_fp32_contract(device: str = "cuda") -> dict:
    if device != "cuda" or not torch.cuda.is_available():
        return {"schema": FP32_SCHEMA, "device": device, "valid": False,
                "problems": ["fp32_contract_requires_cuda"],
                "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    import torch.optim.adam as adam_module

    cases = []
    mismatches: list[str] = []
    for case in _fp32_probe_cases():
        theta, m, v, g, step = (case["theta"], case["m"], case["v"],
                                case["g"], int(case["step"]))
        tp, tm, tv, ts = _torch_adamw_probe(theta, m, v, g, step, device)
        eff = adamw_effective_fp32(theta, m, v, g, step,
                                   m_none=bool(np.all(m == 0) and step == 1),
                                   v_none=bool(np.all(v == 0) and step == 1))
        ok = (np.array_equal(tp, eff["theta"])
              and np.array_equal(tm, eff["m"])
              and np.array_equal(tv, eff["v"])
              and ts == step)
        if not ok:
            bad = int(np.sum(tp != eff["theta"]))
            mismatches.append(f"{case['name']}:theta:{bad}")
        cases.append({
            "name": case["name"], "step": step, "size": int(theta.size),
            "theta_match": bool(np.array_equal(tp, eff["theta"])),
            "m_match": bool(np.array_equal(tm, eff["m"])),
            "v_match": bool(np.array_equal(tv, eff["v"])),
            "step_match": bool(ts == step),
        })
    # two-step recursion, graph state fed forward
    rng = np.random.default_rng(77)
    theta0 = (rng.standard_normal(16) * 0.2).astype(np.float32)
    g1 = (rng.standard_normal(16) * 0.01).astype(np.float32)
    g2 = (rng.standard_normal(16) * 0.01).astype(np.float32)
    t1, m1, v1, s1 = _torch_adamw_probe(theta0, np.zeros(16, np.float32),
                                        np.zeros(16, np.float32), g1, 1,
                                        device)
    t2, m2, v2, s2 = _torch_adamw_probe(t1, m1, v1, g2, 2, device)
    e1 = adamw_effective_fp32(theta0, None, None, g1, 1, m_none=True,
                              v_none=True)
    e2 = adamw_effective_fp32(e1["theta"], e1["m"], e1["v"], g2, 2)
    two_step_ok = (np.array_equal(t1, e1["theta"])
                   and np.array_equal(t2, e2["theta"])
                   and np.array_equal(m2, e2["m"])
                   and np.array_equal(v2, e2["v"]))
    if not two_step_ok:
        mismatches.append("two_step_recursion")
    # subnormal / FTZ facts
    ftz = []
    for value in (1e-38, 1e-40, 1e-45):
        x = np.array([value], np.float32)
        prod = (x * np.float32(1.0)).astype(np.float32)
        ftz.append({"value": float(value), "product": float(prod[0]),
                    "is_subnormal": bool(0 < abs(float(x[0]))
                                         < float(np.finfo(np.float32).tiny))})
    env = environment_facts()
    source = {}
    for name in ("torch.optim.adam", "torch.optim.adamw"):
        try:
            mod = __import__(name, fromlist=["x"])
            source[name] = sha256_file(Path(mod.__file__))
        except (ImportError, OSError):
            source[name] = None
    device_name = torch.cuda.get_device_name(0)
    record = {
        "schema": FP32_SCHEMA,
        "device": device,
        "device_name": device_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": source,
        "adams": dict(ADAM_DEFAULTS),
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "node_graph": NODE_GRAPH_TEXT,
        "host_constants": {
            str(step): {k: v for k, v in host_constants(step).items()
                        if isinstance(v, (int, float))}
            for step in (1, 2)},
        "probe_cases": cases,
        "probe_mismatches": mismatches,
        "two_step_match": two_step_ok,
        "ftz_probe": ftz,
        "dispatch": {"foreach_default": True, "fused_default": False,
                     "capturable": False, "differentiable": False},
        "rules": {
            "rounding": "IEEE binary32 round-to-nearest-ties-to-even",
            "fma": "exact product+sum, one binary32 rounding",
            "subnormal": "gradual underflow (verified by probe)",
            "scalar_constants": "binary64 host constants, binary32 at graph nodes",
            "exact_fallback": "Fraction/Decimal at midpoint-unsafe elements",
        },
        "environment": env,
        "valid": not mismatches,
        "problems": [] if not mismatches else mismatches,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return record


def fp32_contract_problems(contract: Mapping[str, Any] | None,
                           *, recompute: bool = True) -> list[str]:
    if not isinstance(contract, Mapping):
        return ["fp32_contract_missing"]
    problems: list[str] = []
    if contract.get("schema") != FP32_SCHEMA:
        problems.append("fp32_schema")
    if contract.get("device") != "cuda":
        problems.append("fp32_device")
    if contract.get("node_graph") != NODE_GRAPH_TEXT:
        problems.append("fp32_node_graph")
    if contract.get("valid") is not True:
        problems.append("fp32_not_valid")
    if contract.get("probe_mismatches"):
        problems.append("fp32_probe_mismatch")
    if contract.get("two_step_match") is not True:
        problems.append("fp32_two_step")
    if not isinstance(contract.get("source_sha256"), Mapping) or any(
            not isinstance(v, str) for v in
            (contract.get("source_sha256") or {}).values()):
        problems.append("fp32_source_missing")
    if recompute:
        fresh = build_fp32_contract("cuda")
        for key in ("node_graph", "probe_mismatches", "two_step_match",
                    "source_sha256", "valid"):
            if _jsonable(fresh.get(key)) != _jsonable(contract.get(key)):
                problems.append(f"fp32_not_reproducible:{key}")
    return problems


def _ratio_record(ratio: float, *, source: str) -> dict:
    return {"max_abs": float(ratio), "max_relative": float(ratio),
            "max_ratio_of_proposal": float(ratio), "source": source}


def _merge_ratio_records(records: Sequence[Mapping[str, Any]]) -> dict:
    """Keep the largest normalized error over every applicable positive case."""
    usable = [r for r in records if r is not None]
    if not usable:
        return _ratio_record(0.0, source="none")
    best = max(usable, key=lambda r: float(
        r.get("max_ratio_of_proposal", -math.inf)))
    sources = [str(r.get("source")) for r in usable if r.get("source")]
    return {**dict(best), "source": "+".join(dict.fromkeys(sources)),
            "sources": sources}


def _scalar_case(theta: float, m: float, v: float, step: int,
                 g_p: float, g_i: float, *, m_none: bool,
                 post_p: float | None = None,
                 post_i: float | None = None,
                 v_post_p: float | None = None) -> dict:
    host = host_constants(step)
    rp = adamw_effective_fp32(np.array([theta]), np.array([m]),
                              np.array([v]), np.array([g_p]), step,
                              m_none=m_none, v_none=m_none, host=host)
    ri = adamw_effective_fp32(np.array([theta]), np.array([m]),
                              np.array([v]), np.array([g_i]), step,
                              m_none=m_none, v_none=m_none, host=host)
    tp = rp["theta"] if post_p is None else np.array([post_p], np.float32)
    ti = ri["theta"] if post_i is None else np.array([post_i], np.float32)
    mp = rp["m"]
    vi = ri["v"]
    vp = rp["v"] if v_post_p is None else np.array([v_post_p], np.float32)
    d_p = tp.astype(np.float64) - np.float64(theta)
    d_i = ti.astype(np.float64) - np.float64(theta)
    b_p = _delta_write_bound(rp["theta"])
    b_i = _delta_write_bound(ri["theta"])
    gate_same = float(TOLERANCE_PROPOSALS["delta_same_input"]["rtol"])
    gate_cross = float(TOLERANCE_PROPOSALS["delta_cross_path"]["rtol"])
    obs_p, ok_p = _delta_observability(rp["delta_ref64"], d_p, b_p,
                                       rp["theta"], np.array([theta], np.float32),
                                       gate_rtol=gate_same)
    obs_i, ok_i = _delta_observability(ri["delta_ref64"], d_i, b_i,
                                       ri["theta"], np.array([theta], np.float32),
                                       gate_rtol=gate_same)
    p_formula = _formula_state_problems(
        rp, tp, mp, vp, step, m_none=m_none, v_none=m_none,
        expected_step=step)
    i_formula = _formula_state_problems(
        ri, ti, ri["m"], vi, step, m_none=m_none, v_none=m_none,
        expected_step=step)
    zero_problems = _zero_state_problems(rp["m"], rp["v"], mp, vp,
                                         m_none=m_none, v_none=m_none)
    flags = _cross_path_flags(
        rp["m"].astype(np.float64), ri["m"].astype(np.float64),
        rp["v"].astype(np.float64), ri["v"].astype(np.float64),
        rp["theta"].astype(np.float64), ri["theta"].astype(np.float64),
        d_p, rp["delta_ref64"], ri["delta_ref64"], obs_p, obs_i,
        TOLERANCE_PROPOSALS["state_cross_path"],
        delta_rtol=gate_cross)
    theta_spec = TOLERANCE_PROPOSALS["theta_cross_path"]
    tclose = (abs(float(rp["theta"][0]) - float(ri["theta"][0]))
              <= theta_spec["atol"] + theta_spec["rtol"]
              * abs(float(ri["theta"][0])))
    flags["theta_close"] = tclose
    flags["theta_ok"] = bool(flags["theta_ok"] and tclose)
    iv = propagate_intervals(np.array([g_i]), np.array([theta]),
                             np.array([m]), np.array([v]), step,
                             m_none=m_none, v_none=m_none)
    inside = {
        "m": _inside(iv["K_m"], rp["m"]) and _inside(iv["K_m"], ri["m"]),
        "v": _inside(iv["K_v"], rp["v"]) and _inside(iv["K_v"], ri["v"]),
        "theta": _inside(iv["K_theta"], rp["theta"])
                 and _inside(iv["K_theta"], ri["theta"]),
        "delta": _inside(iv["K_delta"], d_p) and _inside(iv["K_delta"], d_i),
    }
    verdict = _classify_adamw(flags, not (p_formula or i_formula), inside,
                              zero_problems=zero_problems)
    p_nz = rp["delta_ref64"] != 0.0
    i_nz = ri["delta_ref64"] != 0.0
    p_bad = (~ok_p & obs_p) | ((~p_nz) & (d_p != 0.0))
    i_bad = (~ok_i & obs_i) | ((~i_nz) & (d_i != 0.0))
    if p_bad.any() or i_bad.any():
        verdict["failures"].append("same_input_delta")
        verdict["classification"] = "direct_failure"
    if (p_nz & ~obs_p).any() or (i_nz & ~obs_i).any():
        if "parameter_delta_precision_gap" not in verdict["gaps"]:
            verdict["gaps"].append("parameter_delta_precision_gap")
        if verdict["classification"] == "passed":
            verdict["classification"] = "audit_incomplete"
    theta_bound = (float(theta_spec["atol"])
                   + float(theta_spec["rtol"]) * abs(float(ri["theta"][0])))
    max_theta_ratio = abs(float(rp["theta"][0]) - float(ri["theta"][0])) \
        / theta_bound
    state_ratios = []
    state_applicable = 0
    state_unbounded = False
    for a, b in ((rp["m"], ri["m"]), (rp["v"], ri["v"])):
        ratio, count = _state_cross_path_ratio(
            a, b, TOLERANCE_PROPOSALS["state_cross_path"])
        state_applicable += count
        if not math.isfinite(ratio):
            state_unbounded = True
        else:
            state_ratios.append(ratio)
    max_state_ratio = (None if state_unbounded
                       else max(state_ratios, default=0.0))
    mask = bool(obs_p[0]) and bool(obs_i[0])
    max_delta_ratio = (abs(float(d_p[0]) - float(ri["delta_ref64"][0]))
                       / (gate_cross
                          * abs(float(ri["delta_ref64"][0])))) if mask else 0.0
    same_ratios = []
    for obs, ref, measured in ((obs_p, rp["delta_ref64"], d_p),
                               (obs_i, ri["delta_ref64"], d_i)):
        if bool(obs[0]):
            same_ratios.append(abs(float(measured[0]) - float(ref[0]))
                               / (gate_same * abs(float(ref[0]))))
    max_same_delta_ratio = max(same_ratios, default=0.0)
    return {
        "classification": verdict["classification"],
        "failures": verdict["failures"],
        "gaps": verdict["gaps"],
        "max_theta_ratio": max_theta_ratio,
        "max_state_ratio": max_state_ratio,
        "state_ratio_applicable": state_applicable,
        "state_ratio_unbounded": state_unbounded,
        "max_delta_ratio": max_delta_ratio,
        "max_same_delta_ratio": max_same_delta_ratio,
        "m_p": float(rp["m"][0]), "m_i": float(ri["m"][0]),
        "v_p": float(vp[0]), "v_i": float(vi[0]),
        "theta_p": float(rp["theta"][0]), "theta_i": float(ri["theta"][0]),
        "delta_ref_p": float(rp["delta_ref64"][0]),
        "delta_ref_i": float(ri["delta_ref64"][0]),
        "delta_measured_p": float(d_p[0]), "delta_measured_i": float(d_i[0]),
        "bound_delta_p": float(b_p[0]), "bound_delta_i": float(b_i[0]),
        "observable_p": bool(obs_p[0]), "observable_i": bool(obs_i[0]),
        "inside": inside, "flags": flags, "p_formula": p_formula,
        "i_formula": i_formula, "zero_problems": zero_problems,
        "interval_recompute_problems": _interval_recompute_problems(
            np.array([g_i]), np.array([theta]), np.array([m]),
            np.array([v]), step, iv, m_none=m_none, v_none=m_none),
    }


def _interval_recompute_problems(g: np.ndarray, theta: np.ndarray,
                                 m: np.ndarray, v: np.ndarray, step: int,
                                 supplied: Mapping[str, Any], *,
                                 m_none: bool = False,
                                 v_none: bool = False) -> list[str]:
    fresh = propagate_intervals(g, theta, m, v, step, m_none=m_none,
                                v_none=v_none)
    problems: list[str] = []
    for key in ("K_m", "K_v", "K_theta", "K_delta"):
        lo, hi = supplied.get(key, (None, None))
        if lo is None or hi is None:
            problems.append(f"{key}_missing")
            continue
        if not (np.array_equal(np.asarray(lo), fresh[key][0])
                and np.array_equal(np.asarray(hi), fresh[key][1])):
            problems.append(f"{key}_not_reproducible")
    return problems


def _row_ratio_observations(row: Mapping[str, Any]) -> dict[str, float]:
    """Per-gate applicable observations from one ``adamw:`` row.

    A row classified ``audit_incomplete`` only because the update delta is
    below the float32 write resolution still ran and passed its m/v
    comparison, so its state ratio is a valid sensitivity observation.
    Filtering by the whole-row classification hides exactly those rows.  A
    gate that failed is not a positive observation, and a direct failure is
    not a numeric-sensitivity sample.
    """
    if row.get("status") == "none_parameter" or row.get("failure"):
        return {}
    flags = row.get("cross_path_flags") or {}
    out: dict[str, float] = {}
    if flags.get("theta_ok") and row.get("max_theta_ratio") is not None:
        out["theta_cross_path"] = float(row["max_theta_ratio"])
    if (int(row.get("state_ratio_applicable") or 0) > 0
            and flags.get("m_ok") and flags.get("v_ok")):
        value = row.get("max_state_ratio")
        out["state_cross_path"] = (math.inf if value is None
                                   else float(value))
    if flags.get("delta_ok") and row.get("max_delta_ratio") is not None:
        out["delta_cross_path"] = float(row["max_delta_ratio"])
    if (row.get("same_input_formula_passed")
            and row.get("max_same_delta_ratio") is not None):
        out["delta_same_input"] = float(row["max_same_delta_ratio"])
    return out


def _observed_from_chain(seed: int, device: str) -> dict:
    config = dict(MODEL_CONFIG)
    config.update(d_model=8, n_layers=1, n_heads=2, ffn_mult=2, time_window=3)
    provider = SyntheticTrainingProvider(config, seed=seed, device=device)
    ratios = {key: 0.0 for key in TOLERANCE_PROPOSALS}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="w1_update_cal_") as tmp:
        root = Path(tmp)
        provider.prepare_fixture(root / "fixture")
        provider.load_fixture()
        roster = provider.roster
        counter = CallCounter(CALL_CAPS)
        budget = {"remaining_s": 10 ** 6, "reserve_s": 0.0}
        p = run_p_stage(root, "P", "cal", _now(), budget, provider, counter,
                        roster, out_dir=root / "P")
        tolerance = {"tolerances": TOLERANCE_PROPOSALS}
        i = run_i_stage(root, "I", "cal", _now(), budget, provider, counter,
                        roster, root / "P", tolerance=tolerance,
                        out_dir=root / "I")
        # alpha ratio is in the P report, not the rows
        p_report = read_json(root / "P" / "p_report.json")
        for model in MODELS:
            for u in UPDATES:
                for micro in p_report["models"][model]["updates"][f"u{u}"]["micros"]:
                    ratios["alpha_x_t"] = max(
                        ratios["alpha_x_t"], float(micro.get("x_t_max_ratio", 0.0)))
        for row in i["rows"]:
            key = row["key"]
            if key.startswith("objective:") and row.get("max_ratio") is not None:
                ratios["loss_scalar"] = max(ratios["loss_scalar"],
                                            float(row["max_ratio"]))
            if key.startswith("cotangent:") and row.get("max_ratio") is not None:
                ratios["output_cotangent"] = max(
                    ratios["output_cotangent"], float(row["max_ratio"]),
                    float(row.get("group_max_ratio") or 0.0))
            if key.startswith("micro_param:"):
                cmp = row.get("comparison") or {}
                if cmp.get("max_ratio") is not None:
                    ratios["parameter_gradient"] = max(
                        ratios["parameter_gradient"], float(cmp["max_ratio"]))
            if key.startswith("accumulated:"):
                for tag in ("sum_comparison", "cross_comparison"):
                    cmp = row.get(tag) or {}
                    if cmp.get("max_ratio") is not None:
                        ratios["parameter_gradient"] = max(
                            ratios["parameter_gradient"],
                            float(cmp["max_ratio"]))
            if key.startswith("clipped:"):
                for tag in ("self_comparison", "cross_comparison"):
                    cmp = row.get(tag) or {}
                    if cmp.get("max_ratio") is not None:
                        ratios["parameter_gradient"] = max(
                            ratios["parameter_gradient"],
                            float(cmp["max_ratio"]))
            if key.startswith("update_global:"):
                for tag in ("norm_comparison", "coef_comparison"):
                    cmp = row.get(tag) or {}
                    if cmp.get("max_ratio") is not None:
                        ratios["global_norm_clip"] = max(
                            ratios["global_norm_clip"], float(cmp["max_ratio"]))
            if key.startswith("adamw:"):
                for name, value in _row_ratio_observations(row).items():
                    ratios[name] = max(ratios[name], value)
    return {key: _ratio_record(value, source="synthetic_chain")
            for key, value in ratios.items()}


def synthetic_calibration(*, seed: int = 20261029,
                          device: str = "cuda") -> dict:
    if device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("synthetic calibration requires the cuda dispatch")
    config = dict(MODEL_CONFIG)
    config.update(d_model=8, n_layers=1, n_heads=2, ffn_mult=2, time_window=3)
    observed = _observed_from_chain(seed, device)
    fp32 = build_fp32_contract(device)
    fp32_for_hash = {k: v for k, v in fp32.items() if k != "created_wall"}
    fp32_sha = _canonical_sha256(fp32_for_hash)
    scalar_plan = _scalar_case(0.0, 0.001, 1.0000000116860974e-7, 2,
                               -0.008999999612569809,
                               -0.008999998681247234, m_none=False)
    scalar_stable = _scalar_case(0.0, 0.002, 3.3e-7, 2, 0.5, 0.5,
                                 m_none=False)
    scalar_near = _scalar_case(0.0, 0.002, 3.3e-7, 2, 0.5,
                               0.5 * (1 + 1e-5), m_none=False)
    for key, tag in (("theta_cross_path", "max_theta_ratio"),
                     ("state_cross_path", "max_state_ratio"),
                     ("delta_cross_path", "max_delta_ratio"),
                     ("delta_same_input", "max_same_delta_ratio")):
        observed[key] = _merge_ratio_records([
            observed.get(key),
            _ratio_record(float(scalar_near[tag]), source="scalar_near_1e-7"),
        ])
    for key, rec in observed.items():
        ratio = float(rec["max_ratio_of_proposal"])
        if not math.isfinite(ratio) or ratio < 0.0 or 4.0 * ratio > 1.0:
            raise ValueError(
                f"calibration {key} ratio {ratio:.6g} violates 4x rule")
    rp_tamper = adamw_effective_fp32(np.array([1.0]),
                                     np.array([0.002]),
                                     np.array([3.3e-7]),
                                     np.array([0.0123]), 2)
    post_tamper = np.nextafter(rp_tamper["theta"], np.float32(np.inf))
    tamper_case = _scalar_case(1.0, 0.002, 3.3e-7, 2, 0.0123, 0.0123,
                               m_none=False,
                               post_p=float(post_tamper[0]))
    twenty = float(rp_tamper["theta"][0] + 0.8 * (
        float(rp_tamper["theta"][0]) - 1.0))
    twenty_case = _scalar_case(1.0, 0.002, 3.3e-7, 2, 0.0123, 0.0123,
                               m_none=False, post_p=twenty)
    local_zero = _scalar_case(0.0, 0.0, 0.0, 1, 1e-17, 1e-17,
                              m_none=True, v_post_p=0.0)
    small = _scalar_case(1.0, 1e-3, 1.0, 2, 1e-3, 1e-3, m_none=False)
    # interval tamper: a narrowed copy must be rejected by raw recomputation
    g_small = np.array([1e-3])
    iv = propagate_intervals(g_small, np.array([1.0]), np.array([1e-3]),
                             np.array([1.0]), 2)
    bad_iv = {"K_m": (iv["K_m"][0] * 0.5, iv["K_m"][1] * 0.5),
              "K_v": iv["K_v"], "K_theta": iv["K_theta"],
              "K_delta": iv["K_delta"]}
    interval_tamper_problems = _interval_recompute_problems(
        g_small, np.array([1.0]), np.array([1e-3]), np.array([1.0]), 2,
        bad_iv)
    evidence = {
        "schema": CALIBRATION_SCHEMA,
        "seed": int(seed),
        "device": device,
        "config": config,
        "observed": observed,
        "fp32_contract_sha256": fp32_sha,
        "fp32_valid": bool(fp32.get("valid")),
        "fp32_probe_cases": fp32.get("probe_cases"),
        "fp32_probe_mismatches": fp32.get("probe_mismatches"),
        "fp32_two_step_match": fp32.get("two_step_match"),
        "fp32_source_sha256": fp32.get("source_sha256"),
        "cases": {
            "plan_1ulp": scalar_plan,
            "stable_pass": scalar_stable,
            "post_state_tamper": tamper_case,
            "update_minus_20_percent": twenty_case,
            "local_state_zero": local_zero,
            "small_unobservable_update": small,
            "interval_tamper_problems": interval_tamper_problems,
        },
        "expected": {
            "plan_1ulp": "cross_path_sensitivity_gap",
            "stable_pass": "passed",
            "post_state_tamper": "direct_failure",
            "update_minus_20_percent": "direct_failure",
            "local_state_zero": "direct_failure",
            "small_unobservable_update": "audit_incomplete",
            "interval_tamper_rejected": bool(interval_tamper_problems),
        },
        "tiny_zero_floors": {"gradient_abs": ZERO_GRADIENT_ABS,
                             "gradient_noise_scale": ZERO_GRADIENT_NOISE_SCALE,
                             "state_abs": ZERO_STATE_ABS},
        "real_dataset_loaded": False,
        "real_checkpoint_loaded": False,
        "real_asset_loaded": False,
        "real_model_calls": 0,
        "boundaries": dict(BOUNDARIES),
    }
    for key, want in (("plan_1ulp", "cross_path_sensitivity_gap"),
                      ("stable_pass", "passed"),
                      ("post_state_tamper", "direct_failure"),
                      ("update_minus_20_percent", "direct_failure"),
                      ("local_state_zero", "direct_failure"),
                      ("small_unobservable_update", "audit_incomplete")):
        if evidence["cases"][key]["classification"] != want:
            raise ValueError(
                f"calibration case {key}: "
                f"{evidence['cases'][key]['classification']} != {want}")
    if not interval_tamper_problems:
        raise ValueError("calibration interval tamper was not rejected")
    evidence_sha = _canonical_sha256(evidence)
    contract = build_tolerance_contract(
        observed=observed, calibration=evidence,
        evidence_sha256=evidence_sha, fp32_contract=fp32_for_hash,
        fp32_contract_sha256=fp32_sha)
    return contract


def _deep_copy_proposals() -> dict:
    import copy

    return copy.deepcopy(TOLERANCE_PROPOSALS)


def build_tolerance_contract(*, observed: Mapping[str, Any] | None = None,
                             calibration: Mapping[str, Any] | None = None,
                             evidence_sha256: str | None = None,
                             fp32_contract: Mapping[str, Any] | None = None,
                             fp32_contract_sha256: str | None = None) -> dict:
    tolerances = _deep_copy_proposals()
    if observed is not None:
        if set(observed) != set(TOLERANCE_PROPOSALS):
            raise ValueError("calibration observed an unexpected key set")
        for key, rec in observed.items():
            ratio = float(rec.get("max_ratio_of_proposal", math.inf))
            if not math.isfinite(ratio) or ratio < 0.0:
                raise ValueError(f"{key}: invalid calibration ratio")
            if 4.0 * ratio > 1.0:
                raise ValueError(
                    f"{key}: 4x synthetic error {ratio:.3g} of proposal "
                    "exceeds the upper bound")
    return {
        "schema": TOLERANCE_SCHEMA,
        "audit_schema": AUDIT_SCHEMA,
        "mode": ("synthetic_calibrated" if calibration is not None
                 else "synthetic_default"),
        "calibrated": calibration is not None,
        "synthetic_only": True,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evidence_sha256": evidence_sha256,
        "tolerances": tolerances,
        "rules": dict(TOLERANCE_RULES),
        "tiny_zero_floors": {"gradient_abs": ZERO_GRADIENT_ABS,
                             "gradient_noise_scale": ZERO_GRADIENT_NOISE_SCALE,
                             "state_abs": ZERO_STATE_ABS},
        "calibration": dict(calibration) if calibration is not None else None,
        "calibration_sha256": evidence_sha256,
        "fp32_contract": dict(fp32_contract) if fp32_contract is not None else None,
        "fp32_contract_sha256": fp32_contract_sha256,
    }


def build_default_tolerance_contract() -> dict:
    return build_tolerance_contract()


def tolerance_contract_problems(contract: Mapping[str, Any] | None,
                                *, recompute: bool = True) -> list[str]:
    problems: list[str] = []
    if not isinstance(contract, Mapping):
        return ["tolerance_contract_not_a_mapping"]
    if contract.get("schema") != TOLERANCE_SCHEMA:
        problems.append("tolerance_contract_schema")
    if contract.get("audit_schema") != AUDIT_SCHEMA:
        problems.append("tolerance_contract_audit_schema")
    if contract.get("synthetic_only") is not True:
        problems.append("tolerance_contract_not_synthetic")
    mode = contract.get("mode")
    if mode not in ("synthetic_default", "synthetic_calibrated"):
        problems.append("tolerance_contract_mode")
    tol = contract.get("tolerances")
    if not isinstance(tol, Mapping) or set(tol) != set(TOLERANCE_PROPOSALS):
        return problems + ["tolerance_contract_keys"]
    for key, want in TOLERANCE_PROPOSALS.items():
        rec = tol.get(key)
        if not isinstance(rec, Mapping):
            problems.append(f"tolerance_record:{key}")
            continue
        if (not _num(rec.get("atol")) or not _num(rec.get("rtol"))
                or float(rec["atol"]) != float(want["atol"])
                or float(rec["rtol"]) != float(want["rtol"])):
            problems.append(f"tolerance_widened_or_changed:{key}")
    if _jsonable(contract.get("rules")) != _jsonable(TOLERANCE_RULES):
        problems.append("tolerance_rules_changed")
    if _jsonable(contract.get("tiny_zero_floors")) != _jsonable(
            {"gradient_abs": ZERO_GRADIENT_ABS,
             "gradient_noise_scale": ZERO_GRADIENT_NOISE_SCALE,
             "state_abs": ZERO_STATE_ABS}):
        problems.append("tolerance_tiny_floors_changed")
    if problems:
        return problems
    if mode == "synthetic_default":
        if contract.get("evidence_sha256") is not None:
            problems.append("default_contract_has_evidence")
        if contract.get("calibration") not in (None, {}):
            problems.append("default_contract_has_calibration")
        return problems
    evidence_sha = contract.get("evidence_sha256")
    if not isinstance(evidence_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", evidence_sha):
        problems.append("calibration_evidence_hash_missing")
    calibration = contract.get("calibration")
    if not isinstance(calibration, Mapping):
        return problems + ["calibration_evidence_missing"]
    if contract.get("calibration_sha256") != evidence_sha:
        problems.append("calibration_sha_mismatch")
    if _canonical_sha256(calibration) != evidence_sha:
        problems.append("calibration_evidence_hash_mismatch")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        problems.append("calibration_schema")
    for flag in ("real_dataset_loaded", "real_checkpoint_loaded",
                 "real_asset_loaded"):
        if calibration.get(flag) is not False:
            problems.append(f"calibration_{flag}")
    if calibration.get("real_model_calls") != 0:
        problems.append("calibration_real_model_calls")
    observed = calibration.get("observed")
    if not isinstance(observed, Mapping) or set(observed) != set(
            TOLERANCE_PROPOSALS):
        problems.append("calibration_observed_keys")
    else:
        for key, rec in observed.items():
            ratio = rec.get("max_ratio_of_proposal") if isinstance(
                rec, Mapping) else None
            if not _num(ratio) or float(ratio) < 0.0 or 4.0 * float(ratio) > 1.0:
                problems.append(f"calibration_margin:{key}")
    cases = calibration.get("cases")
    expected = calibration.get("expected")
    if not isinstance(cases, Mapping) or not isinstance(expected, Mapping):
        problems.append("calibration_cases")
    else:
        for key in ("plan_1ulp", "stable_pass", "post_state_tamper",
                    "update_minus_20_percent", "local_state_zero",
                    "small_unobservable_update"):
            rec = cases.get(key)
            if not isinstance(rec, Mapping) or rec.get("classification") != \
                    expected.get(key):
                problems.append(f"calibration_case:{key}")
        if not cases.get("interval_tamper_problems"):
            problems.append("calibration_interval_tamper")
    fp32 = contract.get("fp32_contract")
    problems += [f"fp32:{p}" for p in fp32_contract_problems(
        fp32, recompute=False)]
    if contract.get("fp32_contract_sha256") != _canonical_sha256(fp32) \
            if isinstance(fp32, Mapping) else True:
        problems.append("fp32_contract_sha")
    seed = calibration.get("seed")
    if not _count(seed):
        problems.append("calibration_seed_invalid")
    elif recompute:
        try:
            fresh = synthetic_calibration(seed=int(seed),
                                          device=str(calibration.get("device")))
        except Exception as exc:  # noqa: BLE001 -- evidence must reproduce
            problems.append(f"calibration_recompute_failed:{type(exc).__name__}")
        else:
            if fresh.get("evidence_sha256") != evidence_sha:
                problems.append("calibration_evidence_not_reproducible")
            if _jsonable(fresh.get("calibration")) != _jsonable(calibration):
                problems.append("calibration_values_not_reproducible")
    return problems


def load_tolerance_contract(path: Path,
                            *, recompute: bool = True
                            ) -> tuple[dict, str, list[str]]:
    p = Path(path)
    if not p.is_file():
        return {}, "", ["tolerance_contract_missing"]
    try:
        contract = read_json(p)
    except (OSError, ValueError):
        return {}, sha256_file(p), ["tolerance_contract_unreadable"]
    return contract, sha256_file(p), tolerance_contract_problems(
        contract, recompute=recompute)

# ---------------------------------------------------------------------------
# authorization and admission
# ---------------------------------------------------------------------------

AUTHORIZATION_REQUIRED = (
    "new_question_cohort_and_parents",
    "fresh_adamw_and_diagnostic_steps",
    "two_continuous_updates_and_common_prestate",
    "real_network_objective_parameter_backward_fk_clip_adamw",
    "call_matrix_and_unsettled_unknown",
    "budgets_attempts_and_resources",
    "synthetic_acceptance_fp32_zeros_propagation",
    "zero_fd_ddim_selection_checkpoint_contact",
    "publishing_three_way_and_withdrawal",
    "partial_coverage_not_ready_no_auto",
)
RESOURCE_CAPS = {"gpu_peak_gib": 22, "host_peak_gib": 32,
                 "artifacts_gib": 16, "disk_free_min_gib": 20}
EXPECTED_UPDATES = {"diagnostic_parameter_update_events": 16,
                    "cohort_hard_cap": 32}


def authorization_template() -> dict:
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "template_only": True,
        "authorized": False,
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "granted_by": None,
        "granted_wall": None,
        **{key: 0 for key in ZERO_DECLARATIONS},
        "checklist": {key: None for key in AUTHORIZATION_REQUIRED},
        "budgets": {stage: {"budget_s": spec["budget_s"],
                            "attempts": spec["attempts"]}
                    for stage, spec in STAGE_BUDGETS.items()},
        "call_caps": dict(CALL_CAPS),
        "updates": dict(EXPECTED_UPDATES),
        "resource_caps": dict(RESOURCE_CAPS),
        "parent_registration_sha256": PARENT_FILES[
            "objective_registration"][1],
        "draft_sha256": None,
        "tolerance_evidence_sha256": None,
        "fp32_contract_sha256": None,
        "admission_package_sha256": None,
        "implementation_files": None,
        "notes": ("Template only. The user must create a separate record with "
                  "authorized=true, every checklist item true, the exact "
                  "frozen budgets/caps/updates, JSON integer zero declarations, "
                  "and the bound parent/draft/tolerance/fp32/admission/"
                  "implementation hashes."),
    }


def authorization_problems(record: Mapping[str, Any] | None) -> list[str]:
    problems: list[str] = []
    if not isinstance(record, Mapping):
        return ["authorization_not_a_mapping"]
    if record.get("schema") != AUTHORIZATION_SCHEMA:
        problems.append("authorization_schema")
    if record.get("authorized") is not True:
        problems.append("not_authorized")
    if record.get("cohort_id") != COHORT_ID:
        problems.append("authorization_cohort")
    if record.get("execution_id") != EXECUTION_ID:
        problems.append("authorization_execution")
    for key in ZERO_DECLARATIONS:
        if type(record.get(key)) is not int or record.get(key) != 0:
            problems.append(f"authorization_zero:{key}")
    checklist = record.get("checklist")
    if not isinstance(checklist, Mapping):
        problems.append("authorization_checklist_missing")
    else:
        for key in AUTHORIZATION_REQUIRED:
            if checklist.get(key) is not True:
                problems.append(f"authorization_checklist:{key}")
    budgets = record.get("budgets")
    if not isinstance(budgets, Mapping) or set(budgets) != set(STAGE_BUDGETS):
        problems.append("authorization_budget_stages")
    else:
        for stage, spec in STAGE_BUDGETS.items():
            rec = budgets.get(stage)
            if not isinstance(rec, Mapping):
                problems.append(f"authorization_budget:{stage}")
                continue
            if not _num(rec.get("budget_s")) or float(rec["budget_s"]) != \
                    spec["budget_s"]:
                problems.append(f"authorization_budget_size:{stage}")
            if type(rec.get("attempts")) is not int or rec["attempts"] != \
                    spec["attempts"]:
                problems.append(f"authorization_attempts:{stage}")
    caps = record.get("call_caps")
    if not isinstance(caps, Mapping) or set(caps) != set(CALL_CAPS):
        problems.append("authorization_call_caps")
    else:
        for key, want in CALL_CAPS.items():
            if type(caps.get(key)) is not int or caps[key] != int(want):
                problems.append(f"authorization_call_cap:{key}")
    updates = record.get("updates")
    if not isinstance(updates, Mapping) or \
            updates.get("diagnostic_parameter_update_events") != 16 or \
            updates.get("cohort_hard_cap") != 32:
        problems.append("authorization_updates")
    if _jsonable(record.get("resource_caps")) != _jsonable(RESOURCE_CAPS):
        problems.append("authorization_resources")
    if record.get("parent_registration_sha256") != PARENT_FILES[
            "objective_registration"][1]:
        problems.append("authorization_parent_registration")
    for key in ("draft_sha256", "tolerance_evidence_sha256",
                "fp32_contract_sha256", "admission_package_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                          value):
            problems.append(f"authorization_binding:{key}")
    impl = record.get("implementation_files")
    if not isinstance(impl, Mapping) or not impl:
        problems.append("authorization_implementation")
    return problems


def load_authorization(path: Path) -> tuple[dict, str, list[str]]:
    p = Path(path)
    if not p.is_file():
        return {}, "", ["authorization_missing"]
    try:
        record = read_json(p)
    except (OSError, ValueError):
        return {}, sha256_file(p), ["authorization_unreadable"]
    return record, sha256_file(p), authorization_problems(record)


def admission_package_problems(package: Mapping[str, Any] | None) -> list[str]:
    problems: list[str] = []
    if not isinstance(package, Mapping):
        return ["admission_not_a_mapping"]
    if package.get("schema") != ADMISSION_SCHEMA:
        problems.append("admission_schema")
    if package.get("passed") is not True:
        problems.append("admission_not_passed")
    if package.get("problems"):
        problems.append("admission_has_problems")
    status = package.get("status")
    if not isinstance(status, Mapping):
        problems.append("admission_status_missing")
    else:
        if status.get("authorized") is not False:
            problems.append("admission_must_be_pre_authorization")
        if status.get("registered") is not False:
            problems.append("admission_must_not_be_registered")
        if status.get("run") != "not_run":
            problems.append("admission_must_not_be_run")
        if status.get("section_10_2_checked") is not False:
            problems.append("admission_section_10_2")
    scope = package.get("read_scope")
    if not isinstance(scope, Mapping):
        problems.append("admission_read_scope_missing")
    else:
        for key in ("window_values_loaded", "checkpoint_torch_load",
                    "cohort_created", "registered_attempt_created",
                    "asset_closure_verified"):
            if scope.get(key) is not False:
                problems.append(f"admission_read_scope:{key}")
        for key in ("conditioner_forward_calls", "model_forward_calls",
                    "vjp_calls", "optimizer_steps"):
            if scope.get(key) != 0:
                problems.append(f"admission_read_scope:{key}")
    for name, rec in (package.get("parent_anchors") or {}).items():
        if not isinstance(rec, Mapping) or rec.get("match") is not True:
            problems.append(f"admission_parent_anchor:{name}")
    roster = package.get("roster") or {}
    if roster.get("row_formula") != "14N+168":
        problems.append("admission_roster_formula")
    if not isinstance(roster.get("sha256"), str):
        problems.append("admission_roster_sha")
    tolerance = package.get("tolerance") or {}
    if tolerance.get("mode") != "synthetic_calibrated" or not \
            tolerance.get("evidence_sha256"):
        problems.append("admission_tolerance_not_calibrated")
    tolerance_contract = tolerance.get("contract") or {}
    fp32 = ((tolerance_contract.get("fp32_contract")
             if isinstance(tolerance_contract, Mapping) else None)
            or tolerance.get("fp32_contract"))
    problems += [f"admission_fp32:{p}" for p in fp32_contract_problems(
        fp32, recompute=False)]
    fp32_sha = (tolerance_contract.get("fp32_contract_sha256")
                if isinstance(tolerance_contract, Mapping) else None)
    if (fp32_sha or tolerance.get("fp32_contract_sha256")) != \
            package.get("fp32_contract_sha256"):
        problems.append("admission_fp32_binding")
    impl = package.get("implementation") or {}
    for rel, sha in (impl.get("files") or {}).items():
        if not isinstance(sha, str):
            problems.append(f"admission_implementation_missing:{rel}")
    runtime = package.get("runtime")
    if not isinstance(runtime, Mapping):
        problems.append("admission_runtime_missing")
    else:
        if runtime.get("applied") is not True:
            problems.append("admission_runtime_not_applied")
        if runtime.get("observed_after_setup") != \
                runtime.get("requirements"):
            problems.append("admission_runtime_observation")
    return problems


def build_admission_package(project_root: Path,
                            tolerance_contract: Mapping[str, Any],
                            *, tolerance_contract_path: Path | None = None,
                            tolerance_contract_sha256: str | None = None
                            ) -> dict:
    root = Path(project_root)
    contract_problems = tolerance_contract_problems(tolerance_contract,
                                                    recompute=True)
    if contract_problems:
        raise ValueError(f"tolerance contract refused: {contract_problems}")
    if tolerance_contract.get("mode") != "synthetic_calibrated":
        raise ValueError("admission requires a completed calibration")
    problems: list[str] = []
    runtime = configure_production_runtime()
    problems += [f"runtime_apply:{p}" for p in runtime["problems"]]
    environment = environment_facts()
    parents = {}
    for name, (rel, want) in PARENT_FILES.items():
        p = root / rel
        actual = sha256_file(p) if p.is_file() else None
        parents[name] = {"path": path_for_record(p), "expected": want,
                         "actual": actual,
                         "match": actual is not None and actual == want}
        if actual != want:
            problems.append(f"parent_drifted:{name}")
    checkpoints = {}
    for model, (rel, want) in CHECKPOINTS.items():
        p = root / rel
        actual = sha256_file(p) if p.is_file() else None
        checkpoints[model] = {"path": path_for_record(p), "expected": want,
                              "actual": actual,
                              "match": actual is not None and actual == want,
                              "step": 4000}
        if actual != want:
            problems.append(f"checkpoint_drifted:{model}")
    config_path = root / TRAIN_CONFIG_PATH
    config_sha = sha256_file(config_path) if config_path.is_file() else None
    if config_sha != TRAIN_CONFIG_SHA:
        problems.append("train_config_sha")
    roster = build_parameter_roster(MODEL_CONFIG)
    catalog = row_catalog(roster)
    if catalog["n_rows"] != 14 * int(roster["n_total"]) + 168:
        problems.append("row_formula")
    bound = _resource_bound(roster)
    if not bound["within_cap"]:
        problems.append("artifact_bound")
    impl = implementation_identity(root)
    missing = [rel for rel, sha in impl["files"].items() if not isinstance(
        sha, str)]
    if missing:
        problems.append(f"implementation_missing:{missing}")
    fp32_sha = tolerance_contract.get("fp32_contract_sha256")
    if fp32_contract_problems(tolerance_contract.get("fp32_contract"),
                              recompute=False):
        problems.append("fp32_contract")
    full_chain = dict(CHAIN_NOMINAL)
    full_chain["scene_setup_mj_forward"] = 0
    for kind, value in full_chain.items():
        if value > int(CALL_CAPS[kind]):
            problems.append(f"full_chain_call_cap:{kind}")
    notes = [
        "both demos are sub10; no cross-subject coverage is claimed",
        "synthetic fixture targets are a deterministic base prediction plus a "
        "registered perturbation; they are not real-batch evidence",
        "the AdamW contract is proven on the CUDA default foreach dispatch; "
        "a CPU-only host cannot authorize real execution",
        "mathematically-zero gradients use calibrated absolute floors "
        f"({ZERO_GRADIENT_ABS:g} gradient, {ZERO_STATE_ABS:g} state); the "
        "draft's 1e-30 proposal was not viable",
        "step-1 updates can be below the 5e-4 float32 write-resolution gate; "
        "those coordinates are honest parameter_delta_precision_gap entries",
    ]
    package = {
        "schema": ADMISSION_SCHEMA,
        "status": {
            "mode": "pre_authorization_readonly",
            "authorized": False, "registered": False, "run": "not_run",
            "package_passed": not problems, "section_10_2_checked": False,
        },
        "passed": not problems,
        "problems": problems,
        "cohort": {
            "cohort_id": COHORT_ID, "execution_id": EXECUTION_ID,
            "audit_schema": AUDIT_SCHEMA,
            "accounting_schema": ACCOUNTING_SCHEMA,
            "history_root": str(HISTORY_ROOT),
            "stage_budgets": {k: dict(v) for k, v in STAGE_BUDGETS.items()},
            "total_budget_s": TOTAL_BUDGET_S,
            "call_caps": dict(CALL_CAPS),
            "full_chain_calls": full_chain,
            "updates": dict(EXPECTED_UPDATES),
            "budget_transfer_from_old_runs": False,
        },
        "windows": [{"id": WINDOW_IDS[i], "demo": spec[0], "start": spec[1],
                     "end": spec[2], "T": spec[3]}
                    for i, spec in enumerate(WINDOW_SPECS)],
        "micro_schedule": {str(u): [list(x) for x in MICRO_SCHEDULE[u]]
                           for u in UPDATES},
        "noise": {"seed": NOISE_SEED, "protocol": NOISE_PROTOCOL,
                  "draw_units_per_path": DRAW_UNITS_PER_PATH,
                  "draw_units_full_chain": 48},
        "roster": {
            "sha256": roster["sha256"], "n_det": roster["n_det"],
            "n_diff": roster["n_diff"], "n_total": roster["n_total"],
            "n_rows": catalog["n_rows"], "row_formula": "14N+168",
            "row_keys_sha256": catalog["keys_sha256"],
            "parameter_roster": roster,
        },
        "schemas": {"rows": catalog},
        "tolerance": {
            "path": (path_for_record(tolerance_contract_path)
                     if tolerance_contract_path is not None else None),
            "sha256": tolerance_contract_sha256,
            "mode": tolerance_contract.get("mode"),
            "evidence_sha256": tolerance_contract.get("evidence_sha256"),
            "fp32_contract_sha256": fp32_sha,
            "contract": dict(tolerance_contract),
        },
        "fp32_contract_sha256": fp32_sha,
        "parent_anchors": parents,
        "checkpoints": checkpoints,
        "config": {"path": path_for_record(config_path), "sha256": config_sha,
                   "expected": TRAIN_CONFIG_SHA},
        "resource_bound": bound,
        "resource_caps": dict(RESOURCE_CAPS),
        "implementation": impl,
        "environment": environment,
        "runtime": {
            "requirements": RUNTIME_REQUIREMENTS,
            "observed_after_setup": dict(runtime["after"]),
            "applied": runtime["applied"],
            "problems": list(runtime["problems"]),
            "applied_by": [
                "scripts/audit_w1_real_training_update.py --real (before any "
                "provider construction)",
                "ireset.eval.w1_real_training_update_audit._worker_main",
            ],
            "verified_by": ["R0 runtime gate", "F parent/worker comparison"],
        },
        "coverage": {
            "objective_coverage": "partial",
            "audited": [
                "shared_update_unit", "full_parameter_roster",
                "two_continuous_updates", "micro_accumulation",
                "joint_global_clip", "effective_float32_adamw",
                "cross_path_input_propagation", "state_zero_semantics",
                "delta_observability", "f_fresh_process_replay",
                "call_counting_and_budget_sealing",
            ],
            "not_audited": [
                "real_checkpoint_network_updates",
                "real_provider_registration", "second_parameter_derivative",
                "parameter_fd", "ddim_sampling", "model_selection",
                "end_to_end_quality", "data_loader_epoch_equivalence",
                "production_integration",
            ],
        },
        "known_limitations": notes,
        "boundaries": dict(BOUNDARIES),
        "authorization": {
            "required": True, "authorized": False,
            "required_fields": list(AUTHORIZATION_REQUIRED),
            "zero_declarations": list(ZERO_DECLARATIONS),
            "budgets_expected": {
                stage: {"budget_s": spec["budget_s"],
                        "attempts": spec["attempts"]}
                for stage, spec in STAGE_BUDGETS.items()},
            "template": authorization_template(),
        },
        "read_scope": {
            "metadata_only": True,
            "window_values_loaded": False,
            "checkpoint_torch_load": False,
            "asset_closure_verified": False,
            "cohort_created": False,
            "registered_attempt_created": False,
            "conditioner_forward_calls": 0,
            "model_forward_calls": 0,
            "vjp_calls": 0,
            "optimizer_steps": 0,
            "model_constructor_calls": 2,
            "note": ("file hashes, frozen JSON metadata and CPU random-init "
                     "roster construction only; no array, asset, cache or "
                     "checkpoint was opened"),
        },
        "r0_obligations": {
            "parent_consumers": ("verify objective/FK parent disk rows and "
                                 "consumer status, not only file hashes"),
            "real_inputs": ("strict-load two checkpoints, bind parent saved "
                            "x0/conditions, and re-check the full roster"),
            "asset_closure": ("real mode must hash locked G1/largebox assets "
                              "before FK work"),
            "call_accounting": ("persist stage reservations before work; an "
                                "unsettled worker leaves history unknown"),
            "f_process": ("fresh spawned Python subprocess with pid/ppid, "
                          "disk re-read and full replay"),
            "runtime": ("TF32 off / matmul highest / autocast off applied by "
                        "the real entry and the F worker"),
        },
        "notes": notes,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return package


def write_package(out_root: Path, package: Mapping[str, Any]
                  ) -> tuple[Path, str]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "index.json"
    rows = []
    if index_path.is_file():
        try:
            data = read_json(index_path)
            rows = [r for r in (data.get("attempts") or [])
                    if isinstance(r, dict)]
        except (OSError, ValueError, TypeError):
            rows = []
    numbers = [int(r["attempt"].split("_")[1]) for r in rows
               if isinstance(r.get("attempt"), str)
               and re.fullmatch(r"attempt_\d+", r["attempt"])]
    attempt = f"attempt_{max(numbers, default=0) + 1:03d}"
    directory = out_root / attempt
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "admission_package.json"
    write_json(path, package)
    sha = sha256_file(path)
    rows.append({"attempt": attempt, "path": path_for_record(path),
                 "sha256": sha, "passed": bool(package.get("passed"))})
    write_json(index_path, {"attempts": rows})
    return path, sha

# ---------------------------------------------------------------------------
# registered R0 -> P -> I -> F orchestration
# ---------------------------------------------------------------------------


def _stage_scope(provider: SyntheticTrainingProvider,
                 roster: Mapping[str, Any]) -> dict:
    return {
        "windows": [{"id": WINDOW_IDS[i], "demo": spec[0], "start": spec[1],
                     "end": spec[2], "T": spec[3]}
                    for i, spec in enumerate(WINDOW_SPECS)],
        "models": list(MODELS), "updates": list(UPDATES),
        "micro_schedule": {str(u): [list(x) for x in MICRO_SCHEDULE[u]]
                           for u in UPDATES},
        "noise_seed": NOISE_SEED, "noise_protocol": NOISE_PROTOCOL,
        "roster_sha256": roster["sha256"],
    }


def _stage_delta(counter: CallCounter, base: Mapping[str, int]) -> dict:
    return {"schema": counter.schema,
            "counts": {k: int(counter.counts[k] - int(base.get(k, 0)))
                       for k in CALL_KINDS},
            "within_caps": True}


def _artifact_hashes(paths: Iterable[Path | None]) -> dict:
    out = {}
    for path in paths:
        if path is None:
            continue
        p = Path(path)
        if p.is_file():
            out[path_for_record(p)] = sha256_file(p)
    return out


def run_registered_audit(run_root: Path, provider: SyntheticTrainingProvider,
                         *, mode: str, tolerance: Mapping[str, Any],
                         authorization_sha: str | None = None,
                         authorization: Mapping[str, Any] | None = None,
                         admission_package: Mapping[str, Any] | None = None,
                         admission_sha: str | None = None,
                         fixed_root: Path = HISTORY_ROOT,
                         project_root: Path = PROJECT_ROOT,
                         r0_started_perf: float | None = None) -> dict:
    if r0_started_perf is not None and (
            not _nonneg(r0_started_perf)
            or float(r0_started_perf) > time.perf_counter()):
        raise ValueError("invalid R0 launch clock")
    run_root = Path(run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if mode not in ("synthetic", "real"):
        raise ValueError(f"unknown mode {mode}")
    contract_problems = tolerance_contract_problems(tolerance,
                                                    recompute=False)
    if contract_problems:
        raise SystemExit(f"tolerance contract refused: {contract_problems}")
    prior = {kind: 0 for kind in CALL_KINDS}
    if mode == "real":
        base = (Path(project_root) / fixed_root).resolve()
        if run_root.parent != base:
            raise SystemExit(
                f"real run root must be a direct child of {base}, got {run_root}")
        if authorization_sha is None:
            raise SystemExit("real audit requires an authorization record")
        auth_problems = authorization_problems(authorization)
        if auth_problems:
            raise SystemExit(f"authorization refused: {auth_problems}")
        if tolerance.get("mode") != "synthetic_calibrated":
            raise SystemExit("real audit requires a completed calibration")
        if authorization.get("tolerance_evidence_sha256") != tolerance.get(
                "evidence_sha256"):
            raise SystemExit("authorization does not bind this calibration")
        if authorization.get("fp32_contract_sha256") != tolerance.get(
                "fp32_contract_sha256"):
            raise SystemExit("authorization does not bind this fp32 contract")
        if admission_package is None or admission_sha is None:
            raise SystemExit("real audit requires a passed admission package")
        pkg_problems = admission_package_problems(admission_package)
        if pkg_problems:
            raise SystemExit(f"admission package refused: {pkg_problems}")
        if authorization.get("admission_package_sha256") != admission_sha:
            raise SystemExit("authorization does not bind this admission package")
        current_impl = implementation_identity(project_root)
        if (admission_package.get("implementation") or {}).get(
                "files") != current_impl.get("files"):
            raise SystemExit("implementation drifted since the admission package")
        if authorization.get("implementation_files") != current_impl.get(
                "files"):
            raise SystemExit("authorization does not bind the implementation")
        if authorization.get("draft_sha256") != current_impl["files"].get(
                PREREG_DRAFT):
            raise SystemExit("authorization does not bind the current draft")
        prior = prior_call_counts(run_root, fixed_root=fixed_root, mode=mode)
        if prior is None:
            raise SystemExit("call history unknown: refusing a fresh quota")
        insufficient = {kind: {"prior": prior[kind],
                               "required": CHAIN_NOMINAL[kind],
                               "cap": CALL_CAPS[kind]}
                        for kind in CALL_KINDS
                        if prior[kind] + CHAIN_NOMINAL[kind] > CALL_CAPS[kind]}
        if insufficient:
            raise SystemExit(
                f"cumulative call budget insufficient: {insufficient}")
        raise SystemExit(
            "real_provider_not_registered_in_ic: the synthetic I/C delivery "
            "does not construct real models or read checkpoints")
    write_cohort(run_root, mode=mode, authorization_sha=authorization_sha)
    counter = CallCounter(CALL_CAPS, prior)
    binder = getattr(provider, "bind_counter", None)
    if callable(binder):
        binder(counter)
    fixture_seconds = None
    if mode == "synthetic":
        t_fx = _now()
        provider.prepare_fixture(run_root / "fixture")
        fixture_seconds = _now() - t_fx
        provider.load_fixture()
    roster = provider.roster
    scope = _stage_scope(provider, roster)
    results: dict[str, Any] = {}
    seals: dict[str, Any] = {}

    def stop(problems: Sequence[str], gaps: Sequence[str],
             failures: Sequence[str] = ()) -> dict:
        return {
            "mode": mode, "conclusion": "audit_incomplete",
            "complete_evidence": False, "covered_contract_passed": False,
            "problems": list(problems), "failures": list(failures),
            "gaps": list(gaps), "stages": results, "seals": seals,
            "fixture_build_s": fixture_seconds,
            "calls": counter.snapshot(),
            "training_updates": int(counter.counts["native_adamw_steps"]
                                    + counter.counts["explicit_reference_updates"]),
            "zero_counters": {k: 0 for k in ZERO_COUNTERS},
            "coverage": dict(BOUNDARIES),
            "boundaries": dict(BOUNDARIES),
        }

    # --- R0 -----------------------------------------------------------------
    t0 = (_now() if r0_started_perf is None else float(r0_started_perf))
    try:
        budget = budget_for(run_root, "R0", fixed_root=fixed_root, mode=mode)
    except SystemExit as exc:
        return stop([str(exc)], [str(exc)])
    attempt = allocate_attempt(run_root, "R0")
    base_counts = dict(counter.counts)
    pre_register_attempt(run_root, "R0", attempt, t0, scope=scope,
                         budget=budget, call_balance=counter.snapshot())
    reserve_stage_calls(run_root, "R0", attempt,
                        {"counts": STAGE_NOMINAL["R0"]})
    r0 = run_r0(run_root, "R0", attempt, t0, budget, provider, counter,
                roster, mode=mode, tolerance=tolerance,
                project_root=project_root,
                admission_package=admission_package,
                authorization=authorization)
    r0_delta = _stage_delta(counter, base_counts)
    settle_stage_calls(run_root, "R0", attempt, r0_delta)
    results["R0"] = r0
    admission_path = run_root / STAGE_DIRS["R0"] / attempt / "admission.json"
    seal = finish_attempt(
        run_root, "R0", attempt, t0, budget,
        {"complete": bool(r0["passed"]),
         "report": path_for_record(admission_path),
         "summary": path_for_record(admission_path),
         "registration_sha256": r0["registration_sha256"]},
        status="done" if r0["passed"] else "incomplete",
        extra={"stage": "R0", "scope": scope,
               "call_balance": counter.snapshot()})
    seals["R0"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                   "problems": seal.problems, "row": seal.row}
    _row, r0_consumer = stage_state(run_root, "R0")
    seals["R0"]["consumer_problems"] = r0_consumer
    if not seal.sealed or not r0["passed"] or r0_consumer:
        blockers = r0["problems"] + seal.problems + r0_consumer
        return stop(blockers, r0["gaps"] + seal.problems + r0_consumer)
    registration_sha = r0["registration_sha256"]

    # --- P ------------------------------------------------------------------
    t0 = _now()
    try:
        budget = budget_for(run_root, "P", fixed_root=fixed_root, mode=mode)
    except SystemExit as exc:
        return stop([str(exc)], [str(exc)])
    attempt = allocate_attempt(run_root, "P")
    base_counts = dict(counter.counts)
    pre_register_attempt(run_root, "P", attempt, t0, scope=scope,
                         budget=budget, registration_sha=registration_sha,
                         call_balance=counter.snapshot())
    reserve_stage_calls(run_root, "P", attempt,
                        {"counts": STAGE_NOMINAL["P"]})
    stage_deadline = t0 + float(budget["remaining_s"]) - float(budget["reserve_s"])
    try:
        p = run_p_stage(run_root, "P", attempt, t0, budget, provider, counter,
                        roster, guard=_stage_unit_guard(
                            run_root, "P", attempt,
                            registration_sha=registration_sha,
                            deadline=stage_deadline))
    except (TimeoutError, UpdateCallBudgetExceeded) as exc:
        p = {"passed": False, "failures": [type(exc).__name__],
             "gaps": [f"p_stage_aborted:{exc}"], "rows": [], "paths": {},
             "report": None, "arrays_sha256": {},
             "complete_evidence": False, "numeric_passed": False}
    p_delta = _stage_delta(counter, base_counts)
    try:
        settle_stage_calls(run_root, "P", attempt, p_delta)
    except (OSError, ValueError) as exc:
        p["passed"] = False
        p["complete_evidence"] = False
        p["failures"] = list(p.get("failures") or []) + [
            f"reservation_settle:{type(exc).__name__}"]
    results["P"] = p
    p_complete = bool(p.get("complete_evidence"))
    p_report = Path(p["report"]) if p.get("report") else None
    seal = finish_attempt(
        run_root, "P", attempt, t0, budget,
        {"complete": p_complete,
         "report": path_for_record(p_report) if p_report else None,
         "summary": path_for_record(p_report) if p_report else None,
         "registration_sha256": registration_sha,
         "artifacts": _artifact_hashes(
             Path(path) for path in (p.get("paths") or {}).values())},
        status="done" if p_complete else "incomplete",
        extra={"stage": "P", "scope": scope,
               "numeric_passed": bool(p.get("numeric_passed")),
               "call_balance": counter.snapshot()})
    seals["P"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                  "problems": seal.problems, "row": seal.row}
    p_row, p_consumer = stage_state(run_root, "P")
    seals["P"]["consumer_problems"] = p_consumer
    if (not seal.terminal or not seal.sealed or p_consumer
            or not p.get("paths")):
        blockers = seal.problems + p_consumer
        if not p.get("paths"):
            blockers.append("p_artifacts_missing")
        return stop(blockers, p["gaps"] + blockers,
                    failures=p["failures"])
    p_dir = stage_dir(run_root, "P") / attempt

    # --- I ------------------------------------------------------------------
    t0 = _now()
    try:
        budget = budget_for(run_root, "I", fixed_root=fixed_root, mode=mode)
    except SystemExit as exc:
        return stop([str(exc)], [str(exc)])
    attempt = allocate_attempt(run_root, "I")
    base_counts = dict(counter.counts)
    pre_register_attempt(run_root, "I", attempt, t0, scope=scope,
                         budget=budget, registration_sha=registration_sha,
                         call_balance=counter.snapshot())
    reserve_stage_calls(run_root, "I", attempt,
                        {"counts": STAGE_NOMINAL["I"]})
    stage_deadline = t0 + float(budget["remaining_s"]) - float(budget["reserve_s"])
    try:
        i = run_i_stage(run_root, "I", attempt, t0, budget, provider, counter,
                        roster, p_dir, tolerance=tolerance,
                        guard=_stage_unit_guard(
                            run_root, "I", attempt,
                            registration_sha=registration_sha,
                            deadline=stage_deadline))
    except (TimeoutError, UpdateCallBudgetExceeded) as exc:
        i = {"passed": False, "failures": [type(exc).__name__],
             "gaps": [f"i_stage_aborted:{exc}"], "rows": [], "paths": {},
             "report": None, "arrays_sha256": {},
             "complete_evidence": False, "numeric_passed": False}
    i_delta = _stage_delta(counter, base_counts)
    try:
        settle_stage_calls(run_root, "I", attempt, i_delta)
    except (OSError, ValueError) as exc:
        i["passed"] = False
        i["complete_evidence"] = False
        i["failures"] = list(i.get("failures") or []) + [
            f"reservation_settle:{type(exc).__name__}"]
    results["I"] = i
    i_complete = bool(i.get("complete_evidence"))
    i_report = Path(i["report"]) if i.get("report") else None
    seal = finish_attempt(
        run_root, "I", attempt, t0, budget,
        {"complete": i_complete,
         "report": path_for_record(i_report) if i_report else None,
         "summary": path_for_record(i_report) if i_report else None,
         "registration_sha256": registration_sha,
         "artifacts": _artifact_hashes(
             Path(path) for path in (i.get("paths") or {}).values())},
        status="done" if i_complete else "incomplete",
        extra={"stage": "I", "scope": scope,
               "numeric_passed": bool(i.get("numeric_passed")),
               "call_balance": counter.snapshot()})
    seals["I"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                  "problems": seal.problems, "row": seal.row}
    i_row, i_consumer = stage_state(run_root, "I")
    seals["I"]["consumer_problems"] = i_consumer
    if (not seal.terminal or not seal.sealed or i_consumer
            or not i.get("paths")):
        blockers = seal.problems + i_consumer
        if not i.get("paths"):
            blockers.append("i_artifacts_missing")
        return stop(blockers, i["gaps"] + blockers,
                    failures=i["failures"])
    i_dir = stage_dir(run_root, "I") / attempt

    # --- F ------------------------------------------------------------------
    t0 = _now()
    try:
        budget = budget_for(run_root, "F", fixed_root=fixed_root, mode=mode)
    except SystemExit as exc:
        return stop([str(exc)], [str(exc)])
    attempt = allocate_attempt(run_root, "F")
    pre_register_attempt(run_root, "F", attempt, t0, scope=scope,
                         budget=budget, registration_sha=registration_sha,
                         call_balance=counter.snapshot())
    reserve_stage_calls(run_root, "F", attempt,
                        {"counts": STAGE_NOMINAL["F"]})
    try:
        f = run_f_stage(run_root, "F", attempt, t0, budget, provider, counter,
                        roster, p_dir, i_dir, tolerance=tolerance,
                        project_root=project_root)
    except (TimeoutError, UpdateCallBudgetExceeded) as exc:
        f = {"conclusion": "audit_incomplete", "complete_evidence": False,
             "covered_contract_passed": False,
             "verification_problems": [f"f_stage_aborted:{exc}"],
             "failures": [], "gaps": [f"f_stage_aborted:{exc}"],
             "report": None, "summary": {}, "worker": {},
             "call_delta": {}}
    snapshot = f.get("call_delta")
    settled, settle_problems = _settle_f_calls(run_root, "F", attempt,
                                               snapshot)
    if settle_problems:
        f.setdefault("verification_problems", []).extend(settle_problems)
    if not settled:
        f["complete_evidence"] = False
        f["covered_contract_passed"] = False
        f["conclusion"] = "audit_incomplete"
    results["F"] = f
    expected_full = dict(CHAIN_NOMINAL)
    if mode == "synthetic":
        expected_full["scene_setup_mj_forward"] = 0
    matrix_problems = [
        f"full_chain_calls:{kind}:{int(counter.counts[kind])}"
        f"!={int(expected_full[kind])}"
        for kind in CALL_KINDS
        if int(counter.counts[kind]) != int(expected_full[kind])]
    if matrix_problems:
        f["complete_evidence"] = False
        f["covered_contract_passed"] = False
        f["conclusion"] = "audit_incomplete"
        f.setdefault("verification_problems", []).extend(matrix_problems)
    f_report = Path(f["report"]) if f.get("report") else None
    pre_seal_veto = list(matrix_problems)
    if not settled:
        pre_seal_veto.append("f_call_settlement_unknown")
    final_balance = counter.snapshot()
    if settled:
        pre_seal_veto += _call_balance_delta_problems(
            _attempt_row(run_root, "F", attempt).get("call_balance"),
            final_balance, snapshot)
    if pre_seal_veto:
        # Retract the just-written pass claim before the terminal seal and
        # rebind the report hash through the same publishing protocol.  The
        # withdrawal stores the post-execution cumulative call snapshot, not
        # the pre-F balance copied from the running row.
        _amend_final_report(f_report, pre_seal_veto)
        wrow, wprobs, proven = withdraw_attempt_claim(
            run_root, "F", attempt, f_report,
            _attempt_row(run_root, "F", attempt),
            pre_seal_veto, reason="pre_seal_veto", t0=t0, budget=budget,
            call_balance=final_balance)
        seals["F"] = {"sealed": False, "terminal": bool(proven),
                      "problems": wprobs, "row": wrow, "withdrawn": True}
        seal = SealResult(False, wrow, wprobs, bool(proven))
    else:
        seal = finish_attempt(
            run_root, "F", attempt, t0, budget,
            {"complete": bool(f["complete_evidence"]),
             "report": path_for_record(f_report) if f_report else None,
             "summary": path_for_record(f_report) if f_report else None,
             "registration_sha256": registration_sha,
             "artifacts": _artifact_hashes([f_report])},
            status="done" if f["complete_evidence"] else "incomplete",
            extra={"stage": "F", "scope": scope,
                   "call_balance": counter.snapshot()})
        seals["F"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                      "problems": seal.problems, "row": seal.row}
    consumer_row, consumer_problems = stage_state(run_root, "F")
    prerequisite = {}
    prerequisite_problems = []
    for stage_name in ("P", "I"):
        _row, state_problems = stage_state(run_root, stage_name)
        prerequisite[stage_name] = state_problems
        prerequisite_problems += [
            f"prerequisite_not_consumable:{stage_name}:{p}"
            for p in state_problems if p != "attempt_not_complete"]
    seals["F"]["consumer_problems"] = consumer_problems
    seals["F"]["prerequisite_consumer_problems"] = prerequisite
    revoke = (list(seal.problems)
              + [p for p in consumer_problems if p != "attempt_not_complete"]
              + prerequisite_problems + matrix_problems)
    if not settled:
        revoke.append("f_call_settlement_unknown")
    if revoke and _report_claims_complete(f_report):
        # A completed claim survived a post-seal veto (or the seal failed):
        # rewrite the report and reseal incomplete with the old cost floor.
        _amend_final_report(f_report, revoke)
        wrow, wprobs, proven = withdraw_attempt_claim(
            run_root, "F", attempt, f_report,
            seals["F"].get("row") or _attempt_row(run_root, "F", attempt),
            revoke, reason="post_seal_veto", t0=t0, budget=budget,
            call_balance=counter.snapshot())
        seals["F"]["post_seal_withdrawal"] = {
            "row": wrow, "problems": wprobs, "proven": bool(proven)}
        seal = SealResult(False, wrow, list(seal.problems) + wprobs,
                          bool(proven))
        seals["F"].update({"sealed": False, "terminal": bool(proven),
                           "problems": seal.problems, "row": wrow})
        if not proven:
            revoke.append("f_withdraw_unproven")
    if f_report is not None and f_report.is_file():
        try:
            f["summary"] = read_json(f_report)
        except (OSError, ValueError, TypeError):
            revoke.append("f_summary_unreadable")
    conclusion = f["conclusion"]
    complete = bool(f["complete_evidence"])
    covered = bool(f["covered_contract_passed"])
    if revoke or not seal.terminal or not seal.sealed and not complete:
        conclusion = "audit_incomplete"
        complete = False
        covered = False
    if not seal.sealed and not complete:
        conclusion = "audit_incomplete"
    disk_problems = []
    if isinstance(f.get("summary"), Mapping):
        disk_problems = list(f["summary"].get("verification_problems") or [])
    return {
        "mode": mode,
        "conclusion": conclusion,
        "complete_evidence": complete,
        "covered_contract_passed": covered,
        "problems": disk_problems + [p for p in revoke
                                     if p not in disk_problems],
        "failures": f.get("failures", []),
        "gaps": f.get("gaps", []) + revoke,
        "stages": results,
        "seals": seals,
        "prerequisite_consumer_states": prerequisite,
        "summary": f.get("summary", {}),
        "summary_path": f.get("report"),
        "fixture_build_s": fixture_seconds,
        "calls": counter.snapshot(),
        "training_updates": int(counter.counts["native_adamw_steps"]
                                + counter.counts["explicit_reference_updates"]),
        "zero_counters": {k: 0 for k in ZERO_COUNTERS},
        "runtime": getattr(provider, "runtime", None),
        "coverage": dict(BOUNDARIES),
        "boundaries": dict(BOUNDARIES),
    }


def synthetic_self_test(out_root: Path, *, seed: int = FIXTURE_SEED,
                        device: str = "cuda",
                        tolerance: Mapping[str, Any] | None = None,
                        config: Mapping[str, Any] | None = None) -> dict:
    """Full analytic R0->P->I->F chain; no real asset/dataset/checkpoint."""
    run_root = Path(out_root).expanduser().resolve()
    if run_root.exists() and any(run_root.iterdir()):
        n = 1
        while (run_root / f"run_{n:03d}").exists():
            n += 1
        run_root = run_root / f"run_{n:03d}"
    run_root.mkdir(parents=True, exist_ok=True)
    provider = SyntheticTrainingProvider(
        dict(MODEL_CONFIG if config is None else config), seed=seed,
        device=device)
    contract = dict(tolerance or build_default_tolerance_contract())
    result = run_registered_audit(
        run_root, provider, mode="synthetic", tolerance=contract,
        fixed_root=HISTORY_ROOT)
    result["run_root"] = path_for_record(run_root)
    result["tolerance_contract"] = contract
    write_json(run_root / "audit_result.json", result)
    return result
