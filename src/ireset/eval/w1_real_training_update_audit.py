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
import importlib.metadata
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
AUDIT_SCHEMA_V2 = "w1_real_training_update_audit_v2"
COHORT_ID = "w1_real_training_update_audit_v1"
EXECUTION_ID = "w1_real_training_update_audit_v1"
AUTHORIZATION_SCHEMA = "w1_real_training_update_authorization_v1"
AUTHORIZATION_SCHEMA_V2 = "w1_real_training_update_authorization_v2"
FP32_SCHEMA = "w1_real_training_update_fp32_contract_v1"
TOLERANCE_SCHEMA = "w1_real_training_update_tolerance_contract_v1"
CALIBRATION_SCHEMA = "w1_real_training_update_synthetic_calibration_v1"
ADMISSION_SCHEMA = "w1_real_training_update_admission_package_v1"
REGISTRATION_SCHEMA = "w1_real_training_update_registration_v1"
REGISTRATION_SCHEMA_V2 = "w1_real_training_update_registration_v2"
ACCOUNTING_SCHEMA = "pre_registered_v1"
ACCOUNTING_SCHEMA_V2 = "pre_registered_binding_v2"

#: The existing real cohort is migrated in place.  The binding generation is
#: a second identity over the same run root and the same fixed history root;
#: it does not create a fresh cohort or a fresh budget.
GENERATION_001 = "generation_001"
GENERATION_002 = "generation_002"
BINDING_SPEC_SCHEMA = "w1_real_training_update_binding_spec_v1"
HISTORY_SNAPSHOT_SCHEMA = "w1_real_training_update_history_snapshot_v1"
MIGRATION_ADMISSION_SCHEMA = (
    "w1_real_training_update_migration_admission_v1")
LEGACY_HEAD_SCHEMA = "w1_real_training_update_legacy_head_v1"
LAUNCH_CLAIM_SCHEMA = "w1_real_training_update_launch_claim_v1"
RETURN_OBSERVATION_SCHEMA = (
    "w1_real_training_update_return_observation_v1")
EVENT_SCHEMA = "w1_real_training_update_execution_event_v1"
BINDING_INTENT_SCHEMA = "w1_real_training_update_binding_intent_v1"
BINDING_ACTIVATION_SCHEMA = "w1_real_training_update_binding_activation_v1"

MIGRATION_SOURCE_RUN = Path(
    "derived/experiments/omniretarget_first/"
    "20261102_2704709_w1_real_training_update_audit")
MIGRATION_SOURCE_MARKER_SHA256 = (
    "79f42f5fc2bb205a4f8327b7897644e4759ff60bf189f4dd6054cb980e9f3f6b")
MIGRATION_SOURCE_INDEX_SHA256 = (
    "78a2f97dc42545c21f2200a9353b4e1ce9f4dea3fde9cf88f27c29548932b950")
MIGRATION_SOURCE_LEDGER_SHA256 = (
    "92773314541bec3b3d751c1627fb8cf78b331ea21303b8c753b8d4a447aa0f34")
MIGRATION_SOURCE_AUTHORIZATION_SHA256 = (
    "199bfb30f11502aaf57acf38f1c49525ac58e511f870f230f223cb12b8525d78")
MIGRATION_SOURCE_REGISTRATION_SHA256 = (
    "6abab029788f0e87859a560e3e9866c9958758f5a8765b7092f34ca5d53848aa")
MIGRATION_SOURCE_COST_S = 67.10273322396097
MIGRATION_SOURCE_UPDATES = 2
MIGRATION_LEGACY_HEAD_SHA256 = (
    "b499a6ce7e5ef8fac06821cc063f27bd109374d9673791551fe92ad56302a424")
MIGRATION_STAGE_PLAN = (
    {"execution_seq": 3, "generation": GENERATION_002,
     "stage": "R0", "attempt": "attempt_002"},
    {"execution_seq": 4, "generation": GENERATION_002,
     "stage": "P", "attempt": "attempt_002"},
    {"execution_seq": 5, "generation": GENERATION_002,
     "stage": "I", "attempt": "attempt_001"},
    {"execution_seq": 6, "generation": GENERATION_002,
     "stage": "F", "attempt": "attempt_001"},
)
MIGRATION_PRIOR_UPDATES = 2
MIGRATION_INCREMENTAL_UPDATE_LIMIT = 16
MIGRATION_COHORT_UPDATE_LIMIT = 18
MIGRATION_STAGE_ATTEMPTS = {"R0": 1, "P": 1, "I": 1, "F": 1}
MIGRATION_METADATA_BYTES_LIMIT = 32 * 2 ** 20
MIGRATION_PLAN = Path(
    "docs/plans/"
    "omniretarget_w1_real_training_update_binding_migration_plan.md")
LAUNCH_JOURNAL_DIR = "launch_journal"
BINDINGS_DIR = "bindings"
EXECUTION_EVENTS_DIR = "execution_events"

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


# ---------------------------------------------------------------------------
# binding migration: canonical JSON, exclusive publication and manifests
# ---------------------------------------------------------------------------


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def strict_json_loads(data: str) -> Any:
    """Parse JSON without accepting duplicate keys.

    The migration documents are hash-bound.  ``json.loads`` silently keeps the
    last duplicate key, which would let two readers disagree about the bytes
    that produced a SHA.  The regular v1 readers keep their historical
    ``read_json`` behaviour; only binding/migration records use this parser.
    """
    return json.loads(data, object_pairs_hook=_reject_duplicate_keys)


def strict_read_json(path: Path) -> Any:
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    """The single canonical ``H()`` encoding from the migration plan."""
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(Path(path)), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_exclusive(path: Path, payload: bytes) -> str:
    """Atomically publish exact bytes once, never overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite {path}") from exc
        _fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return sha256_file(path)


def write_json_exclusive(path: Path, data: Any) -> str:
    """Atomically publish a JSON file once, never overwriting.

    The bytes are written to a sibling temporary file, fsynced, linked into
    place without overwrite, and the directory is fsynced.  A crash therefore
    leaves either no target, a complete target, or an observable ``.tmp``
    sibling; the latter is treated as an unsettled launch/event liability by
    the readers rather than being silently cleaned up.
    """
    payload = (json.dumps(_jsonable(data), indent=2, sort_keys=True,
                          ensure_ascii=False, allow_nan=False) + "\n").encode(
                              "utf-8")
    return write_bytes_exclusive(Path(path), payload)


def _manifest_files(base: Path, *, skip: Sequence[str] = ()) -> tuple[dict, list[str]]:
    """Return ``{relative_path: {sha256, bytes}}`` for regular files."""
    base = Path(base)
    manifest: dict = {}
    problems: list[str] = []
    if not base.is_dir():
        return manifest, problems
    skip_set = {str(Path(s)) for s in skip}
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        if rel in skip_set:
            continue
        try:
            manifest[rel] = {"sha256": sha256_file(path),
                             "bytes": int(path.stat().st_size)}
        except OSError as exc:
            problems.append(f"manifest_unreadable:{rel}:{type(exc).__name__}")
    return manifest, problems


def _manifest_problems(base: Path, manifest: Any, *, require_exact: bool = True
                       ) -> list[str]:
    if not isinstance(manifest, Mapping):
        return ["manifest_not_a_mapping"]
    actual, problems = _manifest_files(base)
    if require_exact and set(actual) != set(manifest):
        problems.append("manifest_file_set")
    for rel, rec in manifest.items():
        if not isinstance(rec, Mapping):
            problems.append(f"manifest_entry_invalid:{rel}")
            continue
        path = Path(base) / str(rel)
        if not path.is_file():
            problems.append(f"manifest_missing:{rel}")
            continue
        if rec.get("sha256") != sha256_file(path):
            problems.append(f"manifest_sha:{rel}")
        try:
            size = int(path.stat().st_size)
        except OSError:
            problems.append(f"manifest_unreadable:{rel}")
            continue
        if type(rec.get("bytes")) is not int or int(rec["bytes"]) != size:
            problems.append(f"manifest_bytes:{rel}")
    return problems


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


ALPHA_EVALUATION = "torch_cpu_float32_scaled_linear_v1"


def effective_alpha_reference(
        config: Mapping[str, Any] | None = None) -> np.ndarray:
    """Rebuild the scheduler's effective constants without reading its buffer.

    Diffusers evaluates linspace, square, subtraction and cumprod in CPU
    float32.  Computing the whole schedule in float64 and rounding only the
    result is a different graph.  Keep ``alpha_cumprod_reference`` for the
    raw parent/source check; independent x_t uses these effective constants
    promoted to float64, as required by preregistration section 3.2.
    """
    config = MODEL_CONFIG if config is None else config
    if config["beta_schedule"] != "scaled_linear":
        raise ValueError("effective alpha requires the frozen scaled_linear schedule")
    betas = torch.linspace(float(config["beta_start"]) ** 0.5,
                           float(config["beta_end"]) ** 0.5,
                           int(config["num_train_timesteps"]),
                           dtype=torch.float32, device="cpu") ** 2
    return torch.cumprod(1.0 - betas, dim=0).numpy().astype(np.float64)


def verify_production_alpha(alpha: torch.Tensor,
                            config: Mapping[str, Any]) -> None:
    """Require the native buffer's storage and values before any forward."""
    if not isinstance(alpha, torch.Tensor) or alpha.dtype != torch.float32:
        raise ValueError("production diffusion alpha dtype differs")
    expected = effective_alpha_reference(config)
    actual = alpha.detach().cpu().numpy()
    if actual.shape != expected.shape:
        raise ValueError("production diffusion alpha shape differs")
    if not np.array_equal(actual.astype(np.float64), expected):
        raise ValueError("production diffusion alpha schedule differs")


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


class SyntheticReferenceBackend:
    """Analytic 29-joint/15-site chain behind the common reference API."""

    schema = "w1_real_training_update_reference_backend_v1"
    mode = "synthetic"

    def __init__(self, chain: SyntheticChain, counter: CallCounter) -> None:
        self.chain = chain
        self.counter = counter

    def positions(self, pos: np.ndarray, R: np.ndarray,
                  q: np.ndarray) -> np.ndarray:
        out = synthetic_fk_np(self.chain, pos, R, q)
        self.counter.add("reference_mj_forward", int(np.shape(pos)[0]))
        return out

    def jacobians(self, pos: np.ndarray, R: np.ndarray,
                  q: np.ndarray) -> np.ndarray:
        _, jac = synthetic_fk_jac_np(self.chain, pos, R, q)
        Rt = np.swapaxes(np.asarray(R, np.float64), 1, 2)
        J35 = np.array(jac, copy=True)
        J35[..., 3:6] = np.einsum("nsck,nkj->nscj", jac[..., 3:6], Rt)
        n = int(np.shape(pos)[0])
        self.counter.add("reference_mj_forward", n)
        self.counter.add("reference_mj_jac_site", n * len(SITE_NAMES))
        return J35


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
        self.reference: SyntheticReferenceBackend | None = None
        self._alpha: np.ndarray | None = None
        self.problems: list[str] = []
        self.fixture_seconds: float | None = None

    def bind_counter(self, counter: CallCounter) -> None:
        self.counter = counter
        if self.chain is not None:
            self.reference = SyntheticReferenceBackend(self.chain, counter)
        if self.reference is not None:
            self.reference.counter = counter

    # --- provider identity ------------------------------------------------

    def spec(self) -> dict:
        return {
            "schema": "w1_real_training_update_provider_v1",
            "mode": "synthetic",
            "fixture_dir": (path_for_record(self.fixture_dir)
                            if self.fixture_dir is not None else None),
            "config": dict(self.config),
            "device": self.device,
            "seed": self.seed,
        }

    def for_stage(self, stage: str) -> "SyntheticTrainingProvider":
        return self

    def materialize(self, *, stage: str, r0_noise: Any = None,
                    counter: CallCounter | None = None,
                    parent_input_admission: Any = None,
                    guard: Any = None,
                    deadline: float | None = None,
                    registered: bool = False) -> dict:
        return {
            "schema": "w1_real_training_update_input_admission_v1",
            "mode": "synthetic",
            "stage": stage, "registered": bool(registered),
            "passed": not self.problems,
            "problems": list(self.problems),
            "read_scope": {"npz_values_read": 0, "checkpoint_payload_read": 0,
                           "forward_calls": 0},
        }

    def reference_positions(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray:
        if self.reference is None:
            self._refresh_reference()
        return self.reference.positions(pos, R, q)

    def reference_jacobians(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray:
        if self.reference is None:
            self._refresh_reference()
        return self.reference.jacobians(pos, R, q)

    def _refresh_reference(self) -> None:
        if self.chain is None:
            raise ValueError("synthetic provider has no chain")
        counter = self.counter if self.counter is not None else CallCounter(
            {})
        self.reference = SyntheticReferenceBackend(self.chain, counter)

    @property
    def alpha_cumprod(self) -> np.ndarray:
        if self._alpha is None:
            self._alpha = alpha_cumprod_reference(self.config)
        return self._alpha

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
        counter = self.counter if self.counter is not None else CallCounter({})
        self.reference = SyntheticReferenceBackend(self.chain, counter)
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
            verify_production_alpha(diff.ddpm.alphas_cumprod, self.config)
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
        alpha_effective = effective_alpha_reference(self.config)
        x_t_ref: list[np.ndarray] = []
        for unit in noise:
            slots = MICRO_SCHEDULE[unit["update"]][unit["micro"]]
            x0 = np.stack([window_arrays[w]["x0"] for w in slots], axis=0)
            x_t_ref.append(independent_x_t(x0, unit["epsilon"].numpy(),
                                           unit["t"].numpy(),
                                           alpha_effective))
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
            "alpha_effective_sha256": _hash_array(alpha_effective),
            "alpha_evaluation": ALPHA_EVALUATION,
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
        if model_type == "diffusion":
            verify_production_alpha(trainer.model.ddpm.alphas_cumprod,
                                    self.config)
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


def _decode_r0_noise(provider: Any, *, path: Path | None,
                     digest: str | None,
                     check: Callable[[str], None]) -> list[dict]:
    """Decode the frozen R0 schedule inside the consuming attempt.

    Only the sealed path and digest cross the stage boundary.  The decode
    runs under the stage's deadline and unit admission, so a slow read is
    charged to the stage that consumes it instead of slipping between two
    registered attempts.
    """
    if path is None or not digest:
        raise ValueError("R0 noise schedule missing")
    check("r0_noise_digest")
    actual = sha256_file(Path(path))
    if actual != digest:
        raise ValueError(f"R0 noise schedule drifted: {actual}")
    check("r0_noise_decode")
    schedule = provider.load_noise_schedule(Path(path))
    check("r0_noise_decoded")
    return list(schedule)


def frozen_noise_source(root: Path, attempt: str,
                        row: Mapping[str, Any] | None
                        ) -> tuple[Path, str]:
    """Return the R0 path and digest carried by the sealed row.

    The digest is read from the sealed artifacts, never recomputed between
    stages: a consumer must re-verify it inside its own registered window.
    """
    path = stage_dir(root, "R0") / attempt / "noise_schedule.npz"
    digest = None
    if isinstance(row, Mapping) and isinstance(row.get("artifacts"),
                                               Mapping):
        digest = row["artifacts"].get(path_for_record(path))
    if not path.is_file():
        raise SystemExit("R0 noise schedule missing")
    if not isinstance(digest, str) or not digest:
        raise SystemExit("R0 noise schedule digest missing")
    return path, digest



def run_p_stage(root: Path, stage: str, attempt: str, t0: float,
                budget: Mapping[str, Any], provider: Any,
                counter: CallCounter, roster: Mapping[str, Any],
                *, out_dir: Path | None = None,
                fixture_error: str | None = None,
                guard: Callable[[str], None] | None = None,
                registered: bool = True,
                r0_noise: Sequence[dict] | None = None,
                r0_noise_path: Path | None = None,
                r0_noise_sha256: str | None = None,
                parent_input_admission: Mapping[str, Any] | None = None,
                write_input_admission: bool = False,
                r0_registration_sha: str | None = None) -> dict:
    base = Path(out_dir or (stage_dir(root, stage) / attempt))
    base.mkdir(parents=True, exist_ok=True)
    _seed_stage_rng()
    deadline = t0 + float(budget["remaining_s"]) - float(budget["reserve_s"])

    def check(unit: str) -> None:
        _ensure_deadline(deadline, 0.0, unit)
        if guard is not None:
            guard(unit)

    binder = getattr(provider, "bind_counter", None)
    if callable(binder):
        binder(counter)
    mode = str(getattr(provider, "mode", "synthetic"))
    results: dict[str, dict] = {}
    rows: list[dict] = []
    failures: list[str] = []
    gaps: list[str] = list([fixture_error] if fixture_error else [])
    admission_path: Path | None = None
    admission_sha: str | None = None
    if mode == "real":
        if r0_noise is None:
            r0_noise = _decode_r0_noise(
                provider, path=r0_noise_path, digest=r0_noise_sha256,
                check=check)
        admission = provider.materialize(
            stage="P", r0_noise=r0_noise, counter=counter, guard=check,
            deadline=deadline, registered=registered,
            parent_input_admission=parent_input_admission,
            r0_registration_sha=r0_registration_sha)
        if not admission.get("passed"):
            failures += [f"p_input_admission:{p}"
                         for p in admission.get("problems") or []]
        admission_path = None
        admission_sha = None
        if write_input_admission:
            admission_path = base / "input_admission.json"
            write_json(admission_path, admission)
            admission_sha = sha256_file(admission_path)
        if not admission.get("passed"):
            rows = []
            report = {
                "schema": "w1_real_training_update_p_stage_v1",
                "mode": mode, "stage": stage, "attempt": attempt,
                "models": {}, "rows": rows, "failures": failures,
                "gaps": gaps, "evidence_problems": ["p_input_admission"],
                "complete_evidence": False, "numeric_passed": False,
                "call_delta": None, "passed": False,
                "input_admission": (
                    {"path": path_for_record(admission_path),
                     "sha256": admission_sha}
                    if admission_path is not None else None),
            }
            write_json(base / "p_rows.json", rows)
            write_json(base / "p_report.json", report)
            return {"passed": False, "failures": failures, "gaps": gaps,
                    "rows": rows, "complete_evidence": False,
                    "evidence_problems": ["p_input_admission"],
                    "numeric_passed": False, "paths": {},
                    "report": path_for_record(base / "p_report.json"),
                    "arrays_sha256": {}, "call_delta": None}
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
                        slots = MICRO_SCHEDULE[u][m]
                        x0 = np.stack([provider.window_arrays[w]["x0"]
                                       for w in slots], axis=0)
                        x_t_ref = independent_x_t(
                            x0, unit["epsilon"].numpy(), unit["t"].numpy(),
                            effective_alpha_reference(provider.config))
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
        "mode": mode,
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
        "input_admission": (
            {"path": path_for_record(admission_path),
             "sha256": admission_sha}
            if admission_path is not None else None),
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
        self.target: dict[str, dict] = {}
        for key in WINDOW_IDS:
            self._check(f"reference_target_{key}")
            x0 = np.asarray(provider.window_arrays[key]["x0"],
                            dtype=np.float32).reshape(-1, 38)
            decoded = independent_decode(x0.astype(np.float64), self.mean,
                                         self.std, self.lo, self.hi)
            # The decode itself can outlast the deadline; re-check before the
            # charged target FK so no reference FK starts after expiry.
            self._check(f"reference_target_{key}")
            kp = provider.reference_positions(decoded["pos"], decoded["R"],
                                              decoded["q"])
            self.target[key] = {"kp": kp.reshape(-1, len(SITE_NAMES), 3),
                                "ok": bool(decoded["ok"].all()),
                                "boundary": bool(decoded["clamp_boundary"].any())}
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
        kp = self.provider.reference_positions(decoded["pos"], decoded["R"],
                                               decoded["q"])
        self._check("reference_jacobian", guard)
        J35 = self.provider.reference_jacobians(decoded["pos"], decoded["R"],
                                                decoded["q"])
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
                budget: Mapping[str, Any], provider: Any,
                counter: CallCounter, roster: Mapping[str, Any],
                p_dir: Path, *, tolerance: Mapping[str, Any],
                out_dir: Path | None = None,
                guard: Callable[[str], None] | None = None,
                registered: bool = True,
                r0_noise: Sequence[dict] | None = None,
                r0_noise_path: Path | None = None,
                r0_noise_sha256: str | None = None,
                parent_input_admission: Mapping[str, Any] | None = None,
                r0_registration_sha: str | None = None) -> dict:
    base = Path(out_dir or (stage_dir(root, stage) / attempt))
    base.mkdir(parents=True, exist_ok=True)
    _seed_stage_rng()
    deadline = t0 + float(budget["remaining_s"]) - float(budget["reserve_s"])

    def check(unit: str) -> None:
        _ensure_deadline(deadline, 0.0, unit)
        if guard is not None:
            guard(unit)

    binder = getattr(provider, "bind_counter", None)
    if callable(binder):
        binder(counter)
    mode = str(getattr(provider, "mode", "synthetic"))
    tol = _declared_tolerance(tolerance)
    failures: list[str] = []
    gaps: list[str] = []
    if mode == "real":
        if r0_noise is None:
            r0_noise = _decode_r0_noise(
                provider, path=r0_noise_path, digest=r0_noise_sha256,
                check=check)
        admission = provider.materialize(
            stage="I", r0_noise=r0_noise, counter=counter, guard=check,
            deadline=deadline, registered=registered,
            parent_input_admission=parent_input_admission,
            r0_registration_sha=r0_registration_sha)
        if not admission.get("passed"):
            failures += [f"i_input_admission:{p}"
                         for p in admission.get("problems") or []]
        if not admission.get("passed"):
            rows: list[dict] = []
            report = {
                "schema": "w1_real_training_update_i_stage_v1",
                "mode": mode, "stage": stage, "attempt": attempt,
                "models": {}, "rows": rows, "failures": failures,
                "gaps": gaps, "evidence_failures": ["i_input_admission"],
                "evidence_problems": ["i_input_admission"],
                "complete_evidence": False, "numeric_passed": False,
                "passed": False,
            }
            write_json(base / "i_rows.json", rows)
            write_json(base / "i_report.json", report)
            return {"passed": False, "failures": failures, "gaps": gaps,
                    "rows": rows, "complete_evidence": False,
                    "evidence_failures": ["i_input_admission"],
                    "evidence_problems": ["i_input_admission"],
                    "numeric_passed": False, "paths": {},
                    "report": path_for_record(base / "i_report.json"),
                    "arrays_sha256": {}}
    p_reports = {model: _p_sidecar(p_dir, model) for model in MODELS}
    check("i_reference_context")
    ref_ctx = ReferenceContext(provider, counter, guard=check)
    failures += list(ref_ctx.problems)
    gaps += list(ref_ctx.problems)
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
                        x_t_ref = independent_x_t(
                            x0, unit["epsilon"].numpy(), unit["t"].numpy(),
                            effective_alpha_reference(provider.config))
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
        "mode": mode,
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

    The current F attempt is deliberately still ``running/consumed=null``:
    candidate checks use the settled R0/P/I deltas plus this worker's local
    ``L_F`` (``provisional_chain_counts``); the parent settles and re-checks
    from disk afterwards (plan section 2.2).
    """
    spec = read_json(Path(spec_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter = CallCounter(spec["remaining_caps"])
    deadline = float(spec["deadline_perf_counter"])
    reserve = float(spec.get("reserve_s", 0.0))
    root = Path(spec["run_root"])
    attempt = str(spec["attempt"])
    mode = str(spec.get("mode", "synthetic"))
    use_binding = bool(spec.get("use_binding"))
    binding = None
    if use_binding:
        binding, binding_problems = load_binding(
            root, generation=str(spec.get("generation") or GENERATION_002),
            require_active=True)
        if binding is None:
            raise SystemExit(f"F worker binding refused: {binding_problems}")
        if spec.get("binding_spec_sha256") != binding.spec_sha256:
            raise SystemExit("F worker binding spec drifted")
    generation = binding.generation if binding is not None else None
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
    precheck: dict = {}
    candidate_attempts = dict(spec.get("candidate_attempts") or {})
    candidate_checks = bool(spec.get("candidate_checks", True)) and bool(
        candidate_attempts)
    provisional: dict = {}
    stage_totals: dict = {}
    cohort_totals: dict = {}
    local_counts = {kind: 0 for kind in CALL_KINDS}

    def guard(unit: str) -> None:
        # ``deadline`` is already the work deadline (the seal reserve was
        # subtracted once when the stage started), so do not subtract it again.
        _ensure_deadline(deadline, 0.0, unit)
        admission = unit_admission(root, "F", attempt, parent_record,
                                   deadline=deadline)
        if admission:
            raise TimeoutError(f"unit_admission:{unit}:{','.join(admission)}")

    def precheck_candidates() -> list[str]:
        found: list[str] = []
        history = verify_history(
            root, allow_current=("F", attempt),
            allow_invocation=spec.get("invocation_id"),
            generation=generation, binding=binding)
        found += [f"f_candidate_history:{p}" for p in history]
        if set(candidate_attempts) != set(STAGE_BUDGETS):
            found.append("f_candidate_keys")
        internal = ({"invocation_id": spec.get("invocation_id"),
                     "allow_current": True}
                    if use_binding else None)
        for name in ("R0", "P", "I"):
            selected = candidate_attempts.get(name)
            if not isinstance(selected, str):
                found.append(f"f_candidate_missing:{name}")
                continue
            row, state = stage_state(
                root, name, selected, allow_current=("F", attempt),
                generation=generation, binding=binding,
                internal_context=internal)
            precheck[name] = {"attempt": selected, "problems": state,
                              "report": (row or {}).get("report")}
            if state:
                found += [f"f_candidate:{name}:{p}" for p in state]
        current = _attempt_row(root, "F", attempt)
        if current.get("status") != "running":
            found.append("f_current_not_running")
        if candidate_attempts.get("F") != attempt:
            found.append("f_candidate_self")
        if _reservation_consumed(root, "F", attempt) is not None:
            found.append("f_current_already_settled")
        precheck["current"] = {
            "attempt": attempt, "status": current.get("status"),
            "registration_sha256": current.get(
                "execution_registration_sha256"),
            "deadline_perf_counter": current.get("deadline_perf_counter"),
        }
        return found

    def provisional_counts() -> list[str]:
        found: list[str] = []
        deltas: dict[str, dict] = {}
        for name in ("R0", "P", "I"):
            stage_deltas = stage_attempt_deltas(
                root, name, current=("F", attempt), generation=generation,
                binding=binding)
            if stage_deltas is None or candidate_attempts.get(name) not in \
                    stage_deltas:
                found.append(f"f_provisional_delta:{name}")
                continue
            deltas[name] = stage_deltas
        if found:
            return found
        chain = {kind: int(deltas["R0"][candidate_attempts["R0"]]["counts"][kind])
                 + int(deltas["P"][candidate_attempts["P"]]["counts"][kind])
                 + int(deltas["I"][candidate_attempts["I"]]["counts"][kind])
                 + int(local_counts[kind])
                 for kind in CALL_KINDS}
        provisional.update({
            "selected_chain_counts": chain,
            "training_updates": int(chain["native_adamw_steps"]
                                    + chain["explicit_reference_updates"]),
        })
        expected_chain = chain_nominal_for_mode(mode)
        for kind in CALL_KINDS:
            if chain[kind] != expected_chain[kind]:
                found.append(
                    f"f_provisional_chain:{kind}:{chain[kind]}"
                    f"!={expected_chain[kind]}")
        for stage in STAGE_BUDGETS:
            totals = {kind: 0 for kind in CALL_KINDS}
            stage_deltas = stage_attempt_deltas(
                root, stage, current=("F", attempt))
            if stage_deltas is None:
                found.append(f"f_stage_delta:{stage}")
                continue
            if stage == "F":
                for kind in CALL_KINDS:
                    totals[kind] = sum(
                        int(rec["counts"][kind])
                        for rec in stage_deltas.values()) + int(
                        local_counts[kind])
            else:
                for rec in stage_deltas.values():
                    for kind in CALL_KINDS:
                        totals[kind] += int(rec["counts"][kind])
            stage_totals[stage] = totals
            nominal = stage_nominal_for_mode(mode, stage)
            for kind in CALL_KINDS:
                if totals[kind] > 2 * nominal[kind]:
                    found.append(f"f_stage_cap:{stage}:{kind}")
            if stage == "F":
                used = len(stage_deltas) + 1
                if stage == "F" and binding is not None:
                    used = len([r for r in stage_deltas.values()
                                if r.get("generation") == generation]) + 1
                if used > STAGE_BUDGETS["F"]["attempts"]:
                    found.append("f_attempt_cap")
        cohort = {kind: 0 for kind in CALL_KINDS}
        for stage in STAGE_BUDGETS:
            for kind in CALL_KINDS:
                cohort[kind] += int(stage_totals.get(stage, {}).get(kind, 0))
        cohort_totals.update(cohort)
        for kind in CALL_KINDS:
            if cohort[kind] > int(CALL_CAPS[kind]):
                found.append(f"f_cohort_cap:{kind}")
        return found

    try:
        guard("worker_start")
        runtime = configure_production_runtime()
        if runtime["problems"]:
            problems += [f"worker_runtime:{p}" for p in runtime["problems"]]
        identity_problems: list[str] = []
        if mode == "real":
            identity_problems = [f"worker_runtime_identity:{p}" for p in
                                 runtime_identity_problems(
                                     (spec.get("provider") or {}).get(
                                         "runtime_identity"))]
            problems += identity_problems
        candidates_problems = (precheck_candidates()
                               if candidate_checks else [])
        if candidates_problems or identity_problems:
            problems += candidates_problems
            raise TimeoutError(
                "candidate_precheck_failed" if candidates_problems
                else "runtime_identity_failed")
        guard("fixture_load")
        if mode == "real":
            from ireset.eval.w1_real_training_update_provider import (
                RealTrainingProvider,
            )
            provider = RealTrainingProvider.from_spec(
                spec["provider"], project_root=Path(spec["project_root"]))
            provider.roster = spec["roster"]
        else:
            provider = SyntheticTrainingProvider(
                spec["config"], device=spec.get("device", "cuda"))
            provider.fixture_dir = Path(spec["fixture_dir"])
            provider.load_fixture()
            provider.roster = read_json(provider.fixture_dir / "roster.json")
        roster = provider.roster
        guard("schedule_rebuild")
        schedule = draw_noise_path(NOISE_SEED)
        counter.add("cpu_draw_units", DRAW_UNITS_PER_PATH)
        schedule_ok = True
        source_problems: list[str] = []
        if mode == "real":
            noise_path = Path(spec.get("r0_noise_path") or "")
            if not noise_path.is_file():
                source_problems.append("schedule_source_missing")
            elif sha256_file(noise_path) != spec.get("r0_noise_sha256"):
                source_problems.append("schedule_source_sha")
            else:
                stored = provider.load_noise_schedule(noise_path)
                for a, b in zip(schedule, stored):
                    if (not torch.equal(a["t"], b["t"])
                            or not torch.equal(a["epsilon"], b["epsilon"])
                            or not torch.equal(a["generator_state_before"],
                                               b["generator_state_before"])
                            or not torch.equal(a["generator_state_after"],
                                               b["generator_state_after"])):
                        schedule_ok = False
                provider.noise = schedule
        else:
            fixture_noise = provider.noise
            for a, b in zip(schedule, fixture_noise):
                if (not torch.equal(a["t"], b["t"])
                        or not torch.equal(a["epsilon"], b["epsilon"])):
                    schedule_ok = False
        if not schedule_ok:
            source_problems.append("schedule_rebuild_mismatch")
        main_admission = None
        if mode == "real":
            admission_path = Path(spec.get("p_input_admission_path") or "")
            if not admission_path.is_file():
                source_problems.append("p_input_admission_missing")
            else:
                actual_sha = sha256_file(admission_path)
                main_admission = read_json(admission_path)
                if actual_sha != spec.get("p_input_admission_sha256"):
                    source_problems.append("p_input_admission_sha")
                if not isinstance(main_admission, Mapping) or \
                        main_admission.get("passed") is not True:
                    source_problems.append("p_input_admission_not_passed")
        if source_problems:
            problems += source_problems
            raise TimeoutError("f_source_precheck_failed")
        p_dir = out / "production_replay"
        i_dir = out / "independent_replay"
        guard("p_replay_start")
        p_replay = run_p_stage(
            out, "P", "F-P", _now(),
            {"remaining_s": max(0.0, deadline - _now() + reserve),
             "reserve_s": reserve}, provider, counter, roster,
            out_dir=p_dir, guard=guard, registered=False,
            r0_noise=(schedule if mode == "real" else None),
            parent_input_admission=main_admission,
            r0_registration_sha=spec.get("registration_sha256"))
        guard("p_replay_compare")
        p_problems = _npz_compare_problems(Path(spec["p_dir"]), p_dir, MODELS,
                                           prefix="p")
        problems += p_problems
        guard("i_replay_start")
        i_replay = run_i_stage(
            out, "I", "F-I", _now(),
            {"remaining_s": max(0.0, deadline - _now() + reserve),
             "reserve_s": reserve}, provider, counter, roster,
            p_dir, tolerance=spec["tolerance"], out_dir=i_dir, guard=guard,
            registered=False,
            r0_noise=(schedule if mode == "real" else None),
            parent_input_admission=main_admission,
            r0_registration_sha=spec.get("registration_sha256"))
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
        failures += list(p_replay["failures"]) + list(i_replay["failures"])
        gaps += list(p_replay["gaps"]) + list(i_replay["gaps"])
        if not p_replay.get("complete_evidence", False):
            problems.append("p_replay_incomplete_evidence")
        if not i_replay.get("complete_evidence", False):
            problems.append("i_replay_incomplete_evidence")
        local_counts = counter.snapshot()["counts"]
        expected_f = stage_nominal_for_mode(mode, "F")
        call_problems = [
            f"f_call_matrix:{kind}:{counts_now}!={expected_f[kind]}"
            for kind, counts_now in local_counts.items()
            if counts_now != expected_f[kind]]
        problems += call_problems
        provisional_problems = (provisional_counts()
                                if candidate_checks else [])
        if provisional_problems:
            problems += provisional_problems
    except TimeoutError as exc:
        aborted = True
        problems.append(f"f_worker_aborted:{exc}")
    except Exception as exc:  # noqa: BLE001 -- a broken worker is evidence
        aborted = True
        problems.append(f"f_worker_exception:{type(exc).__name__}:{exc}")
    finally:
        snapshot = counter.snapshot()
        local_counts = dict(snapshot["counts"])
        result = {
            "schema": "w1_real_training_update_f_worker_v1",
            "mode": mode,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "python": sys.version.split()[0],
            "runtime": runtime,
            "runtime_identity": runtime_identity_facts(),
            "provider": (path_for_record(provider.fixture_dir)
                         if mode == "synthetic" and provider is not None
                         and provider.fixture_dir is not None
                         else (getattr(provider, "spec", lambda: None)()
                               if provider is not None else None)),
            "candidate_attempts": candidate_attempts,
            "precheck": precheck,
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
            "provisional_chain_counts": provisional.get(
                "selected_chain_counts"),
            "provisional_training_updates": provisional.get(
                "training_updates"),
            "stage_call_counts": stage_totals,
            "cohort_call_counts": cohort_totals,
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
                budget: Mapping[str, Any], provider: Any,
                counter: CallCounter, roster: Mapping[str, Any],
                p_dir: Path, i_dir: Path, *, tolerance: Mapping[str, Any],
                out_dir: Path | None = None,
                project_root: Path = PROJECT_ROOT,
                candidate_attempts: Mapping[str, str] | None = None,
                r0_noise_path: Path | None = None,
                r0_noise_sha256: str | None = None,
                p_input_admission_path: Path | None = None,
                p_input_admission_sha256: str | None = None,
                execution: Mapping[str, Any] | None = None) -> dict:
    execution = dict(execution or {})
    base = Path(out_dir or (stage_dir(root, stage) / attempt))
    base.mkdir(parents=True, exist_ok=True)
    deadline = t0 + float(budget["remaining_s"]) - float(budget["reserve_s"])
    mode = str(getattr(provider, "mode", "synthetic"))
    base_counts = counter.snapshot()["counts"]
    remaining = {kind: int(CALL_CAPS[kind]) - int(base_counts[kind])
                 for kind in CALL_KINDS}
    spec: dict[str, Any] = {
        "mode": mode,
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
        "candidate_attempts": dict(candidate_attempts or {}),
        "candidate_checks": candidate_attempts is not None,
        "project_root": path_for_record(Path(project_root)),
        "roster": dict(roster),
        "generation": execution.get("generation") or GENERATION_001,
        "binding_spec_sha256": execution.get("binding_spec_sha256"),
        "execution_seq": execution.get("execution_seq"),
        "invocation_id": execution.get("invocation_id"),
        "legacy_head_sha256": execution.get("legacy_head_sha256"),
        "claim_path": execution.get("claim_path"),
        "claim_sha256": execution.get("claim_sha256"),
        "use_binding": bool(execution),
    }
    if mode == "real":
        spec["provider"] = provider.spec()
        spec["r0_noise_path"] = (path_for_record(r0_noise_path)
                                 if r0_noise_path is not None else None)
        spec["r0_noise_sha256"] = r0_noise_sha256
        spec["p_input_admission_path"] = (
            path_for_record(p_input_admission_path)
            if p_input_admission_path is not None else None)
        spec["p_input_admission_sha256"] = p_input_admission_sha256
    else:
        spec["fixture_dir"] = path_for_record(provider.fixture_dir)
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
    expected_f = stage_nominal_for_mode(mode, "F")
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
        "mode": mode,
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
        "candidate_attempts": dict(candidate_attempts or {}),
        "provisional_chain_counts": result.get("provisional_chain_counts"),
        "provisional_training_updates": result.get(
            "provisional_training_updates"),
        "selected_attempts": None,
        "selected_chain_counts": None,
        "selected_chain_training_updates": None,
        "training_updates_scope": "provisional_selected_chain",
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
# binding migration: history snapshot, legacy head and binding identity
# ---------------------------------------------------------------------------


def _legacy_reference_rows(index_rows: Sequence[Mapping[str, Any]],
                           ledger_rows: Sequence[Mapping[str, Any]]
                           ) -> tuple[Mapping[str, Any], Mapping[str, Any],
                                      Mapping[str, Any], Mapping[str, Any]]:
    def one(rows: Sequence[Mapping[str, Any]], stage: str
            ) -> Mapping[str, Any]:
        hits = [r for r in rows
                if r.get("stage") == stage
                and r.get("attempt") == "attempt_001"]
        if len(hits) != 1:
            raise ValueError(
                f"legacy head requires exactly one {stage}/attempt_001 row")
        return hits[0]

    r0_index = one(index_rows, "R0")
    p_index = one(index_rows, "P")
    r0_ledger = one(ledger_rows, "R0")
    p_ledger = one(ledger_rows, "P")
    for row in (r0_index, p_index, r0_ledger, p_ledger):
        if row.get("accounting_schema") != ACCOUNTING_SCHEMA:
            raise ValueError("legacy head rows must use pre_registered_v1")
        if row.get("cohort_id") != COHORT_ID:
            raise ValueError("legacy head cohort mismatch")
    return r0_index, p_index, r0_ledger, p_ledger


def build_legacy_head_payload(
        root: Path, *, index_rows: Sequence[Mapping[str, Any]] | None = None,
        ledger_rows: Sequence[Mapping[str, Any]] | None = None,
        marker: Mapping[str, Any] | None = None,
        run_root: Path | None = None,
        marker_sha256: str | None = None,
        source_index_sha256: str | None = None,
        source_ledger_sha256: str | None = None,
        reservation_sha256: Mapping[str, str] | None = None) -> dict:
    """Build the immutable seq=1/2 head from the original v1 records.

    The current index/ledger are append-only views, so their whole-file SHA
    changes as soon as seq=3 is registered.  Callers building a snapshot pass
    the source copies' whole-file hashes; callers validating a snapshot pass
    the frozen values.  Row hashes are always recomputed from canonical JSON.
    """
    root = Path(root)
    marker = (dict(marker) if marker is not None
              else strict_read_json(_cohort_path(root)))
    if index_rows is None:
        index_rows = load_index(root)
    if ledger_rows is None:
        ledger_rows = load_ledger(root)
    r0_index, p_index, r0_ledger, p_ledger = _legacy_reference_rows(
        index_rows, ledger_rows)
    if marker_sha256 is None:
        marker_sha256 = sha256_file(_cohort_path(root))
    if source_index_sha256 is None:
        source_index_sha256 = sha256_file(_index_path(root))
    if source_ledger_sha256 is None:
        source_ledger_sha256 = sha256_file(_ledger_path(root))
    records = []
    for seq, (stage, attempt, index_row, ledger_row) in enumerate((
            ("R0", "attempt_001", r0_index, r0_ledger),
            ("P", "attempt_001", p_index, p_ledger)), start=1):
        if reservation_sha256 is not None:
            reservation_sha = reservation_sha256.get(stage)
        else:
            path = stage_dir(root, stage) / attempt / "call_reservations.json"
            reservation_sha = sha256_file(path) if path.is_file() else None
        if not isinstance(reservation_sha, str) or not re.fullmatch(
                r"[0-9a-f]{64}", reservation_sha):
            raise ValueError(f"legacy reservation hash missing: {stage}")
        records.append({
            "execution_seq": seq,
            "stage": stage,
            "attempt": attempt,
            "index_row_sha256": canonical_sha256(index_row),
            "ledger_row_sha256": canonical_sha256(ledger_row),
            "reservation_sha256": reservation_sha,
        })
    after = _call_counts_from_balance(p_ledger.get("call_balance"))
    if after is None:
        raise ValueError("legacy P ledger call_balance is invalid")
    index_after = _call_counts_from_balance(p_index.get("call_balance"))
    if index_after != after:
        raise ValueError("legacy P index/ledger call_balance disagree")
    marker_cohort = marker.get("cohort_id")
    marker_execution = marker.get("execution_id")
    if marker_cohort != COHORT_ID or marker_execution != EXECUTION_ID:
        raise ValueError("legacy marker identity mismatch")
    authorization = marker.get("authorization_sha256")
    if not isinstance(authorization, str) or not re.fullmatch(
            r"[0-9a-f]{64}", authorization):
        raise ValueError("legacy authorization hash missing")
    return {
        "schema": LEGACY_HEAD_SCHEMA,
        "cohort_id": marker_cohort,
        "execution_id": marker_execution,
        "run_root": str(Path(run_root or root).resolve()),
        "marker_sha256": marker_sha256,
        "authorization_sha256": authorization,
        "source_index_sha256": source_index_sha256,
        "source_ledger_sha256": source_ledger_sha256,
        "last_execution_seq": 2,
        "records": records,
        "call_balance_after": after,
    }


def legacy_head_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def _legacy_payload_problems(payload: Any, *, run_root: Path | None = None,
                             expected_sha256: str | None = None
                             ) -> list[str]:
    problems: list[str] = []
    if not isinstance(payload, Mapping):
        return ["legacy_head_not_a_mapping"]
    if payload.get("schema") != LEGACY_HEAD_SCHEMA:
        problems.append("legacy_head_schema")
    if payload.get("cohort_id") != COHORT_ID:
        problems.append("legacy_head_cohort")
    if payload.get("execution_id") != EXECUTION_ID:
        problems.append("legacy_head_execution")
    if run_root is not None:
        try:
            if Path(str(payload.get("run_root"))).resolve() != \
                    Path(run_root).resolve():
                problems.append("legacy_head_run_root")
        except OSError:
            problems.append("legacy_head_run_root")
    if payload.get("last_execution_seq") != 2:
        problems.append("legacy_head_last_seq")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2:
        problems.append("legacy_head_records")
    else:
        for index, want_stage in enumerate(("R0", "P"), start=1):
            rec = records[index - 1]
            if not isinstance(rec, Mapping):
                problems.append(f"legacy_head_record:{index}")
                continue
            if rec.get("execution_seq") != index:
                problems.append(f"legacy_head_seq:{index}")
            if rec.get("stage") != want_stage or \
                    rec.get("attempt") != "attempt_001":
                problems.append(f"legacy_head_stage:{index}")
            for key in ("index_row_sha256", "ledger_row_sha256",
                        "reservation_sha256"):
                if not isinstance(rec.get(key), str) or not re.fullmatch(
                        r"[0-9a-f]{64}", rec[key]):
                    problems.append(f"legacy_head_hash:{index}:{key}")
    after = _call_counts_from_balance({
        "schema": "w1_real_training_update_call_budget_v1",
        "counts": payload.get("call_balance_after"),
        "within_caps": True,
    }) if isinstance(payload.get("call_balance_after"), Mapping) else None
    if after is None:
        problems.append("legacy_head_call_balance")
    if expected_sha256 is not None and \
            legacy_head_sha256(payload) != expected_sha256:
        problems.append("legacy_head_sha256")
    return problems


def _current_legacy_rows_match(snapshot: Mapping[str, Any],
                               index_rows: Sequence[Mapping[str, Any]],
                               ledger_rows: Sequence[Mapping[str, Any]]
                               ) -> list[str]:
    payload = snapshot.get("legacy_head_payload")
    if not isinstance(payload, Mapping):
        return ["snapshot_legacy_head_missing"]
    records = payload.get("records")
    if not isinstance(records, list):
        return ["snapshot_legacy_head_records"]
    index_by = {(r.get("stage"), r.get("attempt")): r for r in index_rows}
    ledger_by = {(r.get("stage"), r.get("attempt")): r for r in ledger_rows}
    problems: list[str] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            problems.append("snapshot_legacy_record")
            continue
        key = (rec.get("stage"), rec.get("attempt"))
        row = index_by.get(key)
        entry = ledger_by.get(key)
        if row is None or entry is None:
            problems.append(f"legacy_record_missing:{key[0]}")
            continue
        if canonical_sha256(row) != rec.get("index_row_sha256"):
            problems.append(f"legacy_index_row_changed:{key[0]}")
        if canonical_sha256(entry) != rec.get("ledger_row_sha256"):
            problems.append(f"legacy_ledger_row_changed:{key[0]}")
    return problems


def _unsealed_artifacts(root: Path, index_rows: Sequence[Mapping[str, Any]]
                        ) -> list[dict]:
    sealed: set[str] = set()
    for row in index_rows:
        artifacts = row.get("artifacts")
        if isinstance(artifacts, Mapping):
            for rel in artifacts:
                try:
                    sealed.add(str((PROJECT_ROOT / str(rel)).resolve()))
                except OSError:
                    continue
    out = []
    for stage in STAGE_BUDGETS:
        base = stage_dir(root, stage)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            try:
                absolute = str(path.resolve())
            except OSError:
                continue
            if absolute not in sealed:
                out.append({
                    "path": path_for_record(path),
                    "sha256": sha256_file(path),
                    "bytes": int(path.stat().st_size),
                })
    return out


def build_history_snapshot(root: Path, *, project_root: Path = PROJECT_ROOT,
                           run_root: Path | None = None) -> dict:
    """Freeze the original v1 prefix and the pre-migration launch liabilities."""
    root = Path(root)
    source_index_bytes = _index_path(root).read_bytes()
    source_ledger_bytes = _ledger_path(root).read_bytes()
    source_marker_bytes = _cohort_path(root).read_bytes()
    index_rows = strict_json_loads(source_index_bytes.decode("utf-8")).get(
        "attempts")
    ledger_rows = strict_json_loads(source_ledger_bytes.decode("utf-8")).get(
        "invocations")
    if not isinstance(index_rows, list) or not isinstance(ledger_rows, list):
        raise ValueError("legacy index/ledger row lists missing")
    marker = strict_json_loads(source_marker_bytes.decode("utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("legacy cohort marker is not an object")
    legacy = build_legacy_head_payload(
        root, index_rows=index_rows, ledger_rows=ledger_rows, marker=marker,
        run_root=run_root or root,
        marker_sha256=sha256_bytes(source_marker_bytes),
        source_index_sha256=sha256_bytes(source_index_bytes),
        source_ledger_sha256=sha256_bytes(source_ledger_bytes))
    journal_dir = root / LAUNCH_JOURNAL_DIR
    journal_manifest, journal_problems = _manifest_files(journal_dir)
    files, file_problems = _manifest_files(
        root, skip=(LAUNCH_JOURNAL_DIR, BINDINGS_DIR, EXECUTION_EVENTS_DIR))
    stage_costs = {}
    for stage in STAGE_BUDGETS:
        rows = [r for r in ledger_rows if r.get("stage") == stage]
        stage_costs[stage] = float(sum(
            float(r.get("measured_wall_s", 0.0)) for r in rows))
    total_cost = float(sum(stage_costs.values()))
    attempts = {
        stage: sorted(str(r.get("attempt")) for r in index_rows
                      if r.get("stage") == stage)
        for stage in STAGE_BUDGETS
    }
    return {
        "schema": HISTORY_SNAPSHOT_SCHEMA,
        "run_root": str(root.resolve()),
        "run_root_realpath": str(root.resolve()),
        "cohort_id": marker.get("cohort_id"),
        "execution_id": marker.get("execution_id"),
        "marker_sha256": sha256_bytes(source_marker_bytes),
        "authorization_sha256": marker.get("authorization_sha256"),
        "source_index_sha256": sha256_bytes(source_index_bytes),
        "source_ledger_sha256": sha256_bytes(source_ledger_bytes),
        "source_marker_sha256": sha256_bytes(source_marker_bytes),
        "source_copies": {
            "index.json": {"sha256": sha256_bytes(source_index_bytes),
                            "bytes": len(source_index_bytes)},
            "ledger.json": {"sha256": sha256_bytes(source_ledger_bytes),
                             "bytes": len(source_ledger_bytes)},
            "cohort.json": {"sha256": sha256_bytes(source_marker_bytes),
                             "bytes": len(source_marker_bytes)},
        },
        "files": files,
        "files_problems": file_problems,
        "launch_journal": {
            "relative_path": LAUNCH_JOURNAL_DIR,
            "files": journal_manifest,
            "manifest_sha256": canonical_sha256(journal_manifest),
            "problems": journal_problems,
        },
        "index_rows": [{
            "stage": r.get("stage"), "attempt": r.get("attempt"),
            "canonical_sha256": canonical_sha256(r),
        } for r in index_rows],
        "ledger_rows": [{
            "stage": r.get("stage"), "attempt": r.get("attempt"),
            "canonical_sha256": canonical_sha256(r),
        } for r in ledger_rows],
        "execution_order": [
            {"execution_seq": rec["execution_seq"], "stage": rec["stage"],
             "attempt": rec["attempt"]}
            for rec in legacy["records"]
        ],
        "legacy_head_payload": legacy,
        "legacy_head_sha256": legacy_head_sha256(legacy),
        "cost_s": {"stages": stage_costs, "total": total_cost},
        "attempts": attempts,
        "updates": {
            "native_adamw_steps": int(sum(
                int(r.get("call_balance", {}).get("counts", {}).get(
                    "native_adamw_steps", 0))
                for r in ledger_rows[-1:])),
            "diagnostic_parameter_update_events": int(sum(
                int(r.get("call_balance", {}).get("counts", {}).get(
                    "diagnostic_parameter_update_events", 0))
                for r in ledger_rows[-1:])),
        },
        "call_balance_after": legacy["call_balance_after"],
        "unsealed_files": _unsealed_artifacts(root, index_rows),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_history_snapshot(out_dir: Path, root: Path
                           ) -> tuple[Path, str, dict]:
    out_dir = Path(out_dir)
    snapshot_path = out_dir / "history_snapshot.json"
    if snapshot_path.exists():
        raise FileExistsError(f"history snapshot already exists: {snapshot_path}")
    snapshot = build_history_snapshot(root)
    source = out_dir / "source"
    source.mkdir(parents=True, exist_ok=True)
    write_bytes_exclusive(source / "index.json", _index_path(root).read_bytes())
    write_bytes_exclusive(source / "ledger.json",
                          _ledger_path(root).read_bytes())
    write_bytes_exclusive(source / "cohort.json", _cohort_path(root).read_bytes())
    sha = write_json_exclusive(snapshot_path, snapshot)
    return snapshot_path, sha, snapshot


def _git_blob_sha256(project_root: Path, rel: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "show", f"HEAD:{rel}"],
            capture_output=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return sha256_bytes(result.stdout)


def build_protected_manifest(project_root: Path, run_root: Path,
                             *, synthetic_attempt: str = "attempt_013",
                             admission_attempt: str = "attempt_013"
                             ) -> dict:
    root = Path(project_root)
    entries: dict[str, dict] = {}

    def add(path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            return
        rel = path_for_record(path)
        if rel in entries:
            return
        entries[rel] = {"sha256": sha256_file(path),
                        "bytes": int(path.stat().st_size)}

    for path in (Path(run_root).rglob("*")):
        if path.is_file():
            add(path)
    audit = root / "derived/audit/w1_real_training_update_audit"
    for rel in ("authorization/attempt_012", "calibration/attempt_013",
                "admission_real/attempt_013"):
        for path in (audit / rel).rglob("*"):
            if path.is_file():
                add(path)
    synthetic = audit / "synthetic" / synthetic_attempt
    for path in synthetic.rglob("*"):
        if path.is_file() and path.suffix not in (
                ".npz", ".pt", ".png", ".log") and path.stat().st_size <= 8 * 2 ** 20:
            add(path)
    for rel in (
            "docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md",
            "docs/plans/omniretarget_w1_real_training_update_provider_execution_plan.md",
            "docs/plans/omniretarget_w1_real_training_update_alpha_fix_plan.md",
            str(MIGRATION_PLAN)):
        add(root / rel)
    preserved_path = (
        audit / "alpha_schedule_fix/attempt_001/preserved_files.json")
    preserved_62 = {}
    if preserved_path.is_file():
        try:
            preserved_record = strict_read_json(preserved_path)
        except (OSError, ValueError, TypeError):
            preserved_record = {}
        raw = preserved_record.get("protected_files")
        if isinstance(raw, Mapping):
            for rel in raw:
                preserved_62[rel] = raw[rel]
                add(root / str(rel))
    expected = [
        "src/ireset/eval/w1_real_training_update_audit.py",
        "src/ireset/eval/w1_real_training_update_provider.py",
        "scripts/audit_w1_real_training_update.py",
        "tests/test_w1_real_training_update_audit.py",
        "tests/test_w1_real_training_update_provider.py",
        "docs/plans/omniretarget_w1_real_training_update_binding_migration_plan.md",
    ]
    baseline = {}
    for rel in expected:
        path = root / rel
        baseline[rel] = {
            "head_sha256": _git_blob_sha256(root, rel),
            "current_sha256": sha256_file(path) if path.is_file() else None,
            "exists": path.is_file(),
        }
    return {
        "schema": "w1_real_training_update_protected_manifest_v1",
        "entries": entries,
        "entry_count": len(entries),
        "manifest_sha256": canonical_sha256(entries),
        "preserved_62": {
            "path": (path_for_record(preserved_path)
                     if preserved_path.is_file() else None),
            "sha256": (sha256_file(preserved_path)
                       if preserved_path.is_file() else None),
            "count": len(preserved_62),
            "covered": bool(preserved_62) and all(
                path_for_record(root / rel) in entries for rel in preserved_62),
        },
        "implementation_baseline": {
            "source_baseline_head": "f808aa975ba2ccbabdddba348f9cbeb2b85b9dfb",
            "expected_changed_files": expected,
            "files": baseline,
        },
        "large_artifact_policy": (
            "NPZ/PT/log bytes are protected by their owning sealed stage/row "
            "hashes; this manifest records metadata and small JSON only."),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_migration_baseline(out_dir: Path, project_root: Path,
                             run_root: Path) -> tuple[Path, str, dict]:
    out_dir = Path(out_dir)
    path = out_dir / "protected_manifest.json"
    if path.exists():
        raise FileExistsError(f"protected manifest already exists: {path}")
    record = build_protected_manifest(project_root, run_root)
    sha = write_json_exclusive(path, record)
    return path, sha, record


def _snapshot_problems(snapshot: Mapping[str, Any] | None, *,
                       run_root: Path | None = None,
                       root: Path | None = None) -> list[str]:
    if not isinstance(snapshot, Mapping):
        return ["snapshot_not_a_mapping"]
    problems: list[str] = []
    if snapshot.get("schema") != HISTORY_SNAPSHOT_SCHEMA:
        problems.append("snapshot_schema")
    if snapshot.get("cohort_id") != COHORT_ID:
        problems.append("snapshot_cohort")
    if snapshot.get("execution_id") != EXECUTION_ID:
        problems.append("snapshot_execution")
    if run_root is not None:
        try:
            if Path(str(snapshot.get("run_root"))).resolve() != \
                    Path(run_root).resolve():
                problems.append("snapshot_run_root")
        except OSError:
            problems.append("snapshot_run_root")
    payload = snapshot.get("legacy_head_payload")
    problems += _legacy_payload_problems(
        payload, run_root=run_root,
        expected_sha256=snapshot.get("legacy_head_sha256"))
    if isinstance(payload, Mapping) and snapshot.get("legacy_head_sha256") != \
            legacy_head_sha256(payload):
        problems.append("snapshot_legacy_head_binding")
    if isinstance(payload, Mapping) and snapshot.get("authorization_sha256") \
            != payload.get("authorization_sha256"):
        problems.append("snapshot_authorization_binding")
    # The frozen file list covers the marker, the authorization-bearing
    # marker and every old artifact.  The index/ledger are the two files the
    # migration legitimately appends to; their old rows are checked below by
    # canonical row hash while every other file must remain byte identical.
    files = snapshot.get("files")
    snapshot_root = _reference_path(snapshot.get("run_root"))
    target_root = Path(run_root).resolve() if run_root is not None else \
        (snapshot_root.resolve() if snapshot_root is not None else None)
    if not isinstance(files, Mapping) or not files:
        problems.append("snapshot_files_missing")
    else:
        if "cohort.json" not in files:
            problems.append("snapshot_marker_file_missing")
        for recorded in snapshot.get("files_problems") or []:
            problems.append(f"snapshot_files_recorded:{recorded}")
        for rel, rec in files.items():
            path = Path(str(rel))
            if rel in ("attempts_index.json", "costs.json"):
                continue
            if target_root is None:
                problems.append("snapshot_files_root")
                break
            path = target_root / path
            if not isinstance(rec, Mapping) or not isinstance(
                    rec.get("sha256"), str):
                problems.append(f"snapshot_file_entry:{rel}")
                continue
            if not path.is_file():
                problems.append(f"snapshot_file_missing:{rel}")
                continue
            try:
                actual = sha256_file(path)
                size = int(path.stat().st_size)
            except OSError:
                problems.append(f"snapshot_file_unreadable:{rel}")
                continue
            if actual != rec.get("sha256"):
                problems.append(f"snapshot_file_sha:{rel}")
            if type(rec.get("bytes")) is not int or int(rec["bytes"]) != size:
                problems.append(f"snapshot_file_bytes:{rel}")
    copies = snapshot.get("source_copies")
    copies = copies if isinstance(copies, Mapping) else {}

    def copy_sha(name: str) -> Any:
        rec = copies.get(name)
        return rec.get("sha256") if isinstance(rec, Mapping) else None

    if snapshot.get("marker_sha256") != copy_sha("cohort.json"):
        problems.append("snapshot_marker_binding")
    if snapshot.get("source_index_sha256") != copy_sha("index.json"):
        problems.append("snapshot_index_binding")
    if snapshot.get("source_ledger_sha256") != copy_sha("ledger.json"):
        problems.append("snapshot_ledger_binding")
    if root is not None:
        try:
            problems += _current_legacy_rows_match(
                snapshot, load_index(root), load_ledger(root))
        except (OSError, ValueError, TypeError):
            problems.append("snapshot_current_rows_unreadable")
        if isinstance(payload, Mapping):
            attempts_record = snapshot.get("attempts")
            attempts_record = (attempts_record
                               if isinstance(attempts_record, Mapping) else {})
            for stage in STAGE_BUDGETS:
                attempts_for_stage = attempts_record.get(stage, [])
                if not isinstance(attempts_for_stage, list):
                    problems.append(f"snapshot_attempts:{stage}")
                    continue
                for attempt in attempts_for_stage:
                    path = stage_dir(root, stage) / attempt / \
                        "call_reservations.json"
                    if not path.is_file():
                        problems.append(f"snapshot_reservation_missing:{stage}")
                        continue
                    rec = next((r for r in payload.get("records", [])
                                if r.get("stage") == stage
                                and r.get("attempt") == attempt), None)
                    if rec is not None and sha256_file(path) != rec.get(
                            "reservation_sha256"):
                        problems.append(
                            f"snapshot_reservation_changed:{stage}")
    return problems


def _budget_from_snapshot(snapshot: Mapping[str, Any], stage: str) -> dict:
    if stage not in STAGE_BUDGETS:
        raise ValueError(f"unknown stage {stage}")
    costs = (snapshot.get("cost_s") or {}).get("stages") or {}
    used_s = float(costs.get(stage, 0.0))
    used_n = len((snapshot.get("attempts") or {}).get(stage) or [])
    spec = STAGE_BUDGETS[stage]
    remaining = float(spec["budget_s"]) - used_s
    return {
        "stage": stage, "budget_s": float(spec["budget_s"]),
        "reserve_s": float(spec["reserve_s"]),
        "attempts": int(spec["attempts"]), "attempts_used": used_n,
        "local_stage_s": used_s, "prior_stage_s": 0.0, "prior_total_s": 0.0,
        "remaining_s": remaining,
    }


def build_binding_spec(project_root: Path, snapshot_path: Path,
                       snapshot_sha256: str, snapshot: Mapping[str, Any],
                       *, generation: str = GENERATION_002,
                       invocation_id: str | None = None,
                       plan_path: Path = MIGRATION_PLAN,
                       gate_b: Mapping[str, Any] | None = None,
                       admission: Mapping[str, Any] | None = None,
                       implementation: Mapping[str, Any] | None = None,
                       runtime: Mapping[str, Any] | None = None) -> dict:
    root = Path(project_root)
    plan = Path(plan_path)
    if not plan.is_absolute():
        plan = root / plan
    impl = dict(implementation or implementation_identity(root))
    runtime_record = dict(runtime or {
        "identity": runtime_identity_facts(),
        "requirements": dict(RUNTIME_REQUIREMENTS),
    })
    journal = snapshot.get("launch_journal") or {}
    budgets = {stage: _budget_from_snapshot(snapshot, stage)
               for stage in STAGE_BUDGETS}
    return {
        "schema": BINDING_SPEC_SCHEMA,
        "generation": generation,
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "run_root": str(Path(str(snapshot.get("run_root"))).resolve()),
        "run_root_realpath": str(Path(str(snapshot.get("run_root"))).resolve()),
        "history_snapshot": {
            "path": path_for_record(snapshot_path),
            "sha256": snapshot_sha256,
        },
        "legacy_head_sha256": snapshot.get("legacy_head_sha256"),
        "first_generation": {
            "generation": GENERATION_001,
            "marker_sha256": snapshot.get("marker_sha256"),
            "authorization_sha256": snapshot.get("authorization_sha256"),
            "last_execution_seq": 2,
        },
        "target_generation": {
            "generation": generation,
            "cohort_id": COHORT_ID,
            "execution_id": EXECUTION_ID,
            "run_root_realpath": str(Path(str(
                snapshot.get("run_root"))).resolve()),
        },
        "stage_plan": [dict(item) for item in MIGRATION_STAGE_PLAN],
        "accounting": {
            "schema": ACCOUNTING_SCHEMA_V2,
            "prior_updates": MIGRATION_PRIOR_UPDATES,
            "incremental_update_limit": MIGRATION_INCREMENTAL_UPDATE_LIMIT,
            "cohort_update_limit_for_this_authorization":
                MIGRATION_COHORT_UPDATE_LIMIT,
            "stage_attempts": dict(MIGRATION_STAGE_ATTEMPTS),
        },
        "budgets": budgets,
        "launch_journal": {
            "relative_path": LAUNCH_JOURNAL_DIR,
            "manifest_sha256": journal.get("manifest_sha256"),
            "files": dict(journal.get("files") or {}),
            "fixed": True,
        },
        "bindings": {
            "plan": {"path": path_for_record(plan),
                     "sha256": sha256_file(plan) if plan.is_file() else None},
            "implementation": impl,
            "runtime": runtime_record,
            "gate_b": dict(gate_b or {}),
            "real_admission": dict(admission or {}),
        },
        "protocol": {
            "publication": ("prepare_revision -> internal prepared verification "
                            "-> stable cost -> one finish + commit -> "
                            "stage_closed candidate -> actual write-after gate"),
            "write_after": {
                "sample": "post_publish_perf_counter - stage_t0",
                "requires": ["finite", "non_negative", "elapsed<remaining_s",
                              "elapsed<=committed_measured_wall_s"],
            },
            "return_observation": {
                "schema": RETURN_OBSERVATION_SCHEMA,
                "source": "outer_wait_or_exec_actual_return",
                "scope": "post_execution_observation",
                "required_for_public_consumption": True,
            },
            "execution_window": {
                "starts_at": "launch_claim.t0",
                "ends_at": "outer actual process return",
                "post_return_observation_can_execute_work": False,
            },
        },
        "invocation": {"invocation_id": invocation_id},
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def binding_spec_problems(spec: Mapping[str, Any] | None, *, root: Path,
                          snapshot: Mapping[str, Any] | None = None,
                          require_current_implementation: bool = False,
                          project_root: Path = PROJECT_ROOT
                          ) -> list[str]:
    if not isinstance(spec, Mapping):
        return ["binding_spec_not_a_mapping"]
    problems: list[str] = []
    if spec.get("schema") != BINDING_SPEC_SCHEMA:
        problems.append("binding_spec_schema")
    if spec.get("generation") != GENERATION_002:
        problems.append("binding_spec_generation")
    if spec.get("cohort_id") != COHORT_ID:
        problems.append("binding_spec_cohort")
    if spec.get("execution_id") != EXECUTION_ID:
        problems.append("binding_spec_execution")
    try:
        if Path(str(spec.get("run_root_realpath"))).resolve() != \
                Path(root).resolve():
            problems.append("binding_spec_run_root")
    except OSError:
        problems.append("binding_spec_run_root")
    if spec.get("stage_plan") != list(MIGRATION_STAGE_PLAN):
        problems.append("binding_spec_stage_plan")
    accounting = spec.get("accounting") or {}
    expected_accounting = {
        "schema": ACCOUNTING_SCHEMA_V2,
        "prior_updates": MIGRATION_PRIOR_UPDATES,
        "incremental_update_limit": MIGRATION_INCREMENTAL_UPDATE_LIMIT,
        "cohort_update_limit_for_this_authorization":
            MIGRATION_COHORT_UPDATE_LIMIT,
        "stage_attempts": dict(MIGRATION_STAGE_ATTEMPTS),
    }
    if _jsonable(accounting) != _jsonable(expected_accounting):
        problems.append("binding_spec_accounting")
    journal = spec.get("launch_journal") or {}
    if journal.get("relative_path") != LAUNCH_JOURNAL_DIR or not \
            journal.get("fixed"):
        problems.append("binding_spec_launch_journal")
    if not isinstance(journal.get("manifest_sha256"), str):
        problems.append("binding_spec_launch_manifest")
    bindings = spec.get("bindings") or {}
    plan = bindings.get("plan") or {}
    if not isinstance(plan.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", plan["sha256"]):
        problems.append("binding_spec_plan")
    impl = bindings.get("implementation") or {}
    if not isinstance(impl.get("files"), Mapping) or not impl["files"]:
        problems.append("binding_spec_implementation")
    elif require_current_implementation:
        if impl.get("files") != implementation_identity(
                project_root).get("files"):
            problems.append("binding_spec_implementation_drift")
    if not isinstance(bindings.get("gate_b"), Mapping):
        problems.append("binding_spec_gate_b")
    if not isinstance(bindings.get("real_admission"), Mapping):
        problems.append("binding_spec_admission")
    runtime = bindings.get("runtime") or {}
    if not runtime.get("identity") or not runtime.get("requirements"):
        problems.append("binding_spec_runtime")
    if snapshot is not None:
        if spec.get("legacy_head_sha256") != snapshot.get("legacy_head_sha256"):
            problems.append("binding_spec_legacy_head")
        for stage, budget in (spec.get("budgets") or {}).items():
            want = _budget_from_snapshot(snapshot, stage)
            if _jsonable(budget) != _jsonable(want):
                problems.append(f"binding_spec_budget:{stage}")
    return problems


@dataclass(frozen=True)
class BindingContext:
    generation: str
    spec: dict
    spec_sha256: str
    snapshot: dict
    snapshot_sha256: str
    snapshot_path: Path
    binding_dir: Path
    legacy_head: dict
    legacy_head_sha256: str
    activation: dict | None = None
    activation_sha256: str | None = None


def _binding_dir(root: Path, generation: str = GENERATION_002) -> Path:
    return Path(root) / BINDINGS_DIR / generation


def load_binding(root: Path, *, generation: str = GENERATION_002,
                 require_active: bool = False) -> tuple[BindingContext | None,
                                                        list[str]]:
    root = Path(root)
    directory = _binding_dir(root, generation)
    intent_path = directory / "intent.json"
    spec_path = directory / "binding_spec.json"
    if not intent_path.is_file() or not spec_path.is_file():
        if require_active:
            return None, ["binding_missing"]
        return None, []
    problems: list[str] = []
    try:
        intent = strict_read_json(intent_path)
        spec = strict_read_json(spec_path)
    except (OSError, ValueError, TypeError) as exc:
        return None, [f"binding_unreadable:{type(exc).__name__}"]
    if intent.get("schema") != BINDING_INTENT_SCHEMA:
        problems.append("binding_intent_schema")
    if intent.get("generation") != generation:
        problems.append("binding_intent_generation")
    spec_sha = sha256_file(spec_path)
    if intent.get("spec_sha256") != spec_sha:
        problems.append("binding_spec_sha")
    snapshot = spec.get("history_snapshot") or {}
    snapshot_path = _reference_path(snapshot.get("path"))
    if snapshot_path is None or not snapshot_path.is_file():
        return None, problems + ["binding_snapshot_missing"]
    snapshot_sha = sha256_file(snapshot_path)
    if snapshot.get("sha256") != snapshot_sha or \
            intent.get("snapshot_sha256") != snapshot_sha:
        problems.append("binding_snapshot_sha")
    try:
        snapshot_data = strict_read_json(snapshot_path)
    except (OSError, ValueError, TypeError) as exc:
        return None, problems + [f"binding_snapshot_unreadable:"
                                 f"{type(exc).__name__}"]
    problems += _snapshot_problems(snapshot_data, run_root=root, root=root)
    source_dir = snapshot_path.parent / "source"
    for name in ("index.json", "ledger.json", "cohort.json"):
        expected = ((snapshot_data.get("source_copies") or {}).get(name)
                    or {}).get("sha256")
        path = source_dir / name
        if not path.is_file():
            problems.append(f"binding_source_copy_missing:{name}")
        elif sha256_file(path) != expected:
            problems.append(f"binding_source_copy_sha:{name}")
    problems += binding_spec_problems(spec, root=root, snapshot=snapshot_data)
    payload = snapshot_data.get("legacy_head_payload")
    if not isinstance(payload, Mapping):
        return None, problems + ["binding_legacy_head_missing"]
    activation = None
    activation_sha = None
    activation_path = directory / "activation.json"
    if activation_path.is_file():
        try:
            activation = strict_read_json(activation_path)
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"binding_activation_unreadable:"
                            f"{type(exc).__name__}")
        else:
            activation_sha = sha256_file(activation_path)
            problems += _activation_problems(
                activation, root=root, spec_sha256=spec_sha,
                snapshot_sha256=snapshot_sha,
                legacy_head_sha256=snapshot_data.get("legacy_head_sha256"),
                generation=generation)
    elif require_active:
        problems.append("binding_activation_missing")
    if problems:
        return None, problems
    return BindingContext(
        generation=generation, spec=spec, spec_sha256=spec_sha,
        snapshot=snapshot_data, snapshot_sha256=snapshot_sha,
        snapshot_path=snapshot_path, binding_dir=directory,
        legacy_head=dict(payload),
        legacy_head_sha256=str(snapshot_data.get("legacy_head_sha256")),
        activation=activation, activation_sha256=activation_sha), []


def load_external_binding_material(
        snapshot_path: Path, snapshot_sha256: str,
        spec_path: Path, spec_sha256: str, *,
        project_root: Path = PROJECT_ROOT,
        require_current_implementation: bool = False
        ) -> tuple[BindingContext | None, list[str]]:
    """Build a read-only context from external snapshot/spec bytes.

    Authorization drafting happens before the original root may contain a
    binding intent/spec.  This loader therefore verifies the external
    snapshot and spec directly and never writes to the run root.
    """
    snapshot_path = Path(snapshot_path).resolve()
    spec_path = Path(spec_path).resolve()
    problems: list[str] = []
    if not snapshot_path.is_file():
        return None, ["external_snapshot_missing"]
    if not spec_path.is_file():
        return None, ["external_spec_missing"]
    if sha256_file(snapshot_path) != snapshot_sha256:
        problems.append("external_snapshot_sha")
    if sha256_file(spec_path) != spec_sha256:
        problems.append("external_spec_sha")
    try:
        snapshot = strict_read_json(snapshot_path)
        spec = strict_read_json(spec_path)
    except (OSError, ValueError, TypeError) as exc:
        return None, problems + [f"external_material_unreadable:"
                                 f"{type(exc).__name__}"]
    if not isinstance(snapshot, Mapping) or not isinstance(spec, Mapping):
        return None, problems + ["external_material_not_mapping"]
    reference = spec.get("history_snapshot") or {}
    ref_path = _reference_path(reference.get("path"))
    if ref_path is None or ref_path.resolve() != snapshot_path:
        problems.append("external_spec_snapshot_path")
    if reference.get("sha256") != snapshot_sha256:
        problems.append("external_spec_snapshot_sha")
    run_root = Path(str(snapshot.get("run_root"))).expanduser().resolve()
    problems += _snapshot_problems(snapshot, run_root=run_root, root=run_root)
    problems += binding_spec_problems(
        spec, root=run_root, snapshot=snapshot,
        require_current_implementation=require_current_implementation,
        project_root=project_root)
    payload = snapshot.get("legacy_head_payload")
    if not isinstance(payload, Mapping):
        problems.append("external_legacy_head_missing")
        return None, problems
    return BindingContext(
        generation=str(spec.get("generation", GENERATION_002)),
        spec=dict(spec), spec_sha256=spec_sha256,
        snapshot=dict(snapshot), snapshot_sha256=snapshot_sha256,
        snapshot_path=snapshot_path, binding_dir=spec_path.parent,
        legacy_head=dict(payload),
        legacy_head_sha256=str(snapshot.get("legacy_head_sha256")),
        activation=None, activation_sha256=None), problems


def _activation_problems(activation: Mapping[str, Any], *, root: Path,
                         spec_sha256: str, snapshot_sha256: str,
                         legacy_head_sha256: str | None,
                         generation: str) -> list[str]:
    problems: list[str] = []
    if activation.get("schema") != BINDING_ACTIVATION_SCHEMA:
        problems.append("activation_schema")
    if activation.get("generation") != generation:
        problems.append("activation_generation")
    if activation.get("spec_sha256") != spec_sha256:
        problems.append("activation_spec")
    if activation.get("snapshot_sha256") != snapshot_sha256:
        problems.append("activation_snapshot")
    if activation.get("legacy_head_sha256") != legacy_head_sha256:
        problems.append("activation_legacy_head")
    try:
        if Path(str(activation.get("run_root_realpath"))).resolve() != \
                Path(root).resolve():
            problems.append("activation_run_root")
    except OSError:
        problems.append("activation_run_root")
    commit = activation.get("commit") or {}
    if not isinstance(commit, Mapping):
        problems.append("activation_commit")
        return problems
    path = _reference_path(commit.get("path"))
    if path is None or not path.is_file():
        problems.append("activation_commit_missing")
    elif sha256_file(path) != commit.get("sha256"):
        problems.append("activation_commit_sha")
    r0 = activation.get("r0") or {}
    for key in ("registration_sha256", "admission_sha256",
                "noise_schedule_sha256", "reservation_sha256"):
        if not isinstance(r0.get(key), str) or not re.fullmatch(
                r"[0-9a-f]{64}", r0[key]):
            problems.append(f"activation_r0:{key}")
    if activation.get("effective") is not True:
        problems.append("activation_not_effective")
    return problems


def stage_binding(root: Path, *, spec_path: Path, spec_sha256: str,
                  snapshot_sha256: str, invocation_id: str,
                  generation: str = GENERATION_002) -> Path:
    """Publish the byte-exact spec copy and intent; no activation yet."""
    directory = _binding_dir(root, generation)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"binding directory is not fresh: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "binding_spec.json"
    source = Path(spec_path)
    target.write_bytes(source.read_bytes())
    if sha256_file(target) != spec_sha256:
        raise ValueError("binding spec copy sha mismatch")
    _fsync_directory(directory)
    write_json_exclusive(directory / "intent.json", {
        "schema": BINDING_INTENT_SCHEMA,
        "generation": generation,
        "run_root_realpath": str(Path(root).resolve()),
        "spec_sha256": spec_sha256,
        "snapshot_sha256": snapshot_sha256,
        "invocation_id": invocation_id,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return target


def activate_binding(root: Path, *, generation: str, commit_path: Path,
                     commit_sha256: str, r0: Mapping[str, Any],
                     invocation_id: str) -> Path:
    directory = _binding_dir(root, generation)
    intent = strict_read_json(directory / "intent.json")
    spec_path = directory / "binding_spec.json"
    activation_path = directory / "activation.json"
    payload = {
        "schema": BINDING_ACTIVATION_SCHEMA,
        "generation": generation,
        "run_root_realpath": str(Path(root).resolve()),
        "spec_sha256": sha256_file(spec_path),
        "snapshot_sha256": intent.get("snapshot_sha256"),
        "legacy_head_sha256": (strict_read_json(
            _reference_path((strict_read_json(spec_path).get(
                "history_snapshot") or {}).get("path"))).get(
                    "legacy_head_sha256")),
        "invocation_id": invocation_id,
        "commit": {"path": path_for_record(Path(commit_path)),
                   "sha256": commit_sha256},
        "r0": {k: r0[k] for k in ("registration_sha256",
                                   "admission_sha256",
                                   "noise_schedule_sha256",
                                   "reservation_sha256")},
        "effective": True,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_exclusive(activation_path, payload)
    return activation_path


# ---------------------------------------------------------------------------
# launch journal and return observation
# ---------------------------------------------------------------------------


def launch_journal_dir(root: Path) -> Path:
    return Path(root) / LAUNCH_JOURNAL_DIR


_RETURN_TIMING_TOL = 1e-6


def _return_timing_coverage_problems(
        record: Mapping[str, Any], *,
        claim: Mapping[str, Any] | None = None) -> list[str]:
    """Derive return coverage from stage timings and committed costs.

    ``coverage_ok`` in the record is only a caller statement; the public
    reader recomputes the relation between the outer return time, each
    stage's inner start/end facts and the commit's billed wall time.
    """
    timings = record.get("stage_timings")
    if not isinstance(timings, Mapping) or set(timings) != set(STAGE_BUDGETS):
        return ["return_stage_timings"]
    commits = record.get("commits")
    if not isinstance(commits, Mapping) or \
            set(commits) != set(STAGE_BUDGETS):
        return ["return_commits_scope"]
    problems: list[str] = []
    parsed: dict[str, tuple[float, float, float]] = {}
    for stage in STAGE_BUDGETS:
        timing = timings.get(stage)
        ref = commits.get(stage)
        if not isinstance(timing, Mapping) or not isinstance(ref, Mapping):
            problems.append(f"return_stage_timing:{stage}")
            continue
        t0 = timing.get("t0_perf_counter")
        end = timing.get("end_perf_counter")
        if not _nonneg(t0) or not _nonneg(end):
            problems.append(f"return_stage_timing:{stage}")
            continue
        t0 = float(t0)
        end = float(end)
        if end + _RETURN_TIMING_TOL < t0:
            problems.append(f"return_stage_time_reverse:{stage}")
            continue
        path = _reference_path(ref.get("path"))
        if path is None or not path.is_file():
            problems.append(f"return_stage_commit:{stage}")
            continue
        try:
            if sha256_file(path) != ref.get("sha256"):
                problems.append(f"return_stage_commit_sha:{stage}")
                continue
            commit = strict_read_json(path)
        except (OSError, ValueError, TypeError):
            problems.append(f"return_stage_commit:{stage}")
            continue
        cost = commit.get("committed_measured_wall_s")
        commit_t0 = commit.get("stage_t0_perf_counter")
        if not _nonneg(cost) or not _nonneg(commit_t0):
            problems.append(f"return_stage_commit_facts:{stage}")
            continue
        if abs(float(commit_t0) - t0) > _RETURN_TIMING_TOL:
            problems.append(f"return_stage_t0:{stage}")
        parsed[stage] = (t0, end, float(cost))
    if set(parsed) != set(STAGE_BUDGETS):
        return problems
    stages = tuple(STAGE_BUDGETS)
    for previous, current in zip(stages, stages[1:]):
        if parsed[current][0] + _RETURN_TIMING_TOL < parsed[previous][1]:
            problems.append(f"return_stage_overlap:{previous}:{current}")
        elif parsed[current][0] - _RETURN_TIMING_TOL > parsed[previous][1]:
            problems.append(f"return_stage_gap:{previous}:{current}")
    finished = record.get("finished_perf_counter")
    started = record.get("started_perf_counter")
    if not _nonneg(finished):
        problems.append("return_time:finished_perf_counter")
    else:
        finished = float(finished)
        if parsed["F"][0] + parsed["F"][2] + _RETURN_TIMING_TOL < finished:
            problems.append("return_coverage:F")
        last_end = record.get("last_stage_end_perf_counter")
        if not _nonneg(last_end) or \
                float(last_end) > finished + _RETURN_TIMING_TOL:
            problems.append("return_time_order")
        elif abs(float(last_end) - parsed["F"][1]) > _RETURN_TIMING_TOL:
            problems.append("return_last_stage_end")
        total = record.get("total_wall_s")
        if _nonneg(started) and _nonneg(total):
            if abs(float(total) - (finished - float(started))) > \
                    _RETURN_TIMING_TOL:
                problems.append("return_total_wall")
        else:
            problems.append("return_time:total_wall_s")
    if claim is not None and _nonneg(claim.get("t0_perf_counter")) and \
            _nonneg(started):
        if abs(float(started) - float(claim["t0_perf_counter"])) > \
                _RETURN_TIMING_TOL:
            problems.append("return_started")
    for stage in ("R0", "P", "I"):
        t0, end, cost = parsed[stage]
        if end - t0 > cost + _RETURN_TIMING_TOL:
            problems.append(f"return_coverage:{stage}")
    if claim is not None and _nonneg(claim.get("t0_perf_counter")):
        if abs(float(claim["t0_perf_counter"]) - parsed["R0"][0]) > \
                _RETURN_TIMING_TOL:
            problems.append("return_claim_t0")
    return problems


def _claim_problems(claim: Mapping[str, Any], *, invocation_id: str,
                    root: Path) -> list[str]:
    problems: list[str] = []
    if claim.get("schema") != LAUNCH_CLAIM_SCHEMA:
        problems.append("claim_schema")
    if claim.get("invocation_id") != invocation_id:
        problems.append("claim_invocation")
    if claim.get("cohort_id") != COHORT_ID or \
            claim.get("execution_id") != EXECUTION_ID:
        problems.append("claim_cohort")
    try:
        if Path(str(claim.get("run_root_realpath"))).resolve() != \
                Path(root).resolve():
            problems.append("claim_run_root")
    except OSError:
        problems.append("claim_run_root")
    for key in ("authorization_sha256", "spec_sha256",
                "legacy_head_sha256", "marker_sha256"):
        if not isinstance(claim.get(key), str) or not re.fullmatch(
                r"[0-9a-f]{64}", claim[key]):
            problems.append(f"claim_hash:{key}")
    if not _nonneg(claim.get("t0_perf_counter")):
        problems.append("claim_t0")
    target = claim.get("target") or {}
    if target.get("stage") != "R0" or \
            target.get("attempt") != "attempt_002" or \
            target.get("execution_seq") != 3:
        problems.append("claim_target")
    if type(claim.get("pid")) is not int or int(claim.get("pid")) <= 0:
        problems.append("claim_pid")
    if type(claim.get("ppid")) is not int or int(claim.get("ppid")) < 0:
        problems.append("claim_ppid")
    return problems


def return_observation_problems(record: Mapping[str, Any] | None, *,
                                invocation_id: str | None = None,
                                claim: Mapping[str, Any] | None = None
                                ) -> list[str]:
    if not isinstance(record, Mapping):
        return ["return_observation_missing"]
    problems: list[str] = []
    if record.get("schema") != RETURN_OBSERVATION_SCHEMA:
        problems.append("return_schema")
    if record.get("scope") != "post_execution_observation":
        problems.append("return_scope")
    if record.get("process_returned") is not True:
        problems.append("return_process_not_returned")
    if invocation_id is not None and \
            record.get("invocation_id") != invocation_id:
        problems.append("return_invocation")
    source = record.get("source")
    if not isinstance(source, Mapping) or source.get("kind") not in (
            "outer_wait", "outer_exec"):
        problems.append("return_source")
    else:
        if type(source.get("exit_code")) is not int:
            problems.append("return_exit_code")
        if source.get("signal") is not None and type(source.get("signal")) is not int:
            problems.append("return_signal")
        if source.get("pid") != record.get("pid"):
            problems.append("return_pid")
        if source.get("ppid") != record.get("ppid"):
            problems.append("return_ppid")
        if record.get("expected_exit_code") is not None:
            if type(record.get("expected_exit_code")) is not int or \
                    source.get("exit_code") != record["expected_exit_code"]:
                problems.append("return_expected_exit_code")
    expected_by_conclusion = {
        "real_training_update_covered_contract_passed": 0,
        "real_training_update_mismatch": 2,
        "audit_incomplete": 3,
    }
    if record.get("conclusion") in expected_by_conclusion and \
            isinstance(source, Mapping) and \
            source.get("exit_code") != expected_by_conclusion[
                record["conclusion"]]:
        problems.append("return_conclusion_exit_code")
    for key in ("started_perf_counter", "finished_perf_counter",
                "last_stage_end_perf_counter", "total_wall_s"):
        if not _nonneg(record.get(key)):
            problems.append(f"return_time:{key}")
    if _nonneg(record.get("finished_perf_counter")) and _nonneg(
            record.get("last_stage_end_perf_counter")) and float(
                record["finished_perf_counter"]) + 1e-9 < float(
                    record["last_stage_end_perf_counter"]):
        problems.append("return_time_order")
    if _nonneg(record.get("finished_perf_counter")) and _nonneg(
            record.get("started_perf_counter")) and float(
                record["finished_perf_counter"]) + 1e-9 < float(
                    record["started_perf_counter"]):
        problems.append("return_time_reverse")
    write_after = record.get("write_after")
    if not isinstance(write_after, Mapping) or not write_after:
        problems.append("return_write_after_missing")
    else:
        if set(STAGE_BUDGETS) - set(write_after) or \
                "terminal" not in write_after:
            problems.append("return_write_after_scope")
        if not all(value is True for value in write_after.values()):
            problems.append("return_write_after_failed")
    for label in ("commits", "candidate_receipts"):
        refs = record.get(label)
        if not isinstance(refs, Mapping) or set(STAGE_BUDGETS) - set(refs):
            problems.append(f"return_{label}_scope")
            continue
        for stage, ref in refs.items():
            if not isinstance(ref, Mapping):
                problems.append(f"return_{label}:{stage}")
                continue
            path = _reference_path(ref.get("path"))
            if path is None or not path.is_file():
                problems.append(f"return_{label}_missing:{stage}")
            elif sha256_file(path) != ref.get("sha256"):
                problems.append(f"return_{label}_sha:{stage}")
    coverage_problems = _return_timing_coverage_problems(record, claim=claim)
    problems += coverage_problems
    if coverage_problems or record.get("coverage_ok") is not True:
        problems.append("return_coverage")
    if record.get("post_return_work_performed") is not False:
        problems.append("return_post_work")
    if claim is not None:
        for key in ("authorization_sha256", "spec_sha256",
                    "legacy_head_sha256"):
            if record.get(key) != claim.get(key):
                problems.append(f"return_claim_binding:{key}")
        if record.get("claim_sha256") != claim.get("_sha256"):
            problems.append("return_claim_sha")
    return problems


_TERMINAL_EXECUTION_SEQ = int(MIGRATION_STAGE_PLAN[-1]["execution_seq"])
_TERMINAL_STAGES = tuple(
    item["stage"] for item in MIGRATION_STAGE_PLAN
    if int(item["execution_seq"]) == _TERMINAL_EXECUTION_SEQ)


def _terminal_write_after_problems(path: Path, *, expected_sha: Any,
                                   label: str) -> list[str]:
    """A terminal candidate is only closed by its own sampled write proof."""
    if not path.is_file():
        return [f"{label}_write_after_missing"]
    try:
        proof = strict_read_json(path)
    except (OSError, ValueError, TypeError):
        return [f"{label}_write_after_unreadable"]
    problems: list[str] = []
    if proof.get("schema") != EVENT_SCHEMA or \
            proof.get("kind") != "terminal_write_after" or \
            proof.get("passed") is not True:
        problems.append(f"{label}_write_after_binding")
    if not isinstance(expected_sha, str) or \
            proof.get("candidate_sha256") != expected_sha:
        problems.append(f"{label}_write_after_sha")
    observed = proof.get("observed_elapsed_s")
    committed = proof.get("committed_measured_wall_s")
    remaining = proof.get("remaining_s")
    if not (_nonneg(observed) and _nonneg(committed) and _nonneg(remaining)):
        problems.append(f"{label}_write_after_facts")
    elif float(observed) > float(committed) or \
            float(observed) >= float(remaining):
        problems.append(f"{label}_write_after_gate")
    return problems


def _terminal_closure_problems(directory: Path, *,
                               claim: Mapping[str, Any]) -> list[str]:
    """Parse the launch/event terminal pair and both write-after proofs.

    ``write_launch_terminal_candidate`` writes a launch-journal candidate that
    points at the event-directory terminal candidate; each side writes its own
    sampled ``terminal_write_after`` proof.  Existence alone is not closure:
    the candidates, their commit/event references and both proofs must agree
    with the claim and with the expected F execution sequence.
    """
    candidate_path = directory / "terminal_candidate.json"
    if not candidate_path.is_file():
        return ["terminal_candidate_missing"]
    try:
        candidate = strict_read_json(candidate_path)
        candidate_sha = sha256_file(candidate_path)
    except (OSError, ValueError, TypeError):
        return ["terminal_candidate_unreadable"]
    problems: list[str] = []
    if candidate.get("schema") != EVENT_SCHEMA or \
            candidate.get("kind") != "terminal_candidate" or \
            candidate.get("candidate") is not True:
        problems.append("terminal_candidate_binding")
    if candidate.get("invocation_id") != directory.name:
        problems.append("terminal_candidate_invocation")
    commit_ref = candidate.get("commit")
    commit_path = None
    if not isinstance(commit_ref, Mapping):
        problems.append("terminal_candidate_commit")
    else:
        commit_path = _reference_path(commit_ref.get("path"))
        if commit_path is None or \
                _file_sha(commit_path) != commit_ref.get("sha256"):
            problems.append("terminal_candidate_commit")
        else:
            try:
                commit = strict_read_json(commit_path)
            except (OSError, ValueError, TypeError):
                problems.append("terminal_candidate_commit_unreadable")
            else:
                if commit.get("schema") != EVENT_SCHEMA or \
                        commit.get("kind") != "commit" or \
                        commit.get("execution_seq") != \
                        _TERMINAL_EXECUTION_SEQ or \
                        commit.get("stage") not in _TERMINAL_STAGES or \
                        commit.get("invocation_id") != directory.name:
                    problems.append("terminal_candidate_commit_binding")
                if commit.get("binding_spec_sha256") != \
                        claim.get("spec_sha256"):
                    problems.append("terminal_candidate_commit_spec")
            if commit_path is not None and \
                    commit_path.parent.name != f"{_TERMINAL_EXECUTION_SEQ:06d}":
                problems.append("terminal_candidate_commit_seq")
    event_ref = candidate.get("event_candidate")
    if not isinstance(event_ref, Mapping):
        problems.append("terminal_event_candidate_missing")
    else:
        event_path = _reference_path(event_ref.get("path"))
        if event_path is None or \
                _file_sha(event_path) != event_ref.get("sha256"):
            problems.append("terminal_event_candidate")
        else:
            try:
                event = strict_read_json(event_path)
            except (OSError, ValueError, TypeError):
                problems.append("terminal_event_candidate_unreadable")
            else:
                if event.get("schema") != EVENT_SCHEMA or \
                        event.get("kind") != "terminal_candidate" or \
                        event.get("execution_seq") != \
                        _TERMINAL_EXECUTION_SEQ or \
                        event.get("invocation_id") != directory.name or \
                        event.get("binding_spec_sha256") != \
                        claim.get("spec_sha256"):
                    problems.append("terminal_event_candidate_binding")
                if isinstance(commit_ref, Mapping) and (
                        (event.get("commit") or {}).get("path") !=
                        commit_ref.get("path") or
                        (event.get("commit") or {}).get("sha256") !=
                        commit_ref.get("sha256")):
                    problems.append("terminal_event_candidate_commit")
            problems += _terminal_write_after_problems(
                event_path.parent / "terminal_write_after.json",
                expected_sha=event_ref.get("sha256"),
                label="terminal_event")
    problems += _terminal_write_after_problems(
        directory / "terminal_write_after.json",
        expected_sha=candidate_sha, label="terminal")
    return problems


def scan_launch_journals(fixed_root: Path, *,
                         allow_invocation: str | None = None
                         ) -> dict:
    """Scan every attributable cohort root's fixed launch journal.

    A claim is a liability until a valid outer return observation closes the
    invocation.  The current execution may name exactly one invocation whose
    final return is not available yet; no other claim is exempt.
    """
    records: list[dict] = []
    problems: list[str] = []
    debts: list[dict] = []
    for sibling in _cohort_roots(Path(fixed_root)):
        journal = launch_journal_dir(sibling)
        try:
            rows_for_migration = (load_index(sibling)
                                  if _index_path(sibling).is_file() else [])
        except (OSError, ValueError, TypeError):
            rows_for_migration = []
        has_v2 = (sibling / BINDINGS_DIR).exists() or any(
            r.get("accounting_schema") == ACCOUNTING_SCHEMA_V2
            for r in rows_for_migration)
        if not journal.is_dir():
            if has_v2:
                problems.append(f"launch_journal_missing:{sibling.name}")
            continue
        try:
            children = sorted(p for p in journal.iterdir() if p.is_dir())
        except OSError:
            problems.append(f"launch_journal_unreadable:{sibling.name}")
            continue
        for directory in children:
            if any(p.name.endswith(".tmp") for p in directory.iterdir()
                   if p.is_file()):
                problems.append(f"launch_journal_temp:{sibling.name}:"
                                f"{directory.name}")
                continue
            claim_path = directory / "claim.json"
            if not claim_path.is_file():
                problems.append(f"launch_claim_missing:{sibling.name}:"
                                f"{directory.name}")
                continue
            try:
                claim = strict_read_json(claim_path)
            except (OSError, ValueError, TypeError) as exc:
                problems.append(f"launch_claim_unreadable:{sibling.name}:"
                                f"{directory.name}:{type(exc).__name__}")
                continue
            claim["_sha256"] = sha256_file(claim_path)
            found = _claim_problems(claim, invocation_id=directory.name,
                                    root=sibling)
            if found:
                problems += [f"launch_claim:{sibling.name}:{p}" for p in found]
                continue
            entry = {"run": path_for_record(sibling),
                     "invocation_id": directory.name,
                     "claim": claim, "claim_sha256": claim["_sha256"]}
            failure_paths = sorted(directory.glob("failure*.json"))
            revoke_paths = sorted(directory.glob("revoke*.json"))
            if failure_paths or revoke_paths:
                entry["unsettled"] = [path_for_record(p)
                                      for p in failure_paths + revoke_paths]
                debts.append(entry)
                if directory.name != allow_invocation:
                    problems.append(f"launch_debt:{sibling.name}:"
                                    f"{directory.name}")
            observation_path = directory / "return_observation.json"
            return_problems: list[str] = []
            if not failure_paths and not revoke_paths:
                registered_path = directory / "registered.json"
                if not registered_path.is_file():
                    return_problems.append("registered_missing")
                else:
                    try:
                        registered = strict_read_json(registered_path)
                    except (OSError, ValueError, TypeError):
                        return_problems.append("registered_unreadable")
                    else:
                        if (registered.get("schema") != EVENT_SCHEMA or
                                registered.get("kind") != "registered" or
                                registered.get("invocation_id") !=
                                directory.name or
                                registered.get("execution_seq") != 3 or
                                registered.get("stage") != "R0" or
                                registered.get("attempt") != "attempt_002"):
                            return_problems.append("registered_binding")
                        start = registered.get("start") or {}
                        start_path = _reference_path(start.get("path"))
                        if start_path is None or not start_path.is_file() or \
                                sha256_file(start_path) != start.get("sha256"):
                            return_problems.append("registered_start")
                return_problems += _terminal_closure_problems(
                    directory, claim=claim)
                for candidate_path in sorted(
                        directory.glob("stage_closed_*.json")):
                    if candidate_path.name.endswith("_write_after.json"):
                        continue
                    try:
                        candidate = strict_read_json(candidate_path)
                        seq = int(candidate_path.stem.split("_")[-1])
                    except (OSError, ValueError, TypeError):
                        return_problems.append(
                            f"stage_closed_unreadable:{candidate_path.name}")
                        continue
                    if candidate.get("schema") != EVENT_SCHEMA or \
                            candidate.get("kind") != "stage_closed_candidate" or \
                            candidate.get("invocation_id") != directory.name or \
                            candidate.get("execution_seq") != seq:
                        return_problems.append(
                            f"stage_closed_binding:{seq}")
                    commit = candidate.get("commit") or {}
                    commit_path = _reference_path(commit.get("path"))
                    if commit_path is None or not commit_path.is_file() or \
                            sha256_file(commit_path) != commit.get("sha256"):
                        return_problems.append(
                            f"stage_closed_commit:{seq}")
                    after_path = directory / (
                        f"stage_closed_{seq:06d}_write_after.json")
                    if not after_path.is_file():
                        return_problems.append(
                            f"stage_closed_write_after:{seq}")
                    else:
                        try:
                            after = strict_read_json(after_path)
                        except (OSError, ValueError, TypeError):
                            return_problems.append(
                                f"stage_closed_write_after_unreadable:{seq}")
                        else:
                            if after.get("passed") is not True or \
                                    after.get("candidate_sha256") != \
                                    sha256_file(candidate_path):
                                return_problems.append(
                                    f"stage_closed_write_after_failed:{seq}")
            if observation_path.is_file():
                try:
                    observation = strict_read_json(observation_path)
                except (OSError, ValueError, TypeError) as exc:
                    return_problems += [f"return_unreadable:"
                                        f"{type(exc).__name__}"]
                else:
                    # Accumulate: a valid observation cannot erase a missing
                    # or malformed registered/terminal/candidate record.
                    return_problems += return_observation_problems(
                        observation, invocation_id=directory.name, claim=claim)
                    entry["return_observation"] = observation
            else:
                return_problems = ["return_observation_missing"]
            entry["return_problems"] = return_problems
            records.append(entry)
            if return_problems and directory.name != allow_invocation:
                problems.append(f"launch_unsettled:{sibling.name}:"
                                f"{directory.name}:"
                                f"{return_problems[0]}")
                debts.append(entry)
    return {"known": not problems, "records": records, "debts": debts,
            "problems": problems}


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
    schema = row.get("accounting_schema")
    if schema not in (ACCOUNTING_SCHEMA, ACCOUNTING_SCHEMA_V2):
        problems.append(f"{kind}_schema_missing")
    if schema == ACCOUNTING_SCHEMA_V2:
        if row.get("generation") != GENERATION_002:
            problems.append(f"{kind}_generation")
        if not isinstance(row.get("binding_spec_sha256"), str) or not \
                re.fullmatch(r"[0-9a-f]{64}",
                             str(row.get("binding_spec_sha256"))):
            problems.append(f"{kind}_binding_spec")
        if type(row.get("execution_seq")) is not int or \
                int(row["execution_seq"]) < 3:
            problems.append(f"{kind}_execution_seq")
        if row.get("predecessor_kind") not in ("legacy_head", "attempt_commit"):
            problems.append(f"{kind}_predecessor_kind")
        if not isinstance(row.get("previous_head_sha256"), str) or not \
                re.fullmatch(r"[0-9a-f]{64}",
                             str(row.get("previous_head_sha256"))):
            problems.append(f"{kind}_previous_head")
        if not isinstance(row.get("invocation_id"), str) or not \
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                             str(row.get("invocation_id"))):
            problems.append(f"{kind}_invocation")
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


def _reservation_problems(root: Path,
                          *, allow_current: tuple[str, str] | None = None
                          ) -> list[str]:
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
            if allow_current == (stage, directory.name):
                continue
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


def chain_nominal_for_mode(mode: str) -> dict[str, int]:
    chain = {kind: int(value) for kind, value in CHAIN_NOMINAL.items()}
    if mode == "synthetic":
        chain["scene_setup_mj_forward"] = 0
    return chain


def stage_nominal_for_mode(mode: str, stage: str) -> dict[str, int]:
    nominal = {kind: int(value)
               for kind, value in STAGE_NOMINAL[stage].items()}
    if mode == "synthetic":
        nominal["scene_setup_mj_forward"] = 0
    return nominal


def remaining_nominal_for_mode(mode: str, resume_stage: str | None
                               ) -> dict[str, int]:
    """Nominal calls of the stages that still have to execute.

    A resume re-runs only ``resume_stage`` and its downstream stages; the
    already-settled upstream calls are part of the cohort history, not of the
    required remainder.
    """
    order = list(STAGE_BUDGETS)
    stages = (order if resume_stage is None
              else order[order.index(resume_stage):])
    total = {kind: 0 for kind in CALL_KINDS}
    for stage in stages:
        for kind, value in stage_nominal_for_mode(mode, stage).items():
            total[kind] += int(value)
    return total


def _attempt_number(attempt: str) -> int:
    try:
        return int(str(attempt).split("_")[1])
    except (IndexError, ValueError):
        return 1 << 30


def _reservation_consumed(root: Path, stage: str, attempt: str
                          ) -> dict[str, int] | None:
    path = stage_dir(root, stage) / attempt / "call_reservations.json"
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, Mapping):
        return None
    counts = data.get("consumed")
    if not isinstance(counts, Mapping) or set(counts) != set(CALL_KINDS):
        return None
    if not all(_json_int(value) and int(value) >= 0
               for value in counts.values()):
        return None
    return {kind: int(counts[kind]) for kind in CALL_KINDS}


def _migration_present(root: Path) -> bool:
    bindings = Path(root) / BINDINGS_DIR
    if bindings.is_dir() and any(
            p.is_dir() for p in bindings.iterdir()):
        return True
    try:
        return any(r.get("accounting_schema") == ACCOUNTING_SCHEMA_V2
                   for r in load_index(root))
    except (OSError, ValueError, TypeError):
        return False


def _event_dir(root: Path, execution_seq: int) -> Path:
    return Path(root) / EXECUTION_EVENTS_DIR / f"{int(execution_seq):06d}"


def _stage_plan_for_seq(execution_seq: int) -> dict | None:
    for item in MIGRATION_STAGE_PLAN:
        if item["execution_seq"] == execution_seq:
            return dict(item)
    return None


def _commit_path(root: Path, execution_seq: int) -> Path:
    return _event_dir(root, execution_seq) / "commit.json"


def _commit_problems(root: Path, execution_seq: int, *,
                     binding: BindingContext | None = None,
                     stage: str | None = None,
                     attempt: str | None = None) -> tuple[dict | None,
                                                          list[str]]:
    path = _commit_path(root, execution_seq)
    if not path.is_file():
        return None, [f"commit_missing:{execution_seq}"]
    try:
        commit = strict_read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        return None, [f"commit_unreadable:{execution_seq}:"
                      f"{type(exc).__name__}"]
    problems: list[str] = []
    if commit.get("schema") != EVENT_SCHEMA or \
            commit.get("kind") != "commit":
        problems.append(f"commit_schema:{execution_seq}")
    if commit.get("execution_seq") != execution_seq:
        problems.append(f"commit_seq:{execution_seq}")
    plan = _stage_plan_for_seq(execution_seq)
    if plan is None:
        problems.append(f"commit_plan:{execution_seq}")
    else:
        if stage is not None and commit.get("stage") != stage:
            problems.append(f"commit_stage:{execution_seq}")
        if attempt is not None and commit.get("attempt") != attempt:
            problems.append(f"commit_attempt:{execution_seq}")
        if commit.get("stage") != plan["stage"] or \
                commit.get("attempt") != plan["attempt"]:
            problems.append(f"commit_plan_stage:{execution_seq}")
    if binding is not None:
        if commit.get("generation") != binding.generation:
            problems.append(f"commit_generation:{execution_seq}")
        if commit.get("binding_spec_sha256") != binding.spec_sha256:
            problems.append(f"commit_spec:{execution_seq}")
        if commit.get("legacy_head_sha256") != binding.legacy_head_sha256:
            problems.append(f"commit_legacy:{execution_seq}")
        if not isinstance(commit.get("invocation_id"), str):
            problems.append(f"commit_invocation:{execution_seq}")
    finish = commit.get("finish") or {}
    finish_path = _reference_path(finish.get("path")) if isinstance(
        finish, Mapping) else None
    if finish_path is None or not finish_path.is_file():
        problems.append(f"commit_finish_missing:{execution_seq}")
    elif sha256_file(finish_path) != finish.get("sha256"):
        problems.append(f"commit_finish_sha:{execution_seq}")
    for key in ("index_row_sha256", "ledger_row_sha256",
                "reservation_sha256"):
        if not isinstance(commit.get(key), str) or not re.fullmatch(
                r"[0-9a-f]{64}", commit[key]):
            problems.append(f"commit_row:{execution_seq}:{key}")
    if not isinstance(commit.get("previous_head_sha256"), str):
        problems.append(f"commit_previous:{execution_seq}")
    if execution_seq == 3:
        if commit.get("predecessor_kind") != "legacy_head":
            problems.append("commit_predecessor_kind:3")
    else:
        if commit.get("predecessor_kind") != "attempt_commit":
            problems.append(f"commit_predecessor_kind:{execution_seq}")
        prev = _commit_path(root, execution_seq - 1)
        if not prev.is_file() or sha256_file(prev) != \
                commit.get("previous_head_sha256"):
            problems.append(f"commit_previous_sha:{execution_seq}")
    if stage is not None and attempt is not None:
        try:
            index = load_index(root)
            ledger = load_ledger(root)
        except (OSError, ValueError, TypeError):
            problems.append(f"commit_rows_unreadable:{execution_seq}")
        else:
            row = next((r for r in index if r.get("stage") == stage
                        and r.get("attempt") == attempt), None)
            entry = next((e for e in ledger if e.get("stage") == stage
                          and e.get("attempt") == attempt), None)
            if row is None or entry is None:
                problems.append(f"commit_row_missing:{execution_seq}")
            else:
                if canonical_sha256(row) != commit.get("index_row_sha256"):
                    problems.append(f"commit_row_changed:{execution_seq}")
                if canonical_sha256(entry) != commit.get("ledger_row_sha256"):
                    problems.append(f"commit_ledger_changed:{execution_seq}")
                reservation = stage_dir(root, stage) / attempt / \
                    "call_reservations.json"
                if not reservation.is_file() or sha256_file(
                        reservation) != commit.get("reservation_sha256"):
                    problems.append(f"commit_reservation:{execution_seq}")
    return commit, problems


def _start_event_problems(root: Path, execution_seq: int,
                          row: Mapping[str, Any],
                          binding: BindingContext | None
                          ) -> list[str]:
    path = _event_dir(root, execution_seq) / "start.json"
    if not path.is_file():
        return [f"start_event_missing:{execution_seq}"]
    try:
        event = strict_read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        return [f"start_event_unreadable:{execution_seq}:"
                f"{type(exc).__name__}"]
    problems: list[str] = []
    if event.get("schema") != EVENT_SCHEMA or event.get("kind") != "start":
        problems.append(f"start_event_schema:{execution_seq}")
    for key, want in (("execution_seq", execution_seq),
                      ("stage", row.get("stage")),
                      ("attempt", row.get("attempt")),
                      ("generation", row.get("generation")),
                      ("binding_spec_sha256", row.get("binding_spec_sha256")),
                      ("predecessor_kind", row.get("predecessor_kind")),
                      ("previous_head_sha256",
                       row.get("previous_head_sha256")),
                      ("invocation_id", row.get("invocation_id"))):
        if event.get(key) != want:
            problems.append(f"start_event_{key}:{execution_seq}")
    snapshot = event.get("row_snapshot")
    if not isinstance(snapshot, Mapping):
        problems.append(f"start_event_row_snapshot:{execution_seq}")
    elif canonical_sha256(snapshot) != event.get("row_sha256"):
        problems.append(f"start_event_row:{execution_seq}")
    elif any(snapshot.get(key) != row.get(key) for key in (
            "stage", "attempt", "generation", "binding_spec_sha256",
            "execution_seq", "predecessor_kind", "previous_head_sha256",
            "invocation_id", "call_balance_before")):
        problems.append(f"start_event_row_binding:{execution_seq}")
    if binding is not None and event.get("legacy_head_sha256") != \
            binding.legacy_head_sha256:
        problems.append(f"start_event_legacy:{execution_seq}")
    return problems


def _execution_records(root: Path, *,
                       binding: BindingContext | None = None,
                       allow_current: tuple[str, str] | None = None,
                       require_commits: bool = True
                       ) -> tuple[list[dict] | None, BindingContext | None,
                                  list[str]]:
    """Return records in actual execution order, not stage/attempt order."""
    root = Path(root)
    try:
        index = load_index(root)
        ledger = load_ledger(root)
        disk = {stage: _disk_attempts(root, stage) for stage in STAGE_BUDGETS}
    except (OSError, ValueError, TypeError):
        return None, binding, ["execution_rows_unreadable"]
    index_by = {(r.get("stage"), r.get("attempt")): r for r in index}
    ledger_by = {(e.get("stage"), e.get("attempt")): e for e in ledger}
    if len(index_by) != len(index) or len(ledger_by) != len(ledger):
        return None, binding, ["execution_duplicate_rows"]
    if set(index_by) != set(ledger_by):
        return None, binding, ["execution_row_sets"]
    for stage in STAGE_BUDGETS:
        want = {key for key in index_by if key[0] == stage}
        if {a for a in disk[stage]} != {key[1] for key in want}:
            return None, binding, [f"execution_disk_set:{stage}"]
    present = _migration_present(root)
    if not present:
        return None, None, []
    if binding is None:
        binding, problems = load_binding(root, require_active=False)
        if binding is None:
            return None, None, problems or ["binding_missing"]
    # Verify the frozen seq=1/2 prefix and its reservations.
    prefix_problems = _snapshot_problems(binding.snapshot, run_root=root,
                                         root=root)
    if prefix_problems:
        return None, binding, prefix_problems
    legacy_records = binding.legacy_head.get("records")
    if not isinstance(legacy_records, list):
        return None, binding, ["legacy_records_missing"]
    records: list[dict] = []
    for rec in legacy_records:
        key = (rec.get("stage"), rec.get("attempt"))
        row = index_by.get(key)
        entry = ledger_by.get(key)
        if row is None or entry is None:
            return None, binding, [f"legacy_record_missing:{key[0]}"]
        if canonical_sha256(row) != rec.get("index_row_sha256") or \
                canonical_sha256(entry) != rec.get("ledger_row_sha256"):
            return None, binding, [f"legacy_record_changed:{key[0]}"]
        reservation = stage_dir(root, key[0]) / key[1] / \
            "call_reservations.json"
        if not reservation.is_file() or sha256_file(reservation) != rec.get(
                "reservation_sha256"):
            return None, binding, [f"legacy_reservation_changed:{key[0]}"]
        records.append({
            "execution_seq": int(rec["execution_seq"]),
            "generation": GENERATION_001,
            "stage": key[0], "attempt": key[1], "index_row": row,
            "ledger_row": entry, "v2": False,
        })
    v2_rows = []
    for key, row in index_by.items():
        if row.get("accounting_schema") != ACCOUNTING_SCHEMA_V2:
            continue
        seq = row.get("execution_seq")
        if type(seq) is not int:
            return None, binding, [f"v2_sequence_invalid:{key[0]}"]
        if row.get("generation") != binding.generation:
            return None, binding, [f"v2_generation:{key[0]}"]
        if row.get("binding_spec_sha256") != binding.spec_sha256:
            return None, binding, [f"v2_spec:{key[0]}"]
        v2_rows.append((seq, key, row))
    v2_rows.sort(key=lambda item: item[0])
    expected_seq = 3
    previous_commit_sha = binding.legacy_head_sha256
    for seq, key, row in v2_rows:
        if seq != expected_seq:
            return None, binding, [f"v2_sequence_gap:{seq}!={expected_seq}"]
        plan = _stage_plan_for_seq(seq)
        if plan is None or key != (plan["stage"], plan["attempt"]) or \
                row.get("generation") != plan["generation"]:
            return None, binding, [f"v2_plan_mismatch:{seq}"]
        if seq == 3:
            if row.get("predecessor_kind") != "legacy_head":
                return None, binding, ["v2_predecessor_kind:3"]
        else:
            if row.get("predecessor_kind") != "attempt_commit":
                return None, binding, [f"v2_predecessor_kind:{seq}"]
        if row.get("previous_head_sha256") != previous_commit_sha:
            return None, binding, [f"v2_previous_head:{seq}"]
        record_problems = _start_event_problems(root, seq, row, binding)
        if record_problems:
            return None, binding, record_problems
        entry = ledger_by[key]
        current = allow_current == key
        if current:
            if row.get("status") not in (
                    "running", "publishing", "done", "incomplete",
                    "crashed", "timeout"):
                return None, binding, [f"v2_current_state:{seq}"]
        elif require_commits:
            commit, commit_problems = _commit_problems(
                root, seq, binding=binding, stage=key[0], attempt=key[1])
            if commit_problems:
                return None, binding, commit_problems
        commit_file = _commit_path(root, seq)
        if commit_file.is_file():
            previous_commit_sha = sha256_file(commit_file)
        elif seq + 1 < 3 + len(v2_rows):
            return None, binding, [f"v2_previous_commit_missing:{seq}"]
        records.append({
            "execution_seq": seq, "generation": binding.generation,
            "stage": key[0], "attempt": key[1], "index_row": row,
            "ledger_row": entry, "v2": True,
        })
        expected_seq += 1
    return records, binding, []


def _cohort_chain(root: Path, *, allow_current: tuple[str, str] | None = None,
                  generation: str | None = None,
                  binding: BindingContext | None = None,
                  require_commits: bool = True
                  ) -> tuple[dict[str, dict[str, dict]], dict[str, int]] | None:
    """Reconstruct per-attempt increments from the cumulative balance chain.

    A migrated root is replayed in execution-sequence order: the frozen v1
    prefix first, then each committed v2 attempt.  An explicit
    ``call_balance_before`` must equal the running cumulative and
    ``after == before + reservation.consumed``.  Legacy rows without the
    explicit field use the uniquely determined ``before = after - consumed``
    and still have to line up with the running cumulative.
    """
    try:
        index = load_index(root)
        ledger = load_ledger(root)
        disk = {stage: _disk_attempts(root, stage) for stage in STAGE_BUDGETS}
    except (OSError, ValueError, TypeError):
        return None
    index_by: dict[str, dict[str, dict]] = {stage: {} for stage in STAGE_BUDGETS}
    ledger_by: dict[str, dict[str, dict]] = {stage: {} for stage in STAGE_BUDGETS}
    for row in index:
        stage = row.get("stage")
        if stage in index_by:
            if row.get("attempt") in index_by[stage]:
                return None
            index_by[stage][row.get("attempt")] = row
    for entry in ledger:
        stage = entry.get("stage")
        if stage in ledger_by:
            if entry.get("attempt") in ledger_by[stage]:
                return None
            ledger_by[stage][entry.get("attempt")] = entry
    for stage in STAGE_BUDGETS:
        if set(index_by[stage]) != set(ledger_by[stage]) \
                or set(index_by[stage]) != disk[stage]:
            return None
    records, binding, record_problems = _execution_records(
        root, binding=binding, allow_current=allow_current,
        require_commits=require_commits)
    if records is None:
        if record_problems:
            return None
        # Historical v1 root without a migration record.
        records = []
        for stage in STAGE_BUDGETS:
            for attempt in sorted(index_by[stage], key=_attempt_number):
                records.append({
                    "execution_seq": None, "generation": GENERATION_001,
                    "stage": stage, "attempt": attempt,
                    "index_row": index_by[stage][attempt],
                    "ledger_row": ledger_by[stage][attempt], "v2": False,
                })
    cumulative = {kind: 0 for kind in CALL_KINDS}
    out: dict[str, dict[str, dict]] = {stage: {} for stage in STAGE_BUDGETS}
    for record in records:
        stage = record["stage"]
        attempt = record["attempt"]
        if allow_current == (stage, attempt):
            continue
        row = record["index_row"]
        entry = record["ledger_row"]
        after = _call_counts_from_balance(row.get("call_balance"))
        ledger_after = _call_counts_from_balance(entry.get("call_balance"))
        if after is None or ledger_after != after:
            return None
        consumed = _reservation_consumed(root, stage, attempt)
        if consumed is None:
            return None
        explicit = row.get("call_balance_before")
        if explicit is not None:
            before = _call_counts_from_balance(explicit)
            if before is None or before != cumulative:
                return None
        else:
            before = {kind: after[kind] - consumed[kind]
                      for kind in CALL_KINDS}
            if any(before[kind] < 0 for kind in CALL_KINDS):
                return None
            if before != cumulative:
                return None
        expected = {kind: cumulative[kind] + consumed[kind]
                    for kind in CALL_KINDS}
        if after != expected:
            return None
        if generation is None or record["generation"] == generation:
            out[stage][attempt] = {
                "before": dict(before), "after": dict(after),
                "counts": {kind: int(consumed[kind]) for kind in CALL_KINDS},
                "generation": record["generation"],
                "execution_seq": record["execution_seq"],
            }
        cumulative = after
    return out, cumulative


def stage_attempt_deltas(root: Path, stage: str,
                         *, allow_current: str | None = None,
                         current: tuple[str, str] | None = None,
                         generation: str | None = None,
                         binding: BindingContext | None = None,
                         require_commits: bool = True
                         ) -> dict[str, dict] | None:
    skip = current if current is not None else (
        (stage, allow_current) if allow_current is not None else None)
    chain = _cohort_chain(root, allow_current=skip, generation=generation,
                          binding=binding, require_commits=require_commits)
    if chain is None:
        return None
    return chain[0].get(stage, {})


def _stage_deltas(root: Path, *, require_commits: bool = True
                  ) -> dict[str, dict[str, dict]] | None:
    out: dict[str, dict[str, dict]] = {}
    for stage in STAGE_BUDGETS:
        deltas = stage_attempt_deltas(root, stage,
                                      require_commits=require_commits)
        if deltas is None:
            return None
        out[stage] = deltas
    return out


def _settled_deltas_with_unsettled_tail(
        root: Path, *, binding: BindingContext | None = None
        ) -> tuple[dict[str, dict[str, dict]] | None, str | None]:
    """Return the settled prefix when a current attempt is still unknown.

    The unsettled row is excluded from both the increment chain and the
    reported totals; callers must expose the result as a known lower bound,
    never as the cohort total.
    """
    full = _stage_deltas(root)
    if full is not None:
        return full, None
    try:
        rows = load_index(root)
    except (OSError, ValueError, TypeError):
        return None, None
    for row in rows:
        if row.get("status") not in ("running", "publishing"):
            continue
        stage = row.get("stage")
        attempt = row.get("attempt")
        if stage not in STAGE_BUDGETS or not isinstance(attempt, str):
            continue
        skip = (stage, attempt)
        out: dict[str, dict[str, dict]] = {}
        failed = False
        for name in STAGE_BUDGETS:
            deltas = stage_attempt_deltas(
                root, name, current=skip, binding=binding,
                require_commits=False)
            if deltas is None:
                failed = True
                break
            out[name] = deltas
        if not failed:
            return out, f"{stage}/{attempt}"
    return None, None


def verify_history(root: Path, *,
                   allow_current: tuple[str, str] | None = None,
                   generation: str | None = None,
                   binding: BindingContext | None = None,
                   allow_invocation: str | None = None,
                   scan_journal: bool = False) -> list[str]:
    """Full-history gate before any resume/selection decision."""
    problems: list[str] = []
    migrated = _migration_present(root)
    if migrated:
        records, loaded, record_problems = _execution_records(
            root, binding=binding, allow_current=allow_current)
        if records is None:
            return record_problems or ["history_migration_unknown"]
        binding = loaded
        if generation is not None and generation not in (
                GENERATION_001, GENERATION_002):
            problems.append(f"history_generation:{generation}")
    # A claim is a liability from the moment it is written, before any
    # binding/intent exists.  Public history therefore scans the fixed journal
    # even for a root that has no v2 rows yet; an in-flight execution passes
    # its own invocation as the sole allowed exception.
    if scan_journal or migrated or launch_journal_dir(root).is_dir():
        journal = scan_launch_journals(
            Path(root).parent, allow_invocation=allow_invocation)
        for problem in journal.get("problems") or []:
            if str(problem).startswith(
                    f"launch_journal_missing:{Path(root).name}") or \
                    f":{Path(root).name}:" in str(problem) or \
                    str(problem).endswith(f":{Path(root).name}"):
                problems.append(f"history_journal:{problem}")
    for stage in STAGE_BUDGETS:
        try:
            index = [r for r in load_index(root) if r.get("stage") == stage]
            ledger = [r for r in load_ledger(root) if r.get("stage") == stage]
            disk = _disk_attempts(root, stage)
        except (OSError, ValueError, TypeError):
            problems.append(f"history_unreadable:{stage}")
            continue
        index_by = {r.get("attempt"): r for r in index}
        ledger_by = {e.get("attempt"): e for e in ledger}
        if (len(index_by) != len(index) or len(ledger_by) != len(ledger)
                or set(index_by) != set(ledger_by)
                or set(disk) != set(index_by)):
            problems.append(f"history_sets:{stage}")
            continue
        for attempt, row in index_by.items():
            if allow_current == (stage, attempt):
                if row.get("status") not in (
                        "running", "publishing", "done", "incomplete",
                        "crashed", "timeout"):
                    problems.append(f"history_current_not_terminal:{stage}")
                continue
            entry = ledger_by[attempt]
            for problem in (_row_identity_problems(row, kind="index")
                            + _row_identity_problems(entry, kind="ledger")):
                problems.append(f"history_row:{stage}:{attempt}:{problem}")
            for problem in _call_balance_problems(row, entry):
                problems.append(f"history_balance:{stage}:{attempt}:{problem}")
            if row.get("status") not in ("done", "crashed", "timeout",
                                         "incomplete"):
                problems.append(f"history_not_terminal:{stage}:{attempt}")
            if row.get("status") != entry.get("status") \
                    or row.get("complete") is not entry.get("complete"):
                problems.append(f"history_state:{stage}:{attempt}")
            if row.get("status") != "done" and row.get("complete") is not False:
                problems.append(f"history_failed_claims_complete:{stage}:{attempt}")
            measured = entry.get("measured_wall_s")
            rounded = entry.get("wall_s")
            index_measured = row.get("measured_wall_s")
            if not (_nonneg(measured) and _nonneg(rounded)
                    and _nonneg(index_measured)):
                problems.append(f"history_cost:{stage}:{attempt}")
            elif float(measured) != float(index_measured) or float(rounded) != \
                    round(float(measured), 1):
                problems.append(f"history_cost:{stage}:{attempt}")
        deltas = stage_attempt_deltas(
            root, stage,
            current=(allow_current if allow_current
                     and allow_current[0] == stage else None),
            binding=binding)
        if deltas is None:
            chain = _cohort_chain(root, allow_current=allow_current,
                                  binding=binding)
            if chain is None:
                problems.append(f"history_deltas:{stage}")
            elif stage not in chain[0]:
                problems.append(f"history_deltas:{stage}")
    return problems


def selected_chain_counts(deltas: Mapping[str, Mapping[str, Mapping[str, Any]]],
                          selected: Mapping[str, str]) -> dict[str, int] | None:
    if set(selected) != set(STAGE_BUDGETS):
        return None
    totals = {kind: 0 for kind in CALL_KINDS}
    for stage, attempt in selected.items():
        rec = (deltas.get(stage) or {}).get(attempt)
        if rec is None:
            return None
        for kind in CALL_KINDS:
            totals[kind] += int(rec["counts"][kind])
    return totals


def stage_call_totals(deltas: Mapping[str, Mapping[str, Mapping[str, Any]]],
                      stage: str) -> dict[str, int]:
    totals = {kind: 0 for kind in CALL_KINDS}
    for rec in (deltas.get(stage) or {}).values():
        for kind in CALL_KINDS:
            totals[kind] += int(rec["counts"][kind])
    return totals


def cohort_call_totals_from_deltas(
        deltas: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, int]:
    totals = {kind: 0 for kind in CALL_KINDS}
    for stage in STAGE_BUDGETS:
        for rec in (deltas.get(stage) or {}).values():
            for kind in CALL_KINDS:
                totals[kind] += int(rec["counts"][kind])
    return totals


def cohort_call_totals(fixed_root: Path, *, exclude_root: Path | None = None
                       ) -> dict[str, int] | None:
    journal = scan_launch_journals(Path(fixed_root))
    if not journal.get("known"):
        return None
    totals = {kind: 0 for kind in CALL_KINDS}
    for sibling in _cohort_roots(fixed_root):
        if exclude_root is not None \
                and sibling.resolve() == Path(exclude_root).resolve():
            continue
        marker: dict = {}
        marker_path = sibling / "cohort.json"
        try:
            if marker_path.is_file():
                marker = read_json(marker_path)
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
        deltas = _stage_deltas(sibling)
        if deltas is None:
            return None
        part = cohort_call_totals_from_deltas(deltas)
        for name in CALL_KINDS:
            totals[name] += int(part[name])
    return totals


def _prior_call_counts_impl(root: Path, *, fixed_root: Path,
                            mode: str = "real") -> dict[str, int] | None:
    if mode == "synthetic":
        deltas = _stage_deltas(Path(root))
        if deltas is None:
            return None
        return cohort_call_totals_from_deltas(deltas)
    return cohort_call_totals(fixed_root, exclude_root=None)


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
    journal = scan_launch_journals(Path(fixed_root))
    if not journal.get("known"):
        return {"known": False, "reason": "launch_journal_unknown",
                "runs": [], "journal_problems": journal.get("problems")}
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
    journal = scan_launch_journals(Path(fixed_root))
    if not journal.get("known"):
        return None
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
    return _prior_call_counts_impl(root, fixed_root=fixed_root, mode=mode)


def _migration_budget_for(root: Path, stage: str, *,
                          binding: BindingContext | None = None,
                          allow_invocation: str | None = None) -> dict:
    if binding is None:
        binding, problems = load_binding(root, require_active=False)
        if binding is None:
            raise SystemExit(f"migration binding unknown: {problems}")
    snapshot = binding.snapshot
    base = _budget_from_snapshot(snapshot, stage)
    journal = scan_launch_journals(Path(root).parent,
                                   allow_invocation=allow_invocation)
    if not journal.get("known"):
        raise SystemExit(f"launch journal unknown: "
                         f"{journal.get('problems')}")
    try:
        index = load_index(root)
        ledger = load_ledger(root)
    except (OSError, ValueError, TypeError):
        raise SystemExit("migration cost rows unreadable")
    v2_rows = [r for r in index
               if r.get("generation") == GENERATION_002
               and r.get("accounting_schema") == ACCOUNTING_SCHEMA_V2]
    v2_ledger = [e for e in ledger
                 if e.get("generation") == GENERATION_002
                 and e.get("accounting_schema") == ACCOUNTING_SCHEMA_V2]
    used_attempts = len([r for r in v2_rows if r.get("stage") == stage])
    used_stage = sum(float(e.get("measured_wall_s", 0.0)) for e in v2_ledger
                     if e.get("stage") == stage)
    used_total = sum(float(e.get("measured_wall_s", 0.0)) for e in v2_ledger)
    remaining_stage = (float(STAGE_BUDGETS[stage]["budget_s"])
                       - float(base["local_stage_s"]) - used_stage)
    remaining_total = (TOTAL_BUDGET_S
                       - float((snapshot.get("cost_s") or {}).get(
                           "total", 0.0)) - used_total)
    remaining = min(remaining_stage, remaining_total)
    attempts_used = int(base["attempts_used"]) + used_attempts
    if attempts_used >= int(STAGE_BUDGETS[stage]["attempts"]):
        raise SystemExit(f"{stage} attempt limit reached: {attempts_used}")
    if remaining <= float(STAGE_BUDGETS[stage]["reserve_s"]):
        raise SystemExit(f"{stage} budget exhausted: {remaining:.6f}s")
    return {
        **base, "local_stage_s": float(base["local_stage_s"]) + used_stage,
        "prior_stage_s": 0.0, "prior_total_s": 0.0,
        "attempts_used": attempts_used, "remaining_s": remaining,
    }


def budget_for(root: Path, stage: str, *, fixed_root: Path,
               mode: str = "real",
               binding: BindingContext | None = None,
               allow_invocation: str | None = None) -> dict:
    if stage not in STAGE_BUDGETS:
        raise ValueError(f"unknown stage {stage}")
    if _migration_present(root):
        return _migration_budget_for(root, stage, binding=binding,
                                     allow_invocation=allow_invocation)
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


def _execution_fields(execution: Mapping[str, Any] | None, *,
                      root: Path, stage: str, attempt: str) -> dict:
    if execution is None:
        return {}
    if not isinstance(execution, Mapping):
        raise ValueError("execution context must be a mapping")
    seq = execution.get("execution_seq")
    generation = execution.get("generation")
    spec_sha = execution.get("binding_spec_sha256")
    invocation = execution.get("invocation_id")
    if type(seq) is not int or seq < 3:
        raise ValueError("execution_seq must be an integer >= 3")
    plan = _stage_plan_for_seq(seq)
    if plan is None or plan["stage"] != stage or plan["attempt"] != attempt:
        raise ValueError(f"execution_seq {seq} does not match {stage}/{attempt}")
    if generation != plan["generation"]:
        raise ValueError("execution generation mismatch")
    if not isinstance(spec_sha, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                         spec_sha):
        raise ValueError("execution binding spec hash missing")
    if not isinstance(invocation, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", invocation):
        raise ValueError("execution invocation id missing")
    predecessor = "legacy_head" if seq == 3 else "attempt_commit"
    if seq == 3:
        previous = execution.get("legacy_head_sha256")
        if not isinstance(previous, str) or not re.fullmatch(
                r"[0-9a-f]{64}", previous):
            raise ValueError("execution legacy head hash missing")
    else:
        previous_path = _commit_path(root, seq - 1)
        if not previous_path.is_file():
            raise ValueError(f"previous commit missing for seq {seq}")
        previous = sha256_file(previous_path)
    return {
        "accounting_schema": ACCOUNTING_SCHEMA_V2,
        "generation": generation,
        "binding_spec_sha256": spec_sha,
        "execution_seq": seq,
        "predecessor_kind": predecessor,
        "previous_head_sha256": previous,
        "invocation_id": invocation,
        **({"claim_path": execution.get("claim_path"),
            "claim_sha256": execution.get("claim_sha256")}
           if execution.get("claim_path") is not None else {}),
    }


def pre_register_attempt(root: Path, stage: str, attempt: str, t0: float,
                         *, scope: dict, budget: dict,
                         registration_sha: str | None = None,
                         call_balance: Mapping[str, Any] | None = None,
                         execution: Mapping[str, Any] | None = None) -> dict:
    if int(budget.get("attempts_used", 0)) >= int(budget.get("attempts", 0)):
        raise SystemExit(f"{stage} attempt limit reached before registration")
    marker = _cohort_path(root)
    if marker.is_file() and read_json(marker).get("cohort_id") not in (None, COHORT_ID):
        raise ValueError("cohort marker mismatch")
    elapsed = max(0.0, time.perf_counter() - t0)
    balance = dict(call_balance) if call_balance else None
    execution_fields = _execution_fields(
        execution, root=root, stage=stage, attempt=attempt)
    entry = {
        "status": "running", "complete": False,
        "measured_wall_s": elapsed,
        "accounting_schema": (ACCOUNTING_SCHEMA_V2 if execution_fields
                              else ACCOUNTING_SCHEMA),
        "cohort_id": COHORT_ID,
        "execution_registration_sha256": registration_sha,
        "scope": scope, "budget": budget,
        "call_balance": balance,
        "call_balance_before": balance,
        "deadline_perf_counter": t0 + float(budget["remaining_s"]),
        "allocated_wall_s": time.time(),
        **execution_fields,
    }
    if execution_fields:
        row_snapshot = {**entry, "stage": stage, "attempt": attempt}
        event = {
            "schema": EVENT_SCHEMA,
            "kind": "start",
            "execution_seq": execution_fields["execution_seq"],
            "generation": execution_fields["generation"],
            "stage": stage,
            "attempt": attempt,
            "binding_spec_sha256": execution_fields["binding_spec_sha256"],
            "predecessor_kind": execution_fields["predecessor_kind"],
            "previous_head_sha256": execution_fields["previous_head_sha256"],
            "invocation_id": execution_fields["invocation_id"],
            "claim_path": execution_fields.get("claim_path"),
            "claim_sha256": execution_fields.get("claim_sha256"),
            "legacy_head_sha256": execution.get("legacy_head_sha256"),
            "row_snapshot": row_snapshot,
            "row_sha256": canonical_sha256(row_snapshot),
            "t0_perf_counter": float(t0),
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json_exclusive(
            _event_dir(root, execution_fields["execution_seq"]) / "start.json",
            event)
    upsert_index(root, stage, attempt, entry)
    upsert_cost(root, stage, attempt, elapsed,
                {"complete": False, "scope": scope, "budget": budget,
                 "call_balance": balance,
                 "call_balance_before": balance,
                 **execution_fields},
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
    existing = _attempt_row(root, stage, attempt)
    if type(existing.get("execution_seq")) is int and \
            _commit_path(root, int(existing["execution_seq"])).is_file():
        raise ValueError(
            "committed attempt cannot be resealed or withdrawn; append a "
            "revoke/failure record instead")
    schema = (ACCOUNTING_SCHEMA_V2
              if existing.get("accounting_schema") == ACCOUNTING_SCHEMA_V2
              else ACCOUNTING_SCHEMA)
    row = {**existing, **row, "stage": stage, "attempt": attempt,
           "accounting_schema": schema, "cohort_id": COHORT_ID}
    for key in ("generation", "binding_spec_sha256", "execution_seq",
                "predecessor_kind", "previous_head_sha256",
                "invocation_id", "claim_path", "claim_sha256"):
        if key in existing:
            row[key] = existing[key]
            extra.setdefault(key, existing[key])
    if schema == ACCOUNTING_SCHEMA_V2:
        extra["accounting_schema"] = ACCOUNTING_SCHEMA_V2
    balance = extra.get("call_balance")
    if isinstance(balance, Mapping):
        row["call_balance"] = dict(balance)
        row["call_balance_after"] = dict(balance)
        extra["call_balance_after"] = dict(balance)
    if isinstance(existing.get("call_balance_before"), Mapping):
        row["call_balance_before"] = dict(existing["call_balance_before"])
        extra["call_balance_before"] = dict(existing["call_balance_before"])
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


@dataclass
class CommitResult:
    committed: bool
    finish_path: Path | None
    finish_sha256: str | None
    commit_path: Path | None
    commit_sha256: str | None
    problems: list[str] = field(default_factory=list)
    observed_elapsed: float | None = None


def _launch_invocation_dir(root: Path, invocation_id: str) -> Path:
    return launch_journal_dir(root) / invocation_id


def claim_launch(root: Path, *, authorization_sha256: str,
                 spec_sha256: str, legacy_head_sha256: str,
                 marker_sha256: str, invocation_id: str,
                 t0_perf_counter: float, target: Mapping[str, Any],
                 pid: int | None = None, ppid: int | None = None,
                 ) -> tuple[Path, str]:
    """Persist the one-shot launch claim before imports or real work."""
    if not _nonneg(t0_perf_counter):
        raise ValueError("launch t0 must be finite and non-negative")
    directory = _launch_invocation_dir(root, invocation_id)
    path = directory / "claim.json"
    payload = {
        "schema": LAUNCH_CLAIM_SCHEMA,
        "invocation_id": invocation_id,
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "run_root_realpath": str(Path(root).resolve()),
        "authorization_sha256": authorization_sha256,
        "spec_sha256": spec_sha256,
        "legacy_head_sha256": legacy_head_sha256,
        "marker_sha256": marker_sha256,
        "t0_perf_counter": float(t0_perf_counter),
        "target": dict(target),
        "pid": int(os.getpid() if pid is None else pid),
        "ppid": int(os.getppid() if ppid is None else ppid),
        "executable": sys.executable,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    sha = write_json_exclusive(path, payload)
    return path, sha


def _prepare_revision_number(directory: Path) -> int:
    numbers = []
    for path in directory.glob("prepare_revision_*.json"):
        match = re.fullmatch(r"prepare_revision_(\d{4})\.json", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


_COST_ONLY_ROW_KEYS = ("measured_wall_s", "sampled_wall_s",
                       "seal_tail_reserve_s", "wall_s")


def _frozen_row_view(row: Mapping[str, Any]) -> dict:
    """Fields that must not change when only the billed cost is re-sealed.

    ``wall_s`` is the ledger's rounded projection of ``measured_wall_s`` and
    is re-written by every cost re-seal, so it must track the billed cost
    instead of being frozen with the semantic row.
    """
    return {key: value for key, value in row.items()
            if key not in _COST_ONLY_ROW_KEYS}


def _billed_cost_problems(row: Mapping[str, Any], *, kind: str,
                          label: str) -> list[str]:
    """The rounded ledger field must remain a projection of the cost."""
    measured = row.get("measured_wall_s")
    if not _nonneg(measured):
        return [f"{label}_{kind}_measured_cost"]
    rounded = row.get("wall_s")
    if rounded is None:
        return [] if kind == "index" else [f"{label}_{kind}_wall_s_missing"]
    if not _nonneg(rounded) or float(rounded) != round(float(measured), 1):
        return [f"{label}_{kind}_wall_s"]
    return []


@dataclass
class PreparedResult:
    row: dict | None
    ledger_row: dict | None
    problems: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)


def _verify_prepared_attempt(root: Path, stage: str, attempt: str, *,
                             binding: BindingContext,
                             execution: Mapping[str, Any] | None = None,
                             record: bool = True) -> PreparedResult:
    """Internal candidate verification; never a public consumable row."""
    execution = dict(execution or {})
    seq = execution.get("execution_seq")
    problems: list[str] = []
    if type(seq) is not int:
        return PreparedResult(None, None, ["prepared_execution_seq"])
    plan = _stage_plan_for_seq(seq)
    if plan is None or plan["stage"] != stage or plan["attempt"] != attempt:
        return PreparedResult(None, None, ["prepared_plan"])
    try:
        rows = [r for r in load_index(root)
                if r.get("stage") == stage and r.get("attempt") == attempt]
        ledger = [r for r in load_ledger(root)
                  if r.get("stage") == stage and r.get("attempt") == attempt]
    except (OSError, ValueError, TypeError):
        return PreparedResult(None, None, ["prepared_rows_unreadable"])
    if len(rows) != 1 or len(ledger) != 1:
        return PreparedResult(None, None, ["prepared_row_missing"])
    row, entry = rows[0], ledger[0]
    problems += _row_identity_problems(row, kind="index")
    problems += _row_identity_problems(entry, kind="ledger")
    problems += _call_balance_problems(row, entry)
    problems += _billed_cost_problems(row, kind="index", label="prepared")
    problems += _billed_cost_problems(entry, kind="ledger", label="prepared")
    for key, want in (("generation", binding.generation),
                      ("binding_spec_sha256", binding.spec_sha256),
                      ("execution_seq", seq)):
        if row.get(key) != want:
            problems.append(f"prepared_binding:{key}")
    if row.get("status") not in ("done", "incomplete", "crashed",
                                 "timeout"):
        problems.append("prepared_not_terminal")
    if row.get("complete") is True and row.get("status") != "done":
        problems.append("prepared_complete_state")
    reservation = stage_dir(root, stage) / attempt / \
        "call_reservations.json"
    if not reservation.is_file():
        problems.append("prepared_reservation_missing")
    else:
        try:
            reservation_data = read_json(reservation)
        except (OSError, ValueError, TypeError):
            problems.append("prepared_reservation_unreadable")
        else:
            if reservation_data.get("consumed") is None:
                problems.append("prepared_reservation_unsettled")
    problems += _sealed_evidence_problems(root, stage, row)
    history = verify_history(
        root, allow_current=(stage, attempt),
        allow_invocation=(str(execution.get("invocation_id"))
                          if execution.get("invocation_id") else None),
        binding=binding, generation=binding.generation)
    problems += history
    consumed = _reservation_consumed(root, stage, attempt)
    if consumed is None:
        problems.append("prepared_call_delta")
    facts = {
        "stage": stage, "attempt": attempt, "execution_seq": seq,
        "generation": binding.generation,
        "binding_spec_sha256": binding.spec_sha256,
        "stage_t0_perf_counter": (
            float(execution["stage_t0_perf_counter"])
            if execution.get("stage_t0_perf_counter") is not None else None),
        "index_row_sha256": canonical_sha256(row),
        "ledger_row_sha256": canonical_sha256(entry),
        "index_view_sha256": canonical_sha256(_frozen_row_view(row)),
        "ledger_view_sha256": canonical_sha256(_frozen_row_view(entry)),
        "measured_wall_s": float(row.get("measured_wall_s", 0.0) or 0.0),
        "reservation_sha256": (sha256_file(reservation)
                               if reservation.is_file() else None),
        "complete": bool(row.get("complete") is True),
        "status": row.get("status"),
        "call_balance": entry.get("call_balance"),
        "call_delta": consumed,
    }
    if record:
        directory = _event_dir(root, seq)
        number = _prepare_revision_number(directory)
        payload = {
            "schema": EVENT_SCHEMA,
            "kind": "prepare_revision",
            "revision": number,
            "stage": stage,
            "attempt": attempt,
            "execution_seq": seq,
            "generation": binding.generation,
            "binding_spec_sha256": binding.spec_sha256,
            "facts": facts,
            "problems": list(problems),
            "public_consumable": False,
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        try:
            write_json_exclusive(
                directory / f"prepare_revision_{number:04d}.json", payload)
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"prepared_revision_write:{type(exc).__name__}")
    return PreparedResult(row, entry, problems, facts)


def _refresh_prepared_view(root: Path, stage: str, attempt: str,
                           prepared: PreparedResult,
                           execution: Mapping[str, Any]) -> PreparedResult:
    """Re-check only the frozen small view after the cost re-seal.

    Full artifact hashing belongs to the priced pre-commit verification.  The
    commit path must not repeat it: it only proves that the rows, state,
    reservation and binding identity are still the ones that were verified,
    while allowing the measured cost fields to advance.
    """
    problems: list[str] = []
    try:
        rows = [r for r in load_index(root)
                if r.get("stage") == stage and r.get("attempt") == attempt]
        ledger = [r for r in load_ledger(root)
                  if r.get("stage") == stage and r.get("attempt") == attempt]
    except (OSError, ValueError, TypeError):
        return PreparedResult(None, None,
                              ["commit_prepared_rows_unreadable"],
                              dict(prepared.facts))
    if len(rows) != 1 or len(ledger) != 1:
        return PreparedResult(None, None, ["commit_prepared_row_missing"],
                              dict(prepared.facts))
    row, entry = rows[0], ledger[0]
    problems += _row_identity_problems(row, kind="index")
    problems += _row_identity_problems(entry, kind="ledger")
    problems += _call_balance_problems(row, entry)
    problems += _billed_cost_problems(row, kind="index",
                                      label="commit_prepared")
    problems += _billed_cost_problems(entry, kind="ledger",
                                      label="commit_prepared")
    prepared_cost = prepared.facts.get("measured_wall_s")
    if _nonneg(prepared_cost) and _nonneg(row.get("measured_wall_s")) and \
            float(row["measured_wall_s"]) < float(prepared_cost):
        problems.append("commit_prepared_cost_decreased")
    if canonical_sha256(_frozen_row_view(row)) != prepared.facts.get(
            "index_view_sha256"):
        problems.append("commit_prepared_index_view")
    if canonical_sha256(_frozen_row_view(entry)) != prepared.facts.get(
            "ledger_view_sha256"):
        problems.append("commit_prepared_ledger_view")
    if bool(row.get("complete") is True) != bool(
            prepared.facts.get("complete")):
        problems.append("commit_prepared_complete")
    if row.get("status") != prepared.facts.get("status"):
        problems.append("commit_prepared_status")
    for key, want in (("generation", execution.get("generation")),
                      ("binding_spec_sha256",
                       execution.get("binding_spec_sha256")),
                      ("execution_seq", execution.get("execution_seq"))):
        if want is not None and row.get(key) != want:
            problems.append(f"commit_prepared_binding:{key}")
    reservation = stage_dir(root, stage) / attempt / \
        "call_reservations.json"
    if not reservation.is_file():
        problems.append("commit_prepared_reservation_missing")
        reservation_sha = None
    else:
        reservation_sha = sha256_file(reservation)
        if reservation_sha != prepared.facts.get("reservation_sha256"):
            problems.append("commit_prepared_reservation")
    facts = {
        **prepared.facts,
        "index_row_sha256": canonical_sha256(row),
        "ledger_row_sha256": canonical_sha256(entry),
        "reservation_sha256": reservation_sha,
        "measured_wall_s": float(row.get("measured_wall_s", 0.0) or 0.0),
    }
    return PreparedResult(row, entry, problems, facts)


def commit_attempt(root: Path, stage: str, attempt: str, *, t0: float,
                   budget: Mapping[str, Any], binding: BindingContext,
                   execution: Mapping[str, Any] | None = None,
                   cost_floor: float | None = None,
                   candidate_problems: Sequence[str] = (),
                   prepared: PreparedResult | None = None) -> CommitResult:
    """Write the single finish/commit pair after internal verification."""
    execution = dict(execution or {})
    seq = execution.get("execution_seq")
    problems: list[str] = list(candidate_problems)
    if type(seq) is not int:
        return CommitResult(False, None, None, None, None,
                            problems + ["commit_execution_seq"])
    plan = _stage_plan_for_seq(seq)
    if plan is None or plan["stage"] != stage or plan["attempt"] != attempt:
        return CommitResult(False, None, None, None, None,
                            problems + ["commit_plan"])
    if prepared is None:
        prepared = _verify_prepared_attempt(
            root, stage, attempt,
            execution={**execution, "stage_t0_perf_counter": t0},
            binding=binding)
    else:
        prepared = _refresh_prepared_view(root, stage, attempt, prepared,
                                          execution)
    problems += prepared.problems
    row = prepared.row
    entry = prepared.ledger_row
    if row is None or entry is None:
        return CommitResult(False, None, None, None, None, problems)
    if problems and row.get("complete") is True:
        return CommitResult(False, None, None, None, None, problems)
    committed_wall = float(row.get("measured_wall_s", 0.0) or 0.0)
    floor = 0.0 if cost_floor is None else float(cost_floor)
    committed_wall = max(committed_wall, floor)
    if not _nonneg(committed_wall):
        return CommitResult(False, None, None, None, None,
                            problems + ["commit_cost_invalid"])
    remaining = float(budget["remaining_s"])
    directory = _event_dir(root, seq)
    finish_path = directory / "finish.json"
    finish = {
        "schema": EVENT_SCHEMA,
        "kind": "finish",
        "execution_seq": seq,
        "stage": stage,
        "attempt": attempt,
        "generation": binding.generation,
        "binding_spec_sha256": binding.spec_sha256,
        "legacy_head_sha256": binding.legacy_head_sha256,
        "invocation_id": execution.get("invocation_id"),
        "predecessor_kind": row.get("predecessor_kind"),
        "previous_head_sha256": row.get("previous_head_sha256"),
        "index_row_snapshot": row,
        "ledger_row_snapshot": entry,
        "index_row_sha256": canonical_sha256(row),
        "ledger_row_sha256": canonical_sha256(entry),
        "reservation_sha256": prepared.facts.get("reservation_sha256"),
        "prepared_facts": prepared.facts,
        "stage_t0_perf_counter": float(t0),
        "committed_measured_wall_s": committed_wall,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        finish_sha = write_json_exclusive(finish_path, finish)
    except (OSError, ValueError, TypeError) as exc:
        return CommitResult(False, finish_path, None, None, None,
                            problems + [f"commit_finish_write:"
                                        f"{type(exc).__name__}"])
    commit_path = _commit_path(root, seq)
    commit = {
        "schema": EVENT_SCHEMA,
        "kind": "commit",
        "execution_seq": seq,
        "stage": stage,
        "attempt": attempt,
        "generation": binding.generation,
        "binding_spec_sha256": binding.spec_sha256,
        "legacy_head_sha256": binding.legacy_head_sha256,
        "invocation_id": execution.get("invocation_id"),
        "claim_path": execution.get("claim_path"),
        "claim_sha256": execution.get("claim_sha256"),
        "predecessor_kind": row.get("predecessor_kind"),
        "previous_head_sha256": row.get("previous_head_sha256"),
        "finish": {"path": path_for_record(finish_path),
                   "sha256": finish_sha},
        "index_row_sha256": canonical_sha256(row),
        "ledger_row_sha256": canonical_sha256(entry),
        "reservation_sha256": prepared.facts.get("reservation_sha256"),
        "stage_t0_perf_counter": float(t0),
        "committed_measured_wall_s": committed_wall,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        commit_sha = write_json_exclusive(commit_path, commit)
    except (OSError, ValueError, TypeError) as exc:
        return CommitResult(False, finish_path, finish_sha, None, None,
                            problems + [f"commit_write:{type(exc).__name__}"])
    observed = max(floor, time.perf_counter() - float(t0))
    if not _nonneg(observed) or observed >= remaining or \
            observed > committed_wall:
        problems.append("commit_write_after_gate")
        append_launch_revoke(
            root, invocation_id=str(execution.get("invocation_id")),
            stage=stage, execution_seq=seq, commit_path=commit_path,
            commit_sha256=commit_sha, candidate_sha256=None,
            observed_elapsed=observed, committed_wall=committed_wall,
            remaining_s=remaining, reason="commit_write_after_gate")
        return CommitResult(False, finish_path, finish_sha, commit_path,
                            commit_sha, problems, observed)
    return CommitResult(True, finish_path, finish_sha, commit_path,
                        commit_sha, problems, observed)


def append_launch_revoke(root: Path, *, invocation_id: str,
                         stage: str, execution_seq: int, commit_path: Path,
                         commit_sha256: str, candidate_sha256: str | None,
                         observed_elapsed: float, committed_wall: float,
                         remaining_s: float, reason: str,
                         candidate_path: Path | None = None) -> Path | None:
    if not isinstance(invocation_id, str) or not invocation_id:
        return None
    directory = _launch_invocation_dir(root, invocation_id)
    directory.mkdir(parents=True, exist_ok=True)
    observed = float(observed_elapsed)
    committed = float(committed_wall)
    payload = {
        "schema": EVENT_SCHEMA,
        "kind": "revoke",
        "reason": reason,
        "invocation_id": invocation_id,
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "stage": stage,
        "execution_seq": int(execution_seq),
        "commit": {"path": path_for_record(Path(commit_path)),
                   "sha256": commit_sha256},
        "candidate_sha256": candidate_sha256,
        "observed_elapsed_s": observed,
        "committed_measured_wall_s": committed,
        "remaining_s": float(remaining_s),
        "uncovered_wall_s_lower_bound": max(0.0, observed - committed),
        "post_return_work_performed": False,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if candidate_path is not None:
        payload["candidate"] = {"path": path_for_record(candidate_path),
                                "sha256": sha256_file(candidate_path)
                                if candidate_path.is_file() else None}
    stem = f"revoke_{int(execution_seq):06d}_{reason}"
    path = directory / f"{stem}.json"
    suffix = 1
    while path.exists():
        path = directory / f"{stem}_{suffix:03d}.json"
        suffix += 1
    try:
        write_json_exclusive(path, payload)
    except (OSError, ValueError, TypeError):
        return None
    return path


def write_stage_closed_candidate(
        root: Path, *, stage: str, attempt: str, binding: BindingContext,
        execution: Mapping[str, Any], commit: CommitResult, t0: float,
        remaining_s: float, committed_measured_wall_s: float
        ) -> tuple[Path | None, str | None, list[str]]:
    """Write and then actually re-sample after the success-receipt write."""
    seq = int(execution.get("execution_seq"))
    directory = _event_dir(root, seq)
    problems: list[str] = []
    if not commit.committed or commit.commit_path is None:
        return None, None, ["stage_closed_without_commit"]
    payload = {
        "schema": EVENT_SCHEMA,
        "kind": "stage_closed_candidate",
        "candidate": True,
        "stage": stage,
        "attempt": attempt,
        "execution_seq": seq,
        "generation": binding.generation,
        "binding_spec_sha256": binding.spec_sha256,
        "invocation_id": execution.get("invocation_id"),
        "commit": {"path": path_for_record(commit.commit_path),
                   "sha256": commit.commit_sha256},
        "committed_measured_wall_s": float(committed_measured_wall_s),
        "write_after": {"passed": True},
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = directory / "stage_closed.json"
    try:
        sha = write_json_exclusive(path, payload)
    except (OSError, ValueError, TypeError) as exc:
        return None, None, [f"stage_closed_write:{type(exc).__name__}"]
    observed = max(0.0, time.perf_counter() - float(t0))
    if not _nonneg(observed) or observed >= float(remaining_s) or \
            observed > float(committed_measured_wall_s):
        problems.append("stage_closed_write_after_gate")
        append_launch_revoke(
            root, invocation_id=str(execution.get("invocation_id")),
            stage=stage, execution_seq=seq, commit_path=commit.commit_path,
            commit_sha256=str(commit.commit_sha256), candidate_sha256=sha,
            candidate_path=path, observed_elapsed=observed,
            committed_wall=float(committed_measured_wall_s),
            remaining_s=float(remaining_s),
            reason="stage_closed_write_after_gate")
        return path, sha, problems
    # The candidate records the intent; the actual gate result is persisted
    # separately so the receipt itself is not mistaken for its own proof.
    result = {
        "schema": EVENT_SCHEMA,
        "kind": "stage_closed_write_after",
        "candidate_sha256": sha,
        "observed_elapsed_s": observed,
        "committed_measured_wall_s": float(committed_measured_wall_s),
        "remaining_s": float(remaining_s),
        "passed": True,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_exclusive(directory / "stage_closed_write_after.json", result)
    # The write-after receipt itself is persistence work.  Sample once more
    # after its fsync/return so the success path cannot hide its own tail.
    release = max(0.0, time.perf_counter() - float(t0))
    if not _nonneg(release) or release >= float(remaining_s) or \
            release > float(committed_measured_wall_s):
        problems.append("stage_closed_write_after_proof_gate")
        append_launch_revoke(
            root, invocation_id=str(execution.get("invocation_id")),
            stage=stage, execution_seq=seq, commit_path=commit.commit_path,
            commit_sha256=str(commit.commit_sha256), candidate_sha256=sha,
            candidate_path=path, observed_elapsed=release,
            committed_wall=float(committed_measured_wall_s),
            remaining_s=float(remaining_s),
            reason="stage_closed_write_after_proof_gate")
    return path, sha, problems


def write_launch_stage_closed_candidate(
        root: Path, *, invocation_id: str, stage: str, execution_seq: int,
        commit: CommitResult, event_candidate: Path | None,
        event_candidate_sha256: str | None, t0: float, remaining_s: float,
        committed_measured_wall_s: float
        ) -> tuple[Path | None, list[str]]:
    directory = _launch_invocation_dir(root, invocation_id)
    path = directory / f"stage_closed_{int(execution_seq):06d}.json"
    payload = {
        "schema": EVENT_SCHEMA,
        "kind": "stage_closed_candidate",
        "candidate": True,
        "invocation_id": invocation_id,
        "stage": stage,
        "execution_seq": int(execution_seq),
        "commit": {"path": path_for_record(commit.commit_path),
                   "sha256": commit.commit_sha256},
        "event_candidate": (
            {"path": path_for_record(event_candidate),
             "sha256": event_candidate_sha256}
            if event_candidate is not None else None),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        sha = write_json_exclusive(path, payload)
    except (OSError, ValueError, TypeError) as exc:
        return None, [f"launch_stage_closed_write:{type(exc).__name__}"]
    observed = max(0.0, time.perf_counter() - float(t0))
    if not _nonneg(observed) or observed >= float(remaining_s) or \
            observed > float(committed_measured_wall_s):
        append_launch_revoke(
            root, invocation_id=invocation_id, stage=stage,
            execution_seq=execution_seq, commit_path=commit.commit_path,
            commit_sha256=str(commit.commit_sha256), candidate_sha256=sha,
            candidate_path=path, observed_elapsed=observed,
            committed_wall=float(committed_measured_wall_s),
            remaining_s=float(remaining_s),
            reason="launch_stage_closed_write_after_gate")
        return path, ["launch_stage_closed_write_after_gate"]
    result = {
        "schema": EVENT_SCHEMA,
        "kind": "stage_closed_write_after",
        "candidate_sha256": sha,
        "observed_elapsed_s": observed,
        "committed_measured_wall_s": float(committed_measured_wall_s),
        "remaining_s": float(remaining_s),
        "passed": True,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_exclusive(
        directory / f"stage_closed_{int(execution_seq):06d}_write_after.json",
        result)
    release = max(0.0, time.perf_counter() - float(t0))
    if not _nonneg(release) or release >= float(remaining_s) or \
            release > float(committed_measured_wall_s):
        append_launch_revoke(
            root, invocation_id=invocation_id, stage=stage,
            execution_seq=int(execution_seq), commit_path=commit.commit_path,
            commit_sha256=str(commit.commit_sha256), candidate_sha256=sha,
            candidate_path=path, observed_elapsed=release,
            committed_wall=float(committed_measured_wall_s),
            remaining_s=float(remaining_s),
            reason="launch_stage_closed_write_after_proof_gate")
        return path, ["launch_stage_closed_write_after_proof_gate"]
    return path, []


def write_launch_terminal_candidate(
        root: Path, *, invocation_id: str, commit: CommitResult,
        event_candidate: Path | None, event_candidate_sha256: str | None,
        t0: float, remaining_s: float, committed_measured_wall_s: float
        ) -> tuple[Path | None, list[str]]:
    directory = _launch_invocation_dir(root, invocation_id)
    path = directory / "terminal_candidate.json"
    payload = {
        "schema": EVENT_SCHEMA,
        "kind": "terminal_candidate",
        "candidate": True,
        "invocation_id": invocation_id,
        "commit": {"path": path_for_record(commit.commit_path),
                   "sha256": commit.commit_sha256},
        "event_candidate": (
            {"path": path_for_record(event_candidate),
             "sha256": event_candidate_sha256}
            if event_candidate is not None else None),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        sha = write_json_exclusive(path, payload)
    except (OSError, ValueError, TypeError) as exc:
        return None, [f"launch_terminal_write:{type(exc).__name__}"]
    observed = max(0.0, time.perf_counter() - float(t0))
    if not _nonneg(observed) or observed >= float(remaining_s) or \
            observed > float(committed_measured_wall_s):
        append_launch_revoke(
            root, invocation_id=invocation_id, stage="F",
            execution_seq=int(commit.commit_path.parent.name), commit_path=commit.commit_path,
            commit_sha256=str(commit.commit_sha256), candidate_sha256=sha,
            candidate_path=path, observed_elapsed=observed,
            committed_wall=float(committed_measured_wall_s),
            remaining_s=float(remaining_s),
            reason="launch_terminal_write_after_gate")
        return path, ["launch_terminal_write_after_gate"]
    write_json_exclusive(directory / "terminal_write_after.json", {
        "schema": EVENT_SCHEMA,
        "kind": "terminal_write_after",
        "candidate_sha256": sha,
        "observed_elapsed_s": observed,
        "committed_measured_wall_s": float(committed_measured_wall_s),
        "remaining_s": float(remaining_s),
        "passed": True,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    release = max(0.0, time.perf_counter() - float(t0))
    if not _nonneg(release) or release >= float(remaining_s) or \
            release > float(committed_measured_wall_s):
        append_launch_revoke(
            root, invocation_id=invocation_id, stage="F",
            execution_seq=int(commit.commit_path.parent.name),
            commit_path=commit.commit_path,
            commit_sha256=str(commit.commit_sha256), candidate_sha256=sha,
            candidate_path=path, observed_elapsed=release,
            committed_wall=float(committed_measured_wall_s),
            remaining_s=float(remaining_s),
            reason="launch_terminal_write_after_proof_gate")
        return path, ["launch_terminal_write_after_proof_gate"]
    return path, []


def write_terminal_candidate(root: Path, *, binding: BindingContext,
                             execution: Mapping[str, Any],
                             commit: CommitResult, t0: float,
                             remaining_s: float,
                             committed_measured_wall_s: float
                             ) -> tuple[Path | None, str | None, list[str]]:
    seq = int(execution.get("execution_seq"))
    directory = _event_dir(root, seq)
    payload = {
        "schema": EVENT_SCHEMA,
        "kind": "terminal_candidate",
        "candidate": True,
        "execution_seq": seq,
        "generation": binding.generation,
        "binding_spec_sha256": binding.spec_sha256,
        "invocation_id": execution.get("invocation_id"),
        "commit": {"path": path_for_record(commit.commit_path),
                   "sha256": commit.commit_sha256},
        "committed_measured_wall_s": float(committed_measured_wall_s),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = directory / "terminal_candidate.json"
    try:
        sha = write_json_exclusive(path, payload)
    except (OSError, ValueError, TypeError) as exc:
        return None, None, [f"terminal_write:{type(exc).__name__}"]
    observed = max(0.0, time.perf_counter() - float(t0))
    problems: list[str] = []
    if not _nonneg(observed) or observed >= float(remaining_s) or \
            observed > float(committed_measured_wall_s):
        problems.append("terminal_write_after_gate")
        append_launch_revoke(
            root, invocation_id=str(execution.get("invocation_id")),
            stage=str(execution.get("stage", "F")), execution_seq=seq,
            commit_path=commit.commit_path,
            commit_sha256=str(commit.commit_sha256), candidate_sha256=sha,
            candidate_path=path, observed_elapsed=observed,
            committed_wall=float(committed_measured_wall_s),
            remaining_s=float(remaining_s),
            reason="terminal_write_after_gate")
        return path, sha, problems
    write_json_exclusive(directory / "terminal_write_after.json", {
        "schema": EVENT_SCHEMA,
        "kind": "terminal_write_after",
        "candidate_sha256": sha,
        "observed_elapsed_s": observed,
        "committed_measured_wall_s": float(committed_measured_wall_s),
        "remaining_s": float(remaining_s),
        "passed": True,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    release = max(0.0, time.perf_counter() - float(t0))
    if not _nonneg(release) or release >= float(remaining_s) or \
            release > float(committed_measured_wall_s):
        problems.append("terminal_write_after_proof_gate")
        append_launch_revoke(
            root, invocation_id=str(execution.get("invocation_id")),
            stage=str(execution.get("stage", "F")), execution_seq=seq,
            commit_path=commit.commit_path,
            commit_sha256=str(commit.commit_sha256), candidate_sha256=sha,
            candidate_path=path, observed_elapsed=release,
            committed_wall=float(committed_measured_wall_s),
            remaining_s=float(remaining_s),
            reason="terminal_write_after_proof_gate")
    return path, sha, problems


def record_return_observation(
        root: Path, *, invocation_id: str, pid: int, ppid: int,
        exit_code: int, signal: int | None, started_perf_counter: float,
        finished_perf_counter: float, last_stage_end_perf_counter: float,
        commits: Mapping[str, Mapping[str, Any]],
        candidate_receipts: Mapping[str, Any],
        write_after: Mapping[str, bool],
        coverage_ok: bool | None = None,
        stage_timings: Mapping[str, Any] | None = None,
        source_kind: str = "outer_wait",
        expected_exit_code: int | None = None,
        conclusion: str | None = None) -> tuple[Path, str, list[str]]:
    """Store only facts observed by the outer caller after the process ends."""
    directory = _launch_invocation_dir(root, invocation_id)
    claim_path = directory / "claim.json"
    if not claim_path.is_file():
        raise FileNotFoundError("launch claim missing")
    claim = strict_read_json(claim_path)
    claim_sha = sha256_file(claim_path)
    claim_problems = _claim_problems(claim, invocation_id=invocation_id,
                                     root=root)
    if claim_problems:
        raise ValueError(f"return observation claim refused: "
                         f"{claim_problems}")
    problems: list[str] = []
    if source_kind not in ("outer_wait", "outer_exec"):
        problems.append("return_source_kind")
    if type(exit_code) is not int:
        problems.append("return_exit_code")
    if signal is not None and type(signal) is not int:
        problems.append("return_signal")
    if not all((_nonneg(started_perf_counter), _nonneg(finished_perf_counter),
                _nonneg(last_stage_end_perf_counter))):
        problems.append("return_timing")
    if _nonneg(started_perf_counter) and _nonneg(finished_perf_counter) and \
            float(finished_perf_counter) + 1e-9 < float(started_perf_counter):
        problems.append("return_time_reverse")
    if _nonneg(last_stage_end_perf_counter) and _nonneg(finished_perf_counter) \
            and float(finished_perf_counter) + 1e-9 < float(
                last_stage_end_perf_counter):
        problems.append("return_time_order")
    if not write_after or not all(value is True for value in write_after.values()):
        problems.append("return_write_after")
    timings = stage_timings if isinstance(stage_timings, Mapping) else {}
    record = {
        "schema": RETURN_OBSERVATION_SCHEMA,
        "scope": "post_execution_observation",
        "invocation_id": invocation_id,
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "run_root_realpath": str(Path(root).resolve()),
        "authorization_sha256": claim.get("authorization_sha256"),
        "spec_sha256": claim.get("spec_sha256"),
        "legacy_head_sha256": claim.get("legacy_head_sha256"),
        "claim_sha256": claim_sha,
        "pid": int(pid), "ppid": int(ppid),
        "source": {"kind": source_kind, "pid": int(pid),
                   "ppid": int(ppid), "exit_code": int(exit_code),
                   "signal": signal},
        "process_returned": True,
        "started_perf_counter": float(started_perf_counter),
        "finished_perf_counter": float(finished_perf_counter),
        "last_stage_end_perf_counter": float(last_stage_end_perf_counter),
        "total_wall_s": max(0.0, float(finished_perf_counter)
                            - float(started_perf_counter)),
        "commits": {k: dict(v) for k, v in commits.items()},
        "candidate_receipts": dict(candidate_receipts),
        "stage_timings": {
            str(stage): dict(rec) for stage, rec in timings.items()
            if isinstance(rec, Mapping)},
        "write_after": {k: bool(v) for k, v in write_after.items()},
        "coverage_ok": False,
        "expected_exit_code": expected_exit_code,
        "conclusion": conclusion,
        "post_return_work_performed": False,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    coverage_problems = _return_timing_coverage_problems(
        record, claim={**claim, "_sha256": claim_sha})
    coverage = not coverage_problems
    if coverage_ok is False:
        coverage = False
    record["coverage_ok"] = bool(coverage)
    problems += coverage_problems
    if not coverage:
        problems.append("return_coverage")
    path = directory / "return_observation.json"
    try:
        sha = write_json_exclusive(path, record)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"return observation write failed: "
                         f"{type(exc).__name__}") from exc
    problems += return_observation_problems(record, invocation_id=invocation_id,
                                            claim={**claim,
                                                   "_sha256": claim_sha})
    return path, sha, problems


def _cohort_mode(root: Path) -> str | None:
    path = _cohort_path(root)
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError, TypeError):
        return None
    return data.get("mode") if isinstance(data, Mapping) else None


def _required_stage_files(root: Path, stage: str) -> list[str]:
    from ireset.eval.w1_real_training_update_provider import (
        STAGE_REQUIRED_FILES,
    )
    names = list(STAGE_REQUIRED_FILES[stage])
    if _cohort_mode(root) == "real":
        if stage == "R0":
            names.append("noise_schedule.npz")
        elif stage == "P":
            names.append("input_admission.json")
    return names


def _artifact_covers(artifacts: Mapping[str, Any], name: str) -> bool:
    for rel in artifacts:
        posix = Path(str(rel)).as_posix()
        if posix == name or posix.endswith("/" + name):
            return True
    return False


def _reference_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_sha(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        return sha256_file(Path(path))
    except OSError:
        return None


def _reference_problems(label: str, path_value: Any,
                        sha_value: Any) -> list[str]:
    path = _reference_path(path_value)
    if path is None or not path.is_file():
        return [f"{label}_reference_missing"]
    if not isinstance(sha_value, str) or sha256_file(path) != sha_value:
        return [f"{label}_reference_drifted"]
    return []


def _report_reference_problems(root: Path, stage: str,
                               report_path: Path | None) -> list[str]:
    """Check the file references embedded in a sealed stage report."""
    if stage == "F":
        return []
    if report_path is None or not Path(report_path).is_file():
        return ["report_missing"]
    try:
        data = read_json(Path(report_path))
    except (OSError, ValueError, TypeError):
        return ["report_unreadable"]
    if not isinstance(data, Mapping):
        return ["report_not_mapping"]
    problems: list[str] = []
    if stage == "R0":
        problems += _reference_problems(
            "registration", data.get("registration"),
            data.get("registration_sha256"))
        registration_path = _reference_path(data.get("registration"))
        if registration_path is None or not registration_path.is_file():
            return problems
        try:
            registration = read_json(registration_path)
        except (OSError, ValueError, TypeError):
            return problems + ["registration_unreadable"]
        if _cohort_mode(root) == "real" and isinstance(registration, Mapping):
            noise = registration.get("noise")
            noise = noise if isinstance(noise, Mapping) else {}
            problems += _reference_problems(
                "noise_schedule", noise.get("schedule_path"),
                noise.get("schedule_sha256"))
        return problems
    models = data.get("models")
    if not isinstance(models, Mapping) or not models:
        return ["report_models"]
    for model in ("deterministic", "diffusion"):
        rec = models.get(model)
        if not isinstance(rec, Mapping):
            problems.append(f"report_model:{model}")
            continue
        problems += _reference_problems(
            f"models.{model}.arrays", rec.get("arrays_path"),
            rec.get("arrays_sha256"))
    if stage == "P":
        admission = data.get("input_admission")
        if admission is None:
            if _cohort_mode(root) == "real":
                problems.append("input_admission_reference_missing")
        elif not isinstance(admission, Mapping):
            problems.append("input_admission_reference")
        else:
            problems += _reference_problems(
                "input_admission", admission.get("path"),
                admission.get("sha256"))
    return problems


def _sealed_evidence_problems(root: Path, stage: str,
                              index_row: Mapping[str, Any]) -> list[str]:
    """Verify the sealed report/summary/artifact/file reference closure.

    A consumer must never accept evidence that changed after sealing; the
    recorded digests are the only binding between the row and its files.
    """
    problems: list[str] = []
    for key in ("report", "summary"):
        rel = index_row.get(key)
        if not rel:
            if key == "report" and index_row.get("complete") is True:
                problems.append("report_missing")
            continue
        path = PROJECT_ROOT / str(rel)
        if not path.is_file():
            problems.append(f"{key}_missing")
            continue
        expected = index_row.get(f"{key}_sha256")
        if not expected:
            problems.append(f"{key}_sha256_missing")
        elif sha256_file(path) != expected:
            problems.append(f"{key}_drifted")
    artifacts = index_row.get("artifacts")
    if isinstance(artifacts, Mapping):
        for rel, expected in artifacts.items():
            path = PROJECT_ROOT / str(rel)
            name = Path(str(rel)).name
            if not path.is_file():
                problems.append(f"artifact_missing:{name}")
            elif not isinstance(expected, str) or sha256_file(path) != expected:
                problems.append(f"artifact_drifted:{name}")
        for name in _required_stage_files(root, stage):
            if not _artifact_covers(artifacts, name):
                problems.append(f"required_file_missing:{name}")
    else:
        problems.append("artifacts_invalid")
    report = index_row.get("report")
    report_path = (PROJECT_ROOT / str(report)) if report else None
    problems += _report_reference_problems(root, stage, report_path)
    return problems


def _charge_final_verification(
        root: Path, stage: str, attempt: str, t0: float,
        budget: Mapping[str, Any], seal: SealResult, seals: dict,
        *, scope: Mapping[str, Any], counter: CallCounter,
        actual_cost: float) -> tuple[SealResult, str | None]:
    """Charge post-seal consumer verification and revoke on overrun.

    The final consumer checks re-hash every sealed artifact.  That work
    happens after the last ``finish_attempt``, so it must be covered by a
    cost re-seal; otherwise a budget can be exceeded while the on-disk cost
    and completion claim stay unchanged.
    """
    row = seals.get("row")
    if not isinstance(row, Mapping) or str(row.get("status")) not in (
            "done", "incomplete", "crashed", "timeout"):
        return seal, None
    recorded_cost = float(row.get("measured_wall_s", 0.0) or 0.0)
    if actual_cost <= recorded_cost:
        return seal, None
    was_complete = row.get("complete") is True
    status = str(row.get("status"))
    extra = {"stage": stage, "scope": dict(scope),
             "call_balance": counter.snapshot()}
    if "selection" in row:
        extra["selection"] = row["selection"]
    sealed = finish_attempt(root, stage, attempt, t0, budget, dict(row),
                            status=status, extra=extra,
                            cost_floor=actual_cost)
    seals["verification_reseal"] = {
        "sealed": sealed.sealed, "terminal": sealed.terminal,
        "problems": sealed.problems, "row": sealed.row,
    }
    combined = list(seal.problems) + list(sealed.problems)
    updated = SealResult(sealed.sealed, sealed.row, combined,
                         sealed.terminal)
    seals.update({"sealed": sealed.sealed, "terminal": sealed.terminal,
                  "problems": combined, "row": sealed.row})
    revoke = None
    if was_complete and not (
            sealed.sealed and sealed.terminal
            and sealed.row.get("complete") is True):
        revoke = "post_seal_verification_budget_exceeded"
    return updated, revoke


def _charge_stage_consumption(
        root: Path, stage: str, attempt: str, t0: float,
        budget: Mapping[str, Any], seal: SealResult, seals: dict,
        *, scope: Mapping[str, Any], counter: CallCounter,
        actual_cost: float, verification_wall: float
        ) -> tuple[SealResult, list[str]]:
    """Charge R0/P/I post-seal consumption before entering downstream.

    ``stage_state`` re-hashes the sealed artifacts.  That time is not part
    of the stage's recorded cost yet, so an expensive consumer check could
    exhaust the budget while the stage still claims completion and the next
    stage starts.  The stage is re-sealed from the actual elapsed time; an
    overrun withdraws its completion claim before the caller continues.
    """
    seals["verification_wall_s"] = verification_wall
    row = seals.get("row")
    if not isinstance(row, Mapping) or str(row.get("status")) not in (
            "done", "incomplete", "crashed", "timeout"):
        return seal, []
    recorded_cost = float(row.get("measured_wall_s", 0.0) or 0.0)
    if actual_cost <= recorded_cost:
        return seal, []
    status = str(row.get("status"))
    extra = {"stage": stage, "scope": dict(scope),
             "call_balance": counter.snapshot()}
    if "numeric_passed" in row:
        extra["numeric_passed"] = row["numeric_passed"]
    resealed = finish_attempt(root, stage, attempt, t0, budget, dict(row),
                              status=status, extra=extra,
                              cost_floor=actual_cost)
    seals["verification_reseal"] = {
        "sealed": resealed.sealed, "terminal": resealed.terminal,
        "problems": resealed.problems, "row": resealed.row,
        "verification_wall_s": verification_wall,
    }
    combined = list(seal.problems) + list(resealed.problems)
    updated = SealResult(resealed.sealed, resealed.row, combined,
                         resealed.terminal)
    seals.update({"sealed": resealed.sealed, "terminal": resealed.terminal,
                  "problems": combined, "row": resealed.row})
    if resealed.sealed and resealed.terminal \
            and resealed.row.get("complete") is True:
        return updated, []
    reason = "post_consumption_verification_budget_exceeded"
    problems = [reason]
    report_path = _reference_path(row.get("report"))
    if report_path is not None and _report_claims_complete(report_path):
        wrow, wprobs, proven = withdraw_attempt_claim(
            root, stage, attempt, report_path, resealed.row, [reason],
            reason=reason, t0=t0, budget=budget,
            call_balance=counter.snapshot())
        problems += wprobs
        seals.update({"sealed": False, "terminal": bool(proven),
                      "problems": problems, "row": wrow})
        return SealResult(False, wrow, problems, bool(proven)), problems
    return updated, problems


def _attempt_consumable(root: Path, stage: str, attempt: str,
                        index_row: Mapping[str, Any],
                        ledger_row: Mapping[str, Any], *,
                        allow_current: tuple[str, str] | None = None
                        ) -> tuple[bool, list[str]]:
    problems: list[str] = []
    problems += _row_identity_problems(index_row, kind="index")
    problems += _row_identity_problems(ledger_row, kind="ledger")
    problems += _call_balance_problems(index_row, ledger_row)
    if index_row.get("complete") is not True:
        problems.append("attempt_not_complete")
    if index_row.get("status") != "done" \
            or ledger_row.get("status") != "done":
        problems.append("attempt_status_not_done")
    problems += _sealed_evidence_problems(root, stage, index_row)
    deltas = stage_attempt_deltas(root, stage, current=allow_current)
    if deltas is None or attempt not in deltas:
        problems.append("attempt_call_delta_unknown")
    return not problems, problems


def _event_json(root: Path, execution_seq: int, name: str) -> Any:
    path = _event_dir(root, execution_seq) / name
    if not path.is_file():
        return None
    try:
        return strict_read_json(path)
    except (OSError, ValueError, TypeError):
        return None


def _invocation_return_problems(root: Path, invocation_id: str) -> list[str]:
    if not isinstance(invocation_id, str) or not invocation_id:
        return ["return_invocation_missing"]
    journal = scan_launch_journals(Path(root).parent)
    if not journal.get("known"):
        return [f"launch_journal:{p}" for p in journal.get("problems") or []]
    for record in journal.get("records") or []:
        if record.get("invocation_id") != invocation_id:
            continue
        p = record.get("return_problems") or []
        return [f"return:{item}" for item in p]
    return ["return_observation_invocation_missing"]


def _v2_public_row_problems(root: Path, stage: str, attempt: str,
                            row: Mapping[str, Any], binding: BindingContext,
                            internal_context: Mapping[str, Any] | None
                            ) -> list[str]:
    seq = row.get("execution_seq")
    if type(seq) is not int:
        return ["v2_execution_seq"]
    problems: list[str] = []
    commit, commit_problems = _commit_problems(
        root, seq, binding=binding, stage=stage, attempt=attempt)
    problems += commit_problems
    candidate = _event_json(root, seq, "stage_closed.json")
    if not isinstance(candidate, Mapping):
        problems.append("stage_closed_missing")
    elif commit is not None:
        if (candidate.get("commit") or {}).get("sha256") != \
                sha256_file(_commit_path(root, seq)):
            problems.append("stage_closed_commit")
        if candidate.get("stage") != stage or \
                candidate.get("attempt") != attempt:
            problems.append("stage_closed_binding")
    after = _event_json(root, seq, "stage_closed_write_after.json")
    if not isinstance(after, Mapping) or after.get("passed") is not True:
        problems.append("stage_closed_write_after_missing")
    current_internal = (
        isinstance(internal_context, Mapping)
        and internal_context.get("invocation_id") == row.get("invocation_id")
        and internal_context.get("stage") in (stage, None)
        and internal_context.get("attempt") in (attempt, None)
        and internal_context.get("allow_current") is True)
    if not current_internal:
        problems += _invocation_return_problems(root, str(
            row.get("invocation_id")))
    return problems


def stage_state(root: Path, stage: str,
                attempt: str | None = None, *,
                allow_current: tuple[str, str] | None = None,
                generation: str | None = None,
                binding: BindingContext | None = None,
                internal_context: Mapping[str, Any] | None = None,
                ) -> tuple[dict | None, list[str]]:
    """Selected-attempt consumer.

    Without ``attempt`` the historical single-attempt contract is preserved
    (exactly one terminal attempt).  With several terminal attempts one
    settled failure may coexist with exactly one consumable attempt; the
    unique consumable attempt is selected.  More than one consumable
    candidate, an unsealed/withdrawn pick or an explicitly wrong attempt is
    refused.
    """
    problems: list[str] = []
    migrated = _migration_present(root)
    if migrated:
        if generation is None:
            return None, ["generation_ambiguous"]
        if generation not in (GENERATION_001, GENERATION_002):
            return None, [f"generation_unknown:{generation}"]
        if binding is None:
            binding, binding_problems = load_binding(
                root, require_active=(generation == GENERATION_002))
            if binding is None:
                return None, binding_problems or ["binding_missing"]
            problems += binding_problems
    try:
        index = [r for r in load_index(root) if r.get("stage") == stage]
        ledger = [r for r in load_ledger(root) if r.get("stage") == stage]
    except (OSError, ValueError, TypeError):
        return None, ["stage_registration_unreadable"]
    if migrated:
        index = [r for r in index
                 if (r.get("generation", GENERATION_001) == generation)]
        ledger = [e for e in ledger
                  if (e.get("generation", GENERATION_001) == generation)]
    disk = _disk_attempts(root, stage)
    all_index = [r for r in load_index(root) if r.get("stage") == stage]
    all_ledger = [e for e in load_ledger(root) if e.get("stage") == stage]
    index_by = {r.get("attempt"): r for r in index}
    ledger_by = {e.get("attempt"): e for e in ledger}
    all_index_by = {r.get("attempt"): r for r in all_index}
    all_ledger_by = {e.get("attempt"): e for e in all_ledger}
    if (len(index_by) != len(index) or len(ledger_by) != len(ledger)
            or set(index_by) != set(ledger_by)
            or set(all_index_by) != set(all_ledger_by)
            or set(disk) != set(all_index_by)):
        return None, ["stage_attempt_sets_disagree"]
    if not index_by:
        return None, ["stage_not_registered"]
    problems += _reservation_problems(root, allow_current=allow_current)
    if attempt is not None and attempt not in index_by:
        if attempt in all_index_by and migrated:
            return None, [f"selected_attempt_generation:{attempt}"]
        return None, ["selected_attempt_missing"]
    internal_invocation = (
        internal_context.get("invocation_id")
        if isinstance(internal_context, Mapping)
        and internal_context.get("allow_current") is True else None)
    history = verify_history(root, allow_current=allow_current,
                             allow_invocation=internal_invocation,
                             binding=binding)
    if history:
        nonterminal = f"history_not_terminal:{stage}"
        if any(p.startswith(nonterminal) for p in history) and not any(
                r.get("status") in ("done", "crashed", "timeout",
                                    "incomplete")
                for r in index_by.values()):
            return None, ["stage_not_terminal"] + problems
        only_delta = all(p.startswith("history_deltas:") for p in history)
        if only_delta and len(index_by) == 1:
            name = next(iter(index_by))
            _allowed, attempt_problems = _attempt_consumable(
                root, stage, name, index_by[name], ledger_by[name],
                allow_current=allow_current)
            if migrated and generation == GENERATION_002:
                attempt_problems += _v2_public_row_problems(
                    root, stage, name, index_by[name], binding,
                    internal_context)
                _allowed = not attempt_problems
            return index_by[name], attempt_problems + history
        return None, history
    terminal = [a for a, r in index_by.items()
                if r.get("status") in ("done", "crashed", "timeout",
                                       "incomplete")]
    consumable: list[str] = []
    per_attempt: dict[str, list[str]] = {}
    for name in terminal:
        allowed, attempt_problems = _attempt_consumable(
            root, stage, name, index_by[name], ledger_by[name],
            allow_current=allow_current)
        if migrated and generation == GENERATION_002:
            attempt_problems += _v2_public_row_problems(
                root, stage, name, index_by[name], binding,
                internal_context)
            allowed = not attempt_problems
        per_attempt[name] = attempt_problems
        if allowed:
            consumable.append(name)
    if attempt is not None:
        if attempt not in terminal:
            return None, ["selected_attempt_not_terminal"] + problems
        if len(consumable) > 1:
            return None, ["stage_multiple_consumable"] + problems
        if consumable != [attempt]:
            return None, per_attempt.get(attempt, []) + problems
        return index_by[attempt], problems
    if len(consumable) > 1:
        return None, ["stage_multiple_consumable"] + problems
    if len(consumable) == 1:
        row = index_by[consumable[0]]
        return row, problems
    # No consumable attempt: preserve the historical diagnostics of the
    # single-terminal case; otherwise report the explicit refusal.
    if len(terminal) == 1:
        name = terminal[0]
        return None, per_attempt.get(name, []) + problems
    return None, ["stage_not_terminal"] + problems


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
                        reserved: Mapping[str, Any],
                        *, execution: Mapping[str, Any] | None = None) -> Path:
    path = stage_dir(root, stage) / attempt / "call_reservations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = reserved.get("counts") if isinstance(reserved, Mapping) else None
    if not isinstance(counts, Mapping) or set(counts) != set(CALL_KINDS):
        raise ValueError("reservation must cover all call kinds")
    for kind, value in counts.items():
        if not _json_int(value):
            raise ValueError(f"reservation {kind} is not a JSON integer")
    execution_fields = _execution_fields(
        execution, root=root, stage=stage, attempt=attempt)
    write_json(path, {"schema": "w1_real_training_update_reservation_v1",
                      "reserved": {k: int(v) for k, v in counts.items()},
                      "consumed": None,
                      **execution_fields})
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
                    selected_attempts=None, selected_chain_counts=None,
                    selected_chain_training_updates=None,
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


def _withdrawn_state_problems(root: Path, stage: str,
                              attempt: str) -> list[str]:
    """Terminal, non-claiming state of one attempt without needing D."""
    problems: list[str] = []
    try:
        index = [r for r in load_index(root)
                 if r.get("stage") == stage and r.get("attempt") == attempt]
        ledger = [r for r in load_ledger(root)
                  if r.get("stage") == stage and r.get("attempt") == attempt]
    except (OSError, ValueError, TypeError):
        return ["stage_unreadable"]
    if len(index) != 1 or len(ledger) != 1:
        return ["stage_row_unreadable"]
    row, entry = index[0], ledger[0]
    if row.get("status") not in ("done", "crashed", "timeout", "incomplete"):
        problems.append("attempt_not_terminal")
    if row.get("status") != entry.get("status"):
        problems.append("attempt_state_mismatch")
    if row.get("complete") is not entry.get("complete"):
        problems.append("attempt_complete_mismatch")
    if row.get("status") != "done" and row.get("complete") is not False:
        problems.append("attempt_claims_complete")
    for key in ("report", "summary"):
        if row.get(key) and not (PROJECT_ROOT / row[key]).is_file():
            problems.append(f"{key}_missing")
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
        state_problems = _withdrawn_state_problems(root, stage, attempt)
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
    "src/ireset/eval/w1_real_training_update_provider.py",
    "scripts/audit_w1_real_training_update.py",
    "tests/test_w1_real_training_update_audit.py",
    "tests/test_w1_real_training_update_provider.py",
    "src/ireset/train/trainer.py",
    "src/ireset/train/losses.py",
    "src/ireset/model/backbone.py",
    "src/ireset/model/conditioner.py",
    "src/ireset/model/deterministic.py",
    "src/ireset/model/tokens.py",
    "src/ireset/diffusion/generator.py",
    "src/ireset/eval/training_objective_audit.py",
    "src/ireset/eval/w1_real_fk_audit.py",
    "src/ireset/eval/w1_real_objective_audit.py",
    "src/ireset/kinematics/torch_adapter.py",
    "src/ireset/kinematics/rotation_adapter.py",
    "src/ireset/geometry/distance_adapter.py",
    "src/ireset/embodied/g1_spec.py",
    "src/ireset/paths.py",
    "pyproject.toml",
    "uv.lock",
    "configs/train/omniretarget_motion.yaml",
    "docs/plans/omniretarget_w1_real_training_update_provider_execution_plan.md",
    "docs/plans/omniretarget_w1_real_training_update_alpha_fix_plan.md",
    "docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md",
)
PREREG_DRAFT = (
    "docs/plans/omniretarget_w1_real_training_update_audit_prereg_draft.md")
EXECUTION_PLAN = (
    "docs/plans/omniretarget_w1_real_training_update_alpha_fix_plan.md")


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


#: Runtime facts whose drift invalidates the frozen numeric calibration.  The
#: check is metadata/source-hash only; it never runs a numeric probe.
RUNTIME_IDENTITY_KEYS = (
    "python", "torch", "numpy", "scipy", "cuda_available",
    "cuda_device_count", "cuda_version", "cpu_threads", "diffusers",
)
ADAM_SOURCE_MODULES = ("torch.optim.adam", "torch.optim.adamw")
SCHEDULER_SOURCE_MODULES = ("diffusers.schedulers.scheduling_ddpm",
                          "diffusers.schedulers.scheduling_ddim")


def _adamw_source_sha256() -> dict:
    source = {}
    for name in ADAM_SOURCE_MODULES:
        try:
            mod = __import__(name, fromlist=["x"])
            source[name] = sha256_file(Path(mod.__file__))
        except (ImportError, OSError, TypeError):
            source[name] = None
    return source


def _scheduler_source_sha256() -> dict:
    source = {}
    for name in SCHEDULER_SOURCE_MODULES:
        try:
            mod = __import__(name, fromlist=["x"])
            source[name] = sha256_file(Path(mod.__file__))
        except (ImportError, OSError, TypeError):
            source[name] = None
    return source


def runtime_identity_facts() -> dict:
    env = environment_facts()
    facts = {key: env.get(key) for key in RUNTIME_IDENTITY_KEYS}
    facts["device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    try:
        facts["mujoco"] = importlib.metadata.version("mujoco")
    except importlib.metadata.PackageNotFoundError:
        facts["mujoco"] = None
    facts["adamw_source_sha256"] = _adamw_source_sha256()
    try:
        facts["diffusers"] = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        facts["diffusers"] = None
    facts["scheduler_source_sha256"] = _scheduler_source_sha256()
    return facts


def runtime_identity_problems(reference: Mapping[str, Any] | None,
                              current: Mapping[str, Any] | None = None
                              ) -> list[str]:
    if not isinstance(reference, Mapping):
        return ["runtime_identity_missing"]
    current = runtime_identity_facts() if current is None else current
    problems = [f"runtime_identity:{key}"
                for key in (*RUNTIME_IDENTITY_KEYS, "device_name", "mujoco")
                if reference.get(key) != current.get(key)]
    reference_sources = reference.get("adamw_source_sha256")
    current_sources = current.get("adamw_source_sha256") or {}
    if not isinstance(reference_sources, Mapping) or any(
            not isinstance(v, str) for v in reference_sources.values()):
        problems.append("runtime_identity:adamw_source_sha256")
    else:
        for name in ADAM_SOURCE_MODULES:
            if reference_sources.get(name) != current_sources.get(name):
                problems.append(f"runtime_identity:source:{name}")
    reference_schedulers = reference.get("scheduler_source_sha256")
    current_schedulers = current.get("scheduler_source_sha256") or {}
    for name in SCHEDULER_SOURCE_MODULES:
        expected = ((reference_schedulers or {}).get(name)
                    if isinstance(reference_schedulers, Mapping) else None)
        if not isinstance(expected, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected) \
                or expected != current_schedulers.get(name):
            problems.append(f"runtime_identity:source:{name}")
    return problems


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
    if fixture.get("alpha_evaluation") != ALPHA_EVALUATION:
        problems.append("fixture_alpha_evaluation")
    if fixture.get("alpha_effective_sha256") != _hash_array(
            effective_alpha_reference(provider.config)):
        problems.append("fixture_alpha_effective_sha256")
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
           budget: Mapping[str, Any], provider: Any,
           counter: CallCounter, roster: Mapping[str, Any],
           *, mode: str, tolerance: Mapping[str, Any],
           project_root: Path = PROJECT_ROOT,
           admission_package: Mapping[str, Any] | None = None,
           authorization: Mapping[str, Any] | None = None,
           execution: Mapping[str, Any] | None = None) -> dict:
    execution = dict(execution or {})
    v2 = bool(execution)
    base = stage_dir(root, stage) / attempt
    base.mkdir(parents=True, exist_ok=True)
    fixture = provider.fixture or {} if mode == "synthetic" else {}
    metadata = provider.metadata() if mode == "real" else {}
    problems: list[str] = list(fixture.get("problems") or [])
    problems += [f"metadata:{p}" for p in metadata.get("problems") or []]
    gaps: list[str] = []
    # tolerance + fp32 contract
    problems += [f"tolerance:{p}" for p in tolerance_contract_problems(
        tolerance, recompute=False)]
    fp32_ok = True
    if mode == "real":
        fp32_ok = bool(tolerance.get("fp32_contract"))
        problems += [f"fp32:{p}" for p in tolerance.get("fp32_problems", [])
                     if p]
        problems += [f"r0_runtime_identity:{p}" for p in
                     runtime_identity_problems(
                         (tolerance.get("fp32_contract") or {}).get(
                             "runtime_identity"))]
        try:
            from ireset.eval.w1_real_training_update_provider import (
                calibration_verification_problems,
            )
            for problem in calibration_verification_problems(
                    (admission_package or {}).get("calibration_verification")):
                problems.append(f"r0_calibration:{problem}")
        except ImportError:  # pragma: no cover
            problems.append("r0_calibration_module_missing")
    # roster/schema/noise
    roster_here = roster_problems(provider.roster, roster)
    problems += [f"roster:{p}" for p in roster_here]
    if mode == "synthetic":
        problems += [f"fixture:{p}" for p in _fixture_schema_problems(provider)]
    schedule = draw_noise_path(NOISE_SEED)
    counter.add("cpu_draw_units", DRAW_UNITS_PER_PATH)
    noise_ok = True
    if mode == "synthetic":
        for a, b in zip(schedule, provider.noise):
            if (not torch.equal(a["t"], b["t"])
                    or not torch.equal(a["epsilon"], b["epsilon"])):
                noise_ok = False
                problems.append("r0_noise_matrix")
                break
    noise_path = None
    if mode == "real":
        noise_path = base / "noise_schedule.npz"
        np.savez(
            noise_path,
            **{f"u{u['update']}m{u['micro']}__t": u["t"].numpy()
               for u in schedule},
            **{f"u{u['update']}m{u['micro']}__epsilon": u["epsilon"].numpy()
               for u in schedule},
            **{f"u{u['update']}m{u['micro']}__state_before":
               u["generator_state_before"].numpy() for u in schedule},
            **{f"u{u['update']}m{u['micro']}__state_after":
               u["generator_state_after"].numpy() for u in schedule})
    alpha = alpha_cumprod_reference(provider.config)
    if _hash_array(alpha) != fixture.get("alpha_cumprod_sha256") \
            and mode == "synthetic":
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
        "schema": (REGISTRATION_SCHEMA_V2 if v2 else REGISTRATION_SCHEMA),
        "mode": mode,
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "stage": stage,
        "attempt": attempt,
        **({"generation": execution.get("generation"),
            "binding_spec_sha256": execution.get("binding_spec_sha256"),
            "execution_seq": execution.get("execution_seq"),
            "predecessor_kind": execution.get("predecessor_kind"),
            "previous_head_sha256": execution.get("previous_head_sha256"),
            "invocation_id": execution.get("invocation_id"),
            "legacy_head_sha256": execution.get("legacy_head_sha256"),
            "claim_path": execution.get("claim_path"),
            "claim_sha256": execution.get("claim_sha256")}
           if v2 else {}),
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
                  "alpha_cumprod_sha256": _hash_array(alpha),
                  "alpha_source": ("config_recompute; parent array value "
                                   "checked by the P input admission"
                                   if mode == "real" else "config_recompute"),
                  "schedule_path": (path_for_record(noise_path)
                                    if noise_path is not None else None),
                  "schedule_sha256": (sha256_file(noise_path)
                                      if noise_path is not None else None)},
        "roster": roster,
        "row_catalog": {k: catalog[k] for k in
                        ("n_rows", "counts", "keys_sha256", "formula")},
        "config": provider.config,
        "budget": dict(budget),
        "call_caps": dict(CALL_CAPS),
        "call_nominal": {k: stage_nominal_for_mode(mode, k)
                         for k in STAGE_BUDGETS},
        "chain_nominal": chain_nominal_for_mode(mode),
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
        "runtime_identity": runtime_identity_facts(),
        "resource_bound": bound,
        "disk_free_bytes": int(free),
        "provider": (metadata if mode == "real" else provider.spec()),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    registration_path = base / "registration.json"
    write_json(registration_path, registration)
    registration_sha = sha256_file(registration_path)
    admission = {
        "schema": ("w1_real_training_update_r0_admission_v2" if v2
                   else "w1_real_training_update_r0_admission_v1"),
        "mode": mode,
        "generation": execution.get("generation") if v2 else GENERATION_001,
        "binding_spec_sha256": (execution.get("binding_spec_sha256")
                                if v2 else None),
        "execution_seq": execution.get("execution_seq") if v2 else None,
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
        "metadata_only": bool(mode == "real"),
        "payload_reads": {
            "checkpoint_torch_load": 0,
            "npz_values_read": 0,
            "model_construction_calls": 0,
            "muJoCo_compile_calls": 0,
        } if mode == "real" else None,
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
        "noise_schedule": (path_for_record(noise_path)
                           if noise_path is not None else None),
        "noise_schedule_sha256": (sha256_file(noise_path)
                                  if noise_path is not None else None),
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
    runtime_identity = runtime_identity_facts()
    source = dict(runtime_identity["adamw_source_sha256"])
    device_name = torch.cuda.get_device_name(0)
    record = {
        "schema": FP32_SCHEMA,
        "device": device,
        "device_name": device_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": source,
        "runtime_identity": runtime_identity,
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
    problems += [f"fp32_{p}" for p in runtime_identity_problems(
        contract.get("runtime_identity"))]
    if recompute:
        fresh = build_fp32_contract("cuda")
        for key in ("node_graph", "probe_mismatches", "two_step_match",
                    "source_sha256", "runtime_identity", "valid"):
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
        "execution_plan_sha256": None,
        "calibration_verification_sha256": None,
        "acceptance_sha256": None,
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


MIGRATION_AUTHORIZATION_KEYS = (
    "run_root_realpath", "source_marker_sha256",
    "source_authorization_sha256", "snapshot_sha256", "spec_sha256",
    "migration_admission_sha256", "legacy_head_sha256",
    "launch_journal", "target_generation", "plan_sha256",
    "implementation_sha256", "runtime_sha256", "gate_b_sha256",
    "real_admission_sha256", "execution_plan_sha256", "prereg_sha256",
    "stage_plan", "prior_updates", "incremental_update_limit",
    "cohort_update_limit_for_this_authorization", "prior_cost_s",
    "remaining_s", "invocation_id", "write_after_protocol",
    "return_observation_required", "execution_window", "unique_entry",
)


def migration_authorization_template() -> dict:
    return {
        "schema": AUTHORIZATION_SCHEMA_V2,
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
        "updates": {
            **EXPECTED_UPDATES,
            "prior_updates": MIGRATION_PRIOR_UPDATES,
            "incremental_update_limit": MIGRATION_INCREMENTAL_UPDATE_LIMIT,
            "cohort_update_limit_for_this_authorization":
                MIGRATION_COHORT_UPDATE_LIMIT,
        },
        "resource_caps": dict(RESOURCE_CAPS),
        "parent_registration_sha256": PARENT_FILES[
            "objective_registration"][1],
        "draft_sha256": None,
        "execution_plan_sha256": None,
        "calibration_verification_sha256": None,
        "acceptance_sha256": None,
        "tolerance_evidence_sha256": None,
        "fp32_contract_sha256": None,
        "admission_package_sha256": None,
        "implementation_files": None,
        "migration": {key: None for key in MIGRATION_AUTHORIZATION_KEYS},
        "notes": ("Template only. This is a second-generation binding over "
                  "the existing cohort; it does not reset budgets, attempts "
                  "or the two already executed diagnostic updates."),
    }


def build_migration_authorization_draft(
        project_root: Path, *, binding: BindingContext,
        snapshot_path: Path, spec_path: Path,
        migration_admission_path: Path,
        migration_admission_sha256: str,
        real_admission_path: Path | None = None,
        real_admission_sha256: str | None = None,
        gate_b_path: Path | None = None,
        gate_b_sha256: str | None = None,
        invocation_id: str | None = None,
        authorized: bool = False) -> dict:
    root = Path(project_root)
    impl = implementation_identity(root)
    runtime = runtime_identity_facts()
    record = migration_authorization_template()
    record.update(
        template_only=not authorized,
        authorized=bool(authorized),
        granted_by=None if not authorized else "USER_REQUIRED",
        granted_wall=None if not authorized else
        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        draft_sha256=sha256_file(root / PREREG_DRAFT),
        execution_plan_sha256=sha256_file(root / MIGRATION_PLAN),
        calibration_verification_sha256=(
            (binding.spec.get("bindings") or {}).get("gate_b") or {}).get(
                "calibration_verification_sha256"),
        acceptance_sha256=((binding.spec.get("bindings") or {}).get(
            "gate_b") or {}).get("acceptance_sha256"),
        tolerance_evidence_sha256=((binding.spec.get("bindings") or {}).get(
            "gate_b") or {}).get("tolerance_evidence_sha256"),
        fp32_contract_sha256=((binding.spec.get("bindings") or {}).get(
            "gate_b") or {}).get("fp32_contract_sha256"),
        admission_package_sha256=real_admission_sha256,
        implementation_files=impl["files"],
    )
    if authorized:
        record["checklist"] = {key: True for key in AUTHORIZATION_REQUIRED}
    record["updates"] = {
        **EXPECTED_UPDATES,
        "prior_updates": MIGRATION_PRIOR_UPDATES,
        "incremental_update_limit": MIGRATION_INCREMENTAL_UPDATE_LIMIT,
        "cohort_update_limit_for_this_authorization":
            MIGRATION_COHORT_UPDATE_LIMIT,
    }
    record["migration"] = {
        "run_root_realpath": str(Path(root / str(
            binding.spec.get("run_root_realpath"))).resolve())
        if not Path(str(binding.spec.get("run_root_realpath"))).is_absolute()
        else str(Path(str(binding.spec.get("run_root_realpath"))).resolve()),
        "source_marker_sha256": binding.snapshot.get("marker_sha256"),
        "source_authorization_sha256": binding.snapshot.get(
            "authorization_sha256"),
        "snapshot_sha256": binding.snapshot_sha256,
        "spec_sha256": binding.spec_sha256,
        "migration_admission_sha256": migration_admission_sha256,
        "legacy_head_sha256": binding.legacy_head_sha256,
        "launch_journal": {
            "relative_path": LAUNCH_JOURNAL_DIR,
            "manifest_sha256": (binding.snapshot.get(
                "launch_journal") or {}).get("manifest_sha256"),
        },
        "target_generation": binding.generation,
        "plan_sha256": sha256_file(root / MIGRATION_PLAN),
        "implementation_sha256": canonical_sha256(impl),
        "runtime_sha256": canonical_sha256(runtime),
        "gate_b_sha256": gate_b_sha256,
        "real_admission_sha256": real_admission_sha256,
        "execution_plan_sha256": sha256_file(root / MIGRATION_PLAN),
        "prereg_sha256": sha256_file(root / PREREG_DRAFT),
        "stage_plan": [dict(item) for item in MIGRATION_STAGE_PLAN],
        "prior_updates": MIGRATION_PRIOR_UPDATES,
        "incremental_update_limit": MIGRATION_INCREMENTAL_UPDATE_LIMIT,
        "cohort_update_limit_for_this_authorization":
            MIGRATION_COHORT_UPDATE_LIMIT,
        "prior_cost_s": MIGRATION_SOURCE_COST_S,
        "remaining_s": TOTAL_BUDGET_S - MIGRATION_SOURCE_COST_S,
        "invocation_id": invocation_id,
        "write_after_protocol": {
            "sample": "post_publish_perf_counter - stage_t0",
            "requires": ["finite", "non_negative", "elapsed<remaining_s",
                          "elapsed<=committed_measured_wall_s"],
        },
        "return_observation_required": True,
        "execution_window": {
            "starts_at": "launch_claim.t0",
            "ends_at": "outer_actual_process_return",
            "post_return_observation_can_execute_work": False,
        },
        "unique_entry": True,
    }
    return record


def _migration_authorization_problems(record: Mapping[str, Any],
                                      *, expected: Mapping[str, Any] | None
                                      ) -> list[str]:
    problems: list[str] = []
    migration = record.get("migration")
    if not isinstance(migration, Mapping):
        return ["authorization_migration_missing"]
    unknown = set(migration) - set(MIGRATION_AUTHORIZATION_KEYS)
    if unknown:
        problems.append(f"authorization_migration_unknown:{sorted(unknown)}")
    missing = [key for key in MIGRATION_AUTHORIZATION_KEYS
               if key not in migration]
    if missing:
        problems.append(f"authorization_migration_missing_keys:{missing}")
    for key in ("run_root_realpath", "source_marker_sha256",
                "source_authorization_sha256", "snapshot_sha256",
                "spec_sha256", "migration_admission_sha256",
                "legacy_head_sha256", "plan_sha256", "implementation_sha256",
                "runtime_sha256", "gate_b_sha256", "real_admission_sha256",
                "execution_plan_sha256", "prereg_sha256",
                "invocation_id"):
        value = migration.get(key)
        if not isinstance(value, str) or not value:
            problems.append(f"authorization_migration_value:{key}")
    for key in ("snapshot_sha256", "spec_sha256",
                "migration_admission_sha256", "legacy_head_sha256",
                "source_marker_sha256", "source_authorization_sha256",
                "plan_sha256", "implementation_sha256", "runtime_sha256",
                "gate_b_sha256", "real_admission_sha256",
                "execution_plan_sha256", "prereg_sha256"):
        value = migration.get(key)
        if isinstance(value, str) and not re.fullmatch(r"[0-9a-f]{64}", value):
            problems.append(f"authorization_migration_hash:{key}")
    if migration.get("target_generation") != GENERATION_002:
        problems.append("authorization_migration_generation")
    if migration.get("stage_plan") != list(MIGRATION_STAGE_PLAN):
        problems.append("authorization_migration_stage_plan")
    if type(migration.get("prior_updates")) is not int or \
            migration.get("prior_updates") != MIGRATION_PRIOR_UPDATES or \
            type(migration.get("incremental_update_limit")) is not int or \
            migration.get("incremental_update_limit") != \
            MIGRATION_INCREMENTAL_UPDATE_LIMIT or \
            type(migration.get(
                "cohort_update_limit_for_this_authorization")) is not int or \
            migration.get("cohort_update_limit_for_this_authorization") != \
            MIGRATION_COHORT_UPDATE_LIMIT:
        problems.append("authorization_migration_update_limits")
    if migration.get("prior_updates") != record.get("updates", {}).get(
            "prior_updates") or migration.get("incremental_update_limit") != \
            record.get("updates", {}).get("incremental_update_limit"):
        problems.append("authorization_migration_update_binding")
    if not _nonneg(migration.get("remaining_s")) or abs(
            float(migration.get("remaining_s", -1.0))
            - (TOTAL_BUDGET_S - MIGRATION_SOURCE_COST_S)) > 1e-9:
        problems.append("authorization_migration_remaining")
    if not _nonneg(migration.get("prior_cost_s")) or abs(
            float(migration.get("prior_cost_s", -1.0))
            - MIGRATION_SOURCE_COST_S) > 1e-9:
        problems.append("authorization_migration_prior_cost")
    journal = migration.get("launch_journal")
    if not isinstance(journal, Mapping) or \
            journal.get("relative_path") != LAUNCH_JOURNAL_DIR or \
            not isinstance(journal.get("manifest_sha256"), str):
        problems.append("authorization_migration_launch_journal")
    protocol = migration.get("write_after_protocol")
    if not isinstance(protocol, Mapping) or \
            protocol.get("sample") != "post_publish_perf_counter - stage_t0":
        problems.append("authorization_migration_write_after")
    window = migration.get("execution_window")
    if not isinstance(window, Mapping) or \
            window.get("ends_at") != "outer_actual_process_return":
        problems.append("authorization_migration_window")
    if migration.get("return_observation_required") is not True or \
            migration.get("unique_entry") is not True:
        problems.append("authorization_migration_return")
    if expected is not None:
        for key, want in expected.items():
            if _jsonable(migration.get(key)) != _jsonable(want):
                problems.append(f"authorization_migration_binding:{key}")
    return problems


def authorization_problems(record: Mapping[str, Any] | None) -> list[str]:
    problems: list[str] = []
    if not isinstance(record, Mapping):
        return ["authorization_not_a_mapping"]
    if record.get("schema") == AUTHORIZATION_SCHEMA_V2:
        problems += _migration_authorization_problems(record, expected=None)
        schema = AUTHORIZATION_SCHEMA_V2
    else:
        schema = AUTHORIZATION_SCHEMA
    if record.get("schema") != schema:
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
    if schema == AUTHORIZATION_SCHEMA_V2 and isinstance(updates, Mapping):
        if type(updates.get("prior_updates")) is not int or \
                updates.get("prior_updates") != MIGRATION_PRIOR_UPDATES or \
                type(updates.get("incremental_update_limit")) is not int or \
                updates.get("incremental_update_limit") != \
                MIGRATION_INCREMENTAL_UPDATE_LIMIT or \
                type(updates.get(
                    "cohort_update_limit_for_this_authorization")) is not int \
                or updates.get(
                    "cohort_update_limit_for_this_authorization") != \
                MIGRATION_COHORT_UPDATE_LIMIT or \
                int(updates.get("prior_updates", -1)) + int(
                    updates.get("incremental_update_limit", -1)) != int(
                        updates.get("cohort_update_limit_for_this_authorization",
                                    -2)):
            problems.append("authorization_updates_v2")
    if _jsonable(record.get("resource_caps")) != _jsonable(RESOURCE_CAPS):
        problems.append("authorization_resources")
    if record.get("parent_registration_sha256") != PARENT_FILES[
            "objective_registration"][1]:
        problems.append("authorization_parent_registration")
    for key in ("draft_sha256", "execution_plan_sha256",
                "calibration_verification_sha256", "acceptance_sha256",
                "tolerance_evidence_sha256", "fp32_contract_sha256",
                "admission_package_sha256"):
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


MIGRATION_ADMISSION_KEYS = (
    "passed", "problems", "status", "run_root", "snapshot", "spec",
    "legacy_head_sha256", "call_balance_after", "budgets", "attempts",
    "launch_journal", "implementation", "gate_b", "real_admission",
    "read_scope", "authorization_required", "created_wall",
)


def migration_admission_problems(record: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(record, Mapping):
        return ["migration_admission_not_a_mapping"]
    problems: list[str] = []
    if record.get("schema") != MIGRATION_ADMISSION_SCHEMA:
        problems.append("migration_admission_schema")
    if record.get("passed") is not True:
        problems.append("migration_admission_not_passed")
    if record.get("problems"):
        problems.append("migration_admission_has_problems")
    status = record.get("status")
    if not isinstance(status, Mapping):
        problems.append("migration_admission_status")
    else:
        if status.get("authorized") is not False or \
                status.get("registered") is not False or \
                status.get("run") != "not_run":
            problems.append("migration_admission_must_be_readonly")
    snapshot = record.get("snapshot") or {}
    spec = record.get("spec") or {}
    for label, rec in (("snapshot", snapshot), ("spec", spec)):
        if not isinstance(rec, Mapping) or not isinstance(
                rec.get("sha256"), str) or not re.fullmatch(
                    r"[0-9a-f]{64}", rec["sha256"]):
            problems.append(f"migration_admission_{label}")
    if snapshot.get("legacy_head_sha256") != record.get("legacy_head_sha256"):
        problems.append("migration_admission_legacy")
    if (record.get("launch_journal") or {}).get("relative_path") != \
            LAUNCH_JOURNAL_DIR:
        problems.append("migration_admission_launch_journal")
    scope = record.get("read_scope")
    if not isinstance(scope, Mapping):
        problems.append("migration_admission_read_scope")
    else:
        for key in ("checkpoint_payload_read", "npz_values_read",
                    "model_constructor_calls", "mujoco_compile_calls",
                    "forward_calls", "vjp_calls", "optimizer_steps",
                    "cohort_created", "registered_attempt_created"):
            if scope.get(key) not in (0, False):
                problems.append(f"migration_admission_read_scope:{key}")
    if record.get("authorization_required") is not True:
        problems.append("migration_admission_authorization_required")
    return problems


def build_migration_admission(
        project_root: Path, *, snapshot_path: Path, snapshot_sha256: str,
        snapshot: Mapping[str, Any], spec_path: Path, spec_sha256: str,
        spec: Mapping[str, Any], real_admission_path: Path | None = None,
        real_admission_sha256: str | None = None,
        real_admission: Mapping[str, Any] | None = None,
        gate_b_path: Path | None = None,
        gate_b_sha256: str | None = None) -> dict:
    root = Path(project_root)
    run_root = Path(str(snapshot.get("run_root"))).expanduser().resolve()
    problems: list[str] = []
    problems += _snapshot_problems(snapshot, run_root=run_root, root=run_root)
    problems += binding_spec_problems(
        spec, root=run_root, snapshot=snapshot,
        require_current_implementation=True, project_root=root)
    if sha256_file(snapshot_path) != snapshot_sha256:
        problems.append("snapshot_file_sha")
    if sha256_file(spec_path) != spec_sha256:
        problems.append("spec_file_sha")
    source_dir = Path(snapshot_path).parent / "source"
    for name, raw_name in (("index.json", "index.json"),
                           ("ledger.json", "ledger.json"),
                           ("cohort.json", "cohort.json")):
        expected = ((snapshot.get("source_copies") or {}).get(name) or {}).get(
            "sha256")
        path = source_dir / raw_name
        if not path.is_file():
            problems.append(f"source_copy_missing:{name}")
        elif sha256_file(path) != expected:
            problems.append(f"source_copy_sha:{name}")
    # The head must be independently reproducible from the frozen copies and
    # the current old rows, not merely trusted from the embedded summary.
    try:
        source_index = strict_read_json(source_dir / "index.json")
        source_ledger = strict_read_json(source_dir / "ledger.json")
        source_marker = strict_read_json(source_dir / "cohort.json")
        source_rows = source_index.get("attempts")
        source_entries = source_ledger.get("invocations")
        if not isinstance(source_rows, list) or not isinstance(
                source_entries, list):
            problems.append("source_copy_rows")
        else:
            rebuilt = build_legacy_head_payload(
                run_root, index_rows=source_rows,
                ledger_rows=source_entries, marker=source_marker,
                run_root=run_root,
                marker_sha256=sha256_file(source_dir / "cohort.json"),
                source_index_sha256=sha256_file(source_dir / "index.json"),
                source_ledger_sha256=sha256_file(source_dir / "ledger.json"))
            if legacy_head_sha256(rebuilt) != snapshot.get(
                    "legacy_head_sha256"):
                problems.append("legacy_head_not_reproducible")
            if _jsonable(rebuilt) != _jsonable(snapshot.get(
                    "legacy_head_payload")):
                problems.append("legacy_head_values_not_reproducible")
    except (OSError, ValueError, TypeError) as exc:
        problems.append(f"source_copy_unreadable:{type(exc).__name__}")
    journal = scan_launch_journals(run_root.parent)
    if not journal.get("known"):
        problems.append("launch_journal_unsettled")
        problems += [f"launch:{p}" for p in journal.get("problems") or []]
    # Exact remaining budget and attempts from the frozen original rows.
    budgets = {stage: _budget_from_snapshot(snapshot, stage)
               for stage in STAGE_BUDGETS}
    remaining_total = TOTAL_BUDGET_S - float(
        (snapshot.get("cost_s") or {}).get("total", 0.0))
    if abs(remaining_total - (TOTAL_BUDGET_S - MIGRATION_SOURCE_COST_S)) > 1e-9:
        problems.append("migration_budget_snapshot")
    if (spec.get("budgets") or {}) != budgets:
        problems.append("migration_budget_spec")
    for stage in STAGE_BUDGETS:
        if int(budgets[stage]["attempts_used"]) != len(
                (snapshot.get("attempts") or {}).get(stage) or []):
            problems.append(f"migration_attempts:{stage}")
        if budgets[stage]["remaining_s"] <= STAGE_BUDGETS[stage]["reserve_s"]:
            problems.append(f"migration_budget_exhausted:{stage}")
    impl = implementation_identity(root)
    if (spec.get("bindings") or {}).get("implementation", {}).get(
            "files") != impl.get("files"):
        problems.append("migration_implementation_drift")
    if gate_b_path is not None:
        if not Path(gate_b_path).is_file() or sha256_file(
                Path(gate_b_path)) != gate_b_sha256:
            problems.append("migration_gate_b")
    if real_admission_path is not None:
        if not Path(real_admission_path).is_file() or sha256_file(
                Path(real_admission_path)) != real_admission_sha256:
            problems.append("migration_real_admission")
        if isinstance(real_admission, Mapping):
            problems += [f"migration_real_admission:{p}" for p in
                         _admission_problems_for(real_admission)]
    elif real_admission_sha256 is not None:
        problems.append("migration_real_admission_path")
    record = {
        "schema": MIGRATION_ADMISSION_SCHEMA,
        "status": {
            "mode": "pre_authorization_migration_readonly",
            "authorized": False, "registered": False, "run": "not_run",
            "section_10_2_checked": False,
        },
        "passed": not problems,
        "problems": problems,
        "run_root": str(run_root),
        "snapshot": {"path": path_for_record(Path(snapshot_path)),
                     "sha256": snapshot_sha256,
                     "legacy_head_sha256": snapshot.get(
                         "legacy_head_sha256")},
        "spec": {"path": path_for_record(Path(spec_path)),
                 "sha256": spec_sha256},
        "legacy_head_sha256": snapshot.get("legacy_head_sha256"),
        "call_balance_after": snapshot.get("call_balance_after"),
        "budgets": budgets,
        "remaining_total_s": remaining_total,
        "attempts": snapshot.get("attempts"),
        "launch_journal": snapshot.get("launch_journal"),
        "implementation": impl,
        "gate_b": ({"path": path_for_record(Path(gate_b_path)),
                    "sha256": gate_b_sha256}
                   if gate_b_path is not None else None),
        "real_admission": ({"path": path_for_record(
            Path(real_admission_path)), "sha256": real_admission_sha256}
            if real_admission_path is not None else None),
        "read_scope": {
            "checkpoint_payload_read": False,
            "npz_values_read": False,
            "model_constructor_calls": 0,
            "mujoco_compile_calls": 0,
            "forward_calls": 0,
            "vjp_calls": 0,
            "optimizer_steps": 0,
            "cohort_created": False,
            "registered_attempt_created": False,
        },
        "authorization_required": True,
        "notes": [
            "same cohort and fixed history root; no budget or attempt reset",
            "all migration cost is charged to the second-generation R0",
            "public consumption additionally requires the outer return "
            "observation",
        ],
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return record


def write_migration_admission(out_root: Path, package: Mapping[str, Any]
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
    path = directory / "migration_admission.json"
    write_json(path, package)
    sha = sha256_file(path)
    rows.append({"attempt": attempt, "path": path_for_record(path),
                 "sha256": sha, "passed": bool(package.get("passed"))})
    write_json(index_path, {"attempts": rows})
    return path, sha


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


def _admission_problems_for(package: Mapping[str, Any] | None) -> list[str]:
    from ireset.eval.w1_real_training_update_provider import (
        REAL_ADMISSION_SCHEMA,
        real_admission_package_problems,
    )
    if isinstance(package, Mapping) \
            and package.get("schema") == REAL_ADMISSION_SCHEMA:
        return real_admission_package_problems(package)
    return admission_package_problems(package)


def _report_array_binding_problems(report_path: Path | None,
                                   files: Mapping[str, Path]) -> list[str]:
    if report_path is None or not Path(report_path).is_file():
        return ["report_missing"]
    try:
        report = read_json(Path(report_path))
    except (OSError, ValueError, TypeError):
        return ["report_unreadable"]
    problems: list[str] = []
    for model, path in files.items():
        record = ((report.get("models") or {}).get(model) or {})
        expected = record.get("arrays_sha256")
        actual = sha256_file(path) if Path(path).is_file() else None
        if expected is None or actual != expected:
            problems.append(f"array_binding:{model}")
    return problems


def _selected_stage_problems(
        deltas: Mapping[str, Mapping[str, Mapping[str, Any]]],
        selected: Mapping[str, str], mode: str) -> list[str]:
    problems: list[str] = []
    for stage in STAGE_BUDGETS:
        rec = (deltas.get(stage) or {}).get(selected.get(stage))
        if rec is None:
            problems.append(f"selected_delta_missing:{stage}")
            continue
        nominal = stage_nominal_for_mode(mode, stage)
        for kind in CALL_KINDS:
            if int(rec["counts"][kind]) != nominal[kind]:
                problems.append(
                    f"selected_stage_matrix:{stage}:{kind}:"
                    f"{rec['counts'][kind]}!={nominal[kind]}")
    chain = selected_chain_counts(deltas, selected)
    expected = chain_nominal_for_mode(mode)
    if chain is None:
        problems.append("selected_chain_missing")
    else:
        for kind in CALL_KINDS:
            if int(chain[kind]) != expected[kind]:
                problems.append(
                    f"full_chain_calls:{kind}:{int(chain[kind])}"
                    f"!={expected[kind]}")
    return problems


def _selected_caps_problems(
        deltas: Mapping[str, Mapping[str, Mapping[str, Any]]],
        mode: str) -> list[str]:
    problems: list[str] = []
    cohort = {kind: 0 for kind in CALL_KINDS}
    for stage in STAGE_BUDGETS:
        totals = stage_call_totals(deltas, stage)
        nominal = stage_nominal_for_mode(mode, stage)
        for kind in CALL_KINDS:
            if totals[kind] > 2 * nominal[kind]:
                problems.append(f"stage_cap:{stage}:{kind}")
            cohort[kind] += totals[kind]
        used = len(deltas.get(stage) or {})
        if used > STAGE_BUDGETS[stage]["attempts"]:
            problems.append(f"attempt_cap:{stage}")
    for kind in CALL_KINDS:
        if cohort[kind] > int(CALL_CAPS[kind]):
            problems.append(f"cohort_cap:{kind}")
    return problems


def run_registered_audit(run_root: Path, provider: Any,
                         *, mode: str, tolerance: Mapping[str, Any],
                         authorization_sha: str | None = None,
                         authorization: Mapping[str, Any] | None = None,
                         admission_package: Mapping[str, Any] | None = None,
                         admission_sha: str | None = None,
                         fixed_root: Path = HISTORY_ROOT,
                         project_root: Path = PROJECT_ROOT,
                         r0_started_perf: float | None = None,
                         resume_stage: str | None = None) -> dict:
    """Registered R0 -> P -> I -> F orchestration with resume support.

    A fresh run executes the four stages once.  ``resume_stage`` reuses the
    unique consumable upstream attempts, refuses unknown history and appends
    one new attempt for the named stage (plan section 2.1).  Every stage
    selection is explicit: no latest-attempt or directory-order guessing.
    """
    if r0_started_perf is not None and (
            not _nonneg(r0_started_perf)
            or float(r0_started_perf) > time.perf_counter()):
        raise ValueError("invalid R0 launch clock")
    run_root = Path(run_root).expanduser().resolve()
    if mode not in ("synthetic", "real"):
        raise ValueError(f"unknown mode {mode}")
    if resume_stage is not None and resume_stage not in STAGE_BUDGETS:
        raise ValueError(f"unknown resume stage {resume_stage}")
    contract_problems = tolerance_contract_problems(tolerance,
                                                    recompute=False)
    if contract_problems:
        raise SystemExit(f"tolerance contract refused: {contract_problems}")
    if mode == "real":
        if isinstance(authorization, Mapping) and authorization.get(
                "schema") == AUTHORIZATION_SCHEMA_V2:
            raise SystemExit(
                "authorization v2 is migration-bound; use the dedicated "
                "binding migration entry, not ordinary resume")
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
        pkg_problems = _admission_problems_for(admission_package)
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
        if authorization.get("execution_plan_sha256") != current_impl[
                "files"].get(EXECUTION_PLAN):
            raise SystemExit(
                "authorization does not bind the execution plan")
        if authorization.get("acceptance_sha256") != admission_package.get(
                "acceptance_sha256"):
            raise SystemExit(
                "authorization does not bind the synthetic acceptance")
        if authorization.get("calibration_verification_sha256") != \
                admission_package.get("calibration_verification_sha256"):
            raise SystemExit(
                "authorization does not bind the calibration verification")
        acceptance = admission_package.get("acceptance") or {}
        from ireset.eval.w1_real_training_update_provider import (
            calibration_verification_problems,
            synthetic_acceptance_problems,
        )
        if synthetic_acceptance_problems(acceptance):
            raise SystemExit(
                "admission synthetic acceptance refused: "
                f"{synthetic_acceptance_problems(acceptance)}")
        if calibration_verification_problems(
                admission_package.get("calibration_verification")):
            raise SystemExit(
                "admission calibration verification refused: "
                f"{calibration_verification_problems(admission_package.get('calibration_verification'))}")
    if resume_stage is None:
        if run_root.exists() and any(run_root.iterdir()):
            raise SystemExit(f"fresh run root is not empty: {run_root}")
    else:
        if not (run_root / "cohort.json").is_file():
            raise SystemExit("resume requires an existing registered cohort")
    run_root.mkdir(parents=True, exist_ok=True)

    selected: dict[str, str] = {}
    if resume_stage is not None:
        marker = read_json(run_root / "cohort.json")
        if marker.get("cohort_id") != COHORT_ID \
                or marker.get("mode") != mode \
                or marker.get("authorization_sha256") != authorization_sha:
            raise SystemExit("resume cohort identity mismatch")
        history = verify_history(run_root)
        if history:
            raise SystemExit(f"resume history unknown: {history}")
        order = list(STAGE_BUDGETS)
        for stage in order[:order.index(resume_stage)]:
            row, problems = stage_state(run_root, stage)
            if problems or row is None:
                raise SystemExit(
                    f"resume upstream {stage} not consumable: {problems}")
            selected[stage] = row["attempt"]
        target_row, target_problems = stage_state(run_root, resume_stage)
        if target_row is not None and not target_problems:
            raise SystemExit(f"{resume_stage} is already consumable")
        for stage in order[order.index(resume_stage) + 1:]:
            if _disk_attempts(run_root, stage):
                raise SystemExit(f"resume has downstream attempts: {stage}")
    if resume_stage is None:
        write_cohort(run_root, mode=mode, authorization_sha=authorization_sha)
    prior = prior_call_counts(run_root, fixed_root=fixed_root, mode=mode)
    if prior is None:
        raise SystemExit("call history unknown: refusing a fresh quota")
    required = remaining_nominal_for_mode(mode, resume_stage)
    insufficient = {kind: {"prior": prior[kind], "required": required[kind],
                           "cap": CALL_CAPS[kind]}
                    for kind in CALL_KINDS
                    if prior[kind] + required[kind] > CALL_CAPS[kind]}
    if insufficient:
        raise SystemExit(
            f"cumulative call budget insufficient: {insufficient}")
    counter = CallCounter(CALL_CAPS, prior)
    binder = getattr(provider, "bind_counter", None)
    if callable(binder):
        binder(counter)
    fixture_seconds = None
    if mode == "real":
        try:
            provider.load_acceptance()
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"real provider acceptance refused: {exc}")
        if provider.roster is None:
            raise SystemExit("real provider has no static roster")
    if mode == "synthetic":
        fixture_dir = run_root / "fixture"
        if (fixture_dir / "fixture.json").is_file():
            provider.fixture_dir = fixture_dir
            provider.load_fixture()
        else:
            t_fx = _now()
            provider.prepare_fixture(fixture_dir)
            fixture_seconds = _now() - t_fx
            provider.load_fixture()
    roster = provider.roster
    scope = _stage_scope(provider, roster)
    results: dict[str, Any] = {}
    seals: dict[str, Any] = {}

    def stop(problems: Sequence[str], gaps: Sequence[str],
             failures: Sequence[str] = ()) -> dict:
        deltas = _stage_deltas(run_root)
        stage_totals = ({stage: stage_call_totals(deltas, stage)
                         for stage in STAGE_BUDGETS} if deltas else {})
        return {
            "mode": mode, "conclusion": "audit_incomplete",
            "complete_evidence": False, "covered_contract_passed": False,
            "problems": list(problems), "failures": list(failures),
            "gaps": list(gaps), "stages": results, "seals": seals,
            "selected_attempts": None,
            "selected_chain_counts": None,
            "selected_chain_training_updates": None,
            "stage_call_counts": stage_totals,
            "cohort_call_counts": (cohort_call_totals_from_deltas(deltas)
                                   if deltas else None),
            "fixture_build_s": fixture_seconds,
            "calls": counter.snapshot(),
            "training_updates": int(counter.counts["native_adamw_steps"]
                                    + counter.counts["explicit_reference_updates"]),
            "training_updates_scope": "cohort",
            "zero_counters": {k: 0 for k in ZERO_COUNTERS},
            "coverage": dict(BOUNDARIES),
            "boundaries": dict(BOUNDARIES),
        }

    def stage_provider(stage: str) -> Any:
        factory = getattr(provider, "for_stage", None)
        stage_p = factory(stage) if callable(factory) else provider
        binder_ = getattr(stage_p, "bind_counter", None)
        if callable(binder_):
            binder_(counter)
        return stage_p

    def r0_noise_frozen(selected_attempt: str,
                        row: Mapping[str, Any] | None
                        ) -> tuple[Path | None, str | None]:
        """Sealed path and digest only; values are decoded in the consumer."""
        if mode == "synthetic":
            return None, None
        return frozen_noise_source(run_root, selected_attempt, row)

    # --- R0 -----------------------------------------------------------------
    first_start = (_now() if r0_started_perf is None
                   else float(r0_started_perf))
    stage_order = list(STAGE_BUDGETS)
    start_index = (stage_order.index(resume_stage)
                   if resume_stage is not None else 0)
    if start_index <= 0:
        t0 = first_start
        try:
            budget = budget_for(run_root, "R0", fixed_root=fixed_root,
                                mode=mode)
        except SystemExit as exc:
            return stop([str(exc)], [str(exc)])
        attempt = allocate_attempt(run_root, "R0")
        base_counts = dict(counter.counts)
        pre_register_attempt(run_root, "R0", attempt, t0, scope=scope,
                             budget=budget, call_balance=counter.snapshot())
        reserve_stage_calls(run_root, "R0", attempt,
                            {"counts": stage_nominal_for_mode(mode, "R0")})
        r0_provider = stage_provider("R0")
        r0 = run_r0(run_root, "R0", attempt, t0, budget, r0_provider, counter,
                    roster, mode=mode, tolerance=tolerance,
                    project_root=project_root,
                    admission_package=admission_package,
                    authorization=authorization)
        r0_delta = _stage_delta(counter, base_counts)
        settle_stage_calls(run_root, "R0", attempt, r0_delta)
        results["R0"] = r0
        admission_path = run_root / STAGE_DIRS["R0"] / attempt / "admission.json"
        r0_dir = stage_dir(run_root, "R0") / attempt
        r0_artifacts = [
            r0_dir / "registration.json", admission_path,
            r0_dir / "call_reservations.json",
        ]
        if mode == "real":
            r0_artifacts.append(r0_dir / "noise_schedule.npz")
        seal = finish_attempt(
            run_root, "R0", attempt, t0, budget,
            {"complete": bool(r0["passed"]),
             "report": path_for_record(admission_path),
             "summary": path_for_record(admission_path),
             "registration_sha256": r0["registration_sha256"],
             "artifacts": _artifact_hashes(r0_artifacts)},
            status="done" if r0["passed"] else "incomplete",
            extra={"stage": "R0", "scope": scope,
                   "call_balance": counter.snapshot()})
        seals["R0"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                       "problems": seal.problems, "row": seal.row}
        r0_consumer_started = _now()
        _row, r0_consumer = stage_state(run_root, "R0")
        r0_verification_wall = _now() - r0_consumer_started
        seal, r0_cost_problems = _charge_stage_consumption(
            run_root, "R0", attempt, t0, budget, seal, seals["R0"],
            scope=scope, counter=counter, actual_cost=_now() - t0,
            verification_wall=r0_verification_wall)
        r0_consumer += r0_cost_problems
        seals["R0"]["consumer_problems"] = r0_consumer
        if not seal.sealed or not r0["passed"] or r0_consumer:
            blockers = r0["problems"] + seal.problems + r0_consumer
            return stop(blockers, r0["gaps"] + seal.problems + r0_consumer)
        r0_row = seal.row
        registration_sha = r0["registration_sha256"]
        selected["R0"] = attempt
    else:
        row, problems = stage_state(run_root, "R0", selected["R0"])
        if problems or row is None:
            return stop([f"selected_r0:{p}" for p in problems],
                        [f"selected_r0:{p}" for p in problems])
        r0_row = row
        registration_sha = row.get("registration_sha256")
        results["R0"] = {"resumed": True, "attempt": selected["R0"]}
        seals["R0"] = {"sealed": True, "terminal": True,
                       "resumed": True, "row": row}
    r0_noise_path, r0_noise_sha = r0_noise_frozen(selected["R0"], r0_row)

    # --- P ------------------------------------------------------------------
    if start_index <= 1:
        t0 = first_start if resume_stage == "P" else _now()
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
                            {"counts": stage_nominal_for_mode(mode, "P")})
        stage_deadline = (t0 + float(budget["remaining_s"])
                          - float(budget["reserve_s"]))
        p_provider = stage_provider("P")
        try:
            p = run_p_stage(
                run_root, "P", attempt, t0, budget, p_provider, counter,
                roster, guard=_stage_unit_guard(
                    run_root, "P", attempt,
                    registration_sha=registration_sha,
                    deadline=stage_deadline),
                registered=True,
                r0_noise=(provider.noise if mode == "synthetic" else None),
                r0_noise_path=r0_noise_path,
                r0_noise_sha256=r0_noise_sha,
                write_input_admission=(mode == "real"),
                r0_registration_sha=registration_sha)
        except (TimeoutError, UpdateCallBudgetExceeded) as exc:
            p = {"passed": False, "failures": [type(exc).__name__],
                 "gaps": [f"p_stage_aborted:{exc}"], "rows": [], "paths": {},
                 "report": None, "arrays_sha256": {},
                 "complete_evidence": False, "numeric_passed": False}
        except Exception as exc:  # noqa: BLE001 -- a broken stage is evidence
            p = {"passed": False, "failures": [type(exc).__name__],
                 "gaps": [f"p_stage_exception:{type(exc).__name__}:{exc}"],
                 "rows": [], "paths": {}, "report": None,
                 "arrays_sha256": {}, "complete_evidence": False,
                 "numeric_passed": False}
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
        p_base = stage_dir(run_root, "P") / attempt
        p_artifacts = [Path(path)
                       for path in (p.get("paths") or {}).values()]
        p_artifacts += [p_report, p_base / "p_rows.json",
                        p_base / "call_reservations.json"]
        if mode == "real":
            p_artifacts.append(p_base / "input_admission.json")
        seal = finish_attempt(
            run_root, "P", attempt, t0, budget,
            {"complete": p_complete,
             "report": path_for_record(p_report) if p_report else None,
             "summary": path_for_record(p_report) if p_report else None,
             "registration_sha256": registration_sha,
             "artifacts": _artifact_hashes(p_artifacts)},
            status="done" if p_complete else "incomplete",
            extra={"stage": "P", "scope": scope,
                   "numeric_passed": bool(p.get("numeric_passed")),
                   "call_balance": counter.snapshot()})
        seals["P"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                      "problems": seal.problems, "row": seal.row}
        p_consumer_started = _now()
        p_row, p_consumer = stage_state(run_root, "P", attempt)
        p_verification_wall = _now() - p_consumer_started
        seal, p_cost_problems = _charge_stage_consumption(
            run_root, "P", attempt, t0, budget, seal, seals["P"],
            scope=scope, counter=counter, actual_cost=_now() - t0,
            verification_wall=p_verification_wall)
        p_consumer += p_cost_problems
        seals["P"]["consumer_problems"] = p_consumer
        if (not seal.terminal or not seal.sealed or p_consumer
                or not p.get("paths")):
            blockers = seal.problems + p_consumer
            if not p.get("paths"):
                blockers.append("p_artifacts_missing")
            return stop(blockers, p["gaps"] + blockers,
                        failures=p["failures"])
        selected["P"] = attempt
        p_dir = stage_dir(run_root, "P") / attempt
    else:
        row, problems = stage_state(run_root, "P", selected["P"])
        if problems or row is None:
            return stop([f"selected_p:{p}" for p in problems],
                        [f"selected_p:{p}" for p in problems])
        results["P"] = {"resumed": True, "attempt": selected["P"]}
        seals["P"] = {"sealed": True, "terminal": True,
                      "resumed": True, "row": row}
        p_dir = stage_dir(run_root, "P") / selected["P"]
    p_input_admission_path = p_dir / "input_admission.json"
    p_input_admission = None
    if mode == "real":
        if not p_input_admission_path.is_file():
            return stop(["p_input_admission_missing"],
                        ["p_input_admission_missing"])
        p_input_admission = read_json(p_input_admission_path)
        if p_input_admission.get("passed") is not True:
            return stop(["p_input_admission_not_passed"],
                        ["p_input_admission_not_passed"])

    # --- I ------------------------------------------------------------------
    if start_index <= 2:
        t0 = first_start if resume_stage == "I" else _now()
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
                            {"counts": stage_nominal_for_mode(mode, "I")})
        stage_deadline = (t0 + float(budget["remaining_s"])
                          - float(budget["reserve_s"]))
        i_provider = stage_provider("I")
        try:
            i = run_i_stage(
                run_root, "I", attempt, t0, budget, i_provider, counter,
                roster, p_dir, tolerance=tolerance,
                guard=_stage_unit_guard(
                    run_root, "I", attempt,
                    registration_sha=registration_sha,
                    deadline=stage_deadline),
                registered=True,
                r0_noise=(provider.noise if mode == "synthetic" else None),
                r0_noise_path=r0_noise_path,
                r0_noise_sha256=r0_noise_sha,
                parent_input_admission=p_input_admission,
                r0_registration_sha=registration_sha)
        except (TimeoutError, UpdateCallBudgetExceeded) as exc:
            i = {"passed": False, "failures": [type(exc).__name__],
                 "gaps": [f"i_stage_aborted:{exc}"], "rows": [], "paths": {},
                 "report": None, "arrays_sha256": {},
                 "complete_evidence": False, "numeric_passed": False}
        except Exception as exc:  # noqa: BLE001 -- a broken stage is evidence
            i = {"passed": False, "failures": [type(exc).__name__],
                 "gaps": [f"i_stage_exception:{type(exc).__name__}:{exc}"],
                 "rows": [], "paths": {}, "report": None,
                 "arrays_sha256": {}, "complete_evidence": False,
                 "numeric_passed": False}
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
        i_base = stage_dir(run_root, "I") / attempt
        i_artifacts = [Path(path)
                       for path in (i.get("paths") or {}).values()]
        i_artifacts += [i_report, i_base / "i_rows.json",
                        i_base / "call_reservations.json"]
        seal = finish_attempt(
            run_root, "I", attempt, t0, budget,
            {"complete": i_complete,
             "report": path_for_record(i_report) if i_report else None,
             "summary": path_for_record(i_report) if i_report else None,
             "registration_sha256": registration_sha,
             "artifacts": _artifact_hashes(i_artifacts)},
            status="done" if i_complete else "incomplete",
            extra={"stage": "I", "scope": scope,
                   "numeric_passed": bool(i.get("numeric_passed")),
                   "call_balance": counter.snapshot()})
        seals["I"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                      "problems": seal.problems, "row": seal.row}
        i_consumer_started = _now()
        i_row, i_consumer = stage_state(run_root, "I", attempt)
        i_verification_wall = _now() - i_consumer_started
        seal, i_cost_problems = _charge_stage_consumption(
            run_root, "I", attempt, t0, budget, seal, seals["I"],
            scope=scope, counter=counter, actual_cost=_now() - t0,
            verification_wall=i_verification_wall)
        i_consumer += i_cost_problems
        seals["I"]["consumer_problems"] = i_consumer
        if (not seal.terminal or not seal.sealed or i_consumer
                or not i.get("paths")):
            blockers = seal.problems + i_consumer
            if not i.get("paths"):
                blockers.append("i_artifacts_missing")
            return stop(blockers, i["gaps"] + blockers,
                        failures=i["failures"])
        selected["I"] = attempt
        i_dir = stage_dir(run_root, "I") / attempt
    else:
        row, problems = stage_state(run_root, "I", selected["I"])
        if problems or row is None:
            return stop([f"selected_i:{p}" for p in problems],
                        [f"selected_i:{p}" for p in problems])
        results["I"] = {"resumed": True, "attempt": selected["I"]}
        seals["I"] = {"sealed": True, "terminal": True,
                      "resumed": True, "row": row}
        i_dir = stage_dir(run_root, "I") / selected["I"]

    # --- F ------------------------------------------------------------------
    t0 = first_start if resume_stage == "F" else _now()
    try:
        budget = budget_for(run_root, "F", fixed_root=fixed_root, mode=mode)
    except SystemExit as exc:
        return stop([str(exc)], [str(exc)])
    attempt = allocate_attempt(run_root, "F")
    candidates = {"R0": selected["R0"], "P": selected["P"],
                  "I": selected["I"], "F": attempt}
    pre_register_attempt(run_root, "F", attempt, t0, scope=scope,
                         budget=budget, registration_sha=registration_sha,
                         call_balance=counter.snapshot())
    reserve_stage_calls(run_root, "F", attempt,
                        {"counts": stage_nominal_for_mode(mode, "F")})
    f_provider = stage_provider("F")
    try:
        f = run_f_stage(
            run_root, "F", attempt, t0, budget, f_provider, counter,
            roster, p_dir, i_dir, tolerance=tolerance,
            project_root=project_root, candidate_attempts=candidates,
            r0_noise_path=r0_noise_path, r0_noise_sha256=r0_noise_sha,
            p_input_admission_path=(p_input_admission_path
                                    if mode == "real" else None),
            p_input_admission_sha256=(sha256_file(p_input_admission_path)
                                      if mode == "real" else None))
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
    pre_seal_veto: list[str] = []
    final_balance = counter.snapshot()
    final_counts = _call_counts_from_balance(final_balance)
    if settled and final_counts is not None:
        current = _attempt_row(run_root, "F", attempt)
        updates = {**current, "call_balance": final_balance,
                   "call_balance_after": final_balance}
        try:
            upsert_index(run_root, "F", attempt, updates)
            upsert_cost(run_root, "F", attempt,
                        float(current.get("measured_wall_s", 0.0)),
                        {"complete": False, "scope": scope,
                         "call_balance": final_balance,
                         "call_balance_before": current.get(
                             "call_balance_before"),
                         "call_balance_after": final_balance},
                        "running")
        except (OSError, ValueError, TypeError):
            pre_seal_veto.append("f_settlement_persist")
    if not settled:
        pre_seal_veto.append("f_call_settlement_unknown")
    if settled and not pre_seal_veto:
        history = verify_history(run_root, allow_current=("F", attempt))
        pre_seal_veto += history
        deltas = _stage_deltas(run_root)
        if deltas is None:
            pre_seal_veto.append("f_call_delta_unknown")
        else:
            current_rec = deltas.get("F", {}).get(attempt)
            if current_rec is None:
                pre_seal_veto.append("f_current_delta_missing")
            else:
                for kind in CALL_KINDS:
                    if int(current_rec["counts"][kind]) != int(
                            (snapshot or {}).get(kind, -1)):
                        pre_seal_veto.append(
                            f"f_settlement_snapshot:{kind}")
            upstream = {"R0": selected["R0"], "P": selected["P"],
                        "I": selected["I"], "F": attempt}
            problems = _selected_stage_problems(deltas, upstream, mode)
            pre_seal_veto += problems
            pre_seal_veto += _selected_caps_problems(deltas, mode)
            worker = f.get("worker_result") or {}
            if (worker.get("provisional_chain_counts")
                    != selected_chain_counts(deltas, upstream)):
                pre_seal_veto.append("f_provisional_chain")
            if (worker.get("precheck") or {}).get("R0", {}).get(
                    "attempt") != selected["R0"]:
                pre_seal_veto.append("f_precheck_binding")
            for stage, attempt_name in (("R0", selected["R0"]),
                                        ("P", selected["P"]),
                                        ("I", selected["I"])):
                row, state = stage_state(run_root, stage, attempt_name,
                                         allow_current=("F", attempt))
                if state:
                    pre_seal_veto += [f"f_parent:{stage}:{p}" for p in state]
                    continue
                if row.get("report") and row.get("report_sha256") and \
                        sha256_file(PROJECT_ROOT / row["report"]) != \
                        row["report_sha256"]:
                    pre_seal_veto.append(f"f_parent_report_sha:{stage}")
            p_report_path = stage_dir(run_root, "P") / selected["P"] / \
                "p_report.json"
            i_report_path = stage_dir(run_root, "I") / selected["I"] / \
                "i_report.json"
            pre_seal_veto += [f"f_array:{p}" for p in
                              _report_array_binding_problems(
                                  p_report_path,
                                  {m: stage_dir(run_root, "P") /
                                   selected["P"] / f"p_{m}.npz"
                                   for m in MODELS})]
            pre_seal_veto += [f"f_array:{p}" for p in
                              _report_array_binding_problems(
                                  i_report_path,
                                  {m: stage_dir(run_root, "I") /
                                   selected["I"] / f"i_{m}.npz"
                                   for m in MODELS})]
            if mode == "real":
                try:
                    p_report_data = read_json(p_report_path)
                    admission_record = p_report_data.get(
                        "input_admission") or {}
                except (OSError, ValueError, TypeError):
                    admission_record = {}
                expected_sha = (sha256_file(p_input_admission_path)
                                if p_input_admission_path.is_file() else None)
                if (not admission_record
                        or admission_record.get("sha256") != expected_sha
                        or Path(admission_record.get("path", "")).resolve()
                        != p_input_admission_path.resolve()):
                    pre_seal_veto.append("f_p_input_admission_binding")
    if not settled:
        f["complete_evidence"] = False
        f["covered_contract_passed"] = False
        f["conclusion"] = "audit_incomplete"
    f_report = Path(f["report"]) if f.get("report") else None
    f_base = stage_dir(run_root, "F") / attempt
    f_artifacts = _artifact_hashes([
        f_report, f_base / "worker_spec.json",
        f_base / "worker" / "worker_result.json",
        f_base / "call_reservations.json",
    ])
    if pre_seal_veto:
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
             "artifacts": dict(f_artifacts)},
            status="done" if f["complete_evidence"] else "incomplete",
            extra={"stage": "F", "scope": scope,
                   "call_balance": counter.snapshot()})
        seals["F"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                      "problems": seal.problems, "row": seal.row}

    # --- post-seal final selection -----------------------------------------
    selected_all: dict[str, str] = {}
    selection_problems: list[str] = []
    selected_chain = None
    stage_totals: dict = {}
    cohort_totals = None
    if seal.terminal and seal.sealed and f.get("complete_evidence"):
        for stage, attempt_name in candidates.items():
            row, state = stage_state(run_root, stage, attempt_name)
            if state or row is None:
                selection_problems += [f"selection:{stage}:{p}" for p in state]
            else:
                selected_all[stage] = attempt_name
        deltas = _stage_deltas(run_root)
        if deltas is None:
            selection_problems.append("selection_delta_unknown")
        else:
            if set(selected_all) == set(STAGE_BUDGETS):
                selected_chain = selected_chain_counts(deltas, selected_all)
                selection_problems += _selected_stage_problems(
                    deltas, selected_all, mode)
            selection_problems += _selected_caps_problems(deltas, mode)
            stage_totals = {stage: stage_call_totals(deltas, stage)
                            for stage in STAGE_BUDGETS}
            cohort_totals = cohort_call_totals_from_deltas(deltas)
    elif seal.terminal and seal.sealed and not pre_seal_veto:
        selection_problems.append("selection_evidence_incomplete")
    if pre_seal_veto:
        selection_problems.append("pre_seal_veto")
    if selection_problems:
        revoke = list(selection_problems)
        if _report_claims_complete(f_report):
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
    else:
        selection_record = {
            "schema": "w1_real_training_update_terminal_selection_v1",
            "selected_attempts": selected_all,
            "selected_chain_counts": selected_chain,
            "selected_chain_training_updates": int(
                (selected_chain or {}).get("native_adamw_steps", 0)
                + (selected_chain or {}).get("explicit_reference_updates", 0)),
            "stage_call_counts": stage_totals,
            "cohort_call_counts": cohort_totals,
            "cohort_training_updates": int(
                (cohort_totals or {}).get("native_adamw_steps", 0)
                + (cohort_totals or {}).get("explicit_reference_updates", 0)),
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if f_report is not None and f_report.is_file():
            try:
                data = read_json(f_report)
                data.update(
                    selected_attempts=selected_all,
                    selected_chain_counts=selected_chain,
                    selected_chain_training_updates=selection_record[
                        "selected_chain_training_updates"],
                    stage_call_counts=stage_totals,
                    cohort_call_counts=cohort_totals,
                    cohort_training_updates=selection_record[
                        "cohort_training_updates"],
                    training_updates_scope="cohort",
                    training_updates=selection_record[
                        "cohort_training_updates"])
                write_json(f_report, data)
            except (OSError, ValueError, TypeError):
                selection_problems.append("selection_report_rewrite")
        f_artifacts = _artifact_hashes([
            f_report, f_base / "worker_spec.json",
            f_base / "worker" / "worker_result.json",
            f_base / "call_reservations.json",
        ])
        previous_cost = float(
            seals["F"].get("row", {}).get("measured_wall_s", 0.0) or 0.0)
        reseal = finish_attempt(
            run_root, "F", attempt, t0, budget,
            {"complete": True,
             "report": path_for_record(f_report) if f_report else None,
             "summary": path_for_record(f_report) if f_report else None,
             "registration_sha256": registration_sha,
             "artifacts": dict(f_artifacts),
             "selection": selection_record},
            status="done",
            extra={"stage": "F", "scope": scope,
                   "call_balance": counter.snapshot(),
                   "selection": selection_record},
            cost_floor=previous_cost)
        seals["F"] = {"sealed": reseal.sealed, "terminal": reseal.terminal,
                      "problems": reseal.problems, "row": reseal.row,
                      "resealed_for_selection": True}
        seal = reseal
        if selection_problems or not reseal.sealed:
            selection_problems += list(reseal.problems)
            if _report_claims_complete(f_report):
                _amend_final_report(f_report, selection_problems)
                wrow, wprobs, proven = withdraw_attempt_claim(
                    run_root, "F", attempt, f_report,
                    reseal.row, selection_problems,
                    reason="post_seal_veto", t0=t0, budget=budget,
                    call_balance=counter.snapshot())
                seals["F"]["post_seal_withdrawal"] = {
                    "row": wrow, "problems": wprobs, "proven": bool(proven)}
                seal = SealResult(False, wrow, list(seal.problems) + wprobs,
                                  bool(proven))
                seals["F"].update({"sealed": False,
                                   "terminal": bool(proven),
                                   "problems": seal.problems, "row": wrow})
    consumer_started = _now()
    consumer_row, consumer_problems = stage_state(run_root, "F", attempt)
    prerequisite = {}
    prerequisite_problems = []
    for stage_name in ("P", "I"):
        _row, state_problems = stage_state(run_root, stage_name,
                                           selected[stage_name])
        prerequisite[stage_name] = state_problems
        prerequisite_problems += [
            f"prerequisite_not_consumable:{stage_name}:{p}"
            for p in state_problems if p != "attempt_not_complete"]
    verification_wall = _now() - consumer_started
    seals["F"]["verification_wall_s"] = verification_wall
    seals["F"]["consumer_problems"] = consumer_problems
    seals["F"]["prerequisite_consumer_problems"] = prerequisite
    seal, cost_revoke = _charge_final_verification(
        run_root, "F", attempt, t0, budget, seal, seals["F"],
        scope=scope, counter=counter, actual_cost=_now() - t0)
    revoke = (list(seal.problems)
              + [p for p in consumer_problems if p != "attempt_not_complete"]
              + prerequisite_problems + selection_problems)
    if cost_revoke is not None:
        revoke.append(cost_revoke)
    if not settled:
        revoke.append("f_call_settlement_unknown")
    if revoke and _report_claims_complete(f_report):
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
        selection_problems.append("f_withdraw_unproven")
    if f_report is not None and f_report.is_file():
        try:
            f["summary"] = read_json(f_report)
        except (OSError, ValueError, TypeError):
            revoke.append("f_summary_unreadable")
    conclusion = f["conclusion"]
    complete = bool(f["complete_evidence"])
    covered = bool(f["covered_contract_passed"])
    if revoke or selection_problems or not seal.terminal or (
            not seal.sealed and not complete):
        conclusion = "audit_incomplete"
        complete = False
        covered = False
    disk_problems = []
    if isinstance(f.get("summary"), Mapping):
        disk_problems = list(f["summary"].get("verification_problems") or [])
    selection_confirmed = (not selection_problems and complete
                           and seal.sealed and seal.terminal)
    deltas = _stage_deltas(run_root)
    stage_totals_final = ({stage: stage_call_totals(deltas, stage)
                           for stage in STAGE_BUDGETS} if deltas else {})
    cohort_final = cohort_call_totals_from_deltas(deltas) if deltas else None
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
        "selected_attempts": selected_all if selection_confirmed else None,
        "selected_chain_counts": (selected_chain
                                  if selection_confirmed else None),
        "selected_chain_training_updates": (
            int((selected_chain or {}).get("native_adamw_steps", 0)
                + (selected_chain or {}).get("explicit_reference_updates", 0))
            if selection_confirmed else None),
        "stage_call_counts": stage_totals_final,
        "cohort_call_counts": cohort_final,
        "fixture_build_s": fixture_seconds,
        "calls": counter.snapshot(),
        "training_updates": int(counter.counts["native_adamw_steps"]
                                + counter.counts["explicit_reference_updates"]),
        "training_updates_scope": "cohort",
        "zero_counters": {k: 0 for k in ZERO_COUNTERS},
        "runtime": getattr(provider, "runtime", None),
        "coverage": dict(BOUNDARIES),
        "boundaries": dict(BOUNDARIES),
    }


def _migration_execution(binding: BindingContext, *,
                         invocation_id: str, claim_path: Path,
                         claim_sha256: str, stage: str) -> dict:
    plan = next((p for p in MIGRATION_STAGE_PLAN if p["stage"] == stage),
                 None)
    if plan is None:
        raise ValueError(f"unknown migration stage {stage}")
    return {
        "generation": binding.generation,
        "binding_spec_sha256": binding.spec_sha256,
        "execution_seq": plan["execution_seq"],
        "invocation_id": invocation_id,
        "legacy_head_sha256": binding.legacy_head_sha256,
        "claim_path": path_for_record(Path(claim_path)),
        "claim_sha256": claim_sha256,
        "stage": stage,
    }


def _migration_stage_seal(
        root: Path, stage: str, attempt: str, t0: float,
        budget: Mapping[str, Any], binding: BindingContext,
        execution: Mapping[str, Any], row: Mapping[str, Any], *,
        complete: bool, report: Path | None,
        artifacts: Sequence[Path], counter: CallCounter,
        scope: Mapping[str, Any], numeric_passed: bool | None = None,
        final: bool = False, actual_cost: float | None = None,
        extra_row: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None
        ) -> tuple[SealResult, CommitResult, Path | None, str | None,
                   list[str]]:
    """finish -> internal prepared verification -> stable cost -> commit."""
    artifact_map = _artifact_hashes(artifacts)
    status = "done" if complete else "incomplete"
    seal = finish_attempt(
        root, stage, attempt, t0, budget,
        {"complete": bool(complete),
         "report": path_for_record(report) if report else None,
         "summary": path_for_record(report) if report else None,
         "registration_sha256": execution.get("registration_sha256"),
         "artifacts": dict(artifact_map),
         **dict(extra_row or {})},
        status=status,
        extra={"stage": stage, "scope": dict(scope),
               "call_balance": counter.snapshot(),
               **dict(extra or {}),
               **({"numeric_passed": bool(numeric_passed)}
                  if numeric_passed is not None else {})})
    if not seal.terminal:
        return seal, CommitResult(False, None, None, None, None,
                                  list(seal.problems)), None, None, []
    prepared = _verify_prepared_attempt(
        root, stage, attempt, binding=binding,
        execution={**execution, "stage_t0_perf_counter": float(t0)})
    problems = list(prepared.problems)
    if complete and problems:
        return seal, CommitResult(False, None, None, None, None,
                                  problems), None, None, problems
    if actual_cost is None:
        actual_cost = max(0.0, time.perf_counter() - float(t0))
    if final:
        seal, cost_revoke = _charge_final_verification(
            root, stage, attempt, t0, budget, seal, {"row": seal.row},
            scope=scope, counter=counter, actual_cost=actual_cost)
        if cost_revoke:
            problems.append(cost_revoke)
    else:
        seal, cost_problems = _charge_stage_consumption(
            root, stage, attempt, t0, budget, seal, {"row": seal.row},
            scope=scope, counter=counter, actual_cost=actual_cost,
            verification_wall=0.0)
        problems += list(cost_problems or [])
    if not seal.sealed or seal.row.get("complete") is not True:
        return seal, CommitResult(False, None, None, None, None,
                                  problems or list(seal.problems)), None, None, \
            problems
    commit = commit_attempt(
        root, stage, attempt, t0=t0, budget=budget, binding=binding,
        execution=execution, cost_floor=float(
            seal.row.get("measured_wall_s", 0.0) or 0.0),
        candidate_problems=[], prepared=prepared)
    problems += commit.problems
    if not commit.committed:
        return seal, commit, None, None, problems
    closed, closed_sha, closed_problems = write_stage_closed_candidate(
        root, stage=stage, attempt=attempt, binding=binding,
        execution=execution, commit=commit, t0=t0,
        remaining_s=float(budget["remaining_s"]),
        committed_measured_wall_s=float(
            seal.row.get("measured_wall_s", 0.0) or 0.0))
    problems += closed_problems
    return seal, commit, closed, closed_sha, problems


def run_binding_migration_audit(
        run_root: Path, provider: Any, *, mode: str,
        tolerance: Mapping[str, Any], authorization_sha: str,
        authorization: Mapping[str, Any],
        admission_package: Mapping[str, Any] | None = None,
        admission_sha: str | None = None,
        binding_spec_path: Path | None = None,
        invocation_id: str | None = None,
        fixed_root: Path = HISTORY_ROOT,
        project_root: Path = PROJECT_ROOT,
        r0_started_perf: float | None = None) -> dict:
    """One registered second-generation R0->P->I->F migration entry.

    The function never creates the outer return observation.  It returns the
    exact commits/candidates/write-after facts that an outer caller must bind
    after the process actually returns.  Synthetic tests may call
    ``record_return_observation`` explicitly; the real CLI does so from its
    launcher.
    """
    run_root = Path(run_root).expanduser().resolve()
    if mode not in ("synthetic", "real"):
        raise ValueError("mode must be synthetic or real")
    if mode == "real":
        runtime = configure_production_runtime()
        if runtime.get("problems"):
            raise SystemExit(f"production runtime cannot be applied: "
                             f"{runtime['problems']}")
    if authorization.get("schema") != AUTHORIZATION_SCHEMA_V2:
        raise SystemExit("migration execution requires authorization v2")
    auth_problems = authorization_problems(authorization)
    if auth_problems:
        raise SystemExit(f"authorization refused: {auth_problems}")
    migration_auth = authorization.get("migration") or {}
    if migration_auth.get("run_root_realpath") != str(run_root):
        raise SystemExit("authorization run root does not match")
    if binding_spec_path is None:
        raise SystemExit("migration execution requires --binding-spec")
    binding_spec_path = Path(binding_spec_path).expanduser().resolve()
    binding, binding_problems = load_binding(
        run_root, generation=str(migration_auth.get("target_generation")))
    if binding is None:
        raise SystemExit(f"binding refused: {binding_problems}")
    if sha256_file(binding_spec_path) != binding.spec_sha256 or \
            (binding.binding_dir / "binding_spec.json").read_bytes() != \
            binding_spec_path.read_bytes():
        raise SystemExit("binding spec path is not the staged byte copy")
    expected_auth = {
        "run_root_realpath": str(run_root),
        "source_marker_sha256": binding.snapshot.get("marker_sha256"),
        "source_authorization_sha256": binding.snapshot.get(
            "authorization_sha256"),
        "snapshot_sha256": binding.snapshot_sha256,
        "spec_sha256": binding.spec_sha256,
        "legacy_head_sha256": binding.legacy_head_sha256,
        "target_generation": binding.generation,
        "stage_plan": list(MIGRATION_STAGE_PLAN),
    }
    for key, want in expected_auth.items():
        if _jsonable(migration_auth.get(key)) != _jsonable(want):
            raise SystemExit(f"authorization migration binding: {key}")
    impl_now = implementation_identity(project_root)
    plan_sha = sha256_file(Path(project_root) / MIGRATION_PLAN)
    if migration_auth.get("implementation_sha256") != canonical_sha256(
            impl_now):
        raise SystemExit("authorization implementation identity drifted")
    if migration_auth.get("execution_plan_sha256") != plan_sha or \
            migration_auth.get("plan_sha256") != plan_sha:
        raise SystemExit("authorization migration plan drifted")
    if migration_auth.get("runtime_sha256") != canonical_sha256(
            runtime_identity_facts()):
        raise SystemExit("authorization runtime identity drifted")
    if migration_auth.get("prereg_sha256") != sha256_file(
            Path(project_root) / PREREG_DRAFT):
        raise SystemExit("authorization prereg drifted")
    spec_bindings = binding.spec.get("bindings") or {}
    if (spec_bindings.get("plan") or {}).get("sha256") != plan_sha:
        raise SystemExit("binding spec plan drifted")
    if migration_auth.get("gate_b_sha256") != \
            (spec_bindings.get("gate_b") or {}).get("sha256"):
        raise SystemExit("authorization gate B binding")
    if mode == "real":
        if admission_package is None or admission_sha is None:
            raise SystemExit("real migration requires a real admission package")
        pkg_problems = _admission_problems_for(admission_package)
        if pkg_problems:
            raise SystemExit(f"admission package refused: {pkg_problems}")
        if migration_auth.get("real_admission_sha256") != admission_sha:
            raise SystemExit("authorization admission binding")
        if authorization.get("admission_package_sha256") != admission_sha:
            raise SystemExit("authorization admission package binding")
        if authorization.get("implementation_files") != \
                implementation_identity(project_root).get("files"):
            raise SystemExit("authorization implementation files drifted")
    if invocation_id is None:
        invocation_id = migration_auth.get("invocation_id")
    if not isinstance(invocation_id, str) or not invocation_id:
        raise SystemExit("migration invocation id missing")
    claim_path = _launch_invocation_dir(run_root, invocation_id) / "claim.json"
    if not claim_path.is_file():
        raise SystemExit("launch claim missing; register the one-shot invocation")
    claim = strict_read_json(claim_path)
    claim_sha = sha256_file(claim_path)
    if claim.get("authorization_sha256") != authorization_sha:
        raise SystemExit("claim authorization differs")
    if claim.get("spec_sha256") != binding.spec_sha256 or \
            claim.get("legacy_head_sha256") != binding.legacy_head_sha256:
        raise SystemExit("claim binding differs")
    if not _nonneg(claim.get("t0_perf_counter")) or \
            float(claim["t0_perf_counter"]) > time.perf_counter():
        raise SystemExit("launch claim clock is invalid")
    journal = scan_launch_journals(
        Path(fixed_root) if mode == "real" else run_root.parent,
        allow_invocation=invocation_id)
    if not journal.get("known"):
        raise SystemExit(f"launch journal not settleable: "
                         f"{journal.get('problems')}")
    before_records, _binding, before_problems = _execution_records(
        run_root, binding=binding)
    if before_records is None:
        raise SystemExit(f"migration history refused: {before_problems}")
    if any(r.get("v2") for r in before_records):
        raise SystemExit("migration has already started")
    order = list(STAGE_BUDGETS)
    results: dict[str, Any] = {}
    commits: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    selected: dict[str, str] = {}
    stage_totals: dict[str, Any] = {}
    stage_timings: dict[str, dict[str, float]] = {}
    cohort_totals = None
    selected_chain = None
    scope = _stage_scope(provider, provider.roster)
    initial_balance = binding.snapshot.get("call_balance_after")
    if not isinstance(initial_balance, Mapping) or set(initial_balance) != \
            set(CALL_KINDS) or not all(
                _json_int(initial_balance[kind]) for kind in CALL_KINDS):
        raise SystemExit("snapshot call balance is invalid")
    counter = CallCounter(CALL_CAPS, initial_balance)
    binder = getattr(provider, "bind_counter", None)
    if callable(binder):
        binder(counter)
    if mode == "real":
        provider.load_acceptance()
    elif provider.fixture_dir is not None:
        provider.load_fixture()
    if r0_started_perf is None:
        r0_started_perf = float(claim.get("t0_perf_counter"))
    t_r0 = float(r0_started_perf)
    # One clock cursor for the whole chain: the next stage starts where the
    # previous one was closed, so handoff work (activation, receipts, budget
    # history scan) is billed to the downstream stage instead of falling into
    # an untimed gap between stage intervals.
    stage_cursor = t_r0
    claim_record = {"claim_path": claim_path, "claim_sha256": claim_sha}

    def failed(problems: Sequence[str], gaps: Sequence[str] = (),
               failures: Sequence[str] = ()) -> dict:
        # Call accounting can be known even when publishing/commit evidence
        # is incomplete, provided every started attempt has a settled
        # reservation and a consistent cumulative balance.
        deltas = _stage_deltas(run_root, require_commits=False)
        known = deltas is not None
        lower_deltas = None
        unsettled: list[str] = []
        if known:
            cohort = cohort_call_totals_from_deltas(deltas)
            lower_deltas = deltas
        else:
            lower_deltas, unsettled_attempt = \
                _settled_deltas_with_unsettled_tail(run_root, binding=binding)
            cohort = None
            if unsettled_attempt:
                unsettled.append(unsettled_attempt)
        chain = (selected_chain_counts(deltas, selected)
                 if known and set(selected) == set(STAGE_BUDGETS) else None)
        exact = int((cohort or {}).get("native_adamw_steps", 0)
                    + (cohort or {}).get("explicit_reference_updates", 0)) \
            if cohort is not None else None
        lower_counts = (cohort_call_totals_from_deltas(lower_deltas)
                        if lower_deltas is not None else
                        {"native_adamw_steps": MIGRATION_SOURCE_UPDATES,
                         "explicit_reference_updates": 0})
        lower = int(lower_counts.get("native_adamw_steps", 0)
                    + lower_counts.get("explicit_reference_updates", 0))
        return {
            "mode": mode, "conclusion": "audit_incomplete",
            "complete_evidence": False, "covered_contract_passed": False,
            "problems": list(problems), "gaps": list(gaps),
            "failures": list(failures), "stages": results,
            "selected_attempts": None, "selected_chain_counts": None,
            "selected_chain_training_updates": None,
            "cohort_call_counts": cohort,
            "cohort_call_counts_known_lower_bound": (
                cohort_call_totals_from_deltas(lower_deltas)
                if lower_deltas is not None else None),
            "cohort_training_updates": exact,
            "training_updates": exact,
            "training_updates_known_lower_bound": lower,
            "training_updates_scope": "cohort",
            "call_accounting_known": known,
            "unsettled_attempts": unsettled,
            "stage_call_counts": (
                {name: stage_call_totals(lower_deltas, name)
                 for name in STAGE_BUDGETS}
                if lower_deltas is not None else stage_totals),
            "commits": commits, "candidates": candidates,
            "invocation_id": invocation_id, "binding_spec_sha256":
                binding.spec_sha256,
        }

    registration_sha = binding.snapshot.get("legacy_head_sha256")
    noise_path = None
    noise_sha = None
    p_dir = None
    p_input_admission = None
    p_input_admission_path = None
    p_input_admission_sha = None
    i_dir = None
    for stage in order:
        attempt = next(p["attempt"] for p in MIGRATION_STAGE_PLAN
                       if p["stage"] == stage)
        try:
            budget = _migration_budget_for(
                run_root, stage, binding=binding,
                allow_invocation=invocation_id)
        except SystemExit as exc:
            return failed([str(exc)], [str(exc)])
        stage_t0 = (t_r0 if stage == "R0" else stage_cursor)
        execution = _migration_execution(
            binding, invocation_id=invocation_id, claim_path=claim_path,
            claim_sha256=claim_sha, stage=stage)
        if stage == "R0":
            base_counts = dict(counter.counts)
            try:
                pre_register_attempt(
                    run_root, stage, attempt, stage_t0, scope=scope,
                    budget=budget, call_balance=counter.snapshot(),
                    execution={**execution,
                               "registration_sha256": registration_sha})
                reserve_stage_calls(
                    run_root, stage, attempt,
                    {"counts": stage_nominal_for_mode(mode, "R0")},
                    execution=execution)
                start_path = _event_dir(
                    run_root, execution["execution_seq"]) / "start.json"
                write_json_exclusive(
                    _launch_invocation_dir(run_root, invocation_id)
                    / "registered.json",
                    {"schema": EVENT_SCHEMA, "kind": "registered",
                     "invocation_id": invocation_id,
                     "execution_seq": execution["execution_seq"],
                     "stage": stage, "attempt": attempt,
                     "start": {"path": path_for_record(start_path),
                               "sha256": sha256_file(start_path)},
                     "created_wall": time.strftime(
                         "%Y-%m-%dT%H:%M:%S%z")})
                r0_provider = (provider.for_stage("R0")
                               if callable(getattr(provider, "for_stage", None))
                               else provider)
                r0 = run_r0(
                    run_root, stage, attempt, stage_t0, budget, r0_provider,
                    counter, provider.roster, mode=mode, tolerance=tolerance,
                    project_root=project_root,
                    admission_package=admission_package,
                    authorization=authorization, execution=execution)
                settle_stage_calls(run_root, stage, attempt,
                                   _stage_delta(counter, base_counts))
                results[stage] = r0
                r0_dir = stage_dir(run_root, stage) / attempt
                artifacts = [r0_dir / "registration.json",
                             r0_dir / "admission.json",
                             r0_dir / "call_reservations.json"]
                if mode == "real":
                    artifacts.append(r0_dir / "noise_schedule.npz")
                seal, commit, closed, closed_sha, problems = \
                    _migration_stage_seal(
                        run_root, stage, attempt, stage_t0, budget, binding,
                        execution, {"complete": bool(r0["passed"])},
                        complete=bool(r0["passed"]),
                        report=Path(r0["admission"]), artifacts=artifacts,
                        counter=counter, scope=scope)
                if problems or not commit.committed:
                    return failed(r0["problems"] + problems,
                                  r0["gaps"] + problems)
                _launch_closed, launch_problems = \
                    write_launch_stage_closed_candidate(
                        run_root, invocation_id=invocation_id, stage=stage,
                        execution_seq=execution["execution_seq"],
                        commit=commit, event_candidate=closed,
                        event_candidate_sha256=closed_sha, t0=stage_t0,
                        remaining_s=float(budget["remaining_s"]),
                        committed_measured_wall_s=float(
                            seal.row.get("measured_wall_s", 0.0) or 0.0))
                if launch_problems:
                    return failed(launch_problems, launch_problems)
                stage_cursor = float(_now())
                stage_timings[stage] = {
                    "t0_perf_counter": float(stage_t0),
                    "end_perf_counter": stage_cursor}
                registration_sha = r0["registration_sha256"]
                noise_path = (Path(r0["noise_schedule"])
                              if r0.get("noise_schedule") else None)
                noise_sha = r0.get("noise_schedule_sha256")
                if mode == "synthetic" and noise_path is None:
                    # Synthetic migration exercises the same binding identity;
                    # the real noise file is not part of the analytic fixture.
                    noise_path = r0_dir / "noise_schedule.npz"
                    np.savez(noise_path, synthetic_fixture=np.array([1], np.int8))
                    noise_sha = sha256_file(noise_path)
                reservation_sha = sha256_file(
                    r0_dir / "call_reservations.json")
                activate_binding(
                    run_root, generation=binding.generation,
                    commit_path=commit.commit_path,
                    commit_sha256=commit.commit_sha256,
                    r0={"registration_sha256": registration_sha,
                        "admission_sha256": sha256_file(
                            r0_dir / "admission.json"),
                        "noise_schedule_sha256": noise_sha,
                        "reservation_sha256": reservation_sha},
                    invocation_id=invocation_id)
                commits[stage] = {"path": path_for_record(commit.commit_path),
                                  "sha256": commit.commit_sha256}
                candidates[stage] = {"path": path_for_record(closed),
                                     "sha256": closed_sha}
                selected[stage] = attempt
            except (TimeoutError, UpdateCallBudgetExceeded, OSError,
                    ValueError, TypeError) as exc:
                return failed([f"migration_r0:{type(exc).__name__}:{exc}"],
                              [f"migration_r0:{type(exc).__name__}"])
            continue
        if stage == "P":
            base_counts = dict(counter.counts)
            stage_deadline = (stage_t0 + float(budget["remaining_s"])
                              - float(budget["reserve_s"]))
            try:
                pre_register_attempt(
                    run_root, stage, attempt, stage_t0, scope=scope,
                    budget=budget, registration_sha=registration_sha,
                    call_balance=counter.snapshot(),
                    execution={**execution,
                               "registration_sha256": registration_sha})
                reserve_stage_calls(
                    run_root, stage, attempt,
                    {"counts": stage_nominal_for_mode(mode, "P")},
                    execution=execution)
                p_provider = (provider.for_stage("P")
                              if callable(getattr(provider, "for_stage", None))
                              else provider)
                p = run_p_stage(
                    run_root, stage, attempt, stage_t0, budget, p_provider,
                    counter, provider.roster, guard=_stage_unit_guard(
                        run_root, stage, attempt,
                        registration_sha=registration_sha,
                        deadline=stage_deadline),
                    registered=True,
                    r0_noise=(provider.noise if mode == "synthetic" else None),
                    r0_noise_path=noise_path, r0_noise_sha256=noise_sha,
                    write_input_admission=(mode == "real"),
                    r0_registration_sha=registration_sha)
                settle_stage_calls(run_root, stage, attempt,
                                   _stage_delta(counter, base_counts))
                results[stage] = p
                p_dir = stage_dir(run_root, stage) / attempt
                p_report = Path(p["report"]) if p.get("report") else None
                artifacts = [Path(path) for path in
                             (p.get("paths") or {}).values()]
                artifacts += [p_report, p_dir / "p_rows.json",
                              p_dir / "call_reservations.json"]
                if mode == "real":
                    artifacts.append(p_dir / "input_admission.json")
                seal, commit, closed, closed_sha, problems = \
                    _migration_stage_seal(
                        run_root, stage, attempt, stage_t0, budget, binding,
                        execution, {"complete": bool(
                            p.get("complete_evidence"))},
                        complete=bool(p.get("complete_evidence")),
                        report=p_report, artifacts=artifacts, counter=counter,
                        scope=scope, numeric_passed=bool(
                            p.get("numeric_passed")))
                if problems or not commit.committed:
                    return failed(list(p.get("gaps") or []) + problems,
                                  list(p.get("gaps") or []) + problems,
                                  list(p.get("failures") or []))
                _launch_closed, launch_problems = \
                    write_launch_stage_closed_candidate(
                        run_root, invocation_id=invocation_id, stage=stage,
                        execution_seq=execution["execution_seq"],
                        commit=commit, event_candidate=closed,
                        event_candidate_sha256=closed_sha, t0=stage_t0,
                        remaining_s=float(budget["remaining_s"]),
                        committed_measured_wall_s=float(
                            seal.row.get("measured_wall_s", 0.0) or 0.0))
                if launch_problems:
                    return failed(launch_problems, launch_problems,
                                  list(p.get("failures") or []))
                stage_cursor = float(_now())
                stage_timings[stage] = {
                    "t0_perf_counter": float(stage_t0),
                    "end_perf_counter": stage_cursor}
                commits[stage] = {"path": path_for_record(commit.commit_path),
                                  "sha256": commit.commit_sha256}
                candidates[stage] = {"path": path_for_record(closed),
                                     "sha256": closed_sha}
                selected[stage] = attempt
                p_input_admission_path = p_dir / "input_admission.json"
                if mode == "real":
                    p_input_admission = strict_read_json(
                        p_input_admission_path)
                    p_input_admission_sha = sha256_file(
                        p_input_admission_path)
            except (TimeoutError, UpdateCallBudgetExceeded, OSError,
                    ValueError, TypeError) as exc:
                return failed([f"migration_p:{type(exc).__name__}:{exc}"],
                              [f"migration_p:{type(exc).__name__}"])
            continue
        if stage == "I":
            base_counts = dict(counter.counts)
            stage_deadline = (stage_t0 + float(budget["remaining_s"])
                              - float(budget["reserve_s"]))
            try:
                pre_register_attempt(
                    run_root, stage, attempt, stage_t0, scope=scope,
                    budget=budget, registration_sha=registration_sha,
                    call_balance=counter.snapshot(),
                    execution={**execution,
                               "registration_sha256": registration_sha})
                reserve_stage_calls(
                    run_root, stage, attempt,
                    {"counts": stage_nominal_for_mode(mode, "I")},
                    execution=execution)
                i_provider = (provider.for_stage("I")
                              if callable(getattr(provider, "for_stage", None))
                              else provider)
                i = run_i_stage(
                    run_root, stage, attempt, stage_t0, budget, i_provider,
                    counter, provider.roster, p_dir, tolerance=tolerance,
                    guard=_stage_unit_guard(
                        run_root, stage, attempt,
                        registration_sha=registration_sha,
                        deadline=stage_deadline),
                    registered=True,
                    r0_noise=(provider.noise if mode == "synthetic" else None),
                    r0_noise_path=noise_path, r0_noise_sha256=noise_sha,
                    parent_input_admission=p_input_admission,
                    r0_registration_sha=registration_sha)
                settle_stage_calls(run_root, stage, attempt,
                                   _stage_delta(counter, base_counts))
                results[stage] = i
                i_dir = stage_dir(run_root, stage) / attempt
                i_report = Path(i["report"]) if i.get("report") else None
                artifacts = [Path(path) for path in
                             (i.get("paths") or {}).values()]
                artifacts += [i_report, i_dir / "i_rows.json",
                              i_dir / "call_reservations.json"]
                seal, commit, closed, closed_sha, problems = \
                    _migration_stage_seal(
                        run_root, stage, attempt, stage_t0, budget, binding,
                        execution, {"complete": bool(
                            i.get("complete_evidence"))},
                        complete=bool(i.get("complete_evidence")),
                        report=i_report, artifacts=artifacts, counter=counter,
                        scope=scope, numeric_passed=bool(
                            i.get("numeric_passed")))
                if problems or not commit.committed:
                    return failed(list(i.get("gaps") or []) + problems,
                                  list(i.get("gaps") or []) + problems,
                                  list(i.get("failures") or []))
                _launch_closed, launch_problems = \
                    write_launch_stage_closed_candidate(
                        run_root, invocation_id=invocation_id, stage=stage,
                        execution_seq=execution["execution_seq"],
                        commit=commit, event_candidate=closed,
                        event_candidate_sha256=closed_sha, t0=stage_t0,
                        remaining_s=float(budget["remaining_s"]),
                        committed_measured_wall_s=float(
                            seal.row.get("measured_wall_s", 0.0) or 0.0))
                if launch_problems:
                    return failed(launch_problems, launch_problems,
                                  list(i.get("failures") or []))
                stage_cursor = float(_now())
                stage_timings[stage] = {
                    "t0_perf_counter": float(stage_t0),
                    "end_perf_counter": stage_cursor}
                commits[stage] = {"path": path_for_record(commit.commit_path),
                                  "sha256": commit.commit_sha256}
                candidates[stage] = {"path": path_for_record(closed),
                                     "sha256": closed_sha}
                selected[stage] = attempt
            except (TimeoutError, UpdateCallBudgetExceeded, OSError,
                    ValueError, TypeError) as exc:
                return failed([f"migration_i:{type(exc).__name__}:{exc}"],
                              [f"migration_i:{type(exc).__name__}"])
            continue
        # F
        base_counts = dict(counter.counts)
        stage_deadline = (stage_t0 + float(budget["remaining_s"])
                          - float(budget["reserve_s"]))
        try:
            pre_register_attempt(
                run_root, stage, attempt, stage_t0, scope=scope,
                budget=budget, registration_sha=registration_sha,
                call_balance=counter.snapshot(),
                execution={**execution,
                           "registration_sha256": registration_sha})
            reserve_stage_calls(
                run_root, stage, attempt,
                {"counts": stage_nominal_for_mode(mode, "F")},
                execution=execution)
            candidates_map = {"R0": selected["R0"], "P": selected["P"],
                              "I": selected["I"], "F": attempt}
            f_provider = (provider.for_stage("F")
                          if callable(getattr(provider, "for_stage", None))
                          else provider)
            f = run_f_stage(
                run_root, stage, attempt, stage_t0, budget, f_provider,
                counter, provider.roster, p_dir, i_dir, tolerance=tolerance,
                project_root=project_root, candidate_attempts=candidates_map,
                r0_noise_path=noise_path, r0_noise_sha256=noise_sha,
                p_input_admission_path=(p_input_admission_path
                                        if mode == "real" else None),
                p_input_admission_sha256=(p_input_admission_sha
                                          if mode == "real" else None),
                execution=execution)
            snapshot = f.get("call_delta")
            settled, settle_problems = _settle_f_calls(
                run_root, stage, attempt, snapshot)
            if settle_problems:
                f.setdefault("verification_problems", []).extend(
                    settle_problems)
            if not settled:
                return failed(["f_call_settlement_unknown"],
                              ["f_call_settlement_unknown"])
            final_balance = counter.snapshot()
            final_counts = _call_counts_from_balance(final_balance)
            if final_counts is None:
                return failed(["f_call_balance_invalid"],
                              ["f_call_balance_invalid"])
            current = _attempt_row(run_root, stage, attempt)
            upsert_index(run_root, stage, attempt, {
                **current, "call_balance": final_balance,
                "call_balance_after": final_balance})
            upsert_cost(
                run_root, stage, attempt,
                float(current.get("measured_wall_s", 0.0)),
                {"complete": False, "scope": scope,
                 "call_balance": final_balance,
                 "call_balance_before": current.get("call_balance_before"),
                 "call_balance_after": final_balance,
                 "accounting_schema": ACCOUNTING_SCHEMA_V2,
                 "generation": binding.generation,
                 "binding_spec_sha256": binding.spec_sha256,
                 "execution_seq": execution["execution_seq"],
                 "predecessor_kind": current.get("predecessor_kind"),
                 "previous_head_sha256": current.get(
                     "previous_head_sha256"),
                 "invocation_id": invocation_id,
                 "claim_path": path_for_record(claim_path),
                 "claim_sha256": claim_sha},
                "running")
            results[stage] = f
            f_dir = stage_dir(run_root, stage) / attempt
            f_report = Path(f["report"]) if f.get("report") else None
            f_evidence_veto: list[str] = []
            if f.get("complete_evidence") is not True:
                f_evidence_veto.append("f_complete_evidence")
            for problem in f.get("verification_problems") or []:
                f_evidence_veto.append(f"f_verification:{problem}")
            if f_evidence_veto:
                _amend_final_report(f_report, f_evidence_veto)
                wrow, wprobs, proven = withdraw_attempt_claim(
                    run_root, stage, attempt, f_report,
                    _attempt_row(run_root, stage, attempt),
                    f_evidence_veto, reason="f_evidence_veto",
                    t0=stage_t0, budget=budget,
                    call_balance=final_balance)
                results["F"]["withdrawal"] = {
                    "problems": wprobs, "proven": bool(proven),
                    "row": wrow}
                return failed(f_evidence_veto + wprobs,
                              f_evidence_veto + wprobs,
                              list(f.get("failures") or []))
            pre_veto: list[str] = []
            history = verify_history(
                run_root, allow_current=(stage, attempt),
                allow_invocation=invocation_id, binding=binding)
            pre_veto += history
            deltas = _stage_deltas(run_root, require_commits=False)
            if deltas is None:
                pre_veto.append("f_call_delta_unknown")
            else:
                current_rec = deltas.get(stage, {}).get(attempt)
                if current_rec is None:
                    pre_veto.append("f_current_delta_missing")
                selected = {"R0": candidates_map["R0"],
                            "P": candidates_map["P"],
                            "I": candidates_map["I"], "F": attempt}
                selected_chain = selected_chain_counts(deltas, selected)
                pre_veto += _selected_stage_problems(deltas, selected, mode)
                pre_veto += _selected_caps_problems(deltas, mode)
                cohort_totals = cohort_call_totals_from_deltas(deltas)
                stage_totals = {name: stage_call_totals(deltas, name)
                                for name in STAGE_BUDGETS}
                worker = f.get("worker_result") or {}
                if worker.get("provisional_chain_counts") != selected_chain:
                    pre_veto.append("f_provisional_chain")
            if pre_veto:
                _amend_final_report(f_report, pre_veto)
                return failed(pre_veto, pre_veto, list(f.get("failures") or []))
            selection_record = {
                "schema": "w1_real_training_update_terminal_selection_v1",
                "selected_attempts": selected,
                "selected_chain_counts": selected_chain,
                "selected_chain_training_updates": int(
                    (selected_chain or {}).get("native_adamw_steps", 0)
                    + (selected_chain or {}).get("explicit_reference_updates",
                                                 0)),
                "stage_call_counts": stage_totals,
                "cohort_call_counts": cohort_totals,
                "cohort_training_updates": int(
                    (cohort_totals or {}).get("native_adamw_steps", 0)
                    + (cohort_totals or {}).get("explicit_reference_updates",
                                                0)),
                "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if f_report is not None and f_report.is_file():
                data = strict_read_json(f_report)
                data.update(
                    selected_attempts=selected,
                    selected_chain_counts=selected_chain,
                    selected_chain_training_updates=selection_record[
                        "selected_chain_training_updates"],
                    stage_call_counts=stage_totals,
                    cohort_call_counts=cohort_totals,
                    cohort_training_updates=selection_record[
                        "cohort_training_updates"],
                    training_updates_scope="cohort",
                    training_updates=selection_record[
                        "cohort_training_updates"])
                write_json(f_report, data)
            f_artifacts = [f_report, f_dir / "worker_spec.json",
                           f_dir / "worker" / "worker_result.json",
                           f_dir / "call_reservations.json"]
            seal, commit, closed, closed_sha, problems = \
                _migration_stage_seal(
                    run_root, stage, attempt, stage_t0, budget, binding,
                    execution, {"complete": True}, complete=True,
                    report=f_report, artifacts=f_artifacts, counter=counter,
                    scope=scope, final=True,
                    extra_row={"selection": selection_record},
                    extra={"selection": selection_record})
            if problems or not commit.committed:
                return failed(problems, problems, list(f.get("failures") or []))
            _launch_closed, launch_problems = \
                write_launch_stage_closed_candidate(
                    run_root, invocation_id=invocation_id, stage=stage,
                    execution_seq=execution["execution_seq"], commit=commit,
                    event_candidate=closed,
                    event_candidate_sha256=closed_sha, t0=stage_t0,
                    remaining_s=float(budget["remaining_s"]),
                    committed_measured_wall_s=float(
                        seal.row.get("measured_wall_s", 0.0) or 0.0))
            if launch_problems:
                return failed(launch_problems, launch_problems,
                              list(f.get("failures") or []))
            terminal_path, terminal_sha, terminal_problems = \
                write_terminal_candidate(
                    run_root, binding=binding, execution={**execution,
                                                          "stage": stage},
                    commit=commit, t0=stage_t0,
                    remaining_s=float(budget["remaining_s"]),
                    committed_measured_wall_s=float(
                        seal.row.get("measured_wall_s", 0.0) or 0.0))
            launch_terminal_path, launch_terminal_problems = \
                write_launch_terminal_candidate(
                    run_root, invocation_id=invocation_id, commit=commit,
                    event_candidate=terminal_path,
                    event_candidate_sha256=terminal_sha, t0=stage_t0,
                    remaining_s=float(budget["remaining_s"]),
                    committed_measured_wall_s=float(
                        seal.row.get("measured_wall_s", 0.0) or 0.0))
            terminal_problems += launch_terminal_problems
            f_write_after = not terminal_problems
            commits[stage] = {"path": path_for_record(commit.commit_path),
                              "sha256": commit.commit_sha256}
            candidates[stage] = {"path": path_for_record(closed),
                                 "sha256": closed_sha}
            selected[stage] = attempt
            if terminal_problems:
                return failed(terminal_problems, terminal_problems,
                              list(f.get("failures") or []))
            f_end = float(_now())
            stage_timings[stage] = {
                "t0_perf_counter": float(stage_t0),
                "end_perf_counter": f_end}
            return {
                "mode": mode,
                "conclusion": ("real_training_update_covered_contract_passed"
                               if f.get("covered_contract_passed")
                               else "audit_incomplete"),
                "complete_evidence": bool(f.get("complete_evidence")),
                "covered_contract_passed": bool(
                    f.get("covered_contract_passed")),
                "problems": [], "failures": list(f.get("failures") or []),
                "gaps": list(f.get("gaps") or []), "stages": results,
                "selected_attempts": selected,
                "selected_chain_counts": selected_chain,
                "selected_chain_training_updates": selection_record[
                    "selected_chain_training_updates"],
                "cohort_call_counts": cohort_totals,
                "cohort_training_updates": selection_record[
                    "cohort_training_updates"],
                "training_updates": selection_record[
                    "cohort_training_updates"],
                "training_updates_known_lower_bound": selection_record[
                    "cohort_training_updates"],
                "training_updates_scope": "cohort",
                "call_accounting_known": True,
                "stage_call_counts": stage_totals,
                "commits": commits, "candidates": candidates,
                "terminal_candidate": {
                    "path": path_for_record(terminal_path),
                    "sha256": terminal_sha},
                "write_after": {**{name: True for name in STAGE_BUDGETS},
                                "terminal": f_write_after},
                "invocation_id": invocation_id,
                "binding_spec_sha256": binding.spec_sha256,
                "stage_timings": stage_timings,
                "last_stage_end_perf_counter": f_end,
            }
        except (TimeoutError, UpdateCallBudgetExceeded, OSError,
                ValueError, TypeError) as exc:
            return failed([f"migration_f:{type(exc).__name__}:{exc}"],
                          [f"migration_f:{type(exc).__name__}"])
    return failed(["migration_incomplete"])


def synthetic_self_test(out_root: Path, *, seed: int = FIXTURE_SEED,
                        device: str = "cuda",
                        tolerance: Mapping[str, Any] | None = None,
                        config: Mapping[str, Any] | None = None,
                        calibration_verification_record: Mapping[str, Any]
                        | None = None) -> dict:
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
    if calibration_verification_record is not None:
        from ireset.eval.w1_real_training_update_provider import (
            synthetic_acceptance,
        )
        acceptance = synthetic_acceptance(
            result, run_root=run_root, tolerance_contract=contract,
            calibration_verification_record=calibration_verification_record,
            roster=provider.roster)
        acceptance_path = run_root / "synthetic_acceptance.json"
        write_json(acceptance_path, acceptance)
        result["synthetic_acceptance"] = {
            "path": path_for_record(acceptance_path),
            "sha256": sha256_file(acceptance_path),
            "passed": acceptance["passed"],
            "problems": acceptance["problems"],
        }
    write_json(run_root / "audit_result.json", result)
    return result
