"""Real-batch W1 training-objective audit harness (pre-registration draft).

This module implements the *infrastructure* for the plan in
``docs/plans/omniretarget_w1_real_objective_audit_prereg_draft.md``.  It is
deliberately split into two entry modes:

* ``synthetic`` -- a self-contained analytic fixture with tiny production
  modules and the audit FK stand-in.  It exercises the complete registration,
  production-capture, batched-VJP, independent-reference, verifier and
  accounting paths without reading a real dataset, cache, checkpoint or
  history root.
* ``real`` -- the registered path.  It refuses to run unless a hash-bound
  authorization record and a frozen synthetic tolerance contract are supplied.
  The draft's section 8 checklist is not completed in this repository state,
  so the real mode is intentionally unreachable here.

The audit measures the **pre-accumulation micro objective**.  It does not run
an optimizer step, parameter finite differences, DDIM sampling, checkpoint
writes or model selection.  All conclusions keep
``objective_coverage="partial"`` and ``full_objective_contract_passed=None``.

Nothing in this module touches disk at import time.  Real asset loading lives
in :class:`RealW1ObjectiveProvider` and is only imported/instantiated by the
explicit real entry after authorization checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from ireset.paths import PROJECT_ROOT, path_for_record
from ireset.train.losses import LossAssembler, LossWeights, HAND_SITE_NAMES
from ireset.embodied.g1_spec import KPT_TO_SMPLH

# ---------------------------------------------------------------------------
# identity and frozen matrix
# ---------------------------------------------------------------------------

AUDIT_SCHEMA = "w1_real_objective_audit_v1"
EXECUTION_ID = "w1_real_objective_audit_v1"
COHORT_ID = "w1_real_objective_audit_v1"
ACCOUNTING_SCHEMA = "pre_registered_v1"
AUDIT_SOURCE = "real_batch_partial_objective"

WINDOW = 60
STRIDE = 30
MODELS = ("deterministic", "diffusion")
TERMS = ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2", "total")
GRADIENT_SLICES = ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "L5")
GRADIENT_SLICE_INDEX = {k: i for i, k in enumerate(GRADIENT_SLICES)}
TERM_INDEX = {k: i for i, k in enumerate(TERMS)}

#: Four windows are selected from the frozen train24 metadata.  The order is
#: demo-name lexicographic, then start ascending.  R0 records the actual keys;
#: these constants only define the selection rule and the diffusion inputs.
WINDOW_COUNT = 4
DEMO_COUNT = 2
WINDOWS_PER_DEMO = 2
NOISE_SEED = 20261008
DIFFUSION_TIMESTEPS = (0, 250, 500, 999)
NOISE_PROTOCOL = "w1_real_objective_audit_v1"

STAGE_BUDGETS = {
    "R0": {"budget_s": 180.0, "reserve_s": 30.0, "attempts": 2},
    "P": {"budget_s": 360.0, "reserve_s": 60.0, "attempts": 2},
    "F": {"budget_s": 360.0, "reserve_s": 60.0, "attempts": 2},
}
TOTAL_BUDGET_S = 900.0
SEAL_TAIL_RESERVE_S = 1.0
SEAL_VERIFY_ROUNDS = 4
STAGE_DIRS = {"R0": "registration", "P": "production", "F": "finalize"}

CALL_CAPS = {
    "conditioner": 4,
    "model": 4,
    "production_vjp": 4,
    "independent_vjp": 4,
}

#: Worst-case calls for one complete R0->P->F chain: two units in P and two
#: in F.  The entry must prove this balance exists before it starts any work;
#: a chain that can only finish by raising AuditCallBudgetExceeded is refused.
FULL_CHAIN_CALLS = {kind: 2 * len(MODELS) for kind in CALL_CAPS}

#: Candidate fixed assets from the draft, section 3.  Full hashes are verified
#: at registration time; they are not used as a substitute for the file
#: closure.  The W1-D execution registration and final are parent anchors.
SOURCE_RUN = Path(
    "derived/experiments/omniretarget_first/"
    "20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8"
)
W1_RUN = Path(
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de"
)
ATTR_RUN = Path(
    "derived/experiments/omniretarget_first/"
    "20261021_worktree_w1_prediction_consistency_r6"
)
HISTORY_ROOT = Path("derived/experiments/omniretarget_first")
TRAIN_CONFIG = Path("configs/train/omniretarget_motion.yaml")
EXPECTED_CHECKPOINTS = {
    "deterministic": (
        W1_RUN / "formal/deterministic/attempt_001/best.pt",
        "7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff",
        4000,
    ),
    "diffusion": (
        W1_RUN / "formal/diffusion/attempt_001/best.pt",
        "1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1",
        4000,
    ),
}
PARENT_ANCHORS = {
    "w1_execution_registration": (
        W1_RUN / "freeze/execution_registration.json",
        "5561f68562189b7ba45f7193c280739e6dde5325b4df2ada82e8a9cf69ba5f7e",
    ),
    "w1_final": (
        W1_RUN / "decisions/attempt_001/w1_de_final.json",
        "7f1a9a595cf7964e99a929c83d3bcaac4aa3482a96e515228921b7b67a576737",
    ),
    "prediction_consistency_registration": (
        ATTR_RUN / "freeze/execution_registration.json",
        "5fcf8b3a655d028799ce2fb35266ed6ee055437eec5f743a36e15186b5d70cea",
    ),
    "prediction_consistency_final": (
        ATTR_RUN / "decisions/attempt_001/prediction_consistency_final.json",
        "ce9843fd653267e1311ac6d89b59aac7fe9f61ed17611b71db29c624a1148f9b",
    ),
}
SOURCE_ANCHORS = {
    "source_training_manifest": (
        SOURCE_RUN / "audit/attempt_001/training_manifest.json",
        "a350a8b8922ca94308de3d97787e60bf8f6a51acff6f20af9f513fd42b10e525",
    ),
}
# Parent anchors are verified against the recorded full SHA-256; a missing
# file or a mismatch is a hard registration problem.  Old rejected runs are
# referenced, never charged to this cohort's budgets.

#: Unified residual domains.  The production loss uses float32; the
#: independent reference uses float64.  These are synthetic-calibrated
#: candidates, written into ``tolerance_contract.json`` before real work.
DEFAULT_TOLERANCES = {
    "encode_float64": {"atol": 1e-9, "rtol": 1e-9, "scale_floor": 0.0},
    "encode_float32": {"atol": 2e-5, "rtol": 2e-5, "scale_floor": 0.0},
    "loss_float64": {"atol": 1e-9, "rtol": 1e-9, "scale_floor": 0.0},
    "loss_float32": {"atol": 2e-5, "rtol": 2e-5, "scale_floor": 0.0},
    "gradient_float64": {"atol": 1e-8, "rtol": 1e-6, "scale_floor": 1e-8},
    "gradient_float32": {"atol": 2e-5, "rtol": 2e-4, "scale_floor": 1e-7},
    "rotation_domain": {"min_angle_rad": 1e-6, "min_sin_theta": 1e-3},
}

TERM_FLOAT32_OVERRIDES = {
    # float32 acos near (but not at) the pi branch loses ~1e-4 rad; the
    # registered geodesic term is theta^2, so its absolute error scales with
    # 2*theta.  This is a precision allowance, not a structural pass: values
    # outside the independent rotation domain are still gaps.
    "geodesic_rad2": {"atol": 2.0e-3, "rtol": 1.0e-4},
    "total": {"atol": 1.0e-3, "rtol": 1.0e-4},
    "L5": {"atol": 1.0e-3, "rtol": 1.0e-4},
}

BOUNDARIES = {
    "objective_coverage": "partial",
    "full_objective_contract_passed": None,
    "attribution": "attribution_insufficient",
    "r3": "not_run",
    "final_interaction_method": "not_ready",
    "production_integration": False,
    "factor_selected": False,
    "auto_execute": False,
    "training_updates": 0,
    "parameter_fd": 0,
    "ddim_calls": 0,
    "checkpoint_writes": 0,
}

# ---------------------------------------------------------------------------
# small serialization helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if value is None or isinstance(value, str):
        return value
    return repr(value)


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False,
                             allow_nan=False))
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _num(value: Any) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)))


def _count(value: Any) -> bool:
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)) and int(value) >= 0)


def _nonneg(value: Any) -> bool:
    """Finite, numeric and non-negative: the only valid cost/effort domain."""
    return _num(value) and float(value) >= 0.0


def _array(value: Any, name: str, *, dtype=None, ndim: int | None = None) -> np.ndarray:
    a = np.asarray(value)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name}: ndim {a.ndim} != {ndim}")
    if dtype is not None and a.dtype != np.dtype(dtype):
        raise ValueError(f"{name}: dtype {a.dtype} != {np.dtype(dtype)}")
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"{name}: non-numeric dtype {a.dtype}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite")
    return a


def _tensor(value: Any, name: str, *, dtype=None, ndim: int | None = None) -> torch.Tensor:
    t = value if torch.is_tensor(value) else torch.as_tensor(value)
    if ndim is not None and t.ndim != ndim:
        raise ValueError(f"{name}: ndim {t.ndim} != {ndim}")
    if dtype is not None and t.dtype != dtype:
        raise ValueError(f"{name}: dtype {t.dtype} != {dtype}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name}: non-finite")
    return t


def _shape(value: Any) -> list[int]:
    return [int(x) for x in np.shape(value)]


def _hash_array(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


# ---------------------------------------------------------------------------
# data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowKey:
    demo: str
    start: int
    end: int
    T: int
    cache_key: str
    split: str
    input_sha: str
    derived_sha: str

    @property
    def key(self) -> str:
        return f"{self.demo}|{self.start}|{self.end}|{self.T}|{self.cache_key}"

    def as_dict(self) -> dict:
        return {
            "key": self.key, "demo": self.demo, "start": int(self.start),
            "end": int(self.end), "T": int(self.T),
            "cache_key": self.cache_key, "split": self.split,
            "input_sha": self.input_sha, "derived_sha": self.derived_sha,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WindowKey":
        return cls(
            demo=str(d["demo"]), start=int(d["start"]), end=int(d["end"]),
            T=int(d["T"]), cache_key=str(d["cache_key"]),
            split=str(d["split"]), input_sha=str(d["input_sha"]),
            derived_sha=str(d["derived_sha"]),
        )


@dataclass
class AuditInputs:
    """Numeric inputs after the four registered windows are loaded."""

    window_keys: list[WindowKey]
    root: np.ndarray             # [4,60,7] world
    q: np.ndarray                # [4,60,29]
    wfc: np.ndarray              # [4,4,4] world_from_clip
    kps_obj: np.ndarray          # [4,60,52,3]
    kps_clip: np.ndarray         # [4,60,52,3]
    box_t_clip: np.ndarray       # [4,60,3]
    box_quat: np.ndarray         # [4,60,4] wxyz
    contact: np.ndarray          # [4,60,4]
    x0: np.ndarray               # [4,60,38] production tokenizer, float32/64
    x0_independent: np.ndarray   # [4,60,38] independent reference, float64
    joint_lo: np.ndarray         # [29]
    joint_hi: np.ndarray         # [29]
    root_mean: np.ndarray        # [3]
    root_std: np.ndarray         # [3]
    t: np.ndarray = field(default_factory=lambda: np.asarray(DIFFUSION_TIMESTEPS, dtype=np.int64))
    noise: np.ndarray = field(default_factory=lambda: np.zeros((WINDOW_COUNT, WINDOW, 38), dtype=np.float32))
    dtype: str = "float32"

    @property
    def B(self) -> int:
        return int(self.root.shape[0])

    @property
    def F(self) -> int:
        return int(self.root.shape[1])

    def condition_batch(self, device="cpu", dtype=None) -> dict:
        """Production condition dict (only the keys the conditioner consumes)."""
        td = torch.float32 if dtype is None else dtype
        return {
            "x0": torch.as_tensor(self.x0, dtype=td, device=device),
            "kps_obj": torch.as_tensor(self.kps_obj, dtype=td, device=device),
            "kps_clip": torch.as_tensor(self.kps_clip, dtype=td, device=device),
            "box_t_clip": torch.as_tensor(self.box_t_clip, dtype=td, device=device),
            "box_quat": torch.as_tensor(self.box_quat, dtype=td, device=device),
            "contact": torch.as_tensor(self.contact, dtype=td, device=device),
            "wfc": torch.as_tensor(self.wfc, dtype=td, device=device),
        }

    def arrays_for_npz(self) -> dict[str, np.ndarray]:
        return {
            "raw_root": self.root,
            "raw_q": self.q,
            "raw_wfc": self.wfc,
            "kps_obj": self.kps_obj,
            "kps_clip": self.kps_clip,
            "box_t_clip": self.box_t_clip,
            "box_quat": self.box_quat,
            "contact": self.contact,
            "x0": self.x0,
            "x0_independent": self.x0_independent,
            "joint_lo": self.joint_lo,
            "joint_hi": self.joint_hi,
            "root_mean": self.root_mean,
            "root_std": self.root_std,
            "t": self.t,
            "noise": self.noise,
        }


@dataclass
class ModelBundle:
    model_type: str
    model: Any
    conditioner: Any
    metadata: dict = field(default_factory=dict)

    def state_dicts(self) -> dict[str, dict]:
        return {"model": self.model.state_dict(),
                "conditioner": self.conditioner.state_dict()}


# ---------------------------------------------------------------------------
# cohort / attempt accounting
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
        raise ValueError(f"measured wall time must be finite and non-negative: {measured}")
    entry = {
        "status": status,
        "complete": bool((extra or {}).get("complete", False)),
        "measured_wall_s": measured,
        "wall_s": round(measured, 1),
        "accounting_schema": ACCOUNTING_SCHEMA,
        **(extra or {}),
    }
    upsert_ledger(root, stage, attempt, entry)


def _disk_attempts(root: Path, stage: str) -> set[str]:
    base = stage_dir(root, stage)
    if not base.is_dir():
        return set()
    return {p.name for p in base.iterdir()
            if p.is_dir() and re.fullmatch(r"attempt_\d+", p.name)}


def attempt_identity(root: Path) -> dict:
    return {
        "execution_id": EXECUTION_ID,
        "cohort_id": COHORT_ID,
        "accounting_schema": ACCOUNTING_SCHEMA,
        "audit_schema": AUDIT_SCHEMA,
    }


def write_cohort(root: Path, *, mode: str, identity: dict | None = None,
                 authorization_sha: str | None = None) -> None:
    path = _cohort_path(root)
    existing = read_json(path) if path.is_file() else {}
    record = {
        **attempt_identity(root),
        "mode": mode,
        "identity": identity or {},
        "authorization_sha256": authorization_sha,
        "stage_budgets": {k: dict(v) for k, v in STAGE_BUDGETS.items()},
        "total_budget_s": TOTAL_BUDGET_S,
        "call_caps": dict(CALL_CAPS),
    }
    if existing and existing.get("cohort_id") != COHORT_ID:
        raise ValueError("cohort.json belongs to another cohort")
    if existing:
        record = {**existing, **record}
    write_json(path, record)


def allocate_attempt(root: Path, stage: str) -> str:
    seen = (_disk_attempts(root, stage)
            | {r.get("attempt") for r in load_index(root)
               if r.get("stage") == stage}
            | {r.get("attempt") for r in load_ledger(root)
               if r.get("stage") == stage})
    nums = []
    for name in seen:
        if isinstance(name, str) and re.fullmatch(r"attempt_\d+", name):
            nums.append(int(name.split("_")[1]))
    n = max(nums, default=0) + 1
    return f"attempt_{n:03d}"


def _stage_rows(root: Path, stage: str) -> tuple[list[dict], list[dict], set[str]]:
    index = [r for r in load_index(root) if r.get("stage") == stage]
    ledger = [r for r in load_ledger(root) if r.get("stage") == stage]
    return index, ledger, _disk_attempts(root, stage)


def _row_identity_problems(row: Mapping[str, Any], *, kind: str) -> list[str]:
    problems: list[str] = []
    if row.get("accounting_schema") != ACCOUNTING_SCHEMA:
        problems.append(f"{kind}_schema_missing")
    if row.get("cohort_id") != COHORT_ID:
        problems.append(f"{kind}_cohort_missing")
    if not isinstance(row.get("complete"), bool):
        problems.append(f"{kind}_completion_invalid")
    status = row.get("status")
    if status not in ("running", "publishing", "done", "crashed",
                      "timeout", "incomplete"):
        problems.append(f"{kind}_status_invalid")
    return problems


def stage_history(root: Path, stage: str, *, allow_current: tuple[str, str] | None = None) -> float | None:
    """Cohort-stage cost only when disk/index/ledger agree on every attempt.

    ``allow_current=(stage, attempt)`` permits the current in-process attempt's
    allocation-time ``running`` row; it is still not treated as completed
    history and its cost is returned separately by :func:`budget_for`.
    """
    try:
        index, ledger, disk = _stage_rows(root, stage)
        index_by = {r.get("attempt"): r for r in index}
        ledger_by = {r.get("attempt"): r for r in ledger}
        if (len(index_by) != len(index) or len(ledger_by) != len(ledger)
                or None in index_by or None in ledger_by):
            return None
        if set(disk) != set(index_by) or set(disk) != set(ledger_by):
            return None
        total = 0.0
        for attempt, row in index_by.items():
            e = ledger_by[attempt]
            current = allow_current == (stage, attempt)
            if current:
                # the current attempt has a pre-registered running row; do
                # not charge it twice and do not infer zero history from it
                continue
            problems = _row_identity_problems(row, kind="index")
            problems += _row_identity_problems(e, kind="ledger")
            if problems:
                return None
            if row.get("status") != e.get("status"):
                return None
            if row.get("complete") is not e.get("complete"):
                return None
            if row.get("status") not in ("done", "crashed", "timeout", "incomplete"):
                return None
            if e.get("status") not in ("done", "crashed", "timeout", "incomplete"):
                return None
            if row.get("status") != "done" and row.get("complete") is not False:
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
            if "wall_s" in row:
                if not _nonneg(row["wall_s"]) or float(row["wall_s"]) != round(float(measured), 1):
                    return None
            total += float(measured)
        return total
    except (OSError, ValueError, TypeError, KeyError):
        return None


def total_history(root: Path, *, allow_current: tuple[str, str] | None = None) -> float | None:
    total = 0.0
    for stage in STAGE_BUDGETS:
        value = stage_history(root, stage, allow_current=allow_current)
        if value is None:
            return None
        total += value
    return total


def _cohort_roots(fixed_root: Path) -> Iterable[Path]:
    base = Path(fixed_root)
    if not base.is_dir():
        return []
    return [p for p in sorted(base.iterdir()) if p.is_dir()]


def cohort_history(fixed_root: Path, *, exclude_root: Path | None = None,
                   allow_current: tuple[str, str] | None = None) -> dict:
    """Scan the fixed history root for this cohort's sealed attempts.

    Only direct children of the fixed root are considered.  A run belongs to
    the cohort when its ``cohort.json`` has the cohort id or when an
    allocation-time ledger/index row carries the cohort id and accounting
    schema.  Unknown/foreign matching attempts make the whole history unknown.
    """
    runs: list[dict] = []
    by_stage = {stage: 0.0 for stage in STAGE_BUDGETS}
    total = 0.0
    base = Path(fixed_root)
    if not base.is_dir():
        return {"known": True, "runs": [], "stage_cost_s": by_stage, "total_s": 0.0}
    for sibling in _cohort_roots(base):
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
        try:
            index = load_index(sibling)
            ledger = load_ledger(sibling)
        except (OSError, ValueError, TypeError):
            return {"known": False, "reason": f"unreadable_ledger:{sibling.name}",
                    "runs": runs}
        matching_marker = marker.get("cohort_id") == COHORT_ID
        matching_rows = any(
            r.get("cohort_id") == COHORT_ID
            and r.get("accounting_schema") == ACCOUNTING_SCHEMA
            for r in (*index, *ledger)
        )
        if not (matching_marker or matching_rows):
            continue
        if marker and marker.get("cohort_id") not in (None, COHORT_ID):
            return {"known": False, "reason": f"foreign_cohort:{sibling.name}",
                    "runs": runs}
        if marker and marker.get("execution_id") not in (None, EXECUTION_ID):
            return {"known": False, "reason": f"foreign_execution:{sibling.name}",
                    "runs": runs}
        stage_costs = {}
        for stage in STAGE_BUDGETS:
            value = stage_history(
                sibling, stage,
                allow_current=allow_current if sibling.resolve() == Path(exclude_root).resolve() else None,
            )
            if value is None:
                return {"known": False,
                        "reason": f"unknown_history:{sibling.name}:{stage}",
                        "runs": runs}
            stage_costs[stage] = value
            by_stage[stage] += value
            total += value
        runs.append({"run": path_for_record(sibling), "stage_cost_s": stage_costs})
    return {"known": True, "runs": runs, "stage_cost_s": by_stage, "total_s": total}


def cohort_attempts(fixed_root: Path, stage: str, *,
                    exclude_root: Path | None = None) -> int | None:
    count = 0
    for sibling in _cohort_roots(fixed_root):
        if exclude_root is not None and sibling.resolve() == Path(exclude_root).resolve():
            continue
        try:
            marker = read_json(sibling / "cohort.json") if (sibling / "cohort.json").is_file() else {}
            index = load_index(sibling)
            ledger = load_ledger(sibling)
        except (OSError, ValueError, TypeError):
            return None
        matching = (marker.get("cohort_id") == COHORT_ID
                    or any(r.get("cohort_id") == COHORT_ID for r in (*index, *ledger)))
        if not matching:
            continue
        index_attempts = {r.get("attempt") for r in index if r.get("stage") == stage}
        ledger_attempts = {e.get("attempt") for e in ledger if e.get("stage") == stage}
        disk = _disk_attempts(sibling, stage)
        if index_attempts != ledger_attempts or index_attempts != disk:
            return None
        if None in index_attempts:
            return None
        count += len(index_attempts)
    return count


def _call_counts_from_balance(balance: Any) -> dict[str, int] | None:
    """Strictly decode one sealed attempt's call ledger; None means unknown."""
    if not isinstance(balance, Mapping):
        return None
    if balance.get("schema") != "w1_real_objective_call_budget_v1":
        return None
    counts = balance.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(CALL_CAPS):
        return None
    out: dict[str, int] = {}
    for kind in CALL_CAPS:
        value = counts.get(kind)
        if not _count(value) or int(value) > int(CALL_CAPS[kind]):
            return None
        out[kind] = int(value)
    if balance.get("within_caps") is not True:
        return None
    return out


def prior_call_counts(root: Path, *, fixed_root: Path,
                      mode: str = "real") -> dict[str, int] | None:
    """Sum cohort-wide call usage from sealed attempts in the fixed history root.

    Every attempted stage must carry a decodable ``call_balance`` in both the
    index and the ledger.  A single missing/contradictory row makes the whole
    history unknown, so the caller stops instead of granting a fresh quota.
    """
    roots: list[Path] = []
    if mode == "real":
        base = Path(fixed_root)
        if not base.is_dir():
            return None
        current = Path(root).resolve()
        for sibling in _cohort_roots(base):
            if sibling.resolve() == current:
                continue
            marker_path = sibling / "cohort.json"
            try:
                marker = read_json(marker_path) if marker_path.is_file() else {}
                index = load_index(sibling)
                ledger = load_ledger(sibling)
            except (OSError, ValueError, TypeError):
                return None
            matching = (isinstance(marker, Mapping)
                        and marker.get("cohort_id") == COHORT_ID)
            matching = matching or any(
                r.get("cohort_id") == COHORT_ID for r in (*index, *ledger))
            if not matching:
                continue
            if isinstance(marker, Mapping) and marker.get("cohort_id") not in (None, COHORT_ID):
                continue
            roots.append(sibling)
    roots.append(Path(root))
    totals = {kind: 0 for kind in CALL_CAPS}
    for run_root in roots:
        try:
            index = load_index(run_root)
            ledger = load_ledger(run_root)
        except (OSError, ValueError, TypeError):
            return None
        index_rows = [r for r in index
                      if r.get("cohort_id") == COHORT_ID
                      and r.get("accounting_schema") == ACCOUNTING_SCHEMA]
        ledger_rows = [r for r in ledger
                       if r.get("cohort_id") == COHORT_ID
                       and r.get("accounting_schema") == ACCOUNTING_SCHEMA]
        index_by = {(r.get("stage"), r.get("attempt")): r for r in index_rows}
        ledger_by = {(e.get("stage"), e.get("attempt")): e for e in ledger_rows}
        if (len(index_by) != len(index_rows) or len(ledger_by) != len(ledger_rows)
                or None in {k[0] for k in index_by}
                or None in {k[1] for k in index_by}
                or None in {k[0] for k in ledger_by}
                or None in {k[1] for k in ledger_by}):
            return None
        if set(index_by) != set(ledger_by):
            return None
        for stage in STAGE_BUDGETS:
            disk = _disk_attempts(run_root, stage)
            idx = {attempt for (s, attempt) in index_by if s == stage}
            led = {attempt for (s, attempt) in ledger_by if s == stage}
            if disk != idx or disk != led:
                return None
        # ``call_balance`` is a cumulative cohort snapshot, not a per-stage
        # delta.  Taking the maximum (and requiring monotonicity in append
        # order) deducts earlier attempts/runs exactly once.
        previous = {kind: 0 for kind in CALL_CAPS}
        for row in index_rows:
            key = (row.get("stage"), row.get("attempt"))
            status = row.get("status")
            if status in ("running", "publishing"):
                return None
            if status not in ("done", "crashed", "timeout", "incomplete"):
                return None
            counts = _call_counts_from_balance(row.get("call_balance"))
            ledger_counts = _call_counts_from_balance(
                ledger_by[key].get("call_balance"))
            if counts is None or ledger_counts != counts:
                return None
            for kind in CALL_CAPS:
                if counts[kind] < previous[kind]:
                    return None
                previous[kind] = counts[kind]
        for kind in CALL_CAPS:
            totals[kind] = max(totals[kind], previous[kind])
    return totals


def budget_for(root: Path, stage: str, *, fixed_root: Path,
               mode: str = "real") -> dict:
    """Read history first, then return the available stage budget.

    ``mode="synthetic"`` uses an isolated root and skips the fixed-root cohort
    scan; the real entry requires the fixed root and a direct-child run root.
    """
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
            f"{prior_stage:.1f}s of {spec['budget_s']}s "
            f"(remaining {remaining:.1f}s <= reserve {spec['reserve_s']}s)")
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
                         unit_parent: dict | None = None,
                         call_balance: Mapping[str, Any] | None = None) -> dict:
    """Write the running index/ledger rows before any substantive work."""
    if "attempts" in budget and "attempts_used" in budget:
        if int(budget["attempts_used"]) >= int(budget["attempts"]):
            raise SystemExit(
                f"{stage} attempt limit reached before registration")
    if (Path(root) / "cohort.json").is_file():
        marker = read_json(Path(root) / "cohort.json")
        if marker.get("cohort_id") not in (None, COHORT_ID):
            raise ValueError("cohort marker mismatch")
    elapsed = max(0.0, time.perf_counter() - t0)
    deadline = t0 + float(budget["remaining_s"])
    entry = {
        "status": "running", "complete": False,
        "measured_wall_s": elapsed,
        "accounting_schema": ACCOUNTING_SCHEMA,
        "cohort_id": COHORT_ID,
        "execution_registration_sha256": registration_sha,
        "scope": scope,
        "budget": budget,
        "call_balance": dict(call_balance) if call_balance else None,
        "deadline_perf_counter": deadline,
        "allocated_wall_s": time.time(),
    }
    if unit_parent is not None:
        entry["unit_parent"] = unit_parent
    upsert_index(root, stage, attempt, entry)
    upsert_cost(root, stage, attempt, elapsed,
                {"complete": False, "cohort_id": COHORT_ID,
                 "scope": scope, "budget": budget,
                 "call_balance": dict(call_balance) if call_balance else None},
                "running")
    return entry


@dataclass
class SealResult:
    """``sealed`` means complete+terminal; ``terminal`` means the row is a
    trustworthy final record even when the attempt is a legitimate partial
    (incomplete) result.  Publishing/running leftovers are neither."""

    sealed: bool
    row: dict
    problems: list[str] = field(default_factory=list)
    terminal: bool = False


#: Seal reasons that mean the attempt was stopped by accounting, not by a
#: measured negative result.  Such a parent must not trigger new model calls
#: in F; F may only reorganise the already-written evidence.
BUDGET_REVOKE_REASONS = (
    "budget_exceeded_after_registration",
    "budget_exceeded_after_publication",
    "budget_exceeded_before_seal",
    "budget_exceeded_after_seal",
    "seal_tail_reserve_insufficient",
    "seal_tail_reserve_unverifiable",
)


def _budget_revoked(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    reason = str(row.get("incomplete_reason") or "")
    return reason.split(":", 1)[0] in BUDGET_REVOKE_REASONS


def _p_budget_exhausted(p: Mapping[str, Any], seal: SealResult) -> str | None:
    """Return the accounting reason that forbids new F model calls, if any."""
    if _budget_revoked(seal.row):
        return str(seal.row.get("incomplete_reason"))
    gap_lists: list[Any] = []
    if isinstance(p, Mapping):
        gap_lists.append(p.get("gaps"))
        report = p.get("report")
        if isinstance(report, Mapping):
            gap_lists.append(report.get("gaps"))
    for gaps in gap_lists:
        if not isinstance(gaps, (list, tuple)):
            continue
        for gap in gaps:
            text = str(gap)
            if text.startswith("unit_deadline:"):
                return text
            if "AuditCallBudgetExceeded" in text:
                return text
    return None


def _persist_row_entry(root: Path, stage: str, attempt: str,
                       row: Mapping[str, Any]) -> list[str]:
    """Persist an updated row in place to both the index and the ledger."""
    problems: list[str] = []
    measured = row.get("measured_wall_s")
    if not _nonneg(measured):
        measured = 0.0
    status = str(row.get("status") or "incomplete")
    if status not in ("running", "publishing", "done", "crashed",
                      "timeout", "incomplete"):
        status = "incomplete"
    complete = bool(row.get("complete")) if status == "done" else False
    payload = {
        key: value for key, value in row.items()
        if key not in ("stage", "attempt", "status", "complete",
                       "measured_wall_s", "wall_s")
    }
    payload["complete"] = complete
    try:
        upsert_index(root, stage, attempt, {
            **payload, "status": status, "measured_wall_s": float(measured)})
    except (OSError, ValueError, TypeError) as exc:
        problems.append(f"row_index_write_failed:{type(exc).__name__}")
    try:
        upsert_cost(root, stage, attempt, float(measured),
                    {**payload, "cohort_id": COHORT_ID}, status)
    except (OSError, ValueError, TypeError) as exc:
        problems.append(f"row_ledger_write_failed:{type(exc).__name__}")
    return problems


def _withdraw_pass_claim(root: Path, stage: str, attempt: str,
                         report_path: Path, row: Mapping[str, Any],
                         gaps: Sequence[str], *, reason: str,
                         extra: Mapping[str, Any] | None = None
                         ) -> tuple[dict, list[str], dict]:
    """Rewrite a pass summary as incomplete and rebind the sealed row hash.

    The on-disk claim is the authoritative one.  If the rewrite or the hash
    rebind cannot be proven, the row is left non-consumable (publishing or
    incomplete, complete=false) rather than keeping a readable pass claim.
    """
    problems: list[str] = []
    try:
        on_disk = read_json(report_path) if Path(report_path).is_file() else {}
    except (OSError, ValueError, TypeError):
        on_disk = {}
    summary = dict(on_disk) if isinstance(on_disk, Mapping) else {}
    summary["conclusion"] = "audit_incomplete"
    summary["producer_conclusion"] = "audit_incomplete"
    summary["complete_evidence"] = False
    summary["covered_contract_passed"] = False
    summary["revoked_reason"] = reason
    existing_gaps = summary.get("gaps")
    summary["gaps"] = (
        (list(existing_gaps) if isinstance(existing_gaps, list) else [])
        + [str(g) for g in gaps] + [reason]
    )
    if extra:
        summary.update(dict(extra))
    try:
        write_json(report_path, summary)
    except (OSError, ValueError, TypeError) as exc:
        problems.append(f"summary_revoke_write_failed:{type(exc).__name__}")
    status = row.get("status")
    if status not in ("running", "publishing", "done", "crashed",
                      "timeout", "incomplete"):
        status = "incomplete"
    if status == "done":
        status = "incomplete"
    updated = {
        **dict(row), "status": status, "complete": False,
        "incomplete_reason": row.get("incomplete_reason") or reason,
    }
    updated = _report_hashes(updated)
    problems += _persist_row_entry(root, stage, attempt, updated)
    if problems:
        # A claim whose withdrawal cannot be proven must not be consumable.
        updated.update(status="publishing", complete=False,
                       incomplete_reason=f"{reason}:revoke_verify_failed")
        problems += _persist_row_entry(root, stage, attempt, updated)
    return updated, problems, summary


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
    artifacts = out.get("artifacts")
    if isinstance(artifacts, dict):
        hashed = {}
        for rel, want in artifacts.items():
            p = Path(rel)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            hashed[rel] = sha256_file(p) if p.is_file() else None
        out["artifact_hashes"] = hashed
    return out


def finish_attempt(root: Path, stage: str, attempt: str, t0: float,
                   budget: dict, row: dict, *, status: str = "done",
                   extra: dict | None = None) -> SealResult:
    """pending -> publishing -> seal with an affordable, checked tail.

    The final terminal content is written under ``publishing`` first.  A
    consumer must refuse that state; only the post-write deadline/cost check
    may flip it to a terminal row.  If the check cannot prove coverage, the
    attempt stays ``publishing/complete=false`` and the call returns failure.
    """
    extra = dict(extra or {})
    attempt_dir = stage_dir(root, stage) / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    try:
        prior_rows = [r for r in load_index(root)
                      if r.get("stage") == stage and r.get("attempt") == attempt]
        prior_registration = (prior_rows[0].get("execution_registration_sha256")
                              if prior_rows else None)
    except (OSError, ValueError, TypeError):
        prior_registration = None
    row = {**row, "stage": stage, "attempt": attempt,
           "accounting_schema": ACCOUNTING_SCHEMA, "cohort_id": COHORT_ID,
           "execution_registration_sha256": row.get("execution_registration_sha256")
           or prior_registration}
    if extra.get("call_balance") is not None:
        row["call_balance"] = extra["call_balance"]
    row = _report_hashes(row)
    writes: list[float] = []
    problems: list[str] = []

    def timed(fn, *args):
        start = time.perf_counter()
        fn(*args)
        writes.append(time.perf_counter() - start)

    def publish(rw: dict) -> dict:
        sampled = max(0.0, time.perf_counter() - t0)
        rw = {**rw, "status": "publishing",
              "measured_wall_s": sampled, "sampled_wall_s": sampled}
        timed(upsert_index, root, stage, attempt, rw)
        timed(upsert_cost, root, stage, attempt, rw["measured_wall_s"],
              {**extra, "complete": False,
               "incomplete_reason": rw.get("incomplete_reason"),
               "sampled_wall_s": sampled,
               "cohort_id": COHORT_ID}, "publishing")
        return rw

    remaining = float(budget["remaining_s"])
    overrun = (time.perf_counter() - t0) >= remaining
    if not overrun:
        pending = {**row, "status": "running", "complete": False,
                   "measured_wall_s": max(0.0, time.perf_counter() - t0)}
        timed(upsert_index, root, stage, attempt, pending)
        timed(upsert_cost, root, stage, attempt, pending["measured_wall_s"],
              {**extra, "complete": False, "cohort_id": COHORT_ID}, "running")
        overrun = (time.perf_counter() - t0) >= remaining
    if overrun:
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_after_registration")
        status = "incomplete"
    row = publish(row)
    if not overrun and (time.perf_counter() - t0) >= remaining:
        overrun = True
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_after_publication")
        status = "incomplete"
        row = publish(row)

    def reserve() -> float:
        return max(SEAL_TAIL_RESERVE_S, 2.0 * max(writes, default=0.0))

    def terminal(rw: dict, st: str, cost: float, tail: float) -> dict:
        sampled = max(0.0, time.perf_counter() - t0)
        rw = {**rw, "status": st, "measured_wall_s": cost,
              "sampled_wall_s": sampled, "seal_tail_reserve_s": tail}
        timed(upsert_index, root, stage, attempt, rw)
        timed(upsert_cost, root, stage, attempt, cost,
              {**extra, "complete": bool(rw.get("complete", False)),
               "incomplete_reason": rw.get("incomplete_reason"),
               "sampled_wall_s": sampled,
               "seal_tail_reserve_s": tail, "cohort_id": COHORT_ID}, st)
        return rw

    def revoke(reason: str) -> dict:
        nonlocal overrun, row
        overrun = True
        tail = reserve()
        row.update(complete=False, incomplete_reason=reason)
        try:
            row = publish(row)
            return terminal(row, "incomplete",
                            time.perf_counter() - t0 + tail, tail)
        except (OSError, ValueError, TypeError):
            # The revoke itself failed: leave the non-consumable publishing
            # row and let the caller return failure.  Never publish done.
            row.update(status="publishing", complete=False,
                       incomplete_reason=f"{reason}:revoke_write_failed")
            try:
                row = publish(row)
            except (OSError, ValueError, TypeError):
                pass
            problems.append("revoke_write_failed")
            return row

    tail = reserve()
    if not overrun and (time.perf_counter() - t0) + tail >= remaining:
        overrun = True
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_before_seal")
        status = "incomplete"
        row = publish(row)
    try:
        row = terminal(row, status, time.perf_counter() - t0 + tail, tail)
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
        post = time.perf_counter() - t0
        if not overrun and post >= remaining:
            row = revoke("budget_exceeded_after_seal")
            continue
        if post <= float(row.get("measured_wall_s", 0.0)):
            break
        tail = reserve()
        if not overrun and post + tail >= remaining:
            row = revoke("seal_tail_reserve_insufficient")
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
    terminal = row.get("status") in ("done", "crashed", "timeout", "incomplete")
    return SealResult(ok, row, problems, terminal)


def stage_state(root: Path, stage: str) -> tuple[dict | None, list[str]]:
    """Consumer view of one stage: non-terminal states are not consumable."""
    problems: list[str] = []
    try:
        index, ledger, disk = _stage_rows(root, stage)
        if not index and not ledger and not disk:
            return None, ["no_attempt"]
        if not index:
            return None, ["no_index"]
        by = {r.get("attempt"): r for r in index}
        eb = {e.get("attempt"): e for e in ledger}
        if len(by) != len(index) or len(eb) != len(ledger):
            problems.append("duplicate_attempt")
        if set(disk) != set(by) or set(disk) != set(eb):
            problems.append("disk_index_ledger_mismatch")
        for row in index:
            if row.get("status") in ("running", "publishing"):
                problems.append(f"historical_attempt_unsealed:{row.get('attempt')}")
        for row in ledger:
            if row.get("status") in ("running", "publishing"):
                problems.append(f"historical_ledger_unsealed:{row.get('attempt')}")
        latest = index[-1]
        attempt = latest.get("attempt")
        e = eb.get(attempt)
        if e is None:
            problems.append("ledger_missing")
            return latest, problems
        if latest.get("status") == "running" or e.get("status") == "running":
            problems.append("attempt_running")
        elif latest.get("status") == "publishing" or e.get("status") == "publishing":
            problems.append("attempt_final_check_pending")
        elif latest.get("status") not in ("done", "crashed", "timeout", "incomplete"):
            problems.append("attempt_status_invalid")
        if latest.get("status") != e.get("status"):
            problems.append("index_ledger_status_mismatch")
        if latest.get("complete") is not e.get("complete"):
            problems.append("index_ledger_completion_mismatch")
        if latest.get("complete") is not True:
            problems.append("attempt_not_complete")
        if latest.get("accounting_schema") != ACCOUNTING_SCHEMA:
            problems.append("attempt_schema_missing")
        if latest.get("cohort_id") != COHORT_ID:
            problems.append("attempt_cohort_missing")
        measured = e.get("measured_wall_s")
        rounded = e.get("wall_s")
        index_measured = latest.get("measured_wall_s")
        if not (_nonneg(measured) and _nonneg(rounded)
                and _nonneg(index_measured)):
            problems.append("attempt_cost_invalid")
        else:
            if float(index_measured) != float(measured):
                problems.append("attempt_cost_mismatch")
            if float(rounded) != round(float(measured), 1):
                problems.append("attempt_cost_rounding_mismatch")
        for key in ("report", "summary"):
            rel = latest.get(key)
            if rel:
                p = Path(rel)
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                if not p.is_file() or sha256_file(p) != latest.get(f"{key}_sha256"):
                    problems.append(f"{key}_hash_mismatch")
        artifacts = latest.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            problems.append("artifacts_not_a_mapping")
        else:
            for rel, want in artifacts.items():
                p = Path(rel)
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                if not p.is_file() or sha256_file(p) != want:
                    problems.append(f"artifact_hash_mismatch:{rel}")
        return latest, problems
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return None, [f"stage_state_unreadable:{type(exc).__name__}"]


def unit_admission(root: Path, stage: str, attempt: str, unit_id: str,
                   *, scope: dict, registration_sha: str | None,
                   deadline: float, output_dir: Path) -> dict:
    """Parent-unit admission: active parent, matching scope, owned output."""
    problems: list[str] = []
    try:
        index = [r for r in load_index(root)
                 if r.get("stage") == stage and r.get("attempt") == attempt]
        ledger = [e for e in load_ledger(root)
                  if e.get("stage") == stage and e.get("attempt") == attempt]
    except (OSError, ValueError, TypeError):
        return {"passed": False, "problems": ["parent_unreadable"]}
    if len(index) != 1 or len(ledger) != 1:
        problems.append("parent_attempt_not_unique")
    else:
        row, e = index[0], ledger[0]
        if row.get("status") != "running" or e.get("status") != "running":
            problems.append("parent_not_running")
        if row.get("complete") is not False or e.get("complete") is not False:
            problems.append("parent_complete")
        if row.get("cohort_id") != COHORT_ID or e.get("cohort_id") != COHORT_ID:
            problems.append("parent_cohort_mismatch")
        if row.get("accounting_schema") != ACCOUNTING_SCHEMA:
            problems.append("parent_schema_missing")
        if registration_sha is not None and row.get("execution_registration_sha256") != registration_sha:
            problems.append("parent_registration_mismatch")
        if row.get("scope") != scope:
            problems.append("parent_scope_mismatch")
        parent_deadline = row.get("deadline_perf_counter")
        if not _num(parent_deadline) or float(deadline) > float(parent_deadline) + 1e-9:
            problems.append("unit_deadline_exceeds_parent")
    expected = stage_dir(root, stage) / attempt / "units" / unit_id
    try:
        output_dir = Path(output_dir).resolve()
        expected = expected.resolve()
        if output_dir != expected:
            problems.append("unit_output_outside_parent")
    except (OSError, ValueError):
        problems.append("unit_output_unreadable")
    return {"passed": not problems, "problems": problems,
            "expected_output": path_for_record(expected)}

# ---------------------------------------------------------------------------
# tolerance contract and comparison rules
# ---------------------------------------------------------------------------


def _deep_copy_tolerances() -> dict:
    import copy

    return copy.deepcopy(DEFAULT_TOLERANCES)


def _apply_observed_tolerances(observed: Mapping[str, Any]) -> dict:
    """The only admitted derivation from synthetic observations to thresholds."""
    tolerances = _deep_copy_tolerances()
    for key, value in observed.items():
        if key not in tolerances or not isinstance(value, Mapping):
            raise ValueError(f"calibration observed an unknown tolerance key: {key}")
        atol = max(float(tolerances[key]["atol"]),
                   4.0 * float(value.get("atol", 0.0)))
        rtol = max(float(tolerances[key]["rtol"]),
                   4.0 * float(value.get("rtol", 0.0)))
        if not (math.isfinite(atol) and math.isfinite(rtol)):
            raise ValueError(f"{key}: non-finite calibrated tolerance")
        tolerances[key]["atol"] = atol
        tolerances[key]["rtol"] = rtol
    return tolerances


def build_tolerance_contract(*, observed: Mapping[str, Any] | None = None,
                             evidence_sha256: str | None = None) -> dict:
    """Freeze the comparison domain before real values are opened.

    ``observed`` is only supplied by the pure synthetic calibration.  The
    resulting thresholds are never inferred from a real result.  A missing
    synthetic calibration hash keeps the contract non-consumable for the real
    entry.
    """
    tolerances = (_apply_observed_tolerances(observed) if observed
                  else _deep_copy_tolerances())
    return {
        "schema": "w1_real_objective_tolerance_contract_v1",
        "audit_schema": AUDIT_SCHEMA,
        "mode": "synthetic_calibrated" if observed or evidence_sha256 else "synthetic_default",
        "calibrated": True,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evidence_sha256": evidence_sha256,
        "synthetic_only": True,
        "tolerances": tolerances,
        "term_overrides": {k: dict(v) for k, v in TERM_FLOAT32_OVERRIDES.items()},
        "rules": {
            "order": ["structure", "dtype", "non_empty", "shape", "domain",
                      "finite", "tolerance"],
            "scalar_formula": "abs(production-independent) <= atol + rtol*abs(independent)",
            "gradient_formula": "max_abs_diff <= atol + rtol*max(independent_norm, scale_floor)",
            "bool_is_not_numeric": True,
            "counts_are_integers": True,
            "near_zero_scale_floor": True,
        },
    }


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")).encode())


def tolerance_contract_problems(contract: Mapping[str, Any] | None) -> list[str]:
    """Validate not just fields but the entire frozen derivation chain.

    A calibrated contract must carry the synthetic evidence that produced it,
    its self-hash must match, the thresholds must equal the documented 4x
    derivation from that evidence, and a fresh deterministic calibration with
    the recorded seed must reproduce both the evidence and the thresholds.
    Replacing evidence+hash while widening a tolerance is therefore rejected.
    """
    problems: list[str] = []
    if not isinstance(contract, Mapping):
        return ["tolerance_contract_not_a_mapping"]
    if contract.get("schema") != "w1_real_objective_tolerance_contract_v1":
        problems.append("tolerance_contract_schema")
    if contract.get("audit_schema") != AUDIT_SCHEMA:
        problems.append("tolerance_contract_audit_schema")
    if contract.get("calibrated") is not True:
        problems.append("tolerance_contract_not_calibrated")
    if contract.get("synthetic_only") is not True:
        problems.append("tolerance_contract_not_synthetic")
    mode = contract.get("mode")
    if mode not in ("synthetic_default", "synthetic_calibrated"):
        problems.append("tolerance_contract_mode")
    tol = contract.get("tolerances")
    if not isinstance(tol, Mapping):
        return problems + ["tolerance_contract_tolerances"]
    required = set(DEFAULT_TOLERANCES)
    if set(tol) != required:
        problems.append(f"tolerance_contract_keys:{sorted(set(tol) ^ required)}")
    for key in required:
        rec = tol.get(key)
        if not isinstance(rec, Mapping):
            problems.append(f"tolerance_record:{key}")
            continue
        if key == "rotation_domain":
            continue
        for field_name in ("atol", "rtol", "scale_floor"):
            value = rec.get(field_name)
            if not _num(value) or float(value) < 0.0:
                problems.append(f"tolerance_value:{key}:{field_name}")
    rot = tol.get("rotation_domain") or {}
    for field_name in ("min_angle_rad", "min_sin_theta"):
        value = rot.get(field_name)
        if not _num(value) or float(value) <= 0.0:
            problems.append(f"tolerance_value:rotation_domain:{field_name}")
    overrides = contract.get("term_overrides")
    if not isinstance(overrides, Mapping):
        problems.append("term_overrides_missing")
    else:
        for key, rec in overrides.items():
            if not isinstance(rec, Mapping):
                problems.append(f"term_override:{key}")
                continue
            for field_name in ("atol", "rtol"):
                value = rec.get(field_name)
                if not _num(value) or float(value) < 0.0:
                    problems.append(f"term_override:{key}:{field_name}")
        if _jsonable(overrides) != _jsonable(TERM_FLOAT32_OVERRIDES):
            problems.append("term_overrides_not_frozen")
    if problems:
        return problems

    evidence_sha = contract.get("evidence_sha256")
    calibration = contract.get("calibration")
    if mode == "synthetic_default":
        if evidence_sha is not None:
            problems.append("default_contract_has_evidence")
        if calibration not in (None, {}):
            problems.append("default_contract_has_calibration")
        if _jsonable(tol) != _jsonable(DEFAULT_TOLERANCES):
            problems.append("default_tolerances_changed")
        return problems

    # mode == "synthetic_calibrated"
    if not isinstance(evidence_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence_sha):
        problems.append("calibration_evidence_hash_missing")
    if not isinstance(calibration, Mapping):
        problems.append("calibration_evidence_missing")
        return problems
    if contract.get("calibration_sha256") != evidence_sha:
        problems.append("calibration_sha_mismatch")
    if _canonical_sha256(calibration) != evidence_sha:
        problems.append("calibration_evidence_hash_mismatch")
    if calibration.get("schema") != "w1_real_objective_synthetic_calibration_v1":
        problems.append("calibration_schema")
    if calibration.get("real_dataset_loaded") is not False:
        problems.append("calibration_real_dataset")
    if calibration.get("real_checkpoint_loaded") is not False:
        problems.append("calibration_real_checkpoint")
    if calibration.get("real_model_calls") != 0:
        problems.append("calibration_real_model_calls")
    observed = calibration.get("observed")
    if not isinstance(observed, Mapping) or not observed:
        problems.append("calibration_observed_missing")
    else:
        try:
            derived = _apply_observed_tolerances(observed)
        except (TypeError, ValueError):
            problems.append("calibration_observed_invalid")
        else:
            if _jsonable(derived) != _jsonable(tol):
                problems.append("calibration_threshold_derivation_mismatch")
    seed = calibration.get("seed")
    if not _count(seed):
        problems.append("calibration_seed_invalid")
    else:
        try:
            fresh = synthetic_calibration(seed=int(seed))
        except Exception as exc:  # noqa: BLE001 -- evidence must be reproducible
            problems.append(f"calibration_recompute_failed:{type(exc).__name__}")
        else:
            if fresh.get("calibration_sha256") != evidence_sha:
                problems.append("calibration_evidence_not_reproducible")
            if _jsonable(fresh.get("calibration")) != _jsonable(calibration):
                problems.append("calibration_values_not_reproducible")
            if _jsonable(fresh.get("tolerances")) != _jsonable(tol):
                problems.append("calibration_contract_not_reproducible")
    return problems


def compare_scalar(name: str, production: Any, independent: Any,
                   tolerance: Mapping[str, Any], *, units: str = "",
                   evidence: Mapping[str, Any] | None = None) -> dict:
    """Typed finite scalar comparison; bool/NaN/Inf never pass."""
    problems: list[str] = []
    if isinstance(production, (bool, np.bool_)) or isinstance(independent, (bool, np.bool_)):
        problems.append("bool_type")
    if not _num(production):
        problems.append("production_not_finite")
    if not _num(independent):
        problems.append("independent_not_finite")
    if problems:
        return {"status": "incomplete", "problems": problems,
                "production": _jsonable(production),
                "independent": _jsonable(independent),
                "within_tolerance": False}
    p, i = float(production), float(independent)
    atol, rtol = float(tolerance["atol"]), float(tolerance["rtol"])
    err = abs(p - i)
    limit = atol + rtol * abs(i)
    ok = err <= limit
    return {
        "status": "within_tolerance" if ok else "mismatch",
        "production": p, "independent": i, "abs_error": err,
        "limit": limit, "within_tolerance": bool(ok),
        "relative_error": (err / abs(i)) if abs(i) > 0.0 else None,
        "units": units, **(dict(evidence or {})),
    }


def compare_array(name: str, production: Any, independent: Any,
                  tolerance: Mapping[str, Any], *, units: str = "") -> dict:
    """Elementwise finite comparison for encoded arrays.

    Unlike the gradient norm rule, each element must satisfy the frozen
    absolute/relative bound; a large total norm cannot hide a small element.
    """
    problems: list[str] = []
    p = np.asarray(production)
    i = np.asarray(independent)
    if p.shape != i.shape:
        return {"status": "incomplete", "problems": ["shape_mismatch"],
                "within_tolerance": False}
    if not np.issubdtype(p.dtype, np.number) or not np.issubdtype(i.dtype, np.number):
        return {"status": "incomplete", "problems": ["non_numeric"],
                "within_tolerance": False}
    p64 = p.astype(np.float64, copy=False)
    i64 = i.astype(np.float64, copy=False)
    if not (np.isfinite(p64).all() and np.isfinite(i64).all()):
        return {"status": "incomplete", "problems": ["nonfinite"],
                "within_tolerance": False}
    atol, rtol = float(tolerance["atol"]), float(tolerance["rtol"])
    err = np.abs(p64 - i64)
    limit = atol + rtol * np.abs(i64)
    bad = err > limit
    max_abs = float(err.max()) if err.size else 0.0
    ok = not bool(bad.any())
    first_bad = None
    if bad.any():
        idx = tuple(int(x) for x in np.argwhere(bad)[0])
        first_bad = {"index": list(idx), "production": float(p64[idx]),
                     "independent": float(i64[idx]), "abs_error": float(err[idx]),
                     "limit": float(limit[idx])}
    return {
        "status": "within_tolerance" if ok else "mismatch",
        "production_norm": float(np.linalg.norm(p64)),
        "independent_norm": float(np.linalg.norm(i64)),
        "max_abs_diff": max_abs,
        "bad_element_count": int(bad.sum()),
        "first_bad_element": first_bad,
        "within_tolerance": bool(ok),
        "production_dtype": str(p.dtype), "independent_dtype": str(i.dtype),
        "shape": list(p.shape), "units": units,
    }


def compare_gradient(name: str, production: torch.Tensor,
                     independent: torch.Tensor, tolerance: Mapping[str, Any]) -> dict:
    """Batched output-gradient comparison with a near-zero scale floor."""
    problems: list[str] = []
    if torch.is_tensor(production):
        if production.dtype == torch.bool or production.is_complex():
            problems.append("production_dtype")
    else:
        problems.append("production_not_tensor")
    if torch.is_tensor(independent):
        if independent.dtype == torch.bool or independent.is_complex():
            problems.append("independent_dtype")
    else:
        problems.append("independent_not_tensor")
    if not problems and production.shape != independent.shape:
        problems.append("shape_mismatch")
    if not problems and (not bool(torch.isfinite(production).all())
                         or not bool(torch.isfinite(independent).all())):
        problems.append("nonfinite")
    if problems:
        return {"status": "incomplete", "problems": problems,
                "within_tolerance": False}
    p = production.detach().double().cpu()
    i = independent.detach().double().cpu()
    diff = (p - i).abs()
    norm_i = float(torch.linalg.vector_norm(i))
    max_abs = float(diff.max()) if diff.numel() else 0.0
    atol, rtol = float(tolerance["atol"]), float(tolerance["rtol"])
    floor = float(tolerance.get("scale_floor", 0.0))
    scale = max(norm_i, floor)
    limit = atol + rtol * scale
    ok = max_abs <= limit
    return {
        "status": "within_tolerance" if ok else "mismatch",
        "production_norm": float(torch.linalg.vector_norm(p)),
        "independent_norm": norm_i,
        "max_abs_diff": max_abs,
        "scale": scale,
        "limit": limit,
        "relative_error": (max_abs / scale) if scale > 0.0 else None,
        "within_tolerance": bool(ok),
        "production_dtype": str(production.dtype).replace("torch.", ""),
        "independent_dtype": str(independent.dtype).replace("torch.", ""),
        "shape": list(production.shape),
    }


# ---------------------------------------------------------------------------
# independent formulas (separate from the audited production implementation)
# ---------------------------------------------------------------------------


def _quat_to_mat_np_independent(q: np.ndarray) -> np.ndarray:
    """wxyz -> rotation matrix, explicit independent NumPy formula."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("quaternion must have last dimension 4")
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-15)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                  2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                  2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                  1 - 2 * (x * x + y * y)], -1),
    ], -2)


def independent_encode(root: np.ndarray, q: np.ndarray, wfc: np.ndarray,
                       lo: np.ndarray, hi: np.ndarray, mean: np.ndarray,
                       std: np.ndarray) -> np.ndarray:
    """Independent CPU/NumPy float64 re-encoding of the 38-dim state.

    Does not call :class:`ireset.model.tokens.StateTokenizer` or the project
    rotation adapter.  The input is one window (or a batch of windows).
    """
    root = np.asarray(root, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    wfc = np.asarray(wfc, dtype=np.float64)
    if root.shape[-1] != 7 or q.shape[-1] != 29 or wfc.shape[-2:] != (4, 4):
        raise ValueError("independent_encode shape mismatch")
    Rz, t = wfc[..., :3, :3], wfc[..., :3, 3]
    delta = root[..., 0:3] - np.expand_dims(t, -2)
    Rz_T = np.swapaxes(Rz, -1, -2)
    # broadcast the per-window matrix over the frame axis before applying it
    Rz_T_b = np.broadcast_to(np.expand_dims(Rz_T, -3),
                             delta.shape[:-1] + (3, 3))
    pos_clip = np.einsum("...ij,...j->...i", Rz_T_b, delta)
    pos_n = (pos_clip - mean) / std
    R_w = _quat_to_mat_np_independent(root[..., 3:7])
    Rz_T_r = np.broadcast_to(np.expand_dims(Rz_T, -3),
                             R_w.shape[:-2] + (3, 3))
    R_c = np.einsum("...ij,...jk->...ik", Rz_T_r, R_w)
    r6 = np.concatenate([R_c[..., :, 0], R_c[..., :, 1]], axis=-1)
    u = 2.0 * (q - lo) / (hi - lo) - 1.0
    return np.concatenate([pos_n, r6, u], axis=-1)


def independent_rotation(theta2: torch.Tensor, domain_ok: torch.Tensor,
                         min_angle: float, min_sin_theta: float) -> dict:
    return {"theta2": theta2, "domain_ok": domain_ok,
            "min_angle_rad": min_angle, "min_sin_theta": min_sin_theta}


def _independent_r6_to_mat(r6: torch.Tensor) -> torch.Tensor:
    """Explicit float64 Gram-Schmidt; separate from the audited decoder."""
    a = r6[..., 0:3]
    b = r6[..., 3:6]
    denom = (a * a).sum(-1, keepdim=True).clamp_min(1e-30)
    b_orth = b - (a * b).sum(-1, keepdim=True) * a / denom
    a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    b_n = b_orth / b_orth.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    c_n = torch.linalg.cross(a_n, b_n, dim=-1)
    c_n = c_n / c_n.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.stack([a_n, b_n, c_n], dim=-1)


def _independent_geodesic(pred: torch.Tensor, target: torch.Tensor,
                          min_angle: float,
                          min_sin_theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """theta^2 from an independent quaternion/trace path.

    The production path calls ``so3_log_torch``.  This implementation derives
    ``sin(theta/2)`` from the relative rotation matrix and evaluates
    ``theta = 2 asin(sin(theta/2))``; it does not share the audited SO(3) log
    implementation.  Near pi the trace branch is ill-conditioned and the
    domain mask marks the element as outside the independent reference.
    """
    R_p = _independent_r6_to_mat(pred[..., 3:9])
    R_t = _independent_r6_to_mat(target[..., 3:9])
    R = R_p @ R_t.transpose(-1, -2)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w2 = ((1.0 + tr) * 0.25).clamp_min(0.0)
    w = torch.sqrt(w2)
    denom = (4.0 * w).clamp_min(1e-12)
    x = (R[..., 2, 1] - R[..., 1, 2]) / denom
    y = (R[..., 0, 2] - R[..., 2, 0]) / denom
    z = (R[..., 1, 0] - R[..., 0, 1]) / denom
    v = torch.stack([x, y, z], dim=-1)
    v_norm = v.norm(dim=-1)
    sin_half = v_norm.clamp(max=1.0)
    theta = 2.0 * torch.asin(sin_half)
    sin_theta = 2.0 * w * sin_half
    domain_ok = (theta >= float(min_angle)) & (sin_theta >= float(min_sin_theta)) \
        & torch.isfinite(theta) & torch.isfinite(sin_theta)
    # Keep the graph finite outside the domain; the caller records a gap and
    # does not interpret the value as evidence.
    theta = torch.where(domain_ok, theta, torch.zeros_like(theta))
    return theta ** 2, domain_ok


def independent_term_elements(pred: torch.Tensor, target: torch.Tensor,
                              *, rotation_domain: Mapping[str, Any]) -> dict:
    """Five differentiable loss elements (float64) from the frozen formulas."""
    if pred.dtype != torch.float64 or target.dtype != torch.float64:
        raise ValueError("independent terms require float64")
    diff2 = (pred - target) ** 2
    geo2, domain = _independent_geodesic(
        pred, target,
        float(rotation_domain["min_angle_rad"]),
        float(rotation_domain["min_sin_theta"]),
    )
    d_pred = pred[:, 1:] - pred[:, :-1]
    d_tgt = target[:, 1:] - target[:, :-1]
    return {
        "pos": diff2[..., 0:3],
        "rot6d": diff2[..., 3:9],
        "joint": diff2[..., 9:38],
        "geodesic_rad2": geo2,
        "vel": (d_pred - d_tgt) ** 2,
        "geodesic_domain_ok": domain,
    }


def independent_scalars(elements: Mapping[str, torch.Tensor],
                        weights: LossWeights,
                        fk_elements: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    out = {k: elements[k].mean() for k in
           ("pos", "rot6d", "joint", "geodesic_rad2", "vel")}
    l5 = (float(weights.w_pos) * out["pos"]
          + float(weights.w_rot6d) * out["rot6d"]
          + float(weights.w_joint) * out["joint"]
          + float(weights.w_geodesic) * out["geodesic_rad2"]
          + float(weights.w_vel) * out["vel"])
    out["L5"] = l5
    if fk_elements is not None:
        out["fk_m2"] = fk_elements.mean()
        out["total"] = l5 + float(weights.w_fk) * out["fk_m2"]
    return out


def independent_vjp(pred: torch.Tensor, target: torch.Tensor,
                    fk_elements: torch.Tensor, weights: LossWeights,
                    *, rotation_domain: Mapping[str, Any]) -> tuple[torch.Tensor, dict]:
    """One batched VJP for the five independent terms plus their L5 sum."""
    pred = pred.detach().double().requires_grad_(True)
    elements = independent_term_elements(pred, target.detach().double(),
                                         rotation_domain=rotation_domain)
    scalars = independent_scalars(elements, weights, fk_elements.detach().double())
    outputs = torch.stack([scalars[k] for k in GRADIENT_SLICES])
    grad = torch.autograd.grad(
        outputs, pred,
        grad_outputs=torch.eye(len(GRADIENT_SLICES), dtype=pred.dtype,
                              device=pred.device),
        is_grads_batched=True, retain_graph=False,
    )[0]
    return grad, {
        "scalars": {k: float(v.detach()) for k, v in scalars.items()},
        "geodesic_domain_ok": bool(elements["geodesic_domain_ok"].all().item()),
        "geodesic_domain_bad_count": int((~elements["geodesic_domain_ok"]).sum().item()),
    }


def production_vjp(capture: Mapping[str, Any], pred: torch.Tensor) -> torch.Tensor:
    """One batched VJP through the ORIGINAL production graph.

    The five live term tensors and the captured pre-FK ``L5`` are used; the
    prediction is never detached and re-attached as a new leaf.
    """
    terms = capture.get("term_tensors")
    if not isinstance(terms, Mapping):
        raise ValueError("production observer did not capture term tensors")
    l5 = capture.get("L5")
    outputs = []
    for key in GRADIENT_SLICES:
        value = l5 if key == "L5" else terms.get(key)
        if not torch.is_tensor(value) or value.numel() != 1:
            raise ValueError(f"production term {key} missing or not scalar")
        outputs.append(value.reshape(()))
    grad = torch.autograd.grad(
        torch.stack(outputs), pred,
        grad_outputs=torch.eye(len(GRADIENT_SLICES), dtype=pred.dtype,
                              device=pred.device),
        is_grads_batched=True, retain_graph=False,
    )[0]
    return grad


# ---------------------------------------------------------------------------
# diffusion noise and input validation
# ---------------------------------------------------------------------------


def independent_beta_schedule(num_steps: int, schedule: str = "scaled_linear",
                              beta_start: float = 1e-4,
                              beta_end: float = 0.02) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if schedule == "scaled_linear":
        return np.linspace(math.sqrt(beta_start), math.sqrt(beta_end),
                           num_steps, dtype=np.float64) ** 2
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    raise ValueError(f"unsupported beta schedule {schedule!r}")


def independent_alpha_cumprod(num_steps: int, schedule: str = "scaled_linear",
                              beta_start: float = 1e-4,
                              beta_end: float = 0.02) -> np.ndarray:
    betas = independent_beta_schedule(num_steps, schedule, beta_start, beta_end)
    return np.cumprod(1.0 - betas)


def independent_x_t(x0: np.ndarray, noise: np.ndarray, t: np.ndarray,
                    alpha_cumprod: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    t = np.asarray(t, dtype=np.int64)
    if x0.shape != noise.shape or x0.shape[0] != t.shape[0]:
        raise ValueError("x_t shape mismatch")
    a = alpha_cumprod[t][:, None, None]
    return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * noise


def generate_noise(seed: int, shape: Sequence[int]) -> np.ndarray:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(tuple(int(x) for x in shape), generator=g,
                       dtype=torch.float32).numpy()


def _rotation6d_valid_np(x0: np.ndarray, atol: float = 1e-6) -> bool:
    a = np.asarray(x0[..., 3:6], dtype=np.float64)
    b = np.asarray(x0[..., 6:9], dtype=np.float64)
    denom = np.maximum((a * a).sum(-1, keepdims=True), 1e-30)
    orth = b - (a * b).sum(-1, keepdims=True) * a / denom
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(orth, axis=-1)
    nc = np.linalg.norm(np.cross(a, orth), axis=-1)
    return bool(np.isfinite(na).all() and np.isfinite(nb).all()
                and np.isfinite(nc).all() and (na > atol).all()
                and (nb > atol).all() and (nc > atol).all())


def validate_inputs(inputs: AuditInputs, *, expected_keys: Sequence[WindowKey] | None = None,
                    tolerance: Mapping[str, Any] | None = None) -> dict:
    """Structural/domain admission for the four-window batch."""
    problems: list[str] = []
    facts: dict[str, Any] = {}
    expected = list(expected_keys if expected_keys is not None else inputs.window_keys)
    try:
        if len(inputs.window_keys) != WINDOW_COUNT:
            problems.append(f"window_count:{len(inputs.window_keys)}")
        keys = [k.key for k in inputs.window_keys]
        if len(set(keys)) != len(keys):
            problems.append("duplicate_window_key")
        if keys != [k.key for k in expected]:
            problems.append("window_key_order_mismatch")
        if len(expected) != WINDOW_COUNT:
            problems.append("registered_window_count")
        for name, arr, shape in (
            ("root", inputs.root, (WINDOW_COUNT, WINDOW, 7)),
            ("q", inputs.q, (WINDOW_COUNT, WINDOW, 29)),
            ("wfc", inputs.wfc, (WINDOW_COUNT, 4, 4)),
            ("kps_obj", inputs.kps_obj, (WINDOW_COUNT, WINDOW, 52, 3)),
            ("kps_clip", inputs.kps_clip, (WINDOW_COUNT, WINDOW, 52, 3)),
            ("box_t_clip", inputs.box_t_clip, (WINDOW_COUNT, WINDOW, 3)),
            ("box_quat", inputs.box_quat, (WINDOW_COUNT, WINDOW, 4)),
            ("contact", inputs.contact, (WINDOW_COUNT, WINDOW, 4)),
            ("x0", inputs.x0, (WINDOW_COUNT, WINDOW, 38)),
            ("x0_independent", inputs.x0_independent, (WINDOW_COUNT, WINDOW, 38)),
            ("joint_lo", inputs.joint_lo, (29,)),
            ("joint_hi", inputs.joint_hi, (29,)),
            ("root_mean", inputs.root_mean, (3,)),
            ("root_std", inputs.root_std, (3,)),
            ("t", inputs.t, (WINDOW_COUNT,)),
            ("noise", inputs.noise, (WINDOW_COUNT, WINDOW, 38)),
        ):
            if tuple(np.shape(arr)) != shape:
                problems.append(f"{name}_shape:{np.shape(arr)}!={shape}")
            elif not np.isfinite(np.asarray(arr)).all():
                problems.append(f"{name}_nonfinite")
        if not len(problems):
            if not np.array_equal(np.asarray(inputs.t, dtype=np.int64),
                                  np.asarray(DIFFUSION_TIMESTEPS, dtype=np.int64)):
                problems.append("diffusion_timesteps")
            if not _rotation6d_valid_np(inputs.x0):
                problems.append("production_rotation6d_invalid")
            if not _rotation6d_valid_np(inputs.x0_independent):
                problems.append("independent_rotation6d_invalid")
            if (np.asarray(inputs.joint_hi) <= np.asarray(inputs.joint_lo)).any():
                problems.append("joint_range_invalid")
            if (np.asarray(inputs.root_std) <= 0).any():
                problems.append("root_std_invalid")
            root_n = np.linalg.norm(inputs.root[..., 3:7], axis=-1)
            if (root_n <= 1e-6).any():
                problems.append("root_quat_zero")
            box_n = np.linalg.norm(inputs.box_quat, axis=-1)
            if (box_n <= 1e-6).any():
                problems.append("box_quat_zero")
            q = inputs.q
            if ((q < inputs.joint_lo - 1e-6) | (q > inputs.joint_hi + 1e-6)).any():
                problems.append("raw_joint_limit")
            u = inputs.x0[..., 9:38]
            if ((u < -1.0 - 1e-6) | (u > 1.0 + 1e-6)).any():
                problems.append("normalized_joint_limit")
            if tolerance is not None:
                cmp = compare_array(
                    "encode", inputs.x0, inputs.x0_independent, tolerance,
                    units="normalized")
                facts["encoding"] = cmp
                if cmp.get("status") != "within_tolerance":
                    problems.append("encoding_mismatch")
        facts.update({
            "shape": list(np.shape(inputs.x0)),
            "dtype": str(np.asarray(inputs.x0).dtype),
            "windows": [k.as_dict() for k in inputs.window_keys],
            "t": [int(x) for x in inputs.t],
            "root_quat_norm_min": float(np.linalg.norm(inputs.root[..., 3:7], axis=-1).min()),
            "box_quat_norm_min": float(np.linalg.norm(inputs.box_quat, axis=-1).min()),
            "raw_joint_limit_max_violation": float(np.maximum(
                inputs.joint_lo - inputs.q, inputs.q - inputs.joint_hi).max()),
        })
    except (ValueError, TypeError, KeyError) as exc:
        problems.append(f"input_unreadable:{type(exc).__name__}:{exc}")
    return {"passed": not problems, "problems": problems, "facts": facts}


def validate_prediction(pred: torch.Tensor, *, dtype: torch.dtype | None = None) -> list[str]:
    problems: list[str] = []
    if not torch.is_tensor(pred):
        return ["prediction_not_tensor"]
    if pred.ndim != 3 or tuple(pred.shape) != (WINDOW_COUNT, WINDOW, 38):
        problems.append(f"prediction_shape:{tuple(pred.shape)}")
    if dtype is not None and pred.dtype != dtype:
        problems.append(f"prediction_dtype:{pred.dtype}!={dtype}")
    if not bool(torch.isfinite(pred).all()):
        problems.append("prediction_nonfinite")
    if pred.numel() and not _rotation6d_valid_np(pred.detach().double().cpu().numpy()):
        problems.append("prediction_rotation6d_invalid")
    return problems


# ---------------------------------------------------------------------------
# call budget and production capture
# ---------------------------------------------------------------------------


class AuditCallBudgetExceeded(RuntimeError):
    def __init__(self, kind: str, count: int, cap: int) -> None:
        super().__init__(f"{kind} call cap exceeded: {count} >= {cap}")
        self.kind, self.count, self.cap = kind, count, cap


class AuditCallBudget:
    """Cumulative call counter shared by R0/P/F; failures still consume.

    ``prior_counts`` seeds calls already consumed by earlier attempts/runs of
    the same cohort.  Caps are cohort-wide, so a retry never receives a fresh
    allowance; missing or malformed history is refused before construction.
    """

    def __init__(self, caps: Mapping[str, int] | None = None,
                 prior_counts: Mapping[str, Any] | None = None) -> None:
        self.caps = {k: int(v) for k, v in (caps or CALL_CAPS).items()}
        seed = {k: 0 for k in self.caps}
        if prior_counts is not None:
            if set(prior_counts) != set(self.caps):
                raise ValueError("prior call counts do not match the frozen cap keys")
            for kind, value in prior_counts.items():
                if not _count(value):
                    raise ValueError(f"prior call count invalid: {kind}={value!r}")
                seed[kind] = int(value)
        self.prior_counts = dict(seed)
        self.counts = dict(seed)
        self.facts: list[dict] = []

    def consume(self, kind: str, **facts: Any) -> int:
        if kind not in self.caps:
            raise ValueError(f"unknown call kind {kind}")
        if self.counts[kind] >= self.caps[kind]:
            raise AuditCallBudgetExceeded(kind, self.counts[kind], self.caps[kind])
        self.counts[kind] += 1
        self.facts.append({"kind": kind, "index": self.counts[kind], **facts})
        return self.counts[kind]

    def snapshot(self) -> dict:
        return {
            "schema": "w1_real_objective_call_budget_v1",
            "caps": dict(self.caps),
            "counts": dict(self.counts),
            "prior_counts": dict(self.prior_counts),
            "remaining": {k: self.caps[k] - self.counts[k] for k in self.caps},
            "facts": list(self.facts),
            "within_caps": all(self.counts[k] <= self.caps[k] for k in self.caps),
        }


def _rng_state() -> dict:
    """Capture restorable process RNG states (not just hashes)."""
    import base64
    import random

    py_version, py_keys, py_gauss = random.getstate()
    np_state = np.random.get_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [
            base64.b64encode(state.cpu().numpy().tobytes()).decode("ascii")
            for state in torch.cuda.get_rng_state_all()
        ]
    return {
        "python": {"version": int(py_version),
                   "keys": [int(x) for x in py_keys],
                   "gauss_next": None if py_gauss is None else float(py_gauss)},
        "numpy": {"name": str(np_state[0]),
                  "keys": [int(x) for x in np_state[1]],
                  "pos": int(np_state[2]), "has_gauss": int(np_state[3]),
                  "cached_gaussian": float(np_state[4])},
        "torch_cpu": base64.b64encode(
            torch.get_rng_state().cpu().numpy().tobytes()).decode("ascii"),
        "torch_cuda": cuda_states,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _rng_state_problems(state: Any) -> list[str]:
    """Structural validation before a replay restores any global RNG state."""
    problems: list[str] = []
    if not isinstance(state, Mapping):
        return ["rng_state_not_a_mapping"]
    py = state.get("python")
    if (not isinstance(py, Mapping)
            or not _count(py.get("version"))
            or not isinstance(py.get("keys"), list)
            or not py["keys"]
            or not all(_count(x) for x in py["keys"])
            or not (py.get("gauss_next") is None or _num(py.get("gauss_next")))):
        problems.append("rng_python_invalid")
    np_state = state.get("numpy")
    if (not isinstance(np_state, Mapping)
            or not isinstance(np_state.get("name"), str)
            or not isinstance(np_state.get("keys"), list)
            or not np_state["keys"]
            or not all(_count(x) for x in np_state["keys"])
            or not _count(np_state.get("pos"))
            or not _count(np_state.get("has_gauss"))
            or not _num(np_state.get("cached_gaussian"))):
        problems.append("rng_numpy_invalid")
    import base64

    def decoded_length(value: Any) -> int | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return len(base64.b64decode(value, validate=True))
        except (ValueError, TypeError):
            return None

    cpu = state.get("torch_cpu")
    expected_len = int(torch.get_rng_state().numel())
    if decoded_length(cpu) != expected_len:
        problems.append("rng_torch_cpu_invalid")
    cuda = state.get("torch_cuda")
    if state.get("cuda_available") is True:
        if (not isinstance(cuda, list) or not cuda
                or not all(decoded_length(x) is not None for x in cuda)):
            problems.append("rng_torch_cuda_invalid")
        elif torch.cuda.is_available():
            cuda_len = {int(s.numel()) for s in torch.cuda.get_rng_state_all()}
            if any(decoded_length(x) not in cuda_len for x in cuda):
                problems.append("rng_torch_cuda_invalid")
    elif cuda is not None:
        problems.append("rng_torch_cuda_unexpected")
    if state.get("cuda_available") is not bool(torch.cuda.is_available()):
        problems.append("rng_cuda_availability_changed")
    return problems


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore the exact P-before state; invalid states raise ValueError."""
    import base64
    import random

    problems = _rng_state_problems(state)
    if problems:
        raise ValueError(f"invalid RNG state: {problems}")
    py = state["python"]
    random.setstate((int(py["version"]), tuple(int(x) for x in py["keys"]),
                     py["gauss_next"]))
    np_state = state["numpy"]
    np.random.set_state((
        np_state["name"],
        np.asarray(np_state["keys"], dtype=np.uint32),
        int(np_state["pos"]), int(np_state["has_gauss"]),
        float(np_state["cached_gaussian"]),
    ))
    cpu = torch.frombuffer(
        bytearray(base64.b64decode(state["torch_cpu"], validate=True)),
        dtype=torch.uint8).clone()
    torch.set_rng_state(cpu)
    if torch.cuda.is_available():
        for index, encoded in enumerate(state["torch_cuda"]):
            cuda = torch.frombuffer(
                bytearray(base64.b64decode(encoded, validate=True)),
                dtype=torch.uint8).clone()
            torch.cuda.set_rng_state(cuda, device=index)


def _rng_facts(state: Mapping[str, Any] | None = None) -> dict:
    """Hash the restorable state captured by :func:`_rng_state`."""
    if state is None:
        state = _rng_state()
    return {
        "python": sha256_bytes(repr(state["python"]).encode()),
        "numpy": sha256_bytes(
            np.asarray(state["numpy"]["keys"], dtype=np.uint32).tobytes()
            + repr((state["numpy"]["name"], state["numpy"]["pos"],
                    state["numpy"]["has_gauss"],
                    state["numpy"]["cached_gaussian"])).encode()),
        "torch_cpu": sha256_bytes(
            __import__("base64").b64decode(state["torch_cpu"])),
        "torch_cuda": (
            sha256_bytes(b"".join(
                __import__("base64").b64decode(x)
                for x in state["torch_cuda"]))
            if state.get("torch_cuda") else None),
        "cuda_available": bool(state.get("cuda_available")),
    }


def _forward_production(bundle: ModelBundle, inputs: AuditInputs,
                        calls: AuditCallBudget, *, device: str,
                        dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    batch = inputs.condition_batch(device=device, dtype=dtype)
    if bundle.model_type == "diffusion":
        t = torch.as_tensor(inputs.t, dtype=torch.int64, device=device)
        noise = torch.as_tensor(inputs.noise, dtype=dtype, device=device)
        calls.consume("conditioner", model_type=bundle.model_type,
                      phase="production", shape=list(batch["x0"].shape))
        memory = bundle.conditioner(batch)
        calls.consume("model", model_type=bundle.model_type,
                      phase="production", input="x_t")
        # Production noise wiring: the real DiffusionG1.add_noise (DDPMScheduler).
        x_t = bundle.model.add_noise(batch["x0"], noise, t)
        pred = bundle.model(x_t, t, memory)
        return pred, x_t, batch
    calls.consume("conditioner", model_type=bundle.model_type,
                  phase="production", shape=list(batch["x0"].shape))
    memory = bundle.conditioner(batch)
    calls.consume("model", model_type=bundle.model_type,
                  phase="production", input="memory")
    pred = bundle.model(memory, frames=inputs.F)
    return pred, None, batch


def _loss_call(loss_fn, pred: torch.Tensor, batch: dict) -> tuple[dict, Any, dict]:
    capture: dict = {}
    loss, parts = loss_fn(pred, batch, observer=capture)
    if not isinstance(parts, Mapping):
        raise ValueError("loss parts are not a mapping")
    if not isinstance(capture.get("term_tensors"), Mapping):
        raise ValueError("loss observer did not capture term tensors")
    return capture, loss, {"loss": loss, "parts": parts}


def _element_mean(value: torch.Tensor, w: int) -> float:
    if value.ndim < 2:
        raise ValueError("term element must have batch and frame axes")
    return float(value[w].detach().double().mean().cpu())


def _per_window_scalars(elements: Mapping[str, torch.Tensor],
                        weights: LossWeights) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    B = int(elements["pos"].shape[0])
    for w in range(B):
        rec = {k: _element_mean(elements[k], w) for k in
               ("pos", "rot6d", "joint", "geodesic_rad2", "vel")}
        rec["L5"] = (float(weights.w_pos) * rec["pos"]
                     + float(weights.w_rot6d) * rec["rot6d"]
                     + float(weights.w_joint) * rec["joint"]
                     + float(weights.w_geodesic) * rec["geodesic_rad2"]
                     + float(weights.w_vel) * rec["vel"])
        if "fk_elements" in elements:
            rec["fk_m2"] = _element_mean(elements["fk_elements"], w)
            rec["total"] = rec["L5"] + float(weights.w_fk) * rec["fk_m2"]
        out[w] = rec
    return out


def _batch_scalars_from_tensors(capture: Mapping[str, Any]) -> dict[str, float]:
    terms = capture["term_tensors"]
    out = {k: float(terms[k].detach().double().cpu().reshape(())) for k in TERMS}
    out["L5"] = float(capture["L5"].detach().double().cpu().reshape(()))
    return out


def _term_denominator(term: str, *, B: int, F: int, fk_sites: int) -> int | None:
    if term == "pos":
        return B * F * 3
    if term == "rot6d":
        return B * F * 6
    if term == "joint":
        return B * F * 29
    if term == "geodesic_rad2":
        return B * F
    if term == "vel":
        return B * (F - 1) * 38
    if term == "fk_m2":
        return B * F * fk_sites
    return None


def _row_id(scope: str, model: str, *parts: Any) -> str:
    return "|".join([scope, model, *[str(p) for p in parts]])


def required_objective_row_keys(window_keys: Sequence[WindowKey]) -> list[str]:
    keys: list[str] = []
    for model in MODELS:
        for wk in window_keys:
            for term in (*TERMS, "L5"):
                keys.append(_row_id("window", model, wk.demo, wk.start, term))
        for term in (*TERMS, "L5"):
            keys.append(_row_id("batch", model, term))
        for term in ("L5_arithmetic", "total_arithmetic"):
            keys.append(_row_id("batch", model, term))
        for term in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2", "total"):
            keys.append(_row_id("batch", model, "window_mean_reconciliation", term))
    return keys


def required_gradient_row_keys() -> list[str]:
    keys = []
    for model in MODELS:
        for term in GRADIENT_SLICES:
            keys.append(_row_id("grad", model, term))
        keys.append(_row_id("grad", model, "l5_weighted_sum"))
    return keys


def objective_array_keys_for(models: Sequence[str]) -> list[str]:
    base = [
        "raw_root", "raw_q", "raw_wfc", "kps_obj", "kps_clip",
        "box_t_clip", "box_quat", "contact", "x0", "x0_independent",
        "joint_lo", "joint_hi", "root_mean", "root_std", "t", "noise",
        "alpha_cumprod",
    ]
    for model in models:
        base += [f"prediction_{model}", f"term_elements_{model}_pos",
                 f"term_elements_{model}_rot6d", f"term_elements_{model}_joint",
                 f"term_elements_{model}_geodesic_rad2",
                 f"term_elements_{model}_vel", f"term_elements_{model}_fk",
                 f"fk_points_pred_{model}", f"fk_points_target_{model}"]
        if model == "diffusion":
            base.append("x_t_diffusion")
    return base


def objective_array_keys() -> list[str]:
    return objective_array_keys_for(MODELS)


def gradient_array_keys_for(models: Sequence[str]) -> list[str]:
    keys = []
    for model in models:
        keys += [f"production_grad_{model}", f"independent_grad_{model}"]
    return keys


def gradient_array_keys() -> list[str]:
    return gradient_array_keys_for(MODELS)


def _audit_weights() -> LossWeights:
    # The W1 config, including w_contact=0.  LossWeights()'s dataclass
    # default is 0.5 and would execute the contact branch.
    return LossWeights(w_pos=1.0, w_rot6d=1.0, w_joint=1.0, w_geodesic=0.5,
                       w_fk=2.0, w_contact=0.0, w_vel=0.5)


def production_objective_unit(bundle: ModelBundle, inputs: AuditInputs,
                              loss_fn, calls: AuditCallBudget, *,
                              device: str, dtype: torch.dtype,
                              tolerance: Mapping[str, Any]) -> dict:
    """One model unit: one production forward, one capture, one VJP."""
    before_state = _rng_state()
    before_rng = _rng_facts(before_state)
    pred, x_t, batch = _forward_production(bundle, inputs, calls,
                                           device=device, dtype=dtype)
    pred_problems = validate_prediction(pred, dtype=dtype)
    capture, loss, loss_meta = _loss_call(loss_fn, pred, batch)
    if capture.get("x0_pred") is not pred:
        raise ValueError("observer prediction is not the live production tensor")
    if bool(capture.get("contact_branch_executed", True)):
        raise ValueError("contact supervision branch executed at w_contact=0")
    weights = _audit_weights()
    if abs(float(getattr(loss_fn.w, "w_contact", 0.0))) > 0.0:
        raise ValueError("audit requires w_contact == 0")
    calls.consume("production_vjp", model_type=bundle.model_type,
                  phase="production", slices=list(GRADIENT_SLICES))
    prod_grad = production_vjp(capture, pred)

    independent_target = torch.as_tensor(inputs.x0_independent, dtype=torch.float64,
                                        device=device)
    fk_elements = capture.get("fk_elements")
    if not torch.is_tensor(fk_elements):
        raise ValueError("observer did not capture FK elements")
    calls.consume("independent_vjp", model_type=bundle.model_type,
                  phase="production", slices=list(GRADIENT_SLICES))
    ind_grad, ind_facts = independent_vjp(
        pred.detach().double(), independent_target,
        fk_elements.detach().double(), weights,
        rotation_domain=tolerance["rotation_domain"],
    )
    after_state = _rng_state()
    after_rng = _rng_facts(after_state)

    elements = {
        k: capture["term_elements"][k].detach().double().cpu()
        for k in ("pos", "rot6d", "joint", "geodesic_rad2", "vel")
    }
    elements["fk_elements"] = fk_elements.detach().double().cpu()
    prod_window = _per_window_scalars(elements, weights)
    ind_elements = independent_term_elements(
        pred.detach().double(), independent_target,
        rotation_domain=tolerance["rotation_domain"],
    )
    ind_scalars = independent_scalars(ind_elements, weights,
                                      fk_elements.detach().double())
    ind_window = _per_window_scalars({**ind_elements,
                                      "fk_elements": fk_elements.detach().double().cpu()},
                                     weights)
    prod_batch = _batch_scalars_from_tensors(capture)
    return {
        "model_type": bundle.model_type,
        "prediction": pred.detach().cpu().numpy(),
        "x_t": None if x_t is None else x_t.detach().cpu().numpy(),
        "fk_points_pred": capture["fk_points_pred"].detach().cpu().numpy(),
        "fk_points_target": capture["fk_points_target"].detach().cpu().numpy(),
        "elements": {k: v.numpy() for k, v in elements.items() if k != "fk_elements"},
        "fk_elements": elements["fk_elements"].numpy(),
        "production_window": prod_window,
        "independent_window": ind_window,
        "production_batch": prod_batch,
        "independent_batch": {k: float(v.detach().double().cpu())
                              for k, v in ind_scalars.items()},
        "production_grad": prod_grad.detach().cpu().numpy(),
        "independent_grad": ind_grad.detach().double().cpu().numpy(),
        "independent_facts": ind_facts,
        "prediction_problems": pred_problems,
        "loss_parts": {k: float(v) for k, v in loss_meta["parts"].items()},
        "rng_before": before_rng,
        "rng_after": after_rng,
        "rng_state_before": before_state,
        "rng_state_after": after_state,
        "contact_branch_executed": bool(capture.get("contact_branch_executed", False)),
        "fk_backend": getattr(getattr(loss_fn, "fk", None), "backend", "production"),
    }


def _scalar_tolerance(tol: Mapping[str, Any], dtype: torch.dtype) -> Mapping[str, Any]:
    return tol["tolerances"]["loss_float64" if dtype == torch.float64 else "loss_float32"]


def _gradient_tolerance(tol: Mapping[str, Any], dtype: torch.dtype) -> Mapping[str, Any]:
    return tol["tolerances"]["gradient_float64" if dtype == torch.float64 else "gradient_float32"]


def _term_scalar_tolerance(stol: Mapping[str, Any], term: str,
                           dtype: torch.dtype,
                           overrides: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if dtype == torch.float32:
        table = overrides if overrides is not None else TERM_FLOAT32_OVERRIDES
        if term in table:
            return {**stol, **table[term]}
    return stol


def build_objective_rows(units: Mapping[str, dict], inputs: AuditInputs,
                         tolerance: Mapping[str, Any], dtype: torch.dtype) -> list[dict]:
    rows: list[dict] = []
    stol = _scalar_tolerance(tolerance, dtype)
    overrides = tolerance.get("term_overrides") or TERM_FLOAT32_OVERRIDES
    for model in units:
        unit = units[model]
        prod_w = unit["production_window"]
        ind_w = unit["independent_window"]
        for wi, wk in enumerate(inputs.window_keys):
            for term in (*TERMS, "L5"):
                p = prod_w[wi].get(term)
                i = ind_w[wi].get(term)
                ttol = _term_scalar_tolerance(stol, term, dtype, overrides)
                cmp = compare_scalar(term, p, i, ttol, units=_term_units(term),
                                     evidence={"window": wk.key,
                                               "denominator": _term_denominator(
                                                   term, B=1, F=inputs.F,
                                                   fk_sites=unit["fk_points_pred"].shape[-2]
                                                   if term == "fk_m2" else 0),
                                               "aggregation_only": term == "fk_m2"})
                rows.append({
                    "id": _row_id("window", model, wk.demo, wk.start, term),
                    "scope": "window", "model": model, "window": wk.as_dict(),
                    "term": term, **cmp,
                    "denominator": _term_denominator(
                        term, B=1, F=inputs.F,
                        fk_sites=unit["fk_points_pred"].shape[-2]
                        if term == "fk_m2" else 0),
                    "weight": _term_weight(term),
                    "aggregation_only": term == "fk_m2",
                })
        for term in (*TERMS, "L5"):
            p = unit["production_batch"].get(term)
            i = unit["independent_batch"].get(term)
            ttol = _term_scalar_tolerance(stol, term, dtype, overrides)
            cmp = compare_scalar(term, p, i, ttol, units=_term_units(term),
                                 evidence={"denominator": _term_denominator(
                                     term, B=inputs.B, F=inputs.F,
                                     fk_sites=unit["fk_points_pred"].shape[-2]
                                     if term == "fk_m2" else 0)})
            rows.append({
                "id": _row_id("batch", model, term),
                "scope": "batch", "model": model, "term": term, **cmp,
                "denominator": _term_denominator(
                    term, B=inputs.B, F=inputs.F,
                    fk_sites=unit["fk_points_pred"].shape[-2] if term == "fk_m2" else 0),
                "weight": _term_weight(term),
                "aggregation_only": term == "fk_m2",
            })
        # L5 and total arithmetic consistency, from the same raw scalars.
        w = _audit_weights()
        l5_manual = (w.w_pos * unit["production_batch"]["pos"]
                     + w.w_rot6d * unit["production_batch"]["rot6d"]
                     + w.w_joint * unit["production_batch"]["joint"]
                     + w.w_geodesic * unit["production_batch"]["geodesic_rad2"]
                     + w.w_vel * unit["production_batch"]["vel"])
        total_manual = (l5_manual + w.w_fk * unit["production_batch"]["fk_m2"])
        for name, actual, expected in (
            ("L5_arithmetic", unit["production_batch"]["L5"], l5_manual),
            ("total_arithmetic", unit["production_batch"]["total"], total_manual),
        ):
            cmp = compare_scalar(name, actual, expected, stol, units="weighted")
            rows.append({"id": _row_id("batch", model, name),
                         "scope": "arithmetic", "model": model,
                         "term": name, **cmp})
        # Batch scalar must be the equal-weight mean of the four window rows
        # for every term with an equal per-window denominator.
        for term in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2", "total"):
            vals = [prod_w[i][term] for i in range(len(inputs.window_keys))]
            mean = sum(vals) / len(vals)
            cmp = compare_scalar(f"{term}_window_mean",
                                 unit["production_batch"][term], mean, stol,
                                 units=_term_units(term))
            rows.append({"id": _row_id("batch", model, "window_mean_reconciliation",
                                       term),
                         "scope": "reconciliation", "model": model,
                         "term": term, **cmp})
    return rows


def _apply_domain_gaps(rows: list[dict], grad_rows: list[dict],
                       bad_models: Mapping[str, int]) -> None:
    """Mark terms whose independent rotation reference is outside its domain."""
    affected_terms = {"geodesic_rad2", "L5", "total", "total_arithmetic"}
    for row in rows:
        model = str(row.get("model", ""))
        term = str(row.get("term", ""))
        if model in bad_models and term in affected_terms:
            row.update(status="incomplete", within_tolerance=False,
                       problems=["independent_rotation_domain"],
                       domain_bad_count=int(bad_models[model]))
    for row in grad_rows:
        model = str(row.get("model", ""))
        slice_name = str(row.get("slice", ""))
        if model in bad_models and slice_name in {
                "geodesic_rad2", "L5", "L5_weighted_sum"}:
            row.update(status="incomplete", within_tolerance=False,
                       problems=["independent_rotation_domain"],
                       domain_bad_count=int(bad_models[model]))


def _term_units(term: str) -> str:
    return {
        "pos": "standardized_position_sq",
        "rot6d": "rot6d_component_sq",
        "joint": "normalized_joint_sq",
        "geodesic_rad2": "rad^2",
        "vel": "normalized_velocity_sq",
        "fk_m2": "m^2",
        "total": "weighted",
        "L5": "weighted",
        "L5_arithmetic": "weighted",
        "total_arithmetic": "weighted",
        "window_mean_reconciliation": "same_as_term",
    }.get(term, "unknown")


def _term_weight(term: str) -> float | None:
    w = _audit_weights()
    return {
        "pos": w.w_pos, "rot6d": w.w_rot6d, "joint": w.w_joint,
        "geodesic_rad2": w.w_geodesic, "vel": w.w_vel, "fk_m2": w.w_fk,
        "total": None, "L5": None,
    }.get(term)


def build_gradient_rows(units: Mapping[str, dict], tolerance: Mapping[str, Any],
                        dtype: torch.dtype) -> list[dict]:
    rows: list[dict] = []
    gtol = _gradient_tolerance(tolerance, dtype)
    w = _audit_weights()
    weights_by_slice = [w.w_pos, w.w_rot6d, w.w_joint, w.w_geodesic,
                        w.w_vel, 1.0]
    for model in units:
        unit = units[model]
        p = torch.as_tensor(unit["production_grad"], dtype=torch.float64)
        i = torch.as_tensor(unit["independent_grad"], dtype=torch.float64)
        for si, name in enumerate(GRADIENT_SLICES):
            cmp = compare_gradient(name, p[si], i[si], gtol)
            rows.append({"id": _row_id("grad", model, name),
                         "scope": "gradient", "model": model, "slice": name,
                         **cmp, "aggregation_only": False})
        weighted = sum(weights_by_slice[k] * i[k] for k in range(5))
        cmp = compare_gradient("l5_weighted_sum", p[5], weighted, gtol)
        rows.append({"id": _row_id("grad", model, "l5_weighted_sum"),
                     "scope": "gradient_arithmetic", "model": model,
                     "slice": "L5_weighted_sum", **cmp,
                     "aggregation_only": True})
    return rows


def _rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError("row without id")
        if rid in out:
            raise ValueError(f"duplicate row id {rid}")
        out[rid] = dict(row)
    return out


def load_rows_checked(path: Path) -> tuple[list[dict], dict[str, dict], str | None]:
    """Read a rows file without crashing on malformed structure."""
    try:
        doc = read_json(path)
        rows = doc.get("rows") if isinstance(doc, Mapping) else None
        if not isinstance(rows, list):
            return [], {}, "rows_not_a_list"
        if not all(isinstance(r, Mapping) for r in rows):
            return [], {}, "row_not_a_mapping"
        return rows, _rows_by_id(rows), None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return [], {}, f"rows_unreadable:{type(exc).__name__}"


def row_multiset_problems(rows: Sequence[Mapping[str, Any]],
                          required: Sequence[str]) -> list[str]:
    """Exact-key multiset check: empty and deleted rows never pass."""
    ids = [r.get("id") for r in rows if isinstance(r, Mapping)]
    if any(not isinstance(x, str) for x in ids):
        return ["row_id_not_string"]
    counts = Counter(ids)
    req = Counter(required)
    problems: list[str] = []
    missing = sorted(req - counts)
    extra = sorted(counts - req)
    duplicate = sorted(k for k, v in counts.items() if v > 1)
    if missing:
        problems.append(f"missing_rows:{missing}")
    if extra:
        problems.append(f"extra_rows:{extra}")
    if duplicate:
        problems.append(f"duplicate_rows:{duplicate}")
    if not required:
        problems.append("empty_required_set")
    return problems


# ---------------------------------------------------------------------------
# identity, environment and config helpers
# ---------------------------------------------------------------------------


def _git_identity(project_root: Path) -> dict:
    try:
        def run(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=project_root, check=True,
                                  capture_output=True, text=True).stdout.strip()
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
        diff = run("diff", "HEAD")
        return {"commit": commit, "dirty": bool(status),
                "status": status.splitlines(),
                "tracked_diff_sha256": sha256_bytes(diff.encode())}
    except Exception as exc:  # noqa: BLE001 -- identity is recorded, not assumed
        return {"commit": None, "error": f"{type(exc).__name__}: {exc}"}


def implementation_identity(project_root: Path) -> dict:
    files = (
        "src/ireset/train/losses.py",
        "src/ireset/model/tokens.py",
        "src/ireset/model/conditioner.py",
        "src/ireset/model/deterministic.py",
        "src/ireset/diffusion/generator.py",
        "src/ireset/eval/w1_real_objective_audit.py",
        "scripts/audit_w1_real_objective.py",
        "tests/test_w1_real_objective_audit.py",
    )
    hashes = {}
    for rel in files:
        p = Path(project_root) / rel
        hashes[rel] = sha256_file(p) if p.is_file() else None
    return {"git": _git_identity(project_root), "files": hashes}


def environment_facts() -> dict:
    import importlib.metadata as md

    def version(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    cuda = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else None),
        "runtime": torch.version.cuda if torch.cuda.is_available() else None,
    }
    lock = Path(PROJECT_ROOT) / "uv.lock"
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": version("mujoco"),
        "diffusers": version("diffusers"),
        "cuda": cuda,
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def verify_parent_anchors(project_root: Path) -> dict:
    problems: list[str] = []
    anchors: dict[str, dict] = {}
    for group in (PARENT_ANCHORS, SOURCE_ANCHORS):
        for name, (rel, want) in group.items():
            path = Path(project_root) / rel
            got = sha256_file(path) if path.is_file() else None
            anchors[name] = {"path": path_for_record(path), "expected": want,
                             "actual": got}
            if not path.is_file():
                problems.append(f"anchor_missing:{name}")
            elif want is not None and got != want:
                problems.append(f"anchor_drifted:{name}")
    return {"passed": not problems, "problems": problems, "anchors": anchors}


def checkpoint_config(ckpt: Mapping[str, Any], *, model_type: str) -> "TrainConfig":
    from ireset.train.trainer import TrainConfig

    raw = ckpt.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint config is not a mapping")
    if raw.get("model_type") != model_type:
        raise ValueError(f"checkpoint model_type {raw.get('model_type')} != {model_type}")
    loss_raw = raw.get("loss")
    if not isinstance(loss_raw, Mapping):
        raise ValueError("checkpoint loss config is not a mapping")
    loss = LossWeights(**{k: float(v) for k, v in loss_raw.items()})
    fields = {k: v for k, v in raw.items() if k != "loss"}
    return TrainConfig(**fields, loss=loss)


def _module_factory(model_type: str, cfg: "TrainConfig", device: str):
    from ireset.diffusion.generator import DiffusionG1
    from ireset.model.conditioner import ConditionEncoder
    from ireset.model.deterministic import DeterministicG1

    if model_type == "diffusion":
        model = DiffusionG1(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            **cfg.backbone_kwargs(),
        ).to(device)
    else:
        model = DeterministicG1(**cfg.backbone_kwargs()).to(device)
    conditioner = ConditionEncoder(cfg.d_model).to(device)
    return model, conditioner


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# real provider (only used after authorization)
# ---------------------------------------------------------------------------


class RealW1ObjectiveProvider:
    """Fixed real assets from the W1-D run, loaded only in the real entry."""

    mode = "real"

    def __init__(self, project_root: Path, *, device: str = "cpu",
                 verify_assets: bool = True) -> None:
        self.root = Path(project_root)
        self.device = device
        self.verify_assets = verify_assets
        self._identity: dict | None = None
        self._manifest: dict | None = None
        self._prereg: dict | None = None
        self._w1_reg: dict | None = None
        self._tokenizer = None
        self._rows: dict[str, dict] | None = None
        self._state: dict[str, dict] | None = None

    def _load_records(self) -> None:
        if self._w1_reg is not None:
            return
        from ireset.data import a2_integrity as integ
        from ireset.data import w1_training as w1

        reg_path = self.root / W1_RUN / "freeze/execution_registration.json"
        if not reg_path.is_file():
            raise FileNotFoundError(f"W1 execution registration missing: {reg_path}")
        self._w1_reg = read_json(reg_path)
        manifest_path = self.root / self._w1_reg["training_manifest"]["path"]
        cache_dir = self.root / self._w1_reg["cache"]["dir"]
        if sha256_file(manifest_path) != self._w1_reg["training_manifest"]["sha256"]:
            raise ValueError("training manifest hash drifted")
        if sha256_file(cache_dir / "cache_identity.json") != self._w1_reg["cache"]["identity_sha256"]:
            raise ValueError("cache identity hash drifted")
        if sha256_file(cache_dir / "root_stats.npz") != self._w1_reg["cache"]["root_stats_sha256"]:
            raise ValueError("root stats hash drifted")
        self._manifest = read_json(manifest_path)
        prereg_path = self.root / self._manifest["preregistration"]["path"]
        if sha256_file(prereg_path) != self._manifest["preregistration"]["sha256"]:
            raise ValueError("preregistration hash drifted")
        self._prereg = read_json(prereg_path)
        check = w1.verify_preregistration(
            self._prereg, self.root, self.root / "resources/data/omniretarget/robot-object",
            self.root / "resources/data/OMOMO_new",
            verify_assets=self.verify_assets,
        )
        if not check["passed"]:
            raise ValueError(f"preregistration re-verification failed: {check['problems']}")
        cache_check = w1.verify_cache_contents(
            cache_dir, self._manifest, self._prereg, self.root)
        if not cache_check["passed"]:
            raise ValueError(f"cache content verification failed: {cache_check['problems']}")
        self._cache_dir = cache_dir
        self._rows = {r["demo"]: r for r in self._manifest.get("rows", [])}

    def verify_identity(self) -> dict:
        self._load_records()
        from ireset.data import w1_training as w1

        assert self._w1_reg and self._manifest and self._prereg
        problems: list[str] = []
        parent = verify_parent_anchors(self.root)
        problems += parent["problems"]
        config_path = self.root / TRAIN_CONFIG
        want_cfg = self._w1_reg.get("configs", {}).get("config_file_sha256")
        got_cfg = sha256_file(config_path) if config_path.is_file() else None
        if got_cfg != want_cfg:
            problems.append("train_config_drifted")
        checkpoint_facts = {}
        for model, (rel, want, step) in EXPECTED_CHECKPOINTS.items():
            p = self.root / rel
            got = sha256_file(p) if p.is_file() else None
            checkpoint_facts[model] = {
                "path": path_for_record(p), "expected_sha256": want,
                "sha256": got, "step": step,
            }
            if got != want:
                problems.append(f"checkpoint_drifted:{model}")
        if self._manifest.get("manifest_name") != w1.MANIFEST_NAME:
            problems.append("manifest_name")
        if self._manifest.get("protocol") != w1.MOTION_PROTOCOL_VERSION:
            problems.append("manifest_protocol")
        sel = self._manifest.get("selection", {})
        train24 = list(sel.get("train24", []))
        recomputed = w1.select_train24(json.loads(
            (self.root / "derived/data/split/train.json").read_text())["demos"])
        if train24 != recomputed:
            problems.append("train24_not_reproducible")
        if sel.get("train24_canonical_sha256") not in (None, w1.canonical_list_sha256(recomputed)):
            problems.append("train24_canonical_sha")
        # old rejected runs are references only; they never fund this cohort
        attr_final = self.root / ATTR_RUN / "decisions/attempt_001/prediction_consistency_final.json"
        attr_status = "unknown"
        if attr_final.is_file():
            try:
                attr_status = read_json(attr_final).get("conclusion", "unknown")
            except (OSError, ValueError):
                attr_status = "unreadable"
        return {
            "passed": not problems, "problems": problems,
            "manifest": {"path": path_for_record(
                self.root / self._w1_reg["training_manifest"]["path"]),
                "sha256": self._w1_reg["training_manifest"]["sha256"]},
            "cache": {"dir": path_for_record(self._cache_dir),
                      "identity_sha256": self._w1_reg["cache"]["identity_sha256"],
                      "root_stats_sha256": self._w1_reg["cache"]["root_stats_sha256"]},
            "preregistration": self._manifest.get("preregistration"),
            "checkpoints": checkpoint_facts,
            "config": {"path": path_for_record(config_path),
                       "sha256": got_cfg},
            "parent_anchors": parent["anchors"],
            "prior_runs": {
                "w1_final": path_for_record(self.root / W1_RUN / "decisions/attempt_001/w1_de_final.json"),
                "attribution_final": path_for_record(attr_final),
                "attribution_conclusion_reference": attr_status,
                "budget_transfer": False,
            },
        }

    def tokenizer(self):
        if self._tokenizer is None:
            from ireset.data.windows import make_tokenizer
            from ireset.model.tokens import RootStats

            self._load_records()
            stats = RootStats.load(self._cache_dir / "root_stats.npz")
            self._tokenizer = make_tokenizer(stats)
        return self._tokenizer

    def select_windows(self) -> list[WindowKey]:
        self._load_records()
        from ireset.data import w1_training as w1

        assert self._manifest and self._prereg and self._rows
        sel = self._manifest.get("selection", {})
        train24 = [d for d in sel.get("train24", []) if d in self._rows]
        rows = {d: self._rows[d] for d in train24 if self._rows[d].get("split") == "train"}
        selected: list[tuple[str, list[int]]] = []
        for demo in sorted(rows):
            row = rows[demo]
            if not row.get("admission", {}).get("motion"):
                continue
            T = int(row.get("T", 0))
            starts = list(range(0, T - WINDOW + 1, STRIDE))
            if len(starts) >= WINDOWS_PER_DEMO:
                selected.append((demo, starts[:WINDOWS_PER_DEMO]))
            if len(selected) == DEMO_COUNT:
                break
        if len(selected) != DEMO_COUNT:
            raise ValueError("fewer than two train demos with two full windows")
        assets_sha = (self._prereg.get("assets") or {}).get("a1_assets_sha256") or ""
        code_sha = w1.condition_code_sha256(self.root)
        keys: list[WindowKey] = []
        for demo, starts in selected:
            row = rows[demo]
            key = w1.schedule_cache_key(
                row["inputs"]["npz_sha256"], row["inputs"]["pt_sha256"],
                assets_sha, code_sha)
            input_sha = sha256_bytes(json.dumps(
                {"npz": row["inputs"]["npz_sha256"],
                 "pt": row["inputs"]["pt_sha256"],
                 "derived": row["derived"]["sha256"]},
                sort_keys=True, separators=(",", ":")).encode())
            for start in starts:
                keys.append(WindowKey(
                    demo=demo, start=int(start), end=int(start + WINDOW),
                    T=int(row["T"]), cache_key=key, split="train",
                    input_sha=input_sha,
                    derived_sha=row["derived"]["sha256"]))
        return keys

    def load_inputs(self, keys: Sequence[WindowKey], *, noise: np.ndarray) -> AuditInputs:
        self._load_records()
        from ireset.data import w1_training as w1
        from ireset.data.teacher import keypoints_in_source_object, world_to_clip

        assert self._manifest and self._prereg and self._rows
        tokenizer = self.tokenizer()
        assets_sha = (self._prereg.get("assets") or {}).get("a1_assets_sha256") or ""
        code_sha = w1.condition_code_sha256(self.root)
        roots, qs, wfcs, kps_o, kps_c = [], [], [], [], []
        box_ts, box_qs, contacts = [], [], []
        for wk in keys:
            row = self._rows[wk.demo]
            clip = w1.load_w1_derived_clip(
                row, self._cache_dir / "schedules", self.root, assets_sha, code_sha)
            cond = clip.condition
            s, e = wk.start, wk.end
            root = np.asarray(clip.root[s:e], dtype=np.float64)
            q = np.asarray(clip.q[s:e], dtype=np.float64)
            wfc = np.asarray(cond.world_from_clip, dtype=np.float64)
            roots.append(root)
            qs.append(q)
            wfcs.append(wfc)
            kps_o.append(keypoints_in_source_object(
                cond.human_kps_world[s:e], cond.source_obj_pose[s:e]))
            kps_c.append(world_to_clip(cond.human_kps_world[s:e], wfc))
            box_ts.append(world_to_clip(
                cond.target_obj_pose[s:e, 0:3], wfc))
            box_qs.append(np.asarray(cond.target_obj_pose[s:e, 3:7], dtype=np.float64))
            contacts.append(np.asarray(cond.contact_flags()[s:e], dtype=np.float64))
        root = np.stack(roots)
        q = np.stack(qs)
        wfc = np.stack(wfcs)
        x0 = np.stack([tokenizer.encode(root[i], q[i], wfc[i])
                       for i in range(len(keys))]).astype(np.float32)
        lo = np.asarray(tokenizer.lo, dtype=np.float64)
        hi = np.asarray(tokenizer.hi, dtype=np.float64)
        mean = np.asarray(tokenizer.root_stats.mean, dtype=np.float64)
        std = np.asarray(tokenizer.root_stats.std, dtype=np.float64)
        x0_ind = independent_encode(root, q, wfc, lo, hi, mean, std)
        return AuditInputs(
            window_keys=list(keys), root=root, q=q, wfc=wfc,
            kps_obj=np.stack(kps_o).astype(np.float32),
            kps_clip=np.stack(kps_c).astype(np.float32),
            box_t_clip=np.stack(box_ts).astype(np.float32),
            box_quat=np.stack(box_qs).astype(np.float32),
            contact=np.stack(contacts).astype(np.float32),
            x0=x0, x0_independent=x0_ind,
            joint_lo=lo, joint_hi=hi, root_mean=mean, root_std=std,
            t=np.asarray(DIFFUSION_TIMESTEPS, dtype=np.int64),
            noise=np.asarray(noise, dtype=np.float32), dtype="float32",
        )

    def model_factory(self) -> dict[str, ModelBundle]:
        self._load_records()
        bundles: dict[str, ModelBundle] = {}
        for model_type, (rel, want, step) in EXPECTED_CHECKPOINTS.items():
            path = self.root / rel
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            if sha256_file(path) != want:
                raise ValueError(f"{model_type}: checkpoint hash drifted")
            if ckpt.get("checkpoint_usable") is not True:
                raise ValueError(f"{model_type}: checkpoint not usable")
            if ckpt.get("parameters_finite") is not True:
                raise ValueError(f"{model_type}: checkpoint parameters not finite")
            if ckpt.get("diagnostic_only") is not False:
                raise ValueError(f"{model_type}: diagnostic-only checkpoint")
            if int(ckpt.get("step", -1)) != step:
                raise ValueError(f"{model_type}: checkpoint step != {step}")
            cfg = checkpoint_config(ckpt, model_type=model_type)
            model, conditioner = _module_factory(model_type, cfg, self.device)
            model.load_state_dict(ckpt["model"], strict=True)
            conditioner.load_state_dict(ckpt["conditioner"], strict=True)
            model.train(); conditioner.train()
            params = [*model.parameters(), *conditioner.parameters()]
            if not all(bool(torch.isfinite(p).all()) for p in params):
                raise ValueError(f"{model_type}: non-finite loaded parameters")
            stats = ckpt.get("root_stats") or {}
            tk = self.tokenizer()
            if not (np.array_equal(np.asarray(stats.get("mean")), tk.root_stats.mean)
                    and np.array_equal(np.asarray(stats.get("std")), tk.root_stats.std)):
                raise ValueError(f"{model_type}: checkpoint root_stats != cache")
            identity = ckpt.get("identity") or {}
            for field_name, want_value in (
                ("config_file_sha256", self._w1_reg["configs"]["config_file_sha256"]),
                ("assets_sha256", (self._prereg.get("assets") or {}).get("a1_assets_sha256")),
            ):
                if want_value is not None and identity.get(field_name) != want_value:
                    raise ValueError(f"{model_type}: checkpoint identity {field_name}")
            if identity.get("training_manifest", {}).get("sha256") != self._w1_reg["training_manifest"]["sha256"]:
                raise ValueError(f"{model_type}: checkpoint manifest identity")
            bundles[model_type] = ModelBundle(
                model_type=model_type, model=model, conditioner=conditioner,
                metadata={
                    "checkpoint_path": path_for_record(path),
                    "checkpoint_sha256": want,
                    "step": step,
                    "config": ckpt.get("config"),
                    "root_stats": stats,
                    "state_sha256": _state_sha(model.state_dict()),
                    "conditioner_state_sha256": _state_sha(conditioner.state_dict()),
                    "parameters_finite": True,
                    "checkpoint_usable": True,
                })
        return bundles

    def loss_factory(self):
        tokenizer = self.tokenizer()

        def build(_model_type: str):
            return LossAssembler(tokenizer, _audit_weights(), self.device, use_fk=True)

        return build


# ---------------------------------------------------------------------------
# synthetic provider (no real dataset/cache/checkpoint/history)
# ---------------------------------------------------------------------------


class _SyntheticConditioner(torch.nn.Module):
    """Tiny differentiable conditioner stand-in with the production signature."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, batch: dict) -> torch.Tensor:
        return batch["x0"] * self.scale


class _SyntheticModel(torch.nn.Module):
    """Tiny model stand-in that stays close to the target.

    The real conditioner/backbone classes are exercised in the real provider
    and in a separate shape test.  This stand-in keeps the synthetic pass away
    from the SO(3) pi branch so the pass/fail separation measures the harness,
    not an untrained network's arbitrary rotation error.
    """

    def __init__(self, model_type: str) -> None:
        super().__init__()
        self.model_type = model_type
        self.bias = torch.nn.Parameter(torch.linspace(-0.02, 0.02, 38,
                                                       dtype=torch.float32))
        if model_type == "diffusion":
            from diffusers import DDPMScheduler

            self.ddpm = DDPMScheduler(
                num_train_timesteps=1000, beta_schedule="scaled_linear",
                beta_start=1e-4, beta_end=0.02,
                prediction_type="sample", clip_sample=False)
        else:
            self.ddpm = None

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        if self.ddpm is None:
            raise RuntimeError("deterministic stand-in has no noise path")
        return self.ddpm.add_noise(x0, noise, t)

    def forward(self, a: torch.Tensor, b: Any = None,
                memory: torch.Tensor | None = None,
                frames: int | None = None) -> torch.Tensor:
        if self.model_type == "diffusion":
            x_t, t, mem = a, b, memory
            out = mem + 0.01 * x_t + 0.001 * (t.float() / 1000.0)[:, None, None]
        else:
            out = a
        return out + self.bias


class SyntheticObjectiveProvider:
    """Analytic four-window fixture using tiny PRODUCTION modules.

    The conditioner, deterministic/diffusion backbones and loss assembly are
    the real classes; only the data and the FK geometry are synthetic.  The
    artifacts are tagged ``scope=synthetic`` and never claim real-batch
    evidence.
    """

    mode = "synthetic"

    def __init__(self, *, seed: int = 20261026, d_model: int = 8,
                 device: str = "cpu") -> None:
        self.seed = int(seed)
        self.d_model = int(d_model)
        self.device = device
        self._tokenizer = None
        self._keys: list[WindowKey] | None = None
        self._state: dict[str, dict] | None = None
        self._worlds: list[Any] | None = None

    def _build_worlds(self):
        if self._worlds is not None:
            return self._worlds
        from ireset.eval.training_objective_audit import build_synthetic_world

        self._worlds = [
            build_synthetic_world(T=WINDOW + STRIDE, seed=self.seed + 1),
            build_synthetic_world(T=WINDOW + STRIDE, seed=self.seed + 2),
        ]
        self._tokenizer = self._worlds[0].tokenizer
        return self._worlds

    def verify_identity(self) -> dict:
        self._build_worlds()
        # Synthetic mode must not read real production history.  Parent
        # anchors are recorded as references only, never loaded.
        return {
            "passed": True, "problems": [],
            "synthetic": True,
            "parent_anchors_for_reference_only": {"loaded": False},
            "checkpoints": {m: {"path": None, "sha256": None, "step": None,
                                "synthetic": True} for m in MODELS},
            "config": {"path": None, "sha256": None, "synthetic": True},
            "prior_runs": {"budget_transfer": False, "reference": "not_loaded"},
        }

    def tokenizer(self):
        self._build_worlds()
        return self._tokenizer

    def select_windows(self) -> list[WindowKey]:
        if self._keys is None:
            keys: list[WindowKey] = []
            for di in range(DEMO_COUNT):
                demo = f"synthetic_demo_{di:02d}"
                for start in (0, STRIDE):
                    keys.append(WindowKey(
                        demo=demo, start=start, end=start + WINDOW,
                        T=WINDOW + STRIDE,
                        cache_key=f"synth_cache_{di}_{start}", split="train",
                        input_sha=sha256_bytes(f"{demo}:{start}:input".encode()),
                        derived_sha=sha256_bytes(f"{demo}:derived".encode())))
            self._keys = keys
        return list(self._keys)

    def load_inputs(self, keys: Sequence[WindowKey], *, noise: np.ndarray) -> AuditInputs:
        worlds = self._build_worlds()
        tokenizer = self.tokenizer()
        roots, qs, wfcs = [], [], []
        kps_o, kps_c, box_ts, box_qs, contacts = [], [], [], [], []
        refs = []
        for wk in keys:
            di = 0 if wk.demo.endswith("00") else 1
            world = worlds[di]
            s, e = wk.start, wk.end
            rng = np.random.default_rng(self.seed + 100 + di)
            kp = rng.normal(0.0, 0.1, size=(WINDOW, 52, 3))
            roots.append(world.root[s:e])
            qs.append(world.q[s:e])
            wfcs.append(world.wfc)
            kps_o.append(kp)
            kps_c.append(kp + np.array([0.05, -0.02, 0.01]))
            box_ts.append(np.broadcast_to(world.center, (WINDOW, 3)).copy())
            box_qs.append(world.box_quat[s:e])
            contacts.append(world.contact[s:e])
            refs.append(world)
        root = np.stack(roots).astype(np.float64)
        q = np.stack(qs).astype(np.float64)
        wfc = np.stack(wfcs).astype(np.float64)
        x0 = np.stack([tokenizer.encode(root[i], q[i], wfc[i])
                       for i in range(len(keys))]).astype(np.float32)
        lo = np.asarray(tokenizer.lo, dtype=np.float64)
        hi = np.asarray(tokenizer.hi, dtype=np.float64)
        mean = np.asarray(tokenizer.root_stats.mean, dtype=np.float64)
        std = np.asarray(tokenizer.root_stats.std, dtype=np.float64)
        x0_ind = independent_encode(root, q, wfc, lo, hi, mean, std)
        return AuditInputs(
            window_keys=list(keys), root=root, q=q, wfc=wfc,
            kps_obj=np.stack(kps_o).astype(np.float32),
            kps_clip=np.stack(kps_c).astype(np.float32),
            box_t_clip=np.stack(box_ts).astype(np.float32),
            box_quat=np.stack(box_qs).astype(np.float32),
            contact=np.stack(contacts).astype(np.float32),
            x0=x0, x0_independent=x0_ind,
            joint_lo=lo, joint_hi=hi, root_mean=mean, root_std=std,
            t=np.asarray(DIFFUSION_TIMESTEPS, dtype=np.int64),
            noise=np.asarray(noise, dtype=np.float32), dtype="float32",
        )

    def model_factory(self) -> dict[str, ModelBundle]:
        self._build_worlds()
        if self._state is None:
            torch.manual_seed(self.seed)
            state: dict[str, dict] = {}
            for model_type in MODELS:
                model = _SyntheticModel(model_type)
                conditioner = _SyntheticConditioner()
                state[model_type] = {
                    "model": {k: v.clone() for k, v in model.state_dict().items()},
                    "conditioner": {k: v.clone() for k, v in conditioner.state_dict().items()},
                    "config": {"model_type": model_type, "synthetic": True},
                }
            self._state = state
        bundles = {}
        for model_type in MODELS:
            rec = self._state[model_type]
            model = _SyntheticModel(model_type)
            conditioner = _SyntheticConditioner()
            model.load_state_dict(rec["model"], strict=True)
            conditioner.load_state_dict(rec["conditioner"], strict=True)
            model.train(); conditioner.train()
            bundles[model_type] = ModelBundle(
                model_type=model_type, model=model, conditioner=conditioner,
                metadata={
                    "checkpoint_path": None, "checkpoint_sha256": None,
                    "step": 0, "config": rec["config"],
                    "root_stats": {"mean": self.tokenizer().root_stats.mean.tolist(),
                                   "std": self.tokenizer().root_stats.std.tolist()},
                    "state_sha256": _state_sha(model.state_dict()),
                    "conditioner_state_sha256": _state_sha(conditioner.state_dict()),
                    "parameters_finite": True, "checkpoint_usable": True,
                    "synthetic": True,
                })
        return bundles

    def loss_factory(self):
        from ireset.eval.training_objective_audit import (
            SyntheticFK, build_synthetic_assembler)

        tokenizer = self.tokenizer()

        def build(_model_type: str):
            asm = build_synthetic_assembler(
                tokenizer, _audit_weights(), dtype=torch.float32,
                use_fk=True, fk=SyntheticFK(torch.float32))
            if asm.fk is not None:
                asm.fk.backend = "synthetic"
            return asm

        return build


# ---------------------------------------------------------------------------
# R0: registration and admission
# ---------------------------------------------------------------------------


def _array_schema() -> dict:
    schema = {
        "raw_root": {"shape": [WINDOW_COUNT, WINDOW, 7], "dtype": "float64", "unit": "m,qwxyz"},
        "raw_q": {"shape": [WINDOW_COUNT, WINDOW, 29], "dtype": "float64", "unit": "rad"},
        "raw_wfc": {"shape": [WINDOW_COUNT, 4, 4], "dtype": "float64", "unit": "m"},
        "kps_obj": {"shape": [WINDOW_COUNT, WINDOW, 52, 3], "dtype": "float32", "unit": "m"},
        "kps_clip": {"shape": [WINDOW_COUNT, WINDOW, 52, 3], "dtype": "float32", "unit": "m"},
        "box_t_clip": {"shape": [WINDOW_COUNT, WINDOW, 3], "dtype": "float32", "unit": "m"},
        "box_quat": {"shape": [WINDOW_COUNT, WINDOW, 4], "dtype": "float32", "unit": "wxyz"},
        "contact": {"shape": [WINDOW_COUNT, WINDOW, 4], "dtype": "float32", "unit": "bool"},
        "x0": {"shape": [WINDOW_COUNT, WINDOW, 38], "dtype": "float32", "unit": "normalized"},
        "x0_independent": {"shape": [WINDOW_COUNT, WINDOW, 38], "dtype": "float64", "unit": "normalized"},
        "joint_lo": {"shape": [29], "dtype": "float64", "unit": "rad"},
        "joint_hi": {"shape": [29], "dtype": "float64", "unit": "rad"},
        "root_mean": {"shape": [3], "dtype": "float64", "unit": "m"},
        "root_std": {"shape": [3], "dtype": "float64", "unit": "m"},
        "t": {"shape": [WINDOW_COUNT], "dtype": "int64", "unit": "timestep"},
        "noise": {"shape": [WINDOW_COUNT, WINDOW, 38], "dtype": "float32", "unit": "normalized"},
        "alpha_cumprod": {"shape": [1000], "dtype": "float64", "unit": "probability"},
    }
    for model in MODELS:
        schema[f"prediction_{model}"] = {
            "shape": [WINDOW_COUNT, WINDOW, 38], "dtype": "float32", "unit": "normalized"}
        for term, shape in {
            "pos": [WINDOW_COUNT, WINDOW, 3],
            "rot6d": [WINDOW_COUNT, WINDOW, 6],
            "joint": [WINDOW_COUNT, WINDOW, 29],
            "geodesic_rad2": [WINDOW_COUNT, WINDOW],
            "vel": [WINDOW_COUNT, WINDOW - 1, 38],
            "fk": [WINDOW_COUNT, WINDOW, len(KPT_TO_SMPLH)],
        }.items():
            schema[f"term_elements_{model}_{term}"] = {
                "shape": shape, "dtype": "float64", "unit": "term_element"}
        schema[f"fk_points_pred_{model}"] = {
            "shape": [WINDOW_COUNT, WINDOW, len(KPT_TO_SMPLH), 3],
            "dtype": "float32", "unit": "m"}
        schema[f"fk_points_target_{model}"] = {
            "shape": [WINDOW_COUNT, WINDOW, len(KPT_TO_SMPLH), 3],
            "dtype": "float32", "unit": "m"}
        if model == "diffusion":
            schema["x_t_diffusion"] = {
                "shape": [WINDOW_COUNT, WINDOW, 38], "dtype": "float32",
                "unit": "normalized_noisy"}
    return schema


def _gradient_schema() -> dict:
    return {
        f"{kind}_grad_{model}": {
            "shape": [len(GRADIENT_SLICES), WINDOW_COUNT, WINDOW, 38],
            "dtype": "float64" if kind == "independent" else "float32",
            "slices": list(GRADIENT_SLICES),
        }
        for model in MODELS for kind in ("production", "independent")
    }


def run_r0(root: Path, stage: str, attempt: str, t0: float, budget: dict,
           provider, calls: AuditCallBudget, tolerance: dict,
           *, mode: str) -> dict:
    """Identity/registration before values, then four-window admission."""
    deadline = t0 + float(budget["remaining_s"])
    attempt_dir = stage_dir(root, stage) / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    gaps: list[str] = []
    identity: dict = {}
    try:
        identity = provider.verify_identity()
        if not identity.get("passed"):
            problems += [f"identity:{p}" for p in identity.get("problems", [])]
    except Exception as exc:  # noqa: BLE001 -- refusal becomes evidence
        problems.append(f"identity_unreadable:{type(exc).__name__}:{exc}")
    keys: list[WindowKey] = []
    try:
        keys = provider.select_windows()
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"window_selection_failed:{type(exc).__name__}:{exc}")
    if len(keys) != WINDOW_COUNT:
        problems.append(f"window_selection_count:{len(keys)}")
    noise = generate_noise(NOISE_SEED, (WINDOW_COUNT, WINDOW, 38))
    alpha = independent_alpha_cumprod(1000, "scaled_linear", 1e-4, 0.02)
    bundles: dict[str, ModelBundle] = {}
    if len(keys) == WINDOW_COUNT:
        try:
            bundles = provider.model_factory()
        except Exception as exc:  # noqa: BLE001 -- refusal becomes registration evidence
            problems.append(f"checkpoint_load:{type(exc).__name__}:{exc}")
    weights = _audit_weights()
    contract_path = attempt_dir / "tolerance_contract.json"
    write_json(contract_path, tolerance)
    contract_sha = sha256_file(contract_path)
    registration = {
        "schema": "w1_real_objective_registration_v1",
        "audit_schema": AUDIT_SCHEMA,
        "cohort_id": COHORT_ID,
        "mode": mode,
        "identity": identity,
        "identity_sha256": sha256_bytes(json.dumps(
            _jsonable(identity), sort_keys=True, separators=(",", ":")).encode()),
        "window_keys": [k.as_dict() for k in keys],
        "selection_rule": "train24 metadata; demo lexicographic; first two full windows",
        "t": [int(x) for x in DIFFUSION_TIMESTEPS],
        "noise_rule": {
            "seed": NOISE_SEED, "device": "cpu", "shape": [WINDOW_COUNT, WINDOW, 38],
            "dtype": "float32", "draws": 1, "protocol": NOISE_PROTOCOL,
            "sha256": _hash_array(noise),
        },
        "alpha_rule": {
            "schedule": "scaled_linear", "beta_start": 1e-4,
            "beta_end": 0.02, "num_train_timesteps": 1000,
            "sha256": _hash_array(alpha),
        },
        "weights": weights.as_dict(),
        "call_caps": dict(CALL_CAPS),
        "stage_budgets": {k: dict(v) for k, v in STAGE_BUDGETS.items()},
        "total_budget_s": TOTAL_BUDGET_S,
        "tolerance_contract": tolerance,
        "tolerance_contract_sha256": contract_sha,
        "effective_config": {
            "weights": weights.as_dict(),
            "micro_batch": WINDOW_COUNT,
            "grad_accum": 4,
            "window": WINDOW,
            "stride": STRIDE,
            "models": list(MODELS),
            "contact_supervision": "disabled",
        },
        "checkpoint_configs": {
            m: (bundles[m].metadata.get("config") if m in bundles else None)
            for m in MODELS
        },
        "array_schema": _array_schema(),
        "gradient_schema": _gradient_schema(),
        "required_objective_rows": required_objective_row_keys(keys),
        "required_gradient_rows": required_gradient_row_keys(),
        "coverage": {
            "objective_coverage": "partial",
            "audited": [
                "clip_encode", "root_standardization", "joint_mapping",
                "pos", "rot6d", "joint", "geodesic_rad2", "vel",
                "fk_m2_aggregation_only", "total_arithmetic",
                "output_space_gradients_five_terms_and_L5",
                "contact_branch_off",
            ],
            "not_audited": [
                "real_fk_geometry", "fk_derivative", "network_parameter_gradient",
                "accumulation", "gradient_clip", "adamw", "optimizer_update",
                "model_selection", "ddim_sampling",
            ],
        },
        "boundaries": BOUNDARIES,
        "implementation": implementation_identity(PROJECT_ROOT),
        "environment": environment_facts(),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "deadline_perf_counter": deadline,
    }
    registration["implementation_sha256"] = sha256_bytes(json.dumps(
        _jsonable(registration["implementation"]), sort_keys=True,
        separators=(",", ":")).encode())
    registration_path = attempt_dir / "registration.json"
    write_json(registration_path, registration)
    write_json(attempt_dir / "window_keys.json", {
        "schema": "w1_real_objective_window_keys_v1",
        "keys": [k.as_dict() for k in keys],
        "t": [int(x) for x in DIFFUSION_TIMESTEPS],
        "noise_seed": NOISE_SEED,
    })
    if time.perf_counter() > deadline:
        gaps.append("r0_deadline_after_registration")
        return {"passed": False, "problems": problems, "gaps": gaps,
                "registration": registration, "registration_path": registration_path,
                "identity": identity}
    def fail_admission(reason: str) -> dict:
        gaps.append(reason)
        admission = {
            "schema": "w1_real_objective_admission_v1",
            "passed": False, "problems": problems, "gaps": gaps,
            "validation": {"passed": False, "problems": [reason], "facts": {}},
            "checkpoints": {m: bundles[m].metadata for m in bundles},
            "weights": weights.as_dict(),
            "registration_sha256": sha256_file(registration_path),
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(attempt_dir / "admission.json", admission)
        return {"passed": False, "problems": problems, "gaps": gaps,
                "registration": registration, "registration_path": registration_path,
                "registration_sha256": admission["registration_sha256"],
                "admission": admission, "inputs": None, "bundles": bundles,
                "identity": identity, "keys": keys}
    if len(keys) != WINDOW_COUNT:
        return fail_admission("admission_skipped_incomplete_window_set")
    try:
        inputs = provider.load_inputs(keys, noise=noise)
    except Exception as exc:  # noqa: BLE001 -- refusal becomes evidence
        return fail_admission(f"load_inputs_failed:{type(exc).__name__}:{exc}")
    validation = validate_inputs(
        inputs, expected_keys=keys,
        tolerance=tolerance["tolerances"]["encode_float32"])
    if not validation["passed"]:
        problems += [f"admission:{p}" for p in validation["problems"]]
    if time.perf_counter() > deadline:
        gaps.append("r0_deadline_after_admission")
    admission = {
        "schema": "w1_real_objective_admission_v1",
        "passed": not problems and not gaps,
        "problems": problems, "gaps": gaps,
        "validation": validation,
        "checkpoints": {m: bundles[m].metadata for m in bundles},
        "weights": weights.as_dict(),
        "root_stats_sha256": _hash_array(np.concatenate([inputs.root_mean, inputs.root_std])),
        "arrays_sha256": {
            "x0": _hash_array(inputs.x0), "x0_independent": _hash_array(inputs.x0_independent),
            "root": _hash_array(inputs.root), "q": _hash_array(inputs.q),
            "wfc": _hash_array(inputs.wfc), "noise": _hash_array(inputs.noise),
            "kps_obj": _hash_array(inputs.kps_obj), "kps_clip": _hash_array(inputs.kps_clip),
            "box_t_clip": _hash_array(inputs.box_t_clip),
            "box_quat": _hash_array(inputs.box_quat),
            "contact": _hash_array(inputs.contact),
        },
        "registration_sha256": sha256_file(registration_path),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(attempt_dir / "admission.json", admission)
    return {
        "passed": admission["passed"], "problems": problems, "gaps": gaps,
        "registration": registration, "registration_path": registration_path,
        "registration_sha256": admission["registration_sha256"],
        "admission": admission, "inputs": inputs, "bundles": bundles,
        "identity": identity, "keys": keys,
    }


# ---------------------------------------------------------------------------
# P: production forward, capture and gradient
# ---------------------------------------------------------------------------


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, path)
    return sha256_file(path)


def _check_arrays(path: Path, required: Sequence[str], schema: Mapping[str, Any]) -> tuple[dict, list[str]]:
    problems: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    if not path.is_file():
        return arrays, ["array_file_missing"]
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            missing = sorted(set(required) - keys)
            extra = sorted(keys - set(required))
            if missing:
                problems.append(f"array_keys_missing:{missing}")
            if extra:
                problems.append(f"array_keys_extra:{extra}")
            for key in required:
                if key in z:
                    arrays[key] = z[key].copy()
    except (OSError, ValueError, TypeError) as exc:
        return arrays, [f"array_file_unreadable:{type(exc).__name__}"]
    for key, rec in schema.items():
        if key not in arrays:
            continue
        a = arrays[key]
        if list(a.shape) != list(rec["shape"]):
            problems.append(f"array_shape:{key}:{list(a.shape)}!={rec['shape']}")
        dtype_ok = str(a.dtype) == rec["dtype"]
        if not dtype_ok:
            problems.append(f"array_dtype:{key}:{a.dtype}!={rec['dtype']}")
        # Type dispatch precedes finiteness: np.isfinite on a string/object
        # array raises TypeError and would abort the report instead of
        # producing an honest incomplete result.
        if not np.issubdtype(a.dtype, np.number) or a.dtype.kind == "b":
            problems.append(f"array_not_numeric:{key}:{a.dtype}")
            continue
        try:
            finite = bool(np.isfinite(a).all())
        except (TypeError, ValueError):
            problems.append(f"array_nonfinite_unreadable:{key}")
            continue
        if not finite:
            problems.append(f"array_nonfinite:{key}")
    return arrays, problems


def run_p_stage(root: Path, stage: str, attempt: str, t0: float, budget: dict,
                r0: Mapping[str, Any], provider, calls: AuditCallBudget,
                tolerance: dict, *, mode: str,
                scope: dict | None = None) -> dict:
    deadline = t0 + float(budget["remaining_s"])
    attempt_dir = stage_dir(root, stage) / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    inputs: AuditInputs = r0["inputs"]
    bundles = r0["bundles"]
    registration_sha = r0["registration_sha256"]
    scope = dict(scope) if scope is not None else _unit_scope(inputs)
    loss_factory = provider.loss_factory()
    alpha = independent_alpha_cumprod(
        1000, "scaled_linear", 1e-4, 0.02)
    units: dict[str, dict] = {}
    problems: list[str] = []
    gaps: list[str] = []
    domain_bad: dict[str, int] = {}
    skipped: set[str] = set()
    stop_after_mismatch: str | None = None
    unit_facts: dict[str, dict] = {}
    unit_artifact_hashes: dict[str, dict[str, str]] = {}
    arrays: dict[str, np.ndarray] = dict(inputs.arrays_for_npz())
    arrays["alpha_cumprod"] = alpha
    gradients: dict[str, np.ndarray] = {}

    for model in MODELS:
        unit_dir = attempt_dir / "units" / model
        unit_dir.mkdir(parents=True, exist_ok=True)
        admission = unit_admission(
            root, stage, attempt, model, scope=scope,
            registration_sha=registration_sha, deadline=deadline,
            output_dir=unit_dir)
        write_json(unit_dir / "request.json", {
            "schema": "w1_real_objective_unit_request_v1",
            "stage": stage, "attempt": attempt, "unit": model,
            "scope": scope, "registration_sha256": registration_sha,
            "deadline_perf_counter": deadline,
            "admission": admission,
        })
        unit_artifact_hashes[model] = {
            "request": sha256_file(unit_dir / "request.json")}
        if not admission["passed"]:
            gaps.append(f"unit_admission:{model}:{admission['problems']}")
            skipped.add(model)
            continue
        if time.perf_counter() > deadline:
            gaps.append(f"unit_deadline:{model}")
            skipped.add(model)
            continue
        try:
            unit = production_objective_unit(
                bundles[model], inputs, loss_factory(model), calls,
                device=str(next(bundles[model].model.parameters()).device),
                dtype=torch.float32,
                tolerance=tolerance["tolerances"],
            )
        except Exception as exc:  # noqa: BLE001 -- a failed unit is evidence
            gaps.append(f"production_unit:{model}:{type(exc).__name__}:{exc}")
            skipped.add(model)
            write_json(unit_dir / "error.json", {
                "schema": "w1_real_objective_unit_error_v1",
                "model": model, "error": f"{type(exc).__name__}: {exc}",
            })
            unit_artifact_hashes[model] = {
                "request": sha256_file(unit_dir / "request.json"),
                "error": sha256_file(unit_dir / "error.json"),
            }
            continue
        units[model] = unit
        if not unit["independent_facts"].get("geodesic_domain_ok", False):
            domain_bad[model] = int(
                unit["independent_facts"].get("geodesic_domain_bad_count", 0))
            gaps.append(f"independent_rotation_domain:{model}:{domain_bad[model]}")
        unit_facts[model] = {
            "model_type": model,
            "prediction_sha256": _hash_array(unit["prediction"]),
            "production_grad_sha256": _hash_array(unit["production_grad"]),
            "independent_grad_sha256": _hash_array(unit["independent_grad"]),
            "prediction_problems": unit["prediction_problems"],
            "fk_backend": unit["fk_backend"],
            "contact_branch_executed": bool(
                unit.get("contact_branch_executed", False)),
            "independent_facts": unit["independent_facts"],
        }
        arrays[f"prediction_{model}"] = unit["prediction"]
        if unit["x_t"] is not None:
            arrays["x_t_diffusion"] = unit["x_t"]
        arrays[f"fk_points_pred_{model}"] = unit["fk_points_pred"]
        arrays[f"fk_points_target_{model}"] = unit["fk_points_target"]
        for term, arr in unit["elements"].items():
            arrays[f"term_elements_{model}_{term}"] = arr
        arrays[f"term_elements_{model}_fk"] = unit["fk_elements"]
        gradients[f"production_grad_{model}"] = unit["production_grad"]
        gradients[f"independent_grad_{model}"] = unit["independent_grad"]
        write_json(unit_dir / "result.json", {
            "schema": "w1_real_objective_unit_result_v1",
            "model": model, **unit_facts[model],
            "rng_before": unit["rng_before"], "rng_after": unit["rng_after"],
            "rng_state_before": unit["rng_state_before"],
            "rng_state_after": unit["rng_state_after"],
            "loss_parts": unit["loss_parts"],
        })
        unit_artifact_hashes[model] = {
            "request": sha256_file(unit_dir / "request.json"),
            "result": sha256_file(unit_dir / "result.json"),
        }
        # Stop new production evaluation after a confirmed mismatch, as
        # required by the draft's section 6.  The completed unit remains
        # available to F; the unexecuted model is an explicit gap.
        tmp_rows = build_objective_rows(
            {model: unit}, inputs, tolerance, torch.float32)
        tmp_grad = build_gradient_rows(
            {model: unit}, tolerance, torch.float32)
        _apply_domain_gaps(tmp_rows, tmp_grad, domain_bad)
        if any(r.get("status") == "mismatch" for r in (*tmp_rows, *tmp_grad)):
            stop_after_mismatch = model
            break

    for model in MODELS:
        if model not in units and model not in skipped:
            reason = (f"stop_after_mismatch:{stop_after_mismatch}"
                      if stop_after_mismatch else "not_executed")
            gaps.append(f"unit_not_executed:{model}:{reason}")

    # Only fully measured units enter the row table; every other unit keeps a
    # gap and cannot be silently replaced by a three-window batch.
    rows: list[dict] = []
    if len(units) == len(MODELS):
        rows = build_objective_rows(units, inputs, tolerance,
                                    torch.float32)
        grad_rows = build_gradient_rows(units, tolerance,
                                        torch.float32)
    else:
        grad_rows = []
        for model in MODELS:
            if model in units:
                rows.extend(build_objective_rows(
                    {model: units[model]}, inputs, tolerance,
                    torch.float32))
                grad_rows.extend(build_gradient_rows(
                    {model: units[model]}, tolerance,
                    torch.float32))
    _apply_domain_gaps(rows, grad_rows, domain_bad)
    required = required_objective_row_keys(inputs.window_keys)
    row_problems = row_multiset_problems(rows, required)
    grad_required = required_gradient_row_keys()
    grad_problems = row_multiset_problems(grad_rows, grad_required)
    if row_problems:
        gaps += [f"objective_rows:{p}" for p in row_problems]
    if grad_problems:
        gaps += [f"gradient_rows:{p}" for p in grad_problems]
    for row in rows:
        if row.get("status") == "mismatch":
            problems.append(f"objective_mismatch:{row['id']}")
        elif row.get("status") == "incomplete":
            gaps.append(f"objective_incomplete:{row['id']}:{row.get('problems')}")
    for row in grad_rows:
        if row.get("status") == "mismatch":
            problems.append(f"gradient_mismatch:{row['id']}")
        elif row.get("status") == "incomplete":
            gaps.append(f"gradient_incomplete:{row['id']}:{row.get('problems')}")

    arrays_path = attempt_dir / "objective_real_arrays.npz"
    grad_path = attempt_dir / "gradient_real_arrays.npz"
    array_sha = _save_npz(arrays_path, arrays) if arrays_path.parent.is_dir() else None
    grad_sha = _save_npz(grad_path, gradients) if gradients else None
    write_json(attempt_dir / "objective_real_rows.json", {
        "schema": "w1_real_objective_rows_v1",
        "rows": rows, "required": required,
        "counts": {"rows": len(rows), "required": len(required)},
    })
    write_json(attempt_dir / "gradient_real_rows.json", {
        "schema": "w1_real_objective_gradient_rows_v1",
        "rows": grad_rows, "required": grad_required,
        "counts": {"rows": len(grad_rows), "required": len(grad_required)},
    })
    trace = calls.snapshot()
    write_json(attempt_dir / "call_trace.json", {
        **trace, "mode": mode, "units": unit_facts,
        "production_forwards": 2 if len(units) == 2 else len(units),
        "parameter_backward": 0, "parameter_fd": 0,
    })
    write_json(attempt_dir / "rng_state_facts.json", {
        "schema": "w1_real_objective_rng_facts_v1",
        "mode": mode, "noise_rule": r0["registration"]["noise_rule"],
        "units": {m: {"before": units[m]["rng_before"],
                      "after": units[m]["rng_after"]} for m in units},
        "autocast": False, "tf32": bool(torch.backends.cuda.matmul.allow_tf32)
        if torch.cuda.is_available() else False,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    })
    complete = not gaps and len(units) == len(MODELS) and not row_problems and not grad_problems
    report = {
        "schema": "w1_real_objective_p_report_v1",
        "attempt": attempt, "mode": mode,
        "complete_evidence": bool(complete),
        "covered_contract_passed": bool(complete and not problems),
        "problems": problems, "gaps": gaps,
        "units": unit_facts,
        "unit_artifacts": unit_artifact_hashes,
        "stop_after_mismatch": stop_after_mismatch,
        "contact_branch_executions": sum(
            1 for f in unit_facts.values() if f.get("contact_branch_executed")),
        "arrays_sha256": array_sha,
        "gradient_arrays_sha256": grad_sha,
        "call_trace_sha256": sha256_file(attempt_dir / "call_trace.json"),
        "rng_facts_sha256": sha256_file(attempt_dir / "rng_state_facts.json"),
        "objective_rows_sha256": sha256_file(attempt_dir / "objective_real_rows.json"),
        "gradient_rows_sha256": sha256_file(attempt_dir / "gradient_real_rows.json"),
        "registration_sha256": registration_sha,
        "boundaries": BOUNDARIES,
    }
    write_json(attempt_dir / "p_report.json", report)
    return {"passed": complete, "problems": problems, "gaps": gaps,
            "units": units, "rows": rows, "gradient_rows": grad_rows,
            "report": report, "paths": {
                "arrays": arrays_path, "gradient_arrays": grad_path,
                "rows": attempt_dir / "objective_real_rows.json",
                "gradient_rows": attempt_dir / "gradient_real_rows.json",
                "call_trace": attempt_dir / "call_trace.json",
                "rng_facts": attempt_dir / "rng_state_facts.json",
                "report": attempt_dir / "p_report.json",
            }}


# ---------------------------------------------------------------------------
# F: independent replay and final verification
# ---------------------------------------------------------------------------


def _inputs_from_arrays(arrays: Mapping[str, np.ndarray],
                        registration: Mapping[str, Any]) -> AuditInputs:
    keys = [WindowKey.from_dict(k) for k in registration["window_keys"]]
    return AuditInputs(
        window_keys=keys,
        root=np.asarray(arrays["raw_root"], dtype=np.float64),
        q=np.asarray(arrays["raw_q"], dtype=np.float64),
        wfc=np.asarray(arrays["raw_wfc"], dtype=np.float64),
        kps_obj=np.asarray(arrays["kps_obj"], dtype=np.float32),
        kps_clip=np.asarray(arrays["kps_clip"], dtype=np.float32),
        box_t_clip=np.asarray(arrays["box_t_clip"], dtype=np.float32),
        box_quat=np.asarray(arrays["box_quat"], dtype=np.float32),
        contact=np.asarray(arrays["contact"], dtype=np.float32),
        x0=np.asarray(arrays["x0"], dtype=np.float32),
        x0_independent=np.asarray(arrays["x0_independent"], dtype=np.float64),
        joint_lo=np.asarray(arrays["joint_lo"], dtype=np.float64),
        joint_hi=np.asarray(arrays["joint_hi"], dtype=np.float64),
        root_mean=np.asarray(arrays["root_mean"], dtype=np.float64),
        root_std=np.asarray(arrays["root_std"], dtype=np.float64),
        t=np.asarray(arrays["t"], dtype=np.int64),
        noise=np.asarray(arrays["noise"], dtype=np.float32),
        dtype="float32",
    )


def _rebuild_unit_from_arrays(model: str, inputs: AuditInputs,
                              arrays: Mapping[str, np.ndarray],
                              gradients: Mapping[str, np.ndarray],
                              weights: LossWeights,
                              tolerances: Mapping[str, Any],
                              fk_elements_override: np.ndarray | None = None) -> dict:
    pred = torch.as_tensor(arrays[f"prediction_{model}"], dtype=torch.float32)
    fk_elements = torch.as_tensor(
        arrays[f"term_elements_{model}_fk"] if fk_elements_override is None
        else fk_elements_override, dtype=torch.float64)
    elements = {
        "pos": torch.as_tensor(arrays[f"term_elements_{model}_pos"], dtype=torch.float64),
        "rot6d": torch.as_tensor(arrays[f"term_elements_{model}_rot6d"], dtype=torch.float64),
        "joint": torch.as_tensor(arrays[f"term_elements_{model}_joint"], dtype=torch.float64),
        "geodesic_rad2": torch.as_tensor(arrays[f"term_elements_{model}_geodesic_rad2"], dtype=torch.float64),
        "vel": torch.as_tensor(arrays[f"term_elements_{model}_vel"], dtype=torch.float64),
    }
    elements["fk_elements"] = fk_elements
    prod_window = _per_window_scalars(elements, weights)
    ind_elements = independent_term_elements(
        pred.detach().double(), torch.as_tensor(inputs.x0_independent, dtype=torch.float64),
        rotation_domain=tolerances["rotation_domain"],
    )
    ind_window = _per_window_scalars(
        {**ind_elements, "fk_elements": fk_elements.double()}, weights)
    ind_scalars = independent_scalars(
        ind_elements, weights, fk_elements.double())
    prod_batch = {
        term: float(torch.as_tensor(
            [prod_window[i][term] for i in range(len(inputs.window_keys))],
            dtype=torch.float64).mean())
        for term in (*TERMS, "L5")
    }
    return {
        "model_type": model,
        "prediction": pred.numpy(),
        "x_t": (np.asarray(arrays["x_t_diffusion"]) if model == "diffusion" else None),
        "fk_points_pred": np.asarray(arrays[f"fk_points_pred_{model}"]),
        "fk_points_target": np.asarray(arrays[f"fk_points_target_{model}"]),
        "elements": {k: v.numpy() for k, v in elements.items() if k != "fk_elements"},
        "fk_elements": fk_elements.numpy(),
        "production_window": prod_window,
        "independent_window": ind_window,
        "production_batch": prod_batch,
        "independent_batch": {k: float(v.detach().double()) for k, v in ind_scalars.items()},
        "production_grad": np.asarray(gradients[f"production_grad_{model}"]),
        "independent_grad": np.asarray(gradients[f"independent_grad_{model}"]),
        "independent_facts": {
            "geodesic_domain_ok": bool(
                ind_elements["geodesic_domain_ok"].all().item()),
            "geodesic_domain_bad_count": int(
                (~ind_elements["geodesic_domain_ok"]).sum().item()),
        },
        "prediction_problems": [],
        "loss_parts": {},
        "rng_before": {}, "rng_after": {},
        "fk_backend": "production" if model in MODELS else "unknown",
    }


def _recompute_fk_elements(arrays: Mapping[str, np.ndarray],
                           model: str) -> np.ndarray:
    """Independent FK term elements from the two saved point sets.

    The production reference is ``((kp_pred - kp_target)**2).sum(-1)`` in
    float32.  Recomputing from the saved points (instead of trusting the saved
    element array) makes a shifted point set visible even when its element
    array was rewritten consistently.
    """
    pred = np.asarray(arrays[f"fk_points_pred_{model}"], dtype=np.float32)
    target = np.asarray(arrays[f"fk_points_target_{model}"], dtype=np.float32)
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[-1] != 3:
        raise ValueError(
            f"fk points shape mismatch: {pred.shape} != {target.shape}")
    return ((pred - target) ** 2).sum(axis=-1, dtype=np.float32).astype(np.float64)


def _gradient_element_problems(name: str, fresh: Any, saved: Any, *,
                               atol: float, rtol: float) -> list[str]:
    """Elementwise raw-gradient reconciliation (never a norm-only summary)."""
    try:
        f = np.asarray(fresh)
        s = np.asarray(saved)
    except (TypeError, ValueError):
        return [f"{name}:unreadable"]
    if f.shape != s.shape:
        return [f"{name}:shape:{f.shape}!={s.shape}"]
    if (not np.issubdtype(f.dtype, np.number) or not np.issubdtype(s.dtype, np.number)
            or f.dtype.kind == "b" or s.dtype.kind == "b"):
        return [f"{name}:non_numeric"]
    try:
        f64 = f.astype(np.float64, copy=False)
        s64 = s.astype(np.float64, copy=False)
        finite = bool(np.isfinite(f64).all() and np.isfinite(s64).all())
    except (TypeError, ValueError):
        return [f"{name}:unreadable"]
    if not finite:
        return [f"{name}:nonfinite"]
    diff = np.abs(f64 - s64)
    limit = atol + rtol * np.abs(s64)
    bad = diff > limit
    if not bool(bad.any()):
        return []
    idx = tuple(int(x) for x in np.argwhere(bad)[0])
    return [f"{name}:element_mismatch:{list(idx)}:"
            f"{float(f64[idx])}!={float(s64[idx])}:"
            f"limit={float(limit[idx])}"]


def _fk_element_problems(arrays: Mapping[str, np.ndarray], model: str,
                         recomputed: np.ndarray) -> list[str]:
    saved = np.asarray(arrays[f"term_elements_{model}_fk"], dtype=np.float64)
    if saved.shape != recomputed.shape:
        return [f"fk_elements_shape:{model}:{saved.shape}!={recomputed.shape}"]
    diff = np.abs(saved - recomputed)
    limit = 1e-9 + 1e-6 * np.abs(recomputed)
    bad = diff > limit
    if not bool(bad.any()):
        return []
    idx = tuple(int(x) for x in np.argwhere(bad)[0])
    return [f"fk_elements_not_recomputed:{model}:{list(idx)}:"
            f"{float(saved[idx])}!={float(recomputed[idx])}"]


def _fields_close(got: Any, want: Any, *, rtol: float = 1e-9,
                  atol: float = 1e-12) -> bool:
    """Recursive strict field equality; bool is never a number."""
    if isinstance(want, Mapping):
        if not isinstance(got, Mapping) or set(got) != set(want):
            return False
        return all(_fields_close(got[k], want[k], rtol=rtol, atol=atol)
                   for k in want)
    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            return False
        return all(_fields_close(g, w, rtol=rtol, atol=atol)
                   for g, w in zip(got, want))
    if isinstance(want, (bool, np.bool_)) or isinstance(got, (bool, np.bool_)):
        return isinstance(got, (bool, np.bool_)) and isinstance(want, (bool, np.bool_)) \
            and bool(got) is bool(want)
    if isinstance(want, (int, np.integer)) and not isinstance(want, bool):
        return (isinstance(got, (int, np.integer)) and not isinstance(got, bool)
                and int(got) == int(want))
    if isinstance(want, (float, np.floating)):
        return (_num(got)
                and abs(float(got) - float(want))
                <= atol + rtol * abs(float(want)))
    return got == want


def _objective_row_field_problems(rid: str, got: Any, want: Mapping[str, Any],
                                  scalar_tol: Mapping[str, Any]) -> list[str]:
    if not isinstance(got, Mapping):
        return [f"p_row_not_mapping:{rid}"]
    problems: list[str] = []
    missing = sorted(set(want) - {"id"} - set(got))
    extra = sorted(set(got) - set(want) - {"id"})
    if missing:
        problems.append(f"p_row_fields_missing:{rid}:{missing}")
    if extra:
        problems.append(f"p_row_fields_extra:{rid}:{extra}")
    derived = {"abs_error", "relative_error", "limit"}
    for key in sorted((set(want) & set(got)) - {"id"} - derived):
        if key in ("production", "independent"):
            p_value, i_value = got[key], want[key]
            if _num(p_value) and _num(i_value):
                cmp = compare_scalar(f"{rid}:{key}", p_value, i_value,
                                     scalar_tol)
                if cmp.get("status") != "within_tolerance":
                    problems.append(f"p_row_not_reproducible:{rid}:{key}")
            elif not _fields_close(p_value, i_value):
                problems.append(f"p_row_field:{rid}:{key}")
        elif not _fields_close(got[key], want[key]):
            problems.append(f"p_row_field:{rid}:{key}")
    # The comparison fields must also be arithmetically self-consistent with
    # this row's own production/independent pair and frozen tolerance.  They
    # are not required to be bitwise equal to the replay's float32 totals.
    p_value, i_value = got.get("production"), got.get("independent")
    if _num(p_value) and _num(i_value):
        abs_error = abs(float(p_value) - float(i_value))
        limit = float(scalar_tol["atol"]) + float(scalar_tol["rtol"]) * abs(i_value)
        within = abs_error <= limit
        if not _fields_close(got.get("abs_error"), abs_error,
                             rtol=1e-6, atol=1e-12):
            problems.append(f"p_row_abs_error_inconsistent:{rid}")
        if not _fields_close(got.get("limit"), limit,
                             rtol=1e-6, atol=1e-12):
            problems.append(f"p_row_limit_inconsistent:{rid}")
        expected_relative = None if abs(i_value) == 0.0 else abs_error / abs(i_value)
        if expected_relative is None:
            if got.get("relative_error") is not None:
                problems.append(f"p_row_relative_error_inconsistent:{rid}")
        elif not _fields_close(got.get("relative_error"), expected_relative,
                               rtol=1e-6, atol=1e-12):
            problems.append(f"p_row_relative_error_inconsistent:{rid}")
        if got.get("within_tolerance") is not within:
            problems.append(f"p_row_within_tolerance_inconsistent:{rid}")
    return problems


def _gradient_row_field_problems(rid: str, got: Any,
                                 want: Mapping[str, Any]) -> list[str]:
    if not isinstance(got, Mapping):
        return [f"p_gradient_row_not_mapping:{rid}"]
    problems: list[str] = []
    missing = sorted(set(want) - {"id"} - set(got))
    extra = sorted(set(got) - set(want) - {"id"})
    if missing:
        problems.append(f"p_gradient_row_fields_missing:{rid}:{missing}")
    if extra:
        problems.append(f"p_gradient_row_fields_extra:{rid}:{extra}")
    for key in sorted((set(want) & set(got)) - {"id"}):
        if not _fields_close(got[key], want[key]):
            problems.append(f"p_gradient_row_field:{rid}:{key}")
    return problems


def _read_p_report(p: Mapping[str, Any]) -> tuple[dict | None, list[str]]:
    """F reads the sealed P report from disk rather than the in-memory unit map."""
    gaps: list[str] = []
    paths = p.get("paths") or {}
    report_path = paths.get("report")
    if report_path is None or not Path(report_path).is_file():
        gaps.append("p_report_artifact_missing")
        report = p.get("report")
        return (dict(report) if isinstance(report, Mapping) else None), gaps
    try:
        report = read_json(Path(report_path))
    except (OSError, ValueError, TypeError):
        gaps.append("p_report_artifact_unreadable")
        return None, gaps
    if not isinstance(report, Mapping):
        gaps.append("p_report_artifact_not_a_mapping")
        return None, gaps
    return dict(report), gaps


def run_f_stage(root: Path, stage: str, attempt: str, t0: float, budget: dict,
                r0: Mapping[str, Any], p: Mapping[str, Any], provider,
                calls: AuditCallBudget, tolerance: dict, *, mode: str,
                scope: dict | None = None, replay: bool = True,
                replay_skip_reason: str | None = None) -> dict:
    deadline = t0 + float(budget["remaining_s"])
    replay_reserve_s = budget.get("reserve_s", SEAL_TAIL_RESERVE_S)
    if not _nonneg(replay_reserve_s):
        replay_reserve_s = SEAL_TAIL_RESERVE_S
    attempt_dir = stage_dir(root, stage) / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    scope = dict(scope) if scope is not None else _unit_scope(r0["inputs"])
    problems: list[str] = []
    gaps: list[str] = []
    replay_enabled = bool(replay)
    replay_stop_reason = replay_skip_reason
    if not replay_enabled:
        gaps.append(
            f"f_replay_skipped:{replay_stop_reason or 'disabled'}")
    registration = r0["registration"]
    inputs = r0["inputs"]
    reg_path = Path(r0.get("registration_path") or
                    (stage_dir(root, "R0") / "attempt_001" / "registration.json"))
    contract_path = reg_path.parent / "tolerance_contract.json"
    if not contract_path.is_file():
        gaps.append("tolerance_contract_artifact_missing")
    else:
        contract_problems = tolerance_contract_problems(read_json(contract_path))
        if contract_problems:
            problems.append(f"tolerance_contract_artifact:{contract_problems}")
        if sha256_file(contract_path) != registration.get("tolerance_contract_sha256"):
            problems.append("tolerance_contract_artifact_drifted")
    supplied_contract_sha = sha256_bytes(json.dumps(
        _jsonable(tolerance), sort_keys=True, separators=(",", ":")).encode())
    registered_contract_sha = sha256_bytes(json.dumps(
        _jsonable(registration.get("tolerance_contract")), sort_keys=True,
        separators=(",", ":")).encode())
    if supplied_contract_sha != registered_contract_sha:
        problems.append("tolerance_argument_mismatch")
    try:
        identity_now = provider.verify_identity()
        if not identity_now.get("passed"):
            problems.append(f"identity_reverification:{identity_now.get('problems')}")
        identity_sha = sha256_bytes(json.dumps(
            _jsonable(identity_now), sort_keys=True, separators=(",", ":")).encode())
        if identity_sha != registration.get("identity_sha256"):
            problems.append("identity_not_reproducible")
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"identity_reverification_failed:{type(exc).__name__}:{exc}")
    impl_now = implementation_identity(PROJECT_ROOT)
    impl_sha = sha256_bytes(json.dumps(
        _jsonable(impl_now), sort_keys=True, separators=(",", ":")).encode())
    if impl_sha != registration.get("implementation_sha256"):
        problems.append("implementation_not_reproducible")
    p_paths = p.get("paths") or {}
    p_report, report_gaps = _read_p_report(p)
    gaps += report_gaps
    # ``units`` is a nested structure from a sealed artifact.  Validate its
    # container type before membership/``.get``: a list merely containing the
    # model names must be a gap, never an AttributeError that aborts F and
    # leaves an unsealed running row.
    p_units: Mapping[str, Any] = {}
    if isinstance(p_report, Mapping):
        raw_units = p_report.get("units")
        if raw_units is None:
            raw_units = {}
        if not isinstance(raw_units, Mapping):
            gaps.append("p_units_not_a_mapping")
        else:
            p_units = raw_units
    reported_models = tuple(
        model for model in MODELS if model in p_units)
    if not reported_models:
        gaps.append("p_units_empty")
    for model in MODELS:
        if model not in reported_models:
            gaps.append(f"p_unit_not_measured:{model}")
    if isinstance(p_report, Mapping):
        if p_report.get("registration_sha256") not in (
                None, r0.get("registration_sha256")):
            problems.append("p_report_registration_mismatch")
        for model in reported_models:
            unit_info = p_units.get(model)
            if not isinstance(unit_info, Mapping):
                problems.append(f"p_unit_metadata_not_mapping:{model}")
    arrays_path = p_paths.get("arrays")
    grad_path = p_paths.get("gradient_arrays")
    arrays: dict[str, np.ndarray] = {}
    gradients: dict[str, np.ndarray] = {}
    arrays_ok = False
    gradients_ok = False
    if reported_models and arrays_path is not None and Path(arrays_path).is_file():
        arrays, problems_ = _check_arrays(
            Path(arrays_path), objective_array_keys_for(reported_models),
            registration["array_schema"])
        gaps += [f"p_arrays:{x}" for x in problems_]
        arrays_ok = not problems_
    elif arrays_path is None or not Path(arrays_path).is_file():
        gaps.append("p_arrays_missing")
    if reported_models and grad_path is not None and Path(grad_path).is_file():
        gradients, problems_ = _check_arrays(
            Path(grad_path), gradient_array_keys_for(reported_models),
            _gradient_schema())
        gaps += [f"p_gradient_arrays:{x}" for x in problems_]
        gradients_ok = not problems_
    elif grad_path is None or not Path(grad_path).is_file():
        gaps.append("p_gradient_arrays_missing")
    p_rows: list[dict] = []
    p_by_id: dict[str, dict] = {}
    p_grad_rows: list[dict] = []
    p_grad_by: dict[str, dict] = {}
    row_error = grad_row_error = "p_rows_not_read"
    if not problems and reported_models and arrays_ok and gradients_ok:
        rebuilt = _inputs_from_arrays(arrays, registration)
        admission_path = reg_path.parent / "admission.json"
        if not admission_path.is_file():
            gaps.append("admission_artifact_missing")
        else:
            try:
                admission = read_json(admission_path)
            except (OSError, ValueError, TypeError):
                admission = {}
                gaps.append("admission_artifact_unreadable")
            if not isinstance(admission, Mapping):
                admission = {}
                gaps.append("admission_artifact_not_a_mapping")
            if admission.get("passed") is not True:
                gaps.append("admission_not_passed")
            declared_arrays = admission.get("arrays_sha256")
            if declared_arrays is not None and not isinstance(
                    declared_arrays, Mapping):
                gaps.append("admission_arrays_sha256_not_a_mapping")
                declared_arrays = {}
            for key, want in (declared_arrays or {}).items():
                if key in arrays and _hash_array(np.asarray(arrays[key])) != want:
                    problems.append(f"admission_array_drifted:{key}")
            if admission.get("registration_sha256") != r0.get("registration_sha256"):
                problems.append("admission_registration_mismatch")
        # R0's registration and admission are the frozen identity.  F
        # independently regenerates the noise, alpha chain and x_t.
        expected_noise = generate_noise(NOISE_SEED, (WINDOW_COUNT, WINDOW, 38))
        if not np.array_equal(expected_noise, np.asarray(arrays["noise"])):
            problems.append("noise_not_reproducible")
        expected_alpha = independent_alpha_cumprod(1000, "scaled_linear", 1e-4, 0.02)
        if not np.array_equal(expected_alpha, np.asarray(arrays["alpha_cumprod"])):
            problems.append("alpha_not_reproducible")
        expected_xt = independent_x_t(arrays["x0"], arrays["noise"],
                                      np.asarray(arrays["t"], dtype=np.int64),
                                      expected_alpha)
        if "x_t_diffusion" in arrays:
            xt_cmp = compare_array(
                "x_t", arrays["x_t_diffusion"], expected_xt,
                tolerance["tolerances"]["loss_float32"], units="normalized_noisy")
            if xt_cmp.get("status") != "within_tolerance":
                problems.append("x_t_not_reproducible")
        elif "diffusion" in reported_models:
            gaps.append("x_t_missing")
        # x0 must be reproducible from raw root/q/wfc and the frozen stats.
        reencoded = independent_encode(
            rebuilt.root, rebuilt.q, rebuilt.wfc,
            rebuilt.joint_lo, rebuilt.joint_hi, rebuilt.root_mean, rebuilt.root_std)
        if not np.array_equal(reencoded, np.asarray(arrays["x0_independent"], dtype=np.float64)):
            problems.append("independent_x0_not_reproducible")
        enc_cmp = compare_array(
            "x0", arrays["x0"], reencoded,
            tolerance["tolerances"]["encode_float32"], units="normalized")
        if enc_cmp.get("status") != "within_tolerance":
            problems.append("x0_encode_mismatch")
        # Model/checkpoint construction happens once, before any per-unit RNG
        # restore, so constructor noise cannot perturb the restored state.  A
        # budget-revoked parent or an already-exhausted stage constructs no
        # models at all.
        fresh = None
        loss_factory = None
        if replay_enabled:
            if time.perf_counter() + replay_reserve_s >= deadline:
                replay_stop_reason = "deadline_exhausted_before_first_unit"
                gaps.append("f_replay_deadline_exhausted:before_first_unit")
                replay_enabled = False
            else:
                try:
                    fresh = provider.model_factory()
                    loss_factory = provider.loss_factory()
                except Exception as exc:  # noqa: BLE001 -- factory failure is evidence
                    gaps.append(
                        f"replay_factory_failed:{type(exc).__name__}:{exc}")
                    replay_enabled = False
        weights = _audit_weights()
        if p_paths.get("call_trace"):
            try:
                trace = read_json(p_paths["call_trace"])
            except (OSError, ValueError, TypeError):
                gaps.append("p_call_trace_unreadable")
            else:
                if not isinstance(trace, Mapping):
                    gaps.append("p_call_trace_not_a_mapping")
                counts = (trace.get("counts")
                          if isinstance(trace, Mapping) else None)
                if not isinstance(counts, Mapping):
                    # Preserve the historical missing-count problems, but do
                    # not call ``.get`` on a list/string artifact.
                    counts = {}
                for kind, cap in CALL_CAPS.items():
                    value = counts.get(kind)
                    if not _count(value) or int(value) > cap:
                        problems.append(f"p_call_count_invalid:{kind}")
                    if not _count(value) or int(value) < len(reported_models):
                        problems.append(f"p_call_count_missing:{kind}")
        raw_artifacts = ((p_report or {}).get("unit_artifacts")
                         if isinstance(p_report, Mapping) else None)
        if raw_artifacts is None:
            unit_artifact_declared: Mapping[str, Any] = {}
        elif isinstance(raw_artifacts, Mapping):
            unit_artifact_declared = raw_artifacts
        else:
            gaps.append("p_unit_artifacts_not_a_mapping")
            unit_artifact_declared = {}
        p_unit_root = Path(p_paths["arrays"]).parent / "units" if p_paths.get("arrays") else None
        fk_recomputed: dict[str, np.ndarray] = {}
        replay_facts: dict[str, dict] = {}
        for model in reported_models:
            unit_dir = attempt_dir / "units" / model
            unit_dir.mkdir(parents=True, exist_ok=True)
            admission = unit_admission(
                root, stage, attempt, model, scope=scope,
                registration_sha=r0.get("registration_sha256"),
                deadline=deadline, output_dir=unit_dir)
            write_json(unit_dir / "request.json", {
                "schema": "w1_real_objective_unit_request_v1",
                "stage": stage, "attempt": attempt, "unit": model,
                "scope": scope,
                "registration_sha256": r0.get("registration_sha256"),
                "deadline_perf_counter": deadline,
                "admission": admission,
            })
            # Recompute the FK elements from the saved point sets.  A shifted
            # point set is visible even if the element array and rows agree.
            try:
                recomputed_fk = _recompute_fk_elements(arrays, model)
            except (ValueError, TypeError, KeyError) as exc:
                gaps.append(f"fk_recompute_unavailable:{model}:{type(exc).__name__}")
            else:
                fk_problems = _fk_element_problems(arrays, model, recomputed_fk)
                problems += fk_problems
                if not fk_problems:
                    fk_recomputed[model] = recomputed_fk
            if p_unit_root is not None:
                declared = unit_artifact_declared.get(model)
                if not isinstance(declared, Mapping):
                    gaps.append(f"p_unit_artifacts_missing:{model}")
                    declared = None
                for kind in ("request", "result"):
                    artifact = p_unit_root / model / f"{kind}.json"
                    if not artifact.is_file():
                        gaps.append(f"p_unit_artifact_missing:{model}:{kind}")
                        continue
                    if declared is None:
                        # No declaration exists to compare against; this is a
                        # gap, not evidence that the artifact drifted.
                        continue
                    want = declared.get(kind)
                    if not isinstance(want, str) or not want:
                        gaps.append(
                            f"p_unit_artifact_declaration_missing:{model}:{kind}")
                        continue
                    if want != sha256_file(artifact):
                        problems.append(f"p_unit_artifact_drifted:{model}:{kind}")
            if not admission["passed"]:
                gaps.append(f"replay_unit_admission:{model}:{admission['problems']}")
                continue
            if not replay_enabled:
                gaps.append(
                    f"replay_skipped:{model}:{replay_stop_reason or 'disabled'}")
                continue
            if time.perf_counter() + replay_reserve_s >= deadline:
                replay_stop_reason = f"deadline_exhausted:{model}"
                gaps.append(f"f_replay_deadline_exhausted:{model}")
                replay_enabled = False
                continue
            unit_result_path = (p_unit_root / model / "result.json"
                                if p_unit_root is not None else None)
            if unit_result_path is None or not unit_result_path.is_file():
                gaps.append(f"p_unit_result_missing:{model}")
                continue
            try:
                saved_unit = read_json(unit_result_path)
            except (OSError, ValueError, TypeError):
                gaps.append(f"p_unit_result_unreadable:{model}")
                continue
            if not isinstance(saved_unit, Mapping):
                gaps.append(f"p_unit_result_not_a_mapping:{model}")
                continue
            state_problems = _rng_state_problems(saved_unit.get("rng_state_before"))
            if state_problems:
                gaps.append(f"p_rng_state_invalid:{model}:{state_problems}")
                continue
            state_problems = _rng_state_problems(saved_unit.get("rng_state_after"))
            if state_problems:
                gaps.append(f"p_rng_state_after_invalid:{model}:{state_problems}")
                continue
            try:
                _restore_rng_state(saved_unit["rng_state_before"])
            except Exception as exc:  # noqa: BLE001 -- bad state is evidence
                gaps.append(
                    f"rng_restore_failed:{model}:{type(exc).__name__}:{exc}")
                continue
            try:
                unit = production_objective_unit(
                    fresh[model], rebuilt, loss_factory(model), calls,
                    device=str(next(fresh[model].model.parameters()).device),
                    dtype=torch.float32, tolerance=tolerance["tolerances"])
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"replay_failed:{model}:{type(exc).__name__}:{exc}")
                continue
            if saved_unit.get("rng_before") != unit.get("rng_before"):
                problems.append(f"replay_rng_before_not_bitwise:{model}")
            if saved_unit.get("rng_after") != unit.get("rng_after"):
                problems.append(f"replay_rng_after_not_bitwise:{model}")
            if not _fields_close(saved_unit.get("rng_state_after"),
                                 unit.get("rng_state_after")):
                problems.append(f"replay_rng_state_after_not_bitwise:{model}")
            # Bind the fresh replay to the *artifact* arrays, never to the
            # in-memory P result object.
            if not np.array_equal(unit["prediction"],
                                  np.asarray(arrays[f"prediction_{model}"])):
                problems.append(f"replay_prediction_not_bitwise:{model}")
            if not np.array_equal(
                    np.asarray(unit["fk_points_pred"]),
                    np.asarray(arrays[f"fk_points_pred_{model}"])):
                problems.append(f"replay_fk_points_pred_not_bitwise:{model}")
            if not np.array_equal(
                    np.asarray(unit["fk_points_target"]),
                    np.asarray(arrays[f"fk_points_target_{model}"])):
                problems.append(f"replay_fk_points_target_not_bitwise:{model}")
            for term in ("pos", "rot6d", "joint", "geodesic_rad2", "vel"):
                if not np.array_equal(
                        np.asarray(unit["elements"][term]),
                        np.asarray(arrays[f"term_elements_{model}_{term}"])):
                    problems.append(
                        f"replay_term_elements_not_bitwise:{model}:{term}")
            if model == "diffusion":
                if not np.array_equal(unit["x_t"],
                                      np.asarray(arrays["x_t_diffusion"])):
                    problems.append(f"replay_x_t_not_bitwise:{model}")
            problems += _gradient_element_problems(
                f"replay_production_grad_not_bitwise:{model}",
                unit["production_grad"],
                gradients[f"production_grad_{model}"], atol=0.0, rtol=0.0)
            problems += _gradient_element_problems(
                f"replay_independent_grad_not_bitwise:{model}",
                unit["independent_grad"],
                gradients[f"independent_grad_{model}"],
                **{k: float(tolerance["tolerances"]["gradient_float64"][k])
                   for k in ("atol", "rtol")})
            replay_facts[model] = {
                "prediction_bitwise": True,
                "x_t_bitwise": model != "diffusion" or bool(
                    np.array_equal(unit["x_t"], arrays["x_t_diffusion"])),
                "production_grad_bitwise": not _gradient_element_problems(
                    f"g:{model}", unit["production_grad"],
                    gradients[f"production_grad_{model}"], atol=0.0, rtol=0.0),
                "independent_grad_bound_to_artifact": not _gradient_element_problems(
                    f"g:{model}", unit["independent_grad"],
                    gradients[f"independent_grad_{model}"],
                    **{k: float(tolerance["tolerances"]["gradient_float64"][k])
                       for k in ("atol", "rtol")}),
                "rng_restored_before_replay": True,
                "call_budget": calls.snapshot(),
            }
        # Recompute the complete row table from raw arrays and compare to P.
        expected_units: dict[str, dict] = {}
        for model in reported_models:
            if model not in fk_recomputed:
                gaps.append(f"expected_unit_unavailable:{model}")
                continue
            try:
                expected_units[model] = _rebuild_unit_from_arrays(
                    model, rebuilt, arrays, gradients, weights,
                    tolerance["tolerances"],
                    fk_elements_override=fk_recomputed[model])
            except (ValueError, TypeError, KeyError) as exc:
                gaps.append(f"expected_unit_unbuildable:{model}:{type(exc).__name__}")
        p_rows, p_by_id, row_error = load_rows_checked(Path(p_paths["rows"])) \
            if p_paths.get("rows") else ([], {}, "p_rows_path_missing")
        p_grad_rows, p_grad_by, grad_row_error = load_rows_checked(
            Path(p_paths["gradient_rows"])) if p_paths.get("gradient_rows") else (
            [], {}, "p_gradient_rows_path_missing")
        try:
            expected_rows = build_objective_rows(
                expected_units, rebuilt, tolerance, torch.float32)
            expected_grad_rows = build_gradient_rows(
                expected_units, tolerance, torch.float32)
            domain_bad = {
                model: int(unit.get("independent_facts", {}).get(
                    "geodesic_domain_bad_count", 0))
                for model, unit in expected_units.items()
                if not unit.get("independent_facts", {}).get(
                    "geodesic_domain_ok", True)}
            _apply_domain_gaps(expected_rows, expected_grad_rows, domain_bad)
            expected_by_id = _rows_by_id(expected_rows)
            expected_grad_by = _rows_by_id(expected_grad_rows)
        except (ValueError, TypeError, KeyError) as exc:
            gaps.append(f"expected_rows_unbuildable:{type(exc).__name__}")
            expected_by_id, expected_grad_by = {}, {}
        # The registered row set is the denominator.  Extra or missing rows
        # are gaps, never invisible to the final verdict.
        objective_multiset = (
            [row_error] if row_error else row_multiset_problems(
                p_rows, registration["required_objective_rows"]))
        gradient_multiset = (
            [grad_row_error] if grad_row_error else row_multiset_problems(
                p_grad_rows, registration["required_gradient_rows"]))
        gaps += [f"p_objective_multiset:{x}" for x in objective_multiset]
        gaps += [f"p_gradient_multiset:{x}" for x in gradient_multiset]
        if row_error:
            gaps.append(f"p_objective_rows:{row_error}")
        if grad_row_error:
            gaps.append(f"p_gradient_rows:{grad_row_error}")
        if not row_error:
            for rid, want in expected_by_id.items():
                got = p_by_id.get(rid)
                if got is None:
                    problems.append(f"p_row_missing:{rid}")
                    continue
                if want.get("scope") in ("arithmetic", "reconciliation"):
                    ttol = tolerance["tolerances"]["loss_float32"]
                else:
                    ttol = _term_scalar_tolerance(
                        tolerance["tolerances"]["loss_float32"],
                        str(want.get("term", "")), torch.float32,
                        tolerance.get("term_overrides"))
                problems += _objective_row_field_problems(rid, got, want, ttol)
            for rid in sorted(set(p_by_id) - set(expected_by_id)):
                problems.append(f"p_row_unexpected:{rid}")
        if not grad_row_error:
            for rid, want in expected_grad_by.items():
                got = p_grad_by.get(rid)
                if got is None:
                    problems.append(f"p_gradient_row_missing:{rid}")
                    continue
                problems += _gradient_row_field_problems(rid, got, want)
            for rid in sorted(set(p_grad_by) - set(expected_grad_by)):
                problems.append(f"p_gradient_row_unexpected:{rid}")
        # P's own row statuses must be consistent with the recomputation.
        for rid, row in p_by_id.items():
            if row.get("status") == "mismatch":
                problems.append(f"p_declared_mismatch:{rid}")
            elif row.get("status") == "incomplete":
                gaps.append(f"p_declared_incomplete:{rid}")
        for rid, row in p_grad_by.items():
            if row.get("status") == "mismatch":
                problems.append(f"p_declared_gradient_mismatch:{rid}")
            elif row.get("status") == "incomplete":
                gaps.append(f"p_declared_gradient_incomplete:{rid}")
    else:
        objective_multiset = ["rows_not_checked"]
        gradient_multiset = ["rows_not_checked"]
        row_error = row_error if reported_models else "p_units_empty"
        grad_row_error = grad_row_error if reported_models else "p_units_empty"
        replay_facts = {}
    consumers = {}
    for prior_stage in ("R0", "P"):
        _, state_problems = stage_state(root, prior_stage)
        consumers[prior_stage] = state_problems
        gaps += [f"consumer:{prior_stage}:{x}" for x in state_problems]
    complete = not gaps
    if problems:
        conclusion = ("real_batch_objective_mismatch" if mode == "real"
                      else "synthetic_pipeline_mismatch")
    elif gaps:
        conclusion = "audit_incomplete"
    else:
        conclusion = ("real_batch_covered_contract_passed" if mode == "real"
                      else "synthetic_contract_passed")
    verification = {
        "schema": "w1_real_objective_independent_verification_v1",
        "attempt": attempt, "mode": mode, "conclusion": conclusion,
        "problems": problems, "gaps": gaps,
        "row_multiset": {
            "objective": objective_multiset,
            "gradient": gradient_multiset,
        },
        "row_errors": {
            "objective": row_error,
            "gradient": grad_row_error,
        },
        "reported_models": list(reported_models),
        "array_sets": {
            "objective_required": objective_array_keys_for(reported_models),
            "gradient_required": gradient_array_keys_for(reported_models),
        },
        "replay": replay_facts,
        "consumer_states": consumers,
        "call_budget": calls.snapshot(),
        "boundaries": BOUNDARIES,
    }
    write_json(attempt_dir / "independent_verification.json", verification)
    failures = problems
    summary = {
        "schema": "w1_real_objective_audit_summary_v1",
        "audit_schema": AUDIT_SCHEMA,
        "cohort_id": COHORT_ID,
        "mode": mode,
        "attempt": attempt,
        "conclusion": conclusion,
        "producer_conclusion": conclusion,
        "complete_evidence": bool(complete),
        "covered_contract_passed": conclusion in (
            "real_batch_covered_contract_passed", "synthetic_contract_passed"),
        "problems": failures, "failures": failures, "gaps": gaps,
        "counts": {
            "windows": len(registration["window_keys"]),
            "models": len(reported_models),
            "objective_rows": len(p_rows),
            "gradient_rows": len(p_grad_rows),
            "required_objective_rows": len(registration["required_objective_rows"]),
            "required_gradient_rows": len(registration["required_gradient_rows"]),
        },
        "coverage": registration["coverage"],
        "boundaries": BOUNDARIES,
        "calls": calls.snapshot(),
        "registration_sha256": r0.get("registration_sha256"),
        "verification_sha256": sha256_file(attempt_dir / "independent_verification.json"),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(attempt_dir / "audit_summary.json", summary)
    return {"passed": conclusion in ("real_batch_covered_contract_passed",
                                     "synthetic_contract_passed"),
            "conclusion": conclusion, "problems": failures, "gaps": gaps,
            "summary": summary, "verification": verification,
            "report": attempt_dir / "audit_summary.json",
            "verification_path": attempt_dir / "independent_verification.json"}


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def _stage_scope() -> dict:
    return {
        "windows": WINDOW_COUNT, "models": list(MODELS),
        "batch": [WINDOW_COUNT, WINDOW, 38],
        "objective_coverage": "partial",
    }


def _unit_scope(inputs: AuditInputs) -> dict:
    return {
        "windows": [k.key for k in inputs.window_keys],
        "models": list(MODELS),
        "batch": [WINDOW_COUNT, WINDOW, 38],
        "objective_coverage": "partial",
    }


def _write_r0_summary(attempt_dir: Path, result: Mapping[str, Any], *,
                      mode: str) -> Path:
    summary = {
        "schema": "w1_real_objective_audit_summary_v1",
        "audit_schema": AUDIT_SCHEMA,
        "mode": mode,
        "conclusion": "audit_incomplete",
        "producer_conclusion": "audit_incomplete",
        "complete_evidence": False,
        "covered_contract_passed": False,
        "problems": list(result.get("problems", [])),
        "failures": list(result.get("problems", [])),
        "gaps": list(result.get("gaps", [])) or ["r0_not_passed"],
        "coverage": {"objective_coverage": "partial"},
        "boundaries": BOUNDARIES,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = attempt_dir / "audit_summary.json"
    write_json(path, summary)
    return path


def run_registered_audit(run_root: Path, provider, *, mode: str,
                         tolerance: Mapping[str, Any],
                         authorization_sha: str | None = None,
                         authorization: Mapping[str, Any] | None = None,
                         fixed_root: Path = HISTORY_ROOT,
                         project_root: Path = PROJECT_ROOT) -> dict:
    """Run R0 -> P -> F in order; every stage is pre-registered and sealed."""
    run_root = Path(run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if mode not in ("synthetic", "real"):
        raise ValueError(f"unknown mode {mode}")
    contract_problems = tolerance_contract_problems(tolerance)
    if contract_problems:
        raise SystemExit(f"tolerance contract refused: {contract_problems}")
    if mode == "real":
        base = (Path(project_root) / fixed_root).resolve()
        if run_root.parent != base:
            raise SystemExit(
                f"real run root must be a direct child of {base}, got {run_root}")
        if authorization_sha is None:
            raise SystemExit("real audit requires an authorization record")
        problems = authorization_problems(authorization)
        if problems:
            raise SystemExit(f"real audit authorization refused: {problems}")
        if (tolerance.get("mode") != "synthetic_calibrated"
                or not tolerance.get("evidence_sha256")):
            raise SystemExit("real audit requires a completed synthetic calibration contract")
    # Prove the remaining quota can fund the whole registered matrix before
    # creating even a cohort marker.  A partially funded chain would spend
    # calls in P/F and only then raise AuditCallBudgetExceeded.
    prior_calls = prior_call_counts(run_root, fixed_root=fixed_root, mode=mode)
    if prior_calls is None:
        raise SystemExit(
            "call history unknown: refusing to grant a fresh call quota")
    exhausted = {kind: prior_calls[kind] for kind in CALL_CAPS
                 if prior_calls[kind] >= CALL_CAPS[kind]}
    if exhausted:
        raise SystemExit(
            f"cumulative call budget exhausted for this cohort: {exhausted}")
    insufficient = {
        kind: {"prior": prior_calls[kind],
               "required": FULL_CHAIN_CALLS[kind],
               "cap": CALL_CAPS[kind]}
        for kind in CALL_CAPS
        if prior_calls[kind] + FULL_CHAIN_CALLS[kind] > CALL_CAPS[kind]
    }
    if insufficient:
        raise SystemExit(
            "cumulative call budget insufficient for a full audit chain: "
            f"{insufficient}")
    calls = AuditCallBudget(prior_counts=prior_calls)
    write_cohort(run_root, mode=mode, authorization_sha=authorization_sha)
    scope = _stage_scope()
    results: dict[str, Any] = {}
    last_seal: dict[str, Any] = {}

    # --- R0 -----------------------------------------------------------------
    t0 = time.perf_counter()
    budget = budget_for(run_root, "R0", fixed_root=fixed_root, mode=mode)
    attempt = allocate_attempt(run_root, "R0")
    pre_register_attempt(run_root, "R0", attempt, t0, scope=scope,
                         budget=budget, call_balance=calls.snapshot())
    r0 = run_r0(run_root, "R0", attempt, t0, budget, provider, calls,
                dict(tolerance), mode=mode)
    results["R0"] = r0
    attempt_dir = stage_dir(run_root, "R0") / attempt
    report = attempt_dir / "r0_report.json"
    write_json(report, {
        "schema": "w1_real_objective_r0_report_v1",
        "attempt": attempt, "mode": mode,
        "passed": bool(r0.get("passed")),
        "problems": r0.get("problems", []), "gaps": r0.get("gaps", []),
        "registration_sha256": r0.get("registration_sha256"),
        "admission_sha256": (sha256_file(attempt_dir / "admission.json")
                             if (attempt_dir / "admission.json").is_file() else None),
        "boundaries": BOUNDARIES,
    })
    complete = bool(r0.get("passed")) and not r0.get("gaps")
    seal = finish_attempt(
        run_root, "R0", attempt, t0, budget,
        {"complete": complete, "report": path_for_record(report),
         "summary": path_for_record(report),
         "registration_sha256": r0.get("registration_sha256")},
        status="done" if complete else "incomplete",
        extra={"stage": "R0", "scope": scope,
               "call_balance": calls.snapshot()},
    )
    last_seal["R0"] = {"sealed": seal.sealed, "problems": seal.problems,
                       "row": seal.row}
    if not seal.sealed or not complete:
        if not (attempt_dir / "audit_summary.json").is_file():
            _write_r0_summary(attempt_dir, r0, mode=mode)
        return {"mode": mode, "conclusion": "audit_incomplete",
                "complete_evidence": False, "covered_contract_passed": False,
                "problems": r0.get("problems", []),
                "gaps": r0.get("gaps", []) + seal.problems,
                "stages": results, "seals": last_seal,
                "summary_path": path_for_record(attempt_dir / "audit_summary.json")}

    # --- P ------------------------------------------------------------------
    p_scope = _unit_scope(r0["inputs"])
    t0 = time.perf_counter()
    budget = budget_for(run_root, "P", fixed_root=fixed_root, mode=mode)
    attempt = allocate_attempt(run_root, "P")
    pre_register_attempt(run_root, "P", attempt, t0, scope=p_scope,
                         budget=budget,
                         registration_sha=r0["registration_sha256"],
                         call_balance=calls.snapshot())
    p = run_p_stage(run_root, "P", attempt, t0, budget, r0, provider, calls,
                    dict(tolerance), mode=mode, scope=p_scope)
    results["P"] = p
    p_complete = bool(p.get("passed"))
    p_artifacts = {}
    for key in ("arrays", "gradient_arrays", "rows", "gradient_rows",
                "call_trace", "rng_facts", "report"):
        path = (p.get("paths") or {}).get(key)
        if path is not None and Path(path).is_file():
            p_artifacts[path_for_record(path)] = sha256_file(path)
    unit_artifacts = (p.get("report") or {}).get("unit_artifacts") or {}
    for model, record in unit_artifacts.items():
        if not isinstance(record, Mapping):
            continue
        for kind, want in record.items():
            path = stage_dir(run_root, "P") / attempt / "units" / model / f"{kind}.json"
            if path.is_file():
                p_artifacts[path_for_record(path)] = sha256_file(path)
    seal = finish_attempt(
        run_root, "P", attempt, t0, budget,
        {"complete": p_complete,
         "report": path_for_record(p["paths"]["report"]),
         "summary": path_for_record(p["paths"]["rows"]),
         "registration_sha256": r0["registration_sha256"],
         "artifacts": p_artifacts},
        status="done" if p_complete else "incomplete",
        extra={"stage": "P", "scope": p_scope,
               "call_balance": calls.snapshot()},
    )
    last_seal["P"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                      "problems": seal.problems, "row": seal.row}
    p_budget_exhausted = _p_budget_exhausted(p, seal)
    if p_budget_exhausted:
        last_seal["P"]["budget_exhausted"] = p_budget_exhausted
    # A terminal P row is legitimate partial evidence even when its
    # ``complete`` flag is false (e.g. stop-after-mismatch).  F must still
    # replay the measured units and preserve the mismatch; only a leftover
    # publishing/running row (unknown history) blocks F.  A parent stopped by
    # accounting (budget/seal/deadline/call cap) is different: F may organise
    # the existing evidence, but must not start new model calls.
    if not seal.terminal:
        return {"mode": mode, "conclusion": "audit_incomplete",
                "complete_evidence": False, "covered_contract_passed": False,
                "problems": p.get("problems", []),
                "gaps": p.get("gaps", []) + seal.problems,
                "stages": results, "seals": last_seal,
                "summary_path": None}
    if not p.get("units"):
        return {"mode": mode, "conclusion": "audit_incomplete",
                "complete_evidence": False, "covered_contract_passed": False,
                "problems": p.get("problems", []),
                "gaps": p.get("gaps", []) + ["p_no_measured_units"],
                "stages": results, "seals": last_seal,
                "summary_path": None}

    # --- F ------------------------------------------------------------------
    t0 = time.perf_counter()
    budget = budget_for(run_root, "F", fixed_root=fixed_root, mode=mode)
    attempt = allocate_attempt(run_root, "F")
    pre_register_attempt(run_root, "F", attempt, t0, scope=p_scope,
                         budget=budget,
                         registration_sha=r0["registration_sha256"],
                         call_balance=calls.snapshot())
    f = run_f_stage(
        run_root, "F", attempt, t0, budget, r0, p, provider,
        calls, dict(tolerance), mode=mode, scope=p_scope,
        replay=p_budget_exhausted is None,
        replay_skip_reason=(
            f"parent_budget_exhausted:{p_budget_exhausted}"
            if p_budget_exhausted else None))
    results["F"] = f
    # A completed mismatch is still complete evidence; an incomplete audit is
    # not.  Both publish a terminal, non-passing summary.  ``complete`` here
    # is the strict evidence flag; ``seal.terminal`` separately records that
    # the negative result was sealed rather than abandoned.
    complete_evidence = bool(f["summary"].get("complete_evidence"))
    seal = finish_attempt(
        run_root, "F", attempt, t0, budget,
        {"complete": complete_evidence,
         "report": path_for_record(f["report"]),
         "summary": path_for_record(f["report"]),
         "registration_sha256": r0["registration_sha256"],
         "artifacts": {
             path_for_record(f["verification_path"]): sha256_file(f["verification_path"]),
         }},
        status="done" if complete_evidence else "incomplete",
        extra={"stage": "F", "scope": p_scope,
               "call_balance": calls.snapshot()},
    )
    last_seal["F"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                      "problems": seal.problems, "row": seal.row}
    pass_conclusion = (
        "real_batch_covered_contract_passed" if mode == "real"
        else "synthetic_contract_passed")
    if not seal.terminal:
        # Unknown history (publishing/running leftover): never claim the
        # pass/negative result as final.  Withdraw any on-disk pass claim too.
        summary = dict(f["summary"])
        revoke_problems: list[str] = []
        if summary.get("conclusion") == pass_conclusion:
            updated_row, revoke_problems, summary = _withdraw_pass_claim(
                run_root, "F", attempt, Path(f["report"]), seal.row,
                ["f_seal_not_terminal", *seal.problems],
                reason="f_seal_not_terminal",
                extra={"final_consumer_problems": list(seal.problems)})
            last_seal["F"]["row"] = updated_row
            last_seal["F"]["problems"] = (
                list(last_seal["F"].get("problems", [])) + revoke_problems)
        f["summary"] = summary
        f["conclusion"] = "audit_incomplete"
        return {"mode": mode, "conclusion": "audit_incomplete",
                "complete_evidence": False, "covered_contract_passed": False,
                "problems": list(f.get("problems", [])) + revoke_problems,
                "gaps": (list(f.get("gaps", [])) + list(seal.problems)
                         + revoke_problems),
                "stages": results, "seals": last_seal,
                "summary": summary,
                "summary_path": path_for_record(f["report"])}
    _, final_consumer_problems = stage_state(run_root, "F")
    summary = dict(f["summary"])
    summary["final_consumer_problems"] = final_consumer_problems
    revoke_problems = []
    if summary.get("conclusion") == pass_conclusion and final_consumer_problems:
        # A pass claim cannot survive a non-consumable final record.  Withdraw
        # the on-disk summary and rebind the seal hash; never mutate only the
        # in-memory copy.
        updated_row, revoke_problems, summary = _withdraw_pass_claim(
            run_root, "F", attempt, Path(f["report"]), seal.row,
            final_consumer_problems,
            reason="final_consumer_not_consumable",
            extra={"final_consumer_problems": final_consumer_problems})
        summary["final_consumer_problems"] = final_consumer_problems
        last_seal["F"]["row"] = updated_row
        last_seal["F"]["problems"] = (
            list(last_seal["F"].get("problems", [])) + revoke_problems)
    if revoke_problems:
        summary["gaps"] = list(summary.get("gaps", [])) + revoke_problems
    f["summary"] = summary
    f["conclusion"] = summary["conclusion"]
    return {
        "mode": mode, "conclusion": summary["conclusion"],
        "complete_evidence": bool(summary.get("complete_evidence")),
        "covered_contract_passed": bool(summary.get("covered_contract_passed")),
        "problems": list(summary.get("problems", [])) + revoke_problems,
        "gaps": summary.get("gaps", []),
        "stages": results, "seals": last_seal,
        "summary": summary, "summary_path": path_for_record(f["report"]),
    }


# ---------------------------------------------------------------------------
# authorization / entry guards
# ---------------------------------------------------------------------------

AUTHORIZATION_REQUIRED = (
    "real_asset_loading", "four_window_forward", "output_space_gradient",
    "cohort_and_history", "budgets_and_attempts", "registration_and_seal",
    "synthetic_reverse_tests", "batched_vjp", "tolerance_calibration",
    "independent_verification", "partial_coverage_acknowledged",
)


def authorization_problems(record: Mapping[str, Any] | None) -> list[str]:
    problems: list[str] = []
    if not isinstance(record, Mapping):
        return ["authorization_not_a_mapping"]
    if record.get("authorized") is not True:
        problems.append("not_authorized")
    if record.get("cohort_id") != COHORT_ID:
        problems.append("authorization_cohort")
    if record.get("execution_id") != EXECUTION_ID:
        problems.append("authorization_execution")
    if record.get("training_updates") != 0 or record.get("parameter_fd") != 0:
        problems.append("authorization_exceeds_scope")
    if record.get("ddim_calls") not in (0, None):
        problems.append("authorization_ddim")
    checklist = record.get("checklist")
    if not isinstance(checklist, Mapping):
        problems.append("authorization_checklist_missing")
    else:
        for key in AUTHORIZATION_REQUIRED:
            if checklist.get(key) is not True:
                problems.append(f"authorization_checklist:{key}")
    budgets = record.get("budgets")
    if not isinstance(budgets, Mapping):
        problems.append("authorization_budgets_missing")
    else:
        for stage, spec in STAGE_BUDGETS.items():
            rec = budgets.get(stage)
            if not isinstance(rec, Mapping):
                problems.append(f"authorization_budget:{stage}")
                continue
            if float(rec.get("budget_s", -1)) != spec["budget_s"]:
                problems.append(f"authorization_budget_size:{stage}")
            if int(rec.get("attempts", -1)) != spec["attempts"]:
                problems.append(f"authorization_attempts:{stage}")
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


def load_tolerance_contract(path: Path) -> tuple[dict, str, list[str]]:
    p = Path(path)
    if not p.is_file():
        return {}, "", ["tolerance_contract_missing"]
    try:
        contract = read_json(p)
    except (OSError, ValueError):
        return {}, sha256_file(p), ["tolerance_contract_unreadable"]
    return contract, sha256_file(p), tolerance_contract_problems(contract)


def synthetic_calibration(*, seed: int = 20261026) -> dict:
    """Pure synthetic calibration of the comparison thresholds.

    This is the only producer of a contract with ``mode=synthetic_calibrated``
    and a non-null evidence hash.  It reads no real dataset/cache/checkpoint.
    """
    provider = SyntheticObjectiveProvider(seed=seed)
    keys = provider.select_windows()
    inputs = provider.load_inputs(
        keys, noise=generate_noise(NOISE_SEED,
                                   (WINDOW_COUNT, WINDOW, 38)))
    bundles = provider.model_factory()
    loss_factory = provider.loss_factory()
    base = build_tolerance_contract()
    calls = AuditCallBudget()
    units: dict[str, dict] = {}
    for model in MODELS:
        units[model] = production_objective_unit(
            bundles[model], inputs, loss_factory(model), calls,
            device="cpu", dtype=torch.float32,
            tolerance=base["tolerances"])
    rows = build_objective_rows(units, inputs, base, torch.float32)
    grad_rows = build_gradient_rows(units, base, torch.float32)
    loss_abs = [float(r["abs_error"]) for r in rows
                if _num(r.get("abs_error"))]
    loss_rel = [float(r["relative_error"]) for r in rows
                if _num(r.get("relative_error"))]
    grad_abs = [float(r["max_abs_diff"]) for r in grad_rows
                if _num(r.get("max_abs_diff"))]
    grad_rel = [float(r["relative_error"]) for r in grad_rows
                if _num(r.get("relative_error"))]
    enc = compare_array("encode", inputs.x0, inputs.x0_independent,
                        base["tolerances"]["encode_float32"])
    enc_abs = float(enc.get("max_abs_diff", 0.0))
    denom = np.maximum(np.abs(inputs.x0_independent), 1e-12)
    enc_rel = float(np.max(np.abs(inputs.x0 - inputs.x0_independent) / denom))
    observed = {
        "encode_float32": {"atol": enc_abs, "rtol": 0.0},
        "loss_float32": {"atol": max(loss_abs, default=0.0), "rtol": 0.0},
        "gradient_float32": {"atol": max(grad_abs, default=0.0), "rtol": 0.0},
    }
    evidence = {
        "schema": "w1_real_objective_synthetic_calibration_v1",
        "seed": seed,
        "observed": observed,
        "observed_relative_errors": {
            "encode_float32": enc_rel,
            "loss_float32_max": max(loss_rel, default=0.0),
            "gradient_float32_max": max(grad_rel, default=0.0),
        },
        "loss_rows": len(rows),
        "gradient_rows": len(grad_rows),
        "encoding": enc,
        "calls": calls.snapshot(),
        "real_dataset_loaded": False,
        "real_checkpoint_loaded": False,
        "real_model_calls": 0,
        "boundaries": BOUNDARIES,
    }
    evidence_sha = sha256_bytes(json.dumps(
        _jsonable(evidence), sort_keys=True, separators=(",", ":")).encode())
    contract = build_tolerance_contract(observed=observed,
                                        evidence_sha256=evidence_sha)
    contract["mode"] = "synthetic_calibrated"
    contract["calibration"] = evidence
    contract["calibration_sha256"] = evidence_sha
    return contract


def synthetic_self_test(out_root: Path, *, seed: int = 20261026) -> dict:
    """Run the complete harness on the analytic fixture; no real assets."""
    run_root = Path(out_root).expanduser().resolve()
    if run_root.exists() and any(run_root.iterdir()):
        n = 1
        while (run_root / f"attempt_{n:03d}").exists():
            n += 1
        run_root = run_root / f"attempt_{n:03d}"
    run_root.mkdir(parents=True, exist_ok=True)
    provider = SyntheticObjectiveProvider(seed=seed)
    contract = build_tolerance_contract()
    result = run_registered_audit(
        run_root, provider, mode="synthetic", tolerance=contract,
        fixed_root=HISTORY_ROOT,
    )
    result["run_root"] = path_for_record(run_root)
    result["tolerance_contract"] = contract
    write_json(run_root / "audit_result.json", result)
    return result


# === END PART 6 ===





