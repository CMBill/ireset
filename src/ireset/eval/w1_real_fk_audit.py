"""W1 real training-decode, FK geometry and output-space derivative audit.

Implementation of the I/C scope in
``docs/plans/omniretarget_w1_real_fk_geometry_derivative_audit_prereg_draft.md``.
It is deliberately split into two entry modes:

* ``synthetic`` -- an asset-free analytic fixture (a 29-joint / 15-site chain,
  the production ``LossAssembler.decode_clip`` path with an injected synthetic
  FK, and an independent NumPy/SciPy reference).  It exercises the full
  registration -> geometry -> derivative -> finalize chain, the 122 geometry
  rows and the 4672 finite-difference rows without reading a real dataset,
  cache, checkpoint, G1 asset or MuJoCo model.
* ``real`` -- the registered path.  It refuses to run unless a hash-bound
  authorization record, a frozen synthetic tolerance contract and a passed
  admission package are supplied.  Section 9.2 of the draft is not checked in
  this repository state, so real mode is intentionally unreachable here.

Coverage remains partial even when the covered contract passes:
``full_objective_contract_passed=None``, ``factor_selected=False``,
``auto_execute=False``, ``final_interaction_method="not_ready"`` and
``production_integration=False``.  Nothing here runs an optimizer update,
parameter finite difference, DDIM sample, checkpoint write or network call.

No disk access happens at import time.
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
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRotation

from ireset.embodied.g1_spec import KPT_TO_SMPLH
from ireset.paths import PROJECT_ROOT, path_for_record
from ireset.train.losses import LossAssembler, LossWeights

# ---------------------------------------------------------------------------
# identity and frozen matrix
# ---------------------------------------------------------------------------

AUDIT_SCHEMA = "w1_real_fk_geometry_derivative_audit_v1"
COHORT_ID = "w1_real_fk_geometry_derivative_audit_v1"
EXECUTION_ID = "w1_real_fk_geometry_derivative_audit_v1"
AUTHORIZATION_SCHEMA = "w1_real_fk_derivative_authorization_v1"
TOLERANCE_SCHEMA = "w1_real_fk_derivative_tolerance_contract_v1"
CALIBRATION_SCHEMA = "w1_real_fk_derivative_synthetic_calibration_v1"
ADMISSION_SCHEMA = "w1_real_fk_derivative_admission_package_v1"
REGISTRATION_SCHEMA = "w1_real_fk_derivative_registration_v1"
ACCOUNTING_SCHEMA = "pre_registered_v1"

WINDOW = 60
STRIDE = 30
WINDOW_IDS = ("W0", "W1", "W2", "W3")
MODELS = ("deterministic", "diffusion")
LOGICAL_INPUTS = ("target", "deterministic", "diffusion")
ROLES = ("target_det", "target_diff", "pred_det", "pred_diff")
ROLE_INPUT = {
    "target_det": "target",
    "target_diff": "target",
    "pred_det": "deterministic",
    "pred_diff": "diffusion",
}
ROLE_MODEL = {
    "target_det": "deterministic",
    "target_diff": "diffusion",
    "pred_det": "deterministic",
    "pred_diff": "diffusion",
}
DECODE_KINDS = ("p", "R", "q")
VARIANT = ("raw", "weighted")
SCOPES = (*WINDOW_IDS, "batch")

SITE_NAMES = (
    "kpt_pelvis", "kpt_l_hip", "kpt_r_hip", "kpt_l_knee", "kpt_r_knee",
    "kpt_l_shoulder", "kpt_r_shoulder", "kpt_l_elbow", "kpt_r_elbow",
    "kpt_l_ankle", "kpt_r_ankle", "kpt_l_toe", "kpt_r_toe",
    "kpt_l_wrist", "kpt_r_wrist",
)
N_SITES = len(SITE_NAMES)
N_JOINTS = 29
X0_DIM = 38
TANGENT_DIM = 35
FK_DENOMINATOR_BATCH = 4 * WINDOW * N_SITES
FK_DENOMINATOR_WINDOW = WINDOW * N_SITES

FD_LOCAL_FRAMES = (0, 29, 30, 59)
FD_STEPS = (1e-5, 1e-6)
FD_STATE_COUNT = len(MODELS) * len(WINDOW_IDS) * len(FD_LOCAL_FRAMES)
FD_TANGENT_ROWS = FD_STATE_COUNT * TANGENT_DIM * len(FD_STEPS)
FD_NORMALIZED_ROWS = FD_STATE_COUNT * X0_DIM * len(FD_STEPS)
FD_ROW_COUNT = FD_TANGENT_ROWS + FD_NORMALIZED_ROWS
FD_POSE_EVALUATIONS = 2 * FD_ROW_COUNT

WINDOW_SPECS = (
    ("sub10_largebox_063", 0, 60, 133),
    ("sub10_largebox_063", 30, 90, 133),
    ("sub10_largebox_085", 0, 60, 148),
    ("sub10_largebox_085", 30, 90, 148),
)

STAGE_BUDGETS = {
    "R0": {"budget_s": 180.0, "reserve_s": 30.0, "attempts": 2},
    "G": {"budget_s": 420.0, "reserve_s": 60.0, "attempts": 2},
    "D": {"budget_s": 600.0, "reserve_s": 90.0, "attempts": 2},
    "F": {"budget_s": 600.0, "reserve_s": 90.0, "attempts": 2},
}
TOTAL_BUDGET_S = 1800.0
SEAL_TAIL_RESERVE_S = 1.0
SEAL_VERIFY_ROUNDS = 4
STAGE_DIRS = {"R0": "registration", "G": "geometry", "D": "derivative",
              "F": "finalize"}

#: Nominal call/work matrix for one G, D and F stage of this cohort.  The
#: cumulative caps are the stage maximum (two attempts) totals.
STAGE_NOMINAL = {
    "G": {
        "loss_assembler": 2, "production_fk_batch": 4,
        "kernel64_fk_batch": 3, "production_vjp": 2,
        "mj_forward": 1200, "mj_jac_site": 7200,
        "fd_pose_eval": 0, "scene_setup_forward": 1,
    },
    "D": {
        "loss_assembler": 0, "production_fk_batch": 0,
        "kernel64_fk_batch": 0, "production_vjp": 0,
        "mj_forward": FD_POSE_EVALUATIONS, "mj_jac_site": 0,
        "fd_pose_eval": FD_POSE_EVALUATIONS, "scene_setup_forward": 0,
    },
    "F": {
        "loss_assembler": 2, "production_fk_batch": 4,
        "kernel64_fk_batch": 3, "production_vjp": 2,
        "mj_forward": 1200 + FD_POSE_EVALUATIONS, "mj_jac_site": 7200,
        "fd_pose_eval": FD_POSE_EVALUATIONS, "scene_setup_forward": 1,
    },
}
CALL_CAPS = {
    "loss_assembler": 8,
    "production_fk_batch": 16,
    "kernel64_fk_batch": 12,
    "production_vjp": 8,
    "mj_forward": 42176,
    "mj_jac_site": 28800,
    "fd_pose_eval": 37376,
    "scene_setup_forward": 4,
}
ZERO_COUNTERS = (
    "checkpoint_loads", "conditioner_calls", "model_calls",
    "parameter_backward", "parameter_fd", "optimizer_steps",
    "training_updates", "ddim_calls", "checkpoint_writes",
    "contact_branch_executions",
)

PARENT_RUN = Path(
    "derived/experiments/omniretarget_first/"
    "20261027_b45cd7c_w1_real_objective_audit"
)
PARENT_FILES = {
    "registration": (
        PARENT_RUN / "registration/attempt_001/registration.json",
        "ecc11368219a33c95a6038769398fd11789dcdd04430760f54ae3a84ba58a66a",
    ),
    "admission": (
        PARENT_RUN / "registration/attempt_001/admission.json",
        "61c5b16d883d0fa084632a548bd7428e2fe580540990feed47926546bd32ce95",
    ),
    "p_report": (
        PARENT_RUN / "production/attempt_001/p_report.json",
        "1ea15ddc2da7a0acb5125f1ecc92e1ea7e0cc74589e7a7ad5055b26055b7cdcb",
    ),
    "objective_arrays": (
        PARENT_RUN / "production/attempt_001/objective_real_arrays.npz",
        "4cf20acb13fcdee207e0492a3b07818cc9ed31202409177fa1ed3f13086fd2c8",
    ),
    "gradient_arrays": (
        PARENT_RUN / "production/attempt_001/gradient_real_arrays.npz",
        "49b21c6c5d82a19e92f2c0d870b1d7f036986604c009c6195e6f86a035765ea7",
    ),
    "audit_summary": (
        PARENT_RUN / "finalize/attempt_001/audit_summary.json",
        "0a839071125a156273b3d579a564965cd0edb196b8f900b05417edd0cb00db97",
    ),
    "independent_verification": (
        PARENT_RUN / "finalize/attempt_001/independent_verification.json",
        "f0bf351bd0bd184f52198ac481293bae9127607ec98864010768430a4dcbd961",
    ),
}
HISTORY_ROOT = Path("derived/experiments/omniretarget_first")
TRAIN_CONFIG = Path("configs/train/omniretarget_motion.yaml")
LOCKED_ASSETS = {
    "urdf": "derived/robots/g1_29dof_locked.urdf",
    "xml": "derived/robots/g1_29dof_locked.xml",
    "scene_xml": "derived/robots/g1_29dof_locked_w_largebox.xml",
    "lock_record": "derived/robots/lock_record.json",
}
AUDIT_ROOT = Path("derived/audit/w1_real_fk_audit")

#: Tolerance proposals from draft section 3.2.  Calibration may only keep or
#: lower the *effective* bound; it can never widen a proposal.  ``observed``
#: records the worst residual *excess* over the proposal's relative term, so
#: the 4x rule is applied to the same combined formula the real gate uses.
TOLERANCE_PROPOSALS = {
    "decode_float32": {"atol": 2e-6, "rtol": 2e-6},
    "decode_rotation_matrix": {"atol": 2e-5, "rtol": 0.0},
    "kernel64_fk": {"atol": 1e-6, "rtol": 1e-6},
    "production_fk": {"atol": 5e-6, "rtol": 2e-6},
    "fk_scalar": {"atol": 1e-8, "rtol": 2e-5},
    "j35_fd": {"atol": 2e-6, "rtol": 2e-5},
    "j38_fd": {"atol": 1e-5, "rtol": 1e-4},
    "fk_gradient": {"atol": 1e-8, "rtol": 1e-4},
}
DOMAIN = {
    "min_6d_a_norm": 1e-6,
    "min_6d_b_perp": 1e-6,
    "min_6d_cross": 1e-12,
}
GROUP_GATE = {
    "relative_l2": 2e-4,
    "cosine_min": 1.0 - 1e-6,
    "tiny_abs": 1e-12,
    "nonzero_reference_norm": 1e-10,
}

BOUNDARIES = {
    "objective_coverage": "partial",
    "full_objective_contract_passed": None,
    "attribution": "attribution_insufficient",
    "factor_selected": False,
    "auto_execute": False,
    "final_interaction_method": "not_ready",
    "production_integration": False,
    "training_updates": 0,
    "parameter_fd": 0,
    "ddim_calls": 0,
    "checkpoint_writes": 0,
}

# ---------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return path_for_record(value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True,
                              allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _num(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _nonneg(value: Any) -> bool:
    return _num(value) and float(value) >= 0.0


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")).encode())


def _hash_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return sha256_bytes(a.tobytes())


# ---------------------------------------------------------------------------
# fixed windows and key generators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowKey:
    window_id: str
    demo: str
    start: int
    end: int
    T: int
    split: str = "train"

    @property
    def local_key(self) -> str:
        return f"{self.demo}|{self.start}-{self.end}|T{self.T}"

    def as_dict(self) -> dict:
        return {
            "window_id": self.window_id, "demo": self.demo,
            "start": int(self.start), "end": int(self.end), "T": int(self.T),
            "split": self.split, "local_key": self.local_key,
        }


def fixed_windows() -> tuple[WindowKey, ...]:
    return tuple(WindowKey(wid, demo, start, end, T)
                 for wid, (demo, start, end, T) in zip(WINDOW_IDS, WINDOW_SPECS))


def _row_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)


def required_geometry_rows() -> list[str]:
    rows: list[str] = []
    for role in ROLES:
        for wid in WINDOW_IDS:
            for kind in DECODE_KINDS:
                rows.append(_row_key("decode", role, wid, kind))
    for role in ROLES:
        for wid in WINDOW_IDS:
            rows.append(_row_key("production_fk", role, wid))
    for inp in LOGICAL_INPUTS:
        for wid in WINDOW_IDS:
            rows.append(_row_key("kernel64_fk", inp, wid))
    for role in ROLES:
        for wid in WINDOW_IDS:
            rows.append(_row_key("parent_fk_binding", role, wid))
    for model in MODELS:
        for scope in SCOPES:
            rows.append(_row_key("fk_scalar", model, scope))
    for model in MODELS:
        for variant in VARIANT:
            for scope in SCOPES:
                rows.append(_row_key("fk_gradient", model, variant, scope))
    return rows


def fd_states() -> list[dict]:
    states: list[dict] = []
    windows = {w.window_id: w for w in fixed_windows()}
    for model in MODELS:
        for wid in WINDOW_IDS:
            w = windows[wid]
            for local in FD_LOCAL_FRAMES:
                states.append({
                    "state_key": _row_key(model, wid, local),
                    "model": model, "window_id": wid, "local_frame": int(local),
                    "abs_frame": int(w.start + local),
                })
    return states


def required_fd_rows() -> list[str]:
    rows: list[str] = []
    for state in fd_states():
        for h in FD_STEPS:
            for dim in range(TANGENT_DIM):
                rows.append(_row_key(
                    "fd", "tangent35", state["state_key"], f"{dim:02d}", h))
            for dim in range(X0_DIM):
                rows.append(_row_key(
                    "fd", "normalized38", state["state_key"], f"{dim:02d}", h))
    return rows


def geometry_array_schema() -> dict:
    schema: dict[str, dict] = {
        "x0": {"dtype": "float32", "shape": [4, 60, 38]},
        "x0_independent": {"dtype": "float64", "shape": [4, 60, 38]},
        "prediction_deterministic": {"dtype": "float32", "shape": [4, 60, 38]},
        "prediction_diffusion": {"dtype": "float32", "shape": [4, 60, 38]},
        "root_mean": {"dtype": "float64", "shape": [3]},
        "root_std": {"dtype": "float64", "shape": [3]},
        "joint_lo": {"dtype": "float64", "shape": [29]},
        "joint_hi": {"dtype": "float64", "shape": [29]},
    }
    for role in ROLES:
        schema[f"decode_pos_{role}"] = {"dtype": "float32", "shape": [4, 60, 3]}
        schema[f"decode_quat_{role}"] = {"dtype": "float32", "shape": [4, 60, 4]}
        schema[f"decode_R_{role}"] = {"dtype": "float32", "shape": [4, 60, 3, 3]}
        schema[f"decode_q_{role}"] = {"dtype": "float32", "shape": [4, 60, 29]}
        schema[f"ref_decode_pos_{role}"] = {"dtype": "float64", "shape": [4, 60, 3]}
        schema[f"ref_decode_R_{role}"] = {"dtype": "float64", "shape": [4, 60, 3, 3]}
        schema[f"ref_decode_q_{role}"] = {"dtype": "float64", "shape": [4, 60, 29]}
        schema[f"prod_fk_{role}"] = {"dtype": "float32", "shape": [4, 60, 15, 3]}
        schema[f"ref_fk_{role}"] = {"dtype": "float64", "shape": [4, 60, 15, 3]}
        schema[f"parent_fk_{role}"] = {"dtype": "float32", "shape": [4, 60, 15, 3]}
    for inp in LOGICAL_INPUTS:
        schema[f"kernel64_fk_{inp}"] = {"dtype": "float64",
                                        "shape": [4, 60, 15, 3]}
    for model in MODELS:
        schema[f"fk_elements_{model}"] = {"dtype": "float64", "shape": [4, 60, 15]}
        schema[f"fk_scalar_production_{model}"] = {"dtype": "float64", "shape": [5]}
        schema[f"fk_scalar_reference_{model}"] = {"dtype": "float64", "shape": [5]}
    return schema


def gradient_array_schema() -> dict:
    schema: dict[str, dict] = {}
    for model in MODELS:
        schema[f"J35_{model}"] = {"dtype": "float64",
                                  "shape": [4, 60, 15, 3, 35]}
        schema[f"J38_{model}"] = {"dtype": "float64",
                                  "shape": [4, 60, 15, 3, 38]}
        schema[f"D_decode_{model}"] = {"dtype": "float64",
                                       "shape": [4, 60, 35, 38]}
        schema[f"domain_6d_{model}"] = {"dtype": "bool", "shape": [4, 60]}
        schema[f"domain_clamp_{model}"] = {"dtype": "bool", "shape": [4, 60, 29]}
        schema[f"grad_production_raw_{model}"] = {"dtype": "float32",
                                                  "shape": [4, 60, 38]}
        schema[f"grad_independent_raw_{model}"] = {"dtype": "float64",
                                                   "shape": [4, 60, 38]}
        schema[f"grad_independent_weighted_{model}"] = {
            "dtype": "float64", "shape": [4, 60, 38]}
        schema[f"fk_elements_{model}"] = {"dtype": "float64", "shape": [4, 60, 15]}
        schema[f"fk_target_points_{model}"] = {"dtype": "float32",
                                               "shape": [4, 60, 15, 3]}
        schema[f"fk_pred_points_{model}"] = {"dtype": "float32",
                                             "shape": [4, 60, 15, 3]}
    return schema


def fd_array_schema() -> dict:
    schema: dict[str, dict] = {}
    for offset, state in enumerate(fd_states()):
        for family, dim in (("tangent35", TANGENT_DIM),
                            ("normalized38", X0_DIM)):
            schema[f"{family}_analytic_{offset:02d}"] = {
                "dtype": "float64", "shape": [15, 3, dim]}
            schema[f"{family}_ok_{offset:02d}"] = {"dtype": "bool",
                                                   "shape": [dim]}
            for h in FD_STEPS:
                schema[f"{family}_fd_{offset:02d}_{h:g}"] = {
                    "dtype": "float64", "shape": [15, 3, dim]}
    return schema


# ---------------------------------------------------------------------------
# independent rotation / decode mathematics
# ---------------------------------------------------------------------------


def _skew_np(v: np.ndarray) -> np.ndarray:
    z = np.zeros(v.shape[:-1])
    return np.stack([
        np.stack([z, -v[..., 2], v[..., 1]], -1),
        np.stack([v[..., 2], z, -v[..., 0]], -1),
        np.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def _cross_columns(dx: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``dx`` [N,3,3] columns cross ``y`` [N,3] -> [N,3,3]."""
    return np.swapaxes(np.cross(np.swapaxes(dx, 1, 2), y[:, None, :]), 1, 2)


def _cross_left(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """``y`` [N,3] cross columns of ``dx`` [N,3,3] -> [N,3,3]."""
    return np.swapaxes(np.cross(y[:, None, :], np.swapaxes(dx, 1, 2)), 1, 2)


def rot6d_domain_np(r6: np.ndarray, *, domain: Mapping[str, float] = DOMAIN
                    ) -> dict:
    r6 = np.asarray(r6, dtype=np.float64)
    a, b = r6[..., 0:3], r6[..., 3:6]
    an = np.linalg.norm(a, axis=-1)
    finite = np.isfinite(r6).all(axis=-1)
    safe = np.maximum(an, 1e-300)
    e0 = a / safe[..., None]
    d = np.sum(e0 * b, axis=-1)
    u = b - d[..., None] * e0
    v = np.linalg.norm(u, axis=-1)
    e1 = u / np.maximum(v, 1e-300)[..., None]
    e2 = np.cross(e0, e1)
    cross_n = np.linalg.norm(e2, axis=-1)
    ok = (finite
          & (an > float(domain["min_6d_a_norm"]))
          & (v > float(domain["min_6d_b_perp"]))
          & (cross_n > float(domain["min_6d_cross"])))
    return {"e0": e0, "e1": e1, "e2": e2, "a_norm": an, "b_perp": v,
            "cross_norm": cross_n, "ok": ok, "u": u}


def independent_rot6d_to_mat(r6: np.ndarray, *, domain: Mapping[str, float] = DOMAIN
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Two-column Gram-Schmidt; invalid rows are marked, never silently zero."""
    info = rot6d_domain_np(r6, domain=domain)
    R = np.stack([info["e0"], info["e1"], info["e2"]], axis=-1)
    ok = info["ok"]
    if not ok.all():
        R = np.where(ok[..., None, None], R, np.eye(3)[None])
    return R, ok


def mat_to_quat_wxyz_np(R: np.ndarray) -> np.ndarray:
    """Stable matrix -> wxyz quaternion via scipy; xyzw->wxyz explicit."""
    quat_xyzw = SciRotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    return np.roll(quat_xyzw, 1, axis=-1)


def quat_to_mat_wxyz_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return SciRotation.from_quat(np.roll(q, -1, axis=-1)).as_matrix()


def so3_exp_np(w: np.ndarray) -> np.ndarray:
    return SciRotation.from_rotvec(np.asarray(w, dtype=np.float64)).as_matrix()


def independent_decode(x0: np.ndarray, mean: np.ndarray, std: np.ndarray,
                       lo: np.ndarray, hi: np.ndarray,
                       *, domain: Mapping[str, float] = DOMAIN) -> dict:
    """Independent float64 decode of the 38-dim normalized state.

    The caller passes the **effective float32 constants** upcast to float64.
    """
    x = np.asarray(x0, dtype=np.float64).reshape(-1, X0_DIM)
    pos = x[:, 0:3] * np.asarray(std, dtype=np.float64) \
        + np.asarray(mean, dtype=np.float64)
    R, ok6d = independent_rot6d_to_mat(x[:, 3:9], domain=domain)
    quat = mat_to_quat_wxyz_np(R)
    u = x[:, 9:38]
    inside = np.abs(u) < 1.0
    boundary = np.abs(u) == 1.0
    q = np.asarray(lo, dtype=np.float64)[None, :] \
        + (np.clip(u, -1.0, 1.0) + 1.0) \
        * (np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64))[None, :] / 2.0
    finite = np.isfinite(x).all(axis=1)
    return {
        "pos": pos, "R": R, "quat": quat, "q": q,
        "ok": ok6d & finite, "ok_6d": ok6d, "finite": finite,
        "clamp_inside": inside, "clamp_boundary": boundary,
    }


def decode_jacobian_np(x0: np.ndarray, std: np.ndarray, lo: np.ndarray,
                       hi: np.ndarray, *, domain: Mapping[str, float] = DOMAIN
                       ) -> tuple[np.ndarray, dict]:
    """Analytic ``d xi / d x`` with ``xi = [dp(3), dtheta(3), dq(29)]``.

    ``dtheta`` is the left-multiplied clip-frame rotation increment
    (``R_next = Exp(dtheta) R``).  The joint block is ``(hi-lo)/2`` strictly
    inside the clamp, zero strictly outside, and a domain gap exactly at
    ``|x_j| == 1``.
    """
    x = np.asarray(x0, dtype=np.float64).reshape(-1, X0_DIM)
    n = x.shape[0]
    D = np.zeros((n, TANGENT_DIM, X0_DIM))
    I3 = np.eye(3)
    D[:, 0:3, 0:3] = np.broadcast_to(np.diag(np.asarray(std, dtype=np.float64)),
                                    (n, 3, 3))
    a, b = x[:, 3:6], x[:, 6:9]
    info = rot6d_domain_np(x[:, 3:9], domain=domain)
    an = np.maximum(info["a_norm"], 1e-300)
    e0, e1 = info["e0"], info["e1"]
    d = np.sum(e0 * b, axis=-1)
    v = np.maximum(info["b_perp"], 1e-300)
    outer0 = e0[:, :, None] * e0[:, None, :]
    outer1 = e1[:, :, None] * e1[:, None, :]
    de0_da = (I3[None] - outer0) / an[:, None, None]
    de0_db = np.zeros((n, 3, 3))
    du_da = -(e0[:, :, None] * b[:, None, :] + d[:, None, None] * I3[None]) @ de0_da
    du_db = I3[None] - outer0
    de1_da = (I3[None] - outer1) @ du_da / v[:, None, None]
    de1_db = (I3[None] - outer1) @ du_db / v[:, None, None]
    de2_da = _cross_columns(de0_da, e1) + _cross_left(e0, de1_da)
    de2_db = _cross_left(e0, de1_db)
    dR_da = np.stack([de0_da, de1_da, de2_da], axis=2)
    dR_db = np.stack([de0_db, de1_db, de2_db], axis=2)
    R = np.stack([e0, e1, e2 := info["e2"]], axis=-1)
    Sa = np.einsum("nclk,nol->ncok", dR_da, R)
    Sb = np.einsum("nclk,nol->ncok", dR_db, R)
    theta_a = np.stack([Sa[:, 2, 1, :], Sa[:, 0, 2, :], Sa[:, 1, 0, :]], axis=1)
    theta_b = np.stack([Sb[:, 2, 1, :], Sb[:, 0, 2, :], Sb[:, 1, 0, :]], axis=1)
    D[:, 3:6, 3:9] = np.concatenate([theta_a, theta_b], axis=2)
    uj = x[:, 9:38]
    inside = np.abs(uj) < 1.0
    boundary = np.abs(uj) == 1.0
    scale = (np.asarray(hi, dtype=np.float64)
             - np.asarray(lo, dtype=np.float64)) / 2.0
    for k in range(N_JOINTS):
        D[:, 6 + k, 9 + k] = np.where(inside[:, k], scale[k], 0.0)
    ok = info["ok"] & np.isfinite(x).all(axis=1) & ~boundary.any(axis=1)
    return D, {
        "ok": ok, "ok_6d": info["ok"], "finite": np.isfinite(x).all(axis=1),
        "clamp_inside": inside, "clamp_outside": np.abs(uj) > 1.0,
        "clamp_boundary": boundary,
    }


def effective_constants(mean: np.ndarray, std: np.ndarray, lo: np.ndarray,
                        hi: np.ndarray) -> dict:
    """Cast constants exactly as the production constructor does (float32)."""
    return {
        "mean32": np.asarray(mean, dtype=np.float32),
        "std32": np.asarray(std, dtype=np.float32),
        "lo32": np.asarray(lo, dtype=np.float32),
        "hi32": np.asarray(hi, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# synthetic 29-joint / 15-site chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticChain:
    axes: np.ndarray          # [29,3] unit
    offsets: np.ndarray       # [30,3], body i origin in parent frame
    site_bodies: tuple[int, ...]
    site_offsets: np.ndarray  # [15,3]

    def as_dict(self) -> dict:
        return {
            "n_joints": int(self.axes.shape[0]),
            "n_sites": len(self.site_bodies),
            "axes_sha256": _hash_array(self.axes),
            "offsets_sha256": _hash_array(self.offsets),
            "site_offsets_sha256": _hash_array(self.site_offsets),
            "site_bodies": list(self.site_bodies),
        }


def build_synthetic_chain() -> SyntheticChain:
    i = np.arange(N_JOINTS, dtype=np.float64)
    raw = np.stack([np.cos(0.7 * i), np.sin(0.7 * i),
                    0.3 + 0.1 * ((i % 5.0) - 2.0)], axis=1)
    axes = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    offsets = np.zeros((N_JOINTS + 1, 3))
    for k in range(N_JOINTS):
        offsets[k + 1] = [0.04 + 0.01 * (k % 3),
                          0.02 * ((k % 4) - 1.5),
                          0.03 * ((k % 5) - 2.0)]
    site_bodies = tuple(range(1, N_SITES + 1))
    site_offsets = np.stack([
        [0.010 + 0.002 * k, 0.005 * ((k % 3) - 1),
         0.020 * ((k % 4) - 1.5)] for k in range(N_SITES)
    ])
    return SyntheticChain(axes=axes, offsets=offsets,
                          site_bodies=site_bodies, site_offsets=site_offsets)


def synthetic_joint_limits() -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(N_JOINTS)
    lo = -1.20 - 0.05 * (k % 4)
    hi = 1.10 + 0.05 * (k % 3)
    return lo.astype(np.float64), hi.astype(np.float64)


def _chain_forward_np(chain: SyntheticChain, pos: np.ndarray, R: np.ndarray,
                      q: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = pos.shape[0]
    Rs = [np.asarray(R, dtype=np.float64).reshape(n, 3, 3)]
    ps = [np.asarray(pos, dtype=np.float64).reshape(n, 3)]
    q = np.asarray(q, dtype=np.float64).reshape(n, N_JOINTS)
    for k in range(N_JOINTS):
        w = chain.axes[k][None, :] * q[:, k:k + 1]
        Rj = so3_exp_np(w)
        Rp, pp = Rs[-1], ps[-1]
        Rs.append(Rp @ Rj)
        ps.append(pp + np.einsum("nij,j->ni", Rp, chain.offsets[k + 1]))
    return Rs, ps


def synthetic_fk_np(chain: SyntheticChain, pos: np.ndarray, R: np.ndarray,
                    q: np.ndarray) -> np.ndarray:
    Rs, ps = _chain_forward_np(chain, pos, R, q)
    out = []
    for body, off in zip(chain.site_bodies, chain.site_offsets):
        out.append(ps[body] + np.einsum("nij,j->ni", Rs[body], off))
    return np.stack(out, axis=1)


def synthetic_fk_jac_np(chain: SyntheticChain, pos: np.ndarray, R: np.ndarray,
                        q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 15 site positions and ``J_nv`` in the MuJoCo local-omega form.

    Column blocks: translation (3), free-joint local angular velocity (3),
    then 29 revolute joints.  ``J35 = J_nv @ blockdiag(I3, R.T, I29)``.
    """
    Rs, ps = _chain_forward_np(chain, pos, R, q)
    R_root = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    p_root = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    n = R_root.shape[0]
    nv = 3 + 3 + N_JOINTS
    kp = np.zeros((n, N_SITES, 3))
    J = np.zeros((n, N_SITES, 3, nv))
    for s, (body, off) in enumerate(zip(chain.site_bodies, chain.site_offsets)):
        x = ps[body] + np.einsum("nij,j->ni", Rs[body], off)
        kp[:, s] = x
        J[:, s, :, 0:3] = np.eye(3)
        r_root = np.einsum("nji,nj->ni", R_root, x - p_root)
        J[:, s, :, 3:6] = -np.einsum("nij,njk->nik", R_root, _skew_np(r_root))
        for k in range(N_JOINTS):
            if body <= k:
                continue  # site is upstream of this joint
            omega = np.einsum("nij,j->ni", Rs[k], chain.axes[k])
            # The fixed offset is applied before the joint rotation, so the
            # rotation axis passes through the child origin ``ps[k+1]``.
            J[:, s, :, 6 + k] = np.cross(omega, x - ps[k + 1])
    return kp, J


def synthetic_fk_torch(chain: SyntheticChain, pos: torch.Tensor,
                       quat_wxyz: torch.Tensor, q: torch.Tensor,
                       site_names: Sequence[str] | None = None) -> torch.Tensor:
    """Production-side synthetic FK stand-in (differentiable, batched)."""
    from ireset.kinematics.rotation_adapter import so3_exp_torch
    from ireset.model.tokens import quat_to_mat_torch

    names = list(site_names if site_names is not None else SITE_NAMES)
    index = {name: i for i, name in enumerate(SITE_NAMES)}
    order = [index[name] for name in names]
    R_root = quat_to_mat_torch(quat_wxyz)
    Rs = [R_root]
    ps = [pos]
    axes = torch.as_tensor(chain.axes, dtype=pos.dtype, device=pos.device)
    offs = torch.as_tensor(chain.offsets, dtype=pos.dtype, device=pos.device)
    for k in range(N_JOINTS):
        w = axes[k][None, :] * q[:, k:k + 1]
        Rj = so3_exp_torch(w)
        Rp, pp = Rs[-1], ps[-1]
        Rs.append(Rp @ Rj)
        ps.append(pp + torch.einsum("nij,j->ni", Rp, offs[k + 1]))
    out = []
    for s in order:
        body = chain.site_bodies[s]
        off = torch.as_tensor(chain.site_offsets[s], dtype=pos.dtype,
                              device=pos.device)
        out.append(ps[body] + torch.einsum("nij,j->ni", Rs[body], off))
    return torch.stack(out, dim=1)


def fk_loss_and_gradient(J38: np.ndarray, kp_pred: np.ndarray,
                         kp_target: np.ndarray) -> dict:
    """Independent FK scalar, raw/weighted gradient and per-site residuals.

    ``elements`` has shape [N, 15] (per-site squared distance).  The batch
    denominator is ``N_sites * N``; with N=240 this is the registered 3600.
    Production ``fk_m2`` is exactly the mean of these elements (not 15x).
    """
    e = np.asarray(kp_pred, dtype=np.float64) - np.asarray(kp_target, dtype=np.float64)
    elements = np.sum(e * e, axis=-1)
    denominator = float(elements.size)
    l_fk = float(elements.mean())
    g_lfk = np.einsum("nsck,nsc->nk", np.asarray(J38, dtype=np.float64), e) \
        * (2.0 / denominator)
    return {
        "elements": elements,
        "denominator": denominator,
        "l_fk": l_fk,
        "l_fk_weighted": 2.0 * l_fk,
        "grad_lfk": g_lfk,
        "grad_raw": g_lfk,
        "grad_weighted": 2.0 * g_lfk,
    }


def gradient_group_stats(got: np.ndarray, ref: np.ndarray,
                         clamped: np.ndarray | None = None) -> dict:
    """Element-wise and grouped checks for a 38-dim input gradient."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        return {"problems": [f"shape:{got.shape}!={ref.shape}"], "groups": {}}
    groups: dict[str, dict] = {}
    problems: list[str] = []
    for name, sl in (("p", slice(0, 3)), ("rot6d", slice(3, 9)),
                     ("joint", slice(9, 38))):
        g = got[..., sl].reshape(-1)
        r = ref[..., sl].reshape(-1)
        diff = g - r
        rn = float(np.linalg.norm(r))
        gn = float(np.linalg.norm(g))
        rel = float(np.linalg.norm(diff) / rn) if rn >= GROUP_GATE["nonzero_reference_norm"] else 0.0
        cos = (float(np.dot(g, r) / (gn * rn))
               if gn > 0.0 and rn > 0.0 else 1.0)
        record = {
            "reference_norm": rn, "production_norm": gn,
            "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "relative_l2": rel, "cosine": cos,
            "all_zero_production": bool(gn == 0.0),
            "all_zero_reference": bool(rn == 0.0),
        }
        if rn >= GROUP_GATE["nonzero_reference_norm"]:
            if not (rel <= GROUP_GATE["relative_l2"]
                    and cos >= GROUP_GATE["cosine_min"] and gn > 0.0):
                problems.append(f"group:{name}")
        elif record["max_abs_diff"] > GROUP_GATE["tiny_abs"]:
            problems.append(f"tiny_group:{name}")
        groups[name] = record
    tiny_problems = []
    if clamped is not None:
        clamped = np.asarray(clamped, dtype=bool)
        if clamped.any():
            got_c = got[..., 9:38][clamped]
            max_got = float(np.max(np.abs(got_c))) if got_c.size else 0.0
            max_ref = float(np.max(np.abs(ref[..., 9:38][clamped]))) if clamped.any() else 0.0
            groups["clamped"] = {
                "count": int(clamped.sum()), "production_max_abs": max_got,
                "reference_max_abs": max_ref,
            }
            if max_got > GROUP_GATE["tiny_abs"] or max_ref > GROUP_GATE["tiny_abs"]:
                tiny_problems.append("clamped_nonzero")
    if tiny_problems:
        problems += tiny_problems
    return {"problems": problems, "groups": groups}


# ---------------------------------------------------------------------------
# call accounting
# ---------------------------------------------------------------------------


class FkCallBudgetExceeded(RuntimeError):
    pass


@dataclass
class CallCounter:
    caps: Mapping[str, int]
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for kind in self.caps:
            self.counts.setdefault(kind, 0)

    def add(self, kind: str, n: int = 1) -> None:
        if kind not in self.caps:
            raise KeyError(f"unknown call kind {kind}")
        if not _count(int(n)) or int(n) < 0:
            raise ValueError(f"invalid call increment {n}")
        value = self.counts[kind] + int(n)
        if value > int(self.caps[kind]):
            raise FkCallBudgetExceeded(
                f"{kind} cumulative cap exceeded: {value} > {self.caps[kind]}")
        self.counts[kind] = value

    def snapshot(self) -> dict:
        return {"schema": "w1_real_fk_call_budget_v1",
                "counts": {k: int(v) for k, v in self.counts.items()},
                "within_caps": all(self.counts[k] <= int(self.caps[k])
                                   for k in self.caps)}


@dataclass
class FkInputs:
    window_keys: list[WindowKey]
    x0: np.ndarray
    prediction: dict[str, np.ndarray]
    x0_independent: np.ndarray
    root_mean: np.ndarray
    root_std: np.ndarray
    joint_lo: np.ndarray
    joint_hi: np.ndarray
    parent_points: dict[str, np.ndarray] = field(default_factory=dict)
    parent_elements: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ProductionPose:
    pos: np.ndarray
    quat: np.ndarray
    q: np.ndarray
    kp: np.ndarray


@dataclass
class ProductionRun:
    target: ProductionPose
    pred: ProductionPose
    fk_m2: float
    fk_elements: np.ndarray
    gradient: np.ndarray | None
    parts: dict
    #: live ``observer['term_tensors']['fk_m2']`` value (the tensor consumed by
    #: the autograd VJP), kept distinct from the reported float and from the
    #: element reduction so the three can be cross-checked.
    fk_m2_live: float = math.nan
    fk_m2_bits: str | None = None


class FkProvider(Protocol):
    mode: str

    def select_windows(self) -> list[WindowKey]: ...
    def load_inputs(self) -> FkInputs: ...
    def production_capture(self, model: str, pred: np.ndarray,
                           *, want_gradient: bool = True) -> ProductionRun: ...
    def reference_positions(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray: ...
    def reference_jacobians(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray: ...
    def kernel64_positions(self, pos: np.ndarray, R: np.ndarray,
                           q: np.ndarray) -> np.ndarray: ...
    def counters(self) -> dict: ...
    def metadata(self) -> dict: ...


# ---------------------------------------------------------------------------
# synthetic provider (asset free)
# ---------------------------------------------------------------------------


class _SyntheticFkAdapter:
    def __init__(self, chain: SyntheticChain) -> None:
        self.chain = chain

    def site_world_positions(self, pos, quat_wxyz, q, site_names):
        return synthetic_fk_torch(self.chain, pos, quat_wxyz, q, site_names)


class _CountingFk:
    """Count production FK batch calls at the actual call site.

    ``_capture_run`` must not add a fixed number: a provider or adapter path
    that skips or duplicates a FK evaluation has to show up in the ledger.
    """

    def __init__(self, fk: Any, owner: Any) -> None:
        self._fk = fk
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fk, name)

    def site_world_positions(self, *args: Any, **kwargs: Any) -> Any:
        counter = getattr(self._owner, "_counter", None) if self._owner is not None else None
        if counter is not None:
            counter.add("production_fk_batch", 1)
        return self._fk.site_world_positions(*args, **kwargs)



def _random_rotation_features(rng: np.random.Generator,
                              shape: Sequence[int]
                              ) -> tuple[np.ndarray, np.ndarray]:
    """SO(3) samples away from the near-pi branch (a registered negative)."""
    axis = rng.normal(size=(*shape, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = rng.uniform(0.0, math.pi - 0.25, size=shape)
    R = so3_exp_np(axis * angle[..., None])
    r6 = np.concatenate([R[..., 0], R[..., 1]], axis=-1)
    return R, r6


def _synthetic_fixture(chain: SyntheticChain, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x0 = np.zeros((4, WINDOW, X0_DIM), dtype=np.float32)
    x0[..., 0:3] = rng.normal(0.0, 0.35, (4, WINDOW, 3))
    R, r6 = _random_rotation_features(rng, (4, WINDOW))
    x0[..., 3:9] = r6
    x0[..., 9:38] = rng.uniform(-0.85, 0.85, (4, WINDOW, N_JOINTS))
    preds = {}
    for model, sigma in (("deterministic", 0.010), ("diffusion", 0.016)):
        pred = x0 + rng.normal(0.0, sigma, x0.shape).astype(np.float32)
        pred[..., 9:38] = np.clip(pred[..., 9:38], -1.25, 1.25)
        preds[model] = pred.astype(np.float32)
    # Deliberate clamp saturations: derivative must be zero there.
    preds["deterministic"][0, 0, 9] = 1.05
    preds["deterministic"][0, 1, 10] = -1.05
    preds["diffusion"][0, 0, 11] = 1.05
    preds["diffusion"][0, 1, 12] = -1.05
    mean = np.array([0.12, -0.18, 0.72], dtype=np.float64)
    std = np.array([0.31, 0.27, 0.19], dtype=np.float64)
    lo, hi = synthetic_joint_limits()
    x0_independent = independent_encode_for_fixture(x0, mean, std, lo, hi)
    return {"x0": x0, "prediction": preds, "root_mean": mean,
            "root_std": std, "joint_lo": lo, "joint_hi": hi,
            "x0_independent": x0_independent}


def independent_encode_for_fixture(x0: np.ndarray, mean: np.ndarray,
                                   std: np.ndarray, lo: np.ndarray,
                                   hi: np.ndarray) -> np.ndarray:
    """Re-encode the physical decode with the independent formulas."""
    dec = independent_decode(x0, mean, std, lo, hi)
    pos_n = (dec["pos"] - mean) / std
    R = dec["R"]
    r6 = np.concatenate([R[..., 0], R[..., 1]], axis=-1)
    u = 2.0 * (dec["q"] - lo) / (hi - lo) - 1.0
    return np.concatenate([pos_n, r6, u], axis=-1)


def _make_synthetic_assembler(chain: SyntheticChain, mean: np.ndarray,
                              std: np.ndarray, lo: np.ndarray,
                              hi: np.ndarray, *, owner: Any = None
                              ) -> LossAssembler:
    from ireset.model.tokens import RootStats, StateTokenizer

    tokenizer = StateTokenizer(
        np.stack([lo, hi], axis=1), RootStats(mean=mean, std=std))
    weights = LossWeights(w_fk=2.0, w_contact=0.0)
    fk: Any = _SyntheticFkAdapter(chain)
    if owner is not None:
        fk = _CountingFk(fk, owner)
    return LossAssembler(tokenizer, weights, "cpu", fk=fk,
                         static_frame=(np.eye(3), np.zeros(3)))


def _capture_run(assembler: LossAssembler, x0: np.ndarray, pred: np.ndarray,
                 counter: CallCounter | None,
                 *, want_gradient: bool) -> ProductionRun:
    if counter is not None:
        counter.add("loss_assembler", 1)
    # The assembler owns the device of every leaf and batch tensor; creating
    # CPU leaves against a CUDA assembler is a device mismatch, not a valid
    # audit path.
    device = getattr(assembler, "device", "cpu")
    pred_leaf = torch.tensor(np.asarray(pred, dtype=np.float32),
                             dtype=torch.float32, device=device,
                             requires_grad=want_gradient)
    x0_t = torch.tensor(np.asarray(x0, dtype=np.float32), dtype=torch.float32,
                        device=device)
    observer: dict = {}
    loss, parts = assembler(pred_leaf, {"x0": x0_t}, observer=observer)
    gradient = None
    if want_gradient:
        if counter is not None:
            counter.add("production_vjp", 1)
        gradient = torch.autograd.grad(
            observer["term_tensors"]["fk_m2"], pred_leaf)[0]
        gradient = gradient.detach().cpu().numpy()

    def pose(prefix: str) -> ProductionPose:
        pos, quat, q = observer[f"decode_{prefix}"]
        return ProductionPose(
            pos=pos.detach().cpu().numpy(),
            quat=quat.detach().cpu().numpy(),
            q=q.detach().cpu().numpy(),
            kp=observer[f"fk_points_{prefix}"].detach().cpu().numpy(),
        )

    elements = observer["fk_elements"].detach().cpu().numpy().astype(np.float64)
    scalar = float(parts["fk_m2"])
    live_tensor = observer.get("term_tensors", {}).get("fk_m2")
    if live_tensor is None:
        raise ValueError("production live fk_m2 tensor missing")
    live = float(live_tensor.detach().cpu().double().item())
    if not math.isfinite(scalar) or not math.isfinite(live):
        raise ValueError("production fk_m2 is not finite")
    if float(np.float32(live)) != float(np.float32(scalar)):
        raise ValueError("reported fk_m2 differs from the live tensor")
    if abs(float(elements.mean()) - scalar) > 1e-6 * max(1.0, abs(scalar)):
        raise ValueError("production fk_m2 does not match its elements")
    if observer.get("contact_branch_executed"):
        raise ValueError("contact branch executed with w_contact=0")
    return ProductionRun(
        target=pose("target"), pred=pose("pred"), fk_m2=scalar,
        fk_elements=elements, gradient=gradient,
        parts={k: float(v) for k, v in parts.items()},
        fk_m2_live=live,
        fk_m2_bits=np.float32(scalar).tobytes().hex(),
    )


class SyntheticFkProvider:
    """Analytic four-window fixture; no real asset or checkpoint is touched."""

    mode = "synthetic"

    def __init__(self, seed: int = 20261028) -> None:
        self.seed = int(seed)
        self.chain = build_synthetic_chain()
        self._fixture = _synthetic_fixture(self.chain, self.seed)
        self._inputs: FkInputs | None = None
        self._assembler: LossAssembler | None = None
        self._counter: CallCounter | None = None
        self.fixture_path: Path | None = None
        self.fixture_sha256: str | None = None
        self.metadata_extra: dict = {}
        self.runtime: dict | None = None

    def bind_counter(self, counter: CallCounter) -> None:
        self._counter = counter

    def prepare_fixture(self, directory: Path) -> None:
        """Create the synthetic 'parent' arrays once; hashes are R0 inputs."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fx = self._fixture
        assembler = _make_synthetic_assembler(
            self.chain, fx["root_mean"], fx["root_std"],
            fx["joint_lo"], fx["joint_hi"])
        arrays: dict[str, np.ndarray] = {
            "x0": fx["x0"],
            "prediction_deterministic": fx["prediction"]["deterministic"],
            "prediction_diffusion": fx["prediction"]["diffusion"],
            "root_mean": fx["root_mean"], "root_std": fx["root_std"],
            "joint_lo": fx["joint_lo"], "joint_hi": fx["joint_hi"],
        }
        for model in MODELS:
            run = _capture_run(assembler, fx["x0"], fx["prediction"][model],
                               None, want_gradient=False)
            arrays[f"fk_points_target_{model}"] = \
                run.target.kp.astype(np.float32)
            arrays[f"fk_points_pred_{model}"] = run.pred.kp.astype(np.float32)
            arrays[f"term_elements_{model}_fk"] = run.fk_elements
        path = directory / "synthetic_parent_arrays.npz"
        np.savez(path, **arrays)
        self.fixture_path = path
        self.fixture_sha256 = sha256_file(path)
        write_json(directory / "synthetic_parent.json", {
            "schema": "w1_real_fk_synthetic_parent_v1",
            "seed": self.seed,
            "fixture_sha256": self.fixture_sha256,
            "chain": self.chain.as_dict(),
            "role_parent_keys": {
                role: f"fk_points_{'target' if role.startswith('target') else 'pred'}_"
                       f"{ROLE_MODEL[role]}"
                for role in ROLES
            },
            "note": ("Synthetic fixture only. It is not a real parent run and "
                     "is not counted as audit evidence outside the analytic "
                     "self-test."),
        })

    def select_windows(self) -> list[WindowKey]:
        return list(fixed_windows())

    def load_inputs(self) -> FkInputs:
        if self._inputs is not None:
            return self._inputs
        if self.fixture_path is None:
            raise RuntimeError("synthetic fixture has not been prepared")
        with np.load(self.fixture_path) as z:
            data = {k: z[k] for k in z.files}
        parent_points = {
            role: np.asarray(
                data[f"fk_points_{'target' if role.startswith('target') else 'pred'}_"
                     f"{ROLE_MODEL[role]}"], dtype=np.float32)
            for role in ROLES
        }
        parent_elements = {
            model: np.asarray(data[f"term_elements_{model}_fk"], dtype=np.float64)
            for model in MODELS
        }
        self._inputs = FkInputs(
            window_keys=self.select_windows(),
            x0=np.asarray(data["x0"], dtype=np.float32),
            prediction={
                "deterministic": np.asarray(
                    data["prediction_deterministic"], dtype=np.float32),
                "diffusion": np.asarray(
                    data["prediction_diffusion"], dtype=np.float32),
            },
            x0_independent=np.asarray(
                independent_encode_for_fixture(
                    data["x0"], data["root_mean"], data["root_std"],
                    data["joint_lo"], data["joint_hi"]), dtype=np.float64
            ).reshape(len(WINDOW_IDS), WINDOW, X0_DIM),
            root_mean=np.asarray(data["root_mean"], dtype=np.float64),
            root_std=np.asarray(data["root_std"], dtype=np.float64),
            joint_lo=np.asarray(data["joint_lo"], dtype=np.float64),
            joint_hi=np.asarray(data["joint_hi"], dtype=np.float64),
            parent_points=parent_points,
            parent_elements=parent_elements,
        )
        return self._inputs

    def _assembler_obj(self, inputs: FkInputs) -> LossAssembler:
        if self._assembler is None:
            self._assembler = _make_synthetic_assembler(
                self.chain, inputs.root_mean, inputs.root_std,
                inputs.joint_lo, inputs.joint_hi, owner=self)
        return self._assembler

    def production_capture(self, model: str, pred: np.ndarray,
                           *, want_gradient: bool = True) -> ProductionRun:
        inputs = self.load_inputs()
        return _capture_run(self._assembler_obj(inputs), inputs.x0, pred,
                            self._counter, want_gradient=want_gradient)

    def reference_positions(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray:
        n = int(np.asarray(pos).reshape(-1, 3).shape[0])
        if self._counter is not None:
            self._counter.add("mj_forward", n)
        return synthetic_fk_np(self.chain, pos, R, q)

    def reference_jacobians(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray:
        n = int(np.asarray(pos).reshape(-1, 3).shape[0])
        if self._counter is not None:
            self._counter.add("mj_forward", n)
            self._counter.add("mj_jac_site", n * N_SITES)
        _, J_nv = synthetic_fk_jac_np(self.chain, pos, R, q)
        Rt = np.swapaxes(np.asarray(R, dtype=np.float64).reshape(n, 3, 3), 1, 2)
        J35 = np.array(J_nv, copy=True)
        J35[..., 3:6] = np.einsum("nsck,nkj->nscj", J_nv[..., 3:6], Rt)
        return J35

    def kernel64_positions(self, pos: np.ndarray, R: np.ndarray,
                           q: np.ndarray) -> np.ndarray:
        if self._counter is not None:
            self._counter.add("kernel64_fk_batch", 1)
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
        q = np.asarray(q, dtype=np.float64).reshape(-1, N_JOINTS)
        if pos.shape[0] > 240:
            raise ValueError("kernel64 batch exceeds 240 states")
        quat = mat_to_quat_wxyz_np(R)
        out = synthetic_fk_torch(
            self.chain,
            torch.tensor(pos, dtype=torch.float64),
            torch.tensor(quat, dtype=torch.float64),
            torch.tensor(q, dtype=torch.float64),
        )
        return out.detach().cpu().numpy()

    def counters(self) -> dict:
        return self._counter.snapshot() if self._counter is not None else {}

    def reference_facts(self) -> dict | None:
        return None

    def metadata(self) -> dict:
        return {
            "mode": "synthetic",
            "seed": self.seed,
            "chain": self.chain.as_dict(),
            "fixture_sha256": self.fixture_sha256,
            "fixture_path": (path_for_record(self.fixture_path)
                             if self.fixture_path is not None else None),
            "real_assets_loaded": False,
            "real_checkpoint_loaded": False,
            "real_dataset_loaded": False,
        }


# ---------------------------------------------------------------------------
# real provider (only reachable after authorization + admission)
# ---------------------------------------------------------------------------


class RealMujocoReference:
    """MuJoCo plain-MJCF reference; compiles the locked asset on first use."""

    def __init__(self, counter: CallCounter | None = None,
                 *, joint_lo: np.ndarray | None = None,
                 joint_hi: np.ndarray | None = None) -> None:
        import mujoco
        from ireset.embodied.g1_spec import G1Spec, JOINT_NAMES, KPT_BODIES

        self.mujoco = mujoco
        self.counter = counter
        self.model = G1Spec().load_plain()
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 36 or self.model.nv != 35:
            raise ValueError(
                f"locked MJCF nq/nv mismatch: {self.model.nq}/{self.model.nv}")
        self.site_ids = {}
        self.site_records = {}
        for name in SITE_NAMES:
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"missing site {name}")
            self.site_ids[name] = sid
            body_id = int(self.model.site_bodyid[sid])
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name != KPT_BODIES[name]:
                raise ValueError(
                    f"site {name} body mismatch: {body_name}")
            self.site_records[name] = {
                "site_id": int(sid), "body": body_name,
                "local_pos": [float(v) for v in self.model.site_pos[sid]],
                "local_quat": [float(v) for v in self.model.site_quat[sid]],
            }
        self.joint_records = []
        for i, name in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"missing joint {name}")
            qadr = int(self.model.jnt_qposadr[jid])
            if qadr != 7 + i:
                raise ValueError(
                    f"joint {name} qpos address {qadr} != {7 + i}")
            if int(self.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"joint {name} is not a hinge")
            if joint_lo is not None and joint_hi is not None:
                lo, hi = (float(v) for v in self.model.jnt_range[jid])
                if abs(lo - float(joint_lo[i])) > 1e-9 \
                        or abs(hi - float(joint_hi[i])) > 1e-9:
                    raise ValueError(
                        f"joint {name} limits differ from the parent constants")
            self.joint_records.append({
                "name": name, "joint_id": int(jid), "qposadr": qadr,
                "range": [float(v) for v in self.model.jnt_range[jid]],
            })

    def facts(self) -> dict:
        return {
            "schema": "w1_real_fk_mujoco_reference_facts_v1",
            "nq": int(self.model.nq), "nv": int(self.model.nv),
            "n_joints": len(self.joint_records),
            "joints": self.joint_records,
            "sites": self.site_records,
            "source": "locked g1_29dof_locked.xml (plain)",
        }

    def positions(self, pos: np.ndarray, R: np.ndarray,
                  q: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
        q = np.asarray(q, dtype=np.float64).reshape(-1, N_JOINTS)
        quat = mat_to_quat_wxyz_np(R)
        out = np.zeros((pos.shape[0], N_SITES, 3))
        for i in range(pos.shape[0]):
            if self.counter is not None:
                self.counter.add("mj_forward", 1)
            self.data.qpos[:3] = pos[i]
            self.data.qpos[3:7] = quat[i]
            self.data.qpos[7:7 + N_JOINTS] = q[i]
            self.mujoco.mj_forward(self.model, self.data)
            for s, name in enumerate(SITE_NAMES):
                out[i, s] = self.data.site_xpos[self.site_ids[name]]
        return out

    def jacobians(self, pos: np.ndarray, R: np.ndarray,
                  q: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
        q = np.asarray(q, dtype=np.float64).reshape(-1, N_JOINTS)
        quat = mat_to_quat_wxyz_np(R)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        out = np.zeros((pos.shape[0], N_SITES, 3, self.model.nv))
        for i in range(pos.shape[0]):
            if self.counter is not None:
                self.counter.add("mj_forward", 1)
            self.data.qpos[:3] = pos[i]
            self.data.qpos[3:7] = quat[i]
            self.data.qpos[7:7 + N_JOINTS] = q[i]
            self.mujoco.mj_forward(self.model, self.data)
            for s, name in enumerate(SITE_NAMES):
                self.mujoco.mj_jacSite(
                    self.model, self.data, jacp, jacr, self.site_ids[name])
                out[i, s] = jacp
                if self.counter is not None:
                    self.counter.add("mj_jac_site", 1)
        Rt = np.swapaxes(R, 1, 2)
        out[..., 3:6] = np.einsum("nsck,nkj->nscj", out[..., 3:6], Rt)
        return out


class RealFkProvider:
    """Registered real provider.  Never constructed without authorization."""

    mode = "real"

    def __init__(self, project_root: Path, *, device: str = "cuda") -> None:
        self.root = Path(project_root)
        self.device = device
        self._inputs: FkInputs | None = None
        self._assembler: LossAssembler | None = None
        self._reference: RealMujocoReference | None = None
        self._kernel: Any = None
        self._counter: CallCounter | None = None
        self._scene_setup_counted = False
        self.runtime: dict | None = None

    def bind_counter(self, counter: CallCounter) -> None:
        self._counter = counter

    def select_windows(self) -> list[WindowKey]:
        return list(fixed_windows())

    def load_inputs(self) -> FkInputs:
        if self._inputs is not None:
            return self._inputs
        arrays_path = self.root / PARENT_FILES["objective_arrays"][0]
        with np.load(arrays_path) as z:
            data = {k: z[k] for k in z.files}
        required = ("x0", "x0_independent", "prediction_deterministic",
                    "prediction_diffusion", "root_mean", "root_std",
                    "joint_lo", "joint_hi")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"parent arrays missing keys: {missing}")
        parent_points = {}
        for role in ROLES:
            key = (f"fk_points_{'target' if role.startswith('target') else 'pred'}_"
                   f"{ROLE_MODEL[role]}")
            if key not in data:
                raise ValueError(f"parent array missing {key}")
            parent_points[role] = np.asarray(data[key], dtype=np.float32)
        parent_elements = {}
        for model in MODELS:
            key = f"term_elements_{model}_fk"
            if key in data:
                parent_elements[model] = np.asarray(data[key], dtype=np.float64)
        self._inputs = FkInputs(
            window_keys=self.select_windows(),
            x0=np.asarray(data["x0"], dtype=np.float32),
            prediction={
                "deterministic": np.asarray(
                    data["prediction_deterministic"], dtype=np.float32),
                "diffusion": np.asarray(
                    data["prediction_diffusion"], dtype=np.float32),
            },
            x0_independent=np.asarray(data["x0_independent"], dtype=np.float64),
            root_mean=np.asarray(data["root_mean"], dtype=np.float64),
            root_std=np.asarray(data["root_std"], dtype=np.float64),
            joint_lo=np.asarray(data["joint_lo"], dtype=np.float64),
            joint_hi=np.asarray(data["joint_hi"], dtype=np.float64),
            parent_points=parent_points,
            parent_elements=parent_elements,
        )
        return self._inputs

    def _assembler_obj(self, inputs: FkInputs) -> LossAssembler:
        if self._assembler is None:
            from ireset.kinematics.torch_adapter import G1TorchFK
            from ireset.model.tokens import RootStats, StateTokenizer

            if self._counter is not None and not self._scene_setup_counted:
                # ``largebox_static_frame`` compiles the locked scene once.
                self._counter.add("scene_setup_forward", 1)
                self._scene_setup_counted = True
            tokenizer = StateTokenizer(
                np.stack([inputs.joint_lo, inputs.joint_hi], axis=1),
                RootStats(mean=inputs.root_mean, std=inputs.root_std))
            weights = LossWeights(w_fk=2.0, w_contact=0.0)
            fk = G1TorchFK(device=self.device, dtype=torch.float32)
            self._assembler = LossAssembler(
                tokenizer, weights, self.device, fk=_CountingFk(fk, self))
        return self._assembler

    def production_capture(self, model: str, pred: np.ndarray,
                           *, want_gradient: bool = True) -> ProductionRun:
        inputs = self.load_inputs()
        return _capture_run(self._assembler_obj(inputs), inputs.x0, pred,
                            self._counter, want_gradient=want_gradient)

    def _reference_obj(self) -> RealMujocoReference:
        if self._reference is None:
            inputs = self.load_inputs()
            self._reference = RealMujocoReference(
                self._counter, joint_lo=inputs.joint_lo,
                joint_hi=inputs.joint_hi)
        return self._reference

    def reference_facts(self) -> dict | None:
        return self._reference.facts() if self._reference is not None else None

    def reference_positions(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray:
        return self._reference_obj().positions(pos, R, q)

    def reference_jacobians(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray:
        return self._reference_obj().jacobians(pos, R, q)

    def kernel64_positions(self, pos: np.ndarray, R: np.ndarray,
                           q: np.ndarray) -> np.ndarray:
        from ireset.kinematics.torch_adapter import G1TorchFK

        if self._kernel is None:
            self._kernel = G1TorchFK(device="cpu", dtype=torch.float64)
        if self._counter is not None:
            self._counter.add("kernel64_fk_batch", 1)
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        if pos.shape[0] > 240:
            raise ValueError("kernel64 batch exceeds 240 states")
        quat = mat_to_quat_wxyz_np(R)
        q = np.asarray(q, dtype=np.float64).reshape(-1, N_JOINTS)
        out = self._kernel.site_world_positions(
            torch.tensor(pos, dtype=torch.float64),
            torch.tensor(quat, dtype=torch.float64),
            torch.tensor(q, dtype=torch.float64),
            list(SITE_NAMES),
        )
        return out.detach().cpu().numpy()

    def counters(self) -> dict:
        return self._counter.snapshot() if self._counter is not None else {}

    def metadata(self) -> dict:
        return {
            "mode": "real", "device": self.device,
            "parent_run": path_for_record(self.root / PARENT_RUN),
            "real_assets_loaded": self._assembler is not None
            or self._reference is not None,
            "real_checkpoint_loaded": False,
            "real_dataset_loaded": False,
        }


# ---------------------------------------------------------------------------
# comparison primitives
# ---------------------------------------------------------------------------


def tolerance_record(tolerance: Mapping[str, Any], key: str) -> dict:
    tolerances = tolerance.get("tolerances") if isinstance(tolerance, Mapping) else None
    if not isinstance(tolerances, Mapping) or key not in tolerances:
        raise ValueError(f"missing tolerance {key}")
    rec = tolerances[key]
    return {"atol": float(rec["atol"]), "rtol": float(rec["rtol"])}


def compare_values(production: Any, reference: Any, tolerance: Mapping[str, Any]
                   ) -> dict:
    """Element-wise ``abs(diff) <= atol + rtol*abs(reference)``."""
    prod = np.asarray(production)
    ref = np.asarray(reference)
    if prod.dtype == bool or ref.dtype == bool:
        return {"status": "mismatch", "problems": ["bool_is_not_numeric"]}
    if prod.shape != ref.shape:
        return {"status": "mismatch",
                "problems": [f"shape:{prod.shape}!={ref.shape}"]}
    if prod.size == 0:
        return {"status": "mismatch", "problems": ["empty_array"]}
    try:
        prod64 = prod.astype(np.float64)
        ref64 = ref.astype(np.float64)
    except (TypeError, ValueError):
        return {"status": "mismatch", "problems": ["non_numeric_array"]}
    if not np.isfinite(prod64).all():
        return {"status": "mismatch", "problems": ["production_nonfinite"]}
    if not np.isfinite(ref64).all():
        return {"status": "mismatch", "problems": ["reference_nonfinite"]}
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])
    diff = np.abs(prod64 - ref64)
    bound = atol + rtol * np.abs(ref64)
    bad = diff > bound
    failures = int(np.count_nonzero(bad))
    scale = np.maximum(np.abs(ref64), 1e-12)
    return {
        "status": "pass" if failures == 0 else "mismatch",
        "failures": failures,
        "max_abs_error": float(diff.max()),
        "max_relative_error": float((diff / scale).max()),
        "bound_max_ratio": float((diff / bound).max()),
        "atol": atol, "rtol": rtol,
        "problems": [] if failures == 0 else ["element_tolerance"],
    }


def _row(key: str, **fields: Any) -> dict:
    return {"key": key, "status": "pass", "problems": [], **fields}


def _summarize(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    failures = [str(r["key"]) for r in rows if r.get("status") == "mismatch"]
    gaps = [str(r["key"]) for r in rows if r.get("status") == "domain_gap"]
    return failures, gaps


def _bitwise_equal(a: Any, b: Any) -> bool:
    x = np.asarray(a)
    y = np.asarray(b)
    return (x.dtype == y.dtype and x.shape == y.shape
            and np.array_equal(x.view(np.uint8), y.view(np.uint8)))


def _scalar_for_scope(elements: np.ndarray, scope: str) -> tuple[float, int, int]:
    """Return ``(l_fk, element_count, denominator)`` for one scope.

    ``elements`` is [4,60,15] per-site squared distances.  ``l_fk`` is the
    mean over elements: 900 per window, 3600 for the batch.
    """
    elements = np.asarray(elements, dtype=np.float64)
    if scope == "batch":
        count = int(elements.size)
        return float(elements.mean()), count, count
    idx = WINDOW_IDS.index(scope)
    part = elements[idx]
    count = int(part.size)
    return float(part.mean()), count, count


# ---------------------------------------------------------------------------
# geometry computation (G stage)
# ---------------------------------------------------------------------------

#: Indirection so a controlled-clock counterexample can expire a deadline
#: *between* compute units without touching the process wall clock.
_CLOCK = time.perf_counter


def _now() -> float:
    return float(_CLOCK())


def _ensure_deadline(deadline: float | None, reserve: float = 0.0,
                     unit: str = "compute") -> None:
    """Refuse to start a new compute unit past the stage deadline."""
    if deadline is not None and _now() + float(reserve) >= float(deadline):
        raise TimeoutError(f"unit_deadline:{unit}")


def _decode_inputs(inputs: FkInputs) -> dict[str, np.ndarray]:
    return {
        "target": np.asarray(inputs.x0, dtype=np.float32),
        "deterministic": np.asarray(
            inputs.prediction["deterministic"], dtype=np.float32),
        "diffusion": np.asarray(inputs.prediction["diffusion"], dtype=np.float32),
    }


def compute_geometry(provider: FkProvider, inputs: FkInputs,
                     tolerance: Mapping[str, Any], *,
                     deadline: float | None = None,
                     reserve: float = 0.0) -> dict:
    """Compute the 122-row geometry matrix and its arrays.

    ``deadline`` is re-checked before every model batch and every reference
    kernel call, so an expired stage cannot keep issuing FK/VJP work.
    """
    _ensure_deadline(deadline, reserve, "geometry_start")
    constants = effective_constants(
        inputs.root_mean, inputs.root_std, inputs.joint_lo, inputs.joint_hi)
    mean32 = constants["mean32"].astype(np.float64)
    std32 = constants["std32"].astype(np.float64)
    lo32 = constants["lo32"].astype(np.float64)
    hi32 = constants["hi32"].astype(np.float64)
    logical = _decode_inputs(inputs)
    ref_decode = {
        inp: independent_decode(x, mean32, std32, lo32, hi32)
        for inp, x in logical.items()
    }
    ref_fk: dict[str, np.ndarray] = {}
    for inp in LOGICAL_INPUTS:
        _ensure_deadline(deadline, reserve, f"reference_positions:{inp}")
        ref_fk[inp] = np.asarray(provider.reference_positions(
            ref_decode[inp]["pos"], ref_decode[inp]["R"],
            ref_decode[inp]["q"])).reshape(
                len(WINDOW_IDS), WINDOW, N_SITES, 3)
    kernel: dict[str, np.ndarray] = {}
    for inp in LOGICAL_INPUTS:
        _ensure_deadline(deadline, reserve, f"kernel64_positions:{inp}")
        kernel[inp] = np.asarray(provider.kernel64_positions(
            ref_decode[inp]["pos"], ref_decode[inp]["R"],
            ref_decode[inp]["q"])).reshape(
                len(WINDOW_IDS), WINDOW, N_SITES, 3)

    rows: list[dict] = []
    grad_rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {
        "x0": inputs.x0.astype(np.float32),
        "x0_independent": inputs.x0_independent.astype(np.float64),
        "prediction_deterministic": inputs.prediction["deterministic"].astype(np.float32),
        "prediction_diffusion": inputs.prediction["diffusion"].astype(np.float32),
        "root_mean": inputs.root_mean.astype(np.float64),
        "root_std": inputs.root_std.astype(np.float64),
        "joint_lo": inputs.joint_lo.astype(np.float64),
        "joint_hi": inputs.joint_hi.astype(np.float64),
    }
    for inp in LOGICAL_INPUTS:
        arrays[f"kernel64_fk_{inp}"] = kernel[inp].astype(np.float64)

    runs: dict[str, ProductionRun] = {}
    ref_jac: dict[str, dict] = {}
    independent_grad: dict[str, dict] = {}

    for model in MODELS:
        _ensure_deadline(deadline, reserve, f"production_capture:{model}")
        run = provider.production_capture(
            model, inputs.prediction[model], want_gradient=True)
        runs[model] = run
        pred_dec = ref_decode[model]
        _ensure_deadline(deadline, reserve, f"reference_jacobians:{model}")
        J35 = provider.reference_jacobians(
            pred_dec["pos"], pred_dec["R"], pred_dec["q"]).reshape(
                len(WINDOW_IDS), WINDOW, N_SITES, 3, TANGENT_DIM)
        D, ddom = decode_jacobian_np(
            logical[model].reshape(-1, X0_DIM), std32, lo32, hi32)
        D = D.reshape(len(WINDOW_IDS), WINDOW, TANGENT_DIM, X0_DIM)
        J38 = np.einsum("bwsck,bwkj->bwscj", J35, D)
        ref_jac[model] = {"J35": J35, "J38": J38, "D": D,
                          "domain": ddom}
        ref_elements = np.sum(
            (ref_fk[model] - ref_fk["target"]) ** 2, axis=-1)
        ref_loss = fk_loss_and_gradient(
            J38.reshape(-1, N_SITES, 3, X0_DIM),
            ref_fk[model].reshape(-1, N_SITES, 3),
            ref_fk["target"].reshape(-1, N_SITES, 3))
        independent_grad[model] = {
            "raw": ref_loss["grad_raw"].reshape(
                len(WINDOW_IDS), WINDOW, X0_DIM),
            "elements": ref_elements,
        }
        prod_elements = np.asarray(run.fk_elements, dtype=np.float64)
        arrays[f"fk_elements_{model}"] = prod_elements
        arrays[f"fk_target_points_{model}"] = run.target.kp.astype(np.float32)
        arrays[f"fk_pred_points_{model}"] = run.pred.kp.astype(np.float32)

        # ---- production FK / parent binding are role-level, not model-level:
        # they are built once after both model runs (see below) -------------
        arrays[f"fk_elements_{model}"] = prod_elements
        arrays[f"fk_target_points_{model}"] = run.target.kp.astype(np.float32)
        arrays[f"fk_pred_points_{model}"] = run.pred.kp.astype(np.float32)

        # ---- scalar rows --------------------------------------------------
        # The batch row must audit the *live* production scalar consumed by
        # the VJP (``run.fk_m2``/``fk_m2_live``), not a re-reduction of the
        # saved elements: a shifted live scalar has to be visible.  Window
        # scopes have no separate production forward, so they use the
        # production element means and are cross-checked against the live
        # scalar through their batch mean.
        run_scalar = float(run.fk_m2)
        run_scalar_live = float(run.fk_m2_live)
        prod_scalars = []
        ref_scalars = []
        for scope in SCOPES:
            prod_from_elements, count, denom = _scalar_for_scope(
                prod_elements, scope)
            ref_m2, _, _ = _scalar_for_scope(ref_elements, scope)
            if scope == "batch":
                prod_m2 = run_scalar
                source = "live_fk_m2"
            else:
                prod_m2 = prod_from_elements
                source = "production_elements_window_mean"
            prod_lfk = prod_m2
            ref_lfk = ref_m2
            cmp = compare_values(
                np.asarray([prod_m2]), np.asarray([ref_m2]),
                tolerance_record(tolerance, "fk_scalar"))
            problems = list(cmp.get("problems", []))
            if scope == "batch":
                if float(np.float32(run_scalar_live)) != float(
                        np.float32(run_scalar)):
                    problems.append("live_tensor_differs_from_reported_scalar")
                if abs(prod_from_elements - run_scalar) > 1e-6 * max(
                        1.0, abs(run_scalar)):
                    problems.append("live_scalar_differs_from_elements")
            row = _row(
                _row_key("fk_scalar", model, scope),
                status="mismatch" if problems else cmp["status"],
                problems=problems,
                production_m2=prod_m2, reference_m2=ref_m2,
                production_lfk=prod_lfk, reference_lfk=ref_lfk,
                production_lfk_weighted=2.0 * prod_lfk,
                reference_lfk_weighted=2.0 * ref_lfk,
                production_m2_from_elements=float(prod_from_elements),
                production_scalar_source=source,
                element_count=count, lfk_denominator=denom,
                mean_factor=1,
                **{k: v for k, v in cmp.items()
                   if k not in ("status", "problems")})
            if scope != "batch":
                row["window_id"] = scope
            else:
                window_means = [
                    float(prod_elements[i].mean()) for i in range(len(WINDOW_IDS))]
                row["production_m2_bits"] = run.fk_m2_bits
                row["batch_equals_window_mean"] = bool(
                    abs(prod_m2 - float(np.mean(window_means)))
                    <= 1e-6 * max(1.0, abs(prod_m2)))
                if not row["batch_equals_window_mean"]:
                    row["status"] = "mismatch"
                    row["problems"] = list(row.get("problems", [])) + [
                        "batch_not_window_mean"]
            rows.append(row)
            prod_scalars.append(prod_m2)
            ref_scalars.append(ref_m2)
        arrays[f"fk_scalar_production_{model}"] = np.asarray(
            prod_scalars, dtype=np.float64)
        arrays[f"fk_scalar_reference_{model}"] = np.asarray(
            ref_scalars, dtype=np.float64)

        # ---- gradient rows ------------------------------------------------
        prod_grad = np.asarray(run.gradient, dtype=np.float64)
        ref_grad = independent_grad[model]["raw"]
        clamped = np.abs(np.asarray(inputs.prediction[model],
                                    dtype=np.float64)[..., 9:38]) >= 1.0
        # A window whose decode derivative is undefined (degenerate 6D or a
        # clamp boundary) cannot support a gradient verdict: record a domain
        # gap instead of a passing gradient row.
        decode_ok = np.asarray(
            ref_jac[model]["domain"]["ok"], dtype=bool).reshape(
                len(WINDOW_IDS), WINDOW)
        for variant in VARIANT:
            for scope in SCOPES:
                prod = prod_grad if variant == "raw" else 2.0 * prod_grad
                ref = ref_grad if variant == "raw" else 2.0 * ref_grad
                if scope != "batch":
                    idx = WINDOW_IDS.index(scope)
                    p_slice, r_slice = prod[idx], ref[idx]
                    c_slice = clamped[idx]
                    scope_ok = bool(decode_ok[idx].all())
                else:
                    p_slice, r_slice, c_slice = prod, ref, clamped
                    scope_ok = bool(decode_ok.all())
                cmp = compare_values(
                    p_slice, r_slice, tolerance_record(tolerance, "fk_gradient"))
                groups = gradient_group_stats(p_slice, r_slice, c_slice)
                problems = list(cmp.get("problems", [])) + groups["problems"]
                if not scope_ok:
                    problems.append("gradient_domain_gap")
                rows.append(_row(
                    _row_key("fk_gradient", model, variant, scope),
                    status=("domain_gap" if not scope_ok
                            else ("pass" if not problems else "mismatch")),
                    problems=problems, groups=groups["groups"],
                    derived_from_vjp=(variant == "weighted"),
                    domain_gap_count=int((~decode_ok).sum()) if scope == "batch"
                    else int((~decode_ok[idx]).sum()),
                    **{k: v for k, v in cmp.items()
                       if k not in ("status", "problems")}))
                grad_rows.append(rows[-1])

        arrays[f"J35_{model}"] = J35.astype(np.float64)
        arrays[f"J38_{model}"] = J38.astype(np.float64)
        arrays[f"D_decode_{model}"] = D.astype(np.float64)
        arrays[f"domain_6d_{model}"] = np.asarray(
            ref_jac[model]["domain"]["ok_6d"]).reshape(
                len(WINDOW_IDS), WINDOW).astype(bool)
        arrays[f"domain_clamp_{model}"] = np.asarray(
            ~ref_jac[model]["domain"]["clamp_inside"]).reshape(
                len(WINDOW_IDS), WINDOW, N_JOINTS).astype(bool)
        arrays[f"grad_production_raw_{model}"] = prod_grad.astype(np.float32)
        arrays[f"grad_independent_raw_{model}"] = ref_grad.astype(np.float64)
        arrays[f"grad_independent_weighted_{model}"] = (
            2.0 * ref_grad).astype(np.float64)

    # ---- float64 kernel isolation (3 logical inputs x 4 windows) ----------
    for inp in LOGICAL_INPUTS:
        for widx, wid in enumerate(WINDOW_IDS):
            cmp = compare_values(
                kernel[inp][widx], ref_fk[inp][widx],
                tolerance_record(tolerance, "kernel64_fk"))
            rows.append(_row(
                _row_key("kernel64_fk", inp, wid),
                status=cmp["status"],
                production_sha256=_hash_array(kernel[inp][widx]),
                reference_sha256=_hash_array(ref_fk[inp][widx]),
                **{k: v for k, v in cmp.items() if k != "status"}))

    # ---- production FK and parent binding (one row per role x window) -----
    for role in ROLES:
        run = runs[ROLE_MODEL[role]]
        inp = ROLE_INPUT[role]
        pose = run.target if role.startswith("target") else run.pred
        prod = np.asarray(pose.kp)
        ref = ref_fk[inp]
        for widx, wid in enumerate(WINDOW_IDS):
            cmp = compare_values(
                prod[widx], ref[widx],
                tolerance_record(tolerance, "production_fk"))
            rows.append(_row(
                _row_key("production_fk", role, wid),
                status=cmp["status"],
                production_sha256=_hash_array(prod[widx]),
                reference_sha256=_hash_array(ref[widx]),
                **{k: v for k, v in cmp.items() if k != "status"}))
        arrays[f"prod_fk_{role}"] = prod.astype(np.float32)
        arrays[f"ref_fk_{role}"] = ref.astype(np.float64)
        parent = inputs.parent_points.get(role)
        for widx, wid in enumerate(WINDOW_IDS):
            key = _row_key("parent_fk_binding", role, wid)
            if parent is None:
                rows.append(_row(key, status="domain_gap",
                                 problems=["parent_points_missing"]))
                continue
            same = _bitwise_equal(
                np.asarray(prod[widx], dtype=np.float32),
                np.asarray(parent[widx], dtype=np.float32))
            rows.append(_row(
                key, status="pass" if same else "mismatch",
                problems=[] if same else ["parent_binding_not_bitwise"],
                production_sha256=_hash_array(np.asarray(prod[widx])),
                parent_sha256=_hash_array(np.asarray(parent[widx])),
                shape=list(np.shape(prod[widx])), dtype="float32"))
        if parent is not None:
            arrays[f"parent_fk_{role}"] = np.asarray(parent, dtype=np.float32)

    # ---- decode rows (all four roles, both target copies included) --------
    target_poses = {
        "target_det": runs["deterministic"].target,
        "target_diff": runs["diffusion"].target,
    }
    target_bitwise = _bitwise_equal(target_poses["target_det"].pos,
                                    target_poses["target_diff"].pos) \
        and _bitwise_equal(target_poses["target_det"].quat,
                           target_poses["target_diff"].quat) \
        and _bitwise_equal(target_poses["target_det"].q,
                           target_poses["target_diff"].q)
    for role in ROLES:
        inp = ROLE_INPUT[role]
        model = ROLE_MODEL[role]
        pose = (runs[model].target if role.startswith("target")
                else runs[model].pred)
        ref = ref_decode[inp]
        ref_pos = ref["pos"].reshape(len(WINDOW_IDS), WINDOW, 3)
        ref_q = ref["q"].reshape(len(WINDOW_IDS), WINDOW, N_JOINTS)
        ref_R = ref["R"].reshape(len(WINDOW_IDS), WINDOW, 3, 3)
        ref_ok = np.asarray(ref["ok"]).reshape(len(WINDOW_IDS), WINDOW)
        prod_pos = np.asarray(pose.pos, dtype=np.float64)
        prod_q = np.asarray(pose.q, dtype=np.float64)
        prod_R = quat_to_mat_wxyz_np(np.asarray(pose.quat, dtype=np.float64))
        for widx, wid in enumerate(WINDOW_IDS):
            okw = bool(ref_ok[widx].all())
            for kind, prod, reference, tol_key in (
                ("p", prod_pos[widx], ref_pos[widx], "decode_float32"),
                ("q", prod_q[widx], ref_q[widx], "decode_float32"),
                ("R", prod_R[widx], ref_R[widx], "decode_rotation_matrix"),
            ):
                cmp = compare_values(
                    prod, reference, tolerance_record(tolerance, tol_key))
                problems = list(cmp.get("problems", []))
                if role.startswith("target") and not target_bitwise:
                    problems.append("target_copies_not_bitwise")
                rows.append(_row(
                    _row_key("decode", role, wid, kind),
                    status="domain_gap" if not okw
                    else ("pass" if not problems else "mismatch"),
                    problems=problems if okw else ["6d_domain_gap"],
                    domain_gap_count=int((~ref_ok[widx]).sum()),
                    **{k: v for k, v in cmp.items()
                       if k not in ("status", "problems")}))
        arrays[f"decode_pos_{role}"] = np.asarray(pose.pos, dtype=np.float32)
        arrays[f"decode_quat_{role}"] = np.asarray(pose.quat, dtype=np.float32)
        arrays[f"decode_R_{role}"] = prod_R.astype(np.float32)
        arrays[f"decode_q_{role}"] = np.asarray(pose.q, dtype=np.float32)
        arrays[f"ref_decode_pos_{role}"] = ref_pos.astype(np.float64)
        arrays[f"ref_decode_R_{role}"] = ref_R.astype(np.float64)
        arrays[f"ref_decode_q_{role}"] = ref_q.astype(np.float64)

    return {
        "rows": rows,
        "grad_rows": grad_rows,
        "arrays": arrays,
        "ref_jac": ref_jac,
        "independent_grad": independent_grad,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# finite-difference computation (D stage)
# ---------------------------------------------------------------------------


def _tangent_perturbation(pos: np.ndarray, R: np.ndarray, q: np.ndarray,
                          dim: int, delta: float) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray]:
    p = np.array(pos, copy=True)
    rot = np.array(R, copy=True)
    joint = np.array(q, copy=True)
    if dim < 3:
        p[dim] += delta
    elif dim < 6:
        axis = np.zeros(3)
        axis[dim - 3] = delta
        rot = so3_exp_np(axis) @ rot
    else:
        joint[dim - 6] += delta
    return p, rot, joint


def compute_fd(provider: FkProvider, inputs: FkInputs,
               grad_arrays: Mapping[str, np.ndarray],
               tolerance: Mapping[str, Any], *,
               deadline: float | None = None,
               reserve: float = 0.0) -> dict:
    """Run the fixed 4672-row state/input FD matrix.

    ``deadline`` is re-checked before every fixed state, so an expired stage
    cannot keep evaluating poses.
    """
    _ensure_deadline(deadline, reserve, "fd_start")
    constants = effective_constants(
        inputs.root_mean, inputs.root_std, inputs.joint_lo, inputs.joint_hi)
    mean32 = constants["mean32"].astype(np.float64)
    std32 = constants["std32"].astype(np.float64)
    lo32 = constants["lo32"].astype(np.float64)
    hi32 = constants["hi32"].astype(np.float64)
    states = fd_states()
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    failures: list[str] = []
    gaps: list[str] = []

    for offset, state in enumerate(states):
        _ensure_deadline(deadline, reserve, f"fd_state:{offset}")
        model = state["model"]
        wid = state["window_id"]
        widx = WINDOW_IDS.index(wid)
        local = state["local_frame"]
        x0 = np.asarray(inputs.prediction[model][widx, local],
                        dtype=np.float64)
        dec = independent_decode(x0[None], mean32, std32, lo32, hi32)
        base_ok = bool(dec["ok"][0])
        J35 = np.asarray(grad_arrays[f"J35_{model}"][widx, local],
                         dtype=np.float64)
        J38 = np.asarray(grad_arrays[f"J38_{model}"][widx, local],
                         dtype=np.float64)
        base_problems = [] if base_ok else ["base_6d_domain_gap"]

        for family, dim in (("tangent35", TANGENT_DIM),
                            ("normalized38", X0_DIM)):
            analytic = J35 if family == "tangent35" else J38
            arrays[f"{family}_analytic_{offset:02d}"] = analytic.astype(np.float64)
            ok_mask = np.zeros(dim, dtype=bool)
            for h in FD_STEPS:
                plus: list[np.ndarray] = []
                minus: list[np.ndarray] = []
                domain_ok = np.ones(dim, dtype=bool)
                for k in range(dim):
                    if family == "tangent35":
                        p_p, R_p, q_p = _tangent_perturbation(
                            dec["pos"][0], dec["R"][0], dec["q"][0], k, h)
                        p_m, R_m, q_m = _tangent_perturbation(
                            dec["pos"][0], dec["R"][0], dec["q"][0], k, -h)
                        plus.append(np.concatenate([p_p, R_p.reshape(-1), q_p]))
                        minus.append(np.concatenate([p_m, R_m.reshape(-1), q_m]))
                    else:
                        xp = np.array(x0, copy=True)
                        xm = np.array(x0, copy=True)
                        xp[k] += h
                        xm[k] -= h
                        dp = independent_decode(xp[None], mean32, std32, lo32, hi32)
                        dm = independent_decode(xm[None], mean32, std32, lo32, hi32)
                        uj = x0[9:38]
                        if 9 <= k < 38:
                            j = k - 9
                            up, um = uj[j] + h, uj[j] - h
                            inside = abs(up) < 1.0 and abs(um) < 1.0
                            outside = abs(up) > 1.0 and abs(um) > 1.0
                            # Both ±1 boundaries and any stencil that lands
                            # exactly on them are undefined; a same-side
                            # interior or saturated pair is fine.
                            if not (inside or outside):
                                domain_ok[k] = False
                        if not (dp["ok"][0] and dm["ok"][0]):
                            domain_ok[k] = False
                        plus.append(np.concatenate(
                            [dp["pos"][0], dp["R"][0].reshape(-1), dp["q"][0]]))
                        minus.append(np.concatenate(
                            [dm["pos"][0], dm["R"][0].reshape(-1), dm["q"][0]]))
                # One reference call per side, all dims batched.
                if family == "tangent35":
                    kp_plus = _eval_batched(provider, plus)
                    kp_minus = _eval_batched(provider, minus)
                else:
                    kp_plus = _eval_batched(provider, plus)
                    kp_minus = _eval_batched(provider, minus)
                if provider_counters(provider):
                    provider_counters(provider).add("fd_pose_eval", 2 * dim)
                fd = np.moveaxis(
                    (kp_plus - kp_minus) / (2.0 * h), 0, -1)  # [15,3,dim]
                arrays[f"{family}_fd_{offset:02d}_{h:g}"] = fd.astype(np.float64)
                for k in range(dim):
                    key = _row_key(
                        "fd", family, state["state_key"], f"{k:02d}", h)
                    if not base_ok:
                        row = _row(key, status="domain_gap",
                                   problems=base_problems, h=h, dim=k,
                                   family=family, **state)
                        gaps.append(key)
                    elif not domain_ok[k]:
                        row = _row(key, status="domain_gap",
                                   problems=["stencil_domain_gap"],
                                   h=h, dim=k, family=family, **state)
                        gaps.append(key)
                    else:
                        cmp = compare_values(
                            fd[..., k], analytic[..., k],
                            tolerance_record(
                                tolerance,
                                "j35_fd" if family == "tangent35" else "j38_fd"))
                        problems = list(cmp.get("problems", []))
                        row = _row(key,
                                   status="pass" if not problems else "mismatch",
                                   problems=problems, h=h, dim=k,
                                   family=family, **state,
                                   **{f: v for f, v in cmp.items()
                                      if f not in ("status", "problems")})
                        if problems:
                            failures.append(key)
                        ok_mask[k] = True
                    rows.append(row)
            arrays[f"{family}_ok_{offset:02d}"] = ok_mask
    return {
        "rows": rows, "arrays": arrays,
        "failures": failures, "gaps": gaps,
        "state_count": len(states),
    }


def _eval_batched(provider: FkProvider, poses: Sequence[np.ndarray]) -> np.ndarray:
    p = np.stack([x[0:3] for x in poses])
    R = np.stack([x[3:12].reshape(3, 3) for x in poses])
    q = np.stack([x[12:41] for x in poses])
    return provider.reference_positions(p, R, q)


def provider_counters(provider: FkProvider) -> CallCounter | None:
    return getattr(provider, "_counter", None)


# ---------------------------------------------------------------------------
# cohort / attempt accounting (semantics carried from the audited ledger)
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


def _read_registration(root: Path, which: str) -> tuple[list[dict] | None, str | None]:
    try:
        rows = load_index(root) if which == "index" else load_ledger(root)
    except (OSError, ValueError, TypeError):
        return None, which
    return rows, None


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
    entry = {
        "status": status,
        "complete": bool((extra or {}).get("complete", False)),
        "measured_wall_s": measured,
        "wall_s": round(measured, 1),
        "accounting_schema": ACCOUNTING_SCHEMA,
        "cohort_id": COHORT_ID,
        **(extra or {}),
    }
    upsert_ledger(root, stage, attempt, entry)


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
    """Index and ledger must publish one identical terminal call snapshot."""
    a = index_row.get("call_balance")
    b = ledger_row.get("call_balance")
    if a is None and b is None:
        return []
    if a is None or b is None:
        return ["call_balance_single_sided"]
    counts_a = _call_counts_from_balance(a)
    counts_b = _call_counts_from_balance(b)
    if counts_a is None or counts_b is None or counts_a != counts_b:
        return ["call_balance_disagreement"]
    return []


def _reservation_problems(root: Path) -> list[str]:
    """An unsettled call reservation means unproven quota: unknown history."""
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
    """Return ``("member"|"foreign"|"legacy"|"unknown", reason)``."""
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
        marker = {}
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
        index, broken_index = _read_registration(sibling, "index")
        ledger, broken_ledger = _read_registration(sibling, "ledger")
        broken = broken_index or broken_ledger
        kind, reason = _attribution(
            marker, index, ledger, broken)
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
                               if sibling.resolve() == Path(exclude_root or "").resolve()
                               else None))
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
        index, broken_index = _read_registration(sibling, "index")
        ledger, broken_ledger = _read_registration(sibling, "ledger")
        kind, _ = _attribution(marker, index, ledger,
                               broken_index or broken_ledger)
        if kind in ("foreign", "legacy"):
            continue
        if kind == "unknown" or broken_index or broken_ledger:
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


def _call_counts_from_balance(balance: Any) -> dict[str, int] | None:
    if not isinstance(balance, Mapping):
        return None
    if balance.get("schema") != "w1_real_fk_call_budget_v1":
        return None
    counts = balance.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(CALL_CAPS):
        return None
    out = {}
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
    totals = {kind: 0 for kind in CALL_CAPS}
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
        previous = {kind: 0 for kind in CALL_CAPS}
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
            for kind in CALL_CAPS:
                if counts[kind] < previous[kind]:
                    return None
                previous[kind] = counts[kind]
        for kind in CALL_CAPS:
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
                   extra: dict | None = None) -> SealResult:
    """running -> publishing -> seal; non-consumable on unproven coverage."""
    extra = dict(extra or {})
    (stage_dir(root, stage) / attempt).mkdir(parents=True, exist_ok=True)
    row = {**row, "stage": stage, "attempt": attempt,
           "accounting_schema": ACCOUNTING_SCHEMA, "cohort_id": COHORT_ID}
    balance = extra.get("call_balance")
    if isinstance(balance, Mapping):
        # The index and the ledger must carry the same terminal snapshot; the
        # index used to keep only the pre-registration balance.
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

    def publish(rw: dict) -> dict:
        sampled = max(0.0, time.perf_counter() - t0)
        rw = {**rw, "status": "publishing", "measured_wall_s": sampled,
              "sampled_wall_s": sampled}
        timed(upsert_index, root, stage, attempt, rw)
        timed(upsert_cost, root, stage, attempt, sampled,
              {**extra, "complete": False,
               "incomplete_reason": rw.get("incomplete_reason"),
               "sampled_wall_s": sampled}, "publishing")
        return rw

    remaining = float(budget["remaining_s"])
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

    def terminal(rw: dict, st: str, cost: float, tail: float) -> dict:
        sampled = max(0.0, time.perf_counter() - t0)
        rw = {**rw, "status": st, "measured_wall_s": cost,
              "sampled_wall_s": sampled, "seal_tail_reserve_s": tail}
        timed(upsert_index, root, stage, attempt, rw)
        timed(upsert_cost, root, stage, attempt, cost,
              {**extra, "complete": bool(rw.get("complete", False)),
               "incomplete_reason": rw.get("incomplete_reason"),
               "sampled_wall_s": sampled, "seal_tail_reserve_s": tail}, st)
        return rw

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
                if r.get("status") in ("done", "crashed", "timeout", "incomplete")]
    if len(terminal) != 1:
        return None, ["stage_not_terminal"]
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
    if time.perf_counter() + float(parent.get("reserve_s", 0.0)) >= deadline:
        problems.append("unit_deadline")
    return problems


# ---------------------------------------------------------------------------
# frozen tolerance contract and pure synthetic calibration
# ---------------------------------------------------------------------------


def _deep_copy_proposals() -> dict:
    import copy

    return copy.deepcopy(TOLERANCE_PROPOSALS)


def calibration_ratio(production: Any, reference: Any,
                      key: str) -> dict:
    prod = np.asarray(production, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if prod.shape != ref.shape or prod.size == 0:
        raise ValueError(f"calibration shape mismatch for {key}")
    spec = TOLERANCE_PROPOSALS[key]
    diff = np.abs(prod - ref)
    bound = spec["atol"] + spec["rtol"] * np.abs(ref)
    ratio = float((diff / bound).max())
    scale = np.maximum(np.abs(ref), 1e-12)
    return {
        "max_abs": float(diff.max()),
        "max_relative": float((diff / scale).max()),
        "max_ratio_of_proposal": ratio,
    }


def build_tolerance_contract(*, observed: Mapping[str, Any] | None = None,
                             calibration: Mapping[str, Any] | None = None,
                             evidence_sha256: str | None = None) -> dict:
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
                    "exceeds the proposal upper bound")
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
        "domain": dict(DOMAIN),
        "group_gate": dict(GROUP_GATE),
        "rules": {
            "order": ["structure", "dtype", "non_empty", "shape", "domain",
                      "finite", "tolerance"],
            "scalar_formula": "abs(prod-ref) <= atol + rtol*abs(ref)",
            "calibration_rule": "4 * observed_max_ratio <= 1.0",
            "thresholds_are_proposals": True,
            "bool_is_not_numeric": True,
            "counts_are_integers": True,
        },
        "calibration": dict(calibration) if calibration is not None else None,
        "calibration_sha256": evidence_sha256,
    }


def build_default_tolerance_contract() -> dict:
    return build_tolerance_contract()


def synthetic_calibration(*, seed: int = 20261028,
                          samples: int = 64) -> dict:
    """Pure synthetic calibration; no real asset, dataset or checkpoint."""
    chain = build_synthetic_chain()
    lo, hi = synthetic_joint_limits()
    mean = np.array([0.12, -0.18, 0.72], dtype=np.float64)
    std = np.array([0.31, 0.27, 0.19], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = int(samples)
    x0 = np.zeros((n, X0_DIM), dtype=np.float32)
    x0[:, 0:3] = rng.normal(0.0, 0.35, (n, 3))
    R, r6 = _random_rotation_features(rng, (n,))
    x0[:, 3:9] = r6
    x0[:, 9:38] = rng.uniform(-0.85, 0.85, (n, N_JOINTS))
    target = x0.copy()
    target[..., 0:3] += rng.normal(0.0, 0.01, (n, 3)).astype(np.float32)
    target[..., 9:38] = np.clip(
        target[..., 9:38] + rng.normal(0.0, 0.01, (n, N_JOINTS)).astype(np.float32),
        -0.95, 0.95)
    constants = effective_constants(mean, std, lo, hi)
    mean32 = constants["mean32"].astype(np.float64)
    std32 = constants["std32"].astype(np.float64)
    lo32 = constants["lo32"].astype(np.float64)
    hi32 = constants["hi32"].astype(np.float64)
    asm = _make_synthetic_assembler(chain, mean, std, lo, hi)
    # LossAssembler expects [B, F, 38]; the independent path uses the flat view.
    b_dim, f_dim = 8, max(1, n // 8)
    x0_flat = x0
    target_flat = target
    x0_batch = x0.reshape(b_dim, f_dim, X0_DIM)
    target_batch = target.reshape(b_dim, f_dim, X0_DIM)
    pred_leaf = torch.tensor(x0_batch, dtype=torch.float32, requires_grad=True)
    observer: dict = {}
    _, parts = asm(pred_leaf, {"x0": torch.tensor(target_batch)},
                   observer=observer)
    prod_pos, prod_quat, prod_q = (
        t.detach().cpu().numpy().reshape(n, -1)
        for t in observer["decode_pred"])
    prod_R = quat_to_mat_wxyz_np(np.asarray(prod_quat, dtype=np.float64))
    prod_kp = observer["fk_points_pred"].detach().cpu().numpy().reshape(
        n, N_SITES, 3)
    prod_kp_target = observer["fk_points_target"].detach().cpu().numpy().reshape(
        n, N_SITES, 3)
    prod_grad = torch.autograd.grad(
        observer["term_tensors"]["fk_m2"], pred_leaf)[0].detach().cpu().numpy()
    prod_grad = prod_grad.reshape(n, X0_DIM)
    dec_pred = independent_decode(x0_flat, mean32, std32, lo32, hi32)
    dec_target = independent_decode(target_flat, mean32, std32, lo32, hi32)
    ref_kp = synthetic_fk_np(chain, dec_pred["pos"], dec_pred["R"], dec_pred["q"])
    ref_kp_target = synthetic_fk_np(
        chain, dec_target["pos"], dec_target["R"], dec_target["q"])
    # float64 kernel isolation (the synthetic analogue of G1TorchFK float64).
    kernel64 = synthetic_fk_torch(
        chain, torch.tensor(dec_pred["pos"], dtype=torch.float64),
        torch.tensor(dec_pred["quat"], dtype=torch.float64),
        torch.tensor(dec_pred["q"], dtype=torch.float64)).detach().numpy()
    _, J_nv = synthetic_fk_jac_np(
        chain, dec_pred["pos"], dec_pred["R"], dec_pred["q"])
    Rt = np.swapaxes(dec_pred["R"], 1, 2)
    J35 = np.array(J_nv, copy=True)
    J35[..., 3:6] = np.einsum("nsck,nkj->nscj", J_nv[..., 3:6], Rt)
    D, ddom = decode_jacobian_np(x0, std32, lo32, hi32)
    J38 = np.einsum("nsck,nkj->nscj", J35, D)
    ref_loss = fk_loss_and_gradient(
        J38, ref_kp.reshape(n, N_SITES, 3),
        ref_kp_target.reshape(n, N_SITES, 3))
    observed = {
        "decode_float32": calibration_ratio(
            np.concatenate([prod_pos, prod_q], axis=1),
            np.concatenate([dec_pred["pos"], dec_pred["q"]], axis=1),
            "decode_float32"),
        "decode_rotation_matrix": calibration_ratio(
            prod_R, dec_pred["R"], "decode_rotation_matrix"),
        "kernel64_fk": calibration_ratio(
            kernel64, ref_kp, "kernel64_fk"),
        "production_fk": calibration_ratio(
            prod_kp, ref_kp, "production_fk"),
        "fk_scalar": calibration_ratio(
            np.asarray([parts["fk_m2"]]),
            np.asarray([ref_loss["l_fk"]]), "fk_scalar"),
        "fk_gradient": calibration_ratio(
            np.asarray(prod_grad, dtype=np.float64), ref_loss["grad_raw"],
            "fk_gradient"),
    }
    # Fixed-step FD on four calibration states (all 35/38 dims, both h).
    j35_ratios = []
    j38_ratios = []
    for state in (0, n // 3, 2 * n // 3, n - 1):
        for h in FD_STEPS:
            for family, dim, analytic in (
                    ("tangent35", TANGENT_DIM, J35[state]),
                    ("normalized38", X0_DIM, J38[state])):
                poses = []
                for k in range(dim):
                    if family == "tangent35":
                        p_p, R_p, q_p = _tangent_perturbation(
                            dec_pred["pos"][state], dec_pred["R"][state],
                            dec_pred["q"][state], k, h)
                        p_m, R_m, q_m = _tangent_perturbation(
                            dec_pred["pos"][state], dec_pred["R"][state],
                            dec_pred["q"][state], k, -h)
                    else:
                        xp = x0[state].astype(np.float64).copy()
                        xm = x0[state].astype(np.float64).copy()
                        xp[k] += h
                        xm[k] -= h
                        dp = independent_decode(xp[None], mean32, std32, lo32, hi32)
                        dm = independent_decode(xm[None], mean32, std32, lo32, hi32)
                        p_p, R_p, q_p = dp["pos"][0], dp["R"][0], dp["q"][0]
                        p_m, R_m, q_m = dm["pos"][0], dm["R"][0], dm["q"][0]
                    poses.append((p_p, R_p, q_p))
                    poses.append((p_m, R_m, q_m))
                p = np.stack([x[0] for x in poses])
                Rr = np.stack([x[1] for x in poses])
                qq = np.stack([x[2] for x in poses])
                kp = synthetic_fk_np(chain, p, Rr, qq)
                fp = kp[0::2]
                fm = kp[1::2]
                fd = np.moveaxis((fp - fm) / (2.0 * h), 0, -1)
                ratios = [calibration_ratio(
                    fd[..., k], analytic[..., k],
                    "j35_fd" if family == "tangent35" else "j38_fd")
                    for k in range(dim)]
                best = max(ratios, key=lambda r: r["max_ratio_of_proposal"])
                if family == "tangent35":
                    j35_ratios.append(best)
                else:
                    j38_ratios.append(best)
    observed["j35_fd"] = max(
        j35_ratios, key=lambda r: r["max_ratio_of_proposal"])
    observed["j38_fd"] = max(
        j38_ratios, key=lambda r: r["max_ratio_of_proposal"])
    # Deliberate negative/edge cases; never used to widen a threshold.
    near_pi = np.zeros((1, 6))
    near_pi[0, 0:3] = [np.cos(np.pi - 1e-5), np.sin(np.pi - 1e-5), 0.0]
    near_pi[0, 3:6] = [-np.sin(np.pi - 1e-5), np.cos(np.pi - 1e-5), 0.0]
    np_dec = independent_decode(
        np.concatenate([np.zeros((1, 3)), near_pi, np.zeros((1, 29))], axis=1),
        mean32, std32, lo32, hi32)
    near_x0 = np.concatenate(
        [np.zeros((1, 3)), near_pi, np.zeros((1, 29))], axis=1).astype(np.float32)
    near_batch = near_x0.reshape(1, 1, X0_DIM)
    near_leaf = torch.tensor(near_batch, dtype=torch.float32)
    near_obs: dict = {}
    asm(near_leaf, {"x0": near_leaf.detach().clone()}, observer=near_obs)
    near_prod_R = quat_to_mat_wxyz_np(
        near_obs["decode_pred"][1].detach().numpy().reshape(1, 4))
    boundary = x0[:1].copy()
    boundary[0, 9] = 1.0
    _, boundary_dom = decode_jacobian_np(boundary, std32, lo32, hi32)
    outside = x0[:1].copy()
    outside[0, 9] = 1.2
    D_out, outside_dom = decode_jacobian_np(outside, std32, lo32, hi32)
    degenerate = x0[:1].copy()
    degenerate[0, 3:6] = 0.0
    _, degenerate_dom = decode_jacobian_np(degenerate, std32, lo32, hi32)
    zero_leaf = torch.tensor(target_batch, dtype=torch.float32,
                             requires_grad=True)
    zero_obs: dict = {}
    _, zero_parts = asm(zero_leaf, {"x0": torch.tensor(target_batch)},
                        observer=zero_obs)
    zero_grad = torch.autograd.grad(
        zero_obs["term_tensors"]["fk_m2"], zero_leaf)[0].detach().numpy()
    excluded = {
        "near_pi": {
            "included_in_tolerance": False,
            "independent_matrix_error": float(
                np.abs(np_dec["R"][0] - quat_to_mat_wxyz_np(
                    np_dec["quat"][:1])[0]).max()),
            "production_matrix_error": float(
                np.abs(near_prod_R[0] - np_dec["R"][0]).max()),
            "production_quat_w": float(
                near_obs["decode_pred"][1].detach().numpy().reshape(-1, 4)[0, 0]),
            "note": ("near-pi branch conditioning is a registered negative "
                     "gate; excluded from tolerance observation"),
        },
        "clamp_boundary": {
            "included_in_tolerance": False,
            "derivative_ok": bool(boundary_dom["ok"][0]),
            "boundary_count": int(boundary_dom["clamp_boundary"].sum()),
            "note": "no unique two-sided derivative exactly at |x_j| = 1",
        },
        "clamp_outside": {
            "included_in_tolerance": True,
            "derivative_ok": bool(outside_dom["ok"][0]),
            "saturated_coordinate_count": int(
                outside_dom["clamp_outside"].sum()),
            "derivative_value": float(D_out[0, 6, 9]),
            "note": "saturated coordinates have exactly zero derivative",
        },
        "degenerate_6d": {
            "included_in_tolerance": False,
            "derivative_ok": bool(degenerate_dom["ok"][0]),
            "first_column_norm": 0.0,
            "note": "zero first column is a domain gap, not a pass",
        },
        "zero_residual": {
            "included_in_tolerance": True,
            "fk_m2": float(zero_parts["fk_m2"]),
            "gradient_max_abs": float(np.abs(zero_grad).max()),
        },
    }
    evidence = {
        "schema": CALIBRATION_SCHEMA,
        "seed": int(seed),
        "samples": n,
        "observed": observed,
        "excluded_cases": excluded,
        "fk_denominator_batch": FK_DENOMINATOR_BATCH,
        "fk_denominator_window": FK_DENOMINATOR_WINDOW,
        "fd_states_used": 4,
        "fd_steps": list(FD_STEPS),
        "positive_rotation_angle_max_rad": math.pi - 0.25,
        "production_calls": 1,
        "kernel64_calls": 1,
        "fd_pose_evaluations": 4 * len(FD_STEPS) * 2 * (TANGENT_DIM + X0_DIM),
        "real_dataset_loaded": False,
        "real_checkpoint_loaded": False,
        "real_asset_loaded": False,
        "real_model_calls": 0,
        "boundaries": dict(BOUNDARIES),
    }
    evidence_sha = _canonical_sha256(evidence)
    contract = build_tolerance_contract(
        observed=observed, calibration=evidence, evidence_sha256=evidence_sha)
    contract["calibration_sha256"] = evidence_sha
    return contract


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
    if not isinstance(contract.get("calibrated"), bool):
        problems.append("tolerance_contract_calibrated_flag")
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
    if _jsonable(contract.get("domain")) != _jsonable(DOMAIN):
        problems.append("tolerance_domain_changed")
    if _jsonable(contract.get("group_gate")) != _jsonable(GROUP_GATE):
        problems.append("tolerance_group_gate_changed")
    if problems:
        return problems
    if mode == "synthetic_default":
        if contract.get("evidence_sha256") is not None:
            problems.append("default_contract_has_evidence")
        if contract.get("calibration") not in (None, {}):
            problems.append("default_contract_has_calibration")
        return problems
    evidence_sha = contract.get("evidence_sha256")
    if not isinstance(evidence_sha, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                             evidence_sha):
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
    if not isinstance(observed, Mapping) or set(observed) != set(TOLERANCE_PROPOSALS):
        problems.append("calibration_observed_keys")
    else:
        for key, rec in observed.items():
            ratio = rec.get("max_ratio_of_proposal") if isinstance(rec, Mapping) else None
            if not _num(ratio) or float(ratio) < 0.0 or 4.0 * float(ratio) > 1.0:
                problems.append(f"calibration_margin:{key}")
    excluded = calibration.get("excluded_cases")
    if not isinstance(excluded, Mapping) or not {
            "near_pi", "clamp_boundary", "clamp_outside", "degenerate_6d",
            "zero_residual"} <= set(excluded):
        problems.append("calibration_excluded_cases")
    seed = calibration.get("seed")
    if not _count(seed):
        problems.append("calibration_seed_invalid")
    elif recompute:
        try:
            fresh = synthetic_calibration(seed=int(seed),
                                          samples=int(calibration.get("samples", 64)))
        except Exception as exc:  # noqa: BLE001 -- evidence must reproduce
            problems.append(f"calibration_recompute_failed:{type(exc).__name__}")
        else:
            if fresh.get("calibration_sha256") != evidence_sha:
                problems.append("calibration_evidence_not_reproducible")
            if _jsonable(fresh.get("calibration")) != _jsonable(calibration):
                problems.append("calibration_values_not_reproducible")
    return problems


def load_tolerance_contract(path: Path,
                            *, recompute: bool = True) -> tuple[dict, str, list[str]]:
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
# implementation / environment identity and key catalog
# ---------------------------------------------------------------------------


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


IMPLEMENTATION_FILES = (
    "src/ireset/eval/w1_real_fk_audit.py",
    "scripts/audit_w1_real_fk.py",
    "tests/test_w1_real_fk_audit.py",
    "src/ireset/train/losses.py",
    "src/ireset/model/tokens.py",
    "src/ireset/kinematics/rotation_adapter.py",
    "src/ireset/kinematics/torch_adapter.py",
    "src/ireset/embodied/g1_spec.py",
    "src/ireset/geometry/distance_adapter.py",
    "configs/train/omniretarget_motion.yaml",
    "docs/plans/omniretarget_w1_real_fk_geometry_derivative_audit_prereg_draft.md",
)
PREREG_DRAFT = IMPLEMENTATION_FILES[-1]


def implementation_identity(project_root: Path) -> dict:
    root = Path(project_root)
    files = {}
    for rel in IMPLEMENTATION_FILES:
        p = root / rel
        files[rel] = sha256_file(p) if p.is_file() else None
    return {"schema": "w1_real_fk_implementation_v1",
            "files": files, "git": _git_identity(root)}


def environment_facts() -> dict:
    import scipy

    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_enabled": _autocast_enabled(),
        "cpu_threads": torch.get_num_threads(),
        "note": ("production path must be CUDA/float32 with TF32 and autocast "
                 "off; the independent reference is CPU/float64"),
    }


#: Frozen production numeric runtime.  ``configure_production_runtime`` makes
#: the settleable flags true for TF32-off and records the values that cannot
#: be forced (deterministic_algorithms, autocast) as hard requirements.
RUNTIME_REQUIREMENTS = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "deterministic_algorithms": False,
    "autocast_enabled": False,
}


def _autocast_enabled() -> bool:
    fn = getattr(torch, "is_autocast_enabled", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001 -- probe signature varies by release
            return False
    return False


def runtime_facts_problems(facts: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(facts, Mapping):
        return ["runtime_facts_missing"]
    return [f"runtime_not_met:{key}"
            for key, want in RUNTIME_REQUIREMENTS.items()
            if facts.get(key) != want]


def configure_production_runtime() -> dict:
    """Apply the frozen production runtime before any real FK/model work.

    The official real entry and the spawned F worker both call this; the
    before/after facts are recorded so the parent/child configuration can be
    cross-checked instead of assumed.
    """
    before = environment_facts()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    after = environment_facts()
    problems = runtime_facts_problems(after)
    return {
        "schema": "w1_real_fk_runtime_v1",
        "requirements": dict(RUNTIME_REQUIREMENTS),
        "before": {k: before.get(k) for k in RUNTIME_REQUIREMENTS},
        "after": {k: after.get(k) for k in RUNTIME_REQUIREMENTS},
        "applied": not problems,
        "problems": problems,
    }


def key_catalog() -> dict:
    geometry = required_geometry_rows()
    fd = required_fd_rows()
    states = fd_states()
    return {
        "geometry_count": len(geometry),
        "geometry_keys": geometry,
        "geometry_multiset_sha256": _canonical_sha256(
            sorted(geometry)),
        "fd_count": len(fd),
        "fd_keys_sha256": _canonical_sha256(sorted(fd)),
        "fd_state_count": len(states),
        "fd_states": states,
        "fd_row_counts": {
            "tangent35": FD_TANGENT_ROWS,
            "normalized38": FD_NORMALIZED_ROWS,
            "total": FD_ROW_COUNT,
        },
        "fd_pose_evaluations": FD_POSE_EVALUATIONS,
        "windows": [w.as_dict() for w in fixed_windows()],
        "roles": list(ROLES),
        "models": list(MODELS),
        "logical_inputs": list(LOGICAL_INPUTS),
        "sites": list(SITE_NAMES),
        "fd_steps": list(FD_STEPS),
        "fd_local_frames": list(FD_LOCAL_FRAMES),
    }


def _key_multiset_problems(rows: Sequence[Mapping[str, Any]],
                           expected: Sequence[str]) -> list[str]:
    got = Counter(str(r.get("key")) for r in rows)
    want = Counter(expected)
    problems = []
    missing = want - got
    extra = got - want
    duplicated = {k: v for k, v in got.items() if v > 1}
    if missing:
        problems.append(f"missing_keys:{len(missing)}")
    if extra:
        problems.append(f"extra_keys:{len(extra)}")
    if duplicated:
        problems.append(f"duplicate_keys:{len(duplicated)}")
    return problems


# ---------------------------------------------------------------------------
# R0 -> G -> D -> F
# ---------------------------------------------------------------------------


def _parent_anchor_facts(project_root: Path, provider: FkProvider) -> dict:
    if getattr(provider, "mode", None) == "synthetic":
        meta = provider.metadata()
        raw = meta.get("fixture_path")
        p = Path(raw) if raw else None
        if p is not None and not p.is_absolute():
            p = Path(project_root) / p
        return {"mode": "synthetic_fixture", "path": raw,
                "sha256": meta.get("fixture_sha256"),
                "exists": bool(p is not None and p.is_file()),
                "note": "synthetic parent; not real evidence"}
    facts = {}
    for name, (rel, want) in PARENT_FILES.items():
        p = Path(project_root) / rel
        actual = sha256_file(p) if p.is_file() else None
        facts[name] = {"path": path_for_record(p), "expected": want,
                       "actual": actual,
                       "match": actual is not None and actual == want}
    return facts


def _implementation_drift(current: Mapping[str, Any],
                          expected: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(expected, Mapping):
        return ["r0_expected_implementation_missing"]
    problems: list[str] = []
    expected_files = expected.get("files")
    if not isinstance(expected_files, Mapping):
        return ["r0_expected_implementation_missing"]
    current_files = current.get("files") or {}
    for rel, want in expected_files.items():
        if current_files.get(rel) != want:
            problems.append(f"implementation_drift:{rel}")
    return problems


#: Parent stage directory names differ from this cohort (P instead of G/D).
_PARENT_STAGE_DIRS = {"R0": "registration", "P": "production", "F": "finalize"}


def _consumer_state_at(run_root: Path, stage: str, dirname: str
                       ) -> tuple[dict | None, list[str]]:
    """Generic disk/index/ledger consumer check for a foreign cohort layout."""
    problems: list[str] = []
    try:
        index = [r for r in load_index(run_root) if r.get("stage") == stage]
        ledger = [r for r in load_ledger(run_root) if r.get("stage") == stage]
    except (OSError, ValueError, TypeError):
        return None, ["registration_unreadable"]
    base = Path(run_root) / dirname
    disk = ({p.name for p in base.iterdir()
             if p.is_dir() and re.fullmatch(r"attempt_\d+", p.name)}
            if base.is_dir() else set())
    index_by = {r.get("attempt"): r for r in index}
    ledger_by = {e.get("attempt"): e for e in ledger}
    if (len(index_by) != len(index) or len(ledger_by) != len(ledger)
            or None in index_by or None in ledger_by
            or set(index_by) != set(ledger_by) or set(disk) != set(index_by)):
        return None, ["attempt_sets_disagree"]
    if not index_by:
        return None, ["stage_not_registered"]
    terminal = [a for a, r in index_by.items()
                if r.get("status") in ("done", "crashed", "timeout", "incomplete")]
    if len(terminal) != 1:
        return None, ["stage_not_terminal"]
    attempt = terminal[0]
    row, entry = index_by[attempt], ledger_by[attempt]
    if row.get("status") != "done" or entry.get("status") != "done":
        problems.append("status_not_done")
    if row.get("complete") is not True or entry.get("complete") is not True:
        problems.append("not_complete")
    if row.get("status") != entry.get("status"):
        problems.append("status_disagreement")
    if row.get("complete") is not entry.get("complete"):
        problems.append("completion_disagreement")
    return row, problems


def _parent_consumer_state(project_root: Path, *, mode: str) -> dict:
    """Consume the parent R0/P/F rows, not just its summary claim."""
    if mode != "real":
        return {"checked": False, "mode": "synthetic_fixture",
                "note": "synthetic parent; no registered run to consume",
                "problems": []}
    base = Path(project_root) / PARENT_RUN
    stages: dict[str, dict] = {}
    problems: list[str] = []
    for stage, dirname in _PARENT_STAGE_DIRS.items():
        row, stage_problems = _consumer_state_at(base, stage, dirname)
        stages[stage] = {
            "attempt": row.get("attempt") if row else None,
            "consumable": row is not None and not stage_problems,
            "problems": stage_problems,
        }
        problems += [f"parent_stage_{stage}:{p}" for p in stage_problems]
    summary: Any = None
    try:
        summary = read_json(Path(project_root) / PARENT_FILES["audit_summary"][0])
    except (OSError, ValueError, TypeError):
        problems.append("parent_summary_unreadable")
    if not isinstance(summary, Mapping):
        problems.append("parent_summary_unreadable")
        conclusion = None
    else:
        conclusion = summary.get("conclusion")
        if conclusion != "real_batch_covered_contract_passed":
            problems.append("parent_conclusion_not_passed")
        if summary.get("complete_evidence") is not True:
            problems.append("parent_evidence_not_complete")
    return {"checked": True, "run": path_for_record(base), "stages": stages,
            "conclusion": conclusion, "problems": problems}


def _asset_closure_facts(project_root: Path, *, mode: str) -> dict:
    """Hash the locked assets and every mesh/include they reference.

    Synthetic mode never opens a real asset, so it records that the closure
    was not checked instead of pretending it was.
    """
    if mode != "real":
        return {"checked": False, "mode": "synthetic_unchecked",
                "problems": [],
                "note": "synthetic mode never opens real assets"}
    import xml.etree.ElementTree as ET

    root = Path(project_root)
    problems: list[str] = []
    locked: dict[str, dict] = {}
    for name, rel in LOCKED_ASSETS.items():
        p = root / rel
        actual = sha256_file(p) if p.is_file() else None
        locked[name] = {"path": path_for_record(p),
                        "exists": actual is not None, "sha256": actual}
        if actual is None:
            problems.append(f"locked_asset_missing:{name}")
    refs: dict[str, dict] = {}
    for name in ("xml", "scene_xml", "urdf"):
        p = root / LOCKED_ASSETS[name]
        if not p.is_file():
            continue
        base = p.parent
        try:
            tree = ET.parse(p)
        except ET.ParseError:
            problems.append(f"asset_parse_failed:{name}")
            continue
        model = tree.getroot()
        meshdir = base
        compiler = model.find("compiler")
        if compiler is not None and compiler.get("meshdir"):
            meshdir = (base / compiler.get("meshdir")).resolve()
        for el in model.iter():
            tag = el.tag.split("}")[-1]
            ref = el.get("file") or el.get("filename")
            if not ref:
                continue
            if tag == "include":
                target = (base / ref).resolve()
            elif tag in ("mesh", "texture", "hfield"):
                if el.get("filename"):  # URDF convention: relative to the file
                    target = (base / ref).resolve()
                elif Path(ref).is_absolute():
                    target = Path(ref)
                else:
                    target = (meshdir / ref).resolve()
            else:
                continue
            key = str(target)
            if key in refs:
                continue
            actual = sha256_file(target) if target.is_file() else None
            refs[key] = {"path": path_for_record(target), "tag": tag,
                         "from": name, "exists": actual is not None,
                         "sha256": actual}
            if actual is None:
                problems.append(f"asset_reference_missing:{target.name}")
    if not refs:
        problems.append("asset_closure_empty")
    return {
        "checked": True, "mode": "real",
        "locked": locked,
        "references": refs,
        "reference_count": len(refs),
        "closure_sha256": _canonical_sha256(
            {k: v["sha256"] for k, v in sorted(refs.items())}),
        "problems": problems,
    }


def _input_schema_facts(inputs: FkInputs) -> dict:
    """R0 schema/effective-constant/domain check on the loaded parent batch."""
    problems: list[str] = []
    arrays: dict[str, dict] = {}

    def check(name: str, array: Any, shape: Sequence[int], dtype: Any, *,
              positive: bool = False) -> None:
        a = np.asarray(array)
        numeric = a.dtype.kind in "fiu"
        record = {"shape": list(a.shape), "dtype": str(a.dtype),
                  "numeric": numeric,
                  "finite": bool(np.isfinite(a).all()) if numeric else False}
        arrays[name] = record
        if a.shape != tuple(shape):
            problems.append(f"input_shape:{name}")
        if a.dtype != np.dtype(dtype):
            problems.append(f"input_dtype:{name}")
        if a.size == 0:
            problems.append(f"input_empty:{name}")
        if not record["finite"]:
            problems.append(f"input_nonfinite:{name}")
        if positive and (not record["finite"] or not bool((a > 0).all())):
            problems.append(f"input_not_positive:{name}")

    check("x0", inputs.x0, (len(WINDOW_IDS), WINDOW, X0_DIM), np.float32)
    for model in MODELS:
        check(f"prediction_{model}", inputs.prediction[model],
              (len(WINDOW_IDS), WINDOW, X0_DIM), np.float32)
    check("x0_independent", inputs.x0_independent,
          (len(WINDOW_IDS), WINDOW, X0_DIM), np.float64)
    check("root_mean", inputs.root_mean, (3,), np.float64)
    check("root_std", inputs.root_std, (3,), np.float64, positive=True)
    check("joint_lo", inputs.joint_lo, (N_JOINTS,), np.float64)
    check("joint_hi", inputs.joint_hi, (N_JOINTS,), np.float64)
    if (arrays["joint_lo"]["finite"] and arrays["joint_hi"]["finite"]
            and not bool((inputs.joint_lo < inputs.joint_hi).all())):
        problems.append("joint_limits_not_ordered")
    for role in ROLES:
        parent = inputs.parent_points.get(role)
        if parent is None:
            problems.append(f"parent_points_missing:{role}")
            continue
        check(f"parent_points_{role}", parent,
              (len(WINDOW_IDS), WINDOW, N_SITES, 3), np.float32)
    for model in MODELS:
        elements = inputs.parent_elements.get(model)
        if elements is None:
            problems.append(f"parent_elements_missing:{model}")
            continue
        check(f"parent_elements_{model}", elements,
              (len(WINDOW_IDS), WINDOW, N_SITES), np.float64)
    constants = effective_constants(
        inputs.root_mean, inputs.root_std, inputs.joint_lo, inputs.joint_hi)
    effective: dict[str, dict] = {}
    for name, array in constants.items():
        a = np.asarray(array)
        effective[name] = {"dtype": str(a.dtype), "shape": list(a.shape),
                           "finite": bool(np.isfinite(a).all())}
        if not effective[name]["finite"]:
            problems.append(f"effective_constant:{name}")
    if bool((np.asarray(constants["std32"]) <= 0).any()):
        problems.append("effective_std_not_positive")
    rot6d: dict[str, dict] = {}
    for name, array in (("target", inputs.x0),
                        ("deterministic", inputs.prediction["deterministic"]),
                        ("diffusion", inputs.prediction["diffusion"])):
        flat = np.asarray(array, dtype=np.float32).reshape(-1, X0_DIM)
        info = rot6d_domain_np(flat[:, 3:9])
        rot6d[name] = {"invalid_6d": int((~info["ok"]).sum()),
                       "min_a_norm": float(info["a_norm"].min()),
                       "min_b_perp": float(info["b_perp"].min())}
    clamp: dict[str, dict] = {}
    for model in MODELS:
        u = np.asarray(inputs.prediction[model], dtype=np.float64)[..., 9:38]
        clamp[model] = {"inside": int((np.abs(u) < 1.0).sum()),
                        "boundary": int((np.abs(u) == 1.0).sum()),
                        "outside": int((np.abs(u) > 1.0).sum())}
    return {"problems": problems, "arrays": arrays,
            "effective_constants": effective,
            "domain": {"rot6d": rot6d, "clamp": clamp},
            "rule": ("shape/dtype/nonempty/finite are checked before any "
                     "comparison; a recorded domain gap is not a pass")}


def run_r0(root: Path, stage: str, attempt: str, t0: float, budget: dict,
           provider: FkProvider, counter: CallCounter,
           tolerance: Mapping[str, Any], *, mode: str,
           project_root: Path = PROJECT_ROOT,
           base_counts: Mapping[str, int] | None = None,
           expected_implementation: Mapping[str, Any] | None = None) -> dict:
    problems: list[str] = []
    gaps: list[str] = []
    catalog = key_catalog()
    contract_problems = tolerance_contract_problems(tolerance)
    problems += [f"tolerance:{p}" for p in contract_problems]
    if catalog["geometry_count"] != 122:
        problems.append(f"geometry_key_count:{catalog['geometry_count']}")
    if catalog["fd_count"] != FD_ROW_COUNT or catalog["fd_count"] != 4672:
        problems.append(f"fd_key_count:{catalog['fd_count']}")
    if catalog["fd_state_count"] != 32:
        problems.append(f"fd_state_count:{catalog['fd_state_count']}")
    if tuple(w.demo for w in provider.select_windows()) != tuple(
            s[0] for s in WINDOW_SPECS):
        problems.append("window_demo_order")
    counts = counter.snapshot()["counts"]
    base = dict(base_counts or {})
    if any(int(counts.get(k, 0)) != int(base.get(k, 0)) for k in CALL_CAPS):
        problems.append("r0_performed_fk_or_model_calls")
    parent = _parent_anchor_facts(project_root, provider)
    if isinstance(parent, Mapping) and "mode" not in parent:
        for name, rec in parent.items():
            if not rec.get("match"):
                problems.append(f"parent_anchor:{name}")
    elif not parent.get("exists"):
        problems.append("synthetic_parent_fixture_missing")
    parent_consumer = _parent_consumer_state(project_root, mode=mode)
    problems += parent_consumer["problems"]
    assets = _asset_closure_facts(project_root, mode=mode)
    problems += assets["problems"]
    try:
        inputs = provider.load_inputs()
        input_facts = _input_schema_facts(inputs)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        input_facts = {"problems": [f"input_load_failed:{type(exc).__name__}"]}
    problems += input_facts.get("problems", [])
    env = environment_facts()
    runtime = getattr(provider, "runtime", None)
    current_impl = implementation_identity(project_root)
    if mode == "real":
        problems += _implementation_drift(current_impl, expected_implementation)
        if not env["cuda_available"]:
            problems.append("real_requires_cuda")
        problems += runtime_facts_problems(env)
        if isinstance(runtime, Mapping):
            problems += [f"provider_{p}" for p in
                         runtime_facts_problems(runtime.get("after"))]
    weights = LossWeights(w_fk=2.0, w_contact=0.0).as_dict()
    if weights.get("w_fk") != 2.0 or weights.get("w_contact") != 0.0:
        problems.append("loss_weights_not_frozen")
    adir = stage_dir(root, stage) / attempt
    adir.mkdir(parents=True, exist_ok=True)
    write_json(adir / "window_keys.json", {
        "schema": "w1_real_fk_window_keys_v1",
        "windows": [w.as_dict() for w in provider.select_windows()],
    })
    write_json(adir / "expected_keys.json", catalog)
    write_json(adir / "tolerance_contract.json", dict(tolerance))
    registration = {
        "schema": REGISTRATION_SCHEMA,
        "audit_schema": AUDIT_SCHEMA,
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "mode": mode,
        "stage": stage,
        "attempt": attempt,
        "windows": [w.as_dict() for w in provider.select_windows()],
        "roles": list(ROLES),
        "models": list(MODELS),
        "logical_inputs": list(LOGICAL_INPUTS),
        "sites": list(SITE_NAMES),
        "key_counts": {"geometry": 122, "fd": 4672, "states": 32},
        "catalog_sha256": _canonical_sha256(catalog),
        "geometry_array_schema": geometry_array_schema(),
        "gradient_array_schema": gradient_array_schema(),
        "fd_array_schema": fd_array_schema(),
        "tolerance_contract_sha256": _canonical_sha256(tolerance),
        "tolerance_evidence_sha256": tolerance.get("evidence_sha256"),
        "parent": parent,
        "parent_consumer_state": parent_consumer,
        "locked_assets": assets,
        "inputs": input_facts,
        "config": {
            "weights": weights, "device": "cuda" if mode == "real" else "cpu",
            "production_dtype": "float32", "reference_dtype": "float64",
            "reference_threads": 1, "fd_steps": list(FD_STEPS),
            "fd_local_frames": list(FD_LOCAL_FRAMES),
        },
        "budget": dict(budget),
        "call_balance": counter.snapshot(),
        "zero_counters": {k: 0 for k in ZERO_COUNTERS},
        "implementation": current_impl,
        "environment": env,
        "runtime": runtime,
        "boundaries": dict(BOUNDARIES),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    registration_path = adir / "registration.json"
    write_json(registration_path, registration)
    admission = {
        "schema": "w1_real_fk_admission_v1",
        "passed": not problems,
        "problems": problems,
        "gaps": gaps,
        "checks": {
            "tolerance_contract": not contract_problems,
            "geometry_key_count": catalog["geometry_count"] == 122,
            "fd_key_count": catalog["fd_count"] == 4672,
            "fd_state_count": catalog["fd_state_count"] == 32,
            "window_order": tuple(w.demo for w in provider.select_windows())
            == tuple(s[0] for s in WINDOW_SPECS),
            "parent_anchors": not any(
                str(p).startswith("parent_anchor") for p in problems),
            "parent_consumer_state": (
                not parent_consumer["problems"]
                if parent_consumer.get("checked") else None),
            "asset_closure": (
                not assets["problems"] if assets.get("checked") else None),
            "input_schema": not input_facts.get("problems"),
            "zero_fk_and_model_calls": not any(
                int(counts.get(k, 0)) != int(base.get(k, 0)) for k in CALL_CAPS),
            "loss_weights": weights.get("w_fk") == 2.0
            and weights.get("w_contact") == 0.0,
            "implementation_bound": not any(
                str(p).startswith("implementation_drift")
                or p == "r0_expected_implementation_missing"
                for p in problems),
            "environment": (not runtime_facts_problems(env)
                            if mode == "real" else True),
        },
        "parent_consumer_state": parent_consumer,
        "asset_closure": assets,
        "inputs": input_facts,
        "call_balance": counter.snapshot(),
        "boundaries": dict(BOUNDARIES),
    }
    write_json(adir / "admission.json", admission)
    return {
        "passed": not problems and not gaps,
        "problems": problems,
        "gaps": gaps,
        "registration_sha256": sha256_file(registration_path),
        "admission_sha256": sha256_file(adir / "admission.json"),
        "registration": registration,
        "admission": admission,
    }


def _ensure_stage_deadline(t0: float, budget: Mapping[str, Any]) -> None:
    remaining = float(budget["remaining_s"])
    reserve = float(budget["reserve_s"])
    if time.perf_counter() - t0 + reserve >= remaining:
        raise TimeoutError("stage deadline/reserve exhausted")


def _reserve_unit_calls(root: Path, stage: str, attempt: str,
                        units: Mapping[str, Mapping[str, Any]]) -> Path:
    """Persist the upper-bound call reservation *before* the work starts.

    An interrupted attempt keeps the reservation on disk (``consumed`` stays
    ``None``), so a later attempt cannot silently reuse unproven quota.
    """
    path = stage_dir(root, stage) / attempt / "call_reservations.json"
    write_json(path, {
        "schema": "w1_real_fk_call_reservations_v1",
        "stage": stage, "attempt": attempt,
        "created_perf_counter": time.perf_counter(),
        "units": {name: dict(rec) for name, rec in units.items()},
        "consumed": None,
        "note": ("upper-bound reservations written before the work; consumed "
                 "is filled only after the unit completes"),
    })
    return path


def _complete_unit_calls(path: Path, consumed: Mapping[str, Any], *,
                         complete: bool = True) -> None:
    data = read_json(path)
    data["consumed"] = {k: int(v) for k, v in consumed.items()}
    data["consumed_complete"] = bool(complete)
    data["completed_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(path, data)


def run_g_stage(root: Path, stage: str, attempt: str, t0: float,
                budget: dict, registration_sha: str,
                provider: FkProvider, counter: CallCounter,
                tolerance: Mapping[str, Any]) -> dict:
    inputs = provider.load_inputs()
    _ensure_stage_deadline(t0, budget)
    reservation_path = _reserve_unit_calls(root, stage, attempt, {
        "production_capture": {
            "units": len(MODELS),
            "reserved": {"loss_assembler": len(MODELS),
                         "production_fk_batch": 2 * len(MODELS),
                         "production_vjp": len(MODELS)},
        },
        "independent_geometry": {
            "units": (len(LOGICAL_INPUTS) * len(WINDOW_IDS) * WINDOW
                      + len(MODELS) * len(WINDOW_IDS) * WINDOW),
            "reserved": {"mj_forward": STAGE_NOMINAL["G"]["mj_forward"]},
        },
        "jacobians": {
            "units": len(MODELS) * len(WINDOW_IDS) * WINDOW,
            "reserved": {"mj_jac_site": STAGE_NOMINAL["G"]["mj_jac_site"]},
        },
        "kernel64": {
            "units": len(LOGICAL_INPUTS),
            "reserved": {"kernel64_fk_batch":
                         STAGE_NOMINAL["G"]["kernel64_fk_batch"]},
        },
    })
    before = counter.snapshot()["counts"]
    delta: dict[str, int] = {}
    reservation_problems: list[str] = []
    completed = False
    try:
        geometry = compute_geometry(
            provider, inputs, tolerance,
            deadline=t0 + float(budget["remaining_s"]),
            reserve=float(budget.get("reserve_s", 0.0)))
        completed = True
    finally:
        # Settle whatever was actually executed even if the unit aborted:
        # an interrupted stage must not look like zero calls.
        after = counter.snapshot()["counts"]
        delta = {k: int(after[k]) - int(before[k]) for k in after}
        try:
            _complete_unit_calls(reservation_path, delta, complete=completed)
        except (OSError, ValueError, TypeError) as exc:
            reservation_problems.append(
                f"call_reservation_unsettled:{type(exc).__name__}")
    nominal = STAGE_NOMINAL["G"]
    problems: list[str] = []
    gaps: list[str] = list(reservation_problems)
    rows = geometry["rows"]
    key_problems = _key_multiset_problems(rows, required_geometry_rows())
    if key_problems:
        gaps += key_problems
    for key in [r["key"] for r in rows if r.get("status") == "domain_gap"]:
        gaps.append(key)
    failures = [r["key"] for r in rows if r.get("status") == "mismatch"]
    for kind, value in delta.items():
        if int(value) > int(nominal[kind]):
            gaps.append(f"stage_call_delta_exceeds_nominal:{kind}:{value}")
    for kind in ("loss_assembler", "production_fk_batch",
                 "production_vjp", "mj_forward", "mj_jac_site"):
        if int(nominal[kind]) > 0 and int(delta.get(kind, 0)) == 0:
            gaps.append(f"stage_call_delta_zero:{kind}")
    adir = stage_dir(root, stage) / attempt
    adir.mkdir(parents=True, exist_ok=True)
    arrays_path = adir / "fk_geometry_arrays.npz"
    np.savez(arrays_path, **geometry["arrays"])
    grad_path = adir / "fk_gradient_arrays.npz"
    np.savez(grad_path, **{k: v for k, v in geometry["arrays"].items()
                           if k.startswith(("J35_", "J38_", "D_decode_",
                                            "domain_", "grad_",
                                            "fk_elements_", "fk_target_points_",
                                            "fk_pred_points_"))})
    rows_path = adir / "geometry_rows.json"
    write_json(rows_path, {
        "schema": "w1_real_fk_geometry_rows_v1",
        "attempt": attempt, "registration_sha256": registration_sha,
        "rows": rows, "gradient_rows": geometry["grad_rows"],
        "failures": failures, "gaps": gaps,
        "call_delta": delta, "call_snapshot": counter.snapshot(),
        "boundaries": dict(BOUNDARIES),
    })
    write_json(adir / "call_trace.json", {
        "schema": "w1_real_fk_call_trace_v1",
        "stage": stage, "attempt": attempt,
        "before": before, "after": after, "delta": delta,
        "nominal": nominal, "snapshot": counter.snapshot(),
    })
    paths = {"arrays": str(arrays_path), "gradient_arrays": str(grad_path),
             "rows": str(rows_path),
             "call_reservations": str(reservation_path)}
    facts = getattr(provider, "reference_facts", lambda: None)()
    if facts is not None:
        facts_path = adir / "reference_facts.json"
        write_json(facts_path, facts)
        paths["reference_facts"] = str(facts_path)
    artifact_hashes = {
        "arrays": sha256_file(arrays_path),
        "gradient_arrays": sha256_file(grad_path),
        "rows": sha256_file(rows_path),
    }
    return {
        "passed": not gaps,
        "problems": problems,
        "failures": failures,
        "gaps": gaps,
        "arrays": geometry["arrays"],
        "rows": rows,
        "gradient_rows": geometry["grad_rows"],
        "paths": paths,
        "artifact_hashes": artifact_hashes,
        "call_delta": delta,
    }


def run_d_stage(root: Path, stage: str, attempt: str, t0: float,
                budget: dict, registration_sha: str, g_result: Mapping[str, Any],
                provider: FkProvider, counter: CallCounter,
                tolerance: Mapping[str, Any]) -> dict:
    inputs = provider.load_inputs()
    _ensure_stage_deadline(t0, budget)
    g_paths = g_result.get("paths") or {}
    grad_path = g_paths.get("gradient_arrays")
    checks: list[str] = []
    expected = (g_result.get("artifact_hashes") or {}).get("gradient_arrays")
    if not grad_path or not Path(grad_path).is_file():
        checks.append("g_gradient_arrays_missing")
    elif not isinstance(expected, str):
        checks.append("g_gradient_arrays_sha_missing")
    elif sha256_file(Path(grad_path)) != expected:
        checks.append("g_gradient_arrays_sha_mismatch")
    grad_arrays: dict[str, np.ndarray] | None = None
    if not checks:
        try:
            with np.load(Path(grad_path)) as z:
                grad_arrays = {k: z[k] for k in z.files}
        except (OSError, ValueError):
            checks.append("g_gradient_arrays_unreadable")
    if grad_arrays is None:
        adir = stage_dir(root, stage) / attempt
        adir.mkdir(parents=True, exist_ok=True)
        rows_path = adir / "fd_rows.json"
        write_json(rows_path, {
            "schema": "w1_real_fk_fd_rows_v1",
            "attempt": attempt, "registration_sha256": registration_sha,
            "rows": [], "failures": [], "gaps": checks,
            "state_count": 0, "row_count": 0,
            "fd_steps": list(FD_STEPS), "fd_local_frames": list(FD_LOCAL_FRAMES),
            "boundaries": dict(BOUNDARIES),
        })
        return {"passed": False, "problems": checks, "failures": [],
                "gaps": checks, "arrays": {}, "rows": [],
                "paths": {"rows": str(rows_path)}, "artifact_hashes": {},
                "call_delta": {}}
    reservation_path = _reserve_unit_calls(root, stage, attempt, {
        "fd_matrix": {
            "units": FD_STATE_COUNT,
            "reserved": {"mj_forward": FD_POSE_EVALUATIONS,
                         "fd_pose_eval": FD_POSE_EVALUATIONS},
        },
    })
    before = counter.snapshot()["counts"]
    delta: dict[str, int] = {}
    reservation_problems: list[str] = []
    completed = False
    try:
        fd = compute_fd(provider, inputs, grad_arrays, tolerance,
                        deadline=t0 + float(budget["remaining_s"]),
                        reserve=float(budget.get("reserve_s", 0.0)))
        completed = True
    finally:
        after = counter.snapshot()["counts"]
        delta = {k: int(after[k]) - int(before[k]) for k in after}
        try:
            _complete_unit_calls(reservation_path, delta, complete=completed)
        except (OSError, ValueError, TypeError) as exc:
            reservation_problems.append(
                f"call_reservation_unsettled:{type(exc).__name__}")
    nominal = STAGE_NOMINAL["D"]
    problems: list[str] = []
    gaps = list(fd["gaps"]) + reservation_problems
    key_problems = _key_multiset_problems(fd["rows"], required_fd_rows())
    if key_problems:
        gaps += key_problems
    for kind, value in delta.items():
        if int(value) > int(nominal[kind]):
            gaps.append(f"stage_call_delta_exceeds_nominal:{kind}:{value}")
    for kind in ("mj_forward", "fd_pose_eval"):
        if int(nominal[kind]) > 0 and int(delta.get(kind, 0)) == 0:
            gaps.append(f"stage_call_delta_zero:{kind}")
    adir = stage_dir(root, stage) / attempt
    adir.mkdir(parents=True, exist_ok=True)
    fd_path = adir / "fd_arrays.npz"
    np.savez(fd_path, **fd["arrays"])
    rows_path = adir / "fd_rows.json"
    write_json(rows_path, {
        "schema": "w1_real_fk_fd_rows_v1",
        "attempt": attempt, "registration_sha256": registration_sha,
        "rows": fd["rows"], "failures": fd["failures"], "gaps": gaps,
        "state_count": fd["state_count"], "row_count": len(fd["rows"]),
        "call_delta": delta, "call_snapshot": counter.snapshot(),
        "fd_steps": list(FD_STEPS), "fd_local_frames": list(FD_LOCAL_FRAMES),
        "boundaries": dict(BOUNDARIES),
    })
    write_json(adir / "call_trace.json", {
        "schema": "w1_real_fk_call_trace_v1",
        "stage": stage, "attempt": attempt,
        "before": before, "after": after, "delta": delta,
        "nominal": nominal, "snapshot": counter.snapshot(),
    })
    return {
        "passed": not gaps,
        "problems": problems,
        "failures": fd["failures"],
        "gaps": gaps,
        "arrays": fd["arrays"],
        "rows": fd["rows"],
        "paths": {"arrays": str(fd_path), "rows": str(rows_path),
                  "call_reservations": str(reservation_path)},
        "artifact_hashes": {"arrays": sha256_file(fd_path),
                            "rows": sha256_file(rows_path)},
        "call_delta": delta,
    }


# --- finalize helpers -------------------------------------------------------


_FLOAT_FIELDS = (
    "max_abs_error", "max_relative_error", "bound_max_ratio",
    "production_m2", "reference_m2", "production_lfk", "reference_lfk",
    "production_lfk_weighted", "reference_lfk_weighted",
    "production_m2_from_elements", "atol", "rtol", "h",
)
_INT_FIELDS = (
    "failures", "domain_gap_count", "element_count", "lfk_denominator",
    "mean_factor", "dim", "local_frame", "abs_frame",
)
_BOOL_FIELDS = (
    "derived_from_vjp", "batch_equals_window_mean",
)
_IGNORED_FIELDS = ("production_sha256", "reference_sha256")


def _compare_row_fields(got: Mapping[str, Any], want: Mapping[str, Any],
                        key: str) -> list[str]:
    problems: list[str] = []
    if set(got) != set(want):
        problems.append(f"{key}:field_set")
    for field in _FLOAT_FIELDS:
        if field in got and field in want:
            a, b = got[field], want[field]
            if a is None and b is None:
                continue
            if not (_num(a) and _num(b)) or abs(float(a) - float(b)) > 1e-12 * max(
                    1.0, abs(float(a)), abs(float(b))):
                problems.append(f"{key}:{field}")
    for field in _INT_FIELDS:
        if field in got and field in want:
            if type(got[field]) is not int or type(want[field]) is not int \
                    or got[field] != want[field]:
                problems.append(f"{key}:{field}")
    for field in _BOOL_FIELDS:
        if field in got and field in want and got[field] is not want[field]:
            problems.append(f"{key}:{field}")
    for field in ("status", "family", "model", "window_id", "state_key"):
        if field in got or field in want:
            if got.get(field) != want.get(field):
                problems.append(f"{key}:{field}")
    if "groups" in got or "groups" in want:
        ga, gb = got.get("groups"), want.get("groups")
        if isinstance(ga, Mapping) and isinstance(gb, Mapping):
            if set(ga) != set(gb):
                problems.append(f"{key}:group_set")
            else:
                for name, rec in ga.items():
                    other = gb[name]
                    if not isinstance(rec, Mapping) or not isinstance(other, Mapping):
                        problems.append(f"{key}:group:{name}")
                        continue
                    for f, v in rec.items():
                        w = other.get(f)
                        if isinstance(v, bool) or isinstance(w, bool):
                            ok = v is w
                        elif _num(v) and _num(w):
                            ok = abs(float(v) - float(w)) <= 1e-12 * max(
                                1.0, abs(float(v)), abs(float(w)))
                        else:
                            ok = v == w
                        if not ok:
                            problems.append(f"{key}:group:{name}:{f}")
        else:
            problems.append(f"{key}:groups")
    handled = (set(_FLOAT_FIELDS) | set(_INT_FIELDS) | set(_BOOL_FIELDS)
               | set(_IGNORED_FIELDS)
               | {"key", "status", "family", "model", "window_id",
                  "state_key", "groups"})
    for field in sorted(set(got) & set(want)):
        if field in handled:
            continue
        a, b = got[field], want[field]
        if isinstance(a, bool) or isinstance(b, bool):
            ok = a is b
        elif _num(a) and _num(b):
            ok = abs(float(a) - float(b)) <= 1e-12 * max(
                1.0, abs(float(a)), abs(float(b)))
        elif isinstance(a, str) or isinstance(b, str) \
                or a is None or b is None:
            ok = a == b
        else:
            continue  # containers are compared by their own field groups
        if not ok:
            problems.append(f"{key}:{field}")
    return problems


def _array_compare_problems(saved: Mapping[str, np.ndarray],
                            fresh: Mapping[str, np.ndarray],
                            *, bitwise_keys: Sequence[str],
                            tolerance: float = 1e-12) -> list[str]:
    problems: list[str] = []
    if set(saved) != set(fresh):
        problems.append(f"array_keys:{len(set(saved) ^ set(fresh))}")
        return problems
    for key in sorted(saved):
        a, b = np.asarray(saved[key]), np.asarray(fresh[key])
        if a.dtype != b.dtype or a.shape != b.shape:
            problems.append(f"array_meta:{key}")
            continue
        if not (np.isfinite(a).all() if a.dtype.kind == "f" else True):
            problems.append(f"array_nonfinite:{key}")
            continue
        if key in bitwise_keys:
            if not _bitwise_equal(a, b):
                problems.append(f"array_not_bitwise:{key}")
            continue
        if a.dtype.kind == "b":
            if not np.array_equal(a, b):
                problems.append(f"array_bool:{key}")
        elif a.dtype.kind == "f":
            if not np.allclose(a, b, atol=tolerance, rtol=tolerance,
                               equal_nan=False):
                problems.append(f"array_value:{key}")
        elif not np.array_equal(a, b):
            problems.append(f"array_value:{key}")
    return problems


def _artifact_read_json(path: Any, tag: str, schema: str,
                        problems: list[str]) -> dict | None:
    p = Path(path) if path else None
    if p is None or not p.is_file():
        problems.append(f"{tag}_missing")
        return None
    try:
        data = read_json(p)
    except (OSError, ValueError, TypeError):
        problems.append(f"{tag}_unreadable")
        return None
    if not isinstance(data, Mapping):
        problems.append(f"{tag}_not_a_mapping")
        return None
    if data.get("schema") != schema:
        problems.append(f"{tag}_schema")
    for key in ("rows", "gradient_rows", "failures", "gaps"):
        if key in data and not isinstance(data[key], list):
            problems.append(f"{tag}_{key}_invalid")
            data = {**data, key: []}
    return dict(data)


def _artifact_rows(data: Mapping[str, Any] | None, key: str, tag: str,
                   problems: list[str]) -> list[dict]:
    if not isinstance(data, Mapping):
        return []
    rows = data.get(key, [])
    if not isinstance(rows, list):
        problems.append(f"{tag}_{key}_invalid")
        return []
    out: list[dict] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("key"), str):
            problems.append(f"{tag}_row_invalid:{i}")
            continue
        out.append(dict(row))
    return out


def _artifact_npz(path: Any, tag: str,
                  problems: list[str],
                  schema: Mapping[str, Mapping[str, Any]] | None = None
                  ) -> dict[str, np.ndarray] | None:
    p = Path(path) if path else None
    if p is None or not p.is_file():
        problems.append(f"{tag}_missing")
        return None
    try:
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
    except (OSError, ValueError, TypeError):
        problems.append(f"{tag}_unreadable")
        return None
    if schema is not None:
        problems += _npz_schema_problems(arrays, schema, tag)
    return arrays


def _geometry_file_schema() -> dict[str, dict[str, Any]]:
    """The geometry NPZ holds both partitions; the gradient NPZ a subset."""
    return {**geometry_array_schema(), **gradient_array_schema()}


def _npz_schema_problems(arrays: Mapping[str, np.ndarray],
                         schema: Mapping[str, Mapping[str, Any]],
                         tag: str) -> list[str]:
    problems: list[str] = []
    missing = sorted(set(schema) - set(arrays))
    extra = sorted(set(arrays) - set(schema))
    if missing:
        problems.append(f"{tag}_schema_missing:{len(missing)}")
    if extra:
        problems.append(f"{tag}_schema_extra:{len(extra)}")
    for key in sorted(set(schema) & set(arrays)):
        arr = np.asarray(arrays[key])
        want = schema[key]
        if str(arr.dtype) != want["dtype"]:
            problems.append(f"{tag}_schema_dtype:{key}")
        if list(arr.shape) != list(want["shape"]):
            problems.append(f"{tag}_schema_shape:{key}")
        if arr.dtype.kind not in "fbiu":
            problems.append(f"{tag}_schema_not_numeric:{key}")
    return problems


def _split_fk_arrays(arrays: Mapping[str, np.ndarray]
                     ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split a full geometry array set into (geometry, gradient) partitions."""
    grad_keys = set(gradient_array_schema())
    return ({k: v for k, v in arrays.items() if k not in grad_keys},
            {k: v for k, v in arrays.items() if k in grad_keys})


def _file_sha(path: Any) -> str | None:
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return None
    return sha256_file(p)


def _artifact_hash_problems(paths: Mapping[str, Any] | None,
                            expected: Mapping[str, Any] | None,
                            keys: Sequence[str], tag: str,
                            problems: list[str]) -> None:
    """Each artifact is checked against its own recorded hash, separately."""
    if not isinstance(expected, Mapping):
        problems.append(f"{tag}_artifact_hashes_missing")
        return
    for key in keys:
        path = (paths or {}).get(key)
        want = expected.get(key)
        if not path:
            problems.append(f"{tag}_{key}_path_missing")
            continue
        if not isinstance(want, str):
            problems.append(f"{tag}_{key}_sha_missing")
            continue
        actual = sha256_file(Path(path)) if Path(path).is_file() else None
        if actual is None:
            problems.append(f"{tag}_{key}_file_missing")
        elif actual != want:
            problems.append(f"{tag}_{key}_sha_mismatch")


def _artifact_overlap_problems(first: Mapping[str, np.ndarray] | None,
                               second: Mapping[str, np.ndarray] | None,
                               tag: str, problems: list[str]) -> None:
    if not first or not second:
        return
    for key in sorted(set(first) & set(second)):
        if not _bitwise_equal(first[key], second[key]):
            problems.append(f"{tag}_overlap_conflict:{key}")


def _recompute_worker_main(spec_path: str, out_dir: str) -> int:
    """Fresh-process G/D recomputation for F; writes arrays, rows and meta."""
    spec = read_json(Path(spec_path))
    out = Path(out_dir)
    started = time.perf_counter()
    mode = spec.get("mode")
    tolerance = spec.get("tolerance") or {}
    counter = CallCounter(CALL_CAPS)
    reserve = float(spec.get("reserve_s", 0.0))
    deadline_perf = float(spec.get("deadline_perf", math.inf))
    deadline_epoch = float(spec.get("deadline_epoch", math.inf))
    parent = {"registration_sha256": spec.get("registration_sha256"),
              "reserve_s": reserve}
    units: list[dict] = []
    provider: FkProvider | None = None
    delta: dict[str, int] = {}
    exit_reason = "ok"
    before = counter.snapshot()["counts"]
    runtime: dict = {"applied": False, "mode": mode, "problems": [],
                     "requirements": dict(RUNTIME_REQUIREMENTS)}

    def admit(name: str) -> list[str]:
        problems: list[str] = []
        if time.time() + reserve >= deadline_epoch:
            problems.append("unit_deadline")
        if time.perf_counter() + reserve >= deadline_perf:
            problems.append("unit_deadline")
        if not problems:
            try:
                problems = unit_admission(
                    Path(spec["root"]), str(spec["stage"]),
                    str(spec["attempt"]), parent, deadline=deadline_perf)
            except (OSError, ValueError, TypeError):
                problems = ["unit_parent_unreadable"]
        units.append({"unit": name, "source": "worker",
                      "admitted": not problems, "problems": problems})
        return problems

    try:
        if mode == "synthetic":
            provider = SyntheticFkProvider(seed=int(spec["seed"]))
            provider.fixture_path = Path(spec["fixture_path"])
            provider.fixture_sha256 = spec.get("fixture_sha256")
        else:
            runtime = configure_production_runtime()
            if runtime["problems"]:
                exit_reason = "runtime_not_applied"
            provider = RealFkProvider(Path(spec["project_root"]),
                                      device=spec.get("device", "cuda"))
        provider.bind_counter(counter)
        if exit_reason == "ok" and admit("recompute_backend"):
            exit_reason = "backend_not_admitted"
        elif exit_reason == "ok":
            before = counter.snapshot()["counts"]
            inputs = provider.load_inputs()
            if admit("geometry_replay"):
                exit_reason = "geometry_not_admitted"
            else:
                geometry = compute_geometry(
                    provider, inputs, tolerance,
                    deadline=deadline_perf, reserve=reserve)
                write_json(out / "fresh_geometry_rows.json", {
                    "schema": "w1_real_fk_fresh_rows_v1",
                    "rows": geometry["rows"],
                    "gradient_rows": geometry["grad_rows"],
                })
                write_json(out / "fresh_reference_facts.json",
                           provider.reference_facts() or {})
                if admit("fd_replay"):
                    exit_reason = "fd_not_admitted"
                else:
                    fd = compute_fd(provider, inputs, geometry["arrays"],
                                    tolerance, deadline=deadline_perf,
                                    reserve=reserve)
                    write_json(out / "fresh_fd_rows.json", {
                        "schema": "w1_real_fk_fresh_rows_v1",
                        "rows": fd["rows"], "failures": fd["failures"],
                        "gaps": fd["gaps"],
                    })
                    np.savez(out / "fresh_geometry_arrays.npz",
                             **geometry["arrays"])
                    np.savez(out / "fresh_fd_arrays.npz", **fd["arrays"])
    except (TimeoutError, FkCallBudgetExceeded) as exc:
        exit_reason = f"{type(exc).__name__}:{exc}"
    except Exception as exc:  # noqa: BLE001 -- must still report honestly
        exit_reason = f"worker_exception:{type(exc).__name__}:{exc}"
    finally:
        # Always settle the calls that actually happened.  An aborted worker
        # used to report an empty delta, letting the parent under-count.
        after = counter.snapshot()["counts"]
        delta = {k: int(after[k]) - int(before[k]) for k in after}
    meta = {
        "schema": "w1_real_fk_recompute_process_v1",
        "pid": os.getpid(), "ppid": os.getppid(),
        "python": sys.executable, "cwd": os.getcwd(),
        "start_method": "spawned_python_subprocess",
        "mode": mode, "exit_reason": exit_reason,
        "units": units, "call_delta": delta,
        "counts": counter.snapshot(),
        "runtime": runtime,
        "elapsed_s": time.perf_counter() - started,
        "fixture_sha256": getattr(provider, "fixture_sha256", None),
        "metadata": provider.metadata() if provider is not None else None,
    }
    write_json(out / "worker_meta.json", meta)
    return 0 if exit_reason == "ok" else 3


def _run_recompute_worker(spec: Mapping[str, Any], out_dir: Path,
                          timeout_s: float, project_root: Path
                          ) -> tuple[dict, list[str]]:
    problems: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "recompute_spec.json"
    write_json(spec_path, dict(spec))
    src = str((Path(project_root) / "src").resolve())
    code = (
        "import sys\n"
        f"sys.path.insert(0, {src!r})\n"
        "from ireset.eval.w1_real_fk_audit import _recompute_worker_main\n"
        "raise SystemExit(_recompute_worker_main(sys.argv[1], sys.argv[2]))\n")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, str(spec_path), str(out_dir)],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(project_root))
    except subprocess.TimeoutExpired:
        return {}, ["f_worker_timeout"]
    except OSError as exc:
        return {}, [f"f_worker_spawn_failed:{type(exc).__name__}"]
    if proc.returncode != 0:
        problems.append(f"f_worker_exit:{proc.returncode}")
    meta_path = out_dir / "worker_meta.json"
    if not meta_path.is_file():
        return {}, problems + ["f_worker_meta_missing"]
    try:
        meta = read_json(meta_path)
    except (OSError, ValueError):
        return {}, problems + ["f_worker_meta_unreadable"]
    if not isinstance(meta, Mapping):
        return {}, problems + ["f_worker_meta_not_a_mapping"]
    meta = dict(meta)
    if meta.get("pid") in (None, os.getpid()):
        problems.append("f_worker_not_separate_process")
    if meta.get("exit_reason") != "ok":
        problems.append(f"f_worker_aborted:{meta.get('exit_reason')}")
    return meta, problems


def run_f_stage(root: Path, stage: str, attempt: str, t0: float,
                budget: dict, registration_sha: str,
                g_result: Mapping[str, Any], d_result: Mapping[str, Any],
                provider: FkProvider, counter: CallCounter,
                tolerance: Mapping[str, Any], *,
                project_root: Path = PROJECT_ROOT,
                provider_factory: Any = None,
                recompute: str = "subprocess",
                g_expected: Mapping[str, Any] | None = None,
                d_expected: Mapping[str, Any] | None = None,
                prerequisite_problems: Sequence[str] = ()) -> dict:
    """Recompute G and D from disk in a fresh backend; do not trust summaries.

    Every compute unit is admitted against the live parent row and the stage
    deadline; with ``recompute="subprocess"`` the heavy work runs in a spawned
    Python process whose identity and call delta are recorded.
    """
    if recompute not in ("subprocess", "in_process"):
        raise ValueError(f"unknown recompute mode {recompute}")
    mode = getattr(provider, "mode", None)
    deadline = t0 + float(budget["remaining_s"])
    reserve = float(budget.get("reserve_s", 0.0))
    parent = {"registration_sha256": registration_sha, "reserve_s": reserve}
    units: list[dict] = []

    def admit(name: str) -> list[str]:
        if time.perf_counter() + reserve >= deadline:
            problems = ["unit_deadline"]
        else:
            problems = unit_admission(root, stage, attempt, parent,
                                      deadline=deadline)
        units.append({"unit": name, "source": "parent",
                      "admitted": not problems, "problems": list(problems)})
        return problems

    verification: list[str] = [str(p) for p in prerequisite_problems]
    if mode == "real":
        parent_runtime = getattr(provider, "runtime", None)
        if not isinstance(parent_runtime, Mapping):
            verification.append("f_parent_runtime_missing")
        else:
            verification += [
                f"f_parent_runtime_mismatch:{p}"
                for p in runtime_facts_problems(parent_runtime.get("after"))]
    gaps: list[str] = []
    failures: list[str] = []
    g_saved_rows: list[dict] = []
    g_saved_grad: list[dict] = []
    d_saved_rows: list[dict] = []
    g_arrays: dict[str, np.ndarray] | None = None
    g_grad: dict[str, np.ndarray] | None = None
    d_arrays: dict[str, np.ndarray] | None = None
    row_counts = {"geometry_rows": 0, "gradient_rows": 0, "fd_rows": 0,
                  "fd_states": None}
    g_ok = isinstance(g_result, Mapping)
    d_ok = isinstance(d_result, Mapping)
    g_paths = (g_result.get("paths") if g_ok else None) or {}
    d_paths = (d_result.get("paths") if d_ok else None) or {}
    if not g_ok:
        verification.append("g_result_not_a_mapping")
    if not d_ok:
        verification.append("d_result_not_a_mapping")
    expected_g = (g_expected if g_expected is not None
                  else (g_result.get("artifact_hashes") if g_ok else None))
    expected_d = (d_expected if d_expected is not None
                  else (d_result.get("artifact_hashes") if d_ok else None))

    blocked = admit("artifact_read")
    if blocked:
        verification += [f"f_unit_not_admitted:artifact_read:{p}"
                         for p in blocked]
    else:
        g_rows = _artifact_read_json(g_paths.get("rows"), "g_rows",
                                     "w1_real_fk_geometry_rows_v1",
                                     verification)
        d_rows = _artifact_read_json(d_paths.get("rows"), "d_rows",
                                     "w1_real_fk_fd_rows_v1", verification)
        g_saved_rows = _artifact_rows(g_rows, "rows", "g_rows", verification)
        g_saved_grad = _artifact_rows(g_rows, "gradient_rows", "g_rows",
                                      verification)
        d_saved_rows = _artifact_rows(d_rows, "rows", "d_rows", verification)
        gaps += [str(g) for g in (d_rows or {}).get("gaps", [])
                 if isinstance(g, str)]
        row_counts = {
            "geometry_rows": len(g_saved_rows),
            "gradient_rows": len(g_saved_grad),
            "fd_rows": len(d_saved_rows),
            "fd_states": (d_rows or {}).get("state_count"),
        }
        _artifact_hash_problems(g_paths, expected_g,
                                ("arrays", "gradient_arrays", "rows"), "g",
                                verification)
        _artifact_hash_problems(d_paths, expected_d, ("arrays", "rows"), "d",
                                verification)
        g_arrays = _artifact_npz(g_paths.get("arrays"), "g_arrays",
                                 verification, _geometry_file_schema())
        g_grad = _artifact_npz(g_paths.get("gradient_arrays"),
                               "g_gradient_arrays", verification,
                               gradient_array_schema())
        d_arrays = _artifact_npz(d_paths.get("arrays"), "d_arrays",
                                 verification, fd_array_schema())
        _artifact_overlap_problems(g_arrays, g_grad, "g", verification)
        # ``_artifact_npz`` validated each file's own schema above; merging
        # alone would let the geometry file mask an emptied gradient file.
        verification += _key_multiset_problems(
            [r for r in g_saved_rows
             if not str(r.get("key", "")).startswith("fk_gradient|")],
            [k for k in required_geometry_rows()
             if not k.startswith("fk_gradient|")])
        verification += _key_multiset_problems(
            g_saved_grad, [k for k in required_geometry_rows()
                           if k.startswith("fk_gradient|")])
        verification += _key_multiset_problems(d_saved_rows, required_fd_rows())

    # 2. recompute in a fresh backend, only after artifacts are consistent.
    fresh_g_rows: list[dict] | None = None
    fresh_g_grad: list[dict] | None = None
    fresh_d_rows: list[dict] | None = None
    fresh_g_arrays: dict[str, np.ndarray] | None = None
    fresh_d_arrays: dict[str, np.ndarray] | None = None
    process: dict = {}
    worker_meta: dict = {}
    f_delta: dict[str, int] = {}
    if not verification:
        out_dir = stage_dir(root, stage) / attempt / "recompute"
        out_dir.mkdir(parents=True, exist_ok=True)
        before = counter.snapshot()["counts"]
        reservation_path: Path | None = None
        delta_settled = True
        recompute_done = False
        try:
            reservation_path = _reserve_unit_calls(root, stage, attempt, {
                "fresh_process": {
                    "units": 1,
                    "reserved": {
                        "loss_assembler": STAGE_NOMINAL["F"]["loss_assembler"],
                        "production_fk_batch":
                            STAGE_NOMINAL["F"]["production_fk_batch"],
                        "production_vjp":
                            STAGE_NOMINAL["F"]["production_vjp"],
                        "kernel64_fk_batch":
                            STAGE_NOMINAL["F"]["kernel64_fk_batch"],
                        "mj_forward": STAGE_NOMINAL["F"]["mj_forward"],
                        "mj_jac_site": STAGE_NOMINAL["F"]["mj_jac_site"],
                    },
                },
                "fd_replay": {
                    "units": FD_STATE_COUNT,
                    "reserved": {"mj_forward": FD_POSE_EVALUATIONS,
                                 "fd_pose_eval": FD_POSE_EVALUATIONS},
                },
            })
        except (OSError, ValueError, TypeError) as exc:
            verification.append(
                f"f_reservation_write_failed:{type(exc).__name__}")
        if not verification and recompute == "in_process":
            unit_hint = "recompute_backend"
            blocked = admit("recompute_backend")
            if not blocked:
                unit_hint = "geometry_replay"
                try:
                    fresh_provider = (provider_factory()
                                      if provider_factory is not None
                                      else _fresh_provider(provider,
                                                           project_root))
                    fresh_provider.bind_counter(counter)
                    fresh_inputs = fresh_provider.load_inputs()
                    blocked = admit("geometry_replay")
                    if not blocked:
                        geometry = compute_geometry(
                            fresh_provider, fresh_inputs, tolerance,
                            deadline=deadline, reserve=reserve)
                        fresh_g_arrays = geometry["arrays"]
                        fresh_g_rows = geometry["rows"]
                        fresh_g_grad = geometry["grad_rows"]
                        unit_hint = "fd_replay"
                        blocked = admit("fd_replay")
                        if not blocked:
                            fd = compute_fd(fresh_provider, fresh_inputs,
                                            fresh_g_arrays, tolerance,
                                            deadline=deadline,
                                            reserve=reserve)
                            fresh_d_arrays = fd["arrays"]
                            fresh_d_rows = fd["rows"]
                            unit_hint = "recompute"
                            recompute_done = True
                except (TimeoutError, FkCallBudgetExceeded) as exc:
                    blocked = [f"{type(exc).__name__}"]
                except Exception as exc:  # noqa: BLE001
                    blocked = [f"recompute_exception:{type(exc).__name__}"]
            if blocked:
                verification += [f"f_unit_not_admitted:{unit_hint}:{p}"
                                 for p in blocked]
            process = {"start_method": "in_process", "pid": os.getpid(),
                       "ppid": os.getppid(), "python": sys.executable,
                       "exit_reason": "ok" if not blocked else "aborted"}
        elif not verification:
            blocked = admit("recompute_backend")
            if blocked:
                verification += [f"f_unit_not_admitted:recompute_backend:{p}"
                                 for p in blocked]
            else:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    verification.append(
                        "f_unit_not_admitted:recompute_backend:unit_deadline")
                else:
                    spec = {
                        "schema": "w1_real_fk_recompute_v1",
                        "mode": mode, "root": str(root), "stage": stage,
                        "attempt": attempt,
                        "registration_sha256": registration_sha,
                        "deadline_perf": deadline,
                        "deadline_epoch": time.time() + remaining,
                        "reserve_s": reserve,
                        "project_root": str(project_root),
                        "device": getattr(provider, "device", "cpu"),
                        "seed": getattr(provider, "seed", None),
                        "fixture_path": (str(provider.fixture_path)
                                         if getattr(provider, "fixture_path",
                                                    None) else None),
                        "fixture_sha256": getattr(provider,
                                                  "fixture_sha256", None),
                        "tolerance": dict(tolerance),
                    }
                    worker_meta, worker_problems = _run_recompute_worker(
                        spec, out_dir, remaining, Path(project_root))
                    for unit in worker_meta.get("units") or []:
                        if isinstance(unit, Mapping):
                            units.append(dict(unit))
                    process = dict(worker_meta)
                    verification += worker_problems
                    # A timed-out or killed worker reports no delta; those
                    # calls are unknown, so the reservation must stay
                    # unsettled instead of being settled at zero.
                    delta_settled = isinstance(
                        worker_meta.get("call_delta"), Mapping)
                    if not delta_settled:
                        verification.append("f_call_settlement_unknown")
                    runtime = worker_meta.get("runtime")
                    if mode == "real":
                        if not isinstance(runtime, Mapping):
                            verification.append("f_worker_runtime_missing")
                        else:
                            after_facts = runtime.get("after")
                            verification += [
                                f"f_worker_runtime_mismatch:{p}"
                                for p in runtime_facts_problems(after_facts)]
                    if not worker_problems and worker_meta.get(
                            "exit_reason") == "ok":
                        geometry_rows = _artifact_read_json(
                            out_dir / "fresh_geometry_rows.json",
                            "fresh_geometry_rows",
                            "w1_real_fk_fresh_rows_v1", verification)
                        fd_rows = _artifact_read_json(
                            out_dir / "fresh_fd_rows.json",
                            "fresh_fd_rows", "w1_real_fk_fresh_rows_v1",
                            verification)
                        fresh_g_rows = _artifact_rows(
                            geometry_rows, "rows", "fresh_geometry_rows",
                            verification)
                        fresh_g_grad = _artifact_rows(
                            geometry_rows, "gradient_rows",
                            "fresh_geometry_rows", verification)
                        fresh_d_rows = _artifact_rows(
                            fd_rows, "rows", "fresh_fd_rows", verification)
                        fresh_g_arrays = _artifact_npz(
                            out_dir / "fresh_geometry_arrays.npz",
                            "fresh_geometry_arrays", verification)
                        fresh_d_arrays = _artifact_npz(
                            out_dir / "fresh_fd_arrays.npz",
                            "fresh_fd_arrays", verification)
                        saved_facts = g_paths.get("reference_facts")
                        fresh_facts = out_dir / "fresh_reference_facts.json"
                        if saved_facts and Path(saved_facts).is_file() \
                                and fresh_facts.is_file():
                            try:
                                if read_json(Path(saved_facts)) != read_json(
                                        fresh_facts):
                                    verification.append(
                                        "reference_facts_drifted")
                            except (OSError, ValueError):
                                verification.append(
                                    "reference_facts_unreadable")
                        recompute_done = (
                            fresh_g_rows is not None
                            and fresh_d_rows is not None
                            and fresh_g_arrays is not None
                            and fresh_d_arrays is not None)
                    raw_delta = worker_meta.get("call_delta") or {}
                    for kind, value in raw_delta.items():
                        if kind not in CALL_CAPS \
                                or not isinstance(value, int) \
                                or isinstance(value, bool) or value < 0:
                            verification.append(
                                f"f_worker_call_unknown:{kind}")
                            continue
                        try:
                            counter.add(kind, int(value))
                        except FkCallBudgetExceeded:
                            verification.append(f"f_worker_call_cap:{kind}")
        after = counter.snapshot()["counts"]
        f_delta = {k: int(after[k]) - int(before[k]) for k in after}
        if reservation_path is not None:
            if delta_settled:
                try:
                    _complete_unit_calls(reservation_path, f_delta,
                                         complete=not verification)
                except (OSError, ValueError, TypeError) as exc:
                    verification.append(
                        f"f_reservation_write_failed:{type(exc).__name__}")
            else:
                # Leave ``consumed`` unset: an interrupted worker's calls are
                # unknown and a later attempt must not get a fresh quota.
                verification.append("f_reservation_left_unsettled")
        if recompute_done:
            for kind in ("loss_assembler", "production_fk_batch",
                         "production_vjp", "mj_forward", "mj_jac_site",
                         "fd_pose_eval"):
                if int(f_delta.get(kind, 0)) == 0:
                    verification.append(f"f_call_delta_zero:{kind}")

    # 3. arrays
    if g_arrays is not None and g_grad is not None:
        saved_arrays = {**g_arrays, **g_grad}
        if fresh_g_arrays is not None:
            bitwise_keys = [k for k in saved_arrays if k.startswith((
                "decode_", "prod_fk_", "parent_fk_", "fk_target_points_",
                "fk_pred_points_", "grad_production_raw_", "x0",
                "prediction_", "kernel64_"))]
            verification += _array_compare_problems(
                saved_arrays, fresh_g_arrays, bitwise_keys=bitwise_keys)
            # The gradient file is also compared as its own partition so a
            # schema-valid but substituted file cannot hide behind the merge.
            _fresh_geo, fresh_grad = _split_fk_arrays(fresh_g_arrays)
            verification += _array_compare_problems(
                g_grad, fresh_grad,
                bitwise_keys=[k for k in g_grad if k in bitwise_keys])
    if d_arrays is not None and fresh_d_arrays is not None:
        verification += _array_compare_problems(
            d_arrays, fresh_d_arrays, bitwise_keys=())
    # 4. rows and all detail fields
    saved_grad_by_key = {str(r["key"]): r for r in g_saved_grad}
    if fresh_g_rows is not None:
        saved_by_key = {str(r["key"]): r for r in g_saved_rows}
        fresh_main = {str(r["key"]): r for r in fresh_g_rows}
        fresh_grad_by_key = {str(r["key"]): r for r in (fresh_g_grad or [])}
        if set(saved_by_key) != set(fresh_main):
            verification.append("geometry_row_key_set")
        else:
            for key in sorted(saved_by_key):
                verification += _compare_row_fields(
                    fresh_main[key], saved_by_key[key], key)
        # The separately saved gradient table is bound to both the main table
        # and the fresh recomputation; existing alone is not evidence.
        if set(saved_grad_by_key) != set(fresh_grad_by_key):
            verification.append("gradient_row_key_set")
        else:
            for key in sorted(saved_grad_by_key):
                verification += _compare_row_fields(
                    fresh_grad_by_key[key], saved_grad_by_key[key], key)
        for key, row in sorted(saved_grad_by_key.items()):
            main_row = saved_by_key.get(key)
            if main_row is None:
                verification.append(f"{key}:gradient_row_not_in_main_table")
            else:
                verification += [
                    f"{problem}:gradient_vs_main_table"
                    for problem in _compare_row_fields(row, main_row, key)]
    if fresh_d_rows is not None:
        saved_d = {str(r["key"]): r for r in d_saved_rows}
        fresh_d_by_key = {str(r["key"]): r for r in fresh_d_rows}
        if set(saved_d) != set(fresh_d_by_key):
            verification.append("fd_row_key_set")
        else:
            for key in sorted(saved_d):
                verification += _compare_row_fields(
                    fresh_d_by_key[key], saved_d[key], key)
    # 5. original numeric failures are preserved, not hidden by replay
    failures = sorted({
        str(r["key"]) for r in [*g_saved_rows, *g_saved_grad, *d_saved_rows]
        if r.get("status") == "mismatch"})
    gaps += sorted({
        str(r["key"]) for r in [*g_saved_rows, *g_saved_grad, *d_saved_rows]
        if r.get("status") == "domain_gap"})
    complete = not verification and not gaps
    covered = complete and not failures
    if verification:
        conclusion = "audit_incomplete"
    elif covered:
        conclusion = ("synthetic_contract_passed"
                      if mode == "synthetic"
                      else "real_saved_fk_covered_contract_passed")
    else:
        conclusion = ("synthetic_pipeline_mismatch"
                      if mode == "synthetic"
                      else "real_saved_fk_mismatch")
    adir = stage_dir(root, stage) / attempt
    adir.mkdir(parents=True, exist_ok=True)
    verification_path = adir / "independent_verification.json"
    write_json(verification_path, {
        "schema": "w1_real_fk_independent_verification_v1",
        "attempt": attempt, "mode": mode,
        "verification_problems": verification,
        "failures": failures, "gaps": gaps,
        "units": units,
        "recompute_process": process,
        "artifact_hashes": {
            "g_expected": expected_g, "d_expected": expected_d,
            "g_actual": {
                "arrays": _file_sha(g_paths.get("arrays")),
                "gradient_arrays": _file_sha(g_paths.get("gradient_arrays")),
                "rows": _file_sha(g_paths.get("rows")),
            },
            "d_actual": {
                "arrays": _file_sha(d_paths.get("arrays")),
                "rows": _file_sha(d_paths.get("rows")),
            },
        },
        "complete_evidence": complete, "covered_contract_passed": covered,
        "calls": counter.snapshot(),
        "boundaries": dict(BOUNDARIES),
    })
    summary = {
        "schema": "w1_real_fk_audit_summary_v1",
        "audit_schema": AUDIT_SCHEMA,
        "mode": mode,
        "conclusion": conclusion,
        "complete_evidence": complete,
        "covered_contract_passed": covered,
        "failures": failures,
        "gaps": gaps,
        "verification_problems": verification,
        "counts": row_counts,
        "units": units,
        "recompute_process": {
            k: process.get(k) for k in
            ("start_method", "pid", "ppid", "python", "exit_reason",
             "elapsed_s", "fixture_sha256")
        } if process else {},
        "calls": counter.snapshot(),
        "zero_counters": {k: 0 for k in ZERO_COUNTERS},
        "coverage": {"objective_coverage": "partial"},
        "boundaries": dict(BOUNDARIES),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    summary_path = adir / "fk_audit_final.json"
    write_json(summary_path, summary)
    return {
        "passed": covered,
        "complete_evidence": complete,
        "covered_contract_passed": covered,
        "conclusion": conclusion,
        "verification_problems": verification,
        "failures": failures,
        "gaps": gaps,
        "summary": summary,
        "report": str(summary_path),
        "verification_path": str(verification_path),
        "call_snapshot": counter.snapshot(),
        "call_delta": f_delta,
        "units": units,
        "recompute_process": process,
    }


# ---------------------------------------------------------------------------
# authorization / admission
# ---------------------------------------------------------------------------

AUTHORIZATION_REQUIRED = (
    "new_question_and_cohort",
    "fixed_windows_and_logical_inputs",
    "prediction_leaf_fk_gradient_only",
    "state_fd_matrix",
    "tolerance_and_domain_gates",
    "locked_assets_and_order",
    "budgets_and_attempts",
    "engineering_counterexamples",
    "partial_coverage_acknowledged",
)
ZERO_DECLARATIONS = (
    "training_updates", "parameter_fd", "ddim_calls", "checkpoint_loads",
    "conditioner_calls", "model_calls", "parameter_backward",
    "optimizer_steps", "checkpoint_writes", "contact_branch_executions",
)


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
        "budgets": {
            stage: {"budget_s": spec["budget_s"],
                    "attempts": spec["attempts"]}
            for stage, spec in STAGE_BUDGETS.items()
        },
        "state_fd": {"states": 32, "rows": 4672,
                     "pose_evaluations": 9344,
                     "steps": list(FD_STEPS),
                     "families": ["tangent35", "normalized38"]},
        "parent_registration_sha256": PARENT_FILES["registration"][1],
        "draft_sha256": None,
        "tolerance_evidence_sha256": None,
        "admission_package_sha256": None,
        "notes": ("Template only. The user must create a separate record with "
                  "authorized=true, every checklist item true, exact frozen "
                  "budgets/attempts, explicit JSON integer zero counters and "
                  "the bound parent/draft/tolerance/admission hashes."),
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
            if not _num(rec.get("budget_s")) \
                    or float(rec["budget_s"]) != spec["budget_s"]:
                problems.append(f"authorization_budget_size:{stage}")
            if type(rec.get("attempts")) is not int \
                    or rec["attempts"] != spec["attempts"]:
                problems.append(f"authorization_attempts:{stage}")
    fd = record.get("state_fd")
    if not isinstance(fd, Mapping):
        problems.append("authorization_state_fd_missing")
    else:
        if fd.get("states") != 32 or fd.get("rows") != 4672 \
                or fd.get("pose_evaluations") != 9344:
            problems.append("authorization_state_fd_shape")
        steps = fd.get("steps")
        if (not isinstance(steps, list) or len(steps) != 2
                or any(not _num(v) for v in steps)
                or [float(v) for v in steps] != list(FD_STEPS)):
            problems.append("authorization_state_fd_steps")
    for key, want in (
        ("parent_registration_sha256", PARENT_FILES["registration"][1]),
    ):
        if record.get(key) != want:
            problems.append(f"authorization_binding:{key}")
    for key in ("draft_sha256", "tolerance_evidence_sha256",
                "admission_package_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            problems.append(f"authorization_binding:{key}")
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
    read_scope = package.get("read_scope")
    if not isinstance(read_scope, Mapping):
        problems.append("admission_read_scope_missing")
    else:
        for key in ("window_values_loaded", "checkpoint_torch_load",
                    "cohort_created", "registered_attempt_created"):
            if read_scope.get(key) is not False:
                problems.append(f"admission_read_scope:{key}")
        for key in ("conditioner_forward_calls", "model_forward_calls",
                    "vjp_calls"):
            if read_scope.get(key) != 0:
                problems.append(f"admission_read_scope:{key}")
    for name, rec in (package.get("parent_anchors") or {}).items():
        if not isinstance(rec, Mapping) or rec.get("match") is not True:
            problems.append(f"admission_parent_anchor:{name}")
    counts = package.get("key_counts") or {}
    if counts.get("geometry") != 122 or counts.get("fd") != 4672 \
            or counts.get("states") != 32:
        problems.append("admission_key_counts")
    impl = package.get("implementation") or {}
    for rel, sha in (impl.get("files") or {}).items():
        if not isinstance(sha, str):
            problems.append(f"admission_implementation_missing:{rel}")
    tolerance = package.get("tolerance") or {}
    if tolerance.get("mode") != "synthetic_calibrated" \
            or not tolerance.get("evidence_sha256"):
        problems.append("admission_tolerance_not_calibrated")
    runtime = package.get("runtime")
    if not isinstance(runtime, Mapping) \
            or runtime.get("requirements") != RUNTIME_REQUIREMENTS:
        problems.append("admission_runtime_contract")
    else:
        if runtime.get("applied") is not True:
            problems.append("admission_runtime_not_applied")
        if runtime.get("observed_after_setup") != RUNTIME_REQUIREMENTS:
            problems.append("admission_runtime_observation")
    return problems


def build_admission_package(project_root: Path,
                            tolerance_contract: Mapping[str, Any],
                            *, tolerance_contract_path: Path | None = None,
                            tolerance_contract_sha256: str | None = None
                            ) -> dict:
    """Metadata-only admission package; no real array/asset/checkpoint read."""
    root = Path(project_root)
    contract_problems = tolerance_contract_problems(
        tolerance_contract, recompute=True)
    if contract_problems:
        raise ValueError(f"tolerance contract refused: {contract_problems}")
    if (tolerance_contract.get("mode") != "synthetic_calibrated"
            or not tolerance_contract.get("evidence_sha256")):
        raise ValueError("admission requires a completed calibration")
    problems: list[str] = []
    # Demonstrate in this process that the frozen production runtime can be
    # applied; a real R0 would otherwise fail on the first attempt.
    runtime_check = configure_production_runtime()
    problems += [f"runtime_apply:{p}" for p in runtime_check["problems"]]
    environment = environment_facts()
    parent: dict[str, dict] = {}
    for name, (rel, want) in PARENT_FILES.items():
        p = root / rel
        actual = sha256_file(p) if p.is_file() else None
        parent[name] = {"path": path_for_record(p), "expected": want,
                        "actual": actual,
                        "match": actual is not None and actual == want}
        if actual is None:
            problems.append(f"parent_missing:{name}")
        elif actual != want:
            problems.append(f"parent_drifted:{name}")
    # Parent window binding (metadata only).
    parent_windows = None
    try:
        reg = read_json(root / PARENT_FILES["registration"][0])
        keys = reg.get("window_keys") or reg.get("windows")
        if isinstance(keys, list) and keys:
            parent_windows = [
                {"demo": str(k.get("demo")), "start": int(k.get("start")),
                 "end": int(k.get("end")), "T": int(k.get("T"))}
                for k in keys if isinstance(k, Mapping)
            ]
    except (OSError, ValueError, TypeError):
        problems.append("parent_registration_unreadable")
    fixed = [{"demo": w.demo, "start": w.start, "end": w.end, "T": w.T}
             for w in fixed_windows()]
    if parent_windows is None:
        problems.append("parent_window_binding_missing")
    elif parent_windows != fixed:
        problems.append("parent_window_binding_mismatch")
    catalog = key_catalog()
    impl = implementation_identity(root)
    missing_impl = [rel for rel, sha in impl["files"].items()
                    if not isinstance(sha, str)]
    if missing_impl:
        problems.append(f"implementation_files_missing:{missing_impl}")
    full_chain = {k: int(STAGE_NOMINAL["G"][k] + STAGE_NOMINAL["D"][k]
                         + STAGE_NOMINAL["F"][k]) for k in CALL_CAPS}
    for kind, value in full_chain.items():
        if value > int(CALL_CAPS[kind]):
            problems.append(f"full_chain_call_cap:{kind}")
    notes = []
    subjects = sorted({w.demo.split("_")[0] for w in fixed_windows()})
    if len(subjects) == 1:
        notes.append(
            f"both candidate demos are from subject {subjects[0]}; no "
            "cross-subject coverage is claimed")
    notes.append(
        "near-pi rotations are a registered negative gate; they are excluded "
        "from tolerance calibration but not from real evidence")
    notes.append(
        "F is a new process/backend with disk re-read; it is not a third "
        "mathematical implementation of the reference")
    package = {
        "schema": ADMISSION_SCHEMA,
        "status": {
            "mode": "pre_authorization_readonly",
            "authorized": False,
            "registered": False,
            "run": "not_run",
            "package_passed": not problems,
            "section_9_2_checked": False,
        },
        "passed": not problems,
        "problems": problems,
        "cohort": {
            "cohort_id": COHORT_ID,
            "execution_id": EXECUTION_ID,
            "audit_schema": AUDIT_SCHEMA,
            "accounting_schema": ACCOUNTING_SCHEMA,
            "history_root": str(HISTORY_ROOT),
            "stage_budgets": {k: dict(v) for k, v in STAGE_BUDGETS.items()},
            "total_budget_s": TOTAL_BUDGET_S,
            "call_caps": dict(CALL_CAPS),
            "full_chain_calls": full_chain,
            "budget_transfer_from_old_runs": False,
        },
        "tolerance": {
            "path": (path_for_record(tolerance_contract_path)
                     if tolerance_contract_path is not None else None),
            "sha256": tolerance_contract_sha256,
            "mode": tolerance_contract.get("mode"),
            "evidence_sha256": tolerance_contract.get("evidence_sha256"),
            "contract": dict(tolerance_contract),
        },
        "parent_anchors": parent,
        "parent_windows": parent_windows,
        "windows": fixed,
        "key_counts": {"geometry": catalog["geometry_count"],
                       "fd": catalog["fd_count"],
                       "states": catalog["fd_state_count"]},
        "key_hashes": {
            "geometry": catalog["geometry_multiset_sha256"],
            "fd": catalog["fd_keys_sha256"],
        },
        "schemas": {
            "geometry": geometry_array_schema(),
            "gradient": gradient_array_schema(),
            "fd": fd_array_schema(),
        },
        "nominal_workload": {
            stage: dict(STAGE_NOMINAL[stage]) for stage in ("G", "D", "F")
        },
        "fixed_matrix": {
            "models": list(MODELS),
            "roles": list(ROLES),
            "logical_inputs": list(LOGICAL_INPUTS),
            "sites": list(SITE_NAMES),
            "fd_steps": list(FD_STEPS),
            "fd_local_frames": list(FD_LOCAL_FRAMES),
            "fd_row_count": FD_ROW_COUNT,
            "fd_pose_evaluations": FD_POSE_EVALUATIONS,
            "fk_denominator_batch": FK_DENOMINATOR_BATCH,
            "fk_denominator_window": FK_DENOMINATOR_WINDOW,
        },
        "weights": LossWeights(w_fk=2.0, w_contact=0.0).as_dict(),
        "zero_counters": {k: 0 for k in ZERO_COUNTERS},
        "implementation": impl,
        "environment": environment,
        "runtime": {
            "requirements": dict(RUNTIME_REQUIREMENTS),
            "observed_before_setup": dict(runtime_check["before"]),
            "observed_after_setup": dict(runtime_check["after"]),
            "applied": runtime_check["applied"],
            "problems": list(runtime_check["problems"]),
            "applied_by": [
                "scripts/audit_w1_real_fk.py --real "
                "(before provider construction)",
                "ireset.eval.w1_real_fk_audit._recompute_worker_main "
                "(spawned F worker)",
            ],
            "verified_by": [
                "R0 environment gate + provider runtime facts",
                "F parent/worker runtime facts comparison",
            ],
        },
        "coverage": {
            "objective_coverage": "partial",
            "audited": [
                "training_decode_clip", "production_float32_fk",
                "cpu_float64_fk_kernel", "mujoco_geometry_reference",
                "fifteen_site_order", "fk_scalar_and_elements",
                "fk_35d_and_38d_state_fd", "fk_input_gradient",
                "parent_fk_bitwise_binding",
            ],
            "not_audited": [
                "network_parameter_gradient", "accumulation",
                "gradient_clip", "adamw", "optimizer_update",
                "model_selection", "ddim_sampling", "end_to_end_quality",
                "asset_truthfulness_vs_real_robot", "collision_contact",
            ],
        },
        "boundaries": dict(BOUNDARIES),
        "known_limitations": [
            "the independent MuJoCo reference and the production torch FK "
            "share the same locked assets; only inter-asset consistency is "
            "audited",
            "F reuses the same NumPy/MuJoCo reference formulas; independence "
            "comes from a fresh process, disk re-read and recomputation",
            "near-pi float32 quaternion conversion remains a negative gate",
            "the FD matrix covers 32 fixed states, not all 480 prediction "
            "states or the full production Jacobian",
            "training updates, parameter gradients and optimizer state are "
            "not audited; parameter_fd=0",
        ],
        "authorization": {
            "required": True,
            "authorized": False,
            "required_fields": list(AUTHORIZATION_REQUIRED),
            "zero_declarations": list(ZERO_DECLARATIONS),
            "budgets_expected": {
                stage: {"budget_s": spec["budget_s"],
                        "attempts": spec["attempts"]}
                for stage, spec in STAGE_BUDGETS.items()
            },
            "template": authorization_template(),
        },
        "read_scope": {
            "metadata_only": True,
            "window_values_loaded": False,
            "checkpoint_torch_load": False,
            "asset_closure_verified": False,
            "asset_closure_deferred_to": "R0 under registration",
            "conditioner_forward_calls": 0,
            "model_forward_calls": 0,
            "vjp_calls": 0,
            "cohort_created": False,
            "registered_attempt_created": False,
            "note": ("file hashes and frozen JSON metadata only; no array, "
                     "asset, cache or checkpoint was opened"),
        },
        "r0_obligations": {
            "asset_closure": ("hash the four locked assets and every mesh/"
                              "include reference; empty or missing closure "
                              "stops real mode"),
            "parent_consumer_state": ("consume parent R0/P/F disk rows and "
                                      "summary, not only the seven file "
                                      "hashes"),
            "input_schema": ("dtype/shape/nonempty/finite on every bound "
                             "array plus effective float32 constants and "
                             "6D/clamp domain facts"),
            "call_accounting": ("production FK is counted at the call site; "
                                "unit reservations are persisted before the "
                                "work and not returned on interruption"),
            "f_process": ("fresh spawned Python subprocess; pid/ppid/python "
                          "recorded; per-unit admission and deadline checks "
                          "run inside the worker"),
            "stage_gate": ("G/D are re-consumed before D/F; a budget-revoked "
                           "stage stops the chain instead of passing"),
            "runtime": ("TF32 off / matmul precision highest / autocast off "
                        "are applied by the real entry and the spawned F "
                        "worker and re-verified in R0 and F"),
        },
        "review_notes": notes,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return package


def write_package(out_root: Path, package: Mapping[str, Any]
                  ) -> tuple[Path, str]:
    out = Path(out_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    index = 1
    while (out / f"attempt_{index:03d}").exists():
        index += 1
    attempt = out / f"attempt_{index:03d}"
    attempt.mkdir()
    body = dict(package)
    body["attempt"] = attempt.name
    path = attempt / "admission_package.json"
    write_json(path, body)
    return path, sha256_file(path)


# ---------------------------------------------------------------------------
# registered execution
# ---------------------------------------------------------------------------


def full_chain_calls() -> dict[str, int]:
    return {kind: int(STAGE_NOMINAL["G"][kind] + STAGE_NOMINAL["D"][kind]
                      + STAGE_NOMINAL["F"][kind]) for kind in CALL_CAPS}


def _rewrite_summary_incomplete(path: Path, reason: str,
                                extra_gaps: Sequence[str] = ()) -> dict:
    try:
        summary = read_json(path) if Path(path).is_file() else {}
    except (OSError, ValueError):
        summary = {}
    if not isinstance(summary, dict):
        summary = {}
    summary["producer_conclusion"] = summary.get("conclusion")
    summary["conclusion"] = "audit_incomplete"
    summary["complete_evidence"] = False
    summary["covered_contract_passed"] = False
    summary["revoked_reason"] = reason
    gaps = summary.get("gaps")
    summary["gaps"] = ((list(gaps) if isinstance(gaps, list) else [])
                       + [str(g) for g in extra_gaps] + [reason])
    write_json(path, summary)
    return summary


def withdraw_attempt_claim(root: Path, stage: str, attempt: str,
                           report_path: Path | None, row: Mapping[str, Any],
                           gaps: Sequence[str], *, reason: str,
                           t0: float, budget: Mapping[str, Any]) -> tuple[dict,
                                                                         list[str],
                                                                         dict]:
    """Withdraw a pass/evidence-complete claim and charge the withdrawal.

    The withdrawal rewrite and row persistence are measured against the same
    attempt budget.  If coverage cannot be proven the row stays ``publishing``
    (unknown history) and the caller must return failure; a terminal row whose
    cost is known to be understated is never published.
    """
    problems: list[str] = []
    writes: list[float] = []

    def timed(fn, *args):
        start = time.perf_counter()
        try:
            fn(*args)
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"withdraw_write_failed:{type(exc).__name__}")
        writes.append(time.perf_counter() - start)

    summary = {}
    if report_path is not None and Path(report_path).is_file():
        try:
            summary = _rewrite_summary_incomplete(
                Path(report_path), reason, gaps)
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"summary_revoke_write_failed:{type(exc).__name__}")
    else:
        problems.append("summary_revoke_target_missing")
    elapsed = max(0.0, time.perf_counter() - t0)
    base_cost = float(row.get("measured_wall_s") or 0.0)
    if not _nonneg(base_cost):
        base_cost = 0.0
    remaining = float(budget.get("remaining_s", math.inf))

    def reserve() -> float:
        return max(SEAL_TAIL_RESERVE_S, 2.0 * max(writes, default=0.0))

    def persist(status: str, cost: float, tail: float) -> dict:
        updated = _report_hashes({
            **dict(row),
            "status": status,
            "complete": False,
            "incomplete_reason": reason,
            "revoked_reason": reason,
            "measured_wall_s": float(cost),
            "sampled_wall_s": elapsed,
            "seal_tail_reserve_s": float(tail),
        })
        timed(upsert_index, root, stage, attempt, updated)
        timed(upsert_cost, root, stage, attempt, float(cost),
              {"complete": False, "incomplete_reason": reason,
               "revoked_reason": reason, "sampled_wall_s": elapsed,
               "seal_tail_reserve_s": float(tail),
               "call_balance": updated.get("call_balance")}, status)
        return updated

    def unknown() -> None:
        if "withdraw_cost_unverifiable" not in problems:
            problems.append("withdraw_cost_unverifiable")

    def publishing() -> dict:
        cost = max(base_cost,
                   max(0.0, time.perf_counter() - t0) + reserve())
        return persist("publishing", cost, reserve())

    # Phase 1: the withdrawal starts non-consumable.  A terminal incomplete
    # row is only written after the cost of every write so far is covered.
    updated = publishing()
    for _ in range(SEAL_VERIFY_ROUNDS):
        post = max(0.0, time.perf_counter() - t0)
        tail = reserve()
        if post + tail >= remaining:
            unknown()
            return publishing(), problems, summary
        if post <= float(updated.get("measured_wall_s", 0.0)):
            break
        cost = max(base_cost, post + tail)
        if cost >= remaining:
            unknown()
            return persist("publishing", cost, tail), problems, summary
        updated = persist("publishing", cost, tail)
    else:
        unknown()
        return publishing(), problems, summary

    # Phase 2: terminal incomplete under the same coverage rule; a write that
    # outruns its own estimate is re-priced or the row falls back to
    # publishing (unknown history), never to an understated terminal cost.
    for _ in range(SEAL_VERIFY_ROUNDS):
        post = max(0.0, time.perf_counter() - t0)
        tail = reserve()
        cost = max(base_cost, post + tail)
        if cost >= remaining:
            unknown()
            return persist("publishing", cost, tail), problems, summary
        updated = persist("incomplete", cost, tail)
        post2 = max(0.0, time.perf_counter() - t0)
        tail2 = reserve()
        if post2 + tail2 >= remaining:
            unknown()
            return persist("publishing", max(base_cost, post2 + tail2),
                           tail2), problems, summary
        if post2 <= float(updated.get("measured_wall_s", 0.0)):
            if problems:
                break
            return updated, problems, summary
    unknown()
    return publishing(), problems, summary


def _fresh_provider(provider: FkProvider, project_root: Path) -> FkProvider:
    """New provider/backend for F; reuses the already registered fixture."""
    if getattr(provider, "mode", None) == "synthetic":
        fresh = SyntheticFkProvider(seed=provider.seed)
        fresh.fixture_path = provider.fixture_path
        fresh.fixture_sha256 = provider.fixture_sha256
        return fresh
    return RealFkProvider(project_root,
                          device=getattr(provider, "device", "cuda"))


def run_registered_audit(run_root: Path, provider: FkProvider, *, mode: str,
                         tolerance: Mapping[str, Any],
                         authorization_sha: str | None = None,
                         authorization: Mapping[str, Any] | None = None,
                         admission_package: Mapping[str, Any] | None = None,
                         admission_sha: str | None = None,
                         fixed_root: Path = HISTORY_ROOT,
                         project_root: Path = PROJECT_ROOT) -> dict:
    run_root = Path(run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if mode not in ("synthetic", "real"):
        raise ValueError(f"unknown mode {mode}")
    contract_problems = tolerance_contract_problems(tolerance)
    if contract_problems:
        raise SystemExit(f"tolerance contract refused: {contract_problems}")
    prior = {kind: 0 for kind in CALL_CAPS}
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
        if tolerance.get("mode") != "synthetic_calibrated" \
                or not tolerance.get("evidence_sha256"):
            raise SystemExit("real audit requires a completed calibration")
        if authorization.get("tolerance_evidence_sha256") != tolerance.get(
                "evidence_sha256"):
            raise SystemExit("authorization does not bind this calibration")
        if admission_package is None or admission_sha is None:
            raise SystemExit("real audit requires a passed admission package")
        pkg_problems = admission_package_problems(admission_package)
        if pkg_problems:
            raise SystemExit(f"admission package refused: {pkg_problems}")
        if authorization.get("admission_package_sha256") != admission_sha:
            raise SystemExit("authorization does not bind this admission package")
        if admission_package.get("tolerance", {}).get("evidence_sha256") != \
                tolerance.get("evidence_sha256"):
            raise SystemExit("admission package does not bind this calibration")
        current_impl = implementation_identity(project_root)
        package_files = (admission_package.get("implementation") or {}).get(
            "files") or {}
        if package_files != current_impl.get("files"):
            raise SystemExit(
                "implementation drifted since the admission package was built")
        if authorization.get("draft_sha256") != current_impl["files"].get(
                PREREG_DRAFT):
            raise SystemExit("authorization does not bind the current draft")
        prior = prior_call_counts(run_root, fixed_root=fixed_root, mode=mode)
        if prior is None:
            raise SystemExit("call history unknown: refusing a fresh quota")
        full = full_chain_calls()
        insufficient = {kind: {"prior": prior[kind], "required": full[kind],
                               "cap": CALL_CAPS[kind]}
                        for kind in CALL_CAPS
                        if prior[kind] + full[kind] > CALL_CAPS[kind]}
        if insufficient:
            raise SystemExit(
                f"cumulative call budget insufficient: {insufficient}")
    write_cohort(run_root, mode=mode, authorization_sha=authorization_sha)
    counter = CallCounter(CALL_CAPS, dict(prior))
    if mode == "real":
        runtime = configure_production_runtime()
        if runtime["problems"]:
            raise SystemExit(
                f"production runtime cannot be applied: {runtime['problems']}")
        setattr(provider, "runtime", runtime)
    binder = getattr(provider, "bind_counter", None)
    if callable(binder):
        # Without this the G/D stages would silently record delta=0 and only
        # F's worker delta would reach the cohort ledger.
        binder(counter)
    base_counts = dict(prior)
    fixture_seconds = None
    if mode == "synthetic":
        t_fx = time.perf_counter()
        provider.prepare_fixture(run_root / "fixture")
        fixture_seconds = time.perf_counter() - t_fx
    results: dict[str, Any] = {}
    seals: dict[str, Any] = {}
    scope = {"windows": [w.as_dict() for w in fixed_windows()],
             "roles": list(ROLES), "models": list(MODELS)}

    def stop(problems: Sequence[str], gaps: Sequence[str],
             failures: Sequence[str] = (),
             summary_path: str | None = None) -> dict:
        return {"mode": mode, "conclusion": "audit_incomplete",
                "complete_evidence": False, "covered_contract_passed": False,
                "problems": list(problems), "failures": list(failures),
                "gaps": list(gaps), "stages": results, "seals": seals,
                "summary_path": summary_path,
                "fixture_build_s": fixture_seconds,
                "calls": counter.snapshot(),
                "zero_counters": {k: 0 for k in ZERO_COUNTERS},
                "runtime": getattr(provider, "runtime", None),
                "boundaries": dict(BOUNDARIES)}

    def sealed_hashes(row: Mapping[str, Any] | None,
                      paths: Mapping[str, Any] | None,
                      keys: Sequence[str]) -> dict:
        artifacts = (row or {}).get("artifacts")
        artifacts = artifacts if isinstance(artifacts, Mapping) else {}
        out = {}
        for key in keys:
            p = (paths or {}).get(key)
            if p:
                out[key] = artifacts.get(path_for_record(Path(p)))
        return out

    # --- R0 -----------------------------------------------------------------
    t0 = time.perf_counter()
    budget = budget_for(run_root, "R0", fixed_root=fixed_root, mode=mode)
    attempt = allocate_attempt(run_root, "R0")
    pre_register_attempt(run_root, "R0", attempt, t0, scope=scope,
                         budget=budget, call_balance=counter.snapshot())
    r0 = run_r0(run_root, "R0", attempt, t0, budget, provider, counter,
                dict(tolerance), mode=mode, project_root=project_root,
                base_counts=base_counts,
                expected_implementation=(
                    admission_package.get("implementation")
                    if admission_package is not None else None))
    results["R0"] = r0
    r0_report = stage_dir(run_root, "R0") / attempt / "admission.json"
    seal = finish_attempt(
        run_root, "R0", attempt, t0, budget,
        {"complete": bool(r0["passed"]),
         "report": path_for_record(r0_report),
         "summary": path_for_record(r0_report),
         "registration_sha256": r0["registration_sha256"]},
        status="done" if r0["passed"] else "incomplete",
        extra={"stage": "R0", "scope": scope,
               "call_balance": counter.snapshot()})
    seals["R0"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                   "problems": seal.problems, "row": seal.row}
    _r0_row, r0_consumer = stage_state(run_root, "R0")
    seals["R0"]["consumer_problems"] = r0_consumer
    if not seal.sealed or not r0["passed"] or r0_consumer:
        blockers = r0["problems"] + seal.problems + r0_consumer
        return stop(blockers, r0["gaps"] + seal.problems + r0_consumer)
    registration_sha = r0["registration_sha256"]

    # --- G ------------------------------------------------------------------
    t0 = time.perf_counter()
    budget = budget_for(run_root, "G", fixed_root=fixed_root, mode=mode)
    attempt = allocate_attempt(run_root, "G")
    pre_register_attempt(run_root, "G", attempt, t0, scope=scope,
                         budget=budget, registration_sha=registration_sha,
                         call_balance=counter.snapshot())
    try:
        g = run_g_stage(run_root, "G", attempt, t0, budget, registration_sha,
                        provider, counter, dict(tolerance))
    except (TimeoutError, FkCallBudgetExceeded) as exc:
        g = {"passed": False, "problems": [type(exc).__name__],
             "failures": [], "gaps": [f"g_stage_aborted:{exc}"],
             "paths": {}, "call_delta": {}}
    results["G"] = g
    g_report = stage_dir(run_root, "G") / attempt / "geometry_rows.json"
    g_artifacts = {}
    for key in ("arrays", "gradient_arrays", "rows", "reference_facts"):
        p = (g.get("paths") or {}).get(key)
        if p and Path(p).is_file():
            g_artifacts[path_for_record(p)] = sha256_file(p)
    seal = finish_attempt(
        run_root, "G", attempt, t0, budget,
        {"complete": bool(g["passed"]),
         "report": path_for_record(g_report) if g_report.is_file() else None,
         "summary": path_for_record(g_report) if g_report.is_file() else None,
         "registration_sha256": registration_sha,
         "artifacts": g_artifacts},
        status="done" if g["passed"] else "incomplete",
        extra={"stage": "G", "scope": scope,
               "call_balance": counter.snapshot()})
    seals["G"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                  "problems": seal.problems, "row": seal.row}
    g_row, g_consumer = stage_state(run_root, "G")
    seals["G"]["consumer_problems"] = g_consumer
    if (not seal.terminal or not seal.sealed or g_consumer
            or not g.get("paths")):
        blockers = seal.problems + g_consumer
        if not g.get("paths"):
            blockers.append("g_artifacts_missing")
        return stop(g["problems"] + blockers, g["gaps"] + blockers,
                    g.get("failures", []))

    # --- D ------------------------------------------------------------------
    t0 = time.perf_counter()
    budget = budget_for(run_root, "D", fixed_root=fixed_root, mode=mode)
    attempt = allocate_attempt(run_root, "D")
    pre_register_attempt(run_root, "D", attempt, t0, scope=scope,
                         budget=budget, registration_sha=registration_sha,
                         call_balance=counter.snapshot())
    try:
        d = run_d_stage(run_root, "D", attempt, t0, budget, registration_sha,
                        {**g, "artifact_hashes": sealed_hashes(
                            g_row, g.get("paths"),
                            ("arrays", "gradient_arrays"))},
                        provider, counter, dict(tolerance))
    except (TimeoutError, FkCallBudgetExceeded) as exc:
        d = {"passed": False, "problems": [type(exc).__name__],
             "failures": [], "gaps": [f"d_stage_aborted:{exc}"],
             "paths": {}, "call_delta": {}}
    results["D"] = d
    d_report = stage_dir(run_root, "D") / attempt / "fd_rows.json"
    d_artifacts = {}
    for key in ("arrays", "rows"):
        p = (d.get("paths") or {}).get(key)
        if p and Path(p).is_file():
            d_artifacts[path_for_record(p)] = sha256_file(p)
    seal = finish_attempt(
        run_root, "D", attempt, t0, budget,
        {"complete": bool(d["passed"]),
         "report": path_for_record(d_report) if d_report.is_file() else None,
         "summary": path_for_record(d_report) if d_report.is_file() else None,
         "registration_sha256": registration_sha,
         "artifacts": d_artifacts},
        status="done" if d["passed"] else "incomplete",
        extra={"stage": "D", "scope": scope,
               "call_balance": counter.snapshot()})
    seals["D"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                  "problems": seal.problems, "row": seal.row}
    d_row, d_consumer = stage_state(run_root, "D")
    seals["D"]["consumer_problems"] = d_consumer
    if (not seal.terminal or not seal.sealed or d_consumer
            or not d.get("paths")):
        blockers = seal.problems + d_consumer
        if not d.get("paths"):
            blockers.append("d_artifacts_missing")
        return stop(d["problems"] + blockers, d["gaps"] + blockers,
                    d.get("failures", []))

    # --- F ------------------------------------------------------------------
    # A budget-revoked prerequisite is not a numeric negative result: F must
    # not spend new backend/FK work on it.  Re-consume G/D immediately before
    # registering F.
    prerequisite = []
    for stage_name, row in (("G", g_row), ("D", d_row)):
        _row, state_problems = stage_state(run_root, stage_name)
        if state_problems:
            prerequisite += [f"prerequisite_not_consumable:{stage_name}:{p}"
                             for p in state_problems]
    if prerequisite:
        return stop(prerequisite, prerequisite,
                    g.get("failures", []) + d.get("failures", []))
    t0 = time.perf_counter()
    budget = budget_for(run_root, "F", fixed_root=fixed_root, mode=mode)
    attempt = allocate_attempt(run_root, "F")
    pre_register_attempt(run_root, "F", attempt, t0, scope=scope,
                         budget=budget, registration_sha=registration_sha,
                         call_balance=counter.snapshot())
    try:
        f = run_f_stage(run_root, "F", attempt, t0, budget, registration_sha,
                        g, d, provider, counter, dict(tolerance),
                        project_root=project_root, recompute="subprocess",
                        g_expected=sealed_hashes(
                            g_row, g.get("paths"),
                            ("arrays", "gradient_arrays", "rows")),
                        d_expected=sealed_hashes(
                            d_row, d.get("paths"), ("arrays", "rows")))
    except (TimeoutError, FkCallBudgetExceeded) as exc:
        f = {"passed": False, "complete_evidence": False,
             "covered_contract_passed": False,
             "conclusion": "audit_incomplete",
             "verification_problems": [f"f_stage_aborted:{exc}"],
             "failures": [], "gaps": [f"f_stage_aborted:{exc}"],
             "summary": {}, "report": None, "verification_path": None,
             "units": [], "recompute_process": {}}
    results["F"] = f
    f_report = Path(f["report"]) if f.get("report") else None
    f_artifacts = {}
    if f.get("verification_path"):
        p = Path(f["verification_path"])
        if p.is_file():
            f_artifacts[path_for_record(p)] = sha256_file(p)
    seal = finish_attempt(
        run_root, "F", attempt, t0, budget,
        {"complete": bool(f["complete_evidence"]),
         "report": path_for_record(f_report) if f_report else None,
         "summary": path_for_record(f_report) if f_report else None,
         "registration_sha256": registration_sha,
         "artifacts": f_artifacts},
        status="done" if f["complete_evidence"] else "incomplete",
        extra={"stage": "F", "scope": scope,
               "call_balance": counter.snapshot()})
    seals["F"] = {"sealed": seal.sealed, "terminal": seal.terminal,
                  "problems": seal.problems, "row": seal.row}
    consumer_row, consumer_problems = stage_state(run_root, "F")
    prerequisite_states: dict[str, list[str]] = {}
    prerequisite_problems: list[str] = []
    for stage_name in ("G", "D"):
        _row, state_problems = stage_state(run_root, stage_name)
        prerequisite_states[stage_name] = state_problems
        prerequisite_problems += [
            f"prerequisite_not_consumable:{stage_name}:{p}"
            for p in state_problems]
    seals["F"]["prerequisite_consumer_problems"] = prerequisite_states
    conclusion = f["conclusion"]
    complete = bool(f["complete_evidence"])
    covered = bool(f["covered_contract_passed"])
    revoke_problems: list[str] = []
    if (not seal.terminal or not seal.sealed or consumer_problems
            or prerequisite_problems):
        revoke_problems = (seal.problems
                           + [p for p in consumer_problems
                              if p != "attempt_not_complete"]
                           + prerequisite_problems)
        conclusion = "audit_incomplete"
        complete = False
        covered = False
        updated_row, withdraw_problems, _ = withdraw_attempt_claim(
            run_root, "F", attempt, f_report, seal.row,
            ["f_seal_or_consumer_not_consumable", *revoke_problems],
            reason="f_seal_or_consumer_not_consumable",
            t0=t0, budget=budget)
        seals["F"]["row"] = updated_row
        revoke_problems += withdraw_problems
    call_matrix_problems: list[str] = []
    if mode == "synthetic":
        expected_calls = full_chain_calls()
        expected_calls["scene_setup_forward"] = 0
        final_counts = counter.snapshot()["counts"]
        call_matrix_problems = [
            f"synthetic_call_matrix:{kind}:"
            f"{int(final_counts.get(kind, 0))}!={int(expected_calls[kind])}"
            for kind in CALL_CAPS
            if int(final_counts.get(kind, 0)) != int(expected_calls[kind])]
    if call_matrix_problems:
        conclusion = "audit_incomplete"
        complete = False
        covered = False
    return {
        "mode": mode,
        "conclusion": conclusion,
        "complete_evidence": complete,
        "covered_contract_passed": covered,
        "problems": list(f.get("verification_problems", []))
        + seal.problems + revoke_problems + call_matrix_problems,
        "failures": f.get("failures", []),
        "gaps": f.get("gaps", []) + seal.problems + revoke_problems
        + call_matrix_problems,
        "stages": results,
        "seals": seals,
        "prerequisite_consumer_states": prerequisite_states,
        "units": f.get("units", []),
        "recompute_process": f.get("recompute_process", {}),
        "summary": f.get("summary", {}),
        "summary_path": f.get("report"),
        "fixture_build_s": fixture_seconds,
        "calls": counter.snapshot(),
        "zero_counters": {k: 0 for k in ZERO_COUNTERS},
        "runtime": getattr(provider, "runtime", None),
        "boundaries": dict(BOUNDARIES),
    }


def synthetic_self_test(out_root: Path, *, seed: int = 20261028) -> dict:
    """Run the complete harness on the analytic fixture; no real assets."""
    run_root = Path(out_root).expanduser().resolve()
    if run_root.exists() and any(run_root.iterdir()):
        n = 1
        while (run_root / f"attempt_{n:03d}").exists():
            n += 1
        run_root = run_root / f"attempt_{n:03d}"
    run_root.mkdir(parents=True, exist_ok=True)
    provider = SyntheticFkProvider(seed=seed)
    contract = build_default_tolerance_contract()
    result = run_registered_audit(
        run_root, provider, mode="synthetic", tolerance=contract,
        fixed_root=HISTORY_ROOT)
    result["run_root"] = path_for_record(run_root)
    result["tolerance_contract"] = contract
    write_json(run_root / "audit_result.json", result)
    return result


# === END ===







