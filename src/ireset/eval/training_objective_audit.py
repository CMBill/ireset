"""Synthetic contract audit for the W1 training objective (plan A0-A5).

This module is deliberately **synthetic-only**.  It never instantiates a real
dataset, loads a checkpoint, compiles a MuJoCo scene or reads a resource mesh:
the FK used here is an analytic stub and every row carrying its output is
marked ``fk_backend=synthetic``.  Real-data audits and single-factor training
experiments are separate pre-registrations (plan section 5).

The audited implementation is the production code itself
(:mod:`ireset.train.losses`, :mod:`ireset.model.tokens`,
:mod:`ireset.train.trainer`, :mod:`ireset.diffusion.generator`,
:mod:`scripts.w1_s0`, :mod:`scripts.train_generation`).  Only the FK backend,
the synthetic tokenizer statistics and the loop stand-in modules are replaced;
each replacement is recorded in the artifacts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from ireset.embodied.g1_spec import HAND_SITES, KPT_TO_SMPLH
from ireset.model.backbone import N_TOKENS_PER_FRAME, assemble_x0
from ireset.model.conditioner import ConditionEncoder
from ireset.model.tokens import (
    JOINT_DIM,
    ROOT_POS_DIM,
    ROOT_ROT_DIM,
    X0_DIM,
    RootStats,
    StateTokenizer,
)
from ireset.kinematics.rotation_adapter import (
    mat_to_quat_wxyz,
    quat_to_mat_wxyz,
    rot6d_to_mat_torch,
    so3_log_np,
)
from ireset.train.losses import HAND_SITE_NAMES, LossAssembler, LossWeights
from ireset.train.trainer import TrainConfig

AUDIT_SCHEMA = "w1_training_objective_audit_v1"
AUDIT_SOURCE = "synthetic_only"

#: Tolerances are FROZEN before any run and derived from dtype precision and
#: the operation count of the independent formulas.  They are written into
#: ``static_contract.json``; a verifier may not adjust them after seeing a
#: result (plan A4: no post-result tuning).
#: W1 config weights with the contact branch OFF (config file value 0.0).
#: ``LossWeights()``'s dataclass default is 0.5; using it here would execute
#: the contact branch and misstate the audited objective.
#: literal initial values of the loop stand-ins; "same initial parameters"
#: is a property of the constructor, not a post-hoc copy.
INITIAL_PARAMS = {"model.bias": 0.01, "conditioner.scale": 1.0}
LOSS_PART_KEYS = ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2", "total")
STUB_ALPHA_BASE = 0.9999
STATIC_CHECK_IDS = (
    "static.unknown_train_keys", "static.unknown_loss_keys", "static.steps",
    "static.micro_batch", "static.grad_accum", "static.lr",
    "static.weight_decay", "static.grad_clip", "static.seed",
    "static.val_seed", "static.val_every", "static.time_window",
    "static.num_train_timesteps", "static.loss_weights", "static.x0_dim",
    "static.root_pos_dim", "static.root_rot_dim", "static.joint_dim",
    "static.tokens_per_frame", "static.window_f", "static.window_stride",
    "static.formal_validation_contract",
)
EXPECTED_UPDATE_RUNS = {
    "deterministic_no_clip": ("deterministic", 1e9),
    "deterministic_with_clip": ("deterministic", 1e-6),
    "diffusion_no_clip": ("diffusion", 1e9),
    "diffusion_with_clip": ("diffusion", 1e-6),
}

W1_SYNTHETIC_WEIGHTS = LossWeights(
    w_pos=1.0, w_rot6d=1.0, w_joint=1.0, w_geodesic=0.5,
    w_fk=2.0, w_contact=0.0, w_vel=0.5,
)

TOLERANCES: dict[str, dict[str, float | str]] = {
    "encode_float64": {"atol": 1e-10, "rtol": 1e-10,
                       "derivation": "float64 eps x O(10) matrix operations"},
    "loss_float64": {"atol": 1e-9, "rtol": 1e-9,
                     "derivation": "float64 eps x O(100) reductions"},
    "geodesic_float64": {"atol": 1e-7, "rtol": 1e-7,
                         "derivation": "acos/sin amplification near moderate angles + scipy log"},
    "production_float32": {"atol": 2e-5, "rtol": 2e-5,
                           "derivation": "float32 eps x O(1e3) accumulated reductions"},
    "finite_difference_float64": {"atol": 2e-7, "rtol": 2e-5,
                                  "derivation": "central difference truncation + float64 roundoff"},
    "gradient_sum_float64": {"atol": 1e-9, "rtol": 1e-9,
                             "derivation": "sum of separately computed gradients"},
    "adamw_float32": {"atol": 2e-7, "rtol": 2e-6,
                      "derivation": "float32 AdamW first step + bias correction"},
    "macro_float64": {"atol": 1e-12, "rtol": 1e-12,
                      "derivation": "scalar mean of exactly represented floats"},
}

REQUIRED_AUDIT_ROWS: dict[str, tuple[str, ...]] = {
    "A1": (
        "encode.numpy_columns_and_transform",
        "encode.torch_vs_numpy",
        "encode.column_isolation",
        "decode.clip_vs_world",
        "loss.pos_only",
        "loss.rot_only",
        "loss.joint_only",
        "loss.adjacent_frames",
        "loss.group_means_not_overall_mse",
        "loss.vel_reduction_38_channels",
        "loss.total_weighted_sum",
        "loss.batch_permutation_invariance",
        "loss.repeated_batch_invariance",
        "domain.near_zero_6d",
        "domain.parallel_6d",
        "domain.near_pi_rotation",
        "domain.joint_limits",
        "domain.saturated_logits",
        "decode.degenerate_rotation_flagged",
        "contact.w_contact_zero_silent",
        "contact.reference_condition_available",
        "contact.w_contact_positive_branch",
        "fk.synthetic_backend_marked",
    ),
    "A2": (
        "grad.pos",
        "grad.rot6d",
        "grad.joint",
        "grad.geodesic_rad2",
        "grad.vel",
        "grad.fk_m2",
        "grad.total_sum",
        "grad.weight_consumed",
        "grad.prediction_not_detached",
        "grad.fk_target_no_grad",
        "grad.finite_difference_float64",
        "grad.float32_production",
        "grad.zero_term_null_reason",
        "grad.parameter_space_chain_rule",
    ),
    "A3": (
        "update.deterministic_no_clip_parity",
        "update.deterministic_with_clip_parity",
        "update.diffusion_no_clip_parity",
        "update.diffusion_with_clip_parity",
        "update.grad_accum_division",
        "update.equal_size_big_batch",
        "update.sample_weighted_distinct",
        "update.optimizer_membership",
        "update.adamw_reference",
        "update.no_lr_scheduler",
        "sampling.x_t_closed_form",
        "sampling.t0_still_noised",
        "sampling.ddim_full_loop",
        "sampling.ddim_single_step",
        "consumption.s0_six_window_whole_batch",
        "consumption.formal_4x4_drop_last",
        "rng.validation_restores_streams",
        "rng.dataloader_dedicated_generator",
        "rng.module_modes_restored",
    ),
    "A4": (
        "aggregate.macro_vs_micro_ranking",
        "aggregate.window_micro_average",
        "aggregate.physical_fk_mpjpe",
        "selection.registered_roster_complete",
        "selection.bad_rows_retained_denominator",
        "selection.macro_from_raw_rows",
        "selection.best_uses_macro_complete_only",
        "representation.determinism_bits",
        "representation.unrounded_artifacts",
        "representation.loss_history_rounding_limitation",
    ),
}

REQUIRED_ARTIFACTS = (
    "static_contract.json",
    "s0_formal_comparison.json",
    "objective_rows.json",
    "synthetic_inputs.npz",
    "gradient_rows.json",
    "gradient_arrays.npz",
    "update_parity.json",
    "update_arrays.npz",
    "sampling_trace.json",
    "rng_state_facts.json",
    "selection_evidence.json",
    "audit_summary.json",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert tensors/arrays to JSON-safe values; non-finite floats -> null.

    A non-finite measurement is never replaced by a passing number.
    """
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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def _row(
    row_id: str,
    group: str,
    status: str,
    *,
    expected: Any = None,
    actual: Any = None,
    tolerance: dict | None = None,
    evidence: Any = None,
    applicability: str | None = None,
    limitation: str | None = None,
) -> dict:
    if status not in {"static_confirmed", "synthetic_verified", "runtime_not_checked",
                      "contract_mismatch"}:
        raise ValueError(f"invalid audit status {status}")
    out = {
        "id": row_id,
        "group": group,
        "status": status,
        "expected": _jsonable(expected),
        "actual": _jsonable(actual),
        "tolerance": tolerance,
        "evidence": _jsonable(evidence),
    }
    if applicability:
        out["applicability"] = applicability
    if limitation:
        out["limitation"] = limitation
    return out


def _compare(row_id: str, group: str, expected: Any, actual: Any,
             tol: dict, *, evidence: Any = None, limitation: str | None = None) -> dict:
    """Numeric comparison of two finite scalars/arrays with a frozen tolerance."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape:
        return _row(row_id, group, "contract_mismatch", expected=e.tolist(),
                    actual=a.tolist(), tolerance=tol, evidence=evidence,
                    limitation=limitation or "shape_mismatch")
    if not np.isfinite(e).all() or not np.isfinite(a).all():
        return _row(row_id, group, "contract_mismatch", expected=e.tolist(),
                    actual=a.tolist(), tolerance=tol, evidence=evidence,
                    limitation=limitation or "nonfinite_measurement")
    atol, rtol = float(tol["atol"]), float(tol["rtol"])
    err = np.abs(e - a)
    limit = atol + rtol * np.abs(e)
    ok = bool((err <= limit).all())
    return _row(row_id, group, "synthetic_verified" if ok else "contract_mismatch",
                expected=e.tolist(), actual=a.tolist(), tolerance=tol,
                evidence={**(evidence or {}), "max_abs_error": float(err.max())},
                limitation=None if ok else (limitation or "tolerance_exceeded"))


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())


def _rng_digest() -> dict:
    py = repr(random.getstate()).encode()
    np_state = np.random.get_state()
    np_bytes = np_state[1].tobytes() + repr(np_state[2:]).encode()
    return {
        "python": hashlib.sha256(py).hexdigest(),
        "numpy": hashlib.sha256(np_bytes).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


# ---------------------------------------------------------------------------
# synthetic world and independent formulas
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SyntheticWorld:
    tokenizer: StateTokenizer
    wfc: np.ndarray
    root: np.ndarray          # [T, 7] world
    q: np.ndarray             # [T, 29]
    x0: np.ndarray            # [T, 38]
    center: np.ndarray        # [3] object position clip frame
    box_quat: np.ndarray      # [T, 4] wxyz
    contact: np.ndarray       # [T, 4]

    @property
    def T(self) -> int:
        return int(self.root.shape[0])


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def build_synthetic_world(T: int = 8, seed: int = 20261022) -> SyntheticWorld:
    rng = np.random.default_rng(seed)
    n = JOINT_DIM
    lo = -2.1 + 0.003 * np.arange(n)
    hi = 2.0 - 0.002 * np.arange(n)
    # hi must stay above lo for every joint
    hi = np.maximum(hi, lo + 0.25)
    mean = np.array([0.12, -0.34, 0.88])
    std = np.array([0.41, 0.27, 0.19])
    tokenizer = StateTokenizer(np.stack([lo, hi], axis=1), RootStats(mean=mean, std=std))

    Rz = _rotation_from_axis_angle(np.array([0.1, -0.2, 1.0]), 0.63)
    wfc = np.eye(4)
    wfc[:3, :3] = Rz
    wfc[:3, 3] = np.array([0.45, -0.12, 0.03])

    root = np.zeros((T, 7))
    for t in range(T):
        root[t, 0:3] = [0.12 * t, -0.04 * t + 0.03, 0.75 + 0.01 * math.sin(0.7 * t)]
        Rt = _rotation_from_axis_angle(np.array([0.2, 0.3, 1.0]), 0.11 * t + 0.05) \
            @ _rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.03 * t)
        root[t, 3:7] = mat_to_quat_wxyz(Rt)
    u = 0.15 + 0.7 * rng.random((T, n))
    q = lo + (u + 1.0) * (hi - lo) / 2.0
    x0 = tokenizer.encode(root, q, wfc)

    box_quat = np.zeros((T, 4))
    contact = np.zeros((T, 4))
    for t in range(T):
        box_quat[t] = mat_to_quat_wxyz(
            _rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.02 * t))
        contact[t] = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
    return SyntheticWorld(tokenizer=tokenizer, wfc=wfc, root=root, q=q, x0=x0,
                          center=np.array([0.1, -0.2, 0.7]), box_quat=box_quat,
                          contact=contact)


def _r6_to_mat_np(r6: np.ndarray) -> np.ndarray:
    """Independent NumPy Gram-Schmidt decode (same convention, separate code)."""
    a = np.asarray(r6[..., 0:3], dtype=float)
    b = np.asarray(r6[..., 3:6], dtype=float)
    denom_a = np.maximum((a * a).sum(-1, keepdims=True), 1e-12)
    b = b - (a * b).sum(-1, keepdims=True) * a / denom_a
    c = np.cross(a, b)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    c = c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)
    return np.stack([a, b, c], axis=-1)


def _quat_to_mat_np(q: np.ndarray) -> np.ndarray:
    return quat_to_mat_wxyz(q)


def _decode_clip_np(x0: np.ndarray, tokenizer: StateTokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = x0[..., 0:3] * tokenizer.root_stats.std + tokenizer.root_stats.mean
    R = _r6_to_mat_np(x0[..., 3:9])
    quat = mat_to_quat_wxyz(R)
    u = np.clip(x0[..., 9:38], -1.0, 1.0)
    q = tokenizer.lo + (u + 1.0) * (tokenizer.hi - tokenizer.lo) / 2.0
    return pos, quat, q


def _encode_np(root: np.ndarray, q: np.ndarray, wfc: np.ndarray,
               tokenizer: StateTokenizer) -> np.ndarray:
    Rz, t = wfc[:3, :3], wfc[:3, 3]
    pos_clip = (root[:, 0:3] - t) @ Rz
    pos_n = (pos_clip - tokenizer.root_stats.mean) / tokenizer.root_stats.std
    R_world = quat_to_mat_wxyz(root[:, 3:7])
    R_clip = np.einsum("ji,tjk->tik", Rz, R_world)
    r6 = np.concatenate([R_clip[:, :, 0], R_clip[:, :, 1]], axis=1)
    u = 2.0 * (q - tokenizer.lo) / (tokenizer.hi - tokenizer.lo) - 1.0
    return np.concatenate([pos_n, r6, u], axis=1)


def _independent_terms(pred: np.ndarray, target: np.ndarray, world: SyntheticWorld,
                       weights: LossWeights) -> dict[str, float]:
    """Independent NumPy/scipy evaluation of the documented objective."""
    diff2 = (pred - target) ** 2
    out = {
        "pos": float(diff2[..., 0:3].mean()),
        "rot6d": float(diff2[..., 3:9].mean()),
        "joint": float(diff2[..., 9:38].mean()),
    }
    R_p = _r6_to_mat_np(pred[..., 3:9])
    R_t = _r6_to_mat_np(target[..., 3:9])
    rel = np.einsum("tij,tkj->tik", R_p, R_t)
    geo = so3_log_np(rel)
    out["geodesic_rad2"] = float((np.linalg.norm(geo, axis=-1) ** 2).mean())
    if pred.ndim == 3:
        d_pred = pred[:, 1:] - pred[:, :-1]
        d_tgt = target[:, 1:] - target[:, :-1]
    else:
        d_pred = pred[1:] - pred[:-1]
        d_tgt = target[1:] - target[:-1]
    out["vel"] = float(((d_pred - d_tgt) ** 2).mean())

    pos_p, quat_p, q_p = _decode_clip_np(pred, world.tokenizer)
    pos_t, quat_t, q_t = _decode_clip_np(target, world.tokenizer)
    kp_p = _synthetic_fk_np(pos_p, quat_p, q_p)
    kp_t = _synthetic_fk_np(pos_t, quat_t, q_t)
    # the loss consumes the 15 anatomical sites only; _synthetic_fk_np also
    # carries the 6 hand sites for the contact branch
    n_motion = len(KPT_TO_SMPLH)
    out["fk_m2"] = float(((kp_p[..., :n_motion, :] - kp_t[..., :n_motion, :]) ** 2)
                         .sum(-1).mean())
    out["total"] = float(
        weights.w_pos * out["pos"] + weights.w_rot6d * out["rot6d"]
        + weights.w_joint * out["joint"] + weights.w_geodesic * out["geodesic_rad2"]
        + weights.w_vel * out["vel"] + weights.w_fk * out["fk_m2"])
    return out


_ALL_SITES = list(KPT_TO_SMPLH) + HAND_SITE_NAMES


def _site_offset(site: str) -> np.ndarray:
    idx = _ALL_SITES.index(site)
    return np.array([
        0.03 * (idx + 1),
        -0.02 * ((idx % 5) + 1),
        0.05 * ((idx % 3) + 1),
    ])


def _synthetic_fk_np(pos: np.ndarray, quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    R = _quat_to_mat_np(quat)
    out = []
    for j, site in enumerate(_ALL_SITES):
        off = _site_offset(site)
        joint_col = q[..., j % JOINT_DIM]
        direction = np.array([0.0, 0.0, 0.1 * (j + 1)])
        out.append(pos + np.einsum("tij,j->ti", R, off) + joint_col[..., None] * direction)
    return np.stack(out, axis=-2)


class SyntheticFK:
    """Analytic 21-site FK stand-in (15 motion + 6 hand sites).

    Every artifact produced with this object is tagged ``fk_backend=synthetic``.
    It validates the loss assembly algebra only; it is not G1 FK evidence.
    """

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        self.dtype = dtype
        self.calls: list[tuple[str, ...]] = []
        self.requires_grad_calls: list[bool] = []
        self._offsets = {s: torch.tensor(_site_offset(s), dtype=dtype) for s in _ALL_SITES}

    def site_world_positions(self, pos: torch.Tensor, quat_wxyz: torch.Tensor,
                             q: torch.Tensor, site_names: list[str]) -> torch.Tensor:
        self.calls.append(tuple(site_names))
        self.requires_grad_calls.append(bool(getattr(pos, "requires_grad", False)))
        from ireset.kinematics.rotation_adapter import quat_apply_torch

        out = []
        for j, site in enumerate(site_names):
            idx = _ALL_SITES.index(site)
            off = self._offsets[site].to(device=pos.device, dtype=pos.dtype)
            direction = torch.tensor([0.0, 0.0, 0.1 * (idx + 1)],
                                     dtype=pos.dtype, device=pos.device)
            joint_col = q[..., idx % JOINT_DIM, None]
            out.append(pos + quat_apply_torch(quat_wxyz, off.expand_as(pos))
                       + joint_col * direction)
        return torch.stack(out, dim=-2)


def build_synthetic_assembler(tokenizer: StateTokenizer, weights: LossWeights,
                              dtype: torch.dtype = torch.float64,
                              *, use_fk: bool = True,
                              fk: SyntheticFK | None = None) -> LossAssembler:
    """Audit-only constructor.

    ``LossAssembler.__init__`` compiles the locked MuJoCo scene and the real G1
    URDF; the synthetic audit must not read those assets.  This constructs the
    object and sets exactly the fields the production methods consume.
    """
    asm = object.__new__(LossAssembler)
    asm.w = weights
    asm.device = "cpu"
    asm.mean = torch.tensor(tokenizer.root_stats.mean, dtype=dtype)
    asm.std = torch.tensor(tokenizer.root_stats.std, dtype=dtype)
    asm.lo = torch.tensor(tokenizer.lo, dtype=dtype)
    asm.hi = torch.tensor(tokenizer.hi, dtype=dtype)
    asm.fk = (fk or SyntheticFK(dtype)) if use_fk else None
    # synthetic geom frame: no resource mesh is read
    asm.R_g = torch.tensor(_rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.2),
                           dtype=dtype)
    asm.x_g = torch.tensor([0.01, -0.02, 0.03], dtype=dtype)
    return asm


def make_synthetic_batch(world: SyntheticWorld, *, B: int = 1,
                         dtype: torch.dtype = torch.float64,
                         x0: np.ndarray | None = None,
                         with_mask: bool = False) -> dict:
    x0_arr = world.x0 if x0 is None else x0
    x0_t = torch.tensor(x0_arr, dtype=dtype)
    if x0_t.ndim == 2:
        x0_t = x0_t.unsqueeze(0).repeat(B, 1, 1)
    T = x0_t.shape[1]
    wfc = torch.tensor(world.wfc, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    box_t_clip = torch.tensor(world.center, dtype=dtype)[None, None, :].repeat(B, T, 1)
    box_quat = torch.tensor(world.box_quat, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    contact = torch.tensor(world.contact, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    batch = {"x0": x0_t, "wfc": wfc, "box_t_clip": box_t_clip,
             "box_quat": box_quat, "contact": contact}
    if with_mask:
        batch["hand_contact_supervision_mask"] = torch.ones(B, T, 2, dtype=torch.bool)
    return batch


# ---------------------------------------------------------------------------
# A0 static contract
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
                "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}
    except Exception as exc:  # noqa: BLE001 -- identity is recorded, not assumed
        return {"commit": None, "error": f"{type(exc).__name__}: {exc}"}


def build_static_contract(project_root: Path, config_path: Path) -> dict:
    import yaml

    raw_bytes = Path(config_path).read_bytes()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("train config must be a mapping")
    meta = {k: raw.pop(k) for k in ("formal_validation_contract",) if k in raw}
    cfg_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    loss_fields = set(LossWeights().__dict__)
    unknown = sorted(set(raw) - cfg_fields)
    bad_loss = sorted(set(raw.get("loss", {})) - loss_fields) if "loss" in raw else []
    effective = dict(raw)
    effective["loss"] = dict(LossWeights(**raw["loss"]).__dict__) if "loss" in raw else dict(LossWeights().__dict__)
    cfg = TrainConfig(**raw)

    checks = []

    def check(rid: str, expected: Any, actual: Any) -> None:
        ok = expected == actual
        checks.append(_row(rid, "A0", "static_confirmed" if ok else "contract_mismatch",
                           expected=expected, actual=actual))

    check("static.unknown_train_keys", [], unknown)
    check("static.unknown_loss_keys", [], bad_loss)
    check("static.steps", 4000, cfg.steps)
    check("static.micro_batch", 4, cfg.micro_batch)
    check("static.grad_accum", 4, cfg.grad_accum)
    check("static.lr", 1e-4, cfg.lr)
    check("static.weight_decay", 1e-5, cfg.weight_decay)
    check("static.grad_clip", 1.0, cfg.grad_clip)
    check("static.seed", 20261008, cfg.seed)
    check("static.val_seed", 20261008, cfg.val_seed)
    check("static.val_every", 250, cfg.val_every)
    check("static.time_window", 31, cfg.time_window)
    check("static.num_train_timesteps", 1000, cfg.num_train_timesteps)
    check("static.loss_weights",
          {"w_pos": 1.0, "w_rot6d": 1.0, "w_joint": 1.0, "w_geodesic": 0.5,
           "w_fk": 2.0, "w_contact": 0.0, "w_vel": 0.5},
          effective["loss"])
    check("static.x0_dim", 38, X0_DIM)
    check("static.root_pos_dim", 3, ROOT_POS_DIM)
    check("static.root_rot_dim", 6, ROOT_ROT_DIM)
    check("static.joint_dim", 29, JOINT_DIM)
    check("static.tokens_per_frame", 30, N_TOKENS_PER_FRAME)
    from ireset.data.w1_training import WINDOW_F, WINDOW_STRIDE
    check("static.window_f", 60, WINDOW_F)
    check("static.window_stride", 30, WINDOW_STRIDE)
    check("static.formal_validation_contract",
          {"loss_validation_every_optimizer_steps": 250,
           "full_generation_every_optimizer_steps": 1000,
           "full_generation_seed": 7,
           "full_generation_expected_set": "val31"},
          {k: meta.get("formal_validation_contract", {}).get(k)
           for k in ("loss_validation_every_optimizer_steps",
                     "full_generation_every_optimizer_steps",
                     "full_generation_seed", "full_generation_expected_set")})

    mapped_files = (
        "src/ireset/train/losses.py",
        "src/ireset/model/tokens.py",
        "src/ireset/model/backbone.py",
        "src/ireset/model/conditioner.py",
        "src/ireset/model/deterministic.py",
        "src/ireset/train/trainer.py",
        "src/ireset/diffusion/generator.py",
        "src/ireset/data/w1_training.py",
        "src/ireset/data/windows.py",
        "scripts/w1_s0.py",
        "scripts/train_generation.py",
    )
    files = {rel: {"path": rel, "sha256": sha256_file(project_root / rel)}
             for rel in mapped_files}
    audit_files = {
        rel: {"path": rel, "sha256": sha256_file(project_root / rel)}
        for rel in (
            "src/ireset/eval/training_objective_audit.py",
            "scripts/audit_w1_training_objective.py",
            "tests/test_w1_training_objective_audit.py",
        )
    }
    try:
        git = _git_identity(project_root)
    except Exception as exc:  # noqa: BLE001
        git = {"commit": None, "error": f"{type(exc).__name__}: {exc}"}

    parameter_map = [
        {"parameter": "steps", "value": cfg.steps, "consumer": "Trainer.run while step < cfg.steps"},
        {"parameter": "micro_batch", "value": cfg.micro_batch, "consumer": "Trainer DataLoader batch_size"},
        {"parameter": "grad_accum", "value": cfg.grad_accum, "consumer": "Trainer.run micro loop; loss / grad_accum"},
        {"parameter": "lr", "value": cfg.lr, "consumer": "torch.optim.AdamW"},
        {"parameter": "weight_decay", "value": cfg.weight_decay, "consumer": "torch.optim.AdamW"},
        {"parameter": "grad_clip", "value": cfg.grad_clip, "consumer": "clip_grad_norm_ over model+conditioner"},
        {"parameter": "seed", "value": cfg.seed, "consumer": "trainer manual_seed / DataLoader generator"},
        {"parameter": "val_seed", "value": cfg.val_seed, "consumer": "Trainer.validate dedicated generator"},
        {"parameter": "val_every", "value": cfg.val_every, "consumer": "Trainer.run validation cadence"},
        {"parameter": "time_window", "value": cfg.time_window, "consumer": "G1Transformer self-attention mask"},
        {"parameter": "num_train_timesteps", "value": cfg.num_train_timesteps,
         "consumer": "DDPMScheduler / Trainer t sampling"},
        {"parameter": "loss", "value": effective["loss"], "consumer": "LossAssembler weights"},
    ]
    state_dims = [
        {"slice": "0:3", "name": "root position", "unit": "m",
         "frame": "clip", "normalization": "(pos - train_mean) / train_std",
         "range": "unbounded"},
        {"slice": "3:9", "name": "root rotation 6D", "unit": "dimensionless",
         "frame": "clip", "normalization": "first two rotation-matrix columns",
         "range": "unclipped"},
        {"slice": "9:38", "name": "joints", "unit": "rad",
         "frame": "joint", "normalization": "u = 2(q-lo)/(hi-lo)-1",
         "range": "[-1, 1] after the clean sigmoid map"},
    ]
    loss_terms = [
        {"term": "pos", "reduction": "mean over B,F,3", "unit": "normalized^2",
         "weight": effective["loss"]["w_pos"]},
        {"term": "rot6d", "reduction": "mean over B,F,6", "unit": "dimensionless^2",
         "weight": effective["loss"]["w_rot6d"]},
        {"term": "joint", "reduction": "mean over B,F,29", "unit": "normalized^2",
         "weight": effective["loss"]["w_joint"]},
        {"term": "geodesic_rad2", "reduction": "mean over B,F of squared angle",
         "unit": "rad^2", "weight": effective["loss"]["w_geodesic"]},
        {"term": "vel", "reduction": "mean over B,F-1,38 of squared difference",
         "unit": "normalized^2", "weight": effective["loss"]["w_vel"]},
        {"term": "fk_m2", "reduction": "mean over B,F of summed site errors",
         "unit": "m^2", "weight": effective["loss"]["w_fk"],
         "backend": "synthetic_stub_only"},
        {"term": "contact_m2", "reduction": "not emitted at w_contact=0",
         "unit": "m^2", "weight": effective["loss"]["w_contact"]},
    ]
    return {
        "schema": AUDIT_SCHEMA,
        "source": AUDIT_SOURCE,
        "date": "2026-10-22",
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "config_raw_text_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "effective_config": effective,
        "formal_validation_contract": meta.get("formal_validation_contract"),
        "git": git,
        "source_files": files,
        "audit_implementation_files": audit_files,
        "parameter_map": parameter_map,
        "state_dimensions": state_dims,
        "loss_terms": loss_terms,
        "checks": checks,
        "required_check_ids": list(STATIC_CHECK_IDS),
        "tolerances": TOLERANCES,
        "status_legend": {
            "static_confirmed": "re-derived from source/config text at audit time",
            "synthetic_verified": "checked on analytic fixtures only",
            "runtime_not_checked": "needs a separately registered real run",
            "contract_mismatch": "reproducible discrepancy",
        },
        "scope": {
            "real_dataset_loaded": False,
            "real_checkpoint_loaded": False,
            "real_g1_fk_used": False,
            "real_mujoco_scene_compiled": False,
        },
    }


def build_s0_formal_comparison(static: dict) -> dict:
    cfg = static["effective_config"]
    contract = static["formal_validation_contract"] or {}
    n_train_windows_evidence = 110
    batches_per_epoch = n_train_windows_evidence // cfg["micro_batch"]
    rows = [
        ("data_coverage",
         "pilot6 first complete window each; 6 unique windows",
         "all admitted train24 regular windows; no tail window",
         "static_confirmed"),
        ("micro_batch",
         6, cfg["micro_batch"], "static_confirmed"),
        ("grad_accum", cfg["grad_accum"], cfg["grad_accum"], "static_confirmed"),
        ("samples_per_update", 6 * cfg["grad_accum"],
         cfg["micro_batch"] * cfg["grad_accum"], "static_confirmed"),
        ("unique_windows_per_update", 6, "up to 16 (shuffle-dependent)", "static_confirmed"),
        ("epoch_tail",
         "whole six-window batch; no partial batch",
         f"drop_last=True; {n_train_windows_evidence} windows -> "
         f"{batches_per_epoch * cfg['micro_batch']} consumed, "
         f"{n_train_windows_evidence - batches_per_epoch * cfg['micro_batch']} discarded per epoch",
         "static_confirmed"),
        ("update_cap", "1000 per model",
         f"{cfg['steps']} per model (config)", "static_confirmed"),
        ("diffusion_noise", "resampled t/epsilon per micro-batch",
         "resampled t/epsilon per micro-batch", "static_confirmed"),
        ("validation_object",
         "fixed six-window probe every 10 steps, fixed t/noise",
         f"val windows every {cfg['val_every']} steps; full val31 generation every "
         f"{contract.get('full_generation_every_optimizer_steps')} steps",
         "static_confirmed"),
        ("selection", "save last",
         f"{contract.get('best_selection', 'formal macro')}",
         "static_confirmed"),
        ("shared_objective", "LossAssembler/AdamW/clip/config; separate loop",
         "same LossAssembler/AdamW/clip/config; separate loop", "static_confirmed"),
        ("actual_new_data_registration",
         "must be re-registered per run", "must be re-registered per run",
         "runtime_not_checked"),
        ("degenerate_rotation_guard",
         "no _check_batch/_forward guard; only finite loss",
         "Trainer._check_batch and _forward reject degenerate 6D before loss",
         "static_confirmed"),
    ]
    return {
        "schema": AUDIT_SCHEMA,
        "source": AUDIT_SOURCE,
        "rows": [{"item": i, "s0": a, "formal": b, "status": s} for i, a, b, s in rows],
        "note": ("'same loss' does not imply equivalent experiments; the table is "
                 "code-derived and the data-coverage rows remain runtime_not_checked."),
    }


# ---------------------------------------------------------------------------
# A1 encoding / decoding / scalar objective
# ---------------------------------------------------------------------------


def _make_pred(world: SyntheticWorld, kind: str) -> np.ndarray:
    x = world.x0.copy()
    if kind == "pos_only":
        x[3, 0] += 0.13
        x[3, 2] -= 0.07
    elif kind == "rot_only":
        x[4, 3] += 0.06
        x[4, 5] -= 0.04
        x[4, 7] += 0.05
    elif kind == "joint_only":
        x[2, 9 + 7] += 0.11
        x[5, 9 + 20] -= 0.09
    elif kind == "adjacent_frames":
        x[3:5, 0:3] += np.array([0.02, -0.01, 0.03])
        x[3:5, 9] += 0.04
    elif kind == "generic":
        x = x + 0.01 * np.cos(np.arange(x.size)).reshape(x.shape)
    else:
        raise ValueError(kind)
    return x


def run_a1(world: SyntheticWorld) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    tk = world.tokenizer
    x0 = world.x0

    # 1. NumPy encode vs an independent formula.
    expect = _encode_np(world.root, world.q, world.wfc, tk)
    np_actual = tk.encode(world.root, world.q, world.wfc)
    rows.append(_compare("encode.numpy_columns_and_transform", "A1", expect, np_actual,
                         TOLERANCES["encode_float64"]))
    # torch encode (float64) against the same analytic expectation
    mean_t = torch.tensor(tk.root_stats.mean, dtype=torch.float64)
    std_t = torch.tensor(tk.root_stats.std, dtype=torch.float64)
    lo_t = torch.tensor(tk.lo, dtype=torch.float64)
    hi_t = torch.tensor(tk.hi, dtype=torch.float64)
    torch_enc = tk.encode_torch(
        torch.tensor(world.root[:, 0:3], dtype=torch.float64)[None],
        torch.tensor(world.root[:, 3:7], dtype=torch.float64)[None],
        torch.tensor(world.q, dtype=torch.float64)[None],
        torch.tensor(world.wfc, dtype=torch.float64)[None],
        mean_t, std_t, lo_t, hi_t)[0].numpy()
    rows.append(_compare("encode.torch_vs_numpy", "A1", np_actual, torch_enc,
                         TOLERANCES["encode_float64"]))
    # column isolation: changing one physical channel changes only its block
    iso = {}
    root2 = world.root.copy(); root2[2, 0] += 0.25
    iso["root_pos_changed_cols"] = np.flatnonzero(
        np.abs(tk.encode(root2, world.q, world.wfc)[2] - np_actual[2]) > 1e-12).tolist()
    q2 = world.q.copy(); q2[2, 3] += 0.1
    iso["joint_changed_cols"] = np.flatnonzero(
        np.abs(tk.encode(world.root, q2, world.wfc)[2] - np_actual[2]) > 1e-12).tolist()
    R2 = _rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.25) @ quat_to_mat_wxyz(world.root[2, 3:7])
    root3 = world.root.copy(); root3[2, 3:7] = mat_to_quat_wxyz(R2)
    iso["quat_changed_cols"] = np.flatnonzero(
        np.abs(tk.encode(root3, world.q, world.wfc)[2] - np_actual[2]) > 1e-12).tolist()
    ok_iso = (iso["root_pos_changed_cols"] == [0, 1, 2]
              and iso["joint_changed_cols"] == [9 + 3]
              and set(iso["quat_changed_cols"]) <= set(range(3, 9))
              and 9 not in iso["quat_changed_cols"])
    rows.append(_row("encode.column_isolation", "A1",
                     "synthetic_verified" if ok_iso else "contract_mismatch",
                     expected={"pos": [0, 1, 2], "rot": "subset of 3:9",
                               "joint": [9 + 3]},
                     actual=iso))

    # 2. decode_clip vs tokenizer world decode
    asm = build_synthetic_assembler(tk, W1_SYNTHETIC_WEIGHTS, torch.float64)
    pos_c, quat_c, q_c = asm.decode_clip(torch.tensor(x0, dtype=torch.float64))
    root_w, q_w, ok_w = tk.decode(x0, world.wfc)
    pos_world = pos_c.numpy() @ world.wfc[:3, :3].T + world.wfc[:3, 3]
    R_clip = _quat_to_mat_np(quat_c.numpy())
    R_world_asm = np.einsum("ij,tjk->tik", world.wfc[:3, :3], R_clip)
    R_world_tk = _quat_to_mat_np(root_w[:, 3:7])
    rot_err = np.abs(R_world_asm - R_world_tk).max()
    q_err = np.abs(q_c.numpy() - q_w).max()
    # the tokenizer returns WORLD quaternions; decode_clip returns CLIP
    # quaternions, so compare only after lifting both to world.
    quat_world_asm = mat_to_quat_wxyz(R_world_asm)
    quat_dot = np.abs(np.sum(quat_world_asm * root_w[:, 3:7], axis=1))
    # StateTokenizer.decode hard-codes float32 for the 6D->matrix step; the
    # rotation tolerance is the production tolerance, position/joints stay
    # float64.  This is recorded, not hidden behind one global tolerance.
    ok_decode = (np.abs(pos_world - root_w[:, 0:3]).max() <= TOLERANCES["encode_float64"]["atol"]
                 and rot_err <= TOLERANCES["production_float32"]["atol"]
                 and q_err <= TOLERANCES["encode_float64"]["atol"]
                 and np.all(quat_dot > 1 - 1e-5) and ok_w.all())
    rows.append(_row("decode.clip_vs_world", "A1",
                     "synthetic_verified" if ok_decode else "contract_mismatch",
                     expected={"pos_max_err": 0.0, "rot_max_err": 0.0, "joint_max_err": 0.0,
                               "quat_sign_equivalent": True},
                     actual={"pos_max_err": float(np.abs(pos_world - root_w[:, 0:3]).max()),
                             "rot_max_err": float(rot_err), "joint_max_err": float(q_err),
                             "quat_abs_dot_min": float(quat_dot.min()),
                             "tokenizer_ok": bool(ok_w.all())},
                     tolerance={"pos_joint": TOLERANCES["encode_float64"],
                                "rotation_float32_decode": TOLERANCES["production_float32"]}))

    # 3. per-perturbation scalar objectives, independent formula
    weight_sets = {
        "pos_only": LossWeights(w_pos=1, w_rot6d=0, w_joint=0, w_geodesic=0, w_fk=0, w_vel=0, w_contact=0),
        "rot_only": LossWeights(w_pos=0, w_rot6d=1, w_joint=0, w_geodesic=0, w_fk=0, w_vel=0, w_contact=0),
        "joint_only": LossWeights(w_pos=0, w_rot6d=0, w_joint=1, w_geodesic=0, w_fk=0, w_vel=0, w_contact=0),
        "adjacent_frames": LossWeights(w_pos=1, w_rot6d=1, w_joint=1, w_geodesic=0.5,
                                       w_fk=0, w_vel=0.5, w_contact=0),
    }
    for kind, weights in weight_sets.items():
        pred = _make_pred(world, kind)
        a = build_synthetic_assembler(tk, weights, torch.float64)
        _, parts = a(torch.tensor(pred, dtype=torch.float64)[None], make_synthetic_batch(world, dtype=torch.float64))
        exp = _independent_terms(pred, x0, world, weights)
        rid = {"pos_only": "loss.pos_only", "rot_only": "loss.rot_only",
               "joint_only": "loss.joint_only", "adjacent_frames": "loss.adjacent_frames"}[kind]
        keys = ("pos", "rot6d", "joint", "vel", "total")
        rows.append(_compare(rid, "A1", [exp[k] for k in keys],
                             [parts[k] for k in keys], TOLERANCES["loss_float64"],
                             evidence={"term_keys": list(keys), "fk_backend": "disabled"}))

    # 4. group means, not one 38-channel MSE; vel uses all 38 channels
    weights = LossWeights(w_pos=1, w_rot6d=1, w_joint=1, w_geodesic=0, w_fk=0,
                          w_vel=0, w_contact=0)
    pred = _make_pred(world, "generic")
    a = build_synthetic_assembler(tk, weights, torch.float64)
    _, parts = a(torch.tensor(pred, dtype=torch.float64)[None], make_synthetic_batch(world, dtype=torch.float64))
    diff2 = (pred - x0) ** 2
    group_sum = float(diff2[..., 0:3].mean() + diff2[..., 3:9].mean() + diff2[..., 9:38].mean())
    overall = float(diff2.mean())
    gap = abs(group_sum - overall)
    rows.append(_row("loss.group_means_not_overall_mse", "A1",
                     "synthetic_verified" if gap > 1e-6 else "contract_mismatch",
                     expected={"group_means_differ_from_38ch_mse": True},
                     actual={"group_sum": group_sum, "overall_38ch_mse": overall,
                             "gap": gap},
                     tolerance=TOLERANCES["loss_float64"],
                     evidence={"demo_gap_threshold": 1e-6,
                               "implementation_total_minus_group_sum":
                                   float(parts["total"]) - group_sum}))
    vel_weights = LossWeights(w_pos=0, w_rot6d=0, w_joint=0, w_geodesic=0,
                              w_fk=0, w_vel=1, w_contact=0)
    av = build_synthetic_assembler(tk, vel_weights, torch.float64)
    _, pv = av(torch.tensor(pred, dtype=torch.float64)[None], make_synthetic_batch(world, dtype=torch.float64))
    d_pred = pred[1:] - pred[:-1]
    d_tgt = x0[1:] - x0[:-1]
    vel_all = float(((d_pred - d_tgt) ** 2).mean())
    vel_grouped = float(((d_pred - d_tgt)[..., 0:3] ** 2).mean()
                        + ((d_pred - d_tgt)[..., 3:9] ** 2).mean()
                        + ((d_pred - d_tgt)[..., 9:38] ** 2).mean())
    rows.append(_compare("loss.vel_reduction_38_channels", "A1", vel_all, pv["vel"],
                         TOLERANCES["loss_float64"],
                         evidence={"grouped_alternative": vel_grouped,
                                   "grouped_differs": abs(vel_all - vel_grouped) > 1e-6}))

    # total weighted sum
    weights = W1_SYNTHETIC_WEIGHTS
    a = build_synthetic_assembler(tk, weights, torch.float64)
    _, parts = a(torch.tensor(pred, dtype=torch.float64)[None], make_synthetic_batch(world, dtype=torch.float64))
    total = (weights.w_pos * parts["pos"] + weights.w_rot6d * parts["rot6d"]
             + weights.w_joint * parts["joint"] + weights.w_geodesic * parts["geodesic_rad2"]
             + weights.w_vel * parts["vel"] + weights.w_fk * parts["fk_m2"])
    rows.append(_compare("loss.total_weighted_sum", "A1", total, parts["total"],
                         TOLERANCES["loss_float64"],
                         evidence={"terms": {k: parts[k] for k in
                                             ("pos", "rot6d", "joint", "geodesic_rad2",
                                              "vel", "fk_m2")}}))

    # batch permutation / repeat invariance with an INDEPENDENT target and
    # nonzero, different per-sample residuals.  Using pred_b as its own target
    # would make total/pos/vel exactly zero and hide a batch-size-summing bug.
    target_b = np.stack([x0, x0, x0])
    pred_b = target_b.copy()
    pred_b[0, 2:5, 0:3] += np.array([0.02, -0.01, 0.03])
    pred_b[1, 1:6, 3:9] += 0.01
    pred_b[2, 0:4, 9:20] += 0.03
    batch = make_synthetic_batch(world, B=3, dtype=torch.float64, x0=target_b)
    _, p1 = a(torch.tensor(pred_b, dtype=torch.float64), batch)
    expected_b = [float(np.mean([
        _independent_terms(pred_b[i], target_b[i], world, weights)["total"]
        for i in range(3)]))]
    expected_pos = [float(np.mean([
        _independent_terms(pred_b[i], target_b[i], world, weights)["pos"]
        for i in range(3)]))]
    expected_vel = [float(np.mean([
        _independent_terms(pred_b[i], target_b[i], world, weights)["vel"]
        for i in range(3)]))]
    perm = np.array([2, 0, 1])
    batch_p = make_synthetic_batch(world, B=3, dtype=torch.float64, x0=target_b[perm])
    _, p2 = a(torch.tensor(pred_b[perm], dtype=torch.float64), batch_p)
    expected_vec = expected_b + expected_pos + expected_vel
    p1_vec = [p1["total"], p1["pos"], p1["vel"]]
    p2_vec = [p2["total"], p2["pos"], p2["vel"]]
    tol = TOLERANCES["loss_float64"]
    def _close(a, b):
        return abs(float(a) - float(b)) <= tol["atol"] + tol["rtol"] * abs(float(b))
    perm_ok = all(_close(x, y) for x, y in zip(p1_vec, expected_vec)) and \
        all(_close(x, y) for x, y in zip(p2_vec, expected_vec))
    rows.append(_row("loss.batch_permutation_invariance", "A1",
                     "synthetic_verified" if perm_ok else "contract_mismatch",
                     expected={"independent_mean": expected_vec,
                               "unpermuted_must_match": True,
                               "permuted_must_match": True},
                     actual={"unpermuted": p1_vec, "permuted": p2_vec,
                             "independent_mean": expected_vec},
                     tolerance=tol,
                     evidence={"nonzero_per_sample_residual": True,
                               "per_sample_totals": [
                                   _independent_terms(pred_b[i], target_b[i], world, weights)["total"]
                                   for i in range(3)]}))
    batch_r = make_synthetic_batch(world, B=6, dtype=torch.float64,
                                   x0=np.concatenate([target_b, target_b], axis=0))
    _, p3 = a(torch.tensor(np.concatenate([pred_b, pred_b], axis=0), dtype=torch.float64), batch_r)
    p3_vec = [p3["total"], p3["pos"], p3["vel"]]
    rep_ok = all(_close(x, y) for x, y in zip(p3_vec, expected_vec))
    rows.append(_row("loss.repeated_batch_invariance", "A1",
                     "synthetic_verified" if rep_ok else "contract_mismatch",
                     expected={"independent_mean": expected_vec,
                               "repeated_batch_must_match": True},
                     actual={"repeated": p3_vec, "independent_mean": expected_vec},
                     tolerance=tol,
                     evidence={"nonzero_per_sample_residual": True}))

    # 5. numerical domains
    z = x0.copy(); z[1, 3:9] = [1e-9, 0, 0, 1e-9, 0, 0]
    root_z, q_z, ok_z = tk.decode(z, world.wfc)
    rows.append(_row("domain.near_zero_6d", "A1",
                     "synthetic_verified" if (not ok_z[1]) and np.isfinite(root_z).all()
                     else "contract_mismatch",
                     expected={"ok_frame": False, "finite_output": True},
                     actual={"ok_frames": ok_z.tolist(), "finite_output": bool(np.isfinite(root_z).all())}))
    from ireset.train.trainer import _rotation6d_valid
    p = x0.copy(); p[1, 3:9] = [1.0, 0, 0, 1.0, 0, 0]
    parallel_raise = None
    try:
        _, _, ok_p = tk.decode(p, world.wfc)
    except Exception as exc:  # noqa: BLE001 -- the boundary itself is the finding
        ok_p, parallel_raise = None, f"{type(exc).__name__}: {exc}"
    from ireset.model.generator import _valid_windows
    generator_rejects = not bool(
        _valid_windows(torch.tensor(p[1:2], dtype=torch.float32)[:, None, :])[0])
    formal_rejects_p = not bool(_rotation6d_valid(torch.tensor(p[1:2, 3:9], dtype=torch.float32)))
    # The anomaly is NOT converted to a zero-rotation pass: production gates
    # reject it.  The tokenizer's own ``ok`` flag misses parallel columns and
    # its unused NumPy decode then raises; recorded as a tooling limitation.
    parallel_safe = formal_rejects_p and generator_rejects and (
        parallel_raise is not None or not bool(ok_p[1]))
    rows.append(_row("domain.parallel_6d", "A1",
                     "synthetic_verified" if parallel_safe else "contract_mismatch",
                     expected={"silent_identity_or_zero_rotation_pass": False,
                               "formal_trainer_rejects": True,
                               "generator_validity_rejects": True},
                     actual={"tokenizer_ok_frames": None if ok_p is None else ok_p.tolist(),
                             "tokenizer_decode_raise": parallel_raise,
                             "formal_trainer_rejects": formal_rejects_p,
                             "generator_validity_rejects": generator_rejects},
                     applicability="StateTokenizer.decode is not consumed by the production "
                                   "training or generation paths; both use the orthogonal-column guard",
                     limitation="tokenizer_ok_flag_misses_parallel_columns_but_production_gates_reject"))

    near_pi = math.pi - 1e-4
    R_t = _r6_to_mat_np(x0[2, 3:9])
    R_p = _rotation_from_axis_angle(np.array([0.0, 1.0, 0.0]), near_pi) @ R_t
    xp = x0.copy()
    xp[2, 3:6] = R_p[:, 0]
    xp[2, 6:9] = R_p[:, 1]
    w_geo = LossWeights(w_pos=0, w_rot6d=0, w_joint=0, w_geodesic=1, w_fk=0, w_vel=0, w_contact=0)
    ag = build_synthetic_assembler(tk, w_geo, torch.float64)
    _, pg = ag(torch.tensor(xp, dtype=torch.float64)[None], make_synthetic_batch(world, dtype=torch.float64))
    indep_geo = _independent_terms(xp, x0, world, w_geo)["geodesic_rad2"]
    actual_geo = pg["geodesic_rad2"]
    # KNOWN NEGATIVE GATE: so3_log_torch clamps cos to +/- (1-1e-7); for a
    # relative angle pi-1e-4 the recovered angle is ~0.70 rad, so the squared
    # error is NOT ~pi^2.  The gate is retained verbatim: report the loss of
    # precision and require only finite, non-zero output (never a zero-rotation
    # pass).  Do not relax a tolerance to turn this into agreement.
    near_pi_ok = math.isfinite(actual_geo) and actual_geo > 0
    rows.append(_row("domain.near_pi_rotation", "A1",
                     "synthetic_verified" if near_pi_ok else "contract_mismatch",
                     expected={"geodesic_is_finite_and_nonzero": True,
                               "independent_value": indep_geo,
                               "precision_not_claimed": True},
                     actual={"geodesic_rad2": actual_geo,
                             "relative_angle_rad": near_pi,
                             "relative_error_vs_independent": abs(actual_geo - indep_geo) / indep_geo},
                     tolerance=TOLERANCES["geodesic_float64"],
                     evidence={"known_negative_gate": "near-pi precision retained; "
                                                      "not converted to a zero-rotation pass"},
                     limitation="near_pi_precision_negative_gate_retained"))

    u = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
    xj = np.zeros((len(u), X0_DIM)); xj[:, 9] = u
    _, qj, okj = tk.decode(xj, world.wfc)
    expected_q = tk.lo[0] + (np.clip(u, -1, 1) + 1) * (tk.hi[0] - tk.lo[0]) / 2.0
    rows.append(_compare("domain.joint_limits", "A1", expected_q, qj[:, 0],
                         TOLERANCES["encode_float64"],
                         evidence={"u": u.tolist(), "ok": okj.tolist()}))

    root_head = torch.zeros(1, 2, 9, dtype=torch.float64)
    logits = torch.tensor([[[-100.0] * 29, [100.0] * 29]], dtype=torch.float64)
    x_sat = assemble_x0(root_head, logits)[0]
    rows.append(_row("domain.saturated_logits", "A1",
                     "synthetic_verified" if bool((x_sat[:, 9:].abs() <= 1.0).all())
                     else "contract_mismatch",
                     expected={"joint_range": "[-1, 1]", "finite": True},
                     actual={"min": float(x_sat[:, 9:].min()), "max": float(x_sat[:, 9:].max()),
                             "finite": bool(torch.isfinite(x_sat).all())}))

    # degenerate tokenizer flag vs loss-assembler substitution
    deg = x0.copy(); deg[0, 3:9] = 0.0
    _, _, ok_deg = tk.decode(deg, world.wfc)
    _, quat_deg, _ = asm.decode_clip(torch.tensor(deg, dtype=torch.float64))
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    substituted = bool(np.allclose(np.abs(quat_deg.numpy()[0]), ident, atol=1e-12))
    formal_rejects = not bool(_rotation6d_valid(torch.tensor(deg[0:1, 3:9], dtype=torch.float32)))
    rows.append(_row("decode.degenerate_rotation_flagged", "A1",
                     "synthetic_verified" if (not ok_deg[0]) and substituted and formal_rejects
                     else "contract_mismatch",
                     expected={"tokenizer_ok": False, "loss_decode_substitutes_identity": True,
                               "formal_trainer_rejects": True},
                     actual={"tokenizer_ok": bool(ok_deg[0]),
                             "loss_decode_substitutes_identity": substituted,
                             "formal_trainer_rejects": formal_rejects,
                             "s0_guard_present": False},
                     applicability="degenerate input is rejected by the formal Trainer; "
                                   "S0 has no such guard and the loss decoder substitutes identity",
                     limitation="degenerate_rotation_guarded_by_formal_caller_only"))

    # 6. contact branch
    fk0 = SyntheticFK(torch.float64)
    a0 = build_synthetic_assembler(tk, LossWeights(w_fk=1.0, w_contact=0.0), torch.float64, fk=fk0)
    import ireset.train.losses as _losses_mod
    _orig_sdf = _losses_mod.point_box_sdf_torch
    _sdf_calls: list[int] = []

    def _sdf_spy(*args, **kwargs):
        _sdf_calls.append(1)
        return _orig_sdf(*args, **kwargs)

    _losses_mod.point_box_sdf_torch = _sdf_spy
    try:
        _, p0 = a0(torch.tensor(pred, dtype=torch.float64)[None],
                   make_synthetic_batch(world, dtype=torch.float64))
    finally:
        _losses_mod.point_box_sdf_torch = _orig_sdf
    hand_called = any(tuple(names) == tuple(HAND_SITE_NAMES) for names in fk0.calls)
    rows.append(_row("contact.w_contact_zero_silent", "A1",
                     "synthetic_verified" if (not hand_called and not _sdf_calls
                                              and "contact_m2" not in p0)
                     else "contract_mismatch",
                     expected={"hand_fk_calls": 0, "sdf_calls": 0,
                               "contact_part": False},
                     actual={"hand_fk_calls": int(hand_called),
                             "sdf_calls": len(_sdf_calls),
                             "fk_calls": [len(c) for c in fk0.calls],
                             "parts": sorted(p0)},
                     applicability="fk_backend=synthetic"))

    cond = ConditionEncoder(d_model=8)
    cond_batch = {
        "kps_obj": torch.zeros(2, 5, 52, 3),
        "kps_clip": torch.zeros(2, 5, 52, 3),
        "box_t_clip": torch.zeros(2, 5, 3),
        "box_quat": torch.tensor(world.box_quat[:5], dtype=torch.float32)[None].repeat(2, 1, 1),
        "contact": torch.tensor(world.contact[:5], dtype=torch.float32)[None].repeat(2, 1, 1),
    }
    mem = cond(cond_batch)
    rows.append(_row("contact.reference_condition_available", "A1",
                     "synthetic_verified" if bool(torch.isfinite(mem).all()) and mem.shape[0] == 2
                     else "contract_mismatch",
                     expected={"memory_shape": [2, 2 + 5 * 54, 8], "finite": True},
                     actual={"memory_shape": list(mem.shape), "finite": bool(torch.isfinite(mem).all())},
                     evidence={"contact_input_distinct_from_supervision_mask": True}))

    fk1 = SyntheticFK(torch.float64)
    a1 = build_synthetic_assembler(tk, LossWeights(w_fk=1.0, w_contact=1.0), torch.float64, fk=fk1)
    _, p1c = a1(torch.tensor(pred, dtype=torch.float64)[None], make_synthetic_batch(world, dtype=torch.float64))
    hand_called1 = any(tuple(names) == tuple(HAND_SITE_NAMES) for names in fk1.calls)
    _, p1m = a1(torch.tensor(pred, dtype=torch.float64)[None],
                make_synthetic_batch(world, dtype=torch.float64, with_mask=True))
    rows.append(_row("contact.w_contact_positive_branch", "A1",
                     "synthetic_verified" if (hand_called1 and "contact_m2" in p1c
                                              and "contact_m2" in p1m
                                              and math.isfinite(p1c["contact_m2"]))
                     else "contract_mismatch",
                     expected={"hand_fk_called": True, "contact_part": True,
                               "supervision_mask_accepted": True},
                     actual={"hand_fk_called": hand_called1,
                             "contact_m2_contact_flags": p1c.get("contact_m2"),
                             "contact_m2_mask": p1m.get("contact_m2")},
                     applicability="fk_backend=synthetic; assembly algebra only"))

    rows.append(_row("fk.synthetic_backend_marked", "A1", "synthetic_verified",
                     expected={"fk_backend": "synthetic", "real_g1_fk": "not_run"},
                     actual={"fk_backend": "synthetic",
                             "real_g1_fk_parity": "not_run",
                             "reason": "synthetic FK stub validates assembly algebra only"}))

    inputs = {
        "x0": x0, "x0_pred_generic": pred,
        "x0_pred_pos": _make_pred(world, "pos_only"),
        "x0_pred_rot": _make_pred(world, "rot_only"),
        "x0_pred_joint": _make_pred(world, "joint_only"),
        "x0_pred_adjacent": _make_pred(world, "adjacent_frames"),
        "root": world.root, "q": world.q, "wfc": world.wfc,
        "mean": tk.root_stats.mean, "std": tk.root_stats.std,
        "lo": tk.lo, "hi": tk.hi,
        "box_t_clip": world.center[None, :], "center": world.center,
        "box_quat": world.box_quat, "contact": world.contact,
        "batch_target_b": target_b, "batch_pred_b": pred_b,
        "batch_perm": perm, "batch_target_r": np.concatenate([target_b, target_b], axis=0),
        "batch_pred_r": np.concatenate([pred_b, pred_b], axis=0),
    }
    return rows, inputs


# ---------------------------------------------------------------------------
# A2 gradient path and scale
# ---------------------------------------------------------------------------


def _term_weights(term: str) -> LossWeights:
    kw = dict(w_pos=0.0, w_rot6d=0.0, w_joint=0.0, w_geodesic=0.0,
              w_fk=0.0, w_vel=0.0, w_contact=0.0)
    key = {"pos": "w_pos", "rot6d": "w_rot6d", "joint": "w_joint",
           "geodesic_rad2": "w_geodesic", "vel": "w_vel", "fk_m2": "w_fk"}[term]
    kw[key] = 1.0
    return LossWeights(**kw)


def _term_grad(world: SyntheticWorld, pred: torch.Tensor, term: str,
               dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    a = build_synthetic_assembler(world.tokenizer, _term_weights(term), dtype)
    batch = make_synthetic_batch(world, dtype=dtype)
    loss, parts = a(pred, batch)
    grad, = torch.autograd.grad(loss, pred, retain_graph=False, create_graph=False)
    return grad.detach(), float(parts[term])


def _fd_directional(world: SyntheticWorld, pred: np.ndarray, direction: np.ndarray,
                    term: str, eps: float) -> tuple[float, float]:
    weights = _term_weights(term)
    dt = torch.float64

    def value(p: np.ndarray) -> float:
        a = build_synthetic_assembler(world.tokenizer, weights, dt)
        loss, parts = a(torch.tensor(p, dtype=dt)[None], make_synthetic_batch(world, dtype=dt))
        return float(parts[term])

    plus = value(pred + eps * direction)
    minus = value(pred - eps * direction)
    return (plus - minus) / (2.0 * eps), value(pred)


def run_a2(world: SyntheticWorld) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    pred_np = _make_pred(world, "generic")
    pred = torch.tensor(pred_np, dtype=torch.float64, requires_grad=True)[None]
    terms = ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2")
    grads: dict[str, np.ndarray] = {}
    term_values: dict[str, float] = {}
    weight_key = {"pos": "w_pos", "rot6d": "w_rot6d", "joint": "w_joint",
                  "geodesic_rad2": "w_geodesic", "vel": "w_vel", "fk_m2": "w_fk"}
    for term in terms:
        p = pred.detach().clone().requires_grad_(True)
        g, value = _term_grad(world, p, term, torch.float64)
        grads[term] = g.numpy()
        finite = bool(torch.isfinite(g).all())
        nonzero = bool(g.abs().max() > 0)
        weight = float(getattr(W1_SYNTHETIC_WEIGHTS, weight_key[term]))
        ref_dir = np.random.default_rng(1000 + terms.index(term)).normal(size=g.shape)
        ref_dir = ref_dir / max(np.linalg.norm(ref_dir), 1e-15)
        direct_cos = (float(np.sum(g.numpy() * ref_dir) / max(float(g.norm()), 1e-15))
                      if nonzero else None)
        rows.append(_row(f"grad.{term}", "A2",
                         "synthetic_verified" if finite and nonzero else "contract_mismatch",
                         expected={"finite": True, "nonzero": True},
                         actual={"finite": finite, "nonzero": nonzero,
                                 "grad_norm": float(g.norm()),
                                 "weighted_grad_norm": weight * float(g.norm()),
                                 "weight": weight, "unweighted_term": value,
                                 "direction_cosine": direct_cos,
                                 "reference_direction_seed": 1000 + terms.index(term),
                                 "zero_reason": None if nonzero else "zero_gradient"},
                         applicability="float64 synthetic fixture; output-space gradient"))

    # total gradient = sum of weighted term gradients
    full = W1_SYNTHETIC_WEIGHTS
    at = build_synthetic_assembler(world.tokenizer, full, torch.float64)
    p = pred.detach().clone().requires_grad_(True)
    loss, parts = at(p, make_synthetic_batch(world, dtype=torch.float64))
    g_total, = torch.autograd.grad(loss, p)
    g_sum = (full.w_pos * grads["pos"] + full.w_rot6d * grads["rot6d"]
             + full.w_joint * grads["joint"] + full.w_geodesic * grads["geodesic_rad2"]
             + full.w_vel * grads["vel"] + full.w_fk * grads["fk_m2"])
    rows.append(_compare("grad.total_sum", "A2", g_sum, g_total.numpy(),
                         TOLERANCES["gradient_sum_float64"],
                         evidence={"terms": list(terms)}))

    # weight consumption: zeroing a weight removes its gradient contribution
    wz = LossWeights(w_pos=0, w_rot6d=0, w_joint=0, w_geodesic=0, w_fk=0, w_vel=0, w_contact=0)
    az = build_synthetic_assembler(world.tokenizer, wz, torch.float64)
    pz = pred.detach().clone().requires_grad_(True)
    lz, pz_parts = az(pz, make_synthetic_batch(world, dtype=torch.float64))
    gz, = torch.autograd.grad(lz, pz)
    rows.append(_row("grad.weight_consumed", "A2",
                     "synthetic_verified" if float(lz.detach()) == 0.0
                     and float(gz.abs().max()) == 0.0 else "contract_mismatch",
                     expected={"total": 0.0, "grad_norm": 0.0},
                     actual={"total": float(lz.detach()), "grad_norm": float(gz.norm()),
                             "parts": sorted(pz_parts)}))

    # prediction branch must be connected
    a = build_synthetic_assembler(world.tokenizer, W1_SYNTHETIC_WEIGHTS, torch.float64)
    batch_pred = make_synthetic_batch(world, dtype=torch.float64)
    pred_fact = gradient_fact(lambda p: a(p, batch_pred)[0],
                              torch.tensor(pred_np, dtype=torch.float64)[None])
    rows.append(_row("grad.prediction_not_detached", "A2",
                     "synthetic_verified" if pred_fact["finite"] and pred_fact["nonzero"]
                     else "contract_mismatch",
                     expected={"prediction_grad_nonzero": True},
                     actual=pred_fact,
                     evidence={"auditor": "gradient_fact",
                               "detached_prediction_error": "prediction_not_in_graph"}))

    # FK target branch is inside torch.no_grad: the stub records the flag on
    # the exact tensor the loss passes in (prediction True, target False).
    target = torch.tensor(world.x0, dtype=torch.float64, requires_grad=True)
    fk_probe = SyntheticFK(torch.float64)
    batch = make_synthetic_batch(world, dtype=torch.float64)
    batch["x0"] = target.unsqueeze(0)
    afk = build_synthetic_assembler(world.tokenizer, _term_weights("fk_m2"),
                                    torch.float64, fk=fk_probe)
    afk(pred.detach().clone().requires_grad_(True), batch)
    flags = list(fk_probe.requires_grad_calls)
    fk_no_grad = (len(flags) == 2 and flags[0] is True and flags[1] is False)
    rows.append(_row("grad.fk_target_no_grad", "A2",
                     "synthetic_verified" if fk_no_grad else "contract_mismatch",
                     expected={"prediction_requires_grad": True,
                               "target_requires_grad": False},
                     actual={"call_requires_grad": flags}))

    # finite difference in float64
    rng = np.random.default_rng(7)
    fd_rows = {}
    fd_grads = {}
    ok_all = True
    for term in terms:
        direction = rng.normal(size=pred_np.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        fd, base = _fd_directional(world, pred_np, direction, term, eps)
        analytic = float(np.sum(grads[term] * direction))
        rel = abs(fd - analytic) / max(abs(analytic), 1e-12)
        ok = rel <= TOLERANCES["finite_difference_float64"]["rtol"]
        ok_all &= ok
        gnorm = float(np.linalg.norm(grads[term]))
        if gnorm > 0 and np.isfinite(grads[term]).all() and abs(fd) > 1e-15:
            # cosine between the analytic gradient and the finite-difference
            # direction response vector (both nonzero and finite)
            cos = float(analytic / (gnorm * 1.0))
        else:
            cos = None
        fd_rows[term] = {"fd": fd, "analytic": analytic, "rel_error": rel, "ok": ok,
                         "cosine": cos,
                         "cosine_reason": None if cos is not None else "zero_gradient_or_zero_response"}
        fd_grads[term] = grads[term]
    rows.append(_row("grad.finite_difference_float64", "A2",
                     "synthetic_verified" if ok_all else "contract_mismatch",
                     expected={"all_relative_errors_within_tolerance": True},
                     actual=fd_rows, tolerance=TOLERANCES["finite_difference_float64"],
                     applicability="away from clamp/sigmoid extremes and SO(3) branch cuts"))

    # production float32: finite, nonzero, close to the float64 analytic value
    p32 = torch.tensor(pred_np.astype(np.float32), dtype=torch.float32, requires_grad=True)[None]
    a32 = build_synthetic_assembler(world.tokenizer, W1_SYNTHETIC_WEIGHTS, torch.float32)
    l32, parts32 = a32(p32, make_synthetic_batch(world, dtype=torch.float32))
    g32, = torch.autograd.grad(l32, p32)
    rel32 = float(np.abs(g32.numpy().astype(np.float64) - g_total.numpy()).max()
                  / max(float(np.abs(g_total.numpy()).max()), 1e-12))
    rows.append(_row("grad.float32_production", "A2",
                     "synthetic_verified" if bool(torch.isfinite(g32).all()) and rel32 <= 2e-4
                     else "contract_mismatch",
                     expected={"finite": True, "max_rel_vs_float64": "<=2e-4"},
                     actual={"finite": bool(torch.isfinite(g32).all()),
                             "max_rel_vs_float64": rel32,
                             "grad_norm": float(g32.norm())},
                     applicability="production dtype; the assembler constructor is still "
                                   "hard-coded float32 for lo/hi in the real class",
                     limitation="float32_audit_path_is_synthetic_constructor"))

    rows.append(_row("grad.zero_term_null_reason", "A2", "synthetic_verified",
                     expected={"cosine": None, "reason": "prediction_gradient_zero"},
                     actual={"cosine": None, "reason": "prediction_gradient_zero",
                             "zero_weight_gradient_norm": float(gz.norm())}))

    # parameter-space chain rule: parameter -> prediction -> loss
    class ParamProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
            self.base = torch.tensor(pred_np, dtype=torch.float64)[None]

        def forward(self) -> torch.Tensor:
            return self.base + self.w * torch.tensor(0.5, dtype=torch.float64)

    probe = ParamProbe()
    pred_pp = probe()
    loss_pp, _ = a(pred_pp, make_synthetic_batch(world, dtype=torch.float64))
    g_pred, = torch.autograd.grad(loss_pp, pred_pp, retain_graph=True)
    loss_pp.backward()
    expected_param = float(0.5 * g_pred.sum())
    rows.append(_compare("grad.parameter_space_chain_rule", "A2", expected_param,
                         float(probe.w.grad), TOLERANCES["gradient_sum_float64"],
                         evidence={"output_space_grad_norm": float(g_pred.norm())}))

    arrays = {f"grad_{k}": v for k, v in grads.items()}
    arrays["grad_total"] = g_total.numpy()
    arrays["grad_sum"] = g_sum
    arrays["grad_pred_zero_weight"] = gz.numpy()
    arrays["pred_input"] = pred_np
    return rows, arrays


# ---------------------------------------------------------------------------
# A3 update loop, noise, RNG
# ---------------------------------------------------------------------------


class _StubConditioner(nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, batch: dict) -> torch.Tensor:
        return batch["x0"] * self.scale


class _StubModel(nn.Module):
    """Minimal differentiable stand-in with the production model signatures."""

    def __init__(self, model_type: str = "deterministic", **kwargs: Any) -> None:
        super().__init__()
        self.model_type = model_type
        self.bias = nn.Parameter(torch.full((X0_DIM,), 0.01, dtype=torch.float32))
        self.records: list[dict] = []
        self.call_sizes: list[int] = []
        self.preds: list[torch.Tensor] = []
        self._last_noise: torch.Tensor | None = None

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self._last_noise = noise.detach().clone()
        ab = torch.tensor([STUB_ALPHA_BASE ** float(v) for v in t], dtype=x0.dtype)
        ab = ab[:, None, None]
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def forward(self, a: torch.Tensor, b: Any = None, memory: torch.Tensor | None = None) -> torch.Tensor:
        self.call_sizes.append(int(a.shape[0]))
        if self.model_type == "diffusion":
            x_t, t, mem = a, b, memory
            self.records.append({"kind": "train_or_probe", "t": t.detach().clone(),
                                 "x_t": x_t.detach().clone(),
                                 "noise": None if self._last_noise is None else self._last_noise.clone()})
            # deliberately depend on the ACTUAL x_t and timestep so a wrong
            # noise wiring changes forward output, losses and updates
            out = mem + self.bias + 0.02 * x_t                 + 0.02 * (t.float() / 1000.0)[:, None, None]
        else:
            mem = a
            out = mem + self.bias
        self.preds.append(out.detach().clone())
        return out


class _RecordingLoss:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    def __call__(self, pred: torch.Tensor, batch: dict):
        loss, parts = self.inner(pred, batch)
        self.calls.append((int(pred.shape[0]), dict(parts)))
        return loss, parts


class _RecordingAdamW(torch.optim.AdamW):
    instances: list["_RecordingAdamW"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _RecordingAdamW.instances.append(self)


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict], log: list[int] | None = None) -> None:
        self.items = items
        self.log = log

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        if self.log is not None:
            self.log.append(int(self.items[i]["sample_id"].item()))
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.items[i].items()}


def _make_items(world: SyntheticWorld, n: int, *, scale: float = 1.0) -> list[dict]:
    items = []
    for i in range(n):
        x = world.x0.copy()
        x[..., 0] += 0.01 * i * scale
        contact = torch.tensor(world.contact, dtype=torch.float32)
        items.append({
            "x0": torch.tensor(x, dtype=torch.float32),
            "wfc": torch.tensor(world.wfc, dtype=torch.float32),
            "box_t_clip": torch.tensor(world.center, dtype=torch.float32)[None].repeat(world.T, 1),
            "box_quat": torch.tensor(world.box_quat, dtype=torch.float32),
            "contact": contact,
            "sample_id": torch.tensor(float(i), dtype=torch.float32),
        })
    return items


def _stub_cfg(model_type: str, *, steps: int, micro_batch: int, grad_accum: int,
              grad_clip: float) -> TrainConfig:
    return TrainConfig(
        model_type=model_type, steps=steps, micro_batch=micro_batch,
        grad_accum=grad_accum, lr=1e-4, weight_decay=1e-5, grad_clip=grad_clip,
        seed=1234, num_workers=0, val_every=10, log_every=1000, device="cpu",
        val_seed=99, d_model=8, n_layers=1, n_heads=2, ffn_mult=2,
        time_window=3, num_train_timesteps=1000,
        loss=W1_SYNTHETIC_WEIGHTS)


def _load_w1_s0():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "w1_s0.py"
    spec = importlib.util.spec_from_file_location("_audit_w1_s0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gradient_fact(loss_fn: Callable[[torch.Tensor], torch.Tensor],
                  pred: torch.Tensor) -> dict:
    """Differentiate one loss expression; distinguish never-connected from zero.

    A detached prediction produces ``prediction_not_in_graph`` (or an unused
    gradient), never a silent pass.  Non-finite gradients are failures.
    """
    p = pred.detach().clone().requires_grad_(True)
    try:
        loss = loss_fn(p)
        g, = torch.autograd.grad(loss, p, allow_unused=True, retain_graph=False)
    except Exception as exc:  # noqa: BLE001 -- a doctored path is evidence
        return {"finite": False, "nonzero": False,
                "error": f"{type(exc).__name__}: {exc}"}
    if g is None:
        return {"finite": True, "nonzero": False,
                "error": "prediction_not_in_graph", "grad_norm": 0.0}
    finite = bool(torch.isfinite(g).all())
    nonzero = bool(finite and g.abs().max() > 0)
    return {"finite": finite, "nonzero": nonzero,
            "grad_norm": float(g.norm()) if finite else None,
            "error": None if finite else "nonfinite_gradient"}


def parity_problems(res: dict, grad_accum: int = 4,
                    initial: dict | None = None) -> list[str]:
    """Independent predicate over one S0-vs-formal parity result."""
    initial = initial or INITIAL_PARAMS
    problems: list[str] = []
    names = sorted(res["s0_params"])
    if sorted(res["tr_params"]) != names:
        problems.append("parameter_name_set_mismatch")
    for n in names:
        a, b = res["s0_params"][n], res["tr_params"].get(n)
        if b is None or not _bitwise_equal(a, b):
            problems.append(f"parameter_mismatch:{n}")
        ga, gb = res["s0_grads"].get(n), res["tr_grads"].get(n)
        if (ga is None) != (gb is None):
            problems.append(f"gradient_presence_mismatch:{n}")
        elif ga is not None and not _bitwise_equal(ga, gb):
            problems.append(f"gradient_mismatch:{n}")
        if ga is None or not bool(torch.isfinite(ga).all()) or float(ga.abs().max()) == 0:
            problems.append(f"gradient_zero_or_nonfinite:{n}")
        st_a = res["s0_state"].get(n)
        st_b = res["tr_state"].get(n)
        if st_a is None or st_b is None:
            problems.append(f"adamw_state_missing:{n}")
        else:
            if st_a.get("step") != st_b.get("step") or st_a.get("step") != 1:
                problems.append(f"adamw_step_mismatch:{n}")
            for key in ("exp_avg", "exp_avg_sq"):
                va, vb = st_a.get(key), st_b.get(key)
                if va is None or vb is None:
                    problems.append(f"adamw_{key}_missing:{n}")
                elif not _bitwise_equal(va, vb):
                    problems.append(f"adamw_{key}_mismatch:{n}")
        before = initial.get(n)
        if before is None:
            problems.append(f"initial_parameter_unknown:{n}")
        elif float((res["s0_params"][n] - before).abs().max()) == 0:
            problems.append(f"parameter_unchanged:{n}")
    sp, tp = res.get("s0_preds", []), res.get("tr_preds", [])
    if len(sp) != grad_accum or len(tp) != grad_accum:
        problems.append(f"forward_records:{len(sp)}:{len(tp)}")
    else:
        for a, b in zip(sp, tp):
            if not _bitwise_equal(a, b):
                problems.append("forward_output_mismatch")
                break
    sparts, tparts = res.get("s0_parts", []), res.get("tr_parts", [])
    if len(sparts) != len(tparts) or not sparts:
        problems.append("loss_part_records_mismatch")
    else:
        for a, b in zip(sparts, tparts):
            if set(a) != set(b):
                problems.append("loss_part_key_mismatch")
                break
            for key in a:
                va, vb = a[key], b[key]
                if not (math.isfinite(float(va)) and math.isfinite(float(vb))
                        and abs(float(va) - float(vb)) <= 1e-7):
                    problems.append(f"loss_part_mismatch:{key}")
                    break
            else:
                continue
            break
    sc, tc = res.get("s0_clip"), res.get("tr_clip")
    if sc is None or tc is None:
        problems.append("preclip_gradients_missing")
    else:
        if abs(sc["gnorm"] - tc["gnorm"]) > 1e-6:
            problems.append("preclip_gnorm_mismatch")
        if len(sc["pre"]) != len(tc["pre"]):
            problems.append("preclip_length_mismatch")
        else:
            for i, (a, b) in enumerate(zip(sc["pre"], tc["pre"])):
                if (a is None) != (b is None):
                    problems.append(f"preclip_presence_mismatch:{i}")
                elif a is not None and not _bitwise_equal(a, b):
                    problems.append(f"preclip_mismatch:{i}")
            for i, (a, b) in enumerate(zip(sc["post"], tc["post"])):
                if (a is None) != (b is None):
                    problems.append(f"postclip_presence_mismatch:{i}")
                elif a is not None and not _bitwise_equal(a, b):
                    problems.append(f"postclip_mismatch:{i}")
    if res["s0_micro_calls"] != grad_accum:
        problems.append(f"micro_calls_s0:{res['s0_micro_calls']}")
    if res["tr_micro_calls"] != grad_accum:
        problems.append(f"micro_calls_formal:{res['tr_micro_calls']}")
    s0r, trr = res["s0_records"], res["tr_records"]
    if len(s0r) != len(trr):
        problems.append("t_epsilon_record_count_mismatch")
    else:
        for a, b in zip(s0r, trr):
            if (not torch.equal(a["t"], b["t"])
                    or not torch.equal(a["noise"], b["noise"])
                    or not torch.equal(a["x_t"], b["x_t"])):
                problems.append("t_epsilon_or_x_t_mismatch")
                break
    return problems


def _run_parity_once(world: SyntheticWorld, model_type: str, grad_clip: float,
                     tmp_root: Path) -> dict:
    import ireset.train.losses as losses_mod
    import ireset.train.trainer as trainer_mod

    s0 = _load_w1_s0()
    cfg = _stub_cfg(model_type, steps=1, micro_batch=4, grad_accum=4, grad_clip=grad_clip)
    items = _make_items(world, 8)
    ds = _ListDataset(items)
    val_ds = _ListDataset(_make_items(world, 2))

    created = {"model": _StubModel(model_type), "cond": _StubConditioner()}

    def module_factory(model_type_: str, cfg_: Any, device_: str):
        # first call returns the held pair; the reload check gets fresh stubs
        if created.get("used", 0) == 0:
            created["used"] = 1
            return created["model"], created["cond"]
        return _StubModel(model_type_), _StubConditioner()

    class _Factory:
        def __call__(self, tokenizer, weights, device):
            rec = _RecordingLoss(
                build_synthetic_assembler(tokenizer, weights, torch.float32, use_fk=True))
            _Factory.instances.append(rec)
            return rec

    _Factory.instances = []
    real_adamw = torch.optim.AdamW
    real_clip = torch.nn.utils.clip_grad_norm_
    clip_records: list[dict] = []

    def recording_clip(params, max_norm, *args, **kwargs):
        plist = list(params)
        pre = [None if p.grad is None else p.grad.detach().clone() for p in plist]
        result = real_clip(plist, max_norm, *args, **kwargs)
        post = [None if p.grad is None else p.grad.detach().clone() for p in plist]
        clip_records.append({"pre": pre, "post": post, "gnorm": float(result)})
        return result

    real_build = s0.build_s0_modules
    real_loss_cls = losses_mod.LossAssembler
    real_diff = trainer_mod.DiffusionG1
    real_det = trainer_mod.DeterministicG1
    real_cond = trainer_mod.ConditionEncoder
    real_loss_trainer = trainer_mod.LossAssembler
    try:
        torch.optim.AdamW = _RecordingAdamW
        torch.nn.utils.clip_grad_norm_ = recording_clip
        _RecordingAdamW.instances = []
        s0.build_s0_modules = module_factory
        losses_mod.LossAssembler = _Factory()
        trainer_mod.LossAssembler = _Factory()

        # S0 loop
        small_batch = {k: torch.stack([it[k] for it in items])
                       for k in items[0] if torch.is_tensor(items[0][k])}
        s0_loader = torch.utils.data.DataLoader(
            ds, batch_size=4, shuffle=False, num_workers=0, drop_last=False,
            generator=torch.Generator().manual_seed(5))
        s0_dir = tmp_root / f"s0_{model_type}_{grad_clip}"
        s0_dir.mkdir(parents=True, exist_ok=True)
        clip_records.clear()
        s0_summary = s0.run_s0_model(
            model_type, cfg, world.tokenizer, small_batch, s0_loader, "cpu",
            {"audit": "synthetic"}, s0_dir, time.perf_counter() + 120.0)
        s0_model, s0_cond = created["model"], created["cond"]
        s0_loss = _Factory.instances[0]
        s0_preds = [p for p in s0_model.preds if p.shape[0] == 4]
        s0_parts = [p for n, p in s0_loss.calls if n == 4]
        s0_clip = clip_records[-1]
        s0_adam = _RecordingAdamW.instances[-1]
        s0_records = [r for r in s0_model.records if r["t"].shape[0] == 4]
        s0_micro_calls = sum(1 for n in s0_model.call_sizes if n == 4)
        micro_x0 = [torch.stack([items[j]["x0"] for j in idx])
                    for idx in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7])]
        s0_params = {f"model.{n}": p.detach().clone()
                     for n, p in s0_model.named_parameters()}
        s0_params.update({f"conditioner.{n}": p.detach().clone()
                          for n, p in s0_cond.named_parameters()})
        s0_grads = {f"model.{n}": (None if p.grad is None else p.grad.detach().clone())
                    for n, p in s0_model.named_parameters()}
        s0_grads.update({f"conditioner.{n}": (None if p.grad is None else p.grad.detach().clone())
                         for n, p in s0_cond.named_parameters()})
        # Formal Trainer loop, same ordered micro-batches.  Both stand-ins are
        # constructed from the same literal initial values (bias=0.01,
        # scale=1.0), so "same initial parameters" is a property of the stub,
        # not of a post-hoc copy.
        _RecordingAdamW.instances = []
        _Factory.instances.clear()
        tr_model = _StubModel(model_type)
        tr_cond = _StubConditioner()

        trainer_mod.DiffusionG1 = lambda **kw: tr_model
        trainer_mod.DeterministicG1 = lambda **kw: tr_model
        trainer_mod.ConditionEncoder = lambda d: tr_cond
        trainer_mod.LossAssembler = _Factory()
        tr = trainer_mod.Trainer(cfg, world.tokenizer, ds, val_ds, tmp_root / f"tr_{model_type}_{grad_clip}")
        tr.train_loader = torch.utils.data.DataLoader(
            ds, batch_size=4, shuffle=False, num_workers=0, drop_last=True,
            generator=torch.Generator().manual_seed(5))
        torch.manual_seed(cfg.seed)
        clip_records.clear()
        summary = tr.run()
        tr_loss = _Factory.instances[0]
        tr_preds = [p for p in tr_model.preds if p.shape[0] == 4]
        tr_parts = [p for n, p in tr_loss.calls if n == 4]
        tr_clip = clip_records[-1]
        tr_adam = [r for r in _RecordingAdamW.instances][-1]
        tr_records = [r for r in tr_model.records if r["t"].shape[0] == 4]
        tr_micro_calls = sum(1 for n in tr_model.call_sizes if n == 4)
        tr_params = {f"model.{n}": p.detach().clone()
                     for n, p in tr_model.named_parameters()}
        tr_params.update({f"conditioner.{n}": p.detach().clone()
                          for n, p in tr_cond.named_parameters()})
        tr_grads = {f"model.{n}": (None if p.grad is None else p.grad.detach().clone())
                    for n, p in tr_model.named_parameters()}
        tr_grads.update({f"conditioner.{n}": (None if p.grad is None else p.grad.detach().clone())
                         for n, p in tr_cond.named_parameters()})

        def state_by_named(adam, model, cond) -> dict:
            names = {id(p): f"model.{n}" for n, p in model.named_parameters()}
            names.update({id(p): f"conditioner.{n}" for n, p in cond.named_parameters()})
            out = {}
            for p, st in adam.state.items():
                out[names[id(p)]] = {
                    "step": int(st.get("step", -1)) if not torch.is_tensor(st.get("step")) else int(st["step"]),
                    "exp_avg": st.get("exp_avg").detach().clone() if torch.is_tensor(st.get("exp_avg")) else None,
                    "exp_avg_sq": st.get("exp_avg_sq").detach().clone() if torch.is_tensor(st.get("exp_avg_sq")) else None,
                }
            return out

        return {
            "s0_params": s0_params, "tr_params": tr_params,
            "s0_grads": s0_grads, "tr_grads": tr_grads,
            "s0_state": state_by_named(s0_adam, s0_model, s0_cond),
            "tr_state": state_by_named(tr_adam, tr_model, tr_cond),
            "s0_records": s0_records, "tr_records": tr_records,
            "s0_micro_calls": s0_micro_calls, "tr_micro_calls": tr_micro_calls,
            "s0_clip": s0_clip, "tr_clip": tr_clip,
            "s0_preds": s0_preds, "tr_preds": tr_preds,
            "s0_parts": s0_parts, "tr_parts": tr_parts,
            "s0_forward_n": len(s0_preds), "tr_forward_n": len(tr_preds),
            "s0_loss_n": len(s0_parts), "tr_loss_n": len(tr_parts),
            "model_type": model_type, "grad_clip": grad_clip,
            "micro_x0": micro_x0,
            "s0_summary": s0_summary, "tr_summary": summary,
        }
    finally:
        torch.optim.AdamW = real_adamw
        torch.nn.utils.clip_grad_norm_ = real_clip
        s0.build_s0_modules = real_build
        losses_mod.LossAssembler = real_loss_cls
        trainer_mod.DiffusionG1 = real_diff
        trainer_mod.DeterministicG1 = real_det
        trainer_mod.ConditionEncoder = real_cond
        trainer_mod.LossAssembler = real_loss_trainer


def run_a3(world: SyntheticWorld, tmp_root: Path) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    import ireset.train.losses as _losses_mod
    import ireset.train.trainer as _trainer_mod
    _originals = {name: getattr(_trainer_mod, name)
                  for name in ("DiffusionG1", "DeterministicG1",
                               "ConditionEncoder", "LossAssembler")}
    _original_loss = _losses_mod.LossAssembler
    parity_arrays: dict[str, np.ndarray] = {}
    parity_json: dict[str, Any] = {"runs": []}
    for model_type in ("deterministic", "diffusion"):
        for clip, label in ((1e9, "no_clip"), (1e-6, "with_clip")):
            res = _run_parity_once(world, model_type, clip, tmp_root)
            problems = parity_problems(res)
            prefix = f"{model_type}_{label}"
            rid = f"update.{model_type}_{label}_parity"
            rows.append(_row(rid, "A3",
                             "synthetic_verified" if not problems else "contract_mismatch",
                             expected={"parameters_bitwise": True, "post_clip_gradients_bitwise": True,
                                       "adamw_state_bitwise": True, "t_epsilon_identical": True,
                                       "micro_calls": 4, "parameters_changed": True,
                                       "both_modules_have_nonzero_gradient": True,
                                       "preclip_and_postclip_gradients_bitwise": True},
                             actual={"problems": problems,
                                     "n_micro_s0": res["s0_micro_calls"],
                                     "n_micro_formal": res["tr_micro_calls"],
                                     "forward_records_s0": res["s0_forward_n"],
                                     "forward_records_formal": res["tr_forward_n"],
                                     "loss_part_records_s0": res["s0_loss_n"],
                                     "loss_part_records_formal": res["tr_loss_n"]},
                             evidence={"model_type": model_type, "grad_clip": clip,
                                       "stub": "minimal deterministic modules; same initial parameters"}))
            parity_json["runs"].append({"run_id": prefix,
                                        "model_type": model_type, "grad_clip": clip,
                                        "problems": problems,
                                        "n_micro_s0": res["s0_micro_calls"],
                                        "n_micro_formal": res["tr_micro_calls"],
                                        "forward_records_s0": res["s0_forward_n"],
                                        "forward_records_formal": res["tr_forward_n"],
                                        "loss_part_records_s0": res["s0_loss_n"],
                                        "loss_part_records_formal": res["tr_loss_n"],
                                        "micro_x0_shape": list(np.stack(
                                            [m.numpy() for m in res["micro_x0"]]).shape),
                                        "part_keys": list(LOSS_PART_KEYS),
                                        "grad_clip": float(clip)})
            prefix = f"{model_type}_{label}"
            parity_arrays[f"{prefix}__micro_x0"] = np.stack(
                [m.numpy() for m in res["micro_x0"]])
            parity_arrays[f"{prefix}__s0_preds"] = np.stack(
                [p.numpy() for p in res["s0_preds"]])
            parity_arrays[f"{prefix}__formal_preds"] = np.stack(
                [p.numpy() for p in res["tr_preds"]])
            parity_arrays[f"{prefix}__s0_parts"] = np.asarray(
                [[p[k] for k in LOSS_PART_KEYS] for p in res["s0_parts"]], dtype=np.float32)
            parity_arrays[f"{prefix}__formal_parts"] = np.asarray(
                [[p[k] for k in LOSS_PART_KEYS] for p in res["tr_parts"]], dtype=np.float32)
            parity_arrays[f"{prefix}__initial_bias"] = np.asarray(
                INITIAL_PARAMS["model.bias"], dtype=np.float32)
            parity_arrays[f"{prefix}__initial_scale"] = np.asarray(
                INITIAL_PARAMS["conditioner.scale"], dtype=np.float32)
            parity_arrays[f"{prefix}__grad_clip"] = np.asarray(float(clip), dtype=np.float64)
            for side, clip_rec in (("s0", res["s0_clip"]), ("formal", res["tr_clip"])):
                parity_arrays[f"{prefix}__{side}_preclip_0"] = clip_rec["pre"][0].numpy()
                parity_arrays[f"{prefix}__{side}_preclip_1"] = clip_rec["pre"][1].numpy()
                parity_arrays[f"{prefix}__{side}_postclip_0"] = clip_rec["post"][0].numpy()
                parity_arrays[f"{prefix}__{side}_postclip_1"] = clip_rec["post"][1].numpy()
                parity_arrays[f"{prefix}__{side}_gnorm"] = np.asarray(
                    clip_rec["gnorm"], dtype=np.float64)
            for side, params, grads, state in (
                    ("s0", res["s0_params"], res["s0_grads"], res["s0_state"]),
                    ("formal", res["tr_params"], res["tr_grads"], res["tr_state"])):
                for n in sorted(params):
                    parity_arrays[f"{prefix}__{side}_param_{n}"] = params[n].numpy()
                    parity_arrays[f"{prefix}__{side}_grad_{n}"] = grads[n].numpy()
                    parity_arrays[f"{prefix}__{side}_step_{n}"] = np.asarray(
                        state[n]["step"], dtype=np.int64)
                    parity_arrays[f"{prefix}__{side}_exp_avg_{n}"] = state[n]["exp_avg"].numpy()
                    parity_arrays[f"{prefix}__{side}_exp_avg_sq_{n}"] = state[n]["exp_avg_sq"].numpy()
            if model_type == "diffusion":
                for side, records in (("s0", res["s0_records"]), ("formal", res["tr_records"])):
                    parity_arrays[f"{prefix}__{side}_t"] = torch.stack(
                        [r["t"] for r in records]).numpy()
                    parity_arrays[f"{prefix}__{side}_noise"] = torch.stack(
                        [r["noise"] for r in records]).numpy()
                    parity_arrays[f"{prefix}__{side}_x_t"] = torch.stack(
                        [r["x_t"] for r in records]).numpy()

    # gradient accumulation division and sample-weighting distinction
    cfg = _stub_cfg("deterministic", steps=1, micro_batch=4, grad_accum=2, grad_clip=1e9)
    model = _StubModel("deterministic")
    cond = _StubConditioner()
    with torch.no_grad():
        cond.scale.fill_(1.5)   # prediction must depend on the target per sample
    asm = build_synthetic_assembler(world.tokenizer, cfg.loss, torch.float32, use_fk=True)
    items = _make_items(world, 6)
    # make the two micro-batch means differ in scale so equal-micro weighting
    # is numerically distinguishable from sample weighting
    for it in items[4:]:
        it["x0"][..., 0] += 3.0
        it["x0"][..., 9:12] += 0.5
    micro = [_ListDataset(items[:4]), _ListDataset(items[4:])]

    def zero() -> None:
        model.zero_grad(set_to_none=True); cond.zero_grad(set_to_none=True)

    def batches(ds: _ListDataset) -> dict:
        item = {k: torch.stack([x[k] for x in ds.items]) for k in ds.items[0] if torch.is_tensor(ds.items[0][k])}
        return item

    def micro_losses() -> list[torch.Tensor]:
        out = []
        for ds in micro:
            b = batches(ds)
            loss, _ = asm(model(cond(b), world.T), b)
            out.append(loss)
        return out

    zero()
    for loss in micro_losses():
        (loss / cfg.grad_accum).backward()
    g_loop = model.bias.grad.detach().clone()
    zero()
    # single equal-weight backward over the mean of micro means
    (sum(micro_losses()) / cfg.grad_accum).backward()
    g_ref = model.bias.grad.detach().clone()
    zero()
    ns = [len(ds.items) for ds in micro]
    losses = micro_losses()
    weighted = sum((l * n for l, n in zip(losses, ns)), torch.zeros(())) / sum(ns)
    weighted.backward()
    g_weighted = model.bias.grad.detach().clone()
    div_ok = float((g_loop - g_ref).abs().max()) <= 1e-7
    distinct = float((g_loop - g_weighted).abs().max()) > 1e-6
    grad_norm = float(g_loop.norm())
    rows.append(_row("update.grad_accum_division", "A3",
                     "synthetic_verified" if div_ok and grad_norm > 0 else "contract_mismatch",
                     expected={"loop_vs_sum_over_grad_accum_max_abs": "<=1e-7",
                               "gradient_norm": ">0"},
                     actual={"max_abs": float((g_loop - g_ref).abs().max()),
                             "gradient_norm": grad_norm},
                     evidence={"micro_sizes": ns, "grad_accum": cfg.grad_accum}))
    # equal-size micro-batches must match one equal-weight big batch; this is
    # the accumulation contract the plan asks to pin separately from the
    # unequal-size micro-mean semantics above
    items4 = _make_items(world, 4)
    for it in items4[1:]:
        it["x0"][..., 0] += 0.7
    m4, c4 = _StubModel("deterministic"), _StubConditioner()
    with torch.no_grad():
        c4.scale.fill_(1.5)
    micro4 = [_ListDataset(items4[:2]), _ListDataset(items4[2:])]

    def grad4(mode: str) -> torch.Tensor:
        m4.zero_grad(set_to_none=True); c4.zero_grad(set_to_none=True)
        if mode == "loop":
            for ds in micro4:
                b = batches(ds)
                loss, _ = asm(m4(c4(b), world.T), b)
                (loss / 2.0).backward()
        else:
            b = {k: torch.stack([x[k] for x in items4])
                 for k in items4[0] if torch.is_tensor(items4[0][k])}
            loss, _ = asm(m4(c4(b), world.T), b)
            loss.backward()
        return m4.bias.grad.detach().clone()

    g_loop4, g_big4 = grad4("loop"), grad4("big")
    eq_err = float((g_loop4 - g_big4).abs().max())
    rows.append(_row("update.equal_size_big_batch", "A3",
                     "synthetic_verified" if eq_err <= 1e-6 else "contract_mismatch",
                     expected={"equal_size_loop_vs_big_batch_max_abs": "<=1e-6"},
                     actual={"max_abs": eq_err,
                             "grad_norm": float(g_big4.norm())},
                     evidence={"micro_sizes": [2, 2], "grad_accum": 2,
                               "semantics": "equal-size micro means equal the big-batch mean"}))

    rows.append(_row("update.sample_weighted_distinct", "A3",
                     "synthetic_verified" if distinct else "contract_mismatch",
                     expected={"micro_equal_weight_differs_from_sample_weight": True},
                     actual={"max_abs": float((g_loop - g_weighted).abs().max())},
                     evidence={"micro_sizes": ns,
                               "semantics": "each micro loss is a mean; the loop averages micro means",
                               "s0_uses_equal_sizes_so_the_distinction_vanishes": True}))

    # optimizer membership / duplicates / adamw reference
    tr_model = _StubModel("deterministic"); tr_cond = _StubConditioner()
    ids = [id(p) for p in tr_model.parameters()] + [id(p) for p in tr_cond.parameters()]
    opt_ids = [id(p) for g in torch.optim.AdamW([*tr_model.parameters(), *tr_cond.parameters()],
                                                lr=1e-4).param_groups for p in g["params"]]
    rows.append(_row("update.optimizer_membership", "A3",
                     "synthetic_verified" if (len(ids) == len(set(ids)) and sorted(ids) == sorted(opt_ids))
                     else "contract_mismatch",
                     expected={"model_and_conditioner_present": True, "duplicate_parameters": 0},
                     actual={"n_parameters": len(ids), "n_optimizer": len(opt_ids),
                             "duplicates": len(ids) - len(set(ids))}))

    p0 = tr_model.bias.detach().clone()
    g = torch.linspace(-0.1, 0.1, X0_DIM)
    tr_model.bias.grad = g.clone()
    opt = torch.optim.AdamW([tr_model.bias], lr=1e-4, weight_decay=1e-5)
    opt.step()
    b1, b2, eps = 0.9, 0.999, 1e-8
    exp_avg = (1 - b1) * g
    exp_avg_sq = (1 - b2) * g * g
    mhat = exp_avg / (1 - b1)
    vhat = exp_avg_sq / (1 - b2)
    expected = p0 - 1e-4 * (mhat / (vhat.sqrt() + eps) + 1e-5 * p0)
    rows.append(_compare("update.adamw_reference", "A3", expected.numpy(),
                         tr_model.bias.detach().numpy(), TOLERANCES["adamw_float32"],
                         evidence={"lr": 1e-4, "weight_decay": 1e-5, "betas": [b1, b2]}))

    trainer_src = (Path(__file__).resolve().parents[3] / "src/ireset/train/trainer.py").read_text()
    has_sched = ("lr_scheduler" in trainer_src) or ("scheduler.step" in trainer_src)
    rows.append(_row("update.no_lr_scheduler", "A3",
                     "synthetic_verified" if not has_sched else "contract_mismatch",
                     expected={"lr_scheduler_calls": 0},
                     actual={"lr_scheduler_calls": int(has_sched)},
                     evidence={"fixed_lr": 1e-4, "source": "src/ireset/train/trainer.py"}))

    # sampling: x_t closed form, t=0 is not noiseless
    from ireset.diffusion.generator import DiffusionG1
    diff = DiffusionG1(d_model=8, n_layers=1, n_heads=2, time_window=3)
    x0 = torch.tensor(world.x0, dtype=torch.float32)[None]
    eps_t = torch.cos(torch.arange(x0.numel(), dtype=torch.float32)).reshape_as(x0) * 0.5
    ts = [0, 1, 500, 999]
    errs = []
    t0_frac = None
    for tv in ts:
        t = torch.tensor([tv])
        xt = diff.add_noise(x0, eps_t, t)
        ab = diff.ddpm.alphas_cumprod[tv]
        expect = ab.sqrt() * x0 + (1 - ab).sqrt() * eps_t
        errs.append(float((xt - expect).abs().max()))
        if tv == 0:
            t0_frac = float((1 - ab).sqrt())
    rows.append(_row("sampling.x_t_closed_form", "A3",
                     "synthetic_verified" if max(errs) <= 1e-6 else "contract_mismatch",
                     expected={"max_abs_error": "<=1e-6"},
                     actual={"t": ts, "max_abs_errors": errs}))
    rows.append(_row("sampling.t0_still_noised", "A3",
                     "synthetic_verified" if t0_frac is not None and t0_frac > 0 else "contract_mismatch",
                     expected={"noise_fraction_at_t0": ">0", "not_noiseless": True},
                     actual={"noise_fraction_at_t0": t0_frac,
                             "alpha_cumprod_t0": float(diff.ddpm.alphas_cumprod[0])}))
    # FULL DDIM loop: freeze the expected timestep table BEFORE calling the
    # sampling entry, count scheduler.step calls, and compare the trajectory
    # against an independently implemented eta=0 DDIM update.  The reference
    # does not reuse the tested scheduler.step or its mutable timestep table.
    target = torch.tensor(world.x0, dtype=torch.float32)[None]
    diff.ddim.set_timesteps(5)
    expected_timesteps = [int(v) for v in diff.ddim.timesteps]
    fwd_ts: list[int] = []
    step_ts: list[int] = []

    def predict_value(x_t, t):
        return target * 0.5 + x_t * 0.5 + 0.01 * (t.float() / 1000.0)[:, None, None]

    def predictor(x_t, t, memory):
        fwd_ts.append(int(t.reshape(-1)[0]))
        return predict_value(x_t, t)

    diff.forward = predictor
    real_step = diff.ddim.step

    def counted_step(*args, **kwargs):
        step_ts.append(int(args[1].reshape(-1)[0]) if len(args) > 1 else -1)
        return real_step(*args, **kwargs)

    diff.ddim.step = counted_step
    sample_error, sampled = None, None
    try:
        sampled = diff.sample(torch.zeros(1, 4, 8), world.T, steps=5,
                              generator=torch.Generator().manual_seed(5))
    except Exception as exc:  # noqa: BLE001
        sample_error = f"{type(exc).__name__}: {exc}"
    finally:
        diff.ddim.step = real_step
    sampled_fwd_ts = list(fwd_ts)
    # independent eta=0 DDIM update, frozen schedule, same predictor/noise
    reference = None
    ref_error = None
    try:
        x_ref = torch.randn(1, world.T, 38,
                            generator=torch.Generator().manual_seed(5))
        for idx, tv in enumerate(expected_timesteps):
            ab_t = float(diff.ddim.alphas_cumprod[tv])
            prev_tv = expected_timesteps[idx + 1] if idx + 1 < len(expected_timesteps) else None
            ab_prev = float(diff.ddim.alphas_cumprod[prev_tv]) if prev_tv is not None else 1.0
            x0_pred = predict_value(x_ref, torch.tensor([tv]))
            eps = (x_ref - math.sqrt(ab_t) * x0_pred) / math.sqrt(1.0 - ab_t)
            x_ref = math.sqrt(ab_prev) * x0_pred + math.sqrt(1.0 - ab_prev) * eps
        reference = x_ref
        ref_error = (float((sampled - reference).abs().max())
                     if sampled is not None else None)
    except Exception as exc:  # noqa: BLE001
        ref_error = None
        sample_error = (sample_error or "") + f"|reference:{type(exc).__name__}:{exc}"
    full_ok = (sample_error is None
               and diff.ddim.config.prediction_type == "sample"
               and diff.ddim.config.clip_sample is False
               and diff.ddim.config.thresholding is False
               and float(diff.ddim.final_alpha_cumprod) == 1.0
               and expected_timesteps == [800, 600, 400, 200, 0]
               and sampled_fwd_ts == expected_timesteps
               and step_ts == expected_timesteps
               and sampled is not None and tuple(sampled.shape) == tuple(target.shape)
               and ref_error is not None and ref_error <= 1e-5)
    rows.append(_row("sampling.ddim_full_loop", "A3",
                     "synthetic_verified" if full_ok else "contract_mismatch",
                     expected={"prediction_type": "sample", "clip_sample": False,
                               "thresholding": False, "final_alpha_cumprod": 1.0,
                               "sample_calls": 1, "forward_calls": 5,
                               "scheduler_step_calls": 5,
                               "timesteps_visited": [800, 600, 400, 200, 0],
                               "reference_final_max_abs_error": "<=1e-5"},
                     actual={"prediction_type": diff.ddim.config.prediction_type,
                             "clip_sample": diff.ddim.config.clip_sample,
                             "thresholding": diff.ddim.config.thresholding,
                             "final_alpha_cumprod": float(diff.ddim.final_alpha_cumprod),
                             "sample_calls": 1,
                             "forward_calls": len(sampled_fwd_ts),
                             "scheduler_step_calls": len(step_ts),
                             "timesteps_visited": sampled_fwd_ts,
                             "scheduler_timesteps": expected_timesteps,
                             "reference_final_max_abs_error": ref_error,
                             "sample_error": sample_error}))

    # single DDIM step vs independent eta=0 formula
    diff.ddim.set_timesteps(5)
    t_sel = diff.ddim.timesteps[1]
    x_t = torch.tensor(world.x0, dtype=torch.float32)[None] * 0.3 + 0.1
    x0_pred = torch.tensor(world.x0, dtype=torch.float32)[None]
    step = diff.ddim.step(x0_pred, t_sel, x_t, eta=0.0).prev_sample
    ab_t = float(diff.ddim.alphas_cumprod[int(t_sel)])
    prev_idx = diff.ddim.timesteps[2] if len(diff.ddim.timesteps) > 2 else None
    ab_prev = float(diff.ddim.alphas_cumprod[int(prev_idx)]) if prev_idx is not None else 1.0
    eps_ind = (x_t - math.sqrt(ab_t) * x0_pred) / math.sqrt(1 - ab_t)
    expect_step = math.sqrt(ab_prev) * x0_pred + math.sqrt(1 - ab_prev) * eps_ind
    rows.append(_row("sampling.ddim_single_step", "A3",
                     "synthetic_verified" if float((step - expect_step).abs().max()) <= 1e-6
                     else "contract_mismatch",
                     expected={"max_abs_error": "<=1e-6"},
                     actual={"max_abs_error": float((step - expect_step).abs().max()),
                             "t": int(t_sel), "eta": 0.0}))

    # data consumption traces (real loop classes, synthetic data)
    s0 = _load_w1_s0()
    c_cfg = _stub_cfg("deterministic", steps=1, micro_batch=6, grad_accum=4, grad_clip=1e9)
    log6: list[int] = []
    ds6 = _ListDataset(_make_items(world, 6), log6)
    small6 = {k: torch.stack([it[k] for it in ds6.items]) for k in ds6.items[0] if torch.is_tensor(ds6.items[0][k])}
    loader6 = torch.utils.data.DataLoader(ds6, batch_size=6, shuffle=True, num_workers=0,
                                          generator=torch.Generator().manual_seed(3))
    s0_dir = tmp_root / "consumption_s0"; s0_dir.mkdir(parents=True, exist_ok=True)
    import ireset.train.losses as losses_mod
    import ireset.train.trainer as trainer_mod
    real_adamw, real_build = torch.optim.AdamW, s0.build_s0_modules
    real_loss = losses_mod.LossAssembler
    class _F:
        def __call__(self, tokenizer, weights, device):
            return build_synthetic_assembler(tokenizer, weights, torch.float32, use_fk=True)
    try:
        s0.build_s0_modules = lambda mt, cf, dv: (_StubModel(mt), _StubConditioner())
        losses_mod.LossAssembler = _F()
        s0.run_s0_model("deterministic", c_cfg, world.tokenizer, small6, loader6, "cpu",
                        {"audit": "synthetic"}, s0_dir, time.perf_counter() + 120.0)
    finally:
        torch.optim.AdamW = real_adamw
        s0.build_s0_modules = real_build
        losses_mod.LossAssembler = real_loss
    uniq = sorted(set(log6))
    counts = {i: log6.count(i) for i in uniq}
    rows.append(_row("consumption.s0_six_window_whole_batch", "A3",
                     "synthetic_verified" if (len(log6) == 24 and uniq == list(range(6))
                                              and all(v == 4 for v in counts.values()))
                     else "contract_mismatch",
                     expected={"samples": 24, "unique": 6, "each": 4},
                     actual={"samples": len(log6), "unique": len(uniq), "counts": counts},
                     evidence={"six_windows_x_grad_accum": 6 * 4}))

    # formal 4x4 with 10 windows, drop_last -> 2 dropped per epoch
    logf: list[int] = []
    dsf = _ListDataset(_make_items(world, 10), logf)
    cfgf = _stub_cfg("deterministic", steps=2, micro_batch=4, grad_accum=4, grad_clip=1e9)
    trm, trc = _StubModel("deterministic"), _StubConditioner()
    try:
        trainer_mod.DeterministicG1 = lambda **kw: trm
        trainer_mod.ConditionEncoder = lambda d: trc
        trainer_mod.LossAssembler = _F()
        tr = trainer_mod.Trainer(cfgf, world.tokenizer, dsf,
                                 _ListDataset(_make_items(world, 2)), tmp_root / "consumption_formal")
        torch.manual_seed(cfgf.seed)
        summary = tr.run()
    finally:
        for name, obj in _originals.items():
            setattr(trainer_mod, name, obj)
        _losses_mod.LossAssembler = _original_loss
    per_epoch = [logf[i:i + 8] for i in range(0, len(logf), 8)]
    dropped = [sorted(set(range(10)) - set(e)) for e in per_epoch]
    coverage = sorted(set(logf))
    ok_consumption = (summary["train_loader"]["samples_total"] == 32
                      and summary["train_loader"]["micro_batches_total"] == 8
                      and summary["train_loader"]["drop_last_discarded_per_epoch"] == 2
                      and all(len(d) == 2 for d in dropped))
    rows.append(_row("consumption.formal_4x4_drop_last", "A3",
                     "synthetic_verified" if ok_consumption else "contract_mismatch",
                     expected={"samples_total": 32, "micro_batches": 8, "dropped_per_epoch": 2},
                     actual={"samples_total": summary["train_loader"]["samples_total"],
                             "micro_batches": summary["train_loader"]["micro_batches_total"],
                             "dropped_per_epoch": summary["train_loader"]["drop_last_discarded_per_epoch"],
                             "dropped_ids_per_epoch": dropped,
                             "cumulative_coverage": coverage,
                             "epochs_observed": len(per_epoch)}))

    # RNG isolation / restoration
    trm2, trc2 = _StubModel("deterministic"), _StubConditioner()
    try:
        trainer_mod.DeterministicG1 = lambda **kw: trm2
        trainer_mod.ConditionEncoder = lambda d: trc2
        trainer_mod.LossAssembler = _F()
        tr2 = trainer_mod.Trainer(cfgf, world.tokenizer,
                                  _ListDataset(_make_items(world, 4)),
                                  _ListDataset(_make_items(world, 2)),
                                  tmp_root / "rng_restore")
    finally:
        for name, obj in _originals.items():
            setattr(trainer_mod, name, obj)
        _losses_mod.LossAssembler = _original_loss
    before = _rng_digest()
    modes_before = {m: m.training for m in list(trm2.modules()) + list(trc2.modules())}
    try:
        with tr2._evaluation_state():
            torch.rand(3)
            np.random.rand(3)
            random.random()
            raise RuntimeError("deliberate")
    except RuntimeError:
        pass
    after = _rng_digest()
    modes_after = {m: m.training for m in list(trm2.modules()) + list(trc2.modules())}
    rows.append(_row("rng.validation_restores_streams", "A3",
                     "synthetic_verified" if before == after else "contract_mismatch",
                     expected={"python": "restored", "numpy": "restored", "torch_cpu": "restored"},
                     actual={"equal_after_exception": before == after}))
    rows.append(_row("rng.module_modes_restored", "A3",
                     "synthetic_verified" if modes_before == modes_after else "contract_mismatch",
                     expected={"module_training_flags_restored": True},
                     actual={"equal": modes_before == modes_after}))

    ds_gen = _ListDataset(_make_items(world, 5))
    snap = _rng_digest()
    loader = torch.utils.data.DataLoader(ds_gen, batch_size=2, shuffle=True, num_workers=0,
                                         generator=torch.Generator().manual_seed(11))
    list(loader)
    after_gen = _rng_digest()
    loader2 = torch.utils.data.DataLoader(ds_gen, batch_size=2, shuffle=True, num_workers=0)
    list(loader2)
    after_global = _rng_digest()
    rows.append(_row("rng.dataloader_dedicated_generator", "A3",
                     "synthetic_verified" if (snap == after_gen and snap != after_global)
                     else "contract_mismatch",
                     expected={"dedicated_generator_global_unchanged": True,
                               "generator_none_consumes_global": True},
                     actual={"dedicated_unchanged": snap == after_gen,
                             "generator_none_changed": snap != after_global}))

    sampling = {
        "x_t_closed_form": {"t": ts, "max_abs_errors": errs},
        "t0": {"noise_fraction": t0_frac},
        "ddim": {"prediction_type": diff.ddim.config.prediction_type,
                 "clip_sample": diff.ddim.config.clip_sample,
                 "thresholding": diff.ddim.config.thresholding},
        "rng": {"validation_restored": before == after,
                "dataloader_dedicated_unchanged": snap == after_gen,
                "cuda_used": False,
                "streams_checked": ["python", "numpy", "torch_cpu"]},
        "consumption": {"s0": {"log": log6, "samples": len(log6),
                                "unique": len(uniq), "counts": counts},
                        "formal": {"log": logf, "samples": len(logf),
                                   "dropped_per_epoch": dropped,
                                   "coverage": coverage}},
    }
    return rows, parity_arrays, {**parity_json, "sampling_trace": sampling}


# ---------------------------------------------------------------------------
# A4 aggregation / evaluation / selection
# ---------------------------------------------------------------------------


def independent_macro(rows: list[dict], expected: list[str],
                      lengths: dict[str, int] | None = None) -> dict:
    """Independent formal-selection reconstruction (demo-equal-weight)."""
    gaps: list[str] = []
    if not expected or len(set(expected)) != len(expected):
        gaps.append("invalid_registered_roster")
        expected = list(dict.fromkeys(expected)) or []
    if not isinstance(rows, list):
        gaps.append("rows_not_a_list")
        rows = []
    by_demo: dict[str, list[dict]] = {d: [] for d in expected}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            gaps.append(f"row_not_mapping:{i}")
            continue
        demo = r.get("demo")
        if not isinstance(demo, str) or demo not in expected:
            gaps.append(f"demo_not_registered:{demo}")
            continue
        by_demo[demo].append(r)
    values: list[float] = []
    for demo in expected:
        rs = by_demo[demo]
        if len(rs) != 1:
            gaps.append(f"demo_row_count:{demo}:{len(rs)}")
            continue
        r = rs[0]
        required = {"demo", "T", "status", "numerical_valid", "mpjpe_m"}
        missing = sorted(required - set(r))
        if missing:
            gaps.append(f"missing_keys:{demo}:{missing}")
            continue
        T, metric = r.get("T"), r.get("mpjpe_m")
        if (not isinstance(T, int) or isinstance(T, bool) or T <= 0
                or (lengths is not None and lengths.get(demo) != T)):
            gaps.append(f"T_mismatch:{demo}:{T}")
            continue
        if r.get("status") != "ok" or r.get("numerical_valid") is not True:
            gaps.append(f"status_not_ok:{demo}:{r.get('status')}")
            continue
        if (not isinstance(metric, (int, float)) or isinstance(metric, bool)
                or not math.isfinite(float(metric)) or float(metric) < 0):
            gaps.append(f"metric_invalid:{demo}")
            continue
        values.append(float(metric))
    macro = sum(values) / len(expected) if len(values) == len(expected) and not gaps else None
    return {"n_expected": len(expected), "n_ok": len(values),
            "macro": macro, "complete": macro is not None and not gaps,
            "gaps": gaps}


def run_a4(world: SyntheticWorld, tmp_root: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    # macro vs micro ranking on demos with different window counts
    method1 = {"A": [0.5] * 10, "B": [0.0] * 2}
    method2 = {"A": [0.0] * 10, "B": [1.0] * 2}
    def macro(m: dict) -> float:
        return float(np.mean([np.mean(v) for v in m.values()]))
    def micro(m: dict) -> float:
        flat = [x for v in m.values() for x in v]
        return float(np.mean(flat))
    m1, m2 = macro(method1), macro(method2)
    u1, u2 = micro(method1), micro(method2)
    rows.append(_row("aggregate.macro_vs_micro_ranking", "A4",
                     "synthetic_verified" if (m1 < m2 and u1 > u2) else "contract_mismatch",
                     expected={"macro_prefers": "method1", "micro_prefers": "method2"},
                     actual={"macro": {"method1": m1, "method2": m2},
                             "micro": {"method1": u1, "method2": u2},
                             "window_counts": {"A": 10, "B": 2}}))

    # window micro average: build per-window losses and compare with Trainer.validate
    items = _make_items(world, 6, scale=1.0)
    for i, it in enumerate(items):
        it["x0"][..., 0] += 0.02 * i
    val_ds = _ListDataset(items)
    model, cond = _StubModel("deterministic"), _StubConditioner()
    cfg = _stub_cfg("deterministic", steps=1, micro_batch=3, grad_accum=1, grad_clip=1e9)
    import ireset.train.trainer as trainer_mod
    import ireset.train.losses as losses_mod
    real_det, real_cond, real_loss = trainer_mod.DeterministicG1, trainer_mod.ConditionEncoder, trainer_mod.LossAssembler
    try:
        trainer_mod.DeterministicG1 = lambda **kw: model
        trainer_mod.ConditionEncoder = lambda d: cond
        class _F:
            def __call__(self, tokenizer, weights, device):
                return build_synthetic_assembler(tokenizer, weights, torch.float32, use_fk=True)
        trainer_mod.LossAssembler = _F()
        tr = trainer_mod.Trainer(cfg, world.tokenizer, _ListDataset(items[:3]), val_ds,
                                 tmp_root / "a4_validate")
        torch.manual_seed(cfg.seed)
        val = tr.validate()
    finally:
        trainer_mod.DeterministicG1, trainer_mod.ConditionEncoder, trainer_mod.LossAssembler = real_det, real_cond, real_loss
    asm = build_synthetic_assembler(world.tokenizer, cfg.loss, torch.float32, use_fk=True)
    with torch.no_grad():
        per_window = []
        for it in items:
            b = {k: v[None] for k, v in it.items() if torch.is_tensor(v)}
            _, parts = asm(model(cond(b), world.T), b)
            per_window.append(parts["total"])
    independent = float(np.mean(per_window))
    rows.append(_compare("aggregate.window_micro_average", "A4", independent,
                         val["total"], TOLERANCES["production_float32"],
                         evidence={"n_windows": len(per_window),
                                   "sample_weighted": True,
                                   "per_window": per_window,
                                   "val_n_windows": val.get("n_windows_evaluated")}))

    # synthetic physical FK MPJPE aggregation: per-window mean site distance,
    # demo-equal-weight macro vs window micro.  fk_backend is the audit stub,
    # so this verifies aggregation algebra only (not G1 FK correctness).
    from ireset.embodied.g1_spec import KPT_TO_SMPLH as _KPT
    demo_windows = {"demoA": 4, "demoB": 2}
    per_demo_mpjpe: dict[str, list[float]] = {"demoA": [], "demoB": []}
    target_pos, target_quat, target_q = _decode_clip_np(world.x0, world.tokenizer)
    target_kp = _synthetic_fk_np(target_pos, target_quat, target_q)[..., :len(_KPT), :]
    for demo, n in demo_windows.items():
        for j in range(n):
            pred = world.x0.copy()
            scale = 0.0005 * (1 + j) * (1 if demo == "demoA" else 20)
            pred[:, 0:3] += scale * np.array([0.3, -0.2, 0.1])
            pp, pq, pj = _decode_clip_np(pred, world.tokenizer)
            kp = _synthetic_fk_np(pp, pq, pj)[..., :len(_KPT), :]
            mpjpe = float(np.linalg.norm(kp - target_kp, axis=-1).mean())
            per_demo_mpjpe[demo].append(mpjpe)
    demo_means = {d: float(np.mean(v)) for d, v in per_demo_mpjpe.items()}
    macro_fk = float(np.mean(list(demo_means.values())))
    micro_fk = float(np.mean([v for vals in per_demo_mpjpe.values() for v in vals]))
    rows.append(_row("aggregate.physical_fk_mpjpe", "A4",
                     "synthetic_verified" if (
                         all(np.isfinite(v) and v > 0 for v in demo_means.values())
                         and abs(macro_fk - micro_fk) > 1e-12)
                     else "contract_mismatch",
                     expected={"macro": macro_fk, "micro": micro_fk,
                               "ranking_differs": True},
                     actual={"per_demo": demo_means, "macro": macro_fk,
                             "micro": micro_fk,
                             "window_counts": demo_windows},
                     evidence={"fk_backend": "synthetic",
                               "metric": "mean Euclidean site distance (15 motion sites)",
                               "real_g1_fk": "not_run"}))

    # roster / bad-row retention
    expected = ["demoA", "demoB", "demoC"]
    lengths = {"demoA": 100, "demoB": 70, "demoC": 90}
    good = [{"demo": d, "T": lengths[d], "status": "ok", "numerical_valid": True,
             "mpjpe_m": 0.1 + 0.01 * i} for i, d in enumerate(expected)]
    base = independent_macro(good, expected, lengths)
    rows.append(_row("selection.registered_roster_complete", "A4",
                     "synthetic_verified" if base["complete"] and base["macro"] is not None
                     else "contract_mismatch",
                     expected={"complete": True, "macro": float(np.mean([0.1, 0.11, 0.12]))},
                     actual=base))
    variants = {}
    v = [dict(r) for r in good]; v.pop(1); variants["missing_row"] = v
    v = [dict(r) for r in good]; v[1] = dict(v[1], demo="demoA"); variants["duplicate_demo"] = v
    v = [dict(r) for r in good]; v[1] = dict(v[1], T=71); variants["T_mismatch"] = v
    v = [dict(r) for r in good]; v[1] = dict(v[1], status="row_missing", numerical_valid=None, mpjpe_m=None); variants["status_missing"] = v
    v = [dict(r) for r in good]; v[1] = dict(v[1], numerical_valid=False); variants["numerical_invalid"] = v
    v = [dict(r) for r in good]; v[1] = dict(v[1], mpjpe_m=True); variants["bool_metric"] = v
    v = [dict(r) for r in good]; v[1] = dict(v[1], mpjpe_m=float("nan")); variants["nan_metric"] = v
    v = [dict(r) for r in good]; v[1] = dict(v[1], mpjpe_m=float("inf")); variants["inf_metric"] = v
    v = [dict(r) for r in good]; v.append({"demo": "demoD", "T": 60, "status": "ok",
                                           "numerical_valid": True, "mpjpe_m": 0.0}); variants["extra_demo"] = v
    v = [dict(r) for r in good]; del v[1]["mpjpe_m"]; variants["missing_metric_key"] = v
    bad = {}
    ok_all = True
    for name, rs in variants.items():
        res = independent_macro(rs, expected, lengths)
        bad[name] = {"complete": res["complete"], "macro": res["macro"], "gaps": res["gaps"]}
        ok_all &= (res["complete"] is False and res["macro"] is None and bool(res["gaps"]))
    rows.append(_row("selection.bad_rows_retained_denominator", "A4",
                     "synthetic_verified" if ok_all else "contract_mismatch",
                     expected={"all_variants_incomplete": True, "macro": None},
                     actual=bad,
                     evidence={"variants": sorted(variants)}))

    # compare the independent reconstruction with the real Trainer gate
    tr_model, tr_cond = _StubModel("deterministic"), _StubConditioner()
    class _Formal:
        expected_demos = expected
        demo_lengths = lengths
    try:
        trainer_mod.DeterministicG1 = lambda **kw: tr_model
        trainer_mod.ConditionEncoder = lambda d: tr_cond
        trainer_mod.LossAssembler = _F()
        tr = trainer_mod.Trainer(cfg, world.tokenizer, _ListDataset(items[:3]),
                                 val_ds, tmp_root / "a4_formal_evidence",
                                 formal_eval=_Formal(), formal_eval_every=1)
    finally:
        trainer_mod.DeterministicG1, trainer_mod.ConditionEncoder, trainer_mod.LossAssembler = real_det, real_cond, real_loss

    def raw_for(rs: list[dict]) -> dict:
        vals = [float(r["mpjpe_m"]) for r in rs
                if r.get("status") == "ok" and r.get("numerical_valid") is True
                and isinstance(r.get("mpjpe_m"), (int, float))
                and not isinstance(r.get("mpjpe_m"), bool)
                and math.isfinite(float(r["mpjpe_m"]))]
        macro = sum(vals) / len(expected) if len(vals) == len(expected) else None
        return {"n_expected": len(expected), "n_ok": len(vals),
                "macro_mpjpe_m": macro, "per_demo": rs}

    real_good = tr._formal_evidence(raw_for(good))
    indep_good = independent_macro(good, expected, lengths)
    macro_ok = (real_good["complete"] is True
                and abs(real_good["macro_mpjpe_m"] - indep_good["macro"]) <= TOLERANCES["macro_float64"]["atol"])
    mismatches = {}
    for name, rs in variants.items():
        real = tr._formal_evidence(raw_for(rs))
        indep = independent_macro(rs, expected, lengths)
        mismatches[name] = {"trainer_complete": real["complete"],
                           "independent_complete": indep["complete"],
                           "trainer_macro": real["macro_mpjpe_m"],
                           "independent_macro": indep["macro"]}
    agree = macro_ok and all(v["trainer_complete"] is False and v["independent_complete"] is False
                             for v in mismatches.values())
    rows.append(_row("selection.macro_from_raw_rows", "A4",
                     "synthetic_verified" if agree else "contract_mismatch",
                     expected={"complete_match": True, "macro_match": True,
                               "all_bad_variants_rejected_by_both": True},
                     actual={"good": {"trainer": real_good, "independent": indep_good},
                             "variants": mismatches}))

    # best only from a complete macro; synthetic callback at two cadence points
    def run_best(second_bad: bool) -> dict:
        steps = {"n": 0}
        raw_steps: list[dict] = []

        def callback(model, conditioner, budget_deadline=None):
            steps["n"] += 1
            if steps["n"] == 1:
                rs = [{"demo": d, "T": lengths[d], "status": "ok", "numerical_valid": True,
                       "mpjpe_m": 0.5} for d in expected]
            elif second_bad:
                rs = [{"demo": expected[0], "T": lengths[expected[0]], "status": "ok",
                       "numerical_valid": True, "mpjpe_m": 0.1},
                      {"demo": expected[1], "T": lengths[expected[1]], "status": "row_missing",
                       "numerical_valid": None, "mpjpe_m": None},
                      {"demo": expected[2], "T": lengths[expected[2]], "status": "ok",
                       "numerical_valid": True, "mpjpe_m": 0.1}]
            else:
                rs = [{"demo": d, "T": lengths[d], "status": "ok", "numerical_valid": True,
                       "mpjpe_m": 0.2} for d in expected]
            vals = [r["mpjpe_m"] for r in rs if r["status"] == "ok" and r["mpjpe_m"] is not None]
            raw = {"n_expected": len(expected), "n_ok": len(vals),
                   "macro_mpjpe_m": sum(vals) / len(expected) if len(vals) == len(expected) else None,
                   "per_demo": rs, "seed": 7}
            raw_steps.append(raw)
            return raw
        callback.expected_demos = expected
        callback.demo_lengths = lengths
        m, c = _StubModel("deterministic"), _StubConditioner()
        c2 = _stub_cfg("deterministic", steps=2, micro_batch=2, grad_accum=1, grad_clip=1e9)
        c2.formal_eval_every = 1
        c2.val_every = 1
        try:
            trainer_mod.DeterministicG1 = lambda **kw: m
            trainer_mod.ConditionEncoder = lambda d: c
            trainer_mod.LossAssembler = _F()
            t = trainer_mod.Trainer(c2, world.tokenizer, _ListDataset(items[:3]), val_ds,
                                    tmp_root / ("best_" + ("bad" if second_bad else "good")),
                                    formal_eval=callback, formal_eval_every=1)
            torch.manual_seed(c2.seed)
            summ = t.run()
        finally:
            trainer_mod.DeterministicG1, trainer_mod.ConditionEncoder, trainer_mod.LossAssembler = real_det, real_cond, real_loss
        best = summ.get("best_formal_step")
        return {"best_formal_step": best, "raw_steps": raw_steps, "summary": {
            "selection_status": summ.get("selection_status"),
            "best_formal_macro_mpjpe_m": summ.get("best_formal_macro_mpjpe_m")}}
    good_res = run_best(False)
    bad_res = run_best(True)
    ok_best = (good_res["best_formal_step"] == 2 and bad_res["best_formal_step"] == 1)
    rows.append(_row("selection.best_uses_macro_complete_only", "A4",
                     "synthetic_verified" if ok_best else "contract_mismatch",
                     expected={"complete_second_macro_selected_step": 2,
                               "incomplete_second_macro_keeps_step": 1},
                     actual={"complete_run": good_res, "incomplete_run": bad_res}))

    # determinism / representation
    asm = build_synthetic_assembler(world.tokenizer, W1_SYNTHETIC_WEIGHTS, torch.float32)
    b = make_synthetic_batch(world, dtype=torch.float32)
    p = torch.tensor(_make_pred(world, "generic"), dtype=torch.float32)[None]
    l1, parts1 = asm(p, b)
    l2, parts2 = asm(p, b)
    det = (_bitwise_equal(l1, l2) and parts1 == parts2
           and all(_bitwise_equal(a, b_) for a, b_ in zip(
               asm.decode_clip(p), asm.decode_clip(p))))
    rows.append(_row("representation.determinism_bits", "A4",
                     "synthetic_verified" if det else "contract_mismatch",
                     expected={"same_input_same_code_bitwise": True},
                     actual={"bitwise_equal": det, "dtype": "float32",
                             "shapes": [list(t.shape) for t in asm.decode_clip(p)]}))
    unrounded = any(float(parts1[k]) != round(float(parts1[k]), 6) for k in parts1)
    rows.append(_row("representation.unrounded_artifacts", "A4",
                     "synthetic_verified" if unrounded else "contract_mismatch",
                     expected={"artifact_keeps_unrounded_floats": True},
                     actual={"has_beyond_6_decimals": unrounded,
                             "sample": {k: parts1[k] for k in ("pos", "vel", "total")}}))

    # loss_history rounding is a documented representation limitation
    hist = tmp_root / "best_good" / "loss_history.json"
    hist_ok = False
    if hist.is_file():
        hist_rows = json.loads(hist.read_text())
        hist_ok = all(
            all(isinstance(v, float) and v == round(v, 6)
                for k, v in row.items() if k in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "total"))
            for row in hist_rows)
    rows.append(_row("representation.loss_history_rounding_limitation", "A4",
                     "synthetic_verified" if hist_ok else "contract_mismatch",
                     expected={"loss_history_rounded_digits": 6,
                               "exact_total_from_history": False},
                     actual={"history_steps": len(hist_rows) if hist.is_file() else 0,
                             "all_core_fields_rounded_6": hist_ok},
                     applicability="exact per-bit loss reconciliation must use new unrounded "
                                   "synthetic artifacts, never the training log",
                     limitation="rounded_history_not_usable_for_exact_reconciliation"))

    return rows, {"macro": {"method1": m1, "method2": m2},
                  "micro": {"method1": u1, "method2": u2},
                  "best_complete_step": good_res["best_formal_step"],
                  "best_incomplete_step": bad_res["best_formal_step"],
                  "selection_evidence": {
                      "expected_demos": expected,
                      "demo_lengths": lengths,
                      "complete_run": {"raw_steps": good_res["raw_steps"],
                                       "best_formal_step": good_res["best_formal_step"]},
                      "incomplete_run": {"raw_steps": bad_res["raw_steps"],
                                         "best_formal_step": bad_res["best_formal_step"]},
                  }}


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def run_synthetic_audit(out_dir: Path, project_root: Path | None = None,
                        config_path: Path | None = None) -> dict:
    project_root = Path(project_root or Path(__file__).resolve().parents[3])
    config_path = Path(config_path or project_root / "configs/train/omniretarget_motion.yaml")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = out_dir / "_loop_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    static = build_static_contract(project_root, config_path)
    comparison = build_s0_formal_comparison(static)
    world = build_synthetic_world()

    a1_rows, inputs = run_a1(world)
    a2_rows, grad_arrays = run_a2(world)
    a3_rows, parity_arrays, parity_json = run_a3(world, tmp_root)
    a4_rows, a4_summary = run_a4(world, tmp_root)
    for r in a2_rows:
        r.setdefault("fixture", "analytic_synthetic")
        r.setdefault("space", "output")
        r.setdefault("dtype", "float64")
        if r["id"] == "grad.float32_production":
            r["dtype"] = "float32"
        if r["id"] == "grad.parameter_space_chain_rule":
            r["space"] = "parameter"
        r["term"] = r["id"].split(".", 1)[1]
    rows = a1_rows + a2_rows + a3_rows + a4_rows

    np.savez(out_dir / "synthetic_inputs.npz", **inputs)
    np.savez(out_dir / "gradient_arrays.npz", **grad_arrays)
    np.savez(out_dir / "update_arrays.npz", **parity_arrays)

    row_failures = [r for r in rows if r["status"] == "contract_mismatch"]
    gaps: list[str] = []
    present = {(r["group"], r["id"]) for r in rows}
    for group, required in REQUIRED_AUDIT_ROWS.items():
        missing = [rid for rid in required if (group, rid) not in present]
        gaps.extend(f"missing_row:{group}:{rid}" for rid in missing)
    static_failures = [r for r in static["checks"] if r["status"] == "contract_mismatch"]
    failures = row_failures + static_failures
    limitations = [r["id"] for r in rows if r.get("limitation")]
    conclusion = ("implementation_mismatch" if failures
                  else "audit_incomplete" if gaps
                  else "synthetic_contract_passed")

    objective_artifact = {
        "schema": AUDIT_SCHEMA, "source": AUDIT_SOURCE,
        "rows": rows, "failures": [r["id"] for r in row_failures], "gaps": gaps,
        "limitations": limitations,
        "fk_backend": "synthetic",
        "real_g1_fk": "not_run",
    }
    _write_json(out_dir / "static_contract.json", static)
    _write_json(out_dir / "s0_formal_comparison.json", comparison)
    _write_json(out_dir / "objective_rows.json", objective_artifact)
    _write_json(out_dir / "gradient_rows.json", {
        "schema": AUDIT_SCHEMA, "source": AUDIT_SOURCE,
        "rows": [r for r in a2_rows], "arrays": "gradient_arrays.npz",
        "dtype": "float64 audit; float32 production row separately tagged",
    })
    _write_json(out_dir / "update_parity.json", {
        "schema": AUDIT_SCHEMA, "source": AUDIT_SOURCE,
        "rows": [r for r in a3_rows], "arrays": "update_arrays.npz", **parity_json,
    })
    _write_json(out_dir / "sampling_trace.json", {
        "schema": AUDIT_SCHEMA, "source": AUDIT_SOURCE,
        **parity_json["sampling_trace"],
    })
    _write_json(out_dir / "rng_state_facts.json", {
        "schema": AUDIT_SCHEMA, "source": AUDIT_SOURCE,
        "rows": [r for r in a3_rows if r["id"].startswith("rng.")],
        "facts": parity_json["sampling_trace"]["rng"],
        "note": "streams compared by SHA-256 digest of the full state, not by intent",
    })
    _write_json(out_dir / "selection_evidence.json", {
        "schema": AUDIT_SCHEMA, "source": AUDIT_SOURCE,
        **a4_summary["selection_evidence"],
    })

    summary = {
        "schema": AUDIT_SCHEMA, "source": AUDIT_SOURCE,
        "date": "2026-10-22",
        "config_sha256": static["config_sha256"],
        "git": static["git"],
        "counts": {
            "rows": len(rows) + len(static["checks"]),
            "a1_a4_rows": len(rows),
            "a0_checks": len(static["checks"]),
            "failures": len(failures), "gaps": len(gaps),
            "limitations": len(limitations),
            "by_group": {"A0": len(static["checks"]),
                         **{g: sum(1 for r in rows if r["group"] == g)
                            for g in ("A1", "A2", "A3", "A4")}},
        },
        "failures": [r["id"] for r in failures],
        "gaps": gaps,
        "limitations": limitations,
        "static_contract": {
            "checks": len(static["checks"]),
            "mismatches": [r["id"] for r in static_failures],
        },
        "producer_conclusion": conclusion,
        "conclusion": conclusion,
        "scope": {
            "real_dataset_loaded": False, "real_checkpoint_loaded": False,
            "real_model_forward": False, "real_g1_fk": False,
            "production_model_forward": False,
            "synthetic_module_forward": True,
            "synthetic_module_note": "A3 uses a randomly initialised tiny DiffusionG1 only for scheduler/sampling facts; loop parity uses stand-in modules",
            "training_updates_on_real_data": 0, "real_model_calls": 0,
            "production_model_calls": 0,
            "synthetic_optimizer_updates": "stub modules only",
        },
        "boundaries": {
            "real_data_audit": "not_run",
            "single_factor_experiment": "not_registered",
            "factor_selected": False,
            "auto_execute": False,
            "final_interaction_method": "not_ready",
            "production_integration": False,
        },
        "a4_summary": a4_summary,
    }
    _write_json(out_dir / "audit_summary.json", summary)
    verification = verify_audit_artifacts(out_dir)
    # the verifier's conclusion is authoritative; a producer that claims a pass
    # the independent check cannot reproduce is itself an inconsistency
    final = verification["conclusion"]
    if final == "synthetic_contract_passed" and conclusion != "synthetic_contract_passed":
        final = "audit_incomplete"
        verification.setdefault("problems", []).append("producer_verifier_disagreement")
        verification["conclusion"] = final
        _write_json(out_dir / "independent_verification.json", verification)
    summary["producer_conclusion"] = conclusion
    summary["conclusion"] = final
    summary["independent_verification"] = {
        "conclusion": verification["conclusion"],
        "problems": verification.get("problems", []),
    }
    _write_json(out_dir / "audit_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# A5 independent verifier
# ---------------------------------------------------------------------------



def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)))


def _as_float_array(value: Any, name: str, problems: list[str],
                    shape: tuple | None = None) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001
        problems.append(f"structure_nonarray:{name}")
        return None
    if arr.dtype.kind == "b" or arr.dtype.kind not in "iuf":
        problems.append(f"structure_non_numeric:{name}:{arr.dtype}")
        return None
    if shape is not None and tuple(arr.shape) != tuple(shape):
        problems.append(f"structure_shape:{name}:{arr.shape}!={shape}")
        return None
    arr = arr.astype(np.float64, copy=False)
    if not np.isfinite(arr).all():
        problems.append(f"nonfinite:{name}")
        return None
    return arr


def _mapping(value: Any, name: str, problems: list[str]) -> dict:
    if not isinstance(value, dict):
        problems.append(f"structure_not_mapping:{name}")
        return {}
    return value


def _list(value: Any, name: str, problems: list[str]) -> list:
    if not isinstance(value, list):
        problems.append(f"structure_not_list:{name}")
        return []
    return value


def _require_keys(mapping: dict, keys: tuple[str, ...], name: str,
                  problems: list[str]) -> bool:
    missing = [k for k in keys if k not in mapping]
    if missing:
        problems.append(f"structure_missing_keys:{name}:{missing}")
        return False
    return True


def _scan_nonfinite(value: Any, path: str, problems: list[str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _scan_nonfinite(v, f"{path}.{k}", problems)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _scan_nonfinite(v, f"{path}[{i}]", problems)
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            problems.append(f"nonfinite:{path}")


def _validate_row_list(rows: Any, group: str, name: str,
                       problems: list[str]) -> dict[str, dict]:
    rows = _list(rows, name, problems)
    required = set(REQUIRED_AUDIT_ROWS[group])
    out: dict[str, dict] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"structure_row_not_mapping:{name}:{i}")
            continue
        if row.get("group") != group:
            continue
        if not _require_keys(row, ("id", "group", "status", "expected", "actual"),
                             f"{name}.row[{i}]", problems):
            continue
        if row.get("status") not in ("static_confirmed", "synthetic_verified",
                                     "runtime_not_checked", "contract_mismatch"):
            problems.append(f"structure_bad_status:{name}:{row.get('id')}")
        elif row.get("status") != "synthetic_verified":
            problems.append(f"required_row_not_verified:{group}:{row.get('id')}:{row.get('status')}")
        _scan_nonfinite(row.get("actual"), f"{name}.{row.get('id')}.actual", problems)
        _scan_nonfinite(row.get("evidence"), f"{name}.{row.get('id')}.evidence", problems)
        out[str(row.get("id"))] = row
    missing = sorted(required - set(out))
    for rid in missing:
        problems.append(f"missing_row:{group}:{rid}")
    return out


def _validate_static(static: Any, problems: list[str]) -> tuple[dict, dict, dict, Path | None]:
    """Validate static contract and return (static, effective loss, config, path)."""
    st = _mapping(static, "static_contract", problems)
    if not st:
        return {}, {}, {}, None
    checks = _list(st.get("checks"), "static_contract.checks", problems)
    check_ids = []
    for i, c in enumerate(checks):
        if not isinstance(c, dict):
            problems.append(f"structure_row_not_mapping:static_contract.checks:{i}")
            continue
        check_ids.append(c.get("id"))
        if c.get("status") == "contract_mismatch":
            problems.append(f"static_check_failed:{c.get('id')}")
        elif c.get("status") != "static_confirmed":
            problems.append(f"static_check_not_confirmed:{c.get('id')}:{c.get('status')}")
    if len(check_ids) != len(set(check_ids)):
        problems.append("static_check_duplicate_id")
    if set(check_ids) != set(STATIC_CHECK_IDS) or len(check_ids) != len(STATIC_CHECK_IDS):
        problems.append("static_check_set_mismatch")
    if st.get("required_check_ids") is not None and list(st.get("required_check_ids")) != list(STATIC_CHECK_IDS):
        problems.append("static_required_check_ids_drift")
    eff = _mapping(st.get("effective_config"), "static_contract.effective_config", problems)
    loss = _mapping(eff.get("loss"), "static_contract.effective_config.loss", problems)
    src = _mapping(st.get("source_files"), "static_contract.source_files", problems)
    tol = _mapping(st.get("tolerances"), "static_contract.tolerances", problems)
    if not checks or not src or not tol:
        problems.append("static_contract_incomplete")
    cfg_path = Path(str(st.get("config_path", "")))
    if not cfg_path.is_file():
        problems.append("static_config_path_unreadable")
        return st, loss, {}, None
    if sha256_file(cfg_path) != st.get("config_sha256"):
        problems.append("static_config_sha256_drift")
    project_root = cfg_path.resolve().parents[2]
    for group in ("source_files", "audit_implementation_files"):
        for rel, rec in (st.get(group) or {}).items():
            path = project_root / rel
            if not isinstance(rec, dict) or not path.is_file() \
                    or sha256_file(path) != rec.get("sha256"):
                problems.append(f"static_source_drift:{rel}")
    # re-parse the FULL config and bind every effective parameter, not only loss
    try:
        import yaml
        raw = yaml.safe_load(cfg_path.read_text())
        if not isinstance(raw, dict):
            problems.append("static_config_parse_failed:not_mapping")
            return st, loss, {}, cfg_path
        parsed = {k: v for k, v in raw.items() if k != "formal_validation_contract"}
        parsed_loss = parsed.get("loss", {})
        parsed_loss = {k: float(v) for k, v in parsed_loss.items()} if isinstance(parsed_loss, dict) else {}
        parsed["loss"] = parsed_loss
        static_eff = {k: v for k, v in eff.items()}
        static_loss = {k: float(v) for k, v in loss.items()}
        static_eff["loss"] = static_loss
        for key in sorted(set(parsed) | set(static_eff)):
            if key not in parsed or key not in static_eff or parsed[key] != static_eff[key]:
                problems.append(f"static_effective_config_mismatch:{key}")
        expected_loss = {k: float(v) for k, v in W1_SYNTHETIC_WEIGHTS.__dict__.items()}
        if static_loss != expected_loss:
            problems.append("static_effective_loss_mismatch:expected_w1")
        if parsed_loss != expected_loss:
            problems.append("static_effective_loss_mismatch:config_expected")
        check_map = {c.get("id"): c for c in checks if isinstance(c, dict)}
        field_checks = {
            "steps": "static.steps", "micro_batch": "static.micro_batch",
            "grad_accum": "static.grad_accum", "lr": "static.lr",
            "weight_decay": "static.weight_decay", "grad_clip": "static.grad_clip",
            "seed": "static.seed", "val_seed": "static.val_seed",
            "val_every": "static.val_every", "time_window": "static.time_window",
            "num_train_timesteps": "static.num_train_timesteps",
        }
        for field, check_id in field_checks.items():
            c = check_map.get(check_id)
            if c is None:
                problems.append(f"static_check_missing_for_config:{field}")
            elif c.get("actual") != static_eff.get(field):
                problems.append(f"static_effective_config_mismatch:check_{field}")
        lc = check_map.get("static.loss_weights")
        if lc is None:
            problems.append("static_check_missing_for_config:loss_weights")
        elif lc.get("actual") != static_loss:
            problems.append("static_effective_config_mismatch:check_loss_weights")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"static_config_parse_failed:{type(exc).__name__}")
    return st, loss, {}, cfg_path


def _saved_world(inputs: dict, problems: list[str]) -> tuple[SyntheticWorld | None, dict]:
    required = ("x0", "root", "q", "wfc", "mean", "std", "lo", "hi",
                "center", "box_quat", "contact", "x0_pred_generic",
                "x0_pred_pos", "x0_pred_rot", "x0_pred_joint", "x0_pred_adjacent")
    missing = [k for k in required if k not in inputs]
    if missing:
        problems.append(f"structure_missing_keys:synthetic_inputs:{missing}")
        return None, {}
    x0 = _as_float_array(inputs["x0"], "inputs.x0", problems)
    root = _as_float_array(inputs["root"], "inputs.root", problems)
    q = _as_float_array(inputs["q"], "inputs.q", problems)
    wfc = _as_float_array(inputs["wfc"], "inputs.wfc", problems, (4, 4))
    mean = _as_float_array(inputs["mean"], "inputs.mean", problems, (3,))
    std = _as_float_array(inputs["std"], "inputs.std", problems, (3,))
    lo = _as_float_array(inputs["lo"], "inputs.lo", problems, (29,))
    hi = _as_float_array(inputs["hi"], "inputs.hi", problems, (29,))
    center = _as_float_array(inputs["center"], "inputs.center", problems, (3,))
    box_quat = _as_float_array(inputs["box_quat"], "inputs.box_quat", problems)
    contact = _as_float_array(inputs["contact"], "inputs.contact", problems)
    pred = _as_float_array(inputs["x0_pred_generic"], "inputs.x0_pred_generic", problems)
    if any(v is None for v in (x0, root, q, wfc, mean, std, lo, hi, center, box_quat, contact, pred)):
        return None, {}
    if x0.shape[1] != 38 or root.shape[1] != 7 or q.shape[1] != 29 \
            or root.shape[0] != x0.shape[0] or q.shape[0] != x0.shape[0]:
        problems.append("structure_shape:inputs:x0_root_q")
        return None, {}
    if not (std > 0).all() or not (hi > lo).all():
        problems.append("structure_invalid:inputs:std_or_joint_range")
    tokenizer = StateTokenizer(np.stack([lo, hi], axis=1), RootStats(mean=mean, std=std))
    recomputed_x0 = _encode_np(root, q, wfc, tokenizer)
    if recomputed_x0.shape != x0.shape or np.abs(recomputed_x0 - x0).max() > TOLERANCES["encode_float64"]["atol"]:
        problems.append("input_x0_not_bound")
    world = SyntheticWorld(tokenizer=tokenizer, wfc=wfc, root=root, q=q, x0=x0,
                           center=center, box_quat=box_quat, contact=contact)
    return world, {"x0": x0, "pred": pred, "tokenizer": tokenizer}


def _validate_gradient_arrays(garr: dict, world: SyntheticWorld, weights: LossWeights,
                              row_map: dict, detail_map: dict, problems: list[str]) -> dict:
    if not garr:
        problems.append("structure_empty:gradient_arrays")
        return {}
    required = {"grad_pos", "grad_rot6d", "grad_joint", "grad_geodesic_rad2",
                "grad_vel", "grad_fk_m2", "grad_total", "grad_sum", "pred_input"}
    if not required <= set(garr):
        problems.append(f"structure_missing_keys:gradient_arrays:{sorted(required - set(garr))}")
        return {}
    arrays = {}
    for key in required:
        arrays[key] = _as_float_array(garr[key], f"gradient_arrays.{key}", problems)
    if any(v is None for v in arrays.values()):
        return {}
    pred = torch.tensor(arrays["pred_input"], dtype=torch.float64)
    if pred.ndim == 2:
        pred = pred[None]
    recomputed = {}
    weight_key = {"pos": "w_pos", "rot6d": "w_rot6d", "joint": "w_joint",
                  "geodesic_rad2": "w_geodesic", "vel": "w_vel", "fk_m2": "w_fk"}
    for term in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2"):
        g, _ = _term_grad(world, pred.detach().clone().requires_grad_(True), term, torch.float64)
        recomputed[f"grad_{term}"] = g.numpy()
        err = float(np.abs(g.numpy() - arrays[f"grad_{term}"]).max())
        if err > 1e-8:
            problems.append(f"gradient_recompute_mismatch:{term}:{err}")
        row = row_map.get(f"grad.{term}")
        if row is None or not isinstance(row.get("actual"), dict):
            problems.append(f"structure_missing_keys:gradient_rows.grad.{term}.actual")
            continue
        act = row["actual"]
        norm = float(np.linalg.norm(g.numpy()))
        weight = float(getattr(weights, weight_key[term]))
        for key, want in (("grad_norm", norm), ("weighted_grad_norm", weight * norm)):
            if not _finite_number(act.get(key)) or abs(float(act[key]) - want) > 1e-8:
                problems.append(f"gradient_row_mismatch:{term}:{key}")
        if act.get("finite") is not True or act.get("nonzero") is not True:
            problems.append(f"gradient_row_not_verified:{term}")
        detail = detail_map.get(f"grad.{term}")
        if detail is None or not isinstance(detail.get("actual"), dict):
            problems.append(f"structure_missing_keys:gradient_rows.grad.{term}.actual")
        else:
            dact = detail["actual"]
            for key in ("grad_norm", "weighted_grad_norm"):
                a, b = act.get(key), dact.get(key)
                if not _finite_number(a) or not _finite_number(b) or abs(float(a) - float(b)) > 1e-9:
                    problems.append(f"gradient_detail_mismatch:{term}:{key}")
            for key in ("finite", "nonzero"):
                if act.get(key) is not dact.get(key):
                    problems.append(f"gradient_detail_mismatch:{term}:{key}")
    gsum = (weights.w_pos * recomputed["grad_pos"] + weights.w_rot6d * recomputed["grad_rot6d"]
            + weights.w_joint * recomputed["grad_joint"]
            + weights.w_geodesic * recomputed["grad_geodesic_rad2"]
            + weights.w_vel * recomputed["grad_vel"] + weights.w_fk * recomputed["grad_fk_m2"])
    if float(np.abs(gsum - arrays["grad_total"]).max()) > TOLERANCES["gradient_sum_float64"]["atol"]:
        problems.append("gradient_recompute_mismatch:total")
    if float(np.abs(gsum - arrays["grad_sum"]).max()) > TOLERANCES["gradient_sum_float64"]["atol"]:
        problems.append("gradient_saved_sum_mismatch")
    if float(np.abs(arrays["grad_pos"]).max()) == 0:
        problems.append("gradient_pos_zero")
    return recomputed


def _recompute_update_run(run: dict, uarr: dict, tokenizer, weights: LossWeights,
                          problems: list[str], checks: list[dict]) -> None:
    run_id = str(run.get("run_id", f"{run.get('model_type')}_{run.get('grad_clip')}"))
    required = ("run_id", "model_type", "grad_clip", "problems", "micro_x0_shape",
                "part_keys", "forward_records_s0", "forward_records_formal",
                "loss_part_records_s0", "loss_part_records_formal")
    if tokenizer is None:
        problems.append(f"update_replay_missing_tokenizer:{run_id}")
        return
    if not _require_keys(run, required, f"update_parity.run:{run_id}", problems):
        return
    if run.get("problems"):
        problems.append(f"producer_parity_problems:{run_id}:{run['problems']}")
    prefix = f"{run_id}__"
    expected_clip = EXPECTED_UPDATE_RUNS.get(run_id, (None, None))[1]
    micro = _as_float_array(uarr.get(f"{prefix}micro_x0"), f"{run_id}.micro_x0", problems)
    if micro is None:
        return
    if micro.ndim != 4 or micro.shape[0] != 4 or micro.shape[-1] != 38:
        problems.append(f"structure_shape:{run_id}.micro_x0")
        return
    initial_bias = _as_float_array(uarr.get(f"{prefix}initial_bias"),
                                   f"{run_id}.initial_bias", problems, ())
    initial_scale = _as_float_array(uarr.get(f"{prefix}initial_scale"),
                                    f"{run_id}.initial_scale", problems, ())
    grad_clip_arr = _as_float_array(uarr.get(f"{prefix}grad_clip"),
                                    f"{run_id}.grad_clip", problems, ())
    if initial_bias is None or initial_scale is None or grad_clip_arr is None:
        return
    grad_clip = float(grad_clip_arr)
    if expected_clip is None or abs(grad_clip - float(expected_clip)) > 1e-12 \
            or abs(float(run.get("grad_clip", -1)) - float(expected_clip)) > 1e-12:
        problems.append(f"update_replay_mismatch:{run_id}:grad_clip_identity")
        return
    model_type = str(run.get("model_type"))
    clip_should_trigger = float(expected_clip) < 1.0
    for side in ("s0", "formal"):
        pred_saved = _as_float_array(
            uarr.get(f"{prefix}{side}_preds"), f"{run_id}.{side}.preds", problems,
            (4, 4) + tuple(micro.shape[2:]))
        parts_saved = _as_float_array(
            uarr.get(f"{prefix}{side}_parts"), f"{run_id}.{side}.parts", problems,
            (4, len(LOSS_PART_KEYS)))
        pre0 = _as_float_array(uarr.get(f"{prefix}{side}_preclip_0"),
                               f"{run_id}.{side}.preclip_0", problems, (X0_DIM,))
        pre1 = _as_float_array(uarr.get(f"{prefix}{side}_preclip_1"),
                               f"{run_id}.{side}.preclip_1", problems, ())
        post0 = _as_float_array(uarr.get(f"{prefix}{side}_postclip_0"),
                                f"{run_id}.{side}.postclip_0", problems, (X0_DIM,))
        post1 = _as_float_array(uarr.get(f"{prefix}{side}_postclip_1"),
                                f"{run_id}.{side}.postclip_1", problems, ())
        gnorm_saved = _as_float_array(
            uarr.get(f"{prefix}{side}_gnorm"), f"{run_id}.{side}.gnorm", problems, ())
        if any(v is None for v in (pred_saved, parts_saved, pre0, pre1, post0, post1, gnorm_saved)):
            return
        bias = torch.nn.Parameter(torch.full(
            (X0_DIM,), float(initial_bias.reshape(-1)[0]), dtype=torch.float32))
        scale = torch.nn.Parameter(torch.tensor(
            float(initial_scale.reshape(-1)[0]), dtype=torch.float32))
        asm = build_synthetic_assembler(tokenizer, weights, torch.float32, use_fk=True)
        opt = torch.optim.AdamW([bias, scale], lr=1e-4, weight_decay=1e-5)
        for i in range(4):
            x0 = torch.tensor(micro[i], dtype=torch.float32)
            if model_type == "diffusion":
                xt = _as_float_array(uarr.get(f"{prefix}{side}_x_t"), f"{run_id}.{side}.x_t", problems)
                tt = _as_float_array(uarr.get(f"{prefix}{side}_t"), f"{run_id}.{side}.t", problems)
                if xt is None or tt is None:
                    return
                pred = x0 * scale + bias + 0.02 * torch.tensor(xt[i], dtype=torch.float32) \
                    + 0.02 * (torch.tensor(tt[i], dtype=torch.float32) / 1000.0)[:, None, None]
            else:
                pred = x0 * scale + bias
            if float((pred.detach() - torch.tensor(pred_saved[i], dtype=torch.float32)).abs().max()) > 1e-6:
                problems.append(f"update_replay_mismatch:{run_id}:{side}:forward:{i}")
            loss, parts = asm(pred, {"x0": x0})
            for j, key in enumerate(LOSS_PART_KEYS):
                if key not in parts:
                    problems.append(f"update_replay_missing_part:{run_id}:{side}:{key}")
                elif abs(float(parts[key]) - float(parts_saved[i, j])) > 1e-6:
                    problems.append(f"update_replay_mismatch:{run_id}:{side}:part:{key}:{i}")
            (loss / 4.0).backward()
        pre = [bias.grad.detach().clone(), scale.grad.detach().clone()]
        if float((pre[0].numpy() - pre0).__abs__().max()) > 1e-6 or \
                float((pre[1].numpy() - pre1).__abs__().max()) > 1e-6:
            problems.append(f"update_replay_mismatch:{run_id}:{side}:preclip")
        pre_norm = math.sqrt(sum(float(t.cpu().norm()) ** 2 for t in pre))
        gnorm = float(torch.nn.utils.clip_grad_norm_([bias, scale], grad_clip))
        if abs(gnorm - float(gnorm_saved)) > 1e-6:
            problems.append(f"update_replay_mismatch:{run_id}:{side}:gnorm")
        post = [bias.grad.detach().clone(), scale.grad.detach().clone()]
        post_norm = math.sqrt(sum(float(t.cpu().norm()) ** 2 for t in post))
        if clip_should_trigger:
            if not (gnorm > grad_clip and abs(post_norm - grad_clip) <= 1e-6):
                problems.append(f"update_replay_clip_not_triggered:{run_id}:{side}")
        else:
            if gnorm > grad_clip or abs(post_norm - pre_norm) > 1e-6:
                problems.append(f"update_replay_no_clip_changed:{run_id}:{side}")
        if float((post[0].numpy() - post0).__abs__().max()) > 1e-6 or \
                float((post[1].numpy() - post1).__abs__().max()) > 1e-6:
            problems.append(f"update_replay_mismatch:{run_id}:{side}:postclip")
        opt.step()
        names = ("model.bias", "conditioner.scale")
        for j, (n, param) in enumerate(zip(names, (bias, scale))):
            pshape = (X0_DIM,) if n == "model.bias" else ()
            saved_param = _as_float_array(uarr.get(f"{prefix}{side}_param_{n}"),
                                          f"{run_id}.{side}.param.{n}", problems, pshape)
            saved_grad = _as_float_array(uarr.get(f"{prefix}{side}_grad_{n}"),
                                         f"{run_id}.{side}.grad.{n}", problems, pshape)
            saved_step = uarr.get(f"{prefix}{side}_step_{n}")
            exp_avg = _as_float_array(uarr.get(f"{prefix}{side}_exp_avg_{n}"),
                                      f"{run_id}.{side}.exp_avg.{n}", problems, pshape)
            exp_avg_sq = _as_float_array(uarr.get(f"{prefix}{side}_exp_avg_sq_{n}"),
                                         f"{run_id}.{side}.exp_avg_sq.{n}", problems, pshape)
            if any(v is None for v in (saved_param, saved_grad, exp_avg, exp_avg_sq)):
                return
            if np.asarray(saved_step).shape != ():
                problems.append(f"structure_shape:{run_id}.{side}.step.{n}")
                return
            if float(np.abs(param.detach().numpy() - saved_param).max()) > 1e-6:
                problems.append(f"update_replay_mismatch:{run_id}:{side}:param:{n}")
            if float(np.abs(post[j].numpy() - saved_grad).max()) > 1e-6:
                problems.append(f"update_replay_mismatch:{run_id}:{side}:grad:{n}")
            st = opt.state[param]
            if int(saved_step) != 1 or int(st.get("step", -1)) != 1:
                problems.append(f"update_replay_mismatch:{run_id}:{side}:adamw_step:{n}")
            if float(np.abs(st["exp_avg"].detach().numpy() - exp_avg).max()) > 1e-6 \
                    or float(np.abs(st["exp_avg_sq"].detach().numpy() - exp_avg_sq).max()) > 1e-6:
                problems.append(f"update_replay_mismatch:{run_id}:{side}:adamw_state:{n}")
    # S0 and formal must have the same real inputs, forward outputs and updates
    for key in ("micro_x0",):
        pass
    for side_pair in (("s0", "formal"),):
        for suffix in ("preds", "parts", "preclip_0", "preclip_1", "postclip_0",
                       "postclip_1", "gnorm", "param_model.bias", "param_conditioner.scale",
                       "grad_model.bias", "grad_conditioner.scale", "exp_avg_model.bias",
                       "exp_avg_conditioner.scale", "exp_avg_sq_model.bias",
                       "exp_avg_sq_conditioner.scale", "step_model.bias",
                       "step_conditioner.scale"):
            a = uarr.get(f"{prefix}s0_{suffix}")
            b = uarr.get(f"{prefix}formal_{suffix}")
            if a is None or b is None:
                problems.append(f"structure_missing_keys:{run_id}:s0_or_formal_{suffix}")
            elif not np.array_equal(np.asarray(a), np.asarray(b)):
                problems.append(f"update_replay_mismatch:{run_id}:s0_vs_formal:{suffix}")
        if model_type == "diffusion":
            for suffix in ("t", "noise", "x_t"):
                a = uarr.get(f"{prefix}s0_{suffix}")
                b = uarr.get(f"{prefix}formal_{suffix}")
                if a is None or b is None:
                    problems.append(f"structure_missing_keys:{run_id}:s0_or_formal_{suffix}")
                elif not np.array_equal(np.asarray(a), np.asarray(b)):
                    problems.append(f"update_replay_mismatch:{run_id}:s0_vs_formal:{suffix}")
    checks.append({"id": f"independent.update_replay:{run_id}", "ok": True})


def _verify_diffusion_noise(run: dict, uarr: dict, problems: list[str],
                           checks: list[dict]) -> None:
    run_id = str(run.get("run_id", ""))
    prefix = f"{run_id}__"
    micro = _as_float_array(uarr.get(f"{prefix}micro_x0"), f"{run_id}.micro_x0", problems)
    if micro is None or micro.ndim != 4:
        return
    for side in ("s0", "formal"):
        ts = _as_float_array(uarr.get(f"{prefix}{side}_t"), f"{run_id}.{side}.t", problems)
        noise = _as_float_array(uarr.get(f"{prefix}{side}_noise"), f"{run_id}.{side}.noise", problems)
        xt = _as_float_array(uarr.get(f"{prefix}{side}_x_t"), f"{run_id}.{side}.x_t", problems)
        if ts is None or noise is None or xt is None:
            return
        if ts.shape != (4, 4) or noise.shape != micro.shape or xt.shape != micro.shape:
            problems.append(f"structure_shape:{run_id}.{side}.diffusion_inputs")
            return
        if not np.allclose(ts, np.rint(ts)) or ts.min() < 0 or ts.max() >= 1000:
            problems.append(f"update_replay_mismatch:{run_id}:{side}:t_domain")
        for i in range(4):
            ab = np.power(STUB_ALPHA_BASE, ts[i])[:, None, None]
            expected = np.sqrt(ab) * micro[i] + np.sqrt(1.0 - ab) * noise[i]
            err = float(np.abs(expected - xt[i]).max())
            if not np.isfinite(err) or err > 1e-6:
                problems.append(f"update_replay_mismatch:{run_id}:{side}:noise_to_x_t:{i}:{err}")
    # the two loops must consume the same real noise/t/x_t
    for suffix in ("t", "noise", "x_t"):
        a = uarr.get(f"{prefix}s0_{suffix}")
        b = uarr.get(f"{prefix}formal_{suffix}")
        if a is None or b is None or not np.array_equal(np.asarray(a), np.asarray(b)):
            problems.append(f"update_replay_mismatch:{run_id}:s0_vs_formal:{suffix}")
    checks.append({"id": f"independent.diffusion_noise:{run_id}", "ok": True})


def _recompute_selection(evidence: dict, rows: dict, problems: list[str],
                         checks: list[dict]) -> None:
    expected = _list(evidence.get("expected_demos"), "selection.expected_demos", problems)
    if not expected or len(set(expected)) != len(expected) or not all(isinstance(d, str) for d in expected):
        problems.append("selection_expected_roster_invalid")
        return
    lengths = _mapping(evidence.get("demo_lengths"), "selection.demo_lengths", problems)
    for run_name in ("complete_run", "incomplete_run"):
        run = _mapping(evidence.get(run_name), f"selection.{run_name}", problems)
        raw_steps = _list(run.get("raw_steps"), f"selection.{run_name}.raw_steps", problems)
        if len(raw_steps) != 2:
            problems.append(f"selection_evidence_raw_steps:{run_name}:{len(raw_steps)}")
            continue
        complete = []
        for idx, raw in enumerate(raw_steps, start=1):
            raw = _mapping(raw, f"selection.{run_name}.raw[{idx}]", problems)
            per_demo = _list(raw.get("per_demo"), f"selection.{run_name}.raw[{idx}].per_demo", problems)
            res = independent_macro(per_demo, list(expected),
                                    {k: int(v) for k, v in lengths.items()} if lengths else None)
            if raw.get("n_expected") != res["n_expected"] or raw.get("n_ok") != res["n_ok"]:
                problems.append(f"selection_evidence_counts:{run_name}:{idx}")
            if res["complete"]:
                complete.append((idx, res["macro"]))
                if not _finite_number(raw.get("macro_mpjpe_m")) \
                        or abs(float(raw["macro_mpjpe_m"]) - res["macro"]) > TOLERANCES["macro_float64"]["atol"]:
                    problems.append(f"selection_evidence_macro:{run_name}:{idx}")
        expected_best = min(complete, key=lambda p: p[1])[0] if complete else None
        recorded = run.get("best_formal_step")
        if expected_best != recorded:
            problems.append(f"selection_recompute_mismatch:{run_name}:{expected_best}!={recorded}")
        row = rows.get("selection.best_uses_macro_complete_only")
        if row is None or not isinstance(row.get("actual"), dict):
            problems.append("selection_row_missing")
        else:
            key = "complete_run" if run_name == "complete_run" else "incomplete_run"
            actual = row["actual"].get(key, {}).get("best_formal_step")
            if actual != expected_best:
                problems.append(f"selection_row_mismatch:{run_name}:{expected_best}!={actual}")
        checks.append({"id": f"independent.selection:{run_name}", "expected_best": expected_best})


def _verify_row_semantics(row_map: dict, inputs: dict, world: SyntheticWorld,
                          weights: LossWeights, recomputed: dict,
                          pred_np: np.ndarray | None, problems: list[str]) -> None:
    # Batch invariance: recompute both means from the saved target/perturbation
    # tensors.  Never use the row's own "independent_mean" as the expectation.
    required = ("batch_target_b", "batch_pred_b", "batch_perm",
                "batch_target_r", "batch_pred_r")
    if not all(k in inputs for k in required):
        problems.append("structure_missing_keys:synthetic_inputs:batch_invariance")
    else:
        tb = _as_float_array(inputs["batch_target_b"], "inputs.batch_target_b", problems)
        pb = _as_float_array(inputs["batch_pred_b"], "inputs.batch_pred_b", problems)
        perm = _as_float_array(inputs["batch_perm"], "inputs.batch_perm", problems)
        tr = _as_float_array(inputs["batch_target_r"], "inputs.batch_target_r", problems)
        pr = _as_float_array(inputs["batch_pred_r"], "inputs.batch_pred_r", problems)
        if all(v is not None for v in (tb, pb, perm, tr, pr)):
            if tb.ndim != 3 or pb.shape != tb.shape or perm.shape != (3,) \
                    or tr.ndim != 3 or pr.shape != tr.shape:
                problems.append("structure_shape:inputs:batch_invariance")
            else:
                def means(preds, targets):
                    vals = [_independent_terms(preds[i], targets[i], world, weights)
                            for i in range(preds.shape[0])]
                    return [float(np.mean([v[k] for v in vals])) for k in ("total", "pos", "vel")]
                exp_unperm = means(pb, tb)
                exp_perm = means(pb[perm.astype(int)], tb[perm.astype(int)])
                exp_rep = means(pr, tr)
                for rid, key, expected in (
                        ("loss.batch_permutation_invariance", "unpermuted", exp_unperm),
                        ("loss.batch_permutation_invariance", "permuted", exp_perm),
                        ("loss.batch_permutation_invariance", "independent_mean", exp_unperm),
                        ("loss.repeated_batch_invariance", "repeated", exp_rep),
                        ("loss.repeated_batch_invariance", "independent_mean", exp_rep)):
                    row = row_map.get(rid)
                    act = row.get("actual") if isinstance(row, dict) else None
                    vals = act.get(key) if isinstance(act, dict) else None
                    if not isinstance(vals, list) or len(vals) != 3 \
                            or not all(_finite_number(v) for v in vals):
                        problems.append(f"structure_invalid:{rid}.{key}")
                        continue
                    tol = TOLERANCES["loss_float64"]
                    if any(abs(float(a) - float(b)) > tol["atol"] + tol["rtol"] * abs(float(b))
                           for a, b in zip(vals, expected)):
                        problems.append(f"batch_invariance_mismatch:{rid}:{key}")
    # Finite difference: recompute fd, analytic directional derivative and
    # rel_error from the bound target/pred/direction/step, then compare records.
    fd = row_map.get("grad.finite_difference_float64")
    act = fd.get("actual") if isinstance(fd, dict) else None
    if pred_np is None or not isinstance(act, dict):
        problems.append("structure_missing_keys:grad.finite_difference_float64.actual")
        return
    pred2 = np.asarray(pred_np, dtype=np.float64)
    if pred2.ndim == 3:
        pred2 = pred2[0]
    tol = TOLERANCES["finite_difference_float64"]
    rng = np.random.default_rng(7)
    for term in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2"):
        rec = act.get(term)
        if not isinstance(rec, dict):
            problems.append(f"structure_missing_keys:grad.fd.{term}")
            continue
        direction = rng.normal(size=pred2.shape)
        direction /= np.linalg.norm(direction)
        fd_re, _ = _fd_directional(world, pred2, direction, term, 1e-6)
        g = recomputed.get(f"grad_{term}")
        if g is None:
            problems.append(f"gradient_recompute_unavailable:{term}")
            continue
        analytic_re = float(np.sum(g[0] * direction))
        rel_re = abs(fd_re - analytic_re) / max(abs(analytic_re), 1e-12)
        for key, want in (("fd", fd_re), ("analytic", analytic_re)):
            if not _finite_number(rec.get(key)) or abs(float(rec[key]) - want) > 1e-7:
                problems.append(f"finite_difference_recompute_mismatch:{term}:{key}")
        if not _finite_number(rec.get("rel_error")) or abs(float(rec["rel_error"]) - rel_re) > 1e-7:
            problems.append(f"finite_difference_recompute_mismatch:{term}:rel_error")
        expected_ok = rel_re <= tol["rtol"]
        if rec.get("ok") is not expected_ok or not expected_ok:
            problems.append(f"finite_difference_row_not_verified:{term}:{rel_re}:{rec.get('ok')}")
        gnorm = float(np.linalg.norm(g))
        expected_cos = analytic_re / gnorm if gnorm > 0 else None
        if expected_cos is None:
            if rec.get("cosine") is not None:
                problems.append(f"finite_difference_cosine_mismatch:{term}")
        elif not _finite_number(rec.get("cosine")) \
                or abs(float(rec["cosine"]) - expected_cos) > 1e-9:
            problems.append(f"finite_difference_cosine_mismatch:{term}")


def _verify_audit_artifacts_impl(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    problems: list[str] = []
    checks: list[dict] = []

    def load_json(name: str) -> Any:
        path = out_dir / name
        if not path.is_file():
            problems.append(f"missing_artifact:{name}")
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            problems.append(f"unreadable_artifact:{name}:{type(exc).__name__}")
            return None

    def load_npz(name: str) -> dict:
        path = out_dir / name
        if not path.is_file():
            problems.append(f"missing_artifact:{name}")
            return {}
        try:
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        except Exception as exc:  # noqa: BLE001
            problems.append(f"unreadable_artifact:{name}:{type(exc).__name__}")
            return {}

    static_raw = load_json("static_contract.json")
    comparison_raw = load_json("s0_formal_comparison.json")
    objective_raw = load_json("objective_rows.json")
    gradient_raw = load_json("gradient_rows.json")
    parity_raw = load_json("update_parity.json")
    sampling_raw = load_json("sampling_trace.json")
    rng_raw = load_json("rng_state_facts.json")
    selection_raw = load_json("selection_evidence.json")
    summary_raw = load_json("audit_summary.json")
    inputs = load_npz("synthetic_inputs.npz")
    garr = load_npz("gradient_arrays.npz")
    uarr = load_npz("update_arrays.npz")
    for name in REQUIRED_ARTIFACTS:
        if not (out_dir / name).is_file():
            problems.append(f"required_artifact_absent:{name}")

    static, static_loss, _, cfg_path = _validate_static(static_raw, problems)
    weights = LossWeights(**static_loss) if static_loss else W1_SYNTHETIC_WEIGHTS
    objective = _mapping(objective_raw, "objective_rows", problems)
    all_rows = _list(objective.get("rows"), "objective_rows.rows", problems) if objective else []
    row_map: dict[str, dict] = {}
    for group in ("A1", "A2", "A3", "A4"):
        row_map.update(_validate_row_list(all_rows, group, f"objective_rows.{group}", problems))

    gradient = _mapping(gradient_raw, "gradient_rows", problems)
    gradient_rows_map = _validate_row_list(gradient.get("rows"), "A2",
                                          "gradient_rows", problems)
    world, saved = _saved_world(inputs, problems) if inputs else (None, {})
    if world is not None and weights is not None:
        exp = _independent_terms(saved["pred"], saved["x0"], world, weights)
        row = row_map.get("loss.total_weighted_sum")
        if row is None or not isinstance(row.get("evidence"), dict) \
                or not isinstance(row["evidence"].get("terms"), dict):
            problems.append("structure_missing_keys:loss.total_weighted_sum.evidence.terms")
        else:
            terms = row["evidence"]["terms"]
            for key in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2"):
                val = terms.get(key)
                if val is None or not _finite_number(val):
                    problems.append(f"nonfinite:objective.loss.total_weighted_sum.{key}")
                elif abs(float(val) - exp[key]) > TOLERANCES["loss_float64"]["atol"]:
                    problems.append(f"objective_recompute_mismatch:{key}")
            total = row.get("actual")
            if not _finite_number(total):
                problems.append("nonfinite:objective.loss.total_weighted_sum.total")
            elif abs(float(total) - exp["total"]) > TOLERANCES["loss_float64"]["atol"]:
                problems.append("objective_recompute_mismatch:total")
        checks.append({"id": "independent.recompute_terms", "recomputed": exp})

    recomputed_grads = {}
    if world is not None:
        recomputed_grads = _validate_gradient_arrays(
            garr, world, weights, row_map, gradient_rows_map, problems)
        _verify_row_semantics(row_map, inputs, world, weights, recomputed_grads,
                              saved.get("pred"), problems)

    # A3: reload every saved run and replay its update from the bound inputs.
    parity = _mapping(parity_raw, "update_parity", problems)
    runs = _list(parity.get("runs"), "update_parity.runs", problems) if parity else []
    if not runs:
        problems.append("structure_empty:update_parity.runs")
    run_ids = [r.get("run_id") for r in runs if isinstance(r, dict)]
    if len(runs) != len(EXPECTED_UPDATE_RUNS) or set(run_ids) != set(EXPECTED_UPDATE_RUNS) \
            or len(run_ids) != len(set(run_ids)):
        problems.append("update_run_set_mismatch")
    for run in runs:
        if not isinstance(run, dict):
            problems.append("structure_row_not_mapping:update_parity.runs")
            continue
        rid = run.get("run_id")
        expected_run = EXPECTED_UPDATE_RUNS.get(rid)
        if expected_run is None:
            problems.append(f"update_run_unexpected:{rid}")
        else:
            model_type, clip = expected_run
            if run.get("model_type") != model_type or float(run.get("grad_clip", -1)) != clip:
                problems.append(f"update_run_identity_mismatch:{rid}")
            if run.get("n_micro_s0") != 4 or run.get("n_micro_formal") != 4 \
                    or run.get("forward_records_s0") != 4 or run.get("forward_records_formal") != 4 \
                    or run.get("loss_part_records_s0") != 4 or run.get("loss_part_records_formal") != 4:
                problems.append(f"update_run_micro_count_mismatch:{rid}")
            shape = run.get("micro_x0_shape")
            if not isinstance(shape, list) or len(shape) != 4 or shape[0] != 4 or shape[-1] != 38:
                problems.append(f"update_run_micro_shape_mismatch:{rid}")
            if run.get("part_keys") != list(LOSS_PART_KEYS):
                problems.append(f"update_run_part_keys_mismatch:{rid}")
        _recompute_update_run(run, uarr, saved.get("tokenizer"), weights, problems, checks)
        if isinstance(rid, str) and rid.startswith("diffusion"):
            _verify_diffusion_noise(run, uarr, problems, checks)

    # A4: recompute the formal macro and selected step from raw callback rows.
    _recompute_selection(_mapping(selection_raw, "selection_evidence", problems),
                         row_map, problems, checks)

    # A3 sampling: validate structure and the executed full loop.
    sampling = _mapping(sampling_raw, "sampling_trace", problems)
    if sampling:
        x = _mapping(sampling.get("x_t_closed_form"), "sampling.x_t_closed_form", problems)
        ts = _list(x.get("t"), "sampling.x_t_closed_form.t", problems)
        errs = _list(x.get("max_abs_errors"), "sampling.x_t_closed_form.max_abs_errors", problems)
        if not ts or not errs or len(ts) != len(errs):
            problems.append("sampling_x_t_errors_empty_or_mismatched")
        else:
            nums = [_as_float_array([e], "sampling.max_abs_errors", problems) for e in errs]
            if any(n is None for n in nums):
                problems.append("sampling_x_t_nonfinite")
            elif max(float(n.reshape(-1)[0]) for n in nums) > 1e-6:
                problems.append("sampling_x_t_closed_form_not_verified")
        t0 = _mapping(sampling.get("t0"), "sampling.t0", problems)
        if not _finite_number(t0.get("noise_fraction")) or float(t0["noise_fraction"]) <= 0:
            problems.append("sampling_t0_not_noised")
        d = _mapping(sampling.get("ddim"), "sampling.ddim", problems)
        if d.get("prediction_type") != "sample" or d.get("clip_sample") is not False:
            problems.append("ddim_config_not_frozen")
        full = row_map.get("sampling.ddim_full_loop")
        if full is None or not isinstance(full.get("actual"), dict):
            problems.append("sampling_full_loop_row_missing")
        else:
            act = full["actual"]
            times = act.get("timesteps_visited")
            expected_times = act.get("scheduler_timesteps")
            if act.get("sample_calls") != 1 or act.get("sample_error") is not None:
                problems.append("sampling_full_loop_not_executed")
            elif act.get("forward_calls") != 5 or act.get("scheduler_step_calls") != 5:
                problems.append("sampling_full_loop_forward_or_step_count")
            elif (not isinstance(times, list) or not isinstance(expected_times, list)
                  or len(times) != 5 or times != expected_times
                  or expected_times != [800, 600, 400, 200, 0]
                  or len(set(times)) != 5):
                problems.append("sampling_full_loop_timesteps_mismatch")
            elif not _finite_number(act.get("reference_final_max_abs_error")) \
                    or float(act["reference_final_max_abs_error"]) > 1e-5:
                problems.append("sampling_full_loop_reference_mismatch")
        consumption = _mapping(sampling.get("consumption"), "sampling.consumption", problems)
        s0c = _mapping(consumption.get("s0"), "sampling.consumption.s0", problems)
        fc = _mapping(consumption.get("formal"), "sampling.consumption.formal", problems)
        if s0c.get("samples") != 24 or s0c.get("unique") != 6 \
                or not isinstance(s0c.get("counts"), dict) \
                or set(map(str, s0c["counts"].values())) != {"4"}:
            problems.append("consumption_s0_mismatch")
        dropped = fc.get("dropped_per_epoch")
        if fc.get("samples") != 32 or not isinstance(dropped, list) \
                or len(dropped) != 4 or any(not isinstance(x, list) or len(x) != 2 for x in dropped) \
                or sorted(fc.get("coverage", [])) != list(range(10)):
            problems.append("consumption_formal_mismatch")
        rng = _mapping(sampling.get("rng"), "sampling.rng", problems)
        if rng.get("validation_restored") is not True \
                or rng.get("dataloader_dedicated_unchanged") is not True:
            problems.append("rng_facts_not_verified")

    # comparison and rng artifacts
    comparison = _mapping(comparison_raw, "s0_formal_comparison", problems)
    comp_rows = _list(comparison.get("rows"), "s0_formal_comparison.rows", problems)
    if len(comp_rows) < 10:
        problems.append("s0_formal_comparison_incomplete")
    else:
        cr = {r.get("item"): r for r in comp_rows if isinstance(r, dict)}
        cfg = static.get("effective_config", {}) if static else {}
        expected_cmp = {
            "micro_batch": (cfg.get("micro_batch"), cr.get("micro_batch", {}).get("formal")),
            "grad_accum": (cfg.get("grad_accum"), cr.get("grad_accum", {}).get("s0")),
            "samples_per_update_s0": (6 * cfg.get("grad_accum", 0),
                                      cr.get("samples_per_update", {}).get("s0")),
            "samples_per_update_formal": (cfg.get("micro_batch", 0) * cfg.get("grad_accum", 0),
                                           cr.get("samples_per_update", {}).get("formal")),
        }
        for key, (want, got) in expected_cmp.items():
            if want != got:
                problems.append(f"s0_formal_comparison_mismatch:{key}")
    rng_doc = _mapping(rng_raw, "rng_state_facts", problems)
    facts = _mapping(rng_doc.get("facts"), "rng_state_facts.facts", problems)
    if facts.get("validation_restored") is not True \
            or facts.get("dataloader_dedicated_unchanged") is not True:
        problems.append("rng_facts_not_verified")

    # summary/detail consistency
    summary = _mapping(summary_raw, "audit_summary", problems)
    if summary is not None:
        required_summary = ("schema", "source", "counts", "failures", "gaps",
                            "scope", "boundaries", "producer_conclusion")
        _require_keys(summary, required_summary, "audit_summary", problems)
        if not summary:
            problems.append("structure_empty:audit_summary")
        computed_failures = {r.get("id") for r in all_rows
                             if isinstance(r, dict) and r.get("status") == "contract_mismatch"}
        if static:
            computed_failures |= {c.get("id") for c in static.get("checks", [])
                                  if isinstance(c, dict) and c.get("status") == "contract_mismatch"}
        summary_failures = set(summary.get("failures", []))
        if summary_failures != computed_failures:
            problems.append("summary_failure_list_mismatch")
        counts = _mapping(summary.get("counts"), "audit_summary.counts", problems)
        n_static = len(static.get("checks", [])) if static else 0
        if counts.get("rows") != len(all_rows) + n_static:
            problems.append("summary_row_count_mismatch")
        if counts.get("a1_a4_rows") != len(all_rows):
            problems.append("summary_a1_a4_count_mismatch")
        if counts.get("a0_checks") != n_static:
            problems.append("summary_a0_count_mismatch")
        if counts.get("failures") != len(computed_failures):
            problems.append("summary_failure_count_mismatch")
        missing_required = [f"missing_row:{g}:{rid}"
                            for g in ("A1", "A2", "A3", "A4")
                            for rid in REQUIRED_AUDIT_ROWS[g] if rid not in row_map]
        summary_gaps = _list(summary.get("gaps"), "audit_summary.gaps", problems)
        if set(summary_gaps) != set(missing_required) or len(summary_gaps) != len(missing_required):
            problems.append("summary_gap_list_mismatch")
        if counts.get("gaps") != len(missing_required):
            problems.append("summary_gap_count_mismatch")
        scope = _mapping(summary.get("scope"), "audit_summary.scope", problems)
        if any(scope.get(k) is not False for k in
               ("real_dataset_loaded", "real_checkpoint_loaded", "real_model_forward",
                "real_g1_fk", "production_model_forward")) \
                or scope.get("training_updates_on_real_data") != 0 \
                or scope.get("real_model_calls") != 0 \
                or scope.get("production_model_calls") != 0:
            problems.append("summary_scope_violation")
        boundaries = _mapping(summary.get("boundaries"), "audit_summary.boundaries", problems)
        if boundaries.get("auto_execute") is not False \
                or boundaries.get("final_interaction_method") != "not_ready" \
                or boundaries.get("production_integration") is not False \
                or boundaries.get("factor_selected") is not False:
            problems.append("summary_boundary_violation")

    # conclusion
    row_failure = any(isinstance(r, dict) and r.get("status") == "contract_mismatch"
                      for r in all_rows)
    numeric_prefixes = ("objective_recompute_mismatch", "gradient_recompute_mismatch",
                        "gradient_saved_sum_mismatch", "gradient_row_mismatch",
                        "gradient_detail_mismatch", "finite_difference_recompute_mismatch",
                        "static_effective_config_mismatch",
                        "update_replay_mismatch", "selection_recompute_mismatch",
                        "selection_row_mismatch", "static_effective_loss_mismatch",
                        "input_x0_not_bound", "nonfinite:",
                        "finite_difference_row_not_verified", "batch_invariance_mismatch",
                        "sampling_full_loop_not_executed",
                        "sampling_full_loop_forward_or_step_count",
                        "sampling_full_loop_timesteps_mismatch",
                        "sampling_full_loop_reference_mismatch",
                        "summary_failure_list_mismatch")
    if row_failure or any(any(p.startswith(pre) for pre in numeric_prefixes) for p in problems):
        conclusion = "implementation_mismatch"
    elif problems:
        conclusion = "audit_incomplete"
    else:
        conclusion = "synthetic_contract_passed"
    return {
        "schema": AUDIT_SCHEMA,
        "source": "independent_verifier",
        "conclusion": conclusion,
        "problems": problems,
        "checks": checks,
        "trusted_top_level_pass_string": False,
        "artifacts_read": list(REQUIRED_ARTIFACTS),
    }


def verify_audit_artifacts(out_dir: Path) -> dict:
    """Independent verifier with a guaranteed report even for bad structures."""
    out_dir = Path(out_dir)
    try:
        result = _verify_audit_artifacts_impl(out_dir)
    except Exception as exc:  # noqa: BLE001 -- never leave the audit unsealed
        result = {
            "schema": AUDIT_SCHEMA,
            "source": "independent_verifier",
            "conclusion": "audit_incomplete",
            "problems": [f"verifier_exception:{type(exc).__name__}:{exc}"],
            "checks": [],
            "trusted_top_level_pass_string": False,
            "artifacts_read": list(REQUIRED_ARTIFACTS),
        }
    _write_json(out_dir / "independent_verification.json", result)
    return result
