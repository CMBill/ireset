"""W1 prediction/window consistency diagnostics (plan §3-§8).

Pure measurement and bookkeeping for the fixed-checkpoint diagnostic:

- exact window/probe key sets and denominators (never inferred from loaded rows);
- teacher 38-dim targets through the production tokenizer;
- production Torch decode from saved normalized predictions;
- per-window training losses (original LossAssembler weights, contact silent),
  physical state errors, frame-aligned geometry and temporal facts;
- same-frame cross-window pairs over ALL positive-weight contributors;
- fused penetration classification and temporal gate facts;
- equal-per-demo aggregation (train6/val6 macros, known case separate);
- deterministic noise seeds for the five fixed denoise probes.

No network is constructed, no fusion candidate is run and no threshold is
invented for teacher-fit acceptance. Measured facts stay facts; the
attribution judgement only reports which mechanism families have evidence
and which comparisons remain undecided (plan §7).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from ireset.kinematics.rotation_adapter import (
    rot6d_to_mat_torch,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from ireset.model.stitch import window_starts, window_weight
from ireset.model.tokens import mat_to_quat_torch

WINDOW = 60
STRIDE = 30
FPS = 30.0
FRAME_DT = 1.0 / FPS
MODELS = ("deterministic", "diffusion")
CONDITION_KEYS = ("kps_obj", "kps_clip", "box_t_clip", "box_quat", "contact")
GEOMETRY_FAMILIES = ("robot_ground", "robot_box", "self")
PROBE_NAMES = ("teacher_roundtrip", "deterministic_generation",
               "diffusion_denoise", "diffusion_ddim25")
DENOISE_TIMESTEPS = (0, 250, 500, 750, 999)
NOISE_SEED = 20261021
NOISE_PROTOCOL = "w1_prediction_consistency_v1"
KEYFRAME_PENETRATION_M = 0.002
ANGULAR_SPEED_LIMIT_RADPS = 7.0
ROOT_SPEED_LIMIT_MPS = 1.8
JOINT_SPEED_LIMIT_RADPS = 12.0
QUAT_NORM_TOL = 1e-5
REPLAY_TOL = {"normalized": 1e-5, "position_m": 1e-5,
              "joints_rad": 1e-5, "rotation_deg": 1e-3}
ORACLE_TOL = {"position_m": 1e-5, "joints_rad": 1e-5,
              "rotation_deg": 1e-3}
F64_METRIC_TOL = {"position_m": 1e-6, "joints_rad": 1e-6,
                  "rotation_rad": 1e-6}
LOCAL_BINS = ((0, 10, "0-10"), (10, 50, "10-50"), (50, 60, "50-60"))

#: Descriptive attribution comparators (NOT acceptance gates). Registered in
#: ``freeze/execution_registration.json``; the final report always carries the
#: absolute magnitudes, per-demo curves and the oracle/teacher comparison.
ATTRIBUTION_COMPARATORS = {
    "training_fit_ratio": 10.0,
    "training_fit_abs_m": 0.01,
    "cross_window_position_m": 0.02,
    "cross_window_rotation_deg": 1.0,
    "coverage_context_m": 0.02,
}


# ---------------------------------------------------------------------------
# exact key sets and denominators
# ---------------------------------------------------------------------------

def regular_starts(T: int) -> list[int]:
    if T < WINDOW:
        raise ValueError(f"T={T} shorter than window {WINDOW}")
    return list(range(0, T - WINDOW + 1, STRIDE))


def inference_starts(T: int) -> list[int]:
    if T < WINDOW:
        raise ValueError(f"T={T} shorter than window {WINDOW}")
    return window_starts(T, WINDOW, STRIDE)


def tail_starts(T: int) -> list[int]:
    regular = set(regular_starts(T))
    return [s for s in inference_starts(T) if s not in regular]


def is_tail_start(T: int, start: int) -> bool:
    return start in tail_starts(T)


def teacher_key(demo: str, start: int) -> str:
    return f"teacher|{demo}|{start}"


def window_key(demo: str, model: str, start: int) -> str:
    return f"window|{demo}|{model}|{start}"


def pair_key(demo: str, model: str, t: int, i: int, j: int) -> str:
    return f"pair|{demo}|{model}|{t}|{i}|{j}"


def probe_key(demo: str, start: int, probe: str, t: int | None = None) -> str:
    suffix = "null" if t is None else f"{t:04d}"
    return f"probe|{demo}|{start}|{probe}|{suffix}"


def probe_keys_for_window(demo: str, start: int) -> list[str]:
    keys = [probe_key(demo, start, "teacher_roundtrip"),
            probe_key(demo, start, "deterministic_generation")]
    keys += [probe_key(demo, start, "diffusion_denoise", t)
             for t in DENOISE_TIMESTEPS]
    keys.append(probe_key(demo, start, "diffusion_ddim25"))
    return keys


def contributor_entries(starts: Sequence[int], T: int) -> list[list[tuple[int, int, float]]]:
    """Per global frame: (window index, local frame, positive weight)."""
    starts = [int(s) for s in starts]
    if T < 0 or any(s < 0 or s + WINDOW > T for s in starts):
        raise ValueError("window start/frame count outside the clip")
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ValueError("window starts must be unique and ordered")
    entries: list[list[tuple[int, int, float]]] = [[] for _ in range(T)]
    n = len(starts)
    for i, start in enumerate(starts):
        weights = window_weight(i, n, WINDOW, STRIDE)
        for j in range(WINDOW):
            if weights[j] > 0:
                entries[start + j].append((i, j, float(weights[j])))
    return entries


def cross_window_pair_keys(demo: str, model: str, starts: Sequence[int],
                           T: int) -> list[str]:
    keys = []
    for t, row in enumerate(contributor_entries(starts, T)):
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                keys.append(pair_key(demo, model, t, row[a][0], row[b][0]))
    return keys


def build_window_plan(t_by_demo: Mapping[str, int], train6: Sequence[str],
                      val6: Sequence[str], known_case: str) -> dict:
    """Rebuild the complete §3 set from real T values; no row inference."""
    demos = list(train6) + list(val6) + [known_case]
    if len(demos) != 13 or len(set(demos)) != 13:
        raise ValueError("the registered denominator must be 13 unique demos")
    groups = {}
    per_demo = {}
    teacher_keys, window_keys, pair_keys_all = [], [], []
    probes = []
    totals = {"regular": 0, "inference": 0, "tail": 0,
              "groups": {"train6": {"regular": 0, "inference": 0, "tail": 0},
                         "val6": {"regular": 0, "inference": 0, "tail": 0},
                         "known_case": {"regular": 0, "inference": 0, "tail": 0}}}
    for demo in demos:
        group = ("train6" if demo in train6 else
                 "val6" if demo in val6 else "known_case")
        if demo not in t_by_demo:
            raise ValueError(f"missing T for {demo}")
        T = int(t_by_demo[demo])
        regular, inference, tail = regular_starts(T), inference_starts(T), tail_starts(T)
        if len(tail) != 1 or tail[0] != T - WINDOW:
            raise ValueError(f"{demo}: expected exactly the inference tail start T-60")
        if regular != list(range(0, T - WINDOW + 1, STRIDE)):
            raise ValueError(f"{demo}: regular starts drifted")
        per_demo[demo] = {"demo": demo, "group": group, "T": T,
                          "regular": regular, "inference": inference,
                          "tail": tail, "window": WINDOW, "stride": STRIDE}
        groups[demo] = group
        for start in inference:
            teacher_keys.append(teacher_key(demo, start))
        for model in MODELS:
            for start in inference:
                window_keys.append(window_key(demo, model, start))
            pair_keys_all.extend(cross_window_pair_keys(demo, model, inference, T))
        for start in inference:
            probes.extend(probe_keys_for_window(demo, start))
        totals["regular"] += len(regular)
        totals["inference"] += len(inference)
        totals["tail"] += len(tail)
        for key, value in (("regular", len(regular)), ("inference", len(inference)),
                           ("tail", len(tail))):
            totals["groups"][group][key] += value
    if totals["regular"] != 52 or totals["inference"] != 65 or totals["tail"] != 13:
        raise ValueError(f"registered window totals changed: {totals}")
    if len(teacher_keys) != 65 or len(window_keys) != 130 or len(probes) != 520:
        raise ValueError("registered key counts changed")
    if len(set(window_keys)) != len(window_keys) or len(set(probes)) != len(probes):
        raise ValueError("duplicate registered keys")
    if len(pair_keys_all) != len(set(pair_keys_all)):
        raise ValueError("duplicate cross-window pair keys")
    return {
        "train6": list(train6), "val6": list(val6), "known_case": known_case,
        "demos": demos, "groups": groups, "per_demo": per_demo,
        "totals": totals, "teacher_keys": teacher_keys,
        "window_keys": window_keys, "probe_keys": probes,
        "pair_keys": pair_keys_all,
        "counts": {"teacher_references": 65, "model_windows": 130,
                   "regular_windows": 52, "tail_windows": 13,
                   "inference_windows": 65, "probe_units": 520,
                   "cross_window_pairs": len(pair_keys_all),
                   "learned_probe_units": 455, "oracle_probe_units": 65},
    }


def expected_ratio_counts() -> dict:
    """Fixed P/F call ratios used to audit actual invocation counts."""
    return {
        "det_sessions": 13, "denoise_batches": 65, "ddim_sessions": 13,
        "high_level_calls": 91, "network_forwards": 403,
        "denoise_rows": 325, "generation_rows": 130, "oracle_rows": 65,
        "replay_demos": ["sub10_largebox_063", "sub10_largebox_100",
                         "sub16_largebox_048"],
        "replay_windows": 11, "replay_units": 88,
        "replay_learned_units": 77, "replay_oracle_units": 11,
        "replay_high_level_calls": 21, "replay_network_forwards": 93,
    }


# ---------------------------------------------------------------------------
# rotation / decode helpers
# ---------------------------------------------------------------------------

def _as_float(a, name: str) -> np.ndarray:
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating) and not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} is not numeric")
    a = a.astype(np.float64, copy=False)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} is not finite")
    return a


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SO(3) relative angle between wxyz quaternions [...,4] in degrees."""
    a = _as_float(a, "quat_a")
    b = _as_float(b, "quat_b")
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError("quaternion shape mismatch")
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if (na < 1e-12).any() or (nb < 1e-12).any():
        raise ValueError("zero-norm quaternion")
    ra = Rot.from_quat(wxyz_to_xyzw(a / na[..., None]))
    rb = Rot.from_quat(wxyz_to_xyzw(b / nb[..., None]))
    return np.rad2deg((ra.inv() * rb).magnitude())


def quat_angle_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.deg2rad(quat_angle_deg(a, b))


def lift_clip_to_world(pos: np.ndarray, quat: np.ndarray,
                       world_from_clip: np.ndarray) -> np.ndarray:
    """clip [F,3]/[F,4] wxyz -> world root [F,7] (production lift)."""
    pos = _as_float(pos, "pos")
    quat = _as_float(quat, "quat")
    wfc = _as_float(world_from_clip, "world_from_clip")
    if pos.ndim != 2 or pos.shape[1] != 3 or quat.shape != (pos.shape[0], 4):
        raise ValueError("clip state shape mismatch")
    pos_world = pos @ wfc[:3, :3].T + wfc[:3, 3]
    q_world = (Rot.from_matrix(wfc[:3, :3])
               * Rot.from_quat(wxyz_to_xyzw(quat))).as_quat()
    return np.concatenate([pos_world, xyzw_to_wxyz(q_world)], axis=1)


def lift_clip_windows_to_world(pos: np.ndarray, quat: np.ndarray,
                               world_from_clip: np.ndarray) -> np.ndarray:
    """[N,F,3]/[N,F,4] clip windows -> [N,F,7] world roots (per-window lift)."""
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[-1] != 3 or quat.shape != (*pos.shape[:2], 4):
        raise ValueError("clip window matrix shape mismatch")
    return np.stack([lift_clip_to_world(pos[i], quat[i], world_from_clip)
                     for i in range(pos.shape[0])])


def world_to_clip_state(root: np.ndarray, world_from_clip: np.ndarray) -> np.ndarray:
    """world root [F,7] -> clip [F,7] (inverse lift)."""
    root = _as_float(root, "root")
    wfc = _as_float(world_from_clip, "world_from_clip")
    if root.ndim != 2 or root.shape[1] != 7:
        raise ValueError("root shape mismatch")
    pos = (root[:, :3] - wfc[:3, 3]) @ wfc[:3, :3]
    quat = (Rot.from_matrix(wfc[:3, :3].T)
            * Rot.from_quat(wxyz_to_xyzw(root[:, 3:7]))).as_quat()
    return np.concatenate([pos, xyzw_to_wxyz(quat)], axis=1)


def validate_normalized(x0, *, frames: int | None = None) -> list[str]:
    problems = []
    a = np.asarray(x0)
    if a.ndim != 2 or a.shape[1] != 38:
        return [f"normalized_shape:{a.shape}"]
    if frames is not None and a.shape[0] != frames:
        problems.append(f"normalized_frames:{a.shape[0]}!={frames}")
    if not np.issubdtype(a.dtype, np.floating):
        problems.append("normalized_not_floating")
    if not np.isfinite(a).all():
        problems.append("normalized_nonfinite")
        return problems
    a64 = a.astype(np.float64, copy=False)
    first = np.linalg.norm(a64[:, 3:6], axis=1)
    second = np.linalg.norm(a64[:, 6:9], axis=1)
    if (first <= 1e-6).any() or (second <= 1e-6).any():
        problems.append("normalized_degenerate_rotation6d")
    return problems


def validate_world_root(root, *, frames: int | None = None) -> list[str]:
    problems = []
    a = np.asarray(root)
    if a.ndim != 2 or a.shape[1] != 7:
        return [f"root_shape:{a.shape}"]
    if frames is not None and a.shape[0] != frames:
        problems.append(f"root_frames:{a.shape[0]}!={frames}")
    if not np.isfinite(a).all():
        problems.append("root_nonfinite")
        return problems
    norm = np.linalg.norm(a[:, 3:7].astype(np.float64), axis=1)
    if (norm < 1e-12).any():
        problems.append("root_zero_norm_quaternion")
    elif (np.abs(norm - 1.0) > QUAT_NORM_TOL).any():
        problems.append("root_quaternion_norm")
    return problems


def validate_joints(q, *, frames: int | None = None) -> list[str]:
    a = np.asarray(q)
    problems = []
    if a.ndim != 2 or a.shape[1] != 29:
        return [f"joints_shape:{a.shape}"]
    if frames is not None and a.shape[0] != frames:
        problems.append(f"joints_frames:{a.shape[0]}!={frames}")
    if not np.isfinite(a).all():
        problems.append("joints_nonfinite")
    return problems


def validate_decoded(pos, quat, q, *, frames: int) -> list[str]:
    problems = []
    for name, value, width in (("decoded_pos", pos, 3), ("decoded_quat", quat, 4),
                               ("decoded_q", q, 29)):
        a = np.asarray(value)
        if a.ndim != 2 or a.shape != (frames, width):
            problems.append(f"{name}_shape:{a.shape}")
        elif not np.isfinite(a).all():
            problems.append(f"{name}_nonfinite")
        elif not np.issubdtype(a.dtype, np.floating):
            problems.append(f"{name}_not_floating")
    if not problems:
        norm = np.linalg.norm(np.asarray(quat, dtype=np.float64), axis=1)
        if (norm < 1e-12).any():
            problems.append("decoded_zero_norm_quaternion")
        elif (np.abs(norm - 1.0) > QUAT_NORM_TOL).any():
            problems.append("decoded_quaternion_norm")
    return problems


def decode_production(x0: np.ndarray, tokenizer, *, device="cpu"):
    """The generator's exact FP32 decode formula (no network, no clipping of root)."""
    a = np.asarray(x0, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 38:
        raise ValueError("x0 must be [F,38]")
    dev = torch.device(device)
    x = torch.as_tensor(a, dtype=torch.float32, device=dev)
    mean, std = tokenizer.root_stats.to_torch(dev)
    lo = torch.tensor(tokenizer.lo, dtype=torch.float32, device=dev)
    hi = torch.tensor(tokenizer.hi, dtype=torch.float32, device=dev)
    if not all(bool(torch.isfinite(v).all()) for v in (mean, std, lo, hi)):
        raise ValueError("non-finite decoder statistics")
    pos = x[..., 0:3] * std + mean
    quat = mat_to_quat_torch(rot6d_to_mat_torch(x[..., 3:9]))
    q = lo + (x[..., 9:38].clamp(-1, 1) + 1) * (hi - lo) / 2
    return (pos.cpu().numpy().astype(np.float32),
            quat.cpu().numpy().astype(np.float32),
            q.cpu().numpy().astype(np.float32))


def encode_teacher(clip, tokenizer) -> np.ndarray:
    x0 = tokenizer.encode(clip.root, clip.q, clip.condition.world_from_clip)
    return np.asarray(x0, dtype=np.float32)


def teacher_windows(x0: np.ndarray, starts: Sequence[int]) -> np.ndarray:
    a = np.asarray(x0)
    if a.ndim != 2 or a.shape[1] != 38:
        raise ValueError("teacher x0 shape")
    out = np.stack([a[int(s):int(s) + WINDOW] for s in starts])
    if out.shape != (len(starts), WINDOW, 38):
        raise ValueError("teacher window shape")
    return out


# ---------------------------------------------------------------------------
# losses and physical facts
# ---------------------------------------------------------------------------

def make_loss_assembler(tokenizer, weights: Mapping[str, float], device):
    from ireset.train.losses import LossAssembler, LossWeights

    if float(weights.get("w_contact", 0.0)) != 0.0:
        raise ValueError("the diagnostic requires w_contact=0 (contact branch silent)")
    w = LossWeights(**{k: float(v) for k, v in weights.items()})
    return LossAssembler(tokenizer, w, device)


def normalized_loss_groups(pred_x0: np.ndarray, target_x0: np.ndarray,
                           assembler) -> dict:
    p = torch.as_tensor(np.asarray(pred_x0), dtype=torch.float32,
                        device=assembler.device)
    t = torch.as_tensor(np.asarray(target_x0), dtype=torch.float32,
                        device=assembler.device)
    if tuple(p.shape) != (WINDOW, 38) or tuple(t.shape) != (WINDOW, 38):
        raise ValueError("loss window shape")
    _, parts = assembler(p[None], {"x0": t[None]})
    out = {k: float(v) for k, v in parts.items()}
    if "contact_m2" in out:
        raise ValueError("contact branch executed with w_contact=0")
    for key in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2", "total"):
        if key not in out or not math.isfinite(out[key]):
            raise ValueError(f"loss group missing/nonfinite: {key}")
    return out


def _per_joint_stats(error: np.ndarray) -> dict:
    e = np.asarray(error, dtype=np.float64)
    return {"mae_rad": float(np.abs(e).mean()),
            "rmse_rad": float(np.sqrt((e ** 2).mean())),
            "per_joint_mae_rad": np.abs(e).mean(0).tolist(),
            "per_joint_rmse_rad": np.sqrt((e ** 2).mean(0)).tolist()}


def _first_exceed(values: np.ndarray, limit: float, *, defined: np.ndarray | None = None):
    a = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(a)
    if defined is not None:
        mask &= np.asarray(defined, dtype=bool)
    hit = np.flatnonzero(mask & (a > limit))
    return int(hit[0]) if len(hit) else None


def physical_window_facts(world_root_pred: np.ndarray, joints_pred: np.ndarray,
                          world_root_teacher: np.ndarray, joints_teacher: np.ndarray,
                          robot_kpts: np.ndarray, teacher_kpts: np.ndarray,
                          frame_dt: float, task_tol_m: float) -> dict:
    pred = _as_float(world_root_pred, "world_root_pred")
    truth = _as_float(world_root_teacher, "world_root_teacher")
    qp = _as_float(joints_pred, "joints_pred")
    qt = _as_float(joints_teacher, "joints_teacher")
    kp = _as_float(robot_kpts, "robot_kpts")
    kt = _as_float(teacher_kpts, "teacher_kpts")
    F = pred.shape[0]
    if (pred.shape != (F, 7) or truth.shape != (F, 7)
            or qp.shape != (F, 29) or qt.shape != (F, 29)
            or kp.shape != (F, 15, 3) or kt.shape != (F, 15, 3)):
        raise ValueError("physical window input shapes")
    dt = float(frame_dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("frame_dt invalid")

    root_err = np.linalg.norm(pred[:, :3] - truth[:, :3], axis=1)
    rot_deg = quat_angle_deg(pred[:, 3:7], truth[:, 3:7])
    joint_err = qp - qt
    kpt_err = np.linalg.norm(kp - kt, axis=-1).mean(1)
    from ireset.embodied.g1_spec import KPT_TO_SMPLH

    kpt_names = list(KPT_TO_SMPLH)
    hand = {}
    for side in ("l", "r"):
        idx = kpt_names.index(f"kpt_{side}_wrist")
        hand[side] = np.linalg.norm(kp[:, idx] - kt[:, idx], axis=1)

    root_speed = np.full(F, np.nan)
    root_ang_speed = np.full(F, np.nan)
    joint_speed = np.full((F, 29), np.nan)
    if F > 1:
        root_speed[1:] = np.linalg.norm(np.diff(pred[:, :3], axis=0), axis=1) / dt
        root_ang_speed[1:] = quat_angle_rad(pred[:-1, 3:7], pred[1:, 3:7]) / dt
        joint_speed[1:] = np.abs(np.diff(qp, axis=0)) / dt
    joint_speed_max = np.full(F, np.nan)
    joint_speed_max[1:] = joint_speed[1:].max(1)

    teacher_root_speed = np.full(F, np.nan)
    teacher_root_ang_speed = np.full(F, np.nan)
    if F > 1:
        teacher_root_speed[1:] = np.linalg.norm(np.diff(truth[:, :3], axis=0), axis=1) / dt
        teacher_root_ang_speed[1:] = quat_angle_rad(truth[:-1, 3:7], truth[1:, 3:7]) / dt

    facts = {
        "root_position": {"mean_m": float(root_err.mean()),
                          "rmse_m": float(np.sqrt((root_err ** 2).mean())),
                          "p95_m": float(np.percentile(root_err, 95)),
                          "max_m": float(root_err.max()),
                          "max_frame": int(np.argmax(root_err))},
        "root_rotation": {"mean_deg": float(rot_deg.mean()),
                          "p95_deg": float(np.percentile(rot_deg, 95)),
                          "max_deg": float(rot_deg.max()),
                          "max_frame": int(np.argmax(rot_deg))},
        "joints": {**_per_joint_stats(joint_err),
                   "max_abs_rad": float(np.abs(joint_err).max()),
                   "max_frame": int(np.argmax(np.abs(joint_err).max(1)))},
        "fk": {"mpjpe_m": float(kpt_err.mean()),
               "p95_m": float(np.percentile(kpt_err, 95)),
               "max_m": float(kpt_err.max()),
               "max_frame": int(np.argmax(kpt_err))},
        "hand_fk_m": {s: float(v.mean()) for s, v in hand.items()},
        "hand_fk_max_m": {s: float(v.max()) for s, v in hand.items()},
        "task_gate": {"tol_m": float(task_tol_m),
                      "first_exceedance_frame": _first_exceed(kpt_err, task_tol_m),
                      "exceedance_frames": int((kpt_err > task_tol_m).sum())},
        "temporal": {
            "root_speed_max_mps": float(np.nanmax(root_speed)),
            "root_speed_max_frame": int(np.nanargmax(root_speed)),
            "root_angular_speed_max_radps": float(np.nanmax(root_ang_speed)),
            "root_angular_speed_max_frame": int(np.nanargmax(root_ang_speed)),
            "joint_speed_max_radps": float(np.nanmax(joint_speed_max)),
            "joint_speed_max_frame": int(np.nanargmax(joint_speed_max)),
            "teacher_root_speed_max_mps": float(np.nanmax(teacher_root_speed)),
            "teacher_root_angular_speed_max_radps": float(np.nanmax(teacher_root_ang_speed)),
            "speed_first_exceedance_frame": {
                "root_speed": _first_exceed(root_speed, ROOT_SPEED_LIMIT_MPS),
                "root_angular_speed": _first_exceed(root_ang_speed,
                                                    ANGULAR_SPEED_LIMIT_RADPS),
                "joint_speed": _first_exceed(joint_speed_max,
                                             JOINT_SPEED_LIMIT_RADPS),
            },
        },
        "curves": {"root_pos_error_m": root_err,
                   "root_rot_error_deg": rot_deg,
                   "joint_abs_error_rad": np.abs(joint_err),
                   "joint_signed_error_rad": joint_err,
                   "kpt_error_m": kpt_err,
                   "hand_error_l_m": hand["l"], "hand_error_r_m": hand["r"],
                   "root_speed_mps": root_speed,
                   "root_angular_speed_radps": root_ang_speed,
                   "joint_speed_radps": joint_speed,
                   "teacher_root_speed_mps": teacher_root_speed,
                   "teacher_root_angular_speed_radps": teacher_root_ang_speed},
    }
    return facts


# ---------------------------------------------------------------------------
# frame-aligned geometry (locked MuJoCo scene / teacher measurement protocol)
# ---------------------------------------------------------------------------

def window_geometry_series(replayer, scene: np.ndarray, human: np.ndarray,
                           target_box: np.ndarray, *, deadline: float | None = None) -> dict:
    """Per-window frame geometry on the locked validator model.

    Mirrors ``w1_de_frame_series`` for penetration / hand gaps / keypoints and
    adds per-family worst pairs and sole-to-ground signed gaps. It is a
    keyframe diagnostic, never an adaptive whole-segment acceptance run.
    """
    scene = np.asarray(scene, dtype=np.float64)
    human = np.asarray(human, dtype=np.float64)
    target_box = np.asarray(target_box, dtype=np.float64)
    F = len(scene)
    if (scene.ndim != 2 or scene.shape[1] != 43 or F == 0
            or human.shape != (F, 52, 3) or target_box.shape != (F, 7)
            or not all(np.isfinite(a).all() for a in (scene, human, target_box))
            or not np.array_equal(scene[:, 36:43], target_box)
            or (np.linalg.norm(scene[:, 3:7], axis=1) < 1e-8).any()
            or (np.linalg.norm(target_box[:, 3:7], axis=1) < 1e-8).any()):
        raise ValueError("unsafe scene/reference passed to window geometry")
    r = replayer
    validator = getattr(r, "validator", None)
    if validator is None or not hasattr(validator, "sole_geoms"):
        raise ValueError("replayer does not expose the locked validator geometry")
    pen = np.zeros(F)
    families = np.zeros((F, 3))
    worst_pairs = np.full((F, 3, 2), -1, dtype=np.int32)
    box_ground = np.zeros(F)
    hand_gap = {"l": np.zeros(F), "r": np.zeros(F)}
    foot_gap = {"l": np.zeros(F), "r": np.zeros(F)}
    robot_kpts = np.zeros((F, len(r.kpt_ids), 3))
    kpt_err = np.zeros(F)
    kpt_err_box = np.zeros(F)
    from ireset.geometry.distance_adapter import world_to_box_geom

    active = [(kind, a, b) for kind in GEOMETRY_FAMILIES
              for (a, b) in r.pairs_by_kind.get(kind, [])]
    for t in range(F):
        if deadline is not None and t % 10 == 0 and _deadline_hit(deadline):
            raise TimeoutError("window_geometry_deadline")
        r._forward(scene, t)
        for k, kind in enumerate(GEOMETRY_FAMILIES):
            worst = -math.inf
            pair = (-1, -1)
            for kk, a, b in active:
                if kk != kind:
                    continue
                depth = -r._dist(a, b)
                if depth > worst:
                    worst, pair = depth, (a, b)
            families[t, k] = worst if math.isfinite(worst) else 0.0
            worst_pairs[t, k] = pair
        pen[t] = float(families[t].max())
        box_ground[t] = -r._dist(r.box_geom, r.ground_geom)
        for side in ("l", "r"):
            hand_gap[side][t] = min(r._dist(g, r.box_geom)
                                    for g in r._hand_geoms[side])
            foot_gap[side][t] = min(r._dist(g, r.ground_geom)
                                    for g in validator.sole_geoms[side])
        kp = r.data.site_xpos[r.kpt_ids].copy()
        robot_kpts[t] = kp
        ref = human[t][r.ref_kpt_idx]
        kpt_err[t] = float(np.linalg.norm(kp - ref, axis=-1).mean())
        kb = world_to_box_geom(kp[None], target_box[t:t + 1])[0]
        rb = world_to_box_geom(ref[None], target_box[t:t + 1])[0]
        kpt_err_box[t] = float(np.linalg.norm(kb - rb, axis=-1).mean())
    arrays = (pen, families, box_ground, hand_gap["l"], hand_gap["r"],
              foot_gap["l"], foot_gap["r"], robot_kpts, kpt_err, kpt_err_box)
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("non-finite window geometry")
    return {"penetration": pen, "penetration_families": families,
            "worst_geom_pairs": worst_pairs, "box_ground": box_ground,
            "hand_gap": hand_gap, "foot_gap": foot_gap,
            "robot_kpts": robot_kpts, "kpt_err_w": kpt_err,
            "kpt_err_b": kpt_err_box}


def _deadline_hit(deadline: float) -> bool:
    import time

    return time.perf_counter() >= deadline


def geometry_window_facts(model_series: dict, teacher_series: dict,
                          human: np.ndarray, teacher_kpts: np.ndarray,
                          start: int, cfg) -> dict:
    """Geometry facts for one window, including teacher-inherited failures."""
    F = len(model_series["penetration"])
    if len(teacher_series["penetration"]) != F:
        raise ValueError("teacher/model geometry length mismatch")
    model_pen = np.maximum(np.asarray(model_series["penetration"], dtype=np.float64), 0.0)
    teacher_pen = np.maximum(np.asarray(teacher_series["penetration"], dtype=np.float64), 0.0)
    m_flag = model_pen > KEYFRAME_PENETRATION_M
    t_flag = teacher_pen > KEYFRAME_PENETRATION_M
    new = m_flag & ~t_flag
    shared = m_flag & t_flag
    same_pair = np.zeros(F, dtype=bool)
    for k in range(3):
        same_pair |= (model_series["worst_geom_pairs"][:, k]
                      == teacher_series["worst_geom_pairs"][:, k]).all(1)
    per_family = {}
    for k, kind in enumerate(GEOMETRY_FAMILIES):
        mp = np.maximum(np.asarray(model_series["penetration_families"][:, k],
                                   dtype=np.float64), 0.0)
        tp = np.maximum(np.asarray(teacher_series["penetration_families"][:, k],
                                   dtype=np.float64), 0.0)
        mf, tf = mp > KEYFRAME_PENETRATION_M, tp > KEYFRAME_PENETRATION_M
        per_family[kind] = {
            "model_max_m": float(mp.max()), "teacher_max_m": float(tp.max()),
            "model_frames": int(mf.sum()), "teacher_frames": int(tf.sum()),
            "shared_frames": int((mf & tf).sum()),
            "new_frames": int((mf & ~tf).sum()),
            "same_worst_pair_frames": int(((mf & tf) & same_pair).sum()),
            "first_model_frame": int(np.flatnonzero(mf)[0]) if mf.any() else None,
        }
    hand = {}
    foot = {}
    for side in ("l", "r"):
        mg = np.asarray(model_series["hand_gap"][side], dtype=np.float64)
        tg = np.asarray(teacher_series["hand_gap"][side], dtype=np.float64)
        hand[side] = {"min_m": float(mg.min()), "teacher_min_m": float(tg.min()),
                      "frames_below_gate": int((mg < cfg.hand_gap_m).sum()),
                      "penetration_frames": int((mg < -KEYFRAME_PENETRATION_M).sum()),
                      "max_penetration_m": float(max(0.0, -mg.min()))}
        mf_gap = np.asarray(model_series["foot_gap"][side], dtype=np.float64)
        tf_gap = np.asarray(teacher_series["foot_gap"][side], dtype=np.float64)
        foot[side] = {"min_m": float(mf_gap.min()),
                      "teacher_min_m": float(tf_gap.min()),
                      "frames_below_gate": int((mf_gap < cfg.foot_ground_gap_m).sum()),
                      "penetration_frames": int((mf_gap < -KEYFRAME_PENETRATION_M).sum()),
                      "max_penetration_m": float(max(0.0, -mf_gap.min()))}
    if teacher_kpts.shape != (F, 15, 3):
        raise ValueError("teacher keypoint window shape")
    hand_err = {}
    from ireset.embodied.g1_spec import KPT_TO_SMPLH

    names = list(KPT_TO_SMPLH)
    for side in ("l", "r"):
        idx = names.index(f"kpt_{side}_wrist")
        hand_err[side] = float(np.linalg.norm(
            model_series["robot_kpts"][:, idx] - teacher_kpts[:, idx], axis=1).mean())
    facts = {
        "penetration": {
            "max_m": float(model_pen.max()),
            "max_frame": int(np.argmax(model_pen)),
            "violation_frames": int(m_flag.sum()),
            "violation_ratio": float(m_flag.mean()),
            "teacher_max_m": float(teacher_pen.max()),
            "teacher_violation_frames": int(t_flag.sum()),
            "new_violation_frames": int(new.sum()),
            "shared_violation_frames": int(shared.sum()),
            "same_worst_pair_frames": int((shared & same_pair).sum()),
            "protocol": "keyframe signed geom distance over robot_ground/robot_box/self",
        },
        "families": per_family,
        "hand_gap": hand,
        "foot_gap": foot,
        "box_ground": {"max_m": float(np.maximum(
            np.asarray(model_series["box_ground"], dtype=np.float64), 0.0).max()),
            "excluded_from_strict": True},
        "hand_fk_m": hand_err,
        "teacher_kpt_error_w_m": float(np.asarray(teacher_series["kpt_err_w"]).mean()),
        "model_kpt_error_w_m": float(np.asarray(model_series["kpt_err_w"]).mean()),
        "model_kpt_error_b_m": float(np.asarray(model_series["kpt_err_b"]).mean()),
        "global_frames": {"start": int(start), "first_model_violation":
                          (int(start + np.flatnonzero(m_flag)[0]) if m_flag.any() else None),
                          "first_new_violation":
                          (int(start + np.flatnonzero(new)[0]) if new.any() else None),
                          "max_frame": int(start + int(np.argmax(model_pen))),
                          "max_teacher_frame": int(start + int(np.argmax(teacher_pen)))},
    }
    return facts


# ---------------------------------------------------------------------------
# reference / bins / curve contract
# ---------------------------------------------------------------------------

def reference_window_facts(schedule, start: int, frames: int = WINDOW) -> dict:
    from ireset.eval.teacher_comparison import _frames_of, _mask_from_ranges

    T_total = max((iv.end for iv in schedule.intervals), default=0)
    T_total = max(T_total, start + frames)
    out = {"ref_grasp_frames": {}, "ref_support_frames": {}}
    for side, full in (("l", "left"), ("r", "right")):
        hand = _mask_from_ranges(_frames_of(schedule, f"{full}_hand", "object"),
                                 T_total)[start:start + frames]
        foot = _mask_from_ranges(_frames_of(schedule, f"{full}_foot", "ground"),
                                 T_total)[start:start + frames]
        out["ref_grasp_frames"][side] = int(hand.sum())
        out["ref_support_frames"][side] = int(foot.sum())
    both = (_mask_from_ranges(_frames_of(schedule, "left_hand", "object"), T_total)
            & _mask_from_ranges(_frames_of(schedule, "right_hand", "object"), T_total))[start:start + frames]
    first = int(np.flatnonzero(both)[0]) if both.any() else None
    out.update({"both_reference_frames": int(both.sum()),
                "first_both_overlap_local_frame": first,
                "frames_before_first_overlap": (first if first is not None else frames),
                "frames_after_first_overlap": (frames - (first + 1) if first is not None else 0)})
    return out


def bin_facts(curves: Mapping[str, np.ndarray]) -> dict:
    """Fixed local position bins [0,10), [10,50), [50,60)."""
    def stats(a, reducer):
        x = np.asarray(a, dtype=np.float64)
        valid = x[np.isfinite(x)]
        return None if valid.size == 0 else float(reducer(valid))

    out = {}
    for lo, hi, name in LOCAL_BINS:
        sl = slice(lo, hi)
        root = np.asarray(curves["root_pos_error_m"])[sl]
        joint = np.abs(np.asarray(curves["joint_abs_error_rad"])[sl])
        kpt = np.asarray(curves["kpt_error_m"])[sl]
        pen = np.maximum(np.asarray(curves["penetration"])[sl], 0.0)
        hands = [np.asarray(curves[f"hand_gap_{s}"])[sl] for s in ("l", "r")]
        feet = [np.asarray(curves[f"foot_gap_{s}"])[sl] for s in ("l", "r")]
        out[name] = {
            "frames": int(hi - lo),
            "root_rmse_m": float(np.sqrt(np.mean(root ** 2))),
            "joint_mae_rad": float(joint.mean()),
            "kpt_mean_m": float(kpt.mean()),
            "penetration_max_m": float(pen.max()),
            "hand_gap_min_m": float(min(np.min(h) for h in hands)),
            "foot_gap_min_m": float(min(np.min(f) for f in feet)),
        }
    return out


CURVE_CONTRACT = {
    "local_frame": "int", "global_frame": "int",
    "root_pos_error_m": "finite", "root_rot_error_deg": "finite",
    "joint_abs_error_rad": "finite", "joint_signed_error_rad": "finite",
    "kpt_error_m": "finite", "hand_error_l_m": "finite",
    "hand_error_r_m": "finite", "penetration": "finite",
    "penetration_families": "finite", "worst_geom_pairs": "int",
    "hand_gap_l": "finite", "hand_gap_r": "finite",
    "foot_gap_l": "finite", "foot_gap_r": "finite",
    "teacher_penetration": "finite", "teacher_hand_gap_l": "finite",
    "teacher_hand_gap_r": "finite", "teacher_foot_gap_l": "finite",
    "teacher_foot_gap_r": "finite", "new_penetration": "mask",
    "shared_penetration": "mask", "root_speed_mps": "nan_first",
    "root_angular_speed_radps": "nan_first", "joint_speed_radps": "nan_first",
    "teacher_root_speed_mps": "nan_first",
    "teacher_root_angular_speed_radps": "nan_first",
    "velocity_defined": "mask",
    "normalized_prediction": "finite", "decoded_pos": "finite",
    "decoded_quat": "finite", "decoded_q": "finite",
    "teacher_x0": "finite",
}


def verify_window_curves(arrays: Mapping[str, np.ndarray]) -> list[str]:
    problems = []
    keys = set(arrays)
    for missing in sorted(set(CURVE_CONTRACT) - keys):
        problems.append(f"curve_missing:{missing}")
    for extra in sorted(keys - set(CURVE_CONTRACT)):
        problems.append(f"curve_unexpected:{extra}")
    if problems:
        return problems
    for name, kind in CURVE_CONTRACT.items():
        a = np.asarray(arrays[name])
        width = 29 if name in ("joint_abs_error_rad", "joint_signed_error_rad",
                               "joint_speed_radps") else None
        shapes = {(WINDOW,), (WINDOW, 3), (WINDOW, 3, 2), (WINDOW, 38), (WINDOW, 3),
                  (WINDOW, 4), (WINDOW, 29)}
        if width is not None:
            expected = (WINDOW, 29)
            if a.shape != expected:
                problems.append(f"curve_shape:{name}:{a.shape}")
                continue
        elif a.shape not in shapes or a.shape[0] != WINDOW:
            problems.append(f"curve_shape:{name}:{a.shape}")
            continue
        if kind == "int":
            if not np.issubdtype(a.dtype, np.integer):
                problems.append(f"curve_not_integer:{name}")
        elif kind == "mask":
            if a.dtype != np.bool_:
                problems.append(f"curve_not_mask:{name}")
        elif kind == "nan_first":
            if np.isinf(a).any():
                problems.append(f"curve_infinite:{name}")
            if a.ndim == 1:
                if not np.isnan(a[0]):
                    problems.append(f"curve_first_frame_not_undefined:{name}")
                if not np.isfinite(a[1:]).all():
                    problems.append(f"curve_nonfinite:{name}")
            else:
                if not np.isnan(a[0]).all():
                    problems.append(f"curve_first_frame_not_undefined:{name}")
                if not np.isfinite(a[1:]).all():
                    problems.append(f"curve_nonfinite:{name}")
        else:
            if not np.isfinite(a).all():
                problems.append(f"curve_nonfinite:{name}")
    return problems


# ---------------------------------------------------------------------------
# window measurement
# ---------------------------------------------------------------------------

def measure_window(*, demo: str, model: str, start: int, T: int, kind: str,
                   group: str, pred_x0: np.ndarray, teacher_x0: np.ndarray,
                   decoded: tuple[np.ndarray, np.ndarray, np.ndarray],
                   world_from_clip: np.ndarray,
                   teacher_root: np.ndarray, teacher_joints: np.ndarray,
                   teacher_kpts: np.ndarray,
                   model_geometry: Mapping[str, np.ndarray],
                   teacher_geometry: Mapping[str, np.ndarray],
                   tokenizer, assembler, schedule,
                   frame_dt: float, task_tol_m: float, hand_gap_m: float,
                   saved_world_root: np.ndarray | None = None,
                   decode_device: str = "cpu") -> dict:
    F = WINDOW
    problems = []
    problems += [f"pred:{p}" for p in validate_normalized(pred_x0, frames=F)]
    problems += [f"teacher:{p}" for p in validate_normalized(teacher_x0, frames=F)]
    problems += validate_decoded(*decoded, frames=F)
    if problems:
        return {"status": "measurement_incomplete", "measurement_gaps": problems,
                "metrics": None, "curves": None}
    pred_x0 = np.asarray(pred_x0, dtype=np.float32)
    teacher_x0 = np.asarray(teacher_x0, dtype=np.float32)
    pos, quat, q = (np.asarray(v, dtype=np.float32) for v in decoded)
    wfc = np.asarray(world_from_clip, dtype=np.float64)
    world_root = lift_clip_to_world(pos.astype(np.float64), quat.astype(np.float64), wfc)
    root_problems = validate_world_root(world_root, frames=F)
    if saved_world_root is not None:
        root_problems += [f"saved_root:{p}" for p in
                          validate_world_root(saved_world_root, frames=F)]
    problems += root_problems
    if problems:
        return {"status": "measurement_incomplete", "measurement_gaps": problems,
                "metrics": None, "curves": None}
    teacher_root_w = np.asarray(teacher_root, dtype=np.float64)[start:start + F]
    teacher_q_w = np.asarray(teacher_joints, dtype=np.float64)[start:start + F]
    teacher_k = np.asarray(teacher_kpts, dtype=np.float64)[start:start + F]
    if teacher_root_w.shape != (F, 7) or teacher_q_w.shape != (F, 29):
        raise ValueError("teacher window slice shape")
    if teacher_k.shape != (F, 15, 3):
        raise ValueError("teacher keypoint slice shape")

    rebuild = decode_production(pred_x0, tokenizer, device=decode_device)
    decode_max = {name: float(np.abs(np.asarray(rebuild[i], dtype=np.float64)
                                     - np.asarray(decoded[i], dtype=np.float64)).max())
                  for i, name in enumerate(("pos", "quat", "q"))}
    decode_bitwise = {name: bool(np.array_equal(np.asarray(rebuild[i]),
                                                np.asarray(decoded[i])))
                      for i, name in enumerate(("pos", "quat", "q"))}
    # physical rotation equivalence is reported independently of the storage.
    rot_equiv = float(quat_angle_deg(world_root[:, 3:7],
                                     lift_clip_to_world(rebuild[0], rebuild[1], wfc)[:, 3:7]).max())

    loss = normalized_loss_groups(pred_x0, teacher_x0, assembler)
    physical = physical_window_facts(
        world_root, q.astype(np.float64), teacher_root_w, teacher_q_w,
        model_geometry["robot_kpts"], teacher_k, frame_dt, task_tol_m)
    geometry = geometry_window_facts(model_geometry, teacher_geometry,
                                     np.zeros((F, 52, 3)), teacher_k, start,
                                     _cfg_like(hand_gap_m))
    reference = reference_window_facts(schedule, start, F)
    curves = {
        "local_frame": np.arange(F), "global_frame": np.arange(start, start + F),
        **physical.pop("curves"),
        "penetration": np.asarray(model_geometry["penetration"], dtype=np.float64),
        "penetration_families": np.asarray(model_geometry["penetration_families"], dtype=np.float64),
        "worst_geom_pairs": np.asarray(model_geometry["worst_geom_pairs"], dtype=np.int32),
        "hand_gap_l": np.asarray(model_geometry["hand_gap"]["l"], dtype=np.float64),
        "hand_gap_r": np.asarray(model_geometry["hand_gap"]["r"], dtype=np.float64),
        "foot_gap_l": np.asarray(model_geometry["foot_gap"]["l"], dtype=np.float64),
        "foot_gap_r": np.asarray(model_geometry["foot_gap"]["r"], dtype=np.float64),
        "teacher_penetration": np.asarray(teacher_geometry["penetration"], dtype=np.float64),
        "teacher_hand_gap_l": np.asarray(teacher_geometry["hand_gap"]["l"], dtype=np.float64),
        "teacher_hand_gap_r": np.asarray(teacher_geometry["hand_gap"]["r"], dtype=np.float64),
        "teacher_foot_gap_l": np.asarray(teacher_geometry["foot_gap"]["l"], dtype=np.float64),
        "teacher_foot_gap_r": np.asarray(teacher_geometry["foot_gap"]["r"], dtype=np.float64),
        "new_penetration": (np.maximum(np.asarray(model_geometry["penetration"]), 0) > KEYFRAME_PENETRATION_M)
                            & ~(np.maximum(np.asarray(teacher_geometry["penetration"]), 0) > KEYFRAME_PENETRATION_M),
        "shared_penetration": (np.maximum(np.asarray(model_geometry["penetration"]), 0) > KEYFRAME_PENETRATION_M)
                              & (np.maximum(np.asarray(teacher_geometry["penetration"]), 0) > KEYFRAME_PENETRATION_M),
        "velocity_defined": np.arange(F) > 0,
        "normalized_prediction": pred_x0,
        "decoded_pos": pos, "decoded_quat": quat, "decoded_q": q,
        "teacher_x0": teacher_x0,
    }
    problems = verify_window_curves(curves)
    if problems:
        return {"status": "measurement_incomplete",
                "measurement_gaps": [f"curve:{p}" for p in problems],
                "metrics": None, "curves": None}
    saved_root_check = None
    if saved_world_root is not None:
        sr = np.asarray(saved_world_root, dtype=np.float64)
        saved_root_check = {
            "max_abs_position_m": float(np.abs(sr[:, :3] - world_root[:, :3]).max()),
            "max_rotation_deg": float(quat_angle_deg(sr[:, 3:7], world_root[:, 3:7]).max()),
            "bitwise": bool(np.array_equal(sr[:, :3], world_root[:, :3])
                            and np.array_equal(sr[:, 3:7], world_root[:, 3:7])),
        }
    metrics = {
        "status": "measured",
        "demo": demo, "model": model, "window_start": int(start), "T": int(T),
        "kind": kind, "group": group,
        "loss": loss,
        "physical": {**physical, "decoded_rebuild": {
            "max_abs": decode_max, "bitwise": decode_bitwise,
            "physical_rotation_equivalence_max_deg": rot_equiv,
        }, "saved_world_root": saved_root_check},
        "geometry": geometry, "reference": reference,
        "bins": bin_facts(curves),
        "measurement_status": "measured", "measurement_gaps": [],
    }
    return {"status": "measured", "measurement_gaps": [], "metrics": metrics,
            "curves": curves}


class _cfg_like:
    """Minimal geometry config holder (hand/foot gates)."""

    def __init__(self, hand_gap_m: float, foot_ground_gap_m: float = 0.005):
        self.hand_gap_m = float(hand_gap_m)
        self.foot_ground_gap_m = float(foot_ground_gap_m)


def decode_and_measure(*, demo: str, model: str, start: int, T: int, kind: str,
                       group: str, pred_x0: np.ndarray, teacher_x0: np.ndarray,
                       world_from_clip: np.ndarray, teacher_root: np.ndarray,
                       teacher_joints: np.ndarray, teacher_kpts: np.ndarray,
                       model_geometry: Mapping[str, np.ndarray],
                       teacher_geometry: Mapping[str, np.ndarray],
                       tokenizer, assembler, schedule, frame_dt: float,
                       task_tol_m: float, hand_gap_m: float,
                       saved_world_root: np.ndarray | None = None,
                       saved_decoded: tuple | None = None,
                       decode_device: str = "cpu") -> dict:
    problems = validate_normalized(pred_x0, frames=WINDOW)
    if problems:
        return {"status": "measurement_incomplete",
                "measurement_gaps": [f"pred:{p}" for p in problems],
                "metrics": None, "curves": None}
    decoded = decode_production(pred_x0, tokenizer, device=decode_device)
    out = measure_window(
        demo=demo, model=model, start=start, T=T, kind=kind, group=group,
        pred_x0=pred_x0, teacher_x0=teacher_x0, decoded=decoded,
        world_from_clip=world_from_clip, teacher_root=teacher_root,
        teacher_joints=teacher_joints, teacher_kpts=teacher_kpts,
        model_geometry=model_geometry, teacher_geometry=teacher_geometry,
        tokenizer=tokenizer, assembler=assembler, schedule=schedule,
        frame_dt=frame_dt, task_tol_m=task_tol_m, hand_gap_m=hand_gap_m,
        saved_world_root=saved_world_root, decode_device=decode_device)
    if out.get("metrics") is not None and saved_decoded is not None:
        saved = [np.asarray(v) for v in saved_decoded]
        rebuilt = decoded
        out["metrics"]["physical"]["decoded_rebuild"]["saved_max_abs"] = {
            name: float(np.abs(saved[i].astype(np.float64)
                               - rebuilt[i].astype(np.float64)).max())
            for i, name in enumerate(("pos", "quat", "q"))}
        out["metrics"]["physical"]["decoded_rebuild"]["saved_bitwise"] = {
            name: bool(np.array_equal(saved[i], rebuilt[i]))
            for i, name in enumerate(("pos", "quat", "q"))}
    return out


# ---------------------------------------------------------------------------
# cross-window pairs and fused temporal facts
# ---------------------------------------------------------------------------

def cross_window_rows(*, demo: str, model: str, starts: Sequence[int], T: int,
                      decoded: tuple[np.ndarray, np.ndarray, np.ndarray],
                      fk_windows: np.ndarray,
                      root_position_error: np.ndarray,
                      root_rotation_error_deg: np.ndarray,
                      frame_dt: float) -> list[dict]:
    pos, quat, joints = decoded
    F = pos.shape[1]
    if (pos.shape != (len(starts), F, 3) or quat.shape != (len(starts), F, 4)
            or joints.shape != (len(starts), F, 29)
            or fk_windows.shape != (len(starts), F, 15, 3)):
        raise ValueError("decoded window matrix shape")
    entries = contributor_entries(starts, T)
    rows = []
    for t, row in enumerate(entries):
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                ia, ja, wa = row[a]
                ib, jb, wb = row[b]
                if ja >= F or jb >= F:
                    raise ValueError("local frame out of range")
                rows.append({
                    "key": pair_key(demo, model, t, ia, ib),
                    "demo": demo, "model": model, "global_frame": int(t),
                    "window_i": int(ia), "window_j": int(ib),
                    "start_i": int(starts[ia]), "start_j": int(starts[ib]),
                    "local_i": int(ja), "local_j": int(jb),
                    "weight_i": float(wa), "weight_j": float(wb),
                    "position_diff_m": float(np.linalg.norm(pos[ia, ja] - pos[ib, jb])),
                    "rotation_diff_deg": float(quat_angle_deg(quat[ia, ja], quat[ib, jb])),
                    "joint_mae_rad": float(np.abs(joints[ia, ja] - joints[ib, jb]).mean()),
                    "joint_max_rad": float(np.abs(joints[ia, ja] - joints[ib, jb]).max()),
                    "fk_mean_diff_m": float(np.linalg.norm(
                        fk_windows[ia, ja] - fk_windows[ib, jb], axis=-1).mean()),
                    "teacher_root_pos_error_i_m": float(root_position_error[ia, ja]),
                    "teacher_root_pos_error_j_m": float(root_position_error[ib, jb]),
                    "teacher_root_rot_error_i_deg": float(root_rotation_error_deg[ia, ja]),
                    "teacher_root_rot_error_j_deg": float(root_rotation_error_deg[ib, jb]),
                    "pair_position_limit_m": float(KEYFRAME_PENETRATION_M),
                    "pair_rotation_limit_deg": float(np.rad2deg(
                        ANGULAR_SPEED_LIMIT_RADPS * frame_dt)),
                })
    expected = len(cross_window_pair_keys(demo, model, starts, T))
    if len(rows) != expected:
        raise ValueError("cross-window pair denominator changed")
    return rows


def same_frame_penetration_classes(*, demo: str, model: str, starts: Sequence[int],
                                   T: int, window_penetration: np.ndarray,
                                   fused_penetration: np.ndarray) -> dict:
    F = WINDOW
    if window_penetration.shape != (len(starts), F):
        raise ValueError("window penetration shape")
    fused = np.asarray(fused_penetration, dtype=np.float64)
    if fused.shape != (T,):
        raise ValueError("fused penetration shape")
    entries = contributor_entries(starts, T)
    counts = {"all_penetrate": 0, "none_penetrate": 0, "mixed": 0}
    fusion = {"all_penetrate_pass": 0, "all_penetrate_fail": 0,
              "none_penetrate_pass": 0, "none_penetrate_fail": 0,
              "mixed_pass": 0, "mixed_fail": 0}
    fusion_only = []
    rows = []
    for t, contributors in enumerate(entries):
        if not contributors:
            raise ValueError(f"uncovered frame {t}")
        pens = [float(max(window_penetration[i, j], 0.0)) for i, j, _ in contributors]
        if all(p > KEYFRAME_PENETRATION_M for p in pens):
            cls = "all_penetrate"
        elif all(p <= KEYFRAME_PENETRATION_M for p in pens):
            cls = "none_penetrate"
        else:
            cls = "mixed"
        counts[cls] += 1
        passed = bool(fused[t] <= KEYFRAME_PENETRATION_M)
        fusion[f"{cls}_{'pass' if passed else 'fail'}"] += 1
        if cls == "none_penetrate" and not passed:
            fusion_only.append(int(t))
        rows.append({"global_frame": int(t), "class": cls,
                     "contributors": [int(i) for i, _, _ in contributors],
                     "window_penetration_m": pens,
                     "fused_penetration_m": float(fused[t]),
                     "fusion_pass": passed})
    if sum(counts.values()) != T:
        raise ValueError("penetration classification does not cover T")
    if sum(fusion.values()) != T:
        raise ValueError("fusion classification does not cover T")
    return {"demo": demo, "model": model, "counts": counts,
            "fusion": fusion, "fusion_only_frames": fusion_only,
            "rows": rows,
            "note": "fusion_only = every positive contributor <=2mm yet fused keyframe >2mm"}


def fused_temporal_facts(*, root: np.ndarray, joints: np.ndarray,
                         starts: Sequence[int], frame_dt: float) -> dict:
    root = _as_float(root, "fused root")
    joints = _as_float(joints, "fused joints")
    T = len(root)
    if root.shape != (T, 7) or joints.shape != (T, 29):
        raise ValueError("fused trajectory shape")
    if not math.isfinite(frame_dt) or frame_dt <= 0:
        raise ValueError("frame_dt invalid")
    speed = np.full(T, np.nan)
    step_deg = np.full(T, np.nan)
    if T > 1:
        step_deg[1:] = quat_angle_deg(root[:-1, 3:7], root[1:, 3:7])
        speed[1:] = np.deg2rad(step_deg[1:]) / frame_dt
    start_set = {int(s) for s in starts if 0 < int(s) < T}
    seam = np.zeros(T, dtype=bool)
    seam[sorted(start_set)] = True
    seam_speed = speed[seam & np.isfinite(speed)]
    interior_speed = speed[(~seam) & np.isfinite(speed)]
    finite_speed = np.isfinite(speed)
    return {
        "frame_dt": float(frame_dt),
        "root_angular_speed_radps": speed,
        "root_rotation_step_deg": step_deg,
        "seam_mask": seam,
        "global_max_radps": float(np.nanmax(speed)),
        "global_max_frame": int(np.nanargmax(speed)),
        "seam_max_radps": float(seam_speed.max()) if seam_speed.size else None,
        "interior_max_radps": float(interior_speed.max()) if interior_speed.size else None,
        "seam_max_step_deg": float(np.nanmax(step_deg[seam])) if seam.any() else None,
        "first_exceedance_frame": _first_exceed(
            np.where(finite_speed, speed, np.nan), ANGULAR_SPEED_LIMIT_RADPS),
        "activation_transition_frames": activation_transition_frames(starts, T),
    }


def activation_transition_frames(starts: Sequence[int], T: int) -> list[int]:
    entries = contributor_entries(starts, T)
    out = []
    for t in range(1, T):
        common = ({i for i, _, _ in entries[t - 1]}
                  & {i for i, _, _ in entries[t]})
        if not common:
            out.append(t)
    return out


def window_internal_jumps(decoded: tuple[np.ndarray, np.ndarray, np.ndarray],
                          starts: Sequence[int], frame_dt: float) -> list[dict]:
    pos, quat, joints = decoded
    out = []
    for i, start in enumerate(starts):
        step_deg = quat_angle_deg(quat[i, :-1], quat[i, 1:])
        speed = np.deg2rad(step_deg) / frame_dt
        joint_step = np.abs(np.diff(joints[i], axis=0)).max(1)
        pos_step = np.linalg.norm(np.diff(pos[i], axis=0), axis=1)
        out.append({
            "window": int(i), "start": int(start),
            "max_rotation_step_deg": float(step_deg.max()),
            "max_rotation_step_local_frame": int(np.argmax(step_deg)) + 1,
            "max_angular_speed_radps": float(speed.max()),
            "max_joint_step_rad": float(joint_step.max()),
            "max_joint_step_local_frame": int(np.argmax(joint_step)) + 1,
            "max_position_step_m": float(pos_step.max()),
        })
    return out


# ---------------------------------------------------------------------------
# noise seeds
# ---------------------------------------------------------------------------

def noise_seed_for_window(demo: str, start: int, base: int = NOISE_SEED) -> int:
    payload = f"{NOISE_PROTOCOL}|{int(base)}|{demo}|{int(start)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") % (2 ** 63)


def epsilon_for_window(demo: str, start: int, *, device="cpu") -> tuple[torch.Tensor, int]:
    seed = noise_seed_for_window(demo, start)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(WINDOW, 38, generator=gen, dtype=torch.float32)
    return eps.to(torch.device(device)), seed


def array_sha256(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# aggregation and attribution judgement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    path: str
    reducer: str  # mean | max | min | count


WINDOW_METRICS = (
    MetricSpec("loss.total", "mean"), MetricSpec("loss.pos", "mean"),
    MetricSpec("loss.rot6d", "mean"), MetricSpec("loss.joint", "mean"),
    MetricSpec("loss.geodesic_rad2", "mean"), MetricSpec("loss.vel", "mean"),
    MetricSpec("loss.fk_m2", "mean"),
    MetricSpec("physical.root_position.rmse_m", "mean"),
    MetricSpec("physical.root_position.max_m", "max"),
    MetricSpec("physical.root_rotation.mean_deg", "mean"),
    MetricSpec("physical.root_rotation.max_deg", "max"),
    MetricSpec("physical.joints.mae_rad", "mean"),
    MetricSpec("physical.joints.max_abs_rad", "max"),
    MetricSpec("physical.fk.mpjpe_m", "mean"),
    MetricSpec("physical.fk.max_m", "max"),
    MetricSpec("physical.hand_fk_m.l", "mean"),
    MetricSpec("physical.hand_fk_m.r", "mean"),
    MetricSpec("geometry.penetration.max_m", "max"),
    MetricSpec("geometry.penetration.violation_frames", "count"),
    MetricSpec("geometry.penetration.new_violation_frames", "count"),
    MetricSpec("geometry.penetration.shared_violation_frames", "count"),
    MetricSpec("geometry.hand_gap.l.min_m", "min"),
    MetricSpec("geometry.hand_gap.r.min_m", "min"),
    MetricSpec("geometry.foot_gap.l.min_m", "min"),
    MetricSpec("geometry.foot_gap.r.min_m", "min"),
)


def _metric_value(metrics: Mapping, path: str):
    cur = metrics
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _reduce(values: list[float], reducer: str) -> float:
    a = np.asarray(values, dtype=np.float64)
    if reducer == "mean":
        return float(a.mean())
    if reducer == "max":
        return float(a.max())
    if reducer == "min":
        return float(a.min())
    if reducer == "count":
        return float(a.sum())
    raise ValueError(reducer)


def aggregate_window_rows(rows: Sequence[Mapping], plan: Mapping,
                          *, value_key: str = "metrics") -> dict:
    expected = {"train6": 6, "val6": 6, "known_case": 1}
    out = {}
    for group, n_demo in expected.items():
        demos = plan[group] if group != "known_case" else [plan["known_case"]]
        group_rows = [r for r in rows if r.get("group") == group]
        out[group] = {}
        for model in MODELS:
            model_rows = [r for r in group_rows if r.get("model") == model]
            for category in ("all", "regular", "tail"):
                selected = [r for r in model_rows
                            if category == "all" or r.get("kind") == category]
                stats = {}
                missing = []
                for spec in WINDOW_METRICS:
                    per_demo = {}
                    for demo in demos:
                        vals = []
                        for r in selected:
                            if r.get("demo") != demo:
                                continue
                            payload = (r.get(value_key) if value_key in r else r) or {}
                            v = _metric_value(payload, spec.path)
                            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                                    and math.isfinite(float(v))
                                    and (spec.reducer != "count"
                                         or float(v).is_integer())):
                                vals.append(float(v))
                        if vals:
                            per_demo[demo] = _reduce(vals, spec.reducer)
                        else:
                            missing.append(f"{demo}:{spec.path}")
                    stats[spec.path] = {
                        "per_demo": per_demo,
                        "macro_equal_demo": (float(np.mean(list(per_demo.values())))
                                             if len(per_demo) == len(demos) else None),
                        "n_demo_measured": len(per_demo),
                        "n_demo_expected": len(demos),
                        "reducer": spec.reducer,
                    }
                measured = len(selected)
                out[group][f"{model}|{category}"] = {
                    "n_expected_windows": _expected_windows(plan, group, category, model),
                    "n_measured_windows": measured,
                    "n_missing_windows": max(0, _expected_windows(plan, group, category, model) - measured),
                    "metrics": stats, "missing": missing[:50],
                }
    return out


def _expected_windows(plan: Mapping, group: str, category: str, model: str) -> int:
    per = plan["per_demo"]
    demos = plan[group] if group != "known_case" else [plan["known_case"]]
    total = 0
    for demo in demos:
        rec = per[demo]
        if category == "all":
            total += len(rec["inference"])
        elif category == "regular":
            total += len(rec["regular"])
        else:
            total += len(rec["tail"])
    return total


def aggregate_probe_rows(rows: Sequence[Mapping], plan: Mapping) -> dict:
    out = {"probe_classes": {}, "groups": {}}
    for probe in PROBE_NAMES:
        selected = [r for r in rows if r.get("probe") == probe]
        out["probe_classes"][probe] = {
            "n_expected": (325 if probe == "diffusion_denoise" else 65),
            "n_measured": len(selected),
            "n_execution_failed": sum(r.get("status") == "execution_failed" for r in selected),
        }
    for group in ("train6", "val6", "known_case"):
        for probe in PROBE_NAMES:
            selected = [r for r in rows
                        if r.get("group") == group and r.get("probe") == probe
                        and r.get("metrics") is not None]
            vals = [float(r["metrics"]["physical"]["fk"]["mpjpe_m"]) for r in selected]
            out["groups"][f"{group}|{probe}"] = {
                "n": len(selected),
                "fk_mpjpe_mean_m": float(np.mean(vals)) if vals else None,
            }
    return out


def denoise_snr_table(rows: Sequence[Mapping]) -> dict:
    """Per-t error-vs-SNR facts; no cross-t average replacement."""
    out = {}
    for t in DENOISE_TIMESTEPS:
        selected = [r for r in rows if r.get("probe") == "diffusion_denoise"
                    and r.get("t") == t and r.get("metrics") is not None]
        if not selected:
            out[str(t)] = None
            continue
        out[str(t)] = {
            "n": len(selected),
            "alpha_cumprod": float(selected[0]["alpha_cumprod"]),
            "snr": float(selected[0]["snr"]),
            "normalized_total_mean": float(np.mean(
                [r["metrics"]["loss"]["total"] for r in selected])),
            "fk_mpjpe_mean_m": float(np.mean(
                [r["metrics"]["physical"]["fk"]["mpjpe_m"] for r in selected])),
        }
    gen = [r for r in rows if r.get("probe") == "diffusion_ddim25"
           and r.get("metrics") is not None]
    out["ddim25"] = ({
        "n": len(gen),
        "normalized_total_mean": float(np.mean(
            [r["metrics"]["loss"]["total"] for r in gen])),
        "fk_mpjpe_mean_m": float(np.mean(
            [r["metrics"]["physical"]["fk"]["mpjpe_m"] for r in gen])),
    } if gen else None)
    return out


def attribution_judgement(*, w_summary: Mapping | None, p_summary: Mapping | None,
                          oracle: Mapping | None,
                          evidence_gaps: Sequence[str]) -> dict:
    """Conservative fact-based status; never upgrades the interaction method."""
    if evidence_gaps:
        return {"status": "attribution_insufficient",
                "mechanisms": {}, "supported": [],
                "undecided": ["evidence_gaps"], "note": "missing evidence blocks attribution"}
    mechanisms: dict[str, dict] = {}
    groups = (w_summary or {}).get("groups", {})
    train = groups.get("train6", {})

    # 1) training fit / physical error on the 26 train regular windows.
    def macro(rec, key: str):
        block = ((rec or {}).get("metrics") or {}).get(key)
        return block.get("macro_equal_demo") if isinstance(block, dict) else None

    train_phys = None
    for key, rec in train.items():
        if key.startswith("deterministic|regular") or key.startswith("diffusion|regular"):
            train_phys = rec
            break
    oracle_fk = (oracle or {}).get("fk_mean_m")
    train_fk = macro(train_phys, "physical.fk.mpjpe_m")
    if train_fk is not None and oracle_fk is not None and oracle_fk >= 0:
        ratio = train_fk / max(oracle_fk, 1e-12)
        mechanisms["training_fit"] = {
            "train_regular_fk_mean_m": float(train_fk),
            "oracle_fk_mean_m": float(oracle_fk),
            "ratio": float(ratio),
            "supported": bool(train_fk > ATTRIBUTION_COMPARATORS["training_fit_abs_m"]
                              and ratio > ATTRIBUTION_COMPARATORS["training_fit_ratio"]),
        }
    else:
        mechanisms["training_fit"] = {"supported": None,
                                      "reason": "fk_oracle_comparison_unavailable"}

    # 2) same-frame cross-window disagreement, over ALL positive contributors.
    pair_summary = (w_summary or {}).get("pair_summary")
    if pair_summary:
        mechanisms["cross_window"] = {
            "max_position_diff_m": float(pair_summary["max_position_diff_m"]),
            "max_rotation_diff_deg": float(pair_summary["max_rotation_diff_deg"]),
            "max_joint_max_rad": float(pair_summary["max_joint_max_rad"]),
            "supported": bool(
                pair_summary["max_position_diff_m"] > ATTRIBUTION_COMPARATORS["cross_window_position_m"]
                or pair_summary["max_rotation_diff_deg"] > ATTRIBUTION_COMPARATORS["cross_window_rotation_deg"]),
        }
    else:
        mechanisms["cross_window"] = {"supported": None,
                                      "reason": "pair_summary_missing"}

    # 3) denoise vs actual DDIM sampling (never merge the two matrices).
    snr = (p_summary or {}).get("denoise_snr") or {}
    ddim = snr.get("ddim25")
    high = snr.get(str(DENOISE_TIMESTEPS[-1]))
    if ddim and high:
        mechanisms["denoise_sampling"] = {
            "ddim_total_mean": float(ddim["normalized_total_mean"]),
            "t999_total_mean": float(high["normalized_total_mean"]),
            "ddim_minus_t999": float(ddim["normalized_total_mean"] - high["normalized_total_mean"]),
            "supported": bool(ddim["normalized_total_mean"] > high["normalized_total_mean"]),
        }
    else:
        mechanisms["denoise_sampling"] = {"supported": None,
                                          "reason": "denoise_or_ddim_missing"}

    # 4) train/val and regular/tail coverage context.
    vals = {}
    for group in ("train6", "val6"):
        for key, rec in groups.get(group, {}).items():
            value = macro(rec, "physical.fk.mpjpe_m")
            if value is not None:
                vals[f"{group}|{key}"] = value
    train_v = next((v for k, v in vals.items() if k.startswith("train6|") and v is not None), None)
    val_v = next((v for k, v in vals.items() if k.startswith("val6|") and v is not None), None)
    reg_v = vals.get("train6|deterministic|regular")
    tail_v = vals.get("train6|deterministic|tail")
    if None not in (train_v, val_v):
        mechanisms["coverage_context"] = {
            "train_fk_mean_m": train_v, "val_fk_mean_m": val_v,
            "abs_difference_m": abs(train_v - val_v),
            "regular_vs_tail_abs_difference_m": (abs(reg_v - tail_v)
                                                 if None not in (reg_v, tail_v) else None),
            "supported": bool(abs(train_v - val_v) > ATTRIBUTION_COMPARATORS["coverage_context_m"]
                              or (None not in (reg_v, tail_v)
                                  and abs(reg_v - tail_v) > ATTRIBUTION_COMPARATORS["coverage_context_m"])),
        }
    else:
        mechanisms["coverage_context"] = {"supported": None,
                                          "reason": "group_comparison_unavailable"}

    # 5) teacher-inherited defects at the same frames.
    teacher_shared = (w_summary or {}).get("teacher_inheritance")
    if teacher_shared:
        mechanisms["teacher_inheritance"] = {
            "shared_penetration_frames": int(teacher_shared.get("shared_violation_frames", 0)),
            "model_new_frames": int(teacher_shared.get("new_violation_frames", 0)),
            "model_penetration_max_m": teacher_shared.get("model_penetration_max_m"),
            "teacher_penetration_max_m": teacher_shared.get("teacher_penetration_max_m"),
            "supported": bool(int(teacher_shared.get("shared_violation_frames", 0)) > 0),
        }
    else:
        mechanisms["teacher_inheritance"] = {"supported": None,
                                             "reason": "teacher_pair_facts_missing"}

    supported = [k for k, v in mechanisms.items() if v.get("supported") is True]
    undecided = [k for k, v in mechanisms.items() if v.get("supported") is None]
    if len(supported) >= 2:
        status = "mixed_evidence"
    elif len(supported) == 1 and not undecided:
        status = "mechanism_supported"
    elif len(supported) == 1 and undecided:
        status = "mixed_evidence"
    else:
        status = "attribution_insufficient"
    return {"status": status, "mechanisms": mechanisms,
            "supported": supported, "undecided": undecided,
            "comparators": dict(ATTRIBUTION_COMPARATORS),
            "note": "descriptive evidence families; no responsibility percentage",
            "final_interaction_method": "not_ready", "training_steps": 0,
            "auto_execute": False}


def _nested(doc, path: str):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def summarize_pairs(pairs: Sequence[Mapping]) -> dict:
    if not pairs:
        return {"n": 0, "max_position_diff_m": None,
                "max_rotation_diff_deg": None, "max_joint_max_rad": None,
                "frames_with_any_pair_above_limits": 0,
                "triple_window_frames": 0}
    pos = np.asarray([p["position_diff_m"] for p in pairs], dtype=np.float64)
    rot = np.asarray([p["rotation_diff_deg"] for p in pairs], dtype=np.float64)
    joint = np.asarray([p["joint_max_rad"] for p in pairs], dtype=np.float64)
    above = np.asarray([p["position_diff_m"] > ATTRIBUTION_COMPARATORS["cross_window_position_m"]
                        or p["rotation_diff_deg"] > ATTRIBUTION_COMPARATORS["cross_window_rotation_deg"]
                        for p in pairs], dtype=bool)
    by_frame: dict[int, set[int]] = {}
    for p in pairs:
        by_frame.setdefault(int(p["global_frame"]), set()).update(
            (int(p["window_i"]), int(p["window_j"])))
    return {
        "n": len(pairs),
        "max_position_diff_m": float(pos.max()),
        "max_rotation_diff_deg": float(rot.max()),
        "max_joint_max_rad": float(joint.max()),
        "frames_with_any_pair_above_limits": int(len({p["global_frame"]
                                                      for p, hit in zip(pairs, above) if hit})),
        "triple_window_frames": int(sum(1 for windows in by_frame.values()
                                        if len(windows) >= 3)),
    }
