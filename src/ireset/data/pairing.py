"""Source/target scene pairing for OmniRetarget data (plan §4.3, stage A1).

The published teacher NPZ stores one scene: ``human_joints`` (the reference
human) and the robot/object ``qpos``. The OMOMO PT stores the *source* scene
the human actually performed in. M0 found (and this module re-verifies per
consumed demo):

* PT human -> NPZ human is an exact similarity transform
  (scale = 1.32 / subject height, upstream ``config_types/robot.py``);
* the same rotation maps the PT object orientation onto the NPZ box
  orientation;
* the NPZ box *position* equals the similarity-transformed PT box position
  plus a per-demo constant offset (z-dominant), so the source scene and the
  target scene are genuinely different inputs even for ``_original`` clips;
* rot/trans variants keep ``human_joints`` identical while robot and box
  move (the teacher is re-optimized), so source/target must never be assumed
  equal.

Pairing gates (plan §4.3, frozen): human residual <= 1e-5 m, box orientation
mapping error <= 1e-3 deg, constant-offset residual <= 1e-5 m. Quaternion
unit-norm error <= 1e-4 is the recordable normalization bound (plan §4.2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# OMOMO PT tensor layout (upstream load_intermimic_data convention).
PT_HUMAN_COLS = slice(162, 318)  # [T, 52, 3]
PT_OBJECT_COLS = slice(318, 325)  # [T, 7] = [x, y, z, qx, qy, qz, qw]

#: Upstream ``config_types/robot.py``: {"g1": {"robot_height": 1.32, ...}}.
G1_ROBOT_HEIGHT_M = 1.32

# Frozen pairing gates (plan §4.3) and basic numeric bounds (plan §4.2).
GATE_HUMAN_RESIDUAL_M = 1e-5
GATE_BOX_ROT_DEG = 1e-3
GATE_BOX_OFFSET_RESIDUAL_M = 1e-5
GATE_QUAT_NORM_ERR = 1e-4
EXPECTED_FPS = 30.0


def sha256_hex(path: Path, chunk: int = 1 << 20) -> str:
    """Full 64-hex SHA-256 of a file (plan §11 identity convention)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_pt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """OMOMO PT -> (human [T, 52, 3], object [T, 7] [x,y,z,qx,qy,qz,qw])."""
    import torch

    data = (
        torch.load(str(path), map_location="cpu", weights_only=False)
        .detach()
        .numpy()
        .astype(np.float64)
    )
    if data.ndim != 2 or data.shape[1] < PT_OBJECT_COLS.stop:
        raise ValueError(f"{path.name}: PT tensor {data.shape} lacks expected columns")
    human = data[:, PT_HUMAN_COLS].reshape(-1, 52, 3)
    obj = data[:, PT_OBJECT_COLS].copy()
    return human, obj


def load_teacher_npz_raw(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Teacher NPZ -> (qpos [T, 43] raw layout, human [T, 52, 3], fps).

    Raw layout: root [0:7] and box [36:43] both ``[qw,qx,qy,qz,x,y,z]``,
    joints [7:36] (M0 §3.2; opposite of the project convention).
    """
    npz = np.load(path, allow_pickle=True)
    q = np.asarray(npz["qpos"], dtype=np.float64)
    human = np.asarray(npz["human_joints"], dtype=np.float64)
    fps = float(npz["fps"])
    if q.ndim != 2 or q.shape[1] != 43:
        raise ValueError(f"{path.name}: qpos {q.shape} != [T, 43]")
    if human.shape[:2] != (q.shape[0], 52) or human.ndim != 3:
        raise ValueError(f"{path.name}: human_joints {human.shape} incompatible")
    return q, human, fps


def pt_obj_to_project(obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PT object [T, 7] ([x,y,z,qx,qy,qz,qw]) -> (pos [T,3], quat wxyz [T,4])."""
    obj = np.asarray(obj, dtype=np.float64)
    return obj[:, 0:3].copy(), obj[:, [6, 3, 4, 5]].copy()


def npz_box_raw(qpos_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw teacher qpos box cols [36:43] -> (pos [T,3], quat wxyz [T,4])."""
    q = np.asarray(qpos_raw, dtype=np.float64)
    return q[:, 40:43].copy(), q[:, 36:40].copy()


def fit_similarity(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Least-squares similarity A->B (uniform scale, rotation, translation).

    Same estimator as the M0 audit (``scripts/audit_data.py``); returns
    ``(scale, R, t, max |s*A@R.T + t - B|)``.
    """
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = float(np.sum(S * np.sign(np.linalg.det(Vt.T @ U.T))) / np.sum(Ac**2))
    t = cb - scale * R @ ca
    resid = float(np.abs(scale * (A @ R.T) + t - B).max())
    return scale, R, t, resid


def _quat_norm_err_max(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.size == 0:
        return 0.0
    return float(np.abs(np.linalg.norm(q, axis=-1) - 1.0).max())


@dataclass(frozen=True)
class ScenePairing:
    """Per-demo source/target scene verification record (plan §4.3)."""

    demo: str
    t_npz: int
    t_pt: int
    fps: float
    scale: float
    scale_expected: float | None
    rotation_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]
    human_residual_max_m: float
    box_rot_err_deg_max: float
    box_offset_mean_m: tuple[float, float, float]
    box_offset_residual_max_m: float
    box_offset_xy_abs_mean_m: tuple[float, float]
    root_quat_norm_err_max: float
    box_quat_norm_err_max: float
    human_all_finite: bool
    teacher_all_finite: bool
    pt_all_finite: bool
    joint_limit_max_violation_rad: float | None
    gates: dict[str, bool] = field(default_factory=dict)
    passed: bool = False
    error: str | None = None

    def to_json_dict(self) -> dict:
        return {
            "demo": self.demo,
            "t_npz": self.t_npz,
            "t_pt": self.t_pt,
            "t_match": self.t_npz == self.t_pt,
            "fps": self.fps,
            "scale": self.scale,
            "scale_expected": self.scale_expected,
            "scale_rel_diff": (
                None
                if self.scale_expected in (None, 0.0)
                else abs(self.scale - self.scale_expected) / abs(self.scale_expected)
            ),
            "rotation_wxyz": list(self.rotation_wxyz),
            "translation_xyz": list(self.translation_xyz),
            "human_residual_max_m": self.human_residual_max_m,
            "box_rot_err_deg_max": self.box_rot_err_deg_max,
            "box_offset_mean_m": list(self.box_offset_mean_m),
            "box_offset_residual_max_m": self.box_offset_residual_max_m,
            "box_offset_xy_abs_mean_m": list(self.box_offset_xy_abs_mean_m),
            "root_quat_norm_err_max": self.root_quat_norm_err_max,
            "box_quat_norm_err_max": self.box_quat_norm_err_max,
            "human_all_finite": self.human_all_finite,
            "teacher_all_finite": self.teacher_all_finite,
            "pt_all_finite": self.pt_all_finite,
            "joint_limit_max_violation_rad": self.joint_limit_max_violation_rad,
            "gates": dict(self.gates),
            "passed": self.passed,
            "error": self.error,
        }


def source_obj_pose_in_target_scene(
    pt_obj: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """PT object trajectory mapped into the NPZ scene: [T, 7] [x,y,z,qw,qx,qy,qz].

    This is the box the reference human motion actually corresponds to (the
    source scene object expressed in the target scene frame); the target box
    itself is the qpos box, the two differ by the per-demo constant offset.
    """
    pos, quat_wxyz = pt_obj_to_project(pt_obj)
    Rpt = Rot.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()  # wxyz -> xyzw
    src_pos = scale * (pos @ R.T) + t
    src_rot = R[None] @ Rpt
    src_quat = Rot.from_matrix(src_rot).as_quat()  # xyzw
    out = np.empty((len(pos), 7), dtype=np.float64)
    out[:, 0:3] = src_pos
    out[:, 3:4] = src_quat[:, 3:4]
    out[:, 4:7] = src_quat[:, 0:3]
    return out


def verify_scene_pairing(
    demo: str,
    qpos_raw: np.ndarray,
    npz_human: np.ndarray,
    pt_human: np.ndarray,
    pt_obj: np.ndarray,
    fps: float,
    joint_range: np.ndarray | None = None,
    subject_height_m: float | None = None,
) -> ScenePairing:
    """Verify the §4.3 pairing structure of one original demo.

    Gates: human similarity residual, box orientation mapping error,
    constant-offset residual, T match, fps, finiteness and quaternion norm
    bounds. ``joint_range`` (locked asset order) and ``subject_height_m``
    (upstream height metadata) add recorded cross-checks.
    """
    t_npz, t_pt = int(qpos_raw.shape[0]), int(pt_human.shape[0])

    def fail(err: str, **kw) -> ScenePairing:
        rec = ScenePairing(
            demo=demo, t_npz=t_npz, t_pt=t_pt, fps=fps, scale=float("nan"),
            scale_expected=None, rotation_wxyz=(0.0, 0.0, 0.0, 0.0),
            translation_xyz=(0.0, 0.0, 0.0), human_residual_max_m=float("inf"),
            box_rot_err_deg_max=float("inf"), box_offset_mean_m=(0.0, 0.0, 0.0),
            box_offset_residual_max_m=float("inf"), box_offset_xy_abs_mean_m=(0.0, 0.0),
            root_quat_norm_err_max=float("inf"), box_quat_norm_err_max=float("inf"),
            human_all_finite=False, teacher_all_finite=False, pt_all_finite=False,
            joint_limit_max_violation_rad=None,
            gates={"t_match": t_npz == t_pt}, passed=False, error=err,
        )
        for k, v in kw.items():
            rec = _replace_field(rec, k, v)
        return rec

    if t_npz != t_pt:
        return fail(f"T mismatch: npz={t_npz} pt={t_pt}")

    teacher_finite = bool(np.isfinite(qpos_raw).all())
    human_finite = bool(np.isfinite(npz_human).all())
    pt_finite = bool(np.isfinite(pt_human).all() and np.isfinite(pt_obj).all())
    if not (teacher_finite and human_finite and pt_finite):
        return fail("non-finite values in teacher/PT arrays",
                    teacher_all_finite=teacher_finite, human_all_finite=human_finite,
                    pt_all_finite=pt_finite)

    scale, Rfit, t, resid = fit_similarity(pt_human.reshape(-1, 3), npz_human.reshape(-1, 3))

    pt_pos, pt_quat_wxyz = pt_obj_to_project(pt_obj)
    box_pos, box_quat_wxyz = npz_box_raw(qpos_raw)
    pt_norm_err = _quat_norm_err_max(pt_quat_wxyz)
    box_norm_err = _quat_norm_err_max(box_quat_wxyz)

    if pt_norm_err > GATE_QUAT_NORM_ERR or box_norm_err > GATE_QUAT_NORM_ERR:
        rot_err = float("inf")  # degenerate quaternion: rotation mapping undefined
    else:
        Rpt = Rot.from_quat(pt_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()  # wxyz -> xyzw
        Rnpz = Rot.from_quat(box_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
        pred = Rfit[None] @ Rpt
        tr = np.trace(pred.transpose(0, 2, 1) @ Rnpz, axis1=1, axis2=2)
        rot_err = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).max())

    pn = scale * (pt_pos @ Rfit.T) + t
    offset = box_pos - pn
    offset_mean = offset.mean(0)
    offset_resid = float(np.abs(offset - offset_mean).max())

    root_q = qpos_raw[:, 0:4]
    rot_fit_quat = Rot.from_matrix(Rfit).as_quat()  # xyzw
    joint_viol = None
    if joint_range is not None:
        lo, hi = joint_range[:, 0], joint_range[:, 1]
        q = qpos_raw[:, 7:36]
        joint_viol = float(np.abs(np.clip(q, lo, hi) - q).max())

    scale_expected = (
        None if subject_height_m is None else G1_ROBOT_HEIGHT_M / subject_height_m
    )

    gates = {
        "t_match": t_npz == t_pt,
        "fps_30": abs(fps - EXPECTED_FPS) < 1e-12,
        "human_residual": resid <= GATE_HUMAN_RESIDUAL_M,
        "box_rot_mapping": rot_err <= GATE_BOX_ROT_DEG,
        "box_offset_constant": offset_resid <= GATE_BOX_OFFSET_RESIDUAL_M,
        "root_quat_norm": _quat_norm_err_max(root_q) <= GATE_QUAT_NORM_ERR,
        "box_quat_norm": box_norm_err <= GATE_QUAT_NORM_ERR,
        "pt_quat_norm": pt_norm_err <= GATE_QUAT_NORM_ERR,
    }
    rec = ScenePairing(
        demo=demo, t_npz=t_npz, t_pt=t_pt, fps=fps, scale=scale,
        scale_expected=scale_expected,
        rotation_wxyz=(float(rot_fit_quat[3]), float(rot_fit_quat[0]),
                       float(rot_fit_quat[1]), float(rot_fit_quat[2])),
        translation_xyz=(float(t[0]), float(t[1]), float(t[2])),
        human_residual_max_m=resid,
        box_rot_err_deg_max=rot_err,
        box_offset_mean_m=(float(offset_mean[0]), float(offset_mean[1]), float(offset_mean[2])),
        box_offset_residual_max_m=offset_resid,
        box_offset_xy_abs_mean_m=(float(abs(offset_mean[0])), float(abs(offset_mean[1]))),
        root_quat_norm_err_max=_quat_norm_err_max(root_q),
        box_quat_norm_err_max=_quat_norm_err_max(box_quat_wxyz),
        human_all_finite=True, teacher_all_finite=True, pt_all_finite=True,
        joint_limit_max_violation_rad=joint_viol,
        gates=gates, passed=all(gates.values()),
    )
    return rec


def _replace_field(rec: ScenePairing, key: str, value) -> ScenePairing:
    import dataclasses

    return dataclasses.replace(rec, **{key: value})


# ---------------------------------------------------------------------------
# Frozen list selection rules (plan §3.3 / §7.1)
# ---------------------------------------------------------------------------

def subject_of(demo: str) -> str:
    return demo.split("_")[0]


def rank_by_hash(demos: list[str], prefix: str) -> list[str]:
    """Sort demo names by SHA256(prefix + demo) hex digest (ascending)."""
    return sorted(demos, key=lambda d: hashlib.sha256(f"{prefix}{d}".encode("utf-8")).hexdigest())


def select_first_per_subject(demos: list[str], prefix: str) -> dict[str, str]:
    """Per subject, the first demo under SHA256(prefix + demo) ordering.

    Plan §3.3 pilot rule uses prefix ``"20261008|teacher_pilot|"``; the val6
    rule (§7.1) uses ``"20261008|"``. Selection is identity-only and must not
    read quality or model results.
    """
    first: dict[str, str] = {}
    for demo in rank_by_hash(demos, prefix):
        first.setdefault(subject_of(demo), demo)
    return dict(sorted(first.items()))
