"""Teacher NPZ loading and condition assembly (plan sections 4.2/5.1).

Each ``resources/data/omniretarget/robot-object/<demo>_<variant>.npz`` holds
``qpos`` [T, 43] with **quaternion-first** freejoints (teacher layout),
``human_joints`` [T, 52, 3] in the same scene, and ``fps``. This module
converts to the project convention and assembles :class:`InteractionClip`.

Source vs target object trajectories: the M0 pairing audit and the A1
re-verification (``derived/experiments/omniretarget_first/<run_id>/freeze/``,
plan omniretarget_first_training_plan §4.3) show that the two scenes are
NOT identical even for ``_original`` clips: the NPZ box equals the
similarity-transformed PT box plus a per-demo constant offset, and rot/trans
variants keep ``human_joints`` fixed while the robot scene (root and box)
moves. ``qpos[36:43]`` is therefore the **target** scene box. Reading it
into ``source_obj_pose`` as well is a known approximation retained only
until the I1/I2 source-scene adapter lands; reference contact inference
against the target box is offset by exactly that constant per-demo offset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ireset.data.schema import InteractionClip, ReferenceCondition
from ireset.embodied.contact_sites import (
    SMPLH_INDEX,
    infer_reference_schedule,
)
from ireset.paths import TEACHER_TRAJ_DIR

_CACHE_DIR = Path("")  # set by windows.py; schedules cached under derived/


def convert_teacher_qpos(row: np.ndarray) -> np.ndarray:
    """Teacher NPZ row [43] (quat-first freejoints) -> project [pos, quat] order."""
    out = np.zeros(43)
    out[0:3] = row[4:7]
    out[3:7] = row[0:4]
    out[7:36] = row[7:36]
    out[36:39] = row[40:43]
    out[39:43] = row[36:40]
    return out


def load_teacher_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """-> scene qpos [T, 43] (converted), human [T, 52, 3], fps."""
    npz = np.load(path, allow_pickle=True)
    q = np.asarray(npz["qpos"], dtype=np.float64)
    human = np.asarray(npz["human_joints"], dtype=np.float64)
    fps = float(npz["fps"])
    if q.ndim != 2 or q.shape[1] != 43:
        raise ValueError(f"{path.name}: qpos {q.shape} != [T, 43]")
    if human.shape[:2] != (q.shape[0], 52):
        raise ValueError(f"{path.name}: human_joints {human.shape} incompatible")
    scene = np.stack([convert_teacher_qpos(r) for r in q])
    return scene, human, fps


def clip_frame_from_human(human: np.ndarray) -> np.ndarray:
    """Fixed clip frame from the reference human (plan 5.1).

    Origin: pelvis horizontal position at frame 0. Yaw: hips -> shoulders
    facing direction at frame 0, about +z. Height and gravity are kept.
    Returns ``world_from_clip`` [4, 4]; degenerate horizontal facing falls
    back to yaw = 0.
    """
    hips = human[0, [SMPLH_INDEX["L_Hip"], SMPLH_INDEX["R_Hip"]]].mean(0)
    shoulders = human[0, [SMPLH_INDEX["L_Shoulder"], SMPLH_INDEX["R_Shoulder"]]].mean(0)
    facing = shoulders - hips
    if np.hypot(facing[0], facing[1]) < 1e-6:
        yaw = 0.0
    else:
        yaw = float(np.arctan2(facing[0], facing[1]))  # +y -> facing, about +z
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz
    T[0, 3], T[1, 3] = human[0, SMPLH_INDEX["Pelvis"], 0], human[0, SMPLH_INDEX["Pelvis"], 1]
    return T


def world_to_clip(points_world: np.ndarray, world_from_clip: np.ndarray) -> np.ndarray:
    """[..., 3] world -> clip frame (inverse of ``world_from_clip``)."""
    R = world_from_clip[:3, :3]
    t = world_from_clip[:3, 3]
    return np.einsum("ji,...j->...i", R, points_world - t)


def keypoints_in_source_object(
    human_kps_world: np.ndarray, source_obj_pose: np.ndarray
) -> np.ndarray:
    """[T, 52, 3] world -> source object frame (plan 5.1)."""
    from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz

    R = quat_to_mat_wxyz(source_obj_pose[:, 3:7])                       # [T,3,3]
    t = source_obj_pose[:, 0:3]
    return np.einsum("tji,tnj->tni", R, human_kps_world - t[:, None, :])


_SPEC: "object | None" = None


def _spec():
    """Process-level G1Spec cache (model compilation is ~0.1 s per call)."""
    global _SPEC
    if _SPEC is None:
        from ireset.embodied.g1_spec import G1Spec

        _SPEC = G1Spec()
    return _SPEC


def build_interaction_clip(
    npz_name: str,
    schedule_cache: Path | None = None,
    clamp_tol_rad: float = 1e-3,
    drop_tol_rad: float = 2e-2,
    teacher_traj_dir: Path = TEACHER_TRAJ_DIR,
) -> InteractionClip:
    """Load one teacher NPZ into an :class:`InteractionClip`.

    Joint-limit handling (plan 7.2): violations <= ``clamp_tol_rad`` are
    clamped (recorded in ``validation``); any violation beyond
    ``drop_tol_rad`` marks the clip invalid (``validation["drop"] = True``)
    so window extraction can exclude it instead of silently clipping.
    """
    spec = _spec()
    lo, hi = spec.joint_range[:, 0], spec.joint_range[:, 1]

    scene, human, fps = load_teacher_npz(teacher_traj_dir / npz_name)
    box = scene[:, 36:43].copy()
    root = scene[:, 0:7].copy()
    q = scene[:, 7:36].copy()

    viol = np.maximum(lo - q, q - hi)
    max_viol = float(viol.max())
    n_viol = int((viol > 0).sum())
    drop = bool(max_viol > drop_tol_rad)
    if max_viol > 0.0:
        q = np.clip(q, lo, hi)

    cached = None
    if schedule_cache is not None:
        cpath = schedule_cache / f"{npz_name}.npz"
        if cpath.exists():
            cached = np.load(cpath, allow_pickle=True)

    if cached is not None:
        schedule = _schedule_from_cache(cached)
    else:
        schedule = infer_reference_schedule(human, box, fps=fps)

    cond = ReferenceCondition(
        source_clip_id=npz_name[: -len(".npz")],
        human_kps_world=human,
        source_obj_pose=box.copy(),
        target_obj_pose=box,
        contacts=schedule.intervals,
        frame_dt=1.0 / fps,
        world_from_clip=clip_frame_from_human(human),
    )
    return InteractionClip(
        condition=cond,
        q=q,
        root=root,
        provenance={"npz": npz_name, "fps": fps, "teacher_traj_dir": str(teacher_traj_dir)},
        validation={
            "joint_limit_max_violation_rad": max_viol,
            "joint_limit_violating_entries": n_viol,
            "clamped": max_viol > 0.0,
            "drop": drop,
        },
    )


def _schedule_from_cache(cached) -> "object":
    from ireset.embodied.contact_sites import ContactInterval, ContactSchedule

    schedule = ContactSchedule()
    arr = cached["intervals"]  # [N, 4] effector_idx, start, end, support_idx
    eff = ("left_hand", "right_hand", "left_foot", "right_foot")
    sup = ("object", "ground")
    for row in arr:
        schedule.intervals.append(
            ContactInterval(eff[int(row[0])], int(row[1]), int(row[2]), sup[int(row[3])])
        )
    return schedule


def schedule_to_cache(schedule) -> np.ndarray:
    """ContactSchedule -> compact [N, 4] array for the cache file."""
    eff = ("left_hand", "right_hand", "left_foot", "right_foot")
    sup = ("object", "ground")
    rows = [
        [eff.index(iv.effector), iv.start, iv.end, sup.index(iv.support)]
        for iv in schedule.intervals
    ]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 4)
