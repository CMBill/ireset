"""Contact schedule inference and anchor extraction (plan sections 8.1/5.1).

Reference side works on SMPLH 52-joint keypoints (the OMOMO/teacher NPZ
``human_joints`` layout, upstream ``SMPLH_DEMO_JOINTS`` order). Robot side
works on locked-model sites. Both sides use the same hysteresis interval
extractor: enter/exit double thresholds plus a minimum duration, so short
threshold crossings are not contacts.

Anchors follow plan section 8.1:

- hand anchors are fixed in the **object (box geom) frame**: the box-surface
  point closest to the hand centroid at contact establishment;
- foot anchors are fixed in the **world frame**: the sole position at
  touchdown, valid after height/velocity gating.

Both are *candidates*: downstream repair/validation commits them only after
the corresponding state passes its checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- SMPLH 52-joint layout (upstream SMPLH_DEMO_JOINTS order) ----------------

SMPLH_JOINTS: tuple[str, ...] = (
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    *(f"L_{chain}{i}" for chain in ("Index", "Middle", "Pinky", "Ring", "Thumb") for i in (1, 2, 3)),
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    *(f"R_{chain}{i}" for chain in ("Index", "Middle", "Pinky", "Ring", "Thumb") for i in (1, 2, 3)),
)
SMPLH_INDEX: dict[str, int] = {n: i for i, n in enumerate(SMPLH_JOINTS)}
assert len(SMPLH_JOINTS) == 52

HAND_TIP_JOINTS: dict[str, list[str]] = {
    "left": ["L_Wrist", "L_Index3", "L_Middle3", "L_Ring3", "L_Pinky3", "L_Thumb3"],
    "right": ["R_Wrist", "R_Index3", "R_Middle3", "R_Ring3", "R_Pinky3", "R_Thumb3"],
}
FOOT_JOINTS: dict[str, list[str]] = {
    "left": ["L_Ankle", "L_Toe"],
    "right": ["R_Ankle", "R_Toe"],
}


@dataclass
class ContactInterval:
    effector: str            # left_hand / right_hand / left_foot / right_foot
    start: int               # inclusive frame
    end: int                 # exclusive frame
    support: str             # object / ground
    anchor: np.ndarray | None = None   # hand: [3] box-geom frame; foot: [3] world
    confidence: float = 1.0


@dataclass
class ContactSchedule:
    intervals: list[ContactInterval] = field(default_factory=list)

    def mask(self, effector: str, T: int) -> np.ndarray:
        m = np.zeros(T, dtype=bool)
        for iv in self.intervals:
            if iv.effector == effector:
                m[iv.start : iv.end] = True
        return m


def hysteresis_intervals(
    metric: np.ndarray, enter: float, exit: float, min_duration: int
) -> list[tuple[int, int]]:
    """Extract [start, end) intervals where metric crosses enter/exit.

    ``metric`` smaller than ``enter`` starts a contact; larger than ``exit``
    ends it. Intervals shorter than ``min_duration`` frames are dropped.
    """
    metric = np.asarray(metric)
    T = metric.shape[0]
    intervals: list[tuple[int, int]] = []
    active = False
    start = 0
    for t in range(T):
        if not active and metric[t] < enter:
            active, start = True, t
        elif active and metric[t] > exit:
            intervals.append((start, t))
            active = False
    if active:
        intervals.append((start, T))
    return [(s, e) for s, e in intervals if e - s >= min_duration]


def infer_reference_schedule(
    human_kpts_world: np.ndarray,      # [T, 52, 3]
    source_obj_pose: np.ndarray,       # [T, 7] project convention [xyz, qwxyz]
    hand_distance_fn=None,             # optional override(world_pts, pose) -> [T] distance
    hand_enter_m: float = 0.06,
    hand_exit_m: float = 0.12,
    foot_enter_m: float = 0.08,
    foot_exit_m: float = 0.14,
    foot_speed_mps: float = 0.35,
    min_duration: int = 3,
    fps: float = 30.0,
) -> ContactSchedule:
    """Infer hand/foot contact intervals from the *reference* human.

    Hand metric: distance from the hand centroid (wrist + fingertips) to the
    box surface in the source object frame. Foot metric: minimum world height
    of ankle/toe, gated by horizontal speed.
    """
    from ireset.geometry.distance_adapter import (
        point_box_surface_distance_numpy,
        world_to_box_geom,
    )

    T = human_kpts_world.shape[0]
    schedule = ContactSchedule()
    for side, hand in (("left", "left_hand"), ("right", "right_hand")):
        ids = [SMPLH_INDEX[n] for n in HAND_TIP_JOINTS[side]]
        centroid = human_kpts_world[:, ids].mean(1)                  # [T,3]
        if hand_distance_fn is not None:
            dist = hand_distance_fn(centroid, source_obj_pose)
        else:
            in_box = world_to_box_geom(centroid, source_obj_pose)     # [T,3] geom frame (AABB proxy)
            dist = point_box_surface_distance_numpy(in_box)
        for s, e in hysteresis_intervals(dist, hand_enter_m, hand_exit_m, min_duration):
            schedule.intervals.append(ContactInterval(hand, s, e, "object"))

    dt = 1.0 / fps
    for side, foot in (("left", "left_foot"), ("right", "right_foot")):
        ids = [SMPLH_INDEX[n] for n in FOOT_JOINTS[side]]
        pts = human_kpts_world[:, ids]                                # [T,2,3]
        height = pts[..., 2].min(1)
        vel = np.linalg.norm(np.diff(pts, axis=0), axis=-1).max(1) / dt
        slow = np.concatenate([[True], vel < foot_speed_mps])
        metric = np.where(slow, height, np.inf)
        for s, e in hysteresis_intervals(metric, foot_enter_m, foot_exit_m, min_duration):
            schedule.intervals.append(ContactInterval(foot, s, e, "ground"))
    return schedule
