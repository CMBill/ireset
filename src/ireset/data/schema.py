"""M3 data-layer schema (plan section 4.3).

The full training-time ``ReferenceCondition`` and ``InteractionClip``. The
repair layer keeps its own minimal :class:`ireset.repair.context.ReferenceCondition`
(pipeline input); :meth:`ReferenceCondition.to_repair_condition` adapts this
schema onto it, so the two never drift apart at call sites.

Coordinate conventions (plan 4.2/5.1):

- all poses ``[x, y, z, qw, qx, qy, qz]`` meters/radians;
- ``world_from_clip`` is built from the **reference human** initial position
  and yaw (never the target G1): the clip frame removes the horizontal
  origin and yaw, keeps height and the gravity direction;
- keypoints are stored world-frame; the model conditioners derive the
  source-object-frame and clip-frame views on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ireset.embodied.contact_sites import ContactInterval, ContactSchedule


@dataclass
class ReferenceCondition:
    """Full condition for one interaction clip (plan 4.3)."""

    source_clip_id: str
    human_kps_world: np.ndarray            # [T, K_h, 3], SMPLH 52 layout
    source_obj_pose: np.ndarray            # [T, 7] human-reference scene object
    target_obj_pose: np.ndarray            # [T, 7] desired G1 scene object
    obj_asset_id: str = "largebox_locked"
    ground_plane: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0]))
    contacts: list[ContactInterval] = field(default_factory=list)
    frame_dt: float = 1.0 / 30.0
    world_from_clip: np.ndarray | None = None   # [4, 4]

    @property
    def T(self) -> int:
        return int(self.human_kps_world.shape[0])

    @property
    def schedule(self) -> ContactSchedule:
        return ContactSchedule(intervals=list(self.contacts))

    def truncate(self, T: int) -> "ReferenceCondition":
        """First-T-frames copy; out-of-range contact intervals become empty."""
        return ReferenceCondition(
            source_clip_id=self.source_clip_id,
            human_kps_world=self.human_kps_world[:T],
            source_obj_pose=self.source_obj_pose[:T],
            target_obj_pose=self.target_obj_pose[:T],
            obj_asset_id=self.obj_asset_id,
            ground_plane=self.ground_plane,
            contacts=[
                ContactInterval(iv.effector, min(iv.start, T), min(iv.end, T), iv.support)
                for iv in self.contacts
            ],
            frame_dt=self.frame_dt,
            world_from_clip=self.world_from_clip,
        )

    def contact_flags(self) -> np.ndarray:
        """[T, 4] boolean schedule for (left_hand, right_hand, left_foot, right_foot)."""
        return np.stack(
            [self.schedule.mask(e, self.T) for e in
             ("left_hand", "right_hand", "left_foot", "right_foot")], axis=-1
        )

    def to_repair_condition(self):
        """Adapt onto the repair pipeline's minimal condition."""
        from ireset.repair.context import ReferenceCondition as RepairCondition

        return RepairCondition(
            human_kpts_world=self.human_kps_world,
            box_pose=self.target_obj_pose,
            ref_schedule=self.schedule,
            fps=1.0 / self.frame_dt,
        )


@dataclass
class InteractionClip:
    """Condition + teacher target + provenance for one NPZ file."""

    condition: ReferenceCondition
    q: np.ndarray                          # [T, 29] teacher joints
    root: np.ndarray                       # [T, 7] teacher base (project convention)
    provenance: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.q.shape[0])

    def scene_qpos(self) -> np.ndarray:
        """[T, 43] MuJoCo scene (robot + box from the target trajectory)."""
        T = self.T
        scene = np.zeros((T, 43))
        scene[:, 0:7] = self.root
        scene[:, 7:36] = self.q
        scene[:, 36:43] = self.condition.target_obj_pose
        return scene


@dataclass
class RobotTrajectory:
    """Generator output (plan 14.2): root/q in the world frame."""

    root: np.ndarray                       # [T, 7] xyz + qwxyz
    q: np.ndarray                          # [T, 29]
    frame_dt: float = 1.0 / 30.0
    metadata: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.q.shape[0])

    def scene_qpos(self, box_pose: np.ndarray) -> np.ndarray:
        """Assemble [T, 43] scene qpos with the known box trajectory."""
        scene = np.zeros((self.T, 43))
        scene[:, 0:7] = self.root
        scene[:, 7:36] = self.q
        scene[:, 36:43] = box_pose[: self.T]
        return scene
