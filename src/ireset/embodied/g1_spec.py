"""Locked G1 29-DoF specification.

Single source of truth for the fixed asset: joint order, limits, qpos
addressing, contact/keypoint site names, and the SMPLH anatomical
correspondence. The derived models are produced by
``scripts/build_locked_assets.py``; this module must stay consistent with
``derived/robots/lock_record.json`` (checked by tests).

Conventions (project-wide, see docs/interaction_retarget_impl_plan.md section 4.2):

- Internal root pose storage is ``[x, y, z, qw, qx, qy, qz]`` in meters/radians.
  This matches MuJoCo's freejoint qpos layout, so qpos slices transfer
  directly. Published datasets storing ``[qw, qx, qy, qz, x, y, z]`` must be
  converted explicitly.
- The scene qpos adds the box freejoint at indices [36:43].
- Site positions for sole/hand proxies sit at collision-sphere *centers*; a
  sphere center is not the contact surface. Acceptance-grade surface gaps
  come from MuJoCo ``mj_geomDistance`` (exact for tilted feet too); the
  recorded radii are asset metadata, not inputs to a subtraction shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from ireset.paths import DERIVED_ROBOTS_DIR

N_JOINTS = 29
NQ_ROBOT = 36          # 7 (free base) + 29
NQ_SCENE = 43          # robot + 7 (largebox freejoint)
ROOT_POS_SLICE = slice(0, 3)
ROOT_QUAT_SLICE = slice(3, 7)  # wxyz
JOINT_QPOS_SLICE = slice(7, 36)
SCENE_BOX_QPOS_SLICE = slice(36, 43)

#: Fixed actuated-joint order, identical to the locked MJCF hinge order.
JOINT_NAMES: tuple[str, ...] = (
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: 15 anatomical keypoints (site name -> SMPLH joint name). Site placement
#: follows the upstream holosoma ("smplh", "g1") body selection.
KPT_TO_SMPLH: dict[str, str] = {
    "kpt_pelvis": "Pelvis",
    "kpt_l_hip": "L_Hip",
    "kpt_r_hip": "R_Hip",
    "kpt_l_knee": "L_Knee",
    "kpt_r_knee": "R_Knee",
    "kpt_l_shoulder": "L_Shoulder",
    "kpt_r_shoulder": "R_Shoulder",
    "kpt_l_elbow": "L_Elbow",
    "kpt_r_elbow": "R_Elbow",
    "kpt_l_ankle": "L_Ankle",
    "kpt_r_ankle": "R_Ankle",
    "kpt_l_toe": "L_Toe",
    "kpt_r_toe": "R_Toe",
    "kpt_l_wrist": "L_Wrist",
    "kpt_r_wrist": "R_Wrist",
}

#: Anatomical keypoint site -> hosting body (authoritative table; the asset
#: build script and kinematics adapters both consume this).
KPT_BODIES: dict[str, str] = {
    "kpt_pelvis": "pelvis_contour_link",
    "kpt_l_hip": "left_hip_pitch_link",
    "kpt_r_hip": "right_hip_pitch_link",
    "kpt_l_knee": "left_knee_link",
    "kpt_r_knee": "right_knee_link",
    "kpt_l_shoulder": "left_shoulder_roll_link",
    "kpt_r_shoulder": "right_shoulder_roll_link",
    "kpt_l_elbow": "left_elbow_link",
    "kpt_r_elbow": "right_elbow_link",
    "kpt_l_ankle": "left_ankle_intermediate_1_link",
    "kpt_r_ankle": "right_ankle_intermediate_1_link",
    "kpt_l_toe": "left_ankle_roll_sphere_5_link",
    "kpt_r_toe": "right_ankle_roll_sphere_5_link",
    "kpt_l_wrist": "left_rubber_hand_link",
    "kpt_r_wrist": "right_rubber_hand_link",
}

#: Sole proxy site -> collision-sphere body (sphere centers, radius 5 mm).
SOLE_BODIES: dict[str, str] = {
    f"sole_{side}_{i}": f"{full}_ankle_roll_sphere_{i}_link"
    for side, full in (("l", "left"), ("r", "right"))
    for i in range(1, 6)
}

#: Foot sole proxy sites (sphere centers) per foot, ordered heel-outside,
#: heel-inside, front-outside, front-inside, toe. Spheres 1-4 are the upstream
#: "foot sticking" set; sphere 5 is the toe.
SOLE_SITES: dict[str, tuple[str, ...]] = {
    side: tuple(f"sole_{side}_{i}" for i in range(1, 6)) for side in ("l", "r")
}

#: Collision radii of the sole proxy spheres (meters). Asset metadata only;
#: gap queries use MuJoCo geomDistance (see module docstring).
SOLE_SPHERE_RADIUS_M = 0.005

#: Hand contact proxy sites. Tips are 1 mm collision spheres; palm is the
#: rubber-hand mesh bbox center (an area proxy, radius not applicable).
HAND_SITES: dict[str, dict[str, str]] = {
    side: {"palm": f"hand_{side}_palm", "thumb": f"hand_{side}_thumb", "pinky": f"hand_{side}_pinky"}
    for side in ("l", "r")
}
HAND_TIP_SPHERE_RADIUS_M = 0.001


@dataclass(frozen=True)
class G1Spec:
    """Locked model metadata plus lazily compiled MuJoCo models."""

    plain_xml: Path = DERIVED_ROBOTS_DIR / "g1_29dof_locked.xml"
    scene_xml: Path = DERIVED_ROBOTS_DIR / "g1_29dof_locked_w_largebox.xml"
    urdf: Path = DERIVED_ROBOTS_DIR / "g1_29dof_locked.urdf"

    joint_names: tuple[str, ...] = JOINT_NAMES
    qposadr: np.ndarray = field(default=None, repr=False)  # [29] per-joint qpos address
    joint_range: np.ndarray = field(default=None, repr=False)  # [29, 2] radians

    def __post_init__(self) -> None:
        model = self.load_plain()
        adr = np.zeros(N_JOINTS, dtype=int)
        rng = np.zeros((N_JOINTS, 2), dtype=float)
        for k, name in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} missing in locked model")
            adr[k] = model.jnt_qposadr[jid]
            rng[k] = model.jnt_range[jid]
        object.__setattr__(self, "qposadr", adr)
        object.__setattr__(self, "joint_range", rng)

    def load_plain(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_path(str(self.plain_xml))

    def load_scene(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_path(str(self.scene_xml))

    @staticmethod
    def split_robot_qpos(scene_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split scene qpos [.., 43] into robot [.., 36] and box [.., 7]."""
        scene_qpos = np.asarray(scene_qpos)
        return scene_qpos[..., :NQ_ROBOT], scene_qpos[..., NQ_SCENE - 7 :]

    @staticmethod
    def split_robot_state(robot_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split robot qpos [.., 36] into pos [.., 3], quat wxyz [.., 4], joints [.., 29]."""
        robot_qpos = np.asarray(robot_qpos)
        return (
            robot_qpos[..., ROOT_POS_SLICE],
            robot_qpos[..., ROOT_QUAT_SLICE],
            robot_qpos[..., JOINT_QPOS_SLICE],
        )
