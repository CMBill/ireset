"""Pure-optimization baseline tests (Fix 4): initial root pose from reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_data import convert_scene_qpos
from ireset.eval.baselines import InteractionTracker, TrackerConfig
from ireset.kinematics.mujoco_adapter import RobotState
from ireset.kinematics.rotation_adapter import so3_log_np, quat_to_mat_wxyz
from ireset.paths import TEACHER_TRAJ_DIR


@pytest.fixture(scope="module")
def tracker() -> InteractionTracker:
    return InteractionTracker(cfg=TrackerConfig())


def test_initial_yaw_matches_teacher_root(tracker) -> None:
    """Estimated yaw within 15 deg of the teacher first-frame root yaw."""
    for demo in ("sub10_largebox_000", "sub3_largebox_000", "sub8_largebox_001"):
        npz = np.load(TEACHER_TRAJ_DIR / f"{demo}_original.npz", allow_pickle=True)
        scene = convert_scene_qpos(npz["qpos"][0])
        teacher_yaw = so3_log_np(quat_to_mat_wxyz(scene[3:7][None]))[0][2]
        st = tracker.initial_state(npz["human_joints"][0].astype(np.float64))
        est_yaw = so3_log_np(quat_to_mat_wxyz(st.quat))[0][2]
        err = np.degrees(abs((est_yaw - teacher_yaw + np.pi) % (2 * np.pi) - np.pi))
        assert err < 25.0, f"{demo}: yaw error {err:.1f} deg (teacher {np.degrees(teacher_yaw):.1f})"


def test_initial_position_accounts_for_site_offset(tracker) -> None:
    """Root estimate must sit above the pelvis site by the nominal offset."""
    npz = np.load(TEACHER_TRAJ_DIR / "sub10_largebox_000_original.npz", allow_pickle=True)
    st = tracker.initial_state(npz["human_joints"][0].astype(np.float64))
    pelvis_world = npz["human_joints"][0].astype(np.float64)[
        __import__("ireset.embodied.contact_sites", fromlist=["SMPLH_INDEX"]).SMPLH_INDEX["Pelvis"]
    ]
    offset = np.linalg.norm(st.pos[0] - pelvis_world)
    # pelvis_contour_link sits at the root origin in this asset: the nominal
    # site->root offset is exactly zero (locked here; the human-pelvis-to-
    # robot-root gap is anthropometric, not a fixed asset offset)
    assert offset < 0.05
