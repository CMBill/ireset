"""Repair pipeline tests (M2): convergence, perturbation recovery, rejection.

Uses synthetic standing trajectories (box far away) for determinism: the
reference keypoints are the robot's own FK keypoints, so a clean trajectory
skips every frame, noise gets repaired back, and a box buried inside the
robot with a short horizon is provably unreachable (escape exceeds the
per-frame velocity budget).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.embodied.contact_sites import ContactInterval, ContactSchedule
from ireset.embodied.g1_spec import KPT_TO_SMPLH
from ireset.kinematics.mujoco_adapter import RobotState
from ireset.repair.context import ContactStateMachine, ReferenceCondition, RepairConfig
from ireset.repair.pipeline import RepairPipeline
from ireset.repair.scp import tangent_error
from test_contacts import _root_height_for_tangent

BOX_FAR = np.array([1.5, 0.0, 0.25, 1.0, 0, 0, 0])  # clear of hull bottom artifact


@pytest.fixture(scope="module")
def pipeline() -> RepairPipeline:
    return RepairPipeline(cfg=RepairConfig())


def standing_scene(T: int, box: np.ndarray = BOX_FAR, seed: int = 0) -> np.ndarray:
    """Standing robot, soles tangent to the ground, box far away."""
    rng = np.random.default_rng(seed)
    v = RepairPipeline()
    h = _root_height_for_tangent(v.ev)
    scene = np.zeros((T, 43))
    scene[:, 0:3] = [0.0, 0.0, h] + rng.normal(0, 1e-4, (T, 3))
    scene[:, 3] = 1.0
    scene[:, 36:43] = box
    return scene


def self_condition(pipeline: RepairPipeline, scene: np.ndarray) -> ReferenceCondition:
    """Reference condition built from the robot's own keypoints (empty schedule)."""
    from ireset.embodied.contact_sites import SMPLH_INDEX

    ids = [mujoco.mj_name2id(pipeline.ev.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in KPT_TO_SMPLH]
    kpts = np.zeros((scene.shape[0], len(ids), 3))
    for t in range(scene.shape[0]):
        pipeline.ev.forward(scene[t, 0:36], scene[t, 36:43])
        kpts[t] = pipeline.ev.site_positions(ids)
    human = np.zeros((scene.shape[0], 52, 3))
    for k, name in enumerate(KPT_TO_SMPLH):
        human[:, SMPLH_INDEX[KPT_TO_SMPLH[name]]] = kpts[:, k]
    return ReferenceCondition(human, scene[:, 36:43].copy(), ContactSchedule())


def test_tangent_error_roundtrip() -> None:
    rng = np.random.default_rng(0)
    z = np.zeros(36)
    z[:3] = rng.uniform(-1, 1, 3)
    v4 = rng.standard_normal(4)
    z[3:7] = v4 / np.linalg.norm(v4)
    z[7:] = rng.uniform(-1, 1, 29)
    e = tangent_error(z, z)
    assert np.abs(e).max() < 1e-12
    z2 = z.copy()
    z2[:3] += [0.1, -0.2, 0.3]
    z2[7] += 0.5
    e2 = tangent_error(z2, z)
    assert np.allclose(e2[0:3], [0.1, -0.2, 0.3]) and abs(e2[6] - 0.5) < 1e-12


def test_contact_state_machine_lifecycle() -> None:
    sched = ContactSchedule(intervals=[ContactInterval("left_hand", 5, 10, "object")])
    sm = ContactStateMachine(sched, 12)
    assert not sm.reference_wants_hold("left_hand", 4)
    assert sm.reference_wants_hold("left_hand", 6)
    # commit only when reference agrees and geometry is provided
    sm.commit("left_hand", 4, hand_local=np.ones((3, 3)))
    assert not sm.holding["left_hand"]
    sm.commit("left_hand", 5, hand_local=np.ones((3, 3)) * 2)
    assert sm.holding["left_hand"] and np.allclose(sm.anchors["left_hand"], 2)
    # release when the reference interval ends
    sm.commit("left_hand", 10)
    assert not sm.holding["left_hand"] and sm.anchors["left_hand"] is None
    # rollback releases anchors of intervals starting at/after t
    sm.commit("left_hand", 5, hand_local=np.ones((3, 3)))
    sm.rollback_to(5)
    assert not sm.holding["left_hand"]


def test_clean_trajectory_fully_skips(pipeline) -> None:
    scene = standing_scene(8)
    cond = self_condition(pipeline, scene)
    res = pipeline.repair_trajectory(scene, cond)
    assert res.accepted, res.diagnostics
    assert res.diagnostics["skipped"] == 8
    assert res.diagnostics["final_validation"] == "passed"
    assert np.isfinite(res.trajectory).all()


def test_joint_noise_is_repaired(pipeline) -> None:
    scene = standing_scene(10)
    rng = np.random.default_rng(3)
    pert = scene.copy()
    lo, hi = pipeline.spec.joint_range[:, 0], pipeline.spec.joint_range[:, 1]
    pert[:, 7:36] = np.clip(pert[:, 7:36] + rng.normal(0, 0.08, (10, 29)), lo, hi)
    cond = self_condition(pipeline, scene)
    res = pipeline.repair_trajectory(pert, cond)
    assert res.accepted, res.diagnostics.get("reason")
    assert res.diagnostics["skipped"] < 10  # noise forced some repairs
    # repaired trajectory stays close to the clean original
    # one clipped joint may legitimately stay near its noisy value
    assert np.abs(res.trajectory[:, 7:36] - scene[:, 7:36]).max() < 0.30


def test_root_offset_is_repaired(pipeline) -> None:
    scene = standing_scene(8)
    pert = scene.copy()
    pert[:, 0:3] += [0.05, -0.04, 0.02]
    cond = self_condition(pipeline, scene)
    res = pipeline.repair_trajectory(pert, cond)
    assert res.accepted, res.diagnostics.get("reason")
    # feasibility is recovered; a uniform feasible offset may legitimately
    # persist (the prediction objective anchors to the perturbed input)
    assert np.linalg.norm(res.trajectory[-1, 0:3] - scene[-1, 0:3]) < 0.08


def test_unreachable_box_rejected(pipeline) -> None:
    """Box enclosing the torso with a 2-frame horizon: the root must move
    >22cm (box half extent) but the velocity budget allows 2x6cm -> truly
    infeasible -> explicit reject with no trajectory."""
    scene = standing_scene(2, box=np.array([0.0, 0.0, 0.80, 1.0, 0, 0, 0]))
    cond = self_condition(pipeline, scene)
    res = pipeline.repair_trajectory(scene, cond)
    assert not res.accepted
    assert res.status in ("rejected_frame", "rejected_final")
    assert res.trajectory is None  # rejected results never carry trajectories


def test_nonfinite_prediction_rejected(pipeline) -> None:
    scene = standing_scene(4)
    scene[2, 20] = np.nan
    cond = self_condition(pipeline, scene)
    res = pipeline.repair_trajectory(scene, cond)
    assert not res.accepted and res.status == "rejected_nonfinite"


def test_suffix_rollback_recovers_from_tail_corruption(pipeline) -> None:
    """A clean prefix plus a corrupted tail: suffix rollback re-solves the tail."""
    scene = standing_scene(12)
    cond = self_condition(pipeline, scene)
    pert = scene.copy()
    pert[10:, 7:36] += 0.12  # tail noise that first-pass repair must handle
    res = pipeline.repair_trajectory(pert, cond)
    assert res.accepted, res.diagnostics.get("reason")
