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


# --- P0.1 L1 semantic decoupling (p0_relaxed_validation_plan section 4) --------




def test_foot_anchor_establishment_speed_gate(pipeline) -> None:
    """A foot sliding >= foot_anchor_establish_speed_mps vs the previous
    accepted sole positions never locks an anchor; a still one does."""
    scene = standing_scene(2)
    z = scene[0, 0:36].copy()
    box = scene[0, 36:43]
    sched = ContactSchedule(intervals=[ContactInterval("left_foot", 0, 2, "ground")])
    contacts = ContactStateMachine(sched, 2)
    pipeline.ev.forward(z, box)
    pts = pipeline.ev.site_positions(pipeline.ev.sole_sites["l"])
    # 1 cm/frame = 0.3 m/s >= 0.05 -> blocked
    fast = {"l": pts - np.array([0.01, 0.0, 0.0]), "r": pts.copy()}
    cands = pipeline._establishment_candidates(z, box, contacts, 0, prev_sole=fast)
    assert "left_foot" not in cands["foot"]
    # previous sole identical -> speed 0 -> candidate allowed (gap in band)
    same = {"l": pts.copy(), "r": pts.copy()}
    cands2 = pipeline._establishment_candidates(z, box, contacts, 0, prev_sole=same)
    assert "left_foot" in cands2["foot"]


def test_validator_support_inference_speed_gate() -> None:
    """foot_entry_speed_mps=0.05: a grounded foot sliding at 0.1 m/s forms no
    support interval (transition, not certified support); 0.01 m/s does."""
    from ireset.repair.validation import IndependentValidator

    validator = IndependentValidator()
    T = 8
    base = standing_scene(T)
    out = {}
    for tag, v in (("fast", 0.1), ("slow", 0.01)):
        scene = base.copy()
        scene[:, 0] += np.arange(T) * v / 30.0   # pure horizontal slide
        sched = validator._infer_robot_schedule(scene)
        n_foot = sum(1 for iv in sched.intervals if iv.effector.endswith("_foot"))
        out[tag] = n_foot
    assert out["fast"] == 0, out
    assert out["slow"] >= 1, out


# --- P0.1 L2 paired mode loading (config_io) --------------------------------------


def test_mode_configs_strict_and_relaxed() -> None:
    from ireset.repair.config_io import check_pair_invariants, load_mode_configs

    rcfg, vcfg = load_mode_configs("strict")
    assert rcfg.foot_contact_gap_m == 0.005 and vcfg.hand_gap_m == 0.01
    check_pair_invariants(rcfg, vcfg)
    rcfg_r, vcfg_r = load_mode_configs("relaxed")
    # paired relaxations move together and keep every pair invariant
    assert rcfg_r.foot_contact_gap_m == vcfg_r.foot_ground_gap_m
    assert rcfg_r.contact_band_gap_m == vcfg_r.hand_gap_m
    assert rcfg_r.hand_anchor_tol_m <= vcfg_r.hand_anchor_error_m
    # strict gates untouched by the relaxed overlay
    assert rcfg_r.foot_speed_mps == rcfg.foot_speed_mps
    assert vcfg_r.foot_speed_mps == vcfg.foot_speed_mps
    assert vcfg_r.penetration_tolerance_m == vcfg.penetration_tolerance_m
    check_pair_invariants(rcfg_r, vcfg_r)
    # a repair band above its validator gate must fail loudly
    from dataclasses import replace as _replace

    with pytest.raises(ValueError, match="pair invariant"):
        check_pair_invariants(_replace(rcfg, foot_contact_gap_m=0.02), vcfg)
    with pytest.raises(ValueError):
        load_mode_configs("bogus")


def test_foot_anchor_released_on_sliding_prediction(pipeline) -> None:
    """A locked foot anchor degrades to touching-without-anchor when the
    prediction slides >= foot_anchor_establish_speed_mps (L1 completion)."""
    sched = ContactSchedule(intervals=[ContactInterval("left_foot", 0, 6, "ground"),
                                        ContactInterval("left_hand", 0, 6, "object")])
    sm = ContactStateMachine(sched, 6)
    sm.commit("left_foot", 0, sole_world=np.zeros((5, 3)))
    assert sm.holding["left_foot"]
    sm.release("left_foot")
    assert not sm.holding["left_foot"] and sm.anchors["left_foot"] is None
    # re-commit works after the reference interval is still active
    sm.commit("left_foot", 1, sole_world=np.ones((5, 3)) * 0.5)
    assert sm.holding["left_foot"] and np.allclose(sm.anchors["left_foot"], 0.5)
    # hands are unaffected by the foot release path
    sm.commit("left_hand", 1, hand_local=np.ones((3, 3)))
    assert sm.holding["left_hand"]


def test_contact_band_classifies_per_pair_geometry(pipeline) -> None:
    """A hand geom within the band upper gets band rows without any anchor;
    beyond it, noncontact clearance -- the "touching-but-not-anchored" state
    is geometric, not anchor-gated (P0.1 L1)."""
    from ireset.repair.scp import FrameContext

    def ctx(**kw):
        base = dict(t=0, box_pose=BOX_FAR, z_pred=np.zeros(36), z_prev=None,
                    ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                    mesh_fixed=None, hand_anchor={}, foot_anchor={},
                    prev_hand_local={}, prev_sole_world={})
        base.update(kw)
        return FrameContext(**base)

    cfg = pipeline.cfg
    # monkey-distance: robot_box pairs get the test distance, everything else
    # is far away (uniform fakes sort stably and the 48-pair cap would keep
    # only the self pairs that come first in ev.pairs)
    real = pipeline.ev.geom_distance
    box_pairs = {(ga, gb) for ga, gb, kind in pipeline.ev.pairs if kind == "robot_box"}

    def fake(ga, gb, distmax=1.0):
        d = pipeline._test_dist if (ga, gb) in box_pairs else 0.5
        return (d, np.zeros(6))

    c = ctx()   # no anchors; classification is purely geometric now
    stand = standing_scene(1)[0]
    for d, want_band in ((0.001, True), (cfg.contact_band_gap_m - 1e-4, True),
                         (cfg.contact_band_gap_m + 0.02, False)):
        pipeline.ev.forward(stand[0:36], stand[36:43])
        pipeline.solver._frame_pairs = None   # rescan pairs under the fake distance
        pipeline._test_dist = d
        pipeline.ev.geom_distance = fake
        try:
            rows = pipeline.solver._collision_rows(c)
        finally:
            pipeline.ev.geom_distance = real
        hand_rows = [r for r in rows if r[3] == cfg.contact_band_gap_m]
        assert bool(hand_rows) == want_band, (d, want_band, len(hand_rows))


def test_root_velocity_rows_are_norm_not_per_component(pipeline) -> None:
    """A diagonal root step passes per-component boxes but violates the norm
    gate; frame rows must match the validator's norm check (P0 fix)."""
    from ireset.repair.scp import FrameContext

    scene = standing_scene(2)
    z_prev = scene[0, 0:36].copy()
    z = scene[1, 0:36].copy()
    z[0:3] = z_prev[0:3] + [0.05, 0.05, 0.05]   # components <= 0.06, norm 0.0866
    ctx = FrameContext(t=1, box_pose=BOX_FAR, z_pred=z, z_prev=z_prev,
                       ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                       mesh_fixed=None, hand_anchor={}, foot_anchor={},
                       prev_hand_local={}, prev_sole_world={})
    pipeline.ev.forward(z, ctx.box_pose)
    val, lo, hi, scl = pipeline.solver.constraint_rows(z, ctx, trust=1.0, mode="accept")
    viol = pipeline.solver.violation(val, lo, hi, scl)
    assert viol > 0, "diagonal step must violate the norm velocity row"
    # axis-aligned step at the same per-component magnitude stays legal
    z2 = scene[1, 0:36].copy()
    z2[0] += 0.05
    ctx2 = FrameContext(t=1, box_pose=BOX_FAR, z_pred=z2, z_prev=z_prev,
                        ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                        mesh_fixed=None, hand_anchor={}, foot_anchor={},
                        prev_hand_local={}, prev_sole_world={})
    pipeline.ev.forward(z2, ctx2.box_pose)
    val2, lo2, hi2, scl2 = pipeline.solver.constraint_rows(z2, ctx2, trust=1.0, mode="accept")
    assert pipeline.solver.violation(val2, lo2, hi2, scl2) == 0.0
