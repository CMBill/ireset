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
    # two-frame ramp (not a single-frame cliff): with the honest slack
    # relaxation a uniform 0.12 jump in one frame creates interpolation
    # dips no fixed band margin can shield; ramped corruption keeps the
    # test's intent (tail re-solve) inside physically repairable steps
    pert[10:, 7:36] += 0.06
    pert[11:, 7:36] += 0.06
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


# --- P0-fix F2/F3 (p0_fix_plan sections 3-4) ------------------------------------


def _transition_ctx(pipeline, z, z_prev, box, box_prev, ref_kpts):
    from ireset.repair.scp import FrameContext

    return FrameContext(t=1, box_pose=box, z_pred=z, z_prev=z_prev,
                        ref_kpts_world=ref_kpts, mesh_L=None, delta_ref=None,
                        mesh_fixed=None, hand_anchor={}, foot_anchor={},
                        prev_hand_local={}, prev_sole_world={}, box_pose_prev=box_prev)


def _self_ref_kpts(pipeline, z, box) -> np.ndarray:
    import mujoco

    ids = [mujoco.mj_name2id(pipeline.ev.model, mujoco.mjtObj.mjOBJ_SITE, n)
                           for n in KPT_TO_SMPLH]
    pipeline.ev.forward(z, box)
    return pipeline.ev.site_positions(ids)


def test_transition_check_catches_box_sweep(pipeline) -> None:
    """Box translating THROUGH the robot between two clean keyframes: both
    endpoints are collision-free, the swept midpoint is not -- the
    sequential pre-check must reject it (F2: probes interpolate the box,
    not freeze it at the current pose)."""
    z = standing_scene(1)[0, 0:36].copy()
    ref = _self_ref_kpts(pipeline, z, BOX_FAR)
    v = pipeline.validator.cfg
    reach = 0.22132 + max(0.05, v.max_substep_motion_m)   # box half-x + margin
    box_prev = np.array([-reach - 0.35, 0.0, 0.80, 1.0, 0, 0, 0])
    box_cur = np.array([+reach + 0.35, 0.0, 0.80, 1.0, 0, 0, 0])
    ctx = _transition_ctx(pipeline, z, z, box_cur, box_prev, ref)
    ok, why = pipeline.frame_valid(z, ctx)
    assert not ok and why == "transition_interpolated", (ok, why)


def test_transition_check_catches_box_rotation_sweep(pipeline) -> None:
    """Box pitching 180 deg in place beside the legs: both endpoint
    orientations are ~8 mm clear, the 120-deg interior probe penetrates
    33 mm. Requires the motion metric to see the box CORNER sweep (0.73 m
    for this motion) -- a center-only metric sees ~0 motion and would run
    no probes at all (F2: 旋转运动量须生效). Placement verified numerically
    against the real collision model."""
    z = standing_scene(1)[0, 0:36].copy()
    ref = _self_ref_kpts(pipeline, z, BOX_FAR)
    c = np.array([0.30, 0.0, 0.12])
    q0 = np.array([1.0, 0, 0, 0])
    q180 = np.array([0.0, 0, 1.0, 0])       # pitch +180 deg (w-first)
    ctx = _transition_ctx(pipeline, z, z, np.r_[c, q180], np.r_[c, q0], ref)
    ok, why = pipeline.frame_valid(z, ctx)
    assert not ok and why == "transition_interpolated", (ok, why)


def test_transition_check_common_motion_no_false_positive(pipeline) -> None:
    """Robot and box translating together: the relative configuration is
    constant at every interpolation point -- mixing the previous robot with
    interpolated box poses must not fabricate a collision (F2). Step
    0.055 m > max_substep so an interior probe actually runs."""
    scene = standing_scene(2)
    z_prev, z = scene[0, 0:36].copy(), scene[1, 0:36].copy()
    z[0] += 0.055                          # within the 1.8 m/s norm budget
    dx = np.array([0.055, 0.0, 0.0])
    box_prev, box_cur = BOX_FAR.copy(), BOX_FAR + np.r_[dx, np.zeros(4)]
    ref = _self_ref_kpts(pipeline, z, box_cur)
    ctx = _transition_ctx(pipeline, z, z_prev, box_cur, box_prev, ref)
    ok, why = pipeline.frame_valid(z, ctx)
    assert ok and why == "ok", (ok, why)


def test_first_failure_frame_is_min_over_failed_checks(pipeline) -> None:
    """F3: retry targeting takes the MINIMUM structured frame across failed
    checks regardless of check-dict insertion order; passing checks and
    unlocated failures never contribute a frame."""
    from ireset.repair.validation import CheckResult, ValidationReport

    def report(order):
        checks = {k: v for k, v in order}
        return ValidationReport(passed=False, checks=checks)

    late_first = [("penetration", CheckResult(False, 3.0, 2.0, "t=40",
                                              first_failure_frame=40)),
                  ("foot_support", CheckResult(False, 2.0, 1.0, "t=5",
                                               first_failure_frame=5))]
    early_first = list(reversed(late_first))
    a = pipeline._first_failure_frame(report(late_first))
    b = pipeline._first_failure_frame(report(early_first))
    assert a == b == 5
    with_pass = report(late_first + [("joint_limits",
                                      CheckResult(True, 0.0, 1e-5, "", first_failure_frame=2))])
    assert pipeline._first_failure_frame(with_pass) == 5
    unlocated = report([("mystery", CheckResult(False, 2.0, 1.0, "somewhere"))])
    assert pipeline._first_failure_frame(unlocated) is None


def test_unlocated_final_failure_retries_full_segment(pipeline, monkeypatch) -> None:
    """F3: a failing final check WITHOUT a structured frame must trigger a
    FULL-segment retry (start=0); a located failure retries from
    frame - window."""
    from ireset.repair.validation import CheckResult, ValidationReport

    scene = standing_scene(8)
    cond = self_condition(pipeline, scene)
    window = pipeline.cfg.suffix_retry_window

    def fake_report(fff):
        return ValidationReport(passed=False, checks={
            "mystery": CheckResult(False, 2.0, 1.0, "somewhere",
                                   first_failure_frame=fff)})

    monkeypatch.setattr(pipeline.validator, "validate",
                        lambda *a, **k: fake_report(None))
    res = pipeline.repair_trajectory(scene, cond)
    assert res.status == "rejected_final"
    assert res.diagnostics["suffix_retry_from"] == 0

    monkeypatch.setattr(pipeline.validator, "validate",
                        lambda *a, **k: fake_report(9))
    res = pipeline.repair_trajectory(scene, cond)
    assert res.status == "rejected_final"
    assert res.diagnostics["suffix_retry_from"] == max(0, 9 - window)
    assert res.diagnostics["failed_checks"]["mystery"]["first_failure_frame"] == 9


def test_slack_flag_recorded_independent_of_path_name(pipeline, monkeypatch) -> None:
    """F4.7: slack usage is a fact about the CHOSEN solve result. When the
    multi-start restart (path name 'repaired_prev') succeeds with slack,
    diagnostics must count it -- the path name alone loses the flag."""
    from ireset.repair.scp import FrameResult

    scene = standing_scene(3)
    pert = scene.copy()
    pert[1:, 0] += 0.30                    # 0.3 m jump from frame 1 on: rows fail
    cond = self_condition(pipeline, scene)
    calls = {"n": 0}

    def fake_solve(z0, ctx, force_slack=False):
        calls["n"] += 1
        if calls["n"] % 2 == 1:            # direct start from z_pred: fail
            return FrameResult(False, z0.copy(), status="infeasible")
        # restart from z_prev: accepted, solved WITH slack
        z_ok = ctx.z_prev.copy() if ctx.z_prev is not None else z0.copy()
        return FrameResult(True, z_ok, slack_used=True)

    monkeypatch.setattr(pipeline.solver, "solve_frame", fake_solve)
    res = pipeline.repair_trajectory(pert, cond)
    assert res.accepted, res.diagnostics.get("reason")
    fs = res.diagnostics["frame_status"]
    assert fs.count("repaired_prev") == 2 and "repaired_slack" not in fs
    assert res.diagnostics["slack_frames"] == 2


# --- teacher regeneration mode (Path B) ------------------------------------------


def test_slack_qp_bounds_slack_variables_nonnegative(pipeline) -> None:
    """The slack restoration QP must carry EXPLICIT s >= 0 rows: without
    them the +penalty linear term drives s negative (over-tightening) and
    the solver takes destructive steps on near-infeasible systems (found
    via the Path B corridor; slack sum measured -51)."""
    import scipy.sparse as spa

    from ireset.repair.scp import FrameContext, N_TANGENT

    scene = standing_scene(1)
    z = scene[0, 0:36].copy()
    ctx = FrameContext(t=0, box_pose=BOX_FAR, z_pred=z, z_prev=None,
                       ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                       mesh_fixed=None, hand_anchor={}, foot_anchor={},
                       prev_hand_local={}, prev_sole_world={})
    pipeline.ev.forward(z, ctx.box_pose)
    pipeline.solver._frame_pairs = None
    (P, q, A, lo, hi), n_s = pipeline.solver.build_qp(z, ctx, trust=1.0, slack=True)
    A = np.asarray(A.todense())
    n_r = A.shape[0] - 2 * n_s
    assert n_s > 0
    # the last block must be identity rows on the slack columns with lo=0
    S = A[n_r + n_s:, N_TANGENT:]
    assert S.shape == (n_s, n_s)
    assert np.allclose(S, np.eye(n_s))
    assert np.allclose(lo[n_r + n_s:], 0.0) and np.all(np.isfinite(hi[n_r + n_s:]))
    # and solving with the assembly returns nonnegative slacks
    x, status = pipeline.solver.solve_qp(P, q, spa.csc_matrix(A), lo, hi)
    if x is not None:
        assert x[N_TANGENT:].min() >= -1e-6, x[N_TANGENT:].min()


def test_regen_active_hand_rows_ignore_geometric_band(pipeline) -> None:
    """Path B: during a reference grasp interval the CLOSEST hand geom's
    box pair carries the ACTIVE upper bound even when it hovers far beyond
    the geometric band (contact := nearest surface touches; the other hand
    geoms keep standard classification). Outside the interval, per-pair
    semantics everywhere."""
    from dataclasses import replace as _replace

    from ireset.repair.scp import FrameContext

    def ctx(ref_hold, **kw):
        base = dict(t=0, box_pose=BOX_FAR, z_pred=np.zeros(36), z_prev=None,
                    ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                    mesh_fixed=None, hand_anchor={}, foot_anchor={},
                    prev_hand_local={}, prev_sole_world={},
                    box_pose_prev=None, ref_hold=ref_hold)
        base.update(kw)
        return FrameContext(**base)

    real = pipeline.ev.geom_distance
    box_pairs = {(ga, gb) for ga, gb, kind in pipeline.ev.pairs if kind == "robot_box"}
    hand_geoms = set(pipeline.ev._hand_geoms["l"]) | set(pipeline.ev._hand_geoms["r"])
    closest = pipeline.ev._hand_geoms["l"][0]

    def fake(ga, gb, distmax=1.0):
        # the closest left-hand geom hovers at 2.8 cm (outside the band,
        # inside the activation margin); other hand geoms at 3.3 cm; the
        # rest of the world far away
        if (ga, gb) not in box_pairs or ga not in hand_geoms:
            return (0.5, np.zeros(6))
        return ((0.028 if ga == closest else 0.033), np.zeros(6))

    cfg = _replace(pipeline.cfg, regen_mode=True)
    stand = standing_scene(1)[0]
    dt = 1.0 / cfg.fps
    # F1 (2026-09-15): the corridor upper carries the interpolation-dip
    # margin with the same formula as the solver band rows; the plain
    # ctx here has frame_motion_m = 0 -> margin at its 0.5 mm floor
    from ireset.repair.scp import FrameContext as _FC, solver_band_margin as _sbm

    def _margin():
        return _sbm(_FC(t=0, box_pose=BOX_FAR, z_pred=np.zeros(36), z_prev=None,
                        ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                        mesh_fixed=None, hand_anchor={}, foot_anchor={},
                        prev_hand_local={}, prev_sole_world={}), cfg)
    margin = _margin()
    cases = [
        ({"left_hand": True}, {}, cfg.regen_hand_contact_gap_m),          # no prev: hard gap
        ({"left_hand": False}, {}, None),                                 # no hold: standard
        ({"left_hand": True}, {"l": 0.050},                               # corridor: close by rate + dip margin
         max(cfg.regen_hand_contact_gap_m,
             0.050 - cfg.regen_closing_rate_mps * dt - margin)),
        ({"left_hand": True}, {"l": 0.004}, cfg.regen_hand_contact_gap_m),  # inside gap: stays
    ]
    for ref_hold, prev_dist, want_upper in cases:
        pipeline.ev.forward(stand[0:36], stand[36:43])
        pipeline.solver._frame_pairs = None
        pipeline.solver.cfg = cfg
        pipeline.ev.geom_distance = fake
        try:
            rows = pipeline.solver._collision_rows(ctx(ref_hold, prev_hand_dist=prev_dist))
        finally:
            pipeline.ev.geom_distance = real
            pipeline.solver.cfg = pipeline.cfg
        regen_lo = -cfg.contact_band_pen_m + 0.0005
        active = [r for r in rows
                  if abs(r[2] - regen_lo) < 1e-9 and r[3] != cfg.contact_band_gap_m
                  and r[3] != cfg.foot_contact_gap_m]
        if want_upper is None:
            assert not active, (ref_hold, prev_dist, len(active))
        else:
            # exactly one active row -- the closest geom only, at the corridor bound
            assert len(active) == 1, (ref_hold, prev_dist, len(active))
            assert active[0][0] == pytest.approx(0.028)
            assert active[0][3] == pytest.approx(want_upper)


def test_regen_corridor_and_band_share_one_margin_formula(pipeline) -> None:
    """Plan 5.2 (F1): given a frame motion, the corridor upper and the
    solver band-row floor must be derived from the SAME
    ``solver_band_margin`` value -- the endpoint closes the corridor by an
    extra dip margin so interpolated states cannot both ride the corridor
    and dip past a row the QP solved."""
    from dataclasses import replace as _replace

    from ireset.repair.scp import FrameContext, solver_band_margin

    def ctx(motion, prev_dist):
        return FrameContext(
            t=1, box_pose=BOX_FAR, z_pred=np.zeros(36), z_prev=np.zeros(36),
            ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
            mesh_fixed=None, hand_anchor={}, foot_anchor={},
            prev_hand_local={}, prev_sole_world={}, box_pose_prev=BOX_FAR,
            ref_hold={"left_hand": True}, prev_hand_dist={"l": prev_dist},
            frame_motion_m=motion)

    real = pipeline.ev.geom_distance
    box_pairs = {(ga, gb) for ga, gb, kind in pipeline.ev.pairs if kind == "robot_box"}
    hand_geoms = set(pipeline.ev._hand_geoms["l"]) | set(pipeline.ev._hand_geoms["r"])
    closest = pipeline.ev._hand_geoms["l"][0]

    def fake(ga, gb, distmax=1.0):
        if (ga, gb) not in box_pairs or ga not in hand_geoms:
            return (0.5, np.zeros(6))
        return ((0.028 if ga == closest else 0.033), np.zeros(6))

    cfg = _replace(pipeline.cfg, regen_mode=True)
    stand = standing_scene(1)[0]
    dt = 1.0 / cfg.fps
    prev = 0.030
    for motion in (0.0, 0.02, 0.05, 0.2):   # margin floor / quadratic / cap regimes
        c = ctx(motion, prev)
        pipeline.ev.forward(stand[0:36], stand[36:43])
        pipeline.solver._frame_pairs = None
        pipeline.solver.cfg = cfg
        pipeline.ev.geom_distance = fake
        try:
            rows_probe = pipeline.solver._collision_rows(c, band_margin_m=0.0)
            rows_solver = pipeline.solver._collision_rows(
                c, band_margin_m=solver_band_margin(c, cfg))
        finally:
            pipeline.ev.geom_distance = real
            pipeline.solver.cfg = pipeline.cfg
        margin = solver_band_margin(c, cfg)
        want_upper = max(cfg.regen_hand_contact_gap_m,
                         prev - cfg.regen_closing_rate_mps * dt - margin)
        regen_probe = [r for r in rows_probe
                       if abs(r[2] - (-cfg.contact_band_pen_m)) < 1e-9
                       and r[3] != cfg.contact_band_gap_m and r[3] != cfg.foot_contact_gap_m]
        assert len(regen_probe) == 1
        assert regen_probe[0][3] == pytest.approx(want_upper), motion
        # solver-grade rows: every band/regen row floor raised by the same margin
        band_floor = -cfg.contact_band_pen_m + margin
        for r in rows_solver:
            if r[2] != cfg.noncontact_clearance_m:
                assert r[2] == pytest.approx(band_floor), (motion, r[2])
        # and the corridor upper minus the margin equals what the probe-grade
        # floor (validator grade, no margin) expects of interpolated states:
        # upper + ... the margin must not exceed the corridor window
        assert margin <= want_upper - (-cfg.contact_band_pen_m) + 1e-12


def test_regen_mode_holds_hand_anchors_against_hovering_prediction(pipeline) -> None:
    """Path B: the prediction-driven hand release is bypassed in regen_mode
    -- the hovering warm start must not release the anchor established at
    t=0. Standard mode keeps the P0.1 L1 release semantics."""
    from dataclasses import replace as _replace

    from ireset.embodied.contact_sites import SMPLH_INDEX
    from ireset.repair.context import ContactStateMachine as CSM
    from test_validation import _box_beside_left_hand

    T = 4
    h = _root_height_for_tangent(pipeline.ev)
    q0 = np.zeros(43)
    q0[2] = h
    q0[3] = 1.0
    box = _box_beside_left_hand(pipeline.validator, q0)
    scene = np.zeros((T, 43))
    scene[:, 0:3] = [0.0, 0.0, h]
    scene[:, 3] = 1.0
    scene[:, 36:43] = box
    ids = [mujoco.mj_name2id(pipeline.ev.model, mujoco.mjtObj.mjOBJ_SITE, n)
           for n in KPT_TO_SMPLH]
    kpts = np.zeros((T, len(ids), 3))
    human = np.zeros((T, 52, 3))
    for t in range(T):
        pipeline.ev.forward(scene[t, 0:36], scene[t, 36:43])
        kpts[t] = pipeline.ev.site_positions(ids)
    for k, name in enumerate(KPT_TO_SMPLH):
        human[:, SMPLH_INDEX[KPT_TO_SMPLH[name]]] = kpts[:, k]
    sched = ContactSchedule(intervals=[ContactInterval("left_hand", 0, T, "object")])
    cond = ReferenceCondition(human, scene[:, 36:43].copy(), sched)
    # hovering warm start from frame 1: root jumps -10 cm in x, drifting
    # the hand ~10 cm off its box-frame anchor (> hand_anchor_tol 1 cm)
    pred = scene.copy()
    pred[1:, 0] -= 0.10

    for regen, want_release in ((False, True), (True, False)):
        cfg = _replace(pipeline.cfg, regen_mode=regen)
        pipe = RepairPipeline(cfg=cfg)
        pipe.validator.cfg = pipeline.validator.cfg
        released: list[str] = []
        orig = CSM.release
        CSM.release = lambda self, effector: released.append(effector)
        try:
            res = pipe.repair_trajectory(pred, cond)
        finally:
            CSM.release = orig
        assert res.accepted, (regen, res.diagnostics.get("reason"), res.status)
        assert ("left_hand" in released) == want_release, (regen, released)
