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


# --- stage 1: dynamic collision-pair activation (m2_m3_completion_plan) ----------


def test_discovered_pair_is_carried_into_next_linearization(pipeline) -> None:
    """A pair that an endpoint check found violating OUTSIDE the attempt's
    frozen candidate set must enter the next solve attempt's rows (stage 1:
    the solve may no longer fail on a blind spot it cannot see)."""
    from ireset.repair.scp import FrameContext

    scene = standing_scene(1)[0]
    box = BOX_FAR.copy()
    ctx = FrameContext(t=0, box_pose=box, z_pred=scene[0:36], z_prev=None,
                       ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                       mesh_fixed=None, hand_anchor={}, foot_anchor={},
                       prev_hand_local={}, prev_sole_world={})
    solver = pipeline.solver
    solver.clear_discovered_pairs()
    pipeline.ev.forward(scene[0:36], box)
    solver._frame_pairs = None
    rows0 = solver._collision_rows(ctx)
    ids0 = {pid for pid in solver.last_row_ids}
    # pick a robot_box pair that is FAR (not in the candidate set): the box
    # sits at (1.5, 0, 0.25), so any left-hand geom vs box qualifies
    far = [(ga, gb, kind) for ga, gb, kind in pipeline.ev.pairs
           if kind == "robot_box" and "left" in pipeline.ev.geom_name(ga)]
    assert far
    assert all((ga, gb) not in ids0 for ga, gb, _k in far)
    # pipeline-side carry-back with a fabricated violating row
    solver.note_discovered_pairs(far[:1])
    assert solver._extra_pairs, "discovered pair not recorded"
    pipeline.ev.forward(scene[0:36], box)
    solver._frame_pairs = None  # next linearization (what note_... scheduled)
    rows1 = solver._collision_rows(ctx)
    ids1 = {pid[:2] for pid in solver.last_row_ids}
    (ga, gb, _k) = far[0]
    assert (ga, gb) in ids1, "carried pair missing from next linearization rows"
    # the carried pair's row is a real constraint row now (noncontact bound)
    k = [i for i, pid in enumerate(solver.last_row_ids) if pid[:2] == (ga, gb)][0]
    assert rows1[k][2] == pytest.approx(pipeline.cfg.noncontact_clearance_m)
    # frame boundary clears the mechanism
    solver.clear_discovered_pairs()
    pipeline.ev.forward(scene[0:36], box)
    solver._frame_pairs = None
    solver._collision_rows(ctx)
    assert all(pid[:2] != (ga, gb) for pid in solver.last_row_ids)


def test_row_budget_reports_truncation_instead_of_silently_dropping(pipeline) -> None:
    """When candidates exceed the QP row budget the overflow must be
    REPORTED (last_truncated), matching the plan's 'no silent disappearance'
    requirement -- and extras survive while far candidates are dropped."""
    from dataclasses import replace as _replace

    from ireset.repair.scp import FrameContext

    scene = standing_scene(1)[0]
    ctx = FrameContext(t=0, box_pose=BOX_FAR, z_pred=scene[0:36], z_prev=None,
                       ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                       mesh_fixed=None, hand_anchor={}, foot_anchor={},
                       prev_hand_local={}, prev_sole_world={})
    solver = pipeline.solver
    cfg = pipeline.cfg
    solver.clear_discovered_pairs()
    # shrink the budget so the standing scene's ~56 active pairs truncate
    tight = _replace(cfg, max_active_pairs=10)
    solver.cfg = tight
    pipeline.ev.forward(scene[0:36], BOX_FAR)
    solver._frame_pairs = None
    try:
        solver._collision_rows(ctx)
        assert solver.last_truncated > 0
        n_rows = len(solver.last_row_ids)
        assert n_rows <= 10
        # a discovered pair must survive the tight budget: carried extras
        # are re-added as candidates and rank with the nearest
        far = [(ga, gb, kind) for ga, gb, kind in pipeline.ev.pairs if kind == "robot_box"]
        solver.note_discovered_pairs(far[:1])
        pipeline.ev.forward(scene[0:36], BOX_FAR)
        solver._frame_pairs = None
        rows = solver._collision_rows(ctx)
        (ga, gb, _k) = far[0]
        assert any(pid[:2] == (ga, gb) for pid in solver.last_row_ids), \
            "discovered pair dropped by row budget"
    finally:
        solver.cfg = cfg


def test_success_return_rescans_full_registry(pipeline) -> None:
    """Stage 1: a state that passes the FROZEN candidate set but violates a
    pair outside it must NOT be returned as accepted. solve_frame's
    successful return path rescans the full registry and re-judges."""
    from unittest.mock import patch

    from ireset.repair.scp import FrameContext

    scene = standing_scene(4)
    cond = self_condition(pipeline, scene)
    solver = pipeline.solver
    calls = {"n": 0}
    real_scan = solver._scan_pairs

    def counting_scan(full):
        if full:
            calls["n"] += 1
        return real_scan(full)

    with patch.object(solver, "_scan_pairs", side_effect=counting_scan):
        res = pipeline.repair_trajectory(scene, cond)
    assert res.accepted
    # frame 0 skip + 3 repaired frames x (per-iteration rescans happen only
    # on the success path) => the full scan must have run at least once per
    # repaired frame beyond the initial linearization scan
    assert calls["n"] >= 4, calls


def test_probe_violating_pair_reflected_in_retry(pipeline) -> None:
    """Stage 1 convergence: when the prediction's endpoint is illegal but a
    legal alternative exists, the probe-rejected chord's violating pair is
    carried back and the retry CONVERGES to the legal state (previously the
    solver was blind to exactly the pair its checks rejected on). The
    accepted frame must stay on the legal side of the box wall."""
    from dataclasses import replace as _replace

    from ireset.embodied.contact_sites import ContactInterval, ContactSchedule, SMPLH_INDEX
    from ireset.repair.context import ReferenceCondition as RC
    from test_validation import _box_beside_left_hand

    T = 3
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
    human = np.zeros((T, 52, 3))
    for t in range(T):
        pipeline.ev.forward(scene[t, 0:36], scene[t, 36:43])
        kpts = pipeline.ev.site_positions(ids)
        for k, name in enumerate(KPT_TO_SMPLH):
            human[t, SMPLH_INDEX[KPT_TO_SMPLH[name]]] = kpts[k]
    sched = ContactSchedule(intervals=[ContactInterval("left_hand", 0, T, "object")])
    cond = RC(human, scene[:, 36:43].copy(), sched)
    pred = scene.copy()
    pred[1:, 0] += 0.16   # hand ends up INSIDE the box at the endpoint
    cfg = _replace(pipeline.cfg, regen_mode=False)
    pipe = RepairPipeline(cfg=cfg)
    pipe.validator.cfg = pipeline.validator.cfg
    pipe.verbose = True   # capture skip_check/attempt records
    res = pipe.repair_trajectory(pred, cond)
    assert res.accepted, (res.status, res.diagnostics.get("t_fail"))
    for t in range(T):
        pipe.ev.forward(res.trajectory[t, 0:36], res.trajectory[t, 36:43])
        dl = min(pipe.ev.geom_distance(g, pipe.ev.box_geom, distmax=1.0)[0]
                 for g in pipe.ev._hand_geoms["l"])
        assert dl > 0.002, (t, dl)   # outside the wall, on the near side
    # the prediction's illegal endpoints were caught before repair ran
    # (multi-start then converged from the legal predecessor)
    reasons = [r.get("reason") for r in pipe.frame_log
               if r.get("kind") == "skip_check" and not r.get("ok")]
    assert "state_constraints" in reasons


# --- stage 2: interpolation corridor semantics (m2_m3_completion_plan) -----------


def test_corridor_upper_is_frame_end_only_mid_states_free(pipeline) -> None:
    """Stage 2: the regen corridor upper binds ENDPOINT solutions only.
    A probe-mode row set (transition checks) keeps the anti-penetration
    floor but must NOT inherit the frame-end closing target."""
    from dataclasses import replace as _replace

    from ireset.repair.scp import FrameContext, INF

    def ctx(ref_hold, prev_dist):
        return FrameContext(
            t=1, box_pose=BOX_FAR, z_pred=np.zeros(36), z_prev=np.zeros(36),
            ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
            mesh_fixed=None, hand_anchor={}, foot_anchor={},
            prev_hand_local={}, prev_sole_world={}, box_pose_prev=BOX_FAR,
            ref_hold=ref_hold, prev_hand_dist=prev_dist, frame_motion_m=0.02)

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
    pipeline.ev.forward(stand[0:36], stand[36:43])
    pipeline.solver._frame_pairs = None
    pipeline.solver.cfg = cfg
    pipeline.ev.geom_distance = fake
    try:
        rows_solve = pipeline.solver._collision_rows(ctx({"left_hand": True}, {"l": 0.030}))
        rows_probe = pipeline.solver._collision_rows(
            ctx({"left_hand": True}, {"l": 0.030}), band_margin_m=0.0, probe_mode=True)
    finally:
        pipeline.ev.geom_distance = real
        pipeline.solver.cfg = pipeline.cfg
    solve_regen = [r for r in rows_solve if r[3] not in
                   (cfg.contact_band_gap_m, cfg.foot_contact_gap_m) and r[3] < INF / 2
                   and r[2] > -cfg.contact_band_pen_m - 1e-9]
    probe_regen = [r for r in rows_probe
                   if abs(r[2] - (-cfg.contact_band_pen_m)) < 1e-9
                   and r[3] > INF / 2]
    assert len(solve_regen) == 1 and solve_regen[0][3] < 0.030, \
        "solve rows must carry the frame-end corridor upper"
    assert len(probe_regen) == 1, \
        "probe rows must keep the regen row for the penetration floor"
    assert probe_regen[0][3] > INF / 2, "probe regen upper must be released (INF)"


def test_legal_approach_30mm_to_24mm_not_rejected_by_midpoint(pipeline) -> None:
    """Plan §5 reproduction: an endpoint closing 30mm -> 24mm (legal, inside
    the corridor at both ends) must pass the transition probe -- the 27mm
    midpoint previously read the 24.178mm frame-end upper and failed."""
    from dataclasses import replace as _replace

    import mujoco as mj

    from ireset.embodied.contact_sites import ContactInterval, ContactSchedule, SMPLH_INDEX
    from ireset.repair.context import ReferenceCondition as RC
    from ireset.repair.scp import FrameContext

    # box pose beside the LEFT hand at a controlled gap: place box surface
    # 30mm from the closest left-hand geom at frame 0, 24mm at frame 1
    h = _root_height_for_tangent(pipeline.ev)
    q0 = np.zeros(43)
    q0[2] = h
    q0[3] = 1.0
    pipe = pipeline

    def hand_gap_for(box_pose):
        pipe.ev.forward(q0[0:36], box_pose)
        return min(pipe.ev.geom_distance(g, pipe.ev.box_geom, distmax=1.0)[0]
                   for g in pipe.ev._hand_geoms["l"])

    # binary-search the box x-offset placing the nearest hand geom at
    # 30mm (frame 0) and 24mm (frame 1) from the box surface
    pipe.ev.forward(q0[0:36], np.array([1.5, 0, 0.25, 1, 0, 0, 0]))
    hx = pipe.ev.data.geom_xpos[pipe.ev._hand_geoms["l"][0]].copy()

    def box_at(off):
        return np.array([hx[0] + off, hx[1], hx[2], 1.0, 0, 0, 0])

    def gap_at(off):
        return hand_gap_for(box_at(off))

    found = {}
    for target in (0.030, 0.024):
        lo, hi = 0.05, 0.8
        glo, ghi = gap_at(lo), gap_at(hi)
        assert glo < target < ghi, (target, glo, ghi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            g = gap_at(mid)
            if (glo - target) * (g - target) <= 0:
                hi, ghi = mid, g
            else:
                lo, glo = mid, g
        found[target] = box_at(0.5 * (lo + hi))
    box0, box1 = found[0.030], found[0.024]
    T = 2
    scene = np.zeros((T, 43))
    scene[:, 0:3] = [0.0, 0.0, h]
    scene[:, 3] = 1.0
    scene[0, 36:43] = box0
    scene[1, 36:43] = box1
    ids = [mj.mj_name2id(pipe.ev.model, mj.mjtObj.mjOBJ_SITE, n) for n in KPT_TO_SMPLH]
    human = np.zeros((T, 52, 3))
    for t in range(T):
        pipe.ev.forward(scene[t, 0:36], scene[t, 36:43])
        kpts = pipe.ev.site_positions(ids)
        for k, name in enumerate(KPT_TO_SMPLH):
            human[t, SMPLH_INDEX[KPT_TO_SMPLH[name]]] = kpts[k]
    sched = ContactSchedule(intervals=[ContactInterval("left_hand", 1, T, "object")])
    cond = RC(human, scene[:, 36:43].copy(), sched)
    # the prediction IS the legal solution: frame 0 at 30mm (interval not
    # yet active), frame 1 at 24mm inside the closing corridor
    # (30 - rate*dt - margin ~= 24.5mm >= 24mm). With the stage-2 semantics
    # the 27mm midpoint carries no frame-end upper and must pass; the old
    # code read the 24.178mm frame-end bound there and rejected.
    cfg = _replace(pipe.cfg, regen_mode=True)
    p2 = RepairPipeline(cfg=cfg)
    p2.validator.cfg = pipe.validator.cfg
    res = p2.repair_trajectory(scene, cond)
    assert res.accepted, (res.status, res.diagnostics.get("t_fail"),
                          res.diagnostics.get("skip_reason"))
