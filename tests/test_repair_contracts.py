"""F0/F1/F2 correctness contracts (m2_m3_stage_gate_fix_plan sections 3-5).

Regression tests for the defects confirmed by the 2026-09-17 reviews:

- F0: check results are separated from carry-back actions, verbose logging
  adds no checks/FK, attempt-level counters over ALL tries are recomputable
  from events, and a rejection reports its FINAL stage (never re-using the
  prediction's ``skip_reason``).
- F1: pair identity and row ROLE are separate (a carried-back ``regen`` row
  must not lose its registry category), the row budget de-duplicates regen
  placeholders and reports overflow identities, and every success return
  (early return, budget exhaustion, historical best) runs the uncapped
  full-registry review.
- F2: the QP linear/quadratic terms equal the true objective gradient /
  Gauss-Newton term of the two tracking residuals, including the SO(3)
  left-Jacobian inverse for the world-frame rotation residual, and the
  distance Jacobian stays valid at exact contact (collapsed fromto).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.kinematics.mujoco_adapter import retract_batch  # noqa: E402
from ireset.kinematics.rotation_adapter import (  # noqa: E402
    mat_to_quat_wxyz,
    so3_exp_np,
    so3_left_jacobian_inverse_np,
)
from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.scp import INF, FrameContext, tangent_error  # noqa: E402
from test_contacts import _root_height_for_tangent  # noqa: E402
from test_repair import BOX_FAR, self_condition, standing_scene  # noqa: E402

BOX_ON_TORSO = np.array([0.0, 0.0, 0.80, 1.0, 0, 0, 0])
EFFECTORS = ("left_hand", "right_hand", "left_foot", "right_foot")


@pytest.fixture(scope="module")
def pipeline() -> RepairPipeline:
    return RepairPipeline(cfg=RepairConfig())


@pytest.fixture(scope="module")
def regen_pipeline() -> RepairPipeline:
    return RepairPipeline(cfg=replace(RepairConfig(), regen_mode=True))


def _standing_ctx(box: np.ndarray, z_prev: np.ndarray | None = None,
                  t: int = 1, **extra) -> FrameContext:
    z = standing_scene(1)[0, 0:36].copy()
    return FrameContext(t=t, box_pose=box, z_pred=z, z_prev=z if z_prev is None else z_prev,
                        ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
                        mesh_fixed=None, hand_anchor={}, foot_anchor={},
                        prev_hand_local={}, prev_sole_world={}, **extra)


def _box_at_gap(ev, z, gid, gap: float, half: float = 1.0) -> np.ndarray:
    """Box pose whose surface sits ``gap`` from geom ``gid`` (bisection along
    the world x-axis through the geom center), w_first identity rotation."""
    ev.forward(z, BOX_FAR)
    p = ev.data.geom_xpos[gid].copy()
    lo, hi = 0.0, half
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cand = np.array([p[0] + mid, p[1], p[2], 1.0, 0, 0, 0])
        ev.forward(z, cand)
        if ev.geom_distance(gid, ev.box_geom, distmax=2.0)[0] < gap:
            lo = mid
        else:
            hi = mid
    return np.array([p[0] + 0.5 * (lo + hi), p[1], p[2], 1.0, 0, 0, 0])


def _regen_ctx(pipeline: RepairPipeline, gap: float = 0.030, hold=("left_hand",),
               prev_dist=None, gid=None) -> FrameContext:
    """FrameContext with an active reference grasp (Path B corridor) and the
    box placed ``gap`` from the nearest left-hand geom."""
    ev = pipeline.ev
    z = standing_scene(1)[0, 0:36].copy()
    ev.forward(z, BOX_FAR)
    gid = ev._hand_geoms["l"][2] if gid is None else gid
    box = _box_at_gap(ev, z, gid, gap)
    measured = ev.geom_distance(gid, ev.box_geom, distmax=2.0)[0]
    return FrameContext(
        t=1, box_pose=box, z_pred=z, z_prev=z,
        ref_kpts_world=np.zeros((15, 3)), mesh_L=None, delta_ref=None,
        mesh_fixed=None, hand_anchor={}, foot_anchor={},
        prev_hand_local={}, prev_sole_world={}, box_pose_prev=box,
        ref_hold={name: (name in hold) for name in EFFECTORS},
        prev_hand_dist=dict(prev_dist if prev_dist is not None
                            else {"l": measured, "r": measured}),
        frame_motion_m=0.02)


def _hand_box_pairs(ev) -> list[tuple[int, int, str]]:
    geoms = set(ev._hand_geoms["l"]) | set(ev._hand_geoms["r"])
    return [(ga, gb, k) for ga, gb, k in ev.pairs
            if k == "robot_box" and ga in geoms]


# --- F1: pair identity, corridor stability, row budget ---------------------------


def test_pair_identity_comes_from_registry_not_row_role(regen_pipeline) -> None:
    """A carried-back pair keeps its REGISTRY category: the ``regen`` label is
    a row role and used to overwrite ``robot_box``, after which the pair lost
    the active-contact selection and the contact-band classification."""
    solver = regen_pipeline.solver
    solver.clear_discovered_pairs()
    (ga, gb, true_kind) = _hand_box_pairs(regen_pipeline.ev)[0]
    assert true_kind == "robot_box"
    solver.note_discovered_pairs([(ga, gb, "regen")])
    assert solver._extra_pairs == [(ga, gb, "robot_box")]
    with pytest.raises(ValueError):
        solver.note_discovered_pairs([(0, 1, "robot_box")])   # unknown identity
    solver.clear_discovered_pairs()


def test_regen_corridor_survives_its_own_carry_back(regen_pipeline) -> None:
    """§3.2 reproduction: repeating the check on an UNCHANGED state must not
    let the contact requirement disappear. Before the fix the first carry-back
    relabelled the active pair as ``regen``; the next scan then skipped the
    nearest-hand-geom selection (kind != robot_box) and the corridor vanished
    even though the pose never moved (sub10 t=90)."""
    solver, ev = regen_pipeline.solver, regen_pipeline.ev
    solver.clear_discovered_pairs()
    ctx = _regen_ctx(regen_pipeline)
    ev.forward(ctx.z_pred, ctx.box_pose)
    seen = []
    for _ in range(3):
        solver._frame_pairs = None
        solver._collision_rows(ctx)
        regen_rows = [n for n in solver.last_row_pairs if n.endswith(":regen")]
        assert len(regen_rows) == 1, regen_rows
        seen.append({s: (c["geom"], round(c["dist_m"], 9), round(c["upper_m"], 9))
                     for s, c in solver.last_corridor.items()})
        solver.note_discovered_pairs(list(solver.last_row_ids))
    assert seen[0] == seen[1] == seen[2], seen
    solver.clear_discovered_pairs()


def test_check_bounds_stable_across_repeat_checks(regen_pipeline) -> None:
    """A check is a function of the STATE alone: two identical checks with a
    carry-back in between must return the same rows and bounds."""
    solver = regen_pipeline.solver
    ctx = _regen_ctx(regen_pipeline)
    solver.clear_discovered_pairs()
    solver._frame_pairs = None
    v1, l1, h1, _s1, names1, ids1 = solver.check_rows(ctx.z_pred, ctx)
    solver.note_discovered_pairs(ids1)
    v2, l2, h2, _s2, names2, _ids2 = solver.check_rows(ctx.z_pred, ctx)
    assert names1 == names2
    assert np.array_equal(l1, l2) and np.array_equal(h1, h2)
    assert np.allclose(v1, v2)
    solver.clear_discovered_pairs()


def test_carried_pair_keeps_contact_band_rows(regen_pipeline) -> None:
    """sub7 t=34 style: a pair a check just found violating keeps its
    registry classification on the next linearization -- a near hand/box pair
    must still be a CONTACT-band row, not a degraded clearance row."""
    solver, ev = regen_pipeline.solver, regen_pipeline.ev
    solver.clear_discovered_pairs()
    z = standing_scene(1)[0, 0:36].copy()
    pinky = ev._hand_geoms["l"][2]
    box = _box_at_gap(ev, z, pinky, gap=0.005)
    ctx = _standing_ctx(box, z_prev=z)
    ev.forward(z, box)
    dist = ev.geom_distance(pinky, ev.box_geom, distmax=2.0)[0]
    assert 0.0 < dist <= regen_pipeline.cfg.contact_band_gap_m, dist
    pair = (pinky, ev.box_geom)
    solver.note_discovered_pairs([pair])
    solver._frame_pairs = None
    rows = solver._collision_rows(ctx)
    k = [i for i, pid in enumerate(solver.last_row_ids) if pid[:2] == pair]
    assert k, "carried pair missing from the rows"
    assert solver.last_row_ids[k[0]][2] == "robot_box"
    cfg = regen_pipeline.cfg
    band_lo = -cfg.contact_band_pen_m + (0.0005 if cfg.regen_mode else 0.0)
    assert rows[k[0]][2] == pytest.approx(band_lo)
    assert rows[k[0]][3] == pytest.approx(cfg.contact_band_gap_m)
    solver.clear_discovered_pairs()


def _box_poses_by_nearest(ev, z, side: str = "l", max_gap: float = 0.05) -> dict:
    """Box poses (sphere grid around the hand centroid) grouped by which hand
    geom of ``side`` is nearest, keeping only poses within ``max_gap``."""
    out: dict[int, np.ndarray] = {}
    ev.forward(z, BOX_FAR)
    geoms = ev._hand_geoms[side]
    c = np.mean([ev.data.geom_xpos[g] for g in geoms], axis=0)
    for radius in (0.26, 0.28, 0.30):
        for i in range(1, 12):
            th = np.pi * i / 12
            for j in range(12):
                ph = 2.0 * np.pi * j / 12
                u = np.array([np.sin(th) * np.cos(ph),
                              np.sin(th) * np.sin(ph), np.cos(th)])
                box = np.r_[c + radius * u, [1.0, 0, 0, 0]]
                ev.forward(z, box)
                ds = {g: ev.geom_distance(g, ev.box_geom, distmax=2.0)[0]
                      for g in geoms}
                g = min(ds, key=ds.get)
                if ds[g] < max_gap and g not in out:
                    out[g] = box
    return out


def test_nearest_hand_geom_reselection_is_reported(regen_pipeline) -> None:
    """When a REAL state change re-selects the nearest hand geom the corridor
    records where it came from (re-selection must never be confused with the
    old label pollution)."""
    solver, ev = regen_pipeline.solver, regen_pipeline.ev
    solver.clear_discovered_pairs()
    z = standing_scene(1)[0, 0:36].copy()
    poses = _box_poses_by_nearest(ev, z)
    assert len(poses) >= 2, "need two hand geoms to be selectable"
    (g1, box1), (g2, box2) = list(poses.items())[:2]
    ctx1 = replace(_regen_ctx(regen_pipeline), box_pose=box1, box_pose_prev=box1)
    ev.forward(ctx1.z_pred, box1)
    solver._frame_pairs = None
    solver._collision_rows(ctx1)
    first = solver.last_corridor["l"]
    assert first["geom"] == ev.geom_name(g1)
    ctx2 = replace(ctx1, box_pose=box2, box_pose_prev=box2)
    ev.forward(ctx2.z_pred, box2)
    solver._frame_pairs = None
    solver._collision_rows(ctx2)
    second = solver.last_corridor["l"]
    assert second["geom"] == ev.geom_name(g2) != first["geom"]
    assert second.get("reselected_from") == first["geom"]
    solver.clear_discovered_pairs()


def test_row_budget_dedups_regen_and_reports_overflow_identities(
        regen_pipeline) -> None:
    """The budget must (a) count a pair that is BOTH the active regen row and
    a carried extra exactly once and (b) name the identities it dropped."""
    solver, ev = regen_pipeline.solver, regen_pipeline.ev
    solver.clear_discovered_pairs()
    ctx = _regen_ctx(regen_pipeline)
    ev.forward(ctx.z_pred, ctx.box_pose)
    solver._frame_pairs = None
    solver._collision_rows(ctx)
    regen_pair = [pid[:2] for pid, role in
                  zip(solver.last_row_ids, solver.last_row_roles) if role == "regen"]
    assert len(regen_pair) == 1
    solver.note_discovered_pairs([regen_pair[0]])       # same pair as the row
    old_cfg, solver.cfg = solver.cfg, replace(regen_pipeline.cfg, max_active_pairs=6)
    try:
        solver._frame_pairs = None
        rows = solver._collision_rows(ctx)
    finally:
        solver.cfg = old_cfg
    ids = [pid[:2] for pid in solver.last_row_ids]
    assert ids.count(regen_pair[0]) == 1, "regen placeholder double-counted"
    assert len(rows) <= 6
    assert solver.last_truncated == len(solver.last_truncated_ids) \
        == len(solver.last_truncated_names)
    if solver.last_truncated:
        assert all("~" in n for n in solver.last_truncated_names)
    solver.clear_discovered_pairs()


def test_truncation_reports_specific_identities(regen_pipeline) -> None:
    """Overflow must name the dropped pairs, never just log a count."""
    solver, ev = regen_pipeline.solver, regen_pipeline.ev
    solver.clear_discovered_pairs()
    ctx = _regen_ctx(regen_pipeline, hold=())
    ev.forward(ctx.z_pred, ctx.box_pose)
    old_cfg, solver.cfg = solver.cfg, replace(regen_pipeline.cfg, max_active_pairs=3)
    try:
        solver._frame_pairs = None
        solver._collision_rows(ctx)
    finally:
        solver.cfg = old_cfg
    assert solver.last_truncated > 0
    assert len(solver.last_truncated_ids) == solver.last_truncated
    assert all(isinstance(pid, tuple) and len(pid) == 3
               for pid in solver.last_truncated_ids)
    solver.clear_discovered_pairs()


# --- F1: unified success review ---------------------------------------------------


def test_success_review_catches_uncapped_violation_and_feeds_back(pipeline) -> None:
    """A state that violates registry pairs must come back False from the
    review with its violating identities pushed into the candidate set. The
    review is uncapped: the box placed on the torso violates far more pairs
    than the QP row budget would carry."""
    solver = pipeline.solver
    solver.clear_discovered_pairs()
    ctx = _standing_ctx(BOX_ON_TORSO)
    z = ctx.z_pred
    pipeline.ev.forward(z, ctx.box_pose)
    solver._frame_pairs = None
    rows = solver._collision_rows(ctx)
    assert len(rows) <= pipeline.cfg.max_active_pairs
    ok, info = solver.success_review(z, ctx)
    assert not ok
    assert info["violated_rows"], info
    assert solver._extra_pairs, "review violations must feed the next solve"
    carried = {(ga, gb) for ga, gb, _k in solver._extra_pairs}
    assert any("largebox" in n for n in info["violated_rows"])
    assert len(carried) >= 1
    solver.clear_discovered_pairs()


def test_best_feasible_return_is_reviewed(monkeypatch) -> None:
    """Budget exhaustion / trust collapse used to return the historical best
    after only a task check. It must go through the SAME review: when the
    review fails, the attempt is an explicit ``review_failed`` rejection."""
    import ireset.repair.scp as scp

    pipe = RepairPipeline(cfg=RepairConfig())
    calls = {"n": 0}
    real = scp.FrameSolver.success_review

    def failing_once(self, z, ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return False, {"viol": 1.0, "violated_rows": ["synthetic"], "rows": []}
        return real(self, z, ctx)

    monkeypatch.setattr(scp.FrameSolver, "success_review", failing_once)
    scene = standing_scene(3)
    rng = np.random.default_rng(5)
    lo, hi = pipe.spec.joint_range[:, 0], pipe.spec.joint_range[:, 1]
    pert = scene.copy()
    pert[:, 7:36] = np.clip(pert[:, 7:36] + rng.normal(0, 0.05, (3, 29)), lo, hi)
    res = pipe._run_sequential(pert, self_condition(pipe, scene))
    assert any(a["status"] == "review_failed"
               for a in res.diagnostics["attempts_detail"]), \
        [a["status"] for a in res.diagnostics["attempts_detail"]]
    assert calls["n"] >= 1


# --- F0: statistics, purity, rejection attribution --------------------------------


def test_failed_attempts_are_counted_and_attributed(pipeline) -> None:
    """A fully rejected segment used to report ``qp_calls_sum = 0`` (only
    adopted successes were accumulated) and re-used the prediction's
    ``skip_reason`` as the failure reason."""
    scene = standing_scene(3)
    scene[2, 36:43] = BOX_ON_TORSO
    res = pipeline.repair_trajectory(scene, self_condition(pipeline, scene))
    assert not res.accepted and res.status == "rejected_frame", res.diagnostics
    d = res.diagnostics
    assert d["t_fail"] == 2
    detail = d["attempts_detail"]
    assert d["attempts"] == len(detail) >= 1
    assert d["qp_calls_total"] == int(sum(a["qp_calls"] for a in detail))
    assert d["iterations_total"] == int(sum(a["iterations"] for a in detail))
    assert d["qp_calls_total"] > 0, "failed attempts must not report 0 QP calls"
    assert d["slack_attempts"] == sum(1 for a in detail if a["slack_used"])
    assert d["unique_processed_frames"] <= d["processed_frame_visits"]
    assert d["input_frames"] == 3
    # structured rejection: final stage + last attempt, not the initial skip
    assert d["reason_source"] == "last_solve_attempt"
    assert d["final_stage"] in ("state_constraints", "transition_interpolated",
                               "task_quality")
    assert d["last_attempt"]["t"] == d["t_fail"]
    assert d["skip_reason"] is not None
    for a in detail:
        assert {"trust", "last_viol", "violated_rows", "truncated_rows",
                "source", "pass", "last_qp_status"} <= set(a)
    assert d["timing_stats"]["p50_ms"] is not None
    assert d["timing_stats"]["max_ms"] >= d["timing_stats"]["p50_ms"]


def test_probe_failure_snapshot_replays_the_violation(pipeline) -> None:
    """F0: a transition rejection carries a replayable snapshot (interpolation
    fraction, mid state, mid box pose). Re-checking the snapshot must
    reproduce the violation -- the evidence is the exact probed state, not a
    re-derivation of the subdivision."""
    from test_repair import _self_ref_kpts, _transition_ctx

    z = standing_scene(1)[0, 0:36].copy()
    ref = _self_ref_kpts(pipeline, z, BOX_FAR)
    v = pipeline.validator.cfg
    reach = 0.22132 + max(0.05, v.max_substep_motion_m)
    box_prev = np.array([-reach - 0.35, 0.0, 0.80, 1.0, 0, 0, 0])
    box_cur = np.array([+reach + 0.35, 0.0, 0.80, 1.0, 0, 0, 0])
    ctx = _transition_ctx(pipeline, z, z, box_cur, box_prev, ref)
    chk = pipeline.frame_valid(z, ctx)
    assert not chk.ok and chk.reason == "transition_interpolated"
    assert chk.probes, "a failing probe must leave a snapshot"
    rec = chk.probes[0]
    assert rec["alpha"] == rec["i"] / rec["n_sub"]
    assert len(rec["q_mid"]) == 36 and len(rec["box_pose_mid"]) == 7
    assert rec["rows"], "snapshot must carry the violating rows"
    # replay: same mid state + mid box pose -> same violation, same pair
    mid_ctx = replace(ctx, box_pose=np.asarray(rec["box_pose_mid"]))
    pipeline.solver._frame_pairs = None
    val, lo, hi, scl, names, _ids = pipeline.solver.check_rows(
        np.asarray(rec["q_mid"]), mid_ctx, band_margin_m=0.0, probe_mode=True)
    n_coll = 99 + len(names)
    viol = pipeline.solver.violation(val[99:n_coll], lo[99:n_coll],
                                     hi[99:n_coll], scl[99:n_coll])
    assert viol == pytest.approx(rec["viol"], rel=1e-6, abs=1e-9)
    assert names[0] in [r["pair"] for r in rec["rows"]]


def test_quiet_and_verbose_runs_are_identical(pipeline) -> None:
    """Verbose logging is pure observation: same states, same counters, same
    number of FK forwards as a quiet run. The old log called
    ``_frame_result_ok`` again, adding a check plus a carry-back side effect."""
    scene = standing_scene(8)
    rng = np.random.default_rng(11)
    lo, hi = pipeline.spec.joint_range[:, 0], pipeline.spec.joint_range[:, 1]
    pert = scene.copy()
    pert[:, 7:36] = np.clip(pert[:, 7:36] + rng.normal(0, 0.06, (8, 29)), lo, hi)
    cond = self_condition(pipeline, scene)

    out = {}
    for verbose in (False, True):
        pipe = RepairPipeline(cfg=RepairConfig())
        pipe.validator.cfg = pipeline.validator.cfg
        pipe.verbose = verbose
        ev, n = pipe.ev, {"fwd": 0}
        real_forward = ev.forward

        def counting_forward(z, box, _real=real_forward, _n=n):
            _n["fwd"] += 1
            return _real(z, box)

        ev.forward = counting_forward
        res = pipe.repair_trajectory(pert, cond)
        out[verbose] = (res, n["fwd"], len(pipe.frame_log))
    res_q, fwd_q, _ = out[False]
    res_v, fwd_v, n_log = out[True]
    assert res_q.accepted == res_v.accepted
    assert fwd_q == fwd_v, f"verbose changed the FK count ({fwd_q} vs {fwd_v})"
    for key in ("attempts", "qp_calls_total", "iterations_total", "skipped_frames",
                "backtracks", "slack_attempts", "processed_frame_visits"):
        assert res_q.diagnostics[key] == res_v.diagnostics[key], key
    if res_q.accepted:
        assert np.array_equal(res_q.trajectory, res_v.trajectory)
    assert n_log >= res_v.diagnostics["attempts"] + res_v.diagnostics["input_frames"]


def test_suffix_retry_keeps_first_pass_costs(pipeline) -> None:
    """A suffix retry must not overwrite the first pass: both passes report
    their own totals and the retry is labelled ``trajectory_pass=1``."""
    scene = standing_scene(12)
    cond = self_condition(pipeline, scene)
    pred = scene.copy()
    j = 7
    pred[9:12, j] += 0.9
    pred[9:12, j] = np.clip(pred[9:12, j],
                            pipeline.spec.joint_range[j - 7, 0],
                            pipeline.spec.joint_range[j - 7, 1])
    res = pipeline.repair_trajectory(pred, cond)
    if "suffix_retries" not in res.diagnostics:
        pytest.skip("scenario did not need a suffix retry")
    assert res.diagnostics["suffix_retries"] == 1
    assert res.diagnostics["trajectory_pass"] == 1
    first = res.diagnostics["first_attempt"]
    assert first["attempts"] >= 0 and first["qp_calls_total"] >= 0
    assert first["trajectory_pass"] == 0
    assert res.diagnostics["processed_frame_visits"] >= res.diagnostics["input_frames"]


# --- F2: QP model vs objective ----------------------------------------------------


def _fd_grad(f, z0: np.ndarray, model, h: float = 1e-6) -> np.ndarray:
    g = np.zeros(35)
    for k in range(35):
        e = np.zeros(35)
        e[k] = h
        zp = retract_batch(model, z0[None], e[None])[0]
        zm = retract_batch(model, z0[None], (-e)[None])[0]
        g[k] = (f(zp) - f(zm)) / (2.0 * h)
    return g


def _tracking_ctx(pipeline: RepairPipeline, rot: bool = True,
                  z_prev: np.ndarray | None = None) -> tuple:
    """``(z, ctx)`` with the current state ``z`` offset from BOTH references:
    a non-zero prediction residual (translation + joint, plus a world-frame
    root rotation when ``rot``) and a non-zero temporal residual."""
    base = standing_scene(1)[0, 0:36].copy()
    z = base.copy()
    if rot:
        ang = np.array([[0.02, -0.015, 0.01]])
        z[3:7] = mat_to_quat_wxyz(so3_exp_np(ang)[0])
    z_pred = base.copy()
    z_pred[0:3] += np.array([0.003, -0.002, 0.0015])
    z_pred[7] += 0.01          # a JOINT (index 6 is a quaternion slot)
    if z_prev is None:
        z_prev = base.copy()
        z_prev[0:3] += np.array([0.001, 0.0, 0.0])
    ctx = FrameContext(t=1, box_pose=BOX_FAR.copy(), z_pred=z_pred, z_prev=z_prev,
                       ref_kpts_world=np.zeros((15, 3)), mesh_L=None,
                       delta_ref=None, mesh_fixed=None, hand_anchor={},
                       foot_anchor={}, prev_hand_local={}, prev_sole_world={})
    return z, ctx


def test_qp_linear_term_equals_objective_gradient(pipeline) -> None:
    """F2 core contract: q == d objective / d xi at the linearization point
    (isolated tracking terms: no mesh, no hand term, regen off). The old code
    multiplied the ALREADY normalized residual by D^-2 again, giving a 50x
    (root translation) / 10x (joint) too-large linear term."""
    solver, ev = pipeline.solver, pipeline.ev
    for rot in (False, True):
        z, ctx = _tracking_ctx(pipeline, rot=rot)
        ev.forward(z, ctx.box_pose)
        (P, q, _A, _lo, _hi), _ = solver.build_qp(z, ctx, trust=1.0, slack=False)
        P = np.asarray(P.todense())

        def f(zz, ctx=ctx):
            return solver.objective(solver.evaluate(zz, ctx))

        for h in (1e-5, 1e-6, 1e-7):
            g = _fd_grad(f, z, ev.model, h)
            assert np.allclose(q, g, atol=1e-6, rtol=1e-4), (rot, h, q - g)
        # closed form for the additive components (prediction + temporal):
        # q = 2 w_p e_pred / D^2 + 2 w_t e_temp / D^2
        cfg = pipeline.cfg
        d2 = solver.D ** 2
        e_pred = tangent_error(z, ctx.z_pred)
        e_temp = tangent_error(z, ctx.z_prev)
        q_closed = (2.0 * cfg.prediction_weight * e_pred
                    + 2.0 * cfg.temporal_weight * e_temp) / d2
        assert q[0:3] == pytest.approx(q_closed[0:3], rel=1e-9, abs=1e-9)
        assert q[7] == pytest.approx(q_closed[7], rel=1e-9, abs=1e-12)
        # the old buggy term had an extra 1/D (2 w e / D^3, 50x at D_t=0.02)
        q_old = (2.0 * cfg.prediction_weight * e_pred) / (solver.D ** 3)
        assert not np.allclose(q[0:3], q_old[0:3], rtol=1e-3), "pre-F2 scale regression"


def test_qp_quadratic_term_is_residual_gauss_newton(pipeline) -> None:
    """P must be the Gauss-Newton term of the SAME residual Jacobian used for
    q: d/dxi (grad f) == 2 w J^T J == P. The additive (translation/joint)
    subspace is checked by finite differences -- the residual Jacobian is
    exactly constant there -- while the rotation block's Jacobian is
    J_l^{-1}(e_rot) (residual-dependent, so its P carries a second-order term
    of size O(|e_rot|) ~ 1e-3) and is verified analytically instead."""
    solver, ev = pipeline.solver, pipeline.ev
    z, ctx = _tracking_ctx(pipeline, rot=True)
    rng = np.random.default_rng(0)
    ev.forward(z, ctx.box_pose)
    (P, _q0, _A, _lo, _hi), _ = solver.build_qp(z, ctx, trust=1.0, slack=False)
    P = np.asarray(P.todense())
    # the regularization term is a step damping, NOT an objective Hessian
    # (it has no q contribution and never enters the merit): subtract it
    # before comparing P against the finite difference of the gradient
    cfg = pipeline.cfg
    damping = 2.0 * cfg.regularization_weight * np.diag(solver.Dinv2)
    assert np.allclose(P - damping, P - damping)  # shape sanity
    assert np.allclose(P[0, 0] - damping[0, 0],
                       2.0 * (cfg.prediction_weight + cfg.temporal_weight)
                       / solver.D[0] ** 2, rtol=1e-9)
    P_obj = P - damping
    d = 1e-6
    for trial in range(3):
        xi = rng.standard_normal(35) * 1e-3
        xi[3:6] = 0.0
        edges = []
        for s in (+1.0, -1.0):
            zp = retract_batch(ev.model, z[None], (s * d * xi)[None])[0]
            ev.forward(zp, ctx.box_pose)
            (Pv, qv, _A2, _l2, _h2), _ = solver.build_qp(zp, ctx, trust=1.0, slack=False)
            assert np.isfinite(Pv.todense()).all()
            edges.append(qv)
        lhs = (edges[0] - edges[1]) / (2.0 * d)
        rhs = P_obj @ xi
        assert np.allclose(lhs, rhs, atol=1e-6, rtol=1e-4), (trial, lhs - rhs)
    # rotation block: P_rot == 2 w (Jinv/D)^T (Jinv/D) (tracking only)
    _w, J_r, r = solver.tracking_qp_blocks(z, ctx)[0]
    Jinv = so3_left_jacobian_inverse_np((r[3:6] * solver.D[3])[None])[0]
    w_sum = cfg.prediction_weight + cfg.temporal_weight
    P_rot = 2.0 * w_sum * (Jinv.T @ Jinv) / solver.D[3] ** 2
    assert np.allclose(P_obj[3:6, 3:6], P_rot, rtol=1e-9, atol=1e-9)


def test_rotation_block_uses_left_jacobian_inverse(pipeline) -> None:
    """The world-frame rotation residual is not additive under the project's
    left-multiplicative retraction: the identity Jacobian is only first-order
    correct and fails the gradient contract above at a 0.025 rad residual."""
    solver, ev = pipeline.solver, pipeline.ev
    z, ctx = _tracking_ctx(pipeline, rot=True)
    ev.forward(z, ctx.box_pose)
    _w, J_r, r = solver.tracking_qp_blocks(z, ctx)[0]
    e_rot = r[3:6] * solver.D[3]
    assert np.abs(e_rot).max() > 0.01, "test needs a non-zero rotation residual"
    Jinv = so3_left_jacobian_inverse_np(e_rot[None])[0]
    assert np.allclose(J_r[3:6, 3:6], Jinv / solver.D[3])
    assert not np.allclose(Jinv, np.eye(3), atol=1e-3), \
        "identity must be measurably wrong here"


def test_temporal_block_absent_without_previous_frame(pipeline) -> None:
    """z_prev=None: the merit's temporal residual is identically zero, so the
    QP must not add a temporal Hessian the merit does not have."""
    solver, ev = pipeline.solver, pipeline.ev
    z, ctx = _tracking_ctx(pipeline, rot=False)
    ev.forward(z, ctx.box_pose)
    (P0, _q0, _A, _l, _h), _ = solver.build_qp(z, ctx, trust=1.0, slack=False)
    (P1, q1, _A1, _l1, _h1), _ = solver.build_qp(
        z, replace(ctx, z_prev=None), trust=1.0, slack=False)
    P0, P1 = np.asarray(P0.todense()), np.asarray(P1.todense())
    cfg = pipeline.cfg
    pred_only = 2.0 * (cfg.prediction_weight + cfg.regularization_weight) * np.diag(
        solver.Dinv2)
    assert np.allclose(P1, pred_only, rtol=1e-9, atol=1e-9)
    temp = 2.0 * cfg.temporal_weight * np.diag(solver.Dinv2)
    assert np.allclose(P0 - P1, temp, rtol=1e-9, atol=1e-9)
    assert q1[3:6] == pytest.approx(0.0, abs=1e-12)


def test_distance_jacobian_valid_at_exact_contact(pipeline) -> None:
    """Collapsed-fromto guard: at exact contact MuJoCo returns p1 == p2 and
    the old code zeroed the row's Jacobian, leaving the QP blind exactly at
    the contact configurations it must regulate (feet sank centimeters into
    the ground for free while the check saw only the penetration). Sphere vs
    ground is the smooth reference case; the mesh-vs-plane rows are treated
    as non-smooth and excluded from the tight tolerance."""
    ev = pipeline.ev
    h = _root_height_for_tangent(ev)
    ground, box = ev.ground_geom, BOX_FAR.copy()
    rng = np.random.default_rng(2)
    checked = 0
    for dz, label in ((0.0, "tangent"), (-0.003, "penetrating"), (0.004, "clear")):
        z = np.zeros(36)
        z[2] = h + dz
        z[3] = 1.0
        ev.forward(z, box)
        for gid in ev.sole_geoms["l"]:
            dist, ft = ev.geom_distance(gid, ground, distmax=1.0)
            J = np.asarray(ev.distance_jacobian(gid, ground, ft, dist)).reshape(-1)
            if abs(dist) < 1e-12:
                assert J[2] == pytest.approx(1.0, abs=1e-9), "collapsed fromto row"
            xi = rng.standard_normal(35) * 1e-3
            fd = 0.0
            for sgn in (+1.0, -1.0):
                zp = retract_batch(ev.model, z[None], (sgn * xi)[None])[0]
                ev.forward(zp, box)
                fd += sgn * ev.geom_distance(gid, ground, distmax=1.0)[0]
            fd /= 2.0
            lin = float(J @ xi)
            assert abs(fd - lin) <= 1e-5 + 1e-3 * abs(fd), (label, gid, fd, lin)
            checked += 1
    assert checked == 15
    z = np.zeros(36)
    z[2] = h
    z[3] = 1.0
    ev.forward(z, box)
    u = ev.geom_separation_direction(ev.sole_geoms["l"][2], ground, 0.0)
    assert u[2] == pytest.approx(1.0, abs=1e-9)   # away from the ground: up


# --- F4: transition-probe constraints (plan section 7.1) --------------------------


def _probe_scenario(pipeline: RepairPipeline, box0: float = 0.120, box1: float = 0.030,
                    rot=(0.12, -0.08, 0.05), dpos=(0.03, -0.02, 0.01), dj: float = 0.15):
    """One high-motion frame: the box closes on the right hand while the robot
    root rotates/translates and a joint moves. Endpoints are row-legal, the
    interpolated path dips into the box -- the F4 regression scenario."""
    ev = pipeline.ev
    z_prev = standing_scene(1)[0, 0:36].copy()
    gid = ev._hand_geoms["r"][0]
    box_prev = _box_at_gap(ev, z_prev, gid, box0)
    z = z_prev.copy()
    z[3:7] = mat_to_quat_wxyz(so3_exp_np(np.asarray(rot, dtype=float)[None])[0])
    z[0:3] += np.asarray(dpos, dtype=float)
    z[7] += dj
    box_cur = _box_at_gap(ev, z, gid, box1)
    from test_repair import _self_ref_kpts

    ref = _self_ref_kpts(pipeline, z, box_cur)
    ctx = FrameContext(t=1, box_pose=box_cur, z_pred=z, z_prev=z_prev,
                       ref_kpts_world=ref, mesh_L=None, delta_ref=None,
                       mesh_fixed=None, hand_anchor={}, foot_anchor={},
                       prev_hand_local={}, prev_sole_world={}, box_pose_prev=box_prev,
                       ref_hold={"right_hand": True},
                       prev_hand_dist={"r": box0, "l": box0})
    return z, ctx, ev._hand_geoms["r"][0]


def test_probe_interpolation_matches_validator(pipeline) -> None:
    """The solver's probe states must BE the validator's interpolated states
    (translation lerp, rotation slerp, joint lerp, box pose lerp) -- otherwise
    probe rows linearize a different path than the checks evaluate."""
    z, ctx, _gid = _probe_scenario(pipeline)
    scene_a = np.zeros(43)
    scene_a[0:36] = ctx.z_prev
    scene_a[36:43] = ctx.box_pose_prev
    scene_b = np.zeros(43)
    scene_b[0:36] = z
    scene_b[36:43] = ctx.box_pose
    for alpha in (0.2, 1 / 3, 0.5, 2 / 3, 0.8):
        v_mid = pipeline.validator._interpolate(scene_a, scene_b, alpha)
        q_mid, box_mid = pipeline.solver._probe_mid(ctx, z, alpha)
        assert np.array_equal(q_mid, v_mid[0:36]), alpha
        assert np.array_equal(box_mid, v_mid[36:43]), alpha


def test_probe_jacobian_matches_finite_differences(pipeline) -> None:
    """The composite chain rule (endpoint increment -> interpolated state ->
    distance) must equal central differences of the true probe distance,
    covering root rotation, box motion, several alphas and a self pair."""
    z, ctx, gid = _probe_scenario(pipeline)
    solver, ev = pipeline.solver, pipeline.ev
    pairs = [(gid, ev.box_geom), (ev._hand_geoms["r"][0], ev.box_geom),
             (ev._hand_geoms["l"][2], ev.box_geom)]
    h = 1e-6
    for ga, gb in pairs:
        for alpha in (0.25, 1 / 3, 0.5, 2 / 3):
            q_mid, box_mid = solver._probe_mid(ctx, z, alpha)
            ev.probe_forward(q_mid, box_mid)
            dist, ft = ev.probe_geom_distance(ga, gb, distmax=1.0)
            J_mid = ev.distance_jacobian(ga, gb, ft, dist, probe=True)
            J = solver._probe_composite_jacobian(J_mid, ctx, z, alpha)
            g = np.zeros(35)
            for k in range(35):
                e = np.zeros(35)
                e[k] = h
                vals = []
                for sign in (1.0, -1.0):
                    z_pert = retract_batch(ev.model, z[None], (sign * e)[None])[0]
                    qm, bm = solver._probe_mid(ctx, z_pert, alpha)
                    ev.probe_forward(qm, bm)
                    d, _ft = ev.probe_geom_distance(ga, gb, distmax=1.0)
                    vals.append(d)
                g[k] = (vals[0] - vals[1]) / (2.0 * h)
            assert np.allclose(J[0], g, atol=1e-5, rtol=1e-3), (ga, gb, alpha,
                                                               J[0] - g)


def test_probe_rows_repair_a_colliding_transition(pipeline) -> None:
    """The F4 contract: an endpoint-only solve returns a state whose
    interpolated transition still penetrates the box (the pipeline then
    rejects the frame). With probe rows the same solve returns a state that
    passes the SAME check -- the violation is repaired, not merely detected,
    and the probe rows are the reason."""
    z, ctx, _gid = _probe_scenario(pipeline)
    solver = pipeline.solver
    out = {}
    for probes in (False, True):
        solver.probe_cfg = pipeline.validator.cfg if probes else None
        solver.clear_probe_pairs()
        solver._frame_pairs = None
        res = solver.solve_frame(z, ctx)
        chk = pipeline.frame_valid(res.z, ctx)
        out[probes] = (res, chk)
    res_off, chk_off = out[False]
    res_on, chk_on = out[True]
    assert res_off.accepted and res_on.accepted
    assert not chk_off.ok and chk_off.reason == "transition_interpolated", \
        (chk_off.ok, chk_off.reason)
    assert chk_on.ok, (chk_on.reason, chk_on.probes)
    assert res_on.probe_rows, "probe rows must have been active"


def test_probe_rows_inert_below_substep_threshold(pipeline) -> None:
    """No interior probe states exist below max_substep_motion_m: the probe
    block stays empty (zero rows, no false constraints) and the mechanism is
    free."""
    z, ctx, _gid = _probe_scenario(pipeline, box0=0.0205, box1=0.020,
                                   rot=(0.004, -0.003, 0.002),
                                   dpos=(0.002, -0.001, 0.001), dj=0.01)
    solver = pipeline.solver
    solver.probe_cfg = pipeline.validator.cfg
    alphas, motion = solver._probe_grid(ctx, z)
    assert motion <= pipeline.validator.cfg.max_substep_motion_m, motion
    assert alphas == []
    solver._frame_pairs = None
    solver.clear_probe_pairs()
    rows = solver._probe_rows(ctx, z, need_jacobian=False)
    assert rows == []
    val, lo, hi, scl = solver._probe_block(ctx, z)
    assert len(val) == 0


# --- F4 section 7.2: grasp-stability soft target -----------------------------------


def _stab_ctx(pipeline: RepairPipeline, hold: bool = True):
    """Regen frame with the previous accepted state and a reference-driven
    box-frame hand displacement."""
    z, ctx, _gid = _probe_scenario(pipeline)
    ctx = replace(ctx, frame_motion_m=0.02)
    ctx.ref_hold = {"left_hand": hold, "right_hand": False,
                    "left_foot": False, "right_foot": False}
    ctx.prev_hand_local = {"l": None, "r": None}
    ev = pipeline.ev
    ev.forward(ctx.z_prev, ctx.box_pose_prev)
    ctx.prev_hand_local["l"] = ev.hand_local("l").copy()
    ctx.ref_hand_delta = {"l": np.array([0.001, -0.0005, 0.0002])}
    return z, ctx


def test_grasp_stability_term_is_reference_driven(regen_pipeline) -> None:
    """The stability residual is zero when the robot hand repeats the
    reference's box-frame displacement, and scales with the deviation."""
    solver, ev = regen_pipeline.solver, regen_pipeline.ev
    z, ctx = _stab_ctx(regen_pipeline)
    # local construction: freeze the box pose (the current one) so the only
    # motion in the box frame is the robot's, then solve for a hand centroid
    # that moved exactly by the reference displacement
    ctx = replace(ctx, box_pose=ctx.box_pose_prev.copy())
    z_ref = ctx.z_prev.copy()
    step_target = ctx.ref_hand_delta["l"]
    for _ in range(60):     # small Newton iterations on the 3-dim hand task
        ev.forward(z_ref, ctx.box_pose)
        r = (ev.hand_local("l").mean(0)
             - ctx.prev_hand_local["l"].mean(0) - step_target)
        if np.linalg.norm(r) < 1e-6:
            break
        J = ev.hand_local_jac("l").mean(0)
        xi = np.linalg.lstsq(J, -r, rcond=None)[0]
        z_ref = retract_batch(ev.model, z_ref[None], xi[None])[0]
    res = solver.evaluate(z_ref, ctx)
    assert np.abs(res["r_stab"]).max() < 1e-3, res["r_stab"]
    # a robot hand that did not move in the box frame deviates by exactly
    # -ref_delta/scale
    res0 = solver.evaluate(ctx.z_prev, ctx)
    expect = -ctx.ref_hand_delta["l"] / solver.cfg.regen_hand_stab_scale_m
    assert np.allclose(res0["r_stab"], expect, atol=1e-6)
    # with the real box motion the displacement is measured in the CURRENT
    # box geom frame: recompute the step independently from hand_local
    z2_, ctx2_ = _stab_ctx(regen_pipeline)
    ev.forward(ctx2_.z_prev, ctx2_.box_pose)
    step = ev.hand_local("l").mean(0) - ctx2_.prev_hand_local["l"].mean(0)
    res1 = solver.evaluate(ctx2_.z_prev, ctx2_)
    assert np.allclose(res1["r_stab"],
                       (step - ctx2_.ref_hand_delta["l"])
                       / solver.cfg.regen_hand_stab_scale_m, atol=1e-9)
    # outside a reference hold the term is absent
    z2, ctx2 = _stab_ctx(regen_pipeline, hold=False)
    assert solver.evaluate(z2, ctx2)["r_stab"] is None


def test_grasp_stability_block_matches_objective_gradient(regen_pipeline) -> None:
    """The QP's stability block is the Gauss-Newton term of the SAME residual
    the objective evaluates: q == d/dxi objective and P == d/dxi (grad)."""
    solver, ev = regen_pipeline.solver, regen_pipeline.ev
    cfg = replace(regen_pipeline.cfg, regen_hand_stab_weight=1.0)
    solver.cfg = cfg
    z, ctx = _stab_ctx(regen_pipeline)
    ev.forward(z, ctx.box_pose)
    (P, q, _A, _lo, _hi), _ = solver.build_qp(z, ctx, trust=1.0, slack=False)
    P = np.asarray(P.todense())
    # isolate the stability block: rebuild the same QP without the term
    cfg_off = replace(cfg, regen_hand_stab_weight=0.0)
    solver.cfg = cfg_off
    (P0, q0, _A0, _l0, _h0), _ = solver.build_qp(z, ctx, trust=1.0, slack=False)
    solver.cfg = cfg
    dP = P - np.asarray(P0.todense())
    dq = q - q0

    r = solver._stab_residual(ctx)
    J = solver._stab_jacobian(ctx)
    assert np.allclose(dq, 2.0 * cfg.regen_hand_stab_weight * (J.T @ r), atol=1e-9)
    assert np.allclose(dP, 2.0 * cfg.regen_hand_stab_weight * (J.T @ J), atol=1e-9)

    def f_of_xi(xi):
        zz = retract_batch(ev.model, z[None], xi[None])[0]
        return cfg.regen_hand_stab_weight * float((solver.evaluate(zz, ctx)["r_stab"] ** 2).sum())

    h = 1e-6
    g = np.zeros(35)
    for k in range(35):
        e = np.zeros(35)
        e[k] = h
        g[k] = (f_of_xi(e) - f_of_xi(-e)) / (2 * h)
    assert np.allclose(dq, g, atol=1e-6, rtol=1e-4), dq - g


# --- box-frame Jacobians (coordinate-axis contraction fix) ------------------------


def test_box_frame_jacobians_match_finite_differences(regen_pipeline) -> None:
    """``hand_local_jac`` and the mesh residual's kpt Jacobian must contract
    the SITE-POINT axis. The old ``"ij,sak->sik"`` einsum summed over it
    instead, producing a rank-1 wrong-direction row (anchor rows and the
    Laplacian mesh block linearized a different function than they judged).
    """
    solver, ev = regen_pipeline.solver, regen_pipeline.ev
    z, ctx = _stab_ctx(regen_pipeline)
    ev.forward(z, ctx.box_pose)
    R_g, _x_g = ev.box_frame()
    sites = ev.hand_sites["l"]
    J = ev.hand_local_jac("l")
    h = 1e-6
    fd = np.zeros((len(sites), 3, 35))
    for site, sid in enumerate(sites):
        for k in range(35):
            e = np.zeros(35)
            e[k] = h
            vals = []
            for sign in (1.0, -1.0):
                zz = retract_batch(ev.model, z[None], (sign * e)[None])[0]
                ev.forward(zz, ctx.box_pose)
                vals.append(R_g.T @ ev.site_positions([sid])[0])
            fd[site][:, k] = (vals[0] - vals[1]) / (2.0 * h)
    assert np.allclose(J, fd, atol=1e-6, rtol=1e-4), np.abs(J - fd).max()

    # the same contraction for the kpt Jacobian feeding the mesh block: an
    # isotropic Laplacian (all-ones row) must reproduce the mean site motion
    L = np.ones((1, 15)) / 15.0
    J_kpt = np.einsum("ij,sjk->sik", R_g.T, ev.site_jacobians_35(ev.kpt_ids))
    J_mesh = solver._mesh_jacobian(L, J_kpt)
    kpt_ids = list(ev.kpt_ids)
    fd_mesh = np.zeros((3, 35))
    for k in range(35):
        e = np.zeros(35)
        e[k] = h
        vals = []
        for sign in (1.0, -1.0):
            zz = retract_batch(ev.model, z[None], (sign * e)[None])[0]
            ev.forward(zz, ctx.box_pose)
            pts = R_g.T @ (ev.site_positions(kpt_ids) - _x_g).T
            vals.append(pts.T.mean(0))
        fd_mesh[:, k] = (vals[0] - vals[1]) / (2.0 * h)
    assert np.allclose(J_mesh.reshape(3, 35), fd_mesh, atol=1e-6, rtol=1e-4)
