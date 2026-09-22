"""F0/F1 correctness contracts (m2_m3_stage_gate_fix_plan sections 3-4).

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
from ireset.kinematics.rotation_adapter import mat_to_quat_wxyz, so3_exp_np  # noqa: E402
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
