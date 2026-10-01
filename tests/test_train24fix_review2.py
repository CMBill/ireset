"""Review-round-2 counterexamples for the train24fix implementation.

Four confirmed defects (2026-09-27 review), each pinned by a test that
reproduces the original failure mode:

- P1: a carried-back FAR hand pair re-enters ``scored`` with the TRUNCATED
  ``distmax`` (60 mm when the true distance is ~100 mm); the regen row then
  "satisfies" a corridor upper the true distance violates, flipping a
  rejection into a pass. The candidate path must only accept real hits
  (``dist < gate``) and fall through to the registry query otherwise.
- P2a: the D-F failure snapshot used the FIRST attempt's result object while
  the recorded check/rec belong to the LAST executed attempt; the snapshot
  must carry the last executed attempt's result and identity.
- P2c: the verbose attempt log was a spread COPY made before the adoption
  verdict, so every logged ``adopted`` stayed None (67/67 in the real sub10
  logs) and the deciding check of solve-adopted frames was unfindable.
- P2b: the D-F offline interpolation audit reused the endpoint's pair cache,
  so a box sweeping through the robot ONLY at interior states audited as
  "pass" while the formal check rejects.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.scp import FrameContext, FrameResult  # noqa: E402
from test_band_floor_separation import _place_box  # noqa: E402
from test_repair import standing_scene  # noqa: E402

REGEN = replace(RepairConfig(), regen_mode=True)
DT = 1.0 / RepairConfig().fps
LIMDT = DT * 12.0


def _ctx(pipe, z, box, prev_dist_l, motion: float = 0.02) -> FrameContext:
    pipe.ev.forward(z, box)
    return FrameContext(
        t=1, box_pose=box.copy(), z_pred=z.copy(), z_prev=z.copy(),
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={}, prev_hand_local={}, prev_sole_world={},
        box_pose_prev=box.copy(), ref_hold={"left_hand": True},
        prev_hand_dist={"l": prev_dist_l}, frame_motion_m=motion)


def _true_hand_dist(pipe, z, box) -> float:
    pipe.ev.forward(z, box)
    return min(pipe.ev.geom_distance(g, pipe.ev.box_geom, distmax=10.0)[0]
               for g in pipe.ev._hand_geoms["l"])


# --- P1: truncated candidate distances must never bind the corridor ----------------


def test_carried_far_pair_truncated_distance_not_selected() -> None:
    """True hand distance ~100 mm, corridor upper 94.5 mm (prev 100 mm):
    the frame MUST be rejected. Carrying the hand pair back (as a check
    discovery would) must not change that -- pre-fix, the carried pair's
    truncated 60 mm return bound the corridor row and the frame PASSED."""
    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", 0.100)
    d_true = _true_hand_dist(pipe, z, box)
    assert 0.099 < d_true < 0.101
    ctx = _ctx(pipe, z, box, prev_dist_l=0.100)
    # upper = max(0.008, 0.100 - 0.15/30 - margin) = 0.0945 at 2 cm motion
    # baseline without the carried pair: rejected via the registry row
    chk0 = pipe.frame_valid(z, ctx)
    assert not chk0.ok
    cor0 = pipe.solver.last_corridor["l"]
    assert cor0["source"] == "registry" and cor0["dist_m"] > 0.0945

    # carry the FAR hand pair back (exactly what a check discovery does) and
    # re-check the same state: the verdict must not flip to pass
    pipe.solver.clear_discovered_pairs()
    g_hand = pipe.ev._hand_geoms["l"][0]
    ident = pipe.solver.pair_identity(g_hand, pipe.ev.box_geom)
    pipe.solver.note_discovered_pairs([ident])
    chk1 = pipe.frame_valid(z, ctx)
    assert not chk1.ok, "truncated carried distance must not satisfy the corridor"
    cor1 = pipe.solver.last_corridor["l"]
    # the selected distance is the true one (real hit), not the 60 mm sentinel
    assert cor1["dist_m"] == pytest.approx(d_true, abs=1e-9)
    assert any(":regen" in str(r.get("pair")) and r["val_m"] > r["hi_m"]
               for r in chk1.rows)


# --- P2a: failure snapshot binds the LAST EXECUTED attempt -------------------------


def test_fail_snapshot_binds_last_executed_attempt(monkeypatch) -> None:
    """Three failing attempts with DISTINCT candidate states: the snapshot's
    candidate must be the THIRD attempt's state, and its identity fields
    (iterations/qp_calls/status) must match the recorded last attempt."""
    pipe = RepairPipeline(cfg=REGEN)
    events: list = []
    pipe.snapshot = lambda kind, data: events.append((kind, data))
    scene = standing_scene(2)
    scene[1, 7 + 26] = scene[0, 7 + 26] + LIMDT + 2e-6   # skip denied (budget)
    cond = _self_cond(pipe, scene)

    calls = {"n": 0}
    j26_base = float(scene[0, 7 + 26])

    def fake_solve(z_start, ctx, force_slack=False):
        calls["n"] += 1
        zz = np.array(z_start, copy=True)
        # EVERY attempt's candidate carries the same over-budget joint-26
        # step (normalization fails -> attempt unusable) plus a per-attempt
        # joint-5 marker so the three candidate states are distinguishable
        zz[7 + 26] = j26_base + LIMDT + 2e-6
        zz[7 + 5] += 0.001 * calls["n"]
        return FrameResult(accepted=True, z=zz, status="solved",
                           qp_calls=calls["n"], iterations=calls["n"])

    monkeypatch.setattr(pipe.solver, "solve_frame", fake_solve)
    res = pipe.repair_trajectory(scene, cond)
    assert res.status == "rejected_frame" and calls["n"] == 3
    fails = [d for k, d in events if k == "fail"]
    assert len(fails) == 1
    snap = fails[0]
    # the recorded attempt and the snapshot's result are the SAME attempt
    assert snap["last_attempt_id"] == [0, 1, snap["last_rec"]["attempt_id"]]
    assert snap["last_res"].qp_calls == snap["last_rec"]["qp_calls"] == 3
    assert snap["last_res"].iterations == snap["last_rec"]["iterations"]
    # the candidate state is the THIRD attempt's, not the first's: the
    # marker offset accumulates per attempt (z_prev carries none of them)
    assert snap["last_res"].z[7 + 5] == pytest.approx(
        snap["z_prev"][7 + 5] + 0.003, abs=1e-12)
    assert snap["last_res"].z[7 + 5] != pytest.approx(
        snap["z_pred"][7 + 5] + 0.001, abs=1e-12)


def _self_cond(pipe, scene):
    import mujoco
    from ireset.embodied.contact_sites import ContactSchedule, SMPLH_INDEX
    from ireset.embodied.g1_spec import KPT_TO_SMPLH
    from ireset.repair.context import ReferenceCondition

    ids = [mujoco.mj_name2id(pipe.ev.model, mujoco.mjtObj.mjOBJ_SITE, n)
           for n in KPT_TO_SMPLH]
    kpts = np.zeros((scene.shape[0], len(ids), 3))
    for t in range(scene.shape[0]):
        pipe.ev.forward(scene[t, 0:36], scene[t, 36:43])
        kpts[t] = pipe.ev.site_positions(ids)
    human = np.zeros((scene.shape[0], 52, 3))
    for k, name in enumerate(KPT_TO_SMPLH):
        human[:, SMPLH_INDEX[KPT_TO_SMPLH[name]]] = kpts[:, k]
    return ReferenceCondition(human, scene[:, 36:43].copy(), ContactSchedule())


# --- P2c: the verbose attempt log carries the adoption verdict ---------------------


def test_verbose_attempt_log_reflects_adoption() -> None:
    """A solve-adopted frame's verbose attempt record must end with
    ``adopted=True`` (the pre-fix spread copy froze None for every attempt,
    so the deciding check of adopted frames was unfindable)."""
    pipe = RepairPipeline(cfg=REGEN)
    pipe.verbose = True
    scene = standing_scene(2)
    scene[1, 7 + 26] = scene[0, 7 + 26] + LIMDT + 2e-6   # forces a solve
    cond = _self_cond(pipe, scene)
    res = pipe.repair_trajectory(scene, cond)
    attempts = [r for r in pipe.frame_log if r.get("kind") == "attempt"]
    assert attempts, "the over-budget skip denial must produce solve attempts"
    if res.accepted:
        # exactly the adopted attempt is flagged, on the record itself
        adopted = [r for r in attempts if r.get("adopted") is True]
        assert adopted and all(
            r.get("adopted") in (True, False) for r in attempts)
        assert res.diagnostics["frame_status"][1] != "skipped"
    else:
        # a full failure leaves every attempt non-adopted -- still explicit
        assert all(r.get("adopted") is False for r in attempts)


# --- P2b: the offline interpolation audit rescans pairs per point ------------------


def test_probe_audit_rescans_per_point() -> None:
    """Endpoint pair cache must not serve the interpolation audit.  The
    trap is only live when the cache is ESTABLISHED first: the audit runs a
    ``check_rows`` at a definite endpoint state (populating ``_frame_pairs``
    with no robot_box candidate -- the box is far there), then the box
    sweeps through the robot ONLY at an interior subdivision state.  The
    per-point rescan must reject and report the middle pair/alpha; the
    pre-fix audit (no per-point reset) reused the endpoint set and passed.

    Review R5 (2026-10-01): the original test never built the endpoint
    cache, so the FIRST interior point scanned anyway on the fresh solver
    and the pre-fix variant still passed -- verified by replaying the
    pre-fix ``_probe_audit`` against this scenario (recorded in
    train24fix2_review_fix_verification.md section 2)."""
    from train24fix_diagnostics import _probe_audit
    from ireset.repair.scp import FrameSolver

    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    pipe.ev.forward(z, np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0]))
    root = pipe.ev.data.geom_xpos[pipe.ev._hand_geoms["l"][0]]
    # subdivision is capped at max_subdivisions=2 -> interior points sit
    # exactly at alpha=1/3 and 2/3; place the box ON the hand at alpha=1/3
    # and far at BOTH endpoints (30 cm / 60 cm -- beyond the ~6 cm
    # activation margin including the box half-extent)
    box_a = np.array([root[0] - 0.30, root[1], root[2], 1.0, 0, 0, 0])
    box_b = np.array([root[0] + 0.60, root[1], root[2], 1.0, 0, 0, 0])
    ctx = FrameContext(
        t=1, box_pose=box_b, z_pred=z.copy(), z_prev=z.copy(),
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={}, prev_hand_local={}, prev_sole_world={},
        box_pose_prev=box_a, ref_hold={}, prev_hand_dist={},
        frame_motion_m=0.5)
    s2 = FrameSolver(pipe.ev, REGEN)
    s2.probe_cfg = pipe.validator.cfg
    # ESTABLISH the endpoint pair cache exactly like a real endpoint check
    # would (this is the state the pre-fix audit kept reusing): no robot_box
    # candidate at the candidate endpoint, and the cache is a live list --
    # not None -- afterwards
    s2.check_rows(z, ctx)
    assert s2._frame_pairs is not None
    assert not any(ga in set(pipe.ev._hand_geoms["l"]) | set(pipe.ev._hand_geoms["r"])
                   and gb == pipe.ev.box_geom
                   for ga, gb, _k in s2._frame_pairs), \
        "scenario broken: endpoint scan already sees the box pair"
    audit = _probe_audit(pipe, s2, ctx, z, z.copy(), pipe.validator.cfg)
    assert audit["checked"] and audit["n_sub"] > 1
    assert audit["pass"] is False, "mid-state sweep must be caught by rescan"
    all_rows = [row for pt in audit["points"] for row in pt["rows"]]
    assert all_rows and any("largebox" in r["pair"] for r in all_rows)
    # the rejection localizes to an interior alpha with the sweeping pair
    bad = [pt for pt in audit["points"] if pt["rows"]]
    assert bad and all(0.0 < pt["alpha"] < 1.0 for pt in bad)
