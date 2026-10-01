"""C2: re-enabled regen hand box-frame speed rows (train24fix2 plan section 4).

Every test must FAIL on the old defect or on a variant that drops part of
the new enable/release criterion (plan section 4.5):

- the old rows gated on the PROJECTED ``prev_hand_dist`` at 6mm -- five of
  the ten frozen speed spikes sat at 8.5-13.3mm projected while the previous
  FULL-SCENE distance was -1.2..+1.2mm (section 2.4);
- the old gate flapped at its threshold and constrained release frames;
- the 2026-09-15 disable came from a bound incompatible with the reference's
  own grip changes during approach: the new bound sqrt(3)*0.02598 =
  0.04499868 < 0.05 admits every row-satisfying candidate through the
  validator's speed entry gate, and the rows only bind inside an enabled
  reference hold interval (pull-in exempt).
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.embodied.contact_sites import ContactInterval, ContactSchedule  # noqa: E402
from ireset.repair.context import (  # noqa: E402
    C2HandSpeedState,
    ReferenceCondition,
    RepairConfig,
)
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.scp import FrameContext, INF  # noqa: E402
from test_band_floor_separation import _place_box  # noqa: E402
from test_repair import self_condition, standing_scene  # noqa: E402

DT = 1.0 / 30.0
V_ROW = 0.02598
LIM_H = DT * V_ROW                      # per-axis bound: 8.66e-4 m
GAP = 0.008
REGEN_C2 = replace(RepairConfig(), regen_mode=True)              # bound on
REGEN_OFF = replace(RepairConfig(), regen_mode=True,
                    regen_hand_speed_mps=0.0)                    # bound off
POST_HAND_SPEED = 78                    # post-block offset of side-l rows


def _sched(*ivs) -> ContactSchedule:
    s = ContactSchedule()
    for eff, a, b in ivs:
        s.intervals.append(ContactInterval(eff, a, b, "object"))
    return s


def _ctx2(pipe, z, box, prev_local_l, *, enabled=True, scene_dist=0.0012,
          proj_dist=0.012, ref_hold=True, reason="hold", t=1,
          motion=0.02) -> FrameContext:
    """Minimal regen context with the C2 fields of ONE side ('l')."""
    pipe.ev.forward(z, box)
    return FrameContext(
        t=t, box_pose=box.copy(), z_pred=z.copy(), z_prev=z.copy(),
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={},
        prev_hand_local={"l": prev_local_l},
        prev_sole_world={},
        box_pose_prev=box.copy(),
        ref_hold={"left_hand": ref_hold},
        prev_hand_dist={"l": proj_dist},
        frame_motion_m=motion,
        c2_hand_enabled={"l": enabled},
        c2_hand_reason={"l": reason},
        prev_hand_scene_dist={"l": scene_dist},
    )


def _post_slice(solver, names, side="l"):
    base = 99 + len(names) + POST_HAND_SPEED + (0 if side == "l" else 9)
    return slice(base, base + 9)


def _bump(z, j, delta):
    s = z.copy()
    s[j] += delta
    return s


def _setup(dist_m=0.006):
    pipe = RepairPipeline(cfg=REGEN_C2)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", dist_m)
    return pipe, z, box


# --- 1. bound derivation, real row values, the acceptance margin ----------


def test_bound_admits_validator_entry_gate() -> None:
    """sqrt(3) * v_row (and with the acceptance margin 1+1e-6) stays below
    the 0.05 entry gate; the exact-bound candidate has all nine rows at the
    boundary and PASSES the real check (inclusive bounds), one nextafter
    above fails -- the rows and the gate cannot disagree."""
    assert math.sqrt(3) * V_ROW < 0.05
    assert math.sqrt(3) * V_ROW * (1 + 1e-6) < 0.05
    pipe, z, box = _setup()
    cur = pipe.ev.hand_local("l").copy()
    # exact bound on every axis of every site: rows satisfied, norm speed is
    # exactly sqrt(3) * v_row (< 0.05)
    prev = cur - LIM_H
    ctx = _ctx2(pipe, z, box, prev, scene_dist=0.001)
    val, lo, hi, scl, names, _ids = pipe.solver.check_rows(z, ctx)
    sl = _post_slice(pipe.solver, names)
    assert np.allclose(val[sl], LIM_H, atol=1e-15)
    assert (lo[sl] == -LIM_H).all() and (hi[sl] == LIM_H).all()
    assert (scl[sl] == LIM_H).all()
    viol_rows = np.maximum(np.abs(val[sl]) - LIM_H, 0.0) / LIM_H
    # ulp cancellation (cur - (cur - LIM_H) can sit 1ulp above LIM_H) leaves
    # a ~5e-14 normalized sum: INSIDE the 1e-6 acceptance gate, the same
    # same-formula phenomenon V1's strict recheck documents
    assert viol_rows.sum() <= 1e-6
    assert pipe.solver.last_hand_speed["l"]["norm_speed_mps"] == pytest.approx(
        math.sqrt(3) * V_ROW)
    assert pipe.solver.last_hand_speed["l"]["norm_speed_mps"] < 0.05
    # a boundary-scale exceedance (one acceptance-budget unit, 1e-6
    # relative -- a nextafter-level excess is below the cur-(cur-x)
    # rounding error and cannot register) drives the SUM past the 1e-6
    # acceptance gate: this is what blocks an adoption
    prev2 = cur - LIM_H * (1 + 1e-6)
    ctx2 = _ctx2(pipe, z, box, prev2, scene_dist=0.001)
    _v2, _l2, h2, s2, names2, _i2 = pipe.solver.check_rows(z, ctx2)
    viol2 = np.maximum(np.abs(_v2[_post_slice(pipe.solver, names2)]) - LIM_H,
                       0.0) / LIM_H
    assert viol2.sum() == pytest.approx(9e-6, rel=1e-3)
    assert viol2.sum() > 1e-6


# --- 2. the spike counterexample: rigid single-axis shift ------------------


def test_rigid_axis_shift_violates_three_rows_and_rejects() -> None:
    """The frozen-spike shape: previous full-scene distance ~1.2mm (deep
    contact), reference holding, and a rigid 2mm box-frame shift along one
    axis -- the same-axis row of EACH of the 3 sites violates (3 rows, not
    one site violating 3 rows), _frame_valid rejects, and the QP assembly
    carries the same values/bounds."""
    pipe, z, box = _setup()
    cur = pipe.ev.hand_local("l").copy()
    prev = cur - np.array([0.002, 0.0, 0.0])       # rigid +2mm on box x
    ctx = _ctx2(pipe, z, box, prev, scene_dist=0.0012, proj_dist=0.0113)
    val, lo, hi, scl, names, _ids = pipe.solver.check_rows(z, ctx)
    sl = _post_slice(pipe.solver, names)
    vals = val[sl].reshape(3, 3)
    assert np.allclose(vals[:, 0], 0.002)          # x rows carry the shift
    assert np.allclose(vals[:, 1:], 0.0)
    viol = np.maximum(np.abs(val[sl]) - LIM_H, 0.0) / LIM_H
    assert (viol.reshape(3, 3)[:, 0] > 0).all()    # 3 same-axis rows
    assert viol.reshape(3, 3)[:, 1:].sum() == 0.0
    assert viol.sum() == pytest.approx(3 * (0.002 - LIM_H) / LIM_H)
    assert pipe.solver.violation(val, lo, hi, scl) > 1e-6
    chk = pipe.frame_valid(z, ctx)
    assert not chk.ok and chk.reason == "state_constraints"
    assert chk.c2["l"]["norm_speed_mps"] == pytest.approx(0.002 / DT)
    assert chk.c2["l"]["row_absmax_m"] == pytest.approx(0.002)
    assert chk.c2["l"]["bound_abs_m"] == pytest.approx(LIM_H)


# --- 3. moving box: the projected field must NOT gate the rows -------------


def test_projected_distance_does_not_gate_rows() -> None:
    """Section 2.4's five counterexamples: previous FULL-SCENE distance
    <= gap while the projected prev_hand_dist is 8.5-13.3mm. The rows stay
    on and reject the 2mm shift; a variant regressing to the old projected
    6mm gate leaves the same frame unconstrained (both halves asserted)."""
    pipe, z, box = _setup()
    cur = pipe.ev.hand_local("l").copy()
    prev = cur - np.array([0.002, 0.0, 0.0])
    ctx = _ctx2(pipe, z, box, prev, scene_dist=0.0004, proj_dist=0.0122)
    val, _lo, _hi, _scl, names, _ids = pipe.solver.check_rows(z, ctx)
    sl = _post_slice(pipe.solver, names)
    assert np.abs(val[sl]).max() == pytest.approx(0.002)
    assert pipe.solver.violation(val, _lo, _hi, _scl) > 1e-6
    # the OLD criterion (projected > 6mm -> skip; no enable state) leaves
    # exactly this frame unconstrained -- the defect these rows replace
    ctx_old = _ctx2(pipe, z, box, prev, enabled=False, reason=None)
    v_o, l_o, h_o, s_o, names_o, _ = pipe.solver.check_rows(z, ctx_old)
    sl_o = _post_slice(pipe.solver, names_o)
    assert np.abs(v_o[sl_o]).max() == 0.0
    assert (l_o[sl_o] == -INF).all() and (h_o[sl_o] == INF).all()
    assert pipe.solver.violation(v_o, l_o, h_o, s_o) <= 1e-6


# --- 4. pull-in exemption ---------------------------------------------------


def test_pull_in_exemption_and_no_reexemption() -> None:
    """Not yet enabled + previous full-scene distance 9mm (> gap 8mm): both
    assembly paths carry no rows (the corridor drives the approach); an
    ENABLED state must never re-enter the exemption."""
    st = C2HandSpeedState(_sched(("left_hand", 5, 12)), 15, GAP)
    on, why = st.step("left_hand", 5, True, 0.009)
    assert (on, why) == (False, "pull_in_exempt")
    pipe, z, box = _setup()
    cur = pipe.ev.hand_local("l").copy()
    prev = cur - np.array([0.002, 0.0, 0.0])       # would violate if on
    ctx = _ctx2(pipe, z, box, prev, enabled=False, reason="pull_in_exempt",
                scene_dist=0.009)
    val, lo, hi, scl, names, _ = pipe.solver.check_rows(z, ctx)
    sl = _post_slice(pipe.solver, names)
    assert np.abs(val[sl]).max() == 0.0 and (hi[sl] == INF).all()
    assert pipe.solver.violation(val, lo, hi, scl) <= 1e-6
    # enable at 8mm, then the same 9mm must HOLD, not re-exempt
    assert st.step("left_hand", 6, True, 0.008) == (True, "enable")
    assert st.step("left_hand", 7, True, 0.009) == (True, "hold")


# --- 5. enable stability at the gap boundary --------------------------------


def test_enable_boundary_and_stability() -> None:
    """6.5mm enables (the old 6mm gate would NOT have); a distance one
    nextafter above the 8mm gap does not enable an untouched interval, but
    once enabled the interval state holds through values beyond the gap --
    no flapping at the threshold, no physical threshold modified."""
    st = C2HandSpeedState(_sched(("left_hand", 0, 10)), 10, GAP)
    assert st.step("left_hand", 1, True, 0.0065) == (True, "enable")
    assert RepairConfig().regen_hand_contact_gap_m == GAP   # corridor gap reused
    st2 = C2HandSpeedState(_sched(("left_hand", 0, 10)), 10, GAP)
    above = float(np.nextafter(GAP, np.inf))
    assert st2.step("left_hand", 1, True, above) == (False, "pull_in_exempt")
    assert st2.step("left_hand", 2, True, GAP) == (True, "enable")
    # enabled: holds at 9mm, at the boundary value and one nextafter above
    for d in (0.009, GAP, above, 0.05):
        assert st2.step("left_hand", 3, True, d) == (True, "hold")


# --- 6. explicit release at interval end ------------------------------------


def test_release_at_reference_end_leaves_departure_free() -> None:
    """Previous distance 4mm but the reference no longer wants hold: rows
    OFF on both paths and a 2mm departure step is not rejected by C2 (other
    constraints still apply -- the frame itself stays acceptable)."""
    st = C2HandSpeedState(_sched(("left_hand", 0, 5)), 8, GAP)
    assert st.step("left_hand", 1, True, 0.004) == (True, "enable")
    on, why = st.step("left_hand", 5, False, 0.004)
    assert (on, why) == (False, "off_no_ref_released")
    assert st.enabled["left_hand"] is False          # state released
    pipe, z, box = _setup()
    cur = pipe.ev.hand_local("l").copy()
    prev = cur - np.array([0.002, 0.0, 0.0])
    ctx = _ctx2(pipe, z, box, prev, enabled=False, ref_hold=False,
                reason="off_no_ref_released", scene_dist=0.004,
                proj_dist=0.004)
    val, lo, hi, scl, names, _ = pipe.solver.check_rows(z, ctx)
    sl = _post_slice(pipe.solver, names)
    assert (hi[sl] == INF).all()
    assert pipe.solver.violation(val, lo, hi, scl) <= 1e-6
    chk = pipe.frame_valid(z, ctx)
    assert chk.ok, chk.rows                          # departure not blocked


# --- 7. interval identity, isolation, failure isolation ---------------------


def test_interval_identity_change_and_side_isolation() -> None:
    """Interval identity change clears the old state and re-judges; sides
    are independent; a new pass (fresh instance) starts clean; the step is
    a pure function of (prefix, reference) -- a failed candidate has no
    channel into it; t=0 / missing previous distance is safe."""
    st = C2HandSpeedState(
        _sched(("left_hand", 2, 5), ("left_hand", 7, 10)), 12, GAP)
    assert st.step("left_hand", 2, True, 0.006) == (True, "enable")
    assert st.step("left_hand", 5, False, 0.004)[0] is False   # interval end
    # new interval: re-judged from its own distance (9mm -> exempt)
    assert st.step("left_hand", 7, True, 0.009) == (False, "pull_in_exempt")
    assert st.step("left_hand", 8, True, 0.006) == (True, "enable")
    # sides independent: right hand never enabled by the left's state
    assert st.step("right_hand", 8, True, 0.030) == (False, "pull_in_exempt")
    assert st.enabled["right_hand"] is False
    # t=0 and missing distance: off, no inheritance
    st2 = C2HandSpeedState(_sched(("left_hand", 0, 10)), 10, GAP)
    assert st2.step("left_hand", 0, True, None) == (False, "off_t0_or_no_prev")
    assert st2.step("left_hand", 1, True, 0.006) == (True, "enable")
    assert st2.step("left_hand", 2, True, None) == (False, "off_t0_or_no_prev")
    assert st2.enabled["left_hand"] is False          # mid-run miss clears
    # purity: the state is a function of the (prefix, reference) inputs --
    # two fresh instances fed the same prefix reach the same state (a
    # failed candidate has no channel into it; step COMMITTING per frame
    # is the frame-order semantics, not candidate dependence)
    a = C2HandSpeedState(_sched(("left_hand", 0, 10)), 10, GAP)
    b = C2HandSpeedState(_sched(("left_hand", 0, 10)), 10, GAP)
    for t in range(1, 4):
        assert a.step("left_hand", t, True, 0.007) == b.step("left_hand", t, True, 0.007)
    assert a.enabled == b.enabled


# --- 8. the two assembly paths agree ----------------------------------------


def test_value_path_and_qp_mirror_agree() -> None:
    """Same ctx -> same enable set, same bounds, same placeholders. The QP
    mirror's bounds plus the value path's values must reconstruct the
    symmetric +-lim window at the linearization state; inactive slots are
    unconstrained in BOTH. The previous array is never mutated."""
    pipe, z, box = _setup()
    cur = pipe.ev.hand_local("l").copy()
    prev = cur - np.array([0.0005, -0.0003, 0.0])   # within bound
    ctx = _ctx2(pipe, z, box, prev.copy(), scene_dist=0.002)
    val, lo, hi, scl, names, _ = pipe.solver.check_rows(z, ctx)
    sl = _post_slice(pipe.solver, names)
    # 'r' side has no prev_local: helper off in BOTH paths
    from ireset.repair.scp import hand_speed_rows_active
    assert hand_speed_rows_active(REGEN_C2, ctx, "l")
    assert not hand_speed_rows_active(REGEN_C2, ctx, "r")
    pipe.ev.forward(z, box)
    (_P, _q, A, qlo, qhi), _ns = pipe.solver.build_qp(
        z, ctx, trust=REGEN_C2.trust_radius_max)
    base_qp = 99 + len(pipe.solver.last_row_pairs) + POST_HAND_SPEED
    slq = slice(base_qp, base_qp + 9)
    # the QP constrains the STEP: Jh @ xi in [-lim - val, lim - val] (the
    # value path's val = cur - prev) -- bounds plus values reconstruct the
    # symmetric +-lim window at the linearization state
    assert np.all(qlo[slq] + val[sl] == pytest.approx(-LIM_H))
    assert np.all(qhi[slq] + val[sl] == pytest.approx(LIM_H))
    assert np.abs(A[slq]).max() > 0.0                # real Jacobian rows
    # inactive side: zero rows, +-INF placeholders in both paths
    slr = slice(base_qp + 9, base_qp + 18)
    assert np.abs(A[slr]).max() == 0.0
    assert (qlo[slr] == -INF).all() and (qhi[slr] == INF).all()
    # the previous-frame array is a snapshot: neither path mutated it
    assert np.array_equal(ctx.prev_hand_local["l"], prev)


# --- 9. disabled states add no queries and no state --------------------------


def test_disabled_states_issue_no_collection(monkeypatch) -> None:
    """Standard mode and regen+parameter-0: the new distance collection is
    NEVER called (the collection point sits on an unconditional path -- an
    ungated query there would change standard-mode call counts), the C2
    accounting stays empty, and the run completes unchanged."""
    for cfg in (RepairConfig(), REGEN_OFF):
        pipe = RepairPipeline(cfg=cfg)
        calls: list[int] = []
        monkeypatch.setattr(pipe, "_collect_prev_scene_dist",
                            lambda t: calls.append(t) or {})
        scene = standing_scene(3)
        cond = self_condition(pipe, scene)
        res = pipe.repair_trajectory(scene, cond)
        assert calls == []
        d = res.diagnostics["c2_hand_speed"]
        assert d["frames_on"] == {"l": 0, "r": 0}
        assert d["enable_events"] == [] and d["release_events"] == []
        assert d["query_miss"] == []


# --- 10. every adoption entrance sees the rows -------------------------------


def test_skip_violation_escalates_and_adopted_frames_comply() -> None:
    """A prediction frame violating the enabled rows fails the SKIP check
    and engages the solver (not merely 'the solver was called'); whatever
    the pipeline adopts satisfies the rows bit-for-frame."""
    pipe = RepairPipeline(cfg=REGEN_C2)
    scene = standing_scene(2)
    z0 = scene[0, 0:36].copy()
    box = _place_box(pipe, z0, "l", 0.006)
    scene[:, 36:43] = box
    # frame 1 prediction: a left-arm joint moves within the velocity limit
    # but shifts the box-frame hand sites beyond the per-axis bound
    pipe.ev.forward(z0, box)
    base_local = pipe.ev.hand_local("l").copy()
    j = next(cand for cand in range(7, 36)
             if (pipe.ev.forward(_bump(z0, cand, 0.05), box),
                 np.abs(pipe.ev.hand_local("l") - base_local).max() > LIM_H)[1])
    scene[1, 0:36] = z0.copy()
    scene[1, j] += 0.05
    base = self_condition(pipe, scene)
    cond = ReferenceCondition(base.human_kpts_world, scene[:, 36:43],
                              _sched(("left_hand", 0, 2)))
    res = pipe.repair_trajectory(scene, cond)
    d = res.diagnostics["c2_hand_speed"]
    assert {"t": 1, "side": "l"} in d["enable_events"]   # enabled at t=1
    assert res.diagnostics["frame_status"][1] != "skipped"
    if res.accepted:
        pipe.ev.forward(res.trajectory[0, 0:36], box)
        h0 = pipe.ev.hand_local("l").copy()
        pipe.ev.forward(res.trajectory[1, 0:36], box)
        step = np.abs(pipe.ev.hand_local("l") - h0)
        assert step.max() <= LIM_H * (1 + 1e-5), step.max()


# --- 11. certification vs joint feasibility counterexamples -------------------


def test_norm_passes_but_component_fails() -> None:
    """The frozen sub10:r t=42 shape: norm speed 0.04709345 passes the
    0.05 entry gate but the 0.04185564 max axis component violates the
    rows -- judging by the norm instead of per-axis rows would certify it."""
    a = 0.04185564 * DT
    n = 0.04709345 * DT
    b = math.sqrt(n * n - a * a)
    pipe, z, box = _setup()
    cur = pipe.ev.hand_local("l").copy()
    prev = cur.copy()
    prev[0] -= np.array([a, b, 0.0])          # one site carries the motion
    ctx = _ctx2(pipe, z, box, prev, scene_dist=0.001)
    val, lo, hi, scl, names, _ = pipe.solver.check_rows(z, ctx)
    viol = np.maximum(np.abs(val[_post_slice(pipe.solver, names)]) - LIM_H,
                      0.0) / LIM_H
    assert viol.sum() > 1e-6
    hs = pipe.solver.last_hand_speed["l"]
    assert hs["norm_speed_mps"] == pytest.approx(0.04709345, rel=1e-6)
    assert hs["norm_speed_mps"] < 0.05        # the entry gate would pass
    assert hs["row_absmax_m"] == pytest.approx(a)
    assert not pipe.frame_valid(z, ctx).ok


def test_clean_rows_do_not_excuse_other_families() -> None:
    """Rows clean (zero hand displacement) but a joint limit or a foot-speed
    violation still rejects the candidate -- C2 never substitutes for the
    other constraint families."""
    pipe, z, box = _setup()
    # joint limit violation, hand static in the box frame
    z_bad = z.copy()
    j = int(np.argmax(pipe.ev.spec.joint_range[:, 1] - z_bad[7:36]))
    z_bad[7 + j] = pipe.ev.spec.joint_range[j, 1] + 0.01
    pipe.ev.forward(z_bad, box)
    prev = pipe.ev.hand_local("l").copy()
    pipe.ev.forward(z, box)
    ctx = _ctx2(pipe, z, box, prev, scene_dist=0.001)
    chk = pipe.frame_valid(z_bad, ctx)
    assert not chk.ok
    assert pipe.solver.last_hand_speed["l"]["row_absmax_m"] == pytest.approx(
        np.abs(pipe.ev.hand_local("l") - prev).max(), abs=1e-12)
    assert pipe.solver.last_hand_speed["l"]["norm_speed_mps"] == pytest.approx(
        0.0, abs=1e-9)
