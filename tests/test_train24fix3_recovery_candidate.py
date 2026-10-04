"""train24fix3 continuation unit 4: the R1 feasibility-first recovery
candidate (plan sections 7.4/7.5).

Pinned here:

- the registry: R1 == the S1 numeric path + ``recovery_policy``; the closed
  field set rejects unknown policies, and an axis-mode recovery profile is
  an unregistered combination;
- the recovery QP objective (plan 7.5 item 2): ONLY the registered step
  regularization in P and a zero tracking linear term; every relaxable
  slack priced ``mu / accept_scale`` through the explicit row->scale map
  (C2 site slacks stay dimensionless at plain mu); a missing map is an
  engineering error, never a silently defaulted scale;
- the recovery-step predicate (item 3): strict real-phi decrease, no
  non-C2 family may worsen (zero families stay zero), task under
  max(gate, mpjpe(z)), non-finite rejects -- the tracking objective is
  absent on purpose;
- the integration path: a deep C2 violation enters the recovery phase, an
  accepted recovery step may RAISE the tracking objective while phi drops,
  the phase machine re-judges per iteration from the real rows, budget and
  QP counts run continuously across the switch, and after feasibility is
  achieved every accepted tracking step keeps phi <= 1e-6 (the hold may
  reject a merit-improving step);
- purity: R1 attached but the frame C2-inactive, standard mode, or C2
  disarmed reproduces the S1/original behaviour exactly (``recovery`` is
  None, no phase machinery in the trace).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from dataclasses import replace  # noqa: E402

from ireset.repair.config_io import load_mode_configs  # noqa: E402
from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.profiles import get_profile, load_profiles  # noqa: E402
from ireset.repair.scp import N_TANGENT  # noqa: E402
from test_band_floor_separation import _place_box  # noqa: E402
from test_hand_speed_row import REGEN_C2, REGEN_OFF  # noqa: E402
from test_repair import standing_scene  # noqa: E402
from test_train24fix3_norm import _ctx_with_prev, _setup  # noqa: E402

DT = 1.0 / 30.0
R1 = get_profile("R1")
B_R1 = DT * R1.c2_norm_speed_mps


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_r1_registry_entry_and_closed_fields(tmp_path):
    import yaml

    profiles = load_profiles()
    r1, s1 = profiles["R1"], profiles["S1"]
    assert r1.recovery_policy == "feasibility_first"
    assert r1.c2_mode == s1.c2_mode == "norm_cut"
    assert r1.c2_norm_speed_mps == s1.c2_norm_speed_mps == 0.045
    assert r1.c2_row_precondition == s1.c2_row_precondition
    assert r1.c2_slack_penalty == s1.c2_slack_penalty == "mu"
    assert r1.solver == s1.solver
    # every historical profile keeps the pure original path (the R
    # candidates are the only ones carrying a recovery policy)
    assert all(p.recovery_policy is None for n, p in profiles.items()
               if n not in ("R1", "R2", "R3"))

    doc = yaml.safe_load(Path(get_profile.__globals__["PROFILES_YAML"]).read_text())
    base = dict(doc["profiles"])
    # unknown policy value -> rejected
    bad = dict(base)
    bad["X1"] = {"base": "R1", "recovery_policy": "wishful"}
    p = tmp_path / "bad1.yaml"
    p.write_text(yaml.safe_dump({"schema": "train24fix3_profiles_v1",
                                 "profiles": bad}))
    with pytest.raises(ValueError, match="recovery_policy"):
        load_profiles(p)
    # an axis-mode recovery profile is an unregistered combination
    bad2 = dict(base)
    bad2["X2"] = {"base": "R1", "c2_mode": "axis", "c2_norm_speed_mps": None}
    p2 = tmp_path / "bad2.yaml"
    p2.write_text(yaml.safe_dump({"schema": "train24fix3_profiles_v1",
                                  "profiles": bad2}))
    with pytest.raises(ValueError, match="recovery_policy requires"):
        load_profiles(p2)


# ---------------------------------------------------------------------------
# the recovery QP objective (plan 7.5 item 2)
# ---------------------------------------------------------------------------


def _violation_state():
    """A C2-active frame whose current hand sits 4 sites-B above the
    previous accepted hand: the per-site NORM rows are deep in violation,
    phi_c2 > 0 -- the recovery phase's entry state."""
    pipe, z, box = _setup(R1)
    d = np.tile((4.0 * B_R1, 0.0, 0.0), (3, 1))
    ctx = _ctx_with_prev(pipe, z, box, d)
    return pipe, z, box, ctx


def test_recovery_objective_only_regularization_and_normalized_slack():
    pipe, z, box, ctx = _violation_state()
    solver = pipe.solver
    solver.ev.forward(z, box)
    solver.trace = {"attempts": []}   # last_slack_rows is trace-gated
    # freeze the accept-scale map exactly as solve_frame would
    val, lo, hi, scl = solver.constraint_rows(z, ctx, 1.0, mode="accept")
    names, ids = solver.row_descriptors()
    phi, phi_c2, fam = solver._phi_parts(val, lo, hi, scl, names, ids)
    assert phi > 1e-6 and phi_c2 > 0.0          # the entry state
    solver._accept_scale_by_name = {
        nm: float(scl[i]) for i, nm in enumerate(names)
        if np.isfinite(scl[i]) and float(scl[i]) > 0.0}

    mu = 360.0
    (Pt, qt, At, lot, hit), n_s_t = solver.build_qp(
        z, ctx, 1.0, slack=True, mu=mu, objective_mode="tracking")
    (Pr, qr, Ar, lor, hir), n_s_r = solver.build_qp(
        z, ctx, 1.0, slack=True, mu=mu, objective_mode="recovery")

    Pd = np.asarray(Pr.todense())
    expected_P = 2.0 * solver.cfg.regularization_weight * np.diag(solver.Dinv2)
    assert np.allclose(Pd[:N_TANGENT, :N_TANGENT], expected_P, atol=1e-12)
    assert np.allclose(Pd[N_TANGENT:, :], 0.0) and np.allclose(
        Pd[:, N_TANGENT:], 0.0, atol=1e-12)
    # NO tracking linear term at all (prediction/temporal/mesh/hand/stab)
    assert np.allclose(qr[:N_TANGENT], 0.0)
    assert not np.allclose(np.asarray(Pt.todense())[:N_TANGENT, :N_TANGENT],
                           expected_P)
    # the two assemblies share the SAME row set and slack layout
    assert Ar.shape == At.shape and n_s_r == n_s_t

    # the slack pricing: C2 site slacks at plain mu, everything else
    # mu/accept_scale, resolved through the row->scale map
    slack_rows = solver.last_slack_rows
    penalties = slack_rows["penalty"]
    assert len(penalties) == n_s_r
    c2_pens, other = [], []
    for k, nm in enumerate(slack_rows["names"]):
        col = slack_rows["slack_of_row"][k]
        if nm.startswith(("hand_norm_box", "hand_norm_cut")):
            c2_pens.append(penalties[col])
        else:
            other.append((nm, penalties[col]))
    assert c2_pens and all(p == mu for p in c2_pens)
    assert other
    for nm, p in other:
        sc = solver._accept_scale_by_name.get(
            nm, solver._accept_scale_by_name.get(f"probe:{nm}"))
        assert sc is not None, nm
        assert p == pytest.approx(mu / sc, rel=1e-12), nm


def test_recovery_objective_requires_the_scale_map():
    pipe, z, box, ctx = _violation_state()
    solver = pipe.solver
    solver.ev.forward(z, box)
    solver._accept_scale_by_name = None
    with pytest.raises(RuntimeError, match="accept-scale map"):
        solver.build_qp(z, ctx, 1.0, slack=True, mu=360.0,
                        objective_mode="recovery")
    # a relaxable row WITHOUT a legitimate accept counterpart is an
    # engineering error, never a silently defaulted scale
    solver._accept_scale_by_name = {"joint_limit[j7]": 0.01}
    with pytest.raises(RuntimeError, match="no legitimate accept scale"):
        solver.build_qp(z, ctx, 1.0, slack=True, mu=360.0,
                        objective_mode="recovery")


# ---------------------------------------------------------------------------
# the recovery-step predicate (plan 7.5 item 3)
# ---------------------------------------------------------------------------


def _solver_for_predicate():
    pipe = RepairPipeline(cfg=REGEN_C2, c2_profile=R1)
    return pipe.solver


def test_recovery_step_ok_predicate():
    s = _solver_for_predicate()
    zero = {"joint_vel": 0.0, "robot_box": 0.0}
    # phi down, families flat, task under the cap -> accepted
    ok, _ = s._recovery_step_ok(1.0, zero, 0.10, 0.5, zero, 0.11)
    assert ok
    # strict decrease required
    ok, _ = s._recovery_step_ok(1.0, zero, 0.10, 1.0, zero, 0.10)
    assert not ok
    # a non-C2 family may not worsen -- the R1 counter-example
    ok, info = s._recovery_step_ok(1.0, zero, 0.10, 0.5,
                                   {"joint_vel": 0.2, "robot_box": 0.0}, 0.11)
    assert not ok and info["families_ok"] is False
    # a zero-violation family must stay at zero (new violation rejected)
    ok, _ = s._recovery_step_ok(1.0, zero, 0.10, 0.5,
                                {"joint_vel": 0.0, "robot_box": 1e-9}, 0.11)
    assert not ok
    # task guard: over max(gate, mpjpe(z)) -> rejected
    ok, _ = s._recovery_step_ok(1.0, zero, 0.10, 0.5, zero, 0.41)
    assert not ok
    ok, _ = s._recovery_step_ok(1.0, zero, 0.55, 0.5, zero, 0.55)
    assert ok                                  # cap follows the WORSE task
    ok, _ = s._recovery_step_ok(1.0, zero, 0.55, 0.5, zero, 0.56)
    assert not ok
    # non-finite phi or task rejects the step outright
    ok, _ = s._recovery_step_ok(1.0, zero, 0.10, float("nan"), zero, 0.11)
    assert not ok
    ok, _ = s._recovery_step_ok(1.0, zero, 0.10, 0.5, zero, float("inf"))
    assert not ok


# ---------------------------------------------------------------------------
# the integration path: phase machine + line search + purity
# ---------------------------------------------------------------------------


def _run_frame(profile, ctx_z_box, *, cfg=None):
    pipe, z, box, ctx = ctx_z_box
    pipe2 = RepairPipeline(cfg=cfg or REGEN_C2, c2_profile=profile)
    pipe2.solver.trace = {"attempts": []}
    pipe2.solver.ev.forward(z, box)
    res = pipe2.solver.solve_frame(z, ctx)
    return res, pipe2.solver.trace


def test_r1_deep_violation_recovers_with_objective_rise_allowed():
    pipe, z, box, ctx = _violation_state()
    res, trace = _run_frame(R1, (pipe, z, box, ctx))
    att = trace["attempts"][0]
    assert att["r1_active"] is True
    its = att["iterations"]
    rec_its = [i for i in its if i.get("r1", {}).get("phase") == "recovery"]
    assert rec_its, "the deep C2 violation must enter the recovery phase"
    # an ACCEPTED recovery step that RAISES the tracking objective while
    # phi strictly drops -- the exact case the merit rule used to reject
    accepted_recovery = [
        ls for i in rec_its for ls in i["line_search"]
        if ls.get("accepted") and "phi_c" in ls]
    assert accepted_recovery
    assert any(ls["objective_c"] > ls["objective_z"]
               for ls in accepted_recovery)
    for i in rec_its:
        for ls in i["line_search"]:
            if ls.get("accepted") and "phi_c" in ls:
                assert ls["phi_c"] < i["r1"]["phi"]
    # the phase machine re-judged from the real rows, budget ran through
    assert res.recovery is not None
    assert res.recovery["recovery_iters"] == len(rec_its)
    assert res.recovery["phase_switches"] >= 1
    assert res.iterations == len(its)
    assert res.qp_calls == len(att["qp"])
    # after feasibility was achieved, every accepted tracking step kept
    # phi <= 1e-6 (the hold is visible in the trace when it bound)
    if res.recovery["recovery_achieved"]:
        post = [ls for i in its
                if i.get("r1", {}).get("phase") == "tracking"
                and any(k.get("feasibility_hold") is False
                        for k in i["line_search"])]
        for i in its:
            for ls in i["line_search"]:
                if ls.get("accepted") and "viol_c" in ls \
                        and "feasibility_hold" in ls:
                    assert ls["viol_c"] <= 1e-6
        # a merit-improving step that the hold REJECTED is the strongest
        # evidence; record its presence when it occurs (not forced)
        _ = post


def test_r1_budget_and_qp_counts_continuous_across_switch():
    pipe, z, box, ctx = _violation_state()
    res, trace = _run_frame(R1, (pipe, z, box, ctx))
    att = trace["attempts"][0]
    its = att["iterations"]
    # iteration indices strictly increase through phase switches; QP ids
    # accumulate monotonically -- no reset at the switch
    assert [i["it"] for i in its] == list(range(len(its)))
    qp_ids = [q["qp_id"] for q in att["qp"]]
    assert qp_ids == sorted(qp_ids) and len(qp_ids) == res.qp_calls
    branches = {q["branch"] for q in att["qp"]}
    assert "recovery" in branches


def test_r1_purity_c2_inactive_frame_matches_s1():
    """R1 attached but the frame carries NO active C2 side: the recovery
    gate is closed and the frame solves under the S1 behaviour exactly."""
    pipe, z, box = _setup(R1)
    ctx = _ctx_with_prev(pipe, z, box, np.zeros((3, 3)))
    # disable both sides: no enable state, no previous-scene proximity
    from ireset.repair.scp import FrameContext  # noqa: PLC0415

    ctx_off = FrameContext(
        t=1, box_pose=ctx.box_pose, z_pred=ctx.z_pred, z_prev=None,
        ref_kpts_world=ctx.ref_kpts_world, mesh_L=None, delta_ref=None,
        mesh_fixed=None, hand_anchor={}, foot_anchor={},
        prev_hand_local={"l": ctx.prev_hand_local["l"]},
        prev_sole_world={}, box_pose_prev=ctx.box_pose_prev,
        ref_hold={}, prev_hand_dist={"l": 0.30},
        frame_motion_m=0.02, c2_hand_enabled={}, c2_hand_reason={},
        prev_hand_scene_dist={"l": 0.30})
    r_s1, _ = _run_frame(get_profile("S1"), (pipe, z, box, ctx_off))
    r_r1, _ = _run_frame(R1, (pipe, z, box, ctx_off))
    assert r_r1.recovery is None and r_s1.recovery is None
    assert r_r1.status == r_s1.status
    assert r_r1.iterations == r_s1.iterations
    assert r_r1.qp_calls == r_s1.qp_calls
    assert np.array_equal(r_r1.z, r_s1.z)
    assert r_r1.term_reason == r_s1.term_reason


def test_r1_purity_standard_mode_and_c2_disarmed():
    """Standard mode and C2-disarmed regen ignore the recovery profile:
    results are bit-identical to running without any profile."""
    pipe, z, box, ctx = _violation_state()
    std = replace(load_mode_configs("strict")[0])     # regen_mode False
    r_none, _ = _run_frame(None, (pipe, z, box, ctx), cfg=std)
    r_r1, _ = _run_frame(R1, (pipe, z, box, ctx), cfg=std)
    assert np.array_equal(r_none.z, r_r1.z) and r_none.qp_calls == r_r1.qp_calls
    assert r_r1.recovery is None
    # C2 disarmed (regen_hand_speed_mps = 0): the gate reads the CONFIG,
    # not just the profile
    r_off_none, _ = _run_frame(None, (pipe, z, box, ctx), cfg=REGEN_OFF)
    r_off_r1, _ = _run_frame(R1, (pipe, z, box, ctx), cfg=REGEN_OFF)
    assert np.array_equal(r_off_none.z, r_off_r1.z)
    assert r_off_r1.recovery is None
