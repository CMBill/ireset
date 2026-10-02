"""train24fix3 N2: cost-preserving C2 row preconditioning (plan section 6.1).

Pinned invariants:

- the transformation is a pure ROW rescale of the SAME linear constraint
  set: A' = A/B, l' = l/B, u' = u/B on the C2 block only; every other
  block (pre / collision / probe / other post families) is untouched, and
  the trust rows are NOT relaxed;
- the REAL acceptance rows stay in metres (val = d, l/u = +-B, scale = B)
  -- check_rows is byte-identical under N0 and N2;
- the C2 recovery slack becomes dimensionless with penalty 10000*B --
  exactly the metre-unit cost (s_m = B * s_norm); other families keep
  10000;
- the tight QP's solution is unchanged (same set, same minimizer up to
  solver arithmetic) and the slack QP's C2 slack satisfies s_m = B*s_norm;
- an N0 profile attached is byte-identical to no profile at all.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.scp import FrameContext  # noqa: E402
from ireset.repair.profiles import get_profile  # noqa: E402
from test_band_floor_separation import _place_box  # noqa: E402
from test_hand_speed_row import REGEN_C2  # noqa: E402
from test_repair import standing_scene  # noqa: E402

DT = 1.0 / 30.0
B = DT * 0.02598


def _prepared():
    """C2-active context with the previous hand array offset INSIDE the
    bound (the rows are real, not vacuous)."""
    pipe = RepairPipeline(cfg=REGEN_C2)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", 0.006)
    pipe.ev.forward(z, box)
    cur = pipe.ev.hand_local("l")
    prev = cur - np.array([0.0005, -0.0003, 0.0])
    ctx = FrameContext(
        t=1, box_pose=box.copy(), z_pred=z.copy(), z_prev=None,
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={},
        prev_hand_local={"l": prev.copy()},
        prev_sole_world={}, box_pose_prev=box.copy(),
        ref_hold={"left_hand": True}, prev_hand_dist={"l": 0.012},
        frame_motion_m=0.02,
        c2_hand_enabled={"l": True}, c2_hand_reason={"l": "hold"},
        prev_hand_scene_dist={"l": 0.0012})
    return z, box, ctx, prev


def _c2_qp_rows(solver, A, ctx):
    names = solver._qp_row_names(A.shape[0], ctx)
    idx = [i for i, n in enumerate(names) if n.startswith("hand_speed[")]
    assert idx, "no C2 rows found in the QP layout"
    return idx, names


def _qp_with_profile(prof_name):
    pipe = RepairPipeline(cfg=REGEN_C2,
                          c2_profile=None if prof_name is None
                          else get_profile(prof_name))
    z, box, ctx, _prev = _prepared()
    pipe.ev.forward(z, box)
    pipe.solver.trace = {}          # activates the slack-row stash
    (P, q, A, lo, hi), n_s = pipe.solver.build_qp(
        z, ctx, trust=REGEN_C2.trust_radius_max, slack=True)
    return pipe, (P, q, A, lo, hi), n_s, ctx


def test_n0_profile_equals_no_profile() -> None:
    z, box, ctx, _ = _prepared()
    outs = []
    for prof in (None, "N0", "N1"):
        pipe = RepairPipeline(cfg=REGEN_C2,
                              c2_profile=None if prof is None
                              else get_profile(prof))
        pipe.ev.forward(z, box)
        (P, q, A, lo, hi), n_s = pipe.solver.build_qp(
            z, ctx, trust=REGEN_C2.trust_radius_max, slack=True)
        outs.append((np.asarray(P.todense()), q, np.asarray(A.todense()),
                     lo, hi, n_s))
    for got in outs[1:]:
        assert np.array_equal(outs[0][0], got[0])
        assert np.array_equal(outs[0][1], got[1])
        assert np.array_equal(outs[0][2], got[2])
        assert np.array_equal(outs[0][3], got[3])
        assert np.array_equal(outs[0][4], got[4])
        assert outs[0][5] == got[5]


def test_n2_row_transformation_and_trust_untouched() -> None:
    pipe0, (P0, q0, A0, lo0, hi0), n0, ctx0 = _qp_with_profile("N1")
    pipe2, (P2, q2, A2, lo2, hi2), n2, _ctx2 = _qp_with_profile("N2")
    A0d, A2d = np.asarray(A0.todense()), np.asarray(A2.todense())
    idx, names = _c2_qp_rows(pipe0.solver, A0, ctx0)
    # C2 block xi-columns: pure /B rescale. The SLACK columns (>= 35) stay
    # -1 in BOTH formulations -- the scaled rows relax by the dimensionless
    # s_norm, the metre rows by s_m = B * s_norm (never B * the -1 too)
    assert np.allclose(A2d[idx][:, :35], A0d[idx][:, :35] / B,
                       rtol=0, atol=1e-15)
    if A2d.shape[1] > 35:
        assert np.array_equal(A2d[idx][:, 35:], A0d[idx][:, 35:])
    def _scaled_bound(x2, x1):
        fin = (np.abs(x1) < 1e8) & (np.abs(x2) < 1e8)
        assert np.array_equal(np.abs(x2[~fin]) >= 1e8 - 1,
                              np.abs(x1[~fin]) >= 1e8 - 1), "sentinel mismatch"
        assert np.allclose(x2[fin], x1[fin] / B, rtol=0, atol=1e-15)

    _scaled_bound(lo2[idx], lo0[idx])
    _scaled_bound(hi2[idx], hi0[idx])
    # everything else byte-identical (P/q included: the objective is not
    # transformed, only the C2 constraint ROWS). The DUPLICATED relaxation
    # rows (>= n_r) carry the scaled C2 rows by construction -- compare
    # within the original row range only
    n_r = A0d.shape[0] - 2 * n0
    other = [i for i in range(n_r) if i not in idx]
    assert np.array_equal(A2d[other], A0d[other])
    assert np.array_equal(lo2[other], lo0[other])
    assert np.array_equal(hi2[other], hi0[other])
    assert np.array_equal(np.asarray(P2.todense()), np.asarray(P0.todense()))
    # q: the xi block is untouched; the slack-penalty block differs ONLY
    # on the C2 entries (10000*B -- pinned exactly in the units test)
    assert np.array_equal(q2[:35], q0[:35])
    st0, st2 = pipe0.solver.last_slack_rows, pipe2.solver.last_slack_rows
    for k, i in enumerate(st2["idx"]):
        if not names[i].startswith("hand_speed["):
            assert q2[35 + k] == q0[35 + k] == 10000.0
    # trust rows are never relaxed and never rescaled
    tnames = pipe0.solver._qp_row_names(A0.shape[0], ctx0)
    trows = [i for i, n in enumerate(tnames)
             if n.startswith("trust[") and i < n_r]
    assert np.array_equal(A2d[trows], A0d[trows])
    assert n0 == n2


def test_n2_slack_penalty_units() -> None:
    pipe0, _qp0, n0, _ctx0 = _qp_with_profile("N1")
    pipe2, (P2, q2, A2, lo2, hi2), n2, ctx2 = _qp_with_profile("N2")
    st0 = pipe0.solver.last_slack_rows
    st2 = pipe2.solver.last_slack_rows
    assert st0["idx"] == st2["idx"] and st0["n_s"] == st2["n_s"]
    names = pipe2.solver._qp_row_names(A2.shape[0], ctx2)
    is_c2 = [names[i].startswith("hand_speed[") for i in st2["idx"]]
    for k, (c2, p0, p2) in enumerate(zip(is_c2, st0["penalty"],
                                        st2["penalty"])):
        if c2:
            assert p2 == pytest.approx(10000.0 * B, rel=0, abs=0)
        else:
            assert p2 == p0 == 10000.0
    assert any(is_c2), "no C2 slack rows were relaxed in this setup"


def test_n2_tight_and_slack_solutions_correspond() -> None:
    z, box, ctx, _ = _prepared()
    # both formulations solved at the SAME high-precision settings: the
    # mathematical equivalence is exact, so the solutions must agree to
    # solver accuracy (at the default eps the two row scalings legitimately
    # differ inside their own tolerance -- that is N2's conditioning point,
    # not an equivalence violation)
    hp = get_profile("N1").solver
    sols = {}
    for prof in ("N1", "N2"):
        pipe = RepairPipeline(cfg=REGEN_C2, c2_profile=get_profile(prof))
        pipe.ev.forward(z, box)
        (P, q, A, lo, hi), _ = pipe.solver.build_qp(
            z, ctx, trust=REGEN_C2.trust_radius_max, slack=False)
        x, status = pipe.solver.solve_qp(P, q, A, lo, hi, settings=hp)
        assert x is not None, status
        (Ps, qs, As, los, his), n_s = pipe.solver.build_qp(
            z, ctx, trust=REGEN_C2.trust_radius_max, slack=True)
        xs, status_s = pipe.solver.solve_qp(Ps, qs, As, los, his, settings=hp)
        assert xs is not None, status_s
        sols[prof] = (x, xs, n_s)
    x1, xs1, n1 = sols["N1"]
    x2, xs2, n2 = sols["N2"]
    # same constraint set -> same tight step
    assert np.allclose(x1, x2, rtol=0, atol=1e-7)
    # slack QP: same step; C2 slacks satisfy s_m = B * s_norm
    assert np.allclose(xs1[:35], xs2[:35], rtol=0, atol=1e-7)
    assert n1 == n2
    pipe = RepairPipeline(cfg=REGEN_C2, c2_profile=get_profile("N2"))
    pipe.ev.forward(z, box)
    pipe.solver.trace = {}
    (_P, _q, A2, _lo, _hi), _ns = pipe.solver.build_qp(
        z, ctx, trust=REGEN_C2.trust_radius_max, slack=True)
    st = pipe.solver.last_slack_rows
    names = pipe.solver._qp_row_names(A2.shape[0], ctx)
    offs = {n: i for i, n in enumerate(names)}
    c2_slack_idx = [k for k, i in enumerate(st["idx"])
                    if names[i].startswith("hand_speed[")]
    other_idx = [k for k in range(n2) if k not in c2_slack_idx]
    for k in c2_slack_idx:
        s_norm = xs2[35 + k]
        s_m = xs1[35 + k]
        assert s_m == pytest.approx(B * s_norm, rel=1e-6, abs=1e-12), (
            f"slack {k}: metre {s_m} != B*norm {B * s_norm}")


def test_real_acceptance_rows_unchanged_under_n2() -> None:
    z, box, ctx, _ = _prepared()
    outs = []
    for prof in (None, get_profile("N2")):
        pipe = RepairPipeline(cfg=REGEN_C2, c2_profile=prof)
        pipe.ev.forward(z, box)
        val, lo, hi, scl, names, _ids = pipe.solver.check_rows(z, ctx)
        outs.append((val, lo, hi, scl, list(names)))
    for a, b in zip(outs[0], outs[1]):
        assert np.array_equal(a, b)



def test_n3_mu_penalty_on_c2_slack_only() -> None:
    """N3: the dimensionless C2 slack penalty equals the CURRENT merit
    weight mu (the one solve_frame computes, incl. the regen multiplier);
    every other family keeps 10000; missing mu is an explicit error, and
    the tight QP is unchanged (mu only prices the recovery slack)."""
    from ireset.repair.profiles import C2Profile, OsqpSettings

    pipe0, _qp0, n0, ctx0 = _qp_with_profile("N2")
    z, box, ctx, _ = _prepared()
    pipe = RepairPipeline(cfg=REGEN_C2, c2_profile=get_profile("N3"))
    pipe.ev.forward(z, box)
    pipe.solver.trace = {}
    mu = 10.0 * max(1.0, 1.0 + 0.1 + 0.1) * 30.0     # solve_frame's formula
    (_P, _q, A, _lo, _hi), _n = pipe.solver.build_qp(
        z, ctx, trust=8.0, slack=True, mu=mu)
    st = pipe.solver.last_slack_rows
    names = pipe.solver._qp_row_names(A.shape[0], ctx)
    seen_c2 = 0
    for k, i in enumerate(st["idx"]):
        if names[i].startswith("hand_speed["):
            assert st["penalty"][k] == mu
            seen_c2 += 1
        else:
            assert st["penalty"][k] == 10000.0
    assert seen_c2 > 0
    # missing mu with the mu mode attached is an explicit error
    with pytest.raises(ValueError, match="mu"):
        pipe.solver.build_qp(z, ctx, trust=8.0, slack=True)
    # the tight QP has no slack to price: N3 == N2 there
    pipe2 = RepairPipeline(cfg=REGEN_C2, c2_profile=get_profile("N2"))
    pipe2.ev.forward(z, box)
    (P3, q3, A3, l3, h3), _ = pipe.solver.build_qp(
        z, ctx, trust=8.0, slack=False)
    (P2, q2, A2, l2, h2), _ = pipe2.solver.build_qp(
        z, ctx, trust=8.0, slack=False)
    assert np.array_equal(np.asarray(P3.todense()), np.asarray(P2.todense()))
    assert np.array_equal(q3, q2)
    assert np.array_equal(np.asarray(A3.todense()), np.asarray(A2.todense()))
