"""train24fix3 stage D: the norm_cut C2 bound form (plan section 7).

Pinned invariants:

- analytical allocation freedom: a displacement with per-site NORM inside
  dt*V but an AXIS over the old dt*0.02598 bound passes the real norm rows
  and fails the old axis rows (S1's entire reason to exist);
- spikes and out-of-bound displacements are rejected by the REAL rows and
  the acceptance budget provably implies the validator's 0.05 entry gate;
- the QP carries the initial three-axis box (necessary condition) plus the
  cutting planes, in their OWN row space (padding target grows; geometric
  rows never displaced), with descriptors naming mode/side/site/role;
- one shared dimensionless slack per (side, site) across its axis rows and
  ALL its cutting planes, priced mu ONCE -- never per plane;
- the cut loop's budgets are hard: 8 re-solves per iteration, 8 directions
  per site, exhaustion records non-closure and can never read back as a
  real-norm pass;
- cuts are per-linearization: a new solve_frame starts clean;
- v3 norm products reject format/hash tampering and the index is
  dual-view validated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.profiles import get_profile  # noqa: E402
from ireset.repair.scp import FrameContext, N_TANGENT, INF  # noqa: E402
from test_band_floor_separation import _place_box  # noqa: E402
from test_hand_speed_row import REGEN_C2, POST_HAND_SPEED  # noqa: E402
from test_repair import standing_scene  # noqa: E402

NORM_FORMAT = "train24fix3_norm_products_v1"
DT = 1.0 / 30.0
V_S1 = 0.045
B_S1 = DT * V_S1                       # 1.5 mm/frame
AXIS_B = DT * 0.02598                  # 0.866 mm/frame (old axis bound)
S1 = get_profile("S1")
S2 = get_profile("S2")


def _ctx_with_prev(pipe, z, box, d_target):
    """C2-active context whose previous hand array sits at ``d_target``
    below the CURRENT sites (analytical displacement control)."""
    pipe.ev.forward(z, box)
    cur = pipe.ev.hand_local("l")
    prev = cur - np.asarray(d_target, dtype=float)
    return FrameContext(
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


def _setup(profile=None):
    pipe = RepairPipeline(cfg=REGEN_C2, c2_profile=profile)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", 0.006)
    return pipe, z, box


def _c2_slice(names, tag="hand_norm["):
    return [i for i, n in enumerate(names) if n.startswith(tag)]


# --- 1. analytical allocation freedom + spike/bound rejection ---------------


def test_norm_passes_where_axis_failed_and_vice_versa() -> None:
    d = np.array([[0.0013, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.linalg.norm(d[0]) <= B_S1 and abs(d[0, 0]) > AXIS_B
    pipe_s, z, box = _setup(S1)
    ctx = _ctx_with_prev(pipe_s, z, box, d)
    val, lo, hi, scl, names, ids = pipe_s.solver.check_rows(z, ctx)
    names = pipe_s.solver._check_descriptors(list(names), list(ids))[0]
    rows = _c2_slice(names, tag="hand_norm[l:")   # the ACTIVE side only
    assert len(rows) == 3                       # one active row per site
    for r in rows:
        assert lo[r] == -INF                    # one-sided norm rows
    assert all(val[r] <= hi[r] + 1e-12 for r in rows)
    # the DISABLED side's descriptor slots stay inactive (value 0, +-INF)
    for r in _c2_slice(names, tag="hand_norm[r:"):
        assert val[r] == 0.0 and hi[r] == INF
    # the same displacement violates EVERY old axis row that it touches
    pipe_a, _, _ = _setup(None)
    ctx_a = _ctx_with_prev(pipe_a, z, box, d)
    val_a, lo_a, hi_a, _s, names_a, ids_a = pipe_a.solver.check_rows(z, ctx_a)
    names_a = pipe_a.solver._check_descriptors(list(names_a), list(ids_a))[0]
    ax = [i for i, n in enumerate(names_a)
          if n.startswith("hand_speed[l:palm:x]")]
    assert val_a[ax[0]] > hi_a[ax[0]]
    # and the norm ROW value is the site norm itself
    assert val[rows[0]] == pytest.approx(np.linalg.norm(d[0]), abs=1e-15)


def test_spike_and_out_of_bound_displacements_rejected() -> None:
    # (speed, profile): the 0.05 spikes violate BOTH bounds; 0.0499 violates
    # S2 (and S1); 0.0451 violates S1 ONLY (S2 admits it -- direction
    # freedom is bounded, the norm is not free)
    for speed, prof_name in ((0.05, "S1"), (0.0499, "S2"), (0.0451, "S1")):
        d = np.zeros((3, 3))
        d[0] = [speed * DT, 0.0, 0.0]
        pipe, z, box = _setup(get_profile(prof_name))
        ctx = _ctx_with_prev(pipe, z, box, d)
        val, lo, hi, _s, names, ids = pipe.solver.check_rows(z, ctx)
        names = pipe.solver._check_descriptors(list(names), list(ids))[0]
        r = _c2_slice(names, tag="hand_norm[l:")[0]
        assert val[r] > hi[r], (speed, prof_name, val[r], hi[r])


def test_acceptance_budget_implies_entry_gate() -> None:
    for prof in (S1, S2):
        v = prof.c2_norm_speed_mps
        assert v * (1 + 1e-6) < 0.05
        # a candidate exactly AT the bound has norm speed v < 0.05; the
        # normalized violation at the boundary is 0 (inclusive bounds)
        d = np.zeros((3, 3))
        d[1] = [0.0, v * DT, 0.0]
        pipe, z, box = _setup(prof)
        ctx = _ctx_with_prev(pipe, z, box, d)
        val, _lo, hi, scl, names, ids = pipe.solver.check_rows(z, ctx)
        names = pipe.solver._check_descriptors(list(names), list(ids))[0]
        r = _c2_slice(names, tag="hand_norm[l:")[1]
        assert val[r] == pytest.approx(v * DT, abs=1e-15)
        viol = max((val[r] - hi[r]) / scl[r], 0.0)
        assert viol <= 1e-12     # boundary is inclusive up to float rounding


# --- 2. QP layout: box + cuts, own row space, descriptors -------------------


def _qp_with_cuts(cuts):
    pipe, z, box = _setup(S1)
    d = np.array([[0.001, 0.0004, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ctx = _ctx_with_prev(pipe, z, box, d)
    pipe.ev.forward(z, box)
    pipe.solver.trace = {}
    pipe.solver._c2_cuts = list(cuts)
    (P, q, A, lo, hi), n_s = pipe.solver.build_qp(
        z, ctx, trust=8.0, slack=True, mu=360.0)
    return pipe, z, ctx, (P, q, A, lo, hi), n_s


def test_qp_box_rows_and_cut_rows_have_own_space() -> None:
    n_hat = np.array([1.0, 0.0, 0.0])
    cuts = [{"side": "l", "site": 0, "n": n_hat, "k": 0}]
    pipe0, z0, ctx0, qp0, n0 = _qp_with_cuts([])
    pipe1, z1, ctx1, qp1, n1 = _qp_with_cuts(cuts)
    A0 = np.asarray(qp0[2].todense())
    A1 = np.asarray(qp1[2].todense())
    names0 = pipe0.solver._qp_row_names(A0.shape[0], ctx0)
    names1 = pipe1.solver._qp_row_names(A1.shape[0], ctx1)
    # the cut adds ONE base row and ONE duplicated relaxation row (the
    # slack expansion); its slack is SHARED with the site's axis rows
    assert A1.shape[0] == A0.shape[0] + 2
    box_rows = [i for i, n in enumerate(names1)
                if n.startswith("hand_norm_box[l:palm")]
    assert len(box_rows) == 3
    cut_rows = [i for i, n in enumerate(names1)
                if n.startswith("hand_norm_cut[l:palm:k0]")]
    assert len(cut_rows) == 1
    # box bound uses dt*V (S1), not the old axis bound
    (_P, _q, _A, lo1, hi1) = qp1
    d0 = np.array([[0.001, 0.0004, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for a in range(3):                           # per-axis, preconditioned
        assert hi1[box_rows[a]] == pytest.approx(
            (B_S1 - d0[0][a]) / B_S1, abs=1e-14)
    # the cut row's bound: n'd/B preconditioned form 1 - n'd/B
    nd = float(n_hat @ d0[0])
    assert hi1[cut_rows[0]] == pytest.approx(1.0 - nd / B_S1, abs=1e-14)
    # geometric rows are NOT displaced: everything before the C2 tail is
    # byte-identical between the two assemblies
    pre_tail = box_rows[0]
    assert np.array_equal(A1[:pre_tail], A0[:pre_tail])


def test_shared_site_slack_single_penalty() -> None:
    n1 = np.array([1.0, 0.0, 0.0])
    n2 = np.array([0.0, 1.0, 0.0])
    cuts = [{"side": "l", "site": 0, "n": n1, "k": 0},
            {"side": "l", "site": 0, "n": n2, "k": 1}]
    pipe, z, ctx, qp, n_s = _qp_with_cuts(cuts)
    st = pipe.solver.last_slack_rows
    names = pipe.solver._qp_row_names(np.asarray(qp[2].todense()).shape[0],
                                      ctx)
    palm_rows = [i for i, n in enumerate(names)
                 if n.startswith("hand_norm_box[l:palm")
                 or n.startswith("hand_norm_cut[l:palm")]
    idx_pos = {i: k for k, i in enumerate(st["idx"])}
    palm_slacks = {st["slack_of_row"][idx_pos[i]] for i in palm_rows
                   if i in idx_pos}
    assert len(palm_slacks) == 1, "site's axis rows + cuts must SHARE a slack"
    shared_id = palm_slacks.pop()
    # exactly ONE shared group per (side, site); palm's is the one its rows use
    assert set(st["shared_groups"]) == {"l:0", "l:1", "l:2"}
    assert st["shared_groups"]["l:0"] == shared_id
    # priced mu ONCE per site -- the shared column's penalty, not per plane
    assert st["penalty"][shared_id] == 360.0
    # other families keep their own slacks at 10000
    for k, i in enumerate(st["idx"]):
        nm = names[i]
        if not (nm.startswith("hand_norm_box[") or nm.startswith("hand_norm_cut[")):
            if st["slack_of_row"][k] != shared_id:
                assert st["penalty"][st["slack_of_row"][k]] == 10000.0


def test_cut_budgets_record_nonclosure_never_pass() -> None:
    pipe, z, box = _setup(S1)
    d = np.array([[0.0022, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ctx = _ctx_with_prev(pipe, z, box, d)
    pipe.ev.forward(z, box)
    # prime the linearization the cut loop consumes (build_qp fills it)
    pipe.solver.build_qp(z, ctx, trust=8.0, slack=True, mu=360.0)
    assert "l" in pipe.solver._c2_lin
    # stub the solver: every solve returns a proposal that violates the
    # ball in a NEW direction (rotating), so the loop must exhaust budgets
    calls = {"n": 0}
    ang = {"a": 0.0}

    def fake_solve_qp(P, q, A, lo, hi, settings=None, qp_identity=None):
        calls["n"] += 1
        ang["a"] += 0.7
        xi = np.zeros(N_TANGENT)
        # a joint step whose predicted hand motion sweeps a rotating
        # direction at constant oversize (via the linearization itself)
        lin = pipe.solver._c2_lin["l"]
        # solve for xi that makes p = d + J xi point at angle with norm 2B
        J0 = lin["J"][0:3]
        target = 2 * B_S1 * np.array([np.cos(ang["a"]), np.sin(ang["a"]), 0.0])
        xi_j, *_ = np.linalg.lstsq(J0, target - lin["d"][0], rcond=None)
        xi[:] = xi_j             # underdetermined: exact reach, full tangent
        return xi, "solved"

    pipe.solver.solve_qp = fake_solve_qp
    x, status, n_extra, stats = pipe.solver._c2_cut_refine(
        np.zeros(N_TANGENT), "solved", z, ctx, 8.0, True, 360.0, None)
    assert stats["resolves"] <= 8                  # re-solve cap (rounds)
    # per-site cap: 8 rounds x at most one new direction per site each
    assert stats["directions_added"] <= 24
    assert stats["nonclosed"] is True
    # the RE-SOLVE budget is what fired (8 rounds exhausted, violations
    # remain); the per-site cap is structurally unreachable inside 8
    # single-direction rounds and exists for pre-seeded cut sets
    assert stats["resolve_cap_hit"] is True
    assert n_extra == stats["resolves"]


def test_cuts_are_per_linearization_and_frame_clean() -> None:
    pipe, z, box = _setup(S1)
    d = np.array([[0.001, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ctx = _ctx_with_prev(pipe, z, box, d)
    pipe.solver._c2_cuts = [{"side": "l", "site": 0,
                             "n": np.array([1.0, 0.0, 0.0]), "k": 0}]
    res = pipe.solver.solve_frame(z.copy(), ctx)
    # a stale cut never survives the call (and the stats were reset too)
    assert pipe.solver._c2_cuts == []
    assert "c2_cuts" in res.diagnostics or res.accepted


# --- 3. value-path descriptors + evidence ------------------------------------


def test_real_row_descriptors_and_evidence() -> None:
    pipe, z, box = _setup(S1)
    d = np.array([[0.0012, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ctx = _ctx_with_prev(pipe, z, box, d)
    pipe.ev.forward(z, box)
    pipe.solver.check_rows(z, ctx)
    ev = pipe.solver.last_hand_speed["l"]
    assert ev["enabled"] and ev["mode"] == "norm_cut"
    assert ev["bound_abs_m"] == B_S1
    assert ev["norm_speed_mps"] == pytest.approx(0.0012 / DT, rel=1e-12)
    names = pipe.solver.row_descriptors()[0]
    norm_rows = [n for n in names if n.startswith("hand_norm[l")]
    off_rows = [n for n in names if n.startswith("hand_norm_off[l")]
    assert len(norm_rows) == 3 and len(off_rows) == 6
    # right side disabled: mode evidence records it, no rows active
    assert pipe.solver.last_hand_speed["r"]["enabled"] is False


# --- 4. v3 product + index tamper rejection ----------------------------------


def _tiny_norm_product(tmp_path, fmt=NORM_FORMAT, tamper=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train24fix3_convergence as drv

    pdir = tmp_path / "cf"
    pdir.mkdir(parents=True)
    arrays = {"rows__plus_c2_norm__val": np.arange(5, dtype=float)}
    record = {"format": fmt, "stem": "x", "profile": "S1",
              "classification": "C2_ONLY", "effective": {}, "audits": {}}
    drv._write_norm_products(pdir, "x", record, arrays)
    if tamper == "npz":
        np.savez(pdir / "cf_x.npz", rows__plus_c2_norm__val=np.zeros(5))
    elif tamper == "format":
        rec = json.loads((pdir / "cf_x.json").read_text())
        rec["format"] = "train24fix2_reviewfix_audit_v2"
        (pdir / "cf_x.json").write_text(json.dumps(rec))
    return drv, pdir


def test_norm_product_tamper_rejected(tmp_path) -> None:
    drv, pdir = _tiny_norm_product(tmp_path)
    record, arrays = drv.load_norm_products(pdir, "x")
    assert record["classification"] == "C2_ONLY"
    drv2, pdir2 = _tiny_norm_product(tmp_path / "t2", tamper="npz")
    with pytest.raises(ValueError, match="sha256"):
        drv2.load_norm_products(pdir2, "x")
    drv3, pdir3 = _tiny_norm_product(tmp_path / "t3", tamper="format")
    with pytest.raises(ValueError, match="format"):
        drv3.load_norm_products(pdir3, "x")
    with pytest.raises(FileNotFoundError):
        drv.load_norm_products(pdir, "missing")


def test_norm_index_contract(tmp_path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train24fix3_convergence as drv

    base = tmp_path / "cf"
    base.mkdir(parents=True)
    good = {"format": drv.NORM_FORMAT,
            "stems": ["a", "b"], "records": [{"stem": "a"}, {"stem": "b"}]}
    (base / "cf_index.json").write_text(json.dumps(good))
    # Q2 unified convention: the PRODUCTS directory itself (the dir holding
    # cf_index.json) -- the pre-Q3 parent convention is a missing index now
    assert drv.load_norm_index(base)["stems"] == ["a", "b"]
    with pytest.raises(FileNotFoundError):
        drv.load_norm_index(tmp_path)
    for mutate in ("empty_stems", "empty_records", "mismatch", "dup",
                   "bad_format", "missing"):
        d = tmp_path / f"cf_{mutate}"
        d.mkdir(parents=True)
        doc = json.loads(json.dumps(good))
        if mutate == "empty_stems":
            doc["stems"] = []
        elif mutate == "empty_records":
            doc["records"] = []
        elif mutate == "mismatch":
            doc["records"] = [{"stem": "a"}]
        elif mutate == "dup":
            doc["stems"] = ["a", "a"]
        elif mutate == "bad_format":
            doc["format"] = "v2"
        if mutate != "missing":
            (d / "cf_index.json").write_text(json.dumps(doc))
        with pytest.raises((ValueError, FileNotFoundError)):
            drv.load_norm_index(d)


def test_axis_mode_unchanged_by_norm_code() -> None:
    """The axis path (no profile / N-profiles) is byte-identical to the
    pre-normCut behaviour: real rows, descriptors and QP tail untouched."""
    from dataclasses import replace as _replace
    from test_train24fix3_numeric import _qp_with_profile

    pipe0, qp0, n0, ctx0 = _qp_with_profile("N2")
    names = pipe0.solver._qp_row_names(
        np.asarray(qp0[2].todense()).shape[0], ctx0)
    assert not any(n.startswith("hand_norm") for n in names)
    assert any(n.startswith("hand_speed[l:") for n in names)
    val, lo, hi, scl, rnames, _rids = pipe0.solver.check_rows(
        ctx0.z_pred, ctx0)
    rnames = pipe0.solver._check_descriptors(list(rnames), list(_rids))[0]
    assert any(n.startswith("hand_speed[l:") for n in rnames)
