"""F1: regen endpoint acceptance band-floor separation (train24fix plan
section 4.2 / the section-2.4 boundary matrix).

Exactly ONE cell of the matrix changes: regen mode, endpoint acceptance
verdict, band rows (ordinary finite-upper contact band or the active regen
row) move -1.5mm -> -1.8mm (``cfg_noncontact``). Everything else is pinned
by tests: the QP DRIVING floor keeps the dynamic solver band margin
(-2mm + m(ctx), i.e. -1.5mm at 2cm motion, +1mm at 10cm, +4mm at 16cm),
standard-mode acceptance stays -2mm, explicit probe floors stay -2mm, the
probe-mode regen row's effective floor stays -1.8mm (infinite upper ->
accept override), and the independent validator keeps -2mm.

The scenario is built by bisecting the box pose so the nearest left-hand
geom sits at a TARGET signed distance (e.g. -1.6mm): the D-F counterexample
class (band-floor-only candidates) reproduced synthetically.
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
from ireset.repair.scp import (  # noqa: E402
    INF,
    FrameContext,
    FrameSolver,
    cfg_noncontact,
    solver_band_margin,
)
from test_repair import standing_scene  # noqa: E402

REGEN = replace(RepairConfig(), regen_mode=True)


def _place_box(pipe: RepairPipeline, z: np.ndarray, side: str,
               target_m: float) -> np.ndarray:
    """Bisect a box translation along +x so the nearest ``side`` hand geom
    sits at ``target_m`` signed distance from the box."""
    pipe.ev.forward(z, np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0]))
    gids = pipe.ev._hand_geoms[side]
    center = pipe.ev.data.geom_xpos[gids[0]].copy()   # rubber_hand geom ref

    def dist_at(t: float) -> float:
        box = np.array([center[0] + 0.25 + t, center[1], center[2], 1.0, 0, 0, 0])
        pipe.ev.forward(z, box)
        return min(pipe.ev.geom_distance(g, pipe.ev.box_geom)[0] for g in gids)

    lo, hi = -0.30, 0.30          # hi: clear, lo: deep inside
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dist_at(mid) > target_m:
            hi = mid
        else:
            lo = mid
    t = 0.5 * (lo + hi)
    box = np.array([center[0] + 0.25 + t, center[1], center[2], 1.0, 0, 0, 0])
    pipe.ev.forward(z, box)
    d = min(pipe.ev.geom_distance(g, pipe.ev.box_geom)[0] for g in gids)
    assert abs(d - target_m) < 2e-5, f"bisection failed: {d} vs {target_m}"
    return box


def _ctx(pipe, z, box, motion: float = 0.02, z_prev=None,
         regen: bool = True, foot_anchor=None, prev_sole=None) -> FrameContext:
    pipe.ev.forward(z, box)
    ref = pipe.ev.site_positions(pipe.ev.kpt_ids).copy()
    hold = {"left_hand": True} if regen else {}
    return FrameContext(
        t=1, box_pose=box.copy(), z_pred=z.copy(), z_prev=z_prev,
        ref_kpts_world=ref, mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor=foot_anchor or {},
        prev_hand_local={}, prev_sole_world=prev_sole or {},
        box_pose_prev=(box.copy() if z_prev is not None else None),
        ref_hold=hold, prev_hand_dist=({"l": 0.008} if regen else {}),
        frame_motion_m=motion)


def _band_rows(solver: FrameSolver, val, lo, hi):
    """Collision-block (name, val, lo, hi) rows with a finite upper bound
    (the codebase's INF sentinel is 1e9, not numpy inf)."""
    names, _ids = solver.row_descriptors()
    out = []
    for k in range(99, 99 + len(solver.last_row_pairs)):
        if hi[k] < INF / 2:
            out.append((names[k], float(val[k]), float(lo[k]), float(hi[k])))
    return out


@pytest.fixture(scope="module")
def scene16():
    """Standing state with the left hand at -1.6mm from the box (the D-F
    band_floor_only class) and the box clear of everything else."""
    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", -0.0016)
    return pipe, z, box


# --- the matrix --------------------------------------------------------------------


def test_accept_regen_band_floor_is_validator_grade(scene16) -> None:
    """F1 cell: regen endpoint ACCEPTANCE floor = cfg_noncontact (-1.8mm)
    for BOTH the active regen row and ordinary finite-upper band rows."""
    pipe, z, box = scene16
    ctx = _ctx(pipe, z, box)
    val, lo, hi, scl, names, _ids = pipe.solver.check_rows(z, ctx)
    rows = _band_rows(pipe.solver, val, lo, hi)
    regen_rows = [r for r in rows if r[0].endswith(":regen")]
    assert regen_rows, "ref_hold must produce an active regen row"
    for name, v, l_, _h in regen_rows:
        assert l_ == pytest.approx(cfg_noncontact(REGEN), abs=1e-15)
        assert v == pytest.approx(-0.0016, abs=2e-5)
    for name, v, l_, _h in rows:
        assert l_ == pytest.approx(cfg_noncontact(REGEN), abs=1e-15)


def test_qp_driving_floor_keeps_dynamic_margin(scene16) -> None:
    """QP driving floor is untouched: -2mm + m(ctx) with m from
    solver_band_margin (0.5 / 3 / 6 mm at 0.02 / 0.1 / 0.16 m motion)."""
    pipe, z, box = scene16
    for motion, want in ((0.02, -0.0015), (0.10, 0.001), (0.16, 0.004)):
        ctx = _ctx(pipe, z, box, motion=motion)
        assert solver_band_margin(ctx, REGEN) == pytest.approx(
            {"0.02": 0.0005, "0.10": 0.003, "0.16": 0.006}[f"{motion:.2f}"],
            abs=1e-12)
        pipe.ev.forward(z, ctx.box_pose)
        rows = pipe.solver._collision_rows(ctx, solver_band_margin(ctx, REGEN))
        floors = {lo for _d, _J, lo, _hi in rows if _hi < INF / 2}
        assert floors and all(abs(f - want) < 1e-12 for f in floors), \
            (motion, floors)


def test_standard_mode_accept_band_floor_unchanged(scene16) -> None:
    """Standard mode keeps the historical -2mm acceptance floor on band
    rows (no regen row exists there)."""
    std_pipe = RepairPipeline(cfg=RepairConfig())
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(std_pipe, z, "l", -0.0016)
    ctx = _ctx(std_pipe, z, box, regen=False)
    val, lo, hi, scl, names, _ids = std_pipe.solver.check_rows(z, ctx)
    rows = _band_rows(std_pipe.solver, val, lo, hi)
    assert rows and not any(r[0].endswith(":regen") for r in rows)
    for _n, v, l_, _h in rows:
        assert l_ == pytest.approx(-0.002, abs=1e-15)


def test_probe_floors_unchanged(scene16) -> None:
    """Explicit probe floor (band_margin 0) stays -2mm on ordinary band
    rows; the probe-mode regen row drops its upper (INF) and its effective
    floor stays the accept override -1.8mm."""
    pipe, z, box = scene16
    ctx = _ctx(pipe, z, box)
    val, lo, hi, scl, names, _ids = pipe.solver.check_rows(
        z, ctx, band_margin_m=0.0, probe_mode=True)
    names_all, _ = pipe.solver.row_descriptors()
    seen_band = seen_regen_probe = False
    for k in range(99, 99 + len(names)):
        nm = names[k - 99]
        if hi[k] >= INF / 2 and not nm.endswith(":regen"):
            continue
        if nm.endswith(":regen"):
            # probe_mode strips the upper: effective floor via accept override
            assert hi[k] >= INF / 2
            assert lo[k] == pytest.approx(cfg_noncontact(REGEN), abs=1e-15)
            seen_regen_probe = True
        else:
            assert lo[k] == pytest.approx(-0.002, abs=1e-15)
            seen_band = True
    assert seen_band and seen_regen_probe


# --- behaviour: the D-F counterexample class ----------------------------------------


def test_band_floor_only_candidate_now_accepts(scene16) -> None:
    """A -1.6mm hand: the endpoint acceptance no longer rejects it (D-F's
    band_floor_only class); the same row arithmetic under the OLD -1.5mm
    floor yields a 0.05 normalized violation (the rejection the frozen run
    produced). A -1.9mm hand is STILL rejected by the acceptance floor."""
    pipe, z, box = scene16
    ctx = _ctx(pipe, z, box, z_prev=None)      # no probes / no velocity gate
    chk = pipe.frame_valid(z, ctx)
    assert chk.ok, chk.rows

    # the old-floor arithmetic, evaluated on the same assembled rows: every
    # finite-upper collision row that now sits at -1.8mm was at -1.5mm, and
    # the -1.6mm value violates it by 0.1mm -> normalized 0.05 >> 1e-6
    val, lo, hi, scl, _n, _i = pipe.solver.check_rows(z, ctx)
    old_lo = lo.copy()
    n_coll = len(_n)
    for k in range(99, 99 + n_coll):
        if hi[k] < INF / 2 and lo[k] == pytest.approx(cfg_noncontact(REGEN)):
            old_lo[k] = -0.0015
    assert (old_lo != lo).any()
    assert pipe.solver.violation(val, old_lo, hi, scl) > 1e-6

    # deeper: -1.9mm still violates the acceptance floor (and the validator)
    pipe.ev.forward(z, np.array([5.0, 0.25, 0.25, 1.0, 0, 0, 0]))
    box19 = _place_box(pipe, z, "l", -0.0019)
    ctx19 = _ctx(pipe, z, box19, z_prev=None)
    chk19 = pipe.frame_valid(z, ctx19)
    assert not chk19.ok and chk19.reason == "state_constraints"
    assert any(":regen" in str(r.get("pair", "")) for r in chk19.rows)


def test_band_legal_but_other_rows_still_reject(scene16) -> None:
    """A band-legal (-1.6mm) candidate is still rejected when a joint limit
    or a support-foot speed row violates: the floor change frees ONLY the
    band cell, the verdict still needs every other row. (The pipeline's
    ``rows`` listing carries collision pairs only, so the blocking rows are
    asserted on the assembled check_rows arrays.)"""
    pipe, z, box = scene16
    # joint limit: push joint 16 past its lower bound
    z_bad = z.copy()
    j = 16
    z_bad[7 + j] = pipe.ev.spec.joint_range[j, 0] - 1e-4
    ctx = _ctx(pipe, z_bad, box, z_prev=None)
    chk = pipe.frame_valid(z_bad, ctx)
    assert not chk.ok and chk.reason == "state_constraints"
    val, lo, hi, _s, _n, _i = pipe.solver.check_rows(z_bad, ctx)
    names, _ = pipe.solver.row_descriptors()
    jl = [k for k in range(29) if val[k] < lo[k] - 1e-12]
    assert jl and all(names[k].startswith("joint_limit") for k in jl)

    # support-foot speed: an anchor plus a distant previous sole -> the
    # per-site displacement rows exceed dt * foot_speed_mps
    pipe.ev.forward(z, box)
    prev_sole = {"l": pipe.ev.site_positions(pipe.ev.sole_sites["l"]) + 0.05}
    foot_anchor = {"l": pipe.ev.site_positions(pipe.ev.sole_sites["l"])}
    ctx_f = _ctx(pipe, z, box, z_prev=z.copy(),
                 foot_anchor=foot_anchor, prev_sole=prev_sole)
    chk_f = pipe.frame_valid(z, ctx_f)
    assert not chk_f.ok
    val_f, lo_f, hi_f, _sf, _nf, _iff = pipe.solver.check_rows(z, ctx_f)
    names_f, _ = pipe.solver.row_descriptors()
    fs = [k for k in range(len(val_f))
          if names_f[k].startswith("foot_speed")
          and (val_f[k] < lo_f[k] - 1e-12 or val_f[k] > hi_f[k] + 1e-12)]
    assert fs


def test_validator_penetration_gate_unchanged(scene16) -> None:
    """The independent validator keeps its own -2mm gate: -1.6mm passes,
    -2.1mm fails -- F1 moved an INTERNAL acceptance floor, never the final
    check."""
    from ireset.repair.validation import IndependentValidator

    pipe, z, box = scene16
    v = IndependentValidator(spec=pipe.spec)
    scene = np.zeros((1, 43))
    scene[0, 0:36] = z
    scene[0, 36:43] = box
    rep = v.validate(scene, exclude=("box_ground",))
    assert rep.checks["penetration"].passed

    pipe.ev.forward(z, np.array([5.0, 0.25, 0.25, 1.0, 0, 0, 0]))
    box_deep = _place_box(pipe, z, "l", -0.0021)
    scene[0, 36:43] = box_deep
    rep_deep = v.validate(scene, exclude=("box_ground",))
    assert not rep_deep.checks["penetration"].passed
