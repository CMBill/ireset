"""R2/R1: active-row coverage for reference-hold sides + the unified
corridor acceptance contract (train24fix plan section 4.3).

D-R frozen evidence this file pins:

- sub10_001 frames 30-32: ``ref_hold`` true with the hand 9.4-12.6 cm away,
  ``corridor`` EMPTY in every check (the ~5 cm activation gate never admits
  the pair), and a fresh solver returning ``accepted=True, qp_calls=0,
  state unchanged`` from the feasible+mpjpe+stationary early return --
  "calling solve_frame" never meant the corridor participated;
- R1 was NOT independently triggered (no adopted frame violated an ACTIVE
  corridor): once R2 makes the row exist everywhere, the existing check
  chain (skip check, solver feasibility, success review, adoption) IS the
  contract -- these tests verify it at every exit.
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
from ireset.repair.scp import FrameContext, FrameSolver  # noqa: E402
from test_band_floor_separation import _place_box  # noqa: E402
from test_repair import standing_scene  # noqa: E402

REGEN = replace(RepairConfig(), regen_mode=True)


def _ctx(pipe, z, box, prev_dist_l, motion: float = 0.02) -> FrameContext:
    pipe.ev.forward(z, box)
    return FrameContext(
        t=1, box_pose=box.copy(), z_pred=z.copy(), z_prev=z.copy(),
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={}, prev_hand_local={}, prev_sole_world={},
        box_pose_prev=box.copy(), ref_hold={"left_hand": True},
        prev_hand_dist={"l": prev_dist_l}, frame_motion_m=motion)


def _hand_dist(pipe, z, box) -> float:
    pipe.ev.forward(z, box)
    return min(pipe.ev.geom_distance(g, pipe.ev.box_geom, distmax=10.0)[0]
               for g in pipe.ev._hand_geoms["l"])


# --- R2: registry selection --------------------------------------------------------


def test_registry_selection_covers_hovering_hands() -> None:
    """A ref-hold side whose hand is far outside the activation gate still
    gets an active row (source=registry); a candidate that stalls (dist ==
    prev) honestly violates the dynamic upper."""
    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    box = np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0])
    d = _hand_dist(pipe, z, box)
    ctx = _ctx(pipe, z, box, prev_dist_l=d)          # stalled at ~4.4 m
    val, lo, hi, scl, names, ids = pipe.solver.check_rows(z, ctx)
    cor = pipe.solver.last_corridor["l"]
    assert cor["source"] == "registry" and cor["geom"] == "left_rubber_hand_link"
    assert "left_rubber_hand_link~largebox:regen" in names
    # the dynamic upper from a stalled prev: prev - 0.15*dt - margin
    margin = min(0.006, max(0.0005, 0.3 * ctx.frame_motion_m ** 2))
    assert cor["upper_m"] == pytest.approx(max(0.008, d - 0.15 / 30.0 - margin))
    k = names.index("left_rubber_hand_link~largebox:regen")
    assert val[99 + k] == pytest.approx(d)
    assert val[99 + k] > hi[99 + k] + 1e-9           # stalled -> violating
    assert pipe.solver.violation(val, lo, hi, scl) > 1e-6
    assert ids[k] is not None                        # a real pair identity


def test_query_failure_is_explicit_never_silent(monkeypatch) -> None:
    """If no real hit exists at any query range, the side reports
    ``query_failed`` and a synthetic always-violating row rejects the
    candidate -- the truncated sentinel is never used as a distance, and the
    row carries no pair identity to carry back."""
    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    box = np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0])
    ctx = _ctx(pipe, z, box, prev_dist_l=0.05)
    monkeypatch.setattr(pipe.solver, "_registry_nearest_hand", lambda g: None)
    val, lo, hi, scl, names, ids = pipe.solver.check_rows(z, ctx)
    k = names.index("regen_query_failed[l]:regen")
    assert val[99 + k] == 10.0 and hi[99 + k] == 0.0
    assert ids[k] is None
    assert pipe.solver.last_corridor["l"]["query_failed"] is True
    assert pipe.solver.violation(val, lo, hi, scl) > 1e-6
    chk = pipe.frame_valid(z, ctx)
    assert not chk.ok
    # the synthetic row must not crash the carry-back path
    pipe.carry_discovered(chk)


def test_near_hand_dedupes_to_one_row() -> None:
    """When the pair IS within the activation gate the selection stays on
    the candidate path and the pair occupies exactly one row (no duplicate
    from the registry fallback)."""
    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", 0.006)            # 6mm: inside the gate
    ctx = _ctx(pipe, z, box, prev_dist_l=0.02)
    _val, _lo, _hi, _scl, names, _ids = pipe.solver.check_rows(z, ctx)
    assert pipe.solver.last_corridor["l"]["source"] == "candidate"
    regen_names = [n for n in names if n.endswith(":regen")]
    assert len(regen_names) == 1


def test_standard_mode_has_no_active_rows() -> None:
    """Standard mode ignores ref_hold entirely: no corridor, no regen rows
    (unchanged historical behaviour)."""
    std = RepairPipeline(cfg=RepairConfig())
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(std, z, "l", 0.05)
    ctx = _ctx(std, z, box, prev_dist_l=0.05)
    _v, _l, _h, _s, names, _i = std.solver.check_rows(z, ctx)
    assert not any(n.endswith(":regen") for n in names)
    assert std.solver.last_corridor == {}


# --- R1: the corridor contract at every exit ---------------------------------------


def test_zero_qp_early_return_blocked_when_corridor_violated() -> None:
    """The D-R counterexample shape: a ref-hold side stalled at hover
    distance. The old code returned accepted=True with ZERO QP calls and an
    unchanged state; with the active row present the early return requires
    corridor feasibility, so a stall either solves (QP > 0, result satisfies
    the dynamic upper) or fails -- never a free pass."""
    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", 0.10)             # hand hovering 10 cm
    d = _hand_dist(pipe, z, box)
    ctx = _ctx(pipe, z, box, prev_dist_l=d)          # stalled: no closing
    s2 = FrameSolver(pipe.ev, REGEN)
    s2.probe_cfg = pipe.validator.cfg
    res = s2.solve_frame(z, ctx)
    if res.accepted:
        assert res.qp_calls >= 1, "corridor violation must not pass with 0 QPs"
        assert _hand_dist(pipe, res.z, box) <= ctx.prev_hand_dist["l"] - 0.15 / 30.0 + 1e-9
    else:
        assert res.status in ("infeasible", "max_iters", "task_quality",
                              "review_failed")
    # the same candidate under a LEGAL closing (prev 1 cm further) may take
    # the early return: the prediction already satisfies the corridor
    ctx_ok = _ctx(pipe, z, box, prev_dist_l=d + 0.010)
    s3 = FrameSolver(pipe.ev, REGEN)
    s3.probe_cfg = pipe.validator.cfg
    res_ok = s3.solve_frame(z, ctx_ok)
    assert res_ok.accepted and res_ok.qp_calls == 0   # free pass is CORRECT here


def test_adoption_check_enforces_corridor_all_paths() -> None:
    """The pipeline's own acceptance check (used by skip, by every attempt
    and by adoption) sees the active row: a stalled hover fails with the
    regen row in the evidence; a legal 30->24 mm approach passes; a 6 mm
    hand passes when the rest of the frame does."""
    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()

    box10 = _place_box(pipe, z, "l", 0.10)
    d10 = _hand_dist(pipe, z, box10)
    chk = pipe.frame_valid(z, _ctx(pipe, z, box10, prev_dist_l=d10))
    assert not chk.ok and chk.reason == "state_constraints"
    assert any(":regen" in str(r.get("pair")) for r in chk.rows)

    # legal approach: previous accepted distance 30 mm, candidate 24 mm
    box24 = _place_box(pipe, z, "l", 0.024)
    chk_ap = pipe.frame_valid(z, _ctx(pipe, z, box24, prev_dist_l=0.030))
    assert chk_ap.ok, chk_ap.rows

    # inside the gap: 6 mm with everything else legal
    box6 = _place_box(pipe, z, "l", 0.006)
    chk6 = pipe.frame_valid(z, _ctx(pipe, z, box6, prev_dist_l=0.012))
    assert chk6.ok, chk6.rows
