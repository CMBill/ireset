"""Per-frame label EVIDENCE contracts (plan stage A): certification notions
that must never substitute one another, the strict proximity operator,
report values recomputable from masks, and the structured velocity
rejection evidence (unrounded value / limit / exceedance / diff frames /
joint index) that replaces rounded location strings.

Companion tests for the label-quality plan
(m2_m3_label_quality_train24_plan.md sections 2-3, acceptance section 5).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.data.labels import (  # noqa: E402
    both_hands_metrics,
    continuous_runs,
    crossings_within_intervals,
    grasp_pair_valid,
    intersection_interval_ids,
    longest_detachment_run,
    side_label_arrays,
)
from ireset.embodied.contact_sites import ContactInterval, ContactSchedule  # noqa: E402
from ireset.repair.context import ReferenceCondition  # noqa: E402
from ireset.repair.validation import IndependentValidator, ValidationConfig  # noqa: E402
from segment_quality_report import side_metrics  # noqa: E402


@pytest.fixture(scope="module")
def validator() -> IndependentValidator:
    return IndependentValidator(cfg=ValidationConfig())

# --- metric counterexamples: certified total vs intersection -----------------------


def test_certified_total_is_not_the_intersection():
    """The counterexample behind the sub10-left misreport: a 3-frame
    certified interval OUTSIDE the reference window gives certified_frames=3
    and certified_in_ref_frames=0 -- both must be reported, neither may
    substitute the other, and the longest continuous intersection is an
    explicit 0."""
    T = 40
    ref = np.zeros(T, dtype=bool)
    ref[20:30] = True
    arrays = side_label_arrays(np.full(T, 0.005), ref, [(5, 8), (22, 26)], 0.01)
    assert arrays["certified"].sum() == 3 + 4
    assert arrays["supervision"].sum() == 4                       # only [22,26)
    assert continuous_runs(arrays["supervision"]) == [(22, 26)]
    assert max(e - s for s, e in continuous_runs(arrays["supervision"])) == 4
    # zero-intersection side is an explicit 0, not None
    arrays0 = side_label_arrays(np.full(T, 0.005), ref, [(5, 8)], 0.01)
    assert arrays0["supervision"].sum() == 0
    assert (arrays0["interval_id"] == -1).all()


def test_proximity_operator_is_strict_less_than():
    """distance == hand_enter_m is NOT proximity (keep criterion operator),
    below it is, above it is not."""
    T = 3
    ref = np.ones(T, dtype=bool)
    dist = np.array([0.009999, 0.01, 0.010001])
    arrays = side_label_arrays(dist, ref, [], 0.01)
    assert arrays["proximity"].tolist() == [True, False, False]


def test_undefined_vs_zero_semantics():
    """Zero reference frames -> ratios undefined; zero certification -> 0."""
    T = 10
    # no reference interval: proximity/supervision masks empty, ids all -1
    arrays = side_label_arrays(np.zeros(T), np.zeros(T, dtype=bool), [(2, 5)], 0.01)
    assert not arrays["proximity"].any() and not arrays["supervision"].any()
    assert (arrays["interval_id"] == -1).all()


def test_intersection_ids_break_at_gaps_and_boundaries():
    T = 20
    ref = np.zeros(T, dtype=bool)
    ref[4:8] = True       # gap at 8
    ref[9:12] = True
    ref[13:17] = True
    ids = intersection_interval_ids([(3, 14), (14, 18)], ref, T)
    # pieces: [4,8) from iv1; [9,12) from iv1; [13,14) from iv1 + [14,17)
    # from iv2 are CONTIGUOUS but cross a certified boundary -> distinct ids
    assert ids[7] != ids[9]                       # reference gap breaks the id
    assert ids[11] != ids[13]                     # ref gap (12 not in ref)
    assert ids[13] != ids[14]                     # certified-interval boundary
    ok = grasp_pair_valid(ids)
    assert not ok[7] and not ok[8]                # 7->8 crosses a ref gap
    assert ok[4] and ok[5] and ok[6]              # inside one piece
    assert ok[14] and ok[15]                      # inside the iv2 piece
    assert not ok[0]                              # frame 0 has no predecessor


def test_crossings_counted_only_inside_the_same_interval():
    within = np.array([True, True, False, False, True, True])
    # adjacent intervals [0,4) and [4,6): the 0->1 transition at the shared
    # boundary is NOT a crossing; the 1->0 inside [0,4) is
    assert crossings_within_intervals(within, [(0, 4), (4, 6)]) == 1
    # the same series with a GAP between the intervals: still 1
    assert crossings_within_intervals(within, [(0, 3), (4, 6)]) == 1
    # detachment runs also stay inside ONE interval: frames 2,3 are
    # consecutive not-within frames but belong to different intervals here
    assert longest_detachment_run(within, [(0, 3), (4, 6)]) == 1
    assert longest_detachment_run(within, [(0, 4), (4, 6)]) == 2


def test_both_hands_listed_separately():
    T = 12
    per_side = {
        "l": {"certified": (np.arange(T) < 6), "supervision": (np.arange(T) >= 2) & (np.arange(T) < 6)},
        "r": {"certified": (np.arange(T) >= 3) & (np.arange(T) < 9),
              "supervision": (np.arange(T) >= 3) & (np.arange(T) < 8)},
    }
    m = both_hands_metrics(per_side)
    assert m["both_certified_frames"] == 3        # [3,6)
    assert m["both_supervision_frames"] == 3      # [3,6)
    assert m["both_longest_supervision_run_frames"] == 3


def test_report_side_metrics_recomputable_from_masks(validator):
    """side_metrics on an injected per-frame series: every coverage number
    must be recomputable from the supervision/proximity masks, the strict-<
    count excludes distance == enter, speed statistics exclude the
    undefined frame-0 differential, and detachment runs stay inside one
    reference interval."""
    T = 40
    enter = validator.cfg.hand_enter_m
    dist = np.full(T, 0.5)
    dist[10:20] = 0.005
    dist[14] = enter                     # exactly at the threshold: NOT within
    dist[15:18] = 0.05                   # 3-frame detachment inside the ref interval
    dist[25:35] = 0.008
    local = np.zeros((T, 3, 3))
    local[:, 0, 0] = np.arange(T) * 0.001
    validator._cache = {"hand_local": {"l": local, "r": local},
                        "hand_dist": {"l": dist, "r": dist}}
    schedule = ContactSchedule(intervals=[
        ContactInterval("left_hand", 5, 8, "object"),      # certified OUTSIDE ref
        ContactInterval("left_hand", 12, 16, "object"),    # certified inside ref
    ])
    cond = ReferenceCondition(np.zeros((T, 52, 3)),
                              np.tile([0, 0, 0.2, 1, 0, 0, 0], (T, 1)),
                              ContactSchedule(intervals=[
                                  ContactInterval("left_hand", 10, 20, "object"),
                                  ContactInterval("left_hand", 25, 35, "object")]))
    rep = types.SimpleNamespace(schedule=schedule)
    m = side_metrics(validator.cfg, np.zeros((T, 43)), cond, validator, "l", rep)

    # recompute from the same series through the label functions
    ref = np.zeros(T, dtype=bool)
    ref[10:20] = True
    ref[25:35] = True
    arrays = side_label_arrays(dist, ref, [(5, 8), (12, 16)], enter)
    assert m["certified_frames"] == int(arrays["certified"].sum()) == 7
    assert m["certified_in_ref_frames"] == int(arrays["supervision"].sum()) == 4
    assert m["intersection_intervals"] == [[12, 16]]
    assert m["intersection_longest_run_frames"] == 4
    assert m["frames_within_enter"] == int(arrays["proximity"].sum())
    # dist[14] == enter is excluded by the strict operator
    assert m["frames_within_enter"] == 6 + 10  # [10,20): 6 within (enter+3 above); [25,35): 10
    assert m["missing_contact_ref_frames"] == 20 - m["frames_within_enter"]
    assert m["longest_detachment_run_frames"] == 4         # frames 14..17 (== enter is detachment too)
    assert m["proximity_recall"] == pytest.approx(
        arrays["proximity"].sum() / ref.sum())
    # frame 0 is outside the ref intervals here; speed stats cover exactly
    # the ref frames with a defined differential
    assert m["speed_ref_mps"]["n"] == 10 + 10
    # the diagnostic bands strictly enlarge the count, never shrink it
    assert m["frames_within_enter"] <= m["frames_within_enter_p2mm"] \
        <= m["frames_within_enter_p5mm"]
    # a side with no reference frames: ratio undefined
    m2 = side_metrics(validator.cfg, np.zeros((T, 43)), ReferenceCondition(
        np.zeros((T, 52, 3)), np.tile([0, 0, 0.2, 1, 0, 0, 0], (T, 1)),
        ContactSchedule()), validator, "r", types.SimpleNamespace(schedule=schedule))
    assert m2["ref_frames"] == 0 and m2["proximity_recall"] is None


def test_speed_stats_exclude_frame_zero(validator):
    """When a reference interval starts at frame 0, the undefined frame-0
    differential must not enter the statistics as a placeholder zero."""
    T = 6
    local = np.zeros((T, 3, 3))
    local[1:, 0, 0] = 0.01 * np.arange(1, T)      # 0.01/frame -> 0.3 m/s
    dist = np.full(T, 0.005)
    validator._cache = {"hand_local": {"l": local, "r": local},
                        "hand_dist": {"l": dist, "r": dist}}
    cond = ReferenceCondition(np.zeros((T, 52, 3)),
                              np.tile([0, 0, 0.2, 1, 0, 0, 0], (T, 1)),
                              ContactSchedule(intervals=[
                                  ContactInterval("left_hand", 0, 6, "object")]))
    rep = types.SimpleNamespace(schedule=ContactSchedule())
    m = side_metrics(validator.cfg, np.zeros((T, 43)), cond, validator, "l", rep)
    assert m["speed_ref_mps"]["n"] == 5
    assert m["speed_ref_mps"]["p50"] == pytest.approx(0.3, abs=1e-6)
    assert m["speed_all_mps"]["n"] == 5


# --- numerical evidence: micro velocity exceedance ----------------------------------


def test_velocity_micro_exceedance_evidence(validator):
    """sub7's failure class: joint 26 at frame 101 exceeds 12 rad/s by
    ~4e-7 -- still REJECTED, and the evidence is unrounded with the diff
    frames and joint index (the location string rounds to 12.00)."""
    T, idx, frame = 105, 26, 101
    q = np.zeros((T, 43))
    q[:, 3] = 1.0
    q[:, 39] = 1.0
    dt = 1.0 / validator.cfg.fps
    target = 12.000000401362618
    q[frame, 7 + idx] = q[frame - 1, 7 + idx] + target * dt
    res = validator._check_velocities(q)
    assert not res.passed
    assert res.first_failure_frame == frame
    exc = res.evidence["velocity_exceedances"]
    assert len(exc) == 1
    rec = exc[0]
    assert rec["family"] == "joint"
    assert rec["joint_index"] == idx
    assert rec["diff_frames"] == [frame - 1, frame]
    assert rec["frame"] == frame
    assert rec["value"] == pytest.approx(target, rel=1e-12)
    assert rec["limit"] == pytest.approx(12.0)
    assert rec["exceedance"] == pytest.approx(4.0136e-7, rel=1e-3)
    # the display string cannot distinguish this from a pass -- evidence, not
    # location, carries the verdict basis
    assert "12.00" in res.location


def test_velocity_pass_has_no_evidence(validator):
    T = 10
    q = np.zeros((T, 43))
    q[:, 3] = 1.0
    q[:, 39] = 1.0
    q[5, 7 + 3] = 0.39 / validator.cfg.fps * validator.cfg.fps  # well under 12 rad/s
    res = validator._check_velocities(q)
    assert bool(res.passed)
    assert res.evidence is None
    assert res.first_failure_frame is None


def test_velocity_verdict_semantics_unchanged(validator):
    """The evidence addition must not alter worst/limit/first-failure-frame:
    a base-linear exceedance reports the same numbers as before."""
    T = 6
    q = np.zeros((T, 43))
    q[:, 3] = 1.0
    q[:, 39] = 1.0
    q[3, 0:3] = [2.0 / validator.cfg.fps, 0, 0]   # 2.0 m/s > 1.8
    res = validator._check_velocities(q)
    assert not res.passed
    assert res.first_failure_frame == 3
    exc = res.evidence["velocity_exceedances"]
    assert exc[0]["family"] == "base_lin"
    assert exc[0]["joint_index"] is None
    assert exc[0]["value"] == pytest.approx(2.0, rel=1e-9)


def test_rejection_evidence_carries_velocity_evidence():
    """Both report rounds keep the structured evidence (R5 path)."""
    from regenerate_teachers import rejection_evidence

    ev = {"velocity_exceedances": [{"family": "joint", "frame": 101,
                                    "diff_frames": [100, 101], "value": 12.000000401362618,
                                    "limit": 12.0, "exceedance": 4.0136e-7,
                                    "joint_index": 26}]}
    for checks in ({"velocities": {"location": "joint=12.00rad/s",
                                   "first_failure_frame": 101, "evidence": ev}},
                   {"velocities": {"location": "joint=12.00rad/s",
                                   "first_failure_frame": 101, "evidence": ev}}):
        diag = {"reason": "final_validation",
                "final_retry_failed_checks": checks,
                "failed_checks": checks}
        rej = rejection_evidence(diag)
        assert rej["validator_failed_checks"]["velocities"]["evidence"] == ev


def test_cli_task_gate_guard(monkeypatch):
    """--assert-task-gate refuses to start under a different effective gate."""
    import regenerate_teachers

    argv = ["regenerate_teachers.py", "--demos", "sub10_largebox_000",
            "--assert-task-gate", "0.113"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="effective task gate"):
        regenerate_teachers.main()


