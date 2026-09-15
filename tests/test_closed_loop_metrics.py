"""P0-fix F4 bookkeeping rules: pairing, denominators, F1 undefinedness,
condition handling and metadata reuse (p0_fix_plan section 5)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from closed_loop_metrics import (hand_f1_report, metadata_block,
                                 metadata_compatible, paired_task_mpjpe,
                                 summarize_method)


def row(demo, accepted, mpjpe=None, f1_l=None, f1_r=None):
    return {"demo": demo, "accepted": accepted,
            "task_mpjpe_box_m": mpjpe, "hand_f1_left": f1_l, "hand_f1_right": f1_r}


def test_paired_comparison_uses_common_accepted_set() -> None:
    """Method accepts {A,B}, teacher accepts {A,C}: paired n must be 1 on
    A; total success counts (2 vs 2) are never the denominator."""
    method = [row("A", True, 0.10), row("B", True, 0.12)]
    base = [row("A", True, 0.08), row("C", True, 0.06)]
    p = paired_task_mpjpe(method, base)
    assert p["n"] == 1 and p["demos"] == ["A"]
    assert abs(p["diff_mean_m"] - 0.02) < 1e-12
    # no intersection -> undefined, never 0
    p0 = paired_task_mpjpe([row("B", True, 0.1)], base)
    assert p0["n"] == 0 and p0["diff_mean_m"] is None


def test_rejected_samples_stay_in_denominator_without_position_error() -> None:
    """Acceptance counts rejected rows; rejected rows never carry an MPJPE."""
    rows = [row("A", True, 0.1), row("B", False, None)]
    s = summarize_method(rows, [row("A", True, 0.05), row("B", True, 0.05)])
    assert s["n"] == 2 and s["n_accepted"] == 1 and s["acceptance"] == 0.5
    assert s["paired_task_mpjpe"]["n"] == 1
    # a common accepted demo missing the metric is evidence loss, not pass
    s2 = summarize_method([row("A", True, None)], [row("A", True, 0.05)])
    assert s2["conditions"]["paired_mpjpe_le_baseline_plus_2cm"] is None


def test_hand_f1_dual_empty_undefined_and_success_weighted_zero() -> None:
    """Dual-empty (None) F1 never fills with 1; rejected and undefined
    samples score 0 in the success-weighted mean over the full set."""
    rows = [row("A", True, 0.1, f1_l=0.8, f1_r=0.6),
            row("B", True, 0.1, f1_l=None, f1_r=0.4),   # left dual-empty
            row("C", False)]
    rep = hand_f1_report(rows)
    left = rep["hand_f1_left"]
    assert left["raw_mean"] == 0.8 and left["n_defined"] == 1 and left["n_undefined"] == 1
    assert abs(left["success_weighted"] - 0.8 / 3) < 1e-12   # B,C score 0
    right = rep["hand_f1_right"]
    assert abs(right["raw_mean"] - 0.5) < 1e-12
    assert abs(rep["hand_macro_success_weighted"] - (0.8 / 3 + 0.5 * 2 / 3) / 2) < 1e-12


def test_conditions_require_complete_evidence() -> None:
    """Every condition needs defined evidence; any missing input leaves it
    unpassed (None), and all_conditions_met is never True then."""
    both = [row("A", True, 0.09, f1_l=0.9, f1_r=0.9)]
    base = [row("A", True, 0.08, f1_l=0.9, f1_r=0.9)]
    s = summarize_method(both, base)
    assert s["conditions"]["acceptance_ge_baseline"] is True
    assert s["conditions"]["paired_mpjpe_le_baseline_plus_2cm"] is True
    assert s["conditions"]["success_weighted_f1_ge_0.95x_baseline"] is True
    assert s["all_conditions_met"] is True
    # method acceptance drops below baseline -> condition False, not met
    s2 = summarize_method([row("A", True, 0.09, 0.9, 0.9), row("B", False)],
                          [row("A", True, 0.08, 0.9, 0.9), row("B", True, 0.1, 0.9, 0.9)])
    assert s2["conditions"]["acceptance_ge_baseline"] is False
    assert s2["all_conditions_met"] is False
    # undefined baseline F1 -> F1 condition undefined, overall None
    s3 = summarize_method(both, [row("A", True, 0.08, None, None)])
    assert s3["conditions"]["success_weighted_f1_ge_0.95x_baseline"] is None
    assert s3["all_conditions_met"] is None


def test_metadata_reuse_requires_exact_match(monkeypatch) -> None:
    monkeypatch.setattr("closed_loop_metrics.git_rev", lambda: "abc1234")
    meta = metadata_block("relaxed", 0.40, ["a.npz", "b.npz"], 7, models={})
    ok, why = metadata_compatible(meta, "relaxed", 0.40, ["b.npz", "a.npz"], 7)
    assert ok, why
    for bad in (dict(mode="strict"), dict(task_mpjpe_tol_m=0.101),
                dict(files=["a.npz"]), dict(seed=8), dict(models={"m": "x"})):
        ok, why = metadata_compatible({**meta, **bad}, "relaxed", 0.40,
                                      ["a.npz", "b.npz"], 7)
        assert not ok, bad
        assert "mismatch" in why
