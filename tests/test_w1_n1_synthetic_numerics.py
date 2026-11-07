"""Synthetic tests for the N1 minimal synthetic-verification module.

No real dataset, cache, checkpoint, G1 asset or MuJoCo model is opened; the
module under test is itself pure-CPU synthetic arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_n1_synthetic_numerics as n1  # noqa: E402
from ireset.eval.w1_n0_coordinate_flags import N0Budget  # noqa: E402


@pytest.fixture()
def summary_and_cases():
    budget = N0Budget(wall_cap_s=900.0, reserve_s=60.0)
    return n1.run_synthetic_verification(budget)


def test_all_classes_pass_within_caps(summary_and_cases):
    summary, cases = summary_and_cases
    assert summary["all_cases_passed"] is True
    assert summary["go_no_go"] == "go_for_conditional_fit_diagnostics_review"
    assert len(cases) <= n1.MAX_TOTAL_CASES
    for cls, v in summary["per_class"].items():
        assert 0 < v["cases"] <= n1.MAX_CASES_PER_CLASS
        assert v["passed"] == v["cases"]
    assert 0 < summary["counters"]["fp32_formula_evals"] <= 384
    assert summary["counters"]["float64_reference_evals"] > 0
    assert summary["counters"]["state_advances"] > 0
    assert summary["counters"]["interval_propagations"] > 0


def test_case_vectors_within_bound(summary_and_cases):
    _summary, cases = summary_and_cases
    for c in cases:
        assert len(c["id"]) > 0 and c["class"] in n1.CLASSES


def test_determinism():
    s1, c1 = n1.run_synthetic_verification(N0Budget(wall_cap_s=900.0, reserve_s=60.0))
    s2, c2 = n1.run_synthetic_verification(N0Budget(wall_cap_s=900.0, reserve_s=60.0))
    assert c1 == c2
    a = {k: v for k, v in s1.items() if k not in ("budget",)}
    b = {k: v for k, v in s2.items() if k not in ("budget",)}
    assert a == b


def test_c5_zero_disagreement_demonstrated(summary_and_cases):
    _summary, cases = summary_and_cases
    c5 = [c for c in cases if c["class"] == "cancellation_within_intervals"]
    failed = [c for c in c5 if c["observed"]["cross_path_gate_failed"]]
    assert len(failed) >= 2, "review requires >=2 gate-failure demonstrations"
    assert all(c["observed"]["inside_intervals"] for c in c5)
    # the audited-vs-independent C1 contrast must have no fallback exclusions
    # hiding node mismatches in this case set
    c1 = [c for c in cases if c["class"] == "same_input_formula_bitwise"]
    assert all(c["observed"]["bitwise_audited_vs_independent"] for c in c1)


def test_c6_all_deviations_caught(summary_and_cases):
    _summary, cases = summary_and_cases
    c6 = [c for c in cases if c["class"] == "interval_violation_deviation"]
    assert len(c6) == 8
    assert all(c["observed"]["bitwise_differs_from_frozen"] for c in c6)
    assert all(c["observed"]["theta_outside_interval"]
               or c["observed"]["delta_outside_interval"] for c in c6)


def test_closure_fields(summary_and_cases):
    summary, _cases = summary_and_cases
    closure = summary["closure"]
    assert closure["same_input_formula_bitwise_verified"] is True
    assert closure["write_erasure_is_precision_not_formula_error"] is True
    assert closure["one_sided_observable_is_sensitivity"] is True
    assert closure["state_zero_sign_is_direct_failure"] is True
    assert closure["cross_path_sensitivity_inside_intervals_reproduced"] is True
    assert closure["implementation_deviations_are_detectable"] is True
    assert closure["strict_cross_path_gates_still_failing_on_real_evidence"] is True
    assert closure["long_horizon_training_and_action_quality_uncovered"] is True
    assert closure["cpu_not_claimed_equivalent_to_production_cuda_foreach"] is True


def test_budget_stop_partial(tmp_path):
    class Countdown(N0Budget):
        def __init__(self, remaining_checks, **kw):
            super().__init__(**kw)
            self.left = remaining_checks

        def check(self, point):
            self.left -= 1
            if self.left <= 0:
                self.exceeded = True
                self.stopped_at = point
                raise __import__("ireset.eval.w1_n0_coordinate_flags",
                                 fromlist=["BudgetExceeded"]).BudgetExceeded(point)

    budget = Countdown(remaining_checks=4, wall_cap_s=900.0, reserve_s=60.0)
    summary, cases = n1.run_synthetic_verification(budget)
    assert summary["budget_exceeded"] is True
    assert summary["all_cases_passed"] is False
    assert summary["go_no_go"] is None
    assert 0 < len(cases) < n1.MAX_TOTAL_CASES
    paths = n1.write_outputs(tmp_path, summary, cases)
    assert Path(paths["summary"]).exists()


def test_cli(tmp_path):
    import subprocess
    script = Path(__file__).resolve().parents[1] / "scripts/verify_w1_n1_synthetic_numerics.py"
    r = subprocess.run([sys.executable, str(script),
                        "--output-root", str(tmp_path / "out")],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["all_cases_passed"] is True
    summary = json.loads(Path(payload["outputs"]["summary"]).read_text())
    assert summary["label_verdict"] == "complete"


def test_seal_withdraws_go_no_go(tmp_path):
    """P2-3: a seal withdrawal must revoke the go/no-go recommendation
    together with the completion claims (correct elapsed-vs-cap semantics:
    t0 is large, elapsed 901s exceeds the 900s cap)."""
    summary, cases = n1.run_synthetic_verification(
        N0Budget(wall_cap_s=900.0, reserve_s=60.0))
    budget = N0Budget(wall_cap_s=900.0, reserve_s=60.0, t0=5000.0)
    ticks = iter([5000.0 + t for t in (10.0, 901.0, 902.0, 905.0, 910.0)])
    paths = n1.write_outputs(tmp_path, summary, cases, budget=budget,
                             clock=lambda: next(ticks))
    disk = paths["summary_doc"]
    assert paths["seal_facts"]["status"] == "budget_exceeded_after_seal"
    assert disk["seal"]["status"] == "budget_exceeded_after_seal"
    assert disk["go_no_go"] == "no_go_budget_seal_exceeded"
    assert disk["all_cases_passed"] is False
    assert disk["label_verdict"] == "budget_exceeded_after_seal"


def test_seal_two_phase_normal(tmp_path):
    """Happy path: publish (sealing) -> verify -> seal; go/no-go preserved."""
    summary, cases = n1.run_synthetic_verification(
        N0Budget(wall_cap_s=900.0, reserve_s=60.0))
    budget = N0Budget(wall_cap_s=900.0, reserve_s=60.0, t0=0.0)
    ticks = iter([10.0, 20.0, 21.0, 30.0, 31.0, 40.0])
    paths = n1.write_outputs(tmp_path, summary, cases, budget=budget,
                             clock=lambda: next(ticks))
    disk = paths["summary_doc"]
    assert disk["seal"]["status"] == "sealed_within_budget"
    assert disk["seal"]["terminal"] is True
    # registered cost covers the actual write and the return tail
    assert disk["seal"]["measured_wall_s_at_summary"] == 54.0
    assert disk["seal"]["measured_wall_s_at_summary"] >= 40.0
    assert disk["go_no_go"] == "go_for_conditional_fit_diagnostics_review"
    assert disk["all_cases_passed"] is True
