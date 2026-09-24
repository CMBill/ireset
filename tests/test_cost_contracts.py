"""Run-level cost contracts (R6): every trajectory pass is totalled.

The report used to export the FINAL pass's counters under unversioned names
and the first pass nested under ``first_attempt``: a suffix retry made the
totals silently wrong (the review's sub12 case reported 65 attempts / 1815
QP where the two rounds actually ran 130 / 3630). These tests pin the
replacement: one explicit totals view per run (schema 2), recomputable from
per-pass events, with visits vs unique frames vs input length kept apart and
per-visit percentiles taken from the merged unrounded series.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import (  # noqa: E402
    PassRecord,
    RepairPipeline,
    _timing_stats,
    cost_view,
)
from ireset.repair.scp import FrameCheck  # noqa: E402
from ireset.repair.validation import (  # noqa: E402
    CheckResult,
    ValidationReport,
)
from test_repair import self_condition, standing_scene  # noqa: E402


def _synthetic_pass(index: int, visits: list[int], timings: list[float],
                    attempts: int = 0, qp: int = 0, iterations: int = 0,
                    slack: int = 0, complete: bool = True) -> PassRecord:
    return PassRecord(
        trajectory_pass=index,
        status="accepted" if complete else "rejected_frame",
        complete=complete, accepted_frames=max(visits) + 1 if visits else 0,
        skipped_frames=0, slack_frames=0, input_frames=max(visits) + 1 if visits else 0,
        attempts=[{"qp_calls": qp // max(attempts, 1), "iterations": iterations,
                   "slack_used": i < slack} for i in range(attempts)],
        checks=[], visits=list(visits), timings_s=list(timings),
        backtracks_adopted=0, backtrack_probes=0,
        t_fail=None if complete else max(visits) if visits else 0)


def test_cost_view_merges_passes_without_double_counting() -> None:
    """R6: visits are summed, unique frames are the UNION (a replayed prefix
    must not be counted twice), the input length never doubles, and the
    per-visit percentiles come from the MERGED series -- not from averaging
    per-pass percentiles (those are not additive)."""
    p0 = _synthetic_pass(0, [0, 1, 2, 3], [0.30, 0.10, 0.20, 0.40],
                         attempts=2, qp=8, iterations=6, slack=1)
    p1 = _synthetic_pass(1, [2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04],
                         attempts=3, qp=12, iterations=9, slack=2)
    view = cost_view([p0, p1], input_frames=6, accepted_pass=1)
    assert view["schema"] == 2
    assert view["input_frames"] == 6
    assert view["processed_frame_visits"] == 8
    assert view["unique_processed_frames"] == 6          # union, not 4 + 4
    assert view["attempts"] == 5
    assert view["qp_calls_total"] == 20
    assert view["iterations_total"] == 39          # per-attempt counters sum
    assert view["slack_attempts"] == 3
    merged = p0.timings_s + p1.timings_s
    assert view["per_visit_ms"] == _timing_stats(merged)
    per_pass_mean_of_p95 = np.mean([_timing_stats(p.timings_s)["p95_ms"]
                                    for p in (p0, p1)])
    assert view["per_visit_ms"]["p95_ms"] != pytest.approx(per_pass_mean_of_p95)
    assert view["solve_s"] == pytest.approx(sum(merged), abs=1e-3)
    assert view["final_output"]["trajectory_pass"] == 1
    assert view["final_output"]["accepted"] is True
    assert [p["trajectory_pass"] for p in view["passes"]] == [0, 1]
    # recomputable from the per-pass events
    assert sum(p["attempts"] for p in view["passes"]) == view["attempts"]
    assert sum(p["qp_calls_total"] for p in view["passes"]) == view["qp_calls_total"]


def _pipe_with_solver(monkeypatch, results):
    """Pipeline whose solver returns prescribed (accepted, status) results."""
    import ireset.repair.scp as scp

    calls = {"n": 0}

    def fake_solve(self, z_init, ctx, force_slack=False):
        ok, status = results[min(calls["n"], len(results) - 1)]
        calls["n"] += 1
        res = scp.FrameResult(bool(ok), np.asarray(z_init, dtype=float), 3,
                              status=status, qp_calls=5)
        res.frame_check = FrameCheck(ok=bool(ok),
                                     reason="ok" if ok else "state_constraints")
        return res

    monkeypatch.setattr(scp.FrameSolver, "solve_frame", fake_solve)
    return calls


def test_cost_totals_cover_both_passes_of_a_suffix_retry(monkeypatch) -> None:
    """R6: a suffix retry must be totalled, not hidden. The old report showed
    only the retry pass (sub12: 65 attempts / 1815 QP instead of 130 / 3630)."""
    pipe = RepairPipeline(cfg=RepairConfig())
    scene = standing_scene(6)
    scene[1, 36:43] = np.array([0.0, 0.0, 0.80, 1.0, 0, 0, 0])
    cond = self_condition(pipe, scene)
    # pass 0 accepts (frame 1 repaired on the first attempt), the validator
    # fails once -> suffix retry replays the whole clip
    calls = _pipe_with_solver(monkeypatch, [(True, "solved")])
    failing = ValidationReport(passed=False, checks={
        "penetration": CheckResult(False, 0.003, 0.002, "t=1",
                                   first_failure_frame=1)})
    n_val = {"n": 0}

    def fake_validate(traj, exclude=()):
        n_val["n"] += 1
        return failing if n_val["n"] == 1 else ValidationReport(passed=True)

    monkeypatch.setattr(pipe.validator, "validate", fake_validate)
    res = pipe.repair_trajectory(scene, cond)
    assert res.accepted, res.diagnostics
    cost = res.diagnostics["cost"]
    # both passes repaired frame 1 -> the total is 2 attempts, 10 QP calls
    assert calls["n"] == 2
    assert cost["attempts"] == 2
    assert cost["qp_calls_total"] == 10
    assert cost["iterations_total"] == 6
    assert cost["processed_frame_visits"] == 12          # 6 + 6
    assert cost["unique_processed_frames"] == 6
    assert cost["input_frames"] == 6
    assert sum(p["attempts"] for p in cost["passes"]) == 2
    assert sum(p["qp_calls_total"] for p in cost["passes"]) == 10
    assert sum(p["processed_frame_visits"] for p in cost["passes"]) == 12
    assert {p["trajectory_pass"] for p in cost["passes"]} == {0, 1}
    assert cost["final_output"]["trajectory_pass"] == 1
    # the nested first_attempt block describes pass 0 ONLY
    assert res.diagnostics["first_attempt"]["attempts"] == 1
    assert res.diagnostics["first_attempt"]["qp_calls_total"] == 5
    assert cost["attempts"] == (res.diagnostics["attempts"]
                                + res.diagnostics["first_attempt"]["attempts"])
    assert n_val["n"] == 2, "one validator call per pass"


def test_cost_view_exists_on_every_exit_path(monkeypatch) -> None:
    """R6: early rejection, first-pass acceptance, retry solve failure and
    retry validation failure all carry a complete totals view; a pre-flight
    rejection reports zeros instead of omitting the field."""
    pipe = RepairPipeline(cfg=RepairConfig())

    # shape / non-finite pre-flight rejections: zero execution
    bad = pipe.repair_trajectory(np.zeros((3, 10)), None)
    assert bad.status == "rejected_shape"
    assert bad.diagnostics["cost"]["rejected_before_run"] is True
    assert bad.diagnostics["cost"]["attempts"] == 0
    assert bad.diagnostics["cost"]["input_frames"] == 0
    assert bad.diagnostics["cost"]["passes"] == []
    nonfinite = np.zeros((3, 43))
    nonfinite[1, 0] = np.nan
    res_nf = pipe.repair_trajectory(nonfinite, None)
    assert res_nf.status == "rejected_nonfinite"
    assert res_nf.diagnostics["cost"]["input_frames"] == 3
    assert res_nf.diagnostics["cost"]["attempts"] == 0

    # first-pass acceptance
    scene = standing_scene(3)
    cond = self_condition(pipe, scene)
    ok = pipe.repair_trajectory(scene, cond)
    assert ok.accepted
    cost = ok.diagnostics["cost"]
    assert cost["final_output"] == {"trajectory_pass": 0, "accepted": True,
                                    "accepted_frames": 3, "skipped_frames": 3,
                                    "slack_frames": 0}
    assert len(cost["passes"]) == 1

    # retry solve failure: totals cover both passes, output not accepted
    pipe2 = RepairPipeline(cfg=RepairConfig())
    scene2 = standing_scene(4)
    scene2[1, 36:43] = np.array([0.0, 0.0, 0.80, 1.0, 0, 0, 0])
    cond2 = self_condition(pipe2, scene2)
    _pipe_with_solver(monkeypatch, [(True, "solved"), (False, "infeasible")])
    failing = ValidationReport(passed=False, checks={
        "penetration": CheckResult(False, 0.003, 0.002, "t=1",
                                   first_failure_frame=1)})
    monkeypatch.setattr(pipe2.validator, "validate",
                        lambda traj, exclude=(): failing)
    res2 = pipe2.repair_trajectory(scene2, cond2)
    assert not res2.accepted
    c2 = res2.diagnostics["cost"]
    assert len(c2["passes"]) == 2
    assert c2["final_output"]["accepted"] is False
    assert c2["final_output"]["trajectory_pass"] is None
    # pass 0 repairs frame 1 once; the retry escalates prediction ->
    # prev_accepted -> slack fallback on the same frame
    assert [p["attempts"] for p in c2["passes"]] == [1, 3]
    assert c2["attempts"] == sum(p["attempts"] for p in c2["passes"]) == 4
    assert res2.diagnostics["first_attempt"]["final_validation_s"] >= 0.0


def test_cost_view_is_serializable_and_event_recomputable(monkeypatch) -> None:
    """R6: the totals view is plain JSON (the report writes it verbatim) and
    every counter can be recomputed from the per-pass event lists."""
    import json

    pipe = RepairPipeline(cfg=RepairConfig())
    scene = standing_scene(4)
    scene[1, 36:43] = np.array([0.0, 0.0, 0.80, 1.0, 0, 0, 0])
    cond = self_condition(pipe, scene)
    _pipe_with_solver(monkeypatch, [(True, "solved")])
    monkeypatch.setattr(pipe.validator, "validate",
                        lambda traj, exclude=(): ValidationReport(passed=True))
    res = pipe.repair_trajectory(scene, cond)
    cost = res.diagnostics["cost"]
    text = json.dumps(cost, default=float)          # serializable as-is
    assert json.loads(text)["schema"] == 2
    for p in cost["passes"]:
        assert p["attempts"] == len(p["attempts_detail"])
        assert p["qp_calls_total"] == sum(a["qp_calls"] for a in p["attempts_detail"])
        assert p["iterations_total"] == sum(a["iterations"]
                                            for a in p["attempts_detail"])
        assert p["slack_attempts"] == sum(1 for a in p["attempts_detail"]
                                          if a["slack_used"])
        assert p["checks_total"] == len(p["checks"])
        assert p["processed_frame_visits"] == len(p["timings_s"])
    assert cost["checks_total"] == sum(p["checks_total"] for p in cost["passes"])
    assert cost["attempts"] == sum(p["attempts"] for p in cost["passes"])
