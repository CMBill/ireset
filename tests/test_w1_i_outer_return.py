"""Synthetic supervisor timing and fail-closed return coverage."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "w1_i_supervisor", ROOT / "scripts/run_w1_i_synthetic_acceptance.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def _report(path, t0, cap, registered, elapsed):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"seal": {"status": "sealed_within_budget", "terminal": True,
                    "t0_perf_counter": t0, "wall_cap_s": cap,
                    "measured_wall_s_at_summary": registered},
           "budget": {"t0_perf_counter": t0, "wall_cap_s": cap,
                      "measured_wall_s_at_summary": registered}}
    path.write_text(json.dumps(doc))
    proof = {"schema": "n0_return_observation_v1", "covered": True,
             "elapsed_s": elapsed, "registered_measured_s": registered,
             "wall_cap_s": cap, "summary_sha256": supervisor._sha(path)}
    (path.parent / "return_observation.json").write_text(json.dumps(proof))
    return doc, proof


def _update_report(path, changes):
    doc = json.loads(path.read_text())
    doc.update(changes)
    path.write_text(json.dumps(doc))
    proof_path = path.parent / "return_observation.json"
    proof = json.loads(proof_path.read_text())
    proof["summary_sha256"] = supervisor._sha(path)
    proof_path.write_text(json.dumps(proof))


def _stage_facts(stage):
    return {"schema": "w1_smallset_fit_diagnostic_v1", "stage": stage,
            "local_complete_evidence": True, "local_mean_fit_passed": True,
            "local_conclusion": "complete_fit", "final_verification_status": "pending_F",
            "call_matrix": {"problems": []}, "problems": [], "failure": None}


def _f_facts():
    return {"schema": "w1_smallset_fit_f_report_v1", "complete_evidence": True,
            "replay_passed": True, "mean_fit_passed": True, "verdict": "fit_pass",
            "final_verification_status": "final_F", "tolerance_policy": "bitwise_no_tolerance",
            "stages": [{"stage": s, "replay_enabled": True, "replay_passed": True,
                        "complete_evidence": True, "mean_fit_passed": True,
                        "local_reverify_problems": [], "replay_problems": []}
                       for s in ("D1", "D6")]}


def _populate(command, mutation):
    out = Path(command[command.index("--output-dir") + 1])
    launch_id = command[command.index("--launch-id") + 1]
    t0 = float.fromhex(command[command.index("--outer-t0-hex") + 1])
    assert t0 == 100.0
    for name in ("d1", "d6"):
        _report(out / name / "stage_report.json", t0, 100.0, 20.0, 4.0)
        _update_report(out / name / "stage_report.json", _stage_facts(name.upper()))
    _report(out / "f/f_report.json", 103.0, 600.0, 5.0, 4.0)
    _update_report(out / "f/f_report.json", _f_facts())
    doc, proof = _report(out / "acceptance_summary.json", t0, 100.0, 20.0, 9.0)
    doc.update({
        "outer_launch": {"launch_id": launch_id, "t0_perf_counter": t0,
                         "cli_entry_perf_counter": 101.0, "supervisor_declared": True},
        "label_verdict": "pending_outer_return", "frozen_plan_defaults_problems": [],
        "counterexample_ids": ["a", "b"],
        "counterexample_battery": {"a": {"caught": True}, "b": {"caught": True}},
        "all_counterexamples_caught": True,
        "d1": {"local_complete_evidence": True, "return_coverage": True,
               "local_mean_fit_passed": True, "local_conclusion": "complete_fit",
               "seal": "sealed_within_budget",
               "stage_returned_elapsed_s": 3.0, "call_matrix_problems": [],
               "report_problems": []},
        "d1_handoff": {"handoff_ready": True, "problems": []},
        "d6": {"local_complete_evidence": True, "return_coverage": True,
               "local_mean_fit_passed": True, "local_conclusion": "complete_fit",
               "seal": "sealed_within_budget",
               "stage_returned_elapsed_s": 3.0, "call_matrix_problems": []},
        "f": {"complete_evidence": True, "replay_passed": True,
              "mean_fit_passed": True, "verdict": "fit_pass",
              "return_coverage": True, "stage_returned_elapsed_s": 3.0,
              "stages": [{"stage": s, "replay_enabled": True, "replay_passed": True,
                          "skip_reason": None, "replay_problems": []}
                         for s in ("D1", "D6")]},
    })
    if mutation == "roster":
        doc["counterexample_ids"].pop()
        doc["counterexample_battery"].pop("b")
    if mutation == "identity":
        doc["outer_launch"]["launch_id"] = "wrong"
    if mutation == "stage_return":
        doc["d1"]["stage_returned_elapsed_s"] = 5.0
    if mutation == "missing_d6":
        doc["f"]["stages"].pop()
    if mutation == "stage_name_list":
        doc["f"]["stages"][0]["stage"] = []
    if mutation == "stage_name_dict":
        doc["f"]["stages"][0]["stage"] = {}
    if mutation == "f_disk_incomplete":
        _update_report(out / "f/f_report.json", {"complete_evidence": False,
                       "replay_passed": False, "mean_fit_passed": None,
                       "verdict": "incomplete", "stages": []})
    if mutation == "d1_disk_incomplete":
        _update_report(out / "d1/stage_report.json", {"local_complete_evidence": False})
    if mutation == "d6_disk_mismatch":
        _update_report(out / "d6/stage_report.json", {"local_mean_fit_passed": False})
    if mutation == "f_disk_stage_invalid":
        changed = _f_facts()["stages"]
        changed[0]["stage"] = []
        _update_report(out / "f/f_report.json", {"stages": changed})
    if mutation == "fit_insufficient":
        _update_report(out / "d6/stage_report.json", {"local_mean_fit_passed": False,
                       "local_conclusion": "fit_insufficient"})
        f = _f_facts()
        f["stages"][1]["mean_fit_passed"] = False
        f.update(mean_fit_passed=False, verdict="fit_insufficient")
        _update_report(out / "f/f_report.json", f)
        doc["d6"].update(local_mean_fit_passed=False, local_conclusion="fit_insufficient")
        doc["f"].update(mean_fit_passed=False, verdict="fit_insufficient")
    path = out / "acceptance_summary.json"
    path.write_text(json.dumps(doc))
    proof["summary_sha256"] = supervisor._sha(path)
    (out / "return_observation.json").write_text(json.dumps(proof))
    if mutation == "bad_proof":
        (out / "return_observation.json").unlink()


@pytest.mark.parametrize("mutation,expected", [
    (None, None),
    ("fit_insufficient", None),
    ("late_wait", "actual_wait_not_covered"),
    ("nonzero", "worker_exit:1"),
    ("roster", "counterexample_battery_incomplete"),
    ("identity", "launch_identity_mismatch"),
    ("stage_return", "d1:actual_stage_return_not_covered"),
    ("missing_d6", "fresh_replay_incomplete"),
    ("stage_name_list", "fresh_replay_incomplete"),
    ("stage_name_dict", "fresh_replay_incomplete"),
    ("f_disk_incomplete", "f:report_facts_mismatch"),
    ("d1_disk_incomplete", "d1:report_facts_mismatch"),
    ("d6_disk_mismatch", "d6:report_facts_mismatch"),
    ("f_disk_stage_invalid", "f:report_facts_mismatch"),
    ("bad_proof", "report_or_proof_unreadable"),
    ("slow_receipt", "receipt_write_not_covered"),
    ("slow_verdict", "verdict_write_not_covered"),
    ("slow_final_stdout", "stdout_not_covered"),
    ("slow_final_stdout_hard", "stdout_not_covered"),
    ("slow_stdout_observation", "stdout_observation_write_not_covered"),
    ("stdout_failure", "stdout_write_failed"),
    ("source_drift", "implementation_changed_during_run"),
])
def test_actual_wait_and_all_evidence_are_required(tmp_path, monkeypatch, capsys, mutation, expected):
    ticks = {"now": 105.0}
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: ticks["now"])
    hashes = iter([{"source": "before"},
                   {"source": "after" if mutation == "source_drift" else "before"}])
    monkeypatch.setattr(supervisor, "_source_hashes", lambda: next(hashes))
    monkeypatch.setattr(supervisor, "_expected_counterexamples", lambda: {"a", "b"})
    real_write = supervisor._write_json
    real_print = print

    def controlled_write(path, obj):
        real_write(path, obj)
        if ((mutation == "slow_receipt" and path.name == "outer_return_observation.json")
                or (mutation == "slow_verdict" and path.name == "acceptance_verdict.json"
                    and obj["verdict"] == "candidate_complete")
                or (mutation == "slow_stdout_observation"
                    and path.name == "outer_stdout_observation.json")):
            ticks["now"] = 121.0

    monkeypatch.setattr(supervisor, "_write_json", controlled_write)

    def controlled_print(value, **kwargs):
        if (mutation == "stdout_failure" and isinstance(value, str)
                and "w1_i_outer_verdict_" in value):
            raise BrokenPipeError("closed stdout")
        real_print(value, **kwargs)
        if (isinstance(value, str) and "w1_i_outer_verdict_" in value
                and mutation in ("slow_final_stdout", "slow_final_stdout_hard")):
            ticks["now"] = 201.0 if mutation == "slow_final_stdout_hard" else 121.0

    monkeypatch.setattr(supervisor, "print", controlled_print, raising=False)

    class Child:
        pid = 1234

        def __init__(self, command, **kwargs):
            assert command[:3] == ["uv", "run", "--locked"]
            assert kwargs["start_new_session"] is True
            _populate(command, mutation)

        def wait(self, timeout=None):
            ticks["now"] = 121.0 if mutation == "late_wait" else 110.0
            return 1 if mutation == "nonzero" else 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    args = SimpleNamespace(output_root=tmp_path, wall_cap_s=100.0, reserve_s=10.0)
    rc = supervisor.run(args, t0=100.0)
    out = tmp_path / "attempt_001"
    verdict = json.loads((out / "acceptance_verdict.json").read_text())
    receipt = json.loads((out / "outer_return_observation.json").read_text())
    printed = capsys.readouterr().out.splitlines()
    assert receipt["wait_return_perf_counter"] == (121.0 if mutation == "late_wait" else 110.0)
    if expected is None:
        assert rc == 0 and verdict["verdict"] == "candidate_complete" and verdict["problems"] == []
        assert receipt["wait_elapsed_s"] == 10.0
        assert receipt["in_process_elapsed_s"] <= receipt["wait_elapsed_s"] <= 20.0
        for stage, name in (("d1", "stage_report.json"), ("d6", "stage_report.json"),
                            ("f", "f_report.json")):
            assert receipt["stage_artifacts"][stage] == {
                "report_sha256": supervisor._sha(out / stage / name),
                "return_observation_sha256": supervisor._sha(out / stage / "return_observation.json")}
        assert verdict["outer_return_observation_sha256"] == supervisor._sha(
            out / "outer_return_observation.json")
        event = json.loads(printed[-1])
        assert event["event"] == "w1_i_outer_verdict_output_pending"
        assert event["verdict"] == "pending_output_coverage"
        assert event["verdict_sha256"] == supervisor._sha(out / "acceptance_verdict.json")
        assert receipt["wait_elapsed_s"] <= event["elapsed_s"] <= 20.0
        output_proof = json.loads((out / "outer_stdout_observation.json").read_text())
        assert output_proof["stdout_flushed_elapsed_s"] >= event["elapsed_s"]
        assert output_proof["stdout_flushed_elapsed_s"] <= 20.0
        assert output_proof["verdict_sha256"] == event["verdict_sha256"]
        assert output_proof["event_sha256"] == supervisor.hashlib.sha256(
            (printed[-1] + "\n").encode()).hexdigest()
    else:
        assert rc == 3 and verdict["verdict"] == "incomplete"
        assert expected in verdict["problems"]
        assert not any('"verdict": "complete"' in line for line in printed)


def test_counterexample_roster_is_read_from_the_frozen_source():
    ids = supervisor._expected_counterexamples()
    assert len(ids) == 115
    assert "registration_tamper" in ids


@pytest.mark.parametrize("trigger,reason", [
    ("stdout", "stdout_not_covered"),
    ("stdout_failure", "stdout_write_failed"),
    ("launcher_stdout_failure", "launcher_stdout_write_failed"),
    ("observation", "stdout_observation_write_not_covered"),
    ("observation_failure", "stdout_observation_write_failed"),
    ("verdict", "verdict_write_not_covered"),
    ("receipt", "receipt_write_not_covered"),
    ("timeout", "worker_hard_deadline_timeout"),
    ("before_uv", "deadline_before_uv"),
])
def test_failure_rechecks_and_registers_slow_withdrawal(
        tmp_path, monkeypatch, trigger, reason):
    ticks = {"now": 191.0 if trigger == "before_uv" else 105.0}
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: ticks["now"])
    monkeypatch.setattr(supervisor, "_source_hashes", lambda: {"source": "same"})
    monkeypatch.setattr(supervisor, "_expected_counterexamples", lambda: {"a", "b"})
    real_write = supervisor._write_json
    withdrawals = []

    def slow_write(path, doc):
        if trigger == "observation_failure" and path.name == "outer_stdout_observation.json":
            ticks["now"] = 121.0
            raise OSError("injected observation write failure")
        real_write(path, doc)
        if path.name == "acceptance_verdict.json" and doc.get("verdict") == "incomplete":
            withdrawals.append(doc.get("registered_measured_s"))
            if len(withdrawals) == 1:
                ticks["now"] = max(201.0, ticks["now"] + 80.0)
        elif ((trigger == "observation" and path.name == "outer_stdout_observation.json")
              or (trigger == "receipt" and path.name == "outer_return_observation.json")
              or (trigger == "verdict" and path.name == "acceptance_verdict.json"
                  and doc.get("verdict") == "candidate_complete")):
            ticks["now"] = 121.0

    def slow_print(value, **kwargs):
        if trigger == "launcher_stdout_failure" and value.startswith("Synthetic attempt:"):
            ticks["now"] = 121.0
            raise BrokenPipeError("injected launcher stdout failure")
        if isinstance(value, str) and "w1_i_outer_verdict_output_pending" in value:
            if trigger == "stdout_failure":
                ticks["now"] = 121.0
                raise BrokenPipeError("injected stdout failure")
            if trigger == "stdout":
                ticks["now"] = 121.0

    class Child:
        pid = 1234

        def __init__(self, command, **kwargs):
            assert trigger not in ("before_uv", "launcher_stdout_failure")
            self.calls = 0
            _populate(command, None)

        def wait(self, timeout=None):
            self.calls += 1
            if trigger == "timeout" and self.calls == 1:
                ticks["now"] = 201.0
                raise supervisor.subprocess.TimeoutExpired("fake uv", timeout)
            if trigger != "timeout":
                ticks["now"] = 110.0
            return -15 if trigger == "timeout" else 0

    monkeypatch.setattr(supervisor, "_write_json", slow_write)
    monkeypatch.setattr(supervisor, "print", slow_print, raising=False)
    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    monkeypatch.setattr(supervisor.os, "killpg", lambda *args: None)
    args = SimpleNamespace(output_root=tmp_path, wall_cap_s=100.0, reserve_s=10.0)
    assert supervisor.run(args, t0=100.0) == 3
    out = tmp_path / "attempt_001"
    verdict = json.loads((out / "acceptance_verdict.json").read_text())
    assert verdict["verdict"] == "incomplete" and reason in verdict["problems"]
    assert verdict["observed_elapsed_s"] >= 101.0
    assert verdict["registered_measured_s"] >= 101.0
    assert verdict["hard_deadline_exceeded"] is True
    assert verdict["cost_coverage_proven"] is False
    assert verdict["cost_coverage_status"] == "pending_external_return"
    assert verdict["sealed_registered_measured_s"] == (
        None if trigger in ("before_uv", "launcher_stdout_failure") else 20.0)
    observation = json.loads((out / "failure_cost_observation.json").read_text())
    assert observation["verdict_sha256"] == supervisor._sha(out / "acceptance_verdict.json")
    assert observation["elapsed_after_verdict_write_s"] >= 101.0
    assert observation["elapsed_after_verdict_write_s"] <= verdict["registered_measured_s"]
    assert len(withdrawals) >= 2


@pytest.mark.parametrize("delay_point", ["observation_write", "verdict_hash"])
def test_failure_rechecks_delayed_cost_proof(tmp_path, monkeypatch, delay_point):
    ticks = {"now": 121.0}
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: ticks["now"])
    real_write, real_sha = supervisor._write_json, supervisor._sha
    injected = {"done": False}

    def slow_write(path, doc):
        real_write(path, doc)
        if (delay_point == "observation_write" and path.name == "failure_cost_observation.json"
                and not injected["done"]):
            injected["done"] = True
            ticks["now"] = 201.0

    def slow_sha(path):
        value = real_sha(path)
        if delay_point == "verdict_hash" and not injected["done"]:
            injected["done"] = True
            ticks["now"] = 201.0
        return value

    monkeypatch.setattr(supervisor, "_write_json", slow_write)
    monkeypatch.setattr(supervisor, "_sha", slow_sha)
    path = tmp_path / "acceptance_verdict.json"
    supervisor._write_incomplete_costed(path, {"registered_measured_s": 20.0,
                                        "problems": ["injected"]}, t0=100.0,
                                        wall_cap_s=100.0)
    verdict = json.loads(path.read_text())
    observation = json.loads((tmp_path / "failure_cost_observation.json").read_text())
    assert injected["done"]
    assert verdict["withdrawal_round"] >= 2
    assert verdict["registered_measured_s"] >= verdict["observed_elapsed_s"] >= 101.0
    assert verdict["hard_deadline_exceeded"] is True
    assert verdict["cost_coverage_proven"] is False
    assert observation["verdict_sha256"] == real_sha(path)
    assert observation["elapsed_after_verdict_write_s"] >= 101.0


def test_failure_cost_exhaustion_remains_explicitly_unproven(tmp_path, monkeypatch):
    ticks = {"now": 121.0}
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: ticks["now"])
    real_write = supervisor._write_json
    costs = []

    def always_slow(path, doc):
        real_write(path, doc)
        if path.name == "acceptance_verdict.json":
            costs.append(doc["registered_measured_s"])
            ticks["now"] = 100.0 + doc["registered_measured_s"] + 100.0

    monkeypatch.setattr(supervisor, "_write_json", always_slow)
    path = tmp_path / "acceptance_verdict.json"
    supervisor._write_incomplete_costed(path, {"registered_measured_s": 20.0,
                                        "problems": ["injected"]}, t0=100.0,
                                        wall_cap_s=100.0)
    verdict = json.loads(path.read_text())
    assert len(costs) == supervisor.FAILURE_COST_ROUNDS + 1
    assert costs == sorted(costs)
    assert verdict["verdict"] == "incomplete"
    assert verdict["cost_coverage_proven"] is False
    assert verdict["cost_coverage_unproven"] is True
    assert verdict["cost_coverage_status"] == "unproven_rounds_exhausted"
    assert verdict["withdrawal_rounds_exhausted"] is True
    assert verdict["observed_elapsed_s"] <= verdict["registered_measured_s"] < ticks["now"] - 100.0
    assert not (tmp_path / "failure_cost_observation.json").exists()


def test_real_subprocess_wait_is_sampled_after_its_last_output(tmp_path, monkeypatch, capsys):
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(f"#!{sys.executable}\n" + r'''
import hashlib, json, sys, time
from pathlib import Path

args = sys.argv
out = Path(args[args.index("--output-dir") + 1])
t0 = float.fromhex(args[args.index("--outer-t0-hex") + 1])
launch_id = args[args.index("--launch-id") + 1]

def report(path, cap, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = 30.0
    doc = {"seal": {"status": "sealed_within_budget", "terminal": True,
                    "t0_perf_counter": t0, "wall_cap_s": cap,
                    "measured_wall_s_at_summary": reg},
           "budget": {"t0_perf_counter": t0, "wall_cap_s": cap,
                      "measured_wall_s_at_summary": reg}}
    doc.update(extra)
    path.write_text(json.dumps(doc))
    proof = {"schema": "n0_return_observation_v1", "covered": True,
             "elapsed_s": time.perf_counter() - t0, "registered_measured_s": reg,
             "wall_cap_s": cap, "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (path.parent / "return_observation.json").write_text(json.dumps(proof))

for name in ("d1", "d6"):
    report(out / name / "stage_report.json", 100.0, {
        "schema": "w1_smallset_fit_diagnostic_v1", "stage": name.upper(),
        "local_complete_evidence": True, "local_mean_fit_passed": True,
        "local_conclusion": "complete_fit", "final_verification_status": "pending_F",
        "call_matrix": {"problems": []}, "problems": [], "failure": None})
report(out / "f/f_report.json", 600.0, {
    "schema": "w1_smallset_fit_f_report_v1", "complete_evidence": True,
    "replay_passed": True, "mean_fit_passed": True, "verdict": "fit_pass",
    "final_verification_status": "final_F", "tolerance_policy": "bitwise_no_tolerance",
    "stages": [{"stage": s, "replay_enabled": True, "replay_passed": True,
                "complete_evidence": True, "mean_fit_passed": True,
                "local_reverify_problems": [], "replay_problems": []} for s in ("D1", "D6")]})
report(out / "acceptance_summary.json", 100.0, {
    "outer_launch": {"launch_id": launch_id, "t0_perf_counter": t0,
                     "cli_entry_perf_counter": time.perf_counter(), "supervisor_declared": True},
    "label_verdict": "pending_outer_return", "frozen_plan_defaults_problems": [],
    "counterexample_ids": ["a"], "counterexample_battery": {"a": {"caught": True}},
    "all_counterexamples_caught": True,
    "d1": {"local_complete_evidence": True, "return_coverage": True,
           "local_mean_fit_passed": True, "local_conclusion": "complete_fit",
           "seal": "sealed_within_budget",
           "stage_returned_elapsed_s": 0.0, "call_matrix_problems": [], "report_problems": []},
    "d1_handoff": {"handoff_ready": True, "problems": []},
    "d6": {"local_complete_evidence": True, "return_coverage": True,
           "local_mean_fit_passed": True, "local_conclusion": "complete_fit",
           "seal": "sealed_within_budget",
           "stage_returned_elapsed_s": 0.0, "call_matrix_problems": []},
    "f": {"complete_evidence": True, "replay_passed": True,
          "mean_fit_passed": True, "verdict": "fit_pass",
          "return_coverage": True, "stage_returned_elapsed_s": 0.0,
          "stages": [{"stage": s, "replay_enabled": True, "replay_passed": True,
                      "skip_reason": None, "replay_problems": []} for s in ("D1", "D6")]}})
time.sleep(0.1)
''')
    fake_uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(supervisor, "_source_hashes", lambda: {"source": "unchanged"})
    monkeypatch.setattr(supervisor, "_expected_counterexamples", lambda: {"a"})
    args = SimpleNamespace(output_root=tmp_path / "attempts", wall_cap_s=100.0, reserve_s=10.0)
    t0 = time.perf_counter()
    assert supervisor.run(args, t0=t0) == 0
    out = tmp_path / "attempts/attempt_001"
    receipt = json.loads((out / "outer_return_observation.json").read_text())
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["wait_return_code"] == 0
    assert receipt["wait_elapsed_s"] >= receipt["in_process_elapsed_s"] + 0.09
    assert receipt["wait_elapsed_s"] <= event["elapsed_s"] <= 30.0
    assert event["verdict_sha256"] == supervisor._sha(out / "acceptance_verdict.json")
    assert event["verdict"] == "pending_output_coverage"
    output_proof = json.loads((out / "outer_stdout_observation.json").read_text())
    assert event["elapsed_s"] <= output_proof["stdout_flushed_elapsed_s"] <= 30.0


@pytest.mark.parametrize("stage", ["D1", "F"])
def test_stage_is_withdrawn_if_caller_regains_control_after_registered_cost(tmp_path, stage):
    from ireset.eval import w1_smallset_fit_diagnostic as diag

    path = tmp_path / ("stage_report.json" if stage == "D1" else "f_report.json")
    doc, proof = _report(path, 100.0, 100.0, 20.0, 19.0)
    doc.update({"local_complete_evidence": True, "local_mean_fit_passed": True,
                "local_conclusion": "complete_fit", "complete_evidence": True,
                "replay_passed": True, "mean_fit_passed": True, "verdict": "fit_pass"})
    path.write_text(json.dumps(doc))
    proof["summary_sha256"] = supervisor._sha(path)
    (tmp_path / "return_observation.json").write_text(json.dumps(proof))
    budget = diag.StageBudget(stage, 100.0, 10.0, t0=100.0, clock=lambda: 121.0)
    withdraw = diag.I_WITHDRAW_ON_EXCEED if stage == "D1" else diag.F_WITHDRAW_ON_EXCEED
    coverage = diag._observe_returned_report(path, budget, withdraw)
    disk = json.loads(path.read_text())
    assert coverage["covered"] is False
    assert disk["seal"]["status"] == "budget_exceeded_after_seal"
    assert disk["seal"]["measured_wall_s_at_summary"] >= 21.0
    if stage == "D1":
        assert disk["local_complete_evidence"] is False
        assert "return_observation_not_covered" in diag._return_proof_problems(tmp_path)
    else:
        assert disk["complete_evidence"] is False
        assert disk["mean_fit_passed"] is None
