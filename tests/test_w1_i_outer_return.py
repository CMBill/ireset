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


def _populate(command, mutation):
    out = Path(command[command.index("--output-dir") + 1])
    launch_id = command[command.index("--launch-id") + 1]
    t0 = float.fromhex(command[command.index("--outer-t0-hex") + 1])
    assert t0 == 100.0
    for name in ("d1", "d6"):
        _report(out / name / "stage_report.json", t0, 100.0, 20.0, 4.0)
    _report(out / "f/f_report.json", 103.0, 600.0, 5.0, 4.0)
    doc, proof = _report(out / "acceptance_summary.json", t0, 100.0, 20.0, 9.0)
    doc.update({
        "outer_launch": {"launch_id": launch_id, "t0_perf_counter": t0,
                         "cli_entry_perf_counter": 101.0, "supervisor_declared": True},
        "label_verdict": "pending_outer_return", "frozen_plan_defaults_problems": [],
        "counterexample_ids": ["a", "b"],
        "counterexample_battery": {"a": {"caught": True}, "b": {"caught": True}},
        "all_counterexamples_caught": True,
        "d1": {"local_complete_evidence": True, "return_coverage": True,
               "stage_returned_elapsed_s": 3.0, "call_matrix_problems": [],
               "report_problems": []},
        "d1_handoff": {"handoff_ready": True, "problems": []},
        "d6": {"local_complete_evidence": True, "return_coverage": True,
               "stage_returned_elapsed_s": 3.0, "call_matrix_problems": []},
        "f": {"complete_evidence": True, "replay_passed": True,
              "return_coverage": True, "stage_returned_elapsed_s": 3.0,
              "stages": [{"stage": "D1", "replay_passed": True},
                         {"stage": "D6", "replay_passed": True}]},
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
    path = out / "acceptance_summary.json"
    path.write_text(json.dumps(doc))
    proof["summary_sha256"] = supervisor._sha(path)
    (out / "return_observation.json").write_text(json.dumps(proof))
    if mutation == "bad_proof":
        (out / "return_observation.json").unlink()


@pytest.mark.parametrize("mutation,expected", [
    (None, None),
    ("late_wait", "actual_wait_not_covered"),
    ("nonzero", "worker_exit:1"),
    ("roster", "counterexample_battery_incomplete"),
    ("identity", "launch_identity_mismatch"),
    ("stage_return", "d1:actual_stage_return_not_covered"),
    ("missing_d6", "fresh_replay_incomplete"),
    ("stage_name_list", "fresh_replay_incomplete"),
    ("stage_name_dict", "fresh_replay_incomplete"),
    ("bad_proof", "report_or_proof_unreadable"),
    ("slow_receipt", "receipt_write_not_covered"),
    ("slow_verdict", "verdict_write_not_covered"),
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

    def controlled_write(path, obj):
        real_write(path, obj)
        if ((mutation == "slow_receipt" and path.name == "outer_return_observation.json")
                or (mutation == "slow_verdict" and path.name == "acceptance_verdict.json"
                    and obj["verdict"] == "candidate_complete")):
            ticks["now"] = 121.0

    monkeypatch.setattr(supervisor, "_write_json", controlled_write)

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
        assert verdict["outer_return_observation_sha256"] == supervisor._sha(
            out / "outer_return_observation.json")
        event = json.loads(printed[-1])
        assert event["event"] == "w1_i_outer_verdict_write_observed"
        assert event["verdict"] == "complete"
        assert event["verdict_sha256"] == supervisor._sha(out / "acceptance_verdict.json")
        assert receipt["wait_elapsed_s"] <= event["elapsed_s"] <= 20.0
    else:
        assert rc == 3 and verdict["verdict"] == "incomplete"
        assert expected in verdict["problems"]
        assert not any("w1_i_outer_verdict_write_observed" in line for line in printed)


def test_counterexample_roster_is_read_from_the_frozen_source():
    ids = supervisor._expected_counterexamples()
    assert len(ids) == 115
    assert "registration_tamper" in ids


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
    report(out / name / "stage_report.json", 100.0, {})
report(out / "f/f_report.json", 600.0, {})
report(out / "acceptance_summary.json", 100.0, {
    "outer_launch": {"launch_id": launch_id, "t0_perf_counter": t0,
                     "cli_entry_perf_counter": time.perf_counter(), "supervisor_declared": True},
    "label_verdict": "pending_outer_return", "frozen_plan_defaults_problems": [],
    "counterexample_ids": ["a"], "counterexample_battery": {"a": {"caught": True}},
    "all_counterexamples_caught": True,
    "d1": {"local_complete_evidence": True, "return_coverage": True,
           "stage_returned_elapsed_s": 0.0, "call_matrix_problems": [], "report_problems": []},
    "d1_handoff": {"handoff_ready": True, "problems": []},
    "d6": {"local_complete_evidence": True, "return_coverage": True,
           "stage_returned_elapsed_s": 0.0, "call_matrix_problems": []},
    "f": {"complete_evidence": True, "replay_passed": True,
          "return_coverage": True, "stage_returned_elapsed_s": 0.0,
          "stages": [{"stage": "D1", "replay_passed": True},
                     {"stage": "D6", "replay_passed": True}]}})
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
