"""Review regressions: only synthetic artifacts, clocks and subprocesses."""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/run_w1_d1_lr_schedule_supervisor.py"
sys.path.insert(0, str(ROOT / "src"))
from ireset.eval import w1_d1_lr_protocol as protocol
from ireset.eval import w1_d1_lr_artifacts as artifacts


def _supervisor():
    spec = importlib.util.spec_from_file_location("lr_review_supervisor", SUPERVISOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("leader_exits", [False, True])
def test_cleanup_reaps_worker_group(tmp_path, leader_exits):
    sup = _supervisor()
    marker = tmp_path / "worker"
    worker = ("import os, signal, sys, time; from pathlib import Path; "
              "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "Path(sys.argv[1]).write_text(f'{os.getpid()} {os.getpgrp()}'); time.sleep(15)")
    leader = """
import runpy, sys, time
from pathlib import Path
spawn = runpy.run_path(sys.argv[1])['InheritedGroupWorker']
p = spawn([sys.executable, '-c', sys.argv[2], sys.argv[3]], start_new_session=True)
while not Path(sys.argv[3]).is_file():
    time.sleep(0.01)
if sys.argv[4] == 'False':
    p.wait(timeout=0.01)
"""
    rc, unclean, pgid = sup.wait_in_owned_group(
        [sys.executable, "-c", leader, str(SUPERVISOR), worker, str(marker), str(leader_exits)],
        deadline=time.perf_counter() + 2.0)
    pid, group = map(int, marker.read_text().split())
    assert group == pgid and pid != pgid and unclean is True
    assert rc == (0 if leader_exits else -signal.SIGKILL)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # neither continuing nor left as a zombie


def test_synthetic_scope_default_and_reserved_paths(tmp_path, monkeypatch):
    sup = _supervisor()
    monkeypatch.setattr(sup, "ROOT", tmp_path)
    seen = []
    monkeypatch.setattr(sup, "_capture_chain_return", lambda args, path: seen.append(path) or 0)
    assert sup.main(["--synthetic"]) == 0
    assert seen == [tmp_path / protocol.AUDIT_ROOT / "synthetic/acceptance_default"]
    candidate = tmp_path / protocol.RUN_ROOT
    admission = tmp_path / protocol.R_ROOT
    alias = tmp_path / "alias"
    alias.symlink_to(candidate)
    for forbidden in (candidate, admission, candidate / "child", alias, tmp_path / "derived/experiments"):
        for inner in ([], ["--inner-t0-hex", sup._T0.hex()], ["--chain-observer-t0-hex", sup._T0.hex()]):
            with pytest.raises(SystemExit) as exc:
                sup.main(["--synthetic", "--output-root", str(forbidden), *inner])
            assert exc.value.code == 2
    assert not candidate.exists() and not admission.exists() and len(seen) == 1


def test_bad_index_closes_f_without_numeric_work(tmp_path):
    from ireset.eval import w1_d1_lr_schedule as lr
    from ireset.eval import w1_smallset_fit_diagnostic as fit

    stage = tmp_path / "d1"
    budget = fit.StageBudget("D1", 1800, 60)
    lr.write_json(stage / "artifacts_index.json", {"schema": fit.ARTIFACTS_INDEX_SCHEMA, "files": []})
    fit.finish_stage(stage, {"stage": "D1", "budget": budget.facts(), "local_complete_evidence": True}, budget)
    baseline = {"files": {str(stage / "stage_report.json"): lr.file_record(stage / "stage_report.json")},
                "artifact_files": {}}
    baseline["sha256"] = fit._canonical_sha256(baseline)
    plan = lr.SchedulePlan("D1")
    calls = []
    f = fit.run_f_verification(
        stage_specs=[{"stage_dir": stage, "plan": plan, "registration": None, "decoder": None,
                      "loss_fn": None, "identity": {},
                      "verify_extension": lambda b: lr.verify_schedule_stage(stage, plan, baseline, b)}],
        out_dir=tmp_path / "f", budget=fit.StageBudget("F", 600, 60),
        runtime_guard=fit.SyntheticRuntimeGuard(), build_modules=lambda: calls.append("built"),
        authorization={"includes": ["D1", "F"]})
    assert f["summary_doc"]["verdict"] == "incomplete"
    assert f["summary_doc"]["stages"][0]["extension_problems"] == ["lr_artifact_index_files_or_schema_invalid"]
    assert f["return_coverage"]["covered"] is True and calls == []
    assert (tmp_path / "f/return_observation.json").is_file()


@pytest.mark.parametrize("stage", ["D1", "F"])
@pytest.mark.parametrize("slow_file", ["costs.json", "return_observation.json", "all"])
def test_ledger_withdraws_cost_monotonically(stage, slow_file, tmp_path, monkeypatch):
    from ireset.eval import w1_d1_lr_workflow as flow
    from ireset.eval import w1_d1_lr_schedule as lr
    from ireset.eval import w1_smallset_fit_diagnostic as fit

    rel, cap = protocol.STAGE_REPORTS[stage]
    clock = fit._FakeClock(0.)
    budget = fit.StageBudget(stage, cap, 60, t0=0., clock=clock)
    ledger = flow.StageLedger(tmp_path / "costs.json")
    ledger.begin(budget)
    report_path = tmp_path / rel
    report_path.parent.mkdir()
    status_key = "local_complete_evidence" if stage == "D1" else "complete_evidence"
    fit._write_summary_sealed(report_path, {status_key: True, "budget": budget.facts()}, budget, clock=clock)
    original = lr.write_json
    seen = []

    def delayed(path, doc):
        original(path, doc)
        if path == ledger.path:
            seen.append(doc["rows"][0]["measured_wall_s"])
        if slow_file == "all":
            clock.t = (clock.t + cap) * 4
        elif clock.t == 0 and path.name == slow_file:
            clock.t = cap + 1

    monkeypatch.setattr(lr, "write_json", delayed)
    withdraw = fit.I_WITHDRAW_ON_EXCEED if stage == "D1" else fit.F_WITHDRAW_ON_EXCEED
    returned = ledger.finish(budget, report_path, withdraw)
    proof = lr.read_object(report_path.parent / "return_observation.json")
    report = lr.read_object(report_path)
    row = lr.read_object(ledger.path)["rows"][0]
    assert returned["covered"] is False and returned["final_doc"] == report
    assert report[status_key] is False and proof["covered"] is False
    assert row == proof["stage_ledger_row"] and proof["summary_sha256"] == protocol.sha(report_path)
    assert (row["measured_wall_s"] == proof["registered_measured_s"]
            == report["seal"]["measured_wall_s_at_summary"] >= cap + 1)
    assert seen == sorted(seen)
    if slow_file == "all":
        assert row["status"] == "cost_unproven" and proof["cost_coverage_unproven"] is True
    else:
        assert clock.t <= row["measured_wall_s"] and row["status"] == "incomplete"


def _write_r_fixture(output, t0, pid, core):
    """The numerical worker is not launched; supply an internally bound R packet."""
    attempt = output / "attempt_001"
    attempt.mkdir()
    cap, reserve = protocol.CAPS["r-admit"]
    reg = 20.
    launch = {"launch_id": "r-fixture", "mode": "r-admit", "t0_perf_counter": t0,
              "wall_cap_s": cap, "reserve_s": reserve, "worker_process_group": pid}
    core._write_json(attempt / "outer_launch.json", launch)
    core._write_json(attempt / "r_admission.json", {
        "admission_passed": True, "problems": [], "outer_launch": launch,
        "seal": {"status": "sealed_within_budget", "terminal": True, "t0_perf_counter": t0,
                 "wall_cap_s": cap, "measured_wall_s_at_summary": reg}})
    admission_sha = protocol.sha(attempt / "r_admission.json")
    core._write_json(attempt / "return_observation.json", {
        "schema": "n0_return_observation_v1", "covered": True, "summary_sha256": admission_sha,
        "wall_cap_s": cap, "registered_measured_s": reg, "elapsed_s": 3.})
    core._write_json(attempt / "outer_return_observation.json", {
        **launch, "wait_return_code": 0, "worker_hard_deadline_timeout": False, "wait_elapsed_s": 4.,
        "problems": [], "summary_sha256": admission_sha, "registered_measured_s": reg,
        "launch_sha256": protocol.sha(attempt / "outer_launch.json"),
        "in_process_proof_sha256": protocol.sha(attempt / "return_observation.json")})
    core._write_json(attempt / "acceptance_verdict.json", {
        **launch, "verdict": "candidate_complete", "problems": [], "summary_sha256": admission_sha,
        "registered_measured_s": reg, "receipt_write_completed_elapsed_s": 5.,
        "outer_return_observation_sha256": protocol.sha(attempt / "outer_return_observation.json")})
    core._write_json(attempt / "outer_stdout_observation.json", {
        **launch, "registered_measured_s": reg, "stdout_flushed_elapsed_s": 6.,
        "verdict_sha256": protocol.sha(attempt / "acceptance_verdict.json")})
    core._write_json(attempt / "authorization_draft.json", {
        "authorized": False, "template_only": True, "admission_package_sha256": admission_sha,
        "r_observer_actual_return": None})


def _r_observer_fixture(output, t0, slow):
    sup = _supervisor()
    core = sup.load_core()
    clock = SimpleNamespace(t=t0 + 8.)
    write = core._write_json

    def slow_write(path, doc):
        write(path, doc)
        if slow and path.name == protocol.SAMPLE_NAME:
            clock.t = t0 + 21.

    def inner(*args, **kwargs):
        _write_r_fixture(output, t0, os.getpid(), core)
        return SimpleNamespace(returncode=0)

    core._write_json = slow_write
    sup.load_core = lambda: core
    sup.time = SimpleNamespace(perf_counter=lambda: clock.t)
    sup.subprocess = SimpleNamespace(run=inner)
    args = SimpleNamespace(mode="r-admit", synthetic=False, admission=None, authorization=None, cache_root=None,
                           wall_cap_s=120.0, reserve_s=15.0)
    return sup._observe(args, output, t0)


@pytest.mark.parametrize("slow", [False, True])
def test_r_observer_requires_actual_wait_and_bound_receipt(tmp_path, monkeypatch, slow):
    from ireset.eval import w1_d1_lr_workflow as flow
    from ireset.eval import w1_smallset_fit_diagnostic as fit

    sup = _supervisor()
    t0 = time.perf_counter() - 10.
    output = tmp_path / protocol.R_ROOT
    monkeypatch.setattr(sup, "_T0", t0)
    monkeypatch.setattr(sup, "_base_command", lambda _args, _output: [
        sys.executable, __file__, str(output), t0.hex(), str(slow)])
    capture_rc = sup._capture_r_return(SimpleNamespace(wall_cap_s=120.), output)
    attempt = output / "attempt_001"
    admission = attempt / "r_admission.json"
    protocol.verify_observer_files(attempt, "r-admit")  # both sets of provisional files look complete
    with pytest.raises(fit.FitDiagnosticError, match="actual_return_missing_or_invalid"):
        flow.verify_admission_return(admission, root=tmp_path)
    assert capture_rc == (3 if slow else 0)
    receipt = output / protocol.R_RETURN_NAME
    if slow:
        assert not receipt.exists() and not (output / "authorization_draft.json").exists()
        with pytest.raises(fit.FitDiagnosticError, match="actual_return_missing_or_invalid"):
            flow.verify_admission_return(admission, {"sha256": "0" * 64, "capture_actual_return_code": 0}, root=tmp_path)
    else:
        draft, _ = protocol._object_and_sha(output / "authorization_draft.json")
        assert draft["authorized"] is False and draft["template_only"] is True
        binding = draft["r_observer_actual_return"]
        with pytest.raises(fit.FitDiagnosticError, match="actual_capture_exit_required"):
            flow.verify_admission_return(admission, binding, root=tmp_path)
        binding["capture_actual_return_code"] = capture_rc
        assert flow.verify_admission_return(admission, binding, root=tmp_path) == binding
        observed, _ = protocol._object_and_sha(receipt)
        assert observed["observer_return_code"] == 0 and observed["observer_pid"] != os.getpid()
        assert (observed["observer_return_perf_counter"] - t0
                == observed["elapsed_after_observer_wait_s"] <= observed["registered_measured_s"])
        sample = output / protocol.SAMPLE_NAME
        sample.write_bytes(sample.read_bytes() + b"\n")
        with pytest.raises(fit.FitDiagnosticError, match="actual_observer_return_invalid"):
            flow.verify_admission_return(admission, binding, root=tmp_path)


@pytest.mark.parametrize("late", ["wait", "receipt", "archive", None])
def test_capture_closes_external_wait_and_both_writes(tmp_path, monkeypatch, late):
    sup = _supervisor()
    t0, registered, pid = 100.0, 20.0, os.getpid() + 1
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    binding = {"launch_id": "test", "t0_perf_counter": t0, "registered_measured_s": registered}
    packet = (binding, "a" * 64, {"elapsed_after_completion_write_s": 15.0}, "b" * 64,
              {"fixture": True, "outer_launch": {"worker_process_group": pid}}, {})
    space = artifacts.ArtifactSpace(tmp_path)
    space.write_json(tmp_path / "cohort_launch_claim.json", {
        "cohort": protocol.COHORT, "mode": "chain", "state": "running", "max_attempts": 1,
        "synthetic": True, "t0_perf_counter": t0, "observer_pid": pid, "collector_pid": os.getpid(),
        "artifact_space": artifacts.policy()})
    monkeypatch.setattr(protocol, "verify_observer_files", lambda *a: packet)
    returned = 121.0 if late == "wait" else 116.0
    clock = SimpleNamespace(t=returned)
    monkeypatch.setattr(sup, "time", SimpleNamespace(perf_counter=lambda: clock.t))
    original = artifacts.ArtifactSpace.write_json

    def write(space, path, doc, **kwargs):
        original(space, path, doc, **kwargs)
        if ((late == "receipt" and path.name == protocol.CHAIN_WAIT_NAME)
                or (late == "archive" and path.name == protocol.CHAIN_ARCHIVE_NAME)):
            clock.t = 121.0

    monkeypatch.setattr(artifacts.ArtifactSpace, "write_json", write)
    rc = sup._record_chain_return(tmp_path, rc=0, unclean=False, pid=pid, returned=returned, t0=t0)
    if late is None:
        assert rc == 0
        assert protocol.verify_chain_actual_return(attempt, collector_return_code=rc) == packet
    else:
        assert rc == 3 and (tmp_path / protocol.CHAIN_CAPTURE_FAILURE).is_file()
        with pytest.raises(ValueError, match="lr_chain_capture_failed"):
            protocol.verify_chain_actual_return(attempt, collector_return_code=0)
        assert (tmp_path / protocol.CHAIN_WAIT_NAME).exists() == (late != "wait")
        assert (tmp_path / protocol.CHAIN_ARCHIVE_NAME).exists() == (late == "archive")


def test_artifact_replacement_counts_old_and_temporary_bytes(tmp_path):
    space = artifacts.ArtifactSpace(tmp_path, cap_bytes=artifacts.CLOSURE_RESERVE_BYTES + 100)
    path = tmp_path / "artifact.json"
    space.write_bytes(path, b"a" * 60)
    with pytest.raises(artifacts.ArtifactLimit, match="need=41:available=40"):
        space.write_bytes(path, b"b" * 41)
    assert path.read_bytes() == b"a" * 60 and not (tmp_path / "artifact.json.tmp").exists()
    space.write_bytes(path, b"b" * 40)
    assert space.used_bytes() == 40


def test_supervised_logs_cannot_outgrow_reserved_slots(tmp_path):
    sup = _supervisor()
    path = tmp_path / "worker_stdout.log"
    err = tmp_path / "worker_stderr.log"
    with path.open("xb") as stdout, err.open("xb") as stderr:
        child = sup.InheritedGroupWorker([sys.executable, "-c",
            f"import os; os.write(1, b'x' * {artifacts.LOG_BYTES + 1})"],
            stdout=stdout, stderr=stderr, limit_logs=True)
        with pytest.raises(artifacts.ArtifactLimit, match="log_slot_exceeded"):
            child.wait()
    assert path.stat().st_size == artifacts.LOG_BYTES


if __name__ == "__main__":
    raise SystemExit(_r_observer_fixture(Path(sys.argv[1]), float.fromhex(sys.argv[2]), sys.argv[3] == "True"))
