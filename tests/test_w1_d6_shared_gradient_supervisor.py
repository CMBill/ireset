import sys
import time
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import supervise_w1_d6_shared_gradient as supervisor


def _supervise(tmp_path, code, *, timeout_s=5., log_cap=1024):
    root = tmp_path / "attempt_001"
    root.mkdir()
    t0 = time.perf_counter()
    witness = supervisor.supervise_process(
        [sys.executable, "-I", "-c", code], output_root=root, t0=t0,
        work_deadline=t0 + timeout_s, hard_deadline=t0 + timeout_s + 2,
        log_cap=log_cap,
    )
    return root, t0, witness


def test_actual_wait_group_cleanup_logs_and_return_witness(tmp_path):
    root, t0, wait = _supervise(tmp_path, "import os; print('ok', flush=True)")
    assert wait.return_code == 0 and wait.group_reaped
    assert wait.pid == wait.process_group and wait.pid != wait.supervisor_pid
    assert t0 <= wait.started_at <= wait.waited_at
    assert wait.waited_at <= wait.cleaned_at
    assert (root / "worker.stdout.log").read_bytes() == b"ok\n"
    returned = supervisor.return_witness(
        wait, t0=t0, registered_s=wait.cleaned_at - t0 + .25,
        work_deadline=t0 + 5, hard_deadline=t0 + 7, final_sample=wait.cleaned_at,
    )
    assert returned["coverage"] == "covered"
    assert returned["waited_at"] <= returned["cleaned_at"] <= returned["final_sample"]
    assert returned["logs"]["stdout"]["bytes"] == 3
    forged = replace(wait, _token=None)
    with pytest.raises(ValueError, match="return_witness_wait_record_untrusted"):
        supervisor.return_witness(forged, t0=t0, registered_s=10, work_deadline=t0 + 5,
                                  hard_deadline=t0 + 7, final_sample=wait.cleaned_at)


def test_timeout_is_waited_and_reaped_but_not_covered_success(tmp_path):
    root = tmp_path / "timeout"
    root.mkdir()
    t0 = time.perf_counter()
    code = "import time; time.sleep(60)"
    wait = supervisor.supervise_process(
        [sys.executable, "-I", "-c", code], output_root=root, t0=t0,
        work_deadline=t0 + .2, hard_deadline=t0 + 2, term_grace_s=.1,
    )
    assert wait.timed_out and wait.return_code != 0 and wait.group_reaped
    with pytest.raises(ValueError, match="return_witness_worker_not_successful"):
        supervisor.return_witness(wait, t0=t0, registered_s=5.5, work_deadline=t0 + 5,
                                  hard_deadline=t0 + 6, final_sample=wait.cleaned_at)


def test_log_overflow_is_drained_but_fails_closed(tmp_path):
    root, t0, wait = _supervise(tmp_path, "import os; os.write(1, b'x' * 4096)", log_cap=128)
    assert wait.return_code == 0 and wait.group_reaped and wait.log_overflow
    assert (root / "worker.stdout.log").stat().st_size == 128
    with pytest.raises(ValueError, match="return_witness_cleanup_or_log_failure"):
        supervisor.return_witness(wait, t0=t0, registered_s=wait.cleaned_at - t0 + .5,
                                  work_deadline=t0 + 5, hard_deadline=t0 + 7,
                                  final_sample=wait.cleaned_at)


def test_child_process_group_is_terminated_and_reaped(tmp_path):
    code = (
        "import signal, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(60)']); "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); time.sleep(60)"
    )
    root = tmp_path / "descendant"
    root.mkdir()
    t0 = time.perf_counter()
    wait = supervisor.supervise_process(
        [sys.executable, "-I", "-c", code], output_root=root, t0=t0,
        work_deadline=t0 + .3, hard_deadline=t0 + 3, term_grace_s=.5,
    )
    assert wait.timed_out and wait.group_reaped


def test_return_witness_rejects_uncovered_final_sample(tmp_path):
    _, t0, wait = _supervise(tmp_path, "pass")
    with pytest.raises(ValueError, match="return_witness_time_coverage"):
        supervisor.return_witness(wait, t0=t0, registered_s=.01, work_deadline=t0 + 5,
                                  hard_deadline=t0 + 7, final_sample=wait.cleaned_at + 1)


def test_pending_return_observation_is_fsynced_read_back_and_requires_live_witness(tmp_path):
    root, t0, wait = _supervise(tmp_path, "print('return proof', flush=True)")
    registered = wait.cleaned_at - t0 + 2
    final = wait.cleaned_at + .1
    receipt = supervisor.seal_external_return(
        root, wait, t0=t0, registered_s=registered, work_deadline=t0 + 5,
        hard_deadline=t0 + 7, clock=lambda: final,
    )
    assert receipt.observation["publication"] == "pending_actual_readback"
    assert "final_sample" not in receipt.observation
    assert receipt.observation["receipt_pre_sample"] <= receipt.final_sample
    assert supervisor.verify_external_return(root, receipt)["coverage"] == "covered"
    with pytest.raises(ValueError, match="external_return_witness_missing"):
        supervisor.verify_external_return(root, None)
    (root / "return_observation.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="external_return_observation_drift"):
        supervisor.verify_external_return(root, receipt)


def test_final_sample_over_cap_leaves_only_non_consumable_pending_record(tmp_path):
    root, t0, wait = _supervise(tmp_path, "pass")
    registered = wait.cleaned_at - t0 + .01
    with pytest.raises(ValueError, match="return_observation_actual_coverage_failed"):
        supervisor.seal_external_return(
            root, wait, t0=t0, registered_s=registered, work_deadline=t0 + 5,
            hard_deadline=t0 + 7, clock=lambda: t0 + registered + .1,
        )
    assert (root / "return_observation.json").is_file()
    import json

    pending = json.loads((root / "return_observation.json").read_text())
    assert pending["publication"] == "pending_actual_readback"
    assert "final_sample" not in pending
    with pytest.raises(ValueError, match="external_return_witness_missing"):
        supervisor.verify_external_return(root, None)


def test_nonfinite_deadline_is_rejected_before_spawn(tmp_path):
    root = tmp_path / "bad_deadline"
    root.mkdir()
    called = []

    def forbidden_spawn(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("must not spawn")

    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="worker_deadline_invalid"):
        supervisor.supervise_process(
            [sys.executable, "-I", "-c", "pass"], output_root=root, t0=t0,
            work_deadline=float("nan"), hard_deadline=t0 + 10, popen=forbidden_spawn,
        )
    assert called == []


def _synthetic_authorized_packet(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "worker_contract.txt"
    source.write_text("synthetic implementation closure v1\n")
    source_raw = source.read_bytes()
    files = {"worker_contract.txt": {"bytes": len(source_raw),
                                      "sha256": hashlib.sha256(source_raw).hexdigest()}}
    package = {
        "schema": supervisor.admission.SCHEMA,
        "execution_mode": "synthetic_test_only",
        "executable": True,
        "metadata_sha256": "",
        "implementation": {"files": files},
        "historical": {"synthetic_ic_acceptance": {"bytes": 1, "sha256": "1" * 64},
                       "verified_outcome": {"bytes": 1, "sha256": "2" * 64}},
        "status": {"real_worker": "implemented_and_reviewed_and_supervised"},
    }
    body = {key: value for key, value in package.items() if key != "metadata_sha256"}
    package["metadata_sha256"] = supervisor._canonical_sha(body)
    auth = {
        "schema": supervisor.admission.AUTH_SCHEMA,
        "cohort": "w1_d6_shared_gradient_step0_v1",
        "authorized": True, "template_only": False,
        "includes": ["single_step0_diagnostic"], "attempts": 1,
        "checklist": dict.fromkeys(supervisor.admission.CHECKLIST, True),
        "approval_reference": "synthetic-test-only",
        "launch_id": "a" * 64,
        "bindings": {"admission_metadata_sha256": package["metadata_sha256"],
                     "implementation_files": files,
                     "ic_acceptance": package["historical"]["synthetic_ic_acceptance"],
                     "historical_d6_outcome": package["historical"]["verified_outcome"]},
    }
    auth_raw = (json.dumps(auth, sort_keys=True, separators=(",", ":")) + "\n").encode()
    auth_path = tmp_path / "authorization.json"
    auth_path.write_bytes(auth_raw)
    return source_root, package, auth_path, hashlib.sha256(auth_raw).hexdigest(), auth


def _candidate_command(package, authorization_sha256, launch_id, closure_sha256):
    candidate = {"schema": supervisor.admission.SCHEMA + "/synthetic_worker_candidate_v1",
                 "status": "candidate_complete", "launch_id": launch_id,
                 "authorization_sha256": authorization_sha256,
                 "admission_metadata_sha256": package["metadata_sha256"],
                 "implementation_closure_sha256": closure_sha256,
                 "parameter_updates": 0}
    literal = repr(json.dumps(candidate, sort_keys=True, separators=(",", ":")))
    code = ("import json, os; from pathlib import Path; "
            f"p=Path('worker_candidate.json'); raw={literal}.encode(); "
            "fd=os.open(p, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600); "
            "f=os.fdopen(fd,'wb'); f.write(raw); f.flush(); os.fsync(f.fileno()); f.close()")
    return [sys.executable, "-I", "-c", code]


def test_synthetic_outer_chain_claim_wait_closure_candidate_and_return(tmp_path):
    source_root, package, auth_path, auth_sha, auth = _synthetic_authorized_packet(tmp_path)
    (tmp_path / "runs").mkdir()
    closure_sha = supervisor._canonical_sha(package["implementation"]["files"])
    command = _candidate_command(package, auth_sha, auth["launch_id"], closure_sha)
    result = supervisor.run_synthetic_attempt(
        authorization_path=auth_path, authorization_sha256=auth_sha,
        package=package, project_root=source_root,
        attempt_root=tmp_path / "runs" / "attempt_001", command=command,
    )
    assert result["status"] == "candidate_complete"
    assert result["claim"]["attempt"] == 1
    assert result["claim"]["parameter_updates"] == 0
    assert result["implementation_pre_sha256"] == result["implementation_post_sha256"]
    assert result["wait"]["return_code"] == 0 and result["wait"]["group_reaped"] is True
    assert result["return"]["coverage"] == "covered"
    attempt = Path(result["attempt_root"])
    assert {p.name for p in attempt.iterdir()} == {
        "claim.json", "worker.stdout.log", "worker.stderr.log",
        "worker_candidate.json", "return_observation.json",
    }
    observation = json.loads((attempt / "return_observation.json").read_text())
    assert observation["publication"] == "pending_actual_readback"
    assert observation["binding_facts"]["parameter_updates"] == 0


def test_unsigned_authorization_is_rejected_before_attempt_claim_or_spawn(tmp_path):
    source_root, package, auth_path, auth_sha, auth = _synthetic_authorized_packet(tmp_path)
    auth["authorized"] = False
    auth_raw = (json.dumps(auth, sort_keys=True, separators=(",", ":")) + "\n").encode()
    auth_path.write_bytes(auth_raw)
    (tmp_path / "runs").mkdir()
    attempt = tmp_path / "runs" / "attempt_001"
    called = []

    def forbidden_spawn(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("unsigned auth must not spawn")

    with pytest.raises(ValueError, match="synthetic_authorization_rejected"):
        supervisor.run_synthetic_attempt(
            authorization_path=auth_path, authorization_sha256=hashlib.sha256(auth_raw).hexdigest(),
            package=package, project_root=source_root, attempt_root=attempt,
            command=[sys.executable, "-I", "-c", "pass"], popen=forbidden_spawn,
        )
    assert not attempt.exists() and called == []


def test_source_closure_change_during_child_blocks_return_coverage(tmp_path):
    source_root, package, auth_path, auth_sha, auth = _synthetic_authorized_packet(tmp_path)
    (tmp_path / "runs").mkdir()
    closure_sha = supervisor._canonical_sha(package["implementation"]["files"])
    command = _candidate_command(package, auth_sha, auth["launch_id"], closure_sha)
    source = source_root / "worker_contract.txt"
    original = source.read_bytes()

    def mutate_after_spawn(cmd, **kwargs):
        process = __import__("subprocess").Popen(cmd, **kwargs)
        source.write_text("synthetic implementation drift\n")
        return process

    with pytest.raises(ValueError, match="implementation_closure_drift"):
        supervisor.run_synthetic_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=source_root,
            attempt_root=tmp_path / "runs" / "attempt_001", command=command,
            popen=mutate_after_spawn,
        )
    assert original != source.read_bytes()
    attempt = tmp_path / "runs" / "attempt_001"
    assert (attempt / "claim.json").is_file()
    assert not (attempt / "return_observation.json").exists()
