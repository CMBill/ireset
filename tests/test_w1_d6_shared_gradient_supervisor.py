import sys
import time
from dataclasses import replace
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
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
    consumed = supervisor.verify_external_return(root, receipt, clock=lambda: final)
    assert receipt.observation["receipt_pre_sample"] <= consumed["final_sample"]
    assert consumed["coverage"] == "covered"
    with pytest.raises(ValueError, match="external_return_witness_missing"):
        supervisor.verify_external_return(root, None)
    (root / "return_observation.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="external_return_observation_drift"):
        supervisor.verify_external_return(root, receipt)


def test_final_sample_over_cap_leaves_only_non_consumable_pending_record(tmp_path):
    root, t0, wait = _supervise(tmp_path, "pass")
    registered = wait.cleaned_at - t0 + .01
    receipt = supervisor.seal_external_return(
        root, wait, t0=t0, registered_s=registered, work_deadline=t0 + 5,
        hard_deadline=t0 + 7, clock=lambda: t0 + registered / 2,
    )
    with pytest.raises(ValueError, match="return_observation_actual_coverage_failed"):
        supervisor.verify_external_return(root, receipt, clock=lambda: t0 + registered + .1)
    assert (root / "return_observation.json").is_file()
    import json

    pending = json.loads((root / "return_observation.json").read_text())
    assert pending["publication"] == "pending_actual_readback"
    assert "final_sample" not in pending
    with pytest.raises(ValueError, match="external_return_witness_missing"):
        supervisor.verify_external_return(root, None)


def test_final_return_sample_covers_second_receipt_readback(tmp_path, monkeypatch):
    root, t0, wait = _supervise(tmp_path, "pass")
    registered = wait.cleaned_at - t0 + 1
    receipt = supervisor.seal_external_return(
        root, wait, t0=t0, registered_s=registered, work_deadline=t0 + 5,
        hard_deadline=t0 + 7, clock=lambda: wait.cleaned_at + .1,
    )
    original_read = supervisor._read_regular
    state = {"readback_elapsed": 0.}

    def delayed_read(path, cap, **kwargs):
        value = original_read(path, cap, **kwargs)
        if Path(path).name == "return_observation.json":
            state["readback_elapsed"] = registered + .1
        return value

    monkeypatch.setattr(supervisor, "_read_regular", delayed_read)
    with pytest.raises(ValueError, match="return_observation_actual_coverage_failed"):
        supervisor.verify_external_return(
            root, receipt, clock=lambda: t0 + state["readback_elapsed"])


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
    code = (
        "import json, os; from pathlib import Path; "
        "s=json.loads(Path('worker_launch.json').read_text()); "
        "c={'schema':'w1_d6_shared_gradient_real_admission_v1/synthetic_worker_candidate_v1', "
        "'status':'candidate_complete', 'launch_id':s['launch_id'], "
        "'authorization_sha256':s['authorization_sha256'], "
        "'admission_metadata_sha256':s['admission_metadata_sha256'], "
        "'implementation_closure_sha256':s['implementation_closure_sha256'], "
        "'ledger_sha256':s['ledger_sha256'], 'parameter_updates':0}; "
        "raw=(json.dumps(c,sort_keys=True,separators=(',',':'))+'\\n').encode(); "
        "fd=os.open('worker_candidate.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); "
        "f=os.fdopen(fd,'wb'); f.write(raw); f.flush(); os.fsync(f.fileno()); f.close()"
    )
    worker_launch = {"argv": [sys.executable, "-I", "-c", code],
                     "env": {"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"},
                     "shell": False, "status": "synthetic_test_only"}
    package = {
        "schema": supervisor.admission.SCHEMA,
        "execution_mode": "synthetic_test_only",
        "executable": True,
        "metadata_sha256": "",
        "implementation": {"files": files},
        "worker_launch": worker_launch,
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
                     "historical_d6_outcome": package["historical"]["verified_outcome"],
                     "worker_launch": worker_launch},
    }
    auth_raw = (json.dumps(auth, sort_keys=True, separators=(",", ":")) + "\n").encode()
    auth_path = tmp_path / "authorization.json"
    auth_path.write_bytes(auth_raw)
    return source_root, package, auth_path, hashlib.sha256(auth_raw).hexdigest(), auth


def test_synthetic_outer_chain_claim_wait_closure_candidate_and_return(tmp_path):
    source_root, package, auth_path, auth_sha, auth = _synthetic_authorized_packet(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "ledger").mkdir()
    result = supervisor.run_synthetic_attempt(
        authorization_path=auth_path, authorization_sha256=auth_sha,
        package=package, project_root=source_root,
        attempt_root=tmp_path / "runs" / "attempt_001", ledger_root=tmp_path / "ledger",
    )
    assert result["status"] == "candidate_complete"
    assert result["claim"]["attempt"] == 1
    assert result["claim"]["parameter_updates"] == 0
    assert result["authorization_ledger"]["record"]["state"] == "consumed_before_worker_launch"
    assert result["implementation_pre_sha256"] == result["implementation_post_sha256"]
    assert result["wait"]["return_code"] == 0 and result["wait"]["group_reaped"] is True
    assert result["return"]["coverage"] == "covered"
    attempt = Path(result["attempt_root"])
    assert {p.name for p in attempt.iterdir()} == {
        "claim.json", "worker_launch.json", "worker.stdout.log", "worker.stderr.log",
        "worker_candidate.json", "return_observation.json",
    }
    observation = json.loads((attempt / "return_observation.json").read_text())
    assert observation["publication"] == "pending_actual_readback"
    assert observation["binding_facts"]["parameter_updates"] == 0
    ledger_dir = tmp_path / "ledger" / result["authorization_ledger"]["directory"]
    assert {path.name for path in ledger_dir.iterdir()} == {
        "consumed.json", "launch_started.json", "candidate_validated.json", "return_candidate.json",
    }


def test_synthetic_outer_chain_uses_injected_clock_for_final_return(tmp_path):
    source_root, package, auth_path, auth_sha, _ = _synthetic_authorized_packet(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "ledger").mkdir()
    result = supervisor.run_synthetic_attempt(
        authorization_path=auth_path, authorization_sha256=auth_sha,
        package=package, project_root=source_root,
        attempt_root=tmp_path / "runs" / "attempt_001", ledger_root=tmp_path / "ledger",
        clock=lambda: 100.0,
    )
    assert result["status"] == "candidate_complete"
    assert result["return"]["coverage"] == "covered"


def test_unsigned_authorization_is_rejected_before_attempt_claim_or_spawn(tmp_path):
    source_root, package, auth_path, auth_sha, auth = _synthetic_authorized_packet(tmp_path)
    auth["authorized"] = False
    auth_raw = (json.dumps(auth, sort_keys=True, separators=(",", ":")) + "\n").encode()
    auth_path.write_bytes(auth_raw)
    (tmp_path / "runs").mkdir()
    (tmp_path / "ledger").mkdir()
    attempt = tmp_path / "runs" / "attempt_001"
    called = []

    def forbidden_spawn(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("unsigned auth must not spawn")

    with pytest.raises(ValueError, match="synthetic_authorization_rejected"):
        supervisor.run_synthetic_attempt(
            authorization_path=auth_path, authorization_sha256=hashlib.sha256(auth_raw).hexdigest(),
            package=package, project_root=source_root, attempt_root=attempt,
            ledger_root=tmp_path / "ledger", popen=forbidden_spawn,
        )
    assert not attempt.exists() and called == []


def test_source_closure_change_during_child_blocks_return_coverage(tmp_path):
    source_root, package, auth_path, auth_sha, auth = _synthetic_authorized_packet(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "ledger").mkdir()
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
            attempt_root=tmp_path / "runs" / "attempt_001", ledger_root=tmp_path / "ledger",
            popen=mutate_after_spawn,
        )
    assert original != source.read_bytes()
    attempt = tmp_path / "runs" / "attempt_001"
    assert (attempt / "claim.json").is_file()
    assert not (attempt / "return_observation.json").exists()


def test_authorization_ledger_blocks_second_attempt_even_at_new_run_path(tmp_path):
    source_root, package, auth_path, auth_sha, _auth = _synthetic_authorized_packet(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "ledger").mkdir()
    first = supervisor.run_synthetic_attempt(
        authorization_path=auth_path, authorization_sha256=auth_sha,
        package=package, project_root=source_root,
        attempt_root=tmp_path / "runs" / "attempt_001", ledger_root=tmp_path / "ledger",
    )
    assert first["status"] == "candidate_complete"
    with pytest.raises(FileExistsError):
        supervisor.run_synthetic_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=source_root,
            attempt_root=tmp_path / "runs" / "attempt_002", ledger_root=tmp_path / "ledger",
        )
    assert not (tmp_path / "runs" / "attempt_002").exists()
    ledger_files = list((tmp_path / "ledger").iterdir())
    assert len(ledger_files) == 1


def test_authorization_ledger_atomic_claim_allows_only_one_concurrent_consumer(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    kwargs = {"authorization_sha256": "a" * 64, "launch_id": "b" * 64,
              "admission_sha256": "c" * 64, "implementation_sha256": "d" * 64,
              "t0": time.perf_counter()}

    def consume(_):
        try:
            return supervisor.consume_authorization_once(ledger, **kwargs)["sha256"]
        except FileExistsError:
            return "consumed_elsewhere"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(consume, range(2)))
    assert sum(value != "consumed_elsewhere" for value in results) == 1
    assert len(list(ledger.iterdir())) == 1
    marker = next(ledger.iterdir())
    assert {path.name for path in marker.iterdir()} == {"consumed.json"}


def test_worker_argv_tampering_is_rejected_before_claim_and_spawn(tmp_path):
    source_root, package, auth_path, auth_sha, _auth = _synthetic_authorized_packet(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "ledger").mkdir()
    package["worker_launch"]["argv"] = [sys.executable, "-I", "-c", "pass"]
    spawned = []

    def forbidden_spawn(*_args, **_kwargs):
        spawned.append(True)
        raise AssertionError("modified package must not spawn")

    with pytest.raises(ValueError, match="synthetic_packet_metadata_hash"):
        supervisor.run_synthetic_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=source_root,
            attempt_root=tmp_path / "runs" / "attempt_001", ledger_root=tmp_path / "ledger",
            popen=forbidden_spawn,
        )
    assert spawned == []
    assert list((tmp_path / "ledger").iterdir()) == []


def test_worker_environment_tampering_is_rejected_before_claim_and_spawn(tmp_path):
    source_root, package, auth_path, auth_sha, _auth = _synthetic_authorized_packet(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "ledger").mkdir()
    package["worker_launch"]["env"]["PYTHONPATH"] = "/tmp/untrusted"
    spawned = []

    def forbidden_spawn(*_args, **_kwargs):
        spawned.append(True)
        raise AssertionError("modified package must not spawn")

    with pytest.raises(ValueError, match="synthetic_packet_metadata_hash"):
        supervisor.run_synthetic_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=source_root,
            attempt_root=tmp_path / "runs" / "attempt_001", ledger_root=tmp_path / "ledger",
            popen=forbidden_spawn,
        )
    assert spawned == [] and list((tmp_path / "ledger").iterdir()) == []


def _real_protocol_fixture(tmp_path):
    project = tmp_path / "real_protocol_project"
    project.mkdir()
    script_dir = project / "scripts"
    script_dir.mkdir()
    worker_script = script_dir / "fake_authorized_worker.py"
    worker_script.write_text(
        "import argparse, json, os\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--launch-spec'); p.add_argument('--admission-snapshot'); p.add_argument('--authorization-snapshot'); p.add_argument('--authorization-sha256'); p.add_argument('--project-root'); a=p.parse_args()\n"
        "spec=json.loads(Path(a.launch_spec).read_text()); pkg=json.loads(Path(a.admission_snapshot).read_text())\n"
        "io=pkg['io_matrix']; cm=pkg['call_matrix']; reads=io['authorized_future_reads']; records={}\n"
        "for row in reads['d6_files']+reads['production_assets']+[reads['root_stats']]: records[row['path']]={'bytes':row['bytes'],'sha256':row['sha256']}\n"
        "artifacts={name:{'bytes':4,'sha256':'0'*64,'shape':[1],'dtype':'<f4'} for name in ([f'{r}_{i}.npy' for r in ('h','g') for i in range(7)]+[f'loss_{k}.npy' for k in ('pos','rot6d','joint','geodesic_rad2','vel','fk_m2')])}\n"
        "worker_io=io['worker_actual']; runtime=pkg['source']['d6_bindings']['runtime']\n"
        "runtime_report={'stage':'D6','kind':'production','device':'cuda:0','identity':runtime['identity'],'gpu':runtime['gpu'],'before':runtime['observed_before_setup'],'setup':{'applied':True,'problems':[]},'endpoint':{'ok':True}}\n"
        "worker_cm={k:cm[k] for k in ('schema','sources','setup_units','trainer_construction','numeric','forbidden')}\n"
        "worker_matrix={'source_file_reads':48,'source_file_hashes':48,'d6_npz_member_reads':60,'root_stats_npz_member_reads':2,'npz_archive_opens':4,'independent_artifact_reads':27,'independent_artifact_hashes':20,'artifact_files':artifacts,'artifact_io':{'npy_writes':20,'json_writes':8},'checkpoint_cpu_loads':1,'strict_model_loads':2,'source_records':records}\n"
        "result={'schema':'w1_d6_shared_gradient_real_worker_v1/worker_result',"
        "'status':'candidate_pending_external_return','launch_id':spec['launch_id'],"
        "'authorization_sha256':spec['authorization_sha256'],"
        "'admission_metadata_sha256':spec['admission_metadata_sha256'],"
        "'independent_review':{'passed':True},'parameter_updates':0,'worker_matrix_sha256':pkg['metadata_sha256'],"
        "'call_matrix':worker_cm,'worker_io':worker_io,'io_matrix':worker_matrix,"
        "'source_records':records,'artifact_io':{'npy_writes':20,'json_writes':8},"
        "'outcome':{'runtime_report':runtime_report,'runtime_sync_count':cm['setup_units']['runtime_syncs'],'files':artifacts}}\n"
        "raw=(json.dumps(result,sort_keys=True,separators=(',',':'))+'\\n').encode()\n"
        "fd=os.open(Path(spec['attempt_root'])/'worker_result.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\n"
        "f=os.fdopen(fd,'wb'); f.write(raw); f.flush(); os.fsync(f.fileno()); f.close()\n"
    )
    source_path = project / "contract.txt"
    source_path.write_text("real protocol test fixture only\n")
    implementation = {}
    for relative in ("contract.txt", "scripts/fake_authorized_worker.py"):
        raw = (project / relative).read_bytes()
        implementation[relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    launch = {
        "execution_mode": "real_worker_v1", "status": "reviewed_executable",
        "executable": sys.executable,
        "script_rel": "scripts/fake_authorized_worker.py",
        "cwd": "attempt_root",
        "env": {"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"},
        "argv_template": [sys.executable, "-I", str(worker_script), "--launch-spec", "{launch_spec}",
                          "--admission-snapshot", "{admission_snapshot}",
                          "--authorization-snapshot", "{authorization_snapshot}",
                          "--authorization-sha256", "{authorization_sha256}",
                          "--project-root", "{project_root}"],
    }
    source_records = []
    for i in range(6):
        source_records.append({"path": f"source/d6/{i}.json", "bytes": i + 1,
                               "sha256": f"{i + 1:064x}", "read": 1, "hash": 1})
    assets = sorted([{"path": f"assets/{i}.obj", "bytes": i + 10, "sha256": f"{i + 10:064x}",
                      "read": 1, "hash": 1} for i in range(41)], key=lambda row: row["path"])
    root_stats = {"path": "source/root_stats.npz", "bytes": 77, "sha256": "a" * 64,
                  "read": 1, "hash": 1, "npz_members": ["mean", "std"]}
    runtime = {"identity": {"python": "3.12.13", "torch": "2.14.0+cu126"},
               "gpu": {"uuid": "test", "capability": [8, 6], "total_memory": 24},
               "observed_before_setup": {"tf32": False}}
    call_matrix = json.loads(json.dumps(supervisor.admission.REAL_CALL_MATRIX))
    worker_actual = {"launch_metadata_reads": 3, "ledger_reads": 2,
                     "implementation_closure_hash_passes": 2, "source_file_reads": 48,
                     "source_file_hashes": 48, "d6_npz_member_reads": 60,
                     "root_stats_npz_member_reads": 2, "npz_archive_opens": 4,
                     "independent_artifact_reads": 27, "independent_artifact_hashes": 20,
                     "artifact_npy_writes": 20, "artifact_json_writes": 8,
                     "checkpoint_cpu_loads": 1, "strict_model_loads": 2,
                     "d6_directory_created": 1, "worker_result_writes": 1,
                     "worker_result_readbacks": 1}
    io_matrix = {"authorized_future_reads": {"d6_files": source_records,
                                              "root_stats": root_stats,
                                              "production_assets": assets},
                 "worker_actual": worker_actual,
                     "supervisor_actual": {
                         **{key: call_matrix["supervisor"][key] for key in
                            ("authorization_file_reads", "implementation_closure_hash_passes",
                             "worker_launches", "actual_wait_calls", "stdout_log_creates",
                             "stderr_log_creates", "worker_result_reads",
                             "return_observation_writes", "return_observation_readbacks",
                             "authorization_consumption_markers", "authorization_ledger_events", "attempts",
                             "process_group_created", "log_streams_drained", "launch_spec_writes",
                             "return_witnesses")},
                         "retries": call_matrix["supervisor"]["worker_retries"],
                         "group_cleanup_checks": 0}
                     }
    package = {
        "schema": supervisor.admission.SCHEMA,
        "state": "test_only_signed_contract_fixture",
        "execution_mode": "real_worker_v1", "executable": True,
        "metadata_sha256": "", "proposed_real_root": str(supervisor.admission.PROPOSED_REAL_ROOT),
        "authorization_consumption_ledger": str(supervisor.admission.AUTHORIZATION_LEDGER_REL),
        "implementation": {"files": implementation},
        "worker_launch": launch,
        "call_matrix": call_matrix,
        "io_matrix": io_matrix,
        "source": {"d6_bindings": {"runtime": runtime},
                   "selected_d6_files": [{"path": row["path"], "bytes": row["bytes"],
                                          "sha256": row["sha256"]} for row in source_records],
                   "root_stats": {"file": {key: root_stats[key] for key in ("path", "bytes", "sha256")}},
                   "production_assets": {row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
                                         for row in assets}},
        "historical": {"synthetic_ic_acceptance": {"bytes": 1, "sha256": "1" * 64},
                       "verified_outcome": {"bytes": 1, "sha256": "2" * 64}},
        "budgets": {"wall_cap_s": 300, "work_cap_s": 240, "attempts": 1,
                    "parameter_updates": 0},
        "status": {"real_worker": "implemented_and_reviewed_and_supervised"},
    }
    package_body = {key: value for key, value in package.items() if key != "metadata_sha256"}
    package["metadata_sha256"] = supervisor._canonical_sha(package_body)
    auth = {
        "schema": supervisor.admission.AUTH_SCHEMA,
        "cohort": "w1_d6_shared_gradient_step0_v1", "authorized": True,
        "template_only": False, "includes": ["single_step0_diagnostic"], "attempts": 1,
        "checklist": dict.fromkeys(supervisor.admission.CHECKLIST, True),
        "approval_reference": "temporary integration fixture",
        "launch_id": "e" * 64,
        "bindings": {"admission_metadata_sha256": package["metadata_sha256"],
                     "implementation_files": implementation,
                     "ic_acceptance": package["historical"]["synthetic_ic_acceptance"],
                     "historical_d6_outcome": package["historical"]["verified_outcome"],
                     "worker_launch": launch},
    }
    auth_raw = (json.dumps(auth, sort_keys=True, separators=(",", ":")) + "\n").encode()
    auth_path = tmp_path / "real_protocol_authorization.json"
    auth_path.write_bytes(auth_raw)
    run_parent = project / supervisor.admission.PROPOSED_REAL_ROOT.parent
    run_parent.mkdir(parents=True)
    ledger_root = project / supervisor.admission.AUTHORIZATION_LEDGER_REL
    ledger_root.mkdir(parents=True)
    return project, package, auth_path, hashlib.sha256(auth_raw).hexdigest(), ledger_root


def test_real_authorization_ledger_and_pinned_worker_launch_end_to_end_on_temp_tree(tmp_path):
    project, package, auth_path, auth_sha, ledger_root = _real_protocol_fixture(tmp_path)
    result = supervisor.run_authorized_attempt(
        authorization_path=auth_path, authorization_sha256=auth_sha,
        package=package, project_root=project, ledger_root=ledger_root,
    )
    assert result["status"] == "candidate_complete"
    assert result["parameter_updates"] == 0
    assert result["wait"]["return_code"] == 0 and result["wait"]["group_reaped"] is True
    assert result["return"]["coverage"] == "covered"


def test_formal_main_dispatches_only_bound_executable_packet_on_temp_tree(tmp_path, capsys):
    project, package, auth_path, auth_sha, ledger_root = _real_protocol_fixture(tmp_path)
    packet_path = project / "admission.json"
    packet_path.write_text(json.dumps(package, sort_keys=True, separators=(",", ":")))
    rc = supervisor.main([
        "--authorization", str(auth_path), "--authorization-sha256", auth_sha,
        "--admission", str(packet_path), "--project-root", str(project), "--run",
    ])
    result = json.loads(capsys.readouterr().out)
    assert rc == 0 and result["status"] == "candidate_complete"
    assert result["parameter_updates"] == 0
    assert result["return"]["coverage"] == "covered"
    ledger = ledger_root / auth_sha
    assert {path.name for path in ledger.iterdir()} == {
        "consumed.json", "launch_started.json", "candidate_validated.json", "return_candidate.json",
    }
    attempt = Path(result["attempt_root"])
    saved_spec = json.loads((attempt / "worker_launch.json").read_text())
    assert saved_spec["execution_mode"] == "real_worker_v1"
    assert saved_spec["authorization_ledger"]["sha256"] == result["authorization_ledger"]["sha256"]
    assert (attempt / "authorization_snapshot.json").read_bytes() == auth_path.read_bytes()
    assert json.loads((attempt / "authorization_snapshot.json").read_text())["authorized"] is True


def test_real_authorization_consumption_persists_after_worker_failure(tmp_path):
    project, package, auth_path, auth_sha, ledger_root = _real_protocol_fixture(tmp_path)
    worker_path = project / package["worker_launch"]["script_rel"]
    worker_path.write_text("raise SystemExit(9)\n")
    raw = worker_path.read_bytes()
    package["implementation"]["files"][package["worker_launch"]["script_rel"]] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    # The launch recipe is signed into the package/auth before the attempted run.
    package["metadata_sha256"] = ""
    body = {key: value for key, value in package.items() if key != "metadata_sha256"}
    package["metadata_sha256"] = supervisor._canonical_sha(body)
    auth = json.loads(auth_path.read_text())
    auth["bindings"]["implementation_files"] = package["implementation"]["files"]
    auth["bindings"]["admission_metadata_sha256"] = package["metadata_sha256"]
    auth["bindings"]["worker_launch"] = package["worker_launch"]
    auth_raw = (json.dumps(auth, sort_keys=True, separators=(",", ":")) + "\n").encode()
    auth_path.write_bytes(auth_raw)
    auth_sha = hashlib.sha256(auth_raw).hexdigest()
    with pytest.raises(ValueError, match="real_worker_failed_or_unclean"):
        supervisor.run_authorized_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=project, ledger_root=ledger_root,
        )
    assert (ledger_root / auth_sha / "consumed.json").is_file()
    with pytest.raises(ValueError, match="real_attempt_root_not_exclusive"):
        supervisor.run_authorized_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=project, ledger_root=ledger_root,
        )


def test_real_authorization_ledger_path_is_package_pinned(tmp_path):
    project, package, auth_path, auth_sha, ledger_root = _real_protocol_fixture(tmp_path)
    wrong_ledger = project / "other-consumption-ledger"
    wrong_ledger.mkdir()
    with pytest.raises(ValueError, match="real_authorization_ledger_root_not_frozen"):
        supervisor.run_authorized_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=project, ledger_root=wrong_ledger,
        )
    assert list(ledger_root.iterdir()) == []
    assert list(wrong_ledger.iterdir()) == []


@pytest.mark.parametrize("mutation", ["call", "io"])
def test_real_package_matrix_drift_is_rejected_before_authorization_or_spawn(tmp_path, mutation):
    project, package, auth_path, auth_sha, ledger_root = _real_protocol_fixture(tmp_path)
    if mutation == "call":
        package["call_matrix"]["numeric"]["fk_call"] += 1
    else:
        package["io_matrix"]["worker_actual"]["source_file_reads"] -= 1
    package["metadata_sha256"] = ""
    package["metadata_sha256"] = supervisor._canonical_sha(
        {key: value for key, value in package.items() if key != "metadata_sha256"})
    launched = []

    def forbidden_spawn(*_args, **_kwargs):
        launched.append(True)
        raise AssertionError("matrix drift must reject before worker launch")

    with pytest.raises(ValueError, match="real_admission_io_or_call_matrix_not_frozen"):
        supervisor.run_authorized_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=project, ledger_root=ledger_root,
            popen=forbidden_spawn,
        )
    assert launched == []
    assert list(ledger_root.iterdir()) == []


def test_real_supervisor_rejects_worker_reported_io_count_drift(tmp_path):
    project, package, auth_path, _auth_sha, ledger_root = _real_protocol_fixture(tmp_path)
    worker_path = project / package["worker_launch"]["script_rel"]
    source = worker_path.read_text()
    assert "worker_io=io['worker_actual']" in source
    worker_path.write_text(source.replace(
        "worker_io=io['worker_actual']",
        "worker_io=dict(io['worker_actual']); worker_io['source_file_reads']=47",
    ))
    raw = worker_path.read_bytes()
    package["implementation"]["files"][package["worker_launch"]["script_rel"]] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    package["metadata_sha256"] = supervisor._canonical_sha(
        {key: value for key, value in package.items() if key != "metadata_sha256"})
    auth = json.loads(auth_path.read_text())
    auth["bindings"]["implementation_files"] = package["implementation"]["files"]
    auth["bindings"]["admission_metadata_sha256"] = package["metadata_sha256"]
    auth["bindings"]["worker_launch"] = package["worker_launch"]
    auth_raw = (json.dumps(auth, sort_keys=True, separators=(",", ":")) + "\n").encode()
    auth_path.write_bytes(auth_raw)
    auth_sha = hashlib.sha256(auth_raw).hexdigest()

    with pytest.raises(ValueError, match="real_worker_io_call_counts_mismatch"):
        supervisor.run_authorized_attempt(
            authorization_path=auth_path, authorization_sha256=auth_sha,
            package=package, project_root=project, ledger_root=ledger_root,
        )
    assert (ledger_root / auth_sha / "consumed.json").is_file()
    assert not (ledger_root / auth_sha / "candidate_validated.json").exists()
