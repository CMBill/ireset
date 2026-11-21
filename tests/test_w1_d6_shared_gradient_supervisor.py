import sys
import time
from dataclasses import replace
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
