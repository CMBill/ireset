"""Fail-closed outer entry and process supervisor for shared-gradient work.

The process primitive is exercised only with synthetic children. The production
CLI remains refusal-only until an exact worker packet and return contract are
independently reviewed and frozen.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import stat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_ADMISSION_SCHEMA = "w1_d6_shared_gradient_real_admission_v1"
_admission_module = None


class _AdmissionProxy:
    def __getattr__(self, name):
        return getattr(_load_admission(), name)


admission = _AdmissionProxy()


def _load_admission():
    global _admission_module
    if _admission_module is None:
        from ireset.eval import w1_d6_shared_gradient_admission
        _admission_module = w1_d6_shared_gradient_admission
    return _admission_module


LOG_CAP = 256 << 10
TERM_GRACE_S = 1.0
GROUP_REAP_ROUNDS = 20
GROUP_REAP_INTERVAL_S = 0.05


@dataclass(frozen=True)
class ExternalWaitWitness:
    supervisor_pid: int
    pid: int
    process_group: int
    return_code: int
    started_at: float
    waited_at: float
    cleaned_at: float
    group_reaped: bool
    group_cleanup_checks: int
    actual_wait_calls: int
    timed_out: bool
    log_overflow: bool
    logs: dict
    _token: object


@dataclass(frozen=True)
class ExternalReturnWitness:
    observation: dict
    observation_sha256: str
    t0: float
    registered_s: float
    work_deadline: float
    hard_deadline: float
    _token: object


_RETURN_TOKEN = object()
_WAIT_TOKEN = object()


def _hash_file(path: Path, expected_identity: tuple[int, int]) -> dict:
    raw, identity = _read_regular(path, LOG_CAP, with_identity=True)
    if identity != expected_identity:
        raise ValueError("supervisor_file_identity_changed")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _read_regular(path: Path, cap: int, *, with_identity=False):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > cap:
            raise ValueError("supervisor_file_type_or_cap")
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                raise ValueError("supervisor_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != \
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("supervisor_file_changed_during_read")
        raw = b"".join(chunks)
        identity = (after.st_dev, after.st_ino)
        return (raw, identity) if with_identity else raw
    finally:
        os.close(fd)


def _real_directory(path: Path) -> Path:
    supplied = Path(path).absolute()
    resolved = supplied.resolve(strict=True)
    if resolved != supplied or not resolved.is_dir():
        raise ValueError("supervisor_directory_not_real")
    return resolved


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def supervise_process(command: list[str], *, output_root: Path, t0: float,
                      work_deadline: float, hard_deadline: float,
                      clock=time.perf_counter, popen=subprocess.Popen,
                      log_cap: int = LOG_CAP, term_grace_s: float = TERM_GRACE_S,
                      cwd: Path | None = None, env: dict[str, str] | None = None) -> ExternalWaitWitness:
    """Wait for one process and prove process-group cleanup with bounded logs."""
    root = _real_directory(output_root)
    if (not isinstance(command, list) or not command or
            not all(type(part) is str and part for part in command)):
        raise ValueError("worker_command_invalid")
    finite = lambda value: type(value) in (int, float) and value == value and abs(value) != float("inf")
    if not (finite(t0) and finite(work_deadline) and finite(hard_deadline)
            and finite(term_grace_s) and t0 <= work_deadline < hard_deadline
            and clock() < work_deadline):
        raise ValueError("worker_deadline_invalid")
    if type(log_cap) is not int or log_cap <= 0 or term_grace_s < 0:
        raise ValueError("worker_supervisor_limits_invalid")
    working_directory = _real_directory(cwd) if cwd is not None else None
    if env is not None and (type(env) is not dict or any(
            type(key) is not str or not key or type(value) is not str or "\x00" in key + value
            for key, value in env.items())):
        raise ValueError("worker_environment_invalid")

    log_paths = {"stdout": root / "worker.stdout.log", "stderr": root / "worker.stderr.log"}
    streams, identities = {}, {}
    for key, path in log_paths.items():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            os.close(fd)
            raise ValueError("supervisor_log_not_regular")
        identities[key] = (file_stat.st_dev, file_stat.st_ino)
        streams[key] = os.fdopen(fd, "wb")
    written = {key: 0 for key in streams}
    overflow = {key: False for key in streams}

    def drain(key, pipe):
        try:
            while True:
                chunk = pipe.read(65536)
                if not chunk:
                    break
                remaining = max(0, log_cap - written[key])
                if remaining:
                    streams[key].write(chunk[:remaining])
                    written[key] += min(remaining, len(chunk))
                if len(chunk) > remaining:
                    overflow[key] = True
        finally:
            pipe.close()

    process = None
    threads = []
    timed_out = False
    cleanup_checks = 0
    wait_calls = 0
    launch_started_at = None
    try:
        launch_started_at = clock()
        process = popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                        cwd=working_directory, env=env)
        pgid = process.pid
        threads = [threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
                   threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True)]
        for thread in threads:
            thread.start()
        remaining = max(0., work_deadline - clock())
        try:
            wait_calls += 1
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            _signal_group(pgid, signal.SIGTERM)
            try:
                wait_calls += 1
                process.wait(timeout=term_grace_s)
            except subprocess.TimeoutExpired:
                pass
            _signal_group(pgid, signal.SIGKILL)
            wait_calls += 1
            return_code = process.wait()
        waited_at = clock()

        _signal_group(pgid, signal.SIGTERM)
        for _ in range(GROUP_REAP_ROUNDS):
            cleanup_checks += 1
            if not _group_exists(pgid):
                break
            if clock() >= hard_deadline:
                break
            time.sleep(min(GROUP_REAP_INTERVAL_S, max(0., hard_deadline - clock())))
        if _group_exists(pgid):
            _signal_group(pgid, signal.SIGKILL)
        for thread in threads:
            thread.join(timeout=max(0., hard_deadline - clock()))
        group_reaped = not _group_exists(pgid) and all(not thread.is_alive() for thread in threads)
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        cleaned_at = clock()
        records = {key: _hash_file(path, identities[key]) for key, path in log_paths.items()}
        return ExternalWaitWitness(os.getpid(), process.pid, pgid, return_code, launch_started_at,
                                   waited_at, cleaned_at, group_reaped, cleanup_checks, wait_calls, timed_out,
                                   any(overflow.values()), records, _WAIT_TOKEN)
    except BaseException:
        if process is not None:
            _signal_group(process.pid, signal.SIGKILL)
            try:
                process.wait()
            except Exception:
                pass
        raise
    finally:
        for pipe_name in ("stdout", "stderr"):
            stream = streams.get(pipe_name)
            if stream and not stream.closed:
                stream.close()


def return_witness(witness: ExternalWaitWitness, *, t0: float, registered_s: float,
                   work_deadline: float, hard_deadline: float, final_sample: float) -> dict:
    """Validate process return, cleanup, log caps, and end-to-end coverage."""
    if not isinstance(witness, ExternalWaitWitness) or witness._token is not _WAIT_TOKEN:
        raise ValueError("return_witness_wait_record_untrusted")
    values = (t0, registered_s, work_deadline, hard_deadline, final_sample,
              witness.started_at, witness.waited_at, witness.cleaned_at)
    if not all(type(value) in (int, float) and value == value and abs(value) != float("inf")
               for value in values):
        raise ValueError("return_witness_nonfinite_time")
    if not (t0 <= witness.started_at <= witness.waited_at <= witness.cleaned_at <= final_sample
            <= t0 + registered_s <= hard_deadline and t0 <= work_deadline < hard_deadline
            and registered_s >= 0 and witness.cleaned_at < work_deadline):
        raise ValueError("return_witness_time_coverage")
    if witness.pid != witness.process_group or witness.pid == witness.supervisor_pid \
            or witness.return_code != 0 or witness.timed_out or witness.actual_wait_calls != 1:
        raise ValueError("return_witness_worker_not_successful")
    if not witness.group_reaped or witness.log_overflow:
        raise ValueError("return_witness_cleanup_or_log_failure")
    if set(witness.logs) != {"stdout", "stderr"} or any(
            type(rec) is not dict or type(rec.get("bytes")) is not int
            or not 0 <= rec["bytes"] <= LOG_CAP
            or not isinstance(rec.get("sha256"), str) or len(rec["sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in rec["sha256"])
            for rec in witness.logs.values()):
        raise ValueError("return_witness_log_contract")
    return {"schema": _load_admission().SCHEMA + "/external_return_witness_v1",
            "supervisor_pid": witness.supervisor_pid, "pid": witness.pid,
             "process_group": witness.process_group,
             "return_code": witness.return_code, "started_at": witness.started_at,
             "waited_at": witness.waited_at, "cleaned_at": witness.cleaned_at,
             "group_cleanup_checks": witness.group_cleanup_checks,
             "actual_wait_calls": witness.actual_wait_calls,
            "final_sample": final_sample, "registered_s": registered_s,
            "wall_cap_s": hard_deadline - t0, "group_reaped": True,
            "timed_out": False, "log_overflow": False, "logs": witness.logs,
            "coverage": "covered"}


def seal_external_return(root: Path, wait: ExternalWaitWitness, *, t0: float,
                         registered_s: float, work_deadline: float,
                         hard_deadline: float, clock=time.perf_counter,
                         binding_facts: dict | None = None) -> ExternalReturnWitness:
    """Publish/readback the pending proof, then take the final no-I/O sample."""
    root = _real_directory(root)
    candidate = return_witness(wait, t0=t0, registered_s=registered_s,
                               work_deadline=work_deadline, hard_deadline=hard_deadline,
                               final_sample=wait.cleaned_at)
    observation = {key: value for key, value in candidate.items() if key != "final_sample"}
    observation.update({"publication": "pending_actual_readback", "nonce": secrets.token_hex(16),
                        "t0": t0, "work_deadline": work_deadline,
                        "hard_deadline": hard_deadline, "receipt_pre_sample": clock()})
    if binding_facts is not None:
        if type(binding_facts) is not dict:
            raise ValueError("return_binding_facts_invalid")
        observation["binding_facts"] = binding_facts
    raw = (json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path = Path(root) / "return_observation.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    receipt_stat = os.fstat(fd)
    if not stat.S_ISREG(receipt_stat.st_mode) or receipt_stat.st_nlink != 1:
        os.close(fd)
        raise ValueError("return_observation_not_regular")
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    directory_fd = os.open(Path(root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    check, receipt_identity = _read_regular(path, 64 << 10, with_identity=True)
    if receipt_identity != (receipt_stat.st_dev, receipt_stat.st_ino):
        raise ValueError("return_observation_identity_changed")
    if check != raw:
        raise ValueError("return_observation_readback_mismatch")
    rec = {"bytes": len(check), "sha256": hashlib.sha256(check).hexdigest()}
    return ExternalReturnWitness(observation, rec["sha256"], t0,
                                registered_s, work_deadline, hard_deadline, _RETURN_TOKEN)


def verify_external_return(root: Path, witness: ExternalReturnWitness, *, clock=time.perf_counter) -> dict:
    """Read back the pending receipt, then take the final no-I/O clock sample."""
    if not isinstance(witness, ExternalReturnWitness) or witness._token is not _RETURN_TOKEN:
        raise ValueError("external_return_witness_missing")
    path = _real_directory(root) / "return_observation.json"
    raw = _read_regular(path, 64 << 10)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != witness.observation_sha256:
        raise ValueError("external_return_observation_drift")
    observation = json.loads(raw)
    receipt_pre_sample = observation.get("receipt_pre_sample")
    if (observation != witness.observation or observation.get("publication") != "pending_actual_readback"
            or observation.get("t0") != witness.t0
            or observation.get("registered_s") != witness.registered_s
            or observation.get("hard_deadline") != witness.hard_deadline
            or type(receipt_pre_sample) not in (int, float)
            or receipt_pre_sample < witness.t0):
        raise ValueError("external_return_observation_uncovered")
    final_sample = clock()
    wait_cleaned_at = observation.get("cleaned_at")
    finite_final = type(final_sample) in (int, float) and final_sample == final_sample \
        and abs(final_sample) != float("inf")
    if (not finite_final or final_sample < receipt_pre_sample
            or final_sample - witness.t0 > witness.registered_s
            or final_sample > witness.hard_deadline
            or type(wait_cleaned_at) not in (int, float)
            or wait_cleaned_at >= witness.work_deadline):
        raise ValueError("return_observation_actual_coverage_failed")
    return {"coverage": "covered", "observation_sha256": digest,
            "final_sample": final_sample, "registered_s": witness.registered_s}


def _read_bounded_regular(path: Path, cap: int) -> bytes:
    return _read_regular(path, cap)


def _canonical_sha(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _implementation_closure(project_root: Path, files: dict) -> dict:
    if type(files) is not dict or not files:
        raise ValueError("implementation_closure_empty")
    closure = {}
    for rel, expected in sorted(files.items()):
        relative = Path(rel)
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or type(expected) is not dict or type(expected.get("bytes")) is not int
                or type(expected.get("sha256")) is not str or len(expected["sha256"]) != 64):
            raise ValueError("implementation_closure_record_invalid")
        raw = _read_bounded_regular(Path(project_root) / relative, expected["bytes"] + 1)
        record = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if record != {"bytes": expected["bytes"], "sha256": expected["sha256"]}:
            raise ValueError("implementation_closure_drift:" + rel)
        closure[rel] = record
    return closure


def _write_exclusive_json(root: Path, name: str, doc: dict) -> dict:
    raw = (json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path = root / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError("supervisor_json_not_regular")
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    observed = _read_regular(path, 1 << 20)
    if observed != raw:
        raise ValueError("supervisor_json_readback_mismatch")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_exclusive_bytes(root: Path, name: str, raw: bytes) -> dict:
    if type(raw) is not bytes or not name or Path(name).name != name:
        raise ValueError("supervisor_bytes_record_invalid")
    path = root / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError("supervisor_bytes_not_regular")
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    observed = _read_regular(path, len(raw) + 1)
    if observed != raw:
        raise ValueError("supervisor_bytes_readback_mismatch")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def consume_authorization_once(ledger_root: Path, *, authorization_sha256: str,
                              launch_id: str, admission_sha256: str,
                              implementation_sha256: str, t0: float) -> dict:
    """Atomically consume one authorization digest; failures remain consumed."""
    ledger_root = _real_directory(ledger_root)
    is_sha = lambda value: isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)
    if not all(is_sha(value) for value in (authorization_sha256, launch_id,
                                            admission_sha256, implementation_sha256)):
        raise ValueError("authorization_ledger_binding_invalid")
    if type(t0) not in (int, float) or t0 != t0 or abs(t0) == float("inf"):
        raise ValueError("authorization_ledger_time_invalid")
    name = authorization_sha256
    receipt = {"schema": _load_admission().SCHEMA + "/authorization_consumption_v1",
               "state": "consumed_before_worker_launch", "authorization_sha256": authorization_sha256,
               "launch_id": launch_id, "admission_metadata_sha256": admission_sha256,
               "implementation_closure_sha256": implementation_sha256, "t0": t0,
               "attempts": 1, "retries": 0, "parameter_updates": 0}
    directory = ledger_root / name
    directory.mkdir(mode=0o700, exist_ok=False)
    dirfd = os.open(ledger_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    record = _write_exclusive_json(directory, "consumed.json", receipt)
    return {"path": name + "/consumed.json", "directory": name,
            "record": receipt, **record}


def _record_authorization_event(ledger_root: Path, ledger: dict, name: str, event: dict) -> dict:
    if name not in {"launch_started.json", "candidate_validated.json", "return_candidate.json",
                    "failed.json"}:
        raise ValueError("authorization_ledger_event_name")
    directory = _real_directory(Path(ledger_root) / ledger.get("directory", ""))
    consumed = _read_regular(directory / "consumed.json", 64 << 10)
    if hashlib.sha256(consumed).hexdigest() != ledger.get("sha256"):
        raise ValueError("authorization_ledger_consumption_drift")
    record = _write_exclusive_json(directory, name, event)
    return {"path": ledger["directory"] + "/" + name, **record}


def _read_authorization(authorization_path: Path, authorization_sha256: str) -> tuple[bytes, dict]:
    raw = _read_bounded_regular(Path(authorization_path), 1 << 20)
    if hashlib.sha256(raw).hexdigest() != authorization_sha256:
        raise ValueError("authorization_sha256_mismatch")
    try:
        auth = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authorization_json_invalid") from exc
    if type(auth) is not dict:
        raise ValueError("authorization_json_not_object")
    return raw, auth


def _approved_worker_argv(package: dict, launch_spec_path: Path,
                          admission_path: Path, authorization_path: Path,
                          authorization_sha256: str, project_root: Path) -> list[str]:
    worker = package.get("worker_launch")
    if type(worker) is not dict:
        raise ValueError("authorized_worker_launcher_not_frozen")
    template = worker.get("argv_template")
    script_rel = worker.get("script_rel")
    script = Path(script_rel) if isinstance(script_rel, str) else Path("/")
    implementation_files = package.get("implementation", {}).get("files", {})
    substitutions = {"{launch_spec}": str(launch_spec_path),
                     "{admission_snapshot}": str(admission_path),
                     "{authorization_snapshot}": str(authorization_path),
                     "{authorization_sha256}": authorization_sha256,
                     "{project_root}": str(project_root)}
    if (worker.get("status") != "reviewed_executable" or worker.get("execution_mode") != "real_worker_v1"
            or worker.get("cwd") != "attempt_root"
            or type(template) is not list or not template
            or not all(type(part) is str and part for part in template)
            or script.is_absolute() or ".." in script.parts or script_rel not in implementation_files
            or len(template) < 3 or template[1] != "-I"
            or template[2] != str(Path(project_root) / script)
            or not Path(template[0]).is_absolute()
            or any(template.count(token) != 1 for token in substitutions)):
        raise ValueError("authorized_worker_launcher_not_frozen")
    command = []
    seen = set()
    for part in template:
        if part in substitutions:
            seen.add(part)
            command.append(substitutions[part])
        elif "{" in part or "}" in part:
            raise ValueError("authorized_worker_launcher_placeholder_invalid")
        else:
            command.append(part)
    if seen != set(substitutions) or command[0] != package.get("worker_launch", {}).get("executable"):
        raise ValueError("authorized_worker_launcher_binding_invalid")
    return command


def _approved_worker_environment(package: dict) -> dict[str, str]:
    worker = package.get("worker_launch")
    env = worker.get("env") if type(worker) is dict else None
    if type(env) is not dict or any(type(key) is not str or not key or type(value) is not str
                                    or "\x00" in key + value
                                    for key, value in env.items()):
        raise ValueError("authorized_worker_environment_not_frozen")
    return dict(env)


def run_authorized_attempt(*, authorization_path: Path, authorization_sha256: str,
                           package: dict, project_root: Path,
                           ledger_root: Path, clock=time.perf_counter,
                           popen=subprocess.Popen) -> dict:
    """Run only a separately approved, package-pinned worker exactly once."""
    t0 = clock()
    def outer_check(point):
        if type(t0) not in (int, float) or t0 != t0 or abs(t0) == float("inf"):
            raise ValueError("real_execution_clock_invalid")
        if clock() - t0 >= 240.:
            raise TimeoutError("real_supervisor_work_deadline:" + point)

    outer_check("project_root_resolve:pre")
    project_root = _real_directory(project_root)
    outer_check("project_root_resolve:post")
    if type(package) is not dict or package.get("schema") != _ADMISSION_SCHEMA:
        raise ValueError("real_admission_package_invalid")
    package_body = {key: value for key, value in package.items() if key != "metadata_sha256"}
    if (package.get("executable") is not True
            or package.get("metadata_sha256") != _canonical_sha(package_body)
            or package.get("status", {}).get("real_worker") != "implemented_and_reviewed_and_supervised"):
        raise ValueError("real_admission_not_executable")
    outer_check("authorization_read:pre")
    auth_raw, auth = _read_authorization(authorization_path, authorization_sha256)
    outer_check("authorization_read:post")
    outer_check("implementation_closure_pre:pre")
    pre = _implementation_closure(project_root, package.get("implementation", {}).get("files"))
    outer_check("implementation_closure_pre:post")
    admission = _load_admission()
    if (admission.real_call_matrix_problems(package.get("call_matrix"))
            or admission.real_io_matrix_problems(package)):
        raise ValueError("real_admission_io_or_call_matrix_not_frozen")
    problems = admission.authorization_problems(auth, package)
    if problems:
        raise ValueError("real_authorization_rejected:" + ",".join(problems))
    outer_check("authorization_validation:post")
    launch_id = auth.get("launch_id")
    if (not isinstance(launch_id, str) or len(launch_id) != 64
            or any(ch not in "0123456789abcdef" for ch in launch_id)):
        raise ValueError("real_authorization_launch_id_invalid")

    proposed = Path(package.get("proposed_real_root", ""))
    if (proposed.is_absolute() or ".." in proposed.parts or not proposed.parts
            or str(project_root / proposed) != str(project_root / admission.PROPOSED_REAL_ROOT)):
        raise ValueError("real_attempt_root_not_frozen")
    attempt_root = project_root / proposed
    parent = _real_directory(attempt_root.parent)
    if parent != attempt_root.parent or attempt_root.exists():
        raise ValueError("real_attempt_root_not_exclusive")
    ledger_root = _real_directory(ledger_root)
    ledger_rel = Path(package.get("authorization_consumption_ledger", ""))
    if (ledger_rel.is_absolute() or ".." in ledger_rel.parts or not ledger_rel.parts
            or ledger_root != project_root / ledger_rel):
        raise ValueError("real_authorization_ledger_root_not_frozen")
    budgets = package.get("budgets")
    if (type(t0) not in (int, float) or t0 != t0 or abs(t0) == float("inf")
            or type(budgets) is not dict
            or type(budgets.get("work_cap_s")) is not int or budgets["work_cap_s"] != 240
            or type(budgets.get("wall_cap_s")) is not int or budgets["wall_cap_s"] != 300
            or type(budgets.get("attempts")) is not int or budgets["attempts"] != 1
            or type(budgets.get("parameter_updates")) is not int or budgets["parameter_updates"] != 0):
        raise ValueError("real_execution_budget_not_frozen")
    work_deadline, hard_deadline = t0 + 240., t0 + 300.
    ledger = consume_authorization_once(
        ledger_root, authorization_sha256=authorization_sha256, launch_id=launch_id,
        admission_sha256=package["metadata_sha256"], implementation_sha256=_canonical_sha(pre), t0=t0,
    )
    outer_check("authorization_consumption_write:post")
    attempt_root.mkdir(mode=0o700, exist_ok=False)
    attempt = _real_directory(attempt_root)
    package_path = attempt / "admission_snapshot.json"
    auth_path = attempt / "authorization_snapshot.json"
    package_record = _write_exclusive_json(attempt, package_path.name, package)
    outer_check("admission_snapshot_write:post")
    auth_record = _write_exclusive_bytes(attempt, auth_path.name, auth_raw)
    outer_check("authorization_snapshot_write:post")
    launch_spec = {"schema": admission.SCHEMA + "/worker_launch_spec_v1",
                   "execution_mode": "real_worker_v1", "launch_id": launch_id,
                   "authorization_sha256": authorization_sha256,
                   "authorization_record": auth_record,
                   "admission_metadata_sha256": package["metadata_sha256"],
                   "admission_record": package_record,
                   "implementation_pre": pre,
                   "implementation_pre_sha256": _canonical_sha(pre),
                   "authorization_ledger": {"path": ledger["path"], "sha256": ledger["sha256"]},
                   "attempt_root": str(attempt), "project_root": str(project_root), "t0": t0,
                   "work_deadline": work_deadline, "hard_deadline": hard_deadline,
                   "parameter_updates": 0}
    launch_spec_record = _write_exclusive_json(attempt, "worker_launch.json", launch_spec)
    outer_check("launch_spec_write:post")
    command = _approved_worker_argv(package, attempt / "worker_launch.json", package_path,
                                    auth_path, authorization_sha256, project_root)
    worker_env = _approved_worker_environment(package)
    launch_event = _record_authorization_event(
        ledger_root, ledger, "launch_started.json",
        {"schema": admission.SCHEMA + "/authorization_event_v1", "event": "launch_started",
         "launch_id": launch_id, "attempt_root": str(attempt),
         "launch_spec_sha256": launch_spec_record["sha256"], "parameter_updates": 0},
    )
    outer_check("launch_event_write:post")
    wait = supervise_process(command, output_root=attempt, t0=t0, work_deadline=work_deadline,
                             hard_deadline=hard_deadline, clock=clock, popen=popen,
                             cwd=attempt, env=worker_env)
    outer_check("worker_wait_and_cleanup:post")
    if wait.return_code != 0 or wait.timed_out or not wait.group_reaped or wait.log_overflow:
        failure_record = None
        failure_path = attempt / "worker_failure.json"
        try:
            if failure_path.exists():
                failure_raw = _read_regular(failure_path, 2 << 20)
                failure_record = {"bytes": len(failure_raw),
                                  "sha256": hashlib.sha256(failure_raw).hexdigest()}
        except BaseException:
            failure_record = None
        try:
            _record_authorization_event(
                ledger_root, ledger, "failed.json",
                {"schema": admission.SCHEMA + "/authorization_event_v1",
                 "event": "failed", "launch_id": launch_id,
                 "return_code": wait.return_code, "timed_out": wait.timed_out,
                 "group_reaped": wait.group_reaped, "log_overflow": wait.log_overflow,
                 "worker_failure": failure_record, "parameter_updates": 0})
        except BaseException:
            pass
        raise ValueError("real_worker_failed_or_unclean")
    result_raw = _read_regular(attempt / "worker_result.json", 2 << 20)
    outer_check("worker_result_read:post")
    try:
        worker_result = json.loads(result_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("real_worker_result_invalid") from exc
    if (type(worker_result) is not dict
            or worker_result.get("schema") != "w1_d6_shared_gradient_real_worker_v1/worker_result"
            or worker_result.get("status") != "candidate_pending_external_return"
            or worker_result.get("launch_id") != launch_id
            or worker_result.get("authorization_sha256") != authorization_sha256
            or worker_result.get("admission_metadata_sha256") != package["metadata_sha256"]
            or type(worker_result.get("parameter_updates")) is not int
            or worker_result.get("parameter_updates") != 0
            or type(worker_result.get("independent_review")) is not dict
            or worker_result["independent_review"].get("passed") is not True):
        raise ValueError("real_worker_result_binding_or_review")
    worker_matrix = package["call_matrix"]
    expected_worker_matrix = {key: worker_matrix[key] for key in
                              ("schema", "sources", "setup_units", "trainer_construction", "numeric", "forbidden")}
    if worker_result.get("call_matrix") != expected_worker_matrix:
        raise ValueError("real_worker_call_matrix_mismatch")
    if worker_result.get("worker_matrix_sha256") != package["metadata_sha256"]:
        raise ValueError("real_worker_matrix_binding_mismatch")
    if worker_result.get("worker_io") != package["io_matrix"].get("worker_actual"):
        raise ValueError("real_worker_io_call_counts_mismatch")
    runtime_report = worker_result.get("outcome", {}).get("runtime_report")
    expected_runtime = package.get("source", {}).get("d6_bindings", {}).get("runtime")
    if (type(runtime_report) is not dict or type(expected_runtime) is not dict
            or runtime_report.get("stage") != "D6" or runtime_report.get("kind") != "production"
            or runtime_report.get("device") != "cuda:0"
            or runtime_report.get("identity") != expected_runtime.get("identity")
            or runtime_report.get("gpu") != expected_runtime.get("gpu")
            or runtime_report.get("before") != expected_runtime.get("observed_before_setup")
            or runtime_report.get("setup", {}).get("applied") is not True
            or runtime_report.get("setup", {}).get("problems") != []
            or runtime_report.get("endpoint") is None
            or worker_result.get("outcome", {}).get("runtime_sync_count") !=
               worker_matrix["setup_units"].get("runtime_syncs")):
        raise ValueError("real_worker_runtime_matrix_mismatch")
    io_actual = worker_result.get("io_matrix")
    expected_source_records = {
        row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for row in package["io_matrix"]["authorized_future_reads"]["d6_files"]
    }
    stats = package["io_matrix"]["authorized_future_reads"]["root_stats"]
    expected_source_records[stats["path"]] = {"bytes": stats["bytes"], "sha256": stats["sha256"]}
    for row in package["io_matrix"]["authorized_future_reads"]["production_assets"]:
        expected_source_records[row["path"]] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    if (type(io_actual) is not dict or io_actual.get("source_records") != expected_source_records
            or worker_result.get("source_records") != io_actual.get("source_records")
            or io_actual.get("source_file_reads") != 48 or io_actual.get("source_file_hashes") != 48
            or io_actual.get("d6_npz_member_reads") != 60
            or io_actual.get("root_stats_npz_member_reads") != 2
                or io_actual.get("npz_archive_opens") != 4
                or io_actual.get("independent_artifact_reads") != 27
            or io_actual.get("independent_artifact_hashes") != 20
            or io_actual.get("checkpoint_cpu_loads") != 1
                or io_actual.get("strict_model_loads") != 2
                or io_actual.get("artifact_io") != {"npy_writes": 20, "json_writes": 8}
            or worker_result.get("artifact_io") != io_actual.get("artifact_io")
            or worker_result.get("outcome", {}).get("files") != io_actual.get("artifact_files")
                or type(io_actual.get("artifact_files")) is not dict
                or set(io_actual["artifact_files"]) != (
                    {f"{role}_{i}.npy" for role in ("h", "g") for i in range(7)}
                    | {f"loss_{name}.npy" for name in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2")}
                )):
        raise ValueError("real_worker_io_matrix_mismatch")
    supervisor_expected = worker_matrix.get("supervisor")
    supervisor_io_expected = package["io_matrix"].get("supervisor_actual")
    supervisor_actual = {
        "authorization_file_reads": 1,
        "implementation_closure_hash_passes": 2,
        "worker_launches": 1,
        "worker_retries": 0,
        "actual_wait_calls": wait.actual_wait_calls,
        "stdout_log_creates": int("stdout" in wait.logs),
        "stderr_log_creates": int("stderr" in wait.logs),
        "worker_result_reads": 1,
        "return_observation_writes": 1,
        "return_observation_readbacks": 2,
        "authorization_consumption_markers": 1,
        "authorization_ledger_events": 3,
        "attempts": 1,
        "process_group_created": int(wait.process_group == wait.pid),
        "group_cleanup_checks": wait.group_cleanup_checks,
        "log_streams_drained": len(wait.logs),
        "launch_spec_writes": 1,
        "return_witnesses": 1,
    }
    supervisor_io_actual = {key: value for key, value in supervisor_actual.items()
                            if key != "worker_retries"}
    supervisor_io_actual["retries"] = supervisor_actual["worker_retries"]
    call_actual_fixed = {key: value for key, value in supervisor_actual.items()
                         if key != "group_cleanup_checks"}
    call_expected_fixed = {key: value for key, value in supervisor_expected.items()
                           if key != "group_cleanup_checks_max"}
    io_actual_fixed = {key: value for key, value in supervisor_io_actual.items()
                       if key != "group_cleanup_checks"}
    io_expected_fixed = {key: value for key, value in supervisor_io_expected.items()
                         if key != "group_cleanup_checks"}
    if (call_actual_fixed != call_expected_fixed or io_actual_fixed != io_expected_fixed
            or supervisor_io_actual.get("group_cleanup_checks") != supervisor_actual["group_cleanup_checks"]
            or supervisor_actual["group_cleanup_checks"] > supervisor_expected["group_cleanup_checks_max"]):
        raise ValueError("real_supervisor_call_or_io_matrix_mismatch")
    post = _implementation_closure(project_root, package["implementation"]["files"])
    if post != pre:
        raise ValueError("implementation_closure_changed_during_worker")
    result_record = {"bytes": len(result_raw), "sha256": hashlib.sha256(result_raw).hexdigest()}
    candidate_event = _record_authorization_event(
        ledger_root, ledger, "candidate_validated.json",
        {"schema": admission.SCHEMA + "/authorization_event_v1", "event": "candidate_validated",
         "launch_id": launch_id, "worker_result_sha256": result_record["sha256"],
         "parameter_updates": 0},
    )
    wait_record = return_witness(wait, t0=t0, registered_s=300., work_deadline=work_deadline,
                                 hard_deadline=hard_deadline, final_sample=wait.cleaned_at)
    return_candidate_event = _record_authorization_event(
        ledger_root, ledger, "return_candidate.json",
        {"schema": admission.SCHEMA + "/authorization_event_v1", "event": "return_candidate",
         "launch_id": launch_id, "implementation_closure_sha256": _canonical_sha(post),
         "worker_result_sha256": result_record["sha256"], "parameter_updates": 0},
    )
    binding_facts = {"authorization_ledger": {"path": ledger["path"], "sha256": ledger["sha256"]},
                     "authorization_ledger_events": [launch_event, candidate_event, return_candidate_event],
                     "launch_spec": launch_spec_record, "worker_result": result_record,
                     "implementation_pre_sha256": _canonical_sha(pre),
                     "implementation_post_sha256": _canonical_sha(post),
                     "call_matrix": expected_worker_matrix,
                     "supervisor_call_matrix": supervisor_actual,
                     "io_matrix": {key: value for key, value in io_actual.items() if key != "artifact_files"},
                     "wait_witness_sha256": _canonical_sha(wait_record), "parameter_updates": 0}
    receipt = seal_external_return(attempt, wait, t0=t0, registered_s=300.,
                                   work_deadline=work_deadline, hard_deadline=hard_deadline,
                                   clock=clock, binding_facts=binding_facts)
    outer_check("return_receipt_seal:post")
    consumed = verify_external_return(attempt, receipt, clock=clock)
    if consumed.get("coverage") != "covered" or receipt.observation.get("binding_facts") != binding_facts:
        raise ValueError("real_external_return_binding_failed")
    return {"status": "candidate_complete", "attempt_root": str(attempt),
            "authorization_ledger": ledger, "launch_spec": launch_spec,
            "authorization_ledger_events": [launch_event, candidate_event, return_candidate_event],
            "worker_result": worker_result, "wait": wait_record,
            "return": consumed, "parameter_updates": 0}


def run_synthetic_attempt(*, authorization_path: Path, authorization_sha256: str,
                          package: dict, project_root: Path, attempt_root: Path,
                          ledger_root: Path, work_cap_s: float = 5.0,
                          hard_cap_s: float = 10.0, clock=time.perf_counter,
                          popen=subprocess.Popen) -> dict:
    """Exercise the complete outer contract with an explicitly synthetic packet."""
    project_root = _real_directory(project_root)
    supplied_attempt = Path(attempt_root).absolute()
    parent = _real_directory(supplied_attempt.parent)
    if supplied_attempt.parent != parent or supplied_attempt.exists():
        raise ValueError("synthetic_attempt_root_not_exclusive")
    if type(package) is not dict or package.get("execution_mode") != "synthetic_test_only":
        raise ValueError("synthetic_packet_required")
    body = {key: value for key, value in package.items() if key != "metadata_sha256"}
    if package.get("metadata_sha256") != _canonical_sha(body):
        raise ValueError("synthetic_packet_metadata_hash")
    if package.get("executable") is not True:
        raise ValueError("synthetic_packet_not_executable")
    if not isinstance(authorization_path, Path):
        raise ValueError("authorization_path_invalid")
    auth_raw = _read_bounded_regular(authorization_path, 1 << 20)
    if hashlib.sha256(auth_raw).hexdigest() != authorization_sha256:
        raise ValueError("authorization_sha256_mismatch")
    try:
        auth = json.loads(auth_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authorization_json_invalid") from exc
    admission_module = _load_admission()
    if type(auth) is not dict or admission_module.authorization_problems(auth, package):
        raise ValueError("synthetic_authorization_rejected")
    admission = admission_module

    t0 = clock()
    if (type(t0) not in (float, int) or t0 != t0 or abs(t0) == float("inf")
            or type(work_cap_s) not in (float, int) or work_cap_s <= 0
            or type(hard_cap_s) not in (float, int) or hard_cap_s <= work_cap_s):
        raise ValueError("synthetic_budget_invalid")
    hard_deadline = t0 + hard_cap_s
    work_deadline = t0 + work_cap_s
    pre = _implementation_closure(project_root, package.get("implementation", {}).get("files"))

    worker_launch = package.get("worker_launch")
    command = worker_launch.get("argv") if type(worker_launch) is dict else None
    worker_env = worker_launch.get("env") if type(worker_launch) is dict else None
    if (type(command) is not list or not command or not all(type(part) is str and part for part in command)
            or type(worker_env) is not dict or any(type(k) is not str or type(v) is not str
                                                   for k, v in worker_env.items())
            or worker_launch.get("shell") is not False or worker_launch.get("status") != "synthetic_test_only"):
        raise ValueError("synthetic_worker_command_not_pinned")
    ledger = consume_authorization_once(
        ledger_root, authorization_sha256=authorization_sha256, launch_id=auth["launch_id"],
        admission_sha256=package["metadata_sha256"], implementation_sha256=_canonical_sha(pre), t0=t0,
    )

    supplied_attempt.mkdir(mode=0o700, exist_ok=False)
    attempt = _real_directory(supplied_attempt)
    launch_id = auth.get("launch_id")
    if not isinstance(launch_id, str) or len(launch_id) != 64:
        raise ValueError("synthetic_launch_id_invalid")
    claim = {"schema": admission.SCHEMA + "/synthetic_claim_v1", "launch_id": launch_id,
             "authorization_sha256": authorization_sha256,
             "admission_metadata_sha256": package["metadata_sha256"],
             "implementation_closure_sha256": _canonical_sha(pre),
             "t0": t0, "work_deadline": work_deadline, "hard_deadline": hard_deadline,
             "attempt": 1, "parameter_updates": 0}
    claim_record = _write_exclusive_json(attempt, "claim.json", claim)
    launch_spec = {"schema": admission.SCHEMA + "/synthetic_worker_launch_v1",
                   "launch_id": launch_id, "authorization_sha256": authorization_sha256,
                   "admission_metadata_sha256": package["metadata_sha256"],
                   "implementation_closure_sha256": _canonical_sha(pre),
                   "ledger_sha256": ledger["sha256"], "parameter_updates": 0}
    launch_spec_record = _write_exclusive_json(attempt, "worker_launch.json", launch_spec)
    launch_event = _record_authorization_event(
        ledger_root, ledger, "launch_started.json",
        {"schema": admission.SCHEMA + "/authorization_event_v1", "event": "launch_started",
         "launch_id": launch_id, "attempt_root": str(attempt),
         "launch_spec_sha256": launch_spec_record["sha256"], "parameter_updates": 0},
    )

    wait = supervise_process(command, output_root=attempt, t0=t0, work_deadline=work_deadline,
                             hard_deadline=hard_deadline, clock=clock, popen=popen, cwd=attempt,
                             env=worker_env)
    if wait.return_code != 0 or wait.timed_out or not wait.group_reaped or wait.log_overflow:
        raise ValueError("synthetic_worker_failed_or_unclean")
    candidate_raw = _read_regular(attempt / "worker_candidate.json", 64 << 10)
    try:
        candidate = json.loads(candidate_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("synthetic_worker_candidate_invalid") from exc
    expected_candidate = {"schema": admission.SCHEMA + "/synthetic_worker_candidate_v1",
                          "status": "candidate_complete", "launch_id": launch_id,
                          "authorization_sha256": authorization_sha256,
                          "admission_metadata_sha256": package["metadata_sha256"],
                          "implementation_closure_sha256": _canonical_sha(pre),
                          "ledger_sha256": ledger["sha256"],
                          "parameter_updates": 0}
    if candidate != expected_candidate:
        raise ValueError("synthetic_worker_candidate_binding")
    candidate_record = {"bytes": len(candidate_raw), "sha256": hashlib.sha256(candidate_raw).hexdigest()}
    candidate_event = _record_authorization_event(
        ledger_root, ledger, "candidate_validated.json",
        {"schema": admission.SCHEMA + "/authorization_event_v1", "event": "candidate_validated",
         "launch_id": launch_id, "candidate_sha256": candidate_record["sha256"],
         "parameter_updates": 0},
    )

    post = _implementation_closure(project_root, package["implementation"]["files"])
    if post != pre:
        raise ValueError("implementation_closure_changed_during_worker")
    wait_record = return_witness(wait, t0=t0, registered_s=hard_cap_s,
                                 work_deadline=work_deadline, hard_deadline=hard_deadline,
                                 final_sample=wait.cleaned_at)
    return_candidate_event = _record_authorization_event(
        ledger_root, ledger, "return_candidate.json",
        {"schema": admission.SCHEMA + "/authorization_event_v1", "event": "return_candidate",
         "launch_id": launch_id, "implementation_closure_sha256": _canonical_sha(post),
         "candidate_sha256": candidate_record["sha256"], "parameter_updates": 0},
    )
    binding_facts = {"claim": claim_record, "ledger": {"path": ledger["path"], "sha256": ledger["sha256"]},
                     "ledger_events": [launch_event, candidate_event, return_candidate_event],
                     "launch_spec": launch_spec_record, "candidate": candidate_record,
                     "implementation_pre_sha256": _canonical_sha(pre),
                     "implementation_post_sha256": _canonical_sha(post),
                     "wait_witness_sha256": _canonical_sha(wait_record),
                     "parameter_updates": 0}
    return_receipt = seal_external_return(
        attempt, wait, t0=t0, registered_s=hard_cap_s,
        work_deadline=work_deadline, hard_deadline=hard_deadline,
        clock=clock, binding_facts=binding_facts,
    )
    consumed = verify_external_return(attempt, return_receipt, clock=clock)
    if (consumed["coverage"] != "covered"
            or return_receipt.observation.get("binding_facts") != binding_facts):
        raise ValueError("synthetic_return_binding_failed")
    return {"status": "candidate_complete", "attempt_root": str(attempt),
            "claim": claim, "claim_sha256": claim_record["sha256"],
            "authorization_ledger": ledger, "launch_spec": launch_spec,
            "authorization_ledger_events": [launch_event, candidate_event, return_candidate_event],
            "candidate": candidate, "candidate_sha256": candidate_record["sha256"],
            "implementation_pre_sha256": _canonical_sha(pre),
            "implementation_post_sha256": _canonical_sha(post),
            "wait": wait_record, "return": consumed, "parameter_updates": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    dispatch_started = False
    refusal = {"schema": _ADMISSION_SCHEMA + "/supervisor_refusal",
               "status": "refused", "authorization_path_opened": False,
               "worker_spawned": False, "checkpoint_or_batch_reads": 0,
               "parameter_updates": 0}
    if not args.run:
        print(json.dumps({**refusal, "reason": "run_flag_required"}, sort_keys=True))
        return 3
    if args.admission is None:
        print(json.dumps({**refusal, "reason": "executable_admission_packet_required"}, sort_keys=True))
        return 3
    try:
        project_root = _real_directory(args.project_root)
        admission_path = Path(args.admission).absolute()
        if admission_path.resolve(strict=True) != admission_path or \
                not admission_path.is_relative_to(project_root):
            raise ValueError("admission_path_not_in_project")
        package_raw = _read_regular(admission_path, 16 << 20)
        package = json.loads(package_raw)
        if type(package) is not dict:
            raise ValueError("real_admission_package_not_object")
        package_body = {key: value for key, value in package.items()
                        if key != "metadata_sha256"}
        if (package.get("schema") != _ADMISSION_SCHEMA
                or package.get("metadata_sha256") != _canonical_sha(package_body)):
            raise ValueError("real_admission_package_hash_invalid")
        if (package.get("executable") is not True
                or package.get("status", {}).get("real_worker") !=
                "implemented_and_reviewed_and_supervised"):
            print(json.dumps({**refusal, "reason": "real_admission_not_executable",
                              "admission_path_opened": True}, sort_keys=True))
            return 3
        ledger_rel = Path(package.get("authorization_consumption_ledger", ""))
        if ledger_rel.is_absolute() or ".." in ledger_rel.parts or not ledger_rel.parts:
            raise ValueError("real_authorization_ledger_path_invalid")
        dispatch_started = True
        result = run_authorized_attempt(
            authorization_path=args.authorization,
            authorization_sha256=args.authorization_sha256,
            package=package,
            project_root=project_root,
            ledger_root=project_root / ledger_rel,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as exc:
        print(json.dumps({"schema": _ADMISSION_SCHEMA + "/supervisor_failure",
                          "status": "incomplete", "error_type": type(exc).__name__,
                          "error": str(exc)[:500],
                          "execution_state": ("dispatch_started_outcome_unknown"
                                              if dispatch_started else "pre_dispatch_rejected"),
                          "parameter_updates": 0},
                         sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
