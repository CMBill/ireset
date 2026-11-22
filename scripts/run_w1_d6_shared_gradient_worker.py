"""Independent, package-pinned CLI for the fixed-six-window D6 worker.

This file uses only the standard library until the package and complete
implementation closure have been verified. It cannot mint authorization or
choose its own source paths, command, output root, or resource limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time


def _read_regular(path: Path, cap: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > cap:
            raise ValueError("worker_input_type_or_size")
        chunks, left = [], before.st_size
        while left:
            part = os.read(fd, min(left, 1 << 20))
            if not part:
                raise ValueError("worker_input_short_read")
            chunks.append(part)
            left -= len(part)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("worker_input_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_project_file(root: Path, relative: Path, cap: int) -> bytes:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("worker_project_path_invalid")
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory_fd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > cap:
                raise ValueError("worker_implementation_type_or_size")
            chunks, left = [], before.st_size
            while left:
                chunk = os.read(fd, min(left, 1 << 20))
                if not chunk:
                    raise ValueError("worker_implementation_short_read")
                chunks.append(chunk)
                left -= len(chunk)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("worker_implementation_changed_during_read")
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(directory_fd)


def _json(raw: bytes, label: str) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(label + "_duplicate_key")
            out[key] = value
        return out

    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    if type(value) is not dict:
        raise ValueError(label + "_not_object")
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(doc: dict) -> str:
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return _sha(raw)


def _implementation_closure(root: Path, files: dict, budget=None) -> dict:
    if type(files) is not dict or not files:
        raise ValueError("worker_implementation_roster_invalid")
    found = {}
    for relative, expected in sorted(files.items()):
        rel = Path(relative)
        if (rel.is_absolute() or not rel.parts or ".." in rel.parts
                or type(expected) is not dict or type(expected.get("bytes")) is not int
                or not 0 < expected["bytes"] <= 32 << 20
                or type(expected.get("sha256")) is not str or len(expected["sha256"]) != 64):
            raise ValueError("worker_implementation_path_invalid")
        if budget is not None:
            budget.check("worker_preimport_hash:" + relative + ":pre")
        raw = _read_project_file(root, rel, expected["bytes"])
        if budget is not None:
            budget.check("worker_preimport_hash:" + relative + ":post")
        record = {"bytes": len(raw), "sha256": _sha(raw)}
        if record != {"bytes": expected.get("bytes"), "sha256": expected.get("sha256")}:
            raise ValueError("worker_implementation_drift:" + relative)
        found[relative] = record
    return found


def _verify_ledger(project_root: Path, package: dict, launch: dict, launch_spec_raw: bytes) -> dict:
    ledger_rel = Path(package.get("authorization_consumption_ledger", ""))
    if ledger_rel.is_absolute() or ".." in ledger_rel.parts or not ledger_rel.parts:
        raise ValueError("worker_ledger_path_invalid")
    directory = ledger_rel / launch["authorization_sha256"]
    consumed_raw = _read_project_file(project_root, directory / "consumed.json", 64 << 10)
    consumed = _json(consumed_raw, "worker_consumption_receipt")
    expected_record = launch.get("authorization_ledger")
    if (type(expected_record) is not dict or _sha(consumed_raw) != expected_record.get("sha256")
            or expected_record.get("path") != launch["authorization_sha256"] + "/consumed.json"
            or consumed.get("authorization_sha256") != launch["authorization_sha256"]
            or consumed.get("launch_id") != launch["launch_id"]
            or consumed.get("admission_metadata_sha256") != launch["admission_metadata_sha256"]
            or consumed.get("implementation_closure_sha256") != launch.get("implementation_pre_sha256")
            or consumed.get("parameter_updates") != 0 or consumed.get("attempts") != 1):
        raise ValueError("worker_consumption_receipt_binding")
    launch_raw = _read_project_file(project_root, directory / "launch_started.json", 64 << 10)
    started = _json(launch_raw, "worker_launch_event")
    if (started.get("event") != "launch_started" or started.get("launch_id") != launch["launch_id"]
            or started.get("launch_spec_sha256") != _sha(launch_spec_raw)
            or started.get("parameter_updates") != 0):
        raise ValueError("worker_launch_event_binding")
    return {"consumed_sha256": _sha(consumed_raw), "launch_event_sha256": _sha(launch_raw),
            "reads": 2}


def _verify_worker_recipe(package: dict, project: Path) -> None:
    worker = package.get("worker_launch")
    if type(worker) is not dict:
        raise ValueError("worker_launch_recipe_missing")
    relative = Path(worker.get("script_rel", ""))
    expected_argv = [sys.executable, "-I", str(project / relative),
                     "--launch-spec", "{launch_spec}",
                     "--admission-snapshot", "{admission_snapshot}",
                     "--authorization-snapshot", "{authorization_snapshot}",
                     "--authorization-sha256", "{authorization_sha256}",
                     "--project-root", "{project_root}"]
    if (worker.get("status") != "reviewed_executable"
            or worker.get("execution_mode") != "real_worker_v1"
            or worker.get("executable") != sys.executable
            or worker.get("cwd") != "attempt_root"
            or worker.get("shell") is not False
            or worker.get("inherit_stdin") is not False
            or worker.get("new_process_group") is not True
            or relative.is_absolute() or not relative.parts or ".." in relative.parts
            or relative.as_posix() != "scripts/run_w1_d6_shared_gradient_worker.py"
            or worker.get("argv_template") != expected_argv
            or worker.get("env") != dict(os.environ)):
        raise ValueError("worker_launch_recipe_mismatch")


class WorkerBudget:
    def __init__(self, t0: float, deadline: float, clock=time.perf_counter):
        self.t0, self.deadline, self.clock = t0, deadline, clock
        self.points = []
        self.units = {}

    def check(self, point: str) -> None:
        if self.clock() >= self.deadline:
            raise TimeoutError("worker_work_deadline:" + point)
        self.points.append(point)

    def unit(self, name, fn, /, *args, **kwargs):
        self.check(name + ":pre")
        self.units[name] = self.units.get(name, 0) + 1
        value = fn(*args, **kwargs)
        self.check(name + ":post")
        return value

    def facts(self) -> dict:
        return {"t0": self.t0, "work_deadline": self.deadline,
                "elapsed_s": self.clock() - self.t0, "unit_calls": dict(self.units),
                "boundary_checks": len(self.points), "last_check": self.points[-1] if self.points else None}


def execute_worker(*, launch_spec_path: Path, admission_snapshot_path: Path,
                   authorization_snapshot_path: Path, authorization_sha256: str,
                   project_root: Path | None = None, clock=time.perf_counter) -> dict:
    """Validate all pinned metadata/ledger facts before importing numerical code."""
    project = Path(project_root).resolve(strict=True) if project_root is not None else Path.cwd().resolve(strict=True)
    started = clock()
    launch_raw = _read_regular(launch_spec_path, 1 << 20)
    metadata_reads = 1
    launch = _json(launch_raw, "worker_launch_spec")
    t0 = launch.get("t0")
    work_deadline = launch.get("work_deadline")
    if (type(t0) not in (int, float) or type(work_deadline) not in (int, float)
            or type(launch.get("hard_deadline")) not in (int, float)
            or t0 != t0 or work_deadline != t0 + 240 or launch["hard_deadline"] != t0 + 300
            or not t0 <= started < work_deadline or clock() >= work_deadline):
        raise TimeoutError("worker_launch_deadline_invalid")
    budget = WorkerBudget(t0, work_deadline, clock)
    budget.check("admission_snapshot:pre")
    admission_raw = _read_regular(admission_snapshot_path, 16 << 20)
    metadata_reads += 1
    budget.check("admission_snapshot:post")
    budget.check("authorization_snapshot:pre")
    authorization_raw = _read_regular(authorization_snapshot_path, 1 << 20)
    metadata_reads += 1
    budget.check("authorization_snapshot:post")
    package = _json(admission_raw, "worker_admission_snapshot")
    auth = _json(authorization_raw, "worker_authorization_snapshot")
    if (launch.get("schema") != "w1_d6_shared_gradient_real_admission_v1/worker_launch_spec_v1"
            or launch.get("execution_mode") != "real_worker_v1"
            or launch.get("project_root") != str(project)
            or launch.get("attempt_root") != str(project / package.get("proposed_real_root", ""))
            or Path.cwd().resolve(strict=True) != Path(launch["attempt_root"]).resolve(strict=True)
            or Path(launch_spec_path).resolve(strict=True) != Path(launch["attempt_root"]) / "worker_launch.json"
            or Path(admission_snapshot_path).resolve(strict=True) != Path(launch["attempt_root"]) / "admission_snapshot.json"
            or Path(authorization_snapshot_path).resolve(strict=True) != Path(launch["attempt_root"]) / "authorization_snapshot.json"
            or authorization_sha256 != launch.get("authorization_sha256")
            or _sha(authorization_raw) != authorization_sha256
            or launch.get("authorization_record") != {"bytes": len(authorization_raw),
                                                       "sha256": authorization_sha256}
            or _sha(admission_raw) != launch.get("admission_record", {}).get("sha256")
            or len(admission_raw) != launch.get("admission_record", {}).get("bytes")
            or launch.get("admission_metadata_sha256") != package.get("metadata_sha256")
            or launch.get("implementation_pre_sha256") != _canonical_sha(launch.get("implementation_pre"))
            or launch.get("parameter_updates") != 0):
        raise ValueError("worker_launch_binding_invalid")
    package_body = {key: value for key, value in package.items() if key != "metadata_sha256"}
    if (package.get("executable") is not True
            or package.get("metadata_sha256") != _canonical_sha(package_body)
            or package.get("status", {}).get("real_worker") != "implemented_and_reviewed_and_supervised"):
        raise ValueError("worker_package_not_executable")

    closure_passes = 1
    closure = _implementation_closure(project, package.get("implementation", {}).get("files"), budget)
    if closure != launch.get("implementation_pre"):
        raise ValueError("worker_preimport_closure_mismatch")

    # Only import project code after the complete byte closure has passed.
    # Numerical modules remain unloaded until the authorization and recipe gates
    # below have also passed.
    sys.path.insert(0, str(project / "src"))
    from ireset.eval import w1_d6_shared_gradient_admission as admission

    problems = admission.authorization_problems(auth, package)
    if problems:
        raise ValueError("worker_authorization_rejected:" + ",".join(problems))
    if (auth.get("launch_id") != launch.get("launch_id")
            or type(launch.get("launch_id")) is not str or len(launch["launch_id"]) != 64
            or any(ch not in "0123456789abcdef" for ch in launch["launch_id"])
            or auth.get("attempts") != 1 or auth.get("includes") != ["single_step0_diagnostic"]):
        raise ValueError("worker_authorization_launch_binding")
    if Path(__file__).resolve(strict=True) != (project / package["worker_launch"]["script_rel"]).resolve(strict=True):
        raise ValueError("worker_entrypoint_not_package_pinned")
    if sys.executable != package["worker_launch"].get("executable"):
        raise ValueError("worker_interpreter_not_package_pinned")
    if dict(os.environ) != package["worker_launch"].get("env"):
        raise ValueError("worker_environment_not_package_pinned")
    _verify_worker_recipe(package, project)
    matrix_problems = admission.real_call_matrix_problems(package.get("call_matrix"))
    matrix_problems.extend(admission.real_io_matrix_problems(package))
    if matrix_problems:
        raise ValueError("worker_matrix_rejected:" + ",".join(matrix_problems))
    ledger_facts = _verify_ledger(project, package, launch, launch_raw)

    budgets = package.get("budgets")
    if (type(budgets) is not dict or budgets.get("wall_cap_s") != 300
            or budgets.get("work_cap_s") != 240 or budgets.get("attempts") != 1
            or budgets.get("parameter_updates") != 0
            or launch.get("hard_deadline") - launch.get("t0") != 300
            or launch.get("work_deadline") - launch.get("t0") != 240
            or not (launch.get("t0") <= started < launch.get("work_deadline"))):
        raise ValueError("worker_budget_binding_invalid")
    budget.check("worker_import_numerics")
    from ireset.eval import w1_d6_shared_gradient_real as worker
    budget.check("worker_import_numerics:post")

    attempt = Path(launch["attempt_root"])
    d6_dir = attempt / "d6"
    d6_dir.mkdir(mode=0o700, exist_ok=False)
    construction = {"entered": [], "returned": []}
    failure_context = {}

    def construction_entry(name):
        budget.check("trainer_construct_" + name + ":pre")
        if name in construction["entered"]:
            raise ValueError("worker_trainer_construction_duplicate")
        construction["entered"].append(name)

    def construction_check(name, module):
        budget.check("trainer_construct_" + name + ":post")
        if module is None:
            raise ValueError("worker_trainer_construction_missing_module")
        construction["returned"].append(name)

    try:
        result = worker.run_authorized_worker(
            authorization_raw=authorization_raw, authorization_sha256=authorization_sha256,
            package=package, project_root=project, work_budget=budget,
            runtime_construction_entry=construction_entry,
            runtime_construction_check=construction_check,
            failure_context=failure_context)
    except BaseException as exc:
        failure = {"schema": "w1_d6_shared_gradient_real_worker_v1/worker_failure",
                   "status": "incomplete", "error_type": type(exc).__name__,
                   "error": str(exc)[:500], "worker_budget": budget.facts(),
                   "construction": construction,
                   "failure_context": failure_context,
                   "parameter_updates": 0}
        try:
            _write_failure(attempt / "worker_failure.json", failure)
        except BaseException:
            pass
        raise
    closure_passes += 1
    expected_entered = ["model", "model_transfer", "conditioner", "conditioner_transfer",
                        "train_loader", "val_loader"]
    expected_returned = ["model", "model_transfer", "conditioner", "conditioner_transfer"]
    if construction != {"entered": expected_entered, "returned": expected_returned}:
        raise ValueError("worker_trainer_construction_matrix")
    if result.get("parameter_updates") != 0:
        raise ValueError("worker_parameter_update_forbidden")
    result["worker_io"] = {"launch_metadata_reads": metadata_reads, "ledger_reads": ledger_facts["reads"],
                            "implementation_closure_hash_passes": closure_passes,
                            "source_file_reads": result["source_io"]["source_file_reads"],
                            "source_file_hashes": result["source_io"]["source_file_hashes"],
                            "d6_npz_member_reads": result["io_matrix"]["d6_npz_member_reads"],
                            "root_stats_npz_member_reads": result["io_matrix"]["root_stats_npz_member_reads"],
                            "npz_archive_opens": result["io_matrix"]["npz_archive_opens"],
                            "independent_artifact_reads": result["io_matrix"]["independent_artifact_reads"],
                            "independent_artifact_hashes": result["io_matrix"]["independent_artifact_hashes"],
                            "artifact_npy_writes": result["artifact_io"]["npy_writes"],
                            "artifact_json_writes": result["artifact_io"]["json_writes"],
                            "checkpoint_cpu_loads": result["io_matrix"]["checkpoint_cpu_loads"],
                            "strict_model_loads": result["io_matrix"]["strict_model_loads"],
                            "d6_directory_created": 1, "worker_result_writes": 1,
                            "worker_result_readbacks": 1}
    result["worker_budget"] = budget.facts()
    result["worker_matrix_sha256"] = package["metadata_sha256"]
    return result


def _write_result(path: Path, result: dict) -> None:
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or len(raw) > 2 << 20:
            raise ValueError("worker_result_type_or_size")
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    dirfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    if _read_regular(path, 2 << 20) != raw:
        raise ValueError("worker_result_readback_mismatch")


def _write_failure(path: Path, failure: dict) -> None:
    """Persist bounded partial facts without publishing a candidate result."""
    raw = (json.dumps(failure, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > 2 << 20:
        raise ValueError("worker_failure_too_large")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if _read_regular(path, 2 << 20) != raw:
        raise ValueError("worker_failure_readback_mismatch")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--launch-spec", type=Path, required=True)
    parser.add_argument("--admission-snapshot", type=Path, required=True)
    parser.add_argument("--authorization-snapshot", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute_worker(launch_spec_path=args.launch_spec,
                                admission_snapshot_path=args.admission_snapshot,
                                authorization_snapshot_path=args.authorization_snapshot,
                                authorization_sha256=args.authorization_sha256,
                                project_root=args.project_root)
        work_deadline = result["worker_budget"]["work_deadline"]
        if time.perf_counter() >= work_deadline:
            raise TimeoutError("worker_result_write:pre")
        _write_result(Path.cwd() / "worker_result.json", result)
        if time.perf_counter() >= work_deadline:
            raise TimeoutError("worker_result_write:post")
    except BaseException as exc:
        print(json.dumps({"schema": "w1_d6_shared_gradient_real_worker_v1/worker_failure",
                          "status": "incomplete", "error_type": type(exc).__name__,
                          "error": str(exc)[:500]}, sort_keys=True), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
