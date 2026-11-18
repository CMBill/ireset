"""D1-only supervisor with an independent parent that waits and archives.

The audited legacy supervisor is loaded as a private module, not modified.
Every comparison stays provisional until the external wait and archive are verified.
One fixed root/attempt is allowed for R′ and B independently; no retries.
"""

import time

_T0 = time.perf_counter()  # before uv, imports, reservation or subprocess

import argparse
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ireset.eval import w1_d1_lr_protocol as protocol
from ireset.eval import w1_d1_lr_artifacts as artifacts

COHORT = protocol.COHORT


class InheritedGroupWorker(subprocess.Popen):
    """Only the outer observer owns the hard deadline and process group."""

    def __init__(self, *args, start_new_session=False, limit_logs=False, **kwargs):
        logs = {key: kwargs[key] for key in ("stdout", "stderr")} if limit_logs else {}
        kwargs.update(dict.fromkeys(logs, subprocess.PIPE))
        super().__init__(*args, start_new_session=False, **kwargs)
        self._log_errors = []
        self._drains = []
        for key, out in logs.items():
            thread = threading.Thread(target=self._drain_log, args=(getattr(self, key), out), daemon=True)
            thread.start()
            self._drains.append(thread)

    def _drain_log(self, stream, out):
        remaining = artifacts.LOG_BYTES
        try:
            with stream:
                while chunk := stream.read1(64 << 10):
                    out.write(chunk[:remaining])
                    if len(chunk) > remaining:
                        raise artifacts.ArtifactLimit("lr_artifact_log_slot_exceeded")
                    remaining -= len(chunk)
        except (OSError, ValueError) as exc:
            self._log_errors.append(str(exc))

    def wait(self, timeout=None):
        # The legacy supervisor's private TERM/grace/KILL loop must not race
        # its parent or create another session. The parent kills this group.
        rc = super().wait()
        for thread in self._drains:
            thread.join(timeout=1.0)
        if self._log_errors or any(t.is_alive() for t in self._drains):
            raise artifacts.ArtifactLimit(f"lr_bounded_log_capture_incomplete:{self._log_errors}")
        return rc


def _adopt_orphans():
    import ctypes

    prctl = ctypes.CDLL(None, use_errno=True).prctl
    if prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def wait_in_owned_group(command, *, deadline, **kwargs):
    """Return only after the leader and adopted group descendants are reaped."""
    _adopt_orphans()
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise InterruptedError("LR supervisor interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        proc = subprocess.Popen(command, start_new_session=True, **kwargs)
        unclean = False
        try:
            rc = proc.wait(timeout=max(0.0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            unclean = True
            _kill_group(proc.pid)
            rc = proc.wait()
        finally:
            if proc.returncode is None:
                _kill_group(proc.pid)
                proc.wait()
            while True:
                try:
                    pid, _ = os.waitpid(-proc.pid, os.WNOHANG)
                    if pid == 0:
                        _kill_group(proc.pid)
                        os.waitpid(-proc.pid, 0)
                except ChildProcessError:
                    break
                unclean = True  # a child outliving the inner leader is not success
        return rc, unclean, proc.pid
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def load_core():
    spec = importlib.util.spec_from_file_location(
        "_lr_private_supervisor", ROOT / "scripts/run_w1_smallset_fit_real_supervisor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.WORKER = ROOT / "scripts/run_w1_d1_lr_schedule.py"
    mod.CHAIN_CAP = 2400.0
    mod.STAGE_CAPS = {"d1": 1800.0, "f": 600.0}
    old_sources = mod._source_hashes
    def sources():
        return {**old_sources(), **{str(p.relative_to(ROOT)): mod._sha(p) for p in (
            Path(__file__).resolve(),
            ROOT / "docs/plans/omniretarget_w1_d1_lr_schedule_diagnostic_plan.md",
            ROOT / "tests/test_w1_d1_lr_schedule.py",
            ROOT / "tests/test_w1_d1_lr_review_regressions.py")}}
    mod._source_hashes = sources
    old_r = mod._r_summary_valid
    old_candidate = mod._candidate_problems

    def r_valid(doc):
        package = doc.get("package")
        if (not isinstance(package, dict)
                or package.get("schema") != "w1_d1_lr_admission_package_v1"
                or package.get("cohort_id") != COHORT
                or not isinstance(package.get("schedule"), dict)
                or package["schedule"].get("id") != "d1_half_after_500_v1"):
            return False
        return old_r({**doc, "package": {**package, "schema": "w1_smallset_fit_admission_package_v1"}})

    def candidate(doc, launch_id, t0, mode, expected_cache_root=None):
        errors = old_candidate(doc, launch_id, t0, mode, expected_cache_root)
        if mode == "chain":
            try:
                if doc.get("authorized_includes") != ["D1", "F"] or doc.get("d6_ran") is not False:
                    errors.append("lr_d6_forbidden")
                binding = doc["comparison"]
                path = Path(doc["_output_dir"]) / "f/comparison.json"
                f, _ = protocol._object_and_sha(path.parent / "f_report.json")
                cmp, _ = protocol._object_and_sha(path)
                protocol.verify_stage_ledger(path.parent.parent, doc)
                if (binding != f.get("comparison") or binding["sha256"] != mod._sha(path)
                        or binding["bytes"] != path.stat().st_size
                        or cmp.get("state") != "pending_outer_return"
                        or cmp.get("conclusion") == "not_comparable"
                        or cmp.get("fit_verdict") != f.get("verdict")
                        or cmp.get("metrics") is None or cmp.get("problems") != []
                        or cmp.get("baseline_sha256") != doc.get("baseline_sha256")):
                    errors.append("lr_comparison_binding_invalid")
            except (OSError, ValueError, TypeError, KeyError):
                errors.append("lr_comparison_unreadable")
        return errors

    # _candidate_problems needs the reserved output directory, supplied by
    # the proof reader (not a self-reported report field).
    old_proof = mod._proof_problems

    def proof(path, *args):
        problems, doc, observation, hashes = old_proof(path, *args)
        if path.name == "acceptance_summary.json":
            doc["_output_dir"] = str(path.parent)
        return problems, doc, observation, hashes

    mod._proof_problems = proof
    mod._r_summary_valid = r_valid
    mod._candidate_problems = candidate
    return mod


def _base_command(args, output):
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", args.mode,
               "--output-root", str(output)]
    for flag, value in (("--admission", args.admission), ("--authorization", args.authorization),
                        ("--cache-root", args.cache_root)):
        if value:
            command += [flag, str(Path(value).resolve())]
    if args.synthetic:
        command.append("--synthetic")
    return command


def _single_attempt(root):
    attempt = root / "attempt_001"
    attempt.mkdir()  # no alternative, even for a direct/parallel inner invocation
    return attempt


def _inner(args, output):
    t0 = float.fromhex(args.inner_t0_hex)
    claim = json.loads((output / "cohort_launch_claim.json").read_bytes())
    expected_group = os.getppid()
    if (not math.isfinite(t0) or not 0 < t0 <= _T0
            or os.getpgrp() != expected_group or claim.get("observer_pid") != os.getppid()
            or claim.get("t0_perf_counter") != t0 or claim.get("cohort") != COHORT
            or claim.get("mode") != args.mode or claim.get("synthetic") is not args.synthetic
            or claim.get("state") != "running" or any(output.glob("attempt_*"))):
        raise ValueError("inner supervisor must join a newly reserved observer launch")
    if args.synthetic:
        # Read real fixture paths for the inner post-wait SHA check.
        fixture = output / "attempt_001/synthetic_materials"
        args.admission = str(fixture / "fixture_admission.json")
        args.authorization = str(fixture / "fixture_authorization.json")
    mod = load_core()
    space = artifacts.ArtifactSpace(output, supervised=True) if args.mode == "chain" else None
    mod.subprocess = SimpleNamespace(
        Popen=lambda *a, **kw: InheritedGroupWorker(*a, limit_logs=space is not None, **kw),
        TimeoutExpired=subprocess.TimeoutExpired)
    mod._attempt_dir = _single_attempt
    write = mod._write_json

    def write_with_group(path, doc):
        if path.name == "outer_launch.json":
            doc = {**doc, "worker_process_group": expected_group}
        if space is None:
            write(path, doc)
        else:
            space.write_json(path, doc, closing=path.name != "outer_launch.json")

    mod._write_json = write_with_group
    return mod.run(args, t0=t0)


def _observe(args, output, t0):
    mod = load_core()
    space = artifacts.ArtifactSpace(output, supervised=True) if args.mode == "chain" else None
    write_closure = mod._write_json if space is None else space.write_closure_json
    output.mkdir(parents=True, exist_ok=True)
    # O_EXCL reservation plus nonempty-root refusal closes parallel launches,
    # failed/crashed/unknown histories and all attempt-renaming retry paths.
    if any(output.iterdir()):
        raise ValueError("once-only attempt consumed or history unknown")
    claim = output / "cohort_launch_claim.json"
    doc = {"cohort": COHORT, "mode": args.mode, "state": "running", "t0_perf_counter": t0,
           "synthetic": args.synthetic, "observer_pid": os.getpid(), "max_attempts": 1,
           "parent_consumption": {"cohort": "w1_patched_smallset_fit_v1",
                                  "run": "derived/experiments/omniretarget_first/attempt_002",
                                  "authorization_consumed": True, "old_incomplete_not_reclassified": True}}
    if space is None:
        with claim.open("x") as out:
            json.dump(doc, out)
            out.flush()
            os.fsync(out.fileno())
        artifacts.sync_directory(output)
    else:
        space.write_json(claim, {**doc, "collector_pid": os.getppid(), "artifact_space": artifacts.policy()},
                         exclusive=True)
    command = _base_command(args, output) + ["--inner-t0-hex", t0.hex()]
    with (output / "supervisor_stdout.log").open("xb") as stdout, (output / "supervisor_stderr.log").open("xb") as stderr:
        if time.perf_counter() >= t0 + args.wall_cap_s - args.reserve_s:
            rc, unclean = None, False
        elif args.mode == "r-admit":
            # The R collector owns this group and its hard deadline; every
            # descendant (observer, inner, uv, worker) inherits that group.
            rc = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr).returncode
            unclean = False
        else:
            # One process group is owned by the collector, including uv and
            # the numerical worker. The observer must not create a nested group.
            inner = InheritedGroupWorker(command, limit_logs=True, cwd=ROOT, stdout=stdout, stderr=stderr)
            rc, unclean = inner.wait(), False
        returned = time.perf_counter()
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    if rc != 0 or unclean or returned - t0 > args.wall_cap_s:
        if space is not None:
            write_closure(output / "supervisor_failure.json", {
                "state": "incomplete", "inner_return_code": rc, "elapsed_s_lower_bound": returned - t0,
                "problems": ["lr_supervisor_group_or_return_incomplete"]})
            return 3
        mod._write_incomplete_costed(output / "supervisor_failure.json", {
            "schema": "w1_d1_lr_observer_failure_v1", "mode": args.mode,
            "supervisor_return_code": rc, "process_group_unclean": unclean,
            "problems": ["lr_supervisor_group_or_return_incomplete"],
        }, t0=t0, wall_cap_s=args.wall_cap_s)
        return 3
    if {p.name for p in output.glob("attempt_*")} != {"attempt_001"}:
        return 3
    attempt = output / "attempt_001"
    try:
        artifact_bytes = (sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                          if space is None else space.used_bytes())
        if artifact_bytes > artifacts.CAP_BYTES:
            return 3
        verdict = json.loads((attempt / "acceptance_verdict.json").read_bytes())
        registered = verdict["registered_measured_s"]
        if (verdict["verdict"] != "candidate_complete" or verdict["problems"] != []
                or not protocol.number(registered) or not returned - t0 <= registered <= args.wall_cap_s):
            return 3
        record = {"schema": protocol.RETURN_SCHEMA, "mode": args.mode, "launch_id": verdict["launch_id"],
                   "supervisor_return_code": rc, "t0_perf_counter": t0, "supervisor_return_perf_counter": returned,
                   "elapsed_after_actual_wait_s": returned - t0,
                  "registered_measured_s": registered, "wall_cap_s": args.wall_cap_s,
                  "comparison_candidate": args.mode == "chain",
                  "state": "requires_observer_actual_return_zero",
                  "artifacts_bytes_at_return": artifact_bytes,
                   "bindings": {rel: mod._sha(attempt / rel) for rel in protocol.attempt_bindings(args.mode)}}
        path = output / protocol.COMPLETION_NAME
        write_closure(path, record)
        after_record = time.perf_counter() - t0
        write_closure(output / protocol.SAMPLE_NAME, {
            "completion_sha256": mod._sha(path), "elapsed_after_completion_write_s": after_record,
            "registered_measured_s": registered, "wall_cap_s": args.wall_cap_s})
        # No output or I/O after the passing final sample. Exit=0 is required
        # together with these immutable bindings; files alone cannot self-certify.
        protocol.verify_observer_files(attempt, args.mode)
        return 0 if time.perf_counter() - t0 <= registered else 3
    except (OSError, ValueError, KeyError, TypeError):
        return 3


def _record_r_return(output, *, rc, pid, returned, t0):
    """Called only by the parent that waited for the observation publisher."""
    if type(rc) is not int or rc != 0:
        return 3
    attempt = output / "attempt_001"
    completion, completion_sha, sample, sample_sha, summary, _ = protocol.verify_observer_files(attempt, "r-admit")
    registered = completion["registered_measured_s"]
    elapsed = returned - t0
    claim, _ = protocol._object_and_sha(output / "cohort_launch_claim.json")
    if (completion["t0_perf_counter"] != t0 or claim.get("observer_pid") != pid
            or summary.get("admission_passed") is not True
            or not sample["elapsed_after_completion_write_s"] <= elapsed <= registered):
        return 3
    # The worker wrote ONLY an unbound template. The separate collector builds
    # the final *unsigned* draft from actual wait evidence, never signs it.
    draft = json.loads((attempt / "authorization_draft.json").read_bytes())
    if (draft.get("authorized") is not False or draft.get("template_only") is not True
            or draft.get("admission_package_sha256") != completion["bindings"]["r_admission.json"]
            or draft.get("r_observer_actual_return") is not None):
        return 3
    mod = load_core()
    path = output / protocol.R_RETURN_NAME
    mod._write_json(path, {
        "schema": protocol.R_RETURN_SCHEMA, "observer_pid": pid, "observer_return_code": rc,
        "launch_id": completion["launch_id"], "t0_perf_counter": t0,
        "observer_return_perf_counter": returned, "elapsed_after_observer_wait_s": elapsed,
        "registered_measured_s": registered, "wall_cap_s": protocol.CAPS["r-admit"][0],
        "r_admission_sha256": completion["bindings"]["r_admission.json"],
        "completion_sha256": completion_sha, "observer_write_observation_sha256": sample_sha,
    })
    draft["r_observer_actual_return"] = {"sha256": mod._sha(path), "capture_actual_return_code": None}
    mod._write_json(output / "authorization_draft.json", draft)
    if time.perf_counter() - t0 <= registered:
        return 0
    mod._write_incomplete_costed(output / protocol.R_RETURN_FAILURE, {
        "schema": "w1_d1_lr_r_capture_failure_v1", "launch_id": completion["launch_id"],
        "registered_measured_s": registered, "problems": ["lr_r_capture_write_not_covered"],
    }, t0=t0, wall_cap_s=protocol.CAPS["r-admit"][0])
    return 3


def _capture_r_return(args, output):
    # Let the observer own and reap its worker group even when it times out.
    # This parent archives the observer's exit, not the inner/core worker exit.
    command = _base_command(args, output) + ["--r-observer-t0-hex", _T0.hex()]
    rc, unclean, pid = wait_in_owned_group(command, deadline=_T0 + args.wall_cap_s, cwd=ROOT)
    returned = time.perf_counter()
    return 3 if unclean else _record_r_return(output, rc=rc, pid=pid, returned=returned, t0=_T0)


def _record_chain_return(output, *, rc, unclean, pid, returned, t0):
    """Publish only this parent's actual wait and the post-fsync observation."""
    attempt = output / "attempt_001"
    space = artifacts.ArtifactSpace(output, supervised=True)
    if type(rc) is not int or rc != 0 or unclean:
        return 3  # no success receipt, and no write to a root another launcher might own
    owned = False
    try:
        packet = protocol.verify_observer_files(attempt, "chain")
        bound = protocol.chain_wait_binding(attempt, packet)
        if (bound["collector_pid"] != os.getpid() or bound["observer_pid"] != pid
                or bound["t0_perf_counter"] != t0):
            return 3
        owned = True
        reg = bound["registered_measured_s"]
        if not packet[2]["elapsed_after_completion_write_s"] <= returned - t0 <= reg:
            raise ValueError("lr_chain_actual_observer_wait_not_covered")
        path = output / protocol.CHAIN_WAIT_NAME
        space.write_json(path, {"schema": protocol.CHAIN_WAIT_SCHEMA, **bound,
                                "observer_actual_return_code": rc, "group_reaped_cleanly": True,
                                "observer_return_perf_counter": returned, "elapsed_s": returned - t0},
                         closing=True, exclusive=True)
        wait_sha = protocol.sha(path)
        synced = time.perf_counter()
        if not returned - t0 <= synced - t0 <= reg:
            raise ValueError("lr_chain_wait_archive_not_covered")
        space.write_json(output / protocol.CHAIN_ARCHIVE_NAME, {
            "schema": protocol.CHAIN_ARCHIVE_SCHEMA, "launch_id": bound["launch_id"], "wait_sha256": wait_sha,
            "archive_synced_perf_counter": synced, "elapsed_s": synced - t0}, closing=True, exclusive=True)
        if time.perf_counter() - t0 <= reg:
            return 0
        raise ValueError("lr_chain_archive_observation_write_not_covered")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # Only publish a failure for a launch this collector owns, including
        # failures after both tentative records are present.
        if owned:
            space.write_failure(output / protocol.CHAIN_CAPTURE_FAILURE, exc)
        return 3


def _capture_chain_return(args, output):
    if not args.synthetic and shutil.disk_usage(ROOT).free < 7 * (1 << 30):
        raise ValueError("lr_disk_margin")
    command = _base_command(args, output) + ["--chain-observer-t0-hex", _T0.hex()]
    rc, unclean, pid = wait_in_owned_group(command, deadline=_T0 + args.wall_cap_s, cwd=ROOT)
    return _record_chain_return(output, rc=rc, unclean=unclean, pid=pid,
                                returned=time.perf_counter(), t0=_T0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("r-admit", "chain"), default="chain")
    ap.add_argument("--output-root")
    ap.add_argument("--admission")
    ap.add_argument("--authorization")
    ap.add_argument("--cache-root")
    ap.add_argument("--synthetic", action="store_true")
    private = ap.add_mutually_exclusive_group()
    private.add_argument("--inner-t0-hex", help=argparse.SUPPRESS)
    private.add_argument("--r-observer-t0-hex", help=argparse.SUPPRESS)
    private.add_argument("--chain-observer-t0-hex", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.fixture, args.impl_sha = args.synthetic, None
    args.wall_cap_s, args.reserve_s = protocol.CAPS[args.mode]
    expected = (ROOT / (protocol.RUN_ROOT if args.mode == "chain" else protocol.R_ROOT)).resolve()
    default = ROOT / protocol.AUDIT_ROOT / "synthetic/acceptance_default" if args.synthetic else expected
    output = Path(args.output_root or default).resolve()
    if args.synthetic and args.mode != "chain":
        ap.error("synthetic acceptance uses chain mode")
    if args.synthetic and not protocol.fixture_output_allowed(output, ROOT):
        ap.error("synthetic output must be isolated from reserved real and historical roots")
    if not args.synthetic and output != expected:
        ap.error("fixed cohort root required; changing directories does not renew quota")
    if args.mode == "chain" and not args.synthetic and (not args.admission or not args.authorization):
        ap.error("NEW admission and explicit authorization required")
    try:
        if args.inner_t0_hex:
            return _inner(args, output)
        observer_t0 = args.r_observer_t0_hex or args.chain_observer_t0_hex
        if observer_t0:
            t0 = float.fromhex(observer_t0)
            mode = "r-admit" if args.r_observer_t0_hex else "chain"
            if (args.mode != mode or os.getpgrp() != os.getpid()
                    or not math.isfinite(t0) or not 0 < t0 <= _T0):
                raise ValueError("invalid observer launch")
            return _observe(args, output, t0)
        if output.exists() and any(output.iterdir()):
            ap.error("once-only attempt consumed or history unknown")
        if args.mode == "r-admit":
            return _capture_r_return(args, output)
        return _capture_chain_return(args, output)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"LR supervisor refused: {exc}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
