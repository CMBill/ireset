"""Supervise the W1 real small-set fit diagnostic from before uv until wait().

Run this standard-library-only launcher with the system Python; the actual
project process is always ``uv run --locked``.  Every emitted verdict is
provisional: a consumer must require this supervisor's successful exit, the
candidate verdict, the stdout observation, and the bound outer return
observation.  Nothing here starts a real-data stage or retries a failure;
the chain mode additionally refuses to launch without an authorization
path (the worker re-verifies its content independently).
"""

import time

_OUTER_T0 = time.perf_counter()  # before uv, project imports, and entry checks

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/run_w1_smallset_fit_real.py"
DEFAULT_R_ROOT = ROOT / "derived/audit/w1_smallset_fit_diagnostic/r_admission"
DEFAULT_CHAIN_ROOT = ROOT / "derived/experiments/omniretarget_first"
FAILURE_COST_ROUNDS = 4
FAILURE_COST_MIN_TAIL_S = 4.0

#: §8.1 per-stage caps the supervisor re-checks the stage seals against.
STAGE_CAPS = {"d1": 1800.0, "d6": 3600.0, "f": 600.0}
R_CAP = 120.0
CHAIN_CAP = 6120.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_hashes() -> dict[str, str]:
    files = sorted((ROOT / "src/ireset").rglob("*.py"))
    files += [WORKER, Path(__file__).resolve(), ROOT / "pyproject.toml",
              ROOT / "uv.lock"]
    return {str(path.relative_to(ROOT)): _sha(path) for path in files}


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as out:
        json.dump(obj, out, indent=1, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_incomplete_costed(path: Path, verdict: dict, *, t0: float,
                             wall_cap_s: float) -> None:
    """Withdraw with monotone cost accounting, including the withdrawal writes."""
    path = Path(path)
    doc = dict(verdict)
    sealed = doc.get("registered_measured_s")
    sealed = float(sealed) if _number(sealed) else None
    charge = sealed if sealed is not None else 0.0
    reserve = FAILURE_COST_MIN_TAIL_S
    doc.update(verdict="incomplete", sealed_registered_measured_s=sealed,
               cost_coverage_proven=False, cost_coverage_unproven=True,
               cost_coverage_status="pending_external_return",
               registered_cost_role="withdrawal_covering_reservation")
    observation_path = path.parent / "failure_cost_observation.json"

    for round_no in range(1, FAILURE_COST_ROUNDS + 1):
        pre = time.perf_counter() - t0
        charge = max(charge, pre + reserve)
        doc.update(registered_measured_s=charge, observed_elapsed_s=pre,
                   hard_deadline_exceeded=pre > wall_cap_s,
                   withdrawal_round=round_no)
        _write_json(path, doc)
        after_verdict = time.perf_counter() - t0
        if after_verdict > charge or (after_verdict > wall_cap_s
                                      and not doc["hard_deadline_exceeded"]):
            reserve = max(FAILURE_COST_MIN_TAIL_S, 2.0 * (after_verdict - pre) + 3.0)
            continue
        verdict_sha = _sha(path)
        before_observation = time.perf_counter() - t0
        if before_observation > charge or (before_observation > wall_cap_s
                                           and not doc["hard_deadline_exceeded"]):
            reserve = max(FAILURE_COST_MIN_TAIL_S, 2.0 * (before_observation - pre) + 3.0)
            continue
        observation = {
            "schema": "w1_real_fit_failure_cost_observation_v1",
            "launch_id": doc.get("launch_id"),
            "state": "pending_external_return", "cost_coverage_proven": False,
            "verdict_sha256": verdict_sha,
            "elapsed_after_verdict_write_s": after_verdict,
            "elapsed_before_observation_write_s": before_observation,
            "registered_measured_s": charge, "wall_cap_s": wall_cap_s,
            "hard_deadline_exceeded": doc["hard_deadline_exceeded"],
        }
        _write_json(observation_path, observation)
        after_observation = time.perf_counter() - t0
        if after_observation <= charge and not (after_observation > wall_cap_s
                                                and not doc["hard_deadline_exceeded"]):
            return
        reserve = max(FAILURE_COST_MIN_TAIL_S, 2.0 * (after_observation - pre) + 3.0)

    last = time.perf_counter() - t0
    doc.update(registered_measured_s=max(charge, last), observed_elapsed_s=last,
               hard_deadline_exceeded=last > wall_cap_s,
               cost_coverage_status="unproven_rounds_exhausted",
               withdrawal_rounds_exhausted=True)
    _write_json(path, doc)


def _attempt_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.removeprefix("attempt_")) for p in root.glob("attempt_*")
                if p.is_dir() and p.name.removeprefix("attempt_").isdigit()]
    start = max(existing, default=0) + 1
    for number in range(start, start + 1000):
        path = root / f"attempt_{number:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("no free attempt directory")


def _object_and_sha(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return data, hashlib.sha256(raw).hexdigest()


def _number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _proof_problems(summary_path: Path, proof_path: Path,
                    expected_t0: float | None, expected_cap: float
                    ) -> tuple[list[str], dict, dict, dict]:
    problems: list[str] = []
    try:
        summary, report_sha = _object_and_sha(summary_path)
        proof, proof_sha = _object_and_sha(proof_path)
        seal, budget = summary.get("seal"), summary.get("budget")
        if not isinstance(seal, dict) or not isinstance(budget, dict):
            raise ValueError("invalid seal or budget")
        reg, cap, elapsed = (seal.get("measured_wall_s_at_summary"),
                             seal.get("wall_cap_s"), proof.get("elapsed_s"))
        if (seal.get("status") != "sealed_within_budget"
                or seal.get("terminal") is not True
                or proof.get("schema") != "n0_return_observation_v1"
                or proof.get("covered") is not True):
            problems.append("non_consumable_seal_or_proof")
        t0 = seal.get("t0_perf_counter")
        if (not _number(t0) or (expected_t0 is not None and t0 != expected_t0)
                or cap != expected_cap
                or budget.get("t0_perf_counter") != t0
                or budget.get("wall_cap_s") != expected_cap
                or budget.get("measured_wall_s_at_summary") != reg
                or proof.get("registered_measured_s") != reg
                or proof.get("wall_cap_s") != cap):
            problems.append("budget_binding_mismatch")
        if not all(_number(v) for v in (reg, cap, elapsed)) or not elapsed <= reg <= cap:
            problems.append("in_process_coverage_invalid")
        if proof.get("summary_sha256") != report_sha:
            problems.append("summary_hash_mismatch")
        return problems, summary, proof, {"report_sha256": report_sha,
                                          "return_observation_sha256": proof_sha}
    except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        return ["report_or_proof_unreadable"], {}, {}, {}


def _matching_fields(left: dict, right: dict, fields: tuple[str, ...]) -> bool:
    return all(key in left and key in right and type(left[key]) is type(right[key])
               and left[key] == right[key] for key in fields)


def _r_summary_valid(summary: dict) -> bool:
    package = summary.get("package")
    return (summary.get("schema") == "w1_smallset_fit_r_admission_v1"
            and summary.get("admission_passed") is True
            and summary.get("problems") == []
            and isinstance(package, dict)
            and package.get("schema") == "w1_smallset_fit_admission_package_v1"
            and isinstance(package.get("windows"), list)
            and len(package.get("windows") or []) == 6
            and isinstance(package.get("d1_window_id"), str)
            and (summary.get("zero_declarations") or {}).get("array_value_reads") == 0
            and (summary.get("zero_declarations") or {}).get("checkpoint_reads") == 0
            and (summary.get("zero_declarations") or {}).get("cohort_created") is False
            and isinstance(package.get("call_matrix"), dict)
            and isinstance(package.get("runtime_binding"), dict)
            and isinstance(package.get("code_identity"), dict))


def _stage_report_matches(stage: str, report: dict, summary: dict) -> bool:
    """Require the proof-bound report and the chain summary to agree."""
    facts = summary.get(stage)
    if not isinstance(facts, dict):
        return False
    if stage in ("d1", "d6"):
        matrix, seal = report.get("call_matrix"), report.get("seal")
        mean = report.get("local_mean_fit_passed")
        materialization = report.get("materialization")
        return (report.get("schema") == "w1_smallset_fit_diagnostic_v1"
                and report.get("stage") == stage.upper()
                and report.get("local_complete_evidence") is True
                and isinstance(mean, bool)
                and report.get("local_conclusion") == ("complete_fit" if mean
                                                       else "fit_insufficient")
                and report.get("final_verification_status") == "pending_F"
                and report.get("problems") == []
                and report.get("failure", "missing") is None
                and isinstance(matrix, dict) and matrix.get("problems") == []
                and facts.get("call_matrix_problems") == matrix["problems"]
                and isinstance(seal, dict) and facts.get("seal") == seal.get("status")
                and isinstance(materialization, dict)
                and _matching_fields(facts, report,
                                     ("local_complete_evidence",
                                      "local_mean_fit_passed",
                                      "local_conclusion")))

    mean = report.get("mean_fit_passed")
    stages, copies = report.get("stages"), facts.get("stages")
    d6_ran = summary.get("d6_ran") is True
    expected = {"D1", "D6"} if d6_ran else {"D1"}
    if (report.get("schema") != "w1_smallset_fit_f_report_v1"
            or report.get("complete_evidence") is not True
            or report.get("replay_passed") is not True or not isinstance(mean, bool)
            or report.get("verdict") != ("fit_pass" if mean else "fit_insufficient")
            or report.get("final_verification_status") != "final_F"
            or report.get("tolerance_policy") != "bitwise_no_tolerance"
            or not _matching_fields(facts, report,
                                    ("complete_evidence", "replay_passed",
                                     "mean_fit_passed", "verdict"))
            or not isinstance(stages, list) or not isinstance(copies, list)
            or not stages or len(stages) != len(copies)
            or {s.get("stage") for s in stages if isinstance(s, dict)} != expected):
        return False
    for entry, copy in zip(stages, copies):
        if (not isinstance(entry, dict) or not isinstance(copy, dict)
                or not isinstance(entry.get("stage"), str)
                or entry.get("stage") != copy.get("stage")
                or entry.get("replay_enabled") is not True
                or entry.get("replay_passed") is not True
                or entry.get("complete_evidence") is not True
                or entry.get("local_reverify_problems") != []
                or entry.get("replay_problems") != []
                or entry.get("skip_reason") is not None
                or not _matching_fields(copy, entry,
                                        ("stage", "replay_enabled",
                                         "replay_passed", "replay_problems"))
                or copy.get("skip_reason") is not None):
            return False
        source = summary.get(str(entry.get("stage")).lower())
        if (not isinstance(source, dict)
                or not isinstance(entry.get("mean_fit_passed"), bool)
                or entry["mean_fit_passed"] is not source.get("local_mean_fit_passed")):
            return False
    return mean is all(entry["mean_fit_passed"] for entry in stages)


def _candidate_problems(doc: dict, launch_id: str, t0: float,
                        mode: str) -> list[str]:
    problems = []
    launch = doc.get("outer_launch")
    if (not isinstance(launch, dict) or launch.get("launch_id") != launch_id
            or launch.get("t0_perf_counter") != t0
            or launch.get("supervisor_declared") is not True
            or not _number(launch.get("cli_entry_perf_counter"))
            or launch["cli_entry_perf_counter"] < t0):
        problems.append("launch_identity_mismatch")
    if doc.get("label_verdict") != "pending_outer_return":
        problems.append("candidate_state_invalid")
    if mode == "r-admit":
        if not _r_summary_valid(doc):
            problems.append("r_admission_incomplete")
        return problems
    admission, authorization = doc.get("admission"), doc.get("authorization")
    if (not isinstance(admission, dict) or not admission.get("sha256")
            or not isinstance(authorization, dict)
            or not authorization.get("sha256")):
        problems.append("admission_or_authorization_binding_missing")
    d1, d6, handoff, f = (doc.get(k) for k in ("d1", "d6", "d1_handoff", "f"))
    if not all(isinstance(v, dict) for v in (d1, d6, handoff, f)):
        return problems + ["stage_facts_invalid"]
    if not isinstance(doc.get("authorized_includes"), list) or \
            not doc.get("authorized_includes"):
        return problems + ["authorized_includes_missing"]
    if doc.get("d6_ran") not in (True, False):
        return problems + ["d6_ran_flag_missing"]
    d6_ran = doc.get("d6_ran") is True
    if (d1.get("local_complete_evidence") is not True
            or d1.get("return_coverage") is not True
            or d1.get("call_matrix_problems") != []
            or f.get("complete_evidence") is not True
            or f.get("replay_passed") is not True
            or f.get("return_coverage") is not True):
        problems.append("stage_facts_incomplete")
    # D6 runs only when D1's mean gate passed, the handoff resolved and the
    # authorization includes D6 (plan §7.2/§7.4 item 3); any other D6 state
    # must be one of the recorded skip reasons.
    if d6_ran:
        if (d1.get("local_mean_fit_passed") is not True
                or handoff.get("handoff_ready") is not True
                or handoff.get("problems") != []
                or d6.get("local_complete_evidence") is not True
                or d6.get("return_coverage") is not True
                or d6.get("call_matrix_problems") != []):
            problems.append("stage_facts_incomplete")
    elif d6.get("skipped_reason") not in (
            "d1_incomplete", "d1_mean_fit_gate_not_passed",
            "d6_not_authorized", "handoff_not_ready"):
        problems.append("d6_state_invalid")
    if not d6_ran and d6.get("skipped_reason") == "d1_mean_fit_gate_not_passed" \
            and d1.get("local_mean_fit_passed") is not False:
        problems.append("d6_state_invalid")
    stages = f.get("stages")
    expected = {"D1", "D6"} if d6_ran else {"D1"}
    if (not isinstance(stages, list) or len(stages) != (2 if d6_ran else 1)
            or any(not isinstance(s, dict) for s in stages)
            or any(not isinstance(s.get("stage"), str) for s in stages)
            or {s.get("stage") for s in stages} != expected
            or any(s.get("replay_passed") is not True for s in stages)):
        problems.append("fresh_replay_incomplete")
    return problems


def run(args, *, t0: float = _OUTER_T0) -> int:
    if (not all(_number(v) for v in (args.wall_cap_s, args.reserve_s))
            or args.wall_cap_s <= args.reserve_s or args.reserve_s <= 0):
        raise ValueError("wall cap and reserve must be finite and 0 < reserve < cap")
    out_dir = _attempt_dir(Path(args.output_root).resolve())
    launch_id = uuid.uuid4().hex
    before = _source_hashes()
    if args.mode == "chain":
        expected_cap = args.wall_cap_s
        if not args.admission or not args.authorization:
            raise ValueError("chain mode requires --admission and --authorization")
        command = ["uv", "run", "--locked", "python", str(WORKER),
                   "--mode", "chain", "--output-dir", str(out_dir),
                   "--admission", str(Path(args.admission).resolve()),
                   "--authorization", str(Path(args.authorization).resolve())]
        if args.impl_sha:
            command += ["--impl-sha", args.impl_sha]
        if args.fixture:
            command += ["--fixture"]
    else:
        expected_cap = args.wall_cap_s
        command = ["uv", "run", "--locked", "python", str(WORKER),
                   "--mode", "r-admit", "--output-dir", str(out_dir)]
    command += ["--outer-t0-hex", t0.hex(), "--launch-id", launch_id,
                "--wall-cap-s", str(args.wall_cap_s),
                "--reserve-s", str(args.reserve_s)]
    launch = {"schema": "w1_real_fit_outer_launch_v1", "state": "running",
              "mode": args.mode, "launch_id": launch_id,
              "t0_perf_counter": t0, "wall_cap_s": args.wall_cap_s,
              "reserve_s": args.reserve_s, "command": command,
              "source_sha256": before}
    launch_path = out_dir / "outer_launch.json"
    _write_json(launch_path, launch)
    verdict_path = out_dir / "acceptance_verdict.json"
    try:
        print(f"Real-fit attempt ({args.mode}): {out_dir}", flush=True)
    except OSError:
        _write_incomplete_costed(verdict_path, {
            "schema": "w1_real_fit_outer_verdict_v1", "mode": args.mode,
            "launch_id": launch_id, "registered_measured_s": None,
            "wall_cap_s": args.wall_cap_s,
            "problems": ["launcher_stdout_write_failed"]}, t0=t0,
            wall_cap_s=args.wall_cap_s)
        return 3
    if time.perf_counter() >= t0 + args.wall_cap_s - args.reserve_s:
        _write_incomplete_costed(verdict_path, {
            "schema": "w1_real_fit_outer_verdict_v1", "mode": args.mode,
            "launch_id": launch_id, "registered_measured_s": None,
            "wall_cap_s": args.wall_cap_s,
            "problems": ["deadline_before_uv"]}, t0=t0,
            wall_cap_s=args.wall_cap_s)
        return 3

    stdout_path, stderr_path = out_dir / "worker_stdout.log", out_dir / "worker_stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        timed_out = False
        try:
            rc = process.wait(timeout=max(0.0, t0 + args.wall_cap_s - time.perf_counter()))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                rc = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rc = process.wait()
        wait_time = time.perf_counter()  # immediately after the actual wait returns
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())

    if args.mode == "r-admit":
        summary_name = "r_admission.json"
    else:
        summary_name = "acceptance_summary.json"
    problems, summary, proof, summary_binding = _proof_problems(
        out_dir / summary_name, out_dir / "return_observation.json",
        t0, expected_cap)
    problems += _candidate_problems(summary, launch_id, t0, args.mode)
    if rc != 0:
        problems.append(f"worker_exit:{rc}")
    if timed_out:
        problems.append("worker_hard_deadline_timeout")
    after = _source_hashes()
    if before != after:
        problems.append("implementation_changed_during_run")
    seal = summary.get("seal") if isinstance(summary.get("seal"), dict) else {}
    registered = seal.get("measured_wall_s_at_summary")
    wait_elapsed = wait_time - t0
    in_process = proof.get("elapsed_s")
    if (not _number(registered) or not _number(in_process)
            or not in_process <= wait_elapsed <= registered <= args.wall_cap_s):
        problems.append("actual_wait_not_covered")
    stage_artifacts = {}
    if args.mode == "chain":
        d6_ran = summary.get("d6_ran") is True
        stage_dirs = [("d1", "stage_report.json"),
                      ("f", "f_report.json")]
        if d6_ran:
            stage_dirs.insert(1, ("d6", "stage_report.json"))
        for subdir, filename in stage_dirs:
            try:
                stage_problems, stage_doc, stage_proof, stage_binding = \
                    _proof_problems(out_dir / subdir / filename,
                                    out_dir / subdir / "return_observation.json",
                                    None, STAGE_CAPS[subdir])
                problems += [f"{subdir}:{p}" for p in stage_problems]
                stage_artifacts[subdir] = stage_binding
                if not _stage_report_matches(subdir, stage_doc, summary):
                    problems.append(f"{subdir}:report_facts_mismatch")
                stage_seal = stage_doc["seal"]
                stage_t0 = stage_seal["t0_perf_counter"]
                facts = summary.get(subdir)
                returned = (facts.get("stage_returned_elapsed_s")
                            if isinstance(facts, dict) else None)
                observed = stage_proof.get("elapsed_s")
                if (not _number(returned) or not _number(observed)
                        or returned > observed
                        or not _number(stage_t0) or not t0 <= stage_t0 <= wait_time
                        or stage_t0 + observed > wait_time):
                    problems.append(f"{subdir}:actual_stage_return_not_covered")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                problems.append(f"{subdir}:report_or_proof_unreadable")
        # the admission/authorization bytes the chain consumed must be the
        # ones the supervisor launched with
        try:
            adm_sha = _sha(Path(args.admission).resolve())
            auth_sha = _sha(Path(args.authorization).resolve())
            if (summary.get("admission") or {}).get("sha256") != adm_sha \
                    or (summary.get("authorization") or {}).get("sha256") != auth_sha:
                problems.append("admission_or_authorization_sha_mismatch")
        except OSError:
            problems.append("admission_or_authorization_unreadable")

    receipt = {"schema": "w1_real_fit_outer_return_observation_v1",
               "mode": args.mode, "launch_id": launch_id,
               "t0_perf_counter": t0, "uv_pid": process.pid,
               "wait_return_code": rc, "worker_hard_deadline_timeout": timed_out,
               "wait_return_perf_counter": wait_time,
               "wait_elapsed_s": wait_elapsed,
               "registered_measured_s": registered,
               "wall_cap_s": args.wall_cap_s,
               "in_process_elapsed_s": in_process,
               "summary_sha256": summary_binding.get("report_sha256"),
               "in_process_proof_sha256": summary_binding.get("return_observation_sha256"),
               "stage_artifacts": stage_artifacts,
               "launch_sha256": _sha(launch_path),
               "stdout_sha256": _sha(stdout_path),
               "stderr_sha256": _sha(stderr_path),
               "source_sha256_after": after,
               "problems": problems}
    receipt_path = out_dir / "outer_return_observation.json"
    _write_json(receipt_path, receipt)
    after_receipt = time.perf_counter() - t0
    if not _number(registered) or after_receipt > registered or after_receipt > args.wall_cap_s:
        problems.append("receipt_write_not_covered")
    verdict = {"schema": "w1_real_fit_outer_verdict_v1", "mode": args.mode,
               "launch_id": launch_id,
               "verdict": "candidate_complete" if not problems else "incomplete",
               "summary_sha256": receipt["summary_sha256"],
               "outer_return_observation_sha256": _sha(receipt_path),
               "receipt_write_completed_elapsed_s": after_receipt,
               "registered_measured_s": registered,
               "wall_cap_s": args.wall_cap_s,
               "problems": problems}
    if problems:
        _write_incomplete_costed(verdict_path, verdict, t0=t0,
                                 wall_cap_s=args.wall_cap_s)
        return 3
    _write_json(verdict_path, verdict)
    verdict_sha = _sha(verdict_path)
    after_verdict = time.perf_counter() - t0
    if not _number(registered) or after_verdict > registered or after_verdict > args.wall_cap_s:
        verdict.update(problems=problems + ["verdict_write_not_covered"])
        _write_incomplete_costed(verdict_path, verdict, t0=t0,
                                 wall_cap_s=args.wall_cap_s)
        return 3
    event = json.dumps({"event": "w1_real_fit_outer_verdict_output_pending",
                        "mode": args.mode, "verdict": "pending_output_coverage",
                        "output_dir": str(out_dir), "launch_id": launch_id,
                        "verdict_sha256": verdict_sha,
                        "elapsed_s": after_verdict,
                        "registered_measured_s": registered,
                        "wall_cap_s": args.wall_cap_s})
    try:
        print(event, flush=True)
    except OSError:
        verdict.update(problems=["stdout_write_failed"])
        _write_incomplete_costed(verdict_path, verdict, t0=t0,
                                 wall_cap_s=args.wall_cap_s)
        return 3
    after_stdout = time.perf_counter() - t0
    if after_stdout > registered or after_stdout > args.wall_cap_s:
        verdict.update(problems=["stdout_not_covered"])
        _write_incomplete_costed(verdict_path, verdict, t0=t0,
                                 wall_cap_s=args.wall_cap_s)
        return 3
    output_observation = {
        "schema": "w1_real_fit_outer_stdout_observation_v1",
        "state": "pending_supervisor_return", "mode": args.mode,
        "launch_id": launch_id, "verdict_sha256": verdict_sha,
        "event_sha256": hashlib.sha256((event + "\n").encode("utf-8")).hexdigest(),
        "stdout_flushed_elapsed_s": after_stdout,
        "registered_measured_s": registered, "wall_cap_s": args.wall_cap_s,
    }
    try:
        _write_json(out_dir / "outer_stdout_observation.json", output_observation)
    except OSError:
        verdict.update(problems=["stdout_observation_write_failed"])
        _write_incomplete_costed(verdict_path, verdict, t0=t0,
                                 wall_cap_s=args.wall_cap_s)
        return 3
    after_observation = time.perf_counter() - t0
    if after_observation > registered or after_observation > args.wall_cap_s:
        verdict.update(problems=["stdout_observation_write_not_covered"])
        _write_incomplete_costed(verdict_path, verdict, t0=t0,
                                 wall_cap_s=args.wall_cap_s)
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("r-admit", "chain"), default="r-admit")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--admission", default=None)
    parser.add_argument("--authorization", default=None)
    parser.add_argument("--impl-sha", default=None)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--wall-cap-s", type=float, default=None)
    parser.add_argument("--reserve-s", type=float, default=None)
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = (DEFAULT_R_ROOT if args.mode == "r-admit"
                            else DEFAULT_CHAIN_ROOT)
    if args.wall_cap_s is None:
        args.wall_cap_s = R_CAP if args.mode == "r-admit" else CHAIN_CAP
    if args.reserve_s is None:
        args.reserve_s = 15.0 if args.mode == "r-admit" else 120.0
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"real-fit supervisor failed: {exc}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
