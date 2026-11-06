"""W1 real training-parameter update audit entry (Gate A/B wiring).

Modes:

* ``--synthetic`` -- pure analytic R0->P->I->F self-test.
* ``--calibrate-synthetic`` -- write/verify the frozen synthetic calibration
  and its recomputation report in a fresh directory.
* ``--admission`` -- synthetic-schema read-only admission package.
* ``--real-admission`` -- metadata-only real admission package (new schema).
* ``--real`` -- registered real R0->P->I->F; refused unless a hash-bound
  section 10.2 authorization, the frozen calibration/fp32 contract, a passed
  real admission package and a fresh run root are supplied.
* ``--real --resume --stage <R0|P|I|F>`` -- append one attempt to an existing
  same-cohort run after all history was proven settled.

This repository state has no authorization record, so real execution remains
unreachable until the user provides one.
"""

from __future__ import annotations

import os
import time

_ENTRY_STARTED_PERF = time.perf_counter()

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Minimal launch bootstrap (stdlib only, before the project/numeric imports)
# ---------------------------------------------------------------------------
# The frozen migration plan (§6.1) requires the one-shot claim to be persisted
# before uv/project imports and before any substantive precheck.  This block
# deliberately duplicates only the claim writer; the audit module remains the
# authority for validation and for every later record.
_LAUNCH_CLAIM_SCHEMA = "w1_real_training_update_launch_claim_v1"
_LAUNCH_COHORT_ID = "w1_real_training_update_audit_v1"
_LAUNCH_EXECUTION_ID = "w1_real_training_update_audit_v1"
_LAUNCH_HISTORY_ROOT = Path("derived/experiments/omniretarget_first")
_LAUNCH_TARGET = {"stage": "R0", "attempt": "attempt_002",
                  "execution_seq": 3}


def _flag_present(flag: str, argv: list[str]) -> bool:
    return any(token == flag or token.startswith(flag + "=")
               for token in argv)


def _flag_value(flag: str, argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == flag:
            return argv[index + 1] if index + 1 < len(argv) else None
        if token.startswith(flag + "="):
            return token[len(flag) + 1:]
    return None


def _hash_path(path: Path | None) -> str:
    if path is None or not Path(path).is_file():
        return "0" * 64
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_optional_json(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_minimal_launch_claim(
        *, root: Path, invocation_id: str, authorization_sha256: str,
        spec_sha256: str, legacy_head_sha256: str, marker_sha256: str,
        target: Mapping[str, Any], t0_perf_counter: float,
        closed_history_head_sha256: str | None = None) -> Path:
    """Persist ``claim.json`` before any substantive precheck.

    The write is byte-for-byte the same shape as ``claim_launch`` and uses the
    same temp + link + directory-fsync publication, so a crash leaves no
    overwrite and an observable ``.tmp`` sibling.
    """
    directory = Path(root) / "launch_journal" / invocation_id
    path = directory / "claim.json"
    if path.exists():
        return path
    payload = {
        "schema": _LAUNCH_CLAIM_SCHEMA,
        "invocation_id": invocation_id,
        "cohort_id": _LAUNCH_COHORT_ID,
        "execution_id": _LAUNCH_EXECUTION_ID,
        "run_root_realpath": str(Path(root).resolve()),
        "authorization_sha256": authorization_sha256,
        "spec_sha256": spec_sha256,
        "legacy_head_sha256": legacy_head_sha256,
        "marker_sha256": marker_sha256,
        **({"closed_history_head_sha256": closed_history_head_sha256}
           if closed_history_head_sha256 is not None else {}),
        "t0_perf_counter": float(t0_perf_counter),
        "target": dict(target),
        "pid": int(os.getpid()),
        "ppid": int(os.getppid()),
        "executable": sys.executable,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    directory.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return path
        directory_fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return path


def _bootstrap_real_migration_launch(
        argv: list[str] | None = None,
        base: Path | None = None) -> Path | None:
    """Write the fixed-root claim before importing the project.

    Only a fresh real migration launch is eligible.  Resume, stage replay and
    worker processes must not consume another one-shot invocation.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(_flag_present(flag, argv) for flag in (
            "--migration-worker", "--retest-worker", "--synthetic-worker",
            "--resume", "--stage")):
        return None
    if not _flag_present("--real", argv) or not (
            _flag_present("--migration", argv)
            or _flag_present("--retest", argv)):
        return None
    out_value = _flag_value("--out", argv)
    invocation_id = _flag_value("--invocation-id", argv)
    authorization_value = _flag_value("--authorization", argv)
    spec_value = _flag_value("--binding-spec", argv)
    if not out_value or not invocation_id or not authorization_value or \
            not spec_value or Path(invocation_id).name != invocation_id or \
            invocation_id in (".", ".."):
        return None
    out = Path(out_value).expanduser().resolve()
    fixed_base = (Path(base) if base is not None
                  else ROOT / _LAUNCH_HISTORY_ROOT).resolve()
    if out.parent != fixed_base:
        return None
    spec_path = Path(spec_value).expanduser()
    spec_path = (spec_path.resolve() if spec_path.is_file() else None)
    spec = _read_optional_json(spec_path)
    snapshot_path = None
    snapshot = None
    if isinstance(spec, dict):
        reference = spec.get("history_snapshot")
        if isinstance(reference, dict):
            raw = reference.get("path")
            if isinstance(raw, str) and raw:
                snapshot_path = Path(raw)
                if not snapshot_path.is_absolute():
                    snapshot_path = ROOT / snapshot_path
                snapshot_path = snapshot_path.expanduser().resolve()
                snapshot = _read_optional_json(snapshot_path)
    legacy = (snapshot or {}).get("legacy_head_sha256")
    marker = (snapshot or {}).get("marker_sha256")
    target = _LAUNCH_TARGET
    if isinstance(spec, dict):
        plan = spec.get("stage_plan")
        if isinstance(plan, list) and plan and isinstance(plan[0], dict):
            target = plan[0]
    authorization_path = Path(authorization_value).expanduser()
    authorization_path = (authorization_path.resolve()
                          if authorization_path.is_file() else None)
    try:
        return _write_minimal_launch_claim(
            root=out, invocation_id=invocation_id,
            authorization_sha256=_hash_path(authorization_path),
            spec_sha256=_hash_path(spec_path),
            legacy_head_sha256=(legacy if isinstance(legacy, str)
                                and len(legacy) == 64 else "0" * 64),
            marker_sha256=(marker if isinstance(marker, str)
                           and len(marker) == 64 else "0" * 64),
            target=target, t0_perf_counter=_ENTRY_STARTED_PERF,
            closed_history_head_sha256=(
                (snapshot or {}).get("closed_history_head_sha256")
                if isinstance(snapshot, dict) else None))
    except OSError as exc:
        print(f"[update-audit] launch claim bootstrap failed: "
              f"{type(exc).__name__}", file=sys.stderr)
        return None


_BOOTSTRAP_CLAIM = _bootstrap_real_migration_launch()

from ireset.eval.w1_real_training_update_audit import (  # noqa: E402
    ACCOUNTING_SCHEMA_V2,
    ADMISSION_SCHEMA,
    AUDIT_SCHEMA,
    AUTHORIZATION_REQUIRED,
    BINDINGS_DIR,
    COHORT_ID,
    EXECUTION_EVENTS_DIR,
    HISTORY_ROOT,
    LAUNCH_JOURNAL_DIR,
    MODEL_CONFIG,
    STAGE_BUDGETS,
    admission_package_problems,
    authorization_problems,
    build_admission_package,
    build_default_tolerance_contract,
    build_binding_spec,
    build_migration_admission,
    build_migration_synthetic_acceptance,
    build_ordinary_synthetic_acceptance,
    build_retest_synthetic_acceptance,
    build_retest_binding_spec,
    build_retest_authorization_draft,
    build_retest_history_snapshot,
    build_retest_admission,
    build_budget_revision,
    build_closed_history_head,
    budget_revision_problems,
    claim_launch,
    compact_migration_result,
    configure_production_runtime,
    implementation_files_sha256,
    implementation_identity,
    launch_claim_record,
    load_authorization,
    load_external_binding_material,
    load_index,
    load_tolerance_contract,
    prepare_migration_synthetic_launch,
    record_return_observation,
    run_binding_migration_audit,
    run_registered_audit,
    scan_launch_journals,
    sha256_file,
    stage_binding,
    strict_read_json,
    SyntheticTrainingProvider,
    synthetic_calibration,
    synthetic_outer_return_record,
    synthetic_self_test,
    tolerance_contract_problems,
    write_history_snapshot,
    write_json,
    write_json_exclusive,
    write_migration_admission,
    write_migration_baseline,
    write_package,
)
from ireset.paths import path_for_record  # noqa: E402

_EXIT = {
    "synthetic_contract_passed": 0,
    "real_training_update_covered_contract_passed": 0,
    "real_training_update_mismatch": 2,
    "audit_incomplete": 3,
}


def _guard(out: Path) -> Path:
    resolved = Path(out).expanduser().resolve()
    experiments = (ROOT / "derived" / "experiments").resolve()
    if resolved == experiments or experiments in resolved.parents:
        raise SystemExit(
            f"refusing to write I/C artifacts into a research experiment root: "
            f"{resolved}")
    for name in ("resources", "third_party", "src", "scripts", "tests"):
        forbidden = (ROOT / name).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise SystemExit(f"refusing to write under {name}/: {resolved}")
    return resolved


def _print_result(result: dict) -> None:
    print(f"[update-audit] schema={AUDIT_SCHEMA} cohort={COHORT_ID}")
    print(f"[update-audit] mode={result.get('mode')} "
          f"conclusion={result.get('conclusion')}")
    print(f"[update-audit] complete_evidence={result.get('complete_evidence')} "
          f"covered_contract_passed={result.get('covered_contract_passed')}")
    for name in ("problems", "failures", "gaps", "verification_problems"):
        values = result.get(name) or []
        if values:
            print(f"[update-audit] {name}={len(values)}")
            for value in values[:20]:
                print(f"[update-audit]   {value}")
    counts = (result.get("calls") or {}).get("counts")
    if counts:
        print(f"[update-audit] calls={ {k: v for k, v in counts.items() if v} }")
    print(f"[update-audit] training_updates={result.get('training_updates')} "
          f"scope={result.get('training_updates_scope')}")
    print(f"[update-audit] selected_attempts={result.get('selected_attempts')}")
    print(f"[update-audit] selected_chain_training_updates="
          f"{result.get('selected_chain_training_updates')}")
    if result.get("summary_path"):
        print(f"[update-audit] summary={result['summary_path']}")


def _resolve_tolerance_for_admission(args) -> tuple[dict, Path, str]:
    path = (args.tolerance_contract or
            ROOT / "derived/audit/w1_real_training_update_audit/"
                   "tolerance_contract.json")
    contract, sha, problems = load_tolerance_contract(
        path, recompute=False)
    if problems:
        raise SystemExit(f"tolerance contract refused: {problems}")
    if contract.get("mode") != "synthetic_calibrated":
        raise SystemExit("run --calibrate-synthetic first")
    return contract, path, sha


def _calibration_verification_path(args,
                                   contract_path: Path) -> Path | None:
    if args.calibration_verification is not None:
        return Path(args.calibration_verification).expanduser().resolve()
    candidate = contract_path.parent / "calibration_verification.json"
    return candidate if candidate.is_file() else None


def _load_calibration_verification(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise SystemExit(f"calibration verification unreadable: {path}")
    from ireset.eval.w1_real_training_update_provider import (
        calibration_verification_problems,
    )
    problems = calibration_verification_problems(record)
    if problems:
        raise SystemExit(f"calibration verification refused: {problems}")
    return record


def _load_synthetic_config(args) -> dict | None:
    if args.synthetic_config_json is None:
        return None
    path = Path(args.synthetic_config_json).expanduser().resolve()
    try:
        record = strict_read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"synthetic config unreadable: {type(exc).__name__}")
    if not isinstance(record, dict):
        raise SystemExit("synthetic config must be a JSON object")
    return record


def _spawn_worker(command: list[str], log_path: Path) -> dict:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = float(time.perf_counter())
    with open(log_path, "wb") as log:
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ))
        pid = int(process.pid)
        exit_code = int(process.wait())
    finished = float(time.perf_counter())
    return {"pid": pid, "ppid": os.getpid(), "exit_code": exit_code,
            "signal": (-exit_code if exit_code < 0 else None),
            "started_perf_counter": started,
            "finished_perf_counter": finished,
            "log_path": path_for_record(log_path)}


def _write_launch_failure(directory: Path, facts: Mapping[str, Any],
                          reason: str) -> Path:
    path = Path(directory) / "failure.json"
    record = {
        "schema": "w1_real_training_update_launch_failure_v1",
        "kind": "failure",
        "reason": reason,
        "observed": dict(facts),
        "claimed_wall_s": max(0.0, float(
            facts.get("finished_perf_counter", 0.0))
            - float(facts.get("started_perf_counter", 0.0))),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_exclusive(path, record)
    return path


def _history_root() -> Path:
    return (ROOT / HISTORY_ROOT).resolve()


def _write_precheck_failure(root: Path, invocation_id: str, t0: float,
                            reason: str) -> Path | None:
    """Leave a billed liability when a substantive precheck refuses.

    The claim is already persisted by the minimal launcher, so a precheck
    failure must not disappear as a clean zero-record journal.
    """
    directory = Path(root) / LAUNCH_JOURNAL_DIR / invocation_id
    claim_path = directory / "claim.json"
    if not claim_path.is_file():
        return None
    try:
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(claim, dict) or claim.get("pid") != os.getpid() or \
            claim.get("t0_perf_counter") != float(t0):
        return None
    finished = float(time.perf_counter())
    facts = {
        "pid": os.getpid(), "ppid": os.getppid(), "exit_code": 1,
        "signal": None, "started_perf_counter": float(t0),
        "finished_perf_counter": finished, "log_path": None,
    }
    try:
        return _write_launch_failure(directory, facts, reason)
    except FileExistsError:
        return directory / "failure.json"
    except OSError as exc:
        print(f"[update-audit] precheck failure record could not be written: "
              f"{type(exc).__name__}", file=sys.stderr)
        return None


def _worker_result(path: Path) -> dict | None:
    if not Path(path).is_file():
        return None
    try:
        record = strict_read_json(path)
    except (OSError, ValueError, TypeError):
        return None
    return record if isinstance(record, dict) else None


def _fresh_migration_root_problems(
        root: Path, *, allow_invocation: str | None = None) -> list[str]:
    problems: list[str] = []
    for name in (BINDINGS_DIR, EXECUTION_EVENTS_DIR):
        if (Path(root) / name).exists():
            problems.append(f"migration_root_has_{name}")
    journal = Path(root) / LAUNCH_JOURNAL_DIR
    if journal.exists():
        allowed = False
        if journal.is_dir() and allow_invocation:
            try:
                entries = sorted(journal.iterdir())
            except OSError:
                entries = []
            if len(entries) == 1 and entries[0].name == allow_invocation \
                    and entries[0].is_dir():
                try:
                    children = sorted(entries[0].iterdir())
                except OSError:
                    children = []
                allowed = (
                    len(children) == 1
                    and children[0].name == "claim.json"
                    and children[0].is_file())
        if not allowed:
            problems.append("migration_root_has_launch_journal")
    try:
        rows = load_index(Path(root))
    except (OSError, ValueError, TypeError):
        rows = []
    if any(row.get("accounting_schema") == ACCOUNTING_SCHEMA_V2
           for row in rows):
        problems.append("migration_root_has_v2_rows")
    return problems


def _record_worker_return(root: Path, invocation_id: str, result: dict,
                          facts: dict) -> tuple[Path | None, str | None,
                                                list[str]]:
    commits = result.get("commits") or {}
    candidates = result.get("candidates") or {}
    timings = result.get("stage_timings") or {}
    write_after = result.get("write_after") or {}
    stages = set(STAGE_BUDGETS)
    if set(commits) != stages or set(candidates) != stages or \
            set(timings) != stages:
        return None, None, ["worker_return_scope"]
    last_end = result.get("last_stage_end_perf_counter")
    if last_end is None:
        last_end = timings["F"].get("end_perf_counter")
    try:
        claim = strict_read_json(
            Path(root) / LAUNCH_JOURNAL_DIR / invocation_id / "claim.json")
        started = float(claim["t0_perf_counter"])
        return record_return_observation(
            root, invocation_id=invocation_id, pid=facts["pid"],
            ppid=facts["ppid"], exit_code=facts["exit_code"],
            signal=facts["signal"],
            started_perf_counter=started,
            finished_perf_counter=float(facts["finished_perf_counter"]),
            last_stage_end_perf_counter=float(last_end),
            commits=commits, candidate_receipts=candidates,
            write_after={**{stage: True for stage in STAGE_BUDGETS},
                         "terminal": bool(write_after.get("terminal", True))},
            stage_timings=timings, source_kind="outer_wait",
            expected_exit_code=facts["exit_code"],
            conclusion=result.get("conclusion"))
    except (OSError, ValueError, TypeError) as exc:
        return None, None, [f"worker_return:{type(exc).__name__}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true",
                      help="run the analytic R0->P->I->F self-test")
    mode.add_argument("--calibrate-synthetic", action="store_true",
                      help="write/verify the frozen synthetic calibration")
    mode.add_argument("--admission", action="store_true",
                      help="write the synthetic-schema read-only admission "
                           "package")
    mode.add_argument("--real-admission", action="store_true",
                      help="write the metadata-only real admission package")
    mode.add_argument("--migration-snapshot", action="store_true",
                      help="freeze the original v1 prefix and migration "
                           "metadata outside the run root")
    mode.add_argument("--migration-admission", action="store_true",
                      help="write the read-only binding migration admission")
    mode.add_argument("--migration-authorization-draft", action="store_true",
                      help="write the non-authorized v2 migration draft")
    mode.add_argument("--migration-synthetic", action="store_true",
                      help="run the synthetic second-generation chain and "
                           "write its migration acceptance")
    mode.add_argument("--retest-synthetic", action="store_true",
                      help="run the synthetic third-generation retest chain "
                           "and write its acceptance")
    mode.add_argument("--retest-chain-worker", action="store_true",
                      help=argparse.SUPPRESS)
    mode.add_argument("--migration-spec", action="store_true",
                      help="freeze the read-only binding spec outside the root")
    mode.add_argument("--retest-materials", action="store_true",
                      help="build the read-only generation_003 retest "
                           "snapshot/spec/admission/authorization draft")
    mode.add_argument("--gate-b-v2", action="store_true",
                      help="bind the ordinary and migration synthetic "
                           "acceptances into Gate B v2")
    mode.add_argument("--migration-worker", action="store_true",
                      help=argparse.SUPPRESS)
    mode.add_argument("--retest-worker", action="store_true",
                      help=argparse.SUPPRESS)
    mode.add_argument("--synthetic-worker", action="store_true",
                      help=argparse.SUPPRESS)
    mode.add_argument("--real", action="store_true",
                      help="registered real audit (authorization + admission)")
    ap.add_argument("--job", type=Path, default=None)
    ap.add_argument("--synthetic-config-json", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tolerance-out", type=Path, default=None)
    ap.add_argument("--tolerance-contract", type=Path, default=None)
    ap.add_argument("--authorization", type=Path, default=None)
    ap.add_argument("--admission-package", type=Path, default=None)
    ap.add_argument("--synthetic-acceptance", type=Path, default=None)
    ap.add_argument("--ordinary-acceptance", type=Path, default=None)
    ap.add_argument("--migration-acceptance", type=Path, default=None)
    ap.add_argument("--retest-acceptance", type=Path, default=None)
    ap.add_argument("--retest-admission", type=Path, default=None)
    ap.add_argument("--calibration-verification", type=Path, default=None)
    ap.add_argument("--migration-root", type=Path, default=None,
                    help="original v1 cohort run root (never written)")
    ap.add_argument("--history-snapshot", type=Path, default=None)
    ap.add_argument("--binding-spec", type=Path, default=None)
    ap.add_argument("--gate-b", type=Path, default=None)
    ap.add_argument("--migration-admission-package", type=Path, default=None)
    ap.add_argument("--calibration-attempt", type=str, default=None)
    ap.add_argument("--synthetic-attempt", type=str, default=None)
    ap.add_argument("--admission-attempt", type=str, default=None)
    ap.add_argument("--invocation-id", type=str, default=None)
    ap.add_argument("--migration", action="store_true",
                    help="use the dedicated second-generation binding entry")
    ap.add_argument("--retest", action="store_true",
                    help="use the third-generation gradient-numerics retest "
                         "entry")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage", type=str, default=None,
                    choices=("R0", "P", "I", "F"))
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--device", type=str, default="auto",
                    choices=("auto", "cpu", "cuda"))
    args = ap.parse_args()

    if args.synthetic_worker:
        if args.out is None or args.tolerance_contract is None:
            raise SystemExit(
                "--synthetic-worker requires --out and --tolerance-contract")
        contract, _sha, problems = load_tolerance_contract(
            args.tolerance_contract, recompute=False)
        if problems:
            raise SystemExit(f"tolerance contract refused: {problems}")
        device = args.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        result = synthetic_self_test(
            args.out, seed=args.seed, device=device, tolerance=contract,
            config=_load_synthetic_config(args))
        _print_result(result)
        return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)

    if args.retest_chain_worker:
        if args.job is None:
            raise SystemExit("--retest-chain-worker requires --job")
        from ireset.eval.w1_real_training_update_audit import (
            run_retest_chain,
        )
        job = strict_read_json(Path(args.job).expanduser().resolve())
        contract, _sha, problems = load_tolerance_contract(
            Path(str(job["tolerance_contract_path"])).expanduser().resolve(),
            recompute=False)
        if problems:
            raise SystemExit(f"worker tolerance contract refused: {problems}")
        device = job.get("device") or "cuda"
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        provider = SyntheticTrainingProvider(
            dict(job.get("config") or MODEL_CONFIG),
            seed=int(job.get("seed", 20261029)), device=device)
        root = Path(str(job["launch"]["root"])).expanduser().resolve()
        fixture = root / "fixture"
        if not (fixture / "fixture.json").is_file():
            provider.prepare_fixture(fixture)
        provider.load_fixture()
        if provider.fixture.get("problems"):
            raise SystemExit(f"worker fixture refused: "
                             f"{provider.fixture['problems']}")
        result = run_retest_chain(
            base=Path(str(job["base"])), provider=provider,
            tolerance=contract, v2_launch=job["launch"],
            v3_invocation=str(job.get("v3_invocation") or "inv-retest-001"))
        write_json(Path(str(job["result_path"])), {
            "result": result, "roster": provider.roster})
        conclusion = (result.get("v3") or {}).get(
            "conclusion", "audit_incomplete")
        print(f"[update-audit] retest chain v3 conclusion={conclusion}")
        return _EXIT.get(conclusion, 3)

    if args.migration_worker or args.retest_worker:
        retest_worker = bool(args.retest_worker)
        if args.job is None:
            raise SystemExit("worker requires --job")
        job = strict_read_json(Path(args.job).expanduser().resolve())
        if not isinstance(job, Mapping):
            raise SystemExit("worker job must be an object")
        root = Path(str(job.get("root"))).expanduser().resolve()
        spec_path = Path(str(job.get("spec_path"))).expanduser().resolve()
        authorization, auth_sha, auth_problems = load_authorization(
            Path(str(job.get("authorization_path"))).expanduser().resolve())
        if auth_problems:
            raise SystemExit(f"worker authorization refused: {auth_problems}")
        contract, _contract_sha, contract_problems = load_tolerance_contract(
            Path(str(job.get("tolerance_contract_path"))).expanduser().resolve(),
            recompute=False)
        if contract_problems:
            raise SystemExit(
                f"worker tolerance contract refused: {contract_problems}")
        invocation_id = job.get("invocation_id")
        result_path = Path(str(job.get("result_path"))).expanduser().resolve()
        mode = job.get("mode")
        if mode == "real":
            from ireset.eval.w1_real_training_update_provider import (
                REAL_ADMISSION_SCHEMA,
                RealTrainingProvider,
                real_admission_package_problems,
            )
            package_path = Path(
                str(job.get("admission_package_path"))).expanduser().resolve()
            package = strict_read_json(package_path)
            package_sha = sha256_file(package_path)
            if package.get("schema") == REAL_ADMISSION_SCHEMA:
                package_problems = real_admission_package_problems(package)
            else:
                package_problems = admission_package_problems(package)
            if package_problems:
                raise SystemExit(f"worker admission refused: {package_problems}")
            acceptance_path = Path(package.get("acceptance_path") or "")
            if not acceptance_path.is_absolute():
                acceptance_path = ROOT / acceptance_path
            provider = RealTrainingProvider(
                ROOT, MODEL_CONFIG, device="cuda",
                acceptance_path=(acceptance_path if acceptance_path.is_file()
                                 else None))
            provider.acceptance = package.get("acceptance")
            provider.acceptance_sha = package.get("acceptance_sha256")
            provider.roster = ((package.get("roster") or {}).get(
                "parameter_roster"))
            retest_package = None
            retest_sha = None
            if retest_worker:
                from ireset.eval.w1_real_training_update_audit import (
                    retest_admission_problems,
                )
                retest_path = Path(str(
                    job.get("retest_admission_path"))).expanduser().resolve()
                if not retest_path.is_file():
                    raise SystemExit("worker retest admission missing")
                retest_sha = sha256_file(retest_path)
                retest_package = strict_read_json(retest_path)
                admission_errors = retest_admission_problems(
                    retest_package, project_root=ROOT, strict=True)
                if admission_errors:
                    raise SystemExit(
                        f"worker retest admission refused: {admission_errors}")
            result = run_binding_migration_audit(
                root, provider, mode="real", tolerance=contract,
                authorization_sha=auth_sha, authorization=authorization,
                admission_package=package, admission_sha=package_sha,
                binding_spec_path=spec_path, invocation_id=invocation_id,
                fixed_root=HISTORY_ROOT, project_root=ROOT,
                retest=retest_worker,
                retest_admission_package=retest_package,
                retest_admission_sha256=retest_sha)
        elif mode == "synthetic":
            config = job.get("config")
            if not isinstance(config, Mapping):
                config = MODEL_CONFIG
            device = job.get("device") or "cuda"
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            provider = SyntheticTrainingProvider(
                dict(config), seed=int(job.get("seed", 20261029)),
                device=device)
            fixture = root / "fixture"
            if not (fixture / "fixture.json").is_file():
                provider.prepare_fixture(fixture)
            provider.load_fixture()
            if provider.fixture.get("problems"):
                raise SystemExit(
                    f"worker fixture refused: {provider.fixture['problems']}")
            result = run_binding_migration_audit(
                root, provider, mode="synthetic", tolerance=contract,
                authorization_sha=auth_sha, authorization=authorization,
                binding_spec_path=spec_path, invocation_id=invocation_id,
                fixed_root=root.parent, project_root=ROOT,
                retest=retest_worker,
                r0_started_perf=float(
                    strict_read_json(
                        root / LAUNCH_JOURNAL_DIR / invocation_id
                        / "claim.json")["t0_perf_counter"]))
        else:
            raise SystemExit("migration worker mode must be real or synthetic")
        write_json(result_path, {
            "result": compact_migration_result(result),
            "roster": provider.roster})
        _print_result(result)
        return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)

    if args.synthetic_config_json is not None and args.real:
        raise SystemExit("--synthetic-config-json is synthetic-only")

    if args.calibrate_synthetic:
        out = (args.tolerance_out or args.out or
               ROOT / "derived/audit/w1_real_training_update_audit/calibration")
        out = Path(out).expanduser().resolve()
        if out.suffix != ".json":
            out = out / "tolerance_contract.json"
        _guard(out.parent)
        if out.exists() or (out.parent /
                            "calibration_verification.json").exists():
            raise SystemExit(
                f"calibration output already exists (choose a fresh "
                f"directory): {out.parent}")
        out.parent.mkdir(parents=True, exist_ok=True)
        contract = synthetic_calibration(seed=args.seed, device="cuda")
        problems = tolerance_contract_problems(contract, recompute=False)
        if problems:
            raise SystemExit(f"calibration invalid: {problems}")
        from ireset.eval.w1_real_training_update_provider import (
            calibration_verification,
        )
        write_json(out, contract)
        fp32_path = out.parent / "adamw_fp32_contract.json"
        # Write exactly the hash-bound (timestamp-free) contract so the file
        # SHA equals ``fp32_contract_sha256``.
        write_json(fp32_path, contract["fp32_contract"])
        verification = calibration_verification(
            contract, out, sha256_file(out))
        verification_path = out.parent / "calibration_verification.json"
        write_json(verification_path, verification)
        print(f"[update-audit] tolerance contract={out}")
        print(f"[update-audit] sha256={sha256_file(out)}")
        print(f"[update-audit] evidence_sha256={contract['evidence_sha256']}")
        print(f"[update-audit] fp32 contract={fp32_path}")
        print(f"[update-audit] fp32 sha256="
              f"{contract['fp32_contract_sha256']}")
        print(f"[update-audit] calibration verification={verification_path}")
        print(f"[update-audit] passed={verification['passed']} "
              f"recompute_problems={len(verification['recompute_problems'])} "
              f"four_x_ok={verification['four_x_ok']}")
        return 0 if verification["passed"] else 2

    if args.admission:
        contract, path, sha = _resolve_tolerance_for_admission(args)
        out = (args.out or
               ROOT / "derived/audit/w1_real_training_update_audit/admission")
        out = _guard(out)
        package = build_admission_package(
            ROOT, contract, tolerance_contract_path=path,
            tolerance_contract_sha256=sha)
        package_path, package_sha = write_package(out, package)
        print(f"[update-audit] admission package={package_path}")
        print(f"[update-audit] sha256={package_sha}")
        print(f"[update-audit] passed={package['passed']} "
              f"problems={len(package['problems'])}")
        for problem in package["problems"][:20]:
            print(f"[update-audit]   {problem}")
        return 0 if package["passed"] else 2

    if args.real_admission:
        from ireset.eval.w1_real_training_update_provider import (
            build_real_admission_package,
            real_admission_package_problems,
            synthetic_acceptance_problems,
            write_real_package,
        )
        contract, path, sha = _resolve_tolerance_for_admission(args)
        if args.synthetic_acceptance is None:
            raise SystemExit("--real-admission requires --synthetic-acceptance")
        acceptance_path = Path(args.synthetic_acceptance).expanduser().resolve()
        if not acceptance_path.is_file():
            raise SystemExit(f"synthetic acceptance missing: {acceptance_path}")
        acceptance = json.loads(acceptance_path.read_text())
        problems = synthetic_acceptance_problems(acceptance)
        if problems:
            raise SystemExit(f"synthetic acceptance refused: {problems}")
        verification_path = _calibration_verification_path(args, path)
        verification = _load_calibration_verification(verification_path)
        if verification is None:
            raise SystemExit(
                "--real-admission requires the calibration_verification.json "
                "written by --calibrate-synthetic")
        out = (args.out or
               ROOT / "derived/audit/w1_real_training_update_audit/"
                      "admission_real")
        out = _guard(out)
        package = build_real_admission_package(
            ROOT, tolerance_contract=contract,
            tolerance_contract_path=path,
            tolerance_contract_sha256=sha,
            calibration_verification_record=verification,
            acceptance_path=acceptance_path, acceptance=acceptance)
        package_problems = real_admission_package_problems(package)
        if package_problems:
            package["passed"] = False
            package["problems"] = list(package["problems"]) + package_problems
        package_path, package_sha = write_real_package(out, package)
        print(f"[update-audit] real admission package={package_path}")
        print(f"[update-audit] sha256={package_sha}")
        print(f"[update-audit] passed={package['passed']} "
              f"problems={len(package['problems'])}")
        for problem in package["problems"][:20]:
            print(f"[update-audit]   {problem}")
        return 0 if package["passed"] else 2

    if args.migration_snapshot:
        from ireset.eval.w1_real_training_update_audit import (
            MIGRATION_SOURCE_RUN,
        )
        run_root = Path(args.migration_root or
                        ROOT / MIGRATION_SOURCE_RUN).expanduser().resolve()
        if not (run_root / "cohort.json").is_file():
            raise SystemExit(f"migration run root missing: {run_root}")
        out = (args.out or
               ROOT / "derived/audit/w1_real_training_update_audit/"
                      "binding_migration/attempt_001")
        out = _guard(out)
        path, sha, snapshot = write_history_snapshot(out, run_root)
        print(f"[update-audit] history snapshot={path}")
        print(f"[update-audit] sha256={sha}")
        print(f"[update-audit] legacy_head_sha256="
              f"{snapshot['legacy_head_sha256']}")
        print(f"[update-audit] cost_s={snapshot['cost_s']}")
        print(f"[update-audit] unsealed_files="
              f"{len(snapshot['unsealed_files'])}")
        baseline_path = out / "protected_manifest.json"
        if not baseline_path.exists():
            selected = {}
            for key, value in (
                    ("calibration_attempt", args.calibration_attempt),
                    ("synthetic_attempt", args.synthetic_attempt),
                    ("admission_attempt", args.admission_attempt)):
                if value is not None:
                    selected[key] = value
            path, baseline_sha, baseline = write_migration_baseline(
                out, ROOT, run_root, **selected)
            print(f"[update-audit] protected manifest={path}")
            print(f"[update-audit] protected_sha256={baseline_sha} "
                  f"entries={baseline['entry_count']}")
        return 0

    if args.migration_admission:
        from ireset.eval.w1_real_training_update_audit import (
            MIGRATION_SOURCE_RUN,
            build_migration_admission,
            migration_admission_problems,
        )
        if args.history_snapshot is None or args.binding_spec is None:
            raise SystemExit(
                "--migration-admission requires --history-snapshot and "
                "--binding-spec")
        snapshot_path = Path(args.history_snapshot).expanduser().resolve()
        spec_path = Path(args.binding_spec).expanduser().resolve()
        if not snapshot_path.is_file() or not spec_path.is_file():
            raise SystemExit("migration snapshot/spec missing")
        snapshot = strict_read_json(snapshot_path)
        spec = strict_read_json(spec_path)
        snapshot_sha = sha256_file(snapshot_path)
        spec_sha = sha256_file(spec_path)
        real_path = (Path(args.admission_package).expanduser().resolve()
                     if args.admission_package is not None else None)
        real_record = strict_read_json(real_path) if real_path else None
        real_sha = sha256_file(real_path) if real_path else None
        gate_path = (Path(args.gate_b).expanduser().resolve()
                     if args.gate_b is not None else None)
        gate_sha = sha256_file(gate_path) if gate_path else None
        out = (args.out or
               ROOT / "derived/audit/w1_real_training_update_audit/"
                      "binding_migration_admission")
        out = _guard(out)
        package = build_migration_admission(
            ROOT, snapshot_path=snapshot_path,
            snapshot_sha256=snapshot_sha, snapshot=snapshot,
            spec_path=spec_path, spec_sha256=spec_sha, spec=spec,
            real_admission_path=real_path,
            real_admission_sha256=real_sha, real_admission=real_record,
            gate_b_path=gate_path, gate_b_sha256=gate_sha)
        package_problems = migration_admission_problems(package)
        if package_problems:
            package["passed"] = False
            package["problems"] = list(package["problems"]) + package_problems
        path, sha = write_migration_admission(out, package)
        print(f"[update-audit] migration admission={path}")
        print(f"[update-audit] sha256={sha}")
        print(f"[update-audit] passed={package['passed']} "
              f"problems={len(package['problems'])}")
        for problem in package["problems"][:30]:
            print(f"[update-audit]   {problem}")
        return 0 if package["passed"] else 2

    if args.migration_authorization_draft:
        from ireset.eval.w1_real_training_update_audit import (
            build_migration_authorization_draft,
        )
        if args.history_snapshot is None or args.binding_spec is None or \
                args.migration_admission_package is None:
            raise SystemExit(
                "--migration-authorization-draft requires --history-snapshot, "
                "--binding-spec and --migration-admission-package")
        snapshot_path = Path(args.history_snapshot).expanduser().resolve()
        spec_path = Path(args.binding_spec).expanduser().resolve()
        snapshot_sha = sha256_file(snapshot_path)
        spec_sha = sha256_file(spec_path)
        binding, problems = load_external_binding_material(
            snapshot_path, snapshot_sha, spec_path, spec_sha,
            project_root=ROOT, require_current_implementation=True)
        if binding is None:
            raise SystemExit(f"migration binding refused: {problems}")
        if problems:
            raise SystemExit(f"external binding material refused: {problems}")
        admission_path = Path(
            args.migration_admission_package).expanduser().resolve()
        admission_sha = sha256_file(admission_path)
        real_path = (Path(args.admission_package).expanduser().resolve()
                     if args.admission_package is not None else None)
        real_sha = sha256_file(real_path) if real_path else None
        gate_path = (Path(args.gate_b).expanduser().resolve()
                     if args.gate_b is not None else None)
        gate_sha = sha256_file(gate_path) if gate_path else None
        draft = build_migration_authorization_draft(
            ROOT, binding=binding,
            snapshot_path=snapshot_path, spec_path=spec_path,
            migration_admission_path=admission_path,
            migration_admission_sha256=admission_sha,
            real_admission_path=real_path,
            real_admission_sha256=real_sha,
            gate_b_path=gate_path, gate_b_sha256=gate_sha,
            invocation_id=args.invocation_id, authorized=False)
        out = (args.out or
               ROOT / "derived/audit/w1_real_training_update_audit/"
                      "binding_migration_authorization")
        out = _guard(out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "authorization_draft.json"
        if path.exists():
            raise SystemExit(f"authorization draft already exists: {path}")
        write_json(path, draft)
        print(f"[update-audit] authorization draft={path}")
        print(f"[update-audit] sha256={sha256_file(path)}")
        print("[update-audit] authorized=false template_only="
              f"{draft['template_only']}")
        return 0

    if args.migration_synthetic:
        if args.out is None:
            raise SystemExit("--migration-synthetic requires --out")
        if args.tolerance_contract is None:
            raise SystemExit(
                "--migration-synthetic requires --tolerance-contract")
        contract, sha, problems = load_tolerance_contract(
            args.tolerance_contract, recompute=False)
        if problems:
            raise SystemExit(f"tolerance contract refused: {problems}")
        if not isinstance(sha, str) or \
                contract.get("mode") != "synthetic_calibrated":
            raise SystemExit("calibrated tolerance contract required")
        contract_path = Path(args.tolerance_contract).expanduser().resolve()
        verification = _load_calibration_verification(
            _calibration_verification_path(args, contract_path))
        if verification is None:
            raise SystemExit(
                "--migration-synthetic requires calibration_verification.json")
        device = args.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        out = _guard(args.out)
        invocation_id = args.invocation_id or "inv-gate-b-migration-001"
        launch = prepare_migration_synthetic_launch(
            out, tolerance=contract,
            calibration_verification_record=verification,
            invocation_id=invocation_id)
        job = {
            "mode": "synthetic", "root": launch["root"],
            "spec_path": launch["spec_path"],
            "authorization_path": launch["authorization_path"],
            "tolerance_contract_path": str(contract_path),
            "invocation_id": invocation_id,
            "result_path": str(out / "migration_worker_result.json"),
            "seed": args.seed, "device": device,
            "config": _load_synthetic_config(args) or dict(MODEL_CONFIG),
        }
        job_path = out / "migration" / "worker_job.json"
        write_json_exclusive(job_path, job)
        facts = _spawn_worker(
            [sys.executable, str(Path(__file__).resolve()),
             "--migration-worker", "--job", str(job_path)],
            out / "migration" / "worker.log")
        document = _worker_result(Path(job["result_path"]))
        if document is None or not isinstance(document.get("result"), Mapping):
            _write_launch_failure(
                Path(launch["root"]) / LAUNCH_JOURNAL_DIR / invocation_id,
                facts, "worker_result_missing")
            print("[update-audit] migration worker did not return a result")
            print(f"[update-audit] worker_exit={facts['exit_code']} "
                  f"log={facts['log_path']}")
            return 3
        result = dict(document["result"])
        roster = document.get("roster") or {}
        return_path, return_sha, return_problems = _record_worker_return(
            Path(launch["root"]), invocation_id, result, facts)
        journal = scan_launch_journals(Path(launch["root"]).parent)
        acceptance = build_migration_synthetic_acceptance(
            out, result, roster=roster, tolerance=contract,
            calibration_verification_record=verification, launch=launch,
            return_path=return_path, return_sha256=return_sha,
            return_problems=return_problems, journal=journal)
        result["migration_acceptance"] = acceptance
        result["worker_return"] = facts
        result["launch_journal"] = {
            "known": bool(journal.get("known")),
            "problems": list(journal.get("problems") or [])}
        write_json(out / "migration_synthetic_result.json", result)
        print(f"[update-audit] migration synthetic conclusion="
              f"{result.get('conclusion')}")
        print(f"[update-audit] selected_attempts={result.get('selected_attempts')}")
        print(f"[update-audit] cohort_training_updates="
              f"{result.get('cohort_training_updates')}")
        print(f"[update-audit] worker_exit={facts['exit_code']} "
              f"pid={facts['pid']} ppid={facts['ppid']}")
        print(f"[update-audit] migration acceptance="
              f"{acceptance.get('path')}")
        print(f"[update-audit] sha256={acceptance.get('sha256')}")
        print(f"[update-audit] passed={acceptance.get('passed')} "
              f"problems={len(acceptance.get('problems') or [])}")
        for problem in (acceptance.get("problems") or [])[:30]:
            print(f"[update-audit]   {problem}")
        return 0 if acceptance.get("passed") else 2

    if args.retest_synthetic:
        if args.out is None:
            raise SystemExit("--retest-synthetic requires --out")
        if args.tolerance_contract is None:
            raise SystemExit(
                "--retest-synthetic requires --tolerance-contract")
        contract, _sha, problems = load_tolerance_contract(
            args.tolerance_contract, recompute=False)
        if problems:
            raise SystemExit(f"tolerance contract refused: {problems}")
        contract_path = Path(args.tolerance_contract).expanduser().resolve()
        verification = _load_calibration_verification(
            _calibration_verification_path(args, contract_path))
        if verification is None:
            raise SystemExit(
                "--retest-synthetic requires calibration_verification.json")
        device = args.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        out = _guard(args.out)
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"retest synthetic root is not fresh: {out}")
        out.mkdir(parents=True, exist_ok=True)
        v2_invocation = "inv-gate-b-migration-001"
        launch = prepare_migration_synthetic_launch(
            out / "prefix", tolerance=contract,
            calibration_verification_record=verification,
            invocation_id=v2_invocation)
        job = {
            "base": str(out),
            "launch": launch,
            "v3_invocation": "inv-retest-001",
            "tolerance_contract_path": str(contract_path),
            "result_path": str(out / "retest_chain_result.json"),
            "seed": args.seed, "device": device,
            "config": _load_synthetic_config(args) or dict(MODEL_CONFIG),
        }
        job_path = out / "retest_chain_job.json"
        write_json_exclusive(job_path, job)
        facts = _spawn_worker(
            [sys.executable, str(Path(__file__).resolve()),
             "--retest-chain-worker", "--job", str(job_path)],
            out / "retest_chain_worker.log")
        document = _worker_result(Path(job["result_path"]))
        if document is None or not isinstance(document.get("result"), Mapping):
            print("[update-audit] retest chain worker returned no result")
            return 3
        result = dict(document["result"])
        v3 = result.get("v3_full") or {}
        v3_launch = result.get("v3_launch") or {}
        if not v3 or not v3_launch:
            print("[update-audit] retest chain v3 result missing")
            return 3
        stage_timings = v3.get("stage_timings") or {}
        last_end = v3.get("last_stage_end_perf_counter")
        return_path = None
        return_sha = None
        return_problems: list[str] = []
        if stage_timings and last_end is not None:
            return_path, return_sha, return_problems = \
                record_return_observation(
                    Path(str(v3_launch["root"])),
                    invocation_id=str(v3_launch["invocation_id"]),
                    pid=facts["pid"], ppid=facts["ppid"],
                    exit_code=facts["exit_code"], signal=facts["signal"],
                    started_perf_counter=float(
                        v3_launch["t0_perf_counter"]),
                    finished_perf_counter=float(
                        facts["finished_perf_counter"]),
                    last_stage_end_perf_counter=float(last_end),
                    commits=v3.get("commits") or {},
                    candidate_receipts=v3.get("candidates") or {},
                    write_after={"R0": True, "P": True, "I": True,
                                 "F": True, "terminal": True},
                    stage_timings=stage_timings, source_kind="outer_wait",
                    expected_exit_code=facts["exit_code"],
                    conclusion=v3.get("conclusion"))
        else:
            return_problems = ["v3_timing_missing"]
        journal = scan_launch_journals(
            Path(str(v3_launch["root"])).parent)
        acceptance = build_retest_synthetic_acceptance(
            out, result, roster=document.get("roster") or {},
            tolerance=contract,
            calibration_verification_record=verification,
            launch=v3_launch, return_path=return_path,
            return_sha256=return_sha, return_problems=return_problems,
            journal=journal)
        result["retest_acceptance"] = acceptance
        result["worker_return"] = facts
        write_json(out / "retest_synthetic_result.json", result)
        print(f"[update-audit] retest synthetic v3 conclusion="
              f"{v3.get('conclusion')}")
        print(f"[update-audit] selected_attempts="
              f"{v3.get('selected_attempts')}")
        print(f"[update-audit] cohort_training_updates="
              f"{v3.get('cohort_training_updates')}")
        print(f"[update-audit] worker_exit={facts['exit_code']} "
              f"pid={facts['pid']} ppid={facts['ppid']}")
        print(f"[update-audit] retest acceptance={acceptance.get('path')}")
        print(f"[update-audit] sha256={acceptance.get('sha256')} "
              f"passed={acceptance.get('passed')}")
        for problem in (acceptance.get("problems") or [])[:30]:
            print(f"[update-audit]   {problem}")
        return 0 if acceptance.get("passed") else 2

    if args.migration_spec:
        if args.out is None or args.history_snapshot is None or \
                args.gate_b is None or args.admission_package is None:
            raise SystemExit(
                "--migration-spec requires --out, --history-snapshot, "
                "--gate-b and --admission-package")
        snapshot_path = Path(args.history_snapshot).expanduser().resolve()
        gate_path = Path(args.gate_b).expanduser().resolve()
        admission_path = Path(args.admission_package).expanduser().resolve()
        for label, path in (("history snapshot", snapshot_path),
                            ("Gate B", gate_path),
                            ("real admission", admission_path)):
            if not path.is_file():
                raise SystemExit(f"{label} missing: {path}")
        snapshot = strict_read_json(snapshot_path)
        snapshot_sha = sha256_file(snapshot_path)
        gate_sha = sha256_file(gate_path)
        admission_sha = sha256_file(admission_path)
        gate_b = {"path": path_for_record(gate_path), "sha256": gate_sha,
                  "acceptance_sha256": gate_sha}
        if args.tolerance_contract is not None:
            contract, contract_sha, problems = load_tolerance_contract(
                args.tolerance_contract, recompute=False)
            if problems:
                raise SystemExit(f"tolerance contract refused: {problems}")
            contract_path = Path(
                args.tolerance_contract).expanduser().resolve()
            gate_b.update(
                tolerance_contract_sha256=contract_sha,
                tolerance_evidence_sha256=contract.get("evidence_sha256"),
                fp32_contract_sha256=contract.get("fp32_contract_sha256"))
            verification_path = _calibration_verification_path(
                args, contract_path)
            if verification_path is not None and verification_path.is_file():
                gate_b["calibration_verification_sha256"] = sha256_file(
                    verification_path)
        spec = build_binding_spec(
            ROOT, snapshot_path, snapshot_sha, snapshot,
            invocation_id=args.invocation_id, gate_b=gate_b,
            admission={"path": path_for_record(admission_path),
                       "sha256": admission_sha})
        out = _guard(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "binding_spec.json"
        if path.exists():
            raise SystemExit(f"binding spec already exists: {path}")
        sha = write_json_exclusive(path, spec)
        print(f"[update-audit] binding spec={path}")
        print(f"[update-audit] sha256={sha}")
        print(f"[update-audit] legacy_head_sha256="
              f"{spec.get('legacy_head_sha256')}")
        return 0

    if args.retest_materials:
        from ireset.eval.w1_real_training_update_audit import (
            MIGRATION_SOURCE_RUN,
            RETEST_PLAN,
            build_retest_admission,
            build_retest_authorization_draft,
            build_retest_protected_manifest,
            fresh_retest_root_problems,
            retest_admission_problems,
            write_retest_admission,
        )
        if args.out is None or args.admission_package is None or \
                args.gate_b is None or args.retest_acceptance is None:
            raise SystemExit(
                "--retest-materials requires --out, --admission-package, "
                "--gate-b and --retest-acceptance")
        run_root = Path(args.migration_root or
                        ROOT / MIGRATION_SOURCE_RUN).expanduser().resolve()
        if not (run_root / "cohort.json").is_file():
            raise SystemExit(f"retest run root missing: {run_root}")
        out = _guard(args.out)
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"retest materials root is not fresh: {out}")
        out.mkdir(parents=True, exist_ok=True)
        fresh = fresh_retest_root_problems(run_root)
        if fresh:
            raise SystemExit(f"retest target paths already exist: {fresh}")
        snapshot = build_retest_history_snapshot(run_root)
        snapshot_file = out / "history_snapshot.json"
        write_json_exclusive(snapshot_file, snapshot)
        snapshot_sha = sha256_file(snapshot_file)
        source = out / "source"
        source.mkdir(parents=True, exist_ok=True)
        for name in ("index.json", "ledger.json", "cohort.json"):
            path = run_root / ("attempts_index.json" if name == "index.json"
                               else "costs.json" if name == "ledger.json"
                               else "cohort.json")
            (source / name).write_bytes(path.read_bytes())
        revision = build_budget_revision(
            snapshot, mode=str(snapshot.get("mode") or "real"))
        revision_file = out / "budget_revision.json"
        write_json_exclusive(revision_file, revision)
        protected = build_retest_protected_manifest(
            ROOT, run_root,
            repair_evidence=[
                ROOT / "derived/audit/w1_real_training_update_audit/"
                       "gradient_numerics_retest/"],
        )
        protected_file = out / "protected_manifest.json"
        write_json_exclusive(protected_file, protected)
        gate_b_path = Path(args.gate_b).expanduser().resolve()
        retest_path = Path(args.retest_acceptance).expanduser().resolve()
        real_path = Path(args.admission_package).expanduser().resolve()
        gate_b = strict_read_json(gate_b_path)
        retest_acceptance = strict_read_json(retest_path)
        real_admission = strict_read_json(real_path)
        gate_b_sha = sha256_file(gate_b_path)
        retest_sha = sha256_file(retest_path)
        real_sha = sha256_file(real_path)
        spec = build_retest_binding_spec(
            ROOT, snapshot_file, snapshot_sha, snapshot,
            invocation_id=args.invocation_id or "inv-retest-real-001",
            gate_b_v2={"path": path_for_record(gate_b_path),
                       "sha256": gate_b_sha,
                       "tolerance_evidence_sha256": (
                           (real_admission.get("tolerance") or {}).get(
                               "evidence_sha256")),
                       "fp32_contract_sha256": real_admission.get(
                           "fp32_contract_sha256"),
                       "calibration_verification_sha256": real_admission.get(
                           "calibration_verification_sha256"),
                       "acceptance_sha256": gate_b.get("sha256")},
            retest_acceptance={"path": path_for_record(retest_path),
                               "sha256": retest_sha},
            admission={"path": path_for_record(real_path),
                       "sha256": real_sha},
            protected_manifest={
                "path": path_for_record(protected_file),
                "sha256": sha256_file(protected_file)})
        spec_file = out / "binding_spec.json"
        write_json_exclusive(spec_file, spec)
        spec_sha = sha256_file(spec_file)
        admission = build_retest_admission(
            ROOT, snapshot_path=snapshot_file,
            snapshot_sha256=snapshot_sha, snapshot=snapshot,
            spec_path=spec_file, spec_sha256=spec_sha, spec=spec,
            budget_revision_path=revision_file,
            budget_revision_sha256=revision["sha256"],
            budget_revision=revision,
            protected_manifest_path=protected_file,
            protected_manifest_sha256=sha256_file(protected_file),
            protected_manifest=protected,
            real_admission_path=real_path,
            real_admission_sha256=real_sha,
            real_admission=real_admission,
            gate_b_v2_path=gate_b_path,
            gate_b_v2_sha256=gate_b_sha, gate_b_v2=gate_b,
            retest_acceptance_path=retest_path,
            retest_acceptance_sha256=retest_sha,
            retest_acceptance=retest_acceptance)
        admission_problems = retest_admission_problems(
            admission, project_root=ROOT, strict=True)
        if admission_problems:
            admission["passed"] = False
            admission["problems"] = list(admission.get("problems") or []) + \
                admission_problems
        admission_path, admission_sha = write_retest_admission(
            out, admission)
        draft = build_retest_authorization_draft(
            ROOT, binding=load_external_binding_material(
                snapshot_file, snapshot_sha, spec_file, spec_sha,
                project_root=ROOT,
                require_current_implementation=True)[0],
            snapshot_path=snapshot_file, spec_path=spec_file,
            retest_admission_path=admission_path,
            retest_admission_sha256=admission_sha,
            protected_manifest_sha256=sha256_file(protected_file),
            gate_b_v2_path=gate_b_path, gate_b_v2_sha256=gate_b_sha,
            retest_acceptance_path=retest_path,
            retest_acceptance_sha256=retest_sha,
            real_admission_path=real_path,
            real_admission_sha256=real_sha,
            invocation_id=args.invocation_id or "inv-retest-real-001",
            authorized=False)
        real_tolerance = real_admission.get("tolerance") or {}
        draft.update(
            calibration_verification_sha256=real_admission.get(
                "calibration_verification_sha256"),
            tolerance_evidence_sha256=real_tolerance.get("evidence_sha256"),
            fp32_contract_sha256=real_admission.get(
                "fp32_contract_sha256"),
            acceptance_sha256=retest_sha,
            admission_package_sha256=real_sha)
        draft_file = out / "authorization_draft.json"
        write_json_exclusive(draft_file, draft)
        draft_problems: list[str] = []
        checklist = draft.get("checklist") or {}
        if set(checklist) != set(AUTHORIZATION_REQUIRED):
            draft_problems.append("authorization_checklist_keys")
        if any(value is not None for value in checklist.values()):
            draft_problems.append("authorization_checklist_not_null")
        if draft.get("authorized") is not False or \
                draft.get("template_only") is not True:
            draft_problems.append("authorization_must_be_template_only")
        real_tolerance = real_admission.get("tolerance") or {}
        expected_top = {
            "calibration_verification_sha256": real_admission.get(
                "calibration_verification_sha256"),
            "tolerance_evidence_sha256": real_tolerance.get(
                "evidence_sha256"),
            "fp32_contract_sha256": real_admission.get(
                "fp32_contract_sha256"),
            "acceptance_sha256": retest_sha,
            "admission_package_sha256": real_sha,
            "draft_sha256": sha256_file(ROOT / "docs/plans/"
                                        "omniretarget_w1_real_training_"
                                        "update_audit_prereg_draft.md"),
            "execution_plan_sha256": sha256_file(
                ROOT / "docs/plans/"
                       "omniretarget_w1_gradient_numerics_retest_plan.md"),
        }
        for key, want in expected_top.items():
            value = draft.get(key)
            if not isinstance(want, str) or not isinstance(value, str) or \
                    value != want or len(value) != 64:
                draft_problems.append(f"authorization_binding:{key}")
        if draft.get("implementation_files") != \
                implementation_identity(ROOT).get("files"):
            draft_problems.append("authorization_implementation")
        expected_retest = {
            "snapshot_sha256": snapshot_sha,
            "spec_sha256": spec_sha,
            "budget_revision_sha256": revision["sha256"],
            "protected_manifest_sha256": sha256_file(protected_file),
            "retest_admission_sha256": admission_sha,
            "gate_b_v2_sha256": gate_b_sha,
            "retest_acceptance_sha256": retest_sha,
            "real_admission_sha256": real_sha,
            "closed_history_head_sha256": snapshot[
                "closed_history_head_sha256"],
            "invocation_id": args.invocation_id or "inv-retest-real-001",
        }
        retest_draft = draft.get("retest") or {}
        for key, want in expected_retest.items():
            if retest_draft.get(key) != want:
                draft_problems.append(f"authorization_retest:{key}")
        preflight = {
            "schema": "w1_real_training_update_retest_preflight_v1",
            "passed": bool(admission.get("passed")) and not draft_problems,
            "problems": list(admission.get("problems") or []) + draft_problems,
            "files": {
                path_for_record(path): {
                    "sha256": sha256_file(path),
                    "bytes": int(path.stat().st_size)}
                for path in (snapshot_file, revision_file, protected_file,
                             spec_file, admission_path, draft_file)},
            "authorization_template_only": draft.get("template_only"),
            "authorization_authorized": draft.get("authorized"),
            "next_action": ("user must review and provide the ten explicit "
                            "section 9 checklist authorizations"),
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        preflight_file = out / "preflight_verification.json"
        write_json_exclusive(preflight_file, preflight)
        print(f"[update-audit] retest snapshot={snapshot_file}")
        print(f"[update-audit] budget revision={revision_file}")
        print(f"[update-audit] protected manifest={protected_file} "
              f"entries={protected['entry_count']}")
        print(f"[update-audit] binding spec={spec_file}")
        print(f"[update-audit] retest admission={admission_path} "
              f"passed={admission.get('passed')}")
        print(f"[update-audit] authorization draft={draft_file} "
              f"authorized={draft.get('authorized')}")
        print(f"[update-audit] preflight={preflight_file} "
              f"passed={preflight['passed']}")
        for problem in preflight["problems"][:30]:
            print(f"[update-audit]   {problem}")
        return 0 if preflight["passed"] else 2

    if args.gate_b_v2:
        if args.out is None or args.ordinary_acceptance is None or \
                args.migration_acceptance is None:
            raise SystemExit(
                "--gate-b-v2 requires --out, --ordinary-acceptance and "
                "--migration-acceptance")
        from ireset.eval.w1_real_training_update_provider import (
            synthetic_acceptance_v2,
            synthetic_acceptance_v2_problems,
        )
        ordinary_path = Path(args.ordinary_acceptance).expanduser().resolve()
        migration_path = Path(args.migration_acceptance).expanduser().resolve()
        for label, path in (("ordinary acceptance", ordinary_path),
                            ("migration acceptance", migration_path)):
            if not path.is_file():
                raise SystemExit(f"{label} missing: {path}")
        ordinary = strict_read_json(ordinary_path)
        migration = strict_read_json(migration_path)
        identity = migration.get("migration_identity") or {}
        for key in ("spec_sha256", "snapshot_sha256", "legacy_head_sha256"):
            value = identity.get(key)
            if not isinstance(value, str):
                raise SystemExit(f"migration identity missing {key}")
        record = synthetic_acceptance_v2(
            ordinary=ordinary, ordinary_path=ordinary_path,
            ordinary_sha256=sha256_file(ordinary_path),
            migration=migration, migration_path=migration_path,
            migration_sha256=sha256_file(migration_path),
            spec_sha256=identity["spec_sha256"],
            snapshot_sha256=identity["snapshot_sha256"],
            legacy_head_sha256=identity["legacy_head_sha256"])
        problems = synthetic_acceptance_v2_problems(record)
        if problems:
            record["passed"] = False
            record["problems"] = list(record.get("problems") or []) + problems
        out = _guard(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "synthetic_acceptance.json"
        if path.exists():
            raise SystemExit(f"Gate B v2 already exists: {path}")
        sha = write_json_exclusive(path, record)
        print(f"[update-audit] Gate B v2={path}")
        print(f"[update-audit] sha256={sha}")
        print(f"[update-audit] passed={record['passed']} "
              f"problems={len(record['problems'])}")
        for problem in record["problems"][:30]:
            print(f"[update-audit]   {problem}")
        return 0 if record["passed"] else 2

    if args.real:
        if args.authorization is None:
            raise SystemExit(
                "--real requires --authorization (section 10.2 is unchecked)")
        if args.tolerance_contract is None:
            raise SystemExit("--real requires --tolerance-contract")
        if args.admission_package is None:
            raise SystemExit("--real requires --admission-package")
        if args.out is None:
            raise SystemExit("--real requires --out")
        if args.resume and args.stage is None:
            raise SystemExit("--resume requires --stage")
        if args.stage is not None and not args.resume:
            raise SystemExit("--stage is only valid with --resume")
        out = Path(args.out).expanduser().resolve()
        base = _history_root()
        migration_launch = bool((args.migration or args.retest)
                                and not args.resume and args.stage is None)
        try:
            if out.parent != base:
                raise SystemExit(
                    f"--out must be a direct child of {base}, got {out}")
            record, auth_sha, auth_problems = load_authorization(
                args.authorization)
            if auth_problems:
                raise SystemExit(f"authorization refused: {auth_problems}")
            contract, contract_sha, contract_problems = load_tolerance_contract(
                args.tolerance_contract, recompute=False)
            if contract_problems:
                raise SystemExit(
                    f"tolerance contract refused: {contract_problems}")
            try:
                package = json.loads(Path(args.admission_package).read_text())
            except (OSError, ValueError):
                raise SystemExit("admission package unreadable")
            package_sha = sha256_file(args.admission_package)
            from ireset.eval.w1_real_training_update_provider import (
                REAL_ADMISSION_SCHEMA,
                RealTrainingProvider,
            )
            if package.get("schema") == REAL_ADMISSION_SCHEMA:
                from ireset.eval.w1_real_training_update_provider import (
                    real_admission_package_problems,
                )
                pkg_problems = real_admission_package_problems(package)
            else:
                pkg_problems = admission_package_problems(package)
            if pkg_problems:
                raise SystemExit(f"admission package refused: {pkg_problems}")
            if args.migration or args.retest:
                if args.resume or args.stage is not None:
                    raise SystemExit(
                        "binding generation entries cannot use --resume")
                if args.binding_spec is None or args.invocation_id is None:
                    raise SystemExit(
                        "generation entry requires --binding-spec and "
                        "--invocation-id")
                spec_path = Path(args.binding_spec).expanduser().resolve()
                if not spec_path.is_file():
                    raise SystemExit(f"binding spec missing: {spec_path}")
                spec = strict_read_json(spec_path)
                spec_sha = sha256_file(spec_path)
                reference = spec.get("history_snapshot") or {}
                snapshot_path = Path(str(reference.get("path") or ""))
                if not snapshot_path.is_absolute():
                    snapshot_path = ROOT / snapshot_path
                snapshot_path = snapshot_path.expanduser().resolve()
                snapshot_sha = reference.get("sha256")
                if not snapshot_path.is_file():
                    raise SystemExit(
                        f"history snapshot missing: {snapshot_path}")
                snapshot = strict_read_json(snapshot_path)
                binding, binding_problems = load_external_binding_material(
                    snapshot_path, snapshot_sha, spec_path, spec_sha,
                    project_root=ROOT, require_current_implementation=True)
                if binding is None or binding_problems:
                    raise SystemExit(
                        f"migration binding refused: {binding_problems}")
                retest = bool(args.retest)
                retest_admission = None
                retest_admission_sha = None
                if retest:
                    from ireset.eval.w1_real_training_update_audit import (
                        retest_admission_problems,
                    )
                    if args.retest_admission is None:
                        raise SystemExit(
                            "--real --retest requires --retest-admission")
                    retest_admission_path = Path(
                        args.retest_admission).expanduser().resolve()
                    if not retest_admission_path.is_file():
                        raise SystemExit(
                            f"retest admission missing: "
                            f"{retest_admission_path}")
                    retest_admission_sha = sha256_file(retest_admission_path)
                    retest_admission = strict_read_json(
                        retest_admission_path)
                    admission_problems = retest_admission_problems(
                        retest_admission, project_root=ROOT, strict=True)
                    if admission_problems:
                        raise SystemExit(
                            f"retest admission refused: {admission_problems}")
                    if (retest_admission.get("spec") or {}).get(
                            "sha256") != spec_sha or \
                            (retest_admission.get("snapshot") or {}).get(
                                "sha256") != snapshot_sha:
                        raise SystemExit(
                            "retest admission spec/snapshot binding")
                    if (retest_admission.get("budget_revision") or {}).get(
                            "sha256") != spec.get(
                                "budget_revision_sha256"):
                        raise SystemExit(
                            "retest admission budget binding")
                    if (retest_admission.get("protected_manifest") or {}).get(
                            "sha256") != ((spec.get("bindings") or {}).get(
                                "protected_manifest") or {}).get("sha256"):
                        raise SystemExit(
                            "retest admission protected-manifest binding")
                migration = (record.get("retest") if retest
                             else record.get("migration")) or {}
                if retest:
                    expected = {
                        "run_root_realpath": str(out),
                        "invocation_id": args.invocation_id,
                        "spec_sha256": spec_sha,
                        "snapshot_sha256": snapshot_sha,
                        "closed_history_head_sha256": snapshot.get(
                            "closed_history_head_sha256"),
                        "budget_revision_sha256": spec.get(
                            "budget_revision_sha256"),
                        "protected_manifest_sha256": (
                            (spec.get("bindings") or {}).get(
                                "protected_manifest") or {}).get("sha256"),
                        "retest_admission_sha256": retest_admission_sha,
                        "gate_b_v2_sha256": ((spec.get("bindings") or {}).get(
                            "gate_b_v2") or {}).get("sha256"),
                        "retest_acceptance_sha256": (
                            (spec.get("bindings") or {}).get(
                                "retest_acceptance") or {}).get("sha256"),
                        "real_admission_sha256": package_sha,
                        "implementation_sha256": implementation_files_sha256(
                            implementation_identity(ROOT)),
                    }
                else:
                    expected = {
                        "run_root_realpath": str(out),
                        "invocation_id": args.invocation_id,
                        "spec_sha256": spec_sha,
                        "snapshot_sha256": snapshot_sha,
                        "legacy_head_sha256": snapshot.get("legacy_head_sha256"),
                        "gate_b_sha256": ((spec.get("bindings") or {}).get(
                            "gate_b") or {}).get("sha256"),
                        "real_admission_sha256": package_sha,
                        "implementation_sha256": implementation_files_sha256(
                            implementation_identity(ROOT)),
                    }
                binding_problems = [
                    f"authorization_migration:{key}" for key, value in
                    expected.items() if migration.get(key) != value]
                if ((spec.get("bindings") or {}).get(
                        "real_admission") or {}).get("sha256") != package_sha:
                    binding_problems.append("spec_real_admission")
                _claim, _claim_sha, claim_problems = launch_claim_record(
                    out, args.invocation_id,
                    authorization_sha256=auth_sha, spec_sha256=spec_sha,
                    legacy_head_sha256=snapshot["legacy_head_sha256"],
                    marker_sha256=snapshot["marker_sha256"],
                    closed_history_head_sha256=(
                        snapshot.get("closed_history_head_sha256")
                        if retest else None))
                if _claim is not None and _claim.get("pid") != os.getpid():
                    claim_problems.append("launch_claim_not_current_process")
                if claim_problems:
                    raise SystemExit(
                        f"launch claim refused: {claim_problems}")
                from ireset.eval.w1_real_training_update_audit import (
                    fresh_retest_root_problems,
                )
                fresh_problems = (
                    fresh_retest_root_problems(
                        out, allow_invocation=args.invocation_id)
                    if retest else
                    _fresh_migration_root_problems(
                        out, allow_invocation=args.invocation_id))
                if fresh_problems:
                    binding_problems += fresh_problems
                if not (out / "cohort.json").is_file():
                    binding_problems.append("migration_root_missing_marker")
                if binding_problems:
                    raise SystemExit(
                        f"migration launch refused: {binding_problems}")
                stage_binding(
                    out, spec_path=spec_path, spec_sha256=spec_sha,
                    snapshot_sha256=snapshot_sha,
                    invocation_id=args.invocation_id,
                    generation=("generation_003" if retest
                                else "generation_002"))
                directory = out / LAUNCH_JOURNAL_DIR / args.invocation_id
                job = {
                    "mode": "real", "root": str(out),
                    "spec_path": str(spec_path),
                    "authorization_path": args.authorization.resolve(),
                    "admission_package_path": args.admission_package.resolve(),
                    "tolerance_contract_path":
                        args.tolerance_contract.resolve(),
                    "invocation_id": args.invocation_id,
                    "retest": retest,
                    "retest_admission_path": (
                        str(retest_admission_path) if retest else None),
                    "retest_admission_sha256": retest_admission_sha,
                    "result_path": str(directory / "worker_result_inner.json"),
                }
                job_path = directory / "worker_job.json"
                write_json_exclusive(job_path, job)
            else:
                if not args.resume and out.exists() and any(out.iterdir()):
                    raise SystemExit(f"real run root is not fresh: {out}")
        except BaseException as exc:
            if migration_launch:
                reason = str(exc) or type(exc).__name__
                _write_precheck_failure(
                    out, args.invocation_id, _ENTRY_STARTED_PERF, reason)
            raise

        if args.migration or args.retest:
            worker_flag = ("--retest-worker" if args.retest
                           else "--migration-worker")
            facts = _spawn_worker(
                [sys.executable, str(Path(__file__).resolve()),
                 worker_flag, "--job", str(job_path)],
                directory / "worker.log")
            document = _worker_result(Path(job["result_path"]))
            if document is None or not isinstance(
                    document.get("result"), Mapping):
                _write_launch_failure(directory, facts,
                                      "worker_result_missing")
                print("[update-audit] migration worker did not return a result")
                print(f"[update-audit] worker_exit={facts['exit_code']} "
                      f"log={facts['log_path']}")
                return 3
            result = dict(document["result"])
            return_path, return_sha, return_problems = _record_worker_return(
                out, args.invocation_id, result, facts)
            journal = scan_launch_journals(out.parent)
            result["return_observation"] = (
                {"path": path_for_record(return_path),
                 "sha256": return_sha,
                 "problems": return_problems} if return_path else None)
            result["worker_return"] = facts
            result["launch_journal"] = {
                "known": bool(journal.get("known")),
                "problems": list(journal.get("problems") or [])}
            write_json(directory / "worker_result_outer.json", result)
            if return_problems or not journal.get("known"):
                print("[update-audit] migration return/journal not closed: "
                      f"return={return_problems} "
                      f"journal={journal.get('problems')}")
            _print_result(result)
            return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)
        runtime = configure_production_runtime()
        if runtime["problems"]:
            raise SystemExit(
                f"production runtime cannot be applied: {runtime['problems']}")
        acceptance_path = Path(package.get("acceptance_path") or "")
        if not acceptance_path.is_absolute():
            acceptance_path = ROOT / acceptance_path
        provider = RealTrainingProvider(
            ROOT, MODEL_CONFIG, device="cuda",
            acceptance_path=(acceptance_path if acceptance_path.is_file()
                             else None))
        provider.acceptance = package.get("acceptance")
        provider.acceptance_sha = package.get("acceptance_sha256")
        provider.roster = ((package.get("roster") or {}).get(
            "parameter_roster"))
        result = run_registered_audit(
            out, provider, mode="real", tolerance=contract,
            authorization_sha=auth_sha, authorization=record,
            admission_package=package, admission_sha=package_sha,
            r0_started_perf=_ENTRY_STARTED_PERF,
            resume_stage=(args.stage if args.resume else None))
        _print_result(result)
        return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)

    if args.authorization is not None or args.admission_package is not None:
        raise SystemExit(
            "authorization/admission inputs are only valid with --real")
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    tolerance = None
    contract_path = None
    if args.tolerance_contract is not None:
        tolerance, _sha, problems = load_tolerance_contract(
            args.tolerance_contract, recompute=False)
        if problems:
            raise SystemExit(f"tolerance contract refused: {problems}")
        contract_path = Path(args.tolerance_contract).expanduser().resolve()
    out = args.out if args.out is not None else (
        ROOT / "derived/audit/w1_real_training_update_audit/synthetic")
    out = _guard(out)
    verification_path = (None if contract_path is None
                         else _calibration_verification_path(args,
                                                             contract_path))
    verification = _load_calibration_verification(verification_path)
    if verification is not None:
        if tolerance is None or contract_path is None:
            raise SystemExit(
                "synthetic acceptance requires --tolerance-contract")
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"synthetic run root is not fresh: {out}")
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--synthetic-worker", "--out", str(out),
            "--tolerance-contract", str(contract_path),
            "--seed", str(args.seed), "--device", device]
        if args.synthetic_config_json is not None:
            command += ["--synthetic-config-json",
                        str(Path(args.synthetic_config_json).expanduser()
                            .resolve())]
        facts = _spawn_worker(
            command, out.parent / f"{out.name}_synthetic_worker.log")
        result = _worker_result(out / "audit_result.json")
        if result is None:
            print("[update-audit] synthetic worker did not return a result")
            print(f"[update-audit] worker_exit={facts['exit_code']} "
                  f"log={facts['log_path']}")
            return 3
        outer = synthetic_outer_return_record(
            started_perf_counter=facts["started_perf_counter"],
            finished_perf_counter=facts["finished_perf_counter"],
            conclusion=result.get("conclusion"), pid=facts["pid"],
            ppid=facts["ppid"], exit_code=facts["exit_code"],
            signal=facts["signal"])
        outer_path = out / "synthetic_return.json"
        write_json(outer_path, outer)
        acceptance = build_ordinary_synthetic_acceptance(
            out, result, roster=result.get("roster") or {},
            tolerance=tolerance,
            calibration_verification_record=verification,
            outer_return_path=outer_path,
            outer_return_sha256=sha256_file(outer_path))
        result["synthetic_acceptance"] = acceptance
        write_json(out / "audit_result.json", result)
        print(f"[update-audit] worker_exit={facts['exit_code']} "
              f"pid={facts['pid']} ppid={facts['ppid']}")
        print(f"[update-audit] synthetic acceptance={acceptance['path']}")
        print(f"[update-audit] sha256={acceptance['sha256']} "
              f"passed={acceptance['passed']}")
        _print_result(result)
        return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)
    result = synthetic_self_test(
        out, seed=args.seed, device=device, tolerance=tolerance)
    _print_result(result)
    return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)


if __name__ == "__main__":
    raise SystemExit(main())
