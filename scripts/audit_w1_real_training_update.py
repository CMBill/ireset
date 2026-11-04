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
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.eval.w1_real_training_update_audit import (  # noqa: E402
    ACCOUNTING_SCHEMA_V2,
    ADMISSION_SCHEMA,
    AUDIT_SCHEMA,
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
    claim_launch,
    compact_migration_result,
    configure_production_runtime,
    implementation_files_sha256,
    implementation_identity,
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


def _worker_result(path: Path) -> dict | None:
    if not Path(path).is_file():
        return None
    try:
        record = strict_read_json(path)
    except (OSError, ValueError, TypeError):
        return None
    return record if isinstance(record, dict) else None


def _fresh_migration_root_problems(root: Path) -> list[str]:
    problems: list[str] = []
    for name in (BINDINGS_DIR, LAUNCH_JOURNAL_DIR, EXECUTION_EVENTS_DIR):
        if (Path(root) / name).exists():
            problems.append(f"migration_root_has_{name}")
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
    mode.add_argument("--migration-spec", action="store_true",
                      help="freeze the read-only binding spec outside the root")
    mode.add_argument("--gate-b-v2", action="store_true",
                      help="bind the ordinary and migration synthetic "
                           "acceptances into Gate B v2")
    mode.add_argument("--migration-worker", action="store_true",
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

    if args.migration_worker:
        if args.job is None:
            raise SystemExit("--migration-worker requires --job")
        job = strict_read_json(Path(args.job).expanduser().resolve())
        if not isinstance(job, Mapping):
            raise SystemExit("migration worker job must be an object")
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
            result = run_binding_migration_audit(
                root, provider, mode="real", tolerance=contract,
                authorization_sha=auth_sha, authorization=authorization,
                admission_package=package, admission_sha=package_sha,
                binding_spec_path=spec_path, invocation_id=invocation_id,
                fixed_root=HISTORY_ROOT, project_root=ROOT)
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
            load_external_binding_material,
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
        record, auth_sha, auth_problems = load_authorization(args.authorization)
        if auth_problems:
            raise SystemExit(f"authorization refused: {auth_problems}")
        contract, contract_sha, contract_problems = load_tolerance_contract(
            args.tolerance_contract, recompute=False)
        if contract_problems:
            raise SystemExit(f"tolerance contract refused: {contract_problems}")
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
        out = Path(args.out).expanduser().resolve()
        base = (ROOT / HISTORY_ROOT).resolve()
        if out.parent != base:
            raise SystemExit(f"--out must be a direct child of {base}, got {out}")
        if args.migration:
            if args.resume or args.stage is not None:
                raise SystemExit("--migration cannot be combined with --resume")
            if args.binding_spec is None or args.invocation_id is None:
                raise SystemExit(
                    "--migration requires --binding-spec and --invocation-id")
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
                raise SystemExit(f"history snapshot missing: {snapshot_path}")
            snapshot = strict_read_json(snapshot_path)
            binding, binding_problems = load_external_binding_material(
                snapshot_path, snapshot_sha, spec_path, spec_sha,
                project_root=ROOT, require_current_implementation=True)
            if binding is None or binding_problems:
                raise SystemExit(
                    f"migration binding refused: {binding_problems}")
            migration = record.get("migration") or {}
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
            if ((spec.get("bindings") or {}).get("real_admission") or {}).get(
                    "sha256") != package_sha:
                binding_problems.append("spec_real_admission")
            if _fresh_migration_root_problems(out):
                binding_problems += _fresh_migration_root_problems(out)
            if not (out / "cohort.json").is_file():
                binding_problems.append("migration_root_missing_marker")
            if binding_problems:
                raise SystemExit(
                    f"migration launch refused: {binding_problems}")
            stage_binding(
                out, spec_path=spec_path, spec_sha256=spec_sha,
                snapshot_sha256=snapshot_sha,
                invocation_id=args.invocation_id)
            claim_launch(
                out, authorization_sha256=auth_sha, spec_sha256=spec_sha,
                legacy_head_sha256=snapshot["legacy_head_sha256"],
                marker_sha256=snapshot["marker_sha256"],
                invocation_id=args.invocation_id,
                t0_perf_counter=_ENTRY_STARTED_PERF,
                target=dict(spec["stage_plan"][0]))
            directory = out / LAUNCH_JOURNAL_DIR / args.invocation_id
            job = {
                "mode": "real", "root": str(out),
                "spec_path": str(spec_path),
                "authorization_path": args.authorization.resolve(),
                "admission_package_path": args.admission_package.resolve(),
                "tolerance_contract_path": args.tolerance_contract.resolve(),
                "invocation_id": args.invocation_id,
                "result_path": str(directory / "worker_result_inner.json"),
            }
            job_path = directory / "worker_job.json"
            write_json_exclusive(job_path, job)
            facts = _spawn_worker(
                [sys.executable, str(Path(__file__).resolve()),
                 "--migration-worker", "--job", str(job_path)],
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
        if not args.resume and out.exists() and any(out.iterdir()):
            raise SystemExit(f"real run root is not fresh: {out}")
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
