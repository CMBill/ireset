"""W1 real training-decode / FK geometry / derivative audit entry (I/C).

Default mode is the pure synthetic self-test.  Real execution is refused
unless a hash-bound authorization record, a frozen synthetic calibration
contract and a passed admission package are supplied; see
``docs/plans/omniretarget_w1_real_fk_geometry_derivative_audit_prereg_draft.md``
section 9.2.  This repository state has neither, so the real path is not
runnable here.

Run the synthetic R0->G->D->F self-test:
  uv run --locked python scripts/audit_w1_real_fk.py --synthetic

Write/verify the frozen synthetic tolerance contract:
  uv run --locked python scripts/audit_w1_real_fk.py --calibrate-synthetic

Write the read-only admission package:
  uv run --locked python scripts/audit_w1_real_fk.py --admission

Real (only after explicit user authorization):
  uv run --locked python scripts/audit_w1_real_fk.py --real \
    --out derived/experiments/omniretarget_first/<run> \
    --authorization <authorization.json> \
    --tolerance-contract <tolerance_contract.json> \
    --admission-package <admission_package.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.eval.w1_real_fk_audit import (  # noqa: E402
    ADMISSION_SCHEMA,
    AUDIT_SCHEMA,
    COHORT_ID,
    HISTORY_ROOT,
    PARENT_FILES,
    RealFkProvider,
    admission_package_problems,
    authorization_problems,
    build_admission_package,
    build_default_tolerance_contract,
    load_authorization,
    load_tolerance_contract,
    run_registered_audit,
    sha256_file,
    synthetic_calibration,
    synthetic_self_test,
    tolerance_contract_problems,
    write_json,
    write_package,
)

_EXIT = {
    "synthetic_contract_passed": 0,
    "real_saved_fk_covered_contract_passed": 0,
    "real_saved_fk_mismatch": 2,
    "synthetic_pipeline_mismatch": 2,
    "audit_incomplete": 3,
}


def _guard(out: Path) -> Path:
    resolved = Path(out).expanduser().resolve()
    experiments = (ROOT / "derived" / "experiments").resolve()
    if resolved == experiments or experiments in resolved.parents:
        raise SystemExit(
            f"refusing to write I/C artifacts into a research experiment root: {resolved}")
    for name in ("resources", "third_party", "src", "scripts", "tests"):
        forbidden = (ROOT / name).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise SystemExit(f"refusing to write under {name}/: {resolved}")
    return resolved


def _print_result(result: dict) -> None:
    print(f"[fk-audit] schema={AUDIT_SCHEMA} cohort={COHORT_ID}")
    print(f"[fk-audit] mode={result.get('mode')} "
          f"conclusion={result.get('conclusion')}")
    print(f"[fk-audit] complete_evidence={result.get('complete_evidence')} "
          f"covered_contract_passed={result.get('covered_contract_passed')}")
    for name in ("problems", "gaps", "verification_problems"):
        values = result.get(name) or []
        if values:
            print(f"[fk-audit] {name}={len(values)}")
            for value in values[:20]:
                print(f"[fk-audit]   {value}")
    counts = (result.get("calls") or {}).get("counts")
    if counts:
        print(f"[fk-audit] calls={counts}")
    if result.get("summary_path"):
        print(f"[fk-audit] summary={result['summary_path']}")


def _resolve_tolerance_for_admission(args) -> tuple[dict, Path, str]:
    path = (args.tolerance_contract or
            ROOT / "derived/audit/w1_real_fk_audit/tolerance_contract.json")
    contract, sha, problems = load_tolerance_contract(path)
    if problems:
        raise SystemExit(f"tolerance contract refused: {problems}")
    if contract.get("mode") != "synthetic_calibrated":
        raise SystemExit("run --calibrate-synthetic first")
    return contract, path, sha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true",
                      help="run the analytic four-window R0->G->D->F self-test")
    mode.add_argument("--calibrate-synthetic", action="store_true",
                      help="write/verify the frozen synthetic calibration")
    mode.add_argument("--admission", action="store_true",
                      help="write the read-only admission package")
    mode.add_argument("--real", action="store_true",
                      help="registered real audit (authorization + admission)")
    ap.add_argument("--out", type=Path, default=None,
                    help="synthetic/admission root or real run root")
    ap.add_argument("--tolerance-out", type=Path, default=None,
                    help="--calibrate-synthetic output file")
    ap.add_argument("--tolerance-contract", type=Path, default=None,
                    help="frozen calibration contract")
    ap.add_argument("--authorization", type=Path, default=None,
                    help="real mode: hash-bound user authorization record")
    ap.add_argument("--admission-package", type=Path, default=None,
                    help="real mode: passed admission package")
    ap.add_argument("--seed", type=int, default=20261028,
                    help="synthetic fixture seed (ignored by real mode)")
    ap.add_argument("--device", type=str, default="auto",
                    choices=("auto", "cpu", "cuda"))
    args = ap.parse_args()

    if args.calibrate_synthetic:
        out = (args.tolerance_out or args.out or
               ROOT / "derived/audit/w1_real_fk_audit/tolerance_contract.json")
        out = Path(out).expanduser().resolve()
        if out.suffix != ".json":
            out = out / "tolerance_contract.json"
        _guard(out.parent)
        contract = synthetic_calibration(seed=args.seed)
        problems = tolerance_contract_problems(contract)
        if problems:
            raise SystemExit(f"calibration invalid: {problems}")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, contract)
        print(f"[fk-audit] tolerance contract={out}")
        print(f"[fk-audit] sha256={sha256_file(out)}")
        print(f"[fk-audit] evidence_sha256={contract['evidence_sha256']}")
        return 0

    if args.admission:
        contract, path, sha = _resolve_tolerance_for_admission(args)
        out = (args.out or
               ROOT / "derived/audit/w1_real_fk_audit/admission")
        out = _guard(out)
        package = build_admission_package(
            ROOT, contract, tolerance_contract_path=path,
            tolerance_contract_sha256=sha)
        package_path, package_sha = write_package(out, package)
        print(f"[fk-audit] admission package={package_path}")
        print(f"[fk-audit] sha256={package_sha}")
        print(f"[fk-audit] passed={package['passed']} "
              f"problems={len(package['problems'])}")
        for problem in package["problems"][:20]:
            print(f"[fk-audit]   {problem}")
        return 0 if package["passed"] else 2

    if args.real:
        if args.authorization is None:
            raise SystemExit("--real requires --authorization (section 9.2 is not checked)")
        if args.tolerance_contract is None:
            raise SystemExit("--real requires --tolerance-contract")
        if args.admission_package is None:
            raise SystemExit("--real requires --admission-package")
        if args.out is None:
            raise SystemExit("--real requires --out (direct child of the history root)")
        # Validate every gate before constructing a provider or run root.
        record, auth_sha, auth_problems = load_authorization(args.authorization)
        if auth_problems:
            raise SystemExit(f"authorization refused: {auth_problems}")
        contract, contract_sha, contract_problems = load_tolerance_contract(
            args.tolerance_contract)
        if contract_problems:
            raise SystemExit(f"tolerance contract refused: {contract_problems}")
        try:
            package = json.loads(Path(args.admission_package).read_text())
        except (OSError, ValueError):
            raise SystemExit("admission package unreadable")
        package_sha = sha256_file(args.admission_package)
        pkg_problems = admission_package_problems(package)
        if pkg_problems:
            raise SystemExit(f"admission package refused: {pkg_problems}")
        if record.get("tolerance_evidence_sha256") != contract.get("evidence_sha256"):
            raise SystemExit("authorization does not bind this calibration")
        if record.get("admission_package_sha256") != package_sha:
            raise SystemExit("authorization does not bind this admission package")
        out = Path(args.out).expanduser().resolve()
        base = (ROOT / HISTORY_ROOT).resolve()
        if out.parent != base:
            raise SystemExit(f"--out must be a direct child of {base}, got {out}")
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"real run root is not fresh: {out}")
        device = args.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        provider = RealFkProvider(ROOT, device=device)
        result = run_registered_audit(
            out, provider, mode="real", tolerance=contract,
            authorization_sha=auth_sha, authorization=record,
            admission_package=package, admission_sha=package_sha)
        result["authorization_sha256"] = auth_sha
        result["tolerance_contract_sha256"] = contract_sha
        result["admission_package_sha256"] = package_sha
        _print_result(result)
        return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)

    if args.authorization is not None or args.admission_package is not None:
        raise SystemExit(
            "authorization/admission inputs are only valid with --real")
    if args.synthetic and args.tolerance_contract is not None:
        raise SystemExit("--tolerance-contract is only valid with --real/--admission")
    out = args.out if args.out is not None else (
        ROOT / "derived/audit/w1_real_fk_audit")
    out = _guard(out)
    result = synthetic_self_test(out, seed=args.seed)
    _print_result(result)
    return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)


if __name__ == "__main__":
    raise SystemExit(main())
