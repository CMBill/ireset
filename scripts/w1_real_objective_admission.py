"""Build the reviewable W1 real-objective audit admission package.

This entry is read-only with respect to real assets: it hashes metadata files,
re-derives the frozen four-window candidates, and writes a package JSON under
``derived/audit/``.  It never loads window arrays, schedules, cache contents or
checkpoints, never runs a model, and never creates a cohort or registered run.
A real run still requires the section 8 authorization record.

  uv run --locked python scripts/w1_real_objective_admission.py \
    --tolerance-contract derived/audit/w1_real_objective_audit/tolerance_contract.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.eval import w1_real_objective_audit as audit  # noqa: E402
from ireset.eval.w1_real_admission import (  # noqa: E402
    build_admission_package,
    write_package,
)


def _guard_out(out: Path) -> Path:
    resolved = Path(out).expanduser().resolve()
    allowed = (ROOT / "derived" / "audit").resolve()
    if not (resolved == allowed or allowed in resolved.parents):
        raise SystemExit(
            f"admission packages must be written under {allowed}, got {resolved}")
    for name in ("derived/experiments", "resources", "third_party",
                 "src", "scripts", "tests"):
        forbidden = (ROOT / name).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise SystemExit(f"refusing to write an admission package under {name}/")
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--tolerance-contract", type=Path,
        default=ROOT / "derived/audit/w1_real_objective_audit/tolerance_contract.json",
        help="frozen synthetic calibration contract")
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "derived/audit/w1_real_objective_audit/admission",
        help="package root under derived/audit/ (attempt dirs never overwritten)")
    args = ap.parse_args()

    contract, contract_sha, problems = audit.load_tolerance_contract(
        args.tolerance_contract)
    if problems:
        raise SystemExit(f"tolerance contract refused: {problems}")
    if (contract.get("mode") != "synthetic_calibrated"
            or not contract.get("evidence_sha256")):
        raise SystemExit(
            "tolerance contract is not a completed synthetic calibration; "
            "run --calibrate-synthetic first")

    out = _guard_out(args.out)
    package = build_admission_package(
        ROOT, tolerance_contract=contract,
        tolerance_contract_path=Path(args.tolerance_contract),
        tolerance_contract_sha256=contract_sha)
    path, package_sha = write_package(out, package)
    audit.write_json(path.parent / "package_index.json", {
        "schema": "w1_real_objective_admission_index_v1",
        "package": audit.path_for_record(path),
        "package_sha256": package_sha,
        "tolerance_contract": audit.path_for_record(args.tolerance_contract),
        "tolerance_contract_sha256": contract_sha,
        "calibration_evidence_sha256": contract.get("evidence_sha256"),
        "passed": package["passed"],
        "problems": package["problems"],
        "authorized": False,
        "run": "not_run",
    })
    print(f"[admission] package={path}")
    print(f"[admission] sha256={package_sha}")
    print(f"[admission] passed={package['passed']} "
          f"problems={len(package['problems'])}")
    for problem in package["problems"][:20]:
        print(f"[admission]   {problem}")
    print(f"[admission] authorized=false run=not_run "
          f"authorization={package['authorization']['entry']}")
    return 0 if package["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
