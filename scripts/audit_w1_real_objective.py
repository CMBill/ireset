"""W1 real training-objective audit entry (pre-registration draft).

Default mode is the pure synthetic self-test.  Real execution is refused
unless a hash-bound authorization record and a frozen tolerance contract are
supplied; see ``docs/plans/omniretarget_w1_real_objective_audit_prereg_draft.md``
section 8.  This repository state intentionally has neither, so the real path
is not runnable here.

Run synthetic self-test:
  uv run --locked python scripts/audit_w1_real_objective.py --synthetic

Calibrate/write the synthetic tolerance contract:
  uv run --locked python scripts/audit_w1_real_objective.py --calibrate-synthetic

Real (only after explicit user authorization):
  uv run --locked python scripts/audit_w1_real_objective.py --real \
    --out derived/experiments/omniretarget_first/<run> \
    --authorization <authorization.json> \
    --tolerance-contract <tolerance_contract.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.eval.w1_real_objective_audit import (  # noqa: E402
    AUDIT_SCHEMA,
    COHORT_ID,
    HISTORY_ROOT,
    RealW1ObjectiveProvider,
    authorization_problems,
    build_tolerance_contract,
    load_authorization,
    load_tolerance_contract,
    run_registered_audit,
    synthetic_calibration,
    synthetic_self_test,
    tolerance_contract_problems,
    write_json,
)

_EXIT = {
    "synthetic_contract_passed": 0,
    "real_batch_covered_contract_passed": 0,
    "real_batch_objective_mismatch": 2,
    "synthetic_pipeline_mismatch": 2,
    "audit_incomplete": 3,
}


def _guard_synthetic_out(out: Path) -> Path:
    resolved = Path(out).expanduser().resolve()
    experiments = (ROOT / "derived" / "experiments").resolve()
    if resolved == experiments or experiments in resolved.parents:
        raise SystemExit(
            "refusing to write a synthetic audit into a research experiment "
            f"root: {resolved}")
    for name in ("resources", "third_party", "src", "scripts", "tests"):
        forbidden = (ROOT / name).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise SystemExit(f"refusing to write audit artifacts under {name}/: {resolved}")
    return resolved


def _print_result(result: dict) -> None:
    print(f"[audit] schema={AUDIT_SCHEMA} cohort={COHORT_ID}")
    print(f"[audit] mode={result.get('mode')} conclusion={result.get('conclusion')}")
    print(f"[audit] complete_evidence={result.get('complete_evidence')} "
          f"covered_contract_passed={result.get('covered_contract_passed')}")
    if result.get("problems"):
        print(f"[audit] problems={len(result['problems'])}")
        for problem in result["problems"][:20]:
            print(f"[audit]   {problem}")
    if result.get("gaps"):
        print(f"[audit] gaps={len(result['gaps'])}")
        for gap in result["gaps"][:20]:
            print(f"[audit]   {gap}")
    if result.get("summary_path"):
        print(f"[audit] summary={result['summary_path']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true",
                      help="run the analytic four-window self-test (default)")
    mode.add_argument("--calibrate-synthetic", action="store_true",
                      help="write the frozen synthetic tolerance contract")
    mode.add_argument("--real", action="store_true",
                      help="registered real audit (authorization required)")
    ap.add_argument("--out", type=Path, default=None,
                    help="synthetic artifact root or real run root")
    ap.add_argument("--authorization", type=Path, default=None,
                    help="real mode: hash-bound user authorization record")
    ap.add_argument("--tolerance-contract", type=Path, default=None,
                    help="real mode: frozen synthetic tolerance contract")
    ap.add_argument("--tolerance-out", type=Path, default=None,
                    help="--calibrate-synthetic output file")
    ap.add_argument("--seed", type=int, default=20261026,
                    help="synthetic fixture seed (ignored by real mode)")
    ap.add_argument("--device", type=str, default="auto",
                    choices=("auto", "cpu", "cuda"))
    args = ap.parse_args()

    if args.calibrate_synthetic:
        out = (args.tolerance_out or
               (args.out if args.out is not None else
                ROOT / "derived/audit/w1_real_objective_audit/tolerance_contract.json"))
        out = Path(out).expanduser().resolve()
        if out.suffix != ".json":
            out = out / "tolerance_contract.json"
        _guard_synthetic_out(out.parent)
        contract = synthetic_calibration()
        problems = tolerance_contract_problems(contract)
        if problems:
            raise SystemExit(f"tolerance contract invalid: {problems}")
        if contract.get("mode") != "synthetic_calibrated" or not contract.get("evidence_sha256"):
            raise SystemExit("synthetic calibration did not produce a frozen contract")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, contract)
        print(f"[audit] tolerance contract={out}")
        print(f"[audit] sha256={__import__('hashlib').sha256(out.read_bytes()).hexdigest()}")
        return 0

    if args.real:
        if args.authorization is None:
            raise SystemExit("--real requires --authorization (section 8 is not checked)")
        if args.tolerance_contract is None:
            raise SystemExit("--real requires --tolerance-contract")
        if args.out is None:
            raise SystemExit("--real requires --out (direct child of the history root)")
        record, auth_sha, auth_problems = load_authorization(args.authorization)
        if auth_problems:
            raise SystemExit(f"authorization refused: {auth_problems}")
        contract, contract_sha, contract_problems = load_tolerance_contract(
            args.tolerance_contract)
        if contract_problems:
            raise SystemExit(f"tolerance contract refused: {contract_problems}")
        if (contract.get("mode") != "synthetic_calibrated"
                or not contract.get("evidence_sha256")):
            raise SystemExit(
                "tolerance contract is not a completed synthetic calibration; "
                "run --calibrate-synthetic first")
        out = Path(args.out).expanduser().resolve()
        base = (ROOT / HISTORY_ROOT).resolve()
        if out.parent != base:
            raise SystemExit(
                f"--out must be a direct child of {base}, got {out}")
        device = args.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        provider = RealW1ObjectiveProvider(ROOT, device=device)
        result = run_registered_audit(
            out, provider, mode="real", tolerance=contract,
            authorization_sha=auth_sha, authorization=record,
        )
        result["authorization_sha256"] = auth_sha
        result["tolerance_contract_sha256"] = contract_sha
        _print_result(result)
        return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)

    # Default and explicit --synthetic are the same pure self-test.
    if args.authorization is not None or args.tolerance_contract is not None:
        raise SystemExit(
            "authorization/tolerance-contract inputs are only valid with --real "
            "(the synthetic self-test does not consume real records)")
    out = args.out if args.out is not None else (
        ROOT / "derived/audit/w1_real_objective_audit")
    out = _guard_synthetic_out(out)
    result = synthetic_self_test(out, seed=args.seed)
    _print_result(result)
    return _EXIT.get(result.get("conclusion", "audit_incomplete"), 3)


if __name__ == "__main__":
    raise SystemExit(main())
