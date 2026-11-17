"""W1 real small-set fit diagnostic -- worker CLI (plan v0.4 §3.2/§6/§7/§8).

Two fail-closed modes:

  --mode r-admit   R metadata admission under the §8.1 R budget (120s/15s):
                   file byte hashes, NPZ headers and JSON only; no array
                   value, checkpoint, tokenizer, training module or FK is
                   touched.  Produces the sealed admission package the
                   §8.3 authorization binds.

  --mode chain     the AUTHORIZED real chain (D1 -> conditional D6 -> F).
                   Refuses without a sealed admission package and a §8.3
                   authorization whose checklist is entirely true; each
                   stage runs under its own §8.1 budget inside one uv
                   process.

The report is a candidate until the independent supervisor
(scripts/run_w1_smallset_fit_real_supervisor.py) proves that the uv
process actually returned within the registered cost.  Exit codes:
0 = pass, 1 = refused/failed gates, 3 = incomplete (resource limit).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_T0 = time.perf_counter()  # sampled before any heavy import below

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_smallset_fit_r_admission as radm  # noqa: E402
from ireset.eval.w1_n0_coordinate_flags import N0Error  # noqa: E402
from ireset.eval.w1_smallset_fit_diagnostic import (  # noqa: E402
    StageBudget, verify_return_coverage)

DEFAULT_R_ROOT = "derived/audit/w1_smallset_fit_diagnostic/r_admission"
DEFAULT_RUN_ROOT = "derived/experiments/omniretarget_first"


def _exclusive_attempt_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(int(p.name.split("_")[1]) for p in out_root.glob("attempt_*")
                      if p.is_dir() and p.name.split("_")[1].isdigit())
    start = (existing[-1] + 1) if existing else 1
    for n in range(start, start + 1000):
        candidate = out_root / f"attempt_{n:03d}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"no free attempt number under {out_root}")


def _run_r(args) -> dict:
    out_dir = Path(args.output_dir) if args.output_dir \
        else _exclusive_attempt_dir(Path(args.output_root or DEFAULT_R_ROOT))
    budget = StageBudget(stage="R", wall_cap_s=args.wall_cap_s,
                         reserve_s=args.reserve_s, t0=args.budget_t0)
    cache_root = Path(args.cache_root) if args.cache_root else None
    out = radm.run_r_admission(out_dir=out_dir, budget=budget,
                               cache_root=cache_root)
    return {"out_dir": str(out_dir), "mode": "r-admit",
            "cache_root": str(cache_root) if cache_root else None, **out}


def _run_chain(args) -> dict:
    run_root = Path(args.output_dir) if args.output_dir \
        else _exclusive_attempt_dir(Path(args.run_root or DEFAULT_RUN_ROOT))
    out = radm.run_real_chain(
        admission_path=Path(args.admission),
        authorization_path=Path(args.authorization),
        run_root=run_root, chain_t0=args.budget_t0,
        device="cpu" if args.fixture else args.device,
        fixture=args.fixture, impl_sha=args.impl_sha or None)
    return {"out_dir": str(run_root), "mode": "chain", **out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("r-admit", "chain"), required=True)
    output = ap.add_mutually_exclusive_group()
    output.add_argument("--output-dir",
                        help="exclusive directory reserved by the supervisor")
    output.add_argument("--output-root")
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT,
                    help="parent for chain mode attempt directories")
    ap.add_argument("--admission", help="sealed r_admission.json (chain mode)")
    ap.add_argument("--authorization", help="§8.3 authorization JSON (chain mode)")
    ap.add_argument("--cache-root",
                    help="r-admit mode: bind this cache attempt dir instead "
                         "of the SOURCE's registered cache (plan §41; the "
                         "bound cache must carry a valid migration record)")
    ap.add_argument("--impl-sha", help="expected implementation identity sha")
    ap.add_argument("--outer-t0-hex", help="supervisor monotonic start as float.hex")
    ap.add_argument("--launch-id", help="supervisor launch identity")
    ap.add_argument("--wall-cap-s", type=float, default=None,
                    help="R-mode wall cap (default 120 per plan §8.1)")
    ap.add_argument("--reserve-s", type=float, default=None,
                    help="R-mode in-cap sealing reserve (default 15)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fixture", action="store_true",
                    help="fixture world (tests only; never a production claim)")
    args = ap.parse_args()

    if bool(args.outer_t0_hex) != bool(args.launch_id):
        ap.error("--outer-t0-hex and --launch-id must be supplied together")
    if args.outer_t0_hex:
        try:
            budget_t0 = float.fromhex(args.outer_t0_hex)
        except ValueError:
            ap.error("invalid supervisor monotonic start")
        if not math.isfinite(budget_t0) or not 0 < budget_t0 <= _T0:
            ap.error("supervisor monotonic start must precede the CLI entry")
    else:
        budget_t0 = _T0
    args.budget_t0 = budget_t0
    if args.mode == "r-admit":
        args.wall_cap_s = 120.0 if args.wall_cap_s is None else args.wall_cap_s
        args.reserve_s = 15.0 if args.reserve_s is None else args.reserve_s
        runner = _run_r
    else:
        if not args.admission or not args.authorization:
            ap.error("--mode chain requires --admission and --authorization")
        if args.cache_root:
            # the chain binds whatever the ADMISSION froze -- a CLI override
            # would bypass the recorded cache_binding (plan §41)
            ap.error("--cache-root is r-admit only; the chain consumes the "
                     "admission package's cache_binding")
        args.wall_cap_s = args.wall_cap_s
        args.reserve_s = args.reserve_s
        runner = _run_chain

    try:
        out = runner(args)
    except (N0Error, radm.FitDiagnosticError) as exc:
        print(f"real fit worker hard failure: {exc}", file=sys.stderr)
        return 1

    summary_path = Path(out["summary"])
    disk = json.loads(summary_path.read_text())
    print(json.dumps({
        "output_dir": out["out_dir"],
        "mode": out["mode"],
        "label_verdict": disk.get("label_verdict"),
        "admission_passed": disk.get("admission_passed"),
        "problems": (disk.get("problems") or [])[:8],
        "cache_binding": (disk.get("package") or {}).get("cache_binding"),
        "d1": {k: disk.get("d1", {}).get(k)
               for k in ("local_complete_evidence", "local_mean_fit_passed",
                         "local_conclusion", "return_coverage")} if disk.get("d1") else None,
        "d1_handoff": disk.get("d1_handoff"),
        "d6": {k: disk.get("d6", {}).get(k)
               for k in ("local_complete_evidence", "local_mean_fit_passed",
                         "local_conclusion", "return_coverage")} if disk.get("d6") else None,
        "f": {k: disk.get("f", {}).get(k)
              for k in ("complete_evidence", "replay_passed", "mean_fit_passed",
                        "verdict", "return_coverage")} if disk.get("f") else None,
        "seal": disk.get("seal"),
        "budget": {k: (disk.get("budget") or {}).get(k)
                   for k in ("measured_wall_s_at_summary", "wall_cap_s",
                             "budget_exceeded", "stopped_at")},
        "outputs": {"summary": str(summary_path)},
    }, indent=1), flush=True)
    # necessary final disk read BEFORE the coverage sample (no I/O after a
    # passing sample; the failure branch re-verifies after its stderr write)
    disk = json.loads(summary_path.read_text())
    withdraw = (radm.R_WITHDRAW_ON_EXCEED if out["mode"] == "r-admit"
                else radm.CHAIN_WITHDRAW_ON_EXCEED)
    coverage = verify_return_coverage(summary_path, StageBudget(
        stage="R" if out["mode"] == "r-admit" else "chain",
        wall_cap_s=args.wall_cap_s or 6120.0,
        reserve_s=args.reserve_s or 60.0, t0=budget_t0),
        withdraw=dict(withdraw))
    if not coverage["covered"]:
        print(json.dumps({"return_coverage": coverage,
                          "decision": "fail_closed"}),
              file=sys.stderr, flush=True)
        verify_return_coverage(summary_path, StageBudget(
            stage="R" if out["mode"] == "r-admit" else "chain",
            wall_cap_s=args.wall_cap_s or 6120.0,
            reserve_s=args.reserve_s or 60.0, t0=budget_t0),
            withdraw=dict(withdraw))
        return 3
    sealed = (disk.get("seal") or {}).get("status") == "sealed_within_budget"
    if not sealed or (disk.get("budget") or {}).get("budget_exceeded"):
        return 3
    if out["mode"] == "r-admit":
        return 0 if disk.get("admission_passed") is True else 1
    d1 = disk.get("d1") or {}
    d6 = disk.get("d6") or {}
    f = disk.get("f") or {}
    f_stages = f.get("stages") or []
    d6_ran = disk.get("d6_ran") is True
    includes = set(disk.get("authorized_includes") or [])
    # evidence completeness decides the exit code -- the SCIENTIFIC verdict
    # (fit_pass / fit_insufficient) lives in the F report; a D6 skipped by
    # the plan's own gates (§7.2/§7.4 item 3) is a complete chain, not a
    # failure of evidence.
    d6_ok = (
        (d6_ran and d1.get("local_mean_fit_passed") is True
         and (disk.get("d1_handoff") or {}).get("handoff_ready") is True
         and d6.get("local_complete_evidence") is True
         and d6.get("return_coverage") is True
         and not d6.get("call_matrix_problems"))
        or (not d6_ran and d6.get("skipped_reason") in
            ("d1_incomplete", "d1_mean_fit_gate_not_passed",
             "d6_not_authorized", "handoff_not_ready")))
    accept = (
        d1.get("local_complete_evidence") is True
        and d1.get("return_coverage") is True
        and not d1.get("call_matrix_problems")
        and d6_ok
        and f.get("complete_evidence") is True
        and f.get("return_coverage") is True
        and f.get("replay_passed") is True
        and len(f_stages) == (2 if d6_ran else 1)
        and {s.get("stage") for s in f_stages}
        == ({"D1", "D6"} if d6_ran else {"D1"})
        and all(s.get("replay_passed") is True for s in f_stages)
        and (d6_ran or d1.get("local_mean_fit_passed") is False
             or d6.get("skipped_reason") == "d6_not_authorized")
        and bool(includes))
    return 0 if accept else 1


if __name__ == "__main__":
    raise SystemExit(main())
