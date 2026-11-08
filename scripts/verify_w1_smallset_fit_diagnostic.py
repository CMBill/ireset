"""W1 I-phase small-set fit diagnostic — synthetic acceptance CLI (plan v0.4 §6).

This entry is SYNTHETIC-ONLY: it builds an offline synthetic world (synthetic
tokenizer/joint range/teacher windows/FK stub) and runs

  mini D1 -> local verification -> handoff resolution -> mini D6 (only when
  the D1 mean gate passed) -> F fresh bitwise replay -> counterexample
  battery,

under one monotonic budget clock.  It never reads real data, never loads a
real checkpoint and never claims CUDA/float32 production equivalence.  The
real R freeze and real D1/D6/F executions are separate, later, authorized
deliverables (plan §8.3) and have no entry here.

The budget clock starts before this script imports any heavy module; the
final summary write is sealed against the hard deadline (costs only
increase) and the return coverage is proven afterwards.

Exit codes: 0 = acceptance passed (frozen defaults + battery + local/F
facts); 3 = incomplete (resource limit); 1 = acceptance failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_T0 = time.perf_counter()  # sampled before any heavy import below

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_smallset_fit_diagnostic as diag  # noqa: E402
from ireset.eval.w1_n0_coordinate_flags import N0Error  # noqa: E402

DEFAULT_OUTPUT_ROOT = "derived/audit/w1_smallset_fit_diagnostic"
MAX_ATTEMPT_PROBES = 1000

WITHDRAW_ON_EXCEED = {
    "kind": "synthetic_acceptance",
    "all_counterexamples_caught": False,
    "label_verdict": "budget_exceeded_after_seal",
}


def _exclusive_attempt_dir(out_root: Path) -> Path:
    existing = sorted(int(p.name.split("_")[1]) for p in out_root.glob("attempt_*")
                      if p.is_dir() and p.name.split("_")[1].isdigit()) \
        if out_root.exists() else []
    start = (existing[-1] + 1) if existing else 1
    for n in range(start, start + MAX_ATTEMPT_PROBES):
        candidate = out_root / f"attempt_{n:03d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"no free attempt number under {out_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--wall-cap-s", type=float, default=900.0)
    ap.add_argument("--reserve-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--d1-steps", type=int, default=300)
    ap.add_argument("--d6-steps", type=int, default=300)
    args = ap.parse_args()

    budget = diag.StageBudget(stage="I-synthetic-acceptance",
                              wall_cap_s=args.wall_cap_s,
                              reserve_s=args.reserve_s, t0=_T0)
    try:
        out_dir = _exclusive_attempt_dir(Path(args.output_root))
        facts = diag.run_synthetic_acceptance(
            budget, out_dir, seed=args.seed, d1_steps=args.d1_steps,
            d6_steps=args.d6_steps)
    except (N0Error, diag.FitDiagnosticError) as e:
        print(f"I synthetic acceptance hard failure: {e}", file=sys.stderr)
        return 1

    incomplete = bool(budget.exceeded)
    if incomplete:
        facts["label_verdict"] = "evidence_incomplete/resource_limit"
        facts["all_counterexamples_caught"] = False
    else:
        facts["label_verdict"] = "complete"
    paths = diag.write_acceptance_outputs(out_dir, facts, budget,
                                          withdraw=dict(WITHDRAW_ON_EXCEED))
    # P1 (round-2 review): the flushed stdout output is part of the process
    # return and must be INSIDE the checked window.  Print the disk-derived
    # payload FIRST (flush=True), only then sample the return coverage; the
    # failure branch re-verifies after its stderr diagnostic so the
    # diagnostic time is covered too (costs only increase).
    disk = paths["summary_doc"]
    print(json.dumps({
        "output_dir": str(out_dir),
        "label_verdict": disk.get("label_verdict"),
        "frozen_plan_defaults_problems": disk.get("frozen_plan_defaults_problems"),
        "d1": disk.get("d1"),
        "d1_handoff": disk.get("d1_handoff"),
        "d6": disk.get("d6"),
        "f": disk.get("f"),
        "all_counterexamples_caught": disk.get("all_counterexamples_caught"),
        "counterexample_ids": disk.get("counterexample_ids"),
        "seal": disk.get("seal"),
        "budget": {k: (disk.get("budget") or {}).get(k)
                   for k in ("measured_wall_s_at_summary", "wall_cap_s",
                             "budget_exceeded", "stopped_at")},
        "outputs": {"summary": paths["summary"]},
    }, indent=1), flush=True)
    cov = diag.verify_return_coverage(Path(paths["summary"]), budget,
                                      withdraw=dict(WITHDRAW_ON_EXCEED))
    if not cov["covered"]:
        print(json.dumps({"return_coverage": cov, "decision": "fail_closed"}),
              file=sys.stderr, flush=True)
        diag.verify_return_coverage(Path(paths["summary"]), budget,
                                    withdraw=dict(WITHDRAW_ON_EXCEED))
        return 3
    # Fail-closed: re-read the on-disk sealed summary AFTER the coverage
    # proof — the final disk state, not pre-write memory, decides the exit.
    disk = json.loads(Path(paths["summary"]).read_text())
    sealed = (disk.get("seal") or {}).get("status") == "sealed_within_budget"
    if not sealed or (disk.get("budget") or {}).get("budget_exceeded"):
        return 3
    accept = (disk.get("frozen_plan_defaults_problems") == []
              and disk.get("all_counterexamples_caught") is True
              and (disk.get("d1") or {}).get("local_complete_evidence") is True
              and not (disk.get("d1") or {}).get("call_matrix_problems")
              and not (disk.get("d1") or {}).get("report_problems")
              and (disk.get("f") or {}).get("replay_passed") is True)
    return 0 if accept else 1


if __name__ == "__main__":
    raise SystemExit(main())
