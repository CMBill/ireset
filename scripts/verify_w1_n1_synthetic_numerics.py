"""W1 N1 minimal synthetic verification CLI (plan v0.4 section 5).

Pure-CPU synthetic arithmetic only; no real network, optimizer, checkpoint or
dataset.  Emits ``n1_synthetic_numerics_v1`` case records and a closure
summary under a new output attempt directory.

Exit codes: 0 = all cases passed; 3 = incomplete (resource limit); 1 = hard
failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_n1_synthetic_numerics as n1  # noqa: E402
from ireset.eval.w1_n0_coordinate_flags import N0Error  # noqa: E402

DEFAULT_OUTPUT_ROOT = "derived/audit/w1_numerics_n0_n1/n1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--wall-cap-s", type=float, default=900.0)
    ap.add_argument("--reserve-s", type=float, default=60.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    budget = n1.N0Budget(wall_cap_s=args.wall_cap_s, reserve_s=args.reserve_s, t0=t0)
    out_root = Path(args.output_root)
    attempt_dirs = sorted(p for p in out_root.glob("attempt_*") if p.is_dir()) if out_root.exists() else []
    out_dir = out_root / f"attempt_{len(attempt_dirs) + 1:03d}"

    try:
        summary, cases = n1.run_synthetic_verification(budget)
    except (N0Error, n1.N1Error) as e:
        print(f"N1 hard failure: {e}", file=sys.stderr)
        return 1

    incomplete = bool(summary.get("budget_exceeded")) or budget.exceeded
    if incomplete:
        summary["all_cases_passed"] = False
        summary["label_verdict"] = "evidence_incomplete/resource_limit"
    else:
        summary["label_verdict"] = ("complete" if summary["all_cases_passed"]
                                    else "synthetic_failure")
    paths = n1.write_outputs(out_dir, summary, cases)

    print(json.dumps({
        "output_dir": str(out_dir),
        "all_cases_passed": summary["all_cases_passed"],
        "label_verdict": summary["label_verdict"],
        "go_no_go": summary["go_no_go"] if not incomplete else None,
        "per_class": summary["per_class"],
        "counters": summary["counters"],
        "budget": {k: summary["budget"][k] for k in
                   ("measured_wall_s_at_summary", "work_deadline_s",
                    "hard_deadline_s", "budget_exceeded", "stopped_at")},
        "outputs": paths,
    }, indent=1))
    if incomplete:
        return 3
    return 0 if summary["all_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
