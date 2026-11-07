"""N0 coordinate-level gap analysis CLI (plan v0.4 section 4).

Read-only CPU analysis over the saved T6 evidence root. No model, no
forward/backward/FK/AdamW, no checkpoint loading. Emits the frozen
``n0_coordinate_flags_v1`` artifacts under a new output attempt directory.

Exit codes: 0 = coverage complete; 3 = incomplete (resource limit or evidence
problems); 1 = hard failure (structural, nothing consumable written).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_n0_coordinate_flags as n0  # noqa: E402

DEFAULT_RUN_ROOT = ("derived/experiments/omniretarget_first/"
                    "20261102_2704709_w1_real_training_update_audit")
DEFAULT_OUTPUT_ROOT = "derived/audit/w1_numerics_n0_n1/n0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--analysis-path",
                    default="derived/audit/w1_real_training_update_audit/"
                            "gradient_numerics_retest/real_analysis/attempt_001/analysis.json")
    ap.add_argument("--wall-cap-s", type=float, default=900.0)
    ap.add_argument("--reserve-s", type=float, default=60.0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    budget = n0.N0Budget(wall_cap_s=args.wall_cap_s, reserve_s=args.reserve_s, t0=t0)
    out_root = Path(args.output_root)
    attempt_dirs = sorted(p for p in out_root.glob("attempt_*") if p.is_dir()) if out_root.exists() else []
    out_dir = out_root / f"attempt_{len(attempt_dirs) + 1:03d}"

    try:
        analyzer = n0.N0Analyzer(Path(args.run_root), out_dir, budget,
                                 analysis_path=Path(args.analysis_path))
        summary = analyzer.run()
    except n0.N0Error as e:
        print(f"N0 hard failure: {e}", file=sys.stderr)
        return 1

    print(json.dumps({
        "output_dir": str(out_dir),
        "coverage_complete": summary["coverage_complete"],
        "label_verdict": summary["label_verdict"],
        "total_coords": summary["total_coords"],
        "n_rows": summary["n_rows"],
        "event_replay": summary["event_replay"],
        "label_counts": summary["label_counts"],
        "budget": {k: summary["budget"][k] for k in
                   ("measured_wall_s_at_summary", "work_deadline_s", "hard_deadline_s",
                    "budget_exceeded", "stopped_at")},
    }, indent=1))
    return 0 if summary["coverage_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
