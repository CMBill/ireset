"""N0 coordinate-level gap analysis CLI (plan v0.4 section 4, format v2).

Read-only CPU analysis over the saved T6 evidence root. No model, no
forward/backward/FK/AdamW, no checkpoint loading. Emits the frozen
``n0_coordinate_flags_v2`` artifacts under a new exclusively-created output
attempt directory.

The budget clock starts before this script imports any heavy module; the
final summary write is sealed against the hard deadline (costs only increase;
a slow write that crosses the hard deadline revokes the completion claim).

Exit codes: 0 = coverage complete; 3 = incomplete (resource limit or evidence
problems); 1 = hard failure (structural, nothing consumable written).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_T0 = time.perf_counter()  # sampled before any heavy import below

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_n0_coordinate_flags as n0  # noqa: E402

DEFAULT_RUN_ROOT = ("derived/experiments/omniretarget_first/"
                    "20261102_2704709_w1_real_training_update_audit")
DEFAULT_OUTPUT_ROOT = "derived/audit/w1_numerics_n0_n1/n0"
MAX_ATTEMPT_PROBES = 1000


def _exclusive_attempt_dir(out_root: Path) -> Path:
    """Create attempt_NNN with exclusive mkdir; never overwrite an existing
    attempt (a pre-existing attempt_003 yields attempt_004, not a rewrite)."""
    existing = sorted(int(p.name.split("_")[1]) for p in out_root.glob("attempt_*")
                      if p.is_dir() and p.name.split("_")[1].isdigit()) if out_root.exists() else []
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
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--analysis-path",
                    default="derived/audit/w1_real_training_update_audit/"
                            "gradient_numerics_retest/real_analysis/attempt_001/analysis.json")
    ap.add_argument("--wall-cap-s", type=float, default=900.0)
    ap.add_argument("--reserve-s", type=float, default=60.0)
    args = ap.parse_args()

    budget = n0.N0Budget(wall_cap_s=args.wall_cap_s, reserve_s=args.reserve_s, t0=_T0)
    try:
        out_dir = _exclusive_attempt_dir(Path(args.output_root))
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
        "old_chain_comparison": summary["old_chain_comparison"],
        "source_closure": summary["source_closure"],
        "label_counts": summary["label_counts"],
        "n_rows_with_anomalies": summary["n_rows_with_anomalies"],
        "seal": summary.get("seal"),
        "budget": {k: summary["budget"][k] for k in
                   ("measured_wall_s_at_summary", "work_deadline_s", "hard_deadline_s",
                    "budget_exceeded", "stopped_at")},
    }, indent=1), flush=True)
    # Independent post-output return-coverage proof (plan §8.2): the summary
    # read, this flushed output and the check itself are all inside the
    # checked elapsed time; the proof is bound to return_observation.json.
    # A violation withdraws the completion claim on disk and fails closed.
    cov = n0.verify_return_coverage(
        out_dir / "summary.json", budget,
        withdraw={"budget_exceeded_after_seal": True,
                  "coverage_complete": False,
                  "label_verdict": "budget_exceeded_after_seal"})
    if not cov["covered"]:
        print(json.dumps({"return_coverage": cov, "decision": "fail_closed"}),
              file=sys.stderr, flush=True)
        return 3
    # Fail-closed: run() re-reads the sealed summary from disk, so this
    # decision uses the on-disk terminal state, never pre-write memory.
    sealed = (summary.get("seal") or {}).get("status") == "sealed_within_budget"
    return 0 if sealed and summary.get("coverage_complete") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
