"""Acceptance summary for the cpu_parallel R3 six-segment parallel run.

Computes the plan section 6.4 gates from the frozen serial baseline and
the measured parallel report:
- throughput: speedup = serial_sum / parallel_wall must reach >= 75% of
  the ideal-parallel upper bound (serial_sum / max_segment);
- per-segment regression: parallel segment wall <= 1.05 x frozen serial
  for the MEDIAN segment (cache must not slow anything down).

Run: uv run --locked python scripts/cpu_parallel_acceptance_summary.py \
    --serial <frozen R3 report> --parallel <acceptance report>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True, type=Path)
    ap.add_argument("--parallel", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s = json.loads(args.serial.read_text())
    p = json.loads(args.parallel.read_text())
    s_wall = {r["demo"]: r["segment_wall_s"] for r in s["rows"]}
    p_wall = {r["demo"]: r["segment_wall_s"] for r in p["rows"]}
    assert set(s_wall) == set(p_wall), "demo sets differ"

    serial_sum = sum(s_wall.values())
    max_seg = max(s_wall.values())
    ideal_wall = max_seg
    ideal_speedup = serial_sum / ideal_wall
    # whole-run parallel wall from the execution block (submit->collect)
    par_wall = (p.get("execution") or {}).get("collect_s") or sum(p_wall.values())
    speedup = serial_sum / par_wall
    frac = speedup / ideal_speedup

    ratios = sorted(p_wall[d] / s_wall[d] for d in s_wall)
    median_ratio = ratios[len(ratios) // 2]

    out = {
        "serial_report": str(args.serial),
        "parallel_report": str(args.parallel),
        "per_segment_wall_s": {"serial": s_wall, "parallel": p_wall},
        "serial_sum_s": round(serial_sum, 1),
        "max_segment_s (ideal wall)": round(ideal_wall, 1),
        "parallel_wall_s": round(par_wall, 1),
        "speedup": round(speedup, 3),
        "ideal_speedup": round(ideal_speedup, 3),
        "fraction_of_ideal": round(frac, 3),
        "gate_throughput_ge_0.75": bool(frac >= 0.75),
        "per_segment_parallel_over_serial": {
            d: round(p_wall[d] / s_wall[d], 3) for d in sorted(s_wall)},
        "median_segment_ratio": round(median_ratio, 3),
        "gate_median_regression_le_1.05": bool(median_ratio <= 1.05),
        "kept": {
            "serial": [r["demo"] for r in s["rows"] if r["accepted"]],
            "parallel": [r["demo"] for r in p["rows"] if r["accepted"]],
        },
    }
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text)
    sys.exit(0 if (out["gate_throughput_ge_0.75"]
                  and out["gate_median_regression_le_1.05"]) else 1)


if __name__ == "__main__":
    main()
