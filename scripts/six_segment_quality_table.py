"""Six-segment quality comparison table across runs (plan sections 6/9).

Aggregates ``segment_quality_report.quality_for`` over every NPZ of each given
run directory: validator verdict, per-side reference/certified coverage,
box-frame speed, anchor drift, contact gaps and task MPJPE. Used for the
"baseline / ablation / final acceptance" table required by the plan; the
frozen v3 products are just another run entry, so the comparison values are
always recomputed from the stored artifacts, never copied from old reports.

Run: uv run --locked python scripts/six_segment_quality_table.py \
        --run v3=derived/data/teachers_regenerated/pilot_diagnosis6_v3_f1 \
        --run f4b=derived/data/teachers_regenerated/fix_20260922_f4b_stab01 \
        --out derived/data/teachers_regenerated/<run>/quality_table.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from segment_quality_report import quality_for


def summarize(q: dict) -> dict:
    sides = {}
    for s, m in q["sides"].items():
        sides[s] = {
            "ref_frames": m["ref_frames"],
            "certified_frames": m["certified_frames"],
            "certified_in_ref_frames": m["certified_in_ref_frames"],
            "n_intervals": len(m["intervals"]),
            "interval_bounds": m["intervals"],
            "contact_recall": q["hand"].get(f"{'left' if s == 'l' else 'right'}"
                                            "_hand_contact_recall"),
            "interval_recall": q["hand"].get(f"{'left' if s == 'l' else 'right'}"
                                             "_hand_interval_recall"),
            "speed_ref_p95_mps": m["speed_ref_mps"].get("p95"),
            "speed_all_p95_mps": m["speed_all_mps"].get("p95"),
            "drift_certified_max_m": m["anchor_drift_certified_m"].get("max"),
            "gap_ref_mm": m["gap_ref_mm"],
            "enter_mm": m.get("enter_mm"),
            "frames_within_enter": m.get("frames_within_enter"),
            "frames_within_enter_p2mm": m.get("frames_within_enter_p2mm"),
            "frames_within_enter_p5mm": m.get("frames_within_enter_p5mm"),
            "enter_threshold_crossings": m.get("enter_threshold_crossings"),
            "missing_contact_ref_frames": m["missing_contact_ref_frames"],
        }
    return {
        "demo": q["demo"],
        "T": q["T"],
        "validator_passed": q["validator_passed"],
        "validator_failed": q["validator_failed"],
        "hand": q["hand"],
        "task_mpjpe": q["task_mpjpe"],
        "growth": q["mpjpe_growth"]["growth"],
        "q1_mean": q["mpjpe_growth"]["q1_mean"],
        "q5_mean": q["mpjpe_growth"]["q5_mean"],
        "sides": sides,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="LABEL=DIR", help="repeatable")
    ap.add_argument("--validation-mode", default="strict")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    table: dict[str, dict] = {}
    for spec in args.run:
        label, _, dirname = spec.partition("=")
        npzs = sorted(Path(dirname).glob("*.npz"))
        rows = []
        for npz in npzs:
            q = quality_for(npz, npz.stem, args.validation_mode)
            rows.append(summarize(q))
            print(f"[{label}] {npz.stem} done", flush=True)
        table[label] = {"dir": dirname, "n": len(rows), "rows": rows}

    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=1, default=float))
        print(f"-> {args.out}")

    # compact side-by-side print
    demos = sorted({r["demo"] for v in table.values() for r in v["rows"]})
    for label, v in table.items():
        print(f"\n== {label} ({v['n']} npz)")
        for demo in demos:
            row = next((r for r in v["rows"] if r["demo"] == demo), None)
            if row is None:
                continue
            sides = " ".join(
                f"{s}: in-ref {m['certified_in_ref_frames']}/{m['ref_frames']} "
                f"cert {m['certified_frames']} cov {m['contact_recall']} "
                f"crossings {m['enter_threshold_crossings']}"
                for s, m in row["sides"].items())
            print(f"  {demo} valid={row['validator_passed']} "
                  f"task p50/p95 {row['task_mpjpe'].get('p50')}/"
                  f"{row['task_mpjpe'].get('p95')} growth {row['growth']} | {sides}")


if __name__ == "__main__":
    main()
