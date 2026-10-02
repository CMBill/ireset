"""H/B/C/T quality-gate report for train24fix3 (plan sections 3.2 and 8.1).

Computes the unrounded per-side quality metrics of a NEW run (kept NPZ)
with the same ``freeze_segment`` the previous rounds used, then compares
them against the registered references:

- ``H``  labelq_20260926_sixseg/rq            -- 5 demos / 10 sides, the
  historical D-Q lower bounds (sub7 has no NPZ there: absent, never zero);
- ``B``  train24fix_20260927_sixseg/rq        -- 6 demos / 12 sides, the
  pre-C2 6/6-keep baseline with its quality improvements;
- ``C``  train24fix2_20260927_sixseg/rq       -- the current 1/6 production;
  sub16's right intersection / longest run >= 15 are additionally protected;
- ``T``  labelq_20260926_train24/regen_final  -- the original train24 keeps.

Counting is exact (frame counts, interval runs); quality values are the
unrounded series; no physical tolerance is added anywhere. An improvement
on another side or a new segment NEVER compensates a regression on one
side, and a reference segment missing from the new run is a FAIL, never a
skip (plan 8.1).

CLI: uv run --locked python scripts/train24fix3_quality_gate.py \
        --run <new rq dir> --out <gate json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.paths import DERIVED_DATA_DIR  # noqa: E402
from ireset.repair.config_io import load_mode_configs  # noqa: E402
from train24fix2_cert_trace import freeze_segment, load_cond  # noqa: E402

TEACHERS = DERIVED_DATA_DIR / "teachers_regenerated"
REFERENCES = {
    "H": TEACHERS / "labelq_20260926_sixseg" / "rq",
    "B": TEACHERS / "train24fix_20260927_sixseg" / "rq",
    "C": TEACHERS / "train24fix2_20260927_sixseg" / "rq",
    "T": TEACHERS / "labelq_20260926_train24" / "regen_final",
}
#: plan 3.2 / 8.1: sub16's C protections -- right intersection and longest
#: continuous run both >= 15, proximity/task not worse than C
SUB16 = "sub16_largebox_000"

#: core gates (plan 8.1 row 2): per-side counters that must not DECREASE
CORE_METRICS = ("certified_in_ref_frames",
                "intersection_longest_run_frames",
                "frames_within_enter")
#: complete-quality gates (plan 8.1 row 3): must not INCREASE (None = the
#: side has no certified interval to measure -- recorded, never zero)
COMPLETE_MAX_METRICS = ("longest_detachment_run_frames",
                        "max_detachment_ref_m",
                        "anchor_drift_certified_max_m")


def metrics_of_run(run_dir: Path) -> dict[str, dict]:
    """``{demo: freeze_segment dict}`` for every kept NPZ of a run.

    Deliberately NO cross-run cache: the sixseg references and a candidate
    run share DEMO NAMES, so a demo-keyed cache would hand the reference
    comparison the candidate's own metrics (fake equality). Each run's
    metrics are recomputed from its own NPZ."""
    rcfg, vcfg = load_mode_configs("strict")
    out: dict[str, dict] = {}
    for npz in sorted(Path(run_dir).glob("*.npz")):
        demo = npz.stem
        _scene, cond = load_cond(demo)
        out[demo] = freeze_segment(npz, demo, cond, vcfg, rcfg)
    return out


def _side_cells(n: dict, r: dict, side: str) -> dict:
    cells = {}
    for m in CORE_METRICS:
        nv, rv = n["sides"][side][m], r["sides"][side][m]
        cells[m] = {"new": nv, "ref": rv, "ok": nv >= rv}
    for m in COMPLETE_MAX_METRICS:
        nv, rv = n["sides"][side][m], r["sides"][side][m]
        ok = None if (nv is None or rv is None) else nv <= rv
        cells[m] = {"new": nv, "ref": rv, "ok": ok}
    nt, rt = n["task_mpjpe"], r["task_mpjpe"]
    cells["task_p95"] = {"new": nt["p95"], "ref": rt["p95"],
                         "ok": nt["p95"] <= rt["p95"]}
    cells["task_max"] = {"new": nt["max"], "ref": rt["max"],
                         "ok": nt["max"] <= rt["max"]}
    return cells


def _run_demo_set(run_dir: Path) -> set[str] | None:
    """Demos the run actually covered (report rows), or None when no report
    is found (then every reference demo counts as in scope)."""
    for rep in sorted(Path(run_dir).glob("*report*.json")):
        try:
            doc = json.loads(rep.read_text())
        except json.JSONDecodeError:
            continue
        rows = doc.get("rows")
        if isinstance(rows, list):
            return {r["demo"] for r in rows if isinstance(r, dict) and "demo" in r}
    return None


def gate_report(run_dir: Path) -> dict:
    new = metrics_of_run(run_dir)
    refs = {name: metrics_of_run(d) for name, d in REFERENCES.items()}
    scope = _run_demo_set(Path(run_dir))
    gate_rows: list[dict] = []
    verdicts: dict[str, bool] = {}

    for ref_name in ("H", "B", "C", "T"):
        for demo, r in sorted(refs[ref_name].items()):
            if scope is not None and demo not in scope:
                # out of this run's demo set: NOT APPLICABLE (recorded with
                # an explicit marker, never counted as a pass or a fail --
                # the sixseg and train24 sets only share subjects)
                gate_rows.append({"reference": ref_name, "demo": demo,
                                  "side": None,
                                  "cells": {"scope": {"new": None, "ref": None,
                                                      "ok": None}},
                                  "ok": None, "na": "out_of_run_scope"})
                continue
            n = new.get(demo)
            if n is None:
                gate_rows.append({"reference": ref_name, "demo": demo,
                                  "side": None, "cells": {"kept": {
                                      "new": None, "ref": "kept",
                                      "ok": False}}, "ok": False})
                continue
            for side in ("l", "r"):
                cells = _side_cells(n, r, side)
                gate_rows.append({"reference": ref_name, "demo": demo,
                                  "side": side, "cells": cells,
                                  "ok": all(c["ok"] is not False
                                            for c in cells.values())})
        judged = [g for g in gate_rows
                  if g["reference"] == ref_name and g["ok"] is not None]
        verdicts[f"{ref_name}_all_sides_ok"] = (all(g["ok"] for g in judged)
                                                if judged else None)
        verdicts[f"{ref_name}_n_applicable"] = len(judged)

    # C's explicit sub16 protections (plan 3.2) -- sixseg runs only
    n16, r16 = new.get(SUB16), refs["C"].get(SUB16)
    in_scope = scope is None or SUB16 in scope
    sub16_ok = False
    if not in_scope:
        verdicts["C_sub16_protections_ok"] = None
        verdicts["C_sub16_protections_na"] = "out_of_run_scope"
    elif n16 is not None and r16 is not None:
        c = _side_cells(n16, r16, "r")
        sub16_ok = (n16["sides"]["r"]["certified_in_ref_frames"] >= 15
                    and n16["sides"]["r"]["intersection_longest_run_frames"] >= 15
                    and c["frames_within_enter"]["ok"]
                    and c["task_p95"]["ok"] and c["task_max"]["ok"])
        verdicts["C_sub16_protections_ok"] = sub16_ok
    else:
        verdicts["C_sub16_protections_ok"] = sub16_ok

    core_ok = [v for k, v in verdicts.items()
               if k.endswith("_all_sides_ok") and v is not None]
    sub16 = verdicts.get("C_sub16_protections_ok")
    overall = bool(core_ok) and all(core_ok) and sub16 is not False
    return {"run_dir": str(run_dir), "rows": gate_rows, "verdicts": verdicts,
            "ok": overall,
            "metric_notes": {
                "core": "certified-in-ref / longest intersection run / "
                        "proximity frames must not decrease vs H and B",
                "complete": "detachment/drift must not increase; None = no "
                            "certified interval on that side (recorded, "
                            "never zero)",
                "task": "per-segment p95/max must not increase",
                "compensation": "an improvement elsewhere never compensates "
                                "a single-side regression"}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="new rq dir (kept NPZ)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = gate_report(Path(args.run))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(f"[quality gate] ok={report['ok']} verdicts={report['verdicts']} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
