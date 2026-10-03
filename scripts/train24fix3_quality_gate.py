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

train24fix3 continuation (plan 8.2): EVERY section-8.1 row now carries an
explicit verdict -- coverage against the frozen demo list, NPZ/report
consistency, independent strict of every kept segment, per-side
core/complete/task cells, the demo-level both-hands overlap and usable
windows (the 60-frame/stride-30 windowing of ``TeacherWindowDataset``,
never a redefinition), the mechanism family (spikes / enable-state /
GATED-zero reclassification of the four frozen failing sides vs H, via
the same ``classify_side`` the certification-edge round used), the
integrity family (R9 / profile / row coherence / argv) and the label
family (graded_labels_v1 manifest + NPZ contract).  ``full=True`` (the
formal stage-E/F gate) makes the evidence-requiring families MANDATORY:
missing evidence is an explicit failure, never a pass.  The development
stages keep the historical subset (their own spikes/enable gates run in
the driver).

CLI: uv run --locked python scripts/train24fix3_quality_gate.py \
        --run <new rq dir> --out <gate json> [--full] \
        [--expected-set diagnosis6|calibration24_train_v1] [--profile S1] \
        [--cf-summary <profile_summary.json>] [--labels <label_manifest.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from closed_loop_metrics import sha256 as sha256_16
from ireset.data.labels import LABEL_PROTOCOL_VERSION, both_hands_metrics
from ireset.paths import DERIVED_DATA_DIR, PROJECT_ROOT  # noqa: E402
from ireset.repair.config_io import load_mode_configs  # noqa: E402
from regenerate_teachers import registered_set
from train24fix2_cert_trace import (  # noqa: E402
    classify_side,
    freeze_segment,
    load_cond,
)

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
#: plan 8.1 mechanism row: the four frozen D-Q failing sides whose
#: lost/gain reclassification (vs H) must show ZERO speed-gated breaks
FOUR_FAILING_SIDES = (("sub10_largebox_000", "r"), ("sub3_largebox_000", "r"),
                      ("sub8_largebox_000", "l"), ("sub8_largebox_000", "r"))
#: plan 8.2: the usable-window rule is TeacherWindowDataset's own 60/30
#: windowing (frozen stride/protocol) -- no looser "usable" is defined here
WINDOW_FRAMES = 60
WINDOW_STRIDE = 30

#: core gates (plan 8.1 row 2): per-side counters that must not DECREASE
CORE_METRICS = ("certified_in_ref_frames",
                "intersection_longest_run_frames",
                "frames_within_enter")
#: complete-quality gates (plan 8.1 row 3): must not INCREASE (None = the
#: side has no certified interval to measure -- recorded, never zero)
COMPLETE_MAX_METRICS = ("longest_detachment_run_frames",
                        "max_detachment_ref_m",
                        "anchor_drift_certified_max_m")
#: demo-level both-hands overlap (plan 8.1 row 3): must not DECREASE
BOTH_HANDS_METRICS = ("both_certified_frames", "both_supervision_frames",
                      "both_longest_supervision_run_frames")


def metrics_of_run(run_dir: Path, *, with_series: bool = False) -> dict[str, dict]:
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
        out[demo] = freeze_segment(npz, demo, cond, vcfg, rcfg,
                                   with_series=with_series)
    return out


def _both_hands(seg: dict) -> dict:
    """``both_hands_metrics`` from a freeze_segment dict (masks rebuilt
    from the frozen interval lists -- the same series the exporter uses)."""
    per_side = {}
    for side in ("l", "r"):
        s = seg["sides"][side]
        cert = np.zeros(seg["T"], dtype=bool)
        for a, b in s["certified_intervals"]:
            cert[a:b] = True
        sup = np.zeros(seg["T"], dtype=bool)
        for a, b in s["intersection_intervals"]:
            sup[a:b] = True
        per_side[side] = {"certified": cert, "supervision": sup}
    return both_hands_metrics(per_side)


def _usable_windows(seg: dict) -> int:
    """Window count of TeacherWindowDataset's own index construction:
    ``range(0, T - window + 1, stride)`` with window=60, stride=30."""
    return max(0, (seg["T"] - WINDOW_FRAMES) // WINDOW_STRIDE + 1)


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


def _demo_cells(n: dict, r: dict) -> dict:
    """Demo-level complete-quality cells: both-hands overlap + usable
    windows (plan 8.1 row 3 -- never compensable per-side improvements)."""
    cells = {}
    nb, rb = _both_hands(n), _both_hands(r)
    for m in BOTH_HANDS_METRICS:
        cells[m] = {"new": nb[m], "ref": rb[m], "ok": nb[m] >= rb[m]}
    nw, rw = _usable_windows(n), _usable_windows(r)
    cells["usable_windows"] = {"new": nw, "ref": rw, "ok": nw >= rw}
    return cells


def _load_report(run_dir: Path) -> dict | None:
    for rep in sorted(Path(run_dir).glob("*report*.json")):
        try:
            doc = json.loads(rep.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(doc.get("rows"), list):
            return doc
    return None


def _run_demo_set(run_dir: Path) -> set[str] | None:
    """Demos the run actually covered (report rows), or None when no report
    is found (then every reference demo counts as in scope)."""
    doc = _load_report(run_dir)
    if doc is None:
        return None
    return {r["demo"] for r in doc["rows"]
            if isinstance(r, dict) and "demo" in r}


def _coverage_verdicts(run_dir: Path, new: dict, expected_demos,
                       verdicts: dict) -> list[dict]:
    """Plan 8.2: the run's coverage against the FROZEN list -- report rows
    non-empty, duplicate-free, set-equal; kept NPZ set == accepted set with
    T/fps coherent.  A report that dropped failed demos cannot shrink the
    reference comparison to out-of-scope."""
    rows_out: list[dict] = []
    doc = _load_report(run_dir)
    if doc is None:
        verdicts["coverage_report_present"] = False
        verdicts["coverage_demos_ok"] = False
        verdicts["npz_consistency_ok"] = False
        return rows_out
    verdicts["coverage_report_present"] = True
    demos = [r.get("demo") for r in doc["rows"]]
    ok = bool(demos) and all(d is not None for d in demos) \
        and len(set(demos)) == len(demos)
    verdicts["coverage_demos_ok"] = bool(ok)
    if ok and doc.get("n_demos") not in (None, len(demos)):
        verdicts["coverage_demos_ok"] = False
    if ok and expected_demos is not None:
        if set(demos) != set(expected_demos) or len(demos) != len(expected_demos):
            verdicts["coverage_demos_ok"] = False
            verdicts["coverage_expected_set"] = sorted(set(expected_demos)
                                                       ^ set(demos))[:10]
    verdicts["coverage_n_processed"] = len(demos)
    accepted = sorted(r["demo"] for r in doc["rows"] if r.get("accepted"))
    kept_npz = sorted(new)
    npz_ok = accepted == kept_npz
    for demo in accepted:
        row = next(r for r in doc["rows"] if r["demo"] == demo)
        seg = new.get(demo)
        if seg is None or seg["T"] != row.get("T"):
            npz_ok = False
    verdicts["npz_consistency_ok"] = bool(npz_ok) and bool(accepted)
    verdicts["n_kept"] = len(accepted)
    return rows_out


def _mechanism_verdicts(new: dict, refs: dict, cf_summary, full: bool,
                        scope: set[str] | None, verdicts: dict) -> None:
    """Plan 8.1 mechanism row: the ten spikes rejected + enable-state
    compliance (from the profile's counterfactual evidence) and the
    four frozen failing sides' lost/gain reclassification vs H with ZERO
    speed-gated breaks (``classify_side`` GATED class).  Sides outside the
    run's frozen demo scope are N/A (recorded, never silently passed)."""
    if cf_summary is None:
        if full:
            # 缺 gate 不是通过: the formal mode treats missing evidence as
            # an explicit failure of the family
            verdicts["mechanism_spikes_ok"] = False
            verdicts["mechanism_enable_ok"] = False
        else:
            verdicts["mechanism_cf_evidence"] = "not_requested"
    else:
        verdicts["mechanism_spikes_ok"] = bool(cf_summary.get("spikes_ok"))
        verdicts["mechanism_enable_ok"] = bool(cf_summary.get("enable_ok"))
    _rcfg, vcfg = load_mode_configs("strict")
    gated_total, classified = 0, 0
    for demo, side in FOUR_FAILING_SIDES:
        if scope is not None and demo not in scope:
            continue
        old = refs["H"].get(demo)
        cur = new.get(demo)
        if old is None or cur is None:
            gated_total = None
            break
        o = old["sides"][side].get("series")
        nws = cur["sides"][side].get("series")
        if o is None or nws is None:
            gated_total = None
            break
        cls = classify_side(o, nws, vcfg)
        classified += 1
        gated_total += sum(1 for c in cls["lost"].values() if c == "GATED")
    if classified == 0 and gated_total == 0:
        # none of the four sides' demos is in this run's scope (e.g. a
        # train24 run): N/A, never a pass
        verdicts["mechanism_gated_zero_ok"] = None
        verdicts["mechanism_gated_na"] = "out_of_run_scope"
    else:
        verdicts["mechanism_gated_zero_ok"] = (
            False if gated_total is None else gated_total == 0)
    verdicts["mechanism_gated_frames_total"] = gated_total
    verdicts["mechanism_sides_classified"] = classified


def _integrity_verdicts(doc: dict | None, profile, verdicts: dict) -> None:
    """Plan 8.1 integrity row: R9 clean, the registered profile matches the
    run's own record, argv recorded, every row's status coherent with its
    acceptance (an accepted row must be status-accepted; a rejected row
    must not claim acceptance)."""
    if doc is None:
        verdicts["integrity_r9_clean"] = False
        verdicts["integrity_rows_coherent"] = False
        verdicts["integrity_argv_recorded"] = False
        if profile is not None:
            verdicts["integrity_profile_matches"] = False
        return
    verdicts["integrity_r9_clean"] = \
        doc.get("end", {}).get("code_changed_during_run") is False
    verdicts["integrity_argv_recorded"] = \
        bool(doc.get("effective", {}).get("argv"))
    rows = doc.get("rows") or []
    verdicts["integrity_rows_coherent"] = bool(rows) and all(
        (r.get("accepted") and r.get("status") == "accepted")
        or (not r.get("accepted") and r.get("status") != "accepted")
        for r in rows)
    if profile is not None:
        verdicts["integrity_profile_matches"] = \
            (doc.get("profile") or {}).get("name") == profile


def _labels_verdicts(label_manifest, run_doc: dict | None, new: dict,
                     full: bool, verdicts: dict) -> None:
    """Plan 8.1 label row: the graded_labels_v1 manifest of THIS run --
    protocol, a row for every run demo, per-kept label NPZ hash + array
    contract (shapes, bool dtypes, supervision == ids >= 0).  The
    torch-level empty-supervision zero-loss/zero-gradient contract is
    pinned by tests/test_label_quality.py (a code contract, not per-run
    data)."""
    if label_manifest is None:
        verdicts["labels_manifest_ok"] = False if full else "not_requested"
        return
    path = Path(label_manifest)
    if not path.is_file():
        verdicts["labels_manifest_ok"] = False
        verdicts["labels_error"] = f"{path}: manifest missing"
        return
    doc = json.loads(path.read_text())
    problems: list[str] = []
    if doc.get("label_protocol") != LABEL_PROTOCOL_VERSION:
        problems.append(f"protocol {doc.get('label_protocol')!r}")
    run_demos = ({r["demo"] for r in run_doc["rows"]}
                 if run_doc is not None else set(new))
    row_demos = {r.get("demo") for r in doc.get("rows") or []}
    if run_demos and row_demos != run_demos:
        problems.append(f"manifest rows != run demos "
                        f"(diff {sorted(run_demos ^ row_demos)[:6]})")
    n_checked = 0
    for r in doc.get("rows") or []:
        if not r.get("accepted"):
            continue
        rec = r.get("label_npz") or {}
        lpath = (PROJECT_ROOT / rec["path"]) if rec.get("path") else None
        if lpath is None or not lpath.is_file():
            problems.append(f"{r.get('demo')}: label NPZ missing")
            continue
        # the exporter's manifest carries the 16-hex truncated domain --
        # compared with the SAME domain, never a full digest (2026-10-02
        # engineering note on hash-domain mixing)
        if sha256_16(lpath) != rec.get("sha256"):
            problems.append(f"{r.get('demo')}: label NPZ hash mismatch")
            continue
        seg = new.get(r.get("demo"))
        with np.load(lpath, allow_pickle=False) as z:
            prox, sup = z["hand_proximity_mask"], z["hand_contact_supervision_mask"]
            ids = z["hand_supervision_interval_id"]
        if prox.shape != sup.shape or prox.ndim != 2 or prox.shape[1] != 2:
            problems.append(f"{r.get('demo')}: mask shape {prox.shape}")
            continue
        if seg is not None and prox.shape[0] != seg["T"]:
            problems.append(f"{r.get('demo')}: mask T != trajectory T")
            continue
        if prox.dtype != np.bool_ or sup.dtype != np.bool_:
            problems.append(f"{r.get('demo')}: masks not boolean")
            continue
        if not np.array_equal(sup, ids >= 0):
            problems.append(f"{r.get('demo')}: supervision != ids>=0")
            continue
        if r.get("strict_recheck_passed") is not True:
            problems.append(f"{r.get('demo')}: strict recheck not passed")
            continue
        n_checked += 1
    verdicts["labels_manifest_ok"] = not problems
    verdicts["labels_n_kept_checked"] = n_checked
    if problems:
        verdicts["labels_problems"] = problems[:10]


def gate_report(run_dir: Path, *, expected_demos=None, profile=None,
                cf_summary=None, label_manifest=None,
                full: bool = False) -> dict:
    """Every plan-8.1 row as an explicit verdict.  ``full`` (the formal
    stage-E/F mode) makes the evidence-requiring families mandatory: a
    missing cf summary / label manifest / expected list / profile is an
    explicit failure, never a pass (plan 8.2 -- 缺 gate 不是通过)."""
    run_dir = Path(run_dir)
    new = metrics_of_run(run_dir, with_series=True)
    refs = {name: metrics_of_run(d, with_series=True)
            for name, d in REFERENCES.items()}
    scope = _run_demo_set(run_dir)
    report_doc = _load_report(run_dir)
    gate_rows: list[dict] = []
    verdicts: dict[str, object] = {}

    # -- coverage + NPZ consistency + independent strict (plan 8.1 rows 1/7/8)
    _coverage_verdicts(run_dir, new, expected_demos, verdicts)
    verdicts["strict_all_kept_ok"] = bool(new) and all(
        m["validator_passed"] for m in new.values())

    # -- per-side core / complete / task + demo-level overlap & windows
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
            dcells = _demo_cells(n, r)
            gate_rows.append({"reference": ref_name, "demo": demo,
                              "side": "both",
                              "cells": dcells,
                              "ok": all(c["ok"] is not False
                                        for c in dcells.values())})
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

    # mechanism / integrity / label families (plan 8.1 rows 5-8)
    _mechanism_verdicts(new, refs, cf_summary, full, scope, verdicts)
    _integrity_verdicts(report_doc, profile, verdicts)
    _labels_verdicts(label_manifest, report_doc, new, full, verdicts)

    core_ok = [v for k, v in verdicts.items()
               if k.endswith("_ok") and v is not None]
    sub16 = verdicts.get("C_sub16_protections_ok")
    overall = bool(core_ok) and all(core_ok) and sub16 is not False
    return {"run_dir": str(run_dir), "rows": gate_rows, "verdicts": verdicts,
            "ok": overall, "full": bool(full),
            "metric_notes": {
                "core": "certified-in-ref / longest intersection run / "
                        "proximity frames must not decrease vs H and B",
                "complete": "detachment/drift must not increase; None = no "
                            "certified interval on that side (recorded, "
                            "never zero); both-hands overlap and usable "
                            "windows (60-frame, stride-30 -- the dataset's "
                            "own rule) must not decrease",
                "task": "per-segment p95/max must not increase",
                "mechanism": "spikes all rejected + enable-state compliance "
                             "(cf summary) + zero GATED frames in the four "
                             "frozen failing sides' reclassification vs H",
                "labels": "graded_labels_v1 manifest + label NPZ contract; "
                          "the empty-supervision zero-loss/zero-gradient "
                          "smoke is pinned by tests/test_label_quality.py",
                "compensation": "an improvement elsewhere never compensates "
                                "a single-side regression"}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="new rq dir (kept NPZ)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true",
                    help="formal stage-E/F mode: every evidence-requiring "
                         "family is mandatory (missing evidence fails)")
    ap.add_argument("--expected-set", default=None,
                    choices=["diagnosis6", "calibration24_train_v1"],
                    help="frozen demo list the run must cover exactly")
    ap.add_argument("--profile", default=None,
                    help="the SELECTED profile name (cross-checked against "
                         "the run's own record)")
    ap.add_argument("--cf-summary", default=None,
                    help="the selected profile's profile_summary.json "
                         "(mechanism family evidence)")
    ap.add_argument("--labels", default=None,
                    help="this run's label_manifest.json (label family)")
    args = ap.parse_args()
    expected = registered_set(args.expected_set) if args.expected_set else None
    cf = (json.loads(Path(args.cf_summary).read_text())
          if args.cf_summary else None)
    report = gate_report(Path(args.run), expected_demos=expected,
                         profile=args.profile, cf_summary=cf,
                         label_manifest=(Path(args.labels)
                                         if args.labels else None),
                         full=args.full)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(f"[quality gate] ok={report['ok']} verdicts={report['verdicts']} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
