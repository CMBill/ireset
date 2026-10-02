"""Bind the failing candidates of the train24fix2 six-segment regression and
classify them with FULL-constraint counterfactuals (train24fix2 plan section
9 risk row 2; REVIEW-FIX round rewrite, plan train24fix2_review_fix_plan.md
sections 2-4).

Review defects this rewrite removes (2026-10-01):

- R1 (P1): the context used to be REBUILT with the evaluator's current
  robot FK as ``ref_kpts_world``, dropping the human reference, the hand
  targets/deltas and the frame motion -- sub12@12/sub3@14 task errors
  collapsed to 0.  Now the ``fail`` snapshot FREEZES the run's own
  FrameContext field-completely (``train24fix2_audit.freeze_context``) and
  every counterfactual restores THAT context, toggling only the C2 fields.
- R3: the C2-off counterfactual checked endpoint rows only.  Now each
  branch gets the full audit (endpoint rows / every interpolation point /
  strict velocities / task under the human reference / PREFIX independent
  final validation) and the section-4.2 classification vocabulary.
- The single-frame re-solve now replays the production escalation (V1
  skip check with reflux -> prediction -> prev_accepted -> slack) instead
  of three bare ``_solve_attempt`` calls.

Outputs (versioned JSON + NPZ, overwrite refused) go to
``derived/data/teachers_regenerated/train24fix2_reviewfix2_20261002/diag/``
by default -- NEVER the frozen ``train24fix2_20260927_diag/`` nor the
superseded ``train24fix2_reviewfix_20261001/diag/``.

Reviewfix2 (2026-10-02) hardening of THIS tool:

- D1: the identity check compares the re-run against the ORIGINAL formal
  report of ``--rq-dir`` (t_fail / last_attempt_id / last_check_id /
  last-attempt status, item by item); any mismatch STOPS the binding.  The
  previous check compared the re-run against its OWN snapshot -- circular,
  a doctored report changed nothing.
- D2/D4/D5: the replay is index-driven, restores the FROZEN effective
  configs from each product, verifies the NPZ sha256 + code/config/asset
  integrity manifest, recomputes the re-solve escalation AND its adopted
  candidate's audit, and compares the COMPLETE records (all rows,
  interpolation points, strict velocities, prefix worst values,
  classification) -- its summary goes to a SEPARATE replay directory.

Run: uv run --locked python scripts/train24fix2_fail_bind.py [--rq-dir DIR]
                                                      [--out DIR] [--replay]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from closed_loop_metrics import git_rev
from experiment_provenance import asset_hashes, git_state
from ireset.data.teacher import load_teacher_npz
from ireset.embodied.contact_sites import infer_reference_schedule
from ireset.paths import DERIVED_DATA_DIR, TEACHER_TRAJ_DIR
from ireset.repair.config_io import load_mode_configs
from ireset.repair.context import ReferenceCondition
from ireset.repair.pipeline import RepairPipeline
from train24fix2_audit import (
    AUDIT_FORMAT,
    arrays_diff,
    assert_snapshot_matches_live,
    attach_integrity,
    classify_branches,
    configs_from_record,
    context_arrays,
    context_from,
    context_meta,
    context_scalar_leaves,
    deep_diff,
    escalate_frame,
    freeze_context,
    full_candidate_audit,
    load_index,
    load_products,
    verify_integrity,
    write_index,
    write_products,
)

REGEN = DERIVED_DATA_DIR / "teachers_regenerated"
DEFAULT_RQ = REGEN / "train24fix2_20260927_sixseg" / "rq"
DEFAULT_OUT = REGEN / "train24fix2_reviewfix2_20261002" / "diag"
INDEX_NAME = "fail_bind2_index"


def c2_off_ctx(ctx):
    """Same frozen context with ONLY the C2 row-enable state cleared (the
    bound parameter stays 0.02598 in the config -- the rows turn off
    through the context, never by replacing the whole config)."""
    return replace(freeze_context(ctx), c2_hand_enabled={}, c2_hand_reason={})


def load_report_rows(rq_dir: Path) -> dict[str, dict]:
    """demo -> row of the ORIGINAL formal report (the frozen rq run's own
    ``regen_report_diagnosis6_strict.json``)."""
    report = json.loads(
        (rq_dir / "regen_report_diagnosis6_strict.json").read_text())
    return {r["demo"]: r for r in report["rows"]}


def identity_vs_report(report_row: dict, captured: dict) -> dict:
    """D1: item-by-item comparison of THIS re-run's failure against the
    ORIGINAL formal report row -- the binding target.  Any False stops the
    binding (never bind 'a similar failure').  The four items mirror what
    the report's ``rejection`` block freezes: failure frame, last attempt
    id, last check id, last-attempt solver status."""
    rej = report_row.get("rejection") or {}
    last_att = rej.get("last_attempt") or {}
    got_attempt = captured["last_attempt_id"]
    got_check = captured["chk"]["check_id"]
    exp = {
        "t_fail": report_row.get("t_fail"),
        "last_attempt_id": rej.get("last_attempt_id"),
        "last_check_id": rej.get("last_check_id"),
        "last_attempt_status": last_att.get("status"),
    }
    got = {
        "t_fail": captured["t"],
        "last_attempt_id": list(got_attempt),
        "last_check_id": got_check,
        "last_attempt_status": captured["res_fields"]["status"],
    }
    return {"match": {k: exp[k] == got[k] for k in exp},
            "report": exp, "this_run": got}


def bind(demo: str, rq_dir: Path, out: Path,
         *, report_row: dict | None = None,
         scene: np.ndarray | None = None,
         cond: ReferenceCondition | None = None) -> dict:
    if scene is None or cond is None:
        _scene, human, _fps = load_teacher_npz(
            TEACHER_TRAJ_DIR / f"{demo}_original.npz")
        scene = _scene
        cond = ReferenceCondition(human, scene[:, 36:43],
                                  infer_reference_schedule(human,
                                                           scene[:, 36:43]))
    rcfg, vcfg = load_mode_configs("strict")
    cfg = replace(rcfg, regen_mode=True)
    assert cfg.regen_hand_speed_mps == 0.02598          # noqa: S101
    pipe = RepairPipeline(cfg=cfg)
    pipe.validator.cfg = vcfg
    captured: dict = {}

    def snap(kind, data):
        if kind != "fail":
            return
        # R1: freeze the run's own context FIELD-COMPLETELY (deep copy,
        # verified against the live object on every field), plus the last
        # executed attempt's candidate/check/record and the accepted prefix
        ctx_frozen = freeze_context(data["ctx"])
        assert_snapshot_matches_live(ctx_frozen, data["ctx"])
        captured["t"] = data["t"]
        captured["pass_index"] = data["pass_index"]
        captured["ctx"] = ctx_frozen
        captured["z_cand"] = np.asarray(data["last_res"].z).copy()
        captured["res_fields"] = {
            "status": data["last_res"].status,
            "term_reason": data["last_res"].term_reason,
            "iter_budget": data["last_res"].iter_budget,
            "iterations": data["last_res"].iterations,
            "qp_calls": data["last_res"].qp_calls,
            "slack_used": bool(data["last_res"].slack_used),
            "trust": float(data["last_res"].trust),
            "last_qp_status": data["last_res"].last_qp_status,
            "last_viol": float(data["last_res"].last_viol)}
        chk = data["last_chk"]
        captured["chk"] = {"ok": chk.ok, "reason": chk.reason,
                           "viol": float(chk.viol), "rows": chk.rows,
                           "c2": {s: dict(v) for s, v in chk.c2.items()},
                           "check_id": chk.check_id}
        captured["last_rec"] = dict(data["last_rec"])
        captured["last_attempt_id"] = list(data["last_attempt_id"])
        captured["accepted"] = [a.copy() for a in data["accepted"]]
        captured["prediction"] = data["prediction"].copy()
        captured["c2_diag"] = {
            k: (dict(v) if isinstance(v, dict) else list(v))
            for k, v in pipe._c2_diag.items()}
    pipe.snapshot = snap
    t0 = time.perf_counter()
    res = pipe.repair_trajectory(scene, cond)
    wall = time.perf_counter() - t0
    if res.accepted or not captured:
        return {"demo": demo, "accepted": bool(res.accepted),
                "note": "no fail event captured"}

    t = captured["t"]
    ctx = captured["ctx"]
    z_cand = captured["z_cand"]
    prefix = captured["accepted"]
    box_poses = captured["prediction"][:, 36:43]

    # -- D1: identity against the ORIGINAL formal report first; a mismatch
    # stops the binding outright (the run-internal circularity check below
    # can never catch a doctored or diverged report)
    if report_row is None:
        report_row = load_report_rows(rq_dir).get(demo)
    if report_row is None:
        raise KeyError(f"{demo}: no row in the formal report of {rq_dir}")
    if report_row.get("accepted"):
        raise ValueError(f"{demo}: formal report row is accepted -- only "
                         f"rejected segments can be bound")
    vs_report = identity_vs_report(report_row, captured)

    # -- run-internal consistency (R1 section 2.3): the final diagnostics
    # must be the snapshot's own failure
    rec: dict = {
        "demo": demo, "t_fail": t, "pass_index": captured["pass_index"],
        "run_status": res.status,
        "run_t_fail": res.diagnostics.get("t_fail"),
        "run_final_stage": res.diagnostics.get("final_stage"),
        "run_reason": res.diagnostics.get("reason"),
        "identity_match": {
            "t_fail": res.diagnostics.get("t_fail") == t,
            "last_attempt_id": (res.diagnostics.get("last_attempt_id")
                                == captured["last_attempt_id"]),
            "last_check_id": (res.diagnostics.get("last_check_id")
                              == captured["chk"]["check_id"]),
            "status": (res.diagnostics.get("last_attempt", {}).get("status")
                       == captured["res_fields"]["status"])},
        "identity_vs_report": vs_report,
        "wall_s": round(wall, 2),
        "frame_status": list(res.diagnostics.get("frame_status") or []),
        "c2_diag": captured["c2_diag"],
        "last_attempt": {"solver_fields": captured["res_fields"],
                         "chk": captured["chk"],
                         "attempt_id": captured["last_attempt_id"]},
    }
    if not all(vs_report["match"].values()) or \
            not all(rec["identity_match"].values()):
        rec["verdict"] = "IDENTITY_MISMATCH"
        failed = ([k for k, v in vs_report["match"].items() if not v]
                  + [f"run_internal:{k}" for k, v in
                     rec["identity_match"].items() if not v])
        rec["identity_failed_items"] = failed
        write_products(out, f"fail_bind2_{demo}", rec,
                       {"z_cand": z_cand, "prefix": np.stack(prefix),
                        "box_poses": box_poses,
                        **context_arrays(ctx)})
        return rec

    # -- full audits: (a) the frozen candidate with C2 ON (the run's
    # condition), (b) the SAME candidate with C2 off
    audits: dict = {}
    arrays: dict = {"z_cand": z_cand,
                    "prefix": np.stack(prefix), "box_poses": box_poses,
                    **context_arrays(ctx)}
    for label, actx in (("c2_on", ctx), ("c2_off", c2_off_ctx(ctx))):
        arec, aarr = full_candidate_audit(
            pipe.spec, cfg, vcfg, actx, z_cand,
            prefix=prefix, box_poses=box_poses, label=label)
        audits[label] = arec
        arrays.update(aarr)

    # -- (c) the frame re-solved without C2: production escalation on a
    # FRESH pipeline (independent evaluator/solver, frozen context)
    pipe_off = RepairPipeline(spec=pipe.spec, cfg=cfg)
    pipe_off.validator.cfg = vcfg
    esc = escalate_frame(pipe_off, c2_off_ctx(ctx), t, ctx.z_pred, ctx.z_prev)
    rec["resolve_without_c2"] = {
        "skipped": esc["skipped"], "adopted": esc["adopted"],
        "adopted_source": esc["adopted_source"],
        "adopted_status": esc["adopted_status"],
        "last_status": esc["last_status"],
        "last_term_reason": esc["last_term_reason"],
        "last_check_reason": esc["last_check_reason"],
        "attempts": [{"source": a.get("source"), "status": a.get("status"),
                      "term_reason": a.get("term_reason"),
                      "slack": a.get("slack_used"),
                      "qp": a.get("qp_calls"),
                      "iters": a.get("iterations"),
                      "check_id": a.get("check_id")}
                     for a in esc["attempts"]],
        "skip": esc["skip"]}
    if esc["adopted"]:
        # the adopted candidate gets its own full audit under its own
        # identity -- never mixed with the frozen original; this covers
        # BOTH adoption shapes (skip pass-through and solved candidate)
        rrec, rarr = full_candidate_audit(
            pipe.spec, cfg, vcfg, c2_off_ctx(ctx), esc["adopted_z"],
            prefix=prefix, box_poses=box_poses, label="resolve_c2_off")
        audits["resolve_c2_off"] = rrec
        arrays.update(rarr)
        arrays["resolve_adopted_z"] = esc["adopted_z"]
    rec["audits"] = audits

    # -- classification (section 4.2): the two same-candidate branches
    # decide the frozen-candidate verdict; the re-solve is a SEPARATE
    # conclusion, never "the segment is recovered"
    rec["classification"] = classify_branches(
        audits.get("c2_off"), audits.get("c2_on"))
    # convention note: classify_branches(current=no-C2 branch, plus=C2)
    rec["classification_basis"] = {
        "no_c2_branch_pass": audits["c2_off"].get("branch_pass"),
        "plus_c2_branch_pass": audits["c2_on"].get("branch_pass")}
    rec["conclusion_strength"] = {
        "frozen_candidate": (
            "rejected_by_c2_rows" if (
                rec["classification"] == "C2_ONLY")
            else rec["classification"]),
        "resolve": ("c2_addition_failed_the_frame_under_fixed_flow"
                    if not esc["adopted"] else "frame_solved_without_c2"),
        "note": ("neither statement is a whole-segment recovery; the "
                 "original 1/6 pause verdict stands until a full rerun"),
    }
    rec["provenance"] = {
        "git": git_state(), "git_rev": git_rev(),
        "assets": asset_hashes(),
        "fps": cfg.fps, "format": AUDIT_FORMAT,
        "context_meta": {**context_meta(ctx),
                         "ctx_scalars": context_scalar_leaves(ctx)},
    }
    attach_integrity(rec, cfg, vcfg)
    write_products(out, f"fail_bind2_{demo}", rec, arrays)
    return rec


def _escalation_summary(esc: dict) -> dict:
    """JSON-safe projection of an escalation result (same shape bind()
    freezes into ``resolve_without_c2``)."""
    return {
        "skipped": esc["skipped"], "adopted": esc["adopted"],
        "adopted_source": esc["adopted_source"],
        "adopted_status": esc["adopted_status"],
        "last_status": esc["last_status"],
        "last_term_reason": esc["last_term_reason"],
        "last_check_reason": esc["last_check_reason"],
        "attempts": [{"source": a.get("source"), "status": a.get("status"),
                      "term_reason": a.get("term_reason"),
                      "slack": a.get("slack_used"),
                      "qp": a.get("qp_calls"),
                      "iters": a.get("iterations"),
                      "check_id": a.get("check_id")}
                     for a in esc["attempts"]],
        "skip": esc["skip"],
    }


def replay(out: Path, replay_dir: Path) -> dict:
    """Strict replay (reviewfix2 D2/D4/D5): candidates come ONLY from the
    generation-time index; every product is loaded with its NPZ sha256 and
    integrity manifest verified; the audits run under the FROZEN effective
    configs restored from the product (never the current YAML); the
    re-solve escalation AND its adopted candidate's audit are recomputed;
    and the comparison covers the COMPLETE records (all row descriptors and
    values, every interpolation point, strict velocities, task, prefix
    checks, classification) plus the arrays BITWISE.  The summary goes to
    ``replay_dir`` -- never the products directory."""
    idx = load_index(out, INDEX_NAME)
    replay_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    ok = True
    for stem in idx["stems"]:
        record, arrays = load_products(out, stem)
        entry: dict = {"stem": stem}
        results[stem] = entry
        integrity = verify_integrity(record)
        entry["integrity_ok"] = not integrity
        entry["integrity_mismatches"] = integrity[:20]
        if "audits" not in record:
            entry["replayed"] = False
            entry["reason"] = "identity mismatch record (no audits frozen)"
            ok = False
            continue
        cfg, vcfg = configs_from_record(record)
        spec = RepairPipeline(cfg=cfg).spec
        ctx = context_from(arrays, record["provenance"]["context_meta"])
        z_cand = arrays["z_cand"]
        prefix = [arrays["prefix"][i] for i in range(arrays["prefix"].shape[0])]
        box_poses = arrays["box_poses"]
        t = record["t_fail"]
        fresh: dict = {}
        fresh_arrays: dict = {}
        for label, actx in (("c2_on", ctx), ("c2_off", c2_off_ctx(ctx))):
            arec, aarr = full_candidate_audit(
                spec, cfg, vcfg, actx, z_cand,
                prefix=prefix, box_poses=box_poses, label=label)
            fresh[label] = arec
            fresh_arrays.update(aarr)
        # the re-solve itself: production escalation on a fresh pipeline
        pipe_off = RepairPipeline(spec=spec, cfg=cfg)
        pipe_off.validator.cfg = vcfg
        esc = escalate_frame(pipe_off, c2_off_ctx(ctx), t,
                             ctx.z_pred, ctx.z_prev)
        resolve_rec = _escalation_summary(esc)
        if esc["adopted"]:
            rrec, rarr = full_candidate_audit(
                spec, cfg, vcfg, c2_off_ctx(ctx), esc["adopted_z"],
                prefix=prefix, box_poses=box_poses, label="resolve_c2_off")
            fresh["resolve_c2_off"] = rrec
            fresh_arrays.update(rarr)
            fresh_arrays["resolve_adopted_z"] = esc["adopted_z"]
        classification = classify_branches(fresh["c2_off"], fresh["c2_on"])
        fresh_record = {
            "audits": fresh,
            "classification": classification,
            "classification_basis": {
                "no_c2_branch_pass": fresh["c2_off"].get("branch_pass"),
                "plus_c2_branch_pass": fresh["c2_on"].get("branch_pass")},
            "resolve_without_c2": resolve_rec,
            "conclusion_strength": {
                "frozen_candidate": (
                    "rejected_by_c2_rows"
                    if classification == "C2_ONLY" else classification),
                "resolve": ("c2_addition_failed_the_frame_under_fixed_flow"
                            if not esc["adopted"]
                            else "frame_solved_without_c2"),
                "note": ("neither statement is a whole-segment recovery; "
                         "the original 1/6 pause verdict stands until a "
                         "full rerun")},
        }
        # complete-record comparison over everything the replay recomputes
        diffs = deep_diff({k: record.get(k) for k in fresh_record},
                          fresh_record)
        # arrays: compare ONLY the recomputable subset (row values/bounds/
        # scales/viols + candidates); ctx/prefix/box_poses are replay INPUTS
        recomp = {k: arrays[k] for k in fresh_arrays if k in arrays}
        arr_diffs = arrays_diff(recomp, fresh_arrays)
        arr_diffs += [f"{k}: recomputed array absent from saved bundle"
                      for k in sorted(fresh_arrays) if k not in arrays]
        entry.update({
            "replayed": True,
            "record_diff_paths": diffs[:40],
            "n_record_diffs": len(diffs),
            "array_diffs": arr_diffs[:40],
            "classification_recomputed": classification,
            "classification_matches": classification
            == record.get("classification"),
            "matches": not diffs and not arr_diffs,
        })
        entry["ok"] = (entry["matches"]
                       and entry["classification_matches"]
                       and entry["integrity_ok"])
        ok = ok and entry["ok"]
    summary = {"ok": ok, "n": len(results),
               "expected": len(idx["stems"]),
               "recompute_scope": ("audits (all rows/interpolation points/"
                                   "strict velocity/task/prefix), resolve "
                                   "escalation + adopted-candidate audit, "
                                   "classification; arrays bitwise; run-level "
                                   "fields (frame_status/c2_diag/last_attempt)"
                                   " are generation-time inputs, not "
                                   "recomputed"),
               "results": results}
    (replay_dir / "fail_bind2_replay.json").write_text(
        json.dumps(summary, indent=1, default=float))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rq-dir", default=str(DEFAULT_RQ),
                    help="formal six-segment rq report dir (read-only)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output diag dir (default: the reviewfix2 20261002 "
                         "dir; never the frozen or superseded earlier ones)")
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--replay", action="store_true",
                    help="strict replay from the index into a SEPARATE "
                         "directory (default: <out>_replay)")
    args = ap.parse_args()
    out = Path(args.out)
    rq_dir = Path(args.rq_dir)
    if args.replay:
        replay_dir = out.parent / f"{out.name}_replay"
        summary = replay(out, replay_dir)
        print(f"[replay] {'ALL MATCH' if summary['ok'] else 'MISMATCH'} "
              f"({summary['n']} records) -> "
              f"{replay_dir / 'fail_bind2_replay.json'}")
        return
    report_rows = load_report_rows(rq_dir)
    failed = [d for d, r in report_rows.items() if not r["accepted"]]
    demos = args.demos or failed
    print(f"binding {len(demos)} rejected segments: {demos}")
    out.mkdir(parents=True, exist_ok=True)
    stems = []
    for demo in demos:
        rec = bind(demo, rq_dir, out, report_row=report_rows[demo])
        if "t_fail" in rec:
            stems.append(f"fail_bind2_{demo}")
        cls = rec.get("classification", rec.get("verdict", "n/a"))
        esc = rec.get("resolve_without_c2", {})
        idr = rec.get("identity_vs_report", {}).get("match", {})
        print(f"[{demo}] t_fail={rec.get('t_fail')} class={cls} "
              f"id_vs_report={all(idr.values()) if idr else None} "
              f"last={rec.get('last_attempt', {}).get('solver_fields', {}).get('status')}"
              f"/{rec.get('last_attempt', {}).get('solver_fields', {}).get('term_reason')} "
              f"resolve_adopted={esc.get('adopted')}")
    if stems:
        write_index(out, INDEX_NAME,
                    {"demos": demos, "stems": stems,
                     "rq_dir": str(rq_dir),
                     "report_sha256": _report_sha(rq_dir)})
    print(f"-> {out}")


def _report_sha(rq_dir: Path) -> str:
    import hashlib
    return hashlib.sha256(
        (rq_dir / "regen_report_diagnosis6_strict.json").read_bytes()
    ).hexdigest()


if __name__ == "__main__":
    main()
