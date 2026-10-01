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
``derived/data/teachers_regenerated/train24fix2_reviewfix_20261001/diag/``
by default -- NEVER the frozen ``train24fix2_20260927_diag/``.

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
    assert_snapshot_matches_live,
    classify_branches,
    context_arrays,
    context_from,
    context_meta,
    context_scalar_leaves,
    escalate_frame,
    freeze_context,
    full_candidate_audit,
    load_products,
    write_products,
)

REGEN = DERIVED_DATA_DIR / "teachers_regenerated"
DEFAULT_RQ = REGEN / "train24fix2_20260927_sixseg" / "rq"
DEFAULT_OUT = REGEN / "train24fix2_reviewfix_20261001" / "diag"


def c2_off_ctx(ctx):
    """Same frozen context with ONLY the C2 row-enable state cleared (the
    bound parameter stays 0.02598 in the config -- the rows turn off
    through the context, never by replacing the whole config)."""
    return replace(freeze_context(ctx), c2_hand_enabled={}, c2_hand_reason={})


def bind(demo: str, rq_dir: Path, out: Path) -> dict:
    scene, human, fps = load_teacher_npz(TEACHER_TRAJ_DIR / f"{demo}_original.npz")
    cond = ReferenceCondition(human, scene[:, 36:43],
                              infer_reference_schedule(human, scene[:, 36:43]))
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

    # -- identity cross-check against the formal run's own report (R1
    # section 2.3): a mismatch stops the binding, never "a similar failure"
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
        "wall_s": round(wall, 2),
        "frame_status": list(res.diagnostics.get("frame_status") or []),
        "c2_diag": captured["c2_diag"],
        "last_attempt": {"solver_fields": captured["res_fields"],
                         "chk": captured["chk"],
                         "attempt_id": captured["last_attempt_id"]},
    }
    if not all(rec["identity_match"].values()):
        rec["verdict"] = "IDENTITY_MISMATCH"
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
    if esc["adopted"] and not esc["skipped"]:
        # the re-solved candidate gets its own full audit under its own
        # identity -- never mixed with the frozen original
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
        "effective_configs": {"strict_repair": vars(cfg),
                              "strict_validation": vars(vcfg)},
        "fps": fps, "format": AUDIT_FORMAT,
        "context_meta": {**context_meta(ctx),
                         "ctx_scalars": context_scalar_leaves(ctx)},
    }
    write_products(out, f"fail_bind2_{demo}", rec, arrays)
    return rec


def replay(out: Path) -> dict:
    """Recompute the same-candidate branches from the frozen files (no
    trajectory re-solve, no re-search) and compare with the saved verdicts
    (plan section 4.3: replay never substitutes a re-found candidate)."""
    rcfg, vcfg = load_mode_configs("strict")
    cfg = replace(rcfg, regen_mode=True)
    results = {}
    for jpath in sorted(out.glob("fail_bind2_*.json")):
        record, arrays = load_products(out, jpath.stem)
        if "audits" not in record:
            results[jpath.stem] = {"replayed": False,
                                   "reason": "identity mismatch record"}
            continue
        spec = RepairPipeline(cfg=cfg).spec
        ctx = context_from(arrays, record["provenance"]["context_meta"])
        z_cand = arrays["z_cand"]
        prefix = [arrays["prefix"][i] for i in range(arrays["prefix"].shape[0])]
        box_poses = arrays["box_poses"]
        rep: dict = {"replayed": True}
        fresh: dict = {}
        for label, actx in (("c2_on", ctx), ("c2_off", c2_off_ctx(ctx))):
            arec, _a = full_candidate_audit(
                spec, cfg, vcfg, actx, z_cand,
                prefix=prefix, box_poses=box_poses, label=label)
            fresh[label] = arec
            saved = record["audits"][label]
            same = (
                arec.get("branch_pass") == saved.get("branch_pass")
                and arec["endpoint"]["viol_sum"] == saved["endpoint"]["viol_sum"]
                and arec["task"]["mpjpe_m"] == saved["task"]["mpjpe_m"]
                and arec["interpolation"]["pass"] == saved["interpolation"]["pass"]
                and arec["prefix_validation"]["passed"]
                == saved["prefix_validation"]["passed"])
            rep[label] = {"matches_saved": bool(same),
                          "branch_pass": arec.get("branch_pass"),
                          "viol_sum": arec["endpoint"]["viol_sum"],
                          "task_mpjpe": arec["task"]["mpjpe_m"]}
        rep["classification_recomputed"] = classify_branches(
            fresh["c2_off"], fresh["c2_on"])
        rep["classification_matches"] = (
            rep["classification_recomputed"] == record["classification"])
        results[jpath.stem] = rep
    (out / "fail_bind2_replay.json").write_text(
        json.dumps(results, indent=1, default=float))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rq-dir", default=str(DEFAULT_RQ),
                    help="formal six-segment rq report dir (read-only)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output diag dir (default: the reviewfix 20261001 "
                         "dir, never the frozen 20260927 one)")
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--replay", action="store_true",
                    help="recompute same-candidate branches from the frozen "
                         "files and compare (no re-solve)")
    args = ap.parse_args()
    out = Path(args.out)
    rq_dir = Path(args.rq_dir)
    if args.replay:
        res = replay(out)
        ok = all(v.get("replayed") and
                 all(b.get("matches_saved") for b in v.values()
                     if isinstance(b, dict) and "matches_saved" in b)
                 for v in res.values())
        print(f"[replay] {'ALL MATCH' if ok else 'MISMATCH'} -> "
              f"{out / 'fail_bind2_replay.json'}")
        return
    report = json.loads((rq_dir / "regen_report_diagnosis6_strict.json").read_text())
    failed = [r["demo"] for r in report["rows"] if not r["accepted"]]
    demos = args.demos or failed
    print(f"binding {len(demos)} rejected segments: {demos}")
    out.mkdir(parents=True, exist_ok=True)
    for demo in demos:
        rec = bind(demo, rq_dir, out)
        cls = rec.get("classification", rec.get("verdict", "n/a"))
        esc = rec.get("resolve_without_c2", {})
        print(f"[{demo}] t_fail={rec.get('t_fail')} class={cls} "
              f"last={rec.get('last_attempt', {}).get('solver_fields', {}).get('status')}"
              f"/{rec.get('last_attempt', {}).get('solver_fields', {}).get('term_reason')} "
              f"resolve_adopted={esc.get('adopted')}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
