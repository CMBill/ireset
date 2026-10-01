"""Bind the failing candidates of the train24fix2 six-segment regression and
classify the joint infeasibility with full-constraint counterfactuals
(train24fix2 plan section 9, risk row 2 -- the pre-registered action when
the six-segment run falls below 6/6: PAUSE, bind the failing candidates,
classify; any strategy change requires a new plan).

For every REJECTED segment of ``train24fix2_20260927_sixseg/rq`` the segment
is re-run deterministically (same code revision, same inputs) with the
read-only snapshot hook capturing, at the ``fail`` event, the LAST executed
attempt's candidate state, the failing check, the frame context (C2 enable
state, previous full-scene distances, previous box-frame hand arrays), the
accepted prefix, and the contact state.

Classification per segment (all rows uncapped, unrounded):

1. FULL row dump of the last candidate -- every violating row with its
   family (pre / collision / post, post split into anchors, foot speed,
   C2 hand speed), not just the collision-family evidence the verbose log
   carries.
2. Counterfactual "candidate without C2": the SAME candidate checked with
   the C2 rows disabled -- does it violate anything else?
3. Counterfactual "frame solved without C2": the same frame, same accepted
   prefix, same discovered-pair-free start, escalated exactly like the run
   (prediction -> prev_accepted -> slack) with the C2 rows off -- feasible
   and accepted would attribute the infeasibility to the C2 joint
   interaction.

Run: uv run --locked python scripts/train24fix2_fail_bind.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from ireset.data.teacher import load_teacher_npz
from ireset.embodied.contact_sites import infer_reference_schedule
from ireset.paths import DERIVED_DATA_DIR, TEACHER_TRAJ_DIR
from ireset.repair.config_io import load_mode_configs
from ireset.repair.context import ReferenceCondition
from ireset.repair.pipeline import RepairPipeline

REGEN = DERIVED_DATA_DIR / "teachers_regenerated"
RQ_DIR = REGEN / "train24fix2_20260927_sixseg" / "rq"
OUT_DIR = REGEN / "train24fix2_20260927_diag"
INF = 1e9


def row_families(solver, val, lo, hi, scl, names) -> list[dict]:
    """Every violating row with family + descriptor (uncapped, unrounded)."""
    viol = np.maximum(np.maximum(lo - val, val - hi), 0.0) / scl
    out = []
    n_coll = len(names)
    for i in np.nonzero(viol > 0)[0]:
        if i < 99:
            if i < 29:
                d = f"pre.joint_limit[j{i + 7}]"
            elif i < 58:
                d = f"pre.joint_vel[j{i - 29 + 7}]"
            elif i in (58, 61):
                d = f"pre.root_vel[{i}]"
            else:
                d = f"pre.trust[{i - 64}]"
            fam = "pre"
        elif i < 99 + n_coll:
            d, fam = f"coll[{names[i - 99]}]", "collision"
        else:
            j = i - 99 - n_coll
            fam = ("post.hand_anchor" if j < 18 else
                   "post.foot_anchor" if j < 48 else
                   "post.foot_speed" if j < 78 else "post.hand_speed")
            d = f"{fam}[{j}]"
        out.append({"family": fam, "row": d, "val": float(val[i]),
                    "lo": float(lo[i]), "hi": float(hi[i]),
                    "viol_norm": float(viol[i])})
    return out


def c2_off_ctx(ctx):
    """Same frame context with ONLY the C2 rows disabled."""
    return replace(ctx, c2_hand_enabled={}, c2_hand_reason={})


def solve_frame_escalated(pipe, ctx, t, z_pred, z_prev):
    """The pipeline's own escalation for ONE frame (fresh pair state)."""
    pipe.solver.clear_discovered_pairs()
    pipe.solver.clear_probe_pairs()
    attempts, checks, frame_recs = [], [], []
    res, chk, _bt, _bti, _rec = pipe._solve_attempt(
        z_pred, "cf_prediction", ctx, t, z_prev, 0, attempts, checks, frame_recs)
    if not chk.ok and z_prev is not None:
        res2, chk2, _b2, _t2, _r2 = pipe._solve_attempt(
            z_prev, "cf_prev_accepted", ctx, t, z_prev,
            len(frame_recs), attempts, checks, frame_recs)
        if not chk2.ok:
            res3, chk3, _b3, _t3, _r3 = pipe._solve_attempt(
                z_prev, "cf_slack_fallback", ctx, t, z_prev,
                len(frame_recs), attempts, checks, frame_recs,
                force_slack=True)
            return res3, chk3, attempts
        return res2, chk2, attempts
    return res, chk, attempts


def bind(demo: str) -> dict:
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
        ctx = data["ctx"]
        res = data["last_res"]
        chk = data["last_chk"]
        captured["t"] = data["t"]
        captured["z_cand"] = np.asarray(res.z).copy()
        captured["status"] = res.status
        captured["chk"] = {"ok": chk.ok, "reason": chk.reason,
                           "viol": float(chk.viol), "rows": chk.rows,
                           "c2": {s: dict(v) for s, v in chk.c2.items()}}
        captured["ctx_fields"] = {
            "t": ctx.t,
            "c2_enabled": dict(ctx.c2_hand_enabled),
            "c2_reason": dict(ctx.c2_hand_reason),
            "prev_scene_dist": dict(ctx.prev_hand_scene_dist),
            "prev_hand_dist": dict(ctx.prev_hand_dist),
            "z_prev": None if ctx.z_prev is None else ctx.z_prev.copy(),
            "z_pred": ctx.z_pred.copy(),
            "box_pose": ctx.box_pose.copy(),
            "box_pose_prev": (None if ctx.box_pose_prev is None
                              else ctx.box_pose_prev.copy()),
            "prev_hand_local": {s: (None if v is None else v.copy())
                                for s, v in ctx.prev_hand_local.items()},
            "prev_sole_world": {s: (None if v is None else v.copy())
                                for s, v in ctx.prev_sole_world.items()},
            "hand_anchor": {s: (None if v is None else v.copy())
                            for s, v in ctx.hand_anchor.items()},
            "foot_anchor": {s: (None if v is None else v.copy())
                            for s, v in ctx.foot_anchor.items()},
            "ref_hold": dict(ctx.ref_hold),
        }
        captured["accepted_prefix"] = [a.copy() for a in data["accepted"]]
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
    rec = {"demo": demo, "t_fail": t, "run_status": res.status,
           "wall_s": round(wall, 2),
           "frame_status": list(res.diagnostics.get("frame_status") or []),
           "c2_diag": captured["c2_diag"],
           "last_attempt": {"status": captured["status"],
                            "chk": captured["chk"]}}
    cf = captured["ctx_fields"]

    # rebuild a ctx object for the row dumps / counterfactual solves
    from ireset.repair.scp import FrameContext
    ctx_on = FrameContext(
        t=cf["t"], box_pose=cf["box_pose"], z_pred=cf["z_pred"],
        z_prev=cf["z_prev"], ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor=cf["hand_anchor"], foot_anchor=cf["foot_anchor"],
        prev_hand_local=cf["prev_hand_local"], prev_sole_world=cf["prev_sole_world"],
        box_pose_prev=cf["box_pose_prev"], ref_hold=cf["ref_hold"],
        prev_hand_dist=cf["prev_hand_dist"],
        c2_hand_enabled=cf["c2_enabled"], c2_hand_reason=cf["c2_reason"],
        prev_hand_scene_dist=cf["prev_scene_dist"])
    # the original ctx carried the run's mesh terms; rebuild them exactly
    L, delta, fixed = pipe._frame_mesh(cond, t)
    ctx_on.mesh_L, ctx_on.delta_ref, ctx_on.mesh_fixed = L, delta, fixed
    z_cand = captured["z_cand"]

    # (1) full row dump of the last candidate (C2 on -- the run's condition)
    val, lo, hi, scl, names, _ids = pipe.solver.check_rows(z_cand, ctx_on)
    rec["candidate_rows_on"] = row_families(pipe.solver, val, lo, hi, scl, names)

    # (2) the SAME candidate with C2 disabled
    val2, lo2, hi2, scl2, names2, _i2 = pipe.solver.check_rows(z_cand, c2_off_ctx(ctx_on))
    rows_off = row_families(pipe.solver, val2, lo2, hi2, scl2, names2)
    rec["candidate_rows_c2off"] = rows_off
    rec["candidate_passes_without_c2"] = bool(
        pipe.solver.violation(val2, lo2, hi2, scl2) <= 1e-6)

    # (3) the frame re-solved with C2 disabled, same prefix escalation
    res3, chk3, attempts3 = solve_frame_escalated(
        pipe, c2_off_ctx(ctx_on), t, cf["z_pred"], cf["z_prev"])
    rec["frame_solved_without_c2"] = {
        "accepted": bool(chk3.ok), "status": res3.status,
        "reason": chk3.reason, "viol": float(chk3.viol),
        "qp_calls": int(res3.qp_calls),
        "attempts": [{"source": a.get("source"),
                      "status": a.get("status"),
                      "slack": a.get("slack_used"),
                      "qp": a.get("qp_calls")} for a in attempts3]}
    return rec


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((RQ_DIR / "regen_report_diagnosis6_strict.json").read_text())
    failed = [r["demo"] for r in report["rows"] if not r["accepted"]]
    print(f"binding {len(failed)} rejected segments: {failed}")
    out = []
    for demo in failed:
        rec = bind(demo)
        (OUT_DIR / f"fail_bind_{demo}.json").write_text(json.dumps(rec, indent=1))
        out.append(rec)
        print(f"[{demo}] t_fail={rec.get('t_fail')} "
              f"last_status={rec.get('last_attempt', {}).get('status')} "
              f"cand_passes_without_c2={rec.get('candidate_passes_without_c2')} "
              f"frame_solves_without_c2={rec.get('frame_solved_without_c2', {}).get('accepted')}")
        fams = {}
        for r in rec.get("candidate_rows_on", []):
            fams[r["family"]] = fams.get(r["family"], 0) + 1
        print(f"   violating families (c2 on): {fams}")
        fams2 = {}
        for r in rec.get("candidate_rows_c2off", []):
            fams2[r["family"]] = fams2.get(r["family"], 0) + 1
        print(f"   violating families (c2 off): {fams2}")
    (OUT_DIR / "fail_bind_summary.json").write_text(json.dumps(out, indent=1))
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
