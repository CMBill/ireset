"""D-stage forensic replays for the train24 failure-fix plan (section 3).

Read-only evidence tooling: replays segments through the SAME pipeline and
configs as the frozen train24 regen run, using the pipeline's opt-in
``snapshot`` hook (never a decision path), and answers the four D questions:

- ``dv``  bind every velocity micro-exceedance frame to its ADOPTION path,
  previous accepted state, slack usage, and the normalization budget V1
  would need (same-formula recheck as the independent validator);
- ``df``  full-constraint recheck of each frame-rejected candidate on a
  FRESH solver (fixed-pre + collision + fixed-post + every protocol
  interpolation point, with units), plus the counterfactual that replaces
  ONLY the plan-section-2.4 cell (regen accept band floor -1.5 -> -1.8 mm)
  and the ``band_floor_only`` verdict;
- ``dr``  per-frame corridor/skip audit of the two recall-failed segments
  across every entry (skip, solve, early return), and the formal zero-QP
  counterexample (fresh solver on a frozen context);
- ``dq``  label-protocol-v1 quality baseline over the frozen six-segment and
  train24 kept NPZs (input-hashed, frozen before any fix changes chains).

Outputs go to ``derived/data/teachers_regenerated/train24fix_20260927_diag/``.
Nothing here feeds back into the pipeline, the solver, or any gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data.teacher import load_teacher_npz  # noqa: E402
from ireset.embodied.contact_sites import infer_reference_schedule  # noqa: E402
from ireset.paths import PROJECT_ROOT, TEACHER_TRAJ_DIR  # noqa: E402
from ireset.repair.config_io import load_mode_configs  # noqa: E402
from ireset.repair.constraints import FrameEvaluator  # noqa: E402
from ireset.repair.context import ReferenceCondition, RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.scp import FrameSolver, cfg_noncontact  # noqa: E402

DIAG_ROOT = (PROJECT_ROOT / "derived/data/teachers_regenerated"
             / "train24fix_20260927_diag")

#: frozen train24 evidence (regen_final report, 2026-09-26): the replay is
#: expected to reproduce these bit-for-bit -- the assert validates the
#: forensic replay, not the other way round
DV_EXPECTED = {
    "sub7_largebox_000": {"frame": 101, "joint": 26,
                          "value": 12.000000401362618},
    "sub12_largebox_077": {"frame": 203, "joint": 16,
                           "value": 12.000000105688033},
    "sub8_largebox_048": {"frame": 12, "joint": 26,
                          "value": 12.000000003086466},
}
V_BUDGET_RAD = 1.0 / 30.0 * 12.0 * 1e-6     # dt*limit*1e-6 = 4e-7 rad (plan 4.1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


class Collector:
    """Snapshot sink: copies arrays it keeps, never feeds anything back."""

    def __init__(self) -> None:
        self.frames: list[dict] = []        # per-frame ctx (live refs, read-only use)
        self.adopted: list[dict] = []
        self.fails: list[dict] = []
        self.final_fails: list[dict] = []

    def __call__(self, kind: str, data: dict) -> None:
        if kind == "frame":
            self.frames.append(data)
        elif kind == "adopted":
            self.adopted.append({**data, "z": np.array(data["z"], copy=True)})
        elif kind == "fail":
            self.fails.append({
                **data,
                "z_pred": np.array(data["z_pred"], copy=True),
                "z_prev": (None if data["z_prev"] is None
                           else np.array(data["z_prev"], copy=True)),
                "z_cand": np.array(data["last_res"].z, copy=True),
            })
        elif kind == "final_fail":
            self.final_fails.append({
                **data,
                "trajectory": np.array(data["trajectory"], copy=True),
            })


def setup_pipe(verbose: bool = False):
    """Exact replay instance of the train24 regen run (mode=regen, strict,
    no task-gate YAML -> effective gate stays the 0.40 default)."""
    rcfg, vcfg = load_mode_configs("strict")
    from dataclasses import replace as _replace

    regen_cfg = _replace(rcfg, regen_mode=True)
    pipe = RepairPipeline(cfg=regen_cfg)
    pipe.validator.cfg = vcfg
    pipe.verbose = verbose
    return pipe, regen_cfg, vcfg


def run_demo(pipe: RepairPipeline, demo: str):
    src = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
    scene, human, fps = load_teacher_npz(src)
    cond = ReferenceCondition(human, scene[:, 36:43],
                              infer_reference_schedule(human, scene[:, 36:43]))
    t0 = time.perf_counter()
    res = pipe.repair_trajectory(scene, cond)
    return res, scene, cond, fps, time.perf_counter() - t0


def fresh_solver(pipe: RepairPipeline) -> FrameSolver:
    """Independent evaluator+solver sharing only the read-only spec/registry
    (plan 3: the analysis must not touch the formal run's caches)."""
    ev2 = FrameEvaluator(pipe.spec, pipe.pairs, pipe.cfg)
    s2 = FrameSolver(ev2, pipe.cfg)
    s2.probe_cfg = pipe.validator.cfg
    return s2


def strict_velocity_exceedances(q: np.ndarray, vcfg) -> list[dict]:
    """Same formula as ``validation._check_velocities`` (strict ``>``),
    unrounded values, attributed to the post-difference frame."""
    dt = 1.0 / vcfg.fps
    if q.shape[0] < 2:
        return []
    dq = np.abs(np.diff(q[:, 7:36], axis=0))            # [T-1, 29]
    series = dq.max(1) / dt
    lim = float(vcfg.joint_velocity_limit_radps)
    out = []
    for f in np.nonzero(series > lim)[0]:
        j = int(np.argmax(dq[f]))
        out.append({"frame": int(f + 1), "diff_frames": [int(f), int(f + 1)],
                    "joint_index": j, "value": float(series[f]),
                    "limit": lim, "exceedance": float(series[f] - lim)})
    return out


# ---------------------------------------------------------------- dv ----------

def cmd_dv(demos: list[str], out: Path) -> None:
    for demo in demos:
        pipe, _rcfg, vcfg = setup_pipe()
        col = Collector()
        pipe.snapshot = col
        res, _scene, _cond, fps, wall = run_demo(pipe, demo)
        rec: dict = {"demo": demo, "status": res.status, "fps": fps,
                     "wall_s": round(wall, 1), "exceedances": []}
        trajs = [{"pass_index": ff["pass_index"],
                  "source": "final_fail_snapshot",
                  "final_summary": ff["final"].summary(),
                  "traj": ff["trajectory"]} for ff in col.final_fails]
        if not trajs and res.accepted:
            # accepted-but-kept-failure segments would land here; the V-class
            # demos are rejected_final so this is a defensive branch
            trajs = [{"pass_index": 0, "source": "accepted_result",
                      "final_summary": "passed", "traj": res.trajectory}]
        adopted = {(a["pass_index"], a["t"]): a for a in col.adopted}
        for tr in trajs:
            q = tr["traj"][:, 7:36]
            for ex in strict_velocity_exceedances(tr["traj"], vcfg):
                f = ex["frame"]
                prev = q[f - 1, ex["joint_index"]]
                cur = q[f, ex["joint_index"]]
                limdt = 1.0 / vcfg.fps * vcfg.joint_velocity_limit_radps
                need = float(abs(cur - np.clip(cur, prev - limdt, prev + limdt)))
                a_cur = adopted.get((tr["pass_index"], f))
                a_prev = adopted.get((tr["pass_index"], f - 1))
                ex_rec = {
                    **ex, "pass": tr["pass_index"],
                    "budget_needed_rad": need,
                    "within_v1_budget": bool(need <= V_BUDGET_RAD),
                    "adoption_cur": (None if a_cur is None else {
                        "status": a_cur["status"], "slack": a_cur["slack"],
                        "qp_calls": a_cur["qp_calls"],
                        "iterations": a_cur["iterations"]}),
                    "adoption_prev": (None if a_prev is None else {
                        "status": a_prev["status"], "slack": a_prev["slack"]}),
                }
                rec["exceedances"].append(ex_rec)
        exp = DV_EXPECTED.get(demo)
        if exp is not None:
            got = [(e["pass"], e["frame"], e["joint_index"], e["value"])
                   for e in rec["exceedances"]]
            want = (1, exp["frame"], exp["joint"], exp["value"])
            rec["replay_matches_frozen_evidence"] = want in got
            if want not in got:
                rec["frozen_evidence_expected"] = want
        # full adopted-state series for offline re-analysis
        np.savez_compressed(out / f"dv_states_{demo}.npz",
                            **{f"p{a['pass_index']}_t{a['t']}": a["z"]
                               for a in col.adopted})
        path = out / f"dv_{demo}.json"
        path.write_text(json.dumps(rec, indent=1, default=float))
        print(f"[dv] {demo}: {res.status}, "
              f"{len(rec['exceedances'])} exceedance(s) -> {path.name}")


# ---------------------------------------------------------------- df ----------

def _dump_rows(val, lo, hi, scl, names_all, n_coll: int) -> list[dict]:
    rows = []
    for k in range(len(val)):
        block = ("fixed_pre" if k < 99 else
                 "collision" if k < 99 + n_coll else "fixed_post_probe")
        rows.append({
            "row": names_all[k] if k < len(names_all) else "?",
            "block": block, "val": float(val[k]), "lo": float(lo[k]),
            "hi": float(hi[k]), "scale": float(scl[k]),
            "viol_lo": float(max(lo[k] - val[k], 0.0)),
            "viol_hi": float(max(val[k] - hi[k], 0.0)),
        })
    return rows


def _probe_audit(pipe, s2, ctx, z, z_prev, vcfg) -> dict:
    """Offline replica of ``_frame_valid``'s interpolation loop: EVERY
    protocol subdivision point is evaluated even when the endpoint failed
    (the pipeline returns early there -- its ``probe=null`` proves nothing)."""
    if z_prev is None or not vcfg.check_interpolated_states:
        return {"checked": False, "reason": "no_prev_or_disabled"}
    scene_a = np.zeros(43)
    scene_a[0:36] = z_prev
    scene_a[36:43] = ctx.box_pose_prev if ctx.box_pose_prev is not None else ctx.box_pose
    scene_b = np.zeros(43)
    scene_b[0:36] = z
    scene_b[36:43] = ctx.box_pose
    motion = pipe.validator._frame_motion_m(scene_a, scene_b)
    n_sub = int(np.clip(np.ceil(motion / vcfg.max_substep_motion_m),
                        1, vcfg.max_subdivisions + 1))
    out = {"checked": True, "motion_m": float(motion), "n_sub": n_sub,
           "points": [], "pass": True}
    for i in range(1, n_sub):
        mid = pipe.validator._interpolate(scene_a, scene_b, i / n_sub)
        ctx_mid = replace(ctx, box_pose=mid[36:43].copy())
        # rescan per probe like the formal _frame_valid: the endpoint's
        # candidate set cannot see pairs that come into range only at
        # interior states (review P2 -- a box sweeping through the robot
        # mid-frame must not audit as "pass" because the endpoint had no
        # robot_box candidate at all)
        s2._frame_pairs = None
        vm, lm, hm, sm, nm, _im = s2.check_rows(
            mid[0:36], ctx_mid, band_margin_m=0.0, probe_mode=True)
        n_coll = 99 + len(nm)
        viol_m = s2.violation(vm[99:n_coll], lm[99:n_coll], hm[99:n_coll],
                              sm[99:n_coll])
        pt = {"i": i, "alpha": i / n_sub, "viol": float(viol_m), "rows": []}
        if viol_m > 1e-6:
            out["pass"] = False
            for k in range(n_coll - 99):
                v, l_, h_ = vm[99 + k], lm[99 + k], hm[99 + k]
                if v < l_ - 1e-9 or v > h_ + 1e-9:
                    pt["rows"].append({"pair": nm[k] if k < len(nm) else "?",
                                       "val_m": float(v), "lo_m": float(l_),
                                       "hi_m": float(h_)})
        out["points"].append(pt)
    return out


def cmd_df(demos: list[str], out: Path) -> None:
    for demo in demos:
        pipe, rcfg, vcfg = setup_pipe()
        col = Collector()
        pipe.snapshot = col
        res, scene, _cond, fps, wall = run_demo(pipe, demo)
        rec: dict = {"demo": demo, "status": res.status, "fps": fps,
                     "wall_s": round(wall, 1), "fail_events": []}
        if res.accepted or not col.fails:
            rec["note"] = "no frame-failure event captured"
        for fe in col.fails:
            t = fe["t"]
            ctx, z, z_prev = fe["ctx"], fe["z_cand"], fe["z_prev"]
            s2 = fresh_solver(pipe)
            val, lo, hi, scl, names, _ids = s2.check_rows(z, ctx)
            names_all, _ids_all = s2.row_descriptors()
            if len(names_all) != len(val):
                raise RuntimeError(f"{demo} t={t}: descriptor/value mismatch")
            n_coll = len(names)
            dec_orig = bool(s2.violation(val, lo, hi, scl) <= 1e-6)
            # counterfactual: ONLY the section-2.4 cell (regen accept band
            # floor) moves -1.5 -> -1.8 mm; every other bound untouched
            band_floor = -(rcfg.contact_band_pen_m) + 0.0005
            lo_cf, scl_cf = lo.copy(), scl.copy()
            changed = []
            for k in range(99, 99 + n_coll):
                if (np.isfinite(hi[k]) and abs(lo[k] - band_floor) < 1e-12):
                    lo_cf[k] = cfg_noncontact(rcfg)
                    scl_cf[k] = max(min(hi[k] - lo_cf[k],
                                        rcfg.penetration_scale_m), 1e-4)
                    changed.append(names[k - 99])
            dec_cf = bool(s2.violation(val, lo_cf, hi, scl_cf) <= 1e-6)
            probes = _probe_audit(pipe, s2, ctx, z, z_prev, vcfg)
            vel = (strict_velocity_exceedances(
                np.stack([z_prev, z]) if z_prev is not None else z[None], vcfg)
                if z_prev is not None else [])
            ev2 = FrameEvaluator(pipe.spec, pipe.pairs, pipe.cfg)
            s3 = FrameSolver(ev2, pipe.cfg)
            s3.probe_cfg = pipe.validator.cfg
            mpjpe = float(s3.evaluate(z, ctx)["mpjpe"])
            task_ok = bool(mpjpe <= rcfg.task_mpjpe_tol_m)
            vel_ok = not vel
            # accepted prefix + candidate: independent validator, PREFIX
            # conclusion (the frame gate itself is the row/probe audit above)
            n_pref = len(fe["accepted"])
            prefix_scene = np.zeros((n_pref + 1, 43))
            for i, a in enumerate(fe["accepted"]):
                prefix_scene[i, 0:36] = a
            prefix_scene[n_pref, 0:36] = z
            prefix_scene[:, 36:43] = fe["prediction"][: n_pref + 1, 36:43]
            prep = pipe.validator.validate(prefix_scene, exclude=("box_ground",))
            rows = _dump_rows(val, lo, hi, scl, names_all, n_coll)
            rows_cf = [r for r in rows if r["row"] in set(changed)]
            band_rows_bad = [r["row"] for r in rows
                             if r["row"] in set(changed)
                             and (r["viol_lo"] > 0 or r["viol_hi"] > 0)]
            deep = [r["row"] for r in rows
                    if r["block"] == "collision" and r["val"] < -0.002]
            if dec_orig:
                cls = "endpoint_already_passing_offline"
            elif probes.get("pass") is False:
                cls = "interpolated"
            elif dec_cf and probes.get("pass") is True and vel_ok and task_ok \
                    and prep.passed:
                # the frame is freed by the floor change ONLY when every
                # applicable check passes -- including the recomputed
                # independent validation of the accepted prefix + candidate
                # (review P2: a prefix-level failure must not hide behind a
                # frame-level counterfactual)
                cls = "band_floor_only"
            elif deep:
                cls = "deep_penetration"
            elif any("foot_speed" in r["row"] and (r["viol_lo"] or r["viol_hi"])
                     for r in rows):
                cls = "foot_speed"
            elif any("joint_limit" in r["row"] and (r["viol_lo"] or r["viol_hi"])
                     for r in rows):
                cls = "mixed_joint_limit"
            else:
                cls = "mixed_other"
            rec["fail_events"].append({
                "t": t, "pass": fe["pass_index"],
                "run_reason": fe["last_chk"].reason,
                "run_status": fe["last_rec"]["status"],
                "classification": cls,
                "endpoint_decision_original": dec_orig,
                "endpoint_decision_counterfactual": dec_cf,
                "counterfactual_rows_changed": changed,
                "band_rows_violating_original": band_rows_bad,
                "deep_rows_beyond_validator_floor": deep,
                "probes": probes,
                "velocity_strict_exceedances": vel,
                "task_mpjpe": mpjpe, "task_ok": task_ok,
                "prefix_validator": {
                    "pass": "prefix_conclusion",
                    "passed": bool(prep.passed), "status": prep.status,
                    "failed": {k: {"location": c.location,
                                   "first_failure_frame": c.first_failure_frame,
                                   "evidence": c.evidence}
                               for k, c in prep.checks.items() if not c.passed}},
                "rows_violating": [r for r in rows
                                   if r["viol_lo"] > 0 or r["viol_hi"] > 0],
                "rows_changed_cf_values": rows_cf,
                "units": ("fixed_pre rows 0:29 rad (limits), 29:64 rad (velocities "
                          "per dt), 64:99 unitless trust; collision rows m; "
                          "fixed_post rows m (anchors) or m/frame (speeds)"),
            })
        path = out / f"df_{demo}.json"
        path.write_text(json.dumps(rec, indent=1, default=float))
        cls_list = [e["classification"] for e in rec["fail_events"]]
        print(f"[df] {demo}: {res.status} -> {path.name} class={cls_list}")


# ---------------------------------------------------------------- dr ----------

def cmd_dr(demos: list[str], out: Path, zero_qp_frames: dict[str, list[int]]) -> None:
    for demo in demos:
        pipe, _rcfg, vcfg = setup_pipe(verbose=True)
        col = Collector()
        pipe.snapshot = col
        res, _scene, _cond, fps, wall = run_demo(pipe, demo)
        frames_by_key = {(f["pass_index"], f["t"]): f for f in col.frames}
        adopted = {(a["pass_index"], a["t"]): a for a in col.adopted}
        # the check record that DECIDED each frame's adoption: the skip check
        # for skipped frames, otherwise the adopted attempt's own check id
        # (attempt records carry check_id + adopted after finalization).
        # Records logged before their id was assigned (backtrack checks and
        # pre-fix logs) fall back to ORDER-based pass segmentation: pass 1
        # records all follow the first "p1:" id in the sequential log.
        checks_by_id = {r.get("check_id"): r for r in pipe.frame_log
                        if r.get("kind") == "check" and r.get("check_id")}
        checks_at: dict[tuple[int, int], list[dict]] = {}
        attempts_at: dict[tuple[int, int], list[dict]] = {}
        running_pass = 0
        for r in pipe.frame_log:
            cid = r.get("check_id") or ""
            if cid.startswith("p") and ":" in cid:
                running_pass = max(running_pass, int(cid.split(":")[0][1:]))
            pass_idx = (int(cid.split(":")[0][1:])
                        if cid.startswith("p") and ":" in cid else running_pass)
            if r.get("kind") == "check":
                checks_at.setdefault((pass_idx, r["t"]), []).append(r)
            elif r.get("kind") == "attempt":
                attempts_at.setdefault((pass_idx, r["t"]), []).append(r)

        def deciding_check(p: int, t: int) -> dict:
            a = adopted.get((p, t))
            recs = checks_at.get((p, t), [])
            if a is not None and a["status"] == "skipped":
                return next((r for r in recs if r.get("caller") == "skip"), {})
            atts = attempts_at.get((p, t), [])
            adopted_att = next((r for r in atts if r.get("adopted")), None)
            if adopted_att is None:
                return {}
            rec = checks_by_id.get(adopted_att.get("check_id"))
            if rec:
                return rec
            # fallback for records serialized before id assignment: the
            # attempt's own source names its check's caller uniquely
            src = adopted_att.get("source")
            cands = [r for r in recs if r.get("caller") == src]
            return cands[-1] if cands else {}

        table, r2_ev, r1_ev = [], [], []
        for (p, t), fr in sorted(frames_by_key.items()):
            ctx = fr["ctx"]
            hold = {k: bool(v) for k, v in ctx.ref_hold.items()
                    if k.endswith("_hand") and v}
            chk = deciding_check(p, t)
            corridor = chk.get("corridor") or {}
            a = adopted.get((p, t))
            entry = {
                "pass": p, "t": t,
                "ref_hold": sorted(hold.keys()) if hold else [],
                "hand_prev_m": {"l": chk.get("hand_l_prev_m"),
                                "r": chk.get("hand_r_prev_m")},
                "hand_cand_m": {"l": chk.get("hand_l_cand_m"),
                                "r": chk.get("hand_r_cand_m")},
                "corridor": corridor,
                "adopted": (None if a is None else {
                    "status": a["status"], "slack": a["slack"],
                    "qp_calls": a["qp_calls"], "iterations": a["iterations"]}),
                "check_ok": chk.get("ok"), "check_reason": chk.get("reason"),
            }
            table.append(entry)
            hold_sides = {"left_hand": "l", "right_hand": "r"}
            for name in hold:
                s = hold_sides[name]
                if s not in corridor:
                    r2_ev.append({"pass": p, "t": t, "side": s,
                                  "hand_cand_m": entry["hand_cand_m"].get(s)})
                else:
                    c = corridor[s]
                    if float(c["dist_m"]) > float(c["upper_m"]) + 1e-9 and a is not None:
                        r1_ev.append({"pass": p, "t": t, "side": s,
                                      "dist_m": c["dist_m"],
                                      "upper_m": c["upper_m"],
                                      "adopted": entry["adopted"]})
        # zero-QP counterexample: fresh solver on the frozen frame context
        zero_qp = []
        for t in zero_qp_frames.get(demo, []):
            fr = next((f for f in col.frames if f["t"] == t), None)
            if fr is None:
                continue
            s2 = fresh_solver(pipe)
            z_pred = np.array(fr["z_pred"], copy=True)
            r = s2.solve_frame(z_pred, fr["ctx"])
            zero_qp.append({
                "t": t, "pass": fr["pass_index"],
                "accepted": bool(r.accepted), "status": r.status,
                "qp_calls": int(r.qp_calls), "iterations": int(r.iterations),
                "slack_used": bool(r.slack_used),
                "max_abs_state_change": float(np.abs(r.z - z_pred).max()),
                "ref_hold": sorted(k for k, v in fr["ctx"].ref_hold.items() if v),
                "prev_hand_dist": {k: float(v) for k, v in
                                   fr["ctx"].prev_hand_dist.items()},
            })
        # R2 across EVERY entry (plan: any check -- skip, attempts,
        # backtracks -- that faced a ref-hold side without its active row)
        r2_any = []
        for (p, t), fr in sorted(frames_by_key.items()):
            hold_sides = {"left_hand": "l", "right_hand": "r"}
            held = [hold_sides[k] for k, v in fr["ctx"].ref_hold.items()
                    if k.endswith("_hand") and v]
            if not held:
                continue
            for r in checks_at.get((p, t), []):
                cor = r.get("corridor") or {}
                for s in held:
                    if s not in cor:
                        r2_any.append({"pass": p, "t": t, "side": s,
                                       "caller": r.get("caller"),
                                       "ok": r.get("ok")})
        rec = {"demo": demo, "status": res.status, "fps": fps,
               "wall_s": round(wall, 1),
               "r2_evidence": r2_ev, "r2_count": len(r2_ev),
               "r2_any_entry": r2_any, "r2_any_count": len(r2_any),
               "r1_evidence": r1_ev, "r1_count": len(r1_ev),
               "zero_qp_counterexamples": zero_qp,
               "frames": table}
        path = out / f"dr_{demo}.json"
        path.write_text(json.dumps(rec, indent=1, default=float))
        print(f"[dr] {demo}: {res.status} r2_decide={len(r2_ev)} "
              f"r2_any={len(r2_any)} r1={len(r1_ev)} "
              f"zero_qp={len(zero_qp)} -> {path.name}")


# ---------------------------------------------------------------- dq ----------

def cmd_dq(out: Path) -> None:
    from segment_quality_report import quality_for

    sources = [
        ("sixseg_rq", PROJECT_ROOT / "derived/data/teachers_regenerated/"
                                     "labelq_20260926_sixseg/rq"),
        ("train24_keep", PROJECT_ROOT / "derived/data/teachers_regenerated/"
                                        "labelq_20260926_train24/regen_final"),
    ]
    rec = {"purpose": "D-Q frozen quality baseline (plan section 3)",
           "label_protocol": "graded_labels_v1",
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "sources": [], "segments": []}
    for name, d in sources:
        npzs = sorted(d.glob("*.npz"))
        rec["sources"].append({"name": name, "dir": str(d),
                               "n_npz": len(npzs)})
        for npz in npzs:
            demo = npz.stem
            q = quality_for(npz, demo)
            seg = {"source": name, "demo": demo, "npz_sha256": sha256(npz),
                   "input_sha256": sha256(TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
                   "T": q["T"], "validator_passed": q["validator_passed"],
                   "task_mpjpe": q["task_mpjpe"],
                   "windows_60_30": (q["T"] - 60) // 30 + 1 if q["T"] >= 60 else 0,
                   "hands": q["hands"],
                   "sides": {s: {k: m[k] for k in (
                       "ref_frames", "certified_frames", "certified_in_ref_frames",
                       "intersection_intervals",
                       "intersection_longest_run_frames",
                       "frames_within_enter", "proximity_recall",
                       "longest_detachment_run_frames",
                       "max_detachment_ref_m", "enter_threshold_crossings")}
                       for s, m in q["sides"].items()}}
            rec["segments"].append(seg)
            print(f"[dq] {name}/{demo}: inter l/r="
                  f"{seg['sides']['l']['certified_in_ref_frames']}/"
                  f"{seg['sides']['r']['certified_in_ref_frames']} within="
                  f"{seg['sides']['l']['frames_within_enter']}/"
                  f"{seg['sides']['r']['frames_within_enter']}")
    path = out / "dq_quality_baseline.json"
    path.write_text(json.dumps(rec, indent=1, default=float))
    print(f"[dq] baseline ({len(rec['segments'])} segments) -> {path.name}")


# ---------------------------------------------------------------- main --------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("dv", "df", "dr", "dq"))
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--out", default=str(DIAG_ROOT))
    ap.add_argument("--zero-qp-frames", nargs="*", default=None,
                    help="dr: demo=frame list, e.g. sub10_largebox_001=30,31,32")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "dv":
        cmd_dv(args.demos or list(DV_EXPECTED), out)
    elif args.command == "df":
        cmd_df(args.demos or [
            "sub12_largebox_004", "sub3_largebox_001", "sub8_largebox_002",
            "sub16_largebox_012", "sub7_largebox_029", "sub10_largebox_075"], out)
    elif args.command == "dr":
        frames = {}
        for item in (args.zero_qp_frames
                     or ["sub10_largebox_001=30,31,32", "sub10_largebox_003="]):
            demo, _, fl = item.partition("=")
            frames[demo] = [int(x) for x in fl.split(",") if x]
        cmd_dr(args.demos or ["sub10_largebox_001", "sub10_largebox_003"],
               out, frames)
    else:
        cmd_dq(out)


if __name__ == "__main__":
    main()
