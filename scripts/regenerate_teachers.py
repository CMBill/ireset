"""Teacher data regeneration (Path B; p0_relaxed_validation_plan section 7,
impl plan 9.1: "用带交互模型约束的修复器/优化器重生成教师轨迹,筛选通过
严格验收的作为新教师").

Method: the OLD teacher trajectory is the warm start (prediction); the
repairer runs in ``regen_mode`` -- during reference grasp intervals
(inferred from the reference HUMAN keypoints, hull metric) the hand-box
pairs carry an ACTIVE upper bound, pulling the hovering teacher onto the
box. Feet keep the standard per-pair + quasi-static semantics (P0-fix
strict 6/6).

KEEP criterion (geometric screening + strict validation, NOT interval
certification): the independent validator (strict by default) must pass
AND, per side with at least ``--min-ref-frames`` reference grasp frames,
the fraction of those frames whose closest hand geom is within the
validator's ``hand_enter_m`` must reach ``--min-hand-recall``. A sample
with no screened side is reported as ``no_reference_grasp`` and not kept.
Certified-interval coverage (quasi-static entry gates) is REPORTED per
side (count, frames, longest continuous interval) but is not a keep
criterion -- zero coverage means "geometric proximity only", which the
report marks explicitly (m2_m3_stage_gate_fix_plan sections 3/6).

Outputs are NEW files only (source data under resources/ is read-only):
``<output-dir>/<demo>.npz`` in the source teacher layout (qpos
quaternion-first, human_joints, fps -- byte-compatible with
``ireset.data.teacher.load_teacher_npz``) plus a provenance report JSON
(source list, git rev, effective config, per-demo screening rows with
cost/timing breakdown and coverage evidence).

Run: uv run --locked python scripts/regenerate_teachers.py --set diagnosis6 \
        --validation-mode strict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from closed_loop_metrics import git_rev
from experiment_provenance import provenance as provenance_snapshot
from ireset.embodied.contact_sites import infer_reference_schedule
from ireset.paths import DERIVED_DATA_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR
from ireset.repair.config_io import MODES, load_mode_configs
from ireset.repair.context import ReferenceCondition
from ireset.repair.pipeline import KPT_IDX, RepairPipeline
from ireset.repair.validation import IndependentValidator
from ireset.data.teacher import load_teacher_npz

SCREEN = DERIVED_DATA_DIR / "screening" / "velocity_screen.json"
SPLIT = DERIVED_DATA_DIR / "split"


def registered_set(name: str) -> list[str]:
    if name == "diagnosis6":
        return json.loads(SCREEN.read_text())["registered"]["p0_diagnosis_6"]
    if name == "calibration24":
        return json.loads(SCREEN.read_text())["registered"]["p02_calibration"]
    if name == "calibration24_train_v1":
        return json.loads(
            (SCREEN.parent / "calibration24_train_v1.json").read_text())["demos"]
    demos = json.loads((SPLIT / f"{name}.json").read_text())
    return demos["demos"] if isinstance(demos, dict) else demos


def to_teacher_layout(scene: np.ndarray) -> np.ndarray:
    """Project scene [pos, quat wxyz] -> teacher NPZ [quat, pos] rows."""
    out = np.zeros_like(scene)
    out[:, 0:4] = scene[:, 3:7]
    out[:, 4:7] = scene[:, 0:3]
    out[:, 7:36] = scene[:, 7:36]
    out[:, 36:40] = scene[:, 39:43]
    out[:, 40:43] = scene[:, 36:39]
    return out


def hand_recall(validator: IndependentValidator, traj: np.ndarray,
                cond: ReferenceCondition) -> dict:
    """Per-side recall of reference grasp frames by inferred robot hand
    intervals, plus the validator's own interval stats.

    Two coverage notions are reported separately (plan section 6):
    ``contact_recall`` (KEEP criterion: physical proximity within the
    validator's enter threshold) and ``interval_recall`` (certification:
    quasi-static entry gates; lags contact by construction).
    """
    T = traj.shape[0]
    rep = validator.validate(traj, exclude=("box_ground",))
    out = {"intervals": rep.schedule.stats.get("hand_intervals", {}),
           "validator_passed": bool(rep.passed),
           "failed": {k: c.location for k, c in rep.checks.items() if not c.passed}}
    for side, eff in (("left", "left_hand"), ("right", "right_hand")):
        ref = np.zeros(T, dtype=bool)
        for iv in cond.ref_schedule.intervals:
            if iv.effector == eff:
                ref[iv.start : iv.end] = True
        rob = rep.schedule.mask(eff, T)
        n_ref = int(ref.sum())
        out[f"{side}_hand_interval_recall"] = (
            float((ref & rob).sum() / n_ref) if n_ref else None)
        out[f"{side}_hand_ref_frames"] = n_ref
        # certified intervals of this side: count, frames, longest span and
        # its reference-grasp overlap (a long interval outside the
        # reference grasp window is not carry evidence)
        ivs = [(iv.start, iv.end) for iv in rep.schedule.intervals
               if iv.effector == eff and iv.support == "object"]
        out[f"{side}_hand_certified_frames"] = int(sum(b - a for a, b in ivs))
        if ivs:
            a, b = max(ivs, key=lambda ab: ab[1] - ab[0])
            out[f"{side}_hand_longest_interval"] = {
                "start": int(a), "end": int(b), "frames": int(b - a),
                "ref_frames_in_span": int(ref[a:b].sum())}
            out[f"{side}_hand_interval_bounds"] = [[int(a), int(b)] for a, b in ivs]
        else:
            out[f"{side}_hand_longest_interval"] = None
            out[f"{side}_hand_interval_bounds"] = []
        if n_ref:
            # contact recall (KEEP criterion): physical grasp presence --
            # closest hand geom within the enter threshold; interval recall
            # (report-only) depends on the quasi-static entry gates and
            # lags contact by construction
            import mujoco

            d = mujoco.MjData(validator.model)
            close = 0
            for t in np.nonzero(ref)[0]:
                d.qpos = traj[t]
                mujoco.mj_forward(validator.model, d)
                dist = min(mujoco.mj_geomDistance(
                    validator.model, d, g, validator.box_geom, 1.0, np.zeros(6))
                    for g in validator._hand_geoms[side[0]])
                close += dist < validator.cfg.hand_enter_m
            out[f"{side}_hand_contact_recall"] = float(close / n_ref)
        else:
            out[f"{side}_hand_contact_recall"] = None
    return out


def task_mpjpe_stats(ev, traj: np.ndarray, cond: ReferenceCondition) -> dict:
    vals = []
    for t in range(traj.shape[0]):
        ev.forward(traj[t, 0:36], traj[t, 36:43])
        kpt_w = ev.site_positions(ev.kpt_ids)
        vals.append(float(np.linalg.norm(kpt_w - cond.human_kpts_world[t][KPT_IDX],
                                         axis=-1).mean()))
    v = np.array(vals)
    return {"p50": round(float(np.percentile(v, 50)), 4),
            "p95": round(float(np.percentile(v, 95)), 4),
            "max": round(float(v.max()), 4)}


def print_frame_log(pipe: RepairPipeline) -> None:
    """--verbose: one compact line per frame (last check call = the decided
    state) plus every failing endpoint/probe record with violating rows."""
    last_check: dict[int, dict] = {}
    failing: list[dict] = []
    for rec in pipe.frame_log:
        if rec["kind"] == "check":
            last_check[rec["t"]] = rec
            if rec.get("rows"):
                failing.append(rec)
        elif rec.get("viol", 0.0) > 1e-6:
            failing.append(rec)
    for t in sorted(last_check):
        r = last_check[t]
        corr = r.get("corridor", {})
        cl = corr.get("l")
        cr = corr.get("r")
        print(f"  [t={t:3d}] viol={r['viol']:.5f} "
              f"handL {r['hand_l_prev_m']*1e3:8.2f}->{r['hand_l_cand_m']*1e3:8.2f}mm "
              f"handR {r['hand_r_prev_m']*1e3:8.2f}->{r['hand_r_cand_m']*1e3:8.2f}mm "
              f"corridor L={('%.2f/%.2f' % (cl['dist_m']*1e3, cl['upper_m']*1e3)) if cl else '-'} "
              f"R={('%.2f/%.2f' % (cr['dist_m']*1e3, cr['upper_m']*1e3)) if cr else '-'} "
              f"margin={r['band_margin_m']*1e3:.2f}mm")
    for r in failing:
        tag = "probe" if r["kind"] == "probe" else "ENDPOINT"
        loc = (f" i={r['i']}/{r['n_sub']}" if r["kind"] == "probe" else "")
        print(f"  !! [t={r['t']:3d}]{tag}{loc} viol={r['viol']:.5f}")
        for row in r.get("rows", []):
            print(f"       {row['pair']}: {row['val_m']*1e3:.3f}mm "
                  f"in [{row['lo_m']*1e3:.3f}, {row['hi_m']*1e3:.3f}]mm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["diagnosis6", "calibration24",
                                      "calibration24_train_v1",
                                      "train", "val", "test"],
                    default="diagnosis6")
    ap.add_argument("--demos", nargs="*", default=None, help="explicit demo list override")
    ap.add_argument("--frames", type=int, default=0, help="0 = full clip")
    ap.add_argument("--validation-mode", choices=MODES, default="strict")
    ap.add_argument("--mode", choices=("regen", "standard"), default="regen",
                    help="regen = Path B regeneration (hand-contact objective, "
                         "strict screening); standard = the same six-segment "
                         "measurement with regen_mode OFF (regression check "
                         "for the stage gate; no NPZ is written)")
    ap.add_argument("--task-gate-yaml", type=str, default=None)
    ap.add_argument("--min-hand-recall", type=float, default=0.5,
                    help="screening threshold per grasping side")
    ap.add_argument("--min-ref-frames", type=int, default=10,
                    help="sides with fewer reference grasp frames are not screened")
    ap.add_argument("--output-dir", type=str,
                    default=str(DERIVED_DATA_DIR / "teachers_regenerated"))
    ap.add_argument("--report", type=str, default=None)
    ap.add_argument("--note", type=str, default=None,
                    help="free-form run note stored in the report")
    ap.add_argument("--stab-weight", type=float, default=None,
                    help="grasp-stability soft-target weight (F4 section 7.2); "
                         "pre-registered comparison: 0 / 0.1 / 1.0. Default: "
                         "the config value (0.0)")
    ap.add_argument("--verbose", action="store_true",
                    help="per-frame hand_dist / corridor_upper / probe details "
                         "(printed summary + full frame_log in the report rows)")
    args = ap.parse_args()

    demos = args.demos if args.demos else registered_set(args.set)
    rcfg, vcfg = load_mode_configs(args.validation_mode)
    if args.task_gate_yaml:
        import yaml as _yaml
        rcfg.task_mpjpe_tol_m = float(
            _yaml.safe_load(Path(args.task_gate_yaml).read_text())
            ["task_quality_gates"]["mpjpe_tol_m"])
    # regen objective tracks the REFERENCE (mesh/keypoints + temporal +
    # regularization), not the old teacher's joints: the teacher's hand
    # drifts in box frame (p90 18cm) and a prediction-tracking objective
    # drags the regenerated hand at 0.1-0.4 m/s -- the quasi-static entry
    # gate (0.02) can then never certify anything.
    # ``--mode standard`` keeps the identical measurement path with the
    # standard objective (regression check for the stage gate; report-only,
    # no NPZ output -- this is not a regeneration run).
    regen_cfg = replace(rcfg, regen_mode=(args.mode == "regen"))
    if args.stab_weight is not None:
        regen_cfg = replace(regen_cfg, regen_hand_stab_weight=args.stab_weight)
    pipe = RepairPipeline(cfg=regen_cfg)
    pipe.validator.cfg = vcfg
    pipe.verbose = args.verbose
    validator = IndependentValidator(cfg=vcfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_keep = 0
    for demo in demos:
        src = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
        scene, human, fps = load_teacher_npz(src)
        if args.frames:
            scene, human = scene[: args.frames], human[: args.frames]
        cond = ReferenceCondition(human, scene[:, 36:43],
                                  infer_reference_schedule(human, scene[:, 36:43]))
        t0 = time.perf_counter()
        res = pipe.repair_trajectory(scene, cond)
        wall = time.perf_counter() - t0
        # stage 0 (2026-09-16): the old ms_per_frame divided the WHOLE wall
        # time by the FULL clip length -- an early rejection amortized its
        # per-frame cost over frames never processed. Report per PROCESSED
        # frame (pipeline's own timings_s includes every attempt), the raw
        # length-normalized value, and both total timings
        frame_stats = res.diagnostics.get("timings_s", [])
        n_processed = max(len(frame_stats), res.diagnostics.get("t_fail", -1) + 1
                          if res.diagnostics.get("t_fail") is not None else len(frame_stats))
        ms_frame_stats = np.asarray(frame_stats, dtype=float) * 1e3
        row = {"demo": demo, "T": int(scene.shape[0]), "fps": fps,
               "accepted": bool(res.accepted), "status": res.status,
               "wall_s": round(wall, 2),
               "n_frames_processed": int(n_processed),
               "ms_per_processed_frame": (
                   round(float(ms_frame_stats.mean()), 1) if ms_frame_stats.size else None),
               "ms_per_processed_frame_p50": (
                   round(float(np.percentile(ms_frame_stats, 50)), 1)
                   if ms_frame_stats.size else None),
               "ms_per_processed_frame_p95": (
                   round(float(np.percentile(ms_frame_stats, 95)), 1)
                   if ms_frame_stats.size else None),
               "ms_per_processed_frame_max": (
                   round(float(ms_frame_stats.max()), 1) if ms_frame_stats.size else None),
               "solve_frames_s": round(float(sum(frame_stats)), 3),
               "final_validation_s": res.diagnostics.get("final_validation_s"),
               "total_s": res.diagnostics.get("total_s"),
               # length-normalized legacy metric (kept for comparability; NOT
               # a per-frame cost for early-rejected segments)
               "ms_per_input_frame": round(wall * 1e3 / scene.shape[0], 1),
               "t_fail": res.diagnostics.get("t_fail"),
               "skip_reason": res.diagnostics.get("skip_reason"),
               "final_stage": res.diagnostics.get("final_stage"),
               "fallbacks": res.diagnostics.get("fallbacks"),
               "slack_frames": res.diagnostics.get("slack_frames"),
               # F0 aggregate over ALL attempts of this pass (a suffix retry
               # never overwrites the first pass; its costs live under
               # "first_attempt"): attempts/qp/iterations/slack/backtrack and
               # processed-frame visits vs unique frames vs input frames
               "cost": {k: res.diagnostics.get(k) for k in (
                   "attempts", "qp_calls_total", "iterations_total",
                   "slack_attempts", "backtracks", "processed_frame_visits",
                   "unique_processed_frames", "input_frames", "skipped_frames",
                   "suffix_retries", "trajectory_pass")},
               "screening_s": None}
        if not res.accepted:
            row["rejection"] = {
                "reason": res.diagnostics.get("reason"),
                "reason_source": res.diagnostics.get("reason_source"),
                "last_attempt": res.diagnostics.get("last_attempt"),
                "violated_rows": res.diagnostics.get("violated_rows"),
                "probe": res.diagnostics.get("probe"),
            }
        if "first_attempt" in res.diagnostics:
            fa = res.diagnostics["first_attempt"]
            row["suffix_retry"] = {
                "from_frame": res.diagnostics.get("suffix_retry_from"),
                "first_attempt_t_fail": fa.get("t_fail"),
                # the first pass was accepted (trajectory-level) but failed
                # the independent validator: this summary + failed_checks
                # describe that failure; the retry's own outcome is below
                "first_attempt_validator": res.diagnostics.get("first_attempt_final"),
                "first_attempt_failed_checks": res.diagnostics.get("failed_checks"),
                "retry_failed_checks": res.diagnostics.get("final_retry_failed_checks"),
                "first_attempt_solve_s": round(float(sum(fa.get("timings_s", []))), 3),
                "first_attempt_rejection": {
                    "reason": fa.get("reason"),
                    "final_stage": fa.get("final_stage"),
                    "violated_rows": fa.get("violated_rows"),
                    "probe": fa.get("probe")},
            }
        if args.verbose:
            print(f"[{demo}] frame log ({len(pipe.frame_log)} records):")
            print_frame_log(pipe)
            log_path = out_dir / f"frame_log_{demo}.json"
            log_path.write_text(json.dumps(pipe.frame_log, indent=1, default=float))
            row["frame_log"] = str(log_path)
        keep = False
        if res.accepted and res.trajectory is not None:
            traj = res.trajectory
            t_screen = time.perf_counter()
            cov = hand_recall(validator, traj, cond)
            row["screening_s"] = round(time.perf_counter() - t_screen, 3)
            row["hand"] = cov
            row["task_mpjpe"] = task_mpjpe_stats(pipe.ev, traj, cond)
            recalls = [cov[f"{s}_hand_contact_recall"] for s in ("left", "right")
                       if cov[f"{s}_hand_ref_frames"] >= args.min_ref_frames]
            screened = all(r is None or r >= args.min_hand_recall for r in recalls) \
                if recalls else False
            grasping = len(recalls) > 0
            row["hand_screen"] = ("pass" if screened and grasping else
                                  "no_reference_grasp" if not grasping else "fail")
            keep = cov["validator_passed"] and screened and grasping
            if keep and args.mode == "regen":
                np.savez(out_dir / f"{demo}.npz",
                         qpos=to_teacher_layout(traj),
                         human_joints=human, fps=fps)
                n_keep += 1
        rows.append(row)
        print(f"[{demo}] accepted={row['accepted']} keep={keep} "
              f"hand={row.get('hand', {}).get('intervals', '-')} "
              f"contact={row.get('hand', {}).get('left_hand_contact_recall', '-')}/"
              f"{row.get('hand', {}).get('right_hand_contact_recall', '-')} "
              f"cert={row.get('hand', {}).get('left_hand_interval_recall', '-')}/"
              f"{row.get('hand', {}).get('right_hand_interval_recall', '-')} "
              f"task_p95={row.get('task_mpjpe', {}).get('p95', '-')} "
              f"({row['ms_per_processed_frame']}ms/proc-f, "
              f"{row['n_frames_processed']}/{row['T']} frames)")

    reg_cfg_pub = {k: v for k, v in asdict(regen_cfg).items()
                   if k in ("regen_mode", "regen_hand_contact_gap_m",
                            "task_mpjpe_tol_m", "contact_band_gap_m",
                            "hand_anchor_tol_m", "scp_max_iters")}
    # code/input/asset provenance for this report (plan section 10): the
    # report must be reproducible without /tmp probes -- exact revision and
    # dirty-diff hash, input NPZ hashes, asset hashes, effective configs
    prov = provenance_snapshot(demos, note=f"{args.set}/{args.validation_mode}/{args.mode}")
    report = {
        "set": args.set, "n_demos": len(demos), "n_kept": n_keep,
        "mode": args.mode,
        "validation_mode": args.validation_mode,
        "min_hand_recall": args.min_hand_recall,
        "min_ref_frames": args.min_ref_frames,
        "regen_config": reg_cfg_pub,
        "source_dir": str(TEACHER_TRAJ_DIR),
        "git_rev": git_rev(),
        "notes": args.note,
        "provenance": {k: prov[k] for k in
                       ("git", "configs", "assets", "env", "uv_lock_sha256")},
        "inputs": prov.get("inputs"),
        "rows": rows,
    }
    report_path = Path(args.report) if args.report else (
        out_dir / f"{args.mode}_report_{args.set}_{args.validation_mode}.json")
    report_path.write_text(json.dumps(report, indent=1, default=float))
    print(f"kept {n_keep}/{len(demos)} -> {out_dir}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
