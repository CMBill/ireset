"""Per-segment quality report for a repaired/regenerated teacher clip.

Plan section 6: a kept segment must be reported beyond "keep = True" --
per side, inside the reference grasp window AND over the full clip:

- box-frame hand site speed (mean/p50/p95/max, m/s),
- anchor drift inside the certified intervals (validator schedule),
- contact gaps (min hand-geom/box distance) and the missing-contact frames
  inside the reference grasp window,
- task MPJPE p50/p95/max and its growth (first vs last quintile).

Reusable for the frozen v3 artifacts and any new run: the input is a teacher
NPZ (same layout as the regenerated outputs) plus the source clip for the
reference schedule. The independent validator is the arbiter; this script
only quantifies what ``keep`` hides.

Run: uv run --locked python scripts/segment_quality_report.py \
        --npz derived/.../sub10_largebox_000.npz --demo sub10_largebox_000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from ireset.embodied.contact_sites import infer_reference_schedule
from ireset.data.teacher import load_teacher_npz
from ireset.paths import PROJECT_ROOT, TEACHER_TRAJ_DIR
from ireset.repair.config_io import MODES, load_mode_configs
from ireset.repair.constraints import FrameEvaluator
from ireset.repair.context import ReferenceCondition, RepairConfig
from ireset.repair.validation import IndependentValidator
from regenerate_teachers import hand_recall, task_mpjpe_series, task_mpjpe_stats
from ireset.repair.pipeline import KPT_IDX


def _series_stats(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "p50": round(float(np.percentile(v, 50)), 4),
            "p95": round(float(np.percentile(v, 95)), 4),
            "max": round(float(v.max()), 4)}


def side_metrics(vcfg, traj: np.ndarray, cond: ReferenceCondition,
                 validator: IndependentValidator, side: str) -> dict:
    """Speed / drift / gap metrics for one hand side."""
    T = traj.shape[0]
    cache = validator._cache
    local = cache["hand_local"][side]                  # [T, 3, 3] box frame
    dist = cache["hand_dist"][side]                    # [T]
    dt = 1.0 / vcfg.fps
    speed = np.zeros(T)
    if T > 1:
        speed[1:] = np.linalg.norm(np.diff(local, axis=0), axis=2).max(1) / dt

    ref = np.zeros(T, dtype=bool)
    eff = "left_hand" if side == "l" else "right_hand"
    for iv in cond.ref_schedule.intervals:
        if iv.effector == eff:
            ref[iv.start:iv.end] = True
    ivs = [(iv.start, iv.end) for iv in validator.validate(
        traj, exclude=("box_ground",)).schedule.intervals
        if iv.effector == eff and iv.support == "object"]
    cert = np.zeros(T, dtype=bool)
    drift = []
    for a, b in ivs:
        cert[a:b] = True
        anchor = local[a]
        drift.append(float(np.linalg.norm(local[a:b] - anchor, axis=-1).max()))
    gap_ref = dist[ref] if ref.any() else np.array([])
    enter = float(vcfg.hand_enter_m)
    within = (gap_ref <= enter) if gap_ref.size else np.array([], dtype=bool)
    return {
        "ref_frames": int(ref.sum()),
        "certified_frames": int(cert.sum()),
        "certified_in_ref_frames": int((cert & ref).sum()),
        "intervals": [[int(a), int(b)] for a, b in ivs],
        "speed_ref_mps": _series_stats(speed[ref]) if ref.any() else {"n": 0},
        "speed_all_mps": _series_stats(speed),
        "anchor_drift_certified_m": (
            {"n": len(drift), "max": round(max(drift), 4)} if drift else {"n": 0}),
        "gap_ref_mm": (
            {"n": int(gap_ref.size), "min": round(float(gap_ref.min()) * 1e3, 3),
             "p50": round(float(np.percentile(gap_ref, 50)) * 1e3, 3),
             "p90": round(float(np.percentile(gap_ref, 90)) * 1e3, 3),
             "p95": round(float(np.percentile(gap_ref, 95)) * 1e3, 3),
             "p99": round(float(np.percentile(gap_ref, 99)) * 1e3, 3),
             "max": round(float(gap_ref.max()) * 1e3, 3)} if gap_ref.size else {"n": 0}),
        # proximity-coverage robustness: the keep criterion counts frames
        # inside the enter threshold, so report the counts at the threshold
        # and in a small band above it, plus the number of threshold
        # crossings -- a coverage dip with near-identical percentiles and
        # many crossings is inference chatter, not lost proximity
        "enter_mm": round(enter * 1e3, 3),
        "frames_within_enter": int(within.sum()) if within.size else 0,
        "frames_within_enter_p2mm": int(
            (gap_ref <= enter + 0.002).sum()) if gap_ref.size else 0,
        "frames_within_enter_p5mm": int(
            (gap_ref <= enter + 0.005).sum()) if gap_ref.size else 0,
        "enter_threshold_crossings": int(
            np.count_nonzero(np.diff(within.astype(int)))) if within.size else 0,
        "missing_contact_ref_frames": int(
            (~within).sum()) if gap_ref.size else 0,
        "gap_all_mm": {"min": round(float(dist.min()) * 1e3, 3),
                       "p95": round(float(np.percentile(dist, 95)) * 1e3, 3)},
    }


def mpjpe_growth(ev, traj: np.ndarray, cond: ReferenceCondition) -> dict:
    """Growth of the task MPJPE over the clip, from the SHARED per-frame series.

    R4: the evaluator must be forwarded per frame -- reading
    ``site_positions`` in a loop without forwarding reports the LAST state's
    error for every t (sub10 reported growth -0.2242 m instead of -0.0030 m).
    Definition unchanged (world frame, the same keypoint map and units); the
    first/last quintile sampling is the historical one, and a clip shorter
    than 5 frames has ``k = 1`` (growth of a single frame is 0 by definition).
    """
    vals = task_mpjpe_series(ev, traj, cond)
    n = vals.size
    k = max(1, n // 5)
    return {"p50": round(float(np.percentile(vals, 50)), 4),
            "p95": round(float(np.percentile(vals, 95)), 4),
            "max": round(float(vals.max()), 4),
            "q1_mean": round(float(vals[:k].mean()), 4),
            "q5_mean": round(float(vals[-k:].mean()), 4),
            "growth": round(float(vals[-k:].mean() - vals[:k].mean()), 4)}


def quality_for(npz: str | Path, demo: str,
                validation_mode: str = "strict") -> dict:
    """Full per-segment quality record (validator + sides + task MPJPE)."""
    # load_teacher_npz accepts the teacher (quaternion-first) layout and
    # returns the internal scene [pos, quat, joints, box pos, box quat]
    traj, human, fps = load_teacher_npz(Path(npz))

    src_scene, _human_src, _fps = load_teacher_npz(
        TEACHER_TRAJ_DIR / f"{demo}_original.npz")
    cond = ReferenceCondition(human, src_scene[:, 36:43],
                              infer_reference_schedule(human, src_scene[:, 36:43]))
    rcfg, vcfg = load_mode_configs(validation_mode)
    validator = IndependentValidator(cfg=vcfg)
    # task MPJPE needs the robot evaluator on the LOCKED scene model
    ev = FrameEvaluator(validator.spec, validator.pairs, rcfg)
    rep = validator.validate(traj, exclude=("box_ground",))

    return {
        "demo": demo,
        "npz": str(Path(npz).resolve()),
        "T": int(traj.shape[0]),
        "validation_mode": validation_mode,
        "validator_passed": bool(rep.passed),
        "validator_failed": {k: {"location": c.location,
                                 "first_failure_frame": c.first_failure_frame}
                             for k, c in rep.checks.items() if not c.passed},
        "max_check_interval_m": rep.max_check_interval_m,
        "hand": hand_recall(validator, traj, cond),
        "task_mpjpe": task_mpjpe_stats(ev, traj, cond),
        "mpjpe_growth": mpjpe_growth(ev, traj, cond),
        "sides": {s: side_metrics(vcfg, traj, cond, validator, s)
                  for s in ("l", "r")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--demo", type=str, required=True,
                    help="source demo id (reference schedule + validator args)")
    ap.add_argument("--validation-mode", choices=MODES, default="strict")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    out = quality_for(args.npz, args.demo, args.validation_mode)
    text = Path(args.output) if args.output else None
    if text:
        text.write_text(json.dumps(out, indent=1, default=float))
        print(f"-> {text}")
    for s in ("l", "r"):
        m = out["sides"][s]
        print(f"[{s}] ref={m['ref_frames']} cert={m['certified_frames']} "
              f"(in-ref {m['certified_in_ref_frames']}) "
              f"speed_ref p95={m['speed_ref_mps'].get('p95')} "
              f"drift_max={m['anchor_drift_certified_m']} "
              f"gap_ref {m['gap_ref_mm']}")
    print(f"validator_passed={out['validator_passed']} "
          f"task={out['task_mpjpe']} growth={out['mpjpe_growth']['growth']}")


if __name__ == "__main__":
    main()
