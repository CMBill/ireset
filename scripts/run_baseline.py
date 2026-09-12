"""Run the pure-optimization baseline on teacher clips and quantify quality.

For each clip: track the reference human keypoints (NPZ ``human_joints``,
target-scene aligned for original variants) with the GN tracker, validate the
result independently, and report keypoint MPJPE + per-frame timing.

Run: uv run --locked python scripts/run_baseline.py [--n 1] [--frames 120]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from audit_data import convert_scene_qpos
from ireset.embodied.g1_spec import KPT_TO_SMPLH
from ireset.embodied.contact_sites import SMPLH_INDEX
from ireset.eval.baselines import InteractionTracker, TrackerConfig
from ireset.kinematics.mujoco_adapter import RobotState
from ireset.paths import TEACHER_TRAJ_DIR
from ireset.repair.validation import IndependentValidator, ValidationConfig

OUT = Path(__file__).resolve().parents[1] / "derived" / "data" / "validation"
KPT_IDX = [SMPLH_INDEX[v] for v in KPT_TO_SMPLH.values()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="clips")
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    npz_names = sorted(p.name for p in TEACHER_TRAJ_DIR.glob("*_original.npz"))[: args.n]
    validator = IndependentValidator(cfg=ValidationConfig())
    tracker = InteractionTracker(cfg=TrackerConfig())

    rows = []
    for name in npz_names:
        demo = name[: -len("_original.npz")]
        npz = np.load(TEACHER_TRAJ_DIR / name, allow_pickle=True)
        scene = np.stack([convert_scene_qpos(r) for r in npz["qpos"]])[: args.frames]
        human = npz["human_joints"].astype(np.float64)[: args.frames]

        t0 = time.perf_counter()
        result = tracker.track(human, scene[:, 36:43])
        dt_ms = (time.perf_counter() - t0) / scene.shape[0] * 1e3

        # first-frame MPJPE after solving frame 0 from the reference-estimated
        # init (review Fix 4): the seeding quality M2's fallback inherits
        init = tracker.initial_state(human[0])
        solved0 = tracker.solve_frame(init, human[0])
        first_err = float(np.linalg.norm(
            tracker.kin.keypoints(solved0)[0] - human[0][KPT_IDX], axis=-1).mean())
        init_err = float(np.linalg.norm(
            tracker.kin.keypoints(init)[0] - human[0][KPT_IDX], axis=-1).mean())

        # keypoint MPJPE against the reference (what the tracker sees)
        st = RobotState(result[:, 0:3], result[:, 3:7], result[:, 7:36])
        achieved = tracker.kin.keypoints(st)
        target = human[:, KPT_IDX]
        mpjpe = float(np.linalg.norm(achieved - target, axis=-1).mean())

        rep = validator.validate(result)
        rows.append({
            "clip": demo,
            "frames": int(scene.shape[0]),
            "mpjpe_m": round(mpjpe, 4),
            "frame0_solved_mpjpe_m": round(first_err, 4),
            "init_raw_mpjpe_m": round(init_err, 4),
            "ms_per_frame": round(dt_ms, 1),
            "passed": bool(rep.passed),
            "summary": rep.summary(),
            "worst": {k: round(v.worst, 5) for k, v in rep.checks.items()},
        })
        print(f"{demo}: mpjpe={mpjpe:.4f}m first={first_err:.4f}m {dt_ms:.0f}ms/frame {rep.summary()} "
              f"pen={rep.checks['penetration'].worst:.4f}")

    agg = {
        "n_clips": len(rows),
        "mpjpe_mean": float(np.mean([r["mpjpe_m"] for r in rows])),
        "frame0_solved_mpjpe_mean": float(np.mean([r["frame0_solved_mpjpe_m"] for r in rows])),
        "init_raw_mpjpe_mean": float(np.mean([r["init_raw_mpjpe_m"] for r in rows])),
        "ms_per_frame_mean": float(np.mean([r["ms_per_frame"] for r in rows])),
        "pass_rate": float(np.mean([r["passed"] for r in rows])),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline_quality.json").write_text(
        json.dumps({"aggregate": agg, "clips": rows}, indent=2, default=float)
    )
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
