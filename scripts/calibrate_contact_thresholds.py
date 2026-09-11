"""Calibrate reference-side contact thresholds against robot-side inference.

The robot-side schedule comes from generated geometry (MuJoCo distances,
sole heights) and is the grounded scheduler (plan 10.1 uses it to score
generated trajectories). The reference-side scheduler must agree with it on
teacher data, so its enter/exit thresholds are swept for macro-F1 across
the four effectors, and the best values are locked.

Writes derived/data/validation/contact_threshold_calibration.json.

Run: uv run --locked python scripts/calibrate_contact_thresholds.py [--n 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from audit_data import convert_scene_qpos
from ireset.embodied.contact_sites import infer_reference_schedule
from ireset.eval.metrics import contact_f1_all
from ireset.paths import TEACHER_TRAJ_DIR
from ireset.repair.validation import IndependentValidator, ValidationConfig

OUT = Path(__file__).resolve().parents[1] / "derived" / "data" / "validation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="original demos per subject")
    args = ap.parse_args()

    validator = IndependentValidator(cfg=ValidationConfig())
    by_subj: dict[str, list[str]] = defaultdict(list)
    for p in sorted(TEACHER_TRAJ_DIR.glob("*_original.npz")):
        by_subj[p.name.split("_")[0]].append(p.name[: -len("_original.npz")])
    sample = [d for _, demos in sorted(by_subj.items()) for d in demos[: args.n]]

    # precompute robot schedules once
    clips = []
    for demo in sample:
        npz = np.load(TEACHER_TRAJ_DIR / f"{demo}_original.npz", allow_pickle=True)
        scene = np.stack([convert_scene_qpos(r) for r in npz["qpos"]])
        rep = validator.validate(scene)
        clips.append({
            "demo": demo,
            "human": npz["human_joints"].astype(np.float64),
            "box_pose": scene[:, 36:43],
            "robot_schedule": rep.schedule,
            "T": scene.shape[0],
        })
    print(f"robot schedules computed for {len(clips)} clips")

    hand_grid = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
    foot_grid = [0.05, 0.06, 0.08, 0.10, 0.12]
    results = []
    for h_enter in hand_grid:
        for f_enter in foot_grid:
            f1s = []
            for c in clips:
                ref = infer_reference_schedule(
                    c["human"], c["box_pose"],
                    hand_enter_m=h_enter, hand_exit_m=h_enter * 1.8,
                    foot_enter_m=f_enter, foot_exit_m=f_enter * 1.6,
                )
                f1s.append(contact_f1_all(c["robot_schedule"], ref, c["T"])["macro_f1"])
            results.append({
                "hand_enter_m": h_enter, "hand_exit_m": round(h_enter * 1.8, 3),
                "foot_enter_m": f_enter, "foot_exit_m": round(f_enter * 1.6, 3),
                "macro_f1_mean": float(np.mean(f1s)),
            })
    results.sort(key=lambda r: -r["macro_f1_mean"])
    best = results[0]
    print(json.dumps(results[:8], indent=2))
    print("BEST:", json.dumps(best))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contact_threshold_calibration.json").write_text(
        json.dumps({"n_clips": len(clips), "grid": results, "best": best}, indent=2)
    )


if __name__ == "__main__":
    main()
