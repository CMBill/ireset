"""Calibrate reference-side contact thresholds against robot-side inference.

Phase C of the M1 review fix plan (docs/plans/m1_fix_plan.md):

- hand grid extended down to 0.005 (the previous optimum sat on the old
  grid's lower boundary — a grid artifact);
- foot thresholds are NOT swept: reference foot height saturates (82% of
  frames below 0.02 m), so F1 cannot calibrate them; they are pinned by the
  acceptance invariant ``foot_enter <= foot_ground_gap`` and documented as
  such;
- objective uses the undefinedness-skipping macro-F1 (Fix 1) with the hull
  reference metric (Fix 3), and records ``n_defined`` plus the exit/enter
  ratio assumption.

Writes derived/data/validation/contact_threshold_calibration.json.

Run: uv run --locked python scripts/calibrate_contact_thresholds.py [--n 3]
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

# exit/enter ratios are an UNCALIBRATED assumption (recorded, not optimized)
HAND_EXIT_RATIO = 1.8
FOOT_ENTER_M = 0.005   # pinned by the acceptance invariant, not by F1
FOOT_EXIT_M = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="original demos per subject")
    args = ap.parse_args()

    validator = IndependentValidator(cfg=ValidationConfig())
    by_subj: dict[str, list[str]] = defaultdict(list)
    for p in sorted(TEACHER_TRAJ_DIR.glob("*_original.npz")):
        by_subj[p.name.split("_")[0]].append(p.name[: -len("_original.npz")])
    sample = [d for _, demos in sorted(by_subj.items()) for d in demos[: args.n]]

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
    print(f"robot schedules computed for {len(clips)} clips "
          f"(robot side: hand {validator.cfg.hand_enter_m}/{validator.cfg.hand_exit_m}, "
          f"foot {validator.cfg.foot_enter_m}/{validator.cfg.foot_exit_m})")

    hand_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.06]
    results = []
    for h_enter in hand_grid:
        f1_defined: list[float] = []
        n_defined_total = 0
        n_effectors = 0
        for c in clips:
            ref = infer_reference_schedule(
                c["human"], c["box_pose"],
                hand_enter_m=h_enter, hand_exit_m=round(h_enter * HAND_EXIT_RATIO, 4),
                foot_enter_m=FOOT_ENTER_M, foot_exit_m=FOOT_EXIT_M,
            )
            out = contact_f1_all(c["robot_schedule"], ref, c["T"])
            if out["macro_f1"] is not None:
                f1_defined.append(out["macro_f1"])
            n_defined_total += out["n_defined"]
            n_effectors += 4
        results.append({
            "hand_enter_m": h_enter,
            "hand_exit_m": round(h_enter * HAND_EXIT_RATIO, 4),
            "foot_enter_m": FOOT_ENTER_M,
            "foot_exit_m": FOOT_EXIT_M,
            "macro_f1_mean_defined": round(float(np.mean(f1_defined)), 4) if f1_defined else None,
            "n_clips_with_defined_macro": len(f1_defined),
            "n_defined_effectors": n_defined_total,
            "n_effectors_total": n_effectors,
        })
        print(results[-1])
    best = max(results, key=lambda r: r["macro_f1_mean_defined"] or -1)
    print("BEST:", json.dumps(best))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contact_threshold_calibration.json").write_text(json.dumps({
        "n_clips": len(clips),
        "objective": "skip-strategy macro-F1 (undefined double-empty effectors excluded), hull reference metric",
        "assumptions": {
            "hand_exit_over_enter": HAND_EXIT_RATIO,
            "foot_thresholds": "pinned by invariant foot_enter<=foot_ground_gap (F1 saturates: 82% of reference frames below 0.02m)",
        },
        "grid": results,
        "best": best,
    }, indent=2))


if __name__ == "__main__":
    main()
