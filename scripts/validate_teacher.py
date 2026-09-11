"""Quantify full teacher trajectory quality with the independent validator.

M1 acceptance: "完整教师轨迹质量可量化". For each sampled clip:

- run IndependentValidator on the converted scene qpos (keyframes +
  interpolated midpoints, all collision pairs, contact inference);
- infer the *reference* schedule from the NPZ human_joints and the target
  box trajectory, and report robot-vs-reference contact F1;
- collect worst values per check.

Writes derived/data/validation/teacher_quality.json.

Run: uv run --locked python scripts/validate_teacher.py [--n 8] [--include-aug]
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
import torch

from audit_data import convert_scene_qpos
from ireset.embodied.contact_sites import infer_reference_schedule
from ireset.eval.metrics import contact_f1_all
from ireset.paths import OMOMO_NEW_DIR, TEACHER_TRAJ_DIR
from ireset.repair.validation import IndependentValidator, ValidationConfig

OUT = Path(__file__).resolve().parents[1] / "derived" / "data" / "validation"


def load_npz_pt(demo: str, variant: str):
    npz = np.load(TEACHER_TRAJ_DIR / f"{demo}_{variant}.npz", allow_pickle=True)
    scene = np.stack([convert_scene_qpos(r) for r in npz["qpos"]])
    human = npz["human_joints"].astype(np.float64)
    return scene, human


def target_box_pose(scene_qpos: np.ndarray) -> np.ndarray:
    """Target-scene box trajectory in project convention [T,7] (xyz+qwxyz).

    Scene qpos box slice is [x,y,z,qw,qx,qy,qz] already (MuJoCo layout).
    """
    return scene_qpos[:, 36:43]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="original demos per subject")
    ap.add_argument("--include-aug", action="store_true")
    args = ap.parse_args()

    validator = IndependentValidator(cfg=ValidationConfig())
    npz_names = sorted(p.name for p in TEACHER_TRAJ_DIR.glob("*.npz"))
    by_subj: dict[str, list[str]] = defaultdict(list)
    for n in npz_names:
        if n.endswith("_original.npz"):
            by_subj[n.split("_")[0]].append(n[: -len("_original.npz")])

    sample: list[tuple[str, str]] = []
    for subj, demos in sorted(by_subj.items()):
        for demo in demos[: args.n]:
            sample.append((demo, "original"))
            if args.include_aug:
                for prefix in ("rot", "trans"):
                    vs = sorted(x[len(demo) + 1 : -4] for x in npz_names if x.startswith(f"{demo}_{prefix}_"))
                    if vs:
                        sample.append((demo, vs[0]))

    rows = []
    for demo, variant in sample:
        scene, human = load_npz_pt(demo, variant)
        rep = validator.validate(scene)
        ref_sched = infer_reference_schedule(human, target_box_pose(scene))
        T = scene.shape[0]
        f1 = contact_f1_all(rep.schedule, ref_sched, T) if rep.schedule else {}
        rows.append({
            "clip": f"{demo}_{variant}",
            "passed": rep.passed,
            "summary": rep.summary(),
            "worst": {k: round(v.worst, 6) for k, v in rep.checks.items()},
            "limits": {k: v.limit for k, v in rep.checks.items()},
            "check_passed": {k: v.passed for k, v in rep.checks.items()},
            "max_check_interval_m": round(rep.max_check_interval_m, 4),
            "robot_intervals": [
                {"eff": iv.effector, "support": iv.support, "start": iv.start, "end": iv.end}
                for iv in (rep.schedule.intervals if rep.schedule else [])
            ],
            "reference_intervals": [
                {"eff": iv.effector, "support": iv.support, "start": iv.start, "end": iv.end}
                for iv in ref_sched.intervals
            ],
            "contact_f1": {k: round(v, 4) for k, v in f1.items()},
        })
        print(rows[-1]["clip"], rows[-1]["summary"],
              {k: round(v, 4) for k, v in rows[-1]["worst"].items() if k in ("penetration", "foot_support", "hand_grasp")})

    agg = {
        "n_clips": len(rows),
        "full_pass_rate": float(np.mean([r["passed"] for r in rows])),
        "per_check_pass": {
            k: float(np.mean([r["check_passed"].get(k, True) for r in rows])) for k in rows[0]["check_passed"]
        },
        "per_check_worst_max": {
            k: max(r["worst"].get(k, 0.0) for r in rows) for k in rows[0]["worst"]
        },
        "contact_f1_macro_mean": float(np.mean([r["contact_f1"].get("macro_f1", 0.0) for r in rows])),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    def _to_py(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(type(o))

    (OUT / "teacher_quality.json").write_text(
        json.dumps({"aggregate": agg, "clips": rows}, indent=2, ensure_ascii=False, default=_to_py)
    )
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
