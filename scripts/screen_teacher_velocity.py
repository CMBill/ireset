"""P0.0.1 teacher velocity screening + registered input sets (p0 plan section 3).

Scans every ``*_original.npz`` teacher trajectory for inter-frame velocity
spikes (root translation, robot joints) against the M2-calibrated bounds.
Spike files are excluded from P0 calibration inputs by a *pre-registered*
data-quality screen: the exclusion list (with reasons) is written next to the
registered sets so no input silently disappears from any denominator.

Outputs ``derived/data/screening/velocity_screen.json``.

Run: uv run --locked python scripts/screen_teacher_velocity.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.paths import DERIVED_DATA_DIR, TEACHER_TRAJ_DIR

# M2-calibrated velocity bounds (configs/repair/qp_default.yaml velocity_limits)
LIM_ROOT_MPS = 1.8
LIM_JOINT_RADPS = 12.0
FPS = 30.0

OUT = DERIVED_DATA_DIR / "screening"


def file_stats(path: Path) -> dict:
    # teacher NPZ qpos is quaternion-first: [qw,qx,qy,qz, x,y,z, 29 joints, box 7]
    q = np.load(path, allow_pickle=True)["qpos"]
    dpos = np.linalg.norm(np.diff(q[:, 4:7], axis=0), axis=1) * FPS
    dq = np.abs(np.diff(q[:, 7:36], axis=0)).max(1) * FPS
    bad = np.where((dpos > LIM_ROOT_MPS) | (dq > LIM_JOINT_RADPS))[0]
    return {
        "max_root_mps": round(float(dpos.max()), 3),
        "max_joint_radps": round(float(dq.max()), 3),
        "n_spike_frames": int(len(bad)),
        "first_spike_frame": int(bad[0]) if len(bad) else None,
    }


def main() -> None:
    files = sorted(TEACHER_TRAJ_DIR.glob("*_original.npz"))
    by_subj: dict[str, list[str]] = defaultdict(list)
    for p in files:
        by_subj[p.name.split("_")[0]].append(p.name[: -len("_original.npz")])

    per_file, excluded = {}, []
    for p in files:
        demo = p.name[: -len("_original.npz")]
        s = file_stats(p)
        per_file[demo] = s
        if s["n_spike_frames"]:
            excluded.append({
                "demo": demo,
                "reason": (f"velocity spike: root {s['max_root_mps']} m/s / joint "
                           f"{s['max_joint_radps']} rad/s exceeds "
                           f"{LIM_ROOT_MPS}/{LIM_JOINT_RADPS}; first at frame "
                           f"{s['first_spike_frame']}"),
            })

    # registered sets: P0 diagnosis = first original per subject (M2 P1 set);
    # P0.2 calibration = first 4 per subject, exclusions applied
    diag = [d for _, ds in sorted(by_subj.items()) for d in ds[:1]]
    calib_all = [d for _, ds in sorted(by_subj.items()) for d in ds[:4]]
    calib = [d for d in calib_all if d not in {e["demo"] for e in excluded}]

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "date": str(date.today()),
        "thresholds": {"root_mps": LIM_ROOT_MPS, "joint_radps": LIM_JOINT_RADPS, "fps": FPS},
        "n_files": len(files),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "registered": {
            "p0_diagnosis_6": diag,
            "p02_calibration_requested": len(calib_all),
            "p02_calibration": calib,
        },
        "per_file": per_file,
    }
    out_path = OUT / "velocity_screen.json"
    # a pre-registered frozen set (stage-gate fix plan section 8) is not a scan
    # result: carry it over verbatim so a re-scan cannot silently drop it
    if out_path.exists():
        old = json.loads(out_path.read_text())
        for k, v in old.get("registered", {}).items():
            if k.startswith("calibration24_train_v") and k not in report["registered"]:
                report["registered"][k] = v
    out_path.write_text(json.dumps(report, indent=1))
    print(f"{len(files)} originals, {len(excluded)} excluded -> {out_path}")
    print(f"diagnosis set ({len(diag)}): {diag}")
    print(f"calibration set ({len(calib)}/{len(calib_all)} after exclusions)")


if __name__ == "__main__":
    main()
