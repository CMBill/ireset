"""P0.2 task-gate calibration (p0 plan section 5; plan section 8.4).

Calibrates the *existing* frame-level task gate ``task_mpjpe_tol_m`` (M2
working value 0.40) from accepted teacher trajectories: the repair-induced
keypoint MPJPE distribution (repaired robot keypoints vs the reference
human keypoints, the exact metric the gate judges in
``RepairPipeline.frame_valid``). Only ``mpjpe_tol_m`` is calibrated -- E_L
and contact F1 stay report metrics per plan 10.1.

Input: trajectory NPZ dir written by ``eval_repair.py --save-dir`` (keys:
trajectory / prediction / human_kpts / box_pose). The accepted set carries
selection bias (relaxed-mode acceptance); the lock comment records it.

Run: uv run --locked python scripts/calibrate_task_gates.py \
        --trajectories derived/data/validation/p0_trajectories_relaxed
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import yaml

from ireset.embodied.contact_sites import SMPLH_INDEX
from ireset.embodied.g1_spec import KPT_TO_SMPLH
from ireset.paths import DERIVED_DATA_DIR, PROJECT_ROOT
from ireset.repair.pipeline import RepairPipeline

KPT_IDX = [SMPLH_INDEX[v] for v in KPT_TO_SMPLH.values()]
GATES_YAML = PROJECT_ROOT / "configs" / "eval" / "task_quality_gates.yaml"


def frame_mpjpe(ev, z: np.ndarray, box: np.ndarray, ref_kpts: np.ndarray) -> float:
    ev.forward(z, box)
    kpts = ev.site_positions(ev.kpt_ids)
    return float(np.linalg.norm(kpts - ref_kpts, axis=-1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=str, required=True)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--output", type=str, default=str(GATES_YAML))
    args = ap.parse_args()

    tdir = Path(args.trajectories)
    files = sorted(tdir.glob("*.npz"))
    if not files:
        sys.exit(f"no NPZ under {tdir}")

    probe = RepairPipeline()
    values, per_seg = [], {}
    for f in files:
        d = np.load(f)
        traj, box, human = d["trajectory"], d["box_pose"], d["human_kpts"]
        seg = [frame_mpjpe(probe.ev, traj[t, 0:36], box[t], human[t][KPT_IDX])
               for t in range(traj.shape[0])]
        values += seg
        per_seg[f.stem] = {"p50": round(float(np.percentile(seg, 50)), 4),
                           "p95": round(float(np.percentile(seg, 95)), 4),
                           "max": round(float(max(seg)), 4)}
    v = np.array(values)
    p95 = float(np.percentile(v, 95))
    tol = round(p95 + args.margin, 3)

    gates = {
        "task_quality_gates": {
            "mpjpe_tol_m": tol,
            "calibration": {
                "date": str(date.today()),
                "n_segments": len(files),
                "n_frames": int(len(v)),
                "repair_mpjpe_p50": round(float(np.percentile(v, 50)), 4),
                "repair_mpjpe_p90": round(float(np.percentile(v, 90)), 4),
                "repair_mpjpe_p95": round(p95, 4),
                "margin_m": args.margin,
                "bias": ("calibrated on relaxed-mode ACCEPTED teacher segments "
                         "-- selection bias toward repairable clips; values "
                         "underestimate the full-distribution tail"),
            },
        },
    }
    Path(args.output).write_text(yaml.safe_dump(gates, sort_keys=False))
    (DERIVED_DATA_DIR / "validation" / "task_gate_calibration.json").write_text(
        json.dumps({"per_segment": per_seg, "tol": tol, "p95": round(p95, 4)}, indent=1))
    print(f"{len(files)} segments / {len(v)} frames: p50={np.percentile(v, 50):.4f} "
          f"p90={np.percentile(v, 90):.4f} p95={p95:.4f} -> mpjpe_tol_m={tol}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
