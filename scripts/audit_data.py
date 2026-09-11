"""M0 data audit: OMOMO PT <-> OmniRetarget teacher NPZ pairing.

Verifies, per sampled clip, the full pairing chain claimed by the plan:

A. File inventory: demos per subject, PT files with/without teacher, variants.
B. NPZ schema: keys, shapes, fps=30, T matches the PT length.
C. Pose layouts: NPZ root/box stored [qw,qx,qy,qz,x,y,z] (unit quats); PT
   object cols [318:325] = [x,y,z,qx,qy,qz,qw] (upstream loader convention).
D. Scene relation: PT human -> NPZ human is an exact similarity transform
   (scale = robot/human height, rotation, translation); the same rotation maps
   PT object orientation onto the NPZ box orientation; the box position
   differs by a constant per-clip offset (z-dominant).
E. Teacher replay in the locked scene model: joint-limit violations, sole
   sphere heights, MuJoCo penetration depths, palm-to-box distances.

Writes derived/data/audit/m0_pairing_report.json and prints a summary.

Run: uv run --locked python scripts/audit_data.py [--per-subject 4] [--include-aug]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from ireset.embodied.g1_spec import G1Spec, HAND_SITES, SOLE_SITES
from ireset.paths import OMOMO_NEW_DIR, TEACHER_TRAJ_DIR

AUDIT_OUT_DIR = Path(__file__).resolve().parents[1] / "derived" / "data" / "audit"


def convert_scene_qpos(row: np.ndarray) -> np.ndarray:
    """Teacher NPZ row [43] ([quat,pos] both freejoints) -> MuJoCo [pos,quat]."""
    out = np.zeros(43)
    out[0:3] = row[4:7]
    out[3:7] = row[0:4]
    out[7:36] = row[7:36]
    out[36:39] = row[40:43]
    out[39:43] = row[36:40]
    return out


def fit_similarity(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Least-squares similarity A->B (uniform scale, rotation, translation)."""
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = float(np.sum(S * np.sign(np.linalg.det(Vt.T @ U.T))) / np.sum(Ac**2))
    t = cb - scale * R @ ca
    resid = float(np.abs(scale * (A @ R.T) + t - B).max())
    return scale, R, t, resid


def load_pt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = torch.load(str(path), map_location="cpu", weights_only=False).detach().numpy().astype(np.float64)
    human = data[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)
    obj = data[:, 318:325]
    return data, human, obj


def audit_clip(demo: str, variant: str, spec: G1Spec, m, geom_ids: dict) -> dict:
    npz_path = TEACHER_TRAJ_DIR / f"{demo}_{variant}.npz"
    pt_path = OMOMO_NEW_DIR / f"{demo}.pt"
    r: dict = {"demo": demo, "variant": variant}

    npz = np.load(npz_path, allow_pickle=True)
    q, nh = npz["qpos"].astype(np.float64), npz["human_joints"].astype(np.float64)
    r["fps"] = int(npz["fps"])
    r["T_npz"], r["T_pt"] = int(q.shape[0]), -1

    data, hpt, obj = load_pt(pt_path)
    r["T_pt"] = int(data.shape[0])
    r["pt_cols"] = int(data.shape[1])

    # B/C. layouts
    r["root_quat_norm"] = float(np.linalg.norm(q[:, 0:4], axis=1).min())
    r["root_quat_norm_max"] = float(np.linalg.norm(q[:, 0:4], axis=1).max())
    r["box_quat_norm_min"] = float(np.linalg.norm(q[:, 36:40], axis=1).min())

    # D. scene relation
    scale, Rfit, t, resid = fit_similarity(hpt.reshape(-1, 3), nh.reshape(-1, 3))
    r["human_sim_scale"] = scale
    r["human_sim_resid_max_m"] = resid
    Rpt = Rot.from_quat(obj[:, 3:7]).as_matrix()
    Rnpz = Rot.from_quat(q[:, 36:40][:, [1, 2, 3, 0]]).as_matrix()
    pred = Rfit[None] @ Rpt
    tr = np.trace(pred.transpose(0, 2, 1) @ Rnpz, axis1=1, axis2=2)
    r["box_rot_err_deg_max"] = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)).max()))
    pn = scale * (obj[:, :3] @ Rfit.T) + t
    offset = q[:, 40:43] - pn
    r["box_pos_offset_mean"] = offset.mean(0).tolist()
    r["box_pos_offset_std_max"] = float(np.abs(offset - offset.mean(0)).max())

    # E. teacher replay
    d = mujoco.MjData(m)
    lo, hi = spec.joint_range[:, 0], spec.joint_range[:, 1]
    viol = float(np.abs(np.clip(q[:, 7:36], lo, hi) - q[:, 7:36]).max())
    r["joint_limit_max_violation_rad"] = viol
    sole_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in SOLE_SITES["l"] + SOLE_SITES["r"]]
    pen_rg = pen_rb = pen_og = 0.0
    min_palm = np.inf
    fromto = np.zeros(6)
    mid = q.shape[0] // 2
    for t_i in range(q.shape[0]):
        d.qpos = convert_scene_qpos(q[t_i])
        mujoco.mj_forward(m, d)
        if t_i in (0, mid, q.shape[0] - 1):
            for g1, g2, bucket in geom_ids["pairs"]:
                dist = mujoco.mj_geomDistance(m, d, g1, g2, 2.0, fromto)
                if dist < 0:
                    if bucket == "robot_ground":
                        pen_rg = max(pen_rg, -dist)
                    elif bucket == "robot_box":
                        pen_rb = max(pen_rb, -dist)
                    else:
                        pen_og = max(pen_og, -dist)
            for pg in geom_ids["palms"]:
                dist = mujoco.mj_geomDistance(m, d, pg, geom_ids["box"], 2.0, fromto)
                min_palm = min(min_palm, dist)
    sole_z = []
    for t_i in (0, mid):
        d.qpos = convert_scene_qpos(q[t_i])
        mujoco.mj_forward(m, d)
        sole_z.append(d.site_xpos[sole_ids][:, 2].min())
    r["sole_min_z_m"] = float(min(sole_z))
    r["penetration_robot_ground_m"] = pen_rg
    r["penetration_robot_box_m"] = pen_rb
    r["penetration_box_ground_m"] = pen_og
    r["palm_min_dist_to_box_m"] = float(min_palm)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=4)
    ap.add_argument("--include-aug", action="store_true", default=True)
    args = ap.parse_args()

    # A. inventory
    npz_names = sorted(p.name for p in TEACHER_TRAJ_DIR.glob("*.npz"))
    originals = sorted(n[: -len("_original.npz")] for n in npz_names if n.endswith("_original.npz"))
    pt_largebox = sorted(p.stem for p in OMOMO_NEW_DIR.glob("*_largebox_*.pt"))
    by_subj: dict[str, list[str]] = defaultdict(list)
    for demo in originals:
        by_subj[demo.split("_")[0]].append(demo)
    missing_teacher = sorted(set(pt_largebox) - set(originals))
    inventory = {
        "npz_total": len(npz_names),
        "original_demos": len(originals),
        "rot_variants": sum(n.endswith((".npz",)) and "_rot_" in n for n in npz_names),
        "trans_variants": sum("_trans_" in n for n in npz_names),
        "pt_largebox_files": len(pt_largebox),
        "pt_without_teacher": missing_teacher,
        "demos_per_subject": {k: len(v) for k, v in sorted(by_subj.items())},
    }

    # sample
    spec = G1Spec()
    m = spec.load_scene()
    box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "largebox")
    ground_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    robot_geoms = [
        g for g in range(m.ngeom)
        if m.geom_contype[g] or m.geom_conaffinity[g]
        if g not in (box_geom, ground_geom)
    ]
    geom_ids = {
        "box": box_geom,
        "palms": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{h}_rubber_hand_link") for h in ("left", "right")],
        "pairs": (
            [(g, ground_geom, "robot_ground") for g in robot_geoms]
            + [(g, box_geom, "robot_box") for g in robot_geoms]
            + [(box_geom, ground_geom, "box_ground")]
        ),
    }

    sample = []
    for subj, demos in sorted(by_subj.items()):
        for demo in demos[: args.per_subject]:
            sample.append((demo, "original"))
            if args.include_aug:
                for prefix in ("rot", "trans"):
                    variants = sorted(
                        n[len(demo) + 1 : -len(".npz")]
                        for n in npz_names
                        if n.startswith(f"{demo}_{prefix}_") and n.endswith(".npz")
                    )
                    if variants:
                        sample.append((demo, variants[0]))

    results = [audit_clip(demo, variant, spec, m, geom_ids) for demo, variant in sample]

    AUDIT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"inventory": inventory, "sample_size": len(results), "clips": results}
    out = AUDIT_OUT_DIR / "m0_pairing_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # summary
    def agg(key: str) -> tuple[float, float, float]:
        vals = np.array([c[key] for c in results])
        return float(vals.min()), float(vals.mean()), float(vals.max())

    print(json.dumps(inventory, indent=2, ensure_ascii=False))
    print(f"\nsample: {len(results)} clips")
    for key in [
        "root_quat_norm", "box_quat_norm_min", "human_sim_resid_max_m", "box_rot_err_deg_max",
        "box_pos_offset_std_max", "joint_limit_max_violation_rad", "sole_min_z_m",
        "penetration_robot_ground_m", "penetration_robot_box_m", "penetration_box_ground_m",
        "palm_min_dist_to_box_m",
    ]:
        mn, mean, mx = agg(key)
        print(f"{key:36s} min={mn:.6f} mean={mean:.6f} max={mx:.6f}")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
