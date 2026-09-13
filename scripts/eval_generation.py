"""M3 acceptance experiment: generation quality and repair coupling (plan 10.2).

For sampled validation-split originals:

1. generate full trajectories with each model (diffusion DDIM-25 by
   default; deterministic single pass), stitched over 60/30 windows;
2. report generation fidelity vs the teacher: root translation RMSE,
   root rotation geodesic, joint MAE, FK keypoint MPJPE (world + target
   box frame), robot-side inferred contact F1 vs the reference schedule;
3. run the bare output through the independent validator (expected to
   fail -- that is the repairer's job, and the failure reasons are the
   evidence);
4. repair the generated output and the teacher-with-noise baselines
   (sigma = 0.02 / 0.05, the M2 "realistic prediction error" protocol)
   with the same repairer; report acceptance, timings and repair
   magnitude; generation timing reported alongside.

Writes derived/data/validation/generation_eval.json.

Run: uv run --locked python scripts/eval_generation.py [--demos 8] [--frames 120]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from ireset.data.teacher import build_interaction_clip
from ireset.data.windows import CACHE_DIR, load_split_files
from ireset.diffusion.generator import DiffusionG1
from ireset.embodied.g1_spec import KPT_TO_SMPLH
from ireset.eval.metrics import contact_f1_all
from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1
from ireset.model.generator import G1Generator
from ireset.model.tokens import RootStats, StateTokenizer
from ireset.data.windows import make_tokenizer
from ireset.kinematics.mujoco_adapter import G1Kinematics
from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz
from ireset.repair.pipeline import RepairPipeline
from ireset.repair.validation import IndependentValidator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "derived" / "data" / "validation"
KPT_IDS = list(KPT_TO_SMPLH)


def load_generator(model_name: str, device: str, ddim_steps: int, models_dir: Path) -> G1Generator:
    ckpt = torch.load(models_dir / model_name / "best.pt",
                      map_location=device, weights_only=False)
    cfg = ckpt["config"]
    bw = dict(d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
              ffn_mult=cfg["ffn_mult"], time_window=cfg["time_window"])
    if model_name == "diffusion":
        model = DiffusionG1(num_train_timesteps=cfg["num_train_timesteps"],
                            beta_schedule=cfg["beta_schedule"], **bw)
    else:
        model = DeterministicG1(**bw)
    model.load_state_dict(ckpt["model"])
    cond = ConditionEncoder(cfg["d_model"])
    cond.load_state_dict(ckpt["conditioner"])
    stats = RootStats(mean=np.asarray(ckpt["root_stats"]["mean"]),
                      std=np.asarray(ckpt["root_stats"]["std"]))
    tokenizer = make_tokenizer(stats)
    return G1Generator(model, cond, tokenizer, device=device, ddim_steps=ddim_steps)


def fk_keypoints(kin: G1Kinematics, scene: np.ndarray) -> np.ndarray:
    from ireset.kinematics.mujoco_adapter import RobotState

    state = RobotState(pos=scene[:, 0:3].copy(), quat=scene[:, 3:7].copy(), q=scene[:, 7:36].copy())
    return kin.site_positions(state, KPT_IDS)


def box_frame(kp: np.ndarray, box_pose: np.ndarray) -> np.ndarray:
    R = quat_to_mat_wxyz(box_pose[:, 3:7])
    return np.einsum("tji,tnj->tni", R, kp - box_pose[:, 0:3][:, None, :])


def geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    Ra = Rot.from_quat(qa[:, [1, 2, 3, 0]])
    Rb = Rot.from_quat(qb[:, [1, 2, 3, 0]])
    tr = np.trace(np.einsum("tij,tjk->tik", np.transpose(Ra.as_matrix(), (0, 2, 1)),
                            Rb.as_matrix()), axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def eval_generated(traj_scene: np.ndarray, teacher_scene: np.ndarray, cond, kin, validator) -> dict:
    root_rmse = float(np.sqrt(((traj_scene[:, 0:3] - teacher_scene[:, 0:3]) ** 2).sum(1).mean()))
    root_geo = float(geodesic_deg(traj_scene[:, 3:7], teacher_scene[:, 3:7]).mean())
    joint_mae = float(np.abs(traj_scene[:, 7:36] - teacher_scene[:, 7:36]).mean())

    kp_p, kp_t = fk_keypoints(kin, traj_scene), fk_keypoints(kin, teacher_scene)
    mpjpe_world = float(np.linalg.norm(kp_p - kp_t, axis=-1).mean())
    box = cond.target_obj_pose
    kp_p_b, kp_t_b = box_frame(kp_p, box), box_frame(kp_t, box)
    mpjpe_box = float(np.linalg.norm(kp_p_b - kp_t_b, axis=-1).mean())

    robot_sched = validator._infer_robot_schedule(traj_scene)
    f1 = contact_f1_all(robot_sched, cond.schedule, traj_scene.shape[0])

    bare = validator.validate(traj_scene, exclude=("box_ground",))
    return {
        "root_rmse_m": root_rmse, "root_geodesic_deg": root_geo, "joint_mae_rad": joint_mae,
        "mpjpe_world_m": mpjpe_world, "mpjpe_box_m": mpjpe_box,
        "contact_f1": {k: v for k, v in f1.items()},
        "bare_validation": {"status": bare.status, "summary": bare.summary()},
    }


def repair_run(pipe, scene, cond, tag: str) -> dict:
    t0 = time.perf_counter()
    res = pipe.repair_trajectory(scene, cond.to_repair_condition())
    wall = time.perf_counter() - t0
    d = res.diagnostics
    row = {"tag": tag, "accepted": bool(res.accepted), "status": res.status,
           "wall_s": round(wall, 2)}
    if res.accepted and res.trajectory is not None:
        rep = res.trajectory
        row["repair_root_m"] = float(np.linalg.norm(rep[:, 0:3] - scene[:, 0:3], axis=1).mean())
        row["repair_joint_rad"] = float(np.abs(rep[:, 7:36] - scene[:, 7:36]).max())
    if "timings_s" in d:
        tm = np.array(d["timings_s"])
        row["ms_per_frame_mean"] = round(float(tm.mean() * 1e3), 1)
    else:
        row["reason"] = d.get("reason")
        row["first_failure_frame"] = d.get("first_failure_frame")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["diffusion", "deterministic"])
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ddim-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models-dir", type=str, default=str(ROOT / "derived" / "models"))
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)
    generators = {m: load_generator(m, device, args.ddim_steps, models_dir) for m in args.models}

    pipe = RepairPipeline()
    kin = G1Kinematics()
    validator = IndependentValidator()
    lo, hi = pipe.spec.joint_range[:, 0], pipe.spec.joint_range[:, 1]

    val_originals = [f for f in load_split_files("val") if f.endswith("_original.npz")][: args.demos]
    print(f"{len(val_originals)} val originals, frames<={args.frames}")

    rows = []
    for name in val_originals:
        clip = build_interaction_clip(name, schedule_cache=CACHE_DIR / "schedules")
        T = min(clip.T, args.frames)
        cond = clip.condition.truncate(T)
        teacher_scene = np.concatenate(
            [clip.root[:T], clip.q[:T], cond.target_obj_pose[:T]], axis=1)

        for model_name, gen in generators.items():
            traj = gen.sample(cond, seed=args.seed)
            scene = traj.scene_qpos(cond.target_obj_pose)
            row = {"npz": name, "model": model_name,
                   "T": T, "ddim_steps": args.ddim_steps if model_name == "diffusion" else None,
                   "generation_wall_s": round(traj.metadata["generation_wall_s"], 3),
                   "gen_ms_per_frame": round(traj.metadata["generation_wall_s"] * 1e3 / T, 1),
                   "boundary": traj.metadata["boundary"],
                   "degenerate_windows": traj.metadata["degenerate_windows"]}
            row.update(eval_generated(scene, teacher_scene, cond, kin, validator))
            row["repair"] = repair_run(pipe, scene, cond, f"{model_name}_bare")
            rows.append(row)
            print(f"[{model_name}] {name}: mpjpe={row['mpjpe_box_m']:.3f}m "
                  f"f1={row['contact_f1']['macro_f1']} "
                  f"repair={row['repair']['accepted']} bare={row['bare_validation']['status']}")

        # realistic-error baselines: teacher + joint noise, same repairer
        if args.skip_baselines:
            continue
        rng = np.random.default_rng(args.seed)
        for sigma in (0.02, 0.05):
            noisy = teacher_scene.copy()
            noisy[:, 7:36] = np.clip(noisy[:, 7:36] + rng.normal(0, sigma, (T, 29)), lo, hi)
            row = repair_run(pipe, noisy, cond, f"teacher_noise_{sigma}")
            row.update({"npz": name, "model": f"teacher_noise_{sigma}", "T": T})
            rows.append(row)
            print(f"[teacher_noise_{sigma}] {name}: repair={row['accepted']}")

    def acc(pred) -> dict:
        sel = [r for r in rows if pred(r)]
        flags = [r.get("accepted", r.get("repair", {}).get("accepted")) for r in sel]
        known = [f for f in flags if f is not None]
        return {"n": len(sel), "acceptance": float(np.mean(known)) if known else None}

    agg = {
        "n_clips": len(val_originals),
        "frames": args.frames,
        "models": {m: acc(lambda r, m=m: r.get("model") == m and "repair" in r)
                   for m in args.models},
        "teacher_noise_0.02": acc(lambda r: r.get("model") == "teacher_noise_0.02"),
        "teacher_noise_0.05": acc(lambda r: r.get("model") == "teacher_noise_0.05"),
        "fidelity": {
            m: {k: float(np.mean([r[k] for r in rows if r.get("model") == m]))
                for k in ("root_rmse_m", "root_geodesic_deg", "joint_mae_rad",
                          "mpjpe_world_m", "mpjpe_box_m")}
            for m in args.models
            if any(r.get("model") == m for r in rows)
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "generation_eval.json").write_text(
        json.dumps({"aggregate": agg, "rows": rows}, indent=2, default=float))
    print(json.dumps(agg, indent=2, default=float))


if __name__ == "__main__":
    main()
