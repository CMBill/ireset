"""M3/P0.3 closed-loop experiment: generation quality and repair coupling.

For sampled validation-split originals:

1. generate full trajectories with each model (diffusion DDIM-25 by
   default; deterministic single pass), stitched over 60/30 windows;
2. RAW fidelity vs the teacher (generation diagnostics only): root RMSE,
   geodesic, joint MAE, FK keypoint MPJPE -- explicitly labeled
   ``teacher_fit`` (robot-vs-teacher-FK error), never used as the task
   metric;
3. repair the generated output, the bare teacher (no-noise repair input,
   the F4.3 baseline) and the teacher-with-noise baselines with the same
   repairer; accepted trajectories get REPAIRED metrics: task keypoint
   MPJPE vs the REFERENCE HUMAN keypoints (box frame + world frame, the
   repairer-gate caliber), re-inferred hand F1, repair magnitude, slack
   fraction and end-to-end timing (rejected rows keep timing, never get a
   position error);
4. P0.3 relative conditions per method vs the teacher baseline: acceptance
   >= baseline, paired task MPJPE <= baseline + 2 cm on the COMMON
   accepted set, success-weighted hand F1 >= 0.95 x baseline; missing or
   undefined evidence leaves a condition unpassed.

Products carry metadata (code rev, mode, task gate, inputs, seeds, model
checksums) and mode/gate-stamped default names; a stored teacher baseline
is reused only when its metadata matches exactly.

Run: uv run --locked python scripts/eval_generation.py [--demos 8] [--frames 120]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from closed_loop_metrics import (hand_f1_report, metadata_block,
                                 metadata_compatible, sha256, summarize_method)
from ireset.data.teacher import build_interaction_clip
from ireset.data.windows import CACHE_DIR, load_split_files
from ireset.diffusion.generator import DiffusionG1
from ireset.embodied.g1_spec import KPT_TO_SMPLH
from ireset.eval.metrics import contact_f1_all, contact_prf
from ireset.geometry.distance_adapter import world_to_box_geom
from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1
from ireset.model.generator import G1Generator
from ireset.model.tokens import RootStats
from ireset.data.windows import make_tokenizer
from ireset.kinematics.mujoco_adapter import G1Kinematics
from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz
from ireset.repair.config_io import MODES, load_mode_configs
from ireset.repair.pipeline import KPT_IDX, RepairPipeline
from ireset.repair.validation import IndependentValidator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "derived" / "data" / "validation"
KPT_IDS = list(KPT_TO_SMPLH)
HAND_EFFECTORS = (("hand_f1_left", "left_hand"), ("hand_f1_right", "right_hand"))


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


def task_keypoint_mpjpe(ev, traj_scene: np.ndarray, cond) -> tuple[float, float]:
    """Task metric: G1 correspondence keypoints vs the REFERENCE HUMAN
    keypoints. Box-frame value is the primary caliber (plan 10.1 task
    frame); the world-frame value matches the repairer's per-frame gate
    formula exactly and is reported for cross-checking."""
    w, b = [], []
    for t in range(traj_scene.shape[0]):
        ev.forward(traj_scene[t, 0:36], traj_scene[t, 36:43])
        kpt_w = ev.site_positions(ev.kpt_ids)
        ref_w = cond.human_kps_world[t][KPT_IDX]
        w.append(np.linalg.norm(kpt_w - ref_w, axis=-1).mean())
        pose = traj_scene[t : t + 1, 36:43]
        kb = world_to_box_geom(kpt_w[None], pose)[0]
        rb = world_to_box_geom(ref_w[None], pose)[0]
        b.append(np.linalg.norm(kb - rb, axis=-1).mean())
    return float(np.mean(b)), float(np.mean(w))


def eval_raw(traj_scene: np.ndarray, teacher_scene: np.ndarray, cond, kin, validator) -> dict:
    """Bare-output diagnostics vs the teacher (fitting error, NOT task)."""
    root_rmse = float(np.sqrt(((traj_scene[:, 0:3] - teacher_scene[:, 0:3]) ** 2).sum(1).mean()))
    root_geo = float(geodesic_deg(traj_scene[:, 3:7], teacher_scene[:, 3:7]).mean())
    joint_mae = float(np.abs(traj_scene[:, 7:36] - teacher_scene[:, 7:36]).mean())
    kp_p, kp_t = fk_keypoints(kin, traj_scene), fk_keypoints(kin, teacher_scene)
    box = cond.target_obj_pose
    mpjpe_box = float(np.linalg.norm(box_frame(kp_p, box) - box_frame(kp_t, box), axis=-1).mean())
    robot_sched = validator._infer_robot_schedule(traj_scene)
    f1 = contact_f1_all(robot_sched, cond.schedule, traj_scene.shape[0])
    bare = validator.validate(traj_scene, exclude=("box_ground",))
    return {
        "root_rmse_m": root_rmse, "root_geodesic_deg": root_geo, "joint_mae_rad": joint_mae,
        "teacher_fit_mpjpe_box_m": mpjpe_box,
        "contact_f1": {k: v for k, v in f1.items()},
        "bare_validation": {"status": bare.status, "summary": bare.summary()},
    }


def repair_run(pipe, validator, scene, cond, tag, teacher_scene=None, kin=None,
               save_npz: Path | None = None) -> dict:
    """One repair invocation -> row. Accepted rows carry repaired metrics;
    rejected rows keep timing/diagnostics and never get a position error."""
    t0 = time.perf_counter()
    res = pipe.repair_trajectory(scene, cond.to_repair_condition())
    wall = time.perf_counter() - t0
    d = res.diagnostics
    row = {"tag": tag, "accepted": bool(res.accepted), "status": res.status,
           "wall_s": round(wall, 2), "end_to_end_s": d.get("total_s"),
           "t_fail": d.get("t_fail"), "skip_reason": d.get("skip_reason"),
           "reason": d.get("reason")}
    tm = d.get("timings_s")
    if tm is not None:
        a = np.asarray(tm) * 1e3
        row.update({
            "frames": int(len(a)),
            "ms_per_frame_mean": round(float(a.mean()), 1),
            "ms_per_frame_p50": round(float(np.percentile(a, 50)), 1),
            "ms_per_frame_p95": round(float(np.percentile(a, 95)), 1),
            "ms_per_frame_max": round(float(a.max()), 1),
            "skip_rate": round(float(d.get("skipped", 0)) / max(1, len(a)), 3),
            "slack_frames": int(d.get("slack_frames", 0)),
            "slack_fraction": round(float(d.get("slack_frames", 0)) / max(1, len(a)), 3),
        })
    if res.accepted and res.trajectory is not None:
        rep = res.trajectory
        if save_npz is not None:
            save_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(save_npz, trajectory=rep, prediction=scene,
                     human_kpts=cond.human_kps_world, box_pose=cond.target_obj_pose)
        row["task_mpjpe_box_m"], row["task_mpjpe_world_m"] = task_keypoint_mpjpe(
            pipe.ev, rep, cond)
        if teacher_scene is not None and kin is not None:
            kp_r, kp_t = fk_keypoints(kin, rep), fk_keypoints(kin, teacher_scene)
            row["teacher_fit_mpjpe_box_m"] = float(np.linalg.norm(
                box_frame(kp_r, rep[:, 36:43]) - box_frame(kp_t, rep[:, 36:43]),
                axis=-1).mean())
        sched = validator._infer_robot_schedule(rep)
        for key, eff in HAND_EFFECTORS:
            prf = contact_prf(sched, cond.schedule, eff, rep.shape[0])
            row[key] = None if prf is None else prf[2]
        row["hand_intervals_inferred"] = sched.stats.get("hand_intervals")
        row["repair_magnitude"] = {
            "root_trans_rmse_m": float(np.sqrt(
                ((rep[:, 0:3] - scene[:, 0:3]) ** 2).sum(1).mean())),
            "root_rot_geodesic_deg_mean": float(
                geodesic_deg(rep[:, 3:7], scene[:, 3:7]).mean()),
            "joint_rmse_rad": float(np.sqrt(
                ((rep[:, 7:36] - scene[:, 7:36]) ** 2).mean())),
            "joint_max_abs_rad": float(np.abs(rep[:, 7:36] - scene[:, 7:36]).max()),
        }
    return row


def load_teacher_baseline(path: Path, mode: str, gate: float, files: list[str], seed: int):
    doc = json.loads(path.read_text())
    ok, why = metadata_compatible(doc.get("metadata", {}), mode, gate, files, seed, {})
    return (ok, why, doc.get("teacher_baseline_rows", []))


def run_w1_raw_eval(args) -> None:
    """W1 raw full-length evaluation (omniretarget_w1_a3bc_plan §5.3).

    Fixed expected-set (the training manifest's val31 — failures and short
    trajectories stay rows in the denominator, never dropped), FULL clip
    length (the stitched trajectory covers the last frame through the
    trailing T-60 window), explicit model/manifest identity verification,
    and ZERO RepairPipeline construction or invocation — raw diagnostics
    only (the frozen IndependentValidator is used for bare-output checks;
    it is not the repairer).
    """
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1
    from ireset.paths import PROJECT_ROOT, path_for_record

    manifest_path = args.training_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    for key, want in (("manifest_name", w1.MANIFEST_NAME),
                      ("protocol", w1.MOTION_PROTOCOL_VERSION),
                      ("experiment_id", w1.EXPERIMENT_ID)):
        if manifest.get(key) != want:
            raise SystemExit(f"training manifest {key} mismatch: "
                             f"{manifest.get(key)!r} != {want!r}")
    prereg_rec = manifest["preregistration"]
    prereg_path = PROJECT_ROOT / prereg_rec["path"]
    if integ.sha256_file(prereg_path) != prereg_rec["sha256"]:
        raise SystemExit("training manifest not bound to its preregistration")
    prereg = json.loads(prereg_path.read_text())
    assets_sha = prereg["assets"]["a1_assets_sha256"]
    code_sha = w1.condition_code_sha256(PROJECT_ROOT)

    if args.expected_set == "val31":
        expected = list(manifest["selection"]["val31"])
    elif args.expected_set:
        expected = json.loads(Path(args.expected_set).read_text())["demos"]
    else:
        raise SystemExit("W1 eval requires --expected-set (val31 or a "
                         "frozen demo-list json) — no default subset")
    rows_by_demo = {r["demo"]: r for r in manifest["rows"] if r.get("split") == "val"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)
    manifest_sha = integ.sha256_file(manifest_path)
    generators = {}
    for m in args.models:
        ckpt = torch.load(models_dir / m / "best.pt", map_location=device,
                          weights_only=False)
        ident = ckpt.get("identity") or {}
        if (ident.get("training_manifest", {}).get("sha256") != manifest_sha
                or ident.get("protocol") != w1.MOTION_PROTOCOL_VERSION):
            raise SystemExit(
                f"model {m}: checkpoint identity does not match this training "
                f"manifest (ckpt={ident.get('training_manifest')}) — a "
                "checkpoint from another manifest/cache is refused")
        generators[m] = load_generator(m, device, args.ddim_steps, models_dir)
    kin = G1Kinematics()
    validator = IndependentValidator(cfg=load_mode_configs("strict")[1])

    out_rows = []
    for demo in expected:
        row_rec = rows_by_demo.get(demo)
        if row_rec is None:
            out_rows.append({"demo": demo, "status": "missing_row"})
            continue
        if not row_rec.get("admission", {}).get("motion"):
            out_rows.append({"demo": demo, "status": "not_admitted",
                             "reasons": row_rec["admission"]["reasons"]})
            continue
        dpath = PROJECT_ROOT / row_rec["derived"]["path"]
        if integ.sha256_file(dpath) != row_rec["derived"]["sha256"]:
            out_rows.append({"demo": demo, "status": "derived_drifted"})
            continue
        clip = w1.load_w1_derived_clip(row_rec, args.cache_dir / "schedules",
                                       PROJECT_ROOT, assets_sha, code_sha)
        T = clip.T if args.full_length else min(clip.T, args.frames)
        if T < 60:
            # short trajectories are recorded, never deleted from the set
            out_rows.append({"demo": demo, "status": "unsupported_short_trajectory",
                             "T": T})
            continue
        cond = clip.condition.truncate(T)
        teacher_scene = np.concatenate(
            [clip.root[:T], clip.q[:T], cond.target_obj_pose[:T]], axis=1)
        for model_name, gen in generators.items():
            rec = {"demo": demo, "model": model_name, "T": T,
                   "ddim_steps": args.ddim_steps if model_name == "diffusion" else None,
                   "seed": args.seed}
            try:
                traj = gen.sample(cond, seed=args.seed)
                scene = traj.scene_qpos(cond.target_obj_pose)
                rec["status"] = "ok"
                rec["generation_wall_s"] = round(traj.metadata["generation_wall_s"], 3)
                rec["boundary"] = traj.metadata["boundary"]
                rec["degenerate_windows"] = traj.metadata["degenerate_windows"]
                rec["covers_last_frame"] = bool(
                    traj.root.shape[0] == T)  # stitched length == clip length
                rec["raw_metrics"] = eval_raw(scene, teacher_scene, cond, kin, validator)
            except Exception as exc:  # noqa: BLE001 -- failure keeps the row
                rec["status"] = "failed"
                rec["error"] = f"{type(exc).__name__}: {exc}"
            out_rows.append(rec)
            print(f"[w1-raw][{model_name}] {demo}: {rec['status']} T={T}")

    processed = [r for r in out_rows if r.get("status") == "ok"]
    report = {
        "stage": "W1 raw full-length generation evaluation",
        "expected_set": {"n": len(expected), "demos": expected},
        "training_manifest": {"path": path_for_record(manifest_path),
                              "sha256": manifest_sha},
        "cache_dir": str(args.cache_dir),
        "full_length": bool(args.full_length),
        "seed": args.seed,
        "models": args.models,
        "counts": {
            "n_expected": len(expected),
            "n_rows": len(out_rows),
            "n_ok": len(processed),
            "n_failed": sum(1 for r in out_rows if r.get("status") == "failed"),
            "n_unsupported": sum(1 for r in out_rows
                                 if r.get("status") == "unsupported_short_trajectory"),
        },
        "aggregate_raw": {
            m: {k: float(np.mean([r["raw_metrics"][k] for r in processed
                                  if r["model"] == m]))
                for k in ("root_rmse_m", "root_geodesic_deg", "joint_mae_rad",
                          "teacher_fit_mpjpe_box_m")}
            for m in args.models
            if any(r["model"] == m for r in processed)
        },
        "rows": out_rows,
        "repair_pipeline_invocations": 0,
    }
    out = Path(args.output) if args.output else OUT / "w1_raw_generation_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report["counts"], indent=2))
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["diffusion", "deterministic"])
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ddim-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models-dir", type=str, default=str(ROOT / "derived" / "models"))
    ap.add_argument("--skip-noise-baselines", action="store_true")
    ap.add_argument("--validation-mode", choices=MODES, default="strict",
                    help="strict = locked gates; relaxed = P0 teacher-baseline "
                         "pairs (never valid for plan 10.1 claims)")
    ap.add_argument("--teacher-baseline-json", type=str, default=None,
                    help="reuse a stored bare-teacher baseline when its metadata "
                         "matches exactly; otherwise recomputed in this run")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--save-dir", type=str, default=None,
                    help="save accepted repaired trajectories as NPZ")
    ap.add_argument("--task-gate-yaml", type=str, default=None,
                    help="override task_mpjpe_tol_m from a task_quality_gates.yaml")
    ap.add_argument("--training-manifest", type=Path, default=None,
                    help="W1 raw mode: A3 training_manifest.json (fixed "
                         "expected-set, full length, zero repair)")
    ap.add_argument("--expected-set", type=str, default=None,
                    help="W1 raw mode: 'val31' (from the manifest) or a "
                         "frozen demo-list json {'demos': [...]}")
    ap.add_argument("--full-length", action="store_true",
                    help="W1 raw mode: evaluate the COMPLETE clip length "
                         "(default keeps the legacy frame cap)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="W1 raw mode: keyed W1 cache attempt dir (schedules)")
    args = ap.parse_args()

    if args.training_manifest is not None:
        if args.cache_dir is None:
            raise SystemExit("W1 raw mode requires --cache-dir (keyed "
                             "schedule cache; no legacy fallback)")
        return run_w1_raw_eval(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)
    generators = {m: load_generator(m, device, args.ddim_steps, models_dir) for m in args.models}
    model_meta = {m: {"checkpoint_sha256_16": sha256(models_dir / m / "best.pt"),
                      "ddim_steps": args.ddim_steps if m == "diffusion" else None}
                  for m in args.models}

    rcfg, vcfg = load_mode_configs(args.validation_mode)
    if args.task_gate_yaml:
        import yaml as _yaml
        rcfg.task_mpjpe_tol_m = float(
            _yaml.safe_load(Path(args.task_gate_yaml).read_text())
            ["task_quality_gates"]["mpjpe_tol_m"])
    pipe = RepairPipeline(cfg=rcfg)
    pipe.validator.cfg = vcfg
    kin = G1Kinematics()
    validator = IndependentValidator(cfg=vcfg)
    lo, hi = pipe.spec.joint_range[:, 0], pipe.spec.joint_range[:, 1]

    val_originals = [f for f in load_split_files("val") if f.endswith("_original.npz")][: args.demos]
    print(f"{len(val_originals)} val originals, frames<={args.frames}, "
          f"mode={args.validation_mode}, task gate={rcfg.task_mpjpe_tol_m}")

    # registered screening rule (velocity screen, P0.0); record any overlap
    screen_path = ROOT / "derived" / "data" / "screening" / "velocity_screen.json"
    screen_overlap: list[str] = []
    if screen_path.exists():
        excl = {e["demo"] if isinstance(e, dict) else e
                for e in json.loads(screen_path.read_text()).get("excluded", [])}
        screen_overlap = sorted(f for f in val_originals if f in excl)

    teacher_rows: list[dict] = []
    reuse = None
    if args.teacher_baseline_json:
        ok, why, rows = load_teacher_baseline(
            Path(args.teacher_baseline_json), args.validation_mode,
            rcfg.task_mpjpe_tol_m, val_originals, args.seed)
        if ok and rows:
            teacher_rows = rows
            reuse = str(args.teacher_baseline_json)
            print(f"teacher baseline reused from {reuse} ({len(rows)} rows)")
        else:
            print(f"teacher baseline NOT reused: {why}")

    rows: list[dict] = []
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for name in val_originals:
        clip = build_interaction_clip(name, schedule_cache=CACHE_DIR / "schedules")
        T = min(clip.T, args.frames)
        cond = clip.condition.truncate(T)
        teacher_scene = np.concatenate(
            [clip.root[:T], clip.q[:T], cond.target_obj_pose[:T]], axis=1)
        demo = name

        if reuse is None:   # bare-teacher baseline: every registered demo
            row = repair_run(pipe, validator, teacher_scene, cond, "teacher_bare",
                             save_npz=save_dir / f"teacher_bare_{demo}.npz" if save_dir else None)
            row.update({"npz": name, "model": "teacher_bare", "demo": demo, "T": T})
            teacher_rows.append(row)
            print(f"[teacher_bare] {name}: repair={row['accepted']}")

        for model_name, gen in generators.items():
            traj = gen.sample(cond, seed=args.seed)
            scene = traj.scene_qpos(cond.target_obj_pose)
            row = {"npz": name, "model": model_name, "demo": demo,
                   "T": T, "ddim_steps": args.ddim_steps if model_name == "diffusion" else None,
                   "generation_wall_s": round(traj.metadata["generation_wall_s"], 3),
                   "gen_ms_per_frame": round(traj.metadata["generation_wall_s"] * 1e3 / T, 1),
                   "boundary": traj.metadata["boundary"],
                   "degenerate_windows": traj.metadata["degenerate_windows"]}
            row["raw_metrics"] = eval_raw(scene, teacher_scene, cond, kin, validator)
            row["repair"] = repair_run(
                pipe, validator, scene, cond, f"{model_name}_bare",
                teacher_scene=teacher_scene, kin=kin,
                save_npz=save_dir / f"{model_name}_{demo}.npz" if save_dir else None)
            rows.append(row)
            rm = row["raw_metrics"]
            rep = row["repair"]
            print(f"[{model_name}] {name}: teacher_fit={rm['teacher_fit_mpjpe_box_m']:.3f}m "
                  f"repair={rep['accepted']} "
                  f"task={rep.get('task_mpjpe_box_m', '-')}")

        if args.skip_noise_baselines:
            continue
        rng = np.random.default_rng(args.seed)
        for sigma in (0.02, 0.05):
            noisy = teacher_scene.copy()
            noisy[:, 7:36] = np.clip(noisy[:, 7:36] + rng.normal(0, sigma, (T, 29)), lo, hi)
            row = repair_run(pipe, validator, noisy, cond, f"teacher_noise_{sigma}",
                             teacher_scene=teacher_scene, kin=kin)
            row.update({"npz": name, "model": f"teacher_noise_{sigma}",
                        "demo": demo, "T": T})
            rows.append(row)
            print(f"[teacher_noise_{sigma}] {name}: repair={row['accepted']}")

    # method rows for the closed-loop summary: flat per-sample dicts whose
    # accepted flags/metrics live at top level (repair_run output shape)
    method_rows = {}
    for m in [*args.models, "teacher_noise_0.02", "teacher_noise_0.05"]:
        flat = []
        for r in rows:
            src = r["repair"] if "repair" in r else r
            if r.get("model") == m:
                flat.append({**src, "demo": r["demo"], "model": m})
        if flat:
            method_rows[m] = flat

    agg = {
        "n_clips": len(val_originals),
        "frames": args.frames,
        "mode": args.validation_mode,
        "task_mpjpe_tol_m": rcfg.task_mpjpe_tol_m,
        "raw_fidelity_teacher_fit": {
            m: {k: float(np.mean([r["raw_metrics"][k] for r in rows if r.get("model") == m]))
                for k in ("root_rmse_m", "root_geodesic_deg", "joint_mae_rad",
                          "teacher_fit_mpjpe_box_m")}
            for m in args.models if any(r.get("model") == m for r in rows)
        },
        "closed_loop": {
            "teacher_baseline": {
                "n": len(teacher_rows),
                "acceptance": float(np.mean([r["accepted"] for r in teacher_rows]))
                if teacher_rows else None,
                "task_mpjpe_box_m": {
                    "mean": float(np.mean([r["task_mpjpe_box_m"] for r in teacher_rows
                                           if r.get("task_mpjpe_box_m") is not None]))
                    if any(r.get("task_mpjpe_box_m") is not None for r in teacher_rows) else None,
                },
                "hand_f1": hand_f1_report(teacher_rows)["hand_macro_success_weighted"],
                "reused_from": reuse,
            },
            "methods": {m: summarize_method(mrows, teacher_rows)
                        for m, mrows in method_rows.items()},
        },
        "screening_excluded_overlap": screen_overlap,
        "metadata": metadata_block(
            args.validation_mode, rcfg.task_mpjpe_tol_m, val_originals, args.seed,
            models=model_meta,
            extra={"ddim_steps": args.ddim_steps, "frames": args.frames,
                   "screening": "derived/data/screening/velocity_screen.json"}),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT / (
        f"generation_eval_{args.validation_mode}_tol{rcfg.task_mpjpe_tol_m:.3f}.json")
    out.write_text(json.dumps({"aggregate": agg, "rows": rows,
                               "teacher_baseline_rows": teacher_rows},
                              indent=2, default=float))
    print(json.dumps(agg["closed_loop"], indent=2, default=float))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
