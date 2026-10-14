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
import math
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


def guard_report_output(path: Path) -> Path:
    """W1 report output guard (review fix 10): the resolved target must sit
    under ``derived/`` (protecting resources/ source data and any other
    tree) and must not already exist — reports never overwrite prior
    evidence; re-runs write to a new path."""
    from ireset.data.a2_integrity import guard_output_dir

    resolved = guard_output_dir(path)
    if resolved.exists():
        raise SystemExit(
            f"report output already exists (never overwritten): {resolved} "
            "-- choose a new path or an attempt directory")
    return resolved


# --- W1-D/E raw matrix (omniretarget_w1_de_plan §§6, 8, 10) ----------------------

W1_DE_PRODUCT_SCHEMA = "w1_de_raw_product_v1"
W1_DE_METHODS = ("teacher", "deterministic", "diffusion")
W1_DE_SAMPLING = {"window": 60, "stride": 30, "ddim_steps": 25,
                  "eta": 0.0, "resample_attempts": 3}
W1_DE_EXCLUDE = ("box_ground",)
W1_DE_ROBOT_CHECKS = ("joint_limits", "velocities", "penetration",
                      "foot_support", "hand_grasp")


def _w1_de_finite(value) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)))


def _w1_de_count(value) -> bool:
    return (_w1_de_finite(value) and float(value) >= 0
            and float(value).is_integer())


def _w1_de_json(value):
    """JSON records nulls; the unmodified non-finite arrays stay in NPZ."""
    if isinstance(value, dict):
        return {str(k): _w1_de_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_w1_de_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _w1_de_json(value.tolist())
    if isinstance(value, torch.Tensor):
        return _w1_de_json(value.detach().cpu().tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _w1_de_digest(value) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(_w1_de_json(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False)
                          .encode()).hexdigest()


def _w1_de_array_digest(value) -> str:
    import hashlib

    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _w1_de_path(path, project_root: Path) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else Path(project_root) / p).resolve()


def _w1_de_record_path(path: Path, project_root: Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(p)


def _w1_de_deadline(deadline: float | None, phase: str) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError(f"budget_deadline:{phase}")


def _w1_de_sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _w1_de_write_report(path: Path, report: dict, *, first: bool = False) -> None:
    """Only this invocation may update its report, including revocation."""
    import os
    import tempfile

    payload = json.dumps(_w1_de_json(report), indent=2, allow_nan=False)
    if first:
        with path.open("x") as stream:
            stream.write(payload)
        return
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def guard_w1_de_outputs(output: Path, products_dir: Path,
                         registration_path: Path, registration: dict,
                         project_root: Path = ROOT) -> tuple[Path, Path]:
    """A fresh raw attempt of this execution, under derived and away from SOURCE."""
    from ireset.data import a2_integrity as integ

    out = integ.guard_output_dir(Path(output))
    products = integ.guard_output_dir(Path(products_dir))
    reg_path = _w1_de_path(registration_path, project_root)
    if reg_path.parent.name != "freeze":
        raise SystemExit("execution registration must be in RUN/freeze")
    run = reg_path.parent.parent
    source_rec = registration.get("source_run_root")
    if not isinstance(source_rec, dict) or not source_rec.get("path"):
        raise SystemExit("registration source_run_root missing")
    source = _w1_de_path(source_rec["path"], project_root)
    if run == source or source in run.parents or run in source.parents:
        raise SystemExit("raw execution root overlaps SOURCE")
    att = out.parent
    if (att.parent != run / "evaluation" / "raw"
            or not att.name.startswith("attempt_") or not att.name[8:].isdigit()):
        raise SystemExit("raw output must be in RUN/evaluation/raw/attempt_NNN")
    if products.parent != att or products.name != "products":
        raise SystemExit("raw products must be in the same attempt/products")
    if out.exists() or products.exists():
        raise SystemExit("raw output/products already exist (never overwritten)")
    if att.exists() and any(p.name not in ("driver_subprocess.log", "driver_call.json")
                            for p in att.iterdir()):
        raise SystemExit("raw attempt contains prior evidence (never overwritten)")
    return out, products


def w1_de_expected_matrix(registration: dict) -> list[dict]:
    """The registered roster defines all keys, including absent labels."""
    ev, demos = registration["evaluation"], registration["rosters"]["val31"]
    if (not isinstance(demos, list) or not demos
            or any(not isinstance(d, str) or not d for d in demos)
            or len(set(demos)) != len(demos)):
        raise ValueError("val31 roster missing, invalid or duplicated")
    if ev.get("methods") != list(W1_DE_METHODS):
        raise ValueError("registered methods must be teacher/deterministic/diffusion")
    if ev.get("main_seed") != 7 or ev.get("appendix_seeds") != [17, 27]:
        raise ValueError("registered seeds must be 7, with appendix 17/27")
    groups = [("teacher", None, "main"), ("deterministic", 7, "main"),
              ("diffusion", 7, "main"), ("diffusion", 17, "appendix"),
              ("diffusion", 27, "appendix")]
    expected = [{"key": f"{m}|{d}|{s}", "method": m, "demo": d,
                 "seed": s, "matrix": scope}
                for d in demos for m, s, scope in groups]
    counts = ev.get("expected_matrix")
    if isinstance(counts, dict) and (counts.get("main_keys") != 3 * len(demos)
                                  or counts.get("appendix_keys") != 2 * len(demos)):
        raise ValueError("expected_matrix counts disagree with roster")
    return expected


def w1_de_reference_facts(condition, T: int | None = None) -> dict:
    """Union counts and both-reference intersection on the full timeline."""
    from ireset.eval.teacher_comparison import _longest_true_run

    T = condition.T if T is None else T
    masks = {e: np.zeros(T, dtype=bool) for e in
             ("left_hand", "right_hand", "left_foot", "right_foot")}
    intervals = []
    for iv in condition.schedule.intervals:
        if (iv.effector not in masks
                or iv.support != ("object" if "hand" in iv.effector else "ground")
                or not _w1_de_count(iv.start) or not _w1_de_count(iv.end)
                or not 0 <= iv.start < iv.end <= T):
            raise ValueError("invalid frozen reference interval")
        st, end = int(iv.start), int(iv.end)
        intervals.append({"eff": iv.effector, "sup": iv.support,
                          "start": st, "end": end})
        masks[iv.effector][st:end] = True
    both = masks["left_hand"] & masks["right_hand"]
    idx, ranges = np.flatnonzero(both), []
    if len(idx):
        for part in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            ranges.append([int(part[0]), int(part[-1]) + 1])
    grasp = {s: int(masks[f"{side}_hand"].sum())
             for s, side in (("l", "left"), ("r", "right"))}
    support = {s: int(masks[f"{side}_foot"].sum())
               for s, side in (("l", "left"), ("r", "right"))}
    return {"reference_intervals": intervals,
            "ref_grasp_frames": grasp, "ref_support_frames": support,
            "ref_frames": {"grasp_l": grasp["l"], "grasp_r": grasp["r"],
                           "support_l": support["l"], "support_r": support["r"]},
            "both_reference": {"denominator_frames": int(both.sum()),
                               "ranges": ranges,
                               "longest_consecutive": _longest_true_run(both)}}


def _w1_de_reference_gaps(row: dict) -> list[str]:
    from ireset.data.schema import ReferenceCondition
    from ireset.embodied.contact_sites import ContactInterval

    T, intervals = row.get("T"), row.get("reference_intervals")
    if not _w1_de_count(T) or T == 0 or not isinstance(intervals, list):
        return ["reference_evidence_missing"]
    try:
        cond = ReferenceCondition("reference_check", np.empty((int(T), 52, 3)),
                                  np.empty((int(T), 7)), np.empty((int(T), 7)),
                                  contacts=[ContactInterval(i["eff"], i["start"],
                                                            i["end"], i["sup"])
                                            for i in intervals])
        want = w1_de_reference_facts(cond)
    except (KeyError, TypeError, ValueError):
        return ["reference_intervals_invalid"]
    gaps = []
    for field in ("ref_grasp_frames", "ref_support_frames", "ref_frames"):
        got = row.get(field)
        if (not isinstance(got, dict) or set(got) != set(want[field])
                or any(not _w1_de_count(got[k]) or got[k] != v
                       for k, v in want[field].items())):
            gaps.append(f"{field}_missing_or_mismatched")
    if row.get("both_reference") != want["both_reference"]:
        gaps.append("both_reference_mismatched")
    both = row.get("both")
    if (not isinstance(both, dict)
            or not _w1_de_count(both.get("denominator_frames"))
            or both["denominator_frames"] != want["both_reference"]["denominator_frames"]):
        gaps.append("both_denominator_missing_or_mismatched")
    return gaps


def _w1_de_field(row: dict, path: str):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def w1_de_row_evidence_gaps(row: dict) -> list[str]:
    """A measured quality rejection is complete; an absent measurement is not."""
    if not isinstance(row, dict):
        return ["row_not_a_mapping"]
    status = row.get("status")
    if status not in ("ok", "numerical_rejected"):
        return [f"row_status:{status}"]
    gaps = _w1_de_reference_gaps(row)
    measurement_gaps = row.get("measurement_gaps", [])
    if not isinstance(measurement_gaps, list) or any(
            not isinstance(reason, str) for reason in measurement_gaps):
        gaps.append("measurement_gaps_invalid")
    else:
        gaps += measurement_gaps
    product = row.get("product")
    if (not isinstance(product, dict) or not product.get("path")
            or not isinstance(product.get("sha256"), str)
            or len(product["sha256"]) != 64):
        gaps.append("product_missing_or_unhashed")
    if row.get("product_verified") is not True:
        gaps.append("product_not_verified")
    if row.get("retry_evidence_problems") != []:
        gaps.append("retry_evidence_missing_or_invalid")
    if status == "numerical_rejected":
        facts = row.get("numerical_facts")
        if (row.get("numerical_valid") is not False
                or row.get("numerical_evidence_verified") is not True
                or not isinstance(facts, dict)
                or not any(v is False for v in facts.values())
                or facts.get("final_invalid_indices_match_output") is False
                or row.get("measurements_status") != "not_applicable_numerical_rejection"
                or row.get("raw_standard_pass") is not False):
            gaps.append("numerical_rejection_not_corroborated")
        return gaps
    if row.get("numerical_valid") is not True:
        gaps.append("ok_row_not_numerically_valid")
    if row.get("measurements_status") != "measured":
        gaps.append("measurements_missing")
    required = ("fitting.root_rmse_m", "fitting.root_geodesic_deg",
                "fitting.joint_mae_rad", "fitting.teacher_fit_fk_mpjpe_box_m",
                "task.world_mpjpe_mean_m", "task.world_mpjpe_p95_m",
                "task.world_mpjpe_max_m", "task.box_mpjpe_mean_m",
                "task.box_mpjpe_p95_m", "task.box_mpjpe_max_m",
                "penetration.max_m", "penetration.keyframe_violation_ratio",
                "box_ground.penetration_max_m", "box_ground.violation_ratio")
    gaps += [f"metric_missing_or_nonfinite:{p}" for p in required
             if not _w1_de_finite(_w1_de_field(row, p))]
    checks = _w1_de_field(row, "robot_strict.checks")
    if not isinstance(checks, dict) or set(checks) != set(W1_DE_ROBOT_CHECKS):
        gaps.append("robot_checks_incomplete")
    else:
        bad_checks = [k for k, v in checks.items() if not isinstance(v, dict)
                      or type(v.get("passed")) is not bool
                      or not _w1_de_finite(v.get("worst"))
                      or not _w1_de_finite(v.get("limit"))]
        gaps += [f"robot_check_invalid:{k}" for k in bad_checks]
        failed = sorted(k for k, v in checks.items()
                        if isinstance(v, dict) and v.get("passed") is False)
        if _w1_de_field(row, "robot_strict.failed_checks") != failed:
            gaps.append("failed_checks_mismatched")
        strict = not failed and not bad_checks
        if row.get("robot_strict_pass") is not strict \
                or _w1_de_field(row, "robot_strict.passed") is not strict:
            gaps.append("strict_pass_not_reproducible")
    task_max, tol = (_w1_de_field(row, "task.world_mpjpe_max_m"),
                     _w1_de_field(row, "task.tol_m"))
    task = bool(_w1_de_finite(task_max) and _w1_de_finite(tol) and task_max <= tol)
    if row.get("task_gate_pass") is not task or _w1_de_field(row, "task.gate_pass") is not task:
        gaps.append("task_gate_not_reproducible")
    if row.get("raw_standard_pass") is not (row.get("robot_strict_pass") is True and task):
        gaps.append("raw_standard_pass_not_reproducible")
    for family, count_field, fields in (
            ("hands", "ref_grasp_frames", ("gap_p95_m", "coverage", "valid_coverage",
                                            "drift_max_m", "relative_speed_max_mps")),
            ("feet", "ref_support_frames", ("gap_p95_m", "drift_max_m"))):
        for s in ("l", "r"):
            count, value = (_w1_de_field(row, f"{count_field}.{s}"),
                            _w1_de_field(row, f"{family}.{s}"))
            if _w1_de_count(count) and count > 0:
                if (not isinstance(value, dict) or value.get("n_ref_frames") != count
                        or any(not _w1_de_finite(value.get(k)) for k in fields)):
                    gaps.append(f"{family}_{s}_missing_with_reference")
                elif family == "feet" and value.get("n_speed_frames", 0) > 0 \
                        and not _w1_de_finite(value.get("speed_p95_mps")):
                    gaps.append(f"feet_{s}_speed_missing")
    denom = _w1_de_field(row, "both.denominator_frames")
    if _w1_de_count(denom) and denom > 0:
        for name in ("valid_coverage", "gap_series_p95_m", "certified_coverage"):
            if not _w1_de_finite(_w1_de_field(row, f"both.{name}")):
                gaps.append(f"both_{name}_missing")
        for name in ("certified_overlap_frames", "longest_certified_overlap",
                     "longest_consecutive_valid"):
            v = _w1_de_field(row, f"both.{name}")
            if not _w1_de_count(v) or v > denom:
                gaps.append(f"both_{name}_invalid")
    f1 = row.get("contact_f1")
    if not isinstance(f1, dict) or any(e not in f1 for e in
            ("left_hand", "right_hand", "left_foot", "right_foot", "n_defined", "macro_f1")):
        gaps.append("contact_f1_missing")
    else:
        defined = []
        undefined_reasons = f1.get("undefined_reasons")
        if not isinstance(undefined_reasons, dict):
            gaps.append("contact_f1_undefined_reasons_invalid")
            undefined_reasons = {}
        for eff in ("left_hand", "right_hand", "left_foot", "right_foot"):
            value = f1[eff]
            if value is not None:
                if not isinstance(value, dict) or not all(
                        _w1_de_finite(value.get(k)) and 0 <= value[k] <= 1
                        for k in ("precision", "recall", "f1")):
                    gaps.append(f"contact_f1_invalid:{eff}")
                else:
                    defined.append(value["f1"])
            elif not isinstance(undefined_reasons.get(eff), str) or not undefined_reasons[eff]:
                gaps.append(f"contact_f1_undefined_reason_missing:{eff}")
        if not _w1_de_count(f1["n_defined"]) or f1["n_defined"] != len(defined):
            gaps.append("contact_f1_count_mismatch")
        if defined and (not _w1_de_finite(f1["macro_f1"])
                        or abs(f1["macro_f1"] - sum(defined) / len(defined)) > 1e-12):
            gaps.append("contact_f1_macro_mismatch")
        elif not defined and f1["macro_f1"] is not None:
            gaps.append("contact_f1_macro_mismatch")
    return sorted(set(gaps))


def w1_de_matrix_completeness(rows: list[dict], registration: dict, *,
                              budget_stopped: bool = False) -> dict:
    from collections import Counter

    expected = w1_de_expected_matrix(registration)
    by_key = {r["key"]: r for r in expected}
    valid_rows, structural_gaps = [], {}
    if not isinstance(rows, list):
        structural_gaps["rows"] = ["rows_not_a_list"]
        rows = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            structural_gaps[f"row[{i}]"] = ["row_not_a_mapping"]
        elif not isinstance(row.get("key"), str) or not row["key"]:
            structural_gaps[f"row[{i}]"] = ["row_key_invalid"]
        else:
            valid_rows.append(row)
    count = Counter(r["key"] for r in valid_rows)
    duplicates = sorted(k for k, n in count.items() if n > 1)
    extra = sorted(k for k in count if k not in by_key)
    gaps = dict(structural_gaps)
    for row in valid_rows:
        key, why = row["key"], w1_de_row_evidence_gaps(row)
        if key in by_key and any(row.get(k) != v for k, v in by_key[key].items()):
            why.append("matrix_identity_mismatch")
        if why:
            gaps[key] = sorted(set(gaps.get(key, []) + why))
    for demo in registration["rosters"]["val31"]:
        refs = [_w1_de_digest({k: r.get(k) for k in
                              ("reference_intervals", "ref_frames", "both_reference")})
                for r in rows if isinstance(r, dict) and r.get("demo") == demo
                and r.get("status") in ("ok", "numerical_rejected")]
        if len(set(refs)) > 1:
            for r in rows:
                if isinstance(r, dict) and r.get("demo") == demo:
                    gaps.setdefault(str(r.get("key")), []).append("shared_reference_disagreement")
    main = [r["key"] for r in expected if r["matrix"] == "main"]
    appendix = [r["key"] for r in expected if r["matrix"] == "appendix"]
    def complete(keys):
        return (not budget_stopped and not structural_gaps and not duplicates and not extra
                and all(count[k] == 1 and k not in gaps for k in keys))
    return {"expected_main_keys": len(main), "expected_appendix_keys": len(appendix),
            "expected_total_keys": len(expected), "got_rows": len(rows),
            "duplicate_keys": duplicates, "unexpected_keys": extra,
            "missing_main_keys": [k for k in main if k not in count],
            "missing_appendix_keys": [k for k in appendix if k not in count],
            "not_run_keys": [r.get("key") for r in rows if isinstance(r, dict)
                             and r.get("status") == "not_run"],
            "rows_without_product": [r.get("key") for r in rows if isinstance(r, dict)
                                     and r.get("status") in ("ok", "numerical_rejected")
                                     and not isinstance(r.get("product"), dict)],
            "evidence_gaps": gaps, "budget_stopped": bool(budget_stopped),
            "all_methods_present": {r.get("method") for r in valid_rows
                                    if isinstance(r.get("method"), str)} == set(W1_DE_METHODS),
            "main_complete": complete(main), "appendix_complete": complete(appendix),
            "complete": complete(main + appendix)}


def w1_de_aggregate_rows(rows: list[dict], registration: dict) -> dict:
    """Demo macros over the full roster, with labeled defined-only diagnostics."""
    groups = {}
    for r in w1_de_expected_matrix(registration):
        groups.setdefault((r["method"], r["seed"], r["matrix"]), []).append(r["demo"])
    fields = {"root_rmse_m": "fitting.root_rmse_m",
              "root_geodesic_deg": "fitting.root_geodesic_deg",
              "joint_mae_rad": "fitting.joint_mae_rad",
              "teacher_fit_fk_mpjpe_box_m": "fitting.teacher_fit_fk_mpjpe_box_m",
              "task_world_mpjpe_mean_m": "task.world_mpjpe_mean_m",
              "task_world_mpjpe_p95_m": "task.world_mpjpe_p95_m",
              "task_world_mpjpe_max_m": "task.world_mpjpe_max_m",
              "task_box_mpjpe_mean_m": "task.box_mpjpe_mean_m",
              "penetration_max_m": "penetration.max_m",
              "box_ground_penetration_max_m": "box_ground.penetration_max_m"}
    for s in ("l", "r"):
        for fam, names in (("hands", ("gap_p95_m", "coverage", "valid_coverage",
                                      "drift_max_m", "certified_coverage")),
                           ("feet", ("gap_p95_m", "speed_p95_mps", "drift_max_m"))):
            for name in names:
                fields[f"{fam}_{s}_{name}"] = f"{fam}.{s}.{name}"
    for name in ("valid_coverage", "certified_coverage", "gap_series_p95_m"):
        fields[f"both_{name}"] = f"both.{name}"
    for eff in ("left_hand", "right_hand", "left_foot", "right_foot"):
        fields[f"contact_f1_{eff}"] = f"contact_f1.{eff}.f1"
    out = {}
    for (method, seed, scope), demos in groups.items():
        grp = [r for r in rows if isinstance(r, dict) and r.get("method") == method
               and r.get("seed") == seed]
        roster_ok = (len(grp) == len(demos)
                     and sorted(r.get("demo", "") for r in grp) == sorted(demos))
        measured = [r for r in grp if r.get("status") == "ok"
                    and r.get("numerical_valid") is True]
        counts = {"expected": len(demos), "n_rows": len(grp),
                  "attempted": sum(r.get("attempted") is True for r in grp),
                  "processed": sum(r.get("status") in ("ok", "numerical_rejected")
                                   and r.get("product_verified") is True for r in grp),
                  "numerical_rejected": sum(r.get("status") == "numerical_rejected" for r in grp),
                  "strict_rejected": sum(r.get("robot_strict_pass") is False for r in measured),
                  "task_rejected": sum(r.get("task_gate_pass") is False for r in measured),
                  "raw_standard_pass": sum(not w1_de_row_evidence_gaps(r)
                                           and r.get("raw_standard_pass") is True for r in measured),
                  "timeout": sum(r.get("status") == "timeout" for r in grp),
                  "not_run": sum(r.get("status") == "not_run" for r in grp),
                  "checkpoint_unavailable": sum(str(r.get("status", "")).startswith("checkpoint_")
                                                 for r in grp),
                  "label_failures": sum(str(r.get("status", "")).startswith("label_")
                                        or r.get("status") == "unsupported_short_trajectory" for r in grp)}
        counts["error"] = sum(r.get("status") not in ("ok", "numerical_rejected", "not_run", "timeout")
                              for r in grp)
        counts["failed"] = counts["error"]
        macros = {}
        for name, path in fields.items():
            vals = [_w1_de_field(r, path) for r in measured]
            vals = [float(v) for v in vals if _w1_de_finite(v)]
            mean = sum(vals) / len(vals) if vals else None
            conditional = path.startswith(("hands.", "feet.", "both.", "contact_f1."))
            eligible = len(vals)
            if path.startswith(("hands.", "feet.")):
                family = "ref_grasp_frames" if path.startswith("hands.") else "ref_support_frames"
                eligible = sum(_w1_de_count(_w1_de_field(r, f"{family}.{path.split('.')[1]}"))
                               and _w1_de_field(r, f"{family}.{path.split('.')[1]}") > 0 for r in grp)
            elif path.startswith("both."):
                eligible = sum(_w1_de_count(_w1_de_field(r, "both.denominator_frames"))
                               and _w1_de_field(r, "both.denominator_frames") > 0 for r in grp)
            macros[name] = {"expected": len(demos), "n_defined": len(vals),
                            "macro_all": mean if roster_ok and len(vals) == len(demos) else None,
                            "defined_only_mean": mean, "conditional": conditional,
                            "n_reference_defined": eligible if conditional else None,
                            "macro_defined": (mean if conditional and roster_ok and vals
                                              and len(vals) == eligible and all(
                                                  not w1_de_row_evidence_gaps(r) for r in grp) else None)}
        out[f"{method}|{seed}"] = {"matrix": scope, "counts": counts, "macros": macros,
                                   "roster_complete": roster_ok}
    return out


def load_w1_de_generator(selection: dict, device: str, ddim_steps: int,
                         cache_dir: Path, manifest_sha: str,
                         cache_ident_sha: str, root_stats_sha: str,
                         registration_sha: str, *, registration: dict | None = None,
                         project_root: Path = ROOT) -> "G1Generator":
    """Explicit SHA-bound checkpoint; complete config, strict state and exact stats."""
    import dataclasses

    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1
    from ireset.train.trainer import TrainConfig

    path = _w1_de_path(selection["path"], project_root)
    want_sha = selection.get("sha256")
    if (not isinstance(want_sha, str) or len(want_sha) != 64
            or integ.sha256_file(path) != want_sha):
        raise ValueError("checkpoint full sha256 mismatch")
    cache_dir = _w1_de_path(cache_dir, project_root)
    if (integ.sha256_file(cache_dir / "cache_identity.json") != cache_ident_sha
            or integ.sha256_file(cache_dir / "root_stats.npz") != root_stats_sha):
        raise ValueError("checkpoint current cache/root_stats hash mismatch")
    cache_ident = json.loads((cache_dir / "cache_identity.json").read_text())
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get("config"), dict) \
            or not isinstance(ck.get("identity"), dict):
        raise ValueError("checkpoint config/identity not mappings")
    cfg, ident = ck["config"], ck["identity"]
    if registration is None:
        reg_rec = ident.get("execution_registration")
        if not isinstance(reg_rec, dict) or not reg_rec.get("path"):
            raise ValueError("checkpoint registered config unavailable")
        reg_path = _w1_de_path(reg_rec["path"], project_root)
        if integ.sha256_file(reg_path) != registration_sha:
            raise ValueError("execution registration sha256 mismatch")
        registration = json.loads(reg_path.read_text())
    model_name = selection.get("model")
    if model_name not in ("deterministic", "diffusion"):
        raise ValueError("checkpoint explicit model type missing")
    registered_cfg = registration.get("configs", {}).get(model_name)
    problems = []
    if (not isinstance(registered_cfg, dict) or cfg != registered_cfg
            or set(cfg) != {f.name for f in dataclasses.fields(TrainConfig)}
            or ident.get("config") != cfg):
        problems.append("config_not_registered_full_config")
    for key, want in (("model_type", model_name),
                      ("protocol", w1.MOTION_PROTOCOL_VERSION),
                      ("experiment_id", w1.EXPERIMENT_ID),
                      ("training_purpose", w1.TRAINING_PURPOSE),
                      ("assets_sha256", cache_ident.get("assets_sha256")),
                      ("condition_code_sha256", w1.condition_code_sha256(project_root))):
        if ident.get(key) != want or want is None:
            problems.append(key)
    for path_key, want in (("training_manifest.sha256", manifest_sha),
                           ("cache.identity_sha256", cache_ident_sha),
                           ("cache.root_stats_sha256", root_stats_sha),
                           ("execution_registration.sha256", registration_sha)):
        if _w1_de_field(ident, path_key) != want:
            problems.append(path_key)
    if (_w1_de_field(ident, "execution_registration.execution_id") != "w1_de_v1"
            or registration.get("execution_id") != "w1_de_v1"
            or registration.get("label_protocol") != w1.MOTION_PROTOCOL_VERSION
            or registration.get("training_purpose") != w1.TRAINING_PURPOSE):
        problems.append("registration_protocol_purpose")
    if (registration.get("condition_code_sha256") != ident.get("condition_code_sha256")
            or _w1_de_field(registration, "training_manifest.sha256") != manifest_sha
            or _w1_de_field(registration, "cache.identity_sha256") != cache_ident_sha
            or _w1_de_field(registration, "cache.root_stats_sha256") != root_stats_sha
            or ident.get("config_file_sha256") != _w1_de_field(
                registration, "configs.config_file_sha256")):
        problems.append("registration_content_binding")
    if (not _w1_de_count(ck.get("step")) or not _w1_de_count(selection.get("step"))
            or ck["step"] != selection["step"]):
        problems.append("step")
    role = selection.get("checkpoint_role")
    if (selection.get("role"), role) not in (("best", "best"), ("last", "last_diagnostic")):
        problems.append("checkpoint_role")
    if cfg.get("model_type") != model_name or ddim_steps != W1_DE_SAMPLING["ddim_steps"]:
        problems.append("sampling_model_type")
    if problems:
        raise ValueError(f"checkpoint identity problems: {problems}")
    # Check buffers as well as parameters, before any GPU or model call.
    for part in ("model", "conditioner"):
        state = ck.get(part)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"checkpoint {part} empty/not a state mapping")
        if not all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
                   for v in state.values()):
            raise ValueError(f"checkpoint {part} non-finite/non-tensor state")
    bw = {k: cfg[k] for k in ("d_model", "n_layers", "n_heads", "ffn_mult", "time_window")}
    model = (DiffusionG1(num_train_timesteps=cfg["num_train_timesteps"],
                         beta_schedule=cfg["beta_schedule"],
                         beta_start=cfg["beta_start"], beta_end=cfg["beta_end"], **bw)
             if model_name == "diffusion" else DeterministicG1(**bw))
    model.load_state_dict(ck["model"], strict=True)
    cond = ConditionEncoder(cfg["d_model"])
    cond.load_state_dict(ck["conditioner"], strict=True)
    with np.load(cache_dir / "root_stats.npz", allow_pickle=False) as rs:
        for name in ("mean", "std"):
            a, b = np.asarray(ck["root_stats"][name]), rs[name]
            if (a.shape != (3,) or a.dtype != b.dtype or a.shape != b.shape
                    or a.tobytes() != b.tobytes() or not np.isfinite(a).all()
                    or (name == "std" and not (a > 0).all())):
                raise ValueError(f"checkpoint embedded root stats differ: {name}")
    tokenizer = make_tokenizer(RootStats(mean=np.asarray(ck["root_stats"]["mean"]),
                                         std=np.asarray(ck["root_stats"]["std"])))
    gen = G1Generator(model, cond, tokenizer, device=device,
                      window=60, stride=30, ddim_steps=ddim_steps, resample_attempts=3)
    gen.w1_de_checkpoint = {"path": _w1_de_record_path(path, project_root),
                            "sha256": want_sha, "step": ck["step"], "role": role,
                            "formal_attempt": selection.get("formal_attempt"),
                            "config_sha256": _w1_de_digest(cfg),
                            "identity": ident}
    return gen


def w1_de_frame_series(replayer, scene: np.ndarray, human: np.ndarray,
                       target_box: np.ndarray, *, deadline: float | None = None) -> dict:
    """Finite-first frame-aligned geometry on the locked validator scene."""
    scene, human, target_box = map(np.asarray, (scene, human, target_box))
    if (scene.ndim != 2 or scene.shape[1] != 43 or len(scene) == 0
            or human.shape != (len(scene), 52, 3)
            or target_box.shape != (len(scene), 7)
            or not all(np.isfinite(a).all() for a in (scene, human, target_box))
            or not np.array_equal(scene[:, 36:43], target_box)
            or (np.linalg.norm(scene[:, 3:7], axis=1) < 1e-8).any()
            or (np.linalg.norm(target_box[:, 3:7], axis=1) < 1e-8).any()):
        raise ValueError("unsafe scene/reference passed to geometry")
    T = scene.shape[0]
    r = replayer
    pen_key = np.zeros(T)
    box_ground = np.zeros(T)
    gap = {"l": np.zeros(T), "r": np.zeros(T)}
    kpt_err_w = np.zeros(T)
    kpt_err_b = np.zeros(T)
    robot_kpts = np.zeros((T, len(r.kpt_ids), 3))
    active_pairs = [(a, b) for kind in ("robot_ground", "robot_box", "self")
                    for (a, b) in r.pairs_by_kind.get(kind, [])]
    for t in range(T):
        _w1_de_deadline(deadline, "frame_geometry")
        r._forward(scene, t)
        worst = 0.0
        for ga, gb in active_pairs:
            worst = max(worst, -r._dist(ga, gb))
        pen_key[t] = worst
        box_ground[t] = -r._dist(r.box_geom, r.ground_geom)
        for s in ("l", "r"):
            gap[s][t] = min(r._dist(g, r.box_geom) for g in r._hand_geoms[s])
        kpt_w = r.data.site_xpos[r.kpt_ids].copy()
        robot_kpts[t] = kpt_w
        ref_w = human[t][r.ref_kpt_idx]
        kpt_err_w[t] = float(np.linalg.norm(kpt_w - ref_w, axis=-1).mean())
        kb = world_to_box_geom(kpt_w[None], target_box[t:t + 1])[0]
        rb = world_to_box_geom(ref_w[None], target_box[t:t + 1])[0]
        kpt_err_b[t] = float(np.linalg.norm(kb - rb, axis=-1).mean())
    if not all(np.isfinite(a).all() for a in
               (pen_key, box_ground, gap["l"], gap["r"], kpt_err_w, kpt_err_b, robot_kpts)):
        raise ValueError("non-finite geometry measurements")
    return {"pen_key": pen_key, "box_ground": box_ground, "gap": gap,
            "kpt_err_w": kpt_err_w, "kpt_err_b": kpt_err_b,
            "robot_kpts": robot_kpts}


def _w1_de_report_metrics(report) -> dict:
    checks = {k: {"passed": bool(v.passed), "worst": _w1_de_json(v.worst),
                  "limit": _w1_de_json(v.limit), "location": v.location,
                  "detail": v.detail, "first_failure_frame": v.first_failure_frame,
                  "evidence": _w1_de_json(v.evidence)}
              for k, v in report.checks.items()}
    failed = sorted(k for k, v in checks.items() if not v["passed"])
    first = [v["first_failure_frame"] for v in checks.values()
             if not v["passed"] and v["first_failure_frame"] is not None]
    return {"status": report.status,
            "passed": bool(report.passed and report.status == "passed" and checks
                           and not failed and all(_w1_de_finite(v["worst"])
                                                  and _w1_de_finite(v["limit"])
                                                  for v in checks.values())),
            "failed_checks": failed, "checks": checks,
            "max_check_interval_m": _w1_de_json(report.max_check_interval_m),
            "subdivision_capped": bool(report.subdivision_capped),
            "first_failure_frame": min(first) if first else None,
            "excluded_pair_kinds": list(W1_DE_EXCLUDE)}


def _w1_de_unmeasured(reference: dict, reason: str, report=None) -> dict:
    """Preserve nonzero reference denominators when no safe geometry is available."""
    both = {"denominator_frames": reference["both_reference"]["denominator_frames"],
            "valid_coverage": None, "gap_series_p95_m": None,
            "longest_consecutive_valid": None, "certified_overlap_frames": None,
            "certified_coverage": None, "longest_certified_overlap": None,
            "reason": reason}
    return {**reference, "fitting": None, "task": None, "penetration": None,
            "box_ground": None, "hands": {"l": None, "r": None},
            "feet": {"l": None, "r": None}, "both": both,
            "contact_f1": {e: None for e in
                           ("left_hand", "right_hand", "left_foot", "right_foot")}
                          | {"n_defined": 0, "macro_f1": None, "reason": reason},
            "robot_strict": _w1_de_report_metrics(report) if report is not None else None,
            "robot_strict_pass": False, "task_gate_pass": None,
            "raw_standard_pass": False}


def w1_de_row_metrics(demo, scene, cond, teacher_scene, teacher_kpts_w,
                      series, replayer, validator, task_tol_m, *, report=None,
                      deadline: float | None = None) -> dict:
    """Measured rows include all checks, exact references and certified intersections."""
    from ireset.eval.teacher_comparison import (
        HAND_VALID_GAP_M, PENETRATION_TOL_M, _frames_of, _mask_from_ranges,
        _longest_true_run, _p95, certified_robot_mask, foot_side_metrics, hand_side_metrics,
    )

    T, cfg = len(scene), validator.cfg
    reference = w1_de_reference_facts(cond, T)
    _w1_de_deadline(deadline, "independent_validator")
    rep = report if report is not None else validator.validate(scene, exclude=W1_DE_EXCLUDE)
    _w1_de_deadline(deadline, "validator_result")
    # Pre-gates return no schedule/cache. A previous trajectory's cache is never evidence.
    if rep.schedule is None:
        return {**_w1_de_unmeasured(reference, rep.status, rep),
                "measurements_status": "validator_pre_gate",
                "measurement_gaps": [f"validator_pre_gate:{rep.status}"]}
    cache = getattr(validator, "_cache", None)
    if not isinstance(cache, dict):
        raise ValueError("validator measurement cache missing")
    expected_arrays = {"sole_pts": (T, 5, 3), "sole_gap": (T,),
                       "hand_local": (T, 3, 3)}
    for family, shape in expected_arrays.items():
        for side in ("l", "r"):
            a = (cache.get(family) or {}).get(side)
            if not isinstance(a, np.ndarray) or a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"validator cache invalid:{family}:{side}")
    arrays = [scene, teacher_scene, teacher_kpts_w,
              series["robot_kpts"], series["kpt_err_w"], series["kpt_err_b"],
              series["pen_key"], series["box_ground"], series["gap"]["l"], series["gap"]["r"]]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("non-finite fitting/task/contact measurement inputs")
    box, fps = cond.target_obj_pose[:T], 1.0 / cond.frame_dt
    fit_per_frame = np.linalg.norm(
        world_to_box_geom(series["robot_kpts"], box)
        - world_to_box_geom(teacher_kpts_w, box), axis=-1).mean(1)
    w_err, b_err = series["kpt_err_w"], series["kpt_err_b"]
    over = np.flatnonzero(w_err > task_tol_m)
    task = {"world_mpjpe_mean_m": float(w_err.mean()),
            "world_mpjpe_p95_m": _p95(w_err), "world_mpjpe_max_m": float(w_err.max()),
            "box_mpjpe_mean_m": float(b_err.mean()), "box_mpjpe_p95_m": _p95(b_err),
            "box_mpjpe_max_m": float(b_err.max()), "gate_frame": "world",
            "tol_m": task_tol_m, "gate_pass": bool(not len(over)),
            "gate_first_exceedance_frame": int(over[0]) if len(over) else None,
            "gate_failure_frames": over.tolist()}
    pen_valid, bg = np.maximum(series["pen_key"], 0.0), np.maximum(series["box_ground"], 0.0)
    bg_over = np.flatnonzero(bg > PENETRATION_TOL_M)
    box_ground = {"penetration_max_m": float(bg.max()),
                  "violation_ratio": float((bg > PENETRATION_TOL_M).mean()),
                  "worst_frame": int(np.argmax(bg)),
                  "worst_box_position_world_m": box[int(np.argmax(bg)), :3].tolist(),
                  "violation_frames": bg_over.tolist(), "excluded_from_strict": True}
    feet, hands, cert = {}, {}, {}
    ref_sched = cond.schedule
    for s, eff in (("l", "left_foot"), ("r", "right_foot")):
        _w1_de_deadline(deadline, "foot_metrics")
        ranges = _frames_of(ref_sched, eff, "ground")
        feet[s] = (foot_side_metrics(cache["sole_pts"][s], cache["sole_gap"][s],
                                      ranges, fps).to_json() if ranges else None)
        if feet[s] is not None and feet[s]["n_speed_frames"] == 0:
            feet[s]["speed_undefined_reason"] = "no_reference_frame_after_first_frame"
    for s, eff in (("l", "left_hand"), ("r", "right_hand")):
        _w1_de_deadline(deadline, "hand_metrics")
        ranges = _frames_of(ref_sched, eff, "object")
        robot_intervals = [iv for iv in rep.schedule.intervals if iv.effector == eff]
        cert[s], _, _ = certified_robot_mask(robot_intervals, series["gap"][s],
                                             cache["hand_local"][s], cfg)
        hands[s] = (hand_side_metrics(series["gap"][s], cache["hand_local"][s],
                                      robot_intervals, ranges, cfg, pen_valid).to_json()
                    if ranges else None)
        if hands[s] is not None:
            mask = _mask_from_ranges(ranges, T)
            speed = np.zeros(T)
            speed[1:] = np.linalg.norm(np.diff(cache["hand_local"][s], axis=0), axis=-1).max(1) * fps
            speed_mask = mask.copy(); speed_mask[0] = False
            hands[s].update(near_frames=int((series["gap"][s] < HAND_VALID_GAP_M).sum()),
                            certified_frames=int(cert[s].sum()),
                            certified_coverage=float((cert[s] & mask).sum() / mask.sum()),
                            relative_speed_max_mps=float(speed[speed_mask].max()) if speed_mask.any() else 0.0,
                            relative_speed_p95_mps=_p95(speed[speed_mask]) if speed_mask.any() else None,
                            certified_overlap_frame_indices=np.flatnonzero(cert[s] & mask).tolist())
    ml = _mask_from_ranges(_frames_of(ref_sched, "left_hand", "object"), T)
    mr = _mask_from_ranges(_frames_of(ref_sched, "right_hand", "object"), T)
    both_ref = ml & mr
    denominator = int(both_ref.sum())
    if denominator:
        gl, gr = np.maximum(series["gap"]["l"], 0.0), np.maximum(series["gap"]["r"], 0.0)
        near_valid = ((gl < HAND_VALID_GAP_M) & (gr < HAND_VALID_GAP_M)
                      & (pen_valid <= PENETRATION_TOL_M) & both_ref)
        certified = cert["l"] & cert["r"] & both_ref
        both = {"denominator_frames": denominator,
                "valid_coverage": float(near_valid.sum() / denominator),
                "gap_series_p95_m": _p95(np.maximum(gl, gr)[both_ref]),
                "longest_consecutive_valid": _longest_true_run(near_valid),
                "certified_overlap_frames": int(certified.sum()),
                "certified_coverage": float(certified.sum() / denominator),
                "longest_certified_overlap": _longest_true_run(certified),
                "certified_overlap_frame_indices": np.flatnonzero(certified).tolist(),
                "valid_frame_indices": np.flatnonzero(near_valid).tolist(),
                "reason": None}
    else:
        both = {"denominator_frames": 0, "valid_coverage": None,
                "gap_series_p95_m": None, "longest_consecutive_valid": 0,
                "certified_overlap_frames": 0, "certified_coverage": None,
                "longest_certified_overlap": 0, "certified_overlap_frame_indices": [],
                "valid_frame_indices": [], "reason": "no_both_grasp_reference_frames"}
    f1 = contact_f1_all(rep.schedule, ref_sched, T)
    f1["undefined_reasons"] = {e: "neither_reference_nor_robot_contact"
                                for e in ("left_hand", "right_hand", "left_foot", "right_foot")
                                if f1[e] is None}
    strict = _w1_de_report_metrics(rep)
    gaps = []
    if set(strict["checks"]) != set(W1_DE_ROBOT_CHECKS):
        gaps.append("validator_checks_incomplete")
    for k, v in strict["checks"].items():
        if not _w1_de_finite(v["worst"]) or not _w1_de_finite(v["limit"]):
            gaps.append(f"validator_measurement_nonfinite:{k}")
    pen_check = rep.checks.get("penetration")
    series["teacher_fit_per_frame"] = fit_per_frame
    series["certified_overlap_l"] = cert["l"] & ml
    series["certified_overlap_r"] = cert["r"] & mr
    series["both_reference_mask"] = both_ref
    series["both_certified_overlap"] = cert["l"] & cert["r"] & both_ref
    return {**reference,
            "fitting": {"root_rmse_m": float(np.sqrt(((scene[:, :3] - teacher_scene[:, :3]) ** 2).sum(1).mean())),
                        "root_geodesic_deg": float(geodesic_deg(scene[:, 3:7], teacher_scene[:, 3:7]).mean()),
                        "joint_mae_rad": float(np.abs(scene[:, 7:36] - teacher_scene[:, 7:36]).mean()),
                        "teacher_fit_fk_mpjpe_box_m": float(fit_per_frame.mean())},
            "task": task, "robot_strict": strict,
            "penetration": {"max_m": _w1_de_json(pen_check.worst) if pen_check else None,
                            "keyframe_violation_ratio": float((pen_valid > PENETRATION_TOL_M).mean()),
                            "protocol": "validator_adaptive_slerp (ratio: keyframes only)"},
            "box_ground": box_ground, "feet": feet, "hands": hands, "both": both,
            "contact_f1": f1, "measurements_status": "measured", "measurement_gaps": gaps,
            "robot_strict_pass": strict["passed"], "task_gate_pass": task["gate_pass"],
            "raw_standard_pass": bool(strict["passed"] and task["gate_pass"] and not gaps)}


def _w1_de_numerical_validity(traj, scene, cond, T: int) -> tuple[bool, dict]:
    """No rotation/geometry call until shapes and finiteness are established."""
    from ireset.model.stitch import window_starts

    root, q = np.asarray(traj.root), np.asarray(traj.q)
    metadata = traj.metadata if isinstance(traj.metadata, dict) else {}
    root_shape, joint_shape = root.shape == (T, 7), q.shape == (T, 29)
    target = np.asarray(cond.target_obj_pose)
    target_ok = target.shape == (T, 7) and np.isfinite(target).all()
    finite = bool(root_shape and joint_shape and np.isfinite(root).all() and np.isfinite(q).all())
    starts = metadata.get("starts")
    teacher = metadata.get("teacher_raw") is True
    starts_ok = (starts == [] if teacher else starts == window_starts(T, 60, 30))
    final_bad = metadata.get("final_invalid_windows", metadata.get("degenerate_windows"))
    facts = {"root_shape": root_shape, "joint_shape": joint_shape,
             "length_matches_T": root_shape and joint_shape, "all_finite": finite,
             "root_quat_norm_ok": bool(finite and np.all(
                 np.abs(np.linalg.norm(root[:, 3:7], axis=1) - 1) <= 1e-5)),
             "target_shape_finite": bool(target_ok),
             "target_quat_norm_ok": bool(target_ok and np.all(
                 np.abs(np.linalg.norm(target[:, 3:7], axis=1) - 1) <= 1e-4)),
             "box_binding_bitwise": bool(scene is not None and np.shape(scene) == (T, 43)
                                          and np.array_equal(scene[:, 36:43], target)),
             "window_starts_registered": bool(starts_ok),
             "covers_last_frame": bool(root_shape and starts_ok and (teacher or starts[-1] + 60 == T)),
             "frame_dt_matches_condition": bool(_w1_de_finite(traj.frame_dt)
                                                 and traj.frame_dt > 0 and traj.frame_dt == cond.frame_dt),
             "no_final_invalid_windows": bool(_w1_de_count(final_bad) and final_bad == 0)}
    return all(facts.values()), facts


def _w1_de_stitch_metrics(traj, T: int) -> dict | None:
    if traj is None:
        return None
    root, q = np.asarray(traj.root), np.asarray(traj.q)
    if (root.shape != (T, 7) or q.shape != (T, 29)
            or not np.isfinite(root).all() or not np.isfinite(q).all()
            or (np.linalg.norm(root[:, 3:7], axis=1) < 1e-8).any()):
        return {"measured": False, "reason": "numerically_invalid_trajectory"}
    starts = list(traj.metadata.get("starts", []))
    boundary = [s for s in starts[1:] if 0 < s < T]
    dp = np.linalg.norm(np.diff(root[:, :3], axis=0), axis=1)
    dr = geodesic_deg(root[:-1, 3:7], root[1:, 3:7]) if T > 1 else np.empty(0)
    dq = np.abs(np.diff(q, axis=0)).max(1)
    inside = np.ones(max(T - 1, 0), dtype=bool)
    if boundary:
        inside[np.asarray(boundary) - 1] = False
    disp = np.linalg.norm(root[:, :3] - root[0, :3], axis=1)
    dt = traj.frame_dt
    teacher = traj.metadata.get("teacher_raw") is True
    return {"measured": True, "window_starts": starts, "n_windows": len(starts),
            "covers_last_frame": bool(root.shape[0] == T and (teacher or starts and starts[-1] + 60 == T)),
            "boundary_position_jump_m": [float(dp[s - 1]) for s in boundary],
            "boundary_rotation_deg": [float(dr[s - 1]) for s in boundary],
            "boundary_joint_max_abs_rad": [float(dq[s - 1]) for s in boundary],
            "interior_root_speed_p95_mps": float(np.percentile(dp[inside] / dt, 95)) if inside.any() else None,
            "interior_angular_speed_p95_degps": float(np.percentile(dr[inside] / dt, 95)) if inside.any() else None,
            "interior_joint_speed_p95_radps": float(np.percentile(dq[inside] / dt, 95)) if inside.any() else None,
            "root_path_length_m": float(dp.sum()),
            "net_root_drift_m": float(disp[-1]), "max_root_excursion_m": float(disp.max()),
            "seam_diag": _w1_de_json(traj.metadata.get("boundary"))}


def _w1_de_retry(metadata: dict) -> dict:
    return {"model_calls": metadata.get("model_calls"),
            "resample_attempts_used": metadata.get("resample_attempts_used"),
            "attempts_max": metadata.get("resample_attempts_max"),
            "per_attempt_invalid_windows": metadata.get("per_attempt_invalid_windows"),
            "per_attempt_invalid_window_indices": metadata.get("per_attempt_invalid_window_indices"),
            "attempt_seeds": metadata.get("attempt_seeds"),
            "final_invalid_windows": metadata.get("final_invalid_windows", metadata.get("degenerate_windows")),
            "final_invalid_window_indices": metadata.get("final_invalid_window_indices"),
            "attempts": metadata.get("attempts")}


def _w1_de_retry_problems(retry: dict, method: str, seed, n_windows: int) -> list[str]:
    if not isinstance(retry, dict):
        return ["retry_not_a_mapping"]
    calls = retry.get("model_calls")
    if not _w1_de_count(calls):
        return ["model_calls_missing_or_invalid"]
    if method == "teacher":
        return [] if calls == 0 and retry.get("final_invalid_windows") == 0 \
            and retry.get("attempt_seeds") == [] else ["teacher_retry_not_zero"]
    maximum = 1 if method == "deterministic" else 3
    if not 1 <= calls <= maximum:
        return ["model_calls_out_of_bounds"]
    problems = []
    if retry.get("resample_attempts_used") != calls or retry.get("attempts_max") != 3:
        problems.append("retry_attempt_count_mismatch")
    if retry.get("attempt_seeds") != [seed + i for i in range(int(calls))]:
        problems.append("retry_seeds_mismatch")
    counts, indices = retry.get("per_attempt_invalid_windows"), retry.get("per_attempt_invalid_window_indices")
    if not isinstance(counts, list) or len(counts) != calls or any(
            not _w1_de_count(c) or c > n_windows for c in counts):
        problems.append("retry_invalid_counts_missing")
    if not isinstance(indices, list) or len(indices) != calls:
        problems.append("retry_invalid_indices_missing")
    elif isinstance(counts, list) and len(counts) == calls:
        previous = set(range(n_windows))
        for c, ix in zip(counts, indices):
            if (not isinstance(ix, list) or any(not _w1_de_count(v) or v >= n_windows for v in ix)
                    or len(set(ix)) != len(ix) or len(ix) != c or not set(ix) <= previous):
                problems.append("retry_indices_not_current_invalid_subset")
                break
            previous = set(ix)
        if retry.get("final_invalid_window_indices") != indices[-1]:
            problems.append("retry_final_indices_mismatch")
    if isinstance(counts, list) and counts and retry.get("final_invalid_windows") != counts[-1]:
        problems.append("retry_final_count_mismatch")
    return problems


def w1_de_product_identity(row: dict, registration: dict, row_record: dict,
                            condition, checkpoint: dict | None,
                            registration_sha: str, cache_identity: dict) -> dict:
    """Content identity sufficient to bind a replay to inputs and checkpoint."""
    from ireset.data import w1_training as w1

    schedule = next((r for r in cache_identity["schedule_rows"] if r["demo"] == row["demo"]), None)
    if schedule is None:
        raise ValueError("registered schedule identity missing")
    return {"schema": W1_DE_PRODUCT_SCHEMA, "layout": "xyz_wxyz_robot36_box7",
            "key": row["key"], "method": row["method"], "demo": row["demo"], "seed": row["seed"],
            "T": condition.T, "protocol": w1.MOTION_PROTOCOL_VERSION,
            "training_purpose": w1.TRAINING_PURPOSE, "execution_id": "w1_de_v1",
            "execution_registration_sha256": registration_sha,
            "training_manifest_sha256": registration["training_manifest"]["sha256"],
            "cache_identity_sha256": registration["cache"]["identity_sha256"],
            "root_stats_sha256": registration["cache"]["root_stats_sha256"],
            "assets_sha256": cache_identity["assets_sha256"],
            "condition_code_sha256": registration["condition_code_sha256"],
            "inputs": row_record["inputs"], "derived": row_record["derived"],
            "schedule": schedule, "checkpoint": checkpoint,
            "sampling": registration["evaluation"]["sampling"],
            "condition": {"human_sha256": _w1_de_array_digest(condition.human_kps_world),
                          "source_box_sha256": _w1_de_array_digest(condition.source_obj_pose),
                          "target_box_sha256": _w1_de_array_digest(condition.target_obj_pose),
                          "world_from_clip_sha256": _w1_de_array_digest(condition.world_from_clip),
                          "reference_sha256": _w1_de_digest(w1_de_reference_facts(condition))}}


def save_w1_de_product(path: Path, row: dict, identity: dict, condition, *,
                       trajectory=None, output=None, metadata: dict | None = None,
                       project_root: Path = ROOT) -> dict:
    """Save available arrays BEFORE any unsafe decode, scipy or MuJoCo call."""
    from ireset.data import a2_integrity as integ

    metadata = metadata if isinstance(metadata, dict) else {}
    arrays = {"target_box": np.asarray(condition.target_obj_pose),
              "frame_dt": np.float64(condition.frame_dt), "fps": np.float64(1 / condition.frame_dt),
              "window_starts": np.asarray(metadata.get("starts", []), dtype=np.int64)}
    kind = "trajectory" if trajectory is not None else "numerical_evidence"
    if trajectory is not None:
        arrays.update(root=np.asarray(trajectory.root), joints=np.asarray(trajectory.q))
    elif isinstance(output, np.ndarray) or isinstance(output, torch.Tensor):
        arrays["normalized_windows"] = (output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else output)
    elif isinstance(output, (tuple, list)) and len(output) in (2, 3):
        names = ("root", "joints") if len(output) == 2 else ("decoded_pos", "decoded_quat", "decoded_joints")
        arrays.update({k: np.asarray(v) for k, v in zip(names, output)})
    elif isinstance(output, dict):
        arrays.update({f"evidence_{k}": np.asarray(v) for k, v in output.items()
                       if isinstance(v, (np.ndarray, torch.Tensor))})
    root, joints = arrays.get("root"), arrays.get("joints")
    if root is not None and joints is not None and root.shape == (condition.T, 7) \
            and joints.shape == (condition.T, 29):
        arrays["scene"] = np.concatenate([root, joints, arrays["target_box"]], axis=1)
    arrays.update(product_kind=np.asarray(kind), method=np.asarray(row["method"]),
                  demo=np.asarray(row["demo"]), seed=np.int64(-1 if row["seed"] is None else row["seed"]),
                  generated=np.bool_(row["method"] != "teacher"),
                  covers_last_frame=np.bool_(root is not None and root.shape == (condition.T, 7)
                                              and (row["method"] == "teacher" or metadata.get("last_frame_covered") is True)),
                  identity_json=np.asarray(json.dumps(_w1_de_json(identity), sort_keys=True, allow_nan=False)),
                  metadata_json=np.asarray(json.dumps(_w1_de_json(metadata), sort_keys=True, allow_nan=False)),
                  retry_json=np.asarray(json.dumps(_w1_de_json(_w1_de_retry(metadata)), sort_keys=True, allow_nan=False)),
                  manifest_sha256=np.asarray(identity["training_manifest_sha256"]),
                  registration_sha256=np.asarray(identity["execution_registration_sha256"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    return {"path": _w1_de_record_path(path, project_root), "sha256": integ.sha256_file(path),
            "kind": kind, "identity": identity}


def verify_w1_de_product(path, row: dict, registration: dict, condition=None, *,
                          project_root: Path = ROOT, expected_identity: dict | None = None) -> dict:
    """Hash, layout and condition identity; invalid numeric arrays remain evidence."""
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1

    arrays, metadata, problems = None, {}, []
    try:
        path = _w1_de_path(path, project_root)
        product = row.get("product")
        if not isinstance(product, dict) or integ.sha256_file(path) != product.get("sha256"):
            problems.append("product_sha256_mismatch")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        identity = json.loads(str(arrays["identity_json"].item()))
        metadata = json.loads(str(arrays["metadata_json"].item()))
        retry = json.loads(str(arrays["retry_json"].item()))
        if not isinstance(identity, dict) or not isinstance(metadata, dict) or not isinstance(retry, dict):
            raise ValueError("product JSON evidence not mappings")
        if product.get("identity") != identity:
            problems.append("product_identity_mismatch")
        if expected_identity is not None:
            if expected_identity.get("schema") == W1_DE_PRODUCT_SCHEMA:
                if identity != expected_identity:
                    problems.append("product_expected_identity_mismatch")
            elif expected_identity == {"method": "teacher_raw"}:
                if identity.get("method") != "teacher" or identity.get("checkpoint") is not None:
                    problems.append("product_teacher_identity_mismatch")
            elif not isinstance(identity.get("checkpoint"), dict) or any(
                    identity["checkpoint"].get(k) != v for k, v in expected_identity.items()):
                problems.append("product_checkpoint_identity_mismatch")
        for field, want in (("schema", W1_DE_PRODUCT_SCHEMA), ("protocol", w1.MOTION_PROTOCOL_VERSION),
                             ("training_purpose", w1.TRAINING_PURPOSE), ("execution_id", "w1_de_v1"),
                             ("layout", "xyz_wxyz_robot36_box7"), ("key", row["key"]),
                             ("method", row["method"]), ("demo", row["demo"]), ("seed", row["seed"]),
                             ("training_manifest_sha256", registration["training_manifest"]["sha256"]),
                             ("cache_identity_sha256", registration["cache"]["identity_sha256"]),
                             ("root_stats_sha256", registration["cache"]["root_stats_sha256"]),
                             ("condition_code_sha256", registration["condition_code_sha256"]),
                             ("sampling", registration["evaluation"]["sampling"])):
            if identity.get(field) != want:
                problems.append(f"product_identity:{field}")
        reg_path = _w1_de_path(_w1_de_field(row, "execution_registration.path")
                                or _w1_de_field(registration, "registration.path")
                                or _w1_de_field(identity, "registration_path")
                                or "", project_root)
        # Registration hash is anchored by the caller's current RUN, never the product's path.
        if reg_path.is_file() and identity.get("execution_registration_sha256") != integ.sha256_file(reg_path):
            problems.append("product_registration_sha256_mismatch")
        if identity.get("assets_sha256") != registration.get("assets_sha256", identity.get("assets_sha256")):
            problems.append("product_assets_mismatch")
        if str(arrays["method"].item()) != row["method"] or str(arrays["demo"].item()) != row["demo"] \
                or int(arrays["seed"].item()) != (-1 if row["seed"] is None else row["seed"]):
            problems.append("product_key_mismatch")
        if row.get("retry") is not None and row["retry"] != retry:
            problems.append("product_retry_mismatch")
        if retry != _w1_de_retry(metadata):
            problems.append("product_retry_metadata_mismatch")
        T = identity.get("T")
        if not _w1_de_count(T) or T < 1 or (row.get("T") is not None and row["T"] != T):
            problems.append("product_T_invalid")
        if not _w1_de_finite(float(arrays["frame_dt"])) or float(arrays["frame_dt"]) <= 0 \
                or float(arrays["fps"]) != 1 / float(arrays["frame_dt"]):
            problems.append("product_fps_frame_dt_invalid")
        if arrays["target_box"].shape != (T, 7):
            problems.append("product_target_box_shape")
        if "scene" in arrays:
            if not all(k in arrays for k in ("root", "joints")):
                problems.append("product_scene_without_robot_arrays")
            else:
                reconstructed = np.concatenate([arrays["root"], arrays["joints"], arrays["target_box"]], axis=1)
                if arrays["scene"].dtype != reconstructed.dtype or arrays["scene"].shape != reconstructed.shape \
                        or arrays["scene"].tobytes() != reconstructed.tobytes():
                    problems.append("product_scene_not_exact_assembly")
        if condition is not None:
            if condition.T != T or float(arrays["frame_dt"]) != condition.frame_dt:
                problems.append("product_condition_length_or_dt")
            hashes = {"human_sha256": _w1_de_array_digest(condition.human_kps_world),
                      "source_box_sha256": _w1_de_array_digest(condition.source_obj_pose),
                      "target_box_sha256": _w1_de_array_digest(condition.target_obj_pose),
                      "world_from_clip_sha256": _w1_de_array_digest(condition.world_from_clip),
                      "reference_sha256": _w1_de_digest(w1_de_reference_facts(condition))}
            if identity.get("condition") != hashes \
                    or not np.array_equal(arrays["target_box"], condition.target_obj_pose):
                problems.append("product_condition_binding")
        n_windows = len(arrays["window_starts"])
        problems += _w1_de_retry_problems(retry, row["method"], row["seed"], n_windows)
        if arrays["window_starts"].tolist() != metadata.get("starts", []):
            problems.append("product_window_starts_mismatch")
        if not any(k in arrays for k in ("root", "normalized_windows", "decoded_pos")):
            problems.append("product_no_output_evidence")
    except (Exception, SystemExit) as exc:
        problems.append(f"product_unreadable:{type(exc).__name__}:{exc}")
    return {"arrays": arrays, "metadata": metadata, "problems": sorted(set(problems)),
            "passed": not problems}


def _w1_de_window_rejection(arrays: dict, metadata: dict, T: int) -> tuple[bool, dict]:
    """Recompute failure on normalized or decoded arrays, without decoding them."""
    output = arrays.get("normalized_windows")
    facts = {"normalized_window_shape": True, "normalized_windows_finite": True,
             "normalized_rotation_valid": True}
    if output is not None:
        from ireset.model.stitch import window_starts
        shape_ok = output.shape == (len(window_starts(T, 60, 30)), 60, 38)
        facts["normalized_window_shape"] = shape_ok
        facts["normalized_windows_finite"] = bool(np.isfinite(output).all())
        if shape_ok:
            from ireset.model.generator import _valid_windows
            valid = _valid_windows(torch.as_tensor(output))
            facts["normalized_rotation_valid"] = bool(valid.all())
            if metadata.get("final_invalid_windows") != int((~valid).sum()) \
                    or metadata.get("final_invalid_window_indices") != np.flatnonzero(~valid).tolist():
                facts["final_invalid_indices_match_output"] = False
        rejected = any(v is False for k, v in facts.items()
                       if k != "final_invalid_indices_match_output")
        return rejected and facts.get("final_invalid_indices_match_output") is not False, facts
    if all(k in arrays for k in ("decoded_pos", "decoded_quat", "decoded_joints")):
        facts = {"decoded_arrays_finite": all(np.isfinite(arrays[k]).all()
                                              for k in ("decoded_pos", "decoded_quat", "decoded_joints"))}
        quat = arrays["decoded_quat"]
        facts["decoded_rotation_valid"] = bool(quat.shape[-1:] == (4,) and np.isfinite(quat).all()
                                              and np.all(np.abs(np.linalg.norm(quat, axis=-1) - 1) <= 1e-5))
        return not all(facts.values()), facts
    return False, {"numerical_output_evidence_available": False}


def recompute_w1_de_product(path, row: dict, registration: dict, clip, validator, replayer, *,
                             project_root: Path = ROOT, expected_identity: dict | None = None,
                             deadline: float | None = None) -> dict:
    """Closure replay: metrics/flags come from NPZ and frozen inputs, not row claims."""
    from ireset.data.schema import RobotTrajectory

    cond, T = clip.condition, clip.T
    verification = verify_w1_de_product(path, row, registration, cond,
                                         project_root=project_root, expected_identity=expected_identity)
    base = {k: row[k] for k in ("key", "method", "demo", "seed", "matrix")}
    reference = w1_de_reference_facts(cond)
    out = {"verification": verification, "series": None, "report": None}
    if not verification["passed"]:
        out["row"] = {**base, "T": T, "status": "product_invalid",
                      "measurement_gaps": verification["problems"]}
        return out
    arrays, metadata = verification["arrays"], verification["metadata"]
    retry = _w1_de_retry(metadata)
    root, joints = arrays.get("root"), arrays.get("joints")
    if row["method"] == "teacher" and (root is None or joints is None
                                       or not np.array_equal(root, clip.root)
                                       or not np.array_equal(joints, clip.q)):
        # The FK self-reference shortcut below is only sound for the frozen
        # label itself; a teacher product that deviates from clip.root/clip.q
        # must never measure its own deviation away.
        verification["problems"].append("teacher_product_not_bitwise_frozen_label")
        verification["passed"] = False
        out["row"] = {**base, "T": T, "status": "product_invalid",
                      "measurement_gaps": verification["problems"]}
        return out
    scene = arrays.get("scene")
    traj = (RobotTrajectory(root, joints, float(arrays["frame_dt"]), metadata)
            if root is not None and joints is not None else None)
    if traj is not None:
        numerical_valid, facts = _w1_de_numerical_validity(traj, scene, cond, T)
        numerical_evidence = not numerical_valid and any(v is False for v in facts.values())
    else:
        numerical_evidence, facts = _w1_de_window_rejection(arrays, metadata, T)
        numerical_valid = False
    shared = {**base, "T": T, "fps": 1 / cond.frame_dt, "attempted": True, "retry": retry,
              "retry_evidence_problems": [], "product": row["product"], "product_verified": True,
              "numerical_valid": numerical_valid, "numerical_facts": facts,
              "numerical_evidence_verified": bool(numerical_evidence),
              "self_reference": row["method"] == "teacher",
              "covers_last_frame": bool(facts.get("covers_last_frame", False)),
              "windows": len(metadata.get("starts", []))}
    if not numerical_valid:
        out["row"] = {**shared, **_w1_de_unmeasured(reference, "numerical_rejected"),
                      "status": "numerical_rejected" if numerical_evidence else "generation_error",
                      "measurements_status": "not_applicable_numerical_rejection",
                      "measurement_gaps": [] if numerical_evidence else ["numerical_rejection_unverifiable"],
                      "stitch": _w1_de_stitch_metrics(traj, T)}
        return out
    _w1_de_deadline(deadline, "before_validator")
    validator._cache = {}  # invalidate previous row even when this call pre-gates
    rep = validator.validate(scene, exclude=tuple(registration["evaluation"]["exclude"]))
    out["report"] = rep
    _w1_de_deadline(deadline, "after_validator")
    if rep.schedule is None:
        out["row"] = {**shared, **_w1_de_unmeasured(reference, rep.status, rep),
                      "status": "measurement_error", "measurements_status": "validator_pre_gate",
                      "measurement_gaps": [f"validator_pre_gate:{rep.status}"]}
        return out
    series = w1_de_frame_series(replayer, scene, cond.human_kps_world, cond.target_obj_pose, deadline=deadline)
    teacher_scene = np.concatenate([clip.root, clip.q, cond.target_obj_pose], axis=1)
    _w1_de_deadline(deadline, "teacher_fit_geometry")
    teacher_series = (series if row["method"] == "teacher" else w1_de_frame_series(
        replayer, teacher_scene, cond.human_kps_world, cond.target_obj_pose, deadline=deadline))
    metrics = w1_de_row_metrics(row["demo"], scene, cond, teacher_scene, teacher_series["robot_kpts"],
                               series, replayer, validator,
                               registration["evaluation"]["task_gate"]["tol_m"],
                               report=rep, deadline=deadline)
    out["row"] = {**shared, **metrics, "status": "ok", "stitch": _w1_de_stitch_metrics(traj, T)}
    out["series"] = series
    return out


def compare_w1_de_row_metrics(recorded: dict, recomputed: dict, *,
                               atol: float = 1e-6, rtol: float = 1e-6) -> list[str]:
    """Finite-before-tolerance for metrics; exact booleans/counts/intervals."""
    fields = ("status", "numerical_valid", "numerical_facts", "numerical_evidence_verified",
              "retry", "retry_evidence_problems", "product_verified", "covers_last_frame",
              "robot_strict_pass", "task_gate_pass", "raw_standard_pass", "measurements_status",
              "reference_intervals", "ref_frames", "ref_grasp_frames", "ref_support_frames",
              "both_reference", "fitting", "task", "robot_strict", "penetration", "box_ground",
              "feet", "hands", "both", "contact_f1", "stitch", "measurement_gaps")
    problems = []
    def compare(a, b, path):
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                problems.append(f"{path}:fields_not_reproducible")
            if isinstance(a, dict):
                for key in b:
                    compare(a.get(key), b[key], f"{path}.{key}")
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                problems.append(f"{path}:list_not_reproducible")
            else:
                for i, (av, bv) in enumerate(zip(a, b)):
                    compare(av, bv, f"{path}[{i}]")
        elif isinstance(b, bool) or b is None or isinstance(b, (str, int)):
            if type(a) is not type(b) or a != b:
                problems.append(f"{path}:not_reproducible")
        elif _w1_de_finite(b):
            if not _w1_de_finite(a) or abs(float(a) - float(b)) > atol + rtol * abs(float(b)):
                problems.append(f"{path}:nonfinite_or_not_reproducible")
        else:
            problems.append(f"{path}:recomputed_nonfinite")
    for name in fields:
        if name not in recorded or name not in recomputed:
            if name in recorded or name in recomputed:
                problems.append(f"{name}:missing")
        else:
            compare(recorded[name], recomputed[name], name)
    return sorted(set(problems))


def _w1_de_validation_problems(registration: dict, validator) -> list[str]:
    import dataclasses
    import inspect

    from ireset.embodied.contact_sites import infer_reference_schedule

    ev, problems = registration["evaluation"], []
    if ev.get("validation_mode") != "strict" or ev.get("validation_config") != dataclasses.asdict(validator.cfg):
        problems.append("registered_validation_config_mismatch")
    if ev.get("exclude") != list(W1_DE_EXCLUDE):
        problems.append("registered_exclude_mismatch")
    if ev.get("sampling") != W1_DE_SAMPLING:
        problems.append("registered_sampling_mismatch")
    if _w1_de_field(ev, "task_gate.frame") != "world" or _w1_de_field(ev, "task_gate.tol_m") != 0.40:
        problems.append("registered_task_gate_mismatch")
    if "collision_pairs" in ev and ev["collision_pairs"] != _w1_de_json(dataclasses.asdict(validator.pairs)):
        problems.append("registered_collision_pairs_mismatch")
    defaults = {k: p.default for k, p in inspect.signature(infer_reference_schedule).parameters.items()
                if p.default is not inspect.Parameter.empty}
    if "reference_schedule_defaults" in ev and ev["reference_schedule_defaults"] != defaults:
        problems.append("registered_reference_defaults_mismatch")
    return problems


def _w1_de_inputs(registration: dict, manifest_path: Path, cache_dir: Path,
                   project_root: Path, *, deadline=None) -> tuple[dict, dict, dict]:
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1
    from ireset.paths import OMOMO_NEW_DIR, TEACHER_TRAJ_DIR

    problems = []
    _w1_de_deadline(deadline, "input_verification")
    for key, want in (("execution_id", "w1_de_v1"), ("label_protocol", w1.MOTION_PROTOCOL_VERSION),
                      ("source_experiment_id", w1.EXPERIMENT_ID), ("training_purpose", w1.TRAINING_PURPOSE)):
        if registration.get(key) != want:
            problems.append(f"execution_{key}_mismatch")
    if (manifest_path != _w1_de_path(registration["training_manifest"]["path"], project_root)
            or integ.sha256_file(manifest_path) != registration["training_manifest"]["sha256"]):
        problems.append("registered_manifest_path_or_hash_mismatch")
    if (cache_dir != _w1_de_path(registration["cache"]["dir"], project_root)
            or integ.sha256_file(cache_dir / "cache_identity.json") != registration["cache"]["identity_sha256"]
            or integ.sha256_file(cache_dir / "root_stats.npz") != registration["cache"]["root_stats_sha256"]):
        problems.append("registered_cache_path_or_hash_mismatch")
    code_sha = w1.condition_code_sha256(project_root)
    if registration.get("condition_code_sha256") != code_sha:
        problems.append("registered_condition_code_mismatch")
    manifest = json.loads(manifest_path.read_text())
    for key, want in (("manifest_name", w1.MANIFEST_NAME), ("protocol", w1.MOTION_PROTOCOL_VERSION),
                      ("experiment_id", w1.EXPERIMENT_ID), ("training_purpose", w1.TRAINING_PURPOSE),
                      ("source_type", w1.SOURCE_TYPE), ("published_revision", w1.PUBLISHED_REVISION)):
        if manifest.get(key) != want:
            problems.append(f"manifest_{key}_mismatch")
    for split, name in (("train", "train24"), ("val", "val31")):
        if manifest["selection"].get(name) != registration["rosters"].get(name):
            problems.append(f"registered_{name}_mismatch")
    registered = set(registration["rosters"]["train24"]) | set(registration["rosters"]["val31"])
    rows = manifest.get("rows")
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        problems.append("manifest_rows_invalid")
    else:
        demos = [r.get("demo") for r in rows]
        if len(set(demos)) != len(demos) or any(d not in registered for d in demos):
            problems.append("manifest_rows_duplicated_or_foreign")
        for r in rows:
            want_split = "train" if r.get("demo") in registration["rosters"]["train24"] else "val"
            if r.get("split") != want_split:
                problems.append(f"manifest_split_mismatch:{r.get('demo')}")
    prereg_path = _w1_de_path(manifest["preregistration"]["path"], project_root)
    if (manifest["preregistration"] != registration["preregistration"]
            or integ.sha256_file(prereg_path) != manifest["preregistration"]["sha256"]):
        problems.append("manifest_preregistration_binding")
    prereg = json.loads(prereg_path.read_text())
    check = w1.verify_preregistration(prereg, project_root, TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
    problems += [f"preregistration:{p}" for p in check["problems"]]
    cache_check = w1.verify_cache_contents(cache_dir, manifest, prereg, project_root)
    problems += [f"cache:{p}" for p in cache_check["problems"]]
    assets_sha = prereg["assets"]["a1_assets_sha256"]
    if registration.get("assets_sha256") != assets_sha:
        problems.append("registered_assets_mismatch")
    source_files = registration.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        problems.append("registered_source_files_missing")
    else:
        for rel, want in source_files.items():
            _w1_de_deadline(deadline, "source_file_hash")
            path = _w1_de_path(rel, project_root)
            if not path.is_file() or integ.sha256_file(path) != want:
                problems.append(f"registered_source_file_missing_or_drifted:{rel}")
    if problems:
        raise ValueError(f"raw input admission failed: {problems}")
    return manifest, prereg, json.loads((cache_dir / "cache_identity.json").read_text())


def run_w1_de_raw_eval(args) -> None:
    """Fixed raw matrix with early evidence saves and independently replayable facts."""
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1
    from ireset.data.schema import RobotTrajectory
    from ireset.eval.teacher_comparison import TeacherQualityReplayer
    from ireset.model.generator import GenerationNumericalError
    from ireset.paths import PROJECT_ROOT

    t0 = time.perf_counter()
    reg_path = _w1_de_path(args.execution_registration, PROJECT_ROOT)
    reg = json.loads(reg_path.read_text())
    registration_sha = integ.sha256_file(reg_path)
    output = getattr(args, "output", None) or getattr(args, "out", None)
    if output is None:
        raise SystemExit("raw evaluation requires --out/--output")
    out, products_dir = guard_w1_de_outputs(Path(output), Path(args.products_dir),
                                             reg_path, reg, PROJECT_ROOT)
    rows = [{**r, "status": "not_run", "attempted": False,
             "execution_registration": {"path": _w1_de_record_path(reg_path, PROJECT_ROOT),
                                        "sha256": registration_sha}}
            for r in w1_de_expected_matrix(reg)]
    budget_arg = getattr(args, "budget_deadline_s", None)
    if budget_arg is not None and (isinstance(budget_arg, bool)
                                  or not _w1_de_finite(float(budget_arg)) or float(budget_arg) < 0):
        raise SystemExit("raw budget must be finite and nonnegative")
    deadline = t0 + float(budget_arg) if budget_arg is not None else None
    manifest_path = _w1_de_path(args.training_manifest, PROJECT_ROOT)
    cache_dir = _w1_de_path(args.cache_dir, PROJECT_ROOT)
    sel_path = _w1_de_path(args.checkpoint_selection, PROJECT_ROOT)
    model_devices = {model: _w1_de_field(reg, f"configs.{model}.device")
                     for model in ("deterministic", "diffusion")}
    report = {"stage": "W1-D/E raw full matrix", "schema": W1_DE_PRODUCT_SCHEMA,
              "execution_registration": {"path": _w1_de_record_path(reg_path, PROJECT_ROOT),
                                         "sha256": registration_sha},
              "training_manifest": reg["training_manifest"], "cache_dir": reg["cache"]["dir"],
              "expected_set": {"n": len(reg["rosters"]["val31"]),
                               "demos": reg["rosters"]["val31"], "val6": reg["rosters"]["val6"]},
              "methods": list(W1_DE_METHODS), "main_seed": 7, "appendix_seeds": [17, 27],
              "validation_config": reg["evaluation"]["validation_config"],
              "exclude": reg["evaluation"]["exclude"], "sampling": reg["evaluation"]["sampling"],
              "task_gate_tol_m": reg["evaluation"]["task_gate"]["tol_m"],
              "admission": {"passed": False, "problems": []}, "rows": rows,
              "repair_pipeline_invocations": 0, "budget_stopped": False,
              "cost": {"device": "registered_per_model", "model_devices": model_devices,
                       "gpu_synchronized": False, "checkpoint_gpu_synchronized": {},
                       "condition_shared_s": 0.0, "checkpoint_load_s": {},
                       "preflight_s": 0.0, "total_s": 0.0,
                       "warmup_performed": False,
                       "teacher_original_generation_cost_s": None}}
    out.parent.mkdir(parents=True, exist_ok=True)
    def persist(first=False):
        report["completeness"] = w1_de_matrix_completeness(
            rows, reg, budget_stopped=report["budget_stopped"])
        if report["admission"]["passed"] is not True:
            report["completeness"].update(complete=False, main_complete=False, appendix_complete=False)
        report["per_method"] = w1_de_aggregate_rows(rows, reg)
        report["matrix"] = {"main": report["completeness"]["expected_main_keys"],
                            "appendix": report["completeness"]["expected_appendix_keys"],
                            "total": len(rows), "complete": report["completeness"]["complete"]}
        report["cost"]["total_s"] = time.perf_counter() - t0
        _w1_de_write_report(out, report, first=first)
    persist(first=True)  # the report exists even if admission/load fails
    generators, checkpoint_errors = {}, {}
    try:
        manifest, prereg, cache_identity = _w1_de_inputs(
            reg, manifest_path, cache_dir, PROJECT_ROOT, deadline=deadline)
        _w1_de_deadline(deadline, "validator_construction")
        validator = IndependentValidator(cfg=load_mode_configs("strict")[1])
        problems = _w1_de_validation_problems(reg, validator)
        if args.ddim_steps != W1_DE_SAMPLING["ddim_steps"]:
            problems.append("CLI_ddim_steps_not_registered")
        if problems:
            raise ValueError(f"raw validation binding failed:{problems}")
        replayer = TeacherQualityReplayer(validator)
        sel = json.loads(sel_path.read_text())
        if not isinstance(sel.get("models"), dict) or set(sel["models"]) != {"deterministic", "diffusion"}:
            raise ValueError("checkpoint_selection model set mismatch")
        if sel.get("execution_registration_sha256") != registration_sha:
            raise ValueError("checkpoint_selection registration mismatch")
        report["checkpoint_selection"] = {"path": _w1_de_record_path(sel_path, PROJECT_ROOT),
                                           "sha256": integ.sha256_file(sel_path), "models": sel["models"]}
        report["admission"] = {"passed": True, "problems": [],
                               "preregistration_reverified": True, "cache_content_reverified": True}
        for model in ("deterministic", "diffusion"):
            _w1_de_deadline(deadline, "checkpoint_load")
            selection = sel["models"][model]
            if selection.get("role") == "unavailable":
                checkpoint_errors[model] = selection.get("reason", "checkpoint unavailable")
                continue
            start = time.perf_counter()
            device, synchronize_device = model_devices[model], False
            try:
                if not isinstance(device, str) or not device:
                    raise ValueError("registered_device_missing")
                parsed_device = torch.device(device)
                if parsed_device.type == "cuda":
                    if not torch.cuda.is_available() or (parsed_device.index is not None
                            and parsed_device.index >= torch.cuda.device_count()):
                        raise ValueError(f"registered_device_unavailable:{device}")
                    synchronize_device = True
                generators[model] = load_w1_de_generator(
                    {**selection, "model": model}, device, args.ddim_steps, cache_dir,
                    reg["training_manifest"]["sha256"], reg["cache"]["identity_sha256"],
                    reg["cache"]["root_stats_sha256"], registration_sha,
                    registration=reg, project_root=PROJECT_ROOT)
            except (Exception, SystemExit) as exc:
                checkpoint_errors[model] = f"{type(exc).__name__}:{exc}"
            finally:
                if synchronize_device:
                    _w1_de_sync(device)
                    report["cost"]["gpu_synchronized"] = True
                report["cost"]["checkpoint_gpu_synchronized"][model] = synchronize_device
                report["cost"]["checkpoint_load_s"][model] = time.perf_counter() - start
            _w1_de_deadline(deadline, "after_checkpoint_load")
        report["checkpoint_errors"] = checkpoint_errors
        report["cost"]["preflight_s"] = time.perf_counter() - t0
        by_demo = {r["demo"]: r for r in manifest["rows"]}
        products_dir.mkdir()
        for demo in reg["rosters"]["val31"]:
            _w1_de_deadline(deadline, "before_demo")
            demo_rows = [r for r in rows if r["demo"] == demo]
            row_record = by_demo.get(demo)
            start = time.perf_counter()
            try:
                if row_record is None:
                    raise ValueError("label_row_missing")
                if row_record.get("admission", {}).get("motion") is not True:
                    raise ValueError("label_not_admitted")
                path = _w1_de_path(row_record["derived"]["path"], PROJECT_ROOT)
                if integ.sha256_file(path) != row_record["derived"]["sha256"]:
                    raise ValueError("label_derived_drifted")
                clip = w1.load_w1_derived_clip(row_record, cache_dir / "schedules", PROJECT_ROOT,
                                               prereg["assets"]["a1_assets_sha256"],
                                               reg["condition_code_sha256"])
                if clip.T < 60:
                    raise ValueError("unsupported_short_trajectory")
                cond = clip.condition
                if (cond.T != clip.T or clip.root.shape != (clip.T, 7) or clip.q.shape != (clip.T, 29)
                        or not all(np.isfinite(a).all() for a in
                                   (clip.root, clip.q, cond.human_kps_world, cond.source_obj_pose,
                                    cond.target_obj_pose, cond.world_from_clip))):
                    raise ValueError("label_or_condition_invalid")
                reference = w1_de_reference_facts(cond)
            except Exception as exc:
                status = str(exc) if str(exc) in ("label_row_missing", "label_not_admitted", "label_derived_drifted",
                                                   "unsupported_short_trajectory") else "label_unloadable"
                for row in demo_rows:
                    row.update(status=status, error=f"{type(exc).__name__}:{exc}")
                persist()
                continue
            condition_s = time.perf_counter() - start
            report["cost"]["condition_shared_s"] += condition_s
            _w1_de_deadline(deadline, "after_condition_load")
            for row in demo_rows:
                _w1_de_deadline(deadline, "before_method")
                row_start = time.perf_counter()
                row.update(T=clip.T, fps=1 / cond.frame_dt, **reference,
                           cost={"cond_load_s": condition_s, "condition_shared": True,
                                 "generation_s": 0.0, "checks_s": 0.0, "save_s": 0.0,
                                 "row_total_s": 0.0})
                model, seed = row["method"], row["seed"]
                if model != "teacher" and model not in generators:
                    row.update(status="checkpoint_unavailable", error=checkpoint_errors.get(model))
                    continue
                device = "cpu" if model == "teacher" else model_devices[model]
                row["cost"].update(device=device, gpu_synchronized=device.startswith("cuda"))
                row["attempted"] = True
                trajectory, evidence, metadata = None, None, {}
                start = time.perf_counter()
                try:
                    _w1_de_sync(device)
                    if model == "teacher":
                        metadata = {"teacher_raw": True, "starts": [], "windows": 0,
                                    "model_calls": 0, "resample_attempts_used": 0,
                                    "resample_attempts_max": 0, "attempt_seeds": [],
                                    "per_attempt_invalid_windows": [], "per_attempt_invalid_window_indices": [],
                                    "final_invalid_windows": 0, "final_invalid_window_indices": [],
                                    "generation_wall_s": 0.0, "last_frame_covered": True}
                        trajectory = RobotTrajectory(clip.root.copy(), clip.q.copy(), cond.frame_dt, metadata)
                    else:
                        trajectory = generators[model].sample(cond, seed=seed)
                        metadata = trajectory.metadata
                except GenerationNumericalError as exc:
                    row["error"] = f"numerical:{exc.reason}"
                    metadata, evidence = exc.metadata, exc.output
                except Exception as exc:
                    row.update(status="generation_error", error=f"{type(exc).__name__}:{exc}")
                    metadata = getattr(generators.get(model), "_last_sample_stats", {})
                finally:
                    _w1_de_sync(device)
                    row["cost"]["generation_s"] = time.perf_counter() - start
                row["retry"] = _w1_de_retry(metadata)
                row["generation_wall_s"] = metadata.get("generation_wall_s")
                row["cost"]["generation_breakdown_s"] = {k: v for k, v in metadata.items()
                                                        if k.endswith("_wall_s") and _w1_de_finite(v)}
                if trajectory is None and evidence is None:
                    row["status"] = "generation_error"
                    row["cost"]["row_total_s"] = time.perf_counter() - row_start
                    persist()
                    _w1_de_deadline(deadline, "after_failed_generation")
                    continue
                identity = w1_de_product_identity(
                    row, reg, row_record, cond,
                    getattr(generators.get(model), "w1_de_checkpoint", None),
                    registration_sha, cache_identity)
                # Available arrays are evidence even when the next deadline/geometry block fails.
                start = time.perf_counter()
                product_path = products_dir / f"{model}__{demo}__seed{seed}.npz"
                try:
                    row["product"] = save_w1_de_product(
                        product_path, row, identity, cond, trajectory=trajectory,
                        output=evidence, metadata=metadata, project_root=PROJECT_ROOT)
                except Exception as exc:
                    row.update(status="product_error", error=f"save:{type(exc).__name__}:{exc}")
                    row["cost"]["save_s"] = time.perf_counter() - start
                    persist()
                    continue
                row["cost"]["save_s"] = time.perf_counter() - start
                try:
                    _w1_de_deadline(deadline, "after_product_save")
                    start = time.perf_counter()
                    result = recompute_w1_de_product(
                        product_path, row, reg, clip, validator, replayer, project_root=PROJECT_ROOT,
                        expected_identity=identity, deadline=deadline)
                    row.update(result["row"])
                    row["cost"]["checks_s"] = time.perf_counter() - start
                    row["evidence_gaps"] = w1_de_row_evidence_gaps(row)
                except TimeoutError as exc:
                    row.update(status="timeout", error=str(exc))
                    report["budget_stopped"] = True
                except Exception as exc:
                    row.update(status="measurement_error", error=f"{type(exc).__name__}:{exc}",
                               measurement_gaps=[f"measurement:{type(exc).__name__}"])
                row["cost"]["row_total_s"] = time.perf_counter() - row_start
                persist()
                print(f"[w1-de-raw] {row['key']}: {row['status']}", flush=True)
                _w1_de_deadline(deadline, "after_row_registration")
    except TimeoutError as exc:
        report["budget_stopped"] = True
        report["stop_reason"] = str(exc)
        for row in rows:
            if row["status"] == "not_run":
                row["not_run_reason"] = "budget_deadline"
    except (Exception, SystemExit) as exc:
        report["admission"]["problems"].append(f"{type(exc).__name__}:{exc}")
        report["admission"]["passed"] = False
        for row in rows:
            if row["status"] == "not_run":
                row.update(status="input_unavailable", error=f"{type(exc).__name__}:{exc}")
    persist()
    if deadline is not None and time.perf_counter() >= deadline:
        report["budget_stopped"] = True
        report["stop_reason"] = "budget_deadline_after_report_registration"
        persist()  # revoke completion on the same report, never overwrite prior attempts
    print(f"-> {out}", flush=True)
    if not report["completeness"]["complete"]:
        raise SystemExit(2)


def run_w1_raw_eval(args) -> None:
    """W1 raw full-length evaluation (omniretarget_w1_a3bc_plan §5.3).

    Fixed expected-set (the training manifest's val31 — failures and short
    trajectories stay rows in the denominator, never dropped), FULL clip
    length (the stitched trajectory covers the last frame through the
    trailing T-60 window), the UNIFIED consumption admission re-run here
    (preregistration anchors/inputs, cache CONTENT re-hashing, model
    identity binding — review fix 8: a checkpoint pointing at this
    manifest but a foreign cache/stats is refused), and ZERO
    RepairPipeline construction or invocation — raw diagnostics only (the
    frozen IndependentValidator is used for bare-output checks; it is not
    the repairer).
    """
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1
    from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record

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
    # unified consumption admission (review fix 8): the same preregistration
    # legs every other consumer runs — drifted inputs here cannot hide
    # behind an intact manifest binding
    check = w1.verify_preregistration(prereg, PROJECT_ROOT,
                                      TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
    if not check["passed"]:
        raise SystemExit(f"preregistration re-verification failed: "
                         f"{check['problems']}")
    cache_check = w1.verify_cache_contents(args.cache_dir, manifest, prereg,
                                           PROJECT_ROOT)
    if not cache_check["passed"]:
        raise SystemExit(f"W1 cache content verification failed: "
                         f"{cache_check['problems']}")
    assets_sha = prereg["assets"]["a1_assets_sha256"]
    code_sha = w1.condition_code_sha256(PROJECT_ROOT)
    cache_ident_sha = integ.sha256_file(args.cache_dir / "cache_identity.json")
    root_stats_sha = integ.sha256_file(args.cache_dir / "root_stats.npz")

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
        problems = []
        if ident.get("training_manifest", {}).get("sha256") != manifest_sha:
            problems.append("manifest")
        if ident.get("protocol") != w1.MOTION_PROTOCOL_VERSION:
            problems.append("protocol")
        if ident.get("cache", {}).get("identity_sha256") != cache_ident_sha:
            problems.append("cache_identity")
        if ident.get("cache", {}).get("root_stats_sha256") != root_stats_sha:
            problems.append("root_stats")
        if ident.get("model_type") != m:
            problems.append("model_type")
        if problems:
            raise SystemExit(
                f"model {m}: checkpoint identity does not match the current "
                f"manifest/cache ({problems}: ckpt={ident.get('training_manifest')}"
                f"/{ident.get('cache')}) — a checkpoint from another "
                "manifest/cache/stats combination is refused")
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
        "admission": {"preregistration_reverified": True,
                      "cache_content_reverified": True},
    }
    if args.output:
        out = guard_report_output(Path(args.output))
    else:
        out = integ.allocate_attempt(
            OUT / "w1_raw") / "w1_raw_generation_eval.json"
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
    ap.add_argument("--output", "--out", dest="output", type=str, default=None)
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
    ap.add_argument("--w1-de-raw", action="store_true",
                    help="W1-D/E raw matrix mode: fixed 93-key main matrix "
                         "+ 62-key diffusion appendix over the registered "
                         "val31, full products, zero repair")
    ap.add_argument("--execution-registration", type=Path, default=None,
                    help="w1-de-raw mode: the execution registration")
    ap.add_argument("--checkpoint-selection", type=Path, default=None,
                    help="w1-de-raw mode: the driver-written explicit "
                         "checkpoint selection record")
    ap.add_argument("--products-dir", type=Path, default=None,
                    help="w1-de-raw mode: per-row full trajectory products")
    ap.add_argument("--budget-deadline-s", type=float, default=None,
                    help="w1-de-raw mode: remaining wall budget (monotonic)")
    args = ap.parse_args()

    if args.w1_de_raw:
        missing = [flag for flag, v in (
            ("--execution-registration", args.execution_registration),
            ("--checkpoint-selection", args.checkpoint_selection),
            ("--training-manifest", args.training_manifest),
            ("--cache-dir", args.cache_dir),
            ("--out", args.output),
            ("--products-dir", args.products_dir)) if v is None]
        if missing:
            raise SystemExit(f"w1-de-raw mode requires explicit {missing}")
        return run_w1_de_raw_eval(args)

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
