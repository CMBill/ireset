"""Frozen-checkpoint W1 failure attribution; no training or production changes.

Register once, then run once in an independent derived directory. Original
decoded windows are captured from the real generator, never sampled again to
construct a counterfactual. Missing evidence keeps the fixed denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import omniretarget_w1_de as driver
from ireset.data.a2_integrity import sha256_file
from ireset.model.stitch import fuse_windows, window_starts, window_weight

PLAN = ROOT / "docs/plans/omniretarget_w1_attribution_plan.md"
SOURCE = ROOT / "derived/experiments/omniretarget_first/20261013_be00c50_w1_de"
EXECUTION_ID = "w1_attribution_v1"
BUDGET_S, RESERVE_S = 3600.0, 60.0
KNOWN_CASE = "sub16_largebox_048"
CONDITION_KEYS = ("kps_obj", "kps_clip", "box_t_clip", "box_quat", "contact")
FAMILIES = ("robot_ground", "robot_box", "self")
EXPECTED_CKPTS = {
    "deterministic": "7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff",
    "diffusion": "1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1",
}


def record_path(path: Path) -> str:
    path = path.absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data, *, exclusive=False):
    driver.atomic_json(path, safe_json(data), exclusive=exclusive)


def file_record(path: Path) -> dict:
    return {"path": record_path(path), "sha256": sha256_file(path)}


def guard_run(path: Path, source: Path) -> Path:
    path, source = path.expanduser().resolve(), source.resolve()
    if (not path.is_relative_to((ROOT / "derived").resolve())
            or path.is_relative_to(source) or source.is_relative_to(path)):
        raise ValueError("diagnosis output must be an independent directory under derived")
    return path


def select_train6(train24: list[str]) -> list[str]:
    if len(train24) != 24 or len(set(train24)) != 24:
        raise ValueError("train24 roster malformed")
    subjects = sorted({d.split("_", 1)[0] for d in train24})
    if len(subjects) != 6:
        raise ValueError("six subjects required")
    return [min(d for d in train24 if d.split("_", 1)[0] == subject) for subject in subjects]


def hash_problems(anchors: dict[str, str]) -> list[str]:
    problems = []
    for rel, want in anchors.items():
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != want:
            problems.append(f"anchor_missing_or_drifted:{rel}")
    return problems


def verify_registration(run: Path) -> dict:
    reg = read_json(run / "freeze/registration.json")
    if reg.get("execution_id") != EXECUTION_ID:
        raise ValueError("wrong attribution execution")
    if reg.get("train6") != select_train6(reg["train24"]):
        raise ValueError("train selection not reproducible")
    expected = reg["train6"] + reg["val6"] + [KNOWN_CASE]
    if len(expected) != 13 or len(set(expected)) != 13 or reg.get("demos") != expected:
        raise ValueError("diagnosis denominator changed")
    problems = hash_problems(reg["anchors"])
    if problems:
        raise ValueError(problems)
    return reg


def register(run: Path, source: Path) -> None:
    import eval_generation as raw

    t0 = time.perf_counter()
    run = guard_run(run, source)
    if run.exists() and any(run.iterdir()):
        raise ValueError("registration requires a new empty run; no implicit retries")
    source_reg_path = source / "freeze/execution_registration.json"
    reg, selection = read_json(source_reg_path), read_json(source / "evaluation/checkpoint_selection.json")
    manifest, prereg, cache = raw._w1_de_inputs(
        reg, ROOT / reg["training_manifest"]["path"], ROOT / reg["cache"]["dir"], ROOT,
        deadline=t0 + BUDGET_S - RESERVE_S)
    source_states = {}
    for stage in driver.STAGE_DIRS:
        _, problems = driver.stage_state(source, stage, ROOT)
        source_states[stage] = problems
    if any(source_states.values()):
        raise ValueError(f"source stage incomplete:{source_states}")
    budgets = {d: driver.domain_history_cost_s(source, d) for d in driver.DOMAINS}
    if any(v is None or v >= driver.DOMAINS[k][1] for k, v in budgets.items()):
        raise ValueError(f"source budget incomplete:{budgets}")
    final = read_json(source / "decisions/attempt_001/w1_de_final.json")
    if (final.get("complete") is not True or final.get("motion_baseline") != "complete"
            or final.get("final_interaction_method", {}).get("verdict") != "not_ready"):
        raise ValueError("source F is not the closed W1 baseline")
    anchors = dict(reg["source_files"])
    for path in (source_reg_path, source / "evaluation/checkpoint_selection.json",
                 source / "evaluation/raw/attempt_001/raw_eval.json",
                 source / "decisions/attempt_001/w1_de_final.json", ROOT / reg["training_manifest"]["path"],
                 ROOT / reg["preregistration"]["path"], ROOT / reg["cache"]["dir"] / "cache_identity.json",
                 ROOT / reg["cache"]["dir"] / "root_stats.npz", PLAN, Path(__file__)):
        anchors[record_path(path)] = sha256_file(path)
    for model, want in EXPECTED_CKPTS.items():
        rec = selection["models"][model]
        if (rec.get("sha256") != want or rec.get("step") != 4000
                or sha256_file(ROOT / rec["path"]) != want):
            raise ValueError(f"unregistered checkpoint:{model}")
        anchors[rec["path"]] = want
    original_rows = read_json(source / "evaluation/raw/attempt_001/raw_eval.json")["rows"]
    if len(original_rows) != 155 or len({r["key"] for r in original_rows}) != 155:
        raise ValueError("source matrix incomplete")
    for row in original_rows:
        rec = row["product"]
        anchors[rec["path"]] = rec["sha256"]
    for row in manifest["rows"]:
        anchors[row["derived"]["path"]] = row["derived"]["sha256"]
    for row in cache["schedule_rows"]:
        # Cache verifier has checked the closed content hashes above.
        from ireset.data import w1_training as w1
        path = ROOT / reg["cache"]["dir"] / "schedules" / w1.schedule_cache_filename(row["demo"], row["key"])
        anchors[record_path(path)] = sha256_file(path)
    # Pin the executed implementation, including legacy fusion and measurement.
    for rel in ("src/ireset/model/generator.py", "src/ireset/model/stitch.py",
                "src/ireset/model/tokens.py", "scripts/eval_generation.py",
                "scripts/omniretarget_w1_de.py", "scripts/omniretarget_w1_de_detail.py"):
        anchors[rel] = sha256_file(ROOT / rel)
    problems = hash_problems(anchors)
    if problems:
        raise ValueError(problems)
    train6, val6 = select_train6(reg["rosters"]["train24"]), reg["rosters"]["val6"]
    registration = {
        "execution_id": EXECUTION_ID, "source_run": record_path(source),
        "source_registration": file_record(source_reg_path), "plan": file_record(PLAN),
        "implementation": driver.implementation_identity(ROOT),
        "environment": driver.env_record(ROOT), "source_budgets_s": budgets,
        "train24": reg["rosters"]["train24"], "train6": train6, "val6": val6,
        "known_case": KNOWN_CASE, "demos": train6 + val6 + [KNOWN_CASE],
        "expected_scene_rows": 78, "seed": 7, "ddim_steps": 25,
        "budget_s": BUDGET_S, "reserve_s": RESERVE_S, "unit_timeout_s": 600,
        "training_steps": 0, "production_changes": False,
        "counterfactual": "positive-weight ordered cumulative SLERP; same decoded predictions",
        "anchors": anchors, "registered_wall_s": time.perf_counter() - t0,
    }
    write_json(run / "freeze/registration.json", registration, exclusive=True)
    write_json(run / "costs.json", [{"stage": "register", "status": "done", "complete": True,
                                    "wall_s": time.perf_counter() - t0}])
    if time.perf_counter() - t0 >= BUDGET_S - RESERVE_S:
        raise TimeoutError("registration exhausted diagnostic budget")
    print(json.dumps({"registered": record_path(run), "train6": train6,
                      "anchors": len(anchors)}, ensure_ascii=False), flush=True)


def overlap_facts(starts: list[int], T: int, F=60, stride=30) -> dict[str, np.ndarray]:
    coverage, positive = np.zeros(T, int), np.zeros(T, int)
    dropped_mass, weights = np.zeros(T), [[] for _ in range(T)]
    for i, start in enumerate(starts):
        w = window_weight(i, len(starts), F, stride)
        for j, value in enumerate(w):
            t = start + j
            coverage[t] += 1
            positive[t] += int(value > 0)
            weights[t].append(value)
    for t, entries in enumerate(weights):
        if len(entries) > 2:
            dropped_mass[t] = sum(entries[1:-1]) / sum(entries)
    return {"coverage": coverage, "positive_contributors": positive,
            "legacy_dropped_rotation_weight": dropped_mass}


def fuse_all_contributors(decoded, starts: list[int], T: int, stride=30):
    """Diagnostic only: preserve production positions/joints; include all rotations."""
    pos, quat, joints, diag = fuse_windows(*[list(a) for a in decoded], starts, T, stride)
    entries = [[] for _ in range(T)]
    F = decoded[0].shape[1]
    for i, start in enumerate(starts):
        for j, weight in enumerate(window_weight(i, len(starts), F, stride)):
            if weight > 0:
                entries[start + j].append((decoded[1][i, j], weight))
    out = np.empty_like(quat)
    for t, row in enumerate(entries):
        if not row:
            raise ValueError("no positive-weight contributor")
        q, mass = row[0]
        for next_q, weight in row[1:]:
            rotations = Rotation.from_quat(np.stack((q, next_q))[:, [1, 2, 3, 0]])
            q = Slerp([0, 1], rotations)([weight / (mass + weight)]).as_quat()[0, [3, 0, 1, 2]]
            mass += weight
        out[t] = q
    return pos, out, joints, diag


def world_state(pos: np.ndarray, quat: np.ndarray, joints: np.ndarray, wfc: np.ndarray):
    shape = pos.shape[:-1]
    world_pos = pos @ wfc[:3, :3].T + wfc[:3, 3]
    world_quat = (Rotation.from_matrix(wfc[:3, :3])
                  * Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]])).as_quat()
    root = np.concatenate((world_pos, world_quat[:, [3, 0, 1, 2]].reshape(*shape, 4)), -1)
    return root, joints.copy()


def capture_prediction(generator, condition, seed=7):
    import torch

    saved, original = {}, generator._sample_windows
    clean_outputs = []
    output_method = "sample" if hasattr(generator.model, "sample") else "forward"
    original_output = getattr(generator.model, output_method)

    def capture_clean(*args, **kwargs):
        value = original_output(*args, **kwargs)
        if torch.is_tensor(value):
            clean_outputs.append(value.detach().cpu().numpy().copy())
        return value

    def capture(self, batch, call_seed):
        saved["condition"] = {k: np.asarray(v).copy() for k, v in batch.items()}
        result = original(batch, call_seed)
        saved["decoded"] = tuple(v.copy() for v in result[0])
        return result

    generator._sample_windows = MethodType(capture, generator)
    setattr(generator.model, output_method, capture_clean)
    try:
        trajectory = generator.sample(condition, seed=seed)
    finally:
        generator._sample_windows = original
        setattr(generator.model, output_method, original_output)
    if not clean_outputs:
        raise ValueError("normalized predictions not captured")
    combined = clean_outputs[0].copy()
    for output, attempt in zip(clean_outputs[1:], trajectory.metadata["attempts"][1:]):
        indices = attempt["replacement_window_indices"]
        combined[indices] = output[indices]
    saved["normalized_prediction"] = combined
    return trajectory, saved


def oracle_roundtrip(tokenizer, clip):
    import torch
    from ireset.model.generator import G1Generator

    starts = window_starts(clip.T, 60, 30)
    x0 = tokenizer.encode(clip.root, clip.q, clip.condition.world_from_clip).astype(np.float32)
    targets = np.stack([x0[s:s + 60] for s in starts])

    class FixedPrediction(torch.nn.Module):
        def forward(self, memory, frames):
            return torch.from_numpy(targets.copy())

    class NoCondition(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros((len(starts), 1, 1))

    generator = G1Generator(FixedPrediction(), NoCondition(), tokenizer, device="cpu")
    return capture_prediction(generator, clip.condition)[0]


def condition_comparison(dataset, demo: str, generator, condition) -> dict:
    import torch

    arrays = generator._condition_batch(condition)
    rows = []
    for index, (ci, start) in enumerate(dataset.index):
        if dataset.clips[ci].demo != demo:
            continue
        item = dataset[index]
        checks = {}
        for key in CONDITION_KEYS:
            expected = torch.as_tensor(arrays[key][start:start + 60], dtype=torch.float32)
            actual = item[key]
            checks[key] = bool(actual.dtype == expected.dtype and actual.shape == expected.shape
                               and torch.equal(actual, expected))
        rows.append({"start": start, "checks": checks, "passed": all(checks.values())})
    starts = window_starts(condition.T, 60, 30)
    expected_starts = list(range(0, condition.T - 60 + 1, 30))
    return {"passed": [r["start"] for r in rows] == expected_starts
                       and bool(rows) and all(r["passed"] for r in rows),
            "rows": rows, "inference_only_starts": sorted(set(starts) - set(expected_starts))}


def family_geometry(replayer, scene, deadline):
    import mujoco

    values, pairs = np.zeros((len(scene), 3)), np.full((len(scene), 3, 2), -1, dtype=np.int32)
    for t in range(len(scene)):
        check_deadline(deadline)
        replayer._forward(scene, t)
        for k, kind in enumerate(FAMILIES):
            for a, b in replayer.pairs_by_kind.get(kind, []):
                pen = -replayer._dist(a, b)
                if pen > values[t, k]:
                    values[t, k], pairs[t, k] = pen, (a, b)
    worst = {}
    for k, kind in enumerate(FAMILIES):
        t = int(values[:, k].argmax())
        worst[kind] = {"max_m": float(values[t, k]), "frame": t,
                       "geoms": [mujoco.mj_id2name(replayer.model, mujoco.mjtObj.mjOBJ_GEOM, int(i))
                                 if i >= 0 else None for i in pairs[t, k]]}
    return values, pairs, worst


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("attribution deadline")


def save_scene(out: Path, name: str, root, joints, starts, clip, reg, validator,
               replayer, teacher_kpts, deadline):
    import eval_generation as raw
    import omniretarget_w1_de_detail as detail

    check_deadline(deadline)
    cond = clip.condition
    scene = np.concatenate([root, joints, cond.target_obj_pose], axis=1)
    teacher = np.concatenate([clip.root, clip.q, cond.target_obj_pose], axis=1)
    series = raw.w1_de_frame_series(replayer, scene, cond.human_kps_world, cond.target_obj_pose,
                                    deadline=deadline)
    report = validator.validate(scene, exclude=raw.W1_DE_EXCLUDE)
    metrics = raw.w1_de_row_metrics(clip.demo_id if hasattr(clip, "demo_id") else cond.source_clip_id,
                                  scene, cond, teacher, teacher_kpts, series, replayer,
                                  validator, reg["evaluation"]["task_gate"]["tol_m"],
                                  report=report, deadline=deadline)
    if metrics.get("measurement_gaps") or report.schedule is None:
        raise ValueError(f"measurement incomplete:{metrics.get('measurement_gaps')}")
    arrays = {"root": root, "joints": joints, "window_starts": np.asarray(starts, dtype=np.int64),
              "task_tol_m": reg["evaluation"]["task_gate"]["tol_m"]}
    curves = detail._detail_curves(arrays, clip, series, report, validator)
    problems = detail._verify_detail_curves(curves, clip.T, hand_gap_m=validator.cfg.hand_gap_m)
    if problems:
        raise ValueError(f"curves invalid:{problems}")
    family, pairs, worst = family_geometry(replayer, scene, deadline)
    curves.update(penetration_families=family, worst_geom_pairs=pairs)
    for side in ("l", "r"):
        index = raw.KPT_IDS.index(f"kpt_{side}_wrist")
        curves[f"hand_fk_error_{side}_m"] = np.linalg.norm(
            series["robot_kpts"][:, index] - teacher_kpts[:, index], axis=1)
    product = out / f"{name}.npz"
    np.savez_compressed(product, **arrays, target_box=cond.target_obj_pose,
                        world_from_clip=cond.world_from_clip, **{f"curve__{k}": v for k, v in curves.items()})
    valid_rot = curves["base_angular_velocity_radps"][1:]
    seam = curves["seam_rotation_jump_deg"]
    rec = {"variant": name, "status": "ok", "product": file_record(product), "metrics": metrics,
           "families": worst, "root_position_max_error_m": float(curves["fit_root_position_m"].max()),
           "root_rotation_max_error_deg": float(curves["fit_root_rotation_deg"].max()),
           "joint_max_error_rad": float(np.abs(joints - clip.q).max()),
           "hand_fk_mean_m": {s: float(curves[f"hand_fk_error_{s}_m"].mean()) for s in ("l", "r")},
           "angular_velocity_max_radps": float(valid_rot.max()),
           "seam_rotation_max_deg": float(np.nanmax(seam)) if np.isfinite(seam).any() else None}
    return rec, curves


def paired_evidence(teacher, model) -> dict:
    out = {}
    for k, kind in enumerate(FAMILIES):
        a, b = teacher["penetration_families"][:, k] > .002, model["penetration_families"][:, k] > .002
        same = (teacher["worst_geom_pairs"][:, k] == model["worst_geom_pairs"][:, k]).all(1)
        out[kind] = {"teacher_frames": int(a.sum()), "model_frames": int(b.sum()),
                     "shared_frames": int((a & b).sum()), "new_frames": int((~a & b).sum()),
                     "same_worst_geom_pair_frames": int((a & b & same).sum())}
    return out


def window_geometry(decoded, starts, clip, replayer, teacher_kpts, deadline):
    import eval_generation as raw

    roots, qs = world_state(*decoded, clip.condition.world_from_clip)
    pens, errors = [], []
    cond = clip.condition
    for i, start in enumerate(starts):
        scene = np.concatenate((roots[i], qs[i], cond.target_obj_pose[start:start + 60]), 1)
        series = raw.w1_de_frame_series(replayer, scene, cond.human_kps_world[start:start + 60],
                                       cond.target_obj_pose[start:start + 60], deadline=deadline)
        pens.append(series["pen_key"])
        errors.append(np.linalg.norm(series["robot_kpts"] - teacher_kpts[start:start + 60], axis=-1).mean(1))
    minimum, maximum = np.full(clip.T, np.inf), np.zeros(clip.T)
    for i, start in enumerate(starts):
        w = window_weight(i, len(starts), 60, 30)
        valid = np.flatnonzero(w > 0)
        minimum[start + valid] = np.minimum(minimum[start + valid], pens[i][valid])
        maximum[start + valid] = np.maximum(maximum[start + valid], pens[i][valid])
    if not np.isfinite(minimum).all():
        raise ValueError("uncovered window geometry")
    return {"root": roots, "joints": qs, "penetration": np.stack(pens),
            "fk_error": np.stack(errors), "pen_min": minimum, "pen_max": maximum}


def unit(run: Path, demo: str, out: Path, deadline: float) -> dict:
    import torch
    import eval_generation as raw
    from ireset.data import w1_training as w1
    from ireset.data.windows import TeacherWindowDataset
    from ireset.eval.teacher_comparison import TeacherQualityReplayer
    from ireset.kinematics.mujoco_adapter import G1Kinematics
    from ireset.repair.validation import IndependentValidator, ValidationConfig

    torch.set_num_threads(1)
    registered = verify_registration(run)
    if demo not in registered["demos"]:
        raise ValueError("unregistered demonstration")
    source = ROOT / registered["source_run"]
    reg = read_json(source / "freeze/execution_registration.json")
    selection = read_json(source / "evaluation/checkpoint_selection.json")
    manifest = read_json(ROOT / reg["training_manifest"]["path"])
    prereg = read_json(ROOT / reg["preregistration"]["path"])
    hits = [r for r in manifest["rows"] if r.get("demo") == demo]
    if len(hits) != 1 or hits[0].get("admission", {}).get("motion") is not True:
        raise ValueError("demo not uniquely admitted")
    row = hits[0]
    cache = ROOT / reg["cache"]["dir"]
    clip = w1.load_w1_derived_clip(row, cache / "schedules", ROOT,
                                 prereg["assets"]["a1_assets_sha256"], reg["condition_code_sha256"])
    out.mkdir(parents=True, exist_ok=False)
    cond = clip.condition
    validator = IndependentValidator(cfg=ValidationConfig(**reg["evaluation"]["validation_config"]))
    replayer = TeacherQualityReplayer(validator)
    teacher_scene = np.concatenate((clip.root, clip.q, cond.target_obj_pose), 1)
    teacher_kpts = raw.fk_keypoints(G1Kinematics(), teacher_scene)
    teacher_rec, teacher_curves = save_scene(out, "teacher", clip.root, clip.q, [], clip, reg,
                                            validator, replayer, teacher_kpts, deadline)
    starts = window_starts(clip.T, 60, 30)
    overlaps = overlap_facts(starts, clip.T)
    result = {"demo": demo, "split": row["split"], "T": clip.T,
              "group": "train6" if demo in registered["train6"] else "val6" if demo in registered["val6"] else "known_case",
              "starts": starts, "triple_overlap_frames": np.flatnonzero(overlaps["coverage"] > 2).tolist(),
              "dropped_weight_max": float(overlaps["legacy_dropped_rotation_weight"].max()),
              "rows": [teacher_rec], "models": {}, "status": "running"}
    write_json(out / "result.json", result, exclusive=True)
    for model in ("deterministic", "diffusion"):
        check_deadline(deadline)
        device = reg["configs"][model]["device"]
        generator = raw.load_w1_de_generator(
            selection["models"][model], device, 25, cache,
            reg["training_manifest"]["sha256"], reg["cache"]["identity_sha256"], reg["cache"]["root_stats_sha256"],
            registered["source_registration"]["sha256"], registration=reg, project_root=ROOT)
        dataset = TeacherWindowDataset(row["split"], generator.tokenizer,
                                       training_manifest=ROOT / reg["training_manifest"]["path"],
                                       schedule_cache_dir=cache / "schedules", verify_assets=True)
        condition_check = condition_comparison(dataset, demo, generator, cond)
        if not condition_check["passed"]:
            raise ValueError(f"training/generation conditions differ:{condition_check}")
        trajectory, captured = capture_prediction(generator, cond, 7)
        decoded = captured["decoded"]
        original, original_curves = save_scene(out, model, trajectory.root, trajectory.q, starts,
                                               clip, reg, validator, replayer, teacher_kpts, deadline)
        alternative = fuse_all_contributors(decoded, starts, clip.T)
        cf_root, cf_q = world_state(*alternative[:3], cond.world_from_clip)
        if not (np.array_equal(cf_root[:, :3], trajectory.root[:, :3]) and np.array_equal(cf_q, trajectory.q)):
            raise ValueError("counterfactual changed positions or joints")
        cf, cf_curves = save_scene(out, model + "_all_contributors", cf_root, cf_q, starts,
                                  clip, reg, validator, replayer, teacher_kpts, deadline)
        local = window_geometry(decoded, starts, clip, replayer, teacher_kpts, deadline)
        changes_deg = raw.geodesic_deg(trajectory.root[:, 3:7], cf_root[:, 3:7])
        # SO(3) trace/arccos has a small nonzero floor for equal float rotations.
        illegal_changes = (changes_deg > 1e-4) & (overlaps["coverage"] <= 2)
        if illegal_changes.any():
            raise ValueError("counterfactual changed a two-contributor frame")
        replay = None
        if row["split"] == "val":
            raw_rows = read_json(source / "evaluation/raw/attempt_001/raw_eval.json")["rows"]
            old = next(r for r in raw_rows if r["demo"] == demo and r["method"] == model and r["seed"] == 7)
            with np.load(ROOT / old["product"]["path"], allow_pickle=False) as z:
                errors = {"root": float(np.abs(z["root"] - trajectory.root).max()),
                          "joints": float(np.abs(z["joints"] - trajectory.q).max())}
            replay = {"max_abs": errors, "passed": all(v <= 1e-5 for v in errors.values()),
                      "bitwise_equal": all(v == 0 for v in errors.values()), "source": old["product"]}
            if not replay["passed"]:
                raise ValueError("production checkpoint replay mismatch")
        fused_bad = original_curves["penetration_keyframe"] > .002
        pred_evidence = {
            "all_active_windows_penetrate_frames": int((local["pen_min"] > .002).sum()),
            "some_active_window_penetrates_frames": int((local["pen_max"] > .002).sum()),
            "fusion_only_penetration_frames": int((fused_bad & (local["pen_max"] <= .002)).sum()),
            "fused_penetration_frames": int(fused_bad.sum()),
            "window_fk_mean_m": float(local["fk_error"].mean()),
            "window_penetration_max_m": float(local["penetration"].max()),
        }
        from ireset.kinematics.rotation_adapter import rot6d_to_mat_torch
        normalized = captured["normalized_prediction"]
        expected_rotation = rot6d_to_mat_torch(torch.as_tensor(normalized[..., 3:9])).numpy()
        actual_rotation = Rotation.from_quat(decoded[1].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
        rotation_error = Rotation.from_matrix(
            np.swapaxes(expected_rotation.reshape(-1, 3, 3), -1, -2) @ actual_rotation).magnitude()
        pred_evidence["decoded_rotation_consistency_max_deg"] = float(np.rad2deg(rotation_error).max())
        windows_path = out / f"{model}_windows.npz"
        np.savez_compressed(windows_path, **{f"decoded_{k}": v for k, v in zip(("pos", "quat", "q"), decoded)},
                            **{f"condition_{k}": v for k, v in captured["condition"].items()},
                            **{f"window_{k}": v for k, v in local.items()}, **overlaps,
                            starts=np.asarray(starts), root_rotation_change_deg=changes_deg,
                            normalized_prediction=normalized, decoded_rotation_error_deg=np.rad2deg(rotation_error))
        result["rows"] += [original, cf]
        result["models"][model] = {
            "condition_check": condition_check, "retry": trajectory.metadata,
            "replay": replay, "windows": file_record(windows_path),
            "paired_penetration": paired_evidence(teacher_curves, original_curves),
            "pre_fusion": pred_evidence,
            "counterfactual_changed_frames": np.flatnonzero(changes_deg > 1e-4).tolist(),
            "counterfactual_max_angle_change_deg": float(changes_deg.max()),
            "cf_seam_rotation_max_deg": cf["seam_rotation_max_deg"],
        }
        if model == "deterministic":
            oracle = oracle_roundtrip(generator.tokenizer, clip)
            oracle_rec, _ = save_scene(out, "teacher_roundtrip", oracle.root, oracle.q, starts, clip,
                                       reg, validator, replayer, teacher_kpts, deadline)
            result["rows"].append(oracle_rec)
            result["roundtrip_passed"] = (oracle_rec["root_position_max_error_m"] <= 1e-5
                                           and oracle_rec["root_rotation_max_error_deg"] <= 1e-3
                                           and oracle_rec["joint_max_error_rad"] <= 1e-5)
        write_json(out / "result.json", result)
        del dataset, generator
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    result["source_anchors_unchanged"] = not hash_problems(registered["anchors"])
    result["status"] = "ok" if result["source_anchors_unchanged"] and len(result["rows"]) == 6 else "incomplete"
    check_deadline(deadline)
    write_json(out / "result.json", result)
    return result


def aggregate(results: list[dict], expected: list[str]) -> dict:
    names = [r.get("demo") for r in results]
    gaps = []
    if len(names) != len(set(names)) or set(names) != set(expected):
        gaps.append("demo_matrix_mismatch")
    variants = {"teacher", "teacher_roundtrip", "deterministic", "diffusion",
                "deterministic_all_contributors", "diffusion_all_contributors"}
    for r in results:
        rows = r.get("rows", [])
        if (r.get("status") != "ok" or len(rows) != 6
                or {x.get("variant") for x in rows} != variants):
            gaps.append(f"scene_matrix_incomplete:{r.get('demo')}")
        if (r.get("source_anchors_unchanged") is not True
                or set(r.get("models", {})) != {"deterministic", "diffusion"}
                or any(m.get("condition_check", {}).get("passed") is not True
                       for m in r.get("models", {}).values())):
            gaps.append(f"unit_binding_incomplete:{r.get('demo')}")
        for row in rows:
            m = row.get("metrics", {})
            fields = (m.get("fitting", {}).get("teacher_fit_fk_mpjpe_box_m"),
                      m.get("penetration", {}).get("max_m"))
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in fields):
                gaps.append(f"metric_missing_or_nonfinite:{r.get('demo')}:{row.get('variant')}")
    groups = {}
    for group in ("train6", "val6", "known_case"):
        selected = [r for r in results if r.get("group") == group]
        if len(selected) != (1 if group == "known_case" else 6):
            gaps.append(f"group_denominator_mismatch:{group}")
        groups[group] = {}
        for variant in sorted(variants):
            rows = [x for r in selected for x in r.get("rows", []) if x.get("variant") == variant]
            valid = [x for x in rows if isinstance(x.get("metrics", {}).get("penetration", {}).get("max_m"), (float, int))]
            groups[group][variant] = {
                "n": len(rows), "expected": 1 if group == "known_case" else 6,
                "raw_standard_pass": sum(x.get("metrics", {}).get("raw_standard_pass") is True for x in rows),
                "teacher_fit_fk_mean_m": float(np.mean([x["metrics"]["fitting"]["teacher_fit_fk_mpjpe_box_m"] for x in valid])) if valid else None,
                "penetration_max_m": max((x["metrics"]["penetration"]["max_m"] for x in valid), default=None),
            }
    return {"complete": not gaps, "evidence_gaps": gaps, "groups": groups,
            "paired_distinct_train_examples": sum(r.get("group") == "train6" and r.get("status") == "ok" for r in results),
            "label_principal_bottleneck_supported": False,
            "attribution_note": "paired co-occurrence is descriptive; no automated principal-label attribution",
            "final_interaction_method": "not_ready", "training_steps": 0,
            "historical_anchors": {"u0": "incomplete", "ur": "no_quality_gain"}, "auto_execute": False}


def verify_local_products(results: list[dict]) -> list[str]:
    problems = []
    for result in results:
        if result.get("status") != "ok":
            continue
        for row in result["rows"]:
            rec = row["product"]
            path = ROOT / rec["path"]
            if not path.is_file() or sha256_file(path) != rec["sha256"]:
                problems.append(f"diagnostic_product_drift:{result['demo']}:{row['variant']}")
                continue
            with np.load(path, allow_pickle=False) as z:
                if (z["root"].shape != (result["T"], 7) or z["joints"].shape != (result["T"], 29)
                        or not all(np.isfinite(z[k]).all() for k in ("root", "joints", "target_box"))):
                    problems.append(f"diagnostic_product_invalid:{result['demo']}:{row['variant']}")
                fit = float(z["curve__fit_fk_mpjpe_box_m"].mean())
                pen = float(z["curve__penetration_adaptive_max_m"])
                if (abs(fit - row["metrics"]["fitting"]["teacher_fit_fk_mpjpe_box_m"]) > 1e-12
                        or pen != row["metrics"]["penetration"]["max_m"]):
                    problems.append(f"diagnostic_metric_not_bound_to_curves:{result['demo']}:{row['variant']}")
        for model in result.get("models", {}).values():
            rec = model["windows"]
            if hash_problems({rec["path"]: rec["sha256"]}):
                problems.append(f"diagnostic_windows_drift:{result['demo']}")
    return problems


def write_summary_products(report: dict, attempt: Path) -> dict:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fields = ("demo", "group", "variant", "teacher_fit_fk_m", "hand_fk_l_m", "hand_fk_r_m",
              "penetration_adaptive_max_m", "penetration_keyframe_ratio", "raw_standard_pass",
              "both_reference_frames", "both_certified_coverage", "seam_rotation_max_deg")
    flat = []
    for result in report["results"]:
        for row in result.get("rows", []):
            metrics = row["metrics"]
            flat.append(dict(zip(fields, (
                result["demo"], result["group"], row["variant"],
                metrics["fitting"]["teacher_fit_fk_mpjpe_box_m"],
                row["hand_fk_mean_m"]["l"], row["hand_fk_mean_m"]["r"],
                metrics["penetration"]["max_m"], metrics["penetration"]["keyframe_violation_ratio"],
                metrics["raw_standard_pass"], metrics["both"]["denominator_frames"],
                metrics["both"]["certified_coverage"], row["seam_rotation_max_deg"]))))
    csv_path = attempt / "paired_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    results = [r for r in report["results"] if r.get("status") == "ok"]
    x = np.arange(len(results))
    methods = ("teacher", "deterministic", "diffusion", "deterministic_all_contributors", "diffusion_all_contributors")
    for i, method in enumerate(methods):
        rows = [next(row for row in r["rows"] if row["variant"] == method) for r in results]
        axes[0].bar(x + (i - 2) * .16, [r["metrics"]["penetration"]["max_m"] * 1000 for r in rows],
                    width=.16, label=method)
        if method != "teacher":
            axes[1].plot(x, [r["metrics"]["fitting"]["teacher_fit_fk_mpjpe_box_m"] * 1000 for r in rows],
                         marker="o", linestyle="--" if "contributors" in method else "-", label=method)
    axes[0].set(ylabel="Adaptive maximum penetration (mm)", title="Fixed-checkpoint paired diagnosis; diagnostic counterfactuals only")
    axes[1].set(ylabel="Teacher-fit FK MPJPE (mm)")
    for ax in axes[:2]:
        ax.set_xticks(x, [r["demo"] for r in results], rotation=35, ha="right", fontsize=8)
        ax.axvline(5.5, color="gray", linestyle=":")
        ax.axvline(11.5, color="gray", linestyle=":")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(axis="y", alpha=.2)
    case = next((r for r in results if r["demo"] == KNOWN_CASE), None)
    if case:
        for method in methods[1:]:
            row = next(r for r in case["rows"] if r["variant"] == method)
            with np.load(ROOT / row["product"]["path"], allow_pickle=False) as z:
                jumps = z["curve__base_angular_velocity_radps"] / 30 * 180 / np.pi
            axes[2].plot(np.arange(len(jumps)), jumps, linestyle="--" if "contributors" in method else "-", label=method)
        axes[2].axvline(108, color="gray", linestyle=":", label="known tail start 108")
        axes[2].set(xlim=(90, min(case["T"] - 1, 140)), ylabel="Root rotation step (degrees)",
                    xlabel="Global frame", title="Known case sub16_largebox_048; same model windows")
        axes[2].legend(fontsize=8, ncol=2)
        axes[2].grid(alpha=.2)
    figure = attempt / "paired_diagnosis.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    return {"csv": file_record(csv_path), "figure": file_record(figure)}


def run_diagnosis(run: Path) -> bool:
    t0 = time.perf_counter()
    reg = verify_registration(run)
    costs = read_json(run / "costs.json")
    if len(costs) != 1 or costs[0].get("status") != "done":
        raise ValueError("history incomplete or a diagnosis was already attempted; no automatic retry")
    spent = costs[0]["wall_s"]
    deadline = t0 + BUDGET_S - RESERVE_S - spent
    attempt = run / "diagnosis/attempt_001"
    attempt.mkdir(parents=True, exist_ok=False)
    report_path = attempt / "attribution.json"
    costs.append({"stage": "diagnosis", "attempt": "attempt_001", "status": "running", "complete": False, "wall_s": 0.0})
    write_json(run / "costs.json", costs)
    write_json(run / "diagnosis/attempts_index.json", {"attempts": [{"attempt": "attempt_001", "status": "running", "complete": False}]}, exclusive=True)
    results, units = [], []
    report = {"status": "running", "complete": False, "registration": file_record(run / "freeze/registration.json"), "results": results}
    write_json(report_path, report, exclusive=True)
    try:
        for demo in reg["demos"]:
            check_deadline(deadline)
            drift = hash_problems(reg["anchors"])
            if drift:
                raise ValueError(drift)
            start = time.perf_counter()
            timeout = min(reg["unit_timeout_s"], deadline - start)
            argv = [sys.executable, str(Path(__file__)), "unit", "--run-root", str(run),
                    "--demo", demo, "--out", str(attempt / demo), "--deadline", str(start + timeout - 2)]
            env = {**os.environ, **{k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
            unit_rec = {"demo": demo, "argv": argv, "status": "running", "wall_s": 0.0}
            units.append(unit_rec)
            write_json(attempt / "units.json", units)
            try:
                with (attempt / f"{demo}.log").open("w") as log:
                    proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                unit_rec.update(status="done" if proc.returncode == 0 else "error", returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                unit_rec.update(status="timeout", returncode=None)
            finally:
                unit_rec["wall_s"] = time.perf_counter() - start
                write_json(attempt / "units.json", units)
            p = attempt / demo / "result.json"
            result = read_json(p) if p.is_file() else {"demo": demo, "status": "incomplete", "rows": []}
            if unit_rec["status"] != "done":
                result["status"] = unit_rec["status"]
            results.append(result)
            report["results"] = results
            write_json(report_path, report)
            print(json.dumps({"demo": demo, "status": result.get("status"), "wall_s": unit_rec["wall_s"]}), flush=True)
        summary = aggregate(results, reg["demos"])
        source_problems = hash_problems(reg["anchors"]) + verify_local_products(results)
        summary["evidence_gaps"] += source_problems
        summary["complete"] = summary["complete"] and not source_problems
        report.update(summary)
        report["source_anchors_unchanged"] = not source_problems
        report["summary_products"] = write_summary_products(report, attempt)
    except Exception as exc:
        report.update(complete=False, evidence_gaps=[f"diagnosis_exception:{type(exc).__name__}:{exc}"])
    finally:
        report.update(status="done" if report.get("complete") else "incomplete", results=results, units=units)
        # This independent, single-attempt diagnostic has no retry scheduler.
        # Publish under running until all substantive output writes are complete;
        # charge a registered final tail, then check coverage after registration.
        report["cost_s"] = time.perf_counter() - t0
        if spent + report["cost_s"] >= BUDGET_S - 2:
            report.update(complete=False, status="incomplete", budget_stopped=True)
        write_json(report_path, report)
        charged = time.perf_counter() - t0 + 2.0
        costs[-1].update(status=report["status"], complete=report.get("complete", False),
                         wall_s=charged, measured_before_tail_s=charged - 2.0, tail_reserve_s=2.0)
        write_json(run / "costs.json", costs)
        write_json(run / "diagnosis/attempts_index.json", {"attempts": [{"attempt": "attempt_001", "status": report["status"],
                    "complete": report.get("complete", False), "report": file_record(report_path), "wall_s": charged}]})
        if time.perf_counter() - t0 > charged or spent + charged >= BUDGET_S:
            report.update(complete=False, status="incomplete", budget_stopped=True)
            costs[-1].update(status="unknown", complete=False, wall_s=time.perf_counter() - t0)
            write_json(report_path, report)
            write_json(run / "costs.json", costs)
            write_json(run / "diagnosis/attempts_index.json", {"attempts": [{"attempt": "attempt_001", "status": "unknown", "complete": False}]})
    return report.get("complete") is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "run", "unit"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, default=SOURCE)
    parser.add_argument("--demo")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    run = guard_run(args.run_root, args.source_run_root)
    if args.stage == "register":
        register(run, args.source_run_root.resolve())
        return 0
    if args.stage == "unit":
        if args.out is None or not args.out.resolve().is_relative_to(run) or args.deadline is None:
            raise ValueError("unit output/deadline missing or outside run")
        result = unit(run, args.demo, args.out, args.deadline)
        return 0 if result.get("status") == "ok" else 2
    return 0 if run_diagnosis(run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
