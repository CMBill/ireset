"""Hash-bound val6 detail and visual replay for W1-D/E (§7, §8, §10).

Teacher uses seed None. Detail never generates: every curve and visual
consumes a hashed main-matrix product. Bounded subprocess units keep native
measurement/render calls inside the parent's monotonic work deadline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import omniretarget_w1_de as driver  # noqa: E402


def _record_path(path: Path, project_root: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(path)


def _json_safe(value):
    """Serialize undefined values as null; validation precedes serialization."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path, kind=dict):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, kind):
        raise ValueError(f"{Path(path).name}: expected {kind.__name__}")
    return value


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("unit_deadline")


def _save_npz(path: Path, **arrays) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _file_record(path: Path, project_root: Path) -> dict:
    from ireset.data.a2_integrity import sha256_file

    return {"path": _record_path(path, project_root), "sha256": sha256_file(path)}


def _load_indexed_report(run_root: Path, stage_name: str,
                         project_root: Path) -> tuple[dict | None, Path | None, dict, list[str]]:
    """Read the indexed attempt, including partial evidence; never fall back."""
    row, problems = driver.stage_state(run_root, driver.STAGES[stage_name], project_root)
    problems = list(problems)
    if row is None:
        return None, None, {}, problems
    record = row.get("report")
    if not isinstance(record, str):
        return row, None, {}, problems + ["report_path_missing"]
    path = (Path(project_root) / record).resolve()
    attempt = (Path(run_root) / driver.STAGE_DIRS[driver.STAGES[stage_name]]
               / row["attempt"]).resolve()
    if attempt not in path.parents:
        return row, path, {}, problems + ["report_outside_indexed_attempt"]
    try:
        report = _read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        return row, path, {}, problems + [f"report_unreadable:{type(exc).__name__}"]
    if row.get("complete") is not True:
        problems.append("attempt_incomplete")
    if row.get("status") in ("crashed", "running", "publishing", "timeout", "incomplete"):
        problems.append(f"attempt_state:{row['status']}")
    if report.get("crashed") or report.get("budget_stopped"):
        problems.append("report_noncompletion_state")
    if report.get("complete") is False:
        problems.append("report_incomplete")
    return row, path, report, problems


def _begin_attempt(run_root: Path, stage_name: str, report_name: str,
                   project_root: Path) -> tuple[Path, Path]:
    from ireset.data import a2_integrity as integ

    stage = driver.STAGES[stage_name]
    root = Path(run_root) / driver.STAGE_DIRS[stage]
    attempt = integ.allocate_attempt(root)
    report = attempt / report_name
    driver._upsert_cost(run_root, stage, attempt.name, 0.0,
                        {"complete": False}, status="running")
    driver.append_stage_index(root, {
        "attempt": attempt.name, "status": "running", "complete": False,
        "scope": {"stage": stage, "model": None, "attempt": attempt.name},
        "execution_registration_sha256": driver._registration_sha(run_root),
        "measured_wall_s": 0.0, "report": _record_path(report, project_root)})
    return attempt, report


def _finish(run_root: Path, stage_name: str, attempt: Path, report_path: Path,
            report: dict, t0: float, budget: dict, project_root: Path,
            artifacts: dict | None = None, exc: BaseException | None = None) -> bool:
    report = dict(report)
    if exc is not None:
        report.update(complete=False, crashed=True,
                      error=f"{type(exc).__name__}: {exc}")
    driver.atomic_json(report_path, _json_safe(report))
    row = {"report": _record_path(report_path, project_root),
           "complete": report.get("complete") is True, "artifacts": artifacts or {}}
    hist = budget.get("historical_s")
    extra = {"complete": row["complete"],
             "budget_consumed_cumulative_s":
                 (hist + time.perf_counter() - t0)
                 if isinstance(hist, (int, float)) else None}
    if exc is not None:
        return driver.crash_attempt(run_root, driver.STAGES[stage_name],
                                    attempt, t0, budget, row, extra, exc)
    return driver.finish_attempt(run_root, driver.STAGES[stage_name], attempt,
                                 t0, budget, row, extra)


def _run_unit(action: str, payload: dict, deadline: float,
              attempt_dir: Path, name: str) -> dict:
    """The parent kills a blocked native/render/GPU unit at the deadline."""
    _check_deadline(deadline)
    request = Path(attempt_dir) / f"unit__{name}.request.json"
    result = Path(attempt_dir) / f"unit__{name}.result.json"
    driver.atomic_json(request, {**payload, "deadline": deadline})
    argv = [sys.executable, str(Path(__file__).resolve()), "--unit", action,
            "--request", str(request), "--result", str(result)]
    timeout = deadline - time.perf_counter()
    if timeout <= 0:
        raise TimeoutError("unit_deadline_before_spawn")
    code, outcome, _ = driver._spawn(
        argv, Path(__file__).resolve().parents[1], timeout,
        Path(attempt_dir) / f"unit__{name}.log")
    if outcome == "timeout":
        raise TimeoutError(f"unit_timeout:{name}")
    _check_deadline(deadline)
    out = _read_json(result)
    if out.get("unit_error"):
        raise RuntimeError(out["unit_error"])
    if code != 0:
        raise RuntimeError(f"unit_exit:{name}:{code}")
    return out


def _load_clip(reg: dict, project_root: Path, demo: str):
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1

    manifest_path = project_root / reg["training_manifest"]["path"]
    if integ.sha256_file(manifest_path) != reg["training_manifest"]["sha256"]:
        raise ValueError("training_manifest_hash_mismatch")
    manifest = _read_json(manifest_path)
    matches = [r for r in manifest.get("rows", [])
               if isinstance(r, dict) and r.get("demo") == demo]
    if len(matches) != 1 or matches[0].get("split") != "val":
        raise ValueError("manifest_demo_not_unique_val_row")
    row = matches[0]
    if row.get("admission", {}).get("motion") is not True:
        raise ValueError("label_not_admitted")
    derived = row["derived"]
    if integ.sha256_file(project_root / derived["path"]) != derived["sha256"]:
        raise ValueError("derived_hash_mismatch")
    prereg_path = project_root / reg["preregistration"]["path"]
    if integ.sha256_file(prereg_path) != reg["preregistration"]["sha256"]:
        raise ValueError("preregistration_hash_mismatch")
    prereg = _read_json(prereg_path)
    cache = project_root / reg["cache"]["dir"]
    for filename, field in (("cache_identity.json", "identity_sha256"),
                            ("root_stats.npz", "root_stats_sha256")):
        if integ.sha256_file(cache / filename) != reg["cache"][field]:
            raise ValueError(f"{filename}:hash_mismatch")
    clip = w1.load_w1_derived_clip(row, cache / "schedules", project_root,
                                  prereg["assets"]["a1_assets_sha256"],
                                  w1.condition_code_sha256(project_root))
    if clip.T != row.get("T") or clip.T <= 0:
        raise ValueError("clip_T_mismatch")
    fps = row.get("fps")
    if (not isinstance(fps, (int, float)) or isinstance(fps, bool)
            or not math.isfinite(fps) or not math.isfinite(clip.condition.frame_dt)
            or clip.condition.frame_dt <= 0
            or abs(1.0 / clip.condition.frame_dt - fps) > 1e-12):
        raise ValueError("clip_frame_dt_mismatch")
    return clip, row


def _expected_identity(reg: dict, selection: dict, row: dict, condition,
                       manifest_row: dict, project_root: Path,
                       registration_sha: str, *, checkpoint_identities=None) -> dict:
    """Rebuild the full product identity from registered sources and checkpoint bytes."""
    import eval_generation as raw
    from ireset.data.a2_integrity import sha256_file

    cache_path = project_root / reg["cache"]["dir"] / "cache_identity.json"
    if sha256_file(cache_path) != reg["cache"]["identity_sha256"]:
        raise ValueError("identity_cache_hash_mismatch")
    method, checkpoint = row["method"], None
    if method != "teacher":
        rec = selection.get("models", {}).get(method)
        if not isinstance(rec, dict) or rec.get("role") == "unavailable":
            raise ValueError("selected_checkpoint_unavailable")
        if (rec.get("role"), rec.get("checkpoint_role")) not in (
                ("best", "best"), ("last", "last_diagnostic")):
            raise ValueError("selected_checkpoint_role_invalid")
        path = project_root / rec["path"]
        if sha256_file(path) != rec["sha256"]:
            raise ValueError("selected_checkpoint_hash_mismatch")
        identities = checkpoint_identities if checkpoint_identities is not None else {}
        if method not in identities:
            import torch

            ck = torch.load(path, map_location="cpu", weights_only=False)
            if (not isinstance(ck, dict) or ck.get("config") != reg["configs"][method]
                    or not isinstance(ck.get("identity"), dict)
                    or ck["identity"].get("config") != ck["config"]
                    or ck["identity"].get("model_type") != method
                    or type(ck.get("step")) is not int or ck["step"] != rec.get("step")):
                raise ValueError("selected_checkpoint_config_or_identity_mismatch")
            identities[method] = {
                "path": _record_path(path, project_root), "sha256": rec["sha256"],
                "step": ck["step"], "role": rec["checkpoint_role"],
                "formal_attempt": rec.get("formal_attempt"),
                "config_sha256": raw._w1_de_digest(ck["config"]), "identity": ck["identity"]}
        checkpoint = identities[method]
    return raw.w1_de_product_identity(row, reg, manifest_row, condition, checkpoint,
                                      registration_sha, _read_json(cache_path))


def _site_speed(points, frame_dt: float):
    """Per-site worst displacement / seconds; frame 0 is undefined."""
    import numpy as np

    points = np.asarray(points)
    if (points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0
            or not np.isfinite(points).all()
            or not math.isfinite(frame_dt) or frame_dt <= 0):
        raise ValueError("invalid_site_speed_input")
    out = np.full(points.shape[0], np.nan)
    out[1:] = np.linalg.norm(np.diff(points, axis=0), axis=-1).max(1) / frame_dt
    return out


def _interval_drift(points, ranges):
    import numpy as np

    out = np.full(points.shape[0], np.nan)
    for start, end in ranges:
        if not (0 <= start < end <= len(points)):
            raise ValueError("invalid_reference_interval")
        out[start:end] = np.linalg.norm(points[start:end] - points[start], axis=-1).max(1)
    return out


def _detail_curves(arrays: dict, clip, series: dict, report, validator) -> dict:
    import numpy as np
    from eval_generation import fk_keypoints, geodesic_deg
    from ireset.eval.teacher_comparison import _frames_of, _mask_from_ranges, certified_robot_mask
    from ireset.geometry.distance_adapter import world_to_box_geom
    from ireset.kinematics.mujoco_adapter import G1Kinematics

    T, dt = clip.T, clip.condition.frame_dt
    root, joints = arrays["root"], arrays["joints"]
    cond = clip.condition.truncate(T)
    teacher_scene = np.concatenate([clip.root, clip.q, cond.target_obj_pose], 1)
    teacher_kpts = fk_keypoints(G1Kinematics(), teacher_scene)
    curves = {
        "frame": np.arange(T), "time_s": np.arange(T) * dt,
        "task_world": series["kpt_err_w"], "task_box": series["kpt_err_b"],
        "fit_root_position_m": np.linalg.norm(root[:, :3] - clip.root[:, :3], axis=1),
        "fit_root_rotation_deg": geodesic_deg(root[:, 3:7], clip.root[:, 3:7]),
        "fit_joint_mae_rad": np.abs(joints - clip.q).mean(1),
        "fit_fk_mpjpe_box_m": np.linalg.norm(
            world_to_box_geom(series["robot_kpts"], cond.target_obj_pose)
            - world_to_box_geom(teacher_kpts, cond.target_obj_pose), axis=-1).mean(1),
        "penetration_keyframe": series["pen_key"],
        "penetration_adaptive_max_m": np.float64(report.checks["penetration"].worst),
        "box_ground": series["box_ground"], "velocity_defined": np.arange(T) > 0,
    }
    ref, certified = {}, {}
    for side, full in (("l", "left"), ("r", "right")):
        hand_ranges = _frames_of(cond.schedule, f"{full}_hand", "object")
        foot_ranges = _frames_of(cond.schedule, f"{full}_foot", "ground")
        robot_hand = [iv for iv in report.schedule.intervals
                      if iv.effector == f"{full}_hand" and iv.support == "object"]
        robot_foot = [iv for iv in report.schedule.intervals
                      if iv.effector == f"{full}_foot" and iv.support == "ground"]
        ref[side] = _mask_from_ranges(hand_ranges, T)
        hand_local = validator._cache["hand_local"][side]
        sole = validator._cache["sole_pts"][side]
        certified[side], _, _ = certified_robot_mask(
            robot_hand, series["gap"][side], hand_local, validator.cfg)
        curves.update({
            f"hand_gap_{side}": series["gap"][side],
            f"hand_reference_{side}": ref[side],
            f"hand_near_{side}": np.maximum(series["gap"][side], 0) < validator.cfg.hand_gap_m,
            f"hand_certified_{side}": certified[side],
            f"certified_overlap_{side}": certified[side] & ref[side],
            f"hand_relative_speed_{side}": _site_speed(hand_local, dt),
            f"hand_reference_drift_{side}": _interval_drift(hand_local, hand_ranges),
            f"hand_robot_anchor_drift_{side}": _interval_drift(
                hand_local, [(iv.start, iv.end) for iv in robot_hand]),
            f"foot_speed_{side}": _site_speed(sole, dt),
            f"foot_gap_{side}": validator._cache["sole_gap"][side].copy(),
            f"foot_reference_support_{side}": _mask_from_ranges(foot_ranges, T),
            f"foot_robot_support_{side}": _mask_from_ranges(
                [(iv.start, iv.end) for iv in robot_foot], T),
            f"foot_reference_drift_{side}": _interval_drift(sole, foot_ranges),
            f"foot_robot_anchor_drift_{side}": _interval_drift(
                sole, [(iv.start, iv.end) for iv in robot_foot]),
        })
    curves["both_reference"] = ref["l"] & ref["r"]
    curves["both_certified"] = certified["l"] & certified["r"]
    curves["both_certified_overlap"] = curves["both_reference"] & curves["both_certified"]
    position_step, rotation_step, joint_step = (np.full(T, np.nan) for _ in range(3))
    position_step[1:] = np.linalg.norm(np.diff(root[:, :3], axis=0), axis=1)
    rotation_step[1:] = geodesic_deg(root[:-1, 3:7], root[1:, 3:7])
    joint_step[1:] = np.abs(np.diff(joints, axis=0)).max(1)
    starts = np.asarray(arrays["window_starts"], dtype=np.int64)
    seam = np.zeros(T, dtype=bool)
    seam[starts[(starts > 0) & (starts < T)]] = True
    curves.update({
        "window_starts": starts, "seam_mask": seam,
        "base_linear_velocity_mps": position_step / dt,
        "base_angular_velocity_radps": np.deg2rad(rotation_step) / dt,
        "joint_max_velocity_radps": joint_step / dt,
        "seam_position_jump_m": np.where(seam, position_step, np.nan),
        "seam_rotation_jump_deg": np.where(seam, rotation_step, np.nan),
        "seam_joint_jump_rad": np.where(seam, joint_step, np.nan),
        "interior_velocity_reference": (np.arange(T) > 0) & ~seam,
        "root_excursion_m": np.linalg.norm(root[:, :3] - root[0, :3], axis=1),
        "root_path_length_m": np.r_[0.0, np.cumsum(position_step[1:])],
        "root_relative_teacher_drift_m": np.linalg.norm(
            (root[:, :3] - clip.root[:, :3]) - (root[0, :3] - clip.root[0, :3]), axis=1),
    })
    firsts = [v.first_failure_frame for v in report.checks.values()
              if not v.passed and v.first_failure_frame is not None]
    task_fail = np.flatnonzero(series["kpt_err_w"] > float(arrays["task_tol_m"]))
    if task_fail.size:
        firsts.append(int(task_fail[0]))
    first = min(firsts) if firsts else None
    curves["first_failure_frame"] = np.int64(-1 if first is None else first)
    curves["first_failure_marker"] = np.arange(T) == (-1 if first is None else first)
    assert set(curves) == set(_DETAIL_CURVE_CONTRACT), sorted(
        set(curves) ^ set(_DETAIL_CURVE_CONTRACT))
    return curves


#: The delivered per-row curve artifact contract: every key the writer
#: produces, with its validation kind. ``finite`` arrays are finite over all
#: T frames; ``mask`` arrays are boolean; NaN-capable kinds must place their
#: undefined entries exactly on the definition mask named by the kind
#: (``velocity_defined``, ``seam_mask`` or a per-side reference mask);
#: ``nan_or_finite`` permits all-NaN (no robot interval) but never inf.
_DETAIL_CURVE_CONTRACT: dict[str, str] = {
    "frame": "int", "time_s": "finite", "task_world": "finite",
    "task_box": "finite", "fit_root_position_m": "finite",
    "fit_root_rotation_deg": "finite", "fit_joint_mae_rad": "finite",
    "fit_fk_mpjpe_box_m": "finite", "penetration_keyframe": "finite",
    "box_ground": "finite", "root_excursion_m": "finite",
    "root_path_length_m": "finite", "root_relative_teacher_drift_m": "finite",
    "penetration_adaptive_max_m": "scalar_finite", "first_failure_frame": "scalar_int",
    "window_starts": "int_array", "velocity_defined": "mask",
    "seam_mask": "mask", "interior_velocity_reference": "mask",
    "first_failure_marker": "mask", "both_reference": "mask",
    "both_certified": "mask", "both_certified_overlap": "mask",
    "base_linear_velocity_mps": "nan_velocity",
    "base_angular_velocity_radps": "nan_velocity",
    "joint_max_velocity_radps": "nan_velocity",
    "seam_position_jump_m": "nan_seam", "seam_rotation_jump_deg": "nan_seam",
    "seam_joint_jump_rad": "nan_seam",
}
for _side in ("l", "r"):
    _DETAIL_CURVE_CONTRACT.update({
        f"hand_gap_{_side}": "finite",
        f"hand_reference_{_side}": "mask", f"hand_near_{_side}": "mask",
        f"hand_certified_{_side}": "mask", f"certified_overlap_{_side}": "mask",
        f"hand_relative_speed_{_side}": "nan_velocity",
        f"hand_reference_drift_{_side}": f"nan_mask:hand_reference_{_side}",
        f"hand_robot_anchor_drift_{_side}": "nan_or_finite",
        f"foot_speed_{_side}": "nan_velocity",
        f"foot_gap_{_side}": "finite",
        f"foot_reference_support_{_side}": "mask",
        f"foot_robot_support_{_side}": "mask",
        f"foot_reference_drift_{_side}": f"nan_mask:foot_reference_support_{_side}",
        f"foot_robot_anchor_drift_{_side}": "nan_or_finite",
    })


def _verify_detail_curves(arrays: dict, T: int, *, hand_gap_m=None) -> list[str]:
    """Full required-field/shape/definition-mask verification of a curve
    artifact — a hash-correct but stripped delivery is incomplete evidence."""
    import numpy as np

    problems = []
    keys = set(arrays)
    for missing in sorted(set(_DETAIL_CURVE_CONTRACT) - keys):
        problems.append(f"curves_field_missing:{missing}")
    for extra in sorted(keys - set(_DETAIL_CURVE_CONTRACT)):
        problems.append(f"curves_field_unexpected:{extra}")
    if problems:
        return problems
    for name, kind in _DETAIL_CURVE_CONTRACT.items():
        a = arrays[name]
        if kind.startswith("scalar"):
            if a.shape != ():
                problems.append(f"curves_field_not_scalar:{name}")
            elif kind == "scalar_finite" and not np.isfinite(a):
                problems.append(f"curves_field_nonfinite:{name}")
            elif kind == "scalar_int" and not (
                    np.issubdtype(a.dtype, np.integer) and int(a) >= -1):
                problems.append(f"curves_field_not_integer:{name}")
            continue
        if a.ndim != 1 or (kind != "int_array" and a.shape[0] != T):
            problems.append(f"curves_field_shape:{name}")
            continue
        if kind in ("int", "int_array"):
            if not np.issubdtype(a.dtype, np.integer):
                problems.append(f"curves_field_not_integer:{name}")
            elif kind == "int_array" and len(a) and (a.min() < 0 or a.max() >= T):
                problems.append(f"curves_field_out_of_range:{name}")
            continue
        finite = np.isfinite(a)
        if kind == "finite" and not finite.all():
            problems.append(f"curves_field_nonfinite:{name}")
        elif kind == "mask":
            if a.dtype != np.bool_:
                problems.append(f"curves_field_not_mask:{name}")
        elif kind in ("nan_velocity", "nan_seam") or kind.startswith("nan_mask:"):
            # Undefined entries are NaN ONLY: ±Inf is a bad measurement, not
            # an undefined slot, and must be rejected before the mask compare.
            if kind == "nan_velocity":
                mask = arrays["velocity_defined"]
            elif kind == "nan_seam":
                mask = arrays["seam_mask"]
            else:
                mask = arrays[kind.split(":", 1)[1]]
            if np.isinf(a).any():
                problems.append(f"curves_field_infinite:{name}")
            elif not np.array_equal(np.isnan(a), ~mask):
                problems.append(f"curves_field_undefined_mask:{name}")
        elif kind == "nan_or_finite" and not (finite | np.isnan(a)).all():
            problems.append(f"curves_field_nonfinite:{name}")
    if problems:
        return problems
    frame = arrays["frame"]
    if not np.array_equal(frame, np.arange(T)):
        problems.append("curves_frame_not_arange")
    if not np.array_equal(arrays["velocity_defined"], np.arange(T) > 0):
        problems.append("curves_velocity_defined_mask_invalid")
    for side in ("l", "r"):
        if not np.array_equal(arrays[f"certified_overlap_{side}"],
                              arrays[f"hand_certified_{side}"] & arrays[f"hand_reference_{side}"]):
            problems.append(f"curves_certified_overlap_invalid:{side}")
        if hand_gap_m is not None and not np.array_equal(
                arrays[f"hand_near_{side}"],
                np.maximum(arrays[f"hand_gap_{side}"], 0) < hand_gap_m):
            problems.append(f"curves_hand_near_invalid:{side}")
    if not np.array_equal(arrays["both_certified_overlap"],
                          arrays["both_reference"] & arrays["hand_certified_l"]
                          & arrays["hand_certified_r"]):
        problems.append("curves_both_intersection_invalid")
    first = int(arrays["first_failure_frame"])
    expected_marker = (np.arange(T) == first) if first >= 0 else np.zeros(T, dtype=bool)
    if not np.array_equal(arrays["first_failure_marker"], expected_marker):
        problems.append("curves_first_failure_marker_invalid")
    return problems


def _reference_product(clip, manifest_row: dict, path: Path,
                       project_root: Path, reg: dict) -> dict:
    import numpy as np

    cond = clip.condition.truncate(clip.T)
    for value in (cond.human_kps_world, cond.source_obj_pose, cond.target_obj_pose,
                  clip.root, clip.q):
        if not np.isfinite(value).all():
            raise ValueError("reference_replay_not_finite")
    with np.load(project_root / manifest_row["derived"]["path"], allow_pickle=False) as z:
        scale = float(z["similarity_scale"])
    intervals = [{"effector": iv.effector, "start": iv.start, "end": iv.end,
                  "support": iv.support} for iv in cond.schedule.intervals]
    _save_npz(path, human_kpts_world=cond.human_kps_world,
              source_box=cond.source_obj_pose, target_box=cond.target_obj_pose,
              source_box_similarity_scale=scale,
              teacher_root=clip.root, teacher_joints=clip.q,
              teacher_scene=clip.scene_qpos(), frame_dt=cond.frame_dt,
              reference_contact_flags=cond.contact_flags(),
              reference_intervals_json=json.dumps(intervals), demo=manifest_row["demo"],
              source_derived_sha256=manifest_row["derived"]["sha256"],
              training_manifest_sha256=reg["training_manifest"]["sha256"])
    return _file_record(path, project_root)


#: The delivered per-demo reference artifact contract (exact key set).
_REFERENCE_FIELDS = frozenset({
    "human_kpts_world", "source_box", "target_box",
    "source_box_similarity_scale", "teacher_root", "teacher_joints",
    "teacher_scene", "frame_dt", "reference_contact_flags",
    "reference_intervals_json", "demo", "source_derived_sha256",
    "training_manifest_sha256",
})


def _verify_reference_artifact(arrays: dict, row: dict, reg: dict,
                               clip=None, manifest_row: dict | None = None,
                               project_root=None) -> list[str]:
    """Complete-field verification of a delivered reference artifact, with
    bitwise binding to the FROZEN clip and the frozen derived scale when the
    caller supplies them — a hash-correct reference stripped of the source
    box, scale or teacher trajectory is an incomplete delivery, not passing
    evidence."""
    import numpy as np

    problems = []
    keys = set(arrays)
    for missing in sorted(_REFERENCE_FIELDS - keys):
        problems.append(f"reference_field_missing:{missing}")
    for extra in sorted(keys - _REFERENCE_FIELDS):
        problems.append(f"reference_field_unexpected:{extra}")
    if problems:
        return problems
    if str(arrays["demo"]) != row.get("demo"):
        problems.append("reference_demo_mismatch")
    if str(arrays["training_manifest_sha256"]) != reg["training_manifest"]["sha256"]:
        problems.append("reference_training_manifest_sha_mismatch")
    scale = arrays["source_box_similarity_scale"]
    if not ((np.isscalar(scale) or scale.shape == ())
            and math.isfinite(float(scale)) and float(scale) > 0):
        problems.append("reference_similarity_scale_invalid")
    elif manifest_row is not None and project_root is not None:
        # Positive is not enough: the delivered scale must BE the frozen
        # derived clip's similarity_scale (bitwise, same source file).
        try:
            with np.load(Path(project_root) / manifest_row["derived"]["path"],
                         allow_pickle=False) as z:
                want_scale = float(z["similarity_scale"])
            if float(scale) != want_scale:
                problems.append("reference_similarity_scale_not_frozen")
        except (OSError, ValueError, KeyError, TypeError):
            problems.append("reference_similarity_scale_unreadable")
    if manifest_row is not None and str(arrays["source_derived_sha256"]) != manifest_row["derived"]["sha256"]:
        problems.append("reference_source_derived_sha_mismatch")
    if clip is None:
        return problems
    cond = clip.condition.truncate(clip.T)
    T = clip.T
    expectations = {
        "human_kpts_world": (cond.human_kps_world, (T, 52, 3)),
        "source_box": (cond.source_obj_pose, (T, 7)),
        "target_box": (cond.target_obj_pose, (T, 7)),
        "teacher_root": (clip.root, (T, 7)),
        "teacher_joints": (clip.q, (T, 29)),
        "teacher_scene": (clip.scene_qpos(), (T, 43)),
        "reference_contact_flags": (cond.contact_flags(), None),
    }
    for name, (want, shape) in expectations.items():
        got = arrays[name]
        if got.shape != want.shape or (shape is not None and got.shape != shape) \
                or not np.array_equal(got, want):
            problems.append(f"reference_not_frozen_clip:{name}")
    if not (np.isscalar(arrays["frame_dt"]) or arrays["frame_dt"].shape == ()) \
            or float(arrays["frame_dt"]) != cond.frame_dt:
        problems.append("reference_frame_dt_not_frozen")
    intervals = [{"effector": iv.effector, "start": iv.start, "end": iv.end,
                  "support": iv.support} for iv in cond.schedule.intervals]
    if str(arrays["reference_intervals_json"]) != json.dumps(intervals):
        problems.append("reference_intervals_not_frozen")
    return problems


def _measure_unit(payload: dict) -> dict:
    import eval_generation as raw
    from ireset.data import a2_integrity as integ
    from ireset.eval.teacher_comparison import TeacherQualityReplayer
    from ireset.repair.validation import IndependentValidator, ValidationConfig

    project, deadline = Path(payload["project_root"]), payload.get("deadline")
    reg_path, report_path = Path(payload["registration_path"]), Path(payload["raw_report_path"])
    reg = _read_json(reg_path)
    if integ.sha256_file(reg_path) != payload["registration_sha256"]:
        raise ValueError("registration_hash_mismatch")
    if integ.sha256_file(report_path) != payload["raw_report_sha256"]:
        raise ValueError("raw_report_hash_mismatch")
    report = _read_json(report_path)
    clip, manifest_row = _load_clip(reg, project, payload["demo"])
    validator = IndependentValidator(cfg=ValidationConfig(**reg["evaluation"]["validation_config"]))
    if abs(validator.cfg.fps - 1.0 / clip.condition.frame_dt) > 1e-12:
        raise ValueError("validator_fps_mismatch")
    replayer = TeacherQualityReplayer(validator)
    selection_path = Path(payload["selection_path"])
    selection = _read_json(selection_path)
    if integ.sha256_file(selection_path) != payload["selection_sha256"]:
        raise ValueError("checkpoint_selection_hash_mismatch")
    result = {"demo": payload["demo"], "rows": [], "artifacts": {}, "reference": None}
    checkpoint_identities = {}
    detail = payload.get("detail") is True
    if detail:
        _check_deadline(deadline)
        path = Path(payload["attempt_dir"]) / f"reference__{payload['demo']}.npz"
        result["reference"] = _reference_product(clip, manifest_row, path, project, reg)
        result["artifacts"][result["reference"]["path"]] = result["reference"]["sha256"]
    for key in payload["keys"]:
        _check_deadline(deadline)
        hits = [r for r in report.get("rows", []) if isinstance(r, dict) and r.get("key") == key]
        if len(hits) != 1:
            result["rows"].append({"key": key, "status": "evidence_incomplete",
                                   "problems": ["raw_row_missing_or_duplicate"]})
            continue
        row = hits[0]
        rec = {"key": key, "matrix_key": key, "demo": row.get("demo"),
               "method": row.get("method"), "seed": row.get("seed"),
               "scene_product": row.get("product"), "raw_status": row.get("status"),
               "reference": result["reference"]}
        product = row.get("product")
        if not isinstance(product, dict) or not isinstance(product.get("path"), str):
            rec.update(status="evidence_incomplete", problems=["raw_product_missing"])
            result["rows"].append(rec)
            continue
        try:
            measured = raw.recompute_w1_de_product(
                project / product["path"], row, reg, clip, validator, replayer,
                project_root=project,
                expected_identity=_expected_identity(
                    reg, selection, row, clip.condition, manifest_row, project,
                    payload["registration_sha256"], checkpoint_identities=checkpoint_identities),
                deadline=deadline)
            verification, recomputed = measured["verification"], measured["row"]
            problems = list(verification.get("problems", []))
            problems.extend(raw.compare_w1_de_row_metrics(row, recomputed))
            rec.update(recomputed=recomputed, problems=problems,
                       consistent_with_raw=not problems,
                       numerical_valid=recomputed.get("numerical_valid"),
                       raw_standard_pass=recomputed.get("raw_standard_pass"), T=clip.T)
            if verification.get("passed") is not True or problems:
                rec["status"] = "evidence_incomplete"
            elif recomputed.get("numerical_valid") is False:
                rec.update(status="numerical_rejected", curves=None,
                           curve_unavailable_reason="numerical rejection; original output retained",
                           first_failure_frame=recomputed.get("first_failure_frame"))
            elif measured.get("series") is None or measured.get("report") is None:
                rec.update(status="evidence_incomplete", problems=["measured_series_or_report_missing"])
            else:
                rec["status"] = "ok"
                rec["first_failure_frame"] = recomputed.get("robot_strict", {}).get("first_failure_frame")
                if detail:
                    _check_deadline(deadline)
                    arrays = dict(verification["arrays"])
                    arrays["task_tol_m"] = reg["evaluation"]["task_gate"]["tol_m"]
                    curves = _detail_curves(arrays, clip, measured["series"], measured["report"], validator)
                    path = Path(payload["attempt_dir"]) / f"curves__{row['method']}__{row['demo']}.npz"
                    _save_npz(path, **curves)
                    rec["curves"] = _file_record(path, project)
                    rec["first_failure_frame"] = (None if int(curves["first_failure_frame"]) < 0
                                                   else int(curves["first_failure_frame"]))
                    result["artifacts"][rec["curves"]["path"]] = rec["curves"]["sha256"]
        except TimeoutError:
            raise
        except (Exception, SystemExit) as exc:
            rec.update(status="evidence_incomplete", consistent_with_raw=False,
                       problems=[f"product_replay_failed:{type(exc).__name__}:{exc}"])
        result["rows"].append(rec)
        _check_deadline(deadline)
    if integ.sha256_file(report_path) != payload["raw_report_sha256"]:
        raise ValueError("raw_report_changed_during_measurement")
    return _json_safe(result)


def _representative_cases(rows: list[dict], demos: list[str]) -> dict:
    cases = {}
    for name, passing in (("pass_case", True), ("fail_case", False)):
        matches = sorted({r["demo"] for r in rows if r.get("demo") in demos
                          and r.get("status") in ("ok", "numerical_rejected")
                          and r.get("consistent_with_raw") is True
                          and r.get("raw_standard_pass") is passing})
        cases[name] = ({"demo": matches[0]} if matches else
                       {"demo": None, "reason": "no verified case of this class in fixed val6"})
    return cases


def _add_visual_reference(renderer, human, source_box, target_box) -> None:
    import mujoco
    import numpy as np
    from ireset.embodied.contact_sites import SMPLH_INDEX

    scene = renderer.scene

    def sphere(point, color, radius=0.012):
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("replay_visual_geometry_capacity")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, radius), np.asarray(point), np.eye(3).ravel(),
                           np.asarray(color, dtype=np.float32))
        scene.ngeom += 1

    def connector(a, b, color):
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("replay_visual_geometry_capacity")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                           np.asarray(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008,
                             np.asarray(a), np.asarray(b))
        scene.ngeom += 1

    orange = (1.0, 0.65, 0.15, 1.0)
    for point in human:
        sphere(point, orange, 0.008)
    chains = (("Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"),
              ("Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe"),
              ("Pelvis", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"),
              ("Chest", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist"),
              ("Chest", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"))
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            connector(human[SMPLH_INDEX[a]], human[SMPLH_INDEX[b]], orange)
    sphere(source_box[:3], (0.1, 1.0, 1.0, 1.0), 0.025)
    sphere(target_box[:3], (1.0, 0.15, 0.85, 1.0), 0.025)
    connector(source_box[:3], target_box[:3], (0.9, 0.9, 0.9, 1.0))


def _render_scene_gif(scene, out_path: Path, stride: int = 3, *, frame_dt=1 / 30,
                      human=None, source_box=None, deadline=None) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from PIL import Image
    from ireset.embodied.g1_spec import G1Spec

    scene = np.asarray(scene)
    if (scene.ndim != 2 or scene.shape[1] != 43 or scene.shape[0] == 0
            or not np.isfinite(scene).all() or stride <= 0
            or not math.isfinite(frame_dt) or frame_dt <= 0):
        raise ValueError("render_scene_not_finite_or_wrong_shape")
    model = G1Spec().load_scene()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    frames = []
    indices = sorted(set(range(0, len(scene), stride)) | {len(scene) - 1})
    durations = [max(1, round((b - a) * frame_dt * 1000))
                 for a, b in zip(indices, indices[1:])] + [max(1, round(frame_dt * 1000))]
    try:
        for t in indices:
            _check_deadline(deadline)
            data.qpos[:] = scene[t]
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = scene[t, :3] + (0.0, 0.0, 0.3)
            camera.distance, camera.azimuth, camera.elevation = 3.5, 130.0, -15.0
            renderer.update_scene(data, camera=camera)
            if human is not None:
                _add_visual_reference(renderer, human[t], source_box[t], scene[t, 36:43])
            frames.append(Image.fromarray(renderer.render().copy()))
        _check_deadline(deadline)
        temporary = Path(out_path).with_suffix(".tmp.gif")
        frames[0].save(temporary, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0)
        os.replace(temporary, out_path)
    finally:
        renderer.close()


def _render_case_unit(payload: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from ireset.data import a2_integrity as integ
    from ireset.embodied.g1_spec import G1Spec

    project = Path(payload["project_root"])
    ref = payload["reference"]
    ref_path = project / ref["path"]
    if integ.sha256_file(ref_path) != ref["sha256"]:
        raise ValueError("render_reference_hash_mismatch")
    with np.load(ref_path, allow_pickle=False) as z:
        human, source, target = z["human_kpts_world"], z["source_box"], z["target_box"]
        teacher = z["teacher_scene"]
        dt, scale = float(z["frame_dt"]), float(z["source_box_similarity_scale"])
    scenes = {}
    for rec in payload["rows"]:
        if rec.get("status") != "ok":
            continue
        product = rec["scene_product"]
        path = project / product["path"]
        if integ.sha256_file(path) != product["sha256"]:
            raise ValueError("render_product_hash_mismatch")
        with np.load(path, allow_pickle=False) as z:
            scene = z["scene"].copy()
        if scene.shape != teacher.shape or not np.isfinite(scene).all():
            raise ValueError("render_product_not_finite_or_wrong_shape")
        scenes[rec["method"]] = scene
    model = G1Spec().load_scene()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    frames, deadline = [], payload.get("deadline")
    try:
        indices = sorted(set(range(0, len(teacher), 3)) | {len(teacher) - 1})
        durations = [max(1, round((b - a) * dt * 1000))
                     for a, b in zip(indices, indices[1:])] + [max(1, round(dt * 1000))]
        for t in indices:
            _check_deadline(deadline)
            panels = []
            for method in ("reference", *driver.METHODS):
                if method != "reference" and method not in scenes:
                    panel = Image.new("RGB", (320, 240), (25, 25, 25))
                    ImageDraw.Draw(panel).text((8, 20), f"{method}: output unavailable\noriginal negative evidence retained", fill="white")
                else:
                    scene = teacher if method == "reference" else scenes[method]
                    data.qpos[:] = scene[t]
                    mujoco.mj_forward(model, data)
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = target[t, :3]
                    camera.distance, camera.azimuth, camera.elevation = 3.5, 130.0, -15.0
                    renderer.update_scene(data, camera=camera)
                    if method == "reference":
                        for i in range(renderer.scene.ngeom):
                            geom = renderer.scene.geoms[i]
                            if (geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0
                                    and model.geom_bodyid[geom.objid] != 0):
                                geom.rgba[3] = 0.0
                        _add_visual_reference(renderer, human[t], source[t], target[t])
                    panel = Image.fromarray(renderer.render().copy())
                    caption = (f"human; cyan source / magenta target; scale {scale:.4f}"
                               if method == "reference" else f"{method} raw")
                    ImageDraw.Draw(panel).text((5, 5), caption, fill="white", stroke_width=1, stroke_fill="black")
                panels.append(panel)
            frame = Image.new("RGB", (1280, 265), (20, 20, 20))
            for i, panel in enumerate(panels):
                frame.paste(panel, (320 * i, 25))
            ImageDraw.Draw(frame).text((6, 6), f"{payload['case']} / {payload['demo']} / frame {t}", fill="white")
            frames.append(frame)
        _check_deadline(deadline)
        gif = Path(payload["gif_path"])
        temporary = gif.with_suffix(".tmp.gif")
        frames[0].save(temporary, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0)
        os.replace(temporary, gif)
    finally:
        renderer.close()
    _check_deadline(deadline)
    return {"demo": payload["demo"], "gif": _file_record(gif, project),
            "frames": indices, "duration_ms": durations,
            "includes_last_frame": indices[-1] == len(teacher) - 1,
            "source_box_visualization": "source/target origins; registered source scale, no unscaled source mesh"}


def stage_detail(args) -> int:
    from ireset.data import a2_integrity as integ
    from ireset.paths import PROJECT_ROOT
    import eval_generation as raw

    t0, project = time.perf_counter(), Path(PROJECT_ROOT)
    run = integ.guard_output_dir(Path(args.run_root))
    reg_path = run / "freeze/execution_registration.json"
    reg = _read_json(reg_path)
    entry_problems = driver.verify_execution_registration(run, project)
    budget = driver._budget_state(run, "detail_f")
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    attempt, report_path = _begin_attempt(run, "detail", "detail_report.json", project)
    report = {"stage": "W1-D/E val6 detail", "complete": False, "rows": [],
              "problems": list(entry_problems), "budget_stopped": False}
    artifacts = {}
    try:
        raw_att, raw_path, raw_report, problems = _load_indexed_report(run, "raw-eval", project)
        report["problems"].extend(f"raw-eval:{p}" for p in problems)
        matrix, val6 = raw.w1_de_expected_matrix(reg), list(reg["rosters"]["val6"])
        if not val6 or len(val6) != len(set(val6)) or not set(val6) <= set(reg["rosters"]["val31"]):
            raise ValueError("registered_val6_empty_duplicate_or_foreign")
        expected = [r for r in matrix if r["matrix"] == "main" and r["demo"] in val6]
        if len(expected) != len(val6) * len(driver.METHODS):
            raise ValueError("val6_main_matrix_contract_mismatch")
        report.update(val6=val6, methods=list(driver.METHODS), main_seed=reg["evaluation"]["main_seed"],
                      raw_attempt=(raw_att or {}).get("attempt"),
                      raw_eval_report=(_file_record(raw_path, project)
                                       if raw_path is not None and raw_path.is_file() else None))
        report["rows"] = [{**r, "matrix_key": r["key"], "status": "not_run"} for r in expected]
        selection_path = run / "evaluation/checkpoint_selection.json"
        selection_sha = integ.sha256_file(selection_path) if selection_path.is_file() else None
        if selection_sha is None:
            report["problems"].append("checkpoint_selection_missing")
        driver.atomic_json(report_path, report)
        by_key = {r["key"]: r for r in report["rows"]}
        if not entry_problems and raw_path is not None and raw_report and selection_sha:
            for demo in sorted(val6):
                try:
                    result = _run_unit("measure", {
                        "project_root": str(project), "demo": demo,
                        "registration_path": str(reg_path), "registration_sha256": integ.sha256_file(reg_path),
                        "raw_report_path": str(raw_path), "raw_report_sha256": integ.sha256_file(raw_path),
                        "selection_path": str(selection_path), "selection_sha256": selection_sha,
                        "keys": [r["key"] for r in expected if r["demo"] == demo],
                        "attempt_dir": str(attempt), "detail": True},
                        deadline, attempt, f"detail__{demo}")
                    for rec in result["rows"]:
                        by_key[rec["key"]].update(rec)
                    artifacts.update(result.get("artifacts", {}))
                except TimeoutError as exc:
                    report["budget_stopped"] = True
                    report["problems"].append(str(exc))
                    break
                except (Exception, SystemExit) as exc:
                    for rec in by_key.values():
                        if rec["demo"] == demo:
                            rec.update(status="evidence_incomplete", problems=[f"detail_unit_failed:{type(exc).__name__}:{exc}"])
                driver.atomic_json(report_path, _json_safe(report))
        cases = _representative_cases(report["rows"], val6)
        for name, case in cases.items():
            if case["demo"] is None:
                continue
            demo_rows = [r for r in report["rows"] if r["demo"] == case["demo"]]
            reference = next((r.get("reference") for r in demo_rows if r.get("reference")), None)
            if not reference:
                case.update(complete=False, render_error="reference_unavailable")
                continue
            try:
                rendered = _run_unit("render", {
                    "project_root": str(project), "demo": case["demo"], "case": name,
                    "reference": reference, "rows": demo_rows,
                    "gif_path": str(attempt / f"summary__{name}__{case['demo']}.gif")},
                    deadline, attempt, f"render__{name}")
                case.update(rendered, complete=True)
                artifacts[rendered["gif"]["path"]] = rendered["gif"]["sha256"]
            except TimeoutError as exc:
                case.update(complete=False, render_error=str(exc))
                report["budget_stopped"] = True
            except (Exception, SystemExit) as exc:
                case.update(complete=False, render_error=f"{type(exc).__name__}:{exc}")
        rows = report["rows"]
        verified = [r for r in rows if r.get("status") in ("ok", "numerical_rejected")
                    and r.get("consistent_with_raw") is True]
        unresolved = [r["key"] for r in rows if r not in verified]
        consistent = bool(rows) and len(verified) == len(expected)
        render_complete = all(c.get("demo") is None or c.get("complete") is True for c in cases.values())
        report.update(representative_cases=cases,
                      counts={"expected_rows": len(expected), "n_rows": len(rows), "processed": len(verified),
                              "n_ok": sum(r.get("status") == "ok" for r in rows),
                              "numerical_rejected": sum(r.get("status") == "numerical_rejected" for r in rows),
                              "n_not_ok": len(unresolved)},
                      replay_consistency={"all_headline_facts_reproduced": consistent, "problems": unresolved},
                      curves_contract={"speed_first_frame": "undefined; NaN plus velocity_defined mask",
                                       "interval_drift_outside_reference": "undefined; NaN",
                                       "penetration_curve": "keyframes; adaptive SLERP maximum and first failure separate",
                                       "numerical_rejections": "source bytes retained; geometric curves unavailable"},
                      artifacts=artifacts,
                      complete=bool(consistent and render_complete and not report["budget_stopped"]
                                    and not entry_problems and not problems))
        finished = _finish(run, "detail", attempt, report_path, report, t0, budget, project, artifacts)
    except BaseException as exc:
        _finish(run, "detail", attempt, report_path, report, t0, budget, project, artifacts, exc)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 2
    print(json.dumps({"detail_complete": finished, "counts": report["counts"], "problems": report["problems"]}, indent=2))
    return 0 if finished else 2


def _unit_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", choices=("measure", "render", "checkpoint"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    payload = _read_json(args.request)
    try:
        if args.unit == "measure":
            result = _measure_unit(payload)
        elif args.unit == "render":
            result = _render_case_unit(payload)
        else:
            from omniretarget_w1_de_finalize import _checkpoint_replay_unit
            result = _checkpoint_replay_unit(payload)
        driver.atomic_json(args.result, _json_safe(result))
        return 0
    except BaseException as exc:
        driver.atomic_json(args.result, {"unit_error": f"{type(exc).__name__}:{exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(_unit_main())
