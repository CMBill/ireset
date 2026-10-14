"""Focused W1-D/E closure boundaries; no formal training or authoritative run."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import eval_generation as raw
import omniretarget_w1_de as driver
import omniretarget_w1_de_detail as detail
import omniretarget_w1_de_finalize as finalize
from ireset.data import w1_training as w1
from ireset.data.a2_integrity import sha256_file
from ireset.data.schema import ReferenceCondition, RobotTrajectory


def _condition(demo="demo", T=60):
    root = np.zeros((T, 7)); root[:, 2] = .8; root[:, 3] = 1
    source = root.copy(); source[:, 0] = .8; source[:, 2] = .6
    target = source.copy(); target[:, 2] += .1
    human = np.zeros((T, 52, 3)); human[..., 2] = .9
    cond = ReferenceCondition(demo, human, source, target, world_from_clip=np.eye(4))
    return cond, root, np.zeros((T, 29))


def _registration(demos):
    return {"execution_id": "w1_de_v1", "label_protocol": w1.MOTION_PROTOCOL_VERSION,
            "training_purpose": w1.TRAINING_PURPOSE,
            "rosters": {"val31": list(demos), "val6": list(demos)},
            "training_manifest": {"sha256": "a" * 64},
            "cache": {"dir": "cache", "identity_sha256": "b" * 64,
                      "root_stats_sha256": "c" * 64},
            "condition_code_sha256": "d" * 64, "assets_sha256": "e" * 64,
            "configs": {"deterministic": {"model_type": "deterministic", "steps": 2}},
            "evaluation": {"methods": ["teacher", "deterministic", "diffusion"],
                           "main_seed": 7, "appendix_seeds": [17, 27],
                           "sampling": raw.W1_DE_SAMPLING}}


def _teacher_metadata():
    return {"teacher_raw": True, "starts": [], "windows": 0,
            "model_calls": 0, "resample_attempts_used": 0, "resample_attempts_max": 0,
            "attempt_seeds": [], "per_attempt_invalid_windows": [],
            "per_attempt_invalid_window_indices": [], "final_invalid_windows": 0,
            "final_invalid_window_indices": [], "last_frame_covered": True}


def test_full_identity_rebuilt_from_source_and_selected_checkpoint(tmp_path):
    cond, root, joints = _condition()
    reg = _registration(["demo"])
    cache = tmp_path / "cache"; cache.mkdir()
    cache_doc = {"assets_sha256": reg["assets_sha256"],
                 "schedule_rows": [{"demo": "demo", "sha256": "f" * 64}]}
    driver.atomic_json(cache / "cache_identity.json", cache_doc)
    reg["cache"]["identity_sha256"] = sha256_file(cache / "cache_identity.json")
    registered_row = {"demo": "demo", "inputs": {"pt_sha256": "1" * 64},
                      "derived": {"path": "derived.npz", "sha256": "2" * 64}}
    row = {"key": "teacher|demo|None", "method": "teacher", "demo": "demo",
           "seed": None, "matrix": "main", "T": 60}
    ident = detail._expected_identity(reg, {"models": {}}, row, cond,
                                     registered_row, tmp_path, "3" * 64)
    assert ident["checkpoint"] is None and ident["seed"] is None
    assert ident["inputs"] == registered_row["inputs"]
    metadata = _teacher_metadata()
    row["product"] = raw.save_w1_de_product(
        tmp_path / "teacher.npz", row, ident, cond,
        trajectory=RobotTrajectory(root, joints, metadata=metadata),
        metadata=metadata, project_root=tmp_path)
    assert raw.verify_w1_de_product(tmp_path / "teacher.npz", row, reg, cond,
        project_root=tmp_path, expected_identity=ident)["passed"]
    # Mutating a source leg while leaving product hashes untouched is caught by full identity.
    changed = {**registered_row, "inputs": {"pt_sha256": "4" * 64}}
    fresh = detail._expected_identity(reg, {}, row, cond, changed, tmp_path, "3" * 64)
    check = raw.verify_w1_de_product(tmp_path / "teacher.npz", row, reg, cond,
                                     project_root=tmp_path, expected_identity=fresh)
    assert not check["passed"] and "product_expected_identity_mismatch" in check["problems"]
    # Selected identity comes from checkpoint bytes, including full config and role.
    cfg = reg["configs"]["deterministic"]
    ck_ident = {"model_type": "deterministic", "config": cfg}
    torch.save({"config": cfg, "identity": ck_ident, "step": 2}, tmp_path / "best.pt")
    selected = {"path": "best.pt", "sha256": sha256_file(tmp_path / "best.pt"),
                "step": 2, "role": "best", "checkpoint_role": "best",
                "formal_attempt": "attempt_001"}
    model_row = {**row, "key": "deterministic|demo|7", "method": "deterministic", "seed": 7}
    fresh = detail._expected_identity(reg, {"models": {"deterministic": selected}},
                                     model_row, cond, registered_row, tmp_path, "3" * 64)
    assert fresh["checkpoint"]["identity"] == ck_ident
    assert fresh["checkpoint"]["config_sha256"] == raw._w1_de_digest(cfg)
    assert fresh["checkpoint"]["role"] == "best"
    with pytest.raises(ValueError, match="checkpoint_hash"):
        detail._expected_identity(reg, {"models": {"deterministic": {**selected, "sha256": "0" * 64}}},
                                  model_row, cond, registered_row, tmp_path, "3" * 64)
    wrong_target = copy.deepcopy(cond); wrong_target.target_obj_pose[:, 2] += .01
    assert not raw.verify_w1_de_product(tmp_path / "teacher.npz", row, reg, wrong_target,
                                       project_root=tmp_path, expected_identity=ident)["passed"]


def test_site_speed_uses_seconds_and_worst_site():
    points = np.zeros((3, 2, 3))
    points[1, 0, 0] = .02; points[1, 1, 0] = -.01
    points[2, 0, 0] = .02; points[2, 1, 0] = .03
    speed = detail._site_speed(points, .02)
    assert np.isnan(speed[0])
    np.testing.assert_array_equal(speed[1:], [1.0, 2.0])
    # Averaging site centers would hide this opposing-site motion.
    assert np.linalg.norm(points[1].mean(0) - points[0].mean(0)) / .02 < speed[1]
    for dt in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            detail._site_speed(points, dt)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_replay_and_evidence_finite_before_tolerance(bad):
    cond, root, joints = _condition(T=2)
    assert finalize._replay_arrays(root, joints, root, joints,
        cond.target_obj_pose, cond.target_obj_pose, T=2)["within_tolerance"]
    poisoned = root.copy(); poisoned[-1, 0] = bad
    out = finalize._replay_arrays(poisoned, joints, root, joints,
                                 cond.target_obj_pose, cond.target_obj_pose, T=2)
    assert not out["all_finite"] and not out["within_tolerance"]
    assert out["max_root_abs_diff"] is None
    assert finalize._compare_evidence({"metric": bad}, {"metric": bad})
    assert not finalize._replay_arrays(root, joints, root, joints,
        cond.target_obj_pose, cond.target_obj_pose, T=2, tol=bad)["within_tolerance"]


def test_replay_shape_target_and_empty_checkpoint_guards(tmp_path, monkeypatch):
    cond, root, joints = _condition(T=2)
    assert not finalize._replay_arrays(root[:-1], joints, root, joints,
        cond.target_obj_pose, cond.target_obj_pose, T=2)["within_tolerance"]
    target = cond.target_obj_pose.copy(); target[-1, 0] += 1e-12
    out = finalize._replay_arrays(root, joints, root, joints, target,
                                 cond.target_obj_pose, T=2)
    assert out["all_finite"] and not out["box_binding_bitwise"] and not out["within_tolerance"]
    monkeypatch.setattr(finalize, "_run_unit", lambda *a, **k: pytest.fail("unavailable checkpoint was executed"))
    reg = _registration(["demo"])
    result = finalize._replay_selected_checkpoints(tmp_path, tmp_path, reg, {"rows": []},
        {"models": {}}, attempt_dir=tmp_path, deadline=time.perf_counter() + 5)
    assert not result["complete"] and set(result["models"]) == set(driver.MODELS)
    assert all(r["verified"] is False for r in result["models"].values())


def _negative_detail(tmp_path):
    demos = [f"demo{i}" for i in range(6)]
    reg = _registration(demos)
    T = 60
    cond, root, joints = _condition(T=T)
    negative = tmp_path / "negative.npz"
    detail._save_npz(negative, normalized_windows=np.full((1, 60, 38), np.nan))
    product = detail._file_record(negative, tmp_path)
    refs = {}
    for demo in demos:
        reference = tmp_path / f"reference__{demo}.npz"
        detail._save_npz(reference, human_kpts_world=cond.human_kps_world,
                         source_box=cond.source_obj_pose, target_box=cond.target_obj_pose,
                         source_box_similarity_scale=.75, teacher_root=root, teacher_joints=joints,
                         teacher_scene=np.concatenate([root, joints, cond.target_obj_pose], 1),
                         frame_dt=cond.frame_dt, reference_contact_flags=np.zeros(T, dtype=bool),
                         reference_intervals_json="[]", demo=demo,
                         source_derived_sha256="2" * 64,
                         training_manifest_sha256=reg["training_manifest"]["sha256"])
        refs[demo] = detail._file_record(reference, tmp_path)
    main = [r for r in raw.w1_de_expected_matrix(reg) if r["matrix"] == "main"]
    sources, rows = [], []
    for wanted in main:
        source = {**wanted, "T": T, "status": "numerical_rejected",
                  "numerical_valid": False, "raw_standard_pass": False, "product": product}
        sources.append(source)
        rows.append({**wanted, "T": T, "status": "numerical_rejected", "scene_product": product,
                     "consistent_with_raw": True, "raw_standard_pass": False,
                     "recomputed": source, "reference": refs[wanted["demo"]], "curves": None,
                     "curve_unavailable_reason": "numerical rejection; original bytes retained"})
    raw_path = tmp_path / "raw.json"; driver.atomic_json(raw_path, {"rows": sources})
    from PIL import Image
    gif = tmp_path / "case.gif"; Image.new("RGB", (8, 8)).save(gif)
    cases = detail._representative_cases(rows, demos)
    cases["fail_case"].update(complete=True, gif=detail._file_record(gif, tmp_path))
    report = {"rows": rows, "val6": demos, "raw_eval_report": detail._file_record(raw_path, tmp_path),
              "representative_cases": cases, "complete": True, "budget_stopped": False}
    return report, reg, raw_path, sources


def test_all18_negative_detail_retains_evidence_without_fabricated_curves(tmp_path):
    report, reg, raw_path, sources = _negative_detail(tmp_path)
    out = finalize._verify_detail(report, reg, raw_path, sources, tmp_path)
    assert out["complete"] and out["processed"] == out["expected"] == 18
    teacher = [r for r in report["rows"] if r["method"] == "teacher"]
    assert len(teacher) == 6 and all(r["seed"] is None for r in teacher)
    assert all(r["curves"] is None for r in report["rows"])
    assert report["representative_cases"]["pass_case"]["demo"] is None
    assert report["representative_cases"]["pass_case"]["reason"]
    assert report["representative_cases"]["fail_case"]["demo"] == "demo0"
    with np.load(tmp_path / "negative.npz") as z:
        assert np.isnan(z["normalized_windows"]).all()


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "teacher_seed", "pass_flag", "unknown", "source_hash"])
def test_detail_matrix_pass_flag_unknown_and_product_tamper_rejected(tmp_path, mutation):
    report, reg, raw_path, sources = _negative_detail(tmp_path)
    if mutation == "drop":
        report["rows"].pop()
    elif mutation == "duplicate":
        report["rows"].append(copy.deepcopy(report["rows"][0]))
    elif mutation == "teacher_seed":
        report["rows"][0].update(key="teacher|demo0|7", seed=7)
    elif mutation == "pass_flag":
        report["rows"][0]["raw_standard_pass"] = True
    elif mutation == "unknown":
        report["rows"][0]["status"] = "not_run"
    else:
        report["rows"][0]["scene_product"] = {"path": "negative.npz", "sha256": "0" * 64}
    out = finalize._verify_detail(report, reg, raw_path, sources, tmp_path)
    assert not out["complete"] and out["problems"]


def test_attribution_uses_macro_dictionary_and_never_auto_executes():
    evaluation = {"evaluation_complete": True, "per_method": {"deterministic|7": {
        "counts": {"strict_rejected": 31, "raw_standard_pass": 0},
        "macros": {"teacher_fit_fk_mpjpe_box_m": {"macro_all": .02, "n_defined": 31}}}}}
    route = finalize._route_judgment({"training_complete": True, "models": {}}, evaluation)
    assert route["auto_execute"] is False
    assert route["primary_attribution"] == "attribution insufficient"
    assert route["label_principal_bottleneck_supported"] is False
    assert route["evidence"][0]["teacher_fit"]["macro_all"] == .02


def test_structured_negative_checkpoint_replay_compares_masks_and_retry_facts():
    from ireset.model.generator import GenerationNumericalError

    output = np.zeros((1, 60, 38), dtype=np.float32)
    output[..., 3] = 1; output[..., 7] = 1; output[0, 0, 0] = np.nan
    metadata = {"starts": [0], "model_calls": 1, "resample_attempts_used": 1,
                "resample_attempts_max": 3, "attempt_seeds": [7],
                "per_attempt_invalid_windows": [1], "per_attempt_invalid_window_indices": [[0]],
                "final_invalid_windows": 1, "final_invalid_window_indices": [0],
                "attempts": [{"attempt_index": 0, "seed": 7,
                              "replacement_window_indices": [0], "invalid_window_indices": [0], "wall_s": .5}]}
    exc = GenerationNumericalError("invalid_windows_after_attempts", metadata, output)
    stored = copy.deepcopy(exc.metadata)
    stored["attempts"][0]["wall_s"] = .8  # runtime duration is not a same-seed numerical fact
    arrays = {"normalized_windows": output.copy(), "window_starts": np.array([0])}
    check = finalize._replay_negative_output(exc, arrays, stored, method="deterministic", seed=7, T=60)
    assert check["verified"] and check["numerical_rejection_reproduced"]
    arrays["normalized_windows"][0, 0, 0] = np.inf
    assert not finalize._replay_negative_output(exc, arrays, stored,
        method="deterministic", seed=7, T=60)["verified"]
    arrays["normalized_windows"] = output.copy()
    arrays["normalized_windows"][0, 1, 0] = .1
    assert not finalize._replay_negative_output(exc, arrays, stored,
        method="deterministic", seed=7, T=60)["verified"]
    arrays["normalized_windows"] = output.copy()
    stored["attempt_seeds"] = [17]
    assert not finalize._replay_negative_output(exc, arrays, stored,
        method="deterministic", seed=7, T=60)["verified"]


def test_training_summary_has_own_hash_binding_and_requires_complete_state(tmp_path, monkeypatch):
    reg = _registration(["val"])
    reg["rosters"]["train24"] = ["train"]
    cfg = {"model_type": "deterministic", "steps": 2, "micro_batch": 2,
           "grad_accum": 1, "lr": 1e-4, "val_every": 1}
    reg["configs"]["deterministic"] = cfg
    reg["configs"]["formal_validation_contract"] = {"full_generation_every_optimizer_steps": 1}
    run = tmp_path / "run"
    driver.atomic_json(run / "freeze/execution_registration.json", reg)
    manifest_path = tmp_path / "manifest.json"
    driver.atomic_json(manifest_path, {"rows": [
        {"demo": "train", "split": "train", "T": 90, "admission": {"motion": True}},
        {"demo": "val", "split": "val", "T": 90, "admission": {"motion": True}}]})
    reg["training_manifest"] = detail._file_record(manifest_path, tmp_path)
    att = run / "formal/deterministic/attempt_001"; att.mkdir(parents=True)
    parts = {key: .1 for key in w1.S0_REQUIRED_LOSS_KEYS}
    losses = [{"step": step, "n_micro": 1, "n_samples": 2, **parts,
               "gnorm": .2, "lr": cfg["lr"], "ms": 1.0} for step in (1, 2)]
    val = [{"step": step, **parts, "n_windows_evaluated": 2,
            "n_val_windows_total": 2, "n_batches": 1, "complete": True} for step in (1, 2)]
    formal = [{"step": step, "seed": 7, "model_type": "deterministic",
               "n_expected": 1, "n_ok": 1, "complete": True, "macro_mpjpe_m": .2,
               "per_demo": [{"demo": "val", "T": 90, "status": "ok",
                             "numerical_valid": True, "mpjpe_m": .2}]} for step in (1, 2)]
    ck = []
    for name, step in (("best.pt", 1), ("last.pt", 2)):
        (att / name).write_bytes(b"checkpoint-deep-load-covered-by-main-fixture")
        ck.append({"name": name, "sha256": sha256_file(att / name), "step": step})
    summary = {"config": cfg, "identity": {"config": cfg}, "steps": 2,
               "stop_reason": None, "complete": True, "budget_stopped": False,
               "failure": None, "preservation_errors": [], "val_history": val,
               "formal_history": formal, "checkpoint_index": ck,
               "n_train_windows": 2, "n_val_windows": 2,
               "best_formal_step": 1, "selection_status": "best_by_formal_macro",
               "best_formal_macro_mpjpe_m": .2,
               "train_loader": {"n_windows": 2, "micro_batch": 2, "batches_per_epoch": 1,
                   "drop_last_discarded_per_epoch": 0, "epochs_started": 2,
                   "epochs_completed": 1, "micro_batches_total": 2, "samples_total": 4,
                   "backward_micro_batches_total": 2}}
    for name, value in (("loss_history.json", losses), ("val_history.json", val),
                        ("formal_history.json", formal), ("checkpoint_index.json", ck),
                        ("summary.json", summary)):
        driver.atomic_json(att / name, value)
    row = {"attempt": att.name, "status": "done", "complete": True,
           "summary": str(att / "summary.json"), "summary_sha256": sha256_file(att / "summary.json"),
           "artifacts": {detail._record_path(path, tmp_path): sha256_file(path)
                         for path in att.iterdir() if path.name != "summary.json"}}
    monkeypatch.setattr(driver, "stage_state", lambda *a: (row, []))
    calls = []
    def checked_loader(selection, *args, **kwargs):
        assert (selection["role"], selection["checkpoint_role"]) in (("best", "best"), ("last", "last_diagnostic"))
        calls.append(selection["role"])
        return object()
    monkeypatch.setattr(raw, "load_w1_de_generator", checked_loader)
    out = finalize._training_evidence(run, tmp_path, reg, "deterministic")
    assert out["complete"] and out["evidence_complete"], out["problems"]
    assert calls == ["best", "last"]
    summary["complete"] = False
    driver.atomic_json(att / "summary.json", summary)
    row["summary_sha256"] = sha256_file(att / "summary.json")
    assert not finalize._training_evidence(run, tmp_path, reg, "deterministic")["complete"]
    row["summary_sha256"] = "0" * 64
    out = finalize._training_evidence(run, tmp_path, reg, "deterministic")
    assert not out["complete"] and "training_summary_unregistered" in out["problems"]


def test_egl_replay_renderer_data_camera_and_last_frame(tmp_path):
    """Exercise the real locked MuJoCo renderer in the bounded production worker."""
    from PIL import Image

    cond, root, joints = _condition(T=5)
    root[:, 0] = np.arange(5) * .04
    human = cond.human_kps_world.copy()
    human[:, :, 0] = np.linspace(-.3, .3, 52)
    human[:, :, 1] = np.linspace(-.2, .2, 52)
    ref_path = tmp_path / "reference.npz"
    scene = np.concatenate([root, joints, cond.target_obj_pose], 1)
    detail._save_npz(ref_path, human_kpts_world=human, source_box=cond.source_obj_pose,
                     target_box=cond.target_obj_pose, teacher_scene=scene,
                     frame_dt=cond.frame_dt, source_box_similarity_scale=.75)
    product = tmp_path / "teacher.npz"; detail._save_npz(product, scene=scene)
    result = detail._run_unit("render", {
        "project_root": str(tmp_path), "demo": "demo", "case": "fail_case",
        "reference": detail._file_record(ref_path, tmp_path),
        "rows": [{"method": "teacher", "status": "ok", "scene_product": detail._file_record(product, tmp_path)},
                 {"method": "deterministic", "status": "numerical_rejected"},
                 {"method": "diffusion", "status": "numerical_rejected"}],
        "gif_path": str(tmp_path / "replay.gif")},
        time.perf_counter() + 45, tmp_path, "egl_smoke")
    assert result["frames"] == [0, 3, 4] and result["includes_last_frame"]
    assert result["duration_ms"] == [100, 33, 33]
    assert result["gif"]["sha256"] == sha256_file(tmp_path / "replay.gif")
    with Image.open(tmp_path / "replay.gif") as gif:
        assert gif.size == (1280, 265) and gif.n_frames == 3
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index); durations.append(gif.info["duration"])
        assert durations == [100, 30, 30]  # GIF encodes centiseconds
