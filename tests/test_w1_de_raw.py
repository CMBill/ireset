"""W1-D/E raw evidence boundaries, independent replay and honest denominators."""
from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_generation as raw
from ireset.data import a2_integrity as integ, w1_training as w1
from ireset.data.schema import InteractionClip, ReferenceCondition, RobotTrajectory
from ireset.embodied.contact_sites import ContactInterval, ContactSchedule
from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1
from ireset.diffusion.generator import DiffusionG1
from ireset.model.generator import GenerationNumericalError
from ireset.repair.config_io import load_mode_configs
from ireset.repair.validation import CheckResult, ValidationReport
from ireset.train.trainer import TrainConfig
from test_omniretarget_w1 import _patch_paths, _sha, build_synthetic_world


def _condition(T=65, contacts=()):
    pose = np.zeros((T, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    return ReferenceCondition("demo", np.zeros((T, 52, 3)), pose.copy(), pose.copy(),
                              contacts=list(contacts), frame_dt=1 / 30, world_from_clip=np.eye(4))


def _registration(demos):
    return {"execution_id": "w1_de_v1", "label_protocol": w1.MOTION_PROTOCOL_VERSION,
            "source_experiment_id": w1.EXPERIMENT_ID, "training_purpose": w1.TRAINING_PURPOSE,
            "assets_sha256": "a" * 64, "condition_code_sha256": "c" * 64,
            "training_manifest": {"path": "manifest.json", "sha256": "m" * 64},
            "cache": {"dir": "cache", "identity_sha256": "i" * 64, "root_stats_sha256": "r" * 64},
            "rosters": {"train24": [], "val31": list(demos), "val6": list(demos[:1])},
            "evaluation": {"methods": list(raw.W1_DE_METHODS), "main_seed": 7,
                           "appendix_seeds": [17, 27], "exclude": ["box_ground"],
                           "sampling": dict(raw.W1_DE_SAMPLING), "validation_mode": "strict",
                           "validation_config": dataclasses.asdict(load_mode_configs("strict")[1]),
                           "task_gate": {"frame": "world", "tol_m": 0.4},
                           "expected_matrix": {"main_keys": 3 * len(demos), "appendix_keys": 2 * len(demos)}}}


def _metadata(method, T=65, seed=7, invalid=False):
    from ireset.model.stitch import window_starts
    starts = [] if method == "teacher" else window_starts(T, 60, 30)
    calls = 0 if method == "teacher" else (1 if method == "deterministic" or not invalid else 3)
    indices = [0] if invalid else []
    return {"teacher_raw": method == "teacher", "starts": starts, "windows": len(starts),
            "last_frame_covered": True, "model_calls": calls,
            "resample_attempts_used": calls, "resample_attempts_max": 0 if method == "teacher" else 3,
            "attempt_seeds": [] if method == "teacher" else [seed + i for i in range(calls)],
            "per_attempt_invalid_windows": [len(indices)] * calls,
            "per_attempt_invalid_window_indices": [indices.copy() for _ in range(calls)],
            "final_invalid_windows": len(indices), "final_invalid_window_indices": indices,
            "numerical_valid": not invalid, "generation_wall_s": 0.0}


def _numerical_row(key, T=65):
    cond = _condition(T)
    refs = raw.w1_de_reference_facts(cond)
    return {**key, "T": T, "attempted": True, **raw._w1_de_unmeasured(refs, "numerical_rejected"),
            "status": "numerical_rejected", "numerical_valid": False,
            "numerical_facts": {"all_finite": False}, "numerical_evidence_verified": True,
            "measurements_status": "not_applicable_numerical_rejection",
            "retry_evidence_problems": [], "product_verified": True,
            "product": {"path": "product.npz", "sha256": "f" * 64}}


def test_matrix_is_155_keys_and_negative_numerical_results_can_complete():
    reg = _registration([f"demo_{i:02d}" for i in range(31)])
    matrix = raw.w1_de_expected_matrix(reg)
    assert len(matrix) == 155 and len({r["key"] for r in matrix}) == 155
    assert sum(r["matrix"] == "main" for r in matrix) == 93
    rows = [_numerical_row(k) for k in matrix]
    result = raw.w1_de_matrix_completeness(rows, reg)
    assert result["main_complete"] and result["appendix_complete"] and result["complete"]
    aggregate = raw.w1_de_aggregate_rows(rows, reg)
    for group in aggregate.values():
        assert group["counts"]["expected"] == 31
        assert group["counts"]["raw_standard_pass"] == 0
        assert group["counts"]["numerical_rejected"] == 31
        assert group["macros"]["teacher_fit_fk_mpjpe_box_m"]["macro_all"] is None


@pytest.mark.parametrize("status", ["label_row_missing", "label_not_admitted", "input_unavailable",
                                   "checkpoint_unavailable", "measurement_error", "not_run", "timeout", "unknown"])
def test_failure_rows_preserve_matrix_but_are_incomplete(status):
    reg = _registration(["demo"])
    rows = [_numerical_row(k) for k in raw.w1_de_expected_matrix(reg)]
    rows[1]["status"] = status
    result = raw.w1_de_matrix_completeness(rows, reg)
    assert result["got_rows"] == 5 and not result["complete"]
    assert rows[1]["key"] in result["evidence_gaps"]


def test_matrix_rejects_duplicates_foreign_identity_and_reference_count_deletion():
    reg = _registration(["demo"])
    rows = [_numerical_row(k) for k in raw.w1_de_expected_matrix(reg)]
    rows[0]["ref_grasp_frames"].pop("l")
    assert not raw.w1_de_matrix_completeness(rows, reg)["complete"]
    rows = [_numerical_row(k) for k in raw.w1_de_expected_matrix(reg)]
    rows[0]["both"]["denominator_frames"] = -0.5
    assert not raw.w1_de_matrix_completeness(rows, reg)["complete"]
    rows = [_numerical_row(k) for k in raw.w1_de_expected_matrix(reg)]
    rows.append(copy.deepcopy(rows[0]))
    assert raw.w1_de_matrix_completeness(rows, reg)["duplicate_keys"]
    rows[0]["method"] = "diffusion"
    assert not raw.w1_de_matrix_completeness(rows[:-1], reg)["complete"]


@pytest.mark.parametrize("malformed", [[], None, "foreign", {"key": []}, {"key": {}}, {"key": None}])
def test_extra_malformed_rows_cannot_be_silently_ignored(malformed):
    reg = _registration([f"demo_{i:02d}" for i in range(31)])
    rows = [_numerical_row(k) for k in raw.w1_de_expected_matrix(reg)]
    rows.append(malformed)
    result = raw.w1_de_matrix_completeness(rows, reg)
    assert result["got_rows"] == 156
    assert not result["main_complete"] and not result["appendix_complete"] and not result["complete"]
    assert result["evidence_gaps"]["row[155]"] in (["row_not_a_mapping"], ["row_key_invalid"])


@pytest.mark.parametrize("measurement_gaps", [None, "reason", [None], [{"reason": "unknown"}]])
def test_malformed_measurement_gaps_are_incomplete_without_crashing(measurement_gaps):
    reg = _registration(["demo"])
    row = _numerical_row(raw.w1_de_expected_matrix(reg)[0])
    row["measurement_gaps"] = measurement_gaps
    assert "measurement_gaps_invalid" in raw.w1_de_row_evidence_gaps(row)
    # Exercise the measured path too; missing metric families remain named gaps.
    row.update(status="ok", numerical_valid=True)
    assert "measurement_gaps_invalid" in raw.w1_de_row_evidence_gaps(row)


@pytest.mark.parametrize("undefined", [[], ["left_hand", "right_hand", "left_foot", "right_foot"], 1])
def test_contact_undefined_reasons_require_measured_mapping_evidence(undefined):
    row = _numerical_row(raw.w1_de_expected_matrix(_registration(["demo"]))[0])
    row.update(status="ok", numerical_valid=True,
               contact_f1={"left_hand": None, "right_hand": None, "left_foot": None, "right_foot": None,
                           "n_defined": 0, "macro_f1": None, "undefined_reasons": undefined})
    assert "contact_f1_undefined_reasons_invalid" in raw.w1_de_row_evidence_gaps(row)


def test_macros_do_not_publish_subset_or_duplicate_means_as_full_roster():
    reg = _registration(["demo_a", "demo_b"])
    rows = []
    for key in raw.w1_de_expected_matrix(reg):
        row = _numerical_row(key)
        if key["demo"] == "demo_a":
            row.update(status="ok", numerical_valid=True,
                       fitting={"teacher_fit_fk_mpjpe_box_m": 0.25})
        rows.append(row)
    group = raw.w1_de_aggregate_rows(rows, reg)["diffusion|7"]
    metric = group["macros"]["teacher_fit_fk_mpjpe_box_m"]
    assert metric["n_defined"] == 1 and metric["macro_all"] is None
    assert metric["defined_only_mean"] == 0.25
    dup = rows + [copy.deepcopy(rows[2])]
    assert raw.w1_de_aggregate_rows(dup, reg)["diffusion|7"]["macros"]["teacher_fit_fk_mpjpe_box_m"]["macro_all"] is None


def test_output_guard_allows_driver_preparation_only_and_refuses_source_history(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    monkeypatch.setattr(integ, "DERIVED_DIR", root / "derived")
    run = root / "derived/experiments/omniretarget_first/new"
    source = root / "derived/experiments/omniretarget_first/source"
    reg_path = run / "freeze/execution_registration.json"
    reg = {"source_run_root": {"path": str(source)}}
    att = run / "evaluation/raw/attempt_001"
    att.mkdir(parents=True)
    (att / "driver_call.json").write_text("{}")
    out, products = att / "raw_eval.json", att / "products"
    assert raw.guard_w1_de_outputs(out, products, reg_path, reg, root) == (out, products)
    (att / "old_product.npz").write_text("evidence")
    with pytest.raises(SystemExit, match="prior evidence"):
        raw.guard_w1_de_outputs(out, products, reg_path, reg, root)
    with pytest.raises(SystemExit):
        raw.guard_w1_de_outputs(source / "new.json", source / "products", reg_path, reg, root)
    with pytest.raises(SystemExit, match="resolve under"):
        raw.guard_w1_de_outputs(tmp_path / "source.json", products, reg_path, reg, root)


@pytest.mark.parametrize("flag", ["--out", "--output"])
def test_cli_alias_uses_output_dest_and_enters_raw_path(flag, tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(raw, "run_w1_de_raw_eval", lambda args: captured.append(args))
    monkeypatch.setattr(sys, "argv", ["eval_generation.py", "--w1-de-raw",
                                    "--execution-registration", "r.json", "--checkpoint-selection", "s.json",
                                    "--training-manifest", "m.json", "--cache-dir", "cache",
                                    flag, str(tmp_path / "report.json"), "--products-dir", "products"])
    raw.main()
    assert len(captured) == 1 and captured[0].output == str(tmp_path / "report.json")


def _product(tmp_path, method="diffusion", invalid=False, windows=False):
    cond = _condition()
    clip = InteractionClip(cond, np.zeros((65, 29)), np.tile([0., 0., 0., 1., 0., 0., 0.], (65, 1)))
    reg = _registration(["demo"])
    key = next(k for k in raw.w1_de_expected_matrix(reg) if k["method"] == method)
    row = {**key, "T": 65, "retry": raw._w1_de_retry(_metadata(method, invalid=invalid))}
    record = {"inputs": {"npz_sha256": "n" * 64, "pt_sha256": "p" * 64},
              "derived": {"path": "derived.npz", "sha256": "d" * 64}}
    cache = {"assets_sha256": "a" * 64, "schedule_rows": [{"demo": "demo", "key": "k", "sha256": "s" * 64}]}
    checkpoint = None if method == "teacher" else {"path": "model.pt", "sha256": "h" * 64,
                                                    "step": 1000, "role": "best", "formal_attempt": "attempt_001"}
    ident = raw.w1_de_product_identity(row, reg, record, cond, checkpoint, "e" * 64, cache)
    metadata = _metadata(method, invalid=invalid)
    traj = RobotTrajectory(clip.root.copy(), clip.q.copy(), cond.frame_dt, metadata)
    if invalid:
        traj.root[0, 0] = np.nan
    evidence = None
    if windows:
        evidence = np.zeros((2, 60, 38), dtype=np.float32)
        evidence[..., 3] = 1
        evidence[..., 7] = 1
        evidence[0, 0, 0] = np.nan
        traj = None
    path = tmp_path / "product.npz"
    row["product"] = raw.save_w1_de_product(path, row, ident, cond, trajectory=traj,
                                            output=evidence, metadata=metadata, project_root=tmp_path)
    return reg, row, clip, path, ident


def test_product_verification_binds_target_bytes_and_rejects_tampering(tmp_path):
    reg, row, clip, path, ident = _product(tmp_path)
    verified = raw.verify_w1_de_product(path, row, reg, clip.condition, project_root=tmp_path,
                                        expected_identity=ident["checkpoint"])
    assert verified["passed"], verified["problems"]
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["target_box"][0, 0] = 1.0
    np.savez(path, **arrays)
    assert "product_sha256_mismatch" in raw.verify_w1_de_product(path, row, reg, clip.condition,
                                                                 project_root=tmp_path)["problems"]
    row["product"]["sha256"] = _sha(path)  # honest rehash cannot fix foreign task input
    assert "product_condition_binding" in raw.verify_w1_de_product(path, row, reg, clip.condition,
                                                                    project_root=tmp_path)["problems"]


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("attempted", [None, False])
def test_invalid_arrays_saved_and_replayed_without_geometry(tmp_path, monkeypatch, windows, attempted):
    reg, row, clip, path, ident = _product(tmp_path, invalid=True, windows=windows)
    if attempted is not None:
        row["attempted"] = attempted  # A row claim cannot suppress a verified attempt.
    monkeypatch.setattr(raw, "w1_de_frame_series", lambda *a, **k: pytest.fail("unsafe geometry"))
    result = raw.recompute_w1_de_product(path, row, reg, clip, object(), object(),
                                          project_root=tmp_path, expected_identity=ident)
    assert result["verification"]["passed"], result["verification"]["problems"]
    measured = result["row"]
    assert measured["status"] == "numerical_rejected"
    assert measured["numerical_valid"] is False and measured["numerical_evidence_verified"] is True
    assert measured["attempted"] is True
    assert raw.w1_de_aggregate_rows([measured], reg)["diffusion|7"]["counts"]["attempted"] == 1
    assert result["report"] is None and result["series"] is None
    assert raw.w1_de_row_evidence_gaps(measured) == []
    assert raw.compare_w1_de_row_metrics(measured, copy.deepcopy(measured)) == []


def test_numerical_rejection_requires_final_invalid_indices_to_match_actual_windows(tmp_path, monkeypatch):
    reg, row, clip, path, ident = _product(tmp_path, invalid=True, windows=True)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    # Keep all metadata/row counts internally consistent but name the wrong window.
    metadata["per_attempt_invalid_window_indices"] = [[1], [1], [1]]
    metadata["final_invalid_window_indices"] = [1]
    row["retry"] = raw._w1_de_retry(metadata)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    arrays["retry_json"] = np.asarray(json.dumps(row["retry"]))
    np.savez(path, **arrays)
    row["product"]["sha256"] = _sha(path)
    verified = raw.verify_w1_de_product(path, row, reg, clip.condition, project_root=tmp_path)
    assert verified["passed"], verified["problems"]
    negative, facts = raw._w1_de_window_rejection(verified["arrays"], metadata, clip.T)
    assert facts["final_invalid_indices_match_output"] is False and negative is False
    monkeypatch.setattr(raw, "w1_de_frame_series", lambda *a, **k: pytest.fail("unsafe geometry"))
    result = raw.recompute_w1_de_product(path, row, reg, clip, object(), object(), project_root=tmp_path)
    assert result["row"]["status"] == "generation_error"
    assert result["row"]["numerical_evidence_verified"] is False
    assert raw.w1_de_row_evidence_gaps(result["row"])
    # Direct forged numerical claims are rejected by matrix completeness as well.
    forged = _numerical_row(row)
    forged["numerical_facts"] = facts
    assert "numerical_rejection_not_corroborated" in raw.w1_de_row_evidence_gaps(forged)


def test_product_failure_cannot_prove_an_attempt_from_row_claim(tmp_path):
    reg, row, clip, path, ident = _product(tmp_path, invalid=True)
    row["attempted"] = True
    row["product"]["sha256"] = "0" * 64
    result = raw.recompute_w1_de_product(path, row, reg, clip, object(), object(), project_root=tmp_path)
    assert result["row"]["status"] == "product_invalid"
    assert result["row"].get("attempted") is not True
    assert raw.w1_de_aggregate_rows([result["row"]], reg)["diffusion|7"]["counts"]["attempted"] == 0


def test_pre_gate_does_not_read_stale_or_absent_validator_cache_and_keeps_refs():
    T = 8
    cond = _condition(T, [ContactInterval("left_hand", 1, 6, "object"),
                           ContactInterval("right_hand", 2, 7, "object")])
    rep = ValidationReport(False, checks={"finite": CheckResult(False, np.inf, 0.)},
                           status="rejected_nonfinite")
    validator = SimpleNamespace(cfg=load_mode_configs("strict")[1], validate=lambda *a, **k: rep)
    scene = np.zeros((T, 43))
    result = raw.w1_de_row_metrics("demo", scene, cond, None, None, None, None, validator, 0.4)
    assert result["ref_grasp_frames"] == {"l": 5, "r": 5}
    assert result["both"]["denominator_frames"] == 4
    assert result["both"]["valid_coverage"] is None
    assert result["measurements_status"] == "validator_pre_gate"
    assert result["robot_strict"]["checks"]["finite"]["worst"] is None


def test_both_certified_and_near_runs_do_not_bridge_gaps_in_reference():
    T = 10
    intervals = [ContactInterval(e, start, end, "object") for e in ("left_hand", "right_hand")
                 for start, end in ((1, 3), (6, 8))]
    cond = _condition(T, intervals)
    robot_schedule = ContactSchedule([ContactInterval(e, 0, T, "object", anchor=np.zeros((3, 3)))
                                      for e in ("left_hand", "right_hand")])
    checks = {name: CheckResult(True, 0., 1.) for name in raw.W1_DE_ROBOT_CHECKS}
    rep = ValidationReport(True, checks, robot_schedule)
    cache = {"sole_pts": {s: np.zeros((T, 5, 3)) for s in ("l", "r")},
             "sole_gap": {s: np.zeros(T) for s in ("l", "r")},
             "hand_local": {s: np.zeros((T, 3, 3)) for s in ("l", "r")}}
    validator = SimpleNamespace(cfg=load_mode_configs("strict")[1], _cache=cache)
    scene = np.zeros((T, 43)); scene[:, 3] = scene[:, 39] = 1
    series = {"robot_kpts": np.zeros((T, 9, 3)), "kpt_err_w": np.zeros(T),
              "kpt_err_b": np.zeros(T), "pen_key": np.zeros(T), "box_ground": np.zeros(T),
              "gap": {s: np.zeros(T) for s in ("l", "r")}}
    result = raw.w1_de_row_metrics("demo", scene, cond, scene, series["robot_kpts"],
                                   series, None, validator, 0.4, report=rep)
    assert result["both"]["denominator_frames"] == 4
    assert result["both"]["certified_overlap_frames"] == 4
    assert result["both"]["longest_certified_overlap"] == 2
    assert result["both"]["longest_consecutive_valid"] == 2
    assert result["both"]["certified_overlap_frame_indices"] == [1, 2, 6, 7]
    assert set(result["robot_strict"]["checks"]) == set(raw.W1_DE_ROBOT_CHECKS)
    series["kpt_err_w"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        raw.w1_de_row_metrics("demo", scene, cond, scene, series["robot_kpts"],
                              series, None, validator, 0.4, report=rep)


def test_metric_comparison_rejects_nan_and_pass_flag_tampering():
    baseline = {"status": "ok", "fitting": {"teacher_fit_fk_mpjpe_box_m": 0.125},
                "raw_standard_pass": False}
    changed = copy.deepcopy(baseline)
    changed["fitting"]["teacher_fit_fk_mpjpe_box_m"] = float("nan")
    assert any("nonfinite" in s for s in raw.compare_w1_de_row_metrics(changed, baseline))
    changed = copy.deepcopy(baseline); changed["raw_standard_pass"] = True
    assert any("not_reproducible" in s for s in raw.compare_w1_de_row_metrics(changed, baseline))


def _checkpoint_world(tmp_path, monkeypatch, model="diffusion"):
    world = build_synthetic_world(tmp_path, n_train=1, n_val=1, T=65)
    _patch_paths(monkeypatch, world)
    cfg = dataclasses.asdict(TrainConfig(model_type=model, d_model=24, n_layers=1, n_heads=2,
                                       ffn_mult=2, time_window=3, device="cpu", num_train_timesteps=100,
                                       beta_start=0.0002, beta_end=0.03))
    reg = _registration(world["val31"])
    reg.update(condition_code_sha256=w1.condition_code_sha256(world["root"]),
               assets_sha256=json.loads(world["prereg_path"].read_text())["assets"]["a1_assets_sha256"],
               training_manifest={"path": str(world["manifest_path"]), "sha256": _sha(world["manifest_path"])},
               preregistration={"path": str(world["prereg_path"]), "sha256": _sha(world["prereg_path"])},
               cache={"dir": str(world["cache_dir"]),
                      "identity_sha256": _sha(world["cache_dir"] / "cache_identity.json"),
                      "root_stats_sha256": _sha(world["cache_dir"] / "root_stats.npz")},
               configs={model: cfg, "config_file_sha256": "f" * 64})
    bw = {k: cfg[k] for k in ("d_model", "n_layers", "n_heads", "ffn_mult", "time_window")}
    module = (DiffusionG1(num_train_timesteps=100, beta_start=0.0002, beta_end=0.03, **bw)
              if model == "diffusion" else DeterministicG1(**bw))
    with np.load(world["cache_dir"] / "root_stats.npz", allow_pickle=False) as z:
        stats = {k: z[k].copy() for k in ("mean", "std")}
    identity = {"experiment_id": w1.EXPERIMENT_ID, "model_type": model,
                "protocol": w1.MOTION_PROTOCOL_VERSION, "training_purpose": w1.TRAINING_PURPOSE,
                "training_manifest": reg["training_manifest"], "cache": reg["cache"],
                "assets_sha256": reg["assets_sha256"], "condition_code_sha256": reg["condition_code_sha256"],
                "config": cfg, "config_file_sha256": "f" * 64,
                "execution_registration": {"path": "reg.json", "sha256": "e" * 64, "execution_id": "w1_de_v1"}}
    checkpoint = {"model": module.state_dict(), "conditioner": ConditionEncoder(24).state_dict(),
                  "config": cfg, "identity": identity, "root_stats": stats, "step": 1000}
    path = tmp_path / "model.pt"; torch.save(checkpoint, path)
    selection = {"path": str(path), "sha256": _sha(path), "step": 1000,
                 "model": model, "role": "last", "checkpoint_role": "last_diagnostic"}
    return world, reg, checkpoint, selection


def _load_checkpoint(world, reg, selection):
    return raw.load_w1_de_generator(selection, "cpu", 25, world["cache_dir"],
                                    reg["training_manifest"]["sha256"], reg["cache"]["identity_sha256"],
                                    reg["cache"]["root_stats_sha256"], "e" * 64,
                                    registration=reg, project_root=world["root"])


def test_checkpoint_full_config_beta_role_and_strict_state_are_rebuilt(tmp_path, monkeypatch):
    world, reg, checkpoint, selection = _checkpoint_world(tmp_path, monkeypatch)
    gen = _load_checkpoint(world, reg, selection)
    assert gen.model.ddim.config.beta_start == 0.0002
    assert gen.model.ddim.config.beta_end == 0.03
    assert gen.model.ddim.config.num_train_timesteps == 100
    assert gen.w1_de_checkpoint["role"] == "last_diagnostic"
    assert gen.w1_de_checkpoint["sha256"] == selection["sha256"]
    selection["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="full sha256"):
        _load_checkpoint(world, reg, selection)


@pytest.mark.parametrize("damage", ["config", "protocol", "purpose", "assets", "condition_code", "state_empty", "state_nan", "stats_dtype", "stats_value"])
def test_checkpoint_foreign_or_corrupt_evidence_is_refused(tmp_path, monkeypatch, damage):
    world, reg, checkpoint, selection = _checkpoint_world(tmp_path, monkeypatch)
    if damage == "config":
        checkpoint["config"] = {**checkpoint["config"], "beta_end": 0.04}
    elif damage in ("protocol", "purpose", "assets", "condition_code"):
        key = {"purpose": "training_purpose", "assets": "assets_sha256", "condition_code": "condition_code_sha256"}.get(damage, damage)
        checkpoint["identity"][key] = "foreign"
    elif damage == "state_empty":
        checkpoint["model"] = {}
    elif damage == "state_nan":
        checkpoint["model"][next(iter(checkpoint["model"]))].flatten()[0] = float("nan")
    elif damage == "stats_dtype":
        checkpoint["root_stats"]["mean"] = checkpoint["root_stats"]["mean"].astype(np.float32)
    else:
        checkpoint["root_stats"]["mean"][0] += 0.125
    torch.save(checkpoint, selection["path"])
    selection["sha256"] = _sha(Path(selection["path"]))
    with pytest.raises((ValueError, RuntimeError)):
        _load_checkpoint(world, reg, selection)


def _raw_world(tmp_path, monkeypatch, n_val=1, T=65):
    world = build_synthetic_world(tmp_path, n_train=1, n_val=n_val, T=T)
    _patch_paths(monkeypatch, world)
    reg = _registration(world["val31"])
    manifest = json.loads(world["manifest_path"].read_text())
    prereg = json.loads(world["prereg_path"].read_text())
    reg.update(assets_sha256=prereg["assets"]["a1_assets_sha256"],
               condition_code_sha256=w1.condition_code_sha256(world["root"]),
               training_manifest={"path": str(world["manifest_path"]), "sha256": _sha(world["manifest_path"])},
               preregistration=manifest["preregistration"],
               cache={"dir": str(world["cache_dir"]), "identity_sha256": _sha(world["cache_dir"] / "cache_identity.json"),
                      "root_stats_sha256": _sha(world["cache_dir"] / "root_stats.npz")},
               source_files={str(world["prereg_path"]): _sha(world["prereg_path"])},
               source_run_root={"path": str(world["run"])},
               configs={model: {"device": "cpu"} for model in ("deterministic", "diffusion")})
    reg["rosters"]["train24"] = world["train24"]
    run = world["root"] / "derived/experiments/omniretarget_first/de_run"
    reg_path = run / "freeze/execution_registration.json"
    reg_path.parent.mkdir(parents=True)
    reg_path.write_text(json.dumps(reg))
    sel_path = run / "evaluation/checkpoint_selection.json"
    sel_path.parent.mkdir(parents=True)
    sel_path.write_text(json.dumps({"execution_registration_sha256": _sha(reg_path),
                                    "models": {m: {"role": "unavailable", "reason": "fixture"}
                                               for m in ("deterministic", "diffusion")}}))
    args = SimpleNamespace(execution_registration=reg_path, checkpoint_selection=sel_path,
                           training_manifest=world["manifest_path"], cache_dir=world["cache_dir"],
                           output=run / "evaluation/raw/attempt_001/raw_eval.json",
                           products_dir=run / "evaluation/raw/attempt_001/products", ddim_steps=25,
                           budget_deadline_s=30.0)
    return world, reg, args


def test_raw_missing_checkpoints_retains_all_rows_and_teacher_product(tmp_path, monkeypatch):
    world, reg, args = _raw_world(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as err:
        raw.run_w1_de_raw_eval(args)
    assert err.value.code == 2
    report = json.loads(Path(args.output).read_text())
    assert len(report["rows"]) == 5
    assert report["admission"]["passed"], report["admission"]
    assert report["completeness"]["complete"] is False
    teacher = next(r for r in report["rows"] if r["method"] == "teacher")
    assert teacher["status"] == "ok", teacher
    assert teacher["product_verified"] and teacher["self_reference"]
    assert not raw.w1_de_row_evidence_gaps(teacher)
    assert sum(r["status"] == "checkpoint_unavailable" for r in report["rows"]) == 4
    assert report["per_method"]["teacher|None"]["counts"]["expected"] == 1


def _enable_failed_numerics(args, monkeypatch, calls=None):
    selection = json.loads(Path(args.checkpoint_selection).read_text())
    selection["execution_registration_sha256"] = _sha(Path(args.execution_registration))
    selection["models"] = {m: {"role": "best"} for m in ("deterministic", "diffusion")}
    Path(args.checkpoint_selection).write_text(json.dumps(selection))

    class FailedNumerics:
        w1_de_checkpoint = {"path": "fixture.pt", "sha256": "b" * 64, "step": 1, "role": "best"}
        def __init__(self, method, device):
            self.method, self.device = method, device
        def sample(self, cond, seed):
            if calls is not None:
                calls.append(("sample", self.method, self.device))
            md = _metadata(self.method, cond.T, seed, invalid=True)
            output = np.zeros((len(md["starts"]), 60, 38), dtype=np.float32)
            output[..., 3] = 1.; output[..., 7] = 1.
            output[0, 0, 0] = np.nan
            raise GenerationNumericalError("invalid_windows_after_attempts", md, output)

    def load(sel, device, *a, **kw):
        if calls is not None:
            calls.append(("load", sel["model"], device))
        return FailedNumerics(sel["model"], device)
    monkeypatch.setattr(raw, "load_w1_de_generator", load)


def test_raw_generation_error_saves_invalid_windows_before_geometry(tmp_path, monkeypatch):
    world, reg, args = _raw_world(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _enable_failed_numerics(args, monkeypatch)
    # Numerical model rows must skip geometry; Teacher raw still runs real geometry.
    original = raw.w1_de_frame_series
    def finite_geometry(replayer, scene, *a, **kw):
        assert np.isfinite(scene).all()
        return original(replayer, scene, *a, **kw)
    monkeypatch.setattr(raw, "w1_de_frame_series", finite_geometry)
    raw.run_w1_de_raw_eval(args)
    report = json.loads(Path(args.output).read_text())
    assert report["completeness"]["complete"], report["completeness"]
    for row in report["rows"]:
        assert row["product_verified"]
        if row["method"] != "teacher":
            assert row["status"] == "numerical_rejected"
            assert row["numerical_evidence_verified"] and not row["raw_standard_pass"]
            with np.load(world["root"] / row["product"]["path"], allow_pickle=False) as z:
                assert np.isnan(z["normalized_windows"][0, 0, 0])


def test_deadline_before_work_preserves_every_not_run_key(tmp_path, monkeypatch):
    world, reg, args = _raw_world(tmp_path, monkeypatch)
    args.budget_deadline_s = 0.0
    with pytest.raises(SystemExit) as err:
        raw.run_w1_de_raw_eval(args)
    assert err.value.code == 2
    report = json.loads(Path(args.output).read_text())
    assert report["budget_stopped"] and not report["completeness"]["complete"]
    assert len(report["rows"]) == 5 and all(r["status"] == "not_run" for r in report["rows"])
    assert not Path(args.products_dir).exists()


def test_measurement_failure_keeps_successful_raw_arrays(tmp_path, monkeypatch):
    world, reg, args = _raw_world(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(raw, "w1_de_frame_series", lambda *a, **k: (_ for _ in ()).throw(ValueError("measurement broke")))
    with pytest.raises(SystemExit):
        raw.run_w1_de_raw_eval(args)
    report = json.loads(Path(args.output).read_text())
    teacher = next(r for r in report["rows"] if r["method"] == "teacher")
    assert teacher["status"] == "measurement_error"
    product = world["root"] / teacher["product"]["path"]
    assert product.is_file() and _sha(product) == teacher["product"]["sha256"]
    with np.load(product, allow_pickle=False) as z:
        assert z["root"].shape == (65, 7) and z["joints"].shape == (65, 29)
    assert not report["completeness"]["complete"]


@pytest.mark.parametrize("device,cuda_available", [("cpu", True), ("cuda", False), ("cuda", True)])
def test_raw_uses_registered_model_device_without_hardware_fallback(tmp_path, monkeypatch, device, cuda_available):
    world, reg, args = _raw_world(tmp_path, monkeypatch)
    for cfg in reg["configs"].values():
        cfg["device"] = device
    Path(args.execution_registration).write_text(json.dumps(reg))
    calls, synced = [], []
    _enable_failed_numerics(args, monkeypatch, calls)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(raw, "_w1_de_sync", lambda dev: synced.append(dev))
    if device == "cuda" and not cuda_available:
        with pytest.raises(SystemExit) as err:
            raw.run_w1_de_raw_eval(args)
        assert err.value.code == 2
    else:
        raw.run_w1_de_raw_eval(args)
    report = json.loads(Path(args.output).read_text())
    assert report["cost"]["model_devices"] == {m: device for m in ("deterministic", "diffusion")}
    teacher = next(r for r in report["rows"] if r["method"] == "teacher")
    assert teacher["status"] == "ok" and teacher["cost"]["device"] == "cpu"
    models = [r for r in report["rows"] if r["method"] != "teacher"]
    if device == "cuda" and not cuda_available:
        assert calls == [] and synced == ["cpu", "cpu"]
        assert all(r["status"] == "checkpoint_unavailable" and not r["attempted"] for r in models)
        assert all("registered_device_unavailable:cuda" in r["error"] for r in models)
        assert not report["completeness"]["complete"]
    else:
        assert sum(c[0] == "load" for c in calls) == 2
        assert sum(c[0] == "sample" for c in calls) == 4
        assert all(c[2] == device for c in calls)
        assert all(r["cost"]["device"] == device for r in models)
        assert report["completeness"]["complete"]
        if device == "cpu":
            assert set(synced) == {"cpu"} and not report["cost"]["gpu_synchronized"]
        else:
            assert "cuda" in synced and report["cost"]["gpu_synchronized"]


def test_full_measured_quality_rejection_can_complete_at_zero_of_31():
    reg = _registration([f"demo_{i:02d}" for i in range(31)])
    rows = []
    for key in raw.w1_de_expected_matrix(reg):
        row = _numerical_row(key)
        checks = {name: {"passed": name != "penetration", "worst": 0.01 if name == "penetration" else 0.,
                         "limit": 0.002 if name == "penetration" else 1.}
                  for name in raw.W1_DE_ROBOT_CHECKS}
        row.update(status="ok", numerical_valid=True, numerical_facts={"all_finite": True},
                   measurements_status="measured", measurement_gaps=[],
                   fitting={"root_rmse_m": 0., "root_geodesic_deg": 0., "joint_mae_rad": 0.,
                            "teacher_fit_fk_mpjpe_box_m": 0.1},
                   task={"world_mpjpe_mean_m": 0.6, "world_mpjpe_p95_m": 0.6, "world_mpjpe_max_m": 0.6,
                         "box_mpjpe_mean_m": 0.6, "box_mpjpe_p95_m": 0.6, "box_mpjpe_max_m": 0.6,
                         "tol_m": 0.4, "gate_pass": False},
                   penetration={"max_m": 0.01, "keyframe_violation_ratio": 0.5},
                   box_ground={"penetration_max_m": 0., "violation_ratio": 0.},
                   robot_strict={"passed": False, "failed_checks": ["penetration"], "checks": checks},
                   robot_strict_pass=False, task_gate_pass=False, raw_standard_pass=False,
                   contact_f1={"left_hand": None, "right_hand": None, "left_foot": None, "right_foot": None,
                               "n_defined": 0, "macro_f1": None,
                               "undefined_reasons": {e: "neither_reference_nor_robot_contact" for e in
                                                     ("left_hand", "right_hand", "left_foot", "right_foot")}})
        assert raw.w1_de_row_evidence_gaps(row) == []
        rows.append(row)
    assert raw.w1_de_matrix_completeness(rows, reg)["complete"]
    for group in raw.w1_de_aggregate_rows(rows, reg).values():
        assert group["counts"]["expected"] == 31 and group["counts"]["raw_standard_pass"] == 0
        assert group["counts"]["strict_rejected"] == group["counts"]["task_rejected"] == 31
        assert group["macros"]["teacher_fit_fk_mpjpe_box_m"]["macro_all"] == pytest.approx(0.1)
    rows[-1]["task"]["world_mpjpe_max_m"] = None
    assert not raw.w1_de_matrix_completeness(rows, reg)["complete"]


def test_deadline_after_product_save_keeps_arrays_and_stops_new_methods(tmp_path, monkeypatch):
    world, reg, args = _raw_world(tmp_path, monkeypatch)
    clock = [100.]
    monkeypatch.setattr(raw.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_save = raw.save_w1_de_product
    def slow_save(*a, **kw):
        product = original_save(*a, **kw)
        clock[0] = 131.  # args budget=30; cross deadline only after bytes are saved.
        return product
    monkeypatch.setattr(raw, "save_w1_de_product", slow_save)
    monkeypatch.setattr(raw, "recompute_w1_de_product", lambda *a, **kw: pytest.fail("checks started after deadline"))
    with pytest.raises(SystemExit) as err:
        raw.run_w1_de_raw_eval(args)
    assert err.value.code == 2
    report = json.loads(Path(args.output).read_text())
    assert report["budget_stopped"] and not report["completeness"]["complete"]
    teacher = next(r for r in report["rows"] if r["method"] == "teacher")
    assert teacher["status"] == "timeout" and teacher["attempted"]
    assert _sha(world["root"] / teacher["product"]["path"]) == teacher["product"]["sha256"]
    assert len(list(Path(args.products_dir).glob("*.npz"))) == 1
    assert all(r["status"] == "not_run" and not r["attempted"]
               for r in report["rows"] if r["method"] != "teacher")


def test_final_report_registration_deadline_revokes_complete_negative_in_place(tmp_path, monkeypatch):
    world, reg, args = _raw_world(tmp_path, monkeypatch)
    _enable_failed_numerics(args, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    clock, complete_writes = [100.], []
    monkeypatch.setattr(raw.time, "perf_counter", lambda: clock[0])
    original_write = raw._w1_de_write_report
    def slow_final_registration(path, report, **kw):
        original_write(path, report, **kw)
        if report["completeness"]["complete"]:
            complete_writes.append(path)
            # Last row registration is first; final whole-report registration is second.
            if len(complete_writes) == 2:
                clock[0] = 131.
    monkeypatch.setattr(raw, "_w1_de_write_report", slow_final_registration)
    with pytest.raises(SystemExit) as err:
        raw.run_w1_de_raw_eval(args)
    assert err.value.code == 2 and len(complete_writes) == 2
    report = json.loads(Path(args.output).read_text())
    assert report["stop_reason"] == "budget_deadline_after_report_registration"
    assert report["budget_stopped"] and not report["completeness"]["complete"]
    assert report["cost"]["total_s"] == 31.
    assert len(list(Path(args.products_dir).glob("*.npz"))) == 5
    assert all(r["product_verified"] for r in report["rows"])
