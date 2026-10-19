"""Attribution controls: counterfactual isolation and honest evidence scope."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_w1_attribution as diag
from ireset.model.stitch import fuse_windows


def _angles(quats):
    return Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)[:, 2]


def _constant_windows(starts, angles):
    n = len(starts)
    quats = np.stack([np.tile(Rotation.from_euler("z", a, degrees=True).as_quat()[[3, 0, 1, 2]], (60, 1)) for a in angles])
    return np.zeros((n, 60, 3)), quats, np.zeros((n, 60, 29))


def test_triple_overlap_reproduces_middle_window_loss_and_localizes_change():
    starts, T = [0, 30, 48], 108
    decoded = _constant_windows(starts, [0, 80, 160])
    original = fuse_windows(*[list(v) for v in decoded], starts, T, 30)
    alternative = diag.fuse_all_contributors(decoded, starts, T)
    before, after = _angles(original[1]), _angles(alternative[1])
    assert abs(before[48] - before[47]) > 40
    assert abs(after[48] - after[47]) < 10
    facts = diag.overlap_facts(starts, T)
    assert facts["coverage"][48] == 3
    assert facts["legacy_dropped_rotation_weight"][48] > .5
    assert np.array_equal(original[0], alternative[0])
    assert np.array_equal(original[2], alternative[2])
    assert np.allclose(before[facts["coverage"] <= 2], after[facts["coverage"] <= 2], atol=1e-8)


@pytest.mark.parametrize("starts,T", [([0], 60), ([0, 13], 73), ([0, 30], 90), ([0, 30, 60], 120)])
def test_one_or_two_contributors_match_original_slerp(starts, T):
    decoded = _constant_windows(starts, np.linspace(10, 150, len(starts)))
    before = fuse_windows(*[list(v) for v in decoded], starts, T, 30)
    after = diag.fuse_all_contributors(decoded, starts, T)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[2], after[2])
    assert np.allclose(_angles(before[1]), _angles(after[1]), atol=1e-8)


def test_identical_truth_windows_preserved_in_three_window_region():
    T, starts = 108, [0, 30, 48]
    angles = np.linspace(-100, 100, T)
    truth = Rotation.from_euler("z", angles[:, None], degrees=True).as_quat()[:, [3, 0, 1, 2]]
    decoded = list(_constant_windows(starts, [0, 0, 0]))
    decoded[1] = np.stack([truth[s:s + 60] for s in starts])
    after = diag.fuse_all_contributors(decoded, starts, T)
    assert np.allclose(_angles(after[1]), angles, atol=1e-8)


def test_train_selection_is_identity_based_and_subject_balanced():
    roster = [f"sub{s}_largebox_{n:03d}" for s in [10, 12, 16, 3, 7, 8] for n in [4, 2, 3, 1]]
    selected = diag.select_train6(list(reversed(roster)))
    assert selected == [f"sub{s}_largebox_001" for s in [10, 12, 16, 3, 7, 8]]
    with pytest.raises(ValueError):
        diag.select_train6(roster[:-1] + [roster[0]])


def test_hash_verification_detects_source_drift(tmp_path):
    path = tmp_path / "anchor"
    path.write_text("frozen")
    anchors = {str(path): diag.sha256_file(path)}
    assert not diag.hash_problems(anchors)
    path.write_text("changed")
    assert "drifted" in diag.hash_problems(anchors)[0]


def test_output_guard_refuses_source_and_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "ROOT", tmp_path)
    derived = tmp_path / "derived"
    derived.mkdir()
    source = derived / "source"
    source.mkdir()
    with pytest.raises(ValueError):
        diag.guard_run(source / "diagnosis", source)
    outside = tmp_path / "external"
    outside.mkdir()
    (derived / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        diag.guard_run(derived / "link/new", source)


def _matrix():
    variants = ("teacher", "teacher_roundtrip", "deterministic", "diffusion",
                "deterministic_all_contributors", "diffusion_all_contributors")
    return [{"demo": "demo", "group": "train6", "status": "ok", "rows": [
        {"variant": variant, "metrics": {"fitting": {"teacher_fit_fk_mpjpe_box_m": .1},
                                         "penetration": {"max_m": .02}, "raw_standard_pass": False}}
        for variant in variants]}]


def test_evidence_empty_or_duplicate_matrix_never_complete():
    assert not diag.aggregate([], ["demo"])["complete"]
    rows = _matrix()
    assert not diag.aggregate(rows + rows, ["demo"])["complete"]
    rows[0]["rows"].pop()
    assert not diag.aggregate(rows, ["demo"])["complete"]


def test_complete_negative_result_keeps_all_groups_and_no_automatic_label_claim():
    rows = []
    for group, n in (("train6", 6), ("val6", 6), ("known_case", 1)):
        for i in range(n):
            row = copy.deepcopy(_matrix()[0])
            row.update(demo=f"{group}_{i}", group=group, source_anchors_unchanged=True,
                       models={m: {"condition_check": {"passed": True}} for m in ("deterministic", "diffusion")})
            rows.append(row)
    summary = diag.aggregate(rows, [r["demo"] for r in rows])
    assert summary["complete"] and summary["paired_distinct_train_examples"] == 6
    assert not summary["label_principal_bottleneck_supported"]
    assert summary["groups"]["train6"]["deterministic"]["raw_standard_pass"] == 0
    assert summary["auto_execute"] is False


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True])
def test_nonfinite_or_missing_metrics_cannot_complete(bad):
    rows = _matrix()
    rows[0]["rows"][0]["metrics"]["penetration"]["max_m"] = bad
    assert not diag.aggregate(rows, ["demo"])["complete"]


def test_condition_comparison_detects_single_channel_tamper():
    arrays = {k: np.zeros((60, 1), dtype=np.float64) for k in diag.CONDITION_KEYS}
    class Dataset:
        index = [(0, 0)]
        clips = [SimpleNamespace(demo="demo")]
        def __getitem__(self, index):
            return {k: torch.from_numpy(v.astype(np.float32).copy()) for k, v in arrays.items()}
    class BadDataset(Dataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item["box_t_clip"][0] += .1
            return item
    generator = SimpleNamespace(_condition_batch=lambda condition: arrays)
    cond = SimpleNamespace(T=60)
    assert diag.condition_comparison(Dataset(), "demo", generator, cond)["passed"]
    assert not diag.condition_comparison(BadDataset(), "demo", generator, cond)["passed"]


def test_paired_families_keep_model_only_failure_separate():
    teacher = {"penetration_families": np.array([[.01, 0, 0], [0, 0, 0]]),
               "worst_geom_pairs": np.zeros((2, 3, 2), dtype=int)}
    model = copy.deepcopy(teacher)
    model["penetration_families"][1, 1] = .1
    result = diag.paired_evidence(teacher, model)
    assert result["robot_ground"]["shared_frames"] == 1
    assert result["robot_box"]["new_frames"] == 1
    assert result["robot_box"]["shared_frames"] == 0


def test_scene_measurement_runs_real_validator_and_curve_contract(tmp_path):
    """Exercise the actual shared metric API, not a mock of its signature."""
    import eval_generation as raw
    from ireset.data.schema import InteractionClip, ReferenceCondition
    from ireset.embodied.g1_spec import G1Spec
    from ireset.eval.teacher_comparison import TeacherQualityReplayer
    from ireset.kinematics.mujoco_adapter import G1Kinematics
    from ireset.repair.validation import IndependentValidator, ValidationConfig

    T = 60
    root = np.tile([0., 0., .8, 1., 0., 0., 0.], (T, 1))
    q = np.tile(G1Spec().joint_range.mean(1), (T, 1))
    box = np.tile([.6, 0., .8, 1., 0., 0., 0.], (T, 1))
    cond = ReferenceCondition("synthetic", np.zeros((T, 52, 3)), box.copy(), box,
                              world_from_clip=np.eye(4))
    clip = InteractionClip(cond, q, root)
    validator = IndependentValidator(cfg=ValidationConfig())
    replayer = TeacherQualityReplayer(validator)
    teacher_kpts = raw.fk_keypoints(G1Kinematics(), np.concatenate([root, q, box], 1))
    rec, curves = diag.save_scene(tmp_path, "teacher", root, q, [], clip,
                                 {"evaluation": {"task_gate": {"tol_m": .4}}},
                                 validator, replayer, teacher_kpts, time.perf_counter() + 60)
    assert rec["status"] == "ok"
    assert rec["metrics"]["fitting"]["teacher_fit_fk_mpjpe_box_m"] == 0
    assert curves["penetration_families"].shape == (T, 3)
    assert curves["hand_fk_error_l_m"].max() == 0
    assert diag.hash_problems({rec["product"]["path"]: rec["product"]["sha256"]}) == []


def test_oracle_uses_production_decode_fusion_and_world_lifting():
    import eval_generation as raw
    from ireset.data.schema import InteractionClip, ReferenceCondition
    from ireset.data.windows import make_tokenizer
    from ireset.model.tokens import RootStats

    T = 108
    tokenizer = make_tokenizer(RootStats(np.array([0., 0., .7]), np.array([.4, .3, .2])))
    wfc = np.eye(4)
    wfc[:3, :3] = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    wfc[:3, 3] = [.3, -.4, 0.]
    root = np.tile([.2, .3, .8, 1., 0., 0., 0.], (T, 1))
    root[:, 0] += np.linspace(0, .1, T)
    q = np.tile((tokenizer.lo + tokenizer.hi) / 2, (T, 1))
    box = np.tile([.6, 0., .8, 1., 0., 0., 0.], (T, 1))
    cond = ReferenceCondition("oracle", np.zeros((T, 52, 3)), box.copy(), box, world_from_clip=wfc)
    clip = InteractionClip(cond, q, root)
    trajectory = diag.oracle_roundtrip(tokenizer, clip)
    assert np.abs(trajectory.root[:, :3] - root[:, :3]).max() < 1e-5
    assert raw.geodesic_deg(trajectory.root[:, 3:7], root[:, 3:7]).max() < 1e-3
    assert np.abs(trajectory.q - q).max() < 1e-5


def test_exhausted_budget_preserves_incomplete_report_without_child(tmp_path, monkeypatch):
    diag.write_json(tmp_path / "costs.json", [{"stage": "register", "status": "done", "wall_s": 3600.}])
    (tmp_path / "freeze").mkdir()
    (tmp_path / "freeze/registration.json").write_text("{}")
    monkeypatch.setattr(diag, "verify_registration", lambda run: {"demos": ["demo"], "unit_timeout_s": 600})
    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **kw: pytest.fail("child launched without budget"))
    assert not diag.run_diagnosis(tmp_path)
    report = diag.read_json(tmp_path / "diagnosis/attempt_001/attribution.json")
    assert report["complete"] is False and report["budget_stopped"] is True
    assert not report["results"]


def test_unknown_history_refused_before_attempt_allocation(tmp_path, monkeypatch):
    diag.write_json(tmp_path / "costs.json", [{"stage": "register", "status": "running", "wall_s": 0.}])
    monkeypatch.setattr(diag, "verify_registration", lambda run: {})
    with pytest.raises(ValueError, match="history incomplete"):
        diag.run_diagnosis(tmp_path)
    assert not (tmp_path / "diagnosis").exists()


def test_framewise_shortest_arc_can_jump_when_smooth_windows_cross_pi():
    """A second mechanism: including all contributors alone cannot fix it."""
    starts, T = [0, 30], 90
    decoded = list(_constant_windows(starts, [0, 0]))
    angles = 170 + np.arange(60) * 20 / 29
    decoded[1][1] = Rotation.from_euler("z", angles[:, None], degrees=True).as_quat()[:, [3, 0, 1, 2]]
    before = fuse_windows(*[list(v) for v in decoded], starts, T, 30)
    after = diag.fuse_all_contributors(decoded, starts, T)
    rotations = Rotation.from_quat(before[1][:, [1, 2, 3, 0]])
    jumps = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())
    assert np.diff(angles).max() < 1
    assert jumps.max() > 150
    assert np.allclose(_angles(before[1]), _angles(after[1]), atol=1e-8)
