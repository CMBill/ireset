"""Synthetic/interface counterexamples for the W1 prediction diagnostic.

No real checkpoint is loaded here. The fixed-set tests use the plan's T table
only as metadata; the measurement tests use an analytic clip-frame state and
synthetic geometry so the math/contracts can be checked without a model call.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ireset.eval import prediction_consistency as pc  # noqa: E402
from ireset.model.tokens import RootStats  # noqa: E402

T_BY_DEMO = {
    "sub10_largebox_063": 133, "sub12_largebox_047": 358,
    "sub16_largebox_000": 89, "sub3_largebox_001": 178,
    "sub7_largebox_003": 147, "sub8_largebox_005": 190,
    "sub10_largebox_100": 73, "sub12_largebox_008": 161,
    "sub16_largebox_052": 147, "sub3_largebox_018": 191,
    "sub7_largebox_032": 184, "sub8_largebox_001": 151,
    "sub16_largebox_048": 168,
}
TRAIN6 = ["sub10_largebox_063", "sub12_largebox_047", "sub16_largebox_000",
          "sub3_largebox_001", "sub7_largebox_003", "sub8_largebox_005"]
VAL6 = ["sub10_largebox_100", "sub12_largebox_008", "sub16_largebox_052",
        "sub3_largebox_018", "sub7_largebox_032", "sub8_largebox_001"]
KNOWN = "sub16_largebox_048"


class FakeTokenizer:
    def __init__(self) -> None:
        self.lo = np.full(29, -1.0)
        self.hi = np.full(29, 1.0)
        self.root_stats = RootStats(mean=np.zeros(3), std=np.ones(3))


class FakeAssembler:
    device = "cpu"

    def __call__(self, pred, batch):
        return torch.zeros((), dtype=torch.float32), {
            "pos": 0.0, "rot6d": 0.0, "joint": 0.0,
            "geodesic_rad2": 0.0, "vel": 0.0, "fk_m2": 0.0, "total": 0.0,
        }


def identity_x0(F=60):
    x0 = np.zeros((F, 38), dtype=np.float32)
    x0[:, 3] = 1.0
    x0[:, 7] = 1.0
    return x0


def geometry(F=60, penetration=0.0, gap=0.05, foot=0.05):
    return {
        "penetration": np.full(F, penetration),
        "penetration_families": np.zeros((F, 3)),
        "worst_geom_pairs": np.full((F, 3, 2), -1, dtype=np.int32),
        "box_ground": np.zeros(F),
        "hand_gap": {"l": np.full(F, gap), "r": np.full(F, gap)},
        "foot_gap": {"l": np.full(F, foot), "r": np.full(F, foot)},
        "robot_kpts": np.zeros((F, 15, 3)),
        "kpt_err_w": np.zeros(F), "kpt_err_b": np.zeros(F),
    }


def synthetic_window():
    tok = FakeTokenizer()
    x0 = identity_x0()
    decoded = pc.decode_production(x0, tok)
    root = pc.lift_clip_to_world(decoded[0].astype(np.float64),
                                 decoded[1].astype(np.float64), np.eye(4))
    teacher_kpts = np.zeros((60, 15, 3))
    out = pc.measure_window(
        demo="demo", model="deterministic", start=0, T=60, kind="regular",
        group="train6", pred_x0=x0, teacher_x0=x0, decoded=decoded,
        world_from_clip=np.eye(4), teacher_root=root,
        teacher_joints=decoded[2].astype(np.float64),
        teacher_kpts=teacher_kpts, model_geometry=geometry(),
        teacher_geometry=geometry(),
        tokenizer=tok, assembler=FakeAssembler(),
        schedule=__import__("ireset.embodied.contact_sites",
                            fromlist=["ContactSchedule"]).ContactSchedule(),
        frame_dt=1 / 30, task_tol_m=0.4, hand_gap_m=0.01,
        saved_world_root=root, decode_device="cpu")
    return out, tok


# ---------------------------------------------------------------------------
# exact sets / denominators
# ---------------------------------------------------------------------------

def test_window_plan_is_exact_52_65_130_520():
    plan = pc.build_window_plan(T_BY_DEMO, TRAIN6, VAL6, KNOWN)
    assert plan["counts"]["regular_windows"] == 52
    assert plan["counts"]["tail_windows"] == 13
    assert plan["counts"]["inference_windows"] == 65
    assert plan["counts"]["teacher_references"] == 65
    assert plan["counts"]["model_windows"] == 130
    assert plan["counts"]["probe_units"] == 520
    assert len(plan["teacher_keys"]) == 65
    assert len(plan["window_keys"]) == 130
    assert len(plan["probe_keys"]) == 520
    assert plan["totals"]["groups"]["train6"] == {
        "regular": 26, "inference": 32, "tail": 6}
    assert plan["totals"]["groups"]["val6"] == {
        "regular": 22, "inference": 28, "tail": 6}
    assert plan["totals"]["groups"]["known_case"] == {
        "regular": 4, "inference": 5, "tail": 1}


def test_missing_or_duplicated_demo_cannot_shrink_the_denominator():
    with pytest.raises(ValueError):
        pc.build_window_plan({k: v for k, v in T_BY_DEMO.items()
                              if k != KNOWN}, TRAIN6, VAL6, KNOWN)
    with pytest.raises(ValueError):
        pc.build_window_plan(T_BY_DEMO, TRAIN6 + [TRAIN6[0]], VAL6, KNOWN)
    changed = dict(T_BY_DEMO)
    changed[KNOWN] = 180
    with pytest.raises(ValueError):
        pc.build_window_plan(changed, TRAIN6, VAL6, KNOWN)
    with pytest.raises(ValueError):
        pc.build_window_plan(dict(T_BY_DEMO), TRAIN6, VAL6[:-1], KNOWN)


def test_tail_start_is_not_masqueraded_as_regular():
    for demo, T in T_BY_DEMO.items():
        regular = pc.regular_starts(T)
        tail = pc.tail_starts(T)
        assert tail == [T - pc.WINDOW]
        assert tail[0] not in regular
        assert pc.inference_starts(T) == regular + tail


def test_probe_key_matrix_has_exact_520_and_five_t_semantics():
    keys = []
    for demo, T in T_BY_DEMO.items():
        for start in pc.inference_starts(T):
            keys += pc.probe_keys_for_window(demo, start)
    assert len(keys) == 520 and len(set(keys)) == 520
    assert sum("teacher_roundtrip" in k for k in keys) == 65
    assert sum("deterministic_generation" in k for k in keys) == 65
    assert sum("diffusion_denoise" in k for k in keys) == 325
    assert sum("diffusion_ddim25" in k for k in keys) == 65
    assert all("null" in pc.probe_key("d", 0, "teacher_roundtrip") for _ in [0])
    assert pc.probe_key("d", 0, "diffusion_denoise", 0).endswith("|0000")


# ---------------------------------------------------------------------------
# overlap / time
# ---------------------------------------------------------------------------

def test_three_window_middle_positive_contributor_and_all_pairs():
    starts = [0, 30, 60, 73]
    T = 133
    entries = pc.contributor_entries(starts, T)
    # t=73: windows 1 and 2 are active, window 3 has weight 0 at its first frame.
    assert {i for i, _, _ in entries[73]} == {1, 2}
    # t=74: all three are active and the middle window is still positive.
    assert {i for i, _, _ in entries[74]} == {1, 2, 3}
    assert all(w > 0 for _, _, w in entries[74])
    keys = pc.cross_window_pair_keys("d", "deterministic", starts, T)
    expected = sum(len(row) * (len(row) - 1) // 2 for row in entries)
    assert len(keys) == expected
    assert pc.pair_key("d", "deterministic", 74, 1, 3) in keys
    assert pc.pair_key("d", "deterministic", 73, 1, 3) not in keys


def test_activation_transition_is_not_deleted():
    starts = [0, 60]
    T = 120
    assert pc.activation_transition_frames([0, 30], 90) == []


def test_penetration_classes_cover_all_frames_and_mixed_case():
    starts = [0, 30]
    T = 90
    window_pen = np.zeros((2, 60))
    window_pen[0, :] = 0.01
    window_pen[1, :] = 0.0
    fused = np.zeros(T)
    fused[31:59] = 0.01
    out = pc.same_frame_penetration_classes(
        demo="d", model="m", starts=starts, T=T,
        window_penetration=window_pen, fused_penetration=fused)
    assert sum(out["counts"].values()) == T
    assert out["counts"]["mixed"] == 28
    assert out["fusion"]["mixed_fail"] == 28
    assert len(out["fusion_only_frames"]) == 0


def test_fusion_only_requires_every_contributor_below_the_gate():
    starts = [0, 30]
    T = 90
    window_pen = np.zeros((2, 60))
    fused = np.zeros(T)
    fused[40] = 0.01
    out = pc.same_frame_penetration_classes(
        demo="d", model="m", starts=starts, T=T,
        window_penetration=window_pen, fused_penetration=fused)
    assert out["fusion_only_frames"] == [40]


def test_angular_speed_is_rad_per_s_and_seam_step_is_degrees():
    q = np.zeros((90, 4))
    q[:, 0] = 1
    q[1:30] = [math.cos(math.radians(30) / 2), math.sin(math.radians(30) / 2), 0, 0]
    q[30:] = [math.cos(math.radians(60) / 2), math.sin(math.radians(60) / 2), 0, 0]
    facts = pc.fused_temporal_facts(
        root=np.concatenate([np.zeros((90, 3)), q], 1),
        joints=np.zeros((90, 29)), starts=[0, 30], frame_dt=1 / 30)
    assert facts["root_rotation_step_deg"][1] == pytest.approx(30, abs=1e-6)
    assert facts["root_angular_speed_radps"][1] == pytest.approx(
        math.radians(30) * 30, rel=1e-6)
    # seam step is a single-step angle, not rad/s
    assert facts["seam_max_step_deg"] == pytest.approx(30, abs=1e-6)
    assert facts["seam_max_radps"] == pytest.approx(math.radians(30) * 30, rel=1e-6)


# ---------------------------------------------------------------------------
# coordinates / representation
# ---------------------------------------------------------------------------

def test_non_trivial_world_clip_roundtrip_and_quaternion_sign():
    yaw = 0.7
    c, s = math.cos(yaw), math.sin(yaw)
    wfc = np.array([[c, -s, 0, 1.2], [s, c, 0, -2.3], [0, 0, 1, 0.8],
                    [0, 0, 0, 1]], dtype=np.float64)
    root = np.array([[0.3, -0.4, 1.1, 0.9238795, 0.0, 0.0, 0.3826834]])
    clip = pc.world_to_clip_state(root, wfc)
    back = pc.lift_clip_to_world(clip[:, :3], clip[:, 3:7], wfc)
    assert np.allclose(back[0, :3], root[0, :3], atol=1e-12)
    assert pc.quat_angle_deg(back[0, 3:7], root[0, 3:7]).max() < 1e-8
    assert pc.quat_angle_deg(root[0, 3:7], -root[0, 3:7]).max() < 1e-8


def test_zero_norm_and_nonfinite_inputs_are_rejected_not_substituted():
    tok = FakeTokenizer()
    bad = identity_x0()
    bad[0, 3:9] = 0.0
    assert any("degenerate" in p for p in pc.validate_normalized(bad))
    with pytest.raises(ValueError):
        pc.quat_angle_deg(np.array([0.0, 0, 0, 0]), np.array([1.0, 0, 0, 0]))
    assert "decoded_pos_nonfinite" in " ".join(
        pc.validate_decoded(np.full((2, 3), np.nan), np.zeros((2, 4)),
                            np.zeros((2, 29)), frames=2))
    assert "decoded_zero_norm_quaternion" in pc.validate_decoded(
        np.zeros((2, 3)), np.zeros((2, 4)), np.zeros((2, 29)), frames=2)


def test_production_decode_roundtrip_is_bitwise_on_the_same_device():
    tok = FakeTokenizer()
    x0 = identity_x0()
    a = pc.decode_production(x0, tok, device="cpu")
    b = pc.decode_production(x0, tok, device="cpu")
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert np.allclose(a[0], 0.0) and np.allclose(a[2], 0.0)
    assert np.allclose(a[1][:, 0], 1.0)


# ---------------------------------------------------------------------------
# measurement / curves
# ---------------------------------------------------------------------------

def test_measure_window_is_finite_and_zero_error_on_an_analytic_target():
    out, _ = synthetic_window()
    assert out["status"] == "measured"
    assert out["metrics"]["loss"]["total"] == 0.0
    assert out["metrics"]["physical"]["root_position"]["max_m"] == pytest.approx(0, abs=1e-6)
    assert out["metrics"]["physical"]["joints"]["max_abs_rad"] == pytest.approx(0, abs=1e-8)
    assert out["metrics"]["geometry"]["penetration"]["max_m"] == 0.0
    assert out["metrics"]["bins"]["0-10"]["frames"] == 10
    assert pc.verify_window_curves(out["curves"]) == []


def test_curve_contract_rejects_nan_inf_missing_and_bad_undefined_mask():
    out, _ = synthetic_window()
    curves = dict(out["curves"])
    curves["root_pos_error_m"] = curves["root_pos_error_m"].copy()
    curves["root_pos_error_m"][3] = np.nan
    assert any("nonfinite" in p for p in pc.verify_window_curves(curves))
    curves = dict(out["curves"])
    curves["root_speed_mps"] = curves["root_speed_mps"].copy()
    curves["root_speed_mps"][0] = 0.0
    assert any("first_frame" in p for p in pc.verify_window_curves(curves))
    curves = dict(out["curves"])
    curves.pop("teacher_x0")
    assert any("missing" in p for p in pc.verify_window_curves(curves))


def test_geometry_facts_separate_new_and_shared_teacher_defects():
    teacher = geometry(penetration=0.01)
    model = geometry(penetration=0.01)
    model["penetration"][:5] = 0.0
    model["penetration_families"][5:, 0] = 0.01
    teacher["penetration_families"][:, 0] = 0.01
    facts = pc.geometry_window_facts(
        model, teacher, np.zeros((60, 52, 3)), np.zeros((60, 15, 3)), 0,
        type("C", (), {"hand_gap_m": 0.01, "foot_ground_gap_m": 0.005})())
    assert facts["penetration"]["violation_frames"] == 55
    assert facts["penetration"]["shared_violation_frames"] == 55
    assert facts["penetration"]["new_violation_frames"] == 0


def test_bins_and_reference_facts_have_fixed_boundaries():
    from ireset.embodied.contact_sites import ContactInterval, ContactSchedule
    sched = ContactSchedule(intervals=[
        ContactInterval("left_hand", 0, 20, "object"),
        ContactInterval("right_hand", 10, 30, "object"),
        ContactInterval("left_foot", 0, 60, "ground"),
    ])
    ref = pc.reference_window_facts(sched, 0, 60)
    assert ref["ref_grasp_frames"] == {"l": 20, "r": 20}
    assert ref["ref_support_frames"] == {"l": 60, "r": 0}
    assert ref["both_reference_frames"] == 10
    assert ref["first_both_overlap_local_frame"] == 10
    out, _ = synthetic_window()
    assert set(out["metrics"]["bins"]) == {"0-10", "10-50", "50-60"}


# ---------------------------------------------------------------------------
# noise / RNG
# ---------------------------------------------------------------------------

def test_noise_seed_is_deterministic_and_t0_is_not_clean():
    a = pc.noise_seed_for_window("sub10_largebox_063", 0)
    b = pc.noise_seed_for_window("sub10_largebox_063", 0)
    c = pc.noise_seed_for_window("sub10_largebox_063", 30)
    assert a == b != c
    eps, seed = pc.epsilon_for_window("sub10_largebox_063", 0)
    assert seed == a
    eps2, seed2 = pc.epsilon_for_window("sub10_largebox_063", 0)
    assert np.array_equal(eps.numpy(), eps2.numpy())
    assert eps.shape == (60, 38) and eps.dtype == torch.float32
    from diffusers import DDPMScheduler
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
                          beta_start=1e-4, beta_end=0.02,
                          prediction_type="sample", clip_sample=False)
    assert float(sched.alphas_cumprod[0]) < 1.0
    assert float(sched.alphas_cumprod[999]) != 1.0


def test_noise_generation_does_not_consume_global_rng():
    before = torch.get_rng_state().clone()
    pc.epsilon_for_window("sub10_largebox_063", 0)
    pc.epsilon_for_window("sub10_largebox_100", 13)
    assert torch.equal(before, torch.get_rng_state())


# ---------------------------------------------------------------------------
# aggregation / attribution
# ---------------------------------------------------------------------------

def test_aggregation_keeps_expected_denominator_and_never_shrinks_macro():
    plan = pc.build_window_plan(T_BY_DEMO, TRAIN6, VAL6, KNOWN)
    row = {
        "key": pc.window_key(TRAIN6[0], "deterministic", 0),
        "demo": TRAIN6[0], "model": "deterministic", "window_start": 0,
        "kind": "regular", "group": "train6", "status": "measured",
        "metrics": {
            "physical": {"fk": {"mpjpe_m": 0.02}},
            "geometry": {"penetration": {"max_m": 0.0,
                                         "violation_frames": 0,
                                         "new_violation_frames": 0,
                                         "shared_violation_frames": 0}},
            "loss": {},
        },
    }
    out = pc.aggregate_window_rows([row], plan)
    rec = out["train6"]["deterministic|regular"]
    assert rec["n_expected_windows"] == 26
    assert rec["n_measured_windows"] == 1
    assert rec["n_missing_windows"] == 25
    assert rec["metrics"]["physical.fk.mpjpe_m"]["macro_equal_demo"] is None
    assert rec["metrics"]["physical.fk.mpjpe_m"]["n_demo_measured"] == 1


def test_attribution_marks_mixed_and_insufficient_without_a_responsibility_share():
    oracle = {"fk_mean_m": 0.001}
    w_summary = {
        "groups": {
            "train6": {"deterministic|regular": {"metrics": {
                "physical.fk.mpjpe_m": {"macro_equal_demo": 0.1}}}},
            "val6": {"deterministic|regular": {"metrics": {
                "physical.fk.mpjpe_m": {"macro_equal_demo": 0.1}}}},
        },
        "pair_summary": {"max_position_diff_m": 0.05,
                         "max_rotation_diff_deg": 2.0,
                         "max_joint_max_rad": 0.1},
        "teacher_inheritance": {"shared_violation_frames": 0,
                                "new_violation_frames": 0},
    }
    p_summary = {"denoise_snr": {
        "ddim25": {"normalized_total_mean": 1.0},
        "999": {"normalized_total_mean": 0.5},
    }}
    out = pc.attribution_judgement(w_summary=w_summary, p_summary=p_summary,
                                   oracle=oracle, evidence_gaps=[])
    assert out["status"] == "mixed_evidence"
    assert "training_fit" in out["supported"]
    assert "cross_window" in out["supported"]
    assert "responsibility_percentage" not in json.dumps(out).lower()
    missing = pc.attribution_judgement(w_summary=None, p_summary=None,
                                       oracle=None,
                                       evidence_gaps=["missing"])
    assert missing["status"] == "attribution_insufficient"


# ---------------------------------------------------------------------------
# script protocol / binding
# ---------------------------------------------------------------------------

def _import_script():
    import diagnose_w1_prediction_consistency as d
    return d


def test_guard_run_and_resolve_run_reject_overlap_and_nesting(tmp_path, monkeypatch):
    d = _import_script()
    monkeypatch.setattr(d, "ROOT", tmp_path)
    (tmp_path / "derived/experiments/omniretarget_first").mkdir(parents=True)
    source = tmp_path / "derived/experiments/omniretarget_first/source"
    source.mkdir()
    monkeypatch.setattr(d, "SOURCE", source)
    monkeypatch.setattr(d, "ATTRIBUTION", source / "att")
    monkeypatch.setattr(d, "FUSION", source / "fus")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ValueError):
        d.guard_run(outside)
    nested = source / "child"
    with pytest.raises(ValueError):
        d.guard_run(nested)
    empty = tmp_path / "derived/experiments/omniretarget_first/run"
    assert d.guard_run(empty) == empty.resolve()
    empty.mkdir()
    (empty / "x").write_text("x")
    with pytest.raises(ValueError):
        d.guard_run(empty)
    assert d.resolve_run(empty) == empty.resolve()


def test_saved_window_loader_rejects_drift_and_bad_starts(tmp_path, monkeypatch):
    d = _import_script()
    path = tmp_path / "windows.npz"
    starts = np.array([0, 30], dtype=np.int64)
    arrays = {
        "decoded_pos": np.zeros((2, 60, 3), np.float32),
        "decoded_quat": np.tile(np.array([1, 0, 0, 0], np.float32), (2, 60, 1)),
        "decoded_q": np.zeros((2, 60, 29), np.float32),
        "normalized_prediction": np.zeros((2, 60, 38), np.float32),
        "window_root": np.zeros((2, 60, 7), np.float64),
        "starts": starts,
    }
    for i in range(2):
        arrays["window_root"][i, :, 3] = 1.0
    np.savez(path, **arrays)
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    att = {"models": {"deterministic": {"windows": {
        "path": str(path), "sha256": sha}}}}
    got = d._load_saved_windows(att, "deterministic", [0, 30])
    assert got["starts"].tolist() == [0, 30]
    with pytest.raises(ValueError):
        d._load_saved_windows(att, "deterministic", [0, 31])
    bad = {"models": {"deterministic": {"windows": {
        "path": str(path), "sha256": "0" * 64}}}}
    with pytest.raises(ValueError):
        d._load_saved_windows(bad, "deterministic", [0, 30])


def test_historical_snapshot_is_verified_against_saved_bytes(tmp_path):
    d = _import_script()
    snap = tmp_path / "snap"
    snap.mkdir()
    f = snap / "impl.py"
    f.write_text("x = 1\n")
    import hashlib
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    reg = {"implementation_snapshot": {
        "files": {"impl.py": {"sha256": sha, "snapshot": str(f)}},
        "tracked_diff": {"path": str(snap / "diff.patch"),
                         "sha256": hashlib.sha256(b"").hexdigest()}}}
    (snap / "diff.patch").write_bytes(b"")
    got = d.historical_implementation(reg)
    assert got["files"]["impl.py"]["sha256"] == sha
    f.write_text("x = 2\n")
    with pytest.raises(ValueError):
        d.historical_implementation(reg)


def test_running_and_publishing_attempts_are_not_consumable(tmp_path):
    d = _import_script()
    run = tmp_path / "run"
    run.mkdir()
    (run / "freeze/attempt_001").mkdir(parents=True)
    d._preregister_attempt(run, "register", "attempt_001", __import__("time")
                           .perf_counter(), cohort="cohort")
    row, problems = d.stage_state(run, "register")
    assert row["status"] == "running"
    assert "attempt_running" in problems
    # Publish a terminal-looking row under the non-consumable status.
    d._replace_index(run, "register", "attempt_001", {
        "status": "publishing", "complete": True, "measured_wall_s": 0.1})
    d._upsert_cost(run, "register", "attempt_001", 0.1,
                   {"complete": True}, "publishing")
    row, problems = d.stage_state(run, "register")
    assert "attempt_final_check_pending" in problems


def test_cost_three_way_disagreement_is_unknown(tmp_path):
    d = _import_script()
    run = tmp_path / "run"
    run.mkdir()
    (run / "freeze/attempt_001").mkdir(parents=True)
    d._preregister_attempt(run, "register", "attempt_001", __import__("time")
                           .perf_counter(), cohort="cohort")
    d._replace_index(run, "register", "attempt_001", {
        "status": "done", "complete": True, "measured_wall_s": 1.0})
    d._upsert_cost(run, "register", "attempt_001", 2.0,
                   {"complete": True}, "done")
    assert d._stage_history(run, "register") is None


def test_finish_stage_seals_and_binds_report_hash(tmp_path):
    d = _import_script()
    run = tmp_path / "run"
    run.mkdir()
    (run / "freeze/attempt_001").mkdir(parents=True)
    report = run / "freeze/attempt_001/report.json"
    report.write_text("{}")
    d._preregister_attempt(run, "register", "attempt_001", __import__("time")
                           .perf_counter(), cohort="cohort")
    budget = {"remaining_s": 100.0}
    ok = d._finish_stage(
        run, "register", "attempt_001", __import__("time").perf_counter(),
        budget, {"complete": True, "status": "done",
                 "report": str(report)}, {"x": 1})
    assert ok is True
    row, problems = d.stage_state(run, "register")
    assert row["status"] == "done" and row["complete"] is True
    assert problems == []


def test_budget_exhaustion_is_incomplete_not_done(tmp_path):
    d = _import_script()
    run = tmp_path / "run"
    run.mkdir()
    (run / "freeze/attempt_001").mkdir(parents=True)
    d._preregister_attempt(run, "register", "attempt_001", __import__("time")
                           .perf_counter(), cohort="cohort")
    ok = d._finish_stage(
        run, "register", "attempt_001", __import__("time").perf_counter(),
        {"remaining_s": 0.0}, {"complete": True, "status": "done"}, {})
    assert ok is False
    row, _ = d.stage_state(run, "register")
    assert row["complete"] is False
    assert row["status"] == "incomplete"


# ---------------------------------------------------------------------------
# additional review counterexamples
# ---------------------------------------------------------------------------

def test_count_metrics_must_be_integers_and_zero_reference_keeps_other_side():
    plan = pc.build_window_plan(T_BY_DEMO, TRAIN6, VAL6, KNOWN)
    row = {
        "key": pc.window_key(TRAIN6[0], "deterministic", 0),
        "demo": TRAIN6[0], "model": "deterministic", "window_start": 0,
        "kind": "regular", "group": "train6", "status": "measured",
        "metrics": {"physical": {"fk": {"mpjpe_m": 0.02}},
                    "geometry": {"penetration": {
                        "max_m": 0.0, "violation_frames": 1.5,
                        "new_violation_frames": 0,
                        "shared_violation_frames": 0}},
                    "loss": {}},
    }
    out = pc.aggregate_window_rows([row], plan)
    rec = out["train6"]["deterministic|regular"]
    assert rec["metrics"]["geometry.penetration.violation_frames"]["n_demo_measured"] == 0
    from ireset.embodied.contact_sites import ContactInterval, ContactSchedule
    sched = ContactSchedule(intervals=[
        ContactInterval("left_hand", 0, 10, "object"),
        ContactInterval("left_foot", 0, 60, "ground")])
    ref = pc.reference_window_facts(sched, 0, 60)
    assert ref["ref_grasp_frames"]["r"] == 0
    assert ref["ref_support_frames"]["r"] == 0
    facts = pc.geometry_window_facts(
        geometry(), geometry(), np.zeros((60, 52, 3)),
        np.zeros((60, 15, 3)), 0,
        type("C", (), {"hand_gap_m": 0.01, "foot_ground_gap_m": 0.005})())
    assert facts["foot_gap"]["r"]["min_m"] == pytest.approx(0.05)


def test_compare_metric_docs_rejects_tampering_but_allows_float_tolerance():
    d = _import_script()
    a = {"physical": {"fk": {"mpjpe_m": 0.01}}, "n": 3,
         "flag": True, "text": "x"}
    b = {"physical": {"fk": {"mpjpe_m": 0.0100005}}, "n": 3,
         "flag": True, "text": "x"}
    assert d._compare_metric_docs(a, b, "row") == []
    c = json.loads(json.dumps(a))
    c["physical"]["fk"]["mpjpe_m"] = 0.02
    assert d._compare_metric_docs(a, c, "row")
    c = json.loads(json.dumps(a))
    c["n"] = 4
    assert d._compare_metric_docs(a, c, "row")


def test_replay_compare_detects_normalized_and_rotation_drift():
    d = _import_script()
    starts = [0]
    normalized = np.zeros((1, 60, 38), np.float32)
    decoded = (np.zeros((1, 60, 3), np.float32),
               np.tile(np.array([1, 0, 0, 0], np.float32), (1, 60, 1)),
               np.zeros((1, 60, 29), np.float32))
    saved = {"normalized_prediction": normalized.copy(),
             "decoded_pos": decoded[0].copy(),
             "decoded_quat": decoded[1].copy(),
             "decoded_q": decoded[2].copy()}
    rep = d._replay_compare(normalized, decoded, saved, starts)
    assert rep["tolerance_pass"] is True
    bad = {k: v.copy() for k, v in saved.items()}
    bad["normalized_prediction"][0, 0, 0] = 0.1
    assert d._replay_compare(normalized, decoded, bad, starts)["tolerance_pass"] is False
    bad = {k: v.copy() for k, v in saved.items()}
    bad["decoded_quat"][0, 0] = np.array([math.cos(0.01), math.sin(0.01), 0, 0],
                                         np.float32)
    assert d._replay_compare(normalized, decoded, bad, starts)["tolerance_pass"] is False


def test_attribution_window_chain_detects_a_self_consistent_replacement(
        tmp_path, monkeypatch):
    d = _import_script()
    att = tmp_path / "att"
    (att / "freeze").mkdir(parents=True)
    (att / "decisions").mkdir()
    attempt = att / "diagnosis/attempt_001"
    attempt.mkdir(parents=True)
    demos = TRAIN6 + VAL6 + [KNOWN]
    results = []
    for demo in demos:
        windows = {}
        for model in ("deterministic", "diffusion"):
            path = attempt / demo / f"{model}_windows.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, x=np.array([float(len(windows))]))
            windows[model] = {"path": str(path),
                              "sha256": d.full_sha(path)}
        result = {"demo": demo, "status": "ok", "starts": [0],
                  "models": {m: {"windows": windows[m]} for m in windows}}
        (attempt / demo / "result.json").write_text(json.dumps(result))
        results.append(result)
    report = {"complete": True, "evidence_gaps": [], "results": results}
    report_path = attempt / "attribution.json"
    report_path.write_text(json.dumps(report))
    closure = {"complete": True,
               "artifacts": {"diagnosis": {
                   "path": str(report_path), "sha256": d.full_sha(report_path)}}}
    (att / "decisions/diagnostic_closure.json").write_text(json.dumps(closure))
    monkeypatch.setattr(d, "ATTRIBUTION", att)
    att_reg = {"demos": demos}
    report_out, windows_out, scenes_out = d._verify_attribution_windows(att_reg)
    assert len(windows_out) == 26 and len(scenes_out) == 13
    # Replace one window and update only its self-reported hash: the chain
    # must still fail because the authoritative record hash changed.
    victim = Path(windows_out[0]["path"])
    np.savez(victim, x=np.array([999.0]))
    with pytest.raises(ValueError):
        d._verify_attribution_windows(att_reg)


def test_near_pi_rotation_is_measured_not_hidden():
    angle = math.pi - 2e-6
    q = np.array([math.cos(angle / 2), math.sin(angle / 2), 0, 0])
    measured = pc.quat_angle_deg(q[None], np.array([[1.0, 0, 0, 0]]))[0]
    assert measured == pytest.approx(math.degrees(angle), abs=1e-4)


def test_condition_arrays_are_exactly_the_five_teacher_free_keys():
    d = _import_script()
    T = 3
    human = np.zeros((T, 52, 3))
    source = np.zeros((T, 7)); source[:, 6] = 1.0
    target = np.zeros((T, 7)); target[:, 6] = 1.0
    cond = type("Cond", (), {
        "human_kps_world": human, "source_obj_pose": source,
        "target_obj_pose": target, "world_from_clip": np.eye(4),
        "contact_flags": lambda self: np.zeros((T, 4), np.float32),
    })()
    clip = type("Clip", (), {"condition": cond, "root": np.zeros((T, 7)),
                             "q": np.zeros((T, 29))})()
    out = d._condition_arrays(clip)
    assert set(out) == set(pc.CONDITION_KEYS)
    assert "x0" not in out and "root" not in out and "teacher" not in out


def test_next_action_refuses_to_interpret_with_failures():
    out = pc.attribution_judgement(w_summary=None, p_summary=None,
                                   oracle=None, evidence_gaps=["x"])
    action = _import_script()._next_action(out, ["failure"], [])
    assert action["priority"] == "fix_evidence"
    assert action["auto_execute"] is False


def test_cohort_history_charges_a_sealed_unregistered_sibling_and_ignores_foreign(
        tmp_path, monkeypatch):
    d = _import_script()
    monkeypatch.setattr(d, "ROOT", tmp_path)
    base = tmp_path / "derived/experiments/omniretarget_first"
    base.mkdir(parents=True)
    target = base / "target"; target.mkdir()
    sibling = base / "sibling"; sibling.mkdir()
    (sibling / "freeze/attempt_001").mkdir(parents=True)
    d._preregister_attempt(sibling, "register", "attempt_001",
                           __import__("time").perf_counter(), cohort="cohort")
    import time
    ok = d._finish_stage(sibling, "register", "attempt_001", time.perf_counter(),
                         {"remaining_s": 100.0},
                         {"complete": True, "status": "done"}, {})
    assert ok is True
    foreign = base / "foreign"; foreign.mkdir()
    (foreign / "cohort.json").write_text(json.dumps({"cohort_id": "other"}))
    (foreign / "freeze/attempt_001").mkdir(parents=True)
    d._preregister_attempt(foreign, "register", "attempt_001",
                           time.perf_counter(), cohort="other")
    history = d._cohort_history(target, "cohort")
    assert history["known"] is True
    assert history["total_s"] > 0
    assert any(r["run"].endswith("sibling") for r in history["runs"])
    assert not any(r["run"].endswith("foreign") for r in history["runs"])


def test_real_sigkill_leaves_a_nonconsumable_running_attempt(tmp_path, monkeypatch):
    d = _import_script()
    monkeypatch.setattr(d, "ROOT", tmp_path)
    base = tmp_path / "derived/experiments/omniretarget_first"
    base.mkdir(parents=True)
    run = base / "killed"; run.mkdir()
    other = base / "other"; other.mkdir()
    child = tmp_path / "child.py"
    child.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "from pathlib import Path\n"
        "import diagnose_w1_prediction_consistency as d\n"
        f"run = Path({str(run)!r})\n"
        "d._preregister_attempt(run, 'register', 'attempt_001', time.perf_counter(), cohort='cohort')\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n")
    import subprocess
    proc = subprocess.Popen([sys.executable, str(child)], cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.kill()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    row, problems = d.stage_state(run, "register")
    assert row is not None and row["status"] == "running"
    assert "attempt_running" in problems
    history = d._cohort_history(other, "cohort")
    assert history["known"] is False


def test_window_matrix_lift_is_per_window_not_a_2d_call():
    yaw = 0.4
    c, s = math.cos(yaw), math.sin(yaw)
    wfc = np.array([[c, -s, 0, 1.0], [s, c, 0, 2.0], [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float64)
    pos = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    quat = np.tile(np.array([1.0, 0, 0, 0]), (2, 2, 1))
    out = pc.lift_clip_windows_to_world(pos, quat, wfc)
    assert out.shape == (2, 2, 7)
    flat = np.concatenate([pc.lift_clip_to_world(pos[i], quat[i], wfc)
                           for i in range(2)])
    assert np.allclose(out.reshape(-1, 7), flat, atol=1e-12)


def test_aggregation_accepts_direct_metrics_rows_used_by_w():
    plan = pc.build_window_plan(T_BY_DEMO, TRAIN6, VAL6, KNOWN)
    rows = []
    for demo in TRAIN6:
        rows.append({
            "status": "measured", "demo": demo, "model": "deterministic",
            "window_start": 0, "kind": "regular", "group": "train6",
            "physical": {"fk": {"mpjpe_m": 0.02}},
            "geometry": {"penetration": {
                "max_m": 0.0, "violation_frames": 0,
                "new_violation_frames": 0, "shared_violation_frames": 0}},
            "loss": {},
        })
    out = pc.aggregate_window_rows(rows, plan)
    rec = out["train6"]["deterministic|regular"]
    assert rec["n_missing_windows"] == 20
    assert rec["metrics"]["physical.fk.mpjpe_m"]["macro_equal_demo"] == pytest.approx(0.02)


def test_curve_file_comparison_detects_tamper_and_missing(tmp_path):
    d = _import_script()
    out, _ = synthetic_window()
    path = tmp_path / "curve.npz"
    np.savez_compressed(path, **out["curves"])
    assert d._compare_curve_file(path, out["curves"], "c") == []
    tampered = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in out["curves"].items()}
    tampered["root_pos_error_m"] = tampered["root_pos_error_m"].copy()
    tampered["root_pos_error_m"][3] += 1.0
    np.savez_compressed(path, **tampered)
    assert any("value" in p for p in d._compare_curve_file(
        path, out["curves"], "c"))
    assert d._compare_curve_file(tmp_path / "missing.npz", out["curves"], "c")


def test_probe_curve_writer_creates_parent_and_session_counts_recompute(tmp_path):
    d = _import_script()
    path = tmp_path / "curves" / "x.npz"
    d._save_arrays(path, {"a": np.zeros(3)})
    assert path.is_file()
    units = tmp_path / "units"
    (units / "demo").mkdir(parents=True)
    (units / "demo" / "result.json").write_text(json.dumps({
        "facts": {"sessions": {
            "deterministic": {"network_forwards": 1, "conditioner_calls": 1},
            "diffusion_denoise": {"windows": 2, "network_forwards": 2,
                                  "conditioner_calls": 2},
            "diffusion_ddim25": {"network_forwards": 25, "conditioner_calls": 1},
        }}}))
    counts = d._session_counts(units)
    assert counts["det_sessions"] == 1
    assert counts["denoise_batches"] == 2
    assert counts["ddim_sessions"] == 1
    assert counts["network_forwards"] == 28
    assert counts["high_level_calls"] == 4
