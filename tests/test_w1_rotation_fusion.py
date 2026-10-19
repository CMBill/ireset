"""Contract tests for the rotation-fusion verification driver (plan §11.7-§11.10)."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import omniretarget_w1_rotation_fusion as fusion
from ireset.model.rotation_fusion import fuse_windows_v2
from ireset.model.stitch import window_starts


DEMOS = [fusion.KNOWN_CASE] + [f"sub{i}_largebox_{i:03d}" for i in range(12)]
VAL31 = [f"sub{i}_largebox_{i:03d}" for i in range(100, 131)]


def metrics_row(*, penetration=0.02, hand_l=None, hand_r=None, both=None,
                failed=None, raw_pass=True, task_pass=True,
                fit=0.05, task_p95=0.1, task_max=0.15):
    zero = {"gap_p95_m": 0.02, "drift_max_m": 0.01,
            "relative_speed_max_mps": 0.2, "relative_speed_p95_mps": 0.15,
            "certified_frames": 5, "certified_overlap_frames": 3,
            "n_ref_frames": 10}
    foot = {"gap_p95_m": 0.01, "drift_max_m": 0.02, "speed_p95_mps": 0.3,
            "n_ref_frames": 10}
    return {
        "raw_standard_pass": raw_pass,
        "robot_strict": {"failed_checks": list(failed or []),
                         "checks": {"velocities": {"passed": True}}},
        "task": {"gate_pass": task_pass, "world_mpjpe_p95_m": task_p95,
                 "world_mpjpe_max_m": task_max},
        "penetration": {"max_m": penetration},
        "fitting": {"teacher_fit_fk_mpjpe_box_m": fit},
        "hands": {"l": hand_l, "r": hand_r},
        "feet": {"l": foot, "r": dict(foot)},
        "both": both or {"denominator_frames": 0, "certified_overlap_frames": 0,
                         "longest_certified_overlap": 0, "valid_coverage": None},
    }


def measured_row(key, metrics, *, penetration_count=1, angle=0.1,
                 hand_fk=0.01):
    return {"key": key, "status": "measured", "metrics": metrics,
            "keyframe_penetration_count": penetration_count,
            "angular_velocity_max_radps": angle,
            "hand_fk_mean_m": {"l": hand_fk, "r": hand_fk},
            "product": {"path": "/dev/null", "sha256": "x"}}


def test_expected_key_sets_are_exact_and_ordered():
    keys = fusion._expected_keys(DEMOS, VAL31)
    s, v = keys["fixed_windows"], keys["val31"]
    assert len(s) == 117 and len(set(s)) == 117
    assert len(v) == 155 and len(set(v)) == 155
    assert sum(k.startswith("model:") for k in s) == 78
    assert sum(k.startswith("roundtrip:") for k in s) == 26
    assert sum(k.startswith("teacher:") for k in s) == 13
    assert s[:13] == [f"teacher:{d}" for d in DEMOS]
    assert v[-1] == f"model:{VAL31[-1]}:diffusion:V2"


def test_rejected_rows_do_not_shrink_the_val31_denominator():
    rows = []
    for demo in VAL31:
        rows.append(measured_row(f"teacher:{demo}", metrics_row()))
        for model in ("deterministic", "diffusion"):
            rows.append(measured_row(f"model:{demo}:{model}:V0", metrics_row()))
            rows.append({"key": f"model:{demo}:{model}:V2",
                         "status": "algorithm_rejected", "metrics": None})
    matrix = fusion._val31_matrix({"val31": VAL31}, rows)
    v2 = matrix["Deterministic V2"]
    assert v2["raw_denominator"] == 31
    assert v2["rows"] == 31
    assert v2["n_computable"] == 0
    assert v2["algorithm_rejected"] == 31
    assert v2["teacher_fit_fk_mean_m_all"] is None
    assert v2["teacher_fit_fk_mean_m_computed_only"] is None


def test_computed_only_macro_never_claims_the_full_set():
    rows = []
    for i, demo in enumerate(VAL31):
        rows.append(measured_row(f"teacher:{demo}", metrics_row()))
        for model in ("deterministic", "diffusion"):
            rows.append(measured_row(f"model:{demo}:{model}:V0", metrics_row()))
            if model == "deterministic" and i == 0:
                rows.append({"key": f"model:{demo}:{model}:V2",
                             "status": "algorithm_rejected", "metrics": None})
            else:
                rows.append(measured_row(f"model:{demo}:{model}:V2", metrics_row()))
    matrix = fusion._val31_matrix({"val31": VAL31}, rows)
    det = matrix["Deterministic V2"]
    assert det["n_computable"] == 30
    assert det["teacher_fit_fk_mean_m_all"] is None
    assert det["teacher_fit_fk_mean_m_computed_only"] == pytest.approx(0.05)


def test_gq_rejects_new_penetration_and_lost_hand_measurement():
    hand = {"gap_p95_m": 0.02, "drift_max_m": 0.01,
            "relative_speed_max_mps": 0.2, "relative_speed_p95_mps": 0.15,
            "certified_frames": 5, "certified_overlap_frames": 3}
    both = {"denominator_frames": 10, "certified_overlap_frames": 3,
            "longest_certified_overlap": 2, "valid_coverage": 0.5}
    v0 = measured_row("model:d:det:V0", metrics_row(hand_l=hand, hand_r=hand,
                                                    both=both))
    v2 = measured_row("model:d:det:V2", metrics_row(hand_l=dict(hand),
                                                    hand_r=dict(hand),
                                                    both=dict(both)))
    assert fusion.gq_compare("d", "det", v0, v2)["passed"]

    worse = measured_row("model:d:det:V2", metrics_row(
        penetration=0.02001, hand_l=dict(hand), hand_r=dict(hand),
        both=dict(both)))
    verdict = fusion.gq_compare("d", "det", v0, worse)
    assert not verdict["passed"]
    assert any("penetration" in f for f in verdict["failures"])

    lost = measured_row("model:d:det:V2", metrics_row(hand_l=None,
                                                      hand_r=dict(hand),
                                                      both=dict(both)))
    verdict = fusion.gq_compare("d", "det", v0, lost)
    assert not verdict["passed"]
    assert any("lost_hand_measurement" in f for f in verdict["failures"])

    zero_ref0 = measured_row("model:d:det:V0", metrics_row())
    zero_ref2 = measured_row("model:d:det:V2", metrics_row())
    assert fusion.gq_compare("d", "det", zero_ref0, zero_ref2)["passed"]


def test_gq_rejects_fractional_or_disagreeing_both_denominator():
    hand = {"gap_p95_m": 0.02, "drift_max_m": 0.01,
            "relative_speed_max_mps": 0.2, "relative_speed_p95_mps": 0.15,
            "certified_frames": 5, "certified_overlap_frames": 3}
    v0 = measured_row("model:d:det:V0", metrics_row(
        hand_l=hand, hand_r=hand,
        both={"denominator_frames": 10.5, "certified_overlap_frames": 3,
              "longest_certified_overlap": 2, "valid_coverage": 0.5}))
    v2 = measured_row("model:d:det:V2", metrics_row(
        hand_l=hand, hand_r=hand,
        both={"denominator_frames": 10.5, "certified_overlap_frames": 3,
              "longest_certified_overlap": 2, "valid_coverage": 0.5}))
    verdict = fusion.gq_compare("d", "det", v0, v2)
    assert not verdict["passed"]
    assert any("denominator_invalid" in g for g in verdict["evidence_gaps"])

    v0["metrics"]["both"]["denominator_frames"] = 10
    v2["metrics"]["both"]["denominator_frames"] = 11
    verdict = fusion.gq_compare("d", "det", v0, v2)
    assert any("denominator_disagreement" in g for g in verdict["evidence_gaps"])


def test_gq_rejects_rejected_v2_even_if_v0_is_fine():
    v0 = measured_row("model:d:det:V0", metrics_row())
    v2 = {"key": "model:d:det:V2", "status": "algorithm_rejected", "metrics": None}
    verdict = fusion.gq_compare("d", "det", v0, v2)
    assert not verdict["passed"]
    assert any("v2_not_measured" in f for f in verdict["failures"])


def _gs_rows(tmp_path: Path):
    product = tmp_path / "product.npz"
    np.savez_compressed(product, value=np.array([1.0]))
    from ireset.data.a2_integrity import sha256_file
    rec = {"path": str(product), "sha256": sha256_file(product)}
    rows, facts = [], {}
    keys = fusion._expected_keys(DEMOS, VAL31)["fixed_windows"]
    for key in keys:
        if key.startswith("teacher:"):
            row = measured_row(key, metrics_row())
            row["product"] = rec
            rows.append(row)
            continue
        if key.startswith("roundtrip:"):
            demo = key.split(":")[1]
            fail = DEMOS.index(demo) < 3
            row = measured_row(key, metrics_row())
            row["product"] = rec
            row["root_position_max_error_m"] = 0.0
            row["root_rotation_max_error_deg"] = 0.002 if fail else 0.0
            row["joint_max_error_rad"] = 0.0
            row["hand_fk_mean_m"] = {"l": 0.0, "r": 0.0}
            if key.endswith(":V0"):
                row["binding"] = {"v0_bitwise": True}
            else:
                row["binding"] = {"v2_position_joint_bitwise": True}
                row["path_evidence"] = {"path": rec, "rejection": None}
            rows.append(row)
            continue
        row = measured_row(key, metrics_row())
        row["product"] = rec
        row["binding"] = ({"v0_bitwise": True, "v1_bitwise": True}
                          if key.endswith((":V0", ":V1")) else
                          {"v2_position_joint_bitwise": True,
                           "inputs_unmodified": True})
        if key.endswith(":V2"):
            row["path_evidence"] = {"path": rec, "rejection": None}
        rows.append(row)
        facts.setdefault(key.split(":")[1], {})
    facts[fusion.KNOWN_CASE] = {"known_case": {
        "deterministic": {"available": True, "frame_107_108_radps": 0.1},
        "diffusion": {"available": True, "frame_107_108_radps": 0.2},
    }}
    return rows, facts, rec


def test_fixed_gs_detects_key_replay_and_roundtrip_breakage(tmp_path):
    rows, facts, rec = _gs_rows(tmp_path)
    reg = {"demos": DEMOS, "expected_keys": {"fixed_windows":
                                             fusion._expected_keys(DEMOS, VAL31)["fixed_windows"]}}
    ok = fusion._fixed_gs(reg, rows, facts, True)
    assert ok["passed"], (ok["failures"], ok["evidence_gaps"])
    assert len(ok["facts"]["original_roundtrip_failures"]) == 3

    deleted = [r for r in rows if not r["key"].endswith(":diffusion:V2")]
    assert not fusion._fixed_gs(reg, deleted, facts, True)["passed"]

    duplicated = rows + [rows[0]]
    assert not fusion._fixed_gs(reg, duplicated, facts, True)["passed"]

    broken = [dict(r) for r in rows]
    next(r for r in broken if r["key"].endswith(":deterministic:V0"))["binding"] = {
        "v0_bitwise": False, "v1_bitwise": True}
    verdict = fusion._fixed_gs(reg, broken, facts, True)
    assert not verdict["passed"]
    assert any("replay_not_bitwise" in f for f in verdict["failures"])

    increment = [dict(r) for r in rows]
    rt2 = next(r for r in increment if r["key"] == f"roundtrip:{DEMOS[3]}:V2")
    rt2["root_rotation_max_error_deg"] = 0.000002
    verdict = fusion._fixed_gs(reg, increment, facts, True)
    assert not verdict["passed"]
    assert any("roundtrip_increment" in f for f in verdict["failures"])

    bad_known = json.loads(json.dumps(facts))
    bad_known[fusion.KNOWN_CASE]["known_case"]["deterministic"]["frame_107_108_radps"] = 8.0
    verdict = fusion._fixed_gs(reg, rows, bad_known, True)
    assert not verdict["passed"]
    assert any("known_case_frame_108_over_speed" in f for f in verdict["failures"])

    no_known = json.loads(json.dumps(facts))
    no_known[fusion.KNOWN_CASE]["known_case"]["diffusion"] = {"available": False}
    verdict = fusion._fixed_gs(reg, rows, no_known, True)
    assert not verdict["passed"]


def test_fixed_gs_marks_missing_metric_as_evidence_gap(tmp_path):
    rows, facts, rec = _gs_rows(tmp_path)
    rows[0]["metrics"] = metrics_row()
    rows[0]["metrics"]["fitting"] = {}
    reg = {"demos": DEMOS, "expected_keys": {"fixed_windows":
                                             fusion._expected_keys(DEMOS, VAL31)["fixed_windows"]}}
    verdict = fusion._fixed_gs(reg, rows, facts, True)
    assert verdict["evidence_gaps"]


def test_rows_are_emitted_in_registered_order():
    expected = ["a", "b", "c"]
    rows = [{"key": "c"}, {"key": "a"}, {"key": "b"}]
    assert [r["key"] for r in fusion._ordered_rows(rows, expected)] == expected
    missing = [{"key": "c"}, {"key": "a"}]
    assert fusion._ordered_rows(missing, expected) == missing
    duplicate = [{"key": "a"}, {"key": "a"}, {"key": "b"}, {"key": "c"}]
    assert fusion._ordered_rows(duplicate, expected) == duplicate


def test_known_case_facts_reports_rejection_and_frame_speeds(tmp_path):
    np.savez_compressed(tmp_path / "deterministic_V2_path.npz",
                        rejection_frame=np.asarray(120))
    speed = np.full(150, 0.1)
    speed[107] = 0.5
    speed[113] = 0.7
    np.savez_compressed(tmp_path / "diffusion_V2_path.npz",
                        angular_speed_radps=speed)
    rows = [
        {"key": f"model:{fusion.KNOWN_CASE}:deterministic:V2",
         "status": "algorithm_rejected",
         "path_evidence": {"rejection": {"code": "quaternion_blend_ill_conditioned",
                                         "frame": 120}}},
        {"key": f"model:{fusion.KNOWN_CASE}:diffusion:V2",
         "status": "measured", "path_evidence": {"rejection": None}},
    ]
    facts = fusion._known_case_facts(rows, tmp_path)
    assert facts["deterministic"]["available"] is False
    assert facts["deterministic"]["rejection"]["code"] == "quaternion_blend_ill_conditioned"
    assert facts["diffusion"]["available"] is True
    assert facts["diffusion"]["frame_107_108_radps"] == pytest.approx(0.5)
    assert facts["diffusion"]["frame_113_114_radps"] == pytest.approx(0.7)


def test_roundtrip_absolute_gate_boundaries():
    base = {"status": "measured", "root_position_max_error_m": 0.0,
            "root_rotation_max_error_deg": 0.0, "joint_max_error_rad": 0.0}
    assert not fusion._roundtrip_fail(base)
    assert not fusion._roundtrip_fail({**base, "root_rotation_max_error_deg": 0.001})
    assert fusion._roundtrip_fail({**base, "root_rotation_max_error_deg": 0.001001})
    assert fusion._roundtrip_fail({**base, "root_position_max_error_m": 0.0000101})
    assert fusion._roundtrip_fail({**base, "joint_max_error_rad": 0.0000101})


def test_guard_run_rejects_non_derived_overlap_symlink_and_nonempty(
        tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        fusion.guard_run(tmp_path / "outside")
    with pytest.raises(ValueError):
        fusion.guard_run(fusion.SOURCE / "inside", fusion.SOURCE)
    derived = tmp_path / "derived"
    derived.mkdir()
    source = derived / "source"
    source.mkdir()
    monkeypatch.setattr(fusion, "ROOT", tmp_path)
    nonempty = derived / "nonempty"
    nonempty.mkdir()
    (nonempty / "file").write_text("x")
    with pytest.raises(ValueError):
        fusion.guard_run(nonempty, source)
    link = derived / "link"
    link.symlink_to(source)
    with pytest.raises(ValueError):
        fusion.guard_run(link, source)


def test_finish_stage_seals_or_reports_incomplete(tmp_path):
    run = tmp_path / "run"
    stage_dir = run / fusion.STAGE_DIRS["register"] / "attempt_001"
    stage_dir.mkdir(parents=True)
    ok = fusion._finish_stage(run, "register", "attempt_001", time.perf_counter(),
                              {"remaining_s": 30.0},
                              {"complete": True, "artifacts": {}}, {})
    assert ok
    row, problems = fusion.stage_state(run, "register")
    assert row["status"] == "done" and row["complete"] is True and not problems
    assert fusion._stage_history(run, "register") > 0

    run2 = tmp_path / "run2"
    (run2 / fusion.STAGE_DIRS["register"] / "attempt_001").mkdir(parents=True)
    ok = fusion._finish_stage(run2, "register", "attempt_001", time.perf_counter(),
                              {"remaining_s": 0.0},
                              {"complete": True, "artifacts": {}}, {})
    assert not ok
    row, problems = fusion.stage_state(run2, "register")
    # An exhausted budget cannot be sealed with a completion claim; the row
    # must be non-consumable (incomplete or publishing/unknown), never done.
    assert row["complete"] is False and row["status"] in ("incomplete", "publishing")


def test_finish_stage_binds_report_hash(tmp_path):
    run = tmp_path / "run"
    (run / "freeze/attempt_001").mkdir(parents=True)
    (run / "freeze/execution_registration.json").write_text("{}")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"complete": True}))
    ok = fusion._finish_stage(run, "register", "attempt_001", time.perf_counter(),
                              {"remaining_s": 30.0},
                              {"complete": True, "report": str(report),
                               "artifacts": {}}, {})
    assert ok
    row, problems = fusion.stage_state(run, "register")
    assert not problems, problems
    assert row["report_sha256"]


def test_stage_state_rejects_running_and_publishing(tmp_path):
    run = tmp_path / "run"
    (run / "freeze").mkdir(parents=True)
    (run / "freeze/execution_registration.json").write_text("{}")
    (run / "freeze/attempt_001").mkdir()
    for status in ("running", "publishing"):
        fusion._append_index(run, "register", f"attempt_001", {
            "status": status, "complete": False,
            "execution_registration_sha256": fusion._registration_sha(run)})
        fusion._upsert_cost(run, "register", "attempt_001", 1.0,
                            {"complete": False}, status)
        row, problems = fusion.stage_state(run, "register")
        assert problems
        # reset for the next status
        fusion._write_index(run, [])
        fusion._write_ledger(run, [])


def test_cohort_history_inherits_same_cohort_costs_only(tmp_path):
    cohort = "c" * 64
    base = tmp_path / "experiments"
    base.mkdir()
    for name, value, status in (("run_a", 100.0, "done"), ("run_b", 50.0, "done")):
        run = base / name
        (run / "freeze/attempt_001").mkdir(parents=True)
        (run / "freeze/execution_registration.json").write_text(json.dumps({
            "execution_id": fusion.EXECUTION_ID, "cohort_id": cohort}))
        fusion._write_index(run, [{
            "stage": "register", "attempt": "attempt_001", "status": status,
            "complete": True, "measured_wall_s": value}])
        fusion._write_ledger(run, [{
            "stage": "register", "attempt": "attempt_001", "status": status,
            "wall_s": value, "measured_wall_s": value, "complete": True,
            "scope": {"stage": "register", "attempt": "attempt_001"}}])
    unknown = base / "run_c"
    (unknown / "freeze").mkdir(parents=True)
    (unknown / "freeze/execution_registration.json").write_text(json.dumps({
        "execution_id": fusion.EXECUTION_ID, "cohort_id": cohort}))
    fusion._write_ledger(unknown, [{
        "stage": "register", "attempt": "attempt_001", "status": "running",
        "wall_s": 0.0, "measured_wall_s": 0.0, "complete": False,
        "scope": {"stage": "register", "attempt": "attempt_001"}}])
    new = base / "run_new"
    history = fusion._cohort_history(new, cohort)
    assert not history["known"]
    assert "unknown_history" in history["reason"]

    other = base / "run_other"
    (other / "freeze").mkdir(parents=True)
    (other / "freeze/execution_registration.json").write_text(json.dumps({
        "execution_id": fusion.EXECUTION_ID, "cohort_id": "d" * 64}))
    # remove the unknown run and re-scan
    import shutil
    shutil.rmtree(unknown)
    history = fusion._cohort_history(new, cohort)
    assert history["known"]
    assert history["total_s"] == pytest.approx(150.0)
    assert set(history["stage_cost_s"]) == set(fusion.STAGE_BUDGETS)


def test_integration_gate_requires_full_no_regression_evidence():
    assert fusion._integration_ready(True, True, True, "done", True)
    assert not fusion._integration_ready(False, True, True, "done", True)
    assert not fusion._integration_ready(True, False, True, "done", True)
    assert not fusion._integration_ready(True, True, False, "done", True)
    assert not fusion._integration_ready(
        True, True, True, "not_run_due_to_contract_gate", True)
    assert not fusion._integration_ready(True, True, True, "done", False)


def _synthetic_v2_path(tmp_path):
    T, F, stride = 90, 60, 30
    starts = [0, 30]
    local = np.arange(F)
    angles = np.deg2rad(5.0 + 0.5 * local)
    q = np.stack([np.cos(angles / 2), np.zeros(F), np.zeros(F),
                  np.sin(angles / 2)], axis=1)
    pos = np.stack([local * 1e-3, np.zeros(F), np.zeros(F)], axis=1)
    decoded = (np.repeat(pos[None], 2, axis=0), np.repeat(q[None], 2, axis=0),
               np.zeros((2, F, 29)))
    result = fuse_windows_v2(decoded[0], decoded[1], decoded[2], starts, T, stride)
    decoded_path = tmp_path / "decoded.npz"
    np.savez_compressed(decoded_path, decoded_pos=decoded[0],
                        decoded_quat=decoded[1], decoded_q=decoded[2],
                        starts=np.asarray(starts))
    path = tmp_path / "path.npz"
    evidence = fusion._save_fusion_path(path, decoded, starts, result)
    from scipy.spatial.transform import Rotation as Rot
    wfc = np.eye(4)
    wfc[:3, :3] = Rot.from_euler("z", 0.3).as_matrix()
    wfc[:3, 3] = np.array([1.0, -2.0, 0.5])
    target_box = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                         (T, 1))
    world_root, world_joints = fusion.attribution.world_state(
        result.pos, result.quat, result.joints, wfc)
    product = tmp_path / "product_V2.npz"
    np.savez_compressed(product, root=world_root, joints=world_joints,
                        world_from_clip=wfc, target_box=target_box,
                        window_starts=np.asarray(starts, dtype=np.int64))
    key = "model:demo:deterministic:V2"
    scene = tmp_path / "fixed_windows/attempt_001/scenes/demo"
    scene.mkdir(parents=True)
    row = {"key": key, "status": "measured",
           "path_evidence": evidence,
           "product": fusion.file_record(product),
           "binding": {"decoded_window": fusion.file_record(decoded_path)}}
    result_doc = {"T": T, "starts": starts, "status": "ok", "rows": [row]}
    (scene / "result.json").write_text(json.dumps(result_doc))
    return decoded_path, path, scene, product, result_doc, wfc, target_box, key


def _synthetic_recompute(tmp_path, path, scene, result_doc, wfc, target_box, key,
                         product=None):
    from types import SimpleNamespace
    row = result_doc["rows"][0]
    row["path_evidence"]["path"] = fusion.file_record(path)
    if product is not None:
        row["product"] = fusion.file_record(product)
    (scene / "result.json").write_text(json.dumps(result_doc))
    reg = {"demos": ["demo"], "val31": [],
           "expected_keys": {"fixed_windows": [key], "val31": []}}
    clip = SimpleNamespace(condition=SimpleNamespace(
        world_from_clip=wfc, target_obj_pose=target_box, frame_dt=1 / 30))
    return fusion._recompute_samples(
        tmp_path, "fixed-windows", reg, result_doc["rows"],
        {"demo": result_doc}, time.perf_counter() + 60, lambda demo: clip)


def test_recompute_samples_rejects_tampered_lift_records(tmp_path):
    decoded_path, path, scene, product, result_doc, wfc, target_box, key = \
        _synthetic_v2_path(tmp_path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    arrays["window_signs"] = -arrays["window_signs"]
    np.savez_compressed(path, **arrays)
    report = _synthetic_recompute(tmp_path, path, scene, result_doc, wfc,
                                  target_box, key, product)
    assert any("recomputed_window_signs" in f for f in report["failures"])


def test_recompute_samples_reproduces_and_checks_rejection_prefix(tmp_path):
    T, F, stride = 60, 60, 30
    starts = [0]
    quat = np.tile(np.array([1.0, 0, 0, 0]), (1, F, 1))
    quat[0, 1] = np.array([0.999e-6, np.sqrt(1 - (0.999e-6) ** 2), 0.0, 0.0])
    decoded = (np.zeros((1, F, 3)), quat, np.zeros((1, F, 29)))
    try:
        fusion.fuse_windows_v2(decoded[0], decoded[1], decoded[2], starts, T, stride)
    except Exception as exc:  # RotationFusionRejected
        rejection = exc
    else:
        raise AssertionError("expected lift ambiguity")
    decoded_path = tmp_path / "decoded.npz"
    np.savez_compressed(decoded_path, decoded_pos=decoded[0],
                        decoded_quat=decoded[1], decoded_q=decoded[2],
                        starts=np.asarray(starts))
    path = tmp_path / "path.npz"
    evidence = fusion._save_fusion_path(path, decoded, starts, None,
                                        rejection=rejection)
    key = "model:demo:deterministic:V2"
    scene = tmp_path / "fixed_windows/attempt_001/scenes/demo"
    scene.mkdir(parents=True)
    row = {"key": key, "status": "algorithm_rejected",
           "path_evidence": evidence,
           "binding": {"decoded_window": fusion.file_record(decoded_path)}}
    result_doc = {"T": T, "starts": starts, "status": "ok", "rows": [row]}
    (scene / "result.json").write_text(json.dumps(result_doc))
    reg = {"demos": ["demo"], "val31": [],
           "expected_keys": {"fixed_windows": [key], "val31": []}}
    from types import SimpleNamespace
    clip = SimpleNamespace(condition=SimpleNamespace(frame_dt=1 / 30))
    report = fusion._recompute_samples(
        tmp_path, "fixed-windows", reg, result_doc["rows"],
        {"demo": result_doc}, time.perf_counter() + 60, lambda demo: clip)
    assert not report["failures"], report["failures"]
    assert report["checked"] == 1 and report["rejected"] == 1

    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    arrays["prefix_quat"] = arrays["prefix_quat"] * -1
    np.savez_compressed(path, **arrays)
    row["path_evidence"]["path"] = fusion.file_record(path)
    (scene / "result.json").write_text(json.dumps(result_doc))
    report = fusion._recompute_samples(
        tmp_path, "fixed-windows", reg, result_doc["rows"],
        {"demo": result_doc}, time.perf_counter() + 60, lambda demo: clip)
    assert any("rejection_prefix" in f for f in report["failures"])


def test_verify_rows_products_detects_drift(tmp_path):
    product = tmp_path / "p.npz"
    np.savez_compressed(product, value=np.array([1.0]))
    rec = fusion.file_record(product)
    rows = [{"key": "x", "status": "measured", "product": rec,
             "path_evidence": {"path": rec}}]
    assert not fusion._verify_rows_products(rows)
    product.write_bytes(b"tampered")
    assert fusion._verify_rows_products(rows)


def test_recompute_samples_detects_physical_tampering(tmp_path, monkeypatch):
    decoded_path, path, scene, product, result_doc, wfc, target_box, key = \
        _synthetic_v2_path(tmp_path)
    report = _synthetic_recompute(tmp_path, path, scene, result_doc, wfc,
                                  target_box, key, product)
    assert report["checked"] == 1 and not report["failures"], report["failures"]

    # A global quaternion sign is a representation choice, not a physical
    # change: SO(3) recomputation must accept it.
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    arrays["fused_quat_clip"] = -arrays["fused_quat_clip"]
    np.savez_compressed(path, **arrays)
    report = _synthetic_recompute(tmp_path, path, scene, result_doc, wfc,
                                  target_box, key, product)
    assert report["checked"] == 1 and not report["failures"], report["failures"]

    # A small physical rotation in the saved path must be detected.
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_rotvec([0.01, 0.0, 0.0])
    damage = (r * Rot.from_quat(arrays["fused_quat_clip"][:, [1, 2, 3, 0]])).as_quat()
    arrays["fused_quat_clip"] = damage[:, [3, 0, 1, 2]]
    np.savez_compressed(path, **arrays)
    report = _synthetic_recompute(tmp_path, path, scene, result_doc, wfc,
                                  target_box, key, product)
    assert report["failures"]
    assert any("recomputed_rotation" in f for f in report["failures"])


def test_recompute_samples_binds_product_to_recomputed_world_path(tmp_path):
    decoded_path, path, scene, product, result_doc, wfc, target_box, key = \
        _synthetic_v2_path(tmp_path)
    # A different physically valid product (e.g. the V0 clip-space result)
    # must not be accepted merely because its own metrics were recomputed.
    with np.load(product, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    arrays["root"] = arrays["root"].copy()
    from scipy.spatial.transform import Rotation as Rot
    arrays["root"][:, 3:7] = (Rot.from_euler("y", 0.05)
                              * Rot.from_quat(arrays["root"][:, [4, 5, 6, 3]])
                              ).as_quat()[:, [3, 0, 1, 2]]
    np.savez_compressed(product, **arrays)
    report = _synthetic_recompute(tmp_path, path, scene, result_doc, wfc,
                                  target_box, key, product)
    assert report["failures"]
    assert any("product_not_bound_to_recomputed_path" in f
               for f in report["failures"])


def test_gq_penetration_violation_count_cannot_increase():
    v0 = measured_row("model:d:det:V0", metrics_row(), penetration_count=5)
    worse = measured_row("model:d:det:V2", metrics_row(), penetration_count=6)
    verdict = fusion.gq_compare("d", "det", v0, worse)
    assert any("keyframe_violation_count" in f for f in verdict["failures"])

    better = measured_row("model:d:det:V2", metrics_row(), penetration_count=4)
    assert fusion.gq_compare("d", "det", v0, better)["passed"]

    # 111 -> 110 is an improvement and must not be flagged.
    high0 = measured_row("model:d:det:V0", metrics_row(), penetration_count=111)
    high2 = measured_row("model:d:det:V2", metrics_row(), penetration_count=110)
    assert fusion.gq_compare("d", "det", high0, high2)["passed"]


def test_gq_compares_feet_even_without_hand_reference():
    v0 = measured_row("model:d:det:V0", metrics_row(hand_l=None, hand_r=None))
    v2 = measured_row("model:d:det:V2", metrics_row(hand_l=None, hand_r=None))
    v2["metrics"]["feet"]["r"]["drift_max_m"] = 0.077702
    verdict = fusion.gq_compare("d", "det", v0, v2)
    assert not verdict["passed"]
    assert any("feet_r_drift_max_m" in f for f in verdict["failures"])


def test_metric_mismatches_covers_hands_counts_task_and_extras():
    import copy
    hand = {"gap_p95_m": 0.02, "drift_max_m": 0.01,
            "relative_speed_max_mps": 0.2, "relative_speed_p95_mps": 0.15,
            "certified_frames": 5, "certified_overlap_frames": 3}
    row = measured_row("model:d:det:V2", metrics_row(hand_l=hand, hand_r=hand))
    row.update({
        "measurement_gaps": [],
        "root_position_max_error_m": 0.1,
        "root_rotation_max_error_deg": 1.0,
        "joint_max_error_rad": 0.2,
        "seam_rotation_max_deg": 1.5,
    })
    metric = copy.deepcopy(row)
    assert fusion._metric_mismatches(row, metric) == []

    for field, change in (
            ("hand_gap", lambda m: m["metrics"]["hands"]["l"].__setitem__(
                "gap_p95_m", 1.02)),
            ("certified", lambda m: m["metrics"]["hands"]["l"].__setitem__(
                "certified_frames", 4)),
            ("violations", lambda m: m.__setitem__(
                "keyframe_penetration_count", 2)),
            ("task_max", lambda m: m["metrics"]["task"].__setitem__(
                "world_mpjpe_max_m", 0.16)),
            ("failed_checks", lambda m: m["metrics"]["robot_strict"].__setitem__(
                "failed_checks", ["penetration"])),
    ):
        changed = copy.deepcopy(metric)
        change(changed)
        mismatches = fusion._metric_mismatches(row, changed)
        if field == "violations":
            assert "keyframe_penetration_count" in mismatches, field
        else:
            assert "metrics" in mismatches, field

    changed = copy.deepcopy(metric)
    changed["seam_rotation_max_deg"] = 2.0
    assert "seam_rotation_max_deg" in fusion._metric_mismatches(row, changed)


def test_scene_section_requires_registered_key_matrix(tmp_path):
    run = tmp_path
    scene = run / "fixed_windows/attempt_001/scenes/demo"
    scene.mkdir(parents=True)
    key = "teacher:demo"
    reg = {"demos": ["demo"], "val31": [],
           "expected_keys": {"fixed_windows": [key], "val31": []}}
    result = {"status": "ok", "rows": [{"key": key}]}
    (scene / "result.json").write_text(json.dumps(result))
    section = fusion._scene_section(run, "fixed-windows", reg)
    assert section["rows"] and not section["failures"]

    (scene / "result.json").write_text(json.dumps(
        {"status": "ok", "rows": []}))
    section = fusion._scene_section(run, "fixed-windows", reg)
    assert any("key_matrix" in f for f in section["failures"])

    (scene / "result.json").write_text(json.dumps(
        {"status": "ok", "rows": [{"key": key}, {"key": key}]}))
    section = fusion._scene_section(run, "fixed-windows", reg)
    assert any("duplicated" in f for f in section["failures"])

    (scene / "result.json").unlink()
    section = fusion._scene_section(run, "fixed-windows", reg)
    assert any("scene_result_missing" in f for f in section["failures"])


def test_experiment_complete_requires_no_verification_failures():
    fixed = {"complete": True}
    val = {"complete": True}
    assert fusion._experiment_complete(fixed, val, [], [])
    assert not fusion._experiment_complete(fixed, val, ["recompute:x"], [])
    assert not fusion._experiment_complete(fixed, val, [], ["gap:x"])
    assert not fusion._experiment_complete({"complete": False}, val, [], [])


def test_stage_history_requires_disk_index_and_ledger(tmp_path):
    run = tmp_path / "run"
    (run / "fixed_windows/attempt_001").mkdir(parents=True)
    assert fusion._stage_history(run, "fixed-windows") is None

    row = {"stage": "fixed-windows", "attempt": "attempt_001",
           "status": "done", "complete": True, "measured_wall_s": 1.0}
    fusion._write_index(run, [row])
    assert fusion._stage_history(run, "fixed-windows") is None

    fusion._write_ledger(run, [dict(row, wall_s=1.0)])
    assert fusion._stage_history(run, "fixed-windows") == pytest.approx(1.0)

    fusion._write_ledger(run, [dict(row, wall_s=1.0, status="running",
                                    complete=False)])
    assert fusion._stage_history(run, "fixed-windows") is None


def test_preregister_attempt_is_visible_before_substantive_work(tmp_path):
    run = tmp_path / "run"
    (run / "fixed_windows/attempt_001").mkdir(parents=True)
    fusion._preregister_attempt(run, "fixed-windows", "attempt_001",
                                time.perf_counter() - 1.0)
    row, problems = fusion.stage_state(run, "fixed-windows")
    assert row["status"] == "running" and "attempt_running" in problems
    assert fusion._stage_history(run, "fixed-windows") is None


def test_cohort_history_marks_unregistered_attempt_unknown(tmp_path):
    base = tmp_path / "experiments"
    base.mkdir()
    cohort = "c" * 64
    aborted = base / "aborted"
    fusion._write_ledger(aborted, [{
        "stage": "register", "attempt": "attempt_001", "status": "running",
        "wall_s": 0.0, "measured_wall_s": 0.0, "complete": False,
        "cohort_id": cohort, "accounting_schema": fusion.ACCOUNTING_SCHEMA}])
    history = fusion._cohort_history(base / "new", cohort)
    assert not history["known"]
    assert "unregistered_attempt" in history["reason"]


def test_cohort_history_ignores_pre_schema_unregistered_rows(tmp_path):
    base = tmp_path / "experiments"
    base.mkdir()
    cohort = "c" * 64
    legacy = base / "legacy_prefix"
    fusion._write_ledger(legacy, [{
        "stage": "register", "attempt": "attempt_001", "status": "running",
        "wall_s": 0.0, "measured_wall_s": 0.0, "complete": False,
        "cohort_id": cohort}])
    history = fusion._cohort_history(base / "new", cohort)
    assert history["known"]
    assert history["total_s"] == 0.0


def test_seam_angle_is_per_frame_not_per_second():
    speed = np.array([30.0, 60.0])
    angle = fusion._seam_angle_deg(speed, np.array([0]), 1 / 30)
    assert angle == pytest.approx(np.rad2deg(1.0))
    assert fusion._seam_angle_deg(speed, np.array([], dtype=int), 1 / 30) == 0.0


def test_val31_skip_branch_propagates_seal_failure(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "fixed_windows/attempt_001").mkdir(parents=True)
    (run / "freeze").mkdir(parents=True)
    (run / "freeze/execution_registration.json").write_text("{}")
    (run / "fixed_windows/attempt_001/summary.json").write_text(json.dumps({
        "gs": {"passed": False}, "status": "done", "complete": True}))
    monkeypatch.setattr(fusion, "verify_registration", lambda run: {
        "val31": [], "expected_keys": {"val31": []}, "cohort_id": "c" * 64})
    monkeypatch.setattr(fusion, "_cohort_history", lambda run, cohort: {
        "known": True, "stage_cost_s": {}, "total_s": 0.0})
    monkeypatch.setattr(fusion, "budget_for", lambda run, stage, cohort: {
        "remaining_s": 30.0, "reserve_s": 1.0})
    monkeypatch.setattr(fusion, "stage_state", lambda run, stage: (
        {"complete": True, "status": "done"}, []))
    monkeypatch.setattr(fusion, "_finish_stage", lambda *args, **kwargs: False)
    assert fusion.val31_stage(run) is False


def test_finalize_reports_register_gap_without_unbound_errors(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "fixed_windows/attempt_001").mkdir(parents=True)
    (run / "val31/attempt_001").mkdir(parents=True)
    (run / "freeze").mkdir(parents=True)
    (run / "freeze/execution_registration.json").write_text("{}")
    (run / "fixed_windows/attempt_001/summary.json").write_text(json.dumps({
        "status": "done", "complete": True, "rows": [],
        "scene_facts": {fusion.KNOWN_CASE: {"known_case": {
            "deterministic": {"available": True, "frame_107_108_radps": 0.0}}}},
        "gq": {"passed": True, "rows": {}, "failures": [],
               "evidence_gaps": []},
        "gs": {"passed": True, "failures": [], "evidence_gaps": [],
               "facts": {}}, "counts": {}}))
    (run / "val31/attempt_001/summary.json").write_text(json.dumps({
        "status": "not_run_due_to_contract_gate", "complete": True,
        "rows": [], "matrix": None, "counts": None}))
    (run / "fixed_windows/attempt_001/rows.json").write_text("[]")
    monkeypatch.setattr(fusion, "verify_registration", lambda run: {
        "cohort_id": "c" * 64, "anchors": {}, "demos": [], "val31": [],
        "expected_keys": {"fixed_windows": [], "val31": []}})
    monkeypatch.setattr(fusion, "_cohort_history", lambda run, cohort: {
        "known": True, "stage_cost_s": {}, "total_s": 0.0})
    monkeypatch.setattr(fusion, "budget_for", lambda run, stage, cohort: {
        "remaining_s": 30.0, "reserve_s": 1.0})
    monkeypatch.setattr(fusion, "stage_state", lambda run, stage: (
        (None, ["no_attempts"]) if stage == "register"
        else ({"complete": True, "status": "done"}, [])))
    monkeypatch.setattr(fusion, "_scene_section", lambda run, stage, reg: {
        "rows": [], "meta": {},
        "failures": [], "root_dir": run / "fixed_windows/attempt_001/scenes"})
    monkeypatch.setattr(fusion, "verify_source_registration",
                        lambda reg: ({}, {}, {}, None))
    monkeypatch.setattr(fusion, "_verify_rows_products", lambda rows: [])
    monkeypatch.setattr(fusion, "_recompute_samples", lambda *args, **kwargs: {
        "failures": [], "evidence_gaps": [], "checked": 0, "expected": 0,
        "measured": 0, "rejected": 0})
    monkeypatch.setattr(fusion, "_remeasure_rows", lambda *args, **kwargs: {
        "rows": [], "failures": [], "evidence_gaps": [],
        "measured": 0, "expected_measured": 0})
    monkeypatch.setattr(fusion, "_gq_rows", lambda reg, rows, demos: {
        "passed": True, "rows": {}, "failures": [], "evidence_gaps": []})
    monkeypatch.setattr(fusion, "_known_case_facts", lambda rows, scene: {
        "deterministic": {"available": False}})
    monkeypatch.setattr(fusion, "_run_synthetic_tests", lambda deadline: {
        "command": "x", "returncode": 0, "stdout_tail": "", "wall_s": 0.0})
    captured = {}

    def fake_fixed_gs(reg, rows, scene_facts, synthetic_ok):
        captured["scene_facts"] = scene_facts
        return {"passed": True, "failures": [], "evidence_gaps": [],
                "facts": {}}

    monkeypatch.setattr(fusion, "_fixed_gs", fake_fixed_gs)
    monkeypatch.setattr(fusion, "_hash_map", lambda anchors: [])
    monkeypatch.setattr(fusion, "_finish_stage", lambda *args, **kwargs: True)
    assert fusion._finalize(run) is False
    final = json.loads(
        (run / "decisions/attempt_001/fusion_final.json").read_text())
    assert final["experiment_complete"] is False
    assert "register_stage_incomplete" in final["failures"]
    assert captured["scene_facts"] == {
        fusion.KNOWN_CASE: {"known_case": {
            "deterministic": {"available": False}}}}
    assert captured["scene_facts"] != json.loads(
        (run / "fixed_windows/attempt_001/summary.json").read_text())["scene_facts"]


def test_recompute_samples_rejects_drifted_decoded_window(tmp_path):
    decoded_path, path, scene, product, result_doc, wfc, target_box, key = \
        _synthetic_v2_path(tmp_path)
    decoded_path.write_bytes(decoded_path.read_bytes() + b"drift")
    report = _synthetic_recompute(tmp_path, path, scene, result_doc, wfc,
                                  target_box, key, product)
    assert any("decoded_windows_unreadable" in f for f in report["failures"])


def test_cohort_history_charges_sealed_unregistered_attempt(tmp_path):
    base = tmp_path / "experiments"
    base.mkdir()
    cohort = "c" * 64
    aborted = base / "aborted"
    (aborted / "freeze/attempt_001").mkdir(parents=True)
    (aborted / "cohort.json").write_text(json.dumps({
        "execution_id": fusion.EXECUTION_ID, "cohort_id": cohort}))
    fusion._write_index(aborted, [{
        "stage": "register", "attempt": "attempt_001", "status": "incomplete",
        "complete": False, "measured_wall_s": 2.0,
        "cohort_id": cohort, "accounting_schema": fusion.ACCOUNTING_SCHEMA}])
    fusion._write_ledger(aborted, [{
        "stage": "register", "attempt": "attempt_001", "status": "incomplete",
        "complete": False, "wall_s": 2.0, "measured_wall_s": 2.0,
        "cohort_id": cohort, "accounting_schema": fusion.ACCOUNTING_SCHEMA}])
    history = fusion._cohort_history(base / "new", cohort)
    assert history["known"]
    assert history["total_s"] == pytest.approx(2.0)
    assert history["runs"][0]["unregistered"] is True


def test_killed_pre_registered_attempt_stops_new_cohort_run(tmp_path):
    import subprocess
    cohort = "c" * 64
    base = tmp_path / "experiments"
    base.mkdir()
    run = base / "killed"
    script = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "import omniretarget_w1_rotation_fusion as f\n"
        "from pathlib import Path\n"
        "run = Path(sys.argv[1]); cohort = sys.argv[2]\n"
        "(run / 'fixed_windows/attempt_001').mkdir(parents=True, exist_ok=True)\n"
        "f._preregister_attempt(run, 'fixed-windows', 'attempt_001', "
        "time.perf_counter(), cohort_id=cohort)\n"
        "time.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, "-c", script, str(run), cohort],
                            cwd=ROOT)
    try:
        deadline = time.perf_counter() + 20
        while time.perf_counter() < deadline:
            if (run / "attempts_index.json").is_file() \
                    and (run / "costs.json").is_file():
                break
            time.sleep(0.05)
        else:
            raise AssertionError("pre-registration was not written")
        proc.kill()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    assert fusion._stage_history(run, "fixed-windows") is None
    history = fusion._cohort_history(base / "new", cohort)
    assert not history["known"]
    assert "unregistered_attempt" in history["reason"]
