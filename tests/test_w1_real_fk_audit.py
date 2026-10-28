"""I/C counterexamples for the W1 real FK geometry/derivative audit.

These tests are pure synthetic: no real dataset, cache, checkpoint, G1 asset
or MuJoCo model is opened.  They cover draft section 9.1 (decode/geometry,
Jacobian/gradient, evidence independence, admission/calls, ledger/seal).
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_real_fk_audit as m


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("fk_fixture")
    provider = m.SyntheticFkProvider(seed=20261028)
    provider.prepare_fixture(root)
    inputs = provider.load_inputs()
    tolerance = m.build_default_tolerance_contract()
    counter = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter)
    geometry = m.compute_geometry(provider, inputs, tolerance)
    counter_d = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter_d)
    fd = m.compute_fd(provider, inputs, geometry["arrays"], tolerance)
    return {
        "provider": provider,
        "inputs": inputs,
        "tolerance": tolerance,
        "geometry": geometry,
        "fd": fd,
        "g_calls": counter.snapshot()["counts"],
        "d_calls": counter_d.snapshot()["counts"],
    }


@pytest.fixture(scope="module")
def self_test(tmp_path_factory):
    root = tmp_path_factory.mktemp("fk_selftest") / "run"
    return m.synthetic_self_test(root, seed=20261028)


# ---------------------------------------------------------------------------
# fixed matrix and synthetic forward chain
# ---------------------------------------------------------------------------


def test_fixed_key_matrix():
    geometry = m.required_geometry_rows()
    fd = m.required_fd_rows()
    assert len(geometry) == 122
    assert len(set(geometry)) == 122
    assert len(fd) == 4672
    assert len(set(fd)) == 4672
    assert len(m.fd_states()) == 32
    assert m.FD_TANGENT_ROWS == 2240
    assert m.FD_NORMALIZED_ROWS == 2432
    assert m.FD_POSE_EVALUATIONS == 9344
    assert m.FK_DENOMINATOR_BATCH == 3600
    assert m.FK_DENOMINATOR_WINDOW == 900


def test_geometry_matrix_passes(synthetic):
    rows = synthetic["geometry"]["rows"]
    grad = synthetic["geometry"]["grad_rows"]
    assert len(rows) == 122
    assert len(grad) == 20
    assert not [r for r in rows if r["status"] != "pass"]
    assert not [r for r in grad if r["status"] != "pass"]
    counts = synthetic["g_calls"]
    nominal = m.STAGE_NOMINAL["G"]
    for kind in ("loss_assembler", "production_fk_batch",
                 "kernel64_fk_batch", "production_vjp", "mj_forward",
                 "mj_jac_site"):
        assert counts[kind] == nominal[kind], kind
    assert counts["fd_pose_eval"] == 0


def test_fd_matrix_passes(synthetic):
    assert len(synthetic["fd"]["rows"]) == 4672
    assert not synthetic["fd"]["failures"]
    assert not synthetic["fd"]["gaps"]
    counts = synthetic["d_calls"]
    assert counts["mj_forward"] == m.FD_POSE_EVALUATIONS
    assert counts["fd_pose_eval"] == m.FD_POSE_EVALUATIONS
    assert counts["mj_jac_site"] == 0


def test_scalar_denominators(synthetic):
    rows = {r["key"]: r for r in synthetic["geometry"]["rows"]
            if r["key"].startswith("fk_scalar|")}
    assert len(rows) == 10
    for model in m.MODELS:
        batch = rows[f"fk_scalar|{model}|batch"]
        assert batch["lfk_denominator"] == 3600
        assert batch["batch_equals_window_mean"] is True
        # The batch row audits the live production scalar; window rows use the
        # production element means and only agree at float32 resolution.
        assert batch["production_scalar_source"] == "live_fk_m2"
        assert batch["production_m2_bits"]
        window_means = []
        for wid in m.WINDOW_IDS:
            row = rows[f"fk_scalar|{model}|{wid}"]
            assert row["lfk_denominator"] == 900
            assert row["production_scalar_source"] == \
                "production_elements_window_mean"
            window_means.append(row["production_lfk"])
        assert batch["production_lfk"] == pytest.approx(
            float(np.mean(window_means)),
            abs=1e-6 * max(1.0, abs(batch["production_lfk"])))
        assert batch["production_m2_from_elements"] == pytest.approx(
            batch["production_m2"],
            abs=1e-6 * max(1.0, abs(batch["production_m2"])))


def test_gradient_denominator_is_not_multiplied_by_three(synthetic):
    """The FK loss denominator is B*F*S=3600, never 3*3600."""
    arrays = synthetic["geometry"]["arrays"]
    for model in m.MODELS:
        elements = arrays[f"fk_elements_{model}"]
        scalar = arrays[f"fk_scalar_production_{model}"][-1]
        assert scalar == pytest.approx(
            float(elements.mean()), abs=1e-6 * max(1.0, abs(scalar)))
        weighted = elements * 3.0
        assert scalar != pytest.approx(float(weighted.mean()), abs=1e-9)


# ---------------------------------------------------------------------------
# decode / rotation conventions
# ---------------------------------------------------------------------------


def test_rot6d_uses_matrix_columns_not_rows():
    rng = np.random.default_rng(3)
    R, r6 = m._random_rotation_features(rng, (8,))
    recovered, ok = m.independent_rot6d_to_mat(r6)
    assert ok.all()
    assert np.abs(recovered - R).max() < 1e-12
    rows = np.concatenate([R[:, 0, :], R[:, 1, :]], axis=1)
    wrong, _ = m.independent_rot6d_to_mat(rows)
    assert np.abs(wrong - R).max() > 1e-2


def test_quaternion_order_wxyz_vs_xyzw():
    rng = np.random.default_rng(4)
    R, _ = m._random_rotation_features(rng, (1,))
    quat = m.mat_to_quat_wxyz_np(R)
    assert np.abs(m.quat_to_mat_wxyz_np(quat) - R).max() < 1e-12
    xyzw = np.roll(quat, -1, axis=-1)
    wrong = m.quat_to_mat_wxyz_np(xyzw)
    assert np.abs(wrong - R).max() > 1e-3


def test_decode_domain_and_clamp():
    lo, hi = m.synthetic_joint_limits()
    std = np.array([0.31, 0.27, 0.19])
    x0 = np.zeros((1, 38))
    x0[0, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    x0[0, 9:38] = 0.0
    normal = m.independent_decode(x0, np.zeros(3), std, lo, hi)
    assert normal["ok"].all()
    boundary = x0.copy()
    boundary[0, 9] = 1.0
    _, dom = m.decode_jacobian_np(boundary, std, lo, hi)
    assert not dom["ok"][0]
    assert dom["clamp_boundary"][0, 0]
    outside = x0.copy()
    outside[0, 9] = 1.2
    D, dom = m.decode_jacobian_np(outside, std, lo, hi)
    assert dom["ok"][0]
    assert D[0, 6, 9] == 0.0
    degenerate = x0.copy()
    degenerate[0, 3:6] = 0.0
    _, dom = m.decode_jacobian_np(degenerate, std, lo, hi)
    assert not dom["ok"][0]
    assert not dom["ok_6d"][0]


def test_random_rotation_positive_domain_stays_away_from_pi():
    rng = np.random.default_rng(5)
    R, _ = m._random_rotation_features(rng, (32,))
    quat = m.mat_to_quat_wxyz_np(R)
    assert np.abs(quat[:, 0]).min() > np.cos((math.pi - 0.25) / 2.0) - 1e-12


# ---------------------------------------------------------------------------
# Jacobian / gradient counterexamples
# ---------------------------------------------------------------------------


def test_wrong_rotation_frame_breaks_j35():
    rng = np.random.default_rng(6)
    chain = m.build_synthetic_chain()
    pos = rng.normal(0, 0.2, (1, 3))
    R, _ = m._random_rotation_features(rng, (1,))
    q = rng.uniform(-0.6, 0.6, (1, 29))
    _, J_nv = m.synthetic_fk_jac_np(chain, pos, R, q)
    fd = []
    h = 1e-6
    for k in range(35):
        pp, Rp, qp = m._tangent_perturbation(pos[0], R[0], q[0], k, h)
        pm, Rm, qm = m._tangent_perturbation(pos[0], R[0], q[0], k, -h)
        fp = m.synthetic_fk_np(chain, pp[None], Rp[None], qp[None])[0]
        fm = m.synthetic_fk_np(chain, pm[None], Rm[None], qm[None])[0]
        fd.append((fp - fm) / (2 * h))
    fd = np.stack(fd, axis=-1)
    Rt = np.swapaxes(R, 1, 2)
    correct = np.array(J_nv, copy=True)
    correct[..., 3:6] = np.einsum("nsck,nkj->nscj", J_nv[..., 3:6], Rt)
    assert np.abs(fd - correct).max() < 1e-8
    missing_rt = np.array(J_nv, copy=True)
    assert np.abs(fd - missing_rt).max() > 1e-3


def test_wrong_joint_pivot_breaks_j35():
    rng = np.random.default_rng(7)
    chain = m.build_synthetic_chain()
    pos = rng.normal(0, 0.2, (1, 3))
    R, _ = m._random_rotation_features(rng, (1,))
    q = rng.uniform(-0.6, 0.6, (1, 29))
    Rs, ps = m._chain_forward_np(chain, pos, R, q)
    kp, J = m.synthetic_fk_jac_np(chain, pos, R, q)
    k = 5
    h = 1e-6
    qp = q.copy(); qm = q.copy()
    qp[0, k] += h; qm[0, k] -= h
    fp = m.synthetic_fk_np(chain, pos, R, qp)[0]
    fm = m.synthetic_fk_np(chain, pos, R, qm)[0]
    fd = (fp - fm) / (2 * h)
    assert np.abs(fd - J[0, :, :, 6 + k]).max() < 1e-8
    omega = Rs[k][0] @ chain.axes[k]
    wrong = np.cross(omega, kp[0] - ps[k][0])
    assert np.abs(fd - wrong).max() > 1e-3


def test_wrong_joint_scale_breaks_j38():
    std = np.array([0.31, 0.27, 0.19])
    lo, hi = m.synthetic_joint_limits()
    rng = np.random.default_rng(8)
    R, r6 = m._random_rotation_features(rng, (1,))
    x0 = np.zeros((1, 38))
    x0[0, 3:9] = r6
    x0[0, 9:38] = rng.uniform(-0.7, 0.7, 29)
    D, _ = m.decode_jacobian_np(x0, std, lo, hi)
    J = rng.normal(size=(15, 3, 35))
    J38 = np.einsum("sck,kj->scj", J, D[0])
    wrong = np.array(D, copy=True)
    wrong[0, 6:, 9:] *= 2.0
    J38_wrong = np.einsum("sck,kj->scj", J, wrong[0])
    assert np.abs(J38 - J38_wrong).max() > 1e-9


def test_gradient_group_gate_rejects_zero_and_clamped():
    ref = np.linspace(-1e-3, 1e-3, 38)
    zero = m.gradient_group_stats(np.zeros(38), ref)
    assert zero["problems"]
    got = ref.copy()
    clamped = np.zeros(29, dtype=bool)
    clamped[0] = True
    got[9] = 1e-3
    bad = m.gradient_group_stats(got, ref, clamped)
    assert "clamped_nonzero" in bad["problems"]
    good = m.gradient_group_stats(ref.copy(), ref)
    assert not good["problems"]


def test_loss_and_gradient_follow_j38():
    rng = np.random.default_rng(9)
    J = rng.normal(size=(1, 15, 3, 38))
    pred = rng.normal(size=(1, 15, 3))
    target = pred + 0.01 * rng.normal(size=(1, 15, 3))
    out = m.fk_loss_and_gradient(J, pred, target)
    assert out["denominator"] == 15.0
    assert out["grad_weighted"] == pytest.approx(2 * out["grad_raw"])
    assert np.abs(out["grad_weighted"]).max() > 0


# ---------------------------------------------------------------------------
# evidence independence / tamper detection
# ---------------------------------------------------------------------------


def test_key_multiset_detects_missing_extra_duplicate():
    keys = m.required_geometry_rows()
    assert not m._key_multiset_problems(
        [{"key": k} for k in keys], keys)
    assert m._key_multiset_problems([{"key": k} for k in keys[:-1]], keys)
    assert m._key_multiset_problems(
        [{"key": k} for k in keys] + [{"key": "extra"}], keys)
    assert m._key_multiset_problems(
        [{"key": k} for k in keys] + [{"key": keys[0]}], keys)


def test_array_compare_detects_tamper():
    saved = {
        "prod": np.ones((2, 3), dtype=np.float32),
        "ref": np.ones((2, 3), dtype=np.float64),
        "mask": np.ones((2,), dtype=bool),
    }
    fresh = {k: v.copy() for k, v in saved.items()}
    assert not m._array_compare_problems(
        saved, fresh, bitwise_keys=("prod",))
    tampered = copy.deepcopy(fresh)
    tampered["prod"][0, 0] += 1e-4
    assert m._array_compare_problems(
        saved, tampered, bitwise_keys=("prod",))
    tampered = copy.deepcopy(fresh)
    tampered["ref"][0, 0] += 1e-3
    assert m._array_compare_problems(
        saved, tampered, bitwise_keys=("prod",))
    tampered = copy.deepcopy(fresh)
    tampered["ref"][0, 0] = np.nan
    assert m._array_compare_problems(
        saved, tampered, bitwise_keys=("prod",))
    tampered = copy.deepcopy(fresh)
    del tampered["mask"]
    assert m._array_compare_problems(
        saved, tampered, bitwise_keys=("prod",))


def test_row_compare_detects_field_changes(synthetic):
    row = next(r for r in synthetic["geometry"]["rows"]
               if r["key"].startswith("fk_scalar|deterministic|W0"))
    assert not m._compare_row_fields(row, copy.deepcopy(row), row["key"])
    for field, value in (
        ("status", "mismatch"),
        ("lfk_denominator", 901),
        ("production_lfk", row["production_lfk"] + 1e-6),
    ):
        other = copy.deepcopy(row)
        other[field] = value
        assert m._compare_row_fields(other, row, row["key"]), field
    other = copy.deepcopy(row)
    del other["lfk_denominator"]
    assert m._compare_row_fields(other, row, row["key"])
    batch = next(r for r in synthetic["geometry"]["rows"]
                 if r["key"] == "fk_scalar|deterministic|batch")
    assert not m._compare_row_fields(batch, copy.deepcopy(batch), batch["key"])
    other = copy.deepcopy(batch)
    other["batch_equals_window_mean"] = False
    assert m._compare_row_fields(other, batch, batch["key"])


def test_production_geometry_mutation_is_detected(synthetic):
    row = next(r for r in synthetic["geometry"]["rows"]
               if r["key"] == "production_fk|pred_det|W0")
    arrays = synthetic["geometry"]["arrays"]
    prod = arrays["prod_fk_pred_det"][0].astype(np.float64)
    ref = arrays["ref_fk_pred_det"][0]
    tol = m.tolerance_record(synthetic["tolerance"], "production_fk")
    ok = m.compare_values(prod, ref, tol)
    assert ok["status"] == "pass"
    bad = prod.copy()
    bad[0, 0, 0] += 0.01
    assert m.compare_values(bad, ref, tol)["status"] == "mismatch"
    transposed = np.swapaxes(prod, 1, 2)
    assert m.compare_values(transposed, ref, tol)["status"] == "mismatch"


def test_parent_binding_requires_bitwise(synthetic):
    arrays = synthetic["geometry"]["arrays"]
    parent = arrays["parent_fk_pred_det"]
    assert m._bitwise_equal(arrays["prod_fk_pred_det"], parent)
    shifted = parent.copy()
    shifted[0, 0, 0, 0] = np.nextafter(shifted[0, 0, 0, 0], np.float32(1.0))
    assert not m._bitwise_equal(arrays["prod_fk_pred_det"], shifted)


def test_calibration_contract_and_margins():
    contract = m.synthetic_calibration(seed=20261028)
    assert contract["mode"] == "synthetic_calibrated"
    assert contract["evidence_sha256"]
    assert not m.tolerance_contract_problems(contract)
    observed = contract["calibration"]["observed"]
    assert set(observed) == set(m.TOLERANCE_PROPOSALS)
    for key, rec in observed.items():
        assert 4.0 * rec["max_ratio_of_proposal"] <= 1.0, key
    excluded = contract["calibration"]["excluded_cases"]
    assert excluded["near_pi"]["included_in_tolerance"] is False
    assert excluded["clamp_boundary"]["derivative_ok"] is False
    assert excluded["degenerate_6d"]["derivative_ok"] is False
    assert excluded["clamp_outside"]["derivative_value"] == 0.0
    assert excluded["zero_residual"]["fk_m2"] == 0.0


def test_tolerance_widening_is_rejected():
    contract = m.synthetic_calibration(seed=20261028)
    widened = copy.deepcopy(contract)
    widened["tolerances"]["production_fk"]["atol"] *= 2.0
    assert m.tolerance_contract_problems(widened)
    default = m.build_default_tolerance_contract()
    assert not m.tolerance_contract_problems(default)
    default["tolerances"]["decode_float32"]["rtol"] = 1e-3
    assert m.tolerance_contract_problems(default)


def test_calibration_evidence_tamper_is_rejected():
    contract = m.synthetic_calibration(seed=20261028)
    tampered = copy.deepcopy(contract)
    tampered["calibration"]["observed"]["production_fk"][
        "max_ratio_of_proposal"] = 0.0
    assert m.tolerance_contract_problems(tampered)
    # widening observed + rehashing still fails the 4x proposal rule
    tampered = copy.deepcopy(contract)
    tampered["calibration"]["observed"]["production_fk"][
        "max_ratio_of_proposal"] = 0.5
    tampered["calibration_sha256"] = m._canonical_sha256(
        tampered["calibration"])
    tampered["evidence_sha256"] = tampered["calibration_sha256"]
    assert m.tolerance_contract_problems(tampered, recompute=False)


def test_calibration_is_deterministic_and_evidence_free_of_real_reads():
    a = m.synthetic_calibration(seed=20261028)
    b = m.synthetic_calibration(seed=20261028)
    assert a["calibration_sha256"] == b["calibration_sha256"]
    assert a["calibration"]["real_asset_loaded"] is False
    assert a["calibration"]["real_checkpoint_loaded"] is False
    assert a["calibration"]["real_dataset_loaded"] is False
    assert a["calibration"]["real_model_calls"] == 0


# ---------------------------------------------------------------------------
# accounting / seal
# ---------------------------------------------------------------------------


def test_negative_and_non_finite_costs_are_rejected(tmp_path):
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.upsert_cost(tmp_path, "G", "attempt_001", value, {}, "running")


def test_publishing_row_is_not_consumable(tmp_path):
    (m.stage_dir(tmp_path, "G") / "attempt_001").mkdir(parents=True)
    row = {
        "status": "publishing", "complete": False,
        "measured_wall_s": 1.0, "wall_s": 1.0,
        "accounting_schema": m.ACCOUNTING_SCHEMA,
        "cohort_id": m.COHORT_ID,
    }
    m.save_index(tmp_path, [{"stage": "G", "attempt": "attempt_001", **row}])
    m.save_ledger(tmp_path, [{"stage": "G", "attempt": "attempt_001", **row}])
    state, problems = m.stage_state(tmp_path, "G")
    assert state is None
    assert "stage_not_terminal" in problems


def test_stage_history_unknown_on_half_damaged_ledger(tmp_path):
    (m.stage_dir(tmp_path, "G") / "attempt_001").mkdir(parents=True)
    row = {
        "status": "done", "complete": True,
        "measured_wall_s": 2.0, "wall_s": 2.0,
        "accounting_schema": m.ACCOUNTING_SCHEMA,
        "cohort_id": m.COHORT_ID,
    }
    m.save_index(tmp_path, [{"stage": "G", "attempt": "attempt_001", **row}])
    (tmp_path / "costs.json").write_text("{not json")
    assert m.stage_history(tmp_path, "G") is None
    assert m.total_history(tmp_path) is None


def test_stage_history_rejects_negative_history(tmp_path):
    (m.stage_dir(tmp_path, "G") / "attempt_001").mkdir(parents=True)
    row = {
        "status": "done", "complete": True,
        "measured_wall_s": -5.0, "wall_s": -5.0,
        "accounting_schema": m.ACCOUNTING_SCHEMA,
        "cohort_id": m.COHORT_ID,
    }
    m.save_index(tmp_path, [{"stage": "G", "attempt": "attempt_001", **row}])
    m.save_ledger(tmp_path, [{"stage": "G", "attempt": "attempt_001", **row}])
    assert m.stage_history(tmp_path, "G") is None


def test_cohort_attribution_foreign_legacy_and_member(tmp_path):
    fixed = tmp_path / "runs"
    member = fixed / "member"
    foreign = fixed / "foreign"
    legacy = fixed / "legacy"
    for p in (member, foreign, legacy):
        (m.stage_dir(p, "G") / "attempt_001").mkdir(parents=True)
    m.write_cohort(member, mode="synthetic")
    row = {
        "status": "done", "complete": True,
        "measured_wall_s": 3.0, "wall_s": 3.0,
        "accounting_schema": m.ACCOUNTING_SCHEMA,
        "cohort_id": m.COHORT_ID,
        "call_balance": m.CallCounter(m.CALL_CAPS).snapshot(),
    }
    m.save_index(member, [{"stage": "G", "attempt": "attempt_001", **row}])
    m.save_ledger(member, [{"stage": "G", "attempt": "attempt_001", **row}])
    m.write_json(foreign / "cohort.json", {
        "cohort_id": "another_cohort", "execution_id": "another",
        "accounting_schema": m.ACCOUNTING_SCHEMA})
    (legacy / "costs.json").write_text("{broken")
    history = m.cohort_history(fixed)
    assert history["known"] is True
    assert history["total_s"] == pytest.approx(3.0)
    assert len(history["runs"]) == 1
    assert history["runs"][0]["run"] == m.path_for_record(member)


def test_attempt_limit_blocks_before_registration(tmp_path):
    budget = {"attempts": 2, "attempts_used": 2, "remaining_s": 100.0,
              "reserve_s": 10.0}
    with pytest.raises(SystemExit):
        m.pre_register_attempt(tmp_path, "G", "attempt_003", 0.0,
                               scope={}, budget=budget)


def test_finish_attempt_seals_consumable_row(tmp_path):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    m.write_cohort(tmp_path, mode="synthetic")
    budget = {"remaining_s": 100.0, "reserve_s": 10.0}
    t0 = m.time.perf_counter()
    attempt = "attempt_001"
    m.pre_register_attempt(tmp_path, "G", attempt, t0, scope={},
                           budget={**budget, "attempts": 2, "attempts_used": 0})
    seal = m.finish_attempt(
        tmp_path, "G", attempt, t0, budget,
        {"complete": True, "report": None, "summary": None},
        status="done", extra={"call_balance": None})
    assert seal.sealed and seal.terminal
    row, problems = m.stage_state(tmp_path, "G")
    assert row is not None and row["complete"] is True
    assert problems == []


def test_budget_for_isolated_synthetic_root(tmp_path):
    m.write_cohort(tmp_path, mode="synthetic")
    budget = m.budget_for(tmp_path, "R0", fixed_root=tmp_path, mode="synthetic")
    assert budget["remaining_s"] == pytest.approx(m.STAGE_BUDGETS["R0"]["budget_s"])


# ---------------------------------------------------------------------------
# authorization / admission
# ---------------------------------------------------------------------------


def test_authorization_template_is_refused():
    template = m.authorization_template()
    assert "not_authorized" in m.authorization_problems(template)
    record = copy.deepcopy(template)
    record["authorized"] = True
    record["checklist"] = {k: True for k in m.AUTHORIZATION_REQUIRED}
    for stage, spec in m.STAGE_BUDGETS.items():
        record["budgets"][stage] = {"budget_s": spec["budget_s"],
                                    "attempts": spec["attempts"]}
    record["draft_sha256"] = "0" * 64
    record["tolerance_evidence_sha256"] = "1" * 64
    record["admission_package_sha256"] = "2" * 64
    assert not m.authorization_problems(record)
    # strict zero declarations: 0.0 / False / "0" are refused
    for value in (0.0, False, "0", None):
        bad = copy.deepcopy(record)
        bad["training_updates"] = value
        assert m.authorization_problems(bad)
    # exact budgets
    for value in (179.9, "180", True, None):
        bad = copy.deepcopy(record)
        bad["budgets"]["R0"]["budget_s"] = value
        assert m.authorization_problems(bad)
    bad = copy.deepcopy(record)
    bad["budgets"]["R0"]["attempts"] = 3
    assert m.authorization_problems(bad)
    bad = copy.deepcopy(record)
    bad["budgets"].pop("D")
    assert m.authorization_problems(bad)
    bad = copy.deepcopy(record)
    bad["state_fd"]["rows"] = 4671
    assert m.authorization_problems(bad)


def test_admission_package_problems():
    assert m.admission_package_problems(None)
    package = {
        "schema": m.ADMISSION_SCHEMA,
        "passed": False,
        "problems": ["x"],
        "status": {"authorized": True, "registered": True, "run": "run"},
        "read_scope": {"window_values_loaded": True,
                       "checkpoint_torch_load": True,
                       "cohort_created": True,
                       "registered_attempt_created": True,
                       "conditioner_forward_calls": 1,
                       "model_forward_calls": 1,
                       "vjp_calls": 1},
        "parent_anchors": {},
        "key_counts": {"geometry": 1, "fd": 2, "states": 3},
        "implementation": {"files": {}},
        "tolerance": {"mode": "synthetic_default"},
    }
    problems = m.admission_package_problems(package)
    for expected in ("admission_not_passed", "admission_has_problems",
                     "admission_must_be_pre_authorization",
                     "admission_must_not_be_registered",
                     "admission_must_not_be_run", "admission_key_counts",
                     "admission_tolerance_not_calibrated",
                     "admission_runtime_contract"):
        assert expected in problems
    applied = copy.deepcopy(package)
    applied["runtime"] = {"requirements": dict(m.RUNTIME_REQUIREMENTS),
                          "applied": False,
                          "observed_after_setup": dict(m.RUNTIME_REQUIREMENTS)}
    assert "admission_runtime_not_applied" in \
        m.admission_package_problems(applied)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_w1_real_fk.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=120)


def test_real_entry_refuses_without_authorization(tmp_path):
    out = tmp_path / "run"
    proc = _run_cli("--real", "--out", str(out))
    assert proc.returncode != 0
    assert "authorization" in (proc.stdout + proc.stderr).lower()
    assert not out.exists()


def test_synthetic_entry_guard_rejects_experiment_roots():
    forbidden = (m.PROJECT_ROOT / "derived/experiments/omniretarget_first/"
                 "fk_guard_probe")
    proc = _run_cli("--synthetic", "--out", str(forbidden))
    assert proc.returncode != 0
    assert not forbidden.exists()
    proc = _run_cli("--synthetic", "--out",
                    str(m.PROJECT_ROOT / "src/ireset/eval/fk_guard_probe"))
    assert proc.returncode != 0


# ---------------------------------------------------------------------------
# end-to-end and tamper resistance
# ---------------------------------------------------------------------------


def test_synthetic_self_test_end_to_end(self_test):
    assert self_test["conclusion"] == "synthetic_contract_passed"
    assert self_test["complete_evidence"] is True
    assert self_test["covered_contract_passed"] is True
    assert not self_test["problems"]
    assert not self_test["gaps"]
    assert self_test["boundaries"]["full_objective_contract_passed"] is None
    assert self_test["boundaries"]["final_interaction_method"] == "not_ready"
    assert self_test["boundaries"]["production_integration"] is False
    calls = self_test["calls"]["counts"]
    zeros = self_test["zero_counters"]
    for key in m.ZERO_COUNTERS:
        assert zeros[key] == 0, key
    assert calls["scene_setup_forward"] == 0  # synthetic: no MuJoCo compile
    for stage in ("R0", "G", "D", "F"):
        assert self_test["seals"][stage]["terminal"] is True
    # R0 now records the input schema/constants/domain facts it used to defer
    registration = self_test["stages"]["R0"]["registration"]
    assert registration["inputs"]["problems"] == []
    assert registration["parent_consumer_state"]["checked"] is False
    assert registration["locked_assets"]["checked"] is False
    checks = self_test["stages"]["R0"]["admission"]["checks"]
    assert checks["input_schema"] is True
    # F recomputed in a separate process and recorded its identity
    process = self_test["recompute_process"]
    assert process["start_method"] == "spawned_python_subprocess"
    assert process["exit_reason"] == "ok"
    assert process["pid"] != m.os.getpid()
    assert process["ppid"] == m.os.getpid()
    assert any(unit.get("source") == "worker" for unit in self_test["units"])
    assert all(unit.get("admitted") for unit in self_test["units"])
    # G and D count their real calls; only F's worker delta used to survive.
    for stage in ("G", "D", "F"):
        delta = self_test["stages"][stage]["call_delta"]
        nominal = m.STAGE_NOMINAL[stage]
        for kind in ("loss_assembler", "production_fk_batch",
                     "production_vjp", "mj_forward", "mj_jac_site",
                     "fd_pose_eval"):
            assert delta.get(kind, 0) == nominal[kind], (stage, kind)
    # The index and the ledger carry the same terminal call snapshot.
    run_root = Path(self_test["run_root"])
    if not run_root.is_absolute():
        run_root = m.PROJECT_ROOT / run_root
    for stage in ("R0", "G", "D", "F"):
        index = [r for r in m.load_index(run_root) if r["stage"] == stage]
        ledger = [r for r in m.load_ledger(run_root) if r["stage"] == stage]
        assert len(index) == len(ledger) == 1
        assert index[0]["call_balance"] == ledger[0]["call_balance"], stage
    assert self_test["calls"]["counts"]["loss_assembler"] == 4
    # F verifies G/D artifacts against the sealed per-file hashes
    assert self_test["prerequisite_consumer_states"] == {"G": [], "D": []}
    verification = json.loads(Path(self_test["stages"]["F"]["verification_path"])
                              .read_text())
    assert verification["artifact_hashes"]["g_expected"]["arrays"]
    assert verification["artifact_hashes"]["d_expected"]["arrays"]


def test_synthetic_self_test_is_asset_free(self_test):
    assert self_test["stages"]["R0"]["registration"]["parent"]["mode"] == \
        "synthetic_fixture"
    assert not self_test["stages"]["R0"]["registration"]["environment"][
        "note"].startswith("real")


def test_synthetic_does_not_call_largebox_static_frame(monkeypatch, tmp_path):
    def boom():
        raise AssertionError("real asset was loaded")

    monkeypatch.setattr("ireset.train.losses.largebox_static_frame", boom)
    provider = m.SyntheticFkProvider(seed=20261028)
    provider.prepare_fixture(tmp_path)
    inputs = provider.load_inputs()
    run = provider.production_capture(
        "deterministic", inputs.prediction["deterministic"],
        want_gradient=False)
    assert run.fk_m2 >= 0.0


def test_finalize_rejects_tampered_saved_artifacts(self_test, tmp_path):
    source = Path(self_test["run_root"])
    if not source.is_absolute():
        source = m.PROJECT_ROOT / source
    target = tmp_path / "tampered"
    shutil.copytree(source, target)
    g = copy.deepcopy(self_test["stages"]["G"])
    d = copy.deepcopy(self_test["stages"]["D"])
    for key in ("arrays", "gradient_arrays", "rows"):
        g["paths"][key] = str(target / Path(g["paths"][key]).relative_to(source))
    for key in ("arrays", "rows"):
        d["paths"][key] = str(target / Path(d["paths"][key]).relative_to(source))
    rows_path = Path(g["paths"]["rows"])
    payload = json.loads(rows_path.read_text())
    payload["rows"][0]["status"] = "mismatch"
    rows_path.write_text(json.dumps(payload))
    npz_path = Path(g["paths"]["arrays"])
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["prod_fk_pred_det"] = arrays["prod_fk_pred_det"].copy()
    arrays["prod_fk_pred_det"][0, 0, 0, 0] += np.float32(0.01)
    np.savez(npz_path, **arrays)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    provider = m.SyntheticFkProvider(seed=20261028)
    provider.fixture_path = source / "fixture" / "synthetic_parent_arrays.npz"
    provider.fixture_sha256 = m.sha256_file(provider.fixture_path)
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process", g_expected=g["artifact_hashes"])
    assert f["conclusion"] == "audit_incomplete"
    assert f["verification_problems"]
    texts = " ".join(f["verification_problems"])
    assert "array_not_bitwise" in texts or "status" in texts or \
        "sha_mismatch" in texts


def _registered_f_attempt(root: Path, registration_sha: str,
                          *, remaining_s: float = 600.0,
                          reserve_s: float = 90.0,
                          attempt: str = "attempt_999"):
    """Pre-register an F attempt so unit admission has a live parent row."""
    m.write_cohort(root, mode="synthetic")
    budget = {"remaining_s": remaining_s, "reserve_s": reserve_s,
              "attempts": 2, "attempts_used": 0}
    t0 = m.time.perf_counter()
    m.pre_register_attempt(root, "F", attempt, t0, scope={}, budget=budget,
                           registration_sha=registration_sha)
    return m.CallCounter(m.CALL_CAPS), t0, budget


# ---------------------------------------------------------------------------
# withdrawal cost / slow persistence
# ---------------------------------------------------------------------------


def _sealed_f_row(tmp_path):
    m.write_cohort(tmp_path, mode="synthetic")
    adir = m.stage_dir(tmp_path, "F") / "attempt_001"
    adir.mkdir(parents=True)
    report = adir / "fk_audit_final.json"
    report.write_text(json.dumps({
        "schema": "w1_real_fk_audit_summary_v1",
        "conclusion": "synthetic_contract_passed",
        "complete_evidence": True,
        "covered_contract_passed": True,
        "failures": [], "gaps": [],
    }))
    row = {
        "status": "done", "complete": True, "measured_wall_s": 1.0,
        "wall_s": 1.0, "accounting_schema": m.ACCOUNTING_SCHEMA,
        "cohort_id": m.COHORT_ID, "report": str(report),
        "summary": str(report),
    }
    m.save_index(tmp_path, [{"stage": "F", "attempt": "attempt_001", **row}])
    m.save_ledger(tmp_path, [{"stage": "F", "attempt": "attempt_001", **row}])
    return report, row


def test_withdraw_attempt_claim_rewrites_and_charges(tmp_path):
    report, row = _sealed_f_row(tmp_path)
    updated, problems, summary = m.withdraw_attempt_claim(
        tmp_path, "F", "attempt_001", report, row, ["revoked_for_test"],
        reason="f_seal_or_consumer_not_consumable",
        t0=m.time.perf_counter() - 0.05,
        budget={"remaining_s": 100.0, "reserve_s": 90.0})
    assert not problems
    assert updated["status"] == "incomplete"
    assert updated["complete"] is False
    assert updated["measured_wall_s"] >= 1.0
    on_disk = json.loads(report.read_text())
    assert on_disk["conclusion"] == "audit_incomplete"
    assert on_disk["complete_evidence"] is False
    assert on_disk["covered_contract_passed"] is False
    assert "revoked_for_test" in on_disk["gaps"]
    state, state_problems = m.stage_state(tmp_path, "F")
    assert state is not None and state["complete"] is False
    assert "attempt_not_complete" in state_problems


def test_slow_withdraw_cost_is_not_published_as_success(tmp_path, monkeypatch):
    report, row = _sealed_f_row(tmp_path)
    real_upsert = m.upsert_index

    def slow_upsert(*args, **kwargs):
        m.time.sleep(0.03)
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(m, "upsert_index", slow_upsert)
    monkeypatch.setattr(m, "SEAL_TAIL_RESERVE_S", 0.001)
    updated, problems, _ = m.withdraw_attempt_claim(
        tmp_path, "F", "attempt_001", report, row, [],
        reason="f_seal_or_consumer_not_consumable",
        t0=m.time.perf_counter(),
        budget={"remaining_s": 0.02, "reserve_s": 0.0})
    assert "withdraw_cost_unverifiable" in problems
    assert updated["status"] == "publishing"
    assert updated["complete"] is False
    state, state_problems = m.stage_state(tmp_path, "F")
    assert state is None
    assert state_problems


def test_slow_seal_is_not_published_as_success(tmp_path, monkeypatch):
    m.write_cohort(tmp_path, mode="synthetic")
    t0 = m.time.perf_counter()
    budget = {"remaining_s": 0.5, "reserve_s": 0.0,
              "attempts": 2, "attempts_used": 0}
    m.pre_register_attempt(tmp_path, "G", "attempt_001", t0, scope={},
                           budget=budget)
    real_upsert = m.upsert_index

    def slow_upsert(*args, **kwargs):
        m.time.sleep(0.03)
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(m, "upsert_index", slow_upsert)
    monkeypatch.setattr(m, "SEAL_TAIL_RESERVE_S", 0.001)
    seal = m.finish_attempt(
        tmp_path, "G", "attempt_001", t0,
        {"remaining_s": 0.02, "reserve_s": 0.0},
        {"complete": True, "report": None, "summary": None},
        status="done", extra={"call_balance": None})
    assert not seal.sealed
    assert seal.row["complete"] is False
    assert seal.row["status"] in ("publishing", "incomplete")
    if seal.row["status"] == "publishing":
        state, problems = m.stage_state(tmp_path, "G")
        assert state is None and problems


def test_implementation_drift_helper():
    current = {"files": {"a": "1", "b": "2"}}
    assert not m._implementation_drift(current, current)
    drift = m._implementation_drift(
        current, {"files": {"a": "x", "b": "2"}})
    assert drift == ["implementation_drift:a"]
    assert m._implementation_drift(current, None) == [
        "r0_expected_implementation_missing"]


# ---------------------------------------------------------------------------
# review round: device / live scalar / call counting / stage gates / F process
# ---------------------------------------------------------------------------


def test_capture_binds_tensors_to_the_assembler_device(monkeypatch, tmp_path):
    provider = m.SyntheticFkProvider(seed=20261028)
    provider.prepare_fixture(tmp_path)
    inputs = provider.load_inputs()
    inner = provider._assembler_obj(inputs)
    recorded = {}

    class DeclaredCudaAssembler:
        device = "cuda:0"

        def __init__(self, inner):
            self.inner = inner

        def __call__(self, pred, batch, observer=None):
            recorded["pred"] = str(pred.device)
            recorded["x0"] = str(batch["x0"].device)
            return self.inner(pred.to("cpu"),
                              {"x0": batch["x0"].to("cpu")},
                              observer=observer)

    real_tensor = torch.tensor

    def spy(data, *args, **kwargs):
        recorded["tensor_device"] = kwargs.get("device")
        kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        return real_tensor(data, *args, **kwargs)

    monkeypatch.setattr(m.torch, "tensor", spy)
    run = m._capture_run(DeclaredCudaAssembler(inner), inputs.x0,
                         inputs.prediction["deterministic"], None,
                         want_gradient=False)
    # The leaves must be created on the assembler device; a hard-coded CPU
    # leaf is exactly the CUDA device conflict this counterexample pins.
    assert recorded["tensor_device"] == "cuda:0"
    assert run.fk_m2 >= 0.0


def test_shifted_live_scalar_fails_the_batch_row(synthetic, monkeypatch):
    provider = m.SyntheticFkProvider(seed=20261028)
    provider.fixture_path = Path(synthetic["provider"].fixture_path)
    provider.fixture_sha256 = synthetic["provider"].fixture_sha256
    inputs = provider.load_inputs()
    original = m.SyntheticFkProvider.production_capture

    def shifted(self, model, pred, *, want_gradient=True):
        from dataclasses import replace
        run = original(self, model, pred, want_gradient=want_gradient)
        return replace(run, fk_m2=run.fk_m2 + 5e-7)

    monkeypatch.setattr(m.SyntheticFkProvider, "production_capture", shifted)
    counter = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter)
    geometry = m.compute_geometry(provider, inputs, synthetic["tolerance"])
    batch = next(r for r in geometry["rows"]
                 if r["key"] == "fk_scalar|deterministic|batch")
    assert batch["status"] == "mismatch"
    assert "live_scalar_differs_from_elements" in batch["problems"] or \
        "element_tolerance" in batch["problems"]
    windows = [r for r in geometry["rows"]
               if r["key"].startswith("fk_scalar|deterministic|W")]
    assert all(r["status"] == "pass" for r in windows)


def test_production_fk_calls_are_counted_at_the_call_site(monkeypatch,
                                                          tmp_path):
    provider = m.SyntheticFkProvider(seed=1)
    provider.prepare_fixture(tmp_path)
    inputs = provider.load_inputs()
    counter = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter)

    class RaisingAssembler:
        device = "cpu"

        def __call__(self, *args, **kwargs):
            raise RuntimeError("no FK call happened")

    # A preset increment would show 2 even though the assembler never ran FK.
    with pytest.raises(RuntimeError):
        m._capture_run(RaisingAssembler(), inputs.x0,
                       inputs.prediction["deterministic"], counter,
                       want_gradient=False)
    assert counter.counts["production_fk_batch"] == 0
    assert counter.counts["loss_assembler"] == 1

    # A successful capture counts exactly the FK evaluations that ran.
    calls = {"n": 0}
    original = m._SyntheticFkAdapter.site_world_positions

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(m._SyntheticFkAdapter, "site_world_positions", counting)
    counter_success = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter_success)
    provider.production_capture("deterministic",
                                inputs.prediction["deterministic"],
                                want_gradient=False)
    assert calls["n"] == 2
    assert counter_success.counts["production_fk_batch"] == calls["n"]


def test_budget_revoked_g_stops_the_chain(monkeypatch, tmp_path):
    def revoked_g(root, stage, attempt, *args, **kwargs):
        adir = m.stage_dir(root, stage) / attempt
        adir.mkdir(parents=True, exist_ok=True)
        rows = adir / "geometry_rows.json"
        rows.write_text("{}")
        return {"passed": False, "problems": [],
                "failures": [],
                "gaps": ["budget_exceeded_after_registration"],
                "paths": {"rows": str(rows)}, "artifact_hashes": {},
                "call_delta": {}}

    monkeypatch.setattr(m, "run_g_stage", revoked_g)
    result = m.synthetic_self_test(tmp_path / "run", seed=20261028)
    assert result["conclusion"] == "audit_incomplete"
    assert "D" not in result["stages"]
    assert "F" not in result["stages"]
    assert any("attempt_not_complete" in p for p in result["gaps"])
    assert result["calls"]["counts"]["loss_assembler"] == 0
    assert not (tmp_path / "run" / "derivative").exists()


def _f_inputs(self_test, tmp_path, *, tamper=None):
    source = Path(self_test["run_root"])
    if not source.is_absolute():
        source = m.PROJECT_ROOT / source
    target = tmp_path / "run"
    shutil.copytree(source, target)
    g = copy.deepcopy(self_test["stages"]["G"])
    d = copy.deepcopy(self_test["stages"]["D"])
    for key in ("arrays", "gradient_arrays", "rows"):
        g["paths"][key] = str(target / Path(g["paths"][key]).relative_to(source))
    for key in ("arrays", "rows"):
        d["paths"][key] = str(target / Path(d["paths"][key]).relative_to(source))
    if tamper is not None:
        tamper(g, d)
    provider = m.SyntheticFkProvider(seed=20261028)
    provider.fixture_path = target / "fixture" / "synthetic_parent_arrays.npz"
    provider.fixture_sha256 = m.sha256_file(provider.fixture_path)
    return target, g, d, provider


def test_f_refuses_an_unregistered_parent_attempt(self_test, tmp_path):
    target, g, d, provider = _f_inputs(self_test, tmp_path)
    counter = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", m.time.perf_counter(),
        {"remaining_s": 600.0, "reserve_s": 90.0},
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process")
    assert f["conclusion"] == "audit_incomplete"
    assert any("f_unit_not_admitted:artifact_read" in p
               for p in f["verification_problems"])
    assert not any(counter.counts.values())


def test_f_deadline_blocks_before_backend(self_test, tmp_path):
    target, g, d, provider = _f_inputs(self_test, tmp_path)
    counter, t0, _ = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0,
        {"remaining_s": 1.0, "reserve_s": 5.0},
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process")
    assert f["conclusion"] == "audit_incomplete"
    assert any("unit_deadline" in p for p in f["verification_problems"])
    assert not f["recompute_process"]
    assert not any(counter.counts.values())


def test_f_verifies_gradient_npz_hash_separately(self_test, tmp_path):
    def tamper(g, d):
        path = Path(g["paths"]["gradient_arrays"])
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["grad_production_raw_deterministic"] = (
            arrays["grad_production_raw_deterministic"].copy())
        arrays["grad_production_raw_deterministic"][0, 0, 0] += np.float32(1e-3)
        np.savez(path, **arrays)

    target, g, d, provider = _f_inputs(self_test, tmp_path, tamper=tamper)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process", g_expected=g["artifact_hashes"])
    assert "g_gradient_arrays_sha_mismatch" in f["verification_problems"]
    assert f["conclusion"] == "audit_incomplete"


def test_f_detects_overlap_conflict_between_geometry_and_gradient_npz(
        self_test, tmp_path):
    def tamper(g, d):
        path = Path(g["paths"]["arrays"])
        with np.load(path) as z:
            saved = {k: z[k] for k in z.files}
        saved["grad_production_raw_deterministic"] = (
            saved["grad_production_raw_deterministic"].copy())
        saved["grad_production_raw_deterministic"][0, 0, 0] += np.float32(1e-3)
        np.savez(path, **saved)
        g["artifact_hashes"]["arrays"] = m.sha256_file(path)

    target, g, d, provider = _f_inputs(self_test, tmp_path, tamper=tamper)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process", g_expected=g["artifact_hashes"])
    assert "g_overlap_conflict:grad_production_raw_deterministic" in \
        f["verification_problems"]
    assert f["conclusion"] == "audit_incomplete"


def test_malformed_fd_rows_become_gaps_not_exceptions(self_test, tmp_path):
    def tamper(g, d):
        Path(d["paths"]["rows"]).write_text("[]")

    target, g, d, provider = _f_inputs(self_test, tmp_path, tamper=tamper)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process", g_expected=g["artifact_hashes"],
        d_expected=d["artifact_hashes"])
    assert "d_rows_not_a_mapping" in f["verification_problems"]
    assert f["conclusion"] == "audit_incomplete"


def test_negative_clamp_boundary_is_a_domain_gap(tmp_path):
    provider = m.SyntheticFkProvider(seed=20261028)
    provider._fixture["prediction"]["deterministic"][0, 0, 9] = -1.0
    provider.prepare_fixture(tmp_path)
    inputs = provider.load_inputs()
    tolerance = m.build_default_tolerance_contract()
    counter = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter)
    geometry = m.compute_geometry(provider, inputs, tolerance)
    grad_row = next(r for r in geometry["grad_rows"]
                    if r["key"] == "fk_gradient|deterministic|raw|W0")
    assert grad_row["status"] == "domain_gap"
    assert "gradient_domain_gap" in grad_row["problems"]
    counter_d = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter_d)
    fd = m.compute_fd(provider, inputs, geometry["arrays"], tolerance)
    rows = [r for r in fd["rows"]
            if r.get("family") == "normalized38"
            and r.get("model") == "deterministic"
            and r.get("window_id") == "W0"
            and r.get("local_frame") == 0 and r.get("dim") == 9]
    assert rows and all(r["status"] == "domain_gap" for r in rows)
    assert not fd["failures"]


def test_asset_closure_hashes_every_reference(tmp_path):
    assert m._asset_closure_facts(tmp_path, mode="synthetic")["checked"] is False
    project = tmp_path / "project"
    robots = project / "derived/robots"
    assets = project / "resources/robots/g1_holosoma/assets"
    robots.mkdir(parents=True)
    assets.mkdir(parents=True)
    (assets / "a.obj").write_text("o a")
    (robots / "assets").mkdir()
    (robots / "assets/a.urdf.obj").write_text("o u")
    xml = ('<mujoco><compiler meshdir="../../resources/robots/g1_holosoma/assets"/>'
           '<asset><mesh file="a.obj"/></asset></mujoco>')
    (robots / "g1_29dof_locked.xml").write_text(xml)
    (robots / "g1_29dof_locked_w_largebox.xml").write_text(xml)
    (robots / "g1_29dof_locked.urdf").write_text(
        '<robot><link><visual><geometry>'
        '<mesh filename="assets/a.urdf.obj"/></geometry></visual></link></robot>')
    (robots / "lock_record.json").write_text("{}")
    facts = m._asset_closure_facts(project, mode="real")
    assert facts["checked"] is True
    assert not facts["problems"]
    assert facts["reference_count"] == 2
    (assets / "a.obj").unlink()
    broken = m._asset_closure_facts(project, mode="real")
    assert any("asset_reference_missing" in p for p in broken["problems"])


def test_parent_consumer_state_requires_complete_rows(tmp_path):
    project = tmp_path / "project"
    base = project / m.PARENT_RUN
    rows = []
    for stage, dirname in m._PARENT_STAGE_DIRS.items():
        (base / dirname / "attempt_001").mkdir(parents=True, exist_ok=True)
        rows.append({"stage": stage, "attempt": "attempt_001",
                     "status": "done", "complete": True})
    m.save_index(base, rows)
    m.save_ledger(base, rows)
    summary = project / m.PARENT_FILES["audit_summary"][0]
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({
        "conclusion": "real_batch_covered_contract_passed",
        "complete_evidence": True}))
    state = m._parent_consumer_state(project, mode="real")
    assert state["checked"] is True
    assert not state["problems"]
    assert m._parent_consumer_state(project, mode="synthetic")["checked"] is False
    rows[0]["complete"] = False
    m.save_index(base, rows)
    broken = m._parent_consumer_state(project, mode="real")
    assert any("parent_stage_R0" in p for p in broken["problems"])


def test_input_schema_facts_reject_bad_shapes(synthetic):
    inputs = synthetic["inputs"]
    assert not m._input_schema_facts(inputs)["problems"]
    bad = m.FkInputs(**{**inputs.__dict__, "x0": inputs.x0[:, :30]})
    assert any("input_shape:x0" in p
               for p in m._input_schema_facts(bad)["problems"])
    bad_limits = m.FkInputs(**{**inputs.__dict__,
                              "joint_hi": inputs.joint_lo.copy()})
    assert any("joint_limits_not_ordered" in p
               for p in m._input_schema_facts(bad_limits)["problems"])


def test_withdraw_publishes_before_any_terminal_claim(tmp_path, monkeypatch):
    report, row = _sealed_f_row(tmp_path)
    statuses = []
    real_index = m.upsert_index

    def record(root, stage, attempt, entry):
        statuses.append(entry.get("status"))
        return real_index(root, stage, attempt, entry)

    monkeypatch.setattr(m, "upsert_index", record)
    t0 = m.time.perf_counter()
    updated, problems, _ = m.withdraw_attempt_claim(
        tmp_path, "F", "attempt_001", report, row, [],
        reason="f_seal_or_consumer_not_consumable",
        t0=t0, budget={"remaining_s": 100.0, "reserve_s": 90.0})
    assert statuses and statuses[0] == "publishing"
    assert updated["status"] == "incomplete"
    assert updated["measured_wall_s"] >= m.time.perf_counter() - t0
    assert not problems


def test_withdraw_stays_publishing_when_terminal_write_outruns_budget(
        tmp_path, monkeypatch):
    report, row = _sealed_f_row(tmp_path)
    clock = {"t": 100.0}
    real_index = m.upsert_index

    def fake_clock():
        return clock["t"]

    def jumpy_upsert(root, stage, attempt, entry):
        result = real_index(root, stage, attempt, entry)
        clock["t"] += 10.0 if entry.get("status") == "incomplete" else 0.01
        return result

    monkeypatch.setattr(m.time, "perf_counter", fake_clock)
    monkeypatch.setattr(m, "upsert_index", jumpy_upsert)
    updated, problems, _ = m.withdraw_attempt_claim(
        tmp_path, "F", "attempt_001", report, row, [],
        reason="f_seal_or_consumer_not_consumable",
        t0=clock["t"], budget={"remaining_s": 5.0, "reserve_s": 0.0})
    assert "withdraw_cost_unverifiable" in problems
    assert updated["status"] == "publishing"
    assert updated["complete"] is False
    state, state_problems = m.stage_state(tmp_path, "F")
    assert state is None and state_problems


# ---------------------------------------------------------------------------
# call accounting: chain-level counts, seal snapshots, reservations
# ---------------------------------------------------------------------------


def _registered_stage_attempt(root: Path, stage: str,
                              registration_sha: str = "0" * 64,
                              *, remaining_s: float = 600.0,
                              reserve_s: float = 90.0) -> tuple[dict, float]:
    m.write_cohort(root, mode="synthetic")
    budget = {"remaining_s": remaining_s, "reserve_s": reserve_s,
              "attempts": 2, "attempts_used": 0}
    t0 = m.time.perf_counter()
    m.pre_register_attempt(root, stage, "attempt_001", t0, scope={},
                           budget=budget, registration_sha=registration_sha)
    return budget, t0


def test_seal_writes_the_same_call_snapshot_to_index_and_ledger(tmp_path):
    budget, t0 = _registered_stage_attempt(tmp_path, "G")
    counter = m.CallCounter(m.CALL_CAPS)
    counter.add("mj_forward", 7)
    seal = m.finish_attempt(
        tmp_path, "G", "attempt_001", t0, budget,
        {"complete": True, "report": None, "summary": None,
         "registration_sha256": "0" * 64},
        status="done", extra={"call_balance": counter.snapshot()})
    assert seal.sealed
    index = m.load_index(tmp_path)[0]
    ledger = m.load_ledger(tmp_path)[0]
    assert index["call_balance"] == ledger["call_balance"]
    assert index["call_balance"]["counts"]["mj_forward"] == 7
    _row, problems = m.stage_state(tmp_path, "G")
    assert not problems

    tampered = dict(index)
    tampered["call_balance"] = m.CallCounter(m.CALL_CAPS).snapshot()
    m.save_index(tmp_path, [tampered])
    _row, problems = m.stage_state(tmp_path, "G")
    assert "call_balance_disagreement" in problems
    assert m.stage_history(tmp_path, "G") is None


def test_prior_calls_unknown_when_a_reservation_is_unsettled(tmp_path):
    budget, t0 = _registered_stage_attempt(tmp_path, "G")
    counter = m.CallCounter(m.CALL_CAPS)
    counter.add("mj_forward", 3)
    m.finish_attempt(
        tmp_path, "G", "attempt_001", t0, budget,
        {"complete": True, "report": None, "summary": None,
         "registration_sha256": "0" * 64},
        status="done", extra={"call_balance": counter.snapshot()})
    assert m.prior_call_counts(tmp_path, fixed_root=tmp_path,
                               mode="synthetic") is not None
    path = m._reserve_unit_calls(tmp_path, "G", "attempt_001", {
        "unit": {"units": 1, "reserved": {"mj_forward": 5}}})
    assert m.prior_call_counts(tmp_path, fixed_root=tmp_path,
                               mode="synthetic") is None
    _row, problems = m.stage_state(tmp_path, "G")
    assert any("reservation_unsettled" in p for p in problems)
    m._complete_unit_calls(path, {"mj_forward": 3})
    assert m.prior_call_counts(tmp_path, fixed_root=tmp_path,
                               mode="synthetic") is not None


def _worker_spec(root: Path, fixture: Path, *, registration_sha: str,
                 reserve_s: float = 0.0) -> dict:
    return {
        "schema": "w1_real_fk_recompute_v1",
        "mode": "synthetic", "root": str(root), "stage": "F",
        "attempt": "attempt_001",
        "registration_sha256": registration_sha,
        "deadline_perf": m.time.perf_counter() + 600.0,
        "deadline_epoch": m.time.time() + 600.0,
        "reserve_s": reserve_s,
        "project_root": str(m.PROJECT_ROOT),
        "device": "cpu", "seed": 20261028,
        "fixture_path": str(fixture),
        "fixture_sha256": m.sha256_file(fixture),
        "tolerance": m.build_default_tolerance_contract(),
    }


def test_worker_settles_known_calls_when_fd_aborts(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "fixture"
    provider = m.SyntheticFkProvider(seed=20261028)
    provider.prepare_fixture(fixture_dir)
    fixture = fixture_dir / "synthetic_parent_arrays.npz"
    root = tmp_path / "run"
    budget, _t0 = _registered_stage_attempt(root, "F",
                                           registration_sha="a" * 64)
    out = tmp_path / "worker"
    out.mkdir()
    spec_path = out / "spec.json"
    spec_path.write_text(json.dumps(_worker_spec(
        root, fixture, registration_sha="a" * 64)))

    def boom(*_args, **_kwargs):
        raise RuntimeError("fd_exploded")

    monkeypatch.setattr(m, "compute_fd", boom)
    code = m._recompute_worker_main(str(spec_path), str(out))
    meta = json.loads((out / "worker_meta.json").read_text())
    assert code == 3
    assert meta["exit_reason"].startswith("worker_exception")
    # Geometry already ran; those calls must not be reported as zero.
    assert meta["call_delta"]["loss_assembler"] == 2
    assert meta["call_delta"]["production_fk_batch"] == 4
    assert meta["call_delta"]["production_vjp"] == 2
    assert meta["call_delta"]["mj_forward"] == 1200
    assert meta["call_delta"]["mj_jac_site"] == 7200
    assert meta["call_delta"]["fd_pose_eval"] == 0


def test_f_merges_an_aborted_worker_delta_and_keeps_unsettled_unknown(
        self_test, tmp_path, monkeypatch):
    target, g, d, provider = _f_inputs(self_test, tmp_path, tamper=lambda *_: None)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])

    def fake_worker(spec, out_dir, timeout_s, project_root):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        full = {kind: 0 for kind in m.CALL_CAPS}
        full["mj_forward"] = 1200
        full["loss_assembler"] = 2
        meta = {"pid": 999999, "ppid": m.os.getpid(),
                "exit_reason": "worker_exception:RuntimeError",
                "units": [], "call_delta": dict(full),
                "counts": {"schema": "w1_real_fk_call_budget_v1",
                           "counts": dict(full), "within_caps": True}}
        (out_dir / "worker_meta.json").write_text(json.dumps(meta))
        return meta, ["f_worker_exit:3"]

    monkeypatch.setattr(m, "_run_recompute_worker", fake_worker)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="subprocess", g_expected=g["artifact_hashes"],
        d_expected=d["artifact_hashes"])
    assert f["conclusion"] == "audit_incomplete"
    assert counter.counts["mj_forward"] == 1200
    assert counter.counts["loss_assembler"] == 2
    reservation = json.loads(
        (m.stage_dir(target, "F") / "attempt_999" /
         "call_reservations.json").read_text())
    assert reservation["consumed"]["mj_forward"] == 1200

    # A worker that reports no delta leaves the reservation unsettled.
    counter2, t0b, budget2 = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"],
        attempt="attempt_1000")

    def fake_worker_no_delta(spec, out_dir, timeout_s, project_root):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"pid": 999998, "ppid": m.os.getpid(),
                "exit_reason": "worker_exception:RuntimeError", "units": []}
        (out_dir / "worker_meta.json").write_text(json.dumps(meta))
        return meta, ["f_worker_exit:3"]

    monkeypatch.setattr(m, "_run_recompute_worker", fake_worker_no_delta)
    f2 = m.run_f_stage(
        target, "F", "attempt_1000", t0b, budget2,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter2, self_test["tolerance_contract"],
        recompute="subprocess", g_expected=g["artifact_hashes"],
        d_expected=d["artifact_hashes"])
    assert "f_call_settlement_unknown" in f2["verification_problems"]
    assert "f_reservation_left_unsettled" in f2["verification_problems"]
    reservation2 = json.loads(
        (m.stage_dir(target, "F") / "attempt_1000" /
         "call_reservations.json").read_text())
    assert reservation2["consumed"] is None


# ---------------------------------------------------------------------------
# per-unit deadlines inside the compute loops
# ---------------------------------------------------------------------------


def test_worker_call_delta_requires_a_complete_consistent_snapshot():
    full = {kind: 0 for kind in m.CALL_CAPS}
    full["mj_forward"] = 11
    delta, problems = m._worker_call_delta(
        {"call_delta": dict(full), "counts": {"counts": dict(full)}})
    assert delta == full and not problems
    # An empty or partial delta must not settle the reservation at zero.
    assert m._worker_call_delta(
        {"call_delta": {}, "counts": {"counts": dict(full)}})[0] is None
    partial = dict(full)
    partial.pop("fd_pose_eval")
    assert m._worker_call_delta(
        {"call_delta": partial, "counts": {"counts": partial}})[0] is None
    # The delta must agree with the worker's own terminal counter snapshot.
    other = dict(full)
    other["mj_forward"] = 12
    assert "f_worker_call_delta_inconsistent" in m._worker_call_delta(
        {"call_delta": dict(full), "counts": {"counts": other}})[1]
    assert "f_worker_call_counts_missing" in m._worker_call_delta(
        {"call_delta": dict(full)})[1]
    bad = dict(full)
    bad["mj_forward"] = True
    assert m._worker_call_delta(
        {"call_delta": bad, "counts": {"counts": bad}})[0] is None


def test_f_does_not_settle_an_unmergeable_worker_delta(
        self_test, tmp_path, monkeypatch):
    target, g, d, provider = _f_inputs(self_test, tmp_path,
                                       tamper=lambda *_: None)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    counter.add("mj_forward", 31632)

    def fake_worker(spec, out_dir, timeout_s, project_root):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        full = {kind: 0 for kind in m.CALL_CAPS}
        full["mj_forward"] = int(m.CALL_CAPS["mj_forward"]) - 31632 + 1
        meta = {"pid": 999997, "ppid": m.os.getpid(),
                "exit_reason": "worker_exception:RuntimeError",
                "units": [], "call_delta": dict(full),
                "counts": {"counts": dict(full)}}
        (out_dir / "worker_meta.json").write_text(json.dumps(meta))
        return meta, ["f_worker_exit:3"]

    monkeypatch.setattr(m, "_run_recompute_worker", fake_worker)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="subprocess", g_expected=g["artifact_hashes"],
        d_expected=d["artifact_hashes"])
    assert f["conclusion"] == "audit_incomplete"
    assert "f_worker_call_cap:mj_forward" in f["verification_problems"]
    assert "f_call_settlement_unknown" in f["verification_problems"]
    # The failed merge does not move the counter and the reservation stays
    # unsettled so later attempts cannot claim a fresh quota.
    assert counter.counts["mj_forward"] == 31632
    reservation = json.loads(
        (m.stage_dir(target, "F") / "attempt_999" /
         "call_reservations.json").read_text())
    assert reservation["consumed"] is None
    assert any("reservation_unsettled:F:attempt_999" in p
               for p in m._reservation_problems(target))


def test_call_matrix_mismatch_withdraws_the_disk_claim(monkeypatch, tmp_path):
    real = m.full_chain_calls

    def inflated():
        calls = real()
        calls["kernel64_fk_batch"] += m.STAGE_NOMINAL["G"][
            "kernel64_fk_batch"]
        return calls

    monkeypatch.setattr(m, "full_chain_calls", inflated)
    result = m.synthetic_self_test(tmp_path / "run", seed=20261028)
    assert result["conclusion"] == "audit_incomplete"
    assert result["complete_evidence"] is False
    assert result["covered_contract_passed"] is False
    assert any("synthetic_call_matrix:kernel64_fk_batch" in p
               for p in result["problems"])
    run_root = Path(result["run_root"])
    if not run_root.is_absolute():
        run_root = m.PROJECT_ROOT / run_root
    summary = json.loads(
        (run_root / "finalize/attempt_001/fk_audit_final.json").read_text())
    assert summary["conclusion"] == "audit_incomplete"
    assert summary["complete_evidence"] is False
    assert summary["covered_contract_passed"] is False
    verification = json.loads(
        (run_root / "finalize/attempt_001/"
         "independent_verification.json").read_text())
    assert verification["complete_evidence"] is False
    _row, problems = m.stage_state(run_root, "F")
    assert "attempt_not_complete" in problems
    assert m.stage_history(run_root, "F") is not None  # terminal, honest


def test_geometry_deadline_stops_the_second_model(synthetic, monkeypatch):
    provider = synthetic["provider"]
    counter = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter)
    ticks = iter(range(0, 10 ** 6))
    monkeypatch.setattr(m, "_CLOCK", lambda: float(next(ticks)))
    # Calls: start(0), 3x reference, 3x kernel, capture-det(7),
    # jac-det(8), capture-diff(9) -> expired at 9.
    with pytest.raises(TimeoutError):
        m.compute_geometry(provider, synthetic["inputs"],
                           synthetic["tolerance"], deadline=9.0)
    assert counter.counts["loss_assembler"] == 1
    assert counter.counts["production_fk_batch"] == 2
    assert counter.counts["production_vjp"] == 1


def test_fd_deadline_stops_before_the_next_state(synthetic, monkeypatch):
    provider = synthetic["provider"]
    counter = m.CallCounter(m.CALL_CAPS)
    provider.bind_counter(counter)
    ticks = iter(range(0, 10 ** 7))
    monkeypatch.setattr(m, "_CLOCK", lambda: float(next(ticks)))
    with pytest.raises(TimeoutError):
        m.compute_fd(provider, synthetic["inputs"],
                     synthetic["geometry"]["arrays"],
                     synthetic["tolerance"], deadline=3.0)
    per_state = 2 * len(m.FD_STEPS) * (m.TANGENT_DIM + m.X0_DIM)
    assert counter.counts["fd_pose_eval"] == 2 * per_state
    assert counter.counts["fd_pose_eval"] < m.FD_POSE_EVALUATIONS


# ---------------------------------------------------------------------------
# F evidence: gradient rows and per-file NPZ schemas
# ---------------------------------------------------------------------------


def test_f_binds_saved_gradient_rows_to_the_main_table(self_test, tmp_path):
    def tamper(g, d):
        path = Path(g["paths"]["rows"])
        payload = json.loads(path.read_text())
        row = payload["gradient_rows"][0]
        row["max_abs_error"] = 123.0
        path.write_text(json.dumps(payload))
        g["artifact_hashes"]["rows"] = m.sha256_file(path)

    target, g, d, provider = _f_inputs(self_test, tmp_path, tamper=tamper)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process", g_expected=g["artifact_hashes"],
        d_expected=d["artifact_hashes"])
    assert f["conclusion"] == "audit_incomplete"
    texts = " ".join(f["verification_problems"])
    assert "gradient_vs_main_table" in texts or "max_abs_error" in texts


def test_f_validates_each_npz_schema_before_merging(self_test, tmp_path):
    def tamper(g, d):
        path = Path(g["paths"]["gradient_arrays"])
        np.savez(path)  # emptied but hash rebound below
        g["artifact_hashes"]["gradient_arrays"] = m.sha256_file(path)

    target, g, d, provider = _f_inputs(self_test, tmp_path, tamper=tamper)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process", g_expected=g["artifact_hashes"],
        d_expected=d["artifact_hashes"])
    assert f["conclusion"] == "audit_incomplete"
    assert any(p.startswith("g_gradient_arrays_schema_missing:")
               for p in f["verification_problems"])


def test_f_missing_npz_reports_a_gap_instead_of_raising(self_test, tmp_path):
    def tamper(g, d):
        Path(g["paths"]["arrays"]).unlink()

    target, g, d, provider = _f_inputs(self_test, tmp_path, tamper=tamper)
    counter, t0, budget = _registered_f_attempt(
        target, self_test["stages"]["R0"]["registration_sha256"])
    provider.bind_counter(counter)
    f = m.run_f_stage(
        target, "F", "attempt_999", t0, budget,
        self_test["stages"]["R0"]["registration_sha256"],
        g, d, provider, counter, self_test["tolerance_contract"],
        recompute="in_process", g_expected=g["artifact_hashes"],
        d_expected=d["artifact_hashes"])
    assert f["conclusion"] == "audit_incomplete"
    texts = " ".join(f["verification_problems"])
    assert "g_arrays_missing" in texts or "g_arrays_file_missing" in texts


# ---------------------------------------------------------------------------
# frozen production runtime (TF32 off)
# ---------------------------------------------------------------------------


def test_production_runtime_is_applied_and_verified():
    original = (torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision())
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        assert m.runtime_facts_problems(m.environment_facts())
        runtime = m.configure_production_runtime()
        assert runtime["applied"] is True, runtime["problems"]
        for key, want in m.RUNTIME_REQUIREMENTS.items():
            assert runtime["after"][key] == want, key
        assert not m.runtime_facts_problems(runtime["after"])
        # Requirements cannot be silently rewritten.
        assert m.runtime_facts_problems({"cudnn_allow_tf32": True})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original[0]
        torch.backends.cudnn.allow_tf32 = original[1]
        torch.set_float32_matmul_precision(original[2])


def test_r0_charges_launch_and_import_time(tmp_path, monkeypatch):
    original = m.run_r0

    def stop_after_r0(*args, **kwargs):
        result = original(*args, **kwargs)
        result["passed"] = False
        result["problems"].append("synthetic_stop_after_r0")
        return result

    monkeypatch.setattr(m, "run_r0", stop_after_r0)
    started = m.time.perf_counter() - 5.0
    root = tmp_path / "run"
    result = m.run_registered_audit(
        root, m.SyntheticFkProvider(seed=20261028), mode="synthetic",
        tolerance=m.build_default_tolerance_contract(),
        r0_started_perf=started)
    assert result["conclusion"] == "audit_incomplete"
    row = m.load_index(root)[0]
    assert row["stage"] == "R0"
    assert row["budget"]["startup_elapsed_s"] >= 5.0
    assert row["measured_wall_s"] >= m.time.perf_counter() - started
    assert m.stage_history(root, "R0") >= 5.0


def test_expired_r0_launch_stops_before_loading_inputs(tmp_path, monkeypatch):
    provider = m.SyntheticFkProvider(seed=20261028)

    def forbidden():
        raise AssertionError("expired R0 must not read input arrays")

    monkeypatch.setattr(provider, "load_inputs", forbidden)
    root = tmp_path / "run"
    result = m.run_registered_audit(
        root, provider, mode="synthetic",
        tolerance=m.build_default_tolerance_contract(),
        r0_started_perf=m.time.perf_counter() - 181.0)
    assert result["conclusion"] == "audit_incomplete"
    assert set(result["stages"]) == {"R0"}
    assert "input_load_failed:TimeoutError" in result["problems"]
    assert all(value == 0 for value in result["calls"]["counts"].values())
    assert m.stage_history(root, "R0") >= 181.0


@pytest.mark.parametrize("started", [float("nan"), float("inf"), True, -1.0])
def test_invalid_r0_launch_clock_creates_no_run(tmp_path, started):
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="invalid R0 launch clock"):
        m.run_registered_audit(
            root, m.SyntheticFkProvider(seed=20261028), mode="synthetic",
            tolerance=m.build_default_tolerance_contract(),
            r0_started_perf=started)
    assert not root.exists()
