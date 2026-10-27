"""Synthetic tests for the W1 real-objective audit harness.

No test touches the real dataset, cache, checkpoints or the fixed production
history root.  The full-pipeline fixture uses the analytic four-window
provider and tiny production-code stand-ins; tamper tests run on copies.
"""

from __future__ import annotations

import copy
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ireset.eval.w1_real_objective_audit as audit
import ireset.train.losses as losses_mod
from ireset.diffusion.generator import DiffusionG1
from ireset.eval.training_objective_audit import (
    W1_SYNTHETIC_WEIGHTS,
    build_synthetic_assembler,
    build_synthetic_world,
    make_synthetic_batch,
)


@pytest.fixture(scope="module")
def audit_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("w1_real_objective_audit")
    result = audit.synthetic_self_test(out)
    assert result["conclusion"] == "synthetic_contract_passed", result
    return Path(result["run_root"])


def _load(path: Path):
    return json.loads(Path(path).read_text())


def _stage_files(run: Path, stage: str, attempt: str = "attempt_001") -> Path:
    return run / audit.STAGE_DIRS[stage] / attempt


def test_full_synthetic_pipeline_passes_and_artifacts(audit_dir: Path):
    summary = _load(_stage_files(audit_dir, "F") / "audit_summary.json")
    assert summary["conclusion"] == "synthetic_contract_passed"
    assert summary["complete_evidence"] is True
    assert summary["covered_contract_passed"] is True
    assert summary["problems"] == [] and summary["gaps"] == []
    assert summary["counts"]["windows"] == 4
    assert summary["counts"]["models"] == 2
    assert summary["counts"]["objective_rows"] == 98
    assert summary["counts"]["gradient_rows"] == 14
    assert summary["boundaries"]["objective_coverage"] == "partial"
    assert summary["boundaries"]["full_objective_contract_passed"] is None
    assert summary["boundaries"]["attribution"] == "attribution_insufficient"
    assert summary["boundaries"]["r3"] == "not_run"
    assert summary["boundaries"]["final_interaction_method"] == "not_ready"
    assert summary["boundaries"]["training_updates"] == 0
    assert summary["boundaries"]["parameter_fd"] == 0
    assert summary["boundaries"]["ddim_calls"] == 0
    assert summary["calls"]["counts"] == {
        "conditioner": 4, "model": 4,
        "production_vjp": 4, "independent_vjp": 4,
    }
    for stage in ("R0", "P", "F"):
        assert _stage_files(audit_dir, stage).is_dir()
        row, problems = audit.stage_state(audit_dir, stage)
        assert row is not None and problems == [], (stage, problems)
    costs = _load(audit_dir / "costs.json")["invocations"]
    assert {c["stage"] for c in costs} == {"R0", "P", "F"}
    assert all(c["status"] == "done" and c["complete"] is True for c in costs)
    assert all(c["measured_wall_s"] >= 0 for c in costs)
    index = _load(audit_dir / "attempts_index.json")["attempts"]
    assert {r["stage"] for r in index} == {"R0", "P", "F"}
    for row in index:
        assert row["accounting_schema"] == audit.ACCOUNTING_SCHEMA
        assert row["cohort_id"] == audit.COHORT_ID
        assert row["execution_registration_sha256"] is not None or row["stage"] == "R0"
    # Raw arrays and rows are present and exact-key checked.
    p = _stage_files(audit_dir, "P")
    arrays, problems = audit._check_arrays(
        p / "objective_real_arrays.npz", audit.objective_array_keys(),
        _load(_stage_files(audit_dir, "R0") / "registration.json")["array_schema"])
    assert problems == []
    grads, problems = audit._check_arrays(
        p / "gradient_real_arrays.npz", audit.gradient_array_keys(),
        audit._gradient_schema())
    assert problems == []
    assert set(arrays) == set(audit.objective_array_keys())
    assert set(grads) == set(audit.gradient_array_keys())
    rows = _load(p / "objective_real_rows.json")
    assert audit.row_multiset_problems(rows["rows"], rows["required"]) == []
    assert all(r["status"] == "within_tolerance" for r in rows["rows"])
    grad_rows = _load(p / "gradient_real_rows.json")
    assert audit.row_multiset_problems(grad_rows["rows"], grad_rows["required"]) == []
    assert all(r["status"] == "within_tolerance" for r in grad_rows["rows"])
    verification = _load(_stage_files(audit_dir, "F") / "independent_verification.json")
    assert verification["problems"] == [] and verification["gaps"] == []
    assert verification["row_multiset"]["objective"] == []
    assert verification["row_multiset"]["gradient"] == []


def test_no_real_asset_access(monkeypatch, tmp_path: Path):
    def boom(*_a, **_k):
        raise AssertionError("synthetic self-test touched a real asset")

    monkeypatch.setattr(losses_mod, "G1TorchFK", boom)
    monkeypatch.setattr(losses_mod, "largebox_static_frame", boom)
    result = audit.synthetic_self_test(tmp_path / "audit")
    assert result["conclusion"] == "synthetic_contract_passed", result


def test_observer_is_bitwise_transparent():
    world = build_synthetic_world()
    batch = make_synthetic_batch(world, dtype=torch.float64)
    asm = build_synthetic_assembler(world.tokenizer, W1_SYNTHETIC_WEIGHTS,
                                    torch.float64)
    pred = (torch.tensor(world.x0, dtype=torch.float64)[None] + 0.01).requires_grad_(True)
    loss_a, parts_a = asm(pred, batch)
    capture: dict = {}
    loss_b, parts_b = asm(pred, batch, observer=capture)
    assert torch.equal(loss_a, loss_b)
    assert parts_a == parts_b
    assert set(capture["term_tensors"]) == {
        "pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2", "total"}
    assert capture["contact_branch_executed"] is False
    w = W1_SYNTHETIC_WEIGHTS
    l5 = (w.w_pos * capture["term_tensors"]["pos"]
          + w.w_rot6d * capture["term_tensors"]["rot6d"]
          + w.w_joint * capture["term_tensors"]["joint"]
          + w.w_geodesic * capture["term_tensors"]["geodesic_rad2"]
          + w.w_vel * capture["term_tensors"]["vel"])
    assert torch.equal(capture["L5"], l5)
    assert torch.equal(capture["total"], loss_a)
    assert capture["x0_pred"] is pred
    grad = audit.production_vjp(capture, pred)
    assert grad.shape == (6, 1, world.T, 38)
    assert torch.isfinite(grad).all()


def test_contact_branch_flag_positive_weight():
    world = build_synthetic_world()
    batch = make_synthetic_batch(world, dtype=torch.float64)
    weights = audit.LossWeights(w_pos=1, w_rot6d=1, w_joint=1,
                                w_geodesic=0.5, w_fk=2, w_contact=0.5,
                                w_vel=0.5)
    asm = build_synthetic_assembler(world.tokenizer, weights, torch.float64)
    pred = (torch.tensor(world.x0, dtype=torch.float64)[None] + 0.01).requires_grad_(True)
    capture: dict = {}
    asm(pred, batch, observer=capture)
    assert capture["contact_branch_executed"] is True
    assert "contact_m2" in capture["term_tensors"]


def test_detached_prediction_is_rejected():
    provider = audit.SyntheticObjectiveProvider(seed=123)
    keys = provider.select_windows()
    inputs = provider.load_inputs(
        keys, noise=audit.generate_noise(audit.NOISE_SEED,
                                         (audit.WINDOW_COUNT, audit.WINDOW, 38)))
    bundles = provider.model_factory()
    asm = provider.loss_factory()("deterministic")

    class DetachingLoss:
        w = asm.w
        fk = asm.fk

        def __call__(self, pred, batch, observer=None):
            return asm(pred.detach(), batch, observer=observer)

    calls = audit.AuditCallBudget()
    with pytest.raises(ValueError, match="live production tensor"):
        audit.production_objective_unit(
            bundles["deterministic"], inputs, DetachingLoss(), calls,
            device="cpu", dtype=torch.float32,
            tolerance=audit.build_tolerance_contract()["tolerances"])


def test_diffusion_noise_matches_independent_formula():
    model = DiffusionG1(num_train_timesteps=1000, beta_schedule="scaled_linear",
                        beta_start=1e-4, beta_end=0.02, d_model=8,
                        n_layers=1, n_heads=2, ffn_mult=1, time_window=31)
    g = torch.Generator().manual_seed(17)
    x0 = torch.randn(4, 60, 38, generator=g)
    noise = torch.randn(4, 60, 38, generator=g)
    t = torch.tensor([0, 250, 500, 999], dtype=torch.int64)
    x_t = model.add_noise(x0, noise, t).detach().numpy()
    alpha = audit.independent_alpha_cumprod(1000, "scaled_linear", 1e-4, 0.02)
    expected = audit.independent_x_t(x0.numpy(), noise.numpy(), t.numpy(), alpha)
    assert np.allclose(x_t, expected, atol=1e-5, rtol=1e-4)
    # t=0 is still the first noise level, not the clean state.
    assert not np.allclose(x_t[0], x0.numpy()[0], atol=1e-3)


def test_near_pi_rotation_is_outside_independent_domain():
    B, F = 1, 60
    target = np.zeros((B, F, 38), dtype=np.float64)
    pred = target.copy()
    target[..., 3:9] = np.tile([1.0, 0, 0, 0, 1.0, 0], (B, F, 1))
    angle = np.pi - 1e-4
    # rotation about x: columns [1,0,0] and [0,cos,sin]
    pred[..., 3:9] = np.tile(
        [1.0, 0, 0, 0, np.cos(angle), np.sin(angle)], (B, F, 1))
    out = audit.independent_term_elements(
        torch.as_tensor(pred), torch.as_tensor(target),
        rotation_domain=audit.DEFAULT_TOLERANCES["rotation_domain"])
    assert bool(out["geodesic_domain_ok"].all()) is False
    assert torch.isfinite(out["geodesic_rad2"]).all()


def test_required_row_multisets_are_exact():
    keys = audit.SyntheticObjectiveProvider(seed=1).select_windows()
    required = audit.required_objective_row_keys(keys)
    assert len(required) == 98
    assert len(set(required)) == len(required)
    grad_required = audit.required_gradient_row_keys()
    assert len(grad_required) == 14
    assert len(set(grad_required)) == len(grad_required)
    assert audit.row_multiset_problems([], required) != []
    rows = [{"id": k} for k in required]
    assert audit.row_multiset_problems(rows, required) == []
    assert audit.row_multiset_problems(rows[:-1], required) != []
    assert audit.row_multiset_problems(rows + [rows[0]], required) != []
    assert audit.row_multiset_problems(rows + [{"id": "extra"}], required) != []


def test_comparators_reject_bool_nan_and_scale():
    tol = {"atol": 1e-6, "rtol": 1e-6, "scale_floor": 1e-8}
    assert audit.compare_scalar("x", True, 1.0, tol)["status"] == "incomplete"
    assert audit.compare_scalar("x", 1.0, float("nan"), tol)["status"] == "incomplete"
    assert audit.compare_scalar("x", 1.0, 1.0 + 1e-3, tol)["status"] == "mismatch"
    p = torch.zeros(6, 4, 60, 38)
    i = torch.ones(6, 4, 60, 38)
    assert audit.compare_gradient("g", p, i, tol)["status"] == "mismatch"
    assert audit.compare_gradient("g", p, torch.full_like(i, float("nan")),
                                  tol)["status"] == "incomplete"
    # Tiny values are judged against the scale floor, not swallowed by a
    # large absolute tolerance.
    small = torch.full((6, 4, 60, 38), 1e-5)
    assert audit.compare_gradient("g", small, torch.zeros_like(small),
                                  {"atol": 1e-8, "rtol": 1e-6,
                                   "scale_floor": 1e-8})["status"] == "mismatch"


def test_tolerance_contract_validation():
    contract = audit.build_tolerance_contract()
    assert audit.tolerance_contract_problems(contract) == []
    bad = json.loads(json.dumps(contract))
    del bad["tolerances"]["loss_float32"]
    assert audit.tolerance_contract_problems(bad) != []
    bad = json.loads(json.dumps(contract))
    bad["calibrated"] = False
    assert audit.tolerance_contract_problems(bad) != []
    bad = json.loads(json.dumps(contract))
    bad["tolerances"]["loss_float32"]["atol"] = float("nan")
    assert audit.tolerance_contract_problems(bad) != []
    assert audit.load_tolerance_contract(Path("/nonexistent/tolerance.json"))[2] != []


def test_authorization_contract_requires_every_checklist_item():
    record = {
        "authorized": True,
        "cohort_id": audit.COHORT_ID,
        "execution_id": audit.EXECUTION_ID,
        "training_updates": 0,
        "parameter_fd": 0,
        "ddim_calls": 0,
        "checklist": {k: True for k in audit.AUTHORIZATION_REQUIRED},
        "budgets": {k: {"budget_s": v["budget_s"], "attempts": v["attempts"]}
                    for k, v in audit.STAGE_BUDGETS.items()},
    }
    assert audit.authorization_problems(record) == []
    bad = json.loads(json.dumps(record))
    bad["checklist"]["tolerance_calibration"] = False
    problems = audit.authorization_problems(bad)
    assert any("tolerance_calibration" in p for p in problems)
    bad = json.loads(json.dumps(record))
    bad["authorized"] = False
    assert "not_authorized" in audit.authorization_problems(bad)
    assert audit.load_authorization(Path("/nonexistent/auth.json"))[2] != []


def _seal_stage(root: Path, stage: str, attempt: str, *,
                complete: bool = True, status: str = "done",
                remaining_s: float = 60.0):
    t0 = time.perf_counter()
    audit.write_cohort(root, mode="synthetic")
    budget = {"stage": stage, "remaining_s": remaining_s}
    audit.pre_register_attempt(root, stage, attempt, t0, scope={"s": 1},
                               budget=budget)
    return audit.finish_attempt(root, stage, attempt, t0, budget,
                                {"complete": complete}, status=status)


def test_accounting_sealed_history_and_attempt_limit(tmp_path: Path):
    root = tmp_path / "run"
    first = _seal_stage(root, "R0", "attempt_001")
    assert first.sealed and first.row["complete"] is True
    row, problems = audit.stage_state(root, "R0")
    assert row is not None and problems == []
    cost = audit.stage_history(root, "R0")
    assert cost is not None and cost >= 0
    second = _seal_stage(root, "R0", "attempt_002")
    assert second.sealed
    with pytest.raises(SystemExit, match="attempt limit"):
        audit.budget_for(root, "R0", fixed_root=tmp_path, mode="synthetic")


def test_running_and_publishing_are_not_consumable(tmp_path: Path):
    root = tmp_path / "run"
    audit.write_cohort(root, mode="synthetic")
    t0 = time.perf_counter()
    budget = {"stage": "R0", "remaining_s": 60.0}
    audit.pre_register_attempt(root, "R0", "attempt_001", t0,
                               scope={"s": 1}, budget=budget)
    assert audit.stage_history(root, "R0") is None
    _, problems = audit.stage_state(root, "R0")
    assert "attempt_running" in problems
    with pytest.raises(SystemExit, match="history unknown"):
        audit.budget_for(root, "R0", fixed_root=tmp_path, mode="synthetic")
    # A publishing row is non-consumable too.
    audit.upsert_index(root, "R0", "attempt_001",
                       {"status": "publishing", "complete": False,
                        "accounting_schema": audit.ACCOUNTING_SCHEMA,
                        "cohort_id": audit.COHORT_ID, "measured_wall_s": 0.1})
    audit.upsert_cost(root, "R0", "attempt_001", 0.1,
                      {"complete": False, "cohort_id": audit.COHORT_ID},
                      "publishing")
    _, problems = audit.stage_state(root, "R0")
    assert "attempt_final_check_pending" in problems


def test_three_way_history_mismatch_is_unknown(tmp_path: Path):
    root = tmp_path / "run"
    audit.write_cohort(root, mode="synthetic")
    (audit.stage_dir(root, "R0") / "attempt_001").mkdir(parents=True)
    assert audit.stage_history(root, "R0") is None
    audit.upsert_index(root, "R0", "attempt_001",
                       {"status": "done", "complete": True,
                        "accounting_schema": audit.ACCOUNTING_SCHEMA,
                        "cohort_id": audit.COHORT_ID,
                        "measured_wall_s": 1.0})
    assert audit.stage_history(root, "R0") is None


def test_budget_overrun_leaves_non_done_attempt(tmp_path: Path):
    root = tmp_path / "run"
    result = _seal_stage(root, "R0", "attempt_001", remaining_s=0.0)
    assert result.sealed is False
    assert result.row["status"] != "done"
    assert result.row["complete"] is False
    row, _ = audit.stage_state(root, "R0")
    assert row is not None and row["status"] != "done"
    assert row["complete"] is False


def test_seal_write_failure_does_not_publish_done(tmp_path: Path, monkeypatch):
    root = tmp_path / "run"
    audit.write_cohort(root, mode="synthetic")

    def fail_terminal(*args, **kwargs):
        status = args[-1] if args else kwargs.get("status")
        if status in ("done", "incomplete"):
            raise OSError("injected write failure")
        return original(*args, **kwargs)

    original = audit.upsert_cost
    monkeypatch.setattr(audit, "upsert_cost", fail_terminal)
    t0 = time.perf_counter()
    budget = {"stage": "R0", "remaining_s": 0.0}
    audit.pre_register_attempt(root, "R0", "attempt_001", t0,
                               scope={"s": 1}, budget=budget)
    result = audit.finish_attempt(root, "R0", "attempt_001", t0, budget,
                                  {"complete": True}, status="done")
    assert result.sealed is False
    assert result.row["status"] != "done"
    assert result.row["complete"] is False
    index = audit.load_index(root)
    assert index and index[0]["status"] != "done"


def test_unit_admission_requires_active_parent(tmp_path: Path):
    root = tmp_path / "run"
    audit.write_cohort(root, mode="synthetic")
    t0 = time.perf_counter()
    budget = {"stage": "P", "remaining_s": 60.0}
    scope = {"windows": ["a"], "models": ["deterministic"], "batch": [4, 60, 38]}
    audit.pre_register_attempt(root, "P", "attempt_001", t0, scope=scope,
                               budget=budget, registration_sha="r" * 64)
    out = audit.stage_dir(root, "P") / "attempt_001" / "units" / "deterministic"
    out.mkdir(parents=True)
    ok = audit.unit_admission(root, "P", "attempt_001", "deterministic",
                              scope=scope, registration_sha="r" * 64,
                              deadline=t0 + 30.0, output_dir=out)
    assert ok["passed"] is True, ok
    bad = audit.unit_admission(root, "P", "attempt_001", "deterministic",
                               scope={**scope, "models": ["diffusion"]},
                               registration_sha="r" * 64,
                               deadline=t0 + 30.0, output_dir=out)
    assert not bad["passed"]
    audit.finish_attempt(root, "P", "attempt_001", t0, budget,
                         {"complete": True}, status="done")
    after = audit.unit_admission(root, "P", "attempt_001", "deterministic",
                                 scope=scope, registration_sha="r" * 64,
                                 deadline=t0 + 30.0, output_dir=out)
    assert not after["passed"]
    assert "parent_not_running" in after["problems"]


def test_input_validation_rejects_encoding_drift():
    provider = audit.SyntheticObjectiveProvider(seed=5)
    keys = provider.select_windows()
    inputs = provider.load_inputs(
        keys, noise=audit.generate_noise(audit.NOISE_SEED,
                                         (audit.WINDOW_COUNT, audit.WINDOW, 38)))
    contract = audit.build_tolerance_contract()
    ok = audit.validate_inputs(
        inputs, expected_keys=keys,
        tolerance=contract["tolerances"]["encode_float32"])
    assert ok["passed"], ok
    inputs.x0[0, 0, 0] += 0.1
    bad = audit.validate_inputs(
        inputs, expected_keys=keys,
        tolerance=contract["tolerances"]["encode_float32"])
    assert not bad["passed"]
    assert "encoding_mismatch" in bad["problems"]


def test_array_checks_reject_missing_extra_shape_and_nonfinite(tmp_path: Path):
    keys = audit.objective_array_keys()
    schema = audit._array_schema()
    arrays = {k: np.zeros(schema[k]["shape"], dtype=schema[k]["dtype"])
              for k in keys}
    path = tmp_path / "arrays.npz"
    np.savez(path, **arrays)
    got, problems = audit._check_arrays(path, keys, schema)
    assert problems == [] and set(got) == set(keys)
    bad = dict(arrays)
    bad.pop(keys[0])
    np.savez(path, **bad)
    _, problems = audit._check_arrays(path, keys, schema)
    assert any("array_keys_missing" in p for p in problems)
    bad = dict(arrays)
    bad["extra"] = np.zeros(1)
    np.savez(path, **bad)
    _, problems = audit._check_arrays(path, keys, schema)
    assert any("array_keys_extra" in p for p in problems)
    bad = dict(arrays)
    bad["x0"] = np.zeros((2, 2), dtype=np.float32)
    np.savez(path, **bad)
    _, problems = audit._check_arrays(path, keys, schema)
    assert any("array_shape" in p for p in problems)
    bad = dict(arrays)
    bad["x0"] = np.full(schema["x0"]["shape"], np.nan, dtype=np.float32)
    np.savez(path, **bad)
    _, problems = audit._check_arrays(path, keys, schema)
    assert any("array_nonfinite" in p for p in problems)


def _replay_contexts(run_dir: Path):
    reg = _load(_stage_files(run_dir, "R0") / "registration.json")
    pdir = _stage_files(run_dir, "P")
    arrays = dict(np.load(pdir / "objective_real_arrays.npz", allow_pickle=False))
    gradients = dict(np.load(pdir / "gradient_real_arrays.npz", allow_pickle=False))
    inputs = audit._inputs_from_arrays(arrays, reg)
    r0 = {
        "registration": reg,
        "registration_sha256": audit.sha256_file(
            _stage_files(run_dir, "R0") / "registration.json"),
        "inputs": inputs,
    }
    p = {
        "paths": {
            "arrays": pdir / "objective_real_arrays.npz",
            "gradient_arrays": pdir / "gradient_real_arrays.npz",
            "rows": pdir / "objective_real_rows.json",
            "gradient_rows": pdir / "gradient_real_rows.json",
            "call_trace": pdir / "call_trace.json",
            "rng_facts": pdir / "rng_state_facts.json",
            "report": pdir / "p_report.json",
        },
        "rows": _load(pdir / "objective_real_rows.json")["rows"],
        "gradient_rows": _load(pdir / "gradient_real_rows.json")["rows"],
        "units": {},
    }
    for model in audit.MODELS:
        p["units"][model] = {
            "prediction": arrays[f"prediction_{model}"],
            "x_t": arrays.get("x_t_diffusion") if model == "diffusion" else None,
            "production_grad": gradients[f"production_grad_{model}"],
            "independent_grad": gradients[f"independent_grad_{model}"],
            "fk_elements": arrays[f"term_elements_{model}_fk"],
        }
    return r0, p, arrays, gradients


def _run_f_on(run: Path, provider, *, attempt: str = "attempt_extra_f",
              calls=None, remaining_s: float = 120.0):
    r0, p, arrays, gradients = _replay_contexts(run)
    t0 = time.perf_counter()
    budget = {"stage": "F", "remaining_s": remaining_s}
    scope = audit._unit_scope(r0["inputs"])
    audit.pre_register_attempt(
        run, "F", attempt, t0, scope=scope, budget=budget,
        registration_sha=r0["registration_sha256"])
    result = audit.run_f_stage(
        run, "F", attempt, t0, budget, r0, p, provider,
        calls if calls is not None else audit.AuditCallBudget(),
        r0["registration"]["tolerance_contract"],
        mode="synthetic", scope=scope)
    return result, r0, p, arrays, gradients


@pytest.mark.parametrize("tamper", ["prediction", "gradient", "x_t"])
def test_f_replay_detects_tampered_raw_artifacts(audit_dir: Path, tmp_path: Path,
                                                 tamper: str):
    run = tmp_path / f"tampered_{tamper}"
    shutil.copytree(audit_dir, run)
    pdir = _stage_files(run, "P")
    r0, p, arrays, gradients = _replay_contexts(run)
    if tamper == "prediction":
        arrays["prediction_deterministic"] = (
            arrays["prediction_deterministic"] + np.float32(0.1))
        p["units"]["deterministic"]["prediction"] = arrays["prediction_deterministic"]
    elif tamper == "gradient":
        gradients["production_grad_deterministic"] = np.zeros_like(
            gradients["production_grad_deterministic"])
        p["units"]["deterministic"]["production_grad"] = gradients[
            "production_grad_deterministic"]
    else:
        arrays["x_t_diffusion"] = np.zeros_like(arrays["x_t_diffusion"])
        p["units"]["diffusion"]["x_t"] = arrays["x_t_diffusion"]
    np.savez(pdir / "objective_real_arrays.npz", **arrays)
    np.savez(pdir / "gradient_real_arrays.npz", **gradients)
    provider = audit.SyntheticObjectiveProvider(seed=20261026)
    calls = audit.AuditCallBudget()
    t0 = time.perf_counter()
    budget = {"stage": "F", "remaining_s": 60.0}
    scope = audit._unit_scope(r0["inputs"])
    audit.pre_register_attempt(
        run, "F", "attempt_tamper", t0, scope=scope, budget=budget,
        registration_sha=r0["registration_sha256"])
    result = audit.run_f_stage(
        run, "F", "attempt_tamper", t0,
        budget, r0, p, provider, calls,
        r0["registration"]["tolerance_contract"], mode="synthetic", scope=scope)
    assert result["problems"], result
    if tamper == "prediction":
        assert any("replay_prediction_not_bitwise" in x for x in result["problems"])
    elif tamper == "gradient":
        assert any("replay_production_grad_not_bitwise" in x for x in result["problems"])
    else:
        assert any("x_t_not_reproducible" in x for x in result["problems"])


def test_synthetic_calibration_builds_frozen_contract():
    contract = audit.synthetic_calibration(seed=123)
    assert contract["mode"] == "synthetic_calibrated"
    assert contract["evidence_sha256"]
    assert contract["calibration_sha256"] == contract["evidence_sha256"]
    assert contract["calibration"]["real_dataset_loaded"] is False
    assert contract["calibration"]["real_model_calls"] == 0
    assert audit.tolerance_contract_problems(contract) == []
    assert contract["calibration"]["loss_rows"] == 98
    assert contract["calibration"]["gradient_rows"] == 14


def test_domain_gap_marks_only_rotation_dependent_terms():
    rows = [
        {"id": "a", "model": "deterministic", "term": "pos",
         "status": "within_tolerance"},
        {"id": "b", "model": "deterministic", "term": "geodesic_rad2",
         "status": "within_tolerance"},
        {"id": "c", "model": "deterministic", "term": "L5",
         "status": "within_tolerance"},
        {"id": "d", "model": "diffusion", "term": "geodesic_rad2",
         "status": "within_tolerance"},
    ]
    grads = [
        {"id": "g1", "model": "deterministic", "slice": "pos",
         "status": "within_tolerance"},
        {"id": "g2", "model": "deterministic", "slice": "L5",
         "status": "within_tolerance"},
    ]
    audit._apply_domain_gaps(rows, grads, {"deterministic": 3})
    by = {r["id"]: r for r in rows}
    assert by["a"]["status"] == "within_tolerance"
    assert by["b"]["status"] == "incomplete"
    assert by["c"]["status"] == "incomplete"
    assert by["d"]["status"] == "within_tolerance"
    gby = {r["id"]: r for r in grads}
    assert gby["g1"]["status"] == "within_tolerance"
    assert gby["g2"]["status"] == "incomplete"


def test_cohort_freezes_budgets_and_call_balance(tmp_path: Path):
    root = tmp_path / "run"
    audit.write_cohort(root, mode="synthetic")
    cohort = _load(root / "cohort.json")
    assert cohort["stage_budgets"] == {
        k: dict(v) for k, v in audit.STAGE_BUDGETS.items()}
    assert cohort["call_caps"] == dict(audit.CALL_CAPS)
    t0 = time.perf_counter()
    budget = {"stage": "R0", "remaining_s": 60.0}
    audit.pre_register_attempt(
        root, "R0", "attempt_001", t0, scope={"s": 1}, budget=budget,
        call_balance=audit.AuditCallBudget().snapshot())
    row = audit.load_index(root)[0]
    assert row["call_balance"]["counts"]["conditioner"] == 0
    assert row["call_balance"]["within_caps"] is True


def test_f_carries_declared_gradient_mismatch_and_call_cap(audit_dir: Path,
                                                           tmp_path: Path):
    run = tmp_path / "declared"
    shutil.copytree(audit_dir, run)
    pdir = _stage_files(run, "P")
    r0, p, arrays, gradients = _replay_contexts(run)
    grad_doc = _load(pdir / "gradient_real_rows.json")
    grad_doc["rows"][0]["status"] = "mismatch"
    (pdir / "gradient_real_rows.json").write_text(json.dumps(grad_doc))
    p["gradient_rows"] = grad_doc["rows"]
    trace = _load(pdir / "call_trace.json")
    trace["counts"]["model"] = 99
    (pdir / "call_trace.json").write_text(json.dumps(trace))
    provider = audit.SyntheticObjectiveProvider(seed=20261026)
    calls = audit.AuditCallBudget()
    t0 = time.perf_counter()
    budget = {"stage": "F", "remaining_s": 60.0}
    scope = audit._unit_scope(r0["inputs"])
    audit.pre_register_attempt(
        run, "F", "attempt_declared", t0, scope=scope, budget=budget,
        registration_sha=r0["registration_sha256"])
    result = audit.run_f_stage(
        run, "F", "attempt_declared", t0, budget, r0, p, provider, calls,
        r0["registration"]["tolerance_contract"], mode="synthetic", scope=scope)
    assert any("p_declared_gradient_mismatch" in x for x in result["problems"])
    assert any("p_call_count_invalid:model" in x for x in result["problems"])


def test_f_bad_row_structure_writes_incomplete_without_crash(audit_dir: Path,
                                                            tmp_path: Path):
    run = tmp_path / "bad_rows"
    shutil.copytree(audit_dir, run)
    pdir = _stage_files(run, "P")
    r0, p, arrays, gradients = _replay_contexts(run)
    (pdir / "objective_real_rows.json").write_text("{}")
    (pdir / "gradient_real_rows.json").write_text("[]")
    provider = audit.SyntheticObjectiveProvider(seed=20261026)
    calls = audit.AuditCallBudget()
    t0 = time.perf_counter()
    budget = {"stage": "F", "remaining_s": 60.0}
    scope = audit._unit_scope(r0["inputs"])
    audit.pre_register_attempt(
        run, "F", "attempt_bad_rows", t0, scope=scope, budget=budget,
        registration_sha=r0["registration_sha256"])
    result = audit.run_f_stage(
        run, "F", "attempt_bad_rows", t0, budget, r0, p, provider, calls,
        r0["registration"]["tolerance_contract"], mode="synthetic", scope=scope)
    assert result["conclusion"] == "audit_incomplete"
    assert any("p_objective_rows" in x for x in result["gaps"])
    assert any("p_gradient_rows" in x for x in result["gaps"])
    assert (run / "finalize/attempt_bad_rows/audit_summary.json").is_file()


def test_r0_failure_seals_incomplete_without_crash(tmp_path: Path, monkeypatch):
    provider = audit.SyntheticObjectiveProvider(seed=7)
    monkeypatch.setattr(provider, "select_windows",
                        lambda: (_ for _ in ()).throw(ValueError("bad metadata")))
    root = tmp_path / "run"
    audit.write_cohort(root, mode="synthetic")
    t0 = time.perf_counter()
    budget = {"stage": "R0", "remaining_s": 60.0}
    audit.pre_register_attempt(root, "R0", "attempt_001", t0,
                               scope={"s": 1}, budget=budget)
    result = audit.run_r0(root, "R0", "attempt_001", t0, budget, provider,
                          audit.AuditCallBudget(),
                          audit.build_tolerance_contract(), mode="synthetic")
    assert result["passed"] is False
    assert any("window_selection_failed" in x for x in result["gaps"])
    assert (root / "registration/attempt_001/admission.json").is_file()


def test_real_cli_refuses_without_authorization(tmp_path: Path):
    out = tmp_path / "real_run"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_w1_real_objective.py"),
         "--real", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "authorization" in (proc.stdout + proc.stderr).lower()
    assert not out.exists()


def test_synthetic_cli_rejects_real_records(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_w1_real_objective.py"),
         "--synthetic", "--out", str(tmp_path / "audit"),
         "--authorization", str(tmp_path / "auth.json")],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "only valid with --real" in (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# Review counterexamples: each one must fail for the stated reason.
# ---------------------------------------------------------------------------


def test_f_counts_every_actual_vjp(audit_dir: Path, tmp_path: Path,
                                   monkeypatch):
    run = tmp_path / "vjp_count"
    shutil.copytree(audit_dir, run)
    calls = audit.AuditCallBudget()
    actual = {"production": 0, "independent": 0}
    original_prod = audit.production_vjp
    original_ind = audit.independent_vjp

    def counted_prod(*args, **kwargs):
        actual["production"] += 1
        return original_prod(*args, **kwargs)

    def counted_ind(*args, **kwargs):
        actual["independent"] += 1
        return original_ind(*args, **kwargs)

    monkeypatch.setattr(audit, "production_vjp", counted_prod)
    monkeypatch.setattr(audit, "independent_vjp", counted_ind)
    result, *_ = _run_f_on(
        run, audit.SyntheticObjectiveProvider(seed=20261026), calls=calls)
    assert result["problems"] == []
    assert actual == {"production": 2, "independent": 2}
    assert calls.snapshot()["counts"]["production_vjp"] == 2
    assert calls.snapshot()["counts"]["independent_vjp"] == 2


def test_f_binds_disk_gradients_not_in_memory_units(audit_dir: Path,
                                                    tmp_path: Path):
    run = tmp_path / "disk_grad"
    shutil.copytree(audit_dir, run)
    r0, p, arrays, gradients = _replay_contexts(run)
    for key in gradients:
        gradients[key] = -gradients[key]
    np.savez(_stage_files(run, "P") / "gradient_real_arrays.npz", **gradients)
    # p["units"] intentionally still carries the original in-memory values.
    result, *_ = _run_f_on(
        run, audit.SyntheticObjectiveProvider(seed=20261026))
    assert any("replay_production_grad_not_bitwise" in x
               for x in result["problems"]), result["problems"]
    assert any("replay_independent_grad_not_bitwise" in x
               for x in result["problems"]), result["problems"]


def test_f_recomputes_fk_from_saved_points(audit_dir: Path, tmp_path: Path):
    run = tmp_path / "fk_shift"
    shutil.copytree(audit_dir, run)
    pdir = _stage_files(run, "P")
    arrays = dict(np.load(pdir / "objective_real_arrays.npz", allow_pickle=False))
    for model in audit.MODELS:
        arrays[f"fk_points_pred_{model}"] = (
            arrays[f"fk_points_pred_{model}"] + np.float32(1.0))
    np.savez(pdir / "objective_real_arrays.npz", **arrays)
    result, *_ = _run_f_on(
        run, audit.SyntheticObjectiveProvider(seed=20261026))
    assert any("fk_elements_not_recomputed" in x
               for x in result["problems"]), result["problems"]


def test_retry_cannot_reset_call_quota(audit_dir: Path):
    with pytest.raises(SystemExit, match="cumulative call budget exhausted"):
        audit.run_registered_audit(
            audit_dir, audit.SyntheticObjectiveProvider(seed=20261026),
            mode="synthetic", tolerance=audit.build_tolerance_contract())


def test_f_restores_rng_before_replay(audit_dir: Path, tmp_path: Path):
    run = tmp_path / "rng_restore"
    shutil.copytree(audit_dir, run)

    class NoisyFactoryProvider(audit.SyntheticObjectiveProvider):
        def model_factory(self):
            torch.rand(3)
            np.random.rand(3)
            random.random()
            return super().model_factory()

    result, *_ = _run_f_on(
        run, NoisyFactoryProvider(seed=20261026))
    assert result["problems"] == [], result["problems"]
    assert result["gaps"] == [], result["gaps"]


def test_valid_negative_result_can_enter_f(tmp_path: Path):
    class MismatchProvider(audit.SyntheticObjectiveProvider):
        def loss_factory(self):
            base = super().loss_factory()

            def build(model_type):
                asm = base(model_type)
                if model_type == "deterministic":
                    asm.w = audit.LossWeights(
                        w_pos=2.0, w_rot6d=1.0, w_joint=1.0,
                        w_geodesic=0.5, w_fk=2.0, w_contact=0.0, w_vel=0.5)
                return asm

            return build

    root = tmp_path / "negative"
    result = audit.run_registered_audit(
        root, MismatchProvider(seed=20261026), mode="synthetic",
        tolerance=audit.build_tolerance_contract())
    assert result["conclusion"] == "synthetic_pipeline_mismatch", result
    p_report = _load(root / "production/attempt_001/p_report.json")
    assert p_report["stop_after_mismatch"] == "deterministic"
    assert set(p_report["units"]) == {"deterministic"}
    assert (root / "finalize/attempt_001/independent_verification.json").is_file()
    verification = _load(
        root / "finalize/attempt_001/independent_verification.json")
    assert verification["reported_models"] == ["deterministic"]
    assert verification["problems"]


def test_calibration_evidence_is_reverified():
    contract = audit.synthetic_calibration(seed=123)
    assert audit.tolerance_contract_problems(contract) == []
    widened = copy.deepcopy(contract)
    widened["tolerances"]["loss_float32"]["atol"] *= 10.0
    assert "calibration_threshold_derivation_mismatch" in \
        audit.tolerance_contract_problems(widened)
    stripped = copy.deepcopy(contract)
    del stripped["calibration"]
    stripped["evidence_sha256"] = "0" * 64
    stripped["calibration_sha256"] = "0" * 64
    assert "calibration_evidence_missing" in \
        audit.tolerance_contract_problems(stripped)
    coordinated = copy.deepcopy(contract)
    coordinated["calibration"]["observed"]["loss_float32"] = {
        "atol": 1.0, "rtol": 0.0}
    coordinated["tolerances"]["loss_float32"] = {
        "atol": 4.0, "rtol": 2e-5, "scale_floor": 0.0}
    coordinated["evidence_sha256"] = coordinated["calibration_sha256"] = \
        audit._canonical_sha256(coordinated["calibration"])
    problems = audit.tolerance_contract_problems(coordinated)
    assert any("calibration_" in p and "reproducible" in p for p in problems), problems


def test_prior_call_counts_are_cohort_wide_and_fail_closed(tmp_path: Path):
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    old = fixed / "old_run"
    current = fixed / "current_run"
    audit.write_cohort(old, mode="real")
    audit.write_cohort(current, mode="real")
    t0 = time.perf_counter()
    budget = {"stage": "R0", "remaining_s": 60.0}
    audit.pre_register_attempt(old, "R0", "attempt_001", t0,
                               scope={"s": 1}, budget=budget)
    balance = audit.AuditCallBudget(prior_counts={
        "conditioner": 1, "model": 1,
        "production_vjp": 1, "independent_vjp": 1}).snapshot()
    sealed = audit.finish_attempt(
        old, "R0", "attempt_001", t0, budget, {"complete": True},
        status="done", extra={"call_balance": balance})
    assert sealed.sealed
    counts = audit.prior_call_counts(current, fixed_root=fixed, mode="real")
    assert counts == {"conditioner": 1, "model": 1,
                      "production_vjp": 1, "independent_vjp": 1}
    ledger = _load(old / "costs.json")
    del ledger["invocations"][0]["call_balance"]
    (old / "costs.json").write_text(json.dumps(ledger))
    assert audit.prior_call_counts(current, fixed_root=fixed, mode="real") is None
    # A disk attempt without any index/ledger row is also unknown evidence.
    orphan = fixed / "orphan_run"
    audit.write_cohort(orphan, mode="real")
    (orphan / "registration" / "attempt_001").mkdir(parents=True)
    assert audit.prior_call_counts(current, fixed_root=fixed, mode="real") is None


def test_negative_costs_are_refused_by_both_entries(tmp_path: Path):
    root = tmp_path / "negative_cost"
    sealed = _seal_stage(root, "R0", "attempt_001")
    assert sealed.sealed
    index = _load(root / "attempts_index.json")
    ledger = _load(root / "costs.json")
    index["attempts"][0]["measured_wall_s"] = -100.0
    index["attempts"][0]["wall_s"] = -100.0
    ledger["invocations"][0]["measured_wall_s"] = -100.0
    ledger["invocations"][0]["wall_s"] = -100.0
    (root / "attempts_index.json").write_text(json.dumps(index))
    (root / "costs.json").write_text(json.dumps(ledger))
    assert audit.stage_history(root, "R0") is None
    _, problems = audit.stage_state(root, "R0")
    assert "attempt_cost_invalid" in problems
    with pytest.raises(SystemExit, match="history unknown"):
        audit.budget_for(root, "R0", fixed_root=tmp_path, mode="synthetic")
    with pytest.raises(ValueError, match="non-negative"):
        audit.upsert_cost(root, "R0", "attempt_002", -1.0)


def test_f_rejects_extra_rows_and_row_field_tampering(audit_dir: Path,
                                                      tmp_path: Path):
    run = tmp_path / "row_fields"
    shutil.copytree(audit_dir, run)
    pdir = _stage_files(run, "P")
    doc = _load(pdir / "objective_real_rows.json")
    rows = doc["rows"]
    extra = copy.deepcopy(rows[0])
    extra["id"] = "unexpected|row"
    rows.append(extra)
    rows[1]["denominator"] = 999999
    rows[2]["weight"] = 999999
    (pdir / "objective_real_rows.json").write_text(json.dumps(doc))
    result, *_ = _run_f_on(
        run, audit.SyntheticObjectiveProvider(seed=20261026))
    assert any("p_objective_multiset:extra_rows" in x
               for x in result["gaps"]), result["gaps"]
    assert any("unexpected|row" in x for x in result["problems"])
    assert any("denominator" in x for x in result["problems"])
    assert any("weight" in x for x in result["problems"])


def test_non_numeric_arrays_do_not_abort_verification(audit_dir: Path,
                                                      tmp_path: Path):
    run = tmp_path / "string_array"
    shutil.copytree(audit_dir, run)
    pdir = _stage_files(run, "P")
    gradients = dict(np.load(pdir / "gradient_real_arrays.npz",
                             allow_pickle=False))
    key = "production_grad_deterministic"
    gradients[key] = np.full(gradients[key].shape, "bad", dtype="<U4")
    np.savez(pdir / "gradient_real_arrays.npz", **gradients)
    result, *_ = _run_f_on(
        run, audit.SyntheticObjectiveProvider(seed=20261026))
    assert result["conclusion"] == "audit_incomplete"
    assert any("array_not_numeric" in x for x in result["gaps"]), result["gaps"]
    assert (run / "finalize/attempt_extra_f/audit_summary.json").is_file()


def test_check_arrays_string_dtype_is_a_normal_problem(tmp_path: Path):
    keys = audit.objective_array_keys()
    schema = audit._array_schema()
    arrays = {k: np.zeros(schema[k]["shape"], dtype=schema[k]["dtype"])
              for k in keys}
    arrays["x0"] = np.full(schema["x0"]["shape"], "bad", dtype="<U4")
    path = tmp_path / "string.npz"
    np.savez(path, **arrays)
    got, problems = audit._check_arrays(path, keys, schema)
    assert got
    assert any("array_not_numeric:x0" in p for p in problems)


def test_budget_revoked_p_does_not_start_f_model_calls(tmp_path: Path,
                                                       monkeypatch):
    """A P attempt revoked by accounting must not fund a fresh F replay."""
    original = audit.finish_attempt

    def forced_p_seal(root, stage, attempt, t0, budget, row, *,
                      status="done", extra=None):
        if stage == "P":
            budget = {**budget, "remaining_s": 0.0}
        return original(root, stage, attempt, t0, budget, row,
                        status=status, extra=extra)

    monkeypatch.setattr(audit, "finish_attempt", forced_p_seal)
    run = tmp_path / "p_budget_revoked"
    result = audit.run_registered_audit(
        run, audit.SyntheticObjectiveProvider(seed=20261026),
        mode="synthetic", tolerance=audit.build_tolerance_contract())
    assert result["conclusion"] == "audit_incomplete", result
    f_summary = _load(_stage_files(run, "F") / "audit_summary.json")
    assert f_summary["complete_evidence"] is False
    assert f_summary["calls"]["counts"] == {
        "conditioner": 2, "model": 2,
        "production_vjp": 2, "independent_vjp": 2,
    }
    assert any("f_replay_skipped" in g and "parent_budget_exhausted" in g
               for g in f_summary["gaps"]), f_summary["gaps"]
    assert any("replay_skipped:deterministic" in g
               for g in f_summary["gaps"]), f_summary["gaps"]


def test_f_live_deadline_stops_new_model_calls(audit_dir: Path,
                                               tmp_path: Path, monkeypatch):
    """The second F unit must not start after the first overruns the deadline."""
    run = tmp_path / "f_deadline"
    shutil.copytree(audit_dir, run)
    calls = audit.AuditCallBudget()
    seen: list[str] = []
    original = audit.production_objective_unit

    def slow_unit(bundle, *args, **kwargs):
        seen.append(bundle.model_type)
        out = original(bundle, *args, **kwargs)
        if len(seen) == 1:
            time.sleep(2.6)
        return out

    monkeypatch.setattr(audit, "production_objective_unit", slow_unit)
    result, *_ = _run_f_on(
        run, audit.SyntheticObjectiveProvider(seed=20261026),
        calls=calls, remaining_s=3.0)
    assert seen == ["deterministic"], seen
    assert calls.snapshot()["counts"]["model"] == 1
    assert result["conclusion"] == "audit_incomplete", result
    assert any("f_replay_deadline_exhausted:diffusion" in g
               for g in result["gaps"]), result["gaps"]


def test_f_seal_timeout_revokes_on_disk_pass_claim(tmp_path: Path,
                                                   monkeypatch):
    """A non-consumable F seal may not leave a pass summary on disk."""
    original = audit.finish_attempt

    def forced_f_seal(root, stage, attempt, t0, budget, row, *,
                      status="done", extra=None):
        if stage == "F":
            budget = {**budget, "remaining_s": 0.0}
        return original(root, stage, attempt, t0, budget, row,
                        status=status, extra=extra)

    monkeypatch.setattr(audit, "finish_attempt", forced_f_seal)
    run = tmp_path / "f_seal_timeout"
    result = audit.run_registered_audit(
        run, audit.SyntheticObjectiveProvider(seed=20261026),
        mode="synthetic", tolerance=audit.build_tolerance_contract())
    assert result["conclusion"] == "audit_incomplete", result
    summary_path = _stage_files(run, "F") / "audit_summary.json"
    summary = _load(summary_path)
    assert summary["conclusion"] == "audit_incomplete"
    assert summary["producer_conclusion"] == "audit_incomplete"
    assert summary["complete_evidence"] is False
    assert summary["covered_contract_passed"] is False
    row, problems = audit.stage_state(run, "F")
    assert row is not None
    assert row["status"] != "done"
    assert row["complete"] is False
    assert "budget_exceeded" in str(row.get("incomplete_reason"))
    assert "report_hash_mismatch" not in problems, problems
    assert "summary_hash_mismatch" not in problems, problems
    # The returned in-memory summary matches the revoked disk artifact.
    assert result["summary"]["conclusion"] == "audit_incomplete"
    assert result["stages"]["F"]["summary"]["complete_evidence"] is False


def test_insufficient_call_quota_stops_before_any_stage(tmp_path: Path):
    """One prior call leaves 3, but a full chain needs 4: refuse up front."""
    root = tmp_path / "quota"
    audit.write_cohort(root, mode="synthetic")
    t0 = time.perf_counter()
    budget = {"stage": "R0", "remaining_s": 60.0}
    audit.pre_register_attempt(root, "R0", "attempt_001", t0,
                               scope={"s": 1}, budget=budget)
    balance = audit.AuditCallBudget(prior_counts={
        kind: 1 for kind in audit.CALL_CAPS}).snapshot()
    sealed = audit.finish_attempt(
        root, "R0", "attempt_001", t0, budget, {"complete": True},
        status="done", extra={"call_balance": balance})
    assert sealed.sealed

    class NoCallsProvider(audit.SyntheticObjectiveProvider):
        def model_factory(self):
            raise AssertionError("the quota gate must stop before model work")

    with pytest.raises(SystemExit, match="insufficient"):
        audit.run_registered_audit(
            root, NoCallsProvider(seed=20261026), mode="synthetic",
            tolerance=audit.build_tolerance_contract())
    assert len(audit.load_index(root)) == 1
    assert not audit.stage_dir(root, "P").exists()
    assert not audit.stage_dir(root, "F").exists()


@pytest.mark.parametrize(
    ("field", "value", "expected_gap"),
    [
        ("units", ["deterministic", "diffusion"], "p_units_not_a_mapping"),
        ("unit_artifacts", ["deterministic"], "p_unit_artifacts_not_a_mapping"),
    ],
)
def test_f_nested_container_types_are_gaps_not_crashes(
        audit_dir: Path, tmp_path: Path, field: str, value, expected_gap: str):
    run = tmp_path / f"bad_{field}"
    shutil.copytree(audit_dir, run)
    report_path = _stage_files(run, "P") / "p_report.json"
    report = _load(report_path)
    report[field] = value
    report_path.write_text(json.dumps(report))
    result, *_ = _run_f_on(
        run, audit.SyntheticObjectiveProvider(seed=20261026))
    assert any(expected_gap in g for g in result["gaps"]), result["gaps"]
    assert result["conclusion"] == "audit_incomplete", result
    assert (run / "finalize/attempt_extra_f/audit_summary.json").is_file()


def test_f_input_gate_blocks_full_chain_model_calls(tmp_path: Path,
                                                    monkeypatch):
    """An admission failure must stop F before conditioner/model/VJP calls."""
    original = audit.run_p_stage

    def tampering_p(*args, **kwargs):
        out = original(*args, **kwargs)
        admission_path = _stage_files(
            Path(kwargs.get("root", args[0])), "R0") / "admission.json"
        admission = _load(admission_path)
        admission["passed"] = False
        admission_path.write_text(json.dumps(admission))
        return out

    monkeypatch.setattr(audit, "run_p_stage", tampering_p)
    run = tmp_path / "admission_gate"
    result = audit.run_registered_audit(
        run, audit.SyntheticObjectiveProvider(seed=20261026),
        mode="synthetic", tolerance=audit.build_tolerance_contract())
    assert result["conclusion"] == "audit_incomplete", result
    summary = _load(_stage_files(run, "F") / "audit_summary.json")
    assert any("admission_not_passed" in g for g in summary["gaps"]), summary["gaps"]
    assert any("f_replay_input_gate_failed" in g
               for g in summary["gaps"]), summary["gaps"]
    # P measured two units; a gated F must add no model/conditioner/VJP call.
    assert summary["calls"]["counts"] == {
        "conditioner": 2, "model": 2,
        "production_vjp": 2, "independent_vjp": 2,
    }, summary["calls"]


@pytest.mark.parametrize("tamper", ["x0_binding", "consumer_publishing"])
def test_f_binding_and_consumer_gate_skip_model_construction(
        audit_dir: Path, tmp_path: Path, tamper: str):
    run = tmp_path / f"gate_{tamper}"
    shutil.copytree(audit_dir, run)
    pdir = _stage_files(run, "P")
    r0, p, arrays, gradients = _replay_contexts(run)
    if tamper == "x0_binding":
        arrays["x0"] = arrays["x0"] + np.float32(0.01)
        np.savez(pdir / "objective_real_arrays.npz", **arrays)
    else:
        index = _load(run / "attempts_index.json")
        ledger = _load(run / "costs.json")
        for row in index["attempts"]:
            if row["stage"] == "P":
                row.update(status="publishing", complete=False)
        for row in ledger["invocations"]:
            if row["stage"] == "P":
                row.update(status="publishing", complete=False)
        (run / "attempts_index.json").write_text(json.dumps(index))
        (run / "costs.json").write_text(json.dumps(ledger))

    class NoModelProvider(audit.SyntheticObjectiveProvider):
        def model_factory(self):
            raise AssertionError("input gate must prevent model construction")

        def loss_factory(self):
            raise AssertionError("input gate must prevent loss construction")

    result, *_ = _run_f_on(run, NoModelProvider(seed=20261026))
    assert result["summary"]["complete_evidence"] is False, result
    assert any("f_replay_input_gate_failed" in g
               for g in result["gaps"]), result["gaps"]


def _withdraw_fixture(tmp_path: Path):
    root = tmp_path / "withdraw_run"
    audit.write_cohort(root, mode="synthetic")
    t0 = time.perf_counter()
    budget = {"stage": "F", "remaining_s": 60.0}
    audit.pre_register_attempt(root, "F", "attempt_001", t0,
                               scope={"s": 1}, budget=budget)
    report = root / "audit_summary.json"
    report.write_text(json.dumps({
        "conclusion": "synthetic_contract_passed",
        "producer_conclusion": "synthetic_contract_passed",
        "complete_evidence": True,
        "covered_contract_passed": True,
        "gaps": [],
    }))
    seal = audit.finish_attempt(
        root, "F", "attempt_001", t0, budget,
        {"complete": True, "report": str(report), "summary": str(report)},
        status="done", extra={"stage": "F"})
    assert seal.sealed
    return root, t0, budget, report, seal


def test_withdrawal_write_cost_is_charged_to_same_attempt(tmp_path: Path,
                                                          monkeypatch):
    root, t0, budget, report, seal = _withdraw_fixture(tmp_path)
    original = audit._persist_row_entry
    delayed = {"done": False}

    def slow_persist(*args, **kwargs):
        if not delayed["done"]:
            delayed["done"] = True
            time.sleep(1.3)
        return original(*args, **kwargs)

    monkeypatch.setattr(audit, "_persist_row_entry", slow_persist)
    updated, problems, summary = audit._withdraw_pass_claim(
        root, "F", "attempt_001", report, seal.row, ["forced_consumer"],
        reason="forced_consumer", t0=t0, budget=budget,
        preserve_conclusion=False)
    actual = time.perf_counter() - t0
    assert problems == [], problems
    assert updated["status"] == "incomplete"
    assert updated["complete"] is False
    assert updated["measured_wall_s"] >= actual, (updated, actual)
    assert audit.stage_history(root, "F") >= actual
    assert summary["complete_evidence"] is False


def test_withdrawal_without_affordable_coverage_stays_publishing(
        tmp_path: Path, monkeypatch):
    root, t0, budget, report, seal = _withdraw_fixture(tmp_path)
    original = audit._persist_row_entry

    def slow_persist(*args, **kwargs):
        time.sleep(1.3)
        return original(*args, **kwargs)

    monkeypatch.setattr(audit, "_persist_row_entry", slow_persist)
    tight = {**budget, "remaining_s": 0.2}
    updated, problems, summary = audit._withdraw_pass_claim(
        root, "F", "attempt_001", report, seal.row, ["forced_consumer"],
        reason="forced_consumer", t0=t0, budget=tight,
        preserve_conclusion=False)
    assert "withdraw_cost_unverifiable" in problems, problems
    assert updated["status"] == "publishing"
    assert updated["complete"] is False
    row, state_problems = audit.stage_state(root, "F")
    assert row is not None and row["status"] == "publishing"
    assert "attempt_final_check_pending" in state_problems, state_problems
    assert summary["complete_evidence"] is False


def test_withdrawal_slow_persist_then_kill_stays_publishing_and_covered(
        tmp_path: Path, monkeypatch):
    """A second slow persistence that lands before termination must leave a
    non-consumable publishing row whose cost still covers that write."""
    root, t0, budget, report, seal = _withdraw_fixture(tmp_path)
    original = audit._persist_row_entry
    calls = {"n": 0, "landed": []}

    def slow_persist(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            time.sleep(1.3)
        out = original(*args, **kwargs)
        calls["landed"].append(time.perf_counter() - t0)
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated termination after the write")
        return out

    monkeypatch.setattr(audit, "_persist_row_entry", slow_persist)
    with pytest.raises(KeyboardInterrupt):
        audit._withdraw_pass_claim(
            root, "F", "attempt_001", report, seal.row, ["forced_consumer"],
            reason="forced_consumer", t0=t0, budget=budget,
            preserve_conclusion=False)
    assert calls["n"] == 2, calls
    landed = calls["landed"][-1]
    row = [r for r in audit.load_index(root)
           if r.get("stage") == "F" and r.get("attempt") == "attempt_001"][-1]
    assert row["status"] == "publishing", row
    assert row["complete"] is False
    assert float(row["measured_wall_s"]) >= landed, (row, landed)
    assert float(row["seal_tail_reserve_s"]) >= 1.0, row
    assert audit.stage_history(root, "F") is None


def test_withdrawal_never_lowers_the_registered_cost(tmp_path: Path):
    """A withdrawal may add cost but must keep the old seal charge as a floor."""
    root, t0, budget, report, seal = _withdraw_fixture(tmp_path)
    old_cost = float(seal.row["measured_wall_s"])
    reserved_cost = max(old_cost, 3.9099)
    old_row = {**seal.row, "measured_wall_s": reserved_cost,
               "sampled_wall_s": reserved_cost, "seal_tail_reserve_s": 2.9}
    updated, problems, summary = audit._withdraw_pass_claim(
        root, "F", "attempt_001", report, old_row, ["forced_consumer"],
        reason="forced_consumer", t0=t0, budget=budget,
        preserve_conclusion=False)
    assert problems == [], problems
    assert updated["status"] == "incomplete", updated
    assert updated["complete"] is False
    assert float(updated["measured_wall_s"]) >= reserved_cost, (
        updated, reserved_cost)
    history = audit.stage_history(root, "F")
    assert history is not None and history >= reserved_cost, (history, reserved_cost)
    assert summary["complete_evidence"] is False


def test_f_identity_exception_blocks_model_replay(tmp_path: Path):
    """Unverifiable identity is a replay gate, not only a recorded gap."""
    calls = {"n": 0}

    class FlakyIdentity(audit.SyntheticObjectiveProvider):
        def verify_identity(self):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise OSError("identity store unavailable")
            return super().verify_identity()

    run = tmp_path / "identity_exception"
    result = audit.run_registered_audit(
        run, FlakyIdentity(seed=20261026), mode="synthetic",
        tolerance=audit.build_tolerance_contract())
    f_summary = _load(_stage_files(run, "F") / "audit_summary.json")
    assert f_summary["calls"]["counts"] == {
        "conditioner": 2, "model": 2,
        "production_vjp": 2, "independent_vjp": 2,
    }, f_summary["calls"]
    assert any("identity_reverification_failed" in g
               for g in f_summary["gaps"]), f_summary["gaps"]
    assert any("f_replay_identity_gate_failed" in g
               for g in f_summary["gaps"]), f_summary["gaps"]
    assert any("replay_skipped:deterministic:identity_unverifiable" in g
               for g in f_summary["gaps"]), f_summary["gaps"]
    assert result["conclusion"] == "audit_incomplete", result


def test_f_negative_result_withdrawing_past_coverage_is_incomplete(
        tmp_path: Path, monkeypatch):
    """A negative conclusion survives withdrawal, but unknown sealing is
    ``audit_incomplete`` rather than a confirmed mismatch."""
    original_p = audit.run_p_stage
    original_seal = audit.finish_attempt
    original_withdraw = audit._withdraw_pass_claim

    def tampering_p(*args, **kwargs):
        out = original_p(*args, **kwargs)
        path = Path(out["paths"]["arrays"])
        arrays = dict(np.load(path, allow_pickle=False))
        arrays["prediction_deterministic"] = (
            arrays["prediction_deterministic"] + np.float32(0.1))
        np.savez(path, **arrays)
        return out

    def forced_f_seal(root, stage, attempt, t0, budget, row, *,
                      status="done", extra=None):
        if stage == "F":
            budget = {**budget, "remaining_s": 0.0}
        return original_seal(root, stage, attempt, t0, budget, row,
                             status=status, extra=extra)

    def exhausted_withdraw(root, stage, attempt, report_path, row, gaps,
                           **kwargs):
        # The withdrawal writes are charged to the same attempt: model the
        # remaining budget being consumed by them.
        kwargs["budget"] = {**kwargs["budget"], "remaining_s": 0.0}
        return original_withdraw(root, stage, attempt, report_path, row, gaps,
                                 **kwargs)

    monkeypatch.setattr(audit, "run_p_stage", tampering_p)
    monkeypatch.setattr(audit, "finish_attempt", forced_f_seal)
    monkeypatch.setattr(audit, "_withdraw_pass_claim", exhausted_withdraw)
    run = tmp_path / "negative_withdraw_publishing"
    result = audit.run_registered_audit(
        run, audit.SyntheticObjectiveProvider(seed=20261026),
        mode="synthetic", tolerance=audit.build_tolerance_contract())
    row, problems = audit.stage_state(run, "F")
    assert row is not None and row["status"] == "publishing", (row, problems)
    assert result["conclusion"] == "audit_incomplete", result
    assert result["seals"]["F"]["terminal"] is False, result["seals"]
    assert result["seals"]["F"]["sealed"] is False, result["seals"]
    summary = _load(_stage_files(run, "F") / "audit_summary.json")
    # The measured mismatch is preserved in the evidence; only the
    # completeness/pass claim is withdrawn.
    assert summary["conclusion"] == "synthetic_pipeline_mismatch", summary
    assert summary["complete_evidence"] is False
    assert any("f_evidence_completeness_withdrawn" in g
               for g in summary["gaps"]), summary["gaps"]


def test_f_negative_result_seal_failure_withdraws_completeness_only(
        tmp_path: Path, monkeypatch):
    """A measured mismatch stays a mismatch, but cannot claim complete evidence."""
    original_p = audit.run_p_stage
    original_seal = audit.finish_attempt

    def tampering_p(*args, **kwargs):
        out = original_p(*args, **kwargs)
        path = Path(out["paths"]["arrays"])
        arrays = dict(np.load(path, allow_pickle=False))
        arrays["prediction_deterministic"] = (
            arrays["prediction_deterministic"] + np.float32(0.1))
        np.savez(path, **arrays)
        return out

    def forced_f_seal(root, stage, attempt, t0, budget, row, *,
                      status="done", extra=None):
        if stage == "F":
            budget = {**budget, "remaining_s": 0.0}
        return original_seal(root, stage, attempt, t0, budget, row,
                             status=status, extra=extra)

    monkeypatch.setattr(audit, "run_p_stage", tampering_p)
    monkeypatch.setattr(audit, "finish_attempt", forced_f_seal)
    run = tmp_path / "negative_seal"
    result = audit.run_registered_audit(
        run, audit.SyntheticObjectiveProvider(seed=20261026),
        mode="synthetic", tolerance=audit.build_tolerance_contract())
    summary = _load(_stage_files(run, "F") / "audit_summary.json")
    assert summary["conclusion"] == "synthetic_pipeline_mismatch", summary
    assert summary["producer_conclusion"] == "synthetic_pipeline_mismatch"
    assert summary["complete_evidence"] is False
    assert summary["covered_contract_passed"] is False
    assert any("f_evidence_completeness_withdrawn" in g
               for g in summary["gaps"]), summary["gaps"]
    assert result["complete_evidence"] is False
    assert result["conclusion"] == "synthetic_pipeline_mismatch"
    row, problems = audit.stage_state(run, "F")
    assert row is not None
    assert "report_hash_mismatch" not in problems, problems
    assert "summary_hash_mismatch" not in problems, problems
