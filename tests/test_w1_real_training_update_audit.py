"""Synthetic acceptance tests for the W1 real training-update audit (I/C).

These tests exercise the full synthetic harness: the shared Trainer update
unit, the frozen effective-float32 AdamW graph, the update-delta and state
zero gates, the cross-path propagation classification, R0/P/I/F accounting,
and the fresh-process F replay.  No real checkpoint, dataset, cache, G1 asset
or MuJoCo model is read.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch

import ireset.eval.w1_real_training_update_audit as A
from ireset.train.trainer import TrainingNumericalError


TINY_CONFIG = dict(A.MODEL_CONFIG)
TINY_CONFIG.update(d_model=8, n_layers=1, n_heads=2, ffn_mult=2,
                   time_window=3)


@pytest.fixture(scope="module")
def tiny_provider(tmp_path_factory):
    root = tmp_path_factory.mktemp("w1_update_fixture")
    provider = A.SyntheticTrainingProvider(TINY_CONFIG, seed=20261029,
                                           device="cuda")
    provider.prepare_fixture(root / "fixture")
    provider.load_fixture()
    assert not provider.fixture["problems"], provider.fixture["problems"]
    return provider


@pytest.fixture(scope="module")
def chain(tiny_provider, tmp_path_factory):
    root = tmp_path_factory.mktemp("w1_update_chain")
    counter = A.CallCounter(A.CALL_CAPS)
    tolerance = {"tolerances": A.TOLERANCE_PROPOSALS}
    budget = {"remaining_s": 10 ** 6, "reserve_s": 0.0}
    p = A.run_p_stage(root, "P", "attempt_001", A._now(), budget,
                      tiny_provider, counter, tiny_provider.roster,
                      out_dir=root / "P")
    i = A.run_i_stage(root, "I", "attempt_001", A._now(), budget,
                      tiny_provider, counter, tiny_provider.roster,
                      root / "P", tolerance=tolerance, out_dir=root / "I")
    return {"root": root, "counter": counter, "tolerance": tolerance,
            "p": p, "i": i, "provider": tiny_provider}


# ---------------------------------------------------------------------------
# shared update unit
# ---------------------------------------------------------------------------


def _two_trainers(provider, model_type):
    first = provider.build_trainer(model_type, with_optimizer=True,
                                   counter=A.CallCounter(A.CALL_CAPS),
                                   loss_mode="production",
                                   out_dir=provider.fixture_dir / "eq1")
    second = provider.build_trainer(model_type, with_optimizer=True,
                                    counter=A.CallCounter(A.CALL_CAPS),
                                    loss_mode="production",
                                    out_dir=provider.fixture_dir / "eq2")
    return first, second


@pytest.mark.parametrize("model_type", ["deterministic", "diffusion"])
def test_observer_is_bitwise_transparent(tiny_provider, model_type):
    first, second = _two_trainers(tiny_provider, model_type)
    rng_before = torch.get_rng_state()
    for trainer, use_observer in ((first, False), (second, True)):
        trainer.begin_update()
        gen = (torch.Generator(device="cpu").manual_seed(A.NOISE_SEED)
               if model_type == "diffusion" else None)
        for micro in range(A.MICRO_COUNT):
            observer = {} if use_observer else None
            trainer.accumulate_micro(
                tiny_provider.micro_batch(1, micro),
                noise_generator=gen, observer=observer)
        trainer.clip_gradients()
        trainer.step_optimizer()
    for name, p1 in first._named_parameters():
        p2 = dict(second._named_parameters())[name]
        assert torch.equal(p1, p2), name
        s1 = first.opt.state[p1]
        s2 = second.opt.state[p2]
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(s1[key], s2[key]), (name, key)
    assert torch.equal(rng_before, torch.get_rng_state())


def test_observer_does_not_change_exception(tiny_provider):
    trainer, other = _two_trainers(tiny_provider, "deterministic")
    batch = tiny_provider.micro_batch(1, 0)
    bad = dict(batch)
    bad["x0"] = batch["x0"].clone()
    bad["x0"][0, 0, 0] = float("nan")
    for target, observer in ((trainer, None), (other, {})):
        target.begin_update()
        with pytest.raises(TrainingNumericalError) as caught:
            target.accumulate_micro(bad, observer=observer)
        assert caught.value.reason == "nonfinite_input"
    trainer.begin_update()
    trainer.accumulate_micro(batch)
    gnorm = trainer.clip_gradients()
    assert gnorm > 0.0


def test_observer_captures_prediction_and_leaf_contributions(tiny_provider):
    trainer = tiny_provider.build_trainer(
        "diffusion", with_optimizer=True,
        counter=A.CallCounter(A.CALL_CAPS), loss_mode="production",
        out_dir=tiny_provider.fixture_dir / "obs")
    trainer.begin_update()
    observer: dict = {}
    gen = torch.Generator(device="cpu").manual_seed(A.NOISE_SEED)
    trainer.accumulate_micro(tiny_provider.micro_batch(1, 0),
                             noise_generator=gen, observer=observer)
    assert observer["prediction_hook_fires"] == 1
    assert len(observer["prediction_grads"]) == 1
    names = [n for n, _ in trainer._named_parameters()]
    assert set(observer["leaf_counts"]) <= set(names)
    for name, grad in observer["accumulated"].items():
        if grad is not None:
            assert name in observer["leaf_counts"]
            assert observer["leaf_counts"][name] == 1
    assert set(observer["forward_capture"]) >= {"x0_pred", "memory", "t",
                                                "noise", "x_t"}


# ---------------------------------------------------------------------------
# roster, schedule and noise
# ---------------------------------------------------------------------------


def test_roster_and_row_formula(tiny_provider):
    roster = tiny_provider.roster
    catalog = A.row_catalog(roster)
    assert catalog["n_rows"] == 14 * roster["n_total"] + 168
    assert len(set(catalog["rows"])) == len(catalog["rows"])
    assert catalog["counts"]["micro_param"] == 8 * roster["n_total"]
    for model in A.MODELS:
        names = [e["name"] for e in roster["models"][model]["parameters"]]
        assert len(names) == roster["models"][model]["n_parameters"]
        assert names[-1].startswith("conditioner.")
        assert names[0].startswith("model.")


def test_noise_protocol_and_micro_schedule():
    noise = A.draw_noise_path(A.NOISE_SEED)
    assert len(noise) == 8
    assert [(u["update"], u["micro"]) for u in noise] == [
        (u, m) for u in A.UPDATES for m in range(A.MICRO_COUNT)]
    for unit in noise:
        assert unit["t"].shape == (4,)
        assert unit["epsilon"].shape == (4, A.WINDOW, 38)
        assert unit["epsilon"].dtype == torch.float32
        assert int(unit["t"].min()) >= 0 and int(unit["t"].max()) < 1000
    for update in A.UPDATES:
        counter = {}
        for micro in A.MICRO_SCHEDULE[update]:
            for window in micro:
                counter[window] = counter.get(window, 0) + 1
        assert counter == {w: 4 for w in A.WINDOW_IDS}
    again = A.draw_noise_path(A.NOISE_SEED)
    for a, b in zip(noise, again):
        assert torch.equal(a["t"], b["t"])
        assert torch.equal(a["epsilon"], b["epsilon"])


# ---------------------------------------------------------------------------
# effective float32 and semantic cases
# ---------------------------------------------------------------------------


def test_plan_scalar_1ulp_is_a_sensitivity_gap():
    case = A._scalar_case(0.0, 0.001, 1.0000000116860974e-7, 2,
                          -0.008999999612569809,
                          -0.008999998681247234, m_none=False)
    assert abs(case["m_p"] - 5.494803223538014e-11) < 1e-20
    assert abs(case["m_i"] - 1.4808029802360068e-10) < 1e-20
    assert 0.62 < abs(case["m_p"] - case["m_i"]) / abs(case["m_i"]) < 0.64
    assert case["classification"] == "cross_path_sensitivity_gap"
    assert case["gaps"] == ["cross_path_sensitivity_gap"]
    assert case["observable_p"] and case["observable_i"]
    assert case["p_formula"] == [] and case["i_formula"] == []
    assert not case["interval_recompute_problems"]


def test_stable_scalar_passes_all_gates():
    case = A._scalar_case(0.0, 0.001, 1.0000000116860974e-7, 2,
                          -0.009, -0.009, m_none=False)
    assert case["classification"] == "passed"
    assert case["p_formula"] == [] and case["i_formula"] == []


def test_post_state_tamper_is_a_direct_failure():
    case = A._scalar_case(1.0, 0.002, 3.3e-7, 2, 0.0123, 0.0123,
                          m_none=False, post_p=1.0)
    assert case["classification"] == "direct_failure"
    assert "theta" in case["p_formula"]


def test_update_minus_twenty_percent_is_rejected():
    reference = A.adamw_effective_fp32(np.array([1.0]), np.array([0.002]),
                                       np.array([3.3e-7]),
                                       np.array([0.0123]), 2)
    theta = float(reference["theta"][0])
    wrong = 1.0 + 0.8 * (theta - 1.0)
    case = A._scalar_case(1.0, 0.002, 3.3e-7, 2, 0.0123, 0.0123,
                          m_none=False, post_p=wrong)
    assert case["classification"] == "direct_failure"
    assert "theta" in case["p_formula"]


def test_local_state_zero_is_rejected():
    case = A._scalar_case(0.0, 0.0, 0.0, 1, 1e-17, 1e-17,
                          m_none=True, v_post_p=0.0)
    ref = A.adamw_effective_fp32(np.array([0.0]), None, None,
                                 np.array([1e-17]), 1, m_none=True,
                                 v_none=True)
    assert float(ref["v"][0]) > 0.0
    assert float(ref["v"][0]) < 1e-36
    assert case["classification"] == "direct_failure"
    assert "v" in case["p_formula"]
    assert "v_nonzero_erased" in case["zero_problems"]


def test_small_correct_update_is_an_honest_precision_gap():
    case = A._scalar_case(1.0, 1e-3, 1.0, 2, 1e-3, 1e-3, m_none=False)
    assert case["p_formula"] == [] and case["i_formula"] == []
    assert case["classification"] == "audit_incomplete"
    assert "parameter_delta_precision_gap" in case["gaps"]
    assert case["bound_delta_p"] > 5e-4 * abs(case["delta_ref_p"])


def test_interval_tamper_is_rejected_from_raw_inputs():
    g = np.array([1e-3])
    iv = A.propagate_intervals(g, np.array([1.0]), np.array([1e-3]),
                               np.array([1.0]), 2)
    bad = {"K_m": (iv["K_m"][0] * 0.5, iv["K_m"][1] * 0.5),
           "K_v": iv["K_v"], "K_theta": iv["K_theta"],
           "K_delta": iv["K_delta"]}
    assert A._interval_recompute_problems(
        g, np.array([1.0]), np.array([1e-3]), np.array([1.0]), 2, bad)
    assert not A._interval_recompute_problems(
        g, np.array([1.0]), np.array([1e-3]), np.array([1.0]), 2, iv)


def test_effective_graph_handles_none_and_zero_states():
    first = A.adamw_effective_fp32(np.array([0.5]), None, None,
                                   np.array([0.01]), 1, m_none=True,
                                   v_none=True)
    assert first["state_initialized"] is False
    assert np.array_equal(first["m"], np.array([0.001], np.float32))
    zero = A.adamw_effective_fp32(np.array([0.5]),
                                  np.zeros(1, np.float32),
                                  np.zeros(1, np.float32),
                                  np.zeros(1, np.float32), 2)
    assert np.array_equal(zero["theta"], np.array([0.5], np.float32))
    assert np.array_equal(zero["m"], np.zeros(1, np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA dispatch")
def test_fp32_contract_matches_cuda_and_rejects_tamper():
    contract = A.build_fp32_contract("cuda")
    assert contract["valid"], contract["probe_mismatches"]
    assert contract["two_step_match"]
    assert A.fp32_contract_problems(contract, recompute=True) == []
    tampered = dict(contract)
    tampered["node_graph"] = "wrong"
    assert A.fp32_contract_problems(tampered, recompute=False)


# ---------------------------------------------------------------------------
# production chain rows and F replay
# ---------------------------------------------------------------------------


def _register_f_parent(root, attempt="attempt_001"):
    A.write_cohort(root, mode="synthetic")
    budget = {"remaining_s": 10 ** 6, "reserve_s": 0.0,
              "attempts": 2, "attempts_used": 0}
    A.pre_register_attempt(root, "F", attempt, A._now(), scope={},
                           budget=budget)
    A.reserve_stage_calls(root, "F", attempt,
                          {"counts": A.STAGE_NOMINAL["F"]})
    return budget


def test_chain_rows_and_call_matrix(chain):
    i_rows = chain["i"]["rows"]
    counts = chain["counter"].snapshot()["counts"]
    assert counts["conditioner_forwards"] == 32
    assert counts["backbone_forwards"] == 32
    assert counts["production_loss_assembler"] == 16
    assert counts["production_float32_fk_batch"] == 32
    assert counts["independent_output_vjp"] == 16
    assert counts["independent_total_vjp"] == 16
    assert counts["native_adamw_steps"] == 4
    assert counts["explicit_reference_updates"] == 4
    assert counts["diagnostic_parameter_update_events"] == 8
    assert counts["reference_mj_forward"] == 7920
    assert counts["reference_mj_jac_site"] == 57600
    assert any(r["key"].startswith("adamw:") for r in i_rows)
    assert not chain["i"]["failures"]


def test_adamw_classification_is_visible(chain):
    rows = [r for r in chain["i"]["rows"] if r["key"].startswith("adamw:")]
    classes = {r["classification"] for r in rows}
    assert classes <= {"passed", "audit_incomplete",
                       "cross_path_sensitivity_gap", "direct_failure",
                       "cross_path_bound_unverifiable"}
    assert "direct_failure" not in classes
    assert any(r["classification"] == "passed" for r in rows)


def test_f_worker_replays_bitwise_and_verifies_rows(chain, tmp_path):
    provider = chain["provider"]
    _register_f_parent(tmp_path)
    counter = A.CallCounter(A.CALL_CAPS)
    result = A.run_f_stage(
        tmp_path, "F", "attempt_001", A._now(),
        {"remaining_s": 10 ** 6, "reserve_s": 0.0}, provider, counter,
        provider.roster, chain["root"] / "P", chain["root"] / "I",
        tolerance=chain["tolerance"], out_dir=tmp_path / "F")
    assert result["complete_evidence"] is True
    assert result["worker"]["exit_code"] == 0
    assert result["worker"]["pid"] != result["worker"]["parent_pid"]
    assert result["worker_result"]["schedule_rebuild_ok"] is True


# ---------------------------------------------------------------------------
# accounting, authorization and tamper detection
# ---------------------------------------------------------------------------


def test_call_counter_caps_and_types():
    counter = A.CallCounter({"a": 2, "b": 1})
    counter.add("a")
    counter.add("a")
    with pytest.raises(A.UpdateCallBudgetExceeded):
        counter.add("a")
    with pytest.raises(ValueError):
        counter.add("b", True)
    with pytest.raises(ValueError):
        counter.add("b", 1.0)
    snap = counter.snapshot()
    assert snap["within_caps"] and snap["counts"]["a"] == 2


def test_stage_state_rejects_running_publishing_and_unsettled(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    attempt = "attempt_001"
    (A.stage_dir(root, "P") / attempt).mkdir(parents=True)
    budget = {"remaining_s": 10.0, "reserve_s": 1.0, "attempts": 2,
              "attempts_used": 0}
    A.pre_register_attempt(root, "P", attempt, A._now(), scope={},
                           budget=budget,
                           call_balance=A.CallCounter(A.CALL_CAPS).snapshot())
    A.reserve_stage_calls(root, "P", attempt,
                          {"counts": A.STAGE_NOMINAL["P"]})
    _row, problems = A.stage_state(root, "P")
    assert "stage_not_terminal" in problems
    assert any(p.startswith("reservation_unsettled") for p in problems)
    row = {"complete": True, "report": None, "summary": None}
    seal = A.finish_attempt(root, "P", attempt, A._now(), budget, row,
                            status="done",
                            extra={"call_balance": A.CallCounter(
                                A.CALL_CAPS).snapshot()})
    assert seal.terminal and seal.sealed
    A.settle_stage_calls(root, "P", attempt,
                         {"counts": A.CallCounter(A.CALL_CAPS).snapshot()["counts"]})
    _row, problems = A.stage_state(root, "P")
    assert problems == []


def test_authorization_template_is_refused_and_faithful_record_accepted(
        tiny_provider):
    template = A.authorization_template()
    assert A.authorization_problems(template)
    roster = tiny_provider.roster
    record = dict(template)
    record.update(template_only=False, authorized=True,
                  granted_by="test", granted_wall="2026-10-29",
                  checklist={k: True for k in A.AUTHORIZATION_REQUIRED},
                  budgets={stage: {"budget_s": spec["budget_s"],
                                   "attempts": spec["attempts"]}
                           for stage, spec in A.STAGE_BUDGETS.items()},
                  call_caps=dict(A.CALL_CAPS),
                  updates=dict(A.EXPECTED_UPDATES),
                  resource_caps=dict(A.RESOURCE_CAPS),
                  draft_sha256="a" * 64,
                  tolerance_evidence_sha256="b" * 64,
                  fp32_contract_sha256="c" * 64,
                  admission_package_sha256="d" * 64,
                  implementation_files={"x": "e" * 64})
    assert A.authorization_problems(record) == []
    bad = dict(record)
    bad["checklist"] = dict(record["checklist"])
    bad["checklist"][A.AUTHORIZATION_REQUIRED[0]] = False
    assert any(p.startswith("authorization_checklist")
               for p in A.authorization_problems(bad))
    bad = dict(record)
    bad["budgets"] = {**record["budgets"],
                      "P": {"budget_s": 901.0, "attempts": 2}}
    assert any(p.startswith("authorization_budget_size")
               for p in A.authorization_problems(bad))


def test_f_worker_detects_tampered_arrays_and_rows(chain, tmp_path):
    import shutil

    provider = chain["provider"]
    tampered_p = tmp_path / "P_tampered"
    shutil.copytree(chain["root"] / "P", tampered_p)
    path = tampered_p / "p_deterministic.npz"
    arrays = A._load_npz(path)
    key = next(k for k in arrays if k.startswith("theta_after/u1/"))
    arrays[key] = arrays[key].copy()
    arrays[key] = arrays[key] + np.float32(1e-3)
    np.savez(path, **arrays)
    _register_f_parent(tmp_path)
    counter = A.CallCounter(A.CALL_CAPS)
    result = A.run_f_stage(
        tmp_path, "F", "attempt_001", A._now(),
        {"remaining_s": 10 ** 6, "reserve_s": 0.0}, provider, counter,
        provider.roster, tampered_p, chain["root"] / "I",
        tolerance=chain["tolerance"], out_dir=tmp_path / "F")
    assert result["complete_evidence"] is False
    assert any("npz_value" in p for p in result["verification_problems"])


def test_row_multiset_and_numeric_type_rejection():
    rows = [{"key": "a", "v": 1}, {"key": "b", "v": 2}]
    catalog = ["a", "b", "c"]
    problems = A._row_compare_problems(rows, rows, catalog)
    assert any(p.startswith("rows_missing") for p in problems)
    problems = A._row_compare_problems(rows, rows + [{"key": "a"}],
                                       ["a", "b"])
    assert "rows_duplicate" in problems
    bad = A.compare_numeric(np.array([True]), np.array([1.0]),
                            A.TOLERANCE_PROPOSALS["loss_scalar"],
                            name="bool")
    assert not bad["ok"]
    nan = A.compare_numeric(np.array([np.nan]), np.array([0.0]),
                            A.TOLERANCE_PROPOSALS["loss_scalar"],
                            name="nan")
    assert "nan:nonfinite" in nan["problems"]


def test_budget_attempt_limit_and_unknown_history(tiny_provider, tmp_path):
    root = tmp_path / "budget"
    A.write_cohort(root, mode="synthetic")
    history = A.stage_history(root, "P")
    assert history == 0.0
    budget = {"remaining_s": 10.0, "reserve_s": 1.0, "attempts": 2,
              "attempts_used": 2}
    with pytest.raises(SystemExit):
        A.pre_register_attempt(root, "P", "attempt_003", A._now(), scope={},
                               budget=budget)
    (A.stage_dir(root, "P") / "attempt_001").mkdir(parents=True)
    A.upsert_index(root, "P", "attempt_001",
                   {"status": "running", "complete": False,
                    "measured_wall_s": 0.0,
                    "accounting_schema": A.ACCOUNTING_SCHEMA,
                    "cohort_id": A.COHORT_ID})
    assert A.stage_history(root, "P") is None


def test_withdraw_rewrites_pass_claim(tmp_path):
    root = tmp_path / "withdraw"
    A.write_cohort(root, mode="synthetic")
    attempt = "attempt_001"
    directory = A.stage_dir(root, "F") / attempt
    directory.mkdir(parents=True)
    report = directory / "final.json"
    A.write_json(report, {"conclusion": "real_training_update_covered_contract_passed",
                          "complete_evidence": True,
                          "covered_contract_passed": True})
    budget = {"remaining_s": 10.0, "reserve_s": 1.0, "attempts": 2,
              "attempts_used": 0}
    t0 = A._now()
    A.pre_register_attempt(root, "F", attempt, t0, scope={}, budget=budget)
    seal = A.finish_attempt(root, "F", attempt, t0, budget,
                            {"complete": True,
                             "report": A.path_for_record(report),
                             "summary": A.path_for_record(report)},
                            status="done")
    assert seal.sealed
    row, problems, _ = A.withdraw_attempt_claim(
        root, "F", attempt, report, seal.row, ["call_matrix"], reason="matrix",
        t0=t0, budget=budget)
    data = A.read_json(report)
    assert data["conclusion"] == "audit_incomplete"
    assert data["complete_evidence"] is False
    assert row["complete"] is False
    assert problems == []


# ---------------------------------------------------------------------------
# review-round counterexamples: numeric gates, F binding, accounting
# ---------------------------------------------------------------------------


def test_zero_floor_cannot_erase_a_nonzero_reference():
    spec = A.TOLERANCE_PROPOSALS["parameter_gradient"]
    ref = np.full(100, 1e-9)
    got = np.zeros(100)
    rec = A.compare_gradient(got, ref, spec, name="erased")
    assert rec["ok"] is False
    assert rec["problems"]
    partial = ref.copy()
    partial[:50] = 0.0
    rec = A.compare_gradient(partial, ref, spec, name="partial")
    assert rec["ok"] is False
    deep = np.full(8, 4e-12)
    rec = A.compare_gradient(np.zeros(8), deep, spec, name="deep")
    assert rec["ok"] is False
    assert rec["groups"]["full_erasure"] is True
    assert any(p.endswith("zero_erased") for p in rec["problems"])


def test_genuine_cancellation_noise_still_passes():
    rng = np.random.default_rng(5)
    ref = rng.normal(0.0, 4e-12, size=8)
    ref[0] = 0.0
    got = ref + rng.normal(0.0, 2e-12, size=8)
    rec = A.compare_gradient(got, ref,
                             A.TOLERANCE_PROPOSALS["parameter_gradient"],
                             name="noise")
    assert rec["ok"] is True
    assert rec["groups"]["zero_noise_floor"] == A.ZERO_GRADIENT_ABS


def test_state_cross_path_floor_does_not_hide_divergence():
    spec = A.TOLERANCE_PROPOSALS["state_cross_path"]
    flags = A._cross_path_flags(
        np.array([0.0]), np.array([1e-9]),
        np.array([0.0]), np.array([1e-9]),
        np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([1e-9]),
        np.array([True]), np.array([True]), spec)
    assert flags["m_ok"] is False
    assert flags["m_zero_disagreement"] is True
    assert flags["delta_ok"] is False
    assert flags["delta_presence_mismatch"] is True
    flags = A._cross_path_flags(
        np.array([1e-8]), np.array([2e-8]),
        np.array([1e-8]), np.array([2e-8]),
        np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]),
        np.array([False]), np.array([False]), spec)
    assert flags["m_ok"] is False and flags["v_ok"] is False
    flags = A._cross_path_flags(
        np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
        np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
        np.array([False]), np.array([False]), spec)
    assert flags["m_ok"] and flags["v_ok"] and flags["theta_ok"]
    assert flags["delta_ok"] and not flags["delta_presence_mismatch"]


def test_interval_rounding_contains_actual_binary32_nodes():
    rng = np.random.default_rng(11)
    for scale in (1e-12, 1e-9, 1e-4, 1e-1, 1.0):
        g = (rng.normal(size=16) * scale).astype(np.float32)
        theta = (rng.normal(size=16) * 0.4).astype(np.float32)
        state = scale >= 1e-9
        m = (rng.normal(size=16) * max(scale, 1e-9)).astype(np.float32)
        v = (np.abs(rng.normal(size=16)) * max(scale, 1e-9) ** 2
             ).astype(np.float32)
        iv = A.propagate_intervals(g, theta, m, v, 2,
                                    m_none=not state, v_none=not state)
        candidates = [g.astype(np.float64)]
        for direction in (np.float32(-np.inf), np.float32(np.inf)):
            neighbour = np.nextafter(g, direction).astype(np.float64)
            inside = ((neighbour >= iv["g_lo"])
                      & (neighbour <= iv["g_hi"]))
            candidates.append(np.where(inside, neighbour,
                                       g.astype(np.float64)))
        for candidate in candidates:
            eff = A.adamw_effective_fp32(
                theta, m, v, candidate.astype(np.float32), 2,
                m_none=not state, v_none=not state)
            assert A._inside(iv["K_m"], eff["m"])
            assert A._inside(iv["K_v"], eff["v"])
            assert A._inside(iv["K_theta"], eff["theta"])
            assert A._inside(iv["K_delta"], eff["delta_ref64"])
    real = rng.normal(size=512) * 10.0 ** rng.uniform(-14, 2, size=512)
    assert np.all(A._r32_down(real) <= real)
    assert np.all(A._r32_up(real) >= real)


def test_clip_coefficient_scales_at_the_equality_boundary():
    from ireset.train.trainer import clip_coefficient

    grad = np.array([1.0, -1.0], np.float32)
    norm = float(np.sqrt(2.0))
    out = A._clip_independent({"g": grad}, norm)
    expected = min(1.0, norm / (norm + 1e-6))
    assert out["triggered"] is True
    assert out["coef"] == pytest.approx(expected, rel=1e-12)
    assert clip_coefficient(norm, norm) == out["coef"]
    assert clip_coefficient(1.0, 0.5) == 1.0
    assert np.array_equal(out["postclip"]["g"],
                          (grad * np.float32(out["coef"])).astype(np.float32))


def test_calibration_state_ratio_is_normalized_and_maxed():
    case = A._scalar_case(0.3, 0.002, 3.3e-7, 2, 0.5, 0.505, m_none=False)
    spec = A.TOLERANCE_PROPOSALS["state_cross_path"]
    m_ratio = (abs(case["m_p"] - case["m_i"])
               / (spec["atol"] + spec["rtol"] * abs(case["m_i"])))
    v_ratio = (abs(case["v_p"] - case["v_i"])
               / (spec["atol"] + spec["rtol"] * abs(case["v_i"])))
    assert case["max_state_ratio"] == pytest.approx(max(m_ratio, v_ratio),
                                                     rel=1e-12)
    # Small states are part of the gate domain; the old absolute mask hid
    # them from the calibration statistic.
    tiny = A._scalar_case(0.0, 1e-9, 1e-18, 1, 3e-7,
                          3e-7 * (1 + 1e-4), m_none=False)
    assert tiny["m_p"] < A.ZERO_STATE_ABS
    assert tiny["max_state_ratio"] > 0.0
    ratio, count = A._state_cross_path_ratio(
        np.array([1e-9]), np.array([1e-9 * (1 + 1e-4)]), spec)
    assert count == 1
    assert ratio == pytest.approx(
        1e-4 / (spec["rtol"] * (1 + 1e-4)), rel=1e-9)
    merged = A._merge_ratio_records([
        A._ratio_record(0.1, source="chain"),
        A._ratio_record(0.4, source="scalar")])
    assert merged["max_ratio_of_proposal"] == 0.4
    assert merged["sources"] == ["chain", "scalar"]


def test_artifact_bound_is_per_class_and_covers_the_chain(tiny_provider,
                                                          chain):
    bound = A._resource_bound(tiny_provider.roster)
    assert bound["cohort_bytes"] == (
        2 * (bound["p_attempt_bytes"] + bound["i_attempt_bytes"]
             + bound["f_attempt_bytes"])
        + bound["fixture_bytes"] + bound["registration_bytes"])
    assert bound["one_chain_bytes"] >= (bound["p_attempt_bytes"]
                                        + bound["i_attempt_bytes"])
    actual = sum(path.stat().st_size for path in chain["root"].rglob("*")
                 if path.is_file())
    assert actual <= bound["one_chain_bytes"]
    assert bound["within_cap"] is True


def test_f_settlement_requires_a_complete_worker_snapshot(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    budget = {"remaining_s": 10.0, "reserve_s": 1.0,
              "attempts": 2, "attempts_used": 0}
    A.pre_register_attempt(root, "F", "attempt_001", A._now(), scope={},
                           budget=budget)
    A.reserve_stage_calls(root, "F", "attempt_001",
                          {"counts": A.STAGE_NOMINAL["F"]})
    ok, problems = A._settle_f_calls(root, "F", "attempt_001", None)
    assert not ok and problems == ["f_call_settlement_unknown"]
    path = A.stage_dir(root, "F") / "attempt_001" / "call_reservations.json"
    assert A.read_json(path)["consumed"] is None
    partial = {kind: 0 for kind in A.CALL_KINDS}
    partial.pop(next(iter(partial)))
    assert not A._settle_f_calls(root, "F", "attempt_001", partial)[0]
    boolean = {kind: 0 for kind in A.CALL_KINDS}
    boolean[next(iter(boolean))] = True
    assert not A._settle_f_calls(root, "F", "attempt_001", boolean)[0]
    negative = {kind: 0 for kind in A.CALL_KINDS}
    negative[next(iter(negative))] = -1
    assert not A._settle_f_calls(root, "F", "attempt_001", negative)[0]
    assert A.read_json(path)["consumed"] is None
    good = {kind: 0 for kind in A.CALL_KINDS}
    ok, problems = A._settle_f_calls(root, "F", "attempt_001", good)
    assert ok and problems == []
    assert A.read_json(path)["consumed"] == good


def test_worker_inherits_deadline_and_parent_admission(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    budget = {"remaining_s": 10.0, "reserve_s": 1.0,
              "attempts": 2, "attempts_used": 0}
    A.pre_register_attempt(root, "F", "attempt_001", A._now(), scope={},
                           budget=budget)
    A.reserve_stage_calls(root, "F", "attempt_001",
                          {"counts": A.STAGE_NOMINAL["F"]})
    spec = {
        "fixture_dir": str(tmp_path / "missing_fixture"),
        "p_dir": str(tmp_path / "P"), "i_dir": str(tmp_path / "I"),
        "config": TINY_CONFIG, "device": "cuda",
        "tolerance": {"tolerances": A.TOLERANCE_PROPOSALS},
        "remaining_caps": dict(A.STAGE_NOMINAL["F"]),
        "deadline_perf_counter": A._now() - 10.0,
        "reserve_s": 0.0, "run_root": str(root), "stage": "F",
        "attempt": "attempt_001", "registration_sha256": None,
    }
    spec_path = tmp_path / "spec.json"
    A.write_json(spec_path, spec)
    out = tmp_path / "worker"
    assert A._worker_main(str(spec_path), str(out)) == 2
    result = A.read_json(out / "worker_result.json")
    assert result["aborted"] is True
    assert any("f_worker_aborted" in p and "budget_deadline" in p
               for p in result["problems"])
    assert set(result["counts"]["counts"]) == set(A.CALL_KINDS)
    problems = A.unit_admission(root, "F", "attempt_001",
                                {"registration_sha256": "wrong"},
                                deadline=A._now() + 10.0)
    assert "unit_parent_registration" in problems


def test_f_rejects_expired_spawn_without_a_worker(chain, tmp_path):
    _register_f_parent(tmp_path)
    result = A.run_f_stage(
        tmp_path, "F", "attempt_001", A._now(),
        {"remaining_s": 0.0, "reserve_s": 0.0}, chain["provider"],
        A.CallCounter(A.CALL_CAPS), chain["provider"].roster,
        chain["root"] / "P", chain["root"] / "I",
        tolerance=chain["tolerance"], out_dir=tmp_path / "F")
    assert result["complete_evidence"] is False
    assert result["call_delta"] is None
    assert result["worker"]["exit_reason"] == "not_spawned"
    assert any("f_stage_deadline_before_worker" in p
               for p in result["verification_problems"])


def test_f_detects_report_rng_and_duplicate_row_tamper(chain, tmp_path):
    import shutil

    saved_p = tmp_path / "P_tampered"
    shutil.copytree(chain["root"] / "P", saved_p)
    report = A.read_json(saved_p / "p_report.json")
    micro = report["models"]["deterministic"]["updates"]["u1"]["micros"][0]
    micro["loss"] = 123.0
    micro["rng_before"] = {
        key: ("0" * 64 if isinstance(value, str) else value)
        for key, value in micro["rng_before"].items()}
    A.write_json(saved_p / "p_report.json", report)
    saved_i = tmp_path / "I_tampered"
    shutil.copytree(chain["root"] / "I", saved_i)
    rows = A.read_json(saved_i / "i_rows.json")
    rows.append(dict(rows[0]))
    A.write_json(saved_i / "i_rows.json", rows)
    _register_f_parent(tmp_path)
    result = A.run_f_stage(
        tmp_path, "F", "attempt_001", A._now(),
        {"remaining_s": 10 ** 6, "reserve_s": 0.0}, chain["provider"],
        A.CallCounter(A.CALL_CAPS), chain["provider"].roster,
        saved_p, saved_i, tolerance=chain["tolerance"],
        out_dir=tmp_path / "F")
    problems = result["verification_problems"]
    assert result["complete_evidence"] is False
    assert any("report_micro_drift" in p and p.endswith(":loss")
               for p in problems)
    assert any("report_micro_drift" in p and p.endswith(":rng_before")
               for p in problems)
    assert any(("saved_rows_duplicate" in p) or ("saved_i_rows_scope" in p)
               for p in problems)


def test_complete_numeric_mismatch_reaches_f(chain, tmp_path):
    tolerance = dict(A.TOLERANCE_PROPOSALS)
    tolerance["parameter_gradient"] = {"atol": 1e-30, "rtol": 0.0}
    counter = A.CallCounter(A.CALL_CAPS)
    i = A.run_i_stage(
        tmp_path / "chain", "I", "attempt_001", A._now(),
        {"remaining_s": 10 ** 6, "reserve_s": 0.0}, chain["provider"],
        counter, chain["provider"].roster, chain["root"] / "P",
        tolerance={"tolerances": tolerance},
        out_dir=tmp_path / "chain" / "I")
    assert i["failures"]
    assert i["complete_evidence"] is True
    assert i["evidence_failures"] == []
    _register_f_parent(tmp_path / "run")
    f = A.run_f_stage(
        tmp_path / "run", "F", "attempt_001", A._now(),
        {"remaining_s": 10 ** 6, "reserve_s": 0.0}, chain["provider"],
        A.CallCounter(A.CALL_CAPS), chain["provider"].roster,
        chain["root"] / "P", tmp_path / "chain" / "I",
        tolerance={"tolerances": tolerance},
        out_dir=tmp_path / "run" / "F")
    assert f["complete_evidence"] is True
    assert f["failures"]
    assert f["conclusion"] in ("audit_incomplete",
                               "real_training_update_mismatch")


def test_withdrawal_charges_the_cost_floor(tmp_path, monkeypatch):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    attempt = "attempt_001"
    directory = A.stage_dir(root, "F") / attempt
    directory.mkdir(parents=True)
    report = directory / "final.json"
    A.write_json(report, {"conclusion":
                          "real_training_update_covered_contract_passed",
                          "complete_evidence": True,
                          "covered_contract_passed": True})
    budget = {"remaining_s": 20.0, "reserve_s": 1.0,
              "attempts": 2, "attempts_used": 0}
    t0 = A._now()
    A.pre_register_attempt(root, "F", attempt, t0, scope={}, budget=budget)
    seal = A.finish_attempt(
        root, "F", attempt, t0, budget,
        {"complete": True, "report": A.path_for_record(report),
         "summary": A.path_for_record(report)}, status="done")
    assert seal.sealed
    old_cost = seal.row["measured_wall_s"]
    real_upsert = A.upsert_cost
    calls = {"n": 0}

    def slow_upsert(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            time.sleep(0.4)
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(A, "upsert_cost", slow_upsert)
    start = A._now()
    row, problems, proven = A.withdraw_attempt_claim(
        root, "F", attempt, report, seal.row, ["post_veto"],
        reason="post_seal_veto", t0=t0, budget=budget)
    elapsed = A._now() - start
    assert proven and problems == []
    assert row["measured_wall_s"] >= old_cost
    ledger = A.load_ledger(root)[0]
    assert ledger["measured_wall_s"] >= elapsed
    assert A.read_json(report)["complete_evidence"] is False
    assert A.stage_state(root, "F")[0]["status"] == "incomplete"


def test_matrix_veto_retracts_the_disk_claim(tmp_path, monkeypatch):
    monkeypatch.setitem(
        A.CHAIN_NOMINAL, "conditioner_forwards",
        A.CHAIN_NOMINAL["conditioner_forwards"] + 1)
    result = A.synthetic_self_test(tmp_path / "run", config=TINY_CONFIG,
                                   device="cuda")
    assert result["complete_evidence"] is False
    assert result["conclusion"] == "audit_incomplete"
    summary = A.read_json(Path(result["summary_path"]))
    assert summary["complete_evidence"] is False
    assert summary["conclusion"] == "audit_incomplete"
    problems = summary.get("verification_problems") or []
    assert any("full_chain_calls" in p for p in problems)
    assert any("full_chain_calls" in r
               for r in (summary.get("withdrawn_reasons") or []))
    rows = [r for r in A.load_index(tmp_path / "run") if r["stage"] == "F"]
    assert rows and rows[0]["status"] == "incomplete"
    assert rows[0]["complete"] is False
    assert rows[0]["incomplete_reason"] == "pre_seal_veto"
    final_calls = result["calls"]["counts"]
    assert rows[0]["call_balance"]["counts"] == final_calls
    ledger = [e for e in A.load_ledger(tmp_path / "run") if e["stage"] == "F"]
    assert len(ledger) == 1
    assert ledger[0]["call_balance"]["counts"] == final_calls
    prior = A.prior_call_counts(tmp_path / "run", fixed_root=A.HISTORY_ROOT,
                                mode="synthetic")
    assert prior is not None
    assert prior == final_calls
    assert prior["native_adamw_steps"] == A.CHAIN_NOMINAL[
        "native_adamw_steps"]
    assert prior["backbone_forwards"] == A.CHAIN_NOMINAL["backbone_forwards"]
    reservation = A.read_json(A.stage_dir(tmp_path / "run", "F")
                              / "attempt_001" / "call_reservations.json")
    assert reservation["consumed"] is not None


# ---------------------------------------------------------------------------
# second review-round counterexamples: full-domain calibration, withdrawal
# snapshot, main-path deadlines, cotangent groups and the shared reserve
# ---------------------------------------------------------------------------


def test_state_calibration_covers_small_states_and_gate_denominator():
    spec = A.TOLERANCE_PROPOSALS["state_cross_path"]
    # m/v of 1e-17 are legitimate AdamW states; the old absolute mask skipped
    # them while the raw gate still compared them.
    ratio, count = A._state_cross_path_ratio(
        np.array([8.772033343932125e-17]),
        np.array([8.770656915392836e-17]), spec)
    assert count == 1
    assert ratio == pytest.approx(1.569e-4 / spec["rtol"], rel=1e-3)
    # A zero reference with a non-zero production value is unbounded, not
    # skipped: a passed row can never contain one.
    ratio, count = A._state_cross_path_ratio(
        np.array([1e-9]), np.array([0.0]), spec)
    assert not np.isfinite(ratio) and count == 1
    ratio, count = A._state_cross_path_ratio(
        np.zeros(3), np.zeros(3), spec)
    assert ratio == 0.0 and count == 0


def test_withdrawal_stores_the_final_call_balance(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    budget = {"remaining_s": 10.0, "reserve_s": 1.0,
              "attempts": 2, "attempts_used": 0}
    t0 = A._now()
    A.pre_register_attempt(root, "F", "attempt_001", t0, scope={},
                           budget=budget,
                           call_balance=A.CallCounter(A.CALL_CAPS).snapshot())
    prior = A._attempt_row(root, "F", "attempt_001")["call_balance"]
    counts = {kind: int(prior["counts"][kind]) for kind in A.CALL_KINDS}
    counts["backbone_forwards"] += 32
    counts["native_adamw_steps"] += 8
    final = {"schema": "w1_real_training_update_call_budget_v1",
             "counts": counts,
             "within_caps": all(counts[k] <= A.CALL_CAPS[k]
                                for k in A.CALL_KINDS)}
    consumed = {kind: counts[kind] - int(prior["counts"][kind])
                for kind in A.CALL_KINDS}
    assert A._call_balance_delta_problems(prior, final, consumed) == []
    bad = dict(consumed)
    bad["backbone_forwards"] += 1
    assert A._call_balance_delta_problems(prior, final, bad) == [
        "call_balance_delta:backbone_forwards"]
    # A running attempt withdrawal must publish the post-execution snapshot,
    # not the pre-F balance copied from the running row.
    row, _problems, _proven = A.withdraw_attempt_claim(
        root, "F", "attempt_001", None,
        A._attempt_row(root, "F", "attempt_001"), ["call_balance"],
        reason="pre_seal_veto", t0=t0, budget=budget,
        call_balance=final)
    assert row["call_balance"]["counts"] == counts
    index = [r for r in A.load_index(root) if r["stage"] == "F"]
    ledger = [e for e in A.load_ledger(root) if e["stage"] == "F"]
    assert len(index) == 1 and len(ledger) == 1
    assert index[0]["call_balance"]["counts"] == counts
    assert ledger[0]["call_balance"]["counts"] == counts


def test_p_stage_checks_deadline_before_the_backward(tiny_provider, tmp_path,
                                                     monkeypatch):
    from ireset.train.trainer import Trainer

    original = Trainer.accumulate_micro
    seen = {"calls": 0}

    def slow_accumulate(self, *args, **kwargs):
        seen["calls"] += 1
        if seen["calls"] == 1:
            time.sleep(1.2)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "accumulate_micro", slow_accumulate)
    counter = A.CallCounter(A.CALL_CAPS)
    with pytest.raises(TimeoutError) as caught:
        A.run_p_stage(tmp_path, "P", "attempt_001", A._now(),
                      {"remaining_s": 1.0, "reserve_s": 0.0},
                      tiny_provider, counter, tiny_provider.roster,
                      out_dir=tmp_path / "P")
    assert "p_backward_1_0" in str(caught.value)
    assert counter.counts["production_parameter_backward"] == 0
    assert counter.counts["native_adamw_steps"] == 0


def test_stage_unit_guard_admits_only_the_running_parent(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    budget = {"remaining_s": 10.0, "reserve_s": 0.0,
              "attempts": 2, "attempts_used": 0}
    A.pre_register_attempt(root, "P", "attempt_001", A._now(), scope={},
                           budget=budget, registration_sha="reg")
    deadline = A._now() + 10.0
    guard = A._stage_unit_guard(root, "P", "attempt_001",
                                registration_sha="reg", deadline=deadline)
    guard("running_ok")
    A.finish_attempt(root, "P", "attempt_001", A._now(), budget,
                     {"complete": True}, status="done")
    with pytest.raises(TimeoutError) as caught:
        guard("after_seal")
    assert "unit_parent_not_running" in str(caught.value)
    unknown = A._stage_unit_guard(root, "P", "attempt_002",
                                  registration_sha="reg",
                                  deadline=deadline)
    with pytest.raises(TimeoutError) as caught:
        unknown("missing")
    assert "unit_parent_not_registered" in str(caught.value)


def test_i_stage_checks_deadline_before_reference_context(chain, tmp_path):
    counter = A.CallCounter(A.CALL_CAPS)
    with pytest.raises(TimeoutError) as caught:
        A.run_i_stage(
            tmp_path / "I", "I", "attempt_001", A._now() - 10.0,
            {"remaining_s": 1.0, "reserve_s": 0.0}, chain["provider"],
            counter, chain["provider"].roster, chain["root"] / "P",
            tolerance={"tolerances": A.TOLERANCE_PROPOSALS},
            out_dir=tmp_path / "I")
    assert "i_reference_context" in str(caught.value)
    assert counter.counts["reference_mj_forward"] == 0


def test_cotangent_group_gate_rejects_a_zeroed_joint_group(chain):
    ref = np.ones((4, 60, 38), np.float64)
    ref[..., 9:] = 1e-9
    got = ref.copy()
    got[..., 9:] = 0.0
    spec = A.TOLERANCE_PROPOSALS["output_cotangent"]
    whole = A.compare_gradient(got, ref, spec, name="cot")
    assert whole["ok"] is True
    rec = A.compare_cotangent_gradient(got, ref, spec, name="cot")
    assert rec["ok"] is False
    assert "cot:group_joint" in rec["group_gate_problems"]
    joint = rec["groups"]["group_gates"]["joint"]
    assert joint["production_norm"] == 0.0
    assert joint["reference_norm"] > 1e-10
    assert joint["ok"] is False
    missing = A.compare_cotangent_gradient(None, ref, spec, name="missing")
    assert missing["ok"] is False
    assert "missing:group_shape" in missing["group_gate_problems"]
    # The I rows carry the three group gates and their verdict binds them.
    rows = [r for r in chain["i"]["rows"]
            if str(r["key"]).startswith("cotangent:")]
    assert rows
    for row in rows:
        assert set(row["group_gates"]) == {"pos", "rot6d", "joint"}
        assert row["status"] == "synthetic_verified"
        assert row["group_gate_problems"] == []
    tampered = [dict(r) for r in rows]
    tampered[0] = dict(tampered[0])
    gates = dict(tampered[0]["group_gates"])
    gates["joint"] = {**gates["joint"], "relative_l2": 123.0}
    tampered[0]["group_gates"] = gates
    drift = A._row_compare_problems(rows, tampered,
                                    [r["key"] for r in rows])
    assert any(p.startswith("row_drift:cotangent") for p in drift)


def test_f_stage_deducts_the_reserve_once(tmp_path, monkeypatch, chain):
    _register_f_parent(tmp_path)
    counts = {kind: 0 for kind in A.CALL_KINDS}
    import os
    import subprocess

    def fake_run(cmd, **kwargs):
        worker = Path(cmd[-1])
        worker.mkdir(parents=True, exist_ok=True)
        A.write_json(worker / "worker_result.json", {
            "schema": "w1_real_training_update_f_worker_v1",
            "pid": os.getpid() + 1, "ppid": os.getpid(),
            "python": "fake", "runtime": {}, "fixture_dir": None,
            "p_replay": None, "i_replay": None, "i_replay_rows": 0,
            "schedule_rebuild_ok": False, "npz_problems": [],
            "row_problems": [], "report_problems": [],
            "call_matrix_problems": [], "problems": ["fake_worker"],
            "aborted": True,
            "counts": {"schema": A.CallCounter.schema, "counts": counts,
                       "within_caps": True},
            "complete_evidence": False, "failures": [], "gaps": [],
            "created_wall": "fake",
        })
        return subprocess.CompletedProcess(cmd, 2, "", "")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    deadline_before = A._now()
    result = A.run_f_stage(
        tmp_path, "F", "attempt_001", A._now(),
        {"remaining_s": 300.0, "reserve_s": 180.0}, chain["provider"],
        A.CallCounter(A.CALL_CAPS), chain["provider"].roster,
        chain["root"] / "P", chain["root"] / "I",
        tolerance=chain["tolerance"], out_dir=tmp_path / "F")
    problems = result["verification_problems"]
    assert "f_stage_deadline_before_worker" not in problems
    assert result["worker"]["exit_reason"] == "nonzero_exit"
    assert result["worker"]["exit_reason"] != "not_spawned"
    spec = A.read_json(tmp_path / "F" / "worker_spec.json")
    remaining = spec["deadline_perf_counter"] - deadline_before
    assert 110.0 < remaining <= 121.0
    assert not any("reservation_settle" in p for p in problems)


# ---------------------------------------------------------------------------
# third review-round counterexamples: per-gate calibration applicability,
# the group deep-noise band and the remaining I-stage deadline units
# ---------------------------------------------------------------------------


def test_calibration_uses_per_gate_applicability():
    row = {
        "status": "synthetic_verified",
        "classification": "audit_incomplete",
        "failure": None,
        "gaps": ["parameter_delta_precision_gap"],
        "cross_path_flags": {"m_ok": True, "v_ok": True,
                             "theta_ok": True, "delta_ok": True},
        "state_ratio_applicable": 128,
        "max_state_ratio": 0.28962621440972725,
        "max_theta_ratio": 0.01,
        "max_delta_ratio": 0.2977519726068185,
        "max_same_delta_ratio": 0.2313849592711368,
        "same_input_formula_passed": True,
    }
    obs = A._row_ratio_observations(row)
    assert obs["state_cross_path"] == pytest.approx(0.28962621440972725)
    assert obs["delta_cross_path"] == pytest.approx(0.2977519726068185)
    assert obs["theta_cross_path"] == pytest.approx(0.01)
    assert obs["delta_same_input"] == pytest.approx(0.2313849592711368)
    # A gate that did not pass is not a positive observation, even when the
    # row has no direct failure.
    failed = dict(row, cross_path_flags={
        "m_ok": False, "v_ok": False, "theta_ok": True, "delta_ok": False})
    failed_obs = A._row_ratio_observations(failed)
    assert "state_cross_path" not in failed_obs
    assert "delta_cross_path" not in failed_obs
    assert failed_obs["theta_cross_path"] == pytest.approx(0.01)
    # Direct failures are not numeric-sensitivity samples.
    assert A._row_ratio_observations(dict(row, failure="same_input_formula")) == {}
    assert A._row_ratio_observations(dict(row, status="none_parameter")) == {}
    # An applicable-but-unbounded state may never be silently dropped.
    unbounded = dict(row, max_state_ratio=None, state_ratio_unbounded=True)
    assert math.isinf(
        A._row_ratio_observations(unbounded)["state_cross_path"])


def test_calibration_chain_applies_the_4x_rule_to_every_gate(chain):
    rows = [r for r in chain["i"]["rows"]
            if str(r["key"]).startswith("adamw:")
            and r.get("status") != "none_parameter"]
    maxima: dict[str, float] = {}
    for row in rows:
        for name, value in A._row_ratio_observations(row).items():
            maxima[name] = max(maxima.get(name, 0.0), value)
    for name in ("theta_cross_path", "state_cross_path",
                 "delta_cross_path", "delta_same_input"):
        assert math.isfinite(maxima[name])
        assert 4.0 * maxima[name] <= 1.0, (name, maxima[name])
    # The precision-gap rows must be inside the statistic: filtering by the
    # whole-row classification leaves both maxima below 0.1.
    assert maxima["state_cross_path"] > 0.1
    assert maxima["delta_cross_path"] > 0.1
    precision = [r for r in rows
                 if r.get("classification") == "audit_incomplete"
                 and r.get("gaps") == ["parameter_delta_precision_gap"]
                 and (r.get("cross_path_flags") or {}).get("m_ok")
                 and (r.get("cross_path_flags") or {}).get("v_ok")]
    assert precision
    for row in precision:
        obs = A._row_ratio_observations(row)
        assert obs["state_cross_path"] == pytest.approx(
            float(row["max_state_ratio"]))


def test_cotangent_group_tiny_gate_requires_both_norms_in_the_deep_band():
    spec = A.TOLERANCE_PROPOSALS["output_cotangent"]
    ref = np.ones((2, 38), np.float64)
    got = ref.copy()
    ref[..., 9:] = 1e-13
    got[..., 9:] = 1e-8
    whole = A.compare_gradient(got, ref, spec, name="cot")
    assert whole["ok"] is True
    rec = A.compare_cotangent_gradient(got, ref, spec, name="cot")
    assert rec["ok"] is False
    assert "cot:tiny_group_joint" in rec["group_gate_problems"]
    joint = rec["groups"]["group_gates"]["joint"]
    assert joint["reference_norm"] < A.ZERO_GRADIENT_NOISE_SCALE
    assert joint["production_norm"] > A.ZERO_GRADIENT_NOISE_SCALE
    assert joint["tiny_noise_scale"] == A.ZERO_GRADIENT_NOISE_SCALE
    assert joint["ok"] is False
    # A genuinely deep-band production group still uses the absolute floor.
    ref_deep = ref.copy()
    got_deep = ref.copy()
    ref_deep[..., 9:] = 1e-13
    got_deep[..., 9:] = 2e-13
    deep = A.compare_cotangent_gradient(got_deep, ref_deep, spec, name="deep")
    assert deep["ok"] is True
    assert deep["group_gate_problems"] == []
    assert deep["groups"]["group_gates"]["joint"]["ok"] is True


def _install_fake_clock(monkeypatch):
    import types
    clock = {"now": 0.0}
    monkeypatch.setattr(A, "time", types.SimpleNamespace(
        perf_counter=lambda: clock["now"],
        strftime=time.strftime, time=time.time))
    return clock


def test_i_stage_checks_deadline_before_output_vjp(chain, tmp_path,
                                                   monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    original_evaluate = A.ReferenceContext.evaluate
    calls = {"n": 0}

    def expire_after_fk(self, *args, **kwargs):
        calls["n"] += 1
        result = original_evaluate(self, *args, **kwargs)
        if calls["n"] == 1:
            clock["now"] = 200.0
        return result

    monkeypatch.setattr(A.ReferenceContext, "evaluate", expire_after_fk)
    counter = A.CallCounter(A.CALL_CAPS)
    with pytest.raises(TimeoutError) as caught:
        A.run_i_stage(
            tmp_path / "I", "I", "attempt_001", 0.0,
            {"remaining_s": 100.0, "reserve_s": 0.0}, chain["provider"],
            counter, chain["provider"].roster, chain["root"] / "P",
            tolerance={"tolerances": A.TOLERANCE_PROPOSALS},
            out_dir=tmp_path / "I")
    assert "i_output_vjp_1_0" in str(caught.value)
    assert counter.counts["independent_output_vjp"] == 0
    assert counter.counts["independent_total_vjp"] == 0


def test_i_stage_checks_deadline_before_clip_and_counts_after_update(
        chain, tmp_path, monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    original_accumulate = A._accumulate_f32
    seen = {"n": 0}

    def expire_after_last_vjp(*args, **kwargs):
        seen["n"] += 1
        result = original_accumulate(*args, **kwargs)
        if seen["n"] == len(A.MODELS) * len(A.UPDATES) * A.MICRO_COUNT:
            clock["now"] = 200.0
        return result

    monkeypatch.setattr(A, "_accumulate_f32", expire_after_last_vjp)
    clip_calls = {"n": 0}
    original_clip = A._clip_independent

    def counted_clip(*args, **kwargs):
        clip_calls["n"] += 1
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(A, "_clip_independent", counted_clip)
    counter = A.CallCounter(A.CALL_CAPS)
    with pytest.raises(TimeoutError) as caught:
        A.run_i_stage(
            tmp_path / "I", "I", "attempt_001", 0.0,
            {"remaining_s": 100.0, "reserve_s": 0.0}, chain["provider"],
            counter, chain["provider"].roster, chain["root"] / "P",
            tolerance={"tolerances": A.TOLERANCE_PROPOSALS},
            out_dir=tmp_path / "I")
    assert "i_clip_2" in str(caught.value)
    assert counter.counts["independent_total_vjp"] == (
        len(A.MODELS) * len(A.UPDATES) * A.MICRO_COUNT)
    assert clip_calls["n"] == 3
    assert counter.counts["independent_clip_formula"] == 3
    assert counter.counts["explicit_reference_updates"] == 3
    assert counter.counts["diagnostic_parameter_update_events"] == 3
    assert counter.counts["optimizer_formula_checks"] == 3
