"""Synthetic acceptance tests for the W1 real training-update audit (I/C).

These tests exercise the full synthetic harness: the shared Trainer update
unit, the frozen effective-float32 AdamW graph, the update-delta and state
zero gates, the cross-path propagation classification, R0/P/I/F accounting,
and the fresh-process F replay.  No real checkpoint, dataset, cache, G1 asset
or MuJoCo model is read.
"""

from __future__ import annotations

import json
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
                       "cross_path_sensitivity_gap", "direct_failure"}
    assert "direct_failure" not in classes
    assert any(r["classification"] == "passed" for r in rows)


def test_f_worker_replays_bitwise_and_verifies_rows(chain, tmp_path):
    provider = chain["provider"]
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
