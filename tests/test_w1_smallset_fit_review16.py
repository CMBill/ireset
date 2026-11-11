"""Round-16 regressions: tiny offline synthetic fixtures only."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_smallset_fit_diagnostic as diag
from ireset.train import trainer as trainer_mod


@pytest.fixture(autouse=True)
def _limit_torch_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _driver(root: Path, *, model_type: str = "deterministic"):
    world = diag.make_synthetic_world(seed=20261118, n_windows=1, d_model=16)
    plan = diag.FitPlan("D1", steps=1, probe_every=1, synthetic=True,
                        d_model=16, n_layers=1, n_heads=4, ffn_mult=2,
                        lr=0.02, model_type=model_type)
    view = world.view_for("D1")
    clock = diag._FakeClock(1.0)
    budget = diag.StageBudget("D1", 100.0, 10.0, t0=0.0, clock=clock)
    stage_dir = root / "d1"
    trainer = world.build_trainer(plan, stage_dir, budget.work_deadline,
                                  train_ds=view["ds"])
    driver = diag.FitDriver(
        trainer=trainer, plan=plan, registration=view["registration"],
        loss_fn=world.loss_fn, decoder=world.decoder,
        probe_batch=view["probe_batch"], budget=budget,
        runtime_guard=diag.SyntheticRuntimeGuard(), identity={"synthetic": True},
        stage_dir=stage_dir, training_gen=torch.Generator().manual_seed(plan.seed),
        dataset_window_keys=view["ds"].window_keys, authorization_includes_d6=True)
    return world, plan, driver, stage_dir, clock, budget


def _run(driver, clock):
    with patch.object(trainer_mod, "time", types.SimpleNamespace(perf_counter=clock)):
        return driver.run()


@pytest.mark.parametrize("model_type", ["deterministic", "diffusion"])
def test_all_trainable_parameters_receive_state_in_normal_update(tmp_path, model_type):
    _, _, driver, _, clock, _ = _driver(tmp_path, model_type=model_type)
    summary = _run(driver, clock)
    assert summary["local_complete_evidence"] is True, summary.get("failure")
    assert summary["call_matrix"]["problems"] == []
    assert all(p.requires_grad and p.grad is not None and p in driver.trainer.opt.state
               for _, p in driver.trainer._named_parameters())
    assert diag._optimizer_state_problems(driver.trainer.opt, 1) == []


@pytest.mark.parametrize("name", ["model.net.root_out.weight",
                                  "conditioner.kpt_proj.weight"])
def test_frozen_parameter_is_refused_before_any_forward_or_update(tmp_path, name):
    _, _, driver, _, clock, _ = _driver(tmp_path)
    target = dict(driver.trainer._named_parameters())[name]
    before = target.detach().clone()
    target.requires_grad_(False)
    summary = _run(driver, clock)
    assert summary["local_complete_evidence"] is False
    assert "optimizer_parameter_frozen" in summary["failure"]["message"]
    assert summary["counters"]["conditioner_forward_calls"] == 0
    assert summary["counters"]["adamw_step_calls"] == 0
    assert any(p is target for p in driver.trainer.opt.param_groups[0]["params"])
    assert target not in driver.trainer.opt.state
    assert torch.equal(before, target.detach())


def test_missing_gradient_and_optimizer_state_are_not_silently_accepted(tmp_path):
    _, _, driver, _, clock, _ = _driver(tmp_path)
    trainer = driver.trainer
    target = dict(trainer._named_parameters())["model.net.root_out.weight"]
    actual_micro = trainer.accumulate_micro
    calls = {"micro": 0, "step": 0}
    actual_step = trainer.opt.step

    def drop_gradient(*args, **kwargs):
        result = actual_micro(*args, **kwargs)
        calls["micro"] += 1
        if calls["micro"] == driver.plan.grad_accum:
            target.grad = None
        return result

    def step(*args, **kwargs):
        calls["step"] += 1
        return actual_step(*args, **kwargs)

    with patch.object(trainer, "accumulate_micro", drop_gradient), \
            patch.object(trainer.opt, "step", step):
        summary = _run(driver, clock)
    assert summary["local_complete_evidence"] is False
    assert "clip_coverage_grad_missing" in summary["failure"]["message"]
    assert calls == {"micro": 4, "step": 0}
    assert summary["counters"]["clip_calls"] == 0

    # A missing AdamW entry after a completed update is also a refusal; an
    # empty state is permitted only before the first step.
    _, _, healthy, _, healthy_clock, _ = _driver(tmp_path / "healthy")
    assert _run(healthy, healthy_clock)["local_complete_evidence"] is True
    param = dict(healthy.trainer._named_parameters())["model.net.root_out.weight"]
    del healthy.trainer.opt.state[param]
    assert any("optimizer_state_roster_mismatch:missing1" in problem
               for problem in diag._optimizer_state_problems(healthy.trainer.opt, 1))
    for _, param in healthy.trainer._named_parameters():
        param.grad.zero_()
    healthy._check_clip_coverage_pre()
    healthy._check_clip_coverage_post()  # present zero gradients are legal
    healthy._check_clip_coverage_pre()
    param.grad = None
    with pytest.raises(diag.FitDiagnosticError, match="clip_coverage_grad_side"):
        healthy._check_clip_coverage_post()


def test_parameter_frozen_mid_step_is_refused_before_update(tmp_path):
    _, _, driver, _, clock, _ = _driver(tmp_path)
    trainer = driver.trainer
    target = dict(trainer._named_parameters())["conditioner.kpt_proj.weight"]
    original = trainer.accumulate_micro
    calls = {"micro": 0, "step": 0}
    actual_step = trainer.opt.step

    def freeze_after_micro(*args, **kwargs):
        value = original(*args, **kwargs)
        calls["micro"] += 1
        if calls["micro"] == 1:
            target.requires_grad_(False)
        return value

    def step(*args, **kwargs):
        calls["step"] += 1
        return actual_step(*args, **kwargs)

    with patch.object(trainer, "accumulate_micro", freeze_after_micro), \
            patch.object(trainer.opt, "step", step):
        summary = _run(driver, clock)
    assert summary["local_complete_evidence"] is False
    assert "optimizer_parameter_frozen" in summary["failure"]["message"]
    assert calls == {"micro": 1, "step": 0}


@pytest.mark.parametrize("injected", [False, True])
def test_pre_backward_runtime_read_is_charged(tmp_path, injected):
    _, _, driver, stage_dir, clock, budget = _driver(tmp_path)
    actual_check = driver.runtime_guard.check_unit
    actual_backward = torch.Tensor.backward
    calls = {"injections": 0, "backward": 0, "late": 0}

    def check(point):
        actual_check(point)
        if injected and point == "step_1_micro_1_pre_backward":
            calls["injections"] += 1
            clock.t = 91.0

    def backward(tensor, *args, **kwargs):
        calls["backward"] += 1
        calls["late"] += int(clock.t >= budget.work_deadline)
        return actual_backward(tensor, *args, **kwargs)

    with patch.object(driver.runtime_guard, "check_unit", check), \
            patch.object(torch.Tensor, "backward", backward):
        summary = _run(driver, clock)
    if not injected:
        assert summary["local_complete_evidence"] is True
        assert calls == {"injections": 0, "backward": 4, "late": 0}
        return
    assert summary["local_complete_evidence"] is False
    assert summary["failure"]["type"] == "BudgetExceeded"
    assert "pre_backward_post_runtime" in summary["failure"]["message"]
    assert calls == {"injections": 1, "backward": 0, "late": 0}
    assert summary["counters"]["backward_calls"] == 0
    assert summary["counters"]["adamw_step_calls"] == 0
    finished = diag.finish_stage(stage_dir, summary, budget)
    assert finished["summary_doc"]["local_complete_evidence"] is False
    assert finished["return_coverage"]["covered"] is True
    assert (stage_dir / "return_observation.json").is_file()


@pytest.mark.parametrize("variant", ["control", "drift_after_metric",
                                     "drift_after_decode", "slow_decode_gate",
                                     "slow_fk_gate"])
def test_teacher_reference_precision_and_deadline_admission(tmp_path, variant):
    _, _, driver, stage_dir, clock, budget = _driver(tmp_path)
    runtime_before = (torch.backends.cuda.matmul.allow_tf32,
                      torch.backends.cudnn.allow_tf32,
                      torch.get_float32_matmul_precision())
    calls = {"injections": 0, "decode": 0, "fk": 0,
             "late_decode": 0, "late_fk": 0, "bad_fk": 0}
    try:
        driver.runtime_guard = diag.ProductionRuntimeGuard(expected_device="cpu")
        assert not driver.runtime_guard.apply_problems
        actual_check = driver.runtime_guard.check_unit
        actual_metric = driver.decoder.decode
        actual_decode = driver._real_decode_clip
        actual_fk = driver._real_fk.site_world_positions

        def check(point):
            actual_check(point)
            target = {"slow_decode_gate": "teacher_window_w0_prod_decode",
                      "slow_fk_gate": "teacher_window_w0_fk"}.get(variant)
            if point == target and calls["injections"] == 0:
                calls["injections"] += 1
                clock.t = 91.0

        def metric(*args, **kwargs):
            value = actual_metric(*args, **kwargs)
            if variant == "drift_after_metric" and calls["injections"] == 0:
                calls["injections"] += 1
                torch.backends.cuda.matmul.allow_tf32 = True
            return value

        def decode(*args, **kwargs):
            calls["decode"] += 1
            calls["late_decode"] += int(clock.t >= budget.work_deadline)
            value = actual_decode(*args, **kwargs)
            if variant == "drift_after_decode" and calls["injections"] == 0:
                calls["injections"] += 1
                torch.backends.cuda.matmul.allow_tf32 = True
            return value

        def fk(*args, **kwargs):
            calls["fk"] += 1
            calls["late_fk"] += int(clock.t >= budget.work_deadline)
            calls["bad_fk"] += int(torch.backends.cuda.matmul.allow_tf32)
            return actual_fk(*args, **kwargs)

        driver._real_decode_clip = decode
        with patch.object(driver.runtime_guard, "check_unit", check), \
                patch.object(driver.decoder, "decode", metric), \
                patch.object(driver._real_fk, "site_world_positions", fk):
            summary = _run(driver, clock)
    finally:
        torch.set_float32_matmul_precision(runtime_before[2])
        torch.backends.cuda.matmul.allow_tf32 = runtime_before[0]
        torch.backends.cudnn.allow_tf32 = runtime_before[1]

    assert calls["late_decode"] == calls["late_fk"] == calls["bad_fk"] == 0
    if variant == "control":
        assert summary["local_complete_evidence"] is True, summary.get("failure")
        assert calls["decode"] > 0 and calls["fk"] > 0
        return
    assert calls["injections"] == 1
    assert summary["local_complete_evidence"] is False
    assert summary["counters"]["conditioner_forward_calls"] == 0
    assert summary["counters"]["adamw_step_calls"] == 0
    if variant in ("drift_after_metric", "slow_decode_gate"):
        assert calls["decode"] == 0
        assert summary["counters"]["production_decode_calls"] == 0
    else:
        assert calls["decode"] == 1
        assert summary["counters"]["production_decode_calls"] == 1
    assert calls["fk"] == summary["counters"]["production_fk_batch_calls"] == 0
    if variant.startswith("drift"):
        assert "runtime_drift" in summary["failure"]["message"]
    else:
        assert summary["failure"]["type"] == "BudgetExceeded"
        assert "post_runtime" in summary["failure"]["message"]
    finished = diag.finish_stage(stage_dir, summary, budget)
    assert finished["summary_doc"]["local_complete_evidence"] is False
    assert (stage_dir / "return_observation.json").is_file()


@pytest.mark.parametrize("variant", ["control", "missing", "list", "null"])
def test_bad_report_identity_refuses_cold_reference_construction(tmp_path, variant):
    from ireset.model.conditioner import ConditionEncoder
    from ireset.model.deterministic import DeterministicG1

    world, plan, driver, stage_dir, clock, budget = _driver(tmp_path / "source")
    summary = _run(driver, clock)
    assert summary["local_complete_evidence"] is True, summary.get("failure")
    diag.finish_stage(stage_dir, summary, budget)
    copydir = tmp_path / "copy"
    shutil.copytree(stage_dir, copydir)
    if variant != "control":
        report_path = copydir / "stage_report.json"
        report = json.loads(report_path.read_text())
        if variant == "missing":
            del report["identity"]
        else:
            report["identity"] = [] if variant == "list" else None
        report_path.write_text(json.dumps(report))
        diag._full_resign(copydir)
    diag._STATE_CONTRACT_CACHE.clear()
    before_rng = torch.get_rng_state().clone()
    real_model, real_cond = DeterministicG1.__init__, ConditionEncoder.__init__
    events = []
    factory = diag.build_modules_factory(world, plan)
    builds = []

    def model_init(obj, *args, **kwargs):
        events.append("model")
        return real_model(obj, *args, **kwargs)

    def cond_init(obj, *args, **kwargs):
        events.append("conditioner")
        return real_cond(obj, *args, **kwargs)

    def build():
        builds.append(1)
        return factory()

    out_dir = tmp_path / "f"
    with patch.object(DeterministicG1, "__init__", model_init), \
            patch.object(ConditionEncoder, "__init__", cond_init):
        out = diag.run_f_verification(
            stage_specs=[{"stage_dir": copydir, "plan": plan,
                          "registration": world.registration_for("D1"),
                          "decoder": world.decoder, "loss_fn": world.loss_fn,
                          "identity": driver.identity}],
            out_dir=out_dir, budget=diag.StageBudget("F", 300.0, 10.0),
            runtime_guard=diag.SyntheticRuntimeGuard(), build_modules=build,
            authorization={"includes": ["D6"]})
    doc = out["summary_doc"]
    assert json.loads((out_dir / "f_report.json").read_text()) == doc
    assert (out_dir / "return_observation.json").is_file()
    if variant == "control":
        assert doc["complete_evidence"] is True
        assert events == ["model", "conditioner", "model", "conditioner"]
        assert builds == [1]
    else:
        assert events == builds == []
        assert torch.equal(before_rng, torch.get_rng_state())
        assert doc["verdict"] == "incomplete" and doc["complete_evidence"] is False
        assert "stage_report_identity_invalid" in doc["stages"][0]["local_reverify_problems"]
