"""Round-16 regressions: tiny offline synthetic fixtures only."""

from __future__ import annotations

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
