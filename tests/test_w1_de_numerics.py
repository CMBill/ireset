"""Focused CPU checks for Trainer/generator failures and accepted numerical parity."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.data.schema import ReferenceCondition
from ireset.diffusion.generator import DiffusionG1
from ireset.kinematics.rotation_adapter import rot6d_to_mat_torch
from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1
from ireset.model.generator import G1Generator, GenerationNumericalError
from ireset.model.tokens import RootStats, StateTokenizer, mat_to_quat_torch
from ireset.train.losses import LossAssembler, LossWeights
from ireset.train.trainer import (
    TrainConfig, Trainer, TrainingBudgetExceeded, TrainingNumericalError,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tokenizer():
    return StateTokenizer(np.tile([-2.0, 2.0], (29, 1)),
                          RootStats(np.array([0.1, -0.2, 0.8]), np.array([0.4, 0.3, 0.15])))


class Windows(Dataset):
    def __init__(self, n=8, F=3):
        g = torch.Generator().manual_seed(123)
        self.data = {
            "x0": torch.randn(n, F, 38, generator=g) * 0.1,
            "kps_obj": torch.randn(n, F, 52, 3, generator=g),
            "kps_clip": torch.randn(n, F, 52, 3, generator=g),
            "box_t_clip": torch.randn(n, F, 3, generator=g),
            "box_quat": torch.zeros(n, F, 4), "contact": torch.zeros(n, F, 4),
            "wfc": torch.eye(4).repeat(n, 1, 1),
        }
        self.data["x0"][..., 3:9] = torch.tensor([1., 0, 0, 0, 1, 0])
        self.data["box_quat"][..., 0] = 1

    def __len__(self):
        return len(self.data["x0"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.data.items()}


def cfg(model_type="deterministic", **kw):
    out = TrainConfig(model_type=model_type, d_model=16, n_layers=1, n_heads=2,
                      ffn_mult=2, time_window=3, device="cpu", micro_batch=2,
                      grad_accum=2, num_workers=0, steps=2, val_every=1,
                      log_every=1000, seed=11, val_seed=13,
                      loss=LossWeights(w_contact=0.0))
    for k, v in kw.items():
        setattr(out, k, v)
    return out


def trainer(tmp_path, **kw):
    return Trainer(cfg(**kw), tokenizer(), Windows(), Windows(3), tmp_path)


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
    assert a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def formal_callback(macros=(0.5,), mutate=None, with_deadline=False):
    calls = []

    def produce(model, conditioner, deadline):
        value = macros[min(len(calls), len(macros) - 1)]
        calls.append(deadline)
        rows = [{"demo": d, "T": 9, "status": "ok", "numerical_valid": True,
                 "mpjpe_m": value} for d in ("a", "b")]
        out = {"n_expected": 2, "n_ok": 2, "macro_mpjpe_m": value, "per_demo": rows}
        if mutate:
            mutate(out)
        return out

    if with_deadline:
        def callback(model, conditioner, budget_deadline=None):
            return produce(model, conditioner, budget_deadline)
    else:
        def callback(model, conditioner):
            return produce(model, conditioner, None)
    callback.expected_demos = ["a", "b"]
    callback.demo_lengths = {"a": 9, "b": 9}
    callback.calls = calls
    return callback


# A frozen normal-loop reference: pre-hardening initialization, exact noise
# sampling, sample-weighted validation and AdamW arithmetic. It intentionally
# contains no new guards or evidence code, so equality detects changed math/RNG.
def accepted_reference(c, tk, train_ds, val_ds):
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    if c.model_type == "diffusion":
        model = DiffusionG1(num_train_timesteps=c.num_train_timesteps,
                            beta_schedule=c.beta_schedule, beta_start=c.beta_start,
                            beta_end=c.beta_end, **c.backbone_kwargs())
    else:
        model = DeterministicG1(**c.backbone_kwargs())
    conditioner = ConditionEncoder(c.d_model)
    loss_fn = LossAssembler(tk, c.loss, "cpu")
    opt = torch.optim.AdamW([*model.parameters(), *conditioner.parameters()],
                            lr=c.lr, weight_decay=c.weight_decay)
    loader = DataLoader(train_ds, batch_size=c.micro_batch, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(c.seed), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=c.micro_batch, shuffle=False,
                            num_workers=0, generator=torch.Generator().manual_seed(c.seed), drop_last=False)
    initial = {"model": copy.deepcopy(model.state_dict()),
               "conditioner": copy.deepcopy(conditioner.state_dict()), "rng": rng_snapshot()}

    def forward(batch, g=None):
        memory = conditioner(batch)
        if c.model_type == "diffusion":
            if g is None:
                t = torch.randint(0, c.num_train_timesteps, (len(batch["x0"]),), device="cpu")
                noise = torch.randn_like(batch["x0"])
            else:
                t = torch.randint(0, c.num_train_timesteps, (len(batch["x0"]),), generator=g)
                noise = torch.randn(batch["x0"].shape, generator=g)
            pred = model(model.add_noise(batch["x0"], noise, t), t, memory)
        else:
            pred = model(memory, batch["x0"].shape[1])
        return loss_fn(pred, batch)

    history, vals = [], []
    it = iter(loader)
    model.train(); conditioner.train()
    for step in range(1, c.steps + 1):
        opt.zero_grad(set_to_none=True)
        acc = {}
        for _ in range(c.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            loss, parts = forward(batch)
            (loss / c.grad_accum).backward()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.) + v / c.grad_accum
        gnorm = torch.nn.utils.clip_grad_norm_([*model.parameters(), *conditioner.parameters()], c.grad_clip)
        opt.step()
        history.append({**{k: round(v, 6) for k, v in acc.items()}, "gnorm": float(gnorm)})
        if step % c.val_every == 0 or step == c.steps:
            model.eval(); conditioner.eval()
            g = torch.Generator().manual_seed(c.val_seed if c.val_seed is not None else c.seed)
            acc, n = {}, 0
            with torch.no_grad():
                for batch in val_loader:
                    _, parts = forward(batch, g)
                    w = len(batch["x0"])
                    for k, v in parts.items():
                        acc[k] = acc.get(k, 0.) + v * w
                    n += w
            vals.append({k: v / max(n, 1) for k, v in acc.items()})
            model.train(); conditioner.train()
    return initial, model, conditioner, history, vals, rng_snapshot(), loader, val_loader


@pytest.mark.parametrize("model_type", ["deterministic", "diffusion"])
def test_first_success_training_initialization_losses_probe_and_rng_match_accepted_loop(tmp_path, model_type):
    c, tk = cfg(model_type, steps=3), tokenizer()
    ds, vd = Windows(7), Windows(3)
    # Python RNG was never seeded by Trainer; keep its independent starting state.
    random.seed(81)
    initial, model, conditioner, hist, vals, final_rng, loader, val_loader = accepted_reference(c, tk, ds, vd)
    tr = Trainer(c, tk, ds, vd, tmp_path)
    for module, reference in ((tr.model, initial["model"]), (tr.conditioner, initial["conditioner"])):
        assert all(torch.equal(v, reference[k]) for k, v in module.state_dict().items())
    assert_rng_equal(rng_snapshot(), initial["rng"])
    summary = tr.run()
    assert summary["complete"] is True and summary["steps"] == 3
    for new, old in ((tr.model, model), (tr.conditioner, conditioner)):
        assert all(torch.equal(v, old.state_dict()[k]) for k, v in new.state_dict().items())
    saved_hist = json.loads((tmp_path / "loss_history.json").read_text())
    saved_vals = json.loads((tmp_path / "val_history.json").read_text())
    for new, old in zip(saved_hist, hist, strict=True):
        assert all(new[k] == v for k, v in old.items())
    for new, old in zip(saved_vals, vals, strict=True):
        assert all(new[k] == v for k, v in old.items())
    assert_rng_equal(rng_snapshot(), final_rng)
    assert torch.equal(tr.train_loader.generator.get_state(), loader.generator.get_state())
    assert torch.equal(tr.val_loader.generator.get_state(), val_loader.generator.get_state())
    assert summary["train_loader"]["micro_batches_total"] == 6
    assert summary["train_loader"]["samples_total"] == 12
    assert summary["train_loader"]["drop_last_discarded_per_epoch"] == 1


@pytest.mark.parametrize("field", ["x0", "kps_obj", "box_quat", "contact"])
def test_nonfinite_input_stops_before_forward_and_keeps_zero_step_evidence(tmp_path, field):
    tr = trainer(tmp_path)
    tr.train_ds.data[field][:] = float("nan")
    tr.conditioner.forward = lambda batch: pytest.fail("invalid input reached conditioner")
    summary = tr.run()
    assert summary["steps"] == 0
    assert summary["stop_reason"].startswith("nonfinite_input")
    assert summary["failure"]["field"] == f"batch.{field}"
    assert json.loads((tmp_path / "loss_history.json").read_text()) == []
    assert torch.load(tmp_path / "last.pt", weights_only=False)["step"] == 0
    assert summary["train_loader"]["micro_batches_total"] == 1
    assert summary["best_val"] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("kind", ["nan_prediction", "inf_joint", "parallel_rotation", "zero_rotation"])
def test_prediction_checked_before_loss_decode_or_joint_clamp(tmp_path, kind):
    tr = trainer(tmp_path)
    out = torch.zeros(2, 3, 38)
    out[..., 3:9] = torch.tensor([1., 0, 0, 0, 1, 0])
    if kind == "nan_prediction":
        out[0, 0, 0] = float("nan")
    elif kind == "inf_joint":
        out[0, 0, 9] = float("inf")
    elif kind == "parallel_rotation":
        out[..., 6:9] = out[..., 3:6] * 2
    else:
        out[..., 3:6] = 0
    tr.model.forward = lambda *a: out
    tr.loss_fn = lambda *a: pytest.fail("invalid prediction reached loss/decoder")
    summary = tr.run()
    assert summary["steps"] == 0 and summary["complete"] is False
    assert "prediction" in summary["stop_reason"]


def test_all_loss_parts_checked_even_when_total_is_finite(tmp_path):
    tr = trainer(tmp_path)
    orig = tr.loss_fn

    def poisoned(pred, batch):
        loss, parts = orig(pred, batch)
        parts["fk_m2"] = float("nan")
        return loss, parts

    tr.loss_fn = poisoned
    summary = tr.run()
    assert summary["steps"] == 0
    assert summary["stop_reason"].startswith("nonfinite_loss_part")
    assert summary["failure"]["field"] == "parts.fk_m2"
    assert summary["train_loader"]["backward_micro_batches_total"] == 0


def test_backward_gradient_failure_prevents_optimizer_update(tmp_path):
    tr = trainer(tmp_path)
    next(tr.model.parameters()).register_hook(lambda g: torch.full_like(g, float("nan")))
    tr.opt.step = lambda: pytest.fail("invalid gradient reached optimizer")
    summary = tr.run()
    assert summary["steps"] == 0 and summary["stop_reason"].startswith("nonfinite_gradient")
    assert summary["train_loader"]["backward_micro_batches_total"] == 1


def test_parameter_failure_after_update_keeps_actual_step_and_loss_row(tmp_path):
    tr = trainer(tmp_path)
    orig = tr.opt.step

    def poisoned_update():
        orig()
        with torch.no_grad():
            next(tr.model.parameters()).flatten()[0] = float("nan")

    tr.opt.step = poisoned_update
    summary = tr.run()
    assert summary["steps"] == 1 and summary["complete"] is False
    assert len(json.loads((tmp_path / "loss_history.json").read_text())) == 1
    ck = torch.load(tmp_path / "last.pt", weights_only=False)
    assert ck["step"] == 1 and ck["parameters_finite"] is False and ck["checkpoint_usable"] is False
    assert summary["checkpoint_index"][0]["checkpoint_usable"] is False


def test_existing_bad_parameter_stops_before_first_batch(tmp_path):
    tr = trainer(tmp_path)
    with torch.no_grad():
        next(tr.conditioner.parameters()).flatten()[0] = float("inf")
    summary = tr.run()
    assert summary["steps"] == 0 and summary["train_loader"]["micro_batches_total"] == 0
    assert summary["stop_reason"].startswith("nonfinite_params")


def test_training_exception_preserves_last_actual_step_and_original_exception(tmp_path):
    tr = trainer(tmp_path, steps=3, val_every=10)
    original_forward, calls = tr._forward, []
    error = OSError("forward storage unavailable")

    def fail_after_one_step(batch, noise_generator=None):
        calls.append(None)
        if len(calls) == 3:
            raise error
        return original_forward(batch, noise_generator)

    tr._forward = fail_after_one_step
    with pytest.raises(OSError) as caught:
        tr.run()
    assert caught.value is error
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["steps"] == 1 and summary["status"] == "crashed"
    assert caught.value.training_summary["steps"] == 1
    assert len(json.loads((tmp_path / "loss_history.json").read_text())) == 1
    assert torch.load(tmp_path / "last.pt", weights_only=False)["step"] == 1


def test_secondary_checkpoint_exception_does_not_replace_training_exception(tmp_path):
    tr = trainer(tmp_path)
    first, second = RuntimeError("original failure"), OSError("saving failed too")

    def fail_forward(*a, **kw):
        raise first

    def fail_save(*a, **kw):
        raise second

    tr._forward, tr.save = fail_forward, fail_save
    with pytest.raises(RuntimeError) as caught:
        tr.run()
    assert caught.value is first
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["steps"] == 0 and summary["preservation_errors"]
    assert summary["checkpoint_index"] == []
    assert json.loads((tmp_path / "loss_history.json").read_text()) == []


@pytest.mark.parametrize("kind", ["validate", "formal"])
@pytest.mark.parametrize("fail", [False, True])
def test_callbacks_restore_independent_modes_global_rng_and_leave_loader_flow(tmp_path, kind, fail):
    tr = trainer(tmp_path)
    tr.model.eval(); tr.conditioner.train()
    model_child = next(m for m in tr.model.modules() if m is not tr.model)
    model_child.train()
    modes = [(m, m.training) for root in (tr.model, tr.conditioner) for m in root.modules()]
    initial = rng_snapshot()
    train_loader_rng = tr.train_loader.generator.get_state().clone()
    val_loader_rng = tr.val_loader.generator.get_state().clone()

    def consume_random():
        random.random(); np.random.rand(); torch.rand(3)
        if fail:
            raise RuntimeError("callback failed")

    if kind == "validate":
        orig = tr._forward

        def forward(*a, **kw):
            consume_random()
            return orig(*a, **kw)

        tr._forward = forward
        if fail:
            with pytest.raises(RuntimeError):
                tr.validate()
        else:
            assert tr.validate()["complete"]
        # The val loader iterator is still created/consumed, never rolled back.
        assert not torch.equal(val_loader_rng, tr.val_loader.generator.get_state())
    else:
        def callback(model, conditioner):
            consume_random()
            return formal_callback()(model, conditioner)

        callback.expected_demos = ["a", "b"]
        callback.demo_lengths = {"a": 9, "b": 9}
        tr.formal_eval = callback
        tr.formal_expected_demos = tuple(callback.expected_demos)
        tr.formal_demo_lengths = callback.demo_lengths
        assert tr._run_formal_eval()["complete"] is (not fail)
        assert torch.equal(val_loader_rng, tr.val_loader.generator.get_state())
    assert all(m.training == mode for m, mode in modes)
    assert_rng_equal(initial, rng_snapshot())
    assert torch.equal(train_loader_rng, tr.train_loader.generator.get_state())


@pytest.mark.parametrize("bad", ["missing", "duplicate", "foreign", "nan", "bool", "wrong_T", "not_numerical", "wrong_macro", "count_only"])
def test_formal_best_requires_exact_roster_and_finite_per_demo_evidence(tmp_path, bad):
    def mutate(out):
        if bad == "missing":
            out["per_demo"].pop()
        elif bad == "duplicate":
            out["per_demo"][1] = dict(out["per_demo"][0])
        elif bad == "foreign":
            out["per_demo"][1]["demo"] = "foreign"
        elif bad == "nan":
            out["per_demo"][0]["mpjpe_m"] = float("nan")
        elif bad == "bool":
            out["per_demo"][0]["mpjpe_m"] = True
        elif bad == "wrong_T":
            out["per_demo"][0]["T"] = 8
        elif bad == "not_numerical":
            out["per_demo"][0]["numerical_valid"] = False
        elif bad == "wrong_macro":
            out["macro_mpjpe_m"] = 0.1
        else:
            out["per_demo"] = []

    callback = formal_callback(mutate=mutate)
    tr = Trainer(cfg(steps=1), tokenizer(), Windows(), Windows(3), tmp_path,
                 formal_eval=callback, formal_eval_every=1)
    summary = tr.run()
    assert summary["best_formal_step"] is None
    assert summary["best_formal_macro_mpjpe_m"] is None
    assert not (tmp_path / "best.pt").exists()
    formal = json.loads((tmp_path / "formal_history.json").read_text())[0]
    assert formal["complete"] is False and formal["evidence_gaps"]
    json.dumps(formal, allow_nan=False)


def test_formal_31_demo_macro_uses_registered_sum_order(tmp_path):
    demos = [f"demo_{i:02}" for i in range(31)]
    values = [0.1] * 31
    macro = sum(values) / len(values)
    # Both are ordinary finite measurements, but the two reduction orders
    # differ by ulps: exact evidence comparison must use the registered one.
    assert macro != float(np.mean(values))

    def callback(model, conditioner):
        return {"n_expected": 31, "n_ok": 31, "macro_mpjpe_m": macro,
                "per_demo": [{"demo": demo, "T": 9, "status": "ok",
                              "numerical_valid": True, "mpjpe_m": value}
                             for demo, value in zip(demos, values, strict=True)]}

    callback.expected_demos = demos
    callback.demo_lengths = {demo: 9 for demo in demos}
    tr = Trainer(cfg(), tokenizer(), Windows(), Windows(3), tmp_path,
                 formal_eval=callback, formal_eval_every=1)
    formal = tr._run_formal_eval()
    assert formal["complete"] is True and formal["macro_mpjpe_m"] == macro
    assert formal["evidence_gaps"] == []


def test_formal_roster_missing_cannot_be_self_registered_by_callback_output(tmp_path):
    callback = formal_callback()
    del callback.expected_demos
    tr = Trainer(cfg(), tokenizer(), Windows(), Windows(3), tmp_path,
                 formal_eval=callback, formal_eval_every=1)
    formal = tr._run_formal_eval()
    assert "registered_roster_missing_or_invalid" in formal["evidence_gaps"]
    assert formal["complete"] is False


def test_formal_strict_improvement_keeps_tie_and_previous_best_after_gap(tmp_path):
    callback = formal_callback((0.5, 0.3, 0.3, 0.1))
    original = callback

    def wrapped(model, conditioner, budget_deadline=None):
        out = original(model, conditioner)
        if len(original.calls) == 4:
            out["per_demo"].pop()
        return out

    wrapped.expected_demos, wrapped.demo_lengths = original.expected_demos, original.demo_lengths
    tr = Trainer(cfg(steps=4, val_every=3), tokenizer(), Windows(), Windows(3), tmp_path,
                 formal_eval=wrapped, formal_eval_every=1)
    summary = tr.run()
    assert summary["best_formal_step"] == 2 and summary["best_formal_macro_mpjpe_m"] == 0.3
    ck = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ck["step"] == 2 and ck["val"] == {}  # no fabricated current val at independent formal cadence
    assert len(json.loads((tmp_path / "formal_history.json").read_text())) == 4
    best = next(e for e in summary["checkpoint_index"] if e["role"] == "best")
    assert best["step"] == 2 and best["sha256"] == hashlib.sha256((tmp_path / "best.pt").read_bytes()).hexdigest()


def test_deadline_passed_to_aware_formal_callback_and_blocks_new_microbatch(tmp_path):
    callback = formal_callback(with_deadline=True)
    deadline = time.perf_counter() + 1000
    tr = Trainer(cfg(), tokenizer(), Windows(), Windows(3), tmp_path,
                 formal_eval=callback, formal_eval_every=1, budget_deadline=deadline)
    assert tr._run_formal_eval()["complete"] is True
    assert callback.calls == [deadline]
    original = tr._forward

    def expire_after_forward(*a, **kw):
        result = original(*a, **kw)
        tr.budget_deadline = time.perf_counter() - 1
        return result

    tr._forward = expire_after_forward
    summary = tr.run()
    assert summary["steps"] == 0 and summary["budget_stopped"] is True
    assert summary["train_loader"]["micro_batches_total"] == 1
    assert summary["train_loader"]["backward_micro_batches_total"] == 0


def test_validation_deadline_keeps_partial_evidence_without_finite_nan_laundering(tmp_path):
    tr = trainer(tmp_path)
    original = tr._forward

    def expire_after_one(*a, **kw):
        result = original(*a, **kw)
        tr.budget_deadline = time.perf_counter() - 1
        return result

    tr._forward = expire_after_one
    with pytest.raises(TrainingBudgetExceeded):
        tr.validate()
    out = tr._last_validation_evidence
    assert out["complete"] is False and out["n_windows_evaluated"] == 2 and out["n_batches"] == 1
    assert "budget_deadline" in out["error"]
    json.dumps(out, allow_nan=False)


def test_checkpoint_save_deadline_does_not_replace_previous_best(tmp_path, monkeypatch):
    tr = trainer(tmp_path)
    tr.save("best.pt", 0, {})
    old = (tmp_path / "best.pt").read_bytes()
    orig = torch.save

    def slow_save(*a, **kw):
        orig(*a, **kw)
        tr.budget_deadline = time.perf_counter() - 1

    monkeypatch.setattr(torch, "save", slow_save)
    with pytest.raises(TrainingBudgetExceeded):
        tr.save("best.pt", 1, {})
    assert (tmp_path / "best.pt").read_bytes() == old
    assert not (tmp_path / "best.pt.tmp").exists()


def test_last_save_tail_deadline_marks_final_completed_steps_incomplete(tmp_path, monkeypatch):
    tr = trainer(tmp_path, steps=1)
    orig = torch.save

    def expire_on_save(*a, **kw):
        orig(*a, **kw)
        tr.budget_deadline = time.perf_counter() - 1

    monkeypatch.setattr(torch, "save", expire_on_save)
    summary = tr.run()
    assert summary["steps"] == 1 and summary["budget_stopped"] is True
    assert summary["complete"] is False
    assert torch.load(tmp_path / "last.pt", weights_only=False)["step"] == 1
    assert not (tmp_path / "best.pt").exists()


# Generator scripts expose genuine DiffusionG1 subclass identity and valid 6D
# columns so the retry tests exercise retry behavior, not malformed fixtures.
class ScriptDiffusion(DiffusionG1):
    def __init__(self, outputs):
        torch.nn.Module.__init__(self)
        self.outputs = outputs
        self.calls, self.seeds, self.actual = 0, [], []

    def sample(self, memory, F, steps=25, generator=None):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)].clone()
        self.calls += 1
        self.seeds.append(generator.initial_seed())
        self.actual.append(out.clone())
        return out


class ScriptDeterministic(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output, self.calls = output, 0

    def forward(self, memory, F):
        self.calls += 1
        return self.output.clone()


class StubConditioner(torch.nn.Module):
    def forward(self, batch):
        return torch.zeros(len(batch["kps_obj"]), 8)


def prediction(n=3, F=3, value=0.):
    out = torch.full((n, F, 38), value)
    out[..., 3:9] = torch.tensor([1., 0, 0, 0, 1, 0])
    return out


def window_batch(n=3, F=3):
    ds = Windows(n, F)
    return {k: v.numpy() for k, v in ds.data.items() if k not in ("x0", "wfc")}


def condition(T=9):
    pose = np.tile([0, 0, 0.5, 1., 0, 0, 0], (T, 1))
    return ReferenceCondition("fixture", np.zeros((T, 52, 3)), pose, pose.copy(),
                              world_from_clip=np.eye(4))


def test_generator_recovers_only_current_invalid_retains_valid_windows_and_seeds():
    first = prediction(value=0.1)
    first[0, 0, 0] = float("nan")
    second = prediction(value=0.4)
    second[1] = float("nan")  # already accepted window cannot be revoked
    model = ScriptDiffusion([first, second])
    gen = G1Generator(model, StubConditioner(), tokenizer(), window=3, stride=2)
    (pos, quat, q), bad = gen._sample_windows(window_batch(), seed=7)
    assert bad == 0 and model.calls == 2 and model.seeds == [7, 8]
    mean, std = tokenizer().root_stats.to_torch()
    assert np.array_equal(pos[1:], (first[..., :3] * std + mean).numpy()[1:])
    assert np.array_equal(pos[0], (second[..., :3] * std + mean).numpy()[0])
    stats = gen._last_sample_stats
    assert stats["per_attempt_invalid_windows"] == [1, 0]
    assert stats["per_attempt_invalid_window_indices"] == [[0], []]
    assert stats["attempts"][1]["replacement_window_indices"] == [0]
    assert stats["numerical_valid"] and np.isfinite(quat).all() and np.isfinite(q).all()


@pytest.mark.parametrize("bad_kind", ["nan", "inf_joint", "collinear"])
def test_generator_rejection_has_all_attempt_evidence_and_actual_output(bad_kind):
    out = prediction()
    if bad_kind == "nan":
        out[0, 0, 0] = float("nan")
    elif bad_kind == "inf_joint":
        out[0, 0, 9] = float("inf")
    else:
        out[0, :, 6:9] = out[0, :, 3:6] * 2
    model = ScriptDiffusion([out])
    gen = G1Generator(model, StubConditioner(), tokenizer(), window=3, stride=2)
    with pytest.raises(GenerationNumericalError) as caught:
        gen._sample_windows(window_batch(), seed=17)
    exc = caught.value
    assert exc.metadata is exc.evidence and exc.metadata["numerical_valid"] is False
    assert exc.metadata["model_calls"] == 3 and exc.metadata["attempt_seeds"] == [17, 18, 19]
    assert exc.metadata["per_attempt_invalid_windows"] == [1, 1, 1]
    assert exc.metadata["final_invalid_window_indices"] == [0]
    assert np.array_equal(exc.output, out.numpy(), equal_nan=True)
    assert exc.metadata["sampling_wall_s"] >= 0
    json.dumps(exc.metadata, allow_nan=False)


def test_generator_deterministic_bad_rotation_one_call_rejected_before_fusion():
    out = prediction(n=4)
    out[2, :, 6:9] = out[2, :, 3:6]
    model = ScriptDeterministic(out)
    gen = G1Generator(model, StubConditioner(), tokenizer(), window=3, stride=2)
    with pytest.raises(GenerationNumericalError) as caught:
        gen.sample(condition(), seed=7)
    md = caught.value.metadata
    assert model.calls == 1 and md["model_calls"] == 1
    assert md["resample_attempts_used"] == 1 and md["degenerate_windows"] == 1
    assert md["starts"] == [0, 2, 4, 6] and md["last_frame_covered"] is True
    assert md["generation_wall_s"] >= md["sampling_wall_s"]


@pytest.mark.parametrize("model_type", ["deterministic", "diffusion"])
def test_generator_first_success_decode_matches_frozen_math_and_no_quality_retry(model_type):
    out = prediction(value=0.2)
    out[..., 3:9] += torch.randn(3, 3, 6, generator=torch.Generator().manual_seed(8)) * 0.1
    tk = tokenizer()
    mean, std = tk.root_stats.to_torch()
    lo, hi = torch.tensor(tk.lo, dtype=torch.float32), torch.tensor(tk.hi, dtype=torch.float32)
    expected = ((out[..., :3] * std + mean).numpy(),
                mat_to_quat_torch(rot6d_to_mat_torch(out[..., 3:9])).numpy(),
                (lo + (out[..., 9:38].clamp(-1, 1) + 1) * (hi - lo) / 2).numpy())
    model = ScriptDiffusion([out]) if model_type == "diffusion" else ScriptDeterministic(out)
    gen = G1Generator(model, StubConditioner(), tk, window=3, stride=2)
    decoded, bad = gen._sample_windows(window_batch(), seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(decoded, expected, strict=True))
    assert bad == 0 and model.calls == 1


def test_generator_timing_boundaries_are_synchronized_and_included(monkeypatch):
    import ireset.model.generator as module

    clock = [0.]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    model = ScriptDeterministic(prediction(n=4))
    gen = G1Generator(model, StubConditioner(), tokenizer(), window=3, stride=2)
    syncs = []

    def synchronize():
        syncs.append(clock[0])
        clock[0] += 0.1

    gen._synchronize = synchronize
    traj = gen.sample(condition(), seed=7)
    md = traj.metadata
    assert len(syncs) >= 6
    assert md["model_wall_s"] == pytest.approx(0.1)
    assert md["decode_wall_s"] == pytest.approx(0.1)
    assert md["generation_wall_s"] >= md["sampling_wall_s"] >= md["model_wall_s"]
    assert md["model_calls"] == 1 and md["numerical_valid"] is True
    assert (md["T"], md["window"], md["stride"]) == (9, 3, 2)


def test_generator_maximum_is_three_and_decoder_overflow_is_rejected():
    with pytest.raises(ValueError, match="at most 3"):
        G1Generator(ScriptDiffusion([prediction()]), StubConditioner(), tokenizer(), resample_attempts=4)
    tk = tokenizer()
    tk.root_stats.std[0] = 3e38
    out = prediction(value=3.)
    gen = G1Generator(ScriptDeterministic(out), StubConditioner(), tk, window=3, stride=2)
    with pytest.raises(GenerationNumericalError, match="invalid_decoded") as caught:
        gen._sample_windows(window_batch(), seed=7)
    assert caught.value.metadata["model_calls"] == 1
    assert caught.value.metadata["numerical_valid"] is False
