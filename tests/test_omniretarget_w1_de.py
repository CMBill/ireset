"""W1-D/E implementation counterexamples (omniretarget_w1_de_plan §10).

The boundaries this file pins (each test failed or laundered a pass under
a naive implementation):

* budget/history — bad (NaN/negative/bool/running/duplicate/unkeyed) ledger
  rows and disk-without-ledger attempts are UNKNOWN history (stop), an
  evidenced zero is legal, and an exhausted domain refuses to start;
* best selection — recomputed from per-demo rows (missing/duplicate/NaN
  rows and non-reproducible macros never select), strictly decreasing with
  ties keeping the earlier step, agreeing with the Trainer's own choice;
* trainer hardening — non-finite loss stops immediately with evidence
  saved; the budget deadline stops NEW units; per-step rows carry the
  micro-batch/sample counts and lr; the formal best rule integrates with
  the recomputation;
* generator numerical retry — only CURRENTLY-invalid windows are retried
  (a recovered window is recovered; valid windows keep their first
  values), the real model-call count is recorded, and a deterministic
  model never retries (a repeated identical forward is not recovery);
* formal mode — debug subsets, step overrides and argv tampering between
  driver and child are refused; the binding check consumes the
  registration's own anchors;
* register stage — R0 recomputation (not old flags) gates R1; the
  registration is exclusive with identical-payload reuse only;
* the full synthetic chain — train both models, label-gap rows keep one
  row PER method/seed, the fixed matrix is complete, detail re-derives
  headline facts from the products, and finalize recomputes every
  recorded pass flag from the row fields (a tampered flag flips the
  motion baseline to incomplete).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import omniretarget_w1_de as drv  # noqa: E402
from test_omniretarget_w1 import (  # noqa: E402
    _FakeDS, _patch_paths, _repin, _resync_cache_identity, _sha,
    _tokenizer_for_losses, _write_demo_files, build_synthetic_world,
    make_synthetic_demo,
)
from ireset.data import w1_training as w1  # noqa: E402
from ireset.diffusion.generator import DiffusionG1  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

_TINY_CONFIG = """\
model_type: diffusion
steps: 2
micro_batch: 2
grad_accum: 1
lr: 1.0e-4
weight_decay: 1.0e-5
grad_clip: 1.0
seed: 20261008
val_seed: 20261008
num_workers: 0
val_every: 1
log_every: 1
device: cpu
d_model: 16
n_layers: 1
n_heads: 2
ffn_mult: 2
time_window: 3
num_train_timesteps: 100
beta_schedule: scaled_linear
beta_start: 1.0e-4
beta_end: 0.02
loss:
  w_pos: 1.0
  w_rot6d: 1.0
  w_joint: 1.0
  w_geodesic: 0.5
  w_fk: 2.0
  w_contact: 0.0
  w_vel: 0.5
formal_validation_contract:
  loss_validation_every_optimizer_steps: 1
  full_generation_every_optimizer_steps: 1
  full_generation_seed: 7
  full_generation_expected_set: val31
  best_selection: test-tiny
"""


# --- §10 budget / history --------------------------------------------------------

def _isolate_project_root(monkeypatch, tmp_path):
    """Point ireset.paths.PROJECT_ROOT at the test sandbox so the cumulative
    budget/history interfaces never read the workspace's REAL W1-D/E run
    (production keeps the cross-run accumulation; tests must not)."""
    import ireset.paths as paths_mod
    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_domain_history_rejects_unknown_and_bad_costs(tmp_path, monkeypatch):
    _isolate_project_root(monkeypatch, tmp_path)
    run = tmp_path / "run"
    det = run / "formal/deterministic"
    (det / "attempt_001").mkdir(parents=True)

    def write_ledger(rows):
        (run / "costs.json").write_text(json.dumps({"invocations": rows}))

    def write_index(attempts):
        (det / "attempts_index.json").write_text(
            json.dumps({"attempts": attempts}))

    ok_row = {"stage": "w1_de_train_deterministic", "attempt": "attempt_001",
              "status": "done", "wall_s": 10.0}
    idx = [{"attempt": "attempt_001"}]
    # honest zero is legal
    write_ledger([{**ok_row, "wall_s": 0.0}])
    write_index(idx)
    assert drv.domain_history_cost_s(run, "d") == 0.0
    # valid row sums
    write_ledger([ok_row])
    assert drv.domain_history_cost_s(run, "d") == 10.0
    for bad_wall in (float("nan"), -5.0, True, None, "10"):
        write_ledger([{**ok_row, "wall_s": bad_wall}])
        assert drv.domain_history_cost_s(run, "d") is None, bad_wall
    write_ledger([{**ok_row, "status": "running"}])
    assert drv.domain_history_cost_s(run, "d") is None       # no exit reg
    write_ledger([ok_row, dict(ok_row)])                      # duplicate rows
    assert drv.domain_history_cost_s(run, "d") is None
    write_ledger([{**ok_row, "attempt": None}])               # unkeyed row
    assert drv.domain_history_cost_s(run, "d") is None
    # disk attempt without index/ledger coverage -> unknown
    write_ledger([ok_row])
    write_index([{"attempt": "attempt_001"}, {"attempt": "attempt_002"}])
    assert drv.domain_history_cost_s(run, "d") is None
    (det / "attempt_002").mkdir()
    assert drv.domain_history_cost_s(run, "d") is None
    write_ledger([ok_row, {**ok_row, "attempt": "attempt_002",
                           "wall_s": 1.0}])
    assert drv.domain_history_cost_s(run, "d") == 11.0


def test_budget_state_refuses_unknown_or_exhausted(tmp_path, monkeypatch):
    _isolate_project_root(monkeypatch, tmp_path)
    run = tmp_path / "run"
    det = run / "formal/deterministic"
    (det / "attempt_001").mkdir(parents=True)
    (det / "attempts_index.json").write_text(
        json.dumps({"attempts": [{"attempt": "attempt_001"}]}))
    # unknown history: disk attempt without a ledger row
    with pytest.raises(SystemExit, match="unknown"):
        drv._budget_state(run, "d")
    (run / "costs.json").write_text(json.dumps({"invocations": [
        {"stage": "w1_de_train_deterministic", "attempt": "attempt_001",
         "status": "done", "wall_s": drv.D_BUDGET_S - 10.0}]}))
    with pytest.raises(SystemExit, match="exhausted"):
        drv._budget_state(run, "d")


# --- §5.3 best selection recompute -----------------------------------------------

def _fh(step, macro, demos=("a", "b"), n_expected=2, status="ok",
        recorded=None, drop=None, dup=False, nan=False):
    rows = [{"demo": d, "T": 8, "status": status, "numerical_valid": True,
             "mpjpe_m": macro} for d in demos]
    if drop is not None:
        rows = [r for r in rows if r["demo"] != drop]
    if dup:
        rows.append(dict(rows[0]))
    if nan:
        rows[0]["mpjpe_m"] = float("nan")
    return {"step": step, "n_expected": n_expected, "macro_mpjpe_m":
            (recorded if recorded is not None else macro), "per_demo": rows}


def test_recompute_best_selection_boundaries():
    rbs = drv.recompute_best_selection
    # strictly decreasing picks the LAST decrease; ties keep the earlier
    hist = [_fh(1000, 0.50), _fh(2000, 0.30), _fh(3000, 0.30), _fh(4000, 0.35)]
    assert rbs(hist) == (2000, pytest.approx(0.30))
    # a missing row / duplicate row / NaN metric never selects
    assert rbs([_fh(1000, 0.5, drop="a")]) == (None, None)
    assert rbs([_fh(1000, 0.5, dup=True)]) == (None, None)
    assert rbs([_fh(1000, 0.5, nan=True)]) == (None, None)
    assert rbs([_fh(1000, 0.5, status="failed")]) == (None, None)
    # the recorded macro must be reproducible from the rows
    assert rbs([_fh(1000, 0.5, recorded=0.2)]) == (None, None)
    # n_expected must match the row count
    assert rbs([_fh(1000, 0.5, n_expected=3)]) == (None, None)
    # an incomplete flag alone does not veto — the ROWS decide
    hist2 = [{**_fh(1000, 0.5), "complete": False}]
    assert rbs(hist2) == (1000, pytest.approx(0.5))


def test_trainer_best_selection_matches_recompute(tmp_path):
    from ireset.train.losses import LossWeights
    from ireset.train.trainer import TrainConfig, Trainer

    out = tmp_path / "out"
    cfg = TrainConfig(model_type="deterministic", d_model=16, n_layers=1,
                      n_heads=2, time_window=3, ffn_mult=2, device="cpu",
                      micro_batch=2, num_workers=0, steps=3, val_every=1,
                      log_every=1, seed=11, val_seed=13,
                      loss=LossWeights(w_contact=0.0))
    macros = [0.5, 0.3, 0.3]
    calls = {"n": 0}

    def formal_eval(model, conditioner):
        macro = macros[calls["n"]]
        calls["n"] += 1
        return {"n_expected": 2, "n_ok": 2, "macro_mpjpe_m": macro,
                "per_demo": [
                    {"demo": "a", "T": 8, "status": "ok", "numerical_valid": True, "mpjpe_m": macro},
                    {"demo": "b", "T": 8, "status": "ok", "numerical_valid": True, "mpjpe_m": macro}]}

    formal_eval.expected_demos = ["a", "b"]
    formal_eval.demo_lengths = {"a": 8, "b": 8}
    tr = Trainer(cfg, _tokenizer_for_losses(), _FakeDS(8), _FakeDS(4), out,
                 formal_eval=formal_eval, formal_eval_every=1)
    summary = tr.run()
    assert calls["n"] == 3
    # 0.5 -> 0.3 (strict) -> 0.3 (tie keeps step 2000)
    assert summary["best_formal_step"] == 2
    assert summary["selection_status"] == "best_by_formal_macro"
    fh = json.loads((out / "formal_history.json").read_text())
    assert len(fh) == 3
    step, macro = drv.recompute_best_selection(fh)
    assert step == summary["best_formal_step"]
    assert macro == pytest.approx(summary["best_formal_macro_mpjpe_m"])
    ck_index = json.loads((out / "checkpoint_index.json").read_text())
    by_role = {e["role"]: e for e in ck_index}
    assert by_role["best"]["step"] == 2
    assert by_role["best"]["sha256"] == _sha(out / "best.pt")


def test_trainer_no_qualified_best_when_formal_incomplete(tmp_path):
    from ireset.train.losses import LossWeights
    from ireset.train.trainer import TrainConfig, Trainer

    out = tmp_path / "out"
    cfg = TrainConfig(model_type="deterministic", d_model=16, n_layers=1,
                      n_heads=2, time_window=3, ffn_mult=2, device="cpu",
                      micro_batch=2, num_workers=0, steps=2, val_every=1,
                      log_every=1, seed=11, val_seed=13,
                      loss=LossWeights(w_contact=0.0))

    def formal_eval(model, conditioner):
        return {"n_expected": 2, "n_ok": 1, "macro_mpjpe_m": None,
                "per_demo": [
                    {"demo": "a", "T": 8, "status": "ok", "numerical_valid": True, "mpjpe_m": 0.5},
                    {"demo": "b", "T": 8, "status": "label_not_admitted",
                     "mpjpe_m": None}]}

    formal_eval.expected_demos = ["a", "b"]
    formal_eval.demo_lengths = {"a": 8, "b": 8}
    tr = Trainer(cfg, _tokenizer_for_losses(), _FakeDS(8), _FakeDS(4), out,
                 formal_eval=formal_eval, formal_eval_every=1)
    summary = tr.run()
    assert summary["selection_status"] == "no_qualified_best"
    assert summary["best_formal_step"] is None
    assert not (out / "best.pt").exists()


# --- §5.1 trainer hardening --------------------------------------------------------

def test_trainer_nonfinite_loss_stops_and_saves_evidence(tmp_path):
    from ireset.train.losses import LossWeights
    from ireset.train.trainer import TrainConfig, Trainer

    out = tmp_path / "out"
    cfg = TrainConfig(model_type="deterministic", d_model=16, n_layers=1,
                      n_heads=2, time_window=3, ffn_mult=2, device="cpu",
                      micro_batch=4, num_workers=0, steps=50, val_every=1000,
                      log_every=1000, seed=11, val_seed=13,
                      loss=LossWeights(w_contact=0.0))
    ds = _FakeDS(8)
    ds.x0[3, 0, 0] = float("nan")
    tr = Trainer(cfg, _tokenizer_for_losses(), ds, _FakeDS(4), out)
    summary = tr.run()
    assert summary["steps"] < 50
    assert summary["stop_reason"].startswith("nonfinite_")
    assert summary["stop_reason"] is not None
    # the completed evidence is preserved: history rows + last.pt at the
    # completed step count + checkpoint index
    history = json.loads((out / "loss_history.json").read_text())
    assert len(history) == summary["steps"]
    assert all("n_micro" in h and "n_samples" in h and "lr" in h
               for h in history)
    assert all(h["n_micro"] == cfg.grad_accum
               and h["n_samples"] == cfg.micro_batch * cfg.grad_accum
               for h in history)
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["step"] == summary["steps"]
    roles = {e["role"] for e in json.loads(
        (out / "checkpoint_index.json").read_text())}
    assert roles == {"last"}


def test_trainer_budget_deadline_stops_new_units(tmp_path):
    from ireset.train.losses import LossWeights
    from ireset.train.trainer import TrainConfig, Trainer

    out = tmp_path / "out"
    cfg = TrainConfig(model_type="deterministic", d_model=16, n_layers=1,
                      n_heads=2, time_window=3, ffn_mult=2, device="cpu",
                      micro_batch=2, num_workers=0, steps=5, val_every=1,
                      log_every=5, seed=11, val_seed=13,
                      loss=LossWeights(w_contact=0.0))
    # deadline already in the past: NO optimizer step, validation or formal
    # unit may start; the run still registers its evidence
    tr = Trainer(cfg, _tokenizer_for_losses(), _FakeDS(8), _FakeDS(4), out,
                 budget_deadline=time.perf_counter() - 1.0)
    summary = tr.run()
    assert summary["steps"] == 0
    assert summary["budget_stopped"] is True
    assert summary["stop_reason"] == "budget_deadline"
    assert summary["skipped_units"]
    assert (out / "summary.json").is_file()
    assert (out / "loss_history.json").is_file()
    assert torch.load(out / "last.pt", weights_only=False)["step"] == 0


# --- §6.3 generator numerical retry ------------------------------------------------

class _FlakyDiffusion(DiffusionG1):
    """DiffusionG1 subclass whose sample() invalidates scripted windows."""

    def __init__(self, plan):
        torch.nn.Module.__init__(self)
        self.plan = plan
        self.calls = 0
        self.outputs = []

    @torch.no_grad()
    def sample(self, memory, F, steps=25, eta=0.0, generator=None):
        out = torch.zeros(memory.shape[0], F, 38)
        out[..., 3] = 1.0
        out[..., 7] = 1.0
        bad = self.plan[self.calls]
        if bad is not None and any(bad):
            out[torch.as_tensor(list(bad))] = float("nan")
        self.calls += 1
        self.outputs.append(out.clone())
        return out


class _StubDet(torch.nn.Module):
    def __init__(self, bad):
        super().__init__()
        self.bad = list(bad)
        self.calls = 0

    def forward(self, memory, F):
        out = torch.zeros(memory.shape[0], F, 38)
        out[..., 3] = 1.0
        out[..., 7] = 1.0
        out[torch.as_tensor(self.bad)] = float("nan")
        self.calls += 1
        return out


class _StubCond(torch.nn.Module):
    def forward(self, batch):
        return torch.zeros(batch["kps_obj"].shape[0], 8)


def _batch(n=3, F=8):
    return {"kps_obj": np.zeros((n, F, 52, 3), np.float32),
            "kps_clip": np.zeros((n, F, 52, 3), np.float32),
            "box_t_clip": np.zeros((n, F, 3), np.float32),
            "box_quat": np.tile(
                np.array([1, 0, 0, 0], np.float32), (n, F, 1)),
            "contact": np.zeros((n, F, 4), np.float32)}


def test_generator_retry_recovers_currently_invalid_only():
    from ireset.model.generator import G1Generator

    model = _FlakyDiffusion([[True, False, False], [False, False, False]])
    gen = G1Generator(model, _StubCond(), _tokenizer_for_losses(),
                      device="cpu", resample_attempts=3)
    (pos, quat, q), n_bad = gen._sample_windows(_batch(), seed=7)
    assert model.calls == 2                       # real model-call count
    assert n_bad == 0                             # window 0 RECOVERED
    stats = gen._last_sample_stats
    assert stats["model_calls"] == 2
    assert stats["per_attempt_invalid_windows"] == [1, 0]
    assert stats["final_invalid_windows"] == 0
    # valid windows keep their FIRST-attempt values (attempt 1 wrote only
    # the invalid window's slot; window 1/2 == attempt-0 outputs)
    assert np.allclose(pos[1], (model.outputs[0][..., 0:3] * 1 + 0).numpy()[1])
    assert np.allclose(pos[2], model.outputs[0][..., 0:3].numpy()[2])


def test_generator_retry_exhaustion_keeps_failure_fact():
    from ireset.model.generator import G1Generator

    model = _FlakyDiffusion([[True, False, False]] * 3)
    gen = G1Generator(model, _StubCond(), _tokenizer_for_losses(),
                      device="cpu", resample_attempts=3)
    from ireset.model.generator import GenerationNumericalError

    with pytest.raises(GenerationNumericalError) as caught:
        gen._sample_windows(_batch(), seed=7)
    assert model.calls == 3
    assert caught.value.metadata["final_invalid_windows"] == 1
    assert gen._last_sample_stats["per_attempt_invalid_windows"] == [1, 1, 1]


def test_generator_deterministic_single_attempt_no_retry():
    from ireset.model.generator import G1Generator

    model = _StubDet(bad=[1])
    gen = G1Generator(model, _StubCond(), _tokenizer_for_losses(),
                      device="cpu", resample_attempts=3)
    from ireset.model.generator import GenerationNumericalError

    with pytest.raises(GenerationNumericalError) as caught:
        gen._sample_windows(_batch(), seed=7)
    assert model.calls == 1
    assert caught.value.metadata["final_invalid_windows"] == 1
    assert gen._last_sample_stats["model_calls"] == 1


def test_generator_metadata_carries_real_call_counts():
    from ireset.data.schema import ReferenceCondition
    from ireset.model.generator import G1Generator

    T = 90
    rng = np.random.default_rng(0)
    quat = np.tile(np.array([1.0, 0, 0, 0]), (T, 1))
    cond = ReferenceCondition(
        source_clip_id="demo", human_kps_world=rng.normal(size=(T, 52, 3)),
        source_obj_pose=np.concatenate([rng.normal(size=(T, 3)), quat], 1),
        target_obj_pose=np.concatenate([rng.normal(size=(T, 3)), quat], 1),
        frame_dt=1.0 / 30.0, world_from_clip=np.eye(4))
    model = _FlakyDiffusion([[True, False], [False, False]])
    gen = G1Generator(model, _StubCond(), _tokenizer_for_losses(),
                      device="cpu")
    traj = gen.sample(cond, seed=7)
    md = traj.metadata
    assert md["model_calls"] == 2
    assert md["per_attempt_invalid_windows"] == [1, 0]
    assert md["degenerate_windows"] == 0
    assert md["resample_attempts_used"] == 2
    assert traj.root.shape[0] == T
    assert np.isfinite(traj.root).all() and np.isfinite(traj.q).all()


# --- §4.2 formal-mode entry refusals -----------------------------------------------

def test_formal_mode_refuses_debug_subsets_and_step_overrides(
        tmp_path, monkeypatch):
    import train_generation as tg

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_TINY_CONFIG)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({
        "configs": {"config_file_sha256": _sha(cfg)},
        "training_manifest": {"sha256": "m" * 64},
        "cache": {"identity_sha256": "c" * 64, "root_stats_sha256": "r" * 64},
        "label_protocol": w1.MOTION_PROTOCOL_VERSION}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, content in (("cache_identity.json", "{}"),
                          ("root_stats.npz", "not-a-real-npz")):
        (cache / name).write_text(content)

    base = ["prog", "--model", "diffusion",
            "--training-manifest", str(manifest), "--config", str(cfg),
            "--cache-dir", str(cache), "--out", str(tmp_path / "out"),
            "--execution-registration", str(reg)]
    with pytest.raises(SystemExit, match="refuses debug subsets"):
        monkeypatch.setattr(sys, "argv", base + ["--max-files", "2"])
        tg.main()
    with pytest.raises(SystemExit, match="refuses debug subsets"):
        monkeypatch.setattr(sys, "argv", base + ["--max-windows", "2"])
        tg.main()
    with pytest.raises(SystemExit, match="steps from the registered config"):
        monkeypatch.setattr(sys, "argv", base + ["--steps", "1"])
        tg.main()
    bad = tmp_path / "bad.yaml"
    bad.write_text(_TINY_CONFIG.replace("steps: 2", "steps: 5000"))
    with pytest.raises(SystemExit, match="outside"):
        monkeypatch.setattr(
            sys, "argv",
            [base[0], *base[1:base.index("--config") + 1], str(bad),
             *base[base.index("--config") + 2:]])
        tg.main()
    # argv tampering between driver and child: a different config file is
    # refused even though the registration file itself is intact
    other = tmp_path / "other.yaml"
    other.write_text(_TINY_CONFIG.replace("lr: 1.0e-4", "lr: 9.0e-4"))
    with pytest.raises(SystemExit, match="config_not_the_registered_one"):
        tg.verify_formal_execution_binding(reg, other, manifest, cache)
    # budget deadline requires the formal binding
    with pytest.raises(SystemExit, match="requires"):
        monkeypatch.setattr(
            sys, "argv", base[:base.index("--execution-registration")]
            + ["--budget-deadline-s", "10"])
        tg.main()


# --- §2.2/§2.3 R0 preflight + exclusive registration --------------------------------

def _fake_readiness(monkeypatch, fail=()):
    import w1_readiness as readiness

    for name in ("registration", "a3", "b", "c"):
        fn = getattr(readiness, f"check_{name}")

        def result(failing):
            return {"passed": not failing,
                    "reasons": ["synthetic_fail"] if failing else []}
        if name == "b":
            def check_b(run_root, cache, failing=name in fail):
                assert cache is not None
                return result(failing)
            monkeypatch.setattr(readiness, "check_b", check_b)
        elif name == "c":
            def check_c(run_root, cache, prereg, failing=name in fail):
                assert cache is not None and isinstance(prereg, dict)
                return result(failing)
            monkeypatch.setattr(readiness, "check_c", check_c)
        else:
            def make(failing):
                def check(run_root):
                    return result(failing)
                return check
            monkeypatch.setattr(readiness, f"check_{name}", make(name in fail))


def _de_world(tmp_path, monkeypatch, n_train=2, n_val=3, T=90):
    """Synthetic world + fake SOURCE/plan anchors + patched driver identity.

    Returns (world, de_run_root). The readiness checkers are faked to pass
    (their real behaviour is covered by the w1_readiness tests and the real
    R0 run); the ANCHOR legs and everything downstream run for real."""
    world = build_synthetic_world(tmp_path, n_train=n_train, n_val=n_val, T=T)
    _patch_paths(monkeypatch, world)
    root = world["root"]
    src = world["run"]
    import w1_readiness as readiness

    monkeypatch.setattr(readiness, "PROJECT_ROOT", root)

    # stand-in anchor files under the synthetic SOURCE root
    (src / "decisions/attempt_001").mkdir(parents=True, exist_ok=True)
    (src / "decisions/attempt_001/w1_readiness.json").write_text("{}")
    for model in ("deterministic", "diffusion"):
        d = src / "s0" / model / "attempt_001"
        d.mkdir(parents=True, exist_ok=True)
        for name in ("s0_summary.json", "last.pt", "probe_replay.pt"):
            (d / name).write_text("stand-in")
        (d / "loss_history.json").write_text("[]")

    source_anchors = {
        "freeze/preregistration.json": _sha(world["prereg_path"]),
        "audit/attempt_001/training_manifest.json": _sha(world["manifest_path"]),
        "audit/attempts_index.json": _sha(src / "audit/attempts_index.json"),
        "cache/attempt_001/cache_identity.json":
            _sha(world["cache_dir"] / "cache_identity.json"),
        "cache/attempt_001/root_stats.npz":
            _sha(world["cache_dir"] / "root_stats.npz"),
        "cache/attempts_index.json": _sha(src / "cache/attempts_index.json"),
        "decisions/attempt_001/w1_readiness.json":
            _sha(src / "decisions/attempt_001/w1_readiness.json"),
        "costs.json": _sha(src / "costs.json"),
    }
    for model in ("deterministic", "diffusion"):
        for name in ("s0_summary.json", "last.pt", "probe_replay.pt"):
            rel = f"s0/{model}/attempt_001/{name}"
            source_anchors[rel] = _sha(src / rel)
    monkeypatch.setattr(drv, "SOURCE_ANCHORS", source_anchors)

    # project-root protocol anchors: the tiny config is the real one the
    # chain trains with; the rest are stand-ins
    (root / "configs/train").mkdir(parents=True, exist_ok=True)
    (root / "configs/train/omniretarget_motion.yaml").write_text(_TINY_CONFIG)
    (root / "configs/labels").mkdir(parents=True, exist_ok=True)
    (root / "configs/labels/omniretarget_motion_v1.yaml").write_text("x")
    (root / "docs/plans").mkdir(parents=True, exist_ok=True)
    for name in ("omniretarget_first_training_plan.md",
                 "omniretarget_a2_closure_plan.md",
                 "omniretarget_w1_a3bc_plan.md"):
        (root / "docs/plans" / name).write_text("stand-in")
    (root / "uv.lock").write_text("stand-in")
    (root / "de_plan.md").write_text("# w1-de test plan\n")
    plan_anchors = {
        "docs/plans/omniretarget_first_training_plan.md":
            _sha(root / "docs/plans/omniretarget_first_training_plan.md"),
        "docs/plans/omniretarget_a2_closure_plan.md":
            _sha(root / "docs/plans/omniretarget_a2_closure_plan.md"),
        "docs/plans/omniretarget_w1_a3bc_plan.md":
            _sha(root / "docs/plans/omniretarget_w1_a3bc_plan.md"),
        "configs/train/omniretarget_motion.yaml":
            _sha(root / "configs/train/omniretarget_motion.yaml"),
        "configs/labels/omniretarget_motion_v1.yaml":
            _sha(root / "configs/labels/omniretarget_motion_v1.yaml"),
        "uv.lock": _sha(root / "uv.lock"),
    }
    monkeypatch.setattr(drv, "PLAN_ANCHORS", plan_anchors)
    monkeypatch.setattr(drv, "_require_formal_contract", lambda *a, **k: None)
    monkeypatch.setattr(drv, "env_record", lambda project_root: {
        "python": "3.12.13", "executable": "fixture", "uv_lock_sha256": _sha(root / "uv.lock"),
        "torch": torch.__version__, "mujoco": "fixture", "cuda_available": False})
    import train_generation as tg
    monkeypatch.setattr(tg, "PROJECT_ROOT", root)
    monkeypatch.setattr(drv, "implementation_identity",
                        lambda project_root: {
                            "revision": "testrev", "status": [],
                            "tracked_diff_sha256": "0" * 64,
                            "untracked": {}})
    _fake_readiness(monkeypatch)
    de_run = root / "derived/experiments/omniretarget_first/de_run"
    return world, de_run


def test_r0_preflight_failing_checker_blocks_registration(tmp_path, monkeypatch):
    world, de_run = _de_world(tmp_path, monkeypatch)
    _fake_readiness(monkeypatch, fail=("b",))
    args = SimpleNamespace(
        source_run_root=world["run"], run_root=de_run,
        plan="de_plan.md", allow_new_attempt=False)
    code = drv.stage_register(args)
    assert code == 2
    assert not (de_run / "freeze/execution_registration.json").exists()
    # the failed preflight is registered as evidence
    idx = json.loads((de_run / "preflight/attempts_index.json").read_text())
    assert idx["attempts"][-1]["passed"] is False
    report = json.loads((de_run / "preflight/attempt_001"
                         / "preflight_report.json").read_text())
    assert report["passed"] is False
    assert "b:synthetic_fail" in report["reasons"]


def test_register_exclusive_write_and_identical_reuse(tmp_path, monkeypatch):
    world, de_run = _de_world(tmp_path, monkeypatch)
    args = SimpleNamespace(
        source_run_root=world["run"], run_root=de_run,
        plan="de_plan.md", allow_new_attempt=False)
    assert drv.stage_register(args) == 0
    reg_path = de_run / "freeze/execution_registration.json"
    reg = json.loads(reg_path.read_text())
    assert reg["execution_id"] == drv.W1_DE_EXECUTION_ID
    assert reg["rosters"]["val31"] == world["val31"]
    assert reg["configs"]["deterministic"]["steps"] == 2
    assert reg["evaluation"]["task_gate"]["tol_m"] == drv.TASK_MPJPE_TOL_M
    # identical payload -> reuse; a drifted payload -> refusal with the old
    # registration kept as evidence
    args2 = SimpleNamespace(**vars(args))
    assert drv.stage_register(args2) == 0
    idx = json.loads((de_run / "preflight/attempts_index.json").read_text())
    assert idx["attempts"][-1]["registration_reused"] is True
    before = reg_path.read_text()
    reg["configs"]["deterministic"]["lr"] = 9.9
    reg_path.write_text(json.dumps(reg))
    with pytest.raises(SystemExit, match="differs"):
        drv.stage_register(args2)
    assert reg_path.read_text() != before        # not overwritten
    # a drifted SOURCE anchor is an R0 failure (fresh recomputation, never
    # the old report's flags)
    (world["run"] / "decisions/attempt_001/w1_readiness.json").write_text('{"drift": true}')
    code = drv.stage_register(SimpleNamespace(**vars(args)))
    assert code == 2


def test_stage_train_refuses_prior_attempt_and_unknown_history(
        tmp_path, monkeypatch):
    world, de_run = _de_world(tmp_path, monkeypatch)
    args = SimpleNamespace(run_root=de_run, allow_new_attempt=False)
    # no registration yet
    with pytest.raises(SystemExit, match="execution registration"):
        drv.stage_train(args, "deterministic")
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=de_run, plan="de_plan.md")) == 0
    # diffusion before deterministic is out of order
    with pytest.raises(SystemExit, match="train-deterministic"):
        drv.stage_train(args, "diffusion")
    # an unregistered disk attempt = unknown D history
    det = de_run / "formal/deterministic"
    (det / "attempt_001").mkdir(parents=True)
    with pytest.raises(SystemExit, match="unknown"):
        drv._budget_state(de_run, "d")
    shutil_rmtree(det)


def shutil_rmtree(p):
    import shutil
    shutil.rmtree(p)


def test_stage_train_budget_exceeded_flips_complete_in_place(
        tmp_path, monkeypatch):
    world, de_run = _de_world(tmp_path, monkeypatch)
    stages, _, _ = drv.DOMAINS["d"]
    monkeypatch.setattr(drv, "DOMAINS", {
        **drv.DOMAINS, "d": (stages, 2.0, 0.5)})
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=de_run, plan="de_plan.md")) == 0

    def slow_spawn(argv, project_root, timeout_s, log_path):
        assert timeout_s <= 1.5
        assert float(argv[argv.index("--budget-deadline-s") + 1]) <= 1.5
        time.sleep(2.6)
        return 0, "exit", 2.6

    monkeypatch.setattr(drv, "_spawn", slow_spawn)
    args = SimpleNamespace(run_root=de_run, allow_new_attempt=False)
    code = drv.stage_train(args, "deterministic")
    assert code == 2
    det = de_run / "formal/deterministic"
    idx = json.loads((det / "attempts_index.json").read_text())["attempts"]
    assert len(idx) == 1                          # in-place flip, no 2nd row
    assert idx[0]["complete"] is False
    assert idx[0]["incomplete_reason"] == "budget_exceeded_after_registration"
    ledger = [e for e in json.loads((de_run / "costs.json").read_text())
              ["invocations"] if e["stage"] == "w1_de_train_deterministic"]
    assert len(ledger) == 1
    assert ledger[0]["complete"] is False
    # the flipped attempt's cost is real history: the budget is spent
    hist = drv.domain_history_cost_s(de_run, "d")
    assert hist is not None and hist >= 2.0


# --- the full synthetic chain (register -> D -> E -> detail -> finalize) ------------

def test_finish_attempt_publication_waits_for_deadline_verdict(tmp_path, monkeypatch):
    """Real-subprocess counterexample (SIGKILL), already-overrun variant: the
    budget was exhausted before publication started, and the kill lands
    inside a stall right after the terminal publication write. The row on
    disk must be the NON-CONSUMABLE publishing row (final deadline check
    never sealed it) — never a terminal done/complete claim, and the domain
    history must stay unknown."""
    import os
    import signal
    import subprocess

    _isolate_project_root(monkeypatch, tmp_path)
    run = tmp_path / "run"
    (run / "freeze").mkdir(parents=True)
    (run / "freeze/execution_registration.json").write_text("{}")
    stage = "w1_de_raw_eval"
    drv._begin_attempt(run, stage)
    script = tmp_path / "stall_child.py"
    script.write_text(f"""\
import sys, time
sys.path.insert(0, {str(REPO / 'src')!r})
sys.path.insert(0, {str(REPO / 'scripts')!r})
import omniretarget_w1_de as drv

real_upsert = drv._upsert_cost
def stalling_upsert(run_root, stage, attempt, wall, extra, status):
    real_upsert(run_root, stage, attempt, wall, extra, status)
    (run_root / "published.marker").write_text(status)
    time.sleep(60)
drv._upsert_cost = stalling_upsert

budget = {{"total_s": 10.0, "historical_s": 0.0, "remaining_s": 0.1, "reserve_s": 0.0}}
drv.finish_attempt(run_root, {stage!r}, attempt_dir,
                   time.perf_counter() - 1.0, budget,
                   {{"complete": True}}, {{}})
""")
    request = {"run_root": str(run), "attempt_dir": str(run / "evaluation/raw/attempt_001")}
    (tmp_path / "payload.json").write_text(json.dumps(request))
    child = subprocess.Popen(
        [sys.executable, "-c",
         (f"import sys, json; from pathlib import Path; "
          f"sys.path.insert(0, {str(REPO / 'src')!r}); "
          f"sys.path.insert(0, {str(REPO / 'scripts')!r}); "
          f"payload = json.load(open({str(tmp_path / 'payload.json')!r})); "
          f"run_root = Path(payload['run_root']); "
          f"attempt_dir = Path(payload['attempt_dir']); "
          f"exec(open({str(script)!r}).read())")],
        cwd=str(REPO))
    try:
        marker = run / "published.marker"
        for _ in range(300):
            if marker.is_file():
                break
            if child.poll() is not None:
                break
            time.sleep(0.05)
        assert marker.is_file(), "child never published a ledger row"
        os.kill(child.pid, signal.SIGKILL)
    finally:
        child.wait()
    entries = [e for e in json.loads((run / "costs.json").read_text())["invocations"]
               if e.get("stage") == stage]
    assert entries, "the publishing row must survive as evidence"
    e = entries[-1]
    # published but unsealed: non-consumable, never done/complete
    assert e.get("status") == "publishing", e
    assert e.get("complete") is False
    idx = json.loads((run / "evaluation/raw/attempts_index.json").read_text())["attempts"]
    assert idx[-1]["status"] == "publishing" and idx[-1]["complete"] is False
    row, problems = drv.stage_state(run, stage, tmp_path)
    assert row is not None
    assert "attempt_final_check_pending" in problems
    assert drv.domain_history_cost_s(run, "e") is None


def test_finish_attempt_pending_row_survives_sigkill_before_final_verdict(
        tmp_path, monkeypatch):
    """Real-subprocess counterexample (SIGKILL), NOT-yet-overrun variant: the
    budget is generous when publication starts, and the kill lands between
    the pending write and the final-verdict write. The pending row (status
    running) must survive and be NON-CONSUMABLE — stage_state flags it and
    the domain history is unknown — instead of leaving a done/complete=true
    claim whose final check never ran."""
    import os
    import signal
    import subprocess

    _isolate_project_root(monkeypatch, tmp_path)
    run = tmp_path / "run"
    (run / "freeze").mkdir(parents=True)
    (run / "freeze/execution_registration.json").write_text("{}")
    stage = "w1_de_raw_eval"
    drv._begin_attempt(run, stage)
    script = tmp_path / "stall_child.py"
    script.write_text(f"""\
import sys, time
sys.path.insert(0, {str(REPO / 'src')!r})
sys.path.insert(0, {str(REPO / 'scripts')!r})
import omniretarget_w1_de as drv

real_replace = drv._replace_index_row
calls = {{"n": 0}}
def stalling_replace(stage_root, attempt, entry):
    calls["n"] += 1
    if calls["n"] >= 2:
        (run_root / "phase2.marker").write_text("before_final_verdict")
        time.sleep(60)
    return real_replace(stage_root, attempt, entry)
drv._replace_index_row = stalling_replace

budget = {{"total_s": 60.0, "historical_s": 0.0, "remaining_s": 60.0, "reserve_s": 0.0}}
drv.finish_attempt(run_root, {stage!r}, attempt_dir,
                   time.perf_counter(), budget,
                   {{"complete": True}}, {{}})
""")
    request = {"run_root": str(run), "attempt_dir": str(run / "evaluation/raw/attempt_001")}
    (tmp_path / "payload.json").write_text(json.dumps(request))
    child = subprocess.Popen(
        [sys.executable, "-c",
         (f"import sys, json; from pathlib import Path; "
          f"sys.path.insert(0, {str(REPO / 'src')!r}); "
          f"sys.path.insert(0, {str(REPO / 'scripts')!r}); "
          f"payload = json.load(open({str(tmp_path / 'payload.json')!r})); "
          f"run_root = Path(payload['run_root']); "
          f"attempt_dir = Path(payload['attempt_dir']); "
          f"exec(open({str(script)!r}).read())")],
        cwd=str(REPO))
    try:
        marker = run / "phase2.marker"
        for _ in range(300):
            if marker.is_file():
                break
            if child.poll() is not None:
                break
            time.sleep(0.05)
        assert marker.is_file(), "child never reached the final-verdict write"
        os.kill(child.pid, signal.SIGKILL)
    finally:
        child.wait()
    entries = [e for e in json.loads((run / "costs.json").read_text())["invocations"]
               if e.get("stage") == stage]
    assert entries and entries[-1]["status"] == "running", entries
    assert entries[-1]["complete"] is False
    assert entries[-1]["measured_wall_s"] >= 0
    idx = json.loads((run / "evaluation/raw/attempts_index.json").read_text())["attempts"]
    assert idx[-1]["status"] == "running" and idx[-1]["complete"] is False
    row, problems = drv.stage_state(run, stage, tmp_path)
    assert "attempt_running" in problems
    # the pending row is not consumable budget history: unknown = stop
    assert drv.domain_history_cost_s(run, "e") is None


def test_finish_attempt_kill_after_terminal_write_lands_leaves_unsealed_row(
        tmp_path, monkeypatch):
    """Real-subprocess counterexample (SIGKILL), the audited window: the
    budget is generous when publication starts, the terminal (done content)
    publication write itself is slow and crosses the deadline, and the kill
    lands right AFTER the terminal write has landed — before the final
    deadline check could run or revoke. The done/complete=true content IS on
    disk, but under the non-consumable publishing status: stage_state flags
    it and the domain history is unknown, so the undersampled cost can never
    be consumed as a completion claim."""
    import os
    import signal
    import subprocess

    _isolate_project_root(monkeypatch, tmp_path)
    run = tmp_path / "run"
    (run / "freeze").mkdir(parents=True)
    (run / "freeze/execution_registration.json").write_text("{}")
    stage = "w1_de_raw_eval"
    drv._begin_attempt(run, stage)
    script = tmp_path / "stall_child.py"
    script.write_text(f"""\
import sys, time
sys.path.insert(0, {str(REPO / 'src')!r})
sys.path.insert(0, {str(REPO / 'scripts')!r})
import omniretarget_w1_de as drv

# warm the imports finish_attempt pays for, so t0 is not eaten by them
drv._registration_sha(run_root)

real_upsert = drv._upsert_cost
def slow_publishing_upsert(run_root, stage, attempt, wall, extra, status):
    if status == "publishing":
        time.sleep(0.2)          # the terminal write itself crosses the budget
    real_upsert(run_root, stage, attempt, wall, extra, status)
    if status == "publishing":
        (run_root / "published.marker").write_text(status)
        time.sleep(60)           # killed here: after the write, before the check
drv._upsert_cost = slow_publishing_upsert

budget = {{"total_s": 10.0, "historical_s": 0.0, "remaining_s": 0.1, "reserve_s": 0.0}}
drv.finish_attempt(run_root, {stage!r}, attempt_dir,
                   time.perf_counter(), budget,
                   {{"complete": True}}, {{}})
""")
    request = {"run_root": str(run), "attempt_dir": str(run / "evaluation/raw/attempt_001")}
    (tmp_path / "payload.json").write_text(json.dumps(request))
    child = subprocess.Popen(
        [sys.executable, "-c",
         (f"import sys, json; from pathlib import Path; "
          f"sys.path.insert(0, {str(REPO / 'src')!r}); "
          f"sys.path.insert(0, {str(REPO / 'scripts')!r}); "
          f"payload = json.load(open({str(tmp_path / 'payload.json')!r})); "
          f"run_root = Path(payload['run_root']); "
          f"attempt_dir = Path(payload['attempt_dir']); "
          f"exec(open({str(script)!r}).read())")],
        cwd=str(REPO))
    try:
        marker = run / "published.marker"
        for _ in range(300):
            if marker.is_file():
                break
            if child.poll() is not None:
                break
            time.sleep(0.05)
        assert marker.is_file() and marker.read_text() == "publishing"
        os.kill(child.pid, signal.SIGKILL)
    finally:
        child.wait()
    entries = [e for e in json.loads((run / "costs.json").read_text())["invocations"]
               if e.get("stage") == stage]
    assert entries, "the publishing row must survive as evidence"
    e = entries[-1]
    # the done content landed — but unsealed and with the write's own cost
    # missing from the recorded measurement (sampled before the slow write)
    assert e.get("status") == "publishing", e
    assert e.get("complete") is True
    assert e.get("measured_wall_s") < 0.1
    idx = json.loads((run / "evaluation/raw/attempts_index.json").read_text())["attempts"]
    assert idx[-1]["status"] == "publishing" and idx[-1]["complete"] is True
    # ... and it is NOT consumable: unsealed terminal content is rejected
    row, problems = drv.stage_state(run, stage, tmp_path)
    assert row is not None
    assert "attempt_final_check_pending" in problems
    assert drv.domain_history_cost_s(run, "e") is None


def test_finish_attempt_publication_overrun_revokes_before_seal(
        tmp_path, monkeypatch):
    """In-process complement of the kill test: when the terminal publication
    write itself crosses the deadline, the final check (which runs AFTER the
    publication) must revoke the completion and seal an honest incomplete
    row — the recursive post-publication revocation must not be the only
    guard, because it never runs under SIGKILL."""
    world, run = _de_world(tmp_path, monkeypatch)
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=run, plan="de_plan.md")) == 0
    stage = drv.STAGES["raw-eval"]
    att = drv._begin_attempt(run, stage)
    original = drv._upsert_cost
    delayed = {"done": False}

    def slow(*args, **kwargs):
        if args and len(args) >= 6 and args[5] == "publishing" and not delayed["done"]:
            delayed["done"] = True
            time.sleep(0.03)
        return original(*args, **kwargs)

    monkeypatch.setattr(drv, "_upsert_cost", slow)
    assert not drv.finish_attempt(run, stage, att, time.perf_counter(),
        {"remaining_s": 0.01}, {"complete": True}, {})
    rows = json.loads((att.parent / "attempts_index.json").read_text())["attempts"]
    assert len(rows) == 1 and rows[0]["complete"] is False
    assert rows[0]["status"] == "incomplete"
    assert rows[0]["incomplete_reason"] == "budget_exceeded_after_publication"
    costs = json.loads((run / "costs.json").read_text())["invocations"]
    assert len([c for c in costs if c["stage"] == stage]) == 1
    assert costs[-1]["status"] == "incomplete"
    assert costs[-1]["measured_wall_s"] >= 0.03
    # the revoked attempt is honest, consumable budget history
    hist = drv.domain_history_cost_s(run, "e")
    assert hist is not None and hist >= 0.03


def _in_process_spawn(monkeypatch):
    """Fake _spawn that runs the spawned entry IN-PROCESS on the synthetic
    world (the real subprocess path is exercised by the real R/D/E/F run;
    here the same entry functions execute with patched data paths)."""
    import eval_generation as eg
    import train_generation as tg

    def spawn(argv, project_root, timeout_s, log_path):
        t0 = time.perf_counter()
        code, outcome = 0, "exit"
        if "train_generation.py" in argv[1]:
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", ["prog"] + argv[2:])
                try:
                    tg.main()
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 1
        elif "eval_generation.py" in argv[1]:
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", ["prog"] + argv[2:])
                try:
                    eg.main()
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 1
        else:
            raise AssertionError(f"unexpected spawn argv {argv}")
        return code, outcome, time.perf_counter() - t0

    monkeypatch.setattr(drv, "_spawn", spawn)


def _in_process_units(monkeypatch):
    import omniretarget_w1_de_detail as detail
    import omniretarget_w1_de_finalize as final

    def run_unit(action, payload, deadline, attempt_dir, name):
        request = attempt_dir / f"unit__{name}.request.json"
        result = attempt_dir / f"unit__{name}.result.json"
        drv.atomic_json(request, {**payload, "deadline": deadline})
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", ["prog", "--unit", action,
                                        "--request", str(request), "--result", str(result)])
            code = detail._unit_main()
        output = json.loads(result.read_text())
        if code or output.get("unit_error"):
            raise RuntimeError(output.get("unit_error", "unit failed"))
        return output

    monkeypatch.setattr(detail, "_run_unit", run_unit)
    monkeypatch.setattr(final, "_run_unit", run_unit)


def test_de_full_chain_synthetic(tmp_path, monkeypatch):
    world, de_run = _de_world(tmp_path, monkeypatch, n_train=2, n_val=3, T=90)

    # R0/R1
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=de_run, plan="de_plan.md")) == 0
    reg = json.loads((de_run / "freeze/execution_registration.json").read_text())
    assert reg["rosters"]["val31"] == world["val31"]
    assert len(reg["rosters"]["val6"]) == 1

    _in_process_spawn(monkeypatch)
    _in_process_units(monkeypatch)
    train_args = SimpleNamespace(run_root=de_run, allow_new_attempt=False)

    # D1 deterministic
    code = drv.stage_train(train_args, "deterministic")
    assert code == 0
    det_dir = de_run / "formal/deterministic/attempt_001"
    det_summary = json.loads((det_dir / "summary.json").read_text())
    assert det_summary["steps"] == 2
    assert det_summary["stop_reason"] is None
    assert det_summary["selection_status"] == "best_by_formal_macro"
    assert det_summary["train_loader"]["n_windows"] == 4
    # per-step rows carry the counts and every configured loss part
    hist = json.loads((det_dir / "loss_history.json").read_text())
    assert [h["step"] for h in hist] == [1, 2]
    for part in w1.S0_REQUIRED_LOSS_KEYS:
        assert all(part in h for h in hist), part
    assert all(h["n_micro"] == 1 and h["n_samples"] == 2 for h in hist)
    # budget ledger: one running->done row, wall measured
    ledger = [e for e in json.loads((de_run / "costs.json").read_text())
              ["invocations"] if e["stage"] == "w1_de_train_deterministic"]
    assert len(ledger) == 1 and ledger[0]["status"] == "done"
    assert ledger[0]["wall_s"] >= 0
    # a second attempt needs the explicit flag
    with pytest.raises(SystemExit, match="allow-new-attempt"):
        drv.stage_train(train_args, "deterministic")

    # D2 diffusion
    code = drv.stage_train(train_args, "diffusion")
    assert code == 0
    dif_summary = json.loads(
        (de_run / "formal/diffusion/attempt_001/summary.json").read_text())
    assert dif_summary["steps"] == 2

    # E: checkpoint selection + raw matrix
    code = drv.stage_raw_eval(SimpleNamespace(run_root=de_run,
                                              allow_new_attempt=False))
    assert code == 0
    sel = json.loads((de_run / "evaluation/checkpoint_selection.json").read_text())
    for model in ("deterministic", "diffusion"):
        assert sel["models"][model]["checkpoint_role"] == "best"
        assert sel["models"][model]["step"] in (1, 2)
    raw_dir = de_run / "evaluation/raw/attempt_001"
    report = json.loads((raw_dir / "raw_eval.json").read_text())
    rows = report["rows"]
    # the fixed matrix: 3 methods x 3 demos (main) + diffusion seeds 17/27
    # x 3 demos (appendix) = 15 keys, no duplicates, no missing
    assert report["completeness"]["complete"] is True
    assert report["completeness"]["expected_main_keys"] == 9
    assert report["completeness"]["expected_appendix_keys"] == 6
    assert len(rows) == 15
    ok_rows = [r for r in rows if r["status"] == "ok"]
    assert len(ok_rows) == 15
    # every ok row: product saved with hash, retry evidence, both task
    # series, box_ground measured, stitch family
    for r in ok_rows:
        assert r["product"]["sha256"] == _sha(
            world["root"] / r["product"]["path"])
        assert r["retry"]["model_calls"] == 0 if r["method"] == "teacher" else r["retry"]["model_calls"] >= 1
        assert r["task"]["gate_pass"] == (
            r["task"]["world_mpjpe_max_m"] <= drv.TASK_MPJPE_TOL_M)
        assert r["box_ground"]["excluded_from_strict"] is True
        assert r["stitch"]["covers_last_frame"] is True
        assert r["raw_standard_pass"] == (
            r["numerical_valid"] and r["robot_strict_pass"]
            and r["task_gate_pass"])
        # teacher rows are their own reference (fitting self-reference)
        if r["method"] == "teacher":
            assert r["self_reference"] is True
            assert r["retry"]["model_calls"] == 0
    # the reference schedule is SHARED by the three methods per demo
    by_demo_method = {(r["demo"], r["method"]): r for r in ok_rows}
    for demo in world["val31"][:2]:
        refs = {m: by_demo_method[(demo, m)]["ref_frames"]
                for m in ("teacher", "deterministic", "diffusion")}
        assert refs["teacher"] == refs["deterministic"] == refs["diffusion"]
    # per-method aggregates keep the full denominator
    pm = report["per_method"]
    for g in ("teacher|None", "deterministic|7", "diffusion|7",
              "diffusion|17", "diffusion|27"):
        assert pm[g]["counts"]["expected"] == 3
        assert pm[g]["counts"]["label_failures"] == 0
        assert pm[g]["counts"]["n_rows"] == 3
    assert report["repair_pipeline_invocations"] == 0

    # detail: val6 x 3 methods re-derived from the products
    import omniretarget_w1_de_detail as det_mod
    code = det_mod.stage_detail(SimpleNamespace(run_root=de_run,
                                                allow_new_attempt=False))
    assert code == 0
    detail_dir = de_run / "evaluation/detail_val6/attempt_001"
    detail = json.loads((detail_dir / "detail_report.json").read_text())
    assert detail["counts"]["n_ok"] == 3
    assert detail["replay_consistency"]["all_headline_facts_reproduced"]

    # finalize
    import omniretarget_w1_de_finalize as fin_mod
    code = fin_mod.stage_finalize(SimpleNamespace(run_root=de_run,
                                                  allow_new_attempt=False))
    assert code == 0
    fin_dir = de_run / "decisions"
    fin_rows = json.loads((fin_dir / "attempts_index.json").read_text())["attempts"]
    final = json.loads((fin_dir / fin_rows[-1]["attempt"]
                        / "w1_de_final.json").read_text())
    assert final["training"]["training_complete"] is True
    assert final["evaluation"]["evaluation_complete"] is True
    assert final["motion_baseline"] == "complete"
    assert final["final_interaction_method"]["verdict"] in (
        "not_ready", "evidence_insufficient")
    # the checkpoint replay leg reproduced the raw product (fixed seed)
    for model in ("deterministic", "diffusion"):
        rep = final["checkpoint_replay"]["models"][model]
        assert rep["status"] == "replayed"
        assert rep["within_tolerance"] is True
        assert rep["covers_last_frame"] is True
    assert final["historical_anchors"]["u0_verdict"] == "incomplete"

    # TAMPER COUNTEREXAMPLE: flipping a pass flag in the raw report without
    # the underlying fields makes the finalize re-derivation refuse — the
    # new attempt registers incomplete instead of laundering the flag
    report["rows"][0]["raw_standard_pass"] = not report["rows"][0]["raw_standard_pass"]
    (raw_dir / "raw_eval.json").write_text(json.dumps(report, default=float))
    code = fin_mod.stage_finalize(SimpleNamespace(run_root=de_run,
                                                  allow_new_attempt=False))
    assert code == 2
    fin_rows = json.loads((fin_dir / "attempts_index.json").read_text())["attempts"]
    final2 = json.loads((fin_dir / fin_rows[-1]["attempt"]
                         / "w1_de_final.json").read_text())
    assert final2["motion_baseline"] == "incomplete"
    assert any("raw_standard_pass" in p
               for p in final2["evaluation"]["recompute_problems"])

    # COUNTEREXAMPLE (budget audit): an unregistered disk attempt in a
    # NON-detail_f domain (a stray D attempt with no index/ledger row) makes
    # the D history unknown — the closure re-audits every domain, not only
    # the one it is about to spend
    (de_run / "formal/deterministic/attempt_002").mkdir()
    code = fin_mod.stage_finalize(SimpleNamespace(run_root=de_run,
                                                  allow_new_attempt=False))
    assert code == 2
    fin_rows = json.loads((fin_dir / "attempts_index.json").read_text())["attempts"]
    final3 = json.loads((fin_dir / fin_rows[-1]["attempt"]
                         / "w1_de_final.json").read_text())
    assert final3["complete"] is False
    assert "budget_domain_history_unknown:d" in final3["problems"]
    assert final3["budget_domains"]["d"]["historical_s"] is None
    shutil_rmtree(de_run / "formal/deterministic/attempt_002")

    # COUNTEREXAMPLE (curves completeness): a hash-correct curve artifact
    # stripped to the five formerly spot-checked fields is an incomplete
    # delivery — the full required field set, shapes and definition masks
    # are verified
    detail_idx_path = de_run / "evaluation/detail_val6/attempts_index.json"
    detail_idx = json.loads(detail_idx_path.read_text())
    detail_report_path = detail_dir / "detail_report.json"
    detail_rep = json.loads(detail_report_path.read_text())
    curves_row = next(r for r in detail_rep["rows"] if r.get("curves"))
    curves_path = world["root"] / curves_row["curves"]["path"]
    curves_orig_bytes = curves_path.read_bytes()
    curves_orig_sha = curves_row["curves"]["sha256"]
    with np.load(curves_path, allow_pickle=False) as z:
        kept = {k: z[k].copy() for k in ("task_world", "both_certified_overlap",
                                         "both_reference", "hand_certified_l",
                                         "hand_certified_r")}
    with open(curves_path, "wb") as stream:
        np.savez(stream, **kept)
    new_sha = _sha(curves_path)
    for r in detail_rep["rows"]:
        if isinstance(r.get("curves"), dict) and r["curves"]["path"] == curves_row["curves"]["path"]:
            r["curves"]["sha256"] = new_sha
    detail_report_path.write_text(json.dumps(detail_rep, default=float))
    for idx_row in detail_idx["attempts"]:
        if idx_row.get("attempt") == "attempt_001":
            idx_row.setdefault("artifacts", {})[curves_row["curves"]["path"]] = new_sha
            idx_row["report_sha256"] = _sha(detail_report_path)
    detail_idx_path.write_text(json.dumps(detail_idx))
    code = fin_mod.stage_finalize(SimpleNamespace(run_root=de_run,
                                                  allow_new_attempt=False))
    assert code == 2
    fin_rows = json.loads((fin_dir / "attempts_index.json").read_text())["attempts"]
    final4 = json.loads((fin_dir / fin_rows[-1]["attempt"]
                         / "w1_de_final.json").read_text())
    assert any(p.endswith(":curves_field_missing:task_box")
               for p in final4["detail"]["problems"])
    # restore the complete curve artifact for the later counterexamples
    curves_path.write_bytes(curves_orig_bytes)
    detail_rep = json.loads(detail_report_path.read_text())
    for r in detail_rep["rows"]:
        if isinstance(r.get("curves"), dict) and r["curves"]["path"] == curves_row["curves"]["path"]:
            r["curves"]["sha256"] = curves_orig_sha
    detail_report_path.write_text(json.dumps(detail_rep, default=float))
    detail_idx = json.loads(detail_idx_path.read_text())
    for idx_row in detail_idx["attempts"]:
        if idx_row.get("attempt") == "attempt_001":
            if curves_row["curves"]["path"] in idx_row.get("artifacts", {}):
                idx_row["artifacts"][curves_row["curves"]["path"]] = curves_orig_sha
            idx_row["report_sha256"] = _sha(detail_report_path)
    detail_idx_path.write_text(json.dumps(detail_idx))

    # COUNTEREXAMPLE (reference completeness): a hash-correct reference
    # artifact stripped of the source box, scale and teacher trajectory is
    # incomplete — the reference is bound bitwise to the frozen clip
    ref_row = next(r for r in detail_rep["rows"] if r.get("reference"))
    ref_path = world["root"] / ref_row["reference"]["path"]
    ref_orig_bytes = ref_path.read_bytes()
    ref_orig_sha = ref_row["reference"]["sha256"]
    with np.load(ref_path, allow_pickle=False) as z:
        kept = {k: z[k].copy() for k in ("human_kpts_world", "target_box")}
    with open(ref_path, "wb") as stream:
        np.savez(stream, **kept)
    ref_sha = _sha(ref_path)
    for r in detail_rep["rows"]:
        if isinstance(r.get("reference"), dict) and r["reference"]["path"] == ref_row["reference"]["path"]:
            r["reference"]["sha256"] = ref_sha
    detail_report_path.write_text(json.dumps(detail_rep, default=float))
    for idx_row in detail_idx["attempts"]:
        if idx_row.get("attempt") == "attempt_001":
            if ref_row["reference"]["path"] in idx_row.get("artifacts", {}):
                idx_row["artifacts"][ref_row["reference"]["path"]] = ref_sha
            idx_row["report_sha256"] = _sha(detail_report_path)
    detail_idx_path.write_text(json.dumps(detail_idx))
    code = fin_mod.stage_finalize(SimpleNamespace(run_root=de_run,
                                                  allow_new_attempt=False))
    assert code == 2
    fin_rows = json.loads((fin_dir / "attempts_index.json").read_text())["attempts"]
    final5 = json.loads((fin_dir / fin_rows[-1]["attempt"]
                         / "w1_de_final.json").read_text())
    assert any(p.endswith(":reference_field_missing:source_box")
               for p in final5["detail"]["problems"])
    # restore the complete reference artifact for the later counterexamples
    ref_path.write_bytes(ref_orig_bytes)
    detail_rep = json.loads(detail_report_path.read_text())
    for r in detail_rep["rows"]:
        if isinstance(r.get("reference"), dict) and r["reference"]["path"] == ref_row["reference"]["path"]:
            r["reference"]["sha256"] = ref_orig_sha
    detail_report_path.write_text(json.dumps(detail_rep, default=float))
    detail_idx = json.loads(detail_idx_path.read_text())
    for idx_row in detail_idx["attempts"]:
        if idx_row.get("attempt") == "attempt_001":
            if ref_row["reference"]["path"] in idx_row.get("artifacts", {}):
                idx_row["artifacts"][ref_row["reference"]["path"]] = ref_orig_sha
            idx_row["report_sha256"] = _sha(detail_report_path)
    detail_idx_path.write_text(json.dumps(detail_idx))

    # COUNTEREXAMPLE (teacher label binding): a teacher product whose root
    # deviates 10 cm from the frozen clip is not self-verifiable evidence —
    # the FK self-reference shortcut requires a bitwise reconciliation first
    teacher_row = next(r for r in rows if r["method"] == "teacher")
    teacher_path = world["root"] / teacher_row["product"]["path"]
    with np.load(teacher_path, allow_pickle=False) as z:
        tarrays = {k: z[k].copy() for k in z.files}
    tarrays["root"] = tarrays["root"].copy()
    tarrays["root"][:, 0] += 0.10
    tarrays["scene"] = np.concatenate(
        [tarrays["root"], tarrays["joints"], tarrays["target_box"]], axis=1)
    with open(teacher_path, "wb") as stream:
        np.savez(stream, **tarrays)
    import eval_generation as eg
    import omniretarget_w1_de_detail as detail_mod
    clip, _ = detail_mod._load_clip(reg, world["root"], teacher_row["demo"])
    measured = eg.recompute_w1_de_product(
        teacher_path,
        {**teacher_row, "product": {**teacher_row["product"],
                                    "sha256": _sha(teacher_path)}},
        reg, clip, validator=None, replayer=None,
        project_root=world["root"],
        expected_identity=teacher_row["product"]["identity"])
    assert measured["row"]["status"] == "product_invalid"
    assert "teacher_product_not_bitwise_frozen_label" in measured["row"]["measurement_gaps"]

    # COUNTEREXAMPLE (frozen similarity scale): rewriting the reference
    # artifact's similarity_scale to 1.0 and re-hashing it must not pass —
    # the delivered scale must BE the frozen derived clip's scale
    # (restore the full reference first)
    detail_rep = json.loads((detail_dir / "detail_report.json").read_text())
    detail_idx = json.loads((de_run / "evaluation/detail_val6/attempts_index.json").read_text())
    detail_idx_path = de_run / "evaluation/detail_val6/attempts_index.json"
    detail_report_path = detail_dir / "detail_report.json"
    scale_row = next(r for r in detail_rep["rows"] if isinstance(r.get("reference"), dict))
    scale_path = world["root"] / scale_row["reference"]["path"]
    with np.load(scale_path, allow_pickle=False) as z:
        sarrays = {k: z[k].copy() for k in z.files}
    frozen_scale = float(sarrays["source_box_similarity_scale"])
    sarrays["source_box_similarity_scale"] = np.float64(1.0)
    with open(scale_path, "wb") as stream:
        np.savez(stream, **sarrays)
    scale_sha = _sha(scale_path)
    assert frozen_scale != 1.0
    for r in detail_rep["rows"]:
        if isinstance(r.get("reference"), dict) and r["reference"]["path"] == scale_row["reference"]["path"]:
            r["reference"]["sha256"] = scale_sha
    detail_report_path.write_text(json.dumps(detail_rep, default=float))
    for idx_row in detail_idx["attempts"]:
        if idx_row.get("attempt") == "attempt_001":
            if scale_row["reference"]["path"] in idx_row.get("artifacts", {}):
                idx_row["artifacts"][scale_row["reference"]["path"]] = scale_sha
            idx_row["report_sha256"] = _sha(detail_report_path)
    detail_idx_path.write_text(json.dumps(detail_idx))
    code = fin_mod.stage_finalize(SimpleNamespace(run_root=de_run,
                                                  allow_new_attempt=False))
    assert code == 2
    fin_rows = json.loads((fin_dir / "attempts_index.json").read_text())["attempts"]
    final6 = json.loads((fin_dir / fin_rows[-1]["attempt"]
                         / "w1_de_final.json").read_text())
    assert any(":reference_similarity_scale_not_frozen" in p
               for p in final6["detail"]["problems"])
    # restore the frozen scale for the remaining counterexamples
    sarrays["source_box_similarity_scale"] = np.float64(frozen_scale)
    with open(scale_path, "wb") as stream:
        np.savez(stream, **sarrays)
    restored_sha = _sha(scale_path)
    detail_rep = json.loads(detail_report_path.read_text())
    for r in detail_rep["rows"]:
        if isinstance(r.get("reference"), dict) and r["reference"]["path"] == scale_row["reference"]["path"]:
            r["reference"]["sha256"] = restored_sha
    detail_report_path.write_text(json.dumps(detail_rep, default=float))
    detail_idx = json.loads(detail_idx_path.read_text())
    for idx_row in detail_idx["attempts"]:
        if idx_row.get("attempt") == "attempt_001":
            if scale_row["reference"]["path"] in idx_row.get("artifacts", {}):
                idx_row["artifacts"][scale_row["reference"]["path"]] = restored_sha
            idx_row["report_sha256"] = _sha(detail_report_path)
    detail_idx_path.write_text(json.dumps(detail_idx))

    # COUNTEREXAMPLE (Inf is not an undefined slot): replacing the first
    # frame's undefined NaN velocity with +Inf must be rejected before the
    # NaN definition mask comparison
    detail_rep = json.loads(detail_report_path.read_text())
    inf_row = next(r for r in detail_rep["rows"] if isinstance(r.get("curves"), dict))
    inf_path = world["root"] / inf_row["curves"]["path"]
    with np.load(inf_path, allow_pickle=False) as z:
        carrays = {k: z[k].copy() for k in z.files}
    carrays["base_linear_velocity_mps"][0] = np.inf
    with open(inf_path, "wb") as stream:
        np.savez(stream, **carrays)
    inf_sha = _sha(inf_path)
    for r in detail_rep["rows"]:
        if isinstance(r.get("curves"), dict) and r["curves"]["path"] == inf_row["curves"]["path"]:
            r["curves"]["sha256"] = inf_sha
    detail_report_path.write_text(json.dumps(detail_rep, default=float))
    detail_idx = json.loads(detail_idx_path.read_text())
    for idx_row in detail_idx["attempts"]:
        if idx_row.get("attempt") == "attempt_001":
            if inf_row["curves"]["path"] in idx_row.get("artifacts", {}):
                idx_row["artifacts"][inf_row["curves"]["path"]] = inf_sha
            idx_row["report_sha256"] = _sha(detail_report_path)
    detail_idx_path.write_text(json.dumps(detail_idx))
    code = fin_mod.stage_finalize(SimpleNamespace(run_root=de_run,
                                                  allow_new_attempt=False))
    assert code == 2
    fin_rows = json.loads((fin_dir / "attempts_index.json").read_text())["attempts"]
    final7 = json.loads((fin_dir / fin_rows[-1]["attempt"]
                         / "w1_de_final.json").read_text())
    assert any(p.endswith(":curves_field_infinite:base_linear_velocity_mps")
               for p in final7["detail"]["problems"])

    # COUNTEREXAMPLE (detail_f history unknown at finalize): a stray
    # unregistered attempt under the finalize stage directory makes the
    # detail_f history unknown — the closure must REGISTER its own refusal
    # as a terminal incomplete attempt (no running residue, no TypeError)
    # instead of crashing without a record
    stray = de_run / "decisions/attempt_099"
    stray.mkdir()
    code = fin_mod.stage_finalize(SimpleNamespace(run_root=de_run,
                                                  allow_new_attempt=False))
    assert code == 2
    fin_rows = json.loads((fin_dir / "attempts_index.json").read_text())["attempts"]
    final8 = json.loads((fin_dir / fin_rows[-1]["attempt"]
                         / "w1_de_final.json").read_text())
    assert final8["complete"] is False
    assert any(p in final8["problems"] for p in
               ("budget_domain_history_unknown:detail_f",
                "budget_domain_detail_f_unavailable"))
    assert final8["budget_domains"]["detail_f"]["historical_s"] is None
    fin_state, fin_state_problems = drv.stage_state(
        de_run, drv.STAGES["finalize"], world["root"])
    assert not fin_state_problems, fin_state_problems
    assert fin_state["status"] in ("incomplete", "crashed")
    assert fin_state["complete"] is False
    shutil_rmtree(stray)
