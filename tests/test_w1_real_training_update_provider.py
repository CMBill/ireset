"""Gate A/B tests for the real provider, resume protocol and F timing.

Everything here is synthetic: temporary files, analytic four-window fixtures,
metadata stubs and a fake MuJoCo module.  No real checkpoint payload, parent
NPZ value, G1 mesh or compiled asset is read.
"""

from __future__ import annotations

import copy
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch

import ireset.eval.w1_real_training_update_audit as A
import ireset.eval.w1_real_training_update_provider as P
from ireset.model.tokens import RootStats, StateTokenizer


TINY_CONFIG = dict(A.MODEL_CONFIG)
TINY_CONFIG.update(d_model=8, n_layers=1, n_heads=2, ffn_mult=2,
                   time_window=3)
def _tolerance():
    return A.build_default_tolerance_contract()


def _provider() -> A.SyntheticTrainingProvider:
    return A.SyntheticTrainingProvider(TINY_CONFIG, seed=20261029,
                                       device="cuda")


def _budget(attempts_used: int = 0) -> dict:
    return {"remaining_s": 10 ** 6, "reserve_s": 0.0, "attempts": 2,
            "attempts_used": attempts_used}


def _zero_snapshot() -> dict:
    return A.CallCounter(A.CALL_CAPS).snapshot()


def _finish(root: Path, stage: str, attempt: str, complete: bool,
            report: Path | None, t0: float | None = None,
            artifacts: dict | None = None) -> None:
    A.finish_attempt(
        root, stage, attempt, t0 if t0 is not None else A._now(),
        _budget(), {"complete": complete,
                    "report": A.path_for_record(report)
                    if report is not None else None,
                    "summary": A.path_for_record(report)
                    if report is not None else None,
                    "artifacts": dict(artifacts or {})},
        status="done" if complete else "incomplete",
        extra={"stage": stage, "call_balance": _zero_snapshot()})


# ---------------------------------------------------------------------------
# header / metadata primitives
# ---------------------------------------------------------------------------


def test_npz_header_reader_does_not_decode_payload(tmp_path):
    path = tmp_path / "arrays.npz"
    np.savez(path, x0=np.zeros((4, 60, 38), np.float32),
             alpha=np.arange(5, dtype=np.float64))
    headers = P.read_npz_headers(path)
    assert headers["x0"] == {"shape": [4, 60, 38], "dtype": "float32",
                             "fortran_order": False, "compressed": False}
    assert headers["alpha"]["shape"] == [5]


def test_window_header_problems_reject_missing_and_wrong_schema():
    good = {key: {"shape": list(shape), "dtype": dtype.name,
                  "fortran_order": False, "compressed": False}
            for key, (dtype, shape) in P.REQUIRED_WINDOW_HEADERS.items()}
    assert P.window_header_problems(good) == []
    missing = dict(good)
    missing.pop("x0")
    assert any(p.startswith("header_missing:x0")
               for p in P.window_header_problems(missing))
    wrong = copy.deepcopy(good)
    wrong["kps_obj"]["shape"] = [4, 60, 52, 2]
    assert any(p.startswith("header_shape:kps_obj")
               for p in P.window_header_problems(wrong))
    compressed = copy.deepcopy(good)
    compressed["x0"]["compressed"] = True
    assert "header_compressed:x0" in P.window_header_problems(compressed)


def test_window_meta_and_checkpoint_config_problems():
    keys = [{"demo": spec[0], "start": spec[1], "end": spec[2],
             "T": spec[3], "split": "train"}
            for spec in A.WINDOW_SPECS]
    assert P._window_meta_problems(keys) == []
    bad = copy.deepcopy(keys)
    bad[1]["start"] = 31
    assert "metadata_window:1" in P._window_meta_problems(bad)
    configs = {}
    for model in A.MODELS:
        configs[model] = {
            **{key: A.MODEL_CONFIG[key] for key in
               ("micro_batch", "grad_accum", "d_model", "n_layers",
                "n_heads", "ffn_mult", "time_window",
                "num_train_timesteps", "lr", "weight_decay", "grad_clip",
                "beta_start", "beta_end", "beta_schedule")},
            "model_type": model, "steps": 4000,
            "loss": {"w_pos": 1.0, "w_rot6d": 1.0, "w_joint": 1.0,
                     "w_geodesic": 0.5, "w_fk": 2.0, "w_contact": 0.0,
                     "w_vel": 0.5},
        }
    assert P._checkpoint_config_problems(configs) == []
    configs["diffusion"]["lr"] = 1e-3
    assert "metadata_checkpoint_config:diffusion:lr" in \
        P._checkpoint_config_problems(configs)


def test_asset_binding_rejects_reference_drift():
    expected = {
        "checked": True, "problems": [],
        "locked": {"xml": {"sha256": "a" * 64}},
        "references": {"/x/a.obj": {"sha256": "b" * 64, "tag": "mesh"}},
        "reference_count": 1, "closure_sha256": "c" * 64,
    }
    actual = copy.deepcopy(expected)
    assert P._asset_binding_problems(actual, expected) == []
    actual["references"]["/x/a.obj"]["sha256"] = "d" * 64
    assert "asset_reference_sha:a.obj" in P._asset_binding_problems(
        actual, expected)
    actual = copy.deepcopy(expected)
    actual["references"]["/x/b.obj"] = {"sha256": "e" * 64, "tag": "mesh"}
    assert "asset_reference_set" in P._asset_binding_problems(actual, expected)


# ---------------------------------------------------------------------------
# provider reference backend wiring
# ---------------------------------------------------------------------------


def test_reference_context_uses_provider_backend_not_synthetic_fk(
        tmp_path, monkeypatch):
    provider = _provider()
    provider.prepare_fixture(tmp_path / "fixture")
    provider.load_fixture()
    calls = {"positions": 0, "jacobians": 0}
    counter = A.CallCounter(A.CALL_CAPS)

    class Backend:
        def positions(self, pos, R, q):
            calls["positions"] += 1
            counter.add("reference_mj_forward", pos.shape[0])
            return np.zeros((pos.shape[0], len(A.SITE_NAMES), 3))

        def jacobians(self, pos, R, q):
            calls["jacobians"] += 1
            counter.add("reference_mj_forward", pos.shape[0])
            counter.add("reference_mj_jac_site",
                        pos.shape[0] * len(A.SITE_NAMES))
            return np.zeros((pos.shape[0], len(A.SITE_NAMES), 3, 35))

    provider.reference = Backend()

    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic FK must not be reached")

    monkeypatch.setattr(A, "synthetic_fk_np", forbidden)
    monkeypatch.setattr(A, "synthetic_fk_jac_np", forbidden)
    context = A.ReferenceContext(provider, counter)
    assert calls["positions"] == 4
    pred = np.zeros((4, 60, 38), np.float32)
    pred[..., 3] = 1.0
    pred[..., 7] = 1.0
    pred[..., 9:38] = 0.0
    result = context.evaluate(pred, A.MICRO_SCHEDULE[1][0])
    assert calls["jacobians"] == 1
    assert result["elements"].shape == (4 * 60, len(A.SITE_NAMES))
    assert counter.counts["reference_mj_forward"] == 720
    assert counter.counts["reference_mj_jac_site"] == 240 * len(
        A.SITE_NAMES)


def test_synthetic_backend_counts_after_success(tmp_path):
    provider = _provider()
    provider.prepare_fixture(tmp_path / "fixture")
    provider.load_fixture()
    counter = A.CallCounter(A.CALL_CAPS)
    provider.bind_counter(counter)
    pos = np.zeros((2, 3), np.float64)
    R = np.tile(np.eye(3), (2, 1, 1))
    q = np.zeros((2, 29), np.float64)
    positions = provider.reference_positions(pos, R, q)
    assert positions.shape == (2, len(A.SITE_NAMES), 3)
    assert counter.counts["reference_mj_forward"] == 2
    jac = provider.reference_jacobians(pos, R, q)
    assert jac.shape == (2, len(A.SITE_NAMES), 3, 35)
    assert counter.counts["reference_mj_forward"] == 4
    assert counter.counts["reference_mj_jac_site"] == 2 * len(A.SITE_NAMES)


def _fake_mujoco():
    module = types.ModuleType("mujoco")

    class MjData:
        def __init__(self, model):
            self.qpos = np.zeros(model.nq)
            self.site_xpos = np.zeros((len(model.site_names), 3))

    module.MjData = MjData
    module.mjtObj = types.SimpleNamespace(mjOBJ_SITE=1, mjOBJ_BODY=2,
                                          mjOBJ_JOINT=3)
    module.mjtJoint = types.SimpleNamespace(mjJNT_HINGE=0)
    module.mj_name2id = lambda model, kind, name: (
        int(np.where(model.site_names == name)[0][0]) if kind == 1
        else int(np.where(model.joint_names == name)[0][0]))
    module.mj_id2name = lambda model, kind, ident: model.body_names[ident]
    state = {"forward": 0, "jac": 0}

    def forward(model, data):
        state["forward"] += 1

    def jac_site(model, data, jacp, jacr, site_id):
        state["jac"] += 1
        jacp[:] = float(site_id + 1)

    module.mj_forward = forward
    module.mj_jacSite = jac_site
    return module, state


def _fake_g1spec(monkeypatch, calls=None):
    from ireset.embodied import g1_spec

    class Model:
        nq = 36
        nv = 35
        site_names = np.array(A.SITE_NAMES)
        joint_names = np.array([f"j{i}" for i in range(29)])
        body_names = np.array([f"b{i}" for i in range(40)])
        site_bodyid = np.arange(len(A.SITE_NAMES))
        site_pos = np.zeros((len(A.SITE_NAMES), 3))
        site_quat = np.tile(np.array([1.0, 0, 0, 0]),
                            (len(A.SITE_NAMES), 1))
        jnt_qposadr = np.array([7 + i for i in range(29)])
        jnt_type = np.zeros(29, dtype=int)
        jnt_range = np.tile(np.array([-1.0, 1.0]), (29, 1))

    class Spec:
        def __init__(self):
            if calls is not None:
                calls.append("first_compile")
            self.model = Model()

        def load_plain(self):
            if calls is not None:
                calls.append("second_compile")
            return self.model

    bodies = {name: f"b{i}" for i, name in enumerate(A.SITE_NAMES)}
    monkeypatch.setattr(g1_spec, "G1Spec", Spec)
    monkeypatch.setattr(g1_spec, "KPT_BODIES", bodies)
    monkeypatch.setattr(g1_spec, "JOINT_NAMES", [f"j{i}" for i in range(29)])


def test_reference_backend_guards_the_second_compile(monkeypatch):
    fake, _state = _fake_mujoco()
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    calls: list[str] = []
    _fake_g1spec(monkeypatch, calls)

    def guard(unit: str) -> None:
        if unit == "reference_backend_recompile":
            raise TimeoutError(unit)

    with pytest.raises(TimeoutError):
        P.RealReferenceBackend(
            A.CallCounter(A.CALL_CAPS), deadline=None, guard=guard,
            joint_lo=np.full(29, -1.0), joint_hi=np.full(29, 1.0))
    assert calls == ["first_compile"]


def test_trainer_internal_units_check_the_guard(tmp_path, monkeypatch):
    calls: list[str] = []

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            observer = kwargs["construction_observer"]
            observer("model")
            observer("conditioner")
            observer("optimizer")

    monkeypatch.setattr(P, "Trainer", FakeTrainer)
    provider = P.RealTrainingProvider(tmp_path)
    provider.tokenizer = StateTokenizer(
        np.tile(np.array([-1.0, 1.0]), (29, 1)),
        RootStats(mean=np.zeros(3), std=np.ones(3)))

    def guard(unit: str) -> None:
        calls.append(unit)
        if unit == "trainer_conditioner_constructed":
            raise TimeoutError(unit)

    provider._guard = guard
    counter = A.CallCounter(A.CALL_CAPS)
    with pytest.raises(TimeoutError):
        provider.build_trainer("deterministic", with_optimizer=True,
                               counter=counter, loss_mode="forbidden")
    assert calls == [
        "trainer_construction", "trainer_construction",
        "trainer_model_constructed", "trainer_conditioner_constructed"]
    assert counter.counts["model_constructions"] == 1
    assert counter.counts["conditioner_constructions"] == 1
    assert counter.counts["native_optimizer_constructions"] == 0


def test_scene_setup_is_counted_at_forward_not_assembler_completion(
        tmp_path, monkeypatch):
    import ireset.geometry.distance_adapter as D
    import ireset.kinematics.torch_adapter as TA

    calls: list[str] = []

    class FakeFK:
        def __init__(self, device="cpu", dtype=None):
            calls.append("fk")

    class OomAssembler:
        def __init__(self, *args, static_frame=None, **kwargs):
            calls.append("assembler")
            raise RuntimeError("simulated tensor transfer oom")

    def fake_scene(*, observer=None):
        calls.append("scene")
        if observer is not None:
            observer()
        return np.eye(3), np.zeros(3)

    monkeypatch.setattr(P, "LossAssembler", OomAssembler)
    monkeypatch.setattr(D, "largebox_static_frame", fake_scene)
    monkeypatch.setattr(TA, "G1TorchFK", FakeFK)
    provider = P.RealTrainingProvider(tmp_path)
    provider.tokenizer = StateTokenizer(
        np.tile(np.array([-1.0, 1.0]), (29, 1)),
        RootStats(mean=np.zeros(3), std=np.ones(3)))
    counter = A.CallCounter(A.CALL_CAPS)
    with pytest.raises(RuntimeError, match="oom"):
        provider._production_loss(counter)
    assert calls == ["fk", "scene", "assembler"]
    assert counter.counts["scene_setup_mj_forward"] == 1
    assert provider._production_assembler is None



    fake, state = _fake_mujoco()
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    _fake_g1spec(monkeypatch)
    counter = A.CallCounter(A.CALL_CAPS)
    # A guard that expires after the first state: only that mj_forward may
    # run, and the count is one (never the batch nominal).
    def guard(unit):
        if unit == "reference_mj_forward_1":
            raise TimeoutError("expired")

    backend = P.RealReferenceBackend(counter, deadline=None, guard=guard,
                                     joint_lo=np.full(29, -1.0),
                                     joint_hi=np.full(29, 1.0))
    pos = np.zeros((2, 3))
    R = np.tile(np.eye(3), (2, 1, 1))
    q = np.zeros((2, 29))
    with pytest.raises(TimeoutError):
        backend.positions(pos, R, q)
    assert counter.counts["reference_mj_forward"] == 1
    assert state["forward"] == 1
    # An absolute deadline is checked before every call.
    backend = P.RealReferenceBackend(counter, deadline=None,
                                     joint_lo=np.full(29, -1.0),
                                     joint_hi=np.full(29, 1.0))
    backend.deadline = A._now()
    with pytest.raises(TimeoutError):
        backend.positions(pos, R, q)
    assert counter.counts["reference_mj_forward"] == 1


# ---------------------------------------------------------------------------
# strict checkpoint payload
# ---------------------------------------------------------------------------


def _checkpoint_payload() -> dict:
    return {
        "model": {}, "conditioner": {}, "step": 4000,
        "config": {
            **{key: TINY_CONFIG[key] for key in
               ("micro_batch", "grad_accum", "lr", "weight_decay",
                "grad_clip", "d_model", "n_layers", "n_heads", "ffn_mult",
                "time_window", "num_train_timesteps", "beta_schedule",
                "beta_start", "beta_end")},
            "model_type": "deterministic",
        },
        "root_stats": {"mean": np.zeros(3), "std": np.ones(3)},
        "parameters_finite": True, "checkpoint_usable": True,
        "diagnostic_only": False, "failure": None,
        "identity": {"model_type": "deterministic"},
    }


def _checkpoint_provider(tmp_path, monkeypatch):
    provider = object.__new__(P.RealTrainingProvider)
    provider.root = tmp_path
    provider.config = dict(TINY_CONFIG)
    provider.device = "cpu"
    provider.counter = None
    provider._deadline = None
    provider._guard = None
    monkeypatch.setattr(P, "MODEL_CONFIG", dict(TINY_CONFIG))
    provider.tokenizer = StateTokenizer(
        np.tile(np.array([-1.0, 1.0]), (29, 1)),
        RootStats(mean=np.zeros(3), std=np.ones(3)))
    path = tmp_path / "fake.pt"
    torch.save(_checkpoint_payload(), path)
    monkeypatch.setattr(P, "CHECKPOINTS", {
        "deterministic": (Path("fake.pt"), A.sha256_file(path)),
        "diffusion": (Path("fake.pt"), A.sha256_file(path)),
    })
    return provider, path


def test_checkpoint_payload_strict_rules(tmp_path, monkeypatch):
    provider, path = _checkpoint_provider(tmp_path, monkeypatch)
    counter = A.CallCounter(A.CALL_CAPS)
    payload = provider._load_checkpoint("deterministic", counter)
    assert payload["step"] == 4000
    assert counter.counts["checkpoint_loads"] == 1
    for mutation in ("step", "finite", "usable", "key", "root"):
        data = _checkpoint_payload()
        if mutation == "step":
            data["step"] = 3999
        elif mutation == "finite":
            data["parameters_finite"] = False
        elif mutation == "usable":
            data["diagnostic_only"] = True
        elif mutation == "key":
            data.pop("conditioner")
        else:
            data["root_stats"] = {"mean": np.ones(3), "std": np.ones(3)}
        torch.save(data, path)
        monkeypatch.setitem(
            P.CHECKPOINTS, "deterministic",
            (Path("fake.pt"), A.sha256_file(path)))
        with pytest.raises(ValueError):
            provider._load_checkpoint("deterministic",
                                      A.CallCounter(A.CALL_CAPS))


# ---------------------------------------------------------------------------
# materialization guards and source digests
# ---------------------------------------------------------------------------


def test_real_materialize_checks_guard_before_reading_anything(tmp_path):
    provider = P.RealTrainingProvider(tmp_path)
    calls: list[str] = []

    def guard(unit: str) -> None:
        calls.append(unit)
        raise TimeoutError(unit)

    with pytest.raises(TimeoutError):
        provider.materialize(
            stage="P", r0_noise=[],
            counter=A.CallCounter(A.CALL_CAPS), guard=guard,
            deadline=time.perf_counter() + 60.0)
    # The first guard unit is the materialization entry; no array was read.
    assert calls == ["materialize_start"]
    assert provider.window_arrays == {}

    expired = P.RealTrainingProvider(tmp_path)
    with pytest.raises(TimeoutError):
        expired.materialize(
            stage="P", r0_noise=[],
            counter=A.CallCounter(A.CALL_CAPS),
            deadline=time.perf_counter() - 1.0)


def test_real_materialize_rehashes_parent_arrays_before_decode(
        tmp_path, monkeypatch):
    arrays_path = tmp_path / "arrays.npz"
    np.savez(arrays_path,
             **{key: np.zeros(1) for key in P.WINDOW_VALUE_KEYS})
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config")
    parents = dict(P.PARENT_FILES)
    expected = A.sha256_file(arrays_path)
    parents["objective_arrays"] = (Path("arrays.npz"), expected)
    monkeypatch.setattr(P, "PARENT_FILES", parents)
    monkeypatch.setattr(P, "TRAIN_CONFIG_PATH", Path("config.yaml"))
    monkeypatch.setattr(P, "TRAIN_CONFIG_SHA", A.sha256_file(config_path))
    provider = P.RealTrainingProvider(tmp_path)
    arrays = provider._load_arrays()
    assert set(arrays) == set(P.WINDOW_VALUE_KEYS)
    assert provider.source_sha256 == expected

    with open(arrays_path, "ab") as handle:
        handle.write(b"drift")
    monkeypatch.setattr(
        P.np, "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("payload decode must not run")))
    with pytest.raises(ValueError, match="drifted"):
        provider._load_arrays()


def test_build_and_checkpoint_units_recheck_the_guard(tmp_path):
    provider = P.RealTrainingProvider(tmp_path)
    provider.tokenizer = StateTokenizer(
        np.tile(np.array([-1.0, 1.0]), (29, 1)),
        RootStats(mean=np.zeros(3), std=np.ones(3)))
    units: list[str] = []

    def guard(unit: str) -> None:
        units.append(unit)
        raise TimeoutError(unit)

    provider._guard = guard
    counter = A.CallCounter(A.CALL_CAPS)
    with pytest.raises(TimeoutError):
        provider.build_trainer("deterministic", with_optimizer=True,
                               counter=counter)
    assert units == ["trainer_construction"]
    assert counter.counts["model_constructions"] == 0

    units.clear()
    with pytest.raises(TimeoutError):
        provider._load_checkpoint("deterministic", counter)
    assert units == ["checkpoint_sha"]
    assert counter.counts["checkpoint_loads"] == 0


def test_fk_deadline_stops_the_scene_compile(tmp_path, monkeypatch):
    import ireset.geometry.distance_adapter as D
    import ireset.kinematics.torch_adapter as TA

    calls: list[str] = []

    class FakeFK:
        def __init__(self, device="cpu", dtype=None):
            calls.append("fk")

        def site_world_positions(self, *args, **kwargs):
            raise AssertionError("FK must not run during construction")

    class FakeAssembler:
        def __init__(self, *args, static_frame=None, **kwargs):
            calls.append("assembler")

    def fake_scene(*, observer=None):
        calls.append("scene")
        if observer is not None:
            observer()
        return np.eye(3), np.zeros(3)

    monkeypatch.setattr(TA, "G1TorchFK", FakeFK)
    monkeypatch.setattr(D, "largebox_static_frame", fake_scene)
    monkeypatch.setattr(P, "LossAssembler", FakeAssembler)
    provider = P.RealTrainingProvider(tmp_path)
    provider.tokenizer = StateTokenizer(
        np.tile(np.array([-1.0, 1.0]), (29, 1)),
        RootStats(mean=np.zeros(3), std=np.ones(3)))
    counter = A.CallCounter(A.CALL_CAPS)

    def expired(unit: str) -> None:
        if unit == "production_scene_setup":
            raise TimeoutError(unit)

    provider._guard = expired
    with pytest.raises(TimeoutError):
        provider._production_loss(counter)
    assert calls == ["fk"]
    assert counter.counts["scene_setup_mj_forward"] == 0
    assert provider._production_assembler is None

    # Once the compile completes it stays counted, even if a later unit
    # refuses to continue.
    calls.clear()
    provider._guard = None
    provider._production_loss(counter)
    assert calls == ["fk", "scene", "assembler"]
    assert counter.counts["scene_setup_mj_forward"] == 1


def test_checkpoint_state_contract_checks_dtype_shape_and_finite():
    module = torch.nn.Linear(3, 2)
    state = module.state_dict()
    P._copy_state_strict(module, state, "model")
    float64 = {name: value.to(torch.float64)
               for name, value in state.items()}
    with pytest.raises(ValueError, match="model_dtype:weight"):
        P._copy_state_strict(module, float64, "model")
    nonfinite = {name: value.clone() for name, value in state.items()}
    nonfinite["weight"].fill_(float("nan"))
    with pytest.raises(ValueError, match="model_nonfinite:weight"):
        P._copy_state_strict(module, nonfinite, "model")
    wrong_shape = {name: value.clone() for name, value in state.items()}
    wrong_shape["weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="model_shape:weight"):
        P._copy_state_strict(module, wrong_shape, "model")
    missing = {name: value.clone() for name, value in state.items()}
    missing.pop("bias")
    with pytest.raises(ValueError, match="model_missing:bias"):
        P._copy_state_strict(module, missing, "model")


def test_materialize_rechecks_the_frozen_asset_closure(tmp_path, monkeypatch):
    import ireset.eval.w1_real_fk_audit as F

    (tmp_path / "lock_record.json").write_text("{}")
    (tmp_path / "robot.urdf").write_text("<robot/>")
    (tmp_path / "scene_body.xml").write_text("<mujoco/>")
    assets = tmp_path / "assets"
    assets.mkdir()
    mesh = assets / "a.obj"
    mesh.write_text("mesh")
    (tmp_path / "include.xml").write_text("<mujoco/>")
    (tmp_path / "scene.xml").write_text(
        '<mujoco><compiler meshdir="assets"/>'
        '<include file="include.xml"/>'
        '<asset><mesh file="a.obj"/></asset></mujoco>')
    monkeypatch.setattr(F, "LOCKED_ASSETS", {
        "lock_record": Path("lock_record.json"),
        "xml": Path("scene.xml"),
        "scene_xml": Path("scene_body.xml"),
        "urdf": Path("robot.urdf"),
    })
    facts = F._asset_closure_facts(tmp_path, mode="real")
    assert facts["problems"] == []
    assert facts["reference_count"] == 2
    registration = tmp_path / "registration.json"
    P.write_json(registration, {"locked_assets": facts})
    parents = dict(P.PARENT_FILES)
    parents["fk_registration"] = (Path("registration.json"),
                                  A.sha256_file(registration))
    monkeypatch.setattr(P, "PARENT_FILES", parents)
    provider = P.RealTrainingProvider(tmp_path)
    provider._verify_asset_closure()

    units: list[str] = []
    provider._guard = lambda unit: units.append(unit)
    provider._verify_asset_closure()
    assert units == ["materialize_asset_closure"]

    mesh.write_text("drifted")
    with pytest.raises(ValueError, match="asset_reference_sha"):
        provider._verify_asset_closure()

    mesh.write_text("mesh")
    (tmp_path / "include.xml").unlink()
    with pytest.raises(ValueError, match="asset_reference_missing"):
        provider._verify_asset_closure()


# ---------------------------------------------------------------------------
# resume / attempt selection / 16-vs-17 accounting
# ---------------------------------------------------------------------------


def test_resume_p_retry_keeps_selected_16_and_cohort_17(tmp_path, monkeypatch):
    provider = _provider()
    root = tmp_path / "cohort"
    original = A.run_p_stage
    state = {"first": True}

    def failing_p(*args, **kwargs):
        if state["first"]:
            state["first"] = False
            counter = args[6]
            counter.add("native_adamw_steps")
            counter.add("diagnostic_parameter_update_events")
            return {"passed": False, "failures": ["injected_p_failure"],
                    "gaps": [], "rows": [], "paths": {}, "report": None,
                    "arrays_sha256": {}, "complete_evidence": False,
                    "numeric_passed": False}
        return original(*args, **kwargs)

    monkeypatch.setattr(A, "run_p_stage", failing_p)
    first = A.run_registered_audit(root, provider, mode="synthetic",
                                   tolerance=_tolerance())
    assert first["conclusion"] == "audit_incomplete"
    p_attempts = [r for r in A.load_index(root) if r["stage"] == "P"]
    assert len(p_attempts) == 1 and p_attempts[0]["status"] == "incomplete"

    monkeypatch.setattr(A, "run_p_stage", original)
    second = A.run_registered_audit(root, provider, mode="synthetic",
                                    tolerance=_tolerance(), resume_stage="P")
    assert second["complete_evidence"] is True, second["problems"]
    assert second["selected_attempts"]["P"] == "attempt_002"
    assert second["selected_chain_training_updates"] == 16
    assert second["training_updates"] == 17
    assert second["selected_chain_counts"]["native_adamw_steps"] == 8
    assert second["selected_chain_counts"]["explicit_reference_updates"] == 8
    assert second["cohort_call_counts"]["native_adamw_steps"] == 9
    assert second["cohort_call_counts"]["explicit_reference_updates"] == 8
    # Both terminal attempts remain on disk; the failure is preserved.
    assert len([r for r in A.load_index(root)
                if r["stage"] == "P"]) == 2


def test_resume_f_reuses_upstream_and_selects_new_attempt(tmp_path,
                                                          monkeypatch):
    provider = _provider()
    root = tmp_path / "cohort"
    original = A.run_f_stage

    def failing_f(*args, **kwargs):
        return {"conclusion": "audit_incomplete",
                "complete_evidence": False,
                "covered_contract_passed": False,
                "verification_problems": ["injected_f_failure"],
                "failures": [], "gaps": ["injected_f_failure"],
                "report": None, "summary": {}, "worker": {},
                "call_delta": {kind: 0 for kind in A.CALL_KINDS}}

    monkeypatch.setattr(A, "run_f_stage", failing_f)
    first = A.run_registered_audit(root, provider, mode="synthetic",
                                   tolerance=_tolerance())
    assert first["conclusion"] == "audit_incomplete"
    assert not [r for r in A.load_index(root)
                if r["stage"] == "F" and r["status"] == "done"]
    monkeypatch.setattr(A, "run_f_stage", original)
    second = A.run_registered_audit(root, provider, mode="synthetic",
                                    tolerance=_tolerance(), resume_stage="F")
    assert second["complete_evidence"] is True, second["problems"]
    assert second["selected_attempts"]["F"] == "attempt_002"
    assert second["selected_attempts"]["R0"] == "attempt_001"
    assert second["selected_attempts"]["P"] == "attempt_001"
    assert second["selected_attempts"]["I"] == "attempt_001"
    assert second["selected_chain_training_updates"] == 16
    assert len([r for r in A.load_index(root)
                if r["stage"] == "F"]) == 2


def test_remaining_nominal_covers_only_stages_still_to_run():
    full = A.remaining_nominal_for_mode("real", None)
    assert full["checkpoint_loads"] == 8
    assert A.remaining_nominal_for_mode("real", "R0") == full
    assert A.remaining_nominal_for_mode("real", "P")["checkpoint_loads"] == 8
    assert A.remaining_nominal_for_mode("real", "I")["checkpoint_loads"] == 6
    assert A.remaining_nominal_for_mode("real", "F")["checkpoint_loads"] == 4
    assert A.remaining_nominal_for_mode("synthetic", "F")[
        "scene_setup_mj_forward"] == 0


def test_resume_f_only_requires_remaining_stage_calls(tmp_path, monkeypatch):
    provider = _provider()
    root = tmp_path / "cohort"
    original_p = A.run_p_stage
    original_f = A.run_f_stage
    state = {"p": 0, "f": 0}

    def failing_p(*args, **kwargs):
        if state["p"] == 0:
            state["p"] = 1
            args[6].add("checkpoint_loads", 2)
            return {"passed": False, "failures": ["injected_p"], "gaps": [],
                    "rows": [], "paths": {}, "report": None,
                    "arrays_sha256": {}, "complete_evidence": False,
                    "numeric_passed": False}
        return original_p(*args, **kwargs)

    def failing_f(*args, **kwargs):
        if state["f"] == 0:
            state["f"] = 1
            args[6].add("checkpoint_loads", 4)
            delta = {kind: 0 for kind in A.CALL_KINDS}
            delta["checkpoint_loads"] = 4
            return {"conclusion": "audit_incomplete",
                    "complete_evidence": False,
                    "covered_contract_passed": False,
                    "verification_problems": ["injected_f_failure"],
                    "failures": [], "gaps": ["injected_f_failure"],
                    "report": None, "summary": {}, "worker": {},
                    "call_delta": delta}
        return original_f(*args, **kwargs)

    monkeypatch.setattr(A, "run_p_stage", failing_p)
    first = A.run_registered_audit(root, provider, mode="synthetic",
                                   tolerance=_tolerance())
    assert first["conclusion"] == "audit_incomplete"
    monkeypatch.setattr(A, "run_p_stage", original_p)
    monkeypatch.setattr(A, "run_f_stage", failing_f)
    second = A.run_registered_audit(root, provider, mode="synthetic",
                                    tolerance=_tolerance(), resume_stage="P")
    assert second["conclusion"] == "audit_incomplete"
    p_row, p_problems = A.stage_state(root, "P")
    assert p_problems == [] and p_row["attempt"] == "attempt_002"
    # History: failed P 2 + retry P 2 + I 2 + failed F 4 = 10 loads.  A
    # resumed F needs 4, so 14 <= 16; the old whole-chain requirement (8)
    # would have refused a legitimate resume.
    monkeypatch.setattr(A, "run_f_stage", original_f)
    third = A.run_registered_audit(root, provider, mode="synthetic",
                                   tolerance=_tolerance(), resume_stage="F")
    assert third["complete_evidence"] is True, third["problems"]
    assert third["selected_attempts"]["F"] == "attempt_002"
    assert third["calls"]["counts"]["checkpoint_loads"] == 14


def test_resume_uses_a_fresh_start_after_the_resumed_stage(tmp_path,
                                                            monkeypatch):
    provider = _provider()
    root = tmp_path / "cohort"
    original_p = A.run_p_stage
    state = {"first": True}

    def failing_p(*args, **kwargs):
        if state["first"]:
            state["first"] = False
            return {"passed": False, "failures": ["injected_p"], "gaps": [],
                    "rows": [], "paths": {}, "report": None,
                    "arrays_sha256": {}, "complete_evidence": False,
                    "numeric_passed": False}
        return original_p(*args, **kwargs)

    monkeypatch.setattr(A, "run_p_stage", failing_p)
    A.run_registered_audit(root, provider, mode="synthetic",
                           tolerance=_tolerance())
    monkeypatch.setattr(A, "run_p_stage", original_p)
    starts: dict[str, float] = {}
    original_pre = A.pre_register_attempt

    def recording_pre(root_, stage, attempt, t0, **kwargs):
        starts[stage] = t0
        return original_pre(root_, stage, attempt, t0, **kwargs)

    monkeypatch.setattr(A, "pre_register_attempt", recording_pre)
    result = A.run_registered_audit(root, provider, mode="synthetic",
                                    tolerance=_tolerance(), resume_stage="P")
    assert result["complete_evidence"] is True, result["problems"]
    assert starts["P"] < starts["I"] < starts["F"]


def test_resume_rejects_unknown_or_already_consumable_stage(tmp_path,
                                                            monkeypatch):
    provider = _provider()
    root = tmp_path / "cohort"
    result = A.run_registered_audit(root, provider, mode="synthetic",
                                    tolerance=_tolerance())
    assert result["complete_evidence"] is True
    with pytest.raises(SystemExit):
        A.run_registered_audit(root, provider, mode="synthetic",
                               tolerance=_tolerance(), resume_stage="F")
    # A corrupted ledger makes history unknown and refuses before any work.
    ledger = A.load_ledger(root)
    ledger[0]["status"] = "running"
    A.save_ledger(root, ledger)
    with pytest.raises(SystemExit):
        A.run_registered_audit(root, provider, mode="synthetic",
                               tolerance=_tolerance(), resume_stage="P")


def _manual_attempt(root: Path, stage: str, attempt: str, *,
                    complete: bool) -> None:
    budget = _budget()
    A.pre_register_attempt(root, stage, attempt, A._now(), scope={},
                           budget=budget, call_balance=_zero_snapshot())
    A.reserve_stage_calls(root, stage, attempt,
                          {"counts": {kind: 0 for kind in A.CALL_KINDS}})
    A.settle_stage_calls(root, stage, attempt,
                         {"counts": {kind: 0 for kind in A.CALL_KINDS}})
    directory = A.stage_dir(root, stage) / attempt
    prefix = "p" if stage == "P" else "i"
    report = directory / f"{prefix}_report.json"
    (directory / f"{prefix}_rows.json").write_text("[]")
    models = {}
    for model in ("deterministic", "diffusion"):
        arrays = directory / f"{prefix}_{model}.npz"
        arrays.write_bytes(f"{prefix}:{model}".encode())
        models[model] = {"arrays_path": A.path_for_record(arrays),
                         "arrays_sha256": A.sha256_file(arrays)}
    report_data = {"complete_evidence": complete, "models": models}
    if stage == "P":
        report_data["input_admission"] = None
    A.write_json(report, report_data)
    artifacts = {
        A.path_for_record(path): A.sha256_file(path)
        for path in directory.rglob("*") if path.is_file()
    }
    _finish(root, stage, attempt, complete, report, artifacts=artifacts)


def test_stage_state_rejects_multiple_consumable_attempts(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    _manual_attempt(root, "P", "attempt_001", complete=True)
    _manual_attempt(root, "P", "attempt_002", complete=True)
    row, problems = A.stage_state(root, "P")
    assert row is None
    assert "stage_multiple_consumable" in problems
    # An explicitly requested attempt cannot bypass the uniqueness check.
    row, problems = A.stage_state(root, "P", "attempt_002")
    assert row is None
    assert "stage_multiple_consumable" in problems


def test_incomplete_attempt_is_never_consumable(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    _manual_attempt(root, "P", "attempt_001", complete=False)
    row, problems = A.stage_state(root, "P")
    assert row is None
    assert "attempt_not_complete" in problems
    row, problems = A.stage_state(root, "P", "attempt_001")
    assert row is None
    assert "attempt_not_complete" in problems
    # Two incomplete attempts stay unconsumable exactly like one.
    _manual_attempt(root, "P", "attempt_002", complete=False)
    row, problems = A.stage_state(root, "P", "attempt_001")
    assert row is None
    assert "attempt_not_complete" in problems
    row, problems = A.stage_state(root, "P")
    assert row is None
    assert problems


def test_stage_state_allows_one_failure_plus_one_consumable(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    _manual_attempt(root, "P", "attempt_001", complete=False)
    _manual_attempt(root, "P", "attempt_002", complete=True)
    row, problems = A.stage_state(root, "P")
    assert problems == []
    assert row["attempt"] == "attempt_002"
    row, problems = A.stage_state(root, "P", "attempt_001")
    assert row is None
    assert "attempt_not_complete" in problems


def test_selected_chain_and_caps_are_separate():
    deltas = {stage: {} for stage in A.STAGE_BUDGETS}
    selected = {}
    for stage in A.STAGE_BUDGETS:
        attempt = f"{stage.lower()}_001"
        selected[stage] = attempt
        deltas[stage][attempt] = {
            "counts": A.stage_nominal_for_mode("synthetic", stage)}
    assert A._selected_stage_problems(deltas, selected, "synthetic") == []
    assert A._selected_caps_problems(deltas, "synthetic") == []
    # A stage failure adds partial calls to the cohort/stage totals without
    # changing the selected chain.
    extra = {"counts": {kind: 0 for kind in A.CALL_KINDS}}
    extra["counts"]["native_adamw_steps"] = 1
    deltas["P"]["p_failed"] = extra
    assert A._selected_stage_problems(deltas, selected, "synthetic") == []
    assert A.cohort_call_totals_from_deltas(deltas)["native_adamw_steps"] == 9
    # The selected chain still has the exact nominal matrix.
    chain = A.selected_chain_counts(deltas, selected)
    assert chain["native_adamw_steps"] == 8
    assert chain["explicit_reference_updates"] == 8
    bad = copy.deepcopy(deltas)
    bad["P"]["p_failed"]["counts"]["native_adamw_steps"] = 100
    assert any(p.startswith("stage_cap:P")
               for p in A._selected_caps_problems(bad, "synthetic"))


# ---------------------------------------------------------------------------
# admission / acceptance schemas
# ---------------------------------------------------------------------------


def test_real_admission_rejects_synthetic_and_missing_asset_closure():
    synthetic = {"schema": A.ADMISSION_SCHEMA, "passed": True,
                 "problems": []}
    assert "admission_schema" in P.real_admission_package_problems(synthetic)
    minimal = {
        "schema": P.REAL_ADMISSION_SCHEMA, "provider_mode": "real",
        "scope": "metadata_only", "passed": True, "problems": [],
        "status": {"authorized": False, "registered": False,
                   "run": "not_run", "section_10_2_checked": False},
        "read_scope": {"metadata_only": True, "npz_values_read": 0,
                       "checkpoint_payload_read": 0, "torch_load_calls": 0,
                       "model_constructor_calls": 0,
                       "muJoCo_compile_calls": 0,
                       "conditioner_forward_calls": 0,
                       "model_forward_calls": 0, "vjp_calls": 0,
                       "optimizer_steps": 0, "cohort_created": False,
                       "registered_attempt_created": False,
                       "asset_closure_verified": False},
    }
    problems = P.real_admission_package_problems(minimal)
    assert "admission_asset_closure" in problems


def test_synthetic_acceptance_schema_rejects_incomplete_stages():
    record = {
        "schema": P.SYNTHETIC_ACCEPTANCE_SCHEMA, "mode": "synthetic",
        "passed": True, "problems": [], "complete_evidence": True,
        "stages": {stage: {"sealed": True, "terminal": True,
                           "complete_evidence": True, "failures": 0,
                           "seal_problems": []}
                   for stage in A.STAGE_BUDGETS},
        "selected_chain_counts": {**A.chain_nominal_for_mode("synthetic")},
        "selected_chain_training_updates": 16,
        "roster": {}, "tolerance": {}, "calibration_verification": {},
        "read_scope": {"real_checkpoint_loaded": False,
                       "real_asset_loaded": False,
                       "real_dataset_loaded": False, "real_model_calls": 0},
    }
    problems = P.synthetic_acceptance_problems(record)
    assert "acceptance_roster" in problems
    bad = copy.deepcopy(record)
    bad["stages"]["F"]["sealed"] = False
    assert "acceptance_stage_seal:F" in P.synthetic_acceptance_problems(bad)


def test_calibration_verification_rejects_real_reads():
    record = {
        "schema": P.CALIBRATION_VERIFICATION_SCHEMA,
        "mode": "synthetic_calibrated", "passed": True,
        "recompute_problems": [], "four_x_ok": True,
        "real_dataset_loaded": False, "real_checkpoint_loaded": True,
        "real_asset_loaded": False, "real_model_calls": 0,
        "contract_sha256": "a" * 64, "evidence_sha256": "b" * 64,
        "fp32_contract_sha256": "c" * 64,
    }
    assert "calibration_verification_real_reads" in \
        P.calibration_verification_problems(record)


def test_runtime_identity_drift_is_refused():
    current = A.runtime_identity_facts()
    assert A.runtime_identity_problems(current, current) == []
    tampered = copy.deepcopy(current)
    tampered["torch"] = "0.0.0"
    assert "runtime_identity:torch" in A.runtime_identity_problems(
        tampered, current)
    tampered = copy.deepcopy(current)
    tampered["mujoco"] = "0.0.0"
    assert "runtime_identity:mujoco" in A.runtime_identity_problems(
        tampered, current)
    tampered = copy.deepcopy(current)
    tampered["cpu_threads"] = int(current["cpu_threads"]) + 1
    assert "runtime_identity:cpu_threads" in A.runtime_identity_problems(
        tampered, current)
    tampered = copy.deepcopy(current)
    tampered["adamw_source_sha256"]["torch.optim.adam"] = "0" * 64
    assert "runtime_identity:source:torch.optim.adam" in \
        A.runtime_identity_problems(tampered, current)
    assert A.runtime_identity_problems(None, current) == [
        "runtime_identity_missing"]
    contract = A.build_fp32_contract("cuda")
    assert A.fp32_contract_problems(contract, recompute=False) == []
    tampered_contract = copy.deepcopy(contract)
    tampered_contract["runtime_identity"]["python"] = "0.0"
    assert "fp32_runtime_identity:python" in A.fp32_contract_problems(
        tampered_contract, recompute=False)


def test_implementation_files_cover_calculation_dependencies():
    required = {
        "src/ireset/model/tokens.py",
        "src/ireset/model/deterministic.py",
        "src/ireset/kinematics/torch_adapter.py",
        "src/ireset/kinematics/rotation_adapter.py",
        "src/ireset/geometry/distance_adapter.py",
        "src/ireset/embodied/g1_spec.py",
        "src/ireset/paths.py",
        "src/ireset/eval/training_objective_audit.py",
        "pyproject.toml",
        "uv.lock",
    }
    assert required <= set(A.IMPLEMENTATION_FILES)
    identity = A.implementation_identity(Path(__file__).resolve().parents[1])
    assert all(identity["files"].get(rel) for rel in required)


def _acceptance_fixture(tmp_path):
    run_root = tmp_path / "synthetic_run"
    selected = {stage: "attempt_001" for stage in A.STAGE_BUDGETS}
    report_names = {"R0": "admission.json", "P": "p_report.json",
                    "I": "i_report.json",
                    "F": "training_update_final.json"}
    stages = {}
    for stage in A.STAGE_BUDGETS:
        directory = run_root / A.STAGE_DIRS[stage] / "attempt_001"
        for name in P.STAGE_REQUIRED_FILES[stage]:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{stage}:{name}")
        stages[stage] = {
            "complete_evidence": True, "failures": [], "gaps": [],
            "report": A.path_for_record(directory / report_names[stage]),
            "worker_result": {"pid": 2, "ppid": 1},
        }
    fixture = run_root / "fixture"
    fixture.mkdir(parents=True)
    for name in P.FIXTURE_REQUIRED_FILES:
        (fixture / name).write_text(name)
    for name in P.RUN_REQUIRED_FILES:
        (run_root / name).write_text(name)
    result = {
        "stages": stages,
        "seals": {stage: {"sealed": True, "terminal": True, "problems": []}
                  for stage in A.STAGE_BUDGETS},
        "selected_attempts": selected,
        "selected_chain_counts": A.chain_nominal_for_mode("synthetic"),
        "complete_evidence": True, "problems": [], "failures": [],
        "conclusion": "audit_incomplete",
    }
    roster = A.build_parameter_roster(TINY_CONFIG)
    tolerance = {"mode": "synthetic_calibrated",
                 "evidence_sha256": "a" * 64,
                 "fp32_contract_sha256": "b" * 64}
    calibration = {
        "schema": P.CALIBRATION_VERIFICATION_SCHEMA,
        "mode": "synthetic_calibrated", "passed": True,
        "recompute_problems": [], "four_x_ok": True,
        "real_dataset_loaded": False, "real_checkpoint_loaded": False,
        "real_asset_loaded": False, "real_model_calls": 0,
        "contract_sha256": "c" * 64, "evidence_sha256": "a" * 64,
        "fp32_contract_sha256": "b" * 64,
        "runtime_identity": A.runtime_identity_facts(),
    }
    return run_root, result, roster, tolerance, calibration


def test_synthetic_acceptance_binds_every_evidence_file(tmp_path):
    run_root, result, roster, tolerance, calibration = \
        _acceptance_fixture(tmp_path)
    record = P.synthetic_acceptance(
        result, run_root=run_root, tolerance_contract=tolerance,
        calibration_verification_record=calibration, roster=roster)
    assert record["passed"] is True, record["problems"]
    assert P.synthetic_acceptance_problems(record) == []

    # Deleting any evidence file breaks the binding.
    victim = run_root / A.STAGE_DIRS["P"] / "attempt_001" / "p_report.json"
    victim.unlink()
    problems = P.synthetic_acceptance_problems(record)
    assert any("file_missing" in p for p in problems)

    # Restoring the path with different bytes is caught by the digest.
    victim.write_text("tampered")
    problems = P.synthetic_acceptance_problems(record)
    assert any("file_sha" in p for p in problems)

    # Replacing every report path also fails (the manifest no longer matches).
    tampered = copy.deepcopy(record)
    for stage in A.STAGE_BUDGETS:
        tampered["stages"][stage]["report"] = str(
            tmp_path / "missing" / "report.json")
    problems = P.synthetic_acceptance_problems(tampered)
    assert any("acceptance_stage_report" in p for p in problems)

    # Runtime drift is a Gate B return, not a warning.
    drifted = copy.deepcopy(record)
    drifted["runtime_identity"]["torch"] = "0.0.0"
    assert any("acceptance_runtime:runtime_identity:torch" in p
               for p in P.synthetic_acceptance_problems(drifted))


def test_worker_aborts_before_replay_when_source_evidence_drifted(
        tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    A.write_cohort(root, mode="real")
    attempt = "attempt_001"
    A.pre_register_attempt(root, "F", attempt, A._now(), scope={},
                           budget=_budget())
    A.reserve_stage_calls(
        root, "F", attempt,
        {"counts": A.stage_nominal_for_mode("real", "F")})
    noise = tmp_path / "noise_schedule.npz"
    noise.write_bytes(b"noise")
    admission = tmp_path / "input_admission.json"
    A.write_json(admission, {"passed": False})
    spec_path = tmp_path / "worker_spec.json"
    A.write_json(spec_path, {
        "mode": "real", "run_root": str(root), "stage": "F",
        "attempt": attempt, "deadline_perf_counter": A._now() + 60.0,
        "reserve_s": 0.0, "remaining_caps": dict(A.CALL_CAPS),
        "registration_sha256": None, "candidate_attempts": {},
        "candidate_checks": False, "project_root": str(tmp_path),
        "roster": {}, "config": TINY_CONFIG, "device": "cpu",
        "provider": {"runtime_identity": A.runtime_identity_facts()},
        "r0_noise_path": str(noise), "r0_noise_sha256": "0" * 64,
        "p_input_admission_path": str(admission),
        "p_input_admission_sha256": "1" * 64,
    })

    class FakeProvider:
        roster: dict = {}
        noise: list = []

        def load_noise_schedule(self, path):
            raise AssertionError("drifted noise must not be decoded")

    monkeypatch.setattr(P.RealTrainingProvider, "from_spec",
                        classmethod(lambda cls, *a, **k: FakeProvider()))

    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not start")

    monkeypatch.setattr(A, "run_p_stage", forbidden)
    out = tmp_path / "worker"
    assert A._worker_main(str(spec_path), str(out)) == 2
    result = A.read_json(out / "worker_result.json")
    assert {"schedule_source_sha", "p_input_admission_sha",
            "p_input_admission_not_passed"} <= set(result["problems"])
    counts = result["counts"]["counts"]
    assert counts["cpu_draw_units"] == A.DRAW_UNITS_PER_PATH
    assert all(int(value) == 0 for kind, value in counts.items()
               if kind != "cpu_draw_units")


def test_from_spec_restores_the_acceptance_identity(tmp_path):
    run_root, result, roster, tolerance, calibration = \
        _acceptance_fixture(tmp_path)
    record = P.synthetic_acceptance(
        result, run_root=run_root, tolerance_contract=tolerance,
        calibration_verification_record=calibration, roster=roster)
    path = tmp_path / "acceptance.json"
    P.write_json(path, record)
    spec = {"mode": "real", "config": TINY_CONFIG, "device": "cpu",
            "acceptance_path": "acceptance.json",
            "acceptance_sha256": A.sha256_file(path)}
    provider = P.RealTrainingProvider.from_spec(spec, project_root=tmp_path)
    assert provider.acceptance_path == path
    assert provider.acceptance is not None
    assert provider.acceptance_sha == A.sha256_file(path)

    drifted = dict(spec)
    drifted["acceptance_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha drifted"):
        P.RealTrainingProvider.from_spec(drifted, project_root=tmp_path)


def test_worker_aborts_on_runtime_identity_before_real_calls(
        tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    A.write_cohort(root, mode="real")
    attempt = "attempt_001"
    A.pre_register_attempt(root, "F", attempt, A._now(), scope={},
                           budget=_budget())
    A.reserve_stage_calls(
        root, "F", attempt,
        {"counts": A.stage_nominal_for_mode("real", "F")})
    spec_path = tmp_path / "worker_spec.json"
    A.write_json(spec_path, {
        "mode": "real", "run_root": str(root), "stage": "F",
        "attempt": attempt, "deadline_perf_counter": A._now() + 60.0,
        "reserve_s": 0.0, "remaining_caps": dict(A.CALL_CAPS),
        "registration_sha256": None, "candidate_attempts": {},
        "candidate_checks": False, "project_root": str(tmp_path),
        "roster": {}, "config": TINY_CONFIG, "device": "cpu",
        "provider": {"runtime_identity": {}},
    })

    def forbidden(*args, **kwargs):
        raise AssertionError("real provider must not be constructed")

    monkeypatch.setattr(P.RealTrainingProvider, "from_spec",
                        classmethod(forbidden))
    out = tmp_path / "worker"
    assert A._worker_main(str(spec_path), str(out)) == 2
    result = A.read_json(out / "worker_result.json")
    assert any(p.startswith("worker_runtime_identity:")
               for p in result["problems"])
    assert result["provider"] is None
    assert all(int(value) == 0 for value in result["counts"]["counts"].values())


def test_calibration_verification_rejects_runtime_drift():
    record = {
        "schema": P.CALIBRATION_VERIFICATION_SCHEMA,
        "mode": "synthetic_calibrated", "passed": True,
        "recompute_problems": [], "four_x_ok": True,
        "real_dataset_loaded": False, "real_checkpoint_loaded": False,
        "real_asset_loaded": False, "real_model_calls": 0,
        "contract_sha256": "a" * 64, "evidence_sha256": "b" * 64,
        "fp32_contract_sha256": "c" * 64,
        "runtime_identity": A.runtime_identity_facts(),
    }
    assert P.calibration_verification_problems(record) == []
    drifted = copy.deepcopy(record)
    drifted["runtime_identity"]["cuda_version"] = "0.0"
    assert "calibration_runtime_identity:cuda_version" in \
        P.calibration_verification_problems(drifted)


def test_cli_refuses_to_overwrite_calibration(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_cli", Path(__file__).resolve().parents[1] /
        "scripts/audit_w1_real_training_update.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    target = tmp_path / "calibration" / "tolerance_contract.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "audit", "--calibrate-synthetic", "--tolerance-out", str(target)])
    with pytest.raises(SystemExit):
        cli.main()


# ---------------------------------------------------------------------------
# F timing: worker precheck -> settle -> terminal check -> selection
# ---------------------------------------------------------------------------


def test_f_worker_exposes_candidate_and_provisional_counts(tmp_path,
                                                           monkeypatch):
    provider = _provider()
    root = tmp_path / "cohort"
    result = A.run_registered_audit(root, provider, mode="synthetic",
                                    tolerance=_tolerance())
    assert result["complete_evidence"] is True
    final = A.read_json(Path(result["summary_path"]))
    assert final["selected_attempts"] == result["selected_attempts"]
    assert final["selected_attempts"]["F"] == "attempt_001"
    worker = final["worker_result"]
    assert worker["candidate_attempts"]["P"] == "attempt_001"
    assert worker["candidate_attempts"]["F"] == "attempt_001"
    assert worker["provisional_chain_counts"] == result["selected_chain_counts"]
    assert worker["provisional_training_updates"] == 16
    assert worker["stage_call_counts"]["P"]["native_adamw_steps"] == 4
    assert worker["cohort_call_counts"]["native_adamw_steps"] == 8
    assert final["training_updates_scope"] == "cohort"
    assert final["training_updates"] == 16
    # Event order: worker snapshot on disk, parent settlement, final seal.
    worker_dir = (A.stage_dir(root, "F") / "attempt_001" / "worker")
    assert (worker_dir / "worker_result.json").is_file()
    reservation = A.read_json(
        A.stage_dir(root, "F") / "attempt_001" / "call_reservations.json")
    assert reservation["consumed"] is not None
    row = [r for r in A.load_index(root)
           if r["stage"] == "F" and r["attempt"] == "attempt_001"][0]
    assert row["status"] == "done"
    assert row["call_balance_before"]["counts"]["native_adamw_steps"] == 4
    assert row["call_balance_after"]["counts"]["native_adamw_steps"] == 8


def test_f_worker_candidate_mutation_is_vetoed(tmp_path, monkeypatch):
    provider = _provider()
    root = tmp_path / "cohort"
    original = A.run_f_stage

    def tampered_f(*args, **kwargs):
        candidates = dict(kwargs.get("candidate_attempts") or {})
        candidates["P"] = "attempt_999"
        kwargs["candidate_attempts"] = candidates
        return original(*args, **kwargs)

    monkeypatch.setattr(A, "run_f_stage", tampered_f)
    result = A.run_registered_audit(root, provider, mode="synthetic",
                                    tolerance=_tolerance())
    assert result["complete_evidence"] is False
    assert result["conclusion"] == "audit_incomplete"
    assert result["selected_attempts"] is None
    assert any("f_candidate" in p for p in result["problems"])
    rows = [r for r in A.load_index(root) if r["stage"] == "F"]
    assert rows and rows[0]["status"] == "incomplete"
    assert rows[0]["complete"] is False
