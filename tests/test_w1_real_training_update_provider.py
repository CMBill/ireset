"""Gate A/B tests for the real provider, resume protocol and F timing.

Everything here is synthetic: temporary files, analytic four-window fixtures,
metadata stubs and a fake MuJoCo module.  No real checkpoint payload, parent
NPZ value, G1 mesh or compiled asset is read.
"""

from __future__ import annotations

import copy
import json
import sys
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
            report: Path | None, t0: float | None = None) -> None:
    A.finish_attempt(
        root, stage, attempt, t0 if t0 is not None else A._now(),
        _budget(), {"complete": complete,
                    "report": A.path_for_record(report)
                    if report is not None else None,
                    "summary": A.path_for_record(report)
                    if report is not None else None},
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


def _fake_g1spec(monkeypatch):
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
        def load_plain(self):
            return Model()

    bodies = {name: f"b{i}" for i, name in enumerate(A.SITE_NAMES)}
    monkeypatch.setattr(g1_spec, "G1Spec", Spec)
    monkeypatch.setattr(g1_spec, "KPT_BODIES", bodies)
    monkeypatch.setattr(g1_spec, "JOINT_NAMES", [f"j{i}" for i in range(29)])


def test_real_reference_backend_partial_counts_and_guards(monkeypatch):
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
    backend = P.RealReferenceBackend(counter, deadline=A._now(),
                                     joint_lo=np.full(29, -1.0),
                                     joint_hi=np.full(29, 1.0))
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
    report = directory / "report.json"
    A.write_json(report, {"complete_evidence": complete})
    _finish(root, stage, attempt, complete, report)


def test_stage_state_rejects_multiple_consumable_attempts(tmp_path):
    root = tmp_path / "run"
    A.write_cohort(root, mode="synthetic")
    _manual_attempt(root, "P", "attempt_001", complete=True)
    _manual_attempt(root, "P", "attempt_002", complete=True)
    row, problems = A.stage_state(root, "P")
    assert row is None
    assert "stage_multiple_consumable" in problems


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
