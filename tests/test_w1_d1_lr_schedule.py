"""Offline I′ acceptance. No real dataset/checkpoint/CUDA work is performed."""

import copy
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.eval import w1_d1_lr_schedule as lr
from ireset.eval import w1_d1_lr_workflow as flow
from ireset.eval import w1_d1_lr_artifacts as artifacts
from ireset.eval import w1_smallset_fit_diagnostic as fit

@pytest.fixture(scope="module")
def control(tmp_path_factory):
    threads = torch.get_num_threads()
    out = tmp_path_factory.mktemp("lr_control")
    world, plan, baseline, admission, authorization = flow.synthetic_materials(out)
    yield out, world, plan, baseline, admission, authorization
    torch.set_num_threads(threads)


def driver_at(out, world, plan, baseline, *, two_groups=False):
    torch.set_num_threads(1)
    view = world.view_for("D1")
    budget = fit.StageBudget("D1", 1800.0, 60.0)
    trainer = world.build_trainer(plan, out, budget_deadline=budget.work_deadline, train_ds=view["ds"])
    if two_groups:
        group = trainer.opt.param_groups[0]
        params = group["params"]
        trainer.opt.param_groups = [{**group, "params": params[:1]}, {**group, "params": params[1:]}]
    driver = lr.ScheduleDriver(baseline=baseline, trainer=trainer, plan=plan,
                               registration=view["registration"], loss_fn=world.loss_fn,
                               decoder=world.decoder, probe_batch=view["probe_batch"],
                               budget=budget, runtime_guard=fit.SyntheticRuntimeGuard(),
                               identity={"fixture": True}, stage_dir=out,
                               dataset_window_keys=view["ds"].window_keys)
    return driver


def test_real_boundary_and_legacy_constant_contract():
    p = lr.SchedulePlan("D1")
    assert [p.lr_at(u) for u in (0, 500, 501, 502, 1000)] == [1e-4, 1e-4, 5e-5, 5e-5, 5e-5]
    assert fit.FitPlan("D1").as_dict().get("schedule_id") is None
    for changes in ({"stage": "D6"}, {"switch_after": 499}, {"multiplier": 0.25}, {"model_type": "diffusion"}):
        with pytest.raises(fit.FitDiagnosticError):
            replace(p, **changes)
    with pytest.raises(TypeError):
        fit.FitPlan("D1", schedule_id=lr.SCHEDULE_ID)


def test_frozen_matrix_is_d1_only():
    matrix = flow.frozen_matrix()
    assert matrix["bounds"] == {"D6_all_calls": 0, "conditioner_forward_calls": 4103,
                                 "backbone_forward_calls": 4103, "production_fk_batch_calls": 8104}
    assert matrix["D1"]["loss_production_decode_calls"] == 8000
    assert matrix["D1"]["prefix_file_comparisons"] == 53
    assert matrix["F"]["checkpoint_inspection_reads"] == 4
    assert flow.CAPS == {"R": [120., 15.], "D1": [1800., 60.], "F": [600., 60.]}


@pytest.fixture(scope="module")
def candidate(control):
    out, world, plan, baseline, admission, authorization = control
    run = out / "B"
    run.mkdir()
    t0 = time.perf_counter()
    result = flow.run_candidate(run_root=run, admission_path=admission,
                                authorization_path=authorization, chain_t0=t0,
                                outer_launch={"launch_id": "fixture", "t0_perf_counter": t0},
                                fixture=True, fixture_world=world)
    return run, result, plan, baseline


def test_native_training_prefix_persisted_and_f_bound(candidate):
    run, result, plan, baseline = candidate
    doc = lr.read_object(Path(result["summary"]))
    assert doc["problems"] == [], doc
    assert doc["d1"]["local_complete_evidence"] is True, lr.read_object(run / "d1/stage_report.json")
    assert doc["f"]["complete_evidence"] is True, lr.read_object(run / "f/f_report.json")
    assert doc["f"]["replay_passed"] is True
    assert doc["d6_ran"] is False and not (run / "d6").exists()
    assert lr.verify_schedule_stage(run / "d1", plan, baseline, fit.StageBudget("F", 600, 60)) == []
    rows = lr.read_object(run / "d1/probe_index.json")["history_unrounded"]
    assert [r["effective_lrs"] for r in rows] == [[.002], [.002], [.001], [.001]]
    f = lr.read_object(run / "f/f_report.json")
    assert f["comparison"]["sha256"] == lr.file_record(run / "f/comparison.json")["sha256"]
    assert doc["comparison"] == f["comparison"]
    assert len(lr.read_object(run / "f/comparison.json")["paired_probes"]) == plan.n_probes
    assert rows[0]["state_step_before"] == 0 and rows[-1]["state_step_after"] == plan.steps


@pytest.mark.parametrize("mutation", ["early", "late", "one_group", "micro", "reset", "data_reset", "new_optimizer"])
def test_actual_optimizer_mutations_refused(tmp_path, control, mutation):
    _, world, plan, baseline, *_ = control
    driver = driver_at(tmp_path / mutation, world, plan, baseline, two_groups=mutation == "one_group")
    original = driver._before_update

    def before(u):
        original(u)
        if mutation == "early" and u == plan.switch_after:
            driver.trainer.opt.param_groups[0]["lr"] *= .5
        if mutation in ("late", "one_group") and u == plan.switch_after + 1:
            driver.trainer.opt.param_groups[-1]["lr"] = plan.lr
        if mutation == "reset" and u == plan.switch_after + 1:
            for entry in driver.trainer.opt.state.values():
                entry["exp_avg"].zero_()
        if mutation == "data_reset" and u == plan.switch_after + 1:
            for entry in driver.trainer.opt.state.values():
                entry["exp_avg"].data.zero_()
        if mutation == "new_optimizer" and u == plan.switch_after + 1:
            driver.trainer.opt = torch.optim.AdamW(
                [*driver.trainer.model.parameters(), *driver.trainer.conditioner.parameters()],
                lr=driver.current_lr, weight_decay=plan.weight_decay)
    driver._before_update = before
    if mutation == "micro":
        method = driver.trainer.accumulate_micro
        def micro(*args, **kw):
            result = method(*args, **kw)
            driver.trainer.opt.param_groups[0]["lr"] *= .5
            return result
        driver.trainer.accumulate_micro = micro
    result = driver.run()
    assert result["local_complete_evidence"] is False
    assert result["counters"]["adamw_step_calls"] < plan.steps
    if mutation in ("late", "one_group", "reset", "data_reset", "new_optimizer"):
        assert result["counters"]["adamw_step_calls"] == plan.switch_after


@pytest.mark.parametrize("scope", ["instance", "global"])
@pytest.mark.parametrize("when", ["pre", "post"])
def test_late_optimizer_step_hooks_refused(tmp_path, control, scope, when):
    _, world, plan, baseline, *_ = control
    driver = driver_at(tmp_path / "B", world, plan, baseline)
    original = driver._optimizer_step_entry
    handles, called = [], []

    def alter_lr(opt, *_):
        called.append(True)
        opt.param_groups[0]["lr"] = plan.lr

    def bind(u):
        original(u)
        if u == plan.switch_after + 1:
            register = (getattr(driver.trainer.opt, f"register_step_{when}_hook") if scope == "instance"
                        else getattr(lr.torch_optimizer, f"register_optimizer_step_{when}_hook"))
            handles.append(register(alter_lr))

    driver._optimizer_step_entry = bind
    try:
        report = driver.run()
    finally:
        for handle in handles:
            handle.remove()
    assert handles and called == []
    assert report["local_complete_evidence"] is False
    assert report["failure"]["message"] == "lr_optimizer_step_hooks_forbidden"
    assert report["counters"]["adamw_step_calls"] == plan.switch_after
    assert len((driver.stage_dir / "lr_history.jsonl").read_text().splitlines()) == plan.switch_after
    assert not (driver.stage_dir / "last.pt").exists()


def test_prefix_receipt_write_cannot_admit_late_state_read(tmp_path, control, monkeypatch):
    _, world, plan, baseline, *_ = control
    driver = driver_at(tmp_path / "B", world, plan, baseline)
    clock = fit._FakeClock(0.)
    driver.budget.t0, driver.budget.clock = 0., clock
    write, digest = lr.write_json, driver._read_state_digest
    late_reads = []

    def slow(path, doc):
        write(path, doc)
        if path.name == "prefix_admission.json":
            clock.t = driver.budget.work_deadline

    def state():
        if clock.t >= driver.budget.work_deadline:
            late_reads.append(clock.t)
        return digest()

    monkeypatch.setattr(lr, "write_json", slow)
    monkeypatch.setattr(driver, "_read_state_digest", state)
    report = driver.run()
    assert (driver.stage_dir / "prefix_admission.json").is_file()
    assert late_reads == [] and report["budget"]["stopped_at"] == "prefix_admission_written"
    assert report["counters"]["loss_calls"] == plan.switch_after * plan.grad_accum
    assert report["counters"]["adamw_step_calls"] == plan.switch_after


@pytest.mark.parametrize("stopping_file", ["teacher_reference.npz", lr.probe_name(0, 0), "initial.pt"])
def test_artifact_cap_refuses_before_write_and_prevents_f(tmp_path, control, monkeypatch, stopping_file):
    _, world, _, baseline, admission, authorization = control
    seq = ("teacher_reference.npz", "probe_inputs.npz", lr.probe_name(0, 0), "initial.pt")
    preceding = seq[:seq.index(stopping_file)]
    work_bytes = (sum(baseline["artifact_files"][rel]["bytes"] for rel in preceding) + 8192
                  if preceding else 1500)
    cap = artifacts.CLOSURE_RESERVE_BYTES + work_bytes
    monkeypatch.setattr(flow.radm, "R_ARTIFACT_CAP_BYTES", cap)

    def forbidden(*_a, **_kw):
        pytest.fail("F entered fresh replay after artifact admission failed")

    monkeypatch.setattr(fit, "_fresh_probe", forbidden)
    run = tmp_path / "B"
    result = flow.run_candidate(run_root=run, admission_path=admission, authorization_path=authorization,
                                chain_t0=time.perf_counter(), outer_launch={}, fixture=True, fixture_world=world)
    d = lr.read_object(run / "d1/stage_report.json")
    assert not d["local_complete_evidence"] and d["counters"]["adamw_step_calls"] == 0
    assert d["counters"].get("checkpoint_saves", 0) == 0
    assert "lr_artifact_cap_exceeded" in d["failure"]["message"] and stopping_file in d["failure"]["message"]
    assert not any((run / "d1" / rel).exists() for rel in (stopping_file, stopping_file + ".tmp"))
    assert all((run / "d1" / rel).is_file() for rel in preceding)
    assert result["summary_doc"]["f"]["complete_evidence"] is False
    cmp = lr.read_object(run / "f/comparison.json")
    assert cmp["conclusion"] == "not_comparable" and cmp["metrics"] is None
    assert artifacts.ArtifactSpace(run, cap_bytes=cap).used_bytes() <= cap


@pytest.mark.parametrize("kind", ["byte", "missing_probe", "history", "fake_lr", "state_counter", "endpoint"])
def test_disk_evidence_refuses_tamper(tmp_path, candidate, kind):
    run, _, plan, baseline = candidate
    import shutil
    stage = tmp_path / "d1"
    shutil.copytree(run / "d1", stage)
    if kind == "byte":
        path = stage / lr.probe_name(0, 0)
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    elif kind == "missing_probe":
        (stage / lr.probe_name(2, 2)).unlink()
    elif kind == "endpoint":
        import numpy as np
        path = stage / lr.probe_name(plan.n_probes - 1, plan.steps)
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["w0__raw_x0_pred"][0, 0] += .01
        np.savez(path, **arrays)
        # The extension binds only the prefix; the normal local/F verifier
        # is responsible for endpoints. Re-signing does not bypass decode.
        fit._full_resign(stage)
        verification = fit.verify_stage_local(stage, plan, fit.FitRegistration(
            "D1", (fit.WindowKey("synth_00", 0, 60),), ("train",)),
            fit.make_synthetic_world(n_windows=1).decoder)
        assert verification["problems"]
        return
    else:
        pidx = lr.read_object(stage / "probe_index.json")
        rows = pidx["history_unrounded"]
        if kind == "history":
            rows.pop(0)
        elif kind == "fake_lr":
            rows[-1]["effective_lrs"] = [plan.lr]
        else:
            rows[-1]["state_step_before"] = 0
        lr.write_json(stage / "probe_index.json", pidx)
        (stage / "lr_history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert lr.verify_schedule_stage(stage, plan, baseline, fit.StageBudget("F", 600, 60))


def test_prefix_drift_stops_before_first_changed_micro(tmp_path, control):
    _, world, plan, baseline, *_ = control
    driver = driver_at(tmp_path / "B", world, plan, baseline)
    original = driver._check_prefix_file
    def check(rel):
        if rel == lr.probe_name(2, 2):
            path = driver.stage_dir / rel
            path.write_bytes(path.read_bytes() + b"x")
        original(rel)
    driver._check_prefix_file = check
    result = driver.run()
    assert result["failure"]["message"].startswith("baseline_prefix_mismatch")
    assert driver.counters["adamw_step_calls"] == plan.switch_after
    assert driver.counters["loss_calls"] == plan.switch_after * plan.grad_accum
    assert (driver.stage_dir / "lr_history.jsonl").is_file()


@pytest.mark.parametrize("b,errors,expected", [
    ([.005, .01, .02, .008], [], "single_window_pass"),
    ([.005, .021, .02, .015], [], "improved_but_insufficient"),
    ([.005, .023, .02, .015], [], "no_prespecified_improvement"),
    ([.005, .021, .02, .015], ["replay_mismatch"], "not_comparable"),
    ([.02, .021, .02, .015], [], "no_prespecified_improvement"),
])
def test_four_exclusive_decisions(b, errors, expected):
    a = dict(zip(lr.GATES, [.002787, .023781, .027639, .016762]))
    result = lr.classify(a, dict(zip(lr.GATES, b)), errors)
    assert result["conclusion"] == expected
    if errors:
        assert result["metrics"] is None and result["fit_verdict"] is None


@pytest.mark.parametrize("change", ["legacy", "d6", "old_sha", "duplicate", "bool_budget", "unsigned"])
def test_new_authorization_strict_and_bound(control, change):
    _, _, _, _, admission_path, auth_path = control
    admission = lr.read_object(admission_path)
    auth = lr.read_object(auth_path)
    assert flow.authorization_problems(auth, admission, admission_path) == []
    if change == "legacy": auth["schema"] = flow.radm.AUTHORIZATION_SCHEMA
    if change == "d6": auth["includes"] = ["D1", "D6", "F"]
    if change == "duplicate": auth["includes"] = ["D1", "F", "F"]
    if change == "old_sha": auth["baseline_sha256"] = "0" * 64
    if change == "bool_budget": auth["budget_caps"]["D1"][0] = True
    if change == "unsigned": auth["authorized"] = False
    assert flow.authorization_problems(auth, admission, admission_path)


def test_once_only_and_unknown_history(tmp_path):
    flow.require_unused_root(tmp_path, fixture=True)
    (tmp_path / "unsealed.tmp").write_text("unknown")
    with pytest.raises(fit.FitDiagnosticError, match="history_unknown"):
        flow.require_unused_root(tmp_path, fixture=True)


def test_deadline_does_not_launch_late_update(tmp_path, control):
    _, world, plan, baseline, *_ = control
    driver = driver_at(tmp_path / "late", world, plan, baseline)
    clock = fit._FakeClock(0.)
    driver.budget.clock = clock
    driver.budget.t0 = 0.
    original = driver._before_update
    def before(u):
        original(u)
        clock.t = driver.budget.work_deadline + 1
    driver._before_update = before
    result = driver.run()
    assert result["local_complete_evidence"] is False
    assert result["counters"]["loss_calls"] == 0


def test_supervisor_normal_fixture_chain(tmp_path):
    output = tmp_path / "supervised"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/run_w1_d1_lr_schedule_supervisor.py"),
                           "--synthetic", "--output-root", str(output)],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    logs = (output / "supervisor_stderr.log").read_text() if (output / "supervisor_stderr.log").exists() else ""
    worker = output / "attempt_001/worker_stderr.log"
    assert proc.returncode == 0, (proc.stdout, proc.stderr, logs, worker.read_text() if worker.exists() else "")
    completion = lr.read_object(output / "completion_verification.json")
    assert completion["supervisor_return_code"] == 0 and completion["comparison_candidate"] is True
    assert lr.consume_comparison(output / "attempt_001", collector_return_code=proc.returncode)["metrics"] is not None
    assert lr.consume_comparison(output / "attempt_001", observer_return_code=0)["metrics"] is None
    assert lr.consume_comparison(output / "attempt_001")["conclusion"] == "not_comparable"
    run = output / "attempt_001"
    admission = lr.read_object(run / "synthetic_materials/fixture_admission.json")
    baseline_dir = Path(admission["package"]["baseline"]["root"]) / "d1"
    for path, problem in (
        (run / "d1" / lr.probe_name(0, 0), "lr_artifact_missing_or_invalid"),
        (baseline_dir / "teacher_reference.npz", "baseline_unreadable"),
        (run / "d1/lr_history.jsonl", "lr_artifact_missing_or_invalid"),
        (run / "f/return_observation.json", "return_observation.json"),
        (output / lr.protocol.CHAIN_WAIT_NAME, lr.protocol.CHAIN_WAIT_NAME),
        (output / lr.protocol.CHAIN_ARCHIVE_NAME, lr.protocol.CHAIN_ARCHIVE_NAME),
    ):
        # No records or observed exit values are rewritten between the two calls.
        saved = path.with_suffix(".held")
        path.rename(saved)
        try:
            refused = lr.consume_comparison(run, collector_return_code=proc.returncode)
            assert refused["conclusion"] == "not_comparable" and refused["metrics"] is None, path
            assert any(problem in p for p in refused["problems"]), refused
        finally:
            saved.rename(path)
    assert lr.consume_comparison(run, collector_return_code=proc.returncode)["metrics"] is not None
    wait_path = output / lr.protocol.CHAIN_WAIT_NAME
    archive_path = output / lr.protocol.CHAIN_ARCHIVE_NAME
    wait_raw, archive_raw = wait_path.read_bytes(), archive_path.read_bytes()
    for tamper in ("wrong_launch", "late_wait", "late_archive"):
        wait, archive = json.loads(wait_raw), json.loads(archive_raw)
        if tamper == "wrong_launch":
            wait["launch_id"] += "-other"
        elif tamper == "late_wait":
            wait["observer_return_perf_counter"] = wait["t0_perf_counter"] + wait["registered_measured_s"] + 1
            wait["elapsed_s"] = wait["observer_return_perf_counter"] - wait["t0_perf_counter"]
        else:
            archive["archive_synced_perf_counter"] = wait["t0_perf_counter"] + wait["registered_measured_s"] + 1
            archive["elapsed_s"] = archive["archive_synced_perf_counter"] - wait["t0_perf_counter"]
        try:
            lr.write_json(wait_path, wait)
            archive["wait_sha256"] = lr.protocol.sha(wait_path)
            lr.write_json(archive_path, archive)
            refused = lr.consume_comparison(run, collector_return_code=0)
            assert refused["conclusion"] == "not_comparable" and refused["metrics"] is None
            assert "lr_chain_actual_return_or_archive_not_covered" in refused["problems"]
        finally:
            wait_path.write_bytes(wait_raw)
            archive_path.write_bytes(archive_raw)
    comparison_path = run / "f/comparison.json"
    modified = lr.read_object(comparison_path)
    modified["conclusion"] = "single_window_pass"
    lr.write_json(comparison_path, modified)
    assert lr.consume_comparison(run, collector_return_code=proc.returncode)["metrics"] is None
    assert completion["elapsed_after_actual_wait_s"] <= completion["registered_measured_s"] <= 2400
    p2 = subprocess.run([sys.executable, str(ROOT / "scripts/run_w1_d1_lr_schedule_supervisor.py"),
                         "--synthetic", "--output-root", str(output)], cwd=ROOT,
                        capture_output=True, text=True, timeout=15)
    assert p2.returncode != 0 and "once-only" in p2.stderr


def test_supervisor_budget_binding_refuses():
    spec = importlib.util.spec_from_file_location("lr_sup_test", ROOT / "scripts/run_w1_d1_lr_schedule_supervisor.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    core = module.load_core()
    assert core.CHAIN_CAP == 2400 and "d6" not in core.STAGE_CAPS


def test_old_chain_budget_cannot_be_rebound(tmp_path, candidate):
    run, *_ = candidate
    import shutil
    shutil.copy(run / "acceptance_summary.json", tmp_path / "acceptance_summary.json")
    # Pure protocol fixture: run_candidate returns before the worker's stdout
    # coverage step, so it deliberately has no top-level return proof yet.
    seal = lr.read_object(tmp_path / "acceptance_summary.json")["seal"]
    lr.write_json(tmp_path / "return_observation.json", {
        "schema": fit.RETURN_OBSERVATION_SCHEMA, "covered": True,
        "summary_sha256": flow.radm.sha256_file(tmp_path / "acceptance_summary.json"),
        "elapsed_s": min(1., seal["measured_wall_s_at_summary"]),
        "registered_measured_s": seal["measured_wall_s_at_summary"], "wall_cap_s": 2400.})
    spec = importlib.util.spec_from_file_location("lr_budget_test", ROOT / "scripts/run_w1_d1_lr_schedule_supervisor.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    core = module.load_core()
    path = tmp_path / "acceptance_summary.json"
    problems, doc, proof, _ = core._proof_problems(path, tmp_path / "return_observation.json", None, 2400.)
    assert problems == []
    doc.pop("_output_dir", None)
    doc["budget"]["wall_cap_s"] = 6120.
    lr.write_json(path, doc)
    proof["summary_sha256"] = core._sha(path)
    lr.write_json(tmp_path / "return_observation.json", proof)
    assert "budget_binding_mismatch" in core._proof_problems(
        path, tmp_path / "return_observation.json", None, 2400.)[0]


def test_mutation_after_prefix_receipt_stops_at_501_boundary(tmp_path, control):
    _, world, plan, baseline, *_ = control
    driver = driver_at(tmp_path / "B", world, plan, baseline)
    original = driver._before_update
    def before(u):
        if u == plan.switch_after + 1:
            path = driver.stage_dir / lr.probe_name(0, 0)
            path.write_bytes(path.read_bytes() + b"late mutation")
        original(u)
    driver._before_update = before
    result = driver.run()
    assert result["counters"]["adamw_step_calls"] == plan.switch_after
    assert "baseline_prefix_mismatch:admission" in result["failure"]["message"]


def test_recorded_lr_drift_stops_before_first_changed_micro(tmp_path, control):
    _, world, plan, baseline, *_ = control
    driver = driver_at(tmp_path / "B", world, plan, baseline)
    original = driver._before_update

    def before(u):
        if u == plan.switch_after + 1:
            path = driver.stage_dir / "lr_history.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1]["effective_lrs"] = [plan.lr_at(u)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        original(u)

    driver._before_update = before
    result = driver.run()
    assert result["failure"]["message"] == "baseline_prefix_mismatch:durable_history"
    assert result["counters"]["loss_calls"] == plan.switch_after * plan.grad_accum
    assert result["counters"]["adamw_step_calls"] == plan.switch_after


def test_stage_close_does_not_rewrite_a_changed_prefix(tmp_path, control):
    _, world, plan, baseline, *_ = control
    driver = driver_at(tmp_path / "B", world, plan, baseline)
    original = driver._write_stage_artifacts
    path = driver.stage_dir / lr.probe_name(0, 0)

    def before_close():
        path.write_bytes(path.read_bytes() + b"drift")
        original()

    driver._write_stage_artifacts = before_close
    result = driver.run()
    assert result["local_complete_evidence"] is False
    assert "baseline_prefix_mismatch:file:" in result["failure"]["message"]
    assert path.read_bytes().endswith(b"drift")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 10**400, "0.1", None])
def test_bad_comparison_numbers_never_escape(value):
    a = dict(zip(lr.GATES, [.002, .024, .028, .017]))
    b = {**a, "root_rotation_mean_rad": value}
    assert lr.classify(a, b, [])["conclusion"] == "not_comparable"


def test_r_extension_metadata_only_and_versioned_recheck(tmp_path, control, monkeypatch):
    spec = importlib.util.spec_from_file_location("lr_source_fixture", ROOT / "tests/test_w1_smallset_fit_r_admission.py")
    fixture_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture_module)
    fx = fixture_module._fixture_project(tmp_path)
    project = fx["proj"]
    (project / "docs/plans").mkdir(parents=True)
    import shutil
    shutil.copy(ROOT / lr.PLAN_PATH, project / lr.PLAN_PATH)
    _, _, plan, baseline, *_ = control
    def forbidden(*_a, **_kw):
        raise AssertionError("metadata R attempted numeric work")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(fit, "Trainer", forbidden)
    reader = fit.MetadataOnlyReader()
    result = flow.run_admission(
        out_dir=tmp_path / "lr_admission", budget=fit.StageBudget("R", 120, 15),
        fixture=True, baseline_root=Path(baseline["root"]), plan=plan,
        project_root=project, source_root=fx["source_root"], teacher_npz_dir=fx["npz_dir"],
        omomo_pt_dir=fx["pt_dir"], reader=reader, runtime_probe=fixture_module._runtime_probe,
        disk_probe=fixture_module._disk_probe(), git_identity=fixture_module._stub_git)
    doc = result["summary_doc"]
    assert doc["admission_passed"] is True, doc["problems"]
    assert doc["package"]["schema"] == flow.PACKAGE_SCHEMA
    assert doc["package"]["call_matrix"]["D6"] == {}
    assert doc["zero_declarations"]["array_value_reads"] == 0
    assert doc["zero_declarations"]["checkpoint_reads"] == 0
    assert doc["package"]["authorization_template"]["authorized"] is False
    args = dict(project_root=project, source_root=fx["source_root"],
                teacher_npz_dir=fx["npz_dir"], omomo_pt_dir=fx["pt_dir"],
                runtime_probe=fixture_module._runtime_probe, disk_probe=fixture_module._disk_probe())
    assert flow.recheck(doc, **args) == []
    changed = copy.deepcopy(doc)
    changed["package"]["call_matrix"]["F"]["backbone_forward_calls"] = 0
    assert "lr_admission_drift:call_matrix" in flow.recheck(changed, **args)


def test_f_endpoint_mismatch_is_not_fit_failure(tmp_path, candidate, control, monkeypatch):
    run, _, plan, baseline = candidate
    _, world, *_ = control
    original = fit._fresh_probe
    def bad_fresh(*args, **kwargs):
        raw, prod = original(*args, **kwargs)
        raw[0, 0, 0] += .001
        return raw, prod
    monkeypatch.setattr(fit, "_fresh_probe", bad_fresh)
    budget = fit.StageBudget("F", 600, 60)
    identity = lr.read_object(run / "d1/stage_report.json")["identity"]
    out = fit.run_f_verification(stage_specs=[{
        "stage_dir": run / "d1", "plan": plan, "registration": world.registration_for("D1"),
        "decoder": world.decoder, "loss_fn": world.loss_fn, "identity": identity,
        "verify_extension": lambda b: lr.verify_schedule_stage(run / "d1", plan, baseline, b)}],
        out_dir=tmp_path / "f", budget=budget, runtime_guard=fit.SyntheticRuntimeGuard(),
        build_modules=fit.build_modules_factory(world, plan), authorization={"includes": ["D1", "F"]})
    final = out["summary_doc"]
    assert final["complete_evidence"] is False and final["verdict"] == "incomplete"
    comparison = lr.build_comparison(stage_dir=run / "d1", baseline=baseline, plan=plan,
                                      f_candidate=final, budget=budget)
    assert comparison["conclusion"] == "not_comparable" and comparison["metrics"] is None


def test_lr_evidence_failure_prevents_f_construction(tmp_path, candidate, control):
    run, _, plan, baseline = candidate
    _, world, *_ = control
    calls = []
    out = fit.run_f_verification(stage_specs=[{
        "stage_dir": run / "d1", "plan": plan, "registration": world.registration_for("D1"),
        "decoder": world.decoder, "loss_fn": world.loss_fn,
        "identity": lr.read_object(run / "d1/stage_report.json")["identity"],
        "verify_extension": lambda b: ["lr_sequence_mismatch:501"]}],
        out_dir=tmp_path / "f", budget=fit.StageBudget("F", 600, 60),
        runtime_guard=fit.SyntheticRuntimeGuard(), build_modules=lambda: calls.append("constructed"),
        authorization={"includes": ["D1", "F"]})
    assert calls == [] and out["summary_doc"]["complete_evidence"] is False


def test_f_comparison_deadline_withdraws_claim(tmp_path):
    clock = fit._FakeClock(0.)
    budget = fit.StageBudget("F", 100, 10, t0=0., clock=clock)
    def slow(final):
        clock.t = 91.
    result = fit.run_f_verification(stage_specs=[], out_dir=tmp_path / "f", budget=budget,
                                    runtime_guard=fit.SyntheticRuntimeGuard(), build_modules=lambda: None,
                                    authorization=None, finalize_evidence=slow)
    assert result["summary_doc"]["complete_evidence"] is False
    assert "f_extension_evidence_complete" in result["summary_doc"]["extension_error"]


def test_registered_source_materialization_to_f(tmp_path, monkeypatch):
    """Exercise the non-injected orchestration on temporary synthetic files."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        spec = importlib.util.spec_from_file_location("lr_materialized_fixture", ROOT / "tests/test_w1_smallset_fit_r_admission.py")
        fxmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(fxmod)
        fx = fxmod._fixture_project(tmp_path)
        project = fx["proj"]
        (project / "docs/plans").mkdir(parents=True)
        import shutil
        shutil.copy(ROOT / lr.PLAN_PATH, project / lr.PLAN_PATH)
        monkeypatch.setattr(flow.radm, "runtime_identity_facts", fxmod._runtime_probe)
        old_r = fxmod._run_r(tmp_path, fx)
        assert old_r["summary_doc"]["admission_passed"]
        kw = dict(project_root=project, source_root=fx["source_root"],
                  teacher_npz_dir=fx["npz_dir"], omomo_pt_dir=fx["pt_dir"])
        billing = fit.StageBudget("D1", 1800, 60)
        world = flow.radm.materialize_real_world(admission=old_r["summary_doc"],
                                                 stage="D1", budget=billing,
                                                 fixture=True, device="cpu", **kw)
        plan = lr.SchedulePlan("D1", steps=4, probe_every=1, switch_after=2,
                                d_model=16, n_heads=4, n_layers=1, ffn_mult=2,
                                lr=.002, synthetic=True)
        view = world.view_for("D1")
        a_dir = tmp_path / "A/d1"
        trainer = world.build_trainer(plan.constant_plan(), a_dir,
                                      budget_deadline=billing.work_deadline, train_ds=view["ds"])
        driver = fit.FitDriver(trainer=trainer, plan=plan.constant_plan(), registration=view["registration"],
                               loss_fn=world.loss_fn, decoder=world.decoder, probe_batch=view["probe_batch"],
                               budget=billing, runtime_guard=fit.SyntheticRuntimeGuard(), identity={"fixture": True},
                               stage_dir=a_dir, dataset_window_keys=view["ds"].window_keys)
        summary = driver.run()
        assert summary["local_complete_evidence"], summary
        fit.finish_stage(a_dir, summary, billing)
        r_out = flow.run_admission(out_dir=tmp_path / "new_R", budget=fit.StageBudget("R", 120, 15),
                                   fixture=True, baseline_root=a_dir.parent, plan=plan,
                                   runtime_probe=fxmod._runtime_probe, disk_probe=fxmod._disk_probe(),
                                   git_identity=fxmod._stub_git, **kw)
        doc = r_out["summary_doc"]
        assert doc["admission_passed"], doc["problems"]
        auth = flow.authorization_draft(doc["package"], flow.radm.sha256_file(Path(r_out["summary"])))
        auth.update(authorized=True, template_only=False, checklist={k: True for k in flow.CHECKLIST})
        auth_path = tmp_path / "authorized_fixture.json"
        lr.write_json(auth_path, auth)
        run = tmp_path / "B"
        run.mkdir()
        result = flow.run_candidate(run_root=run, admission_path=Path(r_out["summary"]),
                                    authorization_path=auth_path, chain_t0=time.perf_counter(),
                                    outer_launch={}, fixture=True, **kw)
        facts = lr.read_object(Path(result["summary"]))
        assert facts["problems"] == [], facts
        assert facts["f"]["complete_evidence"] is True, facts
        assert lr.read_object(run / "d1/stage_report.json")["construction_counts"] == {
            "model_constructions": 1, "conditioner_constructions": 1, "optimizer_constructions": 1}
    finally:
        torch.set_num_threads(threads)
