"""Execution accounting and configuration boundary tests for W1-D/E."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import omniretarget_w1_de as drv
from test_omniretarget_w1_de import _de_world, _fh


def test_history_bad_json_and_duplicate_index_are_unknown(tmp_path):
    run = tmp_path / "run"
    root = run / "formal/deterministic"
    (root / "attempt_001").mkdir(parents=True)
    (root / "attempts_index.json").write_text(json.dumps({"attempts": [
        {"attempt": "attempt_001"}, {"attempt": "attempt_001"}]}))
    (run / "costs.json").write_text(json.dumps({"invocations": [{
        "stage": drv.STAGES["train-deterministic"], "attempt": "attempt_001",
        "status": "done", "wall_s": 0.0}]}))
    assert drv.domain_history_cost_s(run, "d") is None
    (root / "attempts_index.json").write_text(json.dumps({"attempts": [
        {"attempt": "attempt_001"}]}))
    (run / "costs.json").write_text("{")
    assert drv.domain_history_cost_s(run, "d") is None
    (run / "costs.json").write_text("{}")
    assert drv.domain_history_cost_s(run, "d") is None


def test_new_root_keeps_failed_attempt_cost(tmp_path, monkeypatch):
    world, run = _de_world(tmp_path, monkeypatch)
    args = SimpleNamespace(source_run_root=world["run"], run_root=run, plan="de_plan.md")
    assert drv.stage_register(args) == 0
    stage = drv.STAGES["train-deterministic"]
    att = drv._begin_attempt(run, stage)
    t0 = time.perf_counter() - 31.0
    drv.finish_attempt(run, stage, att, t0,
        {"remaining_s": 100.0}, {"complete": False}, {}, "crashed")
    second = run.parent / "another_de_root"
    drv._claim_execution_root(second, world["root"], world["run"], "de_plan.md")
    assert drv.domain_history_cost_s(second, "d") >= 31.0
    row = json.loads((run / "costs.json").read_text())["invocations"][-1]
    assert row["status"] == "crashed"
    assert row["wall_s"] == round(row["measured_wall_s"], 1)


def test_running_ledger_cannot_be_consumed_after_sigkill(tmp_path, monkeypatch):
    world, run = _de_world(tmp_path, monkeypatch)
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=run, plan="de_plan.md")) == 0
    stage = drv.STAGES["train-deterministic"]
    att = drv._begin_attempt(run, stage)
    drv._replace_index_row(att.parent, att.name, {"status": "done", "complete": True})
    row, problems = drv.stage_state(run, stage, world["root"])
    assert row is not None
    assert "attempt_running" in problems
    assert "index_ledger_completion_contradiction" in problems
    assert drv.domain_history_cost_s(run, "d") is None


def test_scope_deletion_and_rounded_cost_tamper_are_unknown(tmp_path, monkeypatch):
    world, run = _de_world(tmp_path, monkeypatch)
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=run, plan="de_plan.md")) == 0
    stage = drv.STAGES["train-deterministic"]
    att = drv._begin_attempt(run, stage)
    drv.finish_attempt(run, stage, att, time.perf_counter(),
                       {"remaining_s": 100.0}, {"complete": False}, {})
    path = run / "costs.json"
    doc = json.loads(path.read_text())
    doc["invocations"][-1].pop("scope")
    path.write_text(json.dumps(doc))
    assert drv.domain_history_cost_s(run, "d") is None


def test_scope_deleted_from_shared_history_cannot_reset_cost(tmp_path, monkeypatch):
    world, run = _de_world(tmp_path, monkeypatch)
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=run, plan="de_plan.md")) == 0
    history = drv._execution_history_path(world["root"])
    history.write_text('{"runs": []}')
    assert drv.domain_history_cost_s(run, "r0r1") is None
    assert drv.domain_history_cost_s(run.parent / "fresh_root", "d") is None


def test_same_attempt_final_write_delay_revokes_completion(tmp_path, monkeypatch):
    world, run = _de_world(tmp_path, monkeypatch)
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=run, plan="de_plan.md")) == 0
    stage = drv.STAGES["raw-eval"]
    att = drv._begin_attempt(run, stage)
    report = att / "raw_eval.json"
    drv.atomic_json(report, {"completeness": {"complete": True}})
    original = drv._upsert_cost
    delayed = {"done": False}

    def slow(*args, **kwargs):
        if not delayed["done"]:
            delayed["done"] = True
            time.sleep(.03)
        return original(*args, **kwargs)

    monkeypatch.setattr(drv, "_upsert_cost", slow)
    assert not drv.finish_attempt(run, stage, att, time.perf_counter(),
        {"remaining_s": .01}, {"complete": True, "report": str(report)}, {})
    rows = json.loads((att.parent / "attempts_index.json").read_text())["attempts"]
    assert len(rows) == 1 and rows[0]["complete"] is False
    assert rows[0]["incomplete_reason"] == "budget_exceeded_after_registration"
    assert json.loads(report.read_text())["completeness"]["complete"] is False
    costs = json.loads((run / "costs.json").read_text())["invocations"]
    assert len([c for c in costs if c["stage"] == stage]) == 1
    assert costs[-1]["measured_wall_s"] >= .03


def test_best_full_roster_wrong_names_and_nan_macro_rejected():
    hist = [_fh(1000, .4, demos=("a", "foreign"))]
    assert drv.recompute_best_selection(hist, expected_demos=["a", "b"]) == (None, None)
    hist = [_fh(1000, .4, recorded=float("nan"))]
    assert drv.recompute_best_selection(hist, expected_demos=["a", "b"]) == (None, None)
    hist = [{**_fh(1000, .4), "seed": 17}]
    assert drv.recompute_best_selection(hist, expected_demos=["a", "b"], seed=7) == (None, None)
    hist = [_fh(1000, .4)]
    hist[0]["per_demo"][0].pop("numerical_valid")
    assert drv.recompute_best_selection(hist, expected_demos=["a", "b"]) == (None, None)
    hist = [_fh(1000, .4)]
    hist[0]["per_demo"][0]["T"] = True
    assert drv.recompute_best_selection(hist, expected_demos=["a", "b"],
                                       demo_lengths={"a": 1, "b": 8}) == (None, None)


def test_formal_unknown_nested_key_rejected(tmp_path):
    from train_generation import load_train_config

    path = tmp_path / "bad.yaml"
    path.write_text("formal_validation_contract:\n  full_generation_sead: 7\n")
    with pytest.raises(SystemExit, match="unknown formal-validation"):
        load_train_config(path)


def test_spawn_timeout_is_capped_for_small_remaining_budget(tmp_path):
    log = tmp_path / "subprocess.log"
    start = time.perf_counter()
    code, outcome, elapsed = drv._spawn(
        [sys.executable, "-c", "import time; time.sleep(5)"], tmp_path, .03, log)
    assert code is None and outcome == "timeout"
    assert elapsed < .75 and time.perf_counter() - start < .75


def test_exclusive_json_retains_existing_bytes(tmp_path):
    path = tmp_path / "registration.json"
    drv.atomic_json(path, {"version": 1}, exclusive=True)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        drv.atomic_json(path, {"version": 2}, exclusive=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("patch,problem", [
    ({"status": "crashed"}, "index_ledger_status_mismatch"),
    ({"complete": False}, "index_ledger_completion_mismatch"),
    ({"scope": {}}, "attempt_scope_mismatch"),
    ({"artifacts": ["bad"]}, "attempt_artifacts_not_a_mapping"),
])
def test_terminal_attempt_contradictions_are_rejected(tmp_path, monkeypatch, patch, problem):
    world, run = _de_world(tmp_path, monkeypatch)
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=run, plan="de_plan.md")) == 0
    stage = drv.STAGES["raw-eval"]
    att = drv._begin_attempt(run, stage)
    assert drv.finish_attempt(run, stage, att, time.perf_counter(),
                              {"remaining_s": 100.0}, {"complete": True}, {})
    drv._replace_index_row(att.parent, att.name, patch)
    row, problems = drv.stage_state(run, stage, world["root"])
    assert row is not None and problem in problems
    if "artifacts" not in patch:
        assert drv.domain_history_cost_s(run, "e") is None


def test_revoked_preflight_cannot_consume_written_registration(tmp_path, monkeypatch):
    world, run = _de_world(tmp_path, monkeypatch)
    assert drv.stage_register(SimpleNamespace(
        source_run_root=world["run"], run_root=run, plan="de_plan.md")) == 0
    assert drv.verify_execution_registration(run, world["root"]) == []
    stage = drv.STAGES["register"]
    row, problems = drv.stage_state(run, stage, world["root"])
    assert not problems
    att = run / "preflight" / row["attempt"]
    assert not drv.finish_attempt(run, stage, att, time.perf_counter() - 2,
                                  {"remaining_s": 1.0}, row, {})
    assert (run / "freeze/execution_registration.json").is_file()
    assert "preflight_not_complete" in drv.verify_execution_registration(run, world["root"])


def test_spawn_timeout_kills_descendant_group(tmp_path):
    import os

    pidfile = tmp_path / "child.pid"
    child = "import time; time.sleep(30)"
    parent = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)"
    )
    code, outcome, elapsed = drv._spawn(
        [sys.executable, "-c", parent, str(pidfile)], tmp_path, .4, tmp_path / "group.log")
    assert code is None and outcome == "timeout" and elapsed < 2.0
    assert pidfile.is_file()
    pid = int(pidfile.read_text())
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        assert stat.read_text().split()[2] == "Z"
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
