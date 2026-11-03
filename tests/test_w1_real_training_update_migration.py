"""Binding migration tests for the W1 real training-update audit.

The fixture constructs the same two-row v1 prefix as the real cohort (R0 done,
P incomplete, two already-billed updates) without running the old chain.  The
migration then executes the second-generation R0/P/I/F under the synthetic
provider, so no real checkpoint, dataset, asset or model is read.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import ireset.eval.w1_real_training_update_audit as A
from ireset.paths import PROJECT_ROOT


TINY_CONFIG = dict(A.MODEL_CONFIG)
TINY_CONFIG.update(d_model=8, n_layers=1, n_heads=2, ffn_mult=2,
                   time_window=3)
OLD_R0_COST = 58.59990523397573
OLD_P_COST = 8.502827989985235
OLD_MARKER_AUTH = "a" * 64


def _balance(counts: dict[str, int]) -> dict:
    return {"schema": "w1_real_training_update_call_budget_v1",
            "counts": {kind: int(counts[kind]) for kind in A.CALL_KINDS},
            "within_caps": all(int(counts[kind]) <= int(A.CALL_CAPS[kind])
                              for kind in A.CALL_KINDS)}


def _zero_counts() -> dict[str, int]:
    return {kind: 0 for kind in A.CALL_KINDS}


def _write_legacy_prefix(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    A.write_cohort(root, mode="synthetic")
    marker = A.read_json(root / "cohort.json")
    marker["authorization_sha256"] = OLD_MARKER_AUTH
    A.write_json(root / "cohort.json", marker)
    r0_delta = {kind: 0 for kind in A.CALL_KINDS}
    r0_delta["cpu_draw_units"] = 8
    # The real P attempt failed after two native updates; it did not consume
    # the full four-update P nominal.  Keeping that partial prefix is the
    # whole point of the migration test.
    p_delta = {kind: 0 for kind in A.CALL_KINDS}
    p_delta.update({
        "checkpoint_loads": 2, "model_constructions": 2,
        "conditioner_constructions": 2,
        "native_optimizer_constructions": 2,
        "conditioner_forwards": 8, "backbone_forwards": 8,
        "native_add_noise_batches": 0, "cpu_draw_units": 8,
        "production_loss_assembler": 8,
        "production_float32_fk_batch": 16,
        "production_parameter_backward": 8,
        "native_global_clip": 2, "native_adamw_steps": 2,
        "diagnostic_parameter_update_events": 2,
        "micro_execution_units": 8,
    })
    r0_after = dict(r0_delta)
    p_after = {kind: r0_delta[kind] + int(p_delta[kind])
               for kind in A.CALL_KINDS}
    rows = []
    ledger = []
    for stage, delta, before, after, cost, complete in (
            ("R0", r0_delta, _zero_counts(), r0_after, OLD_R0_COST, True),
            ("P", p_delta, r0_after, p_after, OLD_P_COST, False)):
        attempt = "attempt_001"
        directory = A.stage_dir(root, stage) / attempt
        directory.mkdir(parents=True, exist_ok=True)
        reservation = {
            "schema": "w1_real_training_update_reservation_v1",
            "reserved": {kind: int(delta[kind]) for kind in A.CALL_KINDS},
            "consumed": {kind: int(delta[kind]) for kind in A.CALL_KINDS},
        }
        A.write_json(directory / "call_reservations.json", reservation)
        status = "done" if complete else "incomplete"
        row = {
            "stage": stage, "attempt": attempt,
            "accounting_schema": A.ACCOUNTING_SCHEMA,
            "cohort_id": A.COHORT_ID,
            "status": status, "complete": bool(complete),
            "measured_wall_s": float(cost),
            "call_balance": _balance(after),
            "call_balance_before": _balance(before),
            "call_balance_after": _balance(after),
        }
        rows.append(row)
        ledger.append({**row, "wall_s": round(float(cost), 1)})
    A.save_index(root, rows)
    A.save_ledger(root, ledger)


def _install_migration(root: Path, provider: A.SyntheticTrainingProvider,
                       base: Path, invocation: str = "inv-migration-001"
                       ) -> tuple[Path, str, Path, dict, dict, float]:
    snapshot_path, snapshot_sha, snapshot = A.write_history_snapshot(
        base / "migration", root)
    spec = A.build_binding_spec(
        PROJECT_ROOT, snapshot_path, snapshot_sha, snapshot,
        invocation_id=invocation,
        gate_b={"sha256": "3" * 64, "tolerance_evidence_sha256": "d" * 64,
                "fp32_contract_sha256": "e" * 64,
                "acceptance_sha256": "c" * 64,
                "calibration_verification_sha256": "b" * 64},
        admission={"sha256": "4" * 64})
    spec_path = base / "migration" / "binding_spec.json"
    A.write_json_exclusive(spec_path, spec)
    spec_sha = A.sha256_file(spec_path)
    A.stage_binding(root, spec_path=spec_path, spec_sha256=spec_sha,
                    snapshot_sha256=snapshot_sha,
                    invocation_id=invocation)
    binding, problems = A.load_binding(root, require_active=False)
    assert binding is not None and problems == []
    t0 = A._now()
    A.claim_launch(
        root, authorization_sha256=OLD_MARKER_AUTH, spec_sha256=spec_sha,
        legacy_head_sha256=binding.legacy_head_sha256,
        marker_sha256=snapshot["marker_sha256"], invocation_id=invocation,
        t0_perf_counter=t0,
        target={"stage": "R0", "attempt": "attempt_002",
                "execution_seq": 3})
    impl = A.implementation_identity(PROJECT_ROOT)
    auth = A.migration_authorization_template()
    auth.update(template_only=False, authorized=True)
    auth["checklist"] = {key: True for key in A.AUTHORIZATION_REQUIRED}
    auth["draft_sha256"] = A.sha256_file(PROJECT_ROOT / A.PREREG_DRAFT)
    auth["execution_plan_sha256"] = A.sha256_file(
        PROJECT_ROOT / A.MIGRATION_PLAN)
    auth["calibration_verification_sha256"] = "b" * 64
    auth["acceptance_sha256"] = "c" * 64
    auth["tolerance_evidence_sha256"] = "d" * 64
    auth["fp32_contract_sha256"] = "e" * 64
    auth["admission_package_sha256"] = "4" * 64
    auth["implementation_files"] = impl["files"]
    migration = auth["migration"]
    migration.update(
        run_root_realpath=str(root),
        source_marker_sha256=snapshot["marker_sha256"],
        source_authorization_sha256=snapshot["authorization_sha256"],
        snapshot_sha256=snapshot_sha, spec_sha256=spec_sha,
        migration_admission_sha256="2" * 64,
        legacy_head_sha256=binding.legacy_head_sha256,
        target_generation=A.GENERATION_002,
        plan_sha256=A.sha256_file(PROJECT_ROOT / A.MIGRATION_PLAN),
        implementation_sha256=A.canonical_sha256(impl),
        runtime_sha256=A.canonical_sha256(A.runtime_identity_facts()),
        gate_b_sha256="3" * 64, real_admission_sha256="4" * 64,
        execution_plan_sha256=A.sha256_file(PROJECT_ROOT / A.MIGRATION_PLAN),
        prereg_sha256=A.sha256_file(PROJECT_ROOT / A.PREREG_DRAFT),
        stage_plan=list(A.MIGRATION_STAGE_PLAN),
        prior_updates=2, incremental_update_limit=16,
        cohort_update_limit_for_this_authorization=18,
        prior_cost_s=A.MIGRATION_SOURCE_COST_S,
        remaining_s=A.TOTAL_BUDGET_S - A.MIGRATION_SOURCE_COST_S,
        invocation_id=invocation,
        launch_journal={
            "relative_path": A.LAUNCH_JOURNAL_DIR,
            "manifest_sha256": snapshot["launch_journal"][
                "manifest_sha256"]},
        write_after_protocol={
            "sample": "post_publish_perf_counter - stage_t0"},
        return_observation_required=True, unique_entry=True,
        execution_window={"ends_at": "outer_actual_process_return"})
    assert A.authorization_problems(auth) == []
    return spec_path, spec_sha, snapshot_path, auth, binding, t0


@pytest.fixture(scope="module")
def migrated(tmp_path_factory):
    base = tmp_path_factory.mktemp("w1_migration")
    root = base / "cohort"
    _write_legacy_prefix(root)
    fixture_dir = base / "fixture"
    provider = A.SyntheticTrainingProvider(
        TINY_CONFIG, seed=A.FIXTURE_SEED, device="cuda")
    provider.prepare_fixture(fixture_dir)
    provider.load_fixture()
    assert provider.fixture["problems"] == []
    spec_path, spec_sha, snapshot_path, auth, binding, t0 = \
        _install_migration(root, provider, base)
    result = A.run_binding_migration_audit(
        root, provider, mode="synthetic",
        tolerance=A.build_default_tolerance_contract(),
        authorization_sha=OLD_MARKER_AUTH, authorization=auth,
        binding_spec_path=spec_path, invocation_id="inv-migration-001",
        fixed_root=A.HISTORY_ROOT, r0_started_perf=t0)
    return {"base": base, "root": root, "provider": provider,
            "result": result, "binding": binding, "t0": t0,
            "snapshot_path": snapshot_path, "spec_path": spec_path,
            "auth": auth}


def test_legacy_head_matches_the_frozen_real_prefix():
    root = PROJECT_ROOT / A.MIGRATION_SOURCE_RUN
    if not root.is_dir():
        pytest.skip("original real cohort is not present")
    payload = A.build_legacy_head_payload(root)
    assert A.legacy_head_sha256(payload) == \
        A.MIGRATION_LEGACY_HEAD_SHA256
    assert payload["last_execution_seq"] == 2
    assert [r["stage"] for r in payload["records"]] == ["R0", "P"]
    assert payload["call_balance_after"][
        "diagnostic_parameter_update_events"] == 2


def test_sequence_starts_after_the_frozen_prefix_and_matches_plan(migrated):
    result = migrated["result"]
    assert result["complete_evidence"] is True, result["problems"]
    assert result["selected_attempts"] == {
        "R0": "attempt_002", "P": "attempt_002",
        "I": "attempt_001", "F": "attempt_001"}
    assert result["selected_chain_training_updates"] == 16
    assert result["cohort_training_updates"] == 18
    assert result["training_updates"] == 18
    assert result["call_accounting_known"] is True
    rows = {(r["stage"], r["attempt"]): r
            for r in A.load_index(migrated["root"])}
    assert rows[("R0", "attempt_001")].get("execution_seq") is None
    assert rows[("P", "attempt_001")].get("execution_seq") is None
    for seq, stage, attempt in ((3, "R0", "attempt_002"),
                                (4, "P", "attempt_002"),
                                (5, "I", "attempt_001"),
                                (6, "F", "attempt_001")):
        row = rows[(stage, attempt)]
        assert row["accounting_schema"] == A.ACCOUNTING_SCHEMA_V2
        assert row["generation"] == A.GENERATION_002
        assert row["execution_seq"] == seq
        assert row["predecessor_kind"] == (
            "legacy_head" if seq == 3 else "attempt_commit")


def test_public_selection_requires_the_outer_return_observation(migrated):
    root = migrated["root"]
    _row, problems = A.stage_state(
        root, "P", generation=A.GENERATION_002)
    assert any("return_observation_missing" in p for p in problems)
    result = migrated["result"]
    path, sha, problems = A.record_return_observation(
        root, invocation_id="inv-migration-001", pid=os.getpid(),
        ppid=os.getppid(), exit_code=3, signal=None,
        started_perf_counter=migrated["t0"], finished_perf_counter=A._now(),
        last_stage_end_perf_counter=result["last_stage_end_perf_counter"],
        commits=result["commits"], candidate_receipts=result["candidates"],
        write_after=result["write_after"],
        coverage_ok=result["complete_evidence"])
    assert problems == []
    for stage in A.STAGE_BUDGETS:
        row, problems = A.stage_state(
            root, stage, generation=A.GENERATION_002)
        assert row is not None and problems == [], (stage, problems)
    old_rows = [r for r in A.load_index(root)
                if r["stage"] == "R0" and r["attempt"] == "attempt_001"]
    assert len(old_rows) == 1 and old_rows[0]["status"] == "done"
    _row, problems = A.stage_state(root, "R0")
    assert "generation_ambiguous" in problems


def test_duplicate_or_gapped_sequence_is_refused(migrated, tmp_path):
    import shutil

    source = migrated["root"]
    root = tmp_path / "tampered"
    shutil.copytree(source, root)
    rows = A.load_index(root)
    for row in rows:
        if row["stage"] == "P" and row["attempt"] == "attempt_002":
            row["execution_seq"] = 5
    A.save_index(root, rows)
    ledger = A.load_ledger(root)
    for entry in ledger:
        if entry["stage"] == "P" and entry["attempt"] == "attempt_002":
            entry["execution_seq"] = 5
    A.save_ledger(root, ledger)
    from dataclasses import replace

    binding = migrated["binding"]
    snapshot = {**binding.snapshot, "run_root": str(root),
                "run_root_realpath": str(root)}
    copied = replace(binding, snapshot=snapshot)
    records, _binding, problems = A._execution_records(
        root, binding=copied, require_commits=False)
    assert records is None
    assert problems


def _fake_clock(monkeypatch, now: float):
    import types
    monkeypatch.setattr(A, "time", types.SimpleNamespace(
        perf_counter=lambda: now,
        strftime=lambda fmt: "2026-11-03T00:00:00+0800",
        time=lambda: now,
        time_ns=lambda: int(now * 1e9)))


def test_stage_closed_candidate_actual_write_gate_and_revoke(tmp_path,
                                                             monkeypatch):
    root = tmp_path / "cohort"
    root.mkdir()
    commit_path = root / "execution_events/000003/commit.json"
    commit_path.parent.mkdir(parents=True)
    A.write_json(commit_path, {"kind": "commit"})
    commit = A.CommitResult(
        True, None, None, commit_path, A.sha256_file(commit_path), [], 0.0)
    _fake_clock(monkeypatch, now=5.0)
    path, problems = A.write_launch_stage_closed_candidate(
        root, invocation_id="inv-slow", stage="R0", execution_seq=3,
        commit=commit, event_candidate=None, event_candidate_sha256=None,
        t0=0.0, remaining_s=10.0, committed_measured_wall_s=1.0)
    assert path is not None
    assert "launch_stage_closed_write_after_gate" in problems
    revokes = list((root / "launch_journal/inv-slow").glob("revoke_*.json"))
    assert revokes
    revoke = A.strict_read_json(revokes[0])
    assert revoke["uncovered_wall_s_lower_bound"] == 4.0


def test_terminal_candidate_actual_write_gate_and_revoke(tmp_path,
                                                         monkeypatch):
    root = tmp_path / "cohort"
    root.mkdir()
    commit_path = root / "execution_events/000006/commit.json"
    commit_path.parent.mkdir(parents=True)
    A.write_json(commit_path, {"kind": "commit"})
    commit = A.CommitResult(
        True, None, None, commit_path, A.sha256_file(commit_path), [], 0.0)
    _fake_clock(monkeypatch, now=5.0)
    path, problems = A.write_launch_terminal_candidate(
        root, invocation_id="inv-slow", commit=commit,
        event_candidate=None, event_candidate_sha256=None,
        t0=0.0, remaining_s=10.0, committed_measured_wall_s=1.0)
    assert path is not None
    assert "launch_terminal_write_after_gate" in problems
    assert list((root / "launch_journal/inv-slow").glob("revoke_*.json"))


def test_return_observation_requires_actual_return_facts(tmp_path):
    commits = {}
    receipts = {}
    for stage in A.STAGE_BUDGETS:
        commit = tmp_path / f"{stage}_commit.json"
        receipt = tmp_path / f"{stage}_receipt.json"
        A.write_json(commit, {"stage": stage})
        A.write_json(receipt, {"stage": stage})
        commits[stage] = {"path": str(commit), "sha256": A.sha256_file(commit)}
        receipts[stage] = {"path": str(receipt),
                           "sha256": A.sha256_file(receipt)}
    base = {
        "schema": A.RETURN_OBSERVATION_SCHEMA,
        "scope": "post_execution_observation",
        "process_returned": True,
        "invocation_id": "inv",
        "pid": 1, "ppid": 0,
        "source": {"kind": "outer_wait", "pid": 1, "ppid": 0,
                   "exit_code": 3, "signal": None},
        "started_perf_counter": 0.0, "finished_perf_counter": 5.0,
        "last_stage_end_perf_counter": 4.0, "total_wall_s": 5.0,
        "write_after": {"R0": True, "P": True, "I": True, "F": True,
                        "terminal": True},
        "commits": commits,
        "candidate_receipts": receipts,
        "coverage_ok": True,
        "post_return_work_performed": False,
    }
    claim = {"authorization_sha256": "a" * 64, "spec_sha256": "b" * 64,
             "legacy_head_sha256": "c" * 64, "_sha256": "d" * 64}
    record = {**base, "authorization_sha256": "a" * 64,
              "spec_sha256": "b" * 64, "legacy_head_sha256": "c" * 64,
              "claim_sha256": "d" * 64}
    assert A.return_observation_problems(record, invocation_id="inv",
                                          claim=claim) == []
    bad = dict(record, coverage_ok=False, process_returned=False)
    problems = A.return_observation_problems(bad, invocation_id="inv",
                                              claim=claim)
    assert "return_coverage" in problems
    assert "return_process_not_returned" in problems
    assert "return_observation_missing" in A.return_observation_problems(None)


def test_unsettled_f_reports_null_total_and_known_lower_bound(migrated,
                                                               tmp_path):
    import copy
    import shutil
    from dataclasses import replace

    root = tmp_path / "unsettled"
    shutil.copytree(migrated["root"], root)
    rows = A.load_index(root)
    ledger = A.load_ledger(root)
    for row in rows:
        if row["stage"] == "F":
            row["status"] = "running"
            row["complete"] = False
            row["measured_wall_s"] = 0.0
            row["call_balance"] = copy.deepcopy(row["call_balance_before"])
    for entry in ledger:
        if entry["stage"] == "F":
            entry["status"] = "running"
            entry["complete"] = False
            entry["measured_wall_s"] = 0.0
            entry["call_balance"] = copy.deepcopy(entry["call_balance_before"])
    A.save_index(root, rows)
    A.save_ledger(root, ledger)
    reservation = A.stage_dir(root, "F") / "attempt_001" / \
        "call_reservations.json"
    data = A.read_json(reservation)
    data["consumed"] = None
    A.write_json(reservation, data)
    binding = migrated["binding"]
    snapshot = copy.deepcopy(binding.snapshot)
    snapshot["run_root"] = str(root)
    snapshot["run_root_realpath"] = str(root)
    payload = snapshot["legacy_head_payload"]
    payload["run_root"] = str(root)
    snapshot["legacy_head_sha256"] = A.legacy_head_sha256(payload)
    copied = replace(binding, snapshot=snapshot)
    deltas, unsettled = A._settled_deltas_with_unsettled_tail(
        root, binding=copied)
    assert unsettled == "F/attempt_001"
    assert deltas is not None
    counts = A.cohort_call_totals_from_deltas(deltas)
    assert int(counts["native_adamw_steps"]
               + counts["explicit_reference_updates"]) == 10


def test_migration_authorization_rejects_budget_and_update_reset():
    template = A.migration_authorization_template()
    assert A.authorization_problems(template)
    auth = json.loads(json.dumps(template))
    auth.update(template_only=False, authorized=True)
    auth["checklist"] = {key: True for key in A.AUTHORIZATION_REQUIRED}
    for key in ("draft_sha256", "execution_plan_sha256",
                "calibration_verification_sha256", "acceptance_sha256",
                "tolerance_evidence_sha256", "fp32_contract_sha256",
                "admission_package_sha256"):
        auth[key] = "0" * 64
    auth["implementation_files"] = {"x": "y"}
    auth["updates"] = {**A.EXPECTED_UPDATES,
                       "prior_updates": 0,
                       "incremental_update_limit": 18,
                       "cohort_update_limit_for_this_authorization": 18}
    auth["migration"] = {key: "0" for key in A.MIGRATION_AUTHORIZATION_KEYS}
    auth["migration"].update(
        run_root_realpath="/tmp/run", target_generation=A.GENERATION_002,
        stage_plan=list(A.MIGRATION_STAGE_PLAN), prior_updates=0,
        incremental_update_limit=18,
        cohort_update_limit_for_this_authorization=18,
        prior_cost_s=0.0, remaining_s=A.TOTAL_BUDGET_S,
        launch_journal={"relative_path": A.LAUNCH_JOURNAL_DIR,
                        "manifest_sha256": "0" * 64},
        write_after_protocol={
            "sample": "post_publish_perf_counter - stage_t0"},
        execution_window={"ends_at": "outer_actual_process_return"},
        return_observation_required=True, unique_entry=True)
    problems = A.authorization_problems(auth)
    assert any("update" in p for p in problems)
    assert any("remaining" in p or "prior_cost" in p for p in problems)
