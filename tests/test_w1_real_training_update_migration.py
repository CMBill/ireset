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
import subprocess
import sys
import time
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


def _load_launch_script(name: str):
    import importlib.util

    script_path = PROJECT_ROOT / "scripts/audit_w1_real_training_update.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


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
        implementation_sha256=A.implementation_files_sha256(impl),
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
    # Once the migration has run, the live index/ledger are append-only and
    # include generation-2 rows.  The frozen v1 prefix must be rebuilt from
    # the snapshot's byte-for-byte source copies, exactly as the migration
    # baseline builder does.  Before any snapshot exists, the live files are
    # still the original prefix and the direct call is valid.
    snapshots = sorted(
        (PROJECT_ROOT / "derived/audit/w1_real_training_update_audit"
         / "binding_migration").glob("attempt_*/history_snapshot.json"))
    if snapshots:
        snapshot_path = snapshots[-1]
        snapshot = A.strict_read_json(snapshot_path)
        source = snapshot_path.parent / "source"
        assert snapshot["legacy_head_sha256"] == \
            A.MIGRATION_LEGACY_HEAD_SHA256
        payload = A.build_legacy_head_payload(
            root,
            index_rows=A.strict_read_json(source / "index.json")["attempts"],
            ledger_rows=A.strict_read_json(
                source / "ledger.json")["invocations"],
            marker=A.strict_read_json(source / "cohort.json"),
            marker_sha256=A.sha256_file(source / "cohort.json"),
            source_index_sha256=A.sha256_file(source / "index.json"),
            source_ledger_sha256=A.sha256_file(source / "ledger.json"))
    else:
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
    root = migrated["root"]
    r0_commit = A.strict_read_json(
        root / "execution_events/000003/commit.json")
    activation_path = (root / A.BINDINGS_DIR / A.GENERATION_002 /
                       "activation.json")
    activation = A.strict_read_json(activation_path)
    assert r0_commit["activation"] == {
        "path": A.path_for_record(activation_path),
        "sha256": A.sha256_file(activation_path)}
    assert activation["r0"]["reservation_sha256"] == \
        r0_commit["reservation_sha256"]


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
        stage_timings=result["stage_timings"],
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
    timings = {}
    for index, stage in enumerate(A.STAGE_BUDGETS):
        commit = tmp_path / f"{stage}_commit.json"
        receipt = tmp_path / f"{stage}_receipt.json"
        t0 = float(index)
        end = float(index + 1)
        A.write_json(commit, {
            "stage": stage,
            "stage_t0_perf_counter": t0,
            "committed_measured_wall_s": 2.5,
        })
        A.write_json(receipt, {"stage": stage})
        commits[stage] = {"path": str(commit), "sha256": A.sha256_file(commit)}
        receipts[stage] = {"path": str(receipt),
                           "sha256": A.sha256_file(receipt)}
        timings[stage] = {"t0_perf_counter": t0, "end_perf_counter": end}
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
        "stage_timings": timings,
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
    # A caller-supplied coverage_ok=True cannot hide an outer return that
    # happened long after the last commit was billed.
    late = dict(record, finished_perf_counter=1_000_000.0,
                total_wall_s=1_000_000.0)
    late_problems = A.return_observation_problems(
        late, invocation_id="inv", claim=claim)
    assert "return_coverage:F" in late_problems
    assert "return_coverage" in late_problems


def test_snapshot_consumption_rehashes_frozen_files(tmp_path):
    root = tmp_path / "cohort"
    _write_legacy_prefix(root)
    snapshot_path, _sha, snapshot = A.write_history_snapshot(
        tmp_path / "migration", root)
    assert A._snapshot_problems(
        snapshot, run_root=root, root=root) == []
    # index/ledger are the only legal append targets.
    rows = A.load_index(root)
    ledger = A.load_ledger(root)
    extra = dict(rows[-1])
    extra.update(stage="P", attempt="attempt_002", status="incomplete",
                 complete=False, execution_seq=3)
    rows.append(extra)
    ledger.append({**extra, "wall_s": round(
        float(extra.get("measured_wall_s", 0.0)), 1)})
    A.save_index(root, rows)
    A.save_ledger(root, ledger)
    assert A._snapshot_problems(
        snapshot, run_root=root, root=root) == []
    marker = A.read_json(root / "cohort.json")
    marker["tampered_after_snapshot"] = True
    A.write_json(root / "cohort.json", marker)
    problems = A._snapshot_problems(snapshot, run_root=root, root=root)
    assert "snapshot_file_sha:cohort.json" in problems
    marker.pop("tampered_after_snapshot")
    A.write_json(root / "cohort.json", marker)
    reservation = A.stage_dir(root, "P") / "attempt_001" / \
        "call_reservations.json"
    data = A.read_json(reservation)
    data["consumed"]["native_adamw_steps"] = 99
    A.write_json(reservation, data)
    problems = A._snapshot_problems(snapshot, run_root=root, root=root)
    assert ("snapshot_file_sha:production/attempt_001/"
            "call_reservations.json") in problems


def test_commit_refresh_reuses_the_priced_prepared_view(migrated, monkeypatch):
    root = migrated["root"]
    row = next(r for r in A.load_index(root)
               if r["stage"] == "F" and r["attempt"] == "attempt_001")
    entry = next(r for r in A.load_ledger(root)
                 if r["stage"] == "F" and r["attempt"] == "attempt_001")
    reservation = A.stage_dir(root, "F") / "attempt_001" / \
        "call_reservations.json"
    prepared = A.PreparedResult(
        row, entry, [], {
            "index_view_sha256": A.canonical_sha256(
                A._frozen_row_view(row)),
            "ledger_view_sha256": A.canonical_sha256(
                A._frozen_row_view(entry)),
            "complete": row.get("complete") is True,
            "status": row.get("status"),
            "reservation_sha256": A.sha256_file(reservation),
            "generation": row.get("generation"),
            "binding_spec_sha256": row.get("binding_spec_sha256"),
            "execution_seq": row.get("execution_seq"),
        })

    def no_full_verify(*_args, **_kwargs):
        raise AssertionError("commit must not repeat full prepared hashing")

    monkeypatch.setattr(A, "_verify_prepared_attempt", no_full_verify)
    refreshed = A._refresh_prepared_view(
        root, "F", "attempt_001", prepared,
        {"generation": row.get("generation"),
         "binding_spec_sha256": row.get("binding_spec_sha256"),
         "execution_seq": row.get("execution_seq")})
    assert refreshed.problems == []
    assert refreshed.facts["index_row_sha256"] == A.canonical_sha256(row)


def test_write_after_proof_write_is_itself_release_checked(tmp_path,
                                                           monkeypatch):
    from types import SimpleNamespace

    root = tmp_path / "cohort"
    root.mkdir()
    commit_path = root / "execution_events/000003/commit.json"
    commit_path.parent.mkdir(parents=True)
    A.write_json(commit_path, {"kind": "commit"})
    commit = A.CommitResult(
        True, None, None, commit_path, A.sha256_file(commit_path), [], 0.0)
    clock = SimpleNamespace(now=0.0)
    original_write = A.write_json_exclusive

    def slow_write(path, data):
        sha = original_write(path, data)
        if str(path).endswith("_write_after.json"):
            clock.now = 100.0
        return sha

    monkeypatch.setattr(A, "time", SimpleNamespace(
        perf_counter=lambda: clock.now,
        strftime=lambda fmt: "2026-11-03T00:00:00+0800",
        time=lambda: clock.now,
        time_ns=lambda: int(clock.now * 1e9)))
    monkeypatch.setattr(A, "write_json_exclusive", slow_write)
    path, problems = A.write_launch_stage_closed_candidate(
        root, invocation_id="inv-proof", stage="R0", execution_seq=3,
        commit=commit, event_candidate=None, event_candidate_sha256=None,
        t0=0.0, remaining_s=200.0, committed_measured_wall_s=10.0)
    assert path is not None
    assert "launch_stage_closed_write_after_proof_gate" in problems
    revokes = list((root / "launch_journal/inv-proof").glob("revoke_*.json"))
    assert revokes
    assert A.strict_read_json(revokes[0])["reason"] == \
        "launch_stage_closed_write_after_proof_gate"
    clock.now = 0.0
    terminal_commit = root / "execution_events/000006/commit.json"
    terminal_commit.parent.mkdir(parents=True)
    A.write_json(terminal_commit, {"kind": "commit"})
    terminal = A.CommitResult(
        True, None, None, terminal_commit,
        A.sha256_file(terminal_commit), [], 0.0)
    _path, problems = A.write_launch_terminal_candidate(
        root, invocation_id="inv-proof", commit=terminal,
        event_candidate=None, event_candidate_sha256=None,
        t0=0.0, remaining_s=200.0, committed_measured_wall_s=10.0)
    assert "launch_terminal_write_after_proof_gate" in problems


def test_f_incomplete_evidence_is_not_committed(tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    _write_legacy_prefix(root)
    fixture_dir = tmp_path / "fixture"
    provider = A.SyntheticTrainingProvider(
        TINY_CONFIG, seed=A.FIXTURE_SEED, device="cuda")
    provider.prepare_fixture(fixture_dir)
    provider.load_fixture()
    spec_path, _spec_sha, _snapshot_path, auth, _binding, t0 = \
        _install_migration(root, provider, tmp_path,
                           invocation="inv-f-veto")
    original = A.run_f_stage

    def forced_incomplete(*args, **kwargs):
        result = original(*args, **kwargs)
        result["complete_evidence"] = False
        result.setdefault("verification_problems", []).append(
            "forced_incomplete")
        return result

    monkeypatch.setattr(A, "run_f_stage", forced_incomplete)
    result = A.run_binding_migration_audit(
        root, provider, mode="synthetic",
        tolerance=A.build_default_tolerance_contract(),
        authorization_sha=OLD_MARKER_AUTH, authorization=auth,
        binding_spec_path=spec_path, invocation_id="inv-f-veto",
        fixed_root=A.HISTORY_ROOT, r0_started_perf=t0)
    assert result["conclusion"] == "audit_incomplete"
    assert result["selected_attempts"] is None
    row = next(r for r in A.load_index(root)
               if r["stage"] == "F" and r["attempt"] == "attempt_001")
    assert row["status"] == "incomplete" and row["complete"] is False
    assert not A._commit_path(root, 6).exists()


def test_scanner_keeps_structural_problems_when_observation_is_valid(tmp_path):
    root = tmp_path / "cohort"
    _write_legacy_prefix(root)
    commits = {}
    receipts = {}
    timings = {}
    for index, stage in enumerate(A.STAGE_BUDGETS):
        commit = tmp_path / f"scan_{stage}_commit.json"
        receipt = tmp_path / f"scan_{stage}_receipt.json"
        t0 = float(index)
        A.write_json(commit, {
            "stage": stage, "stage_t0_perf_counter": t0,
            "committed_measured_wall_s": 2.5})
        A.write_json(receipt, {"stage": stage})
        commits[stage] = {"path": str(commit),
                          "sha256": A.sha256_file(commit)}
        receipts[stage] = {"path": str(receipt),
                           "sha256": A.sha256_file(receipt)}
        timings[stage] = {"t0_perf_counter": t0,
                          "end_perf_counter": float(index + 1)}
    A.claim_launch(
        root, authorization_sha256=OLD_MARKER_AUTH,
        spec_sha256="b" * 64, legacy_head_sha256="c" * 64,
        marker_sha256=A.sha256_file(root / "cohort.json"),
        invocation_id="inv-structure", t0_perf_counter=0.0,
        target={"stage": "R0", "attempt": "attempt_002",
                "execution_seq": 3})
    observation = {
        "schema": A.RETURN_OBSERVATION_SCHEMA,
        "scope": "post_execution_observation",
        "process_returned": True,
        "invocation_id": "inv-structure",
        "authorization_sha256": OLD_MARKER_AUTH,
        "spec_sha256": "b" * 64,
        "legacy_head_sha256": "c" * 64,
        "claim_sha256": A.sha256_file(
            A._launch_invocation_dir(root, "inv-structure") / "claim.json"),
        "pid": 1, "ppid": 0,
        "source": {"kind": "outer_wait", "pid": 1, "ppid": 0,
                   "exit_code": 3, "signal": None},
        "started_perf_counter": 0.0, "finished_perf_counter": 5.0,
        "last_stage_end_perf_counter": 4.0, "total_wall_s": 5.0,
        "commits": commits, "candidate_receipts": receipts,
        "stage_timings": timings,
        "write_after": {"R0": True, "P": True, "I": True, "F": True,
                        "terminal": True},
        "coverage_ok": True,
        "expected_exit_code": 3, "conclusion": "audit_incomplete",
        "post_return_work_performed": False,
    }
    A.write_json(
        A._launch_invocation_dir(root, "inv-structure") /
        "return_observation.json", observation)
    journal = A.scan_launch_journals(tmp_path)
    record = next(r for r in journal["records"]
                  if r["invocation_id"] == "inv-structure")
    assert "registered_missing" in record["return_problems"]
    assert "terminal_candidate_missing" in record["return_problems"]
    assert journal["known"] is False


def test_claim_only_root_is_public_history_debt(tmp_path):
    root = tmp_path / "cohort"
    _write_legacy_prefix(root)
    A.claim_launch(
        root, authorization_sha256=OLD_MARKER_AUTH,
        spec_sha256="b" * 64, legacy_head_sha256="c" * 64,
        marker_sha256=A.sha256_file(root / "cohort.json"),
        invocation_id="inv-claim-only", t0_perf_counter=A._now(),
        target={"stage": "R0", "attempt": "attempt_002",
                "execution_seq": 3})
    problems = A.verify_history(root)
    assert any(p.startswith("history_journal:")
               and "inv-claim-only" in p for p in problems)
    allowed = A.verify_history(root, allow_invocation="inv-claim-only")
    assert not any(p.startswith("history_journal:") for p in allowed)


def test_acceptance_v2_exposes_the_ordinary_leaf_record():
    from ireset.eval.w1_real_training_update_provider import (
        SYNTHETIC_ACCEPTANCE_SCHEMA_V2,
        acceptance_leaf_record,
    )
    leaf = {
        "schema": A.AUDIT_SCHEMA,
        "roster": {"parameter_roster": {"n_total": 1}},
        "tolerance": {"evidence_sha256": "e" * 64},
        "implementation": {"files": {"x": "y"}},
    }
    wrapper = {
        "schema": SYNTHETIC_ACCEPTANCE_SCHEMA_V2,
        "ordinary": {"record": leaf},
        "migration": {"record": {"schema": A.AUDIT_SCHEMA}},
    }
    assert acceptance_leaf_record(wrapper) is leaf
    assert acceptance_leaf_record(leaf) is leaf
    assert acceptance_leaf_record(None) == {}


def test_authorization_draft_reads_external_binding_material(tmp_path):
    root = tmp_path / "cohort"
    _write_legacy_prefix(root)
    snapshot_path, snapshot_sha, snapshot = A.write_history_snapshot(
        tmp_path / "migration", root)
    spec = A.build_binding_spec(
        PROJECT_ROOT, snapshot_path, snapshot_sha, snapshot,
        invocation_id="inv-external",
        gate_b={"sha256": "3" * 64}, admission={"sha256": "4" * 64})
    spec_path = tmp_path / "migration" / "binding_spec.json"
    A.write_json_exclusive(spec_path, spec)
    spec_sha = A.sha256_file(spec_path)
    context, problems = A.load_external_binding_material(
        snapshot_path, snapshot_sha, spec_path, spec_sha,
        project_root=PROJECT_ROOT,
        require_current_implementation=True)
    assert context is not None and problems == []
    assert not (root / A.BINDINGS_DIR).exists()
    marker = A.read_json(root / "cohort.json")
    marker["tampered_after_snapshot"] = True
    A.write_json(root / "cohort.json", marker)
    _context, problems = A.load_external_binding_material(
        snapshot_path, snapshot_sha, spec_path, spec_sha,
        project_root=PROJECT_ROOT,
        require_current_implementation=True)
    assert "snapshot_file_sha:cohort.json" in problems


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


def test_executor_stage_intervals_are_contiguous(migrated):
    """The next stage must start at the previous stage's recorded end."""
    timings = migrated["result"]["stage_timings"]
    stages = tuple(A.STAGE_BUDGETS)
    assert set(timings) == set(stages)
    for previous, current in zip(stages, stages[1:]):
        assert timings[current]["t0_perf_counter"] == \
            timings[previous]["end_perf_counter"]


def _write_valid_journal(root: Path, invocation: str = "inv-valid") -> Path:
    """Build a fully closed launch journal (scan-only, no execution rows)."""
    root.mkdir(parents=True, exist_ok=True)
    A.write_cohort(root, mode="synthetic")
    directory = A._launch_invocation_dir(root, invocation)
    directory.mkdir(parents=True)
    A.claim_launch(
        root, authorization_sha256=OLD_MARKER_AUTH, spec_sha256="b" * 64,
        legacy_head_sha256="c" * 64,
        marker_sha256=A.sha256_file(root / "cohort.json"),
        invocation_id=invocation, t0_perf_counter=0.0,
        target={"stage": "R0", "attempt": "attempt_002",
                "execution_seq": 3})
    claim = A.strict_read_json(directory / "claim.json")
    commits: dict[str, dict] = {}
    receipts: dict[str, dict] = {}
    timings: dict[str, dict] = {}
    for index, stage in enumerate(A.STAGE_BUDGETS):
        seq = index + 3
        event_dir = root / "execution_events" / f"{seq:06d}"
        event_dir.mkdir(parents=True, exist_ok=True)
        commit_path = event_dir / "commit.json"
        commit_payload = {
            "schema": A.EVENT_SCHEMA, "kind": "commit",
            "execution_seq": seq, "stage": stage,
            "attempt": A.MIGRATION_STAGE_PLAN[index]["attempt"],
            "generation": A.GENERATION_002,
            "binding_spec_sha256": claim["spec_sha256"],
            "invocation_id": invocation,
            "stage_t0_perf_counter": float(index),
            "committed_measured_wall_s": 2.5}
        if stage == "F":
            # The terminal proof must bind the commit's frozen cost and the
            # F stage's registered remaining budget, not its own claims.
            ledger_row = {
                "stage": stage,
                "attempt": A.MIGRATION_STAGE_PLAN[index]["attempt"],
                "budget": {"remaining_s": 100.0}}
            finish_path = event_dir / "finish.json"
            A.write_json(finish_path, {
                "schema": A.EVENT_SCHEMA, "kind": "finish",
                "execution_seq": seq, "stage": stage,
                "attempt": A.MIGRATION_STAGE_PLAN[index]["attempt"],
                "generation": A.GENERATION_002,
                "binding_spec_sha256": claim["spec_sha256"],
                "invocation_id": invocation,
                "committed_measured_wall_s": 2.5,
                "ledger_row_snapshot": ledger_row})
            commit_payload["finish"] = {
                "path": str(finish_path),
                "sha256": A.sha256_file(finish_path)}
            commit_payload["ledger_row_sha256"] = A.canonical_sha256(
                ledger_row)
        A.write_json(commit_path, commit_payload)
        commits[stage] = {"path": str(commit_path),
                          "sha256": A.sha256_file(commit_path)}
        receipt_path = directory / f"stage_closed_{seq:06d}.json"
        A.write_json(receipt_path, {
            "schema": A.EVENT_SCHEMA, "kind": "stage_closed_candidate",
            "candidate": True, "invocation_id": invocation,
            "stage": stage, "execution_seq": seq,
            "commit": commits[stage]})
        A.write_json(directory / f"stage_closed_{seq:06d}_write_after.json", {
            "schema": A.EVENT_SCHEMA, "kind": "stage_closed_write_after",
            "candidate_sha256": A.sha256_file(receipt_path),
            "observed_elapsed_s": 1.0, "committed_measured_wall_s": 2.5,
            "remaining_s": 100.0, "passed": True})
        receipts[stage] = {"path": str(receipt_path),
                           "sha256": A.sha256_file(receipt_path)}
        timings[stage] = {"t0_perf_counter": float(index),
                          "end_perf_counter": float(index + 1)}
    start_path = root / "execution_events" / "000003" / "start.json"
    A.write_json(start_path, {"schema": A.EVENT_SCHEMA, "kind": "start"})
    A.write_json(directory / "registered.json", {
        "schema": A.EVENT_SCHEMA, "kind": "registered",
        "invocation_id": invocation, "execution_seq": 3,
        "stage": "R0", "attempt": "attempt_002",
        "start": {"path": str(start_path),
                  "sha256": A.sha256_file(start_path)}})
    final_commit = commits[tuple(A.STAGE_BUDGETS)[-1]]
    event_terminal_path = root / "execution_events" / "000006" / \
        "terminal_candidate.json"
    A.write_json(event_terminal_path, {
        "schema": A.EVENT_SCHEMA, "kind": "terminal_candidate",
        "execution_seq": 6, "generation": A.GENERATION_002,
        "binding_spec_sha256": claim["spec_sha256"],
        "invocation_id": invocation, "commit": final_commit,
        "committed_measured_wall_s": 2.5})
    A.write_json(event_terminal_path.parent / "terminal_write_after.json", {
        "schema": A.EVENT_SCHEMA, "kind": "terminal_write_after",
        "candidate_sha256": A.sha256_file(event_terminal_path),
        "observed_elapsed_s": 1.0, "committed_measured_wall_s": 2.5,
        "remaining_s": 100.0, "passed": True})
    launch_terminal_path = directory / "terminal_candidate.json"
    A.write_json(launch_terminal_path, {
        "schema": A.EVENT_SCHEMA, "kind": "terminal_candidate",
        "candidate": True, "invocation_id": invocation,
        "commit": final_commit,
        "event_candidate": {"path": str(event_terminal_path),
                            "sha256": A.sha256_file(event_terminal_path)}})
    A.write_json(directory / "terminal_write_after.json", {
        "schema": A.EVENT_SCHEMA, "kind": "terminal_write_after",
        "candidate_sha256": A.sha256_file(launch_terminal_path),
        "observed_elapsed_s": 1.0, "committed_measured_wall_s": 2.5,
        "remaining_s": 100.0, "passed": True})
    A.write_json(directory / "return_observation.json", {
        "schema": A.RETURN_OBSERVATION_SCHEMA,
        "scope": "post_execution_observation",
        "process_returned": True,
        "invocation_id": invocation,
        "authorization_sha256": claim["authorization_sha256"],
        "spec_sha256": claim["spec_sha256"],
        "legacy_head_sha256": claim["legacy_head_sha256"],
        "claim_sha256": A.sha256_file(directory / "claim.json"),
        "pid": claim["pid"], "ppid": claim["ppid"],
        "source": {"kind": "outer_wait", "pid": claim["pid"],
                   "ppid": claim["ppid"], "exit_code": 0, "signal": None},
        "started_perf_counter": 0.0, "finished_perf_counter": 5.0,
        "last_stage_end_perf_counter": 4.0, "total_wall_s": 5.0,
        "commits": commits, "candidate_receipts": receipts,
        "stage_timings": timings,
        "write_after": {"R0": True, "P": True, "I": True, "F": True,
                        "terminal": True},
        "coverage_ok": True,
        "expected_exit_code": 0,
        "conclusion": "real_training_update_covered_contract_passed",
        "post_return_work_performed": False})
    return directory


def test_terminal_closure_parses_candidate_event_and_proofs(tmp_path):
    root = tmp_path / "cohort"
    directory = _write_valid_journal(root)
    invocation = directory.name
    journal = A.scan_launch_journals(tmp_path)
    assert journal["known"] is True, journal["problems"]
    assert A.verify_history(root) == []

    saved = (directory / "terminal_candidate.json").read_bytes()
    A.write_json(directory / "terminal_candidate.json", {})
    journal = A.scan_launch_journals(tmp_path)
    assert journal["known"] is False
    record = next(r for r in journal["records"]
                  if r["invocation_id"] == invocation)
    assert "terminal_candidate_binding" in record["return_problems"]
    assert "terminal_event_candidate_missing" in record["return_problems"]
    assert any(p.startswith("launch_unsettled:")
               for p in journal["problems"])
    assert any("history_journal:" in p for p in A.verify_history(root))
    (directory / "terminal_candidate.json").write_bytes(saved)

    (root / "execution_events/000006/terminal_write_after.json").unlink()
    journal = A.scan_launch_journals(tmp_path)
    assert journal["known"] is False
    record = next(r for r in journal["records"]
                  if r["invocation_id"] == invocation)
    assert "terminal_event_write_after_missing" in \
        record["return_problems"]

    (root / "execution_events/000006/terminal_candidate.json").unlink()
    (directory / "terminal_write_after.json").unlink()
    journal = A.scan_launch_journals(tmp_path)
    assert journal["known"] is False
    record = next(r for r in journal["records"]
                  if r["invocation_id"] == invocation)
    assert "terminal_event_candidate" in record["return_problems"]
    assert "terminal_write_after_missing" in record["return_problems"]


def test_return_coverage_rejects_handoff_gap_and_total_wall_drift(tmp_path):
    root = tmp_path / "cohort"
    directory = _write_valid_journal(root)
    record = A.strict_read_json(directory / "return_observation.json")
    claim = A.strict_read_json(directory / "claim.json")
    claim["_sha256"] = A.sha256_file(directory / "claim.json")
    assert A.return_observation_problems(
        record, invocation_id=directory.name, claim=claim) == []

    drifted = dict(record, total_wall_s=record["total_wall_s"] + 7.0)
    problems = A.return_observation_problems(
        drifted, invocation_id=directory.name, claim=claim)
    assert "return_total_wall" in problems

    shifted = dict(record, started_perf_counter=2.0)
    problems = A.return_observation_problems(
        shifted, invocation_id=directory.name, claim=claim)
    assert "return_started" in problems

    gapped = json.loads(json.dumps(record))
    for stage in ("P", "I", "F"):
        gapped["stage_timings"][stage]["t0_perf_counter"] += 20.0
        gapped["stage_timings"][stage]["end_perf_counter"] += 20.0
        # Keep the commit facts consistent so the gap is the only defect.
        commit_path = Path(gapped["commits"][stage]["path"])
        commit = A.strict_read_json(commit_path)
        commit["stage_t0_perf_counter"] += 20.0
        A.write_json(commit_path, commit)
        gapped["commits"][stage]["sha256"] = A.sha256_file(commit_path)
    gapped["last_stage_end_perf_counter"] += 20.0
    gapped["finished_perf_counter"] += 20.0
    gapped["total_wall_s"] += 20.0
    problems = A.return_observation_problems(
        gapped, invocation_id=directory.name, claim=claim)
    assert "return_stage_gap:R0:P" in problems
    assert not any(p.startswith("return_stage_t0") for p in problems)


def test_commit_view_allows_cost_reseal_and_checks_wall_projection(
        migrated, tmp_path):
    import shutil

    root = tmp_path / "cohort"
    shutil.copytree(migrated["root"], root)
    stage, attempt = "F", "attempt_001"
    row = next(r for r in A.load_index(root)
               if r["stage"] == stage and r["attempt"] == attempt)
    entry = next(r for r in A.load_ledger(root)
                 if r["stage"] == stage and r["attempt"] == attempt)
    reservation = A.stage_dir(root, stage) / attempt / \
        "call_reservations.json"
    measured = float(entry["measured_wall_s"])
    prepared = A.PreparedResult(
        row, entry, [], {
            "index_view_sha256": A.canonical_sha256(
                A._frozen_row_view(row)),
            "ledger_view_sha256": A.canonical_sha256(
                A._frozen_row_view(entry)),
            "complete": row.get("complete") is True,
            "status": row.get("status"),
            "measured_wall_s": measured,
            "reservation_sha256": A.sha256_file(reservation),
            "generation": row.get("generation"),
            "binding_spec_sha256": row.get("binding_spec_sha256"),
            "execution_seq": row.get("execution_seq"),
        })
    execution = {"generation": row.get("generation"),
                 "binding_spec_sha256": row.get("binding_spec_sha256"),
                 "execution_seq": row.get("execution_seq")}
    resealed = measured + 5.0
    A.upsert_cost(root, stage, attempt, resealed,
                  {"complete": True, "status": "done",
                   "accounting_schema": entry.get("accounting_schema")},
                  "done")
    index = A.load_index(root)
    for item in index:
        if item["stage"] == stage and item["attempt"] == attempt:
            item["measured_wall_s"] = resealed
    A.save_index(root, index)
    refreshed = A._refresh_prepared_view(
        root, stage, attempt, prepared, execution)
    assert refreshed.problems == [], refreshed.problems
    assert refreshed.facts["measured_wall_s"] == resealed

    ledger = A.load_ledger(root)
    for item in ledger:
        if item["stage"] == stage and item["attempt"] == attempt:
            item["wall_s"] = round(resealed, 1) + 10.0
    A.save_ledger(root, ledger)
    refreshed = A._refresh_prepared_view(
        root, stage, attempt, prepared, execution)
    assert "commit_prepared_ledger_wall_s" in refreshed.problems

    # A reseal may only ever raise the billed cost.
    index = A.load_index(root)
    for item in index:
        if item["stage"] == stage and item["attempt"] == attempt:
            item["measured_wall_s"] = measured - 1.0
    A.save_index(root, index)
    ledger = A.load_ledger(root)
    for item in ledger:
        if item["stage"] == stage and item["attempt"] == attempt:
            item["measured_wall_s"] = measured - 1.0
            item["wall_s"] = round(measured - 1.0, 1)
    A.save_ledger(root, ledger)
    refreshed = A._refresh_prepared_view(
        root, stage, attempt, prepared, execution)
    assert "commit_prepared_cost_decreased" in refreshed.problems


class _MutableClock:
    def __init__(self, now: float) -> None:
        self.now = float(now)

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)

    def perf_counter(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def time_ns(self) -> int:
        return int(self.now * 1e9)

    def strftime(self, fmt: str) -> str:
        return "2026-11-04T00:00:00+0800"


def test_activation_delay_is_charged_to_r0_not_p(tmp_path, monkeypatch,
                                                migrated):
    """Binding publication belongs to R0; a slow activation cannot borrow P."""
    base = tmp_path / "activation_budget"
    root = base / "cohort"
    _write_legacy_prefix(root)
    provider = migrated["provider"]
    clock = _MutableClock(1_000.0)
    monkeypatch.setattr(A, "time", clock)
    spec_path, _spec_sha, _snapshot_path, auth, _binding, t0 = \
        _install_migration(root, provider, base)
    real_activate = A.activate_binding

    def delayed_activate(*args, **kwargs):
        clock.advance(200.0)
        return real_activate(*args, **kwargs)

    monkeypatch.setattr(A, "activate_binding", delayed_activate)
    result = A.run_binding_migration_audit(
        root, provider, mode="synthetic",
        tolerance=A.build_default_tolerance_contract(),
        authorization_sha=OLD_MARKER_AUTH, authorization=auth,
        binding_spec_path=spec_path, invocation_id="inv-migration-001",
        fixed_root=A.HISTORY_ROOT, r0_started_perf=t0)
    assert result["complete_evidence"] is False
    assert result["selected_attempts"] is None
    assert any("budget_exceeded" in problem
               for problem in result["problems"]), result["problems"]
    # P must never start: its own large budget cannot absorb R0 activation.
    assert not (A.stage_dir(root, "P") / "attempt_002").exists()
    rows = {(row["stage"], row["attempt"]): row
            for row in A.load_index(root)}
    r0 = rows[("R0", "attempt_002")]
    assert r0["status"] == "incomplete"
    assert r0["complete"] is False
    assert float(r0["measured_wall_s"]) > 181.40009476602427


def test_terminal_write_after_binds_commit_cost_and_registered_budget(
        tmp_path):
    root = tmp_path / "cohort"
    directory = _write_valid_journal(root)
    proofs = (directory / "terminal_write_after.json",
              root / "execution_events/000006/terminal_write_after.json")
    saved = {path: path.read_bytes() for path in proofs}
    try:
        # Consistent self-declared bounds that differ from the F commit cost
        # and the registered budget must not close the journal.
        for path in proofs:
            proof = A.strict_read_json(path)
            proof["observed_elapsed_s"] = 19.8921
            proof["committed_measured_wall_s"] = 24.8921
            A.write_json(path, proof)
        journal = A.scan_launch_journals(tmp_path)
        assert journal["known"] is False
        record = next(record for record in journal["records"]
                      if record["invocation_id"] == directory.name)
        assert {"terminal_write_after_committed",
                "terminal_event_write_after_committed"} <= set(
                    record["return_problems"])

        for path in proofs:
            path.write_bytes(saved[path])
            proof = A.strict_read_json(path)
            proof["remaining_s"] = 1234.0
            A.write_json(path, proof)
        journal = A.scan_launch_journals(tmp_path)
        assert journal["known"] is False
        record = next(record for record in journal["records"]
                      if record["invocation_id"] == directory.name)
        assert {"terminal_write_after_remaining",
                "terminal_event_write_after_remaining"} <= set(
                    record["return_problems"])
    finally:
        for path, data in saved.items():
            path.write_bytes(data)
    assert A.scan_launch_journals(tmp_path)["known"] is True


def test_non_object_terminal_json_is_unknown_not_crash(migrated):
    root = migrated["root"]
    invocation = "inv-migration-001"
    directory = A._launch_invocation_dir(root, invocation)
    candidate_path = directory / "terminal_candidate.json"
    launch_proof = directory / "terminal_write_after.json"
    event_candidate = root / "execution_events/000006/terminal_candidate.json"
    event_proof = root / "execution_events/000006/terminal_write_after.json"
    commit_path = root / "execution_events/000006/commit.json"
    paths = (candidate_path, launch_proof, event_candidate, event_proof,
             commit_path)
    saved = {path: path.read_bytes() for path in paths}
    if not (directory / "return_observation.json").is_file():
        result = migrated["result"]
        _path, _sha, problems = A.record_return_observation(
            root, invocation_id=invocation, pid=os.getpid(),
            ppid=os.getppid(), exit_code=3, signal=None,
            started_perf_counter=migrated["t0"],
            finished_perf_counter=A._now(),
            last_stage_end_perf_counter=result[
                "last_stage_end_perf_counter"],
            commits=result["commits"],
            candidate_receipts=result["candidates"],
            write_after=result["write_after"],
            stage_timings=result["stage_timings"],
            coverage_ok=result["complete_evidence"])
        assert problems == []
    assert A.scan_launch_journals(root.parent)["known"] is True

    def scan_record():
        journal = A.scan_launch_journals(root.parent)
        assert journal["known"] is False
        return next(record for record in journal["records"]
                    if record["invocation_id"] == invocation)

    try:
        A.write_json(candidate_path, [])
        assert "terminal_candidate_not_object" in \
            scan_record()["return_problems"]
        problems = A.verify_history(root)
        assert any("history_journal" in problem for problem in problems)
        row, problems = A.stage_state(
            root, "F", generation=A.GENERATION_002)
        assert row is None and problems

        candidate_path.write_bytes(saved[candidate_path])
        A.write_json(launch_proof, None)
        assert "terminal_write_after_not_object" in \
            scan_record()["return_problems"]

        launch_proof.write_bytes(saved[launch_proof])
        A.write_json(event_candidate, [])
        # Rebind the launch-side reference so the event object itself is
        # parsed; otherwise only the stale digest would be reported.
        launch = A.strict_read_json(candidate_path)
        launch["event_candidate"]["sha256"] = A.sha256_file(event_candidate)
        A.write_json(candidate_path, launch)
        assert "terminal_event_candidate_not_object" in \
            scan_record()["return_problems"]

        candidate_path.write_bytes(saved[candidate_path])
        event_candidate.write_bytes(saved[event_candidate])
        A.write_json(event_proof, [])
        assert "terminal_event_write_after_not_object" in \
            scan_record()["return_problems"]

        event_proof.write_bytes(saved[event_proof])
        A.write_json(commit_path, [])
        problems = A.verify_history(root)
        assert any("commit_not_object" in problem or
                   "history_journal" in problem for problem in problems)
        row, problems = A.stage_state(
            root, "F", generation=A.GENERATION_002)
        assert row is None and problems
    finally:
        for path, data in saved.items():
            path.write_bytes(data)
    assert A.scan_launch_journals(root.parent)["known"] is True

def test_implementation_identity_binds_files_not_git_state():
    impl = A.implementation_identity(PROJECT_ROOT)
    other = {**impl, "git": {"head": "0" * 40, "dirty": False,
                             "status_sha256": "0" * 64,
                             "tracked_diff_sha256": "0" * 64}}
    assert A.implementation_files_sha256(impl) == \
        A.implementation_files_sha256(other)
    changed = {**impl, "files": {**impl["files"], "x": "y"}}
    assert A.implementation_files_sha256(changed) != \
        A.implementation_files_sha256(impl)


def _gate_b_calibration() -> tuple[Path, Path]:
    base = PROJECT_ROOT / "derived/audit/w1_real_training_update_audit"
    contract = base / "calibration/attempt_019/tolerance_contract.json"
    verification = base / "calibration/attempt_019/calibration_verification.json"
    if not contract.is_file() or not verification.is_file():
        pytest.skip("final Gate B calibration artifacts are absent")
    return contract, verification


def _tiny_config(path: Path) -> Path:
    record = dict(TINY_CONFIG)
    path.write_text(json.dumps(record))
    return path


def test_ordinary_cli_outer_return_is_an_actual_process(tmp_path):
    contract, verification = _gate_b_calibration()
    config = _tiny_config(tmp_path / "tiny.json")
    out = tmp_path / "ordinary"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/audit_w1_real_training_update.py"),
        "--synthetic", "--out", str(out),
        "--tolerance-contract", str(contract),
        "--calibration-verification", str(verification),
        "--synthetic-config-json", str(config),
        "--device", "cuda",
    ]
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)
    proc = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                          text=True, timeout=900, env=env)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    result = json.loads((out / "audit_result.json").read_text())
    assert result["synthetic_acceptance"]["passed"] is True
    outer = json.loads((out / "synthetic_return.json").read_text())
    assert outer["source"]["kind"] == "outer_wait"
    assert outer["pid"] != os.getpid()
    assert outer["pid"] != outer["ppid"]
    assert outer["exit_code"] == 3
    assert outer["source"]["exit_code"] == 3
    assert outer["post_return_work_performed"] is False
    assert outer["finished_perf_counter"] >= outer["started_perf_counter"]


def test_real_migration_claim_precedes_admission_and_failure_is_debt(
        tmp_path, monkeypatch):
    """The minimal launcher claims first; a refused precheck leaves a debt."""
    script = _load_launch_script("w1_update_launch_cli_debt_test")

    base = tmp_path / "history"
    base.mkdir()
    root = base / "cohort"
    _write_legacy_prefix(root)
    invocation = "inv-precheck-debt"
    snapshot_path = tmp_path / "snapshot.json"
    A.write_json(snapshot_path, {
        "legacy_head_sha256": "b" * 64,
        "marker_sha256": A.sha256_file(root / "cohort.json"),
    })
    spec_path = tmp_path / "spec.json"
    A.write_json(spec_path, {
        "history_snapshot": {"path": str(snapshot_path),
                             "sha256": A.sha256_file(snapshot_path)},
        "stage_plan": [{"stage": "R0", "attempt": "attempt_002",
                        "execution_seq": 3}],
    })
    auth_path = tmp_path / "authorization.json"
    A.write_json(auth_path, {})
    contract_path = tmp_path / "tolerance_contract.json"
    A.write_json(contract_path, {})
    admission_path = tmp_path / "admission_package.json"
    A.write_json(admission_path, {})
    argv = [
        "--real", "--migration", "--out", str(root),
        "--authorization", str(auth_path),
        "--tolerance-contract", str(contract_path),
        "--admission-package", str(admission_path),
        "--binding-spec", str(spec_path),
        "--invocation-id", invocation,
    ]
    claim_path = script._bootstrap_real_migration_launch(
        argv=argv, base=base)
    assert claim_path is not None and claim_path.is_file()

    seen = {}

    def admission_probe(package):
        seen["claim_at_admission"] = claim_path.is_file()
        return ["admission_schema"]

    monkeypatch.setattr(script, "admission_package_problems", admission_probe)
    monkeypatch.setattr(script, "_history_root", lambda: base)
    monkeypatch.setattr(
        script, "load_authorization",
        lambda path: ({}, A.sha256_file(auth_path), []))
    monkeypatch.setattr(
        script, "load_tolerance_contract",
        lambda path, recompute=False: ({}, "b" * 64, []))
    monkeypatch.setattr(sys, "argv", ["audit_w1_real_training_update.py",
                                      *argv])
    with pytest.raises(SystemExit):
        script.main()

    assert seen["claim_at_admission"] is True
    directory = root / "launch_journal" / invocation
    assert (directory / "claim.json").is_file()
    failure = A.strict_read_json(directory / "failure.json")
    assert "admission package refused" in failure["reason"]
    journal = A.scan_launch_journals(base)
    assert journal["known"] is False
    assert any(p.startswith("launch_debt:")
               for p in journal["problems"])
    allowed = A.scan_launch_journals(base, allow_invocation=invocation)
    assert allowed["known"] is True


def test_fresh_migration_root_exempts_only_the_current_claim(tmp_path):
    script = _load_launch_script("w1_update_launch_cli_fresh_test")
    base = tmp_path / "history"
    base.mkdir()
    root = base / "cohort"
    _write_legacy_prefix(root)
    invocation = "inv-current"
    A.claim_launch(
        root, authorization_sha256="a" * 64, spec_sha256="b" * 64,
        legacy_head_sha256="c" * 64,
        marker_sha256=A.sha256_file(root / "cohort.json"),
        invocation_id=invocation, t0_perf_counter=A._now(),
        target={"stage": "R0", "attempt": "attempt_002",
                "execution_seq": 3})
    assert script._fresh_migration_root_problems(root) == [
        "migration_root_has_launch_journal"]
    assert script._fresh_migration_root_problems(
        root, allow_invocation=invocation) == []

    other = root / "launch_journal" / "other-invocation"
    other.mkdir()
    assert script._fresh_migration_root_problems(
        root, allow_invocation=invocation) == [
            "migration_root_has_launch_journal"]
    other.rmdir()

    failure = root / "launch_journal" / invocation / "failure.json"
    failure.write_text("{}")
    assert script._fresh_migration_root_problems(
        root, allow_invocation=invocation) == [
            "migration_root_has_launch_journal"]


def test_real_migration_full_precheck_reaches_worker_dispatch(
        tmp_path, monkeypatch):
    script = _load_launch_script("w1_update_launch_cli_dispatch_test")
    base = tmp_path / "history"
    base.mkdir()
    root = base / "cohort"
    _write_legacy_prefix(root)
    invocation = "inv-dispatch"
    auth_path = tmp_path / "authorization.json"
    A.write_json(auth_path, {})
    contract_path = tmp_path / "tolerance_contract.json"
    A.write_json(contract_path, {})
    admission_path = tmp_path / "admission_package.json"
    A.write_json(admission_path, {"schema": "isolated-test-package"})
    package_sha = A.sha256_file(admission_path)
    gate_sha = "3" * 64
    snapshot_path = tmp_path / "snapshot.json"
    A.write_json(snapshot_path, {
        "legacy_head_sha256": "b" * 64,
        "marker_sha256": A.sha256_file(root / "cohort.json"),
    })
    spec_path = tmp_path / "spec.json"
    A.write_json(spec_path, {
        "history_snapshot": {"path": str(snapshot_path),
                             "sha256": A.sha256_file(snapshot_path)},
        "stage_plan": [{"stage": "R0", "attempt": "attempt_002",
                        "execution_seq": 3}],
        "bindings": {
            "gate_b": {"sha256": gate_sha},
            "real_admission": {"sha256": package_sha},
        },
    })
    argv = [
        "--real", "--migration", "--out", str(root),
        "--authorization", str(auth_path),
        "--tolerance-contract", str(contract_path),
        "--admission-package", str(admission_path),
        "--binding-spec", str(spec_path),
        "--invocation-id", invocation,
    ]
    claim_path = script._bootstrap_real_migration_launch(
        argv=argv, base=base)
    assert claim_path is not None and claim_path.is_file()
    assert script._fresh_migration_root_problems(
        root, allow_invocation=invocation) == []

    auth_sha = A.sha256_file(auth_path)
    expected = {
        "run_root_realpath": str(root),
        "invocation_id": invocation,
        "spec_sha256": A.sha256_file(spec_path),
        "snapshot_sha256": A.sha256_file(snapshot_path),
        "legacy_head_sha256": "b" * 64,
        "gate_b_sha256": gate_sha,
        "real_admission_sha256": package_sha,
        "implementation_sha256": script.implementation_files_sha256(
            script.implementation_identity(script.ROOT)),
    }
    record = {"migration": expected}
    seen = {}
    monkeypatch.setattr(script, "_history_root", lambda: base)
    monkeypatch.setattr(
        script, "load_authorization",
        lambda path: (record, auth_sha, []))
    monkeypatch.setattr(
        script, "load_tolerance_contract",
        lambda path, recompute=False: ({}, "b" * 64, []))
    monkeypatch.setattr(
        script, "admission_package_problems", lambda package: [])
    monkeypatch.setattr(
        script, "load_external_binding_material",
        lambda *args, **kwargs: (object(), []))

    def stage_binding(*args, **kwargs):
        seen["stage_binding"] = True

    def spawn(command, log_path):
        seen["spawn"] = list(command)
        now = time.perf_counter()
        return {
            "pid": os.getpid() + 1, "ppid": os.getpid(), "exit_code": 0,
            "signal": None, "started_perf_counter": now,
            "finished_perf_counter": now + 0.1, "log_path": str(log_path),
        }

    monkeypatch.setattr(script, "stage_binding", stage_binding)
    monkeypatch.setattr(script, "_spawn_worker", spawn)
    monkeypatch.setattr(sys, "argv", ["audit_w1_real_training_update.py",
                                      *argv])
    assert script.main() == 3
    assert seen.get("stage_binding") is True
    assert "--migration-worker" in seen["spawn"]
    directory = root / "launch_journal" / invocation
    job = A.strict_read_json(directory / "worker_job.json")
    assert job["invocation_id"] == invocation
    failure = A.strict_read_json(directory / "failure.json")
    assert failure["reason"] == "worker_result_missing"


def test_migration_cli_outer_return_and_acceptance(tmp_path):
    contract, verification = _gate_b_calibration()
    config = _tiny_config(tmp_path / "tiny.json")
    out = tmp_path / "migration"
    invocation = "inv-cli-outer"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/audit_w1_real_training_update.py"),
        "--migration-synthetic", "--out", str(out),
        "--tolerance-contract", str(contract),
        "--calibration-verification", str(verification),
        "--synthetic-config-json", str(config),
        "--device", "cuda", "--invocation-id", invocation,
    ]
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)
    proc = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                          text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    acceptance = json.loads(
        (out / "migration_synthetic_acceptance.json").read_text())
    assert acceptance["passed"] is True
    observation_path = (out / "cohort" / A.LAUNCH_JOURNAL_DIR / invocation
                        / "return_observation.json")
    observation = json.loads(observation_path.read_text())
    assert observation["source"]["kind"] == "outer_wait"
    assert observation["pid"] != os.getpid()
    assert observation["pid"] != observation["ppid"]
    assert observation["source"]["exit_code"] == 3
    assert observation["coverage_ok"] is True
    assert acceptance["publication"]["return_observation"]["sha256"] == \
        hashlib.sha256(observation_path.read_bytes()).hexdigest()
    assert (out / "cohort" / A.BINDINGS_DIR / A.GENERATION_002).is_dir()
    assert (out / "cohort" / A.LAUNCH_JOURNAL_DIR / invocation
            / "claim.json").is_file()
    result = json.loads(
        (out / "migration_synthetic_result.json").read_text())
    assert result["cohort_training_updates"] == 18
    assert result["worker_return"]["pid"] == observation["pid"]

