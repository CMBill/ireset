"""Third-generation gradient-numerics retest over a closed v2 synthetic root.

The fixture reuses the frozen two-row v1 prefix and runs the ordinary
generation_002 migration chain first, then appends generation_003 with the
fixed R0/P/I/F plan.  No real checkpoint, dataset or asset is read.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ireset.eval.w1_real_training_update_audit as A
from ireset.paths import PROJECT_ROOT


def _load_migration_tests():
    path = Path(__file__).resolve().parent / \
        "test_w1_real_training_update_migration.py"
    spec = importlib.util.spec_from_file_location("w1_migration_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MIG = _load_migration_tests()


def _record_return(root: Path, invocation: str, result: dict,
                   t0: float) -> None:
    finished = A._now()
    last_end = (result.get("last_stage_end_perf_counter")
                or result["stage_timings"]["F"]["end_perf_counter"])
    path, _sha, problems = A.record_return_observation(
        root, invocation_id=invocation, pid=12345, ppid=12344,
        exit_code=3, signal=None, started_perf_counter=t0,
        finished_perf_counter=finished,
        last_stage_end_perf_counter=last_end,
        commits=result["commits"], candidate_receipts=result["candidates"],
        write_after={**{stage: True for stage in A.STAGE_BUDGETS},
                     "terminal": True},
        stage_timings=result["stage_timings"], source_kind="outer_wait",
        expected_exit_code=3, conclusion=result["conclusion"])
    assert path.is_file() and not problems, problems


def _retest_authorization(*, binding: A.BindingContext, snapshot_path: Path,
                          spec_path: Path, invocation: str) -> dict:
    draft = A.build_retest_authorization_draft(
        PROJECT_ROOT, binding=binding, snapshot_path=snapshot_path,
        spec_path=spec_path,
        retest_admission_path=spec_path.parent / "retest_admission.json",
        retest_admission_sha256="2" * 64,
        gate_b_v2_sha256="3" * 64,
        retest_acceptance_sha256="c" * 64,
        real_admission_sha256="4" * 64,
        invocation_id=invocation, authorized=True)
    draft.update(
        calibration_verification_sha256="b" * 64,
        tolerance_evidence_sha256="d" * 64,
        fp32_contract_sha256="e" * 64,
        acceptance_sha256="c" * 64,
        admission_package_sha256="4" * 64)
    assert A.authorization_problems(draft) == []
    return draft


@pytest.fixture(scope="module")
def retested(tmp_path_factory):
    base = tmp_path_factory.mktemp("w1_retest")
    root = base / "cohort"
    MIG._write_legacy_prefix(root)
    fixture_dir = base / "fixture"
    provider = A.SyntheticTrainingProvider(
        MIG.TINY_CONFIG, seed=A.FIXTURE_SEED, device="cuda")
    provider.prepare_fixture(fixture_dir)
    provider.load_fixture()
    assert provider.fixture["problems"] == []
    spec_path, _spec_sha, _snapshot_path, auth, _binding, t0 = \
        MIG._install_migration(root, provider, base)
    v2 = A.run_binding_migration_audit(
        root, provider, mode="synthetic",
        tolerance=A.build_default_tolerance_contract(),
        authorization_sha=MIG.OLD_MARKER_AUTH, authorization=auth,
        binding_spec_path=spec_path, invocation_id="inv-migration-001",
        fixed_root=A.HISTORY_ROOT, r0_started_perf=t0)
    assert v2["complete_evidence"] is True, v2["problems"]
    _record_return(root, "inv-migration-001", v2, t0)

    out = base / "retest"
    snapshot_path, snapshot_sha, snapshot = A.write_retest_history_snapshot(
        out, root)
    invocation = "inv-retest-001"
    spec = A.build_retest_binding_spec(
        PROJECT_ROOT, snapshot_path, snapshot_sha, snapshot,
        invocation_id=invocation,
        gate_b_v2={"sha256": "3" * 64,
                   "tolerance_evidence_sha256": "d" * 64,
                   "fp32_contract_sha256": "e" * 64,
                   "calibration_verification_sha256": "b" * 64,
                   "acceptance_sha256": "c" * 64},
        retest_acceptance={"sha256": "c" * 64},
        admission={"sha256": "4" * 64},
        protected_manifest={"sha256": "f" * 64})
    spec_path = out / "binding_spec.json"
    A.write_json_exclusive(spec_path, spec)
    spec_sha = A.sha256_file(spec_path)
    A.stage_binding(root, spec_path=spec_path, spec_sha256=spec_sha,
                    snapshot_sha256=snapshot_sha, invocation_id=invocation,
                    generation=A.GENERATION_003)
    binding, problems = A.load_binding(
        root, generation=A.GENERATION_003, require_active=False)
    assert binding is not None and problems == [], problems
    auth = _retest_authorization(
        binding=binding, snapshot_path=snapshot_path, spec_path=spec_path,
        invocation=invocation)
    auth_path = out / "authorization.json"
    A.write_json_exclusive(auth_path, auth)
    auth_sha = A.sha256_file(auth_path)
    claim_t0 = A._now()
    A.claim_launch(
        root, authorization_sha256=auth_sha, spec_sha256=spec_sha,
        legacy_head_sha256=snapshot["legacy_head_sha256"],
        marker_sha256=snapshot["marker_sha256"], invocation_id=invocation,
        t0_perf_counter=claim_t0,
        target={"stage": "R0", "attempt": "attempt_003",
                "execution_seq": 7},
        closed_history_head_sha256=snapshot["closed_history_head_sha256"])
    result = A.run_binding_migration_audit(
        root, provider, mode="synthetic",
        tolerance=A.build_default_tolerance_contract(),
        authorization_sha=auth_sha, authorization=auth,
        binding_spec_path=spec_path, invocation_id=invocation,
        fixed_root=A.HISTORY_ROOT, r0_started_perf=claim_t0, retest=True)
    _record_return(root, invocation, result, claim_t0)
    return {
        "base": base, "root": root, "provider": provider,
        "v2": v2, "result": result, "binding": binding,
        "snapshot_path": snapshot_path, "snapshot": snapshot,
        "spec_path": spec_path, "spec": spec, "auth": auth,
        "invocation": invocation,
    }


def test_retest_selects_the_fixed_generation_003_attempts(retested):
    result = retested["result"]
    assert result["complete_evidence"] is True, result["problems"]
    assert result["retest"] is True
    assert result["selected_attempts"] == {
        "R0": "attempt_003", "P": "attempt_003",
        "I": "attempt_002", "F": "attempt_002"}
    assert result["selected_chain_training_updates"] == 16
    assert result["cohort_training_updates"] == 34
    rows = A.load_index(retested["root"])
    v3 = [r for r in rows if r.get("generation") == A.GENERATION_003]
    assert {(r["stage"], r["attempt"]) for r in v3} == {
        ("R0", "attempt_003"), ("P", "attempt_003"),
        ("I", "attempt_002"), ("F", "attempt_002")}
    assert all(r.get("accounting_schema") == A.ACCOUNTING_SCHEMA_V3
               for r in v3)
    assert all(r.get("budget_revision_sha256")
               == retested["spec"]["budget_revision_sha256"] for r in v3)
    assert all(r.get("closed_history_head_sha256")
               == retested["snapshot"]["closed_history_head_sha256"]
               for r in v3)


def test_old_history_rows_are_unchanged(retested):
    snapshot = retested["snapshot"]
    index = {(r["stage"], r["attempt"]): r
             for r in A.load_index(retested["root"])}
    for rec in snapshot["index_rows"]:
        key = (rec["stage"], rec["attempt"])
        assert A.canonical_sha256(index[key]) == rec["canonical_sha256"]
    for seq in ("3", "4", "5", "6"):
        ref = snapshot["commits"][seq]
        assert A.sha256_file(A._reference_path(ref["path"])) == ref["sha256"]


def test_focus_sidecar_preserves_structural_zero_key_bias(retested):
    root = retested["root"]
    report = json.loads(
        (root / "independent/attempt_002/i_report.json").read_text())
    focus = report["focus"]
    assert focus["schema"] == A.FOCUS_SCHEMA
    assert len(focus["keys"]) == 48
    assert focus["total_bytes"] <= A.RETEST_FOCUS_BYTES_LIMIT
    config = retested["provider"].config
    arrays = np.load(root / "independent/attempt_002/i_focus_gradients.npz")
    for model in A.MODELS:
        for name in A.focus_parameter_names(config):
            if not name.endswith(".k.bias"):
                continue
            for u in A.UPDATES:
                for m in range(A.MICRO_COUNT):
                    key = A.focus_array_key("I", model, u, m, name)
                    assert key in arrays
                    assert np.all(arrays[key] == 0)
    problems = A.focus_manifest_problems(
        focus, stage="I", config=config,
        arrays={key: arrays[key] for key in arrays.files})
    assert problems == []


def test_retest_draft_binds_real_admission_calibration_fields(retested):
    from dataclasses import replace

    binding = retested["binding"]
    spec = json.loads(json.dumps(binding.spec))
    spec["bindings"]["gate_b_v2"] = {
        "sha256": "3" * 64,
        "calibration_verification_sha256": "b" * 64,
        "tolerance_evidence_sha256": "d" * 64,
        "fp32_contract_sha256": "e" * 64,
    }
    bound = replace(binding, spec=spec)
    draft = A.build_retest_authorization_draft(
        PROJECT_ROOT, binding=bound,
        snapshot_path=Path(retested["snapshot_path"]),
        spec_path=Path(retested["spec_path"]),
        retest_admission_path=Path(retested["spec_path"]).parent /
        "retest_admission.json",
        retest_admission_sha256="2" * 64,
        gate_b_v2_sha256="3" * 64,
        retest_acceptance_sha256="c" * 64,
        real_admission_sha256="4" * 64,
        invocation_id="inv-bindings", authorized=False)
    assert draft["calibration_verification_sha256"] == "b" * 64
    assert draft["tolerance_evidence_sha256"] == "d" * 64
    assert draft["fp32_contract_sha256"] == "e" * 64
    assert draft["admission_package_sha256"] == "4" * 64
    assert draft["acceptance_sha256"] == "c" * 64


def test_generation_002_authorization_is_not_reused(retested):
    auth = dict(retested["auth"])
    auth["schema"] = A.AUTHORIZATION_SCHEMA_V2
    assert A.authorization_problems(auth)
    broken = json.loads(json.dumps(retested["auth"]))
    broken["retest"]["closed_history_head_sha256"] = "not-a-hash"
    assert any("closed_history_head" in p
               for p in A.authorization_problems(broken))
    broken = json.loads(json.dumps(retested["auth"]))
    broken["retest"]["budget_revision_sha256"] = "not-a-hash"
    assert any("budget_revision" in p
               for p in A.authorization_problems(broken))


def test_budget_revision_binds_the_six_closed_rows(retested):
    snapshot = retested["snapshot"]
    revision = retested["spec"]["budget_revision"]
    assert revision["prior_updates"] == 18
    assert revision["incremental_update_limit"] == 16
    assert revision["cohort_update_limit"] == 34
    assert A.budget_revision_problems(
        revision, snapshot=snapshot, mode="synthetic") == []
    broken = json.loads(json.dumps(revision))
    broken["prior_call_counts"]["cpu_draw_units"] += 1
    assert "budget_revision_prior_calls" in A.budget_revision_problems(
        broken, snapshot=snapshot, mode="synthetic")


def _core_admission(retested, tmp_path: Path) -> tuple[dict, Path, Path]:
    snapshot_path = Path(retested["snapshot_path"])
    snapshot_sha = A.sha256_file(snapshot_path)
    snapshot = retested["snapshot"]
    protected = A.build_retest_protected_manifest(
        PROJECT_ROOT, retested["root"])
    protected_path = tmp_path / "protected_manifest.json"
    A.write_json(protected_path, protected)
    spec = A.build_retest_binding_spec(
        PROJECT_ROOT, snapshot_path, snapshot_sha, snapshot,
        invocation_id="inv-admission-test",
        gate_b_v2={"sha256": "3" * 64},
        retest_acceptance={"sha256": "c" * 64},
        admission={"sha256": "4" * 64},
        protected_manifest={"path": A.path_for_record(protected_path),
                            "sha256": A.sha256_file(protected_path)})
    spec_path = tmp_path / "spec.json"
    A.write_json_exclusive(spec_path, spec)
    spec_sha = A.sha256_file(spec_path)
    revision = spec["budget_revision"]
    revision_path = tmp_path / "budget_revision.json"
    A.write_json_exclusive(revision_path, revision)
    record = {
        "schema": A.RETEST_ADMISSION_SCHEMA,
        "status": {"mode": "pre_authorization_retest_readonly",
                   "authorized": False, "registered": False,
                   "run": "not_run", "section_9_checked": False},
        "passed": True,
        "problems": [],
        "snapshot": {"path": A.path_for_record(snapshot_path),
                     "sha256": snapshot_sha,
                     "closed_history_head_sha256": snapshot[
                         "closed_history_head_sha256"]},
        "spec": {"path": A.path_for_record(spec_path),
                 "sha256": spec_sha},
        "budget_revision": {
            "path": A.path_for_record(revision_path),
            "sha256": revision["sha256"],
            "file_sha256": A.sha256_file(revision_path)},
        "protected_manifest": {
            "path": A.path_for_record(protected_path),
            "sha256": A.sha256_file(protected_path),
            "entry_count": protected["entry_count"],
            "consumed_bytes": 0},
        "read_scope": {
            "checkpoint_payload_read": False,
            "npz_values_read": False,
            "model_constructor_calls": 0,
            "mujoco_compile_calls": 0,
            "forward_calls": 0,
            "vjp_calls": 0,
            "optimizer_steps": 0,
            "cohort_created": False,
            "registered_attempt_created": False,
        },
    }
    return record, spec_path, protected_path


def test_retest_admission_deeply_rejects_missing_and_drifted_refs(
        retested, tmp_path):
    record, spec_path, protected_path = _core_admission(retested, tmp_path)
    assert A.retest_admission_problems(
        record, project_root=PROJECT_ROOT, strict=False) == []
    broken = json.loads(json.dumps(record))
    broken["snapshot"]["path"] = str(tmp_path / "missing_snapshot.json")
    assert any("snapshot_path" in p for p in A.retest_admission_problems(
        broken, project_root=PROJECT_ROOT, strict=False))
    broken = json.loads(json.dumps(record))
    broken["protected_manifest"]["path"] = str(tmp_path / "missing_manifest.json")
    assert any("protected_manifest_path" in p
               for p in A.retest_admission_problems(
                   broken, project_root=PROJECT_ROOT, strict=False))
    # A protected entry changed while the admission's summary hash was
    # recomputed: the deep entry check must still reject it.
    manifest = json.loads(protected_path.read_text())
    first = next(iter(manifest["entries"]))
    manifest["entries"][first]["sha256"] = "0" * 64
    manifest["manifest_sha256"] = A.canonical_sha256(manifest["entries"])
    changed_path = tmp_path / "changed_manifest.json"
    A.write_json(changed_path, manifest)
    broken = json.loads(json.dumps(record))
    broken["protected_manifest"]["path"] = A.path_for_record(changed_path)
    broken["protected_manifest"]["sha256"] = A.sha256_file(changed_path)
    problems = A.retest_admission_problems(
        broken, project_root=PROJECT_ROOT, strict=False)
    assert any("protected_manifest_changed" in p for p in problems), problems
    missing_optional = A.retest_admission_problems(
        record, project_root=PROJECT_ROOT, strict=True)
    assert any("real_admission_missing" in p for p in missing_optional)


def _fresh_claim_root(tmp_path: Path, name: str) -> tuple[Path, str]:
    root = tmp_path / name
    root.mkdir()
    A.write_cohort(root, mode="synthetic")
    invocation = f"inv-{name}"
    A.claim_launch(
        root, authorization_sha256="a" * 64, spec_sha256="b" * 64,
        legacy_head_sha256="c" * 64, marker_sha256="d" * 64,
        invocation_id=invocation, t0_perf_counter=A._now(),
        target={"stage": "R0", "attempt": "attempt_003",
                "execution_seq": 7},
        closed_history_head_sha256="e" * 64)
    return root, invocation


def test_fresh_retest_root_exempts_only_the_current_claim(tmp_path):
    root, invocation = _fresh_claim_root(tmp_path, "current")
    assert A.fresh_retest_root_problems(
        root, allow_invocation=invocation) == []
    external = A.fresh_retest_root_problems(root)
    assert "retest_root_journal_unknown" in external


def test_cross_root_same_invocation_is_not_exempt(tmp_path):
    base = tmp_path / "history"
    base.mkdir()
    root1, invocation = _fresh_claim_root(base, "root1")
    root2 = base / "root2"
    root2.mkdir()
    A.write_cohort(root2, mode="synthetic")
    A.claim_launch(
        root2, authorization_sha256="a" * 64, spec_sha256="b" * 64,
        legacy_head_sha256="c" * 64, marker_sha256="d" * 64,
        invocation_id=invocation, t0_perf_counter=A._now(),
        target={"stage": "R0", "attempt": "attempt_003",
                "execution_seq": 7},
        closed_history_head_sha256="e" * 64)
    (root2 / "launch_journal" / invocation / "failure.json").write_text("{}")
    scan = A.scan_launch_journals(
        base, allow_invocation=invocation, allow_root=root1)
    assert scan["known"] is False
    assert any(Path(str(record["run"])).resolve() == root2.resolve()
               for record in scan["debts"])
    assert "retest_root_journal_unknown" in A.fresh_retest_root_problems(
        root1, allow_invocation=invocation)


def test_fresh_retest_root_rejects_debts_and_target_attempts(tmp_path):
    root, invocation = _fresh_claim_root(tmp_path, "debt")
    (root / "launch_journal" / invocation / "failure.json").write_text("{}")
    problems = A.fresh_retest_root_problems(
        root, allow_invocation=invocation)
    assert "retest_root_journal_unknown" in problems
    scan = A.scan_launch_journals(
        root.parent, allow_invocation=invocation, allow_root=root)
    assert not scan["known"]
    assert any("launch_debt" in p for p in scan["problems"])

    root, invocation = _fresh_claim_root(tmp_path, "tmp")
    (root / "launch_journal" / invocation /
     "claim.json.tmp.123.456").write_text("{}")
    problems = A.fresh_retest_root_problems(
        root, allow_invocation=invocation)
    assert any("journal_tmp" in p for p in problems)

    root, invocation = _fresh_claim_root(tmp_path, "extra")
    (root / "launch_journal" / invocation / "registered.json").write_text("{}")
    problems = A.fresh_retest_root_problems(
        root, allow_invocation=invocation)
    assert "retest_root_current_invocation_not_claim_only" in problems

    root, invocation = _fresh_claim_root(tmp_path, "target")
    (root / "production" / "attempt_003").mkdir(parents=True)
    problems = A.fresh_retest_root_problems(
        root, allow_invocation=invocation)
    assert any("retest_root_has_target_attempt:P/attempt_003" in p
               for p in problems)


def test_real_retest_cli_precheck_reaches_worker_dispatch(
        tmp_path, monkeypatch):
    script = MIG._load_launch_script("w1_retest_cli_dispatch_test")
    base = tmp_path / "history"
    base.mkdir()
    root = base / "cohort"
    root.mkdir()
    A.write_cohort(root, mode="synthetic")
    invocation = "inv-retest-dispatch"
    snapshot_path = tmp_path / "snapshot.json"
    A.write_json(snapshot_path, {
        "legacy_head_sha256": "b" * 64,
        "marker_sha256": A.sha256_file(root / "cohort.json"),
        "closed_history_head_sha256": "d" * 64,
    })
    spec_path = tmp_path / "spec.json"
    package_path = tmp_path / "real_admission.json"
    A.write_json(package_path, {"schema": "isolated-test-package"})
    package_sha = A.sha256_file(package_path)
    A.write_json(spec_path, {
        "history_snapshot": {"path": str(snapshot_path),
                             "sha256": A.sha256_file(snapshot_path)},
        "stage_plan": [{"stage": "R0", "attempt": "attempt_003",
                        "execution_seq": 7}],
        "budget_revision_sha256": "1" * 64,
        "bindings": {
            "gate_b_v2": {"sha256": "3" * 64},
            "retest_acceptance": {"sha256": "c" * 64},
            "real_admission": {"sha256": package_sha},
            "protected_manifest": {"sha256": "2" * 64},
        },
    })
    spec_sha = A.sha256_file(spec_path)
    auth_path = tmp_path / "authorization.json"
    A.write_json(auth_path, {})
    contract_path = tmp_path / "tolerance_contract.json"
    A.write_json(contract_path, {})
    retest_path = tmp_path / "retest_admission.json"
    A.write_json(retest_path, {
        "spec": {"sha256": spec_sha},
        "snapshot": {"sha256": A.sha256_file(snapshot_path)},
        "budget_revision": {"sha256": "1" * 64},
        "protected_manifest": {"sha256": "2" * 64},
    })
    retest_sha = A.sha256_file(retest_path)
    argv = [
        "--real", "--retest", "--out", str(root),
        "--authorization", str(auth_path),
        "--tolerance-contract", str(contract_path),
        "--admission-package", str(package_path),
        "--retest-admission", str(retest_path),
        "--binding-spec", str(spec_path),
        "--invocation-id", invocation,
    ]
    claim_path = script._bootstrap_real_migration_launch(
        argv=argv, base=base)
    assert claim_path is not None and claim_path.is_file()
    auth_sha = A.sha256_file(auth_path)
    expected = {
        "run_root_realpath": str(root),
        "invocation_id": invocation,
        "spec_sha256": spec_sha,
        "snapshot_sha256": A.sha256_file(snapshot_path),
        "closed_history_head_sha256": "d" * 64,
        "budget_revision_sha256": "1" * 64,
        "protected_manifest_sha256": "2" * 64,
        "retest_admission_sha256": retest_sha,
        "gate_b_v2_sha256": "3" * 64,
        "retest_acceptance_sha256": "c" * 64,
        "real_admission_sha256": package_sha,
        "implementation_sha256": script.implementation_files_sha256(
            script.implementation_identity(script.ROOT)),
    }
    record = {"retest": expected}
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
    monkeypatch.setattr(
        "ireset.eval.w1_real_training_update_audit."
        "retest_admission_problems", lambda *a, **k: [])

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
    assert "--retest-worker" in seen["spawn"]
    directory = root / "launch_journal" / invocation
    job = A.strict_read_json(directory / "worker_job.json")
    assert job["invocation_id"] == invocation
    assert job["retest_admission_sha256"] == retest_sha
    failure = A.strict_read_json(directory / "failure.json")
    assert failure["reason"] == "worker_result_missing"


def test_focus_manifest_rejects_missing_or_nonzero_key_bias(retested):
    report = json.loads(
        (retested["root"] / "independent/attempt_002/i_report.json").read_text())
    focus = report["focus"]
    arrays = np.load(
        retested["root"] / "independent/attempt_002/i_focus_gradients.npz")
    data = {key: arrays[key] for key in arrays.files}
    config = retested["provider"].config
    broken = json.loads(json.dumps(focus))
    broken["models"]["deterministic"]["parameters"][1]["updates"]["u1"][
        "micros"][0]["present"] = False
    assert A.focus_manifest_problems(
        broken, stage="I", config=config, arrays=data)
    data2 = dict(data)
    key = next(k for k in data2 if ".k.bias" in k)
    data2[key] = np.ones_like(data2[key])
    assert A.focus_manifest_problems(
        focus, stage="I", config=config, arrays=data2)
