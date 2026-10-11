"""A2 closure review round 3 (2026-10-10) — counterexamples for the 6
independent-review gaps (3 P1 + 3 P2). The authoritative verdicts stand
(UR no_quality_gain, stop line, training not_ready); these pin the
verification machinery that backs them.

* P1-1: the C2 merge re-derives the merged UR verdict, the gates and the
  evidence completeness from the report's OWN metric rows and binding map
  (``reverify_verdict_record``), and the two verdict locations must agree —
  a verdict string over failing gates, a disagreeing second location, or
  invalidated rows behind intact flags must not register;
* P1-2: ``verify_diagnosis_record`` re-hashes the snapshot artifacts ON
  DISK and re-derives the §5.5 verdict from the recorded run records —
  recorded pass flags alone prove nothing (editing run 2's canonical ``b``
  while keeping the recorded hashes used to pass both G0 and C2);
* P1-3: ``decide_diagnosis`` requires the full evidence set on BOTH runs
  BEFORE any equality counts — required keys missing from both snapshots
  (``q_a_indices`` / the canonical objective ``c`` / the P and A dumps) are
  ABSENT evidence (``diagnostic_incomplete``), never "consistent";
  ``q_a_nominal`` stays optional;
* P2-4: the retry cap counts ACTUAL invocations (one normal run + at most
  one justified retry), excludes skipped rows, and refuses a retry after a
  success;
* P2-5: the UR budget stops at exhaustion (``>=``) with a comparison
  reserve, and caps the CURRENT call's timeout at the remaining budget —
  the old code launched a 1800s-timeout call with 5s of budget left;
* P2-6: a truncated recorded argv (a value flag dangling at the end) is a
  per-row PARSE ERROR in the registration reconciliation and lands in
  evidence_gaps — never an IndexError that crashes the comparison before a
  report is written.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import a2_integrity as integ  # noqa: E402
from ireset.eval import a2_closure as a2c  # noqa: E402
from ireset.eval import teacher_comparison as tcmp  # noqa: E402

# the round-2 fixtures carry the heavy machinery (CLI runner, UR report
# builder with evaluation-produced gates, on-disk diagnosis snapshots,
# generation harness) — reimported, not duplicated
from test_omniretarget_a2_closure_review2 import (  # noqa: E402
    PILOT, _diag_run, _diagnosis_file, _prereg_file, _recovery_report,
    _run_cli, _run_generation_ur, _saved_run, _ur_freeze,
    _write_full_snapshot,
)
from test_omniretarget_recovery_source import (  # noqa: E402
    PILOT as RS_PILOT, PREREG, SMOKE_OK, _FakeReplayer, _admission,
    _build_ur_root,
)


# --- P1-1: verdict/gates/evidence re-derived from the report itself ----------

def _clean_ur_report(tmp_path) -> dict:
    p = _recovery_report(tmp_path)
    return json.loads(p.read_text())


@pytest.mark.parametrize("tamper,markers,forbidden", [
    (lambda r: None, [], []),
    # the reviewer's counterexample: verdict flipped to local_preferred
    # while the (recomputed) gates still fail — both locations edited so
    # the location check alone would NOT catch it
    (lambda r: (r.__setitem__("verdict", "local_preferred"),
                r["verdict_record"].__setitem__("verdict",
                                                "local_preferred")),
     ["verdict_not_supported_by_gates"], ["verdict_locations_disagree"]),
    # only ONE location edited: the two verdicts disagree
    (lambda r: r.__setitem__("verdict", "local_preferred"),
     ["verdict_locations_disagree", "verdict_not_supported_by_gates"], []),
    # binding invalidated but the recorded complete_evidence flag kept
    (lambda r: r["binding"].__setitem__(PILOT[0], False),
     ["metric_evidence_gaps_not_reproducible",
      "complete_evidence_contradicts_metrics",
      "verdict_contradicts_metrics"], []),
    # metric row invalidated the same way
    (lambda r: r["metrics"]["UR"][PILOT[0]].__setitem__("valid", False),
     ["metric_evidence_gaps_not_reproducible",
      "complete_evidence_contradicts_metrics",
      "verdict_contradicts_metrics"], []),
    # a gate verdict flipped against what the metrics support
    (lambda r: r["verdict_record"]["gates"]
        ["g3_penetration_and_feet"].__setitem__("passed", True),
     ["gate_passed_not_reproducible:g3_penetration_and_feet"], []),
    # a per-demo gate row flipped
    (lambda r: r["verdict_record"]["gates"]["g2_task_mpjpe"]["per_demo"]
        .__setitem__(PILOT[0], False),
     ["gate_rows_not_reproducible:g2_task_mpjpe"], []),
    # a fake METRIC gap injected into the recorded list
    (lambda r: r["verdict_record"]["evidence_gaps"]
        .append(f"{PILOT[0]}:UR_invalid"),
     ["metric_evidence_gaps_not_reproducible"], []),
    # an admission-side gap is EXCLUDED from the metrics comparison (the
    # producer appends those after the evaluation; the admission verdict
    # itself is verified by its own legs) — but since round 4 an
    # admission:* gap on a report whose admission PASSED is itself flagged
    (lambda r: r["verdict_record"]["evidence_gaps"]
        .append("admission:checkout_identity_mismatch"),
     ["admission_gap_without_admission_failure"],
     ["metric_evidence_gaps_not_reproducible"]),
])
def test_reverify_report_rederives_verdict_gates_and_evidence(
        tmp_path, tamper, markers, forbidden):
    report = _clean_ur_report(tmp_path / "rep")
    tamper(report)
    out = tcmp.reverify_verdict_record(report, u_side="UR")
    for m in markers:
        assert m in out["problems"], (m, out["problems"])
    for f in forbidden:
        assert f not in out["problems"], (f, out["problems"])
    # the clean report recomputes the exact recorded shape
    if not markers:
        assert out["recomputed"]["verdict"] == "no_quality_gain"
        assert out["recomputed"]["complete_evidence"] is True
        assert out["recomputed"]["gate_passed"]["g2_task_mpjpe"] is True
        assert out["recomputed"]["gate_passed"]["g5_degradation"] is False


def test_check_report_invariants_requires_verdict_location_agreement():
    from test_omniretarget_a2_closure_review2 import _report
    report = _report()
    assert a2c.check_report_invariants(report, PILOT)["passed"]
    report["verdict_record"]["verdict"] = "no_quality_gain"  # disagree
    inv = a2c.check_report_invariants(report, PILOT)
    assert not inv["passed"]
    assert any(v.startswith("verdict_locations_disagree")
               for v in inv["violations"])


def test_cli_merge_blocks_tampered_verdict_over_failing_gates(
        tmp_path, monkeypatch):
    """End to end: a bound, admission-clean UR report whose verdict says
    local_preferred over failing gates must NOT register the closure."""
    rc1, out, selection = _run_cli(tmp_path, monkeypatch)
    assert rc1 == 0
    snap = out / "baseline" / "snapshots" / "attempt_001" / \
        "baseline_evidence.json"
    diag = _diagnosis_file(tmp_path / "v1", snap)
    prereg = _prereg_file(tmp_path / "v2", diag, selection)
    report = _recovery_report(tmp_path / "v3",
                              prereg_sha=integ.sha256_file(prereg))
    data = json.loads(report.read_text())
    data["verdict"] = "local_preferred"
    data["verdict_record"]["verdict"] = "local_preferred"
    report.write_text(json.dumps(data))
    rc, _out, _sel = _run_cli(tmp_path, monkeypatch, args_extra={
        "diagnosis": diag, "recovery_report": report,
        "preregistration": prereg})
    assert rc == 2
    rec = json.loads(sorted((out / "decisions")
                            .glob("attempt_*/a2_closure.json"))[-1].read_text())
    assert rec["registration_status"]["complete"] is False
    assert any("verdict_not_supported_by_gates" in p for p in rec[
        "registration_status"]["blocking_reasons"])
    assert rec["merge_verification"]["verdict_reverification"]["problems"]


# --- P1-2: verify_diagnosis_record re-hashes disk + re-derives ---------------

def test_verify_diagnosis_record_rehashes_disk_artifacts(tmp_path):
    """The reviewer's counterexample: run 2's canonical ``b`` edited on
    disk while the RECORDED hashes stay — flags-only verification passed;
    re-hashing + re-derivation must not."""
    import diagnose_omniretarget_sub12 as diag
    sa, sb = tmp_path / "snap_a", tmp_path / "snap_b"
    _write_full_snapshot(sa)
    _write_full_snapshot(sb)
    eval_runs = [_diag_run(sa), _diag_run(sb)]
    verdict = diag.decide_diagnosis({"equal": True}, eval_runs)
    baseline = tmp_path / "baseline_evidence.json"
    baseline.write_text("{}")
    good = {
        "verdict": verdict,
        "control": {"comparison": {"equal": True}},
        "runs": [_saved_run(r) for r in eval_runs],
        "baseline": {"baseline_evidence": str(baseline),
                     "baseline_evidence_sha256":
                         integ.sha256_file(baseline)},
    }
    v = diag.verify_diagnosis_record(good)
    assert v["verified"], v["problems"]
    # tamper run 2's canonical b ON DISK; the recorded hashes are untouched
    arr = dict(np.load(sb / "failure_canonical.npz"))
    arr["b"] = arr["b"] + 1e-9
    np.savez(sb / "failure_canonical.npz", **arr)
    bad = diag.verify_diagnosis_record(good)
    assert not bad["verified"]
    assert "artifact_hashes_drifted_vs_disk:run2" in bad["problems"]
    assert "status_not_reproducible_from_artifacts" in bad["problems"]
    assert bad["rederived"]["status"] == "nondeterministic"


def test_verify_diagnosis_record_rejects_flag_only_records(tmp_path):
    """A verdict whose flags all say "consistent" but whose runs were never
    recorded (or don't resolve) cannot be re-derived — not verified."""
    import diagnose_omniretarget_sub12 as diag
    flags_only = {
        "verdict": {"status": "supports_step03_candidate",
                    "same_frame": True, "same_ordinal": True,
                    "key_arrays_equal": {"all_equal": True},
                    "canonical_vectors_equal": {"equal": True},
                    "artifact_sha256": {"equal": True},
                    "problem_shape_equal": True,
                    "phase1_categories": ["supports_step03_candidate",
                                          "supports_step03_candidate"],
                    "rho_run1": 0.2486138173835237,
                    "failing_frame_run1": 171, "ordinal_run1": 5},
        "control": {"comparison": {"equal": True}},
        "baseline": {},
    }
    v = diag.verify_diagnosis_record(flags_only)
    assert not v["verified"]
    assert "runs_not_recorded_for_reverification" in v["problems"]


# --- P1-3: required evidence must EXIST on both runs -------------------------

def _both_snapshots(tmp_path):
    sa, sb = tmp_path / "snap_a", tmp_path / "snap_b"
    _write_full_snapshot(sa)
    _write_full_snapshot(sb)
    return sa, sb


@pytest.mark.parametrize("delete", [
    "q_a_indices",       # variable-column mapping dropped from both ctx
    "canonical_c",       # objective vector dropped from both canonicals
    "P_file",            # failure_canonical_P.npz deleted on both sides
    "A_file",            # failure_canonical_A.npz deleted on both sides
])
def test_absent_required_evidence_is_incomplete_not_consistent(
        tmp_path, delete):
    """The reviewer's counterexample: deleting required evidence from BOTH
    snapshots used to read as both_missing == equal and still yielded
    supports_step03_candidate."""
    import diagnose_omniretarget_sub12 as diag
    sa, sb = _both_snapshots(tmp_path)
    for s in (sa, sb):
        if delete == "q_a_indices":
            ctx = dict(np.load(s / "failure_context.npz"))
            ctx.pop("q_a_indices")
            np.savez(s / "failure_context.npz", **ctx)
        elif delete == "canonical_c":
            can = dict(np.load(s / "failure_canonical.npz"))
            can.pop("c")
            np.savez(s / "failure_canonical.npz", **can)
        elif delete == "P_file":
            (s / "failure_canonical_P.npz").unlink()
        elif delete == "A_file":
            (s / "failure_canonical_A.npz").unlink()
    v = diag.decide_diagnosis({"equal": True},
                              [_diag_run(sa), _diag_run(sb)])
    assert v["status"] == "diagnostic_incomplete"
    assert any(p.startswith("missing_required_evidence")
               for p in v["problems"]), v


def test_optional_q_a_nominal_both_missing_still_supports(tmp_path):
    """``q_a_nominal`` is OPTIONAL (first-round snapshots predate the key):
    both-missing stays comparable evidence for that key alone."""
    import diagnose_omniretarget_sub12 as diag
    sa, sb = _both_snapshots(tmp_path)
    assert "q_a_nominal" not in np.load(sa / "failure_context.npz").files
    v = diag.decide_diagnosis({"equal": True},
                              [_diag_run(sa), _diag_run(sb)])
    assert v["status"] == "supports_step03_candidate"
    assert v["key_arrays_equal"]["per_key"]["q_a_nominal"] == "both_missing"


# --- P2-4: retry cap counts actual invocations --------------------------------

def test_retry_cap_counts_actual_invocations_not_failures(
        tmp_path, monkeypatch):
    """The reviewer's counterexample: one failed run + one SUCCESSFUL run
    still authorized a third call — the cap must count real calls."""
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "ur3"
    prior = [{"demo": PILOT[0], "status": "generation_failed"},
             {"demo": PILOT[0], "status": "ok"}]
    rc, calls = _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir,
                                   prereg, prior_rows=prior)
    assert PILOT[0] not in calls          # two ACTUAL calls: no third
    assert calls == PILOT[1:]
    m = json.loads(sorted(out_dir.glob("attempt_*/generation_manifest.json"))
                   [-1].read_text())
    row0 = next(r for r in m["rows"] if r["demo"] == PILOT[0])
    assert row0["status"] == "skipped_retry_cap"
    assert row0["prior_invocations"] == [
        {"attempt": "attempt_001", "status": "generation_failed"},
        {"attempt": "attempt_002", "status": "ok"}]


def test_skipped_prior_rows_never_count_as_invocations(tmp_path, monkeypatch):
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "ur4"
    prior = [{"demo": PILOT[0], "status": "generation_failed"},
             {"demo": PILOT[0], "status": "skipped_budget"}]
    rc, calls = _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir,
                                   prereg, prior_rows=prior)
    # one real (failed) call + one skipped row: the retry is justified
    assert calls[:1] == [PILOT[0]]


def test_retry_after_success_is_refused(tmp_path, monkeypatch):
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "ur5"
    prior = [{"demo": PILOT[0], "status": "ok"}]
    rc, calls = _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir,
                                   prereg, prior_rows=prior)
    assert PILOT[0] not in calls         # a retry needs a FAILED prior run
    m = json.loads(sorted(out_dir.glob("attempt_*/generation_manifest.json"))
                   [-1].read_text())
    row0 = next(r for r in m["rows"] if r["demo"] == PILOT[0])
    assert row0["status"] == "skipped_retry_cap"
    assert "SUCCEEDED" in row0["reason"]


# --- P2-5: budget exhaustion boundary + per-call timeout cap ------------------

def _run_gen_timed(monkeypatch, tmp_path, freeze, out_dir, prereg, *,
                   prior_rows, prior_wall):
    """Like the round-2 generation harness but records the timeout each
    call was launched with (the budget cap under test)."""
    import generate_omniretarget_teachers as gen
    calls = []
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)

    def fake_generate_one(demo, env_python, attempt_dir, data_path, **kw):
        calls.append((demo, kw.get("timeout_s")))
        return {"demo": demo, "argv": ["--retargeter.step-size", "0.3"],
                "cwd": "", "exit_code": 0, "wall_s": 10.0,
                "config": "fake", "status": "ok",
                "binding": {"passed": True}, "error": None,
                "converted": f"{attempt_dir.name}/{demo}_c.npz"}

    monkeypatch.setattr(gen, "generate_one", fake_generate_one)
    monkeypatch.setattr(integ, "sha256_file", lambda p: "fakehash")
    monkeypatch.setattr(integ, "measure_env_identity",
                        lambda p: {"env_python": "x",
                                   "env_uv_lock_sha256": "locksha",
                                   "env_mujoco_version": "3.10.0",
                                   "env_lock_mujoco_version": "3.10.0",
                                   "env_probe_error": None})
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda a, root=None: {"passed": True})
    monkeypatch.setattr(integ, "verify_checkout_against_freeze",
                        lambda a: {"passed": True})
    monkeypatch.setattr(integ, "verify_inputs_against_freeze",
                        lambda entries, npz_dir, pt_dir: {
                            e["demo"]: {"npz_matches": True,
                                        "pt_matches": True}
                            for e in entries})
    monkeypatch.setattr(gen, "load_source_record", lambda *a, **k: {
        "prereg": {"experiment_id": "a2_recovery_step03_v1",
                   "source_id": "upstream_recovery_step03_v1"},
        "preregistration": "p", "preregistration_sha256": "shaReg",
        "config_smoke": {}, "config_smoke_full": {}})
    monkeypatch.setattr(gen, "run_config_smoke",
                        lambda *a, **k: {"diff": {}, "default": {},
                                         "recovery": {}})
    if prior_rows is not None:
        # round-6: genuine one-attempt-per-invocation layout with wall_s on
        # each row (summing to prior_wall) and the scope registered on both
        # the index entry and the manifest (the history validator enforces
        # exactly this shape)
        per = prior_wall / len(prior_rows)
        for i, r in enumerate(prior_rows, start=1):
            att = out_dir / f"attempt_{i:03d}"
            att.mkdir(parents=True, exist_ok=True)
            row = {"wall_s": per, **r}
            (att / "generation_manifest.json").write_text(json.dumps(
                {"rows": [row], "total_wall_s": row["wall_s"],
                 "attempt_scope": [r["demo"]]}))
        (out_dir / integ.ATTEMPTS_INDEX).write_text(json.dumps(
            {"attempts": [
                {"attempt": f"attempt_{i:03d}", "scope": [r["demo"]],
                 "preregistration_sha256": "shaReg"}
                for i, r in enumerate(prior_rows, start=1)]}))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--freeze", str(freeze), "--out-dir", str(out_dir),
        "--source-record", str(prereg)])
    return gen.main(), calls


def test_budget_stops_at_exact_exhaustion(tmp_path, monkeypatch):
    """The reviewer's boundary counterexample: cumulative time exactly at
    the budget used to launch anyway (`>` comparison)."""
    import generate_omniretarget_teachers as gen
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "b1"
    prior = [{"demo": PILOT[0], "status": "generation_failed"}]
    rc, calls = _run_gen_timed(
        monkeypatch, tmp_path, freeze, out_dir, prereg,
        prior_rows=prior, prior_wall=gen.UR_GENERATION_BUDGET_S)
    assert calls == []                    # exhausted, not "> exhausted"
    m = json.loads(sorted(out_dir.glob("attempt_*/generation_manifest.json"))
                   [-1].read_text())
    assert all(r["status"] == "skipped_budget" for r in m["rows"])


def test_budget_caps_the_current_call_timeout(tmp_path, monkeypatch):
    """5s of budget left: the call must launch with timeout 5 (the old code
    launched with the default 1800s and a 10s stand-in blew the 3h total to
    10805s); after it, the budget is spent and the rest are skipped."""
    import generate_omniretarget_teachers as gen
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "b2"
    prior = [{"demo": PILOT[0], "status": "generation_failed"}]
    rc, calls = _run_gen_timed(
        monkeypatch, tmp_path, freeze, out_dir, prereg,
        prior_rows=prior, prior_wall=gen.UR_GENERATION_BUDGET_S - 5)
    assert len(calls) == 1
    demo, timeout = calls[0]
    assert timeout == 5                    # min(1800, remaining=5)
    m = json.loads(sorted(out_dir.glob("attempt_*/generation_manifest.json"))
                   [-1].read_text())
    row0 = next(r for r in m["rows"] if r["demo"] == demo)
    assert row0["budget_timeout_s"] == 5
    assert row0["budget_remaining_s"] == 5.0
    # the 10s stand-in spent the budget: every later demo is skipped, so
    # generation + comparison reserve never exceeds the 3h experiment cap
    assert all(r["status"] == "skipped_budget" for r in m["rows"]
               if r["demo"] != demo)
    assert (gen.UR_GENERATION_BUDGET_S + gen.UR_COMPARISON_RESERVE_S
            == gen.UR_TOTAL_BUDGET_S)


# --- P2-6: truncated argv is a parse error, not a crash -----------------------

def test_argv_flag_value_and_parse_error_helpers():
    import omniretarget_teacher_comparison as cmp_mod
    truncated = ["py", "robot_retarget.py", "--retargeter.step-size"]
    assert cmp_mod._argv_flag_value(truncated,
                                    "--retargeter.step-size") is None
    assert cmp_mod._argv_parse_error(truncated) == \
        "dangling_flag:--retargeter.step-size"
    ok = ["py", "robot_retarget.py", "--retargeter.step-size", "0.3"]
    assert cmp_mod._argv_flag_value(ok, "--retargeter.step-size") == "0.3"
    assert cmp_mod._argv_parse_error(ok) is None
    assert cmp_mod._argv_flag_value(ok, "--absent-flag") is None


def _truncate_step_flag(manifest):
    """Drop the step-size VALUE from the first consumed row's command —
    the reviewer's exact truncation shape (flag dangling at the end)."""
    manifest["rows"][0]["argv"] = manifest["rows"][0]["argv"][:-1]


def test_truncated_argv_is_a_registered_parse_error_not_a_crash(
        tmp_path, monkeypatch):
    ur, npz_dir, pt_dir = _build_ur_root(
        tmp_path, manifest_mutate=_truncate_step_flag)
    # the exact pre-fix crash site: the reconciliation used to IndexError
    # on `argv[index(flag) + 1]` before any report could be written
    adm = _admission(tmp_path, monkeypatch, ur, npz_dir, pt_dir,
                     smoke=SMOKE_OK)
    assert adm["passed"] is False
    reg = adm["source_registration"]
    assert reg["passed"] is False
    per = reg["per_attempt"]["attempt_001"]
    assert per["argv_parse_errors"] == {
        RS_PILOT[0]: "dangling_flag:--retargeter.step-size"}
    assert per["argv_step_size_ok"] is False
    assert RS_PILOT[0] in per["argv_step_size_bad_demos"]
    # the truncated row never reaches the child smoke; the parse error is
    # itself the recorded reason
    assert per["argv_smoke"][RS_PILOT[0]] == {
        "error": "argv_parse_error:dangling_flag:--retargeter.step-size"}


def test_truncated_argv_lands_in_evidence_gaps(tmp_path, monkeypatch):
    """CLI end to end: the report is WRITTEN with the parse-error reason
    and an incomplete verdict (pre-fix: IndexError, no report at all)."""
    import omniretarget_teacher_comparison as cmp_mod
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path / "derived")
    prereg_path = tmp_path / "preregistration.json"
    prereg_path.write_text(json.dumps(PREREG))
    ur, npz_dir, pt_dir = _build_ur_root(
        tmp_path, prereg_sha=integ.sha256_file(prereg_path),
        manifest_mutate=_truncate_step_flag)
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    expected = integ.read_checkout_identity()["holosoma_revision"]
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit": expected}}))
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [
            {"demo": d,
             "npz_sha256": integ.sha256_file(npz_dir / f"{d}_original.npz"),
             "pt_sha256": integ.sha256_file(pt_dir / f"{d}.pt")}
            for d in RS_PILOT]}))
    np.savez(freeze / "pairing_source_boxes.npz",
             **{d: np.zeros((4, 7)) for d in RS_PILOT})
    monkeypatch.setattr(cmp_mod, "TEACHER_TRAJ_DIR", npz_dir)
    monkeypatch.setattr(cmp_mod, "OMOMO_NEW_DIR", pt_dir)
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda assets: {"passed": True})
    monkeypatch.setattr(integ, "measure_env_identity", lambda p: {
        "env_python": "envy", "env_uv_lock_sha256": "locksha",
        "env_mujoco_version": "3.10.0", "env_lock_mujoco_version": "3.10.0",
        "env_probe_error": None})
    monkeypatch.setattr(cmp_mod, "_argv_config_smoke",
                        lambda p, argv: copy.deepcopy(SMOKE_OK))
    monkeypatch.setattr(cmp_mod, "TeacherQualityReplayer", _FakeReplayer)
    out = tmp_path / "derived" / "recovery"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--freeze", str(freeze), "--generated-dir", str(ur),
        "--source-record", str(prereg_path), "--out", str(out)])
    rc = cmp_mod.main()                    # must not raise
    report = json.loads(
        (out / "cmp" / "attempt_001" / "teacher_selection.json").read_text())
    assert report["verdict"] == "incomplete"
    assert (f"admission:source_registration:attempt_001:argv_parse:"
            f"{RS_PILOT[0]}:dangling_flag:--retargeter.step-size"
            ) in report["verdict_record"]["evidence_gaps"]
