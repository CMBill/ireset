"""A2 closure review round 4 (2026-10-10) — counterexamples for the 4
boundary-recheck gaps (1 P1 + 3 P2). The authoritative verdicts stand
(U0 incomplete, UR no_quality_gain, training not_ready); these pin the
verification machinery that backs them.

* P1-1: the per-run §5.5 category is RE-DERIVED from the phase-one facts
  (status/rho/max_violation through ``classify_phase_one``) and the run
  facts in the record are reconciled with the snapshot's OWN
  failure_meta.json/child_summary.json — editing ``phase_one.status`` to
  ``infeasible`` or ``max_violation`` to 1.0/NaN while keeping the old
  ``category`` used to pass the diagnosis verification, the C2 merge and
  the G0 registration;
* P2-2: with admission PASSED, a report may no longer suppress a
  metrics-complete verdict — the demotion to ``incomplete`` +
  ``complete_evidence=false`` + ``admission:*`` gaps is legitimate only
  when the report's own admission actually FAILED;
* P2-3: ``--demos A,A`` is rejected before any subprocess call (one demo
  twice would bypass the per-demo invocation cap read once before the
  loop), and each actual call updates the in-run count;
* P2-4: an attempt REGISTERED in the index whose manifest cannot be read
  is ``history_incomplete`` — the run stops BEFORE generation instead of
  treating unreadable history as zero invocations and zero cost.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import a2_integrity as integ  # noqa: E402
from ireset.eval import teacher_comparison as tcmp  # noqa: E402

# the round-2/3 fixtures carry the heavy machinery (CLI runner, UR report
# builder, on-disk diagnosis snapshots, generation harness) — reimported
from test_omniretarget_a2_closure_review2 import (  # noqa: E402
    PILOT, _canonical_failure, _diag_run, _diagnosis_file, _prereg_file,
    _recovery_report, _run_cli, _run_generation_ur, _saved_run, _ur_freeze,
    _write_full_snapshot,
)


# --- P1-1: phase_one category re-derived from the facts -----------------------

@pytest.mark.parametrize("tamper", [
    {"status": "infeasible"},                  # reviewer's exact flip
    {"max_violation": 1.0},                    # residual beyond the bound
    {"max_violation": float("nan")},           # non-finite residual
    {"rho": None},                             # facts no longer decide
])
def test_phase1_category_contradicting_its_facts_is_incomplete(tmp_path,
                                                                tamper):
    """The recorded ``category=supports_step03_candidate`` is a CLAIM: the
    aggregation re-classifies from status/rho/max_violation and a recorded
    label that contradicts its own facts is internally inconsistent."""
    import diagnose_omniretarget_sub12 as diag
    sa, sb = tmp_path / "snap_a", tmp_path / "snap_b"
    _write_full_snapshot(sa)
    _write_full_snapshot(sb)
    runs = [_diag_run(sa), _diag_run(sb)]
    for run in runs:      # both runs tampered identically: determinism holds
        run["summary"]["failure"]["phase_one"].update(tamper)
    v = diag.decide_diagnosis({"equal": True}, runs)
    assert v["status"] == "diagnostic_incomplete"
    assert any(p.startswith("phase1_category_not_reproducible")
               for p in v["problems"]), v
    # the recorded label survives in the problem for auditability
    assert any("recorded='supports_step03_candidate'" in p
               for p in v["problems"])


def test_phase1_structural_statuses_rederive_exactly(tmp_path):
    """The helper's own early-exit categories are preserved by the
    re-derivation (they are named, not classified from the trio)."""
    import diagnose_omniretarget_sub12 as diag
    sa, sb = tmp_path / "snap_a", tmp_path / "snap_b"
    _write_full_snapshot(sa)
    _write_full_snapshot(sb)
    for status, category in (
            ("not_single_soc", "phase1_not_single_soc"),
            ("soc_x_not_variable", "phase1_soc_x_not_variable"),
            ("phase1_solver_error", "phase1_solver_error")):
        runs = [_diag_run(sa), _diag_run(sb)]
        for run in runs:
            p1 = run["summary"]["failure"]["phase_one"]
            p1["status"], p1["category"] = status, category
        v = diag.decide_diagnosis({"equal": True}, runs)
        # internally consistent facts: NOT a category problem — the terminal
        # §5.5 row is the no-candidate outcome, never a supporting verdict
        assert v["status"] != "supports_step03_candidate"
        assert not any(p.startswith("phase1_category_not_reproducible")
                       for p in v.get("problems", [])), (status, v)


def test_verify_diagnosis_record_reconciles_run_facts_with_disk(tmp_path):
    """Round-4 leg: the record's whole failure dict (phase_one included)
    must equal the snapshot's OWN failure_meta.json / child_summary.json —
    tampering either side must break verification."""
    import diagnose_omniretarget_sub12 as diag
    baseline = tmp_path / "baseline_evidence.json"
    baseline.write_text("{}")
    diag_file = _diagnosis_file(tmp_path / "d", baseline)
    rec = json.loads(diag_file.read_text())
    assert diag.verify_diagnosis_record(rec)["verified"]

    # (a) reviewer's counterexample in the RECORD: status flipped, category
    # kept — the re-derivation sees facts that no longer classify as
    # supports AND the disk jsons keep the original facts
    rec_a = copy.deepcopy(rec)
    rec_a["runs"][1]["failure"]["phase_one"]["status"] = "infeasible"
    v_a = diag.verify_diagnosis_record(rec_a)
    assert not v_a["verified"]
    assert "status_not_reproducible_from_artifacts" in v_a["problems"]
    assert "run_facts_drifted_vs_disk:run2:meta" in v_a["problems"]
    assert "run_facts_drifted_vs_disk:run2:summary" in v_a["problems"]

    # (b) tamper only the DISK failure_meta (record intact): the metric
    # re-derivation alone would still pass; only the disk leg catches it
    rec_b = json.loads(diag_file.read_text())
    snap_b = Path(rec_b["runs"][1]["snapshot"])
    fm = json.loads((snap_b / "failure_meta.json").read_text())
    fm["failure"]["phase_one"]["rho"] = 0.1
    (snap_b / "failure_meta.json").write_text(json.dumps(fm))
    v_b = diag.verify_diagnosis_record(rec_b)
    assert not v_b["verified"]
    assert "run_facts_drifted_vs_disk:run2:meta" in v_b["problems"]
    assert "run_facts_drifted_vs_disk:run2:summary" not in v_b["problems"]

    # (c) the run-fact jsons missing entirely from the snapshot
    rec_c = json.loads(diag_file.read_text())
    snap_a = Path(rec_c["runs"][0]["snapshot"])
    (snap_a / "failure_meta.json").unlink()
    (snap_a / "child_summary.json").unlink()
    v_c = diag.verify_diagnosis_record(rec_c)
    assert not v_c["verified"]
    assert "run_facts_missing_on_disk:run1" in v_c["problems"]


def test_cli_merge_blocks_tampered_phase_one_status(tmp_path, monkeypatch):
    """End to end: a diagnosis record whose phase_one facts were flipped
    after the fact must NOT register the closure (the C2 merge runs the
    same re-derivation + disk reconciliation)."""
    rc1, out, selection = _run_cli(tmp_path, monkeypatch)
    assert rc1 == 0
    snap = out / "baseline" / "snapshots" / "attempt_001" / \
        "baseline_evidence.json"
    diag_file = _diagnosis_file(tmp_path / "t1", snap)
    data = json.loads(diag_file.read_text())
    for run in data["runs"]:
        run["failure"]["phase_one"]["status"] = "infeasible"
    diag_file.write_text(json.dumps(data))
    prereg = _prereg_file(tmp_path / "t2", diag_file, selection)
    report = _recovery_report(tmp_path / "t3",
                              prereg_sha=integ.sha256_file(prereg))
    rc, _out, _sel = _run_cli(tmp_path, monkeypatch, args_extra={
        "diagnosis": diag_file, "recovery_report": report,
        "preregistration": prereg})
    assert rc == 2
    rec = json.loads(sorted((out / "decisions")
                            .glob("attempt_*/a2_closure.json"))[-1].read_text())
    assert rec["registration_status"]["complete"] is False
    assert any("phase1_category_not_reproducible" in p or
               "status_not_reproducible_from_artifacts" in p
               for p in rec["registration_status"]["blocking_reasons"])


# --- P2-2: admission passing forbids suppressing a complete verdict ----------

def test_admission_passing_forbids_suppressing_complete_verdict(tmp_path):
    """The reviewer's counterexample: admission passed, but both verdict
    locations demoted to incomplete with a fake admission:* gap — the
    re-derivation computes a COMPLETE no_quality_gain from the metrics."""
    report = json.loads(_recovery_report(tmp_path / "r").read_text())
    assert report["admission"]["passed"] is True
    report["verdict"] = "incomplete"
    report["verdict_record"]["verdict"] = "incomplete"
    report["verdict_record"]["complete_evidence"] = False
    report["verdict_record"]["evidence_gaps"].append(
        "admission:checkout_identity_mismatch")
    out = tcmp.reverify_verdict_record(report, u_side="UR")
    assert out["recomputed"]["verdict"] == "no_quality_gain"
    assert out["recomputed"]["complete_evidence"] is True
    assert "verdict_not_supported_by_gates" in out["problems"]
    assert "complete_evidence_not_reproducible" in out["problems"]
    assert "admission_gap_without_admission_failure" in out["problems"]


def test_admission_failure_demotion_still_verifies(tmp_path):
    """The legitimate shape stays legal: when admission actually FAILED the
    comparison entry demotes the metrics-complete verdict to incomplete and
    appends admission:* gaps — re-derivation must not flag that."""
    report = json.loads(_recovery_report(tmp_path / "r2").read_text())
    report["admission"]["passed"] = False
    report["verdict"] = "incomplete"
    report["verdict_record"]["verdict"] = "incomplete"
    report["verdict_record"]["complete_evidence"] = False
    report["verdict_record"]["evidence_gaps"].append(
        "admission:env_lock_drift_vs_generation")
    out = tcmp.reverify_verdict_record(report, u_side="UR")
    assert out["problems"] == []
    assert out["recomputed"]["verdict"] == "no_quality_gain"


def test_clean_report_still_recomputes_exactly(tmp_path):
    """The authoritative shape (admission passed, no gaps) re-derives with
    zero problems — the strict match is not over-tightened."""
    report = json.loads(_recovery_report(tmp_path / "r3").read_text())
    out = tcmp.reverify_verdict_record(report, u_side="UR")
    assert out["problems"] == []
    assert out["recomputed"]["verdict"] == "no_quality_gain"
    assert out["recomputed"]["complete_evidence"] is True


# --- P2-3: duplicate demos rejected before any call ---------------------------

def test_duplicate_demos_rejected_before_generation(tmp_path, monkeypatch):
    """The reviewer's counterexample: after one failure, ``--demos A,A``
    used to call generate_one TWICE more (three actual invocations; the
    cap history was read once before the loop)."""
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "dup"
    prior = [{"demo": PILOT[0], "status": "generation_failed"}]
    with pytest.raises(SystemExit, match="duplicate demo"):
        _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir, prereg,
                           prior_rows=prior,
                           demos=f"{PILOT[0]},{PILOT[0]}")
    # refused BEFORE any subprocess call and BEFORE a new attempt dir
    assert not list(out_dir.glob("attempt_00[2-9]*"))


def test_single_demo_scope_still_runs(tmp_path, monkeypatch):
    """A legitimate singleton retry scope keeps working."""
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "one"
    prior = [{"demo": PILOT[0], "status": "generation_failed"}]
    rc, calls = _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir,
                                   prereg, prior_rows=prior,
                                   demos=PILOT[0])
    assert calls == [PILOT[0]]
    # partial scope keeps the FULL denominator: reports INCOMPLETE, not a
    # shrunken COMPLETE (rc 1 is the designed outcome)
    assert rc == 1


# --- P2-4: unreadable history stops before generation -------------------------

@pytest.mark.parametrize("manifest_text", [
    "{not json",        # corrupt manifest body
    "__missing__",      # manifest file absent although the index lists it
])
def test_unreadable_history_stops_before_generation(tmp_path, monkeypatch,
                                                    manifest_text):
    """The reviewer's counterexample: a normal attempts_index with a
    registered attempt whose manifest cannot be read used to silently skip
    the history — six fresh calls, exit 0, prior_wall_s_in_root=0."""
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "hist"
    with pytest.raises(SystemExit, match="history_incomplete"):
        _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir, prereg,
                           manifest_text=manifest_text)
    # stopped BEFORE any generation call and BEFORE allocating an attempt
    assert not list(out_dir.glob("attempt_002*"))


@pytest.mark.parametrize("manifest_text", [
    "{}",  # parses, but neither rows nor cost — read as zero history
    pytest.param(json.dumps({
        "rows": [{"demo": PILOT[0], "status": "generation_failed"}]}),
        id="rows_without_total_wall_s"),
    pytest.param(json.dumps({"total_wall_s": 100.0}), id="no_rows"),
    pytest.param(json.dumps({"rows": "not-a-list", "total_wall_s": 1.0}),
                 id="rows_not_list"),
    pytest.param(json.dumps({"rows": [{"demo": PILOT[0]}],
                             "total_wall_s": 1.0}), id="row_status_missing"),
    pytest.param(json.dumps({"rows": [{"status": "ok"}],
                             "total_wall_s": 1.0}), id="row_demo_missing"),
    # json.loads ACCEPTS the bare NaN literal — a NaN cost is not a number
    '{"rows": [], "total_wall_s": NaN}',
    # bool is an int subclass — True must not pass as a finite cost
    '{"rows": [], "total_wall_s": true}',
    '"a string manifest"',
])
def test_structurally_deficient_history_stops_before_generation(
        tmp_path, monkeypatch, manifest_text):
    """Round-5 reviewer counterexample: history_errors used to cover only
    read/parse exceptions — a manifest that PARSES but lacks the fields the
    gates consume (``{}``, or rows without ``total_wall_s``) fell through
    the ``or 0.0``/``, []`` defaults as zero invocations and zero cost, so
    a normal index still launched all six generation calls with exit 0."""
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "hist5"
    with pytest.raises(SystemExit, match="history_incomplete"):
        _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir, prereg,
                           manifest_text=manifest_text)
    # stopped BEFORE any generation call and BEFORE allocating an attempt
    assert not list(out_dir.glob("attempt_002*"))


def test_all_skipped_history_is_valid_not_deficient(tmp_path, monkeypatch):
    """Not over-tightened: a genuine all-skipped prior run records
    ``total_wall_s: 0.0`` with skipped rows — zero cost BY EVIDENCE, not by
    a missing field, so the retry path keeps working."""
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "zero"
    prior = [{"demo": PILOT[0], "status": "skipped_retry_cap",
              "wall_s": 0.0}]
    rc, calls = _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir,
                                   prereg, prior_rows=prior, prior_wall=0.0)
    # PILOT[0] was never actually invoked (only skipped) -> eligible now
    assert calls == PILOT
    m = json.loads(sorted(out_dir.glob("attempt_*/generation_manifest.json"))
                   [-1].read_text())
    assert m["prior_wall_s_in_root"] == 0.0


def test_readable_history_still_generates(tmp_path, monkeypatch):
    """The gate is not over-tightened: a readable prior manifest keeps the
    normal retry path (one prior failure -> one justified retry call)."""
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "ok"
    prior = [{"demo": PILOT[0], "status": "generation_failed"}]
    rc, calls = _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir,
                                   prereg, prior_rows=prior)
    assert calls[:1] == [PILOT[0]]
    assert calls == PILOT
