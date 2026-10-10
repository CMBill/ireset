"""A2 closure (C0/C1) contracts — closure plan §10.1 counterexamples.

* anchors/freeze/artifacts tampered or inputs drifted -> the new gap is NOT
  a normal stop-line registration (registration blocked, non-zero exit);
* only the registered sub12 gap -> registration completes while the
  baseline verdict stays ``incomplete`` and training stays ``not_ready``;
* registration_status / exit 0 can never be read as a source pass;
* shrunken denominator, dropped rows or dropped attempts are rejected;
* second invocation writes a new decision attempt; prior evidence bytes
  stay unchanged.
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
from ireset.eval import a2_closure as a2c  # noqa: E402

PILOT = ["subA_largebox_001", "subB_largebox_002", "subC_largebox_003",
         "subD_largebox_004", "subE_largebox_005", "subF_largebox_006"]
KNOWN = "subF_largebox_006"
MEASURABLE = PILOT[:-1]
GATES = {
    "g1_identity_and_availability": {"per_demo": {d: d != KNOWN for d in PILOT},
                                     "passed": False},
    "g2_task_mpjpe": {"per_demo": {d: d != KNOWN for d in PILOT},
                      "passed": False},
    "g3_penetration_and_feet": {
        "per_demo": {d: {"passed": False, "penetration": False,
                         "foot": {"l": True, "r": True}} for d in PILOT},
        "passed": False},
    "g4_hand_improvement": {
        "coverage_gain_pp_by_demo": [-25.0, None, None, None, -17.14, None],
        "gap_rel_drop_by_demo": [-0.17, None, None, None, 0.44, None],
        "n_coverage_gain_demos": 0, "n_gap_drop_demos": 1,
        "primary_metric": "both_valid_coverage_gain>=10pp",
        "passed": False},
    "g5_degradation": {
        "per_demo": {d: {"passed": False,
                         "degradation": {"hand_r": {"p_drift_m": 0.1,
                                                    "u0_drift_m": 0.2}}}
                     for d in MEASURABLE},
        "margin_m": 0.02, "note": "", "passed": False},
}
KNOWN_GAPS = [f"{KNOWN}:U0_invalid", f"{KNOWN}:binding_not_verified",
              f"admission:{KNOWN}:binding_recheck_failed"]


def _report(*, verdict="incomplete", gaps=None, pilot=None) -> dict:
    pilot = PILOT if pilot is None else pilot
    return copy.deepcopy({
        "stage": "A2", "timestamp": "2026-10-10T00:00:00",
        "pilot": pilot, "p_revision": "1352",
        "comparison_attempt": "attempt_001",
        "u1": "not_applicable",
        "metrics": {
            "P": {d: {"demo": d, "valid": True, "task_mpjpe_p95_m": 0.05,
                      "penetration_max_m": 0.001,
                      "both": {"denominator_frames": 30,
                               "valid_coverage": 0.5,
                               "gap_series_p95_m": 0.02}} for d in pilot},
            "U0": {d: ({"demo": d, "valid": False,
                        "error": "no OK U0 artifact in any attempt"}
                       if d == KNOWN else
                       {"demo": d, "valid": True,
                        "task_mpjpe_p95_m": 0.06, "penetration_max_m": 0.02,
                        "both": {"denominator_frames": 30,
                                 "valid_coverage": 0.25,
                                 "gap_series_p95_m": 0.03}})
                   for d in pilot},
        },
        "binding": {d: d != KNOWN for d in pilot},
        "admission": {"passed": False,
                      "u0_coverage": {"missing": [KNOWN],
                                      "by_demo": {d: "attempt_001"
                                                  for d in pilot
                                                  if d != KNOWN}}},
        "verdict_record": {"complete_evidence": not (gaps or KNOWN_GAPS),
                           "verdict": verdict, "gates": GATES,
                           "evidence_gaps": KNOWN_GAPS if gaps is None else gaps},
        "verdict": verdict,
        "cost_s": 100.0,
    })


# --- unit: gap split, invariants, consumption -------------------------------

def test_split_gaps_separates_known_sub12_from_anything_new():
    gaps = KNOWN_GAPS + ["subB_largebox_002:U0_invalid",
                         "admission:frozen_assets_drifted"]
    s = a2c.split_gaps(gaps, known_missing=(KNOWN,))
    assert set(s["known"]) == set(KNOWN_GAPS)
    assert s["unexpected"] == ["subB_largebox_002:U0_invalid",
                               "admission:frozen_assets_drifted"]


def test_report_invariants_reject_shrunken_denominator_and_dropped_rows():
    ok = a2c.check_report_invariants(_report(), PILOT)
    assert ok["passed"]
    shrunk = a2c.check_report_invariants(_report(pilot=PILOT[:-1]), PILOT)
    assert any(v.startswith("pilot_mismatch") for v in shrunk["violations"])
    assert any(v.startswith("rows_missing_or_extra") for v in shrunk["violations"])
    dropped = _report()
    dropped["metrics"]["U0"].pop("subC_largebox_003")
    assert any("rows_missing_or_extra:U0" in v for v in
               a2c.check_report_invariants(dropped, PILOT)["violations"])
    binding_cut = _report()
    binding_cut["binding"].pop(PILOT[0])
    assert any(v.startswith("binding_keys_mismatch") for v in
               a2c.check_report_invariants(binding_cut, PILOT)["violations"])


def test_report_invariants_reject_verdict_evidence_disagreement():
    r = _report()
    r["verdict"] = "local_preferred"
    r["verdict_record"]["verdict"] = "local_preferred"
    v = a2c.check_report_invariants(r, PILOT)
    assert "local_preferred_without_complete_evidence" in v["violations"]
    bogus = _report(verdict="deferred")
    assert any(v.startswith("verdict_out_of_vocabulary") for v in
               a2c.check_report_invariants(bogus, PILOT)["violations"])


def test_consumption_map_detects_changed_missing_set_or_consumer():
    idx = [{"attempt": "attempt_001"}, {"attempt": "attempt_002"}]
    cov = {"by_demo": {d: {"attempt": "attempt_001"} for d in MEASURABLE},
           "missing": [KNOWN]}
    ok = a2c.check_consumption(idx, cov, PILOT, known_missing=(KNOWN,))
    assert ok["passed"] and ok["attempts_retained"] == ["attempt_001",
                                                        "attempt_002"]
    cov2 = {"by_demo": {**{d: {"attempt": "attempt_001"}
                           for d in MEASURABLE[:-1]},
                        MEASURABLE[-1]: {"attempt": "attempt_002"}},
            "missing": [KNOWN]}
    assert any(v.startswith("consumption_changed") for v in
               a2c.check_consumption(idx, cov2, PILOT,
                                     known_missing=(KNOWN,))["violations"])
    cov3 = {"by_demo": {d: {"attempt": "attempt_001"} for d in PILOT},
            "missing": []}
    assert any(v.startswith("missing_set_changed") for v in
               a2c.check_consumption(idx, cov3, PILOT,
                                     known_missing=(KNOWN,))["violations"])


def test_anchor_verification_measures_bytes_not_existence(tmp_path):
    f = tmp_path / "anchor.bin"
    f.write_bytes(b"hello")
    decl = {"k": integ.sha256_file(f)}
    assert a2c.verify_plan_anchors({"k": f}, decl, integ.sha256_file)["passed"]
    f.write_bytes(b"tampered")           # declared hash unchanged, file changed
    bad = a2c.verify_plan_anchors({"k": f}, decl, integ.sha256_file)
    assert not bad["passed"] and not bad["anchors"]["k"]["matches"]
    gone = a2c.verify_plan_anchors({"k": tmp_path / "nope"}, decl,
                                   integ.sha256_file)
    assert gone["anchors"]["k"]["reason"] == "missing"


# --- unit: fresh-vs-authoritative comparison --------------------------------

def test_fresh_comparison_match_ignores_only_run_scoped_fields():
    a = _report()
    f = _report()
    f["timestamp"] = "2026-10-11T09:00:00"
    f["comparison_attempt"] = "attempt_007"
    f["cost_s"] = 137.2
    assert a2c.compare_fresh_to_authoritative(a, f)["matches"]
    f["metrics"]["U0"][PILOT[0]]["task_mpjpe_p95_m"] = 0.07
    cmp1 = a2c.compare_fresh_to_authoritative(a, f)
    assert not cmp1["matches"]
    assert "metrics.U0.subA_largebox_001.task_mpjpe_p95_m" in cmp1["differences"]
    f2 = _report()
    f2["verdict_record"]["gates"]["g2_task_mpjpe"]["per_demo"][PILOT[1]] = False
    assert ("verdict_record.gates.g2_task_mpjpe.per_demo."
            "subB_largebox_002") in a2c.compare_fresh_to_authoritative(
                a, f2)["differences"]
    f3 = _report()
    f3["admission"]["passed"] = True
    assert not a2c.compare_fresh_to_authoritative(a, f3)["matches"]


# --- unit: closure record keeps the three statuses apart --------------------

def _verification(**over):
    v = {"anchors_passed": True, "fresh_matches": True,
         "known_gaps": KNOWN_GAPS, "unexpected_gaps": [],
         "report_violations": [], "consumption_violations": []}
    v.update(over)
    return v


def _baseline():
    return {"report_path": "sel.json", "report_sha256": "x",
            "verdict": "incomplete", "pilot": PILOT,
            "measurable": MEASURABLE, "missing": [KNOWN]}


def test_normal_registration_keeps_incomplete_and_not_ready():
    rec = a2c.build_closure_record(
        baseline=_baseline(),
        verification=_verification(),
        quality_evidence=a2c.summarize_quality_evidence(_report()),
        cost_basis={"failed_runs": [{"attempt": "attempt_001",
                                     "status": "generation_failed"}]})
    assert rec["registration_status"]["complete"] is True
    assert rec["baseline"]["verdict"] == "incomplete"
    assert rec["baseline_verdict_unchanged"] is True
    assert rec["training"]["status"] == "not_ready"
    assert rec["disposition"]["action"] == "stop_u0_expansion"
    md = a2c.render_markdown(rec)
    assert "baseline_verdict = incomplete" in md
    assert "training = not_ready" in md


@pytest.mark.parametrize("over,expect_reason", [
    ({"unexpected_gaps": ["admission:frozen_assets_drifted"]},
     "unexpected_evidence_gap"),
    ({"fresh_matches": False}, "fresh_comparison_differs"),
    ({"anchors_passed": False}, "plan_anchor_mismatch"),
    ({"report_violations": ["pilot_mismatch:..."]}, "pilot_mismatch:..."),
    ({"consumption_violations": ["consumption_changed:subA:attempt_002"]},
     "consumption_changed:subA:attempt_002"),
])
def test_evidence_damage_blocks_registration(over, expect_reason):
    rec = a2c.build_closure_record(
        baseline=_baseline(), verification=_verification(**over),
        quality_evidence=a2c.summarize_quality_evidence(_report()),
        cost_basis={})
    assert rec["registration_status"]["complete"] is False
    assert expect_reason in rec["registration_status"]["blocking_reasons"]


def test_registration_complete_is_not_a_source_pass():
    rec = a2c.build_closure_record(
        baseline=_baseline(), verification=_verification(),
        quality_evidence=a2c.summarize_quality_evidence(_report()),
        cost_basis={})
    # no field anywhere downgrades the baseline verdict or opens training
    assert rec["baseline"]["verdict"] != "local_preferred"
    assert rec["training"]["status"] == "not_ready"
    assert rec["quality_evidence"]["formal_verdict"] == "incomplete"
    assert set(rec["quality_evidence"]["measurable_demos"]) == set(MEASURABLE)
    assert rec["diagnosis"]["status"] == "not_run"
    assert rec["recovery"]["status"] == "not_registered"


# --- CLI end-to-end with a fake fresh-comparison runner ---------------------

UPSTREAM_LOCK = Path(__file__).resolve().parents[1] / "derived" / "upstream" \
    / "holosoma_env" / "uv.lock"


def _write_freeze(root: Path) -> Path:
    freeze = root / "freeze"
    freeze.mkdir(exist_ok=True)
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [{"demo": d} for d in PILOT]}))
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit":
                      "bccd4d7451640a2800ddc77e469d911a84f91994"}}))
    (freeze / "pairing_source_boxes.npz").write_bytes(b"npz")
    (freeze / "summary.json").write_text("{}")
    return freeze


def _write_u0(root: Path, npz_dir: Path) -> Path:
    u0 = root / "u0"
    npz_dir.mkdir(parents=True, exist_ok=True)
    for d in PILOT:
        (npz_dir / f"{d}_original.npz").write_bytes(b"pub")
    att1 = u0 / "attempt_001"
    att1.mkdir(parents=True, exist_ok=True)
    rows1 = []
    for d in MEASURABLE:
        (att1 / f"{d}_original.npz").write_bytes(b"raw")
        (att1 / f"{d}_original_converted.npz").write_bytes(b"conv")
        rows1.append({"demo": d, "status": "ok",
                      "binding": {"passed": True}})
    rows1.append({"demo": KNOWN, "status": "generation_failed",
                  "wall_s": 143.9, "error": "CVXPY solve failed: infeasible"})
    (att1 / "generation_manifest.json").write_text(json.dumps(
        {"rows": rows1, "identity": {}, "input_sha256": {}}))
    att2 = u0 / "attempt_002"
    att2.mkdir(exist_ok=True)
    (att2 / "generation_manifest.json").write_text(json.dumps(
        {"rows": [{"demo": KNOWN, "status": "generation_failed",
                   "wall_s": 144.6,
                   "error": "CVXPY solve failed: infeasible"}],
         "identity": {}, "input_sha256": {}}))
    (u0 / integ.ATTEMPTS_INDEX).write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001"},
                      {"attempt": "attempt_002"}]}))
    return u0


class _Args:
    def __init__(self, freeze, selection, u0, out):
        self.freeze, self.selection = freeze, selection
        self.u0_dir, self.out = u0, out
        self.diagnosis = None
        self.recovery_report = None


def _run_cli(tmp_path, monkeypatch, *, fresh_mutator=None,
             tamper_selection=None):
    import omniretarget_a2_closure as closer_mod
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path / "derived")
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir(exist_ok=True)
    for d in PILOT:
        (pt_dir / f"{d}.pt").write_bytes(b"pt")
    freeze = _write_freeze(tmp_path)
    selection = tmp_path / "teacher_selection.json"
    selection.write_text(json.dumps(_report()))
    declared = {
        "authoritative_selection": integ.sha256_file(selection),
        "a1_frozen_lists": integ.sha256_file(freeze / "frozen_lists.json"),
        "a1_assets": integ.sha256_file(freeze / "assets.json"),
        "a1_pairing_source_boxes":
            integ.sha256_file(freeze / "pairing_source_boxes.npz"),
        "upstream_uv_lock": integ.sha256_file(UPSTREAM_LOCK),
    }
    if tamper_selection is not None:
        tamper_selection(selection)          # AFTER the anchors were pinned
    u0 = _write_u0(tmp_path, tmp_path / "npz")
    monkeypatch.setattr(closer_mod, "TEACHER_TRAJ_DIR", tmp_path / "npz")
    monkeypatch.setattr(closer_mod, "consumed_data_path",
                        lambda root: (pt_dir, False))
    monkeypatch.setattr(closer_mod, "PLAN_ANCHORS", declared)

    def fake_runner(freeze_arg, u0_root, baseline_dir):
        baseline_dir.mkdir(parents=True, exist_ok=True)
        fresh = _report()
        fresh["timestamp"] = "2026-10-11T08:00:00"
        fresh["comparison_attempt"] = "attempt_042"
        fresh["cost_s"] = 12.5
        if fresh_mutator is not None:
            fresh_mutator(fresh)
        out = baseline_dir / "cmp" / "attempt_042" / "teacher_selection.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fresh))
        return out

    out = tmp_path / "derived" / "run"
    rc = closer_mod.run_closure(_Args(freeze, selection, u0, out),
                                comparison_runner=fake_runner)
    return rc, out


def test_cli_normal_registration_exit0_and_reports_written(tmp_path,
                                                           monkeypatch):
    rc, out = _run_cli(tmp_path, monkeypatch)
    assert rc == 0
    rec = json.loads((out / "decisions" / "attempt_001"
                      / "a2_closure.json").read_text())
    assert rec["registration_status"]["complete"] is True
    assert rec["baseline"]["verdict"] == "incomplete"
    assert rec["training"]["status"] == "not_ready"
    be = json.loads((out / "baseline" / "baseline_evidence.json").read_text())
    assert be["passed"] is True
    assert be["known_gaps"] == KNOWN_GAPS
    assert be["unexpected_gaps"] == []
    ev = json.loads((out / "baseline" / "evidence_files.json").read_text())
    assert ev["u0_evidence"]["sub12_teacher"]["missing"] is True
    assert ev["consumed_data_path"]["recorded_in_manifest"] is False
    assert set(ev["published_teachers"]) == set(PILOT)
    md = (out / "decisions" / "attempt_001" / "a2_closure.md").read_text()
    assert "baseline_verdict = incomplete" in md
    assert "training = not_ready" in md
    idx = json.loads((out / "decisions" / integ.ATTEMPTS_INDEX).read_text())
    assert idx["attempts"][0]["decision"] == "stop_u0_expansion"


def test_cli_unexpected_gap_blocks_registration_nonzero(tmp_path, monkeypatch):
    def mutate(fresh):
        fresh["verdict_record"]["evidence_gaps"] = KNOWN_GAPS + [
            "admission:input_hash_mismatch_vs_freeze"]
    rc, out = _run_cli(tmp_path, monkeypatch, fresh_mutator=mutate)
    assert rc == 2
    rec = json.loads((out / "decisions" / "attempt_001"
                      / "a2_closure.json").read_text())
    assert rec["registration_status"]["complete"] is False
    assert "unexpected_evidence_gap" in rec["registration_status"][
        "blocking_reasons"]
    be = json.loads((out / "baseline" / "baseline_evidence.json").read_text())
    assert be["passed"] is False
    assert "admission:input_hash_mismatch_vs_freeze" in be["unexpected_gaps"]


def test_cli_selection_tampered_after_anchor_pinning_blocks(tmp_path,
                                                            monkeypatch):
    def tamper(sel):
        r = json.loads(sel.read_text())
        r["metrics"]["P"][PILOT[0]]["task_mpjpe_p95_m"] = 0.5
        sel.write_text(json.dumps(r))
    rc, out = _run_cli(tmp_path, monkeypatch, tamper_selection=tamper)
    assert rc == 2
    rec = json.loads((out / "decisions" / "attempt_001"
                      / "a2_closure.json").read_text())
    assert rec["registration_status"]["complete"] is False
    assert "plan_anchor_mismatch" in rec["registration_status"][
        "blocking_reasons"]


def test_cli_second_call_new_attempt_old_bytes_unchanged(tmp_path, monkeypatch):
    rc1, out = _run_cli(tmp_path, monkeypatch)
    assert rc1 == 0
    first = out / "decisions" / "attempt_001" / "a2_closure.json"
    first_bytes = first.read_bytes()
    rc2, _ = _run_cli(tmp_path, monkeypatch)   # same --out root
    assert rc2 == 0
    assert (out / "decisions" / "attempt_002" / "a2_closure.json").is_file()
    assert first.read_bytes() == first_bytes
    idx = json.loads((out / "decisions" / integ.ATTEMPTS_INDEX).read_text())
    assert [a["attempt"] for a in idx["attempts"]] == ["attempt_001",
                                                       "attempt_002"]
