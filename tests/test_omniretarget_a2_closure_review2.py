"""A2 closure review round 2 (2026-10-10) — counterexamples for the 13
independent-review gaps.

* C0 snapshots are attempt-scoped and immutable (fix 1);
* C2 merge verifies the evidence CHAIN, not just current hashes (fix 6);
* C1 inherits identity/checkout failures (fix 7) and the training-unmet
  reasons match the merged recovery state (fix 13);
* consumption checks the manifest ROWS, not just directory names (fix 9);
* repo identity pins tracked-diff CONTENT (fix 10);
* D0 runs its full pre-admission and confirms the control BEFORE any sub12
  subprocess (fix 4);
* two-run consistency covers canonical vectors / column mapping / artifact
  hashes / problem shape, and G0 verifies the record (fix 5);
* generation enforces the per-demo retry cap and the UR budget across
  invocations (fix 11).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import a2_integrity as integ  # noqa: E402
from ireset.eval import a2_closure as a2c  # noqa: E402
from ireset.eval import teacher_comparison as tcmp  # noqa: E402
from ireset.eval.teacher_comparison import (  # noqa: E402
    DemoMetrics, SideFootMetrics, SideHandMetrics,
)

PILOT = ["subA_largebox_001", "subB_largebox_002", "subC_largebox_003",
         "subD_largebox_004", "subE_largebox_005", "subF_largebox_006"]
KNOWN = "subF_largebox_006"
MEASURABLE = PILOT[:-1]
KNOWN_GAPS = [f"{KNOWN}:U0_invalid", f"{KNOWN}:binding_not_verified",
              f"admission:{KNOWN}:binding_recheck_failed"]
GATES = {
    "g1_identity_and_availability": {"per_demo": {d: d != KNOWN
                                                  for d in PILOT},
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


def _report(*, verdict="incomplete") -> dict:
    return {
        "stage": "A2", "timestamp": "2026-10-10T00:00:00",
        "pilot": PILOT, "p_revision": "1352",
        "comparison_attempt": "attempt_001",
        "u1": "not_applicable",
        "metrics": {
            "P": {d: {"demo": d, "valid": True} for d in PILOT},
            "U0": {d: ({"demo": d, "valid": False}
                       if d == KNOWN else {"demo": d, "valid": True})
                   for d in PILOT},
        },
        "binding": {d: d != KNOWN for d in PILOT},
        "admission": {"passed": False,
                      "u0_coverage": {"missing": [KNOWN],
                                      "by_demo": {d: "attempt_001"
                                                  for d in MEASURABLE}}},
        "verdict_record": {"complete_evidence": False, "verdict": verdict,
                           "gates": GATES, "evidence_gaps": KNOWN_GAPS},
        "verdict": verdict, "cost_s": 100.0,
    }


# --- fix 7: C1 inherits identity/checkout failures ---------------------------

def _verification(**over):
    v = {"anchors_passed": True, "fresh_matches": True,
         "known_gaps": KNOWN_GAPS, "unexpected_gaps": [],
         "report_violations": [], "consumption_violations": [],
         "identity_stable": True, "checkout_passed": True}
    v.update(over)
    return v


def _baseline():
    return {"report_path": "sel.json", "report_sha256": "x",
            "verdict": "incomplete", "pilot": PILOT,
            "measurable": MEASURABLE, "missing": [KNOWN]}


def _quality():
    return {"formal_verdict": "incomplete",
            "formal_verdict_note": "", "measurable_demos": MEASURABLE,
            "g2_task_in_bound": {}, "g3_per_demo": {}, "g4": {},
            "g5_per_demo": {}, "gates_passed": {}}


def test_c1_inherits_identity_and_checkout_failures():
    for over, reason in (({"identity_stable": False},
                          "identity_unstable_during_run"),
                         ({"checkout_passed": False},
                          "upstream_checkout_mismatch")):
        rec = a2c.build_closure_record(
            baseline=_baseline(), verification=_verification(**over),
            quality_evidence=_quality(), cost_basis={})
        assert rec["registration_status"]["complete"] is False, over
        assert reason in rec["registration_status"]["blocking_reasons"]
    ok = a2c.build_closure_record(
        baseline=_baseline(), verification=_verification(),
        quality_evidence=_quality(), cost_basis={})
    assert ok["registration_status"]["complete"] is True


# --- fix 13: training-unmet reasons match the merged recovery -----------------

def test_training_unmet_matches_merged_recovery_state():
    rec = a2c.build_closure_record(
        baseline=_baseline(), verification=_verification(),
        quality_evidence=_quality(), cost_basis={},
        recovery={"status": "no_quality_gain"})
    unmet = "；".join(rec["training"]["unmet"])
    assert "local_preferred" in unmet and "W1" in unmet
    # the pre-recovery reason ("no source experiment has produced a
    # complete ... verdict") is now FALSE and must not be listed
    assert "no source experiment has produced" not in unmet
    default = a2c.build_closure_record(
        baseline=_baseline(), verification=_verification(),
        quality_evidence=_quality(), cost_basis={})
    assert any("no source experiment has produced" in u
               for u in default["training"]["unmet"])


# --- fix 9: consumption verifies the manifest ROWS ---------------------------

def test_consumption_detects_dropped_failure_rows():
    idx = [{"attempt": "attempt_001", "scope": PILOT},
           {"attempt": "attempt_002", "scope": [KNOWN]}]
    cov = {"by_demo": {d: {"attempt": "attempt_001"} for d in MEASURABLE},
           "missing": [KNOWN]}
    rows = {"attempt_001": [{"demo": d, "status": "ok"} for d in PILOT[:-1]]
            + [{"demo": KNOWN, "status": "generation_failed"}],
            "attempt_002": [{"demo": KNOWN, "status": "generation_failed"}]}
    ok = a2c.check_consumption(idx, cov, PILOT, known_missing=(KNOWN,),
                               attempt_scopes={a["attempt"]: a["scope"]
                                               for a in idx},
                               manifest_rows=rows)
    assert ok["passed"]
    # the second registered failure row is deleted -> the cost basis
    # silently shrinks to one run; the scope promise catches it
    dropped = copy.deepcopy(rows)
    dropped["attempt_002"] = []
    bad = a2c.check_consumption(idx, cov, PILOT, known_missing=(KNOWN,),
                                attempt_scopes={a["attempt"]: a["scope"]
                                                for a in idx},
                                manifest_rows=dropped)
    assert not bad["passed"]
    assert any(v.startswith("failure_record_dropped:attempt_002")
               for v in bad["violations"])
    # both rows gone -> the stop line's cost basis is empty
    emptied = copy.deepcopy(rows)
    emptied["attempt_001"] = [r for r in rows["attempt_001"]
                              if r["demo"] != KNOWN]
    emptied["attempt_002"] = []
    worse = a2c.check_consumption(idx, cov, PILOT, known_missing=(KNOWN,),
                                  attempt_scopes={a["attempt"]: a["scope"]
                                                  for a in idx},
                                  manifest_rows=emptied)
    assert any(v.startswith("cost_basis_empty") for v in worse["violations"])


# --- fix 10: repo identity pins tracked-diff CONTENT --------------------------

def test_repo_identity_pins_tracked_diff_content(tmp_path, monkeypatch):
    import omniretarget_a2_closure as closer
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        subprocess.run(["git", "-C", str(repo), *a], check=True,
                       capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (repo / "tracked.txt").write_text("v1")
    git("add", "-A"); git("commit", "-qm", "init")
    monkeypatch.setattr(closer, "PROJECT_ROOT", repo)
    id1 = closer.measure_repo_identity()
    assert id1["status_lines"] == []
    (repo / "tracked.txt").write_text("v2")
    id2 = closer.measure_repo_identity()
    assert id2["status_lines"] == ["M tracked.txt"]
    (repo / "tracked.txt").write_text("v3 — different content")  # same "M"
    id3 = closer.measure_repo_identity()
    # identical status line, DIFFERENT tracked content: the pre-fix
    # identity compared equal on the status list alone
    assert id3["status_lines"] == id2["status_lines"]
    assert id3["tracked_diff_sha256"] != id2["tracked_diff_sha256"]
    id4 = closer.measure_repo_identity()          # content pinned: stable
    assert id4 == id3


# --- fix 5: two-run consistency over the FULL captured evidence ---------------

def _ctx_npz(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    np.savez(path, q_locked=rng.normal(size=43),
             q_a_n_last=rng.normal(size=29), q_t_last=rng.normal(size=43),
             target_laplacian=rng.normal(size=30),
             obj_pts_local=rng.normal(size=(10, 3)),
             q_a_indices=np.arange(-7, 29))


def _canonical_npz(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    np.savez(path, c=rng.normal(size=798), b=rng.normal(size=949))


def _canonical_failure(*, rho=0.2486138173835237, soc_t=0.2, n_kept=66):
    """The genuine child shape: category ALWAYS consistent with the facts
    (round-4 review fix 1 re-derives it through classify_phase_one) and
    ``max_violation`` present whenever status is optimal."""
    return {"failing_frame_idx": 171, "ordinal_in_frame": 5,
            "phase_one": {
                "category": "supports_step03_candidate",
                "rho": rho, "status": "optimal", "max_violation": 3.5e-17,
                "soc_t": soc_t, "dqa_shape": [36],
                "n_kept_constraints": n_kept, "n_soc_constraints": 1,
                "soc_t_matches_step_size": True}}


def _write_full_snapshot(path: Path, seed: int = 7, *, failure=None):
    """Complete snapshot: the REQUIRED artifact set (context arrays incl.
    q_a_indices, canonical b AND c, P/A dumps) — round-3 review fix 3 —
    PLUS the on-disk run-fact jsons (failure_meta.json / child_summary.json
    carrying the same failure dict) the round-4 disk-reconciliation leg
    compares the record against."""
    path.mkdir(parents=True, exist_ok=True)
    _ctx_npz(path / "failure_context.npz", seed)
    _canonical_npz(path / "failure_canonical.npz", seed + 1)
    rng = np.random.default_rng(seed + 2)
    np.savez(path / "failure_canonical_P.npz",
             data=rng.normal(size=37), indices=np.arange(5),
             indptr=np.arange(6), shape=[798, 798])
    np.savez(path / "failure_canonical_A.npz",
             data=rng.normal(size=41), indices=np.arange(5),
             indptr=np.arange(6), shape=[949, 798])
    if failure is None:
        failure = _canonical_failure()
    (path / "failure_meta.json").write_text(
        json.dumps({"identity": {}, "failure": failure}))
    (path / "child_summary.json").write_text(
        json.dumps({"failure": failure, "outcome": "x"}))


def _diag_run(snapshot: Path, *, rho=0.2486138173835237, soc_t=0.2,
              n_kept=66):
    return {"label": "r", "snapshot": str(snapshot), "timed_out": False,
            "launch_error": None,
            "summary": {"outcome": "RuntimeError: CVXPY solve failed",
                        "failure": _canonical_failure(
                            rho=rho, soc_t=soc_t, n_kept=n_kept)}}


@pytest.fixture()
def two_snapshots(tmp_path):
    sa, sb = tmp_path / "snap_a", tmp_path / "snap_b"
    _write_full_snapshot(sa)
    _write_full_snapshot(sb)
    return sa, sb


def test_decide_catches_tampered_canonical_or_mapping(two_snapshots):
    import diagnose_omniretarget_sub12 as diag
    a, b = two_snapshots
    ok = diag.decide_diagnosis({"equal": True},
                               [_diag_run(a), _diag_run(b)])
    assert ok["status"] == "supports_step03_candidate"
    # canonical b tampered in the second run
    arr = dict(np.load(b / "failure_canonical.npz"))
    arr["b"] = arr["b"] + 1e-9
    np.savez(b / "failure_canonical.npz", **arr)
    bad = diag.decide_diagnosis({"equal": True},
                                [_diag_run(a), _diag_run(b)])
    assert bad["status"] == "nondeterministic"
    assert not bad["canonical_vectors_equal"]["equal"]
    # q_a_indices (variable-column mapping) tampered
    _canonical_npz(b / "failure_canonical.npz", 11)
    ctx = dict(np.load(b / "failure_context.npz"))
    ctx["q_a_indices"] = ctx["q_a_indices"] + 1
    np.savez(b / "failure_context.npz", **ctx)
    bad2 = diag.decide_diagnosis({"equal": True},
                                 [_diag_run(a), _diag_run(b)])
    assert bad2["status"] == "nondeterministic"
    # problem shape differs (SOC t drifts between the two runs)
    _ctx_npz(b / "failure_context.npz", 7)
    bad3 = diag.decide_diagnosis({"equal": True},
                                 [_diag_run(a), _diag_run(b, soc_t=0.3)])
    assert bad3["status"] == "nondeterministic"
    assert not bad3["problem_shape_equal"]


def _saved_run(r: dict) -> dict:
    """The shape diagnose() writes into diagnosis_run.json: the failure
    dict lives at the TOP level of each run record, not under "summary"."""
    s = r["summary"]
    return {"label": r["label"], "snapshot": r["snapshot"],
            "timed_out": r["timed_out"], "launch_error": r["launch_error"],
            "failure": s["failure"], "outcome": s["outcome"],
            "frames_completed": 171}


def test_verify_diagnosis_record_checks_backing_and_parent(two_snapshots,
                                                            tmp_path):
    import diagnose_omniretarget_sub12 as diag
    a, b = two_snapshots
    eval_runs = [_diag_run(a), _diag_run(b)]
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
    # parent snapshot drifted after the diagnosis registered its hash
    baseline.write_text("{} ")
    drifted = diag.verify_diagnosis_record(good)
    assert not drifted["verified"]
    assert "parent_baseline_drifted" in drifted["problems"]
    baseline.write_text("{}")
    # the recorded verdict is not backed: rho outside the candidate window
    bad_verdict = copy.deepcopy(verdict)
    bad_verdict["rho_run1"] = 0.45
    rec2 = copy.deepcopy(good)
    rec2["verdict"] = bad_verdict
    v2 = diag.verify_diagnosis_record(rec2)
    assert not v2["verified"]
    assert "rho_outside_candidate_window" in v2["problems"]
    # evidence fields missing entirely (status string alone is not enough)
    rec3 = copy.deepcopy(good)
    rec3["verdict"] = {"status": "supports_step03_candidate"}
    v3 = diag.verify_diagnosis_record(rec3)
    assert not v3["verified"]
    assert any(p.startswith(("same_frame", "rho")) for p in v3["problems"])


# --- fix 4: D0 pre-admission and control-first ordering -----------------------

def _d0_args(tmp_path, baseline_evidence):
    class _A:
        pass
    a = _A()
    a.baseline_evidence = baseline_evidence
    a.out = tmp_path / "diag"
    a.env_python = tmp_path / "envpy"
    a.demo = KNOWN
    a.control_demo = "subA_largebox_001"
    a.data_path = tmp_path / "pts"
    a.run_timeout_s = 30
    return a


def _write_baseline(tmp_path, *, passed=True):
    sel = tmp_path / "teacher_selection.json"
    sel.write_text(json.dumps(_report()))
    freeze = tmp_path / "freeze"
    freeze.mkdir(exist_ok=True)
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit": "bccd4d7"}}))
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [{"demo": d} for d in PILOT]}))
    u0 = tmp_path / "u0"
    (u0 / "attempt_001").mkdir(parents=True, exist_ok=True)
    (u0 / integ.ATTEMPTS_INDEX).write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001"}]}))
    (u0 / "attempt_001" / "generation_manifest.json").write_text(
        json.dumps({"rows": [], "identity": {"env": {
            "env_uv_lock_sha256": "locksha",
            "env_mujoco_version": "3.10.0"}}}))
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    be = {
        "passed": passed,
        "identity": {"stable_during_run": True},
        "plan_anchors": {"anchors": {"authoritative_selection": {
            "measured_sha256": integ.sha256_file(sel)}}},
    }
    (snap / "baseline_evidence.json").write_text(json.dumps(be))
    (snap / "evidence_files.json").write_text(json.dumps({
        "authoritative_selection": {"path": str(sel),
                                    "sha256": integ.sha256_file(sel)},
        "a1_freeze": {"frozen_lists.json": {
            "path": str(freeze / "frozen_lists.json")}},
        "u0_evidence": {"attempts_index": {"path": str(
            u0 / integ.ATTEMPTS_INDEX)}},
    }))
    return snap / "baseline_evidence.json"


def _patch_d0(monkeypatch, tmp_path):
    import diagnose_omniretarget_sub12 as diag
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)
    monkeypatch.setattr(diag, "RETARGETER_FILE", tmp_path / "retargeter.py")
    (tmp_path / "retargeter.py").write_text("# pinned")
    monkeypatch.setattr(diag, "CHILD_DRIVER", tmp_path / "child.py")
    (tmp_path / "child.py").write_text("# child")
    monkeypatch.setattr(integ, "verify_checkout_against_freeze",
                        lambda a: {"passed": True})
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda a: {"passed": True})
    monkeypatch.setattr(integ, "verify_inputs_against_freeze",
                        lambda entries, npz_dir, pt_dir: {
                            e["demo"]: {"npz_matches": True,
                                        "pt_matches": True}
                            for e in entries})
    monkeypatch.setattr(integ, "measure_env_identity", lambda p: {
        "env_python": "x", "env_uv_lock_sha256": "locksha",
        "env_mujoco_version": "3.10.0", "env_lock_mujoco_version": "3.10.0",
        "env_probe_error": None})
    return diag


def test_diagnose_blocked_on_failed_c0_without_any_child_call(tmp_path,
                                                              monkeypatch):
    diag = _patch_d0(monkeypatch, tmp_path)
    baseline = _write_baseline(tmp_path, passed=False)
    def _no_child(*a, **k):  # any subprocess call is the defect
        raise AssertionError("child launched on blocked evidence")
    monkeypatch.setattr(diag, "run_child", _no_child)
    rc = diag.diagnose(_d0_args(tmp_path, baseline))
    assert rc == 2
    blocked = json.loads(sorted(
        (tmp_path / "diag").glob("attempt_*/diagnosis_run.json"))[-1]
        .read_text())["blocked"]
    assert "baseline_c0_not_passed" in blocked


def test_diagnose_non_equivalent_control_runs_no_sub12(tmp_path, monkeypatch):
    diag = _patch_d0(monkeypatch, tmp_path)
    baseline = _write_baseline(tmp_path)
    calls = []

    def fake_child(env_python, demo, data_path, save_dir, snapshot_dir,
                   retargeter_sha, timeout_s):
        calls.append(demo)
        (Path(snapshot_dir)).mkdir(parents=True, exist_ok=True)
        return {"label": demo, "timed_out": False, "launch_error": None,
                "exit_code": 1, "wall_s": 0.1,
                "snapshot": str(snapshot_dir),
                "summary": {"outcome": "RuntimeError: CVXPY solve failed",
                            "failure": None}}
    monkeypatch.setattr(diag, "run_child", fake_child)
    rc = diag.diagnose(_d0_args(tmp_path, baseline))
    assert rc == 2
    assert calls == ["subA_largebox_001"]      # control only, ZERO sub12
    final = json.loads(sorted(
        (tmp_path / "diag").glob("attempt_*/diagnosis_run.json"))[-1]
        .read_text())
    assert final["verdict"]["status"] == "control_not_equivalent"


# --- fixes 1/6: CLI snapshot immutability + merge-chain verification ----------

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
    return freeze


def _write_u0(root: Path) -> Path:
    u0 = root / "u0"
    att1, att2 = u0 / "attempt_001", u0 / "attempt_002"
    att1.mkdir(parents=True, exist_ok=True)
    att2.mkdir(exist_ok=True)
    for d in MEASURABLE:
        (att1 / f"{d}_original.npz").write_bytes(b"raw")
        (att1 / f"{d}_original_converted.npz").write_bytes(b"conv")
    rows1 = ([{"demo": d, "status": "ok", "binding": {"passed": True}}
              for d in MEASURABLE]
             + [{"demo": KNOWN, "status": "generation_failed",
                 "wall_s": 143.9, "error": "infeasible"}])
    (att1 / "generation_manifest.json").write_text(json.dumps(
        {"rows": rows1, "identity": {}, "input_sha256": {}}))
    (att2 / "generation_manifest.json").write_text(json.dumps(
        {"rows": [{"demo": KNOWN, "status": "generation_failed",
                   "wall_s": 144.6, "error": "infeasible"}],
         "identity": {}, "input_sha256": {}}))
    (u0 / integ.ATTEMPTS_INDEX).write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001", "scope": PILOT},
                      {"attempt": "attempt_002", "scope": [KNOWN]}]}))
    return u0


class _Args:
    def __init__(self, freeze, selection, u0, out, diagnosis=None,
                 recovery_report=None, preregistration=None):
        self.freeze, self.selection, self.u0_dir, self.out = (
            freeze, selection, u0, out)
        self.diagnosis = diagnosis
        self.recovery_report = recovery_report
        self.preregistration = preregistration


def _run_cli(tmp_path, monkeypatch, *, args_extra=None):
    import omniretarget_a2_closure as closer
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
    u0 = _write_u0(tmp_path)
    monkeypatch.setattr(closer, "TEACHER_TRAJ_DIR", tmp_path / "npz")
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir(exist_ok=True)
    for d in PILOT:
        (npz_dir / f"{d}_original.npz").write_bytes(b"pub")
    monkeypatch.setattr(closer, "consumed_data_path",
                        lambda root: (pt_dir, False))
    monkeypatch.setattr(closer, "PLAN_ANCHORS", declared)

    def fake_runner(freeze_arg, u0_root, baseline_dir):
        fresh = _report()
        fresh["timestamp"] = "2026-10-11T08:00:00"
        fresh["comparison_attempt"] = "attempt_042"
        fresh["cost_s"] = 12.5
        out = baseline_dir / "cmp" / "attempt_042" / "teacher_selection.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fresh))
        return out

    out = tmp_path / "derived" / "run"
    rc = closer.run_closure(
        _Args(freeze, selection, u0, out, **(args_extra or {})),
        comparison_runner=fake_runner)
    return rc, out, selection


def test_cli_snapshots_are_attempt_scoped_and_immutable(tmp_path,
                                                        monkeypatch):
    rc1, out, _ = _run_cli(tmp_path, monkeypatch)
    assert rc1 == 0
    snap1 = out / "baseline" / "snapshots" / "attempt_001"
    be_bytes = (snap1 / "baseline_evidence.json").read_bytes()
    ev_bytes = (snap1 / "evidence_files.json").read_bytes()
    rc2, out2, _ = _run_cli(tmp_path, monkeypatch)
    assert rc2 == 0
    snap2 = out / "baseline" / "snapshots" / "attempt_002"
    assert (snap2 / "baseline_evidence.json").is_file()
    # the first snapshot bytes are untouched by the second invocation
    assert (snap1 / "baseline_evidence.json").read_bytes() == be_bytes
    assert (snap1 / "evidence_files.json").read_bytes() == ev_bytes
    costs = json.loads((out / "costs.json").read_text())
    assert len(costs["invocations"]) == 2     # fix 11: costs accumulate


def _ur_metric_row(d, *, valid=True, pen=0.03, task=0.075, drift=0.4,
                   ratio=0.2, failed_checks=("velocities",)):
    """A synthetic UR-side row: decisively WORSE than P on penetration /
    drift / violation-ratio with a new validator hard failure, comparable
    on task — the no_quality_gain shape (g2 pass, g3/g4/g5 fail)."""
    if not valid:
        return DemoMetrics(demo=d, T=-1, fps=float("nan"), valid=False,
                           error="no OK UR artifact")
    return DemoMetrics(
        demo=d, T=100, fps=30.0, valid=True,
        penetration_max_m=pen, penetration_violation_ratio=ratio,
        task_mpjpe_p95_m=task, box_ground_penetration_max_m=0.0,
        foot={"l": SideFootMetrics(10, 0.001, 0.01, 0.01, 9),
              "r": SideFootMetrics(10, 0.001, 0.01, 0.01, 9)},
        hand={"l": SideHandMetrics(10, 0.005, 0.5, 0.5, 5, 5, drift, 1, 1),
              "r": SideHandMetrics(10, 0.005, 0.5, 0.5, 5, 5, drift, 1, 1)},
        both={"denominator_frames": 10, "valid_coverage": 0.5,
              "gap_series_p95_m": 0.01},
        validator_failed_checks=list(failed_checks),
        ref_support_frames={"l": 10, "r": 10},
        ref_grasp_frames={"l": 10, "r": 10})


def _recovery_report(tmp_path, *, verdict="no_quality_gain",
                     admission_passed=True, complete=True, missing=None,
                     prereg_sha="shaReg", drop_row=False):
    """A UR report whose verdict record is BUILT by the real evaluation over
    synthetic metrics (round-3 review fix 1: the merge re-derives gates from
    the metrics, so a hand-written gates dict no longer passes); the
    claimed ``verdict`` stays overridable — that is exactly the tamper the
    re-derivation must catch."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    p_rows = {d: _ur_metric_row(d, pen=0.002, task=0.07, drift=0.05,
                                ratio=0.05, failed_checks=())
              for d in PILOT}
    u_rows = {d: _ur_metric_row(d, valid=d not in (missing or []))
              for d in PILOT}
    binding = {d: d not in (missing or []) for d in PILOT}
    vr = tcmp.evaluate_local_preferred(PILOT, p_rows, u_rows, binding,
                                       u_label="UR")
    if not complete:
        vr["complete_evidence"] = False
    vr["verdict"] = verdict
    report = {
        "stage": "UR", "timestamp": "t", "pilot": PILOT, "p_revision": "r",
        "comparison_attempt": "attempt_001",
        "admission": {"passed": admission_passed,
                      "ur_coverage": {"missing": missing or [],
                                      "by_demo": {d: "attempt_001"
                                                  for d in PILOT}}},
        "metrics": {
            "P": {d: p_rows[d].to_json() for d in PILOT},
            "UR": {d: u_rows[d].to_json()
                   for d in (PILOT[:-1] if drop_row else PILOT)},
        },
        "binding": binding,
        "verdict_record": vr,
        "verdict": verdict, "cost_s": 1.0,
        "ur_identity": {"experiment_id": "a2_recovery_step03_v1",
                        "source_id": "upstream_recovery_step03_v1",
                        "preregistration_sha256": prereg_sha},
    }
    p = tmp_path / "ur_report.json"
    p.write_text(json.dumps(report))
    return p


def _diagnosis_file(tmp_path, baseline_snapshot: Path, *, rho=0.2486138173835237):
    """A diagnosis record backed by REAL on-disk snapshots (round-3 review
    fix 2: verify re-hashes the artifacts and re-derives the §5.5 verdict
    from the run records — flags alone no longer pass; round-4: the on-disk
    failure_meta/child_summary carry the same failure dict the record
    claims)."""
    import diagnose_omniretarget_sub12 as diag
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    failure = _canonical_failure(rho=rho)
    sa, sb = tmp_path / "snap_a", tmp_path / "snap_b"
    _write_full_snapshot(sa, failure=failure)
    _write_full_snapshot(sb, failure=failure)
    eval_runs = [_diag_run(sa, rho=rho), _diag_run(sb, rho=rho)]
    verdict = diag.decide_diagnosis({"equal": True}, eval_runs)
    diag_json = {
        "stage": "D0", "verdict": verdict,
        "control": {"comparison": {"equal": True}},
        "runs": [_saved_run(r) for r in eval_runs],
        "baseline": {
            "baseline_evidence": str(baseline_snapshot),
            "baseline_evidence_sha256":
                integ.sha256_file(baseline_snapshot)},
    }
    p = tmp_path / "diagnosis_run.json"
    p.write_text(json.dumps(diag_json))
    return p


def _prereg_file(tmp_path, diagnosis: Path, selection: Path):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    prereg = {
        "experiment_id": "a2_recovery_step03_v1",
        "source_id": "upstream_recovery_step03_v1",
        "single_config_difference": {"retargeter.step_size":
                                     {"from": 0.2, "to": 0.3}},
        "expected_demos": PILOT,
        "parent": {
            "authoritative_selection": str(selection),
            "authoritative_selection_sha256": integ.sha256_file(selection),
            "diagnosis": str(diagnosis),
            "diagnosis_sha256": integ.sha256_file(diagnosis)},
    }
    p = tmp_path / "preregistration.json"
    p.write_text(json.dumps(prereg))
    return p


def _run_merge(tmp_path, monkeypatch, *, report=None, diag=None):
    """First run C1 to create an immutable snapshot, then run C2 merging a
    diagnosis + recovery report + preregistration chain built against it.
    By default the report is bound to the REAL prereg hash (the pass case);
    callers pass an unbound ``report`` for the blocked variants."""
    rc1, out, selection = _run_cli(tmp_path, monkeypatch)
    assert rc1 == 0
    snap = out / "baseline" / "snapshots" / "attempt_001" / \
        "baseline_evidence.json"
    diag_file = diag if diag is not None else _diagnosis_file(
        tmp_path / "m1", snap)
    prereg_file = _prereg_file(tmp_path / "m2", diag_file, selection)
    report_file = report if report is not None else _recovery_report(
        tmp_path / "m3", prereg_sha=integ.sha256_file(prereg_file))
    rc, _out, _sel = _run_cli(tmp_path, monkeypatch, args_extra={
        "diagnosis": diag_file, "recovery_report": report_file,
        "preregistration": prereg_file})
    return rc, out


def test_cli_merge_chain_passes_on_consistent_evidence(tmp_path, monkeypatch):
    rc, out = _run_merge(tmp_path, monkeypatch)
    assert rc == 0
    final = sorted((out / "decisions").glob("attempt_*/a2_closure.json"))[-1]
    rec = json.loads(final.read_text())
    assert rec["registration_status"]["complete"] is True
    assert rec["merge_verification"]["passed"] is True
    assert rec["recovery"]["status"] == "no_quality_gain"
    assert "W1" in "；".join(rec["training"]["unmet"])


@pytest.mark.parametrize("mutate,marker", [
    # top-level verdict says no_quality_gain but the report underneath is
    # not admissible evidence
    (lambda t: _recovery_report(t, admission_passed=False),
     "no_quality_gain_with_failed_admission"),
    (lambda t: _recovery_report(t, complete=False),
     "no_quality_gain_without_complete_evidence"),
    (lambda t: _recovery_report(t, missing=[KNOWN]),
     "no_quality_gain_with_missing_demos"),
    (lambda t: _recovery_report(t, drop_row=True),
     "rows_missing_or_extra"),
])
def test_cli_merge_blocks_on_inconsistent_recovery_report(tmp_path,
                                                          monkeypatch,
                                                          mutate, marker):
    report = mutate(tmp_path / "rep")
    rc, out = _run_merge(tmp_path, monkeypatch, report=report)
    assert rc == 2
    final = sorted((out / "decisions").glob("attempt_*/a2_closure.json"))[-1]
    rec = json.loads(final.read_text())
    assert rec["registration_status"]["complete"] is False
    assert any(marker in p for p in rec["registration_status"][
        "blocking_reasons"]), marker


def test_cli_merge_blocks_on_broken_registration_chain(tmp_path, monkeypatch):
    # (a) the report is not bound to the provided preregistration
    report = _recovery_report(tmp_path / "c0", prereg_sha="shaOfPrereg")
    rc, out = _run_merge(tmp_path, monkeypatch, report=report)
    assert rc == 2
    rec = json.loads(sorted((out / "decisions")
                            .glob("attempt_*/a2_closure.json"))[-1].read_text())
    assert any("report_not_bound_to_prereg" in p for p in rec[
        "registration_status"]["blocking_reasons"])
    # (b) the registered diagnosis file drifted after registration
    rc1, out, selection = _run_cli(tmp_path, monkeypatch)
    snap = out / "baseline" / "snapshots" / "attempt_001" / \
        "baseline_evidence.json"
    diag = _diagnosis_file(tmp_path / "c1", snap)
    prereg = _prereg_file(tmp_path / "c2", diag, selection)
    real_sha = integ.sha256_file(prereg)
    report2 = _recovery_report(tmp_path / "c3", prereg_sha=real_sha)
    diag.write_text(json.dumps(json.loads(diag.read_text())
                               | {"tampered": True}))
    rc2, out2, _sel = _run_cli(tmp_path, monkeypatch, args_extra={
        "diagnosis": diag, "recovery_report": report2,
        "preregistration": prereg})
    assert rc2 == 2
    rec2 = json.loads(sorted((out / "decisions")
                             .glob("attempt_*/a2_closure.json"))[-1].read_text())
    assert any("registered_diagnosis_drifted" in p for p in rec2[
        "registration_status"]["blocking_reasons"])


def test_cli_merge_blocks_on_disagreeing_reconfirmation(tmp_path, monkeypatch):
    rc1, out, selection = _run_cli(tmp_path, monkeypatch)
    snap = out / "baseline" / "snapshots" / "attempt_001" / \
        "baseline_evidence.json"
    registered = _diagnosis_file(tmp_path / "d1", snap)
    prereg = _prereg_file(tmp_path / "d2", registered, selection)
    real_sha = integ.sha256_file(prereg)
    report = _recovery_report(tmp_path / "d3", prereg_sha=real_sha)
    # a DIFFERENT diagnosis merged whose rho disagrees with the registered
    reconf = _diagnosis_file(tmp_path / "d4", snap, rho=0.26)
    rc, out2, _sel = _run_cli(tmp_path, monkeypatch, args_extra={
        "diagnosis": reconf, "recovery_report": report,
        "preregistration": prereg})
    assert rc == 2
    rec = json.loads(sorted((out / "decisions")
                            .glob("attempt_*/a2_closure.json"))[-1].read_text())
    assert any("reconfirmation_rho_differs" in p for p in rec[
        "registration_status"]["blocking_reasons"])


def test_cli_recovery_report_requires_preregistration(tmp_path, monkeypatch):
    rc1, out, selection = _run_cli(tmp_path, monkeypatch)
    snap = out / "baseline" / "snapshots" / "attempt_001" / \
        "baseline_evidence.json"
    diag = _diagnosis_file(tmp_path, snap)
    report = _recovery_report(tmp_path / "e1")
    with pytest.raises(SystemExit, match="preregistration"):
        _run_cli(tmp_path, monkeypatch, args_extra={
            "diagnosis": diag, "recovery_report": report})


# --- fix 2 (live catch): the smoke's default side is the CANONICAL base ------

def test_canonical_base_argv_exposes_extra_flags():
    """The live defect this round: deriving the smoke's default from the
    recorded argv itself (same argv minus the step flag) put extra flags on
    BOTH sides of the diff — ``--retargeter.penetration-tolerance 0.1``
    still diffed single-field. The base must be the canonical U0 template,
    so an extra flag exists only on the recorded side."""
    import omniretarget_teacher_comparison as cmp_mod
    recorded = ["py", "robot_retarget.py",
                "--task-type", "object_interaction",
                "--task-name", "subA", "--data-format", "smplh",
                "--data-path", "/data", "--save-dir", "/out",
                "--task-config.object-name", "largebox",
                "--retargeter.step-size", "0.3",
                "--retargeter.penetration-tolerance", "0.1"]
    base = cmp_mod._canonical_base_argv(recorded)
    assert "--retargeter.penetration-tolerance" not in base
    assert "--retargeter.step-size" not in base
    assert base[base.index("--task-name") + 1] == "subA"
    # the strip-step derivation (the first round's shape) keeps the extra
    # flag on the default side too — the masked-tamper this replaces
    def legacy_strip(argv):
        out, skip = [], False
        for t in argv:
            if skip:
                skip = False
                continue
            if t == "--retargeter.step-size":
                skip = True
                continue
            out.append(t)
        return out
    assert "--retargeter.penetration-tolerance" in legacy_strip(recorded)


# --- fix 11: generation retry cap + UR budget --------------------------------

def _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir, prereg,
                       *, prior_rows=None, prior_wall=0.0, demo_results=None,
                       demos=None, manifest_text=None):
    import generate_omniretarget_teachers as gen
    calls = []
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)

    def fake_generate_one(demo, env_python, attempt_dir, data_path, **kw):
        calls.append(demo)
        status = (demo_results or {}).get(demo, "ok")
        return {"demo": demo, "argv": ["--retargeter.step-size", "0.3"],
                "cwd": "", "exit_code": 0, "wall_s": 10.0,
                "config": "fake", "status": status,
                "binding": {"passed": status == "ok"}, "error": None,
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
        att = out_dir / "attempt_001"
        att.mkdir(parents=True, exist_ok=True)
        (att / "generation_manifest.json").write_text(
            manifest_text if manifest_text is not None else json.dumps({
                "rows": prior_rows, "total_wall_s": prior_wall}))
        (out_dir / integ.ATTEMPTS_INDEX).write_text(json.dumps(
            {"attempts": [{"attempt": "attempt_001",
                           "preregistration_sha256": "shaReg"}]}))
    elif manifest_text is not None:
        # round-4 review fix 4: an index-registered attempt with an
        # unreadable manifest (corrupt json / missing file) must stop the
        # run before generation, not read as zero history
        att = out_dir / "attempt_001"
        att.mkdir(parents=True, exist_ok=True)
        if manifest_text != "__missing__":
            (att / "generation_manifest.json").write_text(manifest_text)
        (out_dir / integ.ATTEMPTS_INDEX).write_text(json.dumps(
            {"attempts": [{"attempt": "attempt_001",
                           "preregistration_sha256": "shaReg"}]}))
    argv = ["prog", "--freeze", str(freeze), "--out-dir", str(out_dir),
            "--source-record", str(prereg)]
    if demos is not None:
        argv += ["--demos", demos]
    monkeypatch.setattr(sys, "argv", argv)
    return gen.main(), calls


def _ur_freeze(tmp_path):
    freeze = tmp_path / "freeze"
    freeze.mkdir(exist_ok=True)
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [
            {"demo": d, "npz_sha256": "fakehash", "pt_sha256": "fakehash"}
            for d in PILOT]}))
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit": "x"}}))
    prereg = tmp_path / "preregistration.json"
    prereg.write_text("{}")
    return freeze, prereg


def test_generation_retry_cap_skips_the_third_call(tmp_path, monkeypatch):
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "ur"
    # subA already failed TWICE in this root (recorded evidence)
    prior = [{"demo": PILOT[0], "status": "generation_failed"},
             {"demo": PILOT[0], "status": "timeout"}]
    rc, calls = _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir,
                                   prereg, prior_rows=prior)
    assert PILOT[0] not in calls            # no third subprocess call
    assert calls == PILOT[1:]
    m = json.loads(sorted(out_dir.glob("attempt_*/generation_manifest.json"))
                   [-1].read_text())
    row0 = next(r for r in m["rows"] if r["demo"] == PILOT[0])
    assert row0["status"] == "skipped_retry_cap"
    assert row0["prior_invocations"] == [
        {"attempt": "attempt_001", "status": "generation_failed"},
        {"attempt": "attempt_001", "status": "timeout"}]
    assert m["n_skipped_retry_cap"] == 1


def test_generation_ur_budget_gate_skips_past_3h(tmp_path, monkeypatch):
    freeze, prereg = _ur_freeze(tmp_path)
    out_dir = tmp_path / "ur2"
    # one prior FAILED run (a success would hit the retry-justification
    # skip first — a retry is legitimate only after a failure)
    prior = [{"demo": PILOT[0], "status": "generation_failed"}]
    rc, calls = _run_generation_ur(monkeypatch, tmp_path, freeze, out_dir,
                                   prereg, prior_rows=prior,
                                   prior_wall=3 * 3600 + 1)
    assert calls == []                      # budget already exhausted
    m = json.loads(sorted(out_dir.glob("attempt_*/generation_manifest.json"))
                   [-1].read_text())
    assert all(r["status"] == "skipped_budget" for r in m["rows"])
    assert m["n_skipped_budget"] == len(PILOT)
