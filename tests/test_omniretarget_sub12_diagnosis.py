"""D0 sub12 diagnosis contracts — closure plan §10.2 counterexamples.

Pure parts (classifier, decision aggregation, control comparison) run in
the main environment; the cvxpy-dependent ``phase_one`` is exercised in the
upstream environment via the child's ``--mode self-test`` (skipped when the
independent env is absent).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _omniretarget_diagnostic_child import (  # noqa: E402
    RESIDUAL_TOL, classify_phase_one)
from ireset.data import a2_integrity as integ  # noqa: E402
import diagnose_omniretarget_sub12 as diag  # noqa: E402

UPSTREAM_ENV = Path(__file__).resolve().parents[1] / "derived" / "upstream" \
    / "holosoma_env" / ".venv" / "bin" / "python"
CHILD = Path(__file__).resolve().parents[1] / "scripts" \
    / "_omniretarget_diagnostic_child.py"


# --- the §5.5 table, every row ----------------------------------------------

def test_phase_one_classifier_table():
    assert classify_phase_one("optimal", 0.15, 0.0) == "rho_within_original_radius"
    assert classify_phase_one("optimal", 0.2, 1e-9) == "rho_within_original_radius"
    assert classify_phase_one("optimal", 0.2 + 1e-6, 1e-9) == \
        "rho_within_original_radius"           # boundary is inclusive
    assert classify_phase_one("optimal", 0.2 + 2e-6, 1e-9) == \
        "supports_step03_candidate"
    assert classify_phase_one("optimal", 0.3 - 1e-6, 0.0) == \
        "supports_step03_candidate"
    assert classify_phase_one("optimal", 0.3, 0.0) == "rho_exceeds_candidate_range"
    assert classify_phase_one("optimal", 0.45, 0.0) == "rho_exceeds_candidate_range"
    assert classify_phase_one("infeasible", None, None) == "retained_infeasible"
    assert classify_phase_one("infeasible_inaccurate", None, None) == \
        "retained_infeasible"


def test_phase_one_classifier_rejects_unusable_evidence():
    assert classify_phase_one("optimal_inaccurate", 0.25, 0.0) == \
        "diagnostic_incomplete"
    assert classify_phase_one("optimal", None, 0.0) == "diagnostic_incomplete"
    assert classify_phase_one("optimal", float("nan"), 0.0) == \
        "diagnostic_incomplete"
    assert classify_phase_one("optimal", float("inf"), 0.0) == \
        "diagnostic_incomplete"
    assert classify_phase_one("optimal", 0.25, None) == "diagnostic_incomplete"
    assert classify_phase_one("optimal", 0.25, float("nan")) == \
        "diagnostic_incomplete"
    # residual beyond the 1e-6 diagnostic bound is NOT a candidate
    assert classify_phase_one("optimal", 0.25, RESIDUAL_TOL * 10) == \
        "diagnostic_incomplete"
    assert classify_phase_one("optimal", 0.25, RESIDUAL_TOL) == \
        "supports_step03_candidate"


# --- decision aggregation ----------------------------------------------------

def _ctx_npz(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    np.savez(path, q_locked=rng.normal(size=43),
             q_a_n_last=rng.normal(size=29),
             q_t_last=rng.normal(size=43),
             target_laplacian=rng.normal(size=30),
             obj_pts_local=rng.normal(size=(10, 3)))


_DEFAULT = object()


def _run(label: str, snapshot: Path, *, frame=171, ordinal=4,
         category="supports_step03_candidate", rho=0.25, outcome=None,
         timed_out=False, launch_error=None, failure=_DEFAULT):
    if failure is _DEFAULT:
        failure = {"failing_frame_idx": frame, "ordinal_in_frame": ordinal,
                   "phase_one": {"category": category, "rho": rho,
                                 "status": "optimal"}}
    return {"label": label, "snapshot": str(snapshot), "timed_out": timed_out,
            "launch_error": launch_error,
            "summary": None if (timed_out or launch_error) else {
                "outcome": outcome or "RuntimeError: CVXPY solve failed: infeasible",
                "failure": failure}}


@pytest.fixture()
def two_snaphots(tmp_path):
    sa, sb = tmp_path / "snap_a", tmp_path / "snap_b"
    sa.mkdir()
    sb.mkdir()
    _ctx_npz(sa / "failure_context.npz", 7)
    np.savez(sb / "failure_context.npz",
             **{k: v for k, v in np.load(sa / "failure_context.npz").items()})
    return sa, sb


def test_decide_supports_candidate_when_two_runs_consistent(tmp_path,
                                                            two_snaphots):
    a, b = two_snaphots
    v = diag.decide_diagnosis({"equal": True},
                              [_run("r1", a), _run("r2", b, rho=0.25 + 5e-7)])
    assert v["status"] == "supports_step03_candidate"
    assert v["same_frame"] and v["same_ordinal"]
    assert v["key_arrays_equal"]["all_equal"]
    assert v["rho_agreement"]


def test_decide_control_not_equivalent_blocks_attribution(two_snaphots):
    a, b = two_snaphots
    v = diag.decide_diagnosis({"equal": False, "reason": "control_raw_missing"},
                              [_run("r1", a), _run("r2", b)])
    assert v["status"] == "control_not_equivalent"
    assert v["reason"] == "control_raw_missing"


def test_decide_nondeterministic_on_frame_ordinal_or_array_mismatch(
        tmp_path, two_snaphots):
    a, b = two_snaphots
    v = diag.decide_diagnosis({"equal": True},
                              [_run("r1", a), _run("r2", b, frame=172)])
    assert v["status"] == "nondeterministic"
    v = diag.decide_diagnosis({"equal": True},
                              [_run("r1", a), _run("r2", b, ordinal=5)])
    assert v["status"] == "nondeterministic"
    sc = tmp_path / "snap_c"
    sc.mkdir()
    np.savez(sc / "failure_context.npz",
             **{k: v + 1e-3
                for k, v in np.load(a / "failure_context.npz").items()})
    v = diag.decide_diagnosis({"equal": True},
                              [_run("r1", a), _run("r2", sc)])
    assert v["status"] == "nondeterministic"
    assert not v["key_arrays_equal"]["all_equal"]


def test_decide_rho_disagreement_is_nondeterministic(two_snaphots):
    a, b = two_snaphots
    v = diag.decide_diagnosis({"equal": True},
                              [_run("r1", a, rho=0.25),
                               _run("r2", b, rho=0.26)])
    assert v["status"] == "nondeterministic"
    assert "rho" in v["note"]


@pytest.mark.parametrize("make_runs,marker", [
    (lambda a, b: [_run("r1", a), _run("r2", b, timed_out=True)],
     "timeout"),
    (lambda a, b: [_run("r1", a), _run("r2", b, launch_error="x")],
     "launch_failed"),
    (lambda a, b: [_run("r1", a), _run("r2", b, failure={})],
     "unexpected_success"),
    (lambda a, b: [_run("r1", a),
                   _run("r2", b, outcome="instrumentation_not_ready",
                        failure=None)],
     "instrumentation_not_ready"),
    (lambda a, b: [_run("r1", a), {"label": "r2", "timed_out": False,
                                   "launch_error": None, "summary": None}],
     "no_child_summary"),
])
def test_decide_incomplete_on_unusable_runs(make_runs, marker):
    runs = make_runs(None, None)
    v = diag.decide_diagnosis({"equal": True}, runs)
    assert v["status"] == "diagnostic_incomplete"
    assert any(marker in p for p in v["problems"])


def test_decide_terminal_non_candidate_rows(two_snaphots):
    a, b = two_snaphots
    v = diag.decide_diagnosis({"equal": True},
                              [_run("r1", a, rho=0.45,
                                    category="rho_exceeds_candidate_range"),
                               _run("r2", b, rho=0.45,
                                    category="rho_exceeds_candidate_range")])
    assert v["status"] == "no_recovery_candidate:rho_exceeds_candidate_range"
    v = diag.decide_diagnosis({"equal": True},
                              [_run("r1", a, category="retained_infeasible",
                                    rho=None),
                               _run("r2", b, category="retained_infeasible",
                                    rho=None)])
    assert v["status"] == "no_recovery_candidate:retained_infeasible"
    v = diag.decide_diagnosis({"equal": True},
                              [_run("r1", a, rho=0.1,
                                    category="rho_within_original_radius"),
                               _run("r2", b, rho=0.1,
                                    category="rho_within_original_radius")])
    assert v["status"] == "no_recovery_candidate:rho_within_original_radius"


# --- control comparison ------------------------------------------------------

def test_control_comparison_detects_drift_and_difference(tmp_path):
    rng = np.random.default_rng(1)
    ref = tmp_path / "registered.npz"
    np.savez(ref, qpos=rng.normal(size=(5, 43)),
             human_joints=rng.normal(size=(5, 52, 3)), fps=30)
    same = tmp_path / "same.npz"
    np.savez(same, qpos=np.load(ref)["qpos"],
             human_joints=np.load(ref)["human_joints"], fps=30)
    ok = diag.compare_control(same, ref, integ.sha256_file(ref))
    assert ok["equal"] and ok["registered_sha_matches"]
    drifted = tmp_path / "drifted.npz"
    np.savez(drifted, qpos=np.load(ref)["qpos"] + 1e-9,
             human_joints=np.load(ref)["human_joints"], fps=30)
    assert not diag.compare_control(drifted, ref, integ.sha256_file(ref))["equal"]
    assert diag.compare_control(same, ref, "wrongsha")["equal"] is False
    assert diag.compare_control(tmp_path / "missing.npz", ref,
                                integ.sha256_file(ref))["reason"] == \
        "control_raw_missing"


# --- child launch handling ---------------------------------------------------

def test_run_child_timeout_kills_and_records(tmp_path, monkeypatch):
    stub = tmp_path / "stub_child.py"
    stub.write_text("import time\nprint('started', flush=True)\n"
                    "time.sleep(30)\n")
    monkeypatch.setattr(diag, "CHILD_DRIVER", stub)
    rec = diag.run_child(Path(sys.executable), "demoX", tmp_path,
                         tmp_path / "save", tmp_path / "snap",
                         "sha", 3)
    assert rec["timed_out"] is True and rec["exit_code"] is None
    assert (tmp_path / "snap" / "TIMEOUT.txt").is_file()
    assert "started" in (tmp_path / "snap" / "child.stdout.txt").read_text()
    # the process group is gone: no orphan sleeping past the timeout
    probe = subprocess.run(["pgrep", "-f", "stub_child.py"],
                           capture_output=True, text=True)
    assert not probe.stdout.strip()


def test_run_child_missing_interpreter_recorded_not_raised(tmp_path):
    rec = diag.run_child(tmp_path / "no_such_python", "demoX", tmp_path,
                         tmp_path / "save2", tmp_path / "snap2", "sha", 10)
    assert rec.get("launch_error")
    assert "FileNotFoundError" in rec["launch_error"]


# --- phase_one on real cone problems (upstream env only) ---------------------

@pytest.mark.skipif(not UPSTREAM_ENV.is_file(), reason="upstream env absent")
def test_child_self_test_covers_cone_rows():
    proc = subprocess.run([str(UPSTREAM_ENV), str(CHILD), "--mode",
                           "self-test"], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "self-test PASS" in proc.stdout
    payload = json.loads(proc.stdout[:proc.stdout.index("self-test")])
    rows = {r["case"]: r for r in payload["self_test"]}
    assert abs(rows["within"]["rho"] - 0.15) < 1e-6
    assert rows["within"]["category"] == "rho_within_original_radius"
    assert abs(rows["between"]["rho"] - 0.25) < 1e-6
    assert rows["between"]["category"] == "supports_step03_candidate"
    assert abs(rows["beyond"]["rho"] - 0.45) < 1e-6
    assert rows["beyond"]["category"] == "rho_exceeds_candidate_range"
    assert rows["infeasible"]["category"] == "retained_infeasible"
    assert rows["two_soc"]["status"] == "not_single_soc"
    assert all(r["max_violation"] <= RESIDUAL_TOL
               for c, r in rows.items() if "max_violation" in r)
