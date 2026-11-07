"""Synthetic counterexamples for the N0 coordinate-flag analyzer (format v2).

Pure synthetic: no real dataset, cache, checkpoint, G1 asset or MuJoCo model
is opened.  The fixture builds a miniature evidence root in the same layout
the T6 audit produced and checks the frozen ``n0_coordinate_flags_v2``
semantics: bit definitions (including the review-added DELTA_PRESENCE_MISMATCH
and DELTA_ONE_SIDED_OBSERVABLE), observability/ok byte replay, per-side
half-ULP write-bound reconstruction, saved same_input formula-gate consumption,
delta_measured/theta consistency, cross-NPZ binding, classification replay,
old-chain key mapping, sealed SHA anchors, roster set equality, labels,
extrema ordering, capacity prevalidation, budget-stop accounting and
exclusive attempt allocation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_n0_coordinate_flags as n0  # noqa: E402
from ireset.eval.w1_real_training_update_audit import (  # noqa: E402
    _f32_array,
    _fma32_array,
    _half_ulp32,
    host_constants,
)

TOLS = {
    "delta_same_input": {"atol": 0.0, "rtol": 5e-4},
    "delta_cross_path": {"atol": 0.0, "rtol": 5e-4},
    "theta_cross_path": {"atol": 2e-7, "rtol": 2e-6},
    "state_cross_path": {"atol": 0.0, "rtol": 2e-3},
}


def _half_ulp32(x):
    xf = x.astype(np.float64)
    up = np.nextafter(x, np.float32(np.inf)).astype(np.float64)
    down = np.nextafter(x, np.float32(-np.inf)).astype(np.float64)
    return np.maximum(xf - down, up - xf) / 2.0


def _side_ref_bound_pe(theta_pre: np.ndarray, ratio32: np.ndarray, step: int):
    """Exact audit-formula construction: decay=R32(pre*d32) (=pre here),
    ref64 = (decay-pre) + h64*ratio64, pe_theta = fma32(h32, ratio, decay),
    bound = half-ULP(pe_theta).  ratio32 must be exactly representable in
    float32 so the detector's division-based reconstruction round-trips."""
    host = host_constants(step)
    h32 = np.float32(host["h_32"])
    h64 = float(host["h_32"])
    d32 = np.float32(host["d_32"])
    decay = (theta_pre * d32).astype(np.float32)
    base = decay.astype(np.float64) - theta_pre.astype(np.float64)
    ref = base + h64 * ratio32.astype(np.float64)
    pe_theta = _fma32_array(h32, ratio32, decay)[0]
    return ref, _half_ulp32(pe_theta), pe_theta


class ParamWorld:
    """Builds consistent saved bytes for one (model, update, parameter) row."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.n = int(np.prod(shape))

    def build(self, coord_list, step: int):
        def col(name, dtype=None):
            vals = [c[name] for c in coord_list]
            arr = np.asarray(vals, dtype=dtype)
            return arr.reshape(self.shape)

        def interval(name):
            lo = np.asarray([c[name][0] for c in coord_list], np.float64).reshape(self.shape)
            hi = np.asarray([c[name][1] for c in coord_list], np.float64).reshape(self.shape)
            return lo, hi

        theta_pre = col("theta_pre", np.float32)
        # Recorded thetas equal the formula-reference new thetas (the audit's
        # same-input formula gate requires this bitwise); "erased" coordinates
        # arise naturally when |h*ratio| < half-ULP so R32 rounds back to pre.
        ref_p, bound_p, pe_p = _side_ref_bound_pe(theta_pre, col("ratio_p", np.float32), step)
        ref_i, bound_i, pe_i = _side_ref_bound_pe(theta_pre, col("ratio_i", np.float32), step)
        theta_p = pe_p
        theta_i = pe_i
        meas_p = theta_p.astype(np.float64) - theta_pre.astype(np.float64)
        meas_i = theta_i.astype(np.float64) - theta_pre.astype(np.float64)
        rtol_same = TOLS["delta_same_input"]["rtol"]
        changed_p = theta_p.view(np.uint32) != theta_pre.view(np.uint32)
        changed_i = theta_i.view(np.uint32) != theta_pre.view(np.uint32)
        obs_p = ((ref_p != 0) & (bound_p <= 0.25 * rtol_same * np.abs(ref_p)) & changed_p).astype(np.uint8)
        obs_i = ((ref_i != 0) & (bound_i <= 0.25 * rtol_same * np.abs(ref_i)) & changed_i).astype(np.uint8)
        ok_p = np.where(ref_p == 0, meas_p == 0,
                        np.where(obs_p.astype(bool),
                                 np.abs(meas_p - ref_p) <= rtol_same * np.abs(ref_p),
                                 False)).astype(np.uint8)
        ok_i = np.where(ref_i == 0, meas_i == 0,
                        np.where(obs_i.astype(bool),
                                 np.abs(meas_i - ref_i) <= rtol_same * np.abs(ref_i),
                                 False)).astype(np.uint8)

        m_p, m_i = col("m_p", np.float32), col("m_i", np.float32)
        v_p, v_i = col("v_p", np.float32), col("v_i", np.float32)
        K_m_lo, K_m_hi = interval("K_m")
        K_v_lo, K_v_hi = interval("K_v")
        K_theta_lo, K_theta_hi = interval("K_theta")
        K_delta_lo, K_delta_hi = interval("K_delta")
        arrays = {
            "theta_pre": theta_pre, "theta_p": theta_p, "theta_i": theta_i,
            "m_p": m_p, "v_p": v_p, "m_i": m_i, "v_i": v_i,
            "delta_ref_p": ref_p, "delta_ref_i": ref_i,
            "delta_measured_p": meas_p, "delta_measured_i": meas_i,
            "bound_delta_p": bound_p, "bound_delta_i": bound_i,
            "K_m_lo": K_m_lo, "K_m_hi": K_m_hi,
            "K_v_lo": K_v_lo, "K_v_hi": K_v_hi,
            "K_theta_lo": K_theta_lo, "K_theta_hi": K_theta_hi,
            "K_delta_lo": K_delta_lo, "K_delta_hi": K_delta_hi,
            "obs_p": obs_p, "obs_i": obs_i, "ok_p": ok_p, "ok_i": ok_i,
        }
        return arrays


WIDE = (-1e6, 1e6)


def default_coords(**over):
    base = dict(ratio_p=0.0, ratio_i=0.0, theta_pre=1.0,
                m_p=1e-3, m_i=1e-3, v_p=1e-6, v_i=1e-6,
                K_m=WIDE, K_v=WIDE, K_theta=WIDE, K_delta=WIDE)
    base.update(over)
    return base


def write_world(root: Path, params: dict, row_meta: dict, analysis_items=None,
                extra_rows=None):
    (root / "registration/attempt_003").mkdir(parents=True)
    (root / "production/attempt_003").mkdir(parents=True)
    (root / "independent/attempt_002").mkdir(parents=True)
    n_det = len(params["deterministic"])
    n_diff = len(params["diffusion"])
    roster_models = {}
    for model, params_by_name in params.items():
        roster_models[model] = {
            "n_parameters": len(params_by_name),
            "parameters": [{"name": name} for name in sorted(params_by_name)],
        }
    reg = {"roster": {"n_det": n_det, "n_diff": n_diff, "models": roster_models},
           "tolerances": TOLS,
           "config": {"lr": 1e-4, "weight_decay": 1e-5}}
    (root / "registration/attempt_003/registration.json").write_text(json.dumps(reg))
    (root / "independent/attempt_002/i_report.json").write_text(json.dumps({"schema": "x"}))
    rows = []
    if extra_rows:
        rows.extend(extra_rows)
    import hashlib
    for model in ("deterministic", "diffusion"):
        i_members, p_members = {}, {}
        for param, updates in params[model].items():
            for update, arrays in updates.items():
                prefix = f"adamw_{model}_u{update}_{param}"
                for suf, arr in arrays.items():
                    i_members[f"{prefix}/{suf}"] = arr
                i_members[f"{prefix}/same_input_p"] = np.array(1, np.uint8)
                i_members[f"{prefix}/same_input_i"] = np.array(1, np.uint8)
                i_members[f"i_theta/u{update}/{param}"] = arrays["theta_i"]
                i_members[f"i_m/u{update}/{param}"] = arrays["m_i"]
                i_members[f"i_v/u{update}/{param}"] = arrays["v_i"]
                p_members[f"theta_before/u{update}/{param}"] = arrays["theta_pre"]
                p_members[f"theta_after/u{update}/{param}"] = arrays["theta_p"]
                meta = row_meta[(model, param, update)]
                rows.append({"key": f"adamw:{model}:u{update}:{param}",
                             "classification": meta["classification"],
                             "gaps": list(meta["gaps"])})
        i_path = root / f"independent/attempt_002/i_{model}.npz"
        p_path = root / f"production/attempt_003/p_{model}.npz"
        np.savez(i_path, **i_members)
        np.savez(p_path, **p_members)
        # sealed anchors: the P/I reports bind the NPZ file hashes
        for report_path, npz_path, mkey in (
                (root / "independent/attempt_002/i_report.json", i_path, f"i_sha_{model}"),
                (root / "production/attempt_003/p_report.json", p_path, f"p_sha_{model}")):
            data = json.loads(report_path.read_text()) if report_path.exists() else {}
            data.setdefault("models", {})[model] = {
                "arrays_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest()}
            report_path.write_text(json.dumps(data))
    (root / "independent/attempt_002/i_rows.json").write_text(json.dumps(rows))
    analysis = {"gaps": {"items": analysis_items or [], "new_only_items": []}}
    apath = root / "analysis.json"
    apath.write_text(json.dumps(analysis))
    _build_journal(root)
    return apath


RETEST_INV = "w1-retest-20261106-003"


def _sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_journal(root: Path):
    """Seal a minimal T6-retest journal chain around the CURRENT fixture
    bytes: return_observation -> commits{P,I,F} -> commit.json ->
    finish.json -> index_row_snapshot.artifacts (run-root-relative paths)
    plus execution_registration_sha256, and the F final report's replay
    array anchor for the P NPZs.  Idempotent: rebuilding after a tamper
    re-seals the chain to the new bytes (used by the tamper tests that
    exercise downstream detection); a tamper WITHOUT rebuilding is the
    coordinated-modification attack tested separately."""
    journal = root / "launch_journal" / RETEST_INV
    events = root / "execution_events"
    reg_sha = _sha(root / "registration/attempt_003/registration.json")
    fin_dir = root / "finalize/attempt_002"
    fin_dir.mkdir(parents=True, exist_ok=True)
    (fin_dir / "training_update_final.json").write_text(json.dumps({
        "worker_result": {"p_replay": {"arrays_sha256": {
            "deterministic": _sha(root / "production/attempt_003/p_deterministic.npz"),
            "diffusion": _sha(root / "production/attempt_003/p_diffusion.npz"),
        }}}},
    ))
    stage_files = {
        "P": ["production/attempt_003/p_deterministic.npz",
              "production/attempt_003/p_diffusion.npz",
              "production/attempt_003/p_report.json"],
        "I": ["independent/attempt_002/i_deterministic.npz",
              "independent/attempt_002/i_diffusion.npz",
              "independent/attempt_002/i_report.json",
              "independent/attempt_002/i_rows.json"],
    }
    commits = {}
    for stage, seq in (("P", 8), ("I", 9), ("F", 10)):
        ev = events / f"{seq:06d}"
        ev.mkdir(parents=True, exist_ok=True)
        if stage == "F":
            artifacts = {"finalize/attempt_002/training_update_final.json":
                         _sha(fin_dir / "training_update_final.json")}
        else:
            artifacts = {rel: _sha(root / rel) for rel in stage_files[stage]}
        (ev / "finish.json").write_text(json.dumps(
            {"index_row_snapshot": {"artifacts": artifacts,
                                    "execution_registration_sha256": reg_sha}}))
        (ev / "commit.json").write_text(json.dumps(
            {"finish": {"path": f"execution_events/{seq:06d}/finish.json",
                        "sha256": _sha(ev / "finish.json")}}))
        commits[stage] = {"path": f"execution_events/{seq:06d}/commit.json",
                          "sha256": _sha(ev / "commit.json")}
    journal.mkdir(parents=True, exist_ok=True)
    (journal / "return_observation.json").write_text(json.dumps({"commits": commits}))



def _reseal(root: Path, model: str, npz_path: Path, rebuild_chain: bool = True):
    """Re-seal the P/I report anchor after a fixture tamper (the tamper tests
    exercise downstream detection, not the anchor gate, which has its own
    dedicated test).  With rebuild_chain=True the journal chain is rebuilt to
    the new bytes so the source-closure gate stays consistent; the
    coordinated-tamper test passes rebuild_chain=False to leave the chain
    sealed to the pre-tamper bytes."""
    import hashlib
    report = root / ("independent/attempt_002/i_report.json"
                     if npz_path.name.startswith("i_") else
                     "production/attempt_003/p_report.json")
    data = json.loads(report.read_text())
    data["models"][model]["arrays_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    report.write_text(json.dumps(data))
    if rebuild_chain:
        _build_journal(root)

def make_analyzer(root: Path, out: Path, budget=None, analysis_path=None):
    budget = budget or n0.N0Budget(wall_cap_s=900.0, reserve_s=60.0)
    return n0.N0Analyzer(root, out, budget, analysis_path=analysis_path)


def standard_world(root: Path):
    """4 params x 2 updates covering every flag bit family (per-update bytes).

    Reference deltas are constructed exactly as the audit formula does:
    ref = (decay-pre) + h64 * f32(ratio); ratios are exact f32 values so the
    detector's reconstruction round-trips bit-for-bit.
    """
    alpha_coords = [
        # c0: precision, write erased on both sides (|h*ratio| < half-ULP so
        # R32 rounds the reference theta back to pre)
        default_coords(ratio_p=-2e-5, ratio_i=-2e-5),
        # c1: observable, same-input pass
        default_coords(ratio_p=-20.0, ratio_i=-20.0),
        # c2: cross-path sensitivity — asymmetric P/I ratios give theta XP +
        # delta XP failures, both sides observable, all inside intervals
        default_coords(ratio_p=-20.0, ratio_i=-40.0),
    ]
    beta_coords = [
        default_coords(),                                            # c0 clean
        default_coords(m_p=1e-3, m_i=2e-3),                          # c1 M XP inside
        default_coords(m_p=1.0, m_i=1e-3, K_m=(0.0, 2e-3)),          # c2 M XP + outside
    ]
    gamma_coords = [
        # c0 theta XP inside; one-sided: ref_i observable, ref_p zero
        default_coords(theta_p=1.0, theta_i=1.001, ratio_i=-1.0, K_theta=(0.9, 1.1)),
        default_coords(m_p=1e-3, m_i=-1e-3, K_m=(-1e-2, 1e-2)),      # c1 m sign + XP
        default_coords(v_p=0.0, v_i=1e-6, K_v=(0.0, 1e-3)),          # c2 v zero + XP
    ]
    delta_coords = [default_coords(), default_coords()]

    params = {"deterministic": {}, "diffusion": {}}
    for update in (1, 2):
        params["deterministic"]["alpha"] = params["deterministic"].get("alpha", {})
        params["deterministic"]["alpha"][update] = ParamWorld((3,)).build(alpha_coords, update)
        params["deterministic"]["beta"] = params["deterministic"].get("beta", {})
        params["deterministic"]["beta"][update] = ParamWorld((3,)).build(beta_coords, update)
        params["diffusion"]["gamma"] = params["diffusion"].get("gamma", {})
        params["diffusion"]["gamma"][update] = ParamWorld((3,)).build(gamma_coords, update)
        params["diffusion"]["delta_p"] = params["diffusion"].get("delta_p", {})
        params["diffusion"]["delta_p"][update] = ParamWorld((2,)).build(delta_coords, update)

    row_meta = {
        ("deterministic", "alpha", 1): {"classification": "cross_path_sensitivity_gap",
                                        "gaps": ["cross_path_sensitivity_gap",
                                                 "parameter_delta_precision_gap"]},
        ("deterministic", "alpha", 2): {"classification": "cross_path_sensitivity_gap",
                                        "gaps": ["cross_path_sensitivity_gap",
                                                 "parameter_delta_precision_gap"]},
        ("deterministic", "beta", 1): {"classification": "cross_path_bound_unverifiable",
                                       "gaps": ["cross_path_bound_unverifiable"]},
        ("deterministic", "beta", 2): {"classification": "cross_path_bound_unverifiable",
                                       "gaps": ["cross_path_bound_unverifiable"]},
        ("diffusion", "gamma", 1): {"classification": "cross_path_sensitivity_gap",
                                    "gaps": ["cross_path_sensitivity_gap"]},
        ("diffusion", "gamma", 2): {"classification": "cross_path_sensitivity_gap",
                                    "gaps": ["cross_path_sensitivity_gap"]},
        ("diffusion", "delta_p", 1): {"classification": "passed", "gaps": []},
        ("diffusion", "delta_p", 2): {"classification": "passed", "gaps": []},
    }
    analysis_items = [
        {"key": "i_adamw_gap:deterministic:u1:alpha:parameter_delta_precision_gap",
         "kind": "parameter_delta_precision_gap", "state": "resolved"},
        {"key": "i_adamw_gap:diffusion:u2:gamma:cross_path_sensitivity_gap",
         "kind": "cross_path_sensitivity_gap", "state": "new"},
        {"key": "i_adamw_gap:deterministic:u2:alpha:cross_path_bound_unverifiable",
         "kind": "cross_path_bound_unverifiable", "state": "resolved"},
    ]
    new_only_items = [
        {"key": "i_adamw_gap:deterministic:u2:beta:parameter_delta_precision_gap",
         "kind": "parameter_delta_precision_gap"},
    ]
    apath = write_world(root, params, row_meta, analysis_items,
                        extra_rows=[{"key": "cotangent:deterministic:u1:m0",
                                     "classification": "synthetic_verified"}])
    # attach new_only_items to the analysis file
    analysis = json.loads(apath.read_text())
    analysis["gaps"]["new_only_items"] = new_only_items
    apath.write_text(json.dumps(analysis))
    return params, row_meta, apath


@pytest.fixture()
def world(tmp_path):
    root = tmp_path / "run_root"
    out = tmp_path / "out"
    params, row_meta, apath = standard_world(root)
    return root, out, apath


def test_flags_bits_and_labels(world):
    root, out, apath = world
    summary = make_analyzer(root, out, analysis_path=apath).run()
    assert summary["coverage_complete"] is True
    assert summary["n_rows_with_anomalies"] == 0
    flags = np.load(out / "coordinate_flags.npy")
    assert flags.dtype == np.uint32
    index = json.loads((out / "coordinate_index.json").read_text())
    assert index["schema"] == "n0_coordinate_flags_v2"
    offsets = {r["key"]: (r["offset"], r["shape"]) for r in index["rows"]}
    keys = [r["key"] for r in index["rows"]]
    assert keys == sorted(keys, key=lambda k: (k.split(":")[1], int(k.split(":")[2][1:]), k.split(":")[3]))

    def bits(key, i):
        off, shape = offsets[key]
        return int(flags[off + i])

    B = n0
    assert bits("adamw:deterministic:u1:alpha", 0) == (
        B.DELTA_UNOBSERVABLE_P | B.DELTA_UNOBSERVABLE_I | B.DELTA_WRITE_ERASED_P | B.DELTA_WRITE_ERASED_I)
    assert bits("adamw:deterministic:u1:alpha", 1) == 0
    assert bits("adamw:deterministic:u1:alpha", 2) == (
        B.THETA_CROSS_PATH_GATE_FAIL | B.DELTA_CROSS_PATH_GATE_FAIL)
    assert bits("adamw:deterministic:u1:beta", 0) == 0
    assert bits("adamw:deterministic:u1:beta", 1) == B.M_CROSS_PATH_GATE_FAIL
    assert bits("adamw:deterministic:u1:beta", 2) == (B.M_CROSS_PATH_GATE_FAIL | B.INTERVAL_OUTSIDE)
    assert bits("adamw:diffusion:u1:gamma", 0) == (
        B.THETA_CROSS_PATH_GATE_FAIL | B.DELTA_PRESENCE_MISMATCH | B.DELTA_ONE_SIDED_OBSERVABLE)
    assert bits("adamw:diffusion:u1:gamma", 1) == (B.M_CROSS_PATH_GATE_FAIL | B.M_ZERO_OR_SIGN_DISAGREE)
    assert bits("adamw:diffusion:u1:gamma", 2) == (B.V_CROSS_PATH_GATE_FAIL | B.V_ZERO_OR_SIGN_DISAGREE)
    assert bits("adamw:diffusion:u1:delta_p", 0) == 0

    import csv as _csv
    with open(out / "gap_rows.csv") as f:
        recs = {r["key"]: r for r in _csv.DictReader(f)}
    assert recs["adamw:deterministic:u1:alpha"]["labels"] == "cross_path_sensitive|precision_limited"
    assert "implementation_issue_candidate" in recs["adamw:deterministic:u1:beta"]["labels"]
    assert recs["adamw:diffusion:u1:gamma"]["labels"] == "cross_path_sensitive"
    assert recs["adamw:diffusion:u1:gamma"]["n_one_sided_observable"] == "1"
    assert recs["adamw:diffusion:u1:gamma"]["n_presence_mismatch"] == "1"
    assert recs["adamw:diffusion:u1:delta_p"]["labels"] == "no_gap"
    # old-chain mapping (i_adamw_gap keys) and new_only annotation
    assert recs["adamw:deterministic:u1:alpha"]["old_state_precision"] == "resolved"
    assert recs["adamw:diffusion:u2:gamma"]["old_state_sensitivity"] == "new"
    assert recs["adamw:deterministic:u2:beta"]["old_new_only"] == "new_only"
    er = summary["event_replay"]
    assert er["precision_rows"] == 2 and er["sensitivity_rows"] == 6
    assert er["both_rows"] == 2 and er["gap_rows"] == 6
    assert summary["seal"]["status"] == "sealed_within_budget"
    assert summary["seal"]["terminal"] is True
    # source closure: every consumed input journal-bound, analysis recorded
    sc = summary["source_closure"]
    assert sc["all_consumed_inputs_journal_bound"] is True
    assert sc["analysis_journal_bound"] is False
    assert {"p_report", "p_npz", "p_npz_diffusion", "i_report", "i_rows",
            "i_npz", "i_npz_diffusion", "registration",
            "p_npz_f_replay_anchor"} <= set(sc["bindings"])
    assert set(sc["stages"]) == {"P", "I", "F"}
    # old-chain comparison delivers all three gap kinds
    oc = summary["old_chain_comparison"]
    assert oc["items_total"] == 3
    assert oc["items_by_kind"]["cross_path_bound_unverifiable"] == 1
    assert oc["items_by_kind"]["parameter_delta_precision_gap"] == 1
    assert oc["items_by_kind"]["cross_path_sensitivity_gap"] == 1


def test_source_anomaly_on_tampered_ok(world):
    root, out, apath = world
    npz_path = root / "independent/attempt_002/i_deterministic.npz"
    members = dict(np.load(npz_path))
    members["adamw_deterministic_u1_alpha/ok_p"] = members["adamw_deterministic_u1_alpha/ok_p"].copy()
    members["adamw_deterministic_u1_alpha/ok_p"][1] = 0
    np.savez(npz_path, **members)
    _reseal(root, "deterministic", npz_path)
    summary = make_analyzer(root, out, analysis_path=apath).run()
    flags = np.load(out / "coordinate_flags.npy")
    assert all(f & n0.SOURCE_ANOMALY for f in flags[0:3])
    assert summary["coverage_complete"] is False


def test_source_anomaly_on_p_binding_mismatch(world):
    root, out, apath = world
    npz_path = root / "production/attempt_003/p_deterministic.npz"
    members = dict(np.load(npz_path))
    members["theta_before/u1/alpha"] = members["theta_before/u1/alpha"] + np.float32(1e-3)
    np.savez(npz_path, **members)
    _reseal(root, "deterministic", npz_path)
    summary = make_analyzer(root, out, analysis_path=apath).run()
    flags = np.load(out / "coordinate_flags.npy")
    assert all(f & n0.SOURCE_ANOMALY for f in flags[0:3])


def test_source_anomaly_on_bad_half_ulp(world):
    root, out, apath = world
    npz_path = root / "independent/attempt_002/i_deterministic.npz"
    members = dict(np.load(npz_path))
    members["adamw_deterministic_u1_alpha/bound_delta_p"] = (
        members["adamw_deterministic_u1_alpha/bound_delta_p"] * 2.0)
    np.savez(npz_path, **members)
    _reseal(root, "deterministic", npz_path)
    summary = make_analyzer(root, out, analysis_path=apath).run()
    flags = np.load(out / "coordinate_flags.npy")
    assert all(f & n0.SOURCE_ANOMALY for f in flags[0:3])


def test_source_anomaly_on_same_input_flag_false(world):
    root, out, apath = world
    npz_path = root / "independent/attempt_002/i_deterministic.npz"
    members = dict(np.load(npz_path))
    members["adamw_deterministic_u1_alpha/same_input_p"] = np.array(0, np.uint8)
    np.savez(npz_path, **members)
    _reseal(root, "deterministic", npz_path)
    summary = make_analyzer(root, out, analysis_path=apath).run()
    flags = np.load(out / "coordinate_flags.npy")
    assert all(f & n0.SOURCE_ANOMALY for f in flags[0:3])
    import csv as _csv
    with open(out / "gap_rows.csv") as f:
        recs = {r["key"]: r for r in _csv.DictReader(f)}
    assert "implementation_issue_candidate" in recs["adamw:deterministic:u1:alpha"]["labels"]


def test_source_anomaly_on_delta_measured_inconsistent(world):
    root, out, apath = world
    npz_path = root / "independent/attempt_002/i_deterministic.npz"
    members = dict(np.load(npz_path))
    arr = members["adamw_deterministic_u1_alpha/delta_measured_p"].copy()
    arr[1] = arr[1] + 1.0
    members["adamw_deterministic_u1_alpha/delta_measured_p"] = arr
    np.savez(npz_path, **members)
    _reseal(root, "deterministic", npz_path)
    summary = make_analyzer(root, out, analysis_path=apath).run()
    flags = np.load(out / "coordinate_flags.npy")
    assert all(f & n0.SOURCE_ANOMALY for f in flags[0:3])


def test_zero_invented_tamper_detected(world):
    """A recorded theta that moved with a zero reference delta is the
    zero-invention direct-failure semantics (unreachable in self-consistent
    evidence, so exercised by tampering)."""
    root, out, apath = world
    npz_path = root / "independent/attempt_002/i_diffusion.npz"
    members = dict(np.load(npz_path))
    key = "adamw_diffusion_u1_delta_p/theta_p"
    members[key] = members[key] + np.float32(1e-3)
    # keep the saved measured delta consistent with the tampered theta so the
    # ZERO_INVENTED bit (not the measured-delta consistency check) fires
    members["adamw_diffusion_u1_delta_p/delta_measured_p"] = (
        members[key].astype(np.float64)
        - members["adamw_diffusion_u1_delta_p/theta_pre"].astype(np.float64))
    np.savez(npz_path, **members)
    _reseal(root, "diffusion", npz_path)
    summary = make_analyzer(root, out, analysis_path=apath).run()
    flags = np.load(out / "coordinate_flags.npy")
    index = json.loads((out / "coordinate_index.json").read_text())
    off = {r["key"]: r["offset"] for r in index["rows"]}["adamw:diffusion:u1:delta_p"]
    assert all(f & n0.DELTA_ZERO_INVENTED for f in flags[off:off + 2])
    assert all(f & n0.SOURCE_ANOMALY for f in flags[off:off + 2])
    assert summary["coverage_complete"] is False


def test_sealed_sha_anchor_mismatch_rejected(world):
    root, out, apath = world
    report_path = root / "independent/attempt_002/i_report.json"
    report = json.loads(report_path.read_text())
    report["models"]["deterministic"]["arrays_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report))
    with pytest.raises(n0.N0Error, match="sealed anchor mismatch"):
        make_analyzer(root, out, analysis_path=apath).run()


def test_roster_substitution_rejected(world):
    root, out, apath = world
    # same row count, one parameter name swapped -> exact-set gate must fire
    rows_path = root / "independent/attempt_002/i_rows.json"
    rows = json.loads(rows_path.read_text())
    for r in rows:
        if r["key"] == "adamw:deterministic:u1:alpha":
            r["key"] = "adamw:deterministic:u1:alpha_swapped"
            break
    rows_path.write_text(json.dumps(rows))
    with pytest.raises(n0.N0Error, match="row key set != roster"):
        make_analyzer(root, out, analysis_path=apath).run()


def test_prevalidation_cap(world, monkeypatch):
    root, out, apath = world
    monkeypatch.setattr(n0, "MAX_TOTAL_COORDS", 5)
    with pytest.raises(n0.N0Error, match="exceeds frozen cap"):
        make_analyzer(root, out, analysis_path=apath).run()
    assert not (out / "coordinate_flags.npy").exists()


def test_budget_stop_partial_coverage(world):
    root, out, apath = world

    class Countdown(n0.N0Budget):
        def __init__(self, remaining_checks, **kw):
            super().__init__(**kw)
            self.left = remaining_checks

        def check(self, point):
            self.left -= 1
            if self.left <= 0:
                self.exceeded = True
                self.stopped_at = point
                raise n0.BudgetExceeded(f"work deadline reached at {point}")

    budget = Countdown(remaining_checks=6, wall_cap_s=900.0, reserve_s=60.0)
    summary = make_analyzer(root, out, budget=budget, analysis_path=apath).run()
    assert budget.exceeded is True
    assert budget.stopped_at.startswith("row[")
    assert summary["coverage_complete"] is False
    assert summary["label_verdict"] == "evidence_incomplete/resource_limit"
    assert summary["budget"]["budget_exceeded"] is True
    flags = np.load(out / "coordinate_flags.npy")
    assert flags.size > 0


def test_extrema_top16_deterministic(tmp_path):
    root = tmp_path / "run_root_extrema"
    out = tmp_path / "out_extrema"
    n = 20
    coords = []
    for i in range(n):
        coords.append(default_coords(m_p=1e-3 * (n - i), m_i=1e-6, K_m=(0.0, 1.0)))
    params = {"deterministic": {}, "diffusion": {}}
    for update in (1, 2):
        params["deterministic"]["big"] = params["deterministic"].get("big", {})
        params["deterministic"]["big"][update] = ParamWorld((n,)).build(coords, update)
        params["diffusion"]["z"] = params["diffusion"].get("z", {})
        params["diffusion"]["z"][update] = ParamWorld((1,)).build(
            [default_coords()], update)
    row_meta = {
        ("deterministic", "big", 1): {"classification": "cross_path_sensitivity_gap",
                                      "gaps": ["cross_path_sensitivity_gap"]},
        ("deterministic", "big", 2): {"classification": "cross_path_sensitivity_gap",
                                      "gaps": ["cross_path_sensitivity_gap"]},
        ("diffusion", "z", 1): {"classification": "passed", "gaps": []},
        ("diffusion", "z", 2): {"classification": "passed", "gaps": []},
    }
    apath = write_world(root, params, row_meta)
    summary = make_analyzer(root, out, analysis_path=apath).run()
    assert summary["coverage_complete"] is True
    import csv as _csv
    with open(out / "gap_coordinates_extrema.csv") as f:
        recs = [r for r in _csv.DictReader(f) if r["row_key"] == "adamw:deterministic:u1:big"]
    assert len(recs) == 16
    flat = [int(r["flat_index"]) for r in recs]
    scores = [float(r["score"]) for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert flat == list(range(16))
    with open(out / "gap_rows.csv") as f:
        rec = [r for r in _csv.DictReader(f) if r["key"] == "adamw:deterministic:u1:big"][0]
    assert rec["n_extrema_sensitivity"] == "20"
    assert int(rec["n_sensitivity"]) == 20


def test_cli_exclusive_attempts(world, tmp_path):
    import subprocess
    root, out, apath = world
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_w1_n0_coordinate_flags.py"
    out_root = tmp_path / "cli_out"
    (out_root / "attempt_001").mkdir(parents=True)
    (out_root / "attempt_003").mkdir(parents=True)
    r = subprocess.run([sys.executable, str(script), "--run-root", str(root),
                        "--output-root", str(out_root),
                        "--analysis-path", str(apath), "--wall-cap-s", "900"],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    # existing 001/003 must never be overwritten: next exclusive attempt is 004
    assert payload["output_dir"].endswith("attempt_004")
    assert payload["coverage_complete"] is True


def test_coordinated_report_npz_tamper_rejected(world):
    """P2-4 attack: tamper an NPZ AND its report's self-reported anchor so the
    intra-report check passes; the journal-sealed chain must still reject."""
    root, out, apath = world
    npz_path = root / "independent/attempt_002/i_diffusion.npz"
    members = dict(np.load(npz_path))
    members["adamw_diffusion_u2_gamma/theta_i"] = (
        members["adamw_diffusion_u2_gamma/theta_i"] + np.float32(1e-3))
    np.savez(npz_path, **members)
    _reseal(root, "diffusion", npz_path, rebuild_chain=False)
    with pytest.raises(n0.N0Error, match="source closure mismatch"):
        make_analyzer(root, out, analysis_path=apath).run()


def test_registration_tamper_rejected(world):
    root, out, apath = world
    reg_path = root / "registration/attempt_003/registration.json"
    reg = json.loads(reg_path.read_text())
    reg["config"]["lr"] = 2e-4
    reg_path.write_text(json.dumps(reg))
    with pytest.raises(n0.N0Error, match="registration file SHA"):
        make_analyzer(root, out, analysis_path=apath).run()


def test_summary_seal_elapsed_vs_cap_withdrawal(tmp_path):
    """P1-1: elapsed seconds must be compared against the wall cap, never
    against the absolute hard_deadline perf-counter value (t0 is large)."""
    budget = n0.N0Budget(wall_cap_s=900.0, reserve_s=60.0, t0=14467.0)
    # the fake clock returns perf-counter-style ABSOLUTE values; elapsed is
    # value - t0, so 901s elapsed must exceed the 900s cap despite the large t0
    ticks = iter([14467.0 + t for t in (10.0, 901.0, 902.0, 905.0, 910.0)])
    summary = {"coverage_complete": True, "label_verdict": "complete", "budget": {}}
    facts = n0._write_summary_sealed(
        tmp_path / "s.json", summary, budget,
        withdraw={"coverage_complete": False,
                  "label_verdict": "budget_exceeded_after_seal"},
        clock=lambda: next(ticks))
    assert facts["status"] == "budget_exceeded_after_seal"
    assert facts["measured_wall_s"] == 905.0
    disk = json.loads((tmp_path / "s.json").read_text())
    assert disk["seal"]["status"] == "budget_exceeded_after_seal"
    assert disk["seal"]["terminal"] is True
    assert disk["coverage_complete"] is False
    assert disk["label_verdict"] == "budget_exceeded_after_seal"


def test_summary_seal_no_write_after_passing_sample(tmp_path, monkeypatch):
    """P1-2 + P1-1: the terminal sealed claim is only written after the
    non-terminal sealing write passed both checks; once the passing sample
    lands, no further write may happen; and the on-disk registered cost
    (pre + tail reserve, incl. process-return tail) must COVER the actual
    post-write elapsed — not merely echo the pre-write sample."""
    calls = {"n": 0}
    real = n0._write_json_fsync

    def counting(path, doc):
        calls["n"] += 1
        real(path, doc)

    monkeypatch.setattr(n0, "_write_json_fsync", counting)
    budget = n0.N0Budget(wall_cap_s=900.0, reserve_s=60.0, t0=0.0)
    ticks = iter([10.0, 20.0, 21.0, 30.0, 31.0, 40.0])
    summary = {"coverage_complete": True, "label_verdict": "complete", "budget": {}}
    facts = n0._write_summary_sealed(tmp_path / "s.json", summary, budget,
                                     clock=lambda: next(ticks))
    assert facts["status"] == "sealed_within_budget"
    assert facts["measured_wall_s"] == 40.0
    assert calls["n"] == 3
    disk = json.loads((tmp_path / "s.json").read_text())
    assert disk["seal"]["status"] == "sealed_within_budget"
    assert disk["coverage_complete"] is True
    # registered cost covers the actual write (30s) and the return tail:
    # measured = 31 + (3 + 2x10) = 54 >= every observed sample
    assert disk["seal"]["measured_wall_s_at_summary"] == 54.0
    assert disk["budget"]["measured_wall_s_at_summary"] == 54.0
    assert disk["seal"]["seal_tail_reserve_s"] == 23.0
    assert disk["seal"]["measured_wall_s_at_summary"] >= facts["measured_wall_s"]
    assert disk["seal"]["terminal"] is True


def test_summary_seal_slow_sealed_write_revokes(tmp_path, monkeypatch):
    """P1-2: the sealed rewrite itself is verified; if its post-write sample
    crosses the cap the completion claim is withdrawn to exceeded."""
    calls = {"n": 0}
    real = n0._write_json_fsync

    def counting(path, doc):
        calls["n"] += 1
        real(path, doc)

    monkeypatch.setattr(n0, "_write_json_fsync", counting)
    budget = n0.N0Budget(wall_cap_s=900.0, reserve_s=60.0, t0=0.0)
    ticks = iter([10.0, 20.0, 21.0, 905.0, 906.0, 910.0])
    summary = {"coverage_complete": True, "label_verdict": "complete", "budget": {}}
    facts = n0._write_summary_sealed(
        tmp_path / "s.json", summary, budget,
        withdraw={"coverage_complete": False,
                  "label_verdict": "budget_exceeded_after_seal"},
        clock=lambda: next(ticks))
    assert facts["status"] == "budget_exceeded_after_seal"
    assert calls["n"] == 3
    disk = json.loads((tmp_path / "s.json").read_text())
    assert disk["seal"]["status"] == "budget_exceeded_after_seal"
    assert disk["coverage_complete"] is False
    assert disk["label_verdict"] == "budget_exceeded_after_seal"


def test_old_state_bound_column_delivered(world):
    """P2-5: the 25 old bound events have no row of their own in the retest
    result but must appear in the per-row comparison output."""
    root, out, apath = world
    summary = make_analyzer(root, out, analysis_path=apath).run()
    import csv as _csv
    with open(out / "gap_rows.csv") as f:
        recs = {r["key"]: r for r in _csv.DictReader(f)}
    assert recs["adamw:deterministic:u2:alpha"]["old_state_bound"] == "resolved"
    assert recs["adamw:deterministic:u1:alpha"]["old_state_bound"] == ""
    oc = summary["old_chain_comparison"]
    assert oc["items_total"] == 3 and oc["new_only_total"] == 1
    assert oc["rows_annotated"] == 3
    assert oc["kinds"] == ["parameter_delta_precision_gap",
                           "cross_path_sensitivity_gap",
                           "cross_path_bound_unverifiable"]


def test_return_coverage_covered_and_withdrawn(tmp_path):
    """P1-1: the actual return time must be covered by the registered cost
    (plan §8.2).  Covered -> no rewrite; uncovered -> conservative withdrawal
    with the withdraw fields applied and the registered cost updated."""
    budget = n0.N0Budget(wall_cap_s=900.0, reserve_s=60.0, t0=0.0)
    doc = {"coverage_complete": True, "label_verdict": "complete",
           "budget": {"measured_wall_s_at_summary": 100.0},
           "seal": {"status": "sealed_within_budget", "terminal": True,
                    "measured_wall_s_at_summary": 100.0, "wall_cap_s": 900.0}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc))
    raw = p.read_bytes()
    cov = n0.verify_return_coverage(p, budget, clock=lambda: 50.0)
    assert cov == {"covered": True, "elapsed_s": 50.0,
                   "registered_measured_s": 100.0, "wall_cap_s": 900.0,
                   "withdrawn": False}
    assert p.read_bytes() == raw  # covered case performs no write
    cov2 = n0.verify_return_coverage(
        p, budget,
        withdraw={"coverage_complete": False,
                  "label_verdict": "budget_exceeded_after_seal"},
        clock=lambda: 200.0)
    assert cov2["covered"] is False and cov2["withdrawn"] is True
    disk = json.loads(p.read_text())
    assert disk["seal"]["status"] == "budget_exceeded_after_seal"
    assert disk["seal"]["note"] == "return_coverage_withdrawn"
    assert disk["seal"]["measured_wall_s_at_summary"] == 200.0
    assert disk["coverage_complete"] is False
    assert disk["label_verdict"] == "budget_exceeded_after_seal"


def test_registration_binding_strict(world):
    """P2: a missing, malformed or disagreeing execution_registration_sha256
    in ANY stage snapshot must be a hard failure, never silently skipped."""
    root, out, apath = world

    def cascade():
        commits = {}
        for stage, seq in (("P", 8), ("I", 9), ("F", 10)):
            ev = root / "execution_events" / f"{seq:06d}"
            (ev / "commit.json").write_text(json.dumps(
                {"finish": {"path": f"execution_events/{seq:06d}/finish.json",
                            "sha256": _sha(ev / "finish.json")}}))
            commits[stage] = {"path": f"execution_events/{seq:06d}/commit.json",
                              "sha256": _sha(ev / "commit.json")}
        (root / "launch_journal" / RETEST_INV / "return_observation.json").write_text(
            json.dumps({"commits": commits}))

    # missing field in one stage
    fin = root / "execution_events/000009/finish.json"
    doc = json.loads(fin.read_text())
    del doc["index_row_snapshot"]["execution_registration_sha256"]
    fin.write_text(json.dumps(doc))
    cascade()
    with pytest.raises(n0.N0Error, match="execution_registration_sha256 missing or malformed"):
        make_analyzer(root, out, analysis_path=apath).run()

    # malformed (non-hex) value
    _build_journal(root)
    fin = root / "execution_events/000009/finish.json"
    doc = json.loads(fin.read_text())
    doc["index_row_snapshot"]["execution_registration_sha256"] = "XYZ" * 21 + "x"
    fin.write_text(json.dumps(doc))
    cascade()
    with pytest.raises(n0.N0Error, match="missing or malformed"):
        make_analyzer(root, out, analysis_path=apath).run()

    # disagreeing values across stages
    _build_journal(root)
    fin = root / "execution_events/000008/finish.json"
    doc = json.loads(fin.read_text())
    doc["index_row_snapshot"]["execution_registration_sha256"] = "a" * 64
    fin.write_text(json.dumps(doc))
    cascade()
    with pytest.raises(n0.N0Error, match="disagrees across journal snapshots"):
        make_analyzer(root, out, analysis_path=apath).run()
