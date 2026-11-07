"""Synthetic counterexamples for the N0 coordinate-flag analyzer.

Pure synthetic: no real dataset, cache, checkpoint, G1 asset or MuJoCo model
is opened.  The fixture builds a miniature evidence root in the same layout
the T6 audit produced and checks the frozen ``n0_coordinate_flags_v1``
semantics: bit definitions, observability/ok byte replay, half-ULP binding,
cross-NPZ binding, classification replay, labels, extrema ordering, capacity
prevalidation, multiset validation and budget-stop accounting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_n0_coordinate_flags as n0  # noqa: E402

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


class ParamWorld:
    """Builds consistent saved bytes for one (model, update, parameter) row."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.n = int(np.prod(shape))

    def build(self, coord_list):
        """coord_list: list of per-coordinate dicts with keys ref_p, ref_i,
        theta_pre, theta_p, theta_i, m_p, m_i, v_p, v_i and interval tuples
        K_m/K_v/K_theta/K_delta.  Measured deltas are derived from thetas."""
        def col(name, dtype=None):
            vals = [c[name] for c in coord_list]
            arr = np.asarray(vals, dtype=dtype)
            return arr.reshape(self.shape)

        def interval(name):
            lo = np.asarray([c[name][0] for c in coord_list], np.float64).reshape(self.shape)
            hi = np.asarray([c[name][1] for c in coord_list], np.float64).reshape(self.shape)
            return lo, hi

        theta_pre = col("theta_pre", np.float32)
        theta_p = col("theta_p", np.float32)
        theta_i = col("theta_i", np.float32)
        meas_p = theta_p.astype(np.float64) - theta_pre.astype(np.float64)
        meas_i = theta_i.astype(np.float64) - theta_pre.astype(np.float64)
        ref_p = col("ref_p", np.float64)
        ref_i = col("ref_i", np.float64)
        rtol_same = TOLS["delta_same_input"]["rtol"]
        changed_p = theta_p.view(np.uint32) != theta_pre.view(np.uint32)
        changed_i = theta_i.view(np.uint32) != theta_pre.view(np.uint32)
        bound = _half_ulp32(theta_i)
        obs_p = ((ref_p != 0) & (bound <= 0.25 * rtol_same * np.abs(ref_p)) & changed_p).astype(np.uint8)
        obs_i = ((ref_i != 0) & (bound <= 0.25 * rtol_same * np.abs(ref_i)) & changed_i).astype(np.uint8)
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
            "bound_delta_p": bound, "bound_delta_i": bound,
            "K_m_lo": K_m_lo, "K_m_hi": K_m_hi,
            "K_v_lo": K_v_lo, "K_v_hi": K_v_hi,
            "K_theta_lo": K_theta_lo, "K_theta_hi": K_theta_hi,
            "K_delta_lo": K_delta_lo, "K_delta_hi": K_delta_hi,
            "obs_p": obs_p, "obs_i": obs_i, "ok_p": ok_p, "ok_i": ok_i,
        }
        return arrays


WIDE = (-1e6, 1e6)


def default_coords(**over):
    base = dict(ref_p=0.0, ref_i=0.0, theta_pre=1.0, theta_p=1.0, theta_i=1.0,
                m_p=1e-3, m_i=1e-3, v_p=1e-6, v_i=1e-6,
                K_m=WIDE, K_v=WIDE, K_theta=WIDE, K_delta=WIDE)
    base.update(over)
    return base


def write_world(root: Path, params: dict, row_meta: dict, analysis_items=None,
                extra_rows=None):
    """params: {model: {param: {update: ParamWorld.build() output}}}.
    row_meta: {(model, param, update): {"classification": str, "gaps": [...]}}.
    """
    (root / "registration/attempt_003").mkdir(parents=True)
    (root / "production/attempt_003").mkdir(parents=True)
    (root / "independent/attempt_002").mkdir(parents=True)
    n_det = len(params["deterministic"])
    n_diff = len(params["diffusion"])
    reg = {"roster": {"n_det": n_det, "n_diff": n_diff}, "tolerances": TOLS}
    (root / "registration/attempt_003/registration.json").write_text(json.dumps(reg))
    (root / "independent/attempt_002/i_report.json").write_text(json.dumps({"schema": "x"}))
    rows = []
    if extra_rows:
        rows.extend(extra_rows)
    for model in ("deterministic", "diffusion"):
        i_members, p_members = {}, {}
        for param, updates in params[model].items():
            for update, arrays in updates.items():
                prefix = f"adamw_{model}_u{update}_{param}"
                for suf, arr in arrays.items():
                    i_members[f"{prefix}/{suf}"] = arr
                i_members[f"i_theta/u{update}/{param}"] = arrays["theta_i"]
                i_members[f"i_m/u{update}/{param}"] = arrays["m_i"]
                i_members[f"i_v/u{update}/{param}"] = arrays["v_i"]
                p_members[f"theta_before/u{update}/{param}"] = arrays["theta_pre"]
                p_members[f"theta_after/u{update}/{param}"] = arrays["theta_p"]
                meta = row_meta[(model, param, update)]
                rows.append({"key": f"adamw:{model}:u{update}:{param}",
                             "classification": meta["classification"],
                             "gaps": list(meta["gaps"])})
        np.savez(root / f"independent/attempt_002/i_{model}.npz", **i_members)
        np.savez(root / f"production/attempt_003/p_{model}.npz", **p_members)
    (root / "independent/attempt_002/i_rows.json").write_text(json.dumps(rows))
    analysis = {"gaps": {"items": analysis_items or []}}
    apath = root / "analysis.json"
    apath.write_text(json.dumps(analysis))
    return apath


def make_analyzer(root: Path, out: Path, budget=None, analysis_path=None):
    budget = budget or n0.N0Budget(wall_cap_s=900.0, reserve_s=60.0)
    return n0.N0Analyzer(root, out, budget, analysis_path=analysis_path)


def standard_world(root: Path):
    """4 params x 2 updates covering every flag bit family."""
    alpha_c1_ref = float(np.float32(1.02)) - 1.0
    alpha = ParamWorld((3,)).build([
        # c0: precision, write erased on both sides
        default_coords(ref_p=1e-5, ref_i=1e-5, theta_p=1.0, theta_i=1.0),
        # c1: observable, same-input pass (ref equals the measured increment)
        default_coords(theta_p=1.02, theta_i=1.02,
                       ref_p=alpha_c1_ref, ref_i=alpha_c1_ref),
        # c2: observable, same-input gate fail on both sides + delta XP fail
        default_coords(theta_p=1.1, theta_i=1.1, ref_p=0.101, ref_i=0.101),
    ])
    beta = ParamWorld((4,)).build([
        default_coords(),                                            # c0 clean
        default_coords(m_p=1e-3, m_i=2e-3),                          # c1 M XP inside
        default_coords(m_p=1.0, m_i=1e-3, K_m=(0.0, 2e-3)),          # c2 M XP + outside
        default_coords(theta_p=1.001, theta_i=1.001),                # c3 zero invented (P)
    ])
    gamma = ParamWorld((3,)).build([
        # c0 theta XP inside; ref_i observable so no zero-invention
        default_coords(theta_p=1.0, theta_i=1.001, ref_i=1e-3, K_theta=(0.9, 1.1)),
        default_coords(m_p=1e-3, m_i=-1e-3, K_m=(-1e-2, 1e-2)),      # c1 m sign + XP
        default_coords(v_p=0.0, v_i=1e-6, K_v=(0.0, 1e-3)),          # c2 v zero + XP
    ])
    delta_p = ParamWorld((2,)).build([default_coords(), default_coords()])
    params = {
        "deterministic": {"alpha": {1: alpha, 2: alpha}, "beta": {1: beta, 2: beta}},
        "diffusion": {"gamma": {1: gamma, 2: gamma}, "delta_p": {1: delta_p, 2: delta_p}},
    }
    row_meta = {
        ("deterministic", "alpha", 1): {"classification": "direct_failure",
                                        "gaps": ["parameter_delta_precision_gap"]},
        ("deterministic", "alpha", 2): {"classification": "direct_failure",
                                        "gaps": ["parameter_delta_precision_gap"]},
        ("deterministic", "beta", 1): {"classification": "direct_failure", "gaps": []},
        ("deterministic", "beta", 2): {"classification": "direct_failure", "gaps": []},
        ("diffusion", "gamma", 1): {"classification": "cross_path_sensitivity_gap",
                                    "gaps": ["cross_path_sensitivity_gap"]},
        ("diffusion", "gamma", 2): {"classification": "cross_path_sensitivity_gap",
                                    "gaps": ["cross_path_sensitivity_gap"]},
        ("diffusion", "delta_p", 1): {"classification": "passed", "gaps": []},
        ("diffusion", "delta_p", 2): {"classification": "passed", "gaps": []},
    }
    analysis_items = [{"key": "adamw:deterministic:u1:alpha",
                       "kind": "parameter_delta_precision_gap", "state": "resolved"}]
    apath = write_world(root, params, row_meta, analysis_items,
                        extra_rows=[{"key": "cotangent:deterministic:u1:m0",
                                     "classification": "synthetic_verified"}])
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
    flags = np.load(out / "coordinate_flags.npy")
    index = json.loads((out / "coordinate_index.json").read_text())
    offsets = {r["key"]: (r["offset"], r["shape"]) for r in index["rows"]}
    # canonical row order: deterministic before diffusion, u1 before u2
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
        B.DELTA_SAME_INPUT_GATE_FAIL_P | B.DELTA_SAME_INPUT_GATE_FAIL_I | B.DELTA_CROSS_PATH_GATE_FAIL)
    assert bits("adamw:deterministic:u1:beta", 0) == 0
    assert bits("adamw:deterministic:u1:beta", 1) == B.M_CROSS_PATH_GATE_FAIL
    assert bits("adamw:deterministic:u1:beta", 2) == (B.M_CROSS_PATH_GATE_FAIL | B.INTERVAL_OUTSIDE)
    assert bits("adamw:deterministic:u1:beta", 3) == B.DELTA_ZERO_INVENTED
    assert bits("adamw:diffusion:u1:gamma", 0) == B.THETA_CROSS_PATH_GATE_FAIL
    assert bits("adamw:diffusion:u1:gamma", 1) == (B.M_CROSS_PATH_GATE_FAIL | B.M_ZERO_OR_SIGN_DISAGREE)
    assert bits("adamw:diffusion:u1:gamma", 2) == (B.V_CROSS_PATH_GATE_FAIL | B.V_ZERO_OR_SIGN_DISAGREE)
    assert bits("adamw:diffusion:u1:delta_p", 0) == 0

    rows = {r["key"]: r for r in index["rows"]}
    # gap_rows.csv labels
    import csv as _csv
    with open(out / "gap_rows.csv") as f:
        recs = {r["key"]: r for r in _csv.DictReader(f)}
    assert "implementation_issue_candidate" in recs["adamw:deterministic:u1:alpha"]["labels"]
    assert "implementation_issue_candidate" in recs["adamw:deterministic:u1:beta"]["labels"]
    assert recs["adamw:diffusion:u1:gamma"]["labels"] == "cross_path_sensitive"
    assert recs["adamw:diffusion:u1:delta_p"]["labels"] == "no_gap"
    assert recs["adamw:deterministic:u1:alpha"]["old_state_precision"] == "resolved"
    # event replay from flags
    er = summary["event_replay"]
    assert er["precision_rows"] == 2 and er["sensitivity_rows"] == 2
    assert er["both_rows"] == 0 and er["gap_rows"] == 4


def test_source_anomaly_on_tampered_ok(world, tmp_path):
    root, out, apath = world
    # tamper one saved ok byte in the i NPZ for alpha u1
    npz_path = root / "independent/attempt_002/i_deterministic.npz"
    members = dict(np.load(npz_path))
    members["adamw_deterministic_u1_alpha/ok_p"] = members["adamw_deterministic_u1_alpha/ok_p"].copy()
    members["adamw_deterministic_u1_alpha/ok_p"][1] = 0
    np.savez(npz_path, **members)
    summary = make_analyzer(root, out, analysis_path=apath).run()
    flags = np.load(out / "coordinate_flags.npy")
    off = 0  # alpha u1 is the first row
    assert all(f & n0.SOURCE_ANOMALY for f in flags[off:off + 3])
    assert summary["coverage_complete"] is False


def test_source_anomaly_on_p_binding_mismatch(world):
    root, out, apath = world
    npz_path = root / "production/attempt_003/p_deterministic.npz"
    members = dict(np.load(npz_path))
    members["theta_before/u1/alpha"] = members["theta_before/u1/alpha"] + np.float32(1e-3)
    np.savez(npz_path, **members)
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
    summary = make_analyzer(root, out, analysis_path=apath).run()
    flags = np.load(out / "coordinate_flags.npy")
    assert all(f & n0.SOURCE_ANOMALY for f in flags[0:3])


def test_prevalidation_cap(world, tmp_path, monkeypatch):
    root, out, apath = world
    monkeypatch.setattr(n0, "MAX_TOTAL_COORDS", 5)
    with pytest.raises(n0.N0Error, match="exceeds frozen cap"):
        make_analyzer(root, out, analysis_path=apath).run()
    assert not (out / "coordinate_flags.npy").exists()


def test_row_multiset_mismatch(world, tmp_path):
    root, out, apath = world
    reg_path = root / "registration/attempt_003/registration.json"
    reg = json.loads(reg_path.read_text())
    reg["roster"]["n_det"] = 3
    reg_path.write_text(json.dumps(reg))
    with pytest.raises(n0.N0Error, match="row multiset mismatch"):
        make_analyzer(root, out, analysis_path=apath).run()


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
    # completed rows are retained in flags
    flags = np.load(out / "coordinate_flags.npy")
    assert flags.size > 0


def test_extrema_top16_deterministic(tmp_path):
    root = tmp_path / "run_root_extrema"
    out = tmp_path / "out_extrema"
    n = 20
    coords = []
    for i in range(n):
        # descending scores 20-i via m gate ratio; all inside intervals
        coords.append(default_coords(m_p=1e-3 * (n - i), m_i=1e-6, K_m=(0.0, 1.0)))
    world_arrays = ParamWorld((n,)).build(coords)
    params = {"deterministic": {"big": {1: world_arrays, 2: world_arrays}},
              "diffusion": {"z": {1: ParamWorld((1,)).build([default_coords()]),
                                  2: ParamWorld((1,)).build([default_coords()])}}}
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
    # top-16 by score desc, coordinate order tiebreak
    assert flat == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    with open(out / "gap_rows.csv") as f:
        rec = [r for r in _csv.DictReader(f) if r["key"] == "adamw:deterministic:u1:big"][0]
    assert rec["n_extrema_sensitivity"] == "20"
    assert int(rec["n_sensitivity"]) == 20


def test_cli_exit_codes(world, tmp_path):
    import subprocess
    root, out, apath = world
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_w1_n0_coordinate_flags.py"
    r = subprocess.run([sys.executable, str(script), "--run-root", str(root),
                        "--output-root", str(tmp_path / "cli_out"),
                        "--analysis-path", str(apath),
                        "--wall-cap-s", "900"],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["coverage_complete"] is True
    # attempt directories are never overwritten on a second run
    r2 = subprocess.run([sys.executable, str(script), "--run-root", str(root),
                         "--output-root", str(tmp_path / "cli_out"),
                         "--analysis-path", str(apath), "--wall-cap-s", "900"],
                        capture_output=True, text=True, cwd=str(tmp_path))
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert payload["output_dir"] != json.loads(r2.stdout)["output_dir"]
