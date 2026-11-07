"""W1 N0: coordinate-level finite analysis of saved gradient-numerics gap evidence.

Plan: docs/plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md
(v0.4 section 4, format revised to v2 by the 2026-11-07 independent review).

Read-only CPU analysis over the saved T6 evidence root. No model construction,
no forward/backward/FK/AdamW execution, no checkpoint loading, no new real
numerics beyond float64 CPU arithmetic on saved arrays.

Frozen output format ``n0_coordinate_flags_v2`` (v1 was executed once as
``n0/attempt_001`` and is retained as superseded evidence; v2 extends the bit
layout after the independent review):

- Row universe: the 528 rows in ``i_rows.json`` whose key starts with
  ``adamw:``, validated as the EXACT set {parameter} x {update 1, 2} per model
  against ``registration.roster.models[*].parameters`` (set equality, not just
  counts), in canonical order (model, update, parameter).
- ``coordinate_flags.npy``: one ``uint32`` per coordinate, rows concatenated in
  row order, each row expanded C-order over its native NPZ shape.
- Bit definitions (one-hot, frozen before execution):

  b0  DELTA_UNOBSERVABLE_P         delta_ref_p != 0 and obs_p is False
  b1  DELTA_UNOBSERVABLE_I         delta_ref_i != 0 and obs_i is False
  b2  DELTA_WRITE_ERASED_P         delta_ref_p != 0 and delta_measured_p == 0
  b3  DELTA_WRITE_ERASED_I         delta_ref_i != 0 and delta_measured_i == 0
  b4  DELTA_SAME_INPUT_GATE_FAIL_P obs_p and not ok_p (rtol=5e-4)
  b5  DELTA_SAME_INPUT_GATE_FAIL_I obs_i and not ok_i
  b6  DELTA_ZERO_INVENTED          (delta_ref == 0) and (delta_measured != 0)
  b7  THETA_CROSS_PATH_GATE_FAIL   |theta_p - theta_i| > atol + rtol*|theta_i|
  b8  M_CROSS_PATH_GATE_FAIL       same form on (m_p, m_i)
  b9  V_CROSS_PATH_GATE_FAIL       same form on (v_p, v_i)
  b10 DELTA_CROSS_PATH_GATE_FAIL   |delta_measured_p - delta_ref_i| > rtol*|ref_i|
                                   on the obs_p & obs_i mask
  b11 M_ZERO_OR_SIGN_DISAGREE      (m_p==0)!=(m_i==0) or opposite nonzero signs
  b12 V_ZERO_OR_SIGN_DISAGREE      same on v
  b13 THETA_ZERO_OR_SIGN_DISAGREE  same on theta
  b14 INTERVAL_OUTSIDE             any of m/v/theta measured+reference outside
                                   the saved K_* propagation interval
  b15 SOURCE_ANOMALY               recomputed bytes disagree with saved bytes,
                                   nonfinite input, shape/dtype mismatch,
                                   binding mismatch, saved same_input flag
                                   false, or row classification not
                                   reproducible
  b16 DELTA_PRESENCE_MISMATCH      (delta_ref_p != 0) != (delta_ref_i != 0)
  b17 DELTA_ONE_SIDED_OBSERVABLE   obs_p != obs_i

- Row labels derived from flags only (analysis labels, NOT pass exemptions):
  ``evidence_incomplete`` (structural failure) > ``implementation_issue_candidate``
  (SOURCE_ANOMALY, same-input gate fail on observable coordinates, zero-invented
  delta, or INTERVAL_OUTSIDE) > ``precision_limited`` and/or
  ``cross_path_sensitive`` (cross-path gate or delta-existence fails with every
  failing coordinate inside the saved propagation intervals).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ireset.eval.w1_real_training_update_audit import (
    _f32_array,
    _fma32_array,
    _half_ulp32,
    host_constants,
)

SCHEMA = "n0_coordinate_flags_v2"

BIT_NAMES = {
    0: "DELTA_UNOBSERVABLE_P",
    1: "DELTA_UNOBSERVABLE_I",
    2: "DELTA_WRITE_ERASED_P",
    3: "DELTA_WRITE_ERASED_I",
    4: "DELTA_SAME_INPUT_GATE_FAIL_P",
    5: "DELTA_SAME_INPUT_GATE_FAIL_I",
    6: "DELTA_ZERO_INVENTED",
    7: "THETA_CROSS_PATH_GATE_FAIL",
    8: "M_CROSS_PATH_GATE_FAIL",
    9: "V_CROSS_PATH_GATE_FAIL",
    10: "DELTA_CROSS_PATH_GATE_FAIL",
    11: "M_ZERO_OR_SIGN_DISAGREE",
    12: "V_ZERO_OR_SIGN_DISAGREE",
    13: "THETA_ZERO_OR_SIGN_DISAGREE",
    14: "INTERVAL_OUTSIDE",
    15: "SOURCE_ANOMALY",
    16: "DELTA_PRESENCE_MISMATCH",
    17: "DELTA_ONE_SIDED_OBSERVABLE",
}
DELTA_UNOBSERVABLE_P = 1 << 0
DELTA_UNOBSERVABLE_I = 1 << 1
DELTA_WRITE_ERASED_P = 1 << 2
DELTA_WRITE_ERASED_I = 1 << 3
DELTA_SAME_INPUT_GATE_FAIL_P = 1 << 4
DELTA_SAME_INPUT_GATE_FAIL_I = 1 << 5
DELTA_ZERO_INVENTED = 1 << 6
THETA_CROSS_PATH_GATE_FAIL = 1 << 7
M_CROSS_PATH_GATE_FAIL = 1 << 8
V_CROSS_PATH_GATE_FAIL = 1 << 9
DELTA_CROSS_PATH_GATE_FAIL = 1 << 10
M_ZERO_OR_SIGN_DISAGREE = 1 << 11
V_ZERO_OR_SIGN_DISAGREE = 1 << 12
THETA_ZERO_OR_SIGN_DISAGREE = 1 << 13
INTERVAL_OUTSIDE = 1 << 14
SOURCE_ANOMALY = 1 << 15
DELTA_PRESENCE_MISMATCH = 1 << 16
DELTA_ONE_SIDED_OBSERVABLE = 1 << 17
FLAGS_DTYPE = np.uint32
MAX_FLAG_BIT = 17

# Per-row saved-array suffixes consumed from the i NPZ.
I_SUFFIXES = (
    "theta_pre", "theta_p", "theta_i", "m_p", "v_p", "m_i", "v_i",
    "delta_ref_p", "delta_ref_i", "delta_measured_p", "delta_measured_i",
    "bound_delta_p", "bound_delta_i",
    "K_m_lo", "K_m_hi", "K_v_lo", "K_v_hi", "K_theta_lo", "K_theta_hi",
    "K_delta_lo", "K_delta_hi",
    "obs_p", "obs_i", "ok_p", "ok_i",
)
# Scalar (shape ()) saved formula-gate flags: same-input formula bitwise gate.
I_SCALAR_SUFFIXES = ("same_input_p", "same_input_i")
# Reference state duplicates saved under i_theta/i_m/i_v keys.
I_STATE_SUFFIXES = ("i_theta", "i_m", "i_v")
P_CROSSCHECK = ("theta_before", "theta_after")

MAX_TOTAL_COORDS = 16_000_000
FLAGS_NPY_HEADER_RESERVE = 1024
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_GAP_ROWS_CSV_BYTES = 8 * 1024 * 1024
MAX_EXTREMA_CSV_BYTES = 64 * 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_EXTREMA_PER_ROW_CATEGORY = 16
MAX_EXTREMA_RECORD_BYTES = 2048

EXTREMA_COLUMNS = [
    "row_key", "category", "model", "update", "parameter", "flat_index",
    "coord_index", "bits_hex", "score", "max_ratio", "max_abs_err",
    "same_ratio_p", "same_ratio_i", "delta_ref_p", "delta_ref_i",
    "delta_measured_p", "delta_measured_i", "bound_delta_p", "bound_delta_i",
    "theta_ratio", "m_ratio", "v_ratio", "delta_xp_ratio",
    "theta_p", "theta_i", "interval_outside", "source_anomaly",
]

GAP_ROWS_COLUMNS = [
    "key", "model", "update", "parameter", "shape", "gap", "n_coords",
    "n_precision", "n_sensitivity", "n_source_anomaly", "n_interval_outside",
    "n_presence_mismatch", "n_one_sided_observable",
    "labels", "saved_classification", "recomputed_classification",
    "classification_match", "old_state_precision", "old_state_sensitivity",
    "old_new_only", "max_same_delta_ratio", "max_delta_ratio",
    "max_theta_ratio", "max_m_ratio", "max_v_ratio",
    "n_extrema_precision", "n_extrema_sensitivity",
]


class N0Error(RuntimeError):
    """Fail-closed analysis error (structural, not per-coordinate)."""


class BudgetExceeded(N0Error):
    pass


@dataclass
class N0Budget:
    wall_cap_s: float = 900.0
    reserve_s: float = 60.0
    t0: float = field(default_factory=time.perf_counter)
    stopped_at: str = ""
    exceeded: bool = False

    @property
    def work_deadline(self) -> float:
        return self.t0 + self.wall_cap_s - self.reserve_s

    @property
    def hard_deadline(self) -> float:
        return self.t0 + self.wall_cap_s

    def check(self, point: str) -> None:
        if time.perf_counter() >= self.work_deadline:
            self.exceeded = True
            self.stopped_at = point
            raise BudgetExceeded(f"work deadline reached at {point}")

    def facts(self) -> dict:
        return {
            "t0_perf_counter": self.t0,
            "wall_cap_s": self.wall_cap_s,
            "reserve_s": self.reserve_s,
            "work_deadline_s": self.wall_cap_s - self.reserve_s,
            "hard_deadline_s": self.wall_cap_s,
            "measured_wall_s_at_summary": time.perf_counter() - self.t0,
            "budget_exceeded": self.exceeded,
            "stopped_at": self.stopped_at,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_members(path: Path) -> dict:
    """Header-only member listing: name -> (shape tuple, dtype str)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".npy"):
                continue
            with z.open(name) as f:
                ver = np.lib.format.read_magic(f)
                if ver[0] == 1:
                    shape, _, dt = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _, dt = np.lib.format.read_array_header_2_0(f)
            out[name[:-4]] = (tuple(shape), str(dt))
    return out


def _bitwise_ne_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.view(np.uint32) != b.view(np.uint32)


def _ratio(a: np.ndarray, ref: np.ndarray, atol: float, rtol: float):
    denom = atol + rtol * np.abs(ref.astype(np.float64))
    diff = np.abs(a.astype(np.float64) - ref.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, diff / np.where(denom > 0, denom, 1.0),
                         np.where(diff > 0, np.inf, 0.0))
    return ratio, diff


def _xp_fail(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    both_zero = (a64 == 0.0) & (b64 == 0.0)
    bad = np.abs(a64 - b64) > (atol + rtol * np.abs(b64))
    return bad & ~both_zero


def _zero_or_sign_disagree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    return ((a64 == 0.0) != (b64 == 0.0)) | ((a64 * b64) < 0.0)


def _safe_max(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.where(np.isfinite(x), x, np.inf).max())


@dataclass
class RowSpec:
    key: str
    model: str
    update: int
    parameter: str
    shape: tuple
    dtype: str
    offset: int
    numel: int


def _canonical_rows(rows: list, roster: dict) -> list:
    """Select the 528 adamw rows; validate the EXACT key set against roster."""
    selected = []
    for r in rows:
        key = r.get("key")
        if not isinstance(key, str) or not key.startswith("adamw:"):
            continue
        parts = key.split(":")
        if len(parts) != 4:
            raise N0Error(f"malformed adamw row key: {key!r}")
        _, model, u, parameter = parts
        if not (u.startswith("u") and u[1:].isdigit()):
            raise N0Error(f"non-integer update in row key {key!r}")
        selected.append((model, int(u[1:]), parameter, key, r))
    selected.sort(key=lambda t: (t[0], t[1], t[2]))
    keys = [t[3] for t in selected]
    if len(set(keys)) != len(keys):
        raise N0Error("duplicate adamw row keys in rows file")

    models = roster.get("models")
    if not isinstance(models, dict):
        raise N0Error("registration.roster.models missing")
    expected_keys = set()
    for model, info in models.items():
        params = info.get("parameters") if isinstance(info, dict) else None
        if not isinstance(params, list) or not params:
            raise N0Error(f"registration.roster.models.{model}.parameters missing")
        names = []
        for p in params:
            if isinstance(p, dict):
                name = p.get("name")
                if not isinstance(name, str):
                    raise N0Error(f"roster entry without name in {model}: {p!r}")
            elif isinstance(p, str):
                name = p
            else:
                raise N0Error(f"unsupported roster entry in {model}: {p!r}")
            names.append(name)
        if len(set(names)) != len(names):
            raise N0Error(f"duplicate roster parameters in {model}")
        for update in (1, 2):
            for name in names:
                expected_keys.add(f"adamw:{model}:u{update}:{name}")
    got_keys = set(keys)
    if got_keys != expected_keys:
        missing = sorted(expected_keys - got_keys)[:5]
        extra = sorted(got_keys - expected_keys)[:5]
        raise N0Error(
            f"row key set != roster x updates; missing={missing} extra={extra}")
    return selected


def parse_old_gap_key(key: str):
    """`i_adamw_gap:{model}:u{u}:{param}:{kind}` -> (row_key, kind) or None."""
    if not isinstance(key, str) or not key.startswith("i_adamw_gap:"):
        return None
    rest = key[len("i_adamw_gap:"):]
    parts = rest.split(":")
    if len(parts) != 4:
        return None
    model, u, parameter, kind = parts
    if not (u.startswith("u") and u[1:].isdigit()):
        return None
    return f"adamw:{model}:u{int(u[1:])}:{parameter}", kind


def compute_row_flags(a: dict, tols: dict, native_shape: tuple,
                      same_input_p: int, same_input_i: int) -> tuple:
    """Pure per-coordinate detector over flattened saved arrays.

    ``a`` must contain every I_SUFFIXES entry flattened plus ``p_theta_before``,
    ``p_theta_after`` (production bindings) and ``step`` (int).  Returns
    (flags uint32, anomalies list[str]).  Also used by N1 class C6 to feed
    deviation outputs through this exact detector.
    """
    n = a["theta_pre"].size
    flags = np.zeros(n, dtype=FLAGS_DTYPE)
    anomalies: list[str] = []

    def flag_anomaly(reason: str) -> None:
        anomalies.append(reason)

    expected_numel = int(np.prod([int(s) for s in native_shape])) if native_shape else 1
    if n != expected_numel:
        anomalies.append(f"shape_drift:saved_numel{n}:registered{native_shape}")

    for s in I_SUFFIXES:
        arr = a[s]
        if not isinstance(arr, np.ndarray):
            flag_anomaly(f"not_array:{s}")
            continue
        if arr.dtype != np.uint8 and not np.all(np.isfinite(arr)):
            flag_anomaly(f"nonfinite:{s}")
    if same_input_p != 1:
        flag_anomaly("same_input_formula_flag_false_p")
    if same_input_i != 1:
        flag_anomaly("same_input_formula_flag_false_i")
    nonfinite_inputs = any(x.startswith(("nonfinite", "not_array")) for x in anomalies)
    formula_gate_ok = same_input_p == 1 and same_input_i == 1

    theta_pre = a["theta_pre"]
    theta_p, theta_i = a["theta_p"], a["theta_i"]
    delta_ref_p = a["delta_ref_p"].astype(np.float64)
    delta_ref_i = a["delta_ref_i"].astype(np.float64)
    delta_meas_p = a["delta_measured_p"].astype(np.float64)
    delta_meas_i = a["delta_measured_i"].astype(np.float64)
    bound_p = a["bound_delta_p"].astype(np.float64)
    bound_i = a["bound_delta_i"].astype(np.float64)

    if not nonfinite_inputs:
        # Saved measured deltas must equal the float64 theta differences.
        rec_meas_p = theta_p.astype(np.float64) - theta_pre.astype(np.float64)
        rec_meas_i = theta_i.astype(np.float64) - theta_pre.astype(np.float64)
        if not np.array_equal(delta_meas_p, rec_meas_p):
            flag_anomaly("delta_measured_p_not_theta_diff")
        if not np.array_equal(delta_meas_i, rec_meas_i):
            flag_anomaly("delta_measured_i_not_theta_diff")

    if not nonfinite_inputs and formula_gate_ok:
        # Reconstruct the P-side formula-reference new theta from the saved
        # unrounded delta: delta_ref64 = (decay - pre) + h64 * ratio64 with
        # ratio64 an exact float32 value; a single f64 division recovers it,
        # and recomputing delta_ref64 confirms the reconstruction bit-exactly.
        step = int(a["step"])
        host = host_constants(step,
                              lr=float(a.get("lr", 1e-4)),
                              weight_decay=float(a.get("weight_decay", 1e-5)))
        decay = (theta_pre * np.float32(host["d_32"])).astype(np.float32)
        base = decay.astype(np.float64) - theta_pre.astype(np.float64)
        h64 = float(host["h_32"])
        ratio64 = (delta_ref_p - base) / h64
        r32 = _f32_array(ratio64)
        delta_check = base + h64 * r32.astype(np.float64)
        recon_ok = np.array_equal(delta_check, delta_ref_p)
        if not recon_ok:
            flag_anomaly("bound_delta_p_reconstruction_mismatch")
        else:
            pe_theta = _fma32_array(np.float32(host["h_32"]), r32, decay)[0]
            if not np.array_equal(bound_p, _half_ulp32(pe_theta)):
                flag_anomaly("bound_delta_p_not_half_ulp_of_p_reference_theta")
        if not np.array_equal(bound_i, _half_ulp32(theta_i)):
            flag_anomaly("bound_delta_i_not_half_ulp_of_theta_i")

        # Observability replay (audit uses the formula-reference new theta,
        # which equals the recorded theta on the same-input path).
        rtol_same = tols["delta_same_input"]
        obs_p_rec = ((delta_ref_p != 0.0)
                     & (bound_p <= 0.25 * rtol_same * np.abs(delta_ref_p))
                     & _bitwise_ne_f32(theta_p, theta_pre))
        obs_i_rec = ((delta_ref_i != 0.0)
                     & (bound_i <= 0.25 * rtol_same * np.abs(delta_ref_i))
                     & _bitwise_ne_f32(theta_i, theta_pre))
        ok_p_rec = np.where(delta_ref_p == 0.0, delta_meas_p == 0.0,
                            np.where(obs_p_rec,
                                     np.abs(delta_meas_p - delta_ref_p)
                                     <= rtol_same * np.abs(delta_ref_p), False))
        ok_i_rec = np.where(delta_ref_i == 0.0, delta_meas_i == 0.0,
                            np.where(obs_i_rec,
                                     np.abs(delta_meas_i - delta_ref_i)
                                     <= rtol_same * np.abs(delta_ref_i), False))
        if not np.array_equal(a["obs_p"].astype(bool), obs_p_rec):
            flag_anomaly("obs_p_recompute_mismatch")
        if not np.array_equal(a["obs_i"].astype(bool), obs_i_rec):
            flag_anomaly("obs_i_recompute_mismatch")
        if not np.array_equal(a["ok_p"].astype(bool), ok_p_rec):
            flag_anomaly("ok_p_recompute_mismatch")
        if not np.array_equal(a["ok_i"].astype(bool), ok_i_rec):
            flag_anomaly("ok_i_recompute_mismatch")

    obs_p = a["obs_p"].astype(bool)
    obs_i = a["obs_i"].astype(bool)
    ok_p = a["ok_p"].astype(bool)
    ok_i = a["ok_i"].astype(bool)

    with np.errstate(all="ignore"):
        flags |= np.where((delta_ref_p != 0.0) & ~obs_p, DELTA_UNOBSERVABLE_P, 0).astype(FLAGS_DTYPE)
        flags |= np.where((delta_ref_i != 0.0) & ~obs_i, DELTA_UNOBSERVABLE_I, 0).astype(FLAGS_DTYPE)
        flags |= np.where((delta_ref_p != 0.0) & (delta_meas_p == 0.0), DELTA_WRITE_ERASED_P, 0).astype(FLAGS_DTYPE)
        flags |= np.where((delta_ref_i != 0.0) & (delta_meas_i == 0.0), DELTA_WRITE_ERASED_I, 0).astype(FLAGS_DTYPE)
        flags |= np.where(obs_p & ~ok_p, DELTA_SAME_INPUT_GATE_FAIL_P, 0).astype(FLAGS_DTYPE)
        flags |= np.where(obs_i & ~ok_i, DELTA_SAME_INPUT_GATE_FAIL_I, 0).astype(FLAGS_DTYPE)
        flags |= np.where(((delta_ref_p == 0.0) & (delta_meas_p != 0.0))
                          | ((delta_ref_i == 0.0) & (delta_meas_i != 0.0)),
                          DELTA_ZERO_INVENTED, 0).astype(FLAGS_DTYPE)

        if not nonfinite_inputs:
            theta_fail = _xp_fail(theta_p, theta_i, tols["theta_atol"], tols["theta_rtol"])
            m_fail = _xp_fail(a["m_p"], a["m_i"], tols["state_atol"], tols["state_rtol"])
            v_fail = _xp_fail(a["v_p"], a["v_i"], tols["state_atol"], tols["state_rtol"])
            delta_mask = obs_p & obs_i
            delta_fail = delta_mask & (np.abs(delta_meas_p - delta_ref_i)
                                       > tols["delta_cross_path"] * np.abs(delta_ref_i))
            presence = (delta_ref_p != 0.0) != (delta_ref_i != 0.0)
            one_sided = obs_p != obs_i
            flags |= np.where(theta_fail, THETA_CROSS_PATH_GATE_FAIL, 0).astype(FLAGS_DTYPE)
            flags |= np.where(m_fail, M_CROSS_PATH_GATE_FAIL, 0).astype(FLAGS_DTYPE)
            flags |= np.where(v_fail, V_CROSS_PATH_GATE_FAIL, 0).astype(FLAGS_DTYPE)
            flags |= np.where(delta_fail, DELTA_CROSS_PATH_GATE_FAIL, 0).astype(FLAGS_DTYPE)
            flags |= np.where(presence, DELTA_PRESENCE_MISMATCH, 0).astype(FLAGS_DTYPE)
            flags |= np.where(one_sided, DELTA_ONE_SIDED_OBSERVABLE, 0).astype(FLAGS_DTYPE)
            flags |= np.where(_zero_or_sign_disagree(a["m_p"], a["m_i"]),
                              M_ZERO_OR_SIGN_DISAGREE, 0).astype(FLAGS_DTYPE)
            flags |= np.where(_zero_or_sign_disagree(a["v_p"], a["v_i"]),
                              V_ZERO_OR_SIGN_DISAGREE, 0).astype(FLAGS_DTYPE)
            flags |= np.where(_zero_or_sign_disagree(theta_p, theta_i),
                              THETA_ZERO_OR_SIGN_DISAGREE, 0).astype(FLAGS_DTYPE)

            slack = 1e-300

            def outside(x, lo, hi):
                return (x.astype(np.float64) < lo.astype(np.float64) - slack) | (
                    x.astype(np.float64) > hi.astype(np.float64) + slack)

            out_m = outside(a["m_p"], a["K_m_lo"], a["K_m_hi"]) | outside(a["m_i"], a["K_m_lo"], a["K_m_hi"])
            out_v = outside(a["v_p"], a["K_v_lo"], a["K_v_hi"]) | outside(a["v_i"], a["K_v_lo"], a["K_v_hi"])
            out_theta = outside(theta_p, a["K_theta_lo"], a["K_theta_hi"]) | outside(
                theta_i, a["K_theta_lo"], a["K_theta_hi"])
            out_delta = outside(delta_meas_p, a["K_delta_lo"], a["K_delta_hi"]) | outside(
                delta_meas_i, a["K_delta_lo"], a["K_delta_hi"]) | outside(
                delta_ref_p, a["K_delta_lo"], a["K_delta_hi"]) | outside(
                delta_ref_i, a["K_delta_lo"], a["K_delta_hi"])
            flags |= np.where(out_m | out_v | out_theta | out_delta,
                              INTERVAL_OUTSIDE, 0).astype(FLAGS_DTYPE)

    if not nonfinite_inputs:
        if not np.array_equal(a["i_theta_bind"], theta_i):
            flag_anomaly("i_theta_binding_mismatch")
        if not np.array_equal(a["i_m_bind"], a["m_i"]):
            flag_anomaly("i_m_binding_mismatch")
        if not np.array_equal(a["i_v_bind"], a["v_i"]):
            flag_anomaly("i_v_binding_mismatch")
        for name, saved, rec in (("p_theta_before", a["p_theta_before"], theta_pre),
                                 ("p_theta_after", a["p_theta_after"], theta_p)):
            if saved.dtype != rec.dtype or saved.shape != rec.shape:
                flag_anomaly(f"{name}_shape_or_dtype")
            elif _bitwise_ne_f32(saved, rec).any():
                flag_anomaly(f"{name}_binding_mismatch")

    if anomalies:
        flags |= FLAGS_DTYPE(SOURCE_ANOMALY)
    return flags, anomalies


def recompute_classification(flags: np.ndarray) -> tuple:
    any_same_input_fail = bool((flags & (DELTA_SAME_INPUT_GATE_FAIL_P | DELTA_SAME_INPUT_GATE_FAIL_I)).any())
    any_zero_invented = bool((flags & DELTA_ZERO_INVENTED).any())
    any_precision = bool((flags & (DELTA_UNOBSERVABLE_P | DELTA_UNOBSERVABLE_I
                                   | DELTA_WRITE_ERASED_P | DELTA_WRITE_ERASED_I)).any())
    any_xp = bool((flags & (THETA_CROSS_PATH_GATE_FAIL | M_CROSS_PATH_GATE_FAIL
                            | V_CROSS_PATH_GATE_FAIL | DELTA_CROSS_PATH_GATE_FAIL
                            | DELTA_PRESENCE_MISMATCH | DELTA_ONE_SIDED_OBSERVABLE)).any())
    outside = bool((flags & INTERVAL_OUTSIDE).any())
    if any_same_input_fail or any_zero_invented:
        return "direct_failure", {"parameter_delta_precision_gap"} if any_precision else set()
    if any_xp:
        kind = "cross_path_bound_unverifiable" if outside else "cross_path_sensitivity_gap"
        gaps = {kind}
        if any_precision:
            gaps.add("parameter_delta_precision_gap")
        return kind, gaps
    if any_precision:
        return "audit_incomplete", {"parameter_delta_precision_gap"}
    return "passed", set()


class N0Analyzer:
    def __init__(self, run_root: Path, output_dir: Path, budget: N0Budget,
                 analysis_path: Path | None = None):
        self.run_root = Path(run_root)
        self.output_dir = Path(output_dir)
        self.budget = budget
        self.paths = {
            "registration": self.run_root / "registration/attempt_003/registration.json",
            "p_report": self.run_root / "production/attempt_003/p_report.json",
            "p_npz": self.run_root / "production/attempt_003/p_deterministic.npz",
            "p_npz_diffusion": self.run_root / "production/attempt_003/p_diffusion.npz",
            "i_rows": self.run_root / "independent/attempt_002/i_rows.json",
            "i_report": self.run_root / "independent/attempt_002/i_report.json",
            "i_npz": self.run_root / "independent/attempt_002/i_deterministic.npz",
            "i_npz_diffusion": self.run_root / "independent/attempt_002/i_diffusion.npz",
        }
        self.analysis_path = Path(analysis_path) if analysis_path is not None else Path(
            "derived/audit/w1_real_training_update_audit/"
            "gradient_numerics_retest/real_analysis/attempt_001/analysis.json")
        self.notes: list[str] = []

    # ------------------------------------------------------------------ load
    def load_inputs(self) -> dict:
        for name, p in self.paths.items():
            if not p.exists():
                raise N0Error(f"missing required input {name}: {p}")
        reg = json.loads(self.paths["registration"].read_text())
        rows = json.loads(self.paths["i_rows"].read_text())
        if not isinstance(rows, list):
            raise N0Error("i_rows.json must be a list")
        roster = reg.get("roster")
        if not isinstance(roster, dict):
            raise N0Error("registration.roster missing")
        cfg = reg.get("config")
        if not (isinstance(cfg, dict) and isinstance(cfg.get("lr"), (int, float))
                and isinstance(cfg.get("weight_decay"), (int, float))):
            raise N0Error("registration.config.lr/weight_decay missing")
        self.lr = float(cfg["lr"])
        self.weight_decay = float(cfg["weight_decay"])
        tols = reg.get("tolerances")
        if not isinstance(tols, dict):
            raise N0Error("registration.tolerances missing")

        def _gate_tol(key: str) -> tuple:
            v = tols.get(key)
            if not (isinstance(v, dict) and isinstance(v.get("atol"), (int, float))
                    and isinstance(v.get("rtol"), (int, float))):
                raise N0Error(f"registration.tolerances.{key} must be {{atol, rtol}}")
            return float(v["atol"]), float(v["rtol"])

        same_atol, same_rtol = _gate_tol("delta_same_input")
        delta_atol, delta_rtol = _gate_tol("delta_cross_path")
        theta_atol, theta_rtol = _gate_tol("theta_cross_path")
        state_atol, state_rtol = _gate_tol("state_cross_path")
        if same_atol != 0.0 or delta_atol != 0.0:
            raise N0Error("delta gates require atol=0 in the frozen contract")
        self.tols = {
            "delta_same_input": same_rtol,
            "delta_cross_path": delta_rtol,
            "theta_atol": theta_atol, "theta_rtol": theta_rtol,
            "state_atol": state_atol, "state_rtol": state_rtol,
        }
        selected = _canonical_rows(rows, roster)

        # Sealed-anchor cross-check: the sealed P/I reports bind the NPZ file
        # hashes; drift here means the evidence root changed after sealing.
        shas = {name: sha256_file(p) for name, p in self.paths.items()}
        for report_name, npz_names in (
                ("i_report", ("i_npz", "i_npz_diffusion")),
                ("p_report", ("p_npz", "p_npz_diffusion"))):
            report = json.loads(self.paths[report_name].read_text())
            models = report.get("models")
            if not isinstance(models, dict):
                raise N0Error(f"{report_name}.models missing")
            for model, npz_name in zip(("deterministic", "diffusion"), npz_names):
                entry = models.get(model)
                if not isinstance(entry, dict) or not isinstance(entry.get("arrays_sha256"), str):
                    raise N0Error(f"{report_name}.models.{model}.arrays_sha256 missing")
                if entry["arrays_sha256"] != shas[npz_name]:
                    raise N0Error(
                        f"{report_name} sealed anchor mismatch for {model}: "
                        f"recorded {entry['arrays_sha256'][:16]} != current "
                        f"{shas[npz_name][:16]}")
        return {"registration": reg, "rows": rows, "selected": selected, "shas": shas}

    # ------------------------------------------------------- header prevalidation
    def prevalidate(self, selected: list) -> list:
        """Entry check (counted in budget): shapes/dtypes from NPZ headers only."""
        headers = {}
        for model in ("deterministic", "diffusion"):
            suffix = "" if model == "deterministic" else "_diffusion"
            headers[model] = npz_members(self.paths[f"i_npz{suffix}"])
            headers[f"p_{model}"] = npz_members(self.paths[f"p_npz{suffix}"])
        specs: list[RowSpec] = []
        offset = 0
        total = 0
        for model, update, parameter, key, row in selected:
            prefix = f"adamw_{model}_u{update}_{parameter}"
            shape = None
            for suf in I_SUFFIXES:
                got = headers[model].get(f"{prefix}/{suf}")
                if got is None:
                    raise N0Error(f"missing i-NPZ member {prefix}/{suf}")
                if shape is None:
                    shape = got[0]
                elif got[0] != shape:
                    raise N0Error(f"shape mismatch among i-NPZ members of {prefix}")
            for st in I_STATE_SUFFIXES:
                got = headers[model].get(f"{st}/u{update}/{parameter}")
                if got is None:
                    raise N0Error(f"missing i-NPZ member {st}/u{update}/{parameter}")
                if got[0] != shape:
                    raise N0Error(f"shape mismatch on {st}/u{update}/{parameter}")
            for sc in I_SCALAR_SUFFIXES:
                got = headers[model].get(f"{prefix}/{sc}")
                if got is None:
                    raise N0Error(f"missing i-NPZ member {prefix}/{sc}")
                if got[0] != () and got[0] != (1,):
                    raise N0Error(f"{prefix}/{sc} must be a scalar array")
            numel = int(np.prod(shape)) if shape else 1
            total += numel
            if total > MAX_TOTAL_COORDS:
                raise N0Error(
                    f"total coordinate count {total} exceeds frozen cap "
                    f"{MAX_TOTAL_COORDS}; N0 must not start")
            dtypes = {"theta_pre": "float32", "theta_p": "float32", "theta_i": "float32",
                      "m_p": "float32", "v_p": "float32", "m_i": "float32", "v_i": "float32",
                      "obs_p": "uint8", "obs_i": "uint8", "ok_p": "uint8", "ok_i": "uint8",
                      "same_input_p": "uint8", "same_input_i": "uint8"}
            for suf in I_SUFFIXES:
                want = dtypes.get(suf, "float64")
                got = headers[model][f"{prefix}/{suf}"]
                if got[1] != want:
                    raise N0Error(f"{prefix}/{suf} dtype {got[1]} != {want}")
            for sc in I_SCALAR_SUFFIXES:
                if headers[model][f"{prefix}/{sc}"][1] != "uint8":
                    raise N0Error(f"{prefix}/{sc} dtype must be uint8")
            p_key = f"theta_before/u{update}/{parameter}"
            p_key2 = f"theta_after/u{update}/{parameter}"
            for pk in (p_key, p_key2):
                if pk not in headers[f"p_{model}"]:
                    raise N0Error(f"missing p-NPZ member {pk}")
            specs.append(RowSpec(key=key, model=model, update=update,
                                 parameter=parameter, shape=tuple(int(s) for s in shape),
                                 dtype="float32", offset=offset, numel=numel))
            offset += numel
        return specs

    # ------------------------------------------------------------- row analysis
    def analyze_row(self, spec: RowSpec) -> tuple:
        model, u, parameter = spec.model, spec.update, spec.parameter
        prefix = f"adamw_{model}_u{u}_{parameter}"
        suffix = "" if model == "deterministic" else "_diffusion"
        with np.load(self.paths[f"i_npz{suffix}"]) as z:
            native_shape = tuple(int(s) for s in z[f"{prefix}/theta_pre"].shape)
            a = {s: np.asarray(z[f"{prefix}/{s}"]).reshape(-1) for s in I_SUFFIXES}
            a["i_theta_bind"] = np.asarray(z[f"i_theta/u{u}/{parameter}"]).reshape(-1)
            a["i_m_bind"] = np.asarray(z[f"i_m/u{u}/{parameter}"]).reshape(-1)
            a["i_v_bind"] = np.asarray(z[f"i_v/u{u}/{parameter}"]).reshape(-1)
            same_input_p = int(np.asarray(z[f"{prefix}/same_input_p"]).reshape(()))
            same_input_i = int(np.asarray(z[f"{prefix}/same_input_i"]).reshape(()))
        with np.load(self.paths[f"p_npz{suffix}"]) as z:
            a["p_theta_before"] = np.asarray(z[f"theta_before/u{u}/{parameter}"]).reshape(-1)
            a["p_theta_after"] = np.asarray(z[f"theta_after/u{u}/{parameter}"]).reshape(-1)
        a["step"] = u
        a["lr"] = self.lr
        a["weight_decay"] = self.weight_decay
        flags, anomalies = compute_row_flags(a, self.tols, native_shape,
                                             same_input_p, same_input_i)
        return flags, a, anomalies

    # ------------------------------------------------------------------- outputs
    def run(self) -> dict:
        self.budget.check("entry")
        loaded = self.load_inputs()
        self.budget.check("inputs_loaded")
        specs = self.prevalidate(loaded["selected"])
        self.budget.check("header_prevalidation")

        total = sum(s.numel for s in specs)
        flags_all = np.zeros(total, dtype=FLAGS_DTYPE)
        row_records: list[dict] = []

        # Old-chain cross-reference (read-only; keys are i_adamw_gap:*:kind).
        analysis = json.loads(self.analysis_path.read_text())
        old_state: dict = {}
        new_only_keys: set = set()
        for item in analysis.get("gaps", {}).get("items", []):
            parsed = parse_old_gap_key(item.get("key"))
            if parsed is None:
                self.notes.append(f"unparsed_old_gap_key:{item.get('key')}")
                continue
            old_state[parsed] = item.get("state")
        for item in analysis.get("gaps", {}).get("new_only_items", []):
            parsed = parse_old_gap_key(item.get("key"))
            if parsed is None:
                self.notes.append(f"unparsed_new_only_key:{item.get('key')}")
                continue
            new_only_keys.add(parsed)

        extrema_csv = io.StringIO()
        extrema_writer = csv.DictWriter(extrema_csv, fieldnames=EXTREMA_COLUMNS)
        extrema_writer.writeheader()
        n_extrema_records = 0
        label_counts: dict[str, int] = {}

        try:
            for idx, spec in enumerate(specs):
                self.budget.check(f"row[{idx}]:{spec.key}")
                flags, arrs, row_anomalies = self.analyze_row(spec)
                flags_all[spec.offset:spec.offset + spec.numel] = flags

                saved = loaded["selected"][idx][4]
                saved_cls = saved.get("classification")
                saved_gaps = set(saved.get("gaps") or [])
                recomp_cls, recomp_gaps = recompute_classification(flags)
                classification_match = (saved_cls == recomp_cls and saved_gaps == recomp_gaps)
                if not classification_match:
                    row_anomalies.append(
                        f"classification_mismatch:saved={saved_cls}/{sorted(saved_gaps)}:"
                        f"recomputed={recomp_cls}/{sorted(recomp_gaps)}")
                    flags_all[spec.offset:spec.offset + spec.numel] |= FLAGS_DTYPE(SOURCE_ANOMALY)
                    flags = flags_all[spec.offset:spec.offset + spec.numel]

                precision_mask = (flags & (DELTA_UNOBSERVABLE_P | DELTA_UNOBSERVABLE_I
                                           | DELTA_WRITE_ERASED_P | DELTA_WRITE_ERASED_I
                                           | DELTA_SAME_INPUT_GATE_FAIL_P | DELTA_SAME_INPUT_GATE_FAIL_I
                                           | DELTA_ZERO_INVENTED)) != 0
                sensitivity_mask = (flags & (THETA_CROSS_PATH_GATE_FAIL | M_CROSS_PATH_GATE_FAIL
                                             | V_CROSS_PATH_GATE_FAIL | DELTA_CROSS_PATH_GATE_FAIL
                                             | DELTA_PRESENCE_MISMATCH | DELTA_ONE_SIDED_OBSERVABLE
                                             | M_ZERO_OR_SIGN_DISAGREE | V_ZERO_OR_SIGN_DISAGREE
                                             | THETA_ZERO_OR_SIGN_DISAGREE
                                             | INTERVAL_OUTSIDE)) != 0
                n_prec = int(precision_mask.sum())
                n_sens = int(sensitivity_mask.sum())
                n_anom = int(((flags & SOURCE_ANOMALY) != 0).sum())
                n_out = int(((flags & INTERVAL_OUTSIDE) != 0).sum())
                n_presence = int(((flags & DELTA_PRESENCE_MISMATCH) != 0).sum())
                n_onesided = int(((flags & DELTA_ONE_SIDED_OBSERVABLE) != 0).sum())

                with np.errstate(all="ignore"):
                    same_ratio_p, abs_err_p = _ratio(arrs["delta_measured_p"], arrs["delta_ref_p"],
                                                     0.0, self.tols["delta_same_input"])
                    same_ratio_i, abs_err_i = _ratio(arrs["delta_measured_i"], arrs["delta_ref_i"],
                                                     0.0, self.tols["delta_same_input"])
                    theta_ratio, _ = _ratio(arrs["theta_p"], arrs["theta_i"],
                                            self.tols["theta_atol"], self.tols["theta_rtol"])
                    m_ratio, _ = _ratio(arrs["m_p"], arrs["m_i"], self.tols["state_atol"], self.tols["state_rtol"])
                    v_ratio, _ = _ratio(arrs["v_p"], arrs["v_i"], self.tols["state_atol"], self.tols["state_rtol"])
                    delta_xp_ratio, _ = _ratio(arrs["delta_measured_p"], arrs["delta_ref_i"],
                                               0.0, self.tols["delta_cross_path"])
                max_same = max(_safe_max(same_ratio_p), _safe_max(same_ratio_i))
                max_delta_ratio = _safe_max(delta_xp_ratio)
                max_theta_ratio = _safe_max(theta_ratio)
                max_m_ratio = _safe_max(m_ratio)
                max_v_ratio = _safe_max(v_ratio)

                labels: list[str] = []
                has_structural = any(x.startswith(("nonfinite", "not_array", "shape_drift"))
                                     for x in row_anomalies)
                if has_structural:
                    labels.append("evidence_incomplete")
                elif row_anomalies:
                    labels.append("implementation_issue_candidate")
                if bool((flags & (DELTA_SAME_INPUT_GATE_FAIL_P | DELTA_SAME_INPUT_GATE_FAIL_I
                                  | DELTA_ZERO_INVENTED)).any()):
                    labels.append("implementation_issue_candidate")
                if bool((flags & INTERVAL_OUTSIDE).any()):
                    labels.append("implementation_issue_candidate")
                has_precision_bits = bool((flags & (DELTA_UNOBSERVABLE_P | DELTA_UNOBSERVABLE_I
                                                    | DELTA_WRITE_ERASED_P | DELTA_WRITE_ERASED_I)).any())
                has_xp_bits = bool((flags & (THETA_CROSS_PATH_GATE_FAIL | M_CROSS_PATH_GATE_FAIL
                                             | V_CROSS_PATH_GATE_FAIL | DELTA_CROSS_PATH_GATE_FAIL
                                             | DELTA_PRESENCE_MISMATCH | DELTA_ONE_SIDED_OBSERVABLE)).any())
                if has_precision_bits and "implementation_issue_candidate" not in labels \
                        and "evidence_incomplete" not in labels:
                    labels.append("precision_limited")
                if has_xp_bits and not bool((flags & INTERVAL_OUTSIDE).any()) \
                        and "implementation_issue_candidate" not in labels \
                        and "evidence_incomplete" not in labels:
                    labels.append("cross_path_sensitive")
                labels = sorted(set(labels)) or ["no_gap"]

                has_precision_gap = "parameter_delta_precision_gap" in recomp_gaps
                has_sensitivity_gap = ("cross_path_sensitivity_gap" in recomp_gaps
                                       or "cross_path_bound_unverifiable" in recomp_gaps)
                gap = ("both" if has_precision_gap and has_sensitivity_gap else
                       "precision" if has_precision_gap else
                       "sensitivity" if has_sensitivity_gap else "none")
                for lab in labels:
                    label_counts[lab] = label_counts.get(lab, 0) + 1

                old_p = old_state.get((spec.key, "parameter_delta_precision_gap"), "")
                old_s = old_state.get((spec.key, "cross_path_sensitivity_gap"), "")
                old_note = "new_only" if ((spec.key, "parameter_delta_precision_gap") in new_only_keys
                                          or (spec.key, "cross_path_sensitivity_gap") in new_only_keys) else ""

                row_records.append({
                    "key": spec.key, "model": spec.model, "update": spec.update,
                    "parameter": spec.parameter, "shape": "x".join(map(str, spec.shape)),
                    "gap": gap, "n_coords": spec.numel, "n_precision": n_prec,
                    "n_sensitivity": n_sens, "n_source_anomaly": n_anom,
                    "n_interval_outside": n_out, "n_presence_mismatch": n_presence,
                    "n_one_sided_observable": n_onesided,
                    "labels": "|".join(labels),
                    "saved_classification": saved_cls,
                    "recomputed_classification": recomp_cls,
                    "classification_match": classification_match,
                    "old_state_precision": old_p,
                    "old_state_sensitivity": old_s,
                    "old_new_only": old_note,
                    "max_same_delta_ratio": max_same, "max_delta_ratio": max_delta_ratio,
                    "max_theta_ratio": max_theta_ratio, "max_m_ratio": max_m_ratio,
                    "max_v_ratio": max_v_ratio,
                })

                flat = np.arange(spec.numel)
                for category, mask in (("precision", precision_mask), ("sensitivity", sensitivity_mask)):
                    if not mask.any():
                        row_records[-1][f"n_extrema_{category}"] = 0
                        continue
                    idxs = flat[mask]
                    with np.errstate(all="ignore"):
                        if category == "precision":
                            score = np.maximum(same_ratio_p, same_ratio_i)
                            err = np.maximum(abs_err_p, abs_err_i)
                        else:
                            score = np.maximum.reduce([theta_ratio, m_ratio, v_ratio, delta_xp_ratio])
                            err = np.zeros_like(score)
                    score = np.where(np.isfinite(score), score, np.inf)
                    order = np.lexsort((idxs, -err[idxs], -score[idxs]))
                    take = idxs[order[: MAX_EXTREMA_PER_ROW_CATEGORY]]
                    row_records[-1][f"n_extrema_{category}"] = int(len(idxs))
                    for fi in take:
                        rec = {
                            "row_key": spec.key, "category": category,
                            "model": spec.model, "update": spec.update,
                            "parameter": spec.parameter, "flat_index": int(fi),
                            "coord_index": repr(np.unravel_index(int(fi), spec.shape)),
                            "bits_hex": f"0x{int(flags[fi]):06x}",
                            "score": f"{float(score[fi]):.17g}",
                            "max_ratio": f"{float(score[fi]):.17g}",
                            "max_abs_err": f"{float(err[fi]):.17g}",
                            "same_ratio_p": f"{float(same_ratio_p[fi]):.17g}",
                            "same_ratio_i": f"{float(same_ratio_i[fi]):.17g}",
                            "delta_ref_p": f"{float(arrs['delta_ref_p'][fi]):.17g}",
                            "delta_ref_i": f"{float(arrs['delta_ref_i'][fi]):.17g}",
                            "delta_measured_p": f"{float(arrs['delta_measured_p'][fi]):.17g}",
                            "delta_measured_i": f"{float(arrs['delta_measured_i'][fi]):.17g}",
                            "bound_delta_p": f"{float(arrs['bound_delta_p'][fi]):.17g}",
                            "bound_delta_i": f"{float(arrs['bound_delta_i'][fi]):.17g}",
                            "theta_ratio": f"{float(theta_ratio[fi]):.17g}",
                            "m_ratio": f"{float(m_ratio[fi]):.17g}",
                            "v_ratio": f"{float(v_ratio[fi]):.17g}",
                            "delta_xp_ratio": f"{float(delta_xp_ratio[fi]):.17g}",
                            "theta_p": f"{float(arrs['theta_p'][fi]):.17g}",
                            "theta_i": f"{float(arrs['theta_i'][fi]):.17g}",
                            "interval_outside": int(bool(flags[fi] & INTERVAL_OUTSIDE)),
                            "source_anomaly": int(bool(flags[fi] & SOURCE_ANOMALY)),
                        }
                        extrema_writer.writerow(rec)
                        n_extrema_records += 1
        except BudgetExceeded:
            pass

        if not self.budget.exceeded:
            self.budget.check("rows_complete")

        # ---------------------------------------------------------------- writes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        flags_path = self.output_dir / "coordinate_flags.npy"
        np.save(flags_path, flags_all)
        _fsync_dir(self.output_dir)
        flags_sha = sha256_file(flags_path)
        flags_bytes = flags_path.stat().st_size
        if flags_bytes > MAX_TOTAL_COORDS * 4 + FLAGS_NPY_HEADER_RESERVE:
            raise N0Error(f"flags file {flags_bytes} exceeds frozen bound")

        index = {
            "schema": SCHEMA,
            "bit_definitions": {BIT_NAMES[b]: (1 << b) for b in BIT_NAMES},
            "flags_dtype": "uint32",
            "row_order_rule": "(model, update, parameter) ascending; model lexicographic",
            "max_total_coords": MAX_TOTAL_COORDS,
            "total_coords": int(total),
            "n_rows": len(specs),
            "sources": {k: {"path": str(v), "sha256": loaded["shas"][k]}
                        for k, v in self.paths.items()},
            "analysis_path": str(self.analysis_path),
            "analysis_sha256": sha256_file(self.analysis_path),
            "i_suffixes": list(I_SUFFIXES),
            "i_scalar_suffixes": list(I_SCALAR_SUFFIXES),
            "i_state_suffixes": list(I_STATE_SUFFIXES),
            "p_crosscheck": list(P_CROSSCHECK),
            "tolerances_consumed": self.tols,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "rows": [
                {"key": s.key, "model": s.model, "update": s.update,
                 "parameter": s.parameter, "shape": list(s.shape),
                 "dtype": s.dtype, "offset": s.offset, "numel": s.numel,
                 "i_prefix": f"adamw_{s.model}_u{s.update}_{s.parameter}",
                 "i_state_prefix": f"i_{{theta,m,v}}/u{s.update}/{s.parameter}",
                 "p_prefixes": [f"theta_before/u{s.update}/{s.parameter}",
                                f"theta_after/u{s.update}/{s.parameter}"]}
                for s in specs
            ],
        }
        index_path = self.output_dir / "coordinate_index.json"
        index_path.write_text(json.dumps(index, indent=1))
        if index_path.stat().st_size > MAX_INDEX_BYTES:
            raise N0Error("coordinate_index.json exceeds frozen bound")

        gap_rows_path = self.output_dir / "gap_rows.csv"
        with open(gap_rows_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GAP_ROWS_COLUMNS)
            w.writeheader()
            for rec in row_records:
                w.writerow(rec)
        if gap_rows_path.stat().st_size > MAX_GAP_ROWS_CSV_BYTES:
            raise N0Error("gap_rows.csv exceeds frozen bound")

        extrema_path = self.output_dir / "gap_coordinates_extrema.csv"
        extrema_path.write_text(extrema_csv.getvalue())
        if extrema_path.stat().st_size > MAX_EXTREMA_CSV_BYTES:
            raise N0Error("gap_coordinates_extrema.csv exceeds frozen bound")
        _fsync_dir(self.output_dir)

        rows_precision = sum(1 for r in row_records if r["gap"] in ("precision", "both"))
        rows_sensitivity = sum(1 for r in row_records if r["gap"] in ("sensitivity", "both"))
        rows_both = sum(1 for r in row_records if r["gap"] == "both")

        summary = {
            "schema": SCHEMA + "_summary",
            "total_coords": int(total),
            "n_rows": len(specs),
            "event_replay": {
                "precision_rows": rows_precision,
                "sensitivity_rows": rows_sensitivity,
                "both_rows": rows_both,
                "gap_rows": rows_precision + rows_sensitivity - rows_both,
            },
            "label_counts": label_counts,
            "n_extrema_records": n_extrema_records,
            "flags_sha256": flags_sha,
            "flags_bytes": flags_bytes,
            "outputs": {
                "coordinate_flags.npy": flags_sha,
                "coordinate_index.json": sha256_file(index_path),
                "gap_rows.csv": sha256_file(gap_rows_path),
                "gap_coordinates_extrema.csv": sha256_file(extrema_path),
            },
            "budget": self.budget.facts(),
            "notes": self.notes,
            "n_rows_with_anomalies": sum(1 for r in row_records if r["n_source_anomaly"] > 0),
        }
        complete = (not self.budget.exceeded) and all(
            r["classification_match"] for r in row_records)
        summary["coverage_complete"] = bool(complete and summary["n_rows_with_anomalies"] == 0)
        summary["label_verdict"] = ("complete" if summary["coverage_complete"]
                                    else "evidence_incomplete/resource_limit")

        # Hard-deadline and return-coverage seal: costs only increase; a slow
        # write that crosses the hard deadline revokes the completion claim.
        summary_path = self.output_dir / "summary.json"
        _write_summary_sealed(summary_path, summary, self.budget)
        return summary


def _fsync_dir(path: Path) -> None:
    fd = __import__("os").open(path, __import__("os").O_RDONLY)
    try:
        __import__("os").fsync(fd)
    finally:
        __import__("os").close(fd)


def _write_summary_sealed(summary_path: Path, summary: dict, budget: N0Budget) -> None:
    """Write summary, then verify the write is covered by billed cost and
    inside the hard deadline; a violation rewrites the summary as incomplete
    (costs only increase, safety direction) with a final wall sample."""
    now = time.perf_counter()
    summary["budget"]["measured_wall_s_at_summary"] = now - budget.t0
    summary["seal"] = {
        "hard_deadline_s": budget.hard_deadline,
        "measured_wall_s_before_summary_write": now - budget.t0,
    }
    summary_path.write_text(json.dumps(summary, indent=1))
    _fsync_dir(summary_path.parent)
    final = time.perf_counter() - budget.t0
    summary["seal"]["measured_wall_s_final"] = final
    if final > budget.hard_deadline:
        summary["budget_exceeded_after_seal"] = True
        summary["coverage_complete"] = False
        summary["label_verdict"] = "budget_exceeded_after_seal"
    summary["budget"]["measured_wall_s_at_summary"] = final
    summary_path.write_text(json.dumps(summary, indent=1))
    _fsync_dir(summary_path.parent)
