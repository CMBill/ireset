"""W1 N0: coordinate-level finite analysis of saved gradient-numerics gap evidence.

Plan: docs/plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md (v0.4) section 4.

Read-only CPU analysis over the saved T6 evidence root. No model construction,
no forward/backward/FK/AdamW execution, no checkpoint loading, no new real
numerics beyond float64 CPU arithmetic on saved arrays.

Frozen output format ``n0_coordinate_flags_v1``:

- Row universe: the 528 rows in ``i_rows.json`` whose key starts with
  ``adamw:``, in canonical order (model, update, parameter) with model
  lexicographic (deterministic < diffusion), update ascending, parameter
  lexicographic.
- ``coordinate_flags.npy``: one ``uint16`` per coordinate, rows concatenated in
  row order, each row expanded C-order over its native NPZ shape.
- Bit definitions (one-hot, frozen before first execution):

  = ==== ============================== =====================================================
  b  hex name                         meaning (per coordinate)
  = ==== ============================== =====================================================
  0 0x0001 DELTA_UNOBSERVABLE_P         delta_ref_p != 0 and saved/derived obs_p is False
  1 0x0002 DELTA_UNOBSERVABLE_I         delta_ref_i != 0 and saved/derived obs_i is False
  2 0x0004 DELTA_WRITE_ERASED_P         delta_ref_p != 0 and delta_measured_p == 0
  3 0x0008 DELTA_WRITE_ERASED_I         delta_ref_i != 0 and delta_measured_i == 0
  4 0x0010 DELTA_SAME_INPUT_GATE_FAIL_P  obs_p and not ok_p (same-input relative gate, rtol=5e-4)
  5 0x0020 DELTA_SAME_INPUT_GATE_FAIL_I  obs_i and not ok_i
  6 0x0040 DELTA_ZERO_INVENTED           (delta_ref == 0) and (delta_measured != 0), either side
  7 0x0080 THETA_CROSS_PATH_GATE_FAIL    |theta_p - theta_i| > atol + rtol*|theta_i|
  8 0x0100 M_CROSS_PATH_GATE_FAIL        same form on (m_p, m_i)
  9 0x0200 V_CROSS_PATH_GATE_FAIL        same form on (v_p, v_i)
 10 0x0400 DELTA_CROSS_PATH_GATE_FAIL    |delta_measured_p - delta_ref_i| > rtol*|delta_ref_i| on obs mask
 11 0x0800 M_ZERO_OR_SIGN_DISAGREE       (m_p==0)!=(m_i==0) or opposite nonzero signs
 12 0x1000 V_ZERO_OR_SIGN_DISAGREE       same on v
 13 0x2000 THETA_ZERO_OR_SIGN_DISAGREE   same on theta
 14 0x4000 INTERVAL_OUTSIDE              any of m/v/theta measured+reference outside saved K_* interval
 15 0x8000 SOURCE_ANOMALY                recomputed ok/obs/half-ULP/i-state binding/theta cross-NPZ
                                         binding disagrees with saved bytes, nonfinite input, shape
                                         or dtype mismatch, or row classification not reproducible
  = ==== ============================== =====================================================

- Row labels derived from flags only (analysis labels, NOT pass exemptions):
  ``evidence_incomplete`` (structural failure) > ``implementation_issue_candidate``
  (SOURCE_ANOMALY, same-input gate fail on observable coordinates, zero-invented
  delta, or INTERVAL_OUTSIDE) > ``precision_limited`` (only unobservable/erased
  write coordinates) and/or ``cross_path_sensitive`` (cross-path gate fails with
  every failing coordinate inside the saved propagation intervals).
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

SCHEMA = "n0_coordinate_flags_v1"

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

# Per-quantity cross-path gate tolerances (must equal registration.tolerances).
XP_GATE_QUANTITIES = ("theta", "m", "v")
# Saved-array suffixes consumed from i NPZ per row (adamw_{model}_u{u}_{param}/SUFFIX).
I_SUFFIXES = (
    "theta_pre", "theta_p", "theta_i", "m_p", "v_p", "m_i", "v_i",
    "delta_ref_p", "delta_ref_i", "delta_measured_p", "delta_measured_i",
    "bound_delta_p", "bound_delta_i",
    "K_m_lo", "K_m_hi", "K_v_lo", "K_v_hi", "K_theta_lo", "K_theta_hi",
    "K_delta_lo", "K_delta_hi",
    "obs_p", "obs_i", "ok_p", "ok_i",
)
# Cross-binding suffixes: same values saved again under i_theta/i_m/i_v keys.
I_STATE_SUFFIXES = ("i_theta", "i_m", "i_v")
# Production NPZ per-update keys cross-checked bit-for-bit against i NPZ.
P_CROSSCHECK = ("theta_before", "theta_after")

# Capacity bounds (plan section 4), conservative bytes.
MAX_TOTAL_COORDS = 16_000_000
FLAGS_NPY_HEADER_RESERVE = 1024
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_GAP_ROWS_CSV_BYTES = 8 * 1024 * 1024
MAX_EXTREMA_CSV_BYTES = 64 * 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_EXTREMA_PER_ROW_CATEGORY = 16
MAX_EXTREMA_RECORD_BYTES = 2048
WORK_DEADLINE_SLACK_S = 0.0

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
    "labels", "saved_classification", "recomputed_classification",
    "classification_match", "old_state_precision", "old_state_sensitivity",
    "max_same_delta_ratio", "max_delta_ratio", "max_theta_ratio",
    "max_m_ratio", "max_v_ratio", "n_extrema_precision",
    "n_extrema_sensitivity",
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
        if time.perf_counter() >= self.work_deadline - WORK_DEADLINE_SLACK_S:
            self.exceeded = True
            self.stopped_at = point
            raise BudgetExceeded(f"work deadline reached at {point}")

    def facts(self) -> dict:
        now = time.perf_counter()
        return {
            "t0_perf_counter": self.t0,
            "wall_cap_s": self.wall_cap_s,
            "reserve_s": self.reserve_s,
            "work_deadline_s": self.wall_cap_s - self.reserve_s,
            "hard_deadline_s": self.wall_cap_s,
            "measured_wall_s_at_summary": now - self.t0,
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


def _half_ulp32(x: np.ndarray) -> np.ndarray:
    """Half of the larger adjacent float32 spacing around x, in float64."""
    xf = x.astype(np.float64)
    up = np.nextafter(x, np.float32(np.inf)).astype(np.float64)
    down = np.nextafter(x, np.float32(-np.inf)).astype(np.float64)
    spacing = np.maximum(xf - down, up - xf)
    return spacing / 2.0


def _ratio(a: np.ndarray, ref: np.ndarray, atol: float, rtol: float):
    """Elementwise |a-ref|/(atol+rtol*|ref|); inf where ref==0 and a!=0."""
    denom = atol + rtol * np.abs(ref.astype(np.float64))
    diff = np.abs(a.astype(np.float64) - ref.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, diff / np.where(denom > 0, denom, 1.0),
                         np.where(diff > 0, np.inf, 0.0))
    return ratio, diff


def _xp_fail(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    """Elementwise cross-path gate failure; both-exactly-zero passes."""
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    both_zero = (a64 == 0.0) & (b64 == 0.0)
    bad = np.abs(a64 - b64) > (atol + rtol * np.abs(b64))
    return bad & ~both_zero


def _zero_or_sign_disagree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    zero_dis = (a64 == 0.0) != (b64 == 0.0)
    sign_dis = (a64 * b64) < 0.0
    return zero_dis | sign_dis


def _safe_max(x: np.ndarray) -> float:
    """max with nonfinite treated as +inf; 0.0 for empty."""
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


def _canonical_rows(rows: list, expect_counts: dict) -> list:
    """Select and order the 528 adamw rows; validate multiset vs expectation."""
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
        update = int(u[1:])
        selected.append((model, update, parameter, key, r))
    selected.sort(key=lambda t: (t[0], t[1], t[2]))
    counts = {}
    for model, update, _p, _k, _r in selected:
        counts[model] = counts.get(model, 0) + 1
    for model, n in expect_counts.items():
        if counts.get(model, 0) * 0 != 0 or counts.get(model, 0) != n * 2:
            raise N0Error(
                f"row multiset mismatch for model {model}: got {counts.get(model, 0)} "
                f"(= {n} params x 2 updates expected)")
    keys = [t[3] for t in selected]
    if len(set(keys)) != len(keys):
        raise N0Error("duplicate adamw row keys in rows file")
    return selected


class N0Analyzer:
    def __init__(self, run_root: Path, output_dir: Path, budget: N0Budget,
                 analysis_path: Path | None = None):
        self.run_root = Path(run_root)
        self.output_dir = Path(output_dir)
        self.budget = budget
        self.paths = {
            "registration": self.run_root / "registration/attempt_003/registration.json",
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
        self.anomalies: list[str] = []
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
        n_det, n_diff = roster.get("n_det"), roster.get("n_diff")
        if not isinstance(n_det, int) or not isinstance(n_diff, int):
            raise N0Error("registration.roster.n_det/n_diff missing")
        tols = reg.get("tolerances")
        if not isinstance(tols, dict):
            raise N0Error("registration.tolerances missing")

        def _gate_tol(key: str) -> tuple:
            v = tols.get(key)
            if not (isinstance(v, dict)
                    and isinstance(v.get("atol"), (int, float))
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
        selected = _canonical_rows(rows, {"deterministic": n_det, "diffusion": n_diff})
        shas = {name: sha256_file(p) for name, p in self.paths.items()}
        return {"registration": reg, "rows": rows, "selected": selected, "shas": shas}

    # ------------------------------------------------------- header prevalidation
    def prevalidate(self, selected: list) -> list:
        """Entry check (counted in budget): shapes/dtypes from NPZ headers only."""
        headers = {}
        for model in ("deterministic", "diffusion"):
            headers[model] = npz_members(self.paths[f"i_npz{'' if model == 'deterministic' else '_diffusion'}"])
            headers[f"p_{model}"] = npz_members(self.paths[f"p_npz{'' if model == 'deterministic' else '_diffusion'}"])
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
            numel = int(np.prod(shape)) if shape else 1
            total += numel
            if total > MAX_TOTAL_COORDS:
                raise N0Error(
                    f"total coordinate count {total} exceeds frozen cap "
                    f"{MAX_TOTAL_COORDS}; N0 must not start")
            for suf in I_SUFFIXES:
                if f"{prefix}/{suf}" not in headers[model]:
                    raise N0Error(f"missing i-NPZ member {prefix}/{suf}")
            for st in I_STATE_SUFFIXES:
                if f"{st}/u{update}/{parameter}" not in headers[model]:
                    raise N0Error(f"missing i-NPZ member {st}/u{update}/{parameter}")
            dtypes = {"theta_pre": "float32", "theta_p": "float32", "theta_i": "float32",
                      "m_p": "float32", "v_p": "float32", "m_i": "float32", "v_i": "float32",
                      "obs_p": "uint8", "obs_i": "uint8", "ok_p": "uint8", "ok_i": "uint8"}
            for suf in I_SUFFIXES:
                want = dtypes.get(suf, "float64")
                got = headers[model][f"{prefix}/{suf}"]
                if got[1] != want:
                    raise N0Error(f"{prefix}/{suf} dtype {got[1]} != {want}")
            p_key = f"theta_before/u{update}/{parameter}"
            if p_key not in headers[f"p_{model}"]:
                raise N0Error(f"missing p-NPZ member {p_key}")
            p_key2 = f"theta_after/u{update}/{parameter}"
            if p_key2 not in headers[f"p_{model}"]:
                raise N0Error(f"missing p-NPZ member {p_key2}")
            specs.append(RowSpec(key=key, model=model, update=update,
                                 parameter=parameter, shape=tuple(int(s) for s in shape),
                                 dtype="float32", offset=offset, numel=numel))
            offset += numel
        return specs

    # ------------------------------------------------------------- row analysis
    def analyze_row(self, spec: RowSpec) -> tuple:
        """Recompute per-coordinate flags from saved arrays; cross-check saved bytes."""
        model, u, parameter = spec.model, spec.update, spec.parameter
        prefix = f"adamw_{model}_u{u}_{parameter}"
        with np.load(self.paths[f"i_npz{'' if model == 'deterministic' else '_diffusion'}"]) as z:
            native_shape = tuple(int(s) for s in z[f"{prefix}/theta_pre"].shape)
            a = {s: np.asarray(z[f"{prefix}/{s}"]).reshape(-1) for s in I_SUFFIXES}
            a["i_theta_bind"] = np.asarray(z[f"i_theta/u{u}/{parameter}"]).reshape(-1)
            a["i_m_bind"] = np.asarray(z[f"i_m/u{u}/{parameter}"]).reshape(-1)
            a["i_v_bind"] = np.asarray(z[f"i_v/u{u}/{parameter}"]).reshape(-1)
        with np.load(self.paths[f"p_npz{'' if model == 'deterministic' else '_diffusion'}"]) as z:
            p_theta_before = np.asarray(z[f"theta_before/u{u}/{parameter}"]).reshape(-1)
            p_theta_after = np.asarray(z[f"theta_after/u{u}/{parameter}"]).reshape(-1)

        n = spec.numel
        flags = np.zeros(n, dtype=np.uint16)
        row_anomalies: list[str] = []

        def flag_anomaly(reason: str) -> None:
            row_anomalies.append(reason)

        shape = native_shape
        expected = spec.shape
        if shape != expected:
            flag_anomaly(f"shape_drift:saved{shape}:registered{expected}")
        # Nonfinite inputs (float arrays only; obs/ok are uint8).
        for s in I_SUFFIXES:
            arr = a[s]
            if not isinstance(arr, np.ndarray):
                flag_anomaly(f"not_array:{s}")
                continue
            if arr.dtype != np.uint8 and not np.all(np.isfinite(arr)):
                flag_anomaly(f"nonfinite:{s}")
        nonfinite_inputs = any(x.startswith("nonfinite") for x in row_anomalies)

        theta_pre = a["theta_pre"]
        theta_p, theta_i = a["theta_p"], a["theta_i"]
        delta_ref_p = a["delta_ref_p"].astype(np.float64)
        delta_ref_i = a["delta_ref_i"].astype(np.float64)
        delta_meas_p = a["delta_measured_p"].astype(np.float64)
        delta_meas_i = a["delta_measured_i"].astype(np.float64)
        bound_p = a["bound_delta_p"].astype(np.float64)
        bound_i = a["bound_delta_i"].astype(np.float64)

        if not nonfinite_inputs:
            # Recompute observability (identifiability mask) per side.
            rtol_same = self.tols["delta_same_input"]
            obs_p_rec = ((delta_ref_p != 0.0)
                         & (bound_p <= 0.25 * rtol_same * np.abs(delta_ref_p))
                         & _bitwise_ne_f32(theta_p, theta_pre))
            obs_i_rec = ((delta_ref_i != 0.0)
                         & (bound_i <= 0.25 * rtol_same * np.abs(delta_ref_i))
                         & _bitwise_ne_f32(theta_i, theta_pre))
            ok_p_rec = np.where(delta_ref_p == 0.0,
                                delta_meas_p == 0.0,
                                np.where(obs_p_rec,
                                         np.abs(delta_meas_p - delta_ref_p)
                                         <= rtol_same * np.abs(delta_ref_p),
                                         False))
            ok_i_rec = np.where(delta_ref_i == 0.0,
                                delta_meas_i == 0.0,
                                np.where(obs_i_rec,
                                         np.abs(delta_meas_i - delta_ref_i)
                                         <= rtol_same * np.abs(delta_ref_i),
                                         False))
            if not np.array_equal(a["obs_p"].astype(bool), obs_p_rec):
                flag_anomaly("obs_p_recompute_mismatch")
            if not np.array_equal(a["obs_i"].astype(bool), obs_i_rec):
                flag_anomaly("obs_i_recompute_mismatch")
            if not np.array_equal(a["ok_p"].astype(bool), ok_p_rec):
                flag_anomaly("ok_p_recompute_mismatch")
            if not np.array_equal(a["ok_i"].astype(bool), ok_i_rec):
                flag_anomaly("ok_i_recompute_mismatch")
            # Half-ULP write bound cross-check (exact in float64).
            if not np.allclose(bound_p, _half_ulp32(theta_i), rtol=0, atol=0):
                flag_anomaly("bound_delta_p_not_half_ulp_of_theta_i")
            if not np.allclose(bound_i, _half_ulp32(theta_i), rtol=0, atol=0):
                flag_anomaly("bound_delta_i_not_half_ulp_of_theta_i")

        obs_p = a["obs_p"].astype(bool)
        obs_i = a["obs_i"].astype(bool)
        ok_p = a["ok_p"].astype(bool)
        ok_i = a["ok_i"].astype(bool)

        # Precision bits.
        flags |= np.where((delta_ref_p != 0.0) & ~obs_p, DELTA_UNOBSERVABLE_P, 0).astype(np.uint16)
        flags |= np.where((delta_ref_i != 0.0) & ~obs_i, DELTA_UNOBSERVABLE_I, 0).astype(np.uint16)
        flags |= np.where((delta_ref_p != 0.0) & (delta_meas_p == 0.0), DELTA_WRITE_ERASED_P, 0).astype(np.uint16)
        flags |= np.where((delta_ref_i != 0.0) & (delta_meas_i == 0.0), DELTA_WRITE_ERASED_I, 0).astype(np.uint16)
        flags |= np.where(obs_p & ~ok_p, DELTA_SAME_INPUT_GATE_FAIL_P, 0).astype(np.uint16)
        flags |= np.where(obs_i & ~ok_i, DELTA_SAME_INPUT_GATE_FAIL_I, 0).astype(np.uint16)
        flags |= np.where(((delta_ref_p == 0.0) & (delta_meas_p != 0.0))
                          | ((delta_ref_i == 0.0) & (delta_meas_i != 0.0)),
                          DELTA_ZERO_INVENTED, 0).astype(np.uint16)

        # Cross-path gate bits.
        theta_fail = np.zeros(n, dtype=bool)
        with np.errstate(all="ignore"):
            if not nonfinite_inputs:
                theta_fail = _xp_fail(theta_p, theta_i, self.tols["theta_atol"], self.tols["theta_rtol"])
            m_fail = _xp_fail(a["m_p"], a["m_i"], self.tols["state_atol"], self.tols["state_rtol"])
            v_fail = _xp_fail(a["v_p"], a["v_i"], self.tols["state_atol"], self.tols["state_rtol"])
            delta_mask = obs_p & obs_i
            with np.errstate(divide="ignore", invalid="ignore"):
                delta_fail = delta_mask & (np.abs(delta_meas_p - delta_ref_i)
                                           > self.tols["delta_cross_path"] * np.abs(delta_ref_i))
            flags |= np.where(theta_fail, THETA_CROSS_PATH_GATE_FAIL, 0).astype(np.uint16)
            flags |= np.where(m_fail, M_CROSS_PATH_GATE_FAIL, 0).astype(np.uint16)
            flags |= np.where(v_fail, V_CROSS_PATH_GATE_FAIL, 0).astype(np.uint16)
            flags |= np.where(delta_fail, DELTA_CROSS_PATH_GATE_FAIL, 0).astype(np.uint16)
            flags |= np.where(_zero_or_sign_disagree(a["m_p"], a["m_i"]),
                              M_ZERO_OR_SIGN_DISAGREE, 0).astype(np.uint16)
            flags |= np.where(_zero_or_sign_disagree(a["v_p"], a["v_i"]),
                              V_ZERO_OR_SIGN_DISAGREE, 0).astype(np.uint16)
            flags |= np.where(_zero_or_sign_disagree(theta_p, theta_i),
                              THETA_ZERO_OR_SIGN_DISAGREE, 0).astype(np.uint16)

            # Saved propagation intervals (1e-300 slack, as in the audit).
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
            flags |= np.where(out_m | out_v | out_theta | out_delta, INTERVAL_OUTSIDE, 0).astype(np.uint16)

        # Cross-bindings: i_theta/i_m/i_v duplicates and production NPZ.
        if not nonfinite_inputs:
            if not np.array_equal(a["i_theta_bind"], theta_i):
                flag_anomaly("i_theta_binding_mismatch")
            if not np.array_equal(a["i_m_bind"], a["m_i"]):
                flag_anomaly("i_m_binding_mismatch")
            if not np.array_equal(a["i_v_bind"], a["v_i"]):
                flag_anomaly("i_v_binding_mismatch")
            if p_theta_before.dtype != theta_pre.dtype or p_theta_before.shape != theta_pre.shape:
                flag_anomaly("p_theta_before_shape_or_dtype")
            elif not _bitwise_ne_f32(p_theta_before, theta_pre).sum() == 0:
                flag_anomaly("p_theta_before_binding_mismatch")
            if p_theta_after.dtype != theta_p.dtype or p_theta_after.shape != theta_p.shape:
                flag_anomaly("p_theta_after_shape_or_dtype")
            elif not _bitwise_ne_f32(p_theta_after, theta_p).sum() == 0:
                flag_anomaly("p_theta_after_binding_mismatch")

        if row_anomalies:
            flags |= np.uint16(SOURCE_ANOMALY)

        return flags, a, row_anomalies

    # ------------------------------------------------------------- classification
    def recompute_classification(self, flags: np.ndarray) -> tuple:
        any_same_input_fail = bool((flags & (DELTA_SAME_INPUT_GATE_FAIL_P | DELTA_SAME_INPUT_GATE_FAIL_I)).any())
        any_zero_invented = bool((flags & DELTA_ZERO_INVENTED).any())
        any_precision = bool((flags & (DELTA_UNOBSERVABLE_P | DELTA_UNOBSERVABLE_I
                                       | DELTA_WRITE_ERASED_P | DELTA_WRITE_ERASED_I)).any())
        any_xp = bool((flags & (THETA_CROSS_PATH_GATE_FAIL | M_CROSS_PATH_GATE_FAIL
                                | V_CROSS_PATH_GATE_FAIL | DELTA_CROSS_PATH_GATE_FAIL)).any())
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

    # ------------------------------------------------------------------- outputs
    def run(self) -> dict:
        self.budget.check("entry")
        loaded = self.load_inputs()
        self.budget.check("inputs_loaded")
        specs = self.prevalidate(loaded["selected"])
        self.budget.check("header_prevalidation")

        total = sum(s.numel for s in specs)
        flags_all = np.zeros(total, dtype=np.uint16)
        row_records: list[dict] = []

        # Old-chain cross-reference (read-only, no rewriting).
        analysis = json.loads(self.analysis_path.read_text())
        old_state = {}
        for item in analysis.get("gaps", {}).get("items", []):
            old_state[(item.get("key"), item.get("kind"))] = item.get("state")

        extrema_csv = io.StringIO()
        extrema_writer = csv.DictWriter(extrema_csv, fieldnames=EXTREMA_COLUMNS)
        extrema_writer.writeheader()
        n_extrema_records = 0

        precision_total = sensitivity_total = both_total = 0
        label_counts: dict[str, int] = {}

        try:
            for idx, spec in enumerate(specs):
                self.budget.check(f"row[{idx}]:{spec.key}")
                flags, arrs, row_anomalies = self.analyze_row(spec)
                flags_all[spec.offset:spec.offset + spec.numel] = flags

                saved = loaded["selected"][idx][4]
                saved_cls = saved.get("classification")
                saved_gaps = set(saved.get("gaps") or [])
                recomp_cls, recomp_gaps = self.recompute_classification(flags)
                classification_match = (saved_cls == recomp_cls and saved_gaps == recomp_gaps)
                if not classification_match:
                    row_anomalies.append(
                        f"classification_mismatch:saved={saved_cls}/{sorted(saved_gaps)}:"
                        f"recomputed={recomp_cls}/{sorted(recomp_gaps)}")
                    flags_all[spec.offset:spec.offset + spec.numel] |= np.uint16(SOURCE_ANOMALY)
                    flags = flags_all[spec.offset:spec.offset + spec.numel]

                # Per-category coordinate counts and ratios.
                precision_mask = (flags & (DELTA_UNOBSERVABLE_P | DELTA_UNOBSERVABLE_I
                                           | DELTA_WRITE_ERASED_P | DELTA_WRITE_ERASED_I
                                           | DELTA_SAME_INPUT_GATE_FAIL_P | DELTA_SAME_INPUT_GATE_FAIL_I
                                           | DELTA_ZERO_INVENTED)) != 0
                sensitivity_mask = (flags & (THETA_CROSS_PATH_GATE_FAIL | M_CROSS_PATH_GATE_FAIL
                                             | V_CROSS_PATH_GATE_FAIL | DELTA_CROSS_PATH_GATE_FAIL
                                             | M_ZERO_OR_SIGN_DISAGREE | V_ZERO_OR_SIGN_DISAGREE
                                             | THETA_ZERO_OR_SIGN_DISAGREE
                                             | INTERVAL_OUTSIDE)) != 0
                n_prec = int(precision_mask.sum())
                n_sens = int(sensitivity_mask.sum())
                n_anom = int(((flags & SOURCE_ANOMALY) != 0).sum())
                n_out = int(((flags & INTERVAL_OUTSIDE) != 0).sum())

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

                # Labels from flags only (analysis labels, not pass exemptions).
                labels = []
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
                                             | V_CROSS_PATH_GATE_FAIL | DELTA_CROSS_PATH_GATE_FAIL)).any())
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
                if gap in ("precision", "both"):
                    precision_total += 1
                if gap in ("sensitivity", "both"):
                    sensitivity_total += 1
                if gap == "both":
                    both_total += 1
                for lab in labels:
                    label_counts[lab] = label_counts.get(lab, 0) + 1

                row_records.append({
                    "key": spec.key, "model": spec.model, "update": spec.update,
                    "parameter": spec.parameter, "shape": "x".join(map(str, spec.shape)),
                    "gap": gap, "n_coords": spec.numel, "n_precision": n_prec,
                    "n_sensitivity": n_sens, "n_source_anomaly": n_anom,
                    "n_interval_outside": n_out, "labels": "|".join(labels),
                    "saved_classification": saved_cls,
                    "recomputed_classification": recomp_cls,
                    "classification_match": classification_match,
                    "old_state_precision": old_state.get((spec.key, "parameter_delta_precision_gap"), ""),
                    "old_state_sensitivity": old_state.get((spec.key, "cross_path_sensitivity_gap"), ""),
                    "max_same_delta_ratio": max_same, "max_delta_ratio": max_delta_ratio,
                    "max_theta_ratio": max_theta_ratio, "max_m_ratio": max_m_ratio,
                    "max_v_ratio": max_v_ratio,
                })

                # Extrema per category (predefined ordering; not result-cherry-picked).
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
                            "bits_hex": f"0x{int(flags[fi]):04x}",
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
        flags_sha = sha256_file(flags_path)
        flags_bytes = flags_path.stat().st_size
        if flags_bytes > MAX_TOTAL_COORDS * 2 + FLAGS_NPY_HEADER_RESERVE:
            raise N0Error(f"flags file {flags_bytes} exceeds frozen bound")

        index = {
            "schema": SCHEMA,
            "bit_definitions": {BIT_NAMES[b]: (1 << b) for b in BIT_NAMES},
            "row_order_rule": "(model, update, parameter) ascending; model lexicographic",
            "max_total_coords": MAX_TOTAL_COORDS,
            "total_coords": int(total),
            "n_rows": len(specs),
            "sources": {k: {"path": str(v), "sha256": loaded["shas"][k]}
                        for k, v in self.paths.items()},
            "i_suffixes": list(I_SUFFIXES),
            "i_state_suffixes": list(I_STATE_SUFFIXES),
            "p_crosscheck": list(P_CROSSCHECK),
            "tolerances_consumed": self.tols,
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

        # Event-set replay check (350 precision / 189 sensitivity / 392 / 147).
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
        }
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=1))
        if summary_path.stat().st_size > MAX_REPORT_BYTES:
            raise N0Error("summary.json exceeds frozen bound")

        complete = (not self.budget.exceeded) and all(
            r["classification_match"] for r in row_records)
        summary["n_rows_with_anomalies"] = sum(
            1 for r in row_records if r["n_source_anomaly"] > 0)
        summary["coverage_complete"] = bool(complete and summary["n_rows_with_anomalies"] == 0)
        summary["label_verdict"] = ("complete" if complete else "evidence_incomplete/resource_limit")
        summary_path.write_text(json.dumps(summary, indent=1))
        return summary
