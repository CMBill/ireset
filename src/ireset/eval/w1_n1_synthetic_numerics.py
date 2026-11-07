"""W1 N1: minimal synthetic verification and limited numerics-closure evidence.

Plan: docs/plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md (v0.4) section 5.

Pure-CPU synthetic arithmetic over the audited effective-float32 AdamW node
graph and its interval propagation (imported unchanged from
``w1_real_training_update_audit``).  No real network, optimizer, checkpoint,
dataset or production CUDA path is constructed or claimed equivalent.

Six classes x at most 8 cases (48 total); per case a vector of at most 16
coordinates, at most 2 steps and 2 comparison paths; per path per step at most
one effective-float32 formula evaluation and one float64 mathematical
reference evaluation (<=384 vector formula evaluations overall).  Actual case,
evaluation and state-advance counts are registered as nonzero JSON integers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ireset.eval.w1_n0_coordinate_flags import (BudgetExceeded,  # noqa: F401
                                                DELTA_PRESENCE_MISMATCH,
                                                DELTA_ONE_SIDED_OBSERVABLE,
                                                INTERVAL_OUTSIDE,
                                                MAX_REPORT_BYTES,
                                                N0Budget,
                                                SOURCE_ANOMALY,
                                                compute_row_flags,
                                                recompute_classification,
                                                _write_json_fsync,
                                                _write_summary_sealed,
                                                verify_return_coverage)
from ireset.eval.w1_real_training_update_audit import (
    _cross_path_flags,
    _delta_observability,
    _f32_array,
    _half_ulp32,
    _inside,
    _zero_state_problems,
    adamw_effective_fp32,
    adamw_math_reference,
    effective_input_budget,
    host_constants,
    propagate_intervals,
)

SCHEMA = "n1_synthetic_numerics_v1"
CLASSES = (
    "same_input_formula_bitwise",
    "nonzero_write_erasure",
    "one_sided_observable",
    "state_zero_sign_problems",
    "cancellation_within_intervals",
    "interval_violation_deviation",
)
MAX_CASES_PER_CLASS = 8
MAX_TOTAL_CASES = 48
MAX_COORDS_PER_CASE = 16
MAX_STEPS_PER_CASE = 2
DELTA_SAME_INPUT_RTOL = 5e-4
STATE_SPEC = {"atol": 0.0, "rtol": 2e-3}
THETA_SPEC = {"atol": 2e-7, "rtol": 2e-6}
SEED = 20261107

STATE_ADVANCE_NODES = ("theta", "m", "v")


class N1Error(RuntimeError):
    pass


@dataclass
class N1Counters:
    cases: int = 0
    fp32_formula_evals: int = 0
    float64_reference_evals: int = 0
    interval_propagations: int = 0
    state_advances: int = 0

    def as_dict(self) -> dict:
        return {
            "cases": self.cases,
            "fp32_formula_evals": self.fp32_formula_evals,
            "float64_reference_evals": self.float64_reference_evals,
            "interval_propagations": self.interval_propagations,
            "state_advances": self.state_advances,
        }


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _independent_fp32_adamw(theta, m, v, g, step, host):
    """Independent hand-written effective-float32 node graph (N1 C1 contrast).

    Written from the plan's node list, not copied from the audited function:
    z=R32(g-m), m=R32(a*z+m), v=R32(c*g^2+b2*v), s=sqrtf(v),
    b=R32(s/bc2), den=R32(b+eps), ratio=R32(m/den),
    theta=R32(h*ratio+theta_decay), theta_decay=R32(theta*d).
    FMA nodes are emulated as exact-product-plus-sum rounded once in float64
    (this is the naive emulation; the audited function adds exact-rational
    mid-point fallbacks, so nodes with recorded fallbacks are excluded from
    the bitwise comparison by the caller).
    """
    theta32 = _f32(theta)
    g32 = _f32(g)
    m_old = _f32(m)
    v_old = _f32(v)
    a32 = np.float32(host["a_32"])
    b2 = np.float32(host["b2_32"])
    c32 = np.float32(host["one_minus_b2_32"])
    h32 = np.float32(host["h_32"])
    d32 = np.float32(host["d_32"])
    eps32 = np.float32(host["eps_32"])
    bc2 = np.float32(host["bc2_sqrt_32"])

    theta_decay = (theta32 * d32).astype(np.float32)
    z = (g32 - m_old).astype(np.float32)
    m_new = (a32.astype(np.float64) * z.astype(np.float64)
             + m_old.astype(np.float64)).astype(np.float32)
    v_scaled = (b2 * v_old).astype(np.float32)
    g2 = (g32 * g32).astype(np.float32)
    v_new = (c32.astype(np.float64) * g2.astype(np.float64)
             + v_scaled.astype(np.float64)).astype(np.float32)
    s = np.sqrt(v_new.astype(np.float64)).astype(np.float32)
    b = (s.astype(np.float64) / bc2.astype(np.float64)).astype(np.float32)
    denom = (b + eps32).astype(np.float32)
    ratio = (m_new.astype(np.float64) / denom.astype(np.float64)).astype(np.float32)
    theta_new = (h32.astype(np.float64) * ratio.astype(np.float64)
                 + theta_decay.astype(np.float64)).astype(np.float32)
    delta_ref64 = ((theta_decay.astype(np.float64) - theta32.astype(np.float64))
                   + float(h32) * ratio.astype(np.float64))
    return {"theta": theta_new, "m": m_new, "v": v_new, "s": s, "b": b,
            "denom": denom, "ratio": ratio, "delta_ref64": delta_ref64}


def _f16vec(specs) -> np.ndarray:
    v = _f32(specs)
    if v.size > MAX_COORDS_PER_CASE:
        raise N1Error("case vector exceeds 16 coordinates")
    return v


def _bitwise_eq(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


class CaseRunner:
    def __init__(self, budget: N0Budget):
        self.budget = budget
        self.counters = N1Counters()
        self.cases: list[dict] = []

    def record(self, cls: str, cid: str, expected: dict, observed: dict,
               passed: bool, note: str = "") -> None:
        self.counters.cases += 1
        if self.counters.cases > MAX_TOTAL_CASES:
            raise N1Error("total case cap exceeded")
        self.cases.append({
            "class": cls, "id": cid, "expected": expected,
            "observed": observed, "passed": bool(passed), "note": note,
        })

    # ------------------------------------------------------------ class C1
    def c1_same_input_formula(self) -> None:
        """Audited formula vs an independent hand-written node graph.

        A same-function double call has no discriminating power; here the
        second path is an independent implementation written from the plan's
        node list, so a misunderstood node order/dtype would diverge bitwise.
        Nodes where the audited path recorded exact-rational mid-point
        fallbacks are excluded (the naive FMA emulation legitimately differs
        there) and the exclusion is recorded per case.
        """
        cls = "same_input_formula_bitwise"
        g_profiles = [
            np.zeros(16, np.float32),
            _f16vec([1e-38, -1e-38, 1e-40, -1e-40, 1.4e-45, -1.4e-45, 1e-30, -1e-30,
                     1e-20, 1e-10, 1e-6, 1e-3, 1.0, -1.0, 3.0, -3.0]),
            _f16vec([2.0 ** k for k in range(-24, -8)]),
            _f16vec([1.0 + 2.0 ** k for k in range(-24, -8)]),
            _f16vec([-(2.0 ** k) for k in range(-8, 8)]),
            _f16vec([1e-30, 1e-20, 1e-10, 1.0, 1e10, 1e20, 1e30, -1e-30,
                     -1e-20, -1e-10, -1.0, -1e10, 5e-324 / 1e300, 0.0, 0.0, 0.0]),
            _f16vec([1e-12, -1e-12, 1e-11, -1e-11, 1e-10, -1e-10, 1e-9, -1e-9,
                     1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6, 1e-5, -1e-5]),
        ]
        rng = np.random.default_rng(SEED)
        g_profiles.append((rng.uniform(-1.0, 1.0, 16) * 10.0
                           ** rng.integers(-30, 3, 16)).astype(np.float32))
        for pi, g1 in enumerate(g_profiles):
            self.budget.check(f"c1.case[{pi}]")
            theta = _f32(np.full(16, 0.5) * (1.0 + pi * 0.25))
            m_old = _f32(np.full(16, 1e-4) * (pi + 1))
            v_old = _f32(np.full(16, 1e-8) * (pi + 1))
            bitwise_ok = True
            max_math_rel = 0.0
            excluded_nodes: list[str] = []
            for step in (1, 2):
                aud = adamw_effective_fp32(theta, m_old, v_old, g1, step)
                ind = _independent_fp32_adamw(theta, m_old, v_old, g1, step,
                                              host_constants(step))
                self.counters.fp32_formula_evals += 2
                ref = adamw_math_reference(theta, m_old, v_old, g1, step)
                self.counters.float64_reference_evals += 1
                if step == 2:
                    self.counters.state_advances += 1
                fb = aud["rounding_fallbacks"]
                fb_nodes = {k for k, cnt in fb.items() if cnt}
                node_map = {"m": "m", "v": "v", "b": "b", "ratio": "ratio",
                            "theta": "theta", "sqrt": "s", "denom": "denom"}
                for node, key in node_map.items():
                    if node in fb_nodes:
                        excluded_nodes.append(f"step{step}:{node}")
                        continue
                    if not _bitwise_eq(np.asarray(aud[key], np.float32),
                                       np.asarray(ind[key], np.float32)):
                        bitwise_ok = False
                if "ratio" not in fb_nodes and not np.array_equal(
                        aud["delta_ref64"], ind["delta_ref64"]):
                    bitwise_ok = False
                denom64 = np.abs(ref["theta"])
                max_math_rel = max(max_math_rel, float(
                    np.max(np.abs(aud["theta"].astype(np.float64) - ref["theta"])
                           / np.where(denom64 > 0, denom64, 1.0))))
                theta, m_old, v_old = aud["theta"], aud["m"], aud["v"]
            self.record(cls, f"profile_{pi}",
                        expected={"bitwise_audited_vs_independent": True},
                        observed={"bitwise_audited_vs_independent": bitwise_ok,
                                  "max_rel_vs_float64_reference": max_math_rel,
                                  "fallback_excluded_nodes": excluded_nodes},
                        passed=bitwise_ok)

    # ------------------------------------------------------------ class C2
    def c2_write_erasure(self) -> None:
        cls = "nonzero_write_erasure"
        # (theta0, ratio target, expect_erased, expect_observable)
        cfgs = [
            (1.0, 1e-4, True, False),     # delta ~1e-8 < half-ULP(1)=2.98e-8: erased
            (1.0, 1e-3, False, False),    # delta ~6e-8: changed but below identifiability
            (1.0, 5.0, False, True),      # delta ~4e-4: changed and observable
            (1024.0, 1e-1, True, False),  # half-ULP(1024)=6.1e-5: erased
            (-1.0, 1e-4, True, False),
            (1.0, 0.0, True, False),      # zero ref, zero measured: pass control
            (0.5, 1e-4, True, False),
            (2.0 ** -126, 1e-2, False, True),
        ]
        for ci, (theta0, ratio_target, exp_erased, exp_obs) in enumerate(cfgs):
            self.budget.check(f"c2.case[{ci}]")
            step = 1000
            v_old = _f32(np.full(1, 1.0))
            # m chosen so ratio = m/(sqrt(v)/bc2+eps) ~ ratio_target; g = m keeps
            # z = g - m == 0 so the momentum state does not move.
            host = host_constants(step)
            m_target = ratio_target * (np.float32(host["bc2_sqrt_32"])
                                       + np.float32(host["eps_32"]))
            m_old = _f32(np.full(1, float(m_target)))
            g = m_old.copy()
            out = adamw_effective_fp32(theta0, m_old, v_old, g, step)
            self.counters.fp32_formula_evals += 1
            theta_before = _f32([theta0])
            theta_changed = not _bitwise_eq(out["theta"], theta_before)
            erased = not theta_changed
            measured = out["theta"].astype(np.float64) - theta_before.astype(np.float64)
            bound = _half_ulp32(out["theta"])
            obs, ok = _delta_observability(out["delta_ref64"], measured, bound,
                                           out["theta"], theta_before,
                                           gate_rtol=DELTA_SAME_INPUT_RTOL)
            passed = (erased == exp_erased) and (bool(obs[0]) == exp_obs)
            self.record(cls, f"case_{ci}",
                        expected={"theta_changed": not exp_erased,
                                  "observable": exp_obs,
                                  "semantics": "precision" if not exp_obs else "ok"},
                        observed={"theta_changed": bool(theta_changed),
                                  "observable": bool(obs[0]),
                                  "delta_ref": float(out["delta_ref64"][0]),
                                  "bound": float(bound[0])},
                        passed=passed)

    # ------------------------------------------------------------ class C3
    def c3_one_sided_observable(self) -> None:
        cls = "one_sided_observable"
        # (theta_p, theta_i, exp_one_sided) with delta_ref target = -1e-2
        cfgs = [
            (1.0, 1024.0, True),    # obs_p True, obs_i False (write erased at 1024)
            (1024.0, 1.0, True),    # mirrored
            (1.0, 1.0, False),      # both observable control
            (1024.0, 1024.0, False),
            (1.0, 2.0 ** 20, True),
            (2.0 ** -20, 1.0, False),
            (1.0, 4096.0, True),
            (1.0, 0.0, False),
        ]
        for ci, (tp, ti, exp_one_sided) in enumerate(cfgs):
            self.budget.check(f"c3.case[{ci}]")
            step = 1000
            v_old = _f32(np.full(1, 1.0))
            # delta_ref target = -1e-2: ratio = 100 via m = 100*denom; g = m keeps
            # z = 0 so the momentum state does not move.
            host = host_constants(step)
            m_old = _f32(np.full(1, 100.0 * (np.float32(host["bc2_sqrt_32"])
                                             + np.float32(host["eps_32"]))))
            g = m_old.copy()
            p = adamw_effective_fp32(tp, m_old, v_old, g, step)
            i = adamw_effective_fp32(ti, m_old, v_old, g, step)
            self.counters.fp32_formula_evals += 2
            obs_p, ok_p = _delta_observability(
                p["delta_ref64"], p["theta"].astype(np.float64) - _f32([tp]).astype(np.float64),
                _half_ulp32(p["theta"]), p["theta"], _f32([tp]), gate_rtol=DELTA_SAME_INPUT_RTOL)
            obs_i, ok_i = _delta_observability(
                i["delta_ref64"], i["theta"].astype(np.float64) - _f32([ti]).astype(np.float64),
                _half_ulp32(i["theta"]), i["theta"], _f32([ti]), gate_rtol=DELTA_SAME_INPUT_RTOL)
            # theta_new == theta_before on the erased side gives obs False.
            flags = _cross_path_flags(
                p["m"], i["m"], p["v"], i["v"], p["theta"], i["theta"],
                p["theta"].astype(np.float64) - _f32([tp]).astype(np.float64),
                p["delta_ref64"], i["delta_ref64"], obs_p, obs_i,
                STATE_SPEC, delta_rtol=DELTA_SAME_INPUT_RTOL)
            one_sided = flags["delta_one_sided_observable"]
            passed = (one_sided == exp_one_sided)
            if exp_one_sided:
                passed = passed and (flags["delta_ok"] is False)
            self.record(cls, f"case_{ci}",
                        expected={"one_sided_observable": exp_one_sided,
                                  "semantics": "sensitivity" if exp_one_sided else "control"},
                        observed={"one_sided_observable": one_sided,
                                  "obs_p": bool(obs_p[0]), "obs_i": bool(obs_i[0]),
                                  "delta_ok": flags["delta_ok"]},
                        passed=passed)

    # ------------------------------------------------------------ class C4
    def c4_state_zero_sign(self) -> None:
        cls = "state_zero_sign_problems"
        n = 4
        z = np.zeros(n)
        tiny = np.full(n, 1e-12)
        cfgs = [
            (tiny, tiny, tiny, tiny, [], "deep_noise_control"),
            (tiny, tiny, tiny, tiny, [], "all_tiny_control"),
            (np.array([1e-3] * n), tiny, z, tiny, ["m_nonzero_erased"], "m_nonzero_erased"),
            (z, z, z, np.array([1e-6] * n), ["v_zero_invented"], "v_zero_invented"),
            (np.array([1e-3] * n), tiny, np.array([-1e-3] * n), tiny, ["m_sign"], "m_sign"),
            (z, np.array([-1e-6] * n), z, z,
             ["ref_v_negative", "v_nonzero_erased"], "ref_v_negative"),
            (np.array([1e-3] * n), tiny, np.array([-1e-3] * n), z,
             ["m_sign", "v_nonzero_erased"], "combined"),
            (z, z, z, z, [], "m_none_v_none_control"),
        ]
        for ci, (rm, rv, gm, gv, exp, tag) in enumerate(cfgs):
            self.budget.check(f"c4.case[{ci}]")
            m_none = v_none = (tag == "m_none_v_none_control")
            got = _zero_state_problems(rm, rv, gm, gv, m_none=m_none, v_none=v_none)
            passed = got == exp
            self.record(cls, f"case_{ci}",
                        expected={"problems": exp},
                        observed={"problems": got},
                        passed=passed,
                        note="direct-failure semantics, not a gap" if exp else "")

    # ------------------------------------------------------------ class C5
    def c5_cancellation_within_intervals(self) -> None:
        cls = "cancellation_within_intervals"
        rng = np.random.default_rng(SEED + 1)
        cases: list[dict] = []
        # (1) step-1 m_none zero-disagreement: g_p = 0 vs g_i = 1e-40 subnormal.
        cases.append({"theta": _f32([1.0]), "m": None, "v": None, "step": 1,
                      "g_p": _f32([0.0]), "g_i": _f32([1e-40]), "tag": "m_none_zero_disagreement"})
        # (2) step-1 v-side: g_p = 0 vs g_i = 1e-20 -> v_i subnormal != 0.
        cases.append({"theta": _f32([1.0]), "m": None, "v": None, "step": 1,
                      "g_p": _f32([0.0]), "g_i": _f32([1e-20]), "tag": "v_zero_disagreement"})
        # (3-5) deep cancellation with 1-ULP gradient difference at three bands.
        for band, scale in (("deep", 1e-12), ("mid", 1e-8), ("outer", 1e-2)):
            g_p = (rng.uniform(0.5, 1.0, 4) * scale).astype(np.float32)
            g_i = np.nextafter(g_p, np.float32(np.inf)).astype(np.float32)
            cases.append({"theta": _f32(np.full(4, 1.0)),
                          "m": g_p.copy(),                                   # z_p == 0 exactly
                          "v": _f32(np.full(4, scale ** 2)), "step": 5,
                          "g_p": g_p, "g_i": g_i, "tag": f"ulp_cancellation_{band}"})
        # (6-7) two-step recursion from case 3-style start.
        g_p = (rng.uniform(0.5, 1.0, 4) * 1e-8).astype(np.float32)
        g_i = np.nextafter(g_p, np.float32(-np.inf)).astype(np.float32)
        cases.append({"theta": _f32(np.full(4, 1.0)),
                      "m": g_p.copy(),
                      "v": _f32(np.full(4, 1e-16)), "step": 5,
                      "g_p": g_p, "g_i": g_i, "tag": "two_step_recursion", "steps": 2})
        # (8) sign-straddling subnormal m with g straddling the same boundary.
        cases.append({"theta": _f32([1.0]), "m": _f32([5e-45]), "v": _f32([0.0]),
                      "step": 1, "g_p": _f32([-5e-44]), "g_i": _f32([-6e-44]),
                      "tag": "subnormal_boundary"})
        for ci, c in enumerate(cases):
            self.budget.check(f"c5.case[{ci}]")
            steps = c.get("steps", 1)
            theta, m, v = c["theta"], c["m"], c["v"]
            inside_all = True
            gate_failed = False
            g_p, g_i = c["g_p"], c["g_i"]
            for step_i in range(steps):
                step = c["step"] + step_i
                p = adamw_effective_fp32(theta, m, v, g_p, step,
                                         m_none=(m is None and step_i == 0),
                                         v_none=(v is None and step_i == 0))
                i = adamw_effective_fp32(theta, m, v, g_i, step,
                                         m_none=(m is None and step_i == 0),
                                         v_none=(v is None and step_i == 0))
                self.counters.fp32_formula_evals += 2
                if step_i > 0:
                    self.counters.state_advances += 1
                iv = propagate_intervals(g_i.astype(np.float64), theta, m, v, step,
                                         m_none=(m is None and step_i == 0),
                                         v_none=(v is None and step_i == 0))
                self.counters.interval_propagations += 1
                budget_in = effective_input_budget(g_i.astype(np.float64))
                diff = np.abs(g_p.astype(np.float64) - g_i.astype(np.float64))
                if not np.all(diff <= budget_in):
                    inside_all = False  # input outside the frozen gate set
                m_none = (m is None and step_i == 0)
                obs_p, _ = _delta_observability(
                    p["delta_ref64"], p["theta"].astype(np.float64) - theta.astype(np.float64),
                    _half_ulp32(p["theta"]), p["theta"], theta, gate_rtol=DELTA_SAME_INPUT_RTOL)
                obs_i, _ = _delta_observability(
                    i["delta_ref64"], i["theta"].astype(np.float64) - theta.astype(np.float64),
                    _half_ulp32(i["theta"]), i["theta"], theta, gate_rtol=DELTA_SAME_INPUT_RTOL)
                flags = _cross_path_flags(
                    p["m"], i["m"], p["v"], i["v"], p["theta"], i["theta"],
                    p["theta"].astype(np.float64) - theta.astype(np.float64),
                    p["delta_ref64"], i["delta_ref64"], obs_p, obs_i,
                    STATE_SPEC, delta_rtol=DELTA_SAME_INPUT_RTOL)
                gate_failed = gate_failed or not (flags["m_ok"] and flags["v_ok"]
                                                  and flags["theta_ok"] and flags["delta_ok"])
                for key, values in (("K_m", (p["m"], i["m"])), ("K_v", (p["v"], i["v"])),
                                    ("K_theta", (p["theta"], i["theta"]))):
                    for val in values:
                        if not _inside(iv[key], val):
                            inside_all = False
                for val in (p["theta"].astype(np.float64) - theta.astype(np.float64),
                            i["theta"].astype(np.float64) - theta.astype(np.float64),
                            p["delta_ref64"], i["delta_ref64"]):
                    if not _inside(iv["K_delta"], val):
                        inside_all = False
                theta, m, v = p["theta"], p["m"], p["v"]
            must_fail_gate = c["tag"] in ("m_none_zero_disagreement",
                                          "v_zero_disagreement", "subnormal_boundary")
            semantics = ("cross_path_sensitivity_gap_inside_intervals" if gate_failed
                         else "gates_hold")
            passed = inside_all and (gate_failed or not must_fail_gate)
            self.record(cls, f"case_{ci}_{c['tag']}",
                        expected={"inside_intervals": True,
                                  "cross_path_gate_failed": must_fail_gate},
                        observed={"inside_intervals": inside_all,
                                  "cross_path_gate_failed": gate_failed,
                                  "semantics": semantics},
                        passed=passed)

    # ------------------------------------------------------------ class C6
    def c6_interval_violation(self) -> None:
        cls = "interval_violation_deviation"
        theta0 = _f32(np.full(4, 1.0))
        m_old = _f32(np.full(4, 1e-3))
        v_old = _f32(np.full(4, 1e-6))
        g = _f32(np.full(4, 1e-2))
        step = 5
        host = host_constants(step)
        a32, b2, c32 = (np.float32(host["a_32"]), np.float32(host["b2_32"]),
                        np.float32(host["one_minus_b2_32"]))
        h32, d32, eps32 = (np.float32(host["h_32"]), np.float32(host["d_32"]),
                           np.float32(host["eps_32"]))
        bc2 = np.float32(host["bc2_sqrt_32"])
        # Gross formula deviations; each must leave the propagated interval.
        deviations = [
            ("h_sign_flipped", {}, lambda th, m, v, g: (m, v, np.float32(-float(host["h_32"])))),
            ("denom_without_eps", {"v": _f32(np.full(4, 0.0)), "g": _f32(np.full(4, 1e-30))},
             None),  # special-cased below
            ("denom_without_bc2", {}, "denom_without_bc2"),
            ("g_not_squared", {}, "g_not_squared"),
            ("sqrt_skipped", {}, "sqrt_skipped"),
            ("lr_doubled", {}, "lr_doubled"),
            ("ratio_denominator_floor", {}, "ratio_denominator_floor"),
            ("m_uses_g_directly", {}, "m_uses_g_directly"),
        ]
        for ci, (name, overrides, kind) in enumerate(deviations):
            self.budget.check(f"c6.case[{ci}]")
            th = overrides.get("theta", theta0)
            m_o = overrides.get("m", m_old)
            v_o = overrides.get("v", v_old)
            g_o = overrides.get("g", g)
            frozen = adamw_effective_fp32(th, m_o, v_o, g_o, step)
            self.counters.fp32_formula_evals += 1
            iv = propagate_intervals(g_o.astype(np.float64), th, m_o, v_o, step)
            self.counters.interval_propagations += 1
            z = (g_o - m_o).astype(np.float32)
            m_d = (a32 * z + m_o).astype(np.float32)
            g2 = (g_o * g_o).astype(np.float32)
            v_d = (c32 * g2 + b2 * v_o).astype(np.float32)
            s = np.sqrt(v_d.astype(np.float64)).astype(np.float32)
            b = (s / bc2).astype(np.float32)
            denom = (b + eps32).astype(np.float32)
            h = h32
            theta_decay = (th * d32).astype(np.float32)
            if name == "h_sign_flipped":
                h = np.float32(-float(h32))
            elif name == "denom_without_eps":
                denom = b
            elif name == "denom_without_bc2":
                b = s
                denom = (b + eps32).astype(np.float32)
            elif name == "g_not_squared":
                v_d = (c32 * g_o + b2 * v_o).astype(np.float32)
                s = np.sqrt(v_d.astype(np.float64)).astype(np.float32)
                denom = ((s / bc2).astype(np.float32) + eps32).astype(np.float32)
            elif name == "sqrt_skipped":
                denom = ((v_d / bc2).astype(np.float32) + eps32).astype(np.float32)
            elif name == "lr_doubled":
                h = np.float32(2.0 * float(h32))
            elif name == "ratio_denominator_floor":
                denom = (denom + np.float32(1e-3)).astype(np.float32)
            elif name == "m_uses_g_directly":
                m_d = g_o
            ratio = (m_d / denom).astype(np.float32)
            theta_dev = (theta_decay + (h * ratio).astype(np.float64)).astype(np.float32)
            theta_outside = not _inside(iv["K_theta"], theta_dev)
            delta_outside = not _inside(
                iv["K_delta"], theta_dev.astype(np.float64) - th.astype(np.float64))
            bitwise_diff = not _bitwise_eq(theta_dev, frozen["theta"])

            # Feed the deviation through the N0 coordinate detector: the saved
            # same_input flag would be false (formula mismatch) and the theta
            # write must land outside the N0 interval check or break delta
            # existence — i.e. the exact detector used for the real evidence
            # must flag this row as an implementation issue, not stay silent.
            obs = 1 if float(np.asarray(frozen["delta_ref64"]).reshape(-1)[0]) != 0.0 else 0
            detector_input = {
                "theta_pre": th, "theta_p": theta_dev, "theta_i": frozen["theta"],
                "m_p": m_d, "v_p": v_d, "m_i": frozen["m"], "v_i": frozen["v"],
                "delta_ref_p": theta_dev.astype(np.float64) - th.astype(np.float64),
                "delta_ref_i": frozen["delta_ref64"],
                "delta_measured_p": theta_dev.astype(np.float64) - th.astype(np.float64),
                "delta_measured_i": frozen["theta"].astype(np.float64) - th.astype(np.float64),
                "bound_delta_p": _half_ulp32(theta_dev),
                "bound_delta_i": _half_ulp32(frozen["theta"]),
                "K_m_lo": np.zeros(1), "K_m_hi": np.full(1, 1e30),
                "K_v_lo": np.zeros(1), "K_v_hi": np.full(1, 1e30),
                "K_theta_lo": np.asarray(iv["K_theta"][0]).reshape(-1),
                "K_theta_hi": np.asarray(iv["K_theta"][1]).reshape(-1),
                "K_delta_lo": np.asarray(iv["K_delta"][0]).reshape(-1),
                "K_delta_hi": np.asarray(iv["K_delta"][1]).reshape(-1),
                "obs_p": np.array([obs], np.uint8), "obs_i": np.array([obs], np.uint8),
                "ok_p": np.array([1], np.uint8), "ok_i": np.array([1], np.uint8),
                "i_theta_bind": frozen["theta"], "i_m_bind": frozen["m"],
                "i_v_bind": frozen["v"],
                "p_theta_before": th, "p_theta_after": theta_dev,
                "step": step, "lr": 1e-4, "weight_decay": 1e-5,
            }
            d_flags, d_anomalies = compute_row_flags(
                detector_input,
                {"delta_same_input": DELTA_SAME_INPUT_RTOL,
                 "delta_cross_path": DELTA_SAME_INPUT_RTOL,
                 "theta_atol": THETA_SPEC["atol"], "theta_rtol": THETA_SPEC["rtol"],
                 "state_atol": STATE_SPEC["atol"], "state_rtol": STATE_SPEC["rtol"]},
                native_shape=(int(np.asarray(frozen["theta"]).size),),
                same_input_p=0, same_input_i=1)
            cls_out, _gaps = recompute_classification(d_flags)
            nonfinite = any(x.startswith("nonfinite") for x in d_anomalies)
            detected = (bool((d_flags & SOURCE_ANOMALY).any())
                        and (bool((d_flags & INTERVAL_OUTSIDE).any())
                             or bool((d_flags & (DELTA_PRESENCE_MISMATCH
                                                 | DELTA_ONE_SIDED_OBSERVABLE)).any())
                             or nonfinite)
                        and (cls_out in ("direct_failure", "cross_path_bound_unverifiable",
                                         "cross_path_sensitivity_gap") or nonfinite))
            caught = bitwise_diff and (theta_outside or delta_outside) and detected
            self.record(cls, f"case_{ci}_{name}",
                        expected={"caught_as_implementation_issue": True},
                        observed={"bitwise_differs_from_frozen": bitwise_diff,
                                  "theta_outside_interval": theta_outside,
                                  "delta_outside_interval": delta_outside,
                                  "n0_detector_source_anomaly": bool((d_flags & SOURCE_ANOMALY).any()),
                                  "n0_detector_interval_outside": bool((d_flags & INTERVAL_OUTSIDE).any()),
                                  "n0_detector_classification": cls_out,
                                  "n0_detector_anomalies": d_anomalies[:3]},
                        passed=caught,
                        note="deviation fed through the N0 coordinate detector")


def run_synthetic_verification(budget: N0Budget) -> tuple:
    budget.check("entry")
    runner = CaseRunner(budget)
    budget_exceeded = False
    with np.errstate(all="ignore"):
        try:
            for fn in (runner.c1_same_input_formula, runner.c2_write_erasure,
                       runner.c3_one_sided_observable, runner.c4_state_zero_sign,
                       runner.c5_cancellation_within_intervals,
                       runner.c6_interval_violation):
                budget.check(fn.__name__)
                fn()
        except BudgetExceeded:
            budget_exceeded = True
    if not budget_exceeded:
        budget.check("cases_complete")

    per_class = {c: {"cases": 0, "passed": 0} for c in CLASSES}
    for c in runner.cases:
        per_class[c["class"]]["cases"] += 1
        per_class[c["class"]]["passed"] += int(c["passed"])
    for c in CLASSES:
        if per_class[c]["cases"] > MAX_CASES_PER_CLASS:
            raise N1Error(f"class {c} exceeds {MAX_CASES_PER_CLASS} cases")
    all_passed = (not budget_exceeded) and all(
        v["cases"] > 0 and v["passed"] == v["cases"] for v in per_class.values())
    counters = runner.counters.as_dict()
    if counters["cases"] > MAX_TOTAL_CASES:
        raise N1Error("case cap violated")
    if counters["fp32_formula_evals"] > 384:
        raise N1Error("formula evaluation cap exceeded")

    summary = {
        "schema": SCHEMA + "_summary",
        "budget_exceeded": bool(budget_exceeded),
        "all_cases_passed": bool(all_passed),
        "per_class": per_class,
        "counters": counters,
        "caps": {
            "max_cases_per_class": MAX_CASES_PER_CLASS,
            "max_total_cases": MAX_TOTAL_CASES,
            "max_coords_per_case": MAX_COORDS_PER_CASE,
            "max_formula_evals": 384,
        },
        "budget": budget.facts(),
        "closure": {
            "same_input_formula_bitwise_verified": bool(
                per_class["same_input_formula_bitwise"]["cases"] > 0
                and per_class["same_input_formula_bitwise"]["passed"]
                == per_class["same_input_formula_bitwise"]["cases"]),
            "write_erasure_is_precision_not_formula_error":
                per_class["nonzero_write_erasure"]["passed"] > 0,
            "one_sided_observable_is_sensitivity":
                per_class["one_sided_observable"]["passed"] > 0,
            "state_zero_sign_is_direct_failure":
                per_class["state_zero_sign_problems"]["passed"] > 0,
            "cross_path_sensitivity_inside_intervals_reproduced":
                per_class["cancellation_within_intervals"]["passed"] > 0,
            "implementation_deviations_are_detectable":
                per_class["interval_violation_deviation"]["passed"] > 0,
            "strict_cross_path_gates_still_failing_on_real_evidence": True,
            "long_horizon_training_and_action_quality_uncovered": True,
            "cpu_not_claimed_equivalent_to_production_cuda_foreach": True,
        },
    }
    go_no_go = None
    if not budget_exceeded:
        go_no_go = ("go_for_conditional_fit_diagnostics_review" if all_passed
                    else "no_go_investigate_synthetic_failures")
    summary["go_no_go"] = go_no_go
    return summary, runner.cases


# On a seal withdrawal (slow write crossing the wall cap) the go/no-go
# recommendation is revoked together with the completion claims: a budget-
# exceeded run must not publish a go suggestion.
N1_WITHDRAW_ON_EXCEED = {
    "all_cases_passed": False,
    "label_verdict": "budget_exceeded_after_seal",
    "go_no_go": "no_go_budget_seal_exceeded",
}


def write_outputs(output_dir: Path, summary: dict, cases: list,
                  budget: N0Budget | None = None, clock=None) -> dict:
    import os

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "n1_cases.json"
    cases_path.write_text(json.dumps({"schema": SCHEMA, "cases": cases}, indent=1))
    if cases_path.stat().st_size > MAX_REPORT_BYTES:
        raise N1Error("n1_cases.json exceeds frozen report bound")
    summary["outputs"] = {"n1_cases.json": cases_path.name}
    summary_path = output_dir / "n1_summary.json"
    if budget is not None:
        seal_facts = _write_summary_sealed(summary_path, summary, budget,
                                           withdraw=dict(N1_WITHDRAW_ON_EXCEED),
                                           clock=clock)
    else:
        summary_path.write_text(json.dumps(summary, indent=1))
        fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        seal_facts = {"status": "unsealed_no_budget", "measured_wall_s": None,
                      "rounds": 0, "samples": []}
    if summary_path.stat().st_size > MAX_REPORT_BYTES:
        raise N1Error("n1_summary.json exceeds frozen report bound")
    # The on-disk document (post-seal) is returned so the CLI decides from
    # disk state, never from pre-write memory.
    return {"cases": str(cases_path), "summary": str(summary_path),
            "summary_doc": json.loads(summary_path.read_text()),
            "seal_facts": seal_facts}
