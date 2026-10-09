"""P/U0 teacher quality comparison metrics and source-verdict gates.

Plan omniretarget_first_training_plan §3.4 (A2): measure both sources with
the SAME current validator and interpolation protocol, on reference
intervals inferred from the SOURCE scene pairing (plan §4.3 — the A1 freeze
supplies the source-scene box trajectory), then apply the frozen
``local_preferred`` criteria.

Metric families per (source, demo):

* penetration: worst and violation ratio over robot-ground/robot-box/self
  pairs at keyframes + linear midpoints (the validator's interpolated-state
  protocol; ``box_ground`` excluded as a fixed task input, §7.3);
* task: robot-kpt vs reference-human SMPLH keypoint MPJPE (p95 per demo);
* feet, inside reference support intervals: sole-gap p95, foot-speed p95,
  within-interval drift;
* hands, inside reference grasp intervals per side: box-gap p95, coverage,
  *valid* coverage (gap < 0.01 m with penetration <= 0.002 m), certified
  overlap (validator robot schedule ∩ reference), within-interval box-frame
  drift;
* both hands, on the both-grasp reference denominator: valid coverage and
  the ``max(0, left, right)`` gap series p95.

Gates (§3.4 items 1-4 frozen; item 5 is reported evidence, no invented
numeric bound). All comparisons keep the full 6-demo denominator; missing
rows are failures, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ireset.embodied.contact_sites import SMPLH_INDEX, infer_reference_schedule
from ireset.embodied.g1_spec import KPT_TO_SMPLH
from ireset.repair.validation import IndependentValidator, ValidationConfig

HAND_SIDES = ("l", "r")
#: §3.4 fixed numeric bounds (diagnostic working values, frozen here).
HAND_VALID_GAP_M = 0.01
PENETRATION_TOL_M = 0.002
FOOT_SPEED_STRICT_MPS = 0.02
FOOT_GAP_STRICT_M = 0.005
TASK_P95_MARGIN_M = 0.02
COMPARE_TOL = 1e-6
BOTH_COVERAGE_GAIN_PP = 10.0
BOTH_GAP_REL_DROP = 0.25
MIN_GAIN_DEMOS = 3


def _p95(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 95)) if x.size else float("nan")


def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def _frames_of(schedule, effector: str, support: str) -> list[tuple[int, int]]:
    return [(iv.start, iv.end) for iv in schedule.intervals
            if iv.effector == effector and iv.support == support]


def _mask_from_ranges(ranges: list[tuple[int, int]], T: int) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    for s, e in ranges:
        m[s:e] = True
    return m


@dataclass
class SideHandMetrics:
    n_ref_frames: int
    gap_p95_m: float
    coverage: float | None
    valid_coverage: float | None
    certified_overlap_frames: int
    longest_certified_overlap: int
    drift_max_m: float | None

    def to_json(self) -> dict:
        return {
            "n_ref_frames": self.n_ref_frames,
            "gap_p95_m": None if np.isnan(self.gap_p95_m) else round(self.gap_p95_m, 6),
            "coverage": None if self.coverage is None else round(self.coverage, 4),
            "valid_coverage": None if self.valid_coverage is None else round(self.valid_coverage, 4),
            "certified_overlap_frames": self.certified_overlap_frames,
            "longest_certified_overlap": self.longest_certified_overlap,
            "drift_max_m": None if self.drift_max_m is None else round(self.drift_max_m, 6),
        }


@dataclass
class SideFootMetrics:
    n_ref_frames: int
    gap_p95_m: float
    speed_p95_mps: float
    drift_max_m: float

    def to_json(self) -> dict:
        return {
            "n_ref_frames": self.n_ref_frames,
            "gap_p95_m": round(self.gap_p95_m, 6),
            "speed_p95_mps": round(self.speed_p95_mps, 6),
            "drift_max_m": round(self.drift_max_m, 6),
        }


@dataclass
class DemoMetrics:
    demo: str
    T: int
    fps: float
    valid: bool
    error: str | None = None
    penetration_max_m: float = float("nan")
    penetration_violation_ratio: float = float("nan")
    task_mpjpe_p95_m: float = float("nan")
    box_ground_penetration_max_m: float = float("nan")  # fixed-input scene issue, §7.3
    foot: dict[str, SideFootMetrics | None] = field(default_factory=dict)
    hand: dict[str, SideHandMetrics | None] = field(default_factory=dict)
    both: dict = field(default_factory=dict)
    validator_status: str | None = None
    validator_worst: dict = field(default_factory=dict)
    reference_intervals: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "demo": self.demo, "T": self.T, "fps": self.fps, "valid": self.valid,
            "error": self.error,
            "penetration_max_m": None if np.isnan(self.penetration_max_m) else round(self.penetration_max_m, 6),
            "penetration_violation_ratio": None if np.isnan(self.penetration_violation_ratio) else round(self.penetration_violation_ratio, 4),
            "task_mpjpe_p95_m": None if np.isnan(self.task_mpjpe_p95_m) else round(self.task_mpjpe_p95_m, 6),
            "box_ground_penetration_max_m": None if np.isnan(self.box_ground_penetration_max_m) else round(self.box_ground_penetration_max_m, 6),
            "foot": {k: (v.to_json() if v else None) for k, v in self.foot.items()},
            "hand": {k: (v.to_json() if v else None) for k, v in self.hand.items()},
            "both": self.both,
            "validator_status": self.validator_status,
            "validator_worst": {k: round(v, 6) for k, v in self.validator_worst.items()},
            "reference_intervals": [
                {"eff": iv.effector, "start": iv.start, "end": iv.end, "sup": iv.support}
                for iv in self.reference_intervals
            ],
        }


class TeacherQualityReplayer:
    """Per-frame series on the locked scene, reusing the validator's model."""

    def __init__(self, validator: IndependentValidator | None = None):
        self.validator = validator or IndependentValidator(cfg=ValidationConfig())
        self.model = self.validator.model
        self.data = mujoco.MjData(self.model)
        self.box_geom = self.validator.box_geom
        self.ground_geom = self.validator.ground_geom
        sid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
        self.kpt_ids = [sid(n) for n in KPT_TO_SMPLH]
        self.ref_kpt_idx = [SMPLH_INDEX[name] for name in KPT_TO_SMPLH.values()]
        self._hand_geoms = self.validator._hand_geoms
        self.sole_geoms = self.validator.sole_geoms
        self.sole_sites = self.validator.sole_sites
        self.hand_sites = self.validator.hand_sites
        # collision pairs by kind; box_ground is a fixed task input (§7.3)
        self.pairs_by_kind: dict[str, list] = {}
        for ga, gb, kind in self.validator.pairs.all_pairs:
            self.pairs_by_kind.setdefault(kind, []).append((ga, gb))
        self._fromto = np.zeros(6)

    # -- geometry helpers ----------------------------------------------------

    def _dist(self, g1: int, g2: int) -> float:
        return mujoco.mj_geomDistance(self.model, self.data, g1, g2, 2.0, self._fromto)

    def _forward(self, scene: np.ndarray, t: int) -> None:
        self.data.qpos = scene[t]
        mujoco.mj_forward(self.model, self.data)

    # -- main ---------------------------------------------------------------

    def measure(self, demo: str, scene: np.ndarray, human: np.ndarray,
                source_box: np.ndarray, fps: float) -> DemoMetrics:
        """scene [T,43] project convention; source_box [T,7] (A1 pairing)."""
        T = scene.shape[0]
        if not np.isfinite(scene).all() or not np.isfinite(human).all():
            return DemoMetrics(demo, T, fps, valid=False, error="non-finite input")

        ref_sched = infer_reference_schedule(human, source_box, fps=fps)
        rep = self.validator.validate(scene, exclude=("box_ground",))
        if rep.schedule is None:
            return DemoMetrics(demo, T, fps, valid=False,
                               error=f"validator rejected: {rep.status}")

        pen_series = np.zeros(T)          # keyframes
        pen_mid = np.zeros(T - 1) if T > 1 else np.zeros(0)
        box_ground_pen = 0.0
        gap = {s: np.zeros(T) for s in HAND_SIDES}
        sole_gap = {s: np.zeros(T) for s in HAND_SIDES}
        sole_c = {s: np.zeros((T, 3)) for s in HAND_SIDES}
        hand_c_box = {s: np.zeros((T, 3)) for s in HAND_SIDES}
        kpt_err = np.zeros(T)

        active_pairs = [
            (a, b) for kind in ("robot_ground", "robot_box", "self")
            for (a, b) in self.pairs_by_kind.get(kind, [])
        ]
        for t in range(T):
            self._forward(scene, t)
            worst = 0.0
            for ga, gb in active_pairs:
                worst = max(worst, -self._dist(ga, gb))
            pen_series[t] = worst
            box_ground_pen = max(box_ground_pen, -self._dist(self.box_geom, self.ground_geom))
            R_g = self.data.geom_xmat[self.box_geom].reshape(3, 3)
            x_g = self.data.geom_xpos[self.box_geom]
            for s in HAND_SIDES:
                gap[s][t] = min(self._dist(g, self.box_geom) for g in self._hand_geoms[s])
                sole_gap[s][t] = min(self._dist(g, self.ground_geom) for g in self.sole_geoms[s])
                sole_c[s][t] = self.data.site_xpos[self.sole_sites[s]].mean(0)
                hc = self.data.site_xpos[self.hand_sites[s]].mean(0)
                hand_c_box[s][t] = R_g.T @ (hc - x_g)
            kpt_err[t] = float(np.linalg.norm(
                self.data.site_xpos[self.kpt_ids] - human[:, self.ref_kpt_idx], axis=-1
            ).mean())
            if T > 1 and t + 1 < T:
                mid = 0.5 * (scene[t] + scene[t + 1])
                self.data.qpos = mid
                mujoco.mj_forward(self.model, self.data)
                wm = 0.0
                for ga, gb in active_pairs:
                    wm = max(wm, -self._dist(ga, gb))
                pen_mid[t] = wm

        dt = 1.0 / fps
        pen_all = np.concatenate([pen_series, pen_mid]) if len(pen_mid) else pen_series
        pen_valid = np.maximum(pen_series, 0.0)

        m = DemoMetrics(
            demo=demo, T=T, fps=fps, valid=True,
            penetration_max_m=float(pen_all.max()),
            penetration_violation_ratio=float((pen_all > PENETRATION_TOL_M).mean()),
            task_mpjpe_p95_m=_p95(kpt_err),
            box_ground_penetration_max_m=float(box_ground_pen),
            validator_status=rep.status,
            validator_worst={k: v.worst for k, v in rep.checks.items()},
            reference_intervals=ref_sched.intervals,
        )

        # feet (reference support intervals, world-frame series)
        m.foot = {}
        for s, eff in (("l", "left_foot"), ("r", "right_foot")):
            ranges = _frames_of(ref_sched, eff, "ground")
            mask = _mask_from_ranges(ranges, T)
            if not mask.any():
                m.foot[s] = None
                continue
            speed = np.zeros(T)
            cs = sole_c[s]
            speed[1:] = np.linalg.norm(np.diff(cs, axis=0), axis=1) / dt
            drift = 0.0
            for s0, e0 in ranges:
                anchor = cs[s0]
                drift = max(drift, float(np.linalg.norm(cs[s0:e0] - anchor, axis=1).max()))
            m.foot[s] = SideFootMetrics(
                n_ref_frames=int(mask.sum()),
                gap_p95_m=_p95(sole_gap[s][mask]),
                speed_p95_mps=_p95(speed[mask]),
                drift_max_m=drift,
            )

        # hands (reference grasp intervals, box-frame drift)
        m.hand = {}
        for s, eff in (("l", "left_hand"), ("r", "right_hand")):
            ranges = _frames_of(ref_sched, eff, "object")
            mask = _mask_from_ranges(ranges, T)
            if not mask.any():
                m.hand[s] = None
                continue
            g = np.maximum(gap[s], 0.0)
            valid_frames = (g < HAND_VALID_GAP_M) & (pen_valid <= PENETRATION_TOL_M)
            rob = _frames_of(rep.schedule, eff, "object")
            rob_mask = _mask_from_ranges(rob, T)
            overlap = rob_mask & mask
            longest = _longest_true_run(overlap)
            drift = 0.0
            hb = hand_c_box[s]
            for s0, e0 in ranges:
                anchor = hb[s0]
                drift = max(drift, float(np.linalg.norm(hb[s0:e0] - anchor, axis=1).max()))
            m.hand[s] = SideHandMetrics(
                n_ref_frames=int(mask.sum()),
                gap_p95_m=_p95(g[mask]),
                coverage=float((g[mask] < HAND_VALID_GAP_M).mean()),
                valid_coverage=float(valid_frames[mask].mean()),
                certified_overlap_frames=int(overlap.sum()),
                longest_certified_overlap=longest,
                drift_max_m=drift,
            )

        # both hands on the both-grasp reference denominator
        ml = _mask_from_ranges(_frames_of(ref_sched, "left_hand", "object"), T)
        mr = _mask_from_ranges(_frames_of(ref_sched, "right_hand", "object"), T)
        both_ref = ml & mr
        if both_ref.any():
            gl = np.maximum(gap["l"], 0.0)
            gr = np.maximum(gap["r"], 0.0)
            series = np.maximum(np.maximum(gl, gr), 0.0)
            both_valid = (gl < HAND_VALID_GAP_M) & (gr < HAND_VALID_GAP_M) & (pen_valid <= PENETRATION_TOL_M)
            m.both = {
                "denominator_frames": int(both_ref.sum()),
                "valid_coverage": round(float(both_valid[both_ref].mean()), 4),
                "gap_series_p95_m": round(_p95(series[both_ref]), 6),
            }
        else:
            m.both = {"denominator_frames": 0, "valid_coverage": None, "gap_series_p95_m": None}
        return m


def evaluate_local_preferred(
    pilot: list[str], p: dict[str, DemoMetrics], u: dict[str, DemoMetrics],
    binding_passed: dict[str, bool],
) -> dict:
    """Frozen §3.4 criteria. Returns the machine-readable gate record."""
    gates: dict[str, dict] = {}

    def _g1() -> dict:
        rows = {}
        for d in pilot:
            pm, um = p.get(d), u.get(d)
            ok = (
                d in binding_passed and binding_passed[d]
                and um is not None and um.valid and pm is not None and pm.valid
            )
            rows[d] = ok
        return {"per_demo": rows, "passed": all(rows.values())}

    def _g2() -> dict:
        rows = {
            d: (u[d].task_mpjpe_p95_m <= p[d].task_mpjpe_p95_m + TASK_P95_MARGIN_M)
            for d in pilot
        }
        return {"per_demo": rows, "passed": all(rows.values())}

    def _g3() -> dict:
        rows = {}
        for d in pilot:
            pm, um = p[d], u[d]
            ok = um.penetration_max_m <= max(pm.penetration_max_m, PENETRATION_TOL_M) + COMPARE_TOL
            foot_detail = {}
            for s in HAND_SIDES:
                pf, uf = pm.foot.get(s), um.foot.get(s)
                if pf is None or pf.n_ref_frames == 0:
                    foot_detail[s] = "no_reference_support_frames"
                    continue
                if uf is None or uf.n_ref_frames == 0:
                    foot_detail[s] = "u0_missing_reference_frames"
                    ok = False
                    continue
                s_ok = (
                    uf.speed_p95_mps <= max(pf.speed_p95_mps, FOOT_SPEED_STRICT_MPS) + COMPARE_TOL
                    and uf.gap_p95_m <= max(pf.gap_p95_m, FOOT_GAP_STRICT_M) + COMPARE_TOL
                )
                foot_detail[s] = bool(s_ok)
                ok = ok and s_ok
            rows[d] = {"passed": ok, "penetration": ok, "foot": foot_detail}
        return {"per_demo": rows,
                "passed": all(r["passed"] for r in rows.values())}

    def _g4() -> dict:
        cov_gains, gap_rels = [], []
        for d in pilot:
            pb, ub = p[d].both, u[d].both
            if pb.get("denominator_frames", 0) == 0 or ub.get("denominator_frames", 0) == 0:
                cov_gains.append(None)
                gap_rels.append(None)
                continue
            if pb["valid_coverage"] is not None and ub["valid_coverage"] is not None:
                cov_gains.append(100.0 * (ub["valid_coverage"] - pb["valid_coverage"]))
            else:
                cov_gains.append(None)
            pg, ug = pb["gap_series_p95_m"], ub["gap_series_p95_m"]
            if pg is None or ug is None or pg <= 0:
                gap_rels.append(None)
            else:
                gap_rels.append((pg - ug) / pg)
        n_cov = sum(1 for g in cov_gains if g is not None and g >= BOTH_COVERAGE_GAIN_PP)
        n_gap = sum(1 for r in gap_rels if r is not None and r >= BOTH_GAP_REL_DROP)
        return {
            "coverage_gain_pp_by_demo": cov_gains, "gap_rel_drop_by_demo": gap_rels,
            "n_coverage_gain_demos": n_cov, "n_gap_drop_demos": n_gap,
            "primary_metric": "both_valid_coverage_gain>=10pp (fallback gap_p95 drop>=25%)",
            "passed": (n_cov >= MIN_GAIN_DEMOS) or (n_gap >= MIN_GAIN_DEMOS),
        }

    gates["g1_identity_and_availability"] = _g1()
    gates["g2_task_mpjpe"] = _g2()
    gates["g3_penetration_and_feet"] = _g3()
    gates["g4_hand_improvement"] = _g4()
    complete = (
        all(d in p and p[d].valid for d in pilot)
        and all(d in u and u[d].valid for d in pilot)
        and all(binding_passed.get(d, False) for d in pilot)
    )
    if not complete:
        verdict = "incomplete"
    elif all(g["passed"] for g in gates.values()):
        verdict = "local_preferred"
    else:
        verdict = "no_quality_gain"
    return {"gates": gates, "complete_evidence": complete, "verdict": verdict}
