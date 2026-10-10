"""P/U0 teacher quality comparison metrics and source-verdict gates.

Plan omniretarget_first_training_plan §3.4 (A2): measure both sources with
the SAME current validator and interpolation protocol, on reference
intervals inferred from the SOURCE scene pairing (plan §4.3 — the A1 freeze
supplies the source-scene box trajectory), then apply the frozen
``local_preferred`` criteria.

Protocol notes (A2 review fixes 1-3/8-9, 2026-10-09 — every series reuses
the validator's own cached measurements so both sources are measured by one
protocol):

* penetration max comes from ``rep.checks["penetration"]`` — the validator's
  motion-adaptive subdivision + SLERP interpolation; the violation ratio is
  computed on the keyframe series only and is labeled as such (diagnostic);
* task MPJPE is FRAME-ALIGNED: robot keypoints of frame t against the human
  keypoints of frame t (the previous broadcast over the whole human tensor
  inflated every demo by cross-frame distances);
* foot speed and drift are PER-SITE WORST (a pivoting foot keeps its
  centroid fixed while sole sites sweep — centroid speed is blind to it);
  the first frame has no defined speed and is excluded from the stats;
* hand drift is per-site worst inside the box frame;
* certified overlap counts only robot grasp intervals that pass the
  validator's ``_check_hands`` per-interval criteria (gap / anchor drift /
  box-frame relative speed) intersected with the reference intervals —
  inferred contact alone is NOT certification (a slowly drifting grasp
  keeps the interval but fails the anchor check);
* both hands, on the both-grasp reference denominator: valid coverage
  (both gaps < 0.01 m and keyframe penetration <= 0.002 m) and the
  ``max(0, left, right)`` gap series p95.

Gates (§3.4 items 1-5): items 1-4 carry the frozen numeric bounds; item 5
(no buying contact gains with larger penetration/drift/slip/task error) is
enforced by ``g5_degradation`` over three families — within-interval hand
and foot anchor drift (registered working margin), validator hard failures
NEW in U0 relative to P (joint limits / velocities / penetration tolerance
/ hard contact checks), and the keyframe penetration violation ratio. All
comparisons keep the full 6-demo denominator; missing or non-finite
evidence (None / NaN / Inf, on every measured field — round-3 review fix
1) forces ``incomplete`` (never a silent drop, never a KeyError). The
evidence check runs BEFORE the quality gates and every gate is hardened
against deficient rows (round-4 review fix 3) — a None task error or a
both-hands dict without keys yields ``incomplete``, not a TypeError or
KeyError mid-computation; report serialization is None-safe as well.
Round-5 additions (2026-10-10): the both-hands common denominator is
verified for existence, validity and P/U0 AGREEMENT — the denominator is
decided by the shared (binding-verified) human reference, so a missing or
None ``denominator_frames`` is deleted evidence (gap → ``incomplete``),
never "zero reference frames", and P and U0 disagreeing on it is an
evidence gap too. Round-6 (2026-10-10): the denominator must be a
non-negative integer — fractional values are invalid evidence, never
silently truncated to a legal count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ireset.embodied.contact_sites import (
    SMPLH_INDEX, infer_reference_schedule,
)
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
#: §3.4 item 5 working margin: U0 may not exceed P's within-interval drift
#: by more than this on any measured side (no plan-frozen numeric bound;
#: registered here so the degradation gate is decidable, 2026-10-09).
DRIFT_DEGRADATION_MARGIN_M = 0.02


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
    n_robot_intervals: int = 0
    n_robot_intervals_certified: int = 0

    def to_json(self) -> dict:
        return {
            "n_ref_frames": self.n_ref_frames,
            "gap_p95_m": _json_num(self.gap_p95_m),
            "coverage": _json_num(self.coverage, 4),
            "valid_coverage": _json_num(self.valid_coverage, 4),
            "certified_overlap_frames": self.certified_overlap_frames,
            "longest_certified_overlap": self.longest_certified_overlap,
            "drift_max_m": _json_num(self.drift_max_m),
            "n_robot_intervals": self.n_robot_intervals,
            "n_robot_intervals_certified": self.n_robot_intervals_certified,
        }


@dataclass
class SideFootMetrics:
    n_ref_frames: int
    gap_p95_m: float
    speed_p95_mps: float
    drift_max_m: float
    n_speed_frames: int = 0

    def to_json(self) -> dict:
        return {
            "n_ref_frames": self.n_ref_frames,
            "gap_p95_m": _json_num(self.gap_p95_m),
            "speed_p95_mps": _json_num(self.speed_p95_mps),
            "drift_max_m": _json_num(self.drift_max_m),
            "n_speed_frames": self.n_speed_frames,
        }


@dataclass
class DemoMetrics:
    demo: str
    T: int
    fps: float
    valid: bool
    error: str | None = None
    penetration_max_m: float = float("nan")          # validator adaptive protocol
    penetration_violation_ratio: float = float("nan")  # keyframes only (diagnostic)
    task_mpjpe_p95_m: float = float("nan")
    box_ground_penetration_max_m: float = float("nan")  # fixed-input scene issue, §7.3
    foot: dict[str, SideFootMetrics | None] = field(default_factory=dict)
    hand: dict[str, SideHandMetrics | None] = field(default_factory=dict)
    both: dict = field(default_factory=dict)
    validator_status: str | None = None
    validator_worst: dict = field(default_factory=dict)
    #: validator checks that FAILED for this source (round-3 review fix 2:
    #: a U0 that newly fails a check P passed — joint limits, velocities,
    #: hard contact checks — is a §3.4 item 3 degradation even when the
    #: aggregate p95 gates still hold)
    validator_failed_checks: list = field(default_factory=list)
    reference_intervals: list = field(default_factory=list)
    #: reference-interval frame counts per side, recorded even when the
    #: measurement itself is missing — that combination is exactly the
    #: "evidence insufficient" case (§3.4 item 3, review fix 8)
    ref_support_frames: dict[str, int] = field(default_factory=dict)
    ref_grasp_frames: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "demo": self.demo, "T": self.T, "fps": self.fps, "valid": self.valid,
            "error": self.error,
            "penetration_max_m": _json_num(self.penetration_max_m),
            "penetration_violation_ratio": _json_num(self.penetration_violation_ratio, 4),
            "penetration_protocol": "validator_adaptive_slerp (violation ratio: keyframes only)",
            "task_mpjpe_p95_m": _json_num(self.task_mpjpe_p95_m),
            "box_ground_penetration_max_m": _json_num(self.box_ground_penetration_max_m),
            "foot": {k: (v.to_json() if v else None) for k, v in self.foot.items()},
            "hand": {k: (v.to_json() if v else None) for k, v in self.hand.items()},
            "both": self.both,
            "validator_status": self.validator_status,
            "validator_worst": {k: _json_num(v) for k, v in self.validator_worst.items()},
            "validator_failed_checks": list(self.validator_failed_checks),
            "reference_intervals": [
                {"eff": iv.effector, "start": iv.start, "end": iv.end, "sup": iv.support}
                for iv in self.reference_intervals
            ],
            "ref_support_frames": dict(self.ref_support_frames),
            "ref_grasp_frames": dict(self.ref_grasp_frames),
        }


# --- pure per-side metric builders (unit-testable without MuJoCo) ------------


def foot_side_metrics(
    sole_pts: np.ndarray, sole_gap: np.ndarray,
    ranges: list[tuple[int, int]], fps: float,
) -> SideFootMetrics:
    """Per-site worst speed/drift on the reference support intervals.

    ``sole_pts`` is [T, n_sites, 3] (the validator's cached series); the
    speed of frame t is the max per-site displacement between t-1 and t over
    dt — frame 0 has no defined speed and stays out of the statistics. Drift
    is the max per-site displacement from that site's interval-start anchor.
    """
    T = sole_pts.shape[0]
    dt = 1.0 / fps
    mask = _mask_from_ranges(ranges, T)
    speed = np.zeros(T)
    speed[1:] = np.linalg.norm(np.diff(sole_pts, axis=0), axis=-1).max(1) / dt
    sel = mask.copy()
    sel[0] = False                       # placeholder zero never enters stats
    drift = 0.0
    for s0, e0 in ranges:
        seg = sole_pts[s0:e0]
        drift = max(drift, float(np.linalg.norm(seg - seg[0], axis=-1).max()))
    return SideFootMetrics(
        n_ref_frames=int(mask.sum()),
        gap_p95_m=_p95(sole_gap[mask]),
        speed_p95_mps=_p95(speed[sel]),
        drift_max_m=drift,
        n_speed_frames=int(sel.sum()),
    )


def certified_robot_mask(
    intervals, hand_dist: np.ndarray, hand_local: np.ndarray, cfg,
) -> tuple[np.ndarray, int, int]:
    """Frames inside robot grasp intervals that pass the full hand check.

    Mirrors ``IndependentValidator._check_hands`` per-interval criteria
    (gap <= hand_gap_m, per-site anchor drift <= hand_anchor_error_m,
    box-frame relative speed <= hand_relative_speed_mps). Inferred contact
    alone is not certification: a slowly drifting grasp keeps its interval
    but fails the anchor check and contributes no certified frame.
    """
    T = hand_dist.shape[0]
    mask = np.zeros(T, dtype=bool)
    n_iv = n_cert = 0
    dt = 1.0 / cfg.fps
    for iv in intervals:
        if iv.support != "object":
            continue
        n_iv += 1
        seg_local = hand_local[iv.start:iv.end]
        gap_t = hand_dist[iv.start:iv.end]
        ok = float(gap_t.max()) <= cfg.hand_gap_m
        if iv.anchor is not None:
            anchor_err = np.linalg.norm(seg_local - iv.anchor, axis=-1).max(1)
            ok = ok and float(anchor_err.max()) <= cfg.hand_anchor_error_m
        if len(seg_local) > 1:
            rel_speed = np.linalg.norm(np.diff(seg_local, axis=0), axis=-1).max(1) / dt
            ok = ok and float(rel_speed.max()) <= cfg.hand_relative_speed_mps
        if ok:
            mask[iv.start:iv.end] = True
            n_cert += 1
    return mask, n_iv, n_cert


def hand_side_metrics(
    hand_dist: np.ndarray, hand_local: np.ndarray,
    robot_intervals, ref_ranges: list[tuple[int, int]],
    cfg, pen_key: np.ndarray,
) -> SideHandMetrics:
    T = hand_dist.shape[0]
    ref_mask = _mask_from_ranges(ref_ranges, T)
    cert_mask, n_iv, n_cert = certified_robot_mask(
        robot_intervals, hand_dist, hand_local, cfg)
    overlap = cert_mask & ref_mask
    g = np.maximum(hand_dist, 0.0)
    valid_frames = (g < HAND_VALID_GAP_M) & (pen_key <= PENETRATION_TOL_M)
    drift = 0.0
    for s0, e0 in ref_ranges:
        seg = hand_local[s0:e0]
        drift = max(drift, float(np.linalg.norm(seg - seg[0], axis=-1).max()))
    return SideHandMetrics(
        n_ref_frames=int(ref_mask.sum()),
        gap_p95_m=_p95(g[ref_mask]),
        coverage=float((g[ref_mask] < HAND_VALID_GAP_M).mean()),
        valid_coverage=float(valid_frames[ref_mask].mean()),
        certified_overlap_frames=int(overlap.sum()),
        longest_certified_overlap=_longest_true_run(overlap),
        drift_max_m=drift,
        n_robot_intervals=n_iv,
        n_robot_intervals_certified=n_cert,
    )


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
        cache = self.validator._cache
        cfg = self.validator.cfg

        # single forward pass per frame: keyframe penetration series
        # (per-frame validity + diagnostic violation ratio) and the
        # FRAME-ALIGNED task error. The reported penetration MAX comes from
        # the validator's adaptive protocol below, not from this series.
        pen_key = np.zeros(T)
        box_ground_pen = 0.0
        gap = {s: np.zeros(T) for s in HAND_SIDES}
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
            pen_key[t] = worst
            box_ground_pen = max(box_ground_pen, -self._dist(self.box_geom, self.ground_geom))
            for s in HAND_SIDES:
                gap[s][t] = min(self._dist(g, self.box_geom) for g in self._hand_geoms[s])
            # frame t robot keypoints vs frame t human keypoints (review fix 1)
            kpt_err[t] = float(np.linalg.norm(
                self.data.site_xpos[self.kpt_ids] - human[t][self.ref_kpt_idx], axis=-1
            ).mean())
        pen_valid = np.maximum(pen_key, 0.0)

        m = DemoMetrics(
            demo=demo, T=T, fps=fps, valid=True,
            penetration_max_m=float(rep.checks["penetration"].worst),
            penetration_violation_ratio=float((pen_valid > PENETRATION_TOL_M).mean()),
            task_mpjpe_p95_m=_p95(kpt_err),
            box_ground_penetration_max_m=float(box_ground_pen),
            validator_status=rep.status,
            validator_worst={k: v.worst for k, v in rep.checks.items()},
            validator_failed_checks=sorted(
                k for k, v in rep.checks.items() if not v.passed),
            reference_intervals=ref_sched.intervals,
        )

        m.ref_support_frames = {
            s: int(_mask_from_ranges(
                _frames_of(ref_sched, f"{'left' if s == 'l' else 'right'}_foot", "ground"), T).sum())
            for s in HAND_SIDES
        }
        m.ref_grasp_frames = {
            s: int(_mask_from_ranges(
                _frames_of(ref_sched, f"{'left' if s == 'l' else 'right'}_hand", "object"), T).sum())
            for s in HAND_SIDES
        }

        # feet: validator cached series, per-site worst protocols
        m.foot = {}
        for s, eff in (("l", "left_foot"), ("r", "right_foot")):
            ranges = _frames_of(ref_sched, eff, "ground")
            if not ranges:
                m.foot[s] = None
                continue
            m.foot[s] = foot_side_metrics(
                cache["sole_pts"][s], cache["sole_gap"][s], ranges, fps)

        # hands: validator cached box-frame series; certification is per
        # interval (validator criteria), intersected with the reference
        m.hand = {}
        for s, eff in (("l", "left_hand"), ("r", "right_hand")):
            ranges = _frames_of(ref_sched, eff, "object")
            if not ranges:
                m.hand[s] = None
                continue
            m.hand[s] = hand_side_metrics(
                gap[s], cache["hand_local"][s],
                [iv for iv in rep.schedule.intervals if iv.effector == eff],
                ranges, cfg, pen_valid,
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


def _bad(x) -> bool:
    """Missing or non-finite measurement (round-3 review fix 1).

    ``isnan`` alone lets ``inf`` through and ``None`` crashed the checks;
    every measured quantity below is decidable evidence, so None/NaN/Inf
    all mean the comparison lacks evidence — ``incomplete``, never a
    verdict on absent data.
    """
    if x is None:
        return True
    try:
        return not bool(np.isfinite(x))
    except (TypeError, ValueError):
        return True


def _both_denominator(m) -> int | None:
    """Both-hands common denominator frames: present, finite, non-negative.

    Round-5 review fix 2: ``both.get("denominator_frames", 0)`` read a
    MISSING key as "no both-grasp reference frames" (and the value None
    raised TypeError in ``> 0``) — a U0 row with the field deleted kept
    ``complete_evidence`` while g4 silently counted the demo as no-gain,
    so the remaining demos' gains could still carry ``local_preferred``.
    The denominator is decided by the SHARED human reference
    (binding-verified), so it is decidable evidence: missing/None/
    non-finite/negative is a gap, and P and U0 must agree on it exactly.

    Round-6 review fix 2: ``int()`` truncated BEFORE the sign check — -0.5
    read as a legal zero and 30.9 as 30, so P=30 / U0=30.9 "agreed". The
    denominator is an exact frame COUNT: a non-negative integer is
    required, a fractional value is invalid evidence, never silently
    truncated.
    """
    v = m.both.get("denominator_frames")
    if _bad(v):
        return None
    f = float(v)
    if f < 0 or f != int(f):
        return None
    return int(f)


def _json_num(x, nd: int = 6):
    """JSON-safe number: None for missing/non-finite (round-4 review fix 3).

    ``round(None)``/``np.isnan(None)`` raise, so a deficient row (metric
    set to None by a partial measurement) must not crash report
    serialization — it serializes as null and the evidence gaps carry the
    verdict instead.
    """
    return None if _bad(x) else round(float(x), nd)


def _evidence_gaps(
    pilot: list[str], p: dict, u: dict, binding_passed: dict[str, bool],
    u_label: str = "U0",
) -> list[str]:
    """Every way the comparison can lack decisive evidence (§3.4).

    Missing rows, invalid rows, non-finite headline metrics, side
    measurements missing *despite* existing reference frames, and unverified
    bindings all force ``incomplete`` — never a KeyError, never a silent
    ``no_quality_gain`` that would open the W1 path on absent data.
    """
    gaps: list[str] = []
    for d in pilot:
        pm, um = p.get(d), u.get(d)
        for tag, m in (("P", pm), (u_label, um)):
            if m is None:
                gaps.append(f"{d}:{tag}_row_missing")
            elif not m.valid:
                gaps.append(f"{d}:{tag}_invalid")
        if not binding_passed.get(d, False):
            gaps.append(f"{d}:binding_not_verified")
        if pm is None or um is None or not pm.valid or not um.valid:
            continue
        for name in ("task_mpjpe_p95_m", "penetration_max_m",
                     "penetration_violation_ratio"):
            for tag, m in (("P", pm), (u_label, um)):
                if _bad(getattr(m, name)):
                    gaps.append(f"{d}:{tag}_{name}_not_finite")
        for s in HAND_SIDES:
            if pm.ref_support_frames.get(s, 0) > 0:
                for tag, m in (("P", pm), (u_label, um)):
                    f = m.foot.get(s)
                    if (f is None or _bad(f.speed_p95_mps) or _bad(f.gap_p95_m)
                            or _bad(f.drift_max_m)):
                        gaps.append(f"{d}:{tag}_foot_{s}_missing_with_reference_frames")
            if pm.ref_grasp_frames.get(s, 0) > 0:
                for tag, m in (("P", pm), (u_label, um)):
                    h = m.hand.get(s)
                    if (h is None or _bad(h.gap_p95_m) or _bad(h.coverage)
                            or _bad(h.valid_coverage) or _bad(h.drift_max_m)):
                        gaps.append(f"{d}:{tag}_hand_{s}_missing_with_reference_frames")
        for tag, m in (("P", pm), (u_label, um)):
            denom = _both_denominator(m)
            if denom is None:
                gaps.append(f"{d}:{tag}_both_denominator_missing_or_invalid")
            elif denom > 0 and (_bad(m.both.get("valid_coverage"))
                                or _bad(m.both.get("gap_series_p95_m"))):
                gaps.append(f"{d}:{tag}_both_hands_not_finite")
        p_den, u_den = _both_denominator(pm), _both_denominator(um)
        if p_den is not None and u_den is not None and p_den != u_den:
            gaps.append(f"{d}:both_denominator_disagreement"
                        f"_p{p_den}_{u_label.lower()}_{u_den}")
    return gaps


def evaluate_local_preferred(
    pilot: list[str], p: dict[str, DemoMetrics], u: dict[str, DemoMetrics],
    binding_passed: dict[str, bool], u_label: str = "U0",
) -> dict:
    """Frozen §3.4 criteria. Returns the machine-readable gate record.

    Round-4 review fix 3: the EVIDENCE check runs before the quality gates
    and the verdict is taken from it — a deficient row (None task error,
    missing foot fields, a both-hands dict without keys) previously crashed
    g2/g3/g4 mid-comparison (TypeError/KeyError) instead of yielding
    ``incomplete``. The gate lambdas below stay in the record as diagnostics
    and are hardened so they can never raise on the same deficient rows.
    """
    # u_label only names the source in gaps/fields/reasons (the UR
    # comparison reports P vs UR); the quality computation is untouched
    gaps = _evidence_gaps(pilot, p, u, binding_passed, u_label=u_label)
    complete = not gaps
    gates: dict[str, dict] = {}

    def _avail(d) -> bool:
        pm, um = p.get(d), u.get(d)
        return (pm is not None and um is not None and pm.valid and um.valid)

    def _g1() -> dict:
        rows = {}
        for d in pilot:
            ok = (
                d in binding_passed and binding_passed[d] and _avail(d)
            )
            rows[d] = ok
        return {"per_demo": rows, "passed": all(rows.values())}

    def _g2() -> dict:
        rows = {}
        for d in pilot:
            if not _avail(d):
                rows[d] = False
                continue
            pt, ut = p[d].task_mpjpe_p95_m, u[d].task_mpjpe_p95_m
            rows[d] = (False if _bad(pt) or _bad(ut)
                       else bool(ut <= pt + TASK_P95_MARGIN_M))
        return {"per_demo": rows, "passed": all(rows.values())}

    def _g3() -> dict:
        rows = {}
        for d in pilot:
            if not _avail(d):
                rows[d] = {"passed": False, "penetration": False, "foot": {"unavailable": True}}
                continue
            pm, um = p[d], u[d]
            if _bad(pm.penetration_max_m) or _bad(um.penetration_max_m):
                pen_ok = False
            else:
                pen_ok = bool(um.penetration_max_m
                              <= max(pm.penetration_max_m, PENETRATION_TOL_M) + COMPARE_TOL)
            ok = pen_ok
            foot_detail = {}
            for s in HAND_SIDES:
                pf, uf = pm.foot.get(s), um.foot.get(s)
                if pf is None or pf.n_ref_frames == 0:
                    foot_detail[s] = "no_reference_support_frames"
                    continue
                if uf is None or uf.n_ref_frames == 0:
                    foot_detail[s] = f"{u_label.lower()}_missing_reference_frames"
                    ok = False
                    continue
                if (_bad(pf.speed_p95_mps) or _bad(pf.gap_p95_m)
                        or _bad(uf.speed_p95_mps) or _bad(uf.gap_p95_m)):
                    # evidence gap fires for the same row; the gate record
                    # must still be writable, not crash on None arithmetic
                    foot_detail[s] = "foot_metric_not_finite"
                    ok = False
                    continue
                s_ok = (
                    uf.speed_p95_mps <= max(pf.speed_p95_mps, FOOT_SPEED_STRICT_MPS) + COMPARE_TOL
                    and uf.gap_p95_m <= max(pf.gap_p95_m, FOOT_GAP_STRICT_M) + COMPARE_TOL
                )
                foot_detail[s] = bool(s_ok)
                ok = ok and s_ok
            rows[d] = {"passed": ok, "penetration": pen_ok, "foot": foot_detail}
        return {"per_demo": rows,
                "passed": all(r["passed"] for r in rows.values())}

    def _g4() -> dict:
        cov_gains, gap_rels = [], []
        for d in pilot:
            if not _avail(d):
                cov_gains.append(None)
                gap_rels.append(None)
                continue
            pb, ub = p[d].both, u[d].both
            p_den, u_den = _both_denominator(p[d]), _both_denominator(u[d])
            if not p_den or not u_den:
                # None (missing/invalid denominator — the evidence gap fires
                # and forces incomplete) or 0 (no both-grasp reference
                # frames): the demo contributes no gain vote either way
                cov_gains.append(None)
                gap_rels.append(None)
                continue
            pvc, uvc = pb.get("valid_coverage"), ub.get("valid_coverage")
            if not _bad(pvc) and not _bad(uvc):
                cov_gains.append(100.0 * (uvc - pvc))
            else:
                cov_gains.append(None)
            pg, ug = pb.get("gap_series_p95_m"), ub.get("gap_series_p95_m")
            if _bad(pg) or _bad(ug) or pg <= 0:
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

    def _g5() -> dict:
        """§3.4 item 5: no degradation of any family may buy the g4 gains.

        Families:

        * drift — within-interval hand/foot anchor drift, registered margin
          (the aggregate p95 gates of g2/g3 cannot see a drift that stays
          inside the interval bounds);
        * new hard failures — every validator check U0 FAILS that P PASSED
          (joint limits, velocities, penetration tolerance, hard contact
          checks). "P passed, U0 newly fails" is the §3.4 item 3 degradation
          shape the aggregate gates miss;
        * keyframe penetration violation ratio — U0 exceeding P's ratio by
          more than the comparison tolerance (0% -> 100% is the extreme;
          g3's max-penetration gate alone cannot see a mass change at a
          depth below the max).
        """
        rows = {}
        for d in pilot:
            if not _avail(d):
                rows[d] = {"passed": False, "drift": {"unavailable": True}}
                continue
            pm, um = p[d], u[d]
            detail = {}
            ok = True
            for s in HAND_SIDES:
                for kind in ("hand", "foot"):
                    pa, ua = getattr(pm, kind).get(s), getattr(um, kind).get(s)
                    if pa is None or _bad(pa.drift_max_m):
                        continue        # no P reference evidence for this side
                    if ua is None or _bad(ua.drift_max_m):
                        ok = False
                        detail[f"{kind}_{s}"] = f"{u_label.lower()}_measurement_missing"
                        continue
                    if ua.drift_max_m > pa.drift_max_m + DRIFT_DEGRADATION_MARGIN_M:
                        ok = False
                        detail[f"{kind}_{s}"] = {
                            "p_drift_m": pa.drift_max_m,
                            f"{u_label.lower()}_drift_m": ua.drift_max_m}
            new_failures = sorted(
                set(um.validator_failed_checks) - set(pm.validator_failed_checks))
            if new_failures:
                ok = False
                detail["new_validator_failures"] = {
                    "p_failed": list(pm.validator_failed_checks),
                    f"{u_label.lower()}_failed": list(um.validator_failed_checks),
                    "new": new_failures,
                }
            pr, ur = (pm.penetration_violation_ratio,
                      um.penetration_violation_ratio)
            if _bad(pr) or _bad(ur):
                ok = False
                detail["penetration_violation_ratio_missing"] = True
            elif ur > pr + COMPARE_TOL:
                ok = False
                detail["penetration_violation_ratio_increased"] = {
                    "p_ratio": pr, f"{u_label.lower()}_ratio": ur,
                }
            rows[d] = {"passed": ok, "degradation": detail}
        return {
            "per_demo": rows, "margin_m": DRIFT_DEGRADATION_MARGIN_M,
            "note": "§3.4 item 5: drift family + new validator hard failures "
                    "+ keyframe penetration violation ratio; task/slip are "
                    "additionally gated by g2/g3",
            "passed": all(r["passed"] for r in rows.values()),
        }

    gates["g1_identity_and_availability"] = _g1()
    gates["g2_task_mpjpe"] = _g2()
    gates["g3_penetration_and_feet"] = _g3()
    gates["g4_hand_improvement"] = _g4()
    gates["g5_degradation"] = _g5()
    if not complete:
        verdict = "incomplete"
    elif all(g["passed"] for g in gates.values()):
        verdict = "local_preferred"
    else:
        verdict = "no_quality_gain"
    return {"gates": gates, "evidence_gaps": gaps,
            "complete_evidence": complete, "verdict": verdict}
