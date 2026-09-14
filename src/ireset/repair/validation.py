"""Independent trajectory validation (plan sections 6.3, 10.1).

The validator is deliberately independent of any optimizer: it consumes a
scene qpos trajectory (MuJoCo layout, [T, 43] = robot 36 + box 7), infers
contacts from the *generated geometry itself*, and checks:

- penetration over **all** non-excluded collision pairs (self-collision
  included) at keyframes plus motion-adaptive interpolated states;
- joint limits and (provisional) velocity bounds;
- foot support: sole speed, anchor drift, ground gap during inferred support;
- hand grasp: site-to-box distance, anchor drift, box-frame relative speed
  during inferred grasp intervals.

Failure modes return explicit failed checks with locations; nothing is
silently clipped. ``ValidationConfig`` defaults mirror
configs/eval/protocol.yaml (the yaml is the locked source for experiments).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ireset.embodied.contact_sites import ContactInterval, ContactSchedule, hysteresis_intervals
from ireset.embodied.g1_spec import G1Spec, HAND_SITES, SOLE_SITES
from ireset.geometry.collision_pairs import CollisionPairs, build_collision_pairs


@dataclass
class ValidationConfig:
    penetration_tolerance_m: float = 0.002
    joint_limit_tolerance_rad: float = 1e-5
    # velocity bounds calibrated 2026-09-12 from the teacher distribution
    # (48 originals: max 10.6 rad/s / 1.64 m/s / 6.26 rad/s) plus margin
    joint_velocity_limit_radps: float = 12.0
    base_lin_velocity_limit_mps: float = 1.8
    base_ang_velocity_limit_radps: float = 7.0
    # contact inference thresholds. Enter thresholds must not exceed the gate
    # the resulting interval must satisfy (enforced in __post_init__): an
    # interval created at a distance the gate already rejects fails for a
    # configuration reason, not a motion reason.
    hand_enter_m: float = 0.01
    hand_exit_m: float = 0.03
    foot_enter_m: float = 0.005
    foot_exit_m: float = 0.02
    # P0.1 L1 (2026-09-13): interval entry is speed-gated and the entry gate
    # must not exceed the check gate it feeds (same principle as the distance
    # enter<=gate invariant: an interval created at a speed its own check
    # rejects fails for a configuration reason, not a motion reason).
    # Support := quasi-static; fast approach/release stretches near the box
    # or ground are transition, not certified contact. Supersedes the single
    # foot_entry_speed_mps=0.3.
    hand_entry_speed_mps: float = 0.02   # <= hand_relative_speed_mps
    foot_entry_speed_mps: float = 0.02   # <= foot_speed_mps
    min_duration: int = 3
    # contact quality gates (plan 10.1)
    hand_gap_m: float = 0.01
    hand_anchor_error_m: float = 0.01
    hand_relative_speed_mps: float = 0.02
    foot_speed_mps: float = 0.02
    foot_anchor_drift_m: float = 0.01
    foot_ground_gap_m: float = 0.005
    # interpolated-state checks
    check_interpolated_states: bool = True
    max_substep_motion_m: float = 0.05
    max_subdivisions: int = 2
    fps: float = 30.0

    def __post_init__(self) -> None:
        if self.hand_enter_m > self.hand_gap_m:
            raise ValueError(f"hand_enter_m {self.hand_enter_m} > hand_gap_m {self.hand_gap_m}")
        if self.foot_enter_m > self.foot_ground_gap_m:
            raise ValueError(f"foot_enter_m {self.foot_enter_m} > foot_ground_gap_m {self.foot_ground_gap_m}")
        if self.hand_exit_m < self.hand_enter_m or self.foot_exit_m < self.foot_enter_m:
            raise ValueError("exit threshold must not be tighter than enter (hysteresis)")
        if self.hand_entry_speed_mps > self.hand_relative_speed_mps:
            raise ValueError("hand_entry_speed_mps must not exceed hand_relative_speed_mps")
        if self.foot_entry_speed_mps > self.foot_speed_mps:
            raise ValueError("foot_entry_speed_mps must not exceed foot_speed_mps")


@dataclass
class CheckResult:
    passed: bool
    worst: float
    limit: float
    location: str = ""
    detail: str = ""


@dataclass
class ValidationReport:
    passed: bool
    checks: dict[str, CheckResult] = field(default_factory=dict)
    schedule: ContactSchedule | None = None
    #: max Cartesian site displacement between consecutive *checked* states
    #: (post-subdivision), per plan 6.3 "report the max check interval"
    max_check_interval_m: float = 0.0
    #: True when max_subdivisions capped the subdivision below what
    #: max_substep_motion_m would require (reported, never silently passed)
    subdivision_capped: bool = False
    #: passed / rejected_nonfinite / rejected_degenerate_quaternion /
    #: rejected_tolerance
    status: str = "passed"

    def summary(self) -> str:
        if self.status != "passed" and self.status != "rejected_tolerance":
            return self.status
        if self.passed:
            return "passed"
        failed = [k for k, v in self.checks.items() if not v.passed]
        return "failed: " + ", ".join(failed)


class IndependentValidator:
    def __init__(self, spec: G1Spec | None = None, cfg: ValidationConfig | None = None,
                 pairs: CollisionPairs | None = None) -> None:
        self.spec = spec or G1Spec()
        self.cfg = cfg or ValidationConfig()
        self.model = self.spec.load_scene()
        self.pairs = pairs or build_collision_pairs(self.spec, scene=True)
        self._fromto = np.zeros(6)
        ids = lambda names: [  # noqa: E731
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names
        ]
        self.hand_sites = {s: ids(list(v.values())) for s, v in HAND_SITES.items()}
        self.sole_sites = {s: ids(list(v)) for s, v in SOLE_SITES.items()}
        self.box_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "largebox")
        self.ground_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        # sole collision-sphere geoms: the *surface* gap to the ground must
        # come from mj_geomDistance (exact, works for tilted feet); site
        # centers are 5 mm spheres and are not the contact surface.
        self.sole_geoms = {
            s: [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM,
                    f"{'left' if s == 'l' else 'right'}_ankle_roll_sphere_{i}_link",
                )
                for i in range(1, 6)
            ]
            for s in ("l", "r")
        }
        self._motion_data = mujoco.MjData(self.model)
        self._hand_geoms = self._hand_geoms_for()

    # -- core ----------------------------------------------------------------

    def validate(self, scene_qpos: np.ndarray, exclude: tuple[str, ...] = ()) -> ValidationReport:
        """Validate a scene trajectory.

        ``exclude`` names collision-pair kinds to skip (e.g. ``box_ground``:
        the box trajectory is a fixed task input the repairer cannot alter,
        and the convex-hull of the scanned bin over-approximates its bottom
        by several mm — an input/model property, not a repair failure).
        """
        q = np.asarray(scene_qpos, dtype=float)
        T = q.shape[0]
        cfg = self.cfg

        # Explicit pre-gates: non-finite input and degenerate quaternions are
        # rejected before any geometric/scipy call (plan 1.3: non-finite
        # results must be rejected with an explicit status, never a crash or
        # a NaN worst value).
        if not np.isfinite(q).all():
            bad = np.argwhere(~np.isfinite(q))
            return ValidationReport(
                passed=False, status="rejected_nonfinite",
                checks={"finite": CheckResult(False, float("inf"), 0.0,
                        location=f"t={bad[0, 0]} qpos_idx={bad[0, 1]}",
                        detail=f"{len(bad)} non-finite entries")},
            )
        for name, sl in (("root_quat", slice(3, 7)), ("box_quat", slice(39, 43))):
            n = np.linalg.norm(q[:, sl], axis=1)
            if (n < 1e-8).any():
                return ValidationReport(
                    passed=False, status="rejected_degenerate_quaternion",
                    checks={"quaternion": CheckResult(False, float(n.min()), 1e-8,
                            location=f"{name} t={int(np.argmin(n))}")},
                )

        checks: dict[str, CheckResult] = {}
        self._exclude = set(exclude)
        checks["joint_limits"] = self._check_joint_limits(q)
        checks["velocities"] = self._check_velocities(q)
        pen_worst, pen_where, max_interval, capped = self._check_penetration(q)
        checks["penetration"] = CheckResult(
            pen_worst <= cfg.penetration_tolerance_m, pen_worst,
            cfg.penetration_tolerance_m, pen_where,
        )
        report = ValidationReport(
            passed=False, checks=checks, max_check_interval_m=max_interval,
            subdivision_capped=capped, status="rejected_tolerance",
        )

        schedule = self._infer_robot_schedule(q)
        report.schedule = schedule
        checks["foot_support"] = self._check_feet(q, schedule)
        checks["hand_grasp"] = self._check_hands(q, schedule)

        report.passed = all(c.passed for c in checks.values())
        if report.passed:
            report.status = "passed"
        return report

    # -- components ------------------------------------------------------------

    def _check_joint_limits(self, q: np.ndarray) -> CheckResult:
        lo, hi = self.spec.joint_range[:, 0], self.spec.joint_range[:, 1]
        joints = q[:, 7:36]
        viol = np.maximum(lo - joints, joints - hi).max()
        worst_t = int(np.argmax(np.maximum(lo - joints, joints - hi).max(1)))
        return CheckResult(
            viol <= self.cfg.joint_limit_tolerance_rad, viol,
            self.cfg.joint_limit_tolerance_rad, f"t={worst_t}",
        )

    def _check_velocities(self, q: np.ndarray) -> CheckResult:
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        T = q.shape[0]
        if T < 2:
            return CheckResult(True, 0.0, 0.0, "T<2")
        dq = np.abs(np.diff(q[:, 7:36], axis=0)).max() / dt
        dpos = np.linalg.norm(np.diff(q[:, 0:3], axis=0), axis=1).max() / dt
        # world-frame angular increment via retract-style finite difference
        from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz

        R = quat_to_mat_wxyz(q[:, 3:7])
        Rrel = np.einsum("tij,tkj->tik", R[1:], R[:-1])  # R1 @ R0^T
        ang = np.arccos(np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1))
        w = ang.max() / dt
        worst = max(dq / cfg.joint_velocity_limit_radps, dpos / cfg.base_lin_velocity_limit_mps,
                    w / cfg.base_ang_velocity_limit_radps)
        return CheckResult(worst <= 1.0, worst, 1.0, f"joint={dq:.2f}rad/s pos={dpos:.2f}m/s ang={w:.2f}rad/s")

    def _all_site_xpos(self, qp: np.ndarray, d: mujoco.MjData) -> np.ndarray:
        d.qpos = qp
        mujoco.mj_forward(self.model, d)
        return d.site_xpos.copy()

    def _frame_motion_m(self, qa: np.ndarray, qb: np.ndarray) -> float:
        """Max collision-relevant displacement between two states, in meters.

        Replaces the dimensionally inconsistent ``|dpos| + max|dq|`` metric
        (joint radians and base meters are not addable). Robot-side motion is
        the max site displacement; the box contributes its geom-center
        displacement so that box-only motion still triggers subdivision.
        """
        d = self._motion_data
        xa = self._all_site_xpos(qa, d)
        xb = self._all_site_xpos(qb, d)
        motion = float(np.linalg.norm(xb - xa, axis=-1).max())
        ba = self._box_geom_xpos(qa, d)
        bb = self._box_geom_xpos(qb, d)
        motion = max(motion, float(np.linalg.norm(bb - ba)))
        return motion

    def _box_geom_xpos(self, qp: np.ndarray, d: mujoco.MjData) -> np.ndarray:
        d.qpos = qp
        mujoco.mj_forward(self.model, d)
        return d.geom_xpos[self.box_geom].copy()

    def _check_penetration(self, q: np.ndarray) -> tuple[float, str, float, bool]:
        """Max penetration over all pairs at keyframes + adaptive subdivisions.

        Each frame pair is subdivided so that the max site motion between
        checked states stays under ``max_substep_motion_m`` (up to
        ``max_subdivisions`` extra levels); ``max_check_interval_m`` reports
        the *post-subdivision* interval, and ``capped`` flags pairs where the
        budget could not be met.
        """
        cfg = self.cfg
        d = mujoco.MjData(self.model)
        self._motion_data = mujoco.MjData(self.model)
        pairs = self.pairs.all_pairs
        worst = 0.0
        where = ""
        max_interval = 0.0
        capped = False

        def probe(qp: np.ndarray, tag: str) -> float:
            nonlocal worst, where
            d.qpos = qp
            mujoco.mj_forward(self.model, d)
            local = 0.0
            for ga, gb, _kind in pairs:
                if _kind in getattr(self, "_exclude", ()):
                    continue
                dist = mujoco.mj_geomDistance(self.model, d, ga, gb, 1.0, self._fromto)
                if dist < 0:
                    local = max(local, -dist)
            if local > worst:
                worst, where = local, tag
            return local

        for t in range(q.shape[0]):
            probe(q[t], f"t={t}")
        if cfg.check_interpolated_states:
            for t in range(q.shape[0] - 1):
                motion = self._frame_motion_m(q[t], q[t + 1])
                n_sub = int(np.clip(np.ceil(motion / cfg.max_substep_motion_m), 1, cfg.max_subdivisions + 1))
                for i in range(1, n_sub):
                    probe(self._interpolate(q[t], q[t + 1], i / n_sub), f"t={t}+{i}/{n_sub}")
                interval = motion / n_sub
                max_interval = max(max_interval, interval)
                if interval > cfg.max_substep_motion_m:
                    capped = True
        return worst, where, max_interval, capped

    def _interpolate(self, qa: np.ndarray, qb: np.ndarray, s: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation as Rot, Slerp

        out = qa.copy()
        out[0:3] = (1 - s) * qa[0:3] + s * qb[0:3]
        ra = Rot.from_quat(qa[3:7][[1, 2, 3, 0]])
        rb = Rot.from_quat(qb[3:7][[1, 2, 3, 0]])
        out[3:7] = Slerp([0, 1], Rot.concatenate([ra, rb]))(s).as_quat()[[3, 0, 1, 2]]
        out[7:36] = (1 - s) * qa[7:36] + s * qb[7:36]
        out[36:39] = (1 - s) * qa[36:39] + s * qb[36:39]
        ba = Rot.from_quat(qa[39:43][[1, 2, 3, 0]])
        bb = Rot.from_quat(qb[39:43][[1, 2, 3, 0]])
        out[39:43] = Slerp([0, 1], Rot.concatenate([ba, bb]))(s).as_quat()[[3, 0, 1, 2]]
        return out

    def _infer_robot_schedule(self, q: np.ndarray) -> ContactSchedule:
        """Infer contacts from generated geometry, per-site anchors.

        Caches per-site world positions (hands [T,3,3], soles [T,5,3]) and
        per-site box-frame hand positions; sole *gap* uses true MuJoCo
        sphere-surface distance to the ground, not site-center height.
        """
        cfg = self.cfg
        d = mujoco.MjData(self.model)
        T = q.shape[0]
        schedule = ContactSchedule()

        hand_dist = {s: np.full(T, np.inf) for s in self.hand_sites}
        sole_gap = {s: np.full(T, np.inf) for s in self.sole_sites}
        hand_pts = {s: np.zeros((T, 3, 3)) for s in self.hand_sites}
        sole_pts = {s: np.zeros((T, 5, 3)) for s in self.sole_sites}
        hand_local = {s: np.zeros((T, 3, 3)) for s in self.hand_sites}
        for t in range(T):
            d.qpos = q[t]
            mujoco.mj_forward(self.model, d)
            R_g = d.geom_xmat[self.box_geom].reshape(3, 3)
            x_g = d.geom_xpos[self.box_geom]
            for s, ids in self.hand_sites.items():
                pts = d.site_xpos[ids]                                  # [3,3]
                hand_pts[s][t] = pts
                hand_local[s][t] = (R_g.T @ (pts - x_g).T).T
                hand_dist[s][t] = min(
                    mujoco.mj_geomDistance(self.model, d, g, self.box_geom, 1.0, self._fromto)
                    for g in self._hand_geoms[s]
                )
            for s, ids in self.sole_sites.items():
                sole_pts[s][t] = d.site_xpos[ids]                       # [5,3]
                sole_gap[s][t] = min(
                    mujoco.mj_geomDistance(self.model, d, g, self.ground_geom, 1.0, self._fromto)
                    for g in self.sole_geoms[s]
                )

        dt = 1.0 / cfg.fps
        for s, name in (("l", "left_hand"), ("r", "right_hand")):
            # speed-gated entry (P0.1 L1): fast approach/release stretches are
            # transition, not certified grasp; the entry gate never exceeds
            # the rel_speed check it feeds
            hpts = hand_pts[s]
            h_speed = np.concatenate(
                [[np.inf], np.linalg.norm(np.diff(hpts, axis=0), axis=2).max(1) / dt]
            )
            gated = np.where(h_speed < cfg.hand_entry_speed_mps, hand_dist[s], np.inf)
            for a, b in hysteresis_intervals(gated, cfg.hand_enter_m, cfg.hand_exit_m, cfg.min_duration):
                anchor = hand_local[s][a].copy()                        # [3,3] sites
                schedule.intervals.append(ContactInterval(name, a, b, "object", anchor=anchor))
        for s, name in (("l", "left_foot"), ("r", "right_foot")):
            pts = sole_pts[s]
            per_site_speed = np.concatenate(
                [[np.inf], np.linalg.norm(np.diff(pts, axis=0), axis=2).max(1) / dt]
            )
            gated = np.where(per_site_speed < cfg.foot_entry_speed_mps, sole_gap[s], np.inf)
            for a, b in hysteresis_intervals(gated, cfg.foot_enter_m, cfg.foot_exit_m, cfg.min_duration):
                schedule.intervals.append(ContactInterval(name, a, b, "ground", anchor=pts[a].copy()))

        self._cache = dict(hand_dist=hand_dist, sole_gap=sole_gap, hand_pts=hand_pts,
                           sole_pts=sole_pts, hand_local=hand_local)
        return schedule

    def _hand_geoms_for(self) -> dict[str, list[int]]:
        names = {
            s: [f"{'left' if s == 'l' else 'right'}_{part}_link"
                for part in ("rubber_hand", "thumb", "pinky")]
            for s in ("l", "r")
        }
        return {
            s: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in lst]
            for s, lst in names.items()
        }

    def _check_feet(self, q: np.ndarray, schedule: ContactSchedule) -> CheckResult:
        """Per-site support checks (plan 10.1: *all* constrained sites).

        A foot pivoting in place keeps its centroid fixed while individual
        sole sites sweep — centroid metrics would miss it.
        """
        cfg = self.cfg
        cache = self._cache
        dt = 1.0 / cfg.fps
        worst = 0.0
        where = ""
        detail = []
        side_of = {"left_foot": "l", "right_foot": "r"}
        for iv in schedule.intervals:
            if iv.support != "ground" or iv.effector not in side_of:
                continue
            s = side_of[iv.effector]
            pts = cache["sole_pts"][s][iv.start : iv.end]        # [n, 5, 3]
            gap = cache["sole_gap"][s][iv.start : iv.end]
            if len(pts) > 1:
                sp = np.linalg.norm(np.diff(pts, axis=0), axis=-1).max(1) / dt
                t_w = int(np.argmax(sp))
                ratio = float(sp.max() / cfg.foot_speed_mps)
                if ratio > worst:
                    worst, where = ratio, f"{iv.effector} speed t={iv.start + 1 + t_w}"
                detail.append(f"{iv.effector} speed={sp.max():.3f}")
            drift = np.linalg.norm(pts - pts[0], axis=-1).max()
            dr_ratio = float(drift / cfg.foot_anchor_drift_m)
            if dr_ratio > worst:
                worst, where = dr_ratio, f"{iv.effector} drift"
            gap_ratio = float(gap.max() / cfg.foot_ground_gap_m)
            if gap_ratio > worst:
                worst, where = gap_ratio, f"{iv.effector} gap t={iv.start + int(np.argmax(gap))}"
        return CheckResult(worst <= 1.0, worst, 1.0, where, "; ".join(detail[-3:]))

    def _check_hands(self, q: np.ndarray, schedule: ContactSchedule) -> CheckResult:
        """Per-site grasp checks: gap, per-site anchor drift, relative speed."""
        cfg = self.cfg
        cache = self._cache
        dt = 1.0 / cfg.fps
        worst = 0.0
        where = ""
        detail = []
        side_of = {"left_hand": "l", "right_hand": "r"}
        for iv in schedule.intervals:
            if iv.support != "object" or iv.effector not in side_of or iv.anchor is None:
                continue
            s = side_of[iv.effector]
            seg = slice(iv.start, iv.end)
            gap = float(cache["hand_dist"][s][seg].max())
            gap_ratio = gap / cfg.hand_gap_m
            if gap_ratio > worst:
                worst, where = gap_ratio, f"{iv.effector} gap t={seg.start + int(np.argmax(cache['hand_dist'][s][seg]))}"
            local = cache["hand_local"][s][seg]                    # [n, 3 sites, 3]
            anchor_err = np.linalg.norm(local - iv.anchor, axis=-1).max()
            ae_ratio = float(anchor_err / cfg.hand_anchor_error_m)
            if ae_ratio > worst:
                worst, where = ae_ratio, f"{iv.effector} anchor t={seg.start}"
            if local.shape[0] > 1:
                rs = np.linalg.norm(np.diff(local, axis=0), axis=-1).max(1) / dt
                rs_ratio = float(rs.max() / cfg.hand_relative_speed_mps)
                if rs_ratio > worst:
                    worst, where = rs_ratio, f"{iv.effector} rel_speed t={seg.start + 1 + int(np.argmax(rs))}"
            detail.append(f"{iv.effector} gap={gap:.3f} anchor={anchor_err:.3f}")
        return CheckResult(worst <= 1.0, worst, 1.0, where, "; ".join(detail[-3:]))
