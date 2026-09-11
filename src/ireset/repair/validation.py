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
    # provisional velocity bounds (model carries none); locked per plan 10.1 in M2
    joint_velocity_limit_radps: float = 6.0
    base_lin_velocity_limit_mps: float = 1.5
    base_ang_velocity_limit_radps: float = 3.0
    # contact inference thresholds
    hand_enter_m: float = 0.02
    hand_exit_m: float = 0.06
    foot_enter_m: float = 0.012
    foot_exit_m: float = 0.03
    foot_entry_speed_mps: float = 0.3
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
    max_check_interval_m: float = 0.0

    def summary(self) -> str:
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

    # -- core ----------------------------------------------------------------

    def validate(self, scene_qpos: np.ndarray) -> ValidationReport:
        q = np.asarray(scene_qpos, dtype=float)
        T = q.shape[0]
        cfg = self.cfg
        checks: dict[str, CheckResult] = {}

        checks["joint_limits"] = self._check_joint_limits(q)
        checks["velocities"] = self._check_velocities(q)
        pen_worst, pen_where, max_interval = self._check_penetration(q)
        checks["penetration"] = CheckResult(
            pen_worst <= cfg.penetration_tolerance_m, pen_worst,
            cfg.penetration_tolerance_m, pen_where,
        )
        report = ValidationReport(passed=False, checks=checks, max_check_interval_m=max_interval)

        schedule = self._infer_robot_schedule(q)
        report.schedule = schedule
        checks["foot_support"] = self._check_feet(q, schedule)
        checks["hand_grasp"] = self._check_hands(q, schedule)

        report.passed = all(c.passed for c in checks.values())
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

    def _check_penetration(self, q: np.ndarray) -> tuple[float, str, float]:
        """Max penetration over all pairs at keyframes + adaptive midpoints."""
        cfg = self.cfg
        d = mujoco.MjData(self.model)
        pairs = self.pairs.all_pairs
        worst = 0.0
        where = ""
        max_interval = 0.0

        def probe(qp: np.ndarray, tag: str) -> float:
            nonlocal worst, where
            d.qpos = qp
            mujoco.mj_forward(self.model, d)
            local = 0.0
            for ga, gb, _kind in pairs:
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
                lo, hi = q[t], q[t + 1]
                motion = np.linalg.norm(hi[0:3] - lo[0:3]) + np.abs(hi[7:36] - lo[7:36]).max()
                mid = self._interpolate(lo, hi, 0.5)
                probe(mid, f"t={t}.5")
                max_interval = max(max_interval, motion)
        return worst, where, max_interval

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
        cfg = self.cfg
        d = mujoco.MjData(self.model)
        T = q.shape[0]
        schedule = ContactSchedule()

        hand_dist = {s: np.full(T, np.inf) for s in self.hand_sites}
        sole_z = {s: np.full(T, np.inf) for s in self.sole_sites}
        hand_world = {s: np.zeros((T, 3)) for s in self.hand_sites}
        sole_world = {s: np.zeros((T, 3)) for s in self.sole_sites}
        box_R = np.zeros((T, 3, 3)); box_t = np.zeros((T, 3))
        for t in range(T):
            d.qpos = q[t]
            mujoco.mj_forward(self.model, d)
            R_g = d.geom_xmat[self.box_geom].reshape(3, 3)
            x_g = d.geom_xpos[self.box_geom]
            box_R[t], box_t[t] = R_g, x_g
            for s, ids in self.hand_sites.items():
                pts = d.site_xpos[ids]
                hand_world[s][t] = pts.mean(0)
                dists = [
                    mujoco.mj_geomDistance(self.model, d, self._hand_geom(s, k), self.box_geom, 1.0, self._fromto)
                    for k in range(3)
                ]
                hand_dist[s][t] = min(dists)
            for s, ids in self.sole_sites.items():
                sole_world[s][t] = d.site_xpos[ids].mean(0)
                sole_z[s][t] = d.site_xpos[ids][:, 2].min()

        for s, name in (("l", "left_hand"), ("r", "right_hand")):
            for a, b in hysteresis_intervals(hand_dist[s], cfg.hand_enter_m, cfg.hand_exit_m, cfg.min_duration):
                anchor_local = box_R[a].T @ (hand_world[s][a] - box_t[a])
                schedule.intervals.append(ContactInterval(name, a, b, "object", anchor=anchor_local))
        dt = 1.0 / cfg.fps
        for s, name in (("l", "left_foot"), ("r", "right_foot")):
            pts = sole_world[s]
            speed = np.concatenate([[np.inf], np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt])
            gated = np.where(speed < cfg.foot_entry_speed_mps, sole_z[s], np.inf)
            for a, b in hysteresis_intervals(gated, cfg.foot_enter_m, cfg.foot_exit_m, cfg.min_duration):
                schedule.intervals.append(ContactInterval(name, a, b, "ground", anchor=pts[a].copy()))

        self._cache = dict(hand_dist=hand_dist, hand_world=hand_world, sole_world=sole_world,
                           sole_z=sole_z, box_R=box_R, box_t=box_t)
        return schedule

    def _hand_geom(self, side: str, k: int) -> int:
        names = {"l": ("left_rubber_hand_link", "left_thumb_link", "left_pinky_link"),
                 "r": ("right_rubber_hand_link", "right_thumb_link", "right_pinky_link")}
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, names[side][k])

    def _check_feet(self, q: np.ndarray, schedule: ContactSchedule) -> CheckResult:
        cfg = self.cfg
        cache = self._cache
        dt = 1.0 / cfg.fps
        worst = 0.0
        detail = []
        side_of = {"left_foot": "l", "right_foot": "r"}
        for iv in schedule.intervals:
            if iv.support != "ground" or iv.effector not in side_of:
                continue
            s = side_of[iv.effector]
            pts = cache["sole_world"][s][iv.start : iv.end]
            z = cache["sole_z"][s][iv.start : iv.end]
            if len(pts) > 1:
                speed = np.linalg.norm(np.diff(pts, axis=0), axis=1).max() / dt
                worst = max(worst, speed / cfg.foot_speed_mps)
                detail.append(f"{iv.effector} speed={speed:.3f}")
            drift = np.linalg.norm(pts - pts[0], axis=1).max()
            worst = max(worst, drift / cfg.foot_anchor_drift_m)
            gap = z.max()
            worst = max(worst, gap / cfg.foot_ground_gap_m)
        return CheckResult(worst <= 1.0, worst, 1.0, "; ".join(detail[-3:]))

    def _check_hands(self, q: np.ndarray, schedule: ContactSchedule) -> CheckResult:
        cfg = self.cfg
        cache = self._cache
        dt = 1.0 / cfg.fps
        worst = 0.0
        detail = []
        side_of = {"left_hand": "l", "right_hand": "r"}
        for iv in schedule.intervals:
            if iv.support != "object" or iv.effector not in side_of or iv.anchor is None:
                continue
            s = side_of[iv.effector]
            seg = slice(iv.start, iv.end)
            # gap: inferred distance already bounded by enter threshold at start;
            # check it stays within hand_gap_m of the box
            gap = cache["hand_dist"][s][seg].max()
            worst = max(worst, gap / cfg.hand_gap_m)
            # anchor drift in box frame
            local = np.einsum("tij,tj->ti", cache["box_R"][seg], cache["hand_world"][s][seg] - cache["box_t"][seg])
            anchor_err = np.linalg.norm(local - iv.anchor, axis=1).max()
            worst = max(worst, anchor_err / cfg.hand_anchor_error_m)
            # box-frame relative speed of the hand centroid
            if local.shape[0] > 1:
                rel_speed = np.linalg.norm(np.diff(local, axis=0), axis=1).max() / dt
                worst = max(worst, rel_speed / cfg.hand_relative_speed_mps)
            detail.append(f"{iv.effector} gap={gap:.3f} anchor={anchor_err:.3f}")
        return CheckResult(worst <= 1.0, worst, 1.0, "; ".join(detail[-3:]))
