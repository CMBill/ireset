"""Constraint/objective linearization and QP assembly at a state (plan 8.2).

Everything is expressed in the 35-dim tangent xi = (dp, dtheta_world, dq).
Residuals are unit-normalized by the step scale D (root t 0.02 m, root r
0.05 rad, joint 0.1 rad) so weights are dimensionless.

Fixed sparsity: constraint rows are allocated at fixed offsets with inactive
rows set to [-inf, +inf] so one OSQP workspace can be warm-started across
SCP iterations and frames (plan M2: "OSQP 复用").

Row layout (n_rows = N_FIXED + MAX_PAIRS):

  [0 : 29)         joint limits          q + dq in [lo, hi]
  [29 : 58)        joint velocity        |q + dq - q_prev| <= dt*vj
  [58 : 61)        root lin velocity     |p + dp - p_prev| <= dt*vl
  [61 : 64)        root rot velocity     |r_rot_prev + dtheta| <= dt*wa
  [64 : 64+35)     trust region          |D^-1 xi| <= trust
  [99 : 99+M)      active collision      d + J xi in band (clearance/contact)
  [99+M : 99+M+18)  hand anchors          x_local + J xi in anchor +- tol
  [99+M+18 : +30)   foot anchors          x + J xi in anchor +- tol
  [99+M+48 : +30)   foot support speed    |x - x_prev + J xi| <= dt*v_foot
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ireset.embodied.g1_spec import G1Spec, HAND_SITES, SOLE_SITES, KPT_TO_SMPLH
from ireset.geometry.collision_pairs import CollisionPairs
from ireset.kinematics.mujoco_adapter import N_TANGENT, retract_batch
from ireset.kinematics.rotation_adapter import so3_log_np, quat_to_mat_wxyz

MAX_PAIRS = 48
N_HAND_ROWS = 18   # 2 hands x 3 sites x 3 axes
N_FOOT_ROWS = 30   # 2 feet x 5 sites x 3 axes
N_FOOT_SPEED = 30
N_HAND_SPEED = 18  # 2 hands x 3 sites x 3 axes (regen box-frame speed rows)
N_FIXED = 29 + 29 + 3 + 3 + 35
COLLISION_OFF = N_FIXED                       # 99
HAND_OFF = COLLISION_OFF + MAX_PAIRS          # 147
FOOT_OFF = HAND_OFF + N_HAND_ROWS             # 165
FOOT_SPEED_OFF = FOOT_OFF + N_FOOT_ROWS       # 195
HAND_SPEED_OFF = FOOT_SPEED_OFF + N_FOOT_SPEED  # 225
N_ROWS = HAND_SPEED_OFF + N_HAND_SPEED        # 243

KPT_NAMES = list(KPT_TO_SMPLH)


def step_scale_matrix(cfg) -> np.ndarray:
    return np.concatenate([
        np.full(3, cfg.step_scale_root_t_m),
        np.full(3, cfg.step_scale_root_r_rad),
        np.full(29, cfg.step_scale_joint_rad),
    ])


def tangent_error(z: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
    """z ⊖ z_ref for robot qpos [36] -> 35 (world-frame rotation residual)."""
    out = np.empty(N_TANGENT)
    out[0:3] = z[0:3] - z_ref[0:3]
    R = quat_to_mat_wxyz(z[3:7][None])[0]
    Rr = quat_to_mat_wxyz(z_ref[3:7][None])[0]
    out[3:6] = so3_log_np((R @ Rr.T)[None])[0]
    out[6:35] = z[7:36] - z_ref[7:36]
    return out


class FrameEvaluator:
    """Evaluates residuals/constraints at a robot state for one frame.

    Owns the MuJoCo model/data; geometry uses the *fixed* per-frame box pose.
    The robot state is [36] qpos; the scene qpos is assembled internally.
    """

    def __init__(self, spec: G1Spec, pairs: CollisionPairs, cfg) -> None:
        self.spec = spec
        self.cfg = cfg
        self.model = spec.load_scene()
        self.data = mujoco.MjData(self.model)
        self.pairs = pairs.all_pairs
        self.ground_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.box_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "largebox")
        ids = lambda names: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names]  # noqa: E731
        self.hand_sites = {s: ids(list(v.values())) for s, v in HAND_SITES.items()}
        self.sole_sites = {s: ids(list(v)) for s, v in SOLE_SITES.items()}
        self.kpt_ids = ids(KPT_NAMES)
        self._hand_geoms = {
            s: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                 f"{'left' if s == 'l' else 'right'}_{part}_link")
                for part in ("rubber_hand", "thumb", "pinky")]
            for s in ("l", "r")
        }
        self.sole_geoms = {
            s: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                 f"{'left' if s == 'l' else 'right'}_ankle_roll_sphere_{i}_link")
                for i in range(1, 6)]
            for s in ("l", "r")
        }
        self._fromto = np.zeros(6)
        self._jac_buf = np.zeros((3, self.model.nv))
        # transition-probe state (F4, 2026-09-22): an interpolated mid state
        # must be evaluated WITHOUT disturbing the endpoint state the solver
        # linearized at -- own data, own fromto buffer
        self._probe_data = mujoco.MjData(self.model)
        self._probe_fromto = np.zeros(6)
        self._probe_jac_buf = np.zeros((3, self.model.nv))
        self._motion_data = mujoco.MjData(self.model)

    # -- forward -----------------------------------------------------------

    def forward(self, z: np.ndarray, box_pose: np.ndarray) -> None:
        qpos = np.zeros(43)
        qpos[0:36] = z
        qpos[36:39] = box_pose[0:3]
        qpos[39:43] = box_pose[3:7]
        self.data.qpos = qpos
        mujoco.mj_forward(self.model, self.data)
        self._root_R = quat_to_mat_wxyz(z[3:7][None])[0]

    def probe_forward(self, q_robot: np.ndarray, box_pose: np.ndarray) -> None:
        """Forward the PROBE data to an interpolated mid state (F4)."""
        qpos = np.zeros(43)
        qpos[0:36] = q_robot
        qpos[36:39] = box_pose[0:3]
        qpos[39:43] = box_pose[3:7]
        self._probe_data.qpos = qpos
        mujoco.mj_forward(self.model, self._probe_data)
        self._probe_R = quat_to_mat_wxyz(np.asarray(q_robot[3:7], dtype=float)[None])[0]

    def motion_between(self, z_a: np.ndarray, box_a: np.ndarray,
                       z_b: np.ndarray, box_b: np.ndarray) -> float:
        """Max collision-relevant displacement between two scenes (m).

        Same protocol as the independent validator's frame motion: the max
        robot site displacement plus the max box CORNER displacement (a
        center-only metric cannot see a rotating box sweeping through the
        robot). The solver uses it to reproduce the validator's adaptive
        subdivision, so probe rows and checks probe the same states.
        """
        m = self._motion_data
        out = []
        for z, box in ((z_a, box_a), (z_b, box_b)):
            qpos = np.zeros(43)
            qpos[0:36] = z
            qpos[36:39] = box[0:3]
            qpos[39:43] = box[3:7]
            m.qpos = qpos
            mujoco.mj_forward(self.model, m)
            sites = m.site_xpos.copy()
            R = m.geom_xmat[self.box_geom].reshape(3, 3)
            c = m.geom_xpos[self.box_geom].copy()
            out.append((sites, c + self.box_corners() @ R.T))
        (sa, ca), (sb, cb) = out
        robot = float(np.linalg.norm(sb - sa, axis=-1).max()) if len(sa) else 0.0
        corners = float(np.linalg.norm(cb - ca, axis=-1).max())
        return max(robot, corners)

    def box_corners(self) -> np.ndarray:
        """[8, 3] box-local AABB corners (mirrors the validator's metric)."""
        half = self.model.geom_size[self.box_geom, 0:3]
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                          for sz in (-1, 1)], dtype=float)
        return signs * half

    def site_positions(self, ids: list[int]) -> np.ndarray:
        return self.data.site_xpos[ids]

    def site_jacobians_35(self, ids: list[int]) -> np.ndarray:
        """[S, 3, 35] site position Jacobians in the tangent convention."""
        R = self._root_R
        out = np.empty((len(ids), 3, N_TANGENT))
        for s, sid in enumerate(ids):
            self._jac_buf[:] = 0.0
            mujoco.mj_jacSite(self.model, self.data, self._jac_buf, None, sid)
            out[s, :, 0:3] = self._jac_buf[:, 0:3]
            out[s, :, 3:6] = self._jac_buf[:, 3:6] @ R.T
            out[s, :, 6:35] = self._jac_buf[:, 6:35]  # scene nv=42: drop box dofs
        return out

    def geom_jacobian_35(self, gid: int) -> np.ndarray:
        """[3, 35] geom-center linear Jacobian in the tangent convention."""
        R = self._root_R
        self._jac_buf[:] = 0.0
        mujoco.mj_jacGeom(self.model, self.data, self._jac_buf, None, gid)
        J = np.empty((3, N_TANGENT))
        J[:, 0:3] = self._jac_buf[:, 0:3]
        J[:, 3:6] = self._jac_buf[:, 3:6] @ R.T
        J[:, 6:35] = self._jac_buf[:, 6:35]  # scene nv=42: drop box dofs
        return J

    def geom_point_jacobian_35(self, gid: int, point_world: np.ndarray,
                               probe: bool = False) -> np.ndarray:
        """[3, 35] rigid-point Jacobian for a point rigidly attached to the geom.

        ``v_point = v_center + omega x r`` with ``r = point - geom_xpos``:
        for mesh geoms the closest point sits centimeters from the frame
        origin, and ignoring the rotational term mispredicts the distance
        gradient badly enough to stall the SCP. ``probe=True`` evaluates the
        same quantity at the probe (interpolated) state instead.
        """
        if probe:
            data, R, buf = self._probe_data, self._probe_R, self._probe_jac_buf
        else:
            data, R, buf = self.data, self._root_R, self._jac_buf
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model, data, jacp, jacr, gid)
        to35 = lambda J: np.concatenate([  # noqa: E731
            J[:, 0:3], J[:, 3:6] @ R.T, J[:, 6:35]
        ], axis=1)
        Jp = to35(jacp)
        Jr = to35(jacr)
        r = np.asarray(point_world, dtype=float) - data.geom_xpos[gid]
        skew = np.array([
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ])
        return Jp - skew @ Jr

    def box_frame(self) -> tuple[np.ndarray, np.ndarray]:
        R_g = self.data.geom_xmat[self.box_geom].reshape(3, 3)
        x_g = self.data.geom_xpos[self.box_geom]
        return R_g, x_g

    def hand_local(self, side: str) -> np.ndarray:
        R_g, x_g = self.box_frame()
        return (R_g.T @ (self.site_positions(self.hand_sites[side]) - x_g).T).T

    def hand_local_jac(self, side: str) -> np.ndarray:
        """[S, 3, 35] hand-site Jacobians in the BOX GEOM frame."""
        R_g, _ = self.box_frame()
        # 'j' contracts the site-point axis; "sak->sik" summed the coordinate
        return np.einsum("ij,sjk->sik", R_g.T, self.site_jacobians_35(self.hand_sites[side]))

    def geom_distance(self, ga: int, gb: int, distmax: float = 1.0) -> tuple[float, np.ndarray]:
        d = mujoco.mj_geomDistance(self.model, self.data, ga, gb, distmax, self._fromto)
        return float(d), self._fromto.copy()

    def probe_geom_distance(self, ga: int, gb: int, distmax: float = 1.0) -> tuple[float, np.ndarray]:
        """Distance at the PROBE state (F4)."""
        d = mujoco.mj_geomDistance(self.model, self._probe_data, ga, gb,
                                   distmax, self._probe_fromto)
        return float(d), self._probe_fromto.copy()

    def probe_box_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Box geom frame ``(R, x)`` of the PROBE state (F4).

        The box frame depends only on the box pose, so ``probe_forward`` with
        any robot state yields it without disturbing the endpoint state the
        solver linearized at.
        """
        R = self._probe_data.geom_xmat[self.box_geom].reshape(3, 3)
        x = self._probe_data.geom_xpos[self.box_geom].copy()
        return R, x

    def geom_name(self, gid: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)

    def geom_separation_direction(self, ga: int, gb: int, dist: float,
                                  probe: bool = False) -> np.ndarray:
        """Unit direction in which ga's closest point must move to SEPARATE.

        Needed when MuJoCo's ``fromto`` is degenerate: for a pair in exact
        contact (dist ~ 0) the two closest points coincide, so both the
        segment normal and the distance gradient vanish. The old guard
        (``n > 1e-9`` else zeros) zeroed the Jacobian exactly at the contact
        configurations the solver must regulate -- the QP row went inert and
        the solver could sink a supporting foot centimeters into the ground
        for free while the check (true distance) only saw the penetration
        afterwards.

        Fallback order: MuJoCo's contact record for the pair (present
        whenever the pair actually collides), then geometry -- a plane's
        frame z-axis, otherwise the center-to-center direction (exact for
        spheres, the standard convex approximation at contact).
        """
        data = self._probe_data if probe else self.data
        for i in range(data.ncon):
            c = data.contact[i]
            if {int(c.geom1), int(c.geom2)} == {ga, gb}:
                # frame normal points from geom2 to geom1
                n = np.array(c.frame[0:3], dtype=float)
                return n if ga == int(c.geom1) else -n
        for g, other in ((ga, gb), (gb, ga)):
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                R = data.geom_xmat[g].reshape(3, 3)
                n = R[:, 2].copy()
                # orient from the plane toward the other geom
                side = np.dot(n, data.geom_xpos[other] - data.geom_xpos[g])
                if side < 0:
                    n = -n
                # u points from the other side toward ga (its separation
                # direction): along n when the plane is gb, against it when
                # the plane is ga
                return -n if g == ga else n
        d = data.geom_xpos[ga] - data.geom_xpos[gb]
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            return d / n
        return np.array([0.0, 0.0, 1.0])

    def distance_jacobian(self, ga: int, gb: int, fromto: np.ndarray, dist: float,
                          probe: bool = False) -> np.ndarray:
        """Jacobian of the signed distance wrt xi (rigid closest-point form).

        d = s * n . (p2 - p1), s = +1 separated / -1 penetrating: the
        fromto segment always points from p1 to p2, but the signed distance
        is -|p2-p1| inside, which flips the gradient sign. Equivalently the
        gradient is ``u . (Jp1 - Jp2)`` with ``u`` the unit separation
        direction of ga (``geom_separation_direction``), which stays valid
        when the fromto segment collapses at exact contact.
        """
        p1, p2 = fromto[0:3], fromto[3:6]
        diff = p2 - p1
        n = float(np.linalg.norm(diff))
        s = 1.0 if dist >= 0 else -1.0
        if n > 1e-9:
            u = -s * (diff / n)
        else:
            u = self.geom_separation_direction(ga, gb, dist, probe=probe)
        J = np.zeros((1, N_TANGENT))
        for g, point, sign in ((ga, p1, +1.0), (gb, p2, -1.0)):
            if g in (self.box_geom, self.ground_geom):
                continue  # static within a frame
            J += sign * (u[None, :] @ self.geom_point_jacobian_35(g, point, probe=probe))
        return J
