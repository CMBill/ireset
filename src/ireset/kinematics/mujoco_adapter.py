"""MuJoCo kinematics adapter: batched FK, 35-dim retraction and Jacobians.

Tangent-space convention (plan section 6.1)::

    xi = [delta_p_world(3), delta_theta_world(3), delta_q(29)]  in R^35
    p_next = p + delta_p_world
    R_next = Exp(delta_theta_world) @ R
    q_next = q + delta_q

Empirically verified MuJoCo freejoint qvel layout (tests/test_fk.py also
guards this): linear part is **world** frame, angular part is **local body**
frame. Hence ``J35 = J_mjnu @ blkdiag(I3, R^T, I29)`` where ``J_mjnu`` comes
from ``mj_jac*``. The 7-dim quaternion storage derivatives are never used as
velocities.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ireset.embodied.g1_spec import G1Spec, N_JOINTS
from ireset.kinematics.rotation_adapter import mat_to_quat_wxyz, so3_exp_np

N_TANGENT = N_JOINTS + 6  # 35


@dataclass
class RobotState:
    """Plain robot state, [T, ...] arrays: pos [T,3], quat wxyz [T,4], q [T,29]."""

    pos: np.ndarray
    quat: np.ndarray
    q: np.ndarray

    @property
    def T(self) -> int:
        return self.pos.shape[0]

    def to_qpos(self) -> np.ndarray:
        """[T, 36] MuJoCo qpos."""
        out = np.concatenate([self.pos, self.quat, self.q], axis=-1)
        return out


def retract_batch(m: mujoco.MjModel, qpos: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Apply 35-dim world-frame increments to batched robot qpos [T,36].

    ``xi`` is [T,35] or [35]; done analytically (no integration step).
    """
    qpos = np.atleast_2d(np.asarray(qpos, dtype=float).copy())
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    if xi.shape[0] == 1 and qpos.shape[0] > 1:
        xi = np.repeat(xi, qpos.shape[0], axis=0)
    R = np.stack([_quat_mat(qpos[i, 3:7]) for i in range(qpos.shape[0])])
    qpos[:, 0:3] += xi[:, 0:3]
    R_next = np.einsum("tij,tjk->tik", so3_exp_np(xi[:, 3:6]), R)
    qpos[:, 3:7] = np.stack([mat_to_quat_wxyz(R_next[i]) for i in range(qpos.shape[0])])
    qpos[:, 7:36] += xi[:, 6:35]
    return qpos


def _quat_mat(q_wxyz: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.ascontiguousarray(q_wxyz, dtype=float))
    return R.reshape(3, 3)


class G1Kinematics:
    """MuJoCo-backed FK/Jacobians on the locked plain model.

    Matches the plan interface (section 14.2); ``jacobian`` returns
    ``[T, S, 3, 35]`` position Jacobians in the tangent convention above.
    """

    def __init__(self, spec: G1Spec | None = None) -> None:
        self.spec = spec or G1Spec()
        self.model = self.spec.load_plain()
        self.data = mujoco.MjData(self.model)
        self._site_ids_cache: dict[str, int] = {}

    # -- FK ----------------------------------------------------------------

    def _forward(self, qpos: np.ndarray) -> None:
        self.data.qpos = np.asarray(qpos, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def site_positions(self, state: RobotState, site_names: list[str]) -> np.ndarray:
        """World positions of named sites, [T, S, 3]."""
        ids = [self._site_id(n) for n in site_names]
        out = np.empty((state.T, len(ids), 3))
        qpos = state.to_qpos()
        for t in range(state.T):
            self._forward(qpos[t])
            out[t] = self.data.site_xpos[ids]
        return out

    def keypoints(self, state: RobotState) -> np.ndarray:
        from ireset.embodied.g1_spec import KPT_TO_SMPLH

        return self.site_positions(state, list(KPT_TO_SMPLH))

    def body_positions(self, state: RobotState, body_names: list[str]) -> np.ndarray:
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_names]
        out = np.empty((state.T, len(ids), 3))
        qpos = state.to_qpos()
        for t in range(state.T):
            self._forward(qpos[t])
            out[t] = self.data.xpos[ids]
        return out

    # -- Jacobian ------------------------------------------------------------

    def jacobian(self, state: RobotState, site_names: list[str]) -> np.ndarray:
        """Position Jacobians [T, S, 3, 35] w.r.t. xi = (dp, dtheta_world, dq)."""
        ids = [self._site_id(n) for n in site_names]
        qpos = state.to_qpos()
        out = np.empty((state.T, len(ids), 3, N_TANGENT))
        jacp = np.zeros((3, self.model.nv))
        for t in range(state.T):
            self._forward(qpos[t])
            R = _quat_mat(qpos[t, 3:7])
            for s, sid in enumerate(ids):
                jacp[:] = 0.0
                mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
                # nv-space -> tangent-space: angular block right-multiplied by R^T
                J = np.empty((3, N_TANGENT))
                J[:, 0:3] = jacp[:, 0:3]
                J[:, 3:6] = jacp[:, 3:6] @ R.T
                J[:, 6:35] = jacp[:, 6:]
                out[t, s] = J
        return out

    # -- finite-difference validation ----------------------------------------

    def fd_jacobian(
        self, qpos: np.ndarray, site_names: list[str], eps: float = 1e-7
    ) -> np.ndarray:
        """Central-difference Jacobian via the retraction map, [S, 3, 35]."""
        ids = [self._site_id(n) for n in site_names]

        def f(qp: np.ndarray) -> np.ndarray:
            self._forward(qp)
            return self.data.site_xpos[ids].copy()

        S = len(ids)
        J = np.empty((S, 3, N_TANGENT))
        scales = np.concatenate([np.full(3, eps), np.full(3, eps), np.full(N_JOINTS, eps)])
        for k in range(N_TANGENT):
            xi_p = np.zeros(N_TANGENT); xi_p[k] = scales[k]
            xi_m = np.zeros(N_TANGENT); xi_m[k] = -scales[k]
            fp = f(retract_batch(self.model, qpos, xi_p)[0])
            fm = f(retract_batch(self.model, qpos, xi_m)[0])
            J[:, :, k] = (fp - fm) / (2.0 * scales[k])
        return J

    def _site_id(self, name: str) -> int:
        if name not in self._site_ids_cache:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"unknown site {name!r}")
            self._site_ids_cache[name] = sid
        return self._site_ids_cache[name]
