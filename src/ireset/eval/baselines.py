"""Pure-optimization baseline: per-frame Gauss-Newton interaction tracker.

Plan section 10.2, baseline flavor (2): same geometry as the project, no
learned model, no teacher labels beyond the reference condition. For each
frame the 35-dim state is optimized to align 15 anatomical keypoints (G1 FK
sites vs reference human keypoints, SMPLH correspondence) with a damped
Gauss-Newton step in the tangent space of section 6.1:

    xi = -(J^T J + lambda D^T D)^-1 J^T r ,  step clipped to a trust box,
    retracted via the manifold update, joints clamped to limits (documented:
    M2 replaces clamping with hard QP constraints).

Frame 0 starts from the reference-estimated root pose (pelvis keypoint,
shoulder-line yaw) and the nominal posture; later frames warm-start from the
previous solution. The box trajectory is a fixed task input. This is the
timing/quality reference the learned generators (M3) must beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ireset.embodied.g1_spec import KPT_TO_SMPLH, G1Spec
from ireset.embodied.contact_sites import SMPLH_INDEX
from ireset.kinematics.mujoco_adapter import G1Kinematics, N_TANGENT, RobotState, retract_batch

KPT_NAMES = list(KPT_TO_SMPLH)


@dataclass
class TrackerConfig:
    max_iters: int = 10
    damping: float = 1e-3               # LM lambda on the scaled normal equations
    trust_box: tuple[float, float, float] = (0.05, 0.1, 0.2)  # (dp m, dtheta rad, dq rad)
    keypoint_weight_pelvis: float = 4.0  # extra weight on the pelvis keypoint
    fps: float = 30.0


class InteractionTracker:
    """Per-frame GN tracker producing robot state from reference keypoints."""

    def __init__(self, kin: G1Kinematics | None = None, cfg: TrackerConfig | None = None) -> None:
        self.kin = kin or G1Kinematics()
        self.cfg = cfg or TrackerConfig()
        self.spec = self.kin.spec
        self._smplh_ids = [SMPLH_INDEX[KPT_TO_SMPLH[k]] for k in KPT_NAMES]

    def initial_state(self, ref_kpts_world: np.ndarray) -> RobotState:
        """Reference-estimated root pose (pelvis position, shoulder yaw) + nominal q."""
        from ireset.kinematics.rotation_adapter import mat_to_quat_wxyz, so3_exp_np

        kpts = ref_kpts_world[self._smplh_ids]  # [15, 3] in G1 correspondence order
        pos = kpts[0].copy()                    # pelvis site ~ root height offset applied below
        sh_l = kpts[KPT_NAMES.index("kpt_l_shoulder")]
        sh_r = kpts[KPT_NAMES.index("kpt_r_shoulder")]
        yaw = np.arctan2(-(sh_l[1] - sh_r[1]), sh_l[0] - sh_r[0])  # +x along shoulder line
        R = so3_exp_np(np.array([0.0, 0.0, yaw]))
        # pelvis site sits slightly below the root (MJCF pelvis origin); ignore offset
        q = np.zeros((1, 29))
        return RobotState(pos[None].copy(), mat_to_quat_wxyz(R)[None], q)

    def solve_frame(
        self, state: RobotState, ref_kpts_world: np.ndarray, max_iters: int | None = None
    ) -> RobotState:
        """Refine a single-frame state against reference keypoints [K,3] world."""
        cfg = self.cfg
        target = ref_kpts_world[self._smplh_ids]
        w = np.ones(len(KPT_NAMES))
        w[KPT_NAMES.index("kpt_pelvis")] = cfg.keypoint_weight_pelvis
        W = np.repeat(w, 3)

        trust = np.array(
            [cfg.trust_box[0]] * 3 + [cfg.trust_box[1]] * 3 + [cfg.trust_box[2]] * 29
        )
        lo, hi = self.spec.joint_range[:, 0], self.spec.joint_range[:, 1]
        qpos = state.to_qpos()[0]
        for _ in range(max_iters or cfg.max_iters):
            st = RobotState(qpos[0:3][None], qpos[3:7][None], qpos[7:36][None])
            cur = self.kin.site_positions(st, KPT_NAMES)[0]
            r = (cur - target).reshape(-1)
            J = self.kin.jacobian(st, KPT_NAMES)[0].reshape(-1, N_TANGENT)
            Jw = J * W[:, None]
            A = Jw.T @ J + cfg.damping * np.eye(N_TANGENT)
            b = -Jw.T @ r
            xi = np.linalg.solve(A, b)
            xi = np.clip(xi, -trust, trust)
            qpos = retract_batch(self.kin.model, qpos[None], xi[None])[0]
            qpos[7:36] = np.clip(qpos[7:36], lo, hi)
        return RobotState(qpos[0:3][None].copy(), qpos[3:7][None].copy(), qpos[7:36][None].copy())

    def track(
        self, ref_kpts_world: np.ndarray, box_pose: np.ndarray, init: RobotState | None = None
    ) -> np.ndarray:
        """Track a full reference clip; returns scene qpos [T, 43] with the fixed box."""
        T = ref_kpts_world.shape[0]
        state = init or self.initial_state(ref_kpts_world[0])
        out = np.zeros((T, 43))
        for t in range(T):
            state = self.solve_frame(state, ref_kpts_world[t])
            out[t, 0:36] = state.to_qpos()[0]
            out[t, 36:43] = box_pose[t]
        return out
