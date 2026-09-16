"""Rotation conventions and SO(3) helpers.

Project conventions (docs/plans/interaction_retarget_impl_plan.md section 4.2):

- Quaternions stored as ``[qw, qx, qy, qz]`` (MuJoCo order). scipy uses
  ``[x, y, z, w]``; conversion helpers below are the only sanctioned place
  for the reordering.
- Root rotation increments live in the world-frame tangent:
  ``R_next = Exp(delta_theta_world) @ R``.
- 6D rotation representation (Zhou et al.): first two matrix columns.

NumPy paths wrap scipy; torch paths are minimal differentiable supplements
(exp map via Taylor/SVD-free closed form, 6D encoding) used by training
losses. Cross-checked in tests/test_fk.py.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rot


# --- NumPy / scipy wrappers -------------------------------------------------

def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.roll(np.asarray(q, dtype=float), -1, axis=-1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.roll(np.asarray(q, dtype=float), 1, axis=-1)


def quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    """[..., 4] wxyz -> [..., 3, 3]."""
    return _Rot.from_quat(wxyz_to_xyzw(q)).as_matrix()


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    return xyzw_to_wxyz(_Rot.from_matrix(R).as_quat())


def so3_exp_np(w: np.ndarray) -> np.ndarray:
    """Rotation matrices from axis-angle vectors [..., 3]."""
    return _Rot.from_rotvec(np.asarray(w, dtype=float)).as_matrix()


def so3_log_np(R: np.ndarray) -> np.ndarray:
    """Axis-angle vectors from rotation matrices [..., 3, 3] (principal log)."""
    return _Rot.from_matrix(R).as_rotvec()


# --- Torch differentiable helpers --------------------------------------------

def rot6d_to_mat_torch(r6: torch.Tensor) -> torch.Tensor:
    """[..., 6] (two column vectors) -> [..., 3, 3], differentiable."""
    a = r6[..., 0:3]
    b = r6[..., 3:6]
    b = b - (a * b).sum(-1, keepdim=True) * a / (a * a).sum(-1, keepdim=True).clamp_min(1e-12)
    c = torch.linalg.cross(a, b, dim=-1)
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.stack([a, b, c], dim=-1)


def mat_to_rot6d_torch(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def so3_exp_torch(w: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map [..., 3] -> [..., 3, 3]."""
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / theta
    K = skew_torch(k)
    sin, cos = torch.sin(theta), torch.cos(theta)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(*k.shape[:-2], 3, 3)
    return eye + torch.sin(theta.unsqueeze(-1)) * K + (1 - torch.cos(theta.unsqueeze(-1))) * (K @ K)


def so3_log_torch(R: torch.Tensor) -> torch.Tensor:
    """Differentiable log map [..., 3, 3] -> [..., 3] (axis-angle).

    Smooth away from theta = pi (the principal-branch singularity, shared
    with every closed-form log map); near theta = 0 a first-order Taylor
    branch keeps gradients finite. Losses using this should not rely on
    values at rotations near pi from their targets.
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos)
    w = 0.5 * torch.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
        dim=-1,
    )
    sin = torch.sin(theta)
    # theta / sin(theta) -> 1 as theta -> 0; both branches stay finite
    factor = torch.where(theta < 1e-4, torch.ones_like(theta), theta / sin.clamp_min(1e-12))
    return w * factor[..., None]


def skew_torch(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def quat_apply_torch(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` ([..., 3]) by quaternions ``q_wxyz`` ([..., 4]); differentiable.

    ``v' = v + 2 w (xyz x v) + 2 xyz x (xyz x v)``
    """
    w, xyz = q_wxyz[..., 0:1], q_wxyz[..., 1:4]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_conjugate_torch(q_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat([q_wxyz[..., 0:1], -q_wxyz[..., 1:4]], dim=-1)
