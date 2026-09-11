"""Frame transforms between world and object (box) frames.

Pose convention: ``[x, y, z, qw, qx, qy, qz]`` (project standard).
``world_to_object(x, pose) = R^T (x - t)`` with ``R, t`` from ``pose``.
All functions accept [..., 3] points and a single pose or matching batch.
"""

from __future__ import annotations

import numpy as np
import torch

from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz


def pose_to_RT(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pose [..., 7] -> R [..., 3, 3], t [..., 3]."""
    pose = np.atleast_2d(pose)
    R = quat_to_mat_wxyz(pose[:, 3:7])
    return R, pose[:, 0:3]


def world_to_object(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """[T, N, 3] world points -> object frame using per-frame pose [T, 7].

    Applies ``R^T (x - t)``; einsum index layout transposes R.
    """
    R, t = pose_to_RT(pose)
    return np.einsum("tji,tnj->tni", R, points_world - t[:, None, :])


def object_to_world(points_obj: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Object frame -> world: ``R x + t``."""
    R, t = pose_to_RT(pose)
    return np.einsum("tij,tnj->tni", R, points_obj) + t[:, None, :]


def world_to_object_torch(points_world: torch.Tensor, pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """[T, N, 3] world -> object frame given per-frame pos [T,3], quat [T,4].

    Differentiable: applies the conjugate rotation to the translated points.
    """
    from ireset.kinematics.rotation_adapter import quat_apply_torch, quat_conjugate_torch

    v = points_world - pos[:, None, :]
    qbar = quat_conjugate_torch(quat_wxyz)  # [T,4]
    return quat_apply_torch(qbar[:, None, :].expand(-1, v.shape[1], -1), v)
