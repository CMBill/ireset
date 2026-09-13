"""38-dim joint state tokenizer (plan section 7.1/7.2).

Flat diffusion state per frame::

    x0 = [ root_pos_clip_normalized (3) | root_rot6d_clip (6) | u_joints (29) ]

- root position is expressed in the **clip frame** (plan 5.1: horizontal
  origin and yaw removed from the reference human; height kept) and
  standardized by train-split statistics;
- root rotation uses the 6D (two-column) representation in the clip frame —
  a fixed representation, unclipped (scheduler global clipping must stay
  off; see module :mod:`ireset.diffusion.generator`);
- joints are limit-mapped ``u = 2 (q - lo) / (hi - lo) - 1`` on the clean
  side only. Noisy states and root/rot channels are never passed through
  the sigmoid or clipping (plan 7.2).

The model head produces ``2*sigmoid(logits) - 1`` for the clean joint
prediction; the tokenizer's decode applies the same map. Degenerate or
non-finite rotation features (zero 6D vectors) are surfaced as flags so the
generator can reject/resample instead of emitting a broken pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ireset.kinematics.rotation_adapter import (
    mat_to_quat_wxyz,
    quat_to_mat_wxyz,
    rot6d_to_mat_torch,
)

__all__ = [
    "X0_DIM", "RootStats", "StateTokenizer",
    "quat_to_mat_torch", "mat_to_quat_torch",
]

X0_DIM = 38
ROOT_POS_DIM, ROOT_ROT_DIM, JOINT_DIM = 3, 6, 29


@dataclass
class RootStats:
    """Train-split statistics of clip-frame root positions."""

    mean: np.ndarray   # [3]
    std: np.ndarray    # [3]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "RootStats":
        d = np.load(path)
        return cls(mean=d["mean"], std=d["std"])

    @classmethod
    def fit(cls, pos_clip: np.ndarray) -> "RootStats":
        """Per-dim mean/std over frames; floor std to keep the map finite."""
        std = pos_clip.std(0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=pos_clip.mean(0), std=std)

    def to_torch(self, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std, dtype=torch.float32, device=device)
        return mean, std


class StateTokenizer:
    """Physical state <-> normalized x0, in one clip frame."""

    def __init__(self, joint_range: np.ndarray, root_stats: RootStats) -> None:
        self.lo = np.asarray(joint_range[:, 0], dtype=np.float64)
        self.hi = np.asarray(joint_range[:, 1], dtype=np.float64)
        self.root_stats = root_stats

    # --- joint limit maps -----------------------------------------------------

    def joints_to_u(self, q: np.ndarray) -> np.ndarray:
        return 2.0 * (q - self.lo) / (self.hi - self.lo) - 1.0

    def u_to_joints(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, -1.0, 1.0)
        return self.lo + (u + 1.0) * (self.hi - self.lo) / 2.0

    # --- numpy encode/decode --------------------------------------------------

    def encode(self, root: np.ndarray, q: np.ndarray, world_from_clip: np.ndarray) -> np.ndarray:
        """root [T, 7], q [T, 29], world_from_clip [4, 4] -> x0 [T, 38]."""
        Rz, t = world_from_clip[:3, :3], world_from_clip[:3, 3]
        pos_clip = (root[:, 0:3] - t) @ Rz                       # Rz^T (p - t)
        pos_n = (pos_clip - self.root_stats.mean) / self.root_stats.std
        R_world = quat_to_mat_wxyz(root[:, 3:7])                 # [T,3,3]
        R_clip = np.einsum("ji,tjk->tik", Rz, R_world)           # Rz^T R
        r6 = np.concatenate([R_clip[:, :, 0], R_clip[:, :, 1]], axis=1)
        return np.concatenate([pos_n, r6, self.joints_to_u(q)], axis=1)

    def decode(self, x0: np.ndarray, world_from_clip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """x0 [T, 38] -> (root [T, 7] world, q [T, 29], ok [T] bool)."""
        Rz, t = world_from_clip[:3, :3], world_from_clip[:3, 3]
        pos_clip = x0[:, 0:3] * self.root_stats.std + self.root_stats.mean
        pos_world = pos_clip @ Rz.T + t
        r6 = x0[:, 3:9]
        ok = np.isfinite(x0).all(1) & (np.linalg.norm(r6[:, 0:3], axis=1) > 1e-6) \
            & (np.linalg.norm(r6[:, 3:6], axis=1) > 1e-6)
        r6s = np.where(ok[:, None], r6, np.tile([1.0, 0, 0, 0, 1, 0], (len(r6), 1)))
        R_clip = rot6d_to_mat_torch(torch.as_tensor(r6s, dtype=torch.float32)).numpy()
        R_world = np.einsum("ij,tjk->tik", Rz, R_clip)
        quat = mat_to_quat_wxyz(R_world)
        root = np.concatenate([pos_world, quat], axis=1)
        q = self.u_to_joints(x0[:, 9:38])
        return root, q, ok

    # --- torch encode (training targets) --------------------------------------

    def encode_torch(
        self,
        pos: torch.Tensor,               # [B, T, 3] world
        quat_wxyz: torch.Tensor,         # [B, T, 4]
        q: torch.Tensor,                 # [B, T, 29]
        wfc: torch.Tensor,               # [B, 4, 4] world_from_clip
        mean: torch.Tensor, std: torch.Tensor,
        lo: torch.Tensor, hi: torch.Tensor,
    ) -> torch.Tensor:
        Rz, t = wfc[:, :3, :3], wfc[:, :3, 3]
        pos_clip = torch.einsum("bij,btj->bti", Rz.transpose(1, 2), pos - t[:, None, :])
        pos_n = (pos_clip - mean) / std
        R_w = quat_to_mat_torch(quat_wxyz)
        R_c = torch.einsum("bij,btjk->btik", Rz.transpose(1, 2), R_w)
        r6 = torch.cat([R_c[..., :, 0], R_c[..., :, 1]], dim=-1)
        u = 2.0 * (q - lo) / (hi - lo) - 1.0
        return torch.cat([pos_n, r6, u], dim=-1)

    # --- torch decode (losses / sampling) -------------------------------------

    def decode_torch(
        self,
        x0: torch.Tensor,                # [B, T, 38]
        wfc: torch.Tensor,               # [B, 4, 4]
        mean: torch.Tensor, std: torch.Tensor,
        lo: torch.Tensor, hi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> pos [B,T,3] world, quat_wxyz [B,T,4], q [B,T,29]; differentiable."""
        Rz, t = wfc[:, :3, :3], wfc[:, :3, 3]
        pos_clip = x0[..., 0:3] * std + mean
        pos = torch.einsum("bij,btj->bti", Rz, pos_clip) + t[:, None, :]
        R_c = rot6d_to_mat_torch(x0[..., 3:9])
        R_w = torch.einsum("bij,btjk->btik", Rz, R_c)
        quat = _mat_to_quat_torch(R_w)
        q = lo + (x0[..., 9:38].clamp(-1.0, 1.0) + 1.0) * (hi - lo) / 2.0
        return pos, quat, q


def quat_to_mat_torch(q: torch.Tensor) -> torch.Tensor:
    """wxyz [.., 4] -> [.., 3, 3], differentiable (normalized input assumed)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _mat_to_quat_torch(R: torch.Tensor) -> torch.Tensor:
    """[.., 3, 3] -> wxyz [.., 4], differentiable (branch on trace is fine).

    Uses the trace-based branch only; near the pi rotation the normalization
    keeps values finite and the rotation is still recovered up to sign
    (the 6D decode already orthogonalized R, so the input is SO(3)).
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = torch.sqrt((1.0 + tr).clamp_min(1e-12)) / 2.0
    denom = (4.0 * w).clamp_min(1e-12)
    x = (R[..., 2, 1] - R[..., 1, 2]) / denom
    y = (R[..., 0, 2] - R[..., 2, 0]) / denom
    z = (R[..., 1, 0] - R[..., 0, 1]) / denom
    q = torch.stack([w, x, y, z], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


mat_to_quat_torch = _mat_to_quat_torch
