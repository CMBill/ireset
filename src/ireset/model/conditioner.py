"""Condition token encoder (plan sections 5.1/7.1).

Builds the cross-attention memory from one window batch:

- 52 per-frame human keypoint tokens: features are the keypoint in the
  **source object frame** and in the **clip frame** (both derived from the
  reference human; the target G1 root never enters the condition);
- 1 per-frame target-object token: clip-frame translation + 6D rotation;
- 1 per-frame contact token: the four effector schedule flags;
- 2 static tokens: locked box AABB half extents, ground plane.

Frame index sinusoidal encoding is added to per-frame tokens. Token types
are distinguished by learned embeddings so attention can separate them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ireset.geometry.distance_adapter import LARGEBOX_HALF_EXTENTS
from ireset.model.tokens import quat_to_mat_torch as _quat_to_mat

N_KEYPOINTS = 52
N_PER_FRAME = N_KEYPOINTS + 2   # + object pose + contact
N_STATIC = 2


def sinusoidal_frame_encoding(F: int, d_model: int, device=None) -> torch.Tensor:
    """[F, d] positional encoding over frame index."""
    pos = torch.arange(F, dtype=torch.float32, device=device)[:, None]
    i = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)[None, :]
    ang = pos / torch.pow(10000.0, i / d_model)
    enc = torch.zeros(F, d_model, device=device)
    enc[:, 0::2] = torch.sin(ang)
    enc[:, 1::2] = torch.cos(ang[:, : (d_model + 1) // 2])
    return enc


class ConditionEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.kpt_proj = nn.Linear(6, d_model)
        self.obj_proj = nn.Linear(9, d_model)
        self.contact_proj = nn.Linear(4, d_model)
        self.box_proj = nn.Linear(3, d_model)
        self.ground_proj = nn.Linear(4, d_model)
        self.kpt_emb = nn.Embedding(N_KEYPOINTS, d_model)
        self.type_emb = nn.Embedding(4, d_model)   # 0 kpt, 1 object, 2 contact, 3 static

    def forward(self, batch: dict) -> torch.Tensor:
        """Batch window dict -> condition memory [B, 2 + F * N_PER_FRAME, d]."""
        B, F = batch["kps_obj"].shape[:2]
        dev = batch["kps_obj"].device

        kpt = torch.cat([batch["kps_obj"], batch["kps_clip"]], dim=-1)          # [B,F,52,6]
        kpt_tok = self.kpt_proj(kpt) + self.kpt_emb.weight[None, None] + self.type_emb.weight[0]

        R = _quat_to_mat(batch["box_quat"])                                     # [B,F,3,3]
        r6 = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)                     # [B,F,6]
        obj_tok = self.obj_proj(torch.cat([batch["box_t_clip"], r6], dim=-1)) \
            + self.type_emb.weight[1]                                           # [B,F,d]

        con_tok = self.contact_proj(batch["contact"]) + self.type_emb.weight[2]  # [B,F,d]

        box_static = torch.as_tensor(LARGEBOX_HALF_EXTENTS, dtype=torch.float32, device=dev) \
            .expand(B, 3)
        ground = torch.zeros(B, 4, dtype=torch.float32, device=dev)
        ground[:, 2] = 1.0
        static = torch.stack(
            [self.box_proj(box_static) + self.type_emb.weight[3], self.ground_proj(ground) + self.type_emb.weight[3]],
            dim=1,
        )                                                                       # [B,2,d]

        frame = sinusoidal_frame_encoding(F, self.d_model, dev)                 # [F,d]
        per_frame = torch.cat(
            [kpt_tok + frame[:, None], obj_tok[:, :, None] + frame[:, None], con_tok[:, :, None] + frame[:, None]],
            dim=2,
        ).reshape(B, F * N_PER_FRAME, self.d_model)
        return torch.cat([static, per_frame], dim=1)


def _quat_to_mat(q_wxyz: torch.Tensor) -> torch.Tensor:
    q = q_wxyz / q_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
