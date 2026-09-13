"""Training loss assembly for both generators (plan section 7.3).

Terms (units explicit; weights from ``configs/train``):

- masked clean-state reconstruction in the 38-dim normalized space, split
  into position / rotation-6D / joint groups with independent weights;
- root rotation geodesic error (rad^2) on decoded SO(3);
- teacher-FK keypoint error (m^2) on the 15 anatomical sites, clip frame;
- hand contact deviation (m^2): signed AABB SDF of the six hand contact
  sites during reference hand-contact frames (AABB = training proxy, plan
  6.2; acceptance always re-checks with MuJoCo hull queries);
- temporal velocity error on consecutive-frame differences of x0.

Geometry losses never replace repair acceptance. The optional reference
Laplacian energy E_L is left as a planned ablation and is **not**
implemented in the pilot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ireset.embodied.g1_spec import HAND_SITES, KPT_TO_SMPLH
from ireset.geometry.distance_adapter import largebox_static_frame, point_box_sdf_torch
from ireset.kinematics.rotation_adapter import rot6d_to_mat_torch, so3_log_torch
from ireset.kinematics.torch_adapter import G1TorchFK
from ireset.model.tokens import mat_to_quat_torch, quat_to_mat_torch

HAND_SITE_NAMES = [HAND_SITES[s][k] for s in ("l", "r") for k in ("palm", "thumb", "pinky")]


@dataclass
class LossWeights:
    w_pos: float = 1.0
    w_rot6d: float = 1.0
    w_joint: float = 1.0
    w_geodesic: float = 0.5
    w_fk: float = 2.0      # m^2 terms are small (~1e-2); keep them visible
    w_contact: float = 0.5
    w_vel: float = 0.5

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class LossAssembler:
    """Pre-loads tokenizer tensors/FK on device; call per training step."""

    def __init__(self, tokenizer, weights: LossWeights, device, use_fk: bool = True) -> None:
        self.w = weights
        self.device = device
        self.mean, self.std = tokenizer.root_stats.to_torch(device)
        self.lo = torch.tensor(tokenizer.lo, dtype=torch.float32, device=device)
        self.hi = torch.tensor(tokenizer.hi, dtype=torch.float32, device=device)
        self.fk = G1TorchFK(device=device, dtype=torch.float32) if use_fk else None
        R_g, x_g = largebox_static_frame()
        self.R_g = torch.tensor(R_g, dtype=torch.float32, device=device)
        self.x_g = torch.tensor(x_g, dtype=torch.float32, device=device)

    def decode_clip(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x0 [B,F,38] -> clip-frame (pos [B,F,3], quat [B,F,4], q [B,F,29])."""
        pos = x0[..., 0:3] * self.std + self.mean
        R = rot6d_to_mat_torch(x0[..., 3:9])
        quat = mat_to_quat_torch(R)
        q = self.lo + (x0[..., 9:38].clamp(-1.0, 1.0) + 1.0) * (self.hi - self.lo) / 2.0
        return pos, quat, q

    def __call__(self, x0_pred: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
        x0 = batch["x0"]
        parts: dict[str, torch.Tensor] = {}

        diff2 = (x0_pred - x0) ** 2
        parts["pos"] = diff2[..., 0:3].mean()
        parts["rot6d"] = diff2[..., 3:9].mean()
        parts["joint"] = diff2[..., 9:38].mean()
        loss = self.w.w_pos * parts["pos"] + self.w.w_rot6d * parts["rot6d"] \
            + self.w.w_joint * parts["joint"]

        R_p = rot6d_to_mat_torch(x0_pred[..., 3:9])
        R_t = rot6d_to_mat_torch(x0[..., 3:9])
        geo = so3_log_torch(R_p @ R_t.transpose(-1, -2)).norm(dim=-1)
        parts["geodesic_rad2"] = (geo ** 2).mean()
        loss = loss + self.w.w_geodesic * parts["geodesic_rad2"]

        d_pred = x0_pred[:, 1:] - x0_pred[:, :-1]
        d_tgt = x0[:, 1:] - x0[:, :-1]
        parts["vel"] = ((d_pred - d_tgt) ** 2).mean()
        loss = loss + self.w.w_vel * parts["vel"]

        if self.fk is not None and self.w.w_fk > 0:
            pos_p, quat_p, q_p = self.decode_clip(x0_pred)
            B, F = pos_p.shape[:2]
            flat = lambda a: a.reshape(B * F, -1)
            kp_p = self.fk.site_world_positions(
                flat(pos_p), flat(quat_p), flat(q_p), list(KPT_TO_SMPLH)
            ).view(B, F, -1, 3)
            with torch.no_grad():
                pos_t, quat_t, q_t = self.decode_clip(x0)
                kp_t = self.fk.site_world_positions(
                    flat(pos_t), flat(quat_t), flat(q_t), list(KPT_TO_SMPLH)
                ).view(B, F, -1, 3)
            parts["fk_m2"] = ((kp_p - kp_t) ** 2).sum(-1).mean()
            loss = loss + self.w.w_fk * parts["fk_m2"]

            # hand contact deviation in the box geom frame (AABB proxy)
            hand_p = self.fk.site_world_positions(
                flat(pos_p), flat(quat_p), flat(q_p), HAND_SITE_NAMES
            ).view(B, F, -1, 3)                                                     # [B,F,6,3]
            Rz_T = batch["wfc"][:, :3, :3].transpose(1, 2)                              # [B,3,3]
            R_box_clip = torch.einsum("bij,btjk->btik", Rz_T, quat_to_mat_torch(batch["box_quat"]))
            p_rel = hand_p - batch["box_t_clip"][:, :, None, :]                          # [B,F,6,3]
            p_body = torch.einsum("btij,btsj->btsi", R_box_clip, p_rel)
            p_geom = (p_body - self.x_g) @ self.R_g                                      # R_g^T (p - x_g)
            sdf = point_box_sdf_torch(p_geom)                                            # [B,F,6]
            contact = batch["contact"][..., 0:2]                                         # [B,F,2]
            left = sdf[..., 0:3] * contact[..., 0:1]
            right = sdf[..., 3:6] * contact[..., 1:2]
            parts["contact_m2"] = (left ** 2).sum() / contact[..., 0].sum().clamp_min(1.0) / 3.0 \
                + (right ** 2).sum() / contact[..., 1].sum().clamp_min(1.0) / 3.0
            loss = loss + self.w.w_contact * parts["contact_m2"]

        parts["total"] = loss
        return loss, {k: float(v.detach()) for k, v in parts.items()}
