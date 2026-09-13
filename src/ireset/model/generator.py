"""G1Generator: condition -> full-length candidate RobotTrajectory (plan 14.2).

Pipeline per clip: build condition arrays in the clip frame, cut F=60
windows at stride 30 (shared clip frame), batch them through the model
(diffusion DDIM sampling or deterministic forward pass), decode windows to
physical states, crossfade-fuse with quaternion SLERP, and lift back to the
world frame. Degenerate/non-finite windows are resampled (bounded attempts)
instead of emitted; the fused trajectory is a repair candidate, not a final
result.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from ireset.data.schema import ReferenceCondition, RobotTrajectory
from ireset.data.teacher import keypoints_in_source_object, world_to_clip
from ireset.kinematics.rotation_adapter import wxyz_to_xyzw, xyzw_to_wxyz
from ireset.model.stitch import fuse_windows, window_starts


class G1Generator:
    def __init__(
        self,
        model: torch.nn.Module,          # DiffusionG1 | DeterministicG1
        conditioner: torch.nn.Module,
        tokenizer,
        device: str = "cpu",
        window: int = 60,
        stride: int = 30,
        ddim_steps: int = 25,
        resample_attempts: int = 3,
    ) -> None:
        self.model = model.to(device).eval()
        self.conditioner = conditioner.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.window, self.stride = window, stride
        self.ddim_steps = ddim_steps
        self.resample_attempts = resample_attempts

    def _condition_batch(self, cond: ReferenceCondition) -> dict:
        wfc = cond.world_from_clip
        kps_obj = keypoints_in_source_object(cond.human_kps_world, cond.source_obj_pose)
        kps_clip = world_to_clip(cond.human_kps_world, wfc)
        return {
            "kps_obj": kps_obj, "kps_clip": kps_clip,
            "box_t_clip": world_to_clip(cond.target_obj_pose[:, 0:3], wfc),
            "box_quat": cond.target_obj_pose[:, 3:7],
            "contact": cond.contact_flags().astype(np.float32),
        }

    @torch.no_grad()
    def _sample_windows(self, batch: dict, seed: int) -> tuple[np.ndarray, int]:
        """batch over windows -> decoded (pos_clip, quat_clip, q) lists + degenerate count."""
        from ireset.diffusion.generator import DiffusionG1

        Nw, F = batch["kps_obj"].shape[:2]
        dev = self.device

        def to_t(a):
            return torch.as_tensor(a, dtype=torch.float32, device=dev)

        tbatch = {k: to_t(v) for k, v in batch.items()}

        bad = np.zeros(Nw, dtype=bool)
        x0 = None
        for attempt in range(self.resample_attempts):
            gen = torch.Generator(device=dev).manual_seed(seed + attempt)
            if isinstance(self.model, DiffusionG1):
                out = self.model.sample(
                    self.conditioner(tbatch), F, steps=self.ddim_steps, generator=gen
                )
            else:
                out = self.model(self.conditioner(tbatch), F)
            ok = torch.isfinite(out).all(dim=(1, 2)) & \
                (out[..., 3:6].norm(dim=-1) > 1e-6).all(1) & \
                (out[..., 6:9].norm(dim=-1) > 1e-6).all(1)
            if x0 is None:
                x0 = out.clone()
            else:
                x0[~ok] = out[~ok]
            bad = bad | ~ok.cpu().numpy()
            if not bad.any():
                break
        n_bad_final = int(bad.sum())

        # decode windows in the clip frame
        mean, std = self.tokenizer.root_stats.to_torch(dev)
        lo = torch.tensor(self.tokenizer.lo, dtype=torch.float32, device=dev)
        hi = torch.tensor(self.tokenizer.hi, dtype=torch.float32, device=dev)
        from ireset.kinematics.rotation_adapter import rot6d_to_mat_torch
        from ireset.model.tokens import mat_to_quat_torch

        pos = x0[..., 0:3] * std + mean
        quat = mat_to_quat_torch(rot6d_to_mat_torch(x0[..., 3:9]))
        q = lo + (x0[..., 9:38].clamp(-1, 1) + 1) * (hi - lo) / 2
        return (pos.cpu().numpy(), quat.cpu().numpy(), q.cpu().numpy()), n_bad_final

    def sample(self, condition: ReferenceCondition, seed: int = 0) -> RobotTrajectory:
        t0 = time.perf_counter()
        cond = condition
        arrays = self._condition_batch(cond)
        T = cond.T
        F, stride = self.window, self.stride
        starts = window_starts(T, F, stride)
        Nw = len(starts)

        windows = {
            k: np.stack([v[s : s + F] for s in starts])
            for k, v in arrays.items()
        }
        (pos_w, quat_w, q_w), n_bad = self._sample_windows(windows, seed)
        pos, quat, q, diag = fuse_windows(
            list(pos_w), list(quat_w), list(q_w), starts, T, stride
        )

        # clip -> world
        Rz, t = cond.world_from_clip[:3, :3], cond.world_from_clip[:3, 3]
        pos_world = pos @ Rz.T + t
        q_clip = Rot.from_quat(wxyz_to_xyzw(quat))
        q_world = (Rot.from_matrix(Rz) * q_clip).as_quat()   # Rz @ R_clip
        root = np.concatenate([pos_world, xyzw_to_wxyz(q_world)], axis=1)
        gen_wall = time.perf_counter() - t0

        return RobotTrajectory(
            root=root,
            q=q,
            frame_dt=cond.frame_dt,
            metadata={
                "npz": cond.source_clip_id,
                "windows": Nw,
                "starts": starts,
                "degenerate_windows": n_bad,
                "resample_attempts_used": self.resample_attempts,
                "ddim_steps": None if not _is_diffusion(self) else self.ddim_steps,
                "generation_wall_s": gen_wall,
                "boundary": diag,
            },
        )


def _is_diffusion(gen: G1Generator) -> bool:
    from ireset.diffusion.generator import DiffusionG1

    return isinstance(gen.model, DiffusionG1)
