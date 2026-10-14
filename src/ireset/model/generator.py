"""Condition -> full-length raw trajectory with bounded numerical retries.

The accepted decoder, window layout and SLERP fusion stay unchanged. Numerical
failures never trigger quality-based selection or silent repair: diffusion may
replace currently invalid windows using at most three seed+attempt calls;
deterministic inference has one call. Final rejection retains retry evidence.
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


class GenerationNumericalError(ValueError):
    """Rejected raw generation, with evidence and the actual failed output.

    ``metadata``/``evidence`` are the same JSON-compatible dictionary. ``output``
    is normalized windows for a window rejection, or decoded/full arrays when
    a later stage fails. It is evidence, never a usable RobotTrajectory.
    """

    def __init__(self, reason: str, metadata: dict, output=None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = {**metadata, "numerical_valid": False, "numerical_error": reason}
        self.evidence = self.metadata
        self.output = output


def _valid_windows(out: torch.Tensor) -> np.ndarray:
    # Finite-first: even Inf joints that clamp to a limit are invalid raw output.
    finite = torch.isfinite(out).all(dim=(1, 2))
    a, b = out[..., 3:6], out[..., 6:9]
    orth = b - (a * b).sum(-1, keepdim=True) * a / (a * a).sum(-1, keepdim=True).clamp_min(1e-12)
    na, nb = a.norm(dim=-1), orth.norm(dim=-1)
    nc = torch.linalg.cross(a, orth, dim=-1).norm(dim=-1)
    rotation = (torch.isfinite(na) & torch.isfinite(nb) & torch.isfinite(nc)
                & (na > 1e-6) & (nb > 1e-6) & (nc > 1e-12)).all(1)
    return (finite & rotation).cpu().numpy()


class G1Generator:
    def __init__(self, model: torch.nn.Module, conditioner: torch.nn.Module,
                 tokenizer, device: str = "cpu", window: int = 60,
                 stride: int = 30, ddim_steps: int = 25,
                 resample_attempts: int = 3) -> None:
        for name, value in (("window", window), ("stride", stride),
                            ("ddim_steps", ddim_steps), ("resample_attempts", resample_attempts)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if resample_attempts > 3:
            raise ValueError("resample_attempts must be at most 3")
        if stride > window:
            raise ValueError("stride must not leave uncovered frames")
        self.model = model.to(device).eval()
        self.conditioner = conditioner.to(device).eval()
        self.tokenizer, self.device = tokenizer, device
        self.window, self.stride = window, stride
        self.ddim_steps, self.resample_attempts = ddim_steps, resample_attempts
        self._last_sample_stats: dict = {}

    def _synchronize(self) -> None:
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

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
    def _sample_windows(self, batch: dict, seed: int) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], int]:
        """Sample the frozen batch layout; replace only currently invalid slots.

        Keeping the full batch for each seeded call preserves its sampling
        layout. A later invalid value in an already accepted slot is ignored.
        No task, collision, contact or fit metric is read to decide a retry.
        """
        from ireset.kinematics.rotation_adapter import rot6d_to_mat_torch
        from ireset.model.tokens import mat_to_quat_torch

        self._synchronize()
        t0 = time.perf_counter()
        stats = self._last_sample_stats = {
            "model_calls": 0, "conditioner_calls": 0, "resample_attempts_used": 0,
            "resample_attempts_max": self.resample_attempts,
            "per_attempt_invalid_windows": [], "per_attempt_invalid_window_indices": [],
            "attempt_seeds": [], "attempts": [], "final_invalid_windows": None,
            "final_invalid_window_indices": [], "numerical_valid": False,
            "model_wall_s": 0.0, "decode_wall_s": 0.0,
        }
        x0 = None
        try:
            Nw, F = batch["kps_obj"].shape[:2]
            if Nw < 1 or F < 1:
                raise GenerationNumericalError("empty_window_batch", stats)
            dev = self.device
            tbatch = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in batch.items()}
            for key, value in tbatch.items():
                if value.shape[:2] != (Nw, F) or not bool(torch.isfinite(value).all()):
                    raise GenerationNumericalError(f"invalid_condition:{key}", stats)
            for side, module in (("model", self.model), ("conditioner", self.conditioner)):
                for name, p in module.named_parameters():
                    if not bool(torch.isfinite(p).all()):
                        raise GenerationNumericalError(f"nonfinite_parameter:{side}.{name}", stats)
            invalid = np.ones(Nw, dtype=bool)
            max_attempts = self.resample_attempts if _is_diffusion(self) else 1
            for attempt in range(max_attempts):
                self._synchronize()
                call_t0 = time.perf_counter()
                gen = torch.Generator(device=dev).manual_seed(seed + attempt)
                stats["attempt_seeds"].append(int(seed + attempt))
                stats["conditioner_calls"] += 1
                memory = self.conditioner(tbatch)
                if not torch.is_tensor(memory) or not bool(torch.isfinite(memory).all()):
                    raise GenerationNumericalError("nonfinite_condition_memory", stats, x0)
                stats["model_calls"] += 1
                stats["resample_attempts_used"] += 1
                if _is_diffusion(self):
                    out = self.model.sample(memory, F, steps=self.ddim_steps, generator=gen)
                else:
                    out = self.model(memory, F)
                self._synchronize()
                call_wall = time.perf_counter() - call_t0
                stats["model_wall_s"] += call_wall
                if not torch.is_tensor(out) or tuple(out.shape) != (Nw, F, 38):
                    raise GenerationNumericalError("invalid_prediction_shape", stats,
                                                   out.detach().cpu().numpy() if torch.is_tensor(out) else out)
                ok_np = _valid_windows(out)
                attempted_indices = np.flatnonzero(invalid).tolist()
                if x0 is None:
                    x0 = out.clone()
                    invalid = ~ok_np
                else:
                    take = torch.as_tensor(invalid, device=dev)
                    x0[take] = out[take]
                    invalid = invalid & ~ok_np
                remaining = np.flatnonzero(invalid).tolist()
                stats["per_attempt_invalid_windows"].append(len(remaining))
                stats["per_attempt_invalid_window_indices"].append(remaining)
                stats["attempts"].append({"attempt_index": attempt, "seed": int(seed + attempt),
                                          "replacement_window_indices": attempted_indices,
                                          "invalid_window_indices": remaining, "wall_s": call_wall})
                stats["final_invalid_windows"] = len(remaining)
                stats["final_invalid_window_indices"] = remaining
                if not invalid.any():
                    break
            if invalid.any():
                raise GenerationNumericalError("invalid_windows_after_attempts", stats, x0.detach().cpu().numpy())

            # Preserve exact first-success decoder math, checking before and after
            # transformations which could turn an invalid raw value into a number.
            decode_t0 = time.perf_counter()
            mean, std = self.tokenizer.root_stats.to_torch(dev)
            lo = torch.tensor(self.tokenizer.lo, dtype=torch.float32, device=dev)
            hi = torch.tensor(self.tokenizer.hi, dtype=torch.float32, device=dev)
            if (not all(bool(torch.isfinite(v).all()) for v in (mean, std, lo, hi))
                    or not bool((std > 0).all()) or not bool((hi > lo).all())):
                raise GenerationNumericalError("invalid_decoder_statistics", stats, x0.detach().cpu().numpy())
            pos = x0[..., 0:3] * std + mean
            quat = mat_to_quat_torch(rot6d_to_mat_torch(x0[..., 3:9]))
            q = lo + (x0[..., 9:38].clamp(-1, 1) + 1) * (hi - lo) / 2
            self._synchronize()
            stats["decode_wall_s"] = time.perf_counter() - decode_t0
            decoded = (pos.cpu().numpy(), quat.cpu().numpy(), q.cpu().numpy())
            if (not all(np.isfinite(v).all() for v in decoded)
                    or not np.all(np.abs(np.linalg.norm(decoded[1], axis=-1) - 1) <= 1e-5)):
                raise GenerationNumericalError("invalid_decoded_windows", stats, decoded)
            stats["numerical_valid"] = True
            stats["sampling_wall_s"] = time.perf_counter() - t0
            return decoded, 0
        except GenerationNumericalError as exc:
            self._synchronize()
            stats["sampling_wall_s"] = time.perf_counter() - t0
            stats["numerical_valid"] = False
            exc.metadata.update(stats)
            self._last_sample_stats = exc.metadata
            raise

    def sample(self, condition: ReferenceCondition, seed: int = 0) -> RobotTrajectory:
        self._last_sample_stats = {}
        self._synchronize()
        t0 = time.perf_counter()
        cond = condition
        metadata = {"npz": cond.source_clip_id, "seed": int(seed), "numerical_valid": False,
                    "T": cond.T, "window": self.window, "stride": self.stride,
                    "model_calls": 0, "resample_attempts_used": 0,
                    "resample_attempts_max": self.resample_attempts,
                    "ddim_steps": self.ddim_steps if _is_diffusion(self) else None}
        output = None
        try:
            T, F, stride = cond.T, self.window, self.stride
            if T < F:
                raise GenerationNumericalError("clip_shorter_than_window", metadata)
            for name, value in (("human_kps_world", cond.human_kps_world),
                                ("source_obj_pose", cond.source_obj_pose),
                                ("target_obj_pose", cond.target_obj_pose),
                                ("world_from_clip", cond.world_from_clip)):
                if value is None or not np.isfinite(value).all():
                    raise GenerationNumericalError(f"invalid_condition:{name}", metadata)
            if (cond.human_kps_world.shape != (T, 52, 3)
                    or cond.source_obj_pose.shape != (T, 7) or cond.target_obj_pose.shape != (T, 7)
                    or cond.world_from_clip.shape != (4, 4)
                    or isinstance(cond.frame_dt, bool) or not np.isfinite(cond.frame_dt) or cond.frame_dt <= 0):
                raise GenerationNumericalError("invalid_condition_shape_or_frame_dt", metadata)
            for pose in (cond.source_obj_pose, cond.target_obj_pose):
                n = np.linalg.norm(pose[:, 3:7], axis=-1)
                if not (np.isfinite(n).all() and (n > 1e-6).all()):
                    raise GenerationNumericalError("degenerate_condition_quaternion", metadata)
            Rz = cond.world_from_clip[:3, :3]
            if not (np.allclose(Rz.T @ Rz, np.eye(3), atol=1e-6, rtol=0)
                    and abs(np.linalg.det(Rz) - 1) <= 1e-6):
                raise GenerationNumericalError("invalid_clip_rotation", metadata)
            arrays = self._condition_batch(cond)
            starts = window_starts(T, F, stride)
            metadata.update(windows=len(starts), starts=starts,
                            last_frame_covered=starts[-1] + F == T)
            windows = {k: np.stack([v[s:s + F] for s in starts]) for k, v in arrays.items()}
            metadata["condition_wall_s"] = time.perf_counter() - t0
            (pos_w, quat_w, q_w), _ = self._sample_windows(windows, seed)
            metadata.update(self._last_sample_stats)
            fuse_t0 = time.perf_counter()
            output = (pos_w, quat_w, q_w)
            pos, quat, q, diag = fuse_windows(list(pos_w), list(quat_w), list(q_w), starts, T, stride)
            if not all(np.isfinite(v).all() for v in (pos, quat, q)):
                raise GenerationNumericalError("nonfinite_fused_output", metadata, (pos, quat, q))
            # Exact accepted clip -> world lifting.
            t = cond.world_from_clip[:3, 3]
            pos_world = pos @ Rz.T + t
            q_clip = Rot.from_quat(wxyz_to_xyzw(quat))
            q_world = (Rot.from_matrix(Rz) * q_clip).as_quat()
            root = np.concatenate([pos_world, xyzw_to_wxyz(q_world)], axis=1)
            output = (root, q)
            self._synchronize()
            if (root.shape != (T, 7) or q.shape != (T, 29)
                    or not np.isfinite(root).all() or not np.isfinite(q).all()
                    or not np.all(np.abs(np.linalg.norm(root[:, 3:7], axis=-1) - 1) <= 1e-5)):
                raise GenerationNumericalError("invalid_full_output", metadata, output)
            metadata.update(numerical_valid=True, degenerate_windows=0,
                            fusion_wall_s=time.perf_counter() - fuse_t0,
                            generation_wall_s=time.perf_counter() - t0, boundary=diag)
            return RobotTrajectory(root=root, q=q, frame_dt=cond.frame_dt, metadata=metadata)
        except GenerationNumericalError as exc:
            self._synchronize()
            # The exception already carries its latest retry facts. Never replace
            # them with this wrapper's stale pre-sampling zero-call defaults.
            exc.metadata.update({**metadata, **exc.metadata,
                                 "degenerate_windows": exc.metadata.get("final_invalid_windows"),
                                 "generation_wall_s": time.perf_counter() - t0})
            raise
        except Exception as exc:
            self._synchronize()
            metadata.update(self._last_sample_stats)
            metadata["generation_wall_s"] = time.perf_counter() - t0
            raise GenerationNumericalError(f"generation_error:{type(exc).__name__}:{exc}", metadata, output) from exc


def _is_diffusion(gen: G1Generator) -> bool:
    from ireset.diffusion.generator import DiffusionG1

    return isinstance(gen.model, DiffusionG1)
