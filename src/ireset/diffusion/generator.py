"""x0-prediction diffusion wrapper (plan section 7.2).

DDPM training noise + DDIM sampling through ``diffusers`` schedulers, both
configured with ``prediction_type="sample"`` (x0-prediction) and the same
beta schedule. The scheduler-level global clipping is disabled
(``clip_sample=False``; DDIM ``thresholding`` off): joint limits are the
clean-angle sigmoid map in :func:`ireset.model.backbone.assemble_x0`, and
root position/rotation must never be clipped by the sampler.

The last DDIM step returns exactly the model's clean prediction
(``set_alpha_to_one=True`` makes the final previous-SNR 1), matching the
plan's "decode the final step to physical (root, q)".
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler

from ireset.model.backbone import G1Transformer, assemble_x0


class DiffusionG1(nn.Module):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "scaled_linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        **backbone_kwargs,
    ) -> None:
        super().__init__()
        backbone_kwargs.setdefault("use_time", True)
        self.net = G1Transformer(**backbone_kwargs)
        sched = dict(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type="sample",
            clip_sample=False,
        )
        self.ddpm = DDPMScheduler(**sched)
        self.ddim = DDIMScheduler(**sched)   # same beta schedule; thresholding off by default

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Noisy states [B,F,38], timesteps [B] -> x0 prediction [B,F,38]."""
        out = self.net(x_t, t, memory)
        return assemble_x0(out["root"], out["joint_logits"])

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.ddpm.add_noise(x0, noise, t)

    def sample_timesteps(self, B: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randint(0, self.ddpm.config.num_train_timesteps, (B,),
                             device=device, generator=generator)

    @torch.no_grad()
    def sample(
        self,
        memory: torch.Tensor,
        F: int,
        steps: int = 25,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Condition memory -> normalized sample [B, F, 38] via DDIM."""
        B, device = memory.shape[0], memory.device
        self.ddim.set_timesteps(steps)
        x = torch.randn(B, F, 38, device=device, generator=generator)
        for t in self.ddim.timesteps:
            x0_pred = self.forward(x, t.expand(B).to(device), memory)
            x = self.ddim.step(x0_pred, t, x, eta=eta).prev_sample
        return x
