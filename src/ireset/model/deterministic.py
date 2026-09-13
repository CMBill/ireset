"""Deterministic predictor (plan section 7.3).

Same conditioning, output parameterization and backbone capacity as the
diffusion model, minus the noise timestep and sampling loop: a single
forward pass maps the condition memory to the clean normalized state.
Trained with the same loss assembly on the same split; uses the same
repairer and timing convention at evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ireset.model.backbone import G1Transformer, assemble_x0


class DeterministicG1(nn.Module):
    def __init__(self, **backbone_kwargs) -> None:
        super().__init__()
        backbone_kwargs.setdefault("use_time", False)
        self.net = G1Transformer(**backbone_kwargs)

    def forward(self, memory: torch.Tensor, frames: int) -> torch.Tensor:
        """Condition memory [B, N, d], frame count -> normalized x0 [B, F, 38]."""
        out = self.net(None, None, memory, frames=frames)
        return assemble_x0(out["root"], out["joint_logits"])
