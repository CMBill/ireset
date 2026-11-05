"""30-token-per-frame transformer backbone (plan section 7.1).

Per frame the model holds **1 root token + 29 joint tokens** (d_model=240,
4 layers, 6 heads pilot). Root and joint tokens use separate input/output
projections; the root head emits normalized position + 6D rotation, the
joint head emits one logit per joint (clean-angle sigmoid map lives in the
loss/sampling assembly, never on noisy inputs).

Self-attention is windowed over time (``time_window`` frames, default 31 =
plan's temporal attention window); cross-attention reads the condition
memory from :class:`ireset.model.conditioner.ConditionEncoder`. The same
backbone serves the diffusion model (with timestep embedding) and the
deterministic predictor (``use_time=False``, no noisy input: tokens start
from learned embeddings only).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ireset.model.conditioner import sinusoidal_frame_encoding

N_TOKENS_PER_FRAME = 30   # 1 root + 29 joints
ROOT_FEATURES, JOINT_FEATURES = 9, 1


class Attn(nn.Module):
    """Scaled dot-product attention in explicit math form.

    Originally written as explicit ``softmax(QK^T/sqrt(d)) + mask``
    because ``F.scaled_dot_product_attention`` pulled in Triton, which
    could not compile on the pre-migration system interpreter (no
    Python.h). The venv now runs uv-managed CPython 3.12.13 where Triton
    and SDPA work, but this path is kept so pilot runs stay numerically
    reproducible; switching to SDPA is a deferred speed optimization.
    ``mask`` is a boolean keep-mask broadcastable to [*, L, S].
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        kv = x if ctx is None else ctx
        B, L, d = x.shape
        S = kv.shape[1]
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)      # [B,h,L,dk]
        # A linear key bias adds the same vector q.b to every logit of a
        # query, so softmax is invariant to it.  Passing ``bias * 0`` keeps
        # the parameter in the graph with an exactly zero gradient instead of
        # dropping it (an omitted bias would produce grad None and alter the
        # optimizer/step semantics).  This is only valid for this shared
        # linear key projection -- never apply it to Q/V biases or nonlinear
        # key paths.
        key_bias = (None if self.k.bias is None else self.k.bias * 0.0)
        k = F.linear(kv, self.k.weight, key_bias).view(
            B, S, self.h, self.dk).transpose(1, 2)
        v = self.v(kv).view(B, S, self.h, self.dk).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / (self.dk ** 0.5)            # [B,h,L,S]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        y = logits.softmax(dim=-1) @ v
        y = y.transpose(1, 2).reshape(B, L, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = Attn(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = Attn(d_model, n_heads)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(), nn.Linear(ffn_mult * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor, self_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), mask=self_mask)
        x = x + self.cross_attn(self.ln2(x), ctx=mem)
        x = x + self.ffn(self.ln3(x))
        return x


class TimestepEmbed(nn.Module):
    def __init__(self, d_model: int, freq_dim: int = 128) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                                            device=t.device) / half)
        ang = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.mlp(emb)


class G1Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 240,
        n_layers: int = 4,
        n_heads: int = 6,
        ffn_mult: int = 4,
        time_window: int = 31,
        use_time: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.time_window = time_window
        self.use_time = use_time
        self.root_in = nn.Linear(ROOT_FEATURES, d_model)
        self.joint_in = nn.Linear(JOINT_FEATURES, d_model)
        self.type_emb = nn.Embedding(2, d_model)          # 0 root, 1 joint
        self.joint_idx_emb = nn.Embedding(29, d_model)
        self.time_emb = TimestepEmbed(d_model) if use_time else None
        self.mem_ln = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffn_mult) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.root_out = nn.Linear(d_model, ROOT_FEATURES)
        self.joint_out = nn.Linear(d_model, JOINT_FEATURES)
        self._mask_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _self_mask(self, F_: int, device) -> torch.Tensor:
        key = (F_, str(device))
        m = self._mask_cache.get(key)
        if m is None:
            hw = (self.time_window - 1) // 2
            t = torch.arange(F_)
            gap = (t[:, None] - t[None, :]).abs()          # [F,F]
            m = (gap <= hw).repeat_interleave(N_TOKENS_PER_FRAME, 0) \
                .repeat_interleave(N_TOKENS_PER_FRAME, 1).to(device)
            self._mask_cache[key] = m
        return m

    def forward(
        self,
        x: torch.Tensor | None,        # [B, F, 38] noisy (or clean) states; None if use_time=False
        t: torch.Tensor | None,        # [B] diffusion timesteps
        memory: torch.Tensor,          # [B, N, d] condition tokens
        frames: int | None = None,     # required when x is None (deterministic path)
    ) -> dict[str, torch.Tensor]:
        if x is not None:
            B, F_ = x.shape[0], x.shape[1]
        else:
            assert not self.use_time and frames is not None, \
                "deterministic backbone needs the explicit frame count"
            B, F_ = memory.shape[0], frames
        device = memory.device
        if x is None:
            x = torch.zeros(B, F_, 38, device=device)

        root_tok = self.root_in(x[..., :ROOT_FEATURES]) + self.type_emb.weight[0]      # [B,F,d]
        joint_tok = self.joint_in(x[..., ROOT_FEATURES:ROOT_FEATURES + 29][..., None]) \
            + self.joint_idx_emb.weight[None] + self.type_emb.weight[1]                # [B,F,29,d]

        frame = sinusoidal_frame_encoding(F_, self.d_model, device)                    # [F,d]
        tokens = torch.cat(
            [root_tok[:, :, None] + frame[:, None], joint_tok + frame[:, None]], dim=2
        ).reshape(B, F_ * N_TOKENS_PER_FRAME, self.d_model)

        if self.use_time:
            assert t is not None
            tokens = tokens + self.time_emb(t.to(device))[:, None]

        mem = self.mem_ln(memory)
        mask = self._self_mask(F_, device)
        for blk in self.blocks:
            tokens = blk(tokens, mem, mask)
        tokens = self.ln_out(tokens).view(B, F_, N_TOKENS_PER_FRAME, self.d_model)

        return {
            "root": self.root_out(tokens[:, :, 0]),                # [B,F,9]
            "joint_logits": self.joint_out(tokens[:, :, 1:]).squeeze(-1),  # [B,F,29]
        }


def assemble_x0(root: torch.Tensor, joint_logits: torch.Tensor) -> torch.Tensor:
    """Head outputs -> normalized x0 prediction (joint sigmoid map here only)."""
    u = 2.0 * torch.sigmoid(joint_logits) - 1.0
    return torch.cat([root, u], dim=-1)
