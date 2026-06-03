"""Task condition encoder z_t. See design doc §9.

Implements a set-transformer style attention pool over the per-sketch-sample
representations z_v (which share the same shape as z_x).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TaskEncoderConfig:
    z_v_dim: int = 848  # matches d_sem + d_diff + d_act
    hidden_dim: int = 256
    n_queries: int = 4
    n_heads: int = 4
    dropout: float = 0.0


class SetTransformerPool(nn.Module):
    """Multi-head attention with learned query bank, then mean over queries.

    Input : (V, z_v_dim)
    Output: (z_v_dim,)  — same dimensionality as a single z_v, so downstream
            scorer can treat z_t identically to z_x.
    """

    def __init__(self, cfg: TaskEncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or TaskEncoderConfig()
        self.proj_in = nn.Linear(self.cfg.z_v_dim, self.cfg.hidden_dim)
        self.queries = nn.Parameter(torch.randn(self.cfg.n_queries, self.cfg.hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.cfg.hidden_dim,
            num_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )
        self.proj_out = nn.Linear(self.cfg.hidden_dim, self.cfg.z_v_dim)
        self.norm = nn.LayerNorm(self.cfg.z_v_dim)

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        # z_v: (V, d)
        h = self.proj_in(z_v).unsqueeze(0)  # (1, V, hidden)
        q = self.queries.unsqueeze(0)  # (1, n_queries, hidden)
        attn_out, _ = self.attn(q, h, h)  # (1, n_queries, hidden)
        pooled = attn_out.mean(dim=1)  # (1, hidden)
        out = self.proj_out(pooled).squeeze(0)
        return self.norm(out)

    @torch.no_grad()
    def encode_numpy(self, z_v: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(z_v, dtype=torch.float32, device=next(self.parameters()).device)
        out = self.forward(tensor)
        return out.cpu().numpy()
