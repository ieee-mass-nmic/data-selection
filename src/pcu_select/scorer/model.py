"""Scorer network: 3-tower + FiLM + bilinear fusion. See design doc §11."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ScorerConfig:
    z_x_dim: int = 848
    z_p_dim: int = 128
    z_t_dim: int = 848
    h_dim: int = 256
    p_dim: int = 128
    bilin_dim: int = 64
    sigma_floor: float = 1e-3


def _mlp(in_dim: int, out_dim: int, hidden: int | None = None) -> nn.Sequential:
    hidden = hidden or out_dim
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
        nn.LayerNorm(out_dim),
    )


class PCUScorer(nn.Module):
    def __init__(self, cfg: ScorerConfig | None = None):
        super().__init__()
        self.cfg = cfg or ScorerConfig()
        c = self.cfg
        self.f_x = _mlp(c.z_x_dim, c.h_dim)
        self.f_p = _mlp(c.z_p_dim, c.p_dim)
        self.f_t = _mlp(c.z_t_dim, c.h_dim)
        # FiLM modulation parameters from concat(h_p, h_t)
        self.film_gamma = nn.Linear(c.p_dim + c.h_dim, c.h_dim)
        self.film_beta = nn.Linear(c.p_dim + c.h_dim, c.h_dim)
        self.bilinear = nn.Bilinear(c.h_dim, c.p_dim, c.bilin_dim)
        self.head = nn.Sequential(
            nn.Linear(c.h_dim + c.bilin_dim, c.h_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(c.h_dim, 1)
        self.sigma_head = nn.Linear(c.h_dim, 1)

    def forward(self, z_x: torch.Tensor, z_p: torch.Tensor, z_t: torch.Tensor):
        # Broadcast: if z_p / z_t have batch=1, expand.
        if z_p.shape[0] == 1 and z_x.shape[0] != 1:
            z_p = z_p.expand(z_x.shape[0], -1)
        if z_t.shape[0] == 1 and z_x.shape[0] != 1:
            z_t = z_t.expand(z_x.shape[0], -1)
        h_x = self.f_x(z_x)
        h_p = self.f_p(z_p)
        h_t = self.f_t(z_t)
        cond = torch.cat([h_p, h_t], dim=-1)
        gamma = self.film_gamma(cond)
        beta = self.film_beta(cond)
        h_film = gamma * h_x + beta
        h_bilin = self.bilinear(h_x, h_p)
        h = torch.cat([h_film, h_bilin], dim=-1)
        h = self.head(h)
        mu = self.mu_head(h).squeeze(-1)
        sigma = torch.nn.functional.softplus(self.sigma_head(h).squeeze(-1)) + self.cfg.sigma_floor
        return mu, sigma
