"""OOD detection on z_p plus a lightweight calibration head. See design doc §13."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class OODStats:
    mu: np.ndarray  # mean over training z_p
    sigma_inv: np.ndarray  # inverse covariance
    threshold: float  # Mahalanobis squared distance, at quantile q


def fit_ood_stats(z_p_train: np.ndarray, quantile: float = 0.95) -> OODStats:
    mu = z_p_train.mean(axis=0)
    cov = np.cov(z_p_train, rowvar=False) + np.eye(z_p_train.shape[1]) * 1e-3
    cov_inv = np.linalg.inv(cov)
    diff = z_p_train - mu
    d2 = np.einsum("nd,de,ne->n", diff, cov_inv, diff)
    threshold = float(np.quantile(d2, quantile))
    return OODStats(mu=mu, sigma_inv=cov_inv, threshold=threshold)


def is_ood(z_p_target: np.ndarray, stats: OODStats) -> bool:
    diff = z_p_target - stats.mu
    d2 = float(diff @ stats.sigma_inv @ diff.T)
    return d2 > stats.threshold


class CalibrationHead(nn.Module):
    """Linear residual on top of the frozen scorer's μ̂.

    μ_cal = μ̂ + W_cal · [z_x; z_p*; z_t] + b_cal
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, mu_hat: torch.Tensor, z_x: torch.Tensor,
                z_p: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        if z_p.shape[0] == 1 and z_x.shape[0] != 1:
            z_p = z_p.expand(z_x.shape[0], -1)
        if z_t.shape[0] == 1 and z_x.shape[0] != 1:
            z_t = z_t.expand(z_x.shape[0], -1)
        inp = torch.cat([z_x, z_p, z_t], dim=-1)
        return mu_hat + self.linear(inp).squeeze(-1)


def fit_calibration(
    *,
    head: CalibrationHead,
    mu_hat: np.ndarray,
    u_hi: np.ndarray,
    z_x: np.ndarray,
    z_p: np.ndarray,
    z_t: np.ndarray,
    epochs: int = 200,
    lr: float = 5e-3,
    device: str = "cuda",
) -> Path | None:
    """Closed-form-ish linear regression fit; uses GD because module shape is
    convenient. Returns nothing; mutates head in place."""
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    head.to(device_t).train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    mu_t = torch.as_tensor(mu_hat, device=device_t)
    u_t = torch.as_tensor(u_hi, device=device_t)
    zx_t = torch.as_tensor(z_x, device=device_t)
    zp_t = torch.as_tensor(z_p, device=device_t)
    zt_t = torch.as_tensor(z_t, device=device_t)
    for _ in range(epochs):
        pred = head(mu_t, zx_t, zp_t, zt_t)
        loss = ((pred - u_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return None
