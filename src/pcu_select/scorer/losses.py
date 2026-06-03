"""Scorer training losses. See design doc §11.3."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_rank_loss(mu_i: torch.Tensor, mu_j: torch.Tensor,
                       u_i: torch.Tensor, u_j: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(u_i - u_j)
    diff = mu_i - mu_j
    return -F.logsigmoid(sign * diff).mean()


def huber_reg(mu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(mu, u)


def proxy_distill_loss(mu: torch.Tensor, u_lo: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(mu, u_lo)


def heteroscedastic_nll(mu: torch.Tensor, sigma: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    var = sigma.pow(2)
    return (0.5 * (u - mu).pow(2) / var + 0.5 * torch.log(var)).mean()


def combine_losses(*, mu: torch.Tensor, sigma: torch.Tensor,
                   u_hi: torch.Tensor | None, u_lo: torch.Tensor | None,
                   weights: tuple[float, float, float, float],
                   rank_pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                   ) -> tuple[torch.Tensor, dict[str, float]]:
    """rank_pairs: optional (mu_i, mu_j, u_i, u_j) for pairwise sampling."""
    w_rank, w_reg, w_proxy, w_unc = weights
    parts: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=mu.device)
    if rank_pairs is not None and w_rank > 0:
        parts["rank"] = pairwise_rank_loss(*rank_pairs) * w_rank
        total = total + parts["rank"]
    if u_hi is not None and w_reg > 0:
        parts["reg"] = huber_reg(mu, u_hi) * w_reg
        total = total + parts["reg"]
    if u_lo is not None and w_proxy > 0:
        parts["proxy"] = proxy_distill_loss(mu, u_lo) * w_proxy
        total = total + parts["proxy"]
    if u_hi is not None and w_unc > 0:
        parts["unc"] = heteroscedastic_nll(mu, sigma, u_hi) * w_unc
        total = total + parts["unc"]
    return total, {k: float(v.detach()) for k, v in parts.items()}
