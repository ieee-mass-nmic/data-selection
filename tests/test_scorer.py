"""Scorer network + loss function tests (CPU torch, no GPU/model download)."""

from __future__ import annotations

import torch

from pcu_select.scorer.losses import (
    combine_losses,
    heteroscedastic_nll,
    pairwise_rank_loss,
)
from pcu_select.scorer.model import PCUScorer, ScorerConfig


def _scorer() -> PCUScorer:
    torch.manual_seed(0)
    return PCUScorer(ScorerConfig())


def test_forward_output_shapes_and_sigma_floor():
    model = _scorer()
    b = 5
    z_x = torch.randn(b, model.cfg.z_x_dim)
    z_p = torch.randn(b, model.cfg.z_p_dim)
    z_t = torch.randn(b, model.cfg.z_t_dim)
    mu, sigma = model(z_x, z_p, z_t)
    assert mu.shape == (b,)
    assert sigma.shape == (b,)
    # softplus + floor must keep sigma strictly above the floor.
    assert torch.all(sigma >= model.cfg.sigma_floor)
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()


def test_forward_broadcasts_single_peft_and_task():
    model = _scorer()
    b = 4
    z_x = torch.randn(b, model.cfg.z_x_dim)
    z_p = torch.randn(1, model.cfg.z_p_dim)  # broadcast across the batch
    z_t = torch.randn(1, model.cfg.z_t_dim)
    mu, sigma = model(z_x, z_p, z_t)
    assert mu.shape == (b,)
    assert sigma.shape == (b,)


def test_pairwise_rank_loss_prefers_correct_ordering():
    # mu agrees with u ordering -> low loss; disagrees -> high loss.
    u_i = torch.tensor([1.0, 1.0])
    u_j = torch.tensor([0.0, 0.0])
    mu_correct_i = torch.tensor([2.0, 2.0])
    mu_correct_j = torch.tensor([0.0, 0.0])
    mu_wrong_i = torch.tensor([0.0, 0.0])
    mu_wrong_j = torch.tensor([2.0, 2.0])
    good = pairwise_rank_loss(mu_correct_i, mu_correct_j, u_i, u_j)
    bad = pairwise_rank_loss(mu_wrong_i, mu_wrong_j, u_i, u_j)
    assert good < bad


def test_heteroscedastic_nll_rewards_calibrated_sigma():
    # When mu == u, smaller sigma should yield a smaller NLL (until the floor).
    mu = torch.zeros(8)
    u = torch.zeros(8)
    nll_small = heteroscedastic_nll(mu, torch.full((8,), 0.1), u)
    nll_large = heteroscedastic_nll(mu, torch.full((8,), 1.0), u)
    assert nll_small < nll_large


def test_combine_losses_selects_only_active_terms():
    mu = torch.zeros(4, requires_grad=True)
    sigma = torch.ones(4)
    u_lo = torch.ones(4)
    # Only proxy is active (no rank pairs, no u_hi).
    total, parts = combine_losses(
        mu=mu, sigma=sigma, u_hi=None, u_lo=u_lo,
        weights=(1.0, 0.3, 0.5, 0.2), rank_pairs=None,
    )
    assert set(parts.keys()) == {"proxy"}
    assert total.requires_grad
    total.backward()
    assert mu.grad is not None


def test_combine_losses_all_terms_present_with_pairs():
    mu = torch.zeros(4)
    sigma = torch.ones(4)
    u_hi = torch.ones(4)
    u_lo = torch.ones(4)
    pairs = (torch.zeros(2), torch.zeros(2), torch.ones(2), torch.zeros(2))
    total, parts = combine_losses(
        mu=mu, sigma=sigma, u_hi=u_hi, u_lo=u_lo,
        weights=(1.0, 0.3, 0.5, 0.2), rank_pairs=pairs,
    )
    assert set(parts.keys()) == {"rank", "reg", "proxy", "unc"}
    assert torch.isfinite(total)
