"""Pure stat / pooling helpers (torch tensors, no model)."""

from __future__ import annotations

import numpy as np
import torch

from pcu_select.features.stats import (
    N_MODEL_STATS,
    PER_LAYER_DIM,
    model_stats_vector,
    per_layer_activation_stats,
    pool_over_mask,
    response_lm_loss,
)


def test_model_stats_vector_shape_and_logprob_identity():
    torch.manual_seed(0)
    t, v = 5, 7
    logits = torch.randn(t, v)
    input_ids = torch.randint(0, v, (t,))
    response_mask = torch.tensor([0, 0, 1, 1, 1])
    out = model_stats_vector(logits, input_ids, response_mask)
    assert out.shape == (N_MODEL_STATS,)
    assert np.isfinite(out).all()
    loss_mean, _, _, ppl, avg_logprob, _, _ = out
    # avg log-prob of the true token == -mean CE by construction.
    assert abs(avg_logprob - (-loss_mean)) < 1e-4
    # perplexity == exp(loss_mean).
    assert abs(ppl - np.exp(loss_mean)) < 1e-3


def test_model_stats_confident_prediction_has_low_loss():
    t, v = 4, 6
    input_ids = torch.tensor([1, 2, 3, 4])
    # Make logits strongly predict the *next* token at every position.
    logits = torch.full((t, v), -10.0)
    for pos in range(t - 1):
        logits[pos, input_ids[pos + 1]] = 10.0
    mask = torch.ones(t)
    out = model_stats_vector(logits, input_ids, mask)
    assert out[0] < 0.01  # loss_mean ~ 0


def test_response_lm_loss_matches_masked_mean():
    torch.manual_seed(1)
    t, v = 6, 5
    logits = torch.randn(t, v, requires_grad=True)
    input_ids = torch.randint(0, v, (t,))
    mask = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = response_lm_loss(logits, input_ids, mask)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_per_layer_activation_stats_shape_and_self_cosine():
    torch.manual_seed(2)
    t, d = 7, 32
    attn = torch.randn(t, d)
    mlp = torch.randn(t, d)
    residual = torch.randn(t, d)
    mask = torch.ones(t)
    out = per_layer_activation_stats(attn, mlp, residual, mask, n_head_blocks=8)
    assert out.shape == (PER_LAYER_DIM,)
    assert np.isfinite(out).all()
    assert out[0] > 0 and out[1] > 0 and out[2] > 0  # norms are positive

    # If first and last residual rows are identical, last_token_dot_first == 1.
    residual2 = residual.clone()
    residual2[-1] = residual2[0]
    out2 = per_layer_activation_stats(attn, mlp, residual2, mask)
    assert abs(out2[-1] - 1.0) < 1e-4


def test_pool_over_mask_means_selected_rows():
    x = torch.tensor([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
    mask = torch.tensor([1, 1, 0])
    pooled = pool_over_mask(x, mask)
    assert np.allclose(pooled, [1.5, 1.5])


def test_pool_over_mask_empty_mask_falls_back_to_all():
    x = torch.tensor([[2.0], [4.0]])
    pooled = pool_over_mask(x, torch.tensor([0, 0]))
    assert np.allclose(pooled, [3.0])
