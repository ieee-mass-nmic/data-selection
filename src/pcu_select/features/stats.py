"""Pure stat / pooling helpers for selector-model features.

Everything here operates on plain torch tensors and returns numpy, requiring
no model load or network, so the math is unit-testable in isolation. The
model orchestration that produces the input tensors lives in
`features.selector_runner`.

See design doc §7.2 (difficulty stats), §7.3 (activation signature),
§5.1/§5.2 (gradient time-pooling).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Number of model-side difficulty stats (fills d_x slots 4..10).
N_MODEL_STATS = 7
# Per-layer activation-signature width (design doc §7.3).
PER_LAYER_DIM = 8


def _shifted_token_stats(
    logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(response)-token CE, log-prob and entropy under next-token shift.

    logits: (T, V), input_ids: (T,), response_mask: (T,). Returns three 1-D
    tensors over the selected response positions (may be empty if T < 2).
    """
    if logits.shape[0] < 2:
        empty = logits.new_zeros((0,))
        return empty, empty, empty
    shift_logits = logits[:-1]
    shift_labels = input_ids[1:].long()
    shift_mask = response_mask[1:].to(torch.bool)
    if int(shift_mask.sum()) == 0:
        shift_mask = torch.ones_like(shift_mask)
    logp = F.log_softmax(shift_logits.float(), dim=-1)
    tok_lp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    tok_ce = -tok_lp
    tok_entropy = -(logp.exp() * logp).sum(dim=-1)
    return tok_ce[shift_mask], tok_lp[shift_mask], tok_entropy[shift_mask]


def response_lm_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Mean response-token cross-entropy — the scalar used for backward."""
    tok_ce, _, _ = _shifted_token_stats(logits, input_ids, response_mask)
    if tok_ce.numel() == 0:
        return logits.float().sum() * 0.0  # keep graph, zero gradient
    return tok_ce.mean()


def model_stats_vector(
    logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor
) -> np.ndarray:
    """The 7 model-side difficulty stats (design doc §7.2 slots 4..10):

    [loss_mean, loss_std, loss_max, perplexity, avg_logprob,
     entropy_mean, entropy_max].
    """
    tok_ce, tok_lp, tok_ent = _shifted_token_stats(logits, input_ids, response_mask)
    if tok_ce.numel() == 0:
        return np.zeros(N_MODEL_STATS, dtype=np.float32)
    ce = tok_ce.detach()
    lp = tok_lp.detach()
    ent = tok_ent.detach()
    loss_mean = ce.mean()
    loss_std = ce.std(unbiased=False) if ce.numel() > 1 else torch.zeros((), device=ce.device)
    loss_max = ce.max()
    ppl = torch.exp(torch.clamp(loss_mean, max=20.0))
    avg_lp = lp.mean()
    ent_mean = ent.mean()
    ent_max = ent.max()
    vec = torch.stack([loss_mean, loss_std, loss_max, ppl, avg_lp, ent_mean, ent_max])
    return vec.float().cpu().numpy()


def per_layer_activation_stats(
    attn_out: torch.Tensor,
    mlp_out: torch.Tensor,
    residual: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    n_head_blocks: int = 8,
) -> np.ndarray:
    """The 8 per-layer activation-signature stats (design doc §7.3).

    attn_out / mlp_out / residual: (T, d). Stats are computed over response
    tokens (falling back to all tokens if the mask is empty). The two
    attention-weight-derived stats in the design are realized here as
    activation-norm proxies, which need no attention-probability capture and
    stay backbone-agnostic.
    """
    sel = response_mask.to(torch.bool)
    if int(sel.sum()) == 0:
        sel = torch.ones_like(response_mask, dtype=torch.bool)
    a = attn_out.detach().float()
    m = mlp_out.detach().float()
    r = residual.detach().float()
    a_sel, m_sel, r_sel = a[sel], m[sel], r[sel]

    norm_attn = a_sel.norm(dim=-1).mean()
    norm_mlp = m_sel.norm(dim=-1).mean()
    norm_res = r_sel.norm(dim=-1).mean()

    # "attn_entropy" proxy: spread of attn-output energy across response tokens.
    token_norms = a_sel.norm(dim=-1)
    p = token_norms / token_norms.sum().clamp_min(1e-8)
    attn_entropy = -(p * (p + 1e-12).log()).sum()

    # "attn_head_norm_var" proxy: variance of per-head-block norms (d split into
    # n_head_blocks equal chunks), averaged over tokens.
    d = a_sel.shape[-1]
    nb = max(1, min(n_head_blocks, d))
    use = (d // nb) * nb
    blocks = a_sel[:, :use].reshape(a_sel.shape[0], nb, use // nb)
    attn_head_norm_var = blocks.norm(dim=-1).var(dim=-1, unbiased=False).mean()

    mlp_activation_norm = m_sel.abs().mean()
    if r_sel.shape[0] > 1:
        hidden_token_var = r_sel.var(dim=0, unbiased=False).mean()
    else:
        hidden_token_var = torch.zeros((), device=r.device)

    # last-vs-first residual cosine (uses absolute positions, not the mask).
    first, last = r[0], r[-1]
    denom = (first.norm() * last.norm()).clamp_min(1e-8)
    last_token_dot_first = (first @ last) / denom

    vec = torch.stack([
        norm_attn, norm_mlp, norm_res,
        attn_entropy, attn_head_norm_var,
        mlp_activation_norm, hidden_token_var, last_token_dot_first,
    ])
    return vec.float().cpu().numpy()


def pool_over_mask(tensor_td: torch.Tensor, response_mask: torch.Tensor) -> np.ndarray:
    """Mean-pool (T, d) over response tokens → (d,). Design doc §5.2 default."""
    sel = response_mask.to(torch.bool)
    if int(sel.sum()) == 0:
        sel = torch.ones_like(response_mask, dtype=torch.bool)
    return tensor_td.detach().float()[sel].mean(dim=0).cpu().numpy()
