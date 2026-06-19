"""Low-fidelity utility u^lo math (pure numpy, no torch)."""

from __future__ import annotations

import numpy as np

from pcu_select.features.cache import FeatureCache
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.proxy.lo_fidelity import LoFidelityScorer, aggregate_task_grad
from pcu_select.types import PEFTConfig, PEFTRecipe


def test_aggregate_task_grad_returns_unit_rows():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(10, 6, 4)).astype(np.float32)  # (V, |Ω|, d_proj)
    out = aggregate_task_grad(g)
    assert out.shape == (6, 4)
    norms = np.linalg.norm(out, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_lo_fidelity_score_matches_alpha_weighted_cosine(tmp_path):
    # 2 layers x 3 modules = 6 sites; d_proj = 4.
    sites = SiteSpace.uniform(n_layers_total=4, k=2)
    assert len(sites) == 6
    d_proj = 4
    n = 3

    cache = FeatureCache(tmp_path)
    unit = np.zeros((n, d_proj), dtype=np.float32)
    unit[:, 0] = 1.0  # every sample's grad == [1,0,0,0] at every site
    for s in sites.all_sites:
        cache.write_grad_signature(s, unit)
    cache.write_sample_id_index(["a", "b", "c"])

    # Task grad identical unit vector -> cosine == 1 at every site.
    g_t = np.zeros((len(sites), d_proj), dtype=np.float32)
    g_t[:, 0] = 1.0

    peft = PEFTConfig(
        peft_id="lora_t",
        family="lora",
        target_modules=["q_proj", "v_proj"],
        target_layers=list(sites.layer_indices),  # so attn_out sites are active
        rank=8,
        alpha=16,
        recipe=PEFTRecipe(lr=3e-4),
    )

    scorer = LoFidelityScorer(sites, cache)
    res = scorer.score(peft=peft, g_t_per_site=g_t)

    assert res.sample_ids == ["a", "b", "c"]
    # cos == 1 everywhere and normalized alpha sums to 1 over active sites,
    # so u^lo == sum(alpha) == 1 for every sample.
    assert np.allclose(res.u_lo, 1.0, atol=1e-5)


def test_lo_fidelity_orthogonal_task_grad_is_zero(tmp_path):
    sites = SiteSpace.uniform(n_layers_total=4, k=2)
    d_proj = 4
    n = 2
    cache = FeatureCache(tmp_path)
    g_x = np.zeros((n, d_proj), dtype=np.float32)
    g_x[:, 0] = 1.0
    for s in sites.all_sites:
        cache.write_grad_signature(s, g_x)
    cache.write_sample_id_index(["a", "b"])

    g_t = np.zeros((len(sites), d_proj), dtype=np.float32)
    g_t[:, 1] = 1.0  # orthogonal to g_x -> cosine 0

    peft = PEFTConfig(
        peft_id="lora_t",
        family="lora",
        target_modules=["q_proj", "v_proj"],
        target_layers=list(sites.layer_indices),
        rank=8,
        recipe=PEFTRecipe(lr=3e-4),
    )
    res = LoFidelityScorer(sites, cache).score(peft=peft, g_t_per_site=g_t)
    assert np.allclose(res.u_lo, 0.0, atol=1e-6)
