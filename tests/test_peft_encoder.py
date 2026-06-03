"""Sanity tests on pure-Python modules (no torch GPU required)."""

from __future__ import annotations

import numpy as np

from pcu_select.peft_space.encoder import encode_peft
from pcu_select.peft_space.site_mask import SiteSpace, alpha_vector, normalize_alpha, site_mask_of
from pcu_select.types import PEFTConfig, PEFTRecipe


def _lora_cfg() -> PEFTConfig:
    return PEFTConfig(
        peft_id="lora8",
        family="lora",
        target_modules=["q_proj", "v_proj"],
        target_layers=[3, 7, 11, 15, 19, 23, 27, 31],
        rank=8,
        alpha=16,
        recipe=PEFTRecipe(lr=3e-4),
    )


def test_site_space_uniform():
    s = SiteSpace.uniform(n_layers_total=32, k=8)
    assert len(s.layer_indices) == 8
    assert len(s) == 24


def test_site_mask_nonempty_for_matching_lora():
    s = SiteSpace.uniform(32, 8)
    cfg = _lora_cfg()
    mask = site_mask_of(cfg, s)
    # at least the attn_out sites for the targeted layers should be > 0
    pos = sum(v > 0 for v in mask.values())
    assert pos >= 1


def test_normalize_alpha_sums_to_one():
    s = SiteSpace.uniform(32, 8)
    cfg = _lora_cfg()
    raw = site_mask_of(cfg, s)
    if sum(raw.values()) > 0:
        norm = normalize_alpha(raw)
        assert abs(sum(norm.values()) - 1.0) < 1e-5


def test_alpha_vector_shape():
    s = SiteSpace.uniform(32, 8)
    cfg = _lora_cfg()
    v = alpha_vector(cfg, s)
    assert v.shape == (24,)
    assert v.dtype == np.float32


def test_encode_peft_shape():
    s = SiteSpace.uniform(32, 8)
    cfg = _lora_cfg()
    z = encode_peft(cfg, s)
    # 96 (mask) + 16 (cap) + 16 (recipe) = 128 without fingerprint
    assert z.shape[0] == 24 * 4 + 16 + 16
