"""z_p encoding across PEFT families (pure numpy, no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from pcu_select.peft_space.encoder import encode_capacity, encode_peft
from pcu_select.peft_space.site_mask import SiteSpace, alpha_vector
from pcu_select.types import PEFTConfig, PEFTRecipe

SITES = SiteSpace.uniform(32, 8)
Z_P_BASE = 24 * 4 + 16 + 16  # m_p(96) + c_p(16) + r_p(16) = 128


def _cfg(family: str, **kw) -> PEFTConfig:
    defaults = dict(
        peft_id=family,
        family=family,
        target_modules=["q_proj", "v_proj"],
        target_layers=[3, 7, 11, 15, 19, 23, 27, 31],
        recipe=PEFTRecipe(lr=1e-4),
    )
    defaults.update(kw)
    return PEFTConfig(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "cfg",
    [
        _cfg("lora", rank=8, alpha=16),
        _cfg("ia3", target_modules=["q_proj", "v_proj"]),
        _cfg("adapter", target_modules=["mlp"], adapter_bottleneck=64),
        _cfg("prefix", target_modules=["attn"], prefix_len=16),
        _cfg("bitfit", target_modules=["attn"]),
    ],
)
def test_encode_peft_length_is_stable(cfg):
    z = encode_peft(cfg, SITES)
    assert z.shape == (Z_P_BASE,)
    assert z.dtype == np.float32
    assert np.isfinite(z).all()


def test_fingerprint_appends_64_dims():
    cfg = _cfg("lora", rank=8, alpha=16)
    fp = np.ones(64, dtype=np.float32)
    z = encode_peft(cfg, SITES, fingerprint=fp)
    assert z.shape == (Z_P_BASE + 64,)


@pytest.mark.parametrize(
    "family,slot",
    [("lora", 9), ("ia3", 10), ("adapter", 11), ("prefix", 12), ("bitfit", 13)],
)
def test_capacity_family_one_hot(family, slot):
    c = encode_capacity(_cfg(family, rank=8, adapter_bottleneck=64, prefix_len=16))
    assert c[slot] == 1.0


def test_alpha_vector_is_normalized_per_family():
    for family in ("lora", "ia3", "adapter", "prefix"):
        cfg = _cfg(family, rank=8, adapter_bottleneck=64, prefix_len=16)
        v = alpha_vector(cfg, SITES, normalize=True)
        s = float(v.sum())
        # Either no active site (sum 0) or normalized to 1.
        assert abs(s - 1.0) < 1e-5 or abs(s) < 1e-8
