"""Low-fidelity utility u^lo. See design doc §5.5.

Given cached site-wise gradient signatures `g_x^ω` (per sample) and
`g_t^ω` (per task), and the PEFT-conditioned α̃_p^ω weights:

    u^lo(x, p, t) = Σ_ω α̃_p^ω · cos(g_x^ω, g_t^ω)

Because both g's are pre-normalized, the cosine reduces to a dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcu_select.features.cache import FeatureCache
from pcu_select.peft_space.site_mask import SiteSpace, alpha_vector
from pcu_select.types import PEFTConfig


@dataclass
class LoFidelityResult:
    sample_ids: list[str]
    u_lo: np.ndarray  # (N,)


class LoFidelityScorer:
    def __init__(self, sites: SiteSpace, cache: FeatureCache):
        self.sites = sites
        self.cache = cache
        self._sample_ids = cache.read_sample_id_index()
        # Stack per-site grad matrices once on first use.
        self._grads: np.ndarray | None = None  # (N, |Ω|, d_proj)

    def _ensure_grads(self) -> None:
        if self._grads is not None:
            return
        mats = [self.cache.read_grad_signature(s) for s in self.sites.all_sites]
        # mats[i] shape: (N, d_proj)
        self._grads = np.stack(mats, axis=1).astype(np.float32)

    def score(self, *, peft: PEFTConfig, g_t_per_site: np.ndarray) -> LoFidelityResult:
        """g_t_per_site: (|Ω|, d_proj) — already normalized per row."""
        self._ensure_grads()
        assert self._grads is not None  # populated by _ensure_grads
        alpha = alpha_vector(peft, self.sites, normalize=True)  # (|Ω|,)
        # cos(g_x^ω, g_t^ω) = g_x^ω · g_t^ω because both unit-length.
        # einsum: 'noi,oi->no' → (N, |Ω|)
        cos = np.einsum("noi,oi->no", self._grads, g_t_per_site, optimize=True)
        u_lo = (cos * alpha[None, :]).sum(axis=-1)  # (N,)
        return LoFidelityResult(sample_ids=list(self._sample_ids), u_lo=u_lo)


def aggregate_task_grad(per_sample_grads: np.ndarray) -> np.ndarray:
    """Pool sketch sample grads into per-site task grad. Input: (V, |Ω|, d_proj)."""
    avg = per_sample_grads.mean(axis=0)  # (|Ω|, d_proj)
    norms = np.linalg.norm(avg, axis=-1, keepdims=True)
    return avg / np.maximum(norms, 1e-8)
