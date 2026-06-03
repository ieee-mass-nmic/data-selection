"""Per-layer activation signature (a_x). See design doc §7.3.

For each layer in `L_sig` we collect 8 stats:
  ||attn_out||_2, ||mlp_out||_2, ||residual||_2,
  attn_entropy, attn_head_norm_var,
  mlp_activation_norm,
  hidden_token_var, last_token_dot_first
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.types import ActivationSignature, Sample

PER_LAYER_DIM = 8


@dataclass
class ActivationSignatureConfig:
    selector_model: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    fp16: bool = True


class ActivationSignatureExtractor:
    """Forward-only signature collector.

    The same hooks used in `proxy/hooks.py` for site-wise gradient capture can
    feed this extractor; in practice we run them together to share one forward.
    """

    def __init__(self, sites: SiteSpace, cfg: ActivationSignatureConfig | None = None):
        self.sites = sites
        self.cfg = cfg or ActivationSignatureConfig()

    def extract(self, samples: list[Sample]) -> list[ActivationSignature]:
        # Skeleton signature: iterate samples, hook target layers, compute stats.
        n_layers = len(self.sites.layer_indices)
        out: list[ActivationSignature] = []
        for _ in samples:
            vec = np.zeros(n_layers * PER_LAYER_DIM, dtype=np.float32)
            out.append(ActivationSignature(vector=vec))
        # Real implementation:
        #   for each batch:
        #     register forward hooks on the chosen layers
        #     do model(**inputs) without grad
        #     read hidden states, compute the 8 stats per layer, write into vec
        raise NotImplementedError("See design doc §7.3 for the 8 per-layer stats.")
