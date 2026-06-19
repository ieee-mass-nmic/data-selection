"""Per-layer activation signature (a_x). See design doc §7.3.

For each layer in `L_sig` we collect 8 stats:
  ||attn_out||_2, ||mlp_out||_2, ||residual||_2,
  attn_entropy, attn_head_norm_var,
  mlp_activation_norm,
  hidden_token_var, last_token_dot_first
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        self._runner: Any = None

    def _ensure_runner(self) -> None:
        if self._runner is not None:
            return
        from pcu_select.features.selector_runner import SelectorRunner, SelectorRunnerConfig

        self._runner = SelectorRunner(
            self.sites,
            SelectorRunnerConfig(
                selector_model=self.cfg.selector_model,
                device=self.cfg.device,
                dtype="float16" if self.cfg.fp16 else "auto",
            ),
        )

    def extract(self, samples: list[Sample]) -> list[ActivationSignature]:
        """Forward-only: hook the chosen layers, compute the 8 per-layer stats."""
        self._ensure_runner()
        assert self._runner is not None
        out: list[ActivationSignature] = []
        for sample in samples:
            res = self._runner.process(sample, want_grads=False, want_activations=True)
            assert res.activation is not None
            out.append(ActivationSignature(vector=res.activation))
        return out
