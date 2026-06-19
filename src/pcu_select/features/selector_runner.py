"""Selector-model runner: one forward (+optional backward) per sample.

Produces, from a single pass, the three model-side artifacts the design needs:
  - model-side difficulty stats `d_x[4:11]`   (design doc §7.2)
  - per-layer activation signature `a_x`        (design doc §7.3)
  - per-site pooled gradient vectors `g_pool^ω` (design doc §5.1)

Sharing one forward+backward across all three is exactly the design intent
(§7.3 note). Gradient capture reuses `proxy.hooks.SiteHookManager`; the raw
pooled gradients are projected to `d_proj` by the caller (offline pipeline)
via `proxy.projection`.

Heavy imports (`torch` autograd graph, `transformers`) are kept lazy so that
importing this module stays cheap; only `process()` touches the model.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pcu_select.features.stats import (
    model_stats_vector,
    per_layer_activation_stats,
    pool_over_mask,
    response_lm_loss,
)
from pcu_select.features.tokenization import encode_response_lm
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.types import Sample, SiteID


@dataclass
class SelectorRunnerConfig:
    selector_model: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    dtype: str = "auto"  # "auto" | "float32" | "float16" | "bfloat16"
    layers_path: str = "model.layers"
    max_len: int = 1024
    n_head_blocks: int = 8


@dataclass
class SampleResult:
    model_stats: np.ndarray  # (N_MODEL_STATS,)
    activation: np.ndarray | None  # (n_layers * PER_LAYER_DIM,) or None
    site_grads: dict[SiteID, np.ndarray] | None  # raw pooled (d_model,) per site


class SelectorRunner:
    """Loads a causal-LM selector model and extracts per-sample features.

    Construct with the `SiteSpace` whose layers/modules are hooked. For
    stats-only use (no activations, no grads) an empty `SiteSpace` is fine.
    """

    def __init__(self, sites: SiteSpace, cfg: SelectorRunnerConfig | None = None):
        self.sites = sites
        self.cfg = cfg or SelectorRunnerConfig()
        self._tok: Any = None
        self._model: Any = None
        self._hidden_size: int | None = None
        self._num_layers: int | None = None

    # -- model lifecycle ------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.cfg.selector_model)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        torch_dtype = "auto" if self.cfg.dtype == "auto" else getattr(torch, self.cfg.dtype)
        model: Any = AutoModelForCausalLM.from_pretrained(self.cfg.selector_model, torch_dtype=torch_dtype)  # type: ignore[arg-type]
        self._model = model.to(self.cfg.device).eval()
        self._hidden_size = int(self._model.config.hidden_size)
        self._num_layers = int(self._model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        self._ensure_model()
        assert self._hidden_size is not None
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        self._ensure_model()
        assert self._num_layers is not None
        return self._num_layers

    @property
    def tokenizer(self):
        self._ensure_model()
        return self._tok

    # -- per-sample extraction -----------------------------------------
    def process(
        self, sample: Sample, *, want_grads: bool = True, want_activations: bool = True
    ) -> SampleResult:
        from pcu_select.proxy.hooks import SiteHookManager

        self._ensure_model()
        assert self._model is not None and self._hidden_size is not None
        dev = self.cfg.device

        enc = encode_response_lm(
            self._tok, sample.instruction, sample.response, max_len=self.cfg.max_len
        )
        input_ids = torch.tensor([enc["input_ids"]], device=dev)
        attn = torch.tensor([enc["attention_mask"]], device=dev)
        rmask = torch.tensor(enc["response_mask"], device=dev)

        use_hooks = want_grads or want_activations
        mgr = (
            SiteHookManager(self._model, self.sites, layers_path=self.cfg.layers_path)
            if use_hooks and len(self.sites) > 0
            else None
        )
        grad_ctx = torch.enable_grad() if want_grads else torch.no_grad()
        capture_ctx: Any = mgr.capture() if mgr is not None else nullcontext({})

        with grad_ctx, capture_ctx as buffers:
            out = self._model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[0]
            model_stats = model_stats_vector(logits, input_ids[0], rmask)
            if want_grads:
                loss = response_lm_loss(logits, input_ids[0], rmask)
                self._model.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()

        a_vec: np.ndarray | None = None
        if want_activations and mgr is not None:
            parts = []
            for layer in self.sites.layer_indices:
                ao = buffers[(layer, "attn_out")].activation[0]
                mo = buffers[(layer, "mlp_out")].activation[0]
                ro = buffers[(layer, "block_residual")].activation[0]
                parts.append(
                    per_layer_activation_stats(
                        ao, mo, ro, rmask, n_head_blocks=self.cfg.n_head_blocks
                    )
                )
            a_vec = np.concatenate(parts).astype(np.float32)

        site_grads: dict[SiteID, np.ndarray] | None = None
        if want_grads and mgr is not None:
            site_grads = {}
            for site in self.sites.all_sites:
                g = buffers[site].grad
                if g is None:
                    site_grads[site] = np.zeros(self._hidden_size, dtype=np.float32)
                else:
                    site_grads[site] = pool_over_mask(g[0], rmask).astype(np.float32)

        return SampleResult(model_stats=model_stats, activation=a_vec, site_grads=site_grads)
