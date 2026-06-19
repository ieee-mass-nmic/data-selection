"""Forward / backward hooks on selector model for site-wise grad capture.

Design doc §5. Hooks are registered on the chosen `(layer, module)` sites and
expose `.activation` (forward output) and `.grad` (backward gradient) for use
by other modules.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, cast

import torch
from torch import nn

from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.types import SiteID


@dataclass
class SiteCapture:
    """Per-site activation + gradient buffer for one forward/backward pass."""

    activation: torch.Tensor | None = None
    grad: torch.Tensor | None = None


class SiteHookManager:
    """Register the forward/backward hooks for a given selector model.

    Implementation notes:
    - We treat the model as a decoder stack with attribute path
      `model.model.layers[l]`. Subclass / config to support other backbones.
    - For each (l, module_name) we hook a sub-module:
        attn_out      -> layer.self_attn (output[0] before residual add)
        mlp_out       -> layer.mlp.forward output
        block_residual-> layer (the whole TransformerBlock output)
    - The forward output is stored detached; the backward grad is stored only
      if `retain_grad=True` was set or if we registered a `.register_hook` on
      the tensor.
    """

    def __init__(self, model: nn.Module, sites: SiteSpace, *, layers_path: str = "model.layers"):
        self.model = model
        self.sites = sites
        self.layers_path = layers_path
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.buffers: dict[SiteID, SiteCapture] = {}

    # -- module path resolution -----------------------------------------
    def _layer(self, l: int) -> nn.Module:
        cur = self.model
        for part in self.layers_path.split("."):
            cur = getattr(cur, part)
        return cast("nn.ModuleList", cur)[l]

    def _submodule_for(self, layer: nn.Module, m: str) -> nn.Module:
        if m == "attn_out":
            return getattr(layer, "self_attn", None) or getattr(layer, "attention")
        if m == "mlp_out":
            return getattr(layer, "mlp", None) or getattr(layer, "feed_forward")
        if m == "block_residual":
            return layer
        raise ValueError(m)

    # -- registration ---------------------------------------------------
    def _make_fwd_hook(self, site: SiteID):
        def hook(_m, _inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            buf = self.buffers.setdefault(site, SiteCapture())
            t.retain_grad()
            buf.activation = t
            # capture the gradient when it becomes available
            t.register_hook(lambda g, _site=site: self._on_grad(_site, g))
            return out
        return hook

    def _on_grad(self, site: SiteID, grad: torch.Tensor) -> None:
        self.buffers[site].grad = grad.detach()

    @contextmanager
    def capture(self) -> Iterator[dict[SiteID, SiteCapture]]:
        """Context manager: registers hooks on entry, removes on exit."""
        try:
            for (l, m) in self.sites.all_sites:
                layer = self._layer(l)
                sub = self._submodule_for(layer, m)
                h = sub.register_forward_hook(self._make_fwd_hook((l, m)))
                self._handles.append(h)
            self.buffers = {}
            yield self.buffers
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
