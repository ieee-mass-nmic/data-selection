"""High-fidelity short PEFT-only update protocol. See design doc §10.2.

Given a frozen anchor model, attach fresh PEFT parameters, take `h` gradient
steps on a single sample `x`, and measure how much the validation-sketch loss
`L_V` dropped:

    Δ = L_V(θ_a) − L_V(θ_a + Adaptʰ_p(x))

A positive Δ means training on `x` helped the task — exactly the utility signal
the labeler turns into `u^hi`. The base weights stay frozen, so backward only
flows through the small adapter (the design's 50–90 % saving).

`ShortUpdater` holds one loaded anchor model and is reused across many triples
(loading a 7B model per triple would be absurd); the module-level
`run_short_update` is a thin one-shot convenience for tests / small jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pcu_select.features.stats import response_lm_loss
from pcu_select.features.tokenization import encode_response_lm
from pcu_select.hi_fidelity.native_peft import PeftHandle, attach_peft
from pcu_select.types import PEFTConfig, PEFTRecipe, Sample, ValidationSketch


@dataclass
class ShortUpdateConfig:
    horizon: int = 1
    device: str = "cuda"
    dtype: str = "bfloat16"  # informational; model is assumed already cast
    seed: int = 0
    max_len: int = 1024
    grad_clip: float = 1.0


def _build_optimizer(params: list[torch.nn.Parameter], recipe: PEFTRecipe) -> torch.optim.Optimizer:
    """Map a PEFTRecipe to a torch optimizer with blank state (design §10.2)."""
    if recipe.optimizer == "sgd":
        return torch.optim.SGD(params, lr=recipe.lr, weight_decay=recipe.weight_decay)
    # adamw default; adafactor falls back to adamw (no native impl in torch core).
    return torch.optim.AdamW(params, lr=recipe.lr, weight_decay=recipe.weight_decay)


def init_peft_module(model: torch.nn.Module, peft: PEFTConfig, *, seed: int = 0) -> PeftHandle:
    """Attach fresh PEFT params per `peft` to `model`. Returns a removable handle."""
    return attach_peft(model, peft, seed=seed)


class ShortUpdater:
    """Runs the short-update protocol against one frozen anchor model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        device: str = "cuda",
        max_len: int = 1024,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.max_len = max_len
        self.grad_clip = grad_clip
        # Freeze the backbone: backward will only touch attached adapters.
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    # -- encoding -------------------------------------------------------
    def _encode(self, sample: Sample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = encode_response_lm(self.tok, sample.instruction, sample.response, max_len=self.max_len)
        ids = torch.tensor([enc["input_ids"]], device=self.device)
        attn = torch.tensor([enc["attention_mask"]], device=self.device)
        rmask = torch.tensor(enc["response_mask"], device=self.device)
        return ids, attn, rmask

    def _sample_loss(self, sample: Sample) -> torch.Tensor:
        ids, attn, rmask = self._encode(sample)
        out = self.model(input_ids=ids, attention_mask=attn)
        return response_lm_loss(out.logits[0], ids[0], rmask)

    def evaluate_sketch_loss(self, sketch: ValidationSketch) -> float:
        """Forward-only mean response-LM loss over the sketch samples."""
        if not sketch.samples:
            return 0.0
        total = 0.0
        with torch.no_grad():
            for s in sketch.samples:
                total += float(self._sample_loss(s))
        return total / len(sketch.samples)

    # -- protocol -------------------------------------------------------
    def delta(
        self, *, peft: PEFTConfig, sample: Sample, sketch: ValidationSketch,
        horizon: int, seed: int = 0,
    ) -> float:
        """Attach PEFT, take `horizon` steps on `sample`, return L_before − L_after."""
        handle = init_peft_module(self.model, peft, seed=seed)
        try:
            params = handle.parameters()
            l_before = self.evaluate_sketch_loss(sketch)
            opt = _build_optimizer(params, peft.recipe)
            ids, attn, rmask = self._encode(sample)
            with torch.enable_grad():
                for _ in range(max(1, horizon)):
                    opt.zero_grad(set_to_none=True)
                    out = self.model(input_ids=ids, attention_mask=attn)
                    loss = response_lm_loss(out.logits[0], ids[0], rmask)
                    if not loss.requires_grad:
                        break
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                    opt.step()
            l_after = self.evaluate_sketch_loss(sketch)
            return float(l_before - l_after)
        finally:
            handle.remove()


def run_short_update(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    peft: PEFTConfig,
    sample: Sample,
    sketch: ValidationSketch,
    cfg: ShortUpdateConfig,
) -> float:
    """One-shot convenience wrapper around `ShortUpdater` (tests / small jobs).

    For batch labeling, build a single `ShortUpdater` per anchor and call
    `.delta(...)` repeatedly instead of reloading the model each time.
    """
    updater = ShortUpdater(
        model, tokenizer, device=cfg.device, max_len=cfg.max_len, grad_clip=cfg.grad_clip
    )
    return updater.delta(peft=peft, sample=sample, sketch=sketch, horizon=cfg.horizon, seed=cfg.seed)
