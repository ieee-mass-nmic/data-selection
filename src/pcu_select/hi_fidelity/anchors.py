"""Anchor checkpoint management. See design doc §10.1.

Two anchors:
  θ_base : pretrained backbone, untouched.
  θ_warm : θ_base + a base-config LoRA (rank=8, attn-only) trained for 200 steps
           on a 1k-sample slice of the meta-pool — a "lightly fine-tuned"
           representative.

Both anchors are frozen; PEFT params for the short-update protocol start from
them. The warm adapter is stored as an adapter-only state dict (the small set
of LoRA weights), not a full model checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pcu_select.hi_fidelity.native_peft import attach_peft
from pcu_select.types import PEFTConfig, PEFTRecipe, Sample


@dataclass
class AnchorSpec:
    anchor_id: str  # e.g. "base", "warm"
    checkpoint_path: Path
    note: str = ""


class AnchorRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._anchors: dict[str, AnchorSpec] = {}

    def register(self, anchor: AnchorSpec) -> None:
        self._anchors[anchor.anchor_id] = anchor

    def get(self, anchor_id: str) -> AnchorSpec:
        return self._anchors[anchor_id]

    def all(self) -> list[AnchorSpec]:
        return list(self._anchors.values())


def warm_lora_config(rank: int = 8) -> PEFTConfig:
    """The standardized warm-up adapter: rank-8 LoRA on attention q/v, all layers."""
    return PEFTConfig(
        peft_id="warm_lora",
        family="lora",
        target_modules=["q_proj", "v_proj"],
        target_layers=[],  # empty == all layers
        rank=rank,
        alpha=rank,
        recipe=PEFTRecipe(optimizer="adamw", lr=1e-4),
    )


def train_lora_warmup(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[Sample],
    *,
    steps: int = 200,
    rank: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    max_len: int = 1024,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Train a base-config LoRA on `samples` for `steps`; return its state dict.

    Cycles through the (≈1k-sample) slice, one sample per step. The model's base
    weights are frozen — only the LoRA adapter learns. Factored out of
    `build_warm_anchor` so it is testable with a tiny in-memory model.
    """
    from pcu_select.features.stats import response_lm_loss
    from pcu_select.features.tokenization import encode_response_lm

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    peft = warm_lora_config(rank)
    peft.recipe.lr = lr
    handle = attach_peft(model, peft, seed=seed)
    params = handle.parameters()
    opt = torch.optim.AdamW(params, lr=lr)

    n = len(samples)
    with torch.enable_grad():
        for step in range(max(0, steps)):
            sample = samples[step % n] if n else None
            if sample is None:
                break
            enc = encode_response_lm(tokenizer, sample.instruction, sample.response, max_len=max_len)
            ids = torch.tensor([enc["input_ids"]], device=device)
            attn = torch.tensor([enc["attention_mask"]], device=device)
            rmask = torch.tensor(enc["response_mask"], device=device)
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=ids, attention_mask=attn)
            loss = response_lm_loss(out.logits[0], ids[0], rmask)
            if not loss.requires_grad:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    sd = handle.state_dict()
    handle.remove()
    return sd


def build_warm_anchor(
    *,
    base_model_path: str,
    meta_pool_slice_path: Path,
    out_path: Path,
    steps: int = 200,
    rank: int = 8,
    device: str = "cuda",
    max_len: int = 1024,
    seed: int = 0,
    model: Any = None,
    tokenizer: Any = None,
) -> AnchorSpec:
    """Run the standardized warm-up training and persist the LoRA adapter.

    Loads the base model + tokenizer (unless injected, for tests), reads the
    meta-pool slice, trains the warm LoRA, and saves the adapter-only state dict
    to `out_path`. Downstream loads it via `load_anchor_model`.
    """
    from pcu_select.data import JsonlPool

    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        loaded: Any = AutoModelForCausalLM.from_pretrained(base_model_path)
        model = loaded.to(device)

    samples = list(JsonlPool.from_jsonl(meta_pool_slice_path))
    sd = train_lora_warmup(
        model, tokenizer, samples,
        steps=steps, rank=rank, device=device, max_len=max_len, seed=seed,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    return AnchorSpec(
        anchor_id="warm",
        checkpoint_path=out_path,
        note=f"rank-{rank} attn-only LoRA, {steps} steps on {len(samples)} samples",
    )


def load_anchor_model(
    spec: AnchorSpec,
    *,
    base_model_path: str,
    device: str = "cuda",
    rank: int = 8,
    model: Any = None,
    tokenizer: Any = None,
) -> tuple[torch.nn.Module, Any]:
    """Materialize an anchor: base backbone, plus the warm LoRA if applicable.

    For the warm anchor the LoRA adapter is attached and the saved weights are
    loaded, then *merged* conceptually by leaving the adapter in place (frozen).
    `ShortUpdater` will freeze everything and stack its own fresh PEFT on top.
    """
    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        loaded: Any = AutoModelForCausalLM.from_pretrained(base_model_path)
        model = loaded.to(device)

    if spec.anchor_id != "base" and Path(spec.checkpoint_path).exists():
        peft = warm_lora_config(rank)
        handle = attach_peft(model, peft, seed=0)
        handle.load_state_dict(torch.load(spec.checkpoint_path, map_location=device))
    return model, tokenizer
