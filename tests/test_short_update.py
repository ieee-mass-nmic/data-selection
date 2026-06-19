"""Short-update protocol against a tiny in-memory Llama.

Training a fresh LoRA on a sample should lower the loss on a sketch that
contains that sample → Δ > 0. Skipped without transformers.
"""

from __future__ import annotations

import math

import pytest
import torch

from pcu_select.features.selector_runner import SelectorRunner
from pcu_select.hi_fidelity.short_update import ShortUpdateConfig, ShortUpdater, run_short_update
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.types import PEFTConfig, PEFTRecipe, Sample, ValidationSketch


class _CharTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    vocab_size = 64

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [(ord(c) % 60) + 2 for c in text][:24]
        if add_special_tokens:
            ids = [2] + ids
        if not ids:
            ids = [2]
        return {"input_ids": ids}


def _tiny_model(n_layers: int = 2, hidden: int = 32):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=_CharTokenizer.vocab_size, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def _lora(lr: float = 0.1) -> PEFTConfig:
    return PEFTConfig(
        peft_id="lora_test", family="lora",
        target_modules=["q_proj", "v_proj"], target_layers=[],
        rank=4, alpha=8, recipe=PEFTRecipe(optimizer="adamw", lr=lr),
    )


def test_delta_positive_when_training_on_sketch_sample():
    model = _tiny_model()
    updater = ShortUpdater(model, _CharTokenizer(), device="cpu", max_len=64)
    sample = Sample(sample_id="x", instruction="what is two plus two", response="four four four")
    sketch = ValidationSketch(task_id="t", samples=[sample], sketch_seed=0)

    delta = updater.delta(peft=_lora(lr=0.05), sample=sample, sketch=sketch, horizon=25, seed=0)
    assert math.isfinite(delta)
    # Training on the very sample we evaluate should reduce its loss.
    assert delta > 0.0


def test_evaluate_sketch_loss_empty_is_zero():
    model = _tiny_model()
    updater = ShortUpdater(model, _CharTokenizer(), device="cpu", max_len=64)
    empty = ValidationSketch(task_id="t", samples=[], sketch_seed=0)
    assert updater.evaluate_sketch_loss(empty) == 0.0


def test_delta_restores_model_after_run():
    model = _tiny_model()
    updater = ShortUpdater(model, _CharTokenizer(), device="cpu", max_len=64)
    ids = torch.tensor([[2, 5, 9, 13]])
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()

    sample = Sample(sample_id="x", instruction="hello there", response="general kenobi")
    sketch = ValidationSketch(task_id="t", samples=[sample], sketch_seed=0)
    updater.delta(peft=_lora(), sample=sample, sketch=sketch, horizon=2, seed=0)

    with torch.no_grad():
        after = model(input_ids=ids).logits.clone()
    # The adapter is detached after .delta → base model unchanged.
    assert torch.allclose(before, after, atol=1e-5)


def test_run_short_update_convenience_wrapper():
    model = _tiny_model()
    sample = Sample(sample_id="x", instruction="solve equation", response="x equals four")
    sketch = ValidationSketch(task_id="t", samples=[sample], sketch_seed=0)
    delta = run_short_update(
        model=model, tokenizer=_CharTokenizer(), peft=_lora(lr=0.2),
        sample=sample, sketch=sketch,
        cfg=ShortUpdateConfig(horizon=3, device="cpu", max_len=64),
    )
    assert math.isfinite(delta)


def test_runner_and_updater_share_model_class():
    # Smoke: the runner's SiteSpace machinery and the updater coexist (no global
    # hook leakage between feature extraction and short-update).
    sites = SiteSpace.uniform(n_layers_total=2, k=2)
    assert len(sites) == 6
    assert SelectorRunner(sites) is not None
