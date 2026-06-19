"""Integration test for SelectorRunner against a tiny in-memory Llama.

No network / no checkpoint download: we build a small LlamaForCausalLM from a
config and inject it (plus a trivial char tokenizer) into the runner. Skipped
when transformers is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pcu_select.features.selector_runner import SelectorRunner, SelectorRunnerConfig
from pcu_select.features.tokenization import encode_response_lm
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.types import Sample


class _CharTokenizer:
    """Minimal stand-in for an HF tokenizer (char → id)."""

    eos_token_id = 1
    pad_token_id = 0
    vocab_size = 64

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [(ord(c) % 60) + 2 for c in text][:24]
        if add_special_tokens:
            ids = [2] + ids  # BOS
        if not ids:
            ids = [2]
        return {"input_ids": ids}


def _tiny_runner(n_layers: int = 4, hidden: int = 32):
    transformers = pytest.importorskip("transformers")
    LlamaConfig = transformers.LlamaConfig
    LlamaForCausalLM = transformers.LlamaForCausalLM
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=_CharTokenizer.vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    sites = SiteSpace.uniform(n_layers_total=n_layers, k=n_layers)
    runner = SelectorRunner(sites, SelectorRunnerConfig(device="cpu", dtype="float32"))
    # Inject without going through from_pretrained (no download).
    runner._model = model
    runner._tok = _CharTokenizer()
    runner._hidden_size = hidden
    runner._num_layers = n_layers
    return runner, sites, hidden


def test_encode_response_lm_masks_instruction():
    tok = _CharTokenizer()
    enc = encode_response_lm(tok, "abc", "de")
    n = len(enc["input_ids"])
    assert len(enc["response_mask"]) == n
    assert len(enc["attention_mask"]) == n
    # instruction (BOS + 3 chars) masked 0; response (2 chars) + EOS masked 1.
    assert enc["response_mask"][0] == 0
    assert enc["response_mask"][-1] == 1
    assert sum(enc["response_mask"]) == 3  # "d", "e", EOS


def test_runner_process_full_pass():
    runner, sites, hidden = _tiny_runner()
    sample = Sample(sample_id="x", instruction="what is two plus two", response="four")
    res = runner.process(sample, want_grads=True, want_activations=True)

    assert res.model_stats.shape == (7,)
    assert np.isfinite(res.model_stats).all()

    assert res.activation is not None
    assert res.activation.shape == (len(sites.layer_indices) * 8,)
    assert np.isfinite(res.activation).all()

    assert res.site_grads is not None
    assert set(res.site_grads.keys()) == set(sites.all_sites)
    for g in res.site_grads.values():
        assert g.shape == (hidden,)
        assert np.isfinite(g).all()


def test_runner_stats_only_skips_hooks():
    runner, _, _ = _tiny_runner()
    sample = Sample(sample_id="y", instruction="hello", response="world")
    res = runner.process(sample, want_grads=False, want_activations=False)
    assert res.model_stats.shape == (7,)
    assert res.activation is None
    assert res.site_grads is None


def test_runner_grads_differ_across_distinct_samples():
    runner, sites, _ = _tiny_runner()
    a = runner.process(Sample(sample_id="a", instruction="solve the equation", response="x=4"),
                       want_grads=True, want_activations=False)
    b = runner.process(Sample(sample_id="b", instruction="translate to french", response="bonjour"),
                       want_grads=True, want_activations=False)
    site = sites.all_sites[0]
    # Different inputs should yield different gradient signatures.
    assert not np.allclose(a.site_grads[site], b.site_grads[site])
