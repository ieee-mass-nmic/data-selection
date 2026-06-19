"""Native PEFT backend: attach/remove, zero-init identity, state-dict I/O.

Uses a tiny in-memory Llama (no download); skipped without transformers.
"""

from __future__ import annotations

import pytest
import torch

from pcu_select.hi_fidelity.native_peft import attach_peft
from pcu_select.types import PEFTConfig, PEFTRecipe


def _tiny_model(n_layers: int = 2, hidden: int = 32):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def _peft(family: str, **kw) -> PEFTConfig:
    return PEFTConfig(
        peft_id=f"{family}_x", family=family,
        target_modules=kw.pop("target_modules", ["q_proj", "v_proj"]),
        target_layers=kw.pop("target_layers", []),
        recipe=PEFTRecipe(), **kw,
    )


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.clone()


@pytest.mark.parametrize("family,kw", [
    ("lora", {"rank": 4, "alpha": 4}),
    ("ia3", {}),
    ("bitfit", {}),
    ("adapter", {"adapter_bottleneck": 8}),
])
def test_attach_is_identity_then_remove_restores(family, kw):
    model = _tiny_model()
    ids = torch.tensor([[2, 5, 9, 13, 4]])
    before = _logits(model, ids)

    handle = attach_peft(model, _peft(family, **kw), seed=0)
    # All four families are zero/identity-initialized → Δ₀ = 0.
    after_attach = _logits(model, ids)
    assert torch.allclose(before, after_attach, atol=1e-5)
    assert handle.num_trainable() > 0

    handle.remove()
    after_remove = _logits(model, ids)
    assert torch.allclose(before, after_remove, atol=1e-6)
    # remove() is idempotent
    handle.remove()


def test_target_layer_filtering():
    model = _tiny_model(n_layers=2)
    h_all = attach_peft(model, _peft("lora", rank=2, target_layers=[]), seed=0)
    assert len(h_all._sites) == 4  # 2 layers × {q_proj, v_proj}
    h_all.remove()

    h_one = attach_peft(model, _peft("lora", rank=2, target_layers=[0]), seed=0)
    assert len(h_one._sites) == 2
    h_one.remove()


def test_state_dict_round_trip():
    model = _tiny_model()
    ids = torch.tensor([[2, 7, 3, 11]])
    handle = attach_peft(model, _peft("lora", rank=4, alpha=4), seed=0)
    # Perturb the adapter so it is no longer identity.
    for p in handle.parameters():
        with torch.no_grad():
            p.add_(0.3)
    trained = _logits(model, ids)
    sd = handle.state_dict()
    handle.remove()

    # Fresh attach + load → reproduces the perturbed forward.
    handle2 = attach_peft(model, _peft("lora", rank=4, alpha=4), seed=123)
    handle2.load_state_dict(sd)
    assert torch.allclose(_logits(model, ids), trained, atol=1e-5)
    handle2.remove()


def test_unsupported_family_raises():
    model = _tiny_model()
    with pytest.raises(NotImplementedError):
        attach_peft(model, _peft("prefix", prefix_len=4), seed=0)


def test_no_match_raises():
    model = _tiny_model()
    with pytest.raises(ValueError):
        attach_peft(model, _peft("lora", rank=2, target_modules=["nonexistent_proj"]), seed=0)
