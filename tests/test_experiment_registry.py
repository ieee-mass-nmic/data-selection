"""Experiment registry contract tests."""

from __future__ import annotations

from typing import get_args

from pcu_select.experiments import MODELS, PEFT_REGISTRY, resolve_peft
from pcu_select.types import PEFTFamily


VALID_FAMILIES = set(get_args(PEFTFamily))


def test_registry_materializes_for_all_models() -> None:
    for model_name, model_spec in MODELS.items():
        for peft_name, spec in PEFT_REGISTRY.items():
            cfg = resolve_peft(peft_name, model_name)
            cfg_again = resolve_peft(peft_name, model_name)

            assert cfg.peft_id
            assert cfg.peft_id == cfg_again.peft_id
            assert cfg.family == spec.family
            assert cfg.family in VALID_FAMILIES
            assert cfg.target_modules
            assert cfg.target_layers
            assert all(0 <= layer < model_spec.n_layers for layer in cfg.target_layers)
            assert cfg.extra["spec_name"] == peft_name
            assert cfg.extra["group"] == spec.group


def test_registry_rejects_unknown_names() -> None:
    try:
        resolve_peft("missing-peft", "llama2-7b")
    except KeyError as exc:
        assert "unknown PEFT spec" in str(exc)
    else:
        raise AssertionError("resolve_peft should reject unknown PEFT names")

    try:
        resolve_peft("L-r8-qv", "missing-model")
    except KeyError as exc:
        assert "unknown model" in str(exc)
    else:
        raise AssertionError("resolve_peft should reject unknown model names")
