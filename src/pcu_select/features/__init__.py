"""Feature extractors.

`SemanticEncoder` / `ModelStatsExtractor` / `ActivationSignatureExtractor` pull
in heavy optional deps (sentence-transformers / transformers), so they are
loaded lazily via PEP 562 `__getattr__`. Lightweight pieces like
`FeatureCache` (pure pandas/numpy) import eagerly and stay dependency-light —
this lets the proxy / pipeline code touch the cache without paying for the
model stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pcu_select.features.cache import FeatureCache

# public name -> submodule providing it (imported on first access only)
_LAZY = {
    "ActivationSignatureConfig": "pcu_select.features.activation_signature",
    "ActivationSignatureExtractor": "pcu_select.features.activation_signature",
    "DifficultyConfig": "pcu_select.features.difficulty",
    "ModelStatsExtractor": "pcu_select.features.difficulty",
    "quick_text_stats": "pcu_select.features.difficulty",
    "SemanticEncoder": "pcu_select.features.semantic",
    "SemanticEncoderConfig": "pcu_select.features.semantic",
}

if TYPE_CHECKING:  # keep static imports for type checkers / IDEs
    from pcu_select.features.activation_signature import (
        ActivationSignatureConfig,
        ActivationSignatureExtractor,
    )
    from pcu_select.features.difficulty import (
        DifficultyConfig,
        ModelStatsExtractor,
        quick_text_stats,
    )
    from pcu_select.features.semantic import SemanticEncoder, SemanticEncoderConfig


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, name)


__all__ = [
    "ActivationSignatureConfig",
    "ActivationSignatureExtractor",
    "DifficultyConfig",
    "FeatureCache",
    "ModelStatsExtractor",
    "SemanticEncoder",
    "SemanticEncoderConfig",
    "quick_text_stats",
]
