"""Difficulty / quality 16-d statistics (d_x). See design doc §7.2.

Two parts:
- Cheap text statistics (length, ratio, language guesses, format flags).
- Selector-model statistics (loss, perplexity, entropy, mean log-prob).
The latter is computed via a forward pass against the selector model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pcu_select.types import DifficultyStats, Sample


D_DIFF = 16
LANG_IDS = ("en", "zh", "other")


@dataclass
class DifficultyConfig:
    selector_model: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    batch_size: int = 8
    fp16: bool = True


def quick_text_stats(sample: Sample) -> np.ndarray:
    """Length / ratio / format flags. No model required."""
    out = np.zeros(D_DIFF, dtype=np.float32)
    li = len(sample.instruction)
    lr = len(sample.response)
    out[0] = math.log1p(li)
    out[1] = math.log1p(lr)
    out[2] = math.log1p(li + lr)
    out[3] = lr / max(li, 1)
    # slots 4..10 reserved for model-side stats, filled by ModelStatsExtractor.
    out[11] = float("step" in sample.response.lower() or "let's think" in sample.response.lower())
    out[12] = float("```" in sample.response or "def " in sample.response)
    out[13] = float("?" in sample.instruction)
    # language: extremely simple heuristic; replace with langid for production.
    lang_idx = _guess_lang_idx(sample.instruction + " " + sample.response)
    out[14 + lang_idx] = 1.0  # 14, 15 used; "other" overflows to none -> stays zero
    if lang_idx == 2:
        out[14] = 0.0
        out[15] = 0.0  # "other" represented by all zeros in those slots
    return out


def _guess_lang_idx(text: str) -> int:
    has_cjk = any("一" <= ch <= "鿿" for ch in text[:512])
    if has_cjk:
        return 1
    if all(ord(ch) < 128 for ch in text[:512]):
        return 0
    return 2


class ModelStatsExtractor:
    """Run selector model forward to obtain loss / ppl / entropy / mean logprob.

    Fills slots 4..10 of the difficulty vector. The cheap stats from
    `quick_text_stats` are assumed to be already in place.
    """

    def __init__(self, cfg: DifficultyConfig | None = None):
        self.cfg = cfg or DifficultyConfig()
        self._tokenizer = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.selector_model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.selector_model, torch_dtype="auto"
        ).to(self.cfg.device).eval()

    def extract(self, samples: list[Sample], cheap_vectors: list[np.ndarray]) -> list[DifficultyStats]:
        """For each sample, fill in the model-side stats and return DifficultyStats."""
        self._ensure_model()
        # Implementation: batched forward, compute per-sample response-mask
        # CE loss / mean logprob / token-level entropy.
        # NOTE: detailed batching omitted in skeleton; see docstring contract.
        raise NotImplementedError("Hook into selector model forward; see design doc §7.2.")
