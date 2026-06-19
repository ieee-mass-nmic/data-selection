"""Difficulty / quality 16-d statistics (d_x). See design doc §7.2.

Two parts:
- Cheap text statistics (length, ratio, language guesses, format flags).
- Selector-model statistics (loss, perplexity, entropy, mean log-prob).
The latter is computed via a forward pass against the selector model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pcu_select.features.stats import N_MODEL_STATS
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
    `quick_text_stats` are assumed to be already in place. Delegates the actual
    forward to a stats-only `SelectorRunner` (no hooks, no backward).
    """

    def __init__(self, cfg: DifficultyConfig | None = None):
        self.cfg = cfg or DifficultyConfig()
        self._runner: Any = None

    def _ensure_runner(self) -> None:
        if self._runner is not None:
            return
        from pcu_select.features.selector_runner import SelectorRunner, SelectorRunnerConfig
        from pcu_select.peft_space.site_mask import SiteSpace

        self._runner = SelectorRunner(
            SiteSpace(layer_indices=(), modules=()),  # stats-only: no hooked sites
            SelectorRunnerConfig(
                selector_model=self.cfg.selector_model,
                device=self.cfg.device,
                dtype="float16" if self.cfg.fp16 else "auto",
            ),
        )

    def extract(
        self, samples: list[Sample], cheap_vectors: list[np.ndarray]
    ) -> list[DifficultyStats]:
        """For each sample, fill in the model-side stats and return DifficultyStats."""
        self._ensure_runner()
        assert self._runner is not None
        out: list[DifficultyStats] = []
        for sample, cheap in zip(samples, cheap_vectors):
            res = self._runner.process(sample, want_grads=False, want_activations=False)
            vec = np.asarray(cheap, dtype=np.float32).copy()
            vec[4 : 4 + N_MODEL_STATS] = res.model_stats
            out.append(DifficultyStats(vector=vec))
        return out
