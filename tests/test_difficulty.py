"""quick_text_stats — cheap, model-free difficulty features."""

from __future__ import annotations

import numpy as np

from pcu_select.features.difficulty import D_DIFF, quick_text_stats
from pcu_select.types import Sample


def _stats(instruction: str, response: str) -> np.ndarray:
    return quick_text_stats(Sample(sample_id="x", instruction=instruction, response=response))


def test_quick_text_stats_shape():
    out = _stats("hello", "world")
    assert out.shape == (D_DIFF,)
    assert np.isfinite(out).all()


def test_language_one_hot_english():
    out = _stats("what is the answer?", "the answer is 42")
    assert out[14] == 1.0  # en
    assert out[15] == 0.0  # zh


def test_language_one_hot_chinese():
    out = _stats("问题是什么", "答案是四十二")
    assert out[15] == 1.0  # zh
    assert out[14] == 0.0  # en


def test_other_language_does_not_raise_and_is_all_zero():
    # Regression: "other"-language text (non-ASCII, non-CJK) used to index
    # out[16] and raise IndexError. It must now stay both-zero in slots 14/15.
    for instr, resp in [
        ("Привет, как дела", "Хорошо, спасибо"),   # Cyrillic
        ("مرحبا كيف حالك", "بخير شكرا"),             # Arabic
        ("こんにちは", "はい"),                      # Japanese hiragana (no kanji)
        ("안녕하세요 반갑습니다", "네 반갑습니다"),  # Korean Hangul
    ]:
        out = _stats(instr, resp)
        assert out.shape == (D_DIFF,)
        assert out[14] == 0.0
        assert out[15] == 0.0
