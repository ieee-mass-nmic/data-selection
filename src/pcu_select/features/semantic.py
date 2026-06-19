"""Semantic embedding extraction (e_x). See design doc §7.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from pcu_select.types import Sample, SemanticEmbedding


@dataclass
class SemanticEncoderConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_seq_length: int = 512
    device: str = "cuda"
    batch_size: int = 64


class SemanticEncoder:
    """Wraps a sentence-transformer-style encoder.

    Outputs three vectors per sample: instruction-only, response-only, joint.
    """

    def __init__(self, cfg: SemanticEncoderConfig | None = None):
        self.cfg = cfg or SemanticEncoderConfig()
        self._model: Any = None  # lazy SentenceTransformer

    def _ensure_model(self):
        if self._model is not None:
            return
        # Lazy import keeps this module importable without sentence-transformers.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
        self._model.max_seq_length = self.cfg.max_seq_length

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        emb = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def encode_samples(self, samples: Iterable[Sample]) -> list[SemanticEmbedding]:
        samples = list(samples)
        instr_texts = [s.instruction for s in samples]
        resp_texts = [s.response for s in samples]
        joint_texts = [s.instruction + "\n" + s.response for s in samples]
        instr_emb = self.encode_batch(instr_texts)
        resp_emb = self.encode_batch(resp_texts)
        joint_emb = self.encode_batch(joint_texts)
        return [
            SemanticEmbedding(instr=instr_emb[i], resp=resp_emb[i], joint=joint_emb[i])
            for i in range(len(samples))
        ]
