"""Batch scoring for deployment. See design doc §11.1, §15."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from pcu_select.scorer.model import PCUScorer, ScorerConfig


@dataclass
class InferenceConfig:
    batch_size: int = 4096
    device: str = "cuda"
    bf16: bool = True


class ScorerInference:
    def __init__(self, ckpt: Path | str, model_cfg: ScorerConfig | None = None,
                 cfg: InferenceConfig | None = None):
        self.cfg = cfg or InferenceConfig()
        self.model = PCUScorer(model_cfg)
        state = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state)
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        if self.cfg.bf16 and self.device.type == "cuda":
            self.model = self.model.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def score(self, z_x: np.ndarray, z_p: np.ndarray, z_t: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """z_x: (N, d_x). z_p / z_t may be (1, d) or (N, d)."""
        n = z_x.shape[0]
        mus = np.empty(n, dtype=np.float32)
        sigmas = np.empty(n, dtype=np.float32)
        zp_t = torch.as_tensor(z_p, device=self.device)
        zt_t = torch.as_tensor(z_t, device=self.device)
        for s, e in self._batches(n):
            zx_t = torch.as_tensor(z_x[s:e], device=self.device)
            mu, sigma = self.model(zx_t, zp_t, zt_t)
            mus[s:e] = mu.float().cpu().numpy()
            sigmas[s:e] = sigma.float().cpu().numpy()
        return mus, sigmas

    def _batches(self, n: int) -> Iterator[tuple[int, int]]:
        for s in range(0, n, self.cfg.batch_size):
            yield s, min(n, s + self.cfg.batch_size)
