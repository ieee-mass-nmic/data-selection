"""Random projection matrices Φ_ω. See design doc §5.3.

Each site gets its own deterministic Gaussian random projection
Φ_ω ∈ R^{d_proj × d_model}. We persist them so that future runs use the
exact same matrices for samples and tasks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.types import SiteID


@dataclass
class ProjectionConfig:
    d_model: int = 4096
    d_proj: int = 256
    global_seed: int = 0


class ProjectionStore:
    """Lazy generator + on-disk persistence."""

    def __init__(self, cfg: ProjectionConfig, root: Path | str):
        self.cfg = cfg
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, site: SiteID) -> Path:
        return self.root / f"site_l{site[0]:02d}_{site[1]}_d{self.cfg.d_proj}.npy"

    def _seed_for(self, site: SiteID) -> int:
        h = hashlib.sha256(f"{site}-{self.cfg.global_seed}".encode()).digest()
        return int.from_bytes(h[:4], "big")

    def get(self, site: SiteID) -> np.ndarray:
        p = self._path(site)
        if p.exists():
            return np.load(p)
        rng = np.random.default_rng(self._seed_for(site))
        # Johnson-Lindenstrauss scaling.
        phi = rng.normal(scale=1.0 / np.sqrt(self.cfg.d_proj),
                         size=(self.cfg.d_proj, self.cfg.d_model)).astype(np.float32)
        np.save(p, phi)
        return phi

    def ensure_all(self, sites: SiteSpace) -> None:
        for s in sites.all_sites:
            _ = self.get(s)


def project(grad_pooled: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """grad_pooled: (B, d_model). phi: (d_proj, d_model). Returns (B, d_proj)."""
    out = grad_pooled @ phi.T
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norms, 1e-8)
