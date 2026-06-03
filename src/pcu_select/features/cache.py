"""Disk-backed feature cache (parquet + numpy). See design doc §14."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pcu_select.types import ActivationSignature, DifficultyStats, SampleFeatures, SemanticEmbedding


@dataclass
class FeatureCachePaths:
    root: Path

    @property
    def parquet(self) -> Path:
        return self.root / "sample_features.parquet"

    @property
    def grad_dir(self) -> Path:
        return self.root / "sample_grad_signature"


class FeatureCache:
    """Append-only feature store.

    Reading is keyed by `sample_id`. Writing is intended to be done in
    full-pool batches; partial updates are supported by re-writing the parquet.
    """

    def __init__(self, root: Path | str):
        self.paths = FeatureCachePaths(Path(root))
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.grad_dir.mkdir(parents=True, exist_ok=True)

    # ---------- semantic / difficulty / activation -----------
    def write_features(self, features: list[SampleFeatures]) -> None:
        rows = []
        for sf in features:
            rows.append({
                "sample_id": sf.sample_id,
                "e_x_instr": sf.e_x.instr.tolist(),
                "e_x_resp": sf.e_x.resp.tolist(),
                "e_x_joint": sf.e_x.joint.tolist(),
                "d_x": sf.d_x.vector.tolist(),
                "a_x": sf.a_x.vector.tolist(),
            })
        df = pd.DataFrame(rows)
        df.to_parquet(self.paths.parquet)

    def read_features(self) -> dict[str, SampleFeatures]:
        df = pd.read_parquet(self.paths.parquet)
        out: dict[str, SampleFeatures] = {}
        for _, row in df.iterrows():
            sf = SampleFeatures(
                sample_id=row["sample_id"],
                e_x=SemanticEmbedding(
                    instr=np.asarray(row["e_x_instr"], dtype=np.float32),
                    resp=np.asarray(row["e_x_resp"], dtype=np.float32),
                    joint=np.asarray(row["e_x_joint"], dtype=np.float32),
                ),
                d_x=DifficultyStats(vector=np.asarray(row["d_x"], dtype=np.float32)),
                a_x=ActivationSignature(vector=np.asarray(row["a_x"], dtype=np.float32)),
            )
            out[sf.sample_id] = sf
        return out

    # ---------- per-site gradient signature -----------
    def grad_path(self, site_id: tuple[int, str]) -> Path:
        layer, module = site_id
        return self.paths.grad_dir / f"site_l{layer:02d}_{module}.npy"

    def write_grad_signature(self, site_id: tuple[int, str], matrix: np.ndarray) -> None:
        """Matrix shape: (N, d_proj). Row order must match `sample_ids.txt`."""
        np.save(self.grad_path(site_id), matrix.astype(np.float32))

    def read_grad_signature(self, site_id: tuple[int, str]) -> np.ndarray:
        return np.load(self.grad_path(site_id), mmap_mode="r")

    def write_sample_id_index(self, sample_ids: list[str]) -> None:
        (self.paths.grad_dir / "sample_ids.txt").write_text("\n".join(sample_ids))

    def read_sample_id_index(self) -> list[str]:
        return (self.paths.grad_dir / "sample_ids.txt").read_text().splitlines()
