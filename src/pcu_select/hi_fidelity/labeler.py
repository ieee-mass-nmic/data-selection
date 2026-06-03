"""Orchestrates high-fidelity labeling: short updates + RankNorm aggregation.

Design doc §10.3. Produces `HiFidelityLabel` records ready to feed scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pcu_select.hi_fidelity.anchors import AnchorRegistry
from pcu_select.hi_fidelity.sampler import TripleSample
from pcu_select.hi_fidelity.short_update import ShortUpdateConfig, run_short_update
from pcu_select.types import HiFidelityLabel, PEFTConfig, Sample, ValidationSketch


@dataclass
class LabelerConfig:
    horizons: tuple[int, ...] = (1, 4)
    horizon_weights: tuple[float, ...] = (0.4, 0.6)
    seed: int = 0


class HiFidelityLabeler:
    def __init__(
        self,
        *,
        anchors: AnchorRegistry,
        samples_by_id: dict[str, Sample],
        pefts_by_id: dict[str, PEFTConfig],
        sketches_by_id: dict[str, ValidationSketch],
        cfg: LabelerConfig | None = None,
    ):
        self.anchors = anchors
        self.samples_by_id = samples_by_id
        self.pefts_by_id = pefts_by_id
        self.sketches_by_id = sketches_by_id
        self.cfg = cfg or LabelerConfig()

    def _compute_raw_deltas(self, triples: Iterable[TripleSample]) -> pd.DataFrame:
        rows = []
        for tri in triples:
            sample = self.samples_by_id[tri.sample_id]
            peft = self.pefts_by_id[tri.peft_id]
            sketch = self.sketches_by_id[tri.task_id]
            for a_idx, anchor in enumerate(self.anchors.all()):
                for h in self.cfg.horizons:
                    delta = run_short_update(
                        anchor_checkpoint=anchor.checkpoint_path,
                        peft=peft,
                        sample=sample,
                        sketch=sketch,
                        cfg=ShortUpdateConfig(horizon=h, seed=self.cfg.seed),
                    )
                    rows.append({
                        "sample_id": tri.sample_id,
                        "peft_id": tri.peft_id,
                        "task_id": tri.task_id,
                        "anchor_idx": a_idx,
                        "horizon": h,
                        "delta_raw": delta,
                    })
        return pd.DataFrame(rows)

    def _rank_norm_within_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        def _rn(g: pd.DataFrame) -> pd.DataFrame:
            ranks = g["delta_raw"].rank(method="average")
            g = g.copy()
            g["delta_norm"] = (ranks - 1) / max(len(g) - 1, 1)
            return g
        return df.groupby(["peft_id", "task_id", "anchor_idx", "horizon"], group_keys=False).apply(_rn)

    def _aggregate(self, df: pd.DataFrame) -> list[HiFidelityLabel]:
        # mean over anchors, weighted average over horizons, std for σ_est
        h_weights = dict(zip(self.cfg.horizons, self.cfg.horizon_weights))
        out: list[HiFidelityLabel] = []
        for (sid, pid, tid), g in df.groupby(["sample_id", "peft_id", "task_id"]):
            # average anchors per horizon first
            per_h = g.groupby("horizon")["delta_norm"].agg(["mean", "std"]).reset_index()
            mean_weighted = 0.0
            std_acc = []
            for _, row in per_h.iterrows():
                w = h_weights.get(int(row["horizon"]), 0.0)
                mean_weighted += w * row["mean"]
                if not np.isnan(row["std"]):
                    std_acc.append(row["std"])
            sigma_est = float(np.mean(std_acc)) if std_acc else 0.0
            out.append(HiFidelityLabel(
                sample_id=sid, peft_id=pid, task_id=tid,
                u_hi=float(mean_weighted - 0.0),  # β·std penalty handled in scorer if desired
                horizon=int(self.cfg.horizons[-1]),  # representative
                anchor_idx=-1,
                seed=self.cfg.seed,
                delta_raw=float(g["delta_raw"].mean()),
                sigma_est=sigma_est,
            ))
        return out

    def run(self, triples: Iterable[TripleSample]) -> list[HiFidelityLabel]:
        raw = self._compute_raw_deltas(triples)
        normed = self._rank_norm_within_bucket(raw)
        return self._aggregate(normed)

    @staticmethod
    def save_labels(labels: list[HiFidelityLabel], path: Path | str) -> None:
        rows = [vars(l) for l in labels]
        pd.DataFrame(rows).to_parquet(Path(path))
