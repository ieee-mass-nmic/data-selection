"""Three-phase active sampling of (x, p, t) triples. See design doc §10.4."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pcu_select.types import PEFTConfig, TaskConfig


@dataclass
class TripleSample:
    sample_id: str
    peft_id: str
    task_id: str
    phase: int  # 1 / 2 / 3


@dataclass
class PhaseBudget:
    phase1: int
    phase2: int
    phase3: int


def split_budget(q_hi_total: int, ratios: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> PhaseBudget:
    n1 = int(q_hi_total * ratios[0])
    n2 = int(q_hi_total * ratios[1])
    n3 = q_hi_total - n1 - n2
    return PhaseBudget(n1, n2, n3)


def phase1_stratified(
    *,
    sample_ids: list[str],
    sample_cluster: dict[str, int],
    pefts: Sequence[PEFTConfig],
    tasks: Sequence[TaskConfig],
    budget: int,
    seed: int = 0,
) -> list[TripleSample]:
    """Stratified covering across sample-cluster × PEFT family × capacity bucket."""
    rng = random.Random(seed)
    # Strata key = (cluster_id, peft_family, capacity_bucket, task_id)
    keys: dict[tuple, list[tuple[str, str, str]]] = {}
    for sid in sample_ids:
        cluster = sample_cluster.get(sid, 0)
        for p in pefts:
            fam = p.family
            bucket = _capacity_bucket(p)
            for t in tasks:
                keys.setdefault((cluster, fam, bucket, t.task_id), []).append((sid, p.peft_id, t.task_id))
    if not keys:
        return []
    strata = list(keys.values())
    rng.shuffle(strata)
    for triples in strata:
        rng.shuffle(triples)
    out: list[TripleSample] = []
    while len(out) < budget:
        progressed = False
        for triples in strata:
            if not triples or len(out) >= budget:
                continue
            a, b, c = triples.pop()
            out.append(TripleSample(sample_id=a, peft_id=b, task_id=c, phase=1))
            progressed = True
        if not progressed:
            break
    return out


def phase2_uncertainty(
    *,
    candidate_triples: list[TripleSample],
    sigma_hat: np.ndarray,
    u_lo_hat: np.ndarray,
    budget: int,
    gamma: float = 0.5,
) -> list[TripleSample]:
    """Pick triples maximizing σ̂ · (1 + γ · ReLU(û^lo)). Skipped triples already
    queried in phase 1 should be filtered out by the caller."""
    score = sigma_hat * (1.0 + gamma * np.maximum(u_lo_hat, 0.0))
    order = np.argsort(-score)
    return [TripleSample(**candidate_triples[i].__dict__) for i in order[:budget]]


def phase3_boundary(
    *,
    candidate_triples: list[TripleSample],
    scorer_rank: np.ndarray,
    true_rank: np.ndarray,
    budget: int,
) -> list[TripleSample]:
    """Pick triples with largest scorer-vs-truth rank gap in the top-k region."""
    gap = np.abs(scorer_rank - true_rank)
    # weight toward top of the list (smaller true_rank ranks more important)
    weighted = gap / np.maximum(1.0, true_rank)
    order = np.argsort(-weighted)
    return [TripleSample(**candidate_triples[i].__dict__) for i in order[:budget]]


def _capacity_bucket(p: PEFTConfig) -> str:
    rank = p.rank or p.adapter_bottleneck or p.prefix_len or 8
    if rank < 8:
        return "xs"
    if rank < 16:
        return "s"
    if rank < 32:
        return "m"
    return "l"
