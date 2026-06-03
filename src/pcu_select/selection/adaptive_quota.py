"""Adaptive cluster budget allocation. See design doc §12.3, §12.4."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class QuotaConfig:
    alpha: float = 0.6
    top_pct_for_cluster_value: float = 0.10
    min_cluster_size: int | None = None


def compute_cluster_values(
    *,
    q_scores: np.ndarray,
    cluster_ids: np.ndarray,
    top_pct: float,
) -> dict[int, float]:
    """v_k = mean of top-pct fraction of q in cluster k."""
    out: dict[int, float] = {}
    for k in np.unique(cluster_ids):
        mask = cluster_ids == k
        qs = q_scores[mask]
        if qs.size == 0:
            out[int(k)] = 0.0
            continue
        m = max(1, int(math.ceil(qs.size * top_pct)))
        top_m = np.partition(qs, -m)[-m:]
        out[int(k)] = float(np.mean(top_m))
    return out


def allocate_cluster_budgets(
    *,
    cluster_values: dict[int, float],
    cluster_sizes: dict[int, int],
    total_budget: int,
    alpha: float,
) -> dict[int, int]:
    """b_k = round(B · (v_k^+)^α · |C_k|^{1-α} / Z) with overflow redistribution."""
    keys = list(cluster_values.keys())
    weights = np.array([
        max(cluster_values[k], 0.0) ** alpha * (cluster_sizes[k]) ** (1 - alpha)
        for k in keys
    ], dtype=np.float64)
    z = max(weights.sum(), 1e-12)
    raw = total_budget * weights / z
    # round + cap by cluster size; redistribute leftover greedily by weight
    b = {k: min(int(round(raw[i])), cluster_sizes[k]) for i, k in enumerate(keys)}
    remaining = total_budget - sum(b.values())
    if remaining > 0:
        order = np.argsort(-weights)
        for i in order:
            if remaining <= 0:
                break
            k = keys[i]
            slack = cluster_sizes[k] - b[k]
            take = min(slack, remaining)
            b[k] += take
            remaining -= take
    elif remaining < 0:
        # over-allocated; trim from clusters with smallest weight
        order = np.argsort(weights)
        for i in order:
            if remaining >= 0:
                break
            k = keys[i]
            cut = min(b[k], -remaining)
            b[k] -= cut
            remaining += cut
    return b


def pick_top_in_clusters(
    *,
    sample_ids: list[str],
    q_scores: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_budgets: dict[int, int],
) -> list[str]:
    """Within each cluster k, pick top b_k by q. Return list of sample_ids."""
    picked: list[str] = []
    sample_arr = np.array(sample_ids)
    for k, b in cluster_budgets.items():
        if b <= 0:
            continue
        mask = cluster_ids == k
        qs = q_scores[mask]
        ids = sample_arr[mask]
        if qs.size <= b:
            picked.extend(ids.tolist())
            continue
        order = np.argsort(-qs)[:b]
        picked.extend(ids[order].tolist())
    return picked
