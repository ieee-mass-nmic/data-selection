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
    min_cluster_size: int | None = None,
) -> dict[int, int]:
    """b_k = round(B · (v_k^+)^α · |C_k|^{1-α} / Z) with overflow redistribution.

    If `min_cluster_size` is set (design §12.2/§12.3), every cluster is first
    guaranteed a floor of `min(min_cluster_size, |C_k|)` so diverse-but-low-value
    clusters are not starved; the weighted allocation then distributes whatever
    budget remains. Floors are never trimmed below by the overflow logic. If the
    floors alone exceed the budget, they are reduced greedily from the
    lowest-weight clusters so the total still equals `total_budget`.
    """
    keys = list(cluster_values.keys())
    raw_values = np.array([cluster_values[k] for k in keys], dtype=np.float64)
    if raw_values.size:
        # Scores can be globally negative because q = mu - lambda*sigma is not
        # calibrated to be positive. Shift values before exponentiating so only
        # relative cluster quality matters; if all clusters tie, fall back to
        # size-only allocation.
        shifted = raw_values - min(float(raw_values.min()), 0.0)
        if np.all(shifted <= 1e-12):
            value_term = np.ones_like(shifted)
        elif alpha == 0:
            value_term = np.ones_like(shifted)
        else:
            value_term = np.maximum(shifted, 0.0) ** alpha
    else:
        value_term = np.zeros((0,), dtype=np.float64)
    sizes_arr = np.array([cluster_sizes[k] for k in keys], dtype=np.float64)
    weights = value_term * np.maximum(sizes_arr, 0.0) ** (1 - alpha)

    floor = {k: 0 for k in keys}
    if min_cluster_size:
        floor = {k: min(int(min_cluster_size), cluster_sizes[k]) for k in keys}
        over_floor = sum(floor.values()) - total_budget
        if over_floor > 0:
            # too many floors to fit the budget; shed from least-valuable clusters
            for i in np.argsort(weights):
                if over_floor <= 0:
                    break
                k = keys[i]
                cut = min(floor[k], over_floor)
                floor[k] -= cut
                over_floor -= cut

    z = max(weights.sum(), 1e-12)
    remaining_budget = max(total_budget - sum(floor.values()), 0)
    raw = remaining_budget * weights / z
    # round + cap by cluster size; redistribute leftover greedily by weight
    b = {k: min(floor[k] + int(round(raw[i])), cluster_sizes[k]) for i, k in enumerate(keys)}
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
        # over-allocated; trim from clusters with smallest weight, never below floor
        order = np.argsort(weights)
        for i in order:
            if remaining >= 0:
                break
            k = keys[i]
            cut = min(b[k] - floor[k], -remaining)
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
