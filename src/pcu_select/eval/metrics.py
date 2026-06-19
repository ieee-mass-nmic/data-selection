"""Ranking and set-overlap metrics for selection quality (design §1.6 tier 2).

Pure-numpy implementations (no scipy) so the package keeps a light dependency
footprint. All ranking metrics compare a *predicted* score vector against a
*ground-truth* utility vector (the high-fidelity u^hi on held-out triples).
"""

from __future__ import annotations

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (like scipy.stats.rankdata, method='average')."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # resolve ties to their average rank
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0  # average 1-based rank within each tie group
    return avg[inv]


def spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    """Spearman rank correlation ρ."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    if len(pred) < 2:
        return float("nan")
    rp, rt = _rankdata(pred), _rankdata(truth)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    denom = np.sqrt((rp**2).sum() * (rt**2).sum())
    return float((rp * rt).sum() / denom) if denom > 0 else float("nan")


def kendall_tau(pred: np.ndarray, truth: np.ndarray) -> float:
    """Kendall's τ-b via O(n²) concordance count (fine for held-out eval sets)."""
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    n = len(pred)
    if n < 2:
        return float("nan")
    conc = disc = 0
    tp = tt = 0
    for i in range(n - 1):
        dp = pred[i + 1 :] - pred[i]
        dt = truth[i + 1 :] - truth[i]
        s = np.sign(dp) * np.sign(dt)
        conc += int((s > 0).sum())
        disc += int((s < 0).sum())
        tp += int((dp == 0).sum())
        tt += int((dt == 0).sum())
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - tp) * (n0 - tt))
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def ndcg_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    """NDCG@k with gains = (truth rescaled to ≥ 0). Measures whether the
    highest-predicted items are also the highest-utility ones."""
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    n = len(pred)
    if n == 0:
        return float("nan")
    k = min(k, n)
    gains = truth - truth.min()  # non-negative relevance
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    top_pred = np.argsort(-pred)[:k]
    dcg = float((gains[top_pred] * disc).sum())
    top_ideal = np.argsort(-gains)[:k]
    idcg = float((gains[top_ideal] * disc).sum())
    return dcg / idcg if idcg > 0 else 0.0


def topk_hit_rate(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of the true top-k items recovered by the predicted top-k."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    n = len(pred)
    if n == 0:
        return float("nan")
    k = min(k, n)
    pred_top = set(np.argsort(-pred)[:k].tolist())
    true_top = set(np.argsort(-truth)[:k].tolist())
    return len(pred_top & true_top) / k


def pairwise_acc(pred: np.ndarray, truth: np.ndarray, *, n_pairs: int = 5000,
                 seed: int = 0) -> float:
    """Fraction of sampled pairs whose predicted order matches the true order."""
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    n = len(pred)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    same = np.sign(pred[i] - pred[j]) == np.sign(truth[i] - truth[j])
    valid = truth[i] != truth[j]
    return float(same[valid].mean()) if valid.any() else float("nan")


def jaccard(a: list[str] | set[str], b: list[str] | set[str]) -> float:
    """Jaccard overlap of two selected-id sets (design E4-a)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
