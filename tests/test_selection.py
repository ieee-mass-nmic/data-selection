"""Selection / quota logic — pure numpy/sklearn, no torch."""

from __future__ import annotations

import numpy as np

from pcu_select.selection.adaptive_quota import (
    QuotaConfig,
    allocate_cluster_budgets,
    compute_cluster_values,
    pick_top_in_clusters,
)
from pcu_select.selection.cluster import ClusterConfig, cluster_samples


def test_compute_cluster_values_top_pct():
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cid = np.array([0, 0, 0, 1, 1])
    v = compute_cluster_values(q_scores=q, cluster_ids=cid, top_pct=0.5)
    # cluster 0: top-2 of (1,2,3) -> mean(2,3) = 2.5
    # cluster 1: top-1 of (4,5) -> 5.0
    assert abs(v[0] - 2.5) < 1e-6
    assert abs(v[1] - 5.0) < 1e-6


def test_allocate_budgets_respect_sizes_and_sum():
    values = {0: 1.0, 1: 0.5}
    sizes = {0: 100, 1: 50}
    b = allocate_cluster_budgets(cluster_values=values, cluster_sizes=sizes,
                                  total_budget=30, alpha=0.5)
    assert sum(b.values()) == 30
    assert b[0] <= sizes[0]
    assert b[1] <= sizes[1]


def test_pick_top_in_clusters_returns_correct_ids():
    sample_ids = ["a", "b", "c", "d"]
    q = np.array([0.1, 0.9, 0.5, 0.3])
    cid = np.array([0, 0, 1, 1])
    budgets = {0: 1, 1: 1}
    picked = pick_top_in_clusters(
        sample_ids=sample_ids, q_scores=q, cluster_ids=cid, cluster_budgets=budgets
    )
    assert set(picked) == {"b", "c"}


def test_cluster_samples_returns_consistent_shapes():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 32)).astype(np.float32)
    ids = [f"s{i}" for i in range(200)]
    asg = cluster_samples(sample_ids=ids, joint_embeddings=emb,
                          cfg=ClusterConfig(k=10, n_init=1))
    assert asg.cluster_ids.shape == (200,)
    assert asg.centers.shape[0] == 10
