"""Selection / quota logic — pure numpy/sklearn, no torch."""

from __future__ import annotations

import numpy as np

from pcu_select.baselines.selectors import BaselineInputs, score_baseline, select_baseline
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.selection.adaptive_quota import (
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


def test_allocate_budgets_min_cluster_size_floor():
    # Low-value but populous cluster 1 would otherwise be starved; the floor
    # guarantees it at least `min_cluster_size` slots.
    values = {0: 10.0, 1: 0.01}
    sizes = {0: 100, 1: 100}
    b = allocate_cluster_budgets(cluster_values=values, cluster_sizes=sizes,
                                 total_budget=40, alpha=0.6, min_cluster_size=5)
    assert sum(b.values()) == 40
    assert b[1] >= 5
    assert b[0] <= sizes[0] and b[1] <= sizes[1]


def test_allocate_budgets_floor_capped_by_cluster_size():
    # Floor must never exceed the cluster's own size.
    values = {0: 1.0, 1: 1.0}
    sizes = {0: 50, 1: 3}
    b = allocate_cluster_budgets(cluster_values=values, cluster_sizes=sizes,
                                 total_budget=20, alpha=0.5, min_cluster_size=10)
    assert b[1] <= 3
    assert sum(b.values()) == 20


def test_allocate_budgets_floors_exceeding_budget_are_shed():
    # When floors alone exceed the budget, total must still equal the budget.
    values = {0: 5.0, 1: 0.1, 2: 0.1}
    sizes = {0: 100, 1: 100, 2: 100}
    b = allocate_cluster_budgets(cluster_values=values, cluster_sizes=sizes,
                                 total_budget=10, alpha=0.6, min_cluster_size=8)
    assert sum(b.values()) == 10
    assert all(v >= 0 for v in b.values())


def test_allocate_budgets_preserves_order_when_values_are_negative():
    values = {0: -5.0, 1: -1.0, 2: -0.1}
    sizes = {0: 10, 1: 10, 2: 10}
    b = allocate_cluster_budgets(cluster_values=values, cluster_sizes=sizes,
                                 total_budget=6, alpha=0.6)
    assert sum(b.values()) == 6
    assert b[2] >= b[1] >= b[0]


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


def test_cluster_samples_caps_default_k_to_small_pool():
    emb = np.eye(3, dtype=np.float32)
    ids = ["a", "b", "c"]
    asg = cluster_samples(sample_ids=ids, joint_embeddings=emb)
    assert asg.cluster_ids.shape == (3,)
    assert asg.centers.shape[0] == 3


def test_rds_plus_dense_score_matches_selection_rule():
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(8)]
    joint = rng.normal(size=(8, 4)).astype(np.float32)
    inp = BaselineInputs(
        cache=None,  # type: ignore[arg-type]
        sites=SiteSpace.uniform(4, 2),
        sample_ids=ids,
        d_x=np.zeros((8, 16), dtype=np.float32),
        joint=joint,
        task_query_joint=rng.normal(size=4).astype(np.float32),
    )
    selected = select_baseline("rds_plus", inp, budget=3)
    scores = score_baseline("rds_plus", inp)
    assert scores is not None
    expected = [ids[i] for i in np.argsort(-scores)[:3]]
    assert selected == expected
