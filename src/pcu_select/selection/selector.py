"""End-to-end selector that wires scorer → clustering → adaptive quota."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pcu_select.selection.adaptive_quota import (
    QuotaConfig,
    allocate_cluster_budgets,
    compute_cluster_values,
    pick_top_in_clusters,
)
from pcu_select.selection.cluster import ClusterAssignment, ClusterConfig, cluster_samples


@dataclass
class SelectorConfig:
    lambda_unc: float = 0.2
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


@dataclass
class SelectionResult:
    selected_ids: list[str]
    q_scores: np.ndarray
    cluster_assignment: ClusterAssignment
    cluster_budgets: dict[int, int]


def select(
    *,
    sample_ids: list[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    joint_embeddings: np.ndarray,
    budget: int,
    cfg: SelectorConfig | None = None,
) -> SelectionResult:
    cfg = cfg or SelectorConfig()
    q = mu - cfg.lambda_unc * sigma
    asg = cluster_samples(sample_ids=sample_ids, joint_embeddings=joint_embeddings, cfg=cfg.cluster)
    sizes = {int(k): int((asg.cluster_ids == k).sum()) for k in np.unique(asg.cluster_ids)}
    values = compute_cluster_values(
        q_scores=q,
        cluster_ids=asg.cluster_ids,
        top_pct=cfg.quota.top_pct_for_cluster_value,
    )
    budgets = allocate_cluster_budgets(
        cluster_values=values,
        cluster_sizes=sizes,
        total_budget=budget,
        alpha=cfg.quota.alpha,
    )
    picked = pick_top_in_clusters(
        sample_ids=sample_ids,
        q_scores=q,
        cluster_ids=asg.cluster_ids,
        cluster_budgets=budgets,
    )
    return SelectionResult(
        selected_ids=picked,
        q_scores=q,
        cluster_assignment=asg,
        cluster_budgets=budgets,
    )
