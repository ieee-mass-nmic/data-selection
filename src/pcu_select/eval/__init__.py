"""Evaluation for the experiment harness.

- `metrics`      : ranking / set-overlap metrics (pure numpy, no scipy dep).
- `target_train` : fine-tune a PEFT on a selected subset and evaluate it.
"""

from pcu_select.eval.metrics import (
    jaccard,
    kendall_tau,
    ndcg_at_k,
    pairwise_acc,
    spearman,
    topk_hit_rate,
)

__all__ = [
    "jaccard",
    "kendall_tau",
    "ndcg_at_k",
    "pairwise_acc",
    "spearman",
    "topk_hit_rate",
]
