"""Sample-space clustering. See design doc §12.2.

Clusters on the joint semantic embedding `e_x.joint`. MiniBatch KMeans keeps
the cost linear in N. Returns a `(sample_id -> cluster_id)` mapping plus
cluster centers for diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class ClusterConfig:
    k: int | None = None  # default: max(50, sqrt(N))
    seed: int = 0
    batch_size: int = 4096
    n_init: int = 3


@dataclass
class ClusterAssignment:
    sample_ids: list[str]
    cluster_ids: np.ndarray  # (N,)
    centers: np.ndarray      # (k, d)


def cluster_samples(
    *,
    sample_ids: list[str],
    joint_embeddings: np.ndarray,
    cfg: ClusterConfig | None = None,
) -> ClusterAssignment:
    cfg = cfg or ClusterConfig()
    n = joint_embeddings.shape[0]
    k = cfg.k if cfg.k is not None else max(50, int(math.sqrt(n)))
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=cfg.batch_size,
        random_state=cfg.seed,
        n_init=cfg.n_init,
    )
    cluster_ids = km.fit_predict(joint_embeddings)
    return ClusterAssignment(
        sample_ids=list(sample_ids),
        cluster_ids=cluster_ids.astype(np.int32),
        centers=km.cluster_centers_.astype(np.float32),
    )
