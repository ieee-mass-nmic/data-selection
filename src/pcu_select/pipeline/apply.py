"""Apply pipeline: given trained scorer, select subset for target PEFT/task.

See design doc §15.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from pcu_select.cost.accounting import CostAccountant
from pcu_select.features.cache import FeatureCache
from pcu_select.ood.calibration import OODStats, apply_calibration, is_ood, load_calibration
from pcu_select.peft_space.encoder import encode_peft
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.scorer.inference import ScorerInference
from pcu_select.selection.adaptive_quota import QuotaConfig
from pcu_select.selection.cluster import ClusterConfig
from pcu_select.selection.selector import SelectorConfig, select
from pcu_select.types import (
    ApplyConfig,
    DatasetLike,
    PEFTConfig,
    TaskConfig,
    WorkDirLayout,
)
from pcu_select.utils import get_logger


def run_apply(
    *,
    candidate_pool: DatasetLike,
    peft_target: PEFTConfig,
    task_target: TaskConfig,
    budget: int | float,
    scorer_ckpt: Path | str,
    cfg: ApplyConfig,
    workdir: Path | str,
    ood_stats: OODStats | None = None,
    calibration_ckpt: Path | str | None = None,
) -> list[str]:
    log = get_logger("pipeline.apply")
    layout = WorkDirLayout(Path(workdir))
    accountant = CostAccountant(layout.cost / "accounting.jsonl")
    cache = FeatureCache(layout.features)

    candidate_ids = [sample.sample_id for sample in candidate_pool]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_pool contains duplicate sample_id values")
    n = len(candidate_ids)
    budget_int = int(budget) if budget >= 1 else int(round(budget * n))
    sites = SiteSpace.uniform(n_layers_total=cfg.n_layers_total, k=cfg.n_layers_signature)

    with accountant.stage("apply_score", n_samples=n, peft_id=peft_target.peft_id,
                          task_id=task_target.task_id):
        z_x = _load_z_x(cache, candidate_ids)
        z_p = encode_peft(peft_target, sites)[None, :]
        z_t = _load_z_t(layout, task_target)[None, :]
        target_is_ood = ood_stats is not None and is_ood(z_p[0], ood_stats)
        if target_is_ood:
            log.warning("Target PEFT is OOD; consider calibration mode.")
        scorer = ScorerInference(scorer_ckpt)
        mu, sigma = scorer.score(z_x, z_p, z_t)

        # Apply a pre-fit calibration head when available (design §13.2). This
        # makes `enable_calibration` an actual deploy-time behavior rather than
        # an experiment-only capability; without a ckpt we fall back to raw μ̂.
        if cfg.enable_calibration and calibration_ckpt is not None:
            head = load_calibration(calibration_ckpt)
            mu = apply_calibration(head, mu=mu, z_x=z_x, z_p=z_p, z_t=z_t)
            log.info(f"applied calibration head from {calibration_ckpt}")
        elif cfg.enable_calibration and target_is_ood:
            log.warning(
                "enable_calibration=True and target is OOD, but no calibration_ckpt "
                "was provided; scoring with uncalibrated μ̂."
            )

    with accountant.stage("apply_select", n_samples=n, peft_id=peft_target.peft_id,
                          task_id=task_target.task_id):
        sample_ids = candidate_ids
        joint = _load_joint_embeddings(cache, candidate_ids)
        result = select(
            sample_ids=sample_ids,
            mu=mu,
            sigma=sigma,
            joint_embeddings=joint,
            budget=budget_int,
            cfg=SelectorConfig(
                lambda_unc=cfg.lambda_unc,
                quota=QuotaConfig(
                    alpha=cfg.cluster_alpha,
                    min_cluster_size=cfg.min_cluster_size,
                ),
                cluster=ClusterConfig(k=cfg.cluster_k),
            ),
        )
    out_dir = layout.selection / f"{peft_target.peft_id}_{task_target.task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected.txt").write_text("\n".join(result.selected_ids))
    log.info(f"selected {len(result.selected_ids)} samples → {out_dir}")
    return result.selected_ids


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_z_x(cache: FeatureCache, sample_ids: list[str] | None = None) -> np.ndarray:
    feats = cache.read_features()
    ids = sample_ids if sample_ids is not None else cache.read_sample_id_index()
    _ensure_cached(ids, feats)
    if not ids:
        dim = len(next(iter(feats.values())).as_z_x()) if feats else 0
        return np.zeros((0, dim), dtype=np.float32)
    rows = [feats[i].as_z_x() for i in ids]
    return np.stack(rows, axis=0)


def _load_joint_embeddings(cache: FeatureCache, sample_ids: list[str] | None = None) -> np.ndarray:
    feats = cache.read_features()
    ids = sample_ids if sample_ids is not None else cache.read_sample_id_index()
    _ensure_cached(ids, feats)
    if not ids:
        dim = len(next(iter(feats.values())).e_x.joint) if feats else 0
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack([feats[i].e_x.joint for i in ids], axis=0)


def _ensure_cached(ids: list[str], feats: Mapping[str, object]) -> None:
    missing = [sid for sid in ids if sid not in feats]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"candidate_pool has {len(missing)} sample_id(s) missing from feature cache: {preview}{suffix}")


def _load_z_t(layout: WorkDirLayout, task: TaskConfig) -> np.ndarray:
    """Load pre-computed z_t for the task. The offline pipeline persists this
    when sketch features are extracted."""
    p = layout.task / f"z_t_{task.task_id}.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"Pre-computed z_t for task {task.name} not found at {p}. "
            "Run offline pipeline or `scripts/encode_task.py` first."
        )
    return np.load(p)
