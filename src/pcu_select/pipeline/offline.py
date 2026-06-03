"""Offline training pipeline. See design doc §15.1 and §10.

Orchestrates the full meta-training: features → low-fidelity proxy → high-
fidelity labels → scorer training. Each step delegates to its module and
writes intermediate artifacts under `workdir`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pcu_select.cost.accounting import CostAccountant
from pcu_select.features.cache import FeatureCache
from pcu_select.hi_fidelity import (
    AnchorRegistry,
    HiFidelityLabeler,
    phase1_stratified,
    split_budget,
)
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.scorer.model import PCUScorer
from pcu_select.scorer.trainer import TrainerConfig, train_scorer
from pcu_select.types import (
    DatasetLike,
    OfflineConfig,
    PEFTConfig,
    TaskConfig,
    WorkDirLayout,
)
from pcu_select.utils import get_logger, seed_everything


def run_offline(
    *,
    meta_pool: DatasetLike,
    peft_space: Sequence[PEFTConfig],
    tasks: Sequence[TaskConfig],
    cfg: OfflineConfig,
    workdir: Path | str,
) -> Path:
    """End-to-end offline pipeline. Returns path to the final scorer ckpt.

    Steps (each wrapped in a CostAccountant stage):
      1. extract per-sample features (e_x, d_x, a_x) → FeatureCache
      2. compute site-wise gradient signatures for samples and task sketches
      3. compute u^lo for the cartesian product of (sample, peft, task)
      4. sample phase-1 triples, run high-fidelity labeler → u^hi
      5. iterate scorer pretrain (phase A) + joint train (phase B)
      6. (optional) phase-2 / phase-3 active sampling rounds
    """
    log = get_logger("pipeline.offline")
    seed_everything(cfg.global_seed)
    layout = WorkDirLayout(Path(workdir))
    layout.root.mkdir(parents=True, exist_ok=True)
    accountant = CostAccountant(layout.cost / "accounting.jsonl")

    sites = SiteSpace.uniform(n_layers_total=32, k=cfg.n_layers_signature)
    feature_cache = FeatureCache(layout.features)
    anchors = AnchorRegistry(layout.root / "anchors")

    with accountant.stage("feat", n_samples=len(meta_pool)):
        _build_features(meta_pool=meta_pool, cache=feature_cache, sites=sites, cfg=cfg)
    log.info("features done")

    with accountant.stage("lo"):
        _build_lo_labels(pefts=peft_space, tasks=tasks, sites=sites,
                          cache=feature_cache, workdir=layout, cfg=cfg)
    log.info("low-fidelity done")

    with accountant.stage("hi"):
        _build_hi_labels(pefts=peft_space, tasks=tasks, anchors=anchors,
                         workdir=layout, cfg=cfg)
    log.info("high-fidelity done")

    with accountant.stage("scorer_train"):
        ckpt = _train(scorer_workdir=layout.scorer, cfg=cfg)
    log.info(f"scorer trained: {ckpt}")
    return ckpt


# -----------------------------------------------------------------------------
# Subroutines (stubs)
# -----------------------------------------------------------------------------


def _build_features(*, meta_pool, cache, sites, cfg) -> None:
    """Wire SemanticEncoder + ModelStatsExtractor + ActivationSignatureExtractor.

    The full implementation will:
        - iterate the pool in batches
        - run sentence-transformer for e_x
        - run selector model with hooks for a_x and grad signatures
        - persist results to `cache`
    """
    raise NotImplementedError("Skeleton — implement per design doc §7.")


def _build_lo_labels(*, pefts, tasks, sites, cache, workdir, cfg) -> None:
    """For each task, compute g_t per site; for each PEFT, compute u^lo over the
    cached g_x. Write to `workdir.labels / 'lo_fidelity.parquet'`."""
    raise NotImplementedError


def _build_hi_labels(*, pefts, tasks, anchors, workdir, cfg) -> None:
    """Run sampler.phase1_stratified, then HiFidelityLabeler.run. Persist labels."""
    raise NotImplementedError


def _train(*, scorer_workdir: Path, cfg: OfflineConfig) -> Path:
    """Load lo/hi parquet, build TripletDataset, call train_scorer."""
    model = PCUScorer()
    trainer_cfg = TrainerConfig(
        epochs_phase_a=cfg.scorer_epochs_phase_a,
        epochs_phase_b=cfg.scorer_epochs_phase_b,
        lr_phase_a=cfg.scorer_lr_phase_a,
        lr_phase_b=cfg.scorer_lr_phase_b,
        weights_phase_b=cfg.loss_weights,
    )
    # The two DataLoaders are constructed from cached parquet rows; the
    # construction is intentionally left to a separate helper because the
    # exact join schema depends on the experiment.
    raise NotImplementedError("Compose phase A/B DataLoaders, then call train_scorer().")
