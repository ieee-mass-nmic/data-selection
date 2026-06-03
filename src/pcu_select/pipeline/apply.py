"""Apply pipeline: given trained scorer, select subset for target PEFT/task.

See design doc §15.2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pcu_select.cost.accounting import CostAccountant
from pcu_select.features.cache import FeatureCache
from pcu_select.ood.calibration import OODStats, is_ood
from pcu_select.peft_space.encoder import encode_peft
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.scorer.inference import ScorerInference
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
) -> list[str]:
    log = get_logger("pipeline.apply")
    layout = WorkDirLayout(Path(workdir))
    accountant = CostAccountant(layout.cost / "accounting.jsonl")
    cache = FeatureCache(layout.features)

    n = len(candidate_pool)
    budget_int = int(budget) if budget >= 1 else int(round(budget * n))
    sites = SiteSpace.uniform(n_layers_total=32, k=8)

    with accountant.stage("apply_score", n_samples=n, peft_id=peft_target.peft_id,
                          task_id=task_target.task_id):
        z_x = _load_z_x(cache)
        z_p = encode_peft(peft_target, sites)[None, :]
        z_t = _load_z_t(layout, task_target)[None, :]
        if ood_stats is not None and is_ood(z_p[0], ood_stats):
            log.warning("Target PEFT is OOD; consider calibration mode.")
        scorer = ScorerInference(scorer_ckpt)
        mu, sigma = scorer.score(z_x, z_p, z_t)

    with accountant.stage("apply_select", n_samples=n, peft_id=peft_target.peft_id,
                          task_id=task_target.task_id):
        sample_ids = cache.read_sample_id_index()
        joint = _load_joint_embeddings(cache)
        result = select(
            sample_ids=sample_ids,
            mu=mu,
            sigma=sigma,
            joint_embeddings=joint,
            budget=budget_int,
            cfg=SelectorConfig(lambda_unc=cfg.lambda_unc),
        )
    out_dir = layout.selection / f"{peft_target.peft_id}_{task_target.task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected.txt").write_text("\n".join(result.selected_ids))
    log.info(f"selected {len(result.selected_ids)} samples → {out_dir}")
    return result.selected_ids


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_z_x(cache: FeatureCache) -> np.ndarray:
    feats = cache.read_features()
    ids = cache.read_sample_id_index()
    rows = [feats[i].as_z_x() for i in ids]
    return np.stack(rows, axis=0)


def _load_joint_embeddings(cache: FeatureCache) -> np.ndarray:
    feats = cache.read_features()
    ids = cache.read_sample_id_index()
    return np.stack([feats[i].e_x.joint for i in ids], axis=0)


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
