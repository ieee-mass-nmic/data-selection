"""End-to-end closure: cached features + labels → scorer training → apply/select.

Exercises the real P3 wiring (`run_scorer_training` → `ScorerInference` →
`select`) on synthetic disk artifacts, so no model download is needed. This is
the closure the offline pipeline performs at its `scorer_train` + apply stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcu_select.data import JsonlPool
from pcu_select.features.cache import FeatureCache
from pcu_select.peft_space.encoder import encode_peft
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.pipeline.apply import run_apply
from pcu_select.pipeline.offline import run_scorer_training
from pcu_select.types import (
    ActivationSignature,
    ApplyConfig,
    DifficultyStats,
    OfflineConfig,
    PEFTConfig,
    Sample,
    SampleFeatures,
    SemanticEmbedding,
    TaskConfig,
    ValidationSketch,
    WorkDirLayout,
)

JOINT_DIM, A_DIM = 8, 24
Z_X_DIM = JOINT_DIM + 16 + A_DIM  # e_x.joint ⊕ d_x(16) ⊕ a_x


def _write_features(cache: FeatureCache, ids: list[str], rng) -> None:
    feats = []
    for sid in ids:
        joint = rng.standard_normal(JOINT_DIM).astype(np.float32)
        feats.append(SampleFeatures(
            sample_id=sid,
            e_x=SemanticEmbedding(instr=joint.copy(), resp=joint.copy(), joint=joint),
            d_x=DifficultyStats(vector=rng.standard_normal(16).astype(np.float32)),
            a_x=ActivationSignature(vector=rng.standard_normal(A_DIM).astype(np.float32)),
        ))
    cache.write_features(feats)
    cache.write_sample_id_index(ids)


def _setup_workdir(tmp_path):
    rng = np.random.default_rng(0)
    layout = WorkDirLayout(tmp_path)
    cache = FeatureCache(layout.features)
    # N ≥ 50 so the selector's default k = max(50, √N) clustering is valid.
    ids = [f"s{i}" for i in range(64)]
    _write_features(cache, ids, rng)

    sites = SiteSpace.uniform(n_layers_total=32, k=8)
    pefts = [
        PEFTConfig(peft_id="pA", family="lora", target_modules=["q_proj", "v_proj"],
                   target_layers=[3, 7, 11, 15, 19, 23, 27, 31], rank=8, alpha=16),
        PEFTConfig(peft_id="pB", family="ia3", target_modules=["q_proj"],
                   target_layers=[3, 7]),
    ]
    layout.peft.mkdir(parents=True, exist_ok=True)
    for p in pefts:
        np.save(layout.peft / f"z_p_{p.peft_id}.npy", encode_peft(p, sites))

    task_id = "tA"
    layout.task.mkdir(parents=True, exist_ok=True)
    np.save(layout.task / f"z_t_{task_id}.npy", rng.standard_normal(Z_X_DIM).astype(np.float32))

    layout.labels.mkdir(parents=True, exist_ok=True)
    lo_rows = [
        {"sample_id": sid, "peft_id": p.peft_id, "task_id": task_id,
         "u_lo": float(rng.standard_normal())}
        for p in pefts for sid in ids
    ]
    pd.DataFrame(lo_rows).to_parquet(layout.labels / "lo_fidelity.parquet")
    hi_rows = [
        {"sample_id": sid, "peft_id": "pA", "task_id": task_id, "u_hi": float(rng.random()),
         "horizon": 4, "anchor_idx": -1, "seed": 0, "delta_raw": 0.1, "sigma_est": 0.05}
        for sid in ids[:40]
    ]
    pd.DataFrame(hi_rows).to_parquet(layout.labels / "hi_fidelity.parquet")
    return layout, cache, pefts, task_id, ids


def test_train_then_apply_select(tmp_path):
    layout, cache, pefts, task_id, ids = _setup_workdir(tmp_path)

    cfg = OfflineConfig(
        scorer_epochs_phase_a=1, scorer_epochs_phase_b=1,
        scorer_batch_size=16, device="cpu",
    )
    ckpt = run_scorer_training(cache=cache, layout=layout, cfg=cfg)
    assert ckpt.exists()
    assert (layout.scorer / "scorer_config.json").exists()

    samples = [Sample(sample_id=sid, instruction="i", response="r") for sid in ids]
    pool = JsonlPool(samples)
    sketch = ValidationSketch(task_id=task_id, samples=samples[:4], sketch_seed=0)
    task = TaskConfig(name="tA", task_id=task_id, sketch=sketch)

    selected = run_apply(
        candidate_pool=pool, peft_target=pefts[0], task_target=task,
        budget=10, scorer_ckpt=ckpt, cfg=ApplyConfig(), workdir=tmp_path,
    )
    assert 0 < len(selected) <= 10
    assert set(selected) <= set(ids)
    assert len(set(selected)) == len(selected)  # no duplicates


def test_training_infers_scorer_dims(tmp_path):
    from pcu_select.scorer.inference import load_scorer_config

    layout, cache, pefts, _, _ = _setup_workdir(tmp_path)
    cfg = OfflineConfig(scorer_epochs_phase_a=1, scorer_epochs_phase_b=1,
                        scorer_batch_size=16, device="cpu")
    ckpt = run_scorer_training(cache=cache, layout=layout, cfg=cfg)
    saved = load_scorer_config(ckpt)
    assert saved is not None
    assert saved.z_x_dim == Z_X_DIM
    assert saved.z_t_dim == Z_X_DIM
    expected_zp = len(encode_peft(pefts[0], SiteSpace.uniform(32, 8)))
    assert saved.z_p_dim == expected_zp
