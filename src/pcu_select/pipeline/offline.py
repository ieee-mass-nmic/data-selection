"""Offline training pipeline. See design doc §15.1 and §10.

Orchestrates the full meta-training: features → low-fidelity proxy → high-
fidelity labels → scorer training. Each step delegates to its module and
writes intermediate artifacts under `workdir`.

Status: the full pipeline is wired — feature extraction (§7), low-fidelity
proxy (§5), high-fidelity labeler (§10, phase-1 coverage) and two-phase scorer
training (§11). Active-sampling phases 2/3 (which re-label using a trained
scorer) are driven separately and are not part of the single offline pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pcu_select.cost.accounting import CostAccountant
from pcu_select.data import JsonlPool
from pcu_select.features.cache import FeatureCache
from pcu_select.features.difficulty import quick_text_stats
from pcu_select.features.selector_runner import SelectorRunner, SelectorRunnerConfig
from pcu_select.features.semantic import SemanticEncoder, SemanticEncoderConfig
from pcu_select.hi_fidelity import AnchorRegistry
from pcu_select.peft_space.encoder import encode_peft
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.proxy.lo_fidelity import LoFidelityScorer
from pcu_select.proxy.projection import ProjectionConfig, ProjectionStore, project
from pcu_select.scorer.inference import save_scorer_config
from pcu_select.scorer.model import PCUScorer, ScorerConfig
from pcu_select.scorer.trainer import TrainerConfig, TripletDataset, make_loader, train_scorer
from pcu_select.types import (
    ActivationSignature,
    DatasetLike,
    DifficultyStats,
    OfflineConfig,
    PEFTConfig,
    SampleFeatures,
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
      1. extract per-sample features (e_x, d_x, a_x) + grad signatures → cache
      2. compute task grad signatures + z_t, then u^lo for (sample, peft, task)
      3. sample phase-1 triples, run high-fidelity labeler → u^hi   [P2]
      4. iterate scorer pretrain (phase A) + joint train (phase B)  [P3]
    """
    log = get_logger("pipeline.offline")
    seed_everything(cfg.global_seed)
    layout = WorkDirLayout(Path(workdir))
    layout.root.mkdir(parents=True, exist_ok=True)
    accountant = CostAccountant(layout.cost / "accounting.jsonl")

    sites = SiteSpace.uniform(n_layers_total=cfg.n_layers_total, k=cfg.n_layers_signature)
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
        _build_hi_labels(meta_pool=meta_pool, pefts=peft_space, tasks=tasks,
                         anchors=anchors, cache=feature_cache, workdir=layout, cfg=cfg)
    log.info("high-fidelity done")

    with accountant.stage("scorer_train"):
        ckpt = run_scorer_training(cache=feature_cache, layout=layout, cfg=cfg)
    log.info(f"scorer trained: {ckpt}")
    return ckpt


# -----------------------------------------------------------------------------
# Feature extraction (design doc §7, §5)
# -----------------------------------------------------------------------------


def _make_runner(sites: SiteSpace, cfg: OfflineConfig) -> SelectorRunner:
    return SelectorRunner(
        sites,
        SelectorRunnerConfig(selector_model=cfg.selector_model, device=cfg.device),
    )


def _projection_store(cache: FeatureCache, d_model: int, cfg: OfflineConfig) -> ProjectionStore:
    return ProjectionStore(
        ProjectionConfig(d_model=d_model, d_proj=cfg.d_proj, global_seed=cfg.global_seed),
        cache.paths.root / "projections",
    )


def _difficulty_vector(sample, model_stats: np.ndarray) -> np.ndarray:
    vec = quick_text_stats(sample)
    vec[4 : 4 + model_stats.shape[0]] = model_stats
    return vec


def _build_features(*, meta_pool: DatasetLike, cache: FeatureCache,
                    sites: SiteSpace, cfg: OfflineConfig) -> None:
    """Extract e_x / d_x / a_x and per-site gradient signatures for every sample.

    One forward+backward per sample produces the model-side difficulty stats,
    the activation signature and the raw pooled gradients; the gradients are
    then projected to `d_proj` and persisted as per-site shards.
    """
    log = get_logger("pipeline.offline.features")
    samples = list(meta_pool)
    sem = SemanticEncoder(SemanticEncoderConfig(device=cfg.device))
    embeddings = sem.encode_samples(samples)

    runner = _make_runner(sites, cfg)
    d_model = runner.hidden_size
    proj = _projection_store(cache, d_model, cfg)

    feats: list[SampleFeatures] = []
    ids: list[str] = []
    pooled_by_site: dict = {s: [] for s in sites.all_sites}

    for i, sample in enumerate(samples):
        res = runner.process(sample, want_grads=True, want_activations=True)
        assert res.activation is not None and res.site_grads is not None
        feats.append(SampleFeatures(
            sample_id=sample.sample_id,
            e_x=embeddings[i],
            d_x=DifficultyStats(vector=_difficulty_vector(sample, res.model_stats)),
            a_x=ActivationSignature(vector=res.activation),
        ))
        ids.append(sample.sample_id)
        for s in sites.all_sites:
            pooled_by_site[s].append(res.site_grads[s])
        if (i + 1) % 200 == 0:
            log.info(f"features {i + 1}/{len(samples)}")

    cache.write_features(feats)
    cache.write_sample_id_index(ids)
    for s in sites.all_sites:
        pooled = np.stack(pooled_by_site[s], axis=0)  # (N, d_model)
        cache.write_grad_signature(s, project(pooled, proj.get(s)))  # (N, d_proj)
    log.info(f"persisted features + grad signatures for {len(samples)} samples")


# -----------------------------------------------------------------------------
# Task artifacts + low-fidelity labels (design doc §5.4, §9, §5.5)
# -----------------------------------------------------------------------------


def build_task_artifacts(
    *,
    task: TaskConfig,
    sites: SiteSpace,
    layout: WorkDirLayout,
    cfg: OfflineConfig,
    runner: SelectorRunner,
    proj: ProjectionStore,
    sem: SemanticEncoder,
) -> np.ndarray:
    """Compute and persist the task grad signature g_t and the task vector z_t.

    Returns g_t with shape (|Ω|, d_proj). z_t is the mean of the per-sketch
    sample representations z_v (= z_x); the set-transformer pool in
    `task_cond.encoder` is an alternative that would be trained jointly with
    the scorer (future work).
    """
    sketch = task.sketch
    embeddings = sem.encode_samples(sketch.samples)
    all_sites = sites.all_sites
    pooled_by_site: dict = {s: [] for s in all_sites}
    z_v_rows: list[np.ndarray] = []

    for i, sample in enumerate(sketch.samples):
        res = runner.process(sample, want_grads=True, want_activations=True)
        assert res.activation is not None and res.site_grads is not None
        d_vec = _difficulty_vector(sample, res.model_stats)
        z_v_rows.append(np.concatenate([embeddings[i].joint, d_vec, res.activation]))
        for s in all_sites:
            pooled_by_site[s].append(res.site_grads[s])

    g_t = np.zeros((len(all_sites), cfg.d_proj), dtype=np.float32)
    for j, s in enumerate(all_sites):
        projected = project(np.stack(pooled_by_site[s], axis=0), proj.get(s))  # (V, d_proj)
        avg = projected.mean(axis=0)
        g_t[j] = avg / max(float(np.linalg.norm(avg)), 1e-8)

    z_t = np.stack(z_v_rows, axis=0).mean(axis=0).astype(np.float32)

    layout.task.mkdir(parents=True, exist_ok=True)
    np.save(layout.task / f"task_grad_{task.task_id}.npy", g_t)
    np.save(layout.task / f"z_t_{task.task_id}.npy", z_t)
    return g_t


def _build_lo_labels(*, pefts: Sequence[PEFTConfig], tasks: Sequence[TaskConfig],
                     sites: SiteSpace, cache: FeatureCache,
                     workdir: WorkDirLayout, cfg: OfflineConfig) -> None:
    """Per task compute g_t/z_t, then per PEFT compute u^lo over cached g_x.

    Writes `workdir.labels / 'lo_fidelity.parquet'`."""
    log = get_logger("pipeline.offline.lo")
    runner = _make_runner(sites, cfg)
    proj = _projection_store(cache, runner.hidden_size, cfg)
    sem = SemanticEncoder(SemanticEncoderConfig(device=cfg.device))
    scorer = LoFidelityScorer(sites, cache)

    rows: list[dict] = []
    for task in tasks:
        g_t = build_task_artifacts(
            task=task, sites=sites, layout=workdir, cfg=cfg,
            runner=runner, proj=proj, sem=sem,
        )
        for peft in pefts:
            res = scorer.score(peft=peft, g_t_per_site=g_t)
            for sid, u in zip(res.sample_ids, res.u_lo):
                rows.append({"sample_id": sid, "peft_id": peft.peft_id,
                             "task_id": task.task_id, "u_lo": float(u)})
        log.info(f"u^lo for task {task.name}: {len(pefts)} PEFTs × {len(res.sample_ids)} samples")

    # Persist z_p per PEFT so scorer training / apply can look it up by id
    # without reconstructing PEFTConfig objects (design doc §8, §15).
    workdir.peft.mkdir(parents=True, exist_ok=True)
    for peft in pefts:
        np.save(workdir.peft / f"z_p_{peft.peft_id}.npy", encode_peft(peft, sites))

    workdir.labels.mkdir(parents=True, exist_ok=True)
    out = workdir.labels / "lo_fidelity.parquet"
    pd.DataFrame(rows).to_parquet(out)
    log.info(f"wrote {len(rows)} low-fidelity labels → {out}")


# -----------------------------------------------------------------------------
# High-fidelity labels (design doc §10) + two-phase scorer training (§11)
# -----------------------------------------------------------------------------


def _build_hi_labels(*, meta_pool: DatasetLike, pefts: Sequence[PEFTConfig],
                     tasks: Sequence[TaskConfig], anchors: AnchorRegistry,
                     cache: FeatureCache, workdir: WorkDirLayout, cfg: OfflineConfig) -> None:
    """Build anchors, sample phase-1 triples, run the labeler, persist u^hi.

    Only phase 1 (stratified coverage, 50 % of `q_hi_total`) runs offline;
    phases 2 (uncertainty) and 3 (boundary) require a v0 scorer and are driven
    later (P3). Writes `workdir.labels / 'hi_fidelity.parquet'`.
    """
    from pcu_select.hi_fidelity import (
        AnchorSpec,
        HiFidelityLabeler,
        LabelerConfig,
        ShortUpdater,
        build_warm_anchor,
        load_anchor_model,
        phase1_stratified,
        split_budget,
    )
    from pcu_select.selection.cluster import ClusterConfig, cluster_samples

    log = get_logger("pipeline.offline.hi")
    backbone = cfg.backbone_model or cfg.selector_model

    # --- anchors θ_base, θ_warm (design §10.1) -----------------------------
    anchors.register(AnchorSpec(anchor_id="base", checkpoint_path=Path(backbone), note="pretrained"))
    # Design §10.1: warm anchor trains on a RANDOM slice of the meta-pool (not the
    # first N, which biases θ_warm toward whatever sits at the head of the pool).
    # Sampling is seeded for reproducibility.
    n_total = len(meta_pool)
    k_slice = min(cfg.anchor_warm_slice, n_total)
    rng = np.random.default_rng(cfg.global_seed)
    chosen = set(int(i) for i in rng.choice(n_total, size=k_slice, replace=False))
    slice_samples = [s for i, s in enumerate(meta_pool) if i in chosen]
    slice_path = anchors.root / "warm_slice.jsonl"
    JsonlPool(slice_samples).to_jsonl(slice_path)
    warm = build_warm_anchor(
        base_model_path=backbone, meta_pool_slice_path=slice_path,
        out_path=anchors.root / "warm_lora.pt", steps=cfg.anchor_warm_steps,
        rank=cfg.anchor_lora_rank, device=cfg.device, max_len=cfg.max_len, seed=cfg.global_seed,
    )
    anchors.register(warm)

    # --- phase-1 stratified triple sampling (design §10.4) -----------------
    sample_ids = cache.read_sample_id_index()
    features = cache.read_features()
    joint = np.stack([features[sid].e_x.joint for sid in sample_ids], axis=0)
    assign = cluster_samples(sample_ids=sample_ids, joint_embeddings=joint, cfg=ClusterConfig(seed=cfg.global_seed))
    sample_cluster = {sid: int(c) for sid, c in zip(assign.sample_ids, assign.cluster_ids)}
    budget = split_budget(cfg.q_hi_total, cfg.phase_split)
    triples = phase1_stratified(
        sample_ids=sample_ids, sample_cluster=sample_cluster,
        pefts=pefts, tasks=tasks, budget=budget.phase1, seed=cfg.global_seed,
    )
    log.info(f"phase-1 sampled {len(triples)} triples (budget {budget.phase1})")

    # --- short-update labeling --------------------------------------------
    needed_ids = {t.sample_id for t in triples}
    samples_by_id = {s.sample_id: s for s in meta_pool.take(needed_ids)}
    pefts_by_id = {p.peft_id: p for p in pefts}
    sketches_by_id = {t.task_id: t.sketch for t in tasks}

    def _updater_factory(spec: AnchorSpec) -> ShortUpdater:
        model, tok = load_anchor_model(
            spec, base_model_path=backbone, device=cfg.device, rank=cfg.anchor_lora_rank,
        )
        return ShortUpdater(model, tok, device=cfg.device, max_len=cfg.max_len)

    labeler = HiFidelityLabeler(
        anchors=anchors, samples_by_id=samples_by_id, pefts_by_id=pefts_by_id,
        sketches_by_id=sketches_by_id, updater_factory=_updater_factory,
        cfg=LabelerConfig(horizons=cfg.horizons, horizon_weights=cfg.horizon_weights, seed=cfg.global_seed),
    )
    labels = labeler.run(triples)

    workdir.labels.mkdir(parents=True, exist_ok=True)
    out = workdir.labels / "hi_fidelity.parquet"
    HiFidelityLabeler.save_labels(labels, out)
    log.info(f"wrote {len(labels)} high-fidelity labels → {out}")


def _scorer_lookups(
    cache: FeatureCache, layout: WorkDirLayout
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Assemble (z_x, z_p, z_t) lookup tables from disk artifacts.

    z_x comes from the feature cache (e_x ⊕ d_x ⊕ a_x); z_p / z_t are the
    per-PEFT / per-task vectors persisted by `_build_lo_labels` and
    `build_task_artifacts`.
    """
    feats = cache.read_features()
    z_x_by_id = {sid: sf.as_z_x().astype(np.float32) for sid, sf in feats.items()}
    z_p_by_id: dict[str, np.ndarray] = {}
    for p in sorted(layout.peft.glob("z_p_*.npy")):
        z_p_by_id[p.stem[len("z_p_"):]] = np.load(p).astype(np.float32)
    z_t_by_id: dict[str, np.ndarray] = {}
    for p in sorted(layout.task.glob("z_t_*.npy")):
        z_t_by_id[p.stem[len("z_t_"):]] = np.load(p).astype(np.float32)
    return z_x_by_id, z_p_by_id, z_t_by_id


def _label_rows(layout: WorkDirLayout) -> tuple[list[dict], list[dict]]:
    """Load phase-A (low-fidelity) and phase-B (high-fidelity ⋈ u^lo) rows."""
    lo_path = layout.labels / "lo_fidelity.parquet"
    lo_rows = pd.read_parquet(lo_path).to_dict("records") if lo_path.exists() else []
    u_lo_by_key = {
        (r["sample_id"], r["peft_id"], r["task_id"]): r["u_lo"] for r in lo_rows
    }
    hi_path = layout.labels / "hi_fidelity.parquet"
    hi_rows: list[dict] = []
    if hi_path.exists():
        for r in pd.read_parquet(hi_path).to_dict("records"):
            key = (r["sample_id"], r["peft_id"], r["task_id"])
            hi_rows.append({**r, "u_lo": u_lo_by_key.get(key, float("nan"))})
    return lo_rows, hi_rows


def run_scorer_training(*, cache: FeatureCache, layout: WorkDirLayout, cfg: OfflineConfig) -> Path:
    """Two-phase scorer training from cached labels + feature lookups (§11.4).

    Phase A pretrains on low-fidelity proxy labels; phase B jointly fits the
    rank / regression / proxy / uncertainty objective on high-fidelity labels
    (joined with u^lo). Falls back gracefully if one label set is absent. The
    scorer's tower dimensions are inferred from the actual vectors.
    """
    log = get_logger("pipeline.offline.train")
    z_x_by_id, z_p_by_id, z_t_by_id = _scorer_lookups(cache, layout)
    lo_rows, hi_rows = _label_rows(layout)
    if not lo_rows and not hi_rows:
        raise RuntimeError("no labels found under workdir.labels; run lo/hi stages first")
    if not z_p_by_id or not z_t_by_id:
        raise RuntimeError("missing z_p / z_t artifacts; run the lo stage first")

    phase_a_rows = lo_rows or hi_rows
    phase_b_rows = hi_rows or lo_rows
    z_x_dim = len(next(iter(z_x_by_id.values())))
    z_p_dim = len(next(iter(z_p_by_id.values())))
    z_t_dim = len(next(iter(z_t_by_id.values())))
    scorer_cfg = ScorerConfig(z_x_dim=z_x_dim, z_p_dim=z_p_dim, z_t_dim=z_t_dim)
    model = PCUScorer(scorer_cfg)
    layout.scorer.mkdir(parents=True, exist_ok=True)
    save_scorer_config(scorer_cfg, layout.scorer)

    def _dataset(rows: list[dict]) -> TripletDataset:
        return TripletDataset(
            rows=rows, z_x_by_id=z_x_by_id, z_p_by_id=z_p_by_id, z_t_by_id=z_t_by_id
        )

    loader_a = make_loader(_dataset(phase_a_rows), batch_size=cfg.scorer_batch_size)
    loader_b = make_loader(_dataset(phase_b_rows), batch_size=cfg.scorer_batch_size)
    trainer_cfg = TrainerConfig(
        epochs_phase_a=cfg.scorer_epochs_phase_a,
        epochs_phase_b=cfg.scorer_epochs_phase_b,
        lr_phase_a=cfg.scorer_lr_phase_a,
        lr_phase_b=cfg.scorer_lr_phase_b,
        batch_size=cfg.scorer_batch_size,
        weights_phase_b=cfg.loss_weights,
        device=cfg.device,
    )
    log.info(f"training scorer: |A|={len(phase_a_rows)} |B|={len(phase_b_rows)} "
             f"dims=({z_x_dim},{z_p_dim},{z_t_dim})")
    return train_scorer(
        model=model, phase_a_loader=loader_a, phase_b_loader=loader_b,
        cfg=trainer_cfg, ckpt_dir=layout.scorer,
    )
