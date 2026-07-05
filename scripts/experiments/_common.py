"""Shared plumbing for the E1–E5 runner scripts.

The five experiments are just different *loops* over the same primitive:
"select a subset with method M for (PEFT p, task t, budget b, seed s), train the
target PEFT on it, evaluate, and record one ResultRow". That primitive is
`run_cell`. Everything experiment-specific (which cells to run, mismatch
matrices, calibration) stays in the per-experiment scripts.

Assumes an offline run already populated `--workdir` (feature cache, per-task
`z_t`/`task_grad` artifacts, and a trained scorer) — see scripts/build_features,
encode_task, compute_lo_fidelity, train_scorer. Use `--selection-only` to
exercise the matrix/plumbing without loading a 7B backbone (records selection +
ranking metrics + set stats, skips target training).
"""

from __future__ import annotations

import argparse
from importlib import import_module
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pcu_select.baselines import BaselineInputs, score_baseline, select_baseline
from pcu_select.data import JsonlPool, load_sketch
from pcu_select.eval import metrics
from pcu_select.eval.target_train import TargetTrainConfig, load_backbone, train_and_eval
from pcu_select.experiments import MODELS, TASKS, ResultRow, resolve_peft, write_result
from pcu_select.features.cache import FeatureCache
from pcu_select.peft_space.encoder import encode_peft
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.pipeline.apply import run_apply
from pcu_select.scorer.inference import ScorerInference
from pcu_select.types import (
    ApplyConfig,
    PEFTConfig,
    Sample,
    TaskConfig,
    ValidationSketch,
    WorkDirLayout,
)
from pcu_select.utils import get_logger

log = get_logger("experiments")
NDCG_K = 100
TOPK = 100
MetricFactory = Callable[..., Callable[[Any, Any], float | tuple[str, float]]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workdir", type=Path, required=True,
                        help="Offline run dir (feature cache + z_t/task_grad + scorer).")
    parser.add_argument("--pool", type=Path, required=True, help="Candidate pool jsonl.")
    parser.add_argument("--scorer", type=Path, default=None,
                        help="Scorer ckpt for the `pcu` method (default: <workdir>/scorer/ckpt_b.pt).")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Dir with held-out eval sets <task>.jsonl for target eval.")
    parser.add_argument("--task-metric-factory", type=str, default=None,
                        help=("Python callable module:function. It is called once per task as "
                              "factory(task_name=..., task_id=..., metric_name=..., "
                              "eval_set=..., args=...) and must return "
                              "task_metric(model, tokenizer). Required unless --selection-only."))
    parser.add_argument("--model", type=str, default="llama2-7b",
                        help="Backbone tag from experiments.registry.MODELS.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["gsm8k", "humaneval", "mmlu", "tydiqa"])
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sketch-seed", type=int, default=0,
                        help="Which persisted sketch to use as the selection query.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--eval-cap", type=int, default=200, help="Max held-out eval samples.")
    parser.add_argument("--selection-only", action="store_true",
                        help="Skip target training; record selection + ranking metrics only.")
    parser.add_argument("--allow-missing-hi-labels", action="store_true",
                        help=("Allow runs without <workdir>/labels/hi_fidelity.parquet; ranking "
                              "metrics are recorded as NaN. Omit for final experiments."))
    parser.add_argument("--results", type=Path, default=None,
                        help="Results JSONL (default: <workdir>/results/<EXP>.jsonl).")


def _load_callable(spec: str) -> Callable[..., Any]:
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"expected callable spec 'module:function', got {spec!r}")
    module = import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{spec!r} resolved to a non-callable object")
    return obj


# ---------------------------------------------------------------------------
# Context shared across all cells of a run
# ---------------------------------------------------------------------------


@dataclass
class TaskCtx:
    name: str
    task_id: str
    metric_name: str
    sketch: ValidationSketch  # selection query
    eval_set: ValidationSketch  # held-out target eval
    inp: BaselineInputs
    z_t: np.ndarray
    hi_labels: pd.DataFrame | None  # high-fidelity truth for ranking metrics


class RunContext:
    def __init__(self, args: argparse.Namespace, experiment: str):
        self.args = args
        self.experiment = experiment
        self.layout = WorkDirLayout(args.workdir)
        self.cache = FeatureCache(self.layout.features)
        self._metric_factory = (
            _load_callable(args.task_metric_factory) if args.task_metric_factory else None
        )
        if not args.selection_only and self._metric_factory is None:
            raise ValueError(
                "full target evaluation requires --task-metric-factory. "
                "Use --selection-only for selection/ranking smoke tests without target training."
            )
        n_layers = MODELS[args.model].n_layers
        self.sites = SiteSpace.uniform(n_layers_total=n_layers, k=8)
        self.pool = JsonlPool.from_jsonl(args.pool)
        self.scorer_ckpt = args.scorer or (self.layout.scorer / "ckpt_b.pt")
        self.results_path = args.results or (args.workdir / "results" / f"{experiment}.jsonl")
        self._backbone: tuple[Any, Any] | None = None
        self._scorer: ScorerInference | None = None
        self._hi: pd.DataFrame | None = self._load_hi_labels()
        self._tasks: dict[str, TaskCtx] = {}
        self._task_metrics: dict[str, Callable[[Any, Any], float | tuple[str, float]]] = {}

    # ---- lazy heavy resources ----
    def backbone(self) -> tuple[Any, Any]:
        if self._backbone is None:
            spec = MODELS[self.args.model]
            log.info(f"loading backbone {spec.hf_id}")
            self._backbone = load_backbone(spec.hf_id, device=self.args.device)
        return self._backbone

    def scorer(self) -> ScorerInference:
        if self._scorer is None:
            self._scorer = ScorerInference(self.scorer_ckpt)
        return self._scorer

    def _load_hi_labels(self) -> pd.DataFrame | None:
        p = self.layout.labels / "hi_fidelity.parquet"
        if p.exists():
            return pd.read_parquet(p)
        if self.args.allow_missing_hi_labels:
            return None
        raise FileNotFoundError(
            f"missing high-fidelity labels at {p}. Run build_hi_fidelity.py first, or pass "
            "--allow-missing-hi-labels for a selection-only plumbing check."
        )

    def task_metric(self, tc: TaskCtx) -> Callable[[Any, Any], float | tuple[str, float]] | None:
        if self._metric_factory is None:
            return None
        if tc.task_id not in self._task_metrics:
            metric = self._metric_factory(
                task_name=tc.name,
                task_id=tc.task_id,
                metric_name=tc.metric_name,
                eval_set=tc.eval_set,
                args=self.args,
            )
            if not callable(metric):
                raise TypeError("--task-metric-factory must return a callable task metric")
            if not hasattr(metric, "metric_name"):
                try:
                    setattr(metric, "metric_name", tc.metric_name)
                except Exception:
                    pass
            self._task_metrics[tc.task_id] = metric
        return self._task_metrics[tc.task_id]

    # ---- per-task artifacts ----
    def task(self, name: str) -> TaskCtx:
        if name in self._tasks:
            return self._tasks[name]
        if name not in TASKS:
            raise KeyError(f"unknown task {name!r}; known registry tasks: {sorted(TASKS)}")
        metric_name = TASKS[name].metric
        sketch_path = self.layout.task / "sketches" / f"{name}_{self.args.sketch_seed}.json"
        sketch = load_sketch(sketch_path)
        tid = sketch.task_id
        z_t = np.load(self.layout.task / f"z_t_{tid}.npy").astype(np.float32)
        tg_path = self.layout.task / f"task_grad_{tid}.npy"
        task_grad = np.load(tg_path).astype(np.float32) if tg_path.exists() else None
        # Selection-query joint embedding ≈ semantic slice of the pooled task vector.
        joint_dim = self.cache.read_features()[self.cache.read_sample_id_index()[0]].e_x.joint.shape[0]
        inp = BaselineInputs.from_cache(
            self.cache, self.sites,
            task_query_joint=z_t[:joint_dim], task_grad=task_grad,
        )
        eval_set = self._load_eval_set(name, tid)
        hi = None
        if self._hi is not None:
            hi = self._hi[self._hi["task_id"] == tid]
        ctx = TaskCtx(name=name, task_id=tid, metric_name=metric_name,
                      sketch=sketch, eval_set=eval_set,
                      inp=inp, z_t=z_t, hi_labels=hi)
        self._tasks[name] = ctx
        return ctx

    def _load_eval_set(self, name: str, task_id: str) -> ValidationSketch:
        path = self.args.eval_dir / f"{name}.jsonl"
        pool = JsonlPool.from_jsonl(path)
        samples: list[Sample] = list(pool)[: self.args.eval_cap]
        if not samples:
            raise ValueError(f"held-out eval set is empty after --eval-cap for task {name!r}: {path}")
        return ValidationSketch(task_id=task_id, samples=samples, sketch_seed=0)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _pcu_dense(ctx: RunContext, peft: PEFTConfig, tc: TaskCtx,
               scorer: ScorerInference | None = None
               ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Score every cached sample with the scorer for the target (peft, task)."""
    feats = ctx.cache.read_features()
    ids = ctx.cache.read_sample_id_index()
    z_x = np.stack([feats[i].as_z_x() for i in ids], axis=0)
    z_p = encode_peft(peft, ctx.sites)[None, :]
    mu, sigma = (scorer or ctx.scorer()).score(z_x, z_p, tc.z_t[None, :])
    return ids, mu, sigma


def pcu_variant_select(
    ctx: RunContext, peft: PEFTConfig, tc: TaskCtx, budget: float, *,
    scorer: ScorerInference | None = None, strategy: str = "adaptive",
    alpha: float = 0.6, lambda_unc: float = 0.2,
) -> tuple[list[str], np.ndarray]:
    """Apply-time PCU variant for E3/E7 selection-strategy ablations.

    strategy ∈ {"global_topk", "uniform_cluster", "adaptive"}. Returns
    (selected_ids, dense_mu). Representation/condition ablations (-z_p, -z_t,
    lo-only, …) instead pass an alternate `scorer` trained offline.
    """
    from pcu_select.selection.adaptive_quota import QuotaConfig
    from pcu_select.selection.selector import SelectorConfig, select as cluster_select

    ids, mu, sigma = _pcu_dense(ctx, peft, tc, scorer)
    n = len(ids)
    k = int(budget) if budget >= 1 else max(1, int(round(budget * n)))
    q = mu - lambda_unc * sigma
    if strategy == "global_topk":
        top = np.argsort(-q)[:k]
        return [ids[i] for i in top], mu
    joint = ctx.task(tc.name).inp.joint
    a = 0.0 if strategy == "uniform_cluster" else alpha
    res = cluster_select(
        sample_ids=ids, mu=mu, sigma=sigma, joint_embeddings=joint, budget=k,
        cfg=SelectorConfig(lambda_unc=lambda_unc, quota=QuotaConfig(alpha=a)),
    )
    return res.selected_ids, mu


def select(ctx: RunContext, method: str, peft_name: str, tc: TaskCtx, budget: float, seed: int
           ) -> tuple[list[str], float, np.ndarray | None]:
    """Return (selected_ids, select_wall_sec, dense_scores_or_None)."""
    peft = resolve_peft(peft_name, ctx.args.model)
    t0 = time.time()
    if method == "pcu":
        task_cfg = TaskConfig(name=tc.name, task_id=tc.task_id, sketch=tc.sketch)
        ids = run_apply(
            candidate_pool=ctx.pool, peft_target=peft, task_target=task_cfg,
            budget=budget, scorer_ckpt=ctx.scorer_ckpt, cfg=ApplyConfig(),
            workdir=ctx.args.workdir,
        )
        _, mu, _ = _pcu_dense(ctx, peft, tc)
        return ids, time.time() - t0, mu
    ids = select_baseline(method, tc.inp, budget=budget, peft=peft, seed=seed)
    scores = score_baseline(method, tc.inp, peft)
    return ids, time.time() - t0, scores


# ---------------------------------------------------------------------------
# Ranking metrics vs high-fidelity truth
# ---------------------------------------------------------------------------


def ranking_metrics(ctx: RunContext, tc: TaskCtx, peft: PEFTConfig,
                    dense_scores: np.ndarray | None) -> dict[str, float]:
    out = {"spearman": float("nan"), "kendall_tau": float("nan"), "ndcg_at_k": float("nan"),
           "topk_hit_rate": float("nan"), "pairwise_acc": float("nan")}
    if dense_scores is None or tc.hi_labels is None or tc.hi_labels.empty:
        return out
    hi = tc.hi_labels[tc.hi_labels["peft_id"] == peft.peft_id]
    if hi.empty:
        return out
    id_to_idx = {sid: i for i, sid in enumerate(tc.inp.sample_ids)}
    rows = [(id_to_idx[s], u) for s, u in zip(hi["sample_id"], hi["u_hi"]) if s in id_to_idx]
    if len(rows) < 2:
        return out
    idx = np.array([r[0] for r in rows])
    truth = np.array([r[1] for r in rows], dtype=np.float64)
    pred = dense_scores[idx]
    out["spearman"] = metrics.spearman(pred, truth)
    out["kendall_tau"] = metrics.kendall_tau(pred, truth)
    out["ndcg_at_k"] = metrics.ndcg_at_k(pred, truth, NDCG_K)
    out["topk_hit_rate"] = metrics.topk_hit_rate(pred, truth, TOPK)
    out["pairwise_acc"] = metrics.pairwise_acc(pred, truth)
    return out


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


def evaluate_selection(
    ctx: RunContext, *, peft_name: str, task_name: str, budget: float, seed: int,
    ids: list[str], method_tag: str, dense: np.ndarray | None = None,
    select_sec: float = 0.0, extra: dict | None = None, train: bool | None = None,
) -> ResultRow:
    """Given already-selected `ids`, compute ranking metrics, optionally
    train+eval the target PEFT, write and return the ResultRow. Shared by every
    experiment so the train/eval/record path stays identical across E1–E5."""
    tc = ctx.task(task_name)
    peft = resolve_peft(peft_name, ctx.args.model)
    rank = ranking_metrics(ctx, tc, peft, dense)
    row = ResultRow(
        experiment=ctx.experiment, method=method_tag, peft=peft_name,
        task=task_name, budget=budget, seed=seed, model=ctx.args.model,
        select_gpu_h=select_sec / 3600.0, extra={"n_selected": len(ids), **(extra or {})},
        **rank,
    )
    do_train = (not ctx.args.selection_only) if train is None else train
    if do_train:
        model, tok = ctx.backbone()
        samples = ctx.pool.take(ids)
        task_metric = ctx.task_metric(tc)
        if task_metric is None:
            raise ValueError("target training requested but no task metric factory is configured")
        cfg = TargetTrainConfig(
            backbone_model=MODELS[ctx.args.model].hf_id, device=ctx.args.device,
            max_steps=ctx.args.max_steps, batch_size=ctx.args.target_batch_size, seed=seed,
        )
        res = train_and_eval(samples=samples, peft=peft, eval_set=tc.eval_set, cfg=cfg,
                             model=model, tokenizer=tok, task_metric=task_metric)
        row.metric_name, row.metric, row.eval_loss = res.metric_name, res.metric, res.eval_loss
        row.target_train_gpu_h = res.train_wall_sec / 3600.0
        row.extra.update(res.extra)
    write_result(ctx.results_path, row)
    log.info(f"[{ctx.experiment}] {row.method:<16} peft={peft_name:<14} task={task_name:<10} "
             f"b={budget} seed={seed} metric={row.metric:.4f} ndcg={row.ndcg_at_k:.3f}")
    return row


def run_cell(ctx: RunContext, *, method: str, peft_name: str, task_name: str,
             budget: float, seed: int, method_tag: str | None = None,
             extra: dict | None = None) -> ResultRow:
    """Select with `method` then evaluate. Thin wrapper over evaluate_selection."""
    tc = ctx.task(task_name)
    ids, select_sec, dense = select(ctx, method, peft_name, tc, budget, seed)
    return evaluate_selection(
        ctx, peft_name=peft_name, task_name=task_name, budget=budget, seed=seed,
        ids=ids, method_tag=method_tag or method, dense=dense, select_sec=select_sec,
        extra=extra,
    )
