"""Figure 1 labeling — per-PEFT data-value vectors on the estimation pool.

For each (PEFT p, task t), assign every sample in a small estimation pool a data
value, using signals that are **independent of PCU's scorer** (motivation §0.1):

  u_hi   : real short-update Δ = L_V(θ_a) − L_V(θ_a + Adaptʰ_p(x))  — the truth
           (hi_fidelity.short_update). Computed per (anchor, seed) and stored RAW
           so the F1 plot can build the noise floor (§3.3): two independent
           estimates of the SAME PEFT must agree more than two DIFFERENT PEFTs do.
  u_grad : LESS-style per-PEFT influence (baselines `less`) — cheap cross-check.
  u_rds  : RDS+ semantic similarity (PEFT-agnostic) — the "no disagreement"
           control; identical across PEFTs by construction.

u_grad/u_rds need only the feature cache + task_grad (no GPU); u_hi needs the
backbone + anchors. Output: <workdir>/motivation/values.parquet with columns
(signal, peft_name, peft_id, task_id, anchor_id, seed, sample_id, value).
Down-stream: scripts/plots/plot_motivation_f1.py.

Example:
    python scripts/experiments/build_motivation_values.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --model llama2-7b \
        --tasks gsm8k humaneval --n-val 2000 --seeds 0 1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from _motivation import F1_PEFTS

from pcu_select.baselines import BaselineInputs, score_baseline
from pcu_select.data import JsonlPool, load_sketch
from pcu_select.experiments import MODELS, resolve_peft
from pcu_select.features.cache import FeatureCache
from pcu_select.hi_fidelity import (
    AnchorRegistry,
    AnchorSpec,
    ShortUpdater,
    load_anchor_model,
)
from pcu_select.hi_fidelity.native_peft import SUPPORTED_FAMILIES
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.types import WorkDirLayout
from pcu_select.utils import get_logger

AGNOSTIC = "<agnostic>"  # peft_name sentinel for the PEFT-independent RDS+ signal


def _load_task(layout: WorkDirLayout, task: str, sketch_seed: int):
    sk_path = layout.task / "sketches" / f"{task}_{sketch_seed}.json"
    if not sk_path.exists():
        raise SystemExit(f"missing sketch {sk_path}; run encode_task.py for {task} first.")
    sketch = load_sketch(sk_path)
    tg_path = layout.task / f"task_grad_{sketch.task_id}.npy"
    task_grad = np.load(tg_path).astype(np.float32) if tg_path.exists() else None
    z_t_path = layout.task / f"z_t_{sketch.task_id}.npy"
    z_t = np.load(z_t_path).astype(np.float32) if z_t_path.exists() else None
    return sketch, task_grad, z_t


def main() -> None:
    p = argparse.ArgumentParser(description="Build Figure-1 per-PEFT data-value vectors")
    p.add_argument("--workdir", type=Path, required=True,
                   help="Offline run dir (feature cache + per-task sketches/task_grad).")
    p.add_argument("--pool", type=Path, required=True, help="Candidate pool jsonl.")
    p.add_argument("--model", type=str, default="llama2-7b")
    p.add_argument("--tasks", type=str, nargs="+", default=["gsm8k", "humaneval"])
    p.add_argument("--pefts", type=str, nargs="+", default=F1_PEFTS)
    p.add_argument("--signals", type=str, nargs="+", default=["u_hi", "u_grad", "u_rds"],
                   choices=["u_hi", "u_grad", "u_rds"])
    p.add_argument("--n-val", type=int, default=2000, help="Estimation-pool size.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                   help="Independent u_hi seeds → noise floor (≥2 required for the floor).")
    p.add_argument("--horizon", type=int, default=1, help="Short-update steps for u_hi.")
    p.add_argument("--sketch-seed", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=0, help="Estimation-pool sampling seed.")
    p.add_argument("--warm-anchor", type=Path, default=None,
                   help="Optional warm-anchor adapter ckpt; adds a 2nd anchor (helps the "
                        "noise floor for deterministic families like IA3).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=Path, default=None,
                   help="Output parquet (default: <workdir>/motivation/values.parquet).")
    args = p.parse_args()

    log = get_logger("build_motivation_values")
    if args.model not in MODELS:
        raise SystemExit(f"unknown model {args.model!r}; known: {sorted(MODELS)}")
    spec = MODELS[args.model]
    layout = WorkDirLayout(args.workdir)
    cache = FeatureCache(layout.features)
    pool = JsonlPool.from_jsonl(args.pool)
    sites = SiteSpace.uniform(n_layers_total=spec.n_layers, k=8)

    # ---- fixed estimation subset of the cached candidate ids ----
    all_ids = list(cache.read_sample_id_index())
    n = min(args.n_val, len(all_ids))
    val_ids = random.Random(args.sample_seed).sample(all_ids, n)
    log.info(f"estimation pool: {n} samples (requested {args.n_val})")

    tasks = {t: _load_task(layout, t, args.sketch_seed) for t in args.tasks}
    pefts = {name: resolve_peft(name, args.model) for name in args.pefts}
    rows: list[dict] = []

    # ---- cheap signals (no GPU): u_grad (LESS, per-PEFT) + u_rds (PEFT-agnostic) ----
    if "u_grad" in args.signals or "u_rds" in args.signals:
        joint_dim = cache.read_features()[all_ids[0]].e_x.joint.shape[0]
        for task, (sketch_t, task_grad, z_t) in tasks.items():
            tid = sketch_t.task_id
            # RDS+ query = semantic slice of the pooled task vector (as in _common).
            q_joint = z_t[:joint_dim] if z_t is not None else None
            inp = BaselineInputs.from_cache(cache, sites, task_query_joint=q_joint, task_grad=task_grad)
            idx_of = {sid: i for i, sid in enumerate(inp.sample_ids)}
            keep = [(sid, idx_of[sid]) for sid in val_ids if sid in idx_of]
            if "u_rds" in args.signals:
                s = score_baseline("rds_plus", inp)
                if s is None:
                    log.warning(f"u_rds missing for {task} (no task_query_joint); skip")
                else:
                    for sid, i in keep:
                        rows.append(dict(signal="u_rds", peft_name=AGNOSTIC, peft_id=AGNOSTIC,
                                         task_id=tid, anchor_id="-", seed=-1,
                                         sample_id=sid, value=float(s[i])))
            if "u_grad" in args.signals:
                for name, cfg in pefts.items():
                    s = score_baseline("less", inp, cfg)
                    if s is None:
                        log.warning(f"u_grad missing for {task} (no task_grad); skip")
                        break
                    for sid, i in keep:
                        rows.append(dict(signal="u_grad", peft_name=name, peft_id=cfg.peft_id,
                                         task_id=tid, anchor_id="-", seed=-1,
                                         sample_id=sid, value=float(s[i])))

    # ---- truth signal u_hi: real short updates per (anchor, seed) ----
    if "u_hi" in args.signals:
        hi_pefts = {n_: c for n_, c in pefts.items() if c.family in SUPPORTED_FAMILIES}
        skipped = [n_ for n_ in pefts if n_ not in hi_pefts]
        if skipped:
            log.warning(f"u_hi skips non-native families {skipped} (prefix/ptuning).")
        samples_by_id = {s.sample_id: s for s in pool.take(val_ids)}

        anchors = AnchorRegistry(layout.root / "anchors")
        anchors.register(AnchorSpec(anchor_id="base", checkpoint_path=layout.root / "anchors" / "base"))
        if args.warm_anchor is not None:
            anchors.register(AnchorSpec(anchor_id="warm", checkpoint_path=args.warm_anchor))

        n_combos = len(anchors.all()) * len(args.seeds)
        if n_combos < 2:
            log.warning(f"only {n_combos} (anchor×seed) replicate(s); the F1 noise floor "
                        f"needs ≥2 — pass more --seeds or a --warm-anchor.")
        total = len(anchors.all()) * len(hi_pefts) * len(args.seeds) * len(tasks) * len(samples_by_id)
        log.info(f"u_hi: {total} short updates "
                 f"[{len(anchors.all())} anchors × {len(hi_pefts)} pefts × "
                 f"{len(args.seeds)} seeds × {len(tasks)} tasks × {len(samples_by_id)} samples]")

        done = 0
        for anchor in anchors.all():
            model, tok = load_anchor_model(anchor, base_model_path=spec.hf_id, device=args.device)
            updater = ShortUpdater(model, tok, device=args.device)
            for task, (sketch, _tg, _zt) in tasks.items():
                tid = sketch.task_id
                for name, cfg in hi_pefts.items():
                    for seed in args.seeds:
                        for sid, sample in samples_by_id.items():
                            delta = updater.delta(peft=cfg, sample=sample, sketch=sketch,
                                                  horizon=args.horizon, seed=seed)
                            rows.append(dict(signal="u_hi", peft_name=name, peft_id=cfg.peft_id,
                                             task_id=tid, anchor_id=anchor.anchor_id, seed=seed,
                                             sample_id=sid, value=float(delta)))
                            done += 1
                        log.info(f"  u_hi {done}/{total} "
                                 f"(anchor={anchor.anchor_id} task={task} peft={name} seed={seed})")
            del updater, model

    if not rows:
        raise SystemExit("no value rows produced — check --signals and prerequisites.")
    out = args.out or (layout.root / "motivation" / "values.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    log.info(f"wrote {len(df)} value rows ({sorted(df['signal'].unique())}) → {out}")


if __name__ == "__main__":
    main()
