"""CLI: build the offline high-fidelity label set hi_fidelity.parquet (design §10).

This is the "high-fidelity labeling" step between compute_lo_fidelity.py and
train_scorer.py (see scripts/experiments/README prerequisites). It runs the SAME
stage the full offline pipeline uses (`pipeline.offline._build_hi_labels`):
build the θ_base / θ_warm anchors, draw phase-1 stratified (x, p, t) triples
(half of --q-hi-total), run the short-update labeler, and persist u^hi.

Phases 2/3 (uncertainty / boundary active sampling) need a v0 scorer and are
driven separately (design §10.4); this script produces the phase-1 labels that
train_scorer.py consumes.

Prerequisites for --workdir: build_features.py (feature cache) and encode_task.py
(one sketch + z_t per task) must have run first.

Usage:
    python scripts/build_hi_fidelity.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --model llama2-7b \
        --tasks gsm8k humaneval mmlu tydiqa --q-hi-total 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcu_select.data import JsonlPool, load_sketch
from pcu_select.experiments import MODELS, peft_specs_by_group, resolve_peft
from pcu_select.features.cache import FeatureCache
from pcu_select.hi_fidelity import AnchorRegistry
from pcu_select.pipeline.offline import _build_hi_labels
from pcu_select.types import OfflineConfig, TaskConfig, WorkDirLayout
from pcu_select.utils import get_logger


def main() -> None:
    p = argparse.ArgumentParser(description="Build offline high-fidelity labels (phase-1)")
    p.add_argument("--workdir", type=Path, required=True,
                   help="Offline run dir (feature cache + per-task sketches).")
    p.add_argument("--pool", type=Path, required=True, help="Meta-pool jsonl (candidate pool).")
    p.add_argument("--model", type=str, default="llama2-7b",
                   help="Backbone tag from experiments.registry.MODELS.")
    p.add_argument("--pefts", type=str, nargs="+", default=None,
                   help="PEFT registry names for the training support (default: the SEEN group).")
    p.add_argument("--tasks", type=str, nargs="+",
                   default=["gsm8k", "humaneval", "mmlu", "tydiqa"])
    p.add_argument("--sketch-seed", type=int, default=0)
    p.add_argument("--q-hi-total", type=int, default=10000,
                   help="Total high-fidelity triple budget; phase 1 takes half.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--global-seed", type=int, default=0,
                   help="Must match the offline run (anchors / projections / clustering).")
    args = p.parse_args()

    log = get_logger("build_hi_fidelity")
    if args.model not in MODELS:
        raise SystemExit(f"unknown model {args.model!r}; known: {sorted(MODELS)}")
    spec = MODELS[args.model]
    layout = WorkDirLayout(args.workdir)
    cache = FeatureCache(layout.features)
    pool = JsonlPool.from_jsonl(args.pool)

    peft_names = args.pefts or [s.name for s in peft_specs_by_group("seen")]
    pefts = [resolve_peft(name, args.model) for name in peft_names]

    tasks = []
    for name in args.tasks:
        sk_path = layout.task / "sketches" / f"{name}_{args.sketch_seed}.json"
        if not sk_path.exists():
            raise SystemExit(f"missing sketch {sk_path}; run encode_task.py for {name} first.")
        sketch = load_sketch(sk_path)
        tasks.append(TaskConfig(name=name, task_id=sketch.task_id, sketch=sketch))

    cfg = OfflineConfig(
        selector_model=spec.selector_hf_id,
        n_layers_total=spec.n_layers,
        q_hi_total=args.q_hi_total,
        device=args.device,
        global_seed=args.global_seed,
    )
    anchors = AnchorRegistry(layout.root / "anchors")

    log.info(f"high-fidelity labeling: {len(pefts)} pefts × {len(tasks)} tasks, "
             f"Q_H={args.q_hi_total} (phase-1 half), backbone={spec.selector_hf_id}")
    _build_hi_labels(meta_pool=pool, pefts=pefts, tasks=tasks, anchors=anchors,
                     cache=cache, workdir=layout, cfg=cfg)
    log.info(f"wrote high-fidelity labels → {layout.labels / 'hi_fidelity.parquet'}")


if __name__ == "__main__":
    main()
