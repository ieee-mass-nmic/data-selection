"""CLI: select a subset for a target PEFT/task using a trained scorer."""

from __future__ import annotations

import argparse
from pathlib import Path

from pcu_select.data import JsonlPool, load_sketch
from pcu_select.peft_space.schema import load_peft_config
from pcu_select.pipeline.apply import run_apply
from pcu_select.types import ApplyConfig, TaskConfig
from pcu_select.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--peft", type=Path, required=True)
    parser.add_argument("--sketch", type=Path, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--budget", type=float, default=0.1,
                        help="float ∈ (0,1] for fraction, or int for absolute count")
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--lambda-unc", type=float, default=0.2)
    parser.add_argument("--cluster-alpha", type=float, default=0.6)
    args = parser.parse_args()

    log = get_logger("select_subset")
    pool = JsonlPool.from_jsonl(args.pool)
    peft = load_peft_config(args.peft)
    sketch = load_sketch(args.sketch)
    task = TaskConfig(name=args.task_name, task_id=sketch.task_id, sketch=sketch)
    cfg = ApplyConfig(lambda_unc=args.lambda_unc, cluster_alpha=args.cluster_alpha)
    budget = args.budget if args.budget >= 1 else float(args.budget)
    ids = run_apply(
        candidate_pool=pool,
        peft_target=peft,
        task_target=task,
        budget=budget,
        scorer_ckpt=args.scorer,
        cfg=cfg,
        workdir=args.workdir,
    )
    log.info(f"selected {len(ids)} samples")


if __name__ == "__main__":
    main()
