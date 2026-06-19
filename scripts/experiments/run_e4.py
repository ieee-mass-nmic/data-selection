"""E4 — same PEFT family, different configurations (design §5).

The methodological core: does conditioning on the PEFT config actually change
which data is valuable? Three sub-experiments over a LoRA configuration sweep:

  E4-a  selection difference: pairwise Jaccard / top-k overlap of the subsets
        PCU selects per config — contrasted with a PEFT-agnostic baseline
        (rds_plus), whose overlap is ≈ 1 by construction. → E4_overlap.json
  E4-b  per-config performance: PCU vs Random / RDS+ / LESS on each config.
        → E4.jsonl  (method rows)
  E4-c  mismatch matrix: train config j on the subset PCU selected FOR config i.
        Diagonal (correct conditioning) should beat off-diagonal. → E4.jsonl
        (rows tagged method="pcu_mismatch", extra.src_peft=i)

Example:
    python scripts/experiments/run_e4.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval \
        --task gsm8k --configs L-r4-qv L-r8-qv L-r16-qkvo L-r32-qkvo L-r64-all \
                                L-r8-lowlayers L-r8-highlayers
"""

from __future__ import annotations

import argparse
import json
from itertools import product

from _common import RunContext, add_common_args, evaluate_selection, select

from pcu_select.eval.metrics import jaccard

DEFAULT_CONFIGS = ["L-r4-qv", "L-r8-qv", "L-r16-qkvo", "L-r32-qkvo", "L-r64-all",
                   "L-r8-lowlayers", "L-r8-highlayers"]


def main() -> None:
    p = argparse.ArgumentParser(description="E4: same-PEFT-family configuration sensitivity")
    add_common_args(p)
    p.add_argument("--configs", type=str, nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--task", type=str, default="gsm8k")
    p.add_argument("--perf-methods", type=str, nargs="+",
                   default=["random", "rds_plus", "less", "pcu"])
    args = p.parse_args()
    budget, seed = args.budgets[0], args.seeds[0]

    ctx = RunContext(args, experiment="E4")
    tc = ctx.task(args.task)

    # ---- gather per-config selections for pcu and the agnostic baseline ----
    sel: dict[str, dict[str, list[str]]] = {"pcu": {}, "rds_plus": {}}
    for cfg in args.configs:
        for m in ("pcu", "rds_plus"):
            ids, _, _ = select(ctx, m, cfg, tc, budget, seed)
            sel[m][cfg] = ids

    # ---- E4-a: pairwise overlap matrices ----
    overlap = {}
    for m, per_cfg in sel.items():
        mat = {ci: {cj: jaccard(per_cfg[ci], per_cfg[cj]) for cj in args.configs}
               for ci in args.configs}
        overlap[m] = mat
    out = ctx.results_path.with_name("E4_overlap.json")
    out.write_text(json.dumps({"task": args.task, "budget": budget, "overlap": overlap}, indent=2))
    print(f"E4-a → {out}")

    # ---- E4-b: per-config performance ----
    for cfg in args.configs:
        for method in args.perf_methods:
            ids, sec, dense = select(ctx, method, cfg, tc, budget, seed)
            evaluate_selection(ctx, peft_name=cfg, task_name=args.task, budget=budget,
                               seed=seed, ids=ids, method_tag=method, dense=dense,
                               select_sec=sec, extra={"sub": "E4b"})

    # ---- E4-c: mismatch matrix (train config j on subset chosen for config i) ----
    for src, tgt in product(args.configs, args.configs):
        ids = sel["pcu"][src]
        evaluate_selection(ctx, peft_name=tgt, task_name=args.task, budget=budget,
                           seed=seed, ids=ids, method_tag="pcu_mismatch",
                           extra={"sub": "E4c", "src_peft": src, "tgt_peft": tgt})
    print(f"E4 done → {ctx.results_path}")


if __name__ == "__main__":
    main()
