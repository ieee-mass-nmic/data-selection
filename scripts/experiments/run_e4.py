"""E4 — same PEFT family, different configurations (design §5).

The methodological core: does conditioning on the PEFT config actually change
which data is valuable? Three sub-experiments over a LoRA configuration sweep:

  E4-a  selection difference: pairwise Jaccard / top-k overlap of the subsets
        PCU selects per config — contrasted with a PEFT-agnostic baseline
        (rds_plus), whose overlap is ≈ 1 by construction. → E4_overlap.json
        (mechanism metric; computed on the first task / first seed only).
  E4-b  per-config performance: PCU vs Random / RDS+ / LESS on each config.
        → E4.jsonl  (one method row per task × config × method × seed)
  E4-c  mismatch matrix: train config j on the subset PCU selected FOR config i.
        Diagonal (correct conditioning) should beat off-diagonal. → E4.jsonl
        (rows tagged method="pcu_mismatch", extra.src_peft=i)

Per design §1.7 every (config × method × task × seed) cell repeats over
`--seeds` (default 0 1 2) so the tables can report mean ± std. Deterministic
selectors (pcu / rds_plus / less) pick the subset once and only re-train per
seed; random-style baselines re-select each seed.

Example:
    python scripts/experiments/run_e4.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval \
        --tasks gsm8k humaneval --seeds 0 1 2 \
        --configs L-r4-qv L-r8-qv L-r16-qkvo L-r32-qkvo L-r64-all \
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
# Methods whose selected subset depends on the seed; everything else is
# deterministic and selects once, varying only the target-train seed.
SEED_DEPENDENT = {"random", "balanced_random"}


def main() -> None:
    p = argparse.ArgumentParser(description="E4: same-PEFT-family configuration sensitivity")
    add_common_args(p)
    p.add_argument("--configs", type=str, nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--perf-methods", type=str, nargs="+",
                   default=["random", "rds_plus", "less", "pcu"])
    p.set_defaults(tasks=["gsm8k"])  # E4 defaults to the main task; pass more to replicate
    args = p.parse_args()
    budget = args.budgets[0]

    ctx = RunContext(args, experiment="E4")

    # ---- E4-a: selection-difference overlap (mechanism) on the first task/seed ----
    a_task, a_seed = args.tasks[0], args.seeds[0]
    tc0 = ctx.task(a_task)
    sel: dict[str, dict[str, list[str]]] = {"pcu": {}, "rds_plus": {}}
    for cfg in args.configs:
        for m in ("pcu", "rds_plus"):
            ids, _, _ = select(ctx, m, cfg, tc0, budget, a_seed)
            sel[m][cfg] = ids
    overlap = {
        m: {ci: {cj: jaccard(per_cfg[ci], per_cfg[cj]) for cj in args.configs}
            for ci in args.configs}
        for m, per_cfg in sel.items()
    }
    out = ctx.results_path.with_name("E4_overlap.json")
    out.write_text(json.dumps({"task": a_task, "budget": budget, "overlap": overlap}, indent=2))
    print(f"E4-a → {out}")

    # ---- E4-b / E4-c across tasks × seeds ----
    for task in args.tasks:
        tc = ctx.task(task)
        # E4-b: per-config performance.
        for cfg in args.configs:
            for method in args.perf_methods:
                base_ids, base_sec, base_dense = select(ctx, method, cfg, tc, budget, a_seed)
                for seed in args.seeds:
                    if method in SEED_DEPENDENT and seed != a_seed:
                        ids, sec, dense = select(ctx, method, cfg, tc, budget, seed)
                    else:
                        ids, sec, dense = base_ids, base_sec, base_dense
                    evaluate_selection(ctx, peft_name=cfg, task_name=task, budget=budget,
                                       seed=seed, ids=ids, method_tag=method, dense=dense,
                                       select_sec=sec, extra={"sub": "E4b"})
        # E4-c: mismatch matrix. PCU selection is deterministic, so select once
        # per source config, then train every target config on it across seeds.
        pcu_sel = {cfg: select(ctx, "pcu", cfg, tc, budget, a_seed)[0] for cfg in args.configs}
        for src, tgt in product(args.configs, args.configs):
            ids = pcu_sel[src]
            for seed in args.seeds:
                evaluate_selection(ctx, peft_name=tgt, task_name=task, budget=budget,
                                   seed=seed, ids=ids, method_tag="pcu_mismatch",
                                   extra={"sub": "E4c", "src_peft": src, "tgt_peft": tgt})
    print(f"E4 done → {ctx.results_path}")


if __name__ == "__main__":
    main()
