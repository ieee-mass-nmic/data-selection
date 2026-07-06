"""E1 — different PEFTs: PCU-Select vs baselines (design §2).

For every SEEN PEFT, run every method on each main task/budget/seed and record
downstream performance + selection-quality metrics. Produces the big
method × PEFT × task table (T1) and the budget sensitivity curve (F1).

Example:
    python scripts/experiments/run_e1.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval \
        --model llama2-7b --budgets 0.05 0.10 0.30 --seeds 0 1 2
"""

from __future__ import annotations

import argparse

from _common import RunContext, add_common_args, run_cell

from pcu_select.experiments import peft_specs_by_group

# Baselines (design §2.3) + our method. `less` is the strongest per-PEFT rival.
# `lo_proxy_quota` is the simple-scorer control (reviewer 3.3): the raw
# site-weighted low-fidelity proxy fed through PCU-Select's cluster quota, so the
# PCU−proxy gap isolates what the learned scorer buys over the cheap proxy.
# IFD/S2L remain available when their exported scores are present in the cache,
# but they are not part of the default matrix unless those signals are produced.
METHODS = [
    "random", "balanced_random", "length", "loss", "perplexity", "embedding_nn",
    "rds_plus", "diversity", "grad_sim", "less", "lo_proxy_quota", "pcu",
]


def main() -> None:
    p = argparse.ArgumentParser(description="E1: PEFT × method comparison")
    add_common_args(p)
    p.add_argument("--methods", type=str, nargs="+", default=METHODS)
    p.add_argument("--pefts", type=str, nargs="+",
                   default=[s.name for s in peft_specs_by_group("seen")])
    args = p.parse_args()

    ctx = RunContext(args, experiment="E1")
    for peft_name in args.pefts:
        for task in args.tasks:
            for budget in args.budgets:
                for method in args.methods:
                    for seed in args.seeds:
                        run_cell(ctx, method=method, peft_name=peft_name,
                                 task_name=task, budget=budget, seed=seed)
    print(f"E1 done → {ctx.results_path}")


if __name__ == "__main__":
    main()
