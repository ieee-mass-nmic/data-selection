"""E3 — ablations (design §4).

Two kinds of ablation:

  (1) Apply-time ablations — no retraining needed. Toggled here directly:
        - selection strategy: global_topk / uniform_cluster / adaptive   (axis G)
        - cluster α sweep                                                 (axis G)
        - uncertainty penalty λ_unc on/off                               (axis F)
  (2) Representation / condition ablations — need scorers trained offline with a
      component removed (−z_p, −z_t, lo-only, family-onehot, …). Pass them as
      `--scorer-variants tag=path tag=path`; each is scored as a `pcu` variant.
      (The offline pipeline must produce these; this script only consumes them.)

Run on the design's reduced grid (2 tasks, 2 representative PEFTs, budget=10%).

Example:
    python scripts/experiments/run_e3.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval \
        --tasks gsm8k humaneval --pefts L-r16-qkvo AD-b64 \
        --scorer-variants no_zp=runs/exp1/scorer/ckpt_no_zp.pt \
                          lo_only=runs/exp1/scorer/ckpt_lo_only.pt
"""

from __future__ import annotations

import argparse

from _common import RunContext, add_common_args, evaluate_selection, pcu_variant_select

from pcu_select.experiments import resolve_peft
from pcu_select.scorer.inference import ScorerInference

STRATEGIES = ["global_topk", "uniform_cluster", "adaptive"]
ALPHAS = [0.0, 0.3, 0.6, 0.9, 1.0]


def parse_variants(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        tag, _, path = it.partition("=")
        out[tag] = path
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="E3: ablations")
    add_common_args(p)
    p.add_argument("--pefts", type=str, nargs="+", default=["L-r16-qkvo", "AD-b64"])
    p.add_argument("--scorer-variants", type=str, nargs="*", default=[],
                   help="tag=ckpt_path entries for representation/condition ablations.")
    p.set_defaults(tasks=["gsm8k", "humaneval"])  # design §4.3 reduced grid (2 tasks)
    args = p.parse_args()
    variants = parse_variants(args.scorer_variants)

    ctx = RunContext(args, experiment="E3")
    for peft_name in args.pefts:
        peft = resolve_peft(peft_name, args.model)
        for task in args.tasks:
            tc = ctx.task(task)
            for budget in args.budgets:
                for seed in args.seeds:
                    # (1a) selection-strategy ablation (axis G)
                    for strat in STRATEGIES:
                        ids, mu = pcu_variant_select(ctx, peft, tc, budget, strategy=strat)
                        evaluate_selection(ctx, peft_name=peft_name, task_name=task,
                                           budget=budget, seed=seed, ids=ids,
                                           method_tag=f"pcu_strat={strat}", dense=mu,
                                           extra={"axis": "G_strategy"})
                    # (1b) cluster-α sweep (axis G)
                    for a in ALPHAS:
                        ids, mu = pcu_variant_select(ctx, peft, tc, budget,
                                                     strategy="adaptive", alpha=a)
                        evaluate_selection(ctx, peft_name=peft_name, task_name=task,
                                           budget=budget, seed=seed, ids=ids,
                                           method_tag=f"pcu_alpha={a}", dense=mu,
                                           extra={"axis": "G_alpha"})
                    # (1c) uncertainty penalty on/off (axis F)
                    for lam in (0.0, 0.2):
                        ids, mu = pcu_variant_select(ctx, peft, tc, budget, lambda_unc=lam)
                        evaluate_selection(ctx, peft_name=peft_name, task_name=task,
                                           budget=budget, seed=seed, ids=ids,
                                           method_tag=f"pcu_lambda={lam}", dense=mu,
                                           extra={"axis": "F_uncertainty"})
                    # (2) representation/condition ablations via alternate scorers
                    for tag, path in variants.items():
                        scorer = ScorerInference(path)
                        ids, mu = pcu_variant_select(ctx, peft, tc, budget, scorer=scorer)
                        evaluate_selection(ctx, peft_name=peft_name, task_name=task,
                                           budget=budget, seed=seed, ids=ids,
                                           method_tag=f"pcu_{tag}", dense=mu,
                                           extra={"axis": "condition"})
    print(f"E3 done → {ctx.results_path}")


if __name__ == "__main__":
    main()
