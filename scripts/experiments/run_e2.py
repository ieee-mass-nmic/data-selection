"""E2 — cross-PEFT transfer / amortized cost (design §3).

Records what the break-even curve (F2) and performance-vs-cost Pareto (F3) need:

  - PCU-Select pays a one-time offline cost (feat + lo + hi + scorer_train,
    read from cost/accounting.jsonl) and a cheap per-PEFT apply cost.
  - Influence baselines (`less`, `grad_sim`) recompute a per-PEFT gradient/
    feature signal for EVERY target PEFT; that recompute is charged via
    `--per-peft-recompute-h` (PCU-Select amortizes it into the offline stage).

Per-(method, PEFT) selection/train costs land in `E2.jsonl` (normal result
rows). The global cost model (offline total, per-PEFT recompute) is written to
`E2_cost_model.json`. plot_e2.py combines them into T* and the curves.

Example:
    python scripts/experiments/run_e2.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval \
        --methods pcu less grad_sim rds_plus random --per-peft-recompute-h 1.5
"""

from __future__ import annotations

import argparse
import json

from _common import RunContext, add_common_args, run_cell

from pcu_select.cost.accounting import CostAccountant
from pcu_select.experiments import peft_specs_by_group

OFFLINE_STAGES = {"feat", "lo", "hi", "scorer_train"}
INFLUENCE_METHODS = ["less", "grad_sim"]


def offline_gpu_h(workdir) -> float:
    acc = CostAccountant(workdir / "cost" / "accounting.jsonl")
    return sum(e.gpu_hours for e in acc.read_events() if e.stage in OFFLINE_STAGES)


def main() -> None:
    p = argparse.ArgumentParser(description="E2: multi-PEFT total-cost comparison")
    add_common_args(p)
    p.add_argument("--methods", type=str, nargs="+",
                   default=["pcu", "less", "grad_sim", "rds_plus", "random"])
    p.add_argument("--pefts", type=str, nargs="+",
                   default=[s.name for s in peft_specs_by_group("seen")]
                   + ["L-r32-qkvo", "L-r8-lowlayers"])
    p.add_argument("--per-peft-recompute-h", type=float, default=1.5,
                   help="Per-PEFT gradient/feature recompute charged to influence baselines.")
    args = p.parse_args()
    args.tasks = args.tasks[:1]  # one task / seed suffices for cost
    args.seeds = args.seeds[:1]

    ctx = RunContext(args, experiment="E2")
    for peft_name in args.pefts:
        for method in args.methods:
            run_cell(ctx, method=method, peft_name=peft_name,
                     task_name=args.tasks[0], budget=args.budgets[0], seed=args.seeds[0])

    cost_model = {
        "offline_gpu_h": offline_gpu_h(args.workdir),
        "per_peft_recompute_gpu_h": args.per_peft_recompute_h,
        "influence_methods": INFLUENCE_METHODS,
        "T_values": [1, 3, 5, 10],
    }
    out = ctx.results_path.with_name("E2_cost_model.json")
    out.write_text(json.dumps(cost_model, indent=2))
    print(f"E2 done → {ctx.results_path}; cost model → {out} "
          f"(offline={cost_model['offline_gpu_h']:.3f} GPU-h)")


if __name__ == "__main__":
    main()
