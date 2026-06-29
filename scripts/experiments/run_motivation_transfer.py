"""Figure 2 — cross-PEFT transfer matrix (motivation §4).

Does the per-PEFT disagreement in Figure 1 have *downstream consequences*? Pick
a subset selected FOR source PEFT j, train TARGET PEFT i on it, and measure
quality. If data value is PEFT-dependent, the diagonal (i==j, correctly
conditioned) should beat the off-diagonal (mismatched) — "diagonal dominance".

Selection here is **non-PCU** (motivation §0.1): the source subset is chosen by
that source PEFT's own influence signal, never by the project's scorer.
  --select-signal less   : LESS-style per-PEFT influence (default; scales to the
                           full pool, needs only cached grad signatures).
  --select-signal u_hi   : the short-update truth, read from values.parquet
                           (restricted to the labeled estimation pool).

Controls written alongside the transfer cells:
  transfer_random   : random subset per target → row normalization baseline (§4.3).
  transfer_agnostic : RDS+ (PEFT-agnostic) source subset, identical for every j →
                      its matrix has NO diagonal structure (§4.4), the foil.

Reuses the E1–E5 plumbing in _common (RunContext / select / evaluate_selection),
so the train/eval/record path is identical to the main experiments. This mirrors
E4-c's mismatch matrix but with a non-PCU source signal. → results/MOT_F2.jsonl

Example:
    python scripts/experiments/run_motivation_transfer.py \
        --workdir runs/exp1 --pool data/pool_20k.jsonl --eval-dir data/eval \
        --tasks gsm8k humaneval --budgets 0.10 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _common import RunContext, TaskCtx, add_common_args, evaluate_selection, select
from _motivation import F2_PEFTS

from pcu_select.utils import get_logger

log = get_logger("motivation.transfer")
SEED0 = 0  # deterministic source selection happens once, on this seed


def _select_by_uhi(values: pd.DataFrame, peft_name: str, tc: TaskCtx, budget: float,
                   ) -> list[str]:
    """Top-`budget` ids by the source PEFT's u_hi truth (mean over anchor×seed)."""
    sub = values[(values["signal"] == "u_hi") & (values["peft_name"] == peft_name)
                 & (values["task_id"] == tc.task_id)]
    if sub.empty:
        raise SystemExit(f"no u_hi rows for peft={peft_name} task={tc.task_id} in --values; "
                         f"run build_motivation_values.py with --signals u_hi first.")
    agg = sub.groupby("sample_id")["value"].mean()
    k = max(1, int(round(budget * len(agg))))
    return list(agg.sort_values(ascending=False).index[:k])


def _source_select(ctx: RunContext, signal: str, values: pd.DataFrame | None,
                   src: str, tc: TaskCtx, budget: float) -> list[str]:
    if signal == "u_hi":
        assert values is not None
        return _select_by_uhi(values, src, tc, budget)
    ids, _, _ = select(ctx, "less", src, tc, budget, SEED0)  # LESS: per-PEFT, non-PCU
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description="Motivation Figure 2: cross-PEFT transfer matrix")
    add_common_args(p)
    p.add_argument("--pefts", type=str, nargs="+", default=F2_PEFTS)
    p.add_argument("--select-signal", type=str, default="less", choices=["less", "u_hi"],
                   help="Non-PCU source-selection signal (default: LESS).")
    p.add_argument("--values", type=Path, default=None,
                   help="values.parquet for --select-signal u_hi "
                        "(default: <workdir>/motivation/values.parquet).")
    p.set_defaults(tasks=["gsm8k", "humaneval"], results=None)
    args = p.parse_args()
    budget = args.budgets[0]

    ctx = RunContext(args, experiment="MOT_F2")
    values = None
    if args.select_signal == "u_hi":
        vpath = args.values or (ctx.layout.root / "motivation" / "values.parquet")
        if not Path(vpath).exists():
            raise SystemExit(f"--select-signal u_hi needs {vpath}; run build_motivation_values.py.")
        values = pd.read_parquet(vpath)

    for task in args.tasks:
        tc = ctx.task(task)
        # ---- source subsets (deterministic, selected once) ----
        src_sel = {src: _source_select(ctx, args.select_signal, values, src, tc, budget)
                   for src in args.pefts}
        agn_ids, _, _ = select(ctx, "rds_plus", args.pefts[0], tc, budget, SEED0)
        log.info(f"[{task}] selected {len(args.pefts)} source subsets "
                 f"(signal={args.select_signal}) + RDS+ agnostic control")

        for tgt in args.pefts:
            # row-normalization baseline: random subset trained on the target
            for seed in args.seeds:
                ids, sec, dense = select(ctx, "random", tgt, tc, budget, seed)
                evaluate_selection(ctx, peft_name=tgt, task_name=task, budget=budget, seed=seed,
                                   ids=ids, method_tag="transfer_random", dense=dense,
                                   select_sec=sec, extra={"tgt_peft": tgt})
            # transfer cells: train target on each source's subset
            for src in args.pefts:
                for seed in args.seeds:
                    evaluate_selection(ctx, peft_name=tgt, task_name=task, budget=budget,
                                       seed=seed, ids=src_sel[src], method_tag="transfer",
                                       extra={"src_peft": src, "tgt_peft": tgt,
                                              "select_signal": args.select_signal})
            # PEFT-agnostic control: same RDS+ subset for every source
            for seed in args.seeds:
                evaluate_selection(ctx, peft_name=tgt, task_name=task, budget=budget, seed=seed,
                                   ids=agn_ids, method_tag="transfer_agnostic",
                                   extra={"tgt_peft": tgt})
    print(f"motivation F2 done → {ctx.results_path}")


if __name__ == "__main__":
    main()
