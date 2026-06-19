"""E3 figures: T2 (ablation table) + F4 (α / strategy curves).

    python scripts/plots/plot_e3.py --results runs/exp1/results/E3.jsonl --out-dir figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _common import load_df, savefig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = ap.parse_args()
    df = load_df(args.results)

    # ---- T2: per-variant mean metric + NDCG, sorted ----
    t2 = df.groupby("method").agg(metric=("metric", "mean"), ndcg=("ndcg_at_k", "mean"),
                                  n=("metric", "size")).sort_values("metric")
    t2.to_csv(args.out_dir / "T2_ablation.csv")
    print(f"wrote {args.out_dir / 'T2_ablation.csv'}\n{t2.round(4)}")

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(t2) + 1))
    ax.barh(range(len(t2)), t2["metric"])
    ax.set_yticks(range(len(t2)), t2.index, fontsize=7)
    ax.set_xlabel("downstream metric (higher = better)")
    ax.set_title("E3/T2: ablation — each removed component should drop the metric")
    savefig(fig, args.out_dir, "T2_ablation.png")

    # ---- F4: cluster-α sweep (axis G) ----
    alpha = df[df["axis"] == "G_alpha"].copy()
    if not alpha.empty:
        alpha["alpha"] = alpha["method"].str.extract(r"alpha=([0-9.]+)").astype(float)
        g = alpha.groupby("alpha")["metric"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.errorbar(g["alpha"], g["mean"], yerr=g["std"], marker="o", capsize=3)
        ax.set_xlabel(r"cluster quota skew $\alpha$ (0=coverage, 1=utility)")
        ax.set_ylabel("downstream metric")
        ax.set_title("E3/F4: adaptive-cluster α sweep")
        savefig(fig, args.out_dir, "F4_alpha_sweep.png")

    # ---- F4b: selection strategy (axis G) ----
    strat = df[df["axis"] == "G_strategy"].copy()
    if not strat.empty:
        strat["strategy"] = strat["method"].str.replace("pcu_strat=", "", regex=False)
        g = strat.groupby("strategy")["metric"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(g["strategy"], g["mean"], yerr=g["std"], capsize=3)
        ax.set_ylabel("downstream metric")
        ax.set_title("E3/F4b: selection strategy")
        savefig(fig, args.out_dir, "F4b_strategy.png")


if __name__ == "__main__":
    main()
