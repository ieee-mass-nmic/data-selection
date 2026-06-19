"""E1 figures: T1 (method × PEFT table) + F1 (budget sensitivity).

    python scripts/plots/plot_e1.py --results runs/exp1/results/E1.jsonl --out-dir figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import METHOD_LABELS, load_df, method_label, savefig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument("--budget", type=float, default=0.10)
    args = ap.parse_args()
    df = load_df(args.results)

    # ---- T1: method × PEFT mean metric at the main budget (mean over tasks/seeds) ----
    main = df[np.isclose(df["budget"], args.budget)]
    t1 = main.pivot_table(index="method", columns="peft", values="metric", aggfunc="mean")
    # order rows so PCU is last/highlighted
    order = [m for m in METHOD_LABELS if m in t1.index]
    t1 = t1.reindex(order)
    t1.to_csv(args.out_dir / "T1_method_x_peft.csv")
    print(f"wrote {args.out_dir / 'T1_method_x_peft.csv'}\n{t1.round(4)}")

    fig, ax = plt.subplots(figsize=(1.2 * len(t1.columns) + 3, 0.45 * len(t1) + 1))
    im = ax.imshow(t1.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(t1.columns)), t1.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(t1.index)), [method_label(m) for m in t1.index])
    for i in range(len(t1.index)):
        for j in range(len(t1.columns)):
            v = t1.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label="downstream metric")
    ax.set_title(f"E1: method × PEFT (budget={args.budget:.0%})")
    savefig(fig, args.out_dir, "T1_method_x_peft.png")

    # ---- F1: metric vs budget, averaged over PEFT & task ----
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in [m for m in METHOD_LABELS if m in df["method"].unique()]:
        sub = df[df["method"] == method]
        g = sub.groupby("budget")["metric"].agg(["mean", "std"]).reset_index()
        lw = 2.5 if method == "pcu" else 1.0
        ax.errorbar(g["budget"], g["mean"], yerr=g["std"], marker="o",
                    label=method_label(method), linewidth=lw, capsize=2)
    ax.set_xlabel("budget (fraction of pool)")
    ax.set_ylabel("downstream metric")
    ax.set_title("E1/F1: budget sensitivity")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, args.out_dir, "F1_budget_sensitivity.png")


if __name__ == "__main__":
    main()
