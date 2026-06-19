"""E5 figures: F7 (level × mode bars) + F8 (Mahalanobis d² vs degradation).

    python scripts/plots/plot_e5.py --results runs/exp1/results/E5.jsonl --out-dir figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import load_df, savefig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = ap.parse_args()
    df = load_df(args.results)

    pcu = df[df["method"].str.startswith("pcu_")].copy()
    pcu["mode"] = pcu["method"].str.replace("pcu_", "", regex=False)

    # ---- F7: grouped bars, level × mode (zeroshot / cal200 / cal500) ----
    levels = ["L0", "L1", "L2"]
    modes = ["zeroshot", "cal200", "cal500"]
    present = [m for m in modes if (pcu["mode"] == m).any()]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(1, len(present))
    x = np.arange(len(levels))
    for k, mode in enumerate(present):
        means = [pcu[(pcu["level"] == lv) & (pcu["mode"] == mode)]["metric"].mean()
                 for lv in levels]
        ax.bar(x + k * width, means, width, label=mode)
    # reference baselines (LESS / RDS+) per level
    for m, c in (("less", "k"), ("rds_plus", "grey")):
        ref = df[df["method"] == m]
        if not ref.empty:
            ys = [ref[ref["level"] == lv]["metric"].mean() for lv in levels]
            ax.plot(x + width, ys, marker="D", linestyle="--", color=c, label=m)
    ax.set_xticks(x + width, levels)
    ax.set_xlabel("OOD level (L0 interp · L1 extrap · L2 unseen family)")
    ax.set_ylabel("downstream metric")
    ax.set_title("E5/F7: zero-shot vs calibration across OOD levels")
    ax.legend(fontsize=7)
    savefig(fig, args.out_dir, "F7_levels_modes.png")

    # ---- F8: d² vs zero-shot degradation relative to LESS upper bound ----
    zs = pcu[pcu["mode"] == "zeroshot"].copy()
    less = df[df["method"] == "less"].groupby("peft")["metric"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys, labels, colors = [], [], [], []
    for _, r in zs.iterrows():
        ub = less.get(r["peft"], np.nan)
        if np.isnan(ub) or ub == 0:
            continue
        xs.append(r["d2"])
        ys.append(1.0 - r["metric"] / ub)
        labels.append(r["peft"])
        colors.append("crimson" if r["is_ood"] else "steelblue")
    ax.scatter(xs, ys, c=colors)
    for xi, yi, lb in zip(xs, ys, labels):
        ax.annotate(lb, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    if not zs.empty and "ood_threshold" in zs.iloc[0]["extra"]:
        ax.axvline(zs.iloc[0]["extra"]["ood_threshold"], color="grey", ls=":",
                   label="OOD threshold τ")
        ax.legend()
    ax.set_xlabel("Mahalanobis d²(z_p)")
    ax.set_ylabel("zero-shot degradation vs LESS  (1 − metric/LESS)")
    ax.set_title("E5/F8: d² predicts when calibration is needed")
    savefig(fig, args.out_dir, "F8_d2_vs_degradation.png")


if __name__ == "__main__":
    main()
