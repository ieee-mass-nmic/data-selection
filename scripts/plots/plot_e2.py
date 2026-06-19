"""E2 figures: F2 (break-even curve) + F3 (performance-vs-cost Pareto).

Total cost model (design §3.2):
  PCU-Select:     C_offline + T · C_apply
  Influence (LESS/Influence): T · (C_recompute + C_apply_like)
  Cheap baselines (RDS+/Random): T · C_apply_like   (offline ≈ 0)

where per-PEFT C_apply / C_apply_like = mean(select_gpu_h + target_train_gpu_h)
over the PEFTs run, C_recompute comes from E2_cost_model.json, and C_offline is
the summed offline stages for PCU.

    python scripts/plots/plot_e2.py --results runs/exp1/results/E2.jsonl \
        --cost-model runs/exp1/results/E2_cost_model.json --out-dir figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import load_df, method_label, savefig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--cost-model", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = ap.parse_args()
    df = load_df(args.results)
    cm = json.loads(Path(args.cost_model).read_text())
    offline = cm["offline_gpu_h"]
    recompute = cm["per_peft_recompute_gpu_h"]
    influence = set(cm.get("influence_methods", []))
    T_vals = np.array(cm.get("T_values", [1, 3, 5, 10]))

    # per-PEFT apply cost (mean over PEFTs) and mean performance per method
    df["apply_h"] = df["select_gpu_h"].fillna(0) + df["target_train_gpu_h"].fillna(0)
    per_method = df.groupby("method").agg(apply_h=("apply_h", "mean"),
                                          perf=("metric", "mean")).reset_index()

    def total_cost(method: str, apply_h: float, T: np.ndarray) -> np.ndarray:
        if method == "pcu":
            return offline + T * apply_h
        if method in influence:
            return T * (recompute + apply_h)
        return T * apply_h

    # ---- F2: total GPU-h vs T, with break-even markers ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    pcu_apply = float(per_method.loc[per_method.method == "pcu", "apply_h"].iloc[0]) \
        if (per_method.method == "pcu").any() else 0.0
    for _, r in per_method.iterrows():
        y = total_cost(r["method"], r["apply_h"], T_vals)
        lw = 2.5 if r["method"] == "pcu" else 1.2
        ax.plot(T_vals, y, marker="o", label=method_label(r["method"]), linewidth=lw)
    # break-even vs each influence/cheap method (where its line crosses PCU's)
    for _, r in per_method.iterrows():
        if r["method"] == "pcu":
            continue
        c_specific = (recompute + r["apply_h"]) if r["method"] in influence else r["apply_h"]
        denom = c_specific - pcu_apply
        if denom > 1e-9:
            t_star = offline / denom
            ax.axvline(t_star, color="grey", ls=":", lw=0.8)
            ax.text(t_star, ax.get_ylim()[1] * 0.9, f"T*={t_star:.1f}\nvs {method_label(r['method'])}",
                    fontsize=6, rotation=90, va="top")
    ax.set_xlabel("number of target PEFTs served (T)")
    ax.set_ylabel("total GPU-hours")
    ax.set_title("E2/F2: amortized cost & break-even")
    ax.legend(fontsize=7)
    savefig(fig, args.out_dir, "F2_break_even.png")

    # ---- F3: performance vs total cost at T=max (Pareto) ----
    T_max = int(T_vals.max())
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in per_method.iterrows():
        x = float(total_cost(r["method"], r["apply_h"], np.array([T_max]))[0])
        ax.scatter(x, r["perf"], s=90 if r["method"] == "pcu" else 45)
        ax.annotate(method_label(r["method"]), (x, r["perf"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"total GPU-hours @ T={T_max}")
    ax.set_ylabel("mean downstream metric (across PEFTs)")
    ax.set_title("E2/F3: performance vs total cost")
    savefig(fig, args.out_dir, "F3_pareto.png")


if __name__ == "__main__":
    main()
