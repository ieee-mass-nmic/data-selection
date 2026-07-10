#!/usr/bin/env python3
"""Generate the repaired amortized-cost figure for the competition manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data" / "competition_cost_model.json"
OUT = ROOT / "paper" / "Figures" / "fig_cost_break_even.pdf"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )


def main() -> None:
    setup_style()
    model = json.loads(DATA.read_text())
    t_values = np.array(model["t_values"], dtype=float)
    methods = {row["method"]: row for row in model["methods"]}
    order = ["LESS", "PCU-Select", "Influence", "RDS+", "Random"]
    colors = {
        "PCU-Select": "#1f77b4",
        "LESS": "#d62728",
        "Influence": "#2ca02c",
        "RDS+": "#9467bd",
        "Random": "#7f7f7f",
    }
    markers = {
        "PCU-Select": "o",
        "LESS": "s",
        "Influence": "^",
        "RDS+": "D",
        "Random": "x",
    }

    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    for name in order:
        row = methods[name]
        y = row["offline_gpu_h"] + t_values * row["per_configuration_selection_gpu_h"]
        ax.plot(
            t_values,
            y,
            marker=markers[name],
            linewidth=2.0 if name == "PCU-Select" else 1.1,
            markersize=3.5,
            color=colors[name],
            label=name,
        )

    pcu = methods["PCU-Select"]
    less = methods["LESS"]
    t_star = pcu["offline_gpu_h"] / (
        less["per_configuration_selection_gpu_h"]
        - pcu["per_configuration_selection_gpu_h"]
    )
    ax.axvline(t_star, color="0.35", linestyle=":", linewidth=0.8)
    ax.text(t_star + 0.18, ax.get_ylim()[1] * 0.56, f"T*={t_star:.1f}", fontsize=7)

    ax.set_xlabel("target PEFT configurations served")
    ax.set_ylabel("cumulative selection GPU-hours")
    ax.set_xlim(min(t_values), max(t_values))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
