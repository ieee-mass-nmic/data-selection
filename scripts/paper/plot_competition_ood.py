#!/usr/bin/env python3
"""Plot positive performance drops for unseen PEFT configurations."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper" / "data" / "competition_ood_summary.json"
OUTPUT = ROOT / "paper" / "Figures" / "fig_ood_calibration.pdf"


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    groups = payload["groups"]
    modes = ["zero-shot", "cal200", "cal500"]
    labels = {"zero-shot": "zero-shot", "cal200": "cal-200", "cal500": "cal-500"}
    colors = {"zero-shot": "#4C78A8", "cal200": "#F58518", "cal500": "#54A24B"}

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    x = np.arange(len(groups), dtype=float)
    group_offsets = []
    for group in groups:
        available = [mode for mode in modes if group["modes"][mode] is not None]
        positions = np.linspace(-0.24, 0.24, len(available)) if len(available) > 1 else [0.0]
        group_offsets.append(dict(zip(available, positions)))
    fig, ax = plt.subplots(figsize=(3.35, 1.80))
    for mode in modes:
        xs = []
        drops = []
        for idx, group in enumerate(groups):
            value = group["modes"][mode]
            if value is None:
                continue
            xs.append(x[idx] + group_offsets[idx][mode])
            drops.append(-value["gap"])
        ax.bar(
            xs,
            drops,
            width=0.22,
            color=colors[mode],
            label=labels[mode],
        )

    ax.set_xticks(
        x,
        ["L0 near\n(LESS)", "L1 far\n(LESS)", "L2 BitFit\n(LESS)", "L2 Prefix/PT\n(RDS+)"],
    )
    ax.set_ylabel("Drop from reference (points)")
    ax.set_ylim(0.0, 8.2)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.legend(frameon=False, ncol=1, loc="upper left", handletextpad=0.4, labelspacing=0.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
