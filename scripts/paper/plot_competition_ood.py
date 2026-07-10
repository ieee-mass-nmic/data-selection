#!/usr/bin/env python3
"""Plot the paired unseen-configuration gaps from the canonical JSON summary."""

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
            "font.size": 8,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    x = np.arange(len(groups), dtype=float)
    width = 0.23
    fig, ax = plt.subplots(figsize=(3.35, 2.30))
    for offset, mode in zip((-1, 0, 1), modes):
        first = True
        for idx, group in enumerate(groups):
            value = group["modes"][mode]
            if value is None:
                continue
            ax.bar(
                x[idx] + offset * width,
                value["gap"],
                width,
                yerr=value["std"],
                capsize=2,
                color=colors[mode],
                label=labels[mode] if first else None,
            )
            first = False

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(
        x,
        ["near\nLESS", "far\nLESS", "BitFit\nLESS", "Prefix/PT\nRDS+"],
    )
    ax.set_ylabel("gap to reference (points)")
    ax.set_ylim(-8.2, 0.9)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
