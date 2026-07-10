#!/usr/bin/env python3
"""Generate the PCU-Select architecture figure.

This is a manual pipeline diagram rather than a data plot. It replaces the
stale raster figure and keeps the PEFT-code dimension and support-tier policy
consistent with the executable pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "Figures" / "Architecture.pdf"
PNG_OUT = ROOT / "paper" / "Figures" / "Architecture.png"


BLUE = "#1f5aa6"
TEAL = "#187b7b"
ORANGE = "#c45a1a"
PURPLE = "#6b4fa3"
GRAY = "#555555"
LIGHT_BLUE = "#eef5ff"
LIGHT_TEAL = "#edf8f7"
LIGHT_ORANGE = "#fff3eb"
LIGHT_GRAY = "#f6f6f6"


def box(ax, xy, wh, text, edge=BLUE, face="white", fontsize=7.0, lw=1.1):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color=GRAY, lw=1.0, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    box(ax, (0.1, 0.2), (11.25, 8.55), "", edge=BLUE, face="none", lw=1.2)
    box(ax, (11.75, 0.2), (4.15, 8.55), "", edge=TEAL, face="none", lw=1.2)
    ax.plot([11.55, 11.55], [0.35, 8.55], color="#999999", linestyle="--", linewidth=0.8)
    ax.text(5.72, 8.34, "Offline supervision", color=BLUE, ha="center", fontsize=10, weight="bold")
    ax.text(13.83, 8.34, r"Online selection", color=TEAL, ha="center", fontsize=10, weight="bold")
    ax.text(5.72, 8.04, "once per backbone family", color=BLUE, ha="center", fontsize=7.2)
    ax.text(13.83, 8.04, r"per target $p^*,t^*$", color=TEAL, ha="center", fontsize=7.2)

    # Inputs.
    box(ax, (0.35, 6.65), (1.25, 0.7), "Candidate\npool", BLUE, LIGHT_BLUE)
    box(ax, (0.35, 4.5), (1.25, 0.7), "PEFT\nregistry", TEAL, LIGHT_TEAL)
    box(ax, (0.35, 2.35), (1.25, 0.7), "Task\nsketches", BLUE, LIGHT_BLUE)
    box(ax, (2.0, 6.65), (1.15, 0.7), "Feature\nextract", GRAY, LIGHT_GRAY)
    box(ax, (2.0, 4.5), (1.15, 0.7), "PEFT\nencoder", GRAY, LIGHT_GRAY)
    box(ax, (2.0, 2.35), (1.15, 0.7), "Sketch\nencoder", GRAY, LIGHT_GRAY)
    for y in (7.075, 4.875, 2.675):
        arrow(ax, (1.6, y), (2.0, y))

    box(ax, (3.55, 6.25), (2.35, 1.05), "$z_x$ sample\n848 = 768+16+64", BLUE, LIGHT_BLUE, fontsize=6.2)
    box(ax, (3.55, 4.18), (2.35, 1.18), "$z_p$ PEFT code\n128 dims\n96 mask + 16 cap\n16 recipe", TEAL, LIGHT_TEAL, fontsize=5.8)
    box(ax, (3.55, 2.15), (2.35, 1.05), "$z_t$ task\n848 + site grads", BLUE, LIGHT_BLUE, fontsize=6.2)
    for y in (7.075, 4.875, 2.675):
        arrow(ax, (3.15, y), (3.55, y))

    # Shared intervention-site space.
    box(ax, (6.7, 2.0), (1.8, 5.55), "", GRAY, "#fafafa")
    ax.text(7.6, 7.2, "24 sites", ha="center", fontsize=7.5, weight="bold")
    ax.text(7.6, 6.9, "8 layers x 3", ha="center", fontsize=6.5)
    modules = ["attn_out", "mlp_out", "block"]
    for i in range(8):
        y = 2.35 + i * 0.53
        ax.text(6.96, y + 0.12, str(i + 1), fontsize=6.3, color=GRAY)
        for j, m in enumerate(modules):
            box(ax, (7.18 + j * 0.35, y), (0.3, 0.24), "", GRAY, "#e8eef7" if j == 0 else "#eaf7f5" if j == 1 else "#eeeeee", fontsize=4.0, lw=0.5)
    ax.text(7.6, 1.55, "shared coordinates\nsamples / PEFTs / tasks", ha="center", fontsize=6.2)
    arrow(ax, (5.9, 6.85), (6.7, 6.85))
    arrow(ax, (5.9, 4.8), (6.7, 4.8))
    arrow(ax, (5.9, 2.65), (6.7, 2.65))

    # Labels and scorer.
    box(ax, (9.0, 6.15), (1.55, 1.0), "Low-fidelity\nsite-weighted\nproxy", BLUE, LIGHT_BLUE, fontsize=6.6)
    box(ax, (9.0, 4.45), (1.55, 1.15), "High-fidelity\nshort-horizon\nlabels", ORANGE, LIGHT_ORANGE, fontsize=6.6)
    box(ax, (9.0, 2.0), (1.55, 1.25), "Train scorer\nrank + reg\n+ uncertainty", PURPLE, "#f3effb", fontsize=6.6)
    arrow(ax, (8.5, 6.7), (9.0, 6.65))
    arrow(ax, (8.5, 4.8), (9.0, 5.0))
    arrow(ax, (9.78, 6.15), (9.78, 5.6), ORANGE)
    arrow(ax, (9.78, 4.45), (9.78, 3.25), PURPLE)
    ax.text(10.15, 3.55, "multi-fidelity\nlabels", fontsize=6.3, ha="center")

    # Online pipeline.
    online = [
        ("Target PEFT + task", 7.45),
        ("Encode $z_{p^*},z_{t^*}$", 6.55),
        ("Support-tier check\nnear / far / new family", 5.55),
        ("Score pool\n$(\\hat\\mu,\\hat\\sigma)$", 4.45),
        ("Conservative score\n$q=\\hat\\mu-\\lambda\\hat\\sigma$", 3.35),
        ("Cluster + quotas", 2.25),
        ("Top-$b_k$ per cluster", 1.35),
        ("Selected subset", 0.55),
    ]
    prev = None
    for i, (label, y) in enumerate(online):
        edge = ORANGE if i in (4, 7) else TEAL
        face = LIGHT_ORANGE if i in (4, 7) else LIGHT_TEAL
        box(ax, (12.25, y), (3.05, 0.55), label, edge, face, fontsize=6.8)
        if prev is not None:
            arrow(ax, (13.775, prev), (13.775, y + 0.55), TEAL)
        prev = y
    box(ax, (15.48, 5.72), (0.3, 0.42), "cal.", GRAY, "white", fontsize=5.2, lw=0.8)
    arrow(ax, (15.48, 5.92), (15.25, 5.78), GRAY, lw=0.8, style="<|-")
    arrow(ax, (10.55, 2.62), (12.25, 4.72), PURPLE, lw=1.1)
    ax.text(11.25, 3.45, "reused", fontsize=6.2, ha="center", color=PURPLE)

    box(ax, (0.2, -0.02), (15.55, 0.22), "Offline cost is cached and amortized; online selection uses one scorer pass plus clustering.", GRAY, LIGHT_GRAY, fontsize=7.8, lw=0.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PNG_OUT, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {OUT} and {PNG_OUT}")


if __name__ == "__main__":
    main()
