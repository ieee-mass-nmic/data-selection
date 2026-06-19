"""E4 figures: F5 (mismatch heatmap) + F6 (config-diff vs selection-diff).

    python scripts/plots/plot_e4.py --results runs/exp1/results/E4.jsonl \
        --overlap runs/exp1/results/E4_overlap.json --out-dir figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import load_df, savefig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--overlap", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = ap.parse_args()
    df = load_df(args.results)

    # ---- F5: mismatch matrix — train tgt on subset selected for src ----
    mm = df[df["method"] == "pcu_mismatch"].copy()
    if not mm.empty:
        configs = sorted(set(mm["src_peft"]) | set(mm["tgt_peft"]))
        M = np.full((len(configs), len(configs)), np.nan)
        idx = {c: i for i, c in enumerate(configs)}
        for _, r in mm.iterrows():
            M[idx[r["src_peft"]], idx[r["tgt_peft"]]] = r["metric"]
        # normalize each column by its diagonal to expose mismatch penalty
        Mn = M.copy()
        for j in range(len(configs)):
            diag = M[j, j]
            if not np.isnan(diag) and diag != 0:
                Mn[:, j] = M[:, j] / diag
        fig, ax = plt.subplots(figsize=(1.0 * len(configs) + 2, 1.0 * len(configs) + 1))
        im = ax.imshow(Mn, cmap="RdYlGn", vmin=np.nanmin(Mn), vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(configs)), configs, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(configs)), configs, fontsize=7)
        ax.set_xlabel("target config (trained on)")
        ax.set_ylabel("source config (selected for)")
        for i in range(len(configs)):
            for j in range(len(configs)):
                if not np.isnan(Mn[i, j]):
                    ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="metric / diagonal")
        ax.set_title("E4/F5: mismatch matrix (diagonal=1.0 → correct conditioning best)")
        savefig(fig, args.out_dir, "F5_mismatch.png")

    # ---- F6: config distance vs selection difference (1 - Jaccard) ----
    ov = json.loads(Path(args.overlap).read_text())["overlap"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, mat in ov.items():
        configs = list(mat.keys())
        xs, ys = [], []
        for i, ci in enumerate(configs):
            for cj in configs[i + 1:]:
                xs.append(_config_distance(ci, cj))
                ys.append(1.0 - mat[ci][cj])  # selection difference
        ax.scatter(xs, ys, label=method, alpha=0.7,
                   s=60 if method == "pcu" else 30)
    ax.set_xlabel("config distance (heuristic: rank/placement/lr changes)")
    ax.set_ylabel("selection difference (1 − Jaccard)")
    ax.set_title("E4/F6: config change ⇒ selection change (flat line = PEFT-agnostic)")
    ax.legend()
    savefig(fig, args.out_dir, "F6_config_vs_selection.png")


def _config_distance(a: str, b: str) -> int:
    """Crude ordinal distance between two registry config names for the x-axis."""
    def feats(name: str) -> tuple:
        import re
        rank = int(re.search(r"r(\d+)", name).group(1)) if re.search(r"r(\d+)", name) else 0
        place = "low" if "low" in name else "high" if "high" in name else "all"
        mods = "all" if "all" in name else "qkvo" if "qkvo" in name else "qv"
        return (rank, place, mods)
    fa, fb = feats(a), feats(b)
    d = abs(np.log2(max(fa[0], 1)) - np.log2(max(fb[0], 1)))
    d += 1.0 * (fa[1] != fb[1]) + 1.0 * (fa[2] != fb[2])
    return d


if __name__ == "__main__":
    main()
