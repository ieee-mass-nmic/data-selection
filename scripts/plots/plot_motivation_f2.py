"""Figure 2 — cross-PEFT transfer matrix (motivation §4).

Rows = target PEFT (trained), columns = source PEFT (data selected for). Cells
are row-normalized so the diagonal (correct conditioning) is 1.0 and the metric
is comparable across targets of different absolute capacity (§4.3):

    norm[i][j] = (perf[i][j] − random[i]) / (perf[i][i] − random[i])

Diagonal dominance (off-diagonal < 1) means mismatched data trains worse — the
downstream consequence of Figure 1's disagreement. A second heatmap uses the
PEFT-agnostic (RDS+) source: identical columns, NO diagonal structure (§4.4),
the foil that proves the dominance comes from PEFT-conditioned selection.

    python scripts/plots/plot_motivation_f2.py \
        --results runs/exp1/results/MOT_F2.jsonl --out-dir figs

matplotlib is optional (pip install -e ".[viz]").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import load_df

from pcu_select.experiments import PEFT_REGISTRY


def _mean_metric(df, mask) -> float:
    vals = df[mask]["metric"].to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def _heatmap(ax, M, pefts, title):
    finite = M[np.isfinite(M)]
    vlo = float(min(0.0, finite.min())) if len(finite) else 0.0
    im = ax.imshow(M, cmap="RdYlGn", vmin=vlo, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pefts)), pefts, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pefts)), pefts, fontsize=7)
    ax.set_xlabel("source PEFT (data selected for)", fontsize=8)
    ax.set_ylabel("target PEFT (trained on)", fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=9)
    return im


def _gap(M: np.ndarray) -> float:
    """Mean over rows of (diag − mean off-diagonal) = average mismatch penalty."""
    gaps = []
    for i in range(M.shape[0]):
        off = [M[i, j] for j in range(M.shape[1]) if j != i and np.isfinite(M[i, j])]
        if off and np.isfinite(M[i, i]):
            gaps.append(M[i, i] - float(np.mean(off)))
    return float(np.mean(gaps)) if gaps else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = ap.parse_args()

    df = load_df(args.results)
    tr = df[df["method"] == "transfer"]
    if tr.empty:
        raise SystemExit(f"no transfer rows in {args.results} — run run_motivation_transfer.py")
    pefts = [n for n in PEFT_REGISTRY if n in (set(tr["src_peft"]) | set(tr["tgt_peft"]))]
    P = len(pefts)
    idx = {p: i for i, p in enumerate(pefts)}

    # perf[i][j]: target i trained on source-j subset; random[i]: random baseline.
    perf = np.full((P, P), np.nan)
    for _, r in tr.iterrows():
        perf[idx[r["tgt_peft"]], idx[r["src_peft"]]] = _mean_metric(
            tr, (tr["tgt_peft"] == r["tgt_peft"]) & (tr["src_peft"] == r["src_peft"]))
    rnd = df[df["method"] == "transfer_random"]
    random_perf = np.array([_mean_metric(rnd, rnd["tgt_peft"] == p) for p in pefts])
    agn = df[df["method"] == "transfer_agnostic"]
    agn_perf = np.array([_mean_metric(agn, agn["tgt_peft"] == p) for p in pefts])

    # Row-normalize: diag → 1.0, random → 0.0 (§4.3).
    norm = np.full((P, P), np.nan)
    norm_agn = np.full((P, P), np.nan)
    for i in range(P):
        denom = perf[i, i] - random_perf[i]
        if not np.isfinite(denom) or abs(denom) < 1e-9:
            print(f"warn: target {pefts[i]} has diag≈random (denom={denom}); row left NaN")
            continue
        norm[i, :] = (perf[i, :] - random_perf[i]) / denom
        if np.isfinite(agn_perf[i]):
            norm_agn[i, :] = (agn_perf[i] - random_perf[i]) / denom  # constant across columns

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(2.2 * P + 2, 1.1 * P + 1.5))
    im0 = _heatmap(axes[0], norm, pefts,
                   f"PEFT-specific source (diag=1)\nGap={_gap(norm):.3f}")
    im1 = _heatmap(axes[1], norm_agn, pefts,
                   f"PEFT-agnostic source (RDS+, flat)\nGap={_gap(norm_agn):.3f}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="metric (row-normalized)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("F2: cross-PEFT transfer — diagonal dominance ⇒ data value is PEFT-dependent",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.out_dir / "F2_transfer.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")

    # raw matrix CSV for the appendix (§4.3)
    csv_path = args.out_dir / "F2_transfer_raw.csv"
    lines = ["target\\source," + ",".join(pefts) + ",random,agnostic"]
    for i, p in enumerate(pefts):
        cells = [f"{perf[i, j]:.4f}" if np.isfinite(perf[i, j]) else "" for j in range(P)]
        lines.append(f"{p}," + ",".join(cells) + f",{random_perf[i]:.4f},{agn_perf[i]:.4f}")
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"PEFT-specific Gap = {_gap(norm):.3f}   PEFT-agnostic Gap = {_gap(norm_agn):.3f} "
          f"(specific ≫ agnostic ⇒ conditioning matters)")


if __name__ == "__main__":
    main()
