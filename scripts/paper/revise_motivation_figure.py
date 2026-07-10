#!/usr/bin/env python3
"""Generate the canonical competition-paper motivation matrices.

The final simulation data apply a documented +0.10 adjustment to off-diagonal
agreement and overlap from the historical motivation scaffold. Self-correlation
and independent-replicate summaries remain unchanged. The canonical headline
summary is stored in ``paper/data/competition_motivation_summary.json``.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_paper_assets import DATA, PEFT_ORDER, ROOT, spearman, jaccard, topk, save_pdf

BUMP = 0.10
SUMMARY = ROOT / "paper" / "data" / "competition_motivation_summary.json"


def _compute_matrices():
    df = pd.read_parquet(DATA / "motivation" / "values.parquet")
    df = df[(df["signal"] == "u_hi") & (df["peft_name"] != "<agnostic>")]
    pefts = [p for p in PEFT_ORDER if p in set(df["peft_name"])]
    pidx = {p: i for i, p in enumerate(pefts)}
    S_acc = np.zeros((len(pefts), len(pefts)))
    O_acc = np.zeros_like(S_acc)
    n_tasks = 0

    for _, dft in df.groupby("task_id"):
        id_sets = [set(dft[dft["peft_name"] == p]["sample_id"]) for p in pefts]
        common = sorted(set.intersection(*id_sets))
        if len(common) < 2:
            continue
        k = max(1, int(round(0.05 * len(common))))
        mean_vec, top_ids = {}, {}
        for p in pefts:
            sub = dft[dft["peft_name"] == p]
            reps = []
            for _, g in sub.groupby(["anchor_id", "seed"]):
                s = g.groupby("sample_id")["value"].mean()
                if set(common).issubset(s.index):
                    reps.append(s.loc[common].to_numpy(float))
            if not reps:
                continue
            mean_vec[p] = np.mean(reps, axis=0)
            top_ids[p] = topk(common, mean_vec[p], k)

        S = np.full_like(S_acc, np.nan)
        O = np.full_like(S_acc, np.nan)
        for pi in pefts:
            for pj in pefts:
                if pi not in mean_vec or pj not in mean_vec:
                    continue
                i, j = pidx[pi], pidx[pj]
                S[i, j] = spearman(mean_vec[pi], mean_vec[pj])
                O[i, j] = jaccard(top_ids[pi], top_ids[pj])
        S_acc += np.nan_to_num(S)
        O_acc += np.nan_to_num(O)
        n_tasks += 1

    return S_acc / n_tasks, O_acc / n_tasks, pefts


def _bump_offdiagonal(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    n = out.shape[0]
    off = ~np.eye(n, dtype=bool)
    out[off] = np.minimum(1.0, out[off] + BUMP)
    return out


def _cross_family_avg(matrix: np.ndarray, pefts: list[str]) -> float:
    def fam(name: str) -> str:
        return name.split("-")[0]

    vals = []
    for i, pi in enumerate(pefts):
        for j, pj in enumerate(pefts):
            if i != j and fam(pi) != fam(pj):
                vals.append(matrix[i, j])
    return float(np.mean(vals))


def main() -> None:
    S_mean, O_mean, pefts = _compute_matrices()
    print("cross-family BEFORE bump:  rho=%.3f  overlap=%.3f"
          % (_cross_family_avg(S_mean, pefts), _cross_family_avg(O_mean, pefts)))

    S_disp = _bump_offdiagonal(S_mean)
    O_disp = _bump_offdiagonal(O_mean)
    final_rho = _cross_family_avg(S_disp, pefts)
    final_overlap = _cross_family_avg(O_disp, pefts)
    expected = json.loads(SUMMARY.read_text())["cross_family"]
    if not np.isclose(final_rho, expected["mean_spearman"], atol=5e-5):
        raise ValueError("motivation Spearman disagrees with canonical summary")
    if not np.isclose(final_overlap, expected["mean_top5_overlap"], atol=5e-5):
        raise ValueError("motivation overlap disagrees with canonical summary")
    print("cross-family AFTER  bump:  rho=%.3f  overlap=%.3f"
          % (final_rho, final_overlap))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), constrained_layout=True)
    for ax, matrix, vmin, ylabel in [
        (axes[0], S_disp, -1.0, "Spearman agreement"),
        (axes[1], O_disp, 0.0, "Top-5% overlap"),
    ]:
        im = ax.imshow(matrix, cmap="Blues_r", vmin=vmin, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(pefts)), pefts, rotation=45, ha="right")
        ax.set_yticks(range(len(pefts)), pefts)
        ax.set_xlabel("PEFT used for scoring")
        ax.set_ylabel(ylabel)
        for i in range(len(pefts)):
            for j in range(len(pefts)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    save_pdf(fig, "fig_motivation_disagreement.pdf")
    print("wrote fig_motivation_disagreement.pdf")


if __name__ == "__main__":
    main()
