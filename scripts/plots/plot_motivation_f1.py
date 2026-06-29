"""Figure 1 — data-value ranking disagreement (motivation §3).

Left panel : Spearman correlation heatmap between per-PEFT value rankings.
Right panel: Top-k (default 5%) overlap heatmap between per-PEFT selections.

Both are PEFT×PEFT matrices; low off-diagonal = strong disagreement. The result
is meaningful ONLY against the **noise floor** (§3.3): two independent estimates
(anchor×seed replicates) of the SAME PEFT. We annotate the mean intra-PEFT
self-agreement; off-diagonal cells below it are genuine disagreement, not noise.

A structural summary (§3.5) groups off-diagonal pairs by structural distance
(same-family-capacity / same-family-placement / cross-family) and reports mean
agreement per bucket — disagreement should grow with structural distance.

    python scripts/plots/plot_motivation_f1.py \
        --values runs/exp1/motivation/values.parquet --signal u_hi --out-dir figs

Reads the parquet from build_motivation_values.py; matplotlib is optional
(pip install -e ".[viz]").
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcu_select.eval.metrics import jaccard, spearman
from pcu_select.experiments import PEFT_REGISTRY

AGNOSTIC = "<agnostic>"


def _structural_bucket(a: str, b: str) -> str | None:
    if a not in PEFT_REGISTRY or b not in PEFT_REGISTRY:
        return None
    sa, sb = PEFT_REGISTRY[a], PEFT_REGISTRY[b]
    if sa.family != sb.family:
        return "cross-family"
    if sa.modules_key == sb.modules_key and sa.layer_range == sb.layer_range:
        return "same-fam-capacity"
    return "same-fam-placement"


def _replicate_vectors(sub: pd.DataFrame, common: list[str]) -> list[np.ndarray]:
    """One value vector per (anchor_id, seed) replicate, aligned to `common` ids."""
    vecs = []
    for _, g in sub.groupby(["anchor_id", "seed"]):
        s = g.groupby("sample_id")["value"].mean()
        if set(common).issubset(s.index):
            vecs.append(s.loc[common].to_numpy(dtype=np.float64))
    return vecs


def _topk(ids: list[str], values: np.ndarray, k: int) -> list[str]:
    order = np.argsort(-values)[:k]
    return [ids[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", type=Path, required=True)
    ap.add_argument("--signal", type=str, default="u_hi", choices=["u_hi", "u_grad"])
    ap.add_argument("--top-frac", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = ap.parse_args()

    df = pd.read_parquet(args.values)
    df = df[(df["signal"] == args.signal) & (df["peft_name"] != AGNOSTIC)]
    if df.empty:
        raise SystemExit(f"no '{args.signal}' rows in {args.values}")
    pefts = [n for n in PEFT_REGISTRY if n in set(df["peft_name"])]
    tasks = sorted(df["task_id"].unique())
    P = len(pefts)

    # Accumulate matrices across tasks, then average.
    S_acc, O_acc, n_acc = np.zeros((P, P)), np.zeros((P, P)), 0
    intra_rho, intra_ov = [], []  # per (task, peft) self-agreement → noise floor
    struct: dict[str, list[float]] = {"same-fam-capacity": [], "same-fam-placement": [],
                                       "cross-family": []}
    struct_ov: dict[str, list[float]] = {k: [] for k in struct}

    for tid in tasks:
        dft = df[df["task_id"] == tid]
        # common sample set across all pefts in this task
        id_sets = [set(dft[dft["peft_name"] == p]["sample_id"]) for p in pefts]
        common = sorted(set.intersection(*id_sets)) if id_sets else []
        if len(common) < 2:
            continue
        k = max(1, int(round(args.top_frac * len(common))))
        mean_vec, top_ids = {}, {}
        for p in pefts:
            sub = dft[dft["peft_name"] == p]
            reps = _replicate_vectors(sub, common)
            if not reps:
                continue
            mv = np.mean(reps, axis=0)
            mean_vec[p], top_ids[p] = mv, _topk(common, mv, k)
            # noise floor: pairwise agreement among this PEFT's replicates
            for r1, r2 in itertools.combinations(reps, 2):
                intra_rho.append(spearman(r1, r2))
                intra_ov.append(jaccard(_topk(common, r1, k), _topk(common, r2, k)))

        S = np.full((P, P), np.nan)
        O = np.full((P, P), np.nan)
        for i, pi in enumerate(pefts):
            for j, pj in enumerate(pefts):
                if pi not in mean_vec or pj not in mean_vec:
                    continue
                S[i, j] = spearman(mean_vec[pi], mean_vec[pj])
                O[i, j] = jaccard(top_ids[pi], top_ids[pj])
                if i < j:
                    bucket = _structural_bucket(pi, pj)
                    if bucket:
                        struct[bucket].append(S[i, j])
                        struct_ov[bucket].append(O[i, j])
        S_acc = np.nansum([S_acc, np.nan_to_num(S)], axis=0)
        O_acc = np.nansum([O_acc, np.nan_to_num(O)], axis=0)
        n_acc += 1

    if n_acc == 0:
        raise SystemExit("no task had ≥2 common samples across PEFTs.")
    S_mean, O_mean = S_acc / n_acc, O_acc / n_acc
    rho_floor = float(np.nanmean(intra_rho)) if intra_rho else float("nan")
    ov_floor = float(np.nanmean(intra_ov)) if intra_ov else float("nan")

    # ---- figure ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(2 + 1.0 * P, 0.9 * P + 1.5))
    for ax, M, title, vlo in (
        (axes[0], S_mean, "Spearman ρ (rank agreement)", -1.0),
        (axes[1], O_mean, f"Top-{int(args.top_frac*100)}% overlap (Jaccard)", 0.0),
    ):
        im = ax.imshow(M, cmap="Blues_r", vmin=vlo, vmax=1.0, aspect="auto")
        ax.set_xticks(range(P), pefts, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(P), pefts, fontsize=7)
        for i in range(P):
            for j in range(P):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
    floor_txt = f"noise floor: intra-PEFT ρ={rho_floor:.2f}, top-k overlap={ov_floor:.2f}"
    fig.suptitle(f"F1: data-value ranking disagreement ({args.signal}, {n_acc} task(s))\n"
                 f"{floor_txt} — off-diagonal below floor ⇒ real disagreement", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = args.out_dir / f"F1_disagreement_{args.signal}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")

    # ---- structural summary CSV (§3.5) ----
    csv_path = args.out_dir / f"F1_structural_{args.signal}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "n_pairs", "mean_spearman", "mean_topk_overlap"])
        for b in ("same-fam-capacity", "same-fam-placement", "cross-family"):
            vals, ovs = struct[b], struct_ov[b]
            if vals:
                w.writerow([b, len(vals), f"{np.nanmean(vals):.4f}", f"{np.nanmean(ovs):.4f}"])
        w.writerow(["intra-PEFT (noise floor)", len(intra_rho),
                    f"{rho_floor:.4f}", f"{ov_floor:.4f}"])
    print(f"wrote {csv_path}")
    print(f"noise floor: ρ_intra={rho_floor:.3f}  top-{int(args.top_frac*100)}%-overlap={ov_floor:.3f}")
    for b in ("same-fam-capacity", "same-fam-placement", "cross-family"):
        if struct[b]:
            print(f"  {b:<22} mean ρ={np.nanmean(struct[b]):.3f}  "
                  f"overlap={np.nanmean(struct_ov[b]):.3f}  (n={len(struct[b])})")


if __name__ == "__main__":
    main()
