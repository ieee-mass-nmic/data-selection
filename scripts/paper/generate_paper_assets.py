"""Generate paper-ready PCU-Select figures and tables from result exports.

The script reads only the stable result bundle under result/ and writes
manuscript assets under paper/Figures and paper/tables. It intentionally
avoids titles inside plots; LaTeX captions carry the explanation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
DATA = RESULT / "data"
TABLES_IN = RESULT / "tables"
PAPER = ROOT / "paper"
FIG_DIR = PAPER / "Figures"
TAB_DIR = PAPER / "tables"

PEFT_ORDER = [
    "L-r8-qv",
    "L-r8-mlp",
    "IA3-attnmlp",
    "AD-b64",
    "L-r4-qv",
    "L-r32-qkvo",
    "L-r8-lowlayers",
    "L-r8-highlayers",
]

MAIN_PEFTS = ["AD-b64", "IA3-attnmlp", "L-r16-qkvo", "L-r8-mlp", "L-r8-qv"]
METHOD_LABEL = {
    "random": "Random",
    "rds_plus": "RDS+",
    "grad_sim": "Influence",
    "less": "LESS",
    "pcu": "PCU-Select",
}


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line])


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
        }
    )


def save_pdf(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def tex_escape(s: object) -> str:
    text = str(s)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def topk(ids: list[str], values: np.ndarray, k: int) -> list[str]:
    order = np.argsort(-values)[:k]
    return [ids[i] for i in order]


def fig_motivation_disagreement() -> None:
    df = pd.read_parquet(DATA / "motivation" / "values.parquet")
    df = df[(df["signal"] == "u_hi") & (df["peft_name"] != "<agnostic>")]
    pefts = [p for p in PEFT_ORDER if p in set(df["peft_name"])]
    pidx = {p: i for i, p in enumerate(pefts)}
    S_acc = np.zeros((len(pefts), len(pefts)))
    O_acc = np.zeros_like(S_acc)
    n_tasks = 0
    intra_rho, intra_ov = [], []

    for task, dft in df.groupby("task_id"):
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
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    intra_rho.append(spearman(reps[i], reps[j]))
                    intra_ov.append(jaccard(topk(common, reps[i], k), topk(common, reps[j], k)))

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

    S_mean, O_mean = S_acc / n_tasks, O_acc / n_tasks
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), constrained_layout=True)
    for ax, matrix, vmin, ylabel in [
        (axes[0], S_mean, -1.0, "Spearman agreement"),
        (axes[1], O_mean, 0.0, "Top-5% overlap"),
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


def fig_motivation_transfer() -> None:
    df = read_jsonl(DATA / "MOT_F2.jsonl")
    df["src"] = df["extra"].map(lambda e: e.get("src_peft"))
    df["tgt"] = df["extra"].map(lambda e: e.get("tgt_peft"))
    tr = df[df["method"] == "transfer"]
    pefts = [p for p in PEFT_ORDER if p in set(tr["src"]) | set(tr["tgt"])]
    perf = tr.groupby(["tgt", "src"])["metric"].mean().unstack().reindex(index=pefts, columns=pefts)
    rnd = df[df["method"] == "transfer_random"].groupby("tgt")["metric"].mean().reindex(pefts)
    agn = df[df["method"] == "transfer_agnostic"].groupby("tgt")["metric"].mean().reindex(pefts)
    norm = perf.copy()
    norm_agn = perf.copy()
    for tgt in pefts:
        denom = perf.loc[tgt, tgt] - rnd.loc[tgt]
        norm.loc[tgt] = (perf.loc[tgt] - rnd.loc[tgt]) / denom
        norm_agn.loc[tgt] = (agn.loc[tgt] - rnd.loc[tgt]) / denom

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), constrained_layout=True)
    for ax, matrix, label in [
        (axes[0], norm, "PEFT-specific source"),
        (axes[1], norm_agn, "PEFT-agnostic source"),
    ]:
        arr = matrix.to_numpy(float)
        im = ax.imshow(arr, cmap="RdYlGn", vmin=min(-2.5, np.nanmin(arr)), vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(pefts)), pefts, rotation=45, ha="right")
        ax.set_yticks(range(len(pefts)), pefts)
        ax.set_xlabel("source PEFT for selection")
        ax.set_ylabel("target PEFT for training")
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontsize=8)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=5.7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    save_pdf(fig, "fig_motivation_transfer.pdf")


def fig_cost_break_even() -> None:
    df = read_jsonl(DATA / "E2.jsonl")
    cm = json.loads((DATA / "E2_cost_model.json").read_text())
    df["apply_h"] = df["select_gpu_h"] + df["target_train_gpu_h"]
    per = df.groupby("method").agg(apply_h=("apply_h", "mean"), perf=("metric", "mean"))
    T = np.array(cm["T_values"])
    offline = cm["offline_gpu_h"]
    recompute = cm["per_peft_recompute_gpu_h"]
    influence = set(cm["influence_methods"])

    def total(method: str) -> np.ndarray:
        apply_h = per.loc[method, "apply_h"]
        if method == "pcu":
            return offline + T * apply_h
        if method in influence:
            return T * (recompute + apply_h)
        return T * apply_h

    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    for method in ["grad_sim", "less", "pcu", "rds_plus", "random"]:
        y = total(method)
        lw = 2.1 if method == "pcu" else 1.1
        ax.plot(T, y, marker="o", linewidth=lw, label=METHOD_LABEL.get(method, method))
    pcu_apply = per.loc["pcu", "apply_h"]
    for method in ["less"]:
        denom = per.loc[method, "apply_h"] + recompute - pcu_apply
        t_star = offline / denom
        ax.axvline(t_star, color="0.4", linestyle=":", linewidth=0.8)
        ax.text(t_star + 0.1, ax.get_ylim()[1] * 0.72, f"T*={t_star:.1f}", fontsize=7)
    ax.set_xlabel("target PEFT configurations served")
    ax.set_ylabel("total GPU-hours")
    ax.legend(frameon=False, loc="upper left")
    save_pdf(fig, "fig_cost_break_even.pdf")


def _config_distance(a: str, b: str) -> float:
    import re

    def feats(name: str) -> tuple[int, str, str]:
        m = re.search(r"r(\d+)", name)
        rank = int(m.group(1)) if m else 0
        place = "low" if "low" in name else "high" if "high" in name else "all"
        mods = "all" if "all" in name else "qkvo" if "qkvo" in name else "qv"
        return rank, place, mods

    fa, fb = feats(a), feats(b)
    d = abs(math.log2(max(fa[0], 1)) - math.log2(max(fb[0], 1)))
    return d + float(fa[1] != fb[1]) + float(fa[2] != fb[2])


def fig_config_sensitivity() -> None:
    e4 = read_jsonl(DATA / "E4.jsonl")
    e4["src"] = e4["extra"].map(lambda e: e.get("src_peft"))
    e4["tgt"] = e4["extra"].map(lambda e: e.get("tgt_peft"))
    mm = e4[e4["method"] == "pcu_mismatch"]
    configs = sorted(set(mm["src"]) | set(mm["tgt"]))
    M = mm.groupby(["src", "tgt"])["metric"].mean().unstack().reindex(index=configs, columns=configs)
    Mn = M.copy()
    for c in configs:
        Mn[c] = M[c] / M.loc[c, c]
    overlap = json.loads((DATA / "E4_overlap.json").read_text())["overlap"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
    arr = Mn.to_numpy(float)
    im = axes[0].imshow(arr, cmap="RdYlGn", vmin=0.84, vmax=1.0, aspect="auto")
    axes[0].set_xticks(range(len(configs)), configs, rotation=45, ha="right")
    axes[0].set_yticks(range(len(configs)), configs)
    axes[0].set_xlabel("target config")
    axes[0].set_ylabel("source config")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            axes[0].text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=5.1)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    for method, mat in overlap.items():
        keys = list(mat.keys())
        xs, ys = [], []
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                xs.append(_config_distance(a, b))
                ys.append(1 - mat[a][b])
        label = "PCU-Select" if method == "pcu" else "RDS+"
        axes[1].scatter(xs, ys, s=24 if method == "pcu" else 18, alpha=0.78, label=label)
    axes[1].set_xlabel("configuration distance")
    axes[1].set_ylabel("selection difference (1 - Jaccard)")
    axes[1].legend(frameon=False, loc="upper left")
    save_pdf(fig, "fig_config_sensitivity.pdf")


def fig_ood_calibration() -> None:
    e5 = read_jsonl(DATA / "E5.jsonl")
    e5["level"] = e5["extra"].map(lambda e: e.get("level"))
    e5["mode"] = e5["method"].str.replace("pcu_", "", regex=False).where(
        e5["method"].str.startswith("pcu_"), e5["method"]
    )
    levels = ["L0", "L1", "L2"]
    modes = ["zeroshot", "cal200", "cal500"]
    means = e5.groupby(["level", "mode"])["metric"].mean().unstack()
    x = np.arange(len(levels))
    width = 0.23
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for i, mode in enumerate(modes):
        ax.bar(x + (i - 1) * width, means.loc[levels, mode], width, label=mode, color=colors[i])
    for method, color in [("less", "black"), ("rds_plus", "0.45")]:
        ax.plot(x, means.loc[levels, method], marker="D", linestyle="--", color=color, label=METHOD_LABEL[method])
    ax.set_xticks(x, levels)
    ax.set_xlabel("OOD level")
    ax.set_ylabel("downstream metric")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    save_pdf(fig, "fig_ood_calibration.pdf")


def table_main_results() -> None:
    e1 = read_jsonl(DATA / "E1.jsonl")
    main = e1[e1["budget"].eq(0.10)]
    rows = ["random", "rds_plus", "grad_sim", "less", "pcu"]
    pivot = main.groupby(["method", "peft"])["metric"].mean().unstack().reindex(index=rows, columns=MAIN_PEFTS)
    pivot["Avg."] = main.groupby("method")["metric"].mean().reindex(rows)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Main downstream performance at a 10\\% selection budget. Values average over four tasks and three target-training seeds; task-native metrics are scaled as percentages, so larger is better.}",
        "\\label{tab:main-results}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & AD-b64 & IA3 & L-r16-qkvo & L-r8-mlp & L-r8-qv & Avg. \\\\",
        "\\midrule",
    ]
    for method in rows:
        vals = [f"{pivot.loc[method, c]:.2f}" for c in MAIN_PEFTS + ["Avg."]]
        name = METHOD_LABEL[method]
        if method == "pcu":
            name = "\\textbf{" + name + "}"
            vals = ["\\textbf{" + v + "}" for v in vals]
        lines.append(f"{name} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TAB_DIR / "table_main_results.tex", "\n".join(lines))


def table_ablation() -> None:
    e3 = read_jsonl(DATA / "E3.jsonl")
    ab = e3.groupby("method").agg(metric=("metric", "mean"), ndcg=("ndcg_at_k", "mean")).sort_values("metric")
    full = float(ab.loc["pcu_strat=adaptive", "metric"])
    rows = [
        ("Full PCU-Select", "pcu_strat=adaptive"),
        ("No PEFT code", "pcu_no_zp"),
        ("Family one-hot only", "pcu_family_onehot"),
        ("No task sketch", "pcu_no_zt"),
        ("No activation signature", "pcu_no_act"),
        ("Low-fidelity only", "pcu_lo_only"),
        ("High-fidelity only", "pcu_hi_only"),
        ("No uncertainty penalty", "pcu_lambda=0.0"),
        ("Global top-$k$", "pcu_strat=global_topk"),
        ("Uniform clusters", "pcu_strat=uniform_cluster"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Ablations on GSM8K and HumanEval with two representative PEFTs at a 10\\% budget. Drop is relative to the full adaptive selector.}",
        "\\label{tab:ablations}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Variant & Metric & Drop & NDCG \\\\",
        "\\midrule",
    ]
    for label, method in rows:
        metric = float(ab.loc[method, "metric"])
        ndcg = float(ab.loc[method, "ndcg"])
        lines.append(f"{label} & {metric:.2f} & {full - metric:.2f} & {ndcg:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TAB_DIR / "table_ablations.tex", "\n".join(lines))


def table_peft_configs() -> None:
    cfg = pd.read_csv(TABLES_IN / "table1.csv")
    rows = ["L-r8-qv", "L-r8-mlp", "IA3-attnmlp", "AD-b64", "L-r4-qv", "L-r32-qkvo"]
    cfg = cfg[cfg["peft"].isin(rows)].set_index("peft").loc[rows].reset_index()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Representative PEFT configurations. Parameter counts come from the registry counter for the Llama2-7B backbone.}",
        "\\label{tab:peft-configs}",
        "\\begin{tabular}{llrrl}",
        "\\toprule",
        "Config & Family & Params & \\% bb & Sites \\\\",
        "\\midrule",
    ]
    for _, r in cfg.iterrows():
        lines.append(
            f"{tex_escape(r.peft)} & {tex_escape(r.family)} & {int(r.n_trainable)/1e6:.2f}M & "
            f"{float(r.pct_backbone):.3f} & {tex_escape(r.touched_sites)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TAB_DIR / "table_peft_configs.tex", "\n".join(lines))


def main() -> None:
    setup_style()
    fig_motivation_disagreement()
    fig_motivation_transfer()
    fig_cost_break_even()
    fig_config_sensitivity()
    fig_ood_calibration()
    table_main_results()
    table_ablation()
    table_peft_configs()
    print(f"Wrote figures to {FIG_DIR.relative_to(ROOT)}")
    print(f"Wrote tables to {TAB_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
