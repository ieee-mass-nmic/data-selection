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
    "lo_proxy_quota": "Low-fid proxy",
    "pcu": "PCU-Select",
}

# Method rows shown in the main / per-task tables, in display order. The
# low-fidelity-proxy control (reviewer 3.3) is inserted just before PCU-Select
# only when its rows are present in the export, so the manuscript compiles before
# that grid is run and the row appears automatically once it is.
MAIN_TABLE_METHODS = ["random", "rds_plus", "grad_sim", "less", "pcu"]


def _table_methods(present: set[str]) -> list[str]:
    rows = list(MAIN_TABLE_METHODS)
    if "lo_proxy_quota" in present:
        rows.insert(rows.index("pcu"), "lo_proxy_quota")
    return rows


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
    axes[1].set_ylabel(r"1 $-$ Jaccard overlap")
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
    mode_label = {"zeroshot": "zero-shot", "cal200": "cal200", "cal500": "cal500"}
    means = e5.groupby(["level", "mode"])["metric"].mean().unstack()
    x = np.arange(len(levels))
    width = 0.23
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for i, mode in enumerate(modes):
        ax.bar(x + (i - 1) * width, means.loc[levels, mode], width, label=mode_label[mode], color=colors[i])
    for method, color in [("less", "black"), ("rds_plus", "0.45")]:
        ax.plot(x, means.loc[levels, method], marker="D", linestyle="--", color=color, label=METHOD_LABEL[method])
    ax.set_xticks(x, levels)
    ax.set_xlabel("OOD level")
    ax.set_ylabel("downstream metric")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    save_pdf(fig, "fig_ood_calibration.pdf")


def _wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value.

    Uses scipy when available for the exact test; otherwise falls back to a
    normal approximation with continuity and tie corrections.
    """
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(diffs).pvalue)
    except Exception:
        d = diffs[diffs != 0]
        n = len(d)
        if n == 0:
            return float("nan")
        ranks = rankdata(np.abs(d)) + 1.0  # 1-based ranks
        w_plus = ranks[d > 0].sum()
        mean_w = n * (n + 1) / 4.0
        # tie correction on the shared magnitude ranks
        _, counts = np.unique(np.abs(d), return_counts=True)
        tie = (counts ** 3 - counts).sum()
        var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
        z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
        # two-sided normal tail
        return float(math.erfc(z / math.sqrt(2.0)))


def paired_pcu_vs_less(main: pd.DataFrame) -> dict:
    """Paired PCU-vs-LESS comparison across the 20 PEFT x task cells.

    Each cell is the seed-averaged metric. Returns the mean paired difference,
    a 95% paired-bootstrap CI on that mean, the Wilcoxon signed-rank p-value,
    and the win count, so the reframed 'comparable' claim is reproducible.
    """
    cell = main.groupby(["method", "peft", "task"])["metric"].mean()
    pcu = cell.loc["pcu"].reindex(MAIN_PEFTS, level="peft")
    less = cell.loc["less"].reindex(MAIN_PEFTS, level="peft")
    diffs = (pcu - less).to_numpy(float)
    rng = np.random.default_rng(20260706)
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p": _wilcoxon_p(diffs),
        "wins": int((diffs > 0).sum()),
        "n": int(len(diffs)),
    }


def _cell_dispersion(main: pd.DataFrame, rows: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (method, peft) seed-level mean and std.

    Metrics are first averaged over the four tasks within each seed, then the
    mean and sample std are taken over the three target-training seeds, so the
    reported dispersion reflects seed noise at the PEFT-column level.
    """
    seed_col = main.groupby(["method", "peft", "seed"])["metric"].mean()
    mean = seed_col.groupby(["method", "peft"]).mean().unstack().reindex(index=rows, columns=MAIN_PEFTS)
    std = seed_col.groupby(["method", "peft"]).std(ddof=1).unstack().reindex(index=rows, columns=MAIN_PEFTS)
    seed_all = main.groupby(["method", "seed"])["metric"].mean()
    mean["Avg."] = seed_all.groupby("method").mean().reindex(rows)
    std["Avg."] = seed_all.groupby("method").std(ddof=1).reindex(rows)
    return mean, std


def table_main_results() -> None:
    e1 = read_jsonl(DATA / "E1.jsonl")
    main = e1[e1["budget"].eq(0.10)]
    rows = _table_methods(set(main["method"]))
    cols = MAIN_PEFTS + ["Avg."]
    mean, std = _cell_dispersion(main, rows)
    test = paired_pcu_vs_less(main)
    caption = (
        "Main downstream performance at a 10\\% selection budget, reported as "
        "mean$\\pm$std over three target-training seeds (task-native metrics are "
        "scaled as percentages, so larger is better). PCU-Select and per-PEFT "
        "LESS are statistically indistinguishable: the paired difference over the "
        f"{test['n']} PEFT$\\times$task cells is ${test['mean']:.2f}$ points "
        f"(95\\% bootstrap CI $[{test['ci_lo']:.2f}, {test['ci_hi']:.2f}]$; "
        f"Wilcoxon signed-rank $p={test['p']:.2f}$). Boldface marks the best mean "
        "per column."
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{" + caption + "}",
        "\\label{tab:main-results}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & AD-b64 & IA3-attnmlp & L-r16-qkvo & L-r8-mlp & L-r8-qv & Avg. \\\\",
        "\\midrule",
    ]
    best = {c: mean[c].max() for c in cols}
    for method in rows:
        cells = []
        for c in cols:
            body = f"{mean.loc[method, c]:.2f}{{\\scriptsize$\\pm${std.loc[method, c]:.2f}}}"
            if abs(mean.loc[method, c] - best[c]) < 1e-9:
                body = "\\textbf{" + body + "}"
            cells.append(body)
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TAB_DIR / "table_main_results.tex", "\n".join(lines))


TASK_ORDER = ["gsm8k", "humaneval", "mmlu", "tydiqa"]
TASK_LABEL = {
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "mmlu": "MMLU",
    "tydiqa": "TyDiQA",
}
METRIC_LABEL = {
    "exact_match": "EM",
    "pass@1": "Pass@1",
    "accuracy": "Acc",
    "f1": "F1",
}


def _per_task_dispersion(sub: pd.DataFrame, rows: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For a single task: per (method, peft) seed mean and std, plus a PEFT-averaged
    ``Avg.`` column whose dispersion is the seed std of the PEFT-averaged score.

    Kept separate from ``_cell_dispersion`` (which averages over tasks) so the
    appendix surfaces every task's raw per-PEFT numbers rather than the collapsed
    average the reviewer objected to."""
    cell = sub.groupby(["method", "peft", "seed"])["metric"].mean()
    mean = cell.groupby(["method", "peft"]).mean().unstack().reindex(index=rows, columns=MAIN_PEFTS)
    std = cell.groupby(["method", "peft"]).std(ddof=1).unstack().reindex(index=rows, columns=MAIN_PEFTS)
    seed_avg = sub.groupby(["method", "seed"])["metric"].mean()
    mean["Avg."] = seed_avg.groupby("method").mean().reindex(rows)
    std["Avg."] = seed_avg.groupby("method").std(ddof=1).reindex(rows)
    return mean, std


def table_per_task_peft() -> None:
    """Full task x PEFT x method grid at the 10% budget (appendix), so the main
    table's PEFT-and-task average is shown alongside its constituents, not instead
    of them (reviewer 3.2)."""
    e1 = read_jsonl(DATA / "E1.jsonl")
    main = e1[e1["budget"].eq(0.10)]
    rows = _table_methods(set(main["method"]))
    cols = MAIN_PEFTS + ["Avg."]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Full per-task $\\times$ PEFT $\\times$ method downstream results at a "
        "10\\% selection budget (mean$\\pm$std over three target-training seeds, on each "
        "task's native metric). This is the unaveraged source of Table~\\ref{tab:main-results}: "
        "because the four metrics have different scales, ceilings, and seed variance, a point "
        "on one task is not equivalent to a point on another, so we report every cell. "
        "Boldface marks the best mean per column within each task. PCU-Select leads on GSM8K "
        "and MMLU, is within seed noise of LESS on HumanEval, and trails LESS on TyDiQA; the "
        "PEFT-and-task average in Table~\\ref{tab:main-results} smooths over this heterogeneity.}",
        "\\label{tab:per-task-peft}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & AD-b64 & IA3-attnmlp & L-r16-qkvo & L-r8-mlp & L-r8-qv & Avg. \\\\",
    ]
    for task in TASK_ORDER:
        sub = main[main["task"].eq(task)]
        metric_name = str(sub["metric_name"].iloc[0])
        mean, std = _per_task_dispersion(sub, rows)
        best = {c: mean[c].max() for c in cols}
        header = f"{TASK_LABEL[task]} ({METRIC_LABEL.get(metric_name, metric_name)})"
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{header}}}}} \\\\")
        for method in rows:
            cells = []
            for c in cols:
                body = f"{mean.loc[method, c]:.2f}{{\\scriptsize$\\pm${std.loc[method, c]:.2f}}}"
                if abs(mean.loc[method, c] - best[c]) < 1e-9:
                    body = "\\textbf{" + body + "}"
                cells.append(body)
            lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TAB_DIR / "table_per_task_peft.tex", "\n".join(lines))


def _paired_ci(diffs: np.ndarray, seed: int = 20260706) -> tuple[float, float, float]:
    """Mean paired difference and 95% paired-bootstrap CI over the given cells."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def table_per_task_normalized() -> None:
    """Per-task normalized improvement of PCU-Select over Random / RDS+ / LESS
    (reviewer 3.2). Improvements over Random and RDS+ are given as scale-free
    relative gains so cross-task point-differences are not treated as equivalent;
    the head-to-head with LESS is shown as an absolute paired difference with a
    95% bootstrap CI over the five PEFT cells (seed-averaged)."""
    e1 = read_jsonl(DATA / "E1.jsonl")
    main = e1[e1["budget"].eq(0.10)]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\caption{Per-task normalized improvement of PCU-Select at the 10\\% budget. "
        "Absolute scores (Random, RDS+, LESS, PCU) are seed- and PEFT-averaged on each "
        "task's native metric. $\\Delta$Rand and $\\Delta$RDS+ are \\emph{relative} gains "
        "($100\\cdot(\\mathrm{PCU}-b)/b$), which place the four heterogeneous metrics on a "
        "common scale so that cross-task point-differences are not treated as equivalent. "
        "$\\Delta$LESS is the absolute paired difference in native points against the "
        "strongest baseline, with a 95\\% bootstrap CI over the five PEFT cells: a CI that "
        "excludes zero marks a per-task win (GSM8K, MMLU) or loss (TyDiQA), and the sign "
        "flips across tasks, which the averaged Table~\\ref{tab:main-results} conceals.}",
        "\\label{tab:per-task-normalized}",
        "\\begin{tabular}{llrrrrrrl}",
        "\\toprule",
        "Task & Metric & Random & RDS+ & LESS & PCU & $\\Delta$Rand & $\\Delta$RDS+ & "
        "$\\Delta$LESS (95\\% CI) \\\\",
        "\\midrule",
    ]
    for task in TASK_ORDER:
        sub = main[main["task"].eq(task)]
        metric_name = str(sub["metric_name"].iloc[0])
        cell = sub.groupby(["method", "peft"])["metric"].mean()
        pcu = cell.loc["pcu"].reindex(MAIN_PEFTS).to_numpy(float)
        rand = cell.loc["random"].reindex(MAIN_PEFTS).to_numpy(float)
        rds = cell.loc["rds_plus"].reindex(MAIN_PEFTS).to_numpy(float)
        less = cell.loc["less"].reindex(MAIN_PEFTS).to_numpy(float)
        rel_rand = 100.0 * (pcu.mean() - rand.mean()) / rand.mean()
        rel_rds = 100.0 * (pcu.mean() - rds.mean()) / rds.mean()
        d_less, lo, hi = _paired_ci(pcu - less)
        ci = f"${d_less:+.2f}$ $[{lo:+.2f}, {hi:+.2f}]$"
        lines.append(
            f"{TASK_LABEL[task]} & {METRIC_LABEL.get(metric_name, metric_name)} & "
            f"{rand.mean():.2f} & {rds.mean():.2f} & {less.mean():.2f} & {pcu.mean():.2f} & "
            f"{rel_rand:+.1f}\\% & {rel_rds:+.1f}\\% & {ci} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TAB_DIR / "table_per_task_normalized.tex", "\n".join(lines))


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
    # Read the augmented config table (full seen set + the unseen configs used
    # in E4/E5) generated by build_table1.py into paper/data/. The seen rows
    # match the five columns of Table~\ref{tab:main-results}; the unseen rows
    # match the E5 out-of-distribution targets.
    cfg = pd.read_csv(PAPER / "data" / "table1.csv")
    seen = ["L-r8-qv", "L-r16-qkvo", "L-r8-mlp", "IA3-attnmlp", "AD-b64"]
    unseen = ["L-r4-qv", "L-r32-qkvo"]
    cfg = cfg.set_index("peft")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{PEFT configurations used in the experiments. The five "
        "\\emph{seen} configurations are the scorer's training support and the "
        "columns of Table~\\ref{tab:main-results}; the \\emph{unseen} "
        "configurations are held-out targets evaluated in the OOD study "
        "(Sec.~\\ref{sec:analysis}). Parameter counts come from the registry "
        "counter for the Llama2-7B backbone.}",
        "\\label{tab:peft-configs}",
        "\\begin{tabular}{lllrrl}",
        "\\toprule",
        "Config & Group & Family & Params & \\% bb & Sites \\\\",
        "\\midrule",
    ]

    def _row(name: str, group: str) -> str:
        r = cfg.loc[name]
        return (
            f"{tex_escape(name)} & {group} & {tex_escape(r.family)} & "
            f"{int(r.n_trainable)/1e6:.2f}M & {float(r.pct_backbone):.3f} & "
            f"{tex_escape(r.touched_sites)} \\\\"
        )

    for name in seen:
        lines.append(_row(name, "Seen"))
    lines.append("\\midrule")
    for name in unseen:
        lines.append(_row(name, "Unseen"))
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
    table_per_task_peft()
    table_per_task_normalized()
    table_ablation()
    table_peft_configs()
    print(f"Wrote figures to {FIG_DIR.relative_to(ROOT)}")
    print(f"Wrote tables to {TAB_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
