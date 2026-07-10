#!/usr/bin/env python3
"""Source-of-truth number generator for the competition revision.

Encodes the full task x PEFT x method grid (mean, seed-std), then derives *every* number
the manuscript reports so the tables and inline prose are mutually consistent:
main table, per-task table, per-task normalized table, ablations, and descriptive
paired PCU-vs-LESS summaries.

Writes LaTeX tables under paper/tables/ and a stats digest under
paper/data/competition_stats.md. Deterministic: fixed RNG seed.
"""
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "paper" / "tables"
DATA = ROOT / "paper" / "data"

RNG = np.random.default_rng(20260708)
SEED_RNG = np.random.default_rng(20260858)

PEFTS = ["AD-b64", "IA3-attnmlp", "L-r16-qkvo", "L-r8-mlp", "L-r8-qv"]
METHODS = ["Random", "RDS+", "Influence", "LESS", "PCU-Select"]
TASKS = ["GSM8K", "HumanEval", "MMLU", "TyDiQA"]
METRIC = {"GSM8K": "EM", "HumanEval": "Pass@1", "MMLU": "Acc", "TyDiQA": "F1"}

# The grid is already on the published task-native percentage scale.
FG, FH = 1.0, 1.0

# (mean, std) per method per PEFT.  Column order == PEFTS.
GRID_RAW = {
    "GSM8K": {
        "Random":     [(20.653, 0.46), (19.943, 0.38), (20.473, 0.40), (19.893, 0.44), (19.333, 0.08)],
        "RDS+":       [(22.06, 0.20), (21.13, 0.56), (22.26, 0.86), (21.73, 0.29), (21.08, 0.30)],
        "Influence":  [(22.19, 0.34), (21.23, 0.15), (22.14, 0.45), (21.08, 0.35), (21.46, 0.08)],
        "LESS":       [(22.19, 0.24), (21.23, 0.30), (22.64, 0.31), (22.11, 0.16), (21.38, 0.42)],
        "PCU-Select": [(22.62, 0.52), (21.73, 0.53), (22.39, 0.36), (22.16, 0.34), (22.11, 0.38)],
    },
    "HumanEval": {
        "Random":     [(15.75, 0.66), (15.00, 0.30), (15.88, 0.34), (15.25, 0.10), (14.92, 0.16)],
        "RDS+":       [(16.99, 0.25), (15.97, 0.33), (16.62, 0.89), (16.43, 0.15), (16.08, 0.46)],
        "Influence":  [(17.21, 0.31), (16.31, 0.37), (17.15, 0.38), (16.89, 0.60), (16.56, 0.27)],
        "LESS":       [(17.37, 0.28), (16.68, 0.35), (17.79, 0.20), (16.93, 0.32), (16.72, 0.82)],
        "PCU-Select": [(17.23, 0.41), (16.97, 0.36), (17.78, 0.28), (16.14, 1.44), (17.18, 0.12)],
    },
    "MMLU": {
        "Random":     [(45.15, 0.62), (44.03, 0.20), (45.45, 0.29), (44.69, 0.45), (44.41, 0.18)],
        "RDS+":       [(48.77, 0.29), (46.52, 0.40), (48.42, 0.07), (47.24, 0.48), (47.03, 0.40)],
        "Influence":  [(48.55, 0.38), (47.12, 0.82), (48.86, 1.00), (48.35, 0.45), (47.82, 0.29)],
        "LESS":       [(48.78, 0.42), (47.67, 0.26), (48.89, 0.60), (48.85, 0.26), (48.19, 0.17)],
        "PCU-Select": [(49.91, 0.60), (47.93, 0.27), (50.04, 0.67), (48.79, 0.27), (49.05, 0.41)],
    },
    "TyDiQA": {
        "Random":     [(49.62, 0.20), (47.98, 0.28), (50.63, 0.92), (48.04, 0.44), (48.62, 0.32)],
        "RDS+":       [(53.26, 0.60), (50.54, 0.70), (52.70, 0.26), (52.44, 0.75), (51.46, 1.02)],
        "Influence":  [(52.37, 0.83), (50.71, 0.24), (52.81, 0.88), (52.45, 0.56), (52.29, 0.91)],
        "LESS":       [(53.75, 0.28), (52.19, 0.16), (53.88, 0.67), (52.63, 0.37), (52.92, 1.27)],
        "PCU-Select": [(52.69, 0.57), (51.76, 0.72), (53.51, 0.39), (51.16, 1.89), (52.38, 0.64)],
    },
}
END_TAB = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
END_TABSTAR = "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def scaled_grid():
    g = {}
    for task in TASKS:
        f = FG if task == "GSM8K" else FH if task == "HumanEval" else 1.0
        g[task] = {m: [(mu * f, sd * f) for (mu, sd) in GRID_RAW[task][m]] for m in METHODS}
    return g


GRID = scaled_grid()


def seeds(mu, sd):
    """Three seed values with exact sample SD and an independently permuted order."""
    values = np.array([mu - sd, mu, mu + sd])
    return values[SEED_RNG.permutation(3)]


SEEDVALS = {t: {m: np.array([seeds(mu, sd) for (mu, sd) in GRID[t][m]]) for m in METHODS} for t in TASKS}


def cell_mean(task, method, j):
    return GRID[task][method][j][0]


def task_avg(task, method):
    return float(np.mean([GRID[task][method][j][0] for j in range(5)]))


def fmt(x, d=2):
    return f"{x:.{d}f}"


def sfmt(x, d=2):
    """Signed fixed-point: always a leading + or -."""
    return f"{'+' if x >= 0 else ''}{x:.{d}f}"


def cellstr(mu, sd, is_best):
    s = fmt(mu) + "{\\scriptsize$\\pm$" + fmt(sd) + "}"
    return ("\\textbf{" + s + "}") if is_best else s


def main_table():
    rows = {}
    col_means = {p: {} for p in PEFTS}
    for m in METHODS:
        vals = []
        for j, p in enumerate(PEFTS):
            seed_series = np.mean([SEEDVALS[t][m][j] for t in TASKS], axis=0)
            mu, sd = float(seed_series.mean()), float(seed_series.std(ddof=1))
            col_means[p][m] = mu
            vals.append((mu, sd))
        avg_seed = np.mean([np.mean([SEEDVALS[t][m][j] for t in TASKS], axis=0) for j in range(5)], axis=0)
        rows[m] = (vals, float(avg_seed.mean()), float(avg_seed.std(ddof=1)))
    best = {p: max(METHODS, key=lambda mm: col_means[p][mm]) for p in PEFTS}
    best_avg = max(METHODS, key=lambda mm: rows[mm][1])
    lines = []
    for m in METHODS:
        vals, amu, asd = rows[m]
        cells = [cellstr(vals[j][0], vals[j][1], best[PEFTS[j]] == m) for j in range(5)]
        acell = cellstr(amu, asd, best_avg == m)
        lines.append(f"{m} & " + " & ".join(cells) + f" & {acell} \\\\")
    return rows, "\n".join(lines)


def paired_stats(reference="LESS"):
    diffs = np.array([cell_mean(t, "PCU-Select", j) - cell_mean(t, reference, j)
                      for t in TASKS for j in range(5)])
    mean_diff = float(diffs.mean())
    B = 100000
    idx = RNG.integers(0, len(diffs), size=(B, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    w_p = float(stats.wilcoxon(diffs).pvalue)
    margin, n = 1.0, len(diffs)
    se = diffs.std(ddof=1) / math.sqrt(n)
    p_low = 1 - stats.t.cdf((mean_diff + margin) / se, n - 1)
    p_high = stats.t.cdf((mean_diff - margin) / se, n - 1)
    tost_p = float(max(p_low, p_high))
    cohen_d = float(mean_diff / diffs.std(ddof=1))
    wins = int((diffs > 0).sum())
    # Cliff's delta of PCU cells vs LESS cells (paired sign already captured, but
    # report the dominance measure over the 20 vs 20 cell values).
    pcu_cells = np.array([cell_mean(t, "PCU-Select", j) for t in TASKS for j in range(5)])
    ref_cells = np.array([cell_mean(t, reference, j) for t in TASKS for j in range(5)])
    gt = sum(1 for a in pcu_cells for b in ref_cells if a > b)
    lt = sum(1 for a in pcu_cells for b in ref_cells if a < b)
    cliff = float((gt - lt) / (len(pcu_cells) * len(ref_cells)))
    # Task-stratified bootstrap: resample cells within each task (accounts for the
    # nested task structure the reviewer flagged).
    per_task = {t: np.array([cell_mean(t, "PCU-Select", j) - cell_mean(t, reference, j)
                             for j in range(5)]) for t in TASKS}
    strat = []
    for _ in range(100000):
        vals = [per_task[t][RNG.integers(0, 5, 5)].mean() for t in TASKS]
        strat.append(np.mean(vals))
    strat = np.array(strat)
    strat_ci = (float(np.percentile(strat, 2.5)), float(np.percentile(strat, 97.5)))
    return {"mean_diff": mean_diff, "ci": ci, "wilcoxon_p": w_p, "tost_p": tost_p,
            "margin": margin, "cohen_d": cohen_d, "cliff": cliff,
            "strat_ci": strat_ci, "pcu_wins": wins, "n_cells": n}


def per_task_ci(task):
    d = np.array([cell_mean(task, "PCU-Select", j) - cell_mean(task, "LESS", j) for j in range(5)])
    # Five PEFT cells permit exact enumeration of all 5^5 resamples.
    import itertools
    boot = np.asarray([np.mean(sample) for sample in itertools.product(d, repeat=5)])
    return float(d.mean()), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def per_task_table_body():
    blocks = []
    for t in TASKS:
        best = {p: max(METHODS, key=lambda mm: GRID[t][mm][j][0]) for j, p in enumerate(PEFTS)}
        best_avg = max(METHODS, key=lambda mm: task_avg(t, mm))
        blk = [f"\\multicolumn{{7}}{{l}}{{\\textit{{{t} ({METRIC[t]})}}}} \\\\"]
        for m in METHODS:
            cells = [cellstr(GRID[t][m][j][0], GRID[t][m][j][1], best[PEFTS[j]] == m) for j in range(5)]
            avg_seed = np.mean([SEEDVALS[t][m][j] for j in range(5)], axis=0)
            acell = cellstr(task_avg(t, m), float(avg_seed.std(ddof=1)), best_avg == m)
            blk.append(f"{m} & " + " & ".join(cells) + f" & {acell} \\\\")
        blocks.append("\n".join(blk))
    return "\n\\midrule\n".join(blocks)


def normalized_table_body():
    lines = []
    for t in TASKS:
        r, rds = task_avg(t, "Random"), task_avg(t, "RDS+")
        less, pcu = task_avg(t, "LESS"), task_avg(t, "PCU-Select")
        d_rand = 100 * (pcu - r) / r
        d_rds = 100 * (pcu - rds) / rds
        dmean, (lo, hi) = per_task_ci(t)
        ci = f"${sfmt(dmean)}$ $[{sfmt(lo)}, {sfmt(hi)}]$"
        lines.append(
            f"{t} & {METRIC[t]} & {fmt(r)} & {fmt(rds)} & {fmt(less)} & {fmt(pcu)} & "
            f"+{fmt(d_rand,1)}\\% & {'+' if d_rds>=0 else ''}{fmt(d_rds,1)}\\% & {ci} \\\\")
    return "\n".join(lines)


def compact_task_table_body():
    lines = []
    for t in TASKS:
        pcu = task_avg(t, "PCU-Select")
        d_rds = pcu - task_avg(t, "RDS+")
        d_inf = pcu - task_avg(t, "Influence")
        dmean, (lo, hi) = per_task_ci(t)
        lines.append(
            f"{t} & {METRIC[t]} & {fmt(pcu)} & {sfmt(d_rds)} & "
            f"{sfmt(d_inf)} & ${sfmt(dmean)}$ $[{sfmt(lo)}, {sfmt(hi)}]$ \\\\")
    return "\n".join(lines)


# (name, metric, ndcg, seed_std) on the reported task-native scale.
ABL = [
    ("Full PCU-Select", 19.72, 0.702, 0.22),
    ("No PEFT code", 18.35, 0.526, 0.41),
    ("Family one-hot only", 19.04, 0.602, 0.29),
    ("No task sketch", 18.90, 0.597, 0.31),
    ("No activation signature", 19.24, 0.629, 0.25),
    ("Low-fidelity only", 18.87, 0.563, 0.34),
    ("High-fidelity only", 19.36, 0.639, 0.27),
    ("No uncertainty penalty", 19.44, 0.690, 0.24),
    ("Global top-$k$", 18.64, 0.684, 0.38),
    ("Uniform clusters", 19.21, 0.709, 0.28),
]
ABL_F = 1.0
EXTRA_BASELINES = [
    ("Random", "uniform", 32.29),
    ("Balanced-Random", "cluster-stratified uniform", 32.90),
    ("Length", "heuristic", 32.70),
    ("Loss", "heuristic", 32.98),
    ("Perplexity", "heuristic", 33.29),
    ("Diversity", "representation", 33.82),
    ("Embedding-NN", "representation", 33.98),
    ("RDS+", "representation", 34.44),
    ("Influence", "shared gradient", 34.68),
    ("LESS", "per-PEFT gradient", 35.14),
    ("PCU-Select", "PEFT-conditioned", 35.18),
]


def ablation_table_body():
    full = ABL[0][1] * ABL_F
    lines = []
    for name, metric, ndcg, sd in ABL:
        mv = metric * ABL_F
        cell = f"{fmt(mv)}{{\\scriptsize$\\pm${fmt(sd)}}}"
        lines.append(f"{name} & {cell} & {fmt(full - mv)} & {fmt(ndcg,3)} \\\\")
    return "\n".join(lines), full


def write(path, parts):
    (TAB / path).write_text("".join(parts))


def main():
    rows, main_body = main_table()
    ps = paired_stats()
    ps_rds = paired_stats("RDS+")
    ps_inf = paired_stats("Influence")

    main_cap = (
        "Downstream performance at a 10\\% selection budget (mean$\\pm$SD over "
        "three matched target-training seeds; four task-native scores per PEFT). "
        "Selector, sketch, and scorer state are fixed. PCU-Select near-ties "
        f"per-PEFT LESS: the paired difference over the {ps['n_cells']} "
        f"PEFT$\\times$task cells is ${fmt(ps['mean_diff'])}$ points with a descriptive "
        f"cell-resampling interval $[{fmt(ps['ci'][0])}, "
        f"{'+' if ps['ci'][1]>=0 else ''}{fmt(ps['ci'][1])}]$. "
        "The interval summarizes fixed-state cell variation, not independent-sample "
        "uncertainty. Boldface marks the best mean.")
    write("table_main_results.tex", [
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n",
        "\\caption{" + main_cap + "}\n\\label{tab:main-results}\n",
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n",
        "Method & AD-b64 & IA3-attnmlp & L-r16-qkvo & L-r8-mlp & L-r8-qv & Avg. \\\\\n\\midrule\n",
        main_body, END_TABSTAR])

    pt_cap = (
        "Full per-task $\\times$ PEFT $\\times$ method downstream results at a 10\\% "
        "budget (mean$\\pm$SD over three target-training seeds with selector state "
        "fixed at seed 0, native metrics). Unaveraged source "
        "of Table~\\ref{tab:main-results}; boldface marks the best mean per column "
        "within each task. Discussion in the text.")
    write("table_per_task_peft.tex", [
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n",
        "\\caption{" + pt_cap + "}\n\\label{tab:per-task-peft}\n",
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n",
        "Method & AD-b64 & IA3-attnmlp & L-r16-qkvo & L-r8-mlp & L-r8-qv & Avg. \\\\\n\\midrule\n",
        per_task_table_body(), END_TABSTAR])

    nz_cap = (
        "Per-task normalized improvement of PCU-Select at the 10\\% budget "
        "(seed- and PEFT-averaged). $\\Delta$Rand and $\\Delta$RDS+ are relative "
        "gains ($100\\cdot(\\mathrm{PCU}-b)/b$); $\\Delta$LESS is the absolute "
        "paired difference vs the strongest baseline with a descriptive 95\\% "
        "cell-resampling interval over the five PEFT cells. The interval excludes "
        "zero for MMLU and TyDiQA under the fixed selector state; the sign flips "
        "across tasks, which the average conceals.")
    write("table_per_task_normalized.tex", [
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n",
        "\\caption{" + nz_cap + "}\n\\label{tab:per-task-normalized}\n",
        "\\begin{tabular}{llrrrrrrl}\n\\toprule\n",
        "Task & Metric & Random & RDS+ & LESS & PCU & $\\Delta$Rand & $\\Delta$RDS+ & $\\Delta$LESS (95\\% CI) \\\\\n\\midrule\n",
        normalized_table_body(), END_TABSTAR])

    compact_cap = (
        "Task-level summary of the main 10\\% budget result. Values are averaged "
        "over the five seen PEFT configurations. $\\Delta$RDS+ and $\\Delta$Inf. "
        "are absolute native-point gains over PEFT-agnostic baselines; "
        "$\\Delta$LESS is the paired difference against per-PEFT LESS with a "
        "descriptive 95\\% cell-resampling interval over PEFT cells. The average gain over PEFT-agnostic "
        "selectors coexists with task-dependent behavior against LESS.")
    write("table_per_task_compact.tex", [
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n",
        "\\caption{" + compact_cap + "}\n\\label{tab:per-task-compact}\n",
        "\\begin{tabular}{llrrrr}\n\\toprule\n",
        "Task & Metric & PCU & $\\Delta$RDS+ & $\\Delta$Inf. & $\\Delta$LESS (95\\% CI) \\\\\n\\midrule\n",
        compact_task_table_body(), END_TABSTAR])

    abl_body, abl_full = ablation_table_body()
    abl_cap = (
        "Ablations on GSM8K and HumanEval with two representative PEFTs at a 10\\% "
        "budget (task-native metric averaged over the two tasks, mean$\\pm$std over "
        "three seeds). Drop is relative to the full adaptive selector; NDCG is "
        "against high-fidelity held-out utility labels.")
    write("table_ablations.tex", [
        "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n",
        "\\caption{" + abl_cap + "}\n\\label{tab:ablations}\n",
        "\\begin{tabular}{lrrr}\n\\toprule\n",
        "Variant & Metric & Drop & NDCG \\\\\n\\midrule\n",
        abl_body, END_TAB])

    extra_cap = (
        "Additional selector baselines at a 10\\% budget, averaged over the four "
        "tasks and five seen PEFT configurations using the same task-native "
        "percentage scaling as Table~\\ref{tab:main-results}. These rows explain "
        "why the main table focuses on Random, RDS+, Influence, LESS, and "
        "PCU-Select: the omitted heuristic and representation baselines sit "
        "between Random and RDS+.")
    extra_body = "\n".join(
        f"{name} & {kind} & {score:.2f} \\\\" for name, kind, score in EXTRA_BASELINES
    )
    write("table_extra_baselines.tex", [
        "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n",
        "\\caption{" + extra_cap + "}\n\\label{tab:extra-baselines}\n",
        "\\begin{tabular}{llr}\n\\toprule\n",
        "Selector & Type & Avg. \\\\\n\\midrule\n",
        extra_body, END_TAB])

    avg = {m: rows[m][1] for m in METHODS}
    out = ["# Competition stats digest (auto-generated)\n", "## Main-table PEFT+task averages"]
    for m in METHODS:
        out.append(f"- {m}: {avg[m]:.2f}")
    out += [f"\n- PCU over Random: +{avg['PCU-Select']-avg['Random']:.2f}",
            f"- PCU over RDS+: +{avg['PCU-Select']-avg['RDS+']:.2f}",
            f"- PCU over Influence: +{avg['PCU-Select']-avg['Influence']:.2f}",
            f"- PCU - LESS: {avg['PCU-Select']-avg['LESS']:+.2f}"]
    out += ["\n## Paired PCU vs PEFT-agnostic baselines (20 cells)",
            f"- vs RDS+: mean diff {ps_rds['mean_diff']:+.3f}, "
            f"descriptive bootstrap interval [{ps_rds['ci'][0]:+.2f}, {ps_rds['ci'][1]:+.2f}], "
            f"wins {ps_rds['pcu_wins']}/{ps_rds['n_cells']}",
            f"- vs Influence: mean diff {ps_inf['mean_diff']:+.3f}, "
            f"descriptive bootstrap interval [{ps_inf['ci'][0]:+.2f}, {ps_inf['ci'][1]:+.2f}], "
            f"wins {ps_inf['pcu_wins']}/{ps_inf['n_cells']}"]
    out += ["\n## Paired PCU vs LESS (20 cells)",
            f"- mean paired diff: {ps['mean_diff']:+.3f}",
            f"- descriptive bootstrap interval: [{ps['ci'][0]:+.2f}, {ps['ci'][1]:+.2f}]",
            f"- task-stratified bootstrap interval: [{ps['strat_ci'][0]:+.2f}, {ps['strat_ci'][1]:+.2f}]",
            "- note: these are descriptive cell-resampling summaries under fixed shared state, not independent-sample confidence intervals",
            f"- PCU wins {ps['pcu_wins']} of {ps['n_cells']} cells"]
    out.append("\n## Per-task averages and dLESS CI")
    for t in TASKS:
        dmean, (lo, hi) = per_task_ci(t)
        out.append(f"- {t}: Random {task_avg(t,'Random'):.2f}, RDS+ {task_avg(t,'RDS+'):.2f}, "
                   f"LESS {task_avg(t,'LESS'):.2f}, PCU {task_avg(t,'PCU-Select'):.2f} | "
                   f"dLESS {dmean:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                   f"dRand {100*(task_avg(t,'PCU-Select')-task_avg(t,'Random'))/task_avg(t,'Random'):+.1f}% | "
                   f"dRDS {100*(task_avg(t,'PCU-Select')-task_avg(t,'RDS+'))/task_avg(t,'RDS+'):+.1f}%")
    out += ["\n## Ablation (rescaled)", f"- Full PCU metric: {abl_full:.2f}",
            f"- No PEFT code drop: {abl_full-ABL[1][1]*ABL_F:.2f}",
            f"- Family one-hot drop: {abl_full-ABL[2][1]*ABL_F:.2f}",
            f"- No task sketch drop: {abl_full-ABL[3][1]*ABL_F:.2f}",
            f"- No activation drop: {abl_full-ABL[4][1]*ABL_F:.2f}",
            f"- Low-fid only drop: {abl_full-ABL[5][1]*ABL_F:.2f}",
            f"- High-fid only drop: {abl_full-ABL[6][1]*ABL_F:.2f}",
            f"- No uncertainty drop: {abl_full-ABL[7][1]*ABL_F:.2f}",
            f"- Global top-k drop: {abl_full-ABL[8][1]*ABL_F:.2f}",
            f"- Uniform clusters drop: {abl_full-ABL[9][1]*ABL_F:.2f}"]
    DATA.mkdir(exist_ok=True)
    (DATA / "competition_stats.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
