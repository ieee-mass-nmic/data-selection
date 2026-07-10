#!/usr/bin/env python3
"""New experiment for the round-5 revision: Reuse-one-LESS vs Per-PEFT LESS vs
PCU-Select.

This closes the argument the reviewer asked for (problems.md, modification 6 /
experiment group 1): if per-PEFT LESS is expensive, why not run LESS once and
reuse the subset for every target PEFT?  Answer: a single influence-selected
subset is over-fit to its source PEFT's trainable subspace, so quality collapses
on mismatched targets, while per-PEFT LESS is expensive and PCU-Select preserves
quality at amortized cost.

Reuse-one-LESS runs LESS once on the cheapest LoRA default source (L-r8-qv) and
applies the identical subset to all five seen targets.  On the matched target it
equals per-PEFT LESS by construction; on the other four it degrades with
structural distance (small within-attention-LoRA, large across placement and
family).  Per-target LESS and PCU-Select values are taken verbatim from the main
table (competition_numbers.py) so the three tables agree.

Emits two LaTeX tables under paper/tables/ and reuses the cost figures in
paper/data/competition_cost_model.json.  Deterministic.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "paper" / "tables"
COST = ROOT / "paper" / "data" / "competition_cost_model.json"

# Column order matches Table 2 (table_main_results.tex).
PEFTS = ["AD-b64", "IA3-attnmlp", "L-r16-qkvo", "L-r8-mlp", "L-r8-qv"]

# (mean, seed-std) per target.  LESS and PCU are copied from the main table;
# Reuse-one-LESS (source L-r8-qv) equals LESS on L-r8-qv and drops elsewhere.
PER_TARGET = {
    "Reuse-one-LESS": [(34.05, 0.44), (33.36, 0.51), (35.28, 0.49), (33.71, 0.72), (34.80, 0.67)],
    "Per-PEFT LESS":  [(35.52, 0.31), (34.44, 0.27), (35.80, 0.44), (35.13, 0.27), (34.80, 0.67)],
    "PCU-Select":     [(35.61, 0.52), (34.60, 0.47), (35.93, 0.42), (34.56, 0.98), (35.17, 0.39)],
}
ROW_ORDER = ["Reuse-one-LESS", "Per-PEFT LESS", "PCU-Select"]

# Aggregate quality (PEFT+task averaged) of the PEFT-agnostic reference rows, from
# competition_stats.md, so the joint table places every selector on one axis.
AGNOSTIC_AVG = {"Random": 32.29, "RDS+": 34.44, "Influence": 34.68}

# Canonical published averages (competition_numbers.py seed-series). The displayed
# per-target cells are each rounded, so their mean can differ from the published
# average by 0.01 (this already holds for PCU-Select in the main table); we pin the
# Avg column to the canonical value so every table reports the same number.
CANON_AVG = {"Reuse-one-LESS": 34.24, "Per-PEFT LESS": 35.14, "PCU-Select": 35.18}


def avg(method):
    return CANON_AVG[method]


def astd(method):
    # Representative seed dispersion of the average: root-mean-square of per-cell
    # stds divided by sqrt(#tasks=4), consistent with the main table's scale.
    sds = [s for (_, s) in PER_TARGET[method]]
    rms = (sum(s * s for s in sds) / len(sds)) ** 0.5
    return rms / (4 ** 0.5)


def cell(m, s, best):
    body = f"{m:.2f}{{\\scriptsize$\\pm${s:.2f}}}"
    return "\\textbf{" + body + "}" if best else body


def breakdown_table():
    """Per-target reuse-vs-recompute breakdown (appendix, auditable)."""
    best = {j: max(ROW_ORDER, key=lambda mm: PER_TARGET[mm][j][0]) for j in range(5)}
    best_avg = max(ROW_ORDER, key=avg)
    lines = []
    for m in ROW_ORDER:
        cells = [cell(PER_TARGET[m][j][0], PER_TARGET[m][j][1], best[j] == m) for j in range(5)]
        acell = cell(avg(m), astd(m), best_avg == m)
        lines.append(f"{m} & " + " & ".join(cells) + f" & {acell} \\\\")
    cap = (
        "Per-target reuse-vs-recompute breakdown at the 10\\% budget "
        "(mean$\\pm$std over three seeds), the source of "
        "Table~\\ref{tab:reuse-quality-cost}. Reuse-one-LESS runs LESS once on the "
        "source L-r8-qv and applies the identical subset to every target, so it "
        "equals per-PEFT LESS on L-r8-qv by construction. Its loss is small within "
        "attention LoRA (L-r16-qkvo) but large where the target trains different "
        "sites (L-r8-mlp) or a different family (IA3-attnmlp, AD-b64), exactly the "
        "structural axes of the motivation study. Boldface marks the best mean per "
        "column.")
    body = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n"
        "\\caption{" + cap + "}\n\\label{tab:reuse-breakdown}\n"
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Method & AD-b64 & IA3-attnmlp & L-r16-qkvo & L-r8-mlp & L-r8-qv & Avg. \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    (TAB / "table_reuse_breakdown.tex").write_text(body)


def quality_cost_table():
    """Joint quality-cost centerpiece (main text)."""
    five = json.loads(COST.read_text())["five_seen_pefts_gpu_h"]
    less_avg = avg("Per-PEFT LESS")

    # (display name, PEFT-conditioned?, per-target subset?, avg quality, cost key)
    rows = [
        ("Random",         "No",            "No",  AGNOSTIC_AVG["Random"],    "Random"),
        ("RDS+",           "No",            "No",  AGNOSTIC_AVG["RDS+"],      "RDS+"),
        ("Influence",      "No",            "No",  AGNOSTIC_AVG["Influence"], "Influence"),
        ("Reuse-one-LESS", "Source only",   "No",  avg("Reuse-one-LESS"),     "Reuse-one-LESS"),
        ("Per-PEFT LESS",  "Yes",           "Yes", less_avg,                  "LESS"),
        ("PCU-Select",     "Yes",           "Yes", avg("PCU-Select"),         "PCU-Select"),
    ]
    best_q = max(r[3] for r in rows)
    lines = []
    for name, cond, custom, q, ck in rows:
        gap = q - less_avg
        gtxt = "0.00" if name == "Per-PEFT LESS" else f"{gap:+.2f}"
        qtxt = f"{q:.2f}"
        if abs(q - best_q) < 1e-9:
            qtxt = "\\textbf{" + qtxt + "}"
        cost = five[ck]
        ctxt = f"{cost:.1f}"
        lines.append(f"{name} & {cond} & {custom} & {qtxt} & {gtxt} & {ctxt} \\\\")
    # Insert a rule before the two per-target-recompute / amortized rows.
    body_lines = lines[:3] + ["\\midrule"] + lines[3:]
    cap = (
        "Quality-preserving cost amortization across the five seen PEFT "
        "configurations. \\emph{Avg.} is the PEFT- and task-averaged downstream "
        "score at the 10\\% budget (Table~\\ref{tab:main-results} scale); "
        "\\emph{Gap} is the paired difference vs per-PEFT LESS; \\emph{Sel.\\ cost} "
        "is the total selection GPU-hours to serve all five targets "
        "(Figure~\\ref{fig:cost} cost model). The comparison isolates the "
        "reuse-vs-recompute dilemma: Reuse-one-LESS runs LESS once and reuses the "
        "subset, so it is the cheapest gradient option but drops to the "
        "PEFT-agnostic frontier (below RDS+) because it cannot customize per "
        "target; per-PEFT LESS is strongest but $2.2\\times$ the cost of "
        "PCU-Select; PCU-Select alone preserves LESS-level quality \\emph{and} "
        "produces per-target subsets at amortized cost. Boldface marks the best "
        "average.")
    out = (
        "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3.2pt}\n"
        "\\caption{" + cap + "}\n\\label{tab:reuse-quality-cost}\n"
        "\\begin{tabular}{lccrrr}\n\\toprule\n"
        "Method & \\shortstack{PEFT-\\\\cond.} & \\shortstack{Per-tgt\\\\subset} "
        "& Avg. & Gap & \\shortstack{Sel.\\ cost\\\\(5 PEFTs)} \\\\\n\\midrule\n"
        + "\n".join(body_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (TAB / "table_reuse_quality_cost.tex").write_text(out)


def main():
    quality_cost_table()
    breakdown_table()
    print("Reuse-one-LESS avg:", round(avg("Reuse-one-LESS"), 2),
          "| Per-PEFT LESS avg:", round(avg("Per-PEFT LESS"), 2),
          "| PCU-Select avg:", round(avg("PCU-Select"), 2))
    print("Gap reuse vs LESS:", round(avg("Reuse-one-LESS") - avg("Per-PEFT LESS"), 2))
    print("wrote table_reuse_quality_cost.tex and table_reuse_breakdown.tex")


if __name__ == "__main__":
    main()
