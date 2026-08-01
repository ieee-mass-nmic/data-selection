#!/usr/bin/env python3
"""Generate the auditable cost tables from the canonical cost-model JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data" / "competition_cost_model.json"
TABLES = ROOT / "paper" / "tables"


def load_cost_model() -> dict:
    return json.loads(DATA.read_text())


def validate_cost_model(model: dict) -> None:
    scope = model["online_selection_scope"]
    pair = scope["pcu_per_peft_task_gpu_h"]
    four_task = scope["pcu_four_task_per_peft_gpu_h"]
    if abs(model["task_count"] * pair - four_task) > 1e-12:
        raise ValueError("PCU per-task and four-task costs are inconsistent")
    for row in model["pool_scaling_gpu_h"]:
        phase_sum = (
            row["scorer_forward"]
            + row["clustering"]
            + row["unattributed_residual"]
        )
        if abs(phase_sum - row["online_total"]) > 1e-12:
            raise ValueError(f"pool-scaling row {row['pool']} does not sum")


def write_cost_axes(model: dict) -> None:
    scope = model["online_selection_scope"]
    body = f"""\\begin{{table}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{lrr}}
\\toprule
New registry entry & PCU-Select & LESS \\\\
\\midrule
One new task, existing PEFT   & {scope['pcu_per_peft_task_display_gpu_h']:.2f} & {scope['less_ranking_per_peft_task_gpu_h']:.2f} \\\\
One new PEFT, four tasks       & {scope['pcu_four_task_per_peft_gpu_h']:.2f} & {scope['less_four_task_per_peft_gpu_h']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{The amortization advantage is cross-PEFT, not cross-task. Adding a task
to an existing PEFT costs each selector one online-selection pass; adding a PEFT for
four tasks forces LESS to rebuild its datastore while PCU-Select reuses the
offline scorer. Selection-stage GPU-hours; PCU-Select's per-task entry rounds
the exact average $1.51/4=0.3775$.}}
\\label{{tab:cost-axes}}
\\end{{table}}
"""
    (TABLES / "table_cost_axes.tex").write_text(body)


def write_pool_scaling(model: dict) -> None:
    rows = []
    for row in model["pool_scaling_gpu_h"]:
        rows.append(
            f"{row['pool']} & {row['feature_extraction_cached']:.1f} & "
            f"{row['scorer_forward']:.3f} & {row['clustering']:.3f} & "
            f"{row['unattributed_residual']:.3f} & {row['online_total']:.3f} \\\\"
        )
    body = """\\begin{table}[t]
\\centering
\\small
\\setlength{\\tabcolsep}{2.5pt}
\\begin{tabular}{rrrrrr}
\\toprule
Pool & \\shortstack{Feature\\\\ext.\\ (cached)} & \\shortstack{Scorer\\\\forward} & Clustering & Residual & \\shortstack{PCU online\\\\(per $p,t$)} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\caption{Online selection cost versus candidate-pool size on $8{\\times}$H20-96GB.
Feature extraction is cached. Scorer and clustering columns report existing
phase timers. Residual is total minus these timers; at 300K, the revised
four-task measurement gives 0.3775 per pair, while phase timers account for
0.360, leaving 0.0175 pending attribution. Online columns show three decimals.}
\\label{tab:pool-scaling}
\\end{table}
"""
    (TABLES / "table_pool_scaling.tex").write_text(body)


def main() -> None:
    model = load_cost_model()
    validate_cost_model(model)
    TABLES.mkdir(parents=True, exist_ok=True)
    write_cost_axes(model)
    write_pool_scaling(model)
    print("wrote table_cost_axes.tex and table_pool_scaling.tex")


if __name__ == "__main__":
    main()
