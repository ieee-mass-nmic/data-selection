# PCU-Select Result Bundle

This directory contains the result bundle used by the paper figures, tables,
and repository report. The files use the same names and row schema as the
experiment runners, so exported runner outputs can refresh the bundle directly.

Table 1 is computed from the PEFT registry and parameter-counting code. The
E1--E5 and motivation files follow the flat `ResultRow` schema shared by the
experiment and plotting scripts.

## Refresh Figures And Tables

```bash
pip install -e ".[viz]"

python result/validate_result_bundle.py
python scripts/experiments/build_table1.py --model llama2-7b --out-dir result/tables

cd scripts/plots
R=../../result/data
F=../../result/figs
python plot_e1.py            --results $R/E1.jsonl --out-dir $F
python plot_e2.py            --results $R/E2.jsonl --cost-model $R/E2_cost_model.json --out-dir $F
python plot_e3.py            --results $R/E3.jsonl --out-dir $F
python plot_e4.py            --results $R/E4.jsonl --overlap $R/E4_overlap.json --out-dir $F
python plot_e5.py            --results $R/E5.jsonl --out-dir $F
python plot_motivation_f1.py --values  $R/motivation/values.parquet --signal u_hi   --out-dir $F
python plot_motivation_f1.py --values  $R/motivation/values.parquet --signal u_grad --out-dir $F
python plot_motivation_f2.py --results $R/MOT_F2.jsonl --out-dir $F
```

## Directory Layout

```text
result/
  README.md
  validate_result_bundle.py
  data/
    E1.jsonl
    E2.jsonl
    E2_cost_model.json
    E3.jsonl
    E4.jsonl
    E4_overlap.json
    E5.jsonl
    MOT_F2.jsonl
    motivation/values.parquet
  figs/
    F1_budget_sensitivity.png
    F1_disagreement_u_grad.png
    F1_disagreement_u_hi.png
    F2_break_even.png
    F2_transfer.png
    F3_pareto.png
    F4_alpha_sweep.png
    F4b_strategy.png
    F5_mismatch.png
    F6_config_vs_selection.png
    F7_levels_modes.png
    F8_d2_vs_degradation.png
    T1_method_x_peft.png
    T2_ablation.png
  tables/
    table1.{csv,md,tex}
    T1_method_x_peft.csv
    T2_ablation.csv
    F1_structural_u_{grad,hi}.csv
    F2_transfer_raw.csv
```

## Artifact Map

| Artifact | Files | Purpose | Data source |
|---|---|---|---|
| Table 1 | `tables/table1.{csv,md,tex}` | PEFT configuration structure and trainable parameter counts | Registry parameter counter |
| Motivation F1 | `figs/F1_disagreement_u_hi.png`, `figs/F1_disagreement_u_grad.png`, `tables/F1_structural_*.csv` | Ranking disagreement across PEFT configurations | `data/motivation/values.parquet` |
| Motivation F2 | `figs/F2_transfer.png`, `tables/F2_transfer_raw.csv` | Cross-PEFT transfer matrix | `data/MOT_F2.jsonl` |
| E1/T1 | `figs/T1_method_x_peft.png`, `tables/T1_method_x_peft.csv` | Method by PEFT summary at 10 percent budget | `data/E1.jsonl` |
| E1/F1 | `figs/F1_budget_sensitivity.png` | Budget sensitivity at 5, 10, and 30 percent | `data/E1.jsonl` |
| E2/F2 | `figs/F2_break_even.png` | Total GPU-hours as target count increases | `data/E2.jsonl`, `data/E2_cost_model.json` |
| E2/F3 | `figs/F3_pareto.png` | Performance versus total selection cost | `data/E2.jsonl`, `data/E2_cost_model.json` |
| E3/T2 | `figs/T2_ablation.png`, `tables/T2_ablation.csv` | Component ablation summary | `data/E3.jsonl` |
| E3/F4 | `figs/F4_alpha_sweep.png`, `figs/F4b_strategy.png` | Cluster-quota and strategy sensitivity | `data/E3.jsonl` |
| E4/F5 | `figs/F5_mismatch.png` | Source-target PEFT mismatch matrix | `data/E4.jsonl` |
| E4/F6 | `figs/F6_config_vs_selection.png` | Selection overlap across PEFT configurations | `data/E4_overlap.json` |
| E5/F7 | `figs/F7_levels_modes.png` | ID/OOD behavior and calibration modes | `data/E5.jsonl` |
| E5/F8 | `figs/F8_d2_vs_degradation.png` | Mahalanobis distance versus degradation | `data/E5.jsonl` |

## Data Conventions

- `metric` is higher-is-better and uses the task-native unit named by
  `metric_name`.
- `eval_loss` is held-out response-LM loss and remains available for each row.
- Ranking metrics use bounded numeric values for every row so the JSONL files
  stay strict and spreadsheet-friendly.
- Cost fields use GPU-hours. `select_gpu_h` measures selection work,
  `target_train_gpu_h` measures target PEFT training, and
  `offline_gpu_h` is reserved for amortized one-time costs.
- `extra.n_selected` records the number of selected training samples for every
  row.

## Replacement Workflow

To refresh the bundle from a completed experiment, export the runner outputs to
the matching filenames under `result/data/`. Then run the validation and plotting
commands above. The figure and table paths remain stable, so the manuscript can
keep the same includes.
