# 3.1 Main-result advantage unjudgeable -> reframe + variance/significance

**Track: A** (reframe + statistics from existing data; no reruns)

## Reviewer concern

Table 2 shows PCU-Select 42.81 vs LESS 42.73 (0.08 pt), and PCU loses L-r8-mlp
by 0.73 pt; wins only 10/20 cells. No std / SE / CI / paired significance test.
The 0.08-pt gap likely sits inside seed noise. Cannot support "better than the
strongest influence selector."

## Guiding principle (user directive)

Fix is **framing, not method optimization**. Our selling point is low cross-PEFT
transfer cost — approaching the strongest method at lower amortized cost — not
raw performance superiority. So we demote the performance claim and promote the
amortization claim.

## Current state

- `paper/Sections/00_abstract.tex`: "while matching LESS on average".
- `paper/Sections/05_experiments.tex:18`: "improves selected-data quality",
  "matches LESS on average but does not dominate", "wins 10 of 20 cells".
- `paper/tables/table_main_results.tex`: PEFT-averaged only, no dispersion.

## Planned changes

1. **Reframe the headline claim** everywhere it appears (abstract, intro
   contributions, experiments main-results paragraph) to:
   > PCU-Select achieves comparable average downstream performance to per-PEFT
   > LESS while amortizing PEFT-specific selection cost across multiple target
   > configurations.
   Remove wording that implies dominance over LESS ("improves selected-data
   quality" where it reads as beating LESS). Keep gains over Random/RDS+ (those
   are real and defensible).
2. **Add dispersion to Table 2.** Per-cell seed std (or 95% CI) from the 3
   seeds in `E1.jsonl`. Present `mean +/- std` or add a companion appendix
   table with CIs.
3. **Add a paired significance test PCU vs LESS** across the 20 PEFT x task
   cells: Wilcoxon signed-rank + paired bootstrap CI on the mean difference.
   Expected honest outcome: not significant -> which *supports* the "comparable"
   reframe. Report the p-value / CI in a footnote or the results paragraph.
4. **Recast the 10/20 + L-r8-mlp loss** as evidence of parity, not weakness.

## Data sources / files

- Read: `result/data/E1.jsonl` (filter budget=0.1, methods pcu/less/rds_plus/
  influence/random; group by peft,task,seed).
- Extend: `scripts/paper/generate_paper_assets.py` to emit std/CI + the paired
  test result into `paper/tables/` (or `paper/data/`).
- Write: `paper/tables/table_main_results.tex` (dispersion),
  `paper/Sections/00_abstract.tex`, `01_introduction.tex`, `05_experiments.tex`.

## Reruns needed

No. All from existing `E1.jsonl`.

## Note on "influence" vs "grad_sim"

`E1.jsonl` methods include both `grad_sim` and `less`; the paper's "Influence"
row maps to one of them. Confirm the mapping in
`src/pcu_select/baselines/selectors.py` before recomputing so the table labels
stay faithful.

## Acceptance criteria

- Table 2 reports dispersion; a paired PCU-vs-LESS test is reported.
- No sentence claims PCU beats LESS on performance; amortization is the headline.
