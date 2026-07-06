# 3.2 Heterogeneous-metric averaging -> full per-task tables + protocols

**Track: A** (stats from existing data + documentation; no reruns)

## Reviewer concern

GSM8K EM, HumanEval Pass@1, MMLU acc, TyDiQA F1 are scaled to percent and
averaged. Different difficulty/variance/ceiling means "1 pt here != 1 pt there".
The compressed average looks selective. Need: full per-task x PEFT table,
per-task normalized improvement vs random/RDS+/LESS, per-task seed std / CI, and
explicit eval protocols (esp. HumanEval decoding).

## Current state

- Only the PEFT-averaged Table 2 is shown; per-task numbers exist in raw data
  but are not surfaced.
- Only `configs/task/gsm8k.yaml` exists; humaneval/mmlu/tydiqa protocols must be
  reconstructed from code.

## Planned changes

1. **Full appendix table: task x PEFT x method.** 4 tasks x 5 PEFT x key methods
   (random, RDS+, influence, LESS, PCU) at 10% budget, from `E1.jsonl`. This
   replaces reliance on the single compressed average.
2. **Per-task normalized improvement** of PCU vs random / RDS+ / LESS, so
   cross-task point-differences are not treated as equivalent.
3. **Per-task seed std / bootstrap CI.**
4. **Document eval protocols** in the repro appendix:
   - HumanEval Pass@1: temperature, number of samples, greedy vs sampling.
   - GSM8K EM, MMLU acc, TyDiQA F1: decoding + scoring, confirm they match
     public-standard protocols or state deviations.
   Harvest from `src/pcu_select/eval/metrics.py`,
   `src/pcu_select/eval/target_train.py`, `configs/task/gsm8k.yaml`.

## Data sources / files

- Read: `result/data/E1.jsonl`, `src/pcu_select/eval/*`, `configs/task/*`.
- Extend: `scripts/paper/generate_paper_assets.py` for the per-task tables.
- Write: `paper/tables/` (new per-task table + normalized-improvement table),
  repro/appendix text under `paper/`.

## Reruns needed

No for tables (data exists). Protocol documentation is code-reading only.

## Honest-disclosure note

If the HumanEval decoding setting in code is non-standard (e.g. single greedy
sample reported as Pass@1), state it plainly rather than implying a multi-sample
Pass@1.

## Acceptance criteria

- A full per-task x PEFT table exists; averages are shown alongside, not instead.
- Every task's evaluation protocol is documented with concrete numbers.
