# 3.6 Cost claim scoping

**Track: A** (recompute from existing E2 data + reword; no reruns)

## Reviewer concern

1. Online cost "including selection and fixed target training" conflates the
   shared target-training term; strip it when comparing selectors.
2. Random/RDS+ are cheaper; abstract cost claim must not read as "overall more
   efficient".
3. Unseen-PEFT 200/500 calibration-label cost not in the break-even figure.
4. GPU model / batching / caching / parallelism unstated.
5. If a user has only 1-3 PEFT targets, PCU has no cost advantage.

## Current state

- `paper/Sections/05_experiments.tex:31`: offline 11.2 GPU-h, online 0.925
  (includes target training), LESS 0.996 + 6.8 recompute, break-even 1.63.
  Already concedes RDS+/random are cheaper — part of this point is done.
- Review's numbers (31.4 / 18.2 / 5.17) are STALE vs current draft.

## Planned changes

1. **Report selector-only online cost** (strip shared target-training) so the
   selector comparison is clean; keep the full-cost number separately.
2. **Add calibration cost (200/500 labels) to the break-even figure**
   `paper/Figures/fig_cost_break_even.pdf` for unseen-PEFT targets.
3. **Constrain the abstract cost claim** to "amortized advantage vs per-PEFT
   influence/LESS recomputation"; explicitly state Random/RDS+ remain cheaper
   and that advantage requires enough targets to cross break-even.
4. **Document GPU model / batching / caching / parallelism** from
   `src/pcu_select/cost/accounting.py` + `result/data/E2_cost_model.json`.
5. **Reconcile stale numbers**: recompute offline / online / break-even from
   `result/data/E2.jsonl` + `E2_cost_model.json`; ensure text, abstract, and
   figure all agree.

## Data sources / files

- Read: `result/data/E2.jsonl`, `result/data/E2_cost_model.json`,
  `src/pcu_select/cost/accounting.py`.
- Regenerate figure via its plot script (`scripts/plots/plot_e2.py`) or the
  paper-asset exporter; write only under `paper/Figures/`.
- Write: `paper/Sections/00_abstract.tex`, `05_experiments.tex`, cost figure +
  caption.

## Reruns needed

No experiment reruns; only recompute/replot from existing cost data.

## Acceptance criteria

- Selector-only vs full cost separated; calibration cost in break-even figure.
- Cost claim scoped to amortization; no "overall more efficient" reading.
- All cost numbers consistent across text/abstract/figure.
