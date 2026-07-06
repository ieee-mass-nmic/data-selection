# PCU-Select Revision Plan (AAAI-27)

Refinement plan responding to the expert review points 3.1–3.8. One file per
aspect. Each file is tagged **Track A** (presentation + statistics from data we
already have, no reruns, writes only under `paper/` + `scripts/paper/`) or
**Track B** (new compute, touches `src/`/`result/`, needs explicit go before
each run).

## Track split

- **Track A — Presentation + stats from existing data.** ~80% of the review is
  solvable here. `result/data/E1.jsonl` already holds full
  per-task x per-PEFT x per-method x 3-seed x 3-budget values for all 11
  baselines; the main table just collapses it to PEFT-averages.
- **Track B — New compute (authorized items only).** Per user decision:
  - Minimal second backbone (Qwen/Mistral, 2 tasks x 2 PEFT x 5%) — see `07`.
  - Low-fidelity proxy promoted into the full main table — see `04`.
  - NOT authorized: oracle/upper-bound and short-horizon<->downstream
    correlation figure. Handled by wording/limitation instead (see `04`, `05`).

## Write boundaries (from repo + paper CLAUDE.md)

- Paper edits only under `paper/`; figures under `paper/Figures/`; tables under
  `paper/tables/`; intermediate data under `paper/data/`.
- `src/`, `configs/`, `docs/`, `result/` are read-only for paper tasks; Track B
  reruns require explicit instruction and will state exact write scope first.
- Result-export utility lives at `scripts/paper/generate_paper_assets.py` and
  may be extended; it must not modify source results.

## Files

| File | Aspect | Track |
|---|---|---|
| `01_main_result_reframing.md` | 3.1 unjudgeable advantage -> reframe + variance/significance | A |
| `02_metric_averaging.md` | 3.2 heterogeneous-metric averaging -> full per-task tables | A |
| `03_reproducibility.md` | 3.3 reproducibility supplement | A |
| `04_baselines.md` | 3.4 baseline definitions + low-fid proxy in main table | A + B |
| `05_empirical_assumptions.md` | 3.5 unverified assumptions + Fig 1 relabel | A |
| `06_cost_claim.md` | 3.6 cost claim scoping | A |
| `07_generalization.md` | 3.7 cross-backbone | B |
| `08_consistency_polish.md` | 3.8 internal consistency + polish | A |

## Sequencing

1. **Phase 1 (Track A, no reruns):** 01, 08, 05 (Fig 1 relabel), 06.
2. **Phase 2 (Track A, documentation):** 02, 03, 04 (definitions part).
3. **Phase 3 (Track B, authorized compute):** 04 (low-fid proxy run), 07.

## Stale-number reconciliation

The review quotes cost numbers (31.4 GPU-h offline / 18.2 hi-fidelity /
break-even 5.17) that predate the current draft, which already says
11.2 / 4.5 / 1.63 in `paper/Sections/05_experiments.tex`. Part of 3.6 was
revised after this review; plan is written against the current draft. Confirm
current numbers from `result/data/E2.jsonl` + `E2_cost_model.json` before
editing (see `06`).
