# 3.4 Baseline definitions + low-fidelity proxy in main table

**Track: A** (definitions) **+ B** (low-fid proxy run, authorized)

## Reviewer concern

1. RDS+ underspecified (relation to prior work, tuning, same embedding,
   task-conditioned?).
2. "LESS-style" equivalence to original LESS unclear (gradient datastore,
   projection, checkpoints, validation-gradient set, per-PEFT recompute, shared
   pool/sketch?).
3. Missing key baseline in the MAIN experiment: site-weighted low-fidelity proxy
   + cluster quota (currently only a 2-task/2-PEFT ablation). Answers "how much
   does the complex scorer buy over the simple proxy?"
4. Suggested oracle/upper-bound.

## Decisions

- (1)(2) definitions -> **Track A**.
- (3) low-fid proxy into full main table -> **Track B, AUTHORIZED**.
- (4) oracle/upper-bound -> **NOT authorized**. Handle by wording: add one honest
  sentence that no attainable-ceiling reference is reported, list as future work.
  Do not imply a bound we did not measure.

## Track A — definitions (no reruns)

- Read `src/pcu_select/baselines/selectors.py`.
- Write precise definitions for RDS+ (embedding used, whether task-conditioned,
  tuning range) and the LESS-style baseline (what is replicated vs original LESS;
  whether recomputed per-PEFT; shared candidate pool and validation sketch). If
  it diverges from original LESS, keep the name "LESS-style" and state the deltas
  explicitly.
- Write into `paper/Sections/05_experiments.tex` setup + repro appendix.

## Track B — low-fidelity proxy into main table (AUTHORIZED)

- Goal: run `low-fidelity proxy + cluster quota` selection across the full main
  grid (4 tasks x 5 PEFT x 3 seeds, 10% budget) and add it as a main-table row.
- Touches `src/` + `result/`: will state exact write scope and confirm before
  running. Export to `result/data/` matching the `ResultRow` schema, then refresh
  via `scripts/paper/generate_paper_assets.py`.
- Check first whether the existing ablation (`result/data/E3.jsonl`,
  "Low-fidelity only") already covers part of the grid to minimize new compute.
- Cost: cheap (proxy is forward-only, no per-PEFT backward like LESS).

## Reruns needed

Track A: none. Track B: yes (low-fid proxy grid only).

## Acceptance criteria

- RDS+ and LESS-style are precisely defined and honestly labeled.
- Main table includes a low-fidelity-proxy row across all cells.
- Gap-to-ceiling caveat present; no unmeasured oracle claim.

## Implementation status (2026-07-06)

### Track A — DONE (no reruns)

- Precise, honestly-labeled definitions written:
  - `paper/Sections/05_experiments.tex` setup paragraph — RDS+ (standardized
    joint-embedding cosine to sketch mean, task-conditioned, PEFT-agnostic, no
    tuned knobs) and LESS-style (site-weighted `u^lo`, PEFT-specific via
    `α̃_p^ω`, shared datastore + shared sketch), plus the low-fid-proxy control.
  - `paper/Sections/08_appendix.tex` new subsec `\label{sec:appendix-baselines}`
    — full spec + the four explicit deltas of LESS-style from original LESS, and
    the honest note that E2 still charges per-PEFT recompute as a modeling choice.
- Gap-to-ceiling caveat + no-oracle sentence added to Main Results (decision 4).

### Track B — CODE DONE, grid run PENDING (needs GPU)

Exact write scope taken (authorized):

- `src/pcu_select/baselines/selectors.py` — new `lo_proxy_quota` selector
  (`_lo_proxy_quota` / `_lo_proxy_u`), registered in `BASELINES`, dense score in
  `score_baseline`. Signal = same `u^lo` as `less`; selection = PCU cluster quota
  with `σ≡0`, `α=0.6`. So PCU−proxy isolates the learned scorer.
- `tests/test_selection.py` — 3 unit tests (proxy≡less signal, budget/uniqueness,
  high-utility bias). All selection tests pass locally.
- `scripts/experiments/run_e1.py` — `lo_proxy_quota` added to the E1 `METHODS`
  grid. Minimal-compute append (no full rerun):
  `run_e1.py --methods lo_proxy_quota --budgets 0.10 --seeds 0 1 2` over the 4
  tasks × 5 seen PEFTs.
- `scripts/paper/generate_paper_assets.py` — `METHOD_LABEL["lo_proxy_quota"]`
  = "Low-fid proxy"; main + per-task tables include the row **conditionally**
  (only when present in `result/data/E1.jsonl`), so the paper compiles now and
  the row appears automatically after the run.

Not done here (no GPU in this env; no unsupported numbers were written):

- The 60-cell target-training grid (4 tasks × 5 PEFT × 3 seeds, 10% budget) that
  produces the real `metric` values. `--selection-only` also blocked locally: the
  offline artifacts (task sketches, feature cache, `z_t`/`task_grad`, 300K pool,
  `hi_fidelity.parquet`) are not in the working tree — they live in the run env.
- No result rows were written; no numbers were invented. The
  `% TODO(VERIFY)` in `05_experiments.tex` marks the proxy-vs-PCU prose to fill
  once `E1.jsonl` has the rows.

### Reproduce Track B numbers (in the GPU run env)

```
python scripts/experiments/run_e1.py \
  --workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval \
  --model llama2-7b --methods lo_proxy_quota --budgets 0.10 --seeds 0 1 2 \
  --task-metric-factory <module:factory>
# then refresh assets:
python scripts/paper/generate_paper_assets.py
```
Check `result/data/E3.jsonl` `pcu_lo_only` (2 tasks × 2 PEFT) first — that is the
learned lo-only scorer ablation, *not* the raw proxy, so it does not substitute
for this control; it only bounds expected compute.
