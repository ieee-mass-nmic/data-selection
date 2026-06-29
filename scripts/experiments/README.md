# Experiment harness (E1–E5)

Runner + figure scripts for [docs/pcu_select_experiment_design.md](../../docs/pcu_select_experiment_design.md).
Reusable logic lives in the package (`pcu_select.experiments`, `pcu_select.baselines`,
`pcu_select.eval`); these scripts are thin orchestration.

```
scripts/experiments/   run_e1..run_e5, _common.py, dump_peft_registry.py
scripts/plots/         plot_e1..plot_e5, _common.py
src/pcu_select/experiments/   registry.py (PEFT/task/model matrix), results.py
src/pcu_select/baselines/     selectors.py (Random … RDS+ … LESS)
src/pcu_select/eval/          target_train.py (PEFT fine-tune + eval), metrics.py
```

## Prerequisites (offline run, once per backbone family)

E1–E5 assume a populated `--workdir`: feature cache, per-task `z_t`/`task_grad`
artifacts, and a trained scorer. Build it with the existing pipeline scripts:

```bash
WD=runs/exp1
python scripts/build_features.py     --pool data/pool_300k.jsonl --workdir $WD --selector meta-llama/Llama-2-7b-hf
for T in gsm8k humaneval mmlu tydiqa; do
  python scripts/encode_task.py --task-jsonl data/${T}_train.jsonl --task-name $T --workdir $WD
done
python scripts/compute_lo_fidelity.py --workdir $WD --peft-configs configs/peft/registry/seen/*.yaml \
                                      --task-sketches $WD/task/sketches/*.json
python scripts/build_hi_fidelity.py --workdir $WD --pool data/pool_300k.jsonl --model llama2-7b \
                                    --tasks gsm8k humaneval mmlu tydiqa --q-hi-total 10000
python scripts/train_scorer.py --workdir $WD
python scripts/experiments/dump_peft_registry.py --model llama2-7b   # materialize registry yamls
```

Held-out eval sets go in `--eval-dir/<task>.jsonl` (separate from the sketches).

## Motivation experiments (Table 1, Figure 1, Figure 2)

These **precede** E1–E5 and establish the project's premise — *data value is
PEFT-dependent* — using signals **independent of PCU's scorer** (real
short-update Δ truth + LESS influence), so the argument is not circular. See
[docs/pcu_select_motivation_design.md](../../docs/pcu_select_motivation_design.md).

```bash
WD=runs/exp1

# Table 1 — PEFT configs + trainable params (zero-GPU; --from-model for exact counts)
python scripts/experiments/build_table1.py --model llama2-7b --out-dir tables

# Figure 1 — per-PEFT data-value vectors on a small estimation pool.
#   u_hi = short-update truth (needs backbone); 2 seeds → noise floor (§3.3).
#   u_grad/u_rds are cache-only (no GPU). Add --warm-anchor for a 2nd anchor.
python scripts/experiments/build_motivation_values.py --workdir $WD \
    --pool data/pool_300k.jsonl --model llama2-7b --tasks gsm8k humaneval \
    --n-val 2000 --seeds 0 1 --signals u_hi u_grad u_rds
python scripts/plots/plot_motivation_f1.py --values $WD/motivation/values.parquet \
    --signal u_hi --out-dir figs        # left=Spearman, right=Top-5% overlap, +noise floor

# Figure 2 — cross-PEFT transfer matrix (diagonal dominance). Source selection is
# non-PCU (LESS by default; --select-signal u_hi reads values.parquet).
python scripts/experiments/run_motivation_transfer.py --workdir $WD \
    --pool data/pool_20k.jsonl --eval-dir data/eval --tasks gsm8k humaneval \
    --budgets 0.10 --seeds 0 1 2 --select-signal less
python scripts/plots/plot_motivation_f2.py --results $WD/results/MOT_F2.jsonl --out-dir figs
```

## Running the experiments

```bash
COMMON="--workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval --model llama2-7b"

python scripts/experiments/run_e1.py $COMMON --budgets 0.05 0.10 0.30 --seeds 0 1 2
python scripts/experiments/run_e2.py $COMMON --per-peft-recompute-h 1.5
python scripts/experiments/run_e3.py $COMMON --tasks gsm8k humaneval --pefts L-r16-qkvo AD-b64 \
    --scorer-variants no_zp=$WD/scorer/ckpt_no_zp.pt lo_only=$WD/scorer/ckpt_lo_only.pt
python scripts/experiments/run_e4.py $COMMON --tasks gsm8k humaneval --seeds 0 1 2
# E5: first build the small calibration label set (high-fidelity, native families
# only), then run across tasks/seeds. prefix/ptuning (L2) are zero-shot only — see Notes.
python scripts/experiments/build_calib_labels.py --workdir runs/exp1 --pool data/pool_300k.jsonl \
    --model llama2-7b --tasks gsm8k --pefts L-r64-all AD-b256 L-r8-highlayers BF --n-calib 500
python scripts/experiments/run_e5.py $COMMON --tasks gsm8k humaneval mmlu --seeds 0 1 2 \
    --calib-labels runs/exp1/labels/calib.parquet
```

Each writes flat result rows to `runs/exp1/results/E*.jsonl` (+ a couple of
JSON side-files for E2/E4). Add `--selection-only` to exercise the full matrix
**without** loading a 7B backbone (records selection + ranking metrics only) —
handy for smoke-testing the plumbing before committing GPU time.

## Figures

```bash
pip install -e ".[viz]"
python scripts/plots/plot_e1.py --results runs/exp1/results/E1.jsonl --out-dir figs
python scripts/plots/plot_e2.py --results runs/exp1/results/E2.jsonl --cost-model runs/exp1/results/E2_cost_model.json --out-dir figs
python scripts/plots/plot_e3.py --results runs/exp1/results/E3.jsonl --out-dir figs
python scripts/plots/plot_e4.py --results runs/exp1/results/E4.jsonl --overlap runs/exp1/results/E4_overlap.json --out-dir figs
python scripts/plots/plot_e5.py --results runs/exp1/results/E5.jsonl --out-dir figs
```

Figure ↔ design-doc mapping: T1/F1 (E1), F2/F3 (E2), T2/F4 (E3), F5/F6 (E4),
F7/F8 (E5). See design §9.

## Notes / integration points

- **Downstream metric.** Inline eval reports held-out response-LM loss
  (`metric = −eval_loss`). For real EM / pass@k / F1, pass a
  `task_metric(model, tokenizer) -> float` into `eval.target_train.train_and_eval`
  (e.g. an lm-eval-harness wrapper). Documented hook; no harness dependency.
- **Baseline approximations.** `ifd` (length-normalized loss) and `s2l`
  (loss-dispersion) are single-checkpoint proxies; the exact signals
  (instruction-free forward / loss trajectories) are noted in
  `baselines/selectors.py` and can be wired in.
- **Prefix / P-Tuning (E5 L2).** Selection works via z_p and target training
  uses the `peft` library, so *zero-shot* E5 runs fine. But the native
  short-update backend can't train prompt families, so **calibration labels
  cannot be generated for prefix/ptuning** (`build_calib_labels.py` skips them):
  those L2 targets are reported zero-shot only (design §6.5). BitFit (also L2)
  is native, so it calibrates normally.
- **Cost.** Stage GPU-hours come from `cost/accounting.jsonl`; E2's break-even
  charges influence baselines a per-PEFT recompute that PCU amortizes offline.
