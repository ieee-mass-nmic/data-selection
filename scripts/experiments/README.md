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
# (+ high-fidelity labeling, then) :
python scripts/train_scorer.py --workdir $WD
python scripts/experiments/dump_peft_registry.py --model llama2-7b   # materialize registry yamls
```

Held-out eval sets go in `--eval-dir/<task>.jsonl` (separate from the sketches).

## Running the experiments

```bash
COMMON="--workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval --model llama2-7b"

python scripts/experiments/run_e1.py $COMMON --budgets 0.05 0.10 0.30 --seeds 0 1 2
python scripts/experiments/run_e2.py $COMMON --per-peft-recompute-h 1.5
python scripts/experiments/run_e3.py $COMMON --tasks gsm8k humaneval --pefts L-r16-qkvo AD-b64 \
    --scorer-variants no_zp=$WD/scorer/ckpt_no_zp.pt lo_only=$WD/scorer/ckpt_lo_only.pt
python scripts/experiments/run_e4.py $COMMON --task gsm8k
python scripts/experiments/run_e5.py $COMMON --task gsm8k --calib-labels runs/exp1/labels/calib.parquet
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
- **Prefix / P-Tuning (E5 L2).** Selection works via z_p; target training uses
  the `peft` library (native backend defers prompt families). Calibration needs
  a small pre-computed high-fidelity label set (`--calib-labels`).
- **Cost.** Stage GPU-hours come from `cost/accounting.jsonl`; E2's break-even
  charges influence baselines a per-PEFT recompute that PCU amortizes offline.
