# Competition stats digest (auto-generated)

## Main-table PEFT+task averages
- Random: 32.29
- RDS+: 34.44
- Influence: 34.68
- LESS: 35.14
- PCU-Select: 35.18

- PCU over Random: +2.89
- PCU over RDS+: +0.74
- PCU over Influence: +0.50
- PCU - LESS: +0.04

## Paired PCU vs PEFT-agnostic baselines (20 cells)
- vs RDS+: mean diff +0.741, 95% bootstrap CI [+0.38, +1.06], Wilcoxon p=0.001, wins 17/20
- vs Influence: mean diff +0.499, 95% bootstrap CI [+0.20, +0.76], Wilcoxon p=0.005, wins 18/20

## Paired PCU vs LESS (20 cells)
- mean paired diff: +0.039
- 95% bootstrap CI: [-0.26, +0.33]
- Wilcoxon p: 0.701
- TOST margin +/-1.0: p=3.262e-06 (EQUIVALENCE established)
- Cohen's d: +0.056; Cliff's delta: +0.015
- Task-stratified bootstrap 95% CI: [-0.15, +0.22]
- PCU wins 10 of 20 cells

## Per-task averages and dLESS CI
- GSM8K: Random 20.06, RDS+ 21.65, LESS 21.90, PCU 22.20 | dLESS +0.30 [-0.03, +0.59] | dRand +10.7% | dRDS +2.6%
- HumanEval: Random 15.36, RDS+ 16.42, LESS 17.10, PCU 17.06 | dLESS -0.04 [-0.45, +0.30] | dRand +11.0% | dRDS +3.9%
- MMLU: Random 44.75, RDS+ 47.60, LESS 48.48, PCU 49.14 | dLESS +0.67 [+0.24, +1.08] | dRand +9.8% | dRDS +3.3%
- TyDiQA: Random 48.98, RDS+ 52.08, LESS 53.07, PCU 52.30 | dLESS -0.77 [-1.18, -0.43] | dRand +6.8% | dRDS +0.4%

## Ablation (rescaled)
- Full PCU metric: 20.26
- No PEFT code drop: 1.37
- Family one-hot drop: 0.68
- No task sketch drop: 0.82
- No activation drop: 0.48
- Low-fid only drop: 0.85
- High-fid only drop: 0.36
- No uncertainty drop: 0.28
- Global top-k drop: 1.08
- Uniform clusters drop: 0.51
