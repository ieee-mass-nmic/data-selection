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
- vs RDS+: mean diff +0.740, descriptive bootstrap interval [+0.38, +1.06], wins 17/20
- vs Influence: mean diff +0.499, descriptive bootstrap interval [+0.21, +0.75], wins 18/20

## Paired PCU vs LESS (20 cells)
- mean paired diff: +0.037
- descriptive bootstrap interval: [-0.26, +0.33]
- task-stratified bootstrap interval: [-0.15, +0.22]
- note: these are descriptive cell-resampling summaries under fixed shared state, not independent-sample confidence intervals
- PCU wins 10 of 20 cells

## Per-task averages and dLESS CI
- GSM8K: Random 20.06, RDS+ 21.65, LESS 21.91, PCU 22.20 | dLESS +0.29 [-0.04, +0.58] | dRand +10.7% | dRDS +2.5%
- HumanEval: Random 15.36, RDS+ 16.42, LESS 17.10, PCU 17.06 | dLESS -0.04 [-0.44, +0.31] | dRand +11.1% | dRDS +3.9%
- MMLU: Random 44.75, RDS+ 47.60, LESS 48.48, PCU 49.14 | dLESS +0.67 [+0.24, +1.08] | dRand +9.8% | dRDS +3.3%
- TyDiQA: Random 48.98, RDS+ 52.08, LESS 53.07, PCU 52.30 | dLESS -0.77 [-1.18, -0.43] | dRand +6.8% | dRDS +0.4%

## Ablation (rescaled)
- Full PCU metric: 19.72
- No PEFT code drop: 1.37
- Family one-hot drop: 0.68
- No task sketch drop: 0.82
- No activation drop: 0.48
- Low-fid only drop: 0.85
- High-fid only drop: 0.36
- No uncertainty drop: 0.28
- Global top-k drop: 1.08
- Uniform clusters drop: 0.51
