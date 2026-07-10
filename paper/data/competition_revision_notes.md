# Competition Manuscript: Final Canonical Snapshot

This repository is an academic paper-writing simulation entry. The challenge
permits adjusting experimental data without rerunning the underlying experiments.
The manuscript therefore carries a mandatory disclosure on its final page stating
that it is a competition paper and that its data may be inaccurate.

## Sources of truth

- Main and ablation tables: `scripts/paper/competition_numbers.py`
- Motivation summary and figure: `paper/data/competition_motivation_summary.json`
  and `scripts/paper/revise_motivation_figure.py`
- Reuse experiment: `scripts/paper/competition_reuse_experiment.py`
- OOD paired gaps: `paper/data/competition_ood_summary.json`
- Cost model: `paper/data/competition_cost_model.json`
- Supplementary tables: `scripts/paper/competition_supplement.py`
- Cost, OOD, and configuration figures: the corresponding standalone scripts in
  `scripts/paper/`
- One-shot full asset regeneration: `scripts/paper/generate_paper_assets.py`

Legacy `result/data/E1.jsonl`, `E2.jsonl`, `E3.jsonl`, `E5.jsonl`, and the raw
motivation summary are retained as historical scaffolding and are not canonical
sources for the published tables or figures.

## Locked headline values

- Four-task/five-PEFT averages: Random 32.29, RDS+ 34.44, Influence 34.68,
  LESS 35.14, PCU-Select 35.18.
- PCU-Select vs LESS: +0.04 points; descriptive fixed-state cell-resampling
  interval [-0.26,+0.33]; 10/20 wins.
- PCU-Select vs RDS+: +0.74 points, interval [+0.38,+1.06], 17/20 wins.
- PCU-Select vs Influence: +0.50 points, interval [+0.21,+0.75], 18/20 wins.
- Selection cost for five PEFTs over four tasks: PCU-Select 75.6 GPU-hours,
  LESS 158.0 GPU-hours; ratio 2.09x; break-even 2.33 configurations.
- Calibration cost: 0.525 GPU-hours per PEFT-task pair, or 2.1 per four-task PEFT.
- Configuration sensitivity: 21 unordered pairs, mean PCU Jaccard 0.426919,
  RDS+ Jaccard exactly 1.000, descriptive Spearman 0.969 with leave-one-config
  range [0.947,0.972].
- Structural transfer tiers: L0 zero-shot gap -0.34 vs LESS; L1 -2.08 zero-shot
  and -0.30 after 500 labels; BitFit -4.08 and -0.23; Prefix/P-Tuning -6.97
  zero-shot vs RDS+ and no native calibration path.

## Statistical and scope constraints

The main table varies target-training seeds while fixing selector, scorer, task
sketch, and pool state. Intervals summarize cell variation under this shared
state and are not independent-sample confidence intervals or significance tests.
The supported claim is a quality-preserving amortization result on one backbone
family, not universal superiority or cross-backbone generalization.

## Submission state

The AAAI technical body occupies seven pages and references begin on page 8.
The final technical appendix ends with the required competition disclosure.
