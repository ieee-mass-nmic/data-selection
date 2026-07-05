# AAAI-27 Project Context

## Administrative

- Venue: AAAI-27 Main Technical Track
- Main TeX file: main.tex
- Submission stage: anonymous review
- Technical page budget: 7 pages, followed by references
- Citation file: aaai2027.bib
- Current manuscript status: full draft under active revision

## One-Sentence Identity

PCU-Select is a PEFT-conditioned data selector that predicts which instruction examples are useful for a target task under a target PEFT configuration, enabling reusable data-efficient fine-tuning across PEFT settings.

## Research Problem

The paper addresses cross-PEFT data-efficient fine-tuning. Existing selectors usually learn one task-conditioned value per example and reuse it across PEFT methods. This fails structurally because LoRA, adapters, (IA)$^3$, prefix tuning, and related methods expose different trainable sites, operators, and capacities.

## Core Insight

Data value should be represented on intervention sites that connect sample gradients, task sketches, and PEFT update locations. The key abstraction is PEFT-conditioned utility over intervention sites.

## Method

PCU-Select extracts sample representations and site gradient signatures, encodes task sketches, and encodes PEFT configurations by site mask, capacity, and recipe. It builds a low-fidelity utility by weighting sample-task gradient alignment with PEFT-specific site weights. It corrects that proxy with short-horizon high-fidelity labels and trains a conditional scorer with ranking, regression, proxy, and uncertainty losses. At application time it scores the pool, discounts uncertain examples, and allocates budget across semantic clusters.

## Contributions

1. Formulate cross-PEFT data-efficient fine-tuning and show that data-value rankings differ across PEFT configurations.
2. Introduce PCU-Select, a reusable PEFT-conditioned selector based on intervention-site signatures and multi-fidelity utility labels.
3. Evaluate PCU-Select across four tasks, five seen PEFTs, unseen configurations, ablations, and amortized cost.

## Claim-Evidence Map

| Claim ID | Manuscript claim | Required evidence | Current evidence | Status |
|----------|------------------|-------------------|------------------|--------|
| C1 | Data value is PEFT-dependent | Ranking disagreement and transfer mismatch | `result/data/motivation/values.parquet`, `result/data/MOT_F2.jsonl`, `result/tables/F1_structural_u_hi.csv` | verified |
| C2 | PCU-Select improves PEFT-agnostic selection and matches LESS on average | E1 downstream table and ranking metrics | `result/data/E1.jsonl`, `paper/tables/table_main_results.tex` | verified |
| C3 | Offline cost amortizes against per-PEFT influence selectors | E2 GPU-hour curve and cost model | `result/data/E2.jsonl`, `result/data/E2_cost_model.json` | verified |
| C4 | PEFT code, task sketch, multi-fidelity labels, and adaptive quotas matter | E3 ablation table | `result/data/E3.jsonl`, `paper/tables/table_ablations.tex` | verified |
| C5 | Unseen PEFTs need calibration as OOD distance grows | E5 level/mode results | `result/data/E5.jsonl`, `paper/Figures/fig_ood_calibration.pdf` | verified |

## Evaluation

### Datasets

- Candidate pool: 300K mixed instruction examples in the result bundle.
- Tasks: GSM8K, HumanEval, MMLU, TyDiQA.

### Baselines

- Random, Balanced-Random, Length, Loss, Perplexity, IFD, S2L, Embedding-NN, Diversity, RDS+, Influence/gradient similarity, LESS.

### Metrics

- Downstream task-native metrics scaled as percentages.
- Ranking metrics: Spearman, Kendall tau, NDCG@K, Top-K hit rate, pairwise accuracy.
- Cost metrics: selection GPU-hours, target-training GPU-hours, offline GPU-hours, break-even target count.

### Experimental controls

- Number of target-training seeds: 3.
- Main selection budget: 10%; supplementary budgets: 5% and 30%.
- Hyperparameter selection protocol: fixed registry and held-out validation PEFT/task pairs.
- Statistical tests: not currently included in the result bundle; claims use mean comparisons and calibrated scope.

## Headline Results

- At a 10% budget, PCU-Select averages 42.81, improving over random by 3.64 points and RDS+ by 0.98 points; it matches LESS within 0.08 points on average.
- PCU-Select breaks even with per-PEFT LESS/influence recomputation after 5.17 target PEFT configurations.
- Removing the PEFT code drops performance by 2.45 points; removing the task sketch drops 1.47 points.
- For L2 unseen-family targets, zero-shot PCU trails LESS by 5.76 points, while 500 calibration labels recover 5.53 points.

## Limitations

- Main E1--E5 results target one backbone-family registry.
- High-fidelity supervision uses short-horizon utility, not full-training marginal contribution.
- PCU-Select is not cheaper than RDS+ or random on raw cost.
- Far OOD PEFT families require calibration.

## Terminology

| Canonical term | Avoid these alternatives |
|----------------|--------------------------|
| PCU-Select | ours-only phrasing without name |
| PEFT-conditioned utility | universal data value |
| intervention sites | arbitrary hooks |
| task sketch | validation/test set |
| high-fidelity short-horizon label | ground-truth full-training value |

## Locked Decisions

- The official AAAI template must not be modified.
- No result or citation may be invented.
- Critical evidence must appear in the main paper rather than only in supplementary material.
- Anonymous-review mode must not include author names, affiliations, acknowledgments, or repository links.

## Open Questions

- Page count may exceed the 7-page target after full figure insertion.
- Statistical significance tests are not in the current result bundle.
