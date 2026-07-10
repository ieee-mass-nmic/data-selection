# PCU-Select Experimental Design Document

> Companion documents: [Research Plan](pcu_select_research_plan.md) · [Detailed Implementation Version](pcu_select_design.md)
> This document only specifies **how the experiments should be conducted**: experimental matrix, comparisons, protocols, metrics, statistical conventions, and schedule. It does not cover concrete experimental code.
> The method itself (scorer, multi-fidelity utility, site mask, adaptive clustering selection, OOD calibration) follows the implementation-version document.

---

## 0. Experiment Overview

The core proposition this project needs to answer is: **under a fixed backbone family and a stable PEFT subspace, a task-conditioned PEFT-aware data utility scorer obtained through one offline training process can be reused across multiple PEFT configurations, and can be superior in both "performance" and "total cost across multiple PEFTs."**

The five experiments required by the user are mapped to the following research questions (RQs):

| Experiment | Name                                                               | RQ Answered                                                                           | Claim Type              | Main Evidence                                                           |
| ---------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------- |
| **E1**     | This method vs baselines under different PEFTs                     | Does it select more effective data for each PEFT?                                     | Performance-first       | Downstream task metrics + ranking metrics                               |
| **E2**     | Cross-PEFT transfer / amortized cost comparison                    | Is the total cost lower when serving multiple PEFTs?                                  | Cost-first              | GPU-hours curve + break-even                                            |
| **E3**     | Ablation                                                           | Does each key module truly contribute?                                                | Methodology             | Performance drop from removing each module                              |
| **E4**     | Comparison of different configurations within the same PEFT family | Does PEFT conditioning truly capture that "configuration changes sample value"?       | Methodological core     | Selection differences across configurations + mismatch performance drop |
| **E5**     | Performance on unseen PEFTs                                        | Does it generalize to unseen configurations / unseen families, including calibration? | Generalization boundary | ID/OOD stratified results                                               |

**Two iron rules throughout the document:**

1. **Compute-controlled**: All "performance" comparisons must be conducted under a **fully identical training + evaluation protocol for the target PEFT**: same backbone, same PEFT hyperparameter recipe, same epoch/step, same budget, and same seed set. The only variable is the **selected data subset**.
2. **Cost-accounted**: Every stage (features, low fidelity, high fidelity, scorer training, scoring, selection, target training, evaluation) writes to `cost/accounting.jsonl`, which is used for E2 and all "performance/cost" figures.

---

## 1. Common Experimental Configuration (Shared by All Experiments)

### 1.1 Models (All 7B or Above)

The main claim is restricted to a **fixed backbone family and different model sizes within the family** (consistent with implementation version §6.2).

| Role                                                | Main Experiment (Family A: Llama) | Robustness Replication (Family B: Qwen2.5) |
| --------------------------------------------------- | --------------------------------- | ------------------------------------------ |
| Selector model (for gradient signatures / features) | `Llama-2-7B`                      | `Qwen2.5-7B`                               |
| High-fidelity anchor + target fine-tuning backbone  | `Llama-2-7B`, `Llama-2-13B`       | `Qwen2.5-7B`, `Qwen2.5-14B`                |

Notes:

* **The main conclusion is drawn on Family A**; Family B is only used to show that the conclusion is not an accident of a single family. Cross-family zero-shot transfer is **not** performed and is left as future work.
* The selector uses the smallest available model in the family (7B), while target fine-tuning can scale up to 13B/14B, validating the design of "small model for selector, large model for target" (implementation version §6.1). This itself is one ablation axis in E3: selector size.
* Llama-2 is chosen instead of newer models to align with the public implementations of baselines such as LESS / RDS+ / IFD, reducing replication disputes; Qwen2.5 serves as a robustness supplement using a modern model.
* If compute is tight, E2/E4/E5 can be completed only on Family A; for E1/E3, at least one task should be replicated on Family B.

### 1.2 Candidate Data Pool (meta-pool / candidate pool)

* Scale: **N ≈ 300k**, mixed across multiple domains, ensuring the coexistence of semantic redundancy and long tails; otherwise, data selection is meaningless.
* Recommended composition (ratios can be adjusted but must be fixed and disclosed in the paper):

  * General instruction: Tulu-v2-mix / Open-Hermes subset
  * Mathematical reasoning: MetaMathQA / GSM8K-train CoT
  * Code: Magicoder-OSS-Instruct / Evol-Instruct-Code
  * Knowledge QA: FLAN-v2 subset
  * Multilingual: Aya / multilingual instruction subset
  * Safety alignment: safety preference / refusal subset
* **Prevent test set leakage**: perform n-gram + embedding near-duplicate removal between the candidate pool and the test split of all tasks.
* The offline meta-training **meta-pool** and the online application **candidate pool** use the same pool in the main experiment, but high-fidelity labels are sampled only from `Q_H=10k` triplets within it.

### 1.3 Task Set and Evaluation Protocol

Each task requires (a) drawing a validation sketch from `train/dev` (default 32 examples, implementation version §9.1), and (b) an independent `test` for final evaluation. The sketch and test must be strictly separated.

| Task                        | Capability             | Data         | Evaluation Metric          | Decoding                  |
| --------------------------- | ---------------------- | ------------ | -------------------------- | ------------------------- |
| **GSM8K**                   | Mathematical reasoning | GSM8K        | Exact Match (acc)          | greedy, 8-shot→0-shot CoT |
| **MATH** (subset)           | Hard reasoning         | MATH         | Accuracy                   | greedy CoT                |
| **HumanEval + MBPP**        | Code                   | -            | Pass@1, Pass@10            | temp=0.2, n=20            |
| **MMLU**                    | Knowledge              | -            | Accuracy                   | log-likelihood            |
| **TyDiQA-GoldP** (subset)   | Multilingual           | -            | F1 / EM                    | greedy                    |
| **AlpacaEval 2 / MT-Bench** | Instruction following  | -            | LC win-rate / GPT-judge    | official protocol         |
| **Safety** (held-out)       | Safety preservation    | refusal eval | refusal acc / over-refusal | -                         |

* **Main tasks** running through E1–E5: **GSM8K, HumanEval, MMLU, TyDiQA**, covering reasoning/code/knowledge/multilingual capabilities and avoiding single-task overfitting of conclusions.
* AlpacaEval/MT-Bench, MATH, and Safety are supplementary tasks and should appear at least once in E1.
* GPT-judge evaluations fix the judge model and prompt version, and report version numbers.

> **Implementation note**: The target-training path (`pcu_select.eval.target_train`) always records held-out response-LM loss as an auxiliary `eval_loss`, while the primary downstream `metric` comes from a task evaluator injected through `train_and_eval(..., task_metric=...)` or the runner-level `--task-metric-factory`. Full E1-E5 runs therefore use task-native metrics (EM / pass@k / F1 / accuracy / judge score) instead of substituting response-LM loss. See "Downstream metric" in [scripts/experiments/README](../scripts/experiments/README.md).

### 1.4 PEFT Configuration Space (Unified Registry Across All Experiments)

Divide PEFT configurations into three sets. All experiments reuse this registry. `★` indicates inclusion in the scorer offline training support distribution ("seen").

#### A. Training Support Set (SEEN, configurations on which the scorer is trained)

| ID              | family  | modules    | layers | rank/bottleneck | lr   | Notes           |
| --------------- | ------- | ---------- | ------ | --------------- | ---- | --------------- |
| `L-r8-qv` ★     | lora    | q,v        | all    | r=8, α=16       | 2e-4 | LoRA baseline   |
| `L-r16-qkvo` ★  | lora    | q,k,v,o    | all    | r=16, α=32      | 2e-4 | Wider attn      |
| `L-r8-mlp` ★    | lora    | up,down    | all    | r=8, α=16       | 2e-4 | MLP-only        |
| `IA3-attnmlp` ★ | ia3     | attn+ffn   | all    | -               | 5e-4 | IA3 standard    |
| `AD-b64` ★      | adapter | bottleneck | all    | b=64            | 3e-4 | Houlsby adapter |

#### B. Unseen Configurations Within the Same Family (UNSEEN-config, for E4 / E5a)

| ID                   | family  | modules    | layers   | rank/bottleneck | lr   | Difference from SEEN              |
| -------------------- | ------- | ---------- | -------- | --------------- | ---- | --------------------------------- |
| `L-r4-qv`            | lora    | q,v        | all      | r=4             | 2e-4 | Extremely small capacity          |
| `L-r32-qkvo`         | lora    | q,k,v,o    | all      | r=32            | 2e-4 | Extremely large capacity          |
| `L-r64-all`          | lora    | all-linear | all      | r=64            | 1e-4 | Capacity + placement both changed |
| `L-r8-lowlayers`     | lora    | q,v        | low 1/3  | r=8             | 2e-4 | Placement shift                   |
| `L-r8-highlayers`    | lora    | q,v        | high 1/3 | r=8             | 2e-4 | Placement shift                   |
| `L-r16-hlr`          | lora    | q,k,v,o    | all      | r=16            | 5e-4 | Recipe (lr) shift                 |
| `AD-b16` / `AD-b256` | adapter | bottleneck | all      | b=16 / 256      | 3e-4 | Capacity extremes                 |
| `IA3-attnonly`       | ia3     | attn       | all      | -               | 5e-4 | Placement shift                   |

#### C. Unseen Families (OOD-family, for E5b)

| ID        | family  | Description                  |
| --------- | ------- | ---------------------------- |
| `PRE-l16` | prefix  | Prefix tuning, prefix_len=16 |
| `PT-l32`  | ptuning | P-Tuning v2, len=32          |
| `BF`      | bitfit  | bias-only                    |

> Note: Prefix/P-Tuning/BitFit are not included in the main conclusion, and are used only for OOD generalization and failure cases (consistent with research plan §12.3).

### 1.5 Data Budget

* Default three levels: **budget B ∈ {5%, 10%, 30%}** of N.
* The main table uses **10%**; 5%/30% are used for budget sensitivity curves.
* Additional reference points: **100% (full pool)** as upper bound, **random@budget** as lower bound.

### 1.6 Three Layers of Evaluation Metrics

1. **Downstream performance** (final claim): metrics for each task in §1.3.
2. **Selection quality / ranking metrics** (mechanism evidence, compared against high-fidelity ground truth on held-out `(x,p,t)` triplets): Spearman ρ, Kendall τ, NDCG@K, Top-K hit rate, pairwise ranking acc.
3. **Cost**: offline GPU-h, application GPU-h for each target PEFT, total GPU-h, peak memory, persistent storage, target training savings, break-even T.

### 1.7 Statistical and Reproducibility Conventions

* **Seeds**: for each (method × PEFT × task × budget) configuration, run **3 target fine-tuning seeds**, report **mean ± std**; the main table uses paired significance tests (Wilcoxon signed-rank or paired t-test, paired across tasks/configurations).
* **Sketch variance**: for at least one task, redraw the sketch with 3 different sketch seeds and report sketch sensitivity (addressing the data leakage concern in research plan §15.6).
* **Fairness checklist** (fixed and disclosed for each PEFT): same backbone ckpt, same PEFT recipe (lr/warmup/scheduler/steps), same effective batch, same eval protocol, and the same number of training steps under the same budget (i.e., different subsets but identical training compute).
* **Compute-control variants**: baselines are reported in two categories: (i) *unconstrained* (each baseline uses its recommended configuration), and (ii) *compute-matched* (the wall-time of each method's "selection stage" is restricted to the same budget). The latter responds to the question of "why not simply use a cheap method."

---

## 2. E1 — This Method vs Baselines Under Different PEFTs

### 2.1 Objective

Verify that **for every PEFT configuration**, the subset selected by this method yields downstream performance ≥ various baselines after training. This is not only about having an advantage when reusing across PEFTs. It is the foundation of the "performance-first" claim.

### 2.2 Experimental Matrix

```
Methods (≈12) × SEEN PEFTs (5: L-r8-qv, L-r16-qkvo, L-r8-mlp, IA3-attnmlp, AD-b64)
          × Tasks (4 main tasks) × budget (10% main, 5%/30% supplementary) × seed (3)
          × backbone (Llama-2-7B main; 13B sampled replication)
```

The main table fixes budget=10% and Llama-2-7B, producing a large **method × PEFT × task** table, and reports cross-task averages for each PEFT.

### 2.3 Comparison Baselines (Grouped)

| Group                   | Baseline                                                           | Notes                              |
| ----------------------- | ------------------------------------------------------------------ | ---------------------------------- |
| Lower/upper bounds      | Random, Balanced-Random, Full-pool(100%)                           | Anchor the range                   |
| Heuristic               | Length, Loss(high), Perplexity, IFD                                | No training dynamics required      |
| Representation-based    | Embedding-NN-to-sketch, RDS+, Diversity-only clustering            | Only semantic similarity/diversity |
| Training-dynamics-based | **LESS**, Influence/gradient-similarity, S2L                       | PEFT-specific, strongest opponents |
| **This method**         | PCU-Select (adaptive cluster + uncertainty, default configuration) | -                                  |

Key points:

* **LESS / influence baselines are recomputed per PEFT**: gradient features must be recomputed for each PEFT configuration. In E1, they are allowed to "recompute to optimal" (unconstrained), serving as the strongest performance opponents; their recomputation cost is settled in **E2**.
* RDS+/Embedding-NN queries use the same validation sketch, ensuring fairness in task-conditioned inputs.

### 2.4 Success Criteria

* Claim A (strong): this method is ≥ the strongest training-dynamics baseline (LESS) on ≥ 4/5 PEFTs × most tasks, and is significantly better than all representation-based/heuristic baselines.
* Claim B (weak fallback): if it is slightly worse than per-PEFT LESS on some PEFTs, the conclusion becomes "**comparable or close performance, but significantly lower total cost across multiple PEFTs**" (supported by E2).
* Mechanism evidence: on held-out triplets, this method's NDCG@K / Spearman is higher than representation-based baselines.

### 2.5 Failure Analysis Hooks

* If it is outperformed by a pure embedding baseline → check whether the scorer has degenerated into semantic similarity (see E3 ablations removing site-mask / activation).
* If high-fidelity labels are noisy → inspect the multi-fidelity ablation in E3 and the seed/sketch variance in §1.7.

---

## 3. E2 — Transfer / Amortized Cost Comparison Across Multiple PEFTs

### 3.1 Objective

Prove the core economic claim of the method: **offline cost can be amortized across multiple target PEFTs**, and the more PEFTs served, the more cost-saving this method becomes relative to per-PEFT methods. This is the key distinction from LESS/influence-style methods.

### 3.2 Cost Model (Aligned with Implementation Version §16)

* Total cost of this method: `C_offline + T · C_apply`

  * `C_offline = C_feat + C_lo + C_hi + C_scorer` (one-time)
  * `C_apply = C_feat-new(cacheable→≈0) + C_score + C_select + C_target-train`
* Total cost of per-PEFT baselines: `T · C_specific`

  * For LESS: `C_specific = C_grad-feature(per PEFT) + C_select + C_target-train`
  * For RDS+/PPL: `C_specific` is smaller (forward only), making them "cheap but weak" opponents

### 3.3 Experimental Setup

```
T ∈ {1, 3, 5, 10} target PEFTs (sampled from the SEEN ∪ UNSEEN-config registry, covering family/capacity/placement)
For each method: measure GPU-hours for every stage in practice (no estimation), plug into the cost model, and draw two types of curves:
  (a) Total GPU-h vs T            — find intersection point = break-even T*
  (b) Performance (mean across PEFTs) vs total cost  — Pareto frontier
```

### 3.4 Compared Methods

* PCU-Select (including one-time offline cost)
* LESS / influence (recompute gradient features for each PEFT)
* RDS+ / PPL / IFD (recompute forward for each PEFT, cheap)
* Random (zero selection-cost baseline)

### 3.5 Required Outputs

1. **Cost decomposition stacked bar chart**: split `C_feat / C_lo / C_hi / C_scorer / C_apply` for each method.
2. **Break-even curve**: `T* = C_offline / (C_specific - C_apply)`, marking the intersection with LESS and with RDS+ separately.
3. **Pareto plot**: x-axis total GPU-h, y-axis average downstream performance across PEFTs; this method should dominate in the upper-right multi-PEFT region.
4. Comparison table for peak memory / persistent storage / target-train savings.

### 3.6 Success Criteria

* There exists a reasonable `T*` (for example, `T* ≤ 5`) such that when `T > T*`, this method has the lowest total cost.
* If `T*` is too large (offline cost too expensive) → fallback: reduce `Q_H`, reduce anchor count, reduce horizon; report sensitivity of `T*` to these hyperparameters in the paper, echoing research plan §12.3.

---

## 4. E3 — Ablation Experiments

### 4.1 Objective

Remove key modules one by one to prove that each component contributes independently, responding to the concern that "there are too many modules / this is just LESS + a PEFT vector" (research plan §15.1/§15.7).

### 4.2 Ablation Axes (Change Only One at a Time; Others Remain Default)

| # | Ablation Axis                       | Variant                                                                                     | What It Validates                              |
| - | ----------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| A | **PEFT condition**                  | Remove `z_p` (family one-hot / remove all)                                                  | Necessity of PEFT-conditioning (most critical) |
| B | **Task condition**                  | Remove `z_t`; sketch size ∈ {0,8,16,32,64}                                                  | Necessity of task sketch + size turning point  |
| C | **Multi-fidelity**                  | lo-only / hi-only / lo+hi; hi budget ∈ {2k,5k,10k}                                          | Multi-fidelity is not redundant complexity     |
| D | **Sample representation**           | `e_x` only / +`d_x` / +`a_x` (activation signature)                                         | Site interaction requires activation signature |
| E | **PEFT representation granularity** | family one-hot → +site mask → +capacity → +recipe → +fingerprint                            | Incremental gains from structured encoding     |
| F | **Uncertainty**                     | Remove `σ̂` risk penalty (λ_unc=0, λ=0)                                                     | Deployment value of uncertainty                |
| G | **Selection strategy**              | global top-k / uniform-cluster / adaptive-cluster / (DPP optional); α ∈ {0,0.3,0.6,0.9,1.0} | Diversity constraint + α turning point         |
| H | **Training objective**              | Remove rank / remove reg / remove proxy distillation (set each λ=0)                         | Contribution of each loss term                 |
| I | **Selector size**                   | 7B selector → 13B selector (same family)                                                    | Hypothesis that "small selector is sufficient" |
| J | **Temporal pooling / horizon**      | mean vs last-token; horizon {1} / {4} / {1,4}                                               | Low-/high-fidelity details                     |

### 4.3 Setup

* Main ablations are performed on **GSM8K + HumanEval, two tasks; two representative SEEN PEFTs (`L-r16-qkvo`, `AD-b64`); budget=10%**, controlling experimental volume.
* Each variant reports downstream performance (main) + held-out ranking metrics (NDCG@K, mechanism).
* **Critical ablations A and E** must be performed on all 4 main tasks; these are direct evidence for the core contribution.

### 4.4 Success Criteria (Aligned with Minimum Validation in Research Plan §20)

* Removing `z_p`: NDCG@K drops by ≥ 5%, and downstream performance visibly declines.
* lo-only vs lo+hi: lo+hi outperforms either single fidelity in both ranking and downstream performance; otherwise, downgrade hi to an optional component and state this explicitly.
* adaptive-cluster > global top-k (diversity benefit), and there exists an optimal α.
* In E, each component (site mask / capacity / recipe) yields monotonic or near-monotonic gains; if fingerprint brings no gain, downgrade it to an optional module.

---

## 5. E4 — Performance Comparison Across Different Configurations Within the Same PEFT

### 5.1 Objective (Methodological Core of This Project)

Prove the proposition that "**the same sample has different value under different configurations within the same family, and this method can capture that difference**" (research plan premises 1 and 2). This is the direct validation of the entire PEFT-conditioning argument. If the optimal subset is identical across different configurations, then PEFT conditioning is unnecessary.

### 5.2 Configuration Scan (Fixed LoRA Family, Varying One Axis at a Time)

| Axis                       | Values                 | From Registry                            |
| -------------------------- | ---------------------- | ---------------------------------------- |
| rank (capacity)            | 4 / 8 / 16 / 32 / 64   | `L-r4-qv` … `L-r64-all`                  |
| target modules (placement) | qv / qkvo / all-linear | `L-r8-qv` / `L-r16-qkvo` / `L-r64-all`   |
| layer range (placement)    | low / mid / high / all | `L-r8-lowlayers` / `-highlayers` / `-qv` |
| lr (recipe)                | 1e-4 / 2e-4 / 5e-4     | `L-r16-qkvo` / `L-r16-hlr`               |

> Some configurations are in SEEN and others in UNSEEN-config; E4 measures configuration sensitivity and transfer to held-out same-family configurations.

### 5.3 Three Sub-experiments

**E4-a Selection difference (mechanism)**: For every pair of configurations `(p_i, p_j)`, compute **Jaccard overlap** and Top-K Spearman between the subsets selected by this method on the same candidate pool. Expected result: the larger the configuration difference (e.g., rank4 vs rank64, low vs high layers), the lower the overlap. Provide a correlation plot of "configuration difference → selection difference." As a comparison, report RDS+/PPL, which are **PEFT-agnostic** and therefore have overlap constantly equal to 1, highlighting the condition sensitivity of this method.

**E4-b Per-configuration performance (performance)**: For each configuration, compare this method vs Random vs RDS+ vs LESS on downstream performance. The expected result is that this method leads on every configuration, and its advantage grows as the configuration becomes more "deviated from the conventional setting."

**E4-c Mismatch / cross-transfer (decisive evidence)**: Construct a **mismatch matrix**: use the subset selected for configuration `p_i` to train configuration `p_j` (`i≠j`), and compare it with the subset "correctly conditioned for `p_j`."

* Expected: diagonal entries (correct conditioning) > off-diagonal entries (mismatch), and the performance drop quantifies the value of PEFT-conditioning.
* Also provide the same matrix for the PEFT-agnostic baseline (RDS+) as a comparison; it should show no difference between diagonal and off-diagonal entries, further emphasizing the contrast.

### 5.4 Success Criteria

* E4-a: configuration difference and selection difference are significantly positively correlated (Spearman > 0.5).
* E4-c: mismatch performance drop is significant (paired test p<0.05), and PCU's "correct-conditioning gain" is greater than the random fluctuation of PEFT-agnostic baselines.
* If selection is almost unchanged across configurations (E4-a overlap≈1) → the scorer has not truly used `z_p`; return to E3-A/E3-E for diagnosis.

---

## 6. E5 — Performance on Unseen PEFTs

### 6.1 Objective

Characterize generalization boundaries and validate the effectiveness of OOD calibration mode (implementation version §13). It is split into two layers and **must never be mixed together** (research plan §15.5).

### 6.2 Stratified Setup

| Level                   | Test Configuration                                                                                               | Has the scorer seen it?                           | Mode                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------- |
| **L0 near support** | Same-family configurations with one comparatively small structural shift (`L-r32-qkvo`, `AD-b16`) | Family seen, concrete configuration unseen | Direct zero-shot scoring |
| **L1 far support** | Extreme or compound same-family shifts (`L-r64-all`, `AD-b256`, `L-r8-highlayers`) | Family seen, configuration far from support | zero-shot + calibration comparison |
| **L2 unseen family** | `PRE-l16`, `PT-l32`, `BF` (prefix/ptuning/bitfit) | Family completely unseen | Calibration only where native labels exist |

> **Implementation boundary (consistent with implementation version §13.2)**: the high-fidelity short-horizon updates required for calibration only support native families (lora / ia3 / adapter / **bitfit**). `PRE-l16`(prefix) and `PT-l32`(ptuning) cannot generate calibration labels, so they can only be evaluated zero-shot and should be truthfully reported as failure boundaries (§6.5); `BF`(bitfit), also in L2, can be calibrated normally. Calibration labels are generated by `scripts/experiments/build_calib_labels.py`.

### 6.3 Calibration Protocol (L1/L2)

Following implementation version §13.2:

1. Assign the pre-registered structural stratum before evaluation. Report regularized Mahalanobis `d²(p*)` only as a diagnostic; five support configurations do not justify interpreting it as a density estimate or using it to define the strata.
2. After triggering: sample **200 / 500** examples, compute small-scale high-fidelity for `p*` (horizon=1, single anchor, ≈1/4 cost), freeze the scorer body, and train only the calibration head.
3. Compare three modes: **zero-shot direct scoring** / **calibration(200)** / **calibration(500)** / per-PEFT LESS (upper-bound reference).

### 6.4 Experimental Matrix

```
Test configurations (L0:2, L1:3, L2:3) × Tasks (GSM8K, HumanEval, MMLU)
× Modes (zero-shot / cal-200 / cal-500) × seed(3)
Comparisons: Random(lower bound), RDS+(PEFT-agnostic), per-PEFT LESS(upper bound)
```

### 6.5 Success Criteria / Expected Pattern

* **L0**: zero-shot is already close to per-PEFT LESS and clearly better than Random/RDS+ → supports transfer to nearby same-family configurations.
* **L1**: zero-shot degrades, while cal-500 largely recovers the loss → proves that calibration mode is effective and cheap.
* **L2**: zero-shot may fail (**allowed and truthfully reported as a failure case**), and cal-500 should be significantly better than Random; if it still fails, explicitly write this into the "applicability boundary" (research plan §16.2).
* Optional **`d²` vs performance degradation scatter plot**: report regularized Mahalanobis distance only as a diagnostic. Deployment decisions use the preregistered structural tier rather than a fitted density threshold.

---

## 7. Unified Implementation / Fairness Notes (Must Be Fixed and Disclosed in Writing)

1. **Target fine-tuning protocol**: all methods use the same PEFT recipe and the same number of training steps; changing the budget changes the "data subset," not the compute. When necessary, truncate by steps to ensure compute-matched comparison.
2. **Baseline adaptation**: LESS / RDS+ / IFD / S2L use official implementations or carefully verified replications; the required signals are recomputed for every PEFT, which is exactly the cost E2 accounts for.
3. **Sketch as query**: representation-based baselines (RDS+/Embedding-NN) and this method share the same validation sketch as the task query, avoiding unequal task information.
4. **Leakage prevention**: sketch ⟂ test; candidate pool ⟂ test. Report deduplication statistics.
5. **Randomness**: fix global_seed; run 3 target fine-tuning seeds; run 3 sketch-seeds for at least one task.
6. **Accounting**: all stages write to `cost/accounting.jsonl` (fields from implementation version §16.1); E2 reads it directly, and post-hoc estimation is not allowed.

---

## 8. Compute Resources and Schedule Estimate (Rough, for Planning)

> Order-of-magnitude estimate, based on 8×A100-80G; actual values follow `accounting.jsonl`.

| Stage                         | Main Cost Source                                       | Scale                        | Notes                           |
| ----------------------------- | ------------------------------------------------------ | ---------------------------- | ------------------------------- |
| Feature extraction (C_feat)   | One forward pass over N=300k + activation signatures   | Low                          | One-time, cacheable             |
| Low fidelity (C_lo)           | N×forward+backward site signatures                     | Medium                       | One-time, reusable across PEFTs |
| High fidelity (C_hi)          | Q_H=10k × A=2 × horizon{1,4} short updates             | **Medium-high**              | Main offline cost, focus of E2  |
| Scorer training               | Small network with 1.5M parameters                     | Extremely low                | -                               |
| Application per target PEFT   | Scoring(≈0) + selection + target fine-tuning(10% data) | Dominated by target training | Linear in T                     |
| Target fine-tuning evaluation | Inference for each task                                | Medium                       | Pass@k/decoding dominates       |

Scheduling recommendation:

* **First run the minimum closed loop** (research plan §20 / implementation version §20, three experiments) to confirm `u^lo↔u^hi` correlation, effectiveness of `z_p`, and superiority over baselines under equal compute; then scale up E1–E5.
* Priority: **E1 → E4 → E3 → E2 → E5** (first establish performance and mechanism, then cost, and finally generalization boundaries). Family B replication should be placed last.

---

## 9. Result Output Checklist (Paper Figure/Table Templates)

| ID | Form                      | Content                                                                    |
| -- | ------------------------- | -------------------------------------------------------------------------- |
| T1 | Large table               | E1: method × PEFT × task (budget=10%, 7B), including mean and significance |
| F1 | Line chart                | E1: budget(5/10/30%) sensitivity                                           |
| F2 | Line chart + intersection | E2: total GPU-h vs T, break-even                                           |
| F3 | Scatter plot              | E2: performance vs total cost Pareto                                       |
| T2 | Ablation table            | E3: performance drop per axis (performance + NDCG@K)                       |
| F4 | Line chart                | E3-B/E3-G: sketch size / α turning point                                   |
| F5 | Heatmap                   | E4-c: mismatch matrix (diagonal vs off-diagonal)                           |
| F6 | Scatter plot              | E4-a: configuration difference vs selection difference                     |
| F7 | Grouped bar chart         | E5: L0/L1/L2 × (zero-shot/cal-200/cal-500)                                 |
| F8 | Optional diagnostic scatter | E5: Mahalanobis d² vs performance degradation (not a tier definition)    |
| T3 | Table                     | Cost/memory/storage details + fairness checklist                           |

---

## 10. Risk and Fallback Matrix

| Risk                                          | Trigger Signal                                         | Fallback                                                                               |
| --------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| Losing to embedding baseline on a single PEFT | E1 main table falls behind                             | Shift claim to "cost-first" and rely on E2; meanwhile inspect E3-D/E                   |
| Offline cost cannot be sufficiently amortized | E2 has too large T*                                    | Reduce Q_H/anchor/horizon and report sensitivity                                       |
| No selection difference across configurations | E4-a overlap≈1                                         | Diagnose ineffective z_p (E3-A/E), may require stronger site/activation representation |
| OOD family failure                            | E5-L2 zero-shot collapses                              | Restrict claim to fixed-family; list L2 as failure case + calibration fallback         |
| High-fidelity label noise is large            | Large seed/sketch variance, weak u_lo↔u_hi correlation | Add anchors/seeds in high-uncertainty regions, verify with RankNorm                    |
| Evaluation leakage concern                    | Reviewers question the sketch                          | Publish deduplication and sketch protocol, report sketch-seed variance                 |

---

## Appendix: Correspondence with Research Plan §12

This document is a **consolidation and reorganization** of "Experimental Design" in research plan §12. The correspondence is:

* Research plan Experiment 1 (single-PEFT selection) → **E1** in this document
* Research plan Experiment 2 (cross-PEFT reuse) → **E5** (generalization) + **E4** (configuration sensitivity) in this document
* Research plan Experiment 3 (total cost across multiple PEFTs) → **E2** in this document
* Research plan Experiments 4/5/6/7 (multi-fidelity / conditional representation / task sketch / selection-strategy ablations) → axes C/E, B, and G of **E3** in this document
* New contributions: **E4 mismatch matrix** (decisive evidence for PEFT-conditioning), **E5 ID/OOD three-level decomposition + d² effectiveness**, and unified compute-matched and accounting conventions across all experiments.
