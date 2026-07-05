# PCU-Select Motivation Experiment Design Document

> Companion documents: [Research Plan](pcu_select_research_plan.md) · [Detailed Implementation Design](pcu_select_design.md) · [Main Experiment Design](experiment_design.md)
> This document only specifies **how the motivation of the research problem is experimentally validated**: before the method itself (scorer / multi-fidelity / adaptive selection) is introduced, we use cheap signals that are **independent of the proposed method** to prove that the premise of “PEFT-conditioned data selection” is valid.
> Three outputs: **Table 1** (PEFT methods modify different locations and have different parameter counts), **Figure 1** (inconsistency in data-value rankings), and **Figure 2** (diagonal dominance in the cross-PEFT transfer matrix).

---

## 0. What the Motivation Experiments Need to Answer

Research Plan §1.2 grounds the whole project on four premises. Among them, the **most fundamental premise, and also the one most easily dismissed by reviewers in one sentence**, is:

> **Premise 1: The same sample does not have the same value under different PEFT methods.**

If this premise does not hold—that is, if “which data should be selected” is almost identical across different PEFT methods—then the entire main line of PEFT-conditioning (`z_p`), cross-PEFT reuse, and the mismatch matrix (E4-c) loses its meaning, and the method degenerates into an ordinary task-only data selector. Therefore, the sole purpose of the motivation experiments is:

| Output       | Proposition to be argued                                                                                                                                    | Opposing hypothesis to be ruled out                            |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Table 1**  | PEFT methods are **structurally** different: they modify different layers/modules/operators/capacities → `z_p` has encodable structure                      | “PEFT methods differ only by name”                             |
| **Figure 1** | PEFT methods differ in **data value**: per-sample value rankings diverge significantly across PEFT methods, and the divergence is **above the noise floor** | “The rankings look different only because the signal is noisy” |
| **Figure 2** | This divergence has **downstream consequences**: data selected for PEFT *i* causes performance drops when used to train PEFT *j* (diagonal dominance)       | “Data selected using any PEFT is equally good”                 |

The three outputs form a progressive logical chain: **structural difference (T1) → value-ranking difference (F1) → training consequence difference (F2)**. Table 1 is descriptive and does not require training, while Figure 1/2 are empirical.

### 0.1 A Methodological Iron Rule Throughout the Document: No Circular Self-Validation

The motivation experiments are **strictly forbidden from using the proposed scorer `s_φ`** as the data-value signal. Otherwise, “our method thinks different PEFT methods have different values” becomes a tautology and is not persuasive.

Therefore, all data values in this document come from **PEFT-specific ground truth or widely accepted proxies that are independent of PCU**:

* **Main signal (ground truth) `u^hi`**: the validation loss reduction Δ caused by a real short-horizon PEFT update, namely the leave-one-in influence implemented by `ShortUpdater.delta(...)` in [short_update.py](../src/pcu_select/hi_fidelity/short_update.py). This is the physical definition of “whether training this sample is useful for this PEFT,” independent of any learned scorer.
* **Cross-validation signal `u^grad`**: LESS-style per-PEFT gradient similarity ([selectors.py](../src/pcu_select/baselines/selectors.py) `_less`), used to show that the conclusion of F1 does not depend on a single definition of value.
* **PEFT-agnostic control `u^rds`**: RDS+ (semantic similarity, independent of `z_p`). It gives the **same** ranking for all PEFT methods and serves as a “ceiling-like consistency” contrast in F1/F2.

> In one sentence: the motivation experiments use **data values that others would also recognize**, and prove that **this value itself changes with PEFT**—which is precisely the target quantity PCU aims to learn, not the output of PCU.

---

## 1. Common Setup Shared by the Motivation Experiments, Deliberately Small

The motivation experiments are positioned as a “pilot validation” in the spirit of Research Plan §20. They must be **an order of magnitude cheaper than the main experiments** and should quickly produce results before E1–E5 are fully launched. Therefore, based on the common main-experiment configuration ([Experiment Design §1](experiment_design.md)), the setup is narrowed as follows:

| Dimension      | Main experiments                   | Motivation experiments                                                        | Reason                                                              |
| -------------- | ---------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Backbone       | Llama-2-7B/13B + Qwen replication  | **Single Llama-2-7B**                                                         | Motivation does not require robustness replication                  |
| Candidate pool | N≈300k                             | **Valuation pool N_val≈2k** (F1) / selection pool N_sel≈20k (F2)              | Δ labeling cost grows linearly with sample count                    |
| Tasks          | 4 main tasks + supplementary tasks | **GSM8K + HumanEval** (one reasoning task and one code task, sufficient span) | Two tasks are enough to show non-trivial “task × PEFT” interactions |
| Data value     | Learned `s_φ`                      | **Ground-truth Δ + LESS influence, both non-PCU**                             | Iron rule in §0.1                                                   |
| Budget         | 5/10/30%                           | **Single 10% setting** (F2 selection)                                         | One setting is enough to show diagonal dominance                    |
| seed           | 3 target-training seeds            | **F1: 2 anchors + 2 seeds for the noise floor; F2: 3 target-training seeds**  | The noise floor is the key point of F1 (§3.3)                       |

### 1.1 PEFT Set for the Motivation Experiments, a Subset of the Unified Registry

From `PEFT_REGISTRY` in [registry.py](../src/pcu_select/experiments/registry.py), select **8 configurations**, deliberately spread across three structural axes so that “structural distance → value divergence” can be observed:

| ID                | family  | Modified location           | Operator                | Capacity | Axis represented in this set    |
| ----------------- | ------- | --------------------------- | ----------------------- | -------- | ------------------------------- |
| `L-r8-qv`         | lora    | attn: q,v / all layers      | additive low-rank       | r=8      | **Baseline**                    |
| `L-r8-mlp`        | lora    | mlp: up,down / all layers   | additive low-rank       | r=8      | placement: attn↔mlp             |
| `L-r4-qv`         | lora    | attn: q,v / all layers      | additive low-rank       | r=4      | capacity: small                 |
| `L-r32-qkvo`      | lora    | attn: q,k,v,o / all layers  | additive low-rank       | r=32     | capacity+placement: large       |
| `L-r8-lowlayers`  | lora    | attn: q,v / **low 1/3**     | additive low-rank       | r=8      | placement: layer segment        |
| `L-r8-highlayers` | lora    | attn: q,v / **high 1/3**    | additive low-rank       | r=8      | placement: layer segment        |
| `IA3-attnmlp`     | ia3     | k,v,down / all layers       | **multiplicative**      | —        | family/operator: multiplicative |
| `AD-b64`          | adapter | block residual / all layers | **additive bottleneck** | b=64     | family/operator: bottleneck     |

> Among these 8 configurations, 5 are `seen` and 3 are `unseen_config`, but the motivation experiments **do not distinguish seen/unseen**. That is a generalization claim and belongs to E5. Here, only their structural differences are used.

Design intention: the set contains both **cross-family** differences (lora/ia3/adapter, different operators) and **within-family** capacity axes (r4/r8/r32) and placement axes (qv/mlp/lowlayers/highlayers). In this way, Figure 1 can show not only that “LoRA and IA3 have different values,” which is easy, but also the **stronger proposition**: “even two configurations that are both LoRA and differ only in rank or layer segment already show significant divergence in data-value rankings.” This directly blocks the objection that “PEFT differences can be summarized away by a single hyperparameter.”

---

## 2. Table 1 — PEFT Configurations and Trainable Parameters

### 2.1 Purpose

Use a **purely descriptive** table to establish that “PEFT is not a name, but a set of structured and mutually different interventions” (Research Plan §1.2 Premise 2). This table provides direct evidence that `z_p` (site mask + capacity + recipe, [design §8](pcu_select_design.md)) indeed has structure to encode, and it also serves as a quantitative anchor for the term “structural distance” in Figure 1/2.

### 2.2 Column Definitions

For each of the 8 configurations in §1.1, one row is created with the following columns:

| Column                  | Meaning                                                 | Source                                                                        |
| ----------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `PEFT ID`               | Configuration name                                      | registry                                                                      |
| `Family`                | lora / ia3 / adapter                                    | registry                                                                      |
| `Inserted into`         | Which modules are modified (q/k/v/o/up/down/residual)   | `target_modules`                                                              |
| `Layers`                | all / low⅓ / high⅓                                      | `target_layers`                                                               |
| `Operator`              | additive-lowrank / multiplicative / additive-bottleneck | Mapping in [design §2.3](pcu_select_design.md)                                |
| `# Trainable`           | Number of trainable parameters, absolute value          | Formula in §2.3                                                               |
| `% of backbone`         | Proportion of full 7B parameters                        | `#Trainable / 6.7e9`                                                          |
| `Touched sites \|Ω_p\|` | Number of intervention sites touched, out of 24 sites   | `site_mask_of(p)` ([site_mask.py](../src/pcu_select/peft_space/site_mask.py)) |

The last two columns, site count and parameter count, are key for interpreting Figure 1: they quantify “why these two PEFT methods have very different value rankings” as “their modified site sets / capacities differ substantially.”

### 2.3 Parameter Count Calculation (`d=4096, L=32` for Llama-2-7B)

Do not fill manually. Instead, compute from the registry using a script to avoid drift from the configuration:

* **LoRA**: `#trainable = 2 · r · d · n_layers · n_modules`

  * `L-r8-qv`: `2·8·4096·32·2 ≈ 4.19M` (≈0.063%)
  * `L-r4-qv`: ≈2.10M; `L-r32-qkvo`: `2·32·4096·32·4 ≈ 33.6M` (≈0.50%)
  * `L-r8-lowlayers`/`highlayers`: number of layers is ⌈32/3⌉=11 → `2·8·4096·11·2 ≈ 1.44M`
  * `L-r8-mlp`: modules are changed to up/down. Note that the MLP dimension is `d_ffn≈11008` → `2·8·(4096+11008)·32 ≈ 7.73M`
* **IA3** (`IA3-attnmlp`, scaling output-dimension vectors of k/v/down): `#trainable = (d_k + d_v + d_ffn)·L ≈ (4096+4096+11008)·32 ≈ 0.61M` (≈0.009%)
* **Adapter** (`AD-b64`, Houlsby dual bottleneck): `#trainable ≈ 2 · (2·b·d) · L = 2·2·64·4096·32 ≈ 67.1M` (≈1.0%)

> The above numbers are **order-of-magnitude illustrations**. The official numbers should come from the script output. The key point is not the exact values, but that they **span two orders of magnitude** (IA3≈0.6M ↔ Adapter≈67M) and **modify largely non-overlapping locations**—this is the structural premise for “the same sample value changes with PEFT.”

### 2.4 Implementation

* Add `scripts/experiments/build_table1.py`: iterate over `resolve_peft(name, "llama2-7b")`, call `count_trainable_params(peft)` and `site_mask_of(peft)`, and output CSV + LaTeX.
* If `count_trainable_params` does not yet exist, add it to [peft_space/encoder.py](../src/pcu_select/peft_space/encoder.py). It already computes `log(trainable_params)` for `c_p`, so the internal counting logic can be reused.

---

## 3. Figure 1 — Inconsistency in Data-Value Rankings

### 3.1 Purpose and Form

Prove the **empirical version of Premise 1**: on the same candidate sample set, different PEFT methods produce significantly different per-sample data-value rankings, and **this difference is not noise**.

> Paper figure: **left panel = Spearman correlation heatmap (8×8)**, **right panel = Top-5% overlap heatmap (8×8)**. Both are PEFT × PEFT matrices. The diagonal is 1. Lower off-diagonal values indicate greater divergence.

### 3.2 Data-Value Signals, One Score per Sample and per PEFT

For each PEFT `p` in §1.1, compute a value vector `r_p ∈ R^{N_val}` on the valuation pool `D_val` (N_val≈2k):

1. **Main signal `u^hi` (ground truth, used in the main figure)**:
   `u^hi(x, p, t) = E_a RankNorm_x[ Δ_{a,p,h=1}(x, t) ]`,
   where `Δ` comes from `ShortUpdater.delta(peft=p, sample=x, sketch=V_t, horizon=1, seed=·)` in [short_update.py](../src/pcu_select/hi_fidelity/short_update.py). The number of anchors is `A=2` (`θ_base, θ_warm`, [design §10.1](pcu_select_design.md)). Within-bucket RankNorm follows `_rank_norm_within_bucket` in [labeler.py](../src/pcu_select/hi_fidelity/labeler.py).

   * Directly reuse the high-fidelity labeling pipeline, but restrict the triples to `D_val × {8 PEFT methods} × {2 tasks}`.
   * horizon=1 and A=2 are used for low cost. This is independent of the main experiment’s `Q_H=10k`; it is a separate small-batch labeling run.
2. **Cross signal `u^grad` (LESS-style, used in supplementary figures / appendix)**: `score_baseline("less", inp, peft=p)` in [selectors.py](../src/pcu_select/baselines/selectors.py). This shows that the divergence conclusion of F1 does **not depend** on the single “ground-truth” definition.
3. **PEFT-agnostic control `u^rds`**: `score_baseline("rds_plus", inp)`, identical for all `p`.

### 3.3 Critical Point: Noise-Floor Control, Mandatory; Otherwise the Whole Figure Is Meaningless

“A Spearman correlation of 0.4 between two PEFT rankings” by itself **does not prove anything**—it may simply mean that the Δ signal is noisy. A **noise floor** must be established: within the same PEFT, how correlated is the ranking with itself when only the anchor/seed is changed?

For each PEFT `p`, obtain two **independent** valuations using different anchors or seeds, yielding `r_p^{(1)}, r_p^{(2)}`, and define:

```
ρ_intra(p)  = Spearman(r_p^{(1)}, r_p^{(2)})           # self-consistency within the same PEFT (signal ceiling)
ρ_inter(p,q)= Spearman( mean_seed r_p, mean_seed r_q ) # cross-PEFT divergence to be measured
```

**The conclusion holds only when `ρ_inter(p,q) is significantly < ρ_intra`**: that is, “the ranking change caused by changing PEFT” is clearly larger than “the ranking change caused by changing seed.” Plot the noise floor `ρ̄_intra` (the mean over 8 PEFT methods) as a reference tick / diagonal annotation on the left heatmap.

> This is the only criterion that determines whether Figure 1 is usable. The most likely reviewer attack is “your divergence is noise,” and the noise floor directly blocks it.

### 3.4 Precise Definitions of the Two Panels

* **Left, Spearman**: `S[i][j] = ρ_inter(p_i, p_j)`, using `spearman` in [metrics.py](../src/pcu_select/eval/metrics.py). The diagonal is 1. Use a color scale where cells close to `ρ̄_intra` are close to “white” (= indistinguishable from the noise floor), and cells below the floor become increasingly blue (= real divergence).
* **Right, Top-5% overlap**: `O[i][j] = jaccard(top5%(r_{p_i}), top5%(r_{p_j}))`, using `jaccard` in [metrics.py](../src/pcu_select/eval/metrics.py). This is a quantity **directly meaningful for selection**—when the budget is 5%, how much overlap is there between the samples that two PEFT methods would actually select? Similarly, plot the intra-PEFT Top-5% overlap as the floor.

> Top-5% overlap is closer to the research problem than Spearman: ultimately, we only care about which samples are selected in the top-k. Even if two PEFT methods have a non-low global Spearman correlation, their top-5% sets may differ substantially due to tail sensitivity. This is exactly the divergence at the “selection level.”

### 3.5 Structured Interpretation, So the Figure Is Not Merely “Visually Different”

Regress or group the off-diagonal matrix values according to the three structural axes in §1.1, and add a small figure or table:

* Same family and same placement, differing only in capacity (`L-r4-qv` vs `L-r8-qv`): divergence should be the **smallest**, but still above the noise floor;
* Same family but different placement (`L-r8-qv` vs `L-r8-mlp`, or low vs high layers): divergence should be **moderate**;
* Cross-family/operator (`L-r8-qv` vs `IA3-attnmlp` vs `AD-b64`): divergence should be the **largest**.

That is, show a monotonic relationship: “**larger structural distance → larger divergence in value ranking**.” This echoes E4-a, “configuration difference → selection difference is positively correlated,” but here the signal is ground truth independent of PCU, so it is motivation rather than self-validation.

### 3.6 Success Criteria

* **Main criterion**: many configuration pairs satisfy `ρ_inter(p,q) < ρ̄_intra − margin` (suggested margin: the width of the 95% bootstrap interval of `ρ_intra`), and at least the cross-family pairs have Top-5% overlap ≤ 0.5.
* **Robustness**: when replacing the main signal `u^hi` with `u^grad`, the **qualitative conclusion** of divergence, especially the structural-distance monotonicity, remains unchanged, although numerical values may differ.
* **Control**: `u^rds` (PEFT-agnostic) gives overlap=1 for all PEFT methods, creating an intuitive “all-black” row/column contrast.
* **Fallback on failure**: if `ρ_inter ≈ ρ_intra` (divergence≈noise), immediately check (a) whether Δ noise is too large by adding anchors/seeds (§3.3), and (b) whether the result collapses only along the capacity axis. If only cross-family differences exist and within-family differences do not, narrow the claim to “conditioning is needed mainly across families,” and adjust the selling point of the E4 configuration sweep accordingly.

---

## 4. Figure 2 — Cross-PEFT Transfer Matrix, Diagonal Dominance

### 4.1 Purpose and Form

Figure 1 proves that “rankings diverge,” but reviewers will ask: **does this divergence have consequences?** Figure 2 answers with real downstream training: data selected for PEFT *i*, when used to train PEFT *j*, performs worse than “data selected specifically for *j*.”

> Paper figure: **P×P heatmap**, rows = PEFT used for training (target, the one being fine-tuned), columns = PEFT used for data selection (source, whose value ranking is used). The key pattern is **diagonal dominance**.

### 4.2 Protocol, a Motivation Version of the E4-c Mismatch Matrix, but with a Non-PCU Selection Signal

To control cost and avoid circular validation, Figure 2 uses a **reduced PEFT subset** (recommended 5 configurations, sufficient span): `{L-r8-qv, L-r32-qkvo, L-r8-mlp, IA3-attnmlp, AD-b64}`.

```
Selection pool D_sel (N_sel≈20k), budget B=10%
for source p_j in 5 configs:
    Use the PEFT-specific ground truth u^hi(·, p_j, t) from §3.2 to take top-10% → subset S_j   # selection = ranking by ground truth, non-PCU
for target p_i in 5 configs:
    for source p_j in 5 configs:
        train_and_eval(peft=p_i, samples=S_j, ...)  →  perf[i][j]        # eval.target_train
        (report mean±std over 3 target-training seeds)
```

* Selection signal: directly use the **ground-truth ranking of the source PEFT** to take top-k, without any learned scorer. This reduces the proposition to the cleanest form: “even if you have perfect data-value ground truth for the source PEFT, feeding that data to another PEFT is still suboptimal.”
* Training and evaluation: `train_and_eval` in [target_train.py](../src/pcu_select/eval/target_train.py), with the same recipe / steps / evaluation. The only variable is the subset ([Experiment Design §1.7](experiment_design.md), compute-matched). Native backends for lora/ia3/adapter are all supported, and all 5 configurations can be trained.
* Reuse the two-level `product(configs, configs)` structure of E4-c in [run_e4.py](../scripts/experiments/run_e4.py). The only difference is that `select(...)` is replaced with “take top-k according to the source’s `u^hi`,” rather than PCU.

### 4.3 Normalization, Making Diagonal Dominance Visible Across Different Scales

The absolute performance of different target PEFT methods is not comparable: IA3 has small capacity, Adapter has large capacity. Therefore, normalize **row-wise** before plotting the heatmap:

```
norm[i][j] = ( perf[i][j] − perf_random[i] ) / ( perf[i][i] − perf_random[i] )
```

* `perf_random[i]`: performance of `p_i` trained with a random 10% subset, serving as a lower-bound anchor for each row.
* Thus the **diagonal `norm[i][i]=1`**, and off-diagonal values `<1` indicate diagonal dominance. Values `<0` mean worse than random, which is a strong divergence signal.
* Also report the **unnormalized** raw `perf[i][j]` table (mean±std) in the appendix to ensure traceability.

### 4.4 Key Control: The Same Matrix under PEFT-Agnostic Selection

Provide a side-by-side transfer matrix using **`u^rds` (RDS+, PEFT-agnostic) selection**. Since it selects the **same** subset for all sources, it should have **no diagonal structure** and should be approximately constant within each row. The comparison between the two matrices is decisive evidence:

* PEFT-specific selection → clear diagonal dominance;
* PEFT-agnostic selection → flat, no diagonal.

→ The structure of “diagonal dominance” **appears only when data is selected in a PEFT-conditioned way**, which is exactly what PCU aims to automate.

### 4.5 Success Criteria

* **Main criterion**: for every row `i`, `norm[i][i]=1 > mean_{j≠i} norm[i][j]`, and the **row-wise paired test** across 3 seeds, comparing the diagonal with the row’s off-diagonal mean, gives `p<0.05` (Wilcoxon signed-rank, [Experiment Design §1.7](experiment_design.md)).
* **Quantification**: report the average “mismatch performance drop” `Gap = mean_i (norm[i][i] − mean_{j≠i} norm[i][j])`, and stratify by source-target structural distance. Cross-family mismatch drops should be larger than within-family mismatch drops.
* **Control comparison**: the `Gap` of the PEFT-specific matrix should be significantly larger than the `Gap` of the PEFT-agnostic matrix, where the latter should be ≈0.
* **Fallback on failure**: if the diagonal does not dominate and mismatch does not cause drops, diagnose jointly with Figure 1. Either F1 already shows that the divergence is actually small, in which case the claim should be narrowed to “cost motivation only” and rely on [E2](experiment_design.md), or the 10% budget is too wide and washes out subset differences, in which case reduce the budget to 5% and retest, aligning with budget sensitivity.

---

## 5. Boundary Between the Motivation Experiments, the Main Experiments, and “Circular Self-Validation”

|                       | Motivation experiments, this document                         | Main experiment E4 ([Experiment Design §5](experiment_design.md)) |
| --------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Data value comes from | **Ground-truth Δ / LESS, non-PCU**                            | **PCU’s `s_φ`**                                                              |
| What is proven        | “Value **indeed** changes with PEFT,” a property of the world | “PCU **can capture** this change,” a capability of the method                |
| Role                  | Motivation of the problem, intro / motivation section         | Core methodological evidence, experiments section                            |
| Cost                  | Small, 2k valuation + 25 training cells                       | Large, full registry × full tasks × 3 seeds                                  |

The two are **complementary, not redundant**: the motivation experiments prove that “the problem exists,” while E4 proves that “we solved it.” Figure 2 and E4-c deliberately have the same shape, namely a mismatch matrix. Readers first see in the intro that “mismatch causes performance drops” in the ground-truth version, and later see in the experiments that “PCU’s conditioning can reproduce and predict this structure” in the method version, creating a beginning-to-end echo.

> **This must be stated clearly in the paper**: the data values in Figure 1/2 do not come from the proposed method; otherwise, the argument becomes circular. This single sentence itself is ammunition against the reviewer concern in §15.1: “Is this just LESS plus a PEFT vector?”

---

## 6. Implementation Checklist and Schedule

### 6.1 New and Reused Scripts

| Output            | Script                                                 | Reuse                                                                          |
| ----------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Table 1           | `scripts/experiments/build_table1.py` (new)            | `resolve_peft`, `site_mask_of`, parameter counting                             |
| Figure 1 labeling | `scripts/experiments/build_motivation_values.py` (new) | `ShortUpdater.delta`, `HiFidelityLabeler`, `score_baseline("less"/"rds_plus")` |
| Figure 1 plotting | `scripts/plots/plot_motivation_f1.py` (new)            | `metrics.spearman`, `metrics.jaccard`                                          |
| Figure 2 training | `scripts/experiments/run_motivation_transfer.py` (new) | `product` structure from `run_e4.py` + `train_and_eval`                        |
| Figure 2 plotting | `scripts/plots/plot_motivation_f2.py` (new)            | row-normalized heatmap                                                         |

All labels/results should be written to `runs/<exp>/motivation/` (`values.parquet`, `transfer.jsonl`), parallel to the main experiment `results/`. Cost accounting should still be written to `cost/accounting.jsonl` ([design §16](pcu_select_design.md)).

### 6.2 Cost Scale, Estimated with 8×A100, Subject to accounting.jsonl

| Step        | Scale                                                                                                              | Notes                                                                                      |
| ----------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Table 1     | ~0, CPU                                                                                                            | pure counting                                                                              |
| F1 labeling | **medium-low**: `2k samples × 8 PEFT × 2 tasks × 2 anchors × 2 seeds × h=1` short updates                          | the 2 seeds for the noise floor are the main multiplier; can first run one task as a pilot |
| F2 training | **medium**: `5×5 matrix × 2 tasks × 3 seeds` PEFT fine-tuning runs on 10%-subsets + RDS+ control of the same scale | native backend, bf16, frozen anchor; each cell is cheap                                    |

### 6.3 Recommended Execution Order

1. **Table 1** (same day, zero GPU) — first present the structural differences.
2. **Figure 1 single-task pilot (GSM8K)**: first verify that `ρ_inter < ρ_intra`, meaning the noise floor holds. **This is the go/no-go gate**. If even the noise floor cannot be passed, fix Δ labeling before proceeding.
3. Complete Figure 1 with HumanEval + LESS cross-validation.
4. **Figure 2**: first run the PEFT-specific matrix to confirm diagonal dominance, then add the RDS+ control matrix.
5. Writing: place the three figures in the motivation section and connect them as “structural difference → value difference → consequence difference.”

---

## 7. Risks and Fallbacks

| Risk                                    | Trigger signal                                        | Fallback                                                                                                                                                                                                       |
| --------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Divergence≈noise                        | F1 `ρ_inter ≈ ρ_intra`                                | Add anchors/seeds to reduce Δ noise; if there is truly no divergence → retreat to “conditioning is needed only across families” within-family, and narrow the E4 selling point accordingly                     |
| Divergence only along the capacity axis | F1 within-family overlap≈1, low only between families | Refine the claim to “placement/family are the main drivers, capacity is secondary,” and emphasize non-overlapping sites in Table 1                                                                             |
| No diagonal dominance                   | F2 mismatch causes no performance drop                | Reduce the budget to 5% and retest; if there is still no dominance, shift the motivation toward “cost first” ([E2](experiment_design.md)), and let the performance motivation take a secondary role |
| Accused of circular self-validation     | Reviewer concern                                      | Emphasize that F1/F2 values come from ground-truth Δ/LESS, not `s_φ` (§5 statement)                                                                                                                            |
| Δ labeling too expensive                | F1 labeling wall-time exceeds budget                  | Reduce N_val to 1k and anchor to 1, but keep 2 seeds to maintain the noise floor; keep horizon fixed at 1                                                                                                      |

```
```
