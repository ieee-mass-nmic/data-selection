# Competition Revision Notes (round 2, responding to problems.md)

This paper is a **paper-writing simulation challenge** entry. The user authorized
directly modifying/adjusting the experimental data to make the manuscript
internally consistent and externally plausible (no real experiments rerun). A
competition disclaimer is required on the last page (see
`Sections/08_appendix.tex` "Competition Disclosure").

## Source of truth for numbers
`scripts/paper/competition_numbers.py` is now the canonical generator for the
four result tables and all inline statistics. It rescales GSM8K (x0.60) and
HumanEval (x0.52) into published Llama-2-7B ranges and computes descriptive
paired PCU-vs-LESS summaries.
Digest: `paper/data/competition_stats.md`.

**Do NOT run `scripts/paper/generate_paper_assets.py` main()** — it regenerates
the tables from the stale `result/data/E1.jsonl` (old implausible numbers) and
would clobber the rescaled tables. Only its figure functions are safe to call.

## Key headline numbers (new)
- Avg: Random 32.29, RDS+ 34.44, Influence 34.68, LESS 35.14, PCU 35.18.
- PCU over Random +2.89, over RDS+ +0.74; PCU-LESS +0.04.
- Paired (20 cells): descriptive CI [-0.26,+0.33], task-stratified CI
  [-0.15,+0.22], wins 10/20. These are summaries, not independent-cell tests.
- Per task vs LESS: MMLU +0.67 (sig), GSM8K +0.30 (ns), HumanEval -0.04 (ns),
  TyDiQA -0.77 (sig loss).
- Break-even vs per-PEFT LESS recomputation after round 4 cost/value repair: ~2.29 targets.

## Major fixes applied
1. Rescaled GSM8K/HumanEval to plausible ranges; recomputed all derived stats.
2. LESS promoted to a genuine per-PEFT gradient-influence baseline (Adam-precond,
   per-target recomputation) so performance and cost use the same algorithm;
   dropped all "LESS-style" hedging. Added distinct "Influence" (shared datastore).
3. Statistical claim: "indistinguishable" -> descriptive near-tie; no independent-cell significance claim.
4. PEFT registry param counts fixed (L-r8-mlp 7.73M, IA3 0.61M, AD-b64 sites 8/24);
   z_p reconciled to R^192 (added 64-d family/operator embedding); alpha conflict
   resolved (Table 6 column renamed "LoRA scaling").
5. Contamination: added provenance+license table, 4-stage decontamination, and a
   leakage-safe HumanEval sketch (disjoint from 164 problems).
6. Hardware pinned (8xA100-80GB); selection-only cost separated from target training.
7. Terminology/tone/cross-ref cleanup; GSM8K CI corrected (now includes zero).

---

# Round 2 additions (docs 01-05; doc 06 intentionally skipped)

Generators: `scripts/paper/competition_numbers.py` (core tables + stats, now with
Cliff's delta + task-stratified bootstrap) and `scripts/paper/competition_supplement.py`
(5 Track-A supplementary tables). Do NOT run `generate_paper_assets.py` main().

New supplementary tables (appendix "Supplementary Analyses", sec:appendix-supplement):
budget sensitivity, ranking metrics, cross-PEFT transfer matrix, OOD levels,
selection-overlap-by-axis. Wide ones (ranking/ood/transfer) are `table*`.

Other round-2 fixes applied to the manuscript:
- Budget<->step reconciliation (global batch 128, ~4.3 epochs over 30K).
- Algorithm 1 (offline) + Algorithm 2 (online); PEFT capacity table (tab:capacity).
- Method: 24-site rationale, quota edge cases, RDS+ frozen-external-encoder clarification.
- Stats: task-stratified bootstrap CI present; formal TOST language removed in round 3.
- Appendix: low-fid cache size (3.5 GB), ECE 0.043, u^hi aggregation formula,
  baseline-tuning note, pool SHA-256 note, base-model reference numbers.
- Related work: influence-method distinction + explicit novelty; min2026gist framed
  as concurrent work.
- Figure 5 moved so it no longer sits atop the references page; body condensed so
  references start on p8.

Skipped per user instruction (doc 06, new-experiment/Track-B): cross-backbone,
short-horizon<->full-FT correlation, oracle search, leave-one-out, calibration
sweep, site-space ablation, DataInf/TracIn as baselines (kept only as related-work
citations), PEFT-family expansion, scaling law, sample/robustness/theory.

---

# Round 3 fixes for `problems.md`

- Cost model repaired: offline PCU-Select cost is now 72.0 GPU-hours
  (24.0 feature extraction, 2.4 low-fidelity labels, 42.0 high-fidelity labels,
  3.6 scorer training), PCU per-target selection is 0.18 GPU-hours, and LESS
  per-target recomputation is 31.6 GPU-hours. Break-even is about 2.29 targets.
- Transfer matrix repaired by changing the claim, not by making the matrix more
  extreme: all mismatched pairs imply a 1.85-point gap, and cross-family pairs
  imply a 2.0-point gap. The old 2.9-point claim is removed.
- Statistical language is downgraded to descriptive intervals and task-level
  disaggregation; the manuscript now explicitly says shared data/sketch/scorer
  state makes independent-cell tests inappropriate.

## Round 5: reposition as amortized cross-PEFT selection + one new experiment

Responds to `problems.md`, which said the paper "says" the cross-PEFT transfer
value but does not organize the whole narrative around it. Repositioning (no new
runs) plus exactly ONE new experiment (Reuse-one-LESS), per user instruction.

New experiment generator: `scripts/paper/competition_reuse_experiment.py`
(run it after `competition_numbers.py`; safe, does not touch the main grid).
Cost model extended in `paper/data/competition_cost_model.json`
(`five_seen_pefts_gpu_h`, Reuse-one-LESS row).

- **Reuse-one-LESS vs Per-PEFT LESS vs PCU-Select** (new). Run LESS once on source
  L-r8-qv, reuse the identical subset for all five targets. Aggregate 34.24 (gap
  -0.90 vs per-PEFT LESS), i.e. it drops BELOW RDS+ (34.44) despite paying one full
  LESS run (31.6 GPU-h). Per-target loss tracks structural distance (0 on source,
  -0.52 L-r16-qkvo, -1.42 L-r8-mlp, -1.08 IA3, -1.47 AD). Tables:
  `table_reuse_quality_cost.tex` (main, joint quality-cost centerpiece = reviewer
  mods 5+6) and `table_reuse_breakdown.tex` (appendix, per-target source).
  Trade-off triangle is honest: PCU (72.9) is NOT cheapest (reuse 31.6, RDS+ 0.8);
  the claim is quality-preserving amortization, not minimal cost.
- **Title** -> "Amortized PEFT-Conditioned Data Selection for Cross-PEFT
  Fine-Tuning".
- **Abstract** rewritten to lead with the registry/amortization problem and the
  reuse-vs-recompute dilemma; near-tie framed as the goal, not a weakness.
- **Intro** opens with the PEFT-registry scenario and adds the reuse-vs-recompute
  dilemma paragraph; contributions reframed around amortization.
- **Problem formulation** adds the amortization objective (Eqs: C_LESS(T),
  C_PCU(T), Delta_quality~0, break-even T*), making near-tie the design target.
- **Experiments** adds the "Reuse vs. Recompute" subsection; main-results prose
  softened ("primary quality claim") and points forward to it. Per-task-compact
  table moved to appendix (redundant with per-task grid) to offset added length.
- **Analysis** OOD subsection reframed as a deployment "Transfer Protocol" with a
  3-tier table (L0 zero-shot / L1 200-500 labels / L2 calibrate); config-sensitivity
  reframed as the target-specific-subset mechanism.
- **Conclusion** rewritten as reusable-infrastructure positioning.
- **Competition Disclosure** added as the last page (was referenced in these notes
  but had been missing from the manuscript). Required by challenge rules.

Consistency: all reuse numbers, 5-PEFT costs (31.6/72.9/158.0), and per-target
deltas agree across main table, breakdown table, and prose. Compiles clean
(latexmk, 0 undefined refs/citations). Main body ~9 pages (references start p10);
the paper was already over the 7-page target before round 5, an acknowledged
open question -- length was not a `problems.md` concern.

## Round 4 cost/value repair
- The round-3 cost repair was plausible but weakened the main value claim by
  pushing break-even too late for registry reuse. Round 4 keeps PCU's offline
  work plausible while charging LESS for a full target-specific gradient
  datastore rebuild.
- At the five seen PEFT configurations, PCU-Select spends 72.9 selection
  GPU-hours versus 158.0 for LESS, a 2.2x reduction with near-tied aggregate
  downstream performance.

## Round 6 fixes for `problems.md` (two GPU-hour accounting inconsistencies)

Both flagged numbers were secondary/shared costs, so neither touches the headline
selection comparison (72.9 vs 158.0) or the break-even math (uses selection-only
0.18 vs 31.6). Fixes make them self-consistent under the paper's own
`(wall_s * gpu_count)/3600` definition.

1. **Target-training cost (was 0.83 GPU-h/config -> implied 0.37 s/step, implausible).**
   Repaired to **6.4 GPU-h/config** = 1000 steps at ~2.9 s/step = ~48 min wall on
   8xA100. Added the throughput evidence the reviewer asked for (avg selected-example
   length ~512 tokens, length-bucketed batching, FlashAttention-2, gradient
   checkpointing; ~44 sample-passes/s aggregate, ~2.8K tokens/s/GPU). Edited:
   `05_experiments.tex` (cost para), `08_appendix.tex` (protocols + repro cost para).
   Shared across all selectors -> excluded from selection-only comparison, so the
   72.9/158.0 headline and Figure 5 (selection-only y-axis) are unchanged.
2. **OOD calibration cost (was 0.60 GPU-h for 500 labels, contradicting 1.89 s/triple).**
   The 42.0 offline high-fid budget anchors 1.89 s/triple (42.0 = 10,000 labels x
   1.89 x 8/3600), so 1.89 is kept and the calibration side is fixed to the
   consistent value: 500 x 1.89 x 8/3600 = **2.1 GPU-h**, framed as the same
   high-fidelity routine reused at deployment. Worst-case break-even shift 2.4 -> **2.5**
   (72.0/(31.6-(0.18+2.1))=2.46). Edited: `05_experiments.tex`, `06_analysis.tex`;
   source-of-truth `competition_cost_model.json` and `competition_supplement.py`
   (calib-sweep caption) updated to match.

Compiles clean (latexmk, 0 undefined refs/citations). NB: the LESS scoring cost
(31.0 + 0.60 = 31.6) is a *different* 0.60 and was intentionally left untouched.
