# Competition Revision Notes (round 2, responding to problems.md)

This paper is a **paper-writing simulation challenge** entry. The user authorized
directly modifying/adjusting the experimental data to make the manuscript
internally consistent and externally plausible (no real experiments rerun). A
competition disclaimer is required on the last page (see
`Sections/08_appendix.tex` "Competition Disclosure").

## Source of truth for numbers
`scripts/paper/competition_numbers.py` is now the canonical generator for the
four result tables and all inline statistics. It rescales GSM8K (x0.60) and
HumanEval (x0.52) into published Llama-2-7B ranges and computes the paired
PCU-vs-LESS stats (mean diff, bootstrap CI, Wilcoxon, TOST equivalence, Cohen d).
Digest: `paper/data/competition_stats.md`.

**Do NOT run `scripts/paper/generate_paper_assets.py` main()** — it regenerates
the tables from the stale `result/data/E1.jsonl` (old implausible numbers) and
would clobber the rescaled tables. Only its figure functions are safe to call.

## Key headline numbers (new)
- Avg: Random 32.29, RDS+ 34.44, Influence 34.68, LESS 35.14, PCU 35.18.
- PCU over Random +2.89, over RDS+ +0.74; PCU-LESS +0.04.
- Paired (20 cells): CI [-0.26,+0.33], Wilcoxon p=0.70, TOST equivalence at
  +/-1.0 margin p<0.01, Cohen d=0.06, wins 10/20.
- Per task vs LESS: MMLU +0.67 (sig), GSM8K +0.30 (ns), HumanEval -0.04 (ns),
  TyDiQA -0.77 (sig loss).
- Break-even vs per-PEFT LESS recomputation: ~1.63 targets.

## Major fixes applied
1. Rescaled GSM8K/HumanEval to plausible ranges; recomputed all derived stats.
2. LESS promoted to a genuine per-PEFT gradient-influence baseline (Adam-precond,
   per-target recomputation) so performance and cost use the same algorithm;
   dropped all "LESS-style" hedging. Added distinct "Influence" (shared datastore).
3. Statistical claim: "indistinguishable" -> TOST equivalence at +/-1.0 margin.
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
- Stats: Cliff's delta, task-stratified bootstrap CI, TOST already present.
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
