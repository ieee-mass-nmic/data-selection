# 3.3 Reproducibility supplement

**Track: A** (documentation harvest from read-only `src/`; no reruns)

## Reviewer concern

Missing: data pool provenance/license/dedup/contamination; splits; task sketch
construction and HumanEval leakage; backbone/optimizer/lr/steps/scheduler/
grad-accum; full PEFT registry; 24 intervention sites; scorer arch/loss weights/
early stopping/held-out split; cluster count / kmeans params / quota rounding;
hi-fidelity 10K triple sampling ratios; cost GPU model / parallelism. AAAI-27
requires a reproducibility checklist and materials at submission.

## Approach

Harvest from `src/` (read-only) and `configs/`, write into
`paper/ReproducibilityChecklist.tex` + a methods/appendix subsection under
`paper/`. Do not modify source.

## Harvest map

| Item | Source to read |
|---|---|
| PEFT registry (rank, targets, placement, IA3, lr, dropout, init) | `src/pcu_select/peft_space/schema.py`, `src/pcu_select/experiments/registry.py`, `configs/peft/*.yaml` |
| 24 intervention sites (layer choice, projection dim, seed, grad norm) | `src/pcu_select/proxy/projection.py`, `proxy/hooks.py`, `peft_space/site_mask.py` |
| Scorer arch (hidden size, loss weights, epochs, early stopping, held-out split) | `src/pcu_select/scorer/model.py`, `scorer/losses.py`, `scorer/trainer.py` |
| Cluster count, MiniBatch-kmeans params, quota rounding/edge cases | `src/pcu_select/selection/cluster.py`, `selection/adaptive_quota.py` |
| Hi-fidelity triple sampling (coverage/uncertainty/boundary ratios) | `src/pcu_select/hi_fidelity/sampler.py`, `hi_fidelity/labeler.py`, `hi_fidelity/anchors.py` |
| Task sketch construction, 32-sample origin, HumanEval leakage | `src/pcu_select/data/sketch.py` |
| Cost: GPU model, parallelism, caching | `src/pcu_select/cost/accounting.py`, `result/data/E2_cost_model.json` |
| Backbone/optimizer/lr/steps/scheduler/grad-accum | `configs/pipeline/default.yaml`, `src/pcu_select/eval/target_train.py` |
| Candidate pool + splits | `src/pcu_select/data/dataset.py` |

## Two honest-disclosure items (not just documentation)

1. **300K pool contamination check.** If no benchmark-contamination check was
   run, say so explicitly; do not imply one. Document source, license, dedup.
2. **HumanEval has no train/dev.** State exactly how the task sketch is built
   without leakage. If the sketch draws from HumanEval prompts, disclose and
   quantify the overlap. This is a known reviewer trap.

## Reruns needed

No. Pure code-reading + writing. (Any missing number that cannot be found in
code gets a `% TODO(VERIFY)` marker per paper CLAUDE.md, not a guess.)

## Acceptance criteria

- Reproducibility checklist complete; every listed hyperparameter has a source.
- Contamination and HumanEval-leakage handled with explicit statements.
