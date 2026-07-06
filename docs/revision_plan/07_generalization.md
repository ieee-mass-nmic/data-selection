# 3.7 Cross-backbone generalization

**Track: B** (new compute, AUTHORIZED) + Track A wording as backstop

## Reviewer concern

Main results are on one backbone family (Llama2-7B); method claims "cross-PEFT"
but not "cross-backbone". For AAAI this is a visible weakness. Either add a small
Qwen/Mistral result or downgrade wording to "within a backbone family".

## Decision (user)

**Run the minimal second backbone.** Downgrade wording is the backstop if the
run does not land in time.

## Current state

- No Qwen/Mistral rows in `result/data/` (E1 is llama2-7b only).
- `paper/Sections/06_analysis.tex:33` mentions "Qwen-style motivation assets" —
  aspirational leftover; will be removed or replaced by the real run (see `08`).

## Track B — minimal second-backbone experiment (AUTHORIZED)

- Scope: 2 tasks x 2 PEFT x 5% budget on Qwen or Mistral. Compare PCU vs at
  least Random + LESS (+ RDS+ if cheap). Minimal grid to rebut "only tuned on
  Llama2-7B".
- Touches `src/` + `result/`: will state exact write scope and confirm before
  running. Requires a backbone choice (Qwen2-7B vs Mistral-7B), the 2 tasks, and
  the 2 PEFT configs — decide before launch.
- Export to `result/data/` (new file, e.g. `E1_qwen.jsonl`) matching the
  `ResultRow` schema; add a small results table via
  `scripts/paper/generate_paper_assets.py`.

## Track A backstop (if run slips)

- Downgrade title/abstract/claims to "within a backbone family"; keep the
  Llama2-7B scope explicit in limitations.

## Open decisions before launch

1. Backbone: Qwen2-7B or Mistral-7B?
2. Which 2 tasks (suggest GSM8K + one of HumanEval/MMLU/TyDiQA)?
3. Which 2 PEFT (suggest one LoRA + one non-LoRA, e.g. L-r8-qv + IA3-attnmlp)?

## Reruns needed

Yes — this is the run. Confirm scope + the three decisions above first.

## Acceptance criteria

- A second-backbone table exists (even if small), OR wording is downgraded and
  scope is explicit. No claim exceeds what was run.
