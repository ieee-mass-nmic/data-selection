# 3.8 Internal consistency + polish

**Track: A** (cleanup; no reruns)

## Reviewer concern

Naming/consistency issues: Table 1 vs Table 2 vs Figure 1 PEFT sets differ;
Figure 4 matrix values undefined vs caption; "configuration distance" undefined;
internal phrasing "Qwen-style motivation assets" in prose; PCU acronym not
expanded; typography (refs, hyphens, equation line-breaks).

## Confirmed issues (from reading the files)

1. **Table 1 vs Table 2 mismatch (real).**
   - `paper/tables/table_peft_configs.tex` lists 6 configs including
     `L-r4-qv`, `L-r32-qkvo` — these are the UNSEEN/OOD configs used in E5.
   - `paper/tables/table_main_results.tex` columns are `AD-b64`, `IA3`,
     `L-r16-qkvo`, `L-r8-mlp`, `L-r8-qv` — the SEEN set (matches E1's 5 PEFTs).
   - Fix: unify naming (e.g. `IA3` vs `IA3-attnmlp`) and add an explicit
     **seen / unseen** column so representative/seen/unseen relationship is clear
     across Table 1, Table 2, and Figure 1.
2. **Figure 4 caption (`06_analysis.tex:9`)**: matrix shown as 0.84–1.00 but
   caption says "2.42-point diagonal-off-diagonal gap". State whether cells are
   normalized performance / transfer ratio / overlap, and reconcile the two
   scales.
3. **Define "configuration distance"** where used (Fig 3 / analysis).
4. **Remove internal phrasing** "result bundle includes Qwen-style motivation
   assets" from `06_analysis.tex:33` (replace with the real second-backbone
   result from `07`, or delete).
5. **Expand PCU acronym** (PEFT-Conditional Utility) in title/intro.
6. **Typography pass** on final PDF: references, hyphens, equation line-breaks,
   overfull boxes.

## Data sources / files

- Read: `result/data/E1.jsonl` (seen set), `result/data/E5.jsonl` (unseen set),
  `result/data/E4_overlap.json` / `E4.jsonl` (Fig 4 semantics),
  `src/pcu_select/experiments/registry.py` (canonical config names).
- Write: `paper/tables/table_peft_configs.tex`, `table_main_results.tex`,
  `paper/Sections/06_analysis.tex`, title/intro, captions.

## Reruns needed

No. Naming/caption/typography only. Fig 4 numbers recomputed from existing data
if the caption/matrix reconciliation needs it.

## Acceptance criteria

- One consistent PEFT naming + seen/unseen labeling across all tables/figures.
- Fig 4 caption matches the matrix semantics.
- Acronym expanded; internal phrasing removed; clean final PDF.
