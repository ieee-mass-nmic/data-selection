# AAAI-27 Manuscript Operating Rules

## Project

- Target venue: AAAI-27 Main Technical Track.
- Main TeX file: main.tex.
- Current mode: anonymous review submission.
- Existing AAAI-27 author-kit files are authoritative.
- Technical-content budget: at most 7 pages, followed by references.
- The reproducibility checklist must remain present and complete.

Replace `main.tex` above if the actual main file has another name.

## Template integrity

- Never replace or modify the document class, AAAI style files, bibliography
  style files, margins, fonts, spacing, headers, footers, paper size, or
  submission-mode options.
- Never edit `.sty`, `.cls`, or `.bst` files.
- Never add or remove a LaTeX package without explicitly reporting the need
  and verifying that the official AAAI template permits it.
- Do not automatically add packages such as geometry, balance, fontenc,
  caption, subcaption, cleveref, microtype, or times.
- Preserve existing custom commands unless they cause a demonstrated compile
  error.
- Do not rewrite the complete preamble or replace the project structure.

## Scope control

- Edit only the file and section named in the task.
- Before editing, inspect the relevant source and its citations.
- For a review request, do not edit files unless the prompt explicitly asks
  for changes.
- Do not commit or push changes unless explicitly instructed.
- Present a concise summary of changed files and important diffs.

## Scientific integrity

- Never invent citations, BibTeX records, results, datasets, baselines,
  metrics, hyperparameters, theorem assumptions, statistical tests, or
  implementation details.
- Do not infer a missing number from neighboring results.
- Mark missing evidence with a LaTeX comment:
  % TODO(VERIFY): describe the missing evidence
- Claims must be no stronger than the available evidence.
- Preserve calibrated scientific hedging when warranted.
- Do not apply the paper-writing skill's zero-hedging rule blindly.
- Do not describe correlation as causation.
- Verify that every contribution claim maps to a result, proof, or analysis.

## Anonymity

- While in anonymous-review mode, do not introduce author names,
  affiliations, email addresses, acknowledgments, grant identifiers,
  institution-specific URLs, or identifying repository links.
- Do not rewrite self-citations in a way that identifies the authors.
- Do not modify author metadata unless explicitly instructed.

## Figures and tables

- Never modify raw experiment data.
- Generate plots from CSV or JSON rather than hardcoding values.
- Store one reproducible Python script per generated figure.
- Prefer vector PDF output for plots.
- Use readable labels, units, legends, and colorblind-safe encodings.
- Do not place titles inside plots; explanatory text belongs in the caption.
- For a single-column figure, use `width=\columnwidth`.
- For a double-column `figure*`, use `width=\textwidth`.
- Do not use `0.48\textwidth` blindly.
- Do not insert a generated figure into the paper until the figure and caption
  have been reviewed.

## Validation

After each LaTeX edit:

1. Run:
   latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
2. Check for undefined references and citations.
3. Check for overfull boxes.
4. Report compile status.
5. Report files changed.
EOF