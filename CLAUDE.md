# Research Project Agent Instructions

## Repository structure

- `src/`: authoritative implementation of the proposed method.
- `scripts/`: computation, training, experiment, and result-export scripts.
- `docs/`: research design documents and manually maintained notes.
- `result/`: experiment results.
- `paper/`: AAAI manuscript, figures, tables, and review reports.


## Paper-task write boundaries

When performing a paper-writing or figure-generation task:

- Treat `src/`, `configs/`, `docs/`, `experiments/logs/`, and `results/raw/`
  as read-only.
- Do not modify training or evaluation code unless the task explicitly asks
  for a code change.
- Do not rerun expensive experiments unless explicitly instructed.
- Do not delete, rename, overwrite, or normalize raw experiment files.
- Write manuscript changes only under `paper/`.
- Write generated plots only under `paper/figures/`.
- Write generated LaTeX tables only under `paper/tables/`.
- Write intermediate paper datasets only under `paper/data/`.
- A result-export utility may be added under `scripts/paper/`, but it must not
  modify source results.

## Validation

Before completing a paper-related task:

- Report all files read as primary evidence.
- Report all files changed.
- Run relevant result-generation scripts when applicable.
- Compile `paper/main.tex` after LaTeX changes.
- Show unresolved discrepancies and unsupported claims.

## Claude Code-specific behavior

- For repository exploration, begin with read-only analysis.
- Before changing files, state the intended write scope.
- During paper tasks, do not edit files outside `paper/` or `scripts/paper/`
  unless the user explicitly requests it.
EOF