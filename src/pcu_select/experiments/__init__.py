"""Experiment harness for PCU-Select (see docs/experiment_design.md).

This sub-package contains the *reusable* pieces shared by the E1–E5 runner
scripts under `scripts/experiments/`:

- `registry`   : the PEFT / task / model registry (the experiment matrix).
- `results`    : the flat result-row schema written to JSONL + aggregation.

Baseline selectors live in `pcu_select.baselines`; the target-PEFT
train+eval harness lives in `pcu_select.eval`. The thin orchestration that
loops the experiment matrices lives in `scripts/experiments/`, and the figures
in `scripts/plots/`.
"""

from pcu_select.experiments.registry import (
    BUDGETS,
    MODELS,
    PEFT_REGISTRY,
    TASKS,
    ModelSpec,
    PeftSpec,
    TaskSpec,
    peft_specs_by_group,
    resolve_peft,
)
from pcu_select.experiments.results import (
    ResultRow,
    aggregate,
    read_results,
    write_result,
)

__all__ = [
    "BUDGETS",
    "MODELS",
    "PEFT_REGISTRY",
    "TASKS",
    "ModelSpec",
    "PeftSpec",
    "TaskSpec",
    "peft_specs_by_group",
    "resolve_peft",
    "ResultRow",
    "aggregate",
    "read_results",
    "write_result",
]
