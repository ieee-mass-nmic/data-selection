"""Flat result-row schema shared by every experiment + plotting script.

Each evaluated cell of an experiment matrix (one method × PEFT × task × budget ×
seed × backbone) produces exactly one `ResultRow`, appended as a JSON line to
`runs/<exp>/results/<EXPERIMENT>.jsonl`. The plotting scripts read these rows
back with `read_results` and never touch the heavy artifacts directly.

Keeping one flat schema for all of E1–E5 means a single loader/aggregator works
everywhere; experiment-specific quantities (Jaccard overlap, Mahalanobis d²,
calibration mode, …) live in the free-form `extra` dict.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev


@dataclass
class ResultRow:
    experiment: str  # "E1" .. "E5"
    method: str  # "pcu" | "random" | "rds_plus" | "less" | ablation tag | ...
    peft: str  # registry spec name, e.g. "L-r16-qkvo"
    task: str  # registry task name
    budget: float  # fraction in (0, 1]
    seed: int
    model: str  # backbone tag, e.g. "llama2-7b"

    # ---- downstream performance ----
    metric_name: str = "eval_loss"  # what `metric` measures
    metric: float = float("nan")  # primary downstream number (higher = better)
    eval_loss: float = float("nan")  # always-available held-out response-LM loss

    # ---- selection-quality / ranking (vs high-fidelity truth on held-out triples) ----
    spearman: float = float("nan")
    kendall_tau: float = float("nan")
    ndcg_at_k: float = float("nan")
    topk_hit_rate: float = float("nan")
    pairwise_acc: float = float("nan")

    # ---- cost (GPU-hours) ----
    select_gpu_h: float = 0.0
    target_train_gpu_h: float = 0.0
    offline_gpu_h: float = 0.0  # amortized one-time cost attributed to this row's method

    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def write_result(path: Path | str, row: ResultRow) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(row.to_json() + "\n")


def read_results(path: Path | str) -> list[ResultRow]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[ResultRow] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(ResultRow(**json.loads(line)))
    return rows


def aggregate(
    rows: list[ResultRow],
    *,
    by: tuple[str, ...] = ("method", "peft", "task", "budget", "model"),
    value: str = "metric",
) -> dict[tuple, dict[str, float]]:
    """Group `rows` by the `by` keys and reduce `value` to mean/std/n.

    Returns {group_key_tuple: {"mean": ..., "std": ..., "n": ...}}. Used by the
    plotting scripts to collapse the per-seed rows into points with error bars.
    """
    buckets: dict[tuple, list[float]] = {}
    for r in rows:
        key = tuple(getattr(r, k) for k in by)
        v = getattr(r, value)
        if v is None or (isinstance(v, float) and v != v):  # skip NaN
            continue
        buckets.setdefault(key, []).append(float(v))
    out: dict[tuple, dict[str, float]] = {}
    for key, vals in buckets.items():
        out[key] = {
            "mean": mean(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "n": float(len(vals)),
        }
    return out
