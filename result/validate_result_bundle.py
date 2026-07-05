"""Validate the PCU-Select result bundle.

The validator reads committed result files and checks schema, metric ranges,
cost accounting, and cross-artifact consistency. It does not create or alter
experiment measurements.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
TABLES = HERE / "tables"
POOL_SIZE = 300_000

RESULT_FILES = [
    "E1.jsonl",
    "E2.jsonl",
    "E3.jsonl",
    "E4.jsonl",
    "E5.jsonl",
    "MOT_F2.jsonl",
]

NUMERIC_COLUMNS = [
    "metric",
    "eval_loss",
    "spearman",
    "kendall_tau",
    "ndcg_at_k",
    "topk_hit_rate",
    "pairwise_acc",
    "select_gpu_h",
    "target_train_gpu_h",
    "offline_gpu_h",
]

CSV_MIRRORS = [
    "F1_structural_u_hi.csv",
    "F1_structural_u_grad.csv",
    "F2_transfer_raw.csv",
    "T1_method_x_peft.csv",
    "T2_ablation.csv",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty result file")
    return rows


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _group_mean(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, float]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        extra = row.get("extra") or {}
        key_parts = []
        for key in keys:
            if key.startswith("extra."):
                key_parts.append(extra.get(key.split(".", 1)[1]))
            else:
                key_parts.append(row.get(key))
        buckets[tuple(key_parts)].append(float(row["metric"]))
    return {key: sum(vals) / len(vals) for key, vals in buckets.items()}


def _check_rows(name: str, rows: list[dict], errors: list[str]) -> None:
    for idx, row in enumerate(rows, 1):
        for col in NUMERIC_COLUMNS:
            if col not in row or not _finite(row[col]):
                errors.append(f"{name}:{idx}: {col} must be finite")
                continue
            value = float(row[col])
            if col.endswith("_gpu_h") and value < 0:
                errors.append(f"{name}:{idx}: {col} must be non-negative")
            if col in {"ndcg_at_k", "topk_hit_rate", "pairwise_acc"} and not 0 <= value <= 1:
                errors.append(f"{name}:{idx}: {col} must be in [0, 1]")
            if col in {"spearman", "kendall_tau"} and not -1 <= value <= 1:
                errors.append(f"{name}:{idx}: {col} must be in [-1, 1]")

        budget = float(row.get("budget", -1))
        expected = int(round(budget * POOL_SIZE))
        extra = row.get("extra") or {}
        if extra.get("n_selected") != expected:
            errors.append(f"{name}:{idx}: n_selected does not match budget")


def _check_e2(rows: list[dict], errors: list[str]) -> None:
    model_path = DATA / "E2_cost_model.json"
    if not model_path.exists():
        errors.append("E2_cost_model.json is missing")
        return
    cost_model = json.loads(model_path.read_text())
    offline = float(cost_model["offline_gpu_h"])
    recompute = float(cost_model["per_peft_recompute_gpu_h"])
    influence_methods = set(cost_model.get("influence_methods", []))
    t_values = [int(t) for t in cost_model.get("T_values", [])]

    apply_buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        apply_buckets[row["method"]].append(
            float(row["select_gpu_h"]) + float(row["target_train_gpu_h"])
        )
    apply_h = {method: sum(vals) / len(vals) for method, vals in apply_buckets.items()}
    if "pcu" not in apply_h:
        errors.append("E2: missing PCU rows")
        return

    def total(method: str, count: int) -> float:
        if method == "pcu":
            return offline + count * apply_h[method]
        if method in influence_methods:
            return count * (recompute + apply_h[method])
        return count * apply_h[method]

    for method in sorted(influence_methods):
        if method not in apply_h:
            errors.append(f"E2: missing influence method {method}")
            continue
        for count in t_values:
            if count >= 2 and total("pcu", count) >= total(method, count):
                errors.append(f"E2: PCU cost must be lower than {method} at T={count}")


def _check_e4(rows: list[dict], errors: list[str]) -> None:
    overlap_path = DATA / "E4_overlap.json"
    if not overlap_path.exists():
        errors.append("E4_overlap.json is missing")
        return
    overlap = json.loads(overlap_path.read_text()).get("overlap", {})
    for method, matrix in overlap.items():
        for left, row in matrix.items():
            if abs(float(row[left]) - 1.0) > 1e-12:
                errors.append(f"E4_overlap: {method}:{left} diagonal must be 1")
            for right, value in row.items():
                paired = matrix.get(right, {}).get(left)
                if paired is None or abs(float(value) - float(paired)) > 1e-12:
                    errors.append(f"E4_overlap: {method}:{left}/{right} must be symmetric")

    pcu_rows = [
        row for row in rows
        if row["method"] == "pcu" and (row.get("extra") or {}).get("sub") == "E4b"
    ]
    diag_rows = [
        row for row in rows
        if row["method"] == "pcu_mismatch"
        and (row.get("extra") or {}).get("src_peft") == (row.get("extra") or {}).get("tgt_peft")
    ]
    pcu = _group_mean(pcu_rows, ("peft", "task", "seed"))
    diag = _group_mean(diag_rows, ("peft", "task", "seed"))
    for key, value in pcu.items():
        if key not in diag or abs(value - diag[key]) > 1e-12:
            errors.append(f"E4: mismatch diagonal must match PCU row for {key}")


def _check_motivation_f2(rows: list[dict], errors: list[str]) -> None:
    transfer = [row for row in rows if row["method"] == "transfer"]
    means = _group_mean(transfer, ("task", "extra.tgt_peft", "extra.src_peft"))
    by_target: dict[tuple, dict[str, float]] = defaultdict(dict)
    for (task, target, source), value in means.items():
        by_target[(task, target)][source] = value
    for (task, target), sources in by_target.items():
        diag = sources.get(target)
        if diag is None:
            errors.append(f"MOT_F2: missing diagonal for {(task, target)}")
            continue
        for source, value in sources.items():
            if source != target and value > diag + 1e-9:
                errors.append(f"MOT_F2: off-diagonal {source}->{target} exceeds diagonal")


def _check_csv_mirrors(errors: list[str]) -> None:
    for name in CSV_MIRRORS:
        left = FIGS / name
        right = TABLES / name
        if not left.exists() or not right.exists():
            errors.append(f"{name}: missing figs/tables copy")
            continue
        if left.read_bytes() != right.read_bytes():
            errors.append(f"{name}: figs and tables copies differ")


def _check_motivation_parquet(errors: list[str]) -> None:
    path = DATA / "motivation" / "values.parquet"
    if not path.exists():
        errors.append("motivation/values.parquet is missing")
        return
    if path.stat().st_size == 0:
        errors.append("motivation/values.parquet is empty")


def main() -> int:
    errors: list[str] = []
    loaded: dict[str, list[dict]] = {}
    for name in RESULT_FILES:
        path = DATA / name
        if not path.exists():
            errors.append(f"{name}: missing")
            continue
        try:
            rows = _load_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        loaded[name] = rows
        _check_rows(name, rows, errors)

    if "E2.jsonl" in loaded:
        _check_e2(loaded["E2.jsonl"], errors)
    if "E4.jsonl" in loaded:
        _check_e4(loaded["E4.jsonl"], errors)
    if "MOT_F2.jsonl" in loaded:
        _check_motivation_f2(loaded["MOT_F2.jsonl"], errors)
    _check_csv_mirrors(errors)
    _check_motivation_parquet(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("result bundle validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
