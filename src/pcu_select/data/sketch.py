"""Validation-sketch construction & loading.

Sketch protocol (see design doc §9.1):
- Source: train/dev split of the target task, never the test set.
- Default size 32, stratified by length tercile.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pcu_select.types import Sample, ValidationSketch
from pcu_select.utils import task_id_of


def build_sketch(
    source_samples: list[Sample],
    *,
    task_name: str,
    n: int = 32,
    seed: int = 0,
    stratify_by_length: bool = True,
) -> ValidationSketch:
    """Random / stratified sampling. Returns a fresh ValidationSketch object.

    The returned sketch's `task_id` is content-derived from the sampled indices
    so that two calls with the same seed yield the same id.
    """
    rng = random.Random(seed)
    pool = list(source_samples)
    if stratify_by_length and len(pool) >= 3 * n:
        pool_sorted = sorted(pool, key=lambda s: len(s.response))
        terciles = _split_terciles(pool_sorted)
        per_t = n // 3
        leftover = n - per_t * 3
        picked: list[Sample] = []
        for k, group in enumerate(terciles):
            count = per_t + (1 if k < leftover else 0)
            picked.extend(rng.sample(group, min(count, len(group))))
        out = picked
    else:
        out = rng.sample(pool, min(n, len(pool)))

    idx = [_index_of(s, source_samples) for s in out]
    tid = task_id_of(task_name, idx)
    return ValidationSketch(task_id=tid, samples=out, sketch_seed=seed)


def save_sketch(sketch: ValidationSketch, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": sketch.task_id,
        "sketch_seed": sketch.sketch_seed,
        "samples": [
            {"sample_id": s.sample_id, "instruction": s.instruction, "response": s.response,
             "source": s.source, "language": s.language}
            for s in sketch.samples
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_sketch(path: Path | str) -> ValidationSketch:
    path = Path(path)
    payload = json.loads(path.read_text())
    samples = [
        Sample(
            sample_id=s["sample_id"], instruction=s["instruction"], response=s["response"],
            source=s.get("source"), language=s.get("language"),
        )
        for s in payload["samples"]
    ]
    return ValidationSketch(task_id=payload["task_id"], samples=samples,
                            sketch_seed=payload["sketch_seed"])


# -- helpers ----------------------------------------------------------------


def _split_terciles(sorted_samples: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    n = len(sorted_samples)
    a = n // 3
    b = 2 * n // 3
    return sorted_samples[:a], sorted_samples[a:b], sorted_samples[b:]


def _index_of(needle: Sample, haystack: list[Sample]) -> int:
    for i, s in enumerate(haystack):
        if s.sample_id == needle.sample_id:
            return i
    return -1
