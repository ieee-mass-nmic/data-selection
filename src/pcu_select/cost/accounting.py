"""GPU-hours and break-even accounting. See design doc §16."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class CostEvent:
    stage: str  # feat / lo / hi / scorer_train / apply_score / apply_select / target_train
    wall_time_sec: float
    gpu_hours: float
    peak_mem_mb: float = 0.0
    disk_written_mb: float = 0.0
    n_samples: int = 0
    peft_id: str | None = None
    task_id: str | None = None
    extra: dict = field(default_factory=dict)


class CostAccountant:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: CostEvent) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    @contextmanager
    def stage(self, name: str, *, n_samples: int = 0, peft_id: str | None = None,
              task_id: str | None = None) -> Iterator[dict]:
        meta: dict = {}
        t0 = time.time()
        try:
            yield meta
        finally:
            t1 = time.time()
            wall = t1 - t0
            gpu_count = max(1, int(os.environ.get("PCU_GPU_COUNT", "1")))
            self.log(CostEvent(
                stage=name,
                wall_time_sec=wall,
                gpu_hours=wall * gpu_count / 3600.0,
                peak_mem_mb=meta.get("peak_mem_mb", 0.0),
                disk_written_mb=meta.get("disk_written_mb", 0.0),
                n_samples=n_samples,
                peft_id=peft_id,
                task_id=task_id,
                extra=meta.get("extra", {}),
            ))

    def read_events(self) -> list[CostEvent]:
        if not self.path.exists():
            return []
        events: list[CostEvent] = []
        for line in self.path.read_text().splitlines():
            d = json.loads(line)
            events.append(CostEvent(**d))
        return events


def break_even_T(C_offline: float, C_apply: float, C_specific: float) -> float:
    """Return the number of target PEFTs T at which PCU-Select breaks even."""
    denom = C_specific - C_apply
    if denom <= 1e-9:
        return float("inf")
    return C_offline / denom


def total_cost(events: list[CostEvent]) -> float:
    return sum(e.gpu_hours for e in events)
