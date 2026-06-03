"""Candidate / meta pool wrappers.

Implements the `DatasetLike` protocol (see `pcu_select.types`).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from pcu_select.types import Sample, SampleID
from pcu_select.utils import sample_id_of


class JsonlPool:
    """Eager-load jsonl pool. Memory-cheap because we keep raw text only."""

    def __init__(self, samples: list[Sample]):
        self._samples = samples
        self._index: dict[SampleID, int] = {s.sample_id: i for i, s in enumerate(samples)}

    @classmethod
    def from_jsonl(cls, path: Path | str, *, instr_key: str = "instruction",
                   resp_key: str = "response", source_key: str | None = None) -> "JsonlPool":
        path = Path(path)
        samples: list[Sample] = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                instr = rec[instr_key]
                resp = rec[resp_key]
                source = rec.get(source_key) if source_key else None
                sid = sample_id_of(instr, resp, source)
                samples.append(Sample(sample_id=sid, instruction=instr, response=resp, source=source))
        return cls(samples)

    # ------- DatasetLike interface -------
    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def take(self, ids: Iterable[SampleID]) -> list[Sample]:
        return [self._samples[self._index[i]] for i in ids if i in self._index]

    # ------- convenience -------
    def to_jsonl(self, path: Path | str) -> None:
        path = Path(path)
        with path.open("w") as f:
            for s in self._samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
