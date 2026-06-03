"""Hashing + seeding utilities."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np


def stable_hash(obj: Any, *, length: int = 16) -> str:
    """Deterministic content hash (sha256 truncated). Keys are sorted."""
    if isinstance(obj, (dict, list, tuple)):
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    else:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def sample_id_of(instruction: str, response: str, source: str | None = None) -> str:
    return stable_hash({"i": instruction, "r": response, "s": source or ""})


def peft_id_of(payload: dict) -> str:
    return stable_hash(payload)


def task_id_of(name: str, sketch_indices: list[int]) -> str:
    return stable_hash({"name": name, "idx": sorted(sketch_indices)})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
