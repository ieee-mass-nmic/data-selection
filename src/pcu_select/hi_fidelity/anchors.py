"""Anchor checkpoint management. See design doc §10.1.

Two anchors:
  θ_base : pretrained backbone
  θ_warm : after 200 steps of base-config LoRA (rank=8, attn-only) on
           a 1k-sample slice of the meta-pool
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnchorSpec:
    anchor_id: str  # e.g. "base", "warm"
    checkpoint_path: Path
    note: str = ""


class AnchorRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._anchors: dict[str, AnchorSpec] = {}

    def register(self, anchor: AnchorSpec) -> None:
        self._anchors[anchor.anchor_id] = anchor

    def get(self, anchor_id: str) -> AnchorSpec:
        return self._anchors[anchor_id]

    def all(self) -> list[AnchorSpec]:
        return list(self._anchors.values())


def build_warm_anchor(
    *,
    base_model_path: str,
    meta_pool_slice_path: Path,
    out_path: Path,
    steps: int = 200,
    rank: int = 8,
) -> AnchorSpec:
    """Run the standardized warm-up training and persist the resulting weights.

    The implementation is intentionally abstracted out — depending on the
    deployment we might use HuggingFace's `Trainer`, `peft` library, or a
    custom loop. The contract: on success, `out_path` contains a checkpoint
    that downstream high-fidelity short-update routines can load.
    """
    raise NotImplementedError("See design doc §10.1; integrate with chosen training stack.")
