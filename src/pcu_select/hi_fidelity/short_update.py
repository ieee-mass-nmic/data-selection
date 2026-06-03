"""High-fidelity short PEFT-only update protocol. See design doc §10.2.

The routine here produces a single Δ value for a triple (x, p, t) under one
anchor and one horizon. The labeler module in `labeler.py` orchestrates
batches and the RankNorm aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pcu_select.types import PEFTConfig, Sample, ValidationSketch


@dataclass
class ShortUpdateConfig:
    horizon: int = 1
    device: str = "cuda"
    bf16: bool = True
    seed: int = 0


def init_peft_module(model: torch.nn.Module, peft: PEFTConfig) -> torch.nn.Module:
    """Attach PEFT modules per `peft` to `model`. Returns the wrapped model.

    Real implementation will likely delegate to `peft` library
    (`get_peft_model`) with a converted config.
    """
    raise NotImplementedError("Integrate with `peft` library; see design doc §10.2.")


def evaluate_sketch_loss(model: torch.nn.Module, sketch: ValidationSketch) -> float:
    """Forward-only sum of per-sample response-LM losses (mean over sketch)."""
    raise NotImplementedError


def run_short_update(
    *,
    anchor_checkpoint: Path,
    peft: PEFTConfig,
    sample: Sample,
    sketch: ValidationSketch,
    cfg: ShortUpdateConfig,
) -> float:
    """Execute the short-update protocol, return Δ = L_V(θ_a) - L_V(after).

    Pseudocode:
        model = load(anchor_checkpoint, dtype=bf16, device=cfg.device)
        model.eval(); freeze base params
        peft_model = init_peft_module(model, peft)
        optimizer = build_optimizer(peft_model, peft.recipe)
        L_before = evaluate_sketch_loss(peft_model, sketch)
        peft_model.train()
        for _ in range(cfg.horizon):
            forward(sample); loss.backward(); optimizer.step(); zero_grad()
        peft_model.eval()
        L_after = evaluate_sketch_loss(peft_model, sketch)
        return L_before - L_after
    """
    raise NotImplementedError
