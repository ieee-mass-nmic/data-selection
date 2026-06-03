"""PEFTConfig (de)serialization and resolution helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from pcu_select.types import PEFTConfig, PEFTRecipe
from pcu_select.utils import peft_id_of


def load_peft_config(path: Path | str) -> PEFTConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = PEFTConfig(
        peft_id="",  # filled below from content hash
        family=raw["family"],
        target_modules=list(raw["target_modules"]),
        target_layers=list(raw["target_layers"]),
        rank=raw.get("rank"),
        alpha=raw.get("alpha"),
        adapter_bottleneck=raw.get("adapter_bottleneck"),
        prefix_len=raw.get("prefix_len"),
        recipe=PEFTRecipe(**raw.get("recipe", {})),
        use_fingerprint=raw.get("use_fingerprint", False),
        extra=raw.get("extra", {}),
    )
    payload = asdict(cfg)
    payload.pop("peft_id")
    cfg.peft_id = peft_id_of(payload)
    return cfg


def dump_peft_config(cfg: PEFTConfig, path: Path | str) -> None:
    data = asdict(cfg)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def trainable_params_estimate(cfg: PEFTConfig, *, d_model: int = 4096) -> int:
    """Rough trainable-parameter count, used for capacity normalization.

    Treat module count and layer count multiplicatively.
    """
    n_layers = len(cfg.target_layers)
    n_mods = len(cfg.target_modules)
    if cfg.family == "lora":
        r = cfg.rank or 8
        return n_layers * n_mods * 2 * r * d_model
    if cfg.family == "ia3":
        return n_layers * n_mods * d_model
    if cfg.family == "adapter":
        b = cfg.adapter_bottleneck or 64
        return n_layers * 2 * b * d_model
    if cfg.family == "prefix":
        return (cfg.prefix_len or 8) * n_layers * d_model
    if cfg.family == "bitfit":
        return n_layers * n_mods * d_model  # rough
    if cfg.family == "ptuning":
        return (cfg.prefix_len or 8) * d_model
    return n_layers * n_mods * d_model
