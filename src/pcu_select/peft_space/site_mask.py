"""Site mask + α_p^ω computation. See design doc §2, §3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pcu_select.peft_space.schema import trainable_params_estimate
from pcu_select.types import ModuleName, OperatorType, PEFTConfig, SiteID


# Default operator weight prior (design doc §3).
RHO_OP: dict[OperatorType, float] = {
    "additive_low_rank": 1.0,
    "multiplicative": 0.6,
    "additive_bottleneck": 0.8,
    "prefix": 0.4,
    "bias_shift": 0.3,
}

ETA: float = 1.0


@dataclass(frozen=True)
class SiteSpace:
    """The default site set Ω."""

    layer_indices: tuple[int, ...]  # e.g. (3,7,11,15,19,23,27,31) for 32 layers
    modules: tuple[ModuleName, ...] = ("attn_out", "mlp_out", "block_residual")

    @classmethod
    def uniform(cls, n_layers_total: int, k: int = 8) -> "SiteSpace":
        if n_layers_total <= k:
            idx = tuple(range(n_layers_total))
        else:
            step = n_layers_total / k
            idx = tuple(int(round((i + 1) * step) - 1) for i in range(k))
        return cls(layer_indices=idx)

    @property
    def all_sites(self) -> list[SiteID]:
        return [(l, m) for l in self.layer_indices for m in self.modules]

    def __len__(self) -> int:
        return len(self.layer_indices) * len(self.modules)


def operator_of(family: str, module: str) -> OperatorType:
    if family == "lora":
        return "additive_low_rank"
    if family == "ia3":
        return "multiplicative"
    if family == "adapter":
        return "additive_bottleneck"
    if family == "prefix":
        return "prefix"
    if family == "bitfit":
        return "bias_shift"
    if family == "ptuning":
        return "prefix"
    return "additive_low_rank"


def _module_targets_match(cfg: PEFTConfig, site_module: ModuleName) -> bool:
    """Decide whether a PEFT touching certain transformer modules influences the given site."""
    tm = set(cfg.target_modules)
    if site_module == "attn_out":
        return bool(tm & {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "attn", "attention"})
    if site_module == "mlp_out":
        return bool(tm & {"up_proj", "down_proj", "gate_proj", "mlp", "fc1", "fc2"})
    if site_module == "block_residual":
        # adapters typically inserted in residual stream; prefix affects all attn layers
        if cfg.family in ("adapter", "prefix", "bitfit"):
            return True
        return False
    return False


def site_mask_of(cfg: PEFTConfig, sites: SiteSpace) -> dict[SiteID, float]:
    """Return a dense mask α_p^ω (un-normalized) for every site in `sites`."""
    out: dict[SiteID, float] = {}
    total_params = trainable_params_estimate(cfg)
    n_active_sites = max(1, sum(
        1
        for l in sites.layer_indices
        for m in sites.modules
        if l in cfg.target_layers and _module_targets_match(cfg, m)
    ))
    per_site_cap = total_params / n_active_sites

    for l in sites.layer_indices:
        for m in sites.modules:
            mask = 1.0 if (l in cfg.target_layers and _module_targets_match(cfg, m)) else 0.0
            if mask == 0.0:
                out[(l, m)] = 0.0
                continue
            op = operator_of(cfg.family, m)
            rho = RHO_OP.get(op, 0.5)
            cap_norm = math.log1p(per_site_cap / 4096)  # use d_model=4096 default
            out[(l, m)] = mask * rho * math.tanh(ETA * cap_norm)
    return out


def normalize_alpha(alpha_raw: dict[SiteID, float], *, eps: float = 1e-6) -> dict[SiteID, float]:
    total = sum(alpha_raw.values())
    if total <= eps:
        return {k: 0.0 for k in alpha_raw}
    return {k: v / total for k, v in alpha_raw.items()}


def alpha_vector(cfg: PEFTConfig, sites: SiteSpace, *, normalize: bool = True) -> np.ndarray:
    """Return α as a 1-D ndarray aligned with `sites.all_sites` order."""
    raw = site_mask_of(cfg, sites)
    if normalize:
        raw = normalize_alpha(raw)
    return np.asarray([raw[s] for s in sites.all_sites], dtype=np.float32)
