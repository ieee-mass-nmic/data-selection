"""Native (peft-library-free) PEFT injection for the short-update protocol.

The high-fidelity labeler needs to attach trainable PEFT parameters on top of a
frozen anchor model, run a few optimizer steps, then detach them cleanly. We
implement this directly in torch by wrapping the target `nn.Linear` modules,
rather than depending on the `peft` library, because:

  - it keeps the short-update path fully controllable and unit-testable with a
    tiny in-memory model (no checkpoint download, no library version churn);
  - all the families we need (`lora`, `ia3`, `adapter`, `bitfit`) are clean
    `nn.Linear` wrappers; and
  - the base weights stay frozen, so backward only touches the small adapter,
    which is exactly the cost saving the design relies on (§10.2).

`prefix` / `ptuning` are *not* expressible as `nn.Linear` wrappers — they need
KV-cache / input-embedding injection — so this backend raises a clear error for
them; that support is deferred.

See design doc §1, §8 (PEFT families) and §10.2 (short-update protocol).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from pcu_select.types import PEFTConfig

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# Families this native backend can attach. prefix/ptuning are deferred.
SUPPORTED_FAMILIES = ("lora", "ia3", "adapter", "bitfit")


# ---------------------------------------------------------------------------
# Wrapper modules. Each holds a frozen `base` Linear plus a small trainable
# adapter whose parameters are registered under predictable names so that
# `PeftHandle.state_dict()` round-trips.
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """y = base(x) + scaling · (dropout(x) Aᵀ) Bᵀ.   B is zero-init → Δ₀ = 0."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + delta * self.scaling


class IA3Linear(nn.Module):
    """y = base(x) ⊙ s.   s is ones-init → Δ₀ = 0 (multiplicative, design §3)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        dev, dt = base.weight.device, base.weight.dtype
        self.ia3 = nn.Parameter(torch.ones(base.out_features, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) * self.ia3


class BitFitLinear(nn.Module):
    """y = base(x) + b.   b is zero-init → Δ₀ = 0 (bias shift, design §3)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        dev, dt = base.weight.device, base.weight.dtype
        self.delta_bias = nn.Parameter(torch.zeros(base.out_features, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.delta_bias


class AdapterLinear(nn.Module):
    """y = base(x) + up(relu(down(base(x)))).   up is zero-init → Δ₀ = 0."""

    def __init__(self, base: nn.Linear, *, bottleneck: int):
        super().__init__()
        self.base = base
        dev, dt = base.weight.device, base.weight.dtype
        out = base.out_features
        self.down = nn.Linear(out, bottleneck, device=dev, dtype=dt)
        self.up = nn.Linear(bottleneck, out, device=dev, dtype=dt)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.up(torch.relu(self.down(y)))


_WRAPPER_TYPES = (LoRALinear, IA3Linear, BitFitLinear, AdapterLinear)


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------


@dataclass
class PeftHandle:
    """Bookkeeping for an attached PEFT, supporting clean removal + I/O."""

    model: nn.Module
    # (parent_path, leaf_name, original_module, wrapper)
    _sites: list[tuple[str, str, nn.Module, nn.Module]] = field(default_factory=list)
    _removed: bool = False

    def parameters(self) -> list[nn.Parameter]:
        """All trainable adapter parameters (excludes the frozen base)."""
        params: list[nn.Parameter] = []
        for _, _, base, wrapper in self._sites:
            base_ids = {id(p) for p in base.parameters()}
            params.extend(p for p in wrapper.parameters() if id(p) not in base_ids)
        return params

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Adapter-only state, keyed by `<parent>.<leaf>.<param>`."""
        sd: dict[str, torch.Tensor] = {}
        for parent, leaf, base, wrapper in self._sites:
            base_ids = {id(p) for p in base.parameters()}
            prefix = f"{parent}.{leaf}" if parent else leaf
            for name, p in wrapper.named_parameters():
                if id(p) not in base_ids:
                    sd[f"{prefix}.{name}"] = p.detach().cpu().clone()
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        for parent, leaf, base, wrapper in self._sites:
            base_ids = {id(p) for p in base.parameters()}
            prefix = f"{parent}.{leaf}" if parent else leaf
            for name, p in wrapper.named_parameters():
                key = f"{prefix}.{name}"
                if id(p) not in base_ids and key in sd:
                    with torch.no_grad():
                        p.copy_(sd[key].to(p.device, p.dtype))

    def remove(self) -> None:
        """Restore the original modules in place. Idempotent."""
        if self._removed:
            return
        for parent, leaf, base, _ in self._sites:
            parent_mod = self.model.get_submodule(parent) if parent else self.model
            setattr(parent_mod, leaf, base)
        self._removed = True


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------


def _layer_of(name: str) -> int | None:
    m = _LAYER_RE.search("." + name + ".")
    return int(m.group(1)) if m else None


def _find_target_linears(model: nn.Module, peft: PEFTConfig) -> list[tuple[str, str, nn.Linear]]:
    """Locate `nn.Linear` modules whose leaf name and layer index match `peft`.

    An empty `target_layers` means "all layers". Returns (parent_path, leaf,
    module) triples in stable named-module order.
    """
    wanted_mods = set(peft.target_modules)
    wanted_layers = set(peft.target_layers)
    out: list[tuple[str, str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in wanted_mods:
            continue
        layer = _layer_of(name)
        if wanted_layers and (layer is None or layer not in wanted_layers):
            continue
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        out.append((parent, leaf, mod))
    return out


def _make_wrapper(family: str, base: nn.Linear, peft: PEFTConfig) -> nn.Module:
    if family == "lora":
        rank = peft.rank or 8
        alpha = float(peft.alpha if peft.alpha is not None else rank)
        return LoRALinear(base, rank=rank, alpha=alpha, dropout=peft.recipe.dropout)
    if family == "ia3":
        return IA3Linear(base)
    if family == "bitfit":
        return BitFitLinear(base)
    if family == "adapter":
        return AdapterLinear(base, bottleneck=peft.adapter_bottleneck or 64)
    raise ValueError(f"unhandled family {family!r}")


def attach_peft(model: nn.Module, peft: PEFTConfig, *, seed: int = 0) -> PeftHandle:
    """Wrap matching target Linears with `peft`'s adapter; return a handle.

    The adapters are zero/identity-initialized so the model output is unchanged
    at attach time (Δ measured against the anchor is exactly 0 before any step).
    """
    if peft.family not in SUPPORTED_FAMILIES:
        raise NotImplementedError(
            f"native PEFT backend supports {SUPPORTED_FAMILIES}; family "
            f"{peft.family!r} (prefix/ptuning) needs prompt injection — deferred."
        )
    torch.manual_seed(seed)
    targets = _find_target_linears(model, peft)
    if not targets:
        raise ValueError(
            f"PEFT {peft.peft_id!r} matched no Linear modules "
            f"(target_modules={peft.target_modules}, target_layers={peft.target_layers})."
        )
    handle = PeftHandle(model=model)
    for parent, leaf, base in targets:
        wrapper = _make_wrapper(peft.family, base, peft)
        parent_mod = model.get_submodule(parent) if parent else model
        setattr(parent_mod, leaf, wrapper)
        handle._sites.append((parent, leaf, base, wrapper))
    return handle
