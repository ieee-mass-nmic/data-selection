"""The experiment matrix: PEFT registry, task set, model set, budgets.

This is the single source of truth for the configurations referenced by
docs/pcu_select_experiment_design.md §1.4 (PEFT registry), §1.3 (tasks),
§1.1 (models) and §1.5 (budgets). The E1–E5 runner scripts import from here so
that every experiment draws from the *same* registry and nothing drifts.

`PeftSpec` is backbone-agnostic (it stores a `layer_range` like "all"/"low"
rather than concrete indices). Call `PeftSpec.materialize(n_layers_total)` —
or the `resolve_peft(name, model)` helper — to obtain a concrete `PEFTConfig`
whose `target_layers` are filled in for a given backbone depth, and whose
`peft_id` is the content hash used everywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from pcu_select.types import PEFTConfig, PEFTFamily, PEFTRecipe
from pcu_select.utils import peft_id_of

Group = Literal["seen", "unseen_config", "ood_family"]
LayerRange = Literal["all", "low", "mid", "high"]

# Leaf-module name groups (Llama / Qwen naming). site_mask._module_targets_match
# maps these onto the hooked sites Ω.
MODULES = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["up_proj", "down_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    "ia3_attn": ["k_proj", "v_proj"],
    "ia3_attnffn": ["k_proj", "v_proj", "down_proj"],
    # Native bottleneck adapter is wrapped onto the attn + mlp output projections.
    "adapter_sites": ["o_proj", "down_proj"],
}


# ---------------------------------------------------------------------------
# PEFT spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeftSpec:
    """Backbone-agnostic PEFT description. Materializes to a `PEFTConfig`."""

    name: str
    family: PEFTFamily
    modules_key: str
    group: Group
    layer_range: LayerRange = "all"
    rank: int | None = None
    alpha: int | None = None
    adapter_bottleneck: int | None = None
    prefix_len: int | None = None
    lr: float = 2e-4
    note: str = ""

    def layer_indices(self, n_layers_total: int) -> list[int]:
        if self.layer_range == "all":
            return list(range(n_layers_total))
        third = max(1, n_layers_total // 3)
        if self.layer_range == "low":
            return list(range(0, third))
        if self.layer_range == "mid":
            return list(range(third, 2 * third))
        return list(range(n_layers_total - third, n_layers_total))  # high

    def materialize(self, n_layers_total: int = 32) -> PEFTConfig:
        recipe = PEFTRecipe(lr=self.lr, max_steps=1000, warmup_ratio=0.03, scheduler="cosine")
        cfg = PEFTConfig(
            peft_id="",  # filled from content hash below
            family=self.family,
            target_modules=list(MODULES[self.modules_key]),
            target_layers=self.layer_indices(n_layers_total),
            rank=self.rank,
            alpha=self.alpha,
            adapter_bottleneck=self.adapter_bottleneck,
            prefix_len=self.prefix_len,
            recipe=recipe,
            extra={"spec_name": self.name, "group": self.group},
        )
        payload = asdict(cfg)
        payload.pop("peft_id")
        cfg.peft_id = peft_id_of(payload)
        return cfg


# ---------------------------------------------------------------------------
# The registry (design §1.4). `★` rows in the doc are group="seen".
# ---------------------------------------------------------------------------

PEFT_REGISTRY: dict[str, PeftSpec] = {
    # ---- A. SEEN: scorer trains on these ----
    "L-r8-qv": PeftSpec("L-r8-qv", "lora", "qv", "seen", rank=8, alpha=16, lr=2e-4,
                        note="LoRA baseline"),
    "L-r16-qkvo": PeftSpec("L-r16-qkvo", "lora", "qkvo", "seen", rank=16, alpha=32, lr=2e-4,
                           note="wider attn"),
    "L-r8-mlp": PeftSpec("L-r8-mlp", "lora", "mlp", "seen", rank=8, alpha=16, lr=2e-4,
                         note="MLP-only"),
    "IA3-attnmlp": PeftSpec("IA3-attnmlp", "ia3", "ia3_attnffn", "seen", lr=5e-4,
                            note="IA3 standard"),
    "AD-b64": PeftSpec("AD-b64", "adapter", "adapter_sites", "seen", adapter_bottleneck=64,
                       lr=3e-4, note="Houlsby adapter"),
    # ---- B. UNSEEN-config: same family, configuration not in training support ----
    "L-r4-qv": PeftSpec("L-r4-qv", "lora", "qv", "unseen_config", rank=4, alpha=8, lr=2e-4,
                        note="tiny capacity"),
    "L-r32-qkvo": PeftSpec("L-r32-qkvo", "lora", "qkvo", "unseen_config", rank=32, alpha=64,
                           lr=2e-4, note="large capacity, ID-interpolation"),
    "L-r64-all": PeftSpec("L-r64-all", "lora", "all_linear", "unseen_config", rank=64, alpha=128,
                          lr=1e-4, note="capacity + placement extrapolation"),
    "L-r8-lowlayers": PeftSpec("L-r8-lowlayers", "lora", "qv", "unseen_config", layer_range="low",
                               rank=8, alpha=16, lr=2e-4, note="placement shift (low)"),
    "L-r8-highlayers": PeftSpec("L-r8-highlayers", "lora", "qv", "unseen_config",
                                layer_range="high", rank=8, alpha=16, lr=2e-4,
                                note="placement shift (high)"),
    "L-r16-hlr": PeftSpec("L-r16-hlr", "lora", "qkvo", "unseen_config", rank=16, alpha=32,
                          lr=5e-4, note="recipe (lr) shift"),
    "AD-b16": PeftSpec("AD-b16", "adapter", "adapter_sites", "unseen_config",
                       adapter_bottleneck=16, lr=3e-4, note="adapter tiny"),
    "AD-b256": PeftSpec("AD-b256", "adapter", "adapter_sites", "unseen_config",
                        adapter_bottleneck=256, lr=3e-4, note="adapter large, extrapolation"),
    "IA3-attnonly": PeftSpec("IA3-attnonly", "ia3", "ia3_attn", "unseen_config", lr=5e-4,
                             note="placement shift"),
    # ---- C. OOD-family: family never seen during scorer training ----
    "PRE-l16": PeftSpec("PRE-l16", "prefix", "qkvo", "ood_family", prefix_len=16,
                        note="prefix tuning"),
    "PT-l32": PeftSpec("PT-l32", "ptuning", "qkvo", "ood_family", prefix_len=32,
                       note="P-Tuning v2"),
    "BF": PeftSpec("BF", "bitfit", "qkvo", "ood_family", note="bias-only"),
}


def peft_specs_by_group(group: Group) -> list[PeftSpec]:
    return [s for s in PEFT_REGISTRY.values() if s.group == group]


# ---------------------------------------------------------------------------
# Tasks (design §1.3). The four "main" tasks run across E1–E5.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    name: str
    capability: str
    metric: str  # primary downstream metric tag
    is_main: bool = False


TASKS: dict[str, TaskSpec] = {
    "gsm8k": TaskSpec("gsm8k", "math", "exact_match", is_main=True),
    "humaneval": TaskSpec("humaneval", "code", "pass@1", is_main=True),
    "mmlu": TaskSpec("mmlu", "knowledge", "accuracy", is_main=True),
    "tydiqa": TaskSpec("tydiqa", "multilingual", "f1", is_main=True),
    "math": TaskSpec("math", "hard_math", "accuracy"),
    "mbpp": TaskSpec("mbpp", "code", "pass@1"),
    "alpacaeval": TaskSpec("alpacaeval", "instruction", "lc_winrate"),
    "safety": TaskSpec("safety", "safety", "refusal_acc"),
}

MAIN_TASKS = [t.name for t in TASKS.values() if t.is_main]


# ---------------------------------------------------------------------------
# Models — all ≥ 7B (design §1.1). selector == smallest backbone in the family.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str  # short tag used in result rows
    hf_id: str
    n_layers: int
    family_tag: str  # "A" (Llama) / "B" (Qwen)
    selector_hf_id: str  # the selector model used to build features for this family
    role: Literal["selector", "backbone"] = "backbone"


MODELS: dict[str, ModelSpec] = {
    # Family A — Llama-2 (main results)
    "llama2-7b": ModelSpec("llama2-7b", "meta-llama/Llama-2-7b-hf", 32, "A",
                           "meta-llama/Llama-2-7b-hf", role="selector"),
    "llama2-13b": ModelSpec("llama2-13b", "meta-llama/Llama-2-13b-hf", 40, "A",
                            "meta-llama/Llama-2-7b-hf"),
    # Family B — Qwen2.5 (robustness replication)
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", "Qwen/Qwen2.5-7B", 28, "B",
                            "Qwen/Qwen2.5-7B", role="selector"),
    "qwen2.5-14b": ModelSpec("qwen2.5-14b", "Qwen/Qwen2.5-14B", 48, "B",
                             "Qwen/Qwen2.5-7B"),
}

DEFAULT_MODEL = "llama2-7b"


# ---------------------------------------------------------------------------
# Budgets (design §1.5)
# ---------------------------------------------------------------------------

BUDGETS: list[float] = [0.05, 0.10, 0.30]
DEFAULT_BUDGET = 0.10


# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------


def resolve_peft(name: str, model: str = DEFAULT_MODEL) -> PEFTConfig:
    """Materialize a registry PEFT for a concrete backbone depth."""
    if name not in PEFT_REGISTRY:
        raise KeyError(f"unknown PEFT spec {name!r}; known: {sorted(PEFT_REGISTRY)}")
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; known: {sorted(MODELS)}")
    return PEFT_REGISTRY[name].materialize(MODELS[model].n_layers)
