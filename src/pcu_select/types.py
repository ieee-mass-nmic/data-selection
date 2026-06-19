"""Global data contracts for PCU-Select.

All dataclasses here are framework-agnostic (no torch dependency on the
fields), so they can be safely serialized to disk and shared between
offline / apply pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------

SampleID = str
PEFTID = str
TaskID = str
SiteID = tuple[int, str]  # (layer_idx, module_name)

ModuleName = Literal["attn_out", "mlp_out", "block_residual"]
OperatorType = Literal[
    "additive_low_rank",
    "multiplicative",
    "additive_bottleneck",
    "prefix",
    "bias_shift",
]
LossType = Literal["response_lm", "full_lm", "preference"]


# ---------------------------------------------------------------------------
# Sample-side
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    sample_id: SampleID
    instruction: str
    response: str
    source: str | None = None
    language: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class SemanticEmbedding:
    instr: np.ndarray  # (d_sem,)
    resp: np.ndarray
    joint: np.ndarray
    source_onehot: np.ndarray | None = None


@dataclass
class DifficultyStats:
    """Fixed 16-d vector. See design doc §7.2."""

    vector: np.ndarray  # (16,)


@dataclass
class ActivationSignature:
    """Per-layer × per-stat compact descriptor. See design doc §7.3."""

    vector: np.ndarray  # (n_layers_sig * d_layer_stat,)


@dataclass
class SampleFeatures:
    sample_id: SampleID
    e_x: SemanticEmbedding
    d_x: DifficultyStats
    a_x: ActivationSignature

    def as_z_x(self) -> np.ndarray:
        """Concatenate the three components into the scorer input vector."""
        return np.concatenate([self.e_x.joint, self.d_x.vector, self.a_x.vector], axis=-1)


# ---------------------------------------------------------------------------
# PEFT-side
# ---------------------------------------------------------------------------


@dataclass
class PEFTRecipe:
    optimizer: Literal["adamw", "sgd", "adafactor"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    scheduler: Literal["cosine", "linear", "constant", "constant_with_warmup"] = "cosine"
    batch_size: int = 16
    dropout: float = 0.0
    max_steps: int = 1000
    grad_clip: float = 1.0
    init_method: Literal["kaiming", "zero", "default"] = "default"


@dataclass
class PEFTConfig:
    """Structured PEFT description. See design doc §1, §8."""

    peft_id: PEFTID
    family: Literal["lora", "ia3", "adapter", "prefix", "bitfit", "ptuning"]
    target_modules: list[str]  # e.g. ["q_proj", "v_proj"]
    target_layers: list[int]  # which transformer layers
    rank: int | None = None
    alpha: int | None = None
    adapter_bottleneck: int | None = None
    prefix_len: int | None = None
    recipe: PEFTRecipe = field(default_factory=PEFTRecipe)
    use_fingerprint: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task-side
# ---------------------------------------------------------------------------


@dataclass
class ValidationSketch:
    task_id: TaskID
    samples: list[Sample]
    sketch_seed: int


@dataclass
class TaskConfig:
    name: str
    task_id: TaskID
    sketch: ValidationSketch
    loss_type: LossType = "response_lm"
    eval_protocol: str = "exact_match"  # free-form tag for downstream eval


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@dataclass
class LoFidelityLabel:
    sample_id: SampleID
    peft_id: PEFTID
    task_id: TaskID
    u_lo: float


@dataclass
class HiFidelityLabel:
    sample_id: SampleID
    peft_id: PEFTID
    task_id: TaskID
    u_hi: float
    horizon: int
    anchor_idx: int
    seed: int
    delta_raw: float  # before RankNorm, for diagnostics
    sigma_est: float | None = None


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class OfflineConfig:
    selector_model: str = "meta-llama/Llama-2-7b-hf"
    # Backbone trained in the high-fidelity short-update protocol. Defaults to
    # `selector_model` (selector == smallest backbone in the family) when None.
    backbone_model: str | None = None
    n_layers_total: int = 32  # selector model depth; must match the backbone
    n_layers_signature: int = 8
    site_modules: tuple[ModuleName, ...] = ("attn_out", "mlp_out", "block_residual")
    d_proj: int = 256
    device: str = "cuda"
    max_len: int = 1024
    global_seed: int = 0
    horizons: tuple[int, ...] = (1, 4)
    horizon_weights: tuple[float, ...] = (0.4, 0.6)
    anchors: int = 2
    anchor_warm_steps: int = 200
    anchor_warm_slice: int = 1000
    anchor_lora_rank: int = 8
    q_hi_total: int = 10_000
    phase_split: tuple[float, float, float] = (0.5, 0.3, 0.2)
    scorer_epochs_phase_a: int = 3
    scorer_epochs_phase_b: int = 2
    scorer_lr_phase_a: float = 3e-4
    scorer_lr_phase_b: float = 1e-4
    scorer_batch_size: int = 256
    loss_weights: tuple[float, float, float, float] = (1.0, 0.3, 0.5, 0.2)  # rank,reg,proxy,unc


@dataclass
class ApplyConfig:
    lambda_unc: float = 0.2
    cluster_alpha: float = 0.6
    cluster_k: int | None = None  # default: max(50, sqrt(N))
    min_cluster_size: int | None = None
    enable_calibration: bool = True
    ood_quantile: float = 0.95


# ---------------------------------------------------------------------------
# Dataset protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetLike(Protocol):
    """Minimal contract for candidate / meta pools."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Sample]: ...
    def take(self, ids: Iterable[SampleID]) -> list[Sample]: ...


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


@dataclass
class WorkDirLayout:
    """Resolved sub-paths inside `runs/<exp_id>/` (see design doc §14.1)."""

    root: Path

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def task(self) -> Path:
        return self.root / "task"

    @property
    def peft(self) -> Path:
        return self.root / "peft"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def scorer(self) -> Path:
        return self.root / "scorer"

    @property
    def selection(self) -> Path:
        return self.root / "selection"

    @property
    def cost(self) -> Path:
        return self.root / "cost"
