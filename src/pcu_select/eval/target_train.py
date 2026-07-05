"""Train a PEFT on a selected subset and evaluate it (design §1.7, §7).

This is the "target training" half of every performance experiment: given the
`SampleID`s a method selected, fine-tune the *target* PEFT on those samples (and
nothing else) under a fixed recipe, then measure downstream quality. Because the
recipe, step budget and eval are identical across methods, the only variable is
the subset — exactly the compute-matched comparison the design demands.

Backends:
  - native (lora / ia3 / adapter / bitfit): reuse `hi_fidelity.native_peft`,
    which wraps the target `nn.Linear`s and trains only the adapter. A preloaded
    backbone can be reused across many calls (`detach` restores it in place).
  - peft-library (prefix / ptuning): built with the `peft` package on a freshly
    loaded backbone (these need prompt/KV injection the native backend defers).

Downstream metric:
  - `eval_loss`: mean response-LM loss on a held-out eval set, logged as an
    auxiliary diagnostic.
  - `metric`: the task-native score (EM / pass@k / F1 / accuracy / judge score).
    Callers must pass a `task_metric(model, tokenizer)` callback, typically a
    thin wrapper around the task's evaluator. `eval_loss` is never used as a
    replacement for the primary metric.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import numpy as np
import torch

from pcu_select.features.stats import response_lm_loss
from pcu_select.features.tokenization import encode_response_lm
from pcu_select.hi_fidelity.native_peft import SUPPORTED_FAMILIES, attach_peft
from pcu_select.types import PEFTConfig, PEFTRecipe, Sample, ValidationSketch
from pcu_select.utils import get_logger

TaskMetricValue = float | tuple[str, float]
TaskMetric = Callable[[Any, Any], TaskMetricValue]


@dataclass
class TargetTrainConfig:
    backbone_model: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_len: int = 1024
    batch_size: int = 16
    max_steps: int = 1000
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 100


@dataclass
class TargetTrainResult:
    eval_loss: float
    metric: float
    metric_name: str
    n_train: int
    train_wall_sec: float
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model loading (kept here so callers can preload once and reuse for native).
# ---------------------------------------------------------------------------


def load_backbone(model_id: str, *, device: str = "cuda", dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = _resolve_device(device)
    tokenizer_cls = cast(Any, AutoTokenizer)
    model_cls = cast(Any, AutoModelForCausalLM)
    tok = tokenizer_cls.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = getattr(torch, dtype) if resolved_device.startswith("cuda") else torch.float32
    model = model_cls.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(resolved_device)
    return model, tok


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"target training requested device={device!r}, but CUDA is not available. "
            "Pass --device cpu for a CPU run."
        )
    return device


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def _collate(tok: Any, batch: list[Sample], max_len: int, device: str):
    encs = [encode_response_lm(tok, s.instruction, s.response, max_len=max_len) for s in batch]
    width = max(len(e["input_ids"]) for e in encs)
    pad_id = tok.pad_token_id or 0
    ids = torch.full((len(encs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(encs), width), dtype=torch.long)
    rmasks: list[torch.Tensor] = []
    for i, e in enumerate(encs):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        attn[i, :n] = torch.tensor(e["attention_mask"])
        rmasks.append(torch.tensor(e["response_mask"]))
    dev = _resolve_device(device)
    return ids.to(dev), attn.to(dev), rmasks


def _batch_loss(model: Any, ids, attn, rmasks) -> torch.Tensor:
    out = model(input_ids=ids, attention_mask=attn)
    losses = []
    for b, rmask in enumerate(rmasks):
        n = len(rmask)
        losses.append(response_lm_loss(out.logits[b, :n], ids[b, :n], rmask.to(ids.device)))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Trainable PEFT attachment (native vs peft-library)
# ---------------------------------------------------------------------------


class _Trainable:
    """Uniform handle over native / peft-library trainable parameters."""

    def __init__(self, model: Any, params: list[torch.nn.Parameter], detach: Callable[[], None]):
        self.model = model
        self.params = params
        self._detach = detach

    def detach(self) -> None:
        self._detach()


def _attach_native(model: Any, peft: PEFTConfig, seed: int) -> _Trainable:
    for p in model.parameters():
        p.requires_grad_(False)
    handle = attach_peft(model, peft, seed=seed)
    for p in handle.parameters():
        p.requires_grad_(True)
    return _Trainable(model, handle.parameters(), handle.remove)


def _attach_peft_library(peft: PEFTConfig, cfg: TargetTrainConfig) -> _Trainable:
    """Fresh-load backbone + wrap with the peft library (prefix / ptuning)."""
    from peft import PrefixTuningConfig, PromptEncoderConfig, get_peft_model

    model, tok = load_backbone(cfg.backbone_model, device=cfg.device, dtype=cfg.dtype)
    if peft.family == "prefix":
        pc = PrefixTuningConfig(task_type="CAUSAL_LM",
                                num_virtual_tokens=peft.prefix_len or 16)
    else:  # ptuning
        pc = PromptEncoderConfig(task_type="CAUSAL_LM",
                                 num_virtual_tokens=peft.prefix_len or 32)
    pmodel = get_peft_model(model, pc)
    params = [p for p in pmodel.parameters() if p.requires_grad]
    trainable = _Trainable(pmodel, params, lambda: None)
    trainable.extra_tokenizer = tok  # type: ignore[attr-defined]
    return trainable


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_loss(model: Any, tok: Any, sketch: ValidationSketch, cfg: TargetTrainConfig) -> float:
    if not sketch.samples:
        raise ValueError("target evaluation requires a non-empty held-out eval set")
    model.eval()
    total = 0.0
    for s in sketch.samples:
        ids, attn, rmasks = _collate(tok, [s], cfg.max_len, cfg.device)
        total += float(_batch_loss(model, ids, attn, rmasks))
    return total / len(sketch.samples)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def train_and_eval(
    *,
    samples: list[Sample],
    peft: PEFTConfig,
    eval_set: ValidationSketch,
    cfg: TargetTrainConfig,
    model: Any | None = None,
    tokenizer: Any | None = None,
    task_metric: TaskMetric | None = None,
) -> TargetTrainResult:
    """Fine-tune `peft` on `samples`, evaluate on `eval_set`.

    `model`/`tokenizer` may be preloaded and reused for native families (the
    adapter is detached in `finally`, restoring the backbone). prefix/ptuning
    always load a fresh backbone internally.
    """
    if not samples:
        raise ValueError("target training requires at least one selected sample")
    if not eval_set.samples:
        raise ValueError("target evaluation requires a non-empty held-out eval set")
    if task_metric is None:
        raise ValueError(
            "target training requires task_metric; eval_loss is logged as an auxiliary "
            "diagnostic and is not used as the primary downstream metric"
        )

    log = get_logger("eval.target_train")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    use_native = peft.family in SUPPORTED_FAMILIES
    if use_native:
        if model is None or tokenizer is None:
            model, tokenizer = load_backbone(cfg.backbone_model, device=cfg.device, dtype=cfg.dtype)
        trainable = _attach_native(model, peft, cfg.seed)
        tok = tokenizer
    else:
        trainable = _attach_peft_library(peft, cfg)
        model = trainable.model
        tok = trainable.extra_tokenizer  # type: ignore[attr-defined]

    t0 = time.time()
    try:
        opt = _build_optimizer(trainable.params, peft)
        scheduler = _build_scheduler(opt, peft.recipe, cfg.max_steps)
        order = rng.permutation(len(samples))
        model.train()
        step = 0
        cursor = 0
        while step < cfg.max_steps and len(samples) > 0:
            batch_size = max(1, min(cfg.batch_size, len(samples)))
            if cursor + batch_size > len(order):
                order = rng.permutation(len(samples))
                cursor = 0
            batch = [samples[order[cursor + j]] for j in range(batch_size)]
            cursor += batch_size
            ids, attn, rmasks = _collate(tok, batch, cfg.max_len, cfg.device)
            loss = _batch_loss(model, ids, attn, rmasks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable.params, cfg.grad_clip)
            opt.step()
            scheduler.step()
            step += 1
            if step % cfg.log_every == 0:
                log.info(f"step {step}/{cfg.max_steps} loss={float(loss):.4f}")
        train_wall = time.time() - t0

        eval_loss = _eval_loss(model, tok, eval_set, cfg)
        metric_value = task_metric(model, tok)
        if isinstance(metric_value, tuple):
            metric_name, raw_metric = metric_value
            metric = float(raw_metric)
        else:
            metric_name = str(getattr(task_metric, "metric_name", "task_metric"))
            metric = float(metric_value)
    finally:
        trainable.detach()

    return TargetTrainResult(
        eval_loss=eval_loss,
        metric=metric,
        metric_name=metric_name,
        n_train=len(samples),
        train_wall_sec=train_wall,
        extra={"family": peft.family, "backend": "native" if use_native else "peft_lib"},
    )


def _build_optimizer(params: list[torch.nn.Parameter], peft: PEFTConfig) -> torch.optim.Optimizer:
    r = peft.recipe
    if r.optimizer == "sgd":
        return torch.optim.SGD(params, lr=r.lr, weight_decay=r.weight_decay)
    return torch.optim.AdamW(params, lr=r.lr, weight_decay=r.weight_decay)


def _build_scheduler(opt: torch.optim.Optimizer, recipe: PEFTRecipe, num_training_steps: int) -> Any:
    """LR schedule with warmup, per the PEFT recipe (design §8.3 encodes both into r_p).

    `recipe.scheduler` ∈ {cosine, linear, constant, constant_with_warmup} maps
    directly onto `transformers.get_scheduler`; `warmup_ratio` → warmup steps.
    Keeping this on the recipe (not a fixed global) means every method trains the
    target PEFT under the *same* schedule, preserving the compute-matched protocol.
    """
    from transformers import get_scheduler

    num_warmup = max(0, int(recipe.warmup_ratio * num_training_steps))
    return get_scheduler(
        recipe.scheduler,
        optimizer=opt,
        num_warmup_steps=num_warmup,
        num_training_steps=max(1, num_training_steps),
    )
