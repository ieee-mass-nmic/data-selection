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
  - `eval_loss`: always-available mean response-LM loss on a held-out eval set.
  - `metric`: by default `-eval_loss` (higher = better). For real task metrics
    (EM / pass@k / F1), pass a `task_metric(model, tokenizer) -> float` callable
    — e.g. a thin wrapper around lm-eval-harness. This is the documented hook;
    inline eval stays dependency-free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from pcu_select.features.stats import response_lm_loss
from pcu_select.features.tokenization import encode_response_lm
from pcu_select.hi_fidelity.native_peft import SUPPORTED_FAMILIES, attach_peft
from pcu_select.types import PEFTConfig, Sample, ValidationSketch
from pcu_select.utils import get_logger

TaskMetric = Callable[[Any, Any], float]


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

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device if torch.cuda.is_available() else "cpu")
    return model, tok


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
    dev = device if torch.cuda.is_available() else "cpu"
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
        return float("nan")
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
        order = rng.permutation(len(samples))
        model.train()
        step = 0
        cursor = 0
        while step < cfg.max_steps and len(samples) > 0:
            if cursor + cfg.batch_size > len(order):
                order = rng.permutation(len(samples))
                cursor = 0
            batch = [samples[order[cursor + j]] for j in range(cfg.batch_size)]
            cursor += cfg.batch_size
            ids, attn, rmasks = _collate(tok, batch, cfg.max_len, cfg.device)
            loss = _batch_loss(model, ids, attn, rmasks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable.params, cfg.grad_clip)
            opt.step()
            step += 1
            if step % cfg.log_every == 0:
                log.info(f"step {step}/{cfg.max_steps} loss={float(loss):.4f}")
        train_wall = time.time() - t0

        eval_loss = _eval_loss(model, tok, eval_set, cfg)
        if task_metric is not None:
            metric = float(task_metric(model, tok))
            metric_name = "task_metric"
        else:
            metric = -eval_loss
            metric_name = "neg_eval_loss"
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
