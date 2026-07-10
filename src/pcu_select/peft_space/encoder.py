"""PEFT condition vector z_p construction. See design doc §8."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from pcu_select.peft_space.schema import trainable_params_estimate
from pcu_select.peft_space.site_mask import SiteSpace, operator_of, site_mask_of
from pcu_select.types import PEFTConfig

D_CAP = 16
D_REC = 16
SCHEDULER_VOCAB = ("cosine", "linear", "constant", "constant_with_warmup")
OPTIM_VOCAB = ("adamw", "sgd", "adafactor")
INIT_VOCAB = ("kaiming", "zero", "default")


def encode_site_mask(cfg: PEFTConfig, sites: SiteSpace) -> np.ndarray:
    """4 indicators per site: [is_active, additive, multiplicative, prefix]."""
    raw = site_mask_of(cfg, sites)
    n = len(sites.all_sites)
    out = np.zeros((n, 4), dtype=np.float32)
    for i, s in enumerate(sites.all_sites):
        active = 1.0 if raw[s] > 0 else 0.0
        op = operator_of(cfg.family, s[1])
        flags = (
            float(op in ("additive_low_rank", "additive_bottleneck")),
            float(op == "multiplicative"),
            float(op == "prefix"),
        )
        out[i] = (active, *(active * f for f in flags))
    return out.reshape(-1)


def encode_capacity(cfg: PEFTConfig, *, d_model: int = 4096) -> np.ndarray:
    v = np.zeros(D_CAP, dtype=np.float32)
    total = trainable_params_estimate(cfg, d_model=d_model)
    v[0] = math.log1p(total)
    v[1] = total / (1e6)  # ratio surrogate
    v[2] = (cfg.rank or 0) / 64.0
    v[3] = (cfg.alpha or 0) / 64.0
    v[4] = (cfg.adapter_bottleneck or 0) / 256.0
    v[5] = (cfg.prefix_len or 0) / 64.0
    v[6] = math.log1p(total)  # extra FLOPs proxy
    v[7] = math.log1p(total * 4 / (1024 * 1024))  # extra mem MB proxy
    v[8] = 1.0 if cfg.family in ("prefix", "ptuning") else 0.0  # affects KV cache
    # per_op share (5 dims): mostly one-hot on family
    fam_idx = {
        "lora": 9,
        "ia3": 10,
        "adapter": 11,
        "prefix": 12,
        "ptuning": 12,
        "bitfit": 13,
    }.get(cfg.family, 9)
    v[fam_idx] = 1.0
    v[14] = len(cfg.target_layers) / 32.0
    v[15] = len(cfg.target_modules) / 8.0
    return v


def encode_recipe(cfg: PEFTConfig) -> np.ndarray:
    v = np.zeros(D_REC, dtype=np.float32)
    r = cfg.recipe
    v[0] = math.log10(max(r.lr, 1e-8))
    v[1] = r.warmup_ratio
    v[2] = r.weight_decay
    sch = SCHEDULER_VOCAB.index(r.scheduler) if r.scheduler in SCHEDULER_VOCAB else 0
    v[3 + sch] = 1.0  # scheduler one-hot occupies slots 3..6 (4 slots)
    opt = OPTIM_VOCAB.index(r.optimizer) if r.optimizer in OPTIM_VOCAB else 0
    v[7 + opt] = 1.0  # optimizer one-hot occupies slots 7..9 (3 slots)
    v[10] = math.log2(max(r.batch_size, 1))
    v[11] = r.dropout
    v[12] = math.log10(max(r.max_steps, 1))
    v[13] = r.grad_clip
    # init one-hot has 2 reserved slots (14..15); "default" (idx 2 in INIT_VOCAB)
    # is represented as all-zeros to keep within D_REC budget.
    init = INIT_VOCAB.index(r.init_method) if r.init_method in INIT_VOCAB else 2
    if init < 2:
        v[14 + init] = 1.0
    return v


def encode_peft(
    cfg: PEFTConfig,
    sites: SiteSpace,
    *,
    fingerprint: np.ndarray | None = None,
) -> np.ndarray:
    """Concat [m_p; c_p; r_p] and append an optional functional fingerprint."""
    m_p = encode_site_mask(cfg, sites)
    c_p = encode_capacity(cfg)
    r_p = encode_recipe(cfg)
    parts: list[np.ndarray] = [m_p, c_p, r_p]
    if fingerprint is not None:
        parts.append(fingerprint.astype(np.float32))
    return np.concatenate(parts, axis=-1)


def stack_z_p(cfgs: Sequence[PEFTConfig], sites: SiteSpace,
              fingerprints: Sequence[np.ndarray | None] | None = None) -> np.ndarray:
    fps = fingerprints if fingerprints is not None else [None] * len(cfgs)
    rows = [encode_peft(c, sites, fingerprint=fp) for c, fp in zip(cfgs, fps)]
    return np.stack(rows, axis=0)
