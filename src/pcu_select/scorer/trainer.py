"""Two-phase scorer trainer (low-fidelity pretrain → joint finetune).

Design doc §11.4.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pcu_select.scorer.losses import combine_losses
from pcu_select.scorer.model import PCUScorer
from pcu_select.utils import get_logger


@dataclass
class TrainerConfig:
    epochs_phase_a: int = 3
    epochs_phase_b: int = 2
    lr_phase_a: float = 3e-4
    lr_phase_b: float = 1e-4
    batch_size: int = 256
    pair_per_bucket: int = 32
    weights_phase_a: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0)
    weights_phase_b: tuple[float, float, float, float] = (1.0, 0.3, 0.5, 0.2)
    device: str = "cuda"


class TripletDataset(Dataset):
    """Wraps rows from labels parquet plus z_x / z_p / z_t lookup tables.

    Each row exposes: (z_x, z_p, z_t, u_lo, u_hi or NaN, bucket_id).
    `bucket_id` is hash(peft_id || task_id), used for pairwise sampling.
    """

    def __init__(
        self,
        *,
        rows: list[dict],
        z_x_by_id: dict[str, np.ndarray],
        z_p_by_id: dict[str, np.ndarray],
        z_t_by_id: dict[str, np.ndarray],
    ):
        self.rows = rows
        self.z_x_by_id = z_x_by_id
        self.z_p_by_id = z_p_by_id
        self.z_t_by_id = z_t_by_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return {
            "z_x": self.z_x_by_id[r["sample_id"]].astype(np.float32),
            "z_p": self.z_p_by_id[r["peft_id"]].astype(np.float32),
            "z_t": self.z_t_by_id[r["task_id"]].astype(np.float32),
            "u_lo": np.float32(r.get("u_lo", np.nan)),
            "u_hi": np.float32(r.get("u_hi", np.nan)),
            "bucket": r["peft_id"] + "::" + r["task_id"],
        }


def _collate(batch):
    out = {k: None for k in batch[0]}
    out["z_x"] = torch.from_numpy(np.stack([b["z_x"] for b in batch]))
    out["z_p"] = torch.from_numpy(np.stack([b["z_p"] for b in batch]))
    out["z_t"] = torch.from_numpy(np.stack([b["z_t"] for b in batch]))
    out["u_lo"] = torch.from_numpy(np.stack([b["u_lo"] for b in batch]))
    out["u_hi"] = torch.from_numpy(np.stack([b["u_hi"] for b in batch]))
    out["bucket"] = [b["bucket"] for b in batch]
    return out


def sample_rank_pairs(buckets: list[str], u_hi: torch.Tensor, k: int) -> list[tuple[int, int]]:
    """Within each bucket, sample up to k pairs (i, j) with both u_hi != NaN."""
    by_bucket: dict[str, list[int]] = {}
    for i, b in enumerate(buckets):
        if not torch.isnan(u_hi[i]):
            by_bucket.setdefault(b, []).append(i)
    pairs: list[tuple[int, int]] = []
    for idxs in by_bucket.values():
        if len(idxs) < 2:
            continue
        for _ in range(min(k, len(idxs) * (len(idxs) - 1) // 2)):
            i, j = random.sample(idxs, 2)
            pairs.append((i, j))
    return pairs


def train_scorer(
    *,
    model: PCUScorer,
    phase_a_loader: DataLoader,
    phase_b_loader: DataLoader,
    cfg: TrainerConfig,
    ckpt_dir: Path,
) -> Path:
    log = get_logger("scorer.trainer")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    def _run_phase(loader: DataLoader, *, epochs: int, lr: float, weights, tag: str) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            for step, batch in enumerate(loader):
                z_x = batch["z_x"].to(device)
                z_p = batch["z_p"].to(device)
                z_t = batch["z_t"].to(device)
                u_lo = batch["u_lo"].to(device)
                u_hi = batch["u_hi"].to(device)
                mu, sigma = model(z_x, z_p, z_t)
                rank_pairs = None
                u_hi_valid = ~torch.isnan(u_hi)
                if tag == "B" and u_hi_valid.any():
                    pair_idxs = sample_rank_pairs(batch["bucket"], u_hi, cfg.pair_per_bucket)
                    if pair_idxs:
                        ii = torch.tensor([p[0] for p in pair_idxs], device=device)
                        jj = torch.tensor([p[1] for p in pair_idxs], device=device)
                        rank_pairs = (mu[ii], mu[jj], u_hi[ii], u_hi[jj])
                # Replace NaN with the prediction to neutralize that term
                u_hi_safe = torch.where(u_hi_valid, u_hi, mu.detach())
                u_lo_safe = torch.where(~torch.isnan(u_lo), u_lo, mu.detach())
                loss, parts = combine_losses(
                    mu=mu, sigma=sigma,
                    u_hi=u_hi_safe if u_hi_valid.any() else None,
                    u_lo=u_lo_safe,
                    weights=weights,
                    rank_pairs=rank_pairs,
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step % 100 == 0:
                    log.info(f"[phase {tag}][epoch {epoch}][step {step}] "
                             f"loss={float(loss.detach()):.4f} {parts}")

    _run_phase(phase_a_loader, epochs=cfg.epochs_phase_a, lr=cfg.lr_phase_a,
               weights=cfg.weights_phase_a, tag="A")
    torch.save(model.state_dict(), ckpt_dir / "ckpt_a.pt")
    _run_phase(phase_b_loader, epochs=cfg.epochs_phase_b, lr=cfg.lr_phase_b,
               weights=cfg.weights_phase_b, tag="B")
    final = ckpt_dir / "ckpt_b.pt"
    torch.save(model.state_dict(), final)
    return final


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)
