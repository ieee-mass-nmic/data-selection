"""CLI: train scorer from cached lo / hi labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from pcu_select.scorer.model import PCUScorer
from pcu_select.scorer.trainer import (
    TrainerConfig,
    TripletDataset,
    make_loader,
    train_scorer,
)
from pcu_select.types import WorkDirLayout
from pcu_select.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    log = get_logger("train_scorer")
    layout = WorkDirLayout(args.workdir)

    lo = pd.read_parquet(layout.labels / "lo_fidelity.parquet").to_dict("records")
    hi_path = layout.labels / "hi_fidelity.parquet"
    hi = pd.read_parquet(hi_path).to_dict("records") if hi_path.exists() else []

    # NOTE: these lookup tables must be precomputed; see design doc §15.
    z_x_by_id: dict[str, np.ndarray] = {}
    z_p_by_id: dict[str, np.ndarray] = {}
    z_t_by_id: dict[str, np.ndarray] = {}

    ds_lo = TripletDataset(rows=lo, z_x_by_id=z_x_by_id, z_p_by_id=z_p_by_id, z_t_by_id=z_t_by_id)
    ds_hi = TripletDataset(rows=hi or lo, z_x_by_id=z_x_by_id, z_p_by_id=z_p_by_id, z_t_by_id=z_t_by_id)
    loader_a = make_loader(ds_lo, batch_size=args.batch_size)
    loader_b = make_loader(ds_hi, batch_size=args.batch_size)

    model = PCUScorer()
    ckpt = train_scorer(
        model=model,
        phase_a_loader=loader_a,
        phase_b_loader=loader_b,
        cfg=TrainerConfig(),
        ckpt_dir=layout.scorer,
    )
    log.info(f"saved scorer ckpt → {ckpt}")


if __name__ == "__main__":
    main()
