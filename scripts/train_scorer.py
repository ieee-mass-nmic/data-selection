"""CLI: train scorer from cached lo / hi labels + feature lookups.

Reads everything from a finished offline workdir (feature cache + persisted
z_p / z_t artifacts + lo/hi label parquets) and runs the two-phase trainer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcu_select.features.cache import FeatureCache
from pcu_select.pipeline.offline import run_scorer_training
from pcu_select.types import OfflineConfig, WorkDirLayout
from pcu_select.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    log = get_logger("train_scorer")
    layout = WorkDirLayout(args.workdir)
    cache = FeatureCache(layout.features)
    cfg = OfflineConfig(scorer_batch_size=args.batch_size, device=args.device)

    ckpt = run_scorer_training(cache=cache, layout=layout, cfg=cfg)
    log.info(f"saved scorer ckpt → {ckpt}")


if __name__ == "__main__":
    main()
