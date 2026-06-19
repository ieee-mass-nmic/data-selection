"""CLI: build sample features (e_x, d_x, a_x) + grad signatures → FeatureCache.

Runs the offline feature stage (`pipeline.offline._build_features`): one
forward+backward per sample on the selector model yields the model-side
difficulty stats, the activation signature and per-site gradient signatures,
which are persisted under `--workdir/features`.

Usage:
    python scripts/build_features.py \
        --pool data/alpaca_100k.jsonl \
        --workdir runs/exp1 \
        --selector meta-llama/Llama-2-7b-hf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcu_select.data import JsonlPool
from pcu_select.features.cache import FeatureCache
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.pipeline.offline import _build_features
from pcu_select.types import OfflineConfig, WorkDirLayout
from pcu_select.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--selector", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-layers-sig", type=int, default=8)
    parser.add_argument("--n-layers-total", type=int, default=32)
    parser.add_argument("--d-proj", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()

    log = get_logger("build_features")
    layout = WorkDirLayout(args.workdir)
    cache = FeatureCache(layout.features)
    pool = JsonlPool.from_jsonl(args.pool)
    log.info(f"loaded {len(pool)} samples from {args.pool}")

    cfg = OfflineConfig(
        selector_model=args.selector,
        n_layers_total=args.n_layers_total,
        n_layers_signature=args.n_layers_sig,
        d_proj=args.d_proj,
        device=args.device,
        global_seed=args.global_seed,
    )
    sites = SiteSpace.uniform(n_layers_total=args.n_layers_total, k=args.n_layers_sig)

    _build_features(meta_pool=pool, cache=cache, sites=sites, cfg=cfg)
    log.info(f"wrote features + grad signatures → {layout.features}")


if __name__ == "__main__":
    main()
