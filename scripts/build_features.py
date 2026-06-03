"""CLI: build sample features (e_x, d_x, a_x) and persist to FeatureCache.

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
from pcu_select.features import (
    ActivationSignatureConfig,
    ActivationSignatureExtractor,
    DifficultyConfig,
    FeatureCache,
    ModelStatsExtractor,
    SemanticEncoder,
    SemanticEncoderConfig,
    quick_text_stats,
)
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.types import ActivationSignature, DifficultyStats, SampleFeatures, WorkDirLayout
from pcu_select.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--selector", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--n-layers-sig", type=int, default=8)
    parser.add_argument("--n-layers-total", type=int, default=32)
    args = parser.parse_args()

    log = get_logger("build_features")
    layout = WorkDirLayout(args.workdir)
    cache = FeatureCache(layout.features)
    pool = JsonlPool.from_jsonl(args.pool)
    log.info(f"loaded {len(pool)} samples from {args.pool}")

    sites = SiteSpace.uniform(n_layers_total=args.n_layers_total, k=args.n_layers_sig)

    sem = SemanticEncoder(SemanticEncoderConfig())
    stats = ModelStatsExtractor(DifficultyConfig(selector_model=args.selector))
    act = ActivationSignatureExtractor(sites, ActivationSignatureConfig(selector_model=args.selector))

    samples = list(pool)
    emb = sem.encode_samples(samples)
    cheap = [quick_text_stats(s) for s in samples]
    # NOTE: model-side stats and activation signatures need real model integration;
    # they currently raise NotImplementedError. Wire them up before running for real.
    try:
        d_x = stats.extract(samples, cheap)
    except NotImplementedError:
        d_x = [DifficultyStats(vector=c) for c in cheap]
        log.warning("Model stats not implemented; using cheap text stats only.")
    try:
        a_x = act.extract(samples)
    except NotImplementedError:
        from numpy import zeros
        a_x = [ActivationSignature(vector=zeros(args.n_layers_sig * 8)) for _ in samples]
        log.warning("Activation signature extractor not implemented; using zeros.")

    feats = [
        SampleFeatures(sample_id=s.sample_id, e_x=emb[i], d_x=d_x[i], a_x=a_x[i])
        for i, s in enumerate(samples)
    ]
    cache.write_features(feats)
    cache.write_sample_id_index([s.sample_id for s in samples])
    log.info(f"wrote {len(feats)} feature rows → {cache.paths.parquet}")


if __name__ == "__main__":
    main()
