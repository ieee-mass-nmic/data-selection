"""CLI: compute low-fidelity u^lo for all (sample, peft, task) triples.

Assumes:
    - feature cache populated (build_features.py)
    - per-site grad signatures already cached for both samples and sketches
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pcu_select.data.sketch import load_sketch
from pcu_select.features.cache import FeatureCache
from pcu_select.peft_space.schema import load_peft_config
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.proxy.lo_fidelity import LoFidelityScorer
from pcu_select.types import WorkDirLayout
from pcu_select.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--peft-configs", type=Path, nargs="+", required=True)
    parser.add_argument("--task-sketches", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    log = get_logger("compute_lo_fidelity")
    layout = WorkDirLayout(args.workdir)
    cache = FeatureCache(layout.features)
    sites = SiteSpace.uniform(n_layers_total=32, k=8)
    scorer = LoFidelityScorer(sites, cache)

    rows = []
    for sketch_path in args.task_sketches:
        sketch = load_sketch(sketch_path)
        task_grad_path = layout.task / f"task_grad_{sketch.task_id}.npy"
        if not task_grad_path.exists():
            log.warning(f"missing {task_grad_path}; skip")
            continue
        g_t = np.load(task_grad_path)  # (|Ω|, d_proj)
        for peft_path in args.peft_configs:
            cfg = load_peft_config(peft_path)
            result = scorer.score(peft=cfg, g_t_per_site=g_t)
            for sid, u in zip(result.sample_ids, result.u_lo):
                rows.append({
                    "sample_id": sid,
                    "peft_id": cfg.peft_id,
                    "task_id": sketch.task_id,
                    "u_lo": float(u),
                })
    out = layout.labels / "lo_fidelity.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    log.info(f"wrote {len(rows)} u_lo rows → {out}")


if __name__ == "__main__":
    main()
