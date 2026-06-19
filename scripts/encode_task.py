"""CLI: build a task sketch, its grad signature g_t and task vector z_t.

The apply pipeline needs a pre-computed `z_t_<task_id>.npy` (see
`pipeline/apply.py`). This script builds it (and the sketch) from a jsonl split
of the target task. Use the same `--global-seed` as the offline run so the
random projection matrices Φ_ω match.

Usage:
    python scripts/encode_task.py \
        --task-jsonl data/gsm8k_train.jsonl \
        --task-name gsm8k \
        --workdir runs/exp1 \
        --selector meta-llama/Llama-2-7b-hf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcu_select.data import JsonlPool, build_sketch, save_sketch
from pcu_select.features.selector_runner import SelectorRunner, SelectorRunnerConfig
from pcu_select.features.semantic import SemanticEncoder, SemanticEncoderConfig
from pcu_select.peft_space.site_mask import SiteSpace
from pcu_select.pipeline.offline import build_task_artifacts
from pcu_select.proxy.projection import ProjectionConfig, ProjectionStore
from pcu_select.types import OfflineConfig, TaskConfig, WorkDirLayout
from pcu_select.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-jsonl", type=Path, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--selector", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sketch-size", type=int, default=32)
    parser.add_argument("--sketch-seed", type=int, default=0)
    parser.add_argument("--global-seed", type=int, default=0,
                        help="Must match the offline run's global_seed (Φ_ω projections).")
    parser.add_argument("--n-layers-sig", type=int, default=8)
    parser.add_argument("--n-layers-total", type=int, default=32)
    parser.add_argument("--d-proj", type=int, default=256)
    args = parser.parse_args()

    log = get_logger("encode_task")
    layout = WorkDirLayout(args.workdir)
    cfg = OfflineConfig(
        selector_model=args.selector,
        n_layers_total=args.n_layers_total,
        n_layers_signature=args.n_layers_sig,
        d_proj=args.d_proj,
        device=args.device,
        global_seed=args.global_seed,
    )

    pool = JsonlPool.from_jsonl(args.task_jsonl)
    sketch = build_sketch(list(pool), task_name=args.task_name,
                          n=args.sketch_size, seed=args.sketch_seed)
    (layout.task / "sketches").mkdir(parents=True, exist_ok=True)
    save_sketch(sketch, layout.task / "sketches" / f"{args.task_name}_{args.sketch_seed}.json")
    task = TaskConfig(name=args.task_name, task_id=sketch.task_id, sketch=sketch)
    log.info(f"built sketch for {args.task_name}: {len(sketch.samples)} samples, id={sketch.task_id}")

    sites = SiteSpace.uniform(n_layers_total=args.n_layers_total, k=args.n_layers_sig)
    runner = SelectorRunner(sites, SelectorRunnerConfig(selector_model=args.selector, device=args.device))
    proj = ProjectionStore(
        ProjectionConfig(d_model=runner.hidden_size, d_proj=args.d_proj, global_seed=args.global_seed),
        layout.features / "projections",
    )
    sem = SemanticEncoder(SemanticEncoderConfig(device=args.device))

    build_task_artifacts(task=task, sites=sites, layout=layout, cfg=cfg,
                         runner=runner, proj=proj, sem=sem)
    log.info(f"wrote task_grad_{task.task_id}.npy and z_t_{task.task_id}.npy → {layout.task}")


if __name__ == "__main__":
    main()
