"""Materialize the PEFT registry (experiments.registry) to yaml configs.

Writes one `<group>/<name>.yaml` per registry entry under `--out-dir`, resolved
for a given backbone depth, using the same serializer (`dump_peft_config`) as
the hand-written configs in `configs/peft/`. Useful for inspection / reuse by
the offline pipeline, and to keep the python registry and on-disk configs in
sync.

    python scripts/experiments/dump_peft_registry.py --model llama2-7b \
        --out-dir configs/peft/registry
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcu_select.experiments import PEFT_REGISTRY, resolve_peft
from pcu_select.peft_space.schema import dump_peft_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="llama2-7b")
    ap.add_argument("--out-dir", type=Path, default=Path("configs/peft/registry"))
    args = ap.parse_args()

    n = 0
    for name, spec in PEFT_REGISTRY.items():
        cfg = resolve_peft(name, args.model)
        out = args.out_dir / spec.group / f"{name}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        dump_peft_config(cfg, out)
        n += 1
    print(f"wrote {n} PEFT configs → {args.out_dir} (resolved for {args.model})")


if __name__ == "__main__":
    main()
