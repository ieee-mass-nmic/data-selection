"""CLI: build a small high-fidelity calibration label set for E5 (design §13.2).

E5's calibration mode (`run_e5.py --calib-labels`) fits a calibration head on a
SMALL high-fidelity label set for an unseen target PEFT. This script produces
that parquet: for each (target PEFT, task) it samples `--n-calib` candidate
samples and runs the high-fidelity short-update protocol (horizon=1, a single
`base` anchor ≈ 1/4 of full-fidelity cost) to obtain u_hi.

Output columns match `labels/hi_fidelity.parquet`
(sample_id, peft_id, task_id, u_hi, ...) — exactly what `run_e5._calibrate`
reads back.

Implementation boundary: the native short-update backend supports only
lora / ia3 / adapter / bitfit (hi_fidelity.native_peft.SUPPORTED_FAMILIES).
prefix / ptuning targets cannot be labelled and are skipped with a warning, so
E5's L2 prefix/ptuning run zero-shot only (design §6.5).

Example:
    python scripts/experiments/build_calib_labels.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --model llama2-7b \
        --tasks gsm8k --pefts L-r64-all AD-b256 L-r8-highlayers BF \
        --n-calib 500 --out runs/exp1/labels/calib.parquet
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from pcu_select.data import JsonlPool, load_sketch
from pcu_select.experiments import MODELS, resolve_peft
from pcu_select.features.cache import FeatureCache
from pcu_select.hi_fidelity import (
    AnchorRegistry,
    AnchorSpec,
    HiFidelityLabeler,
    LabelerConfig,
    ShortUpdater,
    TripleSample,
    load_anchor_model,
)
from pcu_select.hi_fidelity.native_peft import SUPPORTED_FAMILIES
from pcu_select.types import WorkDirLayout
from pcu_select.utils import get_logger

# E5 calibration targets the native backend can actually label (design §6.2:
# L1 extrapolation + L2 bitfit). prefix/ptuning are excluded by SUPPORTED_FAMILIES.
DEFAULT_PEFTS = ["L-r64-all", "AD-b256", "L-r8-highlayers", "BF"]


def main() -> None:
    p = argparse.ArgumentParser(description="Build E5 calibration high-fidelity labels")
    p.add_argument("--workdir", type=Path, required=True,
                   help="Offline run dir (feature cache + task sketches).")
    p.add_argument("--pool", type=Path, required=True, help="Candidate pool jsonl.")
    p.add_argument("--model", type=str, default="llama2-7b",
                   help="Backbone tag from experiments.registry.MODELS.")
    p.add_argument("--tasks", type=str, nargs="+", default=["gsm8k"])
    p.add_argument("--pefts", type=str, nargs="+", default=DEFAULT_PEFTS,
                   help="Target PEFT registry names to calibrate.")
    p.add_argument("--n-calib", type=int, default=500,
                   help="Samples to label per (peft, task); use the largest E5 --calib-size.")
    p.add_argument("--sketch-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="Sampling + short-update seed.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--warm-anchor", type=Path, default=None,
                   help="Optional warm-anchor adapter ckpt; default uses the base anchor only.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output parquet (default: <workdir>/labels/calib.parquet).")
    args = p.parse_args()

    log = get_logger("build_calib_labels")
    if args.model not in MODELS:
        raise SystemExit(f"unknown model {args.model!r}; known: {sorted(MODELS)}")
    spec = MODELS[args.model]
    layout = WorkDirLayout(args.workdir)
    cache = FeatureCache(layout.features)
    pool = JsonlPool.from_jsonl(args.pool)

    # ---- sample a fixed calibration subset from the cached candidate ids ----
    all_ids = list(cache.read_sample_id_index())
    n = min(args.n_calib, len(all_ids))
    sampled_ids = random.Random(args.seed).sample(all_ids, n)
    samples_by_id = {s.sample_id: s for s in pool.take(sampled_ids)}
    log.info(f"sampled {len(samples_by_id)} calibration samples (requested {args.n_calib})")

    # ---- resolve the target PEFTs the native backend can label ----
    pefts_by_id = {}
    for name in args.pefts:
        peft = resolve_peft(name, args.model)
        if peft.family not in SUPPORTED_FAMILIES:
            log.warning(f"skip {name}: family {peft.family!r} has no native short-update "
                        f"backend (prefix/ptuning) — E5 runs it zero-shot only.")
            continue
        pefts_by_id[peft.peft_id] = peft
    if not pefts_by_id:
        raise SystemExit("no labelable target PEFTs (all prefix/ptuning?); nothing to do.")

    # ---- per-task selection sketches act as the validation set L_V ----
    sketches_by_id = {}
    task_ids = []
    for task in args.tasks:
        sketch = load_sketch(layout.task / "sketches" / f"{task}_{args.sketch_seed}.json")
        sketches_by_id[sketch.task_id] = sketch
        task_ids.append(sketch.task_id)

    # ---- single `base` anchor (design §13.2: single anchor, horizon=1 ≈ 1/4 cost) ----
    anchors = AnchorRegistry(layout.root / "anchors")
    base_ckpt = layout.root / "anchors" / "base"
    anchors.register(AnchorSpec(anchor_id="base", checkpoint_path=base_ckpt))
    if args.warm_anchor is not None:
        anchors.register(AnchorSpec(anchor_id="warm", checkpoint_path=args.warm_anchor))

    def updater_factory(anchor: AnchorSpec) -> ShortUpdater:
        model, tok = load_anchor_model(anchor, base_model_path=spec.hf_id, device=args.device)
        return ShortUpdater(model, tok, device=args.device)

    labeler = HiFidelityLabeler(
        anchors=anchors,
        samples_by_id=samples_by_id,
        pefts_by_id=pefts_by_id,
        sketches_by_id=sketches_by_id,
        updater_factory=updater_factory,
        cfg=LabelerConfig(horizons=(1,), horizon_weights=(1.0,), seed=args.seed),
    )

    triples = [
        TripleSample(sample_id=sid, peft_id=pid, task_id=tid, phase=1)
        for sid in samples_by_id
        for pid in pefts_by_id
        for tid in task_ids
    ]
    log.info(f"labeling {len(triples)} triples "
             f"[{len(pefts_by_id)} pefts × {len(task_ids)} tasks × {len(samples_by_id)} samples]")
    labels = labeler.run(triples)

    out = args.out or (layout.labels / "calib.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    HiFidelityLabeler.save_labels(labels, out)
    log.info(f"wrote {len(labels)} calibration labels → {out}")


if __name__ == "__main__":
    main()
