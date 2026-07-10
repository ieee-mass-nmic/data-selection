"""E5 — unseen PEFT, with OOD detection + calibration (design §6).

Stratifies unseen targets into three levels and reports uncalibrated vs
calibrated behaviour for each, plus the Mahalanobis d² that the OOD detector
uses:

  L0 near support : unseen same-family config with a small structural shift
  L1 far support  : unseen same-family config with an extreme/compound shift
  L2 unseen family: prefix / ptuning / bitfit

Calibration consumes a SMALL pre-computed high-fidelity label set for the target
PEFT (`--calib-labels <parquet>` with columns sample_id, peft_id, task_id,
u_hi). Generate it with scripts/experiments/build_calib_labels.py on 200/500
sampled samples (design §13.2).

Native short-update labels are produced for lora / ia3 / adapter / bitfit.
Prompt families are evaluated through the common target-training path and can
use externally supplied calibration labels with the same parquet schema. If
labels for a (peft, task) are absent, E5 reports the uncalibrated PCU mode and
logs the missing calibration source.

Per design §1.7 each cell repeats over `--seeds` (default 0 1 2). OOD stats and
the per-sample features are task/seed-independent and computed once.

Example:
    python scripts/experiments/run_e5.py \
        --workdir runs/exp1 --pool data/pool_300k.jsonl --eval-dir data/eval \
        --tasks gsm8k humaneval mmlu --seeds 0 1 2 \
        --calib-labels runs/exp1/labels/calib.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import RunContext, add_common_args, evaluate_selection, run_cell

from pcu_select.experiments import peft_specs_by_group, resolve_peft
from pcu_select.ood.calibration import CalibrationHead, fit_calibration, fit_ood_stats
from pcu_select.peft_space.encoder import encode_peft
from pcu_select.selection.selector import SelectorConfig, select as cluster_select

# Pre-registered structural strata. Mahalanobis d² is logged only as a
# diagnostic; it does not define these groups.
L0 = ["L-r32-qkvo", "AD-b16"]
L1 = ["L-r64-all", "AD-b256", "L-r8-highlayers"]
L2 = ["PRE-l16", "PT-l32", "BF"]
LEVEL = {**{n: "L0" for n in L0}, **{n: "L1" for n in L1}, **{n: "L2" for n in L2}}
REFERENCE_METHODS = ["random", "rds_plus", "less"]


def mahalanobis2(z: np.ndarray, mu: np.ndarray, sinv: np.ndarray) -> float:
    d = z - mu
    return float(d @ sinv @ d.T)


def main() -> None:
    p = argparse.ArgumentParser(description="E5: unseen PEFT generalization")
    add_common_args(p)
    p.add_argument("--test-pefts", type=str, nargs="+", default=L0 + L1 + L2)
    p.add_argument("--calib-labels", type=str, default=None)
    p.add_argument("--calib-sizes", type=int, nargs="+", default=[200, 500])
    p.set_defaults(tasks=["gsm8k"])  # E5 defaults to one task; pass more to fill the matrix
    args = p.parse_args()
    budget = args.budgets[0]

    ctx = RunContext(args, experiment="E5")

    # ---- OOD stats from the SEEN PEFT support (task/seed-independent) ----
    seen = [resolve_peft(s.name, args.model) for s in peft_specs_by_group("seen")]
    z_p_seen = np.stack([encode_peft(p, ctx.sites) for p in seen], axis=0)
    ood = fit_ood_stats(z_p_seen, quantile=0.95)

    calib = pd.read_parquet(args.calib_labels) if args.calib_labels else None

    # ---- per-sample features z_x (task/seed-independent) ----
    feats = ctx.cache.read_features()
    ids = ctx.cache.read_sample_id_index()
    z_x = np.stack([feats[i].as_z_x() for i in ids], axis=0)

    for task in args.tasks:
        tc = ctx.task(task)
        z_t = tc.z_t[None, :]
        for name in args.test_pefts:
            peft = resolve_peft(name, args.model)
            z_p = encode_peft(peft, ctx.sites)[None, :]
            d2 = mahalanobis2(z_p[0], ood.mu, ood.sigma_inv)
            meta = {"level": LEVEL.get(name, "?"), "d2": d2, "ood_threshold": ood.threshold,
                    "is_ood": bool(d2 > ood.threshold)}
            mu, sigma = ctx.scorer().score(z_x, z_p, z_t)

            # Uncalibrated PCU selection, repeated target-train per seed.
            res = cluster_select(sample_ids=ids, mu=mu, sigma=sigma,
                                 joint_embeddings=tc.inp.joint, budget=_k(budget, len(ids)),
                                 cfg=SelectorConfig())
            for seed in args.seeds:
                evaluate_selection(ctx, peft_name=name, task_name=task, budget=budget,
                                   seed=seed, ids=res.selected_ids, method_tag="pcu_zeroshot",
                                   dense=mu, extra={**meta, "mode": "zeroshot"})

            # calibrated modes (only if labels for this peft/task exist)
            sub = calib[(calib["peft_id"] == peft.peft_id) & (calib["task_id"] == tc.task_id)] \
                if calib is not None else None
            if sub is None or sub.empty:
                why = "no calib labels" if calib is not None else "no --calib-labels"
                print(f"  {task}/{name}: {why}; continuing with uncalibrated mode")
            else:
                for n_cal in args.calib_sizes:
                    mu_cal = _calibrate(z_x, z_p, z_t, mu, sub, ids, n_cal, args.device)
                    resc = cluster_select(sample_ids=ids, mu=mu_cal, sigma=sigma,
                                          joint_embeddings=tc.inp.joint,
                                          budget=_k(budget, len(ids)), cfg=SelectorConfig())
                    for seed in args.seeds:
                        evaluate_selection(ctx, peft_name=name, task_name=task, budget=budget,
                                           seed=seed, ids=resc.selected_ids,
                                           method_tag=f"pcu_cal{n_cal}", dense=mu_cal,
                                           extra={**meta, "mode": f"cal{n_cal}"})

            # Reference baselines are attempted for every target; unsupported
            # signal combinations are reported once per method.
            for m in REFERENCE_METHODS:
                for seed in args.seeds:
                    try:
                        run_cell(ctx, method=m, peft_name=name, task_name=task,
                                 budget=budget, seed=seed, extra=meta)
                    except (NotImplementedError, ValueError) as e:
                        print(f"  {task}/{name}: baseline {m} skipped ({e})")
                        break  # same failure for every seed; don't repeat
    print(f"E5 done → {ctx.results_path}")


def _k(budget: float, n: int) -> int:
    return int(budget) if budget >= 1 else max(1, int(round(budget * n)))


def _calibrate(z_x, z_p, z_t, mu, sub, ids, n_cal, device) -> np.ndarray:
    """Fit a calibration head on n_cal labelled samples, return calibrated μ."""
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    rows = [(id_to_idx[s], u) for s, u in zip(sub["sample_id"], sub["u_hi"]) if s in id_to_idx]
    rows = rows[:n_cal]
    idx = np.array([r[0] for r in rows])
    u_hi = np.array([r[1] for r in rows], dtype=np.float32)
    head = CalibrationHead(in_dim=z_x.shape[1] + z_p.shape[1] + z_t.shape[1])
    z_p_b = np.repeat(z_p, len(idx), axis=0)
    z_t_b = np.repeat(z_t, len(idx), axis=0)
    fit_calibration(head=head, mu_hat=mu[idx], u_hi=u_hi, z_x=z_x[idx], z_p=z_p_b,
                    z_t=z_t_b, device=device)
    import torch
    with torch.no_grad():
        dev = next(head.parameters()).device
        mu_all = head(torch.as_tensor(mu, device=dev),
                      torch.as_tensor(z_x, device=dev),
                      torch.as_tensor(z_p, device=dev),
                      torch.as_tensor(z_t, device=dev)).cpu().numpy()
    return mu_all


if __name__ == "__main__":
    main()
