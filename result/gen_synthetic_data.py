"""Synthetic-result generator for the PCU-Select paper figures/tables.

==============================  IMPORTANT  ==================================
The numbers produced here are **NOT experimental results**. No 7B backbone,
no PEFT fine-tuning and no short-update labelling were run. This script
fabricates *plausible* numbers (right order of magnitude, realistic noise,
deliberate imperfections) purely so the existing plotting scripts have
something to render while the real offline/GPU pipeline is unavailable.

See result/README_SYNTHETIC_DATA.md for the full disclaimer and the mapping
from each output file to the experiment it stands in for.
============================================================================

Design goals (per the user's brief):
  * realistic magnitudes, drawn from the design docs' own example numbers;
  * genuine seed-to-seed variance (3 target-train seeds → mean ± std);
  * a few cells where our method is NOT the best (no clean sweep);
  * no exaggerated gains; PCU sits just above LESS on average and loses
    on a handful of (peft, task) cells;
  * not every experiment supports the same conclusion (e.g. PCU never beats
    the cheap RDS+ on raw cost in E2; L2 prefix/ptuning fails in E5);
  * everything is driven by ONE fixed master seed → fully reproducible.

Outputs land under result/data/ in exactly the schema the runners would emit
(ResultRow JSONL + the E2/E4 JSON side-files + motivation parquet/JSONL).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pcu_select.experiments import ResultRow, resolve_peft

# --------------------------------------------------------------------------
# Reproducibility: one master seed for the whole synthetic dataset.
# --------------------------------------------------------------------------
MASTER_SEED = 20260629
RNG = np.random.default_rng(MASTER_SEED)


def _det(*key) -> float:
    """Deterministic pseudo-random in [0, 1) from a key.

    Uses md5(repr) instead of the builtin hash() — Python salts str/tuple
    hashing per process (PYTHONHASHSEED), so hash() is NOT reproducible across
    runs. This is: the stable structured-irregularity source used everywhere we
    want a fixed-but-uneven wrinkle (method×task quirks, mismatch asymmetry, …).
    """
    digest = hashlib.md5(repr(key).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def _detc(*key, lo: float = -1.0, hi: float = 1.0) -> float:
    """Deterministic pseudo-random centered in [lo, hi]."""
    return lo + (hi - lo) * _det(*key)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = "llama2-7b"

# --------------------------------------------------------------------------
# Registry slices used across experiments
# --------------------------------------------------------------------------
SEEN = ["L-r8-qv", "L-r16-qkvo", "L-r8-mlp", "IA3-attnmlp", "AD-b64"]
MAIN_TASKS = ["gsm8k", "humaneval", "mmlu", "tydiqa"]
BUDGETS = [0.05, 0.10, 0.30]

E1_METHODS = ["random", "balanced_random", "length", "loss", "perplexity", "ifd", "s2l",
              "embedding_nn", "rds_plus", "diversity", "grad_sim", "less", "pcu"]

# Per-task "Random @ budget=10%" anchor level, in the task's native metric units
# (EM% / pass@1% / accuracy% / F1). These mirror the ballpark in the design docs.
TASK_BASE = {"gsm8k": 33.0, "humaneval": 29.0, "mmlu": 45.0, "tydiqa": 49.0}
TASK_METRIC = {"gsm8k": "exact_match", "humaneval": "pass@1", "mmlu": "accuracy", "tydiqa": "f1"}
TASK_NOISE = {"gsm8k": 0.55, "humaneval": 0.70, "mmlu": 0.40, "tydiqa": 0.60}

# Method quality offset (points above Random). PCU is only ~0.3 over LESS.
METHOD_OFFSET = {
    "random": 0.0, "balanced_random": 0.5, "length": 0.3, "loss": 0.7, "perplexity": 0.6,
    "ifd": 1.2, "s2l": 1.4, "embedding_nn": 1.7, "diversity": 1.5, "rds_plus": 2.1,
    "grad_sim": 2.6, "less": 3.1, "pcu": 4.05,
}

# Ranking-quality (vs high-fidelity truth) per method, drives spearman/ndcg/...
RANK_Q = {
    "random": 0.02, "balanced_random": 0.05, "length": 0.12, "loss": 0.18, "perplexity": 0.16,
    "ifd": 0.28, "s2l": 0.33, "embedding_nn": 0.38, "diversity": 0.30, "rds_plus": 0.46,
    "grad_sim": 0.55, "less": 0.60, "pcu": 0.64,
}

# PEFT capacity ceiling (points). Bigger adapters reach a bit higher.
PEFT_CAP = {
    "L-r8-qv": 0.0, "L-r16-qkvo": 1.2, "L-r8-mlp": 0.4, "IA3-attnmlp": -0.6, "AD-b64": 1.0,
    "L-r4-qv": -0.8, "L-r32-qkvo": 1.8, "L-r64-all": 2.3, "L-r8-lowlayers": -0.5,
    "L-r8-highlayers": -0.2, "L-r16-hlr": 1.0, "AD-b16": -0.4, "AD-b256": 1.6,
    "IA3-attnonly": -0.9, "PRE-l16": -1.5, "PT-l32": -1.3, "BF": -2.0,
}


def _budget_delta(budget: float, task: str) -> float:
    """More data helps with diminishing returns; anchored at budget=10%.

    The slope is task-specific (some tasks saturate faster) so the budget curves
    are not parallel copies of one another.
    """
    slope = 1.5 * (1.0 + 0.20 * _detc(task, "budget_slope"))
    return slope * math.log(budget / 0.10)


def _interaction(method: str, peft: str, task: str) -> float:
    """Per-cell structure so the method ranking is NOT globally identical.

    Combines (a) curated, domain-plausible effects with (b) deterministic
    method×task and method×PEFT pseudo-random offsets. The latter mean the
    column ordering jitters from task to task and PEFT to PEFT — as in real
    tables, a mid baseline occasionally leapfrogs a stronger one on one cell —
    while the zero-mean structure preserves the overall standings on average.
    """
    bump = 0.0
    # (a) curated effects — keep the multilingual task a genuine PCU weak spot
    # (LESS edges it there) and MLP-placement LoRA the clear PCU loss.
    if method == "pcu":
        if peft == "L-r8-mlp":
            bump -= 0.85          # scorer weaker on pure-MLP placement
        if task == "tydiqa":
            bump -= 0.40          # multilingual: PCU slightly behind LESS here
    if method == "less" and task == "tydiqa":
        bump += 0.30
    # cheap heuristics: relatively better on knowledge (MMLU), worse on code
    if method in ("loss", "perplexity", "length") and task == "humaneval":
        bump -= 0.45
    if method in ("loss", "perplexity", "ifd") and task == "mmlu":
        bump += 0.40
    # (b) deterministic pseudo-random texture (reproducible, uneven). `mt` gives
    # per-task column-order jitter — mostly reshuffling the crowded mid-field and
    # occasionally letting a baseline top one task — while `mp` is a small
    # per-PEFT wrinkle. Both amplitudes stay below PCU's lead so it still wins
    # most PEFTs *on the cross-task average* yet loses individual cells.
    bump += 0.45 * _detc(method, task, "mt")
    bump += 0.24 * _detc(method, peft, "mp")
    return bump


def downstream_metric(method: str, peft: str, task: str, budget: float, rng) -> float:
    m = (TASK_BASE[task] + METHOD_OFFSET.get(method, 0.0) + PEFT_CAP.get(peft, 0.0)
         + _budget_delta(budget, task) + _interaction(method, peft, task))
    m += 0.22 * _detc(method, peft, task, "cell")             # fine fixed texture
    # heteroscedastic seed noise: per-cell spread varies, plus occasional outliers
    sigma = TASK_NOISE[task] * (0.80 + 0.55 * _det(method, peft, task, "sig"))
    m += rng.normal(0.0, sigma)
    if rng.random() < 0.05:                                   # rare bad/lucky run
        m += rng.normal(0.0, 1.5)
    return float(round(m, 4))


def eval_loss_for(metric: float, task: str, rng) -> float:
    """A plausible held-out LM loss, loosely (imperfectly) anti-correlated."""
    return float(round(1.95 - 0.012 * metric + rng.normal(0.0, 0.035), 4))


def ranking_metrics(method: str, rng, *, present: bool = True) -> dict:
    if not present:
        nan = float("nan")
        return dict(spearman=nan, kendall_tau=nan, ndcg_at_k=nan,
                    topk_hit_rate=nan, pairwise_acc=nan)
    q = RANK_Q.get(method, 0.1)
    sp = float(np.clip(q + rng.normal(0, 0.055), -0.1, 0.95))
    # the five ranking metrics are correlated but not perfectly redundant
    return dict(
        spearman=round(sp, 4),
        kendall_tau=round(sp * 0.72 + rng.normal(0, 0.02), 4),
        ndcg_at_k=round(float(np.clip(0.35 + 0.55 * q + rng.normal(0, 0.045), 0, 1)), 4),
        topk_hit_rate=round(float(np.clip(0.05 + 0.55 * q + rng.normal(0, 0.05), 0, 1)), 4),
        pairwise_acc=round(float(np.clip(0.5 + 0.45 * q + rng.normal(0, 0.03), 0, 1)), 4),
    )


def _write_rows(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(r.to_json() + "\n")
    print(f"wrote {len(rows):>5} rows → {path.relative_to(HERE)}")


def _row(experiment, method, peft, task, budget, seed, metric, rng, extra=None,
         ranking=True, select_h=0.0, train_h=0.0):
    rk = ranking_metrics(method if method in RANK_Q else "pcu", rng, present=ranking)
    return ResultRow(
        experiment=experiment, method=method, peft=peft, task=task, budget=budget,
        seed=seed, model=MODEL, metric_name=TASK_METRIC.get(task, "exact_match"),
        metric=metric, eval_loss=eval_loss_for(metric, task, rng),
        select_gpu_h=select_h, target_train_gpu_h=train_h,
        extra={"n_selected": extra.pop("n_selected", 0) if extra else 0, **(extra or {})},
        **rk,
    )


# ==========================================================================
# E1 — method × PEFT × task × budget × seed
# ==========================================================================
def gen_e1() -> None:
    rng = np.random.default_rng(MASTER_SEED + 1)
    rows: list[ResultRow] = []
    for peft in SEEN:
        for task in MAIN_TASKS:
            for budget in BUDGETS:
                n_sel = int(round(budget * 300_000))
                for method in E1_METHODS:
                    for seed in (0, 1, 2):
                        metric = downstream_metric(method, peft, task, budget, rng)
                        # selection-quality only meaningful for score-based methods
                        ranking = method not in ("random", "balanced_random", "length")
                        rows.append(_row("E1", method, peft, task, budget, seed, metric, rng,
                                         extra={"n_selected": n_sel}, ranking=ranking,
                                         select_h=round(_select_cost(method), 4),
                                         train_h=round(0.85 + rng.normal(0, 0.03), 4)))
    _write_rows(DATA / "E1.jsonl", rows)


def _select_cost(method: str) -> float:
    # PCU's apply-time selection is a touch dearer than the cheap forward-only
    # baselines (RDS+/PPL): scoring the pool with the scorer + clustering. This
    # makes PCU honestly NEVER break even vs cheap baselines on raw cost — the
    # amortization story only holds against per-PEFT influence methods (LESS).
    return {"pcu": 0.05, "less": 0.11, "grad_sim": 0.10, "rds_plus": 0.02,
            "embedding_nn": 0.03, "random": 0.0, "balanced_random": 0.0}.get(method, 0.02)


# ==========================================================================
# E2 — multi-PEFT amortized cost (one task / seed) + cost-model side file
# ==========================================================================
def gen_e2() -> None:
    rng = np.random.default_rng(MASTER_SEED + 2)
    methods = ["pcu", "less", "grad_sim", "rds_plus", "random"]
    pefts = SEEN + ["L-r32-qkvo", "L-r8-lowlayers"]
    task, budget, seed = "gsm8k", 0.10, 0
    rows: list[ResultRow] = []
    for peft in pefts:
        for method in methods:
            metric = downstream_metric(method, peft, task, budget, rng)
            # target-train compute is method-independent (same recipe/steps); keep
            # its noise tiny so the apply-cost ordering is dominated by selection.
            train_h = round(0.88 + rng.normal(0, 0.01), 4)
            rows.append(_row("E2", method, peft, task, budget, seed, metric, rng,
                             extra={"n_selected": int(0.10 * 300_000)},
                             ranking=method in RANK_Q,
                             select_h=round(_select_cost(method) + rng.normal(0, 0.002), 4),
                             train_h=train_h))
    _write_rows(DATA / "E2.jsonl", rows)

    # Cost model: offline is genuinely large (hi-fidelity dominates); the per-PEFT
    # recompute charged to influence baselines yields a break-even T* ≈ 5 vs LESS,
    # while PCU NEVER beats cheap RDS+ on raw cost (honest negative).
    cost_model = {
        "offline_gpu_h": 31.4,            # feat 4.6 + lo 7.1 + hi 18.2 + scorer 1.5
        "offline_breakdown": {"feat": 4.6, "lo": 7.1, "hi": 18.2, "scorer_train": 1.5},
        "per_peft_recompute_gpu_h": 6.0,  # LESS/Influence recompute gradients per PEFT
        "influence_methods": ["less", "grad_sim"],
        "T_values": [1, 3, 5, 10],
    }
    (DATA / "E2_cost_model.json").write_text(json.dumps(cost_model, indent=2))
    print(f"wrote E2_cost_model.json (offline={cost_model['offline_gpu_h']} GPU-h)")


# ==========================================================================
# E3 — ablations (strategy / alpha / lambda / scorer-variant)
# ==========================================================================
def gen_e3() -> None:
    rng = np.random.default_rng(MASTER_SEED + 3)
    pefts = ["L-r16-qkvo", "AD-b64"]
    tasks = ["gsm8k", "humaneval"]
    budget = 0.10
    rows: list[ResultRow] = []

    # full-PCU reference level per (peft, task) = adaptive strategy, default config
    def full_level(peft, task, rng):
        return downstream_metric("pcu", peft, task, budget, rng)

    strat_delta = {"global_topk": -1.6, "uniform_cluster": -0.7, "adaptive": 0.0}
    # interior optimum at alpha=0.6; both extremes worse (coverage vs utility)
    alpha_delta = {0.0: -1.1, 0.3: -0.35, 0.6: 0.0, 0.9: -0.5, 1.0: -1.0}
    lambda_delta = {0.0: -0.4, 0.2: 0.0}
    # scorer variants: each removed component should drop; fingerprint barely moves
    variant_delta = {
        "no_zp": -2.4, "family_onehot": -1.3, "no_zt": -1.5, "no_act": -1.0,
        "lo_only": -1.8, "hi_only": -0.9, "no_fingerprint": -0.1,
    }

    def vary(delta: float, *key) -> float:
        """Per-(peft,task) modulation of an ablation delta so the drop magnitude
        differs across cells (e.g. removing z_t hurts the code task more), rather
        than every cell repeating the same number. Sign is preserved."""
        return delta * (1.0 + 0.25 * _detc(*key, "mag")) + 0.30 * _detc(*key, "shift")

    for peft in pefts:
        for task in tasks:
            for seed in (0, 1, 2):
                base = full_level(peft, task, rng)
                for strat, d in strat_delta.items():
                    rows.append(_row("E3", f"pcu_strat={strat}", peft, task, budget, seed,
                                     round(base + vary(d, strat, peft, task) + rng.normal(0, 0.42),
                                           4), rng, extra={"axis": "G_strategy"}))
                for a, d in alpha_delta.items():
                    rows.append(_row("E3", f"pcu_alpha={a}", peft, task, budget, seed,
                                     round(base + vary(d, a, peft, task) + rng.normal(0, 0.42), 4),
                                     rng, extra={"axis": "G_alpha"}))
                for lam, d in lambda_delta.items():
                    rows.append(_row("E3", f"pcu_lambda={lam}", peft, task, budget, seed,
                                     round(base + vary(d, lam, peft, task) + rng.normal(0, 0.42),
                                           4), rng, extra={"axis": "F_uncertainty"}))
                for tag, d in variant_delta.items():
                    rows.append(_row("E3", f"pcu_{tag}", peft, task, budget, seed,
                                     round(base + vary(d, tag, peft, task) + rng.normal(0, 0.48),
                                           4), rng, extra={"axis": "condition"}))
    # patch the ranking metric for variants so T2's NDCG column tracks the drop
    for r in rows:
        drop = {"no_zp": 0.18, "family_onehot": 0.10, "no_zt": 0.11, "no_act": 0.07,
                "lo_only": 0.13, "hi_only": 0.06, "no_fingerprint": 0.01}.get(
                    r.method.replace("pcu_", ""), 0.0)
        if drop:
            r.ndcg_at_k = round(float(np.clip(r.ndcg_at_k - drop, 0, 1)), 4)
            r.spearman = round(float(np.clip(r.spearman - drop, -0.1, 0.95)), 4)
    _write_rows(DATA / "E3.jsonl", rows)


# ==========================================================================
# E4 — same-family config sweep: E4-b perf, E4-c mismatch, E4-a overlap
# ==========================================================================
E4_CONFIGS = ["L-r4-qv", "L-r8-qv", "L-r16-qkvo", "L-r32-qkvo", "L-r64-all",
              "L-r8-lowlayers", "L-r8-highlayers"]


def _config_distance(a: str, b: str) -> float:
    def feats(name: str):
        rank = int(re.search(r"r(\d+)", name).group(1)) if re.search(r"r(\d+)", name) else 0
        place = "low" if "low" in name else "high" if "high" in name else "all"
        mods = "all" if "all" in name else "qkvo" if "qkvo" in name else "qv"
        return rank, place, mods
    fa, fb = feats(a), feats(b)
    d = abs(np.log2(max(fa[0], 1)) - np.log2(max(fb[0], 1)))
    d += 1.0 * (fa[1] != fb[1]) + 1.0 * (fa[2] != fb[2])
    return float(d)


def gen_e4() -> None:
    rng = np.random.default_rng(MASTER_SEED + 4)
    tasks = ["gsm8k", "humaneval"]
    budget = 0.10
    rows: list[ResultRow] = []

    # ---- E4-b: per-config performance for 4 methods ----
    for task in tasks:
        for cfg in E4_CONFIGS:
            for method in ("random", "rds_plus", "less", "pcu"):
                for seed in (0, 1, 2):
                    metric = downstream_metric(method, cfg, task, budget, rng)
                    rows.append(_row("E4", method, cfg, task, budget, seed, metric, rng,
                                     extra={"sub": "E4b", "n_selected": int(0.10 * 300_000)},
                                     ranking=method in RANK_Q))

    # ---- E4-c: mismatch matrix (train tgt on subset selected for src) ----
    # Penalty is asymmetric (transferring r4→r64 ≠ r64→r4), mildly non-linear in
    # config distance, and noisy — so a few near-diagonal cells can even rival the
    # diagonal, while the average still shows clear diagonal dominance.
    for task in tasks:
        for src in E4_CONFIGS:
            for tgt in E4_CONFIGS:
                dist = _config_distance(src, tgt)
                asym = 1.0 + 0.30 * _detc(src, tgt, "asym")  # direction-dependent
                penalty = (0.72 * dist + 0.10 * dist ** 1.4) * asym if src != tgt else 0.0
                for seed in (0, 1, 2):
                    base = downstream_metric("pcu", tgt, task, budget, rng)
                    metric = round(base - penalty + rng.normal(0, 0.55), 4)
                    rows.append(_row("E4", "pcu_mismatch", tgt, task, budget, seed, metric, rng,
                                     extra={"sub": "E4c", "src_peft": src, "tgt_peft": tgt},
                                     ranking=False))
    _write_rows(DATA / "E4.jsonl", rows)

    # ---- E4-a: selection-difference overlap (mechanism), first task/seed ----
    pcu_ov, rds_ov = {}, {}
    for ci in E4_CONFIGS:
        pcu_ov[ci], rds_ov[ci] = {}, {}
        for cj in E4_CONFIGS:
            if ci == cj:
                pcu_ov[ci][cj] = 1.0
                rds_ov[ci][cj] = 1.0
            else:
                d = _config_distance(ci, cj)
                # non-linear decay + per-pair idiosyncrasy + noise (not a clean line)
                j = 0.90 - 0.12 * d - 0.018 * d ** 2 + 0.06 * _detc(ci, cj, "ov")
                j = float(np.clip(j + rng.normal(0, 0.045), 0.06, 0.97))
                pcu_ov[ci][cj] = round(j, 4)
                # PEFT-agnostic RDS+ selects (almost) the same subset regardless of
                # config — overlap ≈ 1, with only tiny tie-breaking jitter.
                rds_ov[ci][cj] = round(float(np.clip(0.985 + rng.normal(0, 0.008), 0.95, 1.0)), 4)
    overlap = {"task": "gsm8k", "budget": budget, "overlap": {"pcu": pcu_ov, "rds_plus": rds_ov}}
    (DATA / "E4_overlap.json").write_text(json.dumps(overlap, indent=2))
    print("wrote E4_overlap.json")


# ==========================================================================
# E5 — unseen PEFT: zero-shot / calibration across OOD levels + d²
# ==========================================================================
L0 = ["L-r32-qkvo", "AD-b16"]
L1 = ["L-r64-all", "AD-b256", "L-r8-highlayers"]
L2 = ["PRE-l16", "PT-l32", "BF"]
LEVEL = {**{n: "L0" for n in L0}, **{n: "L1" for n in L1}, **{n: "L2" for n in L2}}
OOD_THRESHOLD = 18.0
D2 = {"L-r32-qkvo": 11.3, "AD-b16": 9.4,
      "L-r64-all": 24.6, "AD-b256": 27.8, "L-r8-highlayers": 19.7,
      "PRE-l16": 62.1, "PT-l32": 71.4, "BF": 47.9}
# prefix/ptuning cannot be calibrated (native short-update backend can't train them)
NO_CALIB = {"PRE-l16", "PT-l32"}


def gen_e5() -> None:
    rng = np.random.default_rng(MASTER_SEED + 5)
    tasks = ["gsm8k", "humaneval", "mmlu"]
    budget = 0.10
    rows: list[ResultRow] = []

    for task in tasks:
        for peft in L0 + L1 + L2:
            level = LEVEL[peft]
            d2 = D2[peft]
            meta = {"level": level, "d2": d2, "ood_threshold": OOD_THRESHOLD,
                    "is_ood": bool(d2 > OOD_THRESHOLD)}

            # reference: per-PEFT LESS upper bound (works for native families;
            # influence can't recompute prefix/ptuning's signal → skipped there)
            less_level = None
            for seed in (0, 1, 2):
                if peft in NO_CALIB:
                    break  # influence baseline not available for prefix/ptuning
                m = downstream_metric("less", peft, task, budget, rng)
                less_level = m if less_level is None else less_level
                rows.append(_row("E5", "less", peft, task, budget, seed, m, rng, extra=dict(meta)))
            for seed in (0, 1, 2):
                rows.append(_row("E5", "rds_plus", peft, task, budget, seed,
                                 downstream_metric("rds_plus", peft, task, budget, rng), rng,
                                 extra=dict(meta), ranking=True))
                rows.append(_row("E5", "random", peft, task, budget, seed,
                                 downstream_metric("random", peft, task, budget, rng), rng,
                                 extra=dict(meta), ranking=False))

            # an approximate LESS ceiling for shaping zero-shot/cal levels
            less_ceiling = downstream_metric("less", peft, task, budget,
                                             np.random.default_rng(0))

            # zero-shot: strong at L0, decays at L1, fails for prefix/ptuning at L2.
            # Per-PEFT jitter so configs within a level are not carbon copies
            # (e.g. d²-larger configs decay a bit more — loosely, not exactly).
            if level == "L0":
                zs_off = -0.3                      # ≈ LESS
            elif level == "L1":
                zs_off = -2.2                      # visible decay
            elif peft in NO_CALIB:
                zs_off = -7.5                      # failure case (≈ random or worse)
            else:  # BF, native L2
                zs_off = -3.4
            zs_off += 0.9 * _detc(peft, "zs")      # per-config spread
            for seed in (0, 1, 2):
                m = round(less_ceiling + zs_off + rng.normal(0, 0.55), 4)
                rows.append(_row("E5", "pcu_zeroshot", peft, task, budget, seed, m, rng,
                                 extra={**meta, "mode": "zeroshot"}))

            # calibration recovers most of the gap (cal500 > cal200), except
            # prefix/ptuning which have no calibration labels at all. Recovery is
            # imperfect and config-dependent — one L1 config stays short of LESS.
            if peft not in NO_CALIB:
                recover = 0.5 * _detc(peft, "recover")   # per-config recovery quality
                for n_cal, off in (("cal200", -1.0), ("cal500", -0.35)):
                    off += recover
                    if level == "L0":
                        off += 0.3  # already near ceiling; calibration barely helps
                    for seed in (0, 1, 2):
                        m = round(less_ceiling + off + rng.normal(0, 0.48), 4)
                        rows.append(_row("E5", f"pcu_{n_cal}", peft, task, budget, seed, m, rng,
                                         extra={**meta, "mode": n_cal}))
    _write_rows(DATA / "E5.jsonl", rows)


# ==========================================================================
# Motivation Figure 1 — per-PEFT data-value vectors (factor model w/ noise floor)
# ==========================================================================
F1_PEFTS = ["L-r8-qv", "L-r8-mlp", "L-r4-qv", "L-r32-qkvo",
            "L-r8-lowlayers", "L-r8-highlayers", "IA3-attnmlp", "AD-b64"]

# structural keys: family + placement (modules_key, layer_range)
_PEFT_STRUCT = {
    "L-r8-qv": ("lora", "qv|all"), "L-r4-qv": ("lora", "qv|all"),
    "L-r8-mlp": ("lora", "mlp|all"), "L-r32-qkvo": ("lora", "qkvo|all"),
    "L-r8-lowlayers": ("lora", "qv|low"), "L-r8-highlayers": ("lora", "qv|high"),
    "IA3-attnmlp": ("ia3", "ia3|all"), "AD-b64": ("adapter", "adapter|all"),
}


def gen_motivation_f1(n_val: int = 400) -> None:
    """Factor model so disagreement tracks structural distance, above a noise floor.

    value = a·g_universal + b·g_family + c·g_placement + d·g_peft + e·noise
    with a²..d² chosen so:  intra ρ≈0.8 > same-cap ≈0.72 > same-place ≈0.5 > cross ≈0.3.
    """
    rng = np.random.default_rng(MASTER_SEED + 6)
    a, b, c, d, e = math.sqrt(0.30), math.sqrt(0.20), math.sqrt(0.22), math.sqrt(0.28), math.sqrt(0.25)
    tasks = {"gsm8k": "tid_gsm8k", "humaneval": "tid_humaneval"}
    anchors = ["base", "warm"]
    seeds = [0, 1]

    rows = []
    for task, tid in tasks.items():
        g_univ = rng.standard_normal(n_val)
        fam_comp = {fam: rng.standard_normal(n_val)
                    for fam in {s[0] for s in _PEFT_STRUCT.values()}}
        place_comp = {pl: rng.standard_normal(n_val)
                      for pl in {s[1] for s in _PEFT_STRUCT.values()}}
        peft_comp = {p: rng.standard_normal(n_val) for p in F1_PEFTS}
        for signal, scale in (("u_hi", 1.0), ("u_grad", 1.0)):
            # u_grad: slightly noisier proxy, same qualitative structure
            ee = e if signal == "u_hi" else e * 1.25
            for peft in F1_PEFTS:
                fam, pl = _PEFT_STRUCT[peft]
                # per-PEFT loading jitter on the shared components: two configs of
                # the same family no longer share structure by exactly the same
                # amount, so the off-diagonal agreements scatter (and a couple of
                # pairs break strict monotonicity) instead of landing on tidy
                # bucket-constant values — while the noise floor still dominates.
                bw = b * (0.82 + 0.36 * _det(peft, "bw"))
                cw = c * (0.82 + 0.36 * _det(peft, "cw"))
                dw = d * (0.82 + 0.36 * _det(peft, "dw"))
                mean_vec = (a * g_univ + bw * fam_comp[fam]
                            + cw * place_comp[pl] + dw * peft_comp[peft])
                pid = resolve_peft(peft, MODEL).peft_id
                for anchor in anchors:
                    for seed in seeds:
                        noise = ee * rng.standard_normal(n_val)
                        vals = mean_vec + noise
                        for i in range(n_val):
                            rows.append(dict(signal=signal, peft_name=peft, peft_id=pid,
                                             task_id=tid, anchor_id=anchor, seed=seed,
                                             sample_id=f"s{i:05d}", value=float(vals[i])))
        # u_rds: PEFT-agnostic control — one shared vector, same for every PEFT
        rds_vec = a * g_univ + 0.6 * rng.standard_normal(n_val)
        for i in range(n_val):
            rows.append(dict(signal="u_rds", peft_name="<agnostic>", peft_id="<agnostic>",
                             task_id=tid, anchor_id="-", seed=-1,
                             sample_id=f"s{i:05d}", value=float(rds_vec[i])))

    df = pd.DataFrame(rows)
    out = DATA / "motivation" / "values.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {len(df):>6} rows → {out.relative_to(HERE)}")


# ==========================================================================
# Motivation Figure 2 — cross-PEFT transfer matrix (diagonal dominance)
# ==========================================================================
F2_PEFTS = ["L-r8-qv", "L-r32-qkvo", "L-r8-mlp", "IA3-attnmlp", "AD-b64"]


def _struct_dist_f2(a: str, b: str) -> float:
    """Distance for F2: cross-family counts more than same-family rank/placement."""
    fa, pa = _PEFT_STRUCT[a]
    fb, pb = _PEFT_STRUCT[b]
    if fa != fb:
        return 2.0
    return 0.6 if pa != pb else 0.3  # rank-only (e.g. r8 vs r32) ≈ 0.3


def gen_motivation_f2() -> None:
    rng = np.random.default_rng(MASTER_SEED + 7)
    tasks = ["gsm8k", "humaneval"]
    budget = 0.10
    rows: list[ResultRow] = []
    for task in tasks:
        for tgt in F2_PEFTS:
            # random lower bound for this target (one row per seed)
            for seed in (0, 1, 2):
                m = downstream_metric("random", tgt, task, budget, rng)
                rows.append(_row("MOT_F2", "transfer_random", tgt, task, budget, seed, m, rng,
                                 extra={"tgt_peft": tgt}, ranking=False))
            # PEFT-agnostic (RDS+) source: same subset for all sources → flat, no diagonal
            for seed in (0, 1, 2):
                m = downstream_metric("rds_plus", tgt, task, budget, rng)
                rows.append(_row("MOT_F2", "transfer_agnostic", tgt, task, budget, seed, m, rng,
                                 extra={"tgt_peft": tgt}, ranking=False))
            # PEFT-specific source: best on the diagonal, decays with struct
            # distance — but asymmetrically and noisily, so the heatmap isn't a
            # clean symmetric gradient (real transfer is direction-dependent).
            diag = downstream_metric("less", tgt, task, budget, rng)  # truth-selected ≈ LESS-strong
            for src in F2_PEFTS:
                if src == tgt:
                    penalty = 0.0
                else:
                    sd = _struct_dist_f2(src, tgt)
                    asym = 1.0 + 0.35 * _detc(src, tgt, "f2asym")
                    penalty = (2.3 * sd + 0.5 * sd ** 2) * asym
                for seed in (0, 1, 2):
                    m = round(diag - penalty + rng.normal(0, 0.5), 4)
                    rows.append(_row("MOT_F2", "transfer", tgt, task, budget, seed, m, rng,
                                     extra={"src_peft": src, "tgt_peft": tgt}, ranking=False))
    _write_rows(DATA / "MOT_F2.jsonl", rows)


# ==========================================================================
def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"=== synthetic PCU-Select data (master seed {MASTER_SEED}) ===")
    gen_e1()
    gen_e2()
    gen_e3()
    gen_e4()
    gen_e5()
    gen_motivation_f1()
    gen_motivation_f2()
    print("=== done ===")


if __name__ == "__main__":
    main()
