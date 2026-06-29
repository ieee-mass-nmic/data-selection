"""Table 1 — PEFT configurations and trainable parameters (motivation §2).

Descriptive, **zero-GPU** by default: shows that PEFT is not a name but a set of
structurally distinct interventions — different inserted modules, operators,
layer placement and trainable-parameter counts (spanning ~two orders of
magnitude). This is the structural premise underneath Figure 1/2 and `z_p`.

Parameter counts come from the codebase's own `trainable_params_estimate`
(`peft_space.schema`) — the same source `z_p`'s capacity vector uses — so the
table can never drift from the encoder. Touched-site counts come from
`site_mask_of`. Pass `--from-model` to instead attach the native adapters to the
real backbone and report the exact `PeftHandle.num_trainable()` (loads a 7B
model; use it once to validate the analytic column).

Example:
    python scripts/experiments/build_table1.py --model llama2-7b --out-dir tables
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _motivation import APPROX_BACKBONE_PARAMS, TABLE1_PEFTS

from pcu_select.experiments import MODELS, PEFT_REGISTRY, resolve_peft
from pcu_select.peft_space.schema import trainable_params_estimate
from pcu_select.peft_space.site_mask import SiteSpace, operator_of, site_mask_of
from pcu_select.types import PEFTConfig
from pcu_select.utils import get_logger

_OPERATOR_LABEL = {
    "additive_low_rank": "add. low-rank",
    "multiplicative": "multiplicative",
    "additive_bottleneck": "add. bottleneck",
    "prefix": "prefix",
    "bias_shift": "bias shift",
}


def _layers_desc(cfg: PEFTConfig, n_layers: int) -> str:
    n = len(cfg.target_layers)
    if n == 0 or n == n_layers:
        return f"all ({n_layers})"
    third = max(1, n_layers // 3)
    if cfg.target_layers == list(range(0, third)):
        return f"low ⅓ ({n})"
    if cfg.target_layers == list(range(n_layers - third, n_layers)):
        return f"high ⅓ ({n})"
    if cfg.target_layers == list(range(third, 2 * third)):
        return f"mid ⅓ ({n})"
    return f"{n} layers"


def _capacity_desc(cfg: PEFTConfig) -> str:
    if cfg.family == "lora":
        return f"r={cfg.rank}"
    if cfg.family == "adapter":
        return f"b={cfg.adapter_bottleneck}"
    if cfg.family in ("prefix", "ptuning"):
        return f"len={cfg.prefix_len}"
    return "—"


def _exact_trainable(cfg: PEFTConfig, hf_id: str, device: str) -> int:
    """Attach the native adapter to the real backbone; return exact param count."""
    from pcu_select.eval.target_train import load_backbone
    from pcu_select.hi_fidelity.native_peft import attach_peft

    model, _ = load_backbone(hf_id, device=device)
    handle = attach_peft(model, cfg, seed=0)
    try:
        return handle.num_trainable()
    finally:
        handle.remove()


def _backbone_total(model_tag: str, from_model: bool, hf_id: str, device: str) -> float:
    if from_model:
        from pcu_select.eval.target_train import load_backbone

        model, _ = load_backbone(hf_id, device=device)
        return float(sum(p.numel() for p in model.parameters()))
    return APPROX_BACKBONE_PARAMS.get(model_tag, float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description="Build Table 1 (PEFT configs + trainable params)")
    p.add_argument("--model", type=str, default="llama2-7b",
                   help="Backbone tag from experiments.registry.MODELS.")
    p.add_argument("--pefts", type=str, nargs="+", default=TABLE1_PEFTS,
                   help="PEFT registry names (default: the motivation 8-config set).")
    p.add_argument("--from-model", action="store_true",
                   help="Attach to the real backbone for EXACT counts (loads a 7B model).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=Path, default=Path("tables"))
    args = p.parse_args()

    log = get_logger("build_table1")
    if args.model not in MODELS:
        raise SystemExit(f"unknown model {args.model!r}; known: {sorted(MODELS)}")
    spec = MODELS[args.model]
    n_layers = spec.n_layers
    sites = SiteSpace.uniform(n_layers_total=n_layers, k=8)
    total_backbone = _backbone_total(args.model, args.from_model, spec.hf_id, args.device)

    rows: list[dict] = []
    for name in args.pefts:
        cfg = resolve_peft(name, args.model)
        spec_reg = PEFT_REGISTRY[name]
        op = operator_of(cfg.family, cfg.target_modules[0] if cfg.target_modules else "")
        mask = site_mask_of(cfg, sites)
        touched = sum(1 for v in mask.values() if v > 0)
        if args.from_model:
            n_train = _exact_trainable(cfg, spec.hf_id, args.device)
        else:
            n_train = trainable_params_estimate(cfg)
        pct = 100.0 * n_train / total_backbone if total_backbone == total_backbone else float("nan")
        rows.append({
            "peft": name,
            "family": cfg.family,
            "inserted_into": ",".join(cfg.target_modules),
            "layers": _layers_desc(cfg, n_layers),
            "operator": _OPERATOR_LABEL.get(op, op),
            "capacity": _capacity_desc(cfg),
            "n_trainable": n_train,
            "pct_backbone": pct,
            "touched_sites": f"{touched}/{len(sites.all_sites)}",
            "group": spec_reg.group,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    count_kind = "exact" if args.from_model else "estimate"

    # ---- CSV ----
    csv_path = args.out_dir / "table1.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- Markdown ----
    md_lines = [
        f"# Table 1 — PEFT configurations and trainable parameters ({args.model}, "
        f"{n_layers} layers; #trainable = {count_kind})",
        "",
        "| PEFT | Family | Inserted into | Layers | Operator | Capacity | "
        "# Trainable | % backbone | Touched sites |Ω_p\\| |",
        "|---|---|---|---|---|---|--:|--:|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['peft']}` | {r['family']} | {r['inserted_into']} | {r['layers']} | "
            f"{r['operator']} | {r['capacity']} | {r['n_trainable']:,} | "
            f"{r['pct_backbone']:.3f}% | {r['touched_sites']} |"
        )
    md_path = args.out_dir / "table1.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    # ---- LaTeX (booktabs) ----
    tex = [
        r"\begin{tabular}{llllllrr l}",
        r"\toprule",
        r"PEFT & Family & Inserted into & Layers & Operator & Capacity & "
        r"\# Trainable & \% bb & Sites \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['peft']} & {r['family']} & {r['inserted_into'].replace('_', chr(92)+'_')} & "
            f"{r['layers']} & {r['operator']} & {r['capacity']} & {r['n_trainable']:,} & "
            f"{r['pct_backbone']:.3f} & {r['touched_sites']} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (args.out_dir / "table1.tex").write_text("\n".join(tex) + "\n")

    print("\n".join(md_lines))
    log.info(f"wrote table1.{{csv,md,tex}} → {args.out_dir} ({count_kind} counts)")


if __name__ == "__main__":
    main()
