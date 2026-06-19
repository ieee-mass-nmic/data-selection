"""Shared helpers for the figure scripts (design §9 figure templates).

Every plot script reads the flat ResultRow JSONL written by the runners, so the
figures never touch heavy artifacts. matplotlib is an optional dependency:
    pip install -e ".[viz]"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pcu_select.experiments import read_results

# A stable, colorblind-friendly order/label for the methods we plot repeatedly.
METHOD_LABELS = {
    "random": "Random", "balanced_random": "Balanced-Random", "length": "Length",
    "loss": "Loss", "perplexity": "PPL", "ifd": "IFD", "s2l": "S2L",
    "embedding_nn": "Emb-NN", "rds_plus": "RDS+", "diversity": "Diversity",
    "grad_sim": "Influence", "less": "LESS", "pcu": "PCU-Select (ours)",
}


def load_df(results_path: Path | str) -> pd.DataFrame:
    rows = read_results(results_path)
    if not rows:
        raise SystemExit(f"no results in {results_path} — run the experiment first.")
    df = pd.DataFrame([r.__dict__ for r in rows])
    # explode commonly-used extra fields into columns for convenience
    for key in ("axis", "sub", "src_peft", "tgt_peft", "level", "d2", "mode",
                "is_ood", "n_selected"):
        df[key] = df["extra"].map(lambda e, k=key: e.get(k))
    return df


def savefig(fig, out_dir: Path | str, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"wrote {path}")
    return path


def method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m)
