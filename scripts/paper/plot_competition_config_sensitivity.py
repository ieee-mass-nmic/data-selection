#!/usr/bin/env python3
"""Regenerate Figure 5 and print its tie-aware descriptive statistics."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from scipy.stats import spearmanr

import generate_paper_assets as assets


ROOT = Path(__file__).resolve().parents[2]


def config_distance(left: str, right: str) -> float:
    def features(name: str) -> tuple[int, str, str]:
        match = re.search(r"r(\d+)", name)
        rank = int(match.group(1)) if match else 0
        placement = "low" if "low" in name else "high" if "high" in name else "all"
        modules = "all" if "all" in name else "qkvo" if "qkvo" in name else "qv"
        return rank, placement, modules

    a, b = features(left), features(right)
    rank_gap = abs(math.log2(max(a[0], 1)) - math.log2(max(b[0], 1)))
    return rank_gap + float(a[1] != b[1]) + float(a[2] != b[2])


def main() -> None:
    payload = json.loads((ROOT / "result" / "data" / "E4_overlap.json").read_text())
    matrix = payload["overlap"]["pcu"]
    keys = list(matrix)
    distances: list[float] = []
    differences: list[float] = []
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            distances.append(config_distance(left, right))
            differences.append(1.0 - float(matrix[left][right]))
    rho = float(spearmanr(distances, differences).statistic)

    assets.setup_style()
    assets.fig_config_sensitivity()
    print(f"Figure 5: n={len(distances)} unordered pairs; Spearman rho={rho:.6f}")


if __name__ == "__main__":
    main()
