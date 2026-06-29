"""Lightweight CLI smoke tests for script entrypoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pcu_select.experiments import PEFT_REGISTRY, resolve_peft
from pcu_select.peft_space.schema import load_peft_config

ROOT = Path(__file__).resolve().parents[1]

HELP_SCRIPTS = (
    "scripts/build_features.py",
    "scripts/encode_task.py",
    "scripts/compute_lo_fidelity.py",
    "scripts/build_hi_fidelity.py",
    "scripts/train_scorer.py",
    "scripts/select_subset.py",
    "scripts/experiments/run_e1.py",
    "scripts/experiments/run_e2.py",
    "scripts/experiments/run_e3.py",
    "scripts/experiments/run_e4.py",
    "scripts/experiments/run_e5.py",
    "scripts/experiments/dump_peft_registry.py",
    "scripts/experiments/build_calib_labels.py",
    "scripts/experiments/build_table1.py",
    "scripts/experiments/build_motivation_values.py",
    "scripts/experiments/run_motivation_transfer.py",
)


@pytest.mark.parametrize("script", HELP_SCRIPTS)
def test_script_help_loads(script: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "usage:" in proc.stdout


def test_build_table1_writes_rows(tmp_path: Path) -> None:
    """Table 1 is zero-GPU and analytic — exercise it end to end."""
    import csv

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/experiments/build_table1.py"),
            "--model", "llama2-7b",
            "--pefts", "L-r8-qv", "IA3-attnmlp", "AD-b64",
            "--out-dir", str(tmp_path),
        ],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    rows = list(csv.DictReader((tmp_path / "table1.csv").open()))
    assert [r["peft"] for r in rows] == ["L-r8-qv", "IA3-attnmlp", "AD-b64"]
    # trainable-param counts must span a wide range (IA3 ≪ Adapter), the whole point.
    counts = {r["peft"]: int(r["n_trainable"]) for r in rows}
    assert counts["IA3-attnmlp"] < counts["L-r8-qv"] < counts["AD-b64"]
    assert (tmp_path / "table1.md").exists() and (tmp_path / "table1.tex").exists()


def test_dump_peft_registry_writes_loadable_configs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/experiments/dump_peft_registry.py"),
            "--model",
            "llama2-7b",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    dumped = sorted(tmp_path.glob("*/*.yaml"))
    assert len(dumped) == len(PEFT_REGISTRY)
    assert {p.stem for p in dumped} == set(PEFT_REGISTRY)

    for path in dumped:
        cfg = load_peft_config(path)
        expected = resolve_peft(path.stem, "llama2-7b")
        assert cfg.peft_id == expected.peft_id
        assert cfg.family == expected.family
        assert cfg.target_modules == expected.target_modules
        assert cfg.target_layers == expected.target_layers
