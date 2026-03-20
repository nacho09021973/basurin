from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _help_text(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "mvp/pipeline.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_s3b_args_subset_of_pipeline_multimode() -> None:
    s3b_source = (REPO_ROOT / "mvp" / "s3b_multimode_estimates.py").read_text(encoding="utf-8")
    multimode_help = _help_text("multimode", "--help")

    scientific_subset = {
        "--n-bootstrap",
        "--seed",
        "--psd-path",
        "--method",
        "--bootstrap-221-residual-strategy",
        "--mode-221-topology",
        "--band-strategy",
        "--max-lnf-span-220",
        "--max-lnq-span-220",
        "--max-lnf-span-221",
        "--max-lnq-span-221",
        "--min-valid-fraction-221",
        "--cv-threshold-221",
    }

    for flag in scientific_subset:
        assert flag in s3b_source

    pipeline_equivalents = {
        "--n-bootstrap": "--s3b-n-bootstrap",
        "--seed": "--s3b-seed",
        "--psd-path": "--psd-path",
        "--method": "--s3b-method",
        "--bootstrap-221-residual-strategy": "--bootstrap-221-residual-strategy",
        "--mode-221-topology": "--mode-221-topology",
        "--band-strategy": "--band-strategy",
        "--max-lnf-span-220": "--max-lnf-span-220",
        "--max-lnq-span-220": "--max-lnq-span-220",
        "--max-lnf-span-221": "--max-lnf-span-221",
        "--max-lnq-span-221": "--max-lnq-span-221",
        "--min-valid-fraction-221": "--min-valid-fraction-221",
        "--cv-threshold-221": "--cv-threshold-221",
    }

    missing = [
        pipeline_flag
        for pipeline_flag in pipeline_equivalents.values()
        if pipeline_flag not in multimode_help
    ]
    assert missing == []


def test_s4_new_args_in_pipeline() -> None:
    multimode_help = _help_text("multimode", "--help")
    assert "--threshold-mode" in multimode_help
    assert "--delta-lnL" in multimode_help
    assert "--informative-threshold" in multimode_help


def test_single_help_includes_s4_threshold_args() -> None:
    single_help = _help_text("single", "--help")
    assert "--threshold-mode" in single_help
    assert "--delta-lnL" in single_help
    assert "--informative-threshold" in single_help


def test_pipeline_help_includes_all_scientific_knobs() -> None:
    multi_help = _help_text("multi", "--help")
    assert "threshold-mode" in multi_help
    assert "delta-lnL" in multi_help
    assert "informative-threshold" in multi_help
