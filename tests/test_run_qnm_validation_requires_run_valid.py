"""Ensure run_qnm_validation.sh fails fast without RUN_VALID."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PY = os.environ.get("PYTHON", "python")


def test_run_qnm_validation_requires_run_valid(tmp_path: Path) -> None:
    run_id = "test_missing_run_valid"

    result = subprocess.run(
        ["bash", "run_qnm_validation.sh", run_id],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "RUN_VALID" in result.stdout or "RUN_VALID" in result.stderr
