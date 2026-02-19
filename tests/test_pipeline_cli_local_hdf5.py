from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE = REPO_ROOT / "mvp" / "pipeline.py"


def test_single_help_exposes_local_hdf5_flag():
    proc = subprocess.run(
        [sys.executable, str(PIPELINE), "single", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )

    assert proc.returncode == 0
    assert "--local-hdf5" in proc.stdout
    assert "unrecognized arguments: --local-hdf5" not in proc.stderr
