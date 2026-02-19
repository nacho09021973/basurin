from __future__ import annotations

import os
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


def test_single_help_exposes_atlas_default_flag():
    proc = subprocess.run(
        [sys.executable, str(PIPELINE), "single", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )

    assert proc.returncode == 0
    assert "--atlas-default" in proc.stdout


def test_single_atlas_default_missing_aborts_with_instruction(tmp_path):
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(tmp_path / "runs")

    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE),
            "single",
            "--event-id",
            "GW150914",
            "--atlas-default",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        check=False,
    )

    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode != 0
    assert "Atlas not found. Generate it by running: python mvp/generate_atlas_from_fits.py" in combined
    assert not (tmp_path / "runs").exists()
