from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_geometry_support_smoke_skips_without_runs_list(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runs_list = repo_root / "runs_50_ids.txt"
    if not runs_list.exists():
        pytest.skip("runs_50_ids.txt not available in this environment")

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(tmp_path / "isolated_runs_root")
    cmd = [
        sys.executable,
        str(repo_root / "experiment" / "analyze_geometry_support.py"),
        "--run-id",
        "geom_support_smoke",
        "--runs-ids",
        str(runs_list),
        "--atlas-path",
        str(repo_root / "docs" / "ringdown" / "atlas" / "atlas_real_v1_s4.json"),
    ]
    res = subprocess.run(cmd, env=env, cwd=repo_root, capture_output=True, text=True)
    # This smoke only validates optional availability. If source runs are absent under temp BASURIN_RUNS_ROOT,
    # the script should fail with actionable upstream path hints.
    assert res.returncode != 0
    assert "Expected:" in (res.stderr + res.stdout)
