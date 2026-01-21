import json
import subprocess
import sys
from pathlib import Path


def test_spectrum_stage_summary_includes_geometry_inputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "run-inputs"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "01_genera_ads_puro.py"),
            "--run",
            run_id,
            "--n",
            "50",
        ],
        check=True,
        cwd=tmp_path,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "03_sturm_liouville.py"),
            "--run",
            run_id,
            "--mode",
            "fixed_mass",
            "--n-modes",
            "2",
        ],
        check=True,
        cwd=tmp_path,
    )

    summary_path = tmp_path / "runs" / run_id / "spectrum" / "stage_summary.json"
    summary = json.loads(summary_path.read_text())
    inputs = summary.get("inputs")
    assert inputs is not None
    assert inputs.get("geometry_path")
    assert inputs.get("geometry_sha256")
