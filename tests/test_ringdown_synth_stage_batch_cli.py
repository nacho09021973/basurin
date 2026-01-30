from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PY = os.environ.get("PYTHON", "python")


def _env_for(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(tmp_path / "runs")
    return env


def test_ringdown_synth_stage_batch_cli(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    run_id = "batch_cli_run"
    batch = {"snr_grid": [10.0], "seeds": [7]}
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "stages/ringdown_synth_stage.py",
            "--run",
            run_id,
            "--batch-json",
            str(batch_path),
            "--out-root",
            "runs",
            "--f-220",
            "240.0",
            "--tau-220",
            "0.004",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr

    stage_dir = tmp_path / "runs" / run_id / "ringdown_synth"
    outputs_dir = stage_dir / "outputs"

    assert (outputs_dir / "synthetic_events.json").exists()
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()

    strain_files = list((outputs_dir / "cases").rglob("strain.npz"))
    assert strain_files
