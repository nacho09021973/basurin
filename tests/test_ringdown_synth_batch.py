from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

PY = os.environ.get("PYTHON", "python")


def _env_for(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(tmp_path / "runs")
    return env


def test_ringdown_synth_stage_batch_outputs(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    run_id = "batch_run"
    batch = {"snr_grid": [8.0, 12.0], "seeds": [1, 2]}
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
            "--f-220",
            "250.0",
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
    events_path = outputs_dir / "synthetic_events.json"
    manifest_path = stage_dir / "manifest.json"
    summary_path = stage_dir / "stage_summary.json"

    assert outputs_dir.exists()
    assert events_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert isinstance(events, list)
    assert len(events) == 4

    sample = events[0]
    case_id = sample["case_id"]
    strain_path = outputs_dir / "cases" / case_id / "strain.npz"
    assert strain_path.exists()
    npz = np.load(strain_path)
    assert np.allclose(npz["strain"], [float(sample["seed"]), float(sample["snr_target"])])
