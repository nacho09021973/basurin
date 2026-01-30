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


def test_ringdown_synth_batch_contract(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    run_id = "batch_contract_run"
    batch = {"snr_grid": [9.0], "seeds": [3]}
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    env = _env_for(tmp_path)
    proc = subprocess.run(
        [
            PY,
            "stages/ringdown_synth_stage.py",
            "--run",
            run_id,
            "--batch-json",
            str(batch_path),
            "--f-220",
            "220.0",
            "--tau-220",
            "0.004",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    run_dir = tmp_path / "runs" / run_id
    events_path = run_dir / "ringdown_synth" / "outputs" / "synthetic_events.json"
    assert events_path.exists()

    events = json.loads(events_path.read_text(encoding="utf-8"))
    assert isinstance(events, list)
    assert events

    event = events[0]
    paths = event.get("paths")
    assert isinstance(paths, dict)
    strain_rel = paths.get("strain_npz")
    assert isinstance(strain_rel, str)
    assert not Path(strain_rel).is_absolute()

    strain_path = run_dir / strain_rel
    assert strain_path.exists()

    npz = np.load(strain_path)
    assert "t" in npz and "h" in npz
    t = npz["t"]
    h = npz["h"]
    assert t.ndim == 1
    assert h.ndim == 1
    assert t.shape == h.shape
    assert t.shape[0] >= 256
    assert np.issubdtype(t.dtype, np.number)
    assert np.issubdtype(h.dtype, np.number)
