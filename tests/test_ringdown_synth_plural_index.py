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


def test_ringdown_synth_writes_plural_index(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    run_id = "plural_index_run"
    env = _env_for(tmp_path)

    proc = subprocess.run(
        [
            PY,
            "stages/ringdown_synth_stage.py",
            "--run",
            run_id,
            "--seed",
            "123",
            "--snr",
            "11.0",
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
    assert proc.returncode == 0, proc.stderr

    outputs_dir = tmp_path / "runs" / run_id / "ringdown_synth" / "outputs"
    event_path = outputs_dir / "synthetic_event.json"
    index_path = outputs_dir / "synthetic_events.json"

    assert event_path.exists()
    assert index_path.exists()

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["n_events"] == 1
    assert index_payload["events"][0]["path"] == "synthetic_event.json"
    assert "truth" in index_payload["events"][0]
    assert index_payload["events"][0]["snr_target"] == 11.0

    # Contract: synthetic_events_list.json must exist and be a list
    list_path = outputs_dir / "synthetic_events_list.json"
    assert list_path.exists(), "synthetic_events_list.json not created"
    list_payload = json.loads(list_path.read_text(encoding="utf-8"))
    assert isinstance(list_payload, list), "synthetic_events_list.json must be a list"
    assert len(list_payload) == 1
    assert list_payload == index_payload["events"]

    # Contract: strain.npz must exist and be referenced in events
    strain_path = outputs_dir / "strain.npz"
    assert strain_path.exists(), "strain.npz not created in non-batch mode"
    assert list_payload[0].get("strain_npz") == "strain.npz", "strain_npz missing or wrong in events"

    # Contract: manifest.json must register strain_npz
    manifest_path = outputs_dir.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "strain_npz" in manifest.get("files", {}), "strain_npz not in manifest.files"
    assert manifest["files"]["strain_npz"] == "outputs/strain.npz"
