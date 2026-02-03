"""
tests/test_ringdown_real_v0_stage.py
------------------------------------
Minimal tests for stages/ringdown_real_v0_stage.py

Verifies:
- Correct output paths (manifest.json, stage_summary.json)
- Contract failure when no real data is provided
- Proper handling of --dry-run mode
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_valid_pass(run_dir: Path) -> None:
    _write_json(
        run_dir / "RUN_VALID" / "outputs" / "run_valid.json",
        {"overall_verdict": "PASS"},
    )


def _write_strain_npz(path: Path, n_samples: int = 2048, fs: float = 4096.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_samples, dtype=float) / fs
    strain = np.sin(2.0 * np.pi * 100.0 * t) + 0.01 * np.random.randn(n_samples)
    np.savez(path, strain=strain.astype(float), t=t.astype(float), fs=fs)


def test_ringdown_real_v0_stage_missing_data_contract(tmp_path: Path) -> None:
    """Test that stage fails with categorized MISSING_REAL_DATA when no data is provided."""
    run_id = "2040-08-01__unit_test__real_v0_missing"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}"

    stage_dir = run_dir / "ringdown_real_v0"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()

    contract_path = stage_dir / "outputs" / "contract_verdict.json"
    assert contract_path.exists()

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["overall_verdict"] == "FAIL"

    data_contract = next(
        (c for c in contract["contracts"] if c["id"] == "REAL_V0_DATA_AVAILABLE"),
        None,
    )
    assert data_contract is not None
    assert data_contract["verdict"] == "FAIL"
    assert "missing_real_v0_inputs" in data_contract["violations"]


def test_ringdown_real_v0_stage_with_valid_data(tmp_path: Path) -> None:
    """Test that stage passes when valid real data is provided."""
    run_id = "2040-08-01__unit_test__real_v0_valid"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    data_source = tmp_path / "real_data"
    for event_id in ["GW150914", "GW170817"]:
        event_dir = data_source / event_id
        _write_strain_npz(event_dir / "strain.npz")

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_v0_stage.py",
        "--run",
        run_id,
        "--data-source-dir",
        str(data_source),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, f"Expected exit code 0, got {res.returncode}. stderr: {res.stderr}"

    stage_dir = run_dir / "ringdown_real_v0"
    outputs_dir = stage_dir / "outputs"

    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (outputs_dir / "real_v0_events_list.json").exists()
    assert (outputs_dir / "contract_verdict.json").exists()

    events_list = json.loads((outputs_dir / "real_v0_events_list.json").read_text(encoding="utf-8"))
    assert isinstance(events_list, list)
    assert len(events_list) == 2

    for event in events_list:
        assert "event_id" in event
        assert "strain_npz" in event
        strain_path = outputs_dir / event["strain_npz"]
        assert strain_path.exists()

    contract = json.loads((outputs_dir / "contract_verdict.json").read_text(encoding="utf-8"))
    assert contract["overall_verdict"] == "PASS"


def test_ringdown_real_v0_stage_dry_run(tmp_path: Path) -> None:
    """Test that --dry-run mode validates without writing full outputs."""
    run_id = "2040-08-01__unit_test__real_v0_dryrun"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_v0_stage.py",
        "--run",
        run_id,
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 2

    stage_dir = run_dir / "ringdown_real_v0"
    outputs_dir = stage_dir / "outputs"

    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (outputs_dir / "contract_verdict.json").exists()


def test_ringdown_real_v0_stage_invalid_strain_files(tmp_path: Path) -> None:
    """Test that stage handles invalid strain files gracefully."""
    run_id = "2040-08-01__unit_test__real_v0_invalid"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    data_source = tmp_path / "real_data"
    valid_dir = data_source / "valid_event"
    _write_strain_npz(valid_dir / "strain.npz")

    invalid_dir = data_source / "invalid_event"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    np.savez(invalid_dir / "strain.npz", other_data=np.array([1, 2, 3]))

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_v0_stage.py",
        "--run",
        run_id,
        "--data-source-dir",
        str(data_source),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0

    stage_dir = run_dir / "ringdown_real_v0"
    outputs_dir = stage_dir / "outputs"

    events_list = json.loads((outputs_dir / "real_v0_events_list.json").read_text(encoding="utf-8"))
    assert len(events_list) == 1
    assert events_list[0]["event_id"] == "valid_event"

    contract = json.loads((outputs_dir / "contract_verdict.json").read_text(encoding="utf-8"))

    io_contract = next(
        (c for c in contract["contracts"] if c["id"] == "REAL_V0_IO_VALID"),
        None,
    )
    assert io_contract is not None
    assert io_contract["verdict"] == "WARN"
    assert io_contract["metrics"]["n_invalid"] == 1
