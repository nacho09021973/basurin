"""
tests/test_ringdown_real_observables_v0_stage.py
-------------------------------------------------
Minimal tests for stages/ringdown_real_observables_v0_stage.py
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


def _write_rd_npz(path: Path, n_samples: int = 1024, fs: float = 4096.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_samples, dtype=float) / fs
    strain = np.cos(2.0 * np.pi * 200.0 * t)
    np.savez(path, strain=strain.astype(float), t=t.astype(float), fs=fs)


def test_stage_real_observables_aborts_without_run_valid(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__real_obs_no_run_valid"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_observables_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode != 0


def test_stage_real_observables_writes_contract_files(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__real_obs_ok"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_rd_npz(inputs_dir / "H1_rd.npz")
    _write_rd_npz(inputs_dir / "L1_rd.npz")
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": 1126259462.4})

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_observables_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, res.stderr

    stage_dir = run_dir / "ringdown_real_observables_v0"
    outputs_dir = stage_dir / "outputs"

    assert (outputs_dir / "observables.jsonl").exists()
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("hashes"), "manifest hashes missing"
    for value in manifest["hashes"].values():
        assert value

    inputs_list = manifest.get("inputs", [])
    assert inputs_list
    for item in inputs_list:
        assert item.get("sha256") is not None

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary.get("verdict") == "PASS"


def test_basurin_where_ringdown_exp08_reports_real_observables_missing(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__where_exp08"
    run_dir = tmp_path / "runs" / run_id

    _write_run_valid_pass(run_dir)
    _write_json(
        run_dir / "ringdown_real_v0" / "outputs" / "real_v0_events_list.json",
        [{"event_id": "GW150914", "strain_npz": "cases/GW150914/strain.npz"}],
    )

    res = subprocess.run(
        [
            "python",
            "tools/basurin_where.py",
            "--run",
            run_id,
            "--out-root",
            str(tmp_path / "runs"),
            "--ringdown-exp08",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 2
    assert "ringdown_real_observables_v0" in res.stdout
    assert "missing: ringdown_real_observables_v0/outputs/observables.jsonl" in res.stdout
