"""
tests/test_ringdown_real_features_v0_stage.py
---------------------------------------------
Minimal tests for stages/ringdown_real_features_v0_stage.py
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


def _write_rd_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strain=np.zeros(8, dtype=float), fs=2.0)


def test_stage_real_features_aborts_without_run_valid(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__real_feat_no_run_valid"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_features_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode != 0


def test_stage_real_features_writes_contract_files(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__real_feat_ok"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    observables = {
        "run_id": run_id,
        "detectors": ["H1", "L1"],
        "fs_hz": 2.0,
        "n_samples": {"H1": 4, "L1": 2},
        "t0_gps": 123.4,
        "rms": {"H1": 0.5, "L1": 1.0},
        "peak_abs": {"H1": 1.0, "L1": 2.0},
    }
    observables_path = (
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl"
    )
    observables_path.parent.mkdir(parents=True, exist_ok=True)
    observables_path.write_text(json.dumps(observables) + "\n", encoding="utf-8")

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_rd_npz(inputs_dir / "H1_rd.npz")
    _write_rd_npz(inputs_dir / "L1_rd.npz")
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": 123.4})

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_features_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, res.stderr

    stage_dir = run_dir / "ringdown_real_features_v0"
    outputs_dir = stage_dir / "outputs"

    assert (outputs_dir / "features.jsonl").exists()
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary.get("verdict") == "PASS"

    features = json.loads(
        (outputs_dir / "features.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert features["duration_s"]["H1"] == 2.0
    assert features["duration_s"]["L1"] == 1.0
    assert features["snr_proxy"]["H1"] == 2.0
    assert features["snr_proxy"]["L1"] == 2.0


def test_basurin_where_ringdown_exp08_reports_real_features_missing(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__where_exp08_real_features"
    run_dir = tmp_path / "runs" / run_id

    _write_run_valid_pass(run_dir)
    _write_json(
        run_dir / "ringdown_real_v0" / "outputs" / "real_v0_events_list.json",
        [{"event_id": "GW150914", "strain_npz": "cases/GW150914/strain.npz"}],
    )
    obs_dir = run_dir / "ringdown_real_observables_v0"
    _write_json(obs_dir / "outputs" / "observables.jsonl", {"ok": True})
    _write_json(obs_dir / "stage_summary.json", {"verdict": "PASS"})
    _write_json(obs_dir / "manifest.json", {"hashes": {"outputs/observables.jsonl": "x"}})

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
    assert "ringdown_real_features_v0" in res.stdout
    assert "missing: ringdown_real_features_v0/outputs/features.jsonl" in res.stdout
