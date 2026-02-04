"""
tests/test_ringdown_real_ringdown_window_v1_stage.py
----------------------------------------------------
Tests for stages/ringdown_real_ringdown_window_v1_stage.py
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


def _write_window_npz(
    path: Path,
    strain: np.ndarray,
    gps_start: float,
    duration_s: float,
    fs_hz: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        strain=strain.astype(float),
        gps_start=float(gps_start),
        duration_s=float(duration_s),
        sample_rate_hz=float(fs_hz),
    )


def test_window_v1_aborts_without_run_valid(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__window_v1_no_run_valid"
    runs_root = tmp_path / "runs"
    (runs_root / run_id).mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    res = subprocess.run(
        [
            "python",
            "stages/ringdown_real_ringdown_window_v1_stage.py",
            "--run",
            run_id,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert res.returncode != 0


def test_window_v1_crops_by_gps_anchor(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__window_v1_crop"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    fs_hz = 4.0
    strain = np.arange(40, dtype=float)
    gps_start = 100.0
    duration_s = 10.0
    t0_gps = 102.0
    dt_start = 0.5
    out_duration = 1.0

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_window_npz(
        inputs_dir / "H1_rd.npz",
        strain=strain,
        gps_start=gps_start,
        duration_s=duration_s,
        fs_hz=fs_hz,
    )
    _write_window_npz(
        inputs_dir / "L1_rd.npz",
        strain=strain,
        gps_start=gps_start,
        duration_s=duration_s,
        fs_hz=fs_hz,
    )
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": t0_gps})

    observables_path = (
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl"
    )
    observables_path.parent.mkdir(parents=True, exist_ok=True)
    observables_path.write_text(json.dumps({"t0_gps": t0_gps}) + "\n", encoding="utf-8")

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    res = subprocess.run(
        [
            "python",
            "stages/ringdown_real_ringdown_window_v1_stage.py",
            "--run",
            run_id,
            "--dt-start-s",
            str(dt_start),
            "--duration-s",
            str(out_duration),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert res.returncode == 0, res.stderr

    outputs_dir = run_dir / "ringdown_real_ringdown_window_v1" / "outputs"
    out_npz = np.load(outputs_dir / "H1_rd.npz")
    expected_start = int(round((t0_gps + dt_start - gps_start) * fs_hz))
    expected_end = expected_start + int(round(out_duration * fs_hz))

    np.testing.assert_allclose(out_npz["strain"], strain[expected_start:expected_end])
    assert float(out_npz["gps_start"]) == t0_gps + dt_start
    assert float(out_npz["duration_s"]) == out_duration
    assert float(out_npz["sample_rate_hz"]) == fs_hz
    assert (outputs_dir / "segments_rd.json").exists()


def test_observables_features_inference_accept_window_stage_param(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__window_stage_param"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    fs_hz = 1024.0
    n_samples = 1024
    t = np.arange(n_samples, dtype=float) / fs_hz
    strain = np.cos(2.0 * np.pi * 200.0 * t)
    gps_start = 100.0
    duration_s = n_samples / fs_hz
    t0_gps = 100.0

    inputs_dir = run_dir / "ringdown_real_ringdown_window_v1" / "outputs"
    _write_window_npz(
        inputs_dir / "H1_rd.npz",
        strain=strain,
        gps_start=gps_start,
        duration_s=duration_s,
        fs_hz=fs_hz,
    )
    _write_window_npz(
        inputs_dir / "L1_rd.npz",
        strain=strain,
        gps_start=gps_start,
        duration_s=duration_s,
        fs_hz=fs_hz,
    )
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": t0_gps})

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}

    res_obs = subprocess.run(
        [
            "python",
            "stages/ringdown_real_observables_v0_stage.py",
            "--run",
            run_id,
            "--window-stage",
            "ringdown_real_ringdown_window_v1",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert res_obs.returncode == 0, res_obs.stderr

    res_feat = subprocess.run(
        [
            "python",
            "stages/ringdown_real_features_v0_stage.py",
            "--run",
            run_id,
            "--window-stage",
            "ringdown_real_ringdown_window_v1",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert res_feat.returncode == 0, res_feat.stderr

    res_inf = subprocess.run(
        [
            "python",
            "stages/ringdown_real_inference_v0_stage.py",
            "--run",
            run_id,
            "--window-stage",
            "ringdown_real_ringdown_window_v1",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert res_inf.returncode == 0, res_inf.stderr
