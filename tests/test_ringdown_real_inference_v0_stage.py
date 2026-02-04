"""
tests/test_ringdown_real_inference_v0_stage.py
----------------------------------------------
Minimal tests for stages/ringdown_real_inference_v0_stage.py
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


def _write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def _write_run_valid_pass(run_dir: Path) -> None:
    _write_json(
        run_dir / "RUN_VALID" / "outputs" / "run_valid.json",
        {"overall_verdict": "PASS"},
    )


def _write_rd_npz(path: Path, strain: np.ndarray, fs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strain=strain.astype(float), fs=fs)


def test_stage_real_inference_aborts_without_run_valid(tmp_path: Path) -> None:
    run_id = "2040-09-01__unit_test__real_inference_no_run_valid"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_inference_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode != 0


def test_stage_real_inference_recovers_f_peak_and_tau_on_synthetic_decay(
    tmp_path: Path,
) -> None:
    run_id = "2040-09-01__unit_test__real_inference_synth"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    fs = 4096.0
    n_samples = 8192
    f_hz = 250.0
    tau_s = 0.25
    t = np.arange(n_samples, dtype=float) / fs
    strain = np.exp(-t / tau_s) * np.sin(2.0 * np.pi * f_hz * t)

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_rd_npz(inputs_dir / "H1_rd.npz", strain, fs)
    _write_rd_npz(inputs_dir / "L1_rd.npz", strain, fs)
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": 1126259462.4})

    observables = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "detectors": ["H1", "L1"],
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "rms": {"H1": 1.0, "L1": 1.0},
        "peak_abs": {"H1": 1.0, "L1": 1.0},
    }
    features = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "duration_s": {"H1": n_samples / fs, "L1": n_samples / fs},
        "snr_proxy": {"H1": 10.0, "L1": 10.0},
    }

    _write_jsonl(
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl",
        observables,
    )
    _write_jsonl(
        run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        features,
    )

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_inference_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    h1_fit = report["fit"]["H1"]
    assert abs(h1_fit["f_peak_hz"] - f_hz) < 5.0
    assert h1_fit["tau_s"] is not None
    assert 0.15 < h1_fit["tau_s"] < 0.4


def test_stage_real_inference_reports_window_duration_from_npz(
    tmp_path: Path,
) -> None:
    run_id = "2040-09-01__unit_test__real_inference_window"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    fs = 4096.0
    duration_s = 0.25
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples, dtype=float) / fs
    strain = np.sin(2.0 * np.pi * 220.0 * t)

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    _write_rd_npz(inputs_dir / "H1_rd.npz", strain, fs)
    _write_rd_npz(inputs_dir / "L1_rd.npz", strain, fs)
    _write_json(inputs_dir / "segments_rd.json", {"t0_gps": 1126259462.4})

    observables = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "detectors": ["H1", "L1"],
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "rms": {"H1": 1.0, "L1": 1.0},
        "peak_abs": {"H1": 1.0, "L1": 1.0},
    }
    features = {
        "run_id": run_id,
        "t0_gps": 1126259462.4,
        "fs_hz": fs,
        "n_samples": {"H1": n_samples, "L1": n_samples},
        "duration_s": {"H1": duration_s, "L1": duration_s},
        "snr_proxy": {"H1": 10.0, "L1": 10.0},
    }

    _write_jsonl(
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl",
        observables,
    )
    _write_jsonl(
        run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        features,
    )

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "stages/ringdown_real_inference_v0_stage.py",
        "--run",
        run_id,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, res.stderr

    report_path = (
        run_dir / "ringdown_real_inference_v0" / "outputs" / "inference_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for det in ["H1", "L1"]:
        expected = report["fit"][det]["n_samples"] / report["fs_hz"]
        assert abs(report["window"]["duration_s"][det] - expected) < 1e-9

    assert "duration_s" not in report["features"]
    assert "features_duration_s" in report["features"]


def test_basurin_where_ringdown_exp08_reports_real_inference_missing(
    tmp_path: Path,
) -> None:
    run_id = "2040-09-01__unit_test__where_exp08_inference"
    run_dir = tmp_path / "runs" / run_id

    _write_run_valid_pass(run_dir)
    _write_json(
        run_dir / "ringdown_real_v0" / "outputs" / "real_v0_events_list.json",
        [{"event_id": "GW150914", "strain_npz": "cases/GW150914/strain.npz"}],
    )

    _write_jsonl(
        run_dir / "ringdown_real_observables_v0" / "outputs" / "observables.jsonl",
        {"detectors": ["H1", "L1"], "fs_hz": 4096.0, "n_samples": {"H1": 1, "L1": 1}},
    )
    _write_json(run_dir / "ringdown_real_observables_v0" / "stage_summary.json", {})
    _write_json(run_dir / "ringdown_real_observables_v0" / "manifest.json", {})

    _write_jsonl(
        run_dir / "ringdown_real_features_v0" / "outputs" / "features.jsonl",
        {"run_id": run_id, "fs_hz": 4096.0},
    )
    _write_json(run_dir / "ringdown_real_features_v0" / "stage_summary.json", {})
    _write_json(run_dir / "ringdown_real_features_v0" / "manifest.json", {})

    exp08_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke" / "outputs"
    )
    _write_json(exp08_dir / "real_v0_smoke_report.json", {"status": "ok"})
    _write_json(exp08_dir / "contract_verdict.json", {"verdict": "PASS"})
    (exp08_dir / "failure_catalog.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (exp08_dir / "failure_catalog.jsonl").write_text("\n", encoding="utf-8")

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
    assert "ringdown_real_inference_v0" in res.stdout
    assert (
        "missing: ringdown_real_inference_v0/outputs/inference_report.json" in res.stdout
    )
