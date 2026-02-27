from __future__ import annotations

import json
import math
import os
import subprocess

import pytest
import sys
from pathlib import Path

from basurin_io import sha256_file, write_json_atomic

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_stage(run_id: str, runs_root: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "mvp" / "s6c_brunete_psd_curvature.py"),
        "--run",
        run_id,
        "--c-window",
        "15.0",
        "--min-points",
        "5",
        "--sigma-switch",
        "0.1",
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)


def _prepare_run(tmp_path: Path) -> tuple[Path, str, Path]:
    runs_root = tmp_path / "det_runs"
    run_id = "it_s6c_brunete"
    run_dir = runs_root / run_id

    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    write_json_atomic(
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "schema_version": "mvp_estimates_v2",
            "event_id": "EVT_SYNTH",
            "per_detector": {
                "H1": {
                    "f_hz": 100.0,
                    "Q": 5.0,
                    "tau_s": 5.0 / (3.141592653589793 * 100.0),
                    "snr_peak": 12.0,
                }
            },
        },
    )

    freqs = [80.0 + i for i in range(41)]
    psd_vals = [f * f for f in freqs]
    write_json_atomic(
        run_dir / "external_inputs" / "psd_model.json",
        {
            "schema_version": "mvp_psd_model_v1",
            "models": {
                "H1": {
                    "frequencies_hz": freqs,
                    "psd_values": psd_vals,
                }
            },
        },
    )
    return runs_root, run_id, run_dir


def test_s6c_brunete_golden_schema_tmp_runs_root(tmp_path: Path) -> None:
    runs_root, run_id, run_dir = _prepare_run(tmp_path)
    proc = _run_stage(run_id=run_id, runs_root=runs_root)
    assert proc.returncode == 0, proc.stderr

    stage_dir = run_dir / "s6c_brunete_psd_curvature"
    outputs_dir = stage_dir / "outputs"
    metrics_path = outputs_dir / "brunete_metrics.json"
    deriv_path = outputs_dir / "psd_derivatives.json"

    assert metrics_path.exists()
    assert deriv_path.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    deriv_payload = json.loads(deriv_path.read_text(encoding="utf-8"))

    assert isinstance(metrics_payload.get("metrics"), list)
    assert isinstance(deriv_payload.get("derivatives"), list)
    assert len(metrics_payload["metrics"]) == 1
    assert len(deriv_payload["derivatives"]) == 1

    row = metrics_payload["metrics"][0]
    required_metrics = {
        "event_id": str,
        "mode": str,
        "f_hz": float,
        "s1": float,
        "kappa": float,
        "sigma": float,
        "chi_psd": float,
        "regime_sigma": str,
        "regime_chi_psd": str,
    }
    for key, expected_type in required_metrics.items():
        assert key in row
        assert isinstance(row[key], expected_type)

    assert "Q" in row or "tau_s" in row
    if row["kappa"] >= 0:
        assert row["sigma"] >= 0

    drow = deriv_payload["derivatives"][0]
    required_derivatives = {
        "method": str,
        "half_window_hz": float,
        "n_points": int,
        "s1": float,
        "kappa": float,
    }
    for key, expected_type in required_derivatives.items():
        assert key in drow
        assert isinstance(drow[key], expected_type)

    repo_runs_dir = REPO_ROOT / "runs" / run_id
    assert not repo_runs_dir.exists(), "stage wrote outside BASURIN_RUNS_ROOT"


def test_s6c_brunete_golden_manifest_hashes(tmp_path: Path) -> None:
    runs_root, run_id, run_dir = _prepare_run(tmp_path)
    proc = _run_stage(run_id=run_id, runs_root=runs_root)
    assert proc.returncode == 0, proc.stderr

    stage_dir = run_dir / "s6c_brunete_psd_curvature"
    manifest_path = stage_dir / "manifest.json"
    assert manifest_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest_payload.get("hashes", {})

    expected = {
        "brunete_metrics": stage_dir / "outputs" / "brunete_metrics.json",
        "psd_derivatives": stage_dir / "outputs" / "psd_derivatives.json",
        "stage_summary": stage_dir / "stage_summary.json",
    }
    for label, path in expected.items():
        assert path.exists(), f"missing {path}"
        assert label in hashes, f"missing hash for {label}"
        assert hashes[label] == sha256_file(path)


def test_s6c_detector_valueerror_insufficient_points_warns_and_passes(tmp_path: Path) -> None:
    runs_root = tmp_path / "det_runs"
    run_id = "it_s6c_brunete_detector_warning"
    run_dir = runs_root / run_id

    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    write_json_atomic(
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "schema_version": "mvp_estimates_v2",
            "event_id": "EVT_SYNTH",
            "per_detector": {
                "H1": {"f_hz": 120.0, "Q": 5.0, "tau_s": 0.01, "snr_peak": 12.0},
                "L1": {"f_hz": 100.0, "Q": 4.0, "tau_s": 0.01, "snr_peak": 10.0},
            },
        },
    )
    write_json_atomic(
        run_dir / "external_inputs" / "psd_model.json",
        {
            "schema_version": "mvp_psd_model_v1",
            "models": {
                "H1": {
                    "frequencies_hz": [100.0, 110.0],
                    "psd_values": [10000.0, 12100.0],
                },
                "L1": {
                    "frequencies_hz": [80.0 + i for i in range(41)],
                    "psd_values": [(80.0 + i) ** 2 for i in range(41)],
                },
            },
        },
    )

    proc = _run_stage(run_id=run_id, runs_root=runs_root)
    assert proc.returncode == 0, proc.stderr

    stage_dir = run_dir / "s6c_brunete_psd_curvature"
    summary_payload = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads(
        (stage_dir / "outputs" / "brunete_metrics.json").read_text(encoding="utf-8")
    )

    warning_records = summary_payload.get("warnings", [])
    assert warning_records == [
        {
            "detector": "H1",
            "code": "PSD_POLYFIT_INSUFFICIENT_POINTS",
            "detail": "Puntos insuficientes en ventana para ajuste grado 2: se requieren al menos 5, encontrados 1 (f0_hz=120.0, half_window_hz=15.0)",
        }
    ]
    assert [row["detector"] for row in metrics_payload["metrics"]] == ["H1", "L1"]
    l1 = [row for row in metrics_payload["metrics"] if row["detector"] == "L1"][0]
    assert "sigma" in l1 and "chi_psd" in l1


def test_s6c_dry_run_does_not_write_stage_outputs(tmp_path: Path) -> None:
    runs_root, run_id, run_dir = _prepare_run(tmp_path)

    proc = _run_stage(run_id=run_id, runs_root=runs_root, extra_args=["--dry-run"])
    assert proc.returncode == 0, proc.stderr
    assert "[dry-run] det=H1" in proc.stdout

    stage_dir = run_dir / "s6c_brunete_psd_curvature"
    assert not stage_dir.exists(), "dry-run must not create stage directory"


def test_s6c_gw150914_like_power_law_ranges(tmp_path: Path) -> None:
    runs_root = tmp_path / "det_runs"
    run_id = "it_s6c_gw150914_like"
    run_dir = runs_root / run_id

    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    write_json_atomic(
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "schema_version": "mvp_estimates_v2",
            "event_id": "GW150914_like",
            "per_detector": {
                "H1": {
                    "f_hz": 251.0,
                    "Q": 4.3,
                    "tau_s": 4.3 / (3.141592653589793 * 251.0),
                    "snr_peak": 12.0,
                }
            },
        },
    )

    freqs = [200.0 + 0.05 * i for i in range(2401)]
    psd_vals = [f ** (-4.0) for f in freqs]
    write_json_atomic(
        run_dir / "external_inputs" / "psd_model.json",
        {
            "schema_version": "mvp_psd_model_v1",
            "models": {
                "H1": {
                    "frequencies_hz": freqs,
                    "psd_values": psd_vals,
                }
            },
        },
    )

    proc = _run_stage(run_id=run_id, runs_root=runs_root)
    assert proc.returncode == 0, proc.stderr

    metrics_payload = json.loads(
        (run_dir / "s6c_brunete_psd_curvature" / "outputs" / "brunete_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    row = metrics_payload["metrics"][0]

    assert -0.05 <= row["sigma"] <= 0.05
    assert 0.01 <= row["chi_psd"] <= 0.15
    assert row["regime_sigma"] == "perturbative"
    assert math.isfinite(row["K"])
    assert row["K"] < 0.0


def test_s6c_closed_form_regime_with_strong_local_curvature(tmp_path: Path) -> None:
    runs_root = tmp_path / "det_runs"
    run_id = "it_s6c_closed_form_regime"
    run_dir = runs_root / run_id

    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    write_json_atomic(
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "schema_version": "mvp_estimates_v2",
            "event_id": "EVT_SPIKE",
            "per_detector": {
                "H1": {
                    "f_hz": 251.0,
                    "Q": 4.3,
                    "tau_s": 4.3 / (3.141592653589793 * 251.0),
                    "rho0": 10.0,
                }
            },
        },
    )

    f0 = 251.0
    freqs = [240.0 + 0.01 * i for i in range(2201)]
    psd_vals = [
        1.0 - 0.999 * math.exp(-((f - f0) / 0.08) ** 2)
        for f in freqs
    ]
    write_json_atomic(
        run_dir / "external_inputs" / "psd_model.json",
        {
            "schema_version": "mvp_psd_model_v1",
            "models": {
                "H1": {
                    "frequencies_hz": freqs,
                    "psd_values": psd_vals,
                }
            },
        },
    )

    proc = _run_stage(run_id=run_id, runs_root=runs_root)
    assert proc.returncode == 0, proc.stderr

    metrics_payload = json.loads(
        (run_dir / "s6c_brunete_psd_curvature" / "outputs" / "brunete_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    row = metrics_payload["metrics"][0]

    assert row["regime_sigma"] == "closed_form"
    assert row["regime_chi_psd"] == "elevated"
    assert math.isfinite(row["K"])
    assert math.isfinite(row["R"])
