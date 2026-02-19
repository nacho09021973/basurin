from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - env dependent
    raise unittest.SkipTest("integration requires numpy")


def _write_min_run(runs_root: Path, run_id: str) -> None:
    run_dir = runs_root / run_id
    rv_dir = run_dir / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    s2_dir = run_dir / "s2_ringdown_window"
    s2_out = s2_dir / "outputs"
    s2_out.mkdir(parents=True, exist_ok=True)

    fs = 4096.0
    t = np.arange(0, 0.25, 1.0 / fs)
    strain = np.exp(-t / 0.05) * np.sin(2.0 * np.pi * 220.0 * t)
    np.savez(s2_out / "H1_rd.npz", strain=strain.astype(np.float64), sample_rate_hz=np.array([fs]))

    (s2_out / "window_meta.json").write_text(
        json.dumps({"sample_rate_hz": fs, "t0_start_s": 0.0, "bandpass_hz": [150.0, 400.0]}),
        encoding="utf-8",
    )
    (s2_dir / "manifest.json").write_text(json.dumps({"artifacts": {"H1_rd": "outputs/H1_rd.npz"}}), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_experiment_t0_sweep_deterministic(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "test_t0_sweep"
    _write_min_run(runs_root, run_id)

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)

    cmd = [
        sys.executable,
        "mvp/experiment_t0_sweep.py",
        "--run-id",
        run_id,
        "--t0-start-ms",
        "0",
        "--t0-stop-ms",
        "10",
        "--t0-step-ms",
        "5",
        "--mode",
        "single",
    ]

    first = subprocess.run(cmd, cwd=repo, env=env, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr

    out_json = runs_root / run_id / "experiment" / "t0_sweep" / "outputs" / "t0_sweep_results.json"
    assert out_json.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "experiment_t0_sweep_v1"
    assert len(payload["points"]) == 3
    assert [p["t0_ms"] for p in payload["points"]] == [0, 5, 10]

    sha_first = _sha(out_json)
    second = subprocess.run(cmd, cwd=repo, env=env, check=False, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    sha_second = _sha(out_json)
    assert sha_first == sha_second
