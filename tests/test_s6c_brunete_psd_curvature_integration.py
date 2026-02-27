from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from basurin_io import sha256_file, write_json_atomic

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_stage(run_id: str, runs_root: Path) -> subprocess.CompletedProcess[str]:
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
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)


def test_s6c_brunete_integration_lite_tmp_runs_root(tmp_path: Path) -> None:
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

    proc = _run_stage(run_id=run_id, runs_root=runs_root)
    assert proc.returncode == 0, proc.stderr

    stage_dir = run_dir / "s6c_brunete_psd_curvature"
    outputs_dir = stage_dir / "outputs"
    metrics = outputs_dir / "brunete_metrics.json"
    deriv = outputs_dir / "psd_derivatives.json"
    summary = stage_dir / "stage_summary.json"
    manifest = stage_dir / "manifest.json"

    for p in (metrics, deriv, summary, manifest):
        assert p.exists(), f"missing {p}"

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["hashes"]["brunete_metrics"] == sha256_file(metrics)
    assert manifest_payload["hashes"]["psd_derivatives"] == sha256_file(deriv)

    repo_runs_dir = REPO_ROOT / "runs" / run_id
    assert not repo_runs_dir.exists(), "stage wrote outside BASURIN_RUNS_ROOT"
