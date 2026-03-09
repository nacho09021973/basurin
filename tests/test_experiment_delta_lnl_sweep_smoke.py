from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_delta_lnl_sweep_smoke(tmp_path: Path) -> None:
    runs_root = tmp_path / "custom_runs"
    run_id = "run_delta_sweep"
    run_dir = runs_root / run_id

    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _write_json(
        run_dir / "s3_ringdown_estimates" / "outputs" / "estimates.json",
        {
            "event_id": "GW_TEST",
            "combined": {"f_hz": 250.0, "Q": 10.0},
            "combined_uncertainty": {
                "sigma_logf": 0.1,
                "sigma_logQ": 0.1,
                "cov_logf_logQ": 0.0,
            },
        },
    )

    atlas_path = tmp_path / "atlas.json"
    _write_json(
        atlas_path,
        [
            {"geometry_id": "g220_a", "f_hz": 250.0, "Q": 10.0, "metadata": {"mode": [2, 2, 0]}},
            {"geometry_id": "g220_b", "f_hz": 350.0, "Q": 10.0, "metadata": {"mode": [2, 2, 0]}},
            {"geometry_id": "g221_a", "f_hz": 250.0, "Q": 10.0, "metadata": {"mode": [2, 2, 1]}},
        ],
    )

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp.experiment_delta_lnL_sweep",
            "--run-id",
            run_id,
            "--atlas-path",
            str(atlas_path),
            "--deltas",
            "0.0,1.0",
            "--modes",
            "(2,2,0);(2,2,1)",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    stage_dir = run_dir / "experiment" / "delta_lnL_sweep"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()

    aggregate_path = stage_dir / "outputs" / "delta_sweep.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    rows = aggregate["rows"]
    assert len(rows) == 4
    assert {row["mode"] for row in rows} == {"(2,2,0)", "(2,2,1)"}
    assert {row["delta"] for row in rows} == {0.0, 1.0}

    assert not (tmp_path / "runs").exists()
