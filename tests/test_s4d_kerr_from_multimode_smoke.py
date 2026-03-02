import json
import os
import subprocess
from pathlib import Path


def test_s4d_smoke_minimal(tmp_path: Path) -> None:
    # Use isolated runs root
    runs_root = tmp_path / "runs"
    os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)

    run_id = "s4d_smoke_pytest"
    run_dir = runs_root / run_id
    (run_dir / "RUN_VALID").mkdir(parents=True)
    (run_dir / "RUN_VALID" / "verdict.json").write_text('{"verdict":"PASS"}\n', encoding="utf-8")

    # Minimal required upstream input
    s3b_out = run_dir / "s3b_multimode_estimates" / "outputs"
    s3b_out.mkdir(parents=True)
    multimode = {
        "estimates": {
            "per_mode": {
                "220": {"f_hz": {"p10": 150.0, "p50": 160.0, "p90": 170.0}, "tau_s": {"p10": 0.003, "p50": 0.004, "p90": 0.005}},
                "221": {"f_hz": {"p10": 240.0, "p50": 250.0, "p90": 260.0}, "tau_s": {"p10": 0.002, "p50": 0.0025, "p90": 0.003}},
            }
        },
        "modes": [
            {"label": "220", "Sigma": [[0.04, 0.01], [0.01, 0.09]]},
            {"label": "221", "Sigma": [[0.04, 0.01], [0.01, 0.09]]},
        ]
    }
    (s3b_out / "multimode_estimates.json").write_text(json.dumps(multimode, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "s3b_multimode_estimates" / "stage_summary.json").write_text(
        json.dumps({
            "stage": "s3b_multimode_estimates",
            "multimode_viability": {
                "class": "MULTIMODE_OK",
                "reasons": [],
                "metrics": {"boundary_fraction": None, "valid_fraction": {"220": 1.0, "221": 1.0}},
            },
        }),
        encoding="utf-8",
    )

    # Run stage as module (captures canonical init/finalize behavior)
    cp = subprocess.run(
        ["python", "-m", "mvp.s4d_kerr_from_multimode", "--run-id", run_id],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert cp.returncode == 0, f"stdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"

    stage_dir = run_dir / "s4d_kerr_from_multimode"
    out_dir = stage_dir / "outputs"
    assert (out_dir / "kerr_from_multimode.json").exists()
    assert (out_dir / "kerr_from_multimode_diagnostics.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (stage_dir / "manifest.json").exists()

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    params = summary.get("parameters", {})
    assert params.get("A_MAX") == 0.99999
    assert params.get("GRID_A_SIZE") == 200
    assert params.get("GRID_M_SIZE") == 200
    assert params.get("gate", {}).get("name") == "KERR_GRID_SATURATION"
    assert params.get("n_accepted", 0) > 0
    assert "boundary_fraction" in params
