import json
import os
import subprocess
import sys
from pathlib import Path


def test_atlas_master_diagnostics(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "diag-run"
    master_id = "atlas-master-diag"

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "01_genera_neutrino_sandbox.py"),
            "--run",
            run_id,
            "--n-delta",
            "20",
            "--n-modes",
            "3",
            "--noise-rel",
            "0.0",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "04_diccionario.py"),
            "--run",
            run_id,
            "--k-features",
            "2",
            "--n-bootstrap",
            "0",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "06_build_atlas_master_stage.py"),
            "--run",
            master_id,
            "--runs",
            run_id,
            "--out-root",
            str(runs_root),
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    summary_path = runs_root / master_id / "atlas_master" / "stage_summary.json"
    summary = json.loads(summary_path.read_text())
    diagnostics = summary["atlas_master_diagnostics"]

    assert "atlas_master_diagnostics" in summary
    assert isinstance(diagnostics["X_explained_var_1"], float)
    assert 0.0 <= diagnostics["X_explained_var_1"] <= 1.0
    assert isinstance(diagnostics["X_singular_values"], list)
    assert diagnostics["X_singular_values"]
    assert isinstance(diagnostics["X_rank"], int)
    assert diagnostics["X_rank"] >= 1
    assert diagnostics["X_effective_dim"] in {1, 2, 3}
    assert diagnostics["X_pairwise_corr_max_offdiag"] >= 0.0
