import json
import os
import subprocess
import sys
from pathlib import Path

from basurin_io import sha256_file


def _write_geometry_json(path: Path) -> None:
    payload = {
        "geometry_type": "ads_like_minimal",
        "d": 3,
        "L": 1.0,
        "z_min": 0.05,
        "z_max": 0.5,
        "notes": {"source": "pytest"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_sturm_liouville(repo_root: Path, run_id: str, cwd: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "03_sturm_liouville.py"),
            "--run",
            run_id,
            "--mode",
            "fixed_mass",
            "--n-modes",
            "2",
            "--n-z",
            "32",
        ],
        check=True,
        cwd=cwd,
        env={**os.environ, "BASURIN_RUNS_ROOT": str(cwd / "runs")},
    )


def test_spectrum_from_geometry_json_is_deterministic(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "run-geometry-json"
    geometry_path = tmp_path / "runs" / run_id / "geometry" / "outputs" / "geometry.json"
    _write_geometry_json(geometry_path)

    _run_sturm_liouville(repo_root, run_id, tmp_path)
    geometry_numeric = tmp_path / "runs" / run_id / "spectrum" / "inputs" / "geometry_numeric.json"
    spectrum_path = tmp_path / "runs" / run_id / "spectrum" / "outputs" / "spectrum.h5"

    assert geometry_numeric.exists()
    assert spectrum_path.exists()

    first_hash = sha256_file(geometry_numeric)
    _run_sturm_liouville(repo_root, run_id, tmp_path)
    second_hash = sha256_file(geometry_numeric)

    assert first_hash == second_hash
