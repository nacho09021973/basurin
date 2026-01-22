import json
import os
import subprocess
import sys
from pathlib import Path


def test_boundary_generation_from_geometry(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "01_generate_boundary_from_geometry.py"

    geometry_path = tmp_path / "runs" / "demo_run" / "geometry" / "outputs" / "geometry.json"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_payload = {
        "geometry_type": "explicit_bulk",
        "dimension": 4,
        "boundary_dimension": 3,
        "details": {"shape": "toy"},
    }
    geometry_path.write_text(json.dumps(geometry_payload, indent=2), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(script), "--run", "demo_run"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        check=True,
    )

    boundary_path = tmp_path / "runs" / "demo_run" / "boundary" / "outputs" / "boundary_data.json"
    summary_path = tmp_path / "runs" / "demo_run" / "boundary" / "stage_summary.json"
    manifest_path = tmp_path / "runs" / "demo_run" / "boundary" / "manifest.json"

    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert boundary["source"] == "explicit_geometry"
    assert boundary["geometry"] == geometry_payload
    assert boundary["boundary"]["point_dimension"] == 3
    assert boundary["boundary"]["points"] == [[0.0, 0.0, 0.0]]

    assert summary["stage"] == "boundary"
    assert summary["run"] == "demo_run"
    assert summary["config"]["geometry_type"] == "explicit_bulk"
    assert summary["config"]["dimension"] == 4
    assert summary["results"]["boundary_generated"] is True
    assert summary["results"]["source"] == "explicit_geometry"
    assert "asserted geometry" in " ".join(summary["notes"])

    assert manifest["stage"] == "boundary"
    assert "boundary_data" in manifest["files"]
