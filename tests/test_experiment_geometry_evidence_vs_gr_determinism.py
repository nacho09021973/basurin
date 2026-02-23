from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("mvp/experiment/geometry_evidence_vs_gr.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_script(repo_root: Path, run_id: str, runs_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    return subprocess.run(
        [sys.executable, str(repo_root / SCRIPT), "--run-id", run_id],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_output_is_stable_across_reruns(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "exp_geom_det"

    _write_json(
        runs_root / run_id / "RUN_VALID" / "verdict.json",
        {"schema_version": "run_valid_v1", "verdict": "PASS", "reason": "test"},
    )
    _write_json(
        runs_root / run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        {
            "schema_version": "mvp_compatible_set_v1",
            "compatible_geometries": [
                {"geometry_id": "GR", "d_conformal": 0.4, "d_conformal_3d": 0.6},
                {"geometry_id": "ALT1", "d_conformal": 0.7, "d_conformal_3d": 0.2},
            ],
        },
    )
    _write_json(
        runs_root / run_id / "s6_information_geometry" / "outputs" / "curvature.json",
        {"schema_version": "curvature_v1", "kappa": 1.0},
    )

    first_run = _run_script(repo_root, run_id, runs_root)
    assert first_run.returncode == 0, first_run.stderr + first_run.stdout

    output_path = runs_root / run_id / "experiment_geometry_evidence_vs_gr" / "outputs" / "evidence_vs_gr.json"
    first_hash = _sha256(output_path)

    second_run = _run_script(repo_root, run_id, runs_root)
    assert second_run.returncode == 0, second_run.stderr + second_run.stdout

    second_hash = _sha256(output_path)
    assert first_hash == second_hash

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "experiment_geometry_evidence_vs_gr_v1"
    assert "created" not in payload
