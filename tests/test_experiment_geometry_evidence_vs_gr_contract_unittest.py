from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment/geometry_evidence_vs_gr.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_script(repo_root: Path, run_id: str, runs_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    return subprocess.run(
        [sys.executable, str(repo_root / SCRIPT), "--run-id", run_id, *extra],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_contract_registered() -> None:
    contract = CONTRACTS.get("experiment_geometry_evidence_vs_gr")
    assert contract is not None
    assert contract.required_inputs == [
        "s4_geometry_filter/outputs/compatible_set.json",
        "s6_information_geometry/outputs/curvature.json",
    ]
    assert contract.produced_outputs == ["outputs/evidence_vs_gr.json"]


def test_fail_when_missing_inputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "exp_geom_missing"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2, result.stderr + result.stdout


def test_happy_path_writes_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    run_id = "exp_geom_ok"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
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
    _write_json(
        runs_root / run_id / "s6b_information_geometry_3d" / "outputs" / "curvature_3d.json",
        {"schema_version": "curvature_3d_v1", "kappa": 0.5},
    )
    _write_json(
        runs_root / run_id / "s6b_information_geometry_3d" / "outputs" / "metric_diagnostics_3d.json",
        {"schema_version": "metric_diag_3d_v1"},
    )

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / "experiment_geometry_evidence_vs_gr"
    manifest = stage_dir / "manifest.json"
    summary = stage_dir / "stage_summary.json"
    output = stage_dir / "outputs" / "evidence_vs_gr.json"

    assert manifest.exists()
    assert summary.exists()
    assert output.exists()

    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["verdict"] == "PASS"
    assert summary_payload["inputs"]
    assert all(entry.get("sha256") for entry in summary_payload["inputs"])
    assert summary_payload["outputs"]
    assert all(entry.get("sha256") for entry in summary_payload["outputs"])

    output_payload = json.loads(output.read_text(encoding="utf-8"))
    assert output_payload["schema_version"] == "experiment_geometry_evidence_vs_gr_v1"
    assert output_payload["metrics"]["n_compatible"] == 2
    assert math.isfinite(output_payload["metrics"]["logB_proxy_alt_vs_GR"])
