from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn no disponible en el entorno de test",
)
def test_features_stage_schema_wrapper(tmp_path: Path) -> None:
    run_id = "features-schema"
    repo_root = Path(__file__).resolve().parents[1]

    atlas_points_dir = tmp_path / "runs" / run_id / "dictionary" / "outputs"
    atlas_points_dir.mkdir(parents=True)
    atlas_points_path = atlas_points_dir / "atlas_points.json"
    atlas_points_path.write_text(
        json.dumps(
            {
                "feature_key": "ratios",
                "points": [
                    {"id": "a", "features": [1.0, 2.0]},
                    {"id": "b", "features": [2.0, 3.0]},
                    {"id": "c", "features": [3.0, 4.0]},
                    {"id": "d", "features": [4.0, 5.0]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "05_build_features_stage.py"),
            "--run",
            run_id,
            "--k-neighbors",
            "2",
            "--out-root",
            "runs",
        ],
        cwd=tmp_path,
        check=True,
    )

    features_path = tmp_path / "runs" / run_id / "features" / "outputs" / "features.json"
    payload_text = features_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)

    assert "metadata" in payload
    assert "features" in payload
    metadata = payload["metadata"]
    assert metadata["schema_version"] == "1.0"
    assert metadata["run"] == run_id
    assert metadata["created_utc"]
    assert "/home/" not in payload_text


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn no disponible en el entorno de test",
)
def test_features_stage_integrates_with_hsc_detector(tmp_path: Path) -> None:
    run_id = "features-hsc"
    repo_root = Path(__file__).resolve().parents[1]

    atlas_points_dir = tmp_path / "runs" / run_id / "dictionary" / "outputs"
    atlas_points_dir.mkdir(parents=True)
    atlas_points_path = atlas_points_dir / "atlas_points.json"
    atlas_points_path.write_text(
        json.dumps(
            {
                "feature_key": "ratios",
                "points": [
                    {"id": "a", "features": [1.0, 2.0]},
                    {"id": "b", "features": [2.0, 3.0]},
                    {"id": "c", "features": [3.0, 4.0]},
                    {"id": "d", "features": [4.0, 5.0]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "05_build_features_stage.py"),
            "--run",
            run_id,
            "--k-neighbors",
            "2",
            "--out-root",
            "runs",
        ],
        cwd=tmp_path,
        check=True,
    )

    run_dir = tmp_path / "runs" / run_id
    run_valid_dir = run_dir / "RUN_VALID" / "outputs"
    run_valid_dir.mkdir(parents=True, exist_ok=True)
    run_valid_payload = {"run": run_id, "verdict": "PASS"}
    (run_valid_dir / "run_valid.json").write_text(
        json.dumps(run_valid_payload, indent=2),
        encoding="utf-8",
    )

    features_path = run_dir / "features" / "outputs" / "features.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiment.hsc_detector.stage_hsc_detector",
            "--run",
            run_id,
            "--input",
            str(features_path),
            "--out-root",
            str(tmp_path / "runs"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={**os.environ, "BASURIN_RUNS_ROOT": str(tmp_path / "runs")},
        check=False,
    )

    assert result.returncode == 0, (
        "hsc_detector failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert (
        run_dir / "experiment" / "hsc_detector" / "outputs" / "report.json"
    ).exists()
