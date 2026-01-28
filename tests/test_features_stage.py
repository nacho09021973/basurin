import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_features_stage_builds_features_json(tmp_path: Path) -> None:
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn no disponible en el entorno de test")
    run_id = "features-stage"
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
                    {"id": "e", "features": [5.0, 6.0]},
                    {"id": "f", "features": [6.0, 7.0]},
                    {"id": "g", "features": [7.0, 8.0]},
                    {"id": "h", "features": [8.0, 9.0]},
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
            "3",
            "--out-root",
            "runs",
        ],
        cwd=tmp_path,
        check=True,
    )

    features_path = tmp_path / "runs" / run_id / "features" / "outputs" / "features.json"
    payload = json.loads(features_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1"
    assert payload["feature_key"] == "tangentes_locales_v1"
    assert "ids" in payload
    assert "Y" in payload
    assert "X_path" in payload
    assert "Y_path" in payload
    assert "shapes" in payload
    assert len(payload["ids"]) == len(payload["Y"])
