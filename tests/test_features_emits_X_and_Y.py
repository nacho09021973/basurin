import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn no disponible en el entorno de test",
)
def test_features_emits_X_and_Y(tmp_path: Path) -> None:
    run_id = "features-emits-x-y"
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

    stage_dir = tmp_path / "runs" / run_id / "features"
    features_path = stage_dir / "outputs" / "features.json"
    manifest_path = stage_dir / "manifest.json"

    payload = json.loads(features_path.read_text(encoding="utf-8"))
    assert "X_path" in payload
    assert "Y_path" in payload
    assert payload["X_path"] == "X.npy"
    assert payload["Y_path"] == "Y.npy"
    assert payload["shapes"]["n"] == len(payload["ids"])

    x_path = features_path.parent / payload["X_path"]
    y_path = features_path.parent / payload["Y_path"]
    assert x_path.exists()
    assert y_path.exists()
    assert np.load(x_path).shape[0] == len(payload["ids"])
    assert np.load(y_path).shape[0] == len(payload["ids"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "X" in manifest["files"]
    assert "Y" in manifest["files"]
