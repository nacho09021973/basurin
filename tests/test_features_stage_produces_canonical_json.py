import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from basurin_io import load_feature_json


def test_features_stage_produces_canonical_json(tmp_path: Path) -> None:
    if importlib.util.find_spec("sklearn") is None:
        import pytest

        pytest.skip("scikit-learn no disponible en el entorno de test")
    run_id = "features-stage-canonical"
    repo_root = Path(__file__).resolve().parents[1]

    atlas_points_dir = tmp_path / "runs" / run_id / "dictionary" / "outputs"
    atlas_points_dir.mkdir(parents=True)
    atlas_points_path = atlas_points_dir / "atlas_points.json"
    atlas_points_path.write_text(
        json.dumps(
            {
                "feature_key": "ratios",
                "points": [
                    {"id": "p0", "features": [0.0, 1.0]},
                    {"id": "p1", "features": [1.0, 2.0]},
                    {"id": "p2", "features": [2.0, 3.0]},
                    {"id": "p3", "features": [3.0, 4.0]},
                    {"id": "p4", "features": [4.0, 5.0]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "05_build_features_stage.py"),
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
    ids, Y, meta = load_feature_json(features_path, kind="features")
    payload = json.loads(features_path.read_text(encoding="utf-8"))

    assert ids == ["p0", "p1", "p2", "p3", "p4"]
    assert np.asarray(Y).shape == (5, 6)
    assert meta["schema_version"] == "1"
    assert meta["feature_key"] == "tangentes_locales_v1"
    assert payload["X_path"] == "X.npy"
    assert payload["Y_path"] == "Y.npy"
    assert payload["shapes"]["n"] == 5
