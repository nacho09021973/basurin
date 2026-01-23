import json
from pathlib import Path

import numpy as np

from basurin_io import load_feature_json


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_feature_json_atlas_contract(tmp_path: Path) -> None:
    atlas_path = tmp_path / "atlas.json"
    payload = {
        "schema_version": "1",
        "feature_key": "ratios",
        "ids": ["a", "b"],
        "X": [[1.0, 2.0], [3.0, 4.0]],
        "meta": {"created": "2024-01-01T00:00:00Z"},
    }
    _write_json(atlas_path, payload)

    ids, X, meta = load_feature_json(atlas_path, kind="atlas")

    assert ids == ["a", "b"]
    assert np.asarray(X).shape == (2, 2)
    assert meta["feature_key"] == "ratios"
    assert meta["schema_version"] == "1"
    assert meta["columns"] == ["ratios_0", "ratios_1"]


def test_load_feature_json_features_contract(tmp_path: Path) -> None:
    features_path = tmp_path / "features.json"
    payload = {
        "schema_version": "1",
        "feature_key": "tangentes_locales_v1",
        "ids": ["a", "b", "c"],
        "Y": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "meta": {
            "feature_key": "tangentes_locales_v1",
            "columns": ["d_eff", "m"],
        },
    }
    _write_json(features_path, payload)

    ids, Y, meta = load_feature_json(features_path, kind="features")

    assert ids == ["a", "b", "c"]
    assert np.asarray(Y).shape == (3, 2)
    assert meta["feature_key"] == "tangentes_locales_v1"
    assert meta["columns"] == ["d_eff", "m"]
