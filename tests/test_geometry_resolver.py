import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basurin_io import resolve_geometry_path


def test_resolve_geometry_path_canonical_and_legacy(tmp_path, monkeypatch):
    run_id = "run-001"
    geometry_path_expected = tmp_path / "runs" / run_id / "geometry" / "outputs" / "ads_puro.h5"
    geometry_path_expected.parent.mkdir(parents=True)
    geometry_path_expected.write_text("geometry")
    monkeypatch.chdir(tmp_path)

    resolved, geometry_path, input_geometry_absolute, geometry_resolution = resolve_geometry_path(
        run_id, "ads_puro.h5"
    )

    assert resolved.resolve() == geometry_path_expected.resolve()
    assert geometry_path == f"runs/{run_id}/geometry/outputs/ads_puro.h5"
    assert input_geometry_absolute is None
    assert geometry_resolution == "canonical"

    run_id_legacy = "run-002"
    legacy_path_expected = tmp_path / "runs" / run_id_legacy / "geometry" / "ads_puro.h5"
    legacy_path_expected.parent.mkdir(parents=True)
    legacy_path_expected.write_text("geometry")

    resolved, geometry_path, input_geometry_absolute, geometry_resolution = resolve_geometry_path(
        run_id_legacy, "ads_puro.h5"
    )

    assert resolved.resolve() == legacy_path_expected.resolve()
    assert geometry_path == f"runs/{run_id_legacy}/geometry/ads_puro.h5"
    assert input_geometry_absolute is None
    assert geometry_resolution == "legacy"


def test_geometry_rejects_parent_traversal(tmp_path, monkeypatch):
    run_id = "run-003"
    base_dir = tmp_path / "runs" / run_id / "geometry"
    base_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=r"no puede contener '\.\.'"):
        resolve_geometry_path(run_id, "../escape.h5")
