import sys
from pathlib import Path

import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "03_sturm_liouville.py"
spec = importlib.util.spec_from_file_location("sturm_liouville", MODULE_PATH)
sturm_liouville = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["sturm_liouville"] = sturm_liouville
spec.loader.exec_module(sturm_liouville)
resolve_geometry_path = sturm_liouville.resolve_geometry_path


def test_geometry_basename_resolves_under_run(tmp_path, monkeypatch):
    run_id = "run-001"
    geometry_path_expected = tmp_path / "runs" / run_id / "geometry" / "ads_puro.h5"
    geometry_path_expected.parent.mkdir(parents=True)
    geometry_path_expected.write_text("geometry")
    monkeypatch.chdir(tmp_path)

    resolved, geometry_path, input_geometry_absolute, geometry_resolution = resolve_geometry_path(
        run_id, "ads_puro.h5"
    )

    assert resolved.resolve() == geometry_path_expected.resolve()
    assert geometry_path == f"runs/{run_id}/geometry/ads_puro.h5"
    assert input_geometry_absolute is None
    assert geometry_resolution == "legacy"


def test_geometry_runs_prefix_does_not_duplicate(tmp_path, monkeypatch):
    run_id = "run-002"
    geometry_path_expected = tmp_path / "runs" / run_id / "geometry" / "ads_puro.h5"
    geometry_path_expected.parent.mkdir(parents=True)
    geometry_path_expected.write_text("geometry")
    monkeypatch.chdir(tmp_path)

    resolved, geometry_path, input_geometry_absolute, geometry_resolution = resolve_geometry_path(
        run_id, f"runs/{run_id}/geometry/ads_puro.h5"
    )

    assert resolved.resolve() == geometry_path_expected.resolve()
    assert geometry_path == f"runs/{run_id}/geometry/ads_puro.h5"
    assert input_geometry_absolute is None
    assert geometry_resolution == "absolute"


def test_geometry_rejects_parent_traversal(tmp_path, monkeypatch):
    run_id = "run-003"
    base_dir = tmp_path / "runs" / run_id / "geometry"
    base_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=r"no puede contener '\.\.'"):
        resolve_geometry_path(run_id, "../escape.h5")
