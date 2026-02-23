from pathlib import Path

from mvp.s5_aggregate import _relpath_under


def test_relpath_under_returns_relative_path_inside_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    p = runs_root / "A" / "s4_geometry_filter" / "outputs" / "compatible_set.json"

    got = _relpath_under(runs_root, p)

    assert got == "A/s4_geometry_filter/outputs/compatible_set.json"
    assert not got.startswith("/")


def test_relpath_under_falls_back_to_absolute_outside_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    outside = tmp_path / "other" / "compatible_set.json"

    got = _relpath_under(runs_root, outside)

    assert got == outside.as_posix()
