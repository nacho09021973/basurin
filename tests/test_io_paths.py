from pathlib import Path

import pytest

from basurin_io import resolve_spectrum_path


def test_resolve_spectrum_path_prefers_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    outputs = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    legacy = run_dir / "spectrum" / "spectrum.h5"
    outputs.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    outputs.write_text("new")
    legacy.write_text("legacy")

    assert resolve_spectrum_path(run_dir) == outputs


def test_resolve_spectrum_path_falls_back_to_legacy(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r2"
    legacy = run_dir / "spectrum" / "spectrum.h5"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy")

    assert resolve_spectrum_path(run_dir) == legacy


def test_resolve_spectrum_path_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r3"

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_spectrum_path(run_dir)

    message = str(excinfo.value)
    assert "outputs/spectrum.h5" in message
    assert "spectrum/spectrum.h5" in message
