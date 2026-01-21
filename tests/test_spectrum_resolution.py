from pathlib import Path

from basurin_io import resolve_spectrum_path


def test_resolve_spectrum_prefers_outputs(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    out_dir = run_dir / "spectrum" / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "spectrum.h5").write_text("ok")
    expected = out_dir / "spectrum.h5"

    assert resolve_spectrum_path(run_dir) == expected


def test_resolve_spectrum_falls_back_to_legacy(tmp_path):
    run_dir = tmp_path / "runs" / "r2"
    legacy = run_dir / "spectrum" / "spectrum.h5"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy")

    expected = run_dir / "spectrum" / "spectrum.h5"
    assert resolve_spectrum_path(run_dir) == expected
