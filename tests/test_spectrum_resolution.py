import importlib.util
import sys
from pathlib import Path


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_in_spectrum_prefers_outputs(tmp_path, monkeypatch):
    mix_module = load_module("mix_module", Path(__file__).parents[1] / "01_mix_spectra.py")
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "runs" / "r1" / "spectrum" / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "spectrum.h5").write_text("ok")
    expected = Path("runs") / "r1" / "spectrum" / "outputs" / "spectrum.h5"

    assert mix_module.resolve_in_spectrum("r1") == expected


def test_resolve_in_spectrum_falls_back_to_legacy(tmp_path, monkeypatch):
    mix_module = load_module("mix_module_legacy", Path(__file__).parents[1] / "01_mix_spectra.py")
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "runs" / "r2" / "spectrum" / "spectrum.h5"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy")

    expected = Path("runs") / "r2" / "spectrum" / "spectrum.h5"
    assert mix_module.resolve_in_spectrum("r2") == expected


def test_resolve_spectrum_path_prefers_outputs(tmp_path, monkeypatch):
    dicc_module = load_module("dicc_module", Path(__file__).parents[1] / "04_diccionario.py")
    monkeypatch.chdir(tmp_path)
    stage_dir = tmp_path / "runs" / "r3" / "spectrum"
    out_file = stage_dir / "outputs" / "spectrum.h5"
    out_file.parent.mkdir(parents=True)
    out_file.write_text("ok")

    expected = Path("runs") / "r3" / "spectrum" / "outputs" / "spectrum.h5"
    assert dicc_module.resolve_spectrum_path("r3", "outputs/spectrum.h5") == expected


def test_resolve_spectrum_path_accepts_absolute(tmp_path):
    dicc_module = load_module("dicc_module_abs", Path(__file__).parents[1] / "04_diccionario.py")
    abs_path = tmp_path / "abs" / "spectrum.h5"
    abs_path.parent.mkdir(parents=True)
    abs_path.write_text("ok")

    assert dicc_module.resolve_spectrum_path("ignored", str(abs_path)) == abs_path


def test_resolve_tangentes_path_prefers_outputs(tmp_path):
    tan_module = load_module("tan_module", Path(__file__).parents[1] / "05_tangentes_locales.py")
    run_dir = tmp_path / "runs" / "r4"
    out_file = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    out_file.parent.mkdir(parents=True)
    out_file.write_text("ok")

    assert tan_module.resolve_spectrum_path(run_dir) == out_file
