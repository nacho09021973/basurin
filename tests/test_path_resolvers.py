import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.resolve_paths import resolve_spectrum_h5, resolve_validation_json


def test_resolvers_prioritize_outputs(tmp_path, monkeypatch):
    run_id = "run-123"
    base = tmp_path / "runs" / run_id

    outputs_spectrum = base / "spectrum" / "outputs" / "spectrum.h5"
    legacy_spectrum = base / "spectrum" / "spectrum.h5"
    outputs_validation = base / "dictionary" / "outputs" / "validation.json"
    legacy_validation = base / "dictionary" / "validation.json"

    outputs_spectrum.parent.mkdir(parents=True, exist_ok=True)
    legacy_spectrum.parent.mkdir(parents=True, exist_ok=True)
    outputs_validation.parent.mkdir(parents=True, exist_ok=True)
    legacy_validation.parent.mkdir(parents=True, exist_ok=True)

    outputs_spectrum.write_text("new")
    legacy_spectrum.write_text("legacy")
    outputs_validation.write_text("new")
    legacy_validation.write_text("legacy")

    monkeypatch.chdir(tmp_path)

    assert resolve_spectrum_h5(run_id).resolve() == outputs_spectrum.resolve()
    assert resolve_validation_json(run_id).resolve() == outputs_validation.resolve()


def test_resolvers_fallback_to_legacy(tmp_path, monkeypatch):
    run_id = "run-456"
    base = tmp_path / "runs" / run_id

    legacy_spectrum = base / "spectrum" / "spectrum.h5"
    legacy_validation = base / "dictionary" / "validation.json"

    legacy_spectrum.parent.mkdir(parents=True)
    legacy_validation.parent.mkdir(parents=True)

    legacy_spectrum.write_text("legacy")
    legacy_validation.write_text("legacy")

    monkeypatch.chdir(tmp_path)

    assert resolve_spectrum_h5(run_id).resolve() == legacy_spectrum.resolve()
    assert resolve_validation_json(run_id).resolve() == legacy_validation.resolve()


def test_resolvers_handle_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_validation_json("missing-run") is None

    try:
        resolve_spectrum_h5("missing-run")
    except FileNotFoundError as exc:
        assert "spectrum.h5 no encontrado" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing spectrum.h5")
