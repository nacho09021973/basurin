# experiment/io_paths.py
from __future__ import annotations
from pathlib import Path

def spectrum_outputs_path(run_dir: Path) -> Path:
    return run_dir / "spectrum" / "outputs" / "spectrum.h5"

def spectrum_legacy_path(run_dir: Path) -> Path:
    return run_dir / "spectrum" / "spectrum.h5"

def resolve_spectrum_path(run_dir: Path) -> Path:
    p_new = spectrum_outputs_path(run_dir)
    if p_new.exists():
        return p_new
    p_old = spectrum_legacy_path(run_dir)
    if p_old.exists():
        return p_old
    raise FileNotFoundError(
        "No se encontró spectrum.h5.\n"
        f"Esperado (canon): {p_new}\n"
        f"Fallback (legacy): {p_old}\n"
        "Ejecuta el stage 'spectrum' para este RUN o materializa el artefacto de forma reproducible."
    )
