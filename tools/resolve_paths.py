from __future__ import annotations

from pathlib import Path

from basurin_io import get_run_dir, resolve_spectrum_path


def resolve_spectrum_h5(run_id: str) -> Path:
    """Resolve the spectrum.h5 path for a run.

    Priority:
    1) runs/<run_id>/spectrum/outputs/spectrum.h5
    2) runs/<run_id>/spectrum/spectrum.h5 (legacy)

    Raises:
        FileNotFoundError: when no candidate exists.
    """
    return resolve_spectrum_path(get_run_dir(run_id))


def resolve_validation_json(run_id: str) -> Path | None:
    """Resolve the validation.json path for a run.

    Priority:
    1) runs/<run_id>/dictionary/outputs/validation.json
    2) runs/<run_id>/dictionary/validation.json (legacy)

    Returns:
        Path when found, otherwise None.
    """
    base = Path("runs") / run_id / "dictionary"
    candidates = [
        base / "outputs" / "validation.json",
        base / "validation.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None
