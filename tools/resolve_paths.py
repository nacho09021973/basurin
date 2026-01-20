from __future__ import annotations

from pathlib import Path


def resolve_spectrum_h5(run_id: str) -> Path:
    """Resolve the spectrum.h5 path for a run.

    Priority:
    1) runs/<run_id>/spectrum/outputs/spectrum.h5
    2) runs/<run_id>/spectrum/spectrum.h5 (legacy)

    Raises:
        FileNotFoundError: when no candidate exists.
    """
    base = Path("runs") / run_id / "spectrum"
    candidates = [
        base / "outputs" / "spectrum.h5",
        base / "spectrum.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "spectrum.h5 no encontrado en outputs/ ni legacy: "
        f"{candidates[0]} | {candidates[1]}"
    )


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
