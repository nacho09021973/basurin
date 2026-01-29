from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "experiment" / "run_valid" / "stage_run_valid.py"


def test_unique_canonical_run_valid_stage() -> None:
    assert CANON.exists(), f"Missing canonical RUN_VALID stage at {CANON}"

    hits = []
    # Buscamos cualquier stage_run_valid.py en el repo (incluye legacy)
    for p in REPO.rglob("stage_run_valid.py"):
        # ignora venvs y caches si existieran
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        hits.append(p)

    # Debe existir exactamente UNO, y debe ser el canónico
    assert hits == [CANON], f"Found non-canonical RUN_VALID stage copies: {hits}"
