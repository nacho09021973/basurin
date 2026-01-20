#!/usr/bin/env python3
"""Wrapper de compatibilidad legacy; el script real vive en experiment/exp_04_tangentes_locales/05_tangentes_locales.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TARGET = _ROOT / "experiment" / "exp_04_tangentes_locales" / "05_tangentes_locales.py"
if not _TARGET.exists():
    raise FileNotFoundError(f"No se encontró el script real en {_TARGET}.")

_IMPORTED_GLOBALS = runpy.run_path(str(_TARGET), run_name="tangentes_locales_impl")
for _name, _value in _IMPORTED_GLOBALS.items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value


def main() -> None:
    runpy.run_path(str(_TARGET), run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())
