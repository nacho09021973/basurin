#!/usr/bin/env python3
"""
exp_05_tangentes_locales wrapper

Este experimento reutiliza el script estable de exp_04 como "script real" y lo ejecuta
desde esta carpeta para mantener el experimento autocontenido y trazable.
"""

import runpy
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TARGET = (_HERE.parent / "exp_04_tangentes_locales" / "05_tangentes_locales.py").resolve()

if not _TARGET.exists():
    raise FileNotFoundError(f"No se encontró el script real en {_TARGET}.")

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
