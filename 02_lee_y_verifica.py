#!/usr/bin/env python3
"""
DEPRECATED: este script ha sido renombrado a 02a_smoke_lee_y_verifica.py

Uso:
    python 02a_smoke_lee_y_verifica.py ...

Motivo:
- Este archivo se mantiene solo por compatibilidad con rutas antiguas.
"""
from pathlib import Path
import runpy
import sys

target = Path(__file__).with_name("02a_smoke_lee_y_verifica.py")
if not target.exists():
    raise SystemExit("ERROR: No se encontró 02a_smoke_lee_y_verifica.py")

# Ejecutar el script nuevo preservando argv
runpy.run_path(str(target), run_name="__main__")
