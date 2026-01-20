#!/usr/bin/env python3
import runpy
from pathlib import Path

if __name__ == "__main__":
    tgt = Path(__file__).resolve().parent / "experiment/exp_05_tangentes_locales/05_tangentes_locales.py"
    runpy.run_path(str(tgt), run_name="__main__")
