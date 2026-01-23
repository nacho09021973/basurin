#!/usr/bin/env python3
"""DEPRECATED: renombrado a 02_mix_spectra.py"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from basurin_io import resolve_spectrum_path as resolve_run_spectrum_path


def resolve_in_spectrum(run_id: str, base_dir: str | Path = "runs") -> Path:
    run_dir = Path(base_dir) / run_id
    try:
        return resolve_run_spectrum_path(run_dir)
    except FileNotFoundError:
        outputs = run_dir / "spectrum" / "outputs" / "spectrum.h5"
        legacy = run_dir / "spectrum" / "spectrum.h5"
        if outputs.exists():
            return outputs
        if legacy.exists():
            return legacy
        raise


def main() -> int:
    target = Path(__file__).with_name("02_mix_spectra.py")
    if not target.exists():
        print("ERROR: no se encuentra 02_mix_spectra.py junto a este wrapper.", file=sys.stderr)
        return 1

    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
