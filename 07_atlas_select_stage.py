#!/usr/bin/env python3
"""Wrapper for work/07_atlas_select_stage.py."""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    target = repo_root / "work" / "07_atlas_select_stage.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
