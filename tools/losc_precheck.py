#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def match_hdf5_files(event_dir: Path) -> dict[str, list[Path]]:
    """Standalone copy of mvp.s1_fetch_strain.match_hdf5_files.

    Kept here to avoid importing the full s1 module (and its deps)
    from a lightweight diagnostic tool.  Any change to the matching
    logic must be mirrored in mvp/s1_fetch_strain.py.
    """
    try:
        all_files = [
            p for p in event_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}
        ]
    except OSError:
        all_files = []
    all_files = sorted(all_files, key=lambda p: p.name)
    return {
        "all": all_files,
        "H1": [p for p in all_files if "H1" in p.name.upper()],
        "L1": [p for p in all_files if "L1" in p.name.upper()],
        "V1": [p for p in all_files if "V1" in p.name.upper()],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only precheck for local LOSC HDF5 visibility and naming")
    ap.add_argument("--event-id", required=True, help="GW event id, e.g. GW150914")
    ap.add_argument(
        "--losc-root",
        default=os.environ.get("BASURIN_LOSC_ROOT", "data/losc"),
        help="LOSC root (default: BASURIN_LOSC_ROOT or data/losc)",
    )
    args = ap.parse_args()

    losc_root = Path(args.losc_root).expanduser().resolve()
    event_dir = (losc_root / args.event_id).resolve()
    matches = match_hdf5_files(event_dir)

    print(f"LOSC_ROOT_EFFECTIVE={losc_root}")
    print(f"EVENT_DIR={event_dir}")
    print(f"h5_count={len(matches['all'])}")
    print(f"match_count_H1={len(matches['H1'])}")
    print(f"match_count_L1={len(matches['L1'])}")
    print(f"match_count_V1={len(matches['V1'])}")

    if not event_dir.is_dir():
        print("FAIL: EVENT_DIR no existe (rama A: mount/symlink roto o no visible).")
        return 2
    detectors_present = [det for det in ("H1", "L1", "V1") if matches[det]]
    if len(detectors_present) >= 2:
        print(
            "PASS: hay al menos dos detectores entre H1/L1/V1 con extensión .h5/.hdf5."
        )
        return 0

    print("FAIL: faltan al menos dos detectores entre H1/L1/V1 (rama B: naming; crear symlinks coherentes por detector).")
    if matches["all"]:
        print("Candidatos detectados:")
        for p in matches["all"]:
            print(f"  - {p.name}")
    print("Hint: revisa docs/readme_rutas.md (Precheck LOSC).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
