#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _iter_event_dirs(losc_root: Path) -> list[Path]:
    try:
        entries = [p for p in losc_root.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return []
    except OSError:
        return []
    return sorted(entries, key=lambda p: p.name)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Read-only canonical listing of visible event directories under data/losc"
        )
    )
    ap.add_argument(
        "--losc-root",
        default=os.environ.get("BASURIN_LOSC_ROOT", "data/losc"),
        help="LOSC root (default: BASURIN_LOSC_ROOT or data/losc)",
    )
    ap.add_argument(
        "--check-nonempty",
        action="store_true",
        help="Return exit code 2 if no visible event directories are found",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    losc_root = Path(args.losc_root).expanduser().resolve()
    event_dirs = _iter_event_dirs(losc_root)

    print(f"LOSC_ROOT_EFFECTIVE={losc_root}")
    print(f"event_dir_count={len(event_dirs)}")

    for event_dir in event_dirs:
        print(event_dir.name)

    if args.check_nonempty and not event_dirs:
        print(
            "FAIL: no hay eventos visibles bajo la vista canónica data/losc/<EVENT_ID>/.\n"
            "Hint: revisa README.md y docs/readme_rutas.md; si la caché real vive en "
            "gw_events/strain/<EVENT_ID>/, expónla primero bajo data/losc/<EVENT_ID>/ "
            "con symlink o bind mount.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
