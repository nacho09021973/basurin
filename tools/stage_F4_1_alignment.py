#!/usr/bin/env python3
"""
BASURIN — Canonical CLI entrypoint for F4-1 bridge alignment.

This script exists ONLY to expose the executable interface under tools/,
so stages are discoverable, auditable, and never searched for again.

Actual implementation lives in:
  experiment/bridge/stage_F4_1_alignment.py

Policy:
- tools/        -> executable entrypoints (CLI)
- experiment/   -> implementation / logic

This wrapper must stay minimal.
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from experiment.bridge.stage_F4_1_alignment import main as _impl_main
    except Exception as e:
        print(
            "ERROR: failed to import experiment.bridge.stage_F4_1_alignment\n"
            "Ensure the module exists and is importable.\n"
            f"Details: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    _impl_main()


if __name__ == "__main__":
    main()
