#!/usr/bin/env python3
"""Localizador determinista de artefactos BASURIN (solo lectura)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


def _default_runs_root() -> Path:
    env_root = os.environ.get("BASURIN_RUNS_ROOT")
    if env_root:
        return Path(env_root)
    return Path("runs")


def _abs(path: Path) -> str:
    return str(path.resolve())


def _load_verdict(verdict_path: Path) -> str | None:
    try:
        payload = json.loads(verdict_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    verdict = payload.get("verdict")
    return verdict if isinstance(verdict, str) else None


def _print_paths(paths: Iterable[Path]) -> bool:
    found = False
    for path in paths:
        if path.exists():
            print(_abs(path))
            found = True
    return found


def cmd_gating(runs_root: Path, run_id: str) -> int:
    path = runs_root / run_id / "RUN_VALID" / "verdict.json"
    if not path.exists():
        print(f"NOT_FOUND: {_abs(path)}")
        return 2
    verdict = _load_verdict(path)
    print(f"GATING: {_abs(path)}")
    if verdict is None:
        print("verdict: <unparseable>")
    else:
        print(f"verdict: {verdict}")
    return 0


def cmd_pass(runs_root: Path) -> int:
    if not runs_root.exists():
        print(f"NOT_FOUND: runs_root does not exist: {_abs(runs_root)}")
        return 2

    matches: list[tuple[str, Path]] = []
    for verdict_path in sorted(runs_root.glob("*/RUN_VALID/verdict.json")):
        verdict = _load_verdict(verdict_path)
        if verdict == "PASS":
            matches.append((verdict_path.parts[-3], verdict_path))

    if not matches:
        print("No PASS runs found.")
        return 2

    print("PASS runs:")
    for run_id, verdict_path in matches:
        print(f"- {run_id} :: {_abs(verdict_path)}")
    return 0


def cmd_h5(runs_root: Path, run_id: str) -> int:
    base = runs_root / run_id / "s1_fetch_strain"
    h1 = base / "inputs" / "H1.h5"
    l1 = base / "inputs" / "L1.h5"
    provenance = base / "outputs" / "provenance.json"

    print("H5 used by s1:")
    found_any = _print_paths([h1, l1])

    if provenance.exists():
        found_any = True
        print(f"provenance: {_abs(provenance)}")
        print("--- provenance.json (first 20 lines) ---")
        for i, line in enumerate(provenance.read_text(encoding="utf-8").splitlines(), start=1):
            if i > 20:
                break
            print(line)
        print("--- end ---")
    else:
        print(f"NOT_FOUND: {_abs(provenance)}")

    return 0 if found_any else 2


def cmd_estimates(runs_root: Path, run_id: str) -> int:
    path = runs_root / run_id / "s3_ringdown_estimates" / "outputs" / "estimates.json"
    if path.exists():
        print(_abs(path))
        return 0
    print(f"NOT_FOUND: {_abs(path)}")
    return 2


def cmd_window(runs_root: Path, run_id: str) -> int:
    root = runs_root / run_id / "s2_ringdown_window" / "outputs"
    candidates = sorted(root.glob("window_meta.json"))
    if not candidates:
        print(f"NOT_FOUND: {_abs(root / 'window_meta.json')}")
        return 2
    for path in candidates:
        print(_abs(path))
    return 0


def cmd_curvature(runs_root: Path, run_id: str) -> int:
    run_root = runs_root / run_id
    patterns = [
        "s6*/outputs/curvature*.json",
        "s6*/outputs/metric_diagnostics*.json",
    ]
    found_paths: list[Path] = []
    for pattern in patterns:
        found_paths.extend(sorted(run_root.glob(pattern)))

    if not found_paths:
        print(f"NOT_FOUND: no curvature/metric diagnostics files under {_abs(run_root)}")
        return 2

    for path in found_paths:
        print(_abs(path))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find canonical BASURIN artifacts (read-only).")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=_default_runs_root(),
        help="Runs root (default: BASURIN_RUNS_ROOT if set, otherwise ./runs)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID for lookups that require a single run.",
    )
    parser.add_argument(
        "--what",
        required=True,
        choices=["gating", "pass", "h5", "estimates", "window", "curvature"],
        help="Artifact group to locate.",
    )
    args = parser.parse_args(argv)

    if args.what != "pass" and not args.run_id:
        parser.error("--run-id is required unless --what pass")

    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code else 0

    runs_root = args.runs_root
    what = args.what

    if what == "gating":
        return cmd_gating(runs_root, args.run_id)
    if what == "pass":
        return cmd_pass(runs_root)
    if what == "h5":
        return cmd_h5(runs_root, args.run_id)
    if what == "estimates":
        return cmd_estimates(runs_root, args.run_id)
    if what == "window":
        return cmd_window(runs_root, args.run_id)
    if what == "curvature":
        return cmd_curvature(runs_root, args.run_id)

    print(f"Unknown --what value: {what}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
