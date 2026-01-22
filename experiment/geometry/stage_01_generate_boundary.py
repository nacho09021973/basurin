#!/usr/bin/env python3
"""Stage 01: Generate boundary data from explicit geometry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a script by ensuring repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    assert_within_runs,
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "0.1.0"


def _read_geometry(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_geometry_fields(geometry: Any) -> dict[str, Any]:
    if not isinstance(geometry, dict):
        return {
            "geometry_type": None,
            "dimension": None,
            "boundary_dimension": None,
        }
    return {
        "geometry_type": geometry.get("geometry_type", geometry.get("type")),
        "dimension": geometry.get("dimension"),
        "boundary_dimension": geometry.get("boundary_dimension"),
    }


def _build_boundary_points(boundary_dimension: Any) -> list[list[float]]:
    if not isinstance(boundary_dimension, int):
        return []
    if boundary_dimension < 0:
        return []
    return [[0.0 for _ in range(boundary_dimension)]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate minimal boundary data from explicit geometry"
    )
    parser.add_argument("--run", required=True, help="run id under runs/<run>")
    parser.add_argument(
        "--out-root",
        default="runs",
        help="Root directory for runs (default: runs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        out_root = resolve_out_root(args.out_root)
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_dir = (out_root / args.run).resolve()
    geometry_path = (run_dir / "geometry" / "outputs" / "geometry.json").resolve()

    try:
        assert_within_runs(run_dir, geometry_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not geometry_path.exists():
        print(f"ERROR: no existe geometry.json: {geometry_path}", file=sys.stderr)
        return 1

    geometry = _read_geometry(geometry_path)
    geometry_fields = _extract_geometry_fields(geometry)

    boundary_points = _build_boundary_points(geometry_fields["boundary_dimension"])

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "boundary", base_dir=out_root)

    boundary_data = {
        "run": args.run,
        "source": "explicit_geometry",
        "geometry": geometry,
        "geometry_declared": {
            "geometry_type": geometry_fields["geometry_type"],
            "dimension": geometry_fields["dimension"],
            "boundary_dimension": geometry_fields["boundary_dimension"],
        },
        "boundary": {
            "points": boundary_points,
            "point_dimension": geometry_fields["boundary_dimension"],
        },
        "notes": [
            "Boundary derived from asserted geometry input.",
            "Boundary is not inferred and not validated.",
        ],
    }

    boundary_path = outputs_dir / "boundary_data.json"
    with open(boundary_path, "w", encoding="utf-8") as f:
        json.dump(boundary_data, f, indent=2)

    summary = {
        "stage": "boundary",
        "run": args.run,
        "version": __version__,
        "config": {
            "geometry_type": geometry_fields["geometry_type"],
            "dimension": geometry_fields["dimension"],
        },
        "inputs": {
            "geometry_path": str(geometry_path.relative_to(run_dir)),
            "geometry_sha256": sha256_file(geometry_path),
        },
        "results": {
            "boundary_generated": True,
            "source": "explicit_geometry",
        },
        "notes": [
            "This boundary is derived from an asserted geometry.",
            "It is not inferred and not validated.",
        ],
    }
    summary_path = write_stage_summary(stage_dir, summary)

    manifest = write_manifest(
        stage_dir,
        {"boundary_data": boundary_path},
        extra={
            "inputs": {
                "geometry_path": str(geometry_path.relative_to(run_dir)),
                "geometry_sha256": sha256_file(geometry_path),
            },
            "summary": str(summary_path.relative_to(stage_dir)),
            "version": __version__,
        },
    )

    print(f"[boundary] OK -> {boundary_path} (manifest: {manifest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
