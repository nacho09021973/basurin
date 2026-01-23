#!/usr/bin/env python3
"""Generate canonical synthetic bridge inputs for F4-1 alignment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow running as a script by ensuring repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    assert_within_runs,
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "bridge_inputs_synth"
GENERATOR_VERSION = "bridge_inputs_synth_v1"


def _governance_error(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical synthetic bridge inputs for F4-1 alignment.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dx", type=int, required=True)
    parser.add_argument("--dy", type=int, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--atlas-feature-key", required=True)
    parser.add_argument("--y-feature-key", required=True)
    return parser.parse_args()


def _validate_governance(run_id: str, out_root: Path) -> None:
    try:
        validate_run_id(run_id, out_root)
    except ValueError as exc:
        _governance_error(str(exc))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _generate_data(
    n: int,
    dx: int,
    dy: int,
    rho: float,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0 or dx <= 0 or dy <= 0:
        raise ValueError("n, dx, dy deben ser positivos")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho debe estar en [-1, 1]")
    if noise < 0:
        raise ValueError("noise debe ser >= 0")

    rng = np.random.default_rng(seed)
    k = min(dx, dy)
    z = rng.standard_normal((n, k))
    a = rng.standard_normal((k, dx))
    b = rng.standard_normal((k, dy))
    x_base = z @ a
    y_base = z @ b
    x = x_base + noise * rng.standard_normal((n, dx))
    y = rho * y_base + noise * rng.standard_normal((n, dy))
    return x, y


def main() -> int:
    args = _parse_args()
    try:
        out_root = resolve_out_root(args.out_root)
    except ValueError as exc:
        _governance_error(str(exc))

    _validate_governance(args.run, out_root)
    run_dir = (out_root / args.run).resolve()
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    try:
        assert_within_runs(run_dir, stage_dir)
        assert_within_runs(run_dir, outputs_dir)
    except ValueError as exc:
        _governance_error(str(exc))

    try:
        x, y = _generate_data(args.n, args.dx, args.dy, args.rho, args.noise, args.seed)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ids = [f"id_{i:06d}" for i in range(args.n)]
    atlas_columns = [f"{args.atlas_feature_key}_{i}" for i in range(args.dx)]
    y_columns = [f"{args.y_feature_key}_{i}" for i in range(args.dy)]

    atlas_payload = {
        "ids": ids,
        "X": x.tolist(),
        "meta": {
            "feature_key": args.atlas_feature_key,
            "schema_version": "1",
            "columns": atlas_columns,
        },
    }
    features_payload = {
        "ids": ids,
        "Y": y.tolist(),
        "meta": {
            "feature_key": args.y_feature_key,
            "schema_version": "1",
            "columns": y_columns,
        },
    }

    atlas_path = outputs_dir / "atlas_bridge.json"
    features_path = outputs_dir / "features_points_k7.json"
    generator_meta_path = outputs_dir / "generator_meta.json"

    _write_json(atlas_path, atlas_payload)
    _write_json(features_path, features_payload)

    hashes = {
        "atlas_bridge.json": sha256_file(atlas_path),
        "features_points_k7.json": sha256_file(features_path),
    }

    generator_meta = {
        "schema_version": "1",
        "generator_version": GENERATOR_VERSION,
        "created": utc_now_iso(),
        "n": args.n,
        "dx": args.dx,
        "dy": args.dy,
        "rho": args.rho,
        "noise": args.noise,
        "seed": args.seed,
        "atlas_feature_key": args.atlas_feature_key,
        "y_feature_key": args.y_feature_key,
        "hashes": hashes,
    }
    _write_json(generator_meta_path, generator_meta)

    artifacts = {
        "atlas": atlas_path,
        "features": features_path,
        "generator_meta": generator_meta_path,
    }
    manifest_path = write_manifest(stage_dir, artifacts)

    summary = {
        "schema_version": "1",
        "generator_version": GENERATOR_VERSION,
        "n": args.n,
        "dx": args.dx,
        "dy": args.dy,
        "rho": args.rho,
        "noise": args.noise,
        "seed": args.seed,
        "atlas_feature_key": args.atlas_feature_key,
        "y_feature_key": args.y_feature_key,
        "outputs": {
            "atlas": str(atlas_path.relative_to(stage_dir)),
            "features": str(features_path.relative_to(stage_dir)),
            "generator_meta": str(generator_meta_path.relative_to(stage_dir)),
            "manifest": str(manifest_path.relative_to(stage_dir)),
        },
        "hashes": hashes,
    }
    write_stage_summary(stage_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
