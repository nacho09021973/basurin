#!/usr/bin/env python3
"""Build canonical features stage from dictionary outputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Allow running as a script by ensuring repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    assert_within_runs,
    ensure_stage_dirs,
    get_runs_root,
    load_feature_json,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "0.2.0"
_MAX_INLINE_VALUES = 10000


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _require_sklearn() -> bool:
    if importlib.util.find_spec("sklearn") is None:
        print("ERROR: sklearn no disponible; instala scikit-learn para generar features.", file=sys.stderr)
        return False
    return True


def _compute_local_features(X: np.ndarray, k_neighbors: int) -> tuple[np.ndarray, int]:
    """Compute local tangent features.

    Returns (features, effective_k) where effective_k may be smaller than
    k_neighbors if dataset is too small.
    """
    if X.ndim != 2:
        raise ValueError("X debe ser matriz 2D.")
    n_rows = X.shape[0]
    if k_neighbors < 1:
        raise ValueError("k_neighbors debe ser >= 1.")
    # Adapt k_neighbors for small datasets
    effective_k = min(k_neighbors, n_rows - 1)
    if effective_k < 1:
        raise ValueError(
            f"n_rows ({n_rows}) insuficiente para calcular features locales (necesita >= 2)."
        )

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=effective_k + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    eps = 1e-12
    features = np.zeros((n_rows, 6), dtype=float)
    for i in range(n_rows):
        neighbors = X[indices[i]]
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        denom = max(effective_k - 1, 1)
        cov = (centered.T @ centered) / float(denom)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        sum_lambda = float(np.sum(eigvals))
        sum_lambda_sq = float(np.sum(eigvals ** 2))
        d_eff = (sum_lambda ** 2) / (sum_lambda_sq + eps)
        lambda1 = float(eigvals[0]) if eigvals.size else 0.0
        m = lambda1 / (sum_lambda + eps)
        parallel = math.sqrt(max(lambda1, 0.0))
        if eigvals.size > 1:
            perp_val = float(np.mean(eigvals[1:]))
            perp = math.sqrt(max(perp_val, 0.0))
        else:
            perp = 0.0
        mean_dist = float(np.mean(distances[i]))
        rho = 1.0 / (mean_dist + eps)
        rho_clipped = float(np.clip(rho, 1e-8, 1e8))
        log10_rho = math.log10(rho_clipped)
        features[i] = [
            d_eff,
            m,
            parallel,
            perp,
            rho_clipped,
            log10_rho,
        ]
    return features, effective_k


def _should_inline(matrix: np.ndarray, max_values: int = _MAX_INLINE_VALUES) -> bool:
    return int(matrix.size) <= max_values


def _load_atlas_inputs(atlas_path: Path, feature_key_hint: str) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    ids, X, meta = load_feature_json(atlas_path, kind="atlas", feature_key_hint=feature_key_hint)
    if ids is None or len(ids) == 0:
        raise ValueError("No se pudieron resolver ids desde atlas.")
    if len(ids) != X.shape[0]:
        raise ValueError("ids no coincide con filas de X.")
    return ids, X, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical features stage from dictionary outputs"
    )
    parser.add_argument("--run", required=True, help="run id under runs/<run>")
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=7,
        help="k de vecinos para tangentes_locales_v1 (default: 7)",
    )
    parser.add_argument(
        "--out-root",
        default="runs",
        help="Root directory for runs (default: runs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        runs_root = get_runs_root()
        out_root = resolve_out_root(args.out_root, runs_root=runs_root)
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_dir = (out_root / args.run).resolve()

    atlas_points_path = (run_dir / "dictionary" / "outputs" / "atlas_points.json").resolve()
    atlas_path = (run_dir / "dictionary" / "outputs" / "atlas.json").resolve()

    try:
        assert_within_runs(run_dir, atlas_points_path)
        assert_within_runs(run_dir, atlas_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if atlas_points_path.exists():
        atlas_source = atlas_points_path
        ids_source = "dictionary/outputs/atlas_points.json"
    elif atlas_path.exists():
        atlas_source = atlas_path
        ids_source = "dictionary/outputs/atlas.json"
    else:
        print(
            "ERROR: faltan inputs aguas arriba; ejecuta scripts 01-04 para generar dictionary/outputs/atlas.json.",
            file=sys.stderr,
        )
        return 1

    if not _require_sklearn():
        return 1

    try:
        ids, X, meta = _load_atlas_inputs(atlas_source, feature_key_hint="ratios")
    except (ValueError, SystemExit) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        Y, effective_k = _compute_local_features(X, args.k_neighbors)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "features", base_dir=out_root)

    features_path = outputs_dir / "features.json"
    x_path = outputs_dir / "X.npy"
    y_path = outputs_dir / "Y.npy"
    feature_key = "tangentes_locales_v1"
    columns = [
        "d_eff",
        "m",
        "parallel",
        "perp",
        "rho_clipped",
        "log10_rho",
    ]
    np.save(x_path, X)
    np.save(y_path, Y)
    shapes = {"n": int(X.shape[0]), "dx": int(X.shape[1]), "dy": int(Y.shape[1])}
    created_utc = utc_now_iso()
    payload = {
        "schema_version": "1",
        "feature_key": feature_key,
        "ids": ids,
        "Y": Y.tolist(),
        "X_path": x_path.name,
        "Y_path": y_path.name,
        "shapes": shapes,
        "meta": {
            "feature_key": feature_key,
            "columns": columns,
            "k_neighbors": effective_k,
            "k_neighbors_requested": args.k_neighbors,
            "k_neighbors_effective": effective_k,
            "created": created_utc,
            "ids_source": ids_source,
            "source_atlas": f"runs/{args.run}/{ids_source}",
            "atlas_feature_key": meta.get("feature_key"),
            "shapes": shapes,
        },
    }
    if _should_inline(X):
        payload["X"] = X.tolist()

    source: dict[str, str] = {}
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    if spectrum_path.exists():
        source["spectrum_path"] = "spectrum/outputs/spectrum.h5"
    validation_path = run_dir / "dictionary" / "outputs" / "validation.json"
    if validation_path.exists():
        source["dictionary_validation_path"] = "dictionary/outputs/validation.json"

    wrapped_payload = {
        "metadata": {
            "schema_version": "1.0",
            "stage": "features",
            "run": args.run,
            "created_utc": created_utc,
            "source": source,
            "config": {
                "k_neighbors": effective_k,
                "k_neighbors_requested": args.k_neighbors,
                "k_neighbors_effective": effective_k,
            },
            "conventions": {},
        },
        "features": payload,
    }

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(wrapped_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    input_hash = sha256_file(atlas_source)
    output_hash = sha256_file(features_path)
    x_hash = sha256_file(x_path)
    y_hash = sha256_file(y_path)
    summary = {
        "stage": "features",
        "version": __version__,
        "script": "05_build_features_stage.py",
        "run": args.run,
        "inputs": {
            "atlas_path": _display_path(atlas_source),
            "atlas_sha256": input_hash,
        },
        "outputs": {
            "features": "outputs/features.json",
            "X": "outputs/X.npy",
            "Y": "outputs/Y.npy",
        },
        "config": {
            "k_neighbors_requested": args.k_neighbors,
            "k_neighbors_effective": effective_k,
            "k_neighbors": effective_k,  # backward compat
            "fallback_reason_k_neighbors": (
                f"k_neighbors reduced from {args.k_neighbors} to {effective_k} (n_rows={len(ids)})"
                if effective_k < args.k_neighbors else None
            ),
            "out_root_requested": args.out_root,
            "out_root_effective": _display_path(out_root),
        },
        "data": {
            "n_rows": len(ids),
            "n_features": int(Y.shape[1]),
            "X_shape": [int(X.shape[0]), int(X.shape[1])],
            "Y_shape": [int(Y.shape[0]), int(Y.shape[1])],
            "n": shapes["n"],
            "dx": shapes["dx"],
            "dy": shapes["dy"],
        },
        "hashes": {
            "outputs/features.json": output_hash,
            "outputs/X.npy": x_hash,
            "outputs/Y.npy": y_hash,
        },
    }
    summary_path = write_stage_summary(stage_dir, summary)

    write_manifest(
        stage_dir,
        {
            "features": features_path,
            "X": x_path,
            "Y": y_path,
            "summary": summary_path,
        },
        extra={
            "version": __version__,
            "inputs": {
                "atlas": _display_path(atlas_source),
                "atlas_sha256": input_hash,
            },
        },
    )

    print(f"features.json escrito en {features_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
