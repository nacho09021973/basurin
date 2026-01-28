#!/usr/bin/env python3
"""Build canonical atlas master stage aggregating multiple runs."""
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
    get_runs_root,
    load_feature_json,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

__version__ = "0.1.0"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _parse_runs(runs_arg: str) -> list[str]:
    runs = [run.strip() for run in runs_arg.split(",") if run.strip()]
    if not runs:
        raise ValueError("--runs debe incluir al menos un run_id.")
    return runs


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _diagnose_X_redundancy(X: np.ndarray, eps_rel: float = 1e-12) -> dict[str, Any]:
    if X.size == 0:
        return {
            "X_singular_values": [],
            "X_explained_var_1": 0.0,
            "X_rank_eps_rel": eps_rel,
            "X_rank": 0,
            "X_pairwise_corr_max_offdiag": 0.0,
            "X_effective_dim": 0,
            "X_is_redundant": False,
        }

    X_centered = X - np.mean(X, axis=0, keepdims=True)
    try:
        _, svals, _ = np.linalg.svd(X_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        svals = np.array([], dtype=float)

    if svals.size:
        s1 = float(svals[0])
        svals_sq = np.square(svals)
        denom = float(svals_sq.sum())
        explained_var_1 = float(svals_sq[0] / denom) if denom > 0 else 0.0
        rank = int(np.count_nonzero(svals > s1 * eps_rel)) if s1 > 0 else 0
    else:
        s1 = 0.0
        explained_var_1 = 0.0
        rank = 0

    ncols = int(X.shape[1]) if X.ndim > 1 else 1
    if ncols > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X_centered, rowvar=False)
        abs_corr = np.abs(corr)
        offdiag_mask = ~np.eye(ncols, dtype=bool)
        offdiag_values = abs_corr[offdiag_mask]
        max_offdiag = float(np.nanmax(offdiag_values)) if offdiag_values.size else 0.0
        if not np.isfinite(max_offdiag):
            max_offdiag = 0.0
    else:
        max_offdiag = 0.0

    if explained_var_1 > 0.98:
        effective_dim = 1
    elif explained_var_1 > 0.90:
        effective_dim = 2
    else:
        effective_dim = min(ncols, 3)

    return {
        "X_singular_values": [float(val) for val in svals.tolist()],
        "X_explained_var_1": explained_var_1,
        "X_rank_eps_rel": eps_rel,
        "X_rank": rank,
        "X_pairwise_corr_max_offdiag": max_offdiag,
        "X_effective_dim": effective_dim,
        "X_is_redundant": bool(max_offdiag > 0.999 or explained_var_1 > 0.98),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical atlas master stage aggregating multiple runs"
    )
    parser.add_argument("--run", required=True, help="run id master under runs/<run>")
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated list of runs to aggregate (run1,run2,...)",
    )
    parser.add_argument(
        "--out-root",
        default="runs",
        help="Root directory for runs (default: runs)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Permite omitir runs sin atlas.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        runs_root = get_runs_root()
        out_root = resolve_out_root(args.out_root, runs_root=runs_root)
        validate_run_id(args.run, out_root)
        runs_list = _parse_runs(args.runs)
        for run_id in runs_list:
            validate_run_id(run_id, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    atlas_feature_key: str | None = None
    features_feature_key: str | None = None
    atlas_dim: int | None = None
    features_dim: int | None = None

    atlas_ids_all: list[Any] = []
    atlas_rows: list[np.ndarray] = []
    features_ids_all: list[Any] = []
    features_rows: list[np.ndarray] = []

    inputs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, str]] = []

    for run_id in runs_list:
        run_dir = (out_root / run_id).resolve()
        atlas_path = (run_dir / "dictionary" / "outputs" / "atlas.json").resolve()
        features_path = (run_dir / "features" / "outputs" / "features.json").resolve()
        verdict_path = (run_dir / "RUN_VALID" / "verdict.json").resolve()

        try:
            assert_within_runs(run_dir, atlas_path)
            assert_within_runs(run_dir, features_path)
            assert_within_runs(run_dir, verdict_path)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if not atlas_path.exists():
            if args.allow_missing:
                skipped_runs.append(
                    {
                        "run": run_id,
                        "reason": "missing dictionary/outputs/atlas.json",
                    }
                )
                continue
            print(
                f"ERROR: falta input canónico: {atlas_path}",
                file=sys.stderr,
            )
            return 1

        try:
            atlas_ids, atlas_matrix, atlas_meta = load_feature_json(
                atlas_path, kind="atlas", feature_key_hint="ratios"
            )
        except (ValueError, SystemExit) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if atlas_ids is None:
            print(f"ERROR: atlas sin ids en {atlas_path}", file=sys.stderr)
            return 1

        if atlas_dim is None:
            atlas_dim = int(atlas_matrix.shape[1])
        elif atlas_dim != int(atlas_matrix.shape[1]):
            print(
                f"ERROR: dimensiones incompatibles en {atlas_path} ({atlas_matrix.shape[1]} != {atlas_dim})",
                file=sys.stderr,
            )
            return 1

        atlas_key = atlas_meta.get("feature_key")
        if atlas_feature_key is None and atlas_key:
            atlas_feature_key = str(atlas_key)
        elif atlas_key and atlas_feature_key != str(atlas_key):
            print(
                f"ERROR: feature_key atlas inconsistente ({atlas_feature_key} vs {atlas_key})",
                file=sys.stderr,
            )
            return 1

        atlas_ids_all.extend(atlas_ids)
        atlas_rows.append(atlas_matrix)

        features_entry: dict[str, Any] | None = None
        if features_path.exists():
            try:
                feat_ids, feat_matrix, feat_meta = load_feature_json(
                    features_path, kind="features", feature_key_hint="tangentes_locales_v1"
                )
            except (ValueError, SystemExit) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

            if feat_ids is None:
                print(f"ERROR: features sin ids en {features_path}", file=sys.stderr)
                return 1

            if list(feat_ids) != list(atlas_ids):
                print(
                    f"ERROR: ids mismatch entre atlas y features en {run_id}",
                    file=sys.stderr,
                )
                return 1

            if features_dim is None:
                features_dim = int(feat_matrix.shape[1])
            elif features_dim != int(feat_matrix.shape[1]):
                print(
                    f"ERROR: dimensiones features incompatibles en {features_path} ({feat_matrix.shape[1]} != {features_dim})",
                    file=sys.stderr,
                )
                return 1

            feat_key = feat_meta.get("feature_key")
            if features_feature_key is None and feat_key:
                features_feature_key = str(feat_key)
            elif feat_key and features_feature_key != str(feat_key):
                print(
                    f"ERROR: feature_key features inconsistente ({features_feature_key} vs {feat_key})",
                    file=sys.stderr,
                )
                return 1

            features_ids_all.extend(feat_ids)
            features_rows.append(feat_matrix)
            features_entry = {
                "path": _display_path(features_path),
                "sha256": sha256_file(features_path),
                "feature_key": features_feature_key,
            }

        verdict_entry: dict[str, Any] | None = None
        if verdict_path.exists():
            try:
                verdict_entry = _load_json(verdict_path)
            except json.JSONDecodeError as exc:
                print(f"ERROR: verdict.json inválido en {verdict_path}: {exc}", file=sys.stderr)
                return 1

        inputs.append(
            {
                "run": run_id,
                "atlas": {
                    "path": _display_path(atlas_path),
                    "sha256": sha256_file(atlas_path),
                    "feature_key": atlas_feature_key,
                },
                "features": features_entry,
                "verdict": {
                    "path": _display_path(verdict_path),
                    "data": verdict_entry,
                    "sha256": sha256_file(verdict_path) if verdict_path.exists() else None,
                }
                if verdict_entry is not None
                else None,
            }
        )

    if not atlas_rows:
        print("ERROR: no hay atlas.json disponibles para agregar.", file=sys.stderr)
        return 1

    atlas_matrix_all = np.vstack(atlas_rows)
    atlas_payload: dict[str, Any] = {
        "schema_version": "1",
        "feature_key": atlas_feature_key or "ratios",
        "ids": atlas_ids_all,
        "X": atlas_matrix_all.tolist(),
        "meta": {
            "created": utc_now_iso(),
            "source_runs": [item["run"] for item in inputs],
            "source_count": len(inputs),
            "feature_key": atlas_feature_key or "ratios",
            "columns": [f"{atlas_feature_key or 'ratios'}_{i}" for i in range(atlas_matrix_all.shape[1])],
        },
        "sources": inputs,
        "skipped_runs": skipped_runs,
    }

    features_matrix_all = None
    if features_rows:
        features_matrix_all = np.vstack(features_rows)
        atlas_payload["features"] = {
            "schema_version": "1",
            "feature_key": features_feature_key or "tangentes_locales_v1",
            "ids": features_ids_all,
            "Y": features_matrix_all.tolist(),
            "meta": {
                "created": utc_now_iso(),
                "source_runs": [item["run"] for item in inputs if item.get("features")],
                "source_count": len([item for item in inputs if item.get("features")]),
                "feature_key": features_feature_key or "tangentes_locales_v1",
                "columns": [
                    f"{features_feature_key or 'tangentes_locales_v1'}_{i}"
                    for i in range(features_matrix_all.shape[1])
                ],
            },
        }

    diagnostics = _diagnose_X_redundancy(atlas_matrix_all)
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "atlas_master", base_dir=out_root)
    output_path = outputs_dir / "BRIDGE_ATLAS_MASTER.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(atlas_payload, f, indent=2)

    output_hash = sha256_file(output_path)

    input_hashes: dict[str, str] = {}
    for entry in inputs:
        atlas_entry = entry.get("atlas")
        if isinstance(atlas_entry, dict) and atlas_entry.get("path") and atlas_entry.get("sha256"):
            input_hashes[str(atlas_entry["path"])] = atlas_entry["sha256"]
        features_entry = entry.get("features")
        if isinstance(features_entry, dict) and features_entry.get("path") and features_entry.get("sha256"):
            input_hashes[str(features_entry["path"])] = features_entry["sha256"]
        verdict_entry = entry.get("verdict")
        if isinstance(verdict_entry, dict) and verdict_entry.get("path") and verdict_entry.get("sha256"):
            input_hashes[str(verdict_entry["path"])] = verdict_entry["sha256"]

    summary = {
        "stage": "atlas_master",
        "version": __version__,
        "script": "06_build_atlas_master_stage.py",
        "run": args.run,
        "inputs": inputs,
        "skipped_runs": skipped_runs,
        "outputs": {
            "atlas_master": "outputs/BRIDGE_ATLAS_MASTER.json",
        },
        "config": {
            "runs_requested": runs_list,
            "runs_included": [item["run"] for item in inputs],
            "allow_missing": args.allow_missing,
            "out_root_requested": args.out_root,
            "out_root_effective": _display_path(out_root),
        },
        "data": {
            "atlas_rows": int(atlas_matrix_all.shape[0]),
            "atlas_dim": int(atlas_matrix_all.shape[1]),
            "features_rows": int(features_matrix_all.shape[0]) if features_matrix_all is not None else 0,
            "features_dim": int(features_matrix_all.shape[1]) if features_matrix_all is not None else 0,
        },
        "atlas_master_diagnostics": diagnostics,
        "hashes": {
            "outputs/BRIDGE_ATLAS_MASTER.json": output_hash,
        },
        "input_hashes": input_hashes,
    }

    summary_path = write_stage_summary(stage_dir, summary)

    write_manifest(
        stage_dir,
        {
            "atlas_master": output_path,
            "summary": summary_path,
        },
        extra={
            "version": __version__,
            "inputs": input_hashes,
            "skipped_runs": skipped_runs,
        },
    )

    print(f"BRIDGE_ATLAS_MASTER.json escrito en {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
