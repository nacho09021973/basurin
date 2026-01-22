#!/usr/bin/env python3
"""Build canonical features stage from dictionary outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

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


def _coerce_vector(value: Any) -> Optional[list[float]]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, dict):
        for key in ["values", "vector", "data", "features", "ratios", "x"]:
            if key in value:
                return _coerce_vector(value[key])
    return None


def _extract_points(atlas: Any) -> Iterable[dict[str, Any]]:
    if isinstance(atlas, dict):
        for key in ["points", "theories", "data", "rows"]:
            if key in atlas and isinstance(atlas[key], list):
                return atlas[key]
    if isinstance(atlas, list):
        return atlas
    return []


def _load_atlas_features(
    atlas_path: Path,
) -> Tuple[list[Any], Optional[np.ndarray], list[str] | None]:
    with open(atlas_path, "r", encoding="utf-8") as f:
        atlas = json.load(f)

    points = list(_extract_points(atlas))
    if not points:
        raise ValueError(
            f"atlas.json sin puntos reconocibles en {atlas_path}. "
            "Se esperaba 'points'/'theories' o lista de dicts."
        )

    ids: list[Any] = []
    rows: list[list[float]] = []
    missing = 0
    for idx, point in enumerate(points):
        if not isinstance(point, dict):
            missing += 1
            continue
        pid = point.get("id", point.get("uid", point.get("name", idx)))
        ids.append(pid)
        vec = None
        for key in ["features", "ratios", "x", "vector"]:
            if key in point:
                vec = point[key]
                break
        vec_list = _coerce_vector(vec)
        if vec_list is None:
            missing += 1
            continue
        rows.append(vec_list)

    if not rows:
        return ids, None, None
    if missing:
        raise ValueError(
            f"atlas.json en {atlas_path} tiene {missing} puntos sin features. "
            "Asegura que todos los puntos contengan features/ratios."
        )

    feature_names: list[str] | None = None
    if isinstance(atlas, dict):
        if isinstance(atlas.get("feature_names"), list):
            feature_names = [str(v) for v in atlas["feature_names"]]
        elif isinstance(atlas.get("columns"), list):
            feature_names = [str(v) for v in atlas["columns"]]
        elif isinstance(atlas.get("feature_key"), str):
            feature_key = atlas["feature_key"]
            feature_names = [f"{feature_key}_{i}" for i in range(len(rows[0]))]

    return ids, np.asarray(rows, dtype=float), feature_names


def _decode_feature_names(raw: Any) -> Optional[list[str]]:
    if raw is None:
        return None
    try:
        values = np.asarray(raw)
    except Exception:
        return None
    if values.ndim == 0:
        return [str(values)]
    decoded = []
    for v in values.tolist():
        if isinstance(v, (bytes, bytearray)):
            decoded.append(v.decode("utf-8"))
        else:
            decoded.append(str(v))
    return decoded


def _find_dataset(h5: Any, names: list[str]) -> Optional[np.ndarray]:
    import h5py

    for name in names:
        if name in h5 and isinstance(h5[name], h5py.Dataset):
            return np.asarray(h5[name])
    if "features" in h5 and isinstance(h5["features"], h5py.Group):
        grp = h5["features"]
        for name in names:
            if name in grp and isinstance(grp[name], h5py.Dataset):
                return np.asarray(grp[name])
    return None


def _load_dictionary_features(dictionary_path: Path) -> tuple[Optional[list[Any]], Optional[np.ndarray], Optional[list[str]]]:
    try:
        import h5py
    except ImportError:
        print("ERROR: pip install h5py", file=sys.stderr)
        raise

    ids = None
    features = None
    feature_names = None

    with h5py.File(dictionary_path, "r") as h5:
        features = _find_dataset(h5, ["X", "features", "metrics"])
        ids_dataset = _find_dataset(h5, ["ids", "id", "uids"])
        if ids_dataset is not None:
            ids = ids_dataset.tolist()
        if "features" in h5 and isinstance(h5["features"], h5py.Group):
            grp = h5["features"]
            if "feature_names" in grp:
                feature_names = _decode_feature_names(grp["feature_names"][...])

    return ids, features, feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical features stage from dictionary outputs"
    )
    parser.add_argument("--run", required=True, help="run id under runs/<run>")
    parser.add_argument(
        "--atlas",
        default=None,
        help="Optional atlas.json path (default runs/<run>/dictionary/outputs/atlas.json)",
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
        out_root = resolve_out_root(args.out_root)
        validate_run_id(args.run, out_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_dir = (out_root / args.run).resolve()

    atlas_path = Path(args.atlas) if args.atlas else run_dir / "dictionary" / "outputs" / "atlas.json"
    if not atlas_path.is_absolute():
        atlas_path = (Path.cwd() / atlas_path).resolve()

    dictionary_path = (run_dir / "dictionary" / "outputs" / "dictionary.h5").resolve()

    try:
        assert_within_runs(run_dir, atlas_path)
        assert_within_runs(run_dir, dictionary_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not atlas_path.exists():
        print(f"ERROR: no existe atlas: {atlas_path}", file=sys.stderr)
        return 1
    if not dictionary_path.exists():
        print(f"ERROR: no existe dictionary.h5: {dictionary_path}", file=sys.stderr)
        return 1

    try:
        atlas_ids, atlas_X, atlas_feature_names = _load_atlas_features(atlas_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        dict_ids, dict_X, dict_feature_names = _load_dictionary_features(dictionary_path)
    except ImportError:
        return 1

    if atlas_X is not None:
        X = atlas_X
    elif dict_X is not None:
        X = dict_X
    else:
        print(
            "ERROR: no se pudo derivar X desde atlas.json o dictionary.h5. "
            "Se esperaba 'points/theories' con features en atlas o datasets X/features/metrics en dictionary.h5.",
            file=sys.stderr,
        )
        return 1

    ids = dict_ids if dict_ids is not None else atlas_ids
    if ids is None:
        print(
            "ERROR: no se pudieron derivar ids desde dictionary.h5 ni atlas.json.",
            file=sys.stderr,
        )
        return 1
    if not ids:
        print(
            "ERROR: ids vacíos después de resolver atlas/dictionary. "
            "Verifica que atlas.json tenga puntos con id o dictionary.h5 contenga ids.",
            file=sys.stderr,
        )
        return 1

    if len(ids) != X.shape[0]:
        print(
            f"ERROR: ids ({len(ids)}) no coinciden con filas de X ({X.shape[0]}).",
            file=sys.stderr,
        )
        return 1

    feature_names = None
    if dict_feature_names and len(dict_feature_names) == X.shape[1]:
        feature_names = dict_feature_names
    elif dict_feature_names:
        print(
            "WARNING: feature_names en dictionary.h5 no coincide con X; "
            "usando fallback.",
            file=sys.stderr,
        )
    if feature_names is None and atlas_feature_names and len(atlas_feature_names) == X.shape[1]:
        feature_names = atlas_feature_names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, "features", base_dir=out_root)

    features_path = outputs_dir / "features.json"
    payload = {
        "run": args.run,
        "source": {
            "atlas_path": _display_path(atlas_path),
            "dictionary_path": _display_path(dictionary_path),
            "atlas_sha256": sha256_file(atlas_path),
            "dictionary_sha256": sha256_file(dictionary_path),
        },
        "feature_names": feature_names,
        "ids": ids,
        "X": X.tolist(),
    }

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "stage": "features",
        "version": __version__,
        "script": "05_build_features_stage.py",
        "run": args.run,
        "inputs": {
            "atlas_path": _display_path(atlas_path),
            "dictionary_path": _display_path(dictionary_path),
            "atlas_sha256": payload["source"]["atlas_sha256"],
            "dictionary_sha256": payload["source"]["dictionary_sha256"],
        },
        "outputs": {"features": "outputs/features.json"},
        "data": {"n_rows": len(ids), "n_features": int(X.shape[1])},
        "hashes": {"outputs/features.json": sha256_file(features_path)},
    }
    summary_path = write_stage_summary(stage_dir, summary)

    write_manifest(
        stage_dir,
        {
            "features": features_path,
            "summary": summary_path,
        },
        extra={"version": __version__},
    )

    print(f"features.json escrito en {features_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
