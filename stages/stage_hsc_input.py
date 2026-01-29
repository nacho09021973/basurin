#!/usr/bin/env python3
"""BASURIN — Derived canonical stage: HSC_INPUT.

Ensamble determinista de features + spectrum para alimentar hsc_detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basurin_io import assert_within_runs, resolve_out_root, sha256_file, validate_run_id


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _list_datasets(h5: h5py.File) -> list[str]:
    datasets: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    h5.visititems(visitor)
    return datasets


def _select_delta_dataset(dataset_paths: Iterable[str]) -> str | None:
    candidates = ["delta_uv", "Delta", "Delta_grid", "delta", "delta_grid"]
    ordered_paths = sorted(dataset_paths)
    for candidate in candidates:
        for path in ordered_paths:
            if Path(path).name == candidate:
                return path
    return None


def _load_spectrum_operators(spectrum_path: Path) -> tuple[list[dict[str, Any]], str]:
    with h5py.File(spectrum_path, "r") as h5:
        dataset_path = _select_delta_dataset(_list_datasets(h5))
        if dataset_path is None:
            raise ValueError("spectrum.h5 no contiene dataset delta_uv ni Delta grid.")
        data = np.asarray(h5[dataset_path][...], dtype=float).ravel()

    values = [float(x) for x in data if np.isfinite(x)]
    if not values:
        raise ValueError("spectrum.h5 no contiene valores finitos en delta_uv/Delta grid.")

    width = max(3, len(str(len(values) - 1)))
    operators = [
        {
            "id": f"op_{idx:0{width}d}",
            "dim": dim,
            "spin": 0,
            "degeneracy": 1,
        }
        for idx, dim in enumerate(values)
    ]
    return operators, dataset_path


def _load_features(features_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(features_path)
    if not isinstance(payload, dict):
        raise ValueError("features.json debe ser un objeto JSON.")
    metadata = payload.get("metadata", {})
    features = payload.get("features")
    if not isinstance(metadata, dict):
        raise ValueError("metadata en features.json debe ser dict.")
    if features is None:
        raise ValueError("features.json no contiene key 'features'.")
    if not isinstance(features, dict):
        raise ValueError("features en features.json debe ser dict.")
    return metadata, features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derived canonical stage HSC_INPUT")
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument("--runs-root", default="runs", help="Runs root path")
    parser.add_argument("--out-stage", default="HSC_INPUT", help="Output stage name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        runs_root = resolve_out_root(args.runs_root)
        validate_run_id(args.run, runs_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_dir = runs_root / args.run
    run_valid_path = run_dir / "RUN_VALID" / "outputs" / "run_valid.json"
    if not run_valid_path.exists():
        print(f"ERROR: RUN_VALID missing at {run_valid_path}", file=sys.stderr)
        return 2

    run_valid_payload = _read_json(run_valid_path)
    verdict = run_valid_payload.get("verdict")
    if verdict != "PASS":
        print(f"ERROR: RUN_VALID verdict {verdict}", file=sys.stderr)
        return 2

    features_path = run_dir / "features" / "outputs" / "features.json"
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    if not features_path.exists():
        print(f"ERROR: features.json missing at {features_path}", file=sys.stderr)
        return 2
    if not spectrum_path.exists():
        print(f"ERROR: spectrum.h5 missing at {spectrum_path}", file=sys.stderr)
        return 2

    try:
        assert_within_runs(run_dir, features_path)
        assert_within_runs(run_dir, spectrum_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    features_metadata, features = _load_features(features_path)
    try:
        operators, dataset_path = _load_spectrum_operators(spectrum_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    stage_dir = run_dir / args.out_stage
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    created_utc = datetime.now(timezone.utc).isoformat()
    features_rel = str(features_path.relative_to(run_dir))
    spectrum_rel = str(spectrum_path.relative_to(run_dir))

    metadata = dict(features_metadata)
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance.update(
        {
            "assembled_by": "stages/stage_hsc_input.py",
            "stage": args.out_stage,
            "created_utc": created_utc,
            "inputs": {
                "features": features_rel,
                "spectrum": spectrum_rel,
            },
            "spectrum_dataset": dataset_path,
        }
    )
    metadata["provenance"] = provenance

    input_payload = {
        "metadata": metadata,
        "features": features,
        "spectrum": {"operators": operators},
    }

    input_path = outputs_dir / "input.json"
    _write_json(input_path, input_payload)

    input_sha = sha256_file(input_path)
    run_valid_sha = sha256_file(run_valid_path)
    features_sha = sha256_file(features_path)
    spectrum_sha = sha256_file(spectrum_path)

    stage_summary_path = stage_dir / "stage_summary.json"
    stage_summary = {
        "stage": args.out_stage,
        "run": args.run,
        "created_utc": created_utc,
        "inputs": {
            "run_valid": {
                "path": str(run_valid_path.relative_to(run_dir)),
                "sha256": run_valid_sha,
                "verdict": verdict,
            },
            "features": {
                "path": features_rel,
                "sha256": features_sha,
            },
            "spectrum": {
                "path": spectrum_rel,
                "sha256": spectrum_sha,
                "dataset": dataset_path,
            },
        },
        "outputs": {
            "input": "outputs/input.json",
        },
        "hashes": {
            "outputs/input.json": input_sha,
        },
        "verdict": "PASS",
    }
    _write_json(stage_summary_path, stage_summary)
    stage_summary_sha = sha256_file(stage_summary_path)

    manifest_path = stage_dir / "manifest.json"
    manifest = {
        "stage": args.out_stage,
        "run": args.run,
        "created_utc": created_utc,
        "files": {
            "input": "outputs/input.json",
            "summary": "stage_summary.json",
            "manifest": "manifest.json",
        },
        "hashes": {
            "outputs/input.json": input_sha,
            "stage_summary.json": stage_summary_sha,
        },
    }
    _write_json(manifest_path, manifest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
