#!/usr/bin/env python3
"""
stages/ringdown_real_observables_v0_stage.py
-----------------------------------------
Canonical stage: derive minimal observables from real ringdown window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- BASURIN import bootstrap (no depende de PYTHONPATH) ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1], _here.parents[2]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break
# -----------------------------------------------------------

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

EXIT_CONTRACT_FAIL = 2
STAGE_NAME_DEFAULT = "ringdown_real_observables_v0"
BAND_SANITY_HZ = [150, 400]


def _abort(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_t0_gps(payload: Any) -> float | None:
    if isinstance(payload, dict):
        for key in ["t0_gps", "t0", "gps_start", "start_gps"]:
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    return None
        if "segments" in payload and isinstance(payload["segments"], list):
            payload = payload["segments"]
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, (list, tuple)) and first:
            try:
                return float(first[0])
            except (TypeError, ValueError):
                return None
        if isinstance(first, dict):
            for key in ["t0_gps", "t0", "gps_start", "start_gps"]:
                if key in first:
                    try:
                        return float(first[key])
                    except (TypeError, ValueError):
                        return None
    return None


def _select_strain_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    keys = list(npz.files)
    for key in ["strain", "h"]:
        if key in npz:
            return np.asarray(npz[key], dtype=float)
    for key in keys:
        arr = np.asarray(npz[key])
        if arr.ndim == 1:
            return np.asarray(arr, dtype=float)
    raise ValueError("npz has no 1D strain-like array")


def _extract_fs(npz: np.lib.npyio.NpzFile, strain: np.ndarray) -> float | None:
    for key in ["fs", "sample_rate", "sample_rate_hz"]:
        if key in npz:
            value = np.asarray(npz[key]).reshape(-1)[0]
            return float(value)
    for key in ["dt", "delta_t"]:
        if key in npz:
            value = float(np.asarray(npz[key]).reshape(-1)[0])
            if value > 0:
                return float(1.0 / value)
    if "t" in npz:
        t = np.asarray(npz["t"], dtype=float)
        if t.size > 1:
            dt = float(np.median(np.diff(t)))
            if dt > 0:
                return float(1.0 / dt)
    if strain.size > 1:
        return None
    return None


def _load_strain_metrics(path: Path) -> dict[str, Any]:
    try:
        data = np.load(path)
    except Exception as exc:
        raise RuntimeError(f"no se pudo leer npz {path}: {exc}") from exc

    strain = _select_strain_array(data)
    if strain.ndim != 1:
        raise ValueError(f"strain array must be 1D in {path}")
    if not np.all(np.isfinite(strain)):
        raise ValueError(f"strain contains NaN/Inf in {path}")

    fs = _extract_fs(data, strain)
    rms = float(np.sqrt(np.mean(np.square(strain))))
    peak_abs = float(np.max(np.abs(strain)))

    return {
        "strain": strain,
        "fs": fs,
        "n_samples": int(strain.size),
        "rms": rms,
        "peak_abs": peak_abs,
    }


def _write_failure(
    stage_dir: Path,
    stage_name: str,
    run_id: str,
    params: dict[str, Any],
    inputs: list[dict[str, str]],
    reason: str,
) -> None:
    summary = {
        "stage": stage_name,
        "run": run_id,
        "created": utc_now_iso(),
        "version": "v0",
        "parameters": params,
        "inputs": inputs,
        "outputs": [],
        "verdict": "FAIL",
        "error": reason,
    }
    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {"stage_summary": summary_path},
        extra={"inputs": inputs, "verdict": "FAIL", "error": reason},
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Canonic stage: minimal observables from real ringdown window"
    )
    ap.add_argument("--run", required=True, help="run_id")
    ap.add_argument(
        "--stage-name",
        default=STAGE_NAME_DEFAULT,
        help=f"stage name (default: {STAGE_NAME_DEFAULT})",
    )
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = (out_root / args.run).resolve()

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        _abort(str(exc))

    stage_name = args.stage_name
    stage_dir, outputs_dir = ensure_stage_dirs(args.run, stage_name, base_dir=out_root)

    inputs_dir = run_dir / "ringdown_real_ringdown_window" / "outputs"
    input_paths = {
        "H1": inputs_dir / "H1_rd.npz",
        "L1": inputs_dir / "L1_rd.npz",
        "segments": inputs_dir / "segments_rd.json",
    }

    missing = [str(path) for path in input_paths.values() if not path.exists()]
    params = {"run": args.run, "stage_name": stage_name}
    inputs_list: list[dict[str, str]] = []
    for label, path in input_paths.items():
        if path.exists():
            inputs_list.append({"path": str(path.relative_to(run_dir)), "sha256": sha256_file(path)})
        else:
            inputs_list.append({"path": str(path.relative_to(run_dir)), "sha256": ""})

    if missing:
        reason = f"missing inputs: {', '.join(missing)}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    try:
        segments_payload = _read_json(input_paths["segments"])
    except Exception as exc:
        reason = f"no se pudo leer segments_rd.json {input_paths['segments']}: {exc}"
        _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
        _abort(reason)

    t0_gps = _extract_t0_gps(segments_payload)

    detectors = ["H1", "L1"]
    metrics: dict[str, dict[str, Any]] = {}
    for det in detectors:
        try:
            metrics[det] = _load_strain_metrics(input_paths[det])
        except Exception as exc:
            reason = f"error leyendo {input_paths[det]}: {exc}"
            _write_failure(stage_dir, stage_name, args.run, params, inputs_list, reason)
            _abort(reason)

    fs_candidates = [metrics[det]["fs"] for det in detectors if metrics[det]["fs"] is not None]
    fs_hz = None
    if fs_candidates:
        fs_hz = float(fs_candidates[0])

    observables = {
        "run_id": args.run,
        "detectors": detectors,
        "fs_hz": fs_hz,
        "n_samples": {det: metrics[det]["n_samples"] for det in detectors},
        "t0_gps": t0_gps,
        "band_sanity_hz": BAND_SANITY_HZ,
        "rms": {det: metrics[det]["rms"] for det in detectors},
        "peak_abs": {det: metrics[det]["peak_abs"] for det in detectors},
    }

    observables_path = outputs_dir / "observables.jsonl"
    with open(observables_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(observables, sort_keys=True))
        f.write("\n")

    outputs_list = [
        {
            "path": str(observables_path.relative_to(run_dir)),
            "sha256": sha256_file(observables_path),
        }
    ]

    summary = {
        "stage": stage_name,
        "run": args.run,
        "created": utc_now_iso(),
        "version": "v0",
        "parameters": params,
        "inputs": inputs_list,
        "outputs": outputs_list,
        "verdict": "PASS",
        "format": {
            "observables_jsonl": "single_record_per_run",
            "detectors": detectors,
        },
    }

    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {"observables": observables_path, "stage_summary": summary_path},
        extra={"inputs": inputs_list},
    )

    print(f"OK: {stage_name} PASS")
    print(f"  outputs: {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
