#!/usr/bin/env python3
"""MVP Stage 2: Crop ringdown window from full strain.

CLI:
    python mvp/s2_ringdown_window.py --run <run_id> --event-id GW150914 \
        [--dt-start-s 0.003] [--duration-s 0.06]

Inputs (from s1_fetch_strain):
    runs/<run>/s1_fetch_strain/outputs/strain.npz

Outputs (runs/<run>/s2_ringdown_window/outputs/):
    H1_rd.npz           Windowed strain for H1
    L1_rd.npz           Windowed strain for L1
    window_meta.json     Window parameters and provenance

Contracts:
    - Window must fall within the strain time range.
    - Windowed arrays must be 1-D, finite, >0 samples.
    - t0 resolved from window_catalog or event_metadata (deterministic).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "s2_ringdown_window"
UPSTREAM_STAGE = "s1_fetch_strain"
EXIT_CONTRACT_FAIL = 2


def _abort(message: str) -> None:
    print(f"[{STAGE_NAME}] ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _resolve_t0_gps(event_id: str, window_catalog_path: Path) -> tuple[float, str]:
    """Resolve coalescence GPS time from catalog or event metadata."""
    # Try window catalog
    if window_catalog_path.exists():
        with open(window_catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        windows = catalog.get("windows", [])
        for w in windows:
            if isinstance(w, dict) and w.get("event_id") == event_id:
                t0_ref = w.get("t0_ref", {})
                if "value_gps" in t0_ref:
                    return float(t0_ref["value_gps"]), str(window_catalog_path)

    # Try local event metadata
    meta_path = Path("docs/ringdown/event_metadata") / f"{event_id}_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("t_coalescence_gps", "t0_ref_gps", "GPS"):
            if key in meta:
                return float(meta[key]), str(meta_path)

    raise RuntimeError(
        f"Cannot resolve t0_gps for {event_id}. "
        f"Checked: {window_catalog_path}, {meta_path}"
    )


def _write_failure(stage_dir: Path, run_id: str, params: dict, inputs: list, reason: str) -> None:
    summary = {
        "stage": STAGE_NAME, "run": run_id, "created": utc_now_iso(),
        "version": "v1", "parameters": params, "inputs": inputs, "outputs": [],
        "verdict": "FAIL", "error": reason,
    }
    sp = write_stage_summary(stage_dir, summary)
    write_manifest(stage_dir, {"stage_summary": sp}, extra={"verdict": "FAIL", "error": reason})


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE_NAME}: crop ringdown window")
    ap.add_argument("--run", required=True)
    ap.add_argument("--event-id", default="GW150914")
    ap.add_argument("--dt-start-s", type=float, default=0.003,
                     help="Offset from t0 (coalescence) to start of ringdown window")
    ap.add_argument("--duration-s", type=float, default=0.06,
                     help="Ringdown window duration in seconds")
    ap.add_argument("--window-catalog", default="docs/ringdown/window_catalog_v1.json")
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)
    run_dir = out_root / args.run

    try:
        require_run_valid(out_root, args.run)
    except Exception as exc:
        _abort(str(exc))

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)

    # Validate inputs from s1
    strain_npz_path = run_dir / UPSTREAM_STAGE / "outputs" / "strain.npz"
    if not strain_npz_path.exists():
        _abort(f"Missing upstream output: {strain_npz_path}")

    params: dict[str, Any] = {
        "event_id": args.event_id,
        "dt_start_s": args.dt_start_s,
        "duration_s": args.duration_s,
        "window_catalog": args.window_catalog,
    }
    inputs_list = [{"path": str(strain_npz_path.relative_to(run_dir)), "sha256": sha256_file(strain_npz_path)}]

    try:
        # Resolve t0
        t0_gps, t0_source = _resolve_t0_gps(args.event_id, Path(args.window_catalog))
        t_start_gps = t0_gps + args.dt_start_s
        t_end_gps = t_start_gps + args.duration_s

        # Load strain
        data = np.load(strain_npz_path)
        gps_start = float(np.asarray(data["gps_start"]).flat[0])
        fs = float(np.asarray(data["sample_rate_hz"]).flat[0])

        if fs <= 0:
            _abort(f"Invalid sample_rate_hz: {fs}")
        if args.duration_s <= 0:
            _abort(f"Invalid duration_s: {args.duration_s}")

        # Determine detectors present in the npz
        detectors = [k for k in data.files if k in ("H1", "L1", "V1")]
        if not detectors:
            _abort("No detector arrays found in strain.npz")

        # Crop window for each detector
        artifacts: dict[str, Path] = {}
        outputs_list: list[dict[str, str]] = []

        for det in detectors:
            strain = np.asarray(data[det], dtype=np.float64)
            if strain.ndim != 1:
                _abort(f"{det} strain is not 1-D: shape={strain.shape}")
            if not np.all(np.isfinite(strain)):
                _abort(f"{det} strain contains NaN/Inf")

            i_start = int(round((t_start_gps - gps_start) * fs))
            n_out = int(round(args.duration_s * fs))
            i_end = i_start + n_out

            if i_start < 0 or i_end > strain.size:
                _abort(
                    f"Window out of range for {det}: "
                    f"i_start={i_start}, i_end={i_end}, n_total={strain.size}"
                )

            windowed = strain[i_start:i_end].copy()
            out_path = outputs_dir / f"{det}_rd.npz"
            np.savez(
                out_path,
                strain=windowed,
                gps_start=np.float64(t_start_gps),
                duration_s=np.float64(args.duration_s),
                sample_rate_hz=np.float64(fs),
            )
            artifacts[f"{det}_rd"] = out_path
            outputs_list.append({
                "path": str(out_path.relative_to(run_dir)),
                "sha256": sha256_file(out_path),
            })

        # Write window metadata
        window_meta = {
            "event_id": args.event_id,
            "t0_gps": t0_gps,
            "t0_source": t0_source,
            "dt_start_s": args.dt_start_s,
            "duration_s": args.duration_s,
            "t_start_gps": t_start_gps,
            "t_end_gps": t_end_gps,
            "sample_rate_hz": fs,
            "detectors": detectors,
            "n_samples": n_out,
        }
        meta_path = outputs_dir / "window_meta.json"
        write_json_atomic(meta_path, window_meta)
        artifacts["window_meta"] = meta_path
        outputs_list.append({
            "path": str(meta_path.relative_to(run_dir)),
            "sha256": sha256_file(meta_path),
        })

        # Summary + manifest
        summary = {
            "stage": STAGE_NAME, "run": args.run, "created": utc_now_iso(),
            "version": "v1", "parameters": params, "inputs": inputs_list,
            "outputs": outputs_list, "verdict": "PASS",
            "window": window_meta,
        }
        sp = write_stage_summary(stage_dir, summary)
        artifacts["stage_summary"] = sp
        write_manifest(stage_dir, artifacts, extra={"inputs": inputs_list})

        print(f"OK: {STAGE_NAME} PASS ({args.event_id}, {detectors}, {n_out} samples)")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        _write_failure(stage_dir, args.run, params, inputs_list, str(exc))
        _abort(str(exc))
        return EXIT_CONTRACT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
