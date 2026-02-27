#!/usr/bin/env python3
"""MVP Stage 2: Crop ringdown window from full strain.

CLI:
    python mvp/s2_ringdown_window.py --run <run_id> --event-id GW150914 \
        [--dt-start-s 0.003] [--duration-s 0.06]

Inputs:  runs/<run>/s1_fetch_strain/outputs/strain.npz
Outputs: runs/<run>/s2_ringdown_window/outputs/{H1,L1}_rd.npz + window_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.contracts import init_stage, check_inputs, finalize, abort
from basurin_io import sha256_file, write_json_atomic

STAGE = "s2_ringdown_window"
DEFAULT_WINDOW_CATALOG = Path(__file__).resolve().parent / "assets" / "window_catalog_v1.json"

def _ensure_window_meta_contract(ctx: Any, artifacts: dict[str, Path], strain_path: Path) -> None:
    ctx.outputs_dir.mkdir(parents=True, exist_ok=True)
    meta_path = ctx.outputs_dir / "window_meta.json"
    if not ctx.inputs_record and strain_path.exists():
        try:
            rel = str(strain_path.relative_to(ctx.run_dir))
        except ValueError:
            rel = str(strain_path)
        ctx.inputs_record = [{"label": "strain_npz", "path": rel, "sha256": sha256_file(strain_path)}]
    if not artifacts or not meta_path.exists() or "window_meta" not in artifacts:
        abort(ctx, "PASS_WITHOUT_OUTPUTS")

def _catalog_schema_info(catalog: Any, max_keys: int = 10) -> tuple[str, list[str]]:
    if isinstance(catalog, dict):
        keys = [str(k) for k in list(catalog.keys())[:max_keys]]
        if "windows" in catalog and isinstance(catalog.get("windows"), list):
            return "legacy_windows", keys
        nested_dict_values = [v for v in catalog.values() if isinstance(v, dict)]
        scalar_values = [v for v in catalog.values() if not isinstance(v, dict)]
        if nested_dict_values and not scalar_values:
            return "event_map_nested", keys
        if scalar_values and not nested_dict_values:
            return "event_map_scalar", keys
        return "dict_mixed", keys
    return type(catalog).__name__, []


def _canonical_event_id(event_id: str) -> str:
    return event_id.split("_", 1)[0] if "_" in event_id else event_id


def _resolve_t0_gps(event_id: str, window_catalog_path: Path) -> tuple[float, str, str]:
    canonical_event_id = _canonical_event_id(event_id)
    lookup_keys = [event_id]
    if canonical_event_id != event_id:
        lookup_keys.append(canonical_event_id)

    if window_catalog_path.exists():
        with open(window_catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        schema, keys = _catalog_schema_info(catalog)

        if isinstance(catalog, dict):
            # Legacy schema: {"windows": [{"event_id": ..., "t0_ref": {"value_gps": ...}}]}
            for w in catalog.get("windows", []):
                if not isinstance(w, dict):
                    continue
                if w.get("event_id") not in lookup_keys:
                    continue
                t0_ref = w.get("t0_ref", {})
                if "value_gps" in t0_ref:
                    return float(t0_ref["value_gps"]), str(window_catalog_path), str(w.get("event_id"))

            # Schema A: {"GW190521": {"t0_gps": 1242442967.4}}
            for lookup_key in lookup_keys:
                if lookup_key in catalog and isinstance(catalog[lookup_key], dict) and "t0_gps" in catalog[lookup_key]:
                    return float(catalog[lookup_key]["t0_gps"]), str(window_catalog_path), lookup_key

            # Schema B: {"GW190521": 1242442967.4}
            for lookup_key in lookup_keys:
                if lookup_key in catalog and isinstance(catalog[lookup_key], (int, float)):
                    return float(catalog[lookup_key]), str(window_catalog_path), lookup_key

        canonical_detail = ""
        if canonical_event_id != event_id:
            canonical_detail = f"; canonical_event_id={canonical_event_id!r}"
        raise RuntimeError(
            "Cannot resolve t0_gps from window catalog "
            f"for event_id={event_id!r}{canonical_detail}; catalog_path={window_catalog_path}; "
            f"detected_schema={schema}; available_keys(first {len(keys)}): {keys}"
        )

    meta_path = Path("docs/ringdown/event_metadata") / f"{event_id}_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("t_coalescence_gps", "t0_ref_gps", "GPS"):
            if key in meta:
                return float(meta[key]), str(meta_path), event_id
    canonical_detail = ""
    if canonical_event_id != event_id:
        canonical_detail = f"; canonical_event_id={canonical_event_id!r}"
    raise RuntimeError(
        f"Cannot resolve t0_gps for event_id={event_id!r}{canonical_detail}; "
        f"catalog_path={window_catalog_path}; detected_schema=missing_catalog; available_keys(first 0): []"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: crop ringdown window")
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT for this invocation")
    ap.add_argument("--event-id", default="GW150914")
    ap.add_argument("--dt-start-s", type=float, default=0.003)
    ap.add_argument("--duration-s", type=float, default=0.06)
    ap.add_argument("--window-catalog", default=str(DEFAULT_WINDOW_CATALOG))
    ap.add_argument(
        "--strain-npz",
        default=None,
        help=(
            "Optional explicit path to the full-strain NPZ. "
            "Defaults to <runs_root>/<run>/s1_fetch_strain/outputs/strain.npz"
        ),
    )
    args = ap.parse_args()

    run_id = args.run_id or args.run
    if not run_id:
        ap.error("one of --run or --run-id is required")
    if args.runs_root:
        os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).expanduser().resolve())

    ctx = init_stage(run_id, STAGE, params={
        "event_id": args.event_id, "dt_start_s": args.dt_start_s,
        "duration_s": args.duration_s, "window_catalog": args.window_catalog,
    })

    default_strain_path = ctx.run_dir / "s1_fetch_strain" / "outputs" / "strain.npz"
    strain_path = Path(args.strain_npz).expanduser().resolve() if args.strain_npz else default_strain_path

    try:
        check_inputs(ctx, {"strain_npz": strain_path})
    except SystemExit:
        if not args.strain_npz and not default_strain_path.exists():
            runs_root = os.environ.get("BASURIN_RUNS_ROOT", "<cwd>/runs")
            abort(
                ctx,
                (
                    "Missing required inputs: "
                    f"strain_npz: {default_strain_path}. "
                    "Hint: this run may be a subrun containing only pre-trimmed s2 outputs. "
                    "Re-run with --strain-npz pointing to the original full-strain file from s1_fetch_strain, "
                    f"or set BASURIN_RUNS_ROOT correctly (current={runs_root!r})."
                ),
            )
        raise

    try:
        import numpy as np

        t0_gps, t0_source, event_id_lookup_key = _resolve_t0_gps(args.event_id, Path(args.window_catalog))
        t_start_gps = t0_gps + args.dt_start_s
        t_end_gps = t_start_gps + args.duration_s

        data = np.load(strain_path)
        gps_start = float(np.asarray(data["gps_start"]).flat[0])
        fs = float(np.asarray(data["sample_rate_hz"]).flat[0])
        if fs <= 0:
            abort(ctx, f"Invalid sample_rate_hz: {fs}")
        if args.duration_s <= 0:
            abort(ctx, f"Invalid duration_s: {args.duration_s}")

        detectors = [k for k in data.files if k in ("H1", "L1", "V1")]
        if not detectors:
            abort(ctx, "No detector arrays found in strain.npz")

        artifacts: dict[str, Path] = {}
        n_out = 0
        for det in detectors:
            strain = np.asarray(data[det], dtype=np.float64)
            if strain.ndim != 1:
                abort(ctx, f"{det} strain is not 1-D: shape={strain.shape}")
            if not np.all(np.isfinite(strain)):
                abort(ctx, f"{det} strain contains NaN/Inf")

            i_start = int(round((t_start_gps - gps_start) * fs))
            n_out = int(round(args.duration_s * fs))
            i_end = i_start + n_out
            if i_start < 0 or i_end > strain.size:
                abort(ctx, f"Window out of range for {det}: i_start={i_start}, i_end={i_end}, n={strain.size}")

            out_path = ctx.outputs_dir / f"{det}_rd.npz"
            np.savez(out_path, strain=strain[i_start:i_end].copy(),
                     gps_start=np.float64(t_start_gps), duration_s=np.float64(args.duration_s),
                     sample_rate_hz=np.float64(fs))
            artifacts[f"{det}_rd"] = out_path

        window_meta = {
            "event_id": args.event_id, "t0_gps": t0_gps, "t0_source": t0_source,
            "event_id_lookup_key": event_id_lookup_key,
            "dt_start_s": args.dt_start_s, "duration_s": args.duration_s,
            "t_start_gps": t_start_gps, "t_end_gps": t_end_gps,
            "sample_rate_hz": fs, "detectors": detectors, "n_samples": n_out,
        }
        meta_path = ctx.outputs_dir / "window_meta.json"
        write_json_atomic(meta_path, window_meta)
        artifacts["window_meta"] = meta_path

        _ensure_window_meta_contract(ctx, artifacts, strain_path)
        finalize(ctx, artifacts, results=window_meta)
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
