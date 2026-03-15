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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.contracts import init_stage, check_inputs, finalize, abort, log_stage_paths
from basurin_io import sha256_file, write_json_atomic

STAGE = "s2_ringdown_window"
DEFAULT_WINDOW_CATALOG = Path(__file__).resolve().parent / "assets" / "window_catalog_v1.json"
DEFAULT_T0_REFERENCE_CATALOG = Path(__file__).resolve().parents[1] / "gwtc_events_t0.json"
DEFAULT_EVENT_METADATA_DIR = Path(__file__).resolve().parents[1] / "docs" / "ringdown" / "event_metadata"
OFFLINE_T0_ERROR = "missing_t0_gps_offline: unable to resolve t0_gps from local sources"


def _stable_path(p: str | None) -> str | None:
    if not p:
        return None
    return Path(p).name


def _format_missing_t0_message(
    *,
    event_id: str,
    offline: bool,
    window_catalog_path: Path,
    sources_attempted: dict[str, Any],
    reason: str | None = None,
    stable_paths: bool = False,
) -> str:
    error_code = "missing_t0_gps_offline" if offline else "missing_t0_gps"
    window_catalog_display = str(window_catalog_path)
    sources_attempted_display = dict(sources_attempted)
    if stable_paths:
        window_catalog_display = _stable_path(window_catalog_display) or window_catalog_display
        for key in ("catalog_path", "legacy_windows_path", "run_cache_path"):
            if key in sources_attempted_display:
                sources_attempted_display[key] = _stable_path(sources_attempted_display.get(key))
    if offline and stable_paths:
        sources_str = json.dumps(sources_attempted_display, sort_keys=True)
        return f"{OFFLINE_T0_ERROR}; sources_attempted={sources_str}"
    parts = [
        f"{error_code}: event_id={event_id}",
        f"window_catalog={window_catalog_display}",
    ]
    if reason:
        normalized_reason = reason.strip()
        for prefix in ("missing_t0_gps_offline:", "missing_t0_gps:"):
            if normalized_reason.startswith(prefix):
                normalized_reason = normalized_reason[len(prefix):].strip()
                break
        if normalized_reason:
            parts.append(normalized_reason)
    parts.append(f"sources_attempted={json.dumps(sources_attempted_display, sort_keys=True)}")
    return "; ".join(parts)

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


def _extract_t0_from_catalog_entry(entry: Any) -> float | None:
    if isinstance(entry, dict):
        for key in ("t0_gps", "GPS", "gps", "event_time_gps", "t_coalescence_gps", "t0_ref_gps", "gpstime", "gps_time"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        t0_ref = entry.get("t0_ref")
        if isinstance(t0_ref, dict):
            for key in ("value_gps", "t0_gps", "gps"):
                value = t0_ref.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
    elif isinstance(entry, (int, float)):
        return float(entry)
    return None


def _resolve_t0_gps_from_event_metadata(
    lookup_keys: list[str],
) -> tuple[float, str, str] | None:
    if not DEFAULT_EVENT_METADATA_DIR.exists():
        return None
    for lookup_key in lookup_keys:
        metadata_path = DEFAULT_EVENT_METADATA_DIR / f"{lookup_key}_metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        t0_gps = _extract_t0_from_catalog_entry(payload)
        if t0_gps is not None:
            return t0_gps, str(metadata_path), lookup_key
    return None


def _resolve_t0_gps_from_reference_catalog(
    event_id: str,
    reference_catalog_path: Path,
    lookup_keys: list[str],
) -> tuple[float, str, str] | None:
    if not reference_catalog_path.exists():
        return None
    with open(reference_catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict):
        return None
    for lookup_key in lookup_keys:
        if lookup_key not in catalog:
            continue
        t0_gps = _extract_t0_from_catalog_entry(catalog[lookup_key])
        if t0_gps is not None:
            return t0_gps, str(reference_catalog_path), lookup_key
    return None


def _fetch_gwosc_event_gps(event_id: str, retries: int = 3, timeout_s: int = 20) -> float:
    last_error = ""
    url = f"https://gwosc.org/api/v2/events/{event_id}"
    headers = {"Accept": "application/json", "User-Agent": "basurin-s2-ringdown-window/1.0"}
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            events = payload.get("events")
            if isinstance(events, list) and events:
                data = events[0]
            elif isinstance(payload, dict):
                data = payload
            else:
                data = {}
            for key in ("gps", "t0_gps", "event_time_gps", "GPS", "gpstime", "gps_time"):
                if key in data:
                    return float(data[key])
            raise RuntimeError(f"GWOSC response missing gps keys for {event_id}: keys={sorted(data.keys())[:12]}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"GWOSC lookup failed for event_id={event_id!r} after {retries} retries: {last_error}")


def _read_or_create_gwosc_cache(run_dir: Path, event_id: str) -> tuple[float, str, bool]:
    cache_path = run_dir / "external_inputs" / "gwosc" / "event_time" / f"{event_id}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        for key in ("t0_gps", "gps", "event_time_gps"):
            if key in cached:
                return float(cached[key]), str(cache_path), False
        raise RuntimeError(f"Invalid GWOSC cache (missing t0 key): {cache_path}")

    t0_gps = _fetch_gwosc_event_gps(event_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cache_path, {
        "event_id": event_id,
        "t0_gps": float(t0_gps),
        "source": "gwosc_api_v2",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return float(t0_gps), str(cache_path), True


def _resolve_t0_gps(
    event_id: str,
    window_catalog_path: Path,
    *,
    offline: bool = False,
    run_dir: Path | None = None,
    stable_error_paths: bool = False,
) -> tuple[float, str, dict[str, Any], bool]:
    canonical_event_id = _canonical_event_id(event_id)
    lookup_keys = [event_id]
    if canonical_event_id != event_id:
        lookup_keys.append(canonical_event_id)

    reference_catalog_path = DEFAULT_T0_REFERENCE_CATALOG
    run_cache_path: str | None = None
    if run_dir is not None:
        run_cache_path = str(run_dir / "external_inputs" / "gwosc" / "event_time" / f"{event_id}.json")
    sources_attempted: dict[str, Any] = {
        "catalog_path": str(window_catalog_path) if window_catalog_path is not None else None,
        "reference_catalog_path": str(reference_catalog_path),
        "event_metadata_candidates": [
            str(DEFAULT_EVENT_METADATA_DIR / f"{lookup_key}_metadata.json")
            for lookup_key in lookup_keys
        ],
        "legacy_windows_path": str(window_catalog_path) if window_catalog_path is not None else None,
        "run_cache_path": run_cache_path,
        "online_fetch_enabled": (not offline),
        "offline": bool(offline),
        "keys_checked": list(lookup_keys),
    }

    try:
        uses_default_window_catalog = window_catalog_path.expanduser().resolve() == DEFAULT_WINDOW_CATALOG.resolve()
    except OSError:
        uses_default_window_catalog = window_catalog_path == DEFAULT_WINDOW_CATALOG

    # The default window catalog is a coarse fallback. When event-specific
    # metadata exists, prefer it to avoid ms-level timing loss for sensitive runs.
    if uses_default_window_catalog:
        event_metadata_lookup = _resolve_t0_gps_from_event_metadata(lookup_keys)
        if event_metadata_lookup is not None:
            t0_gps, source, lookup_key = event_metadata_lookup
            return t0_gps, source, {"lookup_key": lookup_key}, False

    if window_catalog_path.exists():
        with open(window_catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        schema, keys = _catalog_schema_info(catalog)

        if isinstance(catalog, dict):
            # Legacy schema: {"windows": [{"event_id": ..., "t0_ref": {"value_gps": ...}}]}
            legacy_event_ids: list[str] = []
            for w in catalog.get("windows", []):
                if not isinstance(w, dict):
                    continue
                legacy_event_id = w.get("event_id")
                if isinstance(legacy_event_id, str):
                    legacy_event_ids.append(legacy_event_id)
                if legacy_event_id not in lookup_keys:
                    continue
                t0_ref = w.get("t0_ref", {})
                if "value_gps" in t0_ref:
                    return (
                        float(t0_ref["value_gps"]),
                        str(window_catalog_path),
                        {"lookup_key": str(w.get("event_id"))},
                        False,
                    )

            if isinstance(catalog.get("windows"), list) and legacy_event_ids:
                requested = lookup_keys[0]
                if requested not in legacy_event_ids:
                    raise RuntimeError(
                        _format_missing_t0_message(
                            event_id=event_id,
                            offline=offline,
                            window_catalog_path=window_catalog_path,
                            sources_attempted=sources_attempted,
                            reason=(
                                "legacy_windows_event_mismatch: "
                                f"requested_event_id={requested}, "
                                f"available_event_ids={legacy_event_ids}, "
                                f"path={window_catalog_path}"
                            ),
                        )
                    )

            # Schema A: {"GW190521": {"t0_gps": 1242442967.4}}
            for lookup_key in lookup_keys:
                if lookup_key in catalog and isinstance(catalog[lookup_key], dict) and "t0_gps" in catalog[lookup_key]:
                    return float(catalog[lookup_key]["t0_gps"]), str(window_catalog_path), {"lookup_key": lookup_key}, False

            # Schema B: {"GW190521": 1242442967.4}
            for lookup_key in lookup_keys:
                if lookup_key in catalog and isinstance(catalog[lookup_key], (int, float)):
                    return float(catalog[lookup_key]), str(window_catalog_path), {"lookup_key": lookup_key}, False

    reference_lookup = _resolve_t0_gps_from_reference_catalog(
        event_id,
        reference_catalog_path,
        lookup_keys,
    )
    if reference_lookup is not None:
        t0_gps, source, lookup_key = reference_lookup
        return t0_gps, source, {"lookup_key": lookup_key}, False

    if offline:
        raise RuntimeError(
            _format_missing_t0_message(
                event_id=event_id,
                offline=True,
                window_catalog_path=window_catalog_path,
                sources_attempted=sources_attempted,
                reason=OFFLINE_T0_ERROR,
                stable_paths=stable_error_paths,
            )
        )

    effective_run_dir = run_dir if run_dir is not None else Path("runs") / "_s2_tmp"
    t0_gps, source, fetched = _read_or_create_gwosc_cache(effective_run_dir, event_id)
    return t0_gps, source, {"lookup_key": event_id, "gwosc_cache_path": source}, (not fetched)


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: crop ringdown window")
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-root", default=None, help="Override BASURIN_RUNS_ROOT for this invocation")
    ap.add_argument("--event-id", default="GW150914")
    ap.add_argument("--dt-start-s", type=float, default=0.003)
    ap.add_argument("--duration-s", type=float, default=0.06)
    ap.add_argument(
        "--clip-window",
        action="store_true",
        help="Clip per-detector window indices to valid [0, n] range instead of failing on out-of-range.",
    )
    ap.add_argument("--window-catalog", default=str(DEFAULT_WINDOW_CATALOG))
    ap.add_argument("--offline", action="store_true", default=False)
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
        "offline": args.offline,
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

        t0_gps, t0_source, t0_details, t0_used_cache = _resolve_t0_gps(
            args.event_id,
            Path(args.window_catalog),
            offline=bool(args.offline),
            run_dir=ctx.run_dir,
            stable_error_paths=bool(args.offline),
        )
        event_id_lookup_key = str(t0_details.get("lookup_key", args.event_id))
        gwosc_cache_path: str | None = None
        t0_gwosc_cache = t0_details.get("gwosc_cache_path")
        if isinstance(t0_gwosc_cache, str):
            gwosc_cache_path = t0_gwosc_cache
        t0_source_path = Path(t0_source)
        if t0_source_path.exists() and t0_source_path.is_absolute():
            try:
                rel = t0_source_path.relative_to(ctx.run_dir)
                if rel.parts[:3] == ("external_inputs", "gwosc", "event_time"):
                    gwosc_cache_path = str(t0_source_path)
            except ValueError:
                pass
        if gwosc_cache_path is not None:
            gwosc_cache = Path(gwosc_cache_path)
            ctx.inputs_record.append({
                "label": "gwosc_event_time",
                "path": str(gwosc_cache.relative_to(ctx.run_dir)),
                "sha256": sha256_file(gwosc_cache),
            })
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
        window_status: dict[str, dict[str, Any]] = {}
        window_errors: list[str] = []
        n_out = 0
        any_t0_clipped = False
        for det in detectors:
            strain = np.asarray(data[det], dtype=np.float64)
            if strain.ndim != 1:
                abort(ctx, f"{det} strain is not 1-D: shape={strain.shape}")
            if not np.all(np.isfinite(strain)):
                abort(ctx, f"{det} strain contains NaN/Inf")

            i_start = int(round((t_start_gps - gps_start) * fs))
            n_out = int(round(args.duration_s * fs))
            t_start_det = t_start_gps
            t0_det = t0_gps

            if args.clip_window and strain.size > 0:
                i_start_clamped = max(0, min(i_start, int(strain.size) - 1))
                if i_start_clamped != i_start:
                    any_t0_clipped = True
                    i_start = i_start_clamped
                    t_start_det = gps_start + (i_start / fs)
                    t0_det = t_start_det - args.dt_start_s

            i_end = i_start + n_out
            if i_start < 0 or i_end > strain.size:
                err = f"Window out of range for {det}: i_start={i_start}, i_end={i_end}, n={strain.size}"
                status: dict[str, Any] = {
                    "ok": False,
                    "reason": "out_of_range",
                    "i_start": i_start,
                    "i_end": i_end,
                    "n": int(strain.size),
                }
                if args.clip_window:
                    i_start_clipped = max(0, min(i_start, int(strain.size)))
                    i_end_clipped = max(0, min(i_end, int(strain.size)))
                    status.update({
                        "clipped": True,
                        "t0_clipped": bool(t0_det != t0_gps),
                        "t0_gps_original": float(t0_gps),
                        "t0_gps_used": float(t0_det),
                        "t_start_gps_original": float(t_start_gps),
                        "t_start_gps_used": float(t_start_det),
                        "i_start_clipped": i_start_clipped,
                        "i_end_clipped": i_end_clipped,
                        "clip_left_samples": max(0, -i_start),
                        "clip_right_samples": max(0, i_end - int(strain.size)),
                    })
                    if i_end_clipped > i_start_clipped:
                        out_path = ctx.outputs_dir / f"{det}_rd.npz"
                        np.savez(out_path, strain=strain[i_start_clipped:i_end_clipped].copy(),
                                 gps_start=np.float64(gps_start + (i_start_clipped / fs)),
                                 duration_s=np.float64((i_end_clipped - i_start_clipped) / fs),
                                 sample_rate_hz=np.float64(fs))
                        artifacts[f"{det}_rd"] = out_path
                        status["ok"] = True
                        status["reason"] = "clipped"
                    else:
                        status["reason"] = "empty_after_clip"
                        window_errors.append(
                            f"{err}; clip produced empty window (i_start_clipped={i_start_clipped}, "
                            f"i_end_clipped={i_end_clipped})"
                        )
                else:
                    window_errors.append(err)
                window_status[det] = status
                continue

            out_path = ctx.outputs_dir / f"{det}_rd.npz"
            np.savez(out_path, strain=strain[i_start:i_end].copy(),
                     gps_start=np.float64(t_start_gps), duration_s=np.float64(args.duration_s),
                     sample_rate_hz=np.float64(fs))
            artifacts[f"{det}_rd"] = out_path
            window_status[det] = {
                "ok": True,
                "reason": "ok",
                "i_start": i_start,
                "i_end": i_end,
                "n": int(strain.size),
                "clipped": False,
                "t0_clipped": bool(t0_det != t0_gps),
                "t0_gps_original": float(t0_gps),
                "t0_gps_used": float(t0_det),
            }

        window_meta = {
            "event_id": args.event_id, "t0_gps": t0_gps, "t0_source": t0_source,
            "event_id_lookup_key": event_id_lookup_key,
            "t0_details": t0_details,
            "t0_used_cache": bool(t0_used_cache),
            "gwosc_cache_path": gwosc_cache_path,
            "dt_start_s": args.dt_start_s, "duration_s": args.duration_s,
            "t_start_gps": t_start_gps, "t_end_gps": t_end_gps,
            "sample_rate_hz": fs, "detectors": detectors, "n_samples": n_out,
            "clip_window": bool(args.clip_window),
            "t0_clipped": bool(any_t0_clipped),
            "window_status": window_status,
        }
        meta_path = ctx.outputs_dir / "window_meta.json"
        write_json_atomic(meta_path, window_meta)
        artifacts["window_meta"] = meta_path

        _ensure_window_meta_contract(ctx, artifacts, strain_path)
        verdict = "FAIL" if window_errors else "PASS"
        extra_summary = {"error": "; ".join(window_errors)} if window_errors else None
        finalize(ctx, artifacts, verdict=verdict, results=window_meta, extra_summary=extra_summary)
        log_stage_paths(ctx)
        if window_errors:
            return 2
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        try:
            abort(ctx, str(exc))
        except SystemExit:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
