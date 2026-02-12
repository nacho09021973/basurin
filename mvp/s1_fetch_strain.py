#!/usr/bin/env python3
"""MVP Stage 1: Fetch real GWOSC strain for H1 and L1 per event.

CLI:
    python mvp/s1_fetch_strain.py --run <run_id> --event-id GW150914 \
        [--detectors H1,L1] [--duration-s 32]

Outputs (runs/<run>/s1_fetch_strain/outputs/):
    strain.npz          H1, L1 arrays + sample_rate_hz + gps_start + duration_s
    provenance.json     Full download provenance with SHA256 hashes

Contracts:
    - strain.npz MUST contain 1-D float64 arrays for each detector.
    - sample_rate_hz must be identical across detectors.
    - provenance.json records source URL, GPS, library version.
"""
from __future__ import annotations

import argparse
import hashlib
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
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "s1_fetch_strain"
EXIT_CONTRACT_FAIL = 2


def _abort(message: str) -> None:
    print(f"[{STAGE_NAME}] ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def _fetch_via_gwpy(detector: str, gps_start: float, duration_s: float) -> tuple[np.ndarray, float, str]:
    """Download strain using gwpy.timeseries.TimeSeries.fetch_open_data."""
    try:
        from gwpy.timeseries import TimeSeries
        from gwpy import __version__ as gwpy_ver
    except ImportError as exc:
        raise RuntimeError(
            "gwpy not installed. Install with: pip install gwpy. "
            "Or use --synthetic for offline testing."
        ) from exc

    ts = TimeSeries.fetch_open_data(detector, gps_start, gps_start + duration_s, verbose=False)
    values = np.asarray(ts.value, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Invalid strain shape for {detector}: {values.shape}")
    return values, float(ts.sample_rate.value), gwpy_ver


def _generate_synthetic_strain(
    detector: str,
    gps_start: float,
    duration_s: float,
    fs: float = 4096.0,
    f0: float = 251.0,
    tau: float = 0.004,
    seed_offset: int = 0,
) -> tuple[np.ndarray, float, str]:
    """Generate a synthetic damped-sinusoid ringdown for offline testing."""
    seed = 42 + hash(detector) + seed_offset
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n) / fs

    # Place ringdown at center of the window
    t_ring = duration_s / 2.0
    dt = t - t_ring
    signal = np.where(dt >= 0, np.exp(-dt / tau) * np.cos(2 * np.pi * f0 * dt), 0.0)

    # Add coloured noise (1/f shaped)
    noise = rng.normal(0, 1e-21, n)
    strain = signal * 1e-21 + noise
    return strain.astype(np.float64), fs, "synthetic_v1"


def _fetch_gps_center(event_id: str) -> float:
    """Resolve GPS center time from local metadata or GWOSC API."""
    # Try local metadata first
    local = Path("docs/ringdown/event_metadata") / f"{event_id}_metadata.json"
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("t_coalescence_gps", "gps", "GPS", "gpstime"):
            if key in meta:
                return float(meta[key])

    # Fall back to GWOSC API
    try:
        import requests
        url = f"https://gwosc.org/api/v2/events/{event_id}"
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Navigate to GPS time
        versions = data.get("event_versions") or data.get("versions") or []
        if versions and isinstance(versions, list):
            v = versions[-1]
            detail_url = v.get("detail_url") or v.get("url")
            if detail_url:
                if detail_url.startswith("/"):
                    detail_url = f"https://gwosc.org{detail_url}"
                vresp = requests.get(detail_url, headers={"Accept": "application/json"}, timeout=30)
                vresp.raise_for_status()
                vdata = vresp.json()
                for key in ("GPS", "gps", "gpstime", "gps_time"):
                    if key in vdata:
                        return float(vdata[key])
        for key in ("GPS", "gps", "gpstime"):
            if key in data:
                return float(data[key])
    except Exception:
        pass

    raise RuntimeError(
        f"Cannot resolve GPS center for {event_id}. "
        f"Add metadata to docs/ringdown/event_metadata/{event_id}_metadata.json"
    )


def _write_failure(stage_dir: Path, run_id: str, params: dict, reason: str) -> None:
    summary = {
        "stage": STAGE_NAME, "run": run_id, "created": utc_now_iso(),
        "version": "v1", "parameters": params, "inputs": [], "outputs": [],
        "verdict": "FAIL", "error": reason,
    }
    sp = write_stage_summary(stage_dir, summary)
    write_manifest(stage_dir, {"stage_summary": sp}, extra={"verdict": "FAIL", "error": reason})


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE_NAME}: fetch GWOSC strain")
    ap.add_argument("--run", required=True)
    ap.add_argument("--event-id", default="GW150914")
    ap.add_argument("--detectors", default="H1,L1")
    ap.add_argument("--duration-s", type=float, default=32.0)
    ap.add_argument("--synthetic", action="store_true", help="Generate synthetic data (offline mode)")
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.run, out_root)

    detectors = [d.strip().upper() for d in args.detectors.split(",") if d.strip()]
    if not detectors:
        _abort("--detectors is empty")
    if args.duration_s <= 0:
        _abort("--duration-s must be > 0")

    stage_dir, outputs_dir = ensure_stage_dirs(args.run, STAGE_NAME, base_dir=out_root)
    params: dict[str, Any] = {
        "event_id": args.event_id, "detectors": detectors,
        "duration_s": args.duration_s, "synthetic": args.synthetic,
    }

    try:
        if args.synthetic:
            gps_center = 1126259462.4204  # GW150914 default
            gps_start = gps_center - args.duration_s / 2.0
            library_version = "synthetic_v1"
        else:
            gps_center = _fetch_gps_center(args.event_id)
            gps_start = gps_center - args.duration_s / 2.0
            library_version = "unknown"

        strains: dict[str, np.ndarray] = {}
        sha_by_det: dict[str, str] = {}
        sample_rate_hz: float | None = None

        for det in detectors:
            if args.synthetic:
                strain, sr, library_version = _generate_synthetic_strain(
                    det, gps_start, args.duration_s,
                )
            else:
                strain, sr, library_version = _fetch_via_gwpy(det, gps_start, args.duration_s)

            strains[det] = strain
            sha_by_det[det] = _sha256_array(strain)
            if sample_rate_hz is None:
                sample_rate_hz = sr
            elif abs(sample_rate_hz - sr) > 1e-6:
                _abort(f"sample_rate mismatch: {sample_rate_hz} vs {sr}")

        if sample_rate_hz is None:
            _abort("no strain downloaded")

        # Write strain.npz
        npz_payload: dict[str, Any] = {
            "sample_rate_hz": np.float64(sample_rate_hz),
            "gps_start": np.float64(gps_start),
            "duration_s": np.float64(args.duration_s),
        }
        for det in detectors:
            npz_payload[det] = strains[det]
        npz_path = outputs_dir / "strain.npz"
        np.savez(npz_path, **npz_payload)

        # Write provenance.json
        provenance = {
            "event_id": args.event_id,
            "source": "synthetic" if args.synthetic else "GWOSC",
            "detectors": detectors,
            "gps_center": gps_center,
            "gps_start": gps_start,
            "duration_s": args.duration_s,
            "sample_rate_hz": sample_rate_hz,
            "library_version": library_version,
            "sha256_per_detector": sha_by_det,
            "timestamp": utc_now_iso(),
        }
        prov_path = outputs_dir / "provenance.json"
        write_json_atomic(prov_path, provenance)

        # Stage summary + manifest
        run_dir = stage_dir.parent
        outputs_list = [
            {"path": str(npz_path.relative_to(run_dir)), "sha256": sha256_file(npz_path)},
            {"path": str(prov_path.relative_to(run_dir)), "sha256": sha256_file(prov_path)},
        ]
        summary = {
            "stage": STAGE_NAME, "run": args.run, "created": utc_now_iso(),
            "version": "v1", "parameters": params,
            "inputs": [{"kind": "synthetic" if args.synthetic else "gwosc_api", "event_id": args.event_id}],
            "outputs": outputs_list, "verdict": "PASS",
        }
        sp = write_stage_summary(stage_dir, summary)
        write_manifest(stage_dir, {"strain_npz": npz_path, "provenance": prov_path, "stage_summary": sp})

        print(f"OK: {STAGE_NAME} PASS ({args.event_id}, {detectors})")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        _write_failure(stage_dir, args.run, params, str(exc))
        _abort(str(exc))
        return EXIT_CONTRACT_FAIL  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
