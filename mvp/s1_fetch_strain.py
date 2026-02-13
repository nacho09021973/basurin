#!/usr/bin/env python3
"""MVP Stage 1: Fetch real GWOSC strain for H1 and L1 per event.

CLI:
    python mvp/s1_fetch_strain.py --run <run_id> --event-id GW150914 \
        [--detectors H1,L1] [--duration-s 32] [--synthetic]

Outputs (runs/<run>/s1_fetch_strain/outputs/):
    strain.npz          H1, L1 arrays + sample_rate_hz + gps_start + duration_s
    provenance.json     Full download provenance with SHA256 hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.contracts import init_stage, finalize, abort
from basurin_io import write_json_atomic, utc_now_iso

STAGE = "s1_fetch_strain"


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def _fetch_via_gwpy(
    detector: str,
    gps_start: float,
    duration_s: float,
    timeout_s: int = 60,
) -> tuple[np.ndarray, float, str]:
    try:
        from gwpy.timeseries import TimeSeries
        from gwpy import __version__ as gwpy_ver
    except ImportError as exc:
        raise RuntimeError(
            "gwpy not installed. Install with: pip install gwpy. "
            "Or use --synthetic for offline testing."
        ) from exc
    def _handle_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(
            f"GWOSC fetch timeout for detector {detector} after {timeout_s}s"
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(int(timeout_s))
    try:
        print(
            f"[s1_fetch_strain] GWOSC fetch begin: det={detector}, "
            f"gps=[{gps_start}, {gps_start + duration_s}]",
            flush=True,
        )
        ts = TimeSeries.fetch_open_data(detector, gps_start, gps_start + duration_s, verbose=False)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    values = np.asarray(ts.value, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Invalid strain shape for {detector}: {values.shape}")
    print(
        f"[s1_fetch_strain] GWOSC fetch OK: det={detector}, n={values.size}, fs={float(ts.sample_rate.value)}",
        flush=True,
    )
    return values, float(ts.sample_rate.value), gwpy_ver


def _generate_synthetic_strain(
    detector: str, gps_start: float, duration_s: float,
    fs: float = 4096.0, f0: float = 251.0, tau: float = 0.004,
) -> tuple[np.ndarray, float, str]:
    seed = 42 + hash(detector)
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    t_ring = duration_s / 2.0
    dt = t - t_ring
    signal = np.zeros(n)
    mask = dt >= 0
    signal[mask] = np.exp(-dt[mask] / tau) * np.cos(2 * np.pi * f0 * dt[mask])
    noise = rng.normal(0, 1e-21, n)
    return (signal * 1e-21 + noise).astype(np.float64), fs, "synthetic_v1"


def _fetch_gps_center(event_id: str) -> float:
    local = Path("docs/ringdown/event_metadata") / f"{event_id}_metadata.json"
    if local.exists():
        print(f"[s1_fetch_strain] GPS lookup: reading local metadata {local}", flush=True)
        with open(local, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("t_coalescence_gps", "gps", "GPS", "gpstime"):
            if key in meta:
                print(f"[s1_fetch_strain] GPS resolved from local file: {meta[key]}", flush=True)
                return float(meta[key])
    print(f"[s1_fetch_strain] GPS lookup: querying GWOSC API for {event_id} ...", flush=True)
    try:
        import requests
        url = f"https://gwosc.org/api/v2/events/{event_id}"
        print(f"[s1_fetch_strain] GET {url} (timeout=30s)", flush=True)
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        resp.raise_for_status()
        print(f"[s1_fetch_strain] GWOSC API responded: HTTP {resp.status_code}", flush=True)
        data = resp.json()
        versions = data.get("event_versions") or data.get("versions") or []
        if versions and isinstance(versions, list):
            v = versions[-1]
            detail_url = v.get("detail_url") or v.get("url")
            if detail_url:
                if detail_url.startswith("/"):
                    detail_url = f"https://gwosc.org{detail_url}"
                print(f"[s1_fetch_strain] GET {detail_url} (timeout=30s)", flush=True)
                vresp = requests.get(detail_url, headers={"Accept": "application/json"}, timeout=30)
                vresp.raise_for_status()
                vdata = vresp.json()
                for key in ("GPS", "gps", "gpstime", "gps_time"):
                    if key in vdata:
                        print(f"[s1_fetch_strain] GPS resolved from GWOSC: {vdata[key]}", flush=True)
                        return float(vdata[key])
        for key in ("GPS", "gps", "gpstime"):
            if key in data:
                print(f"[s1_fetch_strain] GPS resolved from GWOSC: {data[key]}", flush=True)
                return float(data[key])
    except Exception as exc:
        print(f"[s1_fetch_strain] GWOSC API error: {exc}", flush=True)
    raise RuntimeError(
        f"Cannot resolve GPS center for {event_id}. "
        f"Add metadata to docs/ringdown/event_metadata/{event_id}_metadata.json"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: fetch GWOSC strain")
    ap.add_argument("--run", required=True)
    ap.add_argument("--event-id", default="GW150914")
    ap.add_argument("--detectors", default="H1,L1")
    ap.add_argument("--duration-s", type=float, default=32.0)
    ap.add_argument(
        "--fetch-timeout-s",
        type=int,
        default=60,
        help="Hard timeout for GWOSC strain fetch via gwpy (seconds)",
    )
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    detectors = [d.strip().upper() for d in args.detectors.split(",") if d.strip()]
    if not detectors:
        print("ERROR: --detectors is empty", file=sys.stderr)
        raise SystemExit(2)

    ctx = init_stage(args.run, STAGE, params={
        "event_id": args.event_id, "detectors": detectors,
        "duration_s": args.duration_s, "synthetic": args.synthetic,
    })

    try:
        if args.synthetic:
            gps_center = 1126259462.4204
        else:
            gps_center = _fetch_gps_center(args.event_id)
        gps_start = gps_center - args.duration_s / 2.0

        strains: dict[str, np.ndarray] = {}
        sha_by_det: dict[str, str] = {}
        sample_rate_hz: float | None = None
        library_version = "unknown"

        for i, det in enumerate(detectors, 1):
            print(
                f"[s1_fetch_strain] Detector {i}/{len(detectors)}: {det}",
                flush=True,
            )
            if args.synthetic:
                print(f"[s1_fetch_strain] Generating synthetic strain for {det} ...", flush=True)
                strain, sr, library_version = _generate_synthetic_strain(det, gps_start, args.duration_s)
                print(f"[s1_fetch_strain] Synthetic OK: {det}, n={strain.size}, fs={sr}", flush=True)
            else:
                strain, sr, library_version = _fetch_via_gwpy(
                    det,
                    gps_start,
                    args.duration_s,
                    timeout_s=args.fetch_timeout_s,
                )
            strains[det] = strain
            sha_by_det[det] = _sha256_array(strain)
            if sample_rate_hz is None:
                sample_rate_hz = sr
            elif abs(sample_rate_hz - sr) > 1e-6:
                abort(ctx, f"sample_rate mismatch: {sample_rate_hz} vs {sr}")

        if sample_rate_hz is None:
            abort(ctx, "no strain downloaded")

        npz_payload: dict[str, Any] = {
            "sample_rate_hz": np.float64(sample_rate_hz),
            "gps_start": np.float64(gps_start),
            "duration_s": np.float64(args.duration_s),
        }
        for det in detectors:
            npz_payload[det] = strains[det]
        npz_path = ctx.outputs_dir / "strain.npz"
        np.savez(npz_path, **npz_payload)

        provenance = {
            "event_id": args.event_id,
            "source": "synthetic" if args.synthetic else "GWOSC",
            "detectors": detectors, "gps_center": gps_center,
            "gps_start": gps_start, "duration_s": args.duration_s,
            "sample_rate_hz": sample_rate_hz, "library_version": library_version,
            "sha256_per_detector": sha_by_det, "timestamp": utc_now_iso(),
        }
        prov_path = ctx.outputs_dir / "provenance.json"
        write_json_atomic(prov_path, provenance)

        finalize(ctx, artifacts={"strain_npz": npz_path, "provenance": prov_path},
                 results={"detectors": detectors, "sample_rate_hz": sample_rate_hz})
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
