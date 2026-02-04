#!/usr/bin/env python3
"""
BASURIN validation stage: Fetch GW150914 strain data from GWOSC.

Descarga directa con requests (timeout real) en lugar de gwpy.fetch_open_data
que puede colgarse indefinidamente.

Uso:
    RUN=2025-01-30__ringdown_validation python gw150914_fetch_fixed.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Config ---
EVENT = "GW150914"
T0_GPS = 1126259462.4
SAMPLE_RATE_TARGET = 16384  # Hz (16 kHz version)
DURATION = 32  # seconds (the 32s event files)
DETECTORS = ["H1", "L1"]
TIMEOUT_SECONDS = 60  # per-file download timeout


def abort(msg: str, code: int = 2) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_with_timeout(url: str, dest: Path, timeout: int = TIMEOUT_SECONDS) -> None:
    """Download URL to dest with real timeout via requests."""
    import requests

    print(f"    Downloading: {url}")
    print(f"    -> {dest}")

    try:
        # connect timeout, read timeout
        resp = requests.get(url, timeout=(10, timeout), stream=True)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Download timed out after {timeout}s: {url}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed: {e}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"    {downloaded / (1 << 20):.1f} MB ({pct:.0f}%)", end="\r")

    print()  # newline after progress
    os.replace(tmp, dest)


def get_event_url(detector: str, sample_rate: int = 4096, duration: int = 32) -> str:
    """Get direct URL for event file from GWOSC.
    
    Uses GWTC-1-confident catalog for the official 32s event files.
    
    Args:
        detector: H1 or L1
        sample_rate: Sample rate in Hz (4096 or 16384)
        duration: Duration in seconds (32 for event files)
    """
    try:
        from gwosc.locate import get_event_urls
    except ImportError:
        abort("gwosc package not installed. pip install gwosc")

    # GWTC-1-confident has the official 32s event files
    urls = get_event_urls(
        EVENT,
        catalog="GWTC-1-confident",
        detector=detector,
        duration=duration,
        sample_rate=sample_rate,
        format="hdf5",
    )

    if not urls:
        # Fallback to bulk files from O1
        urls = get_event_urls(
            EVENT,
            detector=detector,
            duration=4096,
            sample_rate=sample_rate,
            format="hdf5",
        )

    if not urls:
        abort(f"No URLs found for {EVENT} {detector} {sample_rate}Hz {duration}s")

    return urls[0]


def convert_gwosc_to_gwpy_format(src_path: Path, dest_path: Path, detector: str) -> dict:
    """
    Read GWOSC HDF5 and write in gwpy-compatible format.
    Returns metadata about the conversion.
    """
    import h5py
    import numpy as np

    with h5py.File(src_path, "r") as f_in:
        # GWOSC format: /strain/Strain dataset
        strain_data = f_in["strain/Strain"][:]
        gps_start = f_in["meta/GPSstart"][()]
        duration = f_in["meta/Duration"][()]
        # Sample rate from attributes or compute
        if "Xspacing" in f_in["strain/Strain"].attrs:
            dt = f_in["strain/Strain"].attrs["Xspacing"]
            sample_rate = int(1.0 / dt)
        else:
            sample_rate = len(strain_data) // duration

    # Write in gwpy TimeSeries format
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest_path, "w") as f_out:
        # gwpy expects a specific structure
        dset = f_out.create_dataset("strain", data=strain_data, dtype=np.float64)
        dset.attrs["x0"] = float(gps_start)
        dset.attrs["dx"] = 1.0 / float(sample_rate)
        dset.attrs["xunit"] = "s"
        dset.attrs["channel"] = f"{detector}:GWOSC-STRAIN"
        dset.attrs["name"] = f"{detector}:GWOSC-STRAIN"

    return {
        "gps_start": float(gps_start),
        "duration": int(duration),
        "sample_rate": int(sample_rate),
        "n_samples": len(strain_data),
    }


def main() -> int:
    # --- Resolve RUN ---
    run_id = os.environ.get("RUN")
    if not run_id:
        abort("env RUN not set")

    runs_root = Path("runs")
    stage_dir = runs_root / run_id / "validation" / "ringdown" / "01_data_fetch_gw150914"
    out_dir = stage_dir / "outputs"

    # --- Check RUN_VALID ---
    run_valid_path = runs_root / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    if not run_valid_path.exists():
        # Try legacy path
        run_valid_path = runs_root / run_id / "RUN_VALID" / "verdict.json"
    if not run_valid_path.exists():
        abort(f"missing RUN_VALID at {run_valid_path}")

    rv = json.loads(run_valid_path.read_text())
    verdict = rv.get("overall_verdict") or rv.get("verdict") or rv.get("status")
    if verdict != "PASS":
        abort(f"RUN_VALID != PASS ({verdict})")

    # --- Import dependencies (late, after basic checks) ---
    try:
        import requests
    except ImportError:
        abort("requests package not installed. pip install requests")

    try:
        import h5py
    except ImportError:
        abort("h5py package not installed. pip install h5py")

    try:
        import gwosc
    except ImportError:
        abort("gwosc package not installed. pip install gwosc")

    # --- Fetch data ---
    out_dir.mkdir(parents=True, exist_ok=True)
    created = utc_now_iso()

    file_meta = {}
    raw_urls = {}

    for i, det in enumerate(DETECTORS, start=1):
        print(f"[{i}/{len(DETECTORS)*2}] Getting URL for {det}...")

        try:
            url = get_event_url(det, sample_rate=4096, duration=DURATION)
        except Exception as e:
            abort(f"Failed to get URL for {det}: {e}")

        raw_urls[det] = url

        # Download to temp, then convert
        raw_path = out_dir / f"{det}_raw.hdf5"
        final_path = out_dir / f"{det}_strain.h5"

        print(f"[{i}/{len(DETECTORS)*2}] Downloading {det}...")
        try:
            download_with_timeout(url, raw_path, timeout=TIMEOUT_SECONDS)
        except TimeoutError as e:
            abort(str(e))
        except RuntimeError as e:
            abort(str(e))

        print(f"[{i + len(DETECTORS)}/{len(DETECTORS)*2}] Converting {det} to gwpy format...")
        try:
            meta = convert_gwosc_to_gwpy_format(raw_path, final_path, det)
            file_meta[det] = meta
            # Remove raw file
            raw_path.unlink()
        except Exception as e:
            abort(f"Conversion failed for {det}: {e}")

    # --- Write metadata ---
    print(f"[{len(DETECTORS)*2 + 1}/{len(DETECTORS)*2 + 2}] Writing metadata...")

    segments = {
        "schema_version": "segments_request_v1",
        "event": EVENT,
        "t0_gps": T0_GPS,
        "window_policy": f"event_file_{DURATION}s",
        "detectors": DETECTORS,
        "file_meta": file_meta,
    }
    write_json(out_dir / "segments.json", segments)

    provenance = {
        "schema_version": "data_provenance_v1",
        "source": "GWOSC",
        "method": "direct_download_via_gwosc.locate.get_event_urls",
        "event": EVENT,
        "t0_gps": T0_GPS,
        "duration": DURATION,
        "urls": raw_urls,
        "versions": {
            "gwosc": gwosc.__version__,
            "h5py": h5py.__version__,
        },
        "file_meta": file_meta,
        "created": created,
    }
    write_json(out_dir / "data_provenance.json", provenance)

    # --- Manifest ---
    manifest_outputs = {}
    for det in DETECTORS:
        det_path = out_dir / f"{det}_strain.h5"
        manifest_outputs[f"outputs/{det}_strain.h5"] = sha256_file(det_path)

    manifest_outputs["outputs/segments.json"] = sha256_file(out_dir / "segments.json")
    manifest_outputs["outputs/data_provenance.json"] = sha256_file(out_dir / "data_provenance.json")

    manifest = {
        "schema_version": "manifest_v1",
        "stage": "01_data_fetch_gw150914",
        "run": run_id,
        "created": created,
        "outputs": manifest_outputs,
    }
    write_json(stage_dir / "manifest.json", manifest)

    # --- Stage summary ---
    summary = {
        "schema_version": "stage_summary_v1",
        "stage": "01_data_fetch_gw150914",
        "run_id": run_id,
        "created": created,
        "intent": "internal_validation_only",
        "publication_intent": "none",
        "params": {
            "event": EVENT,
            "t0_gps": T0_GPS,
            "duration": DURATION,
            "detectors": DETECTORS,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
        "inputs": {
            "RUN_VALID": sha256_file(run_valid_path),
        },
        "provenance": provenance,
        "verdict": {"overall": "PASS"},
    }
    write_json(stage_dir / "stage_summary.json", summary)

    print(f"[{len(DETECTORS)*2 + 2}/{len(DETECTORS)*2 + 2}] Done!")
    print(f"[OK] {stage_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
