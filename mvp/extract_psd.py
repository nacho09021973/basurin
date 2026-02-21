#!/usr/bin/env python3
"""Helper: extract measured PSD from strain data via Welch's method.

This is an optional helper, not a formal pipeline stage. It reads strain
from s1 output and computes a measured PSD that can be passed to s6 via
--psd-path.

CLI:
    python mvp/extract_psd.py --run <run_id> [--detector H1] \
        [--nperseg-s 4.0] [--overlap 0.5]

Inputs:
    runs/<run>/s1_fetch_strain/outputs/strain.npz

Outputs:
    runs/<run>/psd/measured_psd.json  (+ manifest.json)

Note:
    This helper has no formal contract. It is documented as optional
    in the pipeline. The psd/ directory output can be passed to s6 via
    --psd-path runs/<run>/psd/measured_psd.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import resolve_out_root, sha256_file, utc_now_iso, write_json_atomic


def extract_psd(
    strain: np.ndarray,
    fs: float,
    nperseg_s: float = 4.0,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD from strain array.

    Args:
        strain: Time-domain strain array.
        fs: Sample rate in Hz.
        nperseg_s: Segment length in seconds (default 4s).
        overlap: Fractional overlap (default 0.5 = 50%).

    Returns:
        (frequencies_hz, psd_values) as numpy arrays.
    """
    from scipy.signal import welch

    nperseg = min(len(strain), int(nperseg_s * fs))
    nperseg = max(nperseg, 64)
    noverlap = int(nperseg * overlap)

    freqs, psd = welch(
        strain, fs=fs,
        nperseg=nperseg, noverlap=noverlap,
        window="hann", scaling="density",
    )
    return freqs, psd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract measured PSD from strain (optional helper, not a pipeline stage)"
    )
    ap.add_argument("--run", required=True, help="Run ID")
    ap.add_argument(
        "--detector", default=None,
        help="Detector name (H1/L1/V1). If not specified, uses combined strain.npz",
    )
    ap.add_argument(
        "--nperseg-s", type=float, default=4.0,
        help="Welch segment length in seconds (default: 4.0)",
    )
    ap.add_argument(
        "--overlap", type=float, default=0.5,
        help="Fractional overlap for Welch (default: 0.5)",
    )
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    run_dir = out_root / args.run

    # Load strain from s1 output
    strain_path = run_dir / "s1_fetch_strain" / "outputs" / "strain.npz"
    if not strain_path.exists():
        print(f"ERROR: strain not found at {strain_path}", file=sys.stderr)
        return 2

    data = np.load(strain_path)
    strain = np.asarray(data["strain"], dtype=np.float64)
    fs = float(np.asarray(data["sample_rate_hz"]).flat[0])

    # Load provenance to get detector name if not specified
    detector = args.detector
    if detector is None:
        prov_path = run_dir / "s1_fetch_strain" / "outputs" / "provenance.json"
        if prov_path.exists():
            with open(prov_path, "r", encoding="utf-8") as f:
                prov = json.load(f)
            detector = prov.get("detector") or prov.get("detectors", ["H1"])[0]
        else:
            detector = "H1"

    print(f"[extract_psd] run={args.run}, detector={detector}, fs={fs} Hz, "
          f"n={len(strain)}", flush=True)

    freqs, psd = extract_psd(strain, fs, args.nperseg_s, args.overlap)

    # Filter to positive frequencies only
    mask = freqs > 0
    freqs = freqs[mask]
    psd = psd[mask]

    nperseg = min(len(strain), int(args.nperseg_s * fs))

    result: dict[str, Any] = {
        "schema_version": "mvp_measured_psd_v1",
        "run_id": args.run,
        "detector": detector,
        "created_utc": utc_now_iso(),
        "method": "welch",
        "nperseg": nperseg,
        "fs": fs,
        "nperseg_s": args.nperseg_s,
        "overlap": args.overlap,
        "n_samples": len(strain),
        "n_freq_bins": len(freqs),
        "freq_resolution_hz": float(freqs[1] - freqs[0]) if len(freqs) > 1 else None,
        "frequencies_hz": [float(x) for x in freqs],
        "psd_values": [float(x) for x in psd],
    }

    # Write output
    psd_dir = run_dir / "psd"
    psd_dir.mkdir(parents=True, exist_ok=True)
    out_path = psd_dir / "measured_psd.json"
    write_json_atomic(out_path, result)

    # Write manifest
    manifest = {
        "schema_version": "mvp_manifest_v1",
        "stage": "extract_psd",
        "run_id": args.run,
        "created_utc": utc_now_iso(),
        "note": "Optional helper — not a formal pipeline stage",
        "artifacts": [
            {
                "path": str(out_path.relative_to(run_dir)),
                "sha256": sha256_file(out_path),
            }
        ],
        "inputs": [
            {"path": str(strain_path.relative_to(run_dir)), "label": "strain"},
        ],
    }
    write_json_atomic(psd_dir / "manifest.json", manifest)

    print(f"[extract_psd] Output: {out_path}", flush=True)
    print(f"[extract_psd] PSD spans {freqs[0]:.1f}–{freqs[-1]:.1f} Hz "
          f"({len(freqs)} bins)", flush=True)
    print(f"[extract_psd] Use with: --psd-path {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
