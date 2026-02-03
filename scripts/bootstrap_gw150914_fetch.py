#!/usr/bin/env python3
"""
Bootstrap: fetch real GW150914 strain data from GWOSC.

Usage:
    export RUN="2026-02-03__REAL_GW150914_FETCH"
    python scripts/bootstrap_gw150914_fetch.py

Requires:
    - RUN environment variable set
    - BASURIN_RUNS_ROOT (optional, defaults to "runs")
    - RUN_VALID stage must have passed for this run
    - Dependencies: gwpy, gwosc, h5py

Outputs (under $RUNS_ROOT/$RUN/validation/ringdown/01_data_fetch_gw150914/):
    - manifest.json
    - stage_summary.json
    - outputs/H1_strain.h5
    - outputs/L1_strain.h5
    - outputs/segments.json
    - outputs/data_provenance.json
"""
import hashlib
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration: frozen GW150914 window t0 +/- 16s
# ---------------------------------------------------------------------------
EVENT = "GW150914"
T0 = 1126259462.4
GPS_START = T0 - 16.0
GPS_END = T0 + 16.0
SAMPLE_RATE = 16384


def main() -> int:
    run_id = os.environ.get("RUN")
    if not run_id:
        print("ABORT: env RUN not set", file=sys.stderr)
        return 2

    runs_root = Path(os.environ.get("BASURIN_RUNS_ROOT", "runs"))
    stage_dir = runs_root / run_id / "validation" / "ringdown" / "01_data_fetch_gw150914"
    out_dir = stage_dir / "outputs"

    # -----------------------------------------------------------------------
    # Gate: RUN_VALID must exist and be PASS
    # -----------------------------------------------------------------------
    run_valid_path = runs_root / run_id / "RUN_VALID" / "outputs" / "run_valid.json"
    if not run_valid_path.exists():
        print(f"ABORT: missing {run_valid_path}", file=sys.stderr)
        print("       Run: python experiment/run_valid/stage_run_valid.py --run $RUN", file=sys.stderr)
        return 2

    run_valid = json.loads(run_valid_path.read_text())
    if run_valid.get("overall_verdict") != "PASS":
        print(f"ABORT: RUN_VALID != PASS ({run_valid.get('overall_verdict')})", file=sys.stderr)
        return 2

    # -----------------------------------------------------------------------
    # Fetch strain from GWOSC via GWPy
    # -----------------------------------------------------------------------
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        print("       Install: python -m pip install gwpy gwosc h5py", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Fetching H1 strain from GWOSC (GPS {GPS_START:.1f} to {GPS_END:.1f})...")
    h1 = TimeSeries.fetch_open_data("H1", GPS_START, GPS_END, sample_rate=SAMPLE_RATE)

    print(f"[2/4] Fetching L1 strain from GWOSC...")
    l1 = TimeSeries.fetch_open_data("L1", GPS_START, GPS_END, sample_rate=SAMPLE_RATE)

    h1_path = out_dir / "H1_strain.h5"
    l1_path = out_dir / "L1_strain.h5"

    print(f"[3/4] Writing {h1_path}...")
    h1.write(str(h1_path), format="hdf5", overwrite=True)

    print(f"[4/4] Writing {l1_path}...")
    l1.write(str(l1_path), format="hdf5", overwrite=True)

    # -----------------------------------------------------------------------
    # Write metadata files
    # -----------------------------------------------------------------------
    _write_json(out_dir / "segments.json", {
        "schema_version": "segments_request_v1",
        "event": EVENT,
        "t0_gps": T0,
        "window_policy": "t0-16s_to_t0+16s",
        "requested": {"gps_start": GPS_START, "gps_end": GPS_END},
        "detectors": ["H1", "L1"],
    })

    provenance = {
        "schema_version": "data_provenance_v1",
        "source": "GWOSC",
        "method": "gwpy.timeseries.TimeSeries.fetch_open_data",
        "event": EVENT,
        "t0_gps": T0,
        "gps_start": GPS_START,
        "gps_end": GPS_END,
        "sample_rate": SAMPLE_RATE,
    }
    _write_json(out_dir / "data_provenance.json", provenance)

    manifest = {
        "schema_version": "manifest_v1",
        "outputs": {
            "outputs/H1_strain.h5": _sha256_file(h1_path),
            "outputs/L1_strain.h5": _sha256_file(l1_path),
            "outputs/segments.json": _sha256_file(out_dir / "segments.json"),
            "outputs/data_provenance.json": _sha256_file(out_dir / "data_provenance.json"),
        },
    }
    _write_json(stage_dir / "manifest.json", manifest)

    stage_summary = {
        "schema_version": "stage_summary_v1",
        "stage": "01_data_fetch_gw150914",
        "run_id": run_id,
        "intent": "internal_validation_only",
        "publication_intent": "none",
        "params": provenance,
        "inputs": {"RUN_VALID": _sha256_file(run_valid_path)},
        "verdict": {"overall": "PASS"},
    }
    _write_json(stage_dir / "stage_summary.json", stage_summary)

    print(f"[OK] {stage_dir}")
    return 0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
