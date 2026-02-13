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
import shutil
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
from basurin_io import write_json_atomic, sha256_file, utc_now_iso

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
    seed = int.from_bytes(hashlib.sha256(detector.encode("utf-8")).digest()[:8], "little")
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


def _parse_local_hdf5_args(items: list[str]) -> dict[str, Path]:
    local_by_det: dict[str, Path] = {}
    for item in items:
        det_raw, sep, path_raw = item.partition("=")
        det = det_raw.strip().upper()
        path = Path(path_raw.strip())
        if not sep:
            raise ValueError(f"Invalid --local-hdf5 entry '{item}'. Expected DET=PATH")
        if det not in {"H1", "L1"}:
            raise ValueError(f"Invalid detector in --local-hdf5: {det}. Allowed: H1,L1")
        if det in local_by_det:
            raise ValueError(f"Duplicate --local-hdf5 detector: {det}")
        if not path.exists() or not path.is_file():
            raise ValueError(f"--local-hdf5 path for {det} not found or not a file: {path}")
        local_by_det[det] = path
    return local_by_det


def _find_strain_dataset(h5: Any) -> Any:
    if "strain/Strain" in h5:
        return h5["strain/Strain"]
    if "strain" in h5 and hasattr(h5["strain"], "shape"):
        return h5["strain"]

    for key in h5:
        obj = h5[key]
        if hasattr(obj, "keys") and "strain/Strain" in obj:
            return obj["strain/Strain"]
    raise ValueError("missing strain dataset: expected 'strain/Strain' (possibly nested) or 'strain'")


def _load_local_hdf5(path: Path) -> tuple[np.ndarray, float, float | None, str]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py not installed; required for --local-hdf5 mode") from exc

    with h5py.File(path, "r") as h5:
        try:
            ds = _find_strain_dataset(h5)
        except ValueError as exc:
            raise ValueError(f"Unsupported HDF5 structure in {path}: {exc}") from exc

        values = np.asarray(ds[...], dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"Invalid strain shape in {path}: {values.shape}")

        xspacing = ds.attrs.get("Xspacing")
        if xspacing is None:
            raise ValueError(f"Missing Xspacing attr in {path} strain dataset")
        xstart = ds.attrs.get("Xstart")
        gps_start = float(xstart) if xstart is not None else None
    return values, float(1.0 / float(xspacing)), gps_start, "h5py"


def _try_reuse(
    ctx,
    event_id: str,
    detectors: list[str],
    duration_s: float,
    local_input_sha: dict[str, str] | None = None,
) -> bool:
    """Return True if existing outputs match params and pass hash validation.

    Checks:
      1. strain.npz and provenance.json exist in ctx.outputs_dir
      2. provenance.json event_id, duration_s, detectors match current params
      3. strain.npz contains all requested detector keys
      4. Per-detector SHA256 in provenance matches re-computed hashes from npz

    On success, calls finalize() and returns True.
    On any mismatch or missing file, prints reason and returns False.
    """
    npz_path = ctx.outputs_dir / "strain.npz"
    prov_path = ctx.outputs_dir / "provenance.json"

    if not npz_path.exists() or not prov_path.exists():
        print("[s1_fetch_strain] reuse: outputs not found, will fetch", flush=True)
        return False

    try:
        with open(prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[s1_fetch_strain] reuse: cannot read provenance ({exc}), will fetch", flush=True)
        return False

    # Validate params match
    if prov.get("event_id") != event_id:
        print(f"[s1_fetch_strain] reuse: event_id mismatch ({prov.get('event_id')} != {event_id})", flush=True)
        return False
    if abs(prov.get("duration_s", -1) - duration_s) > 1e-9:
        print(f"[s1_fetch_strain] reuse: duration_s mismatch ({prov.get('duration_s')} != {duration_s})", flush=True)
        return False
    prov_dets = sorted(prov.get("detectors", []))
    if prov_dets != sorted(detectors):
        print(f"[s1_fetch_strain] reuse: detectors mismatch ({prov_dets} != {sorted(detectors)})", flush=True)
        return False
    if local_input_sha is not None:
        if prov.get("source") != "local_hdf5":
            print("[s1_fetch_strain] reuse: source mismatch (expected local_hdf5)", flush=True)
            return False
        if prov.get("local_input_sha256", {}) != local_input_sha:
            print("[s1_fetch_strain] reuse: local input SHA mismatch, will fetch", flush=True)
            return False

    # Validate npz contents and hashes
    try:
        npz = np.load(npz_path)
    except Exception as exc:
        print(f"[s1_fetch_strain] reuse: cannot load strain.npz ({exc}), will fetch", flush=True)
        return False

    sha_recorded = prov.get("sha256_per_detector", {})
    for det in detectors:
        if det not in npz.files:
            print(f"[s1_fetch_strain] reuse: detector {det} missing in strain.npz, will fetch", flush=True)
            return False
        if det in sha_recorded:
            actual_sha = _sha256_array(npz[det])
            if actual_sha != sha_recorded[det]:
                print(f"[s1_fetch_strain] reuse: SHA256 mismatch for {det}, will fetch", flush=True)
                return False

    sample_rate_hz = float(npz["sample_rate_hz"]) if "sample_rate_hz" in npz.files else None
    if sample_rate_hz is None:
        print("[s1_fetch_strain] reuse: sample_rate_hz missing in strain.npz, will fetch", flush=True)
        return False

    # All checks passed — finalize with existing artifacts
    print("[s1_fetch_strain] reuse: outputs valid, skipping fetch", flush=True)
    finalize(
        ctx,
        artifacts={"strain_npz": npz_path, "provenance": prov_path},
        results={"detectors": detectors, "sample_rate_hz": sample_rate_hz},
        extra_summary={"reused": True},
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: fetch GWOSC strain")
    ap.add_argument("--run", required=True)
    ap.add_argument("--event-id", default="GW150914")
    ap.add_argument("--detectors", default=None)
    ap.add_argument("--duration-s", type=float, default=32.0)
    ap.add_argument(
        "--fetch-timeout-s",
        type=int,
        default=60,
        help="Hard timeout for GWOSC strain fetch via gwpy (seconds)",
    )
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument(
        "--local-hdf5",
        action="append",
        default=[],
        metavar="DET=PATH",
        help="Use local/offline HDF5 strain per detector (repeatable, e.g. H1=/tmp/H.hdf5)",
    )
    ap.add_argument(
        "--reuse-if-present",
        action="store_true",
        default=False,
        help="Skip fetch if outputs/strain.npz + provenance.json already exist "
             "and event_id/duration_s/detectors match. Validates array hashes.",
    )
    args = ap.parse_args()

    detectors = [d.strip().upper() for d in (args.detectors or "").split(",") if d.strip()]

    local_by_det = _parse_local_hdf5_args(args.local_hdf5)
    if local_by_det and not detectors:
        detectors = sorted(local_by_det.keys())
    if not detectors:
        detectors = ["H1", "L1"]
    if not detectors:
        print("ERROR: --detectors is empty", file=sys.stderr)
        raise SystemExit(2)

    if local_by_det and args.synthetic:
        print("ERROR: --local-hdf5 cannot be used with --synthetic", file=sys.stderr)
        raise SystemExit(2)
    if local_by_det and sorted(local_by_det.keys()) != sorted(detectors):
        print(
            "ERROR: --detectors must match exactly detectors provided by --local-hdf5",
            file=sys.stderr,
        )
        raise SystemExit(2)

    local_input_sha = {det: sha256_file(path) for det, path in sorted(local_by_det.items())} if local_by_det else None

    ctx = init_stage(args.run, STAGE, params={
        "event_id": args.event_id, "detectors": detectors,
        "duration_s": args.duration_s, "synthetic": args.synthetic,
        "local_hdf5": {det: str(path) for det, path in sorted(local_by_det.items())},
    })

    # --- Reuse check (before any network / generation) ---
    if args.reuse_if_present:
        try:
            if _try_reuse(ctx, args.event_id, detectors, args.duration_s, local_input_sha=local_input_sha):
                return 0
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[s1_fetch_strain] reuse check failed ({exc}), proceeding with fetch", flush=True)

    try:
        if local_by_det:
            gps_center = None
            gps_start = None
        elif args.synthetic:
            gps_center = 1126259462.4204
        else:
            gps_center = _fetch_gps_center(args.event_id)
        if gps_center is not None:
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
            elif local_by_det:
                inputs_dir = ctx.stage_dir / "inputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                src = local_by_det[det]
                dst = inputs_dir / src.name
                shutil.copy2(src, dst)
                strain, sr, gps_start_local, library_version = _load_local_hdf5(dst)
                if gps_start is None and gps_start_local is not None:
                    gps_start = gps_start_local
                    gps_center = gps_start + args.duration_s / 2.0
                print(f"[s1_fetch_strain] Local HDF5 OK: {det}, n={strain.size}, fs={sr}", flush=True)
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
        if gps_start is None:
            gps_start = 0.0

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
            "source": "local_hdf5" if local_by_det else ("synthetic" if args.synthetic else "GWOSC"),
            "detectors": detectors, "gps_center": gps_center,
            "gps_start": gps_start, "duration_s": args.duration_s,
            "sample_rate_hz": sample_rate_hz, "library_version": library_version,
            "sha256_per_detector": sha_by_det, "timestamp": utc_now_iso(),
        }
        if local_by_det:
            provenance["local_inputs"] = {
                det: f"inputs/{local_by_det[det].name}" for det in detectors
            }
            provenance["local_input_sha256"] = {
                det: sha256_file(ctx.stage_dir / provenance["local_inputs"][det]) for det in detectors
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
