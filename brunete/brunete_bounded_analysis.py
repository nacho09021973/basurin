#!/usr/bin/env python3
"""BRUNETE — Bounded multimode analysis (220 vs 220+221).

Single-script, single-question analysis:
    For each event in a fixed cohort, does delta_BIC < -10 favour
    a two-mode (220+221) model over a single-mode (220) model?

Inputs:
    --losc-root DIR     Directory with HDF5 strain files: <losc-root>/<EVENT_ID>/*.hdf5
    --cohort FILE       Text file with one event ID per line (default: cohorts/events_support_multi.txt)
    --output FILE       Path for the results JSON (default: brunete_bounded_results.json)

Outputs:
    A single JSON file with per-event results and an aggregate verdict.

Methodology:
    1. Load strain from local HDF5 (H1 preferred, L1 fallback).
    2. Crop ringdown window: t0 + dt_start to t0 + dt_start + duration.
    3. Estimate mode 220 via spectral Lorentzian in lower sub-band.
    4. Subtract 220 template, estimate mode 221 in upper sub-band.
    5. Compute delta_BIC (BIC_2mode - BIC_1mode).
    6. Classify: INFORMATIONAL if delta_BIC < -10, SINGLEMODE_ONLY otherwise.
    7. Aggregate: if zero events cross threshold, dataset verdict = SINGLEMODE_ONLY.

Threshold validation:
    The delta_BIC = -10 threshold has been validated against synthetic injections
    (tests/test_delta_bic_threshold_validation.py): 0% false positive rate on
    1-mode signals and pure noise, 100% detection rate on 2-mode signals.

Reference:
    Kass & Raftery (1995), J. Amer. Statist. Assoc. 90(430):773-795.
    |delta_BIC| > 10 corresponds to "very strong" evidence on their scale.

Cohort provenance:
    The default cohort (events_support_multi.txt, 40 events) was selected by
    geometric viability: events whose ringdown data supports >1 compatible
    geometry in the intersection of mode-220 and mode-221 atlas regions after
    Hawking area filtering. This is a viability pre-filter, not a detection
    result. See docs/SYSTEM_READINESS_ASSESSMENT.md §3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from mvp.s3_ringdown_estimates import estimate_ringdown_spectral
from mvp.s3b_multimode_estimates import (
    _estimate_220_spectral,
    _estimate_221_spectral_two_pass,
    _split_mode_bands,
    _template_220,
    compute_model_comparison,
)
from mvp.s1_fetch_strain import match_hdf5_files, _pick_best_hdf5_candidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELTA_BIC_THRESHOLD = -10.0
BAND_LOW = 150.0
BAND_HIGH = 400.0
FS_DEFAULT = 4096.0
RINGDOWN_DURATION_S = 0.1      # 100 ms analysis window
DT_START_DEFAULT_S = 0.005     # 5 ms after coalescence (conservative)
DETECTORS_PRIORITY = ["H1", "L1", "V1"]
DEFAULT_COHORT = Path(__file__).resolve().parent / "cohorts" / "events_support_multi.txt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_strain_from_hdf5(hdf5_path: Path) -> tuple[np.ndarray, float, float]:
    """Load strain, sample_rate, and GPS start from a GWOSC HDF5 file.

    GWOSC HDF5 layout:
        /strain/Strain   — strain array
        /strain/Strain.attrs['Xspacing'] — 1/sample_rate
        /meta/GPSstart   — GPS start time
    """
    import h5py

    with h5py.File(str(hdf5_path), "r") as f:
        strain_ds = f["strain"]["Strain"]
        strain = np.asarray(strain_ds, dtype=np.float64)
        dt = float(strain_ds.attrs["Xspacing"])
        fs = 1.0 / dt
        gps_start = float(np.asarray(f["meta"]["GPSstart"]))

    return strain, fs, gps_start


def _resolve_hdf5_for_event(
    losc_root: Path, event_id: str,
) -> tuple[Path, str] | None:
    """Find best HDF5 file for an event, preferring H1 > L1 > V1."""
    event_dir = losc_root / event_id
    if not event_dir.is_dir():
        return None

    matches = match_hdf5_files(event_dir)
    for det in DETECTORS_PRIORITY:
        candidates = matches.get(det, [])
        if len(candidates) == 1:
            return candidates[0], det
        if len(candidates) > 1:
            best = _pick_best_hdf5_candidate(candidates)
            if best is not None:
                return best, det
    return None


def _load_t0_catalog() -> dict[str, float]:
    """Load GPS coalescence times from all available local sources.

    Sources (in priority order):
        1. gwtc_events_t0.json (canonical t0 reference catalog)
        2. gwtc_quality_events.csv (broad quality catalog with GPS column)
    """
    repo_root = Path(__file__).resolve().parents[1]
    result: dict[str, float] = {}

    # Source 1: quality CSV (lower priority, loaded first so JSON overrides)
    csv_path = repo_root / "gwtc_quality_events.csv"
    if csv_path.exists():
        import csv
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                event_id = (row.get("event") or "").strip()
                gps_raw = row.get("GPS", "").strip()
                if event_id and gps_raw:
                    try:
                        result[event_id] = float(gps_raw)
                    except ValueError:
                        pass

    # Source 1b: well-known GWTC-1 events not in local catalogs
    # GPS values from GWOSC (https://gwosc.org/eventapi/)
    _GWTC1_GPS: dict[str, float] = {
        "GW150914": 1126259462.4,
        "GW151226": 1135136350.6,
        "GW170104": 1167559936.6,
        "GW170608": 1180922494.5,
        "GW170729": 1185389807.3,
        "GW170809": 1186302519.8,
        "GW170814": 1186741861.5,
        "GW170817": 1187008882.4,
        "GW170818": 1187058327.1,
        "GW170823": 1187529256.5,
    }
    for eid, gps in _GWTC1_GPS.items():
        result.setdefault(eid, gps)

    # Source 2: t0 JSON (higher priority, overrides CSV)
    json_path = repo_root / "gwtc_events_t0.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for event_id, entry in raw.items():
            if isinstance(entry, dict):
                gps = entry.get("GPS") or entry.get("gps")
                if isinstance(gps, (int, float)):
                    result[event_id] = float(gps)
            elif isinstance(entry, (int, float)):
                result[event_id] = float(entry)

    return result


def _crop_ringdown(
    strain: np.ndarray,
    fs: float,
    gps_start: float,
    t0_gps: float,
    dt_start_s: float,
    duration_s: float,
) -> np.ndarray:
    """Crop strain to ringdown window: [t0 + dt_start, t0 + dt_start + duration]."""
    t_ring_start = t0_gps + dt_start_s
    i_start = int((t_ring_start - gps_start) * fs)
    i_end = i_start + int(duration_s * fs)

    if i_start < 0 or i_end > len(strain):
        raise ValueError(
            f"Ringdown window [{i_start}:{i_end}] outside strain "
            f"[0:{len(strain)}] (gps_start={gps_start}, t0={t0_gps})"
        )

    window = strain[i_start:i_end]
    if not np.all(np.isfinite(window)):
        n_bad = int(np.sum(~np.isfinite(window)))
        raise ValueError(f"{n_bad}/{len(window)} non-finite samples in ringdown window")

    return window


# ---------------------------------------------------------------------------
# Per-event analysis
# ---------------------------------------------------------------------------

def _analyze_event(
    event_id: str,
    losc_root: Path,
    t0_catalog: dict[str, float],
    band_low: float,
    band_high: float,
) -> dict[str, Any]:
    """Run bounded 220 vs 220+221 analysis on a single event."""

    # 1. Resolve HDF5
    resolved = _resolve_hdf5_for_event(losc_root, event_id)
    if resolved is None:
        return {"event_id": event_id, "status": "SKIP", "reason": "no_hdf5_found"}
    hdf5_path, detector = resolved

    # 2. Resolve t0
    t0_gps = t0_catalog.get(event_id)
    if t0_gps is None:
        # Try without timestamp suffix
        short_id = event_id.split("_")[0] if "_" in event_id else event_id
        t0_gps = t0_catalog.get(short_id)
    if t0_gps is None:
        return {"event_id": event_id, "status": "SKIP", "reason": "no_t0_gps"}

    # 3. Load strain and crop
    try:
        strain, fs, gps_start = _load_strain_from_hdf5(hdf5_path)
        ringdown = _crop_ringdown(
            strain, fs, gps_start,
            t0_gps=t0_gps,
            dt_start_s=DT_START_DEFAULT_S,
            duration_s=RINGDOWN_DURATION_S,
        )
    except Exception as exc:
        return {"event_id": event_id, "status": "FAIL", "reason": f"load_error: {exc}"}

    # 4. Estimate mode 220 (spectral, lower sub-band)
    try:
        est220 = _estimate_220_spectral(ringdown, fs, band_low=band_low, band_high=band_high)
        f220 = float(est220["f_hz"])
        Q220 = float(est220["Q"])
        tau220 = float(est220["tau_s"])
        ok_220 = math.isfinite(f220) and f220 > 0 and math.isfinite(Q220) and Q220 > 0
    except Exception as exc:
        return {"event_id": event_id, "status": "FAIL", "reason": f"est220_error: {exc}"}

    if not ok_220:
        return {
            "event_id": event_id, "status": "FAIL",
            "reason": "mode_220_not_finite",
            "mode_220": {"f_hz": f220, "Q": Q220},
        }

    # 5. Estimate mode 221 (two-pass spectral: subtract 220 template, fit residual)
    try:
        est221 = _estimate_221_spectral_two_pass(
            ringdown, fs, band_low=band_low, band_high=band_high,
        )
        f221 = float(est221["f_hz"])
        Q221 = float(est221["Q"])
        tau221 = float(est221["tau_s"])
        ok_221 = math.isfinite(f221) and f221 > 0 and math.isfinite(Q221) and Q221 > 0
    except Exception as exc:
        ok_221 = False
        f221 = float("nan")
        Q221 = float("nan")
        tau221 = float("nan")

    # 6. Compute delta_BIC
    mode_220_dict = {
        "ln_f": math.log(f220) if ok_220 else None,
        "ln_Q": math.log(Q220) if ok_220 else None,
        "label": "220",
    }
    mode_221_dict = {
        "ln_f": math.log(f221) if ok_221 else None,
        "ln_Q": math.log(Q221) if ok_221 else None,
        "label": "221",
    }

    mc = compute_model_comparison(
        signal=ringdown,
        fs=fs,
        mode_220=mode_220_dict,
        mode_221=mode_221_dict,
        ok_220=ok_220,
        ok_221=ok_221,
        delta_bic_threshold=DELTA_BIC_THRESHOLD,
    )

    delta_bic = mc.get("delta_bic")
    two_mode_preferred = mc.get("decision", {}).get("two_mode_preferred")

    # 7. Classify
    if delta_bic is not None and two_mode_preferred is True:
        classification = "INFORMATIONAL"
    elif delta_bic is not None:
        classification = "SINGLEMODE_ONLY"
    else:
        classification = "INCONCLUSIVE"

    return {
        "event_id": event_id,
        "status": "OK",
        "detector": detector,
        "hdf5_file": hdf5_path.name,
        "t0_gps": t0_gps,
        "n_samples": len(ringdown),
        "fs_hz": fs,
        "mode_220": {
            "f_hz": f220, "tau_s": tau220, "Q": Q220, "ok": ok_220,
        },
        "mode_221": {
            "f_hz": f221, "tau_s": tau221, "Q": Q221, "ok": ok_221,
        },
        "delta_bic": delta_bic,
        "two_mode_preferred": two_mode_preferred,
        "classification": classification,
        "bic_1mode": mc.get("bic_1mode"),
        "bic_2mode": mc.get("bic_2mode"),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate verdict from per-event results."""
    n_total = len(results)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")

    ok_results = [r for r in results if r["status"] == "OK"]
    n_informational = sum(1 for r in ok_results if r["classification"] == "INFORMATIONAL")
    n_singlemode = sum(1 for r in ok_results if r["classification"] == "SINGLEMODE_ONLY")
    n_inconclusive = sum(1 for r in ok_results if r["classification"] == "INCONCLUSIVE")

    delta_bics = [
        r["delta_bic"] for r in ok_results
        if r["delta_bic"] is not None
    ]

    if n_informational > 0:
        dataset_verdict = "MULTIMODE_EVIDENCE_PRESENT"
    else:
        dataset_verdict = "SINGLEMODE_ONLY"

    return {
        "dataset_verdict": dataset_verdict,
        "threshold_used": DELTA_BIC_THRESHOLD,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "n_informational": n_informational,
        "n_singlemode": n_singlemode,
        "n_inconclusive": n_inconclusive,
        "delta_bic_median": float(np.median(delta_bics)) if delta_bics else None,
        "delta_bic_min": float(np.min(delta_bics)) if delta_bics else None,
        "delta_bic_max": float(np.max(delta_bics)) if delta_bics else None,
        "informational_events": [
            r["event_id"] for r in ok_results if r["classification"] == "INFORMATIONAL"
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="BRUNETE bounded analysis: 220 vs 220+221 via delta_BIC"
    )
    ap.add_argument(
        "--losc-root", type=Path, required=True,
        help="Directory with HDF5 strain files: <losc-root>/<EVENT_ID>/*.hdf5",
    )
    ap.add_argument(
        "--cohort", type=Path, default=DEFAULT_COHORT,
        help="Text file with one event ID per line",
    )
    ap.add_argument(
        "--output", type=Path, default=Path("brunete_bounded_results.json"),
        help="Output JSON path",
    )
    ap.add_argument("--band-low", type=float, default=BAND_LOW)
    ap.add_argument("--band-high", type=float, default=BAND_HIGH)
    args = ap.parse_args()

    # Load cohort
    if not args.cohort.exists():
        print(f"ERROR: cohort file not found: {args.cohort}", file=sys.stderr)
        return 2
    event_ids = [
        line.strip() for line in args.cohort.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not event_ids:
        print("ERROR: empty cohort", file=sys.stderr)
        return 2

    # Load t0 catalog
    t0_catalog = _load_t0_catalog()

    print(f"[brunete_bounded] Cohort: {len(event_ids)} events from {args.cohort.name}")
    print(f"[brunete_bounded] LOSC root: {args.losc_root}")
    print(f"[brunete_bounded] Threshold: delta_BIC < {DELTA_BIC_THRESHOLD}")
    print(f"[brunete_bounded] Band: [{args.band_low}, {args.band_high}] Hz")
    print()

    # Run per-event analysis
    results: list[dict[str, Any]] = []
    for i, event_id in enumerate(event_ids, 1):
        print(f"  [{i:2d}/{len(event_ids)}] {event_id} ... ", end="", flush=True)
        result = _analyze_event(
            event_id, args.losc_root, t0_catalog,
            band_low=args.band_low, band_high=args.band_high,
        )
        status = result["status"]
        if status == "OK":
            delta_bic = result["delta_bic"]
            cls = result["classification"]
            dbic_str = f"ΔBIC={delta_bic:.1f}" if delta_bic is not None else "ΔBIC=N/A"
            print(f"{status} | {dbic_str} | {cls}")
        else:
            print(f"{status} | {result.get('reason', '?')}")
        results.append(result)

    # Aggregate
    aggregate = _aggregate(results)

    # Build output
    output = {
        "schema_version": "brunete_bounded_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_file": args.cohort.name,
        "cohort_size": len(event_ids),
        "losc_root": str(args.losc_root),
        "parameters": {
            "band_hz": [args.band_low, args.band_high],
            "delta_bic_threshold": DELTA_BIC_THRESHOLD,
            "dt_start_s": DT_START_DEFAULT_S,
            "ringdown_duration_s": RINGDOWN_DURATION_S,
            "detectors_priority": DETECTORS_PRIORITY,
        },
        "aggregate": aggregate,
        "per_event": results,
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=True)

    # Print summary
    print()
    print("=" * 60)
    print(f"DATASET VERDICT: {aggregate['dataset_verdict']}")
    print(f"  Processed: {aggregate['n_ok']}/{aggregate['n_total']}")
    print(f"  Skipped:   {aggregate['n_skip']}")
    print(f"  Failed:    {aggregate['n_fail']}")
    if aggregate["n_ok"] > 0:
        print(f"  INFORMATIONAL (ΔBIC < {DELTA_BIC_THRESHOLD}): {aggregate['n_informational']}")
        print(f"  SINGLEMODE_ONLY:  {aggregate['n_singlemode']}")
        if aggregate["delta_bic_median"] is not None:
            print(f"  ΔBIC median: {aggregate['delta_bic_median']:.1f}")
            print(f"  ΔBIC range:  [{aggregate['delta_bic_min']:.1f}, {aggregate['delta_bic_max']:.1f}]")
    if aggregate["informational_events"]:
        print(f"  Events with evidence: {', '.join(aggregate['informational_events'])}")
    print("=" * 60)
    print(f"Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
