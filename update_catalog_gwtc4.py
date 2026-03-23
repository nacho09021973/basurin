#!/usr/bin/env python3
"""
update_catalog_gwtc4.py — Expand BASURIN canonical catalog with GWTC-4.0 events.

Pulls event metadata from GWOSC API, extracts final_spin (af) from the official
LVK PE summary table (Zenodo), applies quality filters, and optionally downloads
4 kHz HDF5 strain files.

Usage:
    # 1. Build/update catalog JSON only (no strain download):
    python update_catalog_gwtc4.py --catalog-only

    # 2. Full run: build catalog + download strain:
    python update_catalog_gwtc4.py

    # 3. Download strain for events already in catalog:
    python update_catalog_gwtc4.py --strain-only

    # 4. Include O4b Discovery Paper events (GW250114 etc.):
    python update_catalog_gwtc4.py --include-discovery-papers

Outputs:
    gw_events/gwtc_events_t1.json      — New canonical catalog (t0 = existing, t1 = expanded)
    gw_events/gwtc4_pe_summary.hdf5    — Cached PE summary table from Zenodo
    gw_events/strain_4k/               — Downloaded 4 kHz HDF5 strain files

Designed for ~/work/basurin/. Adjust BASE_DIR if needed.

Author: BASURIN pipeline (auto-generated 2026-03-22)
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(".")  # Run from ~/work/basurin/
GW_EVENTS_DIR = BASE_DIR / "data" / "losc"
STRAIN_DIR = BASE_DIR / "data" / "losc"
EXISTING_CATALOG = BASE_DIR / "data" / "losc" / "gwtc_events_t0.json"
OUTPUT_CATALOG = BASE_DIR / "data" / "losc" / "gwtc_events_t1.json"
PE_SUMMARY_CACHE = BASE_DIR / "data" / "losc" / "gwtc4_pe_summary.hdf5"

# GWOSC API
GWOSC_ALLEVENTS = "https://gwosc.org/eventapi/json/allevents/"
GWOSC_EVENT_DETAIL = "https://gwosc.org/eventapi/json/{catalog}/{event}/v{version}"

# Zenodo PE summary table (reweighted version — latest as of 2025-09)
PE_SUMMARY_URL = (
    "https://zenodo.org/api/records/17014085/files/"
    "IGWN-GWTC4p0-1a206db3d_721-PESummaryTable.hdf5/content"
)

# Quality filters (Moderate)
SNR_MIN = 8.0
P_ASTRO_MIN = 0.9
FAR_MAX = 1.0  # yr^{-1}

# Pipeline preference for extracting af from PE summary
# (order = decreasing preference)
PIPELINE_PRIORITY = [
    "Mixed+XO4a",
    "Mixed",
    "NRSur7dq4",
    "SEOBNRv5PHM",
    "IMRPhenomXO4a",
    "IMRPhenomXPHM",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[BASURIN] {msg}", flush=True)


def fetch_json(url, retries=3, delay=2.0):
    """Fetch JSON from URL with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                log(f"  Retry {attempt+1}/{retries} for {url}: {e}")
                time.sleep(delay)
            else:
                raise


def download_file(url, dest, retries=3, label=""):
    """Download file with progress indicator and exponential backoff."""
    tmp = Path(str(dest) + ".tmp")
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        with _print_lock:
                            print(f"\r  {label}[{pct:3d}%] {downloaded/(1<<20):.1f} MB", end="", flush=True)
            with _print_lock:
                print(flush=True)
            tmp.rename(dest)
            return True
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log(f"  {label}Retry {attempt+1}/{retries}: {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                log(f"  {label}FAILED after {retries} attempts: {e}")
                return False


# ---------------------------------------------------------------------------
# Step 1: Fetch all events from GWOSC API
# ---------------------------------------------------------------------------

def fetch_gwosc_events(include_discovery=False):
    """Fetch events from GWOSC, return dict keyed by commonName."""
    log("Fetching events from GWOSC API...")
    data = fetch_json(GWOSC_ALLEVENTS)
    raw_events = data.get("events", {})
    log(f"  Raw entries: {len(raw_events)}")

    # Keep only GWTC-4.0 (and optionally O4_Discovery_Papers)
    target_catalogs = {"GWTC-4.0"}
    if include_discovery:
        target_catalogs.add("O4_Discovery_Papers")

    # Group by commonName; prefer GWTC-4.0 over O4_Discovery_Papers
    CATALOG_PRIORITY = {"GWTC-4.0": 0, "O4_Discovery_Papers": 1}
    by_name = {}
    for key, ev in raw_events.items():
        cat = ev.get("catalog.shortName", "")
        if cat not in target_catalogs:
            continue
        name = ev["commonName"]
        version = ev.get("version", 1)
        cat_prio = CATALOG_PRIORITY.get(cat, 99)

        if name not in by_name:
            by_name[name] = ev
            by_name[name]["_cat_prio"] = cat_prio
        else:
            old_prio = by_name[name].get("_cat_prio", 99)
            if cat_prio < old_prio or (cat_prio == old_prio and version > by_name[name].get("version", 0)):
                by_name[name] = ev
                by_name[name]["_cat_prio"] = cat_prio

    log(f"  GWTC-4.0 + Discovery events: {len(by_name)}")
    return by_name


# ---------------------------------------------------------------------------
# Step 2: Load PE summary table for final_spin
# ---------------------------------------------------------------------------

def load_pe_summary():
    """Download (if needed) and parse the PE summary table."""
    import h5py
    import numpy as np

    if not PE_SUMMARY_CACHE.exists():
        log(f"Downloading PE summary table from Zenodo...")
        PE_SUMMARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ok = download_file(PE_SUMMARY_URL, PE_SUMMARY_CACHE)
        if not ok:
            log("FATAL: Could not download PE summary table.")
            sys.exit(1)
    else:
        log(f"Using cached PE summary: {PE_SUMMARY_CACHE}")

    with h5py.File(PE_SUMMARY_CACHE, "r") as f:
        data = f["summary_info"][:]

    # Build dict: event_name -> best af measurement
    af_data = {}
    for row in data:
        name = row["gw_name"].decode().strip()
        label = row["result_label"].decode().strip()
        has_af = not row["final_spin_median.mask"]
        if not has_af:
            continue

        # Determine pipeline priority
        p = 99
        for i, pref in enumerate(PIPELINE_PRIORITY):
            if pref in label:
                p = i
                break

        entry = {
            "af": float(row["final_spin_median"]),
            "af_lower": float(row["final_spin_lower"]),
            "af_upper": float(row["final_spin_upper"]),
            "Mf_pe": float(row["final_mass_source_median"]),
            "pe_pipeline": label,
            "priority": p,
        }

        if name not in af_data or p < af_data[name]["priority"]:
            af_data[name] = entry

    log(f"  PE summary: {len(af_data)} events with final_spin")
    return af_data


# ---------------------------------------------------------------------------
# Step 3: Get strain URLs and detector list per event
# ---------------------------------------------------------------------------

def get_strain_info(ev):
    """Extract 4kHz HDF5 strain URLs and detector list from event data."""
    strain = ev.get("strain", [])
    if not strain:
        # Need to fetch detailed JSON
        jsonurl = ev.get("jsonurl")
        if jsonurl:
            try:
                detail = fetch_json(jsonurl)
                # Navigate nested structure
                events_inner = detail.get("events", {})
                if events_inner:
                    first_ev = next(iter(events_inner.values()))
                    strain = first_ev.get("strain", [])
            except Exception:
                pass

    detectors = sorted(set(s["detector"] for s in strain))
    urls_4k_hdf5 = [
        s for s in strain
        if s.get("sampling_rate") == 4096
        and s.get("format") == "hdf5"
    ]
    return detectors, urls_4k_hdf5


# ---------------------------------------------------------------------------
# Step 4: Apply quality filters and build catalog
# ---------------------------------------------------------------------------

def build_catalog(gwosc_events, af_data, include_discovery=False,
                  snr_min=SNR_MIN, pastro_min=P_ASTRO_MIN, far_max=FAR_MAX):
    """Apply quality filters and build canonical catalog entries."""
    catalog = {}
    skipped = {"no_snr": [], "low_snr": [], "low_pastro": [], "high_far": [],
               "no_pe": [], "accepted": []}

    for name, ev in gwosc_events.items():
        cat = ev.get("catalog.shortName", "")
        snr = ev.get("network_matched_filter_snr")
        pastro = ev.get("p_astro")
        far = ev.get("far")

        # SNR filter
        if snr is None:
            skipped["no_snr"].append(name)
            # O4 Discovery Papers may lack SNR in the summary; include anyway
            if cat != "O4_Discovery_Papers":
                continue
        elif snr < snr_min:
            skipped["low_snr"].append(name)
            continue

        # p_astro filter
        if pastro is not None and pastro < pastro_min:
            skipped["low_pastro"].append(name)
            continue

        # FAR filter
        if far is not None and far > far_max:
            skipped["high_far"].append(name)
            continue

        # Get strain info
        detectors, strain_urls = get_strain_info(ev)

        # Get af from PE summary
        pe = af_data.get(name, {})
        af = pe.get("af")
        af_source = pe.get("pe_pipeline", None)

        # Build canonical entry
        entry = {
            "name": name,
            "catalog": cat,
            "GPS": ev.get("GPS"),
            "SNR": snr,
            "p_astro": pastro,
            "FAR_per_yr": far,
            "m1_source": ev.get("mass_1_source"),
            "m2_source": ev.get("mass_2_source"),
            "chi_eff": ev.get("chi_eff"),
            "Mf_source": ev.get("final_mass_source"),
            "af": af,
            "af_lower": pe.get("af_lower"),
            "af_upper": pe.get("af_upper"),
            "af_source": af_source,
            "DL_Mpc": ev.get("luminosity_distance"),
            "z": ev.get("redshift"),
            "detectors": detectors,
            "strain_urls_4k_hdf5": [s["url"] for s in strain_urls],
        }

        # Flag incomplete events
        flags = []
        if af is None:
            flags.append("af_pending")
        if ev.get("mass_1_source") is None:
            flags.append("pe_incomplete")
        if cat == "O4_Discovery_Papers":
            flags.append("discovery_paper_only")
        if flags:
            entry["flags"] = flags

        catalog[name] = entry
        skipped["accepted"].append(name)

    log(f"\nFilter summary:")
    log(f"  Accepted:    {len(skipped['accepted'])}")
    log(f"  No SNR:      {len(skipped['no_snr'])} {skipped['no_snr'][:5]}")
    log(f"  Low SNR:     {len(skipped['low_snr'])}")
    log(f"  Low p_astro: {len(skipped['low_pastro'])}")
    log(f"  High FAR:    {len(skipped['high_far'])}")

    return catalog


# ---------------------------------------------------------------------------
# Step 5: Merge with existing catalog
# ---------------------------------------------------------------------------

def merge_catalogs(new_catalog, existing_path):
    """Merge new GWTC-4.0 events with existing t0 catalog."""
    existing = {}
    if existing_path.exists():
        log(f"Loading existing catalog: {existing_path}")
        with open(existing_path) as f:
            existing = json.load(f)
        log(f"  Existing events: {len(existing)}")
    else:
        log(f"No existing catalog at {existing_path}, creating from scratch.")

    # Merge: new events take priority for updated fields,
    # but preserve existing events not in GWTC-4.0
    merged = dict(existing)
    new_count = 0
    updated_count = 0
    for name, entry in new_catalog.items():
        if name in merged:
            # Update existing entry with new GWTC-4.0 data
            old = merged[name]
            merged[name] = entry
            # Preserve any custom fields from old entry
            for k in ["glitch_mitigated", "notes"]:
                if k in old and k not in entry:
                    merged[name][k] = old[k]
            updated_count += 1
        else:
            merged[name] = entry
            new_count += 1

    log(f"  New events added: {new_count}")
    log(f"  Existing events updated: {updated_count}")
    log(f"  Total events: {len(merged)}")

    return merged


# ---------------------------------------------------------------------------
# Step 6: Download strain files
# ---------------------------------------------------------------------------

def _download_one_strain(name, url, strain_dir, idx, total):
    """Download a single strain file. Returns (name, url, status)."""
    fname = url.split("/")[-1]
    event_dir = strain_dir / name
    event_dir.mkdir(exist_ok=True)
    dest = event_dir / fname

    if dest.exists():
        return name, url, "skipped"

    label = f"[{idx}/{total}] {name}: {fname} "
    log(label.strip())
    ok = download_file(url, dest, label=label)
    return name, url, "ok" if ok else "failed"


def download_strain(catalog, strain_dir, workers=4):
    """Download 4 kHz HDF5 strain for all events in catalog (parallel)."""
    strain_dir.mkdir(parents=True, exist_ok=True)

    # Build flat list of (name, url) tasks
    tasks = []
    for name, ev in sorted(catalog.items()):
        for url in ev.get("strain_urls_4k_hdf5", []):
            tasks.append((name, url))

    if not tasks:
        log("No strain files to download.")
        return

    total = len(tasks)
    log(f"Strain download: {total} files, {workers} parallel workers")

    skipped = 0
    failed = 0
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one_strain, name, url, strain_dir, i, total): (name, url)
            for i, (name, url) in enumerate(tasks, 1)
        }
        for future in as_completed(futures):
            name, url = futures[future]
            try:
                _, _, status = future.result()
                if status == "skipped":
                    skipped += 1
                elif status == "failed":
                    failed += 1
                done += 1
            except Exception as e:
                log(f"  [ERR] {name}: {e}")
                failed += 1
                done += 1

    log(f"\nStrain download complete:")
    log(f"  Total files:   {total}")
    log(f"  Downloaded:    {done - skipped - failed}")
    log(f"  Skipped (exist): {skipped}")
    log(f"  Failed:        {failed}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expand BASURIN catalog with GWTC-4.0 events"
    )
    parser.add_argument("--catalog-only", action="store_true",
                        help="Build catalog JSON only, skip strain download")
    parser.add_argument("--strain-only", action="store_true",
                        help="Download strain for events already in catalog")
    parser.add_argument("--include-discovery-papers", action="store_true",
                        help="Include O4b Discovery Paper events (GW250114 etc.)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output catalog path (default: gw_events/gwtc_events_t1.json)")
    parser.add_argument("--snr-min", type=float, default=SNR_MIN)
    parser.add_argument("--pastro-min", type=float, default=P_ASTRO_MIN)
    parser.add_argument("--far-max", type=float, default=FAR_MAX)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel strain downloads (default: 4, use 1 for sequential)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_CATALOG

    # Strain-only mode
    if args.strain_only:
        if not output_path.exists():
            log(f"ERROR: Catalog not found at {output_path}. Run without --strain-only first.")
            sys.exit(1)
        with open(output_path) as f:
            catalog = json.load(f)
        download_strain(catalog, STRAIN_DIR, workers=args.workers)
        return

    # --- Build catalog ---
    log("=" * 60)
    log("BASURIN Catalog Expansion: GWTC-4.0")
    log("=" * 60)
    log(f"Filters: SNR >= {args.snr_min}, p_astro >= {args.pastro_min}, FAR < {args.far_max}/yr")
    log("")

    # Step 1: Fetch GWOSC events
    gwosc_events = fetch_gwosc_events(include_discovery=args.include_discovery_papers)

    # Step 2: Load PE summary for af
    try:
        import h5py
        import numpy as np
    except ImportError:
        log("ERROR: h5py and numpy are required. Install with:")
        log("  pip install h5py numpy")
        sys.exit(1)

    af_data = load_pe_summary()

    # Step 3: Build filtered catalog
    new_catalog = build_catalog(gwosc_events, af_data,
                                include_discovery=args.include_discovery_papers,
                                snr_min=args.snr_min,
                                pastro_min=args.pastro_min,
                                far_max=args.far_max)

    # Step 4: Merge with existing
    merged = merge_catalogs(new_catalog, EXISTING_CATALOG)

    # Step 5: Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    log(f"\nCatalog written to: {output_path}")

    # Summary stats
    n_with_af = sum(1 for ev in merged.values() if ev.get("af") is not None)
    n_gwtc4 = sum(1 for ev in merged.values() if ev.get("catalog") == "GWTC-4.0")
    n_disc = sum(1 for ev in merged.values() if ev.get("catalog") == "O4_Discovery_Papers")
    n_legacy = len(merged) - n_gwtc4 - n_disc
    log(f"\nCatalog summary:")
    log(f"  Total events:    {len(merged)}")
    log(f"  Legacy (O1-O3):  {n_legacy}")
    log(f"  GWTC-4.0 (O4a):  {n_gwtc4}")
    log(f"  Discovery papers: {n_disc}")
    log(f"  With af from PE: {n_with_af}")
    flagged = [n for n, ev in merged.items() if ev.get("flags")]
    if flagged:
        log(f"  Flagged events:  {len(flagged)}")
        for n in flagged:
            log(f"    {n}: {merged[n]['flags']}")

    # Step 6: Download strain (unless --catalog-only)
    if not args.catalog_only:
        log("\n" + "=" * 60)
        log("Downloading 4 kHz HDF5 strain files...")
        log("=" * 60)
        download_strain(merged, STRAIN_DIR, workers=args.workers)
    else:
        log("\n(Skipping strain download — use --strain-only later)")

    log("\nDone.")


if __name__ == "__main__":
    main()
