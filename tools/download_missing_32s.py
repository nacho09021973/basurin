#!/usr/bin/env python3
"""Download missing 32s HDF5 strain files for GW events.

Reads the diagnostic output and fetches only the missing 32s files from GWOSC.
Also re-downloads corrupted files (e.g. truncated 4096s).

Usage:
    python tools/download_missing_32s.py [--losc-root data/losc] [--dry-run]
    python tools/download_missing_32s.py --also-4096s   # include 4096s re-downloads for corrupted files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

GWOSC_API_BASE = "https://gwosc.org/api/v2"
USER_AGENT = "basurin-fetch/1.1"
DEFAULT_LOSC_ROOT = Path(__file__).resolve().parents[1] / "data" / "losc"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "gwtc_events_t0.json"

# ── Events needing 32s downloads ──────────────────────────────────────────
# Category 1: Have 4096s but need 32s for H1+L1 (both detectors)
NEED_32S_BOTH = [
    "GW190408_181802",
    "GW190412",
    "GW190421_213856",
    "GW190503_185404",
    "GW190512_180714",
    "GW190513_205428",
    "GW190517_055101",
    "GW190519_153544",
    "GW190521_074359",
    "GW190602_175927",
    "GW190706_222641",
    "GW190707_093326",
    "GW190720_000836",
    "GW190727_060333",
    "GW190828_063405",
    "GW190910_112807",
]

# Category 2: Missing specific detector 32s file
NEED_32S_SINGLE: dict[str, list[str]] = {
    "GW190620_030421": ["H1"],
    "GW190630_185205": ["H1"],
    "GW190708_232457": ["H1"],
    "GW190925_232845": ["L1"],
    "GW191216_213338": ["L1"],
    "GW200112_155838": ["H1"],
    "GW200302_015811": ["L1"],
}

# Category 3: Corrupted 4096s files needing re-download
CORRUPTED_4096S: dict[str, list[str]] = {
    "GW200202_154313": ["H1", "L1"],  # truncated 4096s files
}


def _headers_json() -> dict[str, str]:
    return {"Accept": "application/json", "User-Agent": USER_AGENT}


def _headers_binary() -> dict[str, str]:
    return {"Accept": "application/octet-stream,*/*", "User-Agent": USER_AGENT}


def _http_get_json(url: str, timeout_s: int = 60) -> dict:
    """Fetch JSON from GWOSC API with retries."""
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            req = Request(url, headers=_headers_json())
            with urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed fetching {url}: {last_exc}")


def _download_file(url: str, target: Path, timeout_s: int = 120) -> None:
    """Download a file with retries and progress."""
    for attempt in range(3):
        try:
            req = Request(url, headers=_headers_binary())
            with urlopen(req, timeout=timeout_s) as resp, target.open("wb") as fh:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                for chunk in iter(lambda: resp.read(1024 * 1024), b""):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r    {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
                if total > 0:
                    print()
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            if target.exists():
                target.unlink()
            if attempt == 2:
                raise RuntimeError(f"Download failed after 3 attempts: {url}: {exc}") from exc
            time.sleep(2.0 * (attempt + 1))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_event_version(event_id: str) -> str:
    """Try -v1 then -v2 to find a valid event version."""
    for suffix in ("-v2", "-v1"):
        candidate = f"{event_id}{suffix}"
        url = f"{GWOSC_API_BASE}/event-versions/{candidate}"
        try:
            _http_get_json(url)
            return candidate
        except Exception:
            continue
    raise RuntimeError(f"Cannot resolve event version for {event_id}")


def _find_strain_url(
    event_version: str,
    detector: str,
    duration: int = 32,
    sample_rate: int = 4096,
    fmt: str = "hdf5",
) -> str | None:
    """Find the best matching strain file URL from GWOSC API."""
    url = f"{GWOSC_API_BASE}/event-versions/{event_version}/strain-files?detector={detector}"
    try:
        payload = _http_get_json(url)
    except Exception as exc:
        print(f"    WARN: API error for {detector}: {exc}")
        return None

    # Look through strain_files for best match
    files = payload.get("strain_files") or payload.get("results") or []
    if not isinstance(files, list):
        return None

    best_url: str | None = None
    best_score = -1

    for entry in files:
        if not isinstance(entry, dict):
            continue
        dl_url = entry.get("download_url", "")
        if not dl_url:
            continue

        # Check format
        file_format = str(entry.get("file_format", "")).upper()
        is_hdf = file_format in {"HDF", "HDF5"} or dl_url.lower().endswith((".hdf5", ".h5"))
        if fmt == "hdf5" and not is_hdf:
            continue

        # Score by duration and sample rate match
        score = 0
        dur = entry.get("duration")
        sr = entry.get("sample_rate")
        try:
            if dur is not None and abs(float(dur) - duration) < 1:
                score += 10
            if sr is not None and abs(float(sr) - sample_rate) < 1:
                score += 5
        except (ValueError, TypeError):
            pass

        if score > best_score:
            best_score = score
            best_url = dl_url

    return best_url


def _write_sha256sums(event_dir: Path) -> None:
    h5_files = sorted(
        [p for p in event_dir.iterdir() if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}],
        key=lambda p: p.name,
    )
    if not h5_files:
        return
    lines = [f"{_sha256_file(p)}  {p.name}\n" for p in h5_files]
    (event_dir / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")


def download_for_event(
    event_id: str,
    detectors: list[str],
    losc_root: Path,
    duration: int = 32,
    dry_run: bool = False,
    force: bool = False,
) -> list[str]:
    """Download strain files for one event + detector list."""
    event_dir = losc_root / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{event_id}] detectors={detectors}, duration={duration}s")

    event_version = _resolve_event_version(event_id)
    print(f"    resolved version: {event_version}")

    downloaded: list[str] = []

    for det in detectors:
        # Check if already present (skip if not force).
        # Match any HDF5 with detector name in it for the target duration.
        # This avoids re-downloading when GWOSC returns a different URL/mirror.
        if not force:
            existing = [
                p for p in event_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".h5", ".hdf5"}
                and det.upper() in p.name.upper()
                and (f"-{duration}." in p.name or f"-{duration}-" in p.name)
            ]
            if existing:
                print(f"    {det}: SKIP (already have {existing[0].name})")
                downloaded.append(str(existing[0]))
                continue

        url = _find_strain_url(event_version, det, duration=duration)
        if not url:
            print(f"    {det}: NO {duration}s file available on GWOSC")
            continue

        fname = Path(urlparse(url).path).name
        target = event_dir / fname

        if target.exists() and not force:
            print(f"    {det}: SKIP {fname} (exists)")
            downloaded.append(str(target))
            continue

        if dry_run:
            print(f"    {det}: WOULD download {fname}")
            continue

        print(f"    {det}: downloading {fname}...")
        try:
            _download_file(url, target)
            downloaded.append(str(target))
            print(f"    {det}: OK ({target.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"    {det}: FAILED - {exc}")

    if downloaded and not dry_run:
        _write_sha256sums(event_dir)

    return downloaded


def main() -> int:
    ap = argparse.ArgumentParser(description="Download missing 32s HDF5 strain files for GW events")
    ap.add_argument("--losc-root", type=Path, default=DEFAULT_LOSC_ROOT,
                     help=f"LOSC data root (default: {DEFAULT_LOSC_ROOT})")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    ap.add_argument("--also-4096s", action="store_true",
                     help="Also re-download corrupted 4096s files")
    ap.add_argument("--events", nargs="*", default=None,
                     help="Limit to specific event IDs (default: all missing)")
    ap.add_argument("--force", action="store_true",
                     help="Re-download even if files already exist")
    args = ap.parse_args()

    losc_root = args.losc_root.resolve()
    losc_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DOWNLOAD MISSING 32s DETECTOR DATA")
    print("=" * 60)
    print(f"  LOSC root: {losc_root}")
    print(f"  Dry run:   {args.dry_run}")

    total_downloaded: list[str] = []
    total_failed: list[str] = []

    # ── 1. Events needing 32s for both H1+L1 ─────────────────────────────
    print(f"\n--- Category 1: Need 32s for H1+L1 ({len(NEED_32S_BOTH)} events) ---")
    for event_id in NEED_32S_BOTH:
        if args.events and event_id not in args.events:
            continue
        try:
            dl = download_for_event(event_id, ["H1", "L1"], losc_root,
                                    duration=32, dry_run=args.dry_run, force=args.force)
            total_downloaded.extend(dl)
        except Exception as exc:
            print(f"  [{event_id}] ERROR: {exc}")
            total_failed.append(event_id)

    # ── 2. Events needing 32s for specific detector ───────────────────────
    print(f"\n--- Category 2: Need specific detector 32s ({len(NEED_32S_SINGLE)} events) ---")
    for event_id, dets in NEED_32S_SINGLE.items():
        if args.events and event_id not in args.events:
            continue
        try:
            dl = download_for_event(event_id, dets, losc_root,
                                    duration=32, dry_run=args.dry_run, force=args.force)
            total_downloaded.extend(dl)
        except Exception as exc:
            print(f"  [{event_id}] ERROR: {exc}")
            total_failed.append(event_id)

    # ── 3. Corrupted 4096s re-download ────────────────────────────────────
    if args.also_4096s and CORRUPTED_4096S:
        print(f"\n--- Category 3: Re-download corrupted 4096s ({len(CORRUPTED_4096S)} events) ---")
        for event_id, dets in CORRUPTED_4096S.items():
            if args.events and event_id not in args.events:
                continue
            try:
                dl = download_for_event(event_id, dets, losc_root,
                                        duration=4096, dry_run=args.dry_run, force=True)
                total_downloaded.extend(dl)
            except Exception as exc:
                print(f"  [{event_id}] ERROR: {exc}")
                total_failed.append(event_id)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"  Downloaded: {len(total_downloaded)} files")
    if total_failed:
        print(f"  Failed:     {len(total_failed)} events: {', '.join(total_failed)}")
    print(f"{'=' * 60}")

    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
