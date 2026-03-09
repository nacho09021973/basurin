#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

API_BASE = "https://gwosc.org/api/v2"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "basurin-gwosc-sync/0.1",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    # Descarga simple, sin rangos paralelos: evita el problema de Invalid range header.
    cmd = [
        "curl", "-fL",
        "--retry", "3",
        "--retry-delay", "2",
        "-o", str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    out = subprocess.check_output(["sha256sum", str(path)], text=True)
    return out.split()[0]


def _candidate_versions(event_id: str) -> list[dict]:
    data = _get_json(f"{API_BASE}/events/{urllib.parse.quote(event_id)}")
    versions = data.get("versions", [])
    # Orden descendente por versión numérica; fallback estable si falta el campo.
    versions = sorted(
        versions,
        key=lambda v: int(v.get("version", -1)),
        reverse=True,
    )
    return versions


def _strain_results(version_name: str, detector: str, sample_rate: int, duration: int, file_format: str) -> list[dict]:
    qs = urllib.parse.urlencode(
        {
            "detector": detector,
            "sample-rate": sample_rate,
            "duration": duration,
            "file-format": file_format,
            "pagesize": 20,
        }
    )
    url = f"{API_BASE}/event-versions/{urllib.parse.quote(version_name)}/strain-files?{qs}"
    data = _get_json(url)
    return data.get("results", [])


def _pick_version_with_h1_l1(event_id: str, sample_rate: int, duration: int, file_format: str) -> tuple[str, dict, dict]:
    versions = _candidate_versions(event_id)
    if not versions:
        raise RuntimeError(f"No versions found for event_id={event_id}")

    for v in versions:
        version_name = v.get("name")
        if not version_name:
            detail_url = v.get("detail_url", "")
            if detail_url:
                version_name = detail_url.rstrip("/").split("/")[-1]
        if not version_name and v.get("version") is not None:
            version_name = f"{event_id}-v{v['version']}"
        if not version_name:
            continue
        h1 = _strain_results(version_name, "H1", sample_rate, duration, file_format)
        l1 = _strain_results(version_name, "L1", sample_rate, duration, file_format)
        if h1 and l1:
            return version_name, h1[0], l1[0]

    raise RuntimeError(
        f"No event version with both H1 and L1 found for event_id={event_id} "
        f"(sample_rate={sample_rate}, duration={duration}, format={file_format})"
    )


def _download_event(event_id: str, losc_root: Path, sample_rate: int, duration: int, file_format: str, skip_existing: bool) -> int:
    out_dir = losc_root / event_id
    out_dir.mkdir(parents=True, exist_ok=True)

    version_name, h1_meta, l1_meta = _pick_version_with_h1_l1(
        event_id=event_id,
        sample_rate=sample_rate,
        duration=duration,
        file_format=file_format,
    )

    h1_url = h1_meta.get("download_url") or h1_meta.get("hdf5_url")
    l1_url = l1_meta.get("download_url") or l1_meta.get("hdf5_url")
    if not h1_url or not l1_url:
        raise RuntimeError(f"Missing download_url for event_id={event_id}, version={version_name}")

    h1_name = os.path.basename(urllib.parse.urlparse(h1_url).path)
    l1_name = os.path.basename(urllib.parse.urlparse(l1_url).path)
    h1_path = out_dir / h1_name
    l1_path = out_dir / l1_name

    print(json.dumps({
        "event_id": event_id,
        "event_version": version_name,
        "out_dir": str(out_dir),
        "h1_url": h1_url,
        "l1_url": l1_url,
    }, ensure_ascii=False))

    if not (skip_existing and h1_path.exists()):
        _download(h1_url, h1_path)
    if not (skip_existing and l1_path.exists()):
        _download(l1_url, l1_path)

    print(json.dumps({
        "event_id": event_id,
        "h1_path": str(h1_path),
        "h1_sha256": _sha256(h1_path),
        "l1_path": str(l1_path),
        "l1_sha256": _sha256(l1_path),
    }, ensure_ascii=False))

    return 0


def _iter_empty_event_dirs(losc_root: Path) -> Iterable[str]:
    if not losc_root.exists():
        return []
    for p in sorted(losc_root.iterdir()):
        if not p.is_dir():
            continue
        has_files = any(x.is_file() for x in p.rglob("*"))
        if not has_files:
            yield p.name


def main() -> int:
    ap = argparse.ArgumentParser(description="Populate data/losc/<EVENT_ID>/ from GWOSC")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--event-id", help="Single GWOSC event id, e.g. GW150914")
    g.add_argument("--from-empty-dirs", action="store_true", help="Process empty event dirs under --losc-root")
    ap.add_argument("--losc-root", default="data/losc")
    ap.add_argument("--sample-rate", type=int, default=4, choices=[4, 16])
    ap.add_argument("--duration", type=int, default=4096, choices=[32, 4096])
    ap.add_argument("--file-format", default="hdf5", choices=["hdf5", "gwf", "txt"])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    losc_root = Path(args.losc_root)

    event_ids = [args.event_id] if args.event_id else list(_iter_empty_event_dirs(losc_root))
    if not event_ids:
        print("NO_TARGET_EVENTS", file=sys.stderr)
        return 2

    rc = 0
    for event_id in event_ids:
        try:
            _download_event(
                event_id=event_id,
                losc_root=losc_root,
                sample_rate=args.sample_rate,
                duration=args.duration,
                file_format=args.file_format,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            rc = 1
            print(json.dumps({
                "event_id": event_id,
                "status": "ERROR",
                "error": str(e),
            }, ensure_ascii=False), file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
