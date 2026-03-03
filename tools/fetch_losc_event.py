#!/usr/bin/env python3
"""Fetch canonical LOSC/GWOSC HDF5 strain files for one event.

Usage:
    python tools/fetch_losc_event.py --event-id GW170814 --out-root data/losc
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
from urllib.parse import parse_qsl, urlencode, urlparse
from urllib.request import Request, urlopen

GWOSC_EVENT_VERSIONS_API = "https://gwosc.org/api/v2/event-versions/{event_version}/strain-files"
DEFAULT_USER_AGENT = "basurin-fetch/1.0"


def _json_headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": DEFAULT_USER_AGENT,
    }


def _binary_headers() -> dict[str, str]:
    return {
        "Accept": "application/octet-stream,*/*",
        "Accept-Encoding": "identity",
        "User-Agent": DEFAULT_USER_AGENT,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_event_dir(out_root: Path, event_id: str) -> Path:
    event_dir = (out_root / event_id).resolve()
    out_root_resolved = out_root.resolve()
    try:
        event_dir.relative_to(out_root_resolved)
    except ValueError as exc:
        raise SystemExit(f"ERROR: computed event dir escapes out_root: {event_dir}") from exc
    return event_dir


def _http_get_json(url: str, timeout_s: int = 60) -> dict:
    parsed = urlparse(url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if not any(k == "format" for k, _ in query_pairs):
        query_pairs.append(("format", "api"))
        parsed = parsed._replace(query=urlencode(query_pairs))
    request = Request(parsed.geturl(), headers=_json_headers())
    try:
        with urlopen(request, timeout=timeout_s) as resp:
            content_type = str(getattr(resp, "headers", {}).get("Content-Type", "")).lower()
            body = resp.read().decode("utf-8")
            payload = json.loads(body)
            if content_type and "json" not in content_type:
                raise SystemExit(
                    f"ERROR: GWOSC returned non-JSON content type '{content_type}' for {request.full_url}"
                )
            return payload
    except HTTPError as exc:
        if exc.code == 406:
            print(
                "ERROR: GWOSC HTTP 406 for JSON API request\n"
                f"  url={request.full_url}\n"
                f"  headers={_json_headers()}\n"
                "  hint: pruebe --event-version auto o una versión explícita (p.ej. GW150914-v2)",
                file=sys.stderr,
            )
        raise


def _resolve_event_payload(event_id: str, event_version: str, timeout_s: int) -> tuple[str, dict]:
    if event_version != "auto":
        api_url = GWOSC_EVENT_VERSIONS_API.format(event_version=event_version)
        return event_version, _http_get_json(api_url, timeout_s=timeout_s)

    candidates = [f"{event_id}-v1", f"{event_id}-v2"]
    last_exc: Exception | None = None
    for candidate in candidates:
        api_url = GWOSC_EVENT_VERSIONS_API.format(event_version=candidate)
        try:
            return candidate, _http_get_json(api_url, timeout_s=timeout_s)
        except HTTPError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise SystemExit(
            "ERROR: no se pudo resolver una versión válida del evento. "
            f"Intentadas: {', '.join(candidates)}"
        ) from last_exc
    raise SystemExit("ERROR: no candidate event versions available")


def _download_file(url: str, target: Path, timeout_s: int = 60, retries: int = 3) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(url, headers=_binary_headers())
        try:
            with urlopen(request, timeout=timeout_s) as resp, target.open("wb") as fh:
                for chunk in iter(lambda: resp.read(1024 * 1024), b""):
                    fh.write(chunk)
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(0.2 * attempt)
    raise SystemExit(f"ERROR: failed downloading {url} after {retries} attempts: {last_exc}")


def _detector_from_row(row: dict) -> str | None:
    detector = row.get("detector")
    if isinstance(detector, str) and detector in {"H1", "L1"}:
        return detector
    url = str(row.get("download_url", ""))
    for det in ("H1", "L1"):
        if f"-{det}_" in url or f"/{det}." in url or f"_{det}." in url:
            return det
    return None


def _select_h1_l1_files(payload: dict) -> dict[str, dict]:
    files = payload.get("strain_files")
    if not isinstance(files, list):
        raise SystemExit("ERROR: malformed GWOSC response: missing list strain_files")

    selected: dict[str, dict] = {}
    for row in files:
        if not isinstance(row, dict):
            continue
        url = row.get("download_url")
        if not isinstance(url, str) or not url:
            continue
        det = _detector_from_row(row)
        if det in {"H1", "L1"} and det not in selected:
            selected[det] = row

    missing = [det for det in ("H1", "L1") if det not in selected]
    if missing:
        raise SystemExit(
            "ERROR: missing detector download_url(s): "
            + ", ".join(missing)
            + " in GWOSC event-versions payload"
        )
    return selected


def _write_inventory(event_dir: Path) -> Path:
    inventory_path = event_dir / "INVENTORY.sha256"
    lines = []
    for alias in ("H1.hdf5", "L1.hdf5"):
        alias_path = event_dir / alias
        sha = _sha256_file(alias_path)
        lines.append(f"{sha}  {alias}\n")
    inventory_path.write_text("".join(lines), encoding="utf-8")
    return inventory_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch canonical LOSC HDF5 files for one event")
    ap.add_argument("--event-id", required=True)
    ap.add_argument("--out-root", default="data/losc")
    ap.add_argument("--event-version", default="auto", help="GWOSC event version (default: auto)")
    ap.add_argument("--timeout-s", type=int, default=60, help="HTTP timeout seconds")
    args = ap.parse_args()

    event_id = args.event_id.strip()
    if not event_id:
        print("ERROR: --event-id cannot be empty", file=sys.stderr)
        return 2

    out_root = Path(args.out_root)
    event_dir = _safe_event_dir(out_root, event_id)
    event_dir.mkdir(parents=True, exist_ok=True)

    selected_event_version, payload = _resolve_event_payload(
        event_id=event_id,
        event_version=str(args.event_version).strip() or "auto",
        timeout_s=args.timeout_s,
    )

    selected = _select_h1_l1_files(payload)
    downloaded: dict[str, Path] = {}

    for det in ("H1", "L1"):
        row = selected[det]
        url = str(row["download_url"])
        name = Path(urlparse(url).path).name
        if not name:
            raise SystemExit(f"ERROR: cannot infer filename from URL for {det}: {url}")

        target = event_dir / name
        _download_file(url, target, timeout_s=args.timeout_s)
        downloaded[det] = target

        alias = event_dir / f"{det}.hdf5"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        rel_target = os.path.relpath(target, start=alias.parent)
        alias.symlink_to(rel_target)

    inventory_path = _write_inventory(event_dir)

    print(json.dumps({
        "event_id": event_id,
        "event_version": selected_event_version,
        "event_dir": event_dir.as_posix(),
        "files": {det: path.name for det, path in downloaded.items()},
        "aliases": {det: f"{det}.hdf5" for det in ("H1", "L1")},
        "inventory": inventory_path.name,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
