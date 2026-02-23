from __future__ import annotations

import io
import json
import sys
from urllib.request import Request

import tools.fetch_losc_event as fetch_losc_event


class _FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_http_get_json_adds_format_api_and_accept_header(monkeypatch):
    calls: list[tuple[object, int]] = []

    def fake_urlopen(req, timeout=60):
        calls.append((req, timeout))
        return _FakeHTTPResponse(b'{"ok": true}')

    monkeypatch.setattr(fetch_losc_event, "urlopen", fake_urlopen)

    payload = fetch_losc_event._http_get_json(
        "https://gwosc.org/api/v2/event-versions/GW170814-v1/strain-files"
    )

    assert payload == {"ok": True}
    assert len(calls) == 1

    req, timeout = calls[0]
    assert timeout == 60
    assert isinstance(req, Request)
    assert "format=api" in req.full_url
    assert req.headers["Accept"] == "application/json"


def test_select_h1_l1_files_selects_expected_rows():
    payload = {
        "strain_files": [
            {"detector": "V1", "download_url": "https://example.org/V-V1_TEST-32.hdf5"},
            {"detector": "H1", "download_url": "https://example.org/H-H1_TEST-32.hdf5"},
            {"detector": "L1", "download_url": "https://example.org/L-L1_TEST-32.hdf5"},
        ]
    }

    selected = fetch_losc_event._select_h1_l1_files(payload)

    assert set(selected.keys()) == {"H1", "L1"}
    assert selected["H1"]["download_url"].endswith("H-H1_TEST-32.hdf5")
    assert selected["L1"]["download_url"].endswith("L-L1_TEST-32.hdf5")


def test_fetch_losc_event_creates_aliases_and_inventory(tmp_path, monkeypatch):
    out_root = tmp_path / "data" / "losc"
    event_id = "GW170814"

    payload = {
        "strain_files": [
            {"detector": "H1", "download_url": "https://example.org/H-H1_TEST-32.hdf5"},
            {"detector": "L1", "download_url": "https://example.org/L-L1_TEST-32.hdf5"},
        ]
    }

    def fake_urlopen(req, timeout=60):
        url = req.full_url if isinstance(req, Request) else str(req)
        if "strain-files" in url:
            return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))
        if "H-H1" in url:
            return _FakeHTTPResponse(b"h1-bytes")
        if "L-L1" in url:
            return _FakeHTTPResponse(b"l1-bytes")
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(fetch_losc_event, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_losc_event.py", "--event-id", event_id, "--out-root", str(out_root)],
    )

    assert fetch_losc_event.main() == 0

    event_dir = out_root / event_id
    h1_alias = event_dir / "H1.hdf5"
    l1_alias = event_dir / "L1.hdf5"
    assert h1_alias.is_symlink()
    assert l1_alias.is_symlink()
    assert h1_alias.resolve().read_bytes() == b"h1-bytes"
    assert l1_alias.resolve().read_bytes() == b"l1-bytes"

    inv = event_dir / "INVENTORY.sha256"
    assert inv.exists()
    content = inv.read_text(encoding="utf-8")
    assert "H1.hdf5" in content
    assert "L1.hdf5" in content

    for path in event_dir.rglob("*"):
        path.resolve().relative_to(out_root.resolve())


def test_fetch_losc_event_fails_when_h1_or_l1_missing(tmp_path, monkeypatch):
    out_root = tmp_path / "data" / "losc"

    payload = {
        "strain_files": [
            {"detector": "H1", "download_url": "https://example.org/H-H1_TEST-32.hdf5"}
        ]
    }

    monkeypatch.setattr(
        fetch_losc_event,
        "urlopen",
        lambda *a, **k: _FakeHTTPResponse(json.dumps(payload).encode("utf-8")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_losc_event.py", "--event-id", "GW170814", "--out-root", str(out_root)],
    )

    try:
        fetch_losc_event.main()
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "missing detector" in str(exc)
