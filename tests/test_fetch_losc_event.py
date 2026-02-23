from __future__ import annotations

import io
import json
import sys

import scripts.fetch_losc_event as fetch_losc_event


class _FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fetch_losc_event_creates_aliases_and_inventory(tmp_path, monkeypatch):
    out_root = tmp_path / "data" / "losc"
    event_id = "GW170814"

    payload = {
        "strain_files": [
            {"detector": "H1", "download_url": "https://example.org/H-H1_TEST-32.hdf5"},
            {"detector": "L1", "download_url": "https://example.org/L-L1_TEST-32.hdf5"},
        ]
    }

    def fake_urlopen(url, timeout=60):
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
