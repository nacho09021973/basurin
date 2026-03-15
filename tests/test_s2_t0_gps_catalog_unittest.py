"""Tests for s2_ringdown_window._resolve_t0_gps.

Covers three resolution paths (Gap 6 from test_coverage_proposal.md):
  Path 1 — Legacy windows-array catalog schema
  Path 2 — Canonical reference catalog fallback (`gwtc_events_t0.json`)
  Path 3 — RuntimeError when neither source yields a value

The original tests cover Schema A (nested dict) and Schema B (scalar value).
Additions below cover the legacy windows schema and the canonical reference-catalog fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvp.s2_ringdown_window import _resolve_t0_gps


def _write_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_resolve_t0_gps_schema_a_nested_dict(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog_a.json", {"GW190521": {"t0_gps": 1242442967.4}})

    resolved = _resolve_t0_gps("GW190521", catalog_path, offline=True, run_dir=tmp_path / "runs" / "rid")
    assert isinstance(resolved, tuple) and len(resolved) == 4
    t0_gps, source, _lookup_key, _gwosc_cache = resolved

    assert t0_gps == pytest.approx(1242442967.4)
    assert source == str(catalog_path)


def test_resolve_t0_gps_schema_b_scalar_value(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog_b.json", {"GW190521": 1242442967.4})

    resolved = _resolve_t0_gps("GW190521", catalog_path, offline=True, run_dir=tmp_path / "runs" / "rid")
    assert isinstance(resolved, tuple) and len(resolved) == 4
    t0_gps, source, _lookup_key, _gwosc_cache = resolved

    assert t0_gps == pytest.approx(1242442967.4)
    assert source == str(catalog_path)


def test_resolve_t0_gps_event_missing_includes_sources_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog_missing.json", {"GW150914": {"t0_gps": 1126259462.4}})
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_T0_REFERENCE_CATALOG", tmp_path / "missing_gwtc_events_t0.json")

    with pytest.raises(RuntimeError, match=r"missing_t0_gps_offline") as exc:
        _resolve_t0_gps("GW190521", catalog_path, offline=True, run_dir=tmp_path / "runs" / "rid")

    message = str(exc.value)
    assert str(catalog_path) in message
    assert "sources_attempted" in message


# ---------------------------------------------------------------------------
# Path 1 — Legacy "windows" array catalog schema (previously untested)
# ---------------------------------------------------------------------------


def test_resolve_t0_gps_legacy_windows_schema(tmp_path: Path) -> None:
    """Legacy catalog {'windows': [{event_id, t0_ref: {value_gps}}]} → correct value."""
    catalog_path = _write_catalog(
        tmp_path / "catalog_legacy.json",
        {
            "windows": [
                {
                    "event_id": "GW150914",
                    "t0_ref": {"value_gps": 1126259462.4, "uncertainty_s": 0.001},
                }
            ]
        },
    )

    resolved = _resolve_t0_gps("GW150914", catalog_path)
    assert isinstance(resolved, tuple) and len(resolved) == 4
    t0_gps, source, _lookup_key, _gwosc_cache = resolved

    assert t0_gps == pytest.approx(1126259462.4)
    assert source == str(catalog_path)


def test_resolve_t0_gps_legacy_windows_wrong_event_raises(tmp_path: Path) -> None:
    """Legacy catalog with a different event → RuntimeError for unknown event."""
    catalog_path = _write_catalog(
        tmp_path / "catalog_legacy_miss.json",
        {
            "windows": [
                {
                    "event_id": "GW150914",
                    "t0_ref": {"value_gps": 1126259462.4},
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="GW190521"):
        _resolve_t0_gps("GW190521", catalog_path, offline=True, run_dir=tmp_path / "runs" / "rid")


def test_resolve_t0_gps_legacy_windows_missing_value_gps_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy entry without value_gps is skipped; no match → RuntimeError."""
    catalog_path = _write_catalog(
        tmp_path / "catalog_no_value.json",
        {
            "windows": [
                {
                    "event_id": "GW150914",
                    "t0_ref": {},  # no value_gps key
                }
            ]
        },
    )

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="missing_t0_gps_offline"):
        _resolve_t0_gps("GW150914", catalog_path, offline=True, run_dir=tmp_path / "runs" / "rid")


# ---------------------------------------------------------------------------
# Path 2 — Canonical reference catalog fallback (previously untested in isolation)
# ---------------------------------------------------------------------------


def _write_reference_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_resolve_t0_gps_reference_catalog_t0_gps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_catalog = _write_reference_catalog(
        tmp_path / "gwtc_events_t0.json",
        {"GW_TEST": {"t0_gps": 1234567890.1, "GPS": 9999.9}},
    )
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_T0_REFERENCE_CATALOG", reference_catalog)

    resolved = _resolve_t0_gps("GW_TEST", nonexistent_catalog)
    assert isinstance(resolved, tuple) and len(resolved) == 4
    t0_gps, source, _lookup_key, _gwosc_cache = resolved

    assert t0_gps == pytest.approx(1234567890.1)
    assert source == str(reference_catalog)


def test_resolve_t0_gps_reference_catalog_gps_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_catalog = _write_reference_catalog(
        tmp_path / "gwtc_events_t0.json",
        {"GW_PRIO": {"GPS": 1111111111.0}},
    )
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_T0_REFERENCE_CATALOG", reference_catalog)

    resolved = _resolve_t0_gps("GW_PRIO", nonexistent_catalog)
    assert isinstance(resolved, tuple) and len(resolved) == 4
    t0_gps, source, _lookup_key, _gwosc_cache = resolved

    assert t0_gps == pytest.approx(1111111111.0)
    assert source == str(reference_catalog)


def test_resolve_t0_gps_reference_catalog_event_time_gps_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_catalog = _write_reference_catalog(
        tmp_path / "gwtc_events_t0.json",
        {"GW_EVTIME": {"event_time_gps": 1010101010.5}},
    )
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_T0_REFERENCE_CATALOG", reference_catalog)

    resolved = _resolve_t0_gps("GW_EVTIME", nonexistent_catalog)
    assert isinstance(resolved, tuple) and len(resolved) == 4
    t0_gps, source, _lookup_key, _gwosc_cache = resolved

    assert t0_gps == pytest.approx(1010101010.5)
    assert source == str(reference_catalog)


def test_resolve_t0_gps_default_window_catalog_prefers_event_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_catalog = _write_catalog(tmp_path / "window_catalog_v1.json", {"GW150914": {"t0_gps": 1126259462.4}})
    event_metadata_dir = tmp_path / "docs" / "ringdown" / "event_metadata"
    event_metadata_dir.mkdir(parents=True, exist_ok=True)
    event_metadata_path = event_metadata_dir / "GW150914_metadata.json"
    event_metadata_path.write_text(
        json.dumps({"event_id": "GW150914", "t0_gps": 1126259462.4204}),
        encoding="utf-8",
    )

    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_WINDOW_CATALOG", default_catalog)
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_EVENT_METADATA_DIR", event_metadata_dir)
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_T0_REFERENCE_CATALOG", tmp_path / "missing_gwtc_events_t0.json")

    resolved = _resolve_t0_gps("GW150914", default_catalog, offline=True, run_dir=tmp_path / "runs" / "rid")
    assert isinstance(resolved, tuple) and len(resolved) == 4
    t0_gps, source, details, _gwosc_cache = resolved

    assert t0_gps == pytest.approx(1126259462.4204)
    assert source == str(event_metadata_path)
    assert details["lookup_key"] == "GW150914"


# ---------------------------------------------------------------------------
# Path 3 — RuntimeError when neither catalog nor metadata exists
# ---------------------------------------------------------------------------

def test_resolve_t0_gps_no_catalog_no_metadata_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No window catalog and no canonical reference catalog → RuntimeError mentioning event_id."""
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_T0_REFERENCE_CATALOG", tmp_path / "missing_gwtc_events_t0.json")

    with pytest.raises(RuntimeError, match="GW_UNKNOWN"):
        _resolve_t0_gps("GW_UNKNOWN", nonexistent_catalog, offline=True, run_dir=tmp_path / "runs" / "rid")


def test_resolve_t0_gps_offline_without_sources_raises_stable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_T0_REFERENCE_CATALOG", tmp_path / "missing_gwtc_events_t0.json")

    with pytest.raises(RuntimeError, match="missing_t0_gps_offline") as exc:
        _resolve_t0_gps("GW_OFFLINE", nonexistent_catalog, offline=True, run_dir=tmp_path / "runs" / "r1")

    message = str(exc.value)
    assert "sources_attempted" in message
    assert "window_catalog=" in message


def test_resolve_t0_gps_online_fetch_creates_run_cache_and_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.setattr("mvp.s2_ringdown_window.DEFAULT_T0_REFERENCE_CATALOG", tmp_path / "missing_gwtc_events_t0.json")

    calls: list[str] = []

    def _fake_fetch(event_id: str, retries: int = 3, timeout_s: int = 20) -> float:
        calls.append(event_id)
        return 1234567890.25

    monkeypatch.setattr("mvp.s2_ringdown_window._fetch_gwosc_event_gps", _fake_fetch)
    run_dir = tmp_path / "runs" / "rid"

    t0_gps, source, details, used_cache = _resolve_t0_gps(
        "GW_ONLINE", nonexistent_catalog, offline=False, run_dir=run_dir
    )
    assert t0_gps == pytest.approx(1234567890.25)
    assert details["lookup_key"] == "GW_ONLINE"
    assert details["gwosc_cache_path"] is not None
    assert used_cache is False
    assert source.endswith("GW_ONLINE.json")
    assert calls == ["GW_ONLINE"]

    cache_file = run_dir / "external_inputs" / "gwosc" / "event_time" / "GW_ONLINE.json"
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["event_id"] == "GW_ONLINE"
    assert payload["t0_gps"] == pytest.approx(1234567890.25)
    assert payload["source"] == "gwosc_api_v2"

    # Reuse cache; no additional network call expected.
    t0_again, source_again, details_again, used_cache_again = _resolve_t0_gps(
        "GW_ONLINE", nonexistent_catalog, offline=False, run_dir=run_dir
    )
    assert t0_again == pytest.approx(1234567890.25)
    assert source_again == str(cache_file)
    assert details_again["gwosc_cache_path"] == str(cache_file)
    assert used_cache_again is True
    assert calls == ["GW_ONLINE"]
