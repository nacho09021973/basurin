"""Tests for s2_ringdown_window._resolve_t0_gps.

Covers three resolution paths (Gap 6 from test_coverage_proposal.md):
  Path 1 — Legacy windows-array catalog schema
  Path 2 — Event metadata file (t_coalescence_gps / t0_ref_gps / GPS key priority)
  Path 3 — RuntimeError when neither source yields a value

The original tests cover Schema A (nested dict) and Schema B (scalar value).
Additions below cover the legacy windows schema and the metadata-file fallback.
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

    t0_gps, source = _resolve_t0_gps("GW190521", catalog_path)

    assert t0_gps == pytest.approx(1242442967.4)
    assert source == str(catalog_path)


def test_resolve_t0_gps_schema_b_scalar_value(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog_b.json", {"GW190521": 1242442967.4})

    t0_gps, source = _resolve_t0_gps("GW190521", catalog_path)

    assert t0_gps == pytest.approx(1242442967.4)
    assert source == str(catalog_path)


def test_resolve_t0_gps_event_missing_includes_event_and_catalog_path(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog_missing.json", {"GW150914": {"t0_gps": 1126259462.4}})

    with pytest.raises(RuntimeError, match=r"GW190521") as exc:
        _resolve_t0_gps("GW190521", catalog_path)

    message = str(exc.value)
    assert str(catalog_path) in message
    assert "available_keys" in message
    assert "detected_schema" in message


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

    t0_gps, source = _resolve_t0_gps("GW150914", catalog_path)

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
        _resolve_t0_gps("GW190521", catalog_path)


def test_resolve_t0_gps_legacy_windows_missing_value_gps_raises(tmp_path: Path) -> None:
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

    with pytest.raises(RuntimeError):
        _resolve_t0_gps("GW150914", catalog_path)


# ---------------------------------------------------------------------------
# Path 2 — Metadata file fallback (previously untested in isolation)
# ---------------------------------------------------------------------------


def _write_metadata(tmp_path: Path, event_id: str, payload: dict) -> Path:
    meta_dir = tmp_path / "docs" / "ringdown" / "event_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{event_id}_metadata.json"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    return meta_path


def test_resolve_t0_gps_metadata_t_coalescence_gps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """t_coalescence_gps takes highest priority in metadata file."""
    _write_metadata(tmp_path, "GW_TEST", {"t_coalescence_gps": 1234567890.1, "GPS": 9999.9})
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.chdir(tmp_path)

    t0_gps, source = _resolve_t0_gps("GW_TEST", nonexistent_catalog)

    assert t0_gps == pytest.approx(1234567890.1)


def test_resolve_t0_gps_metadata_t0_ref_gps_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """t0_ref_gps wins when t_coalescence_gps is absent."""
    _write_metadata(tmp_path, "GW_PRIO", {"t0_ref_gps": 1111111111.0, "GPS": 9999.0})
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.chdir(tmp_path)

    t0_gps, source = _resolve_t0_gps("GW_PRIO", nonexistent_catalog)

    assert t0_gps == pytest.approx(1111111111.0)


def test_resolve_t0_gps_metadata_gps_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPS key is used when neither t_coalescence_gps nor t0_ref_gps are present."""
    _write_metadata(tmp_path, "GW_GPS", {"GPS": 1126259462.4})
    nonexistent_catalog = tmp_path / "no_catalog.json"
    monkeypatch.chdir(tmp_path)

    t0_gps, source = _resolve_t0_gps("GW_GPS", nonexistent_catalog)

    assert t0_gps == pytest.approx(1126259462.4)


# ---------------------------------------------------------------------------
# Path 3 — RuntimeError when neither catalog nor metadata exists
# ---------------------------------------------------------------------------


def test_resolve_t0_gps_no_catalog_no_metadata_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No catalog file and no metadata file → RuntimeError mentioning event_id."""
    nonexistent_catalog = tmp_path / "no_catalog.json"
    # chdir ensures docs/ringdown/event_metadata doesn't exist relative to cwd
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="GW_UNKNOWN"):
        _resolve_t0_gps("GW_UNKNOWN", nonexistent_catalog)
