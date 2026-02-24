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
