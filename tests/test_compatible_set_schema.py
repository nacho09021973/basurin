import json
from pathlib import Path

import pytest

from mvp.s5_aggregate import validate_compatible_set_canonical


def test_accepts_canonical_schema(tmp_path: Path) -> None:
    cs_path = tmp_path / "compatible_set.json"
    payload = {
        "schema_version": 1,
        "event_id": "GW150914",
        "compatible_geometry_ids": ["geo_002", "geo_001", "geo_002"],
    }
    cs_path.write_text(json.dumps(payload), encoding="utf-8")

    extracted = validate_compatible_set_canonical(payload, cs_path)

    assert extracted == ["geo_001", "geo_002"]


def test_rejects_legacy_formats(tmp_path: Path) -> None:
    cs_path = tmp_path / "compatible_set.json"
    payload = {
        "event_id": "GW150914",
        "compatible_entries": [{"geometry_id": "geo_001"}],
    }
    cs_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        validate_compatible_set_canonical(payload, cs_path)

    msg = str(excinfo.value)
    assert "schema_version" in msg
    assert cs_path.as_posix() in msg
