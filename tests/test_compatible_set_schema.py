from mvp.schemas import (
    COMPATIBLE_SET_SCHEMA_VERSION,
    extract_compatible_geometry_ids,
    normalize_schema_version,
    validate,
)


def test_compatible_set_accepts_extra_keys() -> None:
    payload = {
        "schema_version": COMPATIBLE_SET_SCHEMA_VERSION,
        "event_id": "GW150914",
        "compatible_geometries": [
            {"geometry_id": "geo_001", "compatible": True},
            {"geometry_id": "geo_002", "compatible": False},
        ],
        "extra": {"debug": True},
    }

    errors = validate("compatible_set", payload)

    assert errors == []
    assert extract_compatible_geometry_ids(payload) == {"geo_001"}


def test_compatible_set_normalizes_int_schema_version() -> None:
    payload = {
        "schema_version": 1,
        "event_id": "GW150914",
        "compatible_geometries": [
            {"id": "geo_001", "compatible": True},
        ],
    }

    normalized = normalize_schema_version("compatible_set", payload)

    assert payload["schema_version"] == 1
    assert normalized["schema_version"] == COMPATIBLE_SET_SCHEMA_VERSION
    assert validate("compatible_set", payload) == []
    assert extract_compatible_geometry_ids(payload) == {"geo_001"}
