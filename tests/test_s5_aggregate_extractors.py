from mvp.s5_aggregate import _extract_compatible_geometry_ids


def test_extract_ids_from_legacy_compatible_geometries():
    payload = {
        "compatible_geometries": [
            {"geometry_id": "geo_005", "compatible": True},
            {"geometry_id": "geo_006", "compatible": False},
            {"id": "geo_007", "compatible": True},
        ]
    }

    assert _extract_compatible_geometry_ids(payload) == {"geo_005", "geo_007"}


def test_extract_ids_from_compatible_entries_and_ids_fallbacks():
    payload_entries = {
        "compatible_entries": [
            {"id": "geo_001"},
            {"geometry_id": "geo_002"},
            "geo_003",
        ]
    }
    payload_ids = {"compatible_ids": [5, "geo_010"]}

    assert _extract_compatible_geometry_ids(payload_entries) == {"geo_001", "geo_002", "geo_003"}
    assert _extract_compatible_geometry_ids(payload_ids) == {"5", "geo_010"}
