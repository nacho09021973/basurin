from mvp.s5_aggregate import _detect_compatible_set_schema, _extract_compatible_geometry_ids


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


def test_detect_schema_accepts_historic_v1_string_and_extracts_ids():
    payload = {
        "atlas_posterior": {},
        "bits_excluded": 0.0,
        "bits_kl": 0.0,
        "chi2_fixed_theta": 0.0,
        "compatible_geometries": [
            {"geometry_id": "geo_101", "compatible": True},
            {"id": "geo_102", "compatible": True},
        ],
        "covariance_logspace": [[1.0, 0.0], [0.0, 1.0]],
        "d2_min": 0.1,
        "distance": 0.2,
        "epsilon": 0.0,
        "event_id": "GW150914",
        "likelihood_stats": {},
        "metric": "mahalanobis_log",
        "metric_params": {},
        "n_atlas": 2,
        "n_compatible": 2,
        "observables": {"f_hz": 250.0, "Q": 8.0},
        "ranked_all": [{"geometry_id": "geo_101"}, {"geometry_id": "geo_102"}],
        "run_id": "run_x",
        "schema_version": "mvp_compatible_set_v1",
        "threshold_d2": 5.99,
    }

    detected, normalized = _detect_compatible_set_schema(payload)

    assert detected == "mvp_compatible_set_v1"
    assert normalized == "compatible_set_v1_canonical"
    assert _extract_compatible_geometry_ids(payload) == {"geo_101", "geo_102"}


def test_detect_schema_accepts_legacy_with_extra_keys_and_minimum_required_subset():
    payload = {
        "schema_version": "mvp_compatible_set_v1",
        "event_id": "GWTEST",
        "compatible_geometries": [
            {"geometry_id": "geo_201", "compatible": True},
            {"geometry_id": "geo_202", "compatible": False},
            {"id": "geo_203", "compatible": True},
        ],
        "diagnostic_extra": {"note": "legacy payloads can include additional keys"},
    }

    detected, normalized = _detect_compatible_set_schema(payload)

    assert detected == "mvp_compatible_set_v1"
    assert normalized == "compatible_set_v1_canonical"
    assert _extract_compatible_geometry_ids(payload) == {"geo_201", "geo_203"}


def test_detect_schema_normalizes_int_schema_version_for_compatible_set():
    payload = {
        "schema_version": 1,
        "event_id": "GWTEST",
        "compatible_geometries": [
            {"geometry_id": "geo_301", "compatible": True},
        ],
    }

    detected, normalized = _detect_compatible_set_schema(payload)

    assert detected == 1
    assert normalized == "compatible_set_v1"
    assert _extract_compatible_geometry_ids(payload) == {"geo_301"}
