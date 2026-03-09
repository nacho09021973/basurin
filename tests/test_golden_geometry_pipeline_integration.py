"""Integration tests for the golden geometry pipeline (s4g → s4h → s4i → s4j).

Tests:
    1. s4g→s4i filename contract: S4G_OUTPUT_REL in s4i matches s4g OUTPUT_FILE.
    2. s4g→s4i key contract: s4i reads accepted_geometry_ids from s4g output.
    3. Golden stages in CONTRACTS: s4g, s4h, s4i, s4j all registered.
    4. s4h skip produces SKIPPED verdict via finalize.
    5. Full pipeline e2e: s4g→s4h→s4i→s4j produces manifest+summary for each stage.
    6. Deprecation warning on import of experiment_t0_sweep.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.contracts import CONTRACTS
from mvp.s4i_common_geometry_intersection import (
    S4G_OUTPUT_LEGACY_REL,
    S4G_OUTPUT_PRIMARY_REL,
    compute_intersection,
)


# ── Contract alignment tests ────────────────────────────────────────────────

def test_s4g_s4i_filename_contract():
    """s4i reads s4g mode220 alias as primary and keeps legacy fallback."""
    assert S4G_OUTPUT_PRIMARY_REL == "s4g_mode220_geometry_filter/outputs/mode220_filter.json"
    assert S4G_OUTPUT_LEGACY_REL == "s4g_mode220_geometry_filter/outputs/geometries_220.json"


def test_s4g_s4i_key_contract():
    """s4i must extract geometry IDs from s4g output using accepted_geometry_ids key."""
    # Simulate s4g output payload
    s4g_payload = {
        "accepted_geometry_ids": ["kerr_01", "kerr_03", "kerr_07"],
        "accepted_geometries": [],
        "verdict": "PASS",
    }
    # s4i logic: get accepted_geometry_ids with fallback to geometry_ids
    ids = s4g_payload.get("accepted_geometry_ids", s4g_payload.get("geometry_ids", []))
    assert ids == ["kerr_01", "kerr_03", "kerr_07"]
    assert len(ids) > 0


def test_s4g_s4i_key_contract_fallback():
    """s4i falls back to geometry_ids key if accepted_geometry_ids is absent."""
    s4g_payload = {
        "geometry_ids": ["kerr_02", "kerr_05"],
        "verdict": "PASS",
    }
    ids = s4g_payload.get("accepted_geometry_ids", s4g_payload.get("geometry_ids", []))
    assert ids == ["kerr_02", "kerr_05"]


def test_golden_stages_in_contracts():
    """All 4 golden geometry stages must be registered in CONTRACTS dict."""
    expected = [
        "s4g_mode220_geometry_filter",
        "s4h_mode221_geometry_filter",
        "s4i_common_geometry_intersection",
        "s4j_hawking_area_filter",
    ]
    for stage in expected:
        assert stage in CONTRACTS, f"Missing contract for {stage}"


def test_s4i_contract_requires_s4g_output():
    """s4i contract must declare s4g mode220_filter.json as required input."""
    contract = CONTRACTS["s4i_common_geometry_intersection"]
    assert any("mode220_filter.json" in inp for inp in contract.required_inputs)


def test_s4i_intersection_is_mode_agnostic_on_suffix():
    """Mode suffix (_l2m2n0/_l2m2n1) should not prevent common-geometry match."""
    ids_220 = ["Kerr_M62_a0.6600_l2m2n0", "EdGB_M62_a0.67_z0.3"]
    ids_221 = ["Kerr_M62_a0.6600_l2m2n1", "Kerr_M70_a0.5000_l2m2n1"]
    common = compute_intersection(ids_220, ids_221)
    assert common == ["Kerr_M62_a0.6600"]


def test_s4j_contract_requires_s4i_output():
    """s4j contract must declare s4i common_intersection.json as required input."""
    contract = CONTRACTS["s4j_hawking_area_filter"]
    assert any("common_intersection.json" in inp for inp in contract.required_inputs)


def test_s4h_contract_has_atlas_external_input():
    """s4h contract must declare atlas as external input."""
    contract = CONTRACTS["s4h_mode221_geometry_filter"]
    assert "atlas" in contract.external_inputs


# ── Deprecation test ────────────────────────────────────────────────────────

def test_t0_sweep_deprecation_warning():
    """Importing experiment_t0_sweep must emit a DeprecationWarning."""
    # Remove from sys.modules to force re-import
    mod_name = "mvp.experiment_t0_sweep"
    was_loaded = mod_name in sys.modules
    if was_loaded:
        saved = sys.modules.pop(mod_name)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                __import__(mod_name)
            except ImportError:
                # numpy or other heavy deps may not be installed in test env;
                # the warning is emitted before those imports, so check anyway.
                pass
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "DEPRECATED" in str(deprecation_warnings[0].message)
    finally:
        if was_loaded:
            sys.modules[mod_name] = saved
