"""Regression tests for adaptive dt_start based on remnant mass."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvp.s2_ringdown_window import estimate_dt_start_from_mass


# ---------------------------------------------------------------------------
# Test 1: estimate_dt_start_from_mass
# ---------------------------------------------------------------------------

class TestEstimateDtStartFromMass:
    """Verify the adaptive dt_start formula against calibration points."""

    def test_gw150914_above_floor(self):
        # GW150914: M_f=63.1, z=0.09 → should be >= 0.003 (floor)
        dt = estimate_dt_start_from_mass(63.1, 0.09)
        assert dt >= 0.003

    def test_gw170814_above_threshold(self):
        # GW170814: M_f=53.2, z=0.12 → should be >= 0.010
        dt = estimate_dt_start_from_mass(53.2, 0.12)
        assert dt >= 0.010

    def test_high_mass_capped(self):
        # High mass: M_f=100, z=0.1 → should be capped at 0.030
        dt = estimate_dt_start_from_mass(100.0, 0.1)
        assert dt <= 0.030

    def test_low_mass_floor(self):
        # Low mass: M_f=10, z=0.01 → should hit floor at 0.003
        dt = estimate_dt_start_from_mass(10.0, 0.01)
        assert dt == 0.003

    def test_gw170814_calibration_anchor(self):
        # GW170814 is the calibration anchor; predicted should be close to 0.015
        dt = estimate_dt_start_from_mass(53.2, 0.12)
        assert abs(dt - 0.015) < 0.002, f"Expected ~0.015, got {dt}"

    def test_monotonic_in_mass(self):
        # Higher detector-frame mass → higher dt_start (within bounds)
        dt_low = estimate_dt_start_from_mass(30.0, 0.1)
        dt_high = estimate_dt_start_from_mass(80.0, 0.1)
        assert dt_high >= dt_low

    def test_no_numpy_dependency(self):
        # Verify the function uses only stdlib
        import inspect
        source = inspect.getsource(estimate_dt_start_from_mass)
        assert "numpy" not in source
        assert "np." not in source


# ---------------------------------------------------------------------------
# Test 2: window_catalog contains GW150914 and GW170814 with fields
# ---------------------------------------------------------------------------

class TestWindowCatalog:
    """Verify window_catalog_v1.json has calibrated entries."""

    @pytest.fixture()
    def catalog(self):
        path = Path(__file__).resolve().parents[1] / "mvp" / "assets" / "window_catalog_v1.json"
        with open(path) as f:
            return json.load(f)

    def test_gw150914_present(self, catalog):
        assert "GW150914" in catalog

    def test_gw150914_calibration_pass(self, catalog):
        assert catalog["GW150914"]["calibration_status"] == "PASS"

    def test_gw150914_dt_start(self, catalog):
        assert catalog["GW150914"]["dt_start_s"] == 0.003

    def test_gw150914_mass(self, catalog):
        assert catalog["GW150914"]["M_f_source_msun"] == 63.1

    def test_gw170814_present(self, catalog):
        assert "GW170814" in catalog

    def test_gw170814_dt_start(self, catalog):
        assert catalog["GW170814"]["dt_start_s"] == 0.015

    def test_gw170814_calibration_pass(self, catalog):
        assert catalog["GW170814"]["calibration_status"] == "PASS"

    def test_gw170814_calibration_criteria(self, catalog):
        crit = catalog["GW170814"]["calibration_criteria"]
        assert abs(crit["f_hz"] - 323.8) < 0.1
        assert abs(crit["Q"] - 3.048) < 0.01


# ---------------------------------------------------------------------------
# Test 3: event_mass_catalog_v1.json
# ---------------------------------------------------------------------------

class TestEventMassCatalog:
    """Verify event_mass_catalog_v1.json has sufficient coverage."""

    @pytest.fixture()
    def catalog(self):
        path = Path(__file__).resolve().parents[1] / "mvp" / "assets" / "event_mass_catalog_v1.json"
        with open(path) as f:
            return json.load(f)

    def test_minimum_event_count(self, catalog):
        assert len(catalog["events"]) >= 40

    def test_schema_version(self, catalog):
        assert catalog["schema_version"] == "event_mass_catalog_v1"

    def test_gw150914_present(self, catalog):
        assert "GW150914" in catalog["events"]

    def test_gw190521_present(self, catalog):
        assert "GW190521" in catalog["events"]

    def test_gw150914_mass(self, catalog):
        assert catalog["events"]["GW150914"]["M_f_source_msun"] == 63.1

    def test_gw150914_redshift(self, catalog):
        assert catalog["events"]["GW150914"]["redshift"] == 0.09

    def test_gw150914_chi_f(self, catalog):
        assert catalog["events"]["GW150914"]["chi_f"] == 0.69

    def test_events_have_required_fields(self, catalog):
        for event_id, ev in catalog["events"].items():
            assert "M_f_source_msun" in ev, f"{event_id} missing M_f_source_msun"
            assert "redshift" in ev, f"{event_id} missing redshift"
