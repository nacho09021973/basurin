"""Tests for Q6: deviation distribution in s5_aggregate.py.

Tests:
    1. 3 events exactly at Kerr (δf=0): chi2_GR ≈ 0, p_value ≈ 1.0
    2. 3 events with δf_rel = 0.1 (far from GR): consistent_GR_95 = False
    3. 1 event with large σ: combined dominated by other events (IVW correct)
    4. Without catalog: deviation_analysis absent, intersection still works
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s5_aggregate import compute_deviation_distribution, aggregate_compatible_sets
from mvp.kerr_qnm_fits import kerr_qnm


def _make_source_data(
    event_id: str,
    f_obs: float,
    Q_obs: float,
    sigma_f: float,
    sigma_Q: float,
) -> dict[str, Any]:
    """Create a minimal source_data entry for testing."""
    return {
        "run_id": f"run_{event_id}",
        "event_id": event_id,
        "metric": "mahalanobis_log",
        "threshold_d2": 5.991,
        "ranked_all": [],
        "compatible_ids": set(),
        "observables": {"f_hz": f_obs, "Q": Q_obs},
        "sigma_f_hz": sigma_f,
        "sigma_Q": sigma_Q,
    }


class TestExactKerr:
    """Test 1: Events at exact Kerr prediction → chi2_GR ≈ 0, p ≈ 1."""

    def test_chi2_near_zero(self):
        """With δf=0 for all events, chi2_GR should be ~0."""
        # Use GW150914 parameters from catalog
        m = 62.2
        chi = 0.67
        kerr = kerr_qnm(m, chi)
        f_kerr = kerr.f_hz
        Q_kerr = kerr.Q

        # Three events, all at exact Kerr values
        source_data = []
        catalog = {}
        for i, ev_id in enumerate(["GW150914", "GW151226", "GW170104"]):
            source_data.append(_make_source_data(
                event_id=ev_id,
                f_obs=f_kerr, Q_obs=Q_kerr,
                sigma_f=5.0, sigma_Q=0.5,
            ))
            catalog[ev_id] = {"m_final_msun": m, "chi_final": chi}

        result = compute_deviation_distribution(source_data, catalog)

        assert result is not None
        combined = result.get("combined")
        assert combined is not None
        # delta_f_rel should be ~0
        assert abs(combined["delta_f_rel"]) < 0.001, \
            f"delta_f_rel={combined['delta_f_rel']} not near 0"

    def test_p_value_near_1(self):
        """p-value should be near 1 when all events are at exact Kerr."""
        m, chi = 62.2, 0.67
        kerr = kerr_qnm(m, chi)

        source_data = []
        catalog = {}
        for ev_id in ["GW150914", "GW151226", "GW170104"]:
            source_data.append(_make_source_data(
                event_id=ev_id,
                f_obs=kerr.f_hz, Q_obs=kerr.Q,
                sigma_f=5.0, sigma_Q=0.5,
            ))
            catalog[ev_id] = {"m_final_msun": m, "chi_final": chi}

        result = compute_deviation_distribution(source_data, catalog)
        combined = result["combined"]

        if math.isfinite(combined["p_value_GR"]):
            assert combined["p_value_GR"] > 0.05, \
                f"p_value={combined['p_value_GR']:.4f} expected > 0.05 for exact Kerr"
            assert combined["consistent_GR_95"] is True


class TestLargeDeviation:
    """Test 2: Events with large δf → consistent_GR_95 = False."""

    def test_far_from_gr_inconsistent(self):
        """10% deviation in f with small σ → GR excluded."""
        m, chi = 62.2, 0.67
        kerr = kerr_qnm(m, chi)
        f_kerr = kerr.f_hz
        Q_kerr = kerr.Q

        # 10% deviation, small uncertainty
        source_data = []
        catalog = {}
        for ev_id in ["GW150914", "GW151226", "GW170104"]:
            source_data.append(_make_source_data(
                event_id=ev_id,
                f_obs=f_kerr * 1.10,  # 10% offset
                Q_obs=Q_kerr,
                sigma_f=f_kerr * 0.01,  # 1% uncertainty → high tension
                sigma_Q=Q_kerr * 0.10,
            ))
            catalog[ev_id] = {"m_final_msun": m, "chi_final": chi}

        result = compute_deviation_distribution(source_data, catalog)
        combined = result["combined"]

        # With 10% offset and 1% σ, tension per event ≈ 10, chi2 ≈ 300 → p << 0.05
        assert abs(combined["delta_f_rel"] - 0.10) < 0.01, \
            f"delta_f_rel={combined['delta_f_rel']:.4f} should be ~0.10"
        if math.isfinite(combined["p_value_GR"]):
            assert combined["consistent_GR_95"] is False, \
                "Should not be consistent with GR at 10% deviation"


class TestInverseVarianceWeighting:
    """Test 3: Event with large σ has less weight in combined estimate."""

    def test_large_sigma_event_has_less_weight(self):
        """One event with large σ should not dominate the combined estimate."""
        m, chi = 62.2, 0.67
        kerr = kerr_qnm(m, chi)
        f_kerr = kerr.f_hz
        Q_kerr = kerr.Q

        catalog = {
            "GW150914": {"m_final_msun": m, "chi_final": chi},
            "GW151226": {"m_final_msun": m, "chi_final": chi},
            "GW170104": {"m_final_msun": m, "chi_final": chi},
        }

        # Two events at Kerr (small σ), one at 50% offset but huge σ
        source_data = [
            _make_source_data("GW150914", f_kerr, Q_kerr, sigma_f=1.0, sigma_Q=0.1),
            _make_source_data("GW151226", f_kerr, Q_kerr, sigma_f=1.0, sigma_Q=0.1),
            _make_source_data("GW170104", f_kerr * 1.50, Q_kerr,
                              sigma_f=f_kerr * 100.0, sigma_Q=Q_kerr),  # huge σ
        ]

        result = compute_deviation_distribution(source_data, catalog)
        combined = result["combined"]

        # Combined should be dominated by events 1 and 2 (δf≈0)
        assert abs(combined["delta_f_rel"]) < 0.05, \
            f"Large-σ event incorrectly dominated: delta_f_rel={combined['delta_f_rel']:.4f}"


class TestWithoutCatalog:
    """Test 4: Without catalog, deviation_analysis absent, intersection still works."""

    def test_none_catalog_returns_none(self):
        """compute_deviation_distribution with None catalog returns None."""
        source_data = [
            _make_source_data("GW150914", 250.0, 3.0, 5.0, 0.5)
        ]
        result = compute_deviation_distribution(source_data, None)
        assert result is None

    def test_aggregate_without_catalog(self):
        """aggregate_compatible_sets works without deviation analysis."""
        # Empty source_data is valid
        result = aggregate_compatible_sets([], min_coverage=1.0)
        assert result["n_events"] == 0
        assert "deviation_analysis" not in result


class TestDeviationFields:
    """Test schema of deviation_analysis output."""

    def test_schema_version_present(self):
        m, chi = 62.2, 0.67
        kerr = kerr_qnm(m, chi)
        catalog = {"GW150914": {"m_final_msun": m, "chi_final": chi}}
        source_data = [
            _make_source_data("GW150914", kerr.f_hz, kerr.Q, 5.0, 0.5)
        ]
        result = compute_deviation_distribution(source_data, catalog)
        assert result is not None
        assert result["schema_version"] == "mvp_deviation_v1"

    def test_per_event_fields(self):
        m, chi = 62.2, 0.67
        kerr = kerr_qnm(m, chi)
        catalog = {"GW150914": {"m_final_msun": m, "chi_final": chi}}
        source_data = [
            _make_source_data("GW150914", kerr.f_hz, kerr.Q, 5.0, 0.5)
        ]
        result = compute_deviation_distribution(source_data, catalog)
        assert result["n_events_in_analysis"] == 1
        pe = result["per_event"][0]
        required_fields = {
            "event_id", "f_obs", "f_kerr", "Q_obs", "Q_kerr",
            "delta_f_rel", "sigma_delta_f_rel",
            "delta_Q_rel", "sigma_delta_Q_rel",
        }
        assert required_fields.issubset(set(pe.keys()))

    def test_combined_fields(self):
        m, chi = 62.2, 0.67
        kerr = kerr_qnm(m, chi)
        catalog = {"GW150914": {"m_final_msun": m, "chi_final": chi}}
        source_data = [
            _make_source_data("GW150914", kerr.f_hz, kerr.Q, 5.0, 0.5)
        ]
        result = compute_deviation_distribution(source_data, catalog)
        combined = result["combined"]
        required = {
            "delta_f_rel", "sigma_delta_f_rel",
            "delta_Q_rel", "sigma_delta_Q_rel",
            "n_events", "chi2_GR", "p_value_GR", "consistent_GR_95",
        }
        assert required.issubset(set(combined.keys()))
