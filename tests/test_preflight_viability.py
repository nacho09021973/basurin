"""Tests for mvp.preflight_viability — Fisher-based informativity prediction.

Tests cover:
  1. Consistency of preflight with known GW150914 failure
  2. Prevention of sweeps outside viable domain
  3. Coherence between preflight, t0_max, and multimode gate
  4. Catalog viability table computation
  5. Alpha calibration
"""
from __future__ import annotations

import math
import pytest

from mvp.preflight_viability import (
    snr_fraction_eta,
    rho_effective,
    rel_iqr_predicted,
    t0_max_informative,
    T_min_resolution,
    band_min_hz,
    assess_mode_viability,
    preflight_viability,
    catalog_viability_table,
    calibrate_alpha_from_runs,
    viable_t0_domain,
    DEFAULT_ALPHA_SAFETY,
    VIABLE,
    MARGINAL,
    INVIABLE,
)
from mvp.kerr_qnm_fits import kerr_qnm


# ---------------------------------------------------------------------------
# GW150914 reference values
# ---------------------------------------------------------------------------
GW150914_M = 62.2
GW150914_CHI = 0.67
GW150914_RHO_TOTAL = 8.0  # approximate ringdown SNR


class TestSnrFractionEta:
    def test_t0_zero_long_window(self):
        """At t0=0 with T >> tau, eta should be ~1."""
        tau = 0.00545
        eta = snr_fraction_eta(0.0, 10.0, tau)
        assert eta == pytest.approx(1.0, abs=1e-6)

    def test_t0_large_eta_near_zero(self):
        """At t0 >> tau, eta should be near zero."""
        tau = 0.00545
        eta = snr_fraction_eta(0.023, 0.06, tau)
        assert eta < 0.001  # less than 0.1%

    def test_gw150914_current_config(self):
        """Reproduce the 0.022% figure from the methodology doc."""
        tau = 0.00545
        eta = snr_fraction_eta(0.023, 0.06, tau)
        assert eta == pytest.approx(2.2e-4, rel=0.3)

    def test_degenerate_inputs(self):
        assert snr_fraction_eta(-1.0, 0.06, 0.005) == 0.0
        assert snr_fraction_eta(0.0, 0.0, 0.005) == 0.0
        assert snr_fraction_eta(0.0, 0.06, 0.0) == 0.0

    def test_monotonic_in_t0(self):
        """eta should decrease as t0 increases."""
        tau = 0.00545
        T = 0.06
        etas = [snr_fraction_eta(t0 / 1000, T, tau) for t0 in range(0, 30, 5)]
        for i in range(len(etas) - 1):
            assert etas[i] >= etas[i + 1]


class TestRhoEffective:
    def test_gw150914_current(self):
        """rho_eff at current config should be << 1."""
        tau = 0.00545
        rho_eff = rho_effective(0.023, 0.06, tau, GW150914_RHO_TOTAL)
        assert rho_eff < 0.2

    def test_gw150914_early_t0(self):
        """rho_eff at t0=3ms should be much higher."""
        tau = 0.00545
        rho_eff = rho_effective(0.003, 0.06, tau, GW150914_RHO_TOTAL)
        assert rho_eff > 2.0


class TestRelIqrPredicted:
    def test_high_qrho_gives_low_iqr(self):
        iqr = rel_iqr_predicted(4.3, 5.0, 2.5)
        assert iqr < 0.1

    def test_low_qrho_gives_high_iqr(self):
        iqr = rel_iqr_predicted(4.3, 0.12, 2.5)
        assert iqr > 1.0

    def test_degenerate(self):
        assert rel_iqr_predicted(0.0, 1.0) == float("inf")
        assert rel_iqr_predicted(1.0, 0.0) == float("inf")


class TestT0MaxInformative:
    def test_gw150914_t0_max(self):
        """t0_max for GW150914 (2,2,0) should be ~11ms with alpha=2 (Berti fits)."""
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        t0_max = t0_max_informative(qnm.tau_s, qnm.Q, GW150914_RHO_TOTAL, alpha_safety=2.0)
        assert 0.008 < t0_max < 0.018  # ~11ms with actual Berti fits

    def test_current_t0_exceeds_max(self):
        """Verify that the current t0=23ms is OUTSIDE the viable domain."""
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        t0_max = t0_max_informative(qnm.tau_s, qnm.Q, GW150914_RHO_TOTAL, alpha_safety=2.5)
        assert 0.023 > t0_max  # 23ms exceeds the max

    def test_very_low_snr_gives_near_zero(self):
        """With very low SNR, t0_max should be near zero (unconstrainable)."""
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        t0_max = t0_max_informative(qnm.tau_s, qnm.Q, 0.5, alpha_safety=2.5)
        assert t0_max < 0.001  # sub-millisecond, practically useless


class TestTMinResolution:
    def test_220(self):
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        T_min = T_min_resolution(qnm.tau_s)
        assert 0.020 < T_min < 0.030  # ~23ms with Berti fits (τ≈3.7ms)


class TestBandMinHz:
    def test_220_band(self):
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        low, high = band_min_hz(qnm.f_hz, qnm.Q)
        assert low < 150
        assert high > 350


class TestAssessModeViability:
    def test_gw150914_current_inviable(self):
        """Current GW150914 config should be INVIABLE."""
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        result = assess_mode_viability(
            f_hz=qnm.f_hz, tau_s=qnm.tau_s, Q=qnm.Q,
            rho_total=GW150914_RHO_TOTAL,
            t0_s=0.023, T_s=0.06,
        )
        assert result["verdict"] == INVIABLE
        assert result["eta"] < 0.001

    def test_gw150914_early_t0_viable(self):
        """GW150914 at t0=5ms should be VIABLE or MARGINAL."""
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        result = assess_mode_viability(
            f_hz=qnm.f_hz, tau_s=qnm.tau_s, Q=qnm.Q,
            rho_total=GW150914_RHO_TOTAL,
            t0_s=0.005, T_s=0.06,
        )
        assert result["verdict"] in (VIABLE, MARGINAL)
        assert result["rho_eff"] > 1.0


class TestPreflightViability:
    def test_gw150914_current(self):
        result = preflight_viability(
            event_id="GW150914",
            m_final_msun=GW150914_M, chi_final=GW150914_CHI,
            rho_total=GW150914_RHO_TOTAL,
            t0_s=0.023, T_s=0.06,
        )
        assert result["schema_version"] == "preflight_viability_v1"
        assert result["overall_verdict"] == INVIABLE
        assert "220" in result["modes"]
        assert "221" in result["modes"]
        # Should recommend a better config
        rec = result["recommended_config"]
        if rec["t0_s"] is not None:
            assert rec["t0_s"] < 0.023

    def test_contains_required_fields(self):
        result = preflight_viability(
            event_id="GW150914",
            m_final_msun=GW150914_M, chi_final=GW150914_CHI,
            rho_total=GW150914_RHO_TOTAL,
            t0_s=0.005, T_s=0.06,
        )
        assert "modes" in result
        assert "overall_verdict" in result
        assert "alpha_safety" in result
        mode_220 = result["modes"]["220"]
        for key in ("eta", "rho_eff", "Q_x_rho_eff", "rel_iqr_predicted",
                     "t0_max_s", "verdict", "qnm_params"):
            assert key in mode_220


class TestCatalogViabilityTable:
    def test_basic(self):
        events = {
            "GW150914": {"m_final_msun": 62.2, "chi_final": 0.67, "snr_network": 24.4},
            "GW151226": {"m_final_msun": 20.8, "chi_final": 0.74, "snr_network": 13.0},
        }
        table = catalog_viability_table(events)
        assert table["schema_version"] == "catalog_viability_table_v1"
        assert len(table["events"]) == 2
        assert "summary" in table
        assert table["summary"]["n_events"] == 2

    def test_high_snr_event_viable(self):
        events = {
            "TEST_HIGH_SNR": {"m_final_msun": 62.2, "chi_final": 0.67, "snr_network": 100.0},
        }
        table = catalog_viability_table(events)
        assert table["events"][0]["mode_220"]["viable"] is True


class TestViableT0Domain:
    def test_gw150914_domain(self):
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        domain = viable_t0_domain(
            tau_s=qnm.tau_s, Q=qnm.Q,
            rho_total=GW150914_RHO_TOTAL, T_s=0.06,
            alpha_safety=2.0,
        )
        assert not domain["domain_empty"]
        assert domain["t0_max_s"] > domain["t0_min_s"]
        assert len(domain["recommended_grid_ms"]) > 0

    def test_prevents_sweep_outside_viable(self):
        """Grid points should all be within viable domain."""
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        domain = viable_t0_domain(
            tau_s=qnm.tau_s, Q=qnm.Q,
            rho_total=GW150914_RHO_TOTAL, T_s=0.06,
        )
        if not domain["domain_empty"]:
            for t0_ms in domain["recommended_grid_ms"]:
                assert t0_ms / 1000 >= domain["t0_min_s"]
                assert t0_ms / 1000 <= domain["t0_max_s"] + 1e-3

    def test_low_snr_domain_empty(self):
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        domain = viable_t0_domain(
            tau_s=qnm.tau_s, Q=qnm.Q,
            rho_total=0.5, T_s=0.06,
        )
        assert domain["domain_empty"]
        assert domain["recommended_grid_ms"] == []


class TestCalibrateAlpha:
    def test_with_observations(self):
        obs = [
            {"rel_iqr_f220_observed": 0.833, "Q_220": 4.3, "rho_eff_estimated": 0.12},
            {"rel_iqr_f220_observed": 0.3, "Q_220": 4.3, "rho_eff_estimated": 2.0},
        ]
        result = calibrate_alpha_from_runs(obs)
        assert result["calibrated"] is True
        assert result["n_observations"] == 2
        assert result["alpha_safety"] > 0

    def test_no_observations(self):
        result = calibrate_alpha_from_runs([])
        assert result["calibrated"] is False
        assert result["alpha_safety"] == DEFAULT_ALPHA_SAFETY


class TestCoherenceBestPointS2S3b:
    """Verify that preflight predictions are coherent with multimode gate logic."""

    def test_viable_implies_possible_informative(self):
        """If preflight says VIABLE, Q*rho_eff should exceed the CR threshold."""
        qnm = kerr_qnm(GW150914_M, GW150914_CHI, (2, 2, 0))
        result = preflight_viability(
            event_id="GW150914",
            m_final_msun=GW150914_M, chi_final=GW150914_CHI,
            rho_total=GW150914_RHO_TOTAL,
            t0_s=0.005, T_s=0.06,
            alpha_safety=2.0,
        )
        mode_220 = result["modes"]["220"]
        if mode_220["verdict"] == VIABLE:
            assert mode_220["Q_x_rho_eff"] > 0.61 * 2.0

    def test_inviable_implies_gate_would_fail(self):
        """If preflight says INVIABLE, predicted rel_iqr should exceed gate threshold."""
        result = preflight_viability(
            event_id="GW150914",
            m_final_msun=GW150914_M, chi_final=GW150914_CHI,
            rho_total=GW150914_RHO_TOTAL,
            t0_s=0.023, T_s=0.06,
        )
        mode_220 = result["modes"]["220"]
        assert mode_220["verdict"] == INVIABLE
        assert mode_220["rel_iqr_predicted"] > 0.5
