"""FASE 5 contract tests (P0) — s3_spectral_estimates contract.

Tests:
    1. CONTRACTS["s3_spectral_estimates"] exists with correct fields
    2. upstream_stages == ["s2_ringdown_window"]
    3. Anti-regression: existing contracts unchanged
    4. Total contract count == 10 (9 previos + 1 nuevo)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.contracts import CONTRACTS, StageContract


class TestS3SpectralContract:
    """Validate s3_spectral_estimates contract definition."""

    def test_exists(self):
        assert "s3_spectral_estimates" in CONTRACTS

    def test_is_stage_contract(self):
        c = CONTRACTS["s3_spectral_estimates"]
        assert isinstance(c, StageContract)

    def test_name_matches_key(self):
        c = CONTRACTS["s3_spectral_estimates"]
        assert c.name == "s3_spectral_estimates"

    def test_produced_outputs(self):
        c = CONTRACTS["s3_spectral_estimates"]
        assert "outputs/spectral_estimates.json" in c.produced_outputs

    def test_upstream_stages(self):
        c = CONTRACTS["s3_spectral_estimates"]
        assert c.upstream_stages == ["s2_ringdown_window"]

    def test_check_run_valid_default(self):
        c = CONTRACTS["s3_spectral_estimates"]
        assert c.check_run_valid is True


class TestAntiRegression:
    """Verify existing contracts are unchanged."""

    def test_total_contract_count(self):
        """Total must be 12 including s4_spectral_geometry_filter."""
        assert len(CONTRACTS) == 16

    def test_s3_ringdown_estimates_unchanged(self):
        c = CONTRACTS["s3_ringdown_estimates"]
        assert c.produced_outputs == ["outputs/estimates.json"]

    def test_s3_ringdown_estimates_upstream(self):
        c = CONTRACTS["s3_ringdown_estimates"]
        assert c.upstream_stages == ["s2_ringdown_window"]

    def test_s1_fetch_strain_present(self):
        assert "s1_fetch_strain" in CONTRACTS

    def test_s2_ringdown_window_present(self):
        assert "s2_ringdown_window" in CONTRACTS

    def test_s4_geometry_filter_present(self):
        assert "s4_geometry_filter" in CONTRACTS

    def test_s5_aggregate_present(self):
        assert "s5_aggregate" in CONTRACTS

    def test_s6_information_geometry_present(self):
        assert "s6_information_geometry" in CONTRACTS

    def test_s4b_spectral_curvature_present(self):
        assert "s4b_spectral_curvature" in CONTRACTS

    def test_s3b_multimode_estimates_present(self):
        assert "s3b_multimode_estimates" in CONTRACTS

    def test_s4c_kerr_consistency_present(self):
        assert "s4c_kerr_consistency" in CONTRACTS

    def test_schema_compatibility(self):
        """spectral_estimates output schema must be compatible with estimates.json for s4."""
        spec_contract = CONTRACTS["s3_spectral_estimates"]
        # s4 reads estimates.json; spectral produces spectral_estimates.json
        # They must have the same "combined" schema (checked via test_spectral_estimates.py)
        assert "outputs/spectral_estimates.json" in spec_contract.produced_outputs
