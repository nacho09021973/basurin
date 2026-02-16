"""FASE 4 contract tests (P0) — s3b_multimode_estimates & s4c_kerr_consistency.

Tests:
    1. Both new contracts exist in CONTRACTS and have correct fields.
    2. Anti-regression: existing contracts unchanged.
    3. Total contract count == 9 (7 prior + 2 new).
    4. DAG integrity: new upstream references are valid, no cycles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.contracts import CONTRACTS, StageContract


# ── Test 1: New contracts exist with correct fields ──────────────────────


class TestS3bMultimodeContract:
    """Validate s3b_multimode_estimates contract definition."""

    def test_exists(self):
        assert "s3b_multimode_estimates" in CONTRACTS

    def test_is_stage_contract(self):
        c = CONTRACTS["s3b_multimode_estimates"]
        assert isinstance(c, StageContract)

    def test_name_matches_key(self):
        c = CONTRACTS["s3b_multimode_estimates"]
        assert c.name == "s3b_multimode_estimates"

    def test_produced_outputs(self):
        c = CONTRACTS["s3b_multimode_estimates"]
        assert c.produced_outputs == ["outputs/multimode_estimates.json"]

    def test_upstream_stages(self):
        c = CONTRACTS["s3b_multimode_estimates"]
        assert c.upstream_stages == ["s2_ringdown_window"]

    def test_required_inputs_empty_dynamic(self):
        """s3b discovers detector files at runtime (like s3)."""
        c = CONTRACTS["s3b_multimode_estimates"]
        assert c.required_inputs == []

    def test_check_run_valid_default(self):
        """s3b checks RUN_VALID (default=True)."""
        c = CONTRACTS["s3b_multimode_estimates"]
        assert c.check_run_valid is True


class TestS4cKerrConsistencyContract:
    """Validate s4c_kerr_consistency contract definition."""

    def test_exists(self):
        assert "s4c_kerr_consistency" in CONTRACTS

    def test_is_stage_contract(self):
        c = CONTRACTS["s4c_kerr_consistency"]
        assert isinstance(c, StageContract)

    def test_name_matches_key(self):
        c = CONTRACTS["s4c_kerr_consistency"]
        assert c.name == "s4c_kerr_consistency"

    def test_produced_outputs(self):
        c = CONTRACTS["s4c_kerr_consistency"]
        assert c.produced_outputs == ["outputs/kerr_consistency.json"]

    def test_required_inputs(self):
        c = CONTRACTS["s4c_kerr_consistency"]
        assert c.required_inputs == [
            "s3_ringdown_estimates/outputs/estimates.json",
            "s3b_multimode_estimates/outputs/multimode_estimates.json",
        ]

    def test_upstream_stages(self):
        c = CONTRACTS["s4c_kerr_consistency"]
        assert c.upstream_stages == [
            "s3_ringdown_estimates",
            "s3b_multimode_estimates",
        ]

    def test_check_run_valid_default(self):
        """s4c checks RUN_VALID (default=True)."""
        c = CONTRACTS["s4c_kerr_consistency"]
        assert c.check_run_valid is True


# ── Test 2: Anti-regression — existing contracts unchanged ───────────────


class TestAntiRegression:
    """Ensure pre-existing contracts are NOT modified by FASE 4 additions."""

    def test_s4_geometry_filter_upstream(self):
        assert CONTRACTS["s4_geometry_filter"].upstream_stages == [
            "s3_ringdown_estimates"
        ]

    def test_s4_geometry_filter_inputs(self):
        assert CONTRACTS["s4_geometry_filter"].required_inputs == [
            "s3_ringdown_estimates/outputs/estimates.json",
        ]

    def test_s4_geometry_filter_outputs(self):
        assert CONTRACTS["s4_geometry_filter"].produced_outputs == [
            "outputs/compatible_set.json",
        ]

    def test_s3_ringdown_estimates_upstream(self):
        assert CONTRACTS["s3_ringdown_estimates"].upstream_stages == [
            "s2_ringdown_window"
        ]

    def test_s3_ringdown_estimates_outputs(self):
        assert CONTRACTS["s3_ringdown_estimates"].produced_outputs == [
            "outputs/estimates.json",
        ]

    def test_s1_fetch_strain_no_upstream(self):
        assert CONTRACTS["s1_fetch_strain"].upstream_stages == []

    def test_s5_aggregate_no_upstream(self):
        assert CONTRACTS["s5_aggregate"].upstream_stages == []

    def test_s4b_spectral_curvature_upstream(self):
        assert CONTRACTS["s4b_spectral_curvature"].upstream_stages == [
            "s3_ringdown_estimates"
        ]

    def test_s6_information_geometry_upstream(self):
        assert CONTRACTS["s6_information_geometry"].upstream_stages == [
            "s3_ringdown_estimates",
            "s4_geometry_filter",
        ]


# ── Test 3: Total contract count ─────────────────────────────────────────


class TestContractCount:
    def test_total_contracts_is_9(self):
        """7 prior stages + 2 new FASE 4 stages = 9."""
        assert len(CONTRACTS) == 9


# ── Test 4: DAG integrity with new stages ────────────────────────────────


class TestDAGIntegrity:
    def test_new_upstream_references_valid(self):
        """All upstream_stages referenced by new contracts exist in CONTRACTS."""
        for name in ("s3b_multimode_estimates", "s4c_kerr_consistency"):
            c = CONTRACTS[name]
            for upstream in c.upstream_stages:
                assert upstream in CONTRACTS, (
                    f"{name} references unknown upstream '{upstream}'"
                )

    def test_no_cycles_with_new_stages(self):
        """Full DAG (including new stages) has no circular dependencies."""
        visited: set[str] = set()

        def _check(name: str, path: list[str]) -> None:
            if name in path:
                raise AssertionError(
                    f"Circular dependency: {' -> '.join(path + [name])}"
                )
            if name in visited:
                return
            for upstream in CONTRACTS[name].upstream_stages:
                _check(upstream, path + [name])
            visited.add(name)

        for name in CONTRACTS:
            _check(name, [])

    def test_s4c_depends_on_s3b(self):
        """s4c_kerr_consistency depends on s3b_multimode_estimates."""
        assert "s3b_multimode_estimates" in CONTRACTS["s4c_kerr_consistency"].upstream_stages

    def test_s3b_parallel_to_s3(self):
        """s3b_multimode_estimates shares the same upstream as s3."""
        assert (
            CONTRACTS["s3b_multimode_estimates"].upstream_stages
            == CONTRACTS["s3_ringdown_estimates"].upstream_stages
        )
