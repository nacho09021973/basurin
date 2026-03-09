"""Tests for mvp/golden_geometry_spec.py — shared golden geometry specification."""
from __future__ import annotations

import pytest

from mvp.golden_geometry_spec import (
    DEFAULT_AREA_TOLERANCE,
    DEFAULT_MODE_CHI2_THRESHOLD_90,
    DEFAULT_MODE_CHI2_THRESHOLD_99,
    GOLDEN_GEOMETRY_SPEC_VERSION,
    MODE_220,
    MODE_221,
    ROBUST_UNIQUE_MIN_SUPPORT_FRACTION,
    VERDICT_NO_DATA,
    VERDICT_NOT_UNIQUE,
    VERDICT_PASS,
    VERDICT_REJECT,
    VERDICT_ROBUST_UNIQUE,
    VERDICT_UNSTABLE_UNIQUE,
    build_common_geometries_payload,
    build_golden_geometries_payload,
    build_mode_filter_payload,
    build_population_consensus_payload,
    build_single_event_robustness_payload,
    chi2_mode,
    delta_area,
    exact_intersection_geometry_ids,
    passes_area_law,
    passes_mode_threshold,
    rank_geometries_by_support,
    robust_unique_verdict,
    singleton_geometry_id,
    support_count_geometry_ids,
)


# ---------------------------------------------------------------------------
# 1. chi2_mode — zero when observed equals predicted
# ---------------------------------------------------------------------------

def test_chi2_mode_zero_when_observed_equals_predicted():
    """chi2 must be exactly 0 when observed == predicted."""
    result = chi2_mode(
        obs_f=250.0, obs_tau=0.01,
        pred_f=250.0, pred_tau=0.01,
        sigma_f=5.0, sigma_tau=0.001,
    )
    assert result == 0.0


# ---------------------------------------------------------------------------
# 2. chi2_mode — mathematical behaviour; no hidden sigma floor
# ---------------------------------------------------------------------------

def test_chi2_mode_uses_sigma_floor_via_caller_not_spec():
    """chi2 uses caller-supplied sigmas directly; verify formula is correct.

    The spec does NOT apply any sigma floor internally.  Callers that need a
    floor must supply one before calling.
    This test verifies the basic formula: chi2 = (Δf/σf)² + (Δτ/στ)²
    """
    obs_f, pred_f, sigma_f = 260.0, 250.0, 5.0      # delta = 10, term = 4
    obs_tau, pred_tau, sigma_tau = 0.012, 0.010, 0.001  # delta = 0.002, term = 4
    expected = (10.0 / 5.0) ** 2 + (0.002 / 0.001) ** 2
    result = chi2_mode(obs_f, obs_tau, pred_f, pred_tau, sigma_f, sigma_tau)
    assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# 3. passes_mode_threshold — True strictly below boundary
# ---------------------------------------------------------------------------

def test_passes_mode_threshold_true_below_boundary():
    assert passes_mode_threshold(4.0, DEFAULT_MODE_CHI2_THRESHOLD_90) is True


# ---------------------------------------------------------------------------
# 4. passes_mode_threshold — False at and above boundary
# ---------------------------------------------------------------------------

def test_passes_mode_threshold_false_above_boundary():
    # At the boundary → reject (conservative choice)
    assert passes_mode_threshold(DEFAULT_MODE_CHI2_THRESHOLD_90, DEFAULT_MODE_CHI2_THRESHOLD_90) is False
    # Clearly above → reject
    assert passes_mode_threshold(DEFAULT_MODE_CHI2_THRESHOLD_99, DEFAULT_MODE_CHI2_THRESHOLD_90) is False


# ---------------------------------------------------------------------------
# 5. delta_area — positive when final > initial
# ---------------------------------------------------------------------------

def test_delta_area_positive():
    result = delta_area(final_area=100.0, initial_total_area=80.0)
    assert result == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 6. passes_area_law — exact zero passes with zero tolerance
# ---------------------------------------------------------------------------

def test_passes_area_law_exact_zero_with_zero_tolerance():
    assert passes_area_law(0.0, DEFAULT_AREA_TOLERANCE) is True


def test_passes_area_law_negative_fails_with_zero_tolerance():
    assert passes_area_law(-1e-10, DEFAULT_AREA_TOLERANCE) is False


def test_passes_area_law_relaxed_tolerance():
    # With tolerance 0.5, a deficit of 0.3 should pass.
    assert passes_area_law(-0.3, 0.5) is True


# ---------------------------------------------------------------------------
# 7. exact_intersection_geometry_ids — non-empty common set
# ---------------------------------------------------------------------------

def test_exact_intersection_geometry_ids_nonempty():
    result = exact_intersection_geometry_ids([
        ["g1", "g2", "g3"],
        ["g2", "g3", "g4"],
        ["g2", "g3"],
    ])
    assert result == ["g2", "g3"]


# ---------------------------------------------------------------------------
# 8. exact_intersection_geometry_ids — empty when no common element
# ---------------------------------------------------------------------------

def test_exact_intersection_geometry_ids_empty():
    result = exact_intersection_geometry_ids([
        ["g1", "g2"],
        ["g3", "g4"],
    ])
    assert result == []


def test_exact_intersection_geometry_ids_empty_inner_iterable():
    """An empty inner iterable makes the intersection empty."""
    result = exact_intersection_geometry_ids([
        ["g1", "g2"],
        [],
    ])
    assert result == []


def test_exact_intersection_geometry_ids_empty_outer():
    assert exact_intersection_geometry_ids([]) == []


# ---------------------------------------------------------------------------
# 9. support_count_geometry_ids — counts correctly
# ---------------------------------------------------------------------------

def test_support_count_geometry_ids_counts_correctly():
    counts = support_count_geometry_ids([
        ["g1", "g2"],
        ["g2", "g3"],
        ["g1", "g3", "g4"],
    ])
    assert counts["g1"] == 2
    assert counts["g2"] == 2
    assert counts["g3"] == 2
    assert counts["g4"] == 1


def test_support_count_geometry_ids_deduplicates_within_inner():
    """Repeated geometry_id within one inner iterable counts only once."""
    counts = support_count_geometry_ids([["g1", "g1", "g2"], ["g1"]])
    assert counts["g1"] == 2   # present in 2 inner lists, not 3
    assert counts["g2"] == 1


# ---------------------------------------------------------------------------
# 10. rank_geometries_by_support — orders descending
# ---------------------------------------------------------------------------

def test_rank_geometries_by_support_orders_descending():
    rows = rank_geometries_by_support([
        ["g1", "g2"],
        ["g2", "g3"],
        ["g2"],
    ])
    # g2 appears 3 times, g1 and g3 appear once each.
    assert rows[0]["geometry_id"] == "g2"
    assert rows[0]["support_count"] == 3
    # g1 and g3 both have count 1; they should be ordered lexicographically.
    ids_rest = [r["geometry_id"] for r in rows[1:]]
    assert ids_rest == sorted(ids_rest)


# ---------------------------------------------------------------------------
# 11. singleton_geometry_id — returns None for non-singleton
# ---------------------------------------------------------------------------

def test_singleton_geometry_id_returns_none_for_non_singleton():
    assert singleton_geometry_id([]) is None
    assert singleton_geometry_id(["g1", "g2"]) is None


def test_singleton_geometry_id_returns_value_for_singleton():
    assert singleton_geometry_id(["g1"]) == "g1"


# ---------------------------------------------------------------------------
# 12. robust_unique_verdict — NO_DATA when n_valid_scenarios == 0
# ---------------------------------------------------------------------------

def test_robust_unique_verdict_no_data():
    result = robust_unique_verdict(singleton_ids=[], n_valid_scenarios=0)
    assert result["robustness_verdict"] == VERDICT_NO_DATA
    assert result["robust_unique_geometry_id"] is None
    assert result["support_fraction"] is None
    assert result["singleton_counts"] == {}


# ---------------------------------------------------------------------------
# 13. robust_unique_verdict — NOT_UNIQUE when no scenario produced a singleton
# ---------------------------------------------------------------------------

def test_robust_unique_verdict_not_unique():
    result = robust_unique_verdict(
        singleton_ids=[None, None, None],
        n_valid_scenarios=3,
    )
    assert result["robustness_verdict"] == VERDICT_NOT_UNIQUE
    assert result["robust_unique_geometry_id"] is None
    assert result["support_fraction"] is None


# ---------------------------------------------------------------------------
# 14. robust_unique_verdict — UNSTABLE_UNIQUE when two geometries compete
# ---------------------------------------------------------------------------

def test_robust_unique_verdict_unstable_unique_when_two_singletons_compete():
    # g1 wins in 2 scenarios, g2 wins in 1 → two distinct geometries
    result = robust_unique_verdict(
        singleton_ids=["g1", "g2", "g1"],
        n_valid_scenarios=3,
    )
    assert result["robustness_verdict"] == VERDICT_UNSTABLE_UNIQUE
    assert result["robust_unique_geometry_id"] is None
    assert result["support_fraction"] is None
    assert "g1" in result["singleton_counts"]
    assert "g2" in result["singleton_counts"]


# ---------------------------------------------------------------------------
# 15. robust_unique_verdict — ROBUST_UNIQUE when single geometry dominates
# ---------------------------------------------------------------------------

def test_robust_unique_verdict_robust_unique_when_single_geometry_dominates():
    # g1 wins in 9 out of 10 scenarios (fraction = 0.9 >= 0.80)
    singletons = ["g1"] * 9 + [None]
    result = robust_unique_verdict(
        singleton_ids=singletons,
        n_valid_scenarios=10,
    )
    assert result["robustness_verdict"] == VERDICT_ROBUST_UNIQUE
    assert result["robust_unique_geometry_id"] == "g1"
    assert result["support_fraction"] == pytest.approx(0.9)
    assert result["singleton_counts"]["g1"] == 9


def test_robust_unique_verdict_unstable_when_below_threshold():
    # g1 wins in 7 out of 10 → fraction = 0.7 < 0.80 → UNSTABLE
    singletons = ["g1"] * 7 + [None] * 3
    result = robust_unique_verdict(
        singleton_ids=singletons,
        n_valid_scenarios=10,
    )
    assert result["robustness_verdict"] == VERDICT_UNSTABLE_UNIQUE
    assert result["robust_unique_geometry_id"] is None


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_constants_are_correct():
    assert GOLDEN_GEOMETRY_SPEC_VERSION == "v1"
    assert MODE_220 == "220"
    assert MODE_221 == "221"
    assert DEFAULT_MODE_CHI2_THRESHOLD_90 == pytest.approx(4.605)
    assert DEFAULT_MODE_CHI2_THRESHOLD_99 == pytest.approx(9.210)
    assert DEFAULT_AREA_TOLERANCE == 0.0
    assert ROBUST_UNIQUE_MIN_SUPPORT_FRACTION == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Payload builder smoke tests
# ---------------------------------------------------------------------------

def test_build_mode_filter_payload_has_required_fields():
    payload = build_mode_filter_payload(
        run_id="test-run",
        stage="s4_geometry_filter",
        mode=MODE_220,
        geometry_ids=["g1", "g2"],
        chi2_threshold=DEFAULT_MODE_CHI2_THRESHOLD_90,
        verdict=VERDICT_PASS,
    )
    assert payload["schema_name"] == "golden_geometry_mode_filter"
    assert payload["schema_version"] == GOLDEN_GEOMETRY_SPEC_VERSION
    assert "created_utc" in payload
    assert payload["run_id"] == "test-run"
    assert payload["stage"] == "s4_geometry_filter"
    assert payload["mode"] == MODE_220
    assert payload["verdict"] == VERDICT_PASS
    assert payload["n_passed"] == 2


def test_build_common_geometries_payload_has_required_fields():
    payload = build_common_geometries_payload(
        run_id="run-abc",
        stage="my_stage",
        common_geometry_ids=["g3"],
        verdict=VERDICT_PASS,
    )
    assert payload["schema_name"] == "golden_geometry_common"
    assert payload["schema_version"] == GOLDEN_GEOMETRY_SPEC_VERSION
    assert payload["n_common"] == 1
    assert "created_utc" in payload


def test_build_golden_geometries_payload_has_required_fields():
    payload = build_golden_geometries_payload(
        run_id="r1",
        stage="s_golden",
        golden_geometry_ids=[],
        verdict="NO_GOLDEN_GEOMETRIES",
    )
    assert payload["schema_name"] == "golden_geometry_per_event"
    assert payload["n_golden"] == 0


def test_build_population_consensus_payload_has_required_fields():
    ranked = rank_geometries_by_support([["g1", "g2"], ["g1"]])
    payload = build_population_consensus_payload(
        experiment_run_id="exp-001",
        experiment_name="test_experiment",
        exact_global_geometry_ids=["g1"],
        ranked_by_support=ranked,
        verdict="EXACT_GLOBAL_GEOMETRY_FOUND",
    )
    assert payload["schema_name"] == "golden_geometry_population_consensus"
    assert payload["experiment_run_id"] == "exp-001"
    assert payload["n_exact_global"] == 1
    assert len(payload["ranked_by_support"]) == 2


def test_build_single_event_robustness_payload_has_required_fields():
    robustness = robust_unique_verdict(
        singleton_ids=["g1"] * 9 + [None],
        n_valid_scenarios=10,
    )
    payload = build_single_event_robustness_payload(
        run_id="r2",
        stage="s_robustness",
        robustness_result=robustness,
        n_valid_scenarios=10,
    )
    assert payload["schema_name"] == "golden_geometry_single_event_robustness"
    assert payload["schema_version"] == GOLDEN_GEOMETRY_SPEC_VERSION
    assert payload["n_valid_scenarios"] == 10
    assert payload["robustness_verdict"] == VERDICT_ROBUST_UNIQUE


def test_payload_builders_accept_created_utc_override():
    fixed = "2024-01-01T00:00:00Z"
    payload = build_mode_filter_payload(
        run_id="r",
        stage="s",
        mode=MODE_220,
        geometry_ids=[],
        chi2_threshold=1.0,
        verdict=VERDICT_REJECT,
        created_utc=fixed,
    )
    assert payload["created_utc"] == fixed


def test_payload_builders_accept_extra_kwargs():
    payload = build_golden_geometries_payload(
        run_id="r",
        stage="s",
        golden_geometry_ids=["g1"],
        verdict=VERDICT_PASS,
        event_id="GW150914",
    )
    assert payload["event_id"] == "GW150914"
