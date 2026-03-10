from __future__ import annotations

import math

from mvp.s8_family_router import FAMILY_BNS, FAMILY_GR_KERR, FAMILY_LOW_MASS_BH, route_family_candidates
from mvp.s8a_family_gr_kerr import assess_gr_kerr_family
from mvp.s8b_family_bns import assess_bns_family
from mvp.s8c_family_low_mass_bh_postmerger import assess_low_mass_bh_family


def _mode_payload(label: str, f_hz: float, tau_s: float, frac: float = 0.02) -> dict:
    return {
        "label": label,
        "fit": {
            "stability": {
                "lnf_p10": math.log(f_hz * (1.0 - frac)),
                "lnf_p50": math.log(f_hz),
                "lnf_p90": math.log(f_hz * (1.0 + frac)),
                "lnQ_p10": math.log(math.pi * f_hz * tau_s * (1.0 - frac)),
                "lnQ_p50": math.log(math.pi * f_hz * tau_s),
                "lnQ_p90": math.log(math.pi * f_hz * tau_s * (1.0 + frac)),
            }
        },
    }


def _multimode_payload(f220_hz: float, tau220_s: float, f221_hz: float, tau221_s: float) -> dict:
    return {
        "modes": [
            _mode_payload("220", f220_hz, tau220_s),
            _mode_payload("221", f221_hz, tau221_s),
        ]
    }


def test_router_prefers_known_bbh_catalog_events() -> None:
    routing = route_family_candidates(
        event_id="GW150914",
        metadata={},
        known_bbh_catalog_entry={"m_final_msun": 62.2, "chi_final": 0.67},
        multimode_viability_class="MULTIMODE_OK",
    )

    assert routing["primary_family"] == FAMILY_GR_KERR
    assert routing["families_to_run"] == [FAMILY_GR_KERR]
    assert routing["routing_mode"] == "catalog_known_bbh"


def test_router_prioritizes_bns_when_metadata_says_bns() -> None:
    routing = route_family_candidates(
        event_id="GW170817",
        metadata={"source_class": "binary_neutron_star", "multimessenger": True},
        known_bbh_catalog_entry=None,
        multimode_viability_class="MULTIMODE_OK",
    )

    assert routing["primary_family"] == FAMILY_BNS
    assert routing["families_to_run"] == [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR]
    assert routing["routing_mode"] == "metadata_bns_or_multimessenger"


def test_router_does_not_emit_generic_domain_status() -> None:
    routing = route_family_candidates(
        event_id="GW170817",
        metadata={"source_class": "binary_neutron_star", "multimessenger": True},
        known_bbh_catalog_entry=None,
        multimode_viability_class="MULTIMODE_OK",
    )

    assert routing["primary_family"] == FAMILY_BNS
    assert "domain_status" not in routing


def test_gr_kerr_family_assessment_supported_when_score_is_consistent() -> None:
    payload = assess_gr_kerr_family(
        router_payload={"primary_family": FAMILY_GR_KERR, "families_to_run": [FAMILY_GR_KERR]},
        ratio_filter={"verdict": "PASS", "kerr_consistency": {"Rf_consistent": True}, "diagnostics": {"informativity_class": "MODERATE"}, "filtering": {"n_input_geometries": 6, "n_ratio_compatible": 4, "n_ratio_excluded": 2, "n_ratio_not_applicable": 0}},
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 62.0, "chi_final": 0.67},
        beyond_kerr_score={"verdict": "GR_CONSISTENT", "chi2_kerr_2dof": 1.2, "independence_class": "NON_INDEPENDENT"},
        kerr_consistency={"status": "OK", "kerr_consistent": True, "source": {"compatible_set_present": True}},
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "SUPPORTED"
    assert payload["m_final_msun"] == 62.0
    assert payload["ratio_rf_consistent"] is True


def test_gr_kerr_family_requires_explicit_monomode_support_before_supported() -> None:
    payload = assess_gr_kerr_family(
        router_payload={"primary_family": FAMILY_GR_KERR, "families_to_run": [FAMILY_GR_KERR]},
        ratio_filter={"verdict": "PASS", "kerr_consistency": {"Rf_consistent": True}, "diagnostics": {"informativity_class": "MODERATE"}, "filtering": {"n_input_geometries": 6, "n_ratio_compatible": 4, "n_ratio_excluded": 2, "n_ratio_not_applicable": 0}},
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 62.0, "chi_final": 0.67},
        beyond_kerr_score={"verdict": "GR_CONSISTENT", "chi2_kerr_2dof": 1.2, "independence_class": "NON_INDEPENDENT"},
        kerr_consistency={"status": "OK", "kerr_consistent": False, "source": {"compatible_set_present": True}},
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "INCONCLUSIVE"
    assert "non-independent" in payload["reason"]
    assert "s4c" in payload["reason"]

    skipped_payload = assess_gr_kerr_family(
        router_payload={"primary_family": FAMILY_GR_KERR, "families_to_run": [FAMILY_GR_KERR]},
        ratio_filter={"verdict": "PASS", "kerr_consistency": {"Rf_consistent": True}, "diagnostics": {"informativity_class": "MODERATE"}, "filtering": {"n_input_geometries": 6, "n_ratio_compatible": 4, "n_ratio_excluded": 2, "n_ratio_not_applicable": 0}},
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 62.0, "chi_final": 0.67},
        beyond_kerr_score={"verdict": "GR_CONSISTENT", "chi2_kerr_2dof": 1.2, "independence_class": "NON_INDEPENDENT"},
        kerr_consistency={"status": "SKIPPED_MULTIMODE_GATE", "kerr_consistent": None, "source": {"compatible_set_present": True}},
    )

    assert skipped_payload["status"] == "EVALUATED"
    assert skipped_payload["assessment"] == "INCONCLUSIVE"
    assert "s4c" in skipped_payload["reason"]


def test_gr_kerr_family_is_inconclusive_without_geometric_support() -> None:
    payload = assess_gr_kerr_family(
        router_payload={"primary_family": FAMILY_GR_KERR, "families_to_run": [FAMILY_GR_KERR]},
        ratio_filter={"verdict": "PASS", "kerr_consistency": {"Rf_consistent": True}, "diagnostics": {"informativity_class": "UNINFORMATIVE"}, "filtering": {"n_input_geometries": 0, "n_ratio_compatible": 0, "n_ratio_excluded": 0, "n_ratio_not_applicable": 0}},
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 144.0, "chi_final": 0.96},
        beyond_kerr_score={"verdict": "GR_CONSISTENT", "chi2_kerr_2dof": 0.15, "independence_class": "NON_INDEPENDENT"},
        kerr_consistency={"status": "OK", "kerr_consistent": True, "source": {"compatible_set_present": True}},
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "INCONCLUSIVE"
    assert "no surviving geometries" in payload["reason"]


def test_gr_kerr_family_is_disfavored_for_astrophysical_inconsistency() -> None:
    payload = assess_gr_kerr_family(
        router_payload={"primary_family": FAMILY_GR_KERR, "families_to_run": [FAMILY_GR_KERR]},
        ratio_filter={"verdict": "PASS", "kerr_consistency": {"Rf_consistent": True}, "diagnostics": {"informativity_class": "LOW"}, "filtering": {"n_input_geometries": 5, "n_ratio_compatible": 2, "n_ratio_excluded": 3, "n_ratio_not_applicable": 0}},
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 144.0, "chi_final": 0.96},
        beyond_kerr_score={"verdict": "ASTRO_INCONSISTENT", "chi2_kerr_2dof": 0.15},
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "DISFAVORED"
    assert "astrophysical" in payload["reason"].lower()


def test_gr_kerr_family_is_disfavored_when_ratio_excludes_all_spin_geometries() -> None:
    payload = assess_gr_kerr_family(
        router_payload={"primary_family": FAMILY_GR_KERR, "families_to_run": [FAMILY_GR_KERR]},
        ratio_filter={"verdict": "PASS", "kerr_consistency": {"Rf_consistent": False}, "diagnostics": {"informativity_class": "HIGH"}, "filtering": {"n_input_geometries": 5, "n_ratio_compatible": 0, "n_ratio_excluded": 5, "n_ratio_not_applicable": 0}},
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 62.0, "chi_final": 0.67},
        beyond_kerr_score={"verdict": "GR_CONSISTENT", "chi2_kerr_2dof": 1.2, "independence_class": "NON_INDEPENDENT"},
        kerr_consistency={"status": "OK", "kerr_consistent": True, "source": {"compatible_set_present": True}},
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "DISFAVORED"
    assert "excludes all surviving spin-bearing geometries" in payload["reason"]


def test_bns_family_handler_supports_matching_candidate() -> None:
    payload = assess_bns_family(
        router_payload={"primary_family": FAMILY_BNS, "families_to_run": [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR]},
        run_provenance={"invocation": {"event_id": "GW170817"}},
        s3b_stage_summary={"multimode_viability": {"class": "MULTIMODE_OK"}},
        multimode_estimates=_multimode_payload(3465.0, 0.0094, 2830.905, 0.00517),
        event_metadata={
            "family_priors": {
                FAMILY_BNS: {
                    "remnant_mass_msun_range": [2.7, 2.7],
                    "radius_1p6_km_range": [12.0, 12.0],
                    "classes": ["HMNS"],
                    "n_mass_points": 1,
                    "n_radius_points": 1,
                    "collapse_time_ms_values": {"HMNS": [20.0]},
                }
            }
        },
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "SUPPORTED"
    assert payload["model_status"] == "PHENOMENOLOGICAL_V1"
    assert payload["atlas_summary"]["n_candidates_compatible"] >= 1
    assert payload["best_candidate"]["remnant_class"] == "HMNS"


def test_bns_family_handler_ignores_generic_upstream_out_of_domain_when_local_overlap_exists() -> None:
    payload = assess_bns_family(
        router_payload={
            "primary_family": FAMILY_BNS,
            "families_to_run": [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR],
            "domain_status": "OUT_OF_DOMAIN",
        },
        run_provenance={"invocation": {"event_id": "GW170817", "key_params": {"band_low": 2500.0, "band_high": 4000.0}}},
        s3b_stage_summary={"multimode_viability": {"class": "MULTIMODE_OK"}},
        multimode_estimates=_multimode_payload(3465.0, 0.0094, 2830.905, 0.00517),
        event_metadata={
            "family_priors": {
                FAMILY_BNS: {
                    "remnant_mass_msun_range": [2.7, 2.7],
                    "radius_1p6_km_range": [12.0, 12.0],
                    "classes": ["HMNS"],
                    "n_mass_points": 1,
                    "n_radius_points": 1,
                    "collapse_time_ms_values": {"HMNS": [20.0]},
                }
            }
        },
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "SUPPORTED"
    assert "domain_status" not in payload


def test_bns_family_handler_disfavors_incompatible_low_frequency_signal() -> None:
    payload = assess_bns_family(
        router_payload={"primary_family": FAMILY_BNS, "families_to_run": [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR]},
        run_provenance={"invocation": {"event_id": "GW170817"}},
        s3b_stage_summary={"multimode_viability": {"class": "MULTIMODE_OK"}},
        multimode_estimates=_multimode_payload(224.0, 0.028, 227.0, 0.026),
        event_metadata={
            "family_priors": {
                FAMILY_BNS: {
                    "remnant_mass_msun_range": [2.5, 2.8],
                    "radius_1p6_km_range": [11.0, 13.0],
                    "classes": ["HMNS", "SMNS"],
                    "n_mass_points": 3,
                    "n_radius_points": 3,
                }
            }
        },
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "DISFAVORED"
    assert payload["atlas_summary"]["n_candidates_compatible"] == 0


def test_bns_family_handler_marks_out_of_domain_when_band_has_no_atlas_overlap() -> None:
    payload = assess_bns_family(
        router_payload={
            "primary_family": FAMILY_BNS,
            "families_to_run": [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR],
        },
        run_provenance={"invocation": {"event_id": "GW170817", "key_params": {"band_low": 150.0, "band_high": 400.0}}},
        s3b_stage_summary={"multimode_viability": {"class": "MULTIMODE_OK"}},
        multimode_estimates=_multimode_payload(224.0, 0.028, 227.0, 0.026),
        event_metadata={
            "family_priors": {
                FAMILY_BNS: {
                    "remnant_mass_msun_range": [2.5, 2.8],
                    "radius_1p6_km_range": [11.0, 13.0],
                    "classes": ["HMNS", "SMNS"],
                    "n_mass_points": 3,
                    "n_radius_points": 3,
                }
            }
        },
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "INCONCLUSIVE"
    assert payload["domain_status"] == "OUT_OF_DOMAIN"
    assert "no physically useful overlap" in payload["reason"]


def test_low_mass_bh_family_supports_matching_low_mass_kerr_solution() -> None:
    payload = assess_low_mass_bh_family(
        router_payload={"primary_family": FAMILY_BNS, "families_to_run": [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR]},
        run_provenance={"invocation": {"event_id": "GW170817"}},
        s3b_stage_summary={"multimode_viability": {"class": "MULTIMODE_OK"}},
        ratio_filter={
            "kerr_consistency": {"Rf_consistent": True},
            "diagnostics": {"informativity_class": "MODERATE"},
            "filtering": {"n_ratio_compatible": 8},
        },
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 2.72, "chi_final": 0.81},
        beyond_kerr_score={"verdict": "GR_CONSISTENT"},
        event_metadata={
            "family_priors": {
                FAMILY_LOW_MASS_BH: {
                    "mass_msun_range": [2.5, 3.0],
                    "chi_range": [0.7, 0.9],
                }
            }
        },
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "SUPPORTED"
    assert payload["mass_in_range"] is True
    assert payload["chi_in_range"] is True


def test_low_mass_bh_family_disfavors_high_mass_solution() -> None:
    payload = assess_low_mass_bh_family(
        router_payload={"primary_family": FAMILY_BNS, "families_to_run": [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR]},
        run_provenance={"invocation": {"event_id": "GW170817"}},
        s3b_stage_summary={"multimode_viability": {"class": "MULTIMODE_OK"}},
        ratio_filter={
            "kerr_consistency": {"Rf_consistent": True},
            "diagnostics": {"informativity_class": "LOW"},
            "filtering": {"n_ratio_compatible": 3},
        },
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 62.0, "chi_final": 0.67},
        beyond_kerr_score={"verdict": "GR_CONSISTENT"},
        event_metadata={"family_priors": {FAMILY_LOW_MASS_BH: {"mass_msun_range": [2.5, 3.0], "chi_range": [0.6, 0.95]}}},
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "DISFAVORED"
    assert payload["mass_in_range"] is False


def test_low_mass_bh_family_marks_out_of_domain_when_band_has_no_kerr_overlap() -> None:
    payload = assess_low_mass_bh_family(
        router_payload={"primary_family": FAMILY_BNS, "families_to_run": [FAMILY_BNS, FAMILY_LOW_MASS_BH, FAMILY_GR_KERR]},
        run_provenance={"invocation": {"event_id": "GW170817", "key_params": {"band_low": 150.0, "band_high": 400.0}}},
        s3b_stage_summary={"multimode_viability": {"class": "MULTIMODE_OK"}},
        ratio_filter={
            "kerr_consistency": {"Rf_consistent": True},
            "diagnostics": {"informativity_class": "LOW"},
            "filtering": {"n_ratio_compatible": 3},
        },
        kerr_extraction={"verdict": "PASS", "M_final_Msun": 2.72, "chi_final": 0.81},
        beyond_kerr_score={"verdict": "GR_CONSISTENT"},
        event_metadata={
            "family_priors": {
                FAMILY_LOW_MASS_BH: {
                    "mass_msun_range": [2.5, 3.0],
                    "chi_range": [0.7, 0.9],
                }
            }
        },
    )

    assert payload["status"] == "EVALUATED"
    assert payload["assessment"] == "INCONCLUSIVE"
    assert payload["domain_status"] == "OUT_OF_DOMAIN"
    assert "no physically useful overlap" in payload["reason"]
