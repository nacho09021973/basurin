from __future__ import annotations

from mvp.contracts import CONTRACTS, audit_contract_completeness


def test_no_undocumented_dynamic_inputs() -> None:
    assert CONTRACTS["s3_ringdown_estimates"].dynamic_inputs == [
        "s2_ringdown_window/outputs/{detector}_rd.npz",
    ]
    assert CONTRACTS["s3b_multimode_estimates"].dynamic_inputs == [
        "s2_ringdown_window/outputs/{detector}_rd.npz",
    ]
    assert CONTRACTS["s3_spectral_estimates"].dynamic_inputs == [
        "s2_ringdown_window/outputs/{detector}_rd.npz",
    ]


def test_dynamic_inputs_have_pattern() -> None:
    for contract in CONTRACTS.values():
        for pattern in contract.dynamic_inputs:
            assert isinstance(pattern, str)
            assert pattern.strip()
            assert ("*" in pattern) or ("{" in pattern and "}" in pattern)


def test_external_inputs_documented() -> None:
    assert CONTRACTS["s4b_spectral_curvature"].external_inputs == ["atlas"]
    assert CONTRACTS["s4_spectral_geometry_filter"].external_inputs == ["atlas"]
    assert CONTRACTS["s6c_brunete_psd_curvature"].external_inputs == ["psd_model"]


def test_all_contracts_have_upstream() -> None:
    expected = {"s0_oracle_mvp", "s1_fetch_strain"}
    actual = set(audit_contract_completeness())
    assert actual == expected


def test_existing_contracts_unchanged() -> None:
    assert CONTRACTS["s4_geometry_filter"].required_inputs == [
        "s3_ringdown_estimates/outputs/estimates.json",
    ]
    assert CONTRACTS["s4_geometry_filter"].produced_outputs == ["outputs/compatible_set.json"]
    assert CONTRACTS["s3_ringdown_estimates"].upstream_stages == ["s2_ringdown_window"]
    assert CONTRACTS["s6_information_geometry"].upstream_stages == [
        "s3_ringdown_estimates",
        "s4_geometry_filter",
    ]
