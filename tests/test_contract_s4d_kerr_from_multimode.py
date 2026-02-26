import mvp.contracts as c


def test_contract_s4d_kerr_from_multimode_minimal():
    sc = c.CONTRACTS["s4d_kerr_from_multimode"]

    assert sc.upstream_stages == ["s3b_multimode_estimates"]

    # Canonical required input (model_comparison.json intentionally optional)
    assert sc.required_inputs == [
        "s3b_multimode_estimates/outputs/multimode_estimates.json",
    ]

    # Canonical produced outputs under runs/<run_id>/s4d_kerr_from_multimode/outputs/
    assert sc.produced_outputs == [
        "outputs/kerr_from_multimode.json",
        "outputs/kerr_from_multimode_diagnostics.json",
    ]
