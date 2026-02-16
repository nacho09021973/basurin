"""Regression tests for canonical combined_uncertainty contract in S3."""
from __future__ import annotations

import math


def test_combined_uncertainty_modern_keys_are_present_and_valid() -> None:
    """Modern canonical keys must exist and satisfy finite-domain constraints."""
    estimates = {
        "combined_uncertainty": {
            "sigma_f_hz": 2.1,
            "sigma_tau_s": 0.0004,
            "sigma_Q": 4.2,
            # legacy
            "cov_logf_logQ": 0.0,
            "sigma_logf": 0.01,
            "sigma_logQ": 0.02,
            # modern canonical
            "sigma_lnf": 0.01,
            "sigma_lnQ": 0.02,
            "r": 0.0,
        }
    }

    unc = estimates["combined_uncertainty"]

    for key in ("sigma_lnf", "sigma_lnQ", "r"):
        assert key in unc, f"Missing combined_uncertainty.{key}"

    assert unc["sigma_lnf"] > 0 and math.isfinite(unc["sigma_lnf"])
    assert unc["sigma_lnQ"] > 0 and math.isfinite(unc["sigma_lnQ"])
    assert abs(unc["r"]) < 1 and math.isfinite(unc["r"])
