from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mvp.s4_geometry_filter import _add_mahalanobis_audit_fields


def test_atlas_posterior_fields_without_fixed_theta() -> None:
    compatible_set = {
        "metric": "mahalanobis_log",
        "d2_min": 0.1343,
        "ranked_all": [
            {"geometry_id": "g0", "d2": 0.1343, "distance": math.sqrt(0.1343)},
            {"geometry_id": "g1", "d2": 6.0, "distance": math.sqrt(6.0)},
            {"geometry_id": "g2", "d2": 12.0, "distance": math.sqrt(12.0)},
        ],
    }

    _add_mahalanobis_audit_fields(compatible_set)

    ranked = compatible_set["ranked_all"]
    best = ranked[0]
    assert best["delta_lnL"] == pytest.approx(0.0)

    weights = [row["posterior_weight"] for row in ranked]
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]

    atlas_posterior = compatible_set["atlas_posterior"]
    assert atlas_posterior["chi2_interpretation"] == "min_over_atlas_not_chi2"
    assert "p_value_best" not in compatible_set
    assert "p_value_best" not in atlas_posterior
    assert compatible_set["chi2_fixed_theta"] is None


def test_fixed_theta_chi2_block_uses_exact_df2_survival() -> None:
    compatible_set = {
        "metric": "mahalanobis_log",
        "d2_min": 0.1343,
        "ranked_all": [
            {"geometry_id": "g0", "d2": 0.1343},
            {"geometry_id": "g1", "d2": 6.0},
            {"geometry_id": "g2", "d2": 12.0},
        ],
    }

    _add_mahalanobis_audit_fields(
        compatible_set,
        fixed_theta0=1,
        theta0_source="inspiral_prior",
    )

    fixed_theta = compatible_set["chi2_fixed_theta"]
    assert fixed_theta is not None
    assert fixed_theta["dof"] == 2
    assert fixed_theta["d2"] == pytest.approx(6.0)
    assert fixed_theta["p_value"] == pytest.approx(math.exp(-6.0 / 2.0))
    assert fixed_theta["classification"] == "tension"
    assert fixed_theta["theta0_source"] == "inspiral_prior"
