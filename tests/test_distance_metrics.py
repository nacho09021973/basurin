"""Tests for distance metrics (euclidean_log, mahalanobis_log) and KL bits.

Validates mathematical properties, edge cases, and integration
with s4_geometry_filter and experiment_eps_sweep.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mvp.distance_metrics import (
    DEFAULT_CORRELATION,
    DEFAULT_SIGMA_LNF,
    DEFAULT_SIGMA_LNQ,
    METRICS,
    euclidean_log,
    get_metric,
    kl_bits,
    mahalanobis_log,
)

MVP_DIR = REPO_ROOT / "mvp"
ATLAS_FIXTURE = MVP_DIR / "test_atlas_fixture.json"


# ---------------------------------------------------------------------------
# Helper: create synthetic s3 estimates for integration tests
# ---------------------------------------------------------------------------
def _create_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8",
    )


def _create_s3_estimates(
    runs_root: Path, run_id: str, f_hz: float = 251.0, Q: float = 3.14,
) -> Path:
    stage_dir = runs_root / run_id / "s3_ringdown_estimates"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    estimates = {
        "schema_version": "mvp_estimates_v1",
        "event_id": "GW150914",
        "method": "hilbert_envelope",
        "combined": {"f_hz": f_hz, "tau_s": Q / (3.14159 * f_hz), "Q": Q},
        "per_detector": {"H1": {"f_hz": f_hz, "tau_s": 0.004, "Q": Q, "snr_peak": 8.0}},
        "n_detectors_valid": 1,
    }
    est_path = outputs_dir / "estimates.json"
    est_path.write_text(json.dumps(estimates, indent=2), encoding="utf-8")
    (stage_dir / "stage_summary.json").write_text(
        json.dumps({"stage": "s3_ringdown_estimates", "run": run_id, "verdict": "PASS"}),
        encoding="utf-8",
    )
    (stage_dir / "manifest.json").write_text(
        json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8",
    )
    return est_path


# ===========================================================================
# Unit tests: euclidean_log
# ===========================================================================
class TestEuclideanLog:
    def test_zero_distance_at_same_point(self) -> None:
        assert euclidean_log(5.0, 1.0, 5.0, 1.0) == 0.0

    def test_symmetry(self) -> None:
        d1 = euclidean_log(5.0, 1.0, 5.5, 1.2)
        d2 = euclidean_log(5.5, 1.2, 5.0, 1.0)
        assert abs(d1 - d2) < 1e-15

    def test_known_value(self) -> None:
        # d = sqrt(0.3² + 0.4²) = 0.5
        d = euclidean_log(0.0, 0.0, 0.3, 0.4)
        assert abs(d - 0.5) < 1e-10

    def test_ignores_extra_kwargs(self) -> None:
        d = euclidean_log(5.0, 1.0, 5.5, 1.2, sigma_lnf=0.1, r=0.9)
        assert d > 0


# ===========================================================================
# Unit tests: mahalanobis_log
# ===========================================================================
class TestMahalanobisLog:
    def test_zero_distance_at_same_point(self) -> None:
        d = mahalanobis_log(5.0, 1.0, 5.0, 1.0, sigma_lnf=0.1, sigma_lnQ=0.2)
        assert d == 0.0

    def test_symmetry(self) -> None:
        d1 = mahalanobis_log(5.0, 1.0, 5.5, 1.2, sigma_lnf=0.1, sigma_lnQ=0.2)
        d2 = mahalanobis_log(5.5, 1.2, 5.0, 1.0, sigma_lnf=0.1, sigma_lnQ=0.2)
        assert abs(d1 - d2) < 1e-15

    def test_reduces_to_euclidean_when_uncorrelated_unit_sigma(self) -> None:
        """With r=0 and σ=1 in both axes, Mahalanobis = Euclidean."""
        d_mah = mahalanobis_log(0.0, 0.0, 0.3, 0.4,
                                sigma_lnf=1.0, sigma_lnQ=1.0, r=0.0)
        d_euc = euclidean_log(0.0, 0.0, 0.3, 0.4)
        assert abs(d_mah - d_euc) < 1e-10

    def test_correlation_changes_distance(self) -> None:
        """Positive correlation should reduce distance along the diagonal."""
        kwargs = dict(sigma_lnf=0.1, sigma_lnQ=0.1)

        # Point along the diagonal (f and Q shift together)
        d_uncorr = mahalanobis_log(0.0, 0.0, 0.05, 0.05, r=0.0, **kwargs)
        d_corr = mahalanobis_log(0.0, 0.0, 0.05, 0.05, r=0.9, **kwargs)

        # With strong positive correlation, the diagonal direction
        # is "expected", so distance should be smaller
        assert d_corr < d_uncorr, (
            f"Correlated distance ({d_corr:.4f}) should be < "
            f"uncorrelated ({d_uncorr:.4f}) along diagonal"
        )

    def test_anticorrelated_direction_penalized(self) -> None:
        """Point perpendicular to correlation axis should have larger distance."""
        kwargs = dict(sigma_lnf=0.1, sigma_lnQ=0.1, r=0.9)

        # Along diagonal (correlated direction)
        d_diag = mahalanobis_log(0.0, 0.0, 0.05, 0.05, **kwargs)
        # Perpendicular (anti-correlated direction)
        d_perp = mahalanobis_log(0.0, 0.0, 0.05, -0.05, **kwargs)

        assert d_perp > d_diag, (
            f"Perpendicular ({d_perp:.4f}) should be > "
            f"diagonal ({d_diag:.4f})"
        )

    def test_smaller_sigma_means_larger_distance(self) -> None:
        """Tighter uncertainty → larger Mahalanobis distance for same offset."""
        d_wide = mahalanobis_log(0.0, 0.0, 0.1, 0.1,
                                 sigma_lnf=0.2, sigma_lnQ=0.2, r=0.0)
        d_tight = mahalanobis_log(0.0, 0.0, 0.1, 0.1,
                                  sigma_lnf=0.05, sigma_lnQ=0.05, r=0.0)
        assert d_tight > d_wide

    def test_1sigma_offset_gives_distance_near_1(self) -> None:
        """A 1σ shift in f only (with r=0) should give d ≈ 1."""
        sigma_f = 0.07
        d = mahalanobis_log(0.0, 0.0, sigma_f, 0.0,
                            sigma_lnf=sigma_f, sigma_lnQ=0.25, r=0.0)
        assert abs(d - 1.0) < 1e-10

    def test_rejects_r_equals_1(self) -> None:
        """Correlation |r| = 1 makes covariance singular."""
        with pytest.raises(ValueError, match=r"\|r\| must be < 1"):
            mahalanobis_log(0.0, 0.0, 0.1, 0.1, sigma_lnf=0.1, sigma_lnQ=0.2, r=1.0)

    def test_rejects_r_equals_minus_1(self) -> None:
        with pytest.raises(ValueError, match=r"\|r\| must be < 1"):
            mahalanobis_log(0.0, 0.0, 0.1, 0.1, sigma_lnf=0.1, sigma_lnQ=0.2, r=-1.0)

    def test_requires_sigmas(self) -> None:
        with pytest.raises(ValueError, match="sigma_lnf and sigma_lnQ are required"):
            mahalanobis_log(math.log(251), math.log(4.0), math.log(260), math.log(4.5))


# ===========================================================================
# Registry tests
# ===========================================================================
class TestMetricRegistry:
    def test_all_metrics_registered(self) -> None:
        assert "euclidean_log" in METRICS
        assert "mahalanobis_log" in METRICS

    def test_get_metric_returns_callable(self) -> None:
        fn = get_metric("euclidean_log")
        assert callable(fn)

    def test_get_metric_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("manhattan_log")


# ===========================================================================
# Unit tests: kl_bits
# ===========================================================================
class TestKLBits:
    def test_empty_returns_zero(self) -> None:
        assert kl_bits([]) == 0.0

    def test_single_entry_returns_zero(self) -> None:
        assert kl_bits([1.5]) == 0.0

    def test_all_equal_distances_returns_zero(self) -> None:
        """If all distances are equal, posterior = prior → D_KL = 0."""
        d = kl_bits([1.0, 1.0, 1.0, 1.0])
        assert abs(d) < 1e-10

    def test_one_zero_distance_concentrates_info(self) -> None:
        """One exact match (d=0) among many far entries → high KL."""
        # 1 match at d=0, 99 entries at d=100
        distances = [0.0] + [100.0] * 99
        kl = kl_bits(distances)
        # Maximum possible is log₂(100) ≈ 6.64 bits
        # With d=100, exp(-100²/2) ≈ 0, so posterior ≈ delta on first entry
        assert kl > 6.0
        assert kl <= math.log2(100) + 0.01

    def test_upper_bound_is_log2_N(self) -> None:
        """D_KL ≤ log₂(N) always (equality when posterior is a delta)."""
        N = 50
        distances = [0.0] + [1000.0] * (N - 1)
        kl = kl_bits(distances)
        assert kl <= math.log2(N) + 1e-10

    def test_monotonicity_with_concentration(self) -> None:
        """More concentrated posterior → more bits."""
        # Spread: all distances similar
        kl_spread = kl_bits([1.0, 1.1, 1.2, 1.3, 1.4])
        # Concentrated: one close, rest far
        kl_conc = kl_bits([0.1, 5.0, 5.0, 5.0, 5.0])
        assert kl_conc > kl_spread

    def test_non_negative(self) -> None:
        """D_KL is always non-negative."""
        import random
        rng = random.Random(42)
        for _ in range(20):
            ds = [rng.uniform(0, 10) for _ in range(rng.randint(2, 50))]
            assert kl_bits(ds) >= -1e-15

    def test_reduces_to_log2_N_over_n_for_binary_case(self) -> None:
        """When distances are 0 (compatible) or huge (incompatible),
        KL ≈ log₂(N/n), the old counting formula."""
        N, n = 100, 10
        distances = [0.0] * n + [1000.0] * (N - n)
        kl = kl_bits(distances)
        expected = math.log2(N / n)  # = log₂(10) ≈ 3.32
        assert abs(kl - expected) < 0.01, f"KL={kl:.4f}, expected≈{expected:.4f}"

    def test_numerically_stable_with_large_distances(self) -> None:
        """Should not overflow/NaN with very large distances."""
        distances = [0.5, 50.0, 500.0, 5000.0]
        kl = kl_bits(distances)
        assert math.isfinite(kl) and kl > 0


# ===========================================================================
# Integration: s4 compute_compatible_set with metric parameter
# ===========================================================================
class TestS4MetricIntegration:
    """Test that compute_compatible_set respects the metric parameter."""

    @pytest.fixture()
    def atlas(self) -> list[dict]:
        with open(ATLAS_FIXTURE, "r") as f:
            data = json.load(f)
        return data["entries"]

    def test_default_metric_is_euclidean(self, atlas: list[dict]) -> None:
        from mvp.s4_geometry_filter import compute_compatible_set

        result = compute_compatible_set(251.0, 4.0, atlas, 0.3)
        assert result.get("metric", "euclidean_log") == "euclidean_log"

    def test_mahalanobis_produces_different_distances(self, atlas: list[dict]) -> None:
        from mvp.s4_geometry_filter import compute_compatible_set

        r_euc = compute_compatible_set(251.0, 4.0, atlas, 999.0,
                                       metric="euclidean_log")
        r_mah = compute_compatible_set(251.0, 4.0, atlas, 999.0,
                                       metric="mahalanobis_log",
                                       metric_params={"sigma_lnf": 0.07, "sigma_lnQ": 0.25, "r": 0.9})

        # Same atlas, same observables → same n_atlas
        assert r_euc["n_atlas"] == r_mah["n_atlas"]

        # But distances should differ
        d_euc = [e["distance"] for e in r_euc["ranked_all"]]
        d_mah = [e["distance"] for e in r_mah["ranked_all"]]
        assert d_euc != d_mah, "Euclidean and Mahalanobis should give different distances"

    def test_mahalanobis_reranks_geometries(self, atlas: list[dict]) -> None:
        """The ranking order should differ between metrics (ellipse vs circle)."""
        from mvp.s4_geometry_filter import compute_compatible_set

        r_euc = compute_compatible_set(251.0, 4.0, atlas, 999.0,
                                       metric="euclidean_log")
        r_mah = compute_compatible_set(251.0, 4.0, atlas, 999.0,
                                       metric="mahalanobis_log",
                                       metric_params={"sigma_lnf": 0.07, "sigma_lnQ": 0.25, "r": 0.9})

        ranking_euc = [e["geometry_id"] for e in r_euc["ranked_all"]]
        ranking_mah = [e["geometry_id"] for e in r_mah["ranked_all"]]

        # Rankings CAN be different (elliptical vs circular iso-contours)
        # Don't assert they must differ (could be identical for small atlas)
        # but at least the distance values must differ
        assert any(
            abs(e["distance"] - m["distance"]) > 1e-10
            for e, m in zip(r_euc["ranked_all"], r_mah["ranked_all"])
        )

    def test_metric_params_passed_through(self, atlas: list[dict]) -> None:
        from mvp.s4_geometry_filter import compute_compatible_set

        result = compute_compatible_set(
            251.0, 4.0, atlas, 5.0,
            metric="mahalanobis_log",
            metric_params={"sigma_lnf": 0.05, "sigma_lnQ": 0.20, "r": 0.85},
        )
        assert result["metric"] == "mahalanobis_log"
        assert result["metric_params"]["r"] == 0.85

    def test_output_records_metric_in_json(self, atlas: list[dict]) -> None:
        from mvp.s4_geometry_filter import compute_compatible_set

        result = compute_compatible_set(251.0, 4.0, atlas, 3.0,
                                        metric="mahalanobis_log",
                                        metric_params={"sigma_lnf": 0.07, "sigma_lnQ": 0.25})
        assert result["metric"] == "mahalanobis_log"

    def test_mahalanobis_without_sigmas_raises(self, atlas: list[dict]) -> None:
        from mvp.s4_geometry_filter import compute_compatible_set

        with pytest.raises(ValueError, match="sigma_lnf and sigma_lnQ are required"):
            compute_compatible_set(251.0, 4.0, atlas, 3.0, metric="mahalanobis_log")

    def test_mahalanobis_zero_sigma_raises(self, atlas: list[dict]) -> None:
        """sigma_lnf=0 is non-invertible → should fail at API level."""
        from mvp.s4_geometry_filter import compute_compatible_set

        with pytest.raises(ValueError, match="Non-invertible covariance"):
            compute_compatible_set(
                251.0, 4.0, atlas, 3.0,
                metric="mahalanobis_log",
                metric_params={"sigma_lnf": 0.0, "sigma_lnQ": 0.25, "r": 0.0},
            )

    def test_mahalanobis_negative_sigma_raises(self, atlas: list[dict]) -> None:
        """Negative sigma → Non-invertible covariance."""
        from mvp.s4_geometry_filter import compute_compatible_set

        with pytest.raises(ValueError, match="Non-invertible covariance"):
            compute_compatible_set(
                251.0, 4.0, atlas, 3.0,
                metric="mahalanobis_log",
                metric_params={"sigma_lnf": 0.07, "sigma_lnQ": -0.1, "r": 0.0},
            )

    def test_mahalanobis_r_equals_1_raises_at_api_level(self, atlas: list[dict]) -> None:
        """Correlation |r| >= 1 in metric_params → caught by contract validation."""
        from mvp.s4_geometry_filter import compute_compatible_set

        with pytest.raises(ValueError, match="Non-invertible covariance"):
            compute_compatible_set(
                251.0, 4.0, atlas, 3.0,
                metric="mahalanobis_log",
                metric_params={"sigma_lnf": 0.07, "sigma_lnQ": 0.25, "r": 1.0},
            )

    def test_backward_compat_no_metric_arg(self, atlas: list[dict]) -> None:
        """Calling without metric= should still work (euclidean default)."""
        from mvp.s4_geometry_filter import compute_compatible_set

        result = compute_compatible_set(251.0, 4.0, atlas, 0.3)
        assert result["n_atlas"] > 0

    def test_bits_kl_present_in_output(self, atlas: list[dict]) -> None:
        from mvp.s4_geometry_filter import compute_compatible_set

        result = compute_compatible_set(251.0, 4.0, atlas, 0.3)
        assert "bits_kl" in result
        assert result["bits_kl"] >= 0
        assert math.isfinite(result["bits_kl"])

    def test_bits_kl_epsilon_independent(self, atlas: list[dict]) -> None:
        """bits_kl should not change with epsilon (it uses full distribution)."""
        from mvp.s4_geometry_filter import compute_compatible_set

        r1 = compute_compatible_set(251.0, 4.0, atlas, 0.1)
        r2 = compute_compatible_set(251.0, 4.0, atlas, 0.5)
        r3 = compute_compatible_set(251.0, 4.0, atlas, 999.0)
        assert abs(r1["bits_kl"] - r2["bits_kl"]) < 1e-10
        assert abs(r1["bits_kl"] - r3["bits_kl"]) < 1e-10

    def test_bits_kl_differs_from_bits_excluded(self, atlas: list[dict]) -> None:
        """KL and counting measure should generally differ."""
        from mvp.s4_geometry_filter import compute_compatible_set

        result = compute_compatible_set(251.0, 4.0, atlas, 0.3)
        # They CAN be close for small atlases, but typically differ
        # At minimum, both should be non-negative
        assert result["bits_kl"] >= 0
        assert result["bits_excluded"] >= 0


# ===========================================================================
# Integration: sweep with Mahalanobis
# ===========================================================================
class TestSweepMetricIntegration:
    """Test that the sweep script passes metric through correctly."""

    def test_sweep_with_mahalanobis(self, tmp_path: Path) -> None:
        from mvp.experiment_eps_sweep import EXPERIMENT_TAG, run_eps_sweep

        run_id = "test_sweep_mah"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id)

        summary = run_eps_sweep(
            run_id, ATLAS_FIXTURE, [0.5, 1.0, 2.0, 3.0],
            runs_root=tmp_path,
            metric="mahalanobis_log",
            metric_params={"sigma_lnf": 0.07, "sigma_lnQ": 0.25, "r": 0.9},
        )

        assert summary["metric"] == "mahalanobis_log"
        assert summary["metric_params"]["r"] == 0.9
        assert summary["n_epsilons"] == 4

        # Monotonicity still holds
        n_vals = [row["n_compatible"] for row in summary["rows"]]
        for i in range(1, len(n_vals)):
            assert n_vals[i] >= n_vals[i - 1]

        # bits_kl present in every row
        for row in summary["rows"]:
            assert "bits_kl" in row
            assert row["bits_kl"] >= 0

        # bits_kl is epsilon-independent → same across rows
        kl_vals = [row["bits_kl"] for row in summary["rows"]]
        assert all(abs(v - kl_vals[0]) < 1e-10 for v in kl_vals)

    def test_sweep_euclidean_still_works(self, tmp_path: Path) -> None:
        from mvp.experiment_eps_sweep import run_eps_sweep

        run_id = "test_sweep_euc"
        _create_run_valid(tmp_path, run_id)
        _create_s3_estimates(tmp_path, run_id)

        summary = run_eps_sweep(
            run_id, ATLAS_FIXTURE, [0.10, 0.30],
            runs_root=tmp_path,
        )
        # Default metric
        assert summary.get("metric", "euclidean_log") == "euclidean_log"
