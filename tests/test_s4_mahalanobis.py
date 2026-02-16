"""Regression tests for s4_geometry_filter Mahalanobis distance.

Test 1: Mahalanobis reduces to Euclidean when Σ = I
Test 2: χ² threshold smoke test (d²=0 always compatible, far point excluded)
Test 3: Singular covariance raises ValueError (FAIL policy)
Test 4: Deterministic hash (identical inputs → identical output)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="legacy")

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"

# Ensure mvp is importable
sys.path.insert(0, str(REPO_ROOT))
from mvp.s4_geometry_filter import compute_compatible_set, CHI2_2DOF_95


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_atlas(entries: list[dict]) -> list[dict]:
    """Wrap entries into atlas format with geometry_id."""
    return [{"geometry_id": f"geo_{i:03d}", **e} for i, e in enumerate(entries)]


# ── Test 1: Mahalanobis == Euclidean² when Σ = I ─────────────────────────

class TestMahalanobisIdentityCovariance:
    def test_d2_equals_euclidean_squared_when_sigma_is_identity(self):
        """With Σ = I (sigma_logf=1, sigma_logQ=1, cov=0),
        d²_mahalanobis == d²_euclidean for every atlas entry."""
        f_obs, Q_obs = 251.0, 4.0
        atlas = _make_atlas([
            {"f_hz": 250.0, "Q": 3.9},
            {"f_hz": 260.0, "Q": 5.0},
            {"f_hz": 300.0, "Q": 7.0},
        ])

        # Mahalanobis with Σ = I
        res_mah = compute_compatible_set(
            f_obs, Q_obs, atlas, epsilon=100.0,
            sigma_logf=1.0, sigma_logQ=1.0, cov_logf_logQ=0.0,
        )
        # Euclidean (no covariance)
        res_euc = compute_compatible_set(f_obs, Q_obs, atlas, epsilon=100.0)

        assert res_mah["metric"] == "mahalanobis"
        assert res_euc["metric"] == "euclidean"

        for m, e in zip(res_mah["ranked_all"], res_euc["ranked_all"]):
            d2_mah = m["d2"]
            d2_euc = e["distance"] ** 2
            assert abs(d2_mah - d2_euc) < 1e-10, (
                f"d²_mah={d2_mah} != d²_euc={d2_euc} for {m['geometry_id']}")


# ── Test 2: χ² threshold smoke test ──────────────────────────────────────

class TestChi2Threshold:
    def test_exact_match_always_compatible(self):
        """Δ = 0 → d² = 0, always compatible regardless of threshold."""
        atlas = _make_atlas([{"f_hz": 251.0, "Q": 4.0}])
        res = compute_compatible_set(
            251.0, 4.0, atlas, epsilon=CHI2_2DOF_95,
            sigma_logf=0.05, sigma_logQ=0.1, cov_logf_logQ=0.0,
        )
        assert res["n_compatible"] == 1
        assert res["ranked_all"][0]["d2"] == 0.0

    def test_far_point_excluded(self):
        """Point far away in log-space → d² >> 5.991 → not compatible."""
        atlas = _make_atlas([{"f_hz": 1000.0, "Q": 100.0}])
        res = compute_compatible_set(
            251.0, 4.0, atlas, epsilon=CHI2_2DOF_95,
            sigma_logf=0.05, sigma_logQ=0.1, cov_logf_logQ=0.0,
        )
        assert res["n_compatible"] == 0
        assert res["ranked_all"][0]["d2"] > CHI2_2DOF_95

    def test_threshold_boundary(self):
        """Verify entries right at the boundary are correctly classified."""
        # Construct a point at known d²
        f_obs, Q_obs = 100.0, 10.0
        sigma_logf, sigma_logQ = 0.1, 0.1

        # A point where Δlogf = sigma_logf, ΔlogQ = 0 → d² = 1.0
        f_atlas = f_obs * math.exp(sigma_logf)  # Δlogf = sigma_logf
        atlas = _make_atlas([{"f_hz": f_atlas, "Q": Q_obs}])

        # With threshold = 1.0, d² = 1.0 should be compatible (<=)
        res = compute_compatible_set(
            f_obs, Q_obs, atlas, epsilon=1.0,
            sigma_logf=sigma_logf, sigma_logQ=sigma_logQ,
        )
        assert res["n_compatible"] == 1
        assert abs(res["ranked_all"][0]["d2"] - 1.0) < 1e-9

        # With threshold = 0.999, d² = 1.0 should NOT be compatible
        res2 = compute_compatible_set(
            f_obs, Q_obs, atlas, epsilon=0.999,
            sigma_logf=sigma_logf, sigma_logQ=sigma_logQ,
        )
        assert res2["n_compatible"] == 0


# ── Test 3: Singular covariance → FAIL ───────────────────────────────────

class TestSingularCovarianceFail:
    def test_sigma_logf_zero_raises(self):
        atlas = _make_atlas([{"f_hz": 251.0, "Q": 4.0}])
        with pytest.raises(ValueError, match="Non-invertible covariance"):
            compute_compatible_set(
                251.0, 4.0, atlas, epsilon=CHI2_2DOF_95,
                sigma_logf=0.0, sigma_logQ=0.1,
            )

    def test_sigma_logQ_zero_raises(self):
        atlas = _make_atlas([{"f_hz": 251.0, "Q": 4.0}])
        with pytest.raises(ValueError, match="Non-invertible covariance"):
            compute_compatible_set(
                251.0, 4.0, atlas, epsilon=CHI2_2DOF_95,
                sigma_logf=0.05, sigma_logQ=0.0,
            )

    def test_inf_sigma_raises(self):
        atlas = _make_atlas([{"f_hz": 251.0, "Q": 4.0}])
        with pytest.raises(ValueError, match="Non-invertible covariance"):
            compute_compatible_set(
                251.0, 4.0, atlas, epsilon=CHI2_2DOF_95,
                sigma_logf=float("inf"), sigma_logQ=0.1,
            )

    def test_singular_via_large_cov_raises(self):
        """det(Σ) ≤ 0 when |cov| > σ_f · σ_Q."""
        atlas = _make_atlas([{"f_hz": 251.0, "Q": 4.0}])
        # cov > sigma_logf * sigma_logQ → det < 0
        with pytest.raises(ValueError, match="Non-invertible covariance"):
            compute_compatible_set(
                251.0, 4.0, atlas, epsilon=CHI2_2DOF_95,
                sigma_logf=0.1, sigma_logQ=0.1,
                cov_logf_logQ=0.02,  # > 0.1 * 0.1 = 0.01 → det < 0
            )

    def test_negative_sigma_raises(self):
        atlas = _make_atlas([{"f_hz": 251.0, "Q": 4.0}])
        with pytest.raises(ValueError, match="Non-invertible covariance"):
            compute_compatible_set(
                251.0, 4.0, atlas, epsilon=CHI2_2DOF_95,
                sigma_logf=-0.05, sigma_logQ=0.1,
            )


# ── Test 4: Deterministic hash ───────────────────────────────────────────

class TestDeterministicHash:
    def test_identical_runs_produce_same_hash(self):
        """Two calls with identical inputs → byte-identical JSON → same SHA-256."""
        f_obs, Q_obs = 251.0, 4.0
        atlas = _make_atlas([
            {"f_hz": 250.0, "Q": 3.9},
            {"f_hz": 260.0, "Q": 5.0},
            {"f_hz": 300.0, "Q": 7.0},
        ])
        kwargs = dict(
            sigma_logf=0.05, sigma_logQ=0.1, cov_logf_logQ=0.0,
        )

        res1 = compute_compatible_set(f_obs, Q_obs, atlas, CHI2_2DOF_95, **kwargs)
        res2 = compute_compatible_set(f_obs, Q_obs, atlas, CHI2_2DOF_95, **kwargs)

        j1 = json.dumps(res1, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(res2, sort_keys=True, separators=(",", ":"))

        h1 = hashlib.sha256(j1.encode()).hexdigest()
        h2 = hashlib.sha256(j2.encode()).hexdigest()

        assert h1 == h2, f"Non-deterministic output: {h1} != {h2}"
        assert j1 == j2, "JSON output differs between runs"


# ── Test 5: Backward compatibility (no covariance → Euclidean) ───────────

class TestBackwardCompatibility:
    def test_no_covariance_uses_euclidean(self):
        """Without sigma_logf/sigma_logQ, falls back to Euclidean metric."""
        atlas = _make_atlas([
            {"f_hz": 250.0, "Q": 3.9},
            {"f_hz": 260.0, "Q": 5.0},
        ])
        res = compute_compatible_set(251.0, 4.0, atlas, epsilon=0.3)
        assert res["metric"] == "euclidean"
        assert "d2" not in res["ranked_all"][0]
        assert "threshold_d2" not in res

    def test_output_schema_additive_only(self):
        """Mahalanobis mode adds fields but doesn't remove existing ones."""
        atlas = _make_atlas([{"f_hz": 250.0, "Q": 3.9}])
        res = compute_compatible_set(
            251.0, 4.0, atlas, epsilon=CHI2_2DOF_95,
            sigma_logf=0.05, sigma_logQ=0.1,
        )
        # All original fields must be present
        for key in ["schema_version", "observables", "epsilon", "n_atlas",
                     "n_compatible", "bits_excluded", "compatible_geometries",
                     "ranked_all"]:
            assert key in res, f"Missing key: {key}"
        # Additive Mahalanobis fields
        assert res["metric"] == "mahalanobis"
        assert "threshold_d2" in res
        assert "d2_min" in res
        assert "covariance_logspace" in res


# ── Test 6: Nonzero covariance support ───────────────────────────────────

class TestNonzeroCovariance:
    def test_positive_cov_changes_d2(self):
        """Nonzero cov_logf_logQ produces different d² than cov=0."""
        f_obs, Q_obs = 251.0, 4.0
        atlas = _make_atlas([{"f_hz": 260.0, "Q": 5.0}])

        res_nocov = compute_compatible_set(
            f_obs, Q_obs, atlas, epsilon=100.0,
            sigma_logf=0.1, sigma_logQ=0.1, cov_logf_logQ=0.0,
        )
        res_cov = compute_compatible_set(
            f_obs, Q_obs, atlas, epsilon=100.0,
            sigma_logf=0.1, sigma_logQ=0.1, cov_logf_logQ=0.005,
        )
        d2_nocov = res_nocov["ranked_all"][0]["d2"]
        d2_cov = res_cov["ranked_all"][0]["d2"]
        assert d2_nocov != d2_cov, (
            f"Nonzero cov should change d²: {d2_nocov} == {d2_cov}")


# ── Test 7: Integration via subprocess (Mahalanobis path) ────────────────

def _run_stage(script, args, env=None):
    cmd = [sys.executable, str(MVP_DIR / script)] + args
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env, cwd=str(REPO_ROOT))


def _create_run_valid(runs_root, run_id):
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(
        json.dumps({"verdict": "PASS", "run_id": run_id}), encoding="utf-8"
    )


ATLAS_FIXTURE = MVP_DIR / "test_atlas_fixture.json"


class TestS4MahalanobisSubprocess:
    def test_subprocess_with_covariance(self, tmp_path):
        """s4 uses Mahalanobis when combined_uncertainty is present."""
        runs_root = tmp_path / "runs"
        run_id = "test_s4_mah"
        _create_run_valid(runs_root, run_id)

        est_dir = runs_root / run_id / "s3_ringdown_estimates"
        est_out = est_dir / "outputs"
        est_out.mkdir(parents=True, exist_ok=True)
        estimates = {
            "schema_version": "mvp_estimates_v1",
            "event_id": "GW150914",
            "combined": {"f_hz": 251.0, "tau_s": 0.004, "Q": math.pi * 251.0 * 0.004},
            "combined_uncertainty": {
                "sigma_f_hz": 5.0,
                "sigma_tau_s": 0.001,
                "sigma_Q": 0.5,
                "sigma_logf": 0.02,
                "sigma_logQ": 0.16,
                "cov_logf_logQ": 0.0,
            },
        }
        (est_out / "estimates.json").write_text(json.dumps(estimates), encoding="utf-8")
        (est_dir / "stage_summary.json").write_text(
            json.dumps({"verdict": "PASS"}), encoding="utf-8"
        )

        result = _run_stage(
            "s4_geometry_filter.py",
            ["--run", run_id, "--atlas-path", str(ATLAS_FIXTURE)],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s4 failed: {result.stderr}"

        cs_path = runs_root / run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json"
        cs = json.loads(cs_path.read_text(encoding="utf-8"))
        assert cs["metric"] == "mahalanobis"
        assert cs["threshold_d2"] == CHI2_2DOF_95
        assert cs["d2_min"] is not None
        assert "covariance_logspace" in cs
        assert cs["n_atlas"] == 20

    def test_subprocess_without_covariance_fallback(self, tmp_path):
        """s4 falls back to Euclidean when no combined_uncertainty."""
        runs_root = tmp_path / "runs"
        run_id = "test_s4_euc"
        _create_run_valid(runs_root, run_id)

        est_dir = runs_root / run_id / "s3_ringdown_estimates"
        est_out = est_dir / "outputs"
        est_out.mkdir(parents=True, exist_ok=True)
        estimates = {
            "schema_version": "mvp_estimates_v1",
            "event_id": "GW150914",
            "combined": {"f_hz": 251.0, "tau_s": 0.004, "Q": math.pi * 251.0 * 0.004},
        }
        (est_out / "estimates.json").write_text(json.dumps(estimates), encoding="utf-8")
        (est_dir / "stage_summary.json").write_text(
            json.dumps({"verdict": "PASS"}), encoding="utf-8"
        )

        result = _run_stage(
            "s4_geometry_filter.py",
            ["--run", run_id, "--atlas-path", str(ATLAS_FIXTURE), "--epsilon", "0.3"],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s4 failed: {result.stderr}"

        cs_path = runs_root / run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json"
        cs = json.loads(cs_path.read_text(encoding="utf-8"))
        assert cs["metric"] == "euclidean"
        assert cs["epsilon"] == 0.3

    def test_subprocess_singular_cov_aborts(self, tmp_path):
        """s4 aborts with exit 2 when covariance is singular."""
        runs_root = tmp_path / "runs"
        run_id = "test_s4_sing"
        _create_run_valid(runs_root, run_id)

        est_dir = runs_root / run_id / "s3_ringdown_estimates"
        est_out = est_dir / "outputs"
        est_out.mkdir(parents=True, exist_ok=True)
        estimates = {
            "schema_version": "mvp_estimates_v1",
            "event_id": "GW150914",
            "combined": {"f_hz": 251.0, "tau_s": 0.004, "Q": math.pi * 251.0 * 0.004},
            "combined_uncertainty": {
                "sigma_f_hz": 5.0,
                "sigma_tau_s": 0.001,
                "sigma_Q": 0.5,
                "sigma_logf": 0.0,  # Singular!
                "sigma_logQ": 0.1,
                "cov_logf_logQ": 0.0,
            },
        }
        (est_out / "estimates.json").write_text(json.dumps(estimates), encoding="utf-8")
        (est_dir / "stage_summary.json").write_text(
            json.dumps({"verdict": "PASS"}), encoding="utf-8"
        )

        result = _run_stage(
            "s4_geometry_filter.py",
            ["--run", run_id, "--atlas-path", str(ATLAS_FIXTURE)],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 2
        assert "Non-invertible covariance" in result.stderr
