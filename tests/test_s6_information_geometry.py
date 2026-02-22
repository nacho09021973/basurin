"""Tests for MVP Stage 6: Information Geometry.

Tests:
    1. Flat-space regression: constant Omega => R == 0 (no numerical artifacts).
    2. Non-zero curvature: frequency-dependent Omega produces R != 0.
    3. Conformal distance differs from flat distance when Omega varies.
    4. Full stage CLI produces contract-compliant outputs.
    5. Stage aborts on missing inputs.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"
ATLAS_FIXTURE = MVP_DIR / "test_atlas_fixture.json"

# Ensure repo root on path for imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s6_information_geometry import (
    conformal_factor,
    scalar_curvature_2d,
    conformal_distance,
    compute_information_geometry,
    PSD_MODELS,
    F_REF_HZ,
)


def _run_stage(script: str, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(MVP_DIR / script)] + args
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env, cwd=str(REPO_ROOT))


def _create_run_valid(runs_root: Path, run_id: str) -> None:
    rv_dir = runs_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    (rv_dir / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")


def _assert_stage_contract(runs_root: Path, run_id: str, stage_name: str) -> dict:
    stage_dir = runs_root / run_id / stage_name
    assert stage_dir.exists(), f"Stage dir missing: {stage_dir}"
    summary_path = stage_dir / "stage_summary.json"
    assert summary_path.exists(), f"stage_summary.json missing in {stage_dir}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = stage_dir / "manifest.json"
    assert manifest_path.exists(), f"manifest.json missing in {stage_dir}"
    outputs_dir = stage_dir / "outputs"
    assert outputs_dir.exists(), f"outputs/ missing in {stage_dir}"
    return summary


# ── Test 1: Flat-space regression ────────────────────────────────────────


class TestFlatSpaceRegression:
    """When Omega is constant, curvature must be exactly zero."""

    def test_curvature_flat_space(self):
        """Constant PSD => constant Omega => R == 0."""
        # Use a constant PSD (returns 1.0 for all f)
        def psd_constant(_f: float) -> float:
            return 1.0

        result = scalar_curvature_2d(
            f_obs=251.0,
            snr_peak=10.0,
            psd_fn=psd_constant,
            delta_log_f=0.01,
        )

        assert result["numerical_valid"] is True
        assert abs(result["R"]) < 1e-12, (
            f"Curvature must be zero for constant Omega, got R={result['R']}"
        )
        assert result["omega_obs"] == pytest.approx(100.0)  # snr^2 * 1/1

    def test_curvature_flat_space_various_snr(self):
        """R == 0 for constant PSD regardless of SNR value."""
        def psd_constant(_f: float) -> float:
            return 5.0

        for snr in [1.0, 10.0, 100.0, 0.5]:
            result = scalar_curvature_2d(
                f_obs=200.0, snr_peak=snr, psd_fn=psd_constant,
            )
            assert abs(result["R"]) < 1e-12, f"R != 0 for SNR={snr}"

    def test_conformal_distance_equals_flat_when_omega_constant(self):
        """With constant Omega, conformal distance is proportional to flat distance."""
        def psd_constant(_f: float) -> float:
            return 1.0

        snr = 10.0
        d = conformal_distance(
            f_obs=251.0, Q_obs=4.0,
            f_atlas=260.0, Q_atlas=5.0,
            snr_peak=snr, psd_fn=psd_constant,
        )

        # Omega is constant = snr^2 = 100, so d_conformal = sqrt(100) * d_flat = 10 * d_flat
        expected_ratio = math.sqrt(snr**2)
        actual_ratio = d["d_conformal"] / d["d_flat"]
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-10)


# ── Test 2: Non-zero curvature ───────────────────────────────────────────


class TestNonZeroCurvature:
    """With frequency-dependent PSD, curvature should be non-zero."""

    def test_simplified_aligo_gives_nonzero_curvature(self):
        """The simplified aLIGO PSD model produces non-zero R."""
        psd_fn = PSD_MODELS["simplified_aligo"]
        result = scalar_curvature_2d(
            f_obs=251.0, snr_peak=10.0, psd_fn=psd_fn,
        )
        assert result["numerical_valid"] is True
        assert result["R"] != 0.0, "Curvature should be non-zero for aLIGO PSD"

    def test_curvature_sign_consistency(self):
        """Curvature sign should be consistent across nearby frequencies."""
        psd_fn = PSD_MODELS["simplified_aligo"]
        signs = []
        for f in [200.0, 220.0, 250.0, 280.0, 300.0]:
            r = scalar_curvature_2d(f_obs=f, snr_peak=10.0, psd_fn=psd_fn)
            if r["numerical_valid"] and r["R"] != 0:
                signs.append(math.copysign(1, r["R"]))
        # All should have the same sign in the detection band
        assert len(set(signs)) == 1, f"Curvature sign inconsistent: {signs}"


# ── Test 3: Conformal distances ──────────────────────────────────────────


class TestConformalDistances:
    def test_conformal_distance_differs_from_flat(self):
        """Conformal and flat distances should differ with frequency-dependent PSD."""
        psd_fn = PSD_MODELS["simplified_aligo"]
        d = conformal_distance(
            f_obs=251.0, Q_obs=4.0,
            f_atlas=300.0, Q_atlas=7.0,
            snr_peak=10.0, psd_fn=psd_fn,
        )
        assert d["d_flat"] > 0
        assert d["d_conformal"] > 0
        # They should not be simply proportional when Omega varies
        ratio_obs = d["omega_obs"]
        ratio_atlas = d["omega_atlas"]
        assert ratio_obs != pytest.approx(ratio_atlas, rel=0.01), (
            "Omega should differ at different frequencies"
        )

    def test_conformal_distance_symmetry(self):
        """d(A,B) == d(B,A) for conformal distance."""
        psd_fn = PSD_MODELS["simplified_aligo"]
        d_ab = conformal_distance(
            f_obs=251.0, Q_obs=4.0,
            f_atlas=300.0, Q_atlas=7.0,
            snr_peak=10.0, psd_fn=psd_fn,
        )
        d_ba = conformal_distance(
            f_obs=300.0, Q_obs=7.0,
            f_atlas=251.0, Q_atlas=4.0,
            snr_peak=10.0, psd_fn=psd_fn,
        )
        assert d_ab["d_conformal"] == pytest.approx(d_ba["d_conformal"], rel=1e-10)


# ── Test 4: Full computation ─────────────────────────────────────────────


class TestComputeInformationGeometry:
    def test_full_computation_returns_valid_structure(self):
        """compute_information_geometry returns correctly structured outputs."""
        geometries = [
            {"geometry_id": "geo_005", "f_hz": 251.0, "Q": 4.2,
             "distance": 0.05, "compatible": True, "metadata": {"family": "kerr"}},
            {"geometry_id": "geo_007", "f_hz": 260.0, "Q": 5.0,
             "distance": 0.15, "compatible": True, "metadata": {"family": "kerr"}},
        ]

        curv, diag = compute_information_geometry(
            f_obs=251.0, Q_obs=3.14,
            snr_peak=10.0,
            compatible_geometries=geometries,
        )

        # Curvature output structure
        assert curv["schema_version"] == "mvp_curvature_v1"
        assert "scalar_curvature_R" in curv
        assert "omega_at_obs" in curv
        assert curv["n_geometries_reranked"] == 2
        assert len(curv["reranked_geometries"]) == 2

        # Each reranked geometry has required fields
        for g in curv["reranked_geometries"]:
            assert "geometry_id" in g
            assert "d_flat" in g
            assert "d_conformal" in g
            assert "rank_conformal" in g
            assert "rank_flat" in g
            assert "rank_delta" in g

        # Diagnostics structure
        assert diag["schema_version"] == "mvp_metric_diagnostics_v1"
        assert "scalar_curvature_R" in diag
        assert "caveats" in diag
        assert len(diag["caveats"]) > 0


# ── Test 5: CLI stage contract compliance ────────────────────────────────


class TestS6CLI:
    def _create_mock_upstream(self, runs_root: Path, run_id: str) -> None:
        """Create mock s3 and s4 outputs for s6 to consume."""
        # s3 estimates
        est_dir = runs_root / run_id / "s3_ringdown_estimates"
        est_out = est_dir / "outputs"
        est_out.mkdir(parents=True, exist_ok=True)
        estimates = {
            "schema_version": "mvp_estimates_v1",
            "event_id": "GW150914",
            "combined": {
                "f_hz": 251.0,
                "tau_s": 0.004,
                "Q": math.pi * 251.0 * 0.004,
            },
            "per_detector": {
                "H1": {"f_hz": 250.8, "tau_s": 0.00397, "Q": 3.12, "snr_peak": 45.2},
                "L1": {"f_hz": 251.2, "tau_s": 0.00403, "Q": 3.13, "snr_peak": 38.1},
            },
            "n_detectors_valid": 2,
        }
        (est_out / "estimates.json").write_text(json.dumps(estimates), encoding="utf-8")
        (est_dir / "stage_summary.json").write_text(
            json.dumps({"verdict": "PASS"}), encoding="utf-8"
        )
        (est_dir / "manifest.json").write_text(
            json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8"
        )

        # s4 compatible set (with atlas entries that have f_hz and Q)
        s4_dir = runs_root / run_id / "s4_geometry_filter"
        s4_out = s4_dir / "outputs"
        s4_out.mkdir(parents=True, exist_ok=True)
        compat = {
            "schema_version": "mvp_compatible_set_v1",
            "event_id": "GW150914",
            "run_id": run_id,
            "observables": {"f_hz": 251.0, "Q": 3.15},
            "epsilon": 0.3,
            "n_atlas": 20,
            "n_compatible": 3,
            "bits_excluded": 2.74,
            "compatible_geometries": [
                {"geometry_id": "geo_005", "distance": 0.05, "compatible": True,
                 "f_hz": 251.0, "Q": 4.2, "metadata": {"family": "kerr", "chi": 0.32}},
                {"geometry_id": "geo_014", "distance": 0.12, "compatible": True,
                 "f_hz": 252.0, "Q": 4.1, "metadata": {"family": "ads_dw", "delta": 2.6}},
                {"geometry_id": "geo_004", "distance": 0.25, "compatible": True,
                 "f_hz": 250.0, "Q": 4.0, "metadata": {"family": "kerr", "chi": 0.3}},
            ],
        }
        (s4_out / "compatible_set.json").write_text(json.dumps(compat), encoding="utf-8")
        (s4_dir / "stage_summary.json").write_text(
            json.dumps({"verdict": "PASS"}), encoding="utf-8"
        )
        (s4_dir / "manifest.json").write_text(
            json.dumps({"stage": "s4_geometry_filter"}), encoding="utf-8"
        )

    def test_s6_produces_contract(self, tmp_path):
        """s6 CLI produces curvature.json + metric_diagnostics.json + contract files."""
        runs_root = tmp_path / "runs"
        run_id = "test_s6"
        _create_run_valid(runs_root, run_id)
        self._create_mock_upstream(runs_root, run_id)

        result = _run_stage(
            "s6_information_geometry.py",
            ["--run", run_id],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0, f"s6 failed: {result.stderr}"

        summary = _assert_stage_contract(runs_root, run_id, "s6_information_geometry")
        assert summary["verdict"] == "PASS"

        # Check both outputs exist
        outputs = runs_root / run_id / "s6_information_geometry" / "outputs"
        curv_path = outputs / "curvature.json"
        diag_path = outputs / "metric_diagnostics.json"
        assert curv_path.exists()
        assert diag_path.exists()

        # Validate curvature.json content
        curv = json.loads(curv_path.read_text(encoding="utf-8"))
        assert curv["schema_version"] == "mvp_curvature_v1"
        assert curv["event_id"] == "GW150914"
        assert curv["run_id"] == run_id
        assert curv["scalar_curvature_R"] != 0.0
        assert curv["curvature_numerical_valid"] is True
        assert curv["n_geometries_reranked"] == 3
        assert curv["snr_peak"] == 45.2  # max(H1, L1)

        # Validate metric_diagnostics.json content
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        assert diag["schema_version"] == "mvp_metric_diagnostics_v1"
        assert diag["psd_model"] == "simplified_aligo"
        assert diag["numerical_valid"] is True
        assert len(diag["caveats"]) >= 1

    def test_s6_aborts_on_missing_estimates(self, tmp_path):
        """s6 aborts with exit 2 when s3 estimates are missing."""
        runs_root = tmp_path / "runs"
        run_id = "test_s6_missing"
        _create_run_valid(runs_root, run_id)
        # Don't create any upstream — s6 should abort

        result = _run_stage(
            "s6_information_geometry.py",
            ["--run", run_id],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 2

    def test_s6_uses_snr_from_per_detector(self, tmp_path):
        """s6 picks max snr_peak from per-detector estimates."""
        runs_root = tmp_path / "runs"
        run_id = "test_s6_snr"
        _create_run_valid(runs_root, run_id)
        self._create_mock_upstream(runs_root, run_id)

        result = _run_stage(
            "s6_information_geometry.py",
            ["--run", run_id],
            env={"BASURIN_RUNS_ROOT": str(runs_root)},
        )
        assert result.returncode == 0

        curv = json.loads(
            (runs_root / run_id / "s6_information_geometry" / "outputs" / "curvature.json")
            .read_text(encoding="utf-8")
        )
        # H1 has snr_peak=45.2, L1 has 38.1 — should pick max
        assert curv["snr_peak"] == 45.2
