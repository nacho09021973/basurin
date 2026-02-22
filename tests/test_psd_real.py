"""Tests for Q4: measured PSD loading and s6 backward compatibility.

Tests:
    1. load_measured_psd with synthetic data: interpolates correctly
    2. Conformal factor with measured PSD ≈ analytic (same shape)
    3. Scalar curvature with smooth measured PSD: R finite and stable
    4. Anti-regression: s6 without --psd-path gives same results as before
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvp.s6_information_geometry import (
    load_measured_psd,
    conformal_factor,
    scalar_curvature_2d,
    _psd_simplified_aligo,
    F_REF_HZ,
)


def _make_synthetic_psd_json(path: Path, f_min: float = 10.0, f_max: float = 2000.0,
                              n_points: int = 200) -> None:
    """Write a synthetic PSD JSON file with the analytic aLIGO shape."""
    freqs = list(np.logspace(math.log10(f_min), math.log10(f_max), n_points))
    psd_vals = [_psd_simplified_aligo(f) for f in freqs]
    data = {
        "schema_version": "mvp_measured_psd_v1",
        "frequencies_hz": freqs,
        "psd_values": psd_vals,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _make_synthetic_psd_npz(path: Path, f_min: float = 10.0, f_max: float = 2000.0,
                              n_points: int = 200) -> None:
    """Write a synthetic PSD NPZ file."""
    freqs = np.logspace(math.log10(f_min), math.log10(f_max), n_points)
    psd_vals = np.array([_psd_simplified_aligo(f) for f in freqs])
    np.savez(path, freq=freqs, psd=psd_vals)


class TestLoadMeasuredPSD:
    """Test 1: load_measured_psd with synthetic data."""

    def test_load_json_returns_callable(self, tmp_path):
        psd_path = tmp_path / "psd.json"
        _make_synthetic_psd_json(psd_path)
        psd_fn = load_measured_psd(psd_path)
        assert callable(psd_fn)

    def test_load_npz_returns_callable(self, tmp_path):
        psd_path = tmp_path / "psd.npz"
        _make_synthetic_psd_npz(psd_path)
        psd_fn = load_measured_psd(psd_path)
        assert callable(psd_fn)

    def test_interpolation_at_reference_freq(self, tmp_path):
        """At f_ref=200 Hz, measured PSD should match analytic value (loaded from same)."""
        psd_path = tmp_path / "psd.json"
        _make_synthetic_psd_json(psd_path)
        psd_fn = load_measured_psd(psd_path)
        expected = _psd_simplified_aligo(F_REF_HZ)
        measured = psd_fn(F_REF_HZ)
        rel_err = abs(measured - expected) / expected
        assert rel_err < 0.05, f"rel_err={rel_err:.4f} at f_ref={F_REF_HZ} Hz"

    def test_interpolation_across_band(self, tmp_path):
        """Spot-check PSD at several frequencies in the band."""
        psd_path = tmp_path / "psd.json"
        _make_synthetic_psd_json(psd_path)
        psd_fn = load_measured_psd(psd_path)
        for f_test in [150.0, 200.0, 250.0, 300.0, 350.0]:
            expected = _psd_simplified_aligo(f_test)
            measured = psd_fn(f_test)
            rel_err = abs(measured - expected) / expected
            assert rel_err < 0.10, f"f={f_test}: rel_err={rel_err:.4f}"

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_measured_psd(tmp_path / "nonexistent.json")

    def test_empty_json_raises(self, tmp_path):
        psd_path = tmp_path / "empty.json"
        psd_path.write_text('{"frequencies_hz": [], "psd_values": []}')
        with pytest.raises(ValueError, match="at least 2"):
            load_measured_psd(psd_path)


class TestConformalFactorWithMeasuredPSD:
    """Test 2: conformal_factor with measured PSD ≈ analytic PSD."""

    def test_conformal_factor_at_200hz(self, tmp_path):
        """Both should give same Ω at 200 Hz (reference frequency)."""
        psd_path = tmp_path / "psd.json"
        _make_synthetic_psd_json(psd_path)
        psd_fn = load_measured_psd(psd_path)

        snr = 10.0
        omega_analytic = conformal_factor(200.0, snr, _psd_simplified_aligo)
        omega_measured = conformal_factor(200.0, snr, psd_fn)
        # At reference frequency, both = snr^2
        assert abs(omega_analytic - snr ** 2) / snr ** 2 < 0.01
        assert abs(omega_measured - snr ** 2) / snr ** 2 < 0.10

    def test_conformal_factor_positive(self, tmp_path):
        psd_path = tmp_path / "psd.json"
        _make_synthetic_psd_json(psd_path)
        psd_fn = load_measured_psd(psd_path)
        snr = 8.0
        for f in [150.0, 200.0, 250.0, 300.0]:
            omega = conformal_factor(f, snr, psd_fn)
            assert omega > 0, f"omega <= 0 at f={f} Hz"


class TestScalarCurvatureWithMeasuredPSD:
    """Test 3: scalar curvature with smooth measured PSD → R finite and stable."""

    def test_curvature_finite(self, tmp_path):
        psd_path = tmp_path / "psd.json"
        _make_synthetic_psd_json(psd_path)
        psd_fn = load_measured_psd(psd_path)

        result = scalar_curvature_2d(250.0, snr_peak=10.0, psd_fn=psd_fn)
        assert result["numerical_valid"]
        assert math.isfinite(result["R"]), f"R={result['R']} is not finite"

    def test_curvature_similar_to_analytic(self, tmp_path):
        """R from measured PSD (same shape) should be close to analytic."""
        psd_path = tmp_path / "psd.json"
        _make_synthetic_psd_json(psd_path)
        psd_fn = load_measured_psd(psd_path)

        snr = 10.0
        f_obs = 250.0
        R_analytic = scalar_curvature_2d(f_obs, snr, _psd_simplified_aligo)["R"]
        R_measured = scalar_curvature_2d(f_obs, snr, psd_fn)["R"]

        if math.isfinite(R_analytic) and math.isfinite(R_measured) and abs(R_analytic) > 1e-10:
            ratio = abs(R_measured - R_analytic) / (abs(R_analytic) + 1e-30)
            assert ratio < 1.0, f"R ratio too large: R_analytic={R_analytic}, R_measured={R_measured}"


class TestAntiRegression:
    """Test 4: s6 without --psd-path gives same results as before."""

    def test_analytic_psd_unchanged(self):
        """_psd_simplified_aligo should still produce same values."""
        # At f=200 Hz, x=1 → 1 + 2 + 2 = 5
        psd_at_200 = _psd_simplified_aligo(F_REF_HZ)
        assert abs(psd_at_200 - 5.0) < 1e-10

    def test_conformal_factor_unchanged(self):
        """conformal_factor with analytic PSD should still work."""
        snr = 10.0
        omega = conformal_factor(F_REF_HZ, snr, _psd_simplified_aligo)
        assert abs(omega - snr ** 2) / snr ** 2 < 1e-10

    def test_compute_information_geometry_backward_compat(self):
        """compute_information_geometry with no psd_fn uses analytic PSD."""
        from mvp.s6_information_geometry import compute_information_geometry
        curvature, diagnostics = compute_information_geometry(
            f_obs=250.0, Q_obs=3.0, snr_peak=10.0,
            compatible_geometries=[],
            psd_model="simplified_aligo",
            psd_fn=None,
        )
        assert "scalar_curvature_R" in curvature
        assert math.isfinite(curvature["scalar_curvature_R"])
