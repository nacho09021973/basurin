"""Tests for Kerr QNM fitting formulas and atlas generation.

Validates Berti (2009) fits against:
  - Known Schwarzschild limit (chi=0)
  - The existing numerical atlas (qnm package, Leaver method)
  - Physical sanity checks
  - Alternative geometry deviations
  - Atlas generator output
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mvp.kerr_qnm_fits import (
    BERTI_FITS,
    MSUN_S,
    QNMResult,
    apply_deviation,
    deviation_dcs,
    deviation_edgb,
    deviation_kerr_newman,
    kerr_Q,
    kerr_omega_dimless,
    kerr_qnm,
    make_atlas_entry,
)


# ---------------------------------------------------------------------------
# Known values for validation
# ---------------------------------------------------------------------------
# Schwarzschild (chi=0) QNM (2,2,0): Mω = 0.3737 - 0.0890i
# (Leaver 1985, Nollert 1993)
SCHW_OMEGA_REAL = 0.3737
SCHW_Q = 2.0  # approximate


class TestBertiCoefficients:
    """Verify fit coefficients are self-consistent."""

    def test_all_modes_have_fits(self) -> None:
        assert (2, 2, 0) in BERTI_FITS
        assert (2, 2, 1) in BERTI_FITS
        assert (3, 3, 0) in BERTI_FITS

    def test_coefficients_are_finite(self) -> None:
        for mode, c in BERTI_FITS.items():
            for name in ("f1", "f2", "f3", "q1", "q2", "q3"):
                val = getattr(c, name)
                assert math.isfinite(val), f"{mode}.{name} = {val}"


class TestSchwarzschildLimit:
    """chi=0 should reproduce known Schwarzschild QNM values."""

    def test_omega_220_at_chi0(self) -> None:
        F = kerr_omega_dimless(0.0, (2, 2, 0))
        # Berti fit: F = 1.5251 + (-1.1568)*(1)^0.1292 = 1.5251 - 1.1568 = 0.3683
        # Known exact: 0.3737
        assert abs(F - SCHW_OMEGA_REAL) / SCHW_OMEGA_REAL < 0.02, f"F={F}"

    def test_Q_220_at_chi0(self) -> None:
        Q = kerr_Q(0.0, (2, 2, 0))
        # Should be ≈ 2.0
        assert abs(Q - SCHW_Q) / SCHW_Q < 0.10, f"Q={Q}"

    def test_f_hz_at_chi0_M62(self) -> None:
        """f ≈ 194 Hz for M=62 M_sun, chi=0 (from existing numerical atlas)."""
        result = kerr_qnm(62.0, 0.0, (2, 2, 0))
        assert 180 < result.f_hz < 210, f"f={result.f_hz}"

    def test_Q_identity(self) -> None:
        """Q = π f τ should hold exactly by construction."""
        result = kerr_qnm(62.0, 0.5)
        assert abs(result.Q - math.pi * result.f_hz * result.tau_s) < 1e-10


class TestAgainstNumericalAtlas:
    """Compare Berti fits against the existing qnm-package atlas (v1)."""

    @pytest.fixture()
    def numerical_atlas(self) -> list[dict]:
        atlas_path = REPO_ROOT / "docs" / "ringdown" / "atlas" / "atlas_real_v1_s4.json"
        if not atlas_path.exists():
            pytest.skip("Numerical atlas not available")
        with open(atlas_path, "r") as f:
            data = json.load(f)
        return data.get("entries", data) if isinstance(data, dict) else data

    def test_kerr_220_f_agreement(self, numerical_atlas: list[dict]) -> None:
        """Berti fits should agree with Leaver numerics within 5% for (2,2,0)."""
        kerr_220 = [e for e in numerical_atlas
                     if e.get("theory") == "GR_Kerr"
                     and e["metadata"].get("mode") == "(2,2,0)"]
        assert len(kerr_220) > 0, "No Kerr (2,2,0) entries in numerical atlas"

        max_rel_err_f = 0.0
        max_rel_err_Q = 0.0

        for entry in kerr_220:
            chi = entry["metadata"]["spin"]
            M = entry["metadata"]["M_remnant_Msun"]
            fit = kerr_qnm(M, chi, (2, 2, 0))

            rel_f = abs(fit.f_hz - entry["f_hz"]) / entry["f_hz"]
            rel_Q = abs(fit.Q - entry["Q"]) / entry["Q"]
            max_rel_err_f = max(max_rel_err_f, rel_f)
            max_rel_err_Q = max(max_rel_err_Q, rel_Q)

        assert max_rel_err_f < 0.05, f"f max relative error: {max_rel_err_f:.4f}"
        assert max_rel_err_Q < 0.10, f"Q max relative error: {max_rel_err_Q:.4f}"

    def test_kerr_221_f_agreement(self, numerical_atlas: list[dict]) -> None:
        """Berti fits for overtone (2,2,1) within 10%."""
        kerr_221 = [e for e in numerical_atlas
                     if e.get("theory") == "GR_Kerr"
                     and e["metadata"].get("mode") == "(2,2,1)"]
        if not kerr_221:
            pytest.skip("No (2,2,1) entries in numerical atlas")

        for entry in kerr_221:
            chi = entry["metadata"]["spin"]
            M = entry["metadata"]["M_remnant_Msun"]
            fit = kerr_qnm(M, chi, (2, 2, 1))

            rel_f = abs(fit.f_hz - entry["f_hz"]) / entry["f_hz"]
            # Overtone fits are less precise
            assert rel_f < 0.10, f"chi={chi}, f_fit={fit.f_hz:.2f}, f_num={entry['f_hz']:.2f}"


class TestMonotonicity:
    """Physical sanity: f increases with chi, Q increases with chi."""

    def test_f_increases_with_chi(self) -> None:
        M = 62.0
        prev_f = 0.0
        for chi in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
            f = kerr_qnm(M, chi).f_hz
            assert f > prev_f, f"f not monotonic at chi={chi}"
            prev_f = f

    def test_Q_increases_with_chi(self) -> None:
        prev_Q = 0.0
        for chi in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
            Q = kerr_Q(chi)
            assert Q > prev_Q, f"Q not monotonic at chi={chi}"
            prev_Q = Q

    def test_f_decreases_with_mass(self) -> None:
        """Larger mass → lower frequency (f ∝ 1/M)."""
        chi = 0.7
        prev_f = float("inf")
        for M in [10, 30, 60, 100, 200]:
            f = kerr_qnm(M, chi).f_hz
            assert f < prev_f, f"f not decreasing at M={M}"
            prev_f = f

    def test_Q_independent_of_mass(self) -> None:
        """Q depends only on chi, not M."""
        chi = 0.7
        q_vals = [kerr_qnm(M, chi).Q for M in [10, 30, 62, 100, 200]]
        for q in q_vals:
            assert abs(q - q_vals[0]) < 1e-10


class TestAlternativeGeometries:
    """Test parametric deviations from Kerr."""

    def test_edgb_shifts_at_zero_coupling(self) -> None:
        df, dt = deviation_edgb(0.7, 0.0)
        assert df == 0.0
        assert dt == 0.0

    def test_edgb_shifts_sign(self) -> None:
        """EdGB: frequency decreases, damping time increases."""
        df, dt = deviation_edgb(0.7, 0.5)
        assert df < 0, "EdGB should decrease f"
        assert dt > 0, "EdGB should increase tau"

    def test_dcs_zero_at_zero_spin(self) -> None:
        """dCS has no effect at chi=0."""
        df, dt = deviation_dcs(0.0, 1.0)
        assert df == 0.0
        assert dt == 0.0

    def test_dcs_nonzero_at_nonzero_spin(self) -> None:
        df, dt = deviation_dcs(0.7, 0.5)
        assert df != 0.0
        assert dt != 0.0

    def test_kerr_newman_shifts_sign(self) -> None:
        """KN: frequency increases with charge."""
        df, dt = deviation_kerr_newman(0.7, 0.3)
        assert df > 0, "KN should increase f"

    def test_apply_deviation_preserves_Q_relation(self) -> None:
        """Q = π f τ after applying deviation."""
        base = kerr_qnm(62.0, 0.7)
        shifted = apply_deviation(base, 0.05, -0.02)
        assert abs(shifted.Q - math.pi * shifted.f_hz * shifted.tau_s) < 1e-10


class TestMakeAtlasEntry:
    """Test atlas entry construction."""

    def test_entry_has_required_fields(self) -> None:
        result = kerr_qnm(62.0, 0.67)
        entry = make_atlas_entry("test_001", "GR_Kerr", result)
        assert "geometry_id" in entry
        assert "f_hz" in entry
        assert "Q" in entry
        assert "phi_atlas" in entry

    def test_phi_atlas_is_log(self) -> None:
        result = kerr_qnm(62.0, 0.67)
        entry = make_atlas_entry("test", "GR_Kerr", result)
        assert abs(entry["phi_atlas"][0] - math.log(result.f_hz)) < 1e-10
        assert abs(entry["phi_atlas"][1] - math.log(result.Q)) < 1e-10


class TestAtlasGeneration:
    """Test the generated atlas file."""

    @pytest.fixture()
    def atlas(self) -> dict:
        atlas_path = REPO_ROOT / "docs" / "ringdown" / "atlas" / "atlas_berti_v2.json"
        if not atlas_path.exists():
            pytest.skip("Atlas not generated yet")
        with open(atlas_path, "r") as f:
            return json.load(f)

    def test_schema_version(self, atlas: dict) -> None:
        assert atlas["schema_version"] == "basurin_atlas_v2_berti_fits"

    def test_entry_count(self, atlas: dict) -> None:
        assert atlas["n_total"] == len(atlas["entries"])
        assert atlas["n_total"] > 2000

    def test_has_kerr_and_alternatives(self, atlas: dict) -> None:
        theories = {e.get("theory") for e in atlas["entries"]}
        assert "GR_Kerr" in theories
        assert "EdGB" in theories
        assert "dCS" in theories
        assert "Kerr-Newman" in theories

    def test_all_entries_have_required_fields(self, atlas: dict) -> None:
        for i, e in enumerate(atlas["entries"]):
            assert "geometry_id" in e, f"Entry {i}: missing geometry_id"
            assert "f_hz" in e, f"Entry {i}: missing f_hz"
            assert "Q" in e, f"Entry {i}: missing Q"
            assert e["f_hz"] > 0, f"Entry {i}: f_hz <= 0"
            assert e["Q"] > 0, f"Entry {i}: Q <= 0"

    def test_gw150914_region_covered(self, atlas: dict) -> None:
        """Atlas should contain entries near GW150914 observables (f~251 Hz, Q~4)."""
        near = [e for e in atlas["entries"]
                if 200 < e["f_hz"] < 300 and 2 < e["Q"] < 8]
        assert len(near) > 20, f"Only {len(near)} entries near GW150914 region"
