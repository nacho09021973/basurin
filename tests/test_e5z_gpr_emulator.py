#!/usr/bin/env python3
"""Governance and functional tests for E5-Z — Gaussian Process Emulator.

Tests verify:
  1. RUN_VALID=PASS enforcement.
  2. Self-abort when R² < 0.90 (SURFACE_UNLEARNABLE).
  3. Correct minimum prediction on synthetic surfaces.
  4. Deterministic results (fixed random_state).
  5. No-hidden-minimum confidence computation.
  6. Insufficient data handling (< 3 geometries).
  7. Multi-family convenience function.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def _make_edgb_atlas(n_chi=5, n_zeta=5, noise=0.0):
    """Generate synthetic EdGB atlas with a known minimum at chi=0.7, zeta=0.15.

    The surface is a simple quadratic bowl:
       d2 = 5 * (chi - 0.7)² + 20 * (zeta - 0.15)² + 1.0 + noise
    """
    rng = np.random.RandomState(42)
    geometries = []
    idx = 0
    for chi in np.linspace(0.1, 0.9, n_chi):
        for zeta in np.linspace(0.01, 0.3, n_zeta):
            d2 = 5.0 * (chi - 0.7) ** 2 + 20.0 * (zeta - 0.15) ** 2 + 1.0
            if noise > 0:
                d2 += rng.normal(0, noise)
            geometries.append({
                "geometry_id": f"edgb_{idx:03d}",
                "family": "edgb",
                "f_hz": 250.0 + chi * 10,
                "Q": 4.0 + zeta * 5,
                "d2": round(d2, 6),
                "delta_lnL": round(-0.5 * (d2 - 1.0), 6),
                "distance": round(math.sqrt(max(d2, 0)), 6),
                "compatible": d2 < 6.0,
                "metadata": {"family": "edgb", "chi": round(chi, 4), "zeta": round(zeta, 4)},
            })
            idx += 1
    return geometries


def _make_kerr_atlas(n_chi=10):
    """Generate synthetic Kerr atlas (1D: only chi)."""
    geometries = []
    for i, chi in enumerate(np.linspace(0.1, 0.95, n_chi)):
        d2 = 3.0 * (chi - 0.68) ** 2 + 0.5
        geometries.append({
            "geometry_id": f"kerr_{i:03d}",
            "family": "kerr",
            "f_hz": 250.0,
            "Q": 4.0,
            "d2": round(d2, 6),
            "distance": round(math.sqrt(d2), 6),
            "compatible": True,
            "metadata": {"family": "kerr", "chi": round(chi, 4)},
        })
    return geometries


@pytest.fixture
def edgb_run(tmp_path):
    """Create a run with a synthetic EdGB atlas on a quadratic bowl."""
    runs_root = tmp_path / "runs"
    run_id = "test_edgb_run"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)

    _write_json(run_dir / "stage_summary.json", {"run_valid": "PASS", "run_id": run_id})

    geoms = _make_edgb_atlas(n_chi=6, n_zeta=6, noise=0.0)
    cs_dir = run_dir / "s4_geometry_filter"
    cs_dir.mkdir()
    _write_json(cs_dir / "compatible_set.json", {
        "schema_version": "mvp_compatible_set_v1",
        "ranked_all": geoms,
        "compatible_geometries": [g for g in geoms if g["compatible"]],
    })

    est_dir = run_dir / "s3b_multimode_estimates"
    est_dir.mkdir()
    _write_json(est_dir / "estimates.json", {"frequency_220": 251.3, "quality_factor_220": 4.2})

    _write_json(run_dir / "verdict.json", {"family_verdicts": {"edgb": {"verdict": "SUPPORTED"}}})

    return runs_root, run_id


@pytest.fixture
def kerr_run(tmp_path):
    """Create a run with a synthetic 1D Kerr atlas."""
    runs_root = tmp_path / "runs"
    run_id = "test_kerr_run"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)

    _write_json(run_dir / "stage_summary.json", {"run_valid": "PASS", "run_id": run_id})

    geoms = _make_kerr_atlas(n_chi=10)
    cs_dir = run_dir / "s4_geometry_filter"
    cs_dir.mkdir()
    _write_json(cs_dir / "compatible_set.json", {"ranked_all": geoms, "compatible_geometries": geoms})

    est_dir = run_dir / "s3b_multimode_estimates"
    est_dir.mkdir()
    _write_json(est_dir / "estimates.json", {"frequency_220": 250.0})

    return runs_root, run_id


@pytest.fixture
def noisy_run(tmp_path):
    """Create a run with very noisy data (should trigger SURFACE_UNLEARNABLE)."""
    runs_root = tmp_path / "runs"
    run_id = "test_noisy_run"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)

    _write_json(run_dir / "stage_summary.json", {"run_valid": "PASS", "run_id": run_id})

    # Extremely noisy → GP can't learn the surface
    rng = np.random.RandomState(123)
    geoms = []
    for i in range(15):
        geoms.append({
            "geometry_id": f"edgb_{i:03d}",
            "family": "edgb",
            "d2": float(rng.uniform(0, 100)),
            "distance": 1.0,
            "compatible": True,
            "metadata": {
                "family": "edgb",
                "chi": round(float(rng.uniform(0.1, 0.9)), 4),
                "zeta": round(float(rng.uniform(0.01, 0.3)), 4),
            },
        })

    cs_dir = run_dir / "s4_geometry_filter"
    cs_dir.mkdir()
    _write_json(cs_dir / "compatible_set.json", {"ranked_all": geoms})

    est_dir = run_dir / "s3b_multimode_estimates"
    est_dir.mkdir()
    _write_json(est_dir / "estimates.json", {})

    return runs_root, run_id


@pytest.fixture
def failed_run(tmp_path):
    """Create a run with RUN_VALID=FAIL."""
    runs_root = tmp_path / "runs"
    run_id = "test_fail_run"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "stage_summary.json", {"run_valid": "FAIL", "run_id": run_id})
    cs_dir = run_dir / "s4_geometry_filter"
    cs_dir.mkdir()
    _write_json(cs_dir / "compatible_set.json", {"ranked_all": _make_edgb_atlas()})
    est_dir = run_dir / "s3b_multimode_estimates"
    est_dir.mkdir()
    _write_json(est_dir / "estimates.json", {})
    return runs_root, run_id


# ── Test: Governance gate ───────────────────────────────────────────────────

class TestGovernance:
    def test_runtime_dependencies_are_materialized_in_requirements(self):
        """Documented E5-Z runtime deps must be installable from requirements.txt."""
        repo_root = Path(__file__).resolve().parent.parent
        requirements = (repo_root / "requirements.txt").read_text().splitlines()
        pinned = {
            line.split("==", 1)[0].strip()
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#") and "==" in line
        }

        for dep in ("scikit-learn", "scipy"):
            assert dep in pinned, f"{dep} must be pinned in requirements.txt for E5-Z"

    def test_rejects_invalid_run(self, failed_run):
        runs_root, run_id = failed_run
        from mvp.experiment.base_contract import GovernanceViolation
        from mvp.experiment.e5z_gpr_emulator import emulate_family
        with pytest.raises(GovernanceViolation, match="RUN_VALID=FAIL"):
            emulate_family(run_id, "edgb", runs_root=str(runs_root))

    def test_no_writes_to_canonical(self):
        """E5-Z source code must not write to canonical directories."""
        from pathlib import Path
        src = (Path(__file__).parent.parent / "mvp" / "experiment" / "e5z_gpr_emulator.py").read_text()
        for canonical in ["s4_geometry_filter", "s3b_multimode_estimates", "s1_fetch_strain"]:
            write_lines = [
                l for l in src.split("\n")
                if canonical in l and ("'w'" in l or '"w"' in l)
            ]
            assert not write_lines, f"VIOLATION: e5z writes to {canonical}"


# ── Test: Core GP functionality ─────────────────────────────────────────────

class TestGPEmulator:
    def test_edgb_quadratic_bowl(self, edgb_run):
        """GP should find the minimum near chi=0.7, zeta=0.15 on a clean quadratic bowl."""
        runs_root, run_id = edgb_run
        from mvp.experiment.e5z_gpr_emulator import emulate_family

        result = emulate_family(run_id, "edgb", target="d2", runs_root=str(runs_root))

        assert result["status"] == "SUCCESS"
        assert result["schema_version"] == "e5z-0.1"
        assert result["r2_score"] >= 0.90

        # Continuous minimum should be close to the true minimum (0.7, 0.15)
        cont = result["continuous_predicted_minimum"]
        assert abs(cont["params"]["chi"] - 0.7) < 0.15, f"chi={cont['params']['chi']}, expected ~0.7"
        assert abs(cont["params"]["zeta"] - 0.15) < 0.05, f"zeta={cont['params']['zeta']}, expected ~0.15"

        # Interpolated d2 should be close to 1.0 (the bowl minimum)
        assert cont["interpolated_d2"] < 1.5, f"d2={cont['interpolated_d2']}, expected ~1.0"

    def test_1d_kerr(self, kerr_run):
        """GP should work on 1D parameter space (Kerr, only chi)."""
        runs_root, run_id = kerr_run
        from mvp.experiment.e5z_gpr_emulator import emulate_family

        result = emulate_family(run_id, "kerr", target="d2", runs_root=str(runs_root))

        assert result["status"] == "SUCCESS"
        assert result["r2_score"] >= 0.90
        assert len(result["param_names"]) == 1
        assert result["param_names"][0] == "chi"

        # Minimum should be near chi=0.68
        cont = result["continuous_predicted_minimum"]
        assert abs(cont["params"]["chi"] - 0.68) < 0.1

    def test_surface_unlearnable(self, noisy_run):
        """Extremely noisy data should trigger SURFACE_UNLEARNABLE."""
        runs_root, run_id = noisy_run
        from mvp.experiment.e5z_gpr_emulator import emulate_family

        result = emulate_family(run_id, "edgb", target="d2", runs_root=str(runs_root))

        assert result["status"] == "SURFACE_UNLEARNABLE"
        assert result["r2_score"] < 0.90

    def test_insufficient_data(self, tmp_path):
        """< 3 geometries should return INSUFFICIENT_DATA."""
        runs_root = tmp_path / "runs"
        run_id = "sparse_run"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "stage_summary.json", {"run_valid": "PASS"})

        geoms = [{
            "geometry_id": "edgb_000", "family": "edgb", "d2": 1.0,
            "distance": 1.0, "compatible": True,
            "metadata": {"family": "edgb", "chi": 0.5, "zeta": 0.1},
        }]
        cs_dir = run_dir / "s4_geometry_filter"
        cs_dir.mkdir()
        _write_json(cs_dir / "compatible_set.json", {"ranked_all": geoms})
        est_dir = run_dir / "s3b_multimode_estimates"
        est_dir.mkdir()
        _write_json(est_dir / "estimates.json", {})

        from mvp.experiment.e5z_gpr_emulator import emulate_family
        result = emulate_family(run_id, "edgb", runs_root=str(runs_root))
        assert result["status"] == "INSUFFICIENT_DATA"


# ── Test: Determinism ───────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_output(self, edgb_run):
        """Two runs with identical input must produce identical results."""
        runs_root, run_id = edgb_run
        from mvp.experiment.e5z_gpr_emulator import emulate_family

        r1 = emulate_family(run_id, "edgb", runs_root=str(runs_root))
        r2 = emulate_family(run_id, "edgb", runs_root=str(runs_root))

        assert r1["r2_score"] == r2["r2_score"]
        assert r1["continuous_predicted_minimum"] == r2["continuous_predicted_minimum"]
        assert r1["discrete_best_geometry"] == r2["discrete_best_geometry"]


# ── Test: No-hidden-minimum confidence ──────────────────────────────────────

class TestHiddenMinimumConfidence:
    def test_confidence_present(self, edgb_run):
        """Result must contain hidden minimum confidence."""
        runs_root, run_id = edgb_run
        from mvp.experiment.e5z_gpr_emulator import emulate_family

        result = emulate_family(run_id, "edgb", runs_root=str(runs_root))
        conf = result["no_hidden_minimum_confidence"]

        assert "confidence_no_hidden_minimum" in conf
        assert "confidence_level" in conf
        assert 0.0 <= conf["confidence_no_hidden_minimum"] <= 1.0

    def test_clean_bowl_high_confidence(self, edgb_run):
        """A clean quadratic bowl should give high confidence."""
        runs_root, run_id = edgb_run
        from mvp.experiment.e5z_gpr_emulator import emulate_family

        result = emulate_family(run_id, "edgb", runs_root=str(runs_root))
        conf = result["no_hidden_minimum_confidence"]

        # A perfectly smooth bowl with well-sampled grid → GP should be confident
        # that no significantly deeper minimum is hidden
        assert conf["confidence_no_hidden_minimum"] > 0.50
        # Relative uncertainty should be small for a smooth surface
        assert conf["relative_uncertainty"] < 1.0


# ── Test: Multi-family ──────────────────────────────────────────────────────

class TestMultiFamily:
    def test_emulate_all(self, edgb_run):
        """emulate_all_families should handle families without data gracefully."""
        runs_root, run_id = edgb_run
        from mvp.experiment.e5z_gpr_emulator import emulate_all_families

        result = emulate_all_families(
            run_id, families=["edgb", "dcs"], runs_root=str(runs_root)
        )

        assert result["per_family_results"]["edgb"]["status"] == "SUCCESS"
        # dcs has no geometries → INSUFFICIENT_DATA
        assert result["per_family_results"]["dcs"]["status"] == "INSUFFICIENT_DATA"


# ── Test: Surface grid output ───────────────────────────────────────────────

class TestSurfaceGrid:
    def test_grid_dimensions(self, edgb_run):
        """Surface grid should have correct dimensions."""
        runs_root, run_id = edgb_run
        from mvp.experiment.e5z_gpr_emulator import emulate_family

        result = emulate_family(
            run_id, "edgb", grid_resolution=10, runs_root=str(runs_root)
        )

        grid = result["surface_grid"]
        assert grid["resolution"] == 10
        assert len(grid["grid"]["chi"]) == 10
        assert len(grid["grid"]["zeta"]) == 10
        assert len(grid["predicted_d2"]) == 10
        assert len(grid["predicted_d2"][0]) == 10

    def test_1d_grid(self, kerr_run):
        """1D family should produce 1D grid."""
        runs_root, run_id = kerr_run
        from mvp.experiment.e5z_gpr_emulator import emulate_family

        result = emulate_family(
            run_id, "kerr", grid_resolution=20, runs_root=str(runs_root)
        )

        grid = result["surface_grid"]
        assert len(grid["param_names"]) == 1
        assert len(grid["grid"]["chi"]) == 20
        assert len(grid["predicted_d2"]) == 20


# ── Test: File output ───────────────────────────────────────────────────────

class TestFileOutput:
    def test_write_outputs(self, edgb_run):
        """run_emulator should write all expected files."""
        runs_root, run_id = edgb_run
        from mvp.experiment.e5z_gpr_emulator import run_emulator

        run_emulator(
            run_id, families=["edgb"],
            grid_resolution=5,
            runs_root=str(runs_root),
        )

        out_dir = runs_root / run_id / "experiment" / "continuous_emulator"
        assert (out_dir / "predicted_minima.json").exists()
        assert (out_dir / "gpr_surface_edgb.json").exists()
        assert (out_dir / "validation_residuals_edgb.json").exists()
        assert (out_dir / "emulator_manifest.json").exists()

        # Verify manifest content
        manifest = json.loads((out_dir / "emulator_manifest.json").read_text())
        assert manifest["schema_version"] == "e5z-0.1"
        assert "edgb" in manifest["families"]
        assert manifest["per_family_status"]["edgb"] == "SUCCESS"
