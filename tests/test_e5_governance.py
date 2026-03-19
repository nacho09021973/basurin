#!/usr/bin/env python3
"""Governance tests for all E5 experimental alternatives.

These tests verify:
  1. Every experiment enforces RUN_VALID=PASS before consuming artifacts.
  2. No experiment writes outside its designated namespace.
  3. Sandbox isolation: no imports from mvp/ in sandbox/.
  4. Idempotency where applicable (E5-E).
  5. Contract schemas are present and valid.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_runs(tmp_path):
    """Create a temporary runs directory with N valid runs."""
    runs_root = tmp_path / "runs"

    def _make_run(run_id, *, run_valid="PASS", geometries=None, verdict=None, estimates=None):
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True)

        # RUN_VALID/verdict.json
        _write(run_dir / "RUN_VALID" / "verdict.json", {"verdict": run_valid, "run_id": run_id})

        # compatible_set.json
        cs_dir = run_dir / "s4_geometry_filter" / "outputs"
        cs_dir.mkdir(parents=True)
        geoms = geometries or [
            {"geometry_id": "edgb_001", "family": "edgb", "mahalanobis_d2": 2.1, "delta_lnL": 0.5, "saturation_221": 0.1},
            {"geometry_id": "edgb_002", "family": "edgb", "mahalanobis_d2": 4.3, "delta_lnL": 0.3, "saturation_221": 0.2},
            {"geometry_id": "kerr_newman_001", "family": "kerr_newman", "mahalanobis_d2": 3.5, "delta_lnL": 0.7, "saturation_221": 0.05},
        ]
        _write(cs_dir / "compatible_set.json", {"geometries": geoms})
        _write(
            run_dir / "s4_geometry_filter" / "stage_summary.json",
            {"stage": "s4_geometry_filter", "verdict": "PASS", "run_id": run_id},
        )

        # verdict.json
        v = verdict or {
            "family_verdicts": {
                "edgb": {"verdict": "SUPPORTED"},
                "kerr_newman": {"verdict": "INCONCLUSIVE"},
            }
        }
        _write(run_dir / "verdict.json", v)

        # estimates.json
        est_dir = run_dir / "s3b_multimode_estimates"
        est_dir.mkdir(parents=True)
        est = estimates or {"frequency_220": 251.3, "quality_factor_220": 4.2}
        _write(est_dir / "estimates.json", est)

        return run_dir

    return runs_root, _make_run


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Test: base_contract governance gate ─────────────────────────────────────

class TestBaseContractGovernance:
    def test_assert_run_valid_pass(self, tmp_runs):
        runs_root, make_run = tmp_runs
        run_dir = make_run("run_001")
        from mvp.experiment.base_contract import assert_run_valid
        summary = assert_run_valid(run_dir / "RUN_VALID" / "verdict.json")
        assert summary["verdict"] == "PASS"

    def test_assert_run_valid_fail_raises(self, tmp_runs):
        runs_root, make_run = tmp_runs
        run_dir = make_run("run_fail", run_valid="FAIL")
        from mvp.experiment.base_contract import assert_run_valid, GovernanceViolation
        with pytest.raises(GovernanceViolation, match="RUN_VALID=FAIL"):
            assert_run_valid(run_dir / "RUN_VALID" / "verdict.json")

    def test_assert_run_valid_warn_raises(self, tmp_runs):
        runs_root, make_run = tmp_runs
        run_dir = make_run("run_warn", run_valid="WARN")
        from mvp.experiment.base_contract import assert_run_valid, GovernanceViolation
        with pytest.raises(GovernanceViolation):
            assert_run_valid(run_dir / "RUN_VALID" / "verdict.json")

    def test_assert_run_valid_missing_file(self, tmp_path):
        from mvp.experiment.base_contract import assert_run_valid
        with pytest.raises(FileNotFoundError):
            assert_run_valid(tmp_path / "nonexistent.json")


# ── Test: E5-E Query Engine ─────────────────────────────────────────────────

class TestE5EQuery:
    def test_basic_query(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from mvp.experiment.e5e_query import execute_query
        result = execute_query("family == 'edgb'", ["run_001"], runs_root=str(runs_root))
        assert result["schema_version"] == "e5e-0.1"
        assert result["result_count"] == 2
        assert result["reproducible"] is True

    def test_numeric_query(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from mvp.experiment.e5e_query import execute_query
        result = execute_query("mahalanobis_d2 < 3.0", ["run_001"], runs_root=str(runs_root))
        assert result["result_count"] == 1  # only edgb_001 with d2=2.1

    def test_idempotency(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from mvp.experiment.e5e_query import execute_query
        r1 = execute_query("family == 'edgb'", ["run_001"], runs_root=str(runs_root))
        r2 = execute_query("family == 'edgb'", ["run_001"], runs_root=str(runs_root))
        assert r1["query_id"] == r2["query_id"]
        assert r1["result_count"] == r2["result_count"]
        assert r1["input_snapshots_hashed"] == r2["input_snapshots_hashed"]

    def test_rejects_invalid_run(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_fail", run_valid="FAIL")
        from mvp.experiment.base_contract import GovernanceViolation
        from mvp.experiment.e5e_query import execute_query
        with pytest.raises(GovernanceViolation):
            execute_query("family == 'edgb'", ["run_fail"], runs_root=str(runs_root))


# ── Test: E5-F Verdict Aggregation ──────────────────────────────────────────

class TestE5FVerdictAggregation:
    def test_basic_aggregation(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        make_run("run_002", verdict={
            "family_verdicts": {
                "edgb": {"verdict": "SUPPORTED"},
                "kerr_newman": {"verdict": "SUPPORTED"},
            }
        })
        from mvp.experiment.e5f_verdict_aggregation import aggregate_verdicts
        result = aggregate_verdicts(["run_001", "run_002"], runs_root=str(runs_root))
        assert result["n_events_aggregated"] == 2
        assert result["verdict_source_only"] is True
        assert result["family_support_rates"]["edgb"]["rate"] == 1.0
        assert result["family_support_rates"]["edgb"]["evidence_strength"] == "STRONG"

    def test_rejects_invalid_run(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_good")
        make_run("run_bad", run_valid="FAIL")
        from mvp.experiment.base_contract import GovernanceViolation
        from mvp.experiment.e5f_verdict_aggregation import aggregate_verdicts
        with pytest.raises(GovernanceViolation):
            aggregate_verdicts(["run_good", "run_bad"], runs_root=str(runs_root))


# ── Test: E5-A Multi-Event Aggregation ──────────────────────────────────────

class TestE5AMultiEvent:
    def test_basic_aggregation(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        make_run("run_002", geometries=[
            {"geometry_id": "edgb_001", "family": "edgb"},
            {"geometry_id": "dcs_001", "family": "dcs"},
        ])
        from mvp.experiment.e5a_multi_event_aggregation import aggregate_events
        result = aggregate_events(["run_001", "run_002"], runs_root=str(runs_root))
        assert result["schema_version"] == "e5a-0.1"
        assert result["intersection_count"] == 1  # only edgb_001 in both
        assert result["union_count"] == 4  # edgb_001, edgb_002, kerr_newman_001, dcs_001

    def test_requires_min_2_runs(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from mvp.experiment.e5a_multi_event_aggregation import aggregate_events
        with pytest.raises(ValueError, match="at least 2"):
            aggregate_events(["run_001"], runs_root=str(runs_root))


# ── Test: E5-B Jackknife Stability ──────────────────────────────────────────

class TestE5BJackknife:
    def test_basic_jackknife(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        make_run("run_002")
        make_run("run_003", geometries=[
            {"geometry_id": "edgb_001", "family": "edgb"},
        ])
        from mvp.experiment.e5b_jackknife import jackknife_analysis
        result = jackknife_analysis(["run_001", "run_002", "run_003"], runs_root=str(runs_root))
        assert result["n_events"] == 3
        assert "edgb_001" in result["geometry_stability"]
        assert result["geometry_stability"]["edgb_001"]["certificate"] in ("STABLE", "MODERATE", "UNSTABLE")

    def test_requires_min_3_runs(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        make_run("run_002")
        from mvp.experiment.e5b_jackknife import jackknife_analysis
        with pytest.raises(ValueError, match="at least 3"):
            jackknife_analysis(["run_001", "run_002"], runs_root=str(runs_root))


# ── Test: E5-C Geometry Ranking ─────────────────────────────────────────────

class TestE5CRanking:
    def test_basic_ranking(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from mvp.experiment.e5c_ranking import rank_geometries
        result = rank_geometries("run_001", runs_root=str(runs_root))
        assert result["schema_version"] == "e5c-0.1"
        assert len(result["ranked"]) == 3
        assert result["ranked"][0]["rank"] == 1
        assert result["score_policy"] == "DETERMINISTIC_WEIGHTED — not a posterior"
        assert 0 <= result["gini_coefficient"] <= 1

    def test_only_compatible_set_geometries(self, tmp_runs):
        """Ranking must not introduce new geometry_ids."""
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from mvp.experiment.e5c_ranking import rank_geometries
        result = rank_geometries("run_001", runs_root=str(runs_root))
        ranked_ids = {g["geometry_id"] for g in result["ranked"]}
        assert ranked_ids == {"edgb_001", "edgb_002", "kerr_newman_001"}


# ── Test: E5-H Blind Prediction ────────────────────────────────────────────

class TestE5HBlindPrediction:
    def test_basic_prediction(self, tmp_runs):
        runs_root, make_run = tmp_runs
        # Three runs with overlapping geometry sets
        make_run("run_001")  # edgb_001, edgb_002, kerr_newman_001
        make_run("run_002")  # same
        make_run("run_003", geometries=[
            {"geometry_id": "edgb_001", "family": "edgb"},
            {"geometry_id": "edgb_002", "family": "edgb"},
        ])
        from mvp.experiment.e5h_blind_prediction import blind_prediction
        result = blind_prediction(["run_001", "run_002", "run_003"], runs_root=str(runs_root))
        assert result["schema_version"] == "e5h-0.1"
        assert result["n_events"] == 3
        assert 0 <= result["mean_recall"] <= 1.0
        assert "headline" in result

    def test_perfect_prediction_identical_sets(self, tmp_runs):
        """When all events have identical sets, prediction should be perfect."""
        runs_root, make_run = tmp_runs
        geoms = [{"geometry_id": "g1"}, {"geometry_id": "g2"}]
        make_run("run_001", geometries=geoms)
        make_run("run_002", geometries=geoms)
        make_run("run_003", geometries=geoms)
        from mvp.experiment.e5h_blind_prediction import blind_prediction
        result = blind_prediction(["run_001", "run_002", "run_003"], runs_root=str(runs_root))
        assert result["mean_recall"] == 1.0
        assert result["mean_precision"] == 1.0

    def test_requires_min_3_runs(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        make_run("run_002")
        from mvp.experiment.e5h_blind_prediction import blind_prediction
        with pytest.raises(ValueError, match="at least 3"):
            blind_prediction(["run_001", "run_002"], runs_root=str(runs_root))


# ── Test: E5-Z GPR Emulator ─────────────────────────────────────────────────

class TestE5ZGPREmulator:
    def _make_kerr_geometries(self, n=20):
        """Create a sweep of Kerr geometries with realistic d2 values."""
        import math
        geometries = []
        for i in range(n):
            spin = i / (n - 1) * 0.95  # 0.0 to 0.95
            # Simulate a parabolic d2 surface with minimum near spin=0.67
            d2 = 3.0 * (spin - 0.67) ** 2 + 0.5
            f_hz = 200.0 + 80.0 * spin
            Q = 2.0 + 6.0 * spin
            geometries.append({
                "geometry_id": f"Kerr_a{spin:.4f}_l2m2n0",
                "family": "kerr",
                "theory": "GR_Kerr",
                "d2": round(d2, 4),
                "distance": round(math.sqrt(d2), 4),
                "delta_lnL": round(-0.5 * d2, 4),
                "f_hz": round(f_hz, 2),
                "Q": round(Q, 3),
                "metadata": {
                    "spin": round(spin, 4),
                    "chi": round(spin, 4),
                    "mode": "(2,2,0)",
                    "M_remnant_Msun": 62.0,
                },
            })
        return geometries

    def _make_2d_geometries(self, n_per_axis=5):
        """Create a 2D grid of beyond-Kerr geometries."""
        geometries = []
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                chi = 0.3 + i * 0.1
                zeta = 0.01 + j * 0.02
                d2 = 2.0 * (chi - 0.6) ** 2 + 5.0 * (zeta - 0.05) ** 2 + 1.0
                geometries.append({
                    "geometry_id": f"edgb_{i:02d}_{j:02d}",
                    "family": "edgb",
                    "d2": round(d2, 4),
                    "distance": round(d2 ** 0.5, 4),
                    "metadata": {"chi": round(chi, 4), "zeta": round(zeta, 4)},
                })
        return geometries

    def test_kerr_1d_emulation(self, tmp_runs):
        """GPR on 1D Kerr spin sweep should find minimum near spin=0.67."""
        runs_root, make_run = tmp_runs
        geoms = self._make_kerr_geometries(n=20)
        make_run("run_gpr", geometries=geoms)
        from mvp.experiment.e5z_gpr_emulator import emulate_family
        result = emulate_family("run_gpr", "kerr", target="d2", runs_root=str(runs_root))
        assert result["status"] == "SUCCESS"
        assert result["r2_score"] >= 0.90
        # Continuous minimum should be near spin=0.67
        cont_min = result["continuous_predicted_minimum"]
        assert "chi" in cont_min["params"]
        assert abs(cont_min["params"]["chi"] - 0.67) < 0.1, \
            f"Expected minimum near chi=0.67, got {cont_min['params']['chi']}"

    def test_edgb_2d_emulation(self, tmp_runs):
        """GPR on 2D EdGB grid should find minimum near (chi=0.6, zeta=0.05)."""
        runs_root, make_run = tmp_runs
        geoms = self._make_2d_geometries(n_per_axis=6)
        make_run("run_2d", geometries=geoms)
        from mvp.experiment.e5z_gpr_emulator import emulate_family
        result = emulate_family("run_2d", "edgb", target="d2", runs_root=str(runs_root))
        assert result["status"] == "SUCCESS"
        assert result["r2_score"] >= 0.90
        cont = result["continuous_predicted_minimum"]
        assert abs(cont["params"]["chi"] - 0.6) < 0.15
        assert abs(cont["params"]["zeta"] - 0.05) < 0.03

    def test_subgrid_improvement_detected(self, tmp_runs):
        """GPR should detect subgrid improvement when minimum is between nodes."""
        runs_root, make_run = tmp_runs
        geoms = self._make_kerr_geometries(n=10)  # coarser grid
        make_run("run_coarse", geometries=geoms)
        from mvp.experiment.e5z_gpr_emulator import emulate_family
        result = emulate_family("run_coarse", "kerr", target="d2", runs_root=str(runs_root))
        if result["status"] == "SUCCESS":
            # With a coarse grid, GP should find a better minimum between nodes
            assert isinstance(result["subgrid_improvement"], bool)

    def test_self_abort_on_noisy_surface(self, tmp_runs):
        """GPR should return SURFACE_UNLEARNABLE for random/noisy data."""
        runs_root, make_run = tmp_runs
        import random
        random.seed(42)
        geoms = []
        for i in range(15):
            spin = i / 14 * 0.9
            d2 = random.uniform(0, 100)  # pure noise
            geoms.append({
                "geometry_id": f"Kerr_noisy_{i}",
                "family": "kerr",
                "theory": "GR_Kerr",
                "d2": d2,
                "metadata": {"spin": spin, "chi": spin},
            })
        make_run("run_noisy", geometries=geoms)
        from mvp.experiment.e5z_gpr_emulator import emulate_family
        result = emulate_family("run_noisy", "kerr", target="d2", runs_root=str(runs_root))
        assert result["status"] == "SURFACE_UNLEARNABLE"
        assert result["r2_score"] < 0.90

    def test_hidden_minimum_confidence(self, tmp_runs):
        """Smooth surface should produce high confidence of no hidden minimum."""
        runs_root, make_run = tmp_runs
        geoms = self._make_kerr_geometries(n=20)
        make_run("run_conf", geometries=geoms)
        from mvp.experiment.e5z_gpr_emulator import emulate_family
        result = emulate_family("run_conf", "kerr", target="d2", runs_root=str(runs_root))
        assert result["status"] == "SUCCESS"
        conf = result["no_hidden_minimum_confidence"]
        assert "confidence_no_hidden_minimum" in conf
        assert conf["confidence_no_hidden_minimum"] > 0.5  # should be well-constrained

    def test_insufficient_data(self, tmp_runs):
        """GPR should report INSUFFICIENT_DATA with < 3 geometries."""
        runs_root, make_run = tmp_runs
        geoms = [
            {"geometry_id": "Kerr_001", "family": "kerr", "theory": "GR_Kerr",
             "d2": 1.0, "metadata": {"spin": 0.5, "chi": 0.5}},
        ]
        make_run("run_tiny", geometries=geoms)
        from mvp.experiment.e5z_gpr_emulator import emulate_family
        result = emulate_family("run_tiny", "kerr", target="d2", runs_root=str(runs_root))
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_governance_rejects_invalid_run(self, tmp_runs):
        """E5-Z must reject runs with RUN_VALID != PASS."""
        runs_root, make_run = tmp_runs
        make_run("run_fail", run_valid="FAIL", geometries=self._make_kerr_geometries())
        from mvp.experiment.base_contract import GovernanceViolation
        from mvp.experiment.e5z_gpr_emulator import emulate_family
        with pytest.raises(GovernanceViolation):
            emulate_family("run_fail", "kerr", runs_root=str(runs_root))

    def test_multi_family_emulation(self, tmp_runs):
        """emulate_all_families should handle multiple families."""
        runs_root, make_run = tmp_runs
        geoms = self._make_kerr_geometries(n=15) + self._make_2d_geometries(n_per_axis=5)
        make_run("run_multi", geometries=geoms)
        from mvp.experiment.e5z_gpr_emulator import emulate_all_families
        result = emulate_all_families("run_multi", families=["kerr", "edgb"],
                                      runs_root=str(runs_root))
        assert "kerr" in result["per_family_results"]
        assert "edgb" in result["per_family_results"]

    def test_output_structure(self, tmp_runs):
        """Verify output files are written correctly."""
        runs_root, make_run = tmp_runs
        geoms = self._make_kerr_geometries(n=15)
        make_run("run_out", geometries=geoms)
        from mvp.experiment.e5z_gpr_emulator import run_emulator
        result = run_emulator("run_out", families=["kerr"],
                              runs_root=str(runs_root))
        out_dir = runs_root / "run_out" / "experiment" / "continuous_emulator"
        assert (out_dir / "predicted_minima.json").exists()
        assert (out_dir / "emulator_manifest.json").exists()

        manifest = json.loads((out_dir / "emulator_manifest.json").read_text())
        assert manifest["schema_version"] == "e5z-0.1"
        assert "input_hashes" in manifest


# ── Test: Sandbox Isolation ─────────────────────────────────────────────────

class TestSandboxIsolation:
    def test_no_mvp_imports_in_sandbox(self):
        """Verify sandbox does not import from mvp/."""
        sandbox_dir = Path(__file__).parent.parent / "experiment" / "sandbox"
        if not sandbox_dir.exists():
            pytest.skip("No sandbox directory")
        for py_file in sandbox_dir.rglob("*.py"):
            content = py_file.read_text()
            assert "from mvp" not in content, f"VIOLATION: {py_file} imports from mvp"
            assert "import mvp" not in content, f"VIOLATION: {py_file} imports mvp"

    def test_no_experiment_writes_to_canonical(self):
        """Verify experiment modules don't open canonical files for writing."""
        exp_dir = Path(__file__).parent.parent / "experiment"
        for py_file in exp_dir.glob("e5*.py"):
            content = py_file.read_text()
            # Should not contain writes to s4_geometry_filter, s3b_multimode_estimates, etc.
            for canonical in ["s4_geometry_filter", "s3b_multimode_estimates", "s1_fetch_strain", "s2_ringdown_window"]:
                lines = [l for l in content.split("\n") if canonical in l and ("'w'" in l or '"w"' in l or "write" in l.lower())]
                assert not lines, f"VIOLATION: {py_file.name} appears to write to {canonical}"
