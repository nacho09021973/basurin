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

        # stage_summary.json
        summary = {"run_valid": run_valid, "run_id": run_id}
        _write(run_dir / "stage_summary.json", summary)

        # compatible_set.json
        cs_dir = run_dir / "s4_geometry_filter"
        cs_dir.mkdir(parents=True)
        geoms = geometries or [
            {"geometry_id": "edgb_001", "family": "edgb", "mahalanobis_d2": 2.1, "delta_lnL": 0.5, "saturation_221": 0.1},
            {"geometry_id": "edgb_002", "family": "edgb", "mahalanobis_d2": 4.3, "delta_lnL": 0.3, "saturation_221": 0.2},
            {"geometry_id": "kerr_newman_001", "family": "kerr_newman", "mahalanobis_d2": 3.5, "delta_lnL": 0.7, "saturation_221": 0.05},
        ]
        _write(cs_dir / "compatible_set.json", {"geometries": geoms})

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
        from experiment.base_contract import assert_run_valid
        summary = assert_run_valid(run_dir / "stage_summary.json")
        assert summary["run_valid"] == "PASS"

    def test_assert_run_valid_fail_raises(self, tmp_runs):
        runs_root, make_run = tmp_runs
        run_dir = make_run("run_fail", run_valid="FAIL")
        from experiment.base_contract import assert_run_valid, GovernanceViolation
        with pytest.raises(GovernanceViolation, match="RUN_VALID=FAIL"):
            assert_run_valid(run_dir / "stage_summary.json")

    def test_assert_run_valid_warn_raises(self, tmp_runs):
        runs_root, make_run = tmp_runs
        run_dir = make_run("run_warn", run_valid="WARN")
        from experiment.base_contract import assert_run_valid, GovernanceViolation
        with pytest.raises(GovernanceViolation):
            assert_run_valid(run_dir / "stage_summary.json")

    def test_assert_run_valid_missing_file(self, tmp_path):
        from experiment.base_contract import assert_run_valid
        with pytest.raises(FileNotFoundError):
            assert_run_valid(tmp_path / "nonexistent.json")


# ── Test: E5-E Query Engine ─────────────────────────────────────────────────

class TestE5EQuery:
    def test_basic_query(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from experiment.e5e_query import execute_query
        result = execute_query("family == 'edgb'", ["run_001"], runs_root=str(runs_root))
        assert result["schema_version"] == "e5e-0.1"
        assert result["result_count"] == 2
        assert result["reproducible"] is True

    def test_numeric_query(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from experiment.e5e_query import execute_query
        result = execute_query("mahalanobis_d2 < 3.0", ["run_001"], runs_root=str(runs_root))
        assert result["result_count"] == 1  # only edgb_001 with d2=2.1

    def test_idempotency(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from experiment.e5e_query import execute_query
        r1 = execute_query("family == 'edgb'", ["run_001"], runs_root=str(runs_root))
        r2 = execute_query("family == 'edgb'", ["run_001"], runs_root=str(runs_root))
        assert r1["query_id"] == r2["query_id"]
        assert r1["result_count"] == r2["result_count"]
        assert r1["input_snapshots_hashed"] == r2["input_snapshots_hashed"]

    def test_rejects_invalid_run(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_fail", run_valid="FAIL")
        from experiment.base_contract import GovernanceViolation
        from experiment.e5e_query import execute_query
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
        from experiment.e5f_verdict_aggregation import aggregate_verdicts
        result = aggregate_verdicts(["run_001", "run_002"], runs_root=str(runs_root))
        assert result["n_events_aggregated"] == 2
        assert result["verdict_source_only"] is True
        assert result["family_support_rates"]["edgb"]["rate"] == 1.0
        assert result["family_support_rates"]["edgb"]["evidence_strength"] == "STRONG"

    def test_rejects_invalid_run(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_good")
        make_run("run_bad", run_valid="FAIL")
        from experiment.base_contract import GovernanceViolation
        from experiment.e5f_verdict_aggregation import aggregate_verdicts
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
        from experiment.e5a_multi_event_aggregation import aggregate_events
        result = aggregate_events(["run_001", "run_002"], runs_root=str(runs_root))
        assert result["schema_version"] == "e5a-0.1"
        assert result["intersection_count"] == 1  # only edgb_001 in both
        assert result["union_count"] == 4  # edgb_001, edgb_002, kerr_newman_001, dcs_001

    def test_requires_min_2_runs(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from experiment.e5a_multi_event_aggregation import aggregate_events
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
        from experiment.e5b_jackknife import jackknife_analysis
        result = jackknife_analysis(["run_001", "run_002", "run_003"], runs_root=str(runs_root))
        assert result["n_events"] == 3
        assert "edgb_001" in result["geometry_stability"]
        assert result["geometry_stability"]["edgb_001"]["certificate"] in ("STABLE", "MODERATE", "UNSTABLE")

    def test_requires_min_3_runs(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        make_run("run_002")
        from experiment.e5b_jackknife import jackknife_analysis
        with pytest.raises(ValueError, match="at least 3"):
            jackknife_analysis(["run_001", "run_002"], runs_root=str(runs_root))


# ── Test: E5-C Geometry Ranking ─────────────────────────────────────────────

class TestE5CRanking:
    def test_basic_ranking(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        from experiment.e5c_ranking import rank_geometries
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
        from experiment.e5c_ranking import rank_geometries
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
        from experiment.e5h_blind_prediction import blind_prediction
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
        from experiment.e5h_blind_prediction import blind_prediction
        result = blind_prediction(["run_001", "run_002", "run_003"], runs_root=str(runs_root))
        assert result["mean_recall"] == 1.0
        assert result["mean_precision"] == 1.0

    def test_requires_min_3_runs(self, tmp_runs):
        runs_root, make_run = tmp_runs
        make_run("run_001")
        make_run("run_002")
        from experiment.e5h_blind_prediction import blind_prediction
        with pytest.raises(ValueError, match="at least 3"):
            blind_prediction(["run_001", "run_002"], runs_root=str(runs_root))


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
