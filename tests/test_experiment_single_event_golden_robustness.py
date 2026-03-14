"""Tests for mvp/experiment_single_event_golden_robustness.py

All tests use tmp_path as BASURIN_RUNS_ROOT.
No network calls.  No subprocess (pure function calls preferred).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

# ── import the experiment module and its helpers ──────────────────────────────
from mvp.experiment_single_event_golden_robustness import (
    _aggregate_robustness,
    _evaluate_single_scenario,
    _inventory_source_runs,
    _load_source_inputs_for_geometry_filters,
    _parse_csv_floats,
    main,
)


# ────────────────────────────────────────────────────────────────────────────
# Synthetic run builder
# ────────────────────────────────────────────────────────────────────────────

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _make_synthetic_source_run(
    runs_root: Path,
    run_id: str,
    *,
    has_run_valid_pass: bool = True,
    # mode 220 observations
    obs_f_220: float = 220.0,
    obs_tau_220: float = 0.004,
    sigma_f_220: float = 5.0,
    sigma_tau_220: float = 0.0005,
    # mode 221 observations (None → no file created)
    obs_f_221: float | None = 200.0,
    obs_tau_221: float | None = 0.003,
    sigma_f_221: float | None = 6.0,
    sigma_tau_221: float | None = 0.0006,
    # area data (geometry_id → {area_final, area_initial})
    area_data: dict[str, dict[str, float]] | None = None,
    # s4g/s4h/s4j filter output geometry_ids (simulate prior canonical run)
    ids_220: list[str] | None = None,
    ids_221: list[str] | None = None,
    ids_golden: list[str] | None = None,
) -> None:
    """Create a minimal run directory with all required canonical inputs."""
    run_dir = runs_root / run_id

    # RUN_VALID
    verdict = "PASS" if has_run_valid_pass else "FAIL"
    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": verdict})

    # s4g canonical output (experiment reads obs values from here)
    s4g_ids = ids_220 if ids_220 is not None else ["geo_001"]
    s4g_payload: dict[str, Any] = {
        "schema_name": "golden_geometry_mode_filter",
        "schema_version": "v1",
        "run_id": run_id,
        "stage": "s4g_mode220_geometry_filter",
        "mode": "220",
        "obs_f_hz": obs_f_220,
        "obs_tau_s": obs_tau_220,
        "sigma_f_hz": sigma_f_220,
        "sigma_tau_s": sigma_tau_220,
        "chi2_threshold": 4.605,
        "geometry_ids": s4g_ids,
        "n_passed": len(s4g_ids),
        "verdict": "PASS",
    }
    _write_json(
        run_dir / "s4g_mode220_geometry_filter" / "outputs" / "mode220_filter.json",
        s4g_payload,
    )

    # s4h canonical output (mode 221; optional)
    if obs_f_221 is not None:
        s4h_ids = ids_221 if ids_221 is not None else ["geo_001"]
        s4h_payload: dict[str, Any] = {
            "schema_name": "golden_geometry_mode_filter",
            "schema_version": "v1",
            "run_id": run_id,
            "stage": "s4h_mode221_geometry_filter",
            "mode": "221",
            "obs_f_hz": obs_f_221,
            "obs_tau_s": obs_tau_221,
            "sigma_f_hz": sigma_f_221,
            "sigma_tau_s": sigma_tau_221,
            "chi2_threshold": 4.605,
            "geometry_ids": s4h_ids,
            "n_passed": len(s4h_ids),
            "verdict": "PASS",
        }
        _write_json(
            run_dir / "s4h_mode221_geometry_filter" / "outputs" / "mode221_filter.json",
            s4h_payload,
        )

    # s4j canonical output (area filter; optional)
    if area_data is not None:
        golden = ids_golden if ids_golden is not None else list(area_data.keys())[:1]
        s4j_payload: dict[str, Any] = {
            "schema_name": "golden_geometry_per_event",
            "schema_version": "v1",
            "run_id": run_id,
            "stage": "s4j_hawking_area_filter",
            "area_tolerance": 0.0,
            "area_data": area_data,
            "golden_geometry_ids": golden,
            "n_golden": len(golden),
            "verdict": "PASS",
        }
        _write_json(
            run_dir / "s4j_hawking_area_filter" / "outputs" / "hawking_area_filter.json",
            s4j_payload,
        )


def _make_minimal_atlas(geometry_ids: list[str]) -> list[dict[str, Any]]:
    """Return a list of atlas entries covering both mode 220 and mode 221.

    All geometries are placed at the same (f, tau) as the default observations,
    so they pass with the default threshold and sigma.
    """
    entries: list[dict[str, Any]] = []
    for gid in geometry_ids:
        entries.append(
            {
                "geometry_id": gid,
                "mode_220": {"f_hz": 220.0, "tau_s": 0.004},
                "mode_221": {"f_hz": 200.0, "tau_s": 0.003},
                "area_final": 1.1,
                "area_initial": 1.0,
            }
        )
    return entries


def _make_far_atlas(geometry_ids: list[str]) -> list[dict[str, Any]]:
    """Atlas entries far from the default observations (chi2 will be huge → not pass)."""
    entries: list[dict[str, Any]] = []
    for gid in geometry_ids:
        entries.append(
            {
                "geometry_id": gid,
                "mode_220": {"f_hz": 9999.0, "tau_s": 9.9},
                "mode_221": {"f_hz": 9999.0, "tau_s": 9.9},
                "area_final": 1.0,
                "area_initial": 1.0,
            }
        )
    return entries


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


class TestInventory:
    def test_inventory_rejects_run_without_run_valid_pass(self, tmp_path: Path) -> None:
        """A run without RUN_VALID PASS must appear as ineligible in inventory."""
        runs_root = tmp_path / "runs"
        run_id = "test_run_no_valid"
        _make_synthetic_source_run(runs_root, run_id, has_run_valid_pass=False)

        records = _inventory_source_runs(
            base_run_id=run_id,
            peer_run_ids=[],
            out_root=runs_root,
        )
        assert len(records) == 1
        rec = records[0]
        assert rec["run_id"] == run_id
        assert rec["eligible"] is False
        assert rec["has_run_valid_pass"] is False
        assert rec["exclusion_reason"] is not None

    def test_inventory_accepts_valid_run(self, tmp_path: Path) -> None:
        runs_root = tmp_path / "runs"
        run_id = "test_run_valid"
        _make_synthetic_source_run(runs_root, run_id, has_run_valid_pass=True)

        records = _inventory_source_runs(
            base_run_id=run_id,
            peer_run_ids=[],
            out_root=runs_root,
        )
        assert records[0]["eligible"] is True
        assert records[0]["has_run_valid_pass"] is True
        assert records[0]["exclusion_reason"] is None

    def test_inventory_includes_peer_runs(self, tmp_path: Path) -> None:
        runs_root = tmp_path / "runs"
        base_id = "run_base"
        peer_id = "run_peer"
        _make_synthetic_source_run(runs_root, base_id)
        _make_synthetic_source_run(runs_root, peer_id)

        records = _inventory_source_runs(
            base_run_id=base_id,
            peer_run_ids=[peer_id],
            out_root=runs_root,
        )
        run_ids = [r["run_id"] for r in records]
        assert base_id in run_ids
        assert peer_id in run_ids
        assert len(records) == 2


class TestScenarioEvaluation:
    def _make_atlas(self, geometry_ids: list[str]) -> list[dict[str, Any]]:
        return _make_minimal_atlas(geometry_ids)

    def test_scenario_evaluation_records_singleton_geometry(self, tmp_path: Path) -> None:
        """A scenario that yields exactly one golden geometry is recorded as singleton."""
        runs_root = tmp_path / "runs"
        run_id = "run_singleton"
        _make_synthetic_source_run(runs_root, run_id)
        inputs = _load_source_inputs_for_geometry_filters(run_id, runs_root)
        assert inputs is not None

        atlas = self._make_atlas(["geo_001"])
        result = _evaluate_single_scenario(
            source_run_id=run_id,
            source_inputs=inputs,
            atlas_entries=atlas,
            threshold_220=4.605,
            threshold_221=4.605,
            sigma_scale=1.0,
            area_tolerance=0.0,
        )

        assert result["status"] == "evaluated"
        assert result["n_golden_geometries"] == 1
        assert result["singleton_geometry_id"] == "geo_001"
        assert result["golden_geometry_ids"] == ["geo_001"]

    def test_scenario_evaluation_records_non_singleton_geometry_set(self, tmp_path: Path) -> None:
        """A scenario with multiple golden geometries has singleton_geometry_id=None."""
        runs_root = tmp_path / "runs"
        run_id = "run_multi"
        _make_synthetic_source_run(runs_root, run_id)
        inputs = _load_source_inputs_for_geometry_filters(run_id, runs_root)
        assert inputs is not None

        # Two geometries both at same point → both pass
        atlas = self._make_atlas(["geo_001", "geo_002"])
        result = _evaluate_single_scenario(
            source_run_id=run_id,
            source_inputs=inputs,
            atlas_entries=atlas,
            threshold_220=4.605,
            threshold_221=4.605,
            sigma_scale=1.0,
            area_tolerance=0.0,
        )

        assert result["status"] == "evaluated"
        assert result["n_golden_geometries"] == 2
        assert result["singleton_geometry_id"] is None
        assert sorted(result["golden_geometry_ids"]) == ["geo_001", "geo_002"]

    def test_scenario_with_no_atlas_entries_gives_empty_golden_set(self, tmp_path: Path) -> None:
        runs_root = tmp_path / "runs"
        run_id = "run_empty_atlas"
        _make_synthetic_source_run(runs_root, run_id)
        inputs = _load_source_inputs_for_geometry_filters(run_id, runs_root)
        assert inputs is not None

        result = _evaluate_single_scenario(
            source_run_id=run_id,
            source_inputs=inputs,
            atlas_entries=[],
            threshold_220=4.605,
            threshold_221=4.605,
            sigma_scale=1.0,
            area_tolerance=0.0,
        )

        assert result["status"] == "evaluated"
        assert result["n_golden_geometries"] == 0
        assert result["singleton_geometry_id"] is None

    def test_sigma_scale_affects_pass_count(self, tmp_path: Path) -> None:
        """Scaling sigma up should include more geometries (wider acceptance region)."""
        runs_root = tmp_path / "runs"
        run_id = "run_sigma_scale"
        _make_synthetic_source_run(
            runs_root, run_id,
            obs_f_220=220.0, obs_tau_220=0.004,
            sigma_f_220=1.0, sigma_tau_220=0.0001,
        )
        inputs = _load_source_inputs_for_geometry_filters(run_id, runs_root)
        assert inputs is not None

        # Entry slightly off from observation
        atlas = [
            {
                "geometry_id": "geo_near",
                "mode_220": {"f_hz": 222.0, "tau_s": 0.00402},
                "mode_221": {"f_hz": 200.0, "tau_s": 0.003},
            }
        ]

        result_tight = _evaluate_single_scenario(
            source_run_id=run_id,
            source_inputs=inputs,
            atlas_entries=atlas,
            threshold_220=4.605,
            threshold_221=4.605,
            sigma_scale=0.01,   # very tight: should fail
            area_tolerance=0.0,
        )
        result_wide = _evaluate_single_scenario(
            source_run_id=run_id,
            source_inputs=inputs,
            atlas_entries=atlas,
            threshold_220=4.605,
            threshold_221=4.605,
            sigma_scale=100.0,  # very wide: should pass
            area_tolerance=0.0,
        )

        assert result_tight["n_geometries_220"] == 0
        assert result_wide["n_geometries_220"] == 1


class TestAggregate:
    def test_aggregate_reports_no_data_when_no_scenarios_evaluated(self) -> None:
        """NO_DATA when n_evaluated == 0."""
        summary = _aggregate_robustness(
            base_run_id="run_base",
            peer_run_ids=[],
            scenarios=[],
        )
        assert summary["robustness_verdict"] == "NO_DATA"
        assert summary["robust_unique_geometry_id"] is None
        assert summary["support_fraction"] is None
        assert summary["n_scenarios_evaluated"] == 0

    def test_aggregate_reports_not_unique_when_no_singleton_scenarios(self) -> None:
        """NOT_UNIQUE when all evaluated scenarios have 0 or >1 golden geometries."""
        scenarios = [
            {
                "status": "evaluated",
                "n_golden_geometries": 2,
                "singleton_geometry_id": None,
                "golden_geometry_ids": ["geo_001", "geo_002"],
            },
            {
                "status": "evaluated",
                "n_golden_geometries": 0,
                "singleton_geometry_id": None,
                "golden_geometry_ids": [],
            },
        ]
        summary = _aggregate_robustness(
            base_run_id="run_base",
            peer_run_ids=[],
            scenarios=scenarios,
        )
        assert summary["robustness_verdict"] == "NOT_UNIQUE"
        assert summary["n_scenarios_evaluated"] == 2
        assert summary["n_scenarios_singleton"] == 0

    def test_aggregate_reports_unstable_unique_when_two_singletons_compete(self) -> None:
        """UNSTABLE_UNIQUE when two different geometries each appear as singleton."""
        scenarios = [
            {
                "status": "evaluated",
                "n_golden_geometries": 1,
                "singleton_geometry_id": "geo_A",
                "golden_geometry_ids": ["geo_A"],
            },
            {
                "status": "evaluated",
                "n_golden_geometries": 1,
                "singleton_geometry_id": "geo_B",
                "golden_geometry_ids": ["geo_B"],
            },
        ]
        summary = _aggregate_robustness(
            base_run_id="run_base",
            peer_run_ids=[],
            scenarios=scenarios,
        )
        assert summary["robustness_verdict"] == "UNSTABLE_UNIQUE"
        assert summary["robust_unique_geometry_id"] is None

    def test_aggregate_reports_robust_unique_when_same_singleton_dominates(self) -> None:
        """ROBUST_UNIQUE when one geometry is singleton in >= 80% of evaluated scenarios."""
        # 8 out of 10 scenarios have geo_A as singleton → fraction 0.8 → ROBUST_UNIQUE
        scenarios = (
            [
                {
                    "status": "evaluated",
                    "n_golden_geometries": 1,
                    "singleton_geometry_id": "geo_A",
                    "golden_geometry_ids": ["geo_A"],
                }
            ] * 8
            + [
                {
                    "status": "evaluated",
                    "n_golden_geometries": 0,
                    "singleton_geometry_id": None,
                    "golden_geometry_ids": [],
                }
            ] * 2
        )
        summary = _aggregate_robustness(
            base_run_id="run_base",
            peer_run_ids=[],
            scenarios=scenarios,
        )
        assert summary["robustness_verdict"] == "ROBUST_UNIQUE"
        assert summary["robust_unique_geometry_id"] == "geo_A"
        assert summary["support_fraction"] == pytest.approx(0.8)

    def test_exact_intersection_over_nonempty_scenarios_is_computed_correctly(self) -> None:
        """exact_intersection contains only geometry_ids present in every non-empty scenario."""
        scenarios = [
            {
                "status": "evaluated",
                "n_golden_geometries": 2,
                "singleton_geometry_id": None,
                "golden_geometry_ids": ["geo_A", "geo_B"],
            },
            {
                "status": "evaluated",
                "n_golden_geometries": 2,
                "singleton_geometry_id": None,
                "golden_geometry_ids": ["geo_A", "geo_C"],
            },
            {
                "status": "evaluated",
                "n_golden_geometries": 0,
                "singleton_geometry_id": None,
                "golden_geometry_ids": [],
            },
        ]
        summary = _aggregate_robustness(
            base_run_id="run_base",
            peer_run_ids=[],
            scenarios=scenarios,
        )
        # Only geo_A is in ALL non-empty sets (geo_B absent from second, geo_C absent from first)
        assert summary["exact_intersection_over_nonempty_scenarios"] == ["geo_A"]
        assert summary["n_scenarios_with_nonempty_golden_set"] == 2


class TestManifest:
    def test_manifest_contains_hashes_for_all_outputs(self, tmp_path: Path) -> None:
        """manifest.json must have SHA-256 hashes for inventory, scenario_results, robustness_summary."""
        runs_root = tmp_path / "runs"
        run_id = "run_manifest_test"
        _make_synthetic_source_run(runs_root, run_id)

        atlas = _make_minimal_atlas(["geo_001"])
        atlas_path = tmp_path / "atlas.json"
        atlas_path.write_text(json.dumps(atlas) + "\n", encoding="utf-8")

        env = os.environ.copy()
        env["BASURIN_RUNS_ROOT"] = str(runs_root)

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", run_id, "--atlas-path", str(atlas_path)])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc == 0

        exp_dir = next((runs_root / run_id / "experiment").iterdir())
        manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
        hashes = manifest.get("hashes", {})
        assert "inventory" in hashes
        assert "scenario_results" in hashes
        assert "robustness_summary" in hashes
        for h in hashes.values():
            assert isinstance(h, str) and len(h) == 64  # SHA-256 hex

    def test_manifest_file_is_present_in_experiment_dir(self, tmp_path: Path) -> None:
        runs_root = tmp_path / "runs"
        run_id = "run_manifest_present"
        _make_synthetic_source_run(runs_root, run_id)

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", run_id])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc == 0
        exp_dir = next((runs_root / run_id / "experiment").iterdir())
        assert (exp_dir / "manifest.json").exists()


class TestExperimentDir:
    def test_experiment_writes_only_under_experiment_dir(self, tmp_path: Path) -> None:
        """No files may be written outside runs/<run_id>/experiment/."""
        runs_root = tmp_path / "runs"
        run_id = "run_isolation"
        _make_synthetic_source_run(runs_root, run_id)

        atlas = _make_minimal_atlas(["geo_001"])
        atlas_path = tmp_path / "atlas.json"
        atlas_path.write_text(json.dumps(atlas) + "\n", encoding="utf-8")

        # Record files before run
        before_files = set(tmp_path.rglob("*"))

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", run_id, "--atlas-path", str(atlas_path)])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc == 0

        after_files = set(tmp_path.rglob("*"))
        new_files = [f for f in (after_files - before_files) if f.is_file()]

        exp_root = runs_root / run_id / "experiment"
        for f in new_files:
            assert str(f).startswith(str(exp_root)), (
                f"File written outside experiment dir: {f}"
            )

    def test_canonical_stage_dirs_not_mutated(self, tmp_path: Path) -> None:
        """s4g/s4h/s4i/s4j canonical outputs must not be touched by the experiment."""
        runs_root = tmp_path / "runs"
        run_id = "run_no_mutate"
        _make_synthetic_source_run(runs_root, run_id)

        # Record mtimes of canonical stage files
        s4g_out = runs_root / run_id / "s4g_mode220_geometry_filter" / "outputs" / "mode220_filter.json"
        assert s4g_out.exists()
        mtime_before = s4g_out.stat().st_mtime

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            main(["--run-id", run_id])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert s4g_out.stat().st_mtime == mtime_before, (
            "s4g canonical output was mutated by the experiment"
        )


class TestPeerRuns:
    def test_peer_runs_are_included_in_scenario_grid(self, tmp_path: Path) -> None:
        """Peer run IDs produce additional scenarios in the grid."""
        runs_root = tmp_path / "runs"
        base_id = "run_base_peer_test"
        peer_id = "run_peer_peer_test"
        _make_synthetic_source_run(runs_root, base_id)
        _make_synthetic_source_run(runs_root, peer_id, obs_f_220=215.0)

        atlas = _make_minimal_atlas(["geo_001"])
        atlas_path = tmp_path / "atlas_peer.json"
        atlas_path.write_text(json.dumps(atlas) + "\n", encoding="utf-8")

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main([
                "--run-id", base_id,
                "--peer-run-id", peer_id,
                "--atlas-path", str(atlas_path),
            ])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc == 0

        exp_dir = next((runs_root / base_id / "experiment").iterdir())
        scenario_results = json.loads((exp_dir / "scenario_results.json").read_text(encoding="utf-8"))
        scenarios = scenario_results["scenarios"]

        source_run_ids = {s["source_run_id"] for s in scenarios}
        assert base_id in source_run_ids
        assert peer_id in source_run_ids

        # With 2 runs and default single-value grid → 2 scenarios
        assert len(scenarios) == 2

    def test_peer_run_without_run_valid_is_skipped(self, tmp_path: Path) -> None:
        """Ineligible peer runs produce skipped scenarios, not crashes."""
        runs_root = tmp_path / "runs"
        base_id = "run_base_skip"
        bad_peer = "run_bad_peer"
        _make_synthetic_source_run(runs_root, base_id)
        _make_synthetic_source_run(runs_root, bad_peer, has_run_valid_pass=False)

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", base_id, "--peer-run-id", bad_peer])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc == 0

        exp_dir = next((runs_root / base_id / "experiment").iterdir())
        scenarios = json.loads((exp_dir / "scenario_results.json").read_text(encoding="utf-8"))["scenarios"]
        statuses = {s["source_run_id"]: s["status"] for s in scenarios}
        assert statuses.get(bad_peer) == "skipped"
        assert statuses.get(base_id) in {"evaluated", "skipped"}


class TestRobustnessSummary:
    def _run_main_and_read_summary(self, runs_root: Path, run_id: str, **kwargs: Any) -> dict:
        atlas = _make_minimal_atlas(["geo_001"])
        atlas_path = runs_root.parent / "atlas_rsumm.json"
        atlas_path.write_text(json.dumps(atlas) + "\n", encoding="utf-8")

        extra_args: list[str] = []
        for k, v in kwargs.items():
            extra_args += [f"--{k}", str(v)]

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", run_id, "--atlas-path", str(atlas_path)] + extra_args)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc == 0
        exp_dir = next((runs_root / run_id / "experiment").iterdir())
        return json.loads((exp_dir / "robustness_summary.json").read_text(encoding="utf-8"))

    def test_summary_has_all_required_fields(self, tmp_path: Path) -> None:
        runs_root = tmp_path / "runs"
        run_id = "run_fields"
        _make_synthetic_source_run(runs_root, run_id)
        summary = self._run_main_and_read_summary(runs_root, run_id)

        required_fields = {
            "schema_name",
            "schema_version",
            "created_utc",
            "base_run_id",
            "peer_run_ids",
            "n_source_runs",
            "n_scenarios_total",
            "n_scenarios_evaluated",
            "n_scenarios_with_nonempty_golden_set",
            "n_scenarios_singleton",
            "exact_intersection_over_nonempty_scenarios",
            "singleton_geometry_support",
            "robust_unique_geometry_id",
            "support_fraction",
            "robustness_verdict",
            "notes",
        }
        for field in required_fields:
            assert field in summary, f"Missing field: {field}"

    def test_summary_singleton_support_is_correct(self, tmp_path: Path) -> None:
        """singleton_geometry_support counts singleton occurrences per geometry_id."""
        # Default run: one geometry in atlas → singleton
        runs_root = tmp_path / "runs"
        run_id = "run_singleton_support"
        _make_synthetic_source_run(runs_root, run_id)
        summary = self._run_main_and_read_summary(runs_root, run_id)

        # With default single-value grid and one geometry, expect singleton
        support = summary["singleton_geometry_support"]
        assert isinstance(support, dict)
        # geo_001 should appear
        assert "geo_001" in support
        assert support["geo_001"] >= 1


class TestParseCSVFloats:
    def test_basic_parse(self) -> None:
        assert _parse_csv_floats("3.5,4.605,6.0") == [3.5, 4.605, 6.0]

    def test_single_value(self) -> None:
        assert _parse_csv_floats("4.605") == [4.605]

    def test_scientific_notation(self) -> None:
        result = _parse_csv_floats("1e-12,1e-9")
        assert len(result) == 2
        assert result[0] == pytest.approx(1e-12)
        assert result[1] == pytest.approx(1e-9)

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_csv_floats("")

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_csv_floats("1.0,abc,2.0")


class TestFailFast:
    def test_fails_fast_when_base_run_has_no_run_valid(self, tmp_path: Path) -> None:
        runs_root = tmp_path / "runs"
        run_id = "run_no_valid_ff"
        # Create a run dir but NO RUN_VALID file
        (runs_root / run_id).mkdir(parents=True, exist_ok=True)

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", run_id])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc != 0

    def test_fails_fast_when_base_run_has_fail_verdict(self, tmp_path: Path) -> None:
        runs_root = tmp_path / "runs"
        run_id = "run_fail_ff"
        _make_synthetic_source_run(runs_root, run_id, has_run_valid_pass=False)

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", run_id])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc != 0


class TestOutputFiles:
    def test_all_four_output_files_are_written(self, tmp_path: Path) -> None:
        """inventory.json, scenario_results.json, robustness_summary.json, manifest.json."""
        runs_root = tmp_path / "runs"
        run_id = "run_four_files"
        _make_synthetic_source_run(runs_root, run_id)

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", run_id])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc == 0
        exp_dir = next((runs_root / run_id / "experiment").iterdir())
        for fname in ("inventory.json", "scenario_results.json", "robustness_summary.json", "manifest.json"):
            assert (exp_dir / fname).exists(), f"Missing output file: {fname}"

    def test_inventory_schema_fields(self, tmp_path: Path) -> None:
        runs_root = tmp_path / "runs"
        run_id = "run_inv_schema"
        _make_synthetic_source_run(runs_root, run_id)

        old_env = os.environ.copy()
        os.environ["BASURIN_RUNS_ROOT"] = str(runs_root)
        try:
            rc = main(["--run-id", run_id])
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        assert rc == 0
        exp_dir = next((runs_root / run_id / "experiment").iterdir())
        inventory = json.loads((exp_dir / "inventory.json").read_text(encoding="utf-8"))
        assert inventory["schema_name"] == "golden_robustness_inventory"
        assert inventory["schema_version"] == "v1"
        assert "base_run_id" in inventory
        assert "peer_run_ids" in inventory
        assert "source_runs" in inventory
        assert isinstance(inventory["source_runs"], list)
