from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from basurin_io import sha256_file
from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase1_geometry_h5.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_source_run(runs_root: Path, run_id: str, *, event_id: str, atlas_path: Path, final_id: str, mode220_ids: list[str], mode221_ids: list[str], common_ids: list[str], golden_ids: list[str]) -> None:
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _write_json(
        runs_root / run_id / "run_provenance.json",
        {
            "run_id": run_id,
            "invocation": {
                "event_id": event_id,
                "atlas_path": str(atlas_path),
                "atlas_sha256": sha256_file(atlas_path),
            },
        },
    )
    _write_json(
        runs_root / run_id / "s4g_mode220_geometry_filter" / "outputs" / "mode220_filter.json",
        {
            "accepted_geometry_ids": mode220_ids,
            "n_geometries_accepted": len(mode220_ids),
        },
    )
    _write_json(
        runs_root / run_id / "s4h_mode221_geometry_filter" / "outputs" / "mode221_filter.json",
        {
            "geometry_ids": mode221_ids,
            "n_passed": len(mode221_ids),
        },
    )
    _write_json(
        runs_root / run_id / "s4i_common_geometry_intersection" / "outputs" / "common_intersection.json",
        {
            "common_geometry_ids": common_ids,
            "mode221_skipped": not mode221_ids,
        },
    )
    _write_json(
        runs_root / run_id / "s4f_area_observation" / "outputs" / "area_obs.json",
        {
            "event_id": event_id,
            "observation_status": "OBSERVED",
            "area_kind": "catalog",
            "initial_total_area_lower_bound": 100.0,
            "n_common_input": len(common_ids),
            "n_area_entries": len(golden_ids),
        },
    )
    _write_json(
        runs_root / run_id / "s4j_hawking_area_filter" / "outputs" / "hawking_area_filter.json",
        {
            "area_constraint_applied": True,
            "area_obs_present": True,
            "n_common_input": len(common_ids),
            "n_missing_area_data": max(len(common_ids) - len(golden_ids), 0),
            "n_golden": len(golden_ids),
            "golden_geometry_ids": golden_ids,
        },
    )
    _write_json(
        runs_root / run_id / "s4k_event_support_region" / "outputs" / "event_support_region.json",
        {
            "event_id": event_id,
            "analysis_path": "MODE220_PLUS_HAWKING",
            "support_region_status": "SUPPORT_REGION_AVAILABLE",
            "domain_status": "UNKNOWN",
            "domain_status_source": "synthetic",
            "downstream_status": {"class": "GEOMETRY_PRESENT_BUT_NONINFORMATIVE", "reasons": []},
            "multimode_viability": {"class": "SINGLEMODE_ONLY"},
            "final_geometry_ids": [final_id],
            "n_final_geometries": 1,
        },
    )


def _run_script(repo_root: Path, run_id: str, runs_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    return subprocess.run(
        [sys.executable, str(repo_root / SCRIPT), "--run-id", run_id],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_contract_registered() -> None:
    contract = CONTRACTS.get("experiment/phase1_geometry_h5")
    assert contract is not None
    assert contract.required_inputs == ["s5_aggregate/outputs/aggregate.json"]
    assert contract.produced_outputs == [
        "outputs/phase1_geometry_cohort.h5",
        "outputs/phase1_geometry_summary.json",
    ]


def test_happy_path_writes_self_contained_h5(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "custom_runs"
    aggregate_run = "agg_phase1_h5"
    atlas_path = tmp_path / "atlas.json"

    _write_json(
        atlas_path,
        {
            "entries": [
                {
                    "geometry_id": "geom_1",
                    "theory": "Kerr",
                    "f_hz": 100.0,
                    "tau_s": 0.1,
                    "Q": 10.0,
                    "metadata": {"family": "kerr", "mode": "(2,2,0)", "M_solar": 50.0, "chi": 0.7},
                },
                {
                    "geometry_id": "geom_2",
                    "theory": "EdGB",
                    "f_hz": 120.0,
                    "tau_s": 0.08,
                    "Q": 9.0,
                    "metadata": {"family": "edgb", "mode": "(2,2,0)", "M_solar": 60.0, "chi": 0.8},
                },
                {
                    "geometry_id": "geom_3",
                    "theory": "dCS",
                    "f_hz": 140.0,
                    "tau_s": 0.06,
                    "Q": 8.0,
                    "metadata": {"family": "dcs", "mode": "(3,3,0)", "M_solar": 70.0, "chi": 0.9},
                },
            ]
        },
    )

    _make_source_run(
        runs_root,
        "src_evt_a",
        event_id="EVT_A",
        atlas_path=atlas_path,
        final_id="geom_2",
        mode220_ids=["geom_1", "geom_2"],
        mode221_ids=["geom_2"],
        common_ids=["geom_2"],
        golden_ids=["geom_2"],
    )
    _make_source_run(
        runs_root,
        "src_evt_b",
        event_id="EVT_B",
        atlas_path=atlas_path,
        final_id="geom_3",
        mode220_ids=["geom_2", "geom_3"],
        mode221_ids=[],
        common_ids=["geom_2", "geom_3"],
        golden_ids=["geom_3"],
    )

    _write_json(runs_root / aggregate_run / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _write_json(
        runs_root / aggregate_run / "s5_aggregate" / "outputs" / "aggregate.json",
        {
            "events": [
                {"event_id": "EVT_A", "run_id": "src_evt_a", "metric": "mahalanobis_log", "threshold_d2": 1.0, "n_atlas": 3},
                {"event_id": "EVT_B", "run_id": "src_evt_b", "metric": "mahalanobis_log", "threshold_d2": 1.0, "n_atlas": 3},
            ],
            "joint_posterior": {
                "joint_ranked_all": [
                    {
                        "geometry_id": "geom_2",
                        "coverage": 1.0,
                        "support_count": 1,
                        "support_fraction": 0.5,
                        "posterior_weight_joint": 0.7,
                        "delta_lnL_joint": 0.0,
                        "d2_sum": 0.1,
                        "log_likelihood_rel_joint": 0.0,
                        "d2_per_event": [0.1, 0.2],
                    },
                    {
                        "geometry_id": "geom_3",
                        "coverage": 0.5,
                        "support_count": 1,
                        "support_fraction": 0.5,
                        "posterior_weight_joint": 0.3,
                        "delta_lnL_joint": -0.4,
                        "d2_sum": 0.2,
                        "log_likelihood_rel_joint": -0.2,
                        "d2_per_event": [None, 0.3],
                    },
                ]
            },
        },
    )

    result = _run_script(repo_root, aggregate_run, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "OUT_ROOT=" in result.stdout
    assert "STAGE_DIR=" in result.stdout

    stage_dir = runs_root / aggregate_run / "experiment" / "phase1_geometry_h5"
    h5_path = stage_dir / "outputs" / "phase1_geometry_cohort.h5"
    summary_path = stage_dir / "outputs" / "phase1_geometry_summary.json"
    manifest_path = stage_dir / "manifest.json"
    stage_summary_path = stage_dir / "stage_summary.json"

    assert h5_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()
    assert stage_summary_path.exists()
    assert not (tmp_path / "runs").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_events"] == 2
    assert summary["n_atlas"] == 3
    assert summary["membership_true_counts"]["final_support_region"] == 2
    assert summary["analysis_path_counts"] == {"MODE220_PLUS_HAWKING": 2}

    stage_summary = json.loads(stage_summary_path.read_text(encoding="utf-8"))
    assert stage_summary["verdict"] == "PASS"
    assert stage_summary["results"]["n_events"] == 2

    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["schema_version"] == "experiment_phase1_geometry_h5_v1"
        assert int(h5.attrs["n_events"]) == 2
        assert int(h5.attrs["n_atlas"]) == 3
        assert [value.decode("utf-8") if isinstance(value, bytes) else value for value in h5["events"]["event_id"][...]] == ["EVT_A", "EVT_B"]
        assert [value.decode("utf-8") if isinstance(value, bytes) else value for value in h5["atlas"]["geometry_id"][...]] == ["geom_1", "geom_2", "geom_3"]

        mode220 = h5["membership"]["mode220"][...]
        final_region = h5["membership"]["final_support_region"][...]
        assert mode220.shape == (2, 3)
        assert final_region.shape == (2, 3)
        assert final_region.sum() == 2
        assert bool(mode220[0, 0]) is True
        assert bool(mode220[1, 2]) is True
        assert bool(final_region[0, 1]) is True
        assert bool(final_region[1, 2]) is True

        joint_available = h5["joint_posterior"]["available"][...]
        assert joint_available.tolist() == [False, True, True]
        raw_s4k = h5["raw_json"]["s4k_event_support_region"][...]
        assert len(raw_s4k) == 2
