from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from basurin_io import write_json_atomic


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_source_run(
    runs_root: Path,
    run_id: str,
    *,
    event_id: str,
    detectors: list[str],
    candidate_basis: str,
    candidates: list[dict[str, object]],
) -> None:
    run_dir = runs_root / run_id
    write_json_atomic(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS", "reason": "synthetic"})
    write_json_atomic(
        run_dir / "s1_fetch_strain" / "stage_summary.json",
        {
            "parameters": {"event_id": event_id, "detectors": detectors},
            "results": {"detectors": detectors},
            "verdict": "PASS",
        },
    )
    write_json_atomic(
        run_dir / "s3b_multimode_estimates" / "stage_summary.json",
        {
            "multimode_viability": {
                "class": "SINGLEMODE_ONLY",
                "reasons": ["mode_221_ok=false: overtone posterior not usable for multimode inference"],
            },
            "verdict": "PASS",
        },
    )
    compatible = candidates if candidate_basis == "compatible_geometries" else []
    ranked = candidates if candidate_basis != "compatible_geometries" else candidates
    write_json_atomic(
        run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json",
        {
            "event_id": event_id,
            "compatible_geometries": compatible,
            "ranked_all": ranked,
            "n_compatible": len(compatible),
            "metric": "mahalanobis_log",
            "threshold_d2": 0.3,
        },
    )
    write_json_atomic(
        run_dir / "s8_family_router" / "outputs" / "family_router.json",
        {
            "event_id": event_id,
            "scientific_outcome": "MULTIMODE_UNAVAILABLE_221",
            "fallback_path": "220_ATLAS",
            "program_classification": "SINGLE_MODE_CONSTRAINED_PROGRAM",
            "primary_family": "GR_KERR_BH",
            "multimode_viability_class": "SINGLEMODE_ONLY",
            "multimode_viability_reasons": ["mode_221_ok=false: overtone posterior not usable for multimode inference"],
        },
    )


def test_feature_foundry_smoke_ranked_fallback_and_path_safety(tmp_path: Path) -> None:
    runs_root = tmp_path / "custom_runs"
    atlas_path = tmp_path / "atlas.json"
    catalog_path = tmp_path / "catalog.csv"

    write_json_atomic(
        atlas_path,
        {
            "entries": [
                {
                    "geometry_id": "geom_big",
                    "f_hz": 100.0,
                    "tau_s": 0.1,
                    "metadata": {"M_solar": 20.0, "chi": 0.1, "mode": "(2,2,1)", "family": "kerr", "source": "test"},
                },
                {
                    "geometry_id": "geom_small",
                    "f_hz": 200.0,
                    "tau_s": 0.05,
                    "metadata": {"M_solar": 3.0, "chi": 0.0, "mode": "(2,2,1)", "family": "kerr", "source": "test"},
                },
            ]
        },
    )
    _write_csv(
        catalog_path,
        ["event", "m1_source", "m2_source", "chi_eff"],
        [{"event": "GW_TEST", "m1_source": 4.0, "m2_source": 3.0, "chi_eff": 0.1}],
    )
    _make_source_run(
        runs_root,
        "src_ranked",
        event_id="GW_TEST_real",
        detectors=["H1", "L1"],
        candidate_basis="ranked_all_fallback",
        candidates=[
            {"geometry_id": "geom_big", "delta_lnL": 0.0, "d2": 1.0, "posterior_weight": 0.7, "prior_weight": 0.02, "compatible": False},
            {"geometry_id": "geom_small", "delta_lnL": -0.5, "d2": 2.0, "posterior_weight": 0.3, "prior_weight": 0.02, "compatible": False},
        ],
    )

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp.experiment_feature_foundry",
            "--run-id",
            "host_feature_foundry",
            "--source-runs",
            "src_ranked",
            "--atlas-path",
            str(atlas_path),
            "--catalog-path",
            str(catalog_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    stage_dir = runs_root / "host_feature_foundry" / "experiment" / "feature_foundry"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()

    event_rows = list(csv.DictReader((stage_dir / "outputs" / "event_summary.csv").open("r", encoding="utf-8")))
    assert len(event_rows) == 1
    assert event_rows[0]["candidate_basis"] == "ranked_all_fallback"
    assert event_rows[0]["detectors_used"] == "H1+L1"
    assert event_rows[0]["scientific_outcome"] == "MULTIMODE_UNAVAILABLE_221"
    assert float(event_rows[0]["initial_area_bound_min"]) < float(event_rows[0]["initial_area_bound_max"])

    candidate_rows = list(csv.DictReader((stage_dir / "outputs" / "candidate_rows.csv").open("r", encoding="utf-8")))
    assert len(candidate_rows) == 2
    geom_big = next(row for row in candidate_rows if row["candidate_id"] == "geom_big")
    geom_small = next(row for row in candidate_rows if row["candidate_id"] == "geom_small")
    assert geom_big["support_count"] == "1"
    assert geom_big["is_common_pre_hawking"] == "True"
    assert geom_big["hawking_interval_status"] == "ROBUST_PASS"
    assert geom_big["area_upper_bound_failure_pattern"] == "PASS_ALL_SEEN"
    assert geom_big["hawking_pass_zero_spin_proxy"] == "True"
    assert geom_small["hawking_pass_area_lower_bound"] == "False"
    assert geom_small["hawking_interval_status"] == "DEFINITE_FAIL"
    assert geom_small["n_fail_events_area_lower_bound"] == "1"
    assert geom_small["area_lower_bound_failure_pattern"] == "FAIL_ALL_SEEN"
    assert geom_small["hawking_pass_zero_spin_proxy"] == "False"

    posthoc = json.loads((stage_dir / "outputs" / "posthoc_checks.json").read_text(encoding="utf-8"))
    assert posthoc["summary"]["n_common_pre_hawking"] == 2
    assert posthoc["summary"]["n_common_hawking_area_lower_bound"] == 1
    assert posthoc["summary"]["n_common_hawking_area_upper_bound"] == 1
    assert posthoc["common_hawking_area_lower_bound_candidate_ids"] == ["geom_big"]
    assert posthoc["common_hawking_zero_spin_candidate_ids"] == ["geom_big"]
    assert posthoc["area_lower_bound_elimination_counts_by_event"] == {"GW_TEST": 1}
    assert posthoc["zero_spin_elimination_counts_by_event"] == {"GW_TEST": 1}

    assert not (tmp_path / "runs").exists()


def test_feature_foundry_multi_run_support_and_hawking_intersection(tmp_path: Path) -> None:
    runs_root = tmp_path / "custom_runs"
    atlas_path = tmp_path / "atlas.json"
    catalog_path = tmp_path / "catalog.csv"

    write_json_atomic(
        atlas_path,
        {
            "entries": [
                {
                    "geometry_id": "geom_a",
                    "f_hz": 90.0,
                    "tau_s": 0.08,
                    "metadata": {"M_solar": 20.0, "chi": 0.1, "mode": "(2,2,1)", "family": "kerr", "source": "test"},
                },
                {
                    "geometry_id": "geom_b",
                    "f_hz": 120.0,
                    "tau_s": 0.06,
                    "metadata": {"M_solar": 7.0, "chi": 0.0, "mode": "(2,2,1)", "family": "kerr", "source": "test"},
                },
                {
                    "geometry_id": "geom_c",
                    "f_hz": 150.0,
                    "tau_s": 0.04,
                    "metadata": {"M_solar": 5.0, "chi": 0.0, "mode": "(2,2,1)", "family": "kerr", "source": "test"},
                },
                {
                    "geometry_id": "geom_alt",
                    "f_hz": 110.0,
                    "tau_s": 0.07,
                    "metadata": {"M_solar": 20.0, "chi": 0.1, "mode": "(2,2,1)", "family": "alt", "source": "test"},
                },
            ]
        },
    )
    _write_csv(
        catalog_path,
        ["event", "m1_source", "m2_source", "chi_eff"],
        [
            {"event": "GW_EVT1", "m1_source": 3.0, "m2_source": 2.0, "chi_eff": 0.0},
            {"event": "GW_EVT2", "m1_source": 6.0, "m2_source": 5.0, "chi_eff": 0.2},
        ],
    )
    _make_source_run(
        runs_root,
        "src_one",
        event_id="GW_EVT1",
        detectors=["H1"],
        candidate_basis="compatible_geometries",
        candidates=[
            {"geometry_id": "geom_a", "delta_lnL": 0.0, "d2": 1.0, "posterior_weight": 0.5, "prior_weight": 0.02, "compatible": True},
            {"geometry_id": "geom_b", "delta_lnL": -0.2, "d2": 1.5, "posterior_weight": 0.4, "prior_weight": 0.02, "compatible": True},
            {"geometry_id": "geom_alt", "delta_lnL": -0.4, "d2": 1.7, "posterior_weight": 0.1, "prior_weight": 0.02, "compatible": True},
        ],
    )
    _make_source_run(
        runs_root,
        "src_two",
        event_id="GW_EVT2",
        detectors=["L1"],
        candidate_basis="compatible_geometries",
        candidates=[
            {"geometry_id": "geom_a", "delta_lnL": 0.0, "d2": 1.0, "posterior_weight": 0.6, "prior_weight": 0.02, "compatible": True},
            {"geometry_id": "geom_b", "delta_lnL": -0.1, "d2": 1.2, "posterior_weight": 0.3, "prior_weight": 0.02, "compatible": True},
            {"geometry_id": "geom_c", "delta_lnL": -0.8, "d2": 3.0, "posterior_weight": 0.1, "prior_weight": 0.02, "compatible": True},
            {"geometry_id": "geom_alt", "delta_lnL": -0.2, "d2": 1.4, "posterior_weight": 0.15, "prior_weight": 0.02, "compatible": True},
        ],
    )

    env = dict(os.environ)
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp.experiment_feature_foundry",
            "--run-id",
            "host_multi",
            "--source-runs",
            "src_one,src_two",
            "--atlas-path",
            str(atlas_path),
            "--catalog-path",
            str(catalog_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    stage_dir = runs_root / "host_multi" / "experiment" / "feature_foundry"
    posthoc = json.loads((stage_dir / "outputs" / "posthoc_checks.json").read_text(encoding="utf-8"))
    assert posthoc["common_pre_hawking_candidate_ids"] == ["geom_a", "geom_alt", "geom_b"]
    assert posthoc["common_hawking_area_lower_bound_candidate_ids"] == ["geom_a", "geom_alt", "geom_b"]
    assert posthoc["common_hawking_zero_spin_candidate_ids"] == ["geom_a", "geom_alt"]
    assert posthoc["zero_spin_elimination_counts_by_event"] == {"GW_EVT2": 1}
    assert posthoc["summary"]["n_common_pre_hawking_kerr_only"] == 2
    assert posthoc["kerr_only"]["common_pre_hawking_candidate_ids"] == ["geom_a", "geom_b"]
    assert posthoc["kerr_only"]["common_hawking_zero_spin_candidate_ids"] == ["geom_a"]

    candidate_rows = list(csv.DictReader((stage_dir / "outputs" / "candidate_rows.csv").open("r", encoding="utf-8")))
    geom_b_rows = [row for row in candidate_rows if row["candidate_id"] == "geom_b"]
    assert len(geom_b_rows) == 2
    assert {row["support_count"] for row in geom_b_rows} == {"2"}
    assert {row["is_common_pre_hawking"] for row in geom_b_rows} == {"True"}
    geom_b_evt2 = next(row for row in geom_b_rows if row["event_id_canonical"] == "GW_EVT2")
    assert geom_b_evt2["hawking_interval_status"] == "BOUND_SENSITIVE"
    assert geom_b_evt2["n_fail_events_area_upper_bound"] == "1"
    assert geom_b_evt2["n_fail_events_area_lower_bound"] == "0"
    assert geom_b_evt2["area_upper_bound_failure_pattern"] == "FAIL_FEW_SEEN"
    assert geom_b_evt2["area_lower_bound_failure_pattern"] == "PASS_ALL_SEEN"
