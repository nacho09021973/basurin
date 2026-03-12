from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from mvp.s5_aggregate import aggregate_compatible_sets

REPO_ROOT = Path(__file__).resolve().parent.parent
MVP_DIR = REPO_ROOT / "mvp"


def _mk_run(
    runs_root: Path,
    run_id: str,
    *,
    ranked: list[int] | None,
    compatible: list[int] | None,
    event_id: str | None = None,
    viability_class: str = "MULTIMODE_OK",
    viability_reasons: list[str] | None = None,
    s3b_verdict: str = "PASS",
    with_s3b: bool = True,
    s4i_common: list[str] | None = None,
    s4k_downstream_status: str | None = None,
    s4k_final: list[str] | None = None,
) -> None:
    event_id = event_id or run_id
    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    s4_out = runs_root / run_id / "s4_geometry_filter" / "outputs"
    s4_out.mkdir(parents=True, exist_ok=True)
    (s4_out / "compatible_set.json").write_text(json.dumps({
        "event_id": event_id,
        "metric": "euclidean_log",
        "n_atlas": 5,
        "ranked_all": [{"geometry_id": f"g{i}"} for i in range(5)],
        "compatible_geometries": [{"geometry_id": "g0", "compatible": True}],
    }), encoding="utf-8")

    s3_stage = runs_root / run_id / "s3_ringdown_estimates"
    s3_stage.mkdir(parents=True, exist_ok=True)
    (s3_stage / "stage_summary.json").write_text(json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8")

    if ranked is not None and compatible is not None:
        s6b_out = runs_root / run_id / "s6b_information_geometry_ranked" / "outputs"
        s6b_out.mkdir(parents=True, exist_ok=True)
        (s6b_out / "ranked_geometries.json").write_text(json.dumps({
            "schema_version": "mvp_s6b_ranked_v1",
            "event_id": event_id,
            "atlas_id": "unknown",
            "n_atlas": 5,
            "ranked": [{"atlas_index": i, "score": 1.0 / (1 + j)} for j, i in enumerate(ranked)],
            "compatible": [{"atlas_index": i, "score": 1.0} for i in compatible],
            "compatibility_criterion": {"name": "test", "params": {}},
        }), encoding="utf-8")

    if with_s3b:
        s3b_stage = runs_root / run_id / "s3b_multimode_estimates"
        s3b_stage.mkdir(parents=True, exist_ok=True)
        (s3b_stage / "stage_summary.json").write_text(json.dumps({
            "verdict": s3b_verdict,
            "multimode_viability": {
                "class": viability_class,
                "reasons": (
                    list(viability_reasons)
                    if viability_reasons is not None
                    else ([] if viability_class == "MULTIMODE_OK" else ["BOUNDARY_FRACTION_HIGH"])
                ),
                "metrics": {"boundary_fraction": None, "valid_fraction": {"220": 1.0, "221": 1.0}},
            }
        }), encoding="utf-8")

    if s4i_common is not None:
        s4i_out = runs_root / run_id / "s4i_common_geometry_intersection" / "outputs"
        s4i_out.mkdir(parents=True, exist_ok=True)
        (s4i_out / "common_intersection.json").write_text(json.dumps({
            "schema_name": "golden_geometry_common",
            "schema_version": "v1",
            "run_id": run_id,
            "stage": "s4i_common_geometry_intersection",
            "common_geometry_ids": list(s4i_common),
            "n_common": len(s4i_common),
            "verdict": "PASS" if s4i_common else "NO_COMMON_GEOMETRIES",
        }), encoding="utf-8")

    if s4k_downstream_status is not None:
        s4k_out = runs_root / run_id / "s4k_event_support_region" / "outputs"
        s4k_out.mkdir(parents=True, exist_ok=True)
        final_ids = list(s4k_final or [])
        (s4k_out / "event_support_region.json").write_text(json.dumps({
            "schema_name": "golden_geometry_event_support",
            "schema_version": "v1",
            "run_id": run_id,
            "stage": "s4k_event_support_region",
            "analysis_path": "MULTIMODE_INTERSECTION",
            "support_region_status": "SUPPORT_REGION_AVAILABLE" if final_ids else "NO_COMMON_REGION",
            "final_geometry_ids": final_ids,
            "downstream_status": {
                "class": s4k_downstream_status,
                "reasons": [f"seeded_for_test:{s4k_downstream_status}"],
            },
        }), encoding="utf-8")


def test_s5_aggregate_uses_s6b_and_warns_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1, 2], compatible=[1, 2], viability_class="MULTIMODE_OK")
    _mk_run(runs_root, "run_b", ranked=[1, 2, 3], compatible=[2, 3], viability_class="SINGLEMODE_ONLY")
    _mk_run(runs_root, "run_c", ranked=None, compatible=None, viability_class="RINGDOWN_NONINFORMATIVE")

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_test",
        "--source-runs",
        "run_a,run_b,run_c",
        "--top-k",
        "3",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    agg_path = runs_root / "agg_test" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))

    assert len(payload["events"]) == 3
    assert all("ranked" in ev and "compatible" in ev for ev in payload["events"])
    assert payload["n_common_geometries"] >= 0
    assert payload["n_common_compatible"] >= 0
    assert any(w.startswith("MISSING_S6B_RANKED:") for w in payload["warnings"])
    assert "NO_COMMON_COMPATIBLE_GEOMETRIES" in payload["warnings"]
    assert payload["multimode_viability"]["counts"]["MULTIMODE_OK"] == 1
    assert payload["multimode_viability"]["counts"]["SINGLEMODE_ONLY"] == 1
    assert payload["multimode_viability"]["counts"]["RINGDOWN_NONINFORMATIVE"] == 1


def test_s5_aggregate_sets_no_common_warning_only_when_data_present(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1], compatible=[0])
    _mk_run(runs_root, "run_b", ranked=[1, 2], compatible=[2])

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_test_2",
        "--source-runs",
        "run_a,run_b",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    agg_path = runs_root / "agg_test_2" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    assert "NO_COMMON_COMPATIBLE_GEOMETRIES" in payload["warnings"]


def test_s5_aggregate_fallback_to_s3_when_s3b_absent(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1], compatible=[0], with_s3b=False)

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_fallback",
        "--source-runs",
        "run_a",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    agg_path = runs_root / "agg_fallback" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    assert agg_path.exists()
    assert payload["multimode_viability"]["counts"]["SINGLEMODE_ONLY"] == 1
    assert "MISSING_S3B_UPSTREAM:run_a" in payload["warnings"]
    assert payload["multimode_viability"]["per_event"]["run_a"]["reasons"] == ["MISSING_S3B_UPSTREAM"]
    assert payload["multimode_viability"]["per_event"]["run_a"]["metrics"] == {}


def test_s5_aggregate_prefers_payload_event_id_over_run_id_suffix(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "mvp_GW200129_065458_area_strict_20260312T132701Z"
    _mk_run(
        runs_root,
        run_id,
        ranked=[0, 1],
        compatible=[0],
        event_id="GW200129_065458",
    )

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_event_id_contract",
        "--source-runs",
        run_id,
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    agg_path = runs_root / "agg_event_id_contract" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    assert payload["events"][0]["event_id"] == "GW200129_065458"


def test_s5_multimode_conditioned_population_is_insufficient_with_one_eligible_event(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1], compatible=[0], viability_class="MULTIMODE_OK")
    _mk_run(runs_root, "run_b", ranked=[0, 1], compatible=[0], viability_class="SINGLEMODE_ONLY")

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_mm_insufficient",
        "--source-runs",
        "run_a,run_b",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    stage_dir = runs_root / "agg_mm_insufficient" / "s5_aggregate"
    payload = json.loads((stage_dir / "outputs" / "aggregate.json").read_text(encoding="utf-8"))
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert payload["multimode_conditioned_population"]["status"] == "INSUFFICIENT_POPULATION"
    assert payload["multimode_conditioned_population"]["n_events_eligible"] == 1
    assert "need at least 2 eligible events" in payload["multimode_conditioned_population"]["reason"]
    assert stage_summary["results"]["multimode_conditioned_status"] == "INSUFFICIENT_POPULATION"


def test_s5_multimode_conditioned_population_is_not_supported_without_s4i_artifacts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1], compatible=[0], viability_class="MULTIMODE_OK")
    _mk_run(runs_root, "run_b", ranked=[0, 1], compatible=[0], viability_class="MULTIMODE_OK")

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_mm_not_supported",
        "--source-runs",
        "run_a,run_b",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((runs_root / "agg_mm_not_supported" / "s5_aggregate" / "outputs" / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["multimode_conditioned_population"]["status"] == "NOT_SUPPORTED"
    assert "s4i_common_geometry_intersection/outputs/common_intersection.json" in payload["multimode_conditioned_population"]["reason"]
    assert payload["multimode_conditioned_population"]["missing_common_intersection_run_ids"] == ["run_a", "run_b"]


def test_s5_multimode_conditioned_population_is_supported_when_s4i_exists_for_eligible_subset(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1], compatible=[0], viability_class="MULTIMODE_OK", s4i_common=["g0", "g1"])
    _mk_run(runs_root, "run_b", ranked=[0, 1], compatible=[0], viability_class="MULTIMODE_OK", s4i_common=["g1", "g2"])

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_mm_supported",
        "--source-runs",
        "run_a,run_b",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((runs_root / "agg_mm_supported" / "s5_aggregate" / "outputs" / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["multimode_conditioned_population"]["status"] == "SUPPORTED"
    assert payload["multimode_conditioned_population"]["artifact_basis"] == "s4i_common_geometry_intersection"
    assert payload["multimode_conditioned_population"]["common_geometry_ids"] == ["g1"]


def test_s5_multimode_conditioned_population_prefers_s4k_when_present(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(
        runs_root,
        "run_a",
        ranked=[0, 1],
        compatible=[0],
        viability_class="MULTIMODE_OK",
        s4i_common=["g0", "g1"],
        s4k_downstream_status="MULTIMODE_USABLE",
        s4k_final=["g1", "g2"],
    )
    _mk_run(
        runs_root,
        "run_b",
        ranked=[0, 1],
        compatible=[0],
        viability_class="MULTIMODE_OK",
        s4i_common=["g1", "g3"],
        s4k_downstream_status="MULTIMODE_USABLE",
        s4k_final=["g2", "g4"],
    )

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_mm_s4k_supported",
        "--source-runs",
        "run_a,run_b",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    stage_dir = runs_root / "agg_mm_s4k_supported" / "s5_aggregate"
    payload = json.loads((stage_dir / "outputs" / "aggregate.json").read_text(encoding="utf-8"))
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert payload["multimode_conditioned_population"]["status"] == "SUPPORTED"
    assert payload["multimode_conditioned_population"]["artifact_basis"] == "s4k_event_support_region"
    assert payload["multimode_conditioned_population"]["common_geometry_ids"] == ["g2"]
    assert payload["golden_geometry_support_region"]["counts"]["MULTIMODE_USABLE"] == 2
    assert payload["golden_geometry_support_region"]["per_event"]["run_a"]["downstream_status_class"] == "MULTIMODE_USABLE"
    assert stage_summary["results"]["multimode_conditioned_artifact_basis"] == "s4k_event_support_region"
    assert stage_summary["results"]["s4k_present_events"] == 2


def test_s5_multimode_conditioned_population_is_not_supported_when_no_event_has_221_enabled(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    overtone_reasons = [
        "mode_221_ok=false: overtone posterior not usable for multimode inference"
    ]
    _mk_run(
        runs_root,
        "run_a",
        ranked=[0, 1],
        compatible=[0],
        viability_class="SINGLEMODE_ONLY",
        viability_reasons=overtone_reasons,
        s3b_verdict="PASS",
    )
    _mk_run(
        runs_root,
        "run_b",
        ranked=[0, 1],
        compatible=[0],
        viability_class="SINGLEMODE_ONLY",
        viability_reasons=overtone_reasons,
        s3b_verdict="PASS",
    )

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_mm_no_221_enabled",
        "--source-runs",
        "run_a,run_b",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    stage_dir = runs_root / "agg_mm_no_221_enabled" / "s5_aggregate"
    payload = json.loads((stage_dir / "outputs" / "aggregate.json").read_text(encoding="utf-8"))
    stage_summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert payload["multimode_viability"]["counts"]["SINGLEMODE_ONLY"] == 2
    assert payload["multimode_conditioned_population"]["n_events_eligible"] == 0
    assert payload["multimode_conditioned_population"]["status"] == "NOT_SUPPORTED"
    assert stage_summary["results"]["multimode_conditioned_status"] == "NOT_SUPPORTED"


def test_s5_aggregate_require_multimode_fails_without_s3b(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _mk_run(runs_root, "run_a", ranked=[0, 1], compatible=[0], with_s3b=False)

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_fallback_strict",
        "--source-runs",
        "run_a",
        "--require-multimode",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "s3b_multimode_estimates/stage_summary.json" in proc.stderr


def test_s6b_common_geometries_match_common_compatible_when_events_match() -> None:
    source_data = [
        {
            "run_id": "run_a",
            "event_id": "event_a",
            "s6b_present": True,
            "ranked_indices": [0, 1, 2],
            "compatible_indices": [0, 1, 2],
        },
        {
            "run_id": "run_b",
            "event_id": "event_b",
            "s6b_present": True,
            "ranked_indices": [0, 1, 2],
            "compatible_indices": [0, 1, 2],
        },
    ]

    agg = aggregate_compatible_sets(source_data, min_coverage=1.0, top_k=3)

    assert all(isinstance(x, int) for x in agg["common_geometries"])
    assert set(agg["common_compatible_geometries"]) == set(agg["common_geometries"])


def test_s6b_common_compatible_uses_min_coverage_support() -> None:
    source_data: list[dict[str, object]] = []
    for i in range(22):
        compatible = [0] if i < 19 else [1]
        source_data.append(
            {
                "run_id": f"run_{i}",
                "event_id": f"event_{i}",
                "metric": "mahalanobis_log",
                "s6b_present": True,
                "ranked_indices": [0, 1],
                "compatible_indices": compatible,
                "ranked_all": [{"geometry_id": "g0", "d2": 1.0}],
            }
        )

    agg = aggregate_compatible_sets(source_data, min_coverage=0.7, top_k=5)

    assert agg["min_count"] == 16
    assert agg["common_compatible_geometries"] == [0]
    assert 0 in agg["common_compatible_geometries"]
    assert agg["n_common_compatible"] == 1
    assert "NO_COMMON_COMPATIBLE_GEOMETRIES" not in agg["warnings"]

def test_s5_aggregate_accepts_historic_compatible_set_schema_v1(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_v1"

    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    s4_out = runs_root / run_id / "s4_geometry_filter" / "outputs"
    s4_out.mkdir(parents=True, exist_ok=True)
    (s4_out / "compatible_set.json").write_text(json.dumps({
        "schema_version": "mvp_compatible_set_v1",
        "atlas_posterior": {},
        "bits_excluded": 0.0,
        "bits_kl": 0.0,
        "chi2_fixed_theta": 0.0,
        "compatible_geometries": [{"geometry_id": "g0", "compatible": True}],
        "covariance_logspace": [[1.0, 0.0], [0.0, 1.0]],
        "d2_min": 0.1,
        "distance": 0.1,
        "epsilon": 0.0,
        "event_id": "GW150914",
        "likelihood_stats": {},
        "metric": "mahalanobis_log",
        "metric_params": {},
        "n_atlas": 1,
        "n_compatible": 1,
        "observables": {"f_hz": 250.0, "Q": 8.0},
        "ranked_all": [{"geometry_id": "g0", "d2": 0.1}],
        "run_id": run_id,
        "threshold_d2": 5.99,
        "diagnostic_extra": {"source": "legacy"},
    }), encoding="utf-8")

    s3_stage = runs_root / run_id / "s3_ringdown_estimates"
    s3_stage.mkdir(parents=True, exist_ok=True)
    (s3_stage / "stage_summary.json").write_text(json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8")

    s3b_stage = runs_root / run_id / "s3b_multimode_estimates"
    s3b_stage.mkdir(parents=True, exist_ok=True)
    (s3b_stage / "stage_summary.json").write_text(json.dumps({
        "multimode_viability": {"class": "MULTIMODE_OK", "reasons": [], "metrics": {}},
    }), encoding="utf-8")

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_v1",
        "--source-runs",
        run_id,
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    agg_path = runs_root / "agg_v1" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    assert payload["compatible_set_schema"]["counts"]["compatible_set_v1_canonical"] == 1
    assert payload["compatible_set_schema"]["per_event"][0]["schema_detected"] == "mvp_compatible_set_v1"
    assert payload["compatible_set_schema"]["per_event"][0]["schema_normalized"] == "compatible_set_v1_canonical"


def test_s5_aggregate_accepts_legacy_int_schema_version(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_legacy_int"

    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    s4_out = runs_root / run_id / "s4_geometry_filter" / "outputs"
    s4_out.mkdir(parents=True, exist_ok=True)
    (s4_out / "compatible_set.json").write_text(json.dumps({
        "schema_version": 1,
        "event_id": "GW150914",
        "compatible_geometries": [{"geometry_id": "g0", "compatible": True}],
        "ranked_all": [{"geometry_id": "g0", "d2": 0.1}],
    }), encoding="utf-8")

    s3_stage = runs_root / run_id / "s3_ringdown_estimates"
    s3_stage.mkdir(parents=True, exist_ok=True)
    (s3_stage / "stage_summary.json").write_text(json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8")

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_legacy_int",
        "--source-runs",
        run_id,
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    agg_path = runs_root / "agg_legacy_int" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    row = payload["compatible_set_schema"]["per_event"][0]
    assert row["schema_detected"] == 1
    assert row["schema_normalized"] == "compatible_set_v1"


def test_s5_aggregate_does_not_fail_schema_when_compatible_geometries_empty(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_empty_compatible"

    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    s4_out = runs_root / run_id / "s4_geometry_filter" / "outputs"
    s4_out.mkdir(parents=True, exist_ok=True)
    (s4_out / "compatible_set.json").write_text(json.dumps({
        "schema_version": "mvp_compatible_set_v1",
        "event_id": "GW150914",
        "compatible_geometries": [],
        "ranked_all": [{"geometry_id": "g0", "d2": 0.1}],
    }), encoding="utf-8")

    s3_stage = runs_root / run_id / "s3_ringdown_estimates"
    s3_stage.mkdir(parents=True, exist_ok=True)
    (s3_stage / "stage_summary.json").write_text(json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8")

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_empty_compatible",
        "--source-runs",
        run_id,
        "--min-coverage",
        "1.0",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)

    assert proc.returncode == 0, proc.stderr
    assert "Invalid compatible_set schema" not in proc.stderr

    agg_path = runs_root / "agg_empty_compatible" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    assert payload["n_common_compatible"] == 0
    assert "NO_COMMON_COMPATIBLE_GEOMETRIES" in payload["warnings"]


def test_s5_aggregate_empty_compatible_set_respects_min_coverage_without_schema_error(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_empty = "run_empty"
    run_non_empty = "run_non_empty"

    for run_id in (run_empty, run_non_empty):
        rv = runs_root / run_id / "RUN_VALID"
        rv.mkdir(parents=True, exist_ok=True)
        (rv / "verdict.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

        s3_stage = runs_root / run_id / "s3_ringdown_estimates"
        s3_stage.mkdir(parents=True, exist_ok=True)
        (s3_stage / "stage_summary.json").write_text(
            json.dumps({"stage": "s3_ringdown_estimates"}), encoding="utf-8"
        )

    empty_s4 = runs_root / run_empty / "s4_geometry_filter" / "outputs"
    empty_s4.mkdir(parents=True, exist_ok=True)
    (empty_s4 / "compatible_set.json").write_text(
        json.dumps({
            "schema_version": "mvp_compatible_set_v1",
            "event_id": "GW-empty",
            "compatible_geometries": [],
            "ranked_all": [{"geometry_id": "g0", "d2": 0.1}],
        }),
        encoding="utf-8",
    )

    non_empty_s4 = runs_root / run_non_empty / "s4_geometry_filter" / "outputs"
    non_empty_s4.mkdir(parents=True, exist_ok=True)
    (non_empty_s4 / "compatible_set.json").write_text(
        json.dumps({
            "schema_version": "mvp_compatible_set_v1",
            "event_id": "GW-non-empty",
            "compatible_geometries": [{"geometry_id": "g0", "compatible": True}],
            "ranked_all": [{"geometry_id": "g0", "d2": 0.1}],
        }),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(MVP_DIR / "s5_aggregate.py"),
        "--out-run",
        "agg_min_coverage_empty",
        "--source-runs",
        f"{run_empty},{run_non_empty}",
        "--min-coverage",
        "1.0",
    ]
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)

    assert proc.returncode == 0, proc.stderr
    assert "Invalid compatible_set schema" not in proc.stderr

    agg_path = runs_root / "agg_min_coverage_empty" / "s5_aggregate" / "outputs" / "aggregate.json"
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    assert payload["n_events"] == 2
    assert payload["n_common_compatible"] == 0
    assert payload["min_coverage"] == 1.0
    assert payload["min_count"] == 2
    assert "NO_COMMON_COMPATIBLE_GEOMETRIES" in payload["warnings"]
