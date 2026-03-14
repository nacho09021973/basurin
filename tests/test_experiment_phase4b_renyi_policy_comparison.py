from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from basurin_io import sha256_file
from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase4b_renyi_policy_comparison.py")
STAGE = "experiment/phase4b_renyi_policy_comparison"
SCHEMA_VERSION = "renyi_policy_comparison_v1"
COMPARISON_ROLE = "epistemic_policy_comparison"
ALPHA_GRID = [0, 1, 2, "inf"]
K_GRID = [1, 3, 5, 10]
TOP_N = 20

POLICY_SPECS = [
    {
        "policy_name": "uniform_support_v1",
        "weight_policy_file": "weight_policy_uniform_support_v1.json",
        "renyi_diversity_file": "renyi_diversity_uniform_support_v1.json",
    },
    {
        "policy_name": "event_frequency_support_v1",
        "weight_policy_file": "weight_policy_event_frequency_support_v1.json",
        "renyi_diversity_file": "renyi_diversity_event_frequency_support_v1.json",
    },
    {
        "policy_name": "event_support_delta_lnL_softmax_mean_v1",
        "weight_policy_file": "weight_policy_event_support_delta_lnL_softmax_mean_v1.json",
        "renyi_diversity_file": "renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json",
    },
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_valid(runs_root: Path, run_id: str, verdict: str = "PASS") -> None:
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": verdict})


def _renyi_metrics(weights: list[float]) -> dict:
    positive = [weight for weight in weights if weight > 0.0]
    h0 = math.log(float(len(positive)))
    h1 = -math.fsum(weight * math.log(weight) for weight in positive)
    h2 = -math.log(math.fsum(weight * weight for weight in positive))
    hinf = -math.log(max(positive))
    return {
        "H_alpha": {"0": h0, "1": h1, "2": h2, "inf": hinf},
        "D_alpha": {
            "0": math.exp(h0),
            "1": math.exp(h1),
            "2": math.exp(h2),
            "inf": math.exp(hinf),
        },
        "p_max": max(positive),
        "n_weighted": len(positive),
        "n_unweighted": len(weights) - len(positive),
        "weight_sum_normalized": float(sum(weights)),
    }


def _geometry_rows(weights: list[float], *, policy_name: str, geometry_ids: list[str] | None = None) -> list[dict]:
    geometry_specs = [
        ("geo_a", "geo_a_norm", "edgb", "EdGB"),
        ("geo_b", "geo_b_norm", "edgb", "EdGB"),
        ("geo_c", "geo_c_norm", "kerr_newman", "Kerr-Newman"),
        ("geo_d", "geo_d_norm", "dcs", "dCS"),
    ]
    rows: list[dict] = []
    for idx, weight in enumerate(weights):
        raw_geometry_id, normalized_geometry_id, family, theory = geometry_specs[idx]
        if geometry_ids is not None:
            raw_geometry_id = geometry_ids[idx]
            normalized_geometry_id = f"{geometry_ids[idx]}_norm"
        rows.append(
            {
                "raw_geometry_id": raw_geometry_id,
                "normalized_geometry_id": normalized_geometry_id,
                "atlas_family": family,
                "atlas_theory": theory,
                "policy_name": policy_name,
                "weight_raw": weight,
                "weight_normalized": weight,
                "weight_status": "WEIGHTED",
                "support_count_events": idx + 1,
                "support_fraction_events": (idx + 1) / 47.0,
                "source_artifacts": ["experiment/phase3_weight_policy_basis/outputs/example.json"],
                "criterion": "fixture_v1",
                "criterion_version": "v1",
                "evidence": {"fixture_index": idx},
            }
        )
    return rows


def _weight_policy_payload(
    weights: list[float],
    *,
    policy_name: str,
    basis_name: str = "final_support_region_union_v1",
    geometry_ids: list[str] | None = None,
) -> dict:
    rows = _geometry_rows(weights, policy_name=policy_name, geometry_ids=geometry_ids)
    n_weighted = sum(1 for weight in weights if weight > 0.0)
    return {
        "schema_version": "weight_policy_basis_v1",
        "basis_name": basis_name,
        "policy_name": policy_name,
        "policy_role": "fixture",
        "coverage_fraction": 1.0,
        "n_rows": len(rows),
        "n_weighted": n_weighted,
        "n_unweighted": len(rows) - n_weighted,
        "weight_sum_raw": float(sum(weights)),
        "weight_sum_normalized": float(sum(weights)),
        "normalization_method": "fixture_v1",
        "source_policy_inputs": ["fixture"],
        "rows": rows,
    }


def _renyi_payload(
    weights: list[float],
    *,
    policy_name: str,
    weight_policy_file: str,
    basis_name: str = "final_support_region_union_v1",
) -> dict:
    return {
        "schema_version": "renyi_diversity_baseline_v1",
        "metric_role": "epistemic_ensemble_diversity",
        "basis_name": basis_name,
        "policy_name": policy_name,
        "weight_policy_file": weight_policy_file,
        "n_rows": len(weights),
        "coverage_fraction": 1.0,
        "alpha_grid": ALPHA_GRID,
        "metrics": _renyi_metrics(weights),
        "notes": [
            "Epistemic ensemble diversity over the full supported basis.",
            "No sector conditioning is applied in this baseline.",
            "This output is not a black-hole thermodynamic entropy claim.",
        ],
    }


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


def _prepare_consistent_run(
    tmp_path: Path,
    *,
    run_id: str,
    missing_artifact: str | None = None,
    basis_name_overrides: dict[str, str] | None = None,
    geometry_id_overrides: dict[str, list[str]] | None = None,
    n_rows_truncate_policy: str | None = None,
) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    _write_run_valid(runs_root, run_id)

    weights_by_policy = {
        "uniform_support_v1": [0.25, 0.25, 0.25, 0.25],
        "event_frequency_support_v1": [0.4, 0.3, 0.2, 0.1],
        "event_support_delta_lnL_softmax_mean_v1": [0.7, 0.1, 0.1, 0.1],
    }
    basis_name_overrides = basis_name_overrides or {}
    geometry_id_overrides = geometry_id_overrides or {}

    for spec in POLICY_SPECS:
        policy_name = spec["policy_name"]
        weights = list(weights_by_policy[policy_name])
        if n_rows_truncate_policy == policy_name:
            weights = [0.5, 0.3, 0.2]
        basis_name = basis_name_overrides.get(policy_name, "final_support_region_union_v1")
        geometry_ids = geometry_id_overrides.get(policy_name)
        weight_payload = _weight_policy_payload(
            weights,
            policy_name=policy_name,
            basis_name=basis_name,
            geometry_ids=geometry_ids,
        )
        renyi_payload = _renyi_payload(
            weights,
            policy_name=policy_name,
            weight_policy_file=spec["weight_policy_file"],
            basis_name=basis_name,
        )

        weight_path = runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "outputs" / spec["weight_policy_file"]
        renyi_path = runs_root / run_id / "experiment" / "phase4_renyi_diversity_baseline" / "outputs" / spec["renyi_diversity_file"]

        if missing_artifact != spec["weight_policy_file"]:
            _write_json(weight_path, weight_payload)
        if missing_artifact != spec["renyi_diversity_file"]:
            _write_json(renyi_path, renyi_payload)

    return repo_root, runs_root


def _normalized_stage_summary(payload: dict, *, input_records: list[dict], output_records: list[dict]) -> dict:
    return {
        "alpha_grid": payload["alpha_grid"],
        "basis_name": payload["basis_name"],
        "comparison_role": payload["comparison_role"],
        "created": "<TIMESTAMP>",
        "family_count": payload["family_count"],
        "inputs": input_records,
        "k_grid": payload["k_grid"],
        "n_rows": payload["n_rows"],
        "outputs": output_records,
        "parameters": payload["parameters"],
        "policies": payload["policies"],
        "results": payload["results"],
        "run": payload["run"],
        "runs_root": "<RUNS_ROOT>",
        "schema_version": payload["schema_version"],
        "stage": payload["stage"],
        "theory_count": payload["theory_count"],
        "top_n": payload["top_n"],
        "verdict": payload["verdict"],
        "version": payload["version"],
    }


def _sorted_weight_rows(weights: list[float]) -> list[dict[str, Any]]:
    rows = _geometry_rows(weights, policy_name="fixture")
    return sorted(rows, key=lambda row: (-float(row["weight_normalized"]), row["raw_geometry_id"]))


def _expected_topk_rows(weights_by_policy: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in POLICY_SPECS:
        policy_name = spec["policy_name"]
        sorted_rows = _sorted_weight_rows(weights_by_policy[policy_name])
        total_mass = math.fsum(float(row["weight_normalized"]) for row in sorted_rows)
        for top_k in K_GRID:
            rows.append(
                {
                    "policy_name": policy_name,
                    "top_k": top_k,
                    "cumulative_mass": math.fsum(float(row["weight_normalized"]) for row in sorted_rows[:top_k]),
                    "normalization_check": total_mass,
                    "criterion": "cumulative_topk_mass_over_weight_normalized",
                    "criterion_version": "v1",
                }
            )
    return rows


def _expected_bucket_rows(
    weights_by_policy: dict[str, list[float]],
    *,
    bucket_key: str,
    mass_field: str,
    count_field: str,
    rank_field: str,
    criterion: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in POLICY_SPECS:
        policy_name = spec["policy_name"]
        sorted_rows = _sorted_weight_rows(weights_by_policy[policy_name])
        bucket_mass: dict[str, float] = {}
        bucket_count: dict[str, int] = {}
        for row in sorted_rows:
            key = str(row[bucket_key])
            bucket_mass[key] = bucket_mass.get(key, 0.0) + float(row["weight_normalized"])
            bucket_count[key] = bucket_count.get(key, 0) + 1
        ordered = sorted(bucket_mass.items(), key=lambda item: (-item[1], item[0]))
        for rank, (bucket, mass) in enumerate(ordered, start=1):
            rows.append(
                {
                    "policy_name": policy_name,
                    bucket_key: bucket,
                    mass_field: mass,
                    count_field: bucket_count[bucket],
                    rank_field: rank,
                    "criterion": criterion,
                    "criterion_version": "v1",
                }
            )
    return rows


def _expected_ranking_rows(weights_by_policy: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in POLICY_SPECS:
        policy_name = spec["policy_name"]
        sorted_rows = _sorted_weight_rows(weights_by_policy[policy_name])
        cumulative_mass = 0.0
        for rank, row in enumerate(sorted_rows[:TOP_N], start=1):
            cumulative_mass += float(row["weight_normalized"])
            rows.append(
                {
                    "policy_name": policy_name,
                    "rank": rank,
                    "raw_geometry_id": row["raw_geometry_id"],
                    "normalized_geometry_id": row["normalized_geometry_id"],
                    "atlas_family": row["atlas_family"],
                    "atlas_theory": row["atlas_theory"],
                    "weight_normalized": float(row["weight_normalized"]),
                    "cumulative_mass": cumulative_mass,
                    "criterion": "weight_normalized_descending_cumulative_mass",
                    "criterion_version": "v1",
                }
            )
    return rows


def test_contract_registered() -> None:
    contract = CONTRACTS.get(STAGE)
    assert contract is not None
    assert contract.required_inputs == [
        "experiment/phase4_renyi_diversity_baseline/outputs/renyi_diversity_uniform_support_v1.json",
        "experiment/phase4_renyi_diversity_baseline/outputs/renyi_diversity_event_frequency_support_v1.json",
        "experiment/phase4_renyi_diversity_baseline/outputs/renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json",
        "experiment/phase3_weight_policy_basis/outputs/weight_policy_uniform_support_v1.json",
        "experiment/phase3_weight_policy_basis/outputs/weight_policy_event_frequency_support_v1.json",
        "experiment/phase3_weight_policy_basis/outputs/weight_policy_event_support_delta_lnL_softmax_mean_v1.json",
    ]
    assert contract.produced_outputs == [
        "outputs/renyi_policy_comparison_v1.json",
        "outputs/topk_mass_by_policy_v1.json",
        "outputs/family_mass_by_policy_v1.json",
        "outputs/theory_mass_by_policy_v1.json",
        "outputs/top_geometry_ranking_by_policy_v1.json",
    ]


def test_happy_path_writes_comparison_outputs_and_normalized_snapshot(tmp_path: Path) -> None:
    run_id = "phase4b_happy_path"
    repo_root, runs_root = _prepare_consistent_run(tmp_path, run_id=run_id)
    weights_by_policy = {
        "uniform_support_v1": [0.25, 0.25, 0.25, 0.25],
        "event_frequency_support_v1": [0.4, 0.3, 0.2, 0.1],
        "event_support_delta_lnL_softmax_mean_v1": [0.7, 0.1, 0.1, 0.1],
    }

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / STAGE
    comparison_path = stage_dir / "outputs" / "renyi_policy_comparison_v1.json"
    topk_path = stage_dir / "outputs" / "topk_mass_by_policy_v1.json"
    family_path = stage_dir / "outputs" / "family_mass_by_policy_v1.json"
    theory_path = stage_dir / "outputs" / "theory_mass_by_policy_v1.json"
    ranking_path = stage_dir / "outputs" / "top_geometry_ranking_by_policy_v1.json"
    summary_path = stage_dir / "stage_summary.json"

    comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    topk_payload = json.loads(topk_path.read_text(encoding="utf-8"))
    family_payload = json.loads(family_path.read_text(encoding="utf-8"))
    theory_payload = json.loads(theory_path.read_text(encoding="utf-8"))
    ranking_payload = json.loads(ranking_path.read_text(encoding="utf-8"))

    expected_metrics_table = {}
    for spec in POLICY_SPECS:
        policy_name = spec["policy_name"]
        metrics = _renyi_metrics(weights_by_policy[policy_name])
        expected_metrics_table[policy_name] = {
            "D_0": float(metrics["D_alpha"]["0"]),
            "D_1": float(metrics["D_alpha"]["1"]),
            "D_2": float(metrics["D_alpha"]["2"]),
            "D_inf": float(metrics["D_alpha"]["inf"]),
            "p_max": float(metrics["p_max"]),
            "n_rows": 4,
            "coverage_fraction": 1.0,
            "weight_policy_file": spec["weight_policy_file"],
            "renyi_diversity_file": spec["renyi_diversity_file"],
        }
    assert comparison_payload == {
        "schema_version": SCHEMA_VERSION,
        "comparison_role": COMPARISON_ROLE,
        "basis_name": "final_support_region_union_v1",
        "policies": [spec["policy_name"] for spec in POLICY_SPECS],
        "alpha_grid": ALPHA_GRID,
        "metrics_table": expected_metrics_table,
        "notes": [
            "Comparison of epistemic ensemble diversity across weight policies.",
            "No sector conditioning is applied.",
            "This comparison is not a black-hole thermodynamic entropy claim.",
        ],
    }

    assert topk_payload == {
        "schema_version": "topk_mass_by_policy_v1",
        "basis_name": "final_support_region_union_v1",
        "k_grid": K_GRID,
        "policies": [spec["policy_name"] for spec in POLICY_SPECS],
        "rows": _expected_topk_rows(weights_by_policy),
    }

    assert family_payload == {
        "schema_version": "family_mass_by_policy_v1",
        "basis_name": "final_support_region_union_v1",
        "policies": [spec["policy_name"] for spec in POLICY_SPECS],
        "rows": _expected_bucket_rows(
            weights_by_policy,
            bucket_key="atlas_family",
            mass_field="family_mass",
            count_field="family_count",
            rank_field="family_mass_rank",
            criterion="mass_sum_by_atlas_family_over_weight_normalized",
        ),
    }
    assert theory_payload == {
        "schema_version": "theory_mass_by_policy_v1",
        "basis_name": "final_support_region_union_v1",
        "policies": [spec["policy_name"] for spec in POLICY_SPECS],
        "rows": _expected_bucket_rows(
            weights_by_policy,
            bucket_key="atlas_theory",
            mass_field="theory_mass",
            count_field="theory_count",
            rank_field="theory_mass_rank",
            criterion="mass_sum_by_atlas_theory_over_weight_normalized",
        ),
    }
    assert ranking_payload == {
        "schema_version": "top_geometry_ranking_by_policy_v1",
        "basis_name": "final_support_region_union_v1",
        "policies": [spec["policy_name"] for spec in POLICY_SPECS],
        "top_n": TOP_N,
        "rows": _expected_ranking_rows(weights_by_policy),
    }

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    input_records = []
    for spec in POLICY_SPECS:
        input_records.append(
            {
                "label": f"{spec['policy_name']}:renyi_diversity",
                "path": f"experiment/phase4_renyi_diversity_baseline/outputs/{spec['renyi_diversity_file']}",
                "sha256": sha256_file(runs_root / run_id / "experiment" / "phase4_renyi_diversity_baseline" / "outputs" / spec["renyi_diversity_file"]),
            }
        )
        input_records.append(
            {
                "label": f"{spec['policy_name']}:weight_policy",
                "path": f"experiment/phase3_weight_policy_basis/outputs/{spec['weight_policy_file']}",
                "sha256": sha256_file(runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "outputs" / spec["weight_policy_file"]),
            }
        )
    output_records = [
        {"path": f"{STAGE}/outputs/renyi_policy_comparison_v1.json", "sha256": sha256_file(comparison_path)},
        {"path": f"{STAGE}/outputs/topk_mass_by_policy_v1.json", "sha256": sha256_file(topk_path)},
        {"path": f"{STAGE}/outputs/family_mass_by_policy_v1.json", "sha256": sha256_file(family_path)},
        {"path": f"{STAGE}/outputs/theory_mass_by_policy_v1.json", "sha256": sha256_file(theory_path)},
        {"path": f"{STAGE}/outputs/top_geometry_ranking_by_policy_v1.json", "sha256": sha256_file(ranking_path)},
    ]
    assert _normalized_stage_summary(summary_payload, input_records=input_records, output_records=output_records) == {
        "alpha_grid": ALPHA_GRID,
        "basis_name": "final_support_region_union_v1",
        "comparison_role": COMPARISON_ROLE,
        "created": "<TIMESTAMP>",
        "family_count": 3,
        "inputs": input_records,
        "k_grid": K_GRID,
        "n_rows": 4,
        "outputs": output_records,
        "parameters": {"alpha_grid": ALPHA_GRID, "k_grid": K_GRID, "top_n": TOP_N},
        "policies": [spec["policy_name"] for spec in POLICY_SPECS],
        "results": {"family_count": 3, "n_rows": 4, "theory_count": 3},
        "run": run_id,
        "runs_root": "<RUNS_ROOT>",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "theory_count": 3,
        "top_n": TOP_N,
        "verdict": "PASS",
        "version": "v1",
    }


def test_fails_if_required_artifact_is_missing(tmp_path: Path) -> None:
    run_id = "phase4b_missing_artifact"
    repo_root, runs_root = _prepare_consistent_run(
        tmp_path,
        run_id=run_id,
        missing_artifact="renyi_diversity_event_frequency_support_v1.json",
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "renyi_diversity_event_frequency_support_v1.json" in (result.stderr + result.stdout)


def test_fails_if_basis_name_mismatches(tmp_path: Path) -> None:
    run_id = "phase4b_basis_mismatch"
    repo_root, runs_root = _prepare_consistent_run(
        tmp_path,
        run_id=run_id,
        basis_name_overrides={"event_support_delta_lnL_softmax_mean_v1": "other_basis_v1"},
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "basis_name mismatch across policies" in (result.stderr + result.stdout)


def test_fails_if_n_rows_mismatch(tmp_path: Path) -> None:
    run_id = "phase4b_n_rows_mismatch"
    repo_root, runs_root = _prepare_consistent_run(
        tmp_path,
        run_id=run_id,
        n_rows_truncate_policy="event_frequency_support_v1",
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "n_rows mismatch across policies" in (result.stderr + result.stdout)


def test_fails_if_geometry_set_mismatches(tmp_path: Path) -> None:
    run_id = "phase4b_geometry_set_mismatch"
    repo_root, runs_root = _prepare_consistent_run(
        tmp_path,
        run_id=run_id,
        geometry_id_overrides={
            "event_support_delta_lnL_softmax_mean_v1": ["geo_a", "geo_b", "geo_c", "geo_x"],
        },
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "geometry key set mismatch" in (result.stderr + result.stdout)
