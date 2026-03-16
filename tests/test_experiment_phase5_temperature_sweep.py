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

SCRIPT = Path("mvp/experiment_phase5_temperature_sweep.py")
STAGE = "experiment/phase5_temperature_sweep"
METRICS_SCHEMA_VERSION = "temperature_sweep_metrics_v1"
TOPK_SCHEMA_VERSION = "temperature_sweep_topk_v1"
FAMILY_SCHEMA_VERSION = "temperature_sweep_family_mass_v1"
THEORY_SCHEMA_VERSION = "temperature_sweep_theory_mass_v1"
TOP_GEOMETRY_SCHEMA_VERSION = "temperature_sweep_top_geometry_v1"
COMPARISON_ROLE = "temperature_sensitivity_of_score_based_policy"
SUPPORT_SCHEMA_VERSION = "support_ontology_basis_v1"
WEIGHT_POLICY_SCHEMA_VERSION = "weight_policy_basis_v1"
RENYI_SCHEMA_VERSION = "renyi_diversity_baseline_v1"
SWEEP_POLICY_NAME = "event_support_delta_lnL_softmax_mean_temperature_v1"
EXISTING_POLICY_NAME = "event_support_delta_lnL_softmax_mean_v1"
TEMPERATURE_GRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
ALPHA_GRID = [0, 1, 2, "inf"]
K_GRID = [1, 3, 5, 10]
TOP_N = 20
TOLERANCE_WEIGHT = 1e-9
TOLERANCE_METRIC = 1e-9

SUPPORT_INPUT_REL = "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json"
AGGREGATE_INPUT_REL = "s5_aggregate/outputs/aggregate.json"
EXISTING_WEIGHT_INPUT_REL = (
    "experiment/phase3_weight_policy_basis/outputs/"
    "weight_policy_event_support_delta_lnL_softmax_mean_v1.json"
)
EXISTING_RENYI_INPUT_REL = (
    "experiment/phase4_renyi_diversity_baseline/outputs/"
    "renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json"
)

BASIS_ROWS = [
    {
        "raw_geometry_id": "geom_alpha",
        "normalized_geometry_id": "geom_alpha_norm",
        "atlas_family": "edgb",
        "atlas_theory": "EdGB",
        "n_events_supported": 2,
        "support_fraction_events": 1.0,
    },
    {
        "raw_geometry_id": "geom_beta",
        "normalized_geometry_id": "geom_beta_norm",
        "atlas_family": "kerr_newman",
        "atlas_theory": "Kerr-Newman",
        "n_events_supported": 2,
        "support_fraction_events": 1.0,
    },
    {
        "raw_geometry_id": "geom_gamma",
        "normalized_geometry_id": "geom_gamma_norm",
        "atlas_family": "dcs",
        "atlas_theory": "dCS",
        "n_events_supported": 2,
        "support_fraction_events": 1.0,
    },
]

EVENT_SPECS = [
    {
        "event_id": "GW_ALPHA",
        "source_run_id": "run_evt_alpha",
        "final_geometry_ids": ["geom_alpha", "geom_beta", "geom_gamma"],
        "ranked_rows": [
            {"geometry_id": "geom_alpha", "delta_lnL": 0.0},
            {"geometry_id": "geom_beta", "delta_lnL": -1.0},
            {"geometry_id": "geom_gamma", "delta_lnL": -2.0},
            {"geometry_id": "geom_outside_alpha", "delta_lnL": -3.0},
        ],
    },
    {
        "event_id": "GW_BETA",
        "source_run_id": "run_evt_beta",
        "final_geometry_ids": ["geom_alpha", "geom_beta", "geom_gamma"],
        "ranked_rows": [
            {"geometry_id": "geom_alpha", "delta_lnL": 0.0},
            {"geometry_id": "geom_beta", "delta_lnL": -0.5},
            {"geometry_id": "geom_gamma", "delta_lnL": -1.5},
            {"geometry_id": "geom_outside_beta", "delta_lnL": -4.0},
        ],
    },
]


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_valid(runs_root: Path, run_id: str, verdict: str = "PASS") -> None:
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": verdict})


def _support_payload() -> dict:
    return {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "n_events": 2,
        "n_rows": len(BASIS_ROWS),
        "family_counts": {"dcs": 1, "edgb": 1, "kerr_newman": 1},
        "theory_counts": {"EdGB": 1, "Kerr-Newman": 1, "dCS": 1},
        "n_joint_available": 0,
        "joint_weight_sum_over_support": 0.0,
        "joint_weight_role": "copied_for_audit_only_not_a_renyi_policy",
        "rows": [
            {
                **row,
                "join_mode": "exact",
                "join_status": "resolved",
                "support_basis_name": "final_support_region_union_v1",
                "in_golden_post_hawking_union": True,
                "in_final_support_region_union": True,
                "joint_available": False,
                "joint_posterior_weight_joint": None,
                "joint_support_count": 0,
                "joint_support_fraction": 0.0,
                "joint_coverage": 0.0,
            }
            for row in BASIS_ROWS
        ],
    }


def _aggregate_payload() -> dict:
    return {
        "schema_version": "mvp_aggregate_v1",
        "events": [
            {"event_id": spec["event_id"], "run_id": spec["source_run_id"]}
            for spec in EVENT_SPECS
        ],
    }


def _temperature_weights(temperature: float) -> dict[str, dict[str, Any]]:
    raw = {row["raw_geometry_id"]: 0.0 for row in BASIS_ROWS}
    for event_spec in EVENT_SPECS:
        support_ids = event_spec["final_geometry_ids"]
        deltas = {
            row["geometry_id"]: float(row["delta_lnL"])
            for row in event_spec["ranked_rows"]
            if row["geometry_id"] in support_ids
        }
        max_delta = max(deltas.values())
        scores = {
            geometry_id: math.exp((deltas[geometry_id] - max_delta) / temperature)
            for geometry_id in support_ids
        }
        score_sum = math.fsum(scores.values())
        for geometry_id in support_ids:
            raw[geometry_id] += scores[geometry_id] / score_sum

    weight_sum_raw = math.fsum(raw.values())
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in BASIS_ROWS:
        raw_geometry_id = row["raw_geometry_id"]
        rows_by_id[raw_geometry_id] = {
            "raw_geometry_id": raw_geometry_id,
            "normalized_geometry_id": row["normalized_geometry_id"],
            "atlas_family": row["atlas_family"],
            "atlas_theory": row["atlas_theory"],
            "weight_raw": raw[raw_geometry_id],
            "weight_normalized": raw[raw_geometry_id] / weight_sum_raw,
        }
    return {
        "weight_sum_raw": weight_sum_raw,
        "weight_sum_normalized": 1.0,
        "rows_by_id": rows_by_id,
    }


def _renyi_metrics(weights: list[float]) -> dict[str, Any]:
    positive = [weight for weight in weights if weight > 0.0]
    h_alpha = {
        "0": math.log(float(len(positive))),
        "1": -math.fsum(weight * math.log(weight) for weight in positive),
        "2": -math.log(math.fsum(weight * weight for weight in positive)),
        "inf": -math.log(max(positive)),
    }
    d_alpha = {key: math.exp(value) for key, value in h_alpha.items()}
    return {
        "H_alpha": h_alpha,
        "D_alpha": d_alpha,
        "p_max": max(positive),
        "n_weighted": len(positive),
        "n_unweighted": len(weights) - len(positive),
        "weight_sum_normalized": math.fsum(weights),
    }


def _sorted_rows(rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows_by_id.values(), key=lambda row: (-float(row["weight_normalized"]), row["raw_geometry_id"]))


def _existing_weight_policy_payload() -> dict:
    temperature_payload = _temperature_weights(1.0)
    rows = []
    for row in BASIS_ROWS:
        data = temperature_payload["rows_by_id"][row["raw_geometry_id"]]
        rows.append(
            {
                "raw_geometry_id": row["raw_geometry_id"],
                "normalized_geometry_id": row["normalized_geometry_id"],
                "atlas_family": row["atlas_family"],
                "atlas_theory": row["atlas_theory"],
                "policy_name": EXISTING_POLICY_NAME,
                "weight_raw": data["weight_raw"],
                "weight_normalized": data["weight_normalized"],
                "weight_status": "WEIGHTED",
                "support_count_events": row["n_events_supported"],
                "support_fraction_events": row["support_fraction_events"],
                "source_artifacts": [
                    SUPPORT_INPUT_REL,
                    AGGREGATE_INPUT_REL,
                    "run_evt_alpha/s4k_event_support_region/outputs/event_support_region.json",
                    "run_evt_alpha/s4_geometry_filter/outputs/ranked_all_full.json",
                    "run_evt_beta/s4k_event_support_region/outputs/event_support_region.json",
                    "run_evt_beta/s4_geometry_filter/outputs/ranked_all_full.json",
                ],
                "criterion": "delta_lnL_softmax_per_event_over_final_support_region",
                "criterion_version": "v1",
                "evidence": {
                    "basis_name": "final_support_region_union_v1",
                    "contributing_event_count": 2,
                    "event_ids_sample": ["GW_ALPHA", "GW_BETA"],
                    "local_score_definition": "exp(delta_lnL_i - max_delta_lnL_event)",
                    "local_normalization": "per_event_softmax",
                    "aggregation": "sum_over_events",
                },
            }
        )
    return {
        "schema_version": WEIGHT_POLICY_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "policy_name": EXISTING_POLICY_NAME,
        "policy_role": "comparison_score_based",
        "coverage_fraction": 1.0,
        "n_rows": len(BASIS_ROWS),
        "n_weighted": len(BASIS_ROWS),
        "n_unweighted": 0,
        "weight_sum_raw": temperature_payload["weight_sum_raw"],
        "weight_sum_normalized": temperature_payload["weight_sum_normalized"],
        "normalization_method": "divide_by_sum_event_normalized_scores_over_all_rows",
        "source_policy_inputs": [
            "s5_aggregate/outputs/aggregate.json",
            "{source_run}/s4k_event_support_region/outputs/event_support_region.json",
            "{source_run}/s4_geometry_filter/outputs/ranked_all_full.json",
            "delta_lnL",
            "softmax per-event over final_support_region",
        ],
        "rows": rows,
    }


def _existing_renyi_payload() -> dict:
    temperature_payload = _temperature_weights(1.0)
    metrics = _renyi_metrics(
        [temperature_payload["rows_by_id"][row["raw_geometry_id"]]["weight_normalized"] for row in BASIS_ROWS]
    )
    return {
        "schema_version": RENYI_SCHEMA_VERSION,
        "metric_role": "epistemic_ensemble_diversity",
        "basis_name": "final_support_region_union_v1",
        "policy_name": EXISTING_POLICY_NAME,
        "weight_policy_file": "weight_policy_event_support_delta_lnL_softmax_mean_v1.json",
        "n_rows": len(BASIS_ROWS),
        "coverage_fraction": 1.0,
        "alpha_grid": ALPHA_GRID,
        "metrics": metrics,
        "notes": [
            "Epistemic ensemble diversity over the full supported basis.",
            "No sector conditioning is applied in this baseline.",
            "This output is not a black-hole thermodynamic entropy claim.",
        ],
    }


def _write_event_artifacts(
    runs_root: Path,
    *,
    source_run_id: str,
    event_id: str,
    final_geometry_ids: list[str] | None,
    ranked_rows: list[dict[str, Any]] | None,
) -> None:
    _write_run_valid(runs_root, source_run_id)
    if final_geometry_ids is not None:
        _write_json(
            runs_root / source_run_id / "s4k_event_support_region" / "outputs" / "event_support_region.json",
            {
                "event_id": event_id,
                "run_id": source_run_id,
                "final_geometry_ids": final_geometry_ids,
            },
        )
    if ranked_rows is not None:
        _write_json(
            runs_root / source_run_id / "s4_geometry_filter" / "outputs" / "ranked_all_full.json",
            ranked_rows,
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


def _prepare_run(
    tmp_path: Path,
    *,
    run_id: str,
    missing_event_support_for: str | None = None,
    missing_ranked_for: str | None = None,
    missing_delta_for: str | None = None,
) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    _write_run_valid(runs_root, run_id)
    _write_json(runs_root / run_id / SUPPORT_INPUT_REL, _support_payload())
    _write_json(runs_root / run_id / AGGREGATE_INPUT_REL, _aggregate_payload())
    _write_json(runs_root / run_id / EXISTING_WEIGHT_INPUT_REL, _existing_weight_policy_payload())
    _write_json(runs_root / run_id / EXISTING_RENYI_INPUT_REL, _existing_renyi_payload())

    for event_spec in EVENT_SPECS:
        ranked_rows = list(event_spec["ranked_rows"])
        if missing_delta_for == event_spec["source_run_id"]:
            ranked_rows = [row for row in ranked_rows if row["geometry_id"] != "geom_gamma"]
        _write_event_artifacts(
            runs_root,
            source_run_id=event_spec["source_run_id"],
            event_id=event_spec["event_id"],
            final_geometry_ids=(
                None if missing_event_support_for == event_spec["source_run_id"] else list(event_spec["final_geometry_ids"])
            ),
            ranked_rows=None if missing_ranked_for == event_spec["source_run_id"] else ranked_rows,
        )
    return repo_root, runs_root


def _expected_metrics_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temperature in TEMPERATURE_GRID:
        payload = _temperature_weights(temperature)
        metrics = _renyi_metrics(
            [payload["rows_by_id"][row["raw_geometry_id"]]["weight_normalized"] for row in BASIS_ROWS]
        )
        rows.append(
            {
                "temperature": temperature,
                "policy_name": SWEEP_POLICY_NAME,
                "normalization_method": "divide_by_sum_event_normalized_scores_over_all_rows",
                "n_rows": len(BASIS_ROWS),
                "coverage_fraction": 1.0,
                "n_weighted": metrics["n_weighted"],
                "n_unweighted": metrics["n_unweighted"],
                "p_max": metrics["p_max"],
                "H_0": metrics["H_alpha"]["0"],
                "H_1": metrics["H_alpha"]["1"],
                "H_2": metrics["H_alpha"]["2"],
                "H_inf": metrics["H_alpha"]["inf"],
                "D_0": metrics["D_alpha"]["0"],
                "D_1": metrics["D_alpha"]["1"],
                "D_2": metrics["D_alpha"]["2"],
                "D_inf": metrics["D_alpha"]["inf"],
                "weight_sum_raw": payload["weight_sum_raw"],
                "weight_sum_normalized": payload["weight_sum_normalized"],
                "criterion": "delta_lnL_softmax_per_event_temperature_sweep_over_final_support_region",
                "criterion_version": "v1",
            }
        )
    return rows


def _expected_topk_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temperature in TEMPERATURE_GRID:
        payload = _temperature_weights(temperature)
        sorted_rows = _sorted_rows(payload["rows_by_id"])
        total_mass = math.fsum(float(row["weight_normalized"]) for row in sorted_rows)
        for top_k in K_GRID:
            rows.append(
                {
                    "temperature": temperature,
                    "top_k": top_k,
                    "cumulative_mass": math.fsum(float(row["weight_normalized"]) for row in sorted_rows[:top_k]),
                    "normalization_check": total_mass,
                    "criterion": "cumulative_topk_mass_over_weight_normalized",
                    "criterion_version": "v1",
                }
            )
    return rows


def _expected_bucket_rows(
    *,
    bucket_key: str,
    mass_field: str,
    count_field: str,
    rank_field: str,
    criterion: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temperature in TEMPERATURE_GRID:
        payload = _temperature_weights(temperature)
        sorted_rows = _sorted_rows(payload["rows_by_id"])
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
                    "temperature": temperature,
                    bucket_key: bucket,
                    mass_field: mass,
                    count_field: bucket_count[bucket],
                    rank_field: rank,
                    "criterion": criterion,
                    "criterion_version": "v1",
                }
            )
    return rows


def _expected_top_geometry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temperature in TEMPERATURE_GRID:
        payload = _temperature_weights(temperature)
        sorted_rows = _sorted_rows(payload["rows_by_id"])
        cumulative_mass = 0.0
        for rank, row in enumerate(sorted_rows[:TOP_N], start=1):
            cumulative_mass += float(row["weight_normalized"])
            rows.append(
                {
                    "temperature": temperature,
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
        "results": payload["results"],
        "run": payload["run"],
        "runs_root": "<RUNS_ROOT>",
        "schema_version": payload["schema_version"],
        "stage": payload["stage"],
        "t1_renyi_match_with_existing_baseline": payload["t1_renyi_match_with_existing_baseline"],
        "t1_weight_match_with_existing_policy": payload["t1_weight_match_with_existing_policy"],
        "temperature_grid": payload["temperature_grid"],
        "theory_count": payload["theory_count"],
        "tolerance_metric": payload["tolerance_metric"],
        "tolerance_weight": payload["tolerance_weight"],
        "top_n": payload["top_n"],
        "verdict": payload["verdict"],
        "version": payload["version"],
    }


def test_contract_registered() -> None:
    contract = CONTRACTS.get(STAGE)
    assert contract is not None
    assert contract.required_inputs == [
        SUPPORT_INPUT_REL,
        AGGREGATE_INPUT_REL,
        EXISTING_WEIGHT_INPUT_REL,
        EXISTING_RENYI_INPUT_REL,
    ]
    assert contract.dynamic_inputs == [
        "{source_run}/s4k_event_support_region/outputs/event_support_region.json",
        "{source_run}/s4_geometry_filter/outputs/ranked_all_full.json",
    ]
    assert contract.produced_outputs == [
        "outputs/temperature_sweep_metrics_v1.json",
        "outputs/temperature_sweep_topk_v1.json",
        "outputs/temperature_sweep_family_mass_v1.json",
        "outputs/temperature_sweep_theory_mass_v1.json",
        "outputs/temperature_sweep_top_geometry_v1.json",
    ]


def test_happy_path_writes_expected_outputs_and_summary(tmp_path: Path) -> None:
    run_id = "phase5_temperature_sweep_ok"
    repo_root, runs_root = _prepare_run(tmp_path, run_id=run_id)

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "OUT_ROOT=" in result.stdout
    assert "STAGE_DIR=" in result.stdout

    stage_dir = runs_root / run_id / STAGE
    metrics_path = stage_dir / "outputs" / "temperature_sweep_metrics_v1.json"
    topk_path = stage_dir / "outputs" / "temperature_sweep_topk_v1.json"
    family_path = stage_dir / "outputs" / "temperature_sweep_family_mass_v1.json"
    theory_path = stage_dir / "outputs" / "temperature_sweep_theory_mass_v1.json"
    top_geometry_path = stage_dir / "outputs" / "temperature_sweep_top_geometry_v1.json"
    summary_path = stage_dir / "stage_summary.json"

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    topk_payload = json.loads(topk_path.read_text(encoding="utf-8"))
    family_payload = json.loads(family_path.read_text(encoding="utf-8"))
    theory_payload = json.loads(theory_path.read_text(encoding="utf-8"))
    top_geometry_payload = json.loads(top_geometry_path.read_text(encoding="utf-8"))

    assert metrics_payload == {
        "schema_version": METRICS_SCHEMA_VERSION,
        "comparison_role": COMPARISON_ROLE,
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": TEMPERATURE_GRID,
        "alpha_grid": ALPHA_GRID,
        "rows": _expected_metrics_rows(),
        "notes": [
            "Temperature sensitivity of the epistemic score-based delta_lnL softmax policy.",
            "No sector conditioning is applied.",
            "This sweep is not a black-hole thermodynamic entropy claim.",
        ],
    }
    assert topk_payload == {
        "schema_version": TOPK_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": TEMPERATURE_GRID,
        "k_grid": K_GRID,
        "rows": _expected_topk_rows(),
    }
    assert family_payload == {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": TEMPERATURE_GRID,
        "rows": _expected_bucket_rows(
            bucket_key="atlas_family",
            mass_field="family_mass",
            count_field="family_count",
            rank_field="family_mass_rank",
            criterion="mass_sum_by_atlas_family_over_weight_normalized",
        ),
    }
    assert theory_payload == {
        "schema_version": THEORY_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": TEMPERATURE_GRID,
        "rows": _expected_bucket_rows(
            bucket_key="atlas_theory",
            mass_field="theory_mass",
            count_field="theory_count",
            rank_field="theory_mass_rank",
            criterion="mass_sum_by_atlas_theory_over_weight_normalized",
        ),
    }
    assert top_geometry_payload == {
        "schema_version": TOP_GEOMETRY_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "temperature_grid": TEMPERATURE_GRID,
        "top_n": TOP_N,
        "rows": _expected_top_geometry_rows(),
    }

    metrics_by_temp = {row["temperature"]: row for row in metrics_payload["rows"]}
    topk_by_temp = {
        (row["temperature"], row["top_k"]): row for row in topk_payload["rows"]
    }
    assert metrics_by_temp[0.25]["D_1"] <= metrics_by_temp[1.0]["D_1"] <= metrics_by_temp[50.0]["D_1"]
    assert metrics_by_temp[0.25]["p_max"] >= metrics_by_temp[1.0]["p_max"] >= metrics_by_temp[50.0]["p_max"]
    assert topk_by_temp[(0.25, 1)]["cumulative_mass"] >= topk_by_temp[(1.0, 1)]["cumulative_mass"] >= topk_by_temp[(50.0, 1)]["cumulative_mass"]

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    input_records = [
        {
            "label": "support_ontology_basis_v1",
            "path": SUPPORT_INPUT_REL,
            "sha256": sha256_file(runs_root / run_id / SUPPORT_INPUT_REL),
        },
        {
            "label": "aggregate",
            "path": AGGREGATE_INPUT_REL,
            "sha256": sha256_file(runs_root / run_id / AGGREGATE_INPUT_REL),
        },
        {
            "label": "existing_weight_policy_t1",
            "path": EXISTING_WEIGHT_INPUT_REL,
            "sha256": sha256_file(runs_root / run_id / EXISTING_WEIGHT_INPUT_REL),
        },
        {
            "label": "existing_renyi_baseline_t1",
            "path": EXISTING_RENYI_INPUT_REL,
            "sha256": sha256_file(runs_root / run_id / EXISTING_RENYI_INPUT_REL),
        },
        {
            "label": "run_evt_alpha:event_support_region",
            "path": "run_evt_alpha/s4k_event_support_region/outputs/event_support_region.json",
            "sha256": sha256_file(runs_root / "run_evt_alpha" / "s4k_event_support_region" / "outputs" / "event_support_region.json"),
        },
        {
            "label": "run_evt_alpha:ranked_all_full",
            "path": "run_evt_alpha/s4_geometry_filter/outputs/ranked_all_full.json",
            "sha256": sha256_file(runs_root / "run_evt_alpha" / "s4_geometry_filter" / "outputs" / "ranked_all_full.json"),
        },
        {
            "label": "run_evt_beta:event_support_region",
            "path": "run_evt_beta/s4k_event_support_region/outputs/event_support_region.json",
            "sha256": sha256_file(runs_root / "run_evt_beta" / "s4k_event_support_region" / "outputs" / "event_support_region.json"),
        },
        {
            "label": "run_evt_beta:ranked_all_full",
            "path": "run_evt_beta/s4_geometry_filter/outputs/ranked_all_full.json",
            "sha256": sha256_file(runs_root / "run_evt_beta" / "s4_geometry_filter" / "outputs" / "ranked_all_full.json"),
        },
    ]
    output_records = [
        {"path": f"{STAGE}/outputs/temperature_sweep_metrics_v1.json", "sha256": sha256_file(metrics_path)},
        {"path": f"{STAGE}/outputs/temperature_sweep_topk_v1.json", "sha256": sha256_file(topk_path)},
        {"path": f"{STAGE}/outputs/temperature_sweep_family_mass_v1.json", "sha256": sha256_file(family_path)},
        {"path": f"{STAGE}/outputs/temperature_sweep_theory_mass_v1.json", "sha256": sha256_file(theory_path)},
        {"path": f"{STAGE}/outputs/temperature_sweep_top_geometry_v1.json", "sha256": sha256_file(top_geometry_path)},
    ]
    assert _normalized_stage_summary(summary_payload, input_records=input_records, output_records=output_records) == {
        "alpha_grid": ALPHA_GRID,
        "basis_name": "final_support_region_union_v1",
        "comparison_role": COMPARISON_ROLE,
        "created": "<TIMESTAMP>",
        "family_count": 3,
        "inputs": input_records,
        "k_grid": K_GRID,
        "n_rows": 3,
        "outputs": output_records,
        "parameters": {
            "alpha_grid": ALPHA_GRID,
            "k_grid": K_GRID,
            "temperature_grid": TEMPERATURE_GRID,
            "tolerance_metric": TOLERANCE_METRIC,
            "tolerance_weight": TOLERANCE_WEIGHT,
            "top_n": TOP_N,
        },
        "results": {
            "family_count": 3,
            "n_rows": 3,
            "t1_renyi_match_with_existing_baseline": True,
            "t1_weight_match_with_existing_policy": True,
            "theory_count": 3,
        },
        "run": run_id,
        "runs_root": "<RUNS_ROOT>",
        "schema_version": METRICS_SCHEMA_VERSION,
        "stage": STAGE,
        "t1_renyi_match_with_existing_baseline": True,
        "t1_weight_match_with_existing_policy": True,
        "temperature_grid": TEMPERATURE_GRID,
        "theory_count": 3,
        "tolerance_metric": TOLERANCE_METRIC,
        "tolerance_weight": TOLERANCE_WEIGHT,
        "top_n": TOP_N,
        "verdict": "PASS",
        "version": "v1",
    }


def test_fails_when_ranked_all_full_is_missing(tmp_path: Path) -> None:
    run_id = "phase5_missing_ranked"
    repo_root, runs_root = _prepare_run(
        tmp_path,
        run_id=run_id,
        missing_ranked_for="run_evt_beta",
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "run_evt_beta/s4_geometry_filter/outputs/ranked_all_full.json" in (result.stderr + result.stdout)

    summary = json.loads((runs_root / run_id / STAGE / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "run_evt_beta/s4_geometry_filter/outputs/ranked_all_full.json" in summary["error"]


def test_fails_when_event_support_region_is_missing(tmp_path: Path) -> None:
    run_id = "phase5_missing_event_support"
    repo_root, runs_root = _prepare_run(
        tmp_path,
        run_id=run_id,
        missing_event_support_for="run_evt_alpha",
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "run_evt_alpha/s4k_event_support_region/outputs/event_support_region.json" in (result.stderr + result.stdout)

    summary = json.loads((runs_root / run_id / STAGE / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "run_evt_alpha/s4k_event_support_region/outputs/event_support_region.json" in summary["error"]


def test_fails_when_event_supported_geometry_lacks_delta_lnl(tmp_path: Path) -> None:
    run_id = "phase5_missing_delta_lnl"
    repo_root, runs_root = _prepare_run(
        tmp_path,
        run_id=run_id,
        missing_delta_for="run_evt_beta",
    )
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "missing_geometry_ids=['geom_gamma']" in (result.stderr + result.stdout)

    summary = json.loads((runs_root / run_id / STAGE / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "missing_geometry_ids=['geom_gamma']" in summary["error"]
