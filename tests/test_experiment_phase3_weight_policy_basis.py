from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from basurin_io import sha256_file
from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase3_weight_policy_basis.py")
SCHEMA_VERSION = "weight_policy_basis_v1"
UNIFORM_POLICY_NAME = "uniform_support_v1"
UNIFORM_POLICY_ROLE = "baseline_canonical"
UNIFORM_NORMALIZATION_METHOD = "divide_by_sum_raw_over_all_rows"
EVENT_FREQUENCY_POLICY_NAME = "event_frequency_support_v1"
EVENT_FREQUENCY_POLICY_ROLE = "comparison_factual"
EVENT_FREQUENCY_NORMALIZATION_METHOD = "divide_by_sum_support_count_events_over_all_rows"
SCORE_POLICY_NAME = "event_support_delta_lnL_softmax_mean_v1"
SCORE_POLICY_ROLE = "comparison_score_based"
SCORE_NORMALIZATION_METHOD = "divide_by_sum_event_normalized_scores_over_all_rows"
SCORE_SOURCE_POLICY_INPUTS = [
    "s5_aggregate/outputs/aggregate.json",
    "{source_run}/s4k_event_support_region/outputs/event_support_region.json",
    "{source_run}/s4_geometry_filter/outputs/ranked_all_full.json",
    "delta_lnL",
    "softmax per-event over final_support_region",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_valid(runs_root: Path, run_id: str, verdict: str = "PASS") -> None:
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": verdict})


def _make_phase2c_payload() -> dict:
    rows: list[dict] = []
    family_specs = (
        ("edgb", "EdGB", "EdGB_geo", 39),
        ("kerr_newman", "Kerr-Newman", "KN_geo", 32),
        ("dcs", "dCS", "dCS_geo", 35),
    )
    for family, theory, prefix, count in family_specs:
        for idx in range(count):
            support_count = (idx % 7) + 1
            rows.append(
                {
                    "raw_geometry_id": f"{prefix}_{idx:03d}",
                    "normalized_geometry_id": f"{prefix}_{idx:03d}_norm",
                    "atlas_family": family,
                    "atlas_theory": theory,
                    "join_mode": "exact_match_v1",
                    "join_status": "RESOLVED",
                    "support_basis_name": "final_support_region_union_v1",
                    "n_events_supported": support_count,
                    "support_fraction_events": support_count / 47.0,
                    "in_golden_post_hawking_union": True,
                    "in_final_support_region_union": True,
                    "joint_available": False,
                    "joint_posterior_weight_joint": None,
                    "joint_support_count": None,
                    "joint_support_fraction": None,
                    "joint_coverage": None,
                }
            )
    return {
        "schema_version": "support_ontology_basis_v1",
        "basis_name": "final_support_region_union_v1",
        "basis_equivalence_check": "final_support_region_union_vs_golden_post_hawking_union_v1",
        "bases_equal": True,
        "n_events": 47,
        "n_rows": len(rows),
        "family_counts": {"dcs": 35, "edgb": 39, "kerr_newman": 32},
        "theory_counts": {"EdGB": 39, "Kerr-Newman": 32, "dCS": 35},
        "n_joint_available": 0,
        "joint_weight_sum_over_support": 0.0,
        "joint_weight_role": "copied_for_audit_only_not_a_renyi_policy",
        "rows": rows,
    }


def _support_count_sum(payload: dict) -> int:
    return sum(int(row["n_events_supported"]) for row in payload["rows"])


def _run_script(
    repo_root: Path,
    run_id: str,
    runs_root: Path,
    *,
    policy_name: str = UNIFORM_POLICY_NAME,
    output_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    cmd = [sys.executable, str(repo_root / SCRIPT), "--run-id", run_id, "--policy-name", policy_name]
    if output_name is not None:
        cmd.extend(["--output-name", output_name])
    return subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _prepare_run(tmp_path: Path, *, run_id: str) -> tuple[Path, Path, dict]:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    _write_run_valid(runs_root, run_id)
    payload = _make_phase2c_payload()
    _write_json(
        runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json",
        payload,
    )
    return repo_root, runs_root, payload


def _normalized_stage_summary(
    payload: dict,
    *,
    input_sha256: str,
    output_sha256: str,
    output_rel_path: str,
    expected_inputs: list[dict] | None = None,
) -> dict:
    normalized = dict(payload)
    normalized["created"] = "<TIMESTAMP>"
    normalized["runs_root"] = "<RUNS_ROOT>"
    return {
        "basis_name": normalized["basis_name"],
        "coverage_fraction": normalized["coverage_fraction"],
        "created": normalized["created"],
        "inputs": expected_inputs
        or [
            {
                "label": "support_ontology_basis_v1",
                "path": "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
                "sha256": input_sha256,
            }
        ],
        "n_rows": normalized["n_rows"],
        "n_unweighted": normalized["n_unweighted"],
        "n_weighted": normalized["n_weighted"],
        "normalization_method": normalized["normalization_method"],
        "outputs": [
            {
                "path": output_rel_path,
                "sha256": output_sha256,
            }
        ],
        "output_name": normalized["output_name"],
        "output_path": normalized["output_path"],
        "parameters": normalized["parameters"],
        "policy_name": normalized["policy_name"],
        "policy_role": normalized["policy_role"],
        "results": normalized["results"],
        "run": normalized["run"],
        "runs_root": normalized["runs_root"],
        "schema_version": normalized["schema_version"],
        "source_policy_inputs": normalized["source_policy_inputs"],
        "stage": normalized["stage"],
        "verdict": normalized["verdict"],
        "version": normalized["version"],
        "weight_sum_normalized": normalized["weight_sum_normalized"],
        "weight_sum_raw": normalized["weight_sum_raw"],
    }


def _make_score_policy_phase2c_payload() -> dict:
    rows = [
        {
            "raw_geometry_id": "geom_alpha",
            "normalized_geometry_id": "geom_alpha_norm",
            "atlas_family": "edgb",
            "atlas_theory": "EdGB",
            "join_mode": "exact_match_v1",
            "join_status": "RESOLVED",
            "support_basis_name": "final_support_region_union_v1",
            "n_events_supported": 1,
            "support_fraction_events": 0.5,
            "in_golden_post_hawking_union": True,
            "in_final_support_region_union": True,
            "joint_available": False,
            "joint_posterior_weight_joint": None,
            "joint_support_count": None,
            "joint_support_fraction": None,
            "joint_coverage": None,
        },
        {
            "raw_geometry_id": "geom_beta",
            "normalized_geometry_id": "geom_beta_norm",
            "atlas_family": "kerr_newman",
            "atlas_theory": "Kerr-Newman",
            "join_mode": "exact_match_v1",
            "join_status": "RESOLVED",
            "support_basis_name": "final_support_region_union_v1",
            "n_events_supported": 2,
            "support_fraction_events": 1.0,
            "in_golden_post_hawking_union": True,
            "in_final_support_region_union": True,
            "joint_available": False,
            "joint_posterior_weight_joint": None,
            "joint_support_count": None,
            "joint_support_fraction": None,
            "joint_coverage": None,
        },
        {
            "raw_geometry_id": "geom_gamma",
            "normalized_geometry_id": "geom_gamma_norm",
            "atlas_family": "dcs",
            "atlas_theory": "dCS",
            "join_mode": "exact_match_v1",
            "join_status": "RESOLVED",
            "support_basis_name": "final_support_region_union_v1",
            "n_events_supported": 1,
            "support_fraction_events": 0.5,
            "in_golden_post_hawking_union": True,
            "in_final_support_region_union": True,
            "joint_available": False,
            "joint_posterior_weight_joint": None,
            "joint_support_count": None,
            "joint_support_fraction": None,
            "joint_coverage": None,
        },
    ]
    return {
        "schema_version": "support_ontology_basis_v1",
        "basis_name": "final_support_region_union_v1",
        "basis_equivalence_check": "final_support_region_union_vs_golden_post_hawking_union_v1",
        "bases_equal": True,
        "n_events": 2,
        "n_rows": len(rows),
        "family_counts": {"dcs": 1, "edgb": 1, "kerr_newman": 1},
        "theory_counts": {"EdGB": 1, "Kerr-Newman": 1, "dCS": 1},
        "n_joint_available": 0,
        "joint_weight_sum_over_support": 0.0,
        "joint_weight_role": "copied_for_audit_only_not_a_renyi_policy",
        "rows": rows,
    }


def _write_score_policy_event_artifacts(
    runs_root: Path,
    source_run_id: str,
    *,
    event_id: str,
    final_geometry_ids: list[str] | None,
    ranked_rows: list[dict] | None,
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


def _prepare_score_policy_run(
    tmp_path: Path,
    *,
    run_id: str,
    missing_event_support_for: str | None = None,
    missing_ranked_for: str | None = None,
    missing_delta_for: str | None = None,
) -> tuple[Path, Path, dict]:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    _write_run_valid(runs_root, run_id)
    payload = _make_score_policy_phase2c_payload()
    _write_json(
        runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json",
        payload,
    )
    _write_json(
        runs_root / run_id / "s5_aggregate" / "outputs" / "aggregate.json",
        {
            "schema_version": "mvp_aggregate_v1",
            "events": [
                {"event_id": "GW_ALPHA", "run_id": "run_evt_alpha"},
                {"event_id": "GW_BETA", "run_id": "run_evt_beta"},
            ],
        },
    )

    alpha_ranked = [
        {"geometry_id": "geom_alpha", "delta_lnL": 0.0},
        {"geometry_id": "geom_beta", "delta_lnL": -1.0},
        {"geometry_id": "geom_outside", "delta_lnL": -5.0},
    ]
    beta_ranked = [
        {"geometry_id": "geom_beta", "delta_lnL": -0.5},
        {"geometry_id": "geom_gamma", "delta_lnL": 0.0},
        {"geometry_id": "geom_outside", "delta_lnL": -3.0},
    ]
    if missing_delta_for == "run_evt_alpha":
        alpha_ranked = [row for row in alpha_ranked if row["geometry_id"] != "geom_beta"]
    if missing_delta_for == "run_evt_beta":
        beta_ranked = [row for row in beta_ranked if row["geometry_id"] != "geom_beta"]

    _write_score_policy_event_artifacts(
        runs_root,
        "run_evt_alpha",
        event_id="GW_ALPHA",
        final_geometry_ids=None if missing_event_support_for == "run_evt_alpha" else ["geom_alpha", "geom_beta"],
        ranked_rows=None if missing_ranked_for == "run_evt_alpha" else alpha_ranked,
    )
    _write_score_policy_event_artifacts(
        runs_root,
        "run_evt_beta",
        event_id="GW_BETA",
        final_geometry_ids=None if missing_event_support_for == "run_evt_beta" else ["geom_beta", "geom_gamma"],
        ranked_rows=None if missing_ranked_for == "run_evt_beta" else beta_ranked,
    )
    return repo_root, runs_root, payload


def test_contract_registered() -> None:
    contract = CONTRACTS.get("experiment/phase3_weight_policy_basis")
    assert contract is not None
    assert contract.required_inputs == [
        "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
    ]
    assert contract.produced_outputs == ["outputs/weight_policy_basis_v1.json"]


def test_happy_path_uniform_support_matches_normalized_snapshot(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_ok"
    repo_root, runs_root, phase2c_payload = _prepare_run(tmp_path, run_id=run_id)

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "OUT_ROOT=" in result.stdout
    assert "STAGE_DIR=" in result.stdout
    assert not (tmp_path / "runs").exists()

    stage_dir = runs_root / run_id / "experiment" / "phase3_weight_policy_basis"
    output_path = stage_dir / "outputs" / "weight_policy_basis_v1.json"
    summary_path = stage_dir / "stage_summary.json"
    manifest_path = stage_dir / "manifest.json"

    assert output_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = output_payload["rows"]
    assert output_payload["schema_version"] == SCHEMA_VERSION
    assert output_payload["basis_name"] == "final_support_region_union_v1"
    assert output_payload["policy_name"] == UNIFORM_POLICY_NAME
    assert output_payload["policy_role"] == UNIFORM_POLICY_ROLE
    assert output_payload["coverage_fraction"] == 1.0
    assert output_payload["n_rows"] == 106
    assert output_payload["n_weighted"] == 106
    assert output_payload["n_unweighted"] == 0
    assert output_payload["weight_sum_raw"] == 106.0
    assert output_payload["weight_sum_normalized"] == 1.0
    assert output_payload["normalization_method"] == UNIFORM_NORMALIZATION_METHOD
    assert output_payload["source_policy_inputs"] == []
    assert len(rows) == 106
    assert all(row["policy_name"] == UNIFORM_POLICY_NAME for row in rows)
    assert all(row["weight_raw"] == 1.0 for row in rows)
    assert all(row["weight_normalized"] == 1.0 / 106.0 for row in rows)
    assert all(row["weight_status"] == "WEIGHTED" for row in rows)
    assert all(row["criterion"] == "uniform_over_support_basis" for row in rows)
    assert all(row["criterion_version"] == "v1" for row in rows)
    assert all(row["source_artifacts"] == ["experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json"] for row in rows)
    assert rows[0]["raw_geometry_id"] == phase2c_payload["rows"][0]["raw_geometry_id"]
    assert rows[38]["atlas_family"] == phase2c_payload["rows"][38]["atlas_family"]
    assert rows[39]["atlas_family"] == phase2c_payload["rows"][39]["atlas_family"]
    assert rows[-1]["atlas_family"] == phase2c_payload["rows"][-1]["atlas_family"]
    assert rows[0]["support_count_events"] == phase2c_payload["rows"][0]["n_events_supported"]
    assert rows[1]["support_count_events"] == phase2c_payload["rows"][1]["n_events_supported"]
    assert rows[0]["support_fraction_events"] == phase2c_payload["rows"][0]["support_fraction_events"]

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    input_sha256 = sha256_file(
        runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
    )
    expected_summary = {
        "basis_name": "final_support_region_union_v1",
        "coverage_fraction": 1.0,
        "created": "<TIMESTAMP>",
        "inputs": [
            {
                "label": "support_ontology_basis_v1",
                "path": "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
                "sha256": input_sha256,
            }
        ],
        "n_rows": 106,
        "n_unweighted": 0,
        "n_weighted": 106,
        "normalization_method": UNIFORM_NORMALIZATION_METHOD,
        "outputs": [
            {
                "path": "experiment/phase3_weight_policy_basis/outputs/weight_policy_basis_v1.json",
                "sha256": sha256_file(output_path),
            }
        ],
        "output_name": "weight_policy_basis_v1.json",
        "output_path": "experiment/phase3_weight_policy_basis/outputs/weight_policy_basis_v1.json",
        "parameters": {
            "input_name": "support_ontology_basis_v1.json",
            "output_name": "weight_policy_basis_v1.json",
            "policy_name": UNIFORM_POLICY_NAME,
        },
        "policy_name": UNIFORM_POLICY_NAME,
        "policy_role": UNIFORM_POLICY_ROLE,
        "results": {
            "coverage_fraction": 1.0,
            "n_rows": 106,
            "n_unweighted": 0,
            "n_weighted": 106,
        },
        "run": run_id,
        "runs_root": "<RUNS_ROOT>",
        "schema_version": SCHEMA_VERSION,
        "source_policy_inputs": [],
        "stage": "experiment/phase3_weight_policy_basis",
        "verdict": "PASS",
        "version": "v1",
        "weight_sum_normalized": 1.0,
        "weight_sum_raw": 106.0,
    }
    assert _normalized_stage_summary(
        summary_payload,
        input_sha256=input_sha256,
        output_sha256=sha256_file(output_path),
        output_rel_path="experiment/phase3_weight_policy_basis/outputs/weight_policy_basis_v1.json",
    ) == expected_summary


def test_happy_path_event_frequency_matches_normalized_snapshot(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_event_frequency"
    repo_root, runs_root, phase2c_payload = _prepare_run(tmp_path, run_id=run_id)
    support_count_sum = _support_count_sum(phase2c_payload)

    result = _run_script(repo_root, run_id, runs_root, policy_name=EVENT_FREQUENCY_POLICY_NAME)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / "experiment" / "phase3_weight_policy_basis"
    output_path = stage_dir / "outputs" / "weight_policy_basis_v1.json"
    summary_path = stage_dir / "stage_summary.json"

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = output_payload["rows"]
    assert output_payload["schema_version"] == SCHEMA_VERSION
    assert output_payload["basis_name"] == "final_support_region_union_v1"
    assert output_payload["policy_name"] == EVENT_FREQUENCY_POLICY_NAME
    assert output_payload["policy_role"] == EVENT_FREQUENCY_POLICY_ROLE
    assert output_payload["coverage_fraction"] == 1.0
    assert output_payload["n_rows"] == 106
    assert output_payload["n_weighted"] == 106
    assert output_payload["n_unweighted"] == 0
    assert output_payload["weight_sum_raw"] == float(support_count_sum)
    assert output_payload["weight_sum_normalized"] == 1.0
    assert output_payload["normalization_method"] == EVENT_FREQUENCY_NORMALIZATION_METHOD
    assert output_payload["source_policy_inputs"] == ["support_count_events_from_phase2c"]
    assert len(rows) == 106
    assert all(row["policy_name"] == EVENT_FREQUENCY_POLICY_NAME for row in rows)
    assert all(row["weight_status"] == "WEIGHTED" for row in rows)
    assert all(row["criterion"] == "support_count_events_over_support_basis" for row in rows)
    assert all(row["criterion_version"] == "v1" for row in rows)
    assert all(row["source_artifacts"] == ["experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json"] for row in rows)
    assert [row["weight_raw"] for row in rows[:5]] == [
        float(phase2c_payload["rows"][idx]["n_events_supported"]) for idx in range(5)
    ]
    assert [row["support_count_events"] for row in rows[:5]] == [
        int(phase2c_payload["rows"][idx]["n_events_supported"]) for idx in range(5)
    ]
    assert rows[0]["support_fraction_events"] == phase2c_payload["rows"][0]["support_fraction_events"]
    assert rows[0]["weight_normalized"] == rows[0]["support_count_events"] / float(support_count_sum)
    assert rows[1]["weight_normalized"] == rows[1]["support_count_events"] / float(support_count_sum)
    assert rows[0]["evidence"] == {
        "basis_name": "final_support_region_union_v1",
        "support_count_events": rows[0]["support_count_events"],
        "support_count_sum_over_basis": support_count_sum,
        "event_frequency_weight_raw": float(rows[0]["support_count_events"]),
        "event_frequency_weight_normalized": rows[0]["weight_normalized"],
    }

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    input_sha256 = sha256_file(
        runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
    )
    expected_summary = {
        "basis_name": "final_support_region_union_v1",
        "coverage_fraction": 1.0,
        "created": "<TIMESTAMP>",
        "inputs": [
            {
                "label": "support_ontology_basis_v1",
                "path": "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
                "sha256": input_sha256,
            }
        ],
        "n_rows": 106,
        "n_unweighted": 0,
        "n_weighted": 106,
        "normalization_method": EVENT_FREQUENCY_NORMALIZATION_METHOD,
        "outputs": [
            {
                "path": "experiment/phase3_weight_policy_basis/outputs/weight_policy_basis_v1.json",
                "sha256": sha256_file(output_path),
            }
        ],
        "output_name": "weight_policy_basis_v1.json",
        "output_path": "experiment/phase3_weight_policy_basis/outputs/weight_policy_basis_v1.json",
        "parameters": {
            "input_name": "support_ontology_basis_v1.json",
            "output_name": "weight_policy_basis_v1.json",
            "policy_name": EVENT_FREQUENCY_POLICY_NAME,
        },
        "policy_name": EVENT_FREQUENCY_POLICY_NAME,
        "policy_role": EVENT_FREQUENCY_POLICY_ROLE,
        "results": {
            "coverage_fraction": 1.0,
            "n_rows": 106,
            "n_unweighted": 0,
            "n_weighted": 106,
        },
        "run": run_id,
        "runs_root": "<RUNS_ROOT>",
        "schema_version": SCHEMA_VERSION,
        "source_policy_inputs": ["support_count_events_from_phase2c"],
        "stage": "experiment/phase3_weight_policy_basis",
        "verdict": "PASS",
        "version": "v1",
        "weight_sum_normalized": 1.0,
        "weight_sum_raw": float(support_count_sum),
    }
    assert _normalized_stage_summary(
        summary_payload,
        input_sha256=input_sha256,
        output_sha256=sha256_file(output_path),
        output_rel_path="experiment/phase3_weight_policy_basis/outputs/weight_policy_basis_v1.json",
    ) == expected_summary


def test_happy_path_event_support_delta_lnl_softmax_mean_matches_normalized_snapshot(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_score_policy"
    output_name = "weight_policy_event_support_delta_lnL_softmax_mean_v1.json"
    repo_root, runs_root, phase2c_payload = _prepare_score_policy_run(tmp_path, run_id=run_id)

    result = _run_script(
        repo_root,
        run_id,
        runs_root,
        policy_name=SCORE_POLICY_NAME,
        output_name=output_name,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / "experiment" / "phase3_weight_policy_basis"
    output_path = stage_dir / "outputs" / output_name
    summary_path = stage_dir / "stage_summary.json"

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = output_payload["rows"]
    rows_by_id = {row["raw_geometry_id"]: row for row in rows}

    alpha_probs = [
        math.exp(0.0) / (math.exp(0.0) + math.exp(-1.0)),
        math.exp(-1.0) / (math.exp(0.0) + math.exp(-1.0)),
    ]
    beta_probs = [
        math.exp(-0.5) / (math.exp(-0.5) + math.exp(0.0)),
        math.exp(0.0) / (math.exp(-0.5) + math.exp(0.0)),
    ]
    expected_raw = {
        "geom_alpha": alpha_probs[0],
        "geom_beta": alpha_probs[1] + beta_probs[0],
        "geom_gamma": beta_probs[1],
    }

    assert output_payload["schema_version"] == SCHEMA_VERSION
    assert output_payload["basis_name"] == "final_support_region_union_v1"
    assert output_payload["policy_name"] == SCORE_POLICY_NAME
    assert output_payload["policy_role"] == SCORE_POLICY_ROLE
    assert output_payload["coverage_fraction"] == 1.0
    assert output_payload["n_rows"] == 3
    assert output_payload["n_weighted"] == 3
    assert output_payload["n_unweighted"] == 0
    assert output_payload["weight_sum_raw"] == 2.0
    assert output_payload["weight_sum_normalized"] == 1.0
    assert output_payload["normalization_method"] == SCORE_NORMALIZATION_METHOD
    assert output_payload["source_policy_inputs"] == SCORE_SOURCE_POLICY_INPUTS

    assert rows_by_id["geom_alpha"]["weight_raw"] == expected_raw["geom_alpha"]
    assert rows_by_id["geom_beta"]["weight_raw"] == expected_raw["geom_beta"]
    assert rows_by_id["geom_gamma"]["weight_raw"] == expected_raw["geom_gamma"]
    assert rows_by_id["geom_alpha"]["weight_normalized"] == expected_raw["geom_alpha"] / 2.0
    assert rows_by_id["geom_beta"]["weight_normalized"] == expected_raw["geom_beta"] / 2.0
    assert rows_by_id["geom_gamma"]["weight_normalized"] == expected_raw["geom_gamma"] / 2.0
    assert all(row["weight_status"] == "WEIGHTED" for row in rows)
    assert all(row["criterion"] == "delta_lnL_softmax_per_event_over_final_support_region" for row in rows)
    assert all(row["criterion_version"] == "v1" for row in rows)

    assert rows_by_id["geom_alpha"]["support_count_events"] == phase2c_payload["rows"][0]["n_events_supported"]
    assert rows_by_id["geom_beta"]["support_count_events"] == phase2c_payload["rows"][1]["n_events_supported"]
    assert rows_by_id["geom_gamma"]["support_count_events"] == phase2c_payload["rows"][2]["n_events_supported"]

    assert rows_by_id["geom_alpha"]["source_artifacts"] == [
        "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
        "s5_aggregate/outputs/aggregate.json",
        "run_evt_alpha/s4k_event_support_region/outputs/event_support_region.json",
        "run_evt_alpha/s4_geometry_filter/outputs/ranked_all_full.json",
    ]
    assert rows_by_id["geom_beta"]["source_artifacts"] == [
        "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
        "s5_aggregate/outputs/aggregate.json",
        "run_evt_alpha/s4k_event_support_region/outputs/event_support_region.json",
        "run_evt_alpha/s4_geometry_filter/outputs/ranked_all_full.json",
        "run_evt_beta/s4k_event_support_region/outputs/event_support_region.json",
        "run_evt_beta/s4_geometry_filter/outputs/ranked_all_full.json",
    ]
    assert rows_by_id["geom_gamma"]["source_artifacts"] == [
        "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
        "s5_aggregate/outputs/aggregate.json",
        "run_evt_beta/s4k_event_support_region/outputs/event_support_region.json",
        "run_evt_beta/s4_geometry_filter/outputs/ranked_all_full.json",
    ]
    assert rows_by_id["geom_beta"]["evidence"] == {
        "basis_name": "final_support_region_union_v1",
        "contributing_event_count": 2,
        "event_ids_sample": ["GW_ALPHA", "GW_BETA"],
        "local_score_definition": "exp(delta_lnL_i - max_delta_lnL_event)",
        "local_normalization": "per_event_softmax",
        "aggregation": "sum_over_events",
    }

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    phase2c_path = runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
    aggregate_path = runs_root / run_id / "s5_aggregate" / "outputs" / "aggregate.json"
    expected_inputs = [
        {
            "label": "support_ontology_basis_v1",
            "path": "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
            "sha256": sha256_file(phase2c_path),
        },
        {
            "label": "aggregate",
            "path": "s5_aggregate/outputs/aggregate.json",
            "sha256": sha256_file(aggregate_path),
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
    expected_summary = {
        "basis_name": "final_support_region_union_v1",
        "coverage_fraction": 1.0,
        "created": "<TIMESTAMP>",
        "inputs": expected_inputs,
        "n_rows": 3,
        "n_unweighted": 0,
        "n_weighted": 3,
        "normalization_method": SCORE_NORMALIZATION_METHOD,
        "outputs": [
            {
                "path": f"experiment/phase3_weight_policy_basis/outputs/{output_name}",
                "sha256": sha256_file(output_path),
            }
        ],
        "output_name": output_name,
        "output_path": f"experiment/phase3_weight_policy_basis/outputs/{output_name}",
        "parameters": {
            "input_name": "support_ontology_basis_v1.json",
            "output_name": output_name,
            "policy_name": SCORE_POLICY_NAME,
        },
        "policy_name": SCORE_POLICY_NAME,
        "policy_role": SCORE_POLICY_ROLE,
        "results": {
            "coverage_fraction": 1.0,
            "n_rows": 3,
            "n_unweighted": 0,
            "n_weighted": 3,
        },
        "run": run_id,
        "runs_root": "<RUNS_ROOT>",
        "schema_version": SCHEMA_VERSION,
        "source_policy_inputs": SCORE_SOURCE_POLICY_INPUTS,
        "stage": "experiment/phase3_weight_policy_basis",
        "verdict": "PASS",
        "version": "v1",
        "weight_sum_normalized": 1.0,
        "weight_sum_raw": 2.0,
    }
    assert _normalized_stage_summary(
        summary_payload,
        input_sha256=sha256_file(phase2c_path),
        output_sha256=sha256_file(output_path),
        output_rel_path=f"experiment/phase3_weight_policy_basis/outputs/{output_name}",
        expected_inputs=expected_inputs,
    ) == expected_summary


def test_rejects_unsupported_policy_name(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_bad_policy"
    repo_root, runs_root, _phase2c_payload = _prepare_run(tmp_path, run_id=run_id)

    result = _run_script(repo_root, run_id, runs_root, policy_name="bad_policy_v1")
    assert result.returncode == 2
    assert "Unsupported policy_name='bad_policy_v1'; supported values are ['event_frequency_support_v1', 'event_support_delta_lnL_softmax_mean_v1', 'uniform_support_v1']" in (
        result.stderr + result.stdout
    )

    stage_dir = runs_root / run_id / "experiment" / "phase3_weight_policy_basis"
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "Unsupported policy_name='bad_policy_v1'" in summary["error"]


def test_distinct_output_names_preserve_multiple_artifacts_without_collision(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_distinct_outputs"
    repo_root, runs_root, _phase2c_payload = _prepare_run(tmp_path, run_id=run_id)
    uniform_output_name = "weight_policy_uniform_support_v1.json"
    event_frequency_output_name = "weight_policy_event_frequency_support_v1.json"

    uniform_result = _run_script(
        repo_root,
        run_id,
        runs_root,
        policy_name=UNIFORM_POLICY_NAME,
        output_name=uniform_output_name,
    )
    assert uniform_result.returncode == 0, uniform_result.stderr + uniform_result.stdout

    event_frequency_result = _run_script(
        repo_root,
        run_id,
        runs_root,
        policy_name=EVENT_FREQUENCY_POLICY_NAME,
        output_name=event_frequency_output_name,
    )
    assert event_frequency_result.returncode == 0, event_frequency_result.stderr + event_frequency_result.stdout

    stage_dir = runs_root / run_id / "experiment" / "phase3_weight_policy_basis"
    uniform_output_path = stage_dir / "outputs" / uniform_output_name
    event_frequency_output_path = stage_dir / "outputs" / event_frequency_output_name
    summary_path = stage_dir / "stage_summary.json"

    assert uniform_output_path.exists()
    assert event_frequency_output_path.exists()
    assert uniform_output_path != event_frequency_output_path
    assert sha256_file(uniform_output_path) != sha256_file(event_frequency_output_path)

    uniform_payload = json.loads(uniform_output_path.read_text(encoding="utf-8"))
    event_frequency_payload = json.loads(event_frequency_output_path.read_text(encoding="utf-8"))
    assert uniform_payload["policy_name"] == UNIFORM_POLICY_NAME
    assert event_frequency_payload["policy_name"] == EVENT_FREQUENCY_POLICY_NAME

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["policy_name"] == EVENT_FREQUENCY_POLICY_NAME
    assert summary_payload["output_name"] == event_frequency_output_name
    assert summary_payload["output_path"] == (
        "experiment/phase3_weight_policy_basis/outputs/weight_policy_event_frequency_support_v1.json"
    )


def test_score_policy_fails_when_ranked_all_full_is_missing(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_missing_ranked"
    repo_root, runs_root, _payload = _prepare_score_policy_run(
        tmp_path,
        run_id=run_id,
        missing_ranked_for="run_evt_beta",
    )

    result = _run_script(repo_root, run_id, runs_root, policy_name=SCORE_POLICY_NAME)
    assert result.returncode == 2
    assert "run_evt_beta/s4_geometry_filter/outputs/ranked_all_full.json" in (result.stderr + result.stdout)

    summary = json.loads(
        (runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "stage_summary.json").read_text(encoding="utf-8")
    )
    assert summary["verdict"] == "FAIL"
    assert "run_evt_beta/s4_geometry_filter/outputs/ranked_all_full.json" in summary["error"]


def test_score_policy_fails_when_event_support_region_is_missing(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_missing_event_support"
    repo_root, runs_root, _payload = _prepare_score_policy_run(
        tmp_path,
        run_id=run_id,
        missing_event_support_for="run_evt_alpha",
    )

    result = _run_script(repo_root, run_id, runs_root, policy_name=SCORE_POLICY_NAME)
    assert result.returncode == 2
    assert "run_evt_alpha/s4k_event_support_region/outputs/event_support_region.json" in (result.stderr + result.stdout)

    summary = json.loads(
        (runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "stage_summary.json").read_text(encoding="utf-8")
    )
    assert summary["verdict"] == "FAIL"
    assert "run_evt_alpha/s4k_event_support_region/outputs/event_support_region.json" in summary["error"]


def test_score_policy_fails_when_event_supported_geometry_lacks_delta_lnl(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_missing_delta_lnl"
    repo_root, runs_root, _payload = _prepare_score_policy_run(
        tmp_path,
        run_id=run_id,
        missing_delta_for="run_evt_alpha",
    )

    result = _run_script(repo_root, run_id, runs_root, policy_name=SCORE_POLICY_NAME)
    assert result.returncode == 2
    assert "missing_geometry_ids=['geom_beta']" in (result.stderr + result.stdout)

    summary = json.loads(
        (runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "stage_summary.json").read_text(encoding="utf-8")
    )
    assert summary["verdict"] == "FAIL"
    assert "missing_geometry_ids=['geom_beta']" in summary["error"]
