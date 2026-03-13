from __future__ import annotations

import json
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
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
) -> dict:
    normalized = dict(payload)
    normalized["created"] = "<TIMESTAMP>"
    normalized["runs_root"] = "<RUNS_ROOT>"
    return {
        "basis_name": normalized["basis_name"],
        "coverage_fraction": normalized["coverage_fraction"],
        "created": normalized["created"],
        "inputs": [
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


def test_rejects_unsupported_policy_name(tmp_path: Path) -> None:
    run_id = "phase3_weight_policy_basis_bad_policy"
    repo_root, runs_root, _phase2c_payload = _prepare_run(tmp_path, run_id=run_id)

    result = _run_script(repo_root, run_id, runs_root, policy_name="bad_policy_v1")
    assert result.returncode == 2
    assert "Unsupported policy_name='bad_policy_v1'; supported values are ['event_frequency_support_v1', 'uniform_support_v1']" in (
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
