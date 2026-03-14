from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from basurin_io import sha256_file
from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase4_renyi_diversity_baseline.py")
SCHEMA_VERSION = "renyi_diversity_baseline_v1"
WEIGHT_POLICY_SCHEMA_VERSION = "weight_policy_basis_v1"
STAGE = "experiment/phase4_renyi_diversity_baseline"
DEFAULT_OUTPUT_NAME = "renyi_diversity_baseline_v1.json"
METRIC_ROLE = "epistemic_ensemble_diversity"
ALPHA_GRID = [0, 1, 2, "inf"]
NORMALIZATION_TOLERANCE = 1e-9


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_valid(runs_root: Path, run_id: str, verdict: str = "PASS") -> None:
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": verdict})


def _policy_role(policy_name: str) -> str:
    if policy_name == "uniform_support_v1":
        return "baseline_canonical"
    if policy_name == "event_frequency_support_v1":
        return "comparison_factual"
    if policy_name == "event_support_delta_lnL_softmax_mean_v1":
        return "comparison_score_based"
    raise AssertionError(policy_name)


def _source_policy_inputs(policy_name: str) -> list[str]:
    if policy_name == "uniform_support_v1":
        return []
    if policy_name == "event_frequency_support_v1":
        return ["support_count_events_from_phase2c"]
    if policy_name == "event_support_delta_lnL_softmax_mean_v1":
        return [
            "s5_aggregate/outputs/aggregate.json",
            "{source_run}/s4k_event_support_region/outputs/event_support_region.json",
            "{source_run}/s4_geometry_filter/outputs/ranked_all_full.json",
            "delta_lnL",
            "softmax per-event over final_support_region",
        ]
    raise AssertionError(policy_name)


def _make_weight_policy_payload(
    weights: list[float],
    *,
    policy_name: str,
    weight_sum_normalized: float | None = None,
) -> dict:
    rows: list[dict] = []
    family_specs = [
        ("geom_alpha", "geom_alpha_norm", "edgb", "EdGB"),
        ("geom_beta", "geom_beta_norm", "kerr_newman", "Kerr-Newman"),
        ("geom_gamma", "geom_gamma_norm", "dcs", "dCS"),
        ("geom_delta", "geom_delta_norm", "edgb", "EdGB"),
    ]
    n_weighted = sum(1 for weight in weights if weight > 0.0)
    for idx, weight in enumerate(weights):
        raw_geometry_id, normalized_geometry_id, family, theory = family_specs[idx]
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
    if weight_sum_normalized is None:
        weight_sum_normalized = float(sum(weights))
    return {
        "schema_version": WEIGHT_POLICY_SCHEMA_VERSION,
        "basis_name": "final_support_region_union_v1",
        "policy_name": policy_name,
        "policy_role": _policy_role(policy_name),
        "coverage_fraction": 1.0,
        "n_rows": len(rows),
        "n_weighted": n_weighted,
        "n_unweighted": len(rows) - n_weighted,
        "weight_sum_raw": float(sum(weights)),
        "weight_sum_normalized": weight_sum_normalized,
        "normalization_method": "fixture_normalization_v1",
        "source_policy_inputs": _source_policy_inputs(policy_name),
        "rows": rows,
    }


def _run_script(
    repo_root: Path,
    run_id: str,
    runs_root: Path,
    *,
    weight_policy_file: str,
    output_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    cmd = [
        sys.executable,
        str(repo_root / SCRIPT),
        "--run-id",
        run_id,
        "--weight-policy-file",
        weight_policy_file,
    ]
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


def _prepare_run(
    tmp_path: Path,
    *,
    run_id: str,
    weight_policy_payload: dict,
    weight_policy_file: str,
) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    _write_run_valid(runs_root, run_id)
    weight_policy_path = (
        runs_root
        / run_id
        / "experiment"
        / "phase3_weight_policy_basis"
        / "outputs"
        / weight_policy_file
    )
    _write_json(weight_policy_path, weight_policy_payload)
    return repo_root, runs_root, weight_policy_path


def _normalized_stage_summary(
    payload: dict,
    *,
    input_sha256: str,
    output_sha256: str,
    output_name: str,
    weight_policy_file: str,
) -> dict:
    return {
        "alpha_grid": payload["alpha_grid"],
        "basis_name": payload["basis_name"],
        "coverage_fraction": payload["coverage_fraction"],
        "created": "<TIMESTAMP>",
        "inputs": [
            {
                "label": "weight_policy_basis_v1",
                "path": f"experiment/phase3_weight_policy_basis/outputs/{weight_policy_file}",
                "sha256": input_sha256,
            }
        ],
        "metrics": payload["metrics"],
        "metric_role": payload["metric_role"],
        "n_rows": payload["n_rows"],
        "output_name": payload["output_name"],
        "output_path": payload["output_path"],
        "outputs": [
            {
                "path": f"{STAGE}/outputs/{output_name}",
                "sha256": output_sha256,
            }
        ],
        "parameters": payload["parameters"],
        "policy_name": payload["policy_name"],
        "results": payload["results"],
        "run": payload["run"],
        "runs_root": "<RUNS_ROOT>",
        "schema_version": payload["schema_version"],
        "stage": payload["stage"],
        "verdict": payload["verdict"],
        "version": payload["version"],
        "weight_policy_file": payload["weight_policy_file"],
    }


def test_contract_registered() -> None:
    contract = CONTRACTS.get(STAGE)
    assert contract is not None
    assert contract.required_inputs == []
    assert contract.dynamic_inputs == [
        "experiment/phase3_weight_policy_basis/outputs/{weight_policy_file}",
    ]
    assert contract.produced_outputs == ["outputs/renyi_diversity_baseline_v1.json"]
    assert contract.upstream_stages == ["experiment/phase3_weight_policy_basis"]


def test_uniform_distribution_known_values(tmp_path: Path) -> None:
    run_id = "phase4_uniform_known"
    weight_policy_file = "weight_policy_uniform_support_v1.json"
    payload = _make_weight_policy_payload([0.25, 0.25, 0.25, 0.25], policy_name="uniform_support_v1")
    repo_root, runs_root, weight_policy_path = _prepare_run(
        tmp_path,
        run_id=run_id,
        weight_policy_payload=payload,
        weight_policy_file=weight_policy_file,
    )

    result = _run_script(repo_root, run_id, runs_root, weight_policy_file=weight_policy_file)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / STAGE
    output_path = stage_dir / "outputs" / DEFAULT_OUTPUT_NAME
    summary_path = stage_dir / "stage_summary.json"
    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    metrics = output_payload["metrics"]

    expected_h = math.log(4.0)
    assert output_payload == {
        "schema_version": SCHEMA_VERSION,
        "metric_role": METRIC_ROLE,
        "basis_name": "final_support_region_union_v1",
        "policy_name": "uniform_support_v1",
        "weight_policy_file": weight_policy_file,
        "n_rows": 4,
        "coverage_fraction": 1.0,
        "alpha_grid": ALPHA_GRID,
        "metrics": {
            "H_alpha": {"0": expected_h, "1": expected_h, "2": expected_h, "inf": expected_h},
            "D_alpha": {"0": 4.0, "1": 4.0, "2": 4.0, "inf": 4.0},
            "p_max": 0.25,
            "n_weighted": 4,
            "n_unweighted": 0,
            "weight_sum_normalized": 1.0,
        },
        "notes": [
            "Epistemic ensemble diversity over the full supported basis.",
            "No sector conditioning is applied in this baseline.",
            "This output is not a black-hole thermodynamic entropy claim.",
        ],
    }
    assert metrics["D_alpha"]["0"] == math.exp(metrics["H_alpha"]["0"])

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert _normalized_stage_summary(
        summary_payload,
        input_sha256=sha256_file(weight_policy_path),
        output_sha256=sha256_file(output_path),
        output_name=DEFAULT_OUTPUT_NAME,
        weight_policy_file=weight_policy_file,
    ) == {
        "alpha_grid": ALPHA_GRID,
        "basis_name": "final_support_region_union_v1",
        "coverage_fraction": 1.0,
        "created": "<TIMESTAMP>",
        "inputs": [
            {
                "label": "weight_policy_basis_v1",
                "path": f"experiment/phase3_weight_policy_basis/outputs/{weight_policy_file}",
                "sha256": sha256_file(weight_policy_path),
            }
        ],
        "metrics": {
            "H_alpha": {"0": expected_h, "1": expected_h, "2": expected_h, "inf": expected_h},
            "D_alpha": {"0": 4.0, "1": 4.0, "2": 4.0, "inf": 4.0},
            "p_max": 0.25,
            "n_weighted": 4,
            "n_unweighted": 0,
            "weight_sum_normalized": 1.0,
        },
        "metric_role": METRIC_ROLE,
        "n_rows": 4,
        "output_name": DEFAULT_OUTPUT_NAME,
        "output_path": f"{STAGE}/outputs/{DEFAULT_OUTPUT_NAME}",
        "outputs": [
            {
                "path": f"{STAGE}/outputs/{DEFAULT_OUTPUT_NAME}",
                "sha256": sha256_file(output_path),
            }
        ],
        "parameters": {
            "alpha_grid": ALPHA_GRID,
            "normalization_tolerance": NORMALIZATION_TOLERANCE,
            "output_name": DEFAULT_OUTPUT_NAME,
            "weight_policy_file": weight_policy_file,
        },
        "policy_name": "uniform_support_v1",
        "results": {
            "coverage_fraction": 1.0,
            "n_rows": 4,
            "n_unweighted": 0,
            "n_weighted": 4,
        },
        "run": run_id,
        "runs_root": "<RUNS_ROOT>",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "verdict": "PASS",
        "version": "v1",
        "weight_policy_file": weight_policy_file,
    }


def test_degenerate_delta_distribution_known_values(tmp_path: Path) -> None:
    run_id = "phase4_delta_known"
    weight_policy_file = "weight_policy_event_frequency_support_v1.json"
    payload = _make_weight_policy_payload([1.0, 0.0, 0.0, 0.0], policy_name="event_frequency_support_v1")
    repo_root, runs_root, _weight_policy_path = _prepare_run(
        tmp_path,
        run_id=run_id,
        weight_policy_payload=payload,
        weight_policy_file=weight_policy_file,
    )

    result = _run_script(repo_root, run_id, runs_root, weight_policy_file=weight_policy_file)
    assert result.returncode == 0, result.stderr + result.stdout

    output_payload = json.loads(
        (runs_root / run_id / STAGE / "outputs" / DEFAULT_OUTPUT_NAME).read_text(encoding="utf-8")
    )
    assert output_payload["policy_name"] == "event_frequency_support_v1"
    assert output_payload["metrics"] == {
        "H_alpha": {"0": 0.0, "1": 0.0, "2": 0.0, "inf": 0.0},
        "D_alpha": {"0": 1.0, "1": 1.0, "2": 1.0, "inf": 1.0},
        "p_max": 1.0,
        "n_weighted": 1,
        "n_unweighted": 3,
        "weight_sum_normalized": 1.0,
    }


def test_fails_on_non_normalized_weights(tmp_path: Path) -> None:
    run_id = "phase4_bad_norm"
    weight_policy_file = "weight_policy_score_support_v1.json"
    payload = _make_weight_policy_payload(
        [0.2, 0.2, 0.2, 0.2],
        policy_name="event_support_delta_lnL_softmax_mean_v1",
        weight_sum_normalized=0.8,
    )
    repo_root, runs_root, _weight_policy_path = _prepare_run(
        tmp_path,
        run_id=run_id,
        weight_policy_payload=payload,
        weight_policy_file=weight_policy_file,
    )

    result = _run_script(repo_root, run_id, runs_root, weight_policy_file=weight_policy_file)
    assert result.returncode == 2
    assert "weight_sum_normalized must be within" in (result.stderr + result.stdout)

    summary = json.loads((runs_root / run_id / STAGE / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "weight_sum_normalized must be within" in summary["error"]


def test_fails_on_negative_weight(tmp_path: Path) -> None:
    run_id = "phase4_negative_weight"
    weight_policy_file = "weight_policy_uniform_support_v1.json"
    payload = _make_weight_policy_payload([0.5, 0.5, -0.1, 0.1], policy_name="uniform_support_v1")
    repo_root, runs_root, _weight_policy_path = _prepare_run(
        tmp_path,
        run_id=run_id,
        weight_policy_payload=payload,
        weight_policy_file=weight_policy_file,
    )

    result = _run_script(repo_root, run_id, runs_root, weight_policy_file=weight_policy_file)
    assert result.returncode == 2
    assert "negative weight_normalized" in (result.stderr + result.stdout)

    summary = json.loads((runs_root / run_id / STAGE / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "negative weight_normalized" in summary["error"]


def test_compatible_with_all_materialized_phase3_policy_names(tmp_path: Path) -> None:
    cases = [
        ("uniform_support_v1", "weight_policy_uniform_support_v1.json"),
        ("event_frequency_support_v1", "weight_policy_event_frequency_support_v1.json"),
        ("event_support_delta_lnL_softmax_mean_v1", "weight_policy_event_support_delta_lnL_softmax_mean_v1.json"),
    ]
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase4_policy_compat"
    _write_run_valid(runs_root, run_id)

    for idx, (policy_name, weight_policy_file) in enumerate(cases):
        payload = _make_weight_policy_payload([0.5, 0.25, 0.25, 0.0], policy_name=policy_name)
        _write_json(
            runs_root / run_id / "experiment" / "phase3_weight_policy_basis" / "outputs" / weight_policy_file,
            payload,
        )
        output_name = f"renyi_{idx}_{policy_name}.json"
        result = _run_script(
            repo_root,
            run_id,
            runs_root,
            weight_policy_file=weight_policy_file,
            output_name=output_name,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        output_payload = json.loads(
            (runs_root / run_id / STAGE / "outputs" / output_name).read_text(encoding="utf-8")
        )
        assert output_payload["policy_name"] == policy_name
        assert output_payload["metrics"]["n_weighted"] == 3
        assert output_payload["metrics"]["n_unweighted"] == 1
        assert output_payload["metrics"]["weight_sum_normalized"] == 1.0

