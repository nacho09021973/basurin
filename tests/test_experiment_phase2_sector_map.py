from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from basurin_io import sha256_file
from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase2_sector_map.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_utf8_dataset(group: object, name: str, values: list[str]) -> None:
    import h5py  # type: ignore

    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, (len(values),), dtype=dt)[...] = values


def _make_phase1_h5(path: Path) -> None:
    h5py = pytest.importorskip("h5py")

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        atlas = h5.create_group("atlas")
        geometry_ids = [
            "geo_hyper",
            "geo_elliptic",
            "geo_euclidean",
            "geo_unknown",
            "geo_conflict",
            "geo_unsupported",
        ]
        _write_utf8_dataset(atlas, "geometry_id", geometry_ids)
        _write_utf8_dataset(
            atlas,
            "entry_json",
            [
                json.dumps({"metadata": {"geometry_sector": "hyperbolic"}}),
                json.dumps({"metadata": {"curvature_sign": 1}}),
                "null",
                "null",
                json.dumps({"geometry_sector": "HYPERBOLIC", "metadata": {"curvature_sign": 1}}),
                json.dumps({"metadata": {"geometry_sector": "ads"}}),
            ],
        )
        _write_utf8_dataset(
            atlas,
            "physical_parameters_json",
            [
                "null",
                "null",
                json.dumps({"curvature_sign": 0}),
                "null",
                "null",
                "null",
            ],
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


def _normalized_stage_summary(payload: dict, output_sha256: str) -> dict:
    normalized = dict(payload)
    normalized["created"] = "<TIMESTAMP>"
    normalized["runs_root"] = "<RUNS_ROOT>"
    return {
        "allowed_sectors": normalized["allowed_sectors"],
        "classification_policy": normalized["classification_policy"],
        "created": normalized["created"],
        "criterion_counts": normalized["criterion_counts"],
        "inputs": normalized["inputs"],
        "outputs": [
            {
                "path": "experiment/phase2_sector_map/outputs/sector_map_v1.json",
                "sha256": output_sha256,
            }
        ],
        "parameters": normalized["parameters"],
        "results": normalized["results"],
        "run": normalized["run"],
        "runs_root": normalized["runs_root"],
        "schema_version": normalized["schema_version"],
        "sector_counts": normalized["sector_counts"],
        "source_h5": normalized["source_h5"],
        "stage": normalized["stage"],
        "verdict": normalized["verdict"],
        "version": normalized["version"],
    }


def test_contract_registered() -> None:
    contract = CONTRACTS.get("experiment/phase2_sector_map")
    assert contract is not None
    assert contract.required_inputs == [
        "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
    ]
    assert contract.produced_outputs == ["outputs/sector_map_v1.json"]


def test_sector_map_happy_path_and_golden_summary(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2_sector_map_ok"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root
        / run_id
        / "experiment"
        / "phase1_geometry_h5"
        / "outputs"
        / "phase1_geometry_cohort.h5"
    )

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "OUT_ROOT=" in result.stdout
    assert "STAGE_DIR=" in result.stdout
    assert not (tmp_path / "runs").exists()

    stage_dir = runs_root / run_id / "experiment" / "phase2_sector_map"
    output_path = stage_dir / "outputs" / "sector_map_v1.json"
    summary_path = stage_dir / "stage_summary.json"
    manifest_path = stage_dir / "manifest.json"

    assert output_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = {row["geometry_id"]: row for row in output_payload["rows"]}

    assert rows["geo_hyper"]["geometry_sector"] == "HYPERBOLIC"
    assert rows["geo_hyper"]["criterion"] == "explicit_intrinsic_sector_label_v1"
    assert rows["geo_hyper"]["evidence_fields_used"] == ["atlas.entry_json.metadata.geometry_sector"]

    assert rows["geo_elliptic"]["geometry_sector"] == "ELLIPTIC"
    assert rows["geo_elliptic"]["criterion"] == "explicit_intrinsic_curvature_sign_v1"

    assert rows["geo_euclidean"]["geometry_sector"] == "EUCLIDEAN"
    assert rows["geo_unknown"]["geometry_sector"] == "UNKNOWN"
    assert rows["geo_unknown"]["criterion"] == "insufficient_intrinsic_metadata_v1"
    assert rows["geo_unknown"]["evidence_fields_used"] == []

    assert rows["geo_conflict"]["geometry_sector"] == "UNKNOWN"
    assert rows["geo_conflict"]["criterion"] == "conflicting_intrinsic_sector_evidence_v1"
    assert rows["geo_unsupported"]["geometry_sector"] == "UNKNOWN"
    assert rows["geo_unsupported"]["criterion"] == "unsupported_intrinsic_sector_value_v1"

    assert output_payload["sector_counts"] == {
        "ELLIPTIC": 1,
        "EUCLIDEAN": 1,
        "HYPERBOLIC": 1,
        "UNKNOWN": 3,
    }

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["verdict"] == "PASS"

    expected_summary = {
        "allowed_sectors": ["HYPERBOLIC", "ELLIPTIC", "EUCLIDEAN", "UNKNOWN"],
        "classification_policy": {
            "accepted_curvature_sign_field_paths": [
                "atlas.curvature_sign",
                "atlas.sectional_curvature_sign",
                "atlas.entry_json.curvature_sign",
                "atlas.entry_json.sectional_curvature_sign",
                "atlas.entry_json.metadata.curvature_sign",
                "atlas.entry_json.metadata.sectional_curvature_sign",
                "atlas.physical_parameters_json.curvature_sign",
                "atlas.physical_parameters_json.sectional_curvature_sign",
                "atlas.physical_parameters_json.metadata.curvature_sign",
                "atlas.physical_parameters_json.metadata.sectional_curvature_sign",
            ],
            "accepted_direct_field_paths": [
                "atlas.geometry_sector",
                "atlas.sector",
                "atlas.constant_curvature_class",
                "atlas.curvature_sector",
                "atlas.entry_json.geometry_sector",
                "atlas.entry_json.sector",
                "atlas.entry_json.constant_curvature_class",
                "atlas.entry_json.curvature_sector",
                "atlas.entry_json.metadata.geometry_sector",
                "atlas.entry_json.metadata.sector",
                "atlas.entry_json.metadata.constant_curvature_class",
                "atlas.entry_json.metadata.curvature_sector",
                "atlas.physical_parameters_json.geometry_sector",
                "atlas.physical_parameters_json.sector",
                "atlas.physical_parameters_json.constant_curvature_class",
                "atlas.physical_parameters_json.curvature_sector",
                "atlas.physical_parameters_json.metadata.geometry_sector",
                "atlas.physical_parameters_json.metadata.sector",
                "atlas.physical_parameters_json.metadata.constant_curvature_class",
                "atlas.physical_parameters_json.metadata.curvature_sector",
            ],
            "name": "explicit_intrinsic_sector_only_v1",
            "never_infer_from": [
                "atlas.geometry_id",
                "atlas.theory",
                "atlas.family",
                "atlas.mode",
                "atlas.zeta",
                "atlas.q_charge",
                "atlas.delta_f_frac",
                "atlas.delta_tau_frac",
                "atlas.phi_atlas",
                "atlas.entry_json.theory",
                "atlas.entry_json.metadata.family",
                "atlas.physical_parameters_json",
            ],
        },
        "created": "<TIMESTAMP>",
        "criterion_counts": {
            "conflicting_intrinsic_sector_evidence_v1": 1,
            "explicit_intrinsic_curvature_sign_v1": 2,
            "explicit_intrinsic_sector_label_v1": 1,
            "insufficient_intrinsic_metadata_v1": 1,
            "unsupported_intrinsic_sector_value_v1": 1,
        },
        "inputs": [
            {
                "label": "phase1_geometry_h5",
                "path": "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
                "sha256": sha256_file(
                    runs_root
                    / run_id
                    / "experiment"
                    / "phase1_geometry_h5"
                    / "outputs"
                    / "phase1_geometry_cohort.h5"
                ),
            }
        ],
        "outputs": [
            {
                "path": "experiment/phase2_sector_map/outputs/sector_map_v1.json",
                "sha256": sha256_file(output_path),
            }
        ],
        "parameters": {
            "input_name": "phase1_geometry_cohort.h5",
            "output_name": "sector_map_v1.json",
            "policy_name": "explicit_intrinsic_sector_only_v1",
        },
        "results": {
            "n_classified_non_unknown": 3,
            "n_geometry_ids": 6,
            "n_unknown": 3,
        },
        "run": "phase2_sector_map_ok",
        "runs_root": "<RUNS_ROOT>",
        "schema_version": "sector_map_v1",
        "sector_counts": {
            "ELLIPTIC": 1,
            "EUCLIDEAN": 1,
            "HYPERBOLIC": 1,
            "UNKNOWN": 3,
        },
        "source_h5": "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
        "stage": "experiment/phase2_sector_map",
        "verdict": "PASS",
        "version": "v1",
    }
    assert _normalized_stage_summary(summary_payload, sha256_file(output_path)) == expected_summary


def test_missing_phase1_h5_reports_expected_path_and_regen_cmd(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2_sector_map_missing_h5"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "expected_path=" in result.stderr
    assert "regen_cmd='python mvp/experiment_phase1_geometry_h5.py --run-id phase2_sector_map_missing_h5'" in result.stderr


def test_abort_when_run_valid_is_not_pass(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2_sector_map_invalid"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "FAIL"})
    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "RUN_VALID check failed" in result.stderr
    assert not (runs_root / run_id / "experiment" / "phase2_sector_map").exists()
