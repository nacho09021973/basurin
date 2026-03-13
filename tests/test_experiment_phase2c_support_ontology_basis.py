from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from basurin_io import sha256_file
from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase2c_support_ontology_basis.py")
SCHEMA_VERSION = "support_ontology_basis_v1"
JOINT_WEIGHT_ROLE = "copied_for_audit_only_not_a_renyi_policy"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_utf8_dataset(group: object, name: str, values: list[str]) -> None:
    import h5py  # type: ignore

    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, (len(values),), dtype=dt)[...] = values


def _make_phase1_h5(path: Path, *, golden_matches_final: bool) -> None:
    h5py = pytest.importorskip("h5py")

    geometry_ids = ["geo_kerr", "geo_edgb", "geo_kn", "geo_dcs"]
    final_support_region = [
        [False, True, False, True],
        [False, True, True, False],
    ]
    golden_post_hawking = (
        final_support_region
        if golden_matches_final
        else [
            [False, True, False, False],
            [False, True, True, False],
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        atlas = h5.create_group("atlas")
        membership = h5.create_group("membership")
        joint = h5.create_group("joint_posterior")

        _write_utf8_dataset(atlas, "geometry_id", geometry_ids)
        _write_utf8_dataset(atlas, "family", ["WRONG_KERR", "WRONG_EDGB", "WRONG_KN", "WRONG_DCS"])
        _write_utf8_dataset(atlas, "theory", ["WRONG_GR", "WRONG_EdGB", "WRONG_KN", "WRONG_dCS"])

        membership.create_dataset("final_support_region", data=final_support_region)
        membership.create_dataset("golden_post_hawking", data=golden_post_hawking)

        joint.create_dataset("available", data=[False, True, True, False])
        joint.create_dataset("posterior_weight_joint", data=[float("nan"), 0.125, 0.375, float("nan")])
        joint.create_dataset("support_count", data=[-1, 2, 1, -1])
        joint.create_dataset("support_fraction", data=[float("nan"), 1.0, 0.5, float("nan")])
        joint.create_dataset("coverage", data=[float("nan"), 1.0, 0.5, float("nan")])


def _family_row(raw_geometry_id: str, normalized_geometry_id: str, family: str, theory: str, join_mode: str) -> dict:
    return {
        "raw_geometry_id": raw_geometry_id,
        "normalized_geometry_id": normalized_geometry_id,
        "join_mode": join_mode,
        "join_status": "RESOLVED",
        "atlas_path": "docs/ringdown/atlas/atlas_berti_v2.json",
        "atlas_geometry_id": normalized_geometry_id,
        "atlas_family": family,
        "atlas_theory": theory,
        "criterion": join_mode,
        "criterion_version": "v1",
        "evidence_fields_used": ["h5.atlas.geometry_id"],
        "evidence": {"h5.atlas.geometry_id": raw_geometry_id},
    }


def _make_family_map(path: Path, *, include_geo_kn: bool = True) -> None:
    rows = [
        _family_row("geo_kerr", "geo_kerr_l2m2n0", "kerr", "GR_Kerr", "normalized_match_l2m2n0_v1"),
        _family_row("geo_edgb", "geo_edgb", "edgb", "EdGB", "exact_match_v1"),
        _family_row("geo_dcs", "geo_dcs", "dcs", "dCS", "exact_match_v1"),
    ]
    if include_geo_kn:
        rows.insert(
            2,
            _family_row(
                "geo_kn",
                "geo_kn_l2m2n0",
                "kerr_newman",
                "Kerr-Newman",
                "normalized_match_l2m2n0_v1",
            ),
        )

    _write_json(
        path,
        {
            "schema_version": "family_map_v1",
            "normalization_policy_name": "exact_or_normalized_l2m2n0_v1",
            "source_h5": "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
            "source_atlas": "docs/ringdown/atlas/atlas_berti_v2.json",
            "n_rows": len(rows),
            "family_counts": {
                "dcs": 1,
                "edgb": 1,
                "kerr": 1,
                **({"kerr_newman": 1} if include_geo_kn else {}),
            },
            "theory_counts": {
                "EdGB": 1,
                "GR_Kerr": 1,
                "dCS": 1,
                **({"Kerr-Newman": 1} if include_geo_kn else {}),
            },
            "join_mode_counts": {
                "exact_match_v1": 2 if include_geo_kn else 2,
                "normalized_match_l2m2n0_v1": 2 if include_geo_kn else 1,
            },
            "unresolved_geometry_ids": [],
            "rows": rows,
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


def _prepare_run(tmp_path: Path, *, run_id: str, golden_matches_final: bool, include_geo_kn: bool = True) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        golden_matches_final=golden_matches_final,
    )
    _make_family_map(
        runs_root / run_id / "experiment" / "phase2a_atlas_family_map" / "outputs" / "family_map_v1.json",
        include_geo_kn=include_geo_kn,
    )
    return repo_root, runs_root


def _normalized_stage_summary(payload: dict, *, phase1_sha256: str, family_map_sha256: str, output_sha256: str) -> dict:
    normalized = dict(payload)
    normalized["created"] = "<TIMESTAMP>"
    normalized["runs_root"] = "<RUNS_ROOT>"
    return {
        "basis_equivalence_check": normalized["basis_equivalence_check"],
        "basis_name": normalized["basis_name"],
        "bases_equal": normalized["bases_equal"],
        "created": normalized["created"],
        "family_counts": normalized["family_counts"],
        "inputs": [
            {
                "label": "phase1_geometry_h5",
                "path": "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
                "sha256": phase1_sha256,
            },
            {
                "label": "family_map_v1",
                "path": "experiment/phase2a_atlas_family_map/outputs/family_map_v1.json",
                "sha256": family_map_sha256,
            },
        ],
        "joint_weight_role": normalized["joint_weight_role"],
        "joint_weight_sum_over_support": normalized["joint_weight_sum_over_support"],
        "n_joint_available": normalized["n_joint_available"],
        "n_rows": normalized["n_rows"],
        "outputs": [
            {
                "path": "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
                "sha256": output_sha256,
            }
        ],
        "parameters": normalized["parameters"],
        "results": normalized["results"],
        "run": normalized["run"],
        "runs_root": normalized["runs_root"],
        "schema_version": normalized["schema_version"],
        "source_family_map": normalized["source_family_map"],
        "source_h5": normalized["source_h5"],
        "stage": normalized["stage"],
        "theory_counts": normalized["theory_counts"],
        "verdict": normalized["verdict"],
        "version": normalized["version"],
    }


def test_contract_registered() -> None:
    contract = CONTRACTS.get("experiment/phase2c_support_ontology_basis")
    assert contract is not None
    assert contract.required_inputs == [
        "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
        "experiment/phase2a_atlas_family_map/outputs/family_map_v1.json",
    ]
    assert contract.produced_outputs == ["outputs/support_ontology_basis_v1.json"]


def test_happy_path_selects_final_union_and_matches_normalized_snapshot(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    run_id = "phase2c_support_basis_ok"
    repo_root, runs_root = _prepare_run(
        tmp_path,
        run_id=run_id,
        golden_matches_final=True,
    )

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "OUT_ROOT=" in result.stdout
    assert "STAGE_DIR=" in result.stdout
    assert not (tmp_path / "runs").exists()

    stage_dir = runs_root / run_id / "experiment" / "phase2c_support_ontology_basis"
    output_path = stage_dir / "outputs" / "support_ontology_basis_v1.json"
    summary_path = stage_dir / "stage_summary.json"
    manifest_path = stage_dir / "manifest.json"

    assert output_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_payload == {
        "schema_version": SCHEMA_VERSION,
        "source_h5": "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
        "source_family_map": "experiment/phase2a_atlas_family_map/outputs/family_map_v1.json",
        "basis_name": "final_support_region_union_v1",
        "basis_equivalence_check": "final_support_region_union_vs_golden_post_hawking_union_v1",
        "bases_equal": True,
        "n_events": 2,
        "n_rows": 3,
        "family_counts": {"dcs": 1, "edgb": 1, "kerr_newman": 1},
        "theory_counts": {"EdGB": 1, "Kerr-Newman": 1, "dCS": 1},
        "n_joint_available": 2,
        "joint_weight_sum_over_support": 0.5,
        "joint_weight_role": JOINT_WEIGHT_ROLE,
        "rows": [
            {
                "raw_geometry_id": "geo_edgb",
                "normalized_geometry_id": "geo_edgb",
                "atlas_family": "edgb",
                "atlas_theory": "EdGB",
                "join_mode": "exact_match_v1",
                "join_status": "RESOLVED",
                "support_basis_name": "final_support_region_union_v1",
                "n_events_supported": 2,
                "support_fraction_events": 1.0,
                "in_golden_post_hawking_union": True,
                "in_final_support_region_union": True,
                "joint_available": True,
                "joint_posterior_weight_joint": 0.125,
                "joint_support_count": 2,
                "joint_support_fraction": 1.0,
                "joint_coverage": 1.0,
            },
            {
                "raw_geometry_id": "geo_kn",
                "normalized_geometry_id": "geo_kn_l2m2n0",
                "atlas_family": "kerr_newman",
                "atlas_theory": "Kerr-Newman",
                "join_mode": "normalized_match_l2m2n0_v1",
                "join_status": "RESOLVED",
                "support_basis_name": "final_support_region_union_v1",
                "n_events_supported": 1,
                "support_fraction_events": 0.5,
                "in_golden_post_hawking_union": True,
                "in_final_support_region_union": True,
                "joint_available": True,
                "joint_posterior_weight_joint": 0.375,
                "joint_support_count": 1,
                "joint_support_fraction": 0.5,
                "joint_coverage": 0.5,
            },
            {
                "raw_geometry_id": "geo_dcs",
                "normalized_geometry_id": "geo_dcs",
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
        ],
    }

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    phase1_sha256 = sha256_file(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5"
    )
    family_map_sha256 = sha256_file(
        runs_root / run_id / "experiment" / "phase2a_atlas_family_map" / "outputs" / "family_map_v1.json"
    )
    expected_summary = {
        "basis_equivalence_check": "final_support_region_union_vs_golden_post_hawking_union_v1",
        "basis_name": "final_support_region_union_v1",
        "bases_equal": True,
        "created": "<TIMESTAMP>",
        "family_counts": {"dcs": 1, "edgb": 1, "kerr_newman": 1},
        "inputs": [
            {
                "label": "phase1_geometry_h5",
                "path": "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
                "sha256": phase1_sha256,
            },
            {
                "label": "family_map_v1",
                "path": "experiment/phase2a_atlas_family_map/outputs/family_map_v1.json",
                "sha256": family_map_sha256,
            },
        ],
        "joint_weight_role": JOINT_WEIGHT_ROLE,
        "joint_weight_sum_over_support": 0.5,
        "n_joint_available": 2,
        "n_rows": 3,
        "outputs": [
            {
                "path": "experiment/phase2c_support_ontology_basis/outputs/support_ontology_basis_v1.json",
                "sha256": sha256_file(output_path),
            }
        ],
        "parameters": {
            "basis_equivalence_check": "final_support_region_union_vs_golden_post_hawking_union_v1",
            "basis_name": "final_support_region_union_v1",
            "family_map_name": "family_map_v1.json",
            "input_name": "phase1_geometry_cohort.h5",
            "output_name": "support_ontology_basis_v1.json",
        },
        "results": {
            "bases_equal": True,
            "n_joint_available": 2,
            "n_rows": 3,
        },
        "run": run_id,
        "runs_root": "<RUNS_ROOT>",
        "schema_version": SCHEMA_VERSION,
        "source_family_map": "experiment/phase2a_atlas_family_map/outputs/family_map_v1.json",
        "source_h5": "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
        "stage": "experiment/phase2c_support_ontology_basis",
        "theory_counts": {"EdGB": 1, "Kerr-Newman": 1, "dCS": 1},
        "verdict": "PASS",
        "version": "v1",
    }
    assert _normalized_stage_summary(
        summary_payload,
        phase1_sha256=phase1_sha256,
        family_map_sha256=family_map_sha256,
        output_sha256=sha256_file(output_path),
    ) == expected_summary


def test_selection_uses_final_union_and_records_basis_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    run_id = "phase2c_support_basis_mismatch"
    repo_root, runs_root = _prepare_run(
        tmp_path,
        run_id=run_id,
        golden_matches_final=False,
    )

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout

    output_path = (
        runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["bases_equal"] is False
    assert [row["raw_geometry_id"] for row in payload["rows"]] == ["geo_edgb", "geo_kn", "geo_dcs"]
    assert payload["rows"][2]["raw_geometry_id"] == "geo_dcs"
    assert payload["rows"][2]["in_final_support_region_union"] is True
    assert payload["rows"][2]["in_golden_post_hawking_union"] is False


def test_joint_fields_are_copied_without_renormalization(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    run_id = "phase2c_support_basis_joint_copy"
    repo_root, runs_root = _prepare_run(
        tmp_path,
        run_id=run_id,
        golden_matches_final=True,
    )

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 0, result.stderr + result.stdout

    output_path = (
        runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
    )
    summary_path = runs_root / run_id / "experiment" / "phase2c_support_ontology_basis" / "stage_summary.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = {row["raw_geometry_id"]: row for row in payload["rows"]}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert rows["geo_edgb"]["joint_posterior_weight_joint"] == 0.125
    assert rows["geo_kn"]["joint_posterior_weight_joint"] == 0.375
    assert rows["geo_dcs"]["joint_posterior_weight_joint"] is None
    assert summary["n_joint_available"] == 2
    assert summary["joint_weight_sum_over_support"] == 0.5
    assert summary["joint_weight_role"] == JOINT_WEIGHT_ROLE


def test_fails_if_supported_geometry_missing_from_family_map(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    run_id = "phase2c_support_basis_missing_family_join"
    repo_root, runs_root = _prepare_run(
        tmp_path,
        run_id=run_id,
        golden_matches_final=True,
        include_geo_kn=False,
    )

    result = _run_script(repo_root, run_id, runs_root)
    assert result.returncode == 2
    assert "Supported geometry_id missing from family_map_v1: geo_kn" in (result.stderr + result.stdout)

    stage_dir = runs_root / run_id / "experiment" / "phase2c_support_ontology_basis"
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "Supported geometry_id missing from family_map_v1: geo_kn" in summary["error"]
