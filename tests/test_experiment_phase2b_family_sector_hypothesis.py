from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase2b_family_sector_hypothesis.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_utf8_dataset(group: object, name: str, values: list[str]) -> None:
    import h5py  # type: ignore

    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, (len(values),), dtype=dt)[...] = values


def _make_phase1_h5(path: Path, geometry_ids: list[str], supported_indices: list[int]) -> None:
    h5py = pytest.importorskip("h5py")

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        atlas = h5.create_group("atlas")
        membership = h5.create_group("membership")
        _write_utf8_dataset(atlas, "geometry_id", geometry_ids)
        matrix = [[False for _ in geometry_ids]]
        for idx in supported_indices:
            matrix[0][idx] = True
        membership.create_dataset("golden_post_hawking", data=matrix)


def _make_family_map(path: Path, rows: list[dict]) -> None:
    counts_family: dict[str, int] = {}
    counts_theory: dict[str, int] = {}
    for row in rows:
        counts_family[row["atlas_family"]] = counts_family.get(row["atlas_family"], 0) + 1
        counts_theory[row["atlas_theory"]] = counts_theory.get(row["atlas_theory"], 0) + 1
    _write_json(
        path,
        {
            "schema_version": "family_map_v1",
            "normalization_policy_name": "exact_or_normalized_l2m2n0_v1",
            "source_h5": "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
            "source_atlas": "atlas_test.json",
            "n_rows": len(rows),
            "family_counts": counts_family,
            "theory_counts": counts_theory,
            "join_mode_counts": {},
            "unresolved_geometry_ids": [],
            "rows": rows,
        },
    )


def _make_rules(path: Path, rules: list[dict]) -> None:
    _write_json(
        path,
        {
            "schema_version": "family_to_sector_rules_v1",
            "rules": rules,
        },
    )


def _run_script(repo_root: Path, run_id: str, runs_root: Path, rules_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / SCRIPT),
            "--run-id",
            run_id,
            "--rules-path",
            str(rules_path),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _family_row(raw_gid: str, family: str, theory: str) -> dict:
    return {
        "raw_geometry_id": raw_gid,
        "normalized_geometry_id": raw_gid,
        "join_mode": "exact_match_v1",
        "join_status": "RESOLVED",
        "atlas_path": "atlas_test.json",
        "atlas_geometry_id": raw_gid,
        "atlas_family": family,
        "atlas_theory": theory,
        "criterion": "exact_match_v1",
        "criterion_version": "v1",
        "evidence_fields_used": ["h5.atlas.geometry_id"],
        "evidence": {"h5.atlas.geometry_id": raw_gid},
    }


def _rule(rule_id: str, family: str, theory: str, sector: str) -> dict:
    return {
        "rule_id": rule_id,
        "match": {
            "atlas_family": family,
            "atlas_theory": theory,
        },
        "proposed_sector": sector,
        "criterion": "test_rule_v1",
        "criterion_version": "v1",
        "rule_source": f"rules.json#{rule_id}",
        "confidence_class": "LOW",
        "notes": "test",
    }


def test_contract_registered() -> None:
    contract = CONTRACTS.get("experiment/phase2b_family_sector_hypothesis")
    assert contract is not None
    assert contract.required_inputs == [
        "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
        "experiment/phase2a_atlas_family_map/outputs/family_map_v1.json",
    ]
    assert contract.external_inputs == ["rules"]
    assert contract.produced_outputs == ["outputs/family_sector_hypothesis_v1.json"]


def test_rule_applied_correctly(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2b_rule_applied"
    rules_path = tmp_path / "rules_applied.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        ["g1"],
        [0],
    )
    _make_family_map(
        runs_root / run_id / "experiment" / "phase2a_atlas_family_map" / "outputs" / "family_map_v1.json",
        [_family_row("g1", "edgb", "EdGB")],
    )
    _make_rules(
        rules_path,
        [_rule("r1", "edgb", "EdGB", "HYPERBOLIC")],
    )

    result = _run_script(repo_root, run_id, runs_root, rules_path)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / "experiment" / "phase2b_family_sector_hypothesis"
    payload = json.loads((stage_dir / "outputs" / "family_sector_hypothesis_v1.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["rule_status"] == "RULE_APPLIED"
    assert row["proposed_sector"] == "HYPERBOLIC"
    assert payload["downstream_ready"] is True


def test_family_supported_without_rule_sets_downstream_ready_false(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2b_no_rule"
    rules_path = tmp_path / "rules_empty.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        ["g1"],
        [0],
    )
    _make_family_map(
        runs_root / run_id / "experiment" / "phase2a_atlas_family_map" / "outputs" / "family_map_v1.json",
        [_family_row("g1", "edgb", "EdGB")],
    )
    _make_rules(rules_path, [])

    result = _run_script(repo_root, run_id, runs_root, rules_path)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / "experiment" / "phase2b_family_sector_hypothesis"
    payload = json.loads((stage_dir / "outputs" / "family_sector_hypothesis_v1.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["rule_status"] == "NO_RULE"
    assert row["proposed_sector"] == "UNKNOWN"
    assert payload["supported_family_coverage"] == 0.0
    assert payload["downstream_ready"] is False


def test_conflict_between_rules_sets_conflict_count(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2b_conflict"
    rules_path = tmp_path / "rules_conflict.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        ["g1"],
        [0],
    )
    _make_family_map(
        runs_root / run_id / "experiment" / "phase2a_atlas_family_map" / "outputs" / "family_map_v1.json",
        [_family_row("g1", "dcs", "dCS")],
    )
    _make_rules(
        rules_path,
        [
            _rule("r1", "dcs", "dCS", "HYPERBOLIC"),
            _rule("r2", "dcs", "dCS", "ELLIPTIC"),
        ],
    )

    result = _run_script(repo_root, run_id, runs_root, rules_path)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / "experiment" / "phase2b_family_sector_hypothesis"
    payload = json.loads((stage_dir / "outputs" / "family_sector_hypothesis_v1.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["rule_status"] == "CONFLICT"
    assert payload["conflict_count"] == 1
    assert payload["downstream_ready"] is False


def test_prohibits_derivation_from_geometry_id_without_rule(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2b_no_geometry_id_inference"
    rules_path = tmp_path / "rules_none.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        ["Kerr_M70_a0.3046_l2m2n0"],
        [0],
    )
    _make_family_map(
        runs_root / run_id / "experiment" / "phase2a_atlas_family_map" / "outputs" / "family_map_v1.json",
        [_family_row("Kerr_M70_a0.3046_l2m2n0", "kerr", "GR_Kerr")],
    )
    _make_rules(rules_path, [])

    result = _run_script(repo_root, run_id, runs_root, rules_path)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / "experiment" / "phase2b_family_sector_hypothesis"
    payload = json.loads((stage_dir / "outputs" / "family_sector_hypothesis_v1.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["raw_geometry_id"] == "Kerr_M70_a0.3046_l2m2n0"
    assert row["rule_status"] == "NO_RULE"
    assert row["proposed_sector"] == "UNKNOWN"
    assert payload["downstream_ready"] is False
