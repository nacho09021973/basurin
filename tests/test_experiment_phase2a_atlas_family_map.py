from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from mvp.contracts import CONTRACTS

SCRIPT = Path("mvp/experiment_phase2a_atlas_family_map.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_utf8_dataset(group: object, name: str, values: list[str]) -> None:
    import h5py  # type: ignore

    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, (len(values),), dtype=dt)[...] = values


def _make_phase1_h5(path: Path, geometry_ids: list[str]) -> None:
    h5py = pytest.importorskip("h5py")

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        atlas = h5.create_group("atlas")
        _write_utf8_dataset(atlas, "geometry_id", geometry_ids)


def _make_atlas(path: Path, entries: list[dict]) -> None:
    _write_json(
        path,
        {
            "schema_version": "atlas_test_v1",
            "entries": entries,
        },
    )


def _run_script(repo_root: Path, run_id: str, runs_root: Path, atlas_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(runs_root)
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / SCRIPT),
            "--run-id",
            run_id,
            "--atlas-path",
            str(atlas_path),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_contract_registered() -> None:
    contract = CONTRACTS.get("experiment/phase2a_atlas_family_map")
    assert contract is not None
    assert contract.required_inputs == [
        "experiment/phase1_geometry_h5/outputs/phase1_geometry_cohort.h5",
    ]
    assert contract.external_inputs == ["atlas"]
    assert contract.produced_outputs == ["outputs/family_map_v1.json"]


def test_family_map_exact_match_preserved(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2a_exact_match"
    atlas_path = tmp_path / "atlas_exact.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        ["Kerr_M35_a0.0000_l2m2n0"],
    )
    _make_atlas(
        atlas_path,
        [
            {
                "geometry_id": "Kerr_M35_a0.0000_l2m2n0",
                "theory": "GR_Kerr",
                "metadata": {"family": "kerr"},
            }
        ],
    )

    result = _run_script(repo_root, run_id, runs_root, atlas_path)
    assert result.returncode == 0, result.stderr + result.stdout

    output_path = runs_root / run_id / "experiment" / "phase2a_atlas_family_map" / "outputs" / "family_map_v1.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["raw_geometry_id"] == "Kerr_M35_a0.0000_l2m2n0"
    assert row["normalized_geometry_id"] == "Kerr_M35_a0.0000_l2m2n0"
    assert row["join_mode"] == "exact_match_v1"
    assert row["join_status"] == "RESOLVED"
    assert row["atlas_family"] == "kerr"
    assert row["atlas_theory"] == "GR_Kerr"


def test_family_map_recovers_truncated_kerr_ids_with_l2m2n0(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2a_norm_match"
    atlas_path = tmp_path / "atlas_norm.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        ["Kerr_M100_a0.6600"],
    )
    _make_atlas(
        atlas_path,
        [
            {
                "geometry_id": "Kerr_M100_a0.6600_l2m2n0",
                "theory": "GR_Kerr",
                "metadata": {"family": "kerr"},
            }
        ],
    )

    result = _run_script(repo_root, run_id, runs_root, atlas_path)
    assert result.returncode == 0, result.stderr + result.stdout

    output_path = runs_root / run_id / "experiment" / "phase2a_atlas_family_map" / "outputs" / "family_map_v1.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["raw_geometry_id"] == "Kerr_M100_a0.6600"
    assert row["normalized_geometry_id"] == "Kerr_M100_a0.6600_l2m2n0"
    assert row["join_mode"] == "normalized_match_l2m2n0_v1"
    assert row["atlas_geometry_id"] == "Kerr_M100_a0.6600_l2m2n0"
    assert row["atlas_family"] == "kerr"
    assert row["atlas_theory"] == "GR_Kerr"


def test_family_map_aborts_on_unresolved_rows(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2a_unresolved"
    atlas_path = tmp_path / "atlas_missing.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        ["Kerr_M200_a0.1234"],
    )
    _make_atlas(atlas_path, [])

    result = _run_script(repo_root, run_id, runs_root, atlas_path)
    assert result.returncode == 2

    stage_dir = runs_root / run_id / "experiment" / "phase2a_atlas_family_map"
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "Unresolved atlas family/theory join" in summary["error"]
    assert not (stage_dir / "outputs" / "family_map_v1.json").exists()


def test_family_map_does_not_infer_from_substrings_without_atlas_join(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2a_no_substring_infer"
    atlas_path = tmp_path / "atlas_other.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        ["EdGB_M62_a0.00_z0.1"],
    )
    _make_atlas(
        atlas_path,
        [
            {
                "geometry_id": "Different_M62_a0.00_z0.1",
                "theory": "GR_Kerr",
                "metadata": {"family": "kerr"},
            }
        ],
    )

    result = _run_script(repo_root, run_id, runs_root, atlas_path)
    assert result.returncode == 2

    stage_dir = runs_root / run_id / "experiment" / "phase2a_atlas_family_map"
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "FAIL"
    assert "EdGB_M62_a0.00_z0.1" in summary["error"]


def test_family_map_stage_summary_counts_match_output_rows(tmp_path: Path) -> None:
    pytest.importorskip("h5py")

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "isolated_runs"
    run_id = "phase2a_counts"
    atlas_path = tmp_path / "atlas_counts.json"

    _write_json(runs_root / run_id / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    geometry_ids = [
        "Kerr_M35_a0.0000_l2m2n0",
        "Kerr_M100_a0.6600",
        "EdGB_M62_a0.00_z0.1",
        "KN_M62_a0.00_q0.1",
        "dCS_M100_a0.80_z0.1",
    ]
    _make_phase1_h5(
        runs_root / run_id / "experiment" / "phase1_geometry_h5" / "outputs" / "phase1_geometry_cohort.h5",
        geometry_ids,
    )
    _make_atlas(
        atlas_path,
        [
            {
                "geometry_id": "Kerr_M35_a0.0000_l2m2n0",
                "theory": "GR_Kerr",
                "metadata": {"family": "kerr"},
            },
            {
                "geometry_id": "Kerr_M100_a0.6600_l2m2n0",
                "theory": "GR_Kerr",
                "metadata": {"family": "kerr"},
            },
            {
                "geometry_id": "EdGB_M62_a0.00_z0.1",
                "theory": "EdGB",
                "metadata": {"family": "edgb"},
            },
            {
                "geometry_id": "KN_M62_a0.00_q0.1",
                "theory": "Kerr-Newman",
                "metadata": {"family": "kerr_newman"},
            },
            {
                "geometry_id": "dCS_M100_a0.80_z0.1",
                "theory": "dCS",
                "metadata": {"family": "dcs"},
            },
        ],
    )

    result = _run_script(repo_root, run_id, runs_root, atlas_path)
    assert result.returncode == 0, result.stderr + result.stdout

    stage_dir = runs_root / run_id / "experiment" / "phase2a_atlas_family_map"
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    output = json.loads((stage_dir / "outputs" / "family_map_v1.json").read_text(encoding="utf-8"))

    family_counts = Counter(row["atlas_family"] for row in output["rows"])
    theory_counts = Counter(row["atlas_theory"] for row in output["rows"])
    join_mode_counts = Counter(row["join_mode"] for row in output["rows"])

    assert summary["n_rows"] == len(output["rows"]) == 5
    assert summary["n_exact_match"] == 4
    assert summary["n_normalized_match"] == 1
    assert summary["n_unresolved"] == 0
    assert summary["family_counts"] == dict(family_counts)
    assert summary["theory_counts"] == dict(theory_counts)
    assert summary["join_mode_counts"] == {
        "exact_match_v1": 4,
        "normalized_match_l2m2n0_v1": 1,
    }
