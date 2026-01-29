import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_atlas_index(path: Path, run_id: str) -> None:
    _write_json(
        path,
        {
            "schema_version": "atlas_index_v1",
            "runs": {
                run_id: {
                    "dim": 4,
                }
            },
            "by_dim": {
                "4": {
                    "path": "runs/atlas_master_dim4/atlas_master/outputs/BRIDGE_ATLAS_MASTER.json",
                }
            },
        },
    )


def test_atlas_select_aborts_on_failed_run(tmp_path: Path) -> None:
    run_id = "atlas-select-fail"
    repo_root = Path(__file__).resolve().parents[1]

    verdict_dir = tmp_path / "runs" / run_id / "RUN_VALID"
    verdict_dir.mkdir(parents=True)
    verdict_path = verdict_dir / "verdict.json"
    _write_json(verdict_path, {"overall_verdict": "FAIL"})

    atlas_index_path = (
        tmp_path / "runs" / "atlas_index_run" / "atlas_index" / "outputs" / "ATLAS_INDEX.json"
    )
    atlas_index_path.parent.mkdir(parents=True)
    _make_atlas_index(atlas_index_path, run_id)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "work" / "07_atlas_select_stage.py"),
            "--run",
            run_id,
            "--atlas-index",
            str(atlas_index_path),
            "--out-root",
            "runs",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2

    output_path = tmp_path / "runs" / run_id / "atlas_select" / "outputs" / "ATLAS_SELECTION.json"
    assert not output_path.exists()


def test_atlas_select_writes_outputs_on_pass(tmp_path: Path) -> None:
    run_id = "atlas-select-pass"
    repo_root = Path(__file__).resolve().parents[1]

    verdict_dir = tmp_path / "runs" / run_id / "RUN_VALID"
    verdict_dir.mkdir(parents=True)
    verdict_path = verdict_dir / "verdict.json"
    _write_json(verdict_path, {"overall_verdict": "PASS"})

    atlas_index_path = (
        tmp_path / "runs" / "atlas_index_run" / "atlas_index" / "outputs" / "ATLAS_INDEX.json"
    )
    atlas_index_path.parent.mkdir(parents=True)
    _make_atlas_index(atlas_index_path, run_id)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "work" / "07_atlas_select_stage.py"),
            "--run",
            run_id,
            "--atlas-index",
            str(atlas_index_path),
            "--out-root",
            "runs",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    output_path = tmp_path / "runs" / run_id / "atlas_select" / "outputs" / "ATLAS_SELECTION.json"
    manifest_path = tmp_path / "runs" / run_id / "atlas_select" / "manifest.json"
    assert output_path.exists()
    assert manifest_path.exists()
