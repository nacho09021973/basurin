import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_atlas_index(path: Path, master_path: str, master_sha: str) -> None:
    _write_json(
        path,
        {
            "schema_version": "atlas_index_v1",
            "items": [
                {
                    "name": "dim4",
                    "path": master_path,
                    "sha256": master_sha,
                },
                {
                    "name": "dim6",
                    "path": "runs/atlas_master_dim6/atlas_master/outputs/BRIDGE_ATLAS_MASTER.json",
                    "sha256": "deadbeef",
                },
            ],
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
    _make_atlas_index(atlas_index_path, "runs/unused.json", "deadbeef")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "07_atlas_select_stage.py"),
            "--run",
            run_id,
            "--atlas-index",
            str(atlas_index_path),
            "--force-dim",
            "4",
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
    master_run = "2026-01-29__ATLAS_MASTER__dim4__v1"
    master_rel_path = (
        f"{master_run}/atlas_master/outputs/BRIDGE_ATLAS_MASTER.json"
    )
    master_path = tmp_path / "runs" / master_rel_path
    master_path.parent.mkdir(parents=True)
    master_payload = b'{"ok": true}\n'
    master_path.write_bytes(master_payload)
    master_sha = _sha256_bytes(master_payload)

    _make_atlas_index(atlas_index_path, master_rel_path, master_sha)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "07_atlas_select_stage.py"),
            "--run",
            run_id,
            "--atlas-index",
            str(atlas_index_path),
            "--force-dim",
            "4",
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

    selection = json.loads(output_path.read_text(encoding="utf-8"))
    assert selection["selected"]["atlas_master_sha256"] == master_sha
    assert selection["selected"]["atlas_master_sha256_computed"] == master_sha


def test_atlas_select_fails_on_sha_mismatch(tmp_path: Path) -> None:
    run_id = "atlas-select-sha-mismatch"
    repo_root = Path(__file__).resolve().parents[1]

    verdict_dir = tmp_path / "runs" / run_id / "RUN_VALID"
    verdict_dir.mkdir(parents=True)
    verdict_path = verdict_dir / "verdict.json"
    _write_json(verdict_path, {"overall_verdict": "PASS"})

    atlas_index_path = (
        tmp_path / "runs" / "atlas_index_run" / "atlas_index" / "outputs" / "ATLAS_INDEX.json"
    )
    atlas_index_path.parent.mkdir(parents=True)

    master_run = "2026-01-29__ATLAS_MASTER__dim4__v1"
    master_rel_path = (
        f"{master_run}/atlas_master/outputs/BRIDGE_ATLAS_MASTER.json"
    )
    master_path = tmp_path / "runs" / master_rel_path
    master_path.parent.mkdir(parents=True)
    master_path.write_text('{"ok": true}\n', encoding="utf-8")

    _make_atlas_index(atlas_index_path, master_rel_path, "badsha")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "07_atlas_select_stage.py"),
            "--run",
            run_id,
            "--atlas-index",
            str(atlas_index_path),
            "--force-dim",
            "4",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2

    output_path = tmp_path / "runs" / run_id / "atlas_select" / "outputs" / "ATLAS_SELECTION.json"
    assert not output_path.exists()
