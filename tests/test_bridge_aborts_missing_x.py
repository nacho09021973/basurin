import json
import subprocess
import sys
from pathlib import Path


def test_bridge_aborts_if_missing_x(tmp_path: Path) -> None:
    run_id = "bridge-missing-x"
    repo_root = Path(__file__).resolve().parents[1]

    atlas_dir = tmp_path / "runs" / run_id / "dictionary" / "outputs"
    atlas_dir.mkdir(parents=True)
    atlas_path = atlas_dir / "atlas.json"
    atlas_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "feature_key": "ratios",
                "ids": ["a", "b", "c"],
                "X": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    features_dir = tmp_path / "runs" / run_id / "features" / "outputs"
    features_dir.mkdir(parents=True)
    features_path = features_dir / "features.json"
    features_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "feature_key": "tangentes_locales_v1",
                "ids": ["a", "b", "c"],
                "Y": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiment" / "bridge" / "stage_F4_1_alignment.py"),
            "--run",
            run_id,
            "--out-root",
            "runs",
            "--no-kill-switch",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0

    summary_path = tmp_path / "runs" / run_id / "bridge_f4_1_alignment" / "stage_summary.json"
    abort_path = tmp_path / "runs" / run_id / "bridge_f4_1_alignment" / "outputs" / "abort_reason.json"
    assert summary_path.exists()
    assert abort_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    abort_payload = json.loads(abort_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ABORT"
    assert summary["abort_reason"] == "MISSING_X"
    assert abort_payload["abort_reason"] == "MISSING_X"
