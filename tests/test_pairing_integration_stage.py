import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_stage_order_policy_ignores_ids(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    run_root = tmp_path / "runs"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = input_dir / "atlas_points.json"
    features_path = input_dir / "event_features.json"

    atlas_payload = {
        "points": [
            {"id": "a", "x": [1.0, 0.0]},
            {"id": "b", "x": [0.0, 1.0]},
            {"id": "c", "x": [1.0, 1.0]},
            {"id": "d", "x": [2.0, 3.0]},
        ]
    }
    features_payload = {
        "events": [
            {"id": "c", "y": [0.0, 2.0]},
            {"id": "a", "y": [2.0, 0.0]},
            {"id": "d", "y": [3.0, 1.0]},
            {"id": "b", "y": [1.0, 3.0]},
        ]
    }

    atlas_path.write_text(json.dumps(atlas_payload))
    features_path.write_text(json.dumps(features_payload))

    stage_path = Path(__file__).resolve().parents[1] / "experiment" / "bridge" / "stage_F4_1_alignment.py"
    result = subprocess.run(
        [
            sys.executable,
            str(stage_path),
            "--run",
            "pairing-order",
            "--atlas",
            str(atlas_path),
            "--features",
            str(features_path),
            "--pairing-policy",
            "order",
            "--bootstrap",
            "2",
            "--perm",
            "2",
            "--k-nn",
            "2",
            "--n-components",
            "1",
            "--seed",
            "7",
            "--no-kill-switch",
            "--out-root",
            str(run_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    outputs_dir = run_root / "pairing-order" / "bridge_f4_1_alignment" / "outputs"
    per_point_path = outputs_dir / "degeneracy_per_point.json"
    per_point = json.loads(per_point_path.read_text())

    assert [row["id"] for row in per_point] == [
        "idx_0",
        "idx_1",
        "idx_2",
        "idx_3",
    ]
    assert per_point[0]["paired_by"] == "order"
    assert [row["pairing_policy"] for row in per_point] == [
        "order",
        "order",
        "order",
        "order",
    ]
    assert [row["atlas_id"] for row in per_point] == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert [row["event_id"] for row in per_point] == [
        "c",
        "a",
        "d",
        "b",
    ]
    assert [row["row_i"] for row in per_point] == [0, 1, 2, 3]
