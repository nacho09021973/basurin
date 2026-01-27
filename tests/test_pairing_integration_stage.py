import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._helpers_features import write_minimal_canonical_features


def test_stage_order_policy_ignores_ids(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    run_root = Path("runs") / f"pytest__{tmp_path.name}"
    run_id = "pairing-order"
    run_dir = run_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    # BASURIN IO governance: inputs must be under runs/<run_id>/inputs
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = input_dir / "atlas_points.json"
    features_path = input_dir / "event_features.json"

    atlas_payload = {
        "ids": ["a", "b", "c", "d"],
        "X": [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 3.0],
        ],
        "meta": {"feature_key": "atlas_vectors"},
    }
    features_payload = {
        "ids": ["c", "a", "d", "b"],
        "Y": [
            [0.0, 2.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ],
        "meta": {"feature_key": "event_vectors"},
        "X_path": "../features/outputs/X.npy",
    }

    atlas_path.write_text(json.dumps(atlas_payload))
    features_path.write_text(json.dumps(features_payload))
    write_minimal_canonical_features(
        run_dir,
        n=len(atlas_payload["ids"]),
        dx=2,
        dy=2,
        feature_key=features_payload["meta"]["feature_key"],
        seed=404,
    )

    repo_root = Path(__file__).resolve().parents[1]
    stage_path = repo_root / "experiment" / "bridge" / "stage_F4_1_alignment.py"
    result = subprocess.run(
        [
            sys.executable,
            str(stage_path),
            "--run",
            run_id,
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
            "--no-kill-switch",  # Bypass RUN_VALID gate (not testing that)
            "--out-root",
            str(run_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    outputs_dir = repo_root / run_root / run_id / "bridge_f4_1_alignment" / "outputs"
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
