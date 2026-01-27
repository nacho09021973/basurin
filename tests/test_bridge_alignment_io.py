import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._helpers_features import write_minimal_canonical_features


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _corrs_within_bounds(corrs: list[float]) -> bool:
    return all(np.isnan(c) or abs(c) <= 1.0 for c in corrs)


def test_bridge_alignment_runs_with_features_points(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    run_root = Path("runs") / f"pytest__{tmp_path.name}"
    run_id = "bridge-io"
    run_dir = run_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    # BASURIN IO governance: inputs must be under runs/<run_id>/inputs
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(9)
    ids = [f"id_{i}" for i in range(30)]
    x_values = rng.normal(size=(30, 3)).tolist()
    atlas = {
        "ids": ids,
        "X": x_values,
        "meta": {"feature_key": "ratios"},
    }
    y_values = rng.normal(size=(30, 6)).tolist()
    features = {
        "ids": ids,
        "Y": y_values,
        "meta": {"feature_key": "tangentes_locales_v1"},
        "X_path": "../features/outputs/X.npy",
    }

    atlas_path = input_dir / "atlas_bridge.json"
    features_path = input_dir / "features_points_k7.json"
    _write_json(atlas_path, atlas)
    _write_json(features_path, features)
    write_minimal_canonical_features(
        run_dir,
        n=len(ids),
        dx=3,
        dy=6,
        feature_key=features["meta"]["feature_key"],
        seed=101,
    )

    stage_path = Path(__file__).resolve().parents[1] / "experiment" / "bridge" / "stage_F4_1_alignment.py"
    repo_root = Path(__file__).resolve().parents[1]
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
            "--no-kill-switch",  # Bypass RUN_VALID gate (not testing that)
            "--bootstrap",
            "2",
            "--perm",
            "2",
            "--k-nn",
            "2",
            "--n-components",
            "2",
            "--seed",
            "11",
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
    assert (outputs_dir / "metrics.json").exists()
    alignment = json.loads((outputs_dir / "alignment_map.json").read_text())
    corrs = alignment["cca"]["canonical_corrs"]
    assert _corrs_within_bounds(corrs)
    manifest = json.loads((outputs_dir.parent / "manifest.json").read_text())
    assert "files" in manifest
    assert manifest["files"]["metrics"] == "outputs/metrics.json"
    assert all(not Path(path).is_absolute() for path in manifest["files"].values())


def test_bridge_alignment_aborts_on_matching_feature_key(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    run_root = Path("runs") / f"pytest__{tmp_path.name}"
    run_id = "bridge-leakage"
    run_dir = run_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    # BASURIN IO governance: inputs must be under runs/<run_id>/inputs
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(3)
    ids = [f"id_{i}" for i in range(20)]
    shared = rng.normal(size=(20, 3)).tolist()
    atlas = {
        "ids": ids,
        "X": shared,
        "meta": {"feature_key": "ratios"},
    }
    features = {
        "ids": ids,
        "Y": shared,
        "meta": {"feature_key": "ratios"},
        "X_path": "../features/outputs/X.npy",
    }

    atlas_path = input_dir / "atlas_bridge.json"
    features_path = input_dir / "features_points_k7.json"
    _write_json(atlas_path, atlas)
    _write_json(features_path, features)
    write_minimal_canonical_features(
        run_dir,
        n=len(ids),
        dx=3,
        dy=3,
        feature_key=features["meta"]["feature_key"],
        seed=202,
    )

    stage_path = Path(__file__).resolve().parents[1] / "experiment" / "bridge" / "stage_F4_1_alignment.py"
    repo_root = Path(__file__).resolve().parents[1]
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
            "--no-kill-switch",  # Bypass RUN_VALID gate (not testing that)
            "--bootstrap",
            "1",
            "--perm",
            "1",
            "--k-nn",
            "2",
            "--n-components",
            "2",
            "--seed",
            "5",
            "--out-root",
            str(run_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2

    outputs_dir = repo_root / run_root / run_id / "bridge_f4_1_alignment" / "outputs"
    assert (outputs_dir / "abort_leakage.json").exists()
    manifest = json.loads((outputs_dir.parent / "manifest.json").read_text())
    assert manifest["files"]["abort_leakage"] == "outputs/abort_leakage.json"


def test_bridge_alignment_filters_constant_columns(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    run_root = Path("runs") / f"pytest__{tmp_path.name}"
    run_id = "bridge-constant-column"
    run_dir = run_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    # BASURIN IO governance: inputs must be under runs/<run_id>/inputs
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(13)
    ids = [f"id_{i}" for i in range(25)]
    x_values = rng.normal(size=(25, 4)).tolist()
    atlas = {
        "ids": ids,
        "X": x_values,
        "meta": {"feature_key": "ratios"},
    }
    y_values = rng.normal(size=(25, 5))
    y_values[:, 2] = 0.0
    features = {
        "ids": ids,
        "Y": y_values.tolist(),
        "meta": {"feature_key": "tangentes_locales_v1"},
        "X_path": "../features/outputs/X.npy",
    }

    atlas_path = input_dir / "atlas_bridge.json"
    features_path = input_dir / "features_points_k7.json"
    _write_json(atlas_path, atlas)
    _write_json(features_path, features)
    write_minimal_canonical_features(
        run_dir,
        n=len(ids),
        dx=4,
        dy=5,
        feature_key=features["meta"]["feature_key"],
        seed=303,
    )

    stage_path = Path(__file__).resolve().parents[1] / "experiment" / "bridge" / "stage_F4_1_alignment.py"
    repo_root = Path(__file__).resolve().parents[1]
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
            "--no-kill-switch",  # Bypass RUN_VALID gate (not testing that)
            "--bootstrap",
            "1",
            "--perm",
            "1",
            "--k-nn",
            "2",
            "--n-components",
            "2",
            "--seed",
            "7",
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
    alignment = json.loads((outputs_dir / "alignment_map.json").read_text())
    data_used = alignment["cca"]["data_used"]
    assert 2 in data_used["y_dropped_idx"]
    assert data_used["y_dim_used"] == data_used["y_dim_original"] - 1
    assert _corrs_within_bounds(alignment["cca"]["canonical_corrs"])
