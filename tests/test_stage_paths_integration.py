import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def _write_spectrum(path: Path, n_points: int, n_features: int) -> None:
    rng = np.random.default_rng(41)
    masses = np.abs(rng.normal(loc=1.0, scale=0.3, size=(n_points, n_features))) + 0.1
    delta_uv = rng.normal(loc=0.5, scale=0.1, size=n_points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("masses", data=masses)
        h5.create_dataset("delta_uv", data=delta_uv)


def test_tangentes_then_bridge_stage_paths(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    run_id = "mini-run"
    run_root = tmp_path / "runs"

    spectrum_path = run_root / run_id / "spectrum" / "outputs" / "spectrum.h5"
    _write_spectrum(spectrum_path, n_points=25, n_features=5)

    atlas_points_path = run_root / run_id / "dictionary" / "outputs" / "atlas_points.json"
    atlas_points_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_points = {
        "source_atlas": f"runs/{run_id}/dictionary/outputs/atlas.json",
        "feature_key": "ratios",
        "n_points": 25,
        "points": [
            {"id": f"atlas_{i}", "features": [float(i), float(i + 1)]}
            for i in range(25)
        ],
    }
    atlas_points_path.write_text(json.dumps(atlas_points, indent=2))

    tangentes_script = Path(__file__).resolve().parents[1] / "05_tangentes_locales.py"
    tangentes = subprocess.run(
        [
            sys.executable,
            str(tangentes_script),
            "--run",
            run_id,
            "--k-neighbors",
            "7",
            "--n-points",
            "25",
            "--n-perturb",
            "3",
            "--k-features",
            "4",
            "--seed",
            "7",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tangentes.returncode == 0, tangentes.stderr

    features_path = (
        run_root
        / run_id
        / "tangentes_locales"
        / "outputs"
        / "features_points_k7.json"
    )
    assert features_path.exists()

    bridge_script = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "bridge"
        / "stage_F4_1_alignment.py"
    )
    bridge = subprocess.run(
        [
            sys.executable,
            str(bridge_script),
            "--run",
            run_id,
            "--atlas",
            str(atlas_points_path),
            "--features",
            str(features_path),
            "--bootstrap",
            "1",
            "--perm",
            "1",
            "--k-nn",
            "2",
            "--n-components",
            "2",
            "--seed",
            "11",
            "--out-root",
            str(run_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bridge.returncode == 0, bridge.stderr

    tangentes_stage = run_root / run_id / "tangentes_locales"
    tangentes_manifest = json.loads((tangentes_stage / "manifest.json").read_text())
    assert (tangentes_stage / "stage_summary.json").exists()
    assert tangentes_manifest["files"]["results"] == "outputs/results.json"
    assert all(
        not Path(path).is_absolute() for path in tangentes_manifest["files"].values()
    )

    bridge_stage = run_root / run_id / "bridge_f4_1_alignment"
    bridge_manifest = json.loads((bridge_stage / "manifest.json").read_text())
    assert (bridge_stage / "stage_summary.json").exists()
    assert bridge_manifest["files"]["metrics"] == "outputs/metrics.json"
    assert all(
        not Path(path).is_absolute() for path in bridge_manifest["files"].values()
    )

    bridge_summary = json.loads((bridge_stage / "stage_summary.json").read_text())
    assert bridge_summary["pairing"]["paired_by"] == "id"
    assert bridge_summary["pairing"]["n_common"] == 25
