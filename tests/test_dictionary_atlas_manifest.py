import json
import os
import subprocess
import sys
from pathlib import Path

from basurin_io import sha256_file


def test_dictionary_atlas_manifest_and_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "atlas-manifest"
    runs_root = tmp_path / "runs"

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "01_genera_neutrino_sandbox.py"),
            "--run",
            run_id,
            "--n-delta",
            "6",
            "--n-modes",
            "3",
            "--noise-rel",
            "0.0",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "04_diccionario.py"),
            "--run",
            run_id,
            "--k-features",
            "2",
            "--n-bootstrap",
            "0",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    stage_dir = runs_root / run_id / "dictionary"
    atlas_path = stage_dir / "outputs" / "atlas.json"
    manifest_path = stage_dir / "manifest.json"
    summary_path = stage_dir / "stage_summary.json"

    assert atlas_path.exists()
    atlas_hash = sha256_file(atlas_path)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"]["atlas"] == "outputs/atlas.json"
    assert manifest["hashes"]["outputs/atlas.json"] == atlas_hash

    summary = json.loads(summary_path.read_text())
    assert summary["hashes"]["outputs/atlas.json"] == atlas_hash
    assert summary["outputs"]["atlas"] == "outputs/atlas.json"
    assert summary["outputs"]["dictionary"] == "outputs/dictionary.h5"

    inputs = summary.get("inputs", {})
    assert inputs.get("spectrum_path")
    assert inputs.get("spectrum_sha256")

    # Anti-regression: atlas.json must have top-level ids/X for features stage
    atlas = json.loads(atlas_path.read_text())
    assert "ids" in atlas, "atlas.json missing top-level 'ids'"
    assert "X" in atlas, "atlas.json missing top-level 'X'"
    assert len(atlas["ids"]) == len(atlas["X"]), "ids/X length mismatch"
    k_features = 2  # from --k-features 2 above
    assert len(atlas["X"][0]) == k_features, f"X[0] should have {k_features} features"
