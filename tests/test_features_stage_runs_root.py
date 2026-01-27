"""Anti-regression test for BASURIN_RUNS_ROOT support in 05_build_features_stage.py."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_features_stage_respects_basurin_runs_root(tmp_path: Path) -> None:
    """Verify 05_build_features_stage.py respects BASURIN_RUNS_ROOT for out-root validation.

    Regression test for: validation was hardcoded to repo_root/runs instead of
    using get_runs_root() to support BASURIN_RUNS_ROOT environment variable.
    """
    pytest.importorskip("sklearn")

    repo_root = Path(__file__).resolve().parents[1]
    run_id = "features-runs-root-test"

    # Create runs_root inside runs_tmp (NOT the default runs/)
    runs_root = repo_root / "runs_tmp" / f"pytest__{tmp_path.name}"
    runs_root.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}

    # Generate minimal pipeline to produce dictionary/outputs/atlas.json
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

    atlas_path = runs_root / run_id / "dictionary" / "outputs" / "atlas.json"
    assert atlas_path.exists(), f"atlas.json not created at {atlas_path}"

    # Run 05_build_features_stage.py with --out-root pointing to runs_tmp/pytest__...
    # This should succeed because BASURIN_RUNS_ROOT is set to that directory
    out_root_rel = runs_root.relative_to(repo_root)
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "05_build_features_stage.py"),
            "--run",
            run_id,
            "--out-root",
            str(out_root_rel),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"05_build_features_stage.py failed with --out-root {out_root_rel}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Verify outputs are written under the correct runs_root
    features_path = runs_root / run_id / "features" / "outputs" / "features.json"
    assert features_path.exists(), f"features.json not created at {features_path}"

    manifest_path = runs_root / run_id / "features" / "manifest.json"
    assert manifest_path.exists(), f"manifest.json not created at {manifest_path}"

    # Verify features.json has expected structure
    features = json.loads(features_path.read_text())
    assert "ids" in features, "features.json missing 'ids'"
    assert "Y" in features, "features.json missing 'Y'"
    assert "X_path" in features, "features.json missing 'X_path'"
    assert "Y_path" in features, "features.json missing 'Y_path'"
    assert features.get("feature_key") == "tangentes_locales_v1"
