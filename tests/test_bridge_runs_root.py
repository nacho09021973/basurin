import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_dictionary_pipeline(tmp_path: Path, run_id: str, out_root: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(repo_root / out_root)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "01_genera_ads_puro.py"),
            "--run",
            run_id,
            "--n",
            "20",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    geom_h5 = repo_root / out_root / run_id / "geometry" / "outputs" / "ads_puro.h5"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "03_sturm_liouville.py"),
            "--run",
            run_id,
            "--geometry-file",
            str(geom_h5),
            "--n-delta",
            "5",
            "--n-modes",
            "4",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    result = subprocess.run(
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
        cwd=repo_root,
        env=env,
        check=False,
    )
    atlas_path = repo_root / out_root / run_id / "dictionary" / "outputs" / "atlas.json"
    if result.returncode != 0 and not atlas_path.exists():
        raise RuntimeError("04_diccionario.py no generó atlas.json en el pipeline de test.")
    return repo_root


def test_bridge_accepts_alternate_runs_root(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    run_id = "bridge-alt-root"
    out_root = Path("runs_tmp") / f"pytest__{tmp_path.name}"
    repo_root = _run_dictionary_pipeline(tmp_path, run_id, out_root)
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(repo_root / out_root)}

    features_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "05_build_features_stage.py"),
            "--run",
            run_id,
            "--out-root",
            str(out_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert features_result.returncode == 0, features_result.stderr

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiment" / "bridge" / "stage_F4_1_alignment.py"),
            "--run",
            run_id,
            "--out-root",
            str(out_root),
            "--no-kill-switch",
            "--bootstrap",
            "1",
            "--perm",
            "1",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr

    summary_path = (
        repo_root
        / out_root
        / run_id
        / "bridge_f4_1_alignment"
        / "stage_summary.json"
    )
    assert summary_path.exists()
