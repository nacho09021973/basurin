import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_dictionary_pipeline(tmp_path: Path, run_id: str) -> Path:
    """Run geometry → spectrum → dictionary pipeline."""
    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path("runs") / f"pytest__{tmp_path.name}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Scripts use env var, not --out-root
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


def _run_spectrum_only_pipeline(tmp_path: Path, run_id: str) -> Path:
    """Run spectrum-only pipeline (01_genera_neutrino_sandbox → 04_diccionario)."""
    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path("runs") / f"pytest__{tmp_path.name}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Scripts use env var, not --out-root
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(repo_root / out_root)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "01_genera_neutrino_sandbox.py"),
            "--run",
            run_id,
            "--n-delta",
            "8",
            "--n-modes",
            "3",
            "--noise-rel",
            "0.0",
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
        raise RuntimeError("04_diccionario.py no generó atlas.json en spectrum_only de test.")
    return repo_root


def test_bridge_alignment_defaults_require_features_stage(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    run_id = "bridge-defaults"
    repo_root = _run_dictionary_pipeline(tmp_path, run_id)
    run_root = Path("runs") / f"pytest__{tmp_path.name}"

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(repo_root / run_root)}
    features_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "05_build_features_stage.py"),
            "--run",
            run_id,
            "--out-root",
            str(run_root),
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
            "--no-kill-switch",  # Bypass RUN_VALID gate (not testing that)
            "--bootstrap",
            "2",
            "--perm",
            "2",
            "--k-nn",
            "2",
            "--n-components",
            "2",
            "--out-root",
            str(run_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "dictionary/outputs" not in result.stderr
    assert "tools/05_build_features_stage.py --run" not in result.stderr


def test_bridge_alignment_multi_run(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    run_x = "bridge-run-x"
    run_y = "bridge-run-y"
    repo_root = _run_dictionary_pipeline(tmp_path, run_x)
    _run_spectrum_only_pipeline(tmp_path, run_y)
    run_root = Path("runs") / f"pytest__{tmp_path.name}"
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(repo_root / run_root)}

    for run_id in (run_x, run_y):
        features_result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "tools" / "05_build_features_stage.py"),
                "--run",
                run_id,
                "--out-root",
                str(run_root),
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
            "--run-x",
            run_x,
            "--run-y",
            run_y,
            "--no-kill-switch",  # Bypass RUN_VALID gate (not testing that)
            "--bootstrap",
            "2",
            "--perm",
            "2",
            "--k-nn",
            "2",
            "--n-components",
            "2",
            "--out-root",
            str(run_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "dictionary/outputs" not in result.stderr
    assert "tools/05_build_features_stage.py --run" not in result.stderr
