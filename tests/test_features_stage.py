import json
import os
import subprocess
import sys
from pathlib import Path


def _run_dictionary_pipeline(tmp_path: Path, run_id: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "01_genera_ads_puro.py"),
            "--run",
            run_id,
            "--n",
            "20",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "03_sturm_liouville.py"),
            "--run",
            run_id,
            "--n-delta",
            "5",
            "--n-modes",
            "4",
        ],
        cwd=tmp_path,
        env={**os.environ, "BASURIN_RUNS_ROOT": str(tmp_path / "runs")},
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "04_diccionario.py"),
            "--run",
            run_id,
        ],
        cwd=tmp_path,
        check=False,
    )
    atlas_path = tmp_path / "runs" / run_id / "dictionary" / "outputs" / "atlas.json"
    if result.returncode != 0 and not atlas_path.exists():
        raise RuntimeError("04_diccionario.py no generó atlas.json en el pipeline de test.")
    return repo_root


def test_features_stage_builds_features_json(tmp_path: Path) -> None:
    run_id = "features-stage"
    repo_root = _run_dictionary_pipeline(tmp_path, run_id)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "05_build_features_stage.py"),
            "--run",
            run_id,
        ],
        cwd=tmp_path,
        check=True,
    )

    features_path = tmp_path / "runs" / run_id / "features" / "outputs" / "features.json"
    payload = json.loads(features_path.read_text(encoding="utf-8"))

    assert "ids" in payload
    assert "X" in payload
    assert "feature_names" in payload
    assert len(payload["ids"]) == len(payload["X"])
