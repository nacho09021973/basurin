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


def test_bridge_alignment_defaults_require_features_stage(tmp_path: Path) -> None:
    run_id = "bridge-defaults"
    repo_root = _run_dictionary_pipeline(tmp_path, run_id)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiment" / "bridge" / "stage_F4_1_alignment.py"),
            "--run",
            run_id,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Primero ejecutar tools/05_build_features_stage.py --run" in result.stderr
