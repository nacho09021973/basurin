import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _write_spectrum(path: Path, *, delta_uv: np.ndarray, m2: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("delta_uv", data=delta_uv)
        h5.create_dataset("M2", data=m2)


def test_mix_wrapper_matches_new(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_new = repo_root / "02_mix_spectra.py"
    script_legacy = repo_root / "01_mix_spectra.py"

    delta_uv = np.array([1.0, 2.0])
    m2_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    m2_b = np.array([[5.0, 6.0], [7.0, 8.0]])

    _write_spectrum(
        tmp_path / "runs" / "run_a" / "spectrum" / "outputs" / "spectrum.h5",
        delta_uv=delta_uv,
        m2=m2_a,
    )
    _write_spectrum(
        tmp_path / "runs" / "run_b" / "spectrum" / "outputs" / "spectrum.h5",
        delta_uv=delta_uv,
        m2=m2_b,
    )

    subprocess.run(
        [
            sys.executable,
            str(script_new),
            "--run-out",
            "run_mix_new",
            "--run-a",
            "run_a",
            "--run-b",
            "run_b",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script_legacy),
            "--run-out",
            "run_mix_legacy",
            "--run-a",
            "run_a",
            "--run-b",
            "run_b",
        ],
        cwd=tmp_path,
        check=True,
    )

    new_path = tmp_path / "runs" / "run_mix_new" / "spectrum" / "outputs" / "spectrum.h5"
    legacy_path = tmp_path / "runs" / "run_mix_legacy" / "spectrum" / "outputs" / "spectrum.h5"

    with h5py.File(new_path, "r") as new_h5, h5py.File(legacy_path, "r") as legacy_h5:
        np.testing.assert_allclose(new_h5["delta_uv"][:], legacy_h5["delta_uv"][:])
        np.testing.assert_allclose(new_h5["M2"][:], legacy_h5["M2"][:])


def test_neutrino_sandbox_stage_metadata(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "01_genera_neutrino_sandbox.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            "sandbox_test",
            "--n-delta",
            "3",
            "--n-modes",
            "2",
            "--n-grid",
            "32",
            "--profiles",
            "vacuum,crust",
        ],
        cwd=tmp_path,
        check=True,
    )

    stage_dir = tmp_path / "runs" / "sandbox_test" / "spectrum"
    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["stage"] == "spectrum"
    assert summary["script"] == "01_genera_neutrino_sandbox.py"
    assert manifest["stage"] == "spectrum"
