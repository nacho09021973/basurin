import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _write_spectrum(path: Path, n_points: int, n_features: int) -> None:
    rng = np.random.default_rng(123)
    masses = np.abs(rng.normal(loc=1.0, scale=0.3, size=(n_points, n_features))) + 0.1
    delta_uv = rng.normal(loc=0.5, scale=0.1, size=n_points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("masses", data=masses)
        h5.create_dataset("delta_uv", data=delta_uv)


def test_tangentes_locales_exports_features_points(tmp_path: Path) -> None:
    run_id = "tangentes-export"
    spectrum_path = tmp_path / "runs" / run_id / "spectrum" / "outputs" / "spectrum.h5"
    _write_spectrum(spectrum_path, n_points=30, n_features=5)

    script_path = Path(__file__).resolve().parents[1] / "05_tangentes_locales.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run",
            run_id,
            "--k-neighbors",
            "7",
            "--n-points",
            "25",
            "--n-perturb",
            "5",
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

    assert result.returncode == 0, result.stderr

    features_path = (
        tmp_path
        / "runs"
        / run_id
        / "tangentes_locales"
        / "outputs"
        / "features_points_k7.json"
    )
    payload = json.loads(features_path.read_text())

    assert set(payload.keys()) == {"ids", "Y", "meta"}
    assert len(payload["ids"]) == 25
    assert len(payload["Y"]) == 25
    assert payload["meta"]["columns"] == [
        "d_eff",
        "m",
        "parallel",
        "perp",
        "rho_clipped",
        "log10_rho",
    ]
    assert payload["meta"]["k_neighbors"] == 7
    assert len(payload["Y"][0]) == len(payload["meta"]["columns"])
