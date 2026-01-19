import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _write_spectrum(path: Path, n_samples: int = 12, n_modes: int = 4) -> None:
    delta_uv = np.linspace(1.0, 2.0, n_samples)
    m2_base = 1.0 + 0.05 * np.arange(n_samples)
    M2 = np.stack([
        m2_base,
        1.6 * m2_base + 0.01,
        2.3 * m2_base + 0.02,
        3.1 * m2_base + 0.03,
    ], axis=1)
    z_grid = np.linspace(0.0, 1.0, 5)
    m2L2 = M2.copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("delta_uv", data=delta_uv)
        h5.create_dataset("m2L2", data=m2L2)
        h5.create_dataset("M2", data=M2)
        h5.create_dataset("z_grid", data=z_grid)
        h5.attrs["d"] = 4
        h5.attrs["L"] = 1.0
        h5.attrs["n_delta"] = n_samples
        h5.attrs["n_modes"] = n_modes


def test_c3_adaptive_validation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "04_diccionario.py"

    run_name = "c3_adaptive_test"
    spectrum_path = tmp_path / "runs" / run_name / "spectrum" / "outputs" / "spectrum.h5"
    _write_spectrum(spectrum_path)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            run_name,
            "--enable-c3",
            "--k-features",
            "3",
            "--direct-model",
            "poly",
            "--direct-degree",
            "2",
            "--c3-adaptive-threshold",
            "--c3-threshold-factor",
            "3.0",
            "--c3-threshold",
            "0.0",
            "--n-bootstrap",
            "5",
            "--seed",
            "123",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1)

    validation_path = tmp_path / "runs" / run_name / "dictionary" / "outputs" / "validation.json"
    assert validation_path.exists()

    data = json.loads(validation_path.read_text())

    bootstrap = data.get("bootstrap", {})
    assert bootstrap.get("sigma_delta_mean") is not None
    assert bootstrap.get("sigma_delta_p90") is not None

    c3 = data.get("C3_spectral", {})
    assert c3.get("c3a_decoder", {}).get("global") is not None
    assert c3.get("c3b_cycle", {}).get("global") is not None
    assert c3.get("threshold_used") is not None
    assert c3.get("tol_cycle") is not None
    assert c3.get("sensitivity", {}).get("s_p90") is not None

    threshold_used = c3.get("threshold_used")
    threshold_user = c3.get("threshold", {}).get("user")
    assert threshold_used != threshold_user
