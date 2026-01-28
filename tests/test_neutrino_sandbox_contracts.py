import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_neutrino_module(repo_root: Path):
    module_path = repo_root / "01_genera_neutrino_sandbox.py"
    spec = spec_from_file_location("neutrino_sandbox", module_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_symmetron_A0_normalization() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_neutrino_module(repo_root)

    rho0 = 2.0
    rho_crit = 4.0
    alpha = 0.5
    rho = np.array([rho0], dtype=np.float64)

    A_norm, A0_raw = module.A_symmetron_normalized(
        rho, alpha_s0=alpha, rho_crit=rho_crit, rho0=rho0
    )
    expected_A0_raw = 1.0 + alpha * (1.0 - rho0 / rho_crit)

    np.testing.assert_allclose(A_norm[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(A0_raw, expected_A0_raw, atol=1e-12)


def test_eft_domain_abort(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "01_genera_neutrino_sandbox.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            "abort_case",
            "--family",
            "eft_power",
            "--n-delta",
            "2",
            "--n-modes",
            "2",
            "--n-grid",
            "32",
            "--profiles",
            "core,crust",
            "--alpha-min",
            "2.0",
            "--alpha-max",
            "2.0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0

    abort_path = tmp_path / "runs" / "abort_case" / "spectrum" / "outputs" / "abort_domain.json"
    assert abort_path.exists()
    payload = json.loads(abort_path.read_text(encoding="utf-8"))
    assert payload["reason"] == "EFT_DOMAIN_VIOLATION"
    assert payload["model"] == "eft_power"
    assert payload["max_abs_delta"] >= payload["threshold"]


def test_eft_domain_pass(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "01_genera_neutrino_sandbox.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            "pass_case",
            "--family",
            "eft_power",
            "--n-delta",
            "2",
            "--n-modes",
            "2",
            "--n-grid",
            "32",
            "--profiles",
            "core,crust",
            "--alpha-min",
            "0.1",
            "--alpha-max",
            "0.1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0

    abort_path = tmp_path / "runs" / "pass_case" / "spectrum" / "outputs" / "abort_domain.json"
    assert not abort_path.exists()
