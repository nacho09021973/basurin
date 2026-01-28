import json
import os
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
    runs_root = tmp_path / "runs"
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}

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
        env=env,
    )

    assert result.returncode != 0

    stage_dir = runs_root / "abort_case" / "spectrum"
    summary_path = stage_dir / "stage_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["reason"] == "EFT_DOMAIN_VIOLATION"
    assert payload["contracts"]["EFT_DOMAIN"]["status"] == "FAIL"
    assert payload["contracts"]["EFT_DOMAIN"]["max_abs_A_minus_1"] >= payload["contracts"]["EFT_DOMAIN"]["threshold"]

    assert not (stage_dir / "outputs" / "spectrum.h5").exists()
    assert not (stage_dir / "manifest.json").exists()


def test_eft_domain_pass(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "01_genera_neutrino_sandbox.py"
    runs_root = tmp_path / "runs"
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}

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
        env=env,
    )

    assert result.returncode == 0

    stage_dir = runs_root / "pass_case" / "spectrum"
    assert (stage_dir / "outputs" / "spectrum.h5").exists()
    assert (stage_dir / "manifest.json").exists()

    summary_path = stage_dir / "stage_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["contracts"]["EFT_DOMAIN"]["status"] == "PASS"
    assert payload["contracts"]["EFT_DOMAIN"]["max_abs_A_minus_1"] < payload["contracts"]["EFT_DOMAIN"]["threshold"]


def _run_neutrino_grid(tmp_path: Path, args: list[str]) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "01_genera_neutrino_sandbox.py"
    runs_root = tmp_path / "runs"
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}

    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr

    stage_dir = runs_root / args[args.index("--run") + 1] / "spectrum"
    summary_path = stage_dir / "stage_summary.json"
    assert summary_path.exists()
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_paired_grid_is_colinear(tmp_path: Path) -> None:
    payload = _run_neutrino_grid(
        tmp_path,
        [
            "--run",
            "paired_case",
            "--n-delta",
            "10",
            "--n-modes",
            "2",
            "--n-grid",
            "32",
            "--profiles",
            "core,crust",
            "--grid-mode",
            "paired",
        ],
    )

    grid = payload["grid"]
    alpha = np.array(grid["alpha_per_point"], dtype=np.float64)
    delta = np.array(grid["delta_per_point"], dtype=np.float64)
    corr = np.corrcoef(alpha, delta)[0, 1]

    assert corr > 0.99


def test_cartesian_grid_breaks_colinearity(tmp_path: Path) -> None:
    payload = _run_neutrino_grid(
        tmp_path,
        [
            "--run",
            "cartesian_case",
            "--n-delta",
            "5",
            "--n-alpha",
            "4",
            "--n-modes",
            "2",
            "--n-grid",
            "32",
            "--profiles",
            "core,crust",
            "--grid-mode",
            "cartesian",
        ],
    )

    grid = payload["grid"]
    assert grid["n_total"] == 20
    alpha = np.array(grid["alpha_per_point"], dtype=np.float64)
    delta = np.array(grid["delta_per_point"], dtype=np.float64)
    corr = np.corrcoef(alpha, delta)[0, 1]

    assert abs(corr) < 0.2


def test_cartesian_config_traces_grid_mapping(tmp_path: Path) -> None:
    payload = _run_neutrino_grid(
        tmp_path,
        [
            "--run",
            "cartesian_config_case",
            "--n-delta",
            "5",
            "--n-alpha",
            "4",
            "--n-modes",
            "2",
            "--n-grid",
            "32",
            "--profiles",
            "core,crust",
            "--grid-mode",
            "cartesian",
        ],
    )

    config = payload["config"]
    assert config["grid_mode"] == "cartesian"
    assert "alpha_values" in config
    assert "delta_values" in config
    assert "grid_order" in config
    assert len(config["alpha_values"]) == int(config["n_alpha"])
    assert len(config["delta_values"]) == int(config["n_delta"])

    alpha_values = np.array(config["alpha_values"], dtype=np.float64)
    delta_values = np.array(config["delta_values"], dtype=np.float64)

    if config["grid_order"] == "delta_outer_alpha_inner":
        delta_per_point = np.repeat(delta_values, int(config["n_alpha"]))
        alpha_per_point = np.tile(alpha_values, int(config["n_delta"]))
    else:
        raise AssertionError(f"Unexpected grid_order: {config['grid_order']}")

    n_total = len(alpha_per_point)
    assert n_total == int(config["n_alpha"]) * int(config["n_delta"])

    corr = np.corrcoef(alpha_per_point, delta_per_point)[0, 1]
    assert abs(corr) < 0.2


def test_dictionary_proxy_hard_contracts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}

    run_name = "dictionary_proxy_case"
    generator = repo_root / "01_genera_neutrino_sandbox.py"
    dictionary = repo_root / "04_diccionario.py"

    gen_result = subprocess.run(
        [
            sys.executable,
            str(generator),
            "--run",
            run_name,
            "--grid-mode",
            "cartesian",
            "--n-delta",
            "5",
            "--n-alpha",
            "4",
            "--n-grid",
            "32",
            "--noise-rel",
            "0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert gen_result.returncode == 0, gen_result.stderr

    dict_result = subprocess.run(
        [sys.executable, str(dictionary), "--run", run_name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert dict_result.returncode == 0, dict_result.stderr

    stage_dir = runs_root / run_name / "dictionary"
    summary_path = stage_dir / "stage_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation_summary = summary["validation_summary"]

    assert validation_summary["hard_contracts_profile"] == "proxy"
    assert validation_summary["all_hard_contracts_pass"] is True

    outputs_dir = stage_dir / "outputs"
    assert (outputs_dir / "dictionary.h5").exists()
    assert (outputs_dir / "atlas.json").exists()
    assert (stage_dir / "manifest.json").exists()
