"""Tests for EXP_RINGDOWN_QNM_01 closed-open limit comparison.

Verifies:
- Contract C3: closed limit recovery (ω_R² → M², ω_I → 0)
- Contract C4: monotonicity (|ω_I| increases with absorption)
- IO contract: all required artifacts created
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PY = os.environ.get("PYTHON", "python")


def _wjson(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _rjson(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _env_for(tmp_path: Path) -> dict:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(tmp_path / "runs")
    return env


def _create_mock_spectrum_h5(path: Path, M2_values: list[float] | None = None) -> None:
    """Create a minimal spectrum.h5 for testing."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        pytest.skip("h5py or numpy not available")

    if M2_values is None:
        # Default: 3 modes with M² = 100, 400, 900 (ω = 10, 20, 30)
        M2_values = [100.0, 400.0, 900.0]

    path.parent.mkdir(parents=True, exist_ok=True)

    n_delta = 1
    n_modes = len(M2_values)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("z_grid", data=np.linspace(0.001, 1.0, 100))
        h5.create_dataset("delta_uv", data=np.array([1.55]))
        h5.create_dataset("m2L2", data=np.array([0.0]))
        h5.create_dataset("M2", data=np.array([M2_values]))

        h5.attrs["d"] = 3
        h5.attrs["L"] = 1.0
        h5.attrs["n_delta"] = n_delta
        h5.attrs["n_modes"] = n_modes
        h5.attrs["bc_uv"] = "dirichlet"
        h5.attrs["bc_ir"] = "dirichlet"


@pytest.fixture()
def repo_root() -> Path:
    return Path.cwd()


def test_creates_required_artifacts(tmp_path: Path, repo_root: Path):
    """Verify all required artifacts are created."""
    run_id = "test_closed_open_artifacts"
    run_dir = tmp_path / "runs" / run_id

    # Create mock spectrum
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    _create_mock_spectrum_h5(spectrum_path)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    stage_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_01_closed_open_limit"
    outputs_dir = stage_dir / "outputs"

    # Check required files
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (outputs_dir / "comparison.json").exists()
    assert (outputs_dir / "per_mode_results.json").exists()
    assert (outputs_dir / "contract_verdict.json").exists()


def test_c3_closed_limit_recovery(tmp_path: Path, repo_root: Path):
    """Contract C3: ω_R² → M² when absorption → 0."""
    run_id = "test_c3_closed_limit"
    run_dir = tmp_path / "runs" / run_id

    # Create mock spectrum with known M² values
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    _create_mock_spectrum_h5(spectrum_path, M2_values=[100.0, 400.0, 900.0])

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
            "--gamma-sweep", "0.0,0.01,0.1,1.0",
            "--mode-indices", "0,1,2",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_01_closed_open_limit" / "outputs"
    verdict = _rjson(outputs_dir / "contract_verdict.json")
    per_mode = _rjson(outputs_dir / "per_mode_results.json")

    # C3 should PASS
    c3 = verdict["contracts"]["C3_closed_limit_recovery"]
    assert c3["verdict"] == "PASS", f"C3 failed: {c3}"

    # Gamma=0 closed limit should recover M² and zero decay
    for mode_key, results in per_mode.items():
        gamma_zero = next(r for r in results if r["gamma"] == 0.0)
        omega_I_abs = abs(gamma_zero["omega_I"])
        rel_error = gamma_zero["omega_R_sq_rel_error"]
        assert omega_I_abs <= 1e-6, f"{mode_key} omega_I too large: {omega_I_abs}"
        assert rel_error <= 0.05, f"{mode_key} ω_R² rel error too large: {rel_error}"


def test_c4_monotonicity(tmp_path: Path, repo_root: Path):
    """Contract C4: |ω_I| increases with absorption."""
    run_id = "test_c4_monotonicity"
    run_dir = tmp_path / "runs" / run_id

    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    _create_mock_spectrum_h5(spectrum_path, M2_values=[100.0])

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
            "--gamma-sweep", "0.0,1.0,10.0,100.0",
            "--mode-indices", "0",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_01_closed_open_limit" / "outputs"
    verdict = _rjson(outputs_dir / "contract_verdict.json")
    per_mode = _rjson(outputs_dir / "per_mode_results.json")

    # C4 should PASS
    c4 = verdict["contracts"]["C4_monotonicity"]
    assert c4["verdict"] == "PASS", f"C4 failed: {c4}"

    # Monotonicity should hold for strictly increasing gamma
    results = sorted(per_mode["mode_0"], key=lambda r: r["gamma"])
    for i in range(len(results) - 1):
        gamma_i = results[i]["gamma"]
        gamma_j = results[i + 1]["gamma"]
        assert gamma_j > gamma_i
        omega_i = abs(results[i]["omega_I"])
        omega_j = abs(results[i + 1]["omega_I"])
        assert omega_j >= omega_i, f"omega_I decreased from {gamma_i} to {gamma_j}"


def test_per_mode_results_structure(tmp_path: Path, repo_root: Path):
    """Verify per_mode_results.json has correct structure."""
    run_id = "test_per_mode_structure"
    run_dir = tmp_path / "runs" / run_id

    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    _create_mock_spectrum_h5(spectrum_path, M2_values=[100.0, 400.0])

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
            "--gamma-sweep", "0.0,1.0",
            "--mode-indices", "0,1",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_01_closed_open_limit" / "outputs"
    per_mode = _rjson(outputs_dir / "per_mode_results.json")

    # Should have results for both modes
    assert "mode_0" in per_mode
    assert "mode_1" in per_mode

    # Each mode should have results for each gamma
    assert len(per_mode["mode_0"]) == 2  # gamma=0, gamma=1
    assert len(per_mode["mode_1"]) == 2

    # Check structure of individual result
    result = per_mode["mode_0"][0]
    assert "gamma" in result
    assert "omega_R" in result
    assert "omega_I" in result
    assert "M2" in result
    assert "omega_R_sq_rel_error" in result
    assert "omega_I_abs" in result
    assert "temporal_convention" in result


def test_comparison_json_structure(tmp_path: Path, repo_root: Path):
    """Verify comparison.json has correct structure."""
    run_id = "test_comparison_structure"
    run_dir = tmp_path / "runs" / run_id

    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    _create_mock_spectrum_h5(spectrum_path)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_01_closed_open_limit" / "outputs"
    comparison = _rjson(outputs_dir / "comparison.json")

    # Check required fields
    assert comparison["schema_version"] == "closed_open_comparison_v1"
    assert "spectrum_source" in comparison
    assert "spectrum_sha256" in comparison
    assert "delta" in comparison
    assert "gamma_values" in comparison
    assert "summary" in comparison
    assert "bloque_b" in comparison

    # Check Bloque B metadata
    assert comparison["bloque_b"]["d"] == 3
    assert comparison["bloque_b"]["bc_uv"] == "dirichlet"


def test_with_upstream_run(tmp_path: Path, repo_root: Path):
    """Test using --upstream-run to specify spectrum source."""
    upstream_run = "upstream_spectrum_run"
    experiment_run = "experiment_run"

    # Create spectrum in upstream run
    upstream_dir = tmp_path / "runs" / upstream_run
    spectrum_path = upstream_dir / "spectrum" / "outputs" / "spectrum.h5"
    _create_mock_spectrum_h5(spectrum_path)

    # Create experiment run directory
    exp_dir = tmp_path / "runs" / experiment_run
    exp_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", experiment_run,
            "--upstream-run", upstream_run,
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    # Verify outputs in experiment run
    stage_dir = exp_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_01_closed_open_limit"
    assert stage_dir.exists()


def test_fails_without_spectrum(tmp_path: Path, repo_root: Path):
    """Test that script fails gracefully without spectrum.h5."""
    run_id = "test_no_spectrum"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2, "Should fail without spectrum.h5"


def test_omega_r_sq_approaches_m2(tmp_path: Path, repo_root: Path):
    """Verify ω_R² is close to M² for small gamma."""
    run_id = "test_omega_r_sq_value"
    run_dir = tmp_path / "runs" / run_id

    # Known M² = 400 → ω_R = 20
    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    _create_mock_spectrum_h5(spectrum_path, M2_values=[400.0])

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
            "--gamma-sweep", "0.0,0.01",
            "--mode-indices", "0",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_01_closed_open_limit" / "outputs"
    per_mode = _rjson(outputs_dir / "per_mode_results.json")

    # For gamma=0, ω_R² should be close to M² = 400
    result_gamma_0 = per_mode["mode_0"][0]
    assert result_gamma_0["gamma"] == 0.0

    omega_R_sq = result_gamma_0["omega_R_sq_fit"]
    M2_target = result_gamma_0["M2_target"]

    rel_error = abs(omega_R_sq - M2_target) / M2_target
    assert rel_error < 0.1, f"ω_R²={omega_R_sq} too far from M²={M2_target}"


def test_deterministic_results(tmp_path: Path, repo_root: Path):
    """Verify same inputs produce same outputs."""
    import shutil

    run_id = "test_deterministic"
    run_dir = tmp_path / "runs" / run_id

    spectrum_path = run_dir / "spectrum" / "outputs" / "spectrum.h5"
    _create_mock_spectrum_h5(spectrum_path)

    env = _env_for(tmp_path)

    # First run
    p1 = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p1.returncode == 0

    stage_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_01_closed_open_limit"
    per_mode_1 = _rjson(stage_dir / "outputs" / "per_mode_results.json")

    # Clear and re-run
    shutil.rmtree(stage_dir)

    p2 = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_01_closed_open_limit.py",
            "--run", run_id,
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p2.returncode == 0

    per_mode_2 = _rjson(stage_dir / "outputs" / "per_mode_results.json")

    # Results should be identical
    assert per_mode_1 == per_mode_2, "Results not deterministic"
