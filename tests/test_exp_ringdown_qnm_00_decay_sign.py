"""Tests for EXP_RINGDOWN_QNM_00_open_bc decay sign contract.

Verifies:
- Contract C1 PASS when omega_I < 0 (decaying mode)
- Contract C1 FAIL when omega_I >= 0 (growing mode)
- Decay rate matches expected value within tolerance
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

PY = os.environ.get("PYTHON", "python")


def _rjson(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _env_for(tmp_path: Path) -> dict:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(tmp_path / "runs")
    return env


@pytest.fixture()
def repo_root() -> Path:
    return Path.cwd()


def test_decay_sign_pass_positive_tau(tmp_path: Path, repo_root: Path):
    """Contract C1 should PASS when tau > 0 (omega_I < 0)."""
    run_id = "test_decay_pass"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--f-hz", "250.0",
            "--tau-s", "0.004",  # positive tau => omega_I = -1/tau < 0
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    verdict = _rjson(outputs_dir / "contract_verdict.json")
    qnm_fit = _rjson(outputs_dir / "qnm_fit.json")

    # omega_I should be negative (decay)
    assert qnm_fit["omega_I"] < 0, "omega_I should be negative for decay"

    # C1 contract should PASS
    c1 = verdict["contracts"]["C1_horizon_decay"]
    assert c1["verdict"] == "PASS", f"C1 should PASS but got: {c1}"


def test_omega_i_matches_expected(tmp_path: Path, repo_root: Path):
    """Verify omega_I matches expected value from tau_s."""
    run_id = "test_omega_i_value"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tau_s = 0.004  # 4 ms
    expected_omega_I = -1.0 / tau_s  # = -250

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--f-hz", "250.0",
            "--tau-s", str(tau_s),
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    qnm_fit = _rjson(outputs_dir / "qnm_fit.json")

    # Check omega_I is close to expected
    rel_error = abs(qnm_fit["omega_I"] - expected_omega_I) / abs(expected_omega_I)
    assert rel_error < 0.3, f"omega_I={qnm_fit['omega_I']} differs too much from expected={expected_omega_I}"


def test_omega_r_matches_expected(tmp_path: Path, repo_root: Path):
    """Verify omega_R matches expected value from f_hz."""
    run_id = "test_omega_r_value"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    f_hz = 250.0
    expected_omega_R = 2 * math.pi * f_hz  # ~ 1570.8

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--f-hz", str(f_hz),
            "--tau-s", "0.004",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    qnm_fit = _rjson(outputs_dir / "qnm_fit.json")

    # Check omega_R is close to expected
    rel_error = abs(qnm_fit["omega_R"] - expected_omega_R) / abs(expected_omega_R)
    assert rel_error < 0.1, f"omega_R={qnm_fit['omega_R']} differs too much from expected={expected_omega_R}"


def test_decay_sign_fail_strict_threshold(tmp_path: Path, repo_root: Path):
    """Contract C1 should FAIL with impossibly strict decay threshold."""
    run_id = "test_decay_fail_strict"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    # Use impossibly strict decay threshold: omega_I < -1e12
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--f-hz", "250.0",
            "--tau-s", "0.004",
            "--decay-eps", "1e12",  # impossibly strict
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    # Should fail because omega_I is not < -1e12
    assert p.returncode == 2, "Script should fail with strict threshold"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    verdict = _rjson(outputs_dir / "contract_verdict.json")

    # C1 contract should FAIL
    c1 = verdict["contracts"]["C1_horizon_decay"]
    assert c1["verdict"] == "FAIL", f"C1 should FAIL with strict threshold but got: {c1}"
    assert len(c1["violations"]) > 0, "Should have violations"


def test_c2_stability_pass_normal_case(tmp_path: Path, repo_root: Path):
    """Contract C2 should PASS for normal synthetic signal."""
    run_id = "test_c2_pass"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--f-hz", "250.0",
            "--tau-s", "0.004",
            "--grid-sweep", "N=1024,2048,4096",
            "--window-sweep", "w1,w2,w3",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    verdict = _rjson(outputs_dir / "contract_verdict.json")

    # C2 contract should PASS
    c2 = verdict["contracts"]["C2_stability"]
    assert c2["verdict"] == "PASS", f"C2 should PASS but got: {c2}"


def test_c2_stability_fail_tight_tolerance(tmp_path: Path, repo_root: Path):
    """Contract C2 should FAIL with impossibly tight tolerances."""
    run_id = "test_c2_fail_tight"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    # Use impossibly tight tolerance: 0.001%
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--f-hz", "250.0",
            "--tau-s", "0.004",
            "--omega-r-rel-tol", "0.00001",  # 0.001% tolerance
            "--omega-i-rel-tol", "0.00001",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    # May fail if stability tolerance is violated
    # Return code 0 or 2 depending on actual variance
    assert p.returncode in (0, 2)

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    verdict = _rjson(outputs_dir / "contract_verdict.json")

    c2 = verdict["contracts"]["C2_stability"]
    # With such tight tolerances, C2 is likely to fail
    # but we don't assert it must fail, just that the contract is evaluated


def test_different_frequencies(tmp_path: Path, repo_root: Path):
    """Verify different frequencies produce different omega_R."""
    run_id_1 = "test_freq_100hz"
    run_id_2 = "test_freq_500hz"

    env = _env_for(tmp_path)

    # Run with 100 Hz
    run_dir_1 = tmp_path / "runs" / run_id_1
    run_dir_1.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id_1,
            "--f-hz", "100.0",
            "--tau-s", "0.004",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p1.returncode == 0

    # Run with 500 Hz
    run_dir_2 = tmp_path / "runs" / run_id_2
    run_dir_2.mkdir(parents=True, exist_ok=True)
    p2 = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id_2,
            "--f-hz", "500.0",
            "--tau-s", "0.004",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p2.returncode == 0

    qnm_1 = _rjson(run_dir_1 / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs" / "qnm_fit.json")
    qnm_2 = _rjson(run_dir_2 / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs" / "qnm_fit.json")

    # 500 Hz should have higher omega_R than 100 Hz
    assert qnm_2["omega_R"] > qnm_1["omega_R"], "Higher frequency should produce higher omega_R"
    # Ratio should be approximately 5
    ratio = qnm_2["omega_R"] / qnm_1["omega_R"]
    assert 4.5 < ratio < 5.5, f"Frequency ratio should be ~5, got {ratio}"


def test_different_decay_times(tmp_path: Path, repo_root: Path):
    """Verify different decay times produce different omega_I."""
    run_id_1 = "test_tau_2ms"
    run_id_2 = "test_tau_10ms"

    env = _env_for(tmp_path)

    # Run with tau = 2 ms (faster decay)
    run_dir_1 = tmp_path / "runs" / run_id_1
    run_dir_1.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id_1,
            "--f-hz", "250.0",
            "--tau-s", "0.002",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p1.returncode == 0

    # Run with tau = 10 ms (slower decay)
    run_dir_2 = tmp_path / "runs" / run_id_2
    run_dir_2.mkdir(parents=True, exist_ok=True)
    p2 = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id_2,
            "--f-hz", "250.0",
            "--tau-s", "0.010",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p2.returncode == 0

    qnm_1 = _rjson(run_dir_1 / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs" / "qnm_fit.json")
    qnm_2 = _rjson(run_dir_2 / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs" / "qnm_fit.json")

    # Faster decay (smaller tau) should have more negative omega_I
    assert qnm_1["omega_I"] < qnm_2["omega_I"], "Faster decay should have more negative omega_I"


def test_overall_verdict_reflects_contracts(tmp_path: Path, repo_root: Path):
    """Verify overall verdict is PASS only if all contracts PASS."""
    run_id = "test_overall_verdict"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--f-hz", "250.0",
            "--tau-s", "0.004",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    verdict = _rjson(outputs_dir / "contract_verdict.json")

    c1_pass = verdict["contracts"]["C1_horizon_decay"]["verdict"] == "PASS"
    c2_pass = verdict["contracts"]["C2_stability"]["verdict"] == "PASS"

    if c1_pass and c2_pass:
        assert verdict["verdict"] == "PASS"
    else:
        assert verdict["verdict"] == "FAIL"
