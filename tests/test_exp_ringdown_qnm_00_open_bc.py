"""Regression tests for EXP_RINGDOWN_QNM_00_open_bc fit metrics."""
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


def _run_open_bc(tmp_path: Path, repo_root: Path, run_id: str, extra_args: list[str]) -> Path:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run",
            run_id,
            *extra_args,
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"
    return run_dir


def test_tau_s_no_overflow(tmp_path: Path, repo_root: Path):
    """Ensure tau_s does not overflow to huge float values."""
    run_dir = _run_open_bc(
        tmp_path,
        repo_root,
        "test_tau_s_no_overflow",
        ["--f-hz", "250.0", "--tau-s", "0.004", "--seed", "42"],
    )

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    per_case_dir = outputs_dir / "per_case"
    for case_path in per_case_dir.glob("case_*.json"):
        case = _rjson(case_path)
        tau_s = case.get("tau_s")
        if tau_s is None:
            continue
        assert math.isfinite(tau_s), f"tau_s should be finite, got {tau_s}"
        assert 0.0 < tau_s < 1e6, f"tau_s out of range: {tau_s}"
        assert tau_s < 1e308, f"tau_s overflow detected: {tau_s}"


def test_fit_r2_defined_on_log_envelope(tmp_path: Path, repo_root: Path):
    """Ensure fit_r2 is computed on log-envelope and is meaningful."""
    run_dir = _run_open_bc(
        tmp_path,
        repo_root,
        "test_fit_r2_defined",
        [
            "--f-hz",
            "250.0",
            "--tau-s",
            "0.004",
            "--grid-sweep",
            "N=1024",
            "--window-sweep",
            "w1",
            "--seed",
            "42",
        ],
    )

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    case = _rjson(outputs_dir / "per_case" / "case_000.json")
    assert case["fit_r2"] >= 0.0
    assert case["fit_r2"] >= 0.9
    assert case["fit_method"] == "log_envelope_linear"
    assert case["fit_eps"] > 0


def test_contract_passes_demo(tmp_path: Path, repo_root: Path):
    """Ensure demo configuration produces PASS verdict."""
    run_dir = _run_open_bc(
        tmp_path,
        repo_root,
        "test_contract_pass_demo",
        [
            "--f-hz",
            "250.0",
            "--tau-s",
            "0.004",
            "--grid-sweep",
            "N=1024",
            "--window-sweep",
            "w1",
            "--seed",
            "42",
        ],
    )
    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    verdict = _rjson(outputs_dir / "contract_verdict.json")
    assert verdict["verdict"] == "PASS"
