"""Tests for EXP_RINGDOWN_QNM_00_open_bc IO contract.

Verifies:
- All outputs are written under runs/<run_id>/
- Required artifacts exist: manifest.json, stage_summary.json, qnm_fit.json, contract_verdict.json
- Manifest hashes match file contents
- No writes outside sandbox
"""
from __future__ import annotations

import hashlib
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


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_for(tmp_path: Path) -> dict:
    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(tmp_path / "runs")
    return env


@pytest.fixture()
def repo_root() -> Path:
    return Path.cwd()


def test_creates_required_artifacts(tmp_path: Path, repo_root: Path):
    """Verify all required artifacts are created."""
    run_id = "test_qnm_00_io_artifacts"
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
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    stage_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc"
    outputs_dir = stage_dir / "outputs"

    # Check required files exist
    assert stage_dir.exists(), "Stage directory not created"
    assert (stage_dir / "manifest.json").exists(), "manifest.json missing"
    assert (stage_dir / "stage_summary.json").exists(), "stage_summary.json missing"
    assert (outputs_dir / "qnm_fit.json").exists(), "qnm_fit.json missing"
    assert (outputs_dir / "contract_verdict.json").exists(), "contract_verdict.json missing"

    # Check per_case directory has files
    per_case_dir = outputs_dir / "per_case"
    assert per_case_dir.exists(), "per_case directory missing"
    case_files = list(per_case_dir.glob("case_*.json"))
    assert len(case_files) > 0, "No per-case results generated"


def test_manifest_hashes_match(tmp_path: Path, repo_root: Path):
    """Verify manifest hashes match actual file hashes."""
    run_id = "test_qnm_00_hash_match"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    stage_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc"
    manifest = _rjson(stage_dir / "manifest.json")

    # Verify hashes for all listed files
    for rel_path, expected_hash in manifest["hashes"].items():
        full_path = stage_dir / rel_path
        assert full_path.exists(), f"File in manifest does not exist: {rel_path}"
        actual_hash = _sha256(full_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}"


def test_no_writes_outside_runs(tmp_path: Path, repo_root: Path):
    """Verify no files are written outside runs/<run_id>/."""
    run_id = "test_qnm_00_sandbox"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Record all files in tmp_path before execution
    before = set()
    for f in tmp_path.rglob("*"):
        if f.is_file():
            before.add(str(f.relative_to(tmp_path)))

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    # Record all files after execution
    after = set()
    for f in tmp_path.rglob("*"):
        if f.is_file():
            after.add(str(f.relative_to(tmp_path)))

    new_files = after - before
    for f in new_files:
        assert f.startswith(f"runs/{run_id}/"), f"File written outside run dir: {f}"


def test_stage_summary_structure(tmp_path: Path, repo_root: Path):
    """Verify stage_summary.json has required fields."""
    run_id = "test_qnm_00_summary"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
            "--seed", "42",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    stage_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc"
    summary = _rjson(stage_dir / "stage_summary.json")

    # Check required fields
    assert "stage" in summary
    assert "run" in summary
    assert "created" in summary
    assert "params" in summary
    assert "outputs" in summary
    assert "verdict" in summary

    # Check verdict is valid
    assert summary["verdict"] in ("PASS", "FAIL")


def test_qnm_fit_structure(tmp_path: Path, repo_root: Path):
    """Verify qnm_fit.json has required fields."""
    run_id = "test_qnm_00_fit"
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
    assert p.returncode == 0, f"Script failed: {p.stderr}"

    outputs_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc" / "outputs"
    qnm_fit = _rjson(outputs_dir / "qnm_fit.json")

    # Check required fields
    assert "schema_version" in qnm_fit
    assert qnm_fit["schema_version"] == "qnm_fit_v1"
    assert "omega_complex" in qnm_fit
    assert "omega_R" in qnm_fit
    assert "omega_I" in qnm_fit
    assert "f_hz" in qnm_fit
    assert "tau_s" in qnm_fit
    assert "n_cases" in qnm_fit
    assert "grid_sizes" in qnm_fit
    assert "window_ids" in qnm_fit

    # omega_complex should be [omega_R, omega_I]
    assert len(qnm_fit["omega_complex"]) == 2
    assert isinstance(qnm_fit["omega_R"], float)
    assert isinstance(qnm_fit["omega_I"], float)


def test_contract_verdict_structure(tmp_path: Path, repo_root: Path):
    """Verify contract_verdict.json has required fields."""
    run_id = "test_qnm_00_verdict"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)
    p = subprocess.run(
        [
            PY,
            "experiment/ringdown/exp_ringdown_qnm_00_open_bc.py",
            "--run", run_id,
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

    # Check required fields
    assert "schema_version" in verdict
    assert verdict["schema_version"] == "contract_verdict_v1"
    assert "verdict" in verdict
    assert verdict["verdict"] in ("PASS", "FAIL")
    assert "contracts" in verdict
    assert "C1_horizon_decay" in verdict["contracts"]
    assert "C2_stability" in verdict["contracts"]
    assert "assumptions" in verdict


def test_deterministic_outputs(tmp_path: Path, repo_root: Path):
    """Verify same inputs produce same outputs."""
    import shutil

    run_id = "test_qnm_00_deterministic"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _env_for(tmp_path)

    # First run
    p1 = subprocess.run(
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
    assert p1.returncode == 0, f"First run failed: {p1.stderr}"

    stage_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_QNM_00_open_bc"
    qnm_fit_1 = _rjson(stage_dir / "outputs" / "qnm_fit.json")
    omega_R_1 = qnm_fit_1["omega_R"]
    omega_I_1 = qnm_fit_1["omega_I"]

    # Clear and re-run
    shutil.rmtree(stage_dir)

    p2 = subprocess.run(
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
    assert p2.returncode == 0, f"Second run failed: {p2.stderr}"

    qnm_fit_2 = _rjson(stage_dir / "outputs" / "qnm_fit.json")
    omega_R_2 = qnm_fit_2["omega_R"]
    omega_I_2 = qnm_fit_2["omega_I"]

    assert omega_R_1 == omega_R_2, "omega_R not deterministic"
    assert omega_I_1 == omega_I_2, "omega_I not deterministic"
