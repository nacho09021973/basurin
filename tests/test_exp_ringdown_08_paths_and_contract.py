"""
tests/test_exp_ringdown_08_paths_and_contract.py
-------------------------------------------------
Tests for experiment/ringdown/exp_ringdown_08_real_v0_smoke.py

Verifies:
- Script writes to correct canonical path
- Generates manifest.json, stage_summary.json, outputs/contract_verdict.json
- When no real data exists: overall_verdict=FAIL, R08_REAL_IO_COMPAT FAIL with missing_real_v0_inputs
- R08_FAIL_CATEGORIZED always PASS (all failures are categorized)
- Tests are deterministic and do not require real data (use fixtures/monkeypatch)
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_valid_pass(run_dir: Path) -> None:
    _write_json(
        run_dir / "RUN_VALID" / "outputs" / "run_valid.json",
        {"overall_verdict": "PASS"},
    )


def _write_strain_npz(
    path: Path,
    f0: float = 220.0,
    tau: float = 0.02,
    seed: int = 42,
    n_samples: int = 2048,
    fs: float = 4096.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_samples, dtype=float) / fs
    signal = np.exp(-t / tau) * np.sin(2.0 * np.pi * f0 * t)
    strain = signal + 0.01 * (seed % 7)
    np.savez(path, strain=strain.astype(float), t=t.astype(float), fs=fs)


def test_exp_ringdown_08_missing_real_data_contract(tmp_path: Path) -> None:
    """
    Test that EXP08 fails with categorized contract when no real data exists.

    Expected:
    - overall_verdict: FAIL
    - R08_REAL_IO_COMPAT: FAIL with 'missing_real_v0_inputs'
    - R08_FAIL_CATEGORIZED: PASS (failure is properly categorized)
    """
    run_id = "2040-08-01__unit_test__exp08_missing"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        run_id,
        "--n-max-cases",
        "3",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}. stderr: {res.stderr}"

    stage_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke"
    )
    outputs_dir = stage_dir / "outputs"

    assert (stage_dir / "manifest.json").exists(), "manifest.json missing"
    assert (stage_dir / "stage_summary.json").exists(), "stage_summary.json missing"
    assert (outputs_dir / "contract_verdict.json").exists(), "contract_verdict.json missing"
    assert (outputs_dir / "real_v0_smoke_report.json").exists(), "real_v0_smoke_report.json missing"
    assert (outputs_dir / "failure_catalog.jsonl").exists(), "failure_catalog.jsonl missing"

    contract = json.loads((outputs_dir / "contract_verdict.json").read_text(encoding="utf-8"))

    assert contract["overall_verdict"] == "FAIL"
    assert "contracts" in contract

    contract_ids = {c["id"] for c in contract["contracts"]}
    assert contract_ids == {"R08_REAL_IO_COMPAT", "R08_PIPELINE_SMOKE", "R08_FAIL_CATEGORIZED"}

    io_compat = next(c for c in contract["contracts"] if c["id"] == "R08_REAL_IO_COMPAT")
    assert io_compat["verdict"] == "FAIL"
    assert "missing_real_v0_inputs" in io_compat["violations"]

    fail_cat = next(c for c in contract["contracts"] if c["id"] == "R08_FAIL_CATEGORIZED")
    assert fail_cat["verdict"] == "PASS"


def test_exp_ringdown_08_with_valid_real_data(tmp_path: Path) -> None:
    """Test that EXP08 passes when valid real data is available."""
    run_id = "2040-08-01__unit_test__exp08_valid"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    real_v0_dir = run_dir / "ringdown_real_v0" / "outputs"
    cases_dir = real_v0_dir / "cases"

    events = []
    for idx, event_id in enumerate(["GW150914_sim", "GW170817_sim", "GW190521_sim"]):
        case_dir = cases_dir / event_id
        strain_path = case_dir / "strain.npz"
        _write_strain_npz(strain_path, f0=200.0 + idx * 20, seed=idx)
        events.append({
            "event_id": event_id,
            "strain_npz": f"cases/{event_id}/strain.npz",
            "fs": 4096.0,
        })

    _write_json(real_v0_dir / "real_v0_events_list.json", events)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        run_id,
        "--n-max-cases",
        "3",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, f"Expected exit code 0, got {res.returncode}. stderr: {res.stderr}"

    stage_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke"
    )
    outputs_dir = stage_dir / "outputs"

    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (outputs_dir / "contract_verdict.json").exists()
    assert (outputs_dir / "real_v0_smoke_report.json").exists()
    assert (outputs_dir / "failure_catalog.jsonl").exists()

    contract = json.loads((outputs_dir / "contract_verdict.json").read_text(encoding="utf-8"))
    assert contract["overall_verdict"] == "PASS"

    io_compat = next(c for c in contract["contracts"] if c["id"] == "R08_REAL_IO_COMPAT")
    assert io_compat["verdict"] == "PASS"
    assert io_compat["metrics"]["n_valid"] == 3

    smoke = next(c for c in contract["contracts"] if c["id"] == "R08_PIPELINE_SMOKE")
    assert smoke["verdict"] == "PASS"
    assert smoke["metrics"]["n_smoke_ok"] == 3

    fail_cat = next(c for c in contract["contracts"] if c["id"] == "R08_FAIL_CATEGORIZED")
    assert fail_cat["verdict"] == "PASS"


def test_exp_ringdown_08_dry_run_mode(tmp_path: Path) -> None:
    """Test that --dry-run mode validates IO without running inference."""
    run_id = "2040-08-01__unit_test__exp08_dryrun"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    real_v0_dir = run_dir / "ringdown_real_v0" / "outputs"
    cases_dir = real_v0_dir / "cases"

    events = []
    for idx, event_id in enumerate(["event_001", "event_002"]):
        case_dir = cases_dir / event_id
        strain_path = case_dir / "strain.npz"
        _write_strain_npz(strain_path, f0=200.0 + idx * 20, seed=idx)
        events.append({
            "event_id": event_id,
            "strain_npz": f"cases/{event_id}/strain.npz",
        })

    _write_json(real_v0_dir / "real_v0_events_list.json", events)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        run_id,
        "--n-max-cases",
        "2",
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, f"Expected exit code 0, got {res.returncode}. stderr: {res.stderr}"

    stage_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke"
    )
    outputs_dir = stage_dir / "outputs"

    contract = json.loads((outputs_dir / "contract_verdict.json").read_text(encoding="utf-8"))
    assert contract["overall_verdict"] == "PASS"

    smoke = next(c for c in contract["contracts"] if c["id"] == "R08_PIPELINE_SMOKE")
    assert smoke["verdict"] == "PASS"
    assert smoke["violations"] == []
    assert smoke["metrics"]["dry_run"] is True
    assert smoke["metrics"]["n_smoke_ok"] == 0


def test_exp_ringdown_08_explicit_real_v0_json_path(tmp_path: Path) -> None:
    """Test using --real-v0-events-json to specify custom path."""
    run_id = "2040-08-01__unit_test__exp08_custom_path"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    custom_data_dir = tmp_path / "custom_real_data"
    cases_dir = custom_data_dir / "cases"

    events = []
    for idx, event_id in enumerate(["custom_event_001"]):
        case_dir = cases_dir / event_id
        strain_path = case_dir / "strain.npz"
        _write_strain_npz(strain_path, f0=250.0, seed=idx)
        events.append({
            "event_id": event_id,
            "strain_npz": f"cases/{event_id}/strain.npz",
        })

    custom_events_json = custom_data_dir / "my_events.json"
    _write_json(custom_events_json, events)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        run_id,
        "--real-v0-events-json",
        str(custom_events_json),
        "--n-max-cases",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    assert res.returncode == 0, f"Expected exit code 0, got {res.returncode}. stderr: {res.stderr}"

    stage_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke"
    )
    outputs_dir = stage_dir / "outputs"

    contract = json.loads((outputs_dir / "contract_verdict.json").read_text(encoding="utf-8"))
    assert contract["overall_verdict"] == "PASS"


def test_exp_ringdown_08_failure_catalog_categorization(tmp_path: Path) -> None:
    """Test that all failures in failure_catalog.jsonl have valid fail_reason_code."""
    run_id = "2040-08-01__unit_test__exp08_catalog"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    real_v0_dir = run_dir / "ringdown_real_v0" / "outputs"
    cases_dir = real_v0_dir / "cases"

    valid_dir = cases_dir / "valid_event"
    _write_strain_npz(valid_dir / "strain.npz", f0=220.0, seed=1)

    invalid_dir = cases_dir / "invalid_event"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    np.savez(invalid_dir / "strain.npz", bad_key=np.array([1, 2, 3]))

    events = [
        {"event_id": "valid_event", "strain_npz": "cases/valid_event/strain.npz"},
        {"event_id": "invalid_event", "strain_npz": "cases/invalid_event/strain.npz"},
    ]
    _write_json(real_v0_dir / "real_v0_events_list.json", events)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        run_id,
        "--n-max-cases",
        "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    stage_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke"
    )
    outputs_dir = stage_dir / "outputs"

    allowed_codes = {
        "exception",
        "invalid_input",
        "nan_inference",
        "nonconvergence",
        "numerical_instability",
        "timeout",
        "missing_real_v0_inputs",
        "",
    }

    with (outputs_dir / "failure_catalog.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") in {"FAIL", "ERROR"}:
                code = row.get("fail_reason_code", "")
                assert code in allowed_codes, f"Invalid fail_reason_code: {code}"


def test_exp_ringdown_08_deterministic_rerun(tmp_path: Path) -> None:
    """Test that running EXP08 twice produces consistent results."""
    run_id = "2040-08-01__unit_test__exp08_deterministic"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    real_v0_dir = run_dir / "ringdown_real_v0" / "outputs"
    cases_dir = real_v0_dir / "cases"

    events = []
    for idx in range(2):
        event_id = f"event_{idx:03d}"
        case_dir = cases_dir / event_id
        _write_strain_npz(case_dir / "strain.npz", f0=200.0 + idx * 10, seed=idx * 100)
        events.append({
            "event_id": event_id,
            "strain_npz": f"cases/{event_id}/strain.npz",
        })
    _write_json(real_v0_dir / "real_v0_events_list.json", events)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        run_id,
        "--n-max-cases",
        "2",
    ]

    res1 = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res1.returncode == 0

    stage_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke"
    )
    outputs_dir = stage_dir / "outputs"

    report1 = json.loads((outputs_dir / "real_v0_smoke_report.json").read_text(encoding="utf-8"))

    res2 = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res2.returncode == 0

    report2 = json.loads((outputs_dir / "real_v0_smoke_report.json").read_text(encoding="utf-8"))

    assert report1["overall_verdict"] == report2["overall_verdict"]
    assert report1["smoke_inference"]["n_smoke_ok"] == report2["smoke_inference"]["n_smoke_ok"]
    assert len(report1["smoke_inference"]["results"]) == len(report2["smoke_inference"]["results"])


def test_exp_ringdown_08_requires_run_valid_pass(tmp_path: Path) -> None:
    """EXP08 must abort if RUN_VALID is missing or fails."""
    runs_root = tmp_path / "runs"

    run_id_missing = "2040-08-01__unit_test__exp08_missing_run_valid"
    run_dir_missing = runs_root / run_id_missing
    run_dir_missing.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd_missing = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        run_id_missing,
        "--dry-run",
    ]
    res_missing = subprocess.run(
        cmd_missing, capture_output=True, text=True, check=False, env=env
    )
    assert res_missing.returncode == 2

    stage_dir_missing = (
        run_dir_missing / "experiment" / "ringdown" / "EXP_RINGDOWN_08__real_v0_smoke"
    )
    outputs_missing = stage_dir_missing / "outputs"
    assert (stage_dir_missing / "manifest.json").exists()
    assert (stage_dir_missing / "stage_summary.json").exists()
    assert (outputs_missing / "contract_verdict.json").exists()
    assert (outputs_missing / "real_v0_smoke_report.json").exists()
    assert (outputs_missing / "failure_catalog.jsonl").exists()

    run_id_fail = "2040-08-01__unit_test__exp08_run_valid_fail"
    run_dir_fail = runs_root / run_id_fail
    _write_json(
        run_dir_fail / "RUN_VALID" / "outputs" / "run_valid.json",
        {"overall_verdict": "FAIL"},
    )

    cmd_fail = [
        "python",
        "experiment/ringdown/exp_ringdown_08_real_v0_smoke.py",
        "--run",
        run_id_fail,
        "--dry-run",
    ]
    res_fail = subprocess.run(
        cmd_fail, capture_output=True, text=True, check=False, env=env
    )
    assert res_fail.returncode == 2
