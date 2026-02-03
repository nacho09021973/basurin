from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_strain_npz(path: Path, f0: float, tau: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 2048
    fs = 4096.0
    t = np.arange(n, dtype=float) / fs
    signal = np.exp(-t / tau) * np.sin(2.0 * np.pi * f0 * t)
    strain = signal + 0.01 * (seed % 7)
    np.savez(path, strain=strain.astype(float), t=t.astype(float))


def test_exp_ringdown_07_paths_and_contract(tmp_path: Path) -> None:
    run_id = "2040-07-01__unit_test__exp07"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_json(run_dir / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    cases = []
    for idx, seed in enumerate([101, 202]):
        case_id = f"case_{idx:03d}"
        snr = 12.0 + idx
        strain_path = run_dir / "ringdown_synth" / "outputs" / "cases" / case_id / "strain.npz"
        _write_strain_npz(strain_path, f0=220.0 + 5.0 * idx, tau=0.02, seed=seed)
        cases.append(
            {
                "case_id": case_id,
                "seed": seed,
                "snr_target": snr,
                "strain_npz": f"cases/{case_id}/strain.npz",
            }
        )

    _write_json(
        run_dir / "ringdown_synth" / "outputs" / "synthetic_events_list.json",
        cases,
    )

    recovery_cases = run_dir / "experiment" / "ringdown_01_injection_recovery" / "outputs" / "recovery_cases.jsonl"
    recovery_cases.parent.mkdir(parents=True, exist_ok=True)
    with recovery_cases.open("w", encoding="utf-8") as f:
        for idx, seed in enumerate([101, 202]):
            row = {
                "case_id": f"case_{idx:03d}",
                "seed": seed,
                "snr_target": 12.0 + idx,
                "truth": {"f_220": 220.0 + 5.0 * idx, "tau_220": 0.02},
            }
            f.write(json.dumps(row) + "\n")

    env = {**os.environ, "BASURIN_RUNS_ROOT": str(runs_root)}
    cmd = [
        "python",
        "experiment/ringdown/exp_ringdown_07_nonstationary_stress.py",
        "--run",
        run_id,
        "--n-max-cases",
        "2",
        "--sweep",
        "v1",
        "--max-fail-rate",
        "1.0",
        "--max-bias-rel",
        "10.0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res.returncode == 0

    stage_dir = run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_07__nonstationary_stress"
    outputs_dir = stage_dir / "outputs"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (outputs_dir / "nonstationary_report.json").exists()
    assert (outputs_dir / "failure_catalog.jsonl").exists()
    assert (outputs_dir / "contract_verdict.json").exists()

    contract = json.loads((outputs_dir / "contract_verdict.json").read_text(encoding="utf-8"))
    assert "contracts" in contract
    assert any(c["id"] == "R07_ROBUST_UNDER_NONSTATIONARY" for c in contract["contracts"])
    assert any(c["id"] == "R07_FAIL_CATEGORIZED" for c in contract["contracts"])

    with (outputs_dir / "failure_catalog.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") in {"FAIL", "ERROR"}:
                assert row.get("fail_reason_code")

    report_first = json.loads((outputs_dir / "nonstationary_report.json").read_text(encoding="utf-8"))
    res2 = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res2.returncode == 0
    report_second = json.loads((outputs_dir / "nonstationary_report.json").read_text(encoding="utf-8"))
    assert report_first["n_total"] == report_second["n_total"]
    assert report_first["n_fail"] == report_second["n_fail"]
    assert len(report_first["variants"]) == len(report_second["variants"])
