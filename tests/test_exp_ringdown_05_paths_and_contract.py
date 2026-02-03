from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_ringdown_npz(path: Path, f0: float, tau: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 4096.0
    t = np.arange(4096) * dt
    signal = np.exp(-t / tau) * np.sin(2.0 * np.pi * f0 * t)
    noise = rng.normal(0.0, 0.01, size=signal.shape)
    strain = (signal + noise).astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strain=strain, dt=dt)


def test_exp_ringdown_05_paths_and_contract(tmp_path: Path) -> None:
    run_id = "2037-01-02__unit_test__exp05"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)

    env = {"BASURIN_RUNS_ROOT": str(runs_root)}

    stage_run_valid = Path("experiment/run_valid/stage_run_valid.py").resolve()
    res = subprocess.run(
        [sys.executable, str(stage_run_valid), "--run", run_id],
        cwd=Path.cwd(),
        env={**env, **dict()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    cases_dir = run_dir / "ringdown_synth" / "outputs" / "cases"
    cases = []
    synth_index = []
    for idx, seed in enumerate([7, 13, 29]):
        case_id = f"case_{idx:03d}"
        _write_ringdown_npz(cases_dir / case_id / "strain.npz", f0=240.0 + idx * 5, tau=0.02, seed=seed)
        snr = 12.0
        cases.append(
            {
                "case_id": case_id,
                "truth": {"f_220": 240.0 + idx * 5, "tau_220": 0.02},
                "seed": seed,
                "snr": snr,
            }
        )
        synth_index.append(
            {
                "seed": seed,
                "snr_target": snr,
                "strain_npz": f"cases/{case_id}/strain.npz",
            }
        )

    synth_index_path = run_dir / "ringdown_synth" / "outputs" / "synthetic_events_list.json"
    synth_index_path.parent.mkdir(parents=True, exist_ok=True)
    synth_index_path.write_text(json.dumps(synth_index, indent=2), encoding="utf-8")

    exp01_outputs = run_dir / "experiment" / "ringdown_01_injection_recovery" / "outputs"
    exp01_outputs.mkdir(parents=True, exist_ok=True)
    recovery_cases_path = exp01_outputs / "recovery_cases.jsonl"
    with open(recovery_cases_path, "w", encoding="utf-8") as f:
        for row in cases:
            f.write(json.dumps(row) + "\n")

    exp05_stage = Path("experiment/ringdown/exp_ringdown_05_prior_hyperparam_sweep.py").resolve()
    res = subprocess.run(
        [
            sys.executable,
            str(exp05_stage),
            "--run",
            run_id,
            "--out-root",
            str(runs_root),
            "--min-cases",
            "2",
        ],
        cwd=Path.cwd(),
        env={**env, **dict()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    exp05_outputs = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_05__prior_hyperparam_sweep"
        / "outputs"
    )
    prior_sweep = exp05_outputs / "prior_sweep.json"
    per_case = exp05_outputs / "per_case.jsonl"
    contract = exp05_outputs / "contract_verdict.json"

    assert prior_sweep.exists()
    assert per_case.exists()
    assert contract.exists()

    payload = json.loads(contract.read_text(encoding="utf-8"))
    assert "R05_PRIOR_SENSITIVITY_BOUNDED" in payload["contracts"]
    assert "R05_FAILURE_MODE_CAP" in payload["contracts"]
    assert "synthetic_events_list" in payload["inputs"]

    for path in [prior_sweep, per_case, contract]:
        assert run_dir in path.resolve().parents
