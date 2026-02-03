from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_strain_npz(path: Path, f0: float, tau: float) -> None:
    dt = 1.0 / 4096.0
    t = np.arange(4096, dtype=float) * dt
    strain = np.exp(-t / tau) * np.sin(2.0 * np.pi * f0 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strain=strain.astype(float), dt=dt)


def test_exp_ringdown_06_paths_and_contract(tmp_path: Path) -> None:
    run_id = "2038-01-01__unit_test__exp06"
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

    synth_outputs = run_dir / "ringdown_synth" / "outputs"
    cases_dir = synth_outputs / "cases"
    events = []
    recovery_rows = []
    for idx, seed in enumerate([11, 37]):
        case_id = f"case_{idx:03d}"
        f0 = 200.0 + 10.0 * idx
        tau = 0.50
        _write_strain_npz(cases_dir / case_id / "strain.npz", f0, tau)
        events.append(
            {
                "case_id": case_id,
                "seed": seed,
                "snr_target": 15.0 + idx,
                "strain_npz": f"cases/{case_id}/strain.npz",
            }
        )
        recovery_rows.append(
            {
                "case_id": case_id,
                "seed": seed,
                "snr_target": 15.0 + idx,
                "truth": {"f_220": f0, "tau_220": tau},
                "status": "OK",
            }
        )

    synth_outputs.mkdir(parents=True, exist_ok=True)
    list_path = synth_outputs / "synthetic_events_list.json"
    list_path.write_text(json.dumps(events, indent=2), encoding="utf-8")

    recovery_path = (
        run_dir
        / "experiment"
        / "ringdown_01_injection_recovery"
        / "outputs"
        / "recovery_cases.jsonl"
    )
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    with open(recovery_path, "w", encoding="utf-8") as f:
        for row in recovery_rows:
            f.write(json.dumps(row) + "\n")

    exp06_stage = Path("experiment/ringdown/exp_ringdown_06_psd_robustness.py").resolve()
    res = subprocess.run(
        [sys.executable, str(exp06_stage), "--run", run_id],
        cwd=Path.cwd(),
        env={**env, **dict()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    exp06_outputs = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_06__psd_robustness"
        / "outputs"
    )
    manifest = exp06_outputs.parent / "manifest.json"
    summary = exp06_outputs.parent / "stage_summary.json"
    sweep = exp06_outputs / "psd_sweep_metrics.json"
    per_case = exp06_outputs / "psd_cases.jsonl"
    contract = exp06_outputs / "contract_verdict.json"

    for path in [manifest, summary, sweep, per_case, contract]:
        assert path.exists()
        assert run_dir in path.resolve().parents

    payload = json.loads(contract.read_text(encoding="utf-8"))
    assert payload["overall_verdict"] in {"PASS", "FAIL"}
    assert {c["id"] for c in payload["contracts"]} == {
        "R06_PSD_ROBUSTNESS",
        "R06_DIAGNOSTICS_COMPLETE",
    }

    manifest_hashes_first = json.loads(manifest.read_text(encoding="utf-8"))["hashes"]

    res = subprocess.run(
        [sys.executable, str(exp06_stage), "--run", run_id],
        cwd=Path.cwd(),
        env={**env, **dict()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    manifest_hashes_second = json.loads(manifest.read_text(encoding="utf-8"))["hashes"]
    assert manifest_hashes_first == manifest_hashes_second
