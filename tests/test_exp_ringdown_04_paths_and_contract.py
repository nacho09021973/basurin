from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_strain_npz(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    strain = rng.normal(0.0, 1.0, size=4096).astype(float)
    dt = 1.0 / 4096.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strain=strain, dt=dt)


def test_exp_ringdown_04_paths_and_contract(tmp_path: Path) -> None:
    run_id = "2037-01-01__unit_test__exp04"
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
    for idx, seed in enumerate([11, 37]):
        case_id = f"case_{idx:03d}"
        _write_strain_npz(cases_dir / case_id / "strain.npz", seed)
        events.append({"case_id": case_id})

    synth_outputs.mkdir(parents=True, exist_ok=True)
    list_path = synth_outputs / "synthetic_events_list.json"
    list_path.write_text(json.dumps(events, indent=2), encoding="utf-8")

    synth_event_path = synth_outputs / "synthetic_event.json"
    synth_event_path.write_text(
        json.dumps({"schema_version": "ringdown_synth_event_v1", "path": "synthetic_event.json"}, indent=2),
        encoding="utf-8",
    )

    exp04_stage = Path("experiment/ringdown/exp_ringdown_04_psd_validity.py").resolve()
    res = subprocess.run(
        [sys.executable, str(exp04_stage), "--run", run_id],
        cwd=Path.cwd(),
        env={**env, **dict()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    exp04_outputs = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_04__psd_validity"
        / "outputs"
    )
    diagnostics = exp04_outputs / "psd_diagnostics.json"
    per_case = exp04_outputs / "per_case_psd.jsonl"
    contract = exp04_outputs / "contract_verdict.json"

    assert diagnostics.exists()
    assert per_case.exists()
    assert contract.exists()

    payload = json.loads(contract.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    for key in [
        "R04_PSD_WELL_CONDITIONED",
        "R04_WHITENING_FINITE",
        "R04_DIAGNOSTICS_COMPLETE",
        "R04_COVERAGE",
    ]:
        assert key in payload["contracts"]

    for path in [diagnostics, per_case, contract]:
        assert run_dir in path.resolve().parents
