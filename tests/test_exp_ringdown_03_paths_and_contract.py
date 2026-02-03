from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_exp_ringdown_03_paths_and_contract(tmp_path: Path) -> None:
    run_id = "2037-01-01__unit_test__exp03"
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

    cases_path = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_01_injection_recovery"
        / "outputs"
        / "recovery_cases.jsonl"
    )
    cases = [
        {
            "case_id": "case_1",
            "status": "OK",
            "estimate": {"f_220_hat": 100.0, "tau_220_hat": 1.0, "Q_220_hat": 314.159},
        },
        {
            "case_id": "case_2",
            "status": "OK",
            "estimate": {"f_220_hat": 200.0, "tau_220_hat": 2.0, "Q_220_hat": 1256.636},
        },
        {
            "case_id": "case_3",
            "status": "OK",
            "estimate": {"f_220_hat": 300.0, "tau_220_hat": 3.0, "Q_220_hat": 2827.433},
        },
    ]
    _write_jsonl(cases_path, cases)

    observables_stage = Path("experiment/ringdown/stage_ringdown_observables_v1.py").resolve()
    res = subprocess.run(
        [sys.executable, str(observables_stage), "--run", run_id],
        cwd=Path.cwd(),
        env={**env, **dict()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    exp03_stage = Path("experiment/ringdown/exp_ringdown_03_observable_minimality.py").resolve()
    res = subprocess.run(
        [
            sys.executable,
            str(exp03_stage),
            "--run",
            run_id,
            "--min-gain",
            "0.05",
        ],
        cwd=Path.cwd(),
        env={**env, **dict()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    exp03_outputs = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_03__observable_minimality"
        / "outputs"
    )
    identifiability = exp03_outputs / "identifiability_report.json"
    ablations = exp03_outputs / "ablations.jsonl"
    contract = exp03_outputs / "contract_verdict.json"

    assert identifiability.exists()
    assert ablations.exists()
    assert contract.exists()

    payload = json.loads(contract.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert "R03_IDENTIFIABILITY_GAIN" in payload["contracts"]
    assert "R03_MINIMALITY" in payload["contracts"]

    for path in [identifiability, ablations, contract]:
        assert run_dir in path.resolve().parents
