from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvp import experiment_oracle_t0_ringdown as oracle_exp


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_run_fails_with_expected_path_and_sweep_command_when_seed_dir_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "BASE_RUN"
    run_root = runs_root / run_id
    _write_json(run_root / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    with pytest.raises(RuntimeError) as exc:
        oracle_exp.run(["--run-id", run_id, "--runs-root", str(runs_root)])

    msg = str(exc.value)
    expected_seed = (run_root / "experiment" / "t0_sweep_full_seed101").resolve()
    assert f"expected path: {expected_seed}" in msg
    assert (
        "generate sweep with: python mvp/experiment_t0_sweep_full.py "
        "--run-id BASE_RUN --phase run --atlas-path <ATLAS_PATH> --t0-grid-ms 0,2,4,6,8 --seed 101"
    ) in msg


def test_run_fails_with_expected_path_and_sweep_command_when_sweep_json_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "BASE_RUN"
    run_root = runs_root / run_id
    _write_json(run_root / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})
    seed_dir = run_root / "experiment" / "t0_sweep_full_seed202"
    seed_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError) as exc:
        oracle_exp.run(["--run-id", run_id, "--runs-root", str(runs_root), "--seed-dir", str(seed_dir)])

    msg = str(exc.value)
    expected_json = seed_dir / "outputs" / "t0_sweep_full_results.json"
    assert f"expected path: {expected_json}" in msg
    assert (
        "generate sweep with: python mvp/experiment_t0_sweep_full.py "
        "--run-id BASE_RUN --phase run --atlas-path <ATLAS_PATH> --t0-grid-ms 0,2,4,6,8 --seed 202"
    ) in msg
