"""
tests/test_exp_ringdown_09_real_atlas_sweep.py
----------------------------------------------
Tests for experiment/ringdown/exp_ringdown_09_real_atlas_sweep.py

Verifies:
- Script writes to correct canonical path
- Generates manifest.json, stage_summary.json, outputs/atlas_cases.jsonl, atlas_summary.json
- Uses tmp_path runs root only
- Monkeypatches subprocess.run to avoid real execution
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import experiment.ringdown.exp_ringdown_09_real_atlas_sweep as exp09


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_valid_pass(run_dir: Path) -> None:
    _write_json(
        run_dir / "RUN_VALID" / "outputs" / "run_valid.json",
        {"overall_verdict": "PASS"},
    )


def test_exp_ringdown_09_sweep_outputs(tmp_path: Path, monkeypatch) -> None:
    run_id = "2040-09-01__unit_test__exp09"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    _write_run_valid_pass(run_dir)

    exp08_report = (
        run_dir
        / "experiment"
        / "ringdown"
        / "EXP_RINGDOWN_08__real_v0_smoke"
        / "outputs"
        / "real_v0_smoke_report.json"
    )
    _write_json(
        exp08_report,
        {"overall_verdict": "PASS", "smoke_inference": {"n_smoke_ok": 2}},
    )

    suffix = exp09._compute_suffix(0.0, 0.25)
    inference_report = (
        run_dir
        / f"ringdown_real_inference_v0__{suffix}"
        / "outputs"
        / "inference_report.json"
    )
    _write_json(
        inference_report,
        {"fit": {"H1": {"f_peak_hz": 200.0}, "L1": {"f_peak_hz": 220.0}}},
    )

    grid = [
        {"dt_start_s": 0.0, "duration_s": 0.25, "band_hz": [150, 400]},
        {"dt_start_s": 0.1, "duration_s": 0.3, "band_hz": [140, 450]},
    ]
    grid_path = tmp_path / "grid.json"
    _write_json(grid_path, grid)

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    calls = []

    def fake_run(command, check=False, capture_output=False, text=False, cwd=None, env=None):
        calls.append(command)
        if command and command[0] == "git":
            return exp09.subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="")
        return exp09.subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(exp09.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exp_ringdown_09_real_atlas_sweep.py",
            "--run",
            run_id,
            "--grid-json",
            str(grid_path),
        ],
    )

    result = exp09.main()

    assert result == 0
    assert any("tools/basurin_run_real.py" in " ".join(cmd) for cmd in calls)

    stage_dir = (
        run_dir / "experiment" / "ringdown" / "EXP_RINGDOWN_09__real_atlas_sweep"
    )
    outputs_dir = stage_dir / "outputs"

    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (outputs_dir / "atlas_cases.jsonl").exists()
    assert (outputs_dir / "atlas_summary.json").exists()
    assert (outputs_dir / "failure_catalog.jsonl").exists()

    cases = [json.loads(line) for line in (outputs_dir / "atlas_cases.jsonl").read_text().splitlines()]
    summary = json.loads((outputs_dir / "atlas_summary.json").read_text(encoding="utf-8"))

    assert len(cases) == 2
    assert summary["n_cases"] == 2
    assert summary["n_pass"] == 2
