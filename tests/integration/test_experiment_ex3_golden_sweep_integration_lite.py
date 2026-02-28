from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from mvp import experiment_ex3_golden_sweep as ex3


def _write_run_valid(runs_root: Path, run_id: str) -> None:
    rv = runs_root / run_id / "RUN_VALID"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "verdict.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")


def test_integration_lite_ex3_golden_writes_only_under_runs_root(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    parent_run = "agg_ex3_it"
    ev1 = "GW150914"
    ev2 = "GW151226"
    run1 = "run_ev1"
    run2 = "run_ev2"

    _write_run_valid(runs_root, parent_run)
    _write_run_valid(runs_root, run1)
    _write_run_valid(runs_root, run2)

    s5_out = runs_root / parent_run / "s5_aggregate" / "outputs"
    s5_out.mkdir(parents=True, exist_ok=True)
    (s5_out / "aggregate.json").write_text(
        json.dumps({"events": [{"event_id": ev1, "run_id": run1}, {"event_id": ev2, "run_id": run2}]}),
        encoding="utf-8",
    )

    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    def _fake_run(cmd: list[str], env: dict[str, str], capture_output: bool, text: bool, check: bool):
        assert capture_output is True
        assert text is True
        assert check is False
        assert env["BASURIN_RUNS_ROOT"] == str(runs_root)
        event_run = cmd[cmd.index("--run") + 1]
        seed = int(cmd[cmd.index("--seed") + 1])
        out = runs_root / event_run / "experiment" / f"t0_sweep_full_seed{seed}" / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        (out / "t0_sweep_full_results.json").write_text(
            json.dumps({"results": [{"t0_ms": 0, "value": 1.0}], "meta": {"ok": True}}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(ex3.subprocess, "run", _fake_run)

    args = argparse.Namespace(
        run_id=parent_run,
        golden_events=f"{ev1},{ev2}",
        atlas_path=None,
        t0_grid_ms=None,
        t0_start_ms=0,
        t0_stop_ms=10,
        t0_step_ms=5,
        n_bootstrap=5,
        seed=12345,
        detector="auto",
        stage_timeout_s=10,
        resume_missing=False,
        max_retries_per_pair=2,
        resume_batch_size=10,
    )

    rc = ex3.run_experiment(args)
    assert rc == 0

    stage_dir = runs_root / parent_run / "experiment" / "ex3_t0_golden"
    outputs_dir = stage_dir / "outputs"
    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "stage_summary.json").exists()
    assert (outputs_dir / "t0_sweep_golden_results.json").exists()
    assert (outputs_dir / "golden_diagnostics.json").exists()
    assert (outputs_dir / "per_event" / f"{ev1}_t0_sweep.json").exists()
    assert (outputs_dir / "per_event" / f"{ev2}_t0_sweep.json").exists()

    payload = json.loads((outputs_dir / "t0_sweep_golden_results.json").read_text(encoding="utf-8"))
    assert payload["golden_events"] == [ev1, ev2]
    assert len(payload["results"]) == 2
    assert all("science_diagnostics" in row for row in payload["results"])

    diagnostics = json.loads((outputs_dir / "golden_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["schema_version"] == "ex3_diagnostics_v1"
    assert diagnostics["parent_run_id"] == parent_run
    assert diagnostics["n_events_attempted"] == 2
    assert diagnostics["n_events_completed"] == 2
    assert diagnostics["n_events_failed"] == 0
    assert diagnostics["failed_events"] == []
    assert diagnostics["completed_events"] == [ev1, ev2]
    assert diagnostics["t0_sweep_exit_codes"] == {ev1: 0, ev2: 0}
    assert set(diagnostics["timing_s"].keys()) == {ev1, ev2}
    assert all(isinstance(v, float) and v >= 0.0 for v in diagnostics["timing_s"].values())
    assert isinstance(diagnostics["created"], str)

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    summary_paths = [item["path"] for item in summary["outputs"]]
    assert "outputs/golden_diagnostics.json" in summary_paths

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["golden_diagnostics"] == "outputs/golden_diagnostics.json"
    assert isinstance(manifest["hashes"]["golden_diagnostics"], str)

    accidental_repo_path = Path.cwd() / "runs" / parent_run / "experiment" / "ex3_t0_golden"
    assert not accidental_repo_path.exists()
