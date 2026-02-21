from __future__ import annotations

import sys
from pathlib import Path

import tools.run_experiment_batch as batch


def test_main_invoca_t0_sweep_full_como_modulo(tmp_path, monkeypatch) -> None:
    atlas_path = tmp_path / "atlas.json"
    atlas_path.write_text("{}", encoding="utf-8")

    recorded_cmds: list[list[str]] = []

    def _fake_run_cmd(cmd: list[str], env: dict[str, str]) -> None:
        recorded_cmds.append(cmd)

    monkeypatch.setattr(batch, "_run_cmd", _fake_run_cmd)
    monkeypatch.setattr(batch, "_select_event_h5", lambda event_id, detector, losc_root: tmp_path / f"{event_id}_{detector}.h5")
    monkeypatch.setattr(batch, "_new_subrun_trace", lambda subrun_id, seed, t0_ms, stages: {"stages": []})
    monkeypatch.setattr(batch, "_write_subrun_trace", lambda trace_path, trace: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_batch.py",
            "--event-id",
            "GW150914",
            "--atlas-path",
            str(atlas_path),
            "--runs-root",
            str(tmp_path / "runs_root"),
            "--batch-run-id",
            "batch_test",
        ],
    )

    rc = batch.main()

    assert rc == 0
    t0_cmd = next(cmd for cmd in recorded_cmds if "mvp.experiment_t0_sweep_full" in cmd)
    assert t0_cmd[0] == sys.executable
    assert "-m" in t0_cmd
    assert "mvp.experiment_t0_sweep_full" in t0_cmd
    assert "mvp/experiment_t0_sweep_full.py" not in t0_cmd
