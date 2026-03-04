from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mvp.experiment_offline_batch as offline_batch
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


def test_offline_batch_invoca_s2_offline_y_escribe_csv_en_runs_root(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    events_file = tmp_path / "events.txt"
    events_file.write_text("GW150914\n", encoding="utf-8")

    t0_catalog = tmp_path / "t0_catalog.json"
    t0_catalog.write_text("{}\n", encoding="utf-8")

    atlas = tmp_path / "atlas.json"
    atlas.write_text("{}\n", encoding="utf-8")

    recorded_cmds: list[list[str]] = []

    def _fake_run_cmd(cmd: list[str], *, env: dict[str, str]) -> None:
        recorded_cmds.append(cmd)

    run_id = "mvp_GW150914_real_offline_20260304T000000Z"
    monkeypatch.setattr(offline_batch, "_run_cmd", _fake_run_cmd)
    monkeypatch.setattr(offline_batch, "_event_run_id", lambda event_id: run_id)
    monkeypatch.setattr(offline_batch, "_read_len_compatible", lambda out_root, run_id: 1)

    rc = offline_batch.main(
        [
            "--batch-run-id",
            "batch_test",
            "--events-file",
            str(events_file),
            "--window-catalog",
            str(t0_catalog),
            "--atlas-path",
            str(atlas),
            "--max-events",
            "1",
        ]
    )

    assert rc == 0

    s2_cmd = next(cmd for cmd in recorded_cmds if "mvp.s2_ringdown_window" in cmd)
    assert "--offline" in s2_cmd
    assert "--window-catalog" in s2_cmd
    assert str(t0_catalog) in s2_cmd

    results_csv = runs_root / "batch_test" / "experiment" / "offline_batch" / "results.csv"
    assert results_csv.exists()

    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "GW150914"
    assert rows[0]["run_id"] == run_id

    summary_path = runs_root / "batch_test" / "experiment" / "offline_batch" / "stage_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["results"]["n_events"] == 1


def test_offline_batch_prioriza_window_catalog_sobre_alias_t0_catalog(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    events_file = tmp_path / "events.txt"
    events_file.write_text("GW150914\n", encoding="utf-8")

    window_catalog = tmp_path / "window_catalog.json"
    window_catalog.write_text("{}\n", encoding="utf-8")

    t0_catalog = tmp_path / "t0_catalog.json"
    t0_catalog.write_text("{}\n", encoding="utf-8")

    atlas = tmp_path / "atlas.json"
    atlas.write_text("{}\n", encoding="utf-8")

    recorded_cmds: list[list[str]] = []

    def _fake_run_cmd(cmd: list[str], *, env: dict[str, str]) -> None:
        recorded_cmds.append(cmd)

    monkeypatch.setattr(offline_batch, "_run_cmd", _fake_run_cmd)
    monkeypatch.setattr(offline_batch, "_event_run_id", lambda event_id: "mvp_GW150914_real_offline_20260304T000000Z")
    monkeypatch.setattr(offline_batch, "_read_len_compatible", lambda out_root, run_id: 1)

    rc = offline_batch.main(
        [
            "--batch-run-id",
            "batch_test",
            "--events-file",
            str(events_file),
            "--window-catalog",
            str(window_catalog),
            "--t0-catalog",
            str(t0_catalog),
            "--atlas-path",
            str(atlas),
            "--max-events",
            "1",
        ]
    )

    assert rc == 0

    s2_cmd = next(cmd for cmd in recorded_cmds if "mvp.s2_ringdown_window" in cmd)
    wc_idx = s2_cmd.index("--window-catalog")
    assert s2_cmd[wc_idx + 1] == str(window_catalog)
    assert "--offline" in s2_cmd


def test_offline_batch_pasa_mode_filter_a_s4(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    events_file = tmp_path / "events.txt"
    events_file.write_text("GW150914\n", encoding="utf-8")

    t0_catalog = tmp_path / "t0_catalog.json"
    t0_catalog.write_text("{}\n", encoding="utf-8")

    atlas = tmp_path / "atlas.json"
    atlas.write_text("{}\n", encoding="utf-8")

    recorded_cmds: list[list[str]] = []

    def _fake_run_cmd(cmd: list[str], *, env: dict[str, str]) -> None:
        recorded_cmds.append(cmd)

    monkeypatch.setattr(offline_batch, "_run_cmd", _fake_run_cmd)
    monkeypatch.setattr(offline_batch, "_event_run_id", lambda event_id: "mvp_GW150914_real_offline_20260304T000000Z")
    monkeypatch.setattr(offline_batch, "_read_len_compatible", lambda out_root, run_id: 1)

    rc = offline_batch.main(
        [
            "--batch-run-id",
            "batch_test",
            "--events-file",
            str(events_file),
            "--window-catalog",
            str(t0_catalog),
            "--atlas-path",
            str(atlas),
            "--mode-filter",
            "(2,2,0)",
            "--max-events",
            "1",
        ]
    )

    assert rc == 0

    s4_cmd = next(cmd for cmd in recorded_cmds if "mvp.s4_geometry_filter" in cmd)
    mf_idx = s4_cmd.index("--mode-filter")
    assert s4_cmd[mf_idx + 1] == "(2,2,0)"

    results_csv = runs_root / "batch_test" / "experiment" / "offline_batch" / "results.csv"
    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["mode_filter"] == "(2,2,0)"


def test_offline_batch_usa_n_compatible_de_s4_en_results_csv(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    events_file = tmp_path / "events.txt"
    events_file.write_text("GW170817\n", encoding="utf-8")

    t0_catalog = tmp_path / "t0_catalog.json"
    t0_catalog.write_text("{}\n", encoding="utf-8")

    atlas = tmp_path / "atlas.json"
    atlas.write_text("{}\n", encoding="utf-8")

    run_id = "mvp_GW170817_real_offline_20260304T000000Z"
    compatible_path = runs_root / run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    compatible_path.parent.mkdir(parents=True, exist_ok=True)
    compatible_path.write_text(
        json.dumps({"n_compatible": 156, "compatible_geometries": list(range(156))}),
        encoding="utf-8",
    )

    monkeypatch.setattr(offline_batch, "_run_cmd", lambda cmd, *, env: None)
    monkeypatch.setattr(offline_batch, "_event_run_id", lambda event_id: run_id)

    rc = offline_batch.main(
        [
            "--batch-run-id",
            "batch_test",
            "--events-file",
            str(events_file),
            "--window-catalog",
            str(t0_catalog),
            "--atlas-path",
            str(atlas),
            "--epsilon-default",
            "2500",
            "--max-events",
            "1",
        ]
    )

    assert rc == 0
    results_csv = runs_root / "batch_test" / "experiment" / "offline_batch" / "results.csv"
    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["event_id"] == "GW170817"
    assert rows[0]["status"] == "PASS"
    assert rows[0]["len_compatible"] == "156"
    assert rows[0]["epsilon_used"] == "2500.0"


def test_offline_batch_reporta_fail_si_compatible_set_es_invalido(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    events_file = tmp_path / "events.txt"
    events_file.write_text("GW170817\n", encoding="utf-8")

    t0_catalog = tmp_path / "t0_catalog.json"
    t0_catalog.write_text("{}\n", encoding="utf-8")

    atlas = tmp_path / "atlas.json"
    atlas.write_text("{}\n", encoding="utf-8")

    run_id = "mvp_GW170817_real_offline_20260304T000000Z"
    compatible_path = runs_root / run_id / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    compatible_path.parent.mkdir(parents=True, exist_ok=True)
    compatible_path.write_text(json.dumps({"compatible": []}), encoding="utf-8")

    monkeypatch.setattr(offline_batch, "_run_cmd", lambda cmd, *, env: None)
    monkeypatch.setattr(offline_batch, "_event_run_id", lambda event_id: run_id)

    rc = offline_batch.main(
        [
            "--batch-run-id",
            "batch_test",
            "--events-file",
            str(events_file),
            "--window-catalog",
            str(t0_catalog),
            "--atlas-path",
            str(atlas),
            "--max-events",
            "1",
        ]
    )

    assert rc == 0
    results_csv = runs_root / "batch_test" / "experiment" / "offline_batch" / "results.csv"
    with results_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["status"] == "FAIL"
    assert rows[0]["error_stage"] == "s4_geometry_filter"
    assert "CompatibleSetSchemaError" in rows[0]["error_message_short"]
