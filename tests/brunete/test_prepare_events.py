from __future__ import annotations

import json
from pathlib import Path

import brunete.brunete_prepare_events as prepare_events


def test_prepare_events_from_events_file_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    events_file = tmp_path / "events.txt"
    events_file.write_text(
        "GW190412\n\n# comment\nGW150914\nGW190412\n",
        encoding="utf-8",
    )

    rc = prepare_events.main([
        "--run-id",
        "prep_file",
        "--events-file",
        str(events_file),
    ])

    assert rc == 0

    stage_dir = runs_root / "prep_file" / "prepare_events"
    verdict = json.loads((stage_dir / "RUN_VALID" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "PASS"

    normalized_events = (stage_dir / "external_inputs" / "events.txt").read_text(encoding="utf-8")
    assert normalized_events == "GW150914\nGW190412\n"

    catalog = json.loads((stage_dir / "outputs" / "events_catalog.json").read_text(encoding="utf-8"))
    assert catalog["event_ids"] == ["GW150914", "GW190412"]
    assert catalog["source_kind"] == "events_file"

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "prepare_events"
    assert manifest["run_id"] == "prep_file"
    assert manifest["verdict"] == "PASS"

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "prepare_events"
    assert summary["run_id"] == "prep_file"
    assert summary["verdict"] == "PASS"
    assert summary["results"]["n_events"] == 2


def test_prepare_events_from_losc_root_discovers_only_event_dirs_with_hdf5(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    losc_root = tmp_path / "data" / "losc"
    (losc_root / "GW190412").mkdir(parents=True)
    (losc_root / "GW150914").mkdir(parents=True)
    (losc_root / "EMPTY_EVENT").mkdir(parents=True)
    (losc_root / "GW190412" / "H1.hdf5").write_text("stub", encoding="utf-8")
    (losc_root / "GW150914" / "L1.h5").write_text("stub", encoding="utf-8")

    rc = prepare_events.main([
        "--run-id",
        "prep_losc",
        "--losc-root",
        str(losc_root),
    ])

    assert rc == 0

    stage_dir = runs_root / "prep_losc" / "prepare_events"
    catalog = json.loads((stage_dir / "outputs" / "events_catalog.json").read_text(encoding="utf-8"))
    assert catalog["source_kind"] == "losc_root"
    assert catalog["event_ids"] == ["GW150914", "GW190412"]

    normalized_events = (stage_dir / "external_inputs" / "events.txt").read_text(encoding="utf-8")
    assert normalized_events == "GW150914\nGW190412\n"


def test_prepare_events_marks_run_valid_fail_when_no_events_are_discovered(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    losc_root = tmp_path / "data" / "losc"
    losc_root.mkdir(parents=True)
    (losc_root / "EMPTY_EVENT").mkdir()

    rc = prepare_events.main([
        "--run-id",
        "prep_fail",
        "--losc-root",
        str(losc_root),
    ])

    assert rc == 2

    stage_dir = runs_root / "prep_fail" / "prepare_events"
    verdict = json.loads((stage_dir / "RUN_VALID" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "FAIL"

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "prepare_events"
    assert summary["run_id"] == "prep_fail"
    assert summary["verdict"] == "FAIL"
    assert "no events discovered" in summary["error"]
