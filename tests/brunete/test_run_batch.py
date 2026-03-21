from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import brunete.brunete_run_batch as run_batch


def test_run_batch_mode_220_writes_batch_outputs(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    _write_prepare_run(runs_root, "prep_ok", ["GW150914", "GW190412"])
    atlas = tmp_path / "atlas.json"
    atlas.write_text("{}\n", encoding="utf-8")
    losc_root = tmp_path / "data" / "losc"
    losc_root.mkdir(parents=True)

    def fake_run_single_event(**kwargs):
        out_root = Path(os.environ["BASURIN_RUNS_ROOT"])
        run_id = kwargs["run_id"]
        run_dir = out_root / run_id
        (run_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
        (run_dir / "RUN_VALID" / "verdict.json").write_text(
            json.dumps({"verdict": "PASS"}),
            encoding="utf-8",
        )
        compatible_dir = run_dir / "s4_geometry_filter" / "outputs"
        compatible_dir.mkdir(parents=True, exist_ok=True)
        (compatible_dir / "compatible_set.json").write_text(
            json.dumps({"n_compatible": 3}),
            encoding="utf-8",
        )
        return 0, run_id

    monkeypatch.setattr(run_batch.pipeline, "run_single_event", fake_run_single_event)

    rc = run_batch.main([
        "--prepare-run",
        "prep_ok",
        "--mode",
        "220",
        "--run-id",
        "batch_220",
        "--atlas-path",
        str(atlas),
        "--losc-root",
        str(losc_root),
    ])

    assert rc == 0

    stage_dir = runs_root / "batch_220" / "run_batch"
    verdict = json.loads((stage_dir / "RUN_VALID" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "PASS"

    results_json = json.loads((stage_dir / "outputs" / "results.json").read_text(encoding="utf-8"))
    assert results_json["summary"]["n_events"] == 2
    assert results_json["summary"]["n_pass"] == 2

    with (stage_dir / "outputs" / "results.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["mode"] == "220"
    assert rows[0]["status"] == "PASS"

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "run_batch"
    assert manifest["run_id"] == "batch_220"
    assert manifest["verdict"] == "PASS"

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "run_batch"
    assert summary["run_id"] == "batch_220"
    assert summary["verdict"] == "PASS"
    assert summary["parameters"]["prepare_run_id"] == "prep_ok"


def test_run_batch_mode_221_uses_multimode_engine_and_records_failures(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    _write_prepare_run(runs_root, "prep_ok", ["GW150914"])
    atlas = tmp_path / "atlas.json"
    atlas.write_text("{}\n", encoding="utf-8")
    losc_root = tmp_path / "data" / "losc"
    losc_root.mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_run_multimode_event(**kwargs):
        captured.update(kwargs)
        out_root = Path(os.environ["BASURIN_RUNS_ROOT"])
        run_id = kwargs["run_id"]
        run_dir = out_root / run_id
        (run_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
        (run_dir / "RUN_VALID" / "verdict.json").write_text(
            json.dumps({"verdict": "FAIL", "reason": "multimode gate failed"}),
            encoding="utf-8",
        )
        return 2, run_id

    monkeypatch.setattr(run_batch.pipeline, "run_multimode_event", fake_run_multimode_event)

    rc = run_batch.main([
        "--prepare-run",
        "prep_ok",
        "--mode",
        "221",
        "--run-id",
        "batch_221",
        "--atlas-path",
        str(atlas),
        "--losc-root",
        str(losc_root),
    ])

    assert rc == 2

    stage_dir = runs_root / "batch_221" / "run_batch"
    verdict = json.loads((stage_dir / "RUN_VALID" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "FAIL"

    with (stage_dir / "outputs" / "results.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["mode"] == "221"
    assert rows[0]["status"] == "FAIL"
    assert "multimode gate failed" in rows[0]["failure_reason"]
    assert captured["minimal_run"] is True

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "run_batch"
    assert summary["run_id"] == "batch_221"
    assert summary["verdict"] == "FAIL"
    assert summary["parameters"]["prepare_run_id"] == "prep_ok"


def _write_prepare_run(runs_root: Path, run_id: str, events: list[str]) -> None:
    stage_dir = runs_root / run_id / "prepare_events"
    (stage_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
    (stage_dir / "RUN_VALID" / "verdict.json").write_text(
        json.dumps({"verdict": "PASS"}),
        encoding="utf-8",
    )
    (stage_dir / "external_inputs").mkdir(parents=True, exist_ok=True)
    (stage_dir / "external_inputs" / "events.txt").write_text(
        "".join(f"{event_id}\n" for event_id in events),
        encoding="utf-8",
    )
