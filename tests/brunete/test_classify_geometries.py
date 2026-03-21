from __future__ import annotations

import csv
import json
from pathlib import Path

import brunete.brunete_classify_geometries as classify_geometries


def test_classify_geometries_writes_joint_summary_for_two_valid_batches(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    _write_batch_run(
        runs_root,
        run_id="batch_220_ok",
        mode="220",
        rows=[
            {
                "event_id": "GW150914",
                "mode": "220",
                "event_run_id": "brunete_GW150914_m220",
                "status": "PASS",
                "n_compatible": 3,
            },
            {
                "event_id": "GW190412",
                "mode": "220",
                "event_run_id": "brunete_GW190412_m220",
                "status": "PASS",
                "n_compatible": 0,
            },
            {
                "event_id": "GW200129_065458",
                "mode": "220",
                "event_run_id": "brunete_GW200129_065458_m220",
                "status": "FAIL",
                "n_compatible": None,
            },
        ],
    )
    _write_batch_run(
        runs_root,
        run_id="batch_221_ok",
        mode="221",
        rows=[
            {
                "event_id": "GW150914",
                "mode": "221",
                "event_run_id": "brunete_GW150914_m221",
                "status": "PASS",
                "n_compatible": 2,
            },
            {
                "event_id": "GW190412",
                "mode": "221",
                "event_run_id": "brunete_GW190412_m221",
                "status": "PASS",
                "n_compatible": 0,
            },
            {
                "event_id": "GW190521_074359",
                "mode": "221",
                "event_run_id": "brunete_GW190521_074359_m221",
                "status": "PASS",
                "n_compatible": 4,
            },
        ],
    )

    rc = classify_geometries.main([
        "--batch-220",
        "batch_220_ok",
        "--batch-221",
        "batch_221_ok",
        "--run-id",
        "classify_ok",
    ])

    assert rc == 0

    stage_dir = runs_root / "classify_ok" / "classify_geometries"
    verdict = json.loads((stage_dir / "RUN_VALID" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "PASS"

    payload = json.loads((stage_dir / "outputs" / "geometry_summary.json").read_text(encoding="utf-8"))
    assert payload["summary"]["n_events_union"] == 4
    assert payload["summary"]["n_events_both"] == 2
    assert payload["summary"]["n_joint_support"] == 1
    assert payload["summary"]["classification_counts"]["common_nonempty_both"] == 1
    assert payload["summary"]["classification_counts"]["common_empty_both"] == 1
    assert payload["summary"]["classification_counts"]["only_batch_221"] == 1
    assert payload["summary"]["classification_counts"]["only_batch_220"] == 1

    rows_by_event = {row["event_id"]: row for row in payload["rows"]}
    assert rows_by_event["GW150914"]["classification"] == "common_nonempty_both"
    assert rows_by_event["GW190412"]["classification"] == "common_empty_both"
    assert rows_by_event["GW190521_074359"]["classification"] == "only_batch_221"
    assert rows_by_event["GW200129_065458"]["classification"] == "only_batch_220"

    with (stage_dir / "outputs" / "geometry_summary.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert any(row["classification"] == "common_nonempty_both" for row in rows)

    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "classify_geometries"
    assert manifest["run_id"] == "classify_ok"
    assert manifest["verdict"] == "PASS"

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "classify_geometries"
    assert summary["run_id"] == "classify_ok"
    assert summary["verdict"] == "PASS"
    assert summary["parameters"]["batch_220_run_id"] == "batch_220_ok"
    assert summary["parameters"]["batch_221_run_id"] == "batch_221_ok"


def test_classify_geometries_fails_when_one_batch_is_not_pass(tmp_path, monkeypatch) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    _write_batch_run(
        runs_root,
        run_id="batch_220_fail",
        mode="220",
        rows=[],
        verdict="FAIL",
    )
    _write_batch_run(
        runs_root,
        run_id="batch_221_ok",
        mode="221",
        rows=[],
        verdict="PASS",
    )

    rc = classify_geometries.main([
        "--batch-220",
        "batch_220_fail",
        "--batch-221",
        "batch_221_ok",
        "--run-id",
        "classify_fail",
    ])

    assert rc == 2

    stage_dir = runs_root / "classify_fail" / "classify_geometries"
    verdict = json.loads((stage_dir / "RUN_VALID" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "FAIL"

    summary = json.loads((stage_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "classify_geometries"
    assert summary["run_id"] == "classify_fail"
    assert summary["verdict"] == "FAIL"
    assert "not PASS" in summary["error"]


def test_classify_geometries_uses_mode_221_support_region_when_n_compatible_is_saturated(
    tmp_path,
    monkeypatch,
) -> None:
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("BASURIN_RUNS_ROOT", str(runs_root))

    _write_batch_run(
        runs_root,
        run_id="batch_220_sat",
        mode="220",
        rows=[
            {
                "event_id": "GW150914",
                "mode": "220",
                "event_run_id": "brunete_GW150914_m220",
                "status": "PASS",
                "n_compatible": 2637,
            },
            {
                "event_id": "GW170814",
                "mode": "220",
                "event_run_id": "brunete_GW170814_m220",
                "status": "PASS",
                "n_compatible": 2637,
            },
            {
                "event_id": "GW190910_112807",
                "mode": "220",
                "event_run_id": "brunete_GW190910_112807_m220",
                "status": "PASS",
                "n_compatible": 2637,
            },
        ],
    )
    _write_batch_run(
        runs_root,
        run_id="batch_221_sat",
        mode="221",
        rows=[
            {
                "event_id": "GW150914",
                "mode": "221",
                "event_run_id": "brunete_GW150914_m221",
                "status": "PASS",
                "n_compatible": 2637,
                "support_region_status": "SUPPORT_REGION_AVAILABLE",
                "support_region_n_final": 5,
            },
            {
                "event_id": "GW170814",
                "mode": "221",
                "event_run_id": "brunete_GW170814_m221",
                "status": "PASS",
                "n_compatible": 2637,
                "support_region_status": "SUPPORT_REGION_AVAILABLE",
                "support_region_n_final": 1,
            },
            {
                "event_id": "GW190910_112807",
                "mode": "221",
                "event_run_id": "brunete_GW190910_112807_m221",
                "status": "PASS",
                "n_compatible": 2637,
                "support_region_status": "NO_COMMON_REGION",
                "support_region_n_final": 0,
            },
        ],
    )

    rc = classify_geometries.main([
        "--batch-220",
        "batch_220_sat",
        "--batch-221",
        "batch_221_sat",
        "--run-id",
        "classify_sat",
    ])

    assert rc == 0

    payload = json.loads(
        (runs_root / "classify_sat" / "classify_geometries" / "outputs" / "geometry_summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows_by_event = {row["event_id"]: row for row in payload["rows"]}

    assert rows_by_event["GW150914"]["n_compatible_220"] == 2637
    assert rows_by_event["GW150914"]["n_compatible_221"] == 2637
    assert rows_by_event["GW150914"]["support_region_n_final_221"] == 5
    assert rows_by_event["GW150914"]["classification"] == "common_nonempty_both_221_support_multi"
    assert rows_by_event["GW150914"]["has_joint_support"] is True

    assert rows_by_event["GW170814"]["classification"] == "common_nonempty_both_221_support_singleton"
    assert rows_by_event["GW170814"]["has_joint_support"] is True

    assert rows_by_event["GW190910_112807"]["classification"] == "common_nonempty_both_221_no_common_region"
    assert rows_by_event["GW190910_112807"]["has_joint_support"] is False

    assert payload["summary"]["n_joint_support"] == 2
    assert payload["summary"]["classification_counts"]["common_nonempty_both_221_support_multi"] == 1
    assert payload["summary"]["classification_counts"]["common_nonempty_both_221_support_singleton"] == 1
    assert payload["summary"]["classification_counts"]["common_nonempty_both_221_no_common_region"] == 1


def _write_batch_run(
    runs_root: Path,
    *,
    run_id: str,
    mode: str,
    rows: list[dict[str, object]],
    verdict: str = "PASS",
) -> None:
    stage_dir = runs_root / run_id / "run_batch"
    (stage_dir / "RUN_VALID").mkdir(parents=True, exist_ok=True)
    (stage_dir / "RUN_VALID" / "verdict.json").write_text(
        json.dumps({"verdict": verdict}),
        encoding="utf-8",
    )

    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": rows,
        "summary": {
            "n_events": len(rows),
            "n_pass": sum(1 for row in rows if row.get("status") == "PASS"),
            "n_fail": sum(1 for row in rows if row.get("status") != "PASS"),
        },
    }
    (outputs_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    (stage_dir / "stage_summary.json").write_text(
        json.dumps({"verdict": verdict, "parameters": {"mode": mode}}),
        encoding="utf-8",
    )
