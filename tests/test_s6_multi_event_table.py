from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _agg_run_id(run_ids: list[str]) -> str:
    digest = hashlib.sha256("\n".join(run_ids).encode("utf-8")).hexdigest()[:16]
    return f"agg_s6_multi_event_{digest}"


def _cmd(repo_root: Path, runs_root: Path, run_ids_file: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mvp.s6_multi_event_table",
        "--runs-root",
        str(runs_root),
        "--run-ids-file",
        str(run_ids_file),
    ]


def test_s6_multi_event_table_concat_two_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"

    run_ids = ["base_run_a", "base_run_b"]
    for idx, run_id in enumerate(run_ids, start=1):
        payload = {
            "schema_version": "s5_event_row_v1",
            "run_id": run_id,
            "seed": idx,
            "t0_ms": idx * 5,
            "s4c": {"verdict": "PASS"},
        }
        _write_json(runs_root / run_id / "s5_event_row" / "outputs" / "event_row.json", payload)

    agg_run = _agg_run_id(run_ids)
    _write_json(runs_root / agg_run / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    run_ids_file = tmp_path / "run_ids.txt"
    run_ids_file.write_text("\n".join(run_ids) + "\n", encoding="utf-8")

    proc = subprocess.run(_cmd(repo_root, runs_root, run_ids_file), cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    stage_root = runs_root / agg_run / "s6_multi_event_table"
    jsonl_path = stage_root / "outputs" / "multi_event.jsonl"
    csv_path = stage_root / "outputs" / "multi_event.csv"

    assert jsonl_path.exists()
    assert csv_path.exists()
    assert (stage_root / "manifest.json").exists()
    assert (stage_root / "stage_summary.json").exists()

    jsonl_lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    csv_lines = [line for line in csv_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(jsonl_lines) == 2
    assert len(csv_lines) == 3


def test_s6_aborts_on_missing_row(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"

    run_ids = ["base_run_ok", "base_run_missing"]
    _write_json(
        runs_root / run_ids[0] / "s5_event_row" / "outputs" / "event_row.json",
        {"schema_version": "s5_event_row_v1", "run_id": run_ids[0]},
    )

    agg_run = _agg_run_id(run_ids)
    _write_json(runs_root / agg_run / "RUN_VALID" / "verdict.json", {"verdict": "PASS"})

    run_ids_file = tmp_path / "run_ids.txt"
    run_ids_file.write_text("\n".join(run_ids) + "\n", encoding="utf-8")

    proc = subprocess.run(_cmd(repo_root, runs_root, run_ids_file), cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "missing" in proc.stderr.lower()
