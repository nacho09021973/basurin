#!/usr/bin/env python3
"""Canonical stage s6_multi_event_table: aggregate multiple event_row files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import (
    ensure_stage_dirs,
    require_run_valid,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)

STAGE = "s6_multi_event_table"


def _fatal(msg: str) -> None:
    print(f"ERROR: [{STAGE}] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _read_run_ids(path: Path) -> list[str]:
    run_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        run_ids.append(item)
    return run_ids


def _agg_run_id(run_ids: list[str]) -> str:
    canonical = "\n".join(run_ids).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:16]
    return f"agg_s6_multi_event_{digest}"


def _flatten(obj: Any, *, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(obj[key], prefix=child_prefix))
        return out
    if isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        return out
    if obj is None:
        out[prefix] = ""
        return out
    out[prefix] = str(obj)
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            fh.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> tuple[list[str], int]:
    flattened = [_flatten(row) for row in rows]
    columns: list[str] = sorted({k for row in flattened for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in flattened:
            writer.writerow({c: row.get(c, "") for c in columns})
    return columns, len(flattened)


def _write_optional_parquet(path: Path, rows: list[dict[str, Any]]) -> str | None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None

    flattened = [_flatten(row) for row in rows]
    columns: list[str] = sorted({k for row in flattened for k in row.keys()})
    table = pa.table({c: [row.get(c, "") for row in flattened] for c in columns})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="MVP s6 multi event table")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--run-ids-file", required=True)
    args = ap.parse_args()

    os.environ["BASURIN_RUNS_ROOT"] = str(Path(args.runs_root).expanduser().resolve())
    out_root = resolve_out_root("runs")

    run_ids_path = Path(args.run_ids_file).expanduser().resolve()
    if not run_ids_path.exists():
        _fatal(f"run-ids-file not found: {run_ids_path}")

    run_ids = _read_run_ids(run_ids_path)
    if not run_ids:
        _fatal("run-ids-file has no run_id entries")

    agg_run_id = _agg_run_id(run_ids)
    validate_run_id(agg_run_id, out_root)
    require_run_valid(out_root, agg_run_id)

    stage_dir, outputs_dir = ensure_stage_dirs(agg_run_id, STAGE, base_dir=out_root)

    rows: list[dict[str, Any]] = []
    input_records: list[dict[str, str]] = []
    missing: list[str] = []

    for run_id in run_ids:
        input_path = out_root / run_id / "s5_event_row" / "outputs" / "event_row.json"
        if not input_path.exists():
            missing.append(run_id)
            continue
        row = _read_json(input_path)
        rows.append(row)
        input_records.append(
            {
                "run_id": run_id,
                "path": str(input_path.relative_to(out_root / agg_run_id)) if input_path.is_relative_to(out_root / agg_run_id) else str(input_path),
                "sha256": sha256_file(input_path),
            }
        )

    if missing:
        _fatal("missing event_row for run_id(s): " + ", ".join(missing))

    jsonl_path = outputs_dir / "multi_event.jsonl"
    csv_path = outputs_dir / "multi_event.csv"

    _write_jsonl(jsonl_path, rows)
    csv_columns, row_count = _write_csv(csv_path, rows)

    outputs: list[dict[str, str]] = [
        {"path": str(jsonl_path.relative_to(out_root / agg_run_id)), "sha256": sha256_file(jsonl_path)},
        {"path": str(csv_path.relative_to(out_root / agg_run_id)), "sha256": sha256_file(csv_path)},
    ]

    parquet_path = outputs_dir / "multi_event.parquet"
    parquet_written = _write_optional_parquet(parquet_path, rows)
    if parquet_written:
        outputs.append({"path": str(parquet_path.relative_to(out_root / agg_run_id)), "sha256": sha256_file(parquet_path)})

    summary = {
        "stage": STAGE,
        "run": agg_run_id,
        "runs_root": str(out_root),
        "created": utc_now_iso(),
        "version": "v1",
        "parameters": {
            "run_ids_file": str(run_ids_path),
            "run_ids_count": len(run_ids),
        },
        "inputs": input_records,
        "outputs": outputs,
        "verdict": "PASS",
        "results": {
            "row_count": row_count,
            "csv_columns": csv_columns,
            "parquet_written": bool(parquet_written),
        },
    }

    summary_path = write_stage_summary(stage_dir, summary)
    write_manifest(
        stage_dir,
        {
            "multi_event_jsonl": jsonl_path,
            "multi_event_csv": csv_path,
            "stage_summary": summary_path,
        },
        extra={"inputs": input_records},
    )

    print(f"OK: {STAGE} PASS run={agg_run_id} rows={row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
