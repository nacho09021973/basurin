#!/usr/bin/env python3
"""Build a joint BRUNETE geometry summary from batch 220 and batch 221.

This public stage consumes two valid BRUNETE batch runs and emits a simple,
explicit joint classification under ``runs/<run_id>/classify_geometries``.

The first cut stays intentionally small:

- cross by ``event_id``
- mark presence in 220 / 221 / both
- expose per-mode status and ``n_compatible``
- assign a clear, extensible classification label per event
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    utc_now_iso,
    write_json_atomic,
)
from brunete.runtime import finalize_stage, init_stage

STAGE_NAME = "classify_geometries"
OUTPUT_SCHEMA_VERSION = "brunete_classify_geometries_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="BRUNETE: classify joint geometry support from a valid 220 batch and a valid 221 batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--batch-220", required=True)
    ap.add_argument("--batch-221", required=True)
    ap.add_argument("--run-id", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE_NAME,
        {
            "batch_220_run_id": args.batch_220,
            "batch_221_run_id": args.batch_221,
        },
    )
    artifacts: dict[str, Path] = {}

    try:
        batch220 = _load_batch_payload(ctx.out_root, args.batch_220, expected_mode="220")
        batch221 = _load_batch_payload(ctx.out_root, args.batch_221, expected_mode="221")

        batch220_copy = _snapshot_json(
            ctx.external_inputs_dir / "batch_220_results.json",
            batch220["payload"],
        )
        batch221_copy = _snapshot_json(
            ctx.external_inputs_dir / "batch_221_results.json",
            batch221["payload"],
        )
        artifacts["batch_220_results_json"] = batch220_copy
        artifacts["batch_221_results_json"] = batch221_copy

        rows_220 = {row["event_id"]: row for row in batch220["payload"]["results"]}
        rows_221 = {row["event_id"]: row for row in batch221["payload"]["results"]}
        event_ids = sorted(set(rows_220) | set(rows_221))

        summary_rows = [
            _classify_event(
                event_id=event_id,
                row_220=rows_220.get(event_id),
                row_221=rows_221.get(event_id),
            )
            for event_id in event_ids
        ]
        summary = _summarize(summary_rows)

        geometry_summary_path = write_json_atomic(ctx.outputs_dir / "geometry_summary.json", {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "created": utc_now_iso(),
            "stage": STAGE_NAME,
            "run_id": args.run_id,
            "batch_220_run_id": args.batch_220,
            "batch_221_run_id": args.batch_221,
            "summary": summary,
            "rows": summary_rows,
        })
        artifacts["geometry_summary_json"] = geometry_summary_path

        geometry_summary_csv_path = ctx.outputs_dir / "geometry_summary.csv"
        _write_summary_csv(geometry_summary_csv_path, summary_rows)
        artifacts["geometry_summary_csv"] = geometry_summary_csv_path

        finalize_stage(
            ctx,
            verdict="PASS",
            reason="classify_geometries completed successfully",
            results=summary,
            artifacts=artifacts,
            notes=[
                "Classification is intentionally minimal and explicit.",
                "This stage crosses event_id, per-mode status, and n_compatible only.",
            ],
        )
        return 0

    except Exception as exc:
        finalize_stage(
            ctx,
            verdict="FAIL",
            reason=str(exc),
            results={
                "n_events_union": 0,
                "n_events_both": 0,
                "n_events_only_220": 0,
                "n_events_only_221": 0,
                "n_joint_support": 0,
                "classification_counts": {},
            },
            artifacts=artifacts,
            error=str(exc),
        )
        print(f"ERROR [{STAGE_NAME}]: {exc}", file=sys.stderr)
        return 2


def _load_batch_payload(out_root: Path, batch_run_id: str, *, expected_mode: str) -> dict[str, Any]:
    stage_dir = out_root / batch_run_id / "run_batch"
    verdict_path = stage_dir / "RUN_VALID" / "verdict.json"
    if not verdict_path.exists():
        raise FileNotFoundError(f"run_batch RUN_VALID verdict not found: {verdict_path}")
    verdict_payload = json.loads(verdict_path.read_text(encoding="utf-8"))
    if verdict_payload.get("verdict") != "PASS":
        raise RuntimeError(
            f"run_batch RUN_VALID verdict is not PASS: {batch_run_id} -> {verdict_payload.get('verdict')!r}"
        )

    summary_path = stage_dir / "stage_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run_batch stage_summary.json not found: {summary_path}")
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary_payload.get("verdict") != "PASS":
        raise RuntimeError(
            f"run_batch stage_summary.json is not PASS: {summary_path} -> {summary_payload.get('verdict')!r}"
        )

    results_path = stage_dir / "outputs" / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"run_batch results.json not found: {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"batch results schema invalid in {results_path}: expected list at key 'results'")

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"batch results row must be a dict in {results_path}: {row!r}")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError(f"batch results row missing event_id in {results_path}: {row!r}")
        mode = str(row.get("mode") or "").strip()
        if mode != expected_mode:
            raise ValueError(
                f"batch mode mismatch in {results_path}: expected {expected_mode!r}, got {mode!r} for event {event_id!r}"
            )

    return {
        "stage_dir": stage_dir,
        "payload": payload,
    }


def _snapshot_json(path: Path, payload: dict[str, Any]) -> Path:
    return write_json_atomic(path, payload)


def _classify_event(
    *,
    event_id: str,
    row_220: dict[str, Any] | None,
    row_221: dict[str, Any] | None,
) -> dict[str, Any]:
    in_220 = row_220 is not None
    in_221 = row_221 is not None
    status_220 = _status_of(row_220)
    status_221 = _status_of(row_221)
    n_compatible_220 = _n_compatible_of(row_220)
    n_compatible_221 = _n_compatible_of(row_221)

    classification = _classification_label(
        in_220=in_220,
        in_221=in_221,
        status_220=status_220,
        status_221=status_221,
        n_compatible_220=n_compatible_220,
        n_compatible_221=n_compatible_221,
    )

    return {
        "event_id": event_id,
        "in_batch_220": in_220,
        "in_batch_221": in_221,
        "in_both_batches": in_220 and in_221,
        "status_220": status_220,
        "status_221": status_221,
        "event_run_id_220": _string_or_none(row_220, "event_run_id"),
        "event_run_id_221": _string_or_none(row_221, "event_run_id"),
        "n_compatible_220": n_compatible_220,
        "n_compatible_221": n_compatible_221,
        "classification": classification,
        "has_joint_support": bool(
            in_220
            and in_221
            and status_220 == "PASS"
            and status_221 == "PASS"
            and (n_compatible_220 or 0) > 0
            and (n_compatible_221 or 0) > 0
        ),
    }


def _classification_label(
    *,
    in_220: bool,
    in_221: bool,
    status_220: str | None,
    status_221: str | None,
    n_compatible_220: int | None,
    n_compatible_221: int | None,
) -> str:
    if in_220 and not in_221:
        return "only_batch_220"
    if in_221 and not in_220:
        return "only_batch_221"

    assert in_220 and in_221

    if status_220 != "PASS" and status_221 != "PASS":
        return "common_failed_both"
    if status_220 != "PASS":
        return "common_failed_220"
    if status_221 != "PASS":
        return "common_failed_221"

    n220 = n_compatible_220 or 0
    n221 = n_compatible_221 or 0
    if n220 > 0 and n221 > 0:
        return "common_nonempty_both"
    if n220 > 0 and n221 <= 0:
        return "common_nonempty_220_only"
    if n220 <= 0 and n221 > 0:
        return "common_nonempty_221_only"
    return "common_empty_both"


def _status_of(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    value = row.get("status")
    if value is None:
        return None
    return str(value).strip().upper()


def _n_compatible_of(row: dict[str, Any] | None) -> int | None:
    if row is None:
        return None
    value = row.get("n_compatible")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(row: dict[str, Any] | None, key: str) -> str | None:
    if row is None:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    for row in rows:
        label = row["classification"]
        class_counts[label] = class_counts.get(label, 0) + 1

    return {
        "n_events_union": len(rows),
        "n_events_both": sum(1 for row in rows if row["in_both_batches"]),
        "n_events_only_220": sum(1 for row in rows if row["classification"] == "only_batch_220"),
        "n_events_only_221": sum(1 for row in rows if row["classification"] == "only_batch_221"),
        "n_joint_support": sum(1 for row in rows if row["has_joint_support"]),
        "classification_counts": class_counts,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "in_batch_220",
        "in_batch_221",
        "in_both_batches",
        "status_220",
        "status_221",
        "event_run_id_220",
        "event_run_id_221",
        "n_compatible_220",
        "n_compatible_221",
        "classification",
        "has_joint_support",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
