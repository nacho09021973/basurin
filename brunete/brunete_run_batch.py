#!/usr/bin/env python3
"""Run a local BRUNETE batch analysis for mode 220 or 221.

This entrypoint consumes the normalized cohort produced by
``brunete_prepare_events.py`` and runs a minimal offline analysis path:

- mode ``220`` -> ``mvp.pipeline.run_single_event``
- mode ``221`` -> ``mvp.pipeline.run_multimode_event``

All event subruns are sandboxed under ``runs/<run_id>/run_batch/event_runs``.
The public batch contract is written under ``runs/<run_id>/run_batch``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basurin_io import (
    utc_now_iso,
    write_json_atomic,
)
from brunete.runtime import finalize_stage, init_stage, load_event_ids_from_text
from mvp import pipeline

STAGE_NAME = "run_batch"
OUTPUT_SCHEMA_VERSION = "brunete_run_batch_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="BRUNETE: run a local offline batch over a prepared cohort for mode 220 or 221.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--prepare-run", required=True)
    ap.add_argument("--mode", choices=["220", "221"], required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--losc-root", default="data/losc")
    ap.add_argument("--atlas-path", default=str(pipeline.DEFAULT_ATLAS_PATH))
    ap.add_argument("--epsilon", type=float, default=2500.0)
    ap.add_argument("--estimator", choices=["spectral", "dual", "hilbert"], default="spectral")
    ap.add_argument("--stage-timeout-s", type=float, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE_NAME,
        {
            "prepare_run_id": args.prepare_run,
            "mode": args.mode,
            "losc_root": str(Path(args.losc_root).expanduser().resolve()),
            "atlas_path": str(Path(args.atlas_path).expanduser().resolve()),
            "epsilon": float(args.epsilon),
            "estimator": args.estimator,
            "stage_timeout_s": args.stage_timeout_s,
        },
    )
    event_runs_root = ctx.stage_dir / "event_runs"
    event_runs_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    try:
        prepare_stage_dir = ctx.out_root / args.prepare_run / "prepare_events"
        _require_prepare_run_pass(prepare_stage_dir)

        events_source = prepare_stage_dir / "external_inputs" / "events.txt"
        if not events_source.exists():
            raise FileNotFoundError(f"prepared cohort missing events.txt: {events_source}")

        copied_events = ctx.external_inputs_dir / "events.txt"
        shutil.copy2(events_source, copied_events)
        artifacts["events_txt"] = copied_events

        prepare_catalog = prepare_stage_dir / "outputs" / "events_catalog.json"
        if prepare_catalog.exists():
            copied_catalog = ctx.external_inputs_dir / "events_catalog.json"
            shutil.copy2(prepare_catalog, copied_catalog)
            artifacts["events_catalog_json"] = copied_catalog

        atlas_path = Path(args.atlas_path).expanduser().resolve()
        if not atlas_path.exists():
            raise FileNotFoundError(f"atlas not found: {atlas_path}")

        losc_root = Path(args.losc_root).expanduser().resolve()
        if not losc_root.exists():
            raise FileNotFoundError(f"losc root not found: {losc_root}")
        if not losc_root.is_dir():
            raise ValueError(f"losc root is not a directory: {losc_root}")

        events = load_event_ids_from_text(copied_events)
        if not events:
            raise ValueError(f"prepared cohort is empty: {copied_events}")

        rows: list[dict[str, Any]] = []
        for event_id in events:
            row = _execute_event(
                event_id=event_id,
                mode=args.mode,
                atlas_path=atlas_path,
                losc_root=losc_root,
                event_runs_root=event_runs_root,
                epsilon=float(args.epsilon),
                estimator=args.estimator,
                stage_timeout_s=args.stage_timeout_s,
            )
            rows.append(row)

        summary = _summarize_rows(rows)
        batch_verdict = "PASS" if summary["n_pass"] > 0 else "FAIL"
        batch_reason = (
            "run_batch completed with at least one PASS event"
            if batch_verdict == "PASS"
            else "run_batch produced zero PASS events"
        )

        results_json_path = write_json_atomic(ctx.outputs_dir / "results.json", {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "created": utc_now_iso(),
            "stage": STAGE_NAME,
            "run_id": args.run_id,
            "prepare_run_id": args.prepare_run,
            "mode": args.mode,
            "results": rows,
            "summary": summary,
        })
        artifacts["results_json"] = results_json_path

        results_csv_path = ctx.outputs_dir / "results.csv"
        _write_results_csv(results_csv_path, rows)
        artifacts["results_csv"] = results_csv_path

        finalize_stage(
            ctx,
            verdict=batch_verdict,
            reason=batch_reason,
            results=summary,
            artifacts=artifacts,
            notes=[
                "BRUNETE run_batch currently uses local offline pipeline execution only.",
                "No t0 bootstrap/catalog dependency is required in this batch path.",
            ],
        )
        return 0 if batch_verdict == "PASS" else 2

    except Exception as exc:
        finalize_stage(
            ctx,
            verdict="FAIL",
            reason=str(exc),
            results={
                "n_events": 0,
                "n_pass": 0,
                "n_fail": 0,
                "n_compatible_sets_present": 0,
            },
            artifacts=artifacts,
            error=str(exc),
        )
        print(f"ERROR [{STAGE_NAME}]: {exc}", file=sys.stderr)
        return 2


def _require_prepare_run_pass(prepare_stage_dir: Path) -> None:
    verdict_path = prepare_stage_dir / "RUN_VALID" / "verdict.json"
    if not verdict_path.exists():
        raise FileNotFoundError(f"prepare_events RUN_VALID verdict not found: {verdict_path}")
    payload = json.loads(verdict_path.read_text(encoding="utf-8"))
    if payload.get("verdict") != "PASS":
        raise RuntimeError(
            f"prepare_events RUN_VALID verdict is not PASS: {prepare_stage_dir} -> {payload.get('verdict')!r}"
        )


def _execute_event(
    *,
    event_id: str,
    mode: str,
    atlas_path: Path,
    losc_root: Path,
    event_runs_root: Path,
    epsilon: float,
    estimator: str,
    stage_timeout_s: float | None,
) -> dict[str, Any]:
    event_run_id = _event_run_id(event_id, mode)
    with _temporary_environ({
        "BASURIN_RUNS_ROOT": str(event_runs_root),
        "BASURIN_LOSC_ROOT": str(losc_root),
    }):
        try:
            if mode == "220":
                exit_code, actual_run_id = pipeline.run_single_event(
                    event_id=event_id,
                    atlas_path=str(atlas_path),
                    run_id=event_run_id,
                    synthetic=False,
                    offline=True,
                    epsilon=epsilon,
                    estimator=estimator,
                    stage_timeout_s=stage_timeout_s,
                )
            else:
                exit_code, actual_run_id = pipeline.run_multimode_event(
                    event_id=event_id,
                    atlas_path=str(atlas_path),
                    run_id=event_run_id,
                    synthetic=False,
                    offline=True,
                    epsilon=epsilon,
                    estimator=estimator,
                    stage_timeout_s=stage_timeout_s,
                    minimal_run=True,
                )
        except Exception as exc:
            return {
                "event_id": event_id,
                "mode": mode,
                "event_run_id": event_run_id,
                "status": "FAIL",
                "exit_code": 2,
                "run_valid_verdict": "FAIL",
                "failure_reason": str(exc),
                "compatible_set_present": False,
                "n_compatible": None,
                "support_region_status": None,
                "support_region_n_final": None,
            }

    event_run_dir = event_runs_root / actual_run_id
    run_valid_payload = _read_json_if_exists(event_run_dir / "RUN_VALID" / "verdict.json") or {}
    compatible_payload = _read_json_if_exists(event_run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json")
    support_payload = _read_json_if_exists(event_run_dir / "s4k_event_support_region" / "outputs" / "event_support_region.json")

    n_compatible = None
    compatible_present = False
    if isinstance(compatible_payload, dict):
        compatible_present = True
        raw_n_compatible = compatible_payload.get("n_compatible")
        if isinstance(raw_n_compatible, int):
            n_compatible = raw_n_compatible
        elif isinstance(compatible_payload.get("compatible_geometries"), list):
            n_compatible = len(compatible_payload["compatible_geometries"])

    support_region_status = None
    support_region_n_final = None
    if isinstance(support_payload, dict):
        status = support_payload.get("support_region_status")
        support_region_status = str(status) if status is not None else None
        raw_n_final = support_payload.get("n_final_geometries")
        if isinstance(raw_n_final, int):
            support_region_n_final = raw_n_final

    run_valid_verdict = str(run_valid_payload.get("verdict") or "").upper() or "UNKNOWN"
    failure_reason = ""
    if exit_code != 0 or run_valid_verdict != "PASS":
        failure_reason = str(run_valid_payload.get("reason") or f"exit={exit_code}")

    return {
        "event_id": event_id,
        "mode": mode,
        "event_run_id": actual_run_id,
        "status": "PASS" if exit_code == 0 and run_valid_verdict == "PASS" else "FAIL",
        "exit_code": int(exit_code),
        "run_valid_verdict": run_valid_verdict,
        "failure_reason": failure_reason,
        "compatible_set_present": compatible_present,
        "n_compatible": n_compatible,
        "support_region_status": support_region_status,
        "support_region_n_final": support_region_n_final,
    }


def _event_run_id(event_id: str, mode: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in event_id).strip("._-")
    safe = safe or "unknown_event"
    safe = safe[:96]
    return f"brunete_{safe}_m{mode}"


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "mode",
        "event_run_id",
        "status",
        "exit_code",
        "run_valid_verdict",
        "failure_reason",
        "compatible_set_present",
        "n_compatible",
        "support_region_status",
        "support_region_n_final",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_events": len(rows),
        "n_pass": sum(1 for row in rows if row["status"] == "PASS"),
        "n_fail": sum(1 for row in rows if row["status"] != "PASS"),
        "n_compatible_sets_present": sum(1 for row in rows if row["compatible_set_present"]),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


@contextmanager
def _temporary_environ(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


if __name__ == "__main__":
    raise SystemExit(main())
