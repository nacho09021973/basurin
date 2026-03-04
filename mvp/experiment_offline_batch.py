#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from basurin_io import (
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_manifest,
    write_stage_summary,
)
from mvp import contracts

DEFAULT_EVENTS_FILE = (
    "runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/gwosc_ready_events.txt"
)
DEFAULT_T0_CATALOG = (
    "runs/audit_gwosc_t0_20260304T115440Z/experiment/losc_quality/t0_catalog_gwosc_v2.json"
)
DEFAULT_ATLAS_PATH = "docs/ringdown/atlas/atlas_berti_v2.json"


@dataclass
class EventResult:
    event_id: str
    run_id: str
    status: str
    len_compatible: int
    epsilon_used: float
    mode_filter: str
    error_stage: str
    error_message_short: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _event_run_id(event_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in event_id)
    return f"mvp_{safe}_real_offline_{_utc_stamp()}"


def _read_events(events_file: Path) -> list[str]:
    if not events_file.exists():
        raise FileNotFoundError(
            "events file missing. "
            f"expected path: {events_file}. "
            "upstream regenerate command: "
            "python -m mvp.audit_gwosc_losc_quality --run-id audit_gwosc_t0_20260304T115440Z"
        )

    events: list[str] = []
    for raw in events_file.read_text(encoding="utf-8").splitlines():
        token = raw.strip()
        if token and not token.startswith("#"):
            events.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for event_id in events:
        if event_id not in seen:
            deduped.append(event_id)
            seen.add(event_id)
    return deduped


def _run_cmd(cmd: list[str], *, env: dict[str, str]) -> None:
    print("[offline_batch] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _read_len_compatible(out_root: Path, run_id: str) -> int:
    ranked_path = (
        out_root / run_id / "s6b_information_geometry_ranked" / "outputs" / "ranked_geometries.json"
    )
    if not ranked_path.exists():
        raise FileNotFoundError(f"ranked output missing: {ranked_path}")

    payload = json.loads(ranked_path.read_text(encoding="utf-8"))
    raw = payload.get("len_compatible", payload.get("n_compatible", 0))
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid len_compatible in {ranked_path}: {raw!r}") from exc


def _build_stage_commands(
    *,
    python_exe: str,
    event_id: str,
    run_id: str,
    window_catalog_path: Path,
    atlas_path: Path,
    epsilon: float,
    mode_filter: str | None,
) -> list[tuple[str, list[str]]]:
    s4_cmd = [
        python_exe,
        "-m",
        "mvp.s4_geometry_filter",
        "--run",
        run_id,
        "--atlas-path",
        str(atlas_path),
        "--epsilon",
        str(float(epsilon)),
        "--metric",
        "mahalanobis_log",
    ]
    if mode_filter is not None:
        s4_cmd.extend(["--mode-filter", mode_filter])

    return [
        (
            "s1_fetch_strain",
            [
                python_exe,
                "-m",
                "mvp.s1_fetch_strain",
                "--run",
                run_id,
                "--event-id",
                event_id,
                "--detectors",
                "H1,L1",
            ],
        ),
        (
            "s2_ringdown_window",
            [
                python_exe,
                "-m",
                "mvp.s2_ringdown_window",
                "--run",
                run_id,
                "--event-id",
                event_id,
                "--window-catalog",
                str(window_catalog_path),
                "--offline",
            ],
        ),
        ("s3_ringdown_estimates", [python_exe, "-m", "mvp.s3_ringdown_estimates", "--run", run_id]),
        (
            "s4_geometry_filter",
            s4_cmd,
        ),
        ("s6_information_geometry", [python_exe, "-m", "mvp.s6_information_geometry", "--run", run_id]),
        (
            "s6b_information_geometry_ranked",
            [python_exe, "-m", "mvp.s6b_information_geometry_ranked", "--run", run_id],
        ),
    ]


def _execute_event(
    *,
    out_root: Path,
    event_id: str,
    window_catalog_path: Path,
    atlas_path: Path,
    epsilon_default: float,
    epsilon_fallback: float,
    mode_filter: str | None,
    env: dict[str, str],
) -> EventResult:
    run_id = _event_run_id(event_id)
    epsilon_used = float(epsilon_default)

    try:
        for stage_name, cmd in _build_stage_commands(
            python_exe=sys.executable,
            event_id=event_id,
            run_id=run_id,
            window_catalog_path=window_catalog_path,
            atlas_path=atlas_path,
            epsilon=epsilon_default,
            mode_filter=mode_filter,
        ):
            _run_cmd(cmd, env=env)

        len_compatible = _read_len_compatible(out_root, run_id)

        if len_compatible == 0:
            _run_cmd(
                _build_stage_commands(
                    python_exe=sys.executable,
                    event_id=event_id,
                    run_id=run_id,
                    window_catalog_path=window_catalog_path,
                    atlas_path=atlas_path,
                    epsilon=epsilon_fallback,
                    mode_filter=mode_filter,
                )[3][1],
                env=env,
            )
            _run_cmd(
                [sys.executable, "-m", "mvp.s6b_information_geometry_ranked", "--run", run_id],
                env=env,
            )
            len_compatible = _read_len_compatible(out_root, run_id)
            epsilon_used = float(epsilon_fallback)

        return EventResult(
            event_id=event_id,
            run_id=run_id,
            status="PASS",
            len_compatible=int(len_compatible),
            epsilon_used=epsilon_used,
            mode_filter=mode_filter or "",
            error_stage="",
            error_message_short="",
        )
    except subprocess.CalledProcessError as exc:
        return EventResult(
            event_id=event_id,
            run_id=run_id,
            status="FAIL",
            len_compatible=0,
            epsilon_used=epsilon_used,
            mode_filter=mode_filter or "",
            error_stage="subprocess",
            error_message_short=f"exit={exc.returncode}",
        )
    except Exception as exc:
        return EventResult(
            event_id=event_id,
            run_id=run_id,
            status="FAIL",
            len_compatible=0,
            epsilon_used=epsilon_used,
            mode_filter=mode_filter or "",
            error_stage="runtime",
            error_message_short=str(exc)[:160],
        )


def _write_results_csv(results: list[EventResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "event_id",
                "run_id",
                "status",
                "len_compatible",
                "epsilon_used",
                "mode_filter",
                "error_stage",
                "error_message_short",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.event_id,
                    row.run_id,
                    row.status,
                    row.len_compatible,
                    row.epsilon_used,
                    row.mode_filter,
                    row.error_stage,
                    row.error_message_short,
                ]
            )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch offline-first runner over GWOSC-ready events")
    ap.add_argument("--batch-run-id", required=True)
    ap.add_argument("--events-file", default=DEFAULT_EVENTS_FILE)
    ap.add_argument("--window-catalog", default=None)
    ap.add_argument("--t0-catalog", default=None)
    ap.add_argument("--atlas-path", default=DEFAULT_ATLAS_PATH)
    ap.add_argument("--epsilon-default", type=float, default=0.3)
    ap.add_argument("--epsilon-fallback", type=float, default=1200.0)
    ap.add_argument("--mode-filter", default=None)
    ap.add_argument("--max-events", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    out_root = resolve_out_root("runs")
    validate_run_id(args.batch_run_id, out_root)

    events_file = Path(args.events_file).expanduser().resolve()
    window_catalog_cli = args.window_catalog if args.window_catalog is not None else args.t0_catalog
    window_catalog_path = Path(window_catalog_cli or DEFAULT_T0_CATALOG).expanduser().resolve()
    atlas_path = Path(args.atlas_path).expanduser().resolve()

    if not window_catalog_path.exists():
        raise SystemExit(
            "ERROR: missing required input. "
            f"expected path: {window_catalog_path}. "
            "upstream regenerate command: "
            "python -m mvp.audit_gwosc_losc_quality --run-id audit_gwosc_t0_20260304T115440Z"
        )
    if not atlas_path.exists():
        raise SystemExit(f"ERROR: atlas-path not found: {atlas_path}")

    events = _read_events(events_file)
    if args.max_events is not None:
        events = events[: max(int(args.max_events), 0)]

    stage_dir = out_root / args.batch_run_id / "experiment" / "offline_batch"
    outputs_dir = stage_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_csv = stage_dir / "results.csv"
    outputs_results_csv = outputs_dir / "results.csv"

    env = os.environ.copy()
    env["BASURIN_RUNS_ROOT"] = str(out_root)

    results: list[EventResult] = []
    for event_id in events:
        results.append(
            _execute_event(
                out_root=out_root,
                event_id=event_id,
                window_catalog_path=window_catalog_path,
                atlas_path=atlas_path,
                epsilon_default=args.epsilon_default,
                epsilon_fallback=args.epsilon_fallback,
                mode_filter=args.mode_filter,
                env=env,
            )
        )

    _write_results_csv(results, results_csv)
    _write_results_csv(results, outputs_results_csv)

    summary_payload: dict[str, Any] = {
        "stage": "experiment_offline_batch",
        "run": args.batch_run_id,
        "runs_root": str(out_root),
        "created": utc_now_iso(),
        "version": "v1",
        "parameters": {
            "events_file": str(events_file),
            "window_catalog": str(window_catalog_path),
            "atlas_path": str(atlas_path),
            "epsilon_default": float(args.epsilon_default),
            "epsilon_fallback": float(args.epsilon_fallback),
            "mode_filter": args.mode_filter,
            "max_events": args.max_events,
        },
        "inputs": [
            {"label": "events_file", "path": str(events_file), "sha256": sha256_file(events_file)},
            {
                "label": "window_catalog",
                "path": str(window_catalog_path),
                "sha256": sha256_file(window_catalog_path),
            },
            {"label": "atlas_path", "path": str(atlas_path), "sha256": sha256_file(atlas_path)},
        ],
        "outputs": [
            {
                "path": "results.csv",
                "sha256": sha256_file(results_csv),
            },
            {
                "path": "outputs/results.csv",
                "sha256": sha256_file(outputs_results_csv),
            },
        ],
        "results": {
            "n_events": len(events),
            "n_pass": sum(1 for r in results if r.status == "PASS"),
            "n_fail": sum(1 for r in results if r.status != "PASS"),
        },
        "verdict": "PASS",
    }
    stage_summary = write_stage_summary(stage_dir, summary_payload)
    write_manifest(
        stage_dir,
        {
            "results_csv": results_csv,
            "outputs_results_csv": outputs_results_csv,
            "stage_summary": stage_summary,
        },
    )

    ctx = contracts.StageContext(
        run_id=args.batch_run_id,
        stage_name="experiment_offline_batch",
        contract=contracts.StageContract(
            name="experiment_offline_batch",
            required_inputs=[],
            produced_outputs=[],
            upstream_stages=[],
        ),
        out_root=out_root,
        run_dir=out_root / args.batch_run_id,
        stage_dir=stage_dir,
        outputs_dir=outputs_dir,
    )
    contracts.log_stage_paths(ctx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
