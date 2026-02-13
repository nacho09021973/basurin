#!/usr/bin/env python3
"""MVP Pipeline Orchestrator: end-to-end ringdown analysis.

Single-event:
    python mvp/pipeline.py single --event-id GW150914 --atlas-path atlas.json \
        [--run-id custom_run_id] [--synthetic]

Multi-event:
    python mvp/pipeline.py multi --events GW150914,GW151226 --atlas-path atlas.json \
        [--min-coverage 1.0] [--synthetic]

What it does:
    1. Creates runs/<run_id>/RUN_VALID/verdict.json
    2. Runs s1_fetch_strain (download or synthetic)
    3. Runs s2_ringdown_window (crop ringdown)
    4. Runs s3_ringdown_estimates (f, tau, Q)
    5. Runs s4_geometry_filter (atlas compatibility)
    6. [multi] Runs s5_aggregate (intersection across events)

Abort semantics:
    If ANY stage exits != 0, the pipeline stops immediately.
    The run's stage_summary.json will have verdict=FAIL.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import resolve_out_root, write_json_atomic

MVP_DIR = Path(__file__).resolve().parent


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _generate_run_id(event_id: str) -> str:
    return f"mvp_{event_id}_{_ts()}"


def _create_run_valid(out_root: Path, run_id: str) -> None:
    rv_dir = out_root / run_id / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_dir / "verdict.json", {
        "verdict": "PASS",
        "created": datetime.now(timezone.utc).isoformat(),
        "note": "MVP pipeline initialization",
    })
    print(f"[pipeline] RUN_VALID created for {run_id}")


def _run_stage(script: str, args: list[str], label: str) -> int:
    cmd = [sys.executable, str(MVP_DIR / script)] + args
    print(f"\n{'=' * 60}")
    print(f"[pipeline] Stage: {label}")
    print(f"[pipeline] Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[pipeline] ABORT: {label} failed (exit={result.returncode})", file=sys.stderr)
    return result.returncode


def run_single_event(
    event_id: str,
    atlas_path: str,
    run_id: str | None = None,
    synthetic: bool = False,
    duration_s: float = 32.0,
    dt_start_s: float = 0.003,
    window_duration_s: float = 0.06,
    band_low: float = 150.0,
    band_high: float = 400.0,
    epsilon: float = 0.3,
) -> tuple[int, str]:
    """Run full pipeline for a single event. Returns (exit_code, run_id)."""
    out_root = resolve_out_root("runs")

    if run_id is None:
        run_id = _generate_run_id(event_id)

    print(f"\n[pipeline] Starting single-event pipeline")
    print(f"[pipeline] event_id={event_id}, run_id={run_id}")
    print(f"[pipeline] atlas={atlas_path}, synthetic={synthetic}")

    _create_run_valid(out_root, run_id)

    # Stage 1: Fetch strain
    s1_args = ["--run", run_id, "--event-id", event_id, "--duration-s", str(duration_s)]
    if synthetic:
        s1_args.append("--synthetic")
    rc = _run_stage("s1_fetch_strain.py", s1_args, "s1_fetch_strain")
    if rc != 0:
        return rc, run_id

    # Stage 2: Ringdown window
    s2_args = [
        "--run", run_id, "--event-id", event_id,
        "--dt-start-s", str(dt_start_s),
        "--duration-s", str(window_duration_s),
    ]
    rc = _run_stage("s2_ringdown_window.py", s2_args, "s2_ringdown_window")
    if rc != 0:
        return rc, run_id

    # Stage 3: Ringdown estimates
    s3_args = ["--run", run_id, "--band-low", str(band_low), "--band-high", str(band_high)]
    rc = _run_stage("s3_ringdown_estimates.py", s3_args, "s3_ringdown_estimates")
    if rc != 0:
        return rc, run_id

    # Stage 4: Geometry filter
    s4_args = ["--run", run_id, "--atlas-path", atlas_path, "--epsilon", str(epsilon)]
    rc = _run_stage("s4_geometry_filter.py", s4_args, "s4_geometry_filter")
    if rc != 0:
        return rc, run_id

    print(f"\n[pipeline] Single-event pipeline COMPLETE: run_id={run_id}")
    return 0, run_id


def run_multi_event(
    events: list[str],
    atlas_path: str,
    agg_run_id: str | None = None,
    min_coverage: float = 1.0,
    **kwargs: Any,
) -> tuple[int, str]:
    """Run pipeline for multiple events, then aggregate."""
    if agg_run_id is None:
        agg_run_id = f"mvp_aggregate_{_ts()}"

    print(f"\n[pipeline] Starting multi-event pipeline")
    print(f"[pipeline] events={events}")
    print(f"[pipeline] aggregate_run={agg_run_id}")

    per_event_runs: list[str] = []

    for event_id in events:
        rc, run_id = run_single_event(event_id=event_id, atlas_path=atlas_path, **kwargs)
        if rc != 0:
            print(f"[pipeline] ABORT: event {event_id} failed", file=sys.stderr)
            return rc, agg_run_id
        per_event_runs.append(run_id)

    # Stage 5: Aggregate
    s5_args = [
        "--out-run", agg_run_id,
        "--source-runs", ",".join(per_event_runs),
        "--min-coverage", str(min_coverage),
    ]
    rc = _run_stage("s5_aggregate.py", s5_args, "s5_aggregate")
    if rc != 0:
        return rc, agg_run_id

    print(f"\n[pipeline] Multi-event pipeline COMPLETE: agg_run={agg_run_id}")
    print(f"[pipeline] Per-event runs: {per_event_runs}")
    return 0, agg_run_id


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MVP Pipeline Orchestrator")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Single event
    sp_single = sub.add_parser("single", help="Run pipeline for one event")
    sp_single.add_argument("--event-id", required=True)
    sp_single.add_argument("--atlas-path", required=True)
    sp_single.add_argument("--run-id", default=None)
    sp_single.add_argument("--synthetic", action="store_true")
    sp_single.add_argument("--duration-s", type=float, default=32.0)
    sp_single.add_argument("--dt-start-s", type=float, default=0.003)
    sp_single.add_argument("--window-duration-s", type=float, default=0.06)
    sp_single.add_argument("--band-low", type=float, default=150.0)
    sp_single.add_argument("--band-high", type=float, default=400.0)
    sp_single.add_argument("--epsilon", type=float, default=0.3)

    # Multi event
    sp_multi = sub.add_parser("multi", help="Run pipeline for multiple events + aggregate")
    sp_multi.add_argument("--events", required=True, help="Comma-separated event IDs")
    sp_multi.add_argument("--atlas-path", required=True)
    sp_multi.add_argument("--agg-run-id", default=None)
    sp_multi.add_argument("--min-coverage", type=float, default=1.0)
    sp_multi.add_argument("--synthetic", action="store_true")
    sp_multi.add_argument("--duration-s", type=float, default=32.0)
    sp_multi.add_argument("--dt-start-s", type=float, default=0.003)
    sp_multi.add_argument("--window-duration-s", type=float, default=0.06)
    sp_multi.add_argument("--band-low", type=float, default=150.0)
    sp_multi.add_argument("--band-high", type=float, default=400.0)
    sp_multi.add_argument("--epsilon", type=float, default=0.3)

    args = parser.parse_args()

    if args.mode == "single":
        rc, run_id = run_single_event(
            event_id=args.event_id,
            atlas_path=args.atlas_path,
            run_id=args.run_id,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
        )
        return rc

    elif args.mode == "multi":
        events = [e.strip() for e in args.events.split(",") if e.strip()]
        rc, agg_run_id = run_multi_event(
            events=events,
            atlas_path=args.atlas_path,
            agg_run_id=args.agg_run_id,
            min_coverage=args.min_coverage,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
        )
        return rc

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
