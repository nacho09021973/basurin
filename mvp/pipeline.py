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
import threading
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
DEFAULT_ATLAS_PATH = Path("docs/ringdown/atlas/atlas_berti_v2.json")


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


def _write_timeline(out_root: Path, run_id: str, timeline: dict[str, Any]) -> None:
    write_json_atomic(out_root / run_id / "pipeline_timeline.json", timeline)


def _heartbeat(label: str, t0: float, stop: threading.Event, interval: float = 5.0) -> None:
    """Print elapsed time every *interval* seconds while a stage runs."""
    while not stop.wait(interval):
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        print(f"[pipeline] {label} ... elapsed {mins:02d}:{secs:02d}", flush=True)


def _run_stage(
    script: str,
    args: list[str],
    label: str,
    out_root: Path,
    run_id: str,
    timeline: dict[str, Any],
    stage_timeout_s: float | None = None,
) -> int:
    cmd = [sys.executable, str(MVP_DIR / script)] + args
    stage_started = datetime.now(timezone.utc).isoformat()
    stage_t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"[pipeline] Stage: {label}")
    print(f"[pipeline] Command: {' '.join(cmd)}")
    if stage_timeout_s is not None:
        print(f"[pipeline] Timeout: {stage_timeout_s:.0f}s")
    print(f"{'=' * 60}", flush=True)

    stop_evt = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(label, stage_t0, stop_evt), daemon=True)
    hb.start()

    timed_out = False
    try:
        proc = subprocess.Popen(cmd)
        try:
            proc.wait(timeout=stage_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
    finally:
        stop_evt.set()
        hb.join(timeout=2)

    returncode = proc.returncode
    elapsed = time.time() - stage_t0
    mins, secs = divmod(int(elapsed), 60)

    stage_entry = {
        "stage": label,
        "script": script,
        "command": cmd,
        "started_utc": stage_started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
    }
    timeline["stages"].append(stage_entry)
    _write_timeline(out_root, run_id, timeline)

    if timed_out:
        print(
            f"[pipeline] TIMEOUT: {label} killed after {mins:02d}:{secs:02d} "
            f"(limit={stage_timeout_s:.0f}s)",
            file=sys.stderr, flush=True,
        )
    elif returncode != 0:
        print(
            f"[pipeline] ABORT: {label} failed (exit={returncode}) after {mins:02d}:{secs:02d}",
            file=sys.stderr, flush=True,
        )
    else:
        print(f"[pipeline] OK: {label} completed in {mins:02d}:{secs:02d}", flush=True)

    return returncode if not timed_out else 124  # 124 = timeout convention


def _run_optional_experiment_t0_sweep(
    out_root: Path,
    run_id: str,
    timeline: dict[str, Any],
    stage_timeout_s: float | None,
) -> None:
    label = "experiment_t0_sweep"
    script = "mvp/experiment_t0_sweep.py"
    script_path = MVP_DIR / "experiment_t0_sweep.py"

    stage_started = datetime.now(timezone.utc).isoformat()

    if not script_path.exists():
        timeline["stages"].append({
            "stage": label,
            "label": label,
            "script": script,
            "started_utc": stage_started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": 0.0,
            "returncode": None,
            "timed_out": False,
            "status": "SKIPPED",
            "message": "missing script",
            "best_effort": True,
        })
        _write_timeline(out_root, run_id, timeline)
        print("[pipeline] WARNING: experiment_t0_sweep.py not found, skipping best-effort experiment", flush=True)
        return

    cmd = [sys.executable, str(script_path), "--run", run_id]
    stage_t0 = time.time()
    status = "OK"
    message = "completed"
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=stage_timeout_s,
            check=False,
        )
        rc = proc.returncode
        if rc != 0:
            status = "FAILED"
            stderr = (proc.stderr or "").strip()
            if len(stderr) > 240:
                stderr = stderr[:240] + "..."
            message = f"exit={rc}: {stderr}" if stderr else f"exit={rc}"
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        status = "FAILED"
        stderr = (exc.stderr or "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr = stderr.strip()
        if len(stderr) > 240:
            stderr = stderr[:240] + "..."
        message = f"timeout after {stage_timeout_s:.0f}s"
        if stderr:
            message = f"{message}: {stderr}"

    timeline["stages"].append({
        "stage": label,
        "label": label,
        "script": script,
        "started_utc": stage_started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": float(time.time() - stage_t0),
        "returncode": rc,
        "timed_out": timed_out,
        "status": status,
        "message": message,
        "best_effort": True,
    })
    _write_timeline(out_root, run_id, timeline)

    if status != "OK":
        print(f"[pipeline] WARNING: {label} failed (exit={rc}), continuing", flush=True)


def _parse_multimode_results(out_root: Path, run_id: str) -> dict[str, Any]:
    results: dict[str, Any] = {
        "kerr_consistent": None,
        "chi_best": None,
        "d2_min": None,
        "extraction_quality": None,
    }

    s4c_path = out_root / run_id / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json"
    try:
        payload = json.loads(s4c_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[pipeline] WARNING: cannot parse {s4c_path}: {exc}", flush=True)
    else:
        for key in ("consistent_kerr_95", "kerr_consistent", "consistent"):
            if key in payload:
                results["kerr_consistent"] = payload[key]
                break
        for key in ("chi_best_fit", "chi_best"):
            if key in payload:
                results["chi_best"] = payload[key]
                break
        if "d2_min" in payload:
            results["d2_min"] = payload["d2_min"]

    s3b_path = out_root / run_id / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    try:
        payload_s3b = json.loads(s3b_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[pipeline] WARNING: cannot parse {s3b_path}: {exc}", flush=True)
    else:
        if isinstance(payload_s3b.get("results"), dict):
            results["extraction_quality"] = payload_s3b["results"].get("verdict")
        elif payload_s3b.get("extraction_quality") is not None:
            results["extraction_quality"] = payload_s3b.get("extraction_quality")

    return results


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
    stage_timeout_s: float | None = None,
    reuse_strain: bool = False,
    with_t0_sweep: bool = False,
    local_hdf5: list[str] | None = None,
    offline: bool = False,
) -> tuple[int, str]:
    """Run full pipeline for a single event. Returns (exit_code, run_id)."""
    out_root = resolve_out_root("runs")

    if run_id is None:
        run_id = _generate_run_id(event_id)

    print(f"\n[pipeline] Starting single-event pipeline")
    print(f"[pipeline] event_id={event_id}, run_id={run_id}")
    print(f"[pipeline] atlas={atlas_path}, synthetic={synthetic}")

    _create_run_valid(out_root, run_id)

    timeline: dict[str, Any] = {
        "schema_version": "mvp_pipeline_timeline_v1",
        "run_id": run_id,
        "mode": "single",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "ended_utc": None,
        "event_id": event_id,
        "atlas_path": atlas_path,
        "synthetic": synthetic,
        "stages": [],
    }
    _write_timeline(out_root, run_id, timeline)

    # Stage 1: Fetch strain
    s1_args = ["--run", run_id, "--event-id", event_id, "--duration-s", str(duration_s)]
    if synthetic:
        s1_args.append("--synthetic")
    if reuse_strain:
        s1_args.append("--reuse-if-present")
    for mapping in (local_hdf5 or []):
        s1_args.extend(["--local-hdf5", mapping])
    if offline:
        s1_args.append("--offline")
    rc = _run_stage("s1_fetch_strain.py", s1_args, "s1_fetch_strain", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    # Stage 2: Ringdown window
    s2_args = [
        "--run", run_id, "--event-id", event_id,
        "--dt-start-s", str(dt_start_s),
        "--duration-s", str(window_duration_s),
    ]
    rc = _run_stage("s2_ringdown_window.py", s2_args, "s2_ringdown_window", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    # Stage 3: Ringdown estimates
    s3_args = ["--run", run_id, "--band-low", str(band_low), "--band-high", str(band_high)]
    rc = _run_stage("s3_ringdown_estimates.py", s3_args, "s3_ringdown_estimates", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    if with_t0_sweep:
        _run_optional_experiment_t0_sweep(out_root, run_id, timeline, stage_timeout_s)

    # Stage 4: Geometry filter
    s4_args = ["--run", run_id, "--atlas-path", atlas_path, "--epsilon", str(epsilon)]
    rc = _run_stage("s4_geometry_filter.py", s4_args, "s4_geometry_filter", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
    _write_timeline(out_root, run_id, timeline)

    print(f"\n[pipeline] Single-event pipeline COMPLETE: run_id={run_id}")
    return 0, run_id


def run_multimode_event(
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
    stage_timeout_s: float | None = None,
    reuse_strain: bool = False,
    with_t0_sweep: bool = False,
    local_hdf5: list[str] | None = None,
    s3b_n_bootstrap: int = 200,
    s3b_seed: int = 12345,
    offline: bool = False,
) -> tuple[int, str]:
    out_root = resolve_out_root("runs")

    if run_id is None:
        run_id = _generate_run_id(event_id)

    print("\n[pipeline] Starting multimode pipeline")
    print(f"[pipeline] event_id={event_id}, run_id={run_id}")
    print(f"[pipeline] atlas={atlas_path}, synthetic={synthetic}")

    _create_run_valid(out_root, run_id)
    timeline: dict[str, Any] = {
        "schema_version": "mvp_pipeline_timeline_v1",
        "run_id": run_id,
        "mode": "multimode",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "ended_utc": None,
        "event_id": event_id,
        "atlas_path": atlas_path,
        "synthetic": synthetic,
        "stages": [],
        "multimode_results": {
            "kerr_consistent": None,
            "chi_best": None,
            "d2_min": None,
            "extraction_quality": None,
        },
    }
    _write_timeline(out_root, run_id, timeline)

    s1_args = ["--run", run_id, "--event-id", event_id, "--duration-s", str(duration_s)]
    if synthetic:
        s1_args.append("--synthetic")
    if reuse_strain:
        s1_args.append("--reuse-if-present")
    for mapping in (local_hdf5 or []):
        s1_args.extend(["--local-hdf5", mapping])
    if offline:
        s1_args.append("--offline")
    rc = _run_stage("s1_fetch_strain.py", s1_args, "s1_fetch_strain", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s2_args = [
        "--run", run_id, "--event-id", event_id,
        "--dt-start-s", str(dt_start_s),
        "--duration-s", str(window_duration_s),
    ]
    rc = _run_stage("s2_ringdown_window.py", s2_args, "s2_ringdown_window", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s3_args = ["--run", run_id, "--band-low", str(band_low), "--band-high", str(band_high)]
    rc = _run_stage("s3_ringdown_estimates.py", s3_args, "s3_ringdown_estimates", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    if with_t0_sweep:
        _run_optional_experiment_t0_sweep(out_root, run_id, timeline, stage_timeout_s)

    s3b_args = [
        "--run-id", run_id,
        "--s3-estimates", f"{run_id}/s3_ringdown_estimates/outputs/estimates.json",
        "--n-bootstrap", str(s3b_n_bootstrap),
        "--seed", str(s3b_seed),
    ]
    rc = _run_stage("s3b_multimode_estimates.py", s3b_args, "s3b_multimode_estimates", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s4_args = ["--run", run_id, "--atlas-path", atlas_path, "--epsilon", str(epsilon)]
    rc = _run_stage("s4_geometry_filter.py", s4_args, "s4_geometry_filter", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    s4c_args = ["--run-id", run_id, "--atlas-path", atlas_path]
    rc = _run_stage("s4c_kerr_consistency.py", s4c_args, "s4c_kerr_consistency", out_root, run_id, timeline, stage_timeout_s)
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, run_id, timeline)
        return rc, run_id

    timeline["multimode_results"] = _parse_multimode_results(out_root, run_id)
    timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
    _write_timeline(out_root, run_id, timeline)
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

    out_root = resolve_out_root("runs")

    print(f"\n[pipeline] Starting multi-event pipeline")
    print(f"[pipeline] events={events}")
    print(f"[pipeline] aggregate_run={agg_run_id}")

    _create_run_valid(out_root, agg_run_id)

    timeline: dict[str, Any] = {
        "schema_version": "mvp_pipeline_timeline_v1",
        "run_id": agg_run_id,
        "mode": "multi",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "ended_utc": None,
        "events": events,
        "atlas_path": atlas_path,
        "synthetic": bool(kwargs.get("synthetic", False)),
        "stages": [],
    }
    _write_timeline(out_root, agg_run_id, timeline)

    per_event_runs: list[str] = []

    for event_id in events:
        rc, run_id = run_single_event(event_id=event_id, atlas_path=atlas_path, **kwargs)
        if rc != 0:
            print(f"[pipeline] ABORT: event {event_id} failed", file=sys.stderr)
            timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_timeline(out_root, agg_run_id, timeline)
            return rc, agg_run_id
        per_event_runs.append(run_id)

    # Stage 5: Aggregate
    s5_args = [
        "--out-run", agg_run_id,
        "--source-runs", ",".join(per_event_runs),
        "--min-coverage", str(min_coverage),
    ]
    rc = _run_stage("s5_aggregate.py", s5_args, "s5_aggregate", out_root, agg_run_id, timeline, kwargs.get("stage_timeout_s"))
    if rc != 0:
        timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_timeline(out_root, agg_run_id, timeline)
        return rc, agg_run_id

    timeline["ended_utc"] = datetime.now(timezone.utc).isoformat()
    _write_timeline(out_root, agg_run_id, timeline)

    print(f"\n[pipeline] Multi-event pipeline COMPLETE: agg_run={agg_run_id}")
    print(f"[pipeline] Per-event runs: {per_event_runs}")
    return 0, agg_run_id


def _resolve_atlas_path(atlas_path: str | None, atlas_default: bool) -> str:
    if atlas_path:
        return atlas_path

    if atlas_default:
        if not DEFAULT_ATLAS_PATH.exists():
            print(
                "Atlas not found. Generate it by running: "
                "python mvp/generate_atlas_from_fits.py",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return str(DEFAULT_ATLAS_PATH)

    raise SystemExit("--atlas-path is required unless --atlas-default is set")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MVP Pipeline Orchestrator")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Single event
    sp_single = sub.add_parser("single", help="Run pipeline for one event")
    sp_single.add_argument("--event-id", required=True)
    sp_single.add_argument("--atlas-path", default=None)
    sp_single.add_argument("--atlas-default", action="store_true", default=False)
    sp_single.add_argument("--run-id", default=None)
    sp_single.add_argument("--synthetic", action="store_true")
    sp_single.add_argument("--duration-s", type=float, default=32.0)
    sp_single.add_argument("--dt-start-s", type=float, default=0.003)
    sp_single.add_argument("--window-duration-s", type=float, default=0.06)
    sp_single.add_argument("--band-low", type=float, default=150.0)
    sp_single.add_argument("--band-high", type=float, default=400.0)
    sp_single.add_argument("--epsilon", type=float, default=0.3)
    sp_single.add_argument(
        "--stage-timeout-s", type=float, default=None,
        help="Kill a stage if it exceeds this many seconds (default: no limit)",
    )
    sp_single.add_argument(
        "--reuse-strain", action="store_true", default=False,
        help="Skip s1 download if outputs already exist and params match",
    )
    sp_single.add_argument("--with-t0-sweep", action="store_true", default=False)
    sp_single.add_argument(
        "--local-hdf5",
        action="append",
        default=[],
        metavar="DET=PATH",
        help="Forward local HDF5 detector mapping(s) to s1_fetch_strain (repeatable)",
    )
    sp_single.add_argument("--offline", action="store_true", default=False)

    # Multi event
    sp_multi = sub.add_parser("multi", help="Run pipeline for multiple events + aggregate")
    sp_multi.add_argument("--events", required=True, help="Comma-separated event IDs")
    sp_multi.add_argument("--atlas-path", default=None)
    sp_multi.add_argument("--atlas-default", action="store_true", default=False)
    sp_multi.add_argument("--agg-run-id", default=None)
    sp_multi.add_argument("--min-coverage", type=float, default=1.0)
    sp_multi.add_argument("--synthetic", action="store_true")
    sp_multi.add_argument("--duration-s", type=float, default=32.0)
    sp_multi.add_argument("--dt-start-s", type=float, default=0.003)
    sp_multi.add_argument("--window-duration-s", type=float, default=0.06)
    sp_multi.add_argument("--band-low", type=float, default=150.0)
    sp_multi.add_argument("--band-high", type=float, default=400.0)
    sp_multi.add_argument("--epsilon", type=float, default=0.3)
    sp_multi.add_argument(
        "--stage-timeout-s", type=float, default=None,
        help="Kill a stage if it exceeds this many seconds (default: no limit)",
    )
    sp_multi.add_argument(
        "--reuse-strain", action="store_true", default=False,
        help="Skip s1 download if outputs already exist and params match",
    )
    sp_multi.add_argument("--with-t0-sweep", action="store_true", default=False)
    sp_multi.add_argument(
        "--local-hdf5",
        action="append",
        default=[],
        metavar="DET=PATH",
        help="Forward local HDF5 detector mapping(s) to per-event s1_fetch_strain (repeatable)",
    )
    sp_multi.add_argument("--offline", action="store_true", default=False)

    # Single event multimode
    sp_multimode = sub.add_parser("multimode", help="Run single-event multimode pipeline")
    sp_multimode.add_argument("--event-id", required=True)
    sp_multimode.add_argument("--atlas-path", default=None)
    sp_multimode.add_argument("--atlas-default", action="store_true", default=False)
    sp_multimode.add_argument("--run-id", default=None)
    sp_multimode.add_argument("--synthetic", action="store_true")
    sp_multimode.add_argument("--duration-s", type=float, default=32.0)
    sp_multimode.add_argument("--dt-start-s", type=float, default=0.003)
    sp_multimode.add_argument("--window-duration-s", type=float, default=0.06)
    sp_multimode.add_argument("--band-low", type=float, default=150.0)
    sp_multimode.add_argument("--band-high", type=float, default=400.0)
    sp_multimode.add_argument("--epsilon", type=float, default=0.3)
    sp_multimode.add_argument(
        "--stage-timeout-s", type=float, default=None,
        help="Kill a stage if it exceeds this many seconds (default: no limit)",
    )
    sp_multimode.add_argument(
        "--reuse-strain", action="store_true", default=False,
        help="Skip s1 download if outputs already exist and params match",
    )
    sp_multimode.add_argument("--with-t0-sweep", action="store_true", default=False)
    sp_multimode.add_argument("--s3b-n-bootstrap", type=int, default=200)
    sp_multimode.add_argument("--s3b-seed", type=int, default=12345)
    sp_multimode.add_argument(
        "--local-hdf5",
        action="append",
        default=[],
        metavar="DET=PATH",
        help="Forward local HDF5 detector mapping(s) to s1_fetch_strain (repeatable)",
    )
    sp_multimode.add_argument("--offline", action="store_true", default=False)

    args = parser.parse_args()

    if args.mode == "single":
        atlas_path = _resolve_atlas_path(args.atlas_path, args.atlas_default)
        rc, run_id = run_single_event(
            event_id=args.event_id,
            atlas_path=atlas_path,
            run_id=args.run_id,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
            stage_timeout_s=args.stage_timeout_s,
            reuse_strain=args.reuse_strain,
            with_t0_sweep=args.with_t0_sweep,
            local_hdf5=args.local_hdf5,
            offline=args.offline,
        )
        return rc

    elif args.mode == "multi":
        atlas_path = _resolve_atlas_path(args.atlas_path, args.atlas_default)
        events = [e.strip() for e in args.events.split(",") if e.strip()]
        rc, agg_run_id = run_multi_event(
            events=events,
            atlas_path=atlas_path,
            agg_run_id=args.agg_run_id,
            min_coverage=args.min_coverage,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
            stage_timeout_s=args.stage_timeout_s,
            reuse_strain=args.reuse_strain,
            with_t0_sweep=args.with_t0_sweep,
            local_hdf5=args.local_hdf5,
            offline=args.offline,
        )
        return rc

    elif args.mode == "multimode":
        atlas_path = _resolve_atlas_path(args.atlas_path, args.atlas_default)
        rc, run_id = run_multimode_event(
            event_id=args.event_id,
            atlas_path=atlas_path,
            run_id=args.run_id,
            synthetic=args.synthetic,
            duration_s=args.duration_s,
            dt_start_s=args.dt_start_s,
            window_duration_s=args.window_duration_s,
            band_low=args.band_low,
            band_high=args.band_high,
            epsilon=args.epsilon,
            stage_timeout_s=args.stage_timeout_s,
            reuse_strain=args.reuse_strain,
            with_t0_sweep=args.with_t0_sweep,
            s3b_n_bootstrap=args.s3b_n_bootstrap,
            s3b_seed=args.s3b_seed,
            local_hdf5=args.local_hdf5,
            offline=args.offline,
        )
        return rc

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
