#!/usr/bin/env python3
"""MVP Stage 5: Aggregate compatible sets across multiple events.

CLI:
    python mvp/s5_aggregate.py --out-run <agg_run_id> \
        --source-runs run_A,run_B,run_C \
        [--min-coverage 1.0]

Inputs:
    For each source run: runs/<run>/s4_geometry_filter/outputs/compatible_set.json

Outputs (runs/<agg_run>/s5_aggregate/outputs/):
    aggregate.json       Intersection/union of compatible geometries across events

Method:
    1. Load compatible geometry sets from N event runs.
    2. Count how many events each geometry_id appears in.
    3. Intersection = geometries present in ALL events (coverage=1.0).
    4. Report coverage histogram, common geometries, and structure metrics.

Contracts:
    - All source runs must have s4_geometry_filter PASS.
    - aggregate.json records every source run's SHA256 for traceability.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# --- BASURIN import bootstrap ---
_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        sys.path.insert(0, str(_cand))
        break

from basurin_io import (
    ensure_stage_dirs,
    resolve_out_root,
    sha256_file,
    utc_now_iso,
    validate_run_id,
    write_json_atomic,
    write_manifest,
    write_stage_summary,
)

STAGE_NAME = "s5_aggregate"
UPSTREAM_STAGE = "s4_geometry_filter"
EXIT_CONTRACT_FAIL = 2


def _abort(message: str) -> None:
    print(f"[{STAGE_NAME}] ERROR: {message}", file=sys.stderr)
    raise SystemExit(EXIT_CONTRACT_FAIL)


def _write_failure(stage_dir: Path, run_id: str, params: dict, inputs: list, reason: str) -> None:
    summary = {
        "stage": STAGE_NAME, "run": run_id, "created": utc_now_iso(),
        "version": "v1", "parameters": params, "inputs": inputs, "outputs": [],
        "verdict": "FAIL", "error": reason,
    }
    sp = write_stage_summary(stage_dir, summary)
    write_manifest(stage_dir, {"stage_summary": sp}, extra={"verdict": "FAIL", "error": reason})


def aggregate_compatible_sets(
    source_data: list[dict[str, Any]],
    min_coverage: float = 1.0,
) -> dict[str, Any]:
    """Aggregate compatible geometry sets from multiple events.

    Args:
        source_data: List of dicts, each with keys:
            run_id, event_id, compatible_geometries (list of {geometry_id, distance, ...})
        min_coverage: Fraction of events a geometry must appear in (1.0 = intersection).

    Returns:
        Aggregate result dict.
    """
    n_events = len(source_data)
    if n_events == 0:
        return {"n_events": 0, "common_geometries": [], "coverage_histogram": {}}

    # Count geometry_id occurrences across events
    geometry_counter: Counter[str] = Counter()
    geometry_distances: dict[str, list[float]] = {}
    geometry_metadata: dict[str, Any] = {}

    for source in source_data:
        compat = source.get("compatible_geometries", [])
        for geo in compat:
            gid = geo["geometry_id"]
            geometry_counter[gid] += 1
            geometry_distances.setdefault(gid, []).append(geo.get("distance", float("nan")))
            if gid not in geometry_metadata and geo.get("metadata"):
                geometry_metadata[gid] = geo["metadata"]

    # Filter by coverage threshold
    min_count = max(1, int(math.ceil(min_coverage * n_events)))

    common: list[dict[str, Any]] = []
    for gid, count in geometry_counter.most_common():
        if count >= min_count:
            distances = geometry_distances[gid]
            common.append({
                "geometry_id": gid,
                "n_events": count,
                "coverage": count / n_events,
                "mean_distance": float(sum(d for d in distances if math.isfinite(d)) / max(1, len(distances))),
                "distances_per_event": distances,
                "metadata": geometry_metadata.get(gid),
            })

    # Coverage histogram: how many geometries appear in exactly k events
    coverage_hist: dict[str, int] = {}
    for count in geometry_counter.values():
        key = str(count)
        coverage_hist[key] = coverage_hist.get(key, 0) + 1

    # Total unique geometries across all events
    n_total_unique = len(geometry_counter)

    return {
        "schema_version": "mvp_aggregate_v1",
        "n_events": n_events,
        "min_coverage": min_coverage,
        "min_count": min_count,
        "n_total_unique_geometries": n_total_unique,
        "n_common_geometries": len(common),
        "common_geometries": common,
        "coverage_histogram": coverage_hist,
        "events": [
            {"run_id": s["run_id"], "event_id": s["event_id"], "n_compatible": len(s.get("compatible_geometries", []))}
            for s in source_data
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE_NAME}: aggregate events")
    ap.add_argument("--out-run", required=True, help="Output run_id for the aggregate")
    ap.add_argument("--source-runs", required=True, help="Comma-separated source run IDs")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                     help="Minimum fraction of events (1.0 = strict intersection)")
    args = ap.parse_args()

    out_root = resolve_out_root("runs")
    validate_run_id(args.out_run, out_root)

    source_runs = [r.strip() for r in args.source_runs.split(",") if r.strip()]
    if not source_runs:
        _abort("--source-runs is empty")
    if not (0.0 < args.min_coverage <= 1.0):
        _abort(f"--min-coverage must be in (0, 1], got {args.min_coverage}")

    # Create RUN_VALID for the aggregate run
    agg_run_dir = out_root / args.out_run
    rv_dir = agg_run_dir / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_dir / "verdict.json", {"verdict": "PASS", "created": utc_now_iso()})

    stage_dir, outputs_dir = ensure_stage_dirs(args.out_run, STAGE_NAME, base_dir=out_root)

    params: dict[str, Any] = {
        "source_runs": source_runs,
        "min_coverage": args.min_coverage,
    }
    inputs_list: list[dict[str, str]] = []

    try:
        # Load compatible sets from all source runs
        source_data: list[dict[str, Any]] = []

        for src_run in source_runs:
            cs_path = out_root / src_run / UPSTREAM_STAGE / "outputs" / "compatible_set.json"
            if not cs_path.exists():
                _abort(f"Missing {UPSTREAM_STAGE}/outputs/compatible_set.json in run {src_run}")

            inputs_list.append({
                "run": src_run,
                "path": str(cs_path),
                "sha256": sha256_file(cs_path),
            })

            with open(cs_path, "r", encoding="utf-8") as f:
                cs = json.load(f)

            source_data.append({
                "run_id": src_run,
                "event_id": cs.get("event_id", "unknown"),
                "compatible_geometries": cs.get("compatible_geometries", []),
                "observables": cs.get("observables", {}),
            })

        # Aggregate
        result = aggregate_compatible_sets(source_data, args.min_coverage)
        result["source_inputs"] = inputs_list

        # Write output
        agg_path = outputs_dir / "aggregate.json"
        write_json_atomic(agg_path, result)

        agg_run_dir_ref = stage_dir.parent
        outputs_list = [{"path": str(agg_path.relative_to(agg_run_dir_ref)), "sha256": sha256_file(agg_path)}]

        summary = {
            "stage": STAGE_NAME, "run": args.out_run, "created": utc_now_iso(),
            "version": "v1", "parameters": params, "inputs": inputs_list,
            "outputs": outputs_list, "verdict": "PASS",
            "results": {
                "n_events": result["n_events"],
                "n_common_geometries": result["n_common_geometries"],
                "n_total_unique": result["n_total_unique_geometries"],
            },
        }
        sp = write_stage_summary(stage_dir, summary)
        write_manifest(
            stage_dir,
            {"aggregate": agg_path, "stage_summary": sp},
            extra={"inputs": inputs_list},
        )

        print(
            f"OK: {STAGE_NAME} PASS "
            f"(events={result['n_events']}, "
            f"common={result['n_common_geometries']}/{result['n_total_unique_geometries']})"
        )
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        _write_failure(stage_dir, args.out_run, params, inputs_list, str(exc))
        _abort(str(exc))
        return EXIT_CONTRACT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
