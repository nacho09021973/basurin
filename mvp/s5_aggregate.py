#!/usr/bin/env python3
"""MVP Stage 5: Aggregate compatible sets across multiple events.

CLI:
    python mvp/s5_aggregate.py --out-run <agg_run_id> \
        --source-runs run_A,run_B [--min-coverage 1.0]

Inputs:  runs/<run>/s4_geometry_filter/outputs/compatible_set.json (per source run)
Outputs: runs/<agg_run>/s5_aggregate/outputs/aggregate.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.contracts import init_stage, check_inputs, finalize, abort
from basurin_io import sha256_file, utc_now_iso, write_json_atomic

STAGE = "s5_aggregate"


def aggregate_compatible_sets(
    source_data: list[dict[str, Any]], min_coverage: float = 1.0,
) -> dict[str, Any]:
    n_events = len(source_data)
    if n_events == 0:
        return {"n_events": 0, "common_geometries": [], "coverage_histogram": {}}

    counter: Counter[str] = Counter()
    distances: dict[str, list[float]] = {}
    metadata: dict[str, Any] = {}

    for src in source_data:
        for geo in src.get("compatible_geometries", []):
            gid = geo["geometry_id"]
            counter[gid] += 1
            distances.setdefault(gid, []).append(geo.get("distance", float("nan")))
            if gid not in metadata and geo.get("metadata"):
                metadata[gid] = geo["metadata"]

    min_count = max(1, int(math.ceil(min_coverage * n_events)))
    common = [
        {
            "geometry_id": gid, "n_events": count,
            "coverage": count / n_events,
            "mean_distance": float(sum(d for d in distances[gid] if math.isfinite(d)) / max(1, len(distances[gid]))),
            "distances_per_event": distances[gid],
            "metadata": metadata.get(gid),
        }
        for gid, count in counter.most_common() if count >= min_count
    ]

    coverage_hist: dict[str, int] = {}
    for c in counter.values():
        coverage_hist[str(c)] = coverage_hist.get(str(c), 0) + 1

    return {
        "schema_version": "mvp_aggregate_v1",
        "n_events": n_events, "min_coverage": min_coverage, "min_count": min_count,
        "n_total_unique_geometries": len(counter),
        "n_common_geometries": len(common), "common_geometries": common,
        "coverage_histogram": coverage_hist,
        "events": [{"run_id": s["run_id"], "event_id": s["event_id"],
                     "n_compatible": len(s.get("compatible_geometries", []))} for s in source_data],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: aggregate events")
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--source-runs", required=True)
    ap.add_argument("--min-coverage", type=float, default=1.0)
    args = ap.parse_args()

    source_runs = [r.strip() for r in args.source_runs.split(",") if r.strip()]
    if not source_runs:
        print("ERROR: --source-runs empty", file=sys.stderr)
        raise SystemExit(2)

    # Create RUN_VALID for aggregate run
    from basurin_io import resolve_out_root
    out_root = resolve_out_root("runs")
    rv_dir = out_root / args.out_run / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_dir / "verdict.json", {"verdict": "PASS", "created": utc_now_iso()})

    ctx = init_stage(args.out_run, STAGE, params={
        "source_runs": source_runs, "min_coverage": args.min_coverage,
    })

    # Collect and validate source compatible_set.json files
    source_paths: dict[str, Path] = {}
    for src in source_runs:
        p = out_root / src / "s4_geometry_filter" / "outputs" / "compatible_set.json"
        source_paths[src] = p
    check_inputs(ctx, source_paths)

    try:
        source_data: list[dict[str, Any]] = []
        for src, p in source_paths.items():
            with open(p, "r", encoding="utf-8") as f:
                cs = json.load(f)
            source_data.append({
                "run_id": src, "event_id": cs.get("event_id", "unknown"),
                "compatible_geometries": cs.get("compatible_geometries", []),
                "observables": cs.get("observables", {}),
            })

        result = aggregate_compatible_sets(source_data, args.min_coverage)
        agg_path = ctx.outputs_dir / "aggregate.json"
        write_json_atomic(agg_path, result)

        finalize(ctx, artifacts={"aggregate": agg_path}, results={
            "n_events": result["n_events"],
            "n_common": result["n_common_geometries"],
            "n_unique": result["n_total_unique_geometries"],
        })
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
