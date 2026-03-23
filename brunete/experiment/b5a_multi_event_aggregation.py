#!/usr/bin/env python3
"""B5-A — Multi-Event Aggregation (BRUNETE port of E5-A).

Computes intersection, union, and Jaccard similarity of compatible geometry
sets across N events, navigating from a BRUNETE classify run to the per-event
BASURIN subruns.

Governance
----------
- Read-only on compatible_set.json per event subrun (all RUN_VALID=PASS).
- Writes only under runs/<classify_run_id>/experiment/multi_event_aggregation/.
- Requires at least 2 events with compatible_set.json present.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from brunete.experiment.base_contract import (
    EVENT_RUN_GATES,
    _write_json_atomic,
    ensure_experiment_dir,
    enumerate_event_runs,
    load_json,
    resolve_classify_run_dir,
    sha256_file,
    validate_classify_run,
    write_manifest,
)

SCHEMA_VERSION = "b5a-0.1"
EXPERIMENT_NAME = "multi_event_aggregation"


def _extract_geometry_ids(compatible_set: Any) -> set[str]:
    if isinstance(compatible_set, list):
        geometries = compatible_set
    else:
        geometries = compatible_set.get("geometries", compatible_set.get("compatible", []))
    ids = set()
    for g in geometries:
        gid = g.get("geometry_id", g.get("id"))
        if gid:
            ids.add(str(gid))
    return ids


def _extract_family(geometry_id: str) -> str:
    parts = geometry_id.rsplit("_", 1)
    return parts[0] if len(parts) > 1 and parts[1].isdigit() else geometry_id


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def aggregate_events(
    classify_run_id: str,
    mode: str = "220",
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute intersection/union/Jaccard across all events in a classify run."""
    event_run_map = enumerate_event_runs(classify_run_id, mode=mode, runs_root=runs_root)

    if len(event_run_map) < 2:
        raise ValueError(
            f"B5-A requires at least 2 valid event runs for mode {mode!r}, "
            f"got {len(event_run_map)}"
        )

    per_event: dict[str, set[str]] = {}
    input_hashes: dict[str, str] = {}

    for event_id, event_run_dir in sorted(event_run_map.items()):
        cs_path = event_run_dir / EVENT_RUN_GATES["compatible_set"]
        if not cs_path.exists():
            continue
        input_hashes[event_id] = sha256_file(cs_path)
        per_event[event_id] = _extract_geometry_ids(load_json(cs_path))

    if len(per_event) < 2:
        raise ValueError(
            f"B5-A requires at least 2 events with compatible_set.json, "
            f"got {len(per_event)}"
        )

    sorted_events = sorted(per_event)
    all_sets = [per_event[e] for e in sorted_events]

    intersection = set.intersection(*all_sets)
    union = set.union(*all_sets)

    # Frequency: how many events contain each geometry
    freq: Counter = Counter()
    for gid_set in all_sets:
        for gid in gid_set:
            freq[gid] += 1

    persistence = {
        str(k): v for k, v in sorted(freq.items(), key=lambda x: -x[1])
    }

    # Jaccard matrix (upper triangle)
    jaccard_matrix: list[dict] = []
    for i, eid_a in enumerate(sorted_events):
        for j, eid_b in enumerate(sorted_events):
            if j <= i:
                continue
            jaccard_matrix.append({
                "event_a": eid_a,
                "event_b": eid_b,
                "jaccard": round(_jaccard(per_event[eid_a], per_event[eid_b]), 4),
            })

    # Family-level persistence
    family_freq: Counter = Counter()
    for gid in union:
        fam = _extract_family(gid)
        family_freq[fam] += freq[gid]

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "classify_run_id": classify_run_id,
        "n_events": len(per_event),
        "events": sorted_events,
        "intersection": sorted(intersection),
        "n_intersection": len(intersection),
        "union": sorted(union),
        "n_union": len(union),
        "jaccard_global": round(
            len(intersection) / len(union) if union else 1.0, 4
        ),
        "jaccard_matrix": jaccard_matrix,
        "persistence_histogram": persistence,
        "family_persistence": {
            k: v for k, v in sorted(family_freq.items(), key=lambda x: -x[1])
        },
        "input_hashes": input_hashes,
    }


def run_b5a(
    classify_run_id: str,
    mode: str = "220",
    runs_root: str | Path | None = None,
) -> Path:
    run_dir = resolve_classify_run_dir(classify_run_id, runs_root)
    exp_dir = ensure_experiment_dir(run_dir, f"{EXPERIMENT_NAME}_{mode}")

    result = aggregate_events(classify_run_id, mode=mode, runs_root=runs_root)

    out_path = exp_dir / "aggregation_result.json"
    _write_json_atomic(out_path, result)
    _write_json_atomic(exp_dir / "jaccard_matrix.json", result["jaccard_matrix"])
    _write_json_atomic(exp_dir / "persistence_histogram.json", result["persistence_histogram"])
    write_manifest(exp_dir, result["input_hashes"],
                   extra={"schema_version": SCHEMA_VERSION, "mode": mode})
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="B5-A: intersection/union/Jaccard across events in a classify run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--classify-run", required=True)
    ap.add_argument("--mode", choices=["220", "221"], default="220")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = aggregate_events(args.classify_run, mode=args.mode, runs_root=args.runs_root)

    if args.dry_run:
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("intersection", "union", "persistence_histogram")},
                         indent=2))
        return 0

    out_path = run_b5a(args.classify_run, mode=args.mode, runs_root=args.runs_root)
    print(f"B5-A written: {out_path}")
    print(f"  mode          : {result['mode']}")
    print(f"  n_events      : {result['n_events']}")
    print(f"  n_intersection: {result['n_intersection']}")
    print(f"  n_union       : {result['n_union']}")
    print(f"  jaccard_global: {result['jaccard_global']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
