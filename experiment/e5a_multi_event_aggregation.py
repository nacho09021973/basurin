#!/usr/bin/env python3
"""E5-A — Multi-Event Aggregation (intersection, union, Jaccard).

Computes intersection, union, and Jaccard similarity of compatible geometry
sets across N events.  First empirical step toward population exclusion:
which geometries survive in ALL events?

Governance:
  - Read-only on compatible_set.json per run (all RUN_VALID=PASS).
  - Writes only under runs/<anchor>/experiment/multi_event_aggregation/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiment.base_contract import (
    REQUIRED_CANONICAL_GATES,
    _write_json_atomic,
    ensure_experiment_dir,
    load_json,
    sha256_file,
    validate_and_load_run,
    write_manifest,
)

SCHEMA_VERSION = "e5a-0.1"
EXPERIMENT_NAME = "multi_event_aggregation"


def _extract_geometry_ids(compatible_set: Any) -> set[str]:
    """Extract geometry_id set from compatible_set.json (handles list or dict)."""
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
    """Heuristic: family is the prefix before the last underscore+digits."""
    parts = geometry_id.rsplit("_", 1)
    return parts[0] if len(parts) > 1 and parts[1].isdigit() else geometry_id


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity coefficient."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def aggregate_events(
    run_ids: list[str],
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute intersection/union/Jaccard across N events."""
    if len(run_ids) < 2:
        raise ValueError("E5-A requires at least 2 runs")

    per_run: dict[str, set[str]] = {}
    input_hashes: dict[str, str] = {}

    for run_id in sorted(run_ids):
        run_dir, _ = validate_and_load_run(run_id, runs_root)
        cs_path = run_dir / REQUIRED_CANONICAL_GATES["compatible_set"]
        if not cs_path.exists():
            raise FileNotFoundError(f"compatible_set.json missing: {cs_path}")

        input_hashes[run_id] = sha256_file(cs_path)
        cs = load_json(cs_path)
        per_run[run_id] = _extract_geometry_ids(cs)

    # Global intersection and union
    all_sets = list(per_run.values())
    intersection = set.intersection(*all_sets)
    union = set.union(*all_sets)

    # Frequency: how many events each geometry appears in
    frequency: Counter = Counter()
    for gids in all_sets:
        frequency.update(gids)

    # Jaccard pairwise matrix
    sorted_ids = sorted(run_ids)
    jaccard_matrix: dict[str, dict[str, float]] = {}
    for i, rid_a in enumerate(sorted_ids):
        jaccard_matrix[rid_a] = {}
        for rid_b in sorted_ids:
            jaccard_matrix[rid_a][rid_b] = round(
                _jaccard(per_run[rid_a], per_run[rid_b]), 6
            )

    # Persistence by family
    family_union: dict[str, set[str]] = defaultdict(set)
    family_intersection: dict[str, set[str]] = defaultdict(set)
    for gid in union:
        fam = _extract_family(gid)
        family_union[fam].add(gid)
    for gid in intersection:
        fam = _extract_family(gid)
        family_intersection[fam].add(gid)

    persistence_by_family = {}
    for fam in sorted(family_union):
        persistence_by_family[fam] = {
            "n_in_union": len(family_union[fam]),
            "n_in_intersection": len(family_intersection.get(fam, set())),
        }

    # Persistence histogram: how many geometries appear in exactly k events
    hist: dict[int, int] = defaultdict(int)
    for count in frequency.values():
        hist[count] += 1
    persistence_histogram = {str(k): v for k, v in sorted(hist.items())}

    jaccard_global = _jaccard(intersection, union) if union else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "run_ids_consumed": sorted_ids,
        "input_hashes": input_hashes,
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard_global": round(jaccard_global, 6),
        "intersection_geometry_ids": sorted(intersection),
        "persistence_by_family": persistence_by_family,
        "persistence_histogram": persistence_histogram,
        "jaccard_matrix": jaccard_matrix,
        "frequency_per_geometry": {gid: count for gid, count in frequency.most_common()},
    }


def run_aggregation(
    run_ids: list[str],
    anchor_run_id: str | None = None,
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full aggregation: validate, compute, write."""
    result = aggregate_events(run_ids, runs_root)

    if dry_run:
        print(json.dumps(result, indent=2))
        return result

    anchor = anchor_run_id or sorted(run_ids)[0]
    run_dir, _ = validate_and_load_run(anchor, runs_root)
    out_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    _write_json_atomic(out_dir / "aggregation_result.json", result)
    _write_json_atomic(out_dir / "jaccard_matrix.json", result["jaccard_matrix"])
    _write_json_atomic(out_dir / "persistence_histogram.json", result["persistence_histogram"])
    write_manifest(out_dir, result["input_hashes"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E5-A: Multi-event geometry aggregation")
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--anchor-run", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_aggregation(
        run_ids=args.run_ids,
        anchor_run_id=args.anchor_run,
        runs_root=args.runs_root,
        dry_run=args.dry_run,
    )
    print(f"Union: {result['union_count']}, Intersection: {result['intersection_count']}, Jaccard: {result['jaccard_global']}")


if __name__ == "__main__":
    main()
