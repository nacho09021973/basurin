#!/usr/bin/env python3
"""E5-B — Jackknife Stability Audit (leave-one-out).

Measures how much the intersection/union changes when one event is removed.
Produces stability certificates per geometry before any population claim.

Governance:
  - Read-only on compatible_set.json + stage_summary.json (all RUN_VALID=PASS).
  - Writes only under runs/<anchor>/experiment/jackknife_stability/.
"""
from __future__ import annotations

import argparse
import json
import math
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
from experiment.e5a_multi_event_aggregation import _extract_geometry_ids

SCHEMA_VERSION = "e5b-0.1"
EXPERIMENT_NAME = "jackknife_stability"

# Stability thresholds
STABLE_VARIANCE_THRESHOLD = 0.05
UNSTABLE_VARIANCE_THRESHOLD = 0.20


def _classify_stability(variance: float, n_events: int) -> str:
    """Classify geometry stability based on jackknife variance."""
    if n_events <= 1:
        return "SINGLETON"
    if variance <= STABLE_VARIANCE_THRESHOLD:
        return "STABLE"
    if variance <= UNSTABLE_VARIANCE_THRESHOLD:
        return "MODERATE"
    return "UNSTABLE"


def jackknife_analysis(
    run_ids: list[str],
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Leave-one-out jackknife over N events."""
    if len(run_ids) < 3:
        raise ValueError("E5-B requires at least 3 runs for meaningful jackknife")

    # Load all compatible sets
    per_run: dict[str, set[str]] = {}
    input_hashes: dict[str, str] = {}

    for run_id in sorted(run_ids):
        run_dir, _ = validate_and_load_run(run_id, runs_root)
        cs_path = run_dir / REQUIRED_CANONICAL_GATES["compatible_set"]
        if not cs_path.exists():
            raise FileNotFoundError(f"compatible_set.json missing: {cs_path}")
        input_hashes[run_id] = sha256_file(cs_path)
        per_run[run_id] = _extract_geometry_ids(load_json(cs_path))

    sorted_ids = sorted(run_ids)
    n = len(sorted_ids)
    all_sets = [per_run[rid] for rid in sorted_ids]
    full_union = set.union(*all_sets)

    # For each geometry: compute presence vector and jackknife variance
    geometry_stability: dict[str, dict[str, Any]] = {}
    for gid in sorted(full_union):
        # Presence: 1 if geometry in event, 0 otherwise
        presence = [1.0 if gid in per_run[rid] else 0.0 for rid in sorted_ids]
        mean_presence = sum(presence) / n

        # Jackknife: leave-one-out means
        jack_means = []
        for i in range(n):
            loo = presence[:i] + presence[i + 1:]
            jack_means.append(sum(loo) / (n - 1))

        # Jackknife variance estimate
        jack_var = ((n - 1) / n) * sum((jm - mean_presence) ** 2 for jm in jack_means)

        geometry_stability[gid] = {
            "mean_presence": round(mean_presence, 4),
            "jackknife_variance": round(jack_var, 6),
            "certificate": _classify_stability(jack_var, n),
        }

    # Influence ranking: which event changes the intersection most when removed?
    full_intersection = set.intersection(*all_sets)
    influence: dict[str, int] = {}
    for i, rid in enumerate(sorted_ids):
        loo_sets = all_sets[:i] + all_sets[i + 1:]
        loo_intersection = set.intersection(*loo_sets)
        influence[rid] = len(loo_intersection) - len(full_intersection)

    # Sort by influence (largest change = most influential)
    high_influence = sorted(influence.keys(), key=lambda r: abs(influence[r]), reverse=True)

    # Summary counts
    n_stable = sum(1 for g in geometry_stability.values() if g["certificate"] == "STABLE")
    n_unstable = sum(1 for g in geometry_stability.values() if g["certificate"] == "UNSTABLE")
    n_moderate = sum(1 for g in geometry_stability.values() if g["certificate"] == "MODERATE")

    return {
        "schema_version": SCHEMA_VERSION,
        "n_events": n,
        "geometry_stability": geometry_stability,
        "high_influence_events": high_influence,
        "influence_delta": {rid: influence[rid] for rid in high_influence},
        "summary": {
            "n_geometries_total": len(full_union),
            "n_stable": n_stable,
            "n_moderate": n_moderate,
            "n_unstable": n_unstable,
        },
        "input_hashes": input_hashes,
    }


def run_jackknife(
    run_ids: list[str],
    anchor_run_id: str | None = None,
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full jackknife: validate, analyze, write."""
    result = jackknife_analysis(run_ids, runs_root)

    if dry_run:
        print(json.dumps(result, indent=2))
        return result

    anchor = anchor_run_id or sorted(run_ids)[0]
    run_dir, _ = validate_and_load_run(anchor, runs_root)
    out_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    _write_json_atomic(out_dir / "stability_per_geometry.json", result["geometry_stability"])
    _write_json_atomic(out_dir / "stability_certificate.json", {
        gid: data["certificate"] for gid, data in result["geometry_stability"].items()
    })
    _write_json_atomic(out_dir / "influence_ranking.json", {
        "ranking": result["high_influence_events"],
        "deltas": result["influence_delta"],
    })
    write_manifest(out_dir, result["input_hashes"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E5-B: Jackknife stability audit")
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--anchor-run", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_jackknife(
        run_ids=args.run_ids,
        anchor_run_id=args.anchor_run,
        runs_root=args.runs_root,
        dry_run=args.dry_run,
    )
    s = result["summary"]
    print(f"Geometries: {s['n_geometries_total']} total, {s['n_stable']} stable, {s['n_unstable']} unstable")
    print(f"Most influential event: {result['high_influence_events'][0]}")


if __name__ == "__main__":
    main()
