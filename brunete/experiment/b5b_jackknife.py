#!/usr/bin/env python3
"""B5-B — Jackknife Stability Audit (BRUNETE port of E5-B).

Leave-one-out analysis over the events in a BRUNETE classify run.
Measures how much the geometry intersection/union changes when one event
is removed, and which event has the most influence.

Governance
----------
- Read-only on compatible_set.json per event subrun (all RUN_VALID=PASS).
- Writes only under runs/<classify_run_id>/experiment/jackknife_stability_<mode>/.
- Requires at least 3 events.
"""
from __future__ import annotations

import argparse
import json
import math
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
from brunete.experiment.b5a_multi_event_aggregation import _extract_geometry_ids

SCHEMA_VERSION = "b5b-0.1"
EXPERIMENT_NAME = "jackknife_stability"

STABLE_VARIANCE_THRESHOLD = 0.05
UNSTABLE_VARIANCE_THRESHOLD = 0.20


def _classify_stability(variance: float, n_events: int) -> str:
    if n_events <= 1:
        return "SINGLETON"
    if variance <= STABLE_VARIANCE_THRESHOLD:
        return "STABLE"
    if variance <= UNSTABLE_VARIANCE_THRESHOLD:
        return "MODERATE"
    return "UNSTABLE"


def jackknife_analysis(
    classify_run_id: str,
    mode: str = "220",
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Leave-one-out jackknife over all events in a classify run."""
    event_run_map = enumerate_event_runs(classify_run_id, mode=mode, runs_root=runs_root)

    if len(event_run_map) < 3:
        raise ValueError(
            f"B5-B requires at least 3 event runs for mode {mode!r}, "
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

    if len(per_event) < 3:
        raise ValueError(
            f"B5-B requires at least 3 events with compatible_set.json, "
            f"got {len(per_event)}"
        )

    sorted_events = sorted(per_event)
    n = len(sorted_events)
    all_sets = [per_event[e] for e in sorted_events]
    full_union = set.union(*all_sets)

    # Per-geometry jackknife variance
    geometry_stability: dict[str, dict[str, Any]] = {}
    for gid in sorted(full_union):
        presence = [1.0 if gid in per_event[e] else 0.0 for e in sorted_events]
        mean_presence = sum(presence) / n
        jack_means = [
            sum(presence[:i] + presence[i + 1:]) / (n - 1)
            for i in range(n)
        ]
        jack_var = ((n - 1) / n) * sum(
            (jm - mean_presence) ** 2 for jm in jack_means
        )
        geometry_stability[gid] = {
            "mean_presence": round(mean_presence, 4),
            "jackknife_variance": round(jack_var, 6),
            "certificate": _classify_stability(jack_var, n),
        }

    # Influence ranking: which event changes intersection most when removed?
    full_intersection = set.intersection(*all_sets)
    influence: dict[str, int] = {}
    for i, event_id in enumerate(sorted_events):
        loo_sets = all_sets[:i] + all_sets[i + 1:]
        loo_intersection = set.intersection(*loo_sets)
        delta = len(loo_intersection) - len(full_intersection)
        influence[event_id] = delta

    influence_ranking = sorted(influence.items(), key=lambda x: -abs(x[1]))

    # Summary counts
    cert_counts: dict[str, int] = {}
    for info in geometry_stability.values():
        cert = info["certificate"]
        cert_counts[cert] = cert_counts.get(cert, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "classify_run_id": classify_run_id,
        "n_events": n,
        "n_geometries_analyzed": len(full_union),
        "stability_certificates": geometry_stability,
        "certificate_summary": cert_counts,
        "influence_ranking": [
            {"event_id": eid, "intersection_delta": delta}
            for eid, delta in influence_ranking
        ],
        "full_intersection_size": len(full_intersection),
        "input_hashes": input_hashes,
    }


def run_b5b(
    classify_run_id: str,
    mode: str = "220",
    runs_root: str | Path | None = None,
) -> Path:
    run_dir = resolve_classify_run_dir(classify_run_id, runs_root)
    exp_dir = ensure_experiment_dir(run_dir, f"{EXPERIMENT_NAME}_{mode}")

    result = jackknife_analysis(classify_run_id, mode=mode, runs_root=runs_root)

    out_path = exp_dir / "stability_per_geometry.json"
    _write_json_atomic(out_path, result["stability_certificates"])
    _write_json_atomic(exp_dir / "stability_certificate.json", {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "n_events": result["n_events"],
        "certificate_summary": result["certificate_summary"],
        "full_intersection_size": result["full_intersection_size"],
    })
    _write_json_atomic(exp_dir / "influence_ranking.json", result["influence_ranking"])
    write_manifest(exp_dir, result["input_hashes"],
                   extra={"schema_version": SCHEMA_VERSION, "mode": mode})
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="B5-B: jackknife stability audit over events in a classify run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--classify-run", required=True)
    ap.add_argument("--mode", choices=["220", "221"], default="220")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = jackknife_analysis(args.classify_run, mode=args.mode, runs_root=args.runs_root)

    if args.dry_run:
        print(json.dumps({k: v for k, v in result.items()
                          if k != "stability_certificates"}, indent=2))
        return 0

    out_path = run_b5b(args.classify_run, mode=args.mode, runs_root=args.runs_root)
    print(f"B5-B written: {out_path}")
    for cert, count in sorted(result["certificate_summary"].items()):
        print(f"  {cert:12s}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
