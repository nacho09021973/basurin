#!/usr/bin/env python3
"""B5-F — Classification Aggregation (BRUNETE port of E5-F).

Aggregates event classifications from a BRUNETE classify_geometries run.
Produces population-level joint-support rates — the core result for the
spectral exclusion paper.

Unlike E5-F (which aggregates per-geometry BASURIN verdict.json), B5-F
aggregates the richer BRUNETE classification labels:
    - has_joint_support (bool)
    - classification label (e.g. "common_nonempty_both_221_support_multi")
    - per-mode n_compatible counts

Governance
----------
- Reads only classify_geometries/outputs/geometry_summary.json (RUN_VALID=PASS).
- Writes only under runs/<classify_run_id>/experiment/classification_aggregation/.
- Evidence classification is DETERMINISTIC_THRESHOLD, not Bayesian.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from brunete.experiment.base_contract import (
    BRUNETE_CLASSIFY_GATES,
    GovernanceViolation,
    _write_json_atomic,
    ensure_experiment_dir,
    load_json,
    sha256_file,
    validate_classify_run,
    write_manifest,
)

SCHEMA_VERSION = "b5f-0.1"
EXPERIMENT_NAME = "classification_aggregation"

STRONG_THRESHOLD = 0.8
MODERATE_THRESHOLD = 0.5


def _classify_evidence(rate: float) -> str:
    if rate >= STRONG_THRESHOLD:
        return "STRONG"
    if rate >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "WEAK"


def aggregate_classifications(
    classify_run_ids: list[str],
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate geometry_summary.json across multiple classify runs.

    Each classify run contributes one data point per event it contains.
    Returns population-level joint-support rates and classification counts.
    """
    joint_support_count = 0
    total_events = 0
    classification_counter: Counter = Counter()
    n_compatible_220_values: list[int] = []
    n_compatible_221_values: list[int] = []
    input_hashes: dict[str, str] = {}
    per_run_summary: list[dict] = []

    for classify_run_id in sorted(classify_run_ids):
        run_dir, geometry_summary = validate_classify_run(classify_run_id, runs_root)
        gs_path = run_dir / BRUNETE_CLASSIFY_GATES["geometry_summary"]
        input_hashes[classify_run_id] = sha256_file(gs_path)

        rows = geometry_summary.get("rows", [])
        run_joint = 0
        for row in rows:
            total_events += 1
            if row.get("has_joint_support"):
                joint_support_count += 1
                run_joint += 1
            label = row.get("classification", "unknown")
            classification_counter[label] += 1
            n220 = row.get("n_compatible_220")
            n221 = row.get("n_compatible_221")
            if isinstance(n220, int):
                n_compatible_220_values.append(n220)
            if isinstance(n221, int):
                n_compatible_221_values.append(n221)

        per_run_summary.append({
            "classify_run_id": classify_run_id,
            "n_events": len(rows),
            "n_joint_support": run_joint,
            "joint_support_rate": round(run_joint / len(rows), 4) if rows else None,
        })

    joint_support_rate = joint_support_count / total_events if total_events > 0 else 0.0

    def _mean(vals: list[int]) -> float | None:
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "schema_version": SCHEMA_VERSION,
        "n_classify_runs": len(classify_run_ids),
        "n_total_events": total_events,
        "joint_support_rate": round(joint_support_rate, 4),
        "evidence_strength": _classify_evidence(joint_support_rate),
        "n_joint_support": joint_support_count,
        "classification_distribution": dict(classification_counter.most_common()),
        "mean_n_compatible_220": _mean(n_compatible_220_values),
        "mean_n_compatible_221": _mean(n_compatible_221_values),
        "per_run_summary": per_run_summary,
        "input_hashes": input_hashes,
    }


def run_b5f(
    classify_run_ids: list[str],
    anchor_run_id: str,
    runs_root: str | Path | None = None,
) -> Path:
    """Execute B5-F and write outputs under the anchor run's experiment dir.

    anchor_run_id is the classify run that hosts the experiment output.
    classify_run_ids is the list of classify runs to aggregate (may include anchor).
    """
    from brunete.experiment.base_contract import resolve_classify_run_dir

    anchor_dir = resolve_classify_run_dir(anchor_run_id, runs_root)
    exp_dir = ensure_experiment_dir(anchor_dir, EXPERIMENT_NAME)

    result = aggregate_classifications(classify_run_ids, runs_root)

    out_path = exp_dir / "population_verdict.json"
    _write_json_atomic(out_path, result)

    manifest_path = write_manifest(
        exp_dir,
        result["input_hashes"],
        extra={
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT_NAME,
            "anchor_run_id": anchor_run_id,
            "classify_run_ids": sorted(classify_run_ids),
        },
    )
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="B5-F: aggregate BRUNETE classification rates across classify runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--classify-runs", nargs="+", required=True,
                    help="classify_run_ids to aggregate")
    ap.add_argument("--anchor", required=True,
                    help="classify_run_id that hosts the output (must be in --classify-runs)")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print result without writing files")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = aggregate_classifications(args.classify_runs, args.runs_root)

    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0

    out_path = run_b5f(args.classify_runs, args.anchor, args.runs_root)
    print(f"B5-F written: {out_path}")
    print(f"  joint_support_rate : {result['joint_support_rate']:.1%}")
    print(f"  evidence_strength  : {result['evidence_strength']}")
    print(f"  n_total_events     : {result['n_total_events']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
