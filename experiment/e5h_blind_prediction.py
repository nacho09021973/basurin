#!/usr/bin/env python3
"""E5-H — Blind Cross-Event Prediction (Leave-One-Out Prediction).

NEW ALTERNATIVE not in original catalog.  Uses N-1 events to PREDICT the
compatible_set of the Nth event before looking at it.  Measures BASURIN's
predictive power — the gold standard in physics.

Unlike E5-B (which measures stability/variance), E5-H measures whether the
population pattern from N-1 events can actually PREDICT what the held-out
event will show.  This is a fundamentally different question.

Key metric: prediction_recall — what fraction of the held-out event's actual
compatible geometries were predicted by the N-1 population intersection?

A result like "BASURIN predicted 78% of compatible geometries for GW170817
using only the other 4 events" is a paper headline.

Governance:
  - Read-only on compatible_set.json (all RUN_VALID=PASS).
  - Writes only under runs/<anchor>/experiment/blind_prediction/.
  - Requires at least 3 events (N-1 >= 2 for meaningful intersection).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mvp.experiment.base_contract import (
    REQUIRED_CANONICAL_GATES,
    _write_json_atomic,
    ensure_experiment_dir,
    load_json,
    sha256_file,
    validate_and_load_run,
    write_manifest,
)
from experiment.e5a_multi_event_aggregation import _extract_family, _extract_geometry_ids

SCHEMA_VERSION = "e5h-0.1"
EXPERIMENT_NAME = "blind_prediction"


def _prediction_metrics(predicted: set[str], actual: set[str]) -> dict[str, Any]:
    """Compute prediction quality metrics."""
    if not actual:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "n_predicted": len(predicted),
            "n_actual": 0,
            "n_true_positive": 0,
            "n_false_positive": len(predicted),
            "n_false_negative": 0,
        }

    tp = predicted & actual
    fp = predicted - actual
    fn = actual - predicted

    recall = len(tp) / len(actual) if actual else 0.0
    precision = len(tp) / len(predicted) if predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "n_predicted": len(predicted),
        "n_actual": len(actual),
        "n_true_positive": len(tp),
        "n_false_positive": len(fp),
        "n_false_negative": len(fn),
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
    }


def blind_prediction(
    run_ids: list[str],
    prediction_strategy: str = "intersection",
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Leave-one-out blind prediction across N events.

    For each event i:
      1. Compute prediction from N-1 other events
      2. Compare prediction vs actual compatible_set of event i
      3. Measure recall, precision, F1

    Prediction strategies:
      - "intersection": predict geometries in ALL other N-1 events
      - "majority": predict geometries in >50% of other N-1 events
      - "frequency_weighted": predict top-K by frequency in N-1 events
    """
    if len(run_ids) < 3:
        raise ValueError("E5-H requires at least 3 runs")

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
    predictions: list[dict[str, Any]] = []
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0

    for held_out in sorted_ids:
        # Build prediction from N-1 events
        training_ids = [rid for rid in sorted_ids if rid != held_out]
        training_sets = [per_run[rid] for rid in training_ids]

        if prediction_strategy == "intersection":
            predicted = set.intersection(*training_sets)
        elif prediction_strategy == "majority":
            from collections import Counter
            freq: Counter = Counter()
            for s in training_sets:
                freq.update(s)
            threshold = len(training_sets) / 2.0
            predicted = {gid for gid, count in freq.items() if count > threshold}
        elif prediction_strategy == "frequency_weighted":
            from collections import Counter
            freq = Counter()
            for s in training_sets:
                freq.update(s)
            # Predict top-K where K = median size of training sets
            median_size = sorted(len(s) for s in training_sets)[len(training_sets) // 2]
            predicted = {gid for gid, _ in freq.most_common(median_size)}
        else:
            raise ValueError(f"Unknown strategy: {prediction_strategy}")

        actual = per_run[held_out]
        metrics = _prediction_metrics(predicted, actual)

        # Per-family breakdown
        family_metrics: dict[str, dict] = {}
        all_families = set()
        for gid in predicted | actual:
            all_families.add(_extract_family(gid))
        for fam in sorted(all_families):
            fam_predicted = {g for g in predicted if _extract_family(g) == fam}
            fam_actual = {g for g in actual if _extract_family(g) == fam}
            family_metrics[fam] = _prediction_metrics(fam_predicted, fam_actual)
            # Remove verbose fields for per-family
            for key in ("true_positives", "false_positives", "false_negatives"):
                family_metrics[fam].pop(key, None)

        predictions.append({
            "held_out_event": held_out,
            "n_training_events": len(training_ids),
            "strategy": prediction_strategy,
            "metrics": metrics,
            "family_metrics": family_metrics,
        })

        total_recall += metrics["recall"]
        total_precision += metrics["precision"]
        total_f1 += metrics["f1"]

    mean_recall = round(total_recall / n, 4)
    mean_precision = round(total_precision / n, 4)
    mean_f1 = round(total_f1 / n, 4)

    return {
        "schema_version": SCHEMA_VERSION,
        "n_events": n,
        "prediction_strategy": prediction_strategy,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "mean_f1": mean_f1,
        "predictions": predictions,
        "input_hashes": input_hashes,
        "headline": (
            f"BASURIN predicted {mean_recall*100:.0f}% of compatible geometries "
            f"(mean recall) using leave-one-out {prediction_strategy} "
            f"across {n} events"
        ),
    }


def run_blind_prediction(
    run_ids: list[str],
    anchor_run_id: str | None = None,
    prediction_strategy: str = "intersection",
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full prediction pipeline: validate, predict, write."""
    result = blind_prediction(run_ids, prediction_strategy, runs_root)

    if dry_run:
        print(json.dumps(result, indent=2))
        return result

    anchor = anchor_run_id or sorted(run_ids)[0]
    run_dir, _ = validate_and_load_run(anchor, runs_root)
    out_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    _write_json_atomic(out_dir / "prediction_results.json", result)
    _write_json_atomic(out_dir / "prediction_summary.json", {
        "mean_recall": result["mean_recall"],
        "mean_precision": result["mean_precision"],
        "mean_f1": result["mean_f1"],
        "strategy": result["prediction_strategy"],
        "headline": result["headline"],
    })

    write_manifest(out_dir, result["input_hashes"], extra={
        "strategy": prediction_strategy,
        "n_events": result["n_events"],
    })

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5-H: Blind cross-event prediction (leave-one-out)"
    )
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--anchor-run", default=None)
    parser.add_argument("--strategy", default="intersection",
                        choices=["intersection", "majority", "frequency_weighted"])
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_blind_prediction(
        run_ids=args.run_ids,
        anchor_run_id=args.anchor_run,
        prediction_strategy=args.strategy,
        runs_root=args.runs_root,
        dry_run=args.dry_run,
    )
    print(result["headline"])
    for p in result["predictions"]:
        m = p["metrics"]
        print(f"  {p['held_out_event']}: recall={m['recall']:.2f} precision={m['precision']:.2f} F1={m['f1']:.2f}")


if __name__ == "__main__":
    main()
