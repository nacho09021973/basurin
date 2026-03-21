#!/usr/bin/env python3
"""B5-H — Blind Cross-Event Prediction (BRUNETE port of E5-H).

Leave-one-out predictive power test: uses N-1 events to predict which
geometries will be compatible for the held-out Nth event.

Measures BRUNETE's predictive power — the gold standard in physics.
A headline like "BRUNETE predicted 78% of compatible geometries for
GW170817 using only 4 other events (F1=0.82)" is paper-publishable.

Governance
----------
- Read-only on compatible_set.json per event subrun (all RUN_VALID=PASS).
- Writes only under runs/<classify_run_id>/experiment/blind_prediction_<mode>/.
- Requires at least 3 events.
"""
from __future__ import annotations

import argparse
import json
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
    write_manifest,
)
from brunete.experiment.b5a_multi_event_aggregation import (
    _extract_family,
    _extract_geometry_ids,
)

SCHEMA_VERSION = "b5h-0.1"
EXPERIMENT_NAME = "blind_prediction"


def _prediction_metrics(predicted: set[str], actual: set[str]) -> dict[str, Any]:
    if not actual:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "n_predicted": len(predicted), "n_actual": 0,
            "n_true_positive": 0,
            "n_false_positive": len(predicted),
            "n_false_negative": 0,
        }
    tp = predicted & actual
    fp = predicted - actual
    fn = actual - predicted
    recall = len(tp) / len(actual) if actual else 0.0
    precision = len(tp) / len(predicted) if predicted else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
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


def _predict(
    training_sets: list[set[str]],
    strategy: str,
) -> set[str]:
    if not training_sets:
        return set()
    if strategy == "intersection":
        return set.intersection(*training_sets)
    if strategy == "majority":
        n = len(training_sets)
        freq: dict[str, int] = {}
        for s in training_sets:
            for gid in s:
                freq[gid] = freq.get(gid, 0) + 1
        return {gid for gid, c in freq.items() if c > n / 2}
    if strategy == "frequency_weighted":
        freq = {}
        for s in training_sets:
            for gid in s:
                freq[gid] = freq.get(gid, 0) + 1
        return {gid for gid, c in freq.items() if c >= 2}
    raise ValueError(f"Unknown prediction strategy: {strategy!r}")


def blind_prediction(
    classify_run_id: str,
    mode: str = "220",
    prediction_strategy: str = "intersection",
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Leave-one-out blind prediction across all events in a classify run."""
    event_run_map = enumerate_event_runs(classify_run_id, mode=mode, runs_root=runs_root)

    if len(event_run_map) < 3:
        raise ValueError(
            f"B5-H requires at least 3 event runs for mode {mode!r}, "
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
            f"B5-H requires at least 3 events with compatible_set.json, "
            f"got {len(per_event)}"
        )

    sorted_events = sorted(per_event)
    all_sets = [per_event[e] for e in sorted_events]

    per_event_results: list[dict] = []
    per_family_tp: dict[str, int] = {}
    per_family_actual: dict[str, int] = {}

    for i, held_out_event in enumerate(sorted_events):
        training_sets = all_sets[:i] + all_sets[i + 1:]
        predicted = _predict(training_sets, prediction_strategy)
        actual = per_event[held_out_event]
        metrics = _prediction_metrics(predicted, actual)

        # Per-family metrics
        for gid in actual:
            fam = _extract_family(gid)
            per_family_actual[fam] = per_family_actual.get(fam, 0) + 1
        for gid in (predicted & actual):
            fam = _extract_family(gid)
            per_family_tp[fam] = per_family_tp.get(fam, 0) + 1

        per_event_results.append({
            "held_out_event": held_out_event,
            "n_training_events": len(training_sets),
            "strategy": prediction_strategy,
            **metrics,
        })

    # Global metrics
    total_actual = sum(r["n_actual"] for r in per_event_results)
    total_tp = sum(r["n_true_positive"] for r in per_event_results)
    total_predicted = sum(r["n_predicted"] for r in per_event_results)
    global_recall = total_tp / total_actual if total_actual else 0.0
    global_precision = total_tp / total_predicted if total_predicted else 0.0
    global_f1 = (
        2 * global_precision * global_recall / (global_precision + global_recall)
        if (global_precision + global_recall) > 0 else 0.0
    )

    per_family_predictions = {}
    for fam in set(per_family_actual) | set(per_family_tp):
        tp_c = per_family_tp.get(fam, 0)
        actual_c = per_family_actual.get(fam, 0)
        per_family_predictions[fam] = {
            "recall": round(tp_c / actual_c, 4) if actual_c else 0.0,
            "n_actual": actual_c,
            "n_true_positive": tp_c,
        }

    best = max(per_event_results, key=lambda r: r["recall"])
    worst = min(per_event_results, key=lambda r: r["recall"])

    headline = (
        f"BRUNETE predicted {global_recall:.1%} of compatible geometries "
        f"across {len(sorted_events)} events "
        f"(strategy: {prediction_strategy}, global F1={global_f1:.2f})"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "classify_run_id": classify_run_id,
        "n_events": len(sorted_events),
        "prediction_strategy": prediction_strategy,
        "prediction_summary": {
            "global_recall": round(global_recall, 4),
            "global_precision": round(global_precision, 4),
            "global_f1": round(global_f1, 4),
            "total_actual": total_actual,
            "total_true_positive": total_tp,
            "headline": headline,
        },
        "best_event": best["held_out_event"],
        "worst_event": worst["held_out_event"],
        "per_event_predictions": per_event_results,
        "per_family_predictions": per_family_predictions,
        "input_hashes": input_hashes,
    }


def run_b5h(
    classify_run_id: str,
    mode: str = "220",
    prediction_strategy: str = "intersection",
    runs_root: str | Path | None = None,
) -> Path:
    run_dir = resolve_classify_run_dir(classify_run_id, runs_root)
    exp_dir = ensure_experiment_dir(run_dir, f"{EXPERIMENT_NAME}_{mode}_{prediction_strategy}")

    result = blind_prediction(
        classify_run_id, mode=mode,
        prediction_strategy=prediction_strategy, runs_root=runs_root,
    )

    out_path = exp_dir / "prediction_summary.json"
    _write_json_atomic(out_path, result["prediction_summary"])
    _write_json_atomic(exp_dir / "per_event_predictions.json", result["per_event_predictions"])
    _write_json_atomic(exp_dir / "per_family_predictions.json", result["per_family_predictions"])
    write_manifest(exp_dir, result["input_hashes"], extra={
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "strategy": prediction_strategy,
    })
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="B5-H: leave-one-out blind prediction across events in a classify run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--classify-run", required=True)
    ap.add_argument("--mode", choices=["220", "221"], default="220")
    ap.add_argument("--strategy", choices=["intersection", "majority", "frequency_weighted"],
                    default="intersection")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = blind_prediction(
        args.classify_run, mode=args.mode,
        prediction_strategy=args.strategy, runs_root=args.runs_root,
    )

    if args.dry_run:
        print(json.dumps(result["prediction_summary"], indent=2))
        return 0

    out_path = run_b5h(
        args.classify_run, mode=args.mode,
        prediction_strategy=args.strategy, runs_root=args.runs_root,
    )
    print(f"B5-H written: {out_path}")
    print(f"  {result['prediction_summary']['headline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
