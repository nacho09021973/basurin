#!/usr/bin/env python3
"""B5-C — Geometry Ranking by Composite Score (BRUNETE port of E5-C).

Deterministic ranking (NOT Bayesian) of compatible geometries per event,
using Mahalanobis distance, delta_lnL, and saturation penalty sourced
from the per-event BASURIN subruns.

Produces ranked list + Lorenz curve + Gini coefficient per event.

Governance
----------
- Reads compatible_set.json + estimates.json + verdict.json per event subrun.
- Writes only under runs/<classify_run_id>/experiment/geometry_ranking_<mode>/.
- Weights are declared in manifest — currently arbitrary, pending Fisher validation.
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
    write_manifest,
)

SCHEMA_VERSION = "b5c-0.1"
EXPERIMENT_NAME = "geometry_ranking"

DEFAULT_WEIGHTS = {"mahalanobis": 0.5, "delta_lnL": 0.4, "saturation_penalty": 0.1}


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    if span == 0:
        return [0.5] * len(values)
    return [(v - vmin) / span for v in values]


def _gini_coefficient(scores: list[float]) -> float:
    if not scores or all(s == 0 for s in scores):
        return 0.0
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    total = sum(sorted_scores)
    gini_sum = sum((2 * (i + 1) - n - 1) * s for i, s in enumerate(sorted_scores))
    return gini_sum / (n * total) if total > 0 else 0.0


def _lorenz_curve(scores: list[float]) -> list[dict[str, float]]:
    if not scores:
        return [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    total = sum(sorted_scores)
    if total == 0:
        return [{"x": i / n, "y": i / n} for i in range(n + 1)]
    points = [{"x": 0.0, "y": 0.0}]
    cumsum = 0.0
    for i, s in enumerate(sorted_scores):
        cumsum += s
        points.append({"x": round((i + 1) / n, 4), "y": round(cumsum / total, 4)})
    return points


def _rank_event(
    event_id: str,
    event_run_dir: Path,
    weights: dict[str, float],
) -> dict[str, Any] | None:
    """Rank geometries for a single event. Returns None if data is missing."""
    cs_path = event_run_dir / EVENT_RUN_GATES["compatible_set"]
    est_path = event_run_dir / EVENT_RUN_GATES["estimates"]
    verd_path = event_run_dir / EVENT_RUN_GATES["verdict"]

    if not cs_path.exists() or not est_path.exists():
        return None

    cs = load_json(cs_path)
    estimates = load_json(est_path)
    verdict = load_json(verd_path) if verd_path.exists() else {}

    # Build lookup by geometry_id
    if isinstance(cs, list):
        compatible_ids = {g.get("geometry_id", g.get("id")) for g in cs}
    else:
        geoms = cs.get("geometries", cs.get("compatible", []))
        compatible_ids = {g.get("geometry_id", g.get("id")) for g in geoms}
    compatible_ids.discard(None)

    est_list = estimates if isinstance(estimates, list) else estimates.get("estimates", [])
    filtered = [
        e for e in est_list
        if str(e.get("geometry_id", e.get("id", ""))) in compatible_ids
    ]

    if not filtered:
        return {"event_id": event_id, "n_compatible": 0, "ranked_geometries": [],
                "gini_coefficient": 0.0, "lorenz_curve": [], "weights": weights}

    maha_vals = [float(e.get("mahalanobis_d2", e.get("d2", 0.0)) or 0.0) for e in filtered]
    dlnl_vals = [float(e.get("delta_lnL", e.get("dlnl", 0.0)) or 0.0) for e in filtered]
    sat_vals = [float(e.get("saturation", 0.0) or 0.0) for e in filtered]

    maha_norm = _normalize(maha_vals)
    dlnl_norm = _normalize([-v for v in dlnl_vals])  # lower delta_lnL is better
    sat_norm = _normalize(sat_vals)

    w_m = weights["mahalanobis"]
    w_d = weights["delta_lnL"]
    w_s = weights["saturation_penalty"]

    ranked = []
    scores = []
    for i, est in enumerate(filtered):
        score = w_m * maha_norm[i] + w_d * dlnl_norm[i] + w_s * sat_norm[i]
        scores.append(score)
        ranked.append({
            "geometry_id": str(est.get("geometry_id", est.get("id", ""))),
            "score": round(score, 6),
            "mahalanobis_d2": maha_vals[i],
            "delta_lnL": dlnl_vals[i],
            "saturation": sat_vals[i],
            "mahalanobis_norm": round(maha_norm[i], 4),
            "delta_lnL_norm": round(dlnl_norm[i], 4),
            "saturation_norm": round(sat_norm[i], 4),
        })

    ranked.sort(key=lambda x: x["score"])

    return {
        "event_id": event_id,
        "n_compatible": len(ranked),
        "ranked_geometries": ranked,
        "gini_coefficient": round(_gini_coefficient(scores), 4),
        "lorenz_curve": _lorenz_curve(scores),
        "weights": weights,
    }


def rank_events(
    classify_run_id: str,
    mode: str = "220",
    weights: dict[str, float] | None = None,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    event_run_map = enumerate_event_runs(classify_run_id, mode=mode, runs_root=runs_root)
    input_hashes: dict[str, str] = {}
    per_event_rankings: list[dict] = []

    for event_id, event_run_dir in sorted(event_run_map.items()):
        cs_path = event_run_dir / EVENT_RUN_GATES["compatible_set"]
        if cs_path.exists():
            input_hashes[event_id] = sha256_file(cs_path)
        ranking = _rank_event(event_id, event_run_dir, weights)
        if ranking is not None:
            per_event_rankings.append(ranking)

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "classify_run_id": classify_run_id,
        "n_events_ranked": len(per_event_rankings),
        "weights": weights,
        "weights_note": "ARBITRARY — pending Fisher information validation",
        "per_event_rankings": per_event_rankings,
        "input_hashes": input_hashes,
    }


def run_b5c(
    classify_run_id: str,
    mode: str = "220",
    weights: dict[str, float] | None = None,
    runs_root: str | Path | None = None,
) -> Path:
    run_dir = resolve_classify_run_dir(classify_run_id, runs_root)
    exp_dir = ensure_experiment_dir(run_dir, f"{EXPERIMENT_NAME}_{mode}")

    result = rank_events(classify_run_id, mode=mode, weights=weights, runs_root=runs_root)

    out_path = exp_dir / "ranked_geometries.json"
    _write_json_atomic(out_path, result)
    write_manifest(exp_dir, result["input_hashes"], extra={
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "weights": result["weights"],
    })
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="B5-C: deterministic geometry ranking per event in a classify run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--classify-run", required=True)
    ap.add_argument("--mode", choices=["220", "221"], default="220")
    ap.add_argument("--weights", nargs=3, type=float, default=None,
                    metavar=("W_MAHA", "W_DLNL", "W_SAT"))
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    weights = None
    if args.weights:
        weights = {
            "mahalanobis": args.weights[0],
            "delta_lnL": args.weights[1],
            "saturation_penalty": args.weights[2],
        }
    result = rank_events(args.classify_run, mode=args.mode,
                         weights=weights, runs_root=args.runs_root)

    if args.dry_run:
        print(json.dumps({k: v for k, v in result.items()
                          if k != "per_event_rankings"}, indent=2))
        return 0

    out_path = run_b5c(args.classify_run, mode=args.mode,
                       weights=weights, runs_root=args.runs_root)
    print(f"B5-C written: {out_path}")
    print(f"  n_events_ranked: {result['n_events_ranked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
