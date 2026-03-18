#!/usr/bin/env python3
"""E5-C — Geometry Ranking by Composite Score.

Deterministic ranking (NOT Bayesian selection) using:
  - Mahalanobis distance (closeness to observation)
  - Normalized delta_lnL (likelihood improvement)
  - Saturation penalty for mode 221

Produces ranked list + Lorenz curve + Gini coefficient.

Governance:
  - Reads compatible_set.json + estimates.json + verdict.json (RUN_VALID=PASS).
  - Writes only under runs/<run_id>/experiment/geometry_ranking/.
  - Weights are declared in manifest — currently arbitrary, pending Fisher validation.
"""
from __future__ import annotations

import argparse
import json
import math
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

SCHEMA_VERSION = "e5c-0.1"
EXPERIMENT_NAME = "geometry_ranking"

# Default weights (ARBITRARY — requires Fisher information validation)
DEFAULT_WEIGHTS = {
    "mahalanobis": 0.5,
    "delta_lnL": 0.4,
    "saturation_penalty": 0.1,
}


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of floats to [0, 1]."""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    if span == 0:
        return [0.5] * len(values)
    return [(v - vmin) / span for v in values]


def _gini_coefficient(scores: list[float]) -> float:
    """Compute Gini coefficient from a list of non-negative scores."""
    if not scores or all(s == 0 for s in scores):
        return 0.0
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    total = sum(sorted_scores)
    cumulative = 0.0
    gini_sum = 0.0
    for i, s in enumerate(sorted_scores):
        cumulative += s
        gini_sum += (2 * (i + 1) - n - 1) * s
    return gini_sum / (n * total) if total > 0 else 0.0


def _lorenz_curve(scores: list[float]) -> list[dict[str, float]]:
    """Compute Lorenz curve coordinates."""
    if not scores:
        return [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    total = sum(sorted_scores)
    if total == 0:
        return [{"x": i / n, "y": i / n} for i in range(n + 1)]
    points = [{"x": 0.0, "y": 0.0}]
    cumulative = 0.0
    for i, s in enumerate(sorted_scores):
        cumulative += s
        points.append({"x": (i + 1) / n, "y": round(cumulative / total, 6)})
    return points


def rank_geometries(
    run_id: str,
    weights: dict[str, float] | None = None,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rank geometries in a single run by composite score."""
    w = weights or DEFAULT_WEIGHTS.copy()

    run_dir, _ = validate_and_load_run(run_id, runs_root)
    cs_path = run_dir / REQUIRED_CANONICAL_GATES["compatible_set"]
    est_path = run_dir / REQUIRED_CANONICAL_GATES["estimates"]
    verdict_path = run_dir / REQUIRED_CANONICAL_GATES["verdict"]

    for p, name in [(cs_path, "compatible_set"), (est_path, "estimates")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} missing: {p}")

    cs = load_json(cs_path)
    estimates = load_json(est_path)

    geometries = cs if isinstance(cs, list) else cs.get("geometries", cs.get("compatible", []))

    # Extract raw component values
    entries = []
    for g in geometries:
        gid = g.get("geometry_id", g.get("id", "unknown"))
        maha = float(g.get("mahalanobis_d2", g.get("distance", 0.0)))
        dlnl = float(g.get("delta_lnL", g.get("delta_log_likelihood", 0.0)))
        sat = float(g.get("saturation_221", g.get("saturation_penalty", 0.0)))
        entries.append({
            "geometry_id": gid,
            "raw_mahalanobis": maha,
            "raw_delta_lnL": dlnl,
            "raw_saturation": sat,
        })

    if not entries:
        return {
            "schema_version": SCHEMA_VERSION,
            "scoring_weights": w,
            "ranked": [],
            "gini_coefficient": 0.0,
            "score_policy": "DETERMINISTIC_WEIGHTED — not a posterior",
        }

    # Normalize components
    maha_raw = [e["raw_mahalanobis"] for e in entries]
    dlnl_raw = [e["raw_delta_lnL"] for e in entries]
    sat_raw = [e["raw_saturation"] for e in entries]

    # For mahalanobis: lower is better → invert normalization
    maha_norm = [1.0 - v for v in _normalize(maha_raw)]
    dlnl_norm = _normalize(dlnl_raw)
    # For saturation: lower is better → invert
    sat_norm = [1.0 - v for v in _normalize(sat_raw)]

    # Compute composite scores
    for i, e in enumerate(entries):
        score = (
            w["mahalanobis"] * maha_norm[i]
            + w["delta_lnL"] * dlnl_norm[i]
            + w["saturation_penalty"] * sat_norm[i]
        )
        e["composite_score"] = round(score, 6)
        e["components"] = {
            "mahalanobis_normalized": round(maha_norm[i], 6),
            "delta_lnL_normalized": round(dlnl_norm[i], 6),
            "saturation_normalized": round(sat_norm[i], 6),
        }

    # Sort by composite score descending
    entries.sort(key=lambda e: e["composite_score"], reverse=True)
    for rank, e in enumerate(entries, 1):
        e["rank"] = rank

    scores = [e["composite_score"] for e in entries]
    gini = _gini_coefficient(scores)
    lorenz = _lorenz_curve(scores)

    input_hashes = {
        "compatible_set": sha256_file(cs_path),
        "estimates": sha256_file(est_path),
    }
    if verdict_path.exists():
        input_hashes["verdict"] = sha256_file(verdict_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scoring_weights": w,
        "ranked": entries,
        "gini_coefficient": round(gini, 6),
        "lorenz_curve": lorenz,
        "score_policy": "DETERMINISTIC_WEIGHTED — not a posterior",
        "input_hashes": input_hashes,
    }


def run_ranking(
    run_id: str,
    weights: dict[str, float] | None = None,
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full ranking: validate, score, write."""
    result = rank_geometries(run_id, weights, runs_root)

    if dry_run:
        print(json.dumps(result, indent=2))
        return result

    run_dir, _ = validate_and_load_run(run_id, runs_root)
    out_dir = ensure_experiment_dir(run_dir, EXPERIMENT_NAME)

    _write_json_atomic(out_dir / "ranked_geometries.json", result["ranked"])
    _write_json_atomic(out_dir / "lorenz_curve.json", result["lorenz_curve"])
    _write_json_atomic(out_dir / "gini_coefficient.json", {
        "gini_coefficient": result["gini_coefficient"],
    })
    write_manifest(out_dir, result["input_hashes"], extra={
        "scoring_weights": result["scoring_weights"],
    })

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E5-C: Geometry ranking by composite score")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--weights", nargs=3, type=float, default=[0.5, 0.4, 0.1],
                        metavar=("MAHA", "DLNL", "SAT"),
                        help="Weights for mahalanobis, delta_lnL, saturation_penalty")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    w = {
        "mahalanobis": args.weights[0],
        "delta_lnL": args.weights[1],
        "saturation_penalty": args.weights[2],
    }

    result = run_ranking(run_id=args.run_id, weights=w, runs_root=args.runs_root, dry_run=args.dry_run)
    print(f"Ranked {len(result['ranked'])} geometries, Gini={result['gini_coefficient']}")
    if result["ranked"]:
        top = result["ranked"][0]
        print(f"Top: {top['geometry_id']} (score={top['composite_score']})")


if __name__ == "__main__":
    main()
