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


def _extract_compatible_geometry_ids(payload: dict[str, Any]) -> set[str]:
    """Extract compatible geometry IDs from legacy and newer compatible_set schemas."""
    compatible_geometries = payload.get("compatible_geometries")
    if isinstance(compatible_geometries, list):
        out: set[str] = set()
        for row in compatible_geometries:
            if not isinstance(row, dict):
                continue
            if "compatible" in row and row["compatible"] is not True:
                continue
            gid = row.get("geometry_id") or row.get("id")
            if gid is not None:
                out.add(str(gid))
        return out

    compatible_entries = payload.get("compatible_entries")
    if isinstance(compatible_entries, list):
        out = set()
        for row in compatible_entries:
            if isinstance(row, dict):
                gid = row.get("geometry_id") or row.get("id")
                if gid is not None:
                    out.add(str(gid))
            elif isinstance(row, str):
                out.add(row)
        return out

    compatible_set = payload.get("compatible_set")
    if isinstance(compatible_set, list):
        out = set()
        for row in compatible_set:
            if isinstance(row, dict):
                gid = row.get("geometry_id") or row.get("id")
                if gid is not None:
                    out.add(str(gid))
            elif isinstance(row, str):
                out.add(row)
        return out

    compatible_ids = payload.get("compatible_ids")
    if isinstance(compatible_ids, list):
        return {str(x) for x in compatible_ids}

    return set()


def aggregate_compatible_sets(
    source_data: list[dict[str, Any]], min_coverage: float = 1.0, top_k: int = 50,
) -> dict[str, Any]:
    n_events = len(source_data)
    if n_events == 0:
        return {
            "schema_version": "mvp_aggregate_v2",
            "n_events": 0,
            "events": [],
            "joint_posterior": {
                "prior_type": "uniform_entries",
                "normalization": "relative_only",
                "combination": "sum_logL_rel",
                "chi2_dof_per_event": 2,
                "chi2_interpretation": "min_over_atlas_not_chi2",
                "best_entry_id": None,
                "d2_sum_min": None,
                "log_likelihood_rel_best": None,
                "common_geometries": [],
            },
            "common_geometries": [],
            "n_common_geometries": 0,
            "coverage_histogram": {},
            "n_total_unique_geometries": 0,
            "min_coverage": min_coverage,
            "min_count": 0,
        }

    counter: Counter[str] = Counter()
    d2_by_geo: dict[str, list[float | None]] = {}
    metadata: dict[str, Any] = {}
    events: list[dict[str, Any]] = []

    def _extract_d2(row: dict[str, Any], metric: str) -> float | None:
        d2 = row.get("d2")
        if isinstance(d2, (int, float)) and math.isfinite(d2):
            return float(d2)
        distance = row.get("distance")
        if metric == "mahalanobis_log" and isinstance(distance, (int, float)) and math.isfinite(distance):
            return float(distance) ** 2
        return None

    for src in source_data:
        metric = str(src.get("metric", ""))
        ranked_all = src.get("ranked_all", [])
        threshold_d2 = src.get("threshold_d2")
        events.append(
            {
                "run_id": src["run_id"],
                "event_id": src["event_id"],
                "metric": metric,
                "threshold_d2": threshold_d2,
                "n_atlas": len(ranked_all),
            }
        )

        current: dict[str, float | None] = {}
        for row in ranked_all:
            gid = row.get("geometry_id")
            if not gid:
                continue
            current[gid] = _extract_d2(row, metric)
            if gid not in metadata and row.get("metadata"):
                metadata[gid] = row["metadata"]

        for gid in current:
            counter[gid] += 1

        for gid in d2_by_geo:
            d2_by_geo[gid].append(current.get(gid))
        for gid, d2 in current.items():
            if gid not in d2_by_geo:
                d2_by_geo[gid] = [None] * (len(events) - 1)
                d2_by_geo[gid].append(d2)

    min_count = max(1, int(math.ceil(min_coverage * n_events)))
    all_rows: list[dict[str, Any]] = []
    for gid in sorted(d2_by_geo):
        per_event = d2_by_geo[gid]
        finite_d2 = [d for d in per_event if isinstance(d, (int, float)) and math.isfinite(d)]
        coverage = counter[gid] / n_events
        d2_sum = float(sum(finite_d2))
        row = {
            "geometry_id": gid,
            "d2_per_event": per_event,
            "d2_sum": d2_sum,
            "coverage": coverage,
            "log_likelihood_rel_joint": -0.5 * d2_sum,
            "support_count": 0,
            "support_fraction": 0.0,
        }
        if metadata.get(gid) is not None:
            row["metadata"] = metadata[gid]
        all_rows.append(row)

    eligible = [r for r in all_rows if r["coverage"] >= min_coverage]
    if eligible:
        best = min(eligible, key=lambda r: (r["d2_sum"], r["geometry_id"]))
        best_logl = best["log_likelihood_rel_joint"]
        logits = [r["log_likelihood_rel_joint"] for r in eligible]
        max_logit = max(logits)
        exp_vals = [math.exp(v - max_logit) for v in logits]
        norm = sum(exp_vals)

        for row, ev in zip(eligible, exp_vals):
            row["posterior_weight_joint"] = ev / norm
            row["delta_lnL_joint"] = row["log_likelihood_rel_joint"] - best_logl
            support_count = 0
            for idx, d2 in enumerate(row["d2_per_event"]):
                threshold = events[idx].get("threshold_d2")
                if (
                    isinstance(d2, (int, float))
                    and math.isfinite(d2)
                    and isinstance(threshold, (int, float))
                    and math.isfinite(threshold)
                    and d2 <= threshold
                ):
                    support_count += 1
            row["support_count"] = support_count
            row["support_fraction"] = support_count / n_events

        ranked = sorted(
            eligible,
            key=lambda r: (-r["posterior_weight_joint"], r["geometry_id"]),
        )
        common = ranked[:max(0, top_k)] if top_k is not None else ranked
        best_entry_id = best["geometry_id"]
        d2_sum_min = best["d2_sum"]
        log_likelihood_rel_best = best_logl
    else:
        ranked = []
        common = []
        best_entry_id = None
        d2_sum_min = None
        log_likelihood_rel_best = None

    coverage_hist: dict[str, int] = {}
    for c in counter.values():
        coverage_hist[str(c)] = coverage_hist.get(str(c), 0) + 1

    return {
        "schema_version": "mvp_aggregate_v2",
        "n_events": n_events, "min_coverage": min_coverage, "min_count": min_count,
        "n_total_unique_geometries": len(counter),
        "n_common_geometries": len(common), "common_geometries": common,
        "joint_posterior": {
            "prior_type": "uniform_entries",
            "normalization": "relative_only",
            "combination": "sum_logL_rel",
            "chi2_dof_per_event": 2,
            "chi2_interpretation": "min_over_atlas_not_chi2",
            "best_entry_id": best_entry_id,
            "d2_sum_min": d2_sum_min,
            "log_likelihood_rel_best": log_likelihood_rel_best,
            "common_geometries": common,
            "joint_ranked_all": ranked,
        },
        "coverage_histogram": coverage_hist,
        "events": events,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: aggregate events")
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--source-runs", required=True)
    ap.add_argument("--min-coverage", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
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
        "source_runs": source_runs, "min_coverage": args.min_coverage, "top_k": args.top_k,
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

            ranked_all = cs.get("ranked_all", [])
            if not ranked_all:
                ranked_all = [{"geometry_id": gid} for gid in sorted(_extract_compatible_geometry_ids(cs))]

            source_data.append({
                "run_id": src, "event_id": cs.get("event_id", "unknown"),
                "metric": cs.get("metric"),
                "threshold_d2": cs.get("threshold_d2"),
                "ranked_all": ranked_all,
                "observables": cs.get("observables", {}),
            })

        result = aggregate_compatible_sets(source_data, args.min_coverage, top_k=args.top_k)
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
