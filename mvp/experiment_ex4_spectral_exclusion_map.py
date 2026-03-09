#!/usr/bin/env python3
"""Experiment EX4: Spectral Exclusion Map — population-level theory exclusion.

CLI:
    python mvp/experiment_ex4_spectral_exclusion_map.py \\
        --run <run_id_del_aggregate> \\
        [--threshold-sigma 3.0] \\
        [--top-k 50]

Input:
    runs/<run_id>/s5_aggregate/outputs/aggregate.json

Outputs (under runs/<run_id>/experiment_ex4_spectral_exclusion/outputs/):
    exclusion_map.json     — matrix of exclusion status per geometry × event
    theory_survival.json   — per-theory survival statistics
    stage_summary.json     — via finalize()
    manifest.json          — via finalize()

Scientific purpose:
    Builds a population-level spectral exclusion map. Given a catalogue of N
    processed events and an atlas of M geometries (theories), determines which
    geometries are compatible with ALL events, which are excluded, and with
    what discriminating strength.

    This is the intersection test that distinguishes BASURIN from event-by-event
    analyses such as pyRing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import sha256_file, utc_now_iso, write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage

STAGE = "experiment_ex4_spectral_exclusion"
SCHEMA_EXCLUSION_MAP = "ex4_exclusion_map_v1"
SCHEMA_THEORY_SURVIVAL = "ex4_theory_survival_v1"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in {path}, got {type(payload).__name__}"
        )
    return payload


def _percentile(vals: list[float], q: float) -> float:
    """Compute the q-th percentile (0 <= q <= 1) of a non-empty list of floats."""
    if not vals:
        raise ValueError("Cannot compute percentile of empty list")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# ── Core computation ──────────────────────────────────────────────────────────


def build_exclusion_matrix(
    events: list[dict[str, Any]],
    joint_ranked_all: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Compute exclusion status for each geometry × event pair.

    Args:
        events: List of event dicts (from aggregate.json), each with threshold_d2.
        joint_ranked_all: Ranked geometries list (from aggregate.json joint_posterior).
        top_k: Maximum number of geometries to include (first top_k by posterior rank).

    Returns:
        List of matrix rows, one per geometry (up to top_k). Each row contains:
            geometry_id, metadata, d2_per_event, status_per_event,
            excess_sigma_per_event, d2_sum, n_compatible, n_excluded,
            n_not_evaluated, survival_fraction, is_globally_compatible,
            max_exclusion_sigma, mean_d2.
    """
    n_events = len(events)
    geom_rows = joint_ranked_all[: max(0, top_k)]
    rows: list[dict[str, Any]] = []

    for geom in geom_rows:
        gid = geom["geometry_id"]
        d2_per_event_raw: list[Any] = geom.get("d2_per_event", [])
        metadata = geom.get("metadata")

        status_per_event: list[str] = []
        excess_sigma_per_event: list[float | None] = []
        n_compatible = 0
        n_excluded = 0

        for i in range(n_events):
            threshold = float(events[i]["threshold_d2"])
            d2_raw = d2_per_event_raw[i] if i < len(d2_per_event_raw) else None

            if d2_raw is None or not (
                isinstance(d2_raw, (int, float)) and math.isfinite(float(d2_raw))
            ):
                status_per_event.append("NOT_EVALUATED")
                excess_sigma_per_event.append(None)
            else:
                d2 = float(d2_raw)
                if d2 <= threshold:
                    status_per_event.append("COMPATIBLE")
                    excess_sigma_per_event.append(0.0)
                    n_compatible += 1
                else:
                    status_per_event.append("EXCLUDED")
                    excess_sigma_per_event.append(math.sqrt(d2 - threshold))
                    n_excluded += 1

        denominator = n_compatible + n_excluded
        survival_fraction: float | None = (
            n_compatible / denominator if denominator > 0 else None
        )
        is_globally_compatible = n_excluded == 0 and n_compatible > 0
        n_not_evaluated = n_events - n_compatible - n_excluded

        finite_sigmas = [s for s in excess_sigma_per_event if s is not None]
        max_exclusion_sigma = max(finite_sigmas) if finite_sigmas else 0.0

        finite_d2s = [
            float(d)
            for d in d2_per_event_raw
            if d is not None
            and isinstance(d, (int, float))
            and math.isfinite(float(d))
        ]
        mean_d2: float | None = (
            sum(finite_d2s) / len(finite_d2s) if finite_d2s else None
        )

        rows.append(
            {
                "geometry_id": gid,
                "metadata": metadata,
                "d2_per_event": list(d2_per_event_raw),
                "status_per_event": status_per_event,
                "excess_sigma_per_event": excess_sigma_per_event,
                "d2_sum": geom.get("d2_sum"),
                "n_compatible": n_compatible,
                "n_excluded": n_excluded,
                "n_not_evaluated": n_not_evaluated,
                "survival_fraction": survival_fraction,
                "is_globally_compatible": is_globally_compatible,
                "max_exclusion_sigma": max_exclusion_sigma,
                "mean_d2": mean_d2,
            }
        )

    return rows


def build_theory_survival(
    matrix_rows: list[dict[str, Any]],
    n_events: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group matrix rows by theory family and compute survival statistics.

    Args:
        matrix_rows: Output of build_exclusion_matrix.
        n_events: Total number of events in the population.

    Returns:
        (per_theory_list, summary_dict) where per_theory_list has one entry per
        theory family and summary_dict contains population-level counts.
    """
    theory_groups: dict[str, list[dict[str, Any]]] = {}
    for row in matrix_rows:
        meta = row.get("metadata")
        theory = (
            meta.get("theory") if isinstance(meta, dict) else None
        ) or "unknown"
        theory_groups.setdefault(theory, []).append(row)

    per_theory: list[dict[str, Any]] = []
    total_globally_compatible = 0
    total_excluded_at_least_once = 0
    globally_compatible_theory_families: list[str] = []
    fully_excluded_theory_families: list[str] = []

    for theory in sorted(theory_groups):
        rows = theory_groups[theory]
        n_entries = len(rows)
        n_globally_compatible = sum(1 for r in rows if r["is_globally_compatible"])
        n_excluded_at_least_once = sum(1 for r in rows if r["n_excluded"] > 0)

        total_globally_compatible += n_globally_compatible
        total_excluded_at_least_once += n_excluded_at_least_once

        # Best geometry: lowest d2_sum among globally compatible, else any row
        compat_rows = [r for r in rows if r["is_globally_compatible"]]
        candidate_pool = compat_rows if compat_rows else rows
        best = min(
            candidate_pool,
            key=lambda r: (
                float(r["d2_sum"])
                if (
                    r["d2_sum"] is not None
                    and math.isfinite(float(r["d2_sum"]))
                )
                else float("inf"),
                r["geometry_id"],
            ),
        )

        # Spin range for globally compatible geometries
        spin_values: list[float] = []
        for r in compat_rows:
            meta = r.get("metadata")
            if isinstance(meta, dict):
                spin = meta.get("spin")
                if isinstance(spin, (int, float)) and math.isfinite(float(spin)):
                    spin_values.append(float(spin))
        spin_range_compatible: list[float] | None = (
            [min(spin_values), max(spin_values)] if spin_values else None
        )

        # Descriptive stats on d2_sum across all rows for this theory
        d2_sums = [
            float(r["d2_sum"])
            for r in rows
            if r["d2_sum"] is not None and math.isfinite(float(r["d2_sum"]))
        ]
        if d2_sums:
            n_d2 = len(d2_sums)
            descriptive_stats: dict[str, float | None] = {
                "d2_sum_median": _percentile(d2_sums, 0.5),
                "d2_sum_p05": _percentile(d2_sums, 0.05) if n_d2 >= 5 else None,
                "d2_sum_p95": _percentile(d2_sums, 0.95) if n_d2 >= 5 else None,
            }
        else:
            descriptive_stats = {
                "d2_sum_median": None,
                "d2_sum_p05": None,
                "d2_sum_p95": None,
            }

        # A theory is fully excluded if:
        #   - no globally compatible geometries
        #   - ALL geometries fully evaluated (n_not_evaluated == 0)
        #   - ALL geometries excluded in at least one event
        all_evaluated = all(r["n_not_evaluated"] == 0 for r in rows)
        all_excluded_somewhere = all(r["n_excluded"] > 0 for r in rows)
        is_fully_excluded = (
            n_globally_compatible == 0 and all_evaluated and all_excluded_somewhere
        )

        if n_globally_compatible > 0:
            globally_compatible_theory_families.append(theory)
        if is_fully_excluded:
            fully_excluded_theory_families.append(theory)

        per_theory.append(
            {
                "theory": theory,
                "n_entries": n_entries,
                "n_globally_compatible": n_globally_compatible,
                "n_excluded_at_least_once": n_excluded_at_least_once,
                "best_geometry_id": best["geometry_id"],
                "best_d2_sum": best["d2_sum"],
                "best_survival_fraction": best["survival_fraction"],
                "spin_range_compatible": spin_range_compatible,
                "descriptive_stats": descriptive_stats,
            }
        )

    summary = {
        "n_globally_compatible_geometries": total_globally_compatible,
        "n_fully_excluded_geometries": total_excluded_at_least_once,
        "globally_compatible_theory_families": globally_compatible_theory_families,
        "fully_excluded_theory_families": fully_excluded_theory_families,
    }

    return per_theory, summary


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Experiment EX4: Spectral exclusion map at population level"
    )
    ap.add_argument("--run", required=True, help="Aggregate run ID")
    ap.add_argument(
        "--threshold-sigma",
        type=float,
        default=3.0,
        help="Reserved parameter (not used in current computation)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top-ranked geometries to include in the exclusion matrix",
    )
    args = ap.parse_args()

    ctx = init_stage(
        args.run,
        STAGE,
        params={"threshold_sigma": args.threshold_sigma, "top_k": args.top_k},
    )

    aggregate_path = ctx.run_dir / "s5_aggregate" / "outputs" / "aggregate.json"
    check_inputs(ctx, {"aggregate": aggregate_path})

    try:
        aggregate = _read_json(aggregate_path)

        # Precheck 1 (already checked by check_inputs): file exists and parseable.

        # Precheck 2: schema_version must start with "mvp_aggregate"
        schema_version = aggregate.get("schema_version", "")
        if not isinstance(schema_version, str) or not schema_version.startswith(
            "mvp_aggregate"
        ):
            abort(
                ctx,
                f"aggregate.json schema_version must start with 'mvp_aggregate', "
                f"got {schema_version!r}. "
                "Regenerate: python mvp/pipeline.py multi --events ... --atlas-path ...",
            )

        # Precheck 3: events[] must have at least 1 event
        events = aggregate.get("events", [])
        if not isinstance(events, list) or len(events) == 0:
            abort(
                ctx,
                "aggregate.json must contain at least 1 event in events[]. "
                "Regenerate: python mvp/pipeline.py multi --events ... --atlas-path ...",
            )

        # Precheck 4: joint_posterior.joint_ranked_all must exist and be non-empty
        joint_posterior = aggregate.get("joint_posterior")
        if not isinstance(joint_posterior, dict):
            abort(
                ctx,
                "aggregate.json missing 'joint_posterior' object. "
                "Regenerate: python mvp/pipeline.py multi --events ... --atlas-path ...",
            )
        joint_ranked_all = joint_posterior.get("joint_ranked_all", [])  # type: ignore[union-attr]
        if not isinstance(joint_ranked_all, list) or len(joint_ranked_all) == 0:
            abort(
                ctx,
                "aggregate.json joint_posterior.joint_ranked_all must exist and be non-empty. "
                "Regenerate: python mvp/pipeline.py multi --events ... --atlas-path ...",
            )

        # Precheck 5: each event must have threshold_d2 as a finite float
        for ev in events:
            if not isinstance(ev, dict):
                continue
            event_id = ev.get("event_id", "<unknown>")
            threshold_d2 = ev.get("threshold_d2")
            if not (
                isinstance(threshold_d2, (int, float))
                and math.isfinite(float(threshold_d2))
            ):
                abort(
                    ctx,
                    f"Event '{event_id}' missing threshold_d2 in aggregate.json. "
                    "Regenerate: python mvp/pipeline.py multi --events ... --atlas-path ...",
                )

        aggregate_sha256 = sha256_file(aggregate_path)
        n_events = len(events)

        # Step 2: Build exclusion matrix (top_k geometries)
        matrix_rows = build_exclusion_matrix(events, joint_ranked_all, args.top_k)

        # Step 3-4: Build theory survival summary
        per_theory, survival_summary = build_theory_survival(matrix_rows, n_events)

        # Step 5: Produce outputs

        # Events list (compact form for output)
        events_out = [
            {
                "event_id": ev.get("event_id"),
                "run_id": ev.get("run_id"),
                "threshold_d2": ev.get("threshold_d2"),
            }
            for ev in events
        ]

        # Output 1: exclusion_map.json
        exclusion_map: dict[str, Any] = {
            "schema_version": SCHEMA_EXCLUSION_MAP,
            "run_id": args.run,
            "aggregate_sha256": aggregate_sha256,
            "parameters": {
                "threshold_source": "per_event_from_aggregate",
                "top_k": args.top_k,
            },
            "n_events": n_events,
            "n_geometries_evaluated": len(matrix_rows),
            "events": events_out,
            "matrix": matrix_rows,
            "created": utc_now_iso(),
        }

        # Output 2: theory_survival.json
        theory_survival: dict[str, Any] = {
            "schema_version": SCHEMA_THEORY_SURVIVAL,
            "run_id": args.run,
            "n_events": n_events,
            "n_theories": len(per_theory),
            "summary": survival_summary,
            "per_theory": per_theory,
            "created": utc_now_iso(),
        }

        exclusion_map_path = ctx.outputs_dir / "exclusion_map.json"
        theory_survival_path = ctx.outputs_dir / "theory_survival.json"
        write_json_atomic(exclusion_map_path, exclusion_map)
        write_json_atomic(theory_survival_path, theory_survival)

        n_globally_compatible = sum(
            1 for r in matrix_rows if r["is_globally_compatible"]
        )
        n_excluded_at_least_once = sum(
            1 for r in matrix_rows if r["n_excluded"] > 0
        )

        finalize(
            ctx,
            artifacts={
                "exclusion_map": exclusion_map_path,
                "theory_survival": theory_survival_path,
            },
            results={
                "n_events": n_events,
                "n_geometries_evaluated": len(matrix_rows),
                "n_globally_compatible": n_globally_compatible,
                "n_excluded_at_least_once": n_excluded_at_least_once,
                "n_theories": len(per_theory),
            },
        )
        print(f"OUT_ROOT={ctx.out_root}")
        print(f"STAGE_DIR={ctx.stage_dir}")
        print(f"OUTPUTS_DIR={ctx.outputs_dir}")
        print(f"STAGE_SUMMARY={ctx.stage_dir / 'stage_summary.json'}")
        print(f"MANIFEST={ctx.stage_dir / 'manifest.json'}")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
