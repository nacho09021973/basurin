#!/usr/bin/env python3
"""Temperature sweep over score-based weight policy (delta_lnL softmax).

Recalculates the score-based policy at multiple temperatures T and measures
stability of Renyi diversity, concentration, family/theory mass, and geometry ranking.

This stage is a sensitivity control; it does NOT replace phase4b.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import require_run_valid, write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "experiment/phase5_temperature_sweep"
COMPARISON_ROLE = "temperature_sensitivity_of_score_based_policy"
T_GRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
ALPHA_GRID: list[int | str] = [0, 1, 2, "inf"]
K_GRID = [1, 3, 5, 10]
TOP_N = 20
NORMALIZATION_TOLERANCE = 1e-9
WEIGHT_TOLERANCE = 1e-9
METRIC_TOLERANCE = 1e-6
CENTRAL_T_RANGE = (0.5, 5.0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Temperature sweep over score-based weight policy")
    ap.add_argument("--run-id", required=True)
    return ap.parse_args(argv)


def _coerce_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _require_text(row: dict[str, Any], field_name: str, *, row_index: int, ctx: Any) -> str:
    value = _coerce_text(row.get(field_name)).strip()
    if not value:
        abort(ctx, f"row[{row_index}] missing {field_name}")
        raise AssertionError("unreachable")
    return value


def _load_json_object(path: Path, *, label: str, ctx: Any) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt {label} at {path}: {exc}")
        raise AssertionError("unreachable")
    if not isinstance(payload, dict):
        abort(ctx, f"Invalid {label} at {path}: expected object")
        raise AssertionError("unreachable")
    return payload


def _load_json_list(path: Path, *, label: str, ctx: Any) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt {label} at {path}: {exc}")
        raise AssertionError("unreachable")
    if not isinstance(payload, list):
        abort(ctx, f"Invalid {label} at {path}: expected list")
        raise AssertionError("unreachable")
    return payload


def _load_phase2c_basis(path: Path, ctx: Any) -> dict[str, Any]:
    payload = _load_json_object(path, label="support_ontology_basis_v1", ctx=ctx)
    if payload.get("schema_version") != "support_ontology_basis_v1":
        abort(ctx, f"Expected schema_version=support_ontology_basis_v1 at {path}")
    basis_name = _coerce_text(payload.get("basis_name")).strip()
    if not basis_name:
        abort(ctx, f"support_ontology_basis_v1 at {path} missing basis_name")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        abort(ctx, f"support_ontology_basis_v1 at {path} missing non-empty rows")
    n_rows_declared = payload.get("n_rows")
    if int(n_rows_declared) != len(rows):
        abort(ctx, f"support_ontology_basis_v1 n_rows mismatch at {path}: declared={n_rows_declared} actual={len(rows)}")
    return {"basis_name": basis_name, "rows": rows}


def _load_aggregate_event_runs(path: Path, *, ctx: Any) -> list[dict[str, str]]:
    aggregate = _load_json_object(path, label="aggregate.json", ctx=ctx)
    event_rows = aggregate.get("events")
    if not isinstance(event_rows, list) or not event_rows:
        abort(ctx, f"aggregate.json has no events at {path}")
        raise AssertionError("unreachable")
    resolved: list[dict[str, str]] = []
    for idx, row in enumerate(event_rows):
        if not isinstance(row, dict):
            abort(ctx, f"aggregate.json event row[{idx}] is not an object at {path}")
        event_id = _require_text(row, "event_id", row_index=idx, ctx=ctx)
        source_run_id = _require_text(row, "run_id", row_index=idx, ctx=ctx)
        resolved.append({"event_id": event_id, "source_run_id": source_run_id})
    return resolved


def _renyi_metrics(positive_weights: list[float]) -> dict[str, Any]:
    n_weighted = len(positive_weights)
    p_max = max(positive_weights)
    h_alpha: dict[str, float] = {
        "0": math.log(float(n_weighted)),
        "1": -math.fsum(p * math.log(p) for p in positive_weights),
        "2": -math.log(math.fsum(p * p for p in positive_weights)),
        "inf": -math.log(p_max),
    }
    d_alpha = {key: math.exp(value) for key, value in h_alpha.items()}
    return {
        "H_alpha": h_alpha,
        "D_alpha": d_alpha,
        "p_max": p_max,
        "n_weighted": n_weighted,
    }


def _compute_weights_at_temperature(
    temperature: float,
    *,
    basis_by_raw_geometry_id: dict[str, dict[str, Any]],
    dynamic_event_paths: list[dict[str, Any]],
    ctx: Any,
) -> dict[str, float]:
    """Compute raw weight for each geometry at a given temperature."""
    raw_weight: dict[str, float] = {gid: 0.0 for gid in basis_by_raw_geometry_id}

    for rec in dynamic_event_paths:
        event_id = str(rec["event_id"])
        source_run_id = str(rec["source_run_id"])
        event_support_path = Path(rec["event_support_path"])
        ranked_path = Path(rec["ranked_path"])

        support_payload = _load_json_object(
            event_support_path, label=f"{source_run_id}:event_support_region.json", ctx=ctx
        )
        final_geometry_ids = support_payload.get("final_geometry_ids")
        if not isinstance(final_geometry_ids, list):
            abort(ctx, f"Invalid event_support_region.json at {event_support_path}: final_geometry_ids must be a list")

        restricted_support_ids = [
            _coerce_text(gid).strip()
            for gid in final_geometry_ids
            if _coerce_text(gid).strip() in basis_by_raw_geometry_id
        ]
        if not restricted_support_ids:
            abort(
                ctx,
                f"event_support_region.json has no overlap with phase2c basis: "
                f"event_id={event_id} source_run_id={source_run_id}",
            )

        ranked_rows = _load_json_list(ranked_path, label=f"{source_run_id}:ranked_all_full.json", ctx=ctx)
        delta_lnL_by_geometry_id: dict[str, float] = {}
        for row in ranked_rows:
            if not isinstance(row, dict):
                continue
            geometry_id = _coerce_text(row.get("geometry_id")).strip()
            if not geometry_id:
                continue
            delta_lnL = row.get("delta_lnL")
            try:
                if delta_lnL is None:
                    continue
                delta_lnL_by_geometry_id[geometry_id] = float(delta_lnL)
            except Exception:
                continue

        missing_delta = [
            gid for gid in restricted_support_ids if gid not in delta_lnL_by_geometry_id
        ]
        if missing_delta:
            abort(
                ctx,
                f"Missing delta_lnL for event-supported geometries: "
                f"event_id={event_id} source_run_id={source_run_id} "
                f"missing_geometry_ids={missing_delta[:5]} total_missing={len(missing_delta)}",
            )

        event_deltas = [delta_lnL_by_geometry_id[gid] for gid in restricted_support_ids]
        max_delta_lnL_event = max(event_deltas)

        local_scores = {
            gid: math.exp((delta_lnL_by_geometry_id[gid] - max_delta_lnL_event) / temperature)
            for gid in restricted_support_ids
        }
        local_score_sum = math.fsum(local_scores.values())
        if local_score_sum <= 0.0:
            abort(ctx, f"Non-positive local score sum at T={temperature}: event_id={event_id}")

        for gid in restricted_support_ids:
            raw_weight[gid] += local_scores[gid] / local_score_sum

    return raw_weight


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "temperature_grid": T_GRID,
            "alpha_grid": ALPHA_GRID,
            "k_grid": K_GRID,
            "top_n": TOP_N,
        },
    )

    # --- resolve inputs ---
    source_basis_rel = Path("experiment") / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
    source_basis_path = ctx.run_dir / source_basis_rel
    aggregate_rel = Path("s5_aggregate") / "outputs" / "aggregate.json"
    aggregate_path = ctx.run_dir / aggregate_rel

    existing_weight_policy_rel = (
        Path("experiment") / "phase3_weight_policy_basis" / "outputs"
        / "weight_policy_event_support_delta_lnL_softmax_mean_v1.json"
    )
    existing_weight_policy_path = ctx.run_dir / existing_weight_policy_rel

    existing_renyi_rel = (
        Path("experiment") / "phase4_renyi_diversity_baseline" / "outputs"
        / "renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json"
    )
    existing_renyi_path = ctx.run_dir / existing_renyi_rel

    runtime_inputs: dict[str, Path] = {
        "support_ontology_basis_v1": source_basis_path,
        "aggregate": aggregate_path,
        "existing_weight_policy": existing_weight_policy_path,
        "existing_renyi_baseline": existing_renyi_path,
    }

    # --- resolve event inputs ---
    if not aggregate_path.exists():
        abort(ctx, f"Missing required input: expected_path={aggregate_path}")
    aggregate_event_runs = _load_aggregate_event_runs(aggregate_path, ctx=ctx)
    dynamic_event_paths: list[dict[str, Any]] = []

    for rec in aggregate_event_runs:
        event_id = rec["event_id"]
        source_run_id = rec["source_run_id"]
        try:
            require_run_valid(ctx.out_root, source_run_id)
        except Exception as exc:
            abort(ctx, f"Source run not PASS: source_run_id={source_run_id}; detail={exc}")

        event_support_path = ctx.out_root / source_run_id / "s4k_event_support_region" / "outputs" / "event_support_region.json"
        ranked_path = ctx.out_root / source_run_id / "s4_geometry_filter" / "outputs" / "ranked_all_full.json"
        if not event_support_path.exists():
            abort(ctx, f"Missing event_support_region.json: event_id={event_id} expected_path={event_support_path}")
        if not ranked_path.exists():
            abort(ctx, f"Missing ranked_all_full.json: event_id={event_id} expected_path={ranked_path}")

        runtime_inputs[f"{source_run_id}:event_support_region"] = event_support_path
        runtime_inputs[f"{source_run_id}:ranked_all_full"] = ranked_path
        dynamic_event_paths.append({
            "event_id": event_id,
            "source_run_id": source_run_id,
            "event_support_path": event_support_path,
            "ranked_path": ranked_path,
        })

    check_inputs(ctx, runtime_inputs)

    # --- load basis ---
    phase2c = _load_phase2c_basis(source_basis_path, ctx)
    basis_name = phase2c["basis_name"]
    basis_rows = phase2c["rows"]
    n_rows = len(basis_rows)

    basis_by_raw_geometry_id: dict[str, dict[str, Any]] = {}
    for idx, brow in enumerate(basis_rows):
        raw_geometry_id = _require_text(brow, "raw_geometry_id", row_index=idx, ctx=ctx)
        if raw_geometry_id in basis_by_raw_geometry_id:
            abort(ctx, f"Duplicate raw_geometry_id={raw_geometry_id!r} in phase2c basis")
        basis_by_raw_geometry_id[raw_geometry_id] = {
            "raw_geometry_id": raw_geometry_id,
            "normalized_geometry_id": _require_text(brow, "normalized_geometry_id", row_index=idx, ctx=ctx),
            "atlas_family": _require_text(brow, "atlas_family", row_index=idx, ctx=ctx),
            "atlas_theory": _require_text(brow, "atlas_theory", row_index=idx, ctx=ctx),
        }

    families = sorted({m["atlas_family"] for m in basis_by_raw_geometry_id.values()})
    theories = sorted({m["atlas_theory"] for m in basis_by_raw_geometry_id.values()})

    # --- load existing policy for T=1 cross-check ---
    existing_weight_payload = _load_json_object(
        existing_weight_policy_path, label="existing_weight_policy", ctx=ctx
    )
    existing_weights_by_gid: dict[str, float] = {}
    for row in existing_weight_payload.get("rows", []):
        gid = _coerce_text(row.get("raw_geometry_id")).strip()
        if gid:
            existing_weights_by_gid[gid] = float(row["weight_normalized"])

    existing_renyi_payload = _load_json_object(
        existing_renyi_path, label="existing_renyi_baseline", ctx=ctx
    )
    existing_d_alpha = existing_renyi_payload.get("metrics", {}).get("D_alpha", {})

    # --- sweep ---
    metrics_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    family_mass_rows: list[dict[str, Any]] = []
    theory_mass_rows: list[dict[str, Any]] = []
    top_geometry_rows: list[dict[str, Any]] = []

    d1_at_t1: float | None = None
    t1_weight_match = False
    t1_renyi_match = False
    per_temperature_gate_status: dict[str, str] = {}

    for temperature in T_GRID:
        policy_name = f"delta_lnL_softmax_T{temperature}"

        raw_weight = _compute_weights_at_temperature(
            temperature,
            basis_by_raw_geometry_id=basis_by_raw_geometry_id,
            dynamic_event_paths=dynamic_event_paths,
            ctx=ctx,
        )

        weight_sum_raw = math.fsum(raw_weight.values())
        if weight_sum_raw <= 0.0:
            abort(ctx, f"Non-positive global raw weight sum at T={temperature}")

        weight_normalized: dict[str, float] = {
            gid: raw_weight[gid] / weight_sum_raw for gid in raw_weight
        }

        weight_sum_normalized = math.fsum(weight_normalized.values())
        if not math.isclose(weight_sum_normalized, 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE):
            abort(ctx, f"Normalization failure at T={temperature}: sum={weight_sum_normalized}")

        positive_weights = [w for w in weight_normalized.values() if w > 0.0]
        if not positive_weights:
            abort(ctx, f"No positive weights at T={temperature}")

        renyi = _renyi_metrics(positive_weights)
        d_alpha = renyi["D_alpha"]
        h_alpha = renyi["H_alpha"]

        metrics_rows.append({
            "temperature": temperature,
            "policy_name": policy_name,
            "normalization_method": "softmax_per_event_sum_then_global_normalize",
            "n_rows": n_rows,
            "coverage_fraction": 1.0,
            "n_weighted": renyi["n_weighted"],
            "n_unweighted": n_rows - renyi["n_weighted"],
            "p_max": renyi["p_max"],
            "H_0": h_alpha["0"],
            "H_1": h_alpha["1"],
            "H_2": h_alpha["2"],
            "H_inf": h_alpha["inf"],
            "D_0": d_alpha["0"],
            "D_1": d_alpha["1"],
            "D_2": d_alpha["2"],
            "D_inf": d_alpha["inf"],
            "weight_sum_raw": weight_sum_raw,
            "weight_sum_normalized": weight_sum_normalized,
            "criterion": "delta_lnL_softmax_temperature_sweep",
            "criterion_version": "v1",
        })

        if temperature == 1.0:
            d1_at_t1 = d_alpha["1"]

        # --- build sorted rows for top-k / ranking ---
        sorted_geom_rows: list[dict[str, Any]] = []
        for gid, meta in basis_by_raw_geometry_id.items():
            sorted_geom_rows.append({
                "raw_geometry_id": gid,
                "normalized_geometry_id": meta["normalized_geometry_id"],
                "atlas_family": meta["atlas_family"],
                "atlas_theory": meta["atlas_theory"],
                "weight_normalized": weight_normalized[gid],
            })
        sorted_geom_rows.sort(key=lambda r: (-r["weight_normalized"], r["raw_geometry_id"]))

        total_mass = math.fsum(r["weight_normalized"] for r in sorted_geom_rows)

        # top-k
        for top_k in K_GRID:
            cumulative_mass = math.fsum(r["weight_normalized"] for r in sorted_geom_rows[:top_k])
            topk_rows.append({
                "temperature": temperature,
                "top_k": top_k,
                "cumulative_mass": cumulative_mass,
                "normalization_check": total_mass,
                "criterion": "cumulative_topk_mass_over_weight_normalized",
                "criterion_version": "v1",
            })

        # family mass
        family_bucket: dict[str, float] = {}
        family_count: dict[str, int] = {}
        for r in sorted_geom_rows:
            fam = r["atlas_family"]
            family_bucket[fam] = family_bucket.get(fam, 0.0) + r["weight_normalized"]
            family_count[fam] = family_count.get(fam, 0) + 1
        ordered_families = sorted(family_bucket.items(), key=lambda item: (-item[1], item[0]))
        for rank, (fam, mass) in enumerate(ordered_families, start=1):
            family_mass_rows.append({
                "temperature": temperature,
                "atlas_family": fam,
                "family_mass": mass,
                "family_count": family_count[fam],
                "family_mass_rank": rank,
                "criterion": "mass_sum_by_atlas_family_temperature_sweep",
                "criterion_version": "v1",
            })

        # theory mass
        theory_bucket: dict[str, float] = {}
        theory_count: dict[str, int] = {}
        for r in sorted_geom_rows:
            th = r["atlas_theory"]
            theory_bucket[th] = theory_bucket.get(th, 0.0) + r["weight_normalized"]
            theory_count[th] = theory_count.get(th, 0) + 1
        ordered_theories = sorted(theory_bucket.items(), key=lambda item: (-item[1], item[0]))
        for rank, (th, mass) in enumerate(ordered_theories, start=1):
            theory_mass_rows.append({
                "temperature": temperature,
                "atlas_theory": th,
                "theory_mass": mass,
                "theory_count": theory_count[th],
                "theory_mass_rank": rank,
                "criterion": "mass_sum_by_atlas_theory_temperature_sweep",
                "criterion_version": "v1",
            })

        # top geometry ranking
        cumulative_mass = 0.0
        for rank, r in enumerate(sorted_geom_rows[:TOP_N], start=1):
            cumulative_mass += r["weight_normalized"]
            top_geometry_rows.append({
                "temperature": temperature,
                "rank": rank,
                "raw_geometry_id": r["raw_geometry_id"],
                "normalized_geometry_id": r["normalized_geometry_id"],
                "atlas_family": r["atlas_family"],
                "atlas_theory": r["atlas_theory"],
                "weight_normalized": r["weight_normalized"],
                "cumulative_mass": cumulative_mass,
                "criterion": "weight_normalized_descending_temperature_sweep",
                "criterion_version": "v1",
            })

        # --- T=1.0 cross-check ---
        if temperature == 1.0:
            max_weight_diff = 0.0
            for gid in basis_by_raw_geometry_id:
                existing_w = existing_weights_by_gid.get(gid, 0.0)
                computed_w = weight_normalized[gid]
                max_weight_diff = max(max_weight_diff, abs(existing_w - computed_w))

            t1_weight_match = max_weight_diff <= WEIGHT_TOLERANCE
            if not t1_weight_match:
                abort(
                    ctx,
                    f"T=1.0 cross-check FAILED: max weight discrepancy={max_weight_diff} "
                    f"exceeds tolerance={WEIGHT_TOLERANCE}",
                )

            max_metric_diff = 0.0
            for alpha_key in ["0", "1", "2", "inf"]:
                existing_d = float(existing_d_alpha.get(alpha_key, 0.0))
                computed_d = d_alpha[alpha_key]
                max_metric_diff = max(max_metric_diff, abs(existing_d - computed_d))
            t1_renyi_match = max_metric_diff <= METRIC_TOLERANCE
            if not t1_renyi_match:
                abort(
                    ctx,
                    f"T=1.0 Renyi cross-check FAILED: max D_alpha discrepancy={max_metric_diff} "
                    f"exceeds tolerance={METRIC_TOLERANCE}",
                )

    # --- gates ---
    assert d1_at_t1 is not None
    d1_lower = 0.5 * d1_at_t1
    d1_upper = 2.0 * d1_at_t1
    overall_gate = True

    for mr in metrics_rows:
        t = mr["temperature"]
        if CENTRAL_T_RANGE[0] <= t <= CENTRAL_T_RANGE[1]:
            gate_d1 = d1_lower <= mr["D_1"] <= d1_upper

            # find family mass for this temperature
            edgb_mass = 0.0
            kn_mass = 0.0
            dcs_mass = 0.0
            for fmr in family_mass_rows:
                if fmr["temperature"] == t:
                    if fmr["atlas_family"] == "edgb":
                        edgb_mass = fmr["family_mass"]
                    elif fmr["atlas_family"] == "kerr_newman":
                        kn_mass = fmr["family_mass"]
                    elif fmr["atlas_family"] == "dcs":
                        dcs_mass = fmr["family_mass"]

            gate_dominant = (edgb_mass + kn_mass) > 0.5
            gate_dcs = dcs_mass < 0.25
            gate_pass = gate_d1 and gate_dominant and gate_dcs
            per_temperature_gate_status[str(t)] = "PASS" if gate_pass else "FAIL"
            if not gate_pass:
                overall_gate = False
        else:
            per_temperature_gate_status[str(t)] = "OUTSIDE_CENTRAL_RANGE"

    overall_gate_status = "PASS" if overall_gate else "FAIL"

    gates = {
        "central_temperature_range": list(CENTRAL_T_RANGE),
        "d1_reference_at_t1": d1_at_t1,
        "d1_lower_bound": d1_lower,
        "d1_upper_bound": d1_upper,
        "edgb_plus_kerr_newman_min_mass": 0.5,
        "dcs_max_mass": 0.25,
        "per_temperature_gate_status": per_temperature_gate_status,
        "overall_gate_status": overall_gate_status,
    }

    # --- build output payloads ---
    metrics_payload = {
        "schema_version": "temperature_sweep_metrics_v1",
        "comparison_role": COMPARISON_ROLE,
        "basis_name": basis_name,
        "temperature_grid": T_GRID,
        "alpha_grid": ALPHA_GRID,
        "rows": metrics_rows,
        "gates": gates,
        "notes": [
            "Temperature sensitivity of delta_lnL softmax score-based policy.",
            "T=1.0 reproduces the closed phase3 policy within tolerance.",
            "This output is not a black-hole thermodynamic entropy claim.",
        ],
    }

    topk_payload = {
        "schema_version": "temperature_sweep_topk_v1",
        "basis_name": basis_name,
        "temperature_grid": T_GRID,
        "k_grid": K_GRID,
        "rows": topk_rows,
    }

    family_payload = {
        "schema_version": "temperature_sweep_family_mass_v1",
        "basis_name": basis_name,
        "temperature_grid": T_GRID,
        "rows": family_mass_rows,
    }

    theory_payload = {
        "schema_version": "temperature_sweep_theory_mass_v1",
        "basis_name": basis_name,
        "temperature_grid": T_GRID,
        "rows": theory_mass_rows,
    }

    top_geometry_payload = {
        "schema_version": "temperature_sweep_top_geometry_v1",
        "basis_name": basis_name,
        "temperature_grid": T_GRID,
        "top_n": TOP_N,
        "rows": top_geometry_rows,
    }

    # --- write outputs ---
    metrics_path = ctx.outputs_dir / "temperature_sweep_metrics_v1.json"
    topk_path = ctx.outputs_dir / "temperature_sweep_topk_v1.json"
    family_path = ctx.outputs_dir / "temperature_sweep_family_mass_v1.json"
    theory_path = ctx.outputs_dir / "temperature_sweep_theory_mass_v1.json"
    top_geometry_path = ctx.outputs_dir / "temperature_sweep_top_geometry_v1.json"

    write_json_atomic(metrics_path, metrics_payload)
    write_json_atomic(topk_path, topk_payload)
    write_json_atomic(family_path, family_payload)
    write_json_atomic(theory_path, theory_payload)
    write_json_atomic(top_geometry_path, top_geometry_payload)

    # --- finalize ---
    finalize(
        ctx,
        artifacts={
            "temperature_sweep_metrics_v1": metrics_path,
            "temperature_sweep_topk_v1": topk_path,
            "temperature_sweep_family_mass_v1": family_path,
            "temperature_sweep_theory_mass_v1": theory_path,
            "temperature_sweep_top_geometry_v1": top_geometry_path,
        },
        results={
            "n_rows": n_rows,
            "family_count": len(families),
            "theory_count": len(theories),
        },
        extra_summary={
            "schema_version": "temperature_sweep_v1",
            "comparison_role": COMPARISON_ROLE,
            "basis_name": basis_name,
            "temperature_grid": T_GRID,
            "alpha_grid": ALPHA_GRID,
            "k_grid": K_GRID,
            "top_n": TOP_N,
            "n_rows": n_rows,
            "family_count": len(families),
            "theory_count": len(theories),
            "t1_weight_match_with_existing_policy": t1_weight_match,
            "t1_renyi_match_with_existing_baseline": t1_renyi_match,
            "tolerance_weight": WEIGHT_TOLERANCE,
            "tolerance_metric": METRIC_TOLERANCE,
            "overall_gate_status": overall_gate_status,
            "verdict": "PASS",
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
