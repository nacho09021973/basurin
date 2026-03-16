#!/usr/bin/env python3
"""Temperature sweep for the score-based delta_lnL softmax policy."""
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
METRICS_SCHEMA_VERSION = "temperature_sweep_metrics_v1"
TOPK_SCHEMA_VERSION = "temperature_sweep_topk_v1"
FAMILY_SCHEMA_VERSION = "temperature_sweep_family_mass_v1"
THEORY_SCHEMA_VERSION = "temperature_sweep_theory_mass_v1"
TOP_GEOMETRY_SCHEMA_VERSION = "temperature_sweep_top_geometry_v1"
SUPPORT_SCHEMA_VERSION = "support_ontology_basis_v1"
WEIGHT_POLICY_SCHEMA_VERSION = "weight_policy_basis_v1"
RENYI_SCHEMA_VERSION = "renyi_diversity_baseline_v1"
COMPARISON_ROLE = "temperature_sensitivity_of_score_based_policy"
SWEEP_POLICY_NAME = "event_support_delta_lnL_softmax_mean_temperature_v1"
EXISTING_POLICY_NAME = "event_support_delta_lnL_softmax_mean_v1"
TEMPERATURE_GRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
ALPHA_GRID: list[int | str] = [0, 1, 2, "inf"]
K_GRID = [1, 3, 5, 10]
TOP_N = 20
TOLERANCE_WEIGHT = 1e-9
TOLERANCE_METRIC = 1e-9

SUPPORT_INPUT_REL = Path("experiment") / "phase2c_support_ontology_basis" / "outputs" / "support_ontology_basis_v1.json"
AGGREGATE_INPUT_REL = Path("s5_aggregate") / "outputs" / "aggregate.json"
EXISTING_WEIGHT_INPUT_REL = (
    Path("experiment")
    / "phase3_weight_policy_basis"
    / "outputs"
    / "weight_policy_event_support_delta_lnL_softmax_mean_v1.json"
)
EXISTING_RENYI_INPUT_REL = (
    Path("experiment")
    / "phase4_renyi_diversity_baseline"
    / "outputs"
    / "renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json"
)

METRICS_OUTPUT_NAME = "temperature_sweep_metrics_v1.json"
TOPK_OUTPUT_NAME = "temperature_sweep_topk_v1.json"
FAMILY_OUTPUT_NAME = "temperature_sweep_family_mass_v1.json"
THEORY_OUTPUT_NAME = "temperature_sweep_theory_mass_v1.json"
TOP_GEOMETRY_OUTPUT_NAME = "temperature_sweep_top_geometry_v1.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep temperature over the delta_lnL score-based softmax policy")
    ap.add_argument("--run-id", required=True)
    return ap.parse_args(argv)


def _coerce_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


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


def _require_text(row: dict[str, Any], field_name: str, *, row_index: int, ctx: Any) -> str:
    value = _coerce_text(row.get(field_name)).strip()
    if not value:
        abort(ctx, f"row[{row_index}] missing {field_name}")
        raise AssertionError("unreachable")
    return value


def _require_number(row: dict[str, Any], field_name: str, *, row_index: int, ctx: Any) -> float:
    try:
        return float(row.get(field_name))
    except Exception:
        abort(ctx, f"row[{row_index}] missing numeric {field_name}")
        raise AssertionError("unreachable")


def _require_positive_int(row: dict[str, Any], field_name: str, *, row_index: int, ctx: Any) -> int:
    value = int(_require_number(row, field_name, row_index=row_index, ctx=ctx))
    if value <= 0:
        abort(ctx, f"row[{row_index}] has non-positive {field_name}={value}")
        raise AssertionError("unreachable")
    return value


def _regen_multimode_cmd(event_id: str, source_run_id: str) -> str:
    return (
        f"python -m mvp.pipeline multimode --event-id {event_id} "
        f"--run-id {source_run_id} --atlas-default --offline --estimator dual"
    )


def _load_support_basis(path: Path, *, ctx: Any) -> dict[str, Any]:
    payload = _load_json_object(path, label="support_ontology_basis_v1", ctx=ctx)
    if payload.get("schema_version") != SUPPORT_SCHEMA_VERSION:
        abort(ctx, f"Expected schema_version={SUPPORT_SCHEMA_VERSION} at {path}")
    basis_name = _coerce_text(payload.get("basis_name")).strip()
    if not basis_name:
        abort(ctx, f"support_ontology_basis_v1 missing basis_name at {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        abort(ctx, f"support_ontology_basis_v1 missing non-empty rows at {path}")
    try:
        n_rows = int(payload.get("n_rows"))
    except Exception:
        abort(ctx, f"support_ontology_basis_v1 missing integer n_rows at {path}")
        raise AssertionError("unreachable")
    if n_rows != len(rows):
        abort(ctx, f"support_ontology_basis_v1 n_rows mismatch at {path}: declared={n_rows} actual={len(rows)}")

    ordered_rows: list[dict[str, Any]] = []
    basis_by_geometry_id: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            abort(ctx, f"support_ontology_basis_v1 row[{idx}] is not an object at {path}")
        raw_geometry_id = _require_text(row, "raw_geometry_id", row_index=idx, ctx=ctx)
        if raw_geometry_id in basis_by_geometry_id:
            abort(ctx, f"Duplicate raw_geometry_id={raw_geometry_id!r} in support_ontology_basis_v1 at {path}")
        ordered_row = {
            "raw_geometry_id": raw_geometry_id,
            "normalized_geometry_id": _require_text(row, "normalized_geometry_id", row_index=idx, ctx=ctx),
            "atlas_family": _require_text(row, "atlas_family", row_index=idx, ctx=ctx),
            "atlas_theory": _require_text(row, "atlas_theory", row_index=idx, ctx=ctx),
            "support_count_events": _require_positive_int(row, "n_events_supported", row_index=idx, ctx=ctx),
            "support_fraction_events": _require_number(row, "support_fraction_events", row_index=idx, ctx=ctx),
        }
        ordered_rows.append(ordered_row)
        basis_by_geometry_id[raw_geometry_id] = ordered_row
    return {
        "basis_name": basis_name,
        "n_rows": n_rows,
        "ordered_rows": ordered_rows,
        "basis_by_geometry_id": basis_by_geometry_id,
    }


def _load_aggregate_event_runs(path: Path, *, ctx: Any) -> list[dict[str, str]]:
    payload = _load_json_object(path, label="aggregate.json", ctx=ctx)
    rows = payload.get("events")
    if not isinstance(rows, list) or not rows:
        abort(ctx, f"aggregate.json has no events at {path}")
    event_runs: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            abort(ctx, f"aggregate.json event row[{idx}] is not an object at {path}")
        event_runs.append(
            {
                "event_id": _require_text(row, "event_id", row_index=idx, ctx=ctx),
                "source_run_id": _require_text(row, "run_id", row_index=idx, ctx=ctx),
            }
        )
    return event_runs


def _load_existing_weight_policy(path: Path, *, ctx: Any) -> dict[str, Any]:
    payload = _load_json_object(path, label="weight_policy_event_support_delta_lnL_softmax_mean_v1", ctx=ctx)
    if payload.get("schema_version") != WEIGHT_POLICY_SCHEMA_VERSION:
        abort(ctx, f"Expected schema_version={WEIGHT_POLICY_SCHEMA_VERSION} at {path}")
    policy_name = _coerce_text(payload.get("policy_name")).strip()
    if policy_name != EXISTING_POLICY_NAME:
        abort(ctx, f"Unexpected policy_name in {path}: expected={EXISTING_POLICY_NAME!r} observed={policy_name!r}")
    basis_name = _coerce_text(payload.get("basis_name")).strip()
    if not basis_name:
        abort(ctx, f"Existing weight policy missing basis_name at {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        abort(ctx, f"Existing weight policy missing non-empty rows at {path}")
    try:
        n_rows = int(payload.get("n_rows"))
    except Exception:
        abort(ctx, f"Existing weight policy missing integer n_rows at {path}")
        raise AssertionError("unreachable")
    if n_rows != len(rows):
        abort(ctx, f"Existing weight policy n_rows mismatch at {path}: declared={n_rows} actual={len(rows)}")
    try:
        weight_sum_normalized = float(payload.get("weight_sum_normalized"))
    except Exception:
        abort(ctx, f"Existing weight policy missing numeric weight_sum_normalized at {path}")
        raise AssertionError("unreachable")
    if not math.isclose(weight_sum_normalized, 1.0, rel_tol=0.0, abs_tol=TOLERANCE_WEIGHT):
        abort(ctx, f"Existing weight policy not normalized at {path}: weight_sum_normalized={weight_sum_normalized}")

    rows_by_geometry_id: dict[str, dict[str, Any]] = {}
    observed_sum = 0.0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            abort(ctx, f"Existing weight policy row[{idx}] is not an object at {path}")
        raw_geometry_id = _require_text(row, "raw_geometry_id", row_index=idx, ctx=ctx)
        if raw_geometry_id in rows_by_geometry_id:
            abort(ctx, f"Duplicate raw_geometry_id={raw_geometry_id!r} in existing weight policy at {path}")
        try:
            weight_normalized = float(row.get("weight_normalized"))
            weight_raw = float(row.get("weight_raw"))
        except Exception:
            abort(ctx, f"Existing weight policy row[{idx}] missing numeric weights at {path}")
            raise AssertionError("unreachable")
        if not math.isfinite(weight_normalized) or weight_normalized < 0.0:
            abort(ctx, f"Existing weight policy row[{idx}] has invalid weight_normalized={weight_normalized!r} at {path}")
        if not math.isfinite(weight_raw) or weight_raw < 0.0:
            abort(ctx, f"Existing weight policy row[{idx}] has invalid weight_raw={weight_raw!r} at {path}")
        observed_sum += weight_normalized
        rows_by_geometry_id[raw_geometry_id] = {
            "raw_geometry_id": raw_geometry_id,
            "normalized_geometry_id": _require_text(row, "normalized_geometry_id", row_index=idx, ctx=ctx),
            "atlas_family": _require_text(row, "atlas_family", row_index=idx, ctx=ctx),
            "atlas_theory": _require_text(row, "atlas_theory", row_index=idx, ctx=ctx),
            "weight_raw": weight_raw,
            "weight_normalized": weight_normalized,
        }
    if not math.isclose(observed_sum, 1.0, rel_tol=0.0, abs_tol=TOLERANCE_WEIGHT):
        abort(ctx, f"Existing weight policy row sum mismatch at {path}: observed={observed_sum}")
    return {
        "basis_name": basis_name,
        "n_rows": n_rows,
        "rows_by_geometry_id": rows_by_geometry_id,
        "weight_sum_normalized": weight_sum_normalized,
    }


def _load_existing_renyi_baseline(path: Path, *, ctx: Any) -> dict[str, Any]:
    payload = _load_json_object(path, label="renyi_diversity_event_support_delta_lnL_softmax_mean_v1", ctx=ctx)
    if payload.get("schema_version") != RENYI_SCHEMA_VERSION:
        abort(ctx, f"Expected schema_version={RENYI_SCHEMA_VERSION} at {path}")
    policy_name = _coerce_text(payload.get("policy_name")).strip()
    if policy_name != EXISTING_POLICY_NAME:
        abort(ctx, f"Unexpected policy_name in {path}: expected={EXISTING_POLICY_NAME!r} observed={policy_name!r}")
    basis_name = _coerce_text(payload.get("basis_name")).strip()
    if not basis_name:
        abort(ctx, f"Existing Renyi baseline missing basis_name at {path}")
    if payload.get("alpha_grid") != ALPHA_GRID:
        abort(ctx, f"Existing Renyi baseline alpha_grid mismatch at {path}: observed={payload.get('alpha_grid')!r}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        abort(ctx, f"Existing Renyi baseline missing metrics at {path}")
    d_alpha = metrics.get("D_alpha")
    h_alpha = metrics.get("H_alpha")
    if not isinstance(d_alpha, dict) or not isinstance(h_alpha, dict):
        abort(ctx, f"Existing Renyi baseline missing H_alpha/D_alpha at {path}")
    return payload


def _validate_basis_against_existing(
    basis: dict[str, Any],
    existing_weight_policy: dict[str, Any],
    existing_renyi_baseline: dict[str, Any],
    *,
    ctx: Any,
) -> None:
    basis_name = basis["basis_name"]
    if existing_weight_policy["basis_name"] != basis_name:
        abort(
            ctx,
            f"basis_name mismatch between support basis and existing weight policy: "
            f"support={basis_name!r} existing={existing_weight_policy['basis_name']!r}",
        )
    renyi_basis_name = _coerce_text(existing_renyi_baseline.get("basis_name")).strip()
    if renyi_basis_name != basis_name:
        abort(
            ctx,
            f"basis_name mismatch between support basis and existing Renyi baseline: "
            f"support={basis_name!r} existing={renyi_basis_name!r}",
        )
    if existing_weight_policy["n_rows"] != basis["n_rows"]:
        abort(
            ctx,
            f"n_rows mismatch between support basis and existing weight policy: "
            f"support={basis['n_rows']} existing={existing_weight_policy['n_rows']}",
        )
    if int(existing_renyi_baseline.get("n_rows")) != basis["n_rows"]:
        abort(
            ctx,
            f"n_rows mismatch between support basis and existing Renyi baseline: "
            f"support={basis['n_rows']} existing={existing_renyi_baseline.get('n_rows')}",
        )
    basis_keys = {row["raw_geometry_id"] for row in basis["ordered_rows"]}
    existing_keys = set(existing_weight_policy["rows_by_geometry_id"])
    if basis_keys != existing_keys:
        abort(
            ctx,
            f"geometry key set mismatch between support basis and existing weight policy: "
            f"missing={sorted(basis_keys - existing_keys)[:5]} extra={sorted(existing_keys - basis_keys)[:5]}",
        )
    for row in basis["ordered_rows"]:
        existing_row = existing_weight_policy["rows_by_geometry_id"][row["raw_geometry_id"]]
        observed_meta = (
            existing_row["normalized_geometry_id"],
            existing_row["atlas_family"],
            existing_row["atlas_theory"],
        )
        expected_meta = (
            row["normalized_geometry_id"],
            row["atlas_family"],
            row["atlas_theory"],
        )
        if observed_meta != expected_meta:
            abort(
                ctx,
                f"geometry metadata mismatch for raw_geometry_id={row['raw_geometry_id']!r}: "
                f"support={expected_meta!r} existing={observed_meta!r}",
            )


def _load_event_records(
    *,
    out_root: Path,
    aggregate_event_runs: list[dict[str, str]],
    basis_by_geometry_id: dict[str, dict[str, Any]],
    ctx: Any,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    event_records: list[dict[str, Any]] = []
    contributing_event_ids_by_geometry_id = {geometry_id: [] for geometry_id in basis_by_geometry_id}
    source_artifacts_by_geometry_id = {
        geometry_id: [str(SUPPORT_INPUT_REL), str(AGGREGATE_INPUT_REL)]
        for geometry_id in basis_by_geometry_id
    }
    runtime_inputs: dict[str, Path] = {}

    for rec in aggregate_event_runs:
        event_id = rec["event_id"]
        source_run_id = rec["source_run_id"]
        try:
            require_run_valid(out_root, source_run_id)
        except Exception as exc:
            abort(
                ctx,
                f"Source run not PASS for temperature sweep: "
                f"expected_path={out_root / source_run_id / 'RUN_VALID' / 'verdict.json'}; "
                f"regen_cmd='{_regen_multimode_cmd(event_id, source_run_id)}'; detail={exc}",
            )

        event_support_rel = Path(source_run_id) / "s4k_event_support_region" / "outputs" / "event_support_region.json"
        ranked_rel = Path(source_run_id) / "s4_geometry_filter" / "outputs" / "ranked_all_full.json"
        event_support_path = out_root / event_support_rel
        ranked_path = out_root / ranked_rel

        if not event_support_path.exists():
            abort(
                ctx,
                f"Missing required input for temperature sweep: expected_path={event_support_path}; "
                f"regen_cmd='{_regen_multimode_cmd(event_id, source_run_id)}'",
            )
        if not ranked_path.exists():
            abort(
                ctx,
                f"Missing required input for temperature sweep: expected_path={ranked_path}; "
                f"regen_cmd='{_regen_multimode_cmd(event_id, source_run_id)}'",
            )

        support_payload = _load_json_object(event_support_path, label=f"{source_run_id}:event_support_region.json", ctx=ctx)
        final_geometry_ids = support_payload.get("final_geometry_ids")
        if not isinstance(final_geometry_ids, list):
            abort(ctx, f"Invalid event_support_region.json at {event_support_path}: final_geometry_ids must be a list")
        restricted_support_ids = [
            _coerce_text(geometry_id).strip()
            for geometry_id in final_geometry_ids
            if _coerce_text(geometry_id).strip() in basis_by_geometry_id
        ]
        if not restricted_support_ids:
            abort(
                ctx,
                f"event_support_region.json has no overlap with support basis: "
                f"event_id={event_id} source_run_id={source_run_id} expected_path={event_support_path}",
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
            geometry_id for geometry_id in restricted_support_ids if geometry_id not in delta_lnL_by_geometry_id
        ]
        if missing_delta:
            abort(
                ctx,
                f"Missing delta_lnL for event-supported geometries: "
                f"event_id={event_id} source_run_id={source_run_id} expected_path={ranked_path} "
                f"missing_geometry_ids={missing_delta[:5]} total_missing={len(missing_delta)}",
            )

        for geometry_id in restricted_support_ids:
            contributing_event_ids_by_geometry_id[geometry_id].append(event_id)
            source_artifacts_by_geometry_id[geometry_id].extend([str(event_support_rel), str(ranked_rel)])

        event_records.append(
            {
                "event_id": event_id,
                "source_run_id": source_run_id,
                "event_support_rel": str(event_support_rel),
                "ranked_rel": str(ranked_rel),
                "support_ids": restricted_support_ids,
                "delta_lnL_by_geometry_id": {
                    geometry_id: delta_lnL_by_geometry_id[geometry_id] for geometry_id in restricted_support_ids
                },
                "max_delta_lnL": max(delta_lnL_by_geometry_id[geometry_id] for geometry_id in restricted_support_ids),
            }
        )

        runtime_inputs[f"{source_run_id}:event_support_region"] = event_support_path
        runtime_inputs[f"{source_run_id}:ranked_all_full"] = ranked_path

    return event_records, contributing_event_ids_by_geometry_id, source_artifacts_by_geometry_id, runtime_inputs


def _normalize_input_records_to_out_root(ctx: Any, runtime_inputs: dict[str, Path]) -> None:
    runtime_input_by_label = dict(runtime_inputs)
    for rec in ctx.inputs_record:
        label = rec.get("label", "")
        input_path = runtime_input_by_label.get(label)
        if input_path is None:
            continue
        if not Path(str(rec.get("path", ""))).is_absolute():
            continue
        try:
            rec["path"] = str(input_path.relative_to(ctx.out_root))
        except ValueError:
            continue


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row["weight_normalized"]), row["raw_geometry_id"]))


def _renyi_metrics_from_weights(weights: list[float], *, n_rows: int) -> dict[str, Any]:
    positive_weights = [weight for weight in weights if weight > 0.0]
    if not positive_weights:
        raise ValueError("No positive weights")
    h_alpha = {
        "0": math.log(float(len(positive_weights))),
        "1": -math.fsum(weight * math.log(weight) for weight in positive_weights),
        "2": -math.log(math.fsum(weight * weight for weight in positive_weights)),
        "inf": -math.log(max(positive_weights)),
    }
    d_alpha = {key: math.exp(value) for key, value in h_alpha.items()}
    return {
        "n_weighted": len(positive_weights),
        "n_unweighted": n_rows - len(positive_weights),
        "p_max": max(positive_weights),
        "H_alpha": h_alpha,
        "D_alpha": d_alpha,
    }


def _compute_temperature_rows(
    *,
    temperature: float,
    basis_rows: list[dict[str, Any]],
    event_records: list[dict[str, Any]],
    ctx: Any,
) -> tuple[list[dict[str, Any]], float, float]:
    raw_weight_by_geometry_id = {row["raw_geometry_id"]: 0.0 for row in basis_rows}

    for event_record in event_records:
        support_ids = event_record["support_ids"]
        max_delta_lnL = float(event_record["max_delta_lnL"])
        local_scores = {
            geometry_id: math.exp((event_record["delta_lnL_by_geometry_id"][geometry_id] - max_delta_lnL) / temperature)
            for geometry_id in support_ids
        }
        local_score_sum = math.fsum(local_scores.values())
        if local_score_sum <= 0.0 or not math.isfinite(local_score_sum):
            abort(
                ctx,
                f"Non-positive local score sum in temperature sweep: "
                f"temperature={temperature} event_id={event_record['event_id']} source_run_id={event_record['source_run_id']}",
            )
        for geometry_id in support_ids:
            raw_weight_by_geometry_id[geometry_id] += local_scores[geometry_id] / local_score_sum

    weight_sum_raw = math.fsum(raw_weight_by_geometry_id.values())
    if weight_sum_raw <= 0.0 or not math.isfinite(weight_sum_raw):
        abort(ctx, f"Non-positive raw weight sum in temperature sweep at temperature={temperature}")

    rows: list[dict[str, Any]] = []
    for row in basis_rows:
        raw_geometry_id = row["raw_geometry_id"]
        weight_raw = raw_weight_by_geometry_id[raw_geometry_id]
        if not math.isfinite(weight_raw) or weight_raw <= 0.0:
            abort(
                ctx,
                f"Temperature sweep did not achieve full coverage: "
                f"temperature={temperature} raw_geometry_id={raw_geometry_id!r} weight_raw={weight_raw!r}",
            )
        weight_normalized = weight_raw / weight_sum_raw
        rows.append(
            {
                "raw_geometry_id": raw_geometry_id,
                "normalized_geometry_id": row["normalized_geometry_id"],
                "atlas_family": row["atlas_family"],
                "atlas_theory": row["atlas_theory"],
                "weight_raw": weight_raw,
                "weight_normalized": weight_normalized,
            }
        )

    observed_weight_sum_normalized = math.fsum(float(row["weight_normalized"]) for row in rows)
    if not math.isclose(observed_weight_sum_normalized, 1.0, rel_tol=0.0, abs_tol=TOLERANCE_WEIGHT):
        abort(
            ctx,
            f"Temperature sweep normalized weight sum mismatch at temperature={temperature}: "
            f"observed={observed_weight_sum_normalized}",
        )
    return rows, weight_sum_raw, 1.0


def _ranked_bucket_rows(
    rows: list[dict[str, Any]],
    *,
    temperature: float,
    key_name: str,
    mass_field: str,
    count_field: str,
    rank_field: str,
    criterion: str,
) -> list[dict[str, Any]]:
    bucket_mass: dict[str, float] = {}
    bucket_count: dict[str, int] = {}
    for row in rows:
        key = str(row[key_name])
        bucket_mass[key] = bucket_mass.get(key, 0.0) + float(row["weight_normalized"])
        bucket_count[key] = bucket_count.get(key, 0) + 1
    ordered = sorted(bucket_mass.items(), key=lambda item: (-item[1], item[0]))
    payload_rows: list[dict[str, Any]] = []
    for rank, (bucket, mass) in enumerate(ordered, start=1):
        payload_rows.append(
            {
                "temperature": temperature,
                key_name: bucket,
                mass_field: mass,
                count_field: bucket_count[bucket],
                rank_field: rank,
                "criterion": criterion,
                "criterion_version": "v1",
            }
        )
    return payload_rows


def _compare_t1_weights(
    t1_rows: list[dict[str, Any]],
    existing_weight_policy: dict[str, Any],
    *,
    ctx: Any,
) -> bool:
    for row in t1_rows:
        raw_geometry_id = row["raw_geometry_id"]
        existing_row = existing_weight_policy["rows_by_geometry_id"].get(raw_geometry_id)
        if existing_row is None:
            abort(ctx, f"T=1 comparison missing geometry in existing weight policy: raw_geometry_id={raw_geometry_id!r}")
        for field_name in ("normalized_geometry_id", "atlas_family", "atlas_theory"):
            if row[field_name] != existing_row[field_name]:
                abort(
                    ctx,
                    f"T=1 weight comparison metadata mismatch for raw_geometry_id={raw_geometry_id!r}: "
                    f"field={field_name!r} observed={row[field_name]!r} existing={existing_row[field_name]!r}",
                )
        if not math.isclose(float(row["weight_normalized"]), float(existing_row["weight_normalized"]), rel_tol=0.0, abs_tol=TOLERANCE_WEIGHT):
            abort(
                ctx,
                f"T=1 weight comparison mismatch for raw_geometry_id={raw_geometry_id!r}: "
                f"observed={row['weight_normalized']} existing={existing_row['weight_normalized']} "
                f"tolerance={TOLERANCE_WEIGHT}",
            )
        if not math.isclose(float(row["weight_raw"]), float(existing_row["weight_raw"]), rel_tol=0.0, abs_tol=TOLERANCE_WEIGHT):
            abort(
                ctx,
                f"T=1 raw weight comparison mismatch for raw_geometry_id={raw_geometry_id!r}: "
                f"observed={row['weight_raw']} existing={existing_row['weight_raw']} "
                f"tolerance={TOLERANCE_WEIGHT}",
            )
    return True


def _compare_t1_renyi(
    metrics_row: dict[str, Any],
    existing_renyi_baseline: dict[str, Any],
    *,
    ctx: Any,
) -> bool:
    metrics = existing_renyi_baseline["metrics"]
    for key in ("0", "1", "2", "inf"):
        observed = float(metrics_row[f"H_{key}" if key != "inf" else "H_inf"])
        existing = float(metrics["H_alpha"][key])
        if not math.isclose(observed, existing, rel_tol=0.0, abs_tol=TOLERANCE_METRIC):
            abort(
                ctx,
                f"T=1 Renyi H_alpha mismatch at order={key}: "
                f"observed={observed} existing={existing} tolerance={TOLERANCE_METRIC}",
            )
    for key in ("0", "1", "2", "inf"):
        observed = float(metrics_row[f"D_{key}" if key != "inf" else "D_inf"])
        existing = float(metrics["D_alpha"][key])
        if not math.isclose(observed, existing, rel_tol=0.0, abs_tol=TOLERANCE_METRIC):
            abort(
                ctx,
                f"T=1 Renyi D_alpha mismatch at order={key}: "
                f"observed={observed} existing={existing} tolerance={TOLERANCE_METRIC}",
            )
    for field_name in ("p_max", "weight_sum_normalized"):
        observed = float(metrics_row[field_name])
        existing = float(metrics[field_name])
        if not math.isclose(observed, existing, rel_tol=0.0, abs_tol=TOLERANCE_METRIC):
            abort(
                ctx,
                f"T=1 Renyi metric mismatch for {field_name}: "
                f"observed={observed} existing={existing} tolerance={TOLERANCE_METRIC}",
            )
    if int(metrics_row["n_weighted"]) != int(metrics["n_weighted"]):
        abort(
            ctx,
            f"T=1 Renyi n_weighted mismatch: observed={metrics_row['n_weighted']} existing={metrics['n_weighted']}",
        )
    if int(metrics_row["n_unweighted"]) != int(metrics["n_unweighted"]):
        abort(
            ctx,
            f"T=1 Renyi n_unweighted mismatch: observed={metrics_row['n_unweighted']} existing={metrics['n_unweighted']}",
        )
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "temperature_grid": TEMPERATURE_GRID,
            "alpha_grid": ALPHA_GRID,
            "k_grid": K_GRID,
            "top_n": TOP_N,
            "tolerance_weight": TOLERANCE_WEIGHT,
            "tolerance_metric": TOLERANCE_METRIC,
        },
    )

    support_path = ctx.run_dir / SUPPORT_INPUT_REL
    aggregate_path = ctx.run_dir / AGGREGATE_INPUT_REL
    existing_weight_path = ctx.run_dir / EXISTING_WEIGHT_INPUT_REL
    existing_renyi_path = ctx.run_dir / EXISTING_RENYI_INPUT_REL

    aggregate_event_runs: list[dict[str, str]] = []
    if aggregate_path.exists():
        aggregate_event_runs = _load_aggregate_event_runs(aggregate_path, ctx=ctx)

    runtime_inputs: dict[str, Path] = {
        "support_ontology_basis_v1": support_path,
        "aggregate": aggregate_path,
        "existing_weight_policy_t1": existing_weight_path,
        "existing_renyi_baseline_t1": existing_renyi_path,
    }

    basis = _load_support_basis(support_path, ctx=ctx) if support_path.exists() else None
    if basis is not None:
        event_records, contributing_event_ids_by_geometry_id, _source_artifacts_by_geometry_id, dynamic_inputs = _load_event_records(
            out_root=ctx.out_root,
            aggregate_event_runs=aggregate_event_runs,
            basis_by_geometry_id=basis["basis_by_geometry_id"],
            ctx=ctx,
        )
        runtime_inputs.update(dynamic_inputs)
    else:
        event_records = []
        contributing_event_ids_by_geometry_id = {}

    check_inputs(ctx, runtime_inputs)
    _normalize_input_records_to_out_root(ctx, runtime_inputs)

    assert basis is not None
    existing_weight_policy = _load_existing_weight_policy(existing_weight_path, ctx=ctx)
    existing_renyi_baseline = _load_existing_renyi_baseline(existing_renyi_path, ctx=ctx)
    _validate_basis_against_existing(basis, existing_weight_policy, existing_renyi_baseline, ctx=ctx)

    support_mismatches: list[str] = []
    for row in basis["ordered_rows"]:
        raw_geometry_id = row["raw_geometry_id"]
        contributing_count = len(contributing_event_ids_by_geometry_id[raw_geometry_id])
        if contributing_count != int(row["support_count_events"]):
            support_mismatches.append(
                f"{raw_geometry_id}:support={row['support_count_events']} observed={contributing_count}"
            )
    if support_mismatches:
        abort(
            ctx,
            f"support_count_events mismatch against event support membership: "
            f"examples={support_mismatches[:5]} total_mismatched={len(support_mismatches)}",
        )

    metrics_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    theory_rows: list[dict[str, Any]] = []
    top_geometry_rows: list[dict[str, Any]] = []
    families = {row["atlas_family"] for row in basis["ordered_rows"]}
    theories = {row["atlas_theory"] for row in basis["ordered_rows"]}

    t1_rows: list[dict[str, Any]] | None = None
    t1_metrics_row: dict[str, Any] | None = None

    for temperature in TEMPERATURE_GRID:
        rows, weight_sum_raw, weight_sum_normalized = _compute_temperature_rows(
            temperature=temperature,
            basis_rows=basis["ordered_rows"],
            event_records=event_records,
            ctx=ctx,
        )
        sorted_rows = _sort_rows(rows)
        renyi_metrics = _renyi_metrics_from_weights(
            [float(row["weight_normalized"]) for row in rows],
            n_rows=basis["n_rows"],
        )
        metrics_row = {
            "temperature": temperature,
            "policy_name": SWEEP_POLICY_NAME,
            "normalization_method": "divide_by_sum_event_normalized_scores_over_all_rows",
            "n_rows": basis["n_rows"],
            "coverage_fraction": 1.0,
            "n_weighted": renyi_metrics["n_weighted"],
            "n_unweighted": renyi_metrics["n_unweighted"],
            "p_max": renyi_metrics["p_max"],
            "H_0": renyi_metrics["H_alpha"]["0"],
            "H_1": renyi_metrics["H_alpha"]["1"],
            "H_2": renyi_metrics["H_alpha"]["2"],
            "H_inf": renyi_metrics["H_alpha"]["inf"],
            "D_0": renyi_metrics["D_alpha"]["0"],
            "D_1": renyi_metrics["D_alpha"]["1"],
            "D_2": renyi_metrics["D_alpha"]["2"],
            "D_inf": renyi_metrics["D_alpha"]["inf"],
            "weight_sum_raw": weight_sum_raw,
            "weight_sum_normalized": weight_sum_normalized,
            "criterion": "delta_lnL_softmax_per_event_temperature_sweep_over_final_support_region",
            "criterion_version": "v1",
        }
        metrics_rows.append(metrics_row)

        total_mass = math.fsum(float(row["weight_normalized"]) for row in sorted_rows)
        for top_k in K_GRID:
            topk_rows.append(
                {
                    "temperature": temperature,
                    "top_k": top_k,
                    "cumulative_mass": math.fsum(float(row["weight_normalized"]) for row in sorted_rows[:top_k]),
                    "normalization_check": total_mass,
                    "criterion": "cumulative_topk_mass_over_weight_normalized",
                    "criterion_version": "v1",
                }
            )

        family_rows.extend(
            _ranked_bucket_rows(
                sorted_rows,
                temperature=temperature,
                key_name="atlas_family",
                mass_field="family_mass",
                count_field="family_count",
                rank_field="family_mass_rank",
                criterion="mass_sum_by_atlas_family_over_weight_normalized",
            )
        )
        theory_rows.extend(
            _ranked_bucket_rows(
                sorted_rows,
                temperature=temperature,
                key_name="atlas_theory",
                mass_field="theory_mass",
                count_field="theory_count",
                rank_field="theory_mass_rank",
                criterion="mass_sum_by_atlas_theory_over_weight_normalized",
            )
        )

        cumulative_mass = 0.0
        for rank, row in enumerate(sorted_rows[:TOP_N], start=1):
            cumulative_mass += float(row["weight_normalized"])
            top_geometry_rows.append(
                {
                    "temperature": temperature,
                    "rank": rank,
                    "raw_geometry_id": row["raw_geometry_id"],
                    "normalized_geometry_id": row["normalized_geometry_id"],
                    "atlas_family": row["atlas_family"],
                    "atlas_theory": row["atlas_theory"],
                    "weight_normalized": float(row["weight_normalized"]),
                    "cumulative_mass": cumulative_mass,
                    "criterion": "weight_normalized_descending_cumulative_mass",
                    "criterion_version": "v1",
                }
            )

        if math.isclose(temperature, 1.0, rel_tol=0.0, abs_tol=1e-12):
            t1_rows = rows
            t1_metrics_row = metrics_row

    if t1_rows is None or t1_metrics_row is None:
        abort(ctx, "Temperature grid must contain T=1.0 for required consistency checks")

    t1_weight_match_with_existing_policy = _compare_t1_weights(t1_rows, existing_weight_policy, ctx=ctx)
    t1_renyi_match_with_existing_baseline = _compare_t1_renyi(t1_metrics_row, existing_renyi_baseline, ctx=ctx)

    metrics_payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "comparison_role": COMPARISON_ROLE,
        "basis_name": basis["basis_name"],
        "temperature_grid": TEMPERATURE_GRID,
        "alpha_grid": ALPHA_GRID,
        "rows": metrics_rows,
        "notes": [
            "Temperature sensitivity of the epistemic score-based delta_lnL softmax policy.",
            "No sector conditioning is applied.",
            "This sweep is not a black-hole thermodynamic entropy claim.",
        ],
    }
    topk_payload = {
        "schema_version": TOPK_SCHEMA_VERSION,
        "basis_name": basis["basis_name"],
        "temperature_grid": TEMPERATURE_GRID,
        "k_grid": K_GRID,
        "rows": topk_rows,
    }
    family_payload = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "basis_name": basis["basis_name"],
        "temperature_grid": TEMPERATURE_GRID,
        "rows": family_rows,
    }
    theory_payload = {
        "schema_version": THEORY_SCHEMA_VERSION,
        "basis_name": basis["basis_name"],
        "temperature_grid": TEMPERATURE_GRID,
        "rows": theory_rows,
    }
    top_geometry_payload = {
        "schema_version": TOP_GEOMETRY_SCHEMA_VERSION,
        "basis_name": basis["basis_name"],
        "temperature_grid": TEMPERATURE_GRID,
        "top_n": TOP_N,
        "rows": top_geometry_rows,
    }

    metrics_output_path = ctx.outputs_dir / METRICS_OUTPUT_NAME
    topk_output_path = ctx.outputs_dir / TOPK_OUTPUT_NAME
    family_output_path = ctx.outputs_dir / FAMILY_OUTPUT_NAME
    theory_output_path = ctx.outputs_dir / THEORY_OUTPUT_NAME
    top_geometry_output_path = ctx.outputs_dir / TOP_GEOMETRY_OUTPUT_NAME

    write_json_atomic(metrics_output_path, metrics_payload)
    write_json_atomic(topk_output_path, topk_payload)
    write_json_atomic(family_output_path, family_payload)
    write_json_atomic(theory_output_path, theory_payload)
    write_json_atomic(top_geometry_output_path, top_geometry_payload)

    finalize(
        ctx,
        artifacts={
            "temperature_sweep_metrics_v1": metrics_output_path,
            "temperature_sweep_topk_v1": topk_output_path,
            "temperature_sweep_family_mass_v1": family_output_path,
            "temperature_sweep_theory_mass_v1": theory_output_path,
            "temperature_sweep_top_geometry_v1": top_geometry_output_path,
        },
        results={
            "n_rows": basis["n_rows"],
            "family_count": len(families),
            "theory_count": len(theories),
            "t1_weight_match_with_existing_policy": t1_weight_match_with_existing_policy,
            "t1_renyi_match_with_existing_baseline": t1_renyi_match_with_existing_baseline,
        },
        extra_summary={
            "schema_version": METRICS_SCHEMA_VERSION,
            "comparison_role": COMPARISON_ROLE,
            "basis_name": basis["basis_name"],
            "temperature_grid": TEMPERATURE_GRID,
            "alpha_grid": ALPHA_GRID,
            "k_grid": K_GRID,
            "top_n": TOP_N,
            "n_rows": basis["n_rows"],
            "family_count": len(families),
            "theory_count": len(theories),
            "t1_weight_match_with_existing_policy": t1_weight_match_with_existing_policy,
            "t1_renyi_match_with_existing_baseline": t1_renyi_match_with_existing_baseline,
            "tolerance_weight": TOLERANCE_WEIGHT,
            "tolerance_metric": TOLERANCE_METRIC,
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
