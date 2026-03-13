#!/usr/bin/env python3
"""Materialize explicit weight policies over the supported basis."""
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

STAGE = "experiment/phase3_weight_policy_basis"
SCHEMA_VERSION = "weight_policy_basis_v1"
WEIGHT_STATUS = "WEIGHTED"
DEFAULT_INPUT_NAME = "support_ontology_basis_v1.json"
DEFAULT_OUTPUT_NAME = "weight_policy_basis_v1.json"
DEFAULT_AGGREGATE_NAME = "aggregate.json"
SUPPORTED_POLICIES = {
    "uniform_support_v1": {
        "policy_role": "baseline_canonical",
        "normalization_method": "divide_by_sum_raw_over_all_rows",
        "source_policy_inputs": [],
        "criterion": "uniform_over_support_basis",
        "criterion_version": "v1",
    },
    "event_frequency_support_v1": {
        "policy_role": "comparison_factual",
        "normalization_method": "divide_by_sum_support_count_events_over_all_rows",
        "source_policy_inputs": ["support_count_events_from_phase2c"],
        "criterion": "support_count_events_over_support_basis",
        "criterion_version": "v1",
    },
    "event_support_delta_lnL_softmax_mean_v1": {
        "policy_role": "comparison_score_based",
        "normalization_method": "divide_by_sum_event_normalized_scores_over_all_rows",
        "source_policy_inputs": [
            "s5_aggregate/outputs/aggregate.json",
            "{source_run}/s4k_event_support_region/outputs/event_support_region.json",
            "{source_run}/s4_geometry_filter/outputs/ranked_all_full.json",
            "delta_lnL",
            "softmax per-event over final_support_region",
        ],
        "criterion": "delta_lnL_softmax_per_event_over_final_support_region",
        "criterion_version": "v1",
    },
}
DEFAULT_POLICY_NAME = "uniform_support_v1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize explicit weight policies over the supported basis")
    ap.add_argument("--run-id", required=True, help="Aggregate run id containing phase2c support basis")
    ap.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    ap.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    ap.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
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
        abort(ctx, f"phase2c support basis row[{row_index}] missing {field_name}")
        raise AssertionError("unreachable")
    return value


def _require_number(row: dict[str, Any], field_name: str, *, row_index: int, ctx: Any) -> float:
    try:
        value = float(row.get(field_name))
    except Exception:
        abort(ctx, f"phase2c support basis row[{row_index}] missing numeric {field_name}")
        raise AssertionError("unreachable")
    return value


def _require_positive_int(row: dict[str, Any], field_name: str, *, row_index: int, ctx: Any) -> int:
    value = int(_require_number(row, field_name, row_index=row_index, ctx=ctx))
    if value <= 0:
        abort(ctx, f"phase2c support basis row[{row_index}] has non-positive {field_name}={value}")
        raise AssertionError("unreachable")
    return value


def _load_phase2c_basis(path: Path, ctx: Any) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt support_ontology_basis_v1 at {path}: {exc}")
        raise AssertionError("unreachable")

    if not isinstance(payload, dict) or payload.get("schema_version") != "support_ontology_basis_v1":
        abort(ctx, f"Expected schema_version=support_ontology_basis_v1 at {path}")
        raise AssertionError("unreachable")

    basis_name = _coerce_text(payload.get("basis_name")).strip()
    if not basis_name:
        abort(ctx, f"support_ontology_basis_v1 at {path} missing basis_name")
        raise AssertionError("unreachable")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        abort(ctx, f"support_ontology_basis_v1 at {path} missing non-empty rows")
        raise AssertionError("unreachable")

    n_rows_declared = payload.get("n_rows")
    if int(n_rows_declared) != len(rows):
        abort(ctx, f"support_ontology_basis_v1 n_rows mismatch at {path}: declared={n_rows_declared} actual={len(rows)}")
        raise AssertionError("unreachable")

    return {
        "basis_name": basis_name,
        "rows": rows,
    }


def _unsupported_policy_message(policy_name: str) -> str:
    supported = ", ".join(repr(name) for name in sorted(SUPPORTED_POLICIES))
    return f"Unsupported policy_name={policy_name!r}; supported values are [{supported}]"


def _output_path_rel(output_name: str) -> Path:
    return Path(STAGE) / "outputs" / output_name


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


def _regen_multimode_cmd(event_id: str, source_run_id: str) -> str:
    return (
        f"python -m mvp.pipeline multimode --event-id {event_id} "
        f"--run-id {source_run_id} --atlas-default --offline --estimator dual"
    )


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
            raise AssertionError("unreachable")
        event_id = _require_text(row, "event_id", row_index=idx, ctx=ctx)
        source_run_id = _require_text(row, "run_id", row_index=idx, ctx=ctx)
        resolved.append({"event_id": event_id, "source_run_id": source_run_id})
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "policy_name": args.policy_name,
            "input_name": args.input_name,
            "output_name": args.output_name,
        },
    )

    if args.policy_name not in SUPPORTED_POLICIES:
        abort(ctx, _unsupported_policy_message(args.policy_name))

    source_basis_rel = Path("experiment") / "phase2c_support_ontology_basis" / "outputs" / args.input_name
    source_basis_path = ctx.run_dir / source_basis_rel
    runtime_inputs: dict[str, Path] = {"support_ontology_basis_v1": source_basis_path}
    aggregate_rel = Path("s5_aggregate") / "outputs" / DEFAULT_AGGREGATE_NAME
    aggregate_path = ctx.run_dir / aggregate_rel
    aggregate_event_runs: list[dict[str, str]] = []
    dynamic_event_paths: list[dict[str, str | Path]] = []

    if args.policy_name == "event_support_delta_lnL_softmax_mean_v1":
        if not aggregate_path.exists():
            abort(
                ctx,
                f"Missing required input for policy {args.policy_name}: "
                f"expected_path={aggregate_path}",
            )
        runtime_inputs["aggregate"] = aggregate_path
        aggregate_event_runs = _load_aggregate_event_runs(aggregate_path, ctx=ctx)
        for rec in aggregate_event_runs:
            event_id = rec["event_id"]
            source_run_id = rec["source_run_id"]
            try:
                require_run_valid(ctx.out_root, source_run_id)
            except Exception as exc:
                abort(
                    ctx,
                    f"Source run not PASS for policy {args.policy_name}: "
                    f"expected_path={ctx.out_root / source_run_id / 'RUN_VALID' / 'verdict.json'}; "
                    f"regen_cmd='{_regen_multimode_cmd(event_id, source_run_id)}'; detail={exc}",
                )

            event_support_rel = Path(source_run_id) / "s4k_event_support_region" / "outputs" / "event_support_region.json"
            ranked_rel = Path(source_run_id) / "s4_geometry_filter" / "outputs" / "ranked_all_full.json"
            event_support_path = ctx.out_root / event_support_rel
            ranked_path = ctx.out_root / ranked_rel
            if not event_support_path.exists():
                abort(
                    ctx,
                    f"Missing required input for policy {args.policy_name}: expected_path={event_support_path}; "
                    f"regen_cmd='{_regen_multimode_cmd(event_id, source_run_id)}'",
                )
            if not ranked_path.exists():
                abort(
                    ctx,
                    f"Missing required input for policy {args.policy_name}: expected_path={ranked_path}; "
                    f"regen_cmd='{_regen_multimode_cmd(event_id, source_run_id)}'",
                )
            runtime_inputs[f"{source_run_id}:event_support_region"] = event_support_path
            runtime_inputs[f"{source_run_id}:ranked_all_full"] = ranked_path
            dynamic_event_paths.append(
                {
                    "event_id": event_id,
                    "source_run_id": source_run_id,
                    "event_support_rel": event_support_rel,
                    "event_support_path": event_support_path,
                    "ranked_rel": ranked_rel,
                    "ranked_path": ranked_path,
                }
            )

    check_inputs(ctx, runtime_inputs)
    if args.policy_name == "event_support_delta_lnL_softmax_mean_v1":
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

    phase2c = _load_phase2c_basis(source_basis_path, ctx)
    basis_rows = phase2c["rows"]
    n_rows = len(basis_rows)
    policy_spec = SUPPORTED_POLICIES[args.policy_name]

    copied_rows: list[dict[str, Any]] = []
    support_count_sum = 0
    basis_by_raw_geometry_id: dict[str, dict[str, Any]] = {}
    for idx, basis_row in enumerate(basis_rows):
        if not isinstance(basis_row, dict):
            abort(ctx, f"phase2c support basis row[{idx}] is not an object")
            raise AssertionError("unreachable")
        raw_geometry_id = _require_text(basis_row, "raw_geometry_id", row_index=idx, ctx=ctx)
        support_count_events = _require_positive_int(basis_row, "n_events_supported", row_index=idx, ctx=ctx)
        support_fraction_events = _require_number(basis_row, "support_fraction_events", row_index=idx, ctx=ctx)
        if raw_geometry_id in basis_by_raw_geometry_id:
            abort(ctx, f"phase2c support basis contains duplicate raw_geometry_id={raw_geometry_id!r}")
        copied_row = {
            "raw_geometry_id": raw_geometry_id,
            "normalized_geometry_id": _require_text(basis_row, "normalized_geometry_id", row_index=idx, ctx=ctx),
            "atlas_family": _require_text(basis_row, "atlas_family", row_index=idx, ctx=ctx),
            "atlas_theory": _require_text(basis_row, "atlas_theory", row_index=idx, ctx=ctx),
            "support_count_events": support_count_events,
            "support_fraction_events": support_fraction_events,
        }
        copied_rows.append(copied_row)
        basis_by_raw_geometry_id[raw_geometry_id] = copied_row
        support_count_sum += support_count_events

    if args.policy_name == "uniform_support_v1":
        weight_sum_raw = float(n_rows)
    elif args.policy_name == "event_frequency_support_v1":
        weight_sum_raw = float(support_count_sum)
        if weight_sum_raw <= 0.0:
            abort(ctx, "event_frequency_support_v1 requires positive total support_count_events over the basis")
    else:
        raw_weight_by_geometry_id = {row["raw_geometry_id"]: 0.0 for row in copied_rows}
        contributing_event_ids_by_geometry_id = {row["raw_geometry_id"]: [] for row in copied_rows}
        source_artifacts_by_geometry_id = {
            row["raw_geometry_id"]: [str(source_basis_rel), str(aggregate_rel)]
            for row in copied_rows
        }

        for rec in dynamic_event_paths:
            event_id = str(rec["event_id"])
            source_run_id = str(rec["source_run_id"])
            event_support_path = Path(rec["event_support_path"])
            ranked_path = Path(rec["ranked_path"])
            event_support_rel = str(rec["event_support_rel"])
            ranked_rel = str(rec["ranked_rel"])

            support_payload = _load_json_object(event_support_path, label=f"{source_run_id}:event_support_region.json", ctx=ctx)
            final_geometry_ids = support_payload.get("final_geometry_ids")
            if not isinstance(final_geometry_ids, list):
                abort(ctx, f"Invalid event_support_region.json at {event_support_path}: final_geometry_ids must be a list")
            restricted_support_ids = [
                _coerce_text(geometry_id).strip()
                for geometry_id in final_geometry_ids
                if _coerce_text(geometry_id).strip() in basis_by_raw_geometry_id
            ]
            if not restricted_support_ids:
                abort(
                    ctx,
                    f"event_support_region.json has no overlap with phase2c basis: "
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

            event_deltas = [delta_lnL_by_geometry_id[geometry_id] for geometry_id in restricted_support_ids]
            max_delta_lnL_event = max(event_deltas)
            local_scores = {
                geometry_id: math.exp(delta_lnL_by_geometry_id[geometry_id] - max_delta_lnL_event)
                for geometry_id in restricted_support_ids
            }
            local_score_sum = math.fsum(local_scores.values())
            if local_score_sum <= 0.0:
                abort(
                    ctx,
                    f"Non-positive local score sum for event_support_delta_lnL_softmax_mean_v1: "
                    f"event_id={event_id} source_run_id={source_run_id}",
                )

            for geometry_id in restricted_support_ids:
                raw_weight_by_geometry_id[geometry_id] += local_scores[geometry_id] / local_score_sum
                contributing_event_ids_by_geometry_id[geometry_id].append(event_id)
                source_artifacts_by_geometry_id[geometry_id].extend([event_support_rel, ranked_rel])

        coverage_failures: list[str] = []
        support_mismatches: list[str] = []
        for copied_row in copied_rows:
            geometry_id = copied_row["raw_geometry_id"]
            contributing_count = len(contributing_event_ids_by_geometry_id[geometry_id])
            if contributing_count <= 0 or raw_weight_by_geometry_id[geometry_id] <= 0.0:
                coverage_failures.append(geometry_id)
            if contributing_count != int(copied_row["support_count_events"]):
                support_mismatches.append(
                    f"{geometry_id}:phase2c={copied_row['support_count_events']} observed={contributing_count}"
                )
        if coverage_failures:
            abort(
                ctx,
                f"event_support_delta_lnL_softmax_mean_v1 did not achieve full coverage over the phase2c basis: "
                f"missing_geometry_ids={coverage_failures[:5]} total_missing={len(coverage_failures)}",
            )
        if support_mismatches:
            abort(
                ctx,
                f"phase2c support_count_events mismatch against contributing events for policy {args.policy_name}: "
                f"examples={support_mismatches[:5]} total_mismatched={len(support_mismatches)}",
            )

        weight_sum_raw = math.fsum(raw_weight_by_geometry_id.values())
        if weight_sum_raw <= 0.0:
            abort(ctx, f"{args.policy_name} produced non-positive global raw weight sum")

    rows: list[dict[str, Any]] = []
    for copied_row in copied_rows:
        if args.policy_name == "uniform_support_v1":
            weight_raw = 1.0
            weight_normalized = 1.0 / float(n_rows)
            evidence = {
                "basis_name": phase2c["basis_name"],
                "uniform_weight_raw": 1.0,
                "uniform_weight_normalized": weight_normalized,
            }
            source_artifacts = [str(source_basis_rel)]
        else:
            if args.policy_name == "event_frequency_support_v1":
                weight_raw = float(copied_row["support_count_events"])
                weight_normalized = weight_raw / weight_sum_raw
                evidence = {
                    "basis_name": phase2c["basis_name"],
                    "support_count_events": copied_row["support_count_events"],
                    "support_count_sum_over_basis": support_count_sum,
                    "event_frequency_weight_raw": weight_raw,
                    "event_frequency_weight_normalized": weight_normalized,
                }
                source_artifacts = [str(source_basis_rel)]
            else:
                geometry_id = copied_row["raw_geometry_id"]
                weight_raw = raw_weight_by_geometry_id[geometry_id]
                weight_normalized = weight_raw / weight_sum_raw
                contributing_event_ids = contributing_event_ids_by_geometry_id[geometry_id]
                evidence = {
                    "basis_name": phase2c["basis_name"],
                    "contributing_event_count": len(contributing_event_ids),
                    "event_ids_sample": contributing_event_ids[:5],
                    "local_score_definition": "exp(delta_lnL_i - max_delta_lnL_event)",
                    "local_normalization": "per_event_softmax",
                    "aggregation": "sum_over_events",
                }
                source_artifacts = source_artifacts_by_geometry_id[geometry_id]

        row = {
            "raw_geometry_id": copied_row["raw_geometry_id"],
            "normalized_geometry_id": copied_row["normalized_geometry_id"],
            "atlas_family": copied_row["atlas_family"],
            "atlas_theory": copied_row["atlas_theory"],
            "policy_name": args.policy_name,
            "weight_raw": weight_raw,
            "weight_normalized": weight_normalized,
            "weight_status": WEIGHT_STATUS,
            "support_count_events": copied_row["support_count_events"],
            "support_fraction_events": copied_row["support_fraction_events"],
            "source_artifacts": source_artifacts,
            "criterion": policy_spec["criterion"],
            "criterion_version": policy_spec["criterion_version"],
            "evidence": evidence,
        }
        rows.append(row)

    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "basis_name": phase2c["basis_name"],
        "policy_name": args.policy_name,
        "policy_role": policy_spec["policy_role"],
        "coverage_fraction": 1.0,
        "n_rows": n_rows,
        "n_weighted": n_rows,
        "n_unweighted": 0,
        "weight_sum_raw": weight_sum_raw,
        "weight_sum_normalized": 1.0,
        "normalization_method": policy_spec["normalization_method"],
        "source_policy_inputs": policy_spec["source_policy_inputs"],
        "rows": rows,
    }

    output_rel = _output_path_rel(args.output_name)
    output_path = ctx.outputs_dir / args.output_name
    write_json_atomic(output_path, output_payload)

    finalize(
        ctx,
        artifacts={"weight_policy_basis_v1": output_path},
        results={
            "coverage_fraction": 1.0,
            "n_rows": n_rows,
            "n_weighted": n_rows,
            "n_unweighted": 0,
        },
        extra_summary={
            "schema_version": SCHEMA_VERSION,
            "basis_name": phase2c["basis_name"],
            "policy_name": args.policy_name,
            "policy_role": policy_spec["policy_role"],
            "coverage_fraction": 1.0,
            "n_rows": n_rows,
            "n_weighted": n_rows,
            "n_unweighted": 0,
            "weight_sum_raw": weight_sum_raw,
            "weight_sum_normalized": 1.0,
            "normalization_method": policy_spec["normalization_method"],
            "source_policy_inputs": policy_spec["source_policy_inputs"],
            "output_name": args.output_name,
            "output_path": str(output_rel),
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
