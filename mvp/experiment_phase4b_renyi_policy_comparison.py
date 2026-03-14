#!/usr/bin/env python3
"""Compare Renyi diversity baselines and weight policies over the supported basis."""
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

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "experiment/phase4b_renyi_policy_comparison"
SCHEMA_VERSION = "renyi_policy_comparison_v1"
RENVI_SCHEMA_VERSION = "renyi_diversity_baseline_v1"
WEIGHT_SCHEMA_VERSION = "weight_policy_basis_v1"
COMPARISON_ROLE = "epistemic_policy_comparison"
ALPHA_GRID: list[int | str] = [0, 1, 2, "inf"]
K_GRID = [1, 3, 5, 10]
TOP_N = 20

POLICY_SPECS = [
    {
        "policy_name": "uniform_support_v1",
        "weight_policy_file": "weight_policy_uniform_support_v1.json",
        "renyi_diversity_file": "renyi_diversity_uniform_support_v1.json",
    },
    {
        "policy_name": "event_frequency_support_v1",
        "weight_policy_file": "weight_policy_event_frequency_support_v1.json",
        "renyi_diversity_file": "renyi_diversity_event_frequency_support_v1.json",
    },
    {
        "policy_name": "event_support_delta_lnL_softmax_mean_v1",
        "weight_policy_file": "weight_policy_event_support_delta_lnL_softmax_mean_v1.json",
        "renyi_diversity_file": "renyi_diversity_event_support_delta_lnL_softmax_mean_v1.json",
    },
]

RENYI_OUTPUT_NAME = "renyi_policy_comparison_v1.json"
TOPK_OUTPUT_NAME = "topk_mass_by_policy_v1.json"
FAMILY_OUTPUT_NAME = "family_mass_by_policy_v1.json"
THEORY_OUTPUT_NAME = "theory_mass_by_policy_v1.json"
RANKING_OUTPUT_NAME = "top_geometry_ranking_by_policy_v1.json"
NORMALIZATION_TOLERANCE = 1e-9


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare multiple Renyi policy baselines over the same supported basis")
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


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row["weight_normalized"]), row["raw_geometry_id"]))


def _validate_weight_policy(
    path: Path,
    *,
    expected_policy_name: str,
    ctx: Any,
) -> dict[str, Any]:
    payload = _load_json_object(path, label=f"{expected_policy_name}:weight_policy_basis_v1", ctx=ctx)
    if payload.get("schema_version") != WEIGHT_SCHEMA_VERSION:
        abort(ctx, f"Expected schema_version={WEIGHT_SCHEMA_VERSION} at {path}")
    if _coerce_text(payload.get("policy_name")).strip() != expected_policy_name:
        abort(
            ctx,
            f"weight policy file policy_name mismatch at {path}: "
            f"expected={expected_policy_name!r} observed={payload.get('policy_name')!r}",
        )
    basis_name = _coerce_text(payload.get("basis_name")).strip()
    if not basis_name:
        abort(ctx, f"weight policy file missing basis_name at {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        abort(ctx, f"weight policy file missing non-empty rows at {path}")
    try:
        n_rows = int(payload.get("n_rows"))
    except Exception:
        abort(ctx, f"weight policy file missing integer n_rows at {path}")
        raise AssertionError("unreachable")
    if n_rows != len(rows):
        abort(ctx, f"weight policy file n_rows mismatch at {path}: declared={n_rows} actual={len(rows)}")
    try:
        coverage_fraction = float(payload.get("coverage_fraction"))
    except Exception:
        abort(ctx, f"weight policy file missing numeric coverage_fraction at {path}")
        raise AssertionError("unreachable")
    try:
        weight_sum_normalized = float(payload.get("weight_sum_normalized"))
    except Exception:
        abort(ctx, f"weight policy file missing numeric weight_sum_normalized at {path}")
        raise AssertionError("unreachable")
    if not math.isclose(weight_sum_normalized, 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE):
        abort(ctx, f"weight policy file not normalized at {path}: weight_sum_normalized={weight_sum_normalized}")

    validated_rows: list[dict[str, Any]] = []
    seen_geometry_ids: set[str] = set()
    observed_weight_sum = 0.0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            abort(ctx, f"weight policy row[{idx}] is not an object at {path}")
        raw_geometry_id = _coerce_text(row.get("raw_geometry_id")).strip()
        normalized_geometry_id = _coerce_text(row.get("normalized_geometry_id")).strip()
        atlas_family = _coerce_text(row.get("atlas_family")).strip()
        atlas_theory = _coerce_text(row.get("atlas_theory")).strip()
        if not raw_geometry_id or not normalized_geometry_id or not atlas_family or not atlas_theory:
            abort(ctx, f"weight policy row[{idx}] missing geometry metadata at {path}")
        if raw_geometry_id in seen_geometry_ids:
            abort(ctx, f"duplicate raw_geometry_id={raw_geometry_id!r} in {path}")
        seen_geometry_ids.add(raw_geometry_id)
        try:
            weight_normalized = float(row.get("weight_normalized"))
        except Exception:
            abort(ctx, f"weight policy row[{idx}] missing numeric weight_normalized at {path}")
            raise AssertionError("unreachable")
        if not math.isfinite(weight_normalized) or weight_normalized < 0.0:
            abort(ctx, f"weight policy row[{idx}] has invalid weight_normalized={weight_normalized!r} at {path}")
        observed_weight_sum += weight_normalized
        validated_rows.append(
            {
                "raw_geometry_id": raw_geometry_id,
                "normalized_geometry_id": normalized_geometry_id,
                "atlas_family": atlas_family,
                "atlas_theory": atlas_theory,
                "weight_normalized": weight_normalized,
            }
        )
    if not math.isclose(observed_weight_sum, 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE):
        abort(ctx, f"weight policy row sum not normalized at {path}: observed={observed_weight_sum}")
    return {
        "basis_name": basis_name,
        "n_rows": n_rows,
        "coverage_fraction": coverage_fraction,
        "rows": validated_rows,
        "weight_policy_file": path.name,
    }


def _validate_renyi_baseline(
    path: Path,
    *,
    expected_policy_name: str,
    expected_weight_policy_file: str,
    ctx: Any,
) -> dict[str, Any]:
    payload = _load_json_object(path, label=f"{expected_policy_name}:renyi_diversity_baseline_v1", ctx=ctx)
    if payload.get("schema_version") != RENVI_SCHEMA_VERSION:
        abort(ctx, f"Expected schema_version={RENVI_SCHEMA_VERSION} at {path}")
    if _coerce_text(payload.get("policy_name")).strip() != expected_policy_name:
        abort(ctx, f"renyi diversity file policy_name mismatch at {path}")
    if _coerce_text(payload.get("weight_policy_file")).strip() != expected_weight_policy_file:
        abort(
            ctx,
            f"renyi diversity file weight_policy_file mismatch at {path}: "
            f"expected={expected_weight_policy_file!r} observed={payload.get('weight_policy_file')!r}",
        )
    if _coerce_text(payload.get("basis_name")).strip() == "":
        abort(ctx, f"renyi diversity file missing basis_name at {path}")
    if payload.get("alpha_grid") != ALPHA_GRID:
        abort(ctx, f"renyi diversity file alpha_grid mismatch at {path}: observed={payload.get('alpha_grid')!r}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        abort(ctx, f"renyi diversity file missing metrics at {path}")
    d_alpha = metrics.get("D_alpha")
    if not isinstance(d_alpha, dict):
        abort(ctx, f"renyi diversity file missing D_alpha at {path}")
    h_alpha = metrics.get("H_alpha")
    if not isinstance(h_alpha, dict):
        abort(ctx, f"renyi diversity file missing H_alpha at {path}")
    return payload


def _ranked_family_or_theory(
    rows: list[dict[str, Any]],
    *,
    policy_name: str,
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
    result: list[dict[str, Any]] = []
    for rank, (bucket, mass) in enumerate(ordered, start=1):
        result.append(
            {
                "policy_name": policy_name,
                key_name: bucket,
                mass_field: mass,
                count_field: bucket_count[bucket],
                rank_field: rank,
                "criterion": criterion,
                "criterion_version": "v1",
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "alpha_grid": ALPHA_GRID,
            "k_grid": K_GRID,
            "top_n": TOP_N,
        },
    )

    input_paths: dict[str, Path] = {}
    policy_records: list[dict[str, Any]] = []
    renyi_dir = Path("experiment") / "phase4_renyi_diversity_baseline" / "outputs"
    weight_dir = Path("experiment") / "phase3_weight_policy_basis" / "outputs"

    for spec in POLICY_SPECS:
        renyi_path = ctx.run_dir / renyi_dir / spec["renyi_diversity_file"]
        weight_path = ctx.run_dir / weight_dir / spec["weight_policy_file"]
        input_paths[f"{spec['policy_name']}:renyi_diversity"] = renyi_path
        input_paths[f"{spec['policy_name']}:weight_policy"] = weight_path
        policy_records.append(
            {
                "policy_name": spec["policy_name"],
                "weight_policy_file": spec["weight_policy_file"],
                "weight_policy_path": weight_path,
                "renyi_diversity_file": spec["renyi_diversity_file"],
                "renyi_diversity_path": renyi_path,
            }
        )

    check_inputs(ctx, input_paths)

    reference_basis_name: str | None = None
    reference_n_rows: int | None = None
    reference_geometry_ids: set[str] | None = None
    reference_geometry_meta: dict[str, tuple[str, str, str]] = {}
    metrics_table: dict[str, dict[str, Any]] = {}
    topk_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    theory_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    families: set[str] = set()
    theories: set[str] = set()
    policies = [spec["policy_name"] for spec in POLICY_SPECS]

    for rec in policy_records:
        policy_name = rec["policy_name"]
        weight_payload = _validate_weight_policy(rec["weight_policy_path"], expected_policy_name=policy_name, ctx=ctx)
        renyi_payload = _validate_renyi_baseline(
            rec["renyi_diversity_path"],
            expected_policy_name=policy_name,
            expected_weight_policy_file=rec["weight_policy_file"],
            ctx=ctx,
        )

        basis_name = weight_payload["basis_name"]
        n_rows = weight_payload["n_rows"]
        geometry_ids = {row["raw_geometry_id"] for row in weight_payload["rows"]}

        if reference_basis_name is None:
            reference_basis_name = basis_name
        elif basis_name != reference_basis_name:
            abort(
                ctx,
                f"basis_name mismatch across policies: reference={reference_basis_name!r} "
                f"observed={basis_name!r} policy_name={policy_name!r}",
            )

        renyi_basis_name = _coerce_text(renyi_payload.get("basis_name")).strip()
        if renyi_basis_name != basis_name:
            abort(
                ctx,
                f"basis_name mismatch between renyi and weight policy for {policy_name!r}: "
                f"renyi={renyi_basis_name!r} weight={basis_name!r}",
            )

        if reference_n_rows is None:
            reference_n_rows = n_rows
        elif n_rows != reference_n_rows:
            abort(
                ctx,
                f"n_rows mismatch across policies: reference={reference_n_rows} observed={n_rows} policy_name={policy_name!r}",
            )

        renyi_n_rows = int(renyi_payload.get("n_rows"))
        if renyi_n_rows != n_rows:
            abort(
                ctx,
                f"n_rows mismatch between renyi and weight policy for {policy_name!r}: "
                f"renyi={renyi_n_rows} weight={n_rows}",
            )

        if reference_geometry_ids is None:
            reference_geometry_ids = geometry_ids
        elif geometry_ids != reference_geometry_ids:
            missing_from_policy = sorted(reference_geometry_ids - geometry_ids)
            extra_in_policy = sorted(geometry_ids - reference_geometry_ids)
            abort(
                ctx,
                f"geometry key set mismatch for {policy_name!r}: "
                f"missing={missing_from_policy[:5]} extra={extra_in_policy[:5]}",
            )

        for row in weight_payload["rows"]:
            geometry_id = row["raw_geometry_id"]
            geometry_meta = (
                row["normalized_geometry_id"],
                row["atlas_family"],
                row["atlas_theory"],
            )
            if geometry_id not in reference_geometry_meta:
                reference_geometry_meta[geometry_id] = geometry_meta
            elif reference_geometry_meta[geometry_id] != geometry_meta:
                abort(
                    ctx,
                    f"geometry metadata mismatch across policies for raw_geometry_id={geometry_id!r}: "
                    f"reference={reference_geometry_meta[geometry_id]!r} observed={geometry_meta!r}",
                )
            families.add(row["atlas_family"])
            theories.add(row["atlas_theory"])

        sorted_rows = _sort_rows(weight_payload["rows"])
        total_mass = math.fsum(float(row["weight_normalized"]) for row in sorted_rows)
        if not math.isclose(total_mass, 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE):
            abort(ctx, f"weight_normalized total mismatch for {policy_name!r}: observed={total_mass}")

        d_alpha = renyi_payload["metrics"]["D_alpha"]
        metrics_table[policy_name] = {
            "D_0": float(d_alpha["0"]),
            "D_1": float(d_alpha["1"]),
            "D_2": float(d_alpha["2"]),
            "D_inf": float(d_alpha["inf"]),
            "p_max": float(renyi_payload["metrics"]["p_max"]),
            "n_rows": n_rows,
            "coverage_fraction": float(weight_payload["coverage_fraction"]),
            "weight_policy_file": rec["weight_policy_file"],
            "renyi_diversity_file": rec["renyi_diversity_file"],
        }

        for top_k in K_GRID:
            cumulative_mass = math.fsum(float(row["weight_normalized"]) for row in sorted_rows[:top_k])
            topk_rows.append(
                {
                    "policy_name": policy_name,
                    "top_k": top_k,
                    "cumulative_mass": cumulative_mass,
                    "normalization_check": total_mass,
                    "criterion": "cumulative_topk_mass_over_weight_normalized",
                    "criterion_version": "v1",
                }
            )

        family_rows.extend(
            _ranked_family_or_theory(
                sorted_rows,
                policy_name=policy_name,
                key_name="atlas_family",
                mass_field="family_mass",
                count_field="family_count",
                rank_field="family_mass_rank",
                criterion="mass_sum_by_atlas_family_over_weight_normalized",
            )
        )
        theory_rows.extend(
            _ranked_family_or_theory(
                sorted_rows,
                policy_name=policy_name,
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
            ranking_rows.append(
                {
                    "policy_name": policy_name,
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

    assert reference_basis_name is not None
    assert reference_n_rows is not None

    renyi_comparison_payload = {
        "schema_version": SCHEMA_VERSION,
        "comparison_role": COMPARISON_ROLE,
        "basis_name": reference_basis_name,
        "policies": policies,
        "alpha_grid": ALPHA_GRID,
        "metrics_table": metrics_table,
        "notes": [
            "Comparison of epistemic ensemble diversity across weight policies.",
            "No sector conditioning is applied.",
            "This comparison is not a black-hole thermodynamic entropy claim.",
        ],
    }
    topk_payload = {
        "schema_version": "topk_mass_by_policy_v1",
        "basis_name": reference_basis_name,
        "k_grid": K_GRID,
        "policies": policies,
        "rows": topk_rows,
    }
    family_payload = {
        "schema_version": "family_mass_by_policy_v1",
        "basis_name": reference_basis_name,
        "policies": policies,
        "rows": family_rows,
    }
    theory_payload = {
        "schema_version": "theory_mass_by_policy_v1",
        "basis_name": reference_basis_name,
        "policies": policies,
        "rows": theory_rows,
    }
    ranking_payload = {
        "schema_version": "top_geometry_ranking_by_policy_v1",
        "basis_name": reference_basis_name,
        "policies": policies,
        "top_n": TOP_N,
        "rows": ranking_rows,
    }

    renyi_output_path = ctx.outputs_dir / RENYI_OUTPUT_NAME
    topk_output_path = ctx.outputs_dir / TOPK_OUTPUT_NAME
    family_output_path = ctx.outputs_dir / FAMILY_OUTPUT_NAME
    theory_output_path = ctx.outputs_dir / THEORY_OUTPUT_NAME
    ranking_output_path = ctx.outputs_dir / RANKING_OUTPUT_NAME
    write_json_atomic(renyi_output_path, renyi_comparison_payload)
    write_json_atomic(topk_output_path, topk_payload)
    write_json_atomic(family_output_path, family_payload)
    write_json_atomic(theory_output_path, theory_payload)
    write_json_atomic(ranking_output_path, ranking_payload)

    finalize(
        ctx,
        artifacts={
            "renyi_policy_comparison_v1": renyi_output_path,
            "topk_mass_by_policy_v1": topk_output_path,
            "family_mass_by_policy_v1": family_output_path,
            "theory_mass_by_policy_v1": theory_output_path,
            "top_geometry_ranking_by_policy_v1": ranking_output_path,
        },
        results={
            "n_rows": reference_n_rows,
            "family_count": len(families),
            "theory_count": len(theories),
        },
        extra_summary={
            "schema_version": SCHEMA_VERSION,
            "comparison_role": COMPARISON_ROLE,
            "basis_name": reference_basis_name,
            "policies": policies,
            "n_rows": reference_n_rows,
            "alpha_grid": ALPHA_GRID,
            "k_grid": K_GRID,
            "top_n": TOP_N,
            "family_count": len(families),
            "theory_count": len(theories),
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
