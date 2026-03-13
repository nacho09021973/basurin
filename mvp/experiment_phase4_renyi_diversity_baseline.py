#!/usr/bin/env python3
"""Compute baseline Renyi diversity metrics over a supported ensemble."""
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

STAGE = "experiment/phase4_renyi_diversity_baseline"
SCHEMA_VERSION = "renyi_diversity_baseline_v1"
WEIGHT_POLICY_SCHEMA_VERSION = "weight_policy_basis_v1"
DEFAULT_WEIGHT_POLICY_FILE = "weight_policy_basis_v1.json"
DEFAULT_OUTPUT_NAME = "renyi_diversity_baseline_v1.json"
ALPHA_GRID: list[int | str] = [0, 1, 2, "inf"]
METRIC_ROLE = "epistemic_ensemble_diversity"
NORMALIZATION_TOLERANCE = 1e-9


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute Renyi diversity metrics over a single weight policy basis")
    ap.add_argument("--run-id", required=True, help="Aggregate run id containing the phase3 weight policy output")
    ap.add_argument("--weight-policy-file", required=True, help="Filename under experiment/phase3_weight_policy_basis/outputs/")
    ap.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return ap.parse_args(argv)


def _coerce_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _require_text(payload: dict[str, Any], field_name: str, *, ctx: Any) -> str:
    value = _coerce_text(payload.get(field_name)).strip()
    if not value:
        abort(ctx, f"weight_policy_basis_v1 missing {field_name}")
        raise AssertionError("unreachable")
    return value


def _load_weight_policy(path: Path, *, ctx: Any) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt weight_policy_basis_v1 at {path}: {exc}")
        raise AssertionError("unreachable")

    if not isinstance(payload, dict) or payload.get("schema_version") != WEIGHT_POLICY_SCHEMA_VERSION:
        abort(ctx, f"Expected schema_version={WEIGHT_POLICY_SCHEMA_VERSION} at {path}")
        raise AssertionError("unreachable")

    basis_name = _require_text(payload, "basis_name", ctx=ctx)
    policy_name = _require_text(payload, "policy_name", ctx=ctx)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        abort(ctx, f"weight_policy_basis_v1 at {path} missing non-empty rows")
        raise AssertionError("unreachable")

    try:
        n_rows = int(payload.get("n_rows"))
    except Exception:
        abort(ctx, f"weight_policy_basis_v1 at {path} missing integer n_rows")
        raise AssertionError("unreachable")
    if n_rows != len(rows):
        abort(ctx, f"weight_policy_basis_v1 n_rows mismatch at {path}: declared={n_rows} actual={len(rows)}")
        raise AssertionError("unreachable")

    try:
        coverage_fraction = float(payload.get("coverage_fraction"))
    except Exception:
        abort(ctx, f"weight_policy_basis_v1 at {path} missing numeric coverage_fraction")
        raise AssertionError("unreachable")

    try:
        declared_weight_sum_normalized = float(payload.get("weight_sum_normalized"))
    except Exception:
        abort(ctx, f"weight_policy_basis_v1 at {path} missing numeric weight_sum_normalized")
        raise AssertionError("unreachable")

    positive_weights: list[float] = []
    row_weight_sum_normalized = 0.0
    validated_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            abort(ctx, f"weight_policy_basis_v1 row[{idx}] is not an object")
            raise AssertionError("unreachable")
        row_policy_name = _coerce_text(row.get("policy_name")).strip()
        if row_policy_name and row_policy_name != policy_name:
            abort(
                ctx,
                f"weight_policy_basis_v1 row[{idx}] policy_name mismatch: "
                f"top_level={policy_name!r} row={row_policy_name!r}",
            )
            raise AssertionError("unreachable")
        try:
            weight_normalized = float(row.get("weight_normalized"))
        except Exception:
            abort(ctx, f"weight_policy_basis_v1 row[{idx}] missing numeric weight_normalized")
            raise AssertionError("unreachable")
        if not math.isfinite(weight_normalized):
            abort(ctx, f"weight_policy_basis_v1 row[{idx}] has non-finite weight_normalized={weight_normalized!r}")
            raise AssertionError("unreachable")
        if weight_normalized < 0.0:
            abort(ctx, f"weight_policy_basis_v1 row[{idx}] has negative weight_normalized={weight_normalized}")
            raise AssertionError("unreachable")
        row_weight_sum_normalized += weight_normalized
        if weight_normalized > 0.0:
            positive_weights.append(weight_normalized)
        validated_rows.append(row)

    if not math.isclose(declared_weight_sum_normalized, 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE):
        abort(
            ctx,
            f"weight_policy_basis_v1 weight_sum_normalized must be within {NORMALIZATION_TOLERANCE} of 1.0: "
            f"observed={declared_weight_sum_normalized}",
        )
        raise AssertionError("unreachable")

    if not math.isclose(row_weight_sum_normalized, 1.0, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE):
        abort(
            ctx,
            f"row weight_normalized sum must be within {NORMALIZATION_TOLERANCE} of 1.0: "
            f"observed={row_weight_sum_normalized}",
        )
        raise AssertionError("unreachable")

    if not math.isclose(row_weight_sum_normalized, declared_weight_sum_normalized, rel_tol=0.0, abs_tol=NORMALIZATION_TOLERANCE):
        abort(
            ctx,
            f"weight_policy_basis_v1 declared weight_sum_normalized mismatch: "
            f"declared={declared_weight_sum_normalized} observed={row_weight_sum_normalized}",
        )
        raise AssertionError("unreachable")

    if not positive_weights:
        abort(ctx, "weight_policy_basis_v1 has no positive normalized weights")
        raise AssertionError("unreachable")

    return {
        "basis_name": basis_name,
        "policy_name": policy_name,
        "n_rows": n_rows,
        "coverage_fraction": coverage_fraction,
        "weight_sum_normalized": declared_weight_sum_normalized,
        "rows": validated_rows,
        "positive_weights": positive_weights,
    }


def _renyi_metrics(positive_weights: list[float]) -> dict[str, Any]:
    n_weighted = len(positive_weights)
    p_max = max(positive_weights)
    h_alpha = {
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
        "n_unweighted": None,
    }


def _output_rel(output_name: str) -> Path:
    return Path(STAGE) / "outputs" / output_name


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "weight_policy_file": args.weight_policy_file,
            "output_name": args.output_name,
            "alpha_grid": ALPHA_GRID,
            "normalization_tolerance": NORMALIZATION_TOLERANCE,
        },
    )

    weight_policy_rel = Path("experiment") / "phase3_weight_policy_basis" / "outputs" / args.weight_policy_file
    weight_policy_path = ctx.run_dir / weight_policy_rel
    check_inputs(ctx, {"weight_policy_basis_v1": weight_policy_path})

    payload = _load_weight_policy(weight_policy_path, ctx=ctx)
    positive_weights = payload["positive_weights"]
    metrics = _renyi_metrics(positive_weights)
    metrics["n_unweighted"] = payload["n_rows"] - metrics["n_weighted"]
    metrics["weight_sum_normalized"] = payload["weight_sum_normalized"]

    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "metric_role": METRIC_ROLE,
        "basis_name": payload["basis_name"],
        "policy_name": payload["policy_name"],
        "weight_policy_file": args.weight_policy_file,
        "n_rows": payload["n_rows"],
        "coverage_fraction": payload["coverage_fraction"],
        "alpha_grid": ALPHA_GRID,
        "metrics": metrics,
        "notes": [
            "Epistemic ensemble diversity over the full supported basis.",
            "No sector conditioning is applied in this baseline.",
            "This output is not a black-hole thermodynamic entropy claim.",
        ],
    }

    output_path = ctx.outputs_dir / args.output_name
    write_json_atomic(output_path, output_payload)

    finalize(
        ctx,
        artifacts={"renyi_diversity_baseline_v1": output_path},
        results={
            "n_rows": payload["n_rows"],
            "coverage_fraction": payload["coverage_fraction"],
            "n_weighted": metrics["n_weighted"],
            "n_unweighted": metrics["n_unweighted"],
        },
        extra_summary={
            "schema_version": SCHEMA_VERSION,
            "metric_role": METRIC_ROLE,
            "basis_name": payload["basis_name"],
            "policy_name": payload["policy_name"],
            "weight_policy_file": args.weight_policy_file,
            "n_rows": payload["n_rows"],
            "coverage_fraction": payload["coverage_fraction"],
            "alpha_grid": ALPHA_GRID,
            "metrics": metrics,
            "output_name": args.output_name,
            "output_path": str(_output_rel(args.output_name)),
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
