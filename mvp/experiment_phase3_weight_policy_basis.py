#!/usr/bin/env python3
"""Materialize the canonical uniform weight policy over the supported basis."""
from __future__ import annotations

import argparse
import json
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

STAGE = "experiment/phase3_weight_policy_basis"
SCHEMA_VERSION = "weight_policy_basis_v1"
SUPPORTED_POLICY_NAME = "uniform_support_v1"
POLICY_ROLE = "baseline_canonical"
NORMALIZATION_METHOD = "divide_by_sum_raw_over_all_rows"
WEIGHT_STATUS = "WEIGHTED"
CRITERION = "uniform_over_support_basis"
CRITERION_VERSION = "v1"
DEFAULT_INPUT_NAME = "support_ontology_basis_v1.json"
DEFAULT_OUTPUT_NAME = "weight_policy_basis_v1.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize the canonical uniform weight policy basis")
    ap.add_argument("--run-id", required=True, help="Aggregate run id containing phase2c support basis")
    ap.add_argument("--policy-name", default=SUPPORTED_POLICY_NAME)
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

    if args.policy_name != SUPPORTED_POLICY_NAME:
        abort(
            ctx,
            f"Unsupported policy_name={args.policy_name!r}; only {SUPPORTED_POLICY_NAME!r} is implemented in this v1",
        )

    source_basis_rel = Path("experiment") / "phase2c_support_ontology_basis" / "outputs" / args.input_name
    source_basis_path = ctx.run_dir / source_basis_rel
    check_inputs(ctx, {"support_ontology_basis_v1": source_basis_path})

    phase2c = _load_phase2c_basis(source_basis_path, ctx)
    basis_rows = phase2c["rows"]
    n_rows = len(basis_rows)
    weight_normalized = 1.0 / float(n_rows)

    rows: list[dict[str, Any]] = []
    for idx, basis_row in enumerate(basis_rows):
        if not isinstance(basis_row, dict):
            abort(ctx, f"phase2c support basis row[{idx}] is not an object")
            raise AssertionError("unreachable")

        row = {
            "raw_geometry_id": _require_text(basis_row, "raw_geometry_id", row_index=idx, ctx=ctx),
            "normalized_geometry_id": _require_text(basis_row, "normalized_geometry_id", row_index=idx, ctx=ctx),
            "atlas_family": _require_text(basis_row, "atlas_family", row_index=idx, ctx=ctx),
            "atlas_theory": _require_text(basis_row, "atlas_theory", row_index=idx, ctx=ctx),
            "policy_name": SUPPORTED_POLICY_NAME,
            "weight_raw": 1.0,
            "weight_normalized": weight_normalized,
            "weight_status": WEIGHT_STATUS,
            "support_count_events": int(_require_number(basis_row, "n_events_supported", row_index=idx, ctx=ctx)),
            "support_fraction_events": _require_number(basis_row, "support_fraction_events", row_index=idx, ctx=ctx),
            "source_artifacts": [str(source_basis_rel)],
            "criterion": CRITERION,
            "criterion_version": CRITERION_VERSION,
            "evidence": {
                "basis_name": phase2c["basis_name"],
                "uniform_weight_raw": 1.0,
                "uniform_weight_normalized": weight_normalized,
            },
        }
        rows.append(row)

    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "basis_name": phase2c["basis_name"],
        "policy_name": SUPPORTED_POLICY_NAME,
        "policy_role": POLICY_ROLE,
        "coverage_fraction": 1.0,
        "n_rows": n_rows,
        "n_weighted": n_rows,
        "n_unweighted": 0,
        "weight_sum_raw": float(n_rows),
        "weight_sum_normalized": 1.0,
        "normalization_method": NORMALIZATION_METHOD,
        "source_policy_inputs": [],
        "rows": rows,
    }

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
            "policy_name": SUPPORTED_POLICY_NAME,
            "policy_role": POLICY_ROLE,
            "coverage_fraction": 1.0,
            "n_rows": n_rows,
            "n_weighted": n_rows,
            "n_unweighted": 0,
            "weight_sum_raw": float(n_rows),
            "weight_sum_normalized": 1.0,
            "normalization_method": NORMALIZATION_METHOD,
            "source_policy_inputs": [],
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
