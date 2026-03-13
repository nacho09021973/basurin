#!/usr/bin/env python3
"""Freeze the factual ontology basis of the supported ensemble."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    h5py = None  # type: ignore[assignment]

from basurin_io import write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "experiment/phase2c_support_ontology_basis"
SCHEMA_VERSION = "support_ontology_basis_v1"
FAMILY_MAP_SCHEMA_VERSION = "family_map_v1"
DEFAULT_INPUT_NAME = "phase1_geometry_cohort.h5"
DEFAULT_FAMILY_MAP_NAME = "family_map_v1.json"
DEFAULT_OUTPUT_NAME = "support_ontology_basis_v1.json"
BASIS_NAME = "final_support_region_union_v1"
BASIS_EQUIVALENCE_CHECK = "final_support_region_union_vs_golden_post_hawking_union_v1"
JOINT_WEIGHT_ROLE = "copied_for_audit_only_not_a_renyi_policy"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Freeze the factual ontology basis of the supported ensemble")
    ap.add_argument("--run-id", required=True, help="Aggregate run id containing phase1 + phase2a artifacts")
    ap.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    ap.add_argument("--family-map-name", default=DEFAULT_FAMILY_MAP_NAME)
    ap.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return ap.parse_args(argv)


def _coerce_text(value: Any) -> str:
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _int_or_none(value: Any) -> int | None:
    try:
        out = int(value)
    except Exception:
        return None
    if out < 0:
        return None
    return out


def _load_family_map(path: Path, ctx: Any) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt family_map_v1 at {path}: {exc}")
        raise AssertionError("unreachable")

    if not isinstance(payload, dict) or payload.get("schema_version") != FAMILY_MAP_SCHEMA_VERSION:
        abort(ctx, f"Expected schema_version={FAMILY_MAP_SCHEMA_VERSION} at {path}")
        raise AssertionError("unreachable")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        abort(ctx, f"Family map at {path} lacks rows list")
        raise AssertionError("unreachable")

    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            abort(ctx, f"Family map at {path} contains non-object row")
            raise AssertionError("unreachable")
        raw_geometry_id = _coerce_text(row.get("raw_geometry_id")).strip()
        if not raw_geometry_id:
            abort(ctx, f"Family map row missing raw_geometry_id at {path}")
            raise AssertionError("unreachable")
        if raw_geometry_id in indexed:
            duplicates.append(raw_geometry_id)
        indexed[raw_geometry_id] = row
    if duplicates:
        abort(ctx, f"Duplicate raw_geometry_id values in family map {path}: {sorted(set(duplicates))[:10]}")
        raise AssertionError("unreachable")
    return indexed


def _require_text(row: dict[str, Any], field_name: str, *, geometry_id: str, ctx: Any) -> str:
    value = _coerce_text(row.get(field_name)).strip()
    if not value:
        abort(ctx, f"family_map_v1 missing {field_name} for geometry_id={geometry_id}")
        raise AssertionError("unreachable")
    return value


def _load_phase1_h5(path: Path, ctx: Any) -> dict[str, Any]:
    assert h5py is not None

    try:
        h5 = h5py.File(path, "r")
    except Exception as exc:
        abort(ctx, f"Unable to open H5 input {path}: {exc}")
        raise AssertionError("unreachable")

    with h5:
        if "atlas" not in h5 or "membership" not in h5 or "joint_posterior" not in h5:
            abort(ctx, f"Missing required atlas/membership/joint_posterior groups in {path}")
            raise AssertionError("unreachable")

        atlas_group = h5["atlas"]
        membership_group = h5["membership"]
        joint_group = h5["joint_posterior"]

        required_atlas = ("geometry_id",)
        required_membership = ("golden_post_hawking", "final_support_region")
        required_joint = ("available", "posterior_weight_joint", "support_count", "support_fraction", "coverage")

        for name in required_atlas:
            if name not in atlas_group:
                abort(ctx, f"Missing required dataset atlas/{name} in {path}")
                raise AssertionError("unreachable")
        for name in required_membership:
            if name not in membership_group:
                abort(ctx, f"Missing required dataset membership/{name} in {path}")
                raise AssertionError("unreachable")
        for name in required_joint:
            if name not in joint_group:
                abort(ctx, f"Missing required dataset joint_posterior/{name} in {path}")
                raise AssertionError("unreachable")

        geometry_ids = [_coerce_text(value).strip() for value in atlas_group["geometry_id"][...]]
        if not geometry_ids or any(not geometry_id for geometry_id in geometry_ids):
            abort(ctx, f"Invalid atlas/geometry_id dataset in {path}")
            raise AssertionError("unreachable")
        n_geometry = len(geometry_ids)

        golden_matrix = membership_group["golden_post_hawking"][...]
        final_matrix = membership_group["final_support_region"][...]
        if len(golden_matrix.shape) != 2 or len(final_matrix.shape) != 2:
            abort(
                ctx,
                f"Membership datasets must be 2D in {path}: "
                f"golden_post_hawking.shape={golden_matrix.shape} final_support_region.shape={final_matrix.shape}",
            )
            raise AssertionError("unreachable")
        if golden_matrix.shape[1] != n_geometry or final_matrix.shape[1] != n_geometry:
            abort(
                ctx,
                f"Membership dataset width mismatch in {path}: "
                f"golden_post_hawking.shape={golden_matrix.shape} "
                f"final_support_region.shape={final_matrix.shape} geometry_ids={n_geometry}",
            )
            raise AssertionError("unreachable")
        if golden_matrix.shape[0] != final_matrix.shape[0]:
            abort(
                ctx,
                f"Membership dataset row mismatch in {path}: "
                f"golden_post_hawking.shape={golden_matrix.shape} final_support_region.shape={final_matrix.shape}",
            )
            raise AssertionError("unreachable")

        n_events = int(final_matrix.shape[0])
        final_union = [bool(final_matrix[:, idx].any()) for idx in range(n_geometry)]
        golden_union = [bool(golden_matrix[:, idx].any()) for idx in range(n_geometry)]
        n_events_supported = [int(final_matrix[:, idx].sum()) for idx in range(n_geometry)]

        available = [bool(value) for value in joint_group["available"][...]]
        posterior_weight_joint = [_float_or_none(value) for value in joint_group["posterior_weight_joint"][...]]
        support_count = [_int_or_none(value) for value in joint_group["support_count"][...]]
        support_fraction = [_float_or_none(value) for value in joint_group["support_fraction"][...]]
        coverage = [_float_or_none(value) for value in joint_group["coverage"][...]]

        for label, values in (
            ("available", available),
            ("posterior_weight_joint", posterior_weight_joint),
            ("support_count", support_count),
            ("support_fraction", support_fraction),
            ("coverage", coverage),
        ):
            if len(values) != n_geometry:
                abort(ctx, f"joint_posterior/{label} length mismatch in {path}: {len(values)} != {n_geometry}")
                raise AssertionError("unreachable")

        return {
            "geometry_ids": geometry_ids,
            "n_events": n_events,
            "final_union": final_union,
            "golden_union": golden_union,
            "n_events_supported": n_events_supported,
            "joint_available": available,
            "joint_posterior_weight_joint": posterior_weight_joint,
            "joint_support_count": support_count,
            "joint_support_fraction": support_fraction,
            "joint_coverage": coverage,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "input_name": args.input_name,
            "family_map_name": args.family_map_name,
            "output_name": args.output_name,
            "basis_name": BASIS_NAME,
            "basis_equivalence_check": BASIS_EQUIVALENCE_CHECK,
        },
    )

    if h5py is None:
        abort(ctx, "h5py is required to read the phase-1 geometry H5")

    source_h5_rel = Path("experiment") / "phase1_geometry_h5" / "outputs" / args.input_name
    source_family_map_rel = Path("experiment") / "phase2a_atlas_family_map" / "outputs" / args.family_map_name
    source_h5_path = ctx.run_dir / source_h5_rel
    source_family_map_path = ctx.run_dir / source_family_map_rel

    check_inputs(
        ctx,
        {
            "phase1_geometry_h5": source_h5_path,
            "family_map_v1": source_family_map_path,
        },
    )

    phase1 = _load_phase1_h5(source_h5_path, ctx)
    family_map_by_gid = _load_family_map(source_family_map_path, ctx)

    rows: list[dict[str, Any]] = []
    family_counts = Counter()
    theory_counts = Counter()
    n_joint_available = 0
    joint_weight_sum_over_support = 0.0

    for idx, geometry_id in enumerate(phase1["geometry_ids"]):
        if not phase1["final_union"][idx]:
            continue

        family_row = family_map_by_gid.get(geometry_id)
        if family_row is None:
            abort(ctx, f"Supported geometry_id missing from family_map_v1: {geometry_id}")
            raise AssertionError("unreachable")

        raw_geometry_id = _require_text(family_row, "raw_geometry_id", geometry_id=geometry_id, ctx=ctx)
        if raw_geometry_id != geometry_id:
            abort(
                ctx,
                f"family_map_v1 raw_geometry_id mismatch for supported geometry: "
                f"expected={geometry_id} actual={raw_geometry_id}",
            )
            raise AssertionError("unreachable")

        normalized_geometry_id = _require_text(
            family_row, "normalized_geometry_id", geometry_id=geometry_id, ctx=ctx
        )
        atlas_family = _require_text(family_row, "atlas_family", geometry_id=geometry_id, ctx=ctx)
        atlas_theory = _require_text(family_row, "atlas_theory", geometry_id=geometry_id, ctx=ctx)
        join_mode = _require_text(family_row, "join_mode", geometry_id=geometry_id, ctx=ctx)
        join_status = _require_text(family_row, "join_status", geometry_id=geometry_id, ctx=ctx)

        joint_available = bool(phase1["joint_available"][idx])
        joint_weight = phase1["joint_posterior_weight_joint"][idx]
        if joint_available:
            n_joint_available += 1
        if joint_weight is not None:
            joint_weight_sum_over_support += joint_weight

        n_events_supported = int(phase1["n_events_supported"][idx])
        support_fraction_events = float(n_events_supported) / float(phase1["n_events"])

        row = {
            "raw_geometry_id": raw_geometry_id,
            "normalized_geometry_id": normalized_geometry_id,
            "atlas_family": atlas_family,
            "atlas_theory": atlas_theory,
            "join_mode": join_mode,
            "join_status": join_status,
            "support_basis_name": BASIS_NAME,
            "n_events_supported": n_events_supported,
            "support_fraction_events": support_fraction_events,
            "in_golden_post_hawking_union": bool(phase1["golden_union"][idx]),
            "in_final_support_region_union": True,
            "joint_available": joint_available,
            "joint_posterior_weight_joint": joint_weight,
            "joint_support_count": phase1["joint_support_count"][idx],
            "joint_support_fraction": phase1["joint_support_fraction"][idx],
            "joint_coverage": phase1["joint_coverage"][idx],
        }
        rows.append(row)
        family_counts.update([atlas_family])
        theory_counts.update([atlas_theory])

    family_counts_payload = {
        family: int(family_counts[family])
        for family in sorted(family_counts)
    }
    theory_counts_payload = {
        theory: int(theory_counts[theory])
        for theory in sorted(theory_counts)
    }
    bases_equal = phase1["final_union"] == phase1["golden_union"]

    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_h5": str(source_h5_rel),
        "source_family_map": str(source_family_map_rel),
        "basis_name": BASIS_NAME,
        "basis_equivalence_check": BASIS_EQUIVALENCE_CHECK,
        "bases_equal": bases_equal,
        "n_events": phase1["n_events"],
        "n_rows": len(rows),
        "family_counts": family_counts_payload,
        "theory_counts": theory_counts_payload,
        "n_joint_available": n_joint_available,
        "joint_weight_sum_over_support": joint_weight_sum_over_support,
        "joint_weight_role": JOINT_WEIGHT_ROLE,
        "rows": rows,
    }

    output_path = ctx.outputs_dir / args.output_name
    write_json_atomic(output_path, output_payload)

    finalize(
        ctx,
        artifacts={"support_ontology_basis_v1": output_path},
        results={
            "n_rows": len(rows),
            "bases_equal": bases_equal,
            "n_joint_available": n_joint_available,
        },
        extra_summary={
            "schema_version": SCHEMA_VERSION,
            "source_h5": str(source_h5_rel),
            "source_family_map": str(source_family_map_rel),
            "basis_name": BASIS_NAME,
            "basis_equivalence_check": BASIS_EQUIVALENCE_CHECK,
            "bases_equal": bases_equal,
            "n_rows": len(rows),
            "family_counts": family_counts_payload,
            "theory_counts": theory_counts_payload,
            "n_joint_available": n_joint_available,
            "joint_weight_sum_over_support": joint_weight_sum_over_support,
            "joint_weight_role": JOINT_WEIGHT_ROLE,
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
