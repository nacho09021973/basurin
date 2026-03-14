#!/usr/bin/env python3
"""Build an auditable geometry_id -> sector map from the phase-1 H5 SSOT."""
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

STAGE = "experiment/phase2_sector_map"
SCHEMA_VERSION = "sector_map_v1"
DEFAULT_INPUT_NAME = "phase1_geometry_cohort.h5"
DEFAULT_OUTPUT_NAME = "sector_map_v1.json"
CRITERION_VERSION = "v1"
ALLOWED_SECTORS = ("HYPERBOLIC", "ELLIPTIC", "EUCLIDEAN", "UNKNOWN")
POLICY_NAME = "explicit_intrinsic_sector_only_v1"
EXPLICIT_LABEL_CRITERION = "explicit_intrinsic_sector_label_v1"
CURVATURE_SIGN_CRITERION = "explicit_intrinsic_curvature_sign_v1"
CONFLICTING_CRITERION = "conflicting_intrinsic_sector_evidence_v1"
UNSUPPORTED_CRITERION = "unsupported_intrinsic_sector_value_v1"
INSUFFICIENT_CRITERION = "insufficient_intrinsic_metadata_v1"

DIRECT_FIELD_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("atlas.geometry_sector", "dataset", ("geometry_sector",)),
    ("atlas.sector", "dataset", ("sector",)),
    ("atlas.constant_curvature_class", "dataset", ("constant_curvature_class",)),
    ("atlas.curvature_sector", "dataset", ("curvature_sector",)),
    ("atlas.entry_json.geometry_sector", "entry_json", ("geometry_sector",)),
    ("atlas.entry_json.sector", "entry_json", ("sector",)),
    ("atlas.entry_json.constant_curvature_class", "entry_json", ("constant_curvature_class",)),
    ("atlas.entry_json.curvature_sector", "entry_json", ("curvature_sector",)),
    ("atlas.entry_json.metadata.geometry_sector", "entry_json", ("metadata", "geometry_sector")),
    ("atlas.entry_json.metadata.sector", "entry_json", ("metadata", "sector")),
    (
        "atlas.entry_json.metadata.constant_curvature_class",
        "entry_json",
        ("metadata", "constant_curvature_class"),
    ),
    ("atlas.entry_json.metadata.curvature_sector", "entry_json", ("metadata", "curvature_sector")),
    (
        "atlas.physical_parameters_json.geometry_sector",
        "physical_parameters_json",
        ("geometry_sector",),
    ),
    ("atlas.physical_parameters_json.sector", "physical_parameters_json", ("sector",)),
    (
        "atlas.physical_parameters_json.constant_curvature_class",
        "physical_parameters_json",
        ("constant_curvature_class",),
    ),
    (
        "atlas.physical_parameters_json.curvature_sector",
        "physical_parameters_json",
        ("curvature_sector",),
    ),
    (
        "atlas.physical_parameters_json.metadata.geometry_sector",
        "physical_parameters_json",
        ("metadata", "geometry_sector"),
    ),
    (
        "atlas.physical_parameters_json.metadata.sector",
        "physical_parameters_json",
        ("metadata", "sector"),
    ),
    (
        "atlas.physical_parameters_json.metadata.constant_curvature_class",
        "physical_parameters_json",
        ("metadata", "constant_curvature_class"),
    ),
    (
        "atlas.physical_parameters_json.metadata.curvature_sector",
        "physical_parameters_json",
        ("metadata", "curvature_sector"),
    ),
)
CURVATURE_SIGN_FIELD_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("atlas.curvature_sign", "dataset", ("curvature_sign",)),
    ("atlas.sectional_curvature_sign", "dataset", ("sectional_curvature_sign",)),
    ("atlas.entry_json.curvature_sign", "entry_json", ("curvature_sign",)),
    ("atlas.entry_json.sectional_curvature_sign", "entry_json", ("sectional_curvature_sign",)),
    ("atlas.entry_json.metadata.curvature_sign", "entry_json", ("metadata", "curvature_sign")),
    (
        "atlas.entry_json.metadata.sectional_curvature_sign",
        "entry_json",
        ("metadata", "sectional_curvature_sign"),
    ),
    (
        "atlas.physical_parameters_json.curvature_sign",
        "physical_parameters_json",
        ("curvature_sign",),
    ),
    (
        "atlas.physical_parameters_json.sectional_curvature_sign",
        "physical_parameters_json",
        ("sectional_curvature_sign",),
    ),
    (
        "atlas.physical_parameters_json.metadata.curvature_sign",
        "physical_parameters_json",
        ("metadata", "curvature_sign"),
    ),
    (
        "atlas.physical_parameters_json.metadata.sectional_curvature_sign",
        "physical_parameters_json",
        ("metadata", "sectional_curvature_sign"),
    ),
)
NEVER_INFER_FROM = (
    "atlas.geometry_id",
    "atlas.theory",
    "atlas.family",
    "atlas.mode",
    "atlas.zeta",
    "atlas.q_charge",
    "atlas.delta_f_frac",
    "atlas.delta_tau_frac",
    "atlas.phi_atlas",
    "atlas.entry_json.theory",
    "atlas.entry_json.metadata.family",
    "atlas.physical_parameters_json",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create an auditable sector map from phase-1 geometry H5")
    ap.add_argument("--run-id", required=True, help="Aggregate run id containing experiment/phase1_geometry_h5")
    ap.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    ap.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return ap.parse_args(argv)


def _coerce_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _is_missing_value(value: Any) -> bool:
    value = _coerce_scalar(value)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "null", "none", "nan"}
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isnan(float(value)) if isinstance(value, float) else False
    return False


def _json_safe_value(value: Any) -> Any:
    value = _coerce_scalar(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _normalize_direct_sector(value: Any) -> str | None:
    value = _coerce_scalar(value)
    if not isinstance(value, str):
        return None
    token = value.strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "HYPERBOLIC": "HYPERBOLIC",
        "ELLIPTIC": "ELLIPTIC",
        "EUCLIDEAN": "EUCLIDEAN",
        "UNKNOWN": "UNKNOWN",
    }
    return mapping.get(token)


def _normalize_curvature_sign(value: Any) -> str | None:
    value = _coerce_scalar(value)
    direct = _normalize_direct_sector(value)
    if direct is not None:
        return direct
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if float(value) < 0.0:
            return "HYPERBOLIC"
        if float(value) > 0.0:
            return "ELLIPTIC"
        return "EUCLIDEAN"
    if not isinstance(value, str):
        return None
    token = value.strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "NEGATIVE": "HYPERBOLIC",
        "NEG": "HYPERBOLIC",
        "_1": "HYPERBOLIC",
        "-1": "HYPERBOLIC",
        "POSITIVE": "ELLIPTIC",
        "POS": "ELLIPTIC",
        "+1": "ELLIPTIC",
        "1": "ELLIPTIC",
        "ZERO": "EUCLIDEAN",
        "0": "EUCLIDEAN",
        "FLAT": "EUCLIDEAN",
    }
    return mapping.get(token)


def _load_json_blob(raw: Any, *, field_path: str, geometry_id: str, ctx: Any, source_h5: Path) -> dict[str, Any] | None:
    if _is_missing_value(raw):
        return None
    text = _coerce_scalar(raw)
    if not isinstance(text, str):
        abort(
            ctx,
            f"Corrupt JSON blob at {field_path} for geometry_id={geometry_id} in {source_h5}: "
            f"expected string, got {type(text).__name__}",
        )
        raise AssertionError("unreachable")
    try:
        payload = json.loads(text)
    except Exception as exc:
        abort(
            ctx,
            f"Corrupt JSON blob at {field_path} for geometry_id={geometry_id} in {source_h5}: {exc}",
        )
        raise AssertionError("unreachable")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        abort(
            ctx,
            f"Unsupported JSON blob at {field_path} for geometry_id={geometry_id} in {source_h5}: "
            f"expected object or null, got {type(payload).__name__}",
        )
        raise AssertionError("unreachable")
    return payload


def _nested_lookup(root: dict[str, Any] | None, parts: tuple[str, ...]) -> tuple[bool, Any]:
    if root is None:
        return False, None
    current: Any = root
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _read_candidate_value(
    atlas_group: Any,
    row_index: int,
    *,
    source_name: str,
    parts: tuple[str, ...],
    entry_json: dict[str, Any] | None,
    physical_parameters_json: dict[str, Any] | None,
) -> tuple[bool, Any]:
    if source_name == "dataset":
        dataset_name = parts[0]
        if dataset_name not in atlas_group:
            return False, None
        return True, _coerce_scalar(atlas_group[dataset_name][row_index])
    if source_name == "entry_json":
        return _nested_lookup(entry_json, parts)
    if source_name == "physical_parameters_json":
        return _nested_lookup(physical_parameters_json, parts)
    raise ValueError(f"Unknown source_name: {source_name}")


def _classify_geometry_row(
    *,
    geometry_id: str,
    atlas_group: Any,
    row_index: int,
    entry_json: dict[str, Any] | None,
    physical_parameters_json: dict[str, Any] | None,
) -> dict[str, Any]:
    mapped_evidence: list[dict[str, Any]] = []
    unsupported_evidence: list[dict[str, Any]] = []

    for field_path, source_name, parts in DIRECT_FIELD_SPECS:
        exists, raw_value = _read_candidate_value(
            atlas_group,
            row_index,
            source_name=source_name,
            parts=parts,
            entry_json=entry_json,
            physical_parameters_json=physical_parameters_json,
        )
        if not exists or _is_missing_value(raw_value):
            continue
        sector = _normalize_direct_sector(raw_value)
        record = {
            "field_path": field_path,
            "raw_value": _json_safe_value(raw_value),
            "kind": "direct",
            "mapped_sector": sector,
        }
        if sector is None:
            unsupported_evidence.append(record)
        else:
            mapped_evidence.append(record)

    for field_path, source_name, parts in CURVATURE_SIGN_FIELD_SPECS:
        exists, raw_value = _read_candidate_value(
            atlas_group,
            row_index,
            source_name=source_name,
            parts=parts,
            entry_json=entry_json,
            physical_parameters_json=physical_parameters_json,
        )
        if not exists or _is_missing_value(raw_value):
            continue
        sector = _normalize_curvature_sign(raw_value)
        record = {
            "field_path": field_path,
            "raw_value": _json_safe_value(raw_value),
            "kind": "curvature_sign",
            "mapped_sector": sector,
        }
        if sector is None:
            unsupported_evidence.append(record)
        else:
            mapped_evidence.append(record)

    if mapped_evidence:
        mapped_sectors = {record["mapped_sector"] for record in mapped_evidence}
        if len(mapped_sectors) == 1:
            sector = next(iter(mapped_sectors))
            criterion = (
                EXPLICIT_LABEL_CRITERION
                if any(record["kind"] == "direct" for record in mapped_evidence)
                else CURVATURE_SIGN_CRITERION
            )
            evidence = mapped_evidence
        else:
            sector = "UNKNOWN"
            criterion = CONFLICTING_CRITERION
            evidence = mapped_evidence
    elif unsupported_evidence:
        sector = "UNKNOWN"
        criterion = UNSUPPORTED_CRITERION
        evidence = unsupported_evidence
    else:
        sector = "UNKNOWN"
        criterion = INSUFFICIENT_CRITERION
        evidence = []

    evidence_fields = [record["field_path"] for record in evidence]
    evidence_payload = {record["field_path"]: record["raw_value"] for record in evidence}
    return {
        "geometry_id": geometry_id,
        "geometry_sector": sector,
        "criterion": criterion,
        "criterion_version": CRITERION_VERSION,
        "evidence_fields_used": evidence_fields,
        "evidence": evidence_payload,
    }


def _classify_h5(source_h5: Path, ctx: Any) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    assert h5py is not None

    try:
        h5 = h5py.File(source_h5, "r")
    except Exception as exc:
        abort(ctx, f"Unable to open H5 input {source_h5}: {exc}")
        raise AssertionError("unreachable")

    with h5:
        if "atlas" not in h5:
            abort(ctx, f"Missing required group atlas in {source_h5}")
            raise AssertionError("unreachable")
        atlas_group = h5["atlas"]
        if "geometry_id" not in atlas_group:
            abort(ctx, f"Missing required dataset atlas/geometry_id in {source_h5}")
            raise AssertionError("unreachable")

        geometry_ids = [_coerce_scalar(value) for value in atlas_group["geometry_id"][...]]
        normalized_ids: list[str] = []
        for raw_id in geometry_ids:
            if not isinstance(raw_id, str) or not raw_id.strip():
                abort(ctx, f"Invalid geometry_id entry in {source_h5}: {raw_id!r}")
                raise AssertionError("unreachable")
            normalized_ids.append(raw_id.strip())

        duplicates = sorted(
            geometry_id
            for geometry_id, count in Counter(normalized_ids).items()
            if count > 1
        )
        if duplicates:
            abort(
                ctx,
                f"Duplicate geometry_id values in {source_h5}: {duplicates[:10]}",
            )
            raise AssertionError("unreachable")

        n_rows = len(normalized_ids)
        rows: list[dict[str, Any]] = []
        sector_counts = Counter()
        criterion_counts = Counter()

        for row_index, geometry_id in enumerate(normalized_ids):
            entry_json = _load_json_blob(
                atlas_group["entry_json"][row_index] if "entry_json" in atlas_group else None,
                field_path="atlas/entry_json",
                geometry_id=geometry_id,
                ctx=ctx,
                source_h5=source_h5,
            )
            physical_parameters_json = _load_json_blob(
                (
                    atlas_group["physical_parameters_json"][row_index]
                    if "physical_parameters_json" in atlas_group
                    else None
                ),
                field_path="atlas/physical_parameters_json",
                geometry_id=geometry_id,
                ctx=ctx,
                source_h5=source_h5,
            )
            row = _classify_geometry_row(
                geometry_id=geometry_id,
                atlas_group=atlas_group,
                row_index=row_index,
                entry_json=entry_json,
                physical_parameters_json=physical_parameters_json,
            )
            rows.append(row)
            sector_counts.update([row["geometry_sector"]])
            criterion_counts.update([row["criterion"]])

        if len(rows) != n_rows:
            abort(ctx, f"Sector map row count mismatch for {source_h5}: {len(rows)} != {n_rows}")
            raise AssertionError("unreachable")

        full_sector_counts = {sector: int(sector_counts.get(sector, 0)) for sector in ALLOWED_SECTORS}
        full_criterion_counts = {
            criterion: int(criterion_counts[criterion])
            for criterion in sorted(criterion_counts)
        }
        return rows, full_sector_counts, full_criterion_counts


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "input_name": args.input_name,
            "output_name": args.output_name,
            "policy_name": POLICY_NAME,
        },
    )

    if h5py is None:
        abort(ctx, "h5py is required to read the phase-1 geometry H5")

    source_h5_rel = Path("experiment") / "phase1_geometry_h5" / "outputs" / args.input_name
    source_h5_path = ctx.run_dir / source_h5_rel
    if not source_h5_path.exists():
        abort(
            ctx,
            f"Missing required input: expected_path={source_h5_path}; "
            f"regen_cmd='python mvp/experiment_phase1_geometry_h5.py --run-id {args.run_id}'",
        )

    check_inputs(ctx, {"phase1_geometry_h5": source_h5_path})

    rows, sector_counts, criterion_counts = _classify_h5(source_h5_path, ctx)
    n_unknown = int(sector_counts.get("UNKNOWN", 0))
    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": args.run_id,
        "source_h5": str(source_h5_rel),
        "allowed_sectors": list(ALLOWED_SECTORS),
        "policy": {
            "name": POLICY_NAME,
            "accepted_direct_field_paths": [field_path for field_path, _, _ in DIRECT_FIELD_SPECS],
            "accepted_curvature_sign_field_paths": [
                field_path for field_path, _, _ in CURVATURE_SIGN_FIELD_SPECS
            ],
            "never_infer_from": list(NEVER_INFER_FROM),
        },
        "n_rows": len(rows),
        "sector_counts": sector_counts,
        "criterion_counts": criterion_counts,
        "rows": rows,
    }

    output_path = ctx.outputs_dir / args.output_name
    write_json_atomic(output_path, output_payload)

    finalize(
        ctx,
        artifacts={"sector_map_v1": output_path},
        results={
            "n_geometry_ids": len(rows),
            "n_unknown": n_unknown,
            "n_classified_non_unknown": len(rows) - n_unknown,
        },
        extra_summary={
            "schema_version": SCHEMA_VERSION,
            "source_h5": str(source_h5_rel),
            "allowed_sectors": list(ALLOWED_SECTORS),
            "classification_policy": {
                "name": POLICY_NAME,
                "accepted_direct_field_paths": [field_path for field_path, _, _ in DIRECT_FIELD_SPECS],
                "accepted_curvature_sign_field_paths": [
                    field_path for field_path, _, _ in CURVATURE_SIGN_FIELD_SPECS
                ],
                "never_infer_from": list(NEVER_INFER_FROM),
            },
            "sector_counts": sector_counts,
            "criterion_counts": criterion_counts,
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
