#!/usr/bin/env python3
"""Build an auditable geometry_id -> family/theory map from phase-1 H5 + source atlas."""
from __future__ import annotations

import argparse
import json
import re
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

STAGE = "experiment/phase2a_atlas_family_map"
SCHEMA_VERSION = "family_map_v1"
DEFAULT_INPUT_NAME = "phase1_geometry_cohort.h5"
DEFAULT_OUTPUT_NAME = "family_map_v1.json"
DEFAULT_ATLAS_PATH = "docs/ringdown/atlas/atlas_berti_v2.json"
NORMALIZATION_POLICY_NAME = "exact_or_normalized_l2m2n0_v1"
EXACT_JOIN_MODE = "exact_match_v1"
NORMALIZED_JOIN_MODE = "normalized_match_l2m2n0_v1"
RESOLVED_STATUS = "RESOLVED"
UNRESOLVED_STATUS = "UNRESOLVED"
CRITERION_VERSION = "v1"
_MODAL_SUFFIX_RE = re.compile(r"_l[^_]+m[^_]+n[^_]+$")


def _repo_root() -> Path:
    return _here.parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create an auditable family/theory map from phase-1 geometry H5")
    ap.add_argument("--run-id", required=True, help="Aggregate run id containing experiment/phase1_geometry_h5")
    ap.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    ap.add_argument("--atlas-path", default=DEFAULT_ATLAS_PATH)
    ap.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    return ap.parse_args(argv)


def _resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    repo_candidate = (_repo_root() / p).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (Path.cwd() / p).resolve()


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(_repo_root()))
    except ValueError:
        return str(path)


def _coerce_text(value: Any) -> str:
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _normalize_geometry_id(raw_geometry_id: str) -> str:
    if _MODAL_SUFFIX_RE.search(raw_geometry_id):
        return raw_geometry_id
    return f"{raw_geometry_id}_l2m2n0"


def _load_atlas_rows(atlas_path: Path, ctx: Any) -> list[dict[str, Any]]:
    try:
        payload = json.loads(atlas_path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt atlas JSON at {atlas_path}: {exc}")
        raise AssertionError("unreachable")

    rows: Any
    if isinstance(payload, dict):
        if isinstance(payload.get("entries"), list):
            rows = payload["entries"]
        elif isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        else:
            abort(ctx, f"Atlas JSON at {atlas_path} lacks entries/rows list")
            raise AssertionError("unreachable")
    elif isinstance(payload, list):
        rows = payload
    else:
        abort(ctx, f"Atlas JSON at {atlas_path} must be a list or dict, got {type(payload).__name__}")
        raise AssertionError("unreachable")

    filtered = [row for row in rows if isinstance(row, dict)]
    if len(filtered) != len(rows):
        abort(ctx, f"Atlas JSON at {atlas_path} contains non-object rows")
        raise AssertionError("unreachable")
    return filtered


def _atlas_by_geometry_id(rows: list[dict[str, Any]], ctx: Any, atlas_path: Path) -> dict[str, dict[str, Any]]:
    counts = Counter()
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        geometry_id = _coerce_text(row.get("geometry_id")).strip()
        if not geometry_id:
            abort(ctx, f"Atlas row missing geometry_id at {atlas_path}")
            raise AssertionError("unreachable")
        counts[geometry_id] += 1
        indexed[geometry_id] = row
    duplicates = sorted(geometry_id for geometry_id, count in counts.items() if count > 1)
    if duplicates:
        abort(ctx, f"Duplicate geometry_id values in atlas {atlas_path}: {duplicates[:10]}")
        raise AssertionError("unreachable")
    return indexed


def _read_h5_geometry_ids(source_h5: Path, ctx: Any) -> list[str]:
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

        geometry_ids: list[str] = []
        for index, raw_value in enumerate(atlas_group["geometry_id"][...]):
            geometry_id = _coerce_text(raw_value).strip()
            if not geometry_id:
                abort(ctx, f"Empty geometry_id at atlas/geometry_id[{index}] in {source_h5}")
                raise AssertionError("unreachable")
            geometry_ids.append(geometry_id)
        return geometry_ids


def _recover_atlas_ontology(
    raw_geometry_id: str,
    *,
    atlas_rows_by_gid: dict[str, dict[str, Any]],
    atlas_display_path: str,
) -> tuple[dict[str, Any] | None, str | None]:
    atlas_row = atlas_rows_by_gid.get(raw_geometry_id)
    if atlas_row is not None:
        return atlas_row, EXACT_JOIN_MODE

    normalized_geometry_id = _normalize_geometry_id(raw_geometry_id)
    atlas_row = atlas_rows_by_gid.get(normalized_geometry_id)
    if atlas_row is not None:
        return atlas_row, NORMALIZED_JOIN_MODE

    return None, None


def _resolved_row(
    *,
    raw_geometry_id: str,
    atlas_row: dict[str, Any],
    join_mode: str,
    atlas_display_path: str,
) -> dict[str, Any]:
    normalized_geometry_id = raw_geometry_id if join_mode == EXACT_JOIN_MODE else _normalize_geometry_id(raw_geometry_id)
    atlas_geometry_id = _coerce_text(atlas_row.get("geometry_id")).strip()
    atlas_theory = _coerce_text(atlas_row.get("theory")).strip()
    atlas_metadata = atlas_row.get("metadata")
    atlas_family = (
        _coerce_text(atlas_metadata.get("family")).strip()
        if isinstance(atlas_metadata, dict)
        else ""
    )
    evidence_fields_used = [
        "h5.atlas.geometry_id",
        "atlas_source.geometry_id",
        "atlas_source.theory",
        "atlas_source.metadata.family",
    ]
    evidence = {
        "h5.atlas.geometry_id": raw_geometry_id,
        "atlas_source.geometry_id": atlas_geometry_id,
        "atlas_source.theory": atlas_theory,
        "atlas_source.metadata.family": atlas_family,
    }
    if join_mode == NORMALIZED_JOIN_MODE:
        evidence_fields_used.insert(1, "normalized_geometry_id")
        evidence["normalized_geometry_id"] = normalized_geometry_id

    return {
        "raw_geometry_id": raw_geometry_id,
        "normalized_geometry_id": normalized_geometry_id,
        "join_mode": join_mode,
        "join_status": RESOLVED_STATUS,
        "atlas_path": atlas_display_path,
        "atlas_geometry_id": atlas_geometry_id,
        "atlas_family": atlas_family,
        "atlas_theory": atlas_theory,
        "criterion": join_mode,
        "criterion_version": CRITERION_VERSION,
        "evidence_fields_used": evidence_fields_used,
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "input_name": args.input_name,
            "atlas_path": args.atlas_path,
            "output_name": args.output_name,
            "normalization_policy_name": NORMALIZATION_POLICY_NAME,
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

    atlas_path = _resolve_path(args.atlas_path)
    if not atlas_path.exists():
        abort(ctx, f"Missing atlas input: expected_path={atlas_path}")

    atlas_display_path = _display_path(atlas_path)
    check_inputs(
        ctx,
        {
            "phase1_geometry_h5": source_h5_path,
            "atlas_source": atlas_path,
        },
    )

    geometry_ids = _read_h5_geometry_ids(source_h5_path, ctx)
    atlas_rows = _load_atlas_rows(atlas_path, ctx)
    atlas_rows_by_gid = _atlas_by_geometry_id(atlas_rows, ctx, atlas_path)

    rows: list[dict[str, Any]] = []
    unresolved_geometry_ids: list[str] = []
    join_mode_counts = Counter()
    family_counts = Counter()
    theory_counts = Counter()

    for raw_geometry_id in geometry_ids:
        atlas_row, join_mode = _recover_atlas_ontology(
            raw_geometry_id,
            atlas_rows_by_gid=atlas_rows_by_gid,
            atlas_display_path=atlas_display_path,
        )
        if atlas_row is None or join_mode is None:
            unresolved_geometry_ids.append(raw_geometry_id)
            continue

        resolved = _resolved_row(
            raw_geometry_id=raw_geometry_id,
            atlas_row=atlas_row,
            join_mode=join_mode,
            atlas_display_path=atlas_display_path,
        )
        if not resolved["atlas_theory"] or not resolved["atlas_family"]:
            unresolved_geometry_ids.append(raw_geometry_id)
            continue

        rows.append(resolved)
        join_mode_counts.update([resolved["join_mode"]])
        family_counts.update([resolved["atlas_family"]])
        theory_counts.update([resolved["atlas_theory"]])

    n_rows = len(geometry_ids)
    n_unresolved = len(unresolved_geometry_ids)
    n_exact_match = int(join_mode_counts.get(EXACT_JOIN_MODE, 0))
    n_normalized_match = int(join_mode_counts.get(NORMALIZED_JOIN_MODE, 0))

    if len(rows) + n_unresolved != n_rows:
        abort(
            ctx,
            f"Family map row accounting mismatch: resolved={len(rows)} unresolved={n_unresolved} total={n_rows}",
        )

    if n_unresolved > 0:
        abort(
            ctx,
            f"Unresolved atlas family/theory join for {n_unresolved}/{n_rows} rows; "
            f"normalization_policy_name={NORMALIZATION_POLICY_NAME}; "
            f"unresolved_geometry_ids={unresolved_geometry_ids[:10]}",
        )

    join_mode_counts_payload = {
        EXACT_JOIN_MODE: n_exact_match,
        NORMALIZED_JOIN_MODE: n_normalized_match,
    }
    family_counts_payload = {
        family: int(family_counts[family])
        for family in sorted(family_counts)
    }
    theory_counts_payload = {
        theory: int(theory_counts[theory])
        for theory in sorted(theory_counts)
    }

    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "normalization_policy_name": NORMALIZATION_POLICY_NAME,
        "source_h5": str(source_h5_rel),
        "source_atlas": atlas_display_path,
        "n_rows": n_rows,
        "family_counts": family_counts_payload,
        "theory_counts": theory_counts_payload,
        "join_mode_counts": join_mode_counts_payload,
        "unresolved_geometry_ids": [],
        "rows": rows,
    }
    output_path = ctx.outputs_dir / args.output_name
    write_json_atomic(output_path, output_payload)

    summary_fields = {
        "schema_version": SCHEMA_VERSION,
        "normalization_policy_name": NORMALIZATION_POLICY_NAME,
        "source_h5": str(source_h5_rel),
        "source_atlas": atlas_display_path,
        "n_rows": n_rows,
        "n_exact_match": n_exact_match,
        "n_normalized_match": n_normalized_match,
        "n_unresolved": 0,
        "family_counts": family_counts_payload,
        "theory_counts": theory_counts_payload,
        "join_mode_counts": join_mode_counts_payload,
        "unresolved_geometry_ids": [],
    }
    finalize(
        ctx,
        artifacts={"family_map_v1": output_path},
        results={
            "n_rows": n_rows,
            "n_exact_match": n_exact_match,
            "n_normalized_match": n_normalized_match,
            "n_unresolved": 0,
        },
        extra_summary=summary_fields,
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
