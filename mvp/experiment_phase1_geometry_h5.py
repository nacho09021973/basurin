#!/usr/bin/env python3
"""Freeze the final phase-1 geometry cohort into a self-contained HDF5 archive."""
from __future__ import annotations

import argparse
import json
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
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime
    h5py = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

from basurin_io import require_run_valid, sha256_file, utc_now_iso, write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths

STAGE = "experiment/phase1_geometry_h5"
SCHEMA_VERSION = "experiment_phase1_geometry_h5_v1"
DEFAULT_OUTPUT_NAME = "phase1_geometry_cohort.h5"
DEFAULT_SUMMARY_NAME = "phase1_geometry_summary.json"


def _repo_root() -> Path:
    return _here.parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Freeze phase-1 geometry outputs into a self-contained HDF5 archive")
    ap.add_argument("--run-id", required=True, help="Aggregate run id containing s5_aggregate/outputs/aggregate.json")
    ap.add_argument(
        "--atlas-path",
        default=None,
        help="Optional atlas JSON path. Defaults to the common atlas recorded in source run provenance.",
    )
    ap.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    ap.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    return ap.parse_args(argv)


def _read_json_object(path: Path, label: str, ctx: Any) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        abort(ctx, f"Corrupt input at {path} ({label}): {exc}")
        raise AssertionError("unreachable")
    if not isinstance(payload, dict):
        abort(ctx, f"Expected JSON object at {path} ({label}), got {type(payload).__name__}")
        raise AssertionError("unreachable")
    return payload


def _resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def _resolve_provenance_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    repo_candidate = (_repo_root() / p).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (Path.cwd() / p).resolve()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if np is not None and not np.isfinite(out):  # type: ignore[truthy-bool]
        return float("nan")
    return out


def _safe_int(value: Any, *, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        token = str(item).strip()
        if token:
            out.append(token)
    return out


def _json_blob(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick_first_text(*values: Any) -> str:
    for value in values:
        text = _safe_str(value).strip()
        if text:
            return text
    return ""


def _pick_first_finite(*values: Any) -> float:
    assert np is not None
    for value in values:
        out = _safe_float(value)
        if np.isfinite(out):
            return float(out)
    return float("nan")


def _atlas_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("entries", "atlas_rows", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError("Atlas payload must be a list or an object with entries/atlas_rows/rows")


def _extract_mode220_ids(payload: dict[str, Any]) -> list[str]:
    ids = _string_list(payload.get("accepted_geometry_ids"))
    if ids:
        return ids
    ids = _string_list(payload.get("geometry_ids"))
    if ids:
        return ids
    rows = payload.get("accepted_geometries")
    if isinstance(rows, list):
        return [
            _safe_str(row.get("geometry_id"))
            for row in rows
            if isinstance(row, dict) and _safe_str(row.get("geometry_id"))
        ]
    return []


def _extract_mode221_ids(payload: dict[str, Any]) -> list[str]:
    return _string_list(payload.get("geometry_ids"))


def _extract_common_ids(payload: dict[str, Any]) -> list[str]:
    ids = _string_list(payload.get("common_geometry_ids"))
    if ids:
        return ids
    nested = payload.get("common_intersection")
    if isinstance(nested, dict):
        return _string_list(nested.get("geometry_ids"))
    return []


def _extract_golden_ids(payload: dict[str, Any]) -> list[str]:
    ids = _string_list(payload.get("golden_geometry_ids"))
    if ids:
        return ids
    nested = payload.get("hawking_filtered_region")
    if isinstance(nested, dict):
        return _string_list(nested.get("golden_geometry_ids"))
    return []


def _extract_final_ids(payload: dict[str, Any]) -> list[str]:
    return _string_list(payload.get("final_geometry_ids"))


def _canonical_event_id(
    aggregate_row: dict[str, Any],
    provenance: dict[str, Any],
    s4f: dict[str, Any],
    s4k: dict[str, Any],
    ctx: Any,
    source_run_id: str,
) -> str:
    candidates = {
        _safe_str(aggregate_row.get("event_id")).strip(),
        _safe_str(((provenance.get("invocation") or {}) if isinstance(provenance.get("invocation"), dict) else {}).get("event_id")).strip(),
        _safe_str(s4f.get("event_id")).strip(),
        _safe_str(s4k.get("event_id")).strip(),
    }
    candidates.discard("")
    if not candidates:
        abort(ctx, f"No canonical event_id found for source run {source_run_id}")
        raise AssertionError("unreachable")
    if len(candidates) > 1:
        abort(
            ctx,
            f"Conflicting event_id values for source run {source_run_id}: {sorted(candidates)}",
        )
        raise AssertionError("unreachable")
    return next(iter(candidates))


def _source_run_stage_paths(out_root: Path, source_run_id: str) -> dict[str, Path]:
    base = out_root / source_run_id
    return {
        f"{source_run_id}:run_valid": base / "RUN_VALID" / "verdict.json",
        f"{source_run_id}:run_provenance": base / "run_provenance.json",
        f"{source_run_id}:s4g": base / "s4g_mode220_geometry_filter" / "outputs" / "mode220_filter.json",
        f"{source_run_id}:s4h": base / "s4h_mode221_geometry_filter" / "outputs" / "mode221_filter.json",
        f"{source_run_id}:s4i": base / "s4i_common_geometry_intersection" / "outputs" / "common_intersection.json",
        f"{source_run_id}:s4f": base / "s4f_area_observation" / "outputs" / "area_obs.json",
        f"{source_run_id}:s4j": base / "s4j_hawking_area_filter" / "outputs" / "hawking_area_filter.json",
        f"{source_run_id}:s4k": base / "s4k_event_support_region" / "outputs" / "event_support_region.json",
    }


def _resolve_atlas_path(
    atlas_arg: str | None,
    source_run_ids: list[str],
    out_root: Path,
    ctx: Any,
) -> tuple[Path, str]:
    known_hashes: set[str] = set()
    provenance_candidates: list[Path] = []

    for source_run_id in source_run_ids:
        provenance_path = out_root / source_run_id / "run_provenance.json"
        provenance = _read_json_object(provenance_path, f"{source_run_id}:run_provenance", ctx)
        invocation = provenance.get("invocation")
        if not isinstance(invocation, dict):
            continue
        atlas_path_raw = invocation.get("atlas_path")
        atlas_sha = invocation.get("atlas_sha256")
        if isinstance(atlas_sha, str) and atlas_sha.strip():
            known_hashes.add(atlas_sha.strip())
        if isinstance(atlas_path_raw, str) and atlas_path_raw.strip():
            provenance_candidates.append(_resolve_provenance_path(atlas_path_raw.strip()))

    if len(known_hashes) > 1:
        abort(ctx, f"Source runs disagree on atlas_sha256: {sorted(known_hashes)}")
        raise AssertionError("unreachable")

    if atlas_arg:
        atlas_path = _resolve_path(atlas_arg)
    else:
        if not provenance_candidates:
            abort(
                ctx,
                "No atlas_path provided and no atlas_path found in source run provenance",
            )
            raise AssertionError("unreachable")
        atlas_path = provenance_candidates[0]

    if not atlas_path.exists():
        abort(
            ctx,
            f"Missing atlas input: expected_path={atlas_path}; "
            "regen_cmd='python -m mvp.pipeline multimode --event-id <EVENT_ID> --run-id <RUN_ID> --atlas-default --offline --estimator dual'",
        )
        raise AssertionError("unreachable")

    atlas_sha = sha256_file(atlas_path)
    if known_hashes and atlas_sha not in known_hashes:
        abort(
            ctx,
            f"Atlas hash mismatch: selected={atlas_sha} provenance={sorted(known_hashes)} path={atlas_path}",
        )
        raise AssertionError("unreachable")

    return atlas_path, atlas_sha


def _write_utf8_dataset(group: Any, name: str, values: list[str], *, compression: str | None = None) -> None:
    assert h5py is not None and np is not None
    dt = h5py.string_dtype(encoding="utf-8")
    ds = group.create_dataset(name, (len(values),), dtype=dt, compression=compression)
    ds[...] = np.asarray(values, dtype=object)


def _write_bool_matrix(group: Any, name: str, matrix: Any) -> None:
    group.create_dataset(name, data=matrix, compression="gzip", shuffle=True)


def _write_float_vector(group: Any, name: str, values: list[float]) -> None:
    assert np is not None
    group.create_dataset(name, data=np.asarray(values, dtype=np.float64), compression="gzip", shuffle=True)


def _write_int_vector(group: Any, name: str, values: list[int]) -> None:
    assert np is not None
    group.create_dataset(name, data=np.asarray(values, dtype=np.int32), compression="gzip", shuffle=True)


def _write_float_matrix(group: Any, name: str, matrix: Any) -> None:
    group.create_dataset(name, data=matrix, compression="gzip", shuffle=True)


def main(argv: list[str] | None = None) -> int:
    if h5py is None or np is None:  # pragma: no cover - depends on runtime env
        print(
            "ERROR: [experiment_phase1_geometry_h5] h5py and numpy are required",
            file=sys.stderr,
        )
        return 2

    args = _parse_args(argv)
    ctx = init_stage(args.run_id, STAGE, params={"atlas_path": args.atlas_path})

    aggregate_path = ctx.run_dir / "s5_aggregate" / "outputs" / "aggregate.json"
    aggregate = _read_json_object(aggregate_path, "aggregate", ctx)
    event_rows = aggregate.get("events")
    if not isinstance(event_rows, list) or not event_rows:
        abort(ctx, f"Aggregate payload has no events at {aggregate_path}")

    source_run_ids: list[str] = []
    aggregate_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(event_rows):
        if not isinstance(row, dict):
            abort(ctx, f"Aggregate event row {idx} is not an object")
        source_run_id = _safe_str(row.get("run_id")).strip()
        if not source_run_id:
            abort(ctx, f"Aggregate event row {idx} missing run_id")
        source_run_ids.append(source_run_id)
        aggregate_rows.append(row)

    atlas_path, atlas_sha = _resolve_atlas_path(args.atlas_path, source_run_ids, ctx.out_root, ctx)

    runtime_inputs: dict[str, Path] = {"aggregate": aggregate_path, "atlas": atlas_path}
    for source_run_id in source_run_ids:
        runtime_inputs.update(_source_run_stage_paths(ctx.out_root, source_run_id))
    check_inputs(ctx, runtime_inputs)

    for source_run_id in source_run_ids:
        try:
            require_run_valid(ctx.out_root, source_run_id)
        except Exception as exc:
            abort(
                ctx,
                f"Source run not PASS: expected_path={ctx.out_root / source_run_id / 'RUN_VALID' / 'verdict.json'}; "
                f"regen_cmd='python -m mvp.pipeline multimode --event-id <EVENT_ID> --run-id {source_run_id} --atlas-default --offline --estimator dual'; "
                f"detail={exc}",
            )

    atlas_payload = json.loads(atlas_path.read_text(encoding="utf-8"))
    try:
        atlas_rows = _atlas_entries(atlas_payload)
    except Exception as exc:
        abort(ctx, f"Invalid atlas payload at {atlas_path}: {exc}")
        raise AssertionError("unreachable")
    if not atlas_rows:
        abort(ctx, f"Atlas has no entries at {atlas_path}")

    atlas_lookup: dict[str, dict[str, Any]] = {}
    atlas_order: list[str] = []
    for idx, entry in enumerate(atlas_rows):
        geometry_id = _safe_str(entry.get("geometry_id")).strip()
        if not geometry_id:
            abort(ctx, f"Atlas entry {idx} missing geometry_id")
        if geometry_id in atlas_lookup:
            abort(ctx, f"Duplicate geometry_id in atlas: {geometry_id}")
        atlas_lookup[geometry_id] = entry
        atlas_order.append(geometry_id)

    joint = aggregate.get("joint_posterior")
    joint_rows = joint.get("joint_ranked_all") if isinstance(joint, dict) else None
    if joint_rows is None:
        joint_rows = []
    if not isinstance(joint_rows, list):
        abort(ctx, "aggregate.joint_posterior.joint_ranked_all must be a list")

    payload_entry_by_gid: dict[str, dict[str, Any]] = {}
    joint_row_by_gid: dict[str, dict[str, Any]] = {}
    observed_geometry_ids: set[str] = set()

    event_records: list[dict[str, Any]] = []
    event_id_values: list[str] = []
    event_run_values: list[str] = []
    event_analysis_path: list[str] = []
    event_support_region_status: list[str] = []
    event_domain_status: list[str] = []
    event_domain_status_source: list[str] = []
    event_downstream_status_class: list[str] = []
    event_multimode_viability_class: list[str] = []
    event_area_observation_status: list[str] = []
    event_area_kind: list[str] = []
    event_metric: list[str] = []
    event_n_mode220: list[int] = []
    event_n_mode221: list[int] = []
    event_n_common_input: list[int] = []
    event_n_area_entries: list[int] = []
    event_n_missing_area_data: list[int] = []
    event_n_golden: list[int] = []
    event_n_final_geometries: list[int] = []
    event_n_atlas: list[int] = []
    event_threshold_d2: list[float] = []
    event_initial_total_area_lower_bound: list[float] = []
    event_area_constraint_applied: list[bool] = []
    event_area_obs_present: list[bool] = []
    event_mode221_skipped: list[bool] = []

    raw_run_provenance: list[str] = []
    raw_s4g: list[str] = []
    raw_s4h: list[str] = []
    raw_s4i: list[str] = []
    raw_s4f: list[str] = []
    raw_s4j: list[str] = []
    raw_s4k: list[str] = []
    raw_aggregate_event_row: list[str] = []

    for source_run_id, aggregate_row in zip(source_run_ids, aggregate_rows):
        base = ctx.out_root / source_run_id
        provenance = _read_json_object(base / "run_provenance.json", f"{source_run_id}:run_provenance", ctx)
        s4g = _read_json_object(base / "s4g_mode220_geometry_filter" / "outputs" / "mode220_filter.json", f"{source_run_id}:s4g", ctx)
        s4h = _read_json_object(base / "s4h_mode221_geometry_filter" / "outputs" / "mode221_filter.json", f"{source_run_id}:s4h", ctx)
        s4i = _read_json_object(base / "s4i_common_geometry_intersection" / "outputs" / "common_intersection.json", f"{source_run_id}:s4i", ctx)
        s4f = _read_json_object(base / "s4f_area_observation" / "outputs" / "area_obs.json", f"{source_run_id}:s4f", ctx)
        s4j = _read_json_object(base / "s4j_hawking_area_filter" / "outputs" / "hawking_area_filter.json", f"{source_run_id}:s4j", ctx)
        s4k = _read_json_object(base / "s4k_event_support_region" / "outputs" / "event_support_region.json", f"{source_run_id}:s4k", ctx)

        event_id = _canonical_event_id(aggregate_row, provenance, s4f, s4k, ctx, source_run_id)
        mode220_ids = _extract_mode220_ids(s4g)
        mode221_ids = _extract_mode221_ids(s4h)
        common_ids = _extract_common_ids(s4i)
        golden_ids = _extract_golden_ids(s4j)
        final_ids = _extract_final_ids(s4k)
        observed_geometry_ids.update(mode220_ids)
        observed_geometry_ids.update(mode221_ids)
        observed_geometry_ids.update(common_ids)
        observed_geometry_ids.update(golden_ids)
        observed_geometry_ids.update(final_ids)

        accepted_geometries = s4g.get("accepted_geometries")
        if isinstance(accepted_geometries, list):
            for row in accepted_geometries:
                if not isinstance(row, dict):
                    continue
                geometry_id = _safe_str(row.get("geometry_id")).strip()
                if geometry_id and geometry_id not in payload_entry_by_gid:
                    payload_entry_by_gid[geometry_id] = row

        downstream_status = s4k.get("downstream_status")
        downstream_status_class = (
            _safe_str(downstream_status.get("class"))
            if isinstance(downstream_status, dict)
            else ""
        )
        multimode_viability = s4k.get("multimode_viability")
        multimode_viability_class = (
            _safe_str(multimode_viability.get("class"))
            if isinstance(multimode_viability, dict)
            else ""
        )

        event_id_values.append(event_id)
        event_run_values.append(source_run_id)
        event_analysis_path.append(_safe_str(s4k.get("analysis_path")))
        event_support_region_status.append(_safe_str(s4k.get("support_region_status")))
        event_domain_status.append(_safe_str(s4k.get("domain_status")))
        event_domain_status_source.append(_safe_str(s4k.get("domain_status_source")))
        event_downstream_status_class.append(downstream_status_class)
        event_multimode_viability_class.append(multimode_viability_class)
        event_area_observation_status.append(_safe_str(s4f.get("observation_status")))
        event_area_kind.append(_safe_str(s4f.get("area_kind")))
        event_metric.append(_safe_str(aggregate_row.get("metric")))
        event_n_mode220.append(_safe_int(s4g.get("n_geometries_accepted"), default=len(mode220_ids)))
        event_n_mode221.append(_safe_int(s4h.get("n_passed"), default=len(mode221_ids)))
        event_n_common_input.append(_safe_int(s4j.get("n_common_input"), default=len(common_ids)))
        event_n_area_entries.append(_safe_int(s4f.get("n_area_entries"), default=-1))
        event_n_missing_area_data.append(_safe_int(s4j.get("n_missing_area_data"), default=-1))
        event_n_golden.append(_safe_int(s4j.get("n_golden"), default=len(golden_ids)))
        event_n_final_geometries.append(_safe_int(s4k.get("n_final_geometries"), default=len(final_ids)))
        event_n_atlas.append(_safe_int(aggregate_row.get("n_atlas"), default=-1))
        event_threshold_d2.append(_safe_float(aggregate_row.get("threshold_d2")))
        event_initial_total_area_lower_bound.append(_safe_float(s4f.get("initial_total_area_lower_bound")))
        event_area_constraint_applied.append(_safe_bool(s4j.get("area_constraint_applied")))
        event_area_obs_present.append(_safe_bool(s4j.get("area_obs_present")))
        event_mode221_skipped.append(_safe_bool(s4i.get("mode221_skipped")))

        raw_run_provenance.append(_json_blob(provenance))
        raw_s4g.append(_json_blob(s4g))
        raw_s4h.append(_json_blob(s4h))
        raw_s4i.append(_json_blob(s4i))
        raw_s4f.append(_json_blob(s4f))
        raw_s4j.append(_json_blob(s4j))
        raw_s4k.append(_json_blob(s4k))
        raw_aggregate_event_row.append(_json_blob(aggregate_row))
        event_records.append(
            {
                "run_id": source_run_id,
                "event_id": event_id,
                "mode220_ids": mode220_ids,
                "mode221_ids": mode221_ids,
                "common_ids": common_ids,
                "golden_ids": golden_ids,
                "final_ids": final_ids,
            }
        )

    for row_idx, row in enumerate(joint_rows):
        if not isinstance(row, dict):
            abort(ctx, f"aggregate.joint_posterior.joint_ranked_all[{row_idx}] is not an object")
        geometry_id = _safe_str(row.get("geometry_id")).strip()
        if not geometry_id:
            abort(ctx, f"aggregate.joint_posterior.joint_ranked_all[{row_idx}] missing geometry_id")
        observed_geometry_ids.add(geometry_id)
        joint_row_by_gid.setdefault(geometry_id, row)

    if not observed_geometry_ids:
        abort(ctx, "No observed geometries found across source runs or aggregate joint posterior")

    atlas_id_set = set(atlas_order)
    geometry_ids = [geometry_id for geometry_id in atlas_order if geometry_id in observed_geometry_ids]
    geometry_ids.extend(sorted(observed_geometry_ids - atlas_id_set))
    geometry_to_index = {geometry_id: idx for idx, geometry_id in enumerate(geometry_ids)}

    atlas_geometry_id: list[str] = []
    atlas_theory: list[str] = []
    atlas_family: list[str] = []
    atlas_mode: list[str] = []
    atlas_source: list[str] = []
    atlas_ref: list[str] = []
    atlas_entry_json: list[str] = []
    atlas_M_solar: list[float] = []
    atlas_chi: list[float] = []
    atlas_f_hz: list[float] = []
    atlas_tau_s: list[float] = []
    atlas_Q: list[float] = []
    atlas_a_over_m: list[float] = []
    atlas_J_over_M2: list[float] = []
    atlas_phi_atlas: list[float] = []
    atlas_zeta: list[float] = []
    atlas_delta_f_frac: list[float] = []
    atlas_delta_tau_frac: list[float] = []
    atlas_modes_available_json: list[str] = []
    atlas_source_geometry_ids_json: list[str] = []
    atlas_physical_parameters_json: list[str] = []
    atlas_present_in_source_atlas: list[bool] = []

    for geometry_id in geometry_ids:
        atlas_entry = atlas_lookup.get(geometry_id)
        payload_entry = payload_entry_by_gid.get(geometry_id)
        joint_row = joint_row_by_gid.get(geometry_id)

        atlas_meta = _dict_or_empty(_dict_or_empty(atlas_entry).get("metadata"))
        payload_meta = _dict_or_empty(_dict_or_empty(payload_entry).get("metadata"))
        joint_meta = _dict_or_empty(_dict_or_empty(joint_row).get("metadata"))

        atlas_geometry_id.append(geometry_id)
        atlas_theory.append(
            _pick_first_text(
                _dict_or_empty(atlas_entry).get("theory"),
                _dict_or_empty(payload_entry).get("theory"),
            )
        )
        atlas_family.append(
            _pick_first_text(
                atlas_meta.get("family"),
                payload_meta.get("family"),
                joint_meta.get("family"),
                _dict_or_empty(atlas_entry).get("family"),
                _dict_or_empty(payload_entry).get("family"),
            )
        )
        atlas_mode.append(
            _pick_first_text(
                atlas_meta.get("mode"),
                payload_meta.get("mode"),
                joint_meta.get("mode"),
                _dict_or_empty(atlas_entry).get("mode"),
                _dict_or_empty(payload_entry).get("mode"),
            )
        )
        atlas_source.append(
            _pick_first_text(
                atlas_meta.get("source"),
                payload_meta.get("source"),
                joint_meta.get("source"),
            )
        )
        atlas_ref.append(_pick_first_text(atlas_meta.get("ref"), payload_meta.get("ref"), joint_meta.get("ref")))
        atlas_entry_json.append(
            _json_blob(atlas_entry or payload_entry or joint_row or {"geometry_id": geometry_id})
        )
        atlas_M_solar.append(
            _pick_first_finite(
                atlas_meta.get("M_solar"),
                _dict_or_empty(atlas_entry).get("M_solar"),
                payload_meta.get("M_solar"),
                _dict_or_empty(payload_entry).get("M_solar"),
                joint_meta.get("M_solar"),
            )
        )
        atlas_chi.append(
            _pick_first_finite(
                atlas_meta.get("chi"),
                _dict_or_empty(atlas_entry).get("chi"),
                payload_meta.get("chi"),
                _dict_or_empty(payload_entry).get("chi"),
                joint_meta.get("chi"),
            )
        )
        atlas_f_hz.append(
            _pick_first_finite(
                _dict_or_empty(atlas_entry).get("f_hz"),
                _dict_or_empty(payload_entry).get("f_hz"),
            )
        )
        atlas_tau_s.append(
            _pick_first_finite(
                _dict_or_empty(atlas_entry).get("tau_s"),
                _dict_or_empty(payload_entry).get("tau_s"),
            )
        )
        atlas_Q.append(
            _pick_first_finite(
                _dict_or_empty(atlas_entry).get("Q"),
                _dict_or_empty(payload_entry).get("Q"),
            )
        )
        atlas_a_over_m.append(_pick_first_finite(_dict_or_empty(atlas_entry).get("a_over_m")))
        atlas_J_over_M2.append(_pick_first_finite(_dict_or_empty(atlas_entry).get("J_over_M2")))
        atlas_phi_atlas.append(
            _pick_first_finite(
                _dict_or_empty(atlas_entry).get("phi_atlas"),
                _dict_or_empty(payload_entry).get("phi_atlas"),
            )
        )
        atlas_zeta.append(_pick_first_finite(atlas_meta.get("zeta"), payload_meta.get("zeta")))
        atlas_delta_f_frac.append(
            _pick_first_finite(atlas_meta.get("delta_f_frac"), payload_meta.get("delta_f_frac"))
        )
        atlas_delta_tau_frac.append(
            _pick_first_finite(atlas_meta.get("delta_tau_frac"), payload_meta.get("delta_tau_frac"))
        )
        atlas_modes_available_json.append(
            _json_blob(
                atlas_meta.get("modes_available")
                if "modes_available" in atlas_meta
                else payload_meta.get("modes_available")
            )
        )
        atlas_source_geometry_ids_json.append(
            _json_blob(
                atlas_meta.get("source_geometry_ids")
                if "source_geometry_ids" in atlas_meta
                else payload_meta.get("source_geometry_ids")
            )
        )
        atlas_physical_parameters_json.append(
            _json_blob(
                atlas_meta.get("physical_parameters")
                if "physical_parameters" in atlas_meta
                else payload_meta.get("physical_parameters")
            )
        )
        atlas_present_in_source_atlas.append(geometry_id in atlas_lookup)

    n_events = len(source_run_ids)
    n_atlas = len(geometry_ids)
    mode220_matrix = np.zeros((n_events, n_atlas), dtype=np.bool_)
    mode221_matrix = np.zeros((n_events, n_atlas), dtype=np.bool_)
    common_matrix = np.zeros((n_events, n_atlas), dtype=np.bool_)
    hawking_matrix = np.zeros((n_events, n_atlas), dtype=np.bool_)
    final_matrix = np.zeros((n_events, n_atlas), dtype=np.bool_)

    for event_idx, record in enumerate(event_records):
        for geometry_id in record["mode220_ids"]:
            mode220_matrix[event_idx, geometry_to_index[geometry_id]] = True
        for geometry_id in record["mode221_ids"]:
            mode221_matrix[event_idx, geometry_to_index[geometry_id]] = True
        for geometry_id in record["common_ids"]:
            common_matrix[event_idx, geometry_to_index[geometry_id]] = True
        for geometry_id in record["golden_ids"]:
            hawking_matrix[event_idx, geometry_to_index[geometry_id]] = True
        for geometry_id in record["final_ids"]:
            final_matrix[event_idx, geometry_to_index[geometry_id]] = True

    joint_available = np.zeros((n_atlas,), dtype=np.bool_)
    joint_coverage = np.full((n_atlas,), np.nan, dtype=np.float64)
    joint_support_count = np.full((n_atlas,), -1, dtype=np.int32)
    joint_support_fraction = np.full((n_atlas,), np.nan, dtype=np.float64)
    joint_posterior_weight = np.full((n_atlas,), np.nan, dtype=np.float64)
    joint_delta_lnL = np.full((n_atlas,), np.nan, dtype=np.float64)
    joint_d2_sum = np.full((n_atlas,), np.nan, dtype=np.float64)
    joint_log_likelihood_rel = np.full((n_atlas,), np.nan, dtype=np.float64)
    joint_d2_per_event = np.full((n_events, n_atlas), np.nan, dtype=np.float64)
    joint_row_json = [""] * n_atlas

    event_index = {event_id: idx for idx, event_id in enumerate(event_id_values)}
    for geometry_id, row in joint_row_by_gid.items():
        geom_idx = geometry_to_index[geometry_id]
        joint_available[geom_idx] = True
        joint_coverage[geom_idx] = _safe_float(row.get("coverage"))
        joint_support_count[geom_idx] = _safe_int(row.get("support_count"), default=-1)
        joint_support_fraction[geom_idx] = _safe_float(row.get("support_fraction"))
        joint_posterior_weight[geom_idx] = _safe_float(row.get("posterior_weight_joint"))
        joint_delta_lnL[geom_idx] = _safe_float(row.get("delta_lnL_joint"))
        joint_d2_sum[geom_idx] = _safe_float(row.get("d2_sum"))
        joint_log_likelihood_rel[geom_idx] = _safe_float(row.get("log_likelihood_rel_joint"))
        joint_row_json[geom_idx] = _json_blob(row)

        d2_per_event = row.get("d2_per_event")
        if isinstance(d2_per_event, list):
            if len(d2_per_event) != n_events:
                abort(
                    ctx,
                    f"d2_per_event length mismatch for {geometry_id}: got {len(d2_per_event)} expected {n_events}",
                )
            for event_idx, value in enumerate(d2_per_event):
                joint_d2_per_event[event_idx, geom_idx] = _safe_float(value)
        elif isinstance(d2_per_event, dict):
            for event_id, value in d2_per_event.items():
                event_idx = event_index.get(str(event_id))
                if event_idx is None:
                    continue
                joint_d2_per_event[event_idx, geom_idx] = _safe_float(value)

    h5_path = ctx.outputs_dir / args.output_name
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["created_utc"] = utc_now_iso()
        h5.attrs["aggregate_run_id"] = args.run_id
        h5.attrs["aggregate_path"] = str(aggregate_path)
        h5.attrs["atlas_path"] = str(atlas_path)
        h5.attrs["atlas_sha256"] = atlas_sha
        h5.attrs["n_events"] = n_events
        h5.attrs["n_atlas"] = n_atlas

        g_events = h5.create_group("events")
        _write_utf8_dataset(g_events, "event_id", event_id_values)
        _write_utf8_dataset(g_events, "run_id", event_run_values)
        _write_utf8_dataset(g_events, "analysis_path", event_analysis_path)
        _write_utf8_dataset(g_events, "support_region_status", event_support_region_status)
        _write_utf8_dataset(g_events, "domain_status", event_domain_status)
        _write_utf8_dataset(g_events, "domain_status_source", event_domain_status_source)
        _write_utf8_dataset(g_events, "downstream_status_class", event_downstream_status_class)
        _write_utf8_dataset(g_events, "multimode_viability_class", event_multimode_viability_class)
        _write_utf8_dataset(g_events, "area_observation_status", event_area_observation_status)
        _write_utf8_dataset(g_events, "area_kind", event_area_kind)
        _write_utf8_dataset(g_events, "metric", event_metric)
        _write_int_vector(g_events, "n_mode220", event_n_mode220)
        _write_int_vector(g_events, "n_mode221", event_n_mode221)
        _write_int_vector(g_events, "n_common_input", event_n_common_input)
        _write_int_vector(g_events, "n_area_entries", event_n_area_entries)
        _write_int_vector(g_events, "n_missing_area_data", event_n_missing_area_data)
        _write_int_vector(g_events, "n_golden", event_n_golden)
        _write_int_vector(g_events, "n_final_geometries", event_n_final_geometries)
        _write_int_vector(g_events, "n_atlas", event_n_atlas)
        _write_float_vector(g_events, "threshold_d2", event_threshold_d2)
        _write_float_vector(g_events, "initial_total_area_lower_bound", event_initial_total_area_lower_bound)
        g_events.create_dataset("area_constraint_applied", data=np.asarray(event_area_constraint_applied, dtype=np.bool_))
        g_events.create_dataset("area_obs_present", data=np.asarray(event_area_obs_present, dtype=np.bool_))
        g_events.create_dataset("mode221_skipped", data=np.asarray(event_mode221_skipped, dtype=np.bool_))

        g_atlas = h5.create_group("atlas")
        _write_utf8_dataset(g_atlas, "geometry_id", atlas_geometry_id)
        _write_utf8_dataset(g_atlas, "theory", atlas_theory)
        _write_utf8_dataset(g_atlas, "family", atlas_family)
        _write_utf8_dataset(g_atlas, "mode", atlas_mode)
        _write_utf8_dataset(g_atlas, "source", atlas_source)
        _write_utf8_dataset(g_atlas, "ref", atlas_ref)
        _write_utf8_dataset(g_atlas, "entry_json", atlas_entry_json, compression="gzip")
        _write_utf8_dataset(g_atlas, "modes_available_json", atlas_modes_available_json, compression="gzip")
        _write_utf8_dataset(g_atlas, "source_geometry_ids_json", atlas_source_geometry_ids_json, compression="gzip")
        _write_utf8_dataset(g_atlas, "physical_parameters_json", atlas_physical_parameters_json, compression="gzip")
        g_atlas.create_dataset("present_in_source_atlas", data=np.asarray(atlas_present_in_source_atlas, dtype=np.bool_))
        _write_float_vector(g_atlas, "M_solar", atlas_M_solar)
        _write_float_vector(g_atlas, "chi", atlas_chi)
        _write_float_vector(g_atlas, "f_hz", atlas_f_hz)
        _write_float_vector(g_atlas, "tau_s", atlas_tau_s)
        _write_float_vector(g_atlas, "Q", atlas_Q)
        _write_float_vector(g_atlas, "a_over_m", atlas_a_over_m)
        _write_float_vector(g_atlas, "J_over_M2", atlas_J_over_M2)
        _write_float_vector(g_atlas, "phi_atlas", atlas_phi_atlas)
        _write_float_vector(g_atlas, "zeta", atlas_zeta)
        _write_float_vector(g_atlas, "delta_f_frac", atlas_delta_f_frac)
        _write_float_vector(g_atlas, "delta_tau_frac", atlas_delta_tau_frac)

        g_membership = h5.create_group("membership")
        _write_bool_matrix(g_membership, "mode220", mode220_matrix)
        _write_bool_matrix(g_membership, "mode221", mode221_matrix)
        _write_bool_matrix(g_membership, "common_pre_hawking", common_matrix)
        _write_bool_matrix(g_membership, "golden_post_hawking", hawking_matrix)
        _write_bool_matrix(g_membership, "final_support_region", final_matrix)

        g_joint = h5.create_group("joint_posterior")
        g_joint.attrs["n_rows_from_aggregate"] = len(joint_rows)
        g_joint.create_dataset("available", data=joint_available)
        g_joint.create_dataset("support_count", data=joint_support_count, compression="gzip", shuffle=True)
        g_joint.create_dataset("coverage", data=joint_coverage, compression="gzip", shuffle=True)
        g_joint.create_dataset("support_fraction", data=joint_support_fraction, compression="gzip", shuffle=True)
        g_joint.create_dataset("posterior_weight_joint", data=joint_posterior_weight, compression="gzip", shuffle=True)
        g_joint.create_dataset("delta_lnL_joint", data=joint_delta_lnL, compression="gzip", shuffle=True)
        g_joint.create_dataset("d2_sum", data=joint_d2_sum, compression="gzip", shuffle=True)
        g_joint.create_dataset("log_likelihood_rel_joint", data=joint_log_likelihood_rel, compression="gzip", shuffle=True)
        _write_float_matrix(g_joint, "d2_per_event", joint_d2_per_event)
        _write_utf8_dataset(g_joint, "row_json", joint_row_json, compression="gzip")

        g_raw = h5.create_group("raw_json")
        g_raw.create_dataset("aggregate", data=_json_blob(aggregate), dtype=h5py.string_dtype(encoding="utf-8"))
        _write_utf8_dataset(g_raw, "aggregate_event_row", raw_aggregate_event_row, compression="gzip")
        _write_utf8_dataset(g_raw, "run_provenance", raw_run_provenance, compression="gzip")
        _write_utf8_dataset(g_raw, "s4g_mode220_geometry_filter", raw_s4g, compression="gzip")
        _write_utf8_dataset(g_raw, "s4h_mode221_geometry_filter", raw_s4h, compression="gzip")
        _write_utf8_dataset(g_raw, "s4i_common_geometry_intersection", raw_s4i, compression="gzip")
        _write_utf8_dataset(g_raw, "s4f_area_observation", raw_s4f, compression="gzip")
        _write_utf8_dataset(g_raw, "s4j_hawking_area_filter", raw_s4j, compression="gzip")
        _write_utf8_dataset(g_raw, "s4k_event_support_region", raw_s4k, compression="gzip")

    analysis_path_counts = dict(Counter(event_analysis_path))
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now_iso(),
        "aggregate_run_id": args.run_id,
        "aggregate_path": str(aggregate_path),
        "atlas_path": str(atlas_path),
        "atlas_sha256": atlas_sha,
        "n_events": n_events,
        "n_atlas": n_atlas,
        "n_joint_rows_from_aggregate": len(joint_rows),
        "n_geometry_entries_missing_from_source_atlas": int(sum(not v for v in atlas_present_in_source_atlas)),
        "analysis_path_counts": analysis_path_counts,
        "membership_true_counts": {
            "mode220": int(mode220_matrix.sum()),
            "mode221": int(mode221_matrix.sum()),
            "common_pre_hawking": int(common_matrix.sum()),
            "golden_post_hawking": int(hawking_matrix.sum()),
            "final_support_region": int(final_matrix.sum()),
        },
        "event_metrics_totals": {
            "n_mode220": int(sum(event_n_mode220)),
            "n_mode221": int(sum(event_n_mode221)),
            "n_common_input": int(sum(event_n_common_input)),
            "n_area_entries": int(sum(max(v, 0) for v in event_n_area_entries)),
            "n_missing_area_data": int(sum(max(v, 0) for v in event_n_missing_area_data)),
            "n_golden": int(sum(event_n_golden)),
            "n_final_geometries": int(sum(event_n_final_geometries)),
        },
        "outputs": {
            "phase1_geometry_cohort_h5": str(h5_path),
        },
        "notes": [
            "The HDF5 archive is self-contained: it includes raw JSON payloads per event plus normalized membership matrices.",
            "The exported geometry universe is the cohort-observed union, enriched from the source atlas when available.",
            "No canonical hyperbolic/elliptic/unknown label exists in the current atlas; phase 2 should derive that classification from the archived geometry metadata.",
        ],
    }
    summary_path = write_json_atomic(ctx.outputs_dir / args.summary_name, summary_payload)

    finalize(
        ctx,
        artifacts={
            "phase1_geometry_cohort_h5": h5_path,
            "phase1_geometry_summary": summary_path,
        },
        results={
            "n_events": n_events,
            "n_atlas": n_atlas,
            "n_final_geometries_total": int(sum(event_n_final_geometries)),
        },
        extra_summary={
            "analysis_path_counts": analysis_path_counts,
        },
    )
    log_stage_paths(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
