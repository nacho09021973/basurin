#!/usr/bin/env python3
"""Build analysis-ready event/candidate tables from BASURIN source runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from basurin_io import sha256_file, write_json_atomic
from mvp.contracts import check_inputs, finalize, init_stage, log_stage_paths

STAGE = "experiment/feature_foundry"
EVENT_SUFFIX_RE = re.compile(r"_real(?:_offline(?:_rescue)?)?$")
DEFAULT_ATLAS_REL = Path("docs/ringdown/atlas/atlas_berti_v2.json")
DEFAULT_CATALOG_RELS = [
    Path("gw_events/gwtc_quality_events.csv"),
    Path("gwtc_quality_events.csv"),
]
EXTREMAL_CHI = 0.999999


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    seen: set[str] = set()
    values: list[str] = []
    for item in raw.split(","):
        token = item.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        values.append(token)
    return values


def _repo_root() -> Path:
    return _here.parents[1]


def _resolve_required_path(raw: str | None, default_rel: Path) -> Path:
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    return (_repo_root() / default_rel).resolve()


def _resolve_optional_catalog(raw: str | None) -> Path | None:
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    for rel in DEFAULT_CATALOG_RELS:
        candidate = (_repo_root() / rel).resolve()
        if candidate.exists():
            return candidate
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json_object(path)


def _canonical_event_id(event_id: str) -> str:
    return EVENT_SUFFIX_RE.sub("", (event_id or "").strip())


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _black_hole_area(m_solar: float | None, chi: float | None) -> float | None:
    if m_solar is None or chi is None:
        return None
    achi = abs(float(chi))
    if achi >= 1.0:
        achi = min(achi, 0.999999)
    return 8.0 * math.pi * float(m_solar) * float(m_solar) * (1.0 + math.sqrt(1.0 - achi * achi))


def _hawking_pass(final_area: float | None, initial_area: float | None) -> bool | None:
    if final_area is None or initial_area is None:
        return None
    return bool(final_area >= initial_area)


def _hawking_interval_status(
    final_area: float | None,
    initial_area_bound_min: float | None,
    initial_area_bound_max: float | None,
) -> str:
    if final_area is None or initial_area_bound_min is None or initial_area_bound_max is None:
        return "UNKNOWN"
    if final_area < initial_area_bound_min:
        return "DEFINITE_FAIL"
    if final_area >= initial_area_bound_max:
        return "ROBUST_PASS"
    return "BOUND_SENSITIVE"


def _is_kerr_family(value: Any) -> bool:
    return str(value or "").strip().lower() == "kerr"


def _failure_pattern(fail_count: int, unknown_count: int, support_count: int) -> str:
    if support_count <= 0:
        return "UNSEEN"
    if fail_count >= support_count:
        return "FAIL_ALL_SEEN"
    if fail_count > 0:
        return "FAIL_FEW_SEEN"
    if unknown_count > 0:
        return "UNKNOWN_PRESENT"
    return "PASS_ALL_SEEN"


def _format_mode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(str(item) for item in value) + ")"
    return str(value)


def _rel_to_out_root(path: Path, out_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(out_root.resolve()))
    except Exception:
        return str(path.resolve())


def _record_existing_input(ctx: Any, label: str, path: Path) -> None:
    if not path.exists():
        return
    ctx.inputs_record.append(
        {
            "label": label,
            "path": _rel_to_out_root(path, ctx.out_root),
            "sha256": sha256_file(path),
        }
    )


def _atlas_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("entries", "atlas_rows", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError("Atlas payload must be a list or an object with entries/atlas_rows/rows")


def _build_atlas_lookup(atlas_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(atlas_path.read_text(encoding="utf-8"))
    rows = _atlas_entries(payload)
    by_id: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(rows):
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        geometry_id = str(entry.get("geometry_id", idx))
        f_hz = _safe_float(entry.get("f_hz"))
        tau_s = _safe_float(entry.get("tau_s"))
        q_val = _safe_float(entry.get("Q"))
        if tau_s is None and q_val is not None and f_hz and f_hz > 0:
            tau_s = q_val / (math.pi * f_hz)
        m_solar = _safe_float(metadata.get("M_solar", entry.get("M_solar")))
        chi = _safe_float(metadata.get("chi", entry.get("chi")))
        by_id[geometry_id] = {
            "geometry_id": geometry_id,
            "atlas_index": idx,
            "M_solar": m_solar,
            "chi": chi,
            "mode": _format_mode(metadata.get("mode", entry.get("mode"))),
            "family": str(metadata.get("family", entry.get("family", ""))),
            "source": str(metadata.get("source", entry.get("source", ""))),
            "f_hz": f_hz,
            "tau_s": tau_s,
            "Q": q_val,
            "final_area_proxy": _black_hole_area(m_solar, chi),
        }
    return rows, by_id


def _load_catalog(catalog_path: Path | None) -> dict[str, dict[str, str]]:
    if catalog_path is None or not catalog_path.exists():
        return {}
    with catalog_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return {str(row.get("event", "")).strip(): row for row in reader if str(row.get("event", "")).strip()}


def _event_initial_areas(event_id: str, catalog: dict[str, dict[str, str]]) -> tuple[bool, dict[str, float | None]]:
    empty = {
        "bound_min": None,
        "bound_max": None,
        "zero_spin_proxy": None,
        "chieff_projection_proxy": None,
    }
    row = catalog.get(event_id)
    if row is None:
        return False, empty
    m1 = _safe_float(row.get("m1_source") or row.get("mass_1_source"))
    m2 = _safe_float(row.get("m2_source") or row.get("mass_2_source"))
    chi_eff = abs(_safe_float(row.get("chi_eff")) or 0.0)
    if m1 is None or m2 is None:
        return False, empty
    area_min_1 = _black_hole_area(m1, EXTREMAL_CHI)
    area_min_2 = _black_hole_area(m2, EXTREMAL_CHI)
    area_zero_1 = _black_hole_area(m1, 0.0)
    area_zero_2 = _black_hole_area(m2, 0.0)
    area_chi_1 = _black_hole_area(m1, chi_eff)
    area_chi_2 = _black_hole_area(m2, chi_eff)
    if None in (area_min_1, area_min_2, area_zero_1, area_zero_2, area_chi_1, area_chi_2):
        return False, empty
    return True, {
        "bound_min": area_min_1 + area_min_2,
        "bound_max": area_zero_1 + area_zero_2,
        "zero_spin_proxy": area_zero_1 + area_zero_2,
        "chieff_projection_proxy": area_chi_1 + area_chi_2,
    }


def _infer_event_id(run_id: str, s1: dict[str, Any] | None, s4: dict[str, Any] | None, router: dict[str, Any] | None) -> str:
    for payload, key in ((router, "event_id"), (s4, "event_id")):
        if isinstance(payload, dict):
            value = payload.get(key)
            if value:
                return str(value)
    if isinstance(s1, dict):
        params = s1.get("parameters")
        if isinstance(params, dict) and params.get("event_id"):
            return str(params["event_id"])
    return run_id


def _detectors_used(s1: dict[str, Any] | None) -> str:
    if not isinstance(s1, dict):
        return ""
    for key in ("results", "parameters"):
        section = s1.get(key)
        if isinstance(section, dict):
            detectors = section.get("detectors")
            if isinstance(detectors, list):
                return "+".join(str(det) for det in detectors)
    return ""


def _choose_candidate_basis(s4: dict[str, Any] | None) -> tuple[list[Any], str]:
    if not isinstance(s4, dict):
        return [], "missing_s4"
    compatible = s4.get("compatible_geometries")
    if isinstance(compatible, list) and compatible:
        return compatible, "compatible_geometries"
    ranked = s4.get("ranked_all")
    if isinstance(ranked, list) and ranked:
        return ranked, "ranked_all_fallback"
    return [], "empty"


def _candidate_to_geometry_id(candidate: Any, atlas_rows: list[dict[str, Any]]) -> str:
    if isinstance(candidate, dict):
        geometry_id = candidate.get("geometry_id")
        if geometry_id:
            return str(geometry_id)
        atlas_index = _safe_int(candidate.get("atlas_index"))
        if atlas_index is not None and 0 <= atlas_index < len(atlas_rows):
            return str(atlas_rows[atlas_index].get("geometry_id", atlas_index))
    elif isinstance(candidate, int) and 0 <= candidate < len(atlas_rows):
        return str(atlas_rows[candidate].get("geometry_id", candidate))
    return str(candidate)


def _candidate_info(candidate: Any, atlas_rows: list[dict[str, Any]], atlas_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    geometry_id = _candidate_to_geometry_id(candidate, atlas_rows)
    atlas_info = atlas_by_id.get(geometry_id, {})
    candidate_meta = candidate.get("metadata") if isinstance(candidate, dict) and isinstance(candidate.get("metadata"), dict) else {}
    m_solar = _safe_float(candidate_meta.get("M_solar", atlas_info.get("M_solar")))
    chi = _safe_float(candidate_meta.get("chi", atlas_info.get("chi")))
    f_hz = _safe_float(candidate.get("f_hz")) if isinstance(candidate, dict) else None
    if f_hz is None:
        f_hz = _safe_float(atlas_info.get("f_hz"))
    tau_s = _safe_float(candidate.get("tau_s")) if isinstance(candidate, dict) else None
    if tau_s is None:
        tau_s = _safe_float(atlas_info.get("tau_s"))
    q_val = _safe_float(candidate.get("Q")) if isinstance(candidate, dict) else None
    if q_val is None:
        q_val = _safe_float(atlas_info.get("Q"))
    if tau_s is None and q_val is not None and f_hz and f_hz > 0:
        tau_s = q_val / (math.pi * f_hz)
    return {
        "candidate_id": geometry_id,
        "M_solar": m_solar,
        "chi": chi,
        "mode": _format_mode(candidate_meta.get("mode", atlas_info.get("mode"))),
        "family": str(candidate_meta.get("family", atlas_info.get("family", ""))),
        "source": str(candidate_meta.get("source", atlas_info.get("source", ""))),
        "f_hz": f_hz,
        "tau_s": tau_s,
        "final_area_proxy": _black_hole_area(m_solar, chi),
    }


def _evaluate_common_metric(
    candidate_ids: list[str],
    candidate_groups: dict[str, list[dict[str, Any]]],
    source_runs: list[str],
    metric_field: str,
) -> tuple[list[str], Counter[str]]:
    survivors: list[str] = []
    elimination_counts: Counter[str] = Counter()
    for candidate_id in candidate_ids:
        rows = candidate_groups.get(candidate_id, [])
        rows_by_run = {str(row["source_run_id"]): row for row in rows}
        metric_ok = True
        for source_run in source_runs:
            row = rows_by_run.get(source_run)
            if row is None or row.get(metric_field) is not True:
                metric_ok = False
                if row is not None and row.get(metric_field) is False:
                    elimination_counts[str(row["event_id_canonical"])] += 1
        if metric_ok:
            survivors.append(candidate_id)
    return sorted(survivors), elimination_counts


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_feature_foundry(
    run_id: str,
    *,
    source_runs: list[str],
    atlas_path_cli: str | None,
    catalog_path_cli: str | None,
) -> dict[str, Any]:
    source_runs_resolved = source_runs or [run_id]
    ctx = init_stage(
        run_id,
        STAGE,
        params={
            "source_runs": source_runs_resolved,
            "atlas_path_override": atlas_path_cli,
            "catalog_path_override": catalog_path_cli,
        },
    )

    atlas_path = _resolve_required_path(atlas_path_cli, DEFAULT_ATLAS_REL)
    catalog_path = _resolve_optional_catalog(catalog_path_cli)
    optional_inputs = {"catalog": catalog_path} if catalog_path is not None else None
    check_inputs(ctx, {"atlas": atlas_path}, optional=optional_inputs)

    for source_run in source_runs_resolved:
        source_dir = ctx.out_root / source_run
        if not source_dir.exists():
            raise FileNotFoundError(
                "Source run not found for feature_foundry. "
                f"expected exact path: {source_dir}. "
                "comando exacto para regenerar upstream: "
                f"python -m mvp.pipeline --run-id {source_run}"
            )

    atlas_rows, atlas_by_id = _build_atlas_lookup(atlas_path)
    catalog = _load_catalog(catalog_path)

    event_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    candidate_support: dict[str, set[str]] = defaultdict(set)
    candidate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    basis_counter: Counter[str] = Counter()
    events_missing_catalog: list[str] = []
    runs_missing_s4: list[str] = []
    runs_missing_router: list[str] = []
    runs_missing_s3b: list[str] = []

    for source_run in source_runs_resolved:
        source_dir = ctx.out_root / source_run
        run_valid_path = source_dir / "RUN_VALID" / "verdict.json"
        s1_path = source_dir / "s1_fetch_strain" / "stage_summary.json"
        s3b_path = source_dir / "s3b_multimode_estimates" / "stage_summary.json"
        s4_path = source_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"
        s8_path = source_dir / "s8_family_router" / "outputs" / "family_router.json"

        for label, path in (
            (f"{source_run}:run_valid", run_valid_path),
            (f"{source_run}:s1", s1_path),
            (f"{source_run}:s3b", s3b_path),
            (f"{source_run}:s4", s4_path),
            (f"{source_run}:s8", s8_path),
        ):
            _record_existing_input(ctx, label, path)

        run_valid = _read_json_optional(run_valid_path)
        s1_summary = _read_json_optional(s1_path)
        s3b_summary = _read_json_optional(s3b_path)
        s4_payload = _read_json_optional(s4_path)
        router_payload = _read_json_optional(s8_path)

        if s4_payload is None:
            runs_missing_s4.append(source_run)
        if router_payload is None:
            runs_missing_router.append(source_run)
        if s3b_summary is None:
            runs_missing_s3b.append(source_run)

        event_id = _infer_event_id(source_run, s1_summary, s4_payload, router_payload)
        event_id_canonical = _canonical_event_id(event_id)
        catalog_found, initial_areas = _event_initial_areas(event_id_canonical, catalog)
        if not catalog_found:
            events_missing_catalog.append(event_id_canonical)

        multimode = {}
        if isinstance(s3b_summary, dict):
            payload = s3b_summary.get("multimode_viability")
            multimode = payload if isinstance(payload, dict) else {}
        router_reasons = []
        if isinstance(router_payload, dict) and isinstance(router_payload.get("multimode_viability_reasons"), list):
            router_reasons = [str(item) for item in router_payload["multimode_viability_reasons"]]
        multimode_reasons = multimode.get("reasons") if isinstance(multimode.get("reasons"), list) else router_reasons

        candidate_basis_rows, candidate_basis = _choose_candidate_basis(s4_payload)
        basis_counter[candidate_basis] += 1

        compatible_rows = s4_payload.get("compatible_geometries") if isinstance(s4_payload, dict) else []
        ranked_rows = s4_payload.get("ranked_all") if isinstance(s4_payload, dict) else []
        n_compatible = len(compatible_rows) if isinstance(compatible_rows, list) else 0
        n_ranked = len(ranked_rows) if isinstance(ranked_rows, list) else 0

        missing_artifacts = []
        for label, payload in (
            ("RUN_VALID", run_valid),
            ("s1_fetch_strain", s1_summary),
            ("s3b_multimode_estimates", s3b_summary),
            ("s4_geometry_filter", s4_payload),
            ("s8_family_router", router_payload),
        ):
            if payload is None:
                missing_artifacts.append(label)

        event_rows.append(
            {
                "source_run_id": source_run,
                "event_id": event_id,
                "event_id_canonical": event_id_canonical,
                "run_valid_verdict": str((run_valid or {}).get("verdict", "MISSING")),
                "run_valid_reason": str((run_valid or {}).get("reason", "")),
                "detectors_used": _detectors_used(s1_summary),
                "scientific_outcome": str((router_payload or {}).get("scientific_outcome", "")),
                "fallback_path": str((router_payload or {}).get("fallback_path", "")),
                "program_classification": str((router_payload or {}).get("program_classification", "")),
                "primary_family": str((router_payload or {}).get("primary_family", "")),
                "multimode_viability_class": str(multimode.get("class", (router_payload or {}).get("multimode_viability_class", ""))),
                "multimode_viability_reasons": "; ".join(str(item) for item in multimode_reasons),
                "candidate_basis": candidate_basis,
                "n_candidates_event": len(candidate_basis_rows),
                "n_compatible": n_compatible,
                "n_ranked": n_ranked,
                "s4_metric": str((s4_payload or {}).get("metric", "")),
                "s4_threshold_d2": _safe_float((s4_payload or {}).get("threshold_d2")),
                "catalog_found": catalog_found,
                "initial_area_bound_min": initial_areas["bound_min"],
                "initial_area_bound_max": initial_areas["bound_max"],
                "initial_area_zero_spin_proxy": initial_areas["zero_spin_proxy"],
                "initial_area_chieff_proxy": initial_areas["chieff_projection_proxy"],
                "initial_area_chieff_projection_proxy": initial_areas["chieff_projection_proxy"],
                "missing_artifacts": "; ".join(missing_artifacts),
            }
        )

        for rank, candidate in enumerate(candidate_basis_rows, start=1):
            info = _candidate_info(candidate, atlas_rows, atlas_by_id)
            final_area = info["final_area_proxy"]
            hawking_lower = _hawking_pass(final_area, initial_areas["bound_min"])
            hawking_upper = _hawking_pass(final_area, initial_areas["bound_max"])
            hawking_zero = _hawking_pass(final_area, initial_areas["zero_spin_proxy"])
            hawking_chieff = _hawking_pass(final_area, initial_areas["chieff_projection_proxy"])
            is_compatible = candidate_basis == "compatible_geometries"
            if isinstance(candidate, dict) and "compatible" in candidate:
                is_compatible = bool(candidate.get("compatible"))
            row = {
                "source_run_id": source_run,
                "event_id": event_id,
                "event_id_canonical": event_id_canonical,
                "candidate_basis": candidate_basis,
                "rank": rank,
                "candidate_id": info["candidate_id"],
                "is_compatible": is_compatible,
                "is_kerr_family": _is_kerr_family(info["family"]),
                "support_count": 0,
                "is_common_pre_hawking": False,
                "delta_lnL": _safe_float(candidate.get("delta_lnL")) if isinstance(candidate, dict) else None,
                "d2": _safe_float(candidate.get("d2")) if isinstance(candidate, dict) else None,
                "posterior_weight": _safe_float(candidate.get("posterior_weight")) if isinstance(candidate, dict) else None,
                "prior_weight": _safe_float(candidate.get("prior_weight")) if isinstance(candidate, dict) else None,
                "M_solar": info["M_solar"],
                "chi": info["chi"],
                "mode": info["mode"],
                "family": info["family"],
                "source": info["source"],
                "f_hz": info["f_hz"],
                "tau_s": info["tau_s"],
                "final_area_proxy": final_area,
                "hawking_pass_area_lower_bound": hawking_lower,
                "hawking_pass_area_upper_bound": hawking_upper,
                "hawking_interval_status": _hawking_interval_status(
                    final_area,
                    initial_areas["bound_min"],
                    initial_areas["bound_max"],
                ),
                "hawking_pass_zero_spin_proxy": hawking_zero,
                "hawking_pass_chieff_projection_proxy": hawking_chieff,
                "hawking_pass_chieff_proxy": hawking_chieff,
                "n_fail_events_area_lower_bound": 0,
                "n_fail_events_area_upper_bound": 0,
                "n_fail_events_zero_spin_proxy": 0,
                "n_fail_events_chieff_projection_proxy": 0,
                "area_lower_bound_failure_pattern": "UNSEEN",
                "area_upper_bound_failure_pattern": "UNSEEN",
                "zero_spin_proxy_failure_pattern": "UNSEEN",
                "chieff_projection_proxy_failure_pattern": "UNSEEN",
            }
            candidate_rows.append(row)
            candidate_support[info["candidate_id"]].add(source_run)
            candidate_groups[info["candidate_id"]].append(row)

    total_runs = len(source_runs_resolved)
    common_pre_hawking = sorted(gid for gid, runs in candidate_support.items() if len(runs) == total_runs)
    common_pre_hawking_set = set(common_pre_hawking)
    metric_fields = {
        "area_lower_bound": "hawking_pass_area_lower_bound",
        "area_upper_bound": "hawking_pass_area_upper_bound",
        "zero_spin_proxy": "hawking_pass_zero_spin_proxy",
        "chieff_projection_proxy": "hawking_pass_chieff_projection_proxy",
    }

    candidate_summary_by_id: dict[str, dict[str, Any]] = {}
    support_rows = []
    candidate_is_kerr: dict[str, bool] = {}
    for candidate_id, runs in sorted(candidate_support.items()):
        rows = candidate_groups.get(candidate_id, [])
        info = atlas_by_id.get(candidate_id, {})
        support_count = len(runs)
        is_kerr_family = bool(rows) and all(bool(row.get("is_kerr_family")) for row in rows)
        candidate_is_kerr[candidate_id] = is_kerr_family
        summary_row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "support_count": support_count,
            "M_solar": info.get("M_solar"),
            "chi": info.get("chi"),
            "family": info.get("family"),
            "source": info.get("source"),
            "f_hz": info.get("f_hz"),
            "tau_s": info.get("tau_s"),
            "final_area_proxy": info.get("final_area_proxy"),
            "is_kerr_family": is_kerr_family,
            "is_common_pre_hawking": candidate_id in common_pre_hawking_set,
        }
        for metric_label, metric_field in metric_fields.items():
            fail_events = sorted(
                str(row["event_id_canonical"])
                for row in rows
                if row.get(metric_field) is False
            )
            fail_count = len(fail_events)
            unknown_count = sum(1 for row in rows if row.get(metric_field) is None)
            summary_row[f"n_fail_events_{metric_label}"] = fail_count
            summary_row[f"n_unknown_events_{metric_label}"] = unknown_count
            summary_row[f"{metric_label}_failure_pattern"] = _failure_pattern(
                fail_count,
                unknown_count,
                support_count,
            )
            summary_row[f"{metric_label}_fail_events"] = fail_events
        candidate_summary_by_id[candidate_id] = summary_row
        support_rows.append(summary_row)

    for row in candidate_rows:
        support_count = len(candidate_support.get(row["candidate_id"], set()))
        row["support_count"] = support_count
        row["is_common_pre_hawking"] = row["candidate_id"] in common_pre_hawking_set
        summary_row = candidate_summary_by_id.get(str(row["candidate_id"]), {})
        row["n_fail_events_area_lower_bound"] = int(summary_row.get("n_fail_events_area_lower_bound", 0))
        row["n_fail_events_area_upper_bound"] = int(summary_row.get("n_fail_events_area_upper_bound", 0))
        row["n_fail_events_zero_spin_proxy"] = int(summary_row.get("n_fail_events_zero_spin_proxy", 0))
        row["n_fail_events_chieff_projection_proxy"] = int(summary_row.get("n_fail_events_chieff_projection_proxy", 0))
        row["area_lower_bound_failure_pattern"] = str(summary_row.get("area_lower_bound_failure_pattern", "UNSEEN"))
        row["area_upper_bound_failure_pattern"] = str(summary_row.get("area_upper_bound_failure_pattern", "UNSEEN"))
        row["zero_spin_proxy_failure_pattern"] = str(summary_row.get("zero_spin_proxy_failure_pattern", "UNSEEN"))
        row["chieff_projection_proxy_failure_pattern"] = str(
            summary_row.get("chieff_projection_proxy_failure_pattern", "UNSEEN")
        )

    common_metric_candidates: dict[str, list[str]] = {}
    common_metric_eliminations: dict[str, dict[str, int]] = {}
    for metric_label, metric_field in metric_fields.items():
        survivors, eliminations = _evaluate_common_metric(
            common_pre_hawking,
            candidate_groups,
            source_runs_resolved,
            metric_field,
        )
        common_metric_candidates[metric_label] = survivors
        common_metric_eliminations[metric_label] = dict(sorted(eliminations.items()))

    kerr_only_candidate_ids = sorted(candidate_id for candidate_id, is_kerr in candidate_is_kerr.items() if is_kerr)
    common_pre_hawking_kerr = [candidate_id for candidate_id in common_pre_hawking if candidate_is_kerr.get(candidate_id, False)]
    kerr_metric_candidates: dict[str, list[str]] = {}
    for metric_label, metric_field in metric_fields.items():
        survivors, _ = _evaluate_common_metric(
            common_pre_hawking_kerr,
            candidate_groups,
            source_runs_resolved,
            metric_field,
        )
        kerr_metric_candidates[metric_label] = survivors

    event_rows.sort(key=lambda row: source_runs_resolved.index(row["source_run_id"]))
    candidate_rows.sort(key=lambda row: (source_runs_resolved.index(row["source_run_id"]), int(row["rank"]), row["candidate_id"]))

    event_fieldnames = [
        "source_run_id",
        "event_id",
        "event_id_canonical",
        "run_valid_verdict",
        "run_valid_reason",
        "detectors_used",
        "scientific_outcome",
        "fallback_path",
        "program_classification",
        "primary_family",
        "multimode_viability_class",
        "multimode_viability_reasons",
        "candidate_basis",
        "n_candidates_event",
        "n_compatible",
        "n_ranked",
        "s4_metric",
        "s4_threshold_d2",
        "catalog_found",
        "initial_area_bound_min",
        "initial_area_bound_max",
        "initial_area_zero_spin_proxy",
        "initial_area_chieff_proxy",
        "initial_area_chieff_projection_proxy",
        "missing_artifacts",
    ]
    candidate_fieldnames = [
        "source_run_id",
        "event_id",
        "event_id_canonical",
        "candidate_basis",
        "rank",
        "candidate_id",
        "is_compatible",
        "is_kerr_family",
        "support_count",
        "is_common_pre_hawking",
        "delta_lnL",
        "d2",
        "posterior_weight",
        "prior_weight",
        "M_solar",
        "chi",
        "mode",
        "family",
        "source",
        "f_hz",
        "tau_s",
        "final_area_proxy",
        "hawking_pass_area_lower_bound",
        "hawking_pass_area_upper_bound",
        "hawking_interval_status",
        "hawking_pass_zero_spin_proxy",
        "hawking_pass_chieff_projection_proxy",
        "hawking_pass_chieff_proxy",
        "n_fail_events_area_lower_bound",
        "n_fail_events_area_upper_bound",
        "n_fail_events_zero_spin_proxy",
        "n_fail_events_chieff_projection_proxy",
        "area_lower_bound_failure_pattern",
        "area_upper_bound_failure_pattern",
        "zero_spin_proxy_failure_pattern",
        "chieff_projection_proxy_failure_pattern",
    ]

    event_csv = ctx.outputs_dir / "event_summary.csv"
    candidate_csv = ctx.outputs_dir / "candidate_rows.csv"
    _write_csv(event_csv, event_rows, event_fieldnames)
    _write_csv(candidate_csv, candidate_rows, candidate_fieldnames)

    posthoc = {
        "schema_version": "experiment_feature_foundry_v2",
        "stage": STAGE,
        "host_run_id": run_id,
        "source_runs": source_runs_resolved,
        "atlas_path": str(atlas_path),
        "catalog_path": str(catalog_path) if catalog_path is not None else None,
        "summary": {
            "n_source_runs": total_runs,
            "n_event_rows": len(event_rows),
            "n_candidate_rows": len(candidate_rows),
            "n_unique_candidates": len(candidate_support),
            "n_unique_candidates_kerr_only": len(kerr_only_candidate_ids),
            "n_common_pre_hawking": len(common_pre_hawking),
            "n_common_pre_hawking_kerr_only": len(common_pre_hawking_kerr),
            "n_common_hawking_area_lower_bound": len(common_metric_candidates["area_lower_bound"]),
            "n_common_hawking_area_upper_bound": len(common_metric_candidates["area_upper_bound"]),
            "n_common_hawking_zero_spin_proxy": len(common_metric_candidates["zero_spin_proxy"]),
            "n_common_hawking_chieff_projection_proxy": len(common_metric_candidates["chieff_projection_proxy"]),
            "n_common_hawking_chieff_proxy": len(common_metric_candidates["chieff_projection_proxy"]),
            "n_common_hawking_area_lower_bound_kerr_only": len(kerr_metric_candidates["area_lower_bound"]),
            "n_common_hawking_area_upper_bound_kerr_only": len(kerr_metric_candidates["area_upper_bound"]),
            "candidate_basis_counts": dict(sorted(basis_counter.items())),
            "n_runs_missing_s4": len(runs_missing_s4),
            "n_runs_missing_s8": len(runs_missing_router),
            "n_runs_missing_s3b": len(runs_missing_s3b),
            "n_events_missing_catalog": len(set(events_missing_catalog)),
        },
        "common_pre_hawking_candidate_ids": common_pre_hawking,
        "common_hawking_area_lower_bound_candidate_ids": common_metric_candidates["area_lower_bound"],
        "common_hawking_area_upper_bound_candidate_ids": common_metric_candidates["area_upper_bound"],
        "common_hawking_zero_spin_candidate_ids": common_metric_candidates["zero_spin_proxy"],
        "common_hawking_chieff_candidate_ids": common_metric_candidates["chieff_projection_proxy"],
        "common_hawking_chieff_projection_candidate_ids": common_metric_candidates["chieff_projection_proxy"],
        "events_missing_catalog": sorted(set(events_missing_catalog)),
        "runs_missing_s4": sorted(runs_missing_s4),
        "runs_missing_s8": sorted(runs_missing_router),
        "runs_missing_s3b": sorted(runs_missing_s3b),
        "area_lower_bound_elimination_counts_by_event": common_metric_eliminations["area_lower_bound"],
        "area_upper_bound_elimination_counts_by_event": common_metric_eliminations["area_upper_bound"],
        "zero_spin_elimination_counts_by_event": common_metric_eliminations["zero_spin_proxy"],
        "chieff_elimination_counts_by_event": common_metric_eliminations["chieff_projection_proxy"],
        "chieff_projection_elimination_counts_by_event": common_metric_eliminations["chieff_projection_proxy"],
        "per_candidate_support": support_rows,
        "kerr_only": {
            "candidate_ids": kerr_only_candidate_ids,
            "common_pre_hawking_candidate_ids": common_pre_hawking_kerr,
            "common_hawking_area_lower_bound_candidate_ids": kerr_metric_candidates["area_lower_bound"],
            "common_hawking_area_upper_bound_candidate_ids": kerr_metric_candidates["area_upper_bound"],
            "common_hawking_zero_spin_candidate_ids": kerr_metric_candidates["zero_spin_proxy"],
            "common_hawking_chieff_projection_candidate_ids": kerr_metric_candidates["chieff_projection_proxy"],
        },
        "notes": [
            "common_pre_hawking uses exact candidate_id intersection across source runs on the chosen per-event basis",
            "initial_area_bound_min and initial_area_bound_max define a physical Kerr-area interval at fixed source masses",
            "hawking_pass_area_upper_bound is identical to the zero-spin conservative proxy by construction",
            "hawking_pass_chieff_projection_proxy is a post-hoc projection proxy derived from catalog source masses and |chi_eff|",
        ],
    }
    posthoc_path = ctx.outputs_dir / "posthoc_checks.json"
    write_json_atomic(posthoc_path, posthoc)

    finalize(
        ctx,
        artifacts={
            "event_summary_csv": event_csv,
            "candidate_rows_csv": candidate_csv,
            "posthoc_checks_json": posthoc_path,
        },
        results={
            "n_source_runs": total_runs,
            "n_event_rows": len(event_rows),
            "n_candidate_rows": len(candidate_rows),
            "n_unique_candidates": len(candidate_support),
            "n_common_pre_hawking": len(common_pre_hawking),
            "n_common_hawking_area_lower_bound": len(common_metric_candidates["area_lower_bound"]),
            "n_common_hawking_area_upper_bound": len(common_metric_candidates["area_upper_bound"]),
            "n_common_hawking_zero_spin_proxy": len(common_metric_candidates["zero_spin_proxy"]),
            "n_common_hawking_chieff_projection_proxy": len(common_metric_candidates["chieff_projection_proxy"]),
        },
    )
    log_stage_paths(ctx)
    return posthoc


def main() -> int:
    ap = argparse.ArgumentParser(description="Build analysis-ready feature tables from BASURIN source runs")
    ap.add_argument("--run-id", required=True, help="Host run_id where experiment outputs are written")
    ap.add_argument(
        "--source-runs",
        default=None,
        help="Comma-separated source run_ids to summarize. Defaults to --run-id itself.",
    )
    ap.add_argument(
        "--atlas-path",
        default=None,
        help="Atlas JSON path. Defaults to docs/ringdown/atlas/atlas_berti_v2.json",
    )
    ap.add_argument(
        "--catalog-path",
        default=None,
        help="Optional catalog CSV with event,m1_source,m2_source,chi_eff columns.",
    )
    args = ap.parse_args()

    source_runs = _parse_csv_tokens(args.source_runs) or [args.run_id]
    run_feature_foundry(
        args.run_id,
        source_runs=source_runs,
        atlas_path_cli=args.atlas_path,
        catalog_path_cli=args.catalog_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
