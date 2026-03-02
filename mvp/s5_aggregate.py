#!/usr/bin/env python3
"""MVP Stage 5: Aggregate compatible sets across multiple events.

CLI:
    python mvp/s5_aggregate.py --out-run <agg_run_id> \
        --source-runs run_A,run_B [--min-coverage 1.0]

Inputs:  runs/<run>/s4_geometry_filter/outputs/compatible_set.json (per source run)
Outputs: runs/<agg_run>/s5_aggregate/outputs/aggregate.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in [_here.parents[0], _here.parents[1]]:
    if (_cand / "basurin_io.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from mvp.contracts import init_stage, check_inputs, finalize, abort
from basurin_io import sha256_file, utc_now_iso, write_json_atomic

STAGE = "s5_aggregate"


def _relpath_under(root: Path, p: Path) -> str:
    """Return POSIX path relative to root when possible, else absolute POSIX path."""
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _extract_compatible_geometry_ids(
    payload: dict[str, Any], source_path: Path | None = None,
) -> set[str]:
    """Extract compatible geometry IDs supporting canonical and legacy payloads."""
    required_keys = {"schema_version", "event_id", "compatible_geometry_ids"}
    present_keys = sorted(payload.keys())
    source_display = source_path.as_posix() if source_path is not None else "<unknown>"

    def _schema_error(reason: str) -> RuntimeError:
        hint = ""
        if source_path is not None and len(source_path.parts) >= 5:
            hint = (
                " Hint: include 'schema_version': 1 and regenerate upstream with "
                f"`python -m mvp.s4_geometry_filter --run {source_path.parts[-5]} --atlas-path <atlas.json>`"
            )
        return RuntimeError(
            "Invalid compatible_set schema at "
            f"{source_display}: {reason}. "
            f"Present keys={present_keys}."
            f"{hint}"
        )

    if "schema_version" in payload:
        if set(payload.keys()) != required_keys:
            raise _schema_error("unexpected keys or missing required keys")
        if payload.get("schema_version") != 1:
            raise _schema_error("'schema_version' must be integer 1")

        event_id = payload.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise _schema_error("'event_id' must be a non-empty string")

        ids = payload.get("compatible_geometry_ids")
        if not isinstance(ids, list) or not ids:
            raise _schema_error("'compatible_geometry_ids' must be a non-empty array")
        if any(not isinstance(gid, str) or not gid.strip() for gid in ids):
            raise _schema_error("'compatible_geometry_ids' must contain only non-empty strings")
        return {gid for gid in ids}

    compatible_geometries = payload.get("compatible_geometries")
    if isinstance(compatible_geometries, list):
        out: set[str] = set()
        for row in compatible_geometries:
            if not isinstance(row, dict) or row.get("compatible") is not True:
                continue
            geometry_id = row.get("geometry_id") if "geometry_id" in row else row.get("id")
            if isinstance(geometry_id, str) and geometry_id:
                out.add(geometry_id)
        return out

    compatible_entries = payload.get("compatible_entries")
    if isinstance(compatible_entries, list):
        out: set[str] = set()
        for row in compatible_entries:
            if isinstance(row, str) and row:
                out.add(row)
                continue
            if isinstance(row, dict):
                geometry_id = row.get("id") if "id" in row else row.get("geometry_id")
                if isinstance(geometry_id, str) and geometry_id:
                    out.add(geometry_id)
        return out

    compatible_ids = payload.get("compatible_ids")
    if isinstance(compatible_ids, list):
        return {str(x) for x in compatible_ids}

    raise _schema_error(
        "missing supported keys: expected canonical schema or legacy keys "
        "('compatible_geometries', 'compatible_entries', 'compatible_ids')"
    )


def _parse_s6b_indices(rows: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("atlas_index"), int):
            out.append(int(row["atlas_index"]))
    return out




def _extract_mode(multimode: dict[str, Any], label: str) -> dict[str, Any] | None:
    for row in multimode.get("modes", []):
        if isinstance(row, dict) and str(row.get("label")) == label:
            return row
    return None


def _censoring_from_multimode(multimode: dict[str, Any]) -> dict[str, Any]:
    flags = multimode.get("results", {}).get("quality_flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(f) for f in flags]

    mode220 = _extract_mode(multimode, "220")
    mode221 = _extract_mode(multimode, "221")

    has_220 = mode220 is not None and isinstance(mode220.get("ln_f"), (int, float))
    has_221_coords = mode221 is not None and isinstance(mode221.get("ln_f"), (int, float)) and isinstance(mode221.get("ln_Q"), (int, float))
    flagged_221 = [f for f in flags if "221" in f]

    has_221 = bool(has_221_coords and not flagged_221)
    reason = None if has_221 else (flagged_221[0] if flagged_221 else (flags[0] if flags else "mode_221_missing_or_degraded"))

    return {
        "has_220": has_220,
        "has_221": has_221,
        "reason": reason,
        "weight_scalar": 1.0 if has_220 else 0.0,
        "weight_vector4": 1.0 if has_221 else 0.0,
    }


def _first_number(mapping: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        val = mapping.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)
    return None


def _extract_vector4(s4c_payload: dict[str, Any]) -> dict[str, Any]:
    deltas = s4c_payload.get("deltas", {})
    if not isinstance(deltas, dict):
        deltas = {}
    return {
        "delta_f_220": _first_number(deltas, ["delta_f_220", "delta_logfreq"]),
        "delta_tau_220": _first_number(deltas, ["delta_tau_220", "delta_logtau"]),
        "delta_f_221": _first_number(deltas, ["delta_f_221"]),
        "delta_tau_221": _first_number(deltas, ["delta_tau_221"]),
        "cov4": s4c_payload.get("cov4") if isinstance(s4c_payload.get("cov4"), list) else None,
    }
def aggregate_compatible_sets(
    source_data: list[dict[str, Any]], min_coverage: float = 1.0, top_k: int = 50,
) -> dict[str, Any]:
    n_events = len(source_data)
    s6b_ready = [src for src in source_data if src.get("s6b_present")]
    use_s6b_mode = len(s6b_ready) > 0

    warnings: list[str] = []
    if use_s6b_mode:
        ranked_sets = [set(src.get("ranked_indices", [])) for src in s6b_ready]
        compatible_sets = [set(src.get("compatible_indices", [])) for src in s6b_ready]

        common_ranked_ids = sorted(set.intersection(*ranked_sets)) if ranked_sets else []
        common_compatible_ids = sorted(set.intersection(*compatible_sets)) if compatible_sets else []

        common_ranked_geometries = common_ranked_ids
        common_compatible_geometries = common_compatible_ids

        for src in source_data:
            if not src.get("s6b_present"):
                warnings.append(f"MISSING_S6B_RANKED:{src['run_id']}")
    else:
        compatible_sets = [set(map(str, src.get("compatible_ids", set()))) for src in source_data]

        ranked_sets: list[set[str]] = []
        for src in source_data:
            ranked_all = src.get("ranked_all", [])
            ranked_rows = ranked_all if top_k is None else ranked_all[:max(0, top_k)]
            ranked_sets.append(
                {
                    str(row.get("geometry_id"))
                    for row in ranked_rows
                    if isinstance(row, dict) and row.get("geometry_id") is not None
                }
            )

        common_ranked_ids = sorted(set.intersection(*ranked_sets)) if ranked_sets else []
        common_compatible_ids = sorted(set.intersection(*compatible_sets)) if compatible_sets else []

        common_ranked_geometries = common_ranked_ids
        common_compatible_geometries = common_compatible_ids

    if n_events == 0:
        return {
            "schema_version": "mvp_aggregate_v2",
            "n_events": 0,
            "events": [],
            "joint_posterior": {
                "prior_type": "uniform_entries",
                "normalization": "relative_only",
                "combination": "sum_logL_rel",
                "chi2_dof_per_event": 2,
                "chi2_interpretation": "min_over_atlas_not_chi2",
                "best_entry_id": None,
                "d2_sum_min": None,
                "log_likelihood_rel_best": None,
                "common_geometries": [],
                "common_ranked_geometries": [],
                "common_compatible_geometries": [],
            },
            "n_common_ranked": 0,
            "common_ranked_geometries": [],
            "n_common_compatible": 0,
            "common_compatible_geometries": [],
            "common_geometries": [],
            "n_common_geometries": 0,
            "coverage_histogram": {},
            "coverage_histogram_basis": "ranked_all",
            "n_total_unique_geometries": 0,
            "min_coverage": min_coverage,
            "min_count": 0,
            "warnings": ["NO_EVENTS_TO_AGGREGATE"],
        }

    counter: Counter[str] = Counter()
    d2_by_geo: dict[str, list[float | None]] = {}
    metadata: dict[str, Any] = {}
    events: list[dict[str, Any]] = []

    def _extract_d2(row: dict[str, Any], metric: str) -> float | None:
        d2 = row.get("d2")
        if isinstance(d2, (int, float)) and math.isfinite(d2):
            return float(d2)
        distance = row.get("distance")
        if metric == "mahalanobis_log" and isinstance(distance, (int, float)) and math.isfinite(distance):
            return float(distance) ** 2
        return None

    for src in source_data:
        metric = str(src.get("metric", ""))
        ranked_all = src.get("ranked_all", [])
        if top_k is None:
            ranked_rows = ranked_all
        else:
            ranked_rows = ranked_all[:max(0, top_k)]
        threshold_d2 = src.get("threshold_d2")
        censoring = _censoring_from_multimode(src.get("multimode", {}) if isinstance(src.get("multimode"), dict) else {})
        raw_vote = str((src.get("s4c", {}) if isinstance(src.get("s4c"), dict) else {}).get("verdict") or "INCONCLUSIVE").upper()
        vote_kerr = raw_vote if censoring["has_221"] and raw_vote in {"PASS", "FAIL", "INCONCLUSIVE"} else "INCONCLUSIVE"
        if not censoring["has_221"]:
            vote_kerr = "INCONCLUSIVE"
            censoring["weight_vector4"] = 0.0

        scalar_obj = {
            "kerr_tension": _first_number(src.get("s4c", {}) if isinstance(src.get("s4c"), dict) else {}, ["kerr_tension", "chi_best", "d2_min"]),
            "kerr_tension_pvalue": _first_number(src.get("s4c", {}) if isinstance(src.get("s4c"), dict) else {}, ["kerr_tension_pvalue", "p_value", "pvalue"]),
        }
        vector4_obj = _extract_vector4(src.get("s4c", {}) if isinstance(src.get("s4c"), dict) else {})

        events.append(
            {
                "run_id": src["run_id"],
                "event_id": src["event_id"],
                "metric": metric,
                "threshold_d2": threshold_d2,
                "n_atlas": int(src.get("n_atlas", len(ranked_rows))),
                "ranked": list(src.get("ranked_indices", [])),
                "compatible": list(src.get("compatible_indices", [])),
                "censoring": {
                    "has_221": censoring["has_221"],
                    "vote_kerr": vote_kerr,
                    "weight_scalar": float(censoring["weight_scalar"]),
                    "weight_vector4": float(censoring["weight_vector4"]),
                    "reason": censoring["reason"],
                },
                "quality": {"ringdown_snr": src.get("ringdown_snr")},
                "scalar": scalar_obj,
                "vector4": vector4_obj,
            }
        )

        current: dict[str, float | None] = {}
        for row in ranked_rows:
            gid = row.get("geometry_id")
            if not gid:
                continue
            current[gid] = _extract_d2(row, metric)
            if gid not in metadata and row.get("metadata"):
                metadata[gid] = row["metadata"]

        for gid in current:
            counter[gid] += 1

        for gid in d2_by_geo:
            d2_by_geo[gid].append(current.get(gid))
        for gid, d2 in current.items():
            if gid not in d2_by_geo:
                d2_by_geo[gid] = [None] * (len(events) - 1)
                d2_by_geo[gid].append(d2)

    min_count = max(1, int(math.ceil(min_coverage * n_events)))
    all_rows: list[dict[str, Any]] = []
    for gid in sorted(d2_by_geo):
        per_event = d2_by_geo[gid]
        finite_d2 = [d for d in per_event if isinstance(d, (int, float)) and math.isfinite(d)]
        coverage = counter[gid] / n_events
        d2_sum = float(sum(finite_d2))
        row = {
            "geometry_id": gid,
            "d2_per_event": per_event,
            "d2_sum": d2_sum,
            "coverage": coverage,
            "log_likelihood_rel_joint": -0.5 * d2_sum,
            "support_count": 0,
            "support_fraction": 0.0,
        }
        if metadata.get(gid) is not None:
            row["metadata"] = metadata[gid]
        all_rows.append(row)

    eligible = [r for r in all_rows if r["coverage"] >= min_coverage]
    if eligible:
        best = min(eligible, key=lambda r: (r["d2_sum"], r["geometry_id"]))
        best_logl = best["log_likelihood_rel_joint"]
        logits = [r["log_likelihood_rel_joint"] for r in eligible]
        max_logit = max(logits)
        exp_vals = [math.exp(v - max_logit) for v in logits]
        norm = sum(exp_vals)

        for row, ev in zip(eligible, exp_vals):
            row["posterior_weight_joint"] = ev / norm
            row["delta_lnL_joint"] = row["log_likelihood_rel_joint"] - best_logl
            support_count = 0
            for idx, d2 in enumerate(row["d2_per_event"]):
                threshold = events[idx].get("threshold_d2")
                if (
                    isinstance(d2, (int, float))
                    and math.isfinite(d2)
                    and isinstance(threshold, (int, float))
                    and math.isfinite(threshold)
                    and d2 <= threshold
                ):
                    support_count += 1
            row["support_count"] = support_count
            row["support_fraction"] = support_count / n_events

        ranked = sorted(
            eligible,
            key=lambda r: (-r["posterior_weight_joint"], r["geometry_id"]),
        )
        common = ranked[:max(0, top_k)] if top_k is not None else ranked
        best_entry_id = best["geometry_id"]
        d2_sum_min = best["d2_sum"]
        log_likelihood_rel_best = best_logl
    else:
        ranked = []
        common = []
        best_entry_id = None
        d2_sum_min = None
        log_likelihood_rel_best = None

    coverage_hist: dict[str, int] = {}
    for c in counter.values():
        coverage_hist[str(c)] = coverage_hist.get(str(c), 0) + 1

    if use_s6b_mode:
        compat_counter: Counter[int] = Counter()
        for e in events:
            for idx in e.get("compatible", []):
                compat_counter[int(idx)] += 1
        common_compatible_ids = sorted([idx for idx, c in compat_counter.items() if c >= min_count])
        common_compatible_geometries = common_compatible_ids

    n_common_compatible = len(common_compatible_geometries)
    if n_common_compatible == 0 and "NO_COMMON_COMPATIBLE_GEOMETRIES" not in warnings:
        warnings.append("NO_COMMON_COMPATIBLE_GEOMETRIES")

    return {
        "schema_version": "mvp_aggregate_v2",
        "n_events": n_events, "min_coverage": min_coverage, "min_count": min_count,
        "n_total_unique_geometries": len(counter),
        # Backward-compatible alias: common_geometries === ranked intersection.
        "n_common_geometries": len(common_ranked_geometries), "common_geometries": common_ranked_geometries,
        "n_common_ranked": len(common_ranked_geometries),
        "common_ranked_geometries": common_ranked_geometries,
        "n_common_compatible": n_common_compatible,
        "common_compatible_geometries": common_compatible_geometries,
        "joint_posterior": {
            "prior_type": "uniform_entries",
            "normalization": "relative_only",
            "combination": "sum_logL_rel",
            "chi2_dof_per_event": 2,
            "chi2_interpretation": "min_over_atlas_not_chi2",
            "best_entry_id": best_entry_id,
            "d2_sum_min": d2_sum_min,
            "log_likelihood_rel_best": log_likelihood_rel_best,
            "common_geometries": common_ranked_geometries,
            "common_ranked_geometries": common_ranked_geometries,
            "common_compatible_geometries": common_compatible_geometries,
            "joint_ranked_all": ranked,
        },
        "coverage_histogram": coverage_hist,
        "coverage_histogram_basis": "ranked_all",
        "warnings": warnings,
        "events": events,
    }


def compute_deviation_distribution(
    source_data: list[dict[str, Any]],
    catalog: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any] | None:
    """Compute δf and δQ deviations from Kerr predictions per event.

    For each event i with observed (f_i, Q_i, σ_f_i, σ_Q_i) and
    catalog (M_i, χ_i), compute:
        f_Kerr_i = kerr_qnm(M_i, χ_i).f_hz
        δf_i = (f_i - f_Kerr_i) / f_Kerr_i
        σ_δf_i = σ_f_i / f_Kerr_i

    Then combine via inverse-variance weighting:
        δf_combined = Σ(w_i · δf_i) / Σ(w_i)   where w_i = 1/σ_δf_i²

    Also computes χ² test for GR consistency (δf=0, δQ=0).

    Args:
        source_data: List of per-event data dicts (from s5 main()).
        catalog: Dict {event_id: {m_final_msun, chi_final}}.

    Returns:
        deviation_analysis dict to embed in aggregate.json,
        or None if catalog is None or no events match.
    """
    if catalog is None or not catalog:
        return None

    from mvp.kerr_qnm_fits import kerr_qnm

    per_event_rows: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    for src in source_data:
        event_id = src.get("event_id", "unknown")
        cat_entry = catalog.get(event_id)
        if cat_entry is None:
            warnings_list.append(f"Event {event_id} not in catalog; skipped")
            continue

        m_final = float(cat_entry.get("m_final_msun", 0))
        chi_final = float(cat_entry.get("chi_final", 0))
        if m_final <= 0 or not (0 <= chi_final < 1):
            warnings_list.append(
                f"Event {event_id}: invalid catalog params M={m_final}, χ={chi_final}; skipped"
            )
            continue

        obs = src.get("observables", {})
        f_obs = obs.get("f_hz")
        Q_obs = obs.get("Q")

        # Fall back to ranked_all best entry if observables not stored
        if f_obs is None:
            ranked = src.get("ranked_all", [])
            if ranked:
                first = ranked[0]
                f_obs = first.get("f_hz")
                Q_obs = first.get("Q")

        if f_obs is None or Q_obs is None:
            warnings_list.append(f"Event {event_id}: missing observables; skipped")
            continue

        f_obs = float(f_obs)
        Q_obs = float(Q_obs)

        # Get uncertainties from source compatible_set metadata
        sigma_f = src.get("sigma_f_hz")
        sigma_Q = src.get("sigma_Q")

        if sigma_f is None or not math.isfinite(sigma_f) or sigma_f <= 0:
            # Use a conservative 10% uncertainty if not available
            sigma_f = 0.10 * f_obs
            warnings_list.append(
                f"Event {event_id}: σ_f not available, using 10% relative"
            )
        if sigma_Q is None or not math.isfinite(sigma_Q) or sigma_Q <= 0:
            sigma_Q = 0.30 * Q_obs
            warnings_list.append(
                f"Event {event_id}: σ_Q not available, using 30% relative"
            )

        # Kerr prediction
        try:
            kerr = kerr_qnm(m_final, chi_final)
        except Exception as exc:
            warnings_list.append(f"Event {event_id}: kerr_qnm failed: {exc}; skipped")
            continue

        f_kerr = kerr.f_hz
        Q_kerr = kerr.Q

        if f_kerr <= 0 or Q_kerr <= 0:
            warnings_list.append(f"Event {event_id}: invalid Kerr prediction; skipped")
            continue

        delta_f_rel = (f_obs - f_kerr) / f_kerr
        delta_Q_rel = (Q_obs - Q_kerr) / Q_kerr
        sigma_delta_f_rel = float(sigma_f) / f_kerr
        sigma_delta_Q_rel = float(sigma_Q) / Q_kerr

        per_event_rows.append({
            "event_id": event_id,
            "f_obs": f_obs, "f_kerr": f_kerr,
            "Q_obs": Q_obs, "Q_kerr": Q_kerr,
            "m_final_msun": m_final, "chi_final": chi_final,
            "delta_f_rel": delta_f_rel,
            "sigma_delta_f_rel": sigma_delta_f_rel,
            "delta_Q_rel": delta_Q_rel,
            "sigma_delta_Q_rel": sigma_delta_Q_rel,
        })

    if not per_event_rows:
        return {
            "schema_version": "mvp_deviation_v1",
            "parameter": "delta_f220_rel",
            "n_events_in_analysis": 0,
            "per_event": [],
            "combined": None,
            "warnings": warnings_list,
            "note": "No events matched catalog entries",
        }

    # Inverse-variance weighted combination for f
    w_f = [1.0 / r["sigma_delta_f_rel"] ** 2 for r in per_event_rows]
    sum_w_f = sum(w_f)
    delta_f_combined = sum(w * r["delta_f_rel"] for w, r in zip(w_f, per_event_rows)) / sum_w_f
    sigma_delta_f_combined = 1.0 / math.sqrt(sum_w_f)

    # Inverse-variance weighted combination for Q
    w_Q = [1.0 / r["sigma_delta_Q_rel"] ** 2 for r in per_event_rows]
    sum_w_Q = sum(w_Q)
    delta_Q_combined = sum(w * r["delta_Q_rel"] for w, r in zip(w_Q, per_event_rows)) / sum_w_Q
    sigma_delta_Q_combined = 1.0 / math.sqrt(sum_w_Q)

    # χ² test under GR null hypothesis (δf=0 per event)
    chi2_f = sum(r["delta_f_rel"] ** 2 / r["sigma_delta_f_rel"] ** 2
                 for r in per_event_rows)
    n_events = len(per_event_rows)

    # p-value from chi2 distribution with n_events DOF
    try:
        from scipy.stats import chi2 as scipy_chi2
        p_value_f = float(scipy_chi2.sf(chi2_f, df=n_events))
    except Exception:
        # Approximation without scipy: use simple lookup
        # chi2 sf ≈ 1 - CDF; for large chi2: p < 0.05
        p_value_f = float("nan")

    consistent_gr_95 = (not math.isnan(p_value_f)) and (p_value_f > 0.05)

    if math.isnan(p_value_f):
        interpretation = "GR test inconclusive (scipy unavailable)"
    elif p_value_f > 0.05:
        interpretation = "GR consistent"
    elif p_value_f > 0.003:
        interpretation = "GR tension"
    else:
        interpretation = "GR excluded"

    return {
        "schema_version": "mvp_deviation_v1",
        "parameter": "delta_f220_rel",
        "catalog_source": "GWTC catalog (posterior medians)",
        "n_events_in_analysis": n_events,
        "per_event": per_event_rows,
        "combined": {
            "delta_f_rel": delta_f_combined,
            "sigma_delta_f_rel": sigma_delta_f_combined,
            "delta_Q_rel": delta_Q_combined,
            "sigma_delta_Q_rel": sigma_delta_Q_combined,
            "n_events": n_events,
            "chi2_GR": chi2_f,
            "p_value_GR": p_value_f,
            "consistent_GR_95": consistent_gr_95,
        },
        "interpretation": interpretation,
        "warnings": warnings_list,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=f"MVP {STAGE}: aggregate events")
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--source-runs", required=True)
    ap.add_argument("--min-coverage", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument(
        "--catalog-path", default=None,
        help="Optional JSON catalog {event_id: {m_final_msun, chi_final}} for deviation analysis",
    )
    args = ap.parse_args()

    source_runs = [r.strip() for r in args.source_runs.split(",") if r.strip()]
    if not source_runs:
        print("ERROR: --source-runs empty", file=sys.stderr)
        raise SystemExit(2)

    # Create RUN_VALID for aggregate run
    from basurin_io import resolve_out_root
    out_root = resolve_out_root("runs")
    rv_dir = out_root / args.out_run / "RUN_VALID"
    rv_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(rv_dir / "verdict.json", {"verdict": "PASS", "created": utc_now_iso()})

    ctx = init_stage(args.out_run, STAGE, params={
        "source_runs": source_runs, "min_coverage": args.min_coverage, "top_k": args.top_k,
        "catalog_path": args.catalog_path,
    })

    # Load optional catalog for deviation analysis
    catalog: dict[str, Any] | None = None
    if args.catalog_path:
        try:
            with open(args.catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            print(f"[s5] Loaded catalog with {len(catalog)} events from {args.catalog_path}",
                  flush=True)
        except Exception as exc:
            print(f"[s5] WARNING: could not load catalog {args.catalog_path}: {exc}",
                  file=sys.stderr, flush=True)
    else:
        # Try to use built-in GWTC_EVENTS as default catalog
        try:
            from mvp.gwtc_events import GWTC_EVENTS
            catalog = GWTC_EVENTS
            print(f"[s5] Using built-in GWTC catalog ({len(catalog)} events)", flush=True)
        except Exception:
            pass

    # Collect and validate source files.
    source_paths: dict[str, Path] = {}
    ranked_paths: dict[str, Path] = {}
    for src in source_runs:
        p = out_root / src / "s4_geometry_filter" / "outputs" / "compatible_set.json"
        source_paths[src] = p
        ranked_paths[src] = out_root / src / "s6b_information_geometry_ranked" / "outputs" / "ranked_geometries.json"
    check_inputs(ctx, source_paths, optional=ranked_paths)
    # Keep metadata portable/auditable: prefer paths relative to runs_root for upstream inputs.
    for rec in ctx.inputs_record:
        label = rec.get("label", "")
        src_path = source_paths.get(label)
        if src_path is not None:
            rec["path"] = _relpath_under(out_root, src_path)

    try:
        source_data: list[dict[str, Any]] = []
        for src, p in source_paths.items():
            with open(p, "r", encoding="utf-8") as f:
                cs = json.load(f)

            extracted_ids = _extract_compatible_geometry_ids(cs, p)
            if os.environ.get("BASURIN_DEBUG_S5"):
                print(
                    f"[s5 debug] run={src} payload_keys={sorted(cs.keys())} extracted_ids={sorted(extracted_ids)}",
                    file=sys.stderr,
                )

            ranked_all = cs.get("ranked_all", [])
            if not ranked_all:
                ranked_all = [{"geometry_id": gid} for gid in sorted(extracted_ids)]

            observables = cs.get("observables", {})

            # Extract uncertainties from the estimates file if available
            sigma_f_hz = None
            sigma_Q = None
            estimates_path_candidates = [
                out_root / src / "s3_ringdown_estimates" / "outputs" / "estimates.json",
                out_root / src / "s3_spectral_estimates" / "outputs" / "spectral_estimates.json",
            ]
            for est_path in estimates_path_candidates:
                if est_path.exists():
                    try:
                        with open(est_path, "r", encoding="utf-8") as ef:
                            est_data = json.load(ef)
                        unc = est_data.get("combined_uncertainty", {})
                        sigma_f_hz = unc.get("sigma_f_hz")
                        sigma_Q = unc.get("sigma_Q")
                        # Also get observables from here if not in compatible_set
                        if not observables.get("f_hz") and est_data.get("combined"):
                            combined = est_data["combined"]
                            observables = {
                                "f_hz": combined.get("f_hz"),
                                "Q": combined.get("Q"),
                            }
                        break
                    except Exception:
                        pass

            multimode_payload: dict[str, Any] = {}
            s4c_payload: dict[str, Any] = {}
            ringdown_snr = None
            for extra_path in [
                out_root / src / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json",
                out_root / src / "s4c_kerr_consistency" / "outputs" / "kerr_consistency.json",
                out_root / src / "s3_ringdown_estimates" / "outputs" / "estimates.json",
            ]:
                if not extra_path.exists():
                    continue
                try:
                    with open(extra_path, "r", encoding="utf-8") as ef:
                        payload = json.load(ef)
                    if extra_path.name == "multimode_estimates.json" and isinstance(payload, dict):
                        multimode_payload = payload
                    elif extra_path.name == "kerr_consistency.json" and isinstance(payload, dict):
                        s4c_payload = payload
                    elif extra_path.name == "estimates.json" and isinstance(payload, dict):
                        ringdown_snr = _first_number(payload.get("combined", {}) if isinstance(payload.get("combined"), dict) else {}, ["snr_peak"])
                except Exception:
                    continue

            source_data.append({
                "run_id": src, "event_id": cs.get("event_id", "unknown"),
                "metric": cs.get("metric"),
                "threshold_d2": cs.get("threshold_d2"),
                "ranked_all": ranked_all,
                "compatible_ids": extracted_ids,
                "n_atlas": cs.get("n_atlas", len(ranked_all)),
                "observables": observables,
                "sigma_f_hz": sigma_f_hz,
                "sigma_Q": sigma_Q,
                "multimode": multimode_payload,
                "s4c": s4c_payload,
                "ringdown_snr": ringdown_snr,
                "s6b_present": False,
                "ranked_indices": [],
                "compatible_indices": [],
            })

            ranked_path = ranked_paths[src]
            if ranked_path.exists():
                try:
                    ranked_payload = json.loads(ranked_path.read_text(encoding="utf-8"))
                    source_data[-1]["s6b_present"] = True
                    source_data[-1]["ranked_indices"] = _parse_s6b_indices(ranked_payload.get("ranked"))
                    source_data[-1]["compatible_indices"] = _parse_s6b_indices(ranked_payload.get("compatible"))
                except Exception:
                    pass

        result = aggregate_compatible_sets(source_data, args.min_coverage, top_k=args.top_k)

        # Compute deviation distribution if catalog available
        deviation_analysis = compute_deviation_distribution(source_data, catalog)
        if deviation_analysis is not None:
            if result.get("n_common_compatible", 0) == 0:
                warnings_list = deviation_analysis.setdefault("warnings", [])
                if "NO_COMMON_COMPATIBLE_GEOMETRIES" not in warnings_list:
                    warnings_list.append("NO_COMMON_COMPATIBLE_GEOMETRIES")
                deviation_analysis["interpretation"] = "INSUFFICIENT_SUPPORT"
                combined = deviation_analysis.get("combined")
                if isinstance(combined, dict):
                    combined["p_value_GR"] = None
                    combined["consistent_GR_95"] = None
            result["deviation_analysis"] = deviation_analysis
            if deviation_analysis.get("combined"):
                comb = deviation_analysis["combined"]
                p_value = comb.get("p_value_GR")
                p_text = "null" if p_value is None else f"{p_value:.3f}"
                print(
                    f"[s5] GR test: δf_rel={comb['delta_f_rel']:.4f}±{comb['sigma_delta_f_rel']:.4f}, "
                    f"χ²={comb['chi2_GR']:.2f}, p={p_text}, "
                    f"consistent_GR_95={comb['consistent_GR_95']}",
                    flush=True,
                )

        agg_path = ctx.outputs_dir / "aggregate.json"
        write_json_atomic(agg_path, result)

        scalar_missing = sum(
            1 for ev in result.get("events", [])
            if isinstance(ev, dict)
            and isinstance(ev.get("scalar"), dict)
            and ev["scalar"].get("kerr_tension") is None
        )
        summary_results: dict[str, Any] = {
            "n_events": result["n_events"],
            "n_common": result["n_common_geometries"],
            "n_unique": result["n_total_unique_geometries"],
            "diagnostics": {
                "missing_scalar_kerr_tension": scalar_missing,
            },
        }
        if deviation_analysis and deviation_analysis.get("combined"):
            comb = deviation_analysis["combined"]
            summary_results["delta_f_rel"] = comb["delta_f_rel"]
            summary_results["p_value_GR"] = comb["p_value_GR"]
            summary_results["consistent_GR_95"] = comb["consistent_GR_95"]

        finalize(ctx, artifacts={"aggregate": agg_path}, results=summary_results)
        print(f"OUT_ROOT={ctx.out_root}")
        print(f"STAGE_DIR={ctx.stage_dir}")
        print(f"OUTPUTS_DIR={ctx.outputs_dir}")
        print(f"STAGE_SUMMARY={ctx.stage_dir / 'stage_summary.json'}")
        print(f"MANIFEST={ctx.stage_dir / 'manifest.json'}")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
