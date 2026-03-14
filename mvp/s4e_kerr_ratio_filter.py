#!/usr/bin/env python3
"""Kerr-specific multimode ratio filter for the canonical pipeline."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
for _cand in (_here.parents[0], _here.parents[1]):
    if (_cand / "basurin_io.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from basurin_io import sha256_file, utc_now_iso, write_json_atomic
from mvp.contracts import abort, check_inputs, finalize, init_stage, log_stage_paths
from mvp.kerr_qnm_fits import kerr_ratio_curve

STAGE = "s4e_kerr_ratio_filter"
OUTPUT_FILE = "ratio_filter_result.json"
MODE220 = "220"
MODE221 = "221"
SPIN_RE = re.compile(r"(?:^|_)(?:a|chi)(-?\d+(?:\.\d+)?)$")
EXTREME_RATIO_WARNING = "EXTREME_RATIO"
KERR_RF_INCONSISTENT = "KERR_Rf_INCONSISTENT"
RF_UNINFORMATIVE = "Rf_UNINFORMATIVE"
NO_EXCLUSION_POWER = "NO_EXCLUSION_POWER"
NO_SPIN_IN_ATLAS = "NO_SPIN_IN_ATLAS"
NO_INPUT_GEOMETRIES = "NO_INPUT_GEOMETRIES"


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _extract_quantile_block(mode_obj: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    direct = mode_obj.get("estimates")
    if isinstance(direct, dict):
        f_hz = direct.get("f_hz")
        tau_s = direct.get("tau_s")
        q_factor = direct.get("Q")
        if isinstance(f_hz, dict) and isinstance(tau_s, dict):
            f_vals = {q: _to_float(f_hz.get(q)) for q in ("p10", "p50", "p90")}
            tau_vals = {q: _to_float(tau_s.get(q)) for q in ("p10", "p50", "p90")}
            q_vals = {q: _to_float(q_factor.get(q)) for q in ("p10", "p50", "p90")} if isinstance(q_factor, dict) else None
            if all(f_vals[q] is not None and tau_vals[q] is not None for q in ("p10", "p50", "p90")):
                if q_vals is None or any(q_vals[q] is None for q in ("p10", "p50", "p90")):
                    q_vals = {
                        q: float(math.pi * float(f_vals[q]) * float(tau_vals[q]))
                        for q in ("p10", "p50", "p90")
                    }
                return {
                    "f_hz": f_vals,  # type: ignore[return-value]
                    "tau_s": tau_vals,  # type: ignore[return-value]
                    "Q": q_vals,
                }

    stability = (((mode_obj.get("fit") or {}).get("stability") or {}))
    if isinstance(stability, dict):
        lnf = {q: _to_float(stability.get(f"lnf_{q}")) for q in ("p10", "p50", "p90")}
        lnq = {q: _to_float(stability.get(f"lnQ_{q}")) for q in ("p10", "p50", "p90")}
        if all(lnf[q] is not None and lnq[q] is not None for q in ("p10", "p50", "p90")):
            f_vals = {q: float(math.exp(float(lnf[q]))) for q in ("p10", "p50", "p90")}
            q_vals = {q: float(math.exp(float(lnq[q]))) for q in ("p10", "p50", "p90")}
            tau_vals = {
                q: float(q_vals[q] / (math.pi * f_vals[q]))
                for q in ("p10", "p50", "p90")
            }
            return {"f_hz": f_vals, "tau_s": tau_vals, "Q": q_vals}

    ln_f = _to_float(mode_obj.get("ln_f"))
    ln_q = _to_float(mode_obj.get("ln_Q"))
    if ln_f is not None and ln_q is not None:
        f = float(math.exp(ln_f))
        q_factor = float(math.exp(ln_q))
        tau = float(q_factor / (math.pi * f))
        one_f = {"p10": f, "p50": f, "p90": f}
        one_tau = {"p10": tau, "p50": tau, "p90": tau}
        one_q = {"p10": q_factor, "p50": q_factor, "p90": q_factor}
        return {"f_hz": one_f, "tau_s": one_tau, "Q": one_q}

    return None


def _extract_mode_quantiles(multimode: dict[str, Any], label: str) -> dict[str, dict[str, float]]:
    estimates = multimode.get("estimates")
    if isinstance(estimates, dict):
        per_mode = estimates.get("per_mode")
        if isinstance(per_mode, dict):
            node = per_mode.get(label)
            if isinstance(node, dict):
                vals = _extract_quantile_block(node)
                if vals is not None:
                    return vals

    modes = multimode.get("modes")
    if isinstance(modes, list):
        for node in modes:
            if not isinstance(node, dict):
                continue
            if str(node.get("label")) != label:
                continue
            vals = _extract_quantile_block(node)
            if vals is not None:
                return vals

    raise ValueError(
        f"Mode ({label[0]},{label[1]},{label[2]}) not found in multimode_estimates.json. "
        f"s4e requires multimode estimates from s3b."
    )


def _summarize_mode(multimode: dict[str, Any], label: str) -> dict[str, Any]:
    quantiles = _extract_mode_quantiles(multimode, label)

    def _require(block: str, q: str) -> float:
        value = _to_float((quantiles.get(block) or {}).get(q))
        if value is None or value <= 0.0:
            raise ValueError(f"Invalid {block}.{q} for mode {label}")
        return value

    f_p10 = _require("f_hz", "p10")
    f_p50 = _require("f_hz", "p50")
    f_p90 = _require("f_hz", "p90")
    q_p10 = _require("Q", "p10")
    q_p50 = _require("Q", "p50")
    q_p90 = _require("Q", "p90")
    tau_p10 = _require("tau_s", "p10")
    tau_p50 = _require("tau_s", "p50")
    tau_p90 = _require("tau_s", "p90")

    return {
        "f_hz": f_p50,
        "sigma_f_hz": max((f_p90 - f_p10) / 2.0, 0.0),
        "Q": q_p50,
        "sigma_Q": max((q_p90 - q_p10) / 2.0, 0.0),
        "tau_s": tau_p50,
        "sigma_tau_s": max((tau_p90 - tau_p10) / 2.0, 0.0),
        "quantiles": quantiles,
    }


def _ratio_interval(central: float, sigma: float | None, sigma_scale: float) -> tuple[float, float]:
    if sigma is None or sigma <= 0.0:
        return central, central
    return central - (sigma_scale * sigma), central + (sigma_scale * sigma)


def _compute_ratio(
    numerator: float,
    sigma_num: float,
    denominator: float,
    sigma_den: float,
    sigma_scale: float,
) -> dict[str, Any]:
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("Observed ratio requires finite positive numerator and denominator")
    central = numerator / denominator
    sigma = central * math.sqrt(
        ((sigma_num / numerator) ** 2 if sigma_num > 0.0 else 0.0) +
        ((sigma_den / denominator) ** 2 if sigma_den > 0.0 else 0.0)
    )
    lo, hi = _ratio_interval(central, sigma, sigma_scale)
    return {
        "central": central,
        "lo": lo,
        "hi": hi,
        "sigma": sigma,
        "n_samples": None,
    }


def compute_observed_ratios(
    multimode: dict[str, Any],
    *,
    sigma_rf: float,
    sigma_rq: float,
) -> dict[str, Any]:
    mode220 = _summarize_mode(multimode, MODE220)
    mode221 = _summarize_mode(multimode, MODE221)
    rf = _compute_ratio(
        numerator=float(mode221["f_hz"]),
        sigma_num=float(mode221["sigma_f_hz"]),
        denominator=float(mode220["f_hz"]),
        sigma_den=float(mode220["sigma_f_hz"]),
        sigma_scale=sigma_rf,
    )
    rq = _compute_ratio(
        numerator=float(mode221["Q"]),
        sigma_num=float(mode221["sigma_Q"]),
        denominator=float(mode220["Q"]),
        sigma_den=float(mode220["sigma_Q"]),
        sigma_scale=sigma_rq,
    )
    return {
        "estimation_method": "delta_method",
        "mode_220": mode220,
        "mode_221": mode221,
        "Rf": rf,
        "RQ": rq,
    }


def _overlap_metrics(lo: float, hi: float, ref_lo: float, ref_hi: float) -> tuple[float, float, float]:
    width = max(hi - lo, 0.0)
    ref_width = max(ref_hi - ref_lo, 0.0)
    overlap = max(0.0, min(hi, ref_hi) - max(lo, ref_lo))
    overlap_frac = overlap / width if width > 0.0 else (1.0 if ref_lo <= lo <= ref_hi else 0.0)
    informativity = 1.0 - min(1.0, (width / ref_width)) if ref_width > 0.0 else 0.0
    return overlap, overlap_frac, informativity


def classify_informativity(informativity: float) -> str:
    if informativity >= 0.5:
        return "HIGH"
    if informativity >= 0.3:
        return "MODERATE"
    if informativity >= 0.1:
        return "LOW"
    return "UNINFORMATIVE"


def _geometry_index_from_ranked(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        gid = row.get("geometry_id")
        if isinstance(gid, str) and gid:
            out[gid] = dict(row)
    return out


def extract_compatible_geometries(
    compatible_payload: dict[str, Any],
    ranked_all_full: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ranked_index = _geometry_index_from_ranked(ranked_all_full or [])
    compatible = compatible_payload.get("compatible_geometries")
    if isinstance(compatible, list):
        out: list[dict[str, Any]] = []
        for row in compatible:
            if not isinstance(row, dict):
                continue
            gid = row.get("geometry_id")
            base = dict(ranked_index.get(str(gid), {}))
            base.update(row)
            out.append(base)
        return out

    compatible_ids = compatible_payload.get("compatible_geometry_ids")
    if isinstance(compatible_ids, list):
        out = []
        for raw in compatible_ids:
            gid = str(raw)
            base = dict(ranked_index.get(gid, {}))
            base.setdefault("geometry_id", gid)
            out.append(base)
        return out
    return []


def extract_geometry_spin(row: dict[str, Any]) -> float | None:
    candidates = [row]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata)
    for source in candidates:
        for key in ("spin", "chi", "a_over_m", "J_over_M2"):
            value = _to_float(source.get(key))
            if value is not None and 0.0 <= value <= 1.0:
                return value
    gid = row.get("geometry_id")
    if isinstance(gid, str):
        match = SPIN_RE.search(gid)
        if match:
            value = _to_float(match.group(1))
            if value is not None and 0.0 <= value <= 1.0:
                return value
    return None


def _kerr_prediction_for_spin(spin: float) -> tuple[float, float]:
    curve = kerr_ratio_curve(chi_grid=[spin])
    rf_grid = curve.get("Rf_grid")
    rq_grid = curve.get("RQ_grid")
    if not isinstance(rf_grid, list) or not rf_grid or not isinstance(rq_grid, list) or not rq_grid:
        raise ValueError(f"Cannot evaluate Kerr ratio at chi={spin}")
    return float(rf_grid[0]), float(rq_grid[0])


def _tension(predicted: float, lo: float, hi: float, sigma: float | None) -> float | None:
    if lo <= predicted <= hi:
        return 0.0
    if sigma is None or sigma <= 0.0:
        return None
    if predicted < lo:
        return (lo - predicted) / sigma
    return (predicted - hi) / sigma


def _filter_geometry(
    row: dict[str, Any],
    observed_ratios: dict[str, Any],
    *,
    apply_rq: bool,
) -> dict[str, Any]:
    spin = extract_geometry_spin(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if spin is None:
        return {
            "geometry_id": row.get("geometry_id"),
            "spin": None,
            "Rf_predicted": None,
            "RQ_predicted": None,
            "tension_Rf": None,
            "tension_RQ": None,
            "status": "RATIO_NOT_APPLICABLE",
            "metadata": metadata,
        }

    rf_pred, rq_pred = _kerr_prediction_for_spin(spin)
    rf_obs = observed_ratios["Rf"]
    rq_obs = observed_ratios["RQ"]
    rf_ok = float(rf_obs["lo"]) <= rf_pred <= float(rf_obs["hi"])
    rq_ok = float(rq_obs["lo"]) <= rq_pred <= float(rq_obs["hi"])
    ratio_ok = rf_ok and rq_ok if apply_rq else rf_ok

    payload = {
        "geometry_id": row.get("geometry_id"),
        "spin": spin,
        "Rf_predicted": rf_pred,
        "RQ_predicted": rq_pred,
        "tension_Rf": _tension(rf_pred, float(rf_obs["lo"]), float(rf_obs["hi"]), _to_float(rf_obs.get("sigma"))),
        "tension_RQ": _tension(rq_pred, float(rq_obs["lo"]), float(rq_obs["hi"]), _to_float(rq_obs.get("sigma"))),
        "status": "RATIO_COMPATIBLE" if ratio_ok else "RATIO_EXCLUDED",
        "metadata": metadata,
    }
    for key in ("distance", "d2", "f_hz", "Q"):
        if key in row:
            payload[key] = row[key]
    return payload


def run_ratio_filter(
    *,
    run_id: str,
    multimode_estimates: dict[str, Any],
    compatible_payload: dict[str, Any],
    ranked_all_full: list[dict[str, Any]] | None,
    estimates_sha256: str,
    compatible_set_sha256: str,
    sigma_rf: float,
    sigma_rq: float,
    chi_grid_points: int,
    apply_rq: bool,
) -> dict[str, Any]:
    observed_ratios = compute_observed_ratios(
        multimode_estimates,
        sigma_rf=sigma_rf,
        sigma_rq=sigma_rq,
    )
    curve = kerr_ratio_curve(n_points=chi_grid_points)
    rf_range = curve["Rf_range"]
    rq_range = curve["RQ_range"]

    rf_obs = observed_ratios["Rf"]
    rq_obs = observed_ratios["RQ"]
    rf_overlap, rf_overlap_frac, rf_informativity = _overlap_metrics(
        float(rf_obs["lo"]),
        float(rf_obs["hi"]),
        float(rf_range["min"]),
        float(rf_range["max"]),
    )
    rq_overlap, rq_overlap_frac, rq_informativity = _overlap_metrics(
        float(rq_obs["lo"]),
        float(rq_obs["hi"]),
        float(rq_range["min"]),
        float(rq_range["max"]),
    )

    geometries = extract_compatible_geometries(compatible_payload, ranked_all_full=ranked_all_full)
    filtered = [
        _filter_geometry(row, observed_ratios, apply_rq=apply_rq)
        for row in geometries
    ]
    compatible_after_ratio = [row for row in filtered if row["status"] != "RATIO_EXCLUDED"]
    excluded_after_ratio = [row for row in filtered if row["status"] == "RATIO_EXCLUDED"]
    not_applicable = [row for row in filtered if row["status"] == "RATIO_NOT_APPLICABLE"]
    ratio_compatible = [row for row in filtered if row["status"] == "RATIO_COMPATIBLE"]

    spins_before = sorted(row["spin"] for row in filtered if isinstance(row.get("spin"), float))
    spins_after = sorted(row["spin"] for row in ratio_compatible if isinstance(row.get("spin"), float))
    chi_before = [spins_before[0], spins_before[-1]] if spins_before else None
    chi_after = [spins_after[0], spins_after[-1]] if spins_after else None
    spin_reduction = None
    if chi_before is not None and chi_after is not None:
        before_width = float(chi_before[1] - chi_before[0])
        after_width = float(chi_after[1] - chi_after[0])
        if before_width > 0.0:
            spin_reduction = 1.0 - min(1.0, after_width / before_width)

    diagnostics_warnings: list[str] = []
    if not (0.5 <= float(rf_obs["central"]) <= 1.5):
        diagnostics_warnings.append(EXTREME_RATIO_WARNING)
    if rf_overlap <= 0.0:
        diagnostics_warnings.append(KERR_RF_INCONSISTENT)
    if classify_informativity(rf_informativity) == "UNINFORMATIVE":
        diagnostics_warnings.append(RF_UNINFORMATIVE)
    if excluded_after_ratio == [] and ratio_compatible:
        diagnostics_warnings.append(NO_EXCLUSION_POWER)
    if not spins_before:
        diagnostics_warnings.append(NO_SPIN_IN_ATLAS)
    if not geometries:
        diagnostics_warnings.append(NO_INPUT_GEOMETRIES)

    return {
        "schema_version": "s4e_ratio_filter_v1",
        "run_id": run_id,
        "created": utc_now_iso(),
        "inputs": {
            "estimates_sha256": estimates_sha256,
            "compatible_set_sha256": compatible_set_sha256,
            "estimation_method": observed_ratios["estimation_method"],
        },
        "parameters": {
            "sigma_Rf": sigma_rf,
            "sigma_RQ": sigma_rq,
            "chi_grid_points": chi_grid_points,
            "apply_RQ": apply_rq,
        },
        "observed_ratios": {
            "Rf": observed_ratios["Rf"],
            "RQ": observed_ratios["RQ"],
        },
        "kerr_reference": {
            "Rf_range": rf_range,
            "RQ_range": rq_range,
            "chi_grid_points": chi_grid_points,
        },
        "kerr_consistency": {
            "Rf_consistent": rf_overlap > 0.0,
            "RQ_consistent": rq_overlap > 0.0 if apply_rq else None,
            "overlap_Rf": rf_overlap,
            "overlap_frac_obs_Rf": rf_overlap_frac,
            "informativity_Rf": rf_informativity,
            "overlap_RQ": rq_overlap if apply_rq else None,
            "overlap_frac_obs_RQ": rq_overlap_frac if apply_rq else None,
            "informativity_RQ": rq_informativity if apply_rq else None,
        },
        "filtering": {
            "n_input_geometries": len(filtered),
            "n_ratio_compatible": len(ratio_compatible),
            "n_ratio_excluded": len(excluded_after_ratio),
            "n_ratio_not_applicable": len(not_applicable),
            "reduction_fraction": (len(excluded_after_ratio) / len(filtered)) if filtered else 0.0,
            "compatible_geometries": compatible_after_ratio,
            "excluded_geometries": excluded_after_ratio,
        },
        "spin_constraints": {
            "chi_min_compatible": spins_after[0] if spins_after else None,
            "chi_max_compatible": spins_after[-1] if spins_after else None,
            "chi_range_before_ratio": chi_before,
            "chi_range_after_ratio": chi_after,
            "spin_range_reduction_fraction": spin_reduction,
        },
        "diagnostics": {
            "warning_codes": diagnostics_warnings,
            "is_informative": classify_informativity(rf_informativity) != "UNINFORMATIVE",
            "informativity_class": classify_informativity(rf_informativity),
        },
        "verdict": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"MVP {STAGE}: Kerr ratio filter for multimode runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sigma-Rf", type=float, default=2.0)
    parser.add_argument("--sigma-RQ", type=float, default=2.0)
    parser.add_argument("--chi-grid-points", type=int, default=200)
    parser.add_argument("--apply-RQ", action="store_true")
    args = parser.parse_args(argv)

    ctx = init_stage(
        args.run_id,
        STAGE,
        params={
            "sigma_Rf": args.sigma_Rf,
            "sigma_RQ": args.sigma_RQ,
            "chi_grid_points": args.chi_grid_points,
            "apply_RQ": args.apply_RQ,
        },
    )
    multimode_path = ctx.run_dir / "s3b_multimode_estimates" / "outputs" / "multimode_estimates.json"
    compatible_path = ctx.run_dir / "s4_geometry_filter" / "outputs" / "compatible_set.json"
    ranked_full_path = ctx.run_dir / "s4_geometry_filter" / "outputs" / "ranked_all_full.json"

    try:
        check_inputs(
            ctx,
            {
                "multimode_estimates": multimode_path,
                "compatible_set": compatible_path,
            },
            optional={"ranked_all_full": ranked_full_path},
        )
        multimode = _load_json_object(multimode_path)
        compatible_payload = _load_json_object(compatible_path)
        ranked_all_full = None
        if ranked_full_path.exists():
            payload = json.loads(ranked_full_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                ranked_all_full = [row for row in payload if isinstance(row, dict)]

        result = run_ratio_filter(
            run_id=args.run_id,
            multimode_estimates=multimode,
            compatible_payload=compatible_payload,
            ranked_all_full=ranked_all_full,
            estimates_sha256=sha256_file(multimode_path),
            compatible_set_sha256=sha256_file(compatible_path),
            sigma_rf=float(args.sigma_Rf),
            sigma_rq=float(args.sigma_RQ),
            chi_grid_points=max(int(args.chi_grid_points), 2),
            apply_rq=bool(args.apply_RQ),
        )

        out_path = ctx.outputs_dir / OUTPUT_FILE
        write_json_atomic(out_path, result)
        finalize(
            ctx,
            artifacts={"ratio_filter_result": out_path},
            verdict="PASS",
            results={
                "n_input_geometries": result["filtering"]["n_input_geometries"],
                "n_ratio_compatible": result["filtering"]["n_ratio_compatible"],
                "n_ratio_excluded": result["filtering"]["n_ratio_excluded"],
                "informativity_class": result["diagnostics"]["informativity_class"],
                "rf_consistent": result["kerr_consistency"]["Rf_consistent"],
            },
        )
        log_stage_paths(ctx)
        print(
            f"[{STAGE}] n_input={result['filtering']['n_input_geometries']} "
            f"n_ratio_compatible={result['filtering']['n_ratio_compatible']} "
            f"n_ratio_excluded={result['filtering']['n_ratio_excluded']}"
        )
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        abort(ctx, str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
