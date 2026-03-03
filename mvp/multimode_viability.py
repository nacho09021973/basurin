"""Utilidades puras para clasificación multimodo y canal de evidencia científica.

Este módulo implementa funciones deterministas y libres de IO para el flujo
multimodo (220+221) descrito en ``informe_multimode_viability_v3.md``.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_THRESHOLDS = {
    "MIN_VALID_FRAC_220": 0.50,
    "MIN_VALID_FRAC_221": 0.30,
    "MAX_REL_IQR_F220": 0.50,
    "MAX_REL_IQR_F221": 0.60,
    "MAX_SPIN_FLOOR_FRAC": 0.30,
    "INFORMATIVE_THRESHOLD": 0.30,
    "DELTA_BIC_SUPPORTIVE": 2.0,
    "STABILITY_221_MAX": 0.60,
    "SEVERE_COUNT_LIMIT": 2,
    "T_T0_STD_MAX": 5.0,
    "T_CHI_PSD_MAX": 0.10,
    "T_Q221_MIN": 1.5,
}


def classify_multimode_viability(
    inputs: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Responsabilidad única: clasificar viabilidad multimodo sin emitir claims físicos."""

    t = _merged_thresholds(thresholds)
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    vf_220 = _finite_or_none(inputs.get("valid_fraction_220"))
    metrics["valid_fraction_220"] = vf_220
    if vf_220 is None:
        reasons.append("valid_fraction_220 missing or non-finite")
        return _viability_result("RINGDOWN_NONINFORMATIVE", reasons, metrics, t)
    if vf_220 < t["MIN_VALID_FRAC_220"]:
        reasons.append(
            f"valid_fraction_220={vf_220:.3f} < {t['MIN_VALID_FRAC_220']}: "
            "fundamental mode posterior unreliable"
        )
        return _viability_result("RINGDOWN_NONINFORMATIVE", reasons, metrics, t)

    rel_iqr_f220 = _safe_ratio(
        _finite_or_none(inputs.get("f_220_iqr")),
        _finite_or_none(inputs.get("f_220_median")),
    )
    metrics["rel_iqr_f220"] = rel_iqr_f220
    if rel_iqr_f220 is None:
        reasons.append("rel_iqr_f220 not computable (missing/invalid f_220 inputs)")
        return _viability_result("RINGDOWN_NONINFORMATIVE", reasons, metrics, t)
    if rel_iqr_f220 > t["MAX_REL_IQR_F220"]:
        reasons.append(
            f"rel_iqr_f220={rel_iqr_f220:.3f} > {t['MAX_REL_IQR_F220']}: "
            "fundamental frequency poorly constrained"
        )
        return _viability_result("RINGDOWN_NONINFORMATIVE", reasons, metrics, t)

    vf_221 = _finite_or_none(inputs.get("valid_fraction_221"))
    metrics["valid_fraction_221"] = vf_221
    if vf_221 is None:
        reasons.append("valid_fraction_221 missing or non-finite")
        return _viability_result("SINGLEMODE_ONLY", reasons, metrics, t)
    if vf_221 < t["MIN_VALID_FRAC_221"]:
        reasons.append(
            f"valid_fraction_221={vf_221:.3f} < {t['MIN_VALID_FRAC_221']}: "
            "overtone posterior insufficient"
        )
        return _viability_result("SINGLEMODE_ONLY", reasons, metrics, t)

    severe: list[str] = []

    sf = _finite_or_none(inputs.get("spin_at_floor_frac_221"))
    metrics["spin_at_floor_frac_221"] = sf
    if sf is not None and sf > t["MAX_SPIN_FLOOR_FRAC"]:
        severe.append("SPIN_AT_PHYSICAL_FLOOR")
        reasons.append(
            f"spin_at_floor_frac_221={sf:.3f} > {t['MAX_SPIN_FLOOR_FRAC']}"
        )

    dbic = _finite_or_none(inputs.get("delta_bic"))
    metrics["delta_bic"] = dbic
    if dbic is not None and dbic < t["DELTA_BIC_SUPPORTIVE"]:
        severe.append("DELTA_BIC_UNSUPPORTIVE")
        reasons.append(
            f"delta_bic={dbic:.2f} < {t['DELTA_BIC_SUPPORTIVE']}: "
            "data do not prefer two-mode model"
        )

    stability_221 = _safe_ratio(
        _finite_or_none(inputs.get("f_221_iqr")),
        _finite_or_none(inputs.get("f_221_median")),
    )
    metrics["stability_221"] = stability_221
    if stability_221 is not None and stability_221 > t["STABILITY_221_MAX"]:
        severe.append("UNSTABLE_221_POSTERIOR")
        reasons.append(f"stability_221={stability_221:.3f} > {t['STABILITY_221_MAX']}")

    rf_q = inputs.get("Rf_bootstrap_quantiles")
    rf_kerr = _as_float_pair(inputs.get("Rf_kerr_band"))
    informativity = None

    if isinstance(rf_q, dict) and rf_kerr is not None:
        rf_lo = _finite_or_none(rf_q.get("q05"))
        rf_hi = _finite_or_none(rf_q.get("q95"))
        if rf_lo is None or rf_hi is None or rf_hi < rf_lo:
            severe.append("RF_NOT_COMPUTABLE")
            reasons.append("Rf quantiles invalid")
            metrics["informativity_Rf"] = None
            metrics["kerr_consistent"] = None
        else:
            d_rf_obs = rf_hi - rf_lo
            d_rf_kerr = rf_kerr[1] - rf_kerr[0]
            if d_rf_kerr <= 0.0:
                informativity = None
                reasons.append("delta_Rf_kerr <= 0; informativity not computable")
            else:
                informativity = 1.0 - min(1.0, d_rf_obs / d_rf_kerr)
            metrics["informativity_Rf"] = informativity
            metrics["delta_Rf_obs"] = d_rf_obs
            metrics["delta_Rf_kerr"] = d_rf_kerr

            overlap_lo = max(rf_lo, rf_kerr[0])
            overlap_hi = min(rf_hi, rf_kerr[1])
            overlap = max(0.0, overlap_hi - overlap_lo)
            metrics["overlap_with_kerr"] = overlap
            metrics["overlap_frac_obs"] = _safe_ratio(overlap, d_rf_obs)
            metrics["overlap_frac_kerr"] = _safe_ratio(overlap, d_rf_kerr)
            metrics["kerr_consistent"] = overlap > 0.0

            if informativity is not None and informativity < t["INFORMATIVE_THRESHOLD"]:
                severe.append("UNINFORMATIVE_RF")
                reasons.append(
                    f"informativity_Rf={informativity:.3f} < {t['INFORMATIVE_THRESHOLD']}"
                )
    else:
        metrics["informativity_Rf"] = None
        metrics["kerr_consistent"] = None
        severe.append("RF_NOT_COMPUTABLE")
        reasons.append("Rf_obs or Rf_kerr_band not available")

    metrics["n_severe_flags"] = len(severe)
    metrics["severe_flags"] = list(severe)

    if len(severe) >= int(t["SEVERE_COUNT_LIMIT"]):
        reasons.append(
            f"n_severe={len(severe)} >= {int(t['SEVERE_COUNT_LIMIT'])}: "
            "degraded to SINGLEMODE_ONLY"
        )
        return _viability_result("SINGLEMODE_ONLY", reasons, metrics, t)

    if metrics.get("kerr_consistent") is False:
        reasons.append(
            "Rf_obs outside Kerr band: potential inconsistency "
            "(verify systematics before interpreting)"
        )

    reasons.append("overtone passes minimum viability checks")
    return _viability_result("MULTIMODE_OK", reasons, metrics, t)


def evaluate_systematics_gate(
    inputs: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Responsabilidad única: evaluar checks de sistemáticas con veredicto automático."""

    t = _merged_thresholds(thresholds)
    checks: dict[str, Any] = {}

    t0 = inputs.get("t0_plateau")
    if isinstance(t0, dict):
        plateau = t0.get("plateau_detected") is True
        t0_std = _finite_or_none(t0.get("f_std_over_plateau_hz"))
        if t0_std is None:
            checks["t0_plateau"] = {"verdict": "NA", "metric": None, "threshold": t["T_T0_STD_MAX"]}
        else:
            t0_pass = plateau and t0_std < t["T_T0_STD_MAX"]
            checks["t0_plateau"] = {
                "verdict": "PASS" if t0_pass else "FAIL",
                "metric": t0_std,
                "threshold": t["T_T0_STD_MAX"],
            }
    else:
        checks["t0_plateau"] = {"verdict": "NA", "metric": None, "threshold": None}

    psd = _finite_or_none(inputs.get("chi_psd_at_f221"))
    if psd is None:
        checks["psd_sanity"] = {"verdict": "NA", "metric": None, "threshold": None}
    else:
        psd_pass = psd < t["T_CHI_PSD_MAX"]
        checks["psd_sanity"] = {
            "verdict": "PASS" if psd_pass else "FAIL",
            "metric": psd,
            "threshold": t["T_CHI_PSD_MAX"],
        }

    q221 = _finite_or_none(inputs.get("Q_221_median"))
    if q221 is None:
        checks["estimator_resolution"] = {"verdict": "NA", "metric": None, "threshold": None}
    else:
        res_pass = q221 > t["T_Q221_MIN"]
        checks["estimator_resolution"] = {
            "verdict": "PASS" if res_pass else "FAIL",
            "metric": q221,
            "threshold": t["T_Q221_MIN"],
        }

    verdicts = {item["verdict"] for item in checks.values()}
    if "FAIL" in verdicts:
        verdict_auto = "FAIL"
    elif verdicts == {"NA"}:
        verdict_auto = "NOT_AVAILABLE"
    elif "PASS" in verdicts:
        verdict_auto = "PASS"
    else:
        verdict_auto = "NOT_AVAILABLE"

    return {
        "schema_version": "systematics_gate_v1",
        "verdict_auto": verdict_auto,
        "checks": checks,
        "thresholds_used": {
            "T_T0_STD_MAX": t["T_T0_STD_MAX"],
            "T_CHI_PSD_MAX": t["T_CHI_PSD_MAX"],
            "T_Q221_MIN": t["T_Q221_MIN"],
        },
    }


def evaluate_science_evidence(
    viability: dict[str, Any],
    systematics: dict[str, Any],
    rf_bootstrap_quantiles: dict[str, Any] | None,
    rf_kerr_grid: list[float] | tuple[float, ...],
    chi_grid: list[float] | tuple[float, ...],
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Responsabilidad única: producir `science_evidence` respetando gating y override."""

    reason_if_skipped: list[str] = []

    if viability.get("class") != "MULTIMODE_OK":
        reason_if_skipped.append("MULTIMODE_GATE")

    verdict_auto = systematics.get("verdict_auto")
    verdict_human = override.get("verdict_human") if isinstance(override, dict) else None
    verdict_final = _combine_verdicts(verdict_auto, verdict_human)

    if verdict_final != "PASS":
        if verdict_final == "FAIL":
            reason_if_skipped.append("SYSTEMATICS_FAIL")
        else:
            reason_if_skipped.append("SYSTEMATICS_NOT_AVAILABLE")

    if reason_if_skipped:
        return _science_not_evaluated(reason_if_skipped)

    if not isinstance(rf_bootstrap_quantiles, dict):
        return _science_not_evaluated(["RF_NOT_COMPUTABLE"])

    rf_med = _finite_or_none(rf_bootstrap_quantiles.get("q50"))
    rf_lo = _finite_or_none(rf_bootstrap_quantiles.get("q05"))
    rf_hi = _finite_or_none(rf_bootstrap_quantiles.get("q95"))
    if rf_med is None or rf_lo is None or rf_hi is None or rf_hi < rf_lo:
        return _science_not_evaluated(["RF_NOT_COMPUTABLE"])

    if len(rf_kerr_grid) != len(chi_grid) or not rf_kerr_grid:
        return _science_not_evaluated(["RF_KERR_GRID_INVALID"])

    best_idx = None
    best_dist = None
    for idx, rf_k in enumerate(rf_kerr_grid):
        rf_k_f = _finite_or_none(rf_k)
        chi_f = _finite_or_none(chi_grid[idx])
        if rf_k_f is None or chi_f is None:
            continue
        dist = abs(rf_med - rf_k_f)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx is None:
        return _science_not_evaluated(["RF_KERR_GRID_INVALID"])

    chi_star = float(chi_grid[best_idx])
    rf_kerr_star = float(rf_kerr_grid[best_idx])
    delta_rf = rf_med - rf_kerr_star
    delta_rf_lo = rf_lo - rf_kerr_star
    delta_rf_hi = rf_hi - rf_kerr_star

    return {
        "schema_version": "science_evidence_v1",
        "status": "EVALUATED",
        "reason_if_skipped": [],
        "H1_min": {
            "delta_Rf": {
                "value": delta_rf,
                "interval": [delta_rf_lo, delta_rf_hi],
                "quantiles": [0.05, 0.95],
                "chi_star": chi_star,
                "Rf_kerr_at_chi_star": rf_kerr_star,
                "contains_zero": delta_rf_lo <= 0.0 <= delta_rf_hi,
                "definition": "Rf_obs_median - Rf_Kerr(chi_star)",
            }
        },
        "future_slots": {
            "delta_f_221": None,
            "delta_tau_221": None,
            "log_bayes_factor": None,
        },
    }


def _combine_verdicts(verdict_auto: Any, verdict_human: Any) -> str:
    """Responsabilidad única: combinar auto/humano con regla de override solo degradante."""

    if verdict_auto == "NOT_AVAILABLE":
        return "NOT_AVAILABLE"
    if verdict_auto == "FAIL":
        return "FAIL"
    if verdict_auto != "PASS":
        return "NOT_AVAILABLE"
    if verdict_human == "FAIL":
        return "FAIL"
    return "PASS"


def _viability_result(
    cls: str,
    reasons: list[str],
    metrics: dict[str, Any],
    thresholds_used: dict[str, float],
) -> dict[str, Any]:
    return {
        "class": cls,
        "reasons": list(reasons),
        "metrics": metrics,
        "thresholds_used": dict(thresholds_used),
        "schema_version": "multimode_viability_v1",
    }


def _science_not_evaluated(reason_if_skipped: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "science_evidence_v1",
        "status": "NOT_EVALUATED",
        "reason_if_skipped": reason_if_skipped,
        "H1_min": {"delta_Rf": None},
        "future_slots": {
            "delta_f_221": None,
            "delta_tau_221": None,
            "log_bayes_factor": None,
        },
    }


def _merged_thresholds(custom: dict[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_THRESHOLDS)
    if custom:
        for key, value in custom.items():
            if key in merged and _is_finite_number(value):
                merged[key] = float(value)
    return merged


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0:
        return None
    val = num / den
    return val if _is_finite_number(val) else None


def _finite_or_none(value: Any) -> float | None:
    if _is_finite_number(value):
        return float(value)
    return None


def _as_float_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = _finite_or_none(value[0])
    high = _finite_or_none(value[1])
    if low is None or high is None:
        return None
    return (low, high)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


# Example usage (sin IO)
def _example_usage() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    viability_inputs = {
        "valid_fraction_220": 0.8,
        "valid_fraction_221": 0.55,
        "f_220_median": 250.0,
        "f_220_iqr": 30.0,
        "f_221_median": 410.0,
        "f_221_iqr": 90.0,
        "Q_220_median": 8.0,
        "Q_220_iqr": 2.0,
        "Q_221_median": 2.3,
        "Q_221_iqr": 0.9,
        "Rf_bootstrap_quantiles": {"q05": 0.90, "q50": 0.94, "q95": 0.99},
        "Rf_kerr_band": [0.88, 1.00],
        "spin_at_floor_frac_221": 0.05,
        "delta_bic": 3.2,
        "two_mode_preferred": True,
    }
    viability = classify_multimode_viability(viability_inputs)

    systematics = evaluate_systematics_gate(
        {
            "t0_plateau": {
                "plateau_detected": True,
                "f_std_over_plateau_hz": 1.8,
            },
            "chi_psd_at_f221": 0.03,
            "Q_221_median": viability_inputs["Q_221_median"],
        }
    )

    science = evaluate_science_evidence(
        viability=viability,
        systematics=systematics,
        rf_bootstrap_quantiles=viability_inputs["Rf_bootstrap_quantiles"],
        rf_kerr_grid=[0.88, 0.91, 0.94, 0.97, 1.00],
        chi_grid=[0.0, 0.25, 0.5, 0.75, 0.95],
        override=None,
    )
    return viability, systematics, science
