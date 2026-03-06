"""Preflight viability: Fisher-based informativity prediction before pipeline execution.

Pure-function module (no IO) that predicts whether a given (t0, T, band) configuration
can produce an informative ringdown run, based on the Cramér-Rao bound and the SNR
captured in the analysis window.

The key insight: the multimode viability gate rejects when rel_iqr_f220 > 0.5.
The Cramér-Rao bound predicts rel_iqr ≈ 0.304 * alpha_safety / (Q * rho_eff).
So informativiy requires Q * rho_eff > 0.61 * alpha_safety.

This module provides:
  - eta(t0, T, tau): fraction of SNR² captured in window [t0, t0+T]
  - rho_eff(t0, T, tau, rho_total): effective SNR in window
  - t0_max(tau, Q, rho_total, alpha): max t0 for informative analysis
  - preflight_viability(): full viability assessment per mode
  - catalog_viability_table(): batch assessment for N events

Reference: docs/metodologia_informatividad_predictiva.md
"""
from __future__ import annotations

import math
from typing import Any

from mvp.kerr_qnm_fits import kerr_qnm, QNMResult


# ---------------------------------------------------------------------------
# Default thresholds — aligned with multimode_viability.py
# ---------------------------------------------------------------------------
DEFAULT_REL_IQR_THRESHOLD = 0.50       # MAX_REL_IQR_F220 from multimode_viability
DEFAULT_ALPHA_SAFETY = 2.5             # conservative; calibrate with runs
DEFAULT_T0_MIN_CONTAMINATION_S = 0.003 # 3 ms minimum to avoid merger contamination
CR_COEFFICIENT = 0.304                 # 1.35 / (pi * sqrt(2))
INFORMATIVE_Q_RHO_MIN = 0.61          # CR_COEFFICIENT / DEFAULT_REL_IQR_THRESHOLD

# Viability verdicts
VIABLE = "VIABLE"
MARGINAL = "MARGINAL"
INVIABLE = "INVIABLE"
DOMAIN_EMPTY = "DOMAIN_EMPTY"

# Marginal band: VIABLE if Q*rho_eff > alpha * 0.61,
#                MARGINAL if Q*rho_eff > 0.61 (but < alpha * 0.61)
#                INVIABLE if Q*rho_eff < 0.61


# ---------------------------------------------------------------------------
# Core analytic functions (pure math, no IO)
# ---------------------------------------------------------------------------

def snr_fraction_eta(t0_s: float, T_s: float, tau_s: float) -> float:
    """Fraction of total ringdown SNR² captured in window [t0, t0+T].

    η(t0, T) = e^{-2t0/τ} * (1 - e^{-2T/τ})

    Returns 0.0 for degenerate inputs.
    """
    if tau_s <= 0.0 or T_s <= 0.0 or t0_s < 0.0:
        return 0.0
    ratio_t0 = 2.0 * t0_s / tau_s
    ratio_T = 2.0 * T_s / tau_s
    # Guard against overflow in exp
    if ratio_t0 > 700.0:
        return 0.0
    return math.exp(-ratio_t0) * (1.0 - math.exp(-ratio_T))


def rho_effective(t0_s: float, T_s: float, tau_s: float, rho_total: float) -> float:
    """Effective SNR of ringdown mode in window [t0, t0+T].

    ρ_eff = ρ_total × √η(t0, T)
    """
    eta = snr_fraction_eta(t0_s, T_s, tau_s)
    return rho_total * math.sqrt(eta)


def rel_iqr_predicted(Q: float, rho_eff: float, alpha_safety: float = DEFAULT_ALPHA_SAFETY) -> float:
    """Predicted rel_iqr_f based on Cramér-Rao bound.

    rel_iqr ≈ 0.304 × alpha_safety / (Q × ρ_eff)

    Returns inf for degenerate inputs.
    """
    denom = Q * rho_eff
    if denom <= 0.0:
        return float("inf")
    return CR_COEFFICIENT * alpha_safety / denom


def t0_max_informative(
    tau_s: float,
    Q: float,
    rho_total: float,
    alpha_safety: float = DEFAULT_ALPHA_SAFETY,
    T_s: float | None = None,
) -> float:
    """Maximum t0 (seconds) for which the analysis can be informative.

    t0_max = (τ/2) × ln(Q² × ρ²_total × correction / (0.61 × α)²)

    With T correction: correction = (1 - e^{-2T/τ}); for T >> τ this ≈ 1.

    Returns 0.0 if the mode is intrinsically unconstrainable at this SNR.
    """
    threshold = INFORMATIVE_Q_RHO_MIN * alpha_safety
    numerator = (Q * rho_total) ** 2
    if T_s is not None and tau_s > 0.0:
        ratio_T = 2.0 * T_s / tau_s
        correction = 1.0 - math.exp(-ratio_T)
        numerator *= correction
    denominator = threshold ** 2
    if numerator <= denominator or tau_s <= 0.0:
        return 0.0
    return (tau_s / 2.0) * math.log(numerator / denominator)


def T_min_resolution(tau_s: float) -> float:
    """Minimum window duration for spectral resolution: T > 2πτ."""
    return 2.0 * math.pi * tau_s


def band_min_hz(f_hz: float, Q: float, n_gamma: float = 5.0) -> tuple[float, float]:
    """Minimum frequency band [f - n*Γ, f + n*Γ] to capture the Lorentzian.

    Γ = f / (2Q) is the half-width of the Lorentzian.
    """
    gamma = f_hz / (2.0 * Q) if Q > 0.0 else f_hz
    return (max(0.0, f_hz - n_gamma * gamma), f_hz + n_gamma * gamma)


# ---------------------------------------------------------------------------
# Viability assessment (per mode, per configuration)
# ---------------------------------------------------------------------------

def assess_mode_viability(
    *,
    f_hz: float,
    tau_s: float,
    Q: float,
    rho_total: float,
    t0_s: float,
    T_s: float,
    alpha_safety: float = DEFAULT_ALPHA_SAFETY,
    t0_min_s: float = DEFAULT_T0_MIN_CONTAMINATION_S,
    rel_iqr_threshold: float = DEFAULT_REL_IQR_THRESHOLD,
) -> dict[str, Any]:
    """Assess viability of a single mode at a given configuration.

    Returns a dict with all diagnostic fields needed for preflight_viability.json.
    """
    eta = snr_fraction_eta(t0_s, T_s, tau_s)
    rho_eff = rho_total * math.sqrt(eta)
    q_rho = Q * rho_eff
    threshold_qrho = INFORMATIVE_Q_RHO_MIN * alpha_safety
    iqr_pred = rel_iqr_predicted(Q, rho_eff, alpha_safety)
    t0_max = t0_max_informative(tau_s, Q, rho_total, alpha_safety, T_s)
    T_min = T_min_resolution(tau_s)
    band = band_min_hz(f_hz, Q)

    # Check constraints
    resolution_ok = T_s >= T_min
    contamination_ok = t0_s >= t0_min_s
    informative_ok = iqr_pred < rel_iqr_threshold

    # Viability verdict
    if not resolution_ok:
        verdict = INVIABLE
        reason = f"T={T_s*1000:.1f}ms < T_min={T_min*1000:.1f}ms (spectral resolution)"
    elif iqr_pred < rel_iqr_threshold:
        if q_rho >= threshold_qrho:
            verdict = VIABLE
            reason = f"Q*rho_eff={q_rho:.2f} >= {threshold_qrho:.2f}"
        else:
            verdict = MARGINAL
            reason = f"Q*rho_eff={q_rho:.2f} (viable by CR, marginal with alpha)"
    elif q_rho >= INFORMATIVE_Q_RHO_MIN:
        verdict = MARGINAL
        reason = f"Q*rho_eff={q_rho:.2f} >= {INFORMATIVE_Q_RHO_MIN} but predicted rel_iqr={iqr_pred:.3f} > {rel_iqr_threshold}"
    else:
        verdict = INVIABLE
        reason = f"Q*rho_eff={q_rho:.2f} < {INFORMATIVE_Q_RHO_MIN}"

    return {
        "eta": eta,
        "rho_eff": rho_eff,
        "Q_x_rho_eff": q_rho,
        "rel_iqr_predicted": iqr_pred,
        "t0_max_s": t0_max,
        "T_min_s": T_min,
        "band_min_hz": list(band),
        "resolution_ok": resolution_ok,
        "contamination_ok": contamination_ok,
        "informative_ok": informative_ok,
        "verdict": verdict,
        "reason": reason,
        "t0_over_tau": t0_s / tau_s if tau_s > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Full preflight viability (per event)
# ---------------------------------------------------------------------------

def preflight_viability(
    *,
    event_id: str,
    m_final_msun: float,
    chi_final: float,
    rho_total: float,
    t0_s: float,
    T_s: float,
    alpha_safety: float = DEFAULT_ALPHA_SAFETY,
    t0_min_s: float = DEFAULT_T0_MIN_CONTAMINATION_S,
    modes: list[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Full preflight viability assessment for an event.

    Parameters
    ----------
    event_id : str
        Event identifier (e.g. "GW150914").
    m_final_msun : float
        Remnant mass in solar masses.
    chi_final : float
        Remnant dimensionless spin.
    rho_total : float
        Estimated total ringdown SNR.
    t0_s : float
        Start time of analysis window (seconds post-peak).
    T_s : float
        Duration of analysis window (seconds).
    alpha_safety : float
        Safety factor for CR → bootstrap degradation.
    t0_min_s : float
        Minimum t0 for merger contamination avoidance.
    modes : list of (l,m,n) tuples
        Modes to assess. Default: [(2,2,0), (2,2,1)].

    Returns
    -------
    dict
        Complete preflight viability assessment.
    """
    if modes is None:
        modes = [(2, 2, 0), (2, 2, 1)]

    mode_assessments: dict[str, Any] = {}
    for mode in modes:
        label = f"{mode[0]}{mode[1]}{mode[2]}"
        qnm = kerr_qnm(m_final_msun, chi_final, mode)
        assessment = assess_mode_viability(
            f_hz=qnm.f_hz,
            tau_s=qnm.tau_s,
            Q=qnm.Q,
            rho_total=rho_total,
            t0_s=t0_s,
            T_s=T_s,
            alpha_safety=alpha_safety,
            t0_min_s=t0_min_s,
        )
        assessment["qnm_params"] = {
            "f_hz": qnm.f_hz,
            "tau_s": qnm.tau_s,
            "Q": qnm.Q,
        }
        mode_assessments[label] = assessment

    # Overall verdict: driven by the fundamental mode (220)
    primary_mode = "220"
    primary = mode_assessments.get(primary_mode, {})
    overall_verdict = primary.get("verdict", INVIABLE)

    # Recommended configuration: t0 that maximizes rho_eff for 220,
    # constrained by t0_min
    primary_qnm = mode_assessments.get(primary_mode, {}).get("qnm_params", {})
    primary_tau = primary_qnm.get("tau_s", 0.005)
    primary_Q = primary_qnm.get("Q", 4.0)
    t0_max_220 = t0_max_informative(primary_tau, primary_Q, rho_total, alpha_safety, T_s)

    recommended_t0_s: float | None = None
    recommended_rho_eff: float | None = None
    recommended_rel_iqr: float | None = None

    if t0_max_220 > t0_min_s:
        # Recommend the midpoint of [t0_min, t0_max] as a conservative choice
        recommended_t0_s = t0_min_s + 0.3 * (t0_max_220 - t0_min_s)
        recommended_rho_eff = rho_effective(recommended_t0_s, T_s, primary_tau, rho_total)
        recommended_rel_iqr = rel_iqr_predicted(primary_Q, recommended_rho_eff, alpha_safety)

    return {
        "schema_version": "preflight_viability_v1",
        "event_id": event_id,
        "source_params": {
            "m_final_msun": m_final_msun,
            "chi_final": chi_final,
            "rho_total": rho_total,
        },
        "current_config": {
            "t0_s": t0_s,
            "T_s": T_s,
        },
        "alpha_safety": alpha_safety,
        "modes": mode_assessments,
        "overall_verdict": overall_verdict,
        "recommended_config": {
            "t0_s": recommended_t0_s,
            "T_s": T_s,
            "t0_max_220_s": t0_max_220,
        },
        "recommended_rho_eff_220": recommended_rho_eff,
        "recommended_rel_iqr_predicted_220": recommended_rel_iqr,
    }


# ---------------------------------------------------------------------------
# Catalog viability table (batch, pure computation)
# ---------------------------------------------------------------------------

def catalog_viability_table(
    events: dict[str, dict[str, float]],
    *,
    T_s: float = 0.06,
    alpha_safety: float = DEFAULT_ALPHA_SAFETY,
    t0_min_s: float = DEFAULT_T0_MIN_CONTAMINATION_S,
    rho_ringdown_fraction: float = 0.33,
) -> dict[str, Any]:
    """Compute viability table for a catalog of events.

    Parameters
    ----------
    events : dict
        {event_id: {"m_final_msun": ..., "chi_final": ..., "snr_network": ...}}
    T_s : float
        Default window duration.
    alpha_safety : float
        Safety factor.
    t0_min_s : float
        Minimum t0 for contamination avoidance.
    rho_ringdown_fraction : float
        Fraction of network SNR attributable to ringdown (conservative estimate).

    Returns
    -------
    dict with schema_version and per-event viability summaries.
    """
    rows: list[dict[str, Any]] = []
    for event_id, params in sorted(events.items()):
        rho_total = params.get("snr_network", 0.0) * rho_ringdown_fraction
        m_final = params.get("m_final_msun", 0.0)
        chi_final = params.get("chi_final", 0.0)

        qnm_220 = kerr_qnm(m_final, chi_final, (2, 2, 0))
        qnm_221 = kerr_qnm(m_final, chi_final, (2, 2, 1))

        t0_max_220 = t0_max_informative(qnm_220.tau_s, qnm_220.Q, rho_total, alpha_safety, T_s)
        t0_max_221 = t0_max_informative(qnm_221.tau_s, qnm_221.Q, rho_total, alpha_safety, T_s)

        viable_220 = t0_max_220 > t0_min_s
        viable_221 = t0_max_221 > t0_min_s

        row = {
            "event_id": event_id,
            "m_final_msun": m_final,
            "chi_final": chi_final,
            "rho_total_estimated": rho_total,
            "mode_220": {
                "f_hz": qnm_220.f_hz,
                "tau_ms": qnm_220.tau_s * 1000,
                "Q": qnm_220.Q,
                "t0_max_ms": t0_max_220 * 1000,
                "T_min_ms": T_min_resolution(qnm_220.tau_s) * 1000,
                "viable": viable_220,
            },
            "mode_221": {
                "f_hz": qnm_221.f_hz,
                "tau_ms": qnm_221.tau_s * 1000,
                "Q": qnm_221.Q,
                "t0_max_ms": t0_max_221 * 1000,
                "T_min_ms": T_min_resolution(qnm_221.tau_s) * 1000,
                "viable": viable_221,
            },
        }
        rows.append(row)

    return {
        "schema_version": "catalog_viability_table_v1",
        "params": {
            "T_s": T_s,
            "alpha_safety": alpha_safety,
            "t0_min_s": t0_min_s,
            "rho_ringdown_fraction": rho_ringdown_fraction,
        },
        "events": rows,
        "summary": {
            "n_events": len(rows),
            "n_viable_220": sum(1 for r in rows if r["mode_220"]["viable"]),
            "n_viable_221": sum(1 for r in rows if r["mode_221"]["viable"]),
        },
    }


# ---------------------------------------------------------------------------
# Alpha calibration from existing runs
# ---------------------------------------------------------------------------

def calibrate_alpha_from_runs(
    observations: list[dict[str, Any]],
    percentile: float = 0.90,
) -> dict[str, Any]:
    """Calibrate alpha_safety from existing run observations.

    Parameters
    ----------
    observations : list of dicts
        Each with: rel_iqr_f220_observed, Q_220, rho_eff_estimated
    percentile : float
        Percentile of observed alphas to use as alpha_safety.

    Returns
    -------
    dict with calibrated alpha, stats, and individual observations.
    """
    alphas: list[float] = []
    details: list[dict[str, Any]] = []
    for obs in observations:
        rel_iqr = obs.get("rel_iqr_f220_observed")
        Q = obs.get("Q_220")
        rho_eff = obs.get("rho_eff_estimated")
        if rel_iqr is None or Q is None or rho_eff is None:
            continue
        if Q <= 0 or rho_eff <= 0:
            continue
        alpha_obs = rel_iqr * Q * rho_eff / CR_COEFFICIENT
        alphas.append(alpha_obs)
        details.append({
            "rel_iqr_observed": rel_iqr,
            "Q": Q,
            "rho_eff": rho_eff,
            "alpha_observed": alpha_obs,
        })

    if not alphas:
        return {
            "alpha_safety": DEFAULT_ALPHA_SAFETY,
            "calibrated": False,
            "reason": "no valid observations",
            "n_observations": 0,
        }

    alphas_sorted = sorted(alphas)
    n = len(alphas_sorted)
    idx = min(int(math.ceil(percentile * n)) - 1, n - 1)
    idx = max(idx, 0)
    alpha_calibrated = alphas_sorted[idx]

    return {
        "alpha_safety": alpha_calibrated,
        "calibrated": True,
        "percentile": percentile,
        "n_observations": n,
        "alpha_median": alphas_sorted[n // 2],
        "alpha_min": alphas_sorted[0],
        "alpha_max": alphas_sorted[-1],
        "details": details,
    }


# ---------------------------------------------------------------------------
# T0 sweep domain restriction
# ---------------------------------------------------------------------------

def viable_t0_domain(
    *,
    tau_s: float,
    Q: float,
    rho_total: float,
    T_s: float,
    alpha_safety: float = DEFAULT_ALPHA_SAFETY,
    t0_min_s: float = DEFAULT_T0_MIN_CONTAMINATION_S,
) -> dict[str, Any]:
    """Compute the viable t0 domain for a sweep.

    Returns the intersection of [t0_min, t0_max] where informative analysis
    is possible. If empty, the mode is unconstrainable.
    """
    t0_max = t0_max_informative(tau_s, Q, rho_total, alpha_safety, T_s)
    domain_empty = t0_max <= t0_min_s

    return {
        "t0_min_s": t0_min_s,
        "t0_max_s": t0_max,
        "domain_empty": domain_empty,
        "domain_width_ms": max(0.0, (t0_max - t0_min_s) * 1000),
        "recommended_grid_ms": _suggested_grid(t0_min_s, t0_max) if not domain_empty else [],
    }


def _suggested_grid(t0_min_s: float, t0_max_s: float, n_points: int = 7) -> list[int]:
    """Suggest a t0 grid (in ms) within the viable domain."""
    t0_min_ms = t0_min_s * 1000
    t0_max_ms = t0_max_s * 1000
    if t0_max_ms <= t0_min_ms:
        return []
    step = max(1, int((t0_max_ms - t0_min_ms) / (n_points - 1)))
    grid = list(range(int(math.ceil(t0_min_ms)), int(t0_max_ms) + 1, step))
    return grid[:n_points]
